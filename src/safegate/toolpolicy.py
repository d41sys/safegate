"""Deterministic capability and data-flow control for tool calls (CaMeL-lite).

``spec.py`` checks the structured spec a model emits against contract bounds. This module
checks the *action* the agent is about to take. Tools are declared in YAML with a capability
class and a policy per argument; every argument value arrives with provenance labels
(``safegate.flow``). A call is allowed only when each argument satisfies its policy:

* ``untrusted: forbid`` (the default): the value must not derive from untrusted text; with
  ``untrusted: allow`` it may, from any source, or only from the sources named in
  ``untrusted_sources`` (``untrusted_sources: [web]`` admits ``untrusted:web`` and nothing else).
* numeric ``min`` / ``max``, an ``allowed_values`` list, ``allowed_hosts`` for URLs,
  ``allowed_email_domains`` for recipients.

A session check (Meta's "Rule of Two", Willison's "lethal trifecta") adds a second, coarser
rule: a session must not combine untrusted input, sensitive data and a consequential action
(``state-change`` or ``external-send``). The call that would complete the combination is
refused or escalated to a human.

A tool may also declare ``untrusted_session: escalate`` (or ``refuse``): a ``state-change`` or
``external-send`` tool whose arguments hold nothing worth checking (``reboot_inverter`` takes
none) is then escalated (or refused) whenever the session has ingested untrusted data, sensitive
or not.

``check_tool_call`` returns the same ``Decision`` as the text gate and ``check_spec``.

Limits, by design:

* Text-only attacks are out of scope of the per-call checks: a poisoned summary or a phishing
  link in the reply makes no tool call. The ``egress`` block (on in the default policy) covers
  the link case deterministically (``check_egress_text``): every host named in the reply (or in a
  text argument marked ``scan_hosts``) must be allow-listed or must already appear in the owner's
  own text. It says nothing about false claims without a link. Text sent to the owner's own
  domain (``egress.internal_domains``, via the recipient argument) is not scanned: egress to the
  trusted principal is not exfiltration.
* When the user legitimately routes untrusted data into an argument the policy allows
  (``untrusted: allow``), the gate checks bounds, not truth, as ``spec.py`` does. With
  ``untrusted_sources`` it checks the source label as well, which cannot tell a poisoned page
  from a good one when both come from an allowed source (``J12`` planted in the tariff page).
* The Rule of Two is session-wide and coarse. It cannot tell that a state change was
  triggered by an injection when the arguments are clean (no arguments, or constants).
  ``untrusted_session`` closes that for the tools that declare it, at the price of escalating
  every use of the tool in a session that read untrusted data, whatever triggered it.
* Labels come from the harness (``safegate.flow``). A wrong label cannot be detected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from safegate.flow import (
    UNLABELED,
    UNTRUSTED_PREFIX,
    Labeled,
    Labels,
    is_sensitive,
    is_untrusted,
    parse_labels,
    union,
    untrusted_labels,
    validate_label,
)
from safegate.gate import Decision
from safegate.rules import ACTION_PRECEDENCE

TOOL_CATEGORY = "TOOL_POLICY"
CAPABILITIES = ("read", "write", "external-send", "state-change")
CONSEQUENTIAL = frozenset({"external-send", "state-change"})

_SEVERITY: dict[str, int] = {action: rank for action, rank in ACTION_PRECEDENCE.items()}
_VIOLATION_ACTIONS = ("warn", "refuse", "escalate")
_RULE_OF_TWO_MODES = ("off", "refuse", "escalate")
_ARG_TYPES = ("number", "string", "boolean")
_UNTRUSTED_MODES = ("forbid", "allow")
_UNTRUSTED_SESSION_MODES = ("allow", "escalate", "refuse")

_ROOT_KEYS = {"version", "on_violation", "rule_of_two", "egress", "tools"}
_EGRESS_KEYS = {
    "allowed_hosts",
    "internal_domains",
    "trust_user_text",
    "include_tool_hosts",
    "on_violation",
}
_TOOL_KEYS = {"capability", "description", "result_labels", "args", "untrusted_session"}
_ARG_KEYS = {
    "type",
    "required",
    "untrusted",
    "untrusted_sources",
    "min",
    "max",
    "allowed_values",
    "allowed_hosts",
    "allowed_email_domains",
    "recipient_of",
    "scan_hosts",
}

_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$")
_EMAIL_RE = re.compile(r"^[^\s@,;<>()\"'\\]+@([A-Za-z0-9.-]+)$")
_URL_FORBIDDEN_RE = re.compile(r"[\\\s\x00-\x1f\x7f]")

# Hosts named in free text (egress check). A URL with a scheme, an e-mail address, or a bare
# dotted name such as ``evil.example/verify``. Labels are Unicode on purpose: a look-alike
# host is found, and is never on an (ASCII) allow-list.
_URL_CHARS = r"[^\s<>\"'`()\[\]{}|^\\]"
_TEXT_URL_RE = re.compile(r"\b[a-z][a-z0-9+.\-]*://" + _URL_CHARS + "*", re.IGNORECASE)
_LABEL = r"[^\W_](?:[\w\-]*[^\W_])?"
_DOMAIN = rf"(?:{_LABEL}\.)+{_LABEL}"
_TEXT_EMAIL_RE = re.compile(rf"[\w.+%\-]+@({_DOMAIN})(?![\w\-])", re.IGNORECASE)
_TEXT_BARE_RE = re.compile(
    rf"(?<![\w@.\-])({_DOMAIN})(?![\w\-])(?::\d{{1,5}})?(/{_URL_CHARS}*)?", re.IGNORECASE
)
# Dotted names that are file names, not hosts (``report.csv``).
_FILE_SUFFIXES = frozenset(
    {"csv", "json", "log", "md", "pdf", "png", "jpg", "py", "txt", "xml", "yaml", "yml", "zip"}
)
UNPARSEABLE_HOST = "(unparseable)"
REDACTION = "[link removed]"


class ToolPolicyError(ValueError):
    """Raised when a tool policy contains unknown keys, bad types or inconsistent values."""


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgPolicy:
    """Policy for one tool argument."""

    type: str
    required: bool = True
    allow_untrusted: bool = False
    untrusted_sources: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str | float, ...] | None = None
    allowed_hosts: tuple[str, ...] | None = None
    allowed_email_domains: tuple[str, ...] | None = None
    recipient_of: tuple[str, ...] | None = None
    scan_hosts: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """One declared tool: its capability class, the labels its result carries, its arguments.

    ``untrusted_session`` (``allow`` by default) is the action when the tool is called in a
    session that already holds untrusted data; only a consequential tool may set it.
    """

    name: str
    capability: str
    args: dict[str, ArgPolicy]
    result_labels: Labels = frozenset()
    description: str = ""
    untrusted_session: str = "allow"

    @property
    def consequential(self) -> bool:
        """True for ``state-change`` and ``external-send``."""
        return self.capability in CONSEQUENTIAL


@dataclass(frozen=True)
class EgressPolicy:
    """Output allow-list: which hosts text leaving the agent may name.

    A host named in a reply (or a ``scan_hosts`` argument) is allowed when it is in
    ``allowed_hosts`` (``*.suffix`` matches subdomains), when it appears in the owner's own
    text (``trust_user_text``: the user prompt is the trusted provenance) or when a tool's
    ``allowed_hosts`` lists it (``include_tool_hosts``). ``on_violation`` is the action for a
    reply that names any other host; the reply is redacted unless a human approves.

    ``internal_domains`` (``*.suffix`` matches subdomains) are the trusted principal's own
    domains. A ``scan_hosts`` argument is not scanned when the call's recipient argument (one
    with ``allowed_email_domains``, or one declaring ``recipient_of``) is an address in them:
    text sent to the owner leaves no trust boundary. The domains are not added to the
    allow-list, and the reply check ignores them.
    """

    allowed_hosts: tuple[str, ...] = ()
    internal_domains: tuple[str, ...] = ()
    trust_user_text: bool = True
    include_tool_hosts: bool = True
    on_violation: str = "refuse"


@dataclass(frozen=True)
class ToolPolicy:
    """Declared tools plus the action taken on a violation. Tools not listed are refused.

    ``egress`` is optional: without it replies and text arguments are not scanned for hosts.
    """

    tools: dict[str, ToolSpec]
    on_violation: str = "refuse"
    rule_of_two: str = "escalate"
    egress: EgressPolicy | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolPolicy:
        """Build a policy from parsed YAML; unknown keys and bad values raise."""
        _no_unknown(data, _ROOT_KEYS, "policy")
        if data.get("version", 1) != 1:
            raise ToolPolicyError(f"unsupported policy version: {data.get('version')!r}")
        on_violation = data.get("on_violation", "refuse")
        if on_violation not in _VIOLATION_ACTIONS:
            raise ToolPolicyError(f"on_violation must be one of {_VIOLATION_ACTIONS}")
        rule_of_two = data.get("rule_of_two", "escalate")
        if rule_of_two is False:  # YAML 1.1 reads an unquoted `off` as the boolean false
            rule_of_two = "off"
        if rule_of_two not in _RULE_OF_TWO_MODES:
            raise ToolPolicyError(f"rule_of_two must be one of {_RULE_OF_TWO_MODES}")
        raw_tools = data.get("tools")
        if not isinstance(raw_tools, Mapping) or not raw_tools:
            raise ToolPolicyError("'tools' must be a non-empty mapping")
        tools = {str(n): _parse_tool(str(n), body) for n, body in raw_tools.items()}
        egress = _parse_egress(data["egress"]) if "egress" in data else None
        if egress is None:
            for tool_name, spec in tools.items():
                for arg_name, arg in spec.args.items():
                    if arg.scan_hosts:
                        raise ToolPolicyError(
                            f"tools.{tool_name}.args.{arg_name}: scan_hosts needs an egress block"
                        )
        for tool_name, spec in tools.items():
            for arg_name, arg in spec.args.items():
                for target in arg.recipient_of or ():
                    if target == arg_name or target not in spec.args:
                        raise ToolPolicyError(
                            f"tools.{tool_name}.args.{arg_name}.recipient_of: "
                            f"{target!r} is not another argument of the tool"
                        )
        return cls(tools=tools, on_violation=on_violation, rule_of_two=rule_of_two, egress=egress)

    def label_result(
        self, tool: str, value: Any, arg_labels: Iterable[Iterable[str]] = ()
    ) -> Labeled[Any]:
        """Label a tool result: the tool's declared ``result_labels`` plus the argument labels."""
        spec = self.tools.get(tool)
        if spec is None:
            raise KeyError(f"unknown tool {tool!r}")
        return Labeled(value, union(spec.result_labels, *arg_labels))


def _no_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ToolPolicyError(f"Unknown keys in {where}: {sorted(map(str, unknown))}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolPolicyError(f"{where} must be a mapping")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolPolicyError(f"{where} must be a finite number")
    return float(value)


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ToolPolicyError(f"{where} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ToolPolicyError(f"{where} entries must be non-empty strings")
        out.append(item.casefold())
    return tuple(out)


def _parse_arg(tool: str, name: str, raw: Any) -> ArgPolicy:
    where = f"tools.{tool}.args.{name}"
    body = _mapping(raw, where)
    _no_unknown(body, _ARG_KEYS, where)
    arg_type = body.get("type")
    if arg_type not in _ARG_TYPES:
        raise ToolPolicyError(f"{where}.type must be one of {_ARG_TYPES}")
    required = body.get("required", True)
    if not isinstance(required, bool):
        raise ToolPolicyError(f"{where}.required must be true or false")
    mode = body.get("untrusted", "forbid")
    if mode not in _UNTRUSTED_MODES:
        raise ToolPolicyError(f"{where}.untrusted must be one of {_UNTRUSTED_MODES}")
    sources: tuple[str, ...] | None = None
    if "untrusted_sources" in body:
        if mode != "allow":
            raise ToolPolicyError(f"{where}.untrusted_sources needs untrusted: allow")
        sources = _string_list(body["untrusted_sources"], f"{where}.untrusted_sources")
        for source in sources:
            if UNTRUSTED_PREFIX + source == UNLABELED:
                raise ToolPolicyError(f"{where}.untrusted_sources: {source!r} is never vouched for")
            try:
                validate_label(UNTRUSTED_PREFIX + source)
            except ValueError as exc:
                raise ToolPolicyError(f"{where}.untrusted_sources: {exc}") from exc

    minimum = _number(body["min"], f"{where}.min") if "min" in body else None
    maximum = _number(body["max"], f"{where}.max") if "max" in body else None
    if (minimum is not None or maximum is not None) and arg_type != "number":
        raise ToolPolicyError(f"{where}: min/max need type number")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ToolPolicyError(f"{where}: min is above max")

    allowed_values: tuple[str | float, ...] | None = None
    if "allowed_values" in body:
        raw_values = body["allowed_values"]
        if not isinstance(raw_values, list) or not raw_values:
            raise ToolPolicyError(f"{where}.allowed_values must be a non-empty list")
        values: list[str | float] = []
        for v in raw_values:
            if arg_type == "string" and isinstance(v, str):
                values.append(v)
            elif arg_type == "number":
                values.append(_number(v, f"{where}.allowed_values entry"))
            else:
                raise ToolPolicyError(f"{where}.allowed_values entries must match type {arg_type}")
        allowed_values = tuple(values)

    hosts = emails = None
    if "allowed_hosts" in body:
        if arg_type != "string":
            raise ToolPolicyError(f"{where}: allowed_hosts needs type string")
        hosts = _string_list(body["allowed_hosts"], f"{where}.allowed_hosts")
        for h in hosts:
            if not _HOST_RE.match(h.removeprefix("*.")):
                raise ToolPolicyError(f"{where}.allowed_hosts: bad host {h!r}")
    if "allowed_email_domains" in body:
        if arg_type != "string":
            raise ToolPolicyError(f"{where}: allowed_email_domains needs type string")
        emails = _string_list(body["allowed_email_domains"], f"{where}.allowed_email_domains")
        for d in emails:
            if not _HOST_RE.match(d):
                raise ToolPolicyError(f"{where}.allowed_email_domains: bad domain {d!r}")
    recipient_of = None
    if "recipient_of" in body:
        raw_of = body["recipient_of"]
        if arg_type != "string":
            raise ToolPolicyError(f"{where}: recipient_of needs type string")
        names = [raw_of] if isinstance(raw_of, str) else raw_of
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) and n for n in names)
        ):
            raise ToolPolicyError(f"{where}.recipient_of must name one or more arguments")
        recipient_of = tuple(names)
    scan_hosts = body.get("scan_hosts", False)
    if not isinstance(scan_hosts, bool):
        raise ToolPolicyError(f"{where}.scan_hosts must be true or false")
    if scan_hosts and arg_type != "string":
        raise ToolPolicyError(f"{where}: scan_hosts needs type string")
    return ArgPolicy(
        type=arg_type,
        required=required,
        allow_untrusted=mode == "allow",
        untrusted_sources=sources,
        minimum=minimum,
        maximum=maximum,
        allowed_values=allowed_values,
        allowed_hosts=hosts,
        allowed_email_domains=emails,
        recipient_of=recipient_of,
        scan_hosts=scan_hosts,
    )


def _parse_egress(raw: Any) -> EgressPolicy:
    where = "egress"
    body = _mapping(raw, where)
    _no_unknown(body, _EGRESS_KEYS, where)
    hosts: list[str] = []
    raw_hosts = body.get("allowed_hosts", [])
    if not isinstance(raw_hosts, list):
        raise ToolPolicyError(f"{where}.allowed_hosts must be a list")
    if raw_hosts:
        hosts = list(_string_list(raw_hosts, f"{where}.allowed_hosts"))
    for h in hosts:
        if not _HOST_RE.match(h.removeprefix("*.")):
            raise ToolPolicyError(f"{where}.allowed_hosts: bad host {h!r}")
    internal: list[str] = []
    raw_internal = body.get("internal_domains", [])
    if not isinstance(raw_internal, list):
        raise ToolPolicyError(f"{where}.internal_domains must be a list")
    if raw_internal:
        internal = [d.casefold() for d in _string_list(raw_internal, f"{where}.internal_domains")]
    for d in internal:
        if not _HOST_RE.match(d.removeprefix("*.")):
            raise ToolPolicyError(f"{where}.internal_domains: bad domain {d!r}")
    flags: dict[str, bool] = {}
    for key in ("trust_user_text", "include_tool_hosts"):
        value = body.get(key, True)
        if not isinstance(value, bool):
            raise ToolPolicyError(f"{where}.{key} must be true or false")
        flags[key] = value
    on_violation = body.get("on_violation", "refuse")
    if on_violation not in _VIOLATION_ACTIONS:
        raise ToolPolicyError(f"{where}.on_violation must be one of {_VIOLATION_ACTIONS}")
    return EgressPolicy(
        allowed_hosts=tuple(hosts),
        internal_domains=tuple(internal),
        trust_user_text=flags["trust_user_text"],
        include_tool_hosts=flags["include_tool_hosts"],
        on_violation=on_violation,
    )


def _parse_tool(name: str, raw: Any) -> ToolSpec:
    where = f"tools.{name}"
    body = _mapping(raw, where)
    _no_unknown(body, _TOOL_KEYS, where)
    capability = body.get("capability")
    if capability not in CAPABILITIES:
        raise ToolPolicyError(f"{where}.capability must be one of {CAPABILITIES}")
    description = body.get("description", "")
    if not isinstance(description, str):
        raise ToolPolicyError(f"{where}.description must be a string")
    raw_labels = body.get("result_labels", [])
    if not isinstance(raw_labels, list):
        raise ToolPolicyError(f"{where}.result_labels must be a list")
    try:
        result_labels = parse_labels(raw_labels)
    except ValueError as exc:
        raise ToolPolicyError(f"{where}.result_labels: {exc}") from exc
    raw_args = body.get("args", {})
    args_map = _mapping(raw_args, f"{where}.args")
    args = {str(a): _parse_arg(name, str(a), p) for a, p in args_map.items()}
    untrusted_session = body.get("untrusted_session", "allow")
    if untrusted_session not in _UNTRUSTED_SESSION_MODES:
        raise ToolPolicyError(
            f"{where}.untrusted_session must be one of {_UNTRUSTED_SESSION_MODES}"
        )
    if untrusted_session != "allow" and capability not in CONSEQUENTIAL:
        raise ToolPolicyError(
            f"{where}.untrusted_session needs a state-change or external-send tool"
        )
    return ToolSpec(name, capability, args, result_labels, description, untrusted_session)


def load_tool_policy(source: Path | str) -> ToolPolicy:
    """Load a policy from a YAML file path or a YAML string. Fails closed on any error."""
    if isinstance(source, Path) or ("\n" not in source and Path(source).exists()):
        content = Path(source).read_text(encoding="utf-8")
    else:
        content = str(source)
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ToolPolicyError(f"Failed to parse YAML tool policy: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ToolPolicyError("Tool policy root must be a YAML mapping")
    return ToolPolicy.from_dict(parsed)


# --------------------------------------------------------------------------------------
# Calls and sessions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A proposed tool call: the tool name and plain argument values."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        """Strictly parse ``{"tool": str, "args": {...}}``."""
        tool = data.get("tool")
        args = data.get("args", {})
        if not isinstance(tool, str) or not tool:
            raise ValueError("call.tool must be a non-empty string")
        if not isinstance(args, Mapping):
            raise ValueError("call.args must be an object")
        return cls(tool, dict(args))


def call_from_labeled(
    tool: str, args: Mapping[str, Labeled[Any]]
) -> tuple[ToolCall, dict[str, Labels]]:
    """Split labeled arguments into a ``ToolCall`` and the per-argument label mapping."""
    for name, arg in args.items():
        if not isinstance(arg, Labeled):
            raise TypeError(f"argument {name!r} is not Labeled: wrap it with user() or system()")
    return (
        ToolCall(tool, {n: a.value for n, a in args.items()}),
        {n: a.labels for n, a in args.items()},
    )


@dataclass
class Session:
    """What a session has seen and done so far, for the Rule-of-Two check.

    ``labels`` is the union of the labels of everything admitted into the agent's context
    (arguments and declared results of admitted calls, plus whatever ``observe`` adds).
    ``consequential`` is set once a ``state-change`` or ``external-send`` call was admitted.
    """

    labels: set[str] = field(default_factory=set)
    consequential: bool = False

    def observe(self, labels: Iterable[str]) -> None:
        """Add labels of data that entered the agent's context."""
        self.labels.update(parse_labels(labels))

    def record(
        self, call: ToolCall, labels: Mapping[str, Iterable[str]], policy: ToolPolicy
    ) -> None:
        """Record a call that was executed (for example after a human approved an escalation)."""
        spec = policy.tools[call.tool]
        self.labels.update(_call_labels(call, labels), spec.result_labels)
        self.consequential = self.consequential or spec.consequential

    def admit(
        self, call: ToolCall, labels: Mapping[str, Iterable[str]], policy: ToolPolicy
    ) -> Decision:
        """Check ``call``; when it is allowed (or only warned), record it in the session."""
        decision = check_tool_call(call, labels, policy, self)
        if decision.action in ("allow", "warn"):
            self.record(call, labels, policy)
        return decision


# --------------------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------------------


@dataclass
class _Violations:
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, reason: str, arg: str | None = None) -> None:
        self.codes.append(code)
        self.reasons.append(reason)
        self.details.append({"code": code, "arg": arg, "reason": reason})


def _call_labels(call: ToolCall, labels: Mapping[str, Iterable[str]]) -> Labels:
    """Labels of all arguments of the call; a missing argument label counts as unlabeled."""
    return union(*(_arg_labels(name, labels) for name in call.args))


def _arg_labels(name: str, labels: Mapping[str, Iterable[str]]) -> Labels:
    raw = labels.get(name)
    return frozenset({UNLABELED}) if raw is None else parse_labels(raw)


def _host_ok(host: str, allowed: tuple[str, ...]) -> bool:
    for entry in allowed:
        if entry.startswith("*."):
            if host.endswith(entry[1:]):
                return True
        elif host == entry:
            return True
    return False


def _check_url(name: str, value: str, allowed: tuple[str, ...], out: _Violations) -> None:
    if _URL_FORBIDDEN_RE.search(value):
        out.add(
            "tool-arg-host", f"{name}: URL has backslash, whitespace or control characters", name
        )
        return
    try:
        parts = urlsplit(value)
        host = parts.hostname
        _ = parts.port  # raises ValueError on a bad port
    except ValueError:
        out.add("tool-arg-host", f"{name}: URL does not parse", name)
        return
    if parts.scheme not in ("http", "https") or not host:
        out.add("tool-arg-host", f"{name}: needs an absolute http(s) URL", name)
    elif "@" in parts.netloc:
        out.add("tool-arg-host", f"{name}: URL carries user info before the host", name)
    elif not _HOST_RE.match(host) or not _host_ok(host, allowed):
        out.add("tool-arg-host", f"{name}: host {host!r} is not in allowed_hosts", name)


def _check_email(name: str, value: str, allowed: tuple[str, ...], out: _Violations) -> None:
    match = _EMAIL_RE.match(value)
    if match is None:
        out.add("tool-arg-domain", f"{name}: not a single plain e-mail address", name)
    elif match.group(1).casefold() not in allowed:
        out.add("tool-arg-domain", f"{name}: domain {match.group(1)!r} is not allowed", name)


# --------------------------------------------------------------------------------------
# Egress: hosts named in text that leaves the agent
# --------------------------------------------------------------------------------------


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _is_ipv4(host: str) -> bool:
    parts = host.split(".")
    return len(parts) == 4 and all(p.isascii() and p.isdigit() and len(p) <= 3 for p in parts)


def hosts_in_text(text: str) -> list[tuple[int, int, str]]:
    """Every host the text names, as ``(start, end, host)``; ``host`` is lower-cased.

    Finds URLs with a scheme (host after any ``user@``), e-mail addresses (the domain) and bare
    dotted names (``evil.example/verify``; file names such as ``report.csv`` are skipped).
    A scheme URL with no parseable host yields ``UNPARSEABLE_HOST``.
    """
    found: list[tuple[int, int, str]] = []
    work = text
    for m in _TEXT_URL_RE.finditer(text):
        end = m.start() + len(m.group().rstrip(".,;:!?"))
        try:
            host = urlsplit(text[m.start() : end]).hostname
        except ValueError:
            host = None
        found.append((m.start(), end, host.rstrip(".") if host else UNPARSEABLE_HOST))
        work = _blank(work, m.start(), end)
    for m in _TEXT_EMAIL_RE.finditer(work):
        found.append((m.start(), m.end(), m.group(1).casefold()))
        work = _blank(work, m.start(), m.end())
    for m in _TEXT_BARE_RE.finditer(work):
        host = m.group(1).casefold()
        suffix = host.rsplit(".", 1)[1]
        if suffix.isdigit():
            if not _is_ipv4(host):
                continue  # a number such as 11.2 or a version string
        elif len(suffix) < 2 or (suffix in _FILE_SUFFIXES and m.group(2) is None):
            continue
        found.append((m.start(), m.start() + len(m.group().rstrip(".,;:!?")), host))
    return sorted(found)


def egress_allowed_hosts(policy: ToolPolicy, trusted_text: Iterable[str] = ()) -> tuple[str, ...]:
    """The hosts text may name under ``policy.egress``: its own list, the hosts and e-mail
    domains the tool arguments allow-list, and the hosts in ``trusted_text`` (the owner's own
    words). Empty without an egress block."""
    egress = policy.egress
    if egress is None:
        return ()
    allowed = list(egress.allowed_hosts)
    if egress.include_tool_hosts:
        for spec in policy.tools.values():
            for arg in spec.args.values():
                allowed += arg.allowed_hosts or ()
                allowed += arg.allowed_email_domains or ()
    if egress.trust_user_text:
        for text in trusted_text:
            allowed += [h for _, _, h in hosts_in_text(text) if h != UNPARSEABLE_HOST]
    return tuple(dict.fromkeys(allowed))


def _blocked_hosts(text: str, allowed: tuple[str, ...]) -> list[tuple[int, int, str]]:
    return [
        (s, e, h)
        for s, e, h in hosts_in_text(text)
        if h == UNPARSEABLE_HOST or not _host_ok(h, allowed)
    ]


def check_egress_text(
    text: str,
    policy: ToolPolicy,
    trusted_text: Iterable[str] = (),
    scope: str = "reply",
) -> Decision:
    """Check outgoing text (the final reply) against the egress allow-list.

    ``allow`` when the text names no host outside ``egress_allowed_hosts(policy, trusted_text)``
    (or when the policy has no egress block); otherwise ``policy.egress.on_violation`` with rule
    id ``egress-host``. ``metadata["blocked"]`` lists the offending hosts. Deterministic: no
    labels are consulted, so a mislabeled harness cannot open the gate.
    """
    egress = policy.egress
    if egress is None:
        return Decision(
            action="allow", rule_ids=[], reasons=[], metadata={"scope": scope, "enforced": False}
        )
    allowed = egress_allowed_hosts(policy, trusted_text)
    blocked = list(dict.fromkeys(h for _, _, h in _blocked_hosts(text, allowed)))
    metadata: dict[str, Any] = {
        "scope": scope,
        "enforced": True,
        "allowed_hosts": list(allowed),
        "blocked": blocked,
    }
    if not blocked:
        return Decision(action="allow", rule_ids=[], reasons=[], metadata=metadata)
    return Decision(
        action=egress.on_violation,
        rule_ids=["egress-host"],
        reasons=[f"{scope}: host {h!r} is not allow-listed" for h in blocked],
        category=TOOL_CATEGORY,
        score=1.0,
        metadata=metadata,
    )


def redact_egress_text(text: str, policy: ToolPolicy, trusted_text: Iterable[str] = ()) -> str:
    """``text`` with every URL, address or host outside the allow-list replaced by
    ``REDACTION``; unchanged without an egress block."""
    if policy.egress is None:
        return text
    out = text
    for start, end, _ in reversed(_blocked_hosts(text, egress_allowed_hosts(policy, trusted_text))):
        out = out[:start] + REDACTION + out[end:]
    return out


def _type_ok(value: Any, arg_type: str) -> bool:
    if arg_type == "number":
        return (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    if arg_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, str)


def _internal_scope(spec: ToolSpec, call: ToolCall, egress: EgressPolicy | None) -> frozenset[str]:
    """Arguments of ``call`` whose text goes only to the owner's own domain, so ``scan_hosts``
    does not apply to them. A recipient argument is one with ``allowed_email_domains`` (it
    covers every ``scan_hosts`` argument of the tool) or one declaring ``recipient_of`` (it
    covers the arguments named). Every recipient that covers an argument must be a single plain
    e-mail address in ``egress.internal_domains``; anything else leaves the scan on."""
    if egress is None or not egress.internal_domains:
        return frozenset()
    scanned = {n for n, a in spec.args.items() if a.scan_hosts and n in call.args}
    exempt: set[str] = set()
    external: set[str] = set()
    for name, arg in spec.args.items():
        if arg.allowed_email_domains is None and arg.recipient_of is None:
            continue
        covers = set(arg.recipient_of) if arg.recipient_of is not None else scanned
        value = call.args.get(name)
        match = _EMAIL_RE.match(value) if isinstance(value, str) else None
        if match is not None and _host_ok(match.group(1).casefold(), egress.internal_domains):
            exempt |= covers
        else:
            external |= covers
    return frozenset((exempt & scanned) - external)


def _check_arg(
    name: str,
    value: Any,
    labels: Labels,
    policy: ArgPolicy,
    out: _Violations,
    egress_hosts: tuple[str, ...] | None = None,
) -> None:
    bad = sorted(untrusted_labels(labels))
    if bad and not policy.allow_untrusted:
        out.add("tool-arg-untrusted", f"{name}: derives from untrusted input ({bad})", name)
    elif bad and policy.untrusted_sources is not None:
        allowed = {UNTRUSTED_PREFIX + s for s in policy.untrusted_sources}
        other = [label for label in bad if label not in allowed]
        if other:
            out.add(
                "tool-arg-untrusted",
                f"{name}: derives from untrusted input ({other}) outside the allowed "
                f"sources ({sorted(allowed)})",
                name,
            )
    if not _type_ok(value, policy.type):
        out.add(
            "tool-arg-type", f"{name}: expected {policy.type}, got {type(value).__name__}", name
        )
        return
    if policy.type == "number":
        if policy.minimum is not None and value < policy.minimum:
            out.add("tool-arg-range", f"{name}: {value} is below min {policy.minimum}", name)
        if policy.maximum is not None and value > policy.maximum:
            out.add("tool-arg-range", f"{name}: {value} is above max {policy.maximum}", name)
    if policy.allowed_values is not None and value not in policy.allowed_values:
        out.add("tool-arg-value", f"{name}: {value!r} is not an allowed value", name)
    if policy.allowed_hosts is not None:
        _check_url(name, value, policy.allowed_hosts, out)
    if policy.allowed_email_domains is not None:
        _check_email(name, value, policy.allowed_email_domains, out)
    if policy.scan_hosts and egress_hosts is not None:
        for _, _, host in _blocked_hosts(value, egress_hosts):
            out.add(
                "tool-arg-egress", f"{name}: names host {host!r}, which is not allow-listed", name
            )


def check_tool_call(
    call: ToolCall,
    labels: Mapping[str, Iterable[str]],
    policy: ToolPolicy,
    session: Session | None = None,
    trusted_text: Iterable[str] = (),
) -> Decision:
    """Check one proposed tool call against ``policy``; return a safegate ``Decision``.

    ``labels`` maps each argument name to its provenance labels; an argument without an entry
    counts as ``untrusted:unlabeled``. Without ``session`` only the per-argument policies run.
    With one, the Rule of Two is checked as well: the call is refused (or escalated, per
    ``policy.rule_of_two``) when the session, this call's arguments and the call's declared
    result labels together hold untrusted input and sensitive data, and the session has taken
    or this call would take a consequential action. The session is not modified; use
    ``Session.admit`` to check and record.

    A tool with ``untrusted_session: escalate | refuse`` is also escalated or refused (rule id
    ``tool-untrusted-session``) when the session labels and this call's argument labels hold
    untrusted data, whether or not the session holds sensitive data. Without ``session`` it is
    not checked.

    ``allow`` when everything holds, otherwise the most severe of ``policy.on_violation`` (per
    argument problems), ``policy.rule_of_two`` and the tool's ``untrusted_session``. An unknown
    tool is always refused.

    With an ``egress`` block, arguments marked ``scan_hosts`` must not name a host outside
    ``egress_allowed_hosts(policy, trusted_text)`` (rule id ``tool-arg-egress``).
    """
    spec = policy.tools.get(call.tool)
    if spec is None:
        return Decision(
            action="refuse",
            rule_ids=["tool-unknown"],
            reasons=[f"tool {call.tool!r} is not declared in the policy"],
            category=TOOL_CATEGORY,
            score=1.0,
            metadata={"tool": call.tool, "capability": None, "violations": []},
        )

    out = _Violations()
    egress_hosts = (
        egress_allowed_hosts(policy, trusted_text)
        if policy.egress is not None and any(a.scan_hosts for a in spec.args.values())
        else None
    )
    internal = _internal_scope(spec, call, policy.egress)
    for name in call.args:
        if name not in spec.args:
            out.add("tool-arg-unknown", f"{name}: argument is not declared for {call.tool}", name)
    for name, arg_policy in spec.args.items():
        if name not in call.args:
            if arg_policy.required:
                out.add("tool-arg-missing", f"{name}: required argument is missing", name)
            continue
        _check_arg(
            name,
            call.args[name],
            _arg_labels(name, labels),
            arg_policy,
            out,
            None if name in internal else egress_hosts,
        )

    actions: list[str] = []
    if out.codes:
        actions.append(policy.on_violation)

    call_labels = _call_labels(call, labels)
    if (
        session is not None
        and spec.untrusted_session != "allow"
        and is_untrusted(union(session.labels, call_labels))
    ):
        out.add(
            "tool-untrusted-session",
            f"{call.tool} is not run in a session that has read untrusted data",
        )
        actions.append(spec.untrusted_session)
    two: dict[str, bool] | None = None
    if session is not None and policy.rule_of_two != "off":
        seen = union(session.labels, call_labels, spec.result_labels)
        two = {
            "untrusted_input": is_untrusted(seen),
            "sensitive_data": is_sensitive(seen),
            "consequential_action": session.consequential or spec.consequential,
        }
        if all(two.values()):
            out.add(
                "tool-rule-of-two",
                "session would combine untrusted input, sensitive data and a "
                f"consequential action ({spec.capability} {call.tool})",
            )
            actions.append(policy.rule_of_two)

    metadata: dict[str, Any] = {
        "tool": call.tool,
        "capability": spec.capability,
        "labels": {n: sorted(_arg_labels(n, labels)) for n in call.args},
        "rule_of_two": two,
        "violations": out.details,
    }
    if not out.codes:
        return Decision(action="allow", rule_ids=[], reasons=[], metadata=metadata)
    action = max(actions, key=_SEVERITY.__getitem__)
    return Decision(
        action=action,
        rule_ids=list(dict.fromkeys(out.codes)),
        reasons=out.reasons,
        category=TOOL_CATEGORY,
        score=1.0,
        metadata=metadata,
    )

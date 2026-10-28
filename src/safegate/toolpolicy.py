"""Deterministic capability and data-flow control for tool calls (CaMeL-lite).

``spec.py`` checks the structured spec a model emits against contract bounds. This module
checks the *action* the agent is about to take. Tools are declared in YAML with a capability
class and a policy per argument; every argument value arrives with provenance labels
(``safegate.flow``). A call is allowed only when each argument satisfies its policy:

* ``untrusted: forbid`` (the default): the value must not derive from untrusted text,
* numeric ``min`` / ``max``, an ``allowed_values`` list, ``allowed_hosts`` for URLs,
  ``allowed_email_domains`` for recipients.

A session check (Meta's "Rule of Two", Willison's "lethal trifecta") adds a second, coarser
rule: a session must not combine untrusted input, sensitive data and a consequential action
(``state-change`` or ``external-send``). The call that would complete the combination is
refused or escalated to a human.

``check_tool_call`` returns the same ``Decision`` as the text gate and ``check_spec``.

Limits, by design:

* Text-only attacks are out of scope: a poisoned summary or a phishing link in the reply
  makes no tool call.
* When the user legitimately routes untrusted data into an argument the policy allows
  (``untrusted: allow``), the gate checks bounds, not truth, as ``spec.py`` does.
* The Rule of Two is session-wide and coarse. It cannot tell that a state change was
  triggered by an injection when the arguments are clean (no arguments, or constants).
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
    Labeled,
    Labels,
    is_sensitive,
    is_untrusted,
    parse_labels,
    union,
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

_ROOT_KEYS = {"version", "on_violation", "rule_of_two", "tools"}
_TOOL_KEYS = {"capability", "description", "result_labels", "args"}
_ARG_KEYS = {
    "type",
    "required",
    "untrusted",
    "min",
    "max",
    "allowed_values",
    "allowed_hosts",
    "allowed_email_domains",
}

_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$")
_EMAIL_RE = re.compile(r"^[^\s@,;<>()\"'\\]+@([A-Za-z0-9.-]+)$")
_URL_FORBIDDEN_RE = re.compile(r"[\\\s\x00-\x1f\x7f]")


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
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str | float, ...] | None = None
    allowed_hosts: tuple[str, ...] | None = None
    allowed_email_domains: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolSpec:
    """One declared tool: its capability class, the labels its result carries, its arguments."""

    name: str
    capability: str
    args: dict[str, ArgPolicy]
    result_labels: Labels = frozenset()
    description: str = ""

    @property
    def consequential(self) -> bool:
        """True for ``state-change`` and ``external-send``."""
        return self.capability in CONSEQUENTIAL


@dataclass(frozen=True)
class ToolPolicy:
    """Declared tools plus the action taken on a violation. Tools not listed are refused."""

    tools: dict[str, ToolSpec]
    on_violation: str = "refuse"
    rule_of_two: str = "escalate"

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
        return cls(tools=tools, on_violation=on_violation, rule_of_two=rule_of_two)

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
    return ArgPolicy(
        type=arg_type,
        required=required,
        allow_untrusted=mode == "allow",
        minimum=minimum,
        maximum=maximum,
        allowed_values=allowed_values,
        allowed_hosts=hosts,
        allowed_email_domains=emails,
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
    return ToolSpec(name, capability, args, result_labels, description)


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


def _type_ok(value: Any, arg_type: str) -> bool:
    if arg_type == "number":
        return (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    if arg_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, str)


def _check_arg(name: str, value: Any, labels: Labels, policy: ArgPolicy, out: _Violations) -> None:
    if not policy.allow_untrusted and is_untrusted(labels):
        bad = sorted(label for label in labels if label.startswith("untrusted:"))
        out.add("tool-arg-untrusted", f"{name}: derives from untrusted input ({bad})", name)
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


def check_tool_call(
    call: ToolCall,
    labels: Mapping[str, Iterable[str]],
    policy: ToolPolicy,
    session: Session | None = None,
) -> Decision:
    """Check one proposed tool call against ``policy``; return a safegate ``Decision``.

    ``labels`` maps each argument name to its provenance labels; an argument without an entry
    counts as ``untrusted:unlabeled``. Without ``session`` only the per-argument policies run.
    With one, the Rule of Two is checked as well: the call is refused (or escalated, per
    ``policy.rule_of_two``) when the session, this call's arguments and the call's declared
    result labels together hold untrusted input and sensitive data, and the session has taken
    or this call would take a consequential action. The session is not modified; use
    ``Session.admit`` to check and record.

    ``allow`` when everything holds, otherwise the most severe of ``policy.on_violation`` (per
    argument problems) and ``policy.rule_of_two``. An unknown tool is always refused.
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
    for name in call.args:
        if name not in spec.args:
            out.add("tool-arg-unknown", f"{name}: argument is not declared for {call.tool}", name)
    for name, arg_policy in spec.args.items():
        if name not in call.args:
            if arg_policy.required:
                out.add("tool-arg-missing", f"{name}: required argument is missing", name)
            continue
        _check_arg(name, call.args[name], _arg_labels(name, labels), arg_policy, out)

    actions: list[str] = []
    if out.codes:
        actions.append(policy.on_violation)

    call_labels = _call_labels(call, labels)
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

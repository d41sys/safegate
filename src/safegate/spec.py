"""Deterministic gate on a machine-readable specification emitted by an LLM.

The text gate (``SafetyGate``) reads the words that go into the model. This module checks
the structured output that comes out, before it reaches an optimiser or controller: a list
of constraints such as ``{"kind": "power_cap_kw", "value": 30, "start": "07:00",
"end": "10:00", "source_id": "m1"}``. A strict YAML policy of contract bounds says which
kinds may appear, in which value range and time window, how far they may move away from
the last trusted spec, and which provenance they must carry.

Limit, by design: the gate checks bounds, not truth. A forged edit that stays inside every
bound is accepted. The policy only narrows how much harm such an edit can do.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from safegate.gate import Decision

SPEC_CATEGORY = "SPEC_VIOLATION"
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_VIOLATION_ACTIONS = ("warn", "refuse", "escalate")

_ROOT_KEYS = {"version", "count", "provenance", "on_violation", "kinds"}
_COUNT_KEYS = {"min", "max"}
_PROVENANCE_KEYS = {"require_source_id", "trusted_source_ids"}
_KIND_KEYS = {"value", "window", "max_count", "max_delta"}
_RANGE_KEYS = {"min", "max"}
_WINDOW_KEYS = {"earliest", "latest"}
_DELTA_KEYS = {"value", "minutes"}


class SpecPolicyError(ValueError):
    """Raised when a spec policy contains unknown keys, bad types, or inconsistent bounds."""


def parse_hhmm(text: str) -> int:
    """Parse ``HH:MM`` into minutes since midnight. ``24:00`` is allowed as an end of day."""
    match = _TIME_RE.match(text.strip())
    if match is None:
        raise ValueError(f"not a HH:MM time: {text!r}")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes > 59 or hours > 24 or (hours == 24 and minutes != 0):
        raise ValueError(f"time out of range: {text!r}")
    return hours * 60 + minutes


@dataclass(frozen=True)
class KindBounds:
    """Bounds for one constraint kind."""

    value_min: float
    value_max: float
    earliest: int | None = None  # minutes since midnight
    latest: int | None = None
    max_count: int | None = None
    max_delta_value: float | None = None
    max_delta_minutes: int | None = None


@dataclass(frozen=True)
class SpecPolicy:
    """Contract bounds for an emitted specification. Kinds not listed are refused."""

    kinds: dict[str, KindBounds]
    count_min: int | None = None
    count_max: int | None = None
    require_source_id: bool = False
    trusted_source_ids: frozenset[str] = frozenset()
    on_violation: str = "refuse"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpecPolicy:
        """Build a policy from parsed YAML; unknown keys and bad values raise."""
        _no_unknown(data, _ROOT_KEYS, "policy")
        version = data.get("version", 1)
        if version != 1:
            raise SpecPolicyError(f"unsupported policy version: {version!r}")

        on_violation = data.get("on_violation", "refuse")
        if on_violation not in _VIOLATION_ACTIONS:
            raise SpecPolicyError(f"on_violation must be one of {_VIOLATION_ACTIONS}")

        kinds_raw = data.get("kinds")
        if not isinstance(kinds_raw, Mapping) or not kinds_raw:
            raise SpecPolicyError("'kinds' must be a non-empty mapping")
        kinds = {str(name): _parse_kind(str(name), body) for name, body in kinds_raw.items()}

        count_min, count_max = _parse_count(data.get("count"))
        require_source_id, trusted = _parse_provenance(data.get("provenance"))
        return cls(
            kinds=kinds,
            count_min=count_min,
            count_max=count_max,
            require_source_id=require_source_id,
            trusted_source_ids=trusted,
            on_violation=on_violation,
        )


def _no_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise SpecPolicyError(f"Unknown keys in {where}: {sorted(map(str, unknown))}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecPolicyError(f"{where} must be a mapping")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecPolicyError(f"{where} must be a finite number")
    return float(value)


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecPolicyError(f"{where} must be a non-negative integer")
    return value


def _clock(value: Any, where: str) -> int:
    if not isinstance(value, str):
        raise SpecPolicyError(f"{where} must be a quoted HH:MM string")
    try:
        return parse_hhmm(value)
    except ValueError as exc:
        raise SpecPolicyError(f"{where}: {exc}") from exc


def _parse_count(raw: Any) -> tuple[int | None, int | None]:
    if raw is None:
        return None, None
    body = _mapping(raw, "count")
    _no_unknown(body, _COUNT_KEYS, "count")
    low = _integer(body["min"], "count.min") if "min" in body else None
    high = _integer(body["max"], "count.max") if "max" in body else None
    if low is not None and high is not None and low > high:
        raise SpecPolicyError("count.min exceeds count.max")
    return low, high


def _parse_provenance(raw: Any) -> tuple[bool, frozenset[str]]:
    if raw is None:
        return False, frozenset()
    body = _mapping(raw, "provenance")
    _no_unknown(body, _PROVENANCE_KEYS, "provenance")
    require = body.get("require_source_id", False)
    if not isinstance(require, bool):
        raise SpecPolicyError("provenance.require_source_id must be a boolean")
    ids = body.get("trusted_source_ids", [])
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise SpecPolicyError("provenance.trusted_source_ids must be a list of non-empty strings")
    if ids and not require:
        raise SpecPolicyError("provenance.trusted_source_ids needs require_source_id: true")
    return require, frozenset(ids)


def _parse_kind(name: str, raw: Any) -> KindBounds:
    where = f"kinds.{name}"
    body = _mapping(raw, where)
    _no_unknown(body, _KIND_KEYS, where)
    if "value" not in body:
        raise SpecPolicyError(f"{where}.value (min/max) is required")
    value = _mapping(body["value"], f"{where}.value")
    _no_unknown(value, _RANGE_KEYS, f"{where}.value")
    if set(value) != _RANGE_KEYS:
        raise SpecPolicyError(f"{where}.value needs both min and max")
    value_min = _number(value["min"], f"{where}.value.min")
    value_max = _number(value["max"], f"{where}.value.max")
    if value_min > value_max:
        raise SpecPolicyError(f"{where}.value.min exceeds max")

    earliest = latest = None
    if "window" in body:
        window = _mapping(body["window"], f"{where}.window")
        _no_unknown(window, _WINDOW_KEYS, f"{where}.window")
        if set(window) != _WINDOW_KEYS:
            raise SpecPolicyError(f"{where}.window needs both earliest and latest")
        earliest = _clock(window["earliest"], f"{where}.window.earliest")
        latest = _clock(window["latest"], f"{where}.window.latest")
        if earliest >= latest:
            raise SpecPolicyError(f"{where}.window.earliest must be before latest")

    max_count = _integer(body["max_count"], f"{where}.max_count") if "max_count" in body else None

    delta_value = delta_minutes = None
    if "max_delta" in body:
        delta = _mapping(body["max_delta"], f"{where}.max_delta")
        _no_unknown(delta, _DELTA_KEYS, f"{where}.max_delta")
        if set(delta) != _DELTA_KEYS:
            raise SpecPolicyError(f"{where}.max_delta needs both value and minutes")
        delta_value = _number(delta["value"], f"{where}.max_delta.value")
        delta_minutes = _integer(delta["minutes"], f"{where}.max_delta.minutes")
        if delta_value < 0:
            raise SpecPolicyError(f"{where}.max_delta.value must be >= 0")

    return KindBounds(
        value_min=value_min,
        value_max=value_max,
        earliest=earliest,
        latest=latest,
        max_count=max_count,
        max_delta_value=delta_value,
        max_delta_minutes=delta_minutes,
    )


def load_spec_policy(source: Path | str) -> SpecPolicy:
    """Load a policy from a YAML file path or a YAML string. Fails closed on any error."""
    if isinstance(source, Path) or ("\n" not in source and Path(source).exists()):
        content = Path(source).read_text(encoding="utf-8")
    else:
        content = str(source)
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise SpecPolicyError(f"Failed to parse YAML spec policy: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SpecPolicyError("Spec policy root must be a YAML mapping")
    return SpecPolicy.from_dict(parsed)


@dataclass(frozen=True)
class SpecConstraint:
    """One validated constraint of an emitted spec."""

    kind: str
    value: float
    start: int  # minutes since midnight
    end: int
    source_id: str | None


@dataclass
class _Violations:
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, reason: str, index: int | None = None) -> None:
        self.codes.append(code)
        self.reasons.append(reason)
        self.details.append({"code": code, "index": index, "reason": reason})


def _parse_constraint(raw: Any) -> SpecConstraint:
    """Strictly parse one emitted constraint; raise ValueError with the reason."""
    if not isinstance(raw, Mapping):
        raise ValueError("constraint is not an object")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("'kind' must be a non-empty string")
    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("'value' must be a finite number")
    start, end = raw.get("start"), raw.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("'start' and 'end' must be HH:MM strings")
    source_id = raw.get("source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("'source_id' must be a string")
    return SpecConstraint(kind, float(value), parse_hhmm(start), parse_hhmm(end), source_id)


def _parse_spec(spec: Any) -> list[SpecConstraint]:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("constraints"), list):
        raise ValueError("spec must be an object with a 'constraints' list")
    parsed: list[SpecConstraint] = []
    for i, raw in enumerate(spec["constraints"]):
        try:
            parsed.append(_parse_constraint(raw))
        except ValueError as exc:
            raise ValueError(f"constraints[{i}]: {exc}") from exc
    return parsed


def _within_delta(new: SpecConstraint, old: SpecConstraint, bounds: KindBounds) -> bool:
    if bounds.max_delta_value is not None and abs(new.value - old.value) > bounds.max_delta_value:
        return False
    if bounds.max_delta_minutes is not None:
        limit = bounds.max_delta_minutes
        return abs(new.start - old.start) <= limit and abs(new.end - old.end) <= limit
    return True


def _check_constraint(
    i: int,
    c: SpecConstraint,
    policy: SpecPolicy,
    previous: list[SpecConstraint] | None,
    out: _Violations,
) -> None:
    bounds = policy.kinds.get(c.kind)
    if bounds is None:
        out.add("spec-kind", f"constraints[{i}]: kind {c.kind!r} is not allowed by the policy", i)
        return
    if not bounds.value_min <= c.value <= bounds.value_max:
        out.add(
            "spec-value",
            f"constraints[{i}] {c.kind}: value {c.value:g} outside "
            f"[{bounds.value_min:g}, {bounds.value_max:g}]",
            i,
        )
    if c.start >= c.end:
        out.add("spec-window", f"constraints[{i}] {c.kind}: start is not before end", i)
    elif (bounds.earliest is not None and c.start < bounds.earliest) or (
        bounds.latest is not None and c.end > bounds.latest
    ):
        out.add("spec-window", f"constraints[{i}] {c.kind}: window outside the allowed hours", i)

    if policy.require_source_id:
        if not c.source_id:
            out.add("spec-provenance", f"constraints[{i}] {c.kind}: no source_id", i)
        elif policy.trusted_source_ids and c.source_id not in policy.trusted_source_ids:
            out.add(
                "spec-provenance",
                f"constraints[{i}] {c.kind}: source_id {c.source_id!r} is not trusted",
                i,
            )

    has_delta = bounds.max_delta_value is not None or bounds.max_delta_minutes is not None
    if previous is not None and has_delta:
        same_kind = [p for p in previous if p.kind == c.kind]
        if not same_kind:
            out.add("spec-delta", f"constraints[{i}] {c.kind}: no counterpart in previous spec", i)
        elif not any(_within_delta(c, p, bounds) for p in same_kind):
            out.add("spec-delta", f"constraints[{i}] {c.kind}: moved too far from previous spec", i)


def check_spec(
    spec: Mapping[str, Any],
    policy: SpecPolicy,
    previous: Mapping[str, Any] | None = None,
) -> Decision:
    """Check an emitted spec against ``policy`` and, if given, the last trusted ``previous`` spec.

    Returns a ``Decision`` in the safegate action vocabulary: ``allow`` when every bound
    holds, otherwise ``policy.on_violation``. A spec that does not parse is refused.
    A malformed ``previous`` spec raises ``ValueError``: it is trusted input, not model output.
    Delta bounds are skipped when ``previous`` is None.
    """
    prev = None
    if previous is not None:
        try:
            prev = _parse_spec(previous)
        except ValueError as exc:
            raise ValueError(f"previous spec is malformed: {exc}") from exc

    out = _Violations()
    try:
        constraints = _parse_spec(spec)
    except ValueError as exc:
        out.add("spec-schema", str(exc))
        constraints = []

    if not out.codes:
        n = len(constraints)
        if policy.count_min is not None and n < policy.count_min:
            out.add("spec-count", f"{n} constraints, policy needs at least {policy.count_min}")
        if policy.count_max is not None and n > policy.count_max:
            out.add("spec-count", f"{n} constraints, policy allows at most {policy.count_max}")
        for i, c in enumerate(constraints):
            _check_constraint(i, c, policy, prev, out)
        for kind, bounds in policy.kinds.items():
            n_kind = sum(c.kind == kind for c in constraints)
            if bounds.max_count is not None and n_kind > bounds.max_count:
                out.add("spec-kind-count", f"{n_kind} x {kind}, policy allows {bounds.max_count}")

    metadata: dict[str, Any] = {
        "n_constraints": len(constraints),
        "delta_checked": prev is not None,
        "violations": out.details,
    }
    if not out.codes:
        return Decision(action="allow", rule_ids=[], reasons=[], metadata=metadata)
    return Decision(
        action=policy.on_violation,
        rule_ids=list(dict.fromkeys(out.codes)),
        reasons=out.reasons,
        category=SPEC_CATEGORY,
        score=1.0,
        metadata=metadata,
    )

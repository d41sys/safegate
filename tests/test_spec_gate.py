"""Offline tests for the spec gate, on the synthetic fixtures in examples/spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from safegate.cli import main
from safegate.spec import SpecPolicy, SpecPolicyError, check_spec, load_spec_policy, parse_hhmm

EXAMPLES = Path(__file__).parent.parent / "examples" / "spec"
POLICY = load_spec_policy(EXAMPLES / "policy.yaml")


def _load(name: str) -> dict[str, Any]:
    data = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _cap(**overrides: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "kind": "power_cap_kw",
        "value": 40,
        "start": "07:00",
        "end": "10:00",
        "source_id": "msg-001",
    }
    c.update(overrides)
    return {"constraints": [c]}


def test_genuine_spec_is_allowed_with_previous() -> None:
    d = check_spec(_load("genuine.json"), POLICY, _load("previous.json"))
    assert d.action == "allow"
    assert d.rule_ids == []
    assert d.metadata["delta_checked"] is True


def test_out_of_bounds_forgery_is_refused_with_every_broken_bound() -> None:
    d = check_spec(_load("forged_out_of_bounds.json"), POLICY, _load("previous.json"))
    assert d.action == "refuse"
    assert d.category == "SPEC_VIOLATION"
    assert set(d.rule_ids) == {
        "spec-value",
        "spec-window",
        "spec-provenance",
        "spec-delta",
    }


def test_in_bounds_forgery_passes_by_design() -> None:
    """The stated limit: a forged edit that respects every bound is accepted."""
    d = check_spec(_load("forged_in_bounds.json"), POLICY, _load("previous.json"))
    assert d.action == "allow"


def test_delta_bound_needs_previous_and_is_skipped_without_it() -> None:
    far = _cap(value=60)  # in range, 20 kW from previous 40 > max_delta 10
    assert check_spec(far, POLICY).action == "allow"
    d = check_spec(far, POLICY, _load("previous.json"))
    assert d.action == "refuse"
    assert d.rule_ids == ["spec-delta"]


def test_kind_absent_from_previous_breaks_delta() -> None:
    policy = SpecPolicy.from_dict(
        {
            "kinds": {
                "reserve_frac": {
                    "value": {"min": 0, "max": 1},
                    "max_delta": {"value": 0.1, "minutes": 30},
                }
            }
        }
    )
    spec = {
        "constraints": [{"kind": "reserve_frac", "value": 0.2, "start": "08:00", "end": "09:00"}]
    }
    assert check_spec(spec, policy).action == "allow"
    assert check_spec(spec, policy, _cap()).rule_ids == ["spec-delta"]


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        (_cap(kind="unit_outage"), "spec-kind"),
        (_cap(value=19.9), "spec-value"),
        (_cap(value=60.1, start="07:00"), "spec-value"),
        (_cap(start="05:59"), "spec-window"),
        (_cap(end="12:01"), "spec-window"),
        (_cap(start="10:00", end="07:00"), "spec-window"),
        (_cap(source_id=None), "spec-provenance"),
        (_cap(source_id="msg-002"), "spec-provenance"),
        ({"constraints": []}, "spec-count"),
        ({"constraints": [_cap()["constraints"][0]] * 3}, "spec-count"),
    ],
)
def test_each_bound_fires_its_own_rule(spec: dict[str, Any], code: str) -> None:
    d = check_spec(spec, POLICY)
    assert d.action == "refuse"
    assert code in d.rule_ids


def test_max_count_per_kind() -> None:
    two = {"constraints": [_cap()["constraints"][0], _cap(value=41)["constraints"][0]]}
    d = check_spec(two, POLICY)
    assert d.rule_ids == ["spec-kind-count"]


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"constraints": "nope"},
        {"constraints": [1]},
        {
            "constraints": [
                {"kind": "power_cap_kw", "value": True, "start": "07:00", "end": "08:00"}
            ]
        },
        {
            "constraints": [
                {"kind": "power_cap_kw", "value": float("nan"), "start": "07:00", "end": "08:00"}
            ]
        },
        {"constraints": [{"kind": "power_cap_kw", "value": 30, "start": "7am", "end": "08:00"}]},
        {
            "constraints": [
                {"kind": "power_cap_kw", "value": "30", "start": "07:00", "end": "08:00"}
            ]
        },
    ],
)
def test_malformed_spec_is_refused_not_raised(spec: dict[str, Any]) -> None:
    d = check_spec(spec, POLICY)
    assert d.action == "refuse"
    assert d.rule_ids == ["spec-schema"]


def test_malformed_previous_raises() -> None:
    with pytest.raises(ValueError, match="previous spec is malformed"):
        check_spec(_cap(), POLICY, {"constraints": [{"kind": 1}]})


def test_on_violation_action_vocabulary() -> None:
    warn = SpecPolicy.from_dict(
        {"on_violation": "warn", "kinds": {"power_cap_kw": {"value": {"min": 0, "max": 1}}}}
    )
    assert check_spec(_cap(), warn).action == "warn"
    with pytest.raises(SpecPolicyError, match="on_violation"):
        SpecPolicy.from_dict(
            {"on_violation": "allow", "kinds": {"k": {"value": {"min": 0, "max": 1}}}}
        )


@pytest.mark.parametrize(
    "yaml_text",
    [
        "version: 1\n",  # no kinds
        "version: 2\nkinds: {k: {value: {min: 0, max: 1}}}\n",
        "kinds: {k: {value: {min: 0, max: 1}}}\nextra: 1\n",
        "kinds: {k: {value: {min: 0, max: 1}, typo: 1}}\n",
        "kinds: {k: {value: {min: 2, max: 1}}}\n",
        "kinds: {k: {value: {min: 0}}}\n",
        "kinds: {k: {}}\n",
        "kinds: {k: {value: {min: 0, max: 1}, window: {earliest: '10:00', latest: '09:00'}}}\n",
        "kinds: {k: {value: {min: 0, max: 1}, window: {earliest: 600, latest: 700}}}\n",
        "kinds: {k: {value: {min: 0, max: 1}, max_delta: {value: 1}}}\n",
        "kinds: {k: {value: {min: 0, max: 1}}}\ncount: {min: 3, max: 1}\n",
        "kinds: {k: {value: {min: 0, max: 1}}}\nprovenance: {trusted_source_ids: [a]}\n",
        "kinds: {k: {value: {min: .nan, max: 1}}}\n",
        "- not a mapping\n",
        "kinds: [unclosed\n",
    ],
)
def test_policy_fails_closed(yaml_text: str) -> None:
    with pytest.raises(SpecPolicyError):
        load_spec_policy(yaml_text)


def test_parse_hhmm() -> None:
    assert parse_hhmm("00:00") == 0
    assert parse_hhmm("7:05") == 425
    assert parse_hhmm("24:00") == 1440
    for bad in ("24:01", "12:60", "noon", "7", ""):
        with pytest.raises(ValueError, match=r"HH:MM|range"):
            parse_hhmm(bad)


def test_cli_exit_codes_and_output(capsys: pytest.CaptureFixture[str]) -> None:
    policy, prev = str(EXAMPLES / "policy.yaml"), str(EXAMPLES / "previous.json")
    assert (
        main(["spec-check", str(EXAMPLES / "genuine.json"), "--policy", policy, "--previous", prev])
        == 0
    )
    assert "action:   allow" in capsys.readouterr().out

    assert (
        main(["spec-check", str(EXAMPLES / "forged_out_of_bounds.json"), "--policy", policy]) == 1
    )
    out = capsys.readouterr().out
    assert "action:   refuse" in out
    assert "delta:    skipped" in out

    assert (
        main(
            [
                "spec-check",
                str(EXAMPLES / "forged_in_bounds.json"),
                "--policy",
                policy,
                "--previous",
                prev,
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["action"] == "allow"


def test_cli_bad_input_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert main(["spec-check", str(bad), "--policy", str(EXAMPLES / "policy.yaml")]) == 2
    assert "top-level JSON value" in capsys.readouterr().err
    assert (
        main(
            [
                "spec-check",
                str(tmp_path / "missing.json"),
                "--policy",
                str(EXAMPLES / "policy.yaml"),
            ]
        )
        == 2
    )

"""Offline tests for provenance labels, the tool policy and the Rule-of-Two session check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from safegate import flow
from safegate.cli import main
from safegate.flow import Labeled, LabelError
from safegate.toolpolicy import (
    Session,
    ToolCall,
    ToolPolicyError,
    call_from_labeled,
    check_tool_call,
    load_tool_policy,
)

EXAMPLES = Path(__file__).parent.parent / "examples" / "tools"
POLICY = load_tool_policy(EXAMPLES / "energy.yaml")


def _check(tool: str, args: dict[str, Any], labels: dict[str, list[str]], **kw: Any) -> Any:
    return check_tool_call(ToolCall(tool, args), labels, POLICY, **kw)


# --- labels ----------------------------------------------------------------------------------


def test_labels_validate() -> None:
    assert flow.untrusted("email") == "untrusted:email"
    assert flow.sensitive("meter") == "sensitive:meter"
    for bad in ("untrusted:", "untrusted:a b", "admin", "trusted", ""):
        with pytest.raises(LabelError):
            flow.validate_label(bad)
    with pytest.raises(LabelError):
        flow.parse_labels("trusted_user")  # type: ignore[arg-type]


def test_combine_unions_labels_through_a_transformation() -> None:
    body = flow.from_untrusted("cap is 0.0 kw", "email")
    number = flow.combine(lambda t: float(t.split()[2]), body)
    assert number.value == 0.0
    assert number.labels == {"untrusted:email"}
    scaled = flow.combine(lambda a, b: a * b, number, flow.user(2))
    assert scaled.labels == {"untrusted:email", "trusted_user"}


def test_concat_fmt_and_derive_union_labels() -> None:
    a, b = flow.user("a"), flow.from_untrusted("b", "web")
    assert flow.concat(a, b, sep="-").value == "a-b"
    assert flow.concat(a, b).labels == {"trusted_user", "untrusted:web"}
    text = flow.fmt("sum {x}", x=b)
    assert text.value == "sum b"
    assert text.labels == {"system", "untrusted:web"}
    assert flow.derive(3, a, b).labels == {"trusted_user", "untrusted:web"}


def test_labeled_refuses_the_ways_a_label_is_dropped() -> None:
    x = flow.user("v")
    with pytest.raises(TypeError):
        str(x)
    with pytest.raises(TypeError):
        f"{x}"
    with pytest.raises(TypeError):
        x + x  # type: ignore[operator]
    with pytest.raises(TypeError):
        flow.combine(len, "plain")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        flow.concat(x, "plain")  # type: ignore[arg-type]


# --- policy loading --------------------------------------------------------------------------


def test_example_policy_loads() -> None:
    assert POLICY.tools["set_export_cap"].capability == "state-change"
    assert POLICY.tools["read_inbox"].result_labels == {"untrusted:email"}
    assert POLICY.tools["add_note"].args["text"].allow_untrusted is True
    assert POLICY.tools["set_export_cap"].args["cap_kw"].allow_untrusted is False
    assert POLICY.tools["reboot_inverter"].untrusted_session == "escalate"
    # off unless declared: schedule_charge allows untrusted arguments on purpose
    assert POLICY.tools["schedule_charge"].untrusted_session == "allow"
    assert all(
        s.untrusted_session == "allow" for n, s in POLICY.tools.items() if n != "reboot_inverter"
    )


@pytest.mark.parametrize(
    "text",
    [
        "tools: {}",
        "version: 2\ntools: {t: {capability: read}}",
        "bogus: 1\ntools: {t: {capability: read}}",
        "tools: {t: {capability: network}}",
        "tools: {t: {capability: read, extra: 1}}",
        "on_violation: allow\ntools: {t: {capability: read}}",
        "rule_of_two: maybe\ntools: {t: {capability: read}}",
        "tools: {t: {capability: read, args: {a: {type: number, min: 5, max: 1}}}}",
        "tools: {t: {capability: read, args: {a: {type: string, min: 1}}}}",
        "tools: {t: {capability: read, args: {a: {type: number, allowed_hosts: [x.example]}}}}",
        "tools: {t: {capability: read, args: {a: {type: string, untrusted: sometimes}}}}",
        "tools: {t: {capability: read, args: {a: {type: string, allowed_hosts: ['a b']}}}}",
        "tools: {t: {capability: read, result_labels: [secret]}}",
        "tools: {t: {capability: read, args: {a: {type: number, min: .nan}}}}",
        "tools: {t: {capability: read, untrusted_session: escalate}}",
        "tools: {t: {capability: write, untrusted_session: refuse}}",
        "tools: {t: {capability: state-change, untrusted_session: maybe}}",
        "- a list",
    ],
)
def test_bad_policies_fail_closed(text: str) -> None:
    with pytest.raises(ToolPolicyError):
        load_tool_policy(text)


# --- per-argument checks ---------------------------------------------------------------------


def test_trusted_in_bounds_call_is_allowed() -> None:
    d = _check("set_export_cap", {"cap_kw": 2.5}, {"cap_kw": ["trusted_user"]})
    assert d.action == "allow"
    assert d.rule_ids == []


def test_untrusted_value_in_forbidden_argument_is_refused_even_in_bounds() -> None:
    d = _check("set_export_cap", {"cap_kw": 0.0}, {"cap_kw": ["untrusted:email"]})
    assert d.action == "refuse"
    assert d.rule_ids == ["tool-arg-untrusted"]
    assert d.category == "TOOL_POLICY"


def test_mixed_labels_are_untrusted() -> None:
    d = _check("set_export_cap", {"cap_kw": 1}, {"cap_kw": ["trusted_user", "untrusted:web"]})
    assert d.rule_ids == ["tool-arg-untrusted"]


def test_missing_labels_fail_closed() -> None:
    d = _check("set_export_cap", {"cap_kw": 2.5}, {})
    assert d.action == "refuse"
    assert d.metadata["labels"] == {"cap_kw": ["untrusted:unlabeled"]}


def test_allowed_untrusted_argument_still_checks_bounds() -> None:
    ok = _check("schedule_charge", {"start_hour": 2}, {"start_hour": ["untrusted:web"]})
    bad = _check("schedule_charge", {"start_hour": 31}, {"start_hour": ["untrusted:web"]})
    assert ok.action == "allow"
    assert bad.rule_ids == ["tool-arg-range"]


def test_range_value_type_and_shape_checks() -> None:
    t = {"cap_kw": ["trusted_user"]}
    assert _check("set_export_cap", {"cap_kw": 25}, t).rule_ids == ["tool-arg-range"]
    assert _check("set_export_cap", {"cap_kw": -1}, t).rule_ids == ["tool-arg-range"]
    assert _check("set_export_cap", {"cap_kw": "2"}, t).rule_ids == ["tool-arg-type"]
    assert _check("set_export_cap", {"cap_kw": True}, t).rule_ids == ["tool-arg-type"]
    assert _check("set_export_cap", {"cap_kw": float("nan")}, t).rule_ids == ["tool-arg-type"]
    assert _check("set_export_cap", {}, t).rule_ids == ["tool-arg-missing"]
    extra = _check("set_export_cap", {"cap_kw": 1, "force": True}, t)
    assert "tool-arg-unknown" in extra.rule_ids


def test_unknown_tool_is_refused() -> None:
    d = _check("delete_logs", {}, {})
    assert d.action == "refuse"
    assert d.rule_ids == ["tool-unknown"]


def test_allowed_values() -> None:
    ok = _check(
        "set_setpoint_recipient", {"recipient": "aggregator-backup"}, {"recipient": ["system"]}
    )
    bad = _check(
        "set_setpoint_recipient", {"recipient": "aggregator-evil"}, {"recipient": ["system"]}
    )
    assert ok.action == "allow"
    assert bad.rule_ids == ["tool-arg-value"]


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://api.grid-operator.example/v1/readings", True),
        ("https://API.grid-operator.example./x", False),  # trailing dot is not the listed host
        ("https://api.grid-operator.example.evil.example/", False),
        ("https://evil.example/api.grid-operator.example", False),
        ("https://api.grid-operator.example@evil.example/", False),
        ("https://evil.example\\@api.grid-operator.example/", False),
        ("https://api.grid-operator.example:99999/", False),
        ("ftp://api.grid-operator.example/", False),
        ("//api.grid-operator.example/", False),
        ("https://api.grid-operator.example /x", False),
        ("https://xn--grid-operator.example/", False),
        ("", False),
    ],
)
def test_url_host_allowlist(url: str, ok: bool) -> None:
    d = _check(
        "http_post", {"url": url, "payload": "x"}, {"url": ["system"], "payload": ["system"]}
    )
    assert (d.action == "allow") is ok, d.reasons


def test_wildcard_host_matches_subdomains_not_the_apex_or_lookalikes() -> None:
    def fetch(url: str) -> str:
        d = _check("fetch_url", {"url": url}, {"url": ["trusted_user"]})
        return str(d.action)

    assert fetch("https://tariff.provider.example/today") == "allow"
    assert fetch("https://a.home.example/x") == "allow"
    assert fetch("https://home.example/x") == "refuse"
    assert fetch("https://evilhome.example/x") == "refuse"
    assert fetch("https://a.home.example.evil.example/x") == "refuse"


@pytest.mark.parametrize(
    ("to", "ok"),
    [
        ("owner@home.example", True),
        ("Owner@HOME.example", True),
        ("owner@home.example,evil@evil.example", False),
        ("owner@home.example;evil@evil.example", False),
        ("<evil@evil.example>", False),
        ("evil@evil.example", False),
        ("owner@home.example.evil.example", False),
        ("owner@evil.example@home.example", False),
        ("home.example", False),
    ],
)
def test_email_domain_allowlist(to: str, ok: bool) -> None:
    labels = {"to": ["trusted_user"], "subject": ["system"], "body": ["system"]}
    d = _check("send_email", {"to": to, "subject": "s", "body": "b"}, labels)
    assert (d.action == "allow") is ok, d.reasons


def test_on_violation_action_is_configurable() -> None:
    text = (
        (EXAMPLES / "energy.yaml")
        .read_text(encoding="utf-8")
        .replace("on_violation: refuse", "on_violation: escalate")
    )
    policy = load_tool_policy(text)
    d = check_tool_call(
        ToolCall("set_export_cap", {"cap_kw": 0}), {"cap_kw": ["untrusted:x"]}, policy
    )
    assert d.action == "escalate"
    # an unknown tool is refused whatever on_violation says
    assert check_tool_call(ToolCall("nope"), {}, policy).action == "refuse"


# --- Rule of Two -----------------------------------------------------------------------------


def test_rule_of_two_needs_all_three_legs() -> None:
    clean = {"labels": ["system"]}
    call = ToolCall("set_battery_setpoint", {"power_kw": 0})
    labels = {"power_kw": ["trusted_user"]}

    def run(labels_seen: list[str], consequential: bool = False) -> str:
        s = Session()
        s.observe(labels_seen)
        s.consequential = consequential
        return str(check_tool_call(call, labels, POLICY, s).action)

    assert run(clean["labels"]) == "allow"
    assert run(["untrusted:email"]) == "allow"  # untrusted + state change, no sensitive data
    assert run(["sensitive:meter"]) == "allow"  # sensitive + state change, no untrusted input
    assert run(["untrusted:email", "sensitive:meter"]) == "escalate"


def test_rule_of_two_is_skipped_without_a_session_or_when_off() -> None:
    call = ToolCall("set_battery_setpoint", {"power_kw": 0})
    labels = {"power_kw": ["trusted_user"]}
    assert check_tool_call(call, labels, POLICY).action == "allow"
    off = load_tool_policy(
        (EXAMPLES / "energy.yaml")
        .read_text(encoding="utf-8")
        .replace("rule_of_two: escalate", "rule_of_two: off")
    )
    s = Session()
    s.observe(["untrusted:email", "sensitive:meter"])
    assert check_tool_call(call, labels, off, s).action == "allow"


def test_rule_of_two_refuse_mode_and_severity_merge() -> None:
    text = (
        (EXAMPLES / "energy.yaml")
        .read_text(encoding="utf-8")
        .replace("rule_of_two: escalate", "rule_of_two: refuse")
    )
    policy = load_tool_policy(text)
    s = Session()
    s.observe(["untrusted:email", "sensitive:meter"])
    d = check_tool_call(
        ToolCall("set_battery_setpoint", {"power_kw": 0}), {"power_kw": ["trusted_user"]}, policy, s
    )
    assert d.action == "refuse"
    assert d.rule_ids == ["tool-rule-of-two"]
    # argument violation (refuse) and Rule of Two (escalate): the stricter wins, both reported
    d2 = check_tool_call(
        ToolCall("set_export_cap", {"cap_kw": 0}), {"cap_kw": ["untrusted:email"]}, POLICY, s
    )
    assert d2.action == "refuse"
    assert set(d2.rule_ids) == {"tool-arg-untrusted", "tool-rule-of-two"}


# --- untrusted_session -----------------------------------------------------------------------


def _with_untrusted_session(mode: str, **replace: str) -> Any:
    text = (EXAMPLES / "energy.yaml").read_text(encoding="utf-8")
    text = text.replace("untrusted_session: escalate ", f"untrusted_session: {mode} ")
    for old, new in replace.items():
        text = text.replace(old, new)
    return load_tool_policy(text)


def test_untrusted_session_needs_untrusted_data_but_not_sensitive_data() -> None:
    call = ToolCall("reboot_inverter")

    def run(labels_seen: list[str], consequential: bool = False) -> Any:
        s = Session()
        s.observe(labels_seen)
        s.consequential = consequential
        return check_tool_call(call, {}, POLICY, s)

    assert run([]).action == "allow"
    assert run(["sensitive:meter"]).action == "allow"
    assert run(["system", "trusted_user"], consequential=True).action == "allow"
    d = run(["untrusted:web"])  # no sensitive data: the Rule of Two is silent
    assert d.action == "escalate"
    assert d.rule_ids == ["tool-untrusted-session"]
    assert d.metadata["rule_of_two"] == {
        "untrusted_input": True,
        "sensitive_data": False,
        "consequential_action": True,
    }
    # both rules fire: the Rule of Two's rule id is added, the action is unchanged
    d = run(["untrusted:email", "sensitive:meter"])
    assert d.action == "escalate"
    assert d.rule_ids == ["tool-untrusted-session", "tool-rule-of-two"]


def test_untrusted_session_is_not_checked_without_a_session_and_is_off_by_default() -> None:
    assert check_tool_call(ToolCall("reboot_inverter"), {}, POLICY).action == "allow"
    allow = _with_untrusted_session("allow")
    s = Session()
    s.observe(["untrusted:email"])
    assert check_tool_call(ToolCall("reboot_inverter"), {}, allow, s).action == "allow"
    # a tool that does not declare it is unaffected by the same session
    d = check_tool_call(
        ToolCall("set_battery_setpoint", {"power_kw": 0}), {"power_kw": ["trusted_user"]}, POLICY, s
    )
    assert d.action == "allow"


def test_untrusted_session_counts_the_call_arguments_as_well_as_the_session() -> None:
    policy = load_tool_policy(
        "tools: {t: {capability: external-send, untrusted_session: refuse,"
        " args: {a: {type: string, untrusted: allow}}}}"
    )
    clean = check_tool_call(ToolCall("t", {"a": "x"}), {"a": ["trusted_user"]}, policy, Session())
    assert clean.action == "allow"
    d = check_tool_call(ToolCall("t", {"a": "x"}), {"a": ["untrusted:web"]}, policy, Session())
    assert d.action == "refuse"
    assert d.rule_ids == ["tool-untrusted-session"]


def test_untrusted_session_refuse_mode_and_severity_merge() -> None:
    s = Session()
    s.observe(["untrusted:email"])
    refuse = _with_untrusted_session("refuse")
    assert check_tool_call(ToolCall("reboot_inverter"), {}, refuse, s).action == "refuse"
    # escalate here, refuse for the Rule of Two: the stricter action wins
    both = _with_untrusted_session("escalate", **{"rule_of_two: escalate": "rule_of_two: refuse"})
    s.observe(["sensitive:meter"])
    d = check_tool_call(ToolCall("reboot_inverter"), {}, both, s)
    assert d.action == "refuse"
    assert set(d.rule_ids) == {"tool-untrusted-session", "tool-rule-of-two"}


def test_untrusted_session_admit_does_not_record_an_escalated_reboot() -> None:
    reboot = ToolCall("reboot_inverter")
    owner = Session()  # the owner asks for the reboot and the session read nothing untrusted
    assert owner.admit(reboot, {}, POLICY).action == "allow"
    assert owner.consequential is True
    s = Session()
    assert s.admit(ToolCall("read_inbox"), {}, POLICY).action == "allow"
    assert s.admit(reboot, {}, POLICY).action == "escalate"
    assert s.consequential is False  # it did not run


def test_session_admit_records_only_executed_calls_and_completes_the_trifecta_in_order() -> None:
    s = Session()
    read_mail = ToolCall("read_inbox")
    read_meter = ToolCall("get_meter_readings")
    push = ToolCall("http_post", {"url": "https://api.grid-operator.example/r", "payload": "x"})
    push_labels = {"url": ["system"], "payload": ["sensitive:meter"]}

    assert s.admit(read_mail, {}, POLICY).action == "allow"
    assert s.labels == {"untrusted:email"}
    assert s.admit(read_meter, {}, POLICY).action == "allow"
    assert s.consequential is False
    blocked = s.admit(push, push_labels, POLICY)
    assert blocked.action == "escalate"
    assert s.consequential is False  # an escalated call did not run

    fresh = Session()
    assert fresh.admit(read_meter, {}, POLICY).action == "allow"
    assert fresh.admit(push, push_labels, POLICY).action == "allow"
    assert fresh.consequential is True
    # reading untrusted input after a consequential action in a sensitive session is blocked too
    assert fresh.admit(read_mail, {}, POLICY).action == "escalate"


def test_label_result_carries_declared_and_argument_labels() -> None:
    r = POLICY.label_result("fetch_url", "<html>", [["trusted_user"]])
    assert r.labels == {"untrusted:web", "trusted_user"}
    with pytest.raises(KeyError):
        POLICY.label_result("nope", 1)


def test_call_from_labeled_round_trip() -> None:
    call, labels = call_from_labeled(
        "set_export_cap", {"cap_kw": flow.from_untrusted(0.0, "email")}
    )
    assert call.args == {"cap_kw": 0.0}
    assert check_tool_call(call, labels, POLICY).action == "refuse"
    with pytest.raises(TypeError):
        call_from_labeled("set_export_cap", {"cap_kw": 1.0})  # type: ignore[dict-item]
    assert isinstance(flow.user(1), Labeled)


# --- CLI -------------------------------------------------------------------------------------


def test_tool_check_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    policy = str(EXAMPLES / "energy.yaml")
    assert main(["tool-check", str(EXAMPLES / "call_user_cap.json"), "--policy", policy]) == 0
    assert main(["tool-check", str(EXAMPLES / "call_email_cap.json"), "--policy", policy]) == 1
    assert (
        main(["tool-check", str(EXAMPLES / "call_email_recipient.json"), "--policy", policy]) == 1
    )
    capsys.readouterr()
    rc = main(
        ["tool-check", str(EXAMPLES / "call_reboot_session.json"), "--policy", policy, "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["action"] == "escalate"
    assert out["rule_ids"] == ["tool-untrusted-session", "tool-rule-of-two"]
    # untrusted data only, no sensitive data: only the per-tool rule fires
    rc = main(
        ["tool-check", str(EXAMPLES / "call_reboot_untrusted.json"), "--policy", policy, "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["action"] == "escalate"
    assert out["rule_ids"] == ["tool-untrusted-session"]
    noegress = str(EXAMPLES / "energy-noegress.yaml")
    call = str(EXAMPLES / "call_reboot_untrusted.json")
    assert main(["tool-check", call, "--policy", noegress]) == 0  # the earlier default


def test_tool_check_cli_bad_input_exits_2(tmp_path: Path) -> None:
    policy = str(EXAMPLES / "energy.yaml")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tool": "set_export_cap", "args": {}, "extra": 1}))
    assert main(["tool-check", str(bad), "--policy", policy]) == 2
    bad.write_text(json.dumps({"tool": "x", "labels": {"a": "untrusted:x"}}))
    assert main(["tool-check", str(bad), "--policy", policy]) == 2
    bad.write_text(
        json.dumps(
            {"tool": "set_export_cap", "labels": {"cap_kw": ["bogus"]}, "args": {"cap_kw": 1}}
        )
    )
    assert main(["tool-check", str(bad), "--policy", policy]) == 2
    assert main(["tool-check", str(tmp_path / "missing.json"), "--policy", policy]) == 2
    assert main(["tool-check", str(bad), "--policy", str(tmp_path / "nope.yaml")]) == 2

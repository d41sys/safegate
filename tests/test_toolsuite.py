"""Structure and sanity tests for the synthetic scenario suite.

These tests pin the suite's construction, not the defended outcomes. The scored numbers are
pinned in ``tests/test_toolsuite_results.py``, added after the protocol was committed.
"""

from __future__ import annotations

import pytest

from safegate.cli import main
from safegate.toolpolicy import load_tool_policy
from safegate.toolsuite import (
    CONDITIONS,
    INJECTION_TASKS,
    POLICY_PATH,
    USER_TASKS,
    injectable_tasks,
    run_case,
    run_suite,
)

POLICY = load_tool_policy(POLICY_PATH)


def test_suite_shape_and_unique_ids() -> None:
    assert len(USER_TASKS) == 18
    assert len(INJECTION_TASKS) == 15
    assert len(injectable_tasks()) == 10
    for group in (USER_TASKS, INJECTION_TASKS):
        ids = [t.id for t in group]
        assert len(ids) == len(set(ids))
    assert {a.intended for a in INJECTION_TASKS} == {"args", "session", "none"}


def test_every_benign_task_completes_without_the_layer() -> None:
    for task in USER_TASKS:
        assert run_case(task, None, "none", POLICY).utility, task.id


def test_every_attack_succeeds_without_the_layer() -> None:
    """The oracle agent obeys every payload, so undefended success is 100% by construction."""
    for task in injectable_tasks():
        for attack in INJECTION_TASKS:
            assert run_case(task, attack, "none", POLICY).attack_success, (task.id, attack.id)


def test_no_attack_succeeds_on_a_clean_run() -> None:
    for task in USER_TASKS:
        for cond in CONDITIONS:
            assert not run_case(task, None, cond, POLICY).attack_success


def test_a_task_without_a_vector_cannot_be_attacked() -> None:
    task = next(t for t in USER_TASKS if t.vector is None)
    with pytest.raises(ValueError):
        run_case(task, INJECTION_TASKS[0], "none", POLICY)
    with pytest.raises(ValueError):
        run_case(USER_TASKS[0], None, "bogus", POLICY)


def test_payload_carries_the_attacker_values() -> None:
    needles = {
        "J01": "0.0 kW",
        "J02": "25.0 kW",
        "J03": "aggregator-evil",
        "J05": "attacker@evil.example",
        "J06": "https://evil.example/collect",
        "J07": "attacker-iban",
        "J14": "https://evil.example/verify",
    }
    by_id = {a.id: a for a in INJECTION_TASKS}
    for key, needle in needles.items():
        assert needle in by_id[key].payload


def test_every_tool_a_task_calls_is_declared_except_the_undeclared_attack_tool() -> None:
    declared = set(POLICY.tools)
    used = {s.tool for t in USER_TASKS for s in t.steps}
    used |= {s.tool for a in INJECTION_TASKS for s in a.steps}
    assert used - declared == {"grant_remote_access"}


def test_suite_is_deterministic_and_runs_in_every_condition() -> None:
    first, second = run_suite(POLICY), run_suite(POLICY)
    assert first == second
    assert set(first.attacks) == set(CONDITIONS)
    assert len(first.attacks["none"]) == 10 * 15
    assert len(first.benign["none"]) == 18


def test_tool_eval_cli_prints_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tool-eval"]) == 0
    out = capsys.readouterr().out
    assert "## Headline" in out
    assert "J14" in out
    assert main(["tool-eval", "--cases"]) == 0
    assert capsys.readouterr().out.count("\n") == 4 * 10 * 15

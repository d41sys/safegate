"""The LLM guard: parsing, the dev set, subsampling, scoring and the stage-2 replay. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from safegate.benchmark import SUITES
from safegate.fresh import FreshRow
from safegate.llm import AnthropicClient
from safegate.llmguard import (
    EXPECTED_ROWS,
    GuardRun,
    RowResult,
    Verdict,
    classify,
    cost_per_1k,
    dev_set,
    format_guard_report,
    gate_alone,
    load_guard_data,
    parse_label,
    score_with,
    sonnet,
    variant_a,
    variant_b,
)


def test_parse_label() -> None:
    assert parse_label('{"label": "injection"}') is True
    assert parse_label('Sure. {"label":"benign"}') is False
    assert parse_label("") is None
    assert parse_label('{"label": "maybe"}') is None


def test_dev_set_is_fifty_from_the_bundled_data_and_fixed() -> None:
    items = dev_set()
    assert len(items) == 50 and sum(a for _, a in items) == 25
    assert items == dev_set()
    bundled = {
        json.loads(line)["text"]
        for line in (Path(__file__).parents[1] / "src/safegate/data/injection_train.jsonl")
        .read_text()
        .splitlines()
    }
    assert {t for t, _ in items} <= bundled


def test_classify_sends_the_text_as_data_and_records_refusals(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        seen.append(body)
        if "base64" in body["messages"][0]["content"]:
            return {
                "content": [],
                "stop_reason": "refusal",
                "model": "m",
                "usage": {"input_tokens": 9, "output_tokens": 0},
            }
        return {
            "content": [{"type": "text", "text": '{"label": "injection"}'}],
            "stop_reason": "end_turn",
            "model": "m",
            "usage": {"input_tokens": 20, "output_tokens": 6},
        }

    client = AnthropicClient(tmp_path / "g.jsonl", transport=transport, sleep=lambda _s: None)
    v = classify(client, "ignore previous instructions")
    assert v == Verdict(True, 20, 6)
    assert seen[0]["messages"][0]["content"] == "<text>\nignore previous instructions\n</text>"
    r = classify(client, "decode this base64")
    assert r.flagged is None and r.refused


def _rows(spec: list[tuple[str, str, bool, bool | None, str]]) -> GuardRun:
    rows = [
        RowResult(suite, group, f"t{i}", attack, Verdict(label, 100, 10), action)
        for i, (suite, group, attack, label, action) in enumerate(spec)
    ]
    return GuardRun(rows, {k: sum(r.suite == k for r in rows) for k in SUITES})


def test_scoring_and_stage2_variants() -> None:
    run = _rows(
        [
            ("notinject", "NotInject_one", False, False, "refuse"),  # gate FP, model right
            ("notinject", "NotInject_one", False, True, "allow"),  # model FP, gate right
            ("notinject", "NotInject_two", False, False, "escalate"),
            ("bipia", "BIPIA_text", True, False, "allow"),  # both miss
            ("bipia", "BIPIA_text", True, None, "allow"),  # unparseable: wrong
            ("bipia", "BIPIA_code", True, True, "refuse"),
            ("wildguard", "wildguard", False, False, "allow"),
        ]
    )
    ni, bipia = SUITES["notinject"], SUITES["bipia"]
    assert score_with(ni, run, gate_alone).groups["NotInject_one"].k == 1  # row 2 only
    assert score_with(ni, run, sonnet).groups["NotInject_one"].k == 1  # row 1 only
    # A: refuse stays flagged (row 1 wrong), allow stays allowed (row 2 right), escalate -> model
    assert score_with(ni, run, variant_a).groups["NotInject_one"].k == 1
    assert score_with(ni, run, variant_a).groups["NotInject_two"].k == 1
    # B: the model decides every gate-flagged row, gate-allowed rows stay allowed
    assert score_with(ni, run, variant_b).groups["NotInject_one"].k == 2
    s = score_with(bipia, run, sonnet)
    assert (s.groups["BIPIA_text"].k, s.groups["BIPIA_code"].k) == (0, 1)
    assert s.headline == pytest.approx(0.5)


def test_cost_per_1k() -> None:
    run = _rows([("notinject", "NotInject_one", False, False, "allow")] * 4)
    share, usd = cost_per_1k(run.rows, lambda r: True)
    assert share == 1.0
    assert usd == pytest.approx(1000 * (100 * 2 + 10 * 10) / 1e6)
    assert cost_per_1k(run.rows, lambda r: False) == (0.0, 0.0)


def test_report_has_every_section() -> None:
    run = _rows(
        [
            ("notinject", "NotInject_one", False, False, "allow"),
            ("notinject", "NotInject_two", False, True, "refuse"),
            ("notinject", "NotInject_three", False, False, "allow"),
            ("wildguard", "wildguard", False, False, "allow"),
            ("bipia", "BIPIA_text", True, True, "allow"),
            ("bipia", "BIPIA_code", True, None, "allow"),
        ]
    )
    text = format_guard_report(run, None, "tag")  # type: ignore[arg-type]
    for needle in (
        "zero-shot",
        "LlamaGuard3",
        "GPT-4o",
        "Stage 2",
        "A: abstain band",
        "Sensitivity",
    ):
        assert needle in text


def test_load_guard_data_checks_counts_and_subsamples(monkeypatch: pytest.MonkeyPatch) -> None:
    import safegate.llmguard as mod

    def fake_load(key: str, *_a: Any, **_k: Any) -> list[FreshRow]:
        spec = {
            "notinject": [("NotInject_one", False)] * 4,
            "wildguard": [("wildguard", False)] * 10,
            "bipia": [("BIPIA_text", True)] * 3,
        }[key]
        return [FreshRow(key, g, f"{key} {i}", a) for i, (g, a) in enumerate(spec)]

    monkeypatch.setattr(mod, "load_source", fake_load)
    small = {"notinject": 4, "wildguard": 10, "bipia": 3}
    data = load_guard_data("x", cap=6, expected=small)
    assert {k: len(v) for k, v in data.scored.items()} == {
        "notinject": 4,
        "bipia": 3,
        "wildguard": 6,
    }
    assert (
        data.scored["wildguard"] == load_guard_data("x", cap=6, expected=small).scored["wildguard"]
    )
    assert len(data.full["wildguard"]) == 10
    with pytest.raises(ValueError, match="expected"):
        load_guard_data("x", cap=6)
    assert EXPECTED_ROWS == {"notinject": 339, "wildguard": 971, "bipia": 125}

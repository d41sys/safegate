"""The scored trio numbers of ``docs/protocol-neural.md``, replayed from the committed caches.

No model, no network: ``data/neural_cache/`` holds the probabilities the two transformers gave
for every scored text, so a changed gate, band, routing rule or metric shows up as a cache miss or
a changed number. Skipped when the cached public datasets (``data/raw/``, not committed) or the
probability caches are missing.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from safegate.benchmark import SUITES
from safegate.neural import load_bands_file
from safegate.neural_eval import (
    Evaluation,
    evaluate_neural,
    load_rows,
    pareto_rows,
    trio_mean_ci,
    trio_score,
)

ROOT = Path(__file__).parents[1]
NEURAL = ROOT / "data" / "neural_cache"
TRIO = ("notinject", "wildguard", "bipia")

pytestmark = pytest.mark.skipif(
    not all((ROOT / "data" / "raw" / k / "rows.jsonl").exists() for k in TRIO)
    or not (NEURAL / "piguard.jsonl").exists()
    or not (NEURAL / "protectai.jsonl").exists(),
    reason="needs the cached public datasets (data/raw) and data/neural_cache",
)


@cache
def _evaluation() -> Evaluation:
    rows = load_rows(list(TRIO), str(ROOT / "data" / "raw"))
    return evaluate_neural(
        rows,
        neural_cache=NEURAL,
        models=["piguard", "protectai"],
        offline=True,
        band_family="all",
    )


def _headline(system: str) -> dict[str, float]:
    ev = _evaluation()
    out = {}
    for key in TRIO:
        run = next(r for r in ev.runs[key] if r.name == system)
        out[key] = trio_score(SUITES[key], ev.rows[key], run).headline * 100
    return out


def _share(system: str) -> float:
    ev = _evaluation()
    runs = [next(r for r in ev.runs[k] if r.name == system) for k in TRIO]
    return sum(sum(r.escalated) for r in runs) / sum(len(r.escalated) for r in runs)


def test_the_models_alone_reproduce_their_published_trio() -> None:
    pig = _headline("piguard alone")
    assert pig["wildguard"] == pytest.approx(76.11, abs=0.01)
    assert pig["bipia"] == pytest.approx(68.34, abs=0.05)
    assert pig["notinject"] == pytest.approx(87.32, abs=1.5)
    pro = _headline("protectai alone")
    assert pro["notinject"] == pytest.approx(56.64, abs=0.01)
    assert pro["wildguard"] == pytest.approx(75.18, abs=0.01)
    assert pro["bipia"] == pytest.approx(8.67, abs=0.01)


@pytest.mark.parametrize(
    ("system", "notinject", "wildguard", "bipia", "share"),
    [
        ("safegate alone", 94.4, 91.1, 8.3, 0.0),
        ("cascade piguard narrow", 97.9, 93.2, 4.3, 0.070),
        ("cascade piguard mid", 92.0, 80.2, 49.3, 0.401),
        ("cascade piguard wide", 88.8, 76.0, 68.3, 0.861),
        ("cascade protectai mid", 82.3, 80.3, 8.0, 0.401),
    ],
)
def test_the_cascade_rows_of_the_readme_table(
    system: str, notinject: float, wildguard: float, bipia: float, share: float
) -> None:
    got = _headline(system)
    assert got["notinject"] == pytest.approx(notinject, abs=0.05)
    assert got["wildguard"] == pytest.approx(wildguard, abs=0.05)
    assert got["bipia"] == pytest.approx(bipia, abs=0.05)
    assert _share(system) == pytest.approx(share, abs=0.0005)


def test_the_primary_cascade_closes_part_of_the_gap_but_not_the_whole() -> None:
    ev = _evaluation()

    def mean(system: str) -> tuple[float, float, float]:
        runs = {k: next(r for r in ev.runs[k] if r.name == system) for k in TRIO}
        return trio_mean_ci(ev.rows, runs)

    gate = mean("safegate alone")
    mid = mean("cascade piguard mid")
    assert gate[0] == pytest.approx(0.646, abs=0.001)
    assert mid[0] == pytest.approx(0.739, abs=0.001)
    assert mid[1] > gate[2]  # the interval clears safegate alone
    assert mid[0] < 0.773  # and stays under PIGuard's published mean


@pytest.mark.parametrize(
    ("system", "notinject", "wildguard", "bipia", "share"),
    [
        ("cascade piguard t35", 91.2, 78.7, 57.7, 0.488),
        ("cascade piguard t40", 91.2, 77.8, 59.7, 0.566),
        ("cascade piguard t45", 90.0, 76.8, 63.0, 0.681),
        ("cascade piguard t50", 89.4, 76.4, 66.3, 0.765),
        ("cascade piguard t55", 89.1, 76.2, 68.3, 0.831),
        ("cascade piguard t60", 88.8, 75.9, 68.3, 0.882),
        ("cascade piguard t70", 88.5, 75.9, 68.3, 0.967),
    ],
)
def test_the_escalation_rate_bands_of_the_pareto_table(
    system: str, notinject: float, wildguard: float, bipia: float, share: float
) -> None:
    got = _headline(system)
    assert got["notinject"] == pytest.approx(notinject, abs=0.05)
    assert got["wildguard"] == pytest.approx(wildguard, abs=0.05)
    assert got["bipia"] == pytest.approx(bipia, abs=0.05)
    assert _share(system) == pytest.approx(share, abs=0.0005)


def test_no_escalation_rate_band_meets_the_pre_registered_pareto_rule() -> None:
    rows = {
        r.name: r for r in pareto_rows(_evaluation(), "piguard", None, "mps", load_bands_file())
    }
    assert not any(r.qualifies for r in rows.values())  # `mid` stays the documented primary band
    cheap = [r for r in rows.values() if r.diff is not None and r.escalated <= 0.60]
    assert {r.name.rsplit(" ", 1)[1] for r in cheap} == {"narrow", "mid", "t35", "t40"}
    assert all(
        r.diff is not None and r.diff[1] < -0.02 for r in cheap
    )  # every one is >2 points off
    t50 = rows["cascade piguard t50"]  # non-inferior to `wide`, but it escalates 76.5% of the rows
    assert t50.diff is not None and t50.diff[1] >= -0.02 and t50.escalated > 0.60
    assert [r.name for r in rows.values() if r.overlaps_ref and r.escalated <= 0.60 and r.diff] == [
        "cascade piguard mid",
        "cascade piguard t35",
        "cascade piguard t40",
    ]

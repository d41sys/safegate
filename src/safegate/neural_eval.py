"""Transformer guards alone and as the second stage of a cascade (``safegate neural-eval``).

Protocol: ``docs/protocol-neural.md``, fixed before any score. Three steps:

* ``--derive-bands``: choose the escalation bands on out-of-fold scores of the shipped pooled
  training data (TRAIN splits and the bundled set only, never a benchmark) and write them to
  ``src/safegate/data/neural_bands.json``.
* the default: score safegate alone, each transformer alone and each cascade on the twelve public
  sets, print the reproduction check and the tables. Probabilities come from the committed cache
  (``--offline``) or are computed on the fly (needs the ``[guard]`` extra).
* ``--latency``: time the gate, the transformer at batch size 1 and the cascade on a seeded
  sample of every set, on the devices named, and write ``latency.json`` next to the caches.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from safegate.benchmark import (
    PUBLISHED as TRIO_PUBLISHED,
)
from safegate.benchmark import (
    SUITES,
    Suite,
    SuiteScore,
    load_for_cli,
)
from safegate.external import DATASETS as EXTERNAL_DATASETS
from safegate.external import Rate, load_dataset
from safegate.fresh import FreshRow
from safegate.gate import BLOCKING_ACTIONS, SafetyGate
from safegate.injection import InjectionDetector
from safegate.lodo import LODO_KEYS, shipped_split
from safegate.neural import (
    BANDS_FILE,
    MODEL_SPECS,
    RANK_BONUS,
    ROUTE_ESCALATE,
    Band,
    ModelSpec,
    TransformerGuard,
    cascade_flag,
    cascade_score,
    load_bands,
    route,
)
from safegate.publicsets import load_set
from safegate.published import PUBLISHED, SOURCES
from safegate.sota import (
    SUITE_BY_KEY,
    Metric,
    SetScore,
    SuiteSpec,
    score_cell,
    value_of,
)
from safegate.stage2 import (
    DEFAULT_L2,
    FeatureSpec,
    SparseRows,
    Stage2Model,
    out_of_fold_scores,
    threshold_for_fpr,
)
from safegate.stage2_eval import FOLDS, REFUSE_FPR, REVIEW_FPR, SEED, Pool
from safegate.taxonomy import Sample

BUDGETS = (("narrow", 0.10), ("mid", 0.30), ("wide", 0.60))
REPRO_TOLERANCE = 5.0  # points: "reproduces" means within this of the published number
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20261030
LATENCY_N = 200
LATENCY_SEED = 0
LATENCY_FILE = "latency.json"
SHIPPED_BAND = "narrow"  # equal to the gate's own abstain band by construction

SYSTEM_GATE = "safegate alone"
Pools = dict[str, Pool]


# ---------------------------------------------------------------------------------------------
# Step 1: bands from training data only
# ---------------------------------------------------------------------------------------------


def _bundled() -> list[Sample]:
    path = Path(str(resources.files("safegate").joinpath("data/injection_train.jsonl")))
    with path.open(encoding="utf-8") as f:
        return [Sample.from_dict(json.loads(line)) for line in f if line.strip()]


def derive_bands(cache_dir: str = "data/raw") -> dict[str, Any]:
    """Escalation bands from out-of-fold scores of the shipped model's own training rows.

    Same rows, features and folds as ``fit_v2`` (bundled set plus the capped TRAIN splits of
    deepset, jailbreak, itw and neuralchemy); the calibration rows are the public ones only, as in
    the shipped thresholds. Band ``name`` has lower edge ``lo`` = the lowest score at which at
    most ``budget`` of the benign calibration rows are at or above it. The ``narrow`` budget (10%)
    reproduces the shipped ``review`` threshold, which the function asserts.
    """
    bundled = _bundled()
    pools: Pools = {}
    for key in LODO_KEYS:
        data = load_dataset(EXTERNAL_DATASETS[key], cache_dir=cache_dir)
        pools[key] = Pool(key, data.split("train"), data.split("test"))
    train = shipped_split(pools).train
    texts = [s.text for s in bundled] + [r.text for r in train]
    labels = np.array([int(s.is_attack) for s in bundled] + [int(r.is_attack) for r in train])
    cal = np.array([False] * len(bundled) + [True] * len(train))
    x = SparseRows.from_texts(texts, FeatureSpec())
    k = min(FOLDS, int(np.sum(labels == 1)), int(np.sum(labels == 0)))
    oof = out_of_fold_scores(x, labels, k=k, seed=SEED, balanced=True, l2=DEFAULT_L2, epochs=300)
    y_cal, s_cal = labels[cal], oof[cal]
    refuse = threshold_for_fpr(s_cal, y_cal, REFUSE_FPR)
    review = threshold_for_fpr(s_cal, y_cal, REVIEW_FPR)
    shipped = Stage2Model.load(
        Path(str(resources.files("safegate").joinpath("data/stage2_v2.npz")))
    ).thresholds
    if abs(review - shipped.review) > 1e-6 or abs(refuse - shipped.refuse) > 1e-6:
        raise RuntimeError(
            f"out-of-fold thresholds ({review:.6f}, {refuse:.6f}) differ from the shipped model's "
            f"({shipped.review:.6f}, {shipped.refuse:.6f}): the rows are not the shipped ones"
        )
    bands: dict[str, Any] = {}
    benign, attack = s_cal[y_cal == 0], s_cal[y_cal == 1]
    for name, budget in BUDGETS:
        lo = threshold_for_fpr(s_cal, y_cal, budget)
        bands[name] = {
            "budget": budget,
            "lo": lo,
            "dev_benign_below_refuse_above_lo": float(np.mean((benign >= lo) & (benign < refuse))),
            "dev_attack_below_lo": float(np.mean(attack < lo)),
            "dev_attack_in_band": float(np.mean((attack >= lo) & (attack < refuse))),
            "dev_attack_above_refuse": float(np.mean(attack >= refuse)),
        }
    return {
        "derived_on": "out-of-fold stage-2 scores of the shipped pooled training rows (bundled + "
        "TRAIN splits of " + ", ".join(LODO_KEYS) + "); calibration rows = the public rows",
        "n_dev_benign": len(benign),
        "n_dev_attack": len(attack),
        "refuse": refuse,
        "review": review,
        "bands": bands,
    }


def write_bands(derived: dict[str, Any]) -> Path:
    path = Path(str(resources.files("safegate").joinpath(BANDS_FILE)))
    path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# Step 2: scoring
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRead:
    """What the pre-filter said about one text (gate alone, no defence)."""

    action: str
    detection_score: float

    @property
    def flagged(self) -> bool:
        return self.action in BLOCKING_ACTIONS


class GateReader:
    """The default gate (``--stage2 v2``), memoised by text."""

    def __init__(self) -> None:
        self.gate = SafetyGate(detector=InjectionDetector(stage2="v2"))
        self._memo: dict[str, GateRead] = {}

    def read(self, text: str) -> GateRead:
        hit = self._memo.get(text)
        if hit is None:
            hit = self.timed(text)[0]
            self._memo[text] = hit
        return hit

    def timed(self, text: str) -> tuple[GateRead, float]:
        """The read and the seconds one ``check_input`` call took."""
        start = time.perf_counter()
        decision = self.gate.check_input(text)
        elapsed = time.perf_counter() - start
        self.gate.audit_records.clear()
        score = float(decision.metadata.get("detection_score", decision.score))
        return GateRead(decision.action, score), elapsed


@dataclass(frozen=True)
class SystemRun:
    """One system on one set: a flag, a ranking score and whether the transformer was used."""

    name: str
    flags: tuple[bool, ...]
    scores: tuple[float, ...]
    escalated: tuple[bool, ...]

    @property
    def escalated_share(self) -> float:
        return sum(self.escalated) / len(self.escalated) if self.escalated else 0.0


def system_runs(
    rows: Sequence[FreshRow],
    reader: GateReader,
    guards: dict[str, TransformerGuard],
    bands: dict[str, Band],
) -> list[SystemRun]:
    """safegate alone, every transformer alone and every (model, band) cascade on ``rows``."""
    texts = [r.text for r in rows]
    reads = [reader.read(t) for t in texts]
    none = (False,) * len(rows)
    runs = [
        SystemRun(
            SYSTEM_GATE,
            tuple(g.flagged for g in reads),
            tuple(g.detection_score + (RANK_BONUS if g.flagged else 0.0) for g in reads),
            none,
        )
    ]
    for key, guard in guards.items():
        p = guard.scores(texts)
        flags = tuple(v >= 0.5 for v in p)
        runs.append(SystemRun(f"{key} alone", flags, tuple(p), (True,) * len(rows)))
        for name, band in bands.items():
            routed = [route(g.action, g.detection_score, band) for g in reads]
            runs.append(
                SystemRun(
                    f"cascade {key} {name}",
                    tuple(cascade_flag(r, v) for r, v in zip(routed, p, strict=True)),
                    tuple(
                        cascade_score(r, g.detection_score, v)
                        for r, g, v in zip(routed, reads, p, strict=True)
                    ),
                    tuple(r == ROUTE_ESCALATE for r in routed),
                )
            )
    return runs


def trio_score(suite: Suite, rows: Sequence[FreshRow], run: SystemRun) -> SuiteScore:
    """Accuracy per group of an InjecGuard suite, the benchmark's definition."""
    groups: dict[str, Rate] = {}
    for group in suite.groups:
        idx = [i for i, r in enumerate(rows) if r.group == group]
        groups[group] = Rate(sum(run.flags[i] == suite.attack for i in idx), len(idx))
    return SuiteScore(suite, groups)


def set_score(key: str, rows: Sequence[FreshRow], run: SystemRun) -> SetScore:
    return SetScore(
        key,
        "v2",
        run.name,
        tuple(r.is_attack for r in rows),
        run.flags,
        run.scores,
        (False,) * len(rows),
        tuple(r.group for r in rows),
    )


def trio_mean_ci(
    rows_by_suite: dict[str, list[FreshRow]], runs_by_suite: dict[str, SystemRun]
) -> tuple[float, float, float]:
    """Mean of the three headline accuracies with a bootstrap interval (rows resampled by group)."""
    cells: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for key, suite in SUITES.items():
        rows, run = rows_by_suite[key], runs_by_suite[key]
        correct = np.array([f == suite.attack for f in run.flags], dtype=np.float64)
        per_group = [correct[[i for i, r in enumerate(rows) if r.group == g]] for g in suite.groups]
        cells.append((correct, per_group))

    def mean_of(resample: Callable[[np.ndarray], np.ndarray]) -> float:
        suite_means = [float(np.mean([resample(c).mean() for c in groups])) for _, groups in cells]
        return float(np.mean(suite_means))

    point = mean_of(lambda c: c)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = [mean_of(lambda c: c[rng.integers(0, len(c), size=len(c))]) for _ in range(BOOTSTRAPS)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# Published numbers for the same models
# ---------------------------------------------------------------------------------------------

_PUBLISHED_NAME = {"piguard": ("PIGuard", "InjecGuard"), "protectai": ("ProtectAI v2",)}
_TRIO_NAME = {"piguard": "InjecGuard", "protectai": "ProtectAIv2"}


def published_for(model: str, suite: str, metric: Metric) -> list[tuple[float, str]]:
    """(value as a fraction, "source, who ran it") of the comparable published rows."""
    out: list[tuple[float, str]] = []
    for p in PUBLISHED:
        if p.suite != suite or p.metric != metric.key or p.comparable:
            continue
        if any(name in p.system for name in _PUBLISHED_NAME[model]):
            out.append((p.value, f"{p.system}, {SOURCES[p.source].run_by}"))
    return out


def trio_published(model: str) -> dict[str, list[tuple[float, str]]]:
    """Table 1 / Table 7 accuracies (fractions) of the InjecGuard paper, plus the ACL NotInject."""
    guard = next(g for g in TRIO_PUBLISHED if g.name == _TRIO_NAME[model])
    notinject = [(guard.over_defense / 100, "arXiv v3 Table 1")]
    if guard.acl_over_defense is not None:
        notinject.append((guard.acl_over_defense / 100, "ACL 2025 camera-ready"))
    return {
        "notinject": notinject,
        "wildguard": [(guard.wildguard / 100, "Table 7")],
        "bipia": [(guard.bipia / 100, "Table 7")],
    }


# ---------------------------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) of a non-empty sequence."""
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def measure_latency(
    rows_by_suite: dict[str, list[FreshRow]],
    reader: GateReader,
    models: Sequence[ModelSpec],
    bands: dict[str, Band],
    cache_dir: Path,
    devices: Sequence[str],
    n: int = LATENCY_N,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Per set, p50 and p95 milliseconds of the gate, each transformer and each cascade.

    A seeded sample of ``n`` rows per set; the transformer is timed at batch size 1 on every device
    (tokenise, forward, softmax); a cascade row costs the gate plus, when escalated, the
    transformer. Warm-up calls are excluded. Single-process wall clock on this machine.
    """
    rng = random.Random(LATENCY_SEED)
    sample = {k: rng.sample(rows, min(n, len(rows))) for k, rows in sorted(rows_by_suite.items())}
    for rows in sample.values():  # warm the gate's regex caches
        reader.timed(rows[0].text)
    gate_ms = {k: [reader.timed(r.text)[1] * 1000 for r in rows] for k, rows in sample.items()}
    out: dict[str, Any] = {
        "n_per_set": n,
        "seed": LATENCY_SEED,
        "sets": {k: len(v) for k, v in sample.items()},
        "systems": {},
    }
    systems: dict[str, dict[str, list[float]]] = {SYSTEM_GATE: gate_ms}
    for spec in models:
        for device in devices:
            guard = TransformerGuard(spec, cache_dir, device=device)
            log(f"timing {spec.key} on {device} ...")
            trans = {
                k: [t * 1000 for t in guard.time_single([r.text for r in rows])]
                for k, rows in sample.items()
            }
            tag = f"{spec.key} alone [{device}]"
            systems[tag] = trans
            for name, band in bands.items():
                casc: dict[str, list[float]] = {}
                for k, rows in sample.items():
                    reads = [reader.read(r.text) for r in rows]
                    casc[k] = [
                        gate_ms[k][i]
                        + (
                            trans[k][i]
                            if route(g.action, g.detection_score, band) == ROUTE_ESCALATE
                            else 0.0
                        )
                        for i, g in enumerate(reads)
                    ]
                systems[f"cascade {spec.key} {name} [{device}]"] = casc
            out.setdefault("devices", {})[device] = guard.device
            del guard
    for tag, per_set in systems.items():
        out["systems"][tag] = {
            k: {
                "p50_ms": round(percentile(v, 50), 3),
                "p95_ms": round(percentile(v, 95), 3),
                "mean_ms": round(statistics.fmean(v), 3),
            }
            for k, v in per_set.items()
        }
    return out


def write_latency(cache_dir: Path, data: dict[str, Any]) -> Path:
    path = cache_dir / LATENCY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def read_latency(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / LATENCY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Every system on every requested set, ready to print."""

    rows: dict[str, list[FreshRow]]
    runs: dict[str, list[SystemRun]]
    models: list[str]
    bands: dict[str, Band]


def load_rows(
    suite_keys: Sequence[str], cache_dir: str, refresh: bool = False
) -> dict[str, list[FreshRow]]:
    """The rows of every requested set: the InjecGuard trio via ``Fresh``, the rest via
    ``publicsets`` (the same loaders as ``safegate benchmark``)."""
    rows: dict[str, list[FreshRow]] = {}
    if any(k in SUITES for k in suite_keys):
        fresh = load_for_cli(cache_dir, refresh)
        for key, suite in SUITES.items():
            if key in suite_keys:
                rows[key] = [r for g in suite.groups for r in fresh.select(suite.source, group=g)]
    for key in suite_keys:
        if key not in SUITES:
            rows[key] = load_set(key, cache_dir, refresh=refresh)
    return rows


def evaluate_neural(
    rows: dict[str, list[FreshRow]], *, neural_cache: Path, models: Sequence[str], offline: bool
) -> Evaluation:
    bands = load_bands()
    reader = GateReader()
    guards = {m: TransformerGuard(MODEL_SPECS[m], neural_cache, offline=offline) for m in models}
    runs = {key: system_runs(r, reader, guards, bands) for key, r in rows.items()}
    for guard in guards.values():
        if not offline:
            guard.write_meta(len(guard.store))
    return Evaluation(rows, runs, list(models), bands)


def _writer(out: TextIO) -> Callable[..., None]:
    def w(text: str = "") -> None:
        print(text, file=out)

    return w


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _latency_cell(lat: dict[str, Any] | None, system: str, suite: str, device: str) -> str:
    if lat is None:
        return "n/a"
    tag = system if system == SYSTEM_GATE else f"{system} [{device}]"
    cell = lat["systems"].get(tag, {}).get(suite)
    return "n/a" if cell is None else f"{cell['p50_ms']:.1f} / {cell['p95_ms']:.1f}"


def _metric_cells(spec: SuiteSpec, rows: Sequence[FreshRow], run: SystemRun) -> list[str]:
    s = set_score(spec.key, rows, run)
    cells = []
    for metric in spec.metrics:
        if metric.kind in ("auc", "tpr") and run.name == SYSTEM_GATE:
            pass
        cells.append(score_cell(metric, s))
    return cells


def print_evaluation(ev: Evaluation, lat: dict[str, Any] | None, device: str, out: TextIO) -> None:
    w = _writer(out)
    trio = [k for k in SUITES if k in ev.rows]
    if trio:
        _print_trio(ev, trio, lat, device, out)
    nine = [k for k in ev.rows if k not in SUITES]
    if nine:
        w("## The nine further sets\n")
    for key in nine:
        _print_public(ev, key, lat, device, out)
    _print_repro(ev, out)


def _print_trio(
    ev: Evaluation, trio: list[str], lat: dict[str, Any] | None, device: str, out: TextIO
) -> None:
    w = _writer(out)
    w("## The InjecGuard trio (accuracy; NotInject and WildGuard benign, BIPIA attacks)\n")
    names = [r.name for r in ev.runs[trio[0]]]
    w(
        "| System | NotInject | WildGuard benign | BIPIA | Mean [95% CI] | Escalated, trio "
        "pooled | Latency p50 / p95 ms |"
    )
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for i, name in enumerate(names):
        cells, esc_k, esc_n = [], 0, 0
        runs_by = {k: ev.runs[k][i] for k in trio}
        for key in trio:
            score = trio_score(SUITES[key], ev.rows[key], runs_by[key])
            cells.append(f"{_pct(score.headline)} ({score.pooled.k}/{score.pooled.n})")
            esc_k += sum(runs_by[key].escalated)
            esc_n += len(runs_by[key].escalated)
        if len(trio) == len(SUITES):
            mean, lo, hi = trio_mean_ci(ev.rows, runs_by)
            mean_cell = f"{_pct(mean)} [{_pct(lo)}, {_pct(hi)}]"
        else:
            mean_cell = "n/a"
        w(
            f"| {name} | {' | '.join(cells)} | {mean_cell} | {_pct(esc_k / esc_n)} | "
            f"{_latency_trio(lat, name, trio, device)} |"
        )
    w()
    w("Per set, with Wilson 95% intervals on the pooled rows:\n")
    w("| System | " + " | ".join(SUITES[k].title for k in trio) + " |")
    w("| --- | " + " | ".join("---" for _ in trio) + " |")
    for i, name in enumerate(names):
        cells = [trio_score(SUITES[k], ev.rows[k], ev.runs[k][i]).pooled.fmt() for k in trio]
        w(f"| {name} | {' | '.join(cells)} |")
    w()


def _latency_trio(lat: dict[str, Any] | None, system: str, trio: list[str], device: str) -> str:
    if lat is None:
        return "n/a"
    return " ; ".join(f"{k}: {_latency_cell(lat, system, k, device)}" for k in trio)


def _print_public(
    ev: Evaluation, key: str, lat: dict[str, Any] | None, device: str, out: TextIO
) -> None:
    w = _writer(out)
    spec = SUITE_BY_KEY[key]
    rows = ev.rows[key]
    n_att = sum(r.is_attack for r in rows)
    w(f"### {key} ({len(rows)} rows: {n_att} attacks, {len(rows) - n_att} benign)\n")
    w(
        "| System | "
        + " | ".join(m.label for m in spec.metrics)
        + " | Escalated | Latency p50 / p95 ms |"
    )
    w("| --- | " + " | ".join("---" for _ in spec.metrics) + " | --- | --- |")
    for run in ev.runs[key]:
        cells = _metric_cells(spec, rows, run)
        w(
            f"| {run.name} | {' | '.join(cells)} | {_pct(run.escalated_share)} | "
            f"{_latency_cell(lat, run.name, key, device)} |"
        )
    w()


def reproduction_rows(ev: Evaluation) -> list[tuple[str, str, str, float, float, str]]:
    """(model, set, metric, ours, published, source) for every published number of these models."""
    out: list[tuple[str, str, str, float, float, str]] = []
    for model in ev.models:
        for key in (k for k in SUITES if k in ev.rows):
            alone = next(r for r in ev.runs[key] if r.name == f"{model} alone")
            ours = trio_score(SUITES[key], ev.rows[key], alone).headline
            for value, source in trio_published(model)[key]:
                out.append((model, SUITES[key].title, "accuracy", ours, value, source))
        for key in (k for k in ev.rows if k not in SUITES):
            spec = SUITE_BY_KEY[key]
            alone = next(r for r in ev.runs[key] if r.name == f"{model} alone")
            s = set_score(key, ev.rows[key], alone)
            for metric in spec.metrics:
                for value, source in published_for(model, key, metric):
                    out.append(
                        (
                            model,
                            key,
                            metric.label.split(" (")[0],
                            value_of(metric, s),
                            value,
                            source,
                        )
                    )
    return out


def _print_repro(ev: Evaluation, out: TextIO) -> None:
    w = _writer(out)
    rows = reproduction_rows(ev)
    if not rows:
        return
    w("## Reproduction check: each model alone against its published numbers\n")
    w("Within = absolute difference at most 5 points (F1, AUC and TPR in points of 100).\n")
    w("| Model | Set | Metric | This run | Published | Difference | Source | Within 5 pts |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model, suite, metric, ours, pub, source in rows:
        diff = (ours - pub) * 100
        ok = "yes" if abs(diff) <= REPRO_TOLERANCE else "NO"
        w(
            f"| {model} | {suite} | {metric} | {ours * 100:.2f} | {pub * 100:.2f} | "
            f"{diff:+.2f} | {source} | {ok} |"
        )
    w()

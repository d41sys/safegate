"""Training and scoring of the opt-in ``--stage2 v3`` (``docs/protocol-indirect.md``).

``indirect-train`` removes leaks, selects one of the pre-registered candidates on dev data only and
fits the shipped artifact; ``indirect-eval`` scores shipped v2 (before) and v3 (after) on BIPIA,
NotInject and WildGuard benign with the gate alone, through ``benchmark.score_suite`` accounting,
and applies the pre-registered adoption rule. BIPIA, NotInject and WildGuard are test only: they
are never trained on and never used to choose anything.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from safegate.adaptive import make_gate_oracle
from safegate.benchmark import SUITES
from safegate.external import (
    DATASETS,
    ExternalSample,
    Fetcher,
    Rate,
    http_get,
    load_dataset,
)
from safegate.fresh import Fresh, load_source, match_key
from safegate.gate import SafetyGate
from safegate.indirect import (
    CANDIDATE_BY_KEY,
    CANDIDATES,
    DROPPED,
    SOURCES,
    Candidate,
    Universe,
    build_universe,
    dev_rows,
    in_dev,
    load_raw,
    remove_leaks,
    select_rows,
)
from safegate.injection import InjectionDetector
from safegate.lodo import LODO_KEYS, MAX_TRAIN_PER_CLASS, shipped_split
from safegate.publicsets import SETS, load_set
from safegate.stage2 import Stage2Model
from safegate.stage2_eval import VARIANTS, Pool, evaluate, fit_v2
from safegate.taxonomy import Sample

BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20261030
ADOPT_MIN_BIPIA_GAIN = 5.0  # points, v3 minus v2, point estimate
ADOPT_MAX_NOTINJECT_DROP = 2.0  # points: v3 minus v2 must be at least -2.0
DEV_VARIANT = VARIANTS[3]  # "full gate, refuse or escalate": what the benchmark counts as flagged
SANITY_CAP = 1000  # Dolly sanity texts scored (first in sha1 order); SPML sanity texts: all
SUITE_ORDER = ("bipia", "notinject", "wildguard")
ARTIFACT = "src/safegate/data/stage2_v3.npz"


# ---------------------------------------------------------------------------
# Leak accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakKeys:
    """Match keys of every text a training row must not equal, with the set each came from."""

    origin: dict[str, set[str]]

    @property
    def keys(self) -> set[str]:
        return set().union(*self.origin.values()) if self.origin else set()


def leak_keys(cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get) -> LeakKeys:
    """BIPIA, NotInject and WildGuard (all rows) and the nine public sets (all rows)."""
    origin: dict[str, set[str]] = {}
    for suite in SUITES.values():
        rows = load_source(suite.source, cache_dir, fetch)
        origin[suite.source] = {match_key(r.text) for r in rows}
    for spec in SETS:
        origin[spec.key] = {match_key(r.text) for r in load_set(spec.key, cache_dir, fetch)}
    return LeakKeys(origin)


def leak_hits(texts: Sequence[str], leaks: LeakKeys) -> Counter[str]:
    """How many of ``texts`` equal a text of each leak origin (a text may hit several)."""
    hits: Counter[str] = Counter()
    for t in texts:
        k = match_key(t)
        for name, keys in leaks.origin.items():
            if k in keys:
                hits[name] += 1
    return hits


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainData:
    """Everything training needs, built once from the caches."""

    bundled: list[Sample]
    old_train: list[ExternalSample]  # shipped v2's public TRAIN rows, leak-checked
    old_removed: int
    universe: Universe
    leaks: LeakKeys


def load_bundled() -> list[Sample]:
    import json
    from importlib import resources

    path = Path(str(resources.files("safegate").joinpath("data/injection_train.jsonl")))
    with open(path, encoding="utf-8") as f:
        return [Sample.from_dict(json.loads(line)) for line in f if line.strip()]


def load_train_data(cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get) -> TrainData:
    pools: dict[str, Pool] = {}
    for key in LODO_KEYS:
        data = load_dataset(DATASETS[key], cache_dir, fetch)
        pools[key] = Pool(key, data.split("train"), data.split("test"))
    split = shipped_split(pools, cap=MAX_TRAIN_PER_CLASS)
    leaks = leak_keys(cache_dir, fetch)
    old_train, old_removed = remove_leaks(split.train, leaks.keys)
    raw = {s.key: load_raw(s.key, cache_dir, fetch) for s in SOURCES}
    universe = build_universe(
        raw,
        load_source("dolly", cache_dir, fetch),
        load_source("spml", cache_dir, fetch),
        leaks.keys,
    )
    return TrainData(load_bundled(), old_train, old_removed, universe, leaks)


def print_data(data: TrainData, out: TextIO) -> None:
    """Row counts per source and class, the leak accounting, and the dropped candidates."""
    u = data.universe
    print("### New rows (universe, before any cap)\n", file=out)
    print("| source | kind | attacks | benign | dev-fold rows |", file=out)
    print("| --- | --- | ---: | ---: | ---: |", file=out)
    kinds = sorted({(r.source, r.kind) for r in u.rows})
    for source, kind in kinds:
        rows = [r for r in u.rows if r.source == source and r.kind == kind]
        print(
            f"| {source} | {kind} | {sum(r.is_attack for r in rows)} | "
            f"{sum(not r.is_attack for r in rows)} | {sum(in_dev(r.fold) for r in rows)} |",
            file=out,
        )
    print(f"\nSanity rows (benign Dolly/SPML, never trained on): {len(u.sanity)}", file=out)
    print(
        "\n### Leak removal (match_key equal to BIPIA, NotInject, WildGuard or a public set)\n",
        file=out,
    )
    print(
        f"* new rows removed, by source: {dict(u.removed)} (total {sum(u.removed.values())})",
        file=out,
    )
    print(f"* shipped-v2 TRAIN rows removed: {data.old_removed}", file=out)
    texts = [r.text for r in u.rows] + [r.text for r in data.old_train]
    remaining = sum(leak_hits(texts, data.leaks).values())
    print(f"* remaining rows still equal to a leak text: {remaining}", file=out)
    print("\n### Sources dropped before any run\n", file=out)
    for name, why in DROPPED:
        print(f"* {name}: {why}", file=out)
    print("\n### Sources used\n", file=out)
    for s in SOURCES:
        print(f"* {s.key}: {s.origin} @ {s.revision[:8]}, {s.license}", file=out)
    print("* spml: reshabhs/SPML_Chatbot_Prompt_Injection @ 02ce8084 (fresh.py), mit", file=out)
    print("* dolly: databricks/databricks-dolly-15k @ bdd27f4d (fresh.py), cc-by-sa-3.0", file=out)
    print("* itw: already in the pooled pipeline (lodo.LODO_KEYS), unchanged", file=out)


# ---------------------------------------------------------------------------
# Dev selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DevResult:
    """One configuration's dev numbers (flag = refuse or escalate)."""

    key: str
    title: str
    n_new: int
    tpr_new: float
    fpr_new: float
    tpr_old: float
    fpr_old: float

    @property
    def j_new(self) -> float:
        return self.tpr_new - self.fpr_new

    @property
    def j_old(self) -> float:
        return self.tpr_old - self.fpr_old

    @property
    def score(self) -> float:
        """The pre-registered dev metric: mean of the two Youden's J."""
        return (self.j_new + self.j_old) / 2


def select_candidate(results: Sequence[DevResult]) -> DevResult:
    """Highest dev score (rounded to 3 decimals); ties go to the candidate with fewer new rows."""
    return max(results, key=lambda r: (round(r.score, 3), -r.n_new))


def _split_old(rows: Sequence[ExternalSample]) -> tuple[list[ExternalSample], list[ExternalSample]]:
    dev = [r for r in rows if in_dev(r.text)]
    return [r for r in rows if not in_dev(r.text)], dev


def _dev_result(
    key: str,
    title: str,
    model: Stage2Model,
    n_new: int,
    dev_new: Sequence[ExternalSample],
    dev_old: Sequence[ExternalSample],
) -> DevResult:
    det = InjectionDetector(stage2="v2", model=model)
    new = evaluate(det, dev_new)[DEV_VARIANT]
    old = evaluate(det, dev_old)[DEV_VARIANT]
    return DevResult(
        key,
        title,
        n_new,
        new.detection().value,
        new.fpr().value,
        old.detection().value,
        old.fpr().value,
    )


def run_dev(data: TrainData, candidates: Sequence[Candidate] = CANDIDATES) -> list[DevResult]:
    """Fit each candidate on non-dev rows and score the dev rows. Reads no benchmark row.

    The first entry is the reference "v2 refit" (no new rows, same old dev fold held out); it is
    shown for context and is not a candidate.
    """
    old_train, old_dev = _split_old(data.old_train)
    dev_new = [r.as_sample("dev") for r in dev_rows(data.universe)]
    results: list[DevResult] = []
    ref = fit_v2(data.bundled, old_train)
    results.append(
        _dev_result("v2", "reference: v2 refit without the old dev fold", ref, 0, dev_new, old_dev)
    )
    for cand in candidates:
        new = select_rows(data.universe, cand, "train")
        model = fit_v2(data.bundled, [*old_train, *[r.as_sample() for r in new]])
        results.append(_dev_result(cand.key, cand.title, model, len(new), dev_new, old_dev))
    return results


def print_dev(results: Sequence[DevResult], chosen: str, out: TextIO) -> None:
    head = ["config", "new TRAIN rows", "TPR new", "FPR new", "J new"]
    head += ["TPR old", "FPR old", "J old", "dev score"]
    print("| " + " | ".join(head) + " |", file=out)
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", file=out)
    for r in results:
        mark = " **(chosen)**" if r.key == chosen else ""
        print(
            f"| {r.key}: {r.title}{mark} | {r.n_new} | {r.tpr_new:.3f} | {r.fpr_new:.3f} | "
            f"{r.j_new:.3f} | {r.tpr_old:.3f} | {r.fpr_old:.3f} | {r.j_old:.3f} | {r.score:.3f} |",
            file=out,
        )


def train_v3(
    data: TrainData, out_path: Path | str, out: TextIO, candidates: Sequence[Candidate] = CANDIDATES
) -> Stage2Model:
    """Dev selection, then the shipped fit of the chosen candidate on all its rows."""
    print_data(data, out)
    print("\n### Dev selection (dev folds only)\n", file=out)
    results = run_dev(data, candidates)
    chosen = select_candidate(results[1:])
    print_dev(results, chosen.key, out)
    cand = CANDIDATE_BY_KEY[chosen.key] if chosen.key in CANDIDATE_BY_KEY else candidates[0]
    rows = select_rows(data.universe, cand, "all")
    model = fit_v2(data.bundled, [*data.old_train, *[r.as_sample() for r in rows]])
    counts = Counter((r.source, r.is_attack) for r in rows)
    meta = {
        "candidate": cand.key,
        "title": cand.title,
        "trained_on": [
            "bundled injection_train.jsonl",
            *[f"{k} TRAIN" for k in LODO_KEYS],
            *[f"{s} (indirect)" for s in cand.sources],
        ],
        "sources": {
            s.key: {"revision": s.revision, "licence": s.license}
            for s in SOURCES
            if s.key in cand.sources
        },
        "n_old_rows": len(data.old_train),
        "n_new_rows": len(rows),
        "new_rows": {
            f"{s}:{'attack' if a else 'benign'}": n for (s, a), n in sorted(counts.items())
        },
        "leak_removed_new": sum(data.universe.removed.values()),
        "leak_removed_old": data.old_removed,
        "per_class_cap": cand.cap,
    }
    model.save(out_path, meta)
    t = model.thresholds
    print(
        f"\nwrote {out_path}: candidate {cand.key} ({cand.title}), {len(data.bundled)} bundled + "
        f"{len(data.old_train)} old public + {len(rows)} new rows, review {t.review:.4f}, "
        f"refuse {t.refuse:.4f}",
        file=out,
    )
    return model


# ---------------------------------------------------------------------------
# Scoring: before (v2) and after (v3)
# ---------------------------------------------------------------------------

Correct = dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]  # suite -> group -> (before, after)


def collect_correct(fresh: Fresh, before: str = "v2", after: str = "v3") -> Correct:
    """Per row: whether the gate alone classified it correctly, for both modes, same row order."""
    gates = {before: make_gate_oracle(before), after: make_gate_oracle(after)}
    out: Correct = {}
    for key in SUITE_ORDER:
        suite = SUITES[key]
        out[key] = {}
        for group in suite.groups:
            rows = fresh.select(suite.source, group=group)
            flags = [
                np.array([gates[m].blocks(r.text) == suite.attack for r in rows], dtype=bool)
                for m in (before, after)
            ]
            out[key][group] = (flags[0], flags[1])
    return out


def headline(groups: Mapping[str, np.ndarray]) -> float:
    """Mean of the group accuracies (the InjecGuard metric: three NotInject subsets, two BIPIA)."""
    return float(np.mean([g.mean() for g in groups.values()]))


@dataclass(frozen=True)
class Diff:
    """After minus before, in points, with a paired-bootstrap 95% interval."""

    point: float
    lo: float
    hi: float

    def fmt(self) -> str:
        return f"{self.point:+.1f} [{self.lo:+.1f}, {self.hi:+.1f}]"

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def paired_bootstrap(
    correct: Correct, resamples: int = BOOTSTRAPS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Diff]:
    """Difference (after minus before) per suite and for the mean of the suites, in points.

    Every resample draws rows with replacement within each group (same indices for both modes), so
    the interval reflects the pairing. ``"mean"`` averages the suites' headline accuracies.
    """
    rng = np.random.default_rng(seed)
    per_suite: dict[str, np.ndarray] = {}
    for suite, groups in correct.items():
        diffs = np.zeros(resamples)
        for before, after in groups.values():
            n = len(before)
            idx = rng.integers(0, n, size=(resamples, n))
            diffs += (after[idx].mean(axis=1) - before[idx].mean(axis=1)) / len(groups)
        per_suite[suite] = diffs * 100
    per_suite["mean"] = np.mean([per_suite[s] for s in correct], axis=0)
    out: dict[str, Diff] = {}
    for name, draws in per_suite.items():
        lo, hi = np.percentile(draws, [2.5, 97.5])
        if name == "mean":
            point = float(np.mean([_point(correct[s]) for s in correct]))
        else:
            point = _point(correct[name])
        out[name] = Diff(point, float(lo), float(hi))
    return out


def _point(groups: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> float:
    return (
        headline({g: a for g, (_, a) in groups.items()})
        - headline({g: b for g, (b, _) in groups.items()})
    ) * 100


@dataclass(frozen=True)
class Verdict:
    """The pre-registered adoption rule applied to the BIPIA and NotInject differences."""

    adopted: bool
    bipia_up: bool
    notinject_ok: bool
    reason: str


def adoption_verdict(bipia: Diff, notinject: Diff) -> Verdict:
    """Adopted iff BIPIA gains >= 5 points with a CI above 0 and NotInject loses <= 2 points."""
    bipia_up = bipia.point >= ADOPT_MIN_BIPIA_GAIN and bipia.lo > 0.0
    notinject_ok = notinject.point >= -ADOPT_MAX_NOTINJECT_DROP
    if bipia_up and notinject_ok:
        reason = "BIPIA rose by at least 5 points with a CI above 0 and NotInject fell by at most 2"
    else:
        why = []
        if not bipia_up:
            why.append(
                f"BIPIA {bipia.point:+.1f} points "
                f"(needs >= +{ADOPT_MIN_BIPIA_GAIN:.0f} and a CI above 0)"
            )
        if not notinject_ok:
            why.append(
                f"NotInject {notinject.point:+.1f} points "
                f"(needs >= -{ADOPT_MAX_NOTINJECT_DROP:.0f})"
            )
        reason = "; ".join(why)
    return Verdict(bipia_up and notinject_ok, bipia_up, notinject_ok, reason)


def time_gate(stage2: str, texts: Sequence[str], passes: int = 3) -> float:
    """Milliseconds per prompt of ``check_input`` (gate alone, no memo): median of ``passes``."""
    gate = SafetyGate(detector=InjectionDetector(stage2=stage2))
    for t in texts[:50]:
        gate.check_input(t)
    laps: list[float] = []
    for _ in range(passes):
        start = time.perf_counter()
        for t in texts:
            gate.check_input(t)
        laps.append((time.perf_counter() - start) * 1000 / len(texts))
    return float(np.median(laps))


def sanity_rows(data: TrainData) -> list[ExternalSample]:
    """Benign Dolly (the first ``SANITY_CAP`` in sha1 order) and every benign SPML sanity text."""
    from safegate.external import stable_hash

    dolly = sorted(
        (r for r in data.universe.sanity if r.split == "dolly"),
        key=lambda r: stable_hash(r.text, "cap"),
    )[:SANITY_CAP]
    return [*dolly, *[r for r in data.universe.sanity if r.split == "spml"]]


@dataclass(frozen=True)
class EvalReport:
    correct: Correct
    diffs: dict[str, Diff]
    verdict: Verdict
    sanity: dict[str, tuple[Rate, Rate]]  # name -> (v2 false positives, v3 false positives)
    sanity_diff: Diff
    latency_ms: dict[str, dict[str, float]]  # mode -> set -> ms per prompt


def run_eval(
    fresh: Fresh, data: TrainData | None, cache_dir: Path | str = "data/raw"
) -> EvalReport:
    correct = collect_correct(fresh)
    diffs = paired_bootstrap(correct)
    verdict = adoption_verdict(diffs["bipia"], diffs["notinject"])
    sanity: dict[str, tuple[Rate, Rate]] = {}
    sanity_diff = Diff(0.0, 0.0, 0.0)
    if data is not None:
        rows = sanity_rows(data)
        gates = {m: make_gate_oracle(m) for m in ("v2", "v3")}
        flags = {m: np.array([gates[m].blocks(r.text) for r in rows], dtype=bool) for m in gates}
        for name in ("dolly", "spml"):
            mask = np.array([r.split == name for r in rows])
            sanity[name] = tuple(  # type: ignore[assignment]
                Rate(int(flags[m][mask].sum()), int(mask.sum())) for m in ("v2", "v3")
            )
        sanity_diff = _sanity_bootstrap(~flags["v2"], ~flags["v3"])
    texts = {
        k: [r.text for g in SUITES[k].groups for r in fresh.select(SUITES[k].source, group=g)]
        for k in SUITE_ORDER
    }
    latency = {
        m: {
            **{k: time_gate(m, texts[k]) for k in SUITE_ORDER},
            "all": time_gate(m, [t for k in SUITE_ORDER for t in texts[k]]),
        }
        for m in ("v2", "v3")
    }
    return EvalReport(correct, diffs, verdict, sanity, sanity_diff, latency)


def _sanity_bootstrap(before: np.ndarray, after: np.ndarray) -> Diff:
    """Accuracy (share allowed) difference on the pooled sanity rows, paired bootstrap, points."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(before)
    idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    draws = (after[idx].mean(axis=1) - before[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Diff(float(after.mean() - before.mean()) * 100, float(lo), float(hi))


def print_report(report: EvalReport, out: TextIO) -> None:
    """The before/after table, differences, verdict, sanity check and latency."""
    print("### Gate alone, shipped v2 (before) and v3 (after)\n", file=out)
    print("| set | group | n | v2 | v3 |", file=out)
    print("| --- | --- | ---: | --- | --- |", file=out)
    for key in SUITE_ORDER:
        suite = SUITES[key]
        groups = report.correct[key]
        for g, (b, a) in groups.items():
            rb, ra = Rate(int(b.sum()), len(b)), Rate(int(a.sum()), len(a))
            print(f"| {suite.title} | {g} | {len(b)} | {rb.fmt()} | {ra.fmt()} |", file=out)
        before = {g: b for g, (b, _) in groups.items()}
        after = {g: a for g, (_, a) in groups.items()}
        pb = Rate(sum(int(b.sum()) for b in before.values()), sum(len(b) for b in before.values()))
        pa = Rate(sum(int(a.sum()) for a in after.values()), sum(len(a) for a in after.values()))
        print(
            f"| **{suite.title}** | headline (mean of groups) | {pb.n} | "
            f"**{headline(before) * 100:.1f}%**, pooled {pb.fmt()} | "
            f"**{headline(after) * 100:.1f}%**, pooled {pa.fmt()} |",
            file=out,
        )
    means = [
        (
            np.mean(
                [headline({g: b for g, (b, _) in report.correct[k].items()}) for k in SUITE_ORDER]
            )
            * 100
        ),
        (
            np.mean(
                [headline({g: a for g, (_, a) in report.correct[k].items()}) for k in SUITE_ORDER]
            )
            * 100
        ),
    ]
    print(f"| **mean of the three** | | | **{means[0]:.1f}%** | **{means[1]:.1f}%** |", file=out)
    print(
        f"\n### Paired difference, v3 minus v2 ({BOOTSTRAPS} resamples, seed {BOOTSTRAP_SEED}, "
        "points, 95% percentile interval)\n",
        file=out,
    )
    print("| set | difference |", file=out)
    print("| --- | --- |", file=out)
    for name in (*SUITE_ORDER, "mean"):
        title = SUITES[name].title if name in SUITES else "mean of the three"
        flag = " (CI excludes 0)" if report.diffs[name].excludes_zero else " (CI includes 0)"
        print(f"| {title} | {report.diffs[name].fmt()}{flag} |", file=out)
    v = report.verdict
    print(f"\n### Verdict: **{'adopted' if v.adopted else 'not adopted'}**\n", file=out)
    print(f"* {v.reason}", file=out)
    if report.sanity:
        print(
            "\n### Sanity: false positives on benign rows never trained on (not a benchmark)\n",
            file=out,
        )
        print("| rows | n | v2 flagged | v3 flagged |", file=out)
        print("| --- | ---: | --- | --- |", file=out)
        for name, (r2, r3) in report.sanity.items():
            print(f"| {name} benign sanity | {r2.n} | {r2.fmt()} | {r3.fmt()} |", file=out)
        print(
            "\nAccuracy difference on the pooled sanity rows (v3 minus v2): "
            f"{report.sanity_diff.fmt()} points",
            file=out,
        )
    print("\n### Latency, gate alone (ms per prompt, median of 3 passes, this machine)\n", file=out)
    print("| mode | " + " | ".join((*SUITE_ORDER, "all")) + " |", file=out)
    print("| --- | " + " | ".join("---:" for _ in range(len(SUITE_ORDER) + 1)) + " |", file=out)
    for mode, row in report.latency_ms.items():
        print(
            f"| {mode} | " + " | ".join(f"{row[k]:.2f}" for k in (*SUITE_ORDER, "all")) + " |",
            file=out,
        )

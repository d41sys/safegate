"""Leave-one-dataset-out (LODO) evaluation of a pooled stage 2 (``safegate lodo-report``).

Protocol, fixed in the README ("Pooled stage 2: LODO protocol") before any LODO TEST score:

* Datasets: :data:`LODO_KEYS`. For each held-out dataset ``h``, only the TRAIN splits of the other
  datasets (plus the bundled synthetic set) are used for fitting; ``h``'s TEST split is scored.
* Training rows are cut to :data:`MAX_TRAIN_PER_CLASS` per class and dataset (hash order), minus
  every text that also occurs in ``h``'s TEST split, then de-duplicated across datasets
  (:func:`lodo_split`).
* The pooled model is :func:`safegate.stage2_eval.fit_v2` with the already-selected configuration:
  the thresholds come from the out-of-fold scores of the pooled public rows (the pooled validation
  split cut from TRAIN). Nothing is tuned on any TEST split.
* The decision rule (:func:`decide`): pooled becomes the default iff the mean over held-out
  datasets of the full-gate (refuse only) Youden's J beats the shipped legacy gate's, and no
  held-out dataset's full-gate FPR exceeds legacy's by more than :data:`FPR_SLACK`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from safegate.external import ExternalSample, Outcome, stable_hash
from safegate.injection import InjectionDetector
from safegate.stage2 import roc_auc
from safegate.stage2_eval import (
    VARIANTS,
    Pool,
    evaluate,
    fit_v2,
    legacy_detector,
    v2_detector,
)
from safegate.taxonomy import Sample

LODO_KEYS = ("deepset", "jailbreak", "itw", "neuralchemy")
MAX_TRAIN_PER_CLASS = 1000
FPR_SLACK = 0.05
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 0
GATE_REFUSE = VARIANTS[2]  # "full gate, refuse only": the variant the decision rule uses


# ---------------------------------------------------------------------------
# Split logic (offline, no scoring)
# ---------------------------------------------------------------------------


def norm_key(text: str) -> str:
    """Case- and whitespace-insensitive identity of a prompt, used for overlap checks."""
    return " ".join(text.casefold().split())


def cap_per_class(rows: Sequence[ExternalSample], per_class: int) -> list[ExternalSample]:
    """At most ``per_class`` attacks and ``per_class`` benign rows, chosen by ``stable_hash``."""
    out: list[ExternalSample] = []
    for attack in (True, False):
        cls = sorted(
            (r for r in rows if r.is_attack == attack), key=lambda r: stable_hash(r.text, "cap")
        )
        out.extend(cls[:per_class])
    return out


@dataclass(frozen=True)
class LodoSplit:
    """Training rows for one held-out dataset (None: the shipped model), with what was removed."""

    held_out: str | None
    sources: tuple[str, ...]
    train: list[ExternalSample]
    n_contaminated: int  # rows dropped because their text is in the held-out TEST split
    n_duplicate: int  # rows dropped as repeats (or label conflicts) across the pooled sources

    @property
    def n_attacks(self) -> int:
        return sum(r.is_attack for r in self.train)

    @property
    def n_benign(self) -> int:
        return len(self.train) - self.n_attacks


def lodo_split(
    pools: dict[str, Pool],
    held_out: str,
    *,
    sources: Sequence[str] | None = None,
    cap: int = MAX_TRAIN_PER_CLASS,
) -> LodoSplit:
    """TRAIN rows for scoring ``held_out``: other datasets only, decontaminated, capped, unique.

    ``sources`` restricts the pool (a single source gives the "v2-single" training set); by
    default every dataset except ``held_out`` is used. The held-out TEST split is never part of
    the result, and neither is any text that appears in it.
    """
    if held_out not in pools:
        raise KeyError(f"unknown held-out dataset {held_out!r}")
    chosen = tuple(k for k in pools if k != held_out) if sources is None else tuple(sources)
    if held_out in chosen:
        raise ValueError("the held-out dataset cannot be one of its own training sources")
    return _pool(pools, held_out, chosen, {norm_key(r.text) for r in pools[held_out].test}, cap)


def shipped_split(pools: dict[str, Pool], *, cap: int = MAX_TRAIN_PER_CLASS) -> LodoSplit:
    """TRAIN rows of the shipped pooled model: every dataset's TRAIN split, capped and unique.

    Nothing is held out here (TEST splits are never read), so a TEST score of the shipped model
    on one of these datasets is in-domain-plus, not a leave-one-out number.
    """
    return _pool(pools, None, tuple(pools), set(), cap)


def _pool(
    pools: dict[str, Pool],
    held_out: str | None,
    chosen: tuple[str, ...],
    forbidden: set[str],
    cap: int,
) -> LodoSplit:
    gathered: list[ExternalSample] = []
    n_contaminated = 0
    for key in chosen:
        clean = [r for r in pools[key].train if norm_key(r.text) not in forbidden]
        n_contaminated += len(pools[key].train) - len(clean)
        gathered.extend(cap_per_class(clean, cap))
    labels: dict[str, set[bool]] = {}
    for r in gathered:
        labels.setdefault(norm_key(r.text), set()).add(r.is_attack)
    seen: set[str] = set()
    unique: list[ExternalSample] = []
    for r in gathered:
        k = norm_key(r.text)
        if k in seen or len(labels[k]) > 1:
            continue
        seen.add(k)
        unique.append(r)
    return LodoSplit(held_out, chosen, unique, n_contaminated, len(gathered) - len(unique))


# ---------------------------------------------------------------------------
# Decision rule (pure arithmetic on already-computed rates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetScore:
    """One system's full-gate detection rate and FPR on one held-out TEST split."""

    key: str
    detection: float
    fpr: float

    @property
    def youden(self) -> float:
        return self.detection - self.fpr


@dataclass(frozen=True)
class RuleOutcome:
    mean_legacy: float
    mean_pooled: float
    beats_mean: bool
    fpr_violations: tuple[str, ...]  # held-out datasets where pooled FPR > legacy FPR + slack

    @property
    def passes(self) -> bool:
        return self.beats_mean and not self.fpr_violations


def decide(
    legacy: Sequence[DatasetScore], pooled: Sequence[DatasetScore], slack: float = FPR_SLACK
) -> RuleOutcome:
    """Pooled becomes the default iff mean J beats legacy and no FPR exceeds legacy's by > slack."""
    if [s.key for s in legacy] != [s.key for s in pooled] or not legacy:
        raise ValueError("legacy and pooled scores must cover the same, non-empty datasets")
    mean_legacy = float(np.mean([s.youden for s in legacy]))
    mean_pooled = float(np.mean([s.youden for s in pooled]))
    violations = tuple(
        lg.key for lg, pl in zip(legacy, pooled, strict=True) if pl.fpr > lg.fpr + slack + 1e-12
    )
    return RuleOutcome(mean_legacy, mean_pooled, mean_pooled > mean_legacy, violations)


def dataset_score(key: str, outcome: Outcome) -> DatasetScore:
    return DatasetScore(key, outcome.detection().value, outcome.fpr().value)


def bootstrap_mean_dj(
    pairs: Sequence[tuple[Outcome, Outcome]],
    n_boot: int = BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% bootstrap interval of mean-over-datasets (J_b - J_a); informational, not the rule.

    Each pair scores the same TEST rows. Attacks and benign rows are resampled separately inside
    every dataset, so the paired difference keeps its correlation.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for a, b in pairs:
        atk = np.array([s.is_attack for s, _ in a.results])
        blk_a = np.array([x for _, x in a.results])
        blk_b = np.array([x for _, x in b.results])
        prepared.append((np.flatnonzero(atk), np.flatnonzero(~atk), blk_a, blk_b))
    draws = np.zeros(n_boot)
    for i in range(n_boot):
        total = 0.0
        for pos, neg, blk_a, blk_b in prepared:
            p = rng.choice(pos, size=len(pos))
            q = rng.choice(neg, size=len(neg))
            total += (blk_b[p].mean() - blk_b[q].mean()) - (blk_a[p].mean() - blk_a[q].mean())
        draws[i] = total / len(prepared)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class SystemResult:
    name: str
    outcomes: dict[str, Outcome]
    auc: float  # stage-2 score AUC on the held-out TEST rows (threshold-free, post hoc)


@dataclass
class HeldOut:
    key: str
    n_test: int
    pooled_split: LodoSplit
    legacy: SystemResult  # shipped legacy, bundled only
    legacy_pooled: SystemResult  # legacy model refit on the pooled rows (ablation)
    singles: list[tuple[str, LodoSplit, SystemResult]]  # v2 trained on bundled + one source
    pooled: SystemResult


def _as_samples(rows: Sequence[ExternalSample]) -> list[Sample]:
    return [Sample(f"x{i}", r.text, None, r.is_attack) for i, r in enumerate(rows)]


def _result(name: str, detector: InjectionDetector, rows: Sequence[ExternalSample]) -> SystemResult:
    y = np.array([int(r.is_attack) for r in rows], dtype=np.int64)
    auc = roc_auc(np.array([detector.stage2_score(r.text) for r in rows]), y)
    return SystemResult(name, evaluate(detector, rows), auc)


def lodo_scenarios(
    bundled: Sequence[Sample],
    pools: dict[str, Pool],
    *,
    keys: Sequence[str] = LODO_KEYS,
    log: TextIO | None = None,
) -> list[HeldOut]:
    """Score every held-out dataset once for each system. TEST splits are read only here."""
    out: list[HeldOut] = []
    for h in keys:
        test = pools[h].test
        if log:
            print(f"  held out: {h} (test n={len(test)})", file=log, flush=True)
        pooled_split = lodo_split(pools, h)
        pooled_det = v2_detector(fit_v2(bundled, pooled_split.train))
        legacy_pooled = legacy_detector(list(bundled) + _as_samples(pooled_split.train))
        singles: list[tuple[str, LodoSplit, SystemResult]] = []
        for s in pooled_split.sources:
            sp = lodo_split(pools, h, sources=[s])
            det = v2_detector(fit_v2(bundled, sp.train))
            singles.append((s, sp, _result(f"v2-single ({s})", det, test)))
        out.append(
            HeldOut(
                h,
                len(test),
                pooled_split,
                _result("legacy (shipped)", legacy_detector(), test),
                _result("legacy refit pooled", legacy_pooled, test),
                singles,
                _result("v2 pooled", pooled_det, test),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_split_table(
    bundled: Sequence[Sample], pools: dict[str, Pool], out: TextIO, keys: Sequence[str] = LODO_KEYS
) -> None:
    """Dataset sizes and the pooled training composition per held-out dataset (no scoring)."""
    print("| Dataset | TRAIN rows (attacks / benign) | TEST rows (attacks / benign) |", file=out)
    print("| --- | --- | --- |", file=out)
    for k in keys:
        tr, te = pools[k].train, pools[k].test
        a_tr, a_te = sum(r.is_attack for r in tr), sum(r.is_attack for r in te)
        print(
            f"| {k} | {len(tr)} ({a_tr} / {len(tr) - a_tr}) | "
            f"{len(te)} ({a_te} / {len(te) - a_te}) |",
            file=out,
        )
    print(file=out)
    print(
        f"Bundled synthetic set: {len(bundled)} rows, always in training. Per-class cap "
        f"{MAX_TRAIN_PER_CLASS} per dataset.\n",
        file=out,
    )
    print(
        "| Held out | Pooled training sources | Pooled rows (attacks / benign) | "
        "Dropped: text in held-out TEST | Dropped: duplicate across sources |",
        file=out,
    )
    print("| --- | --- | --- | --- | --- |", file=out)
    for h in keys:
        sp = lodo_split(pools, h)
        print(
            f"| {h} | {', '.join(sp.sources)} | {len(sp.train)} ({sp.n_attacks} / {sp.n_benign}) "
            f"| {sp.n_contaminated} | {sp.n_duplicate} |",
            file=out,
        )
    print(file=out)


def _rows(res: SystemResult) -> list[tuple[str, str]]:
    variants = VARIANTS if res.name.startswith("v2") else (VARIANTS[0], VARIANTS[2])
    return [(v, v) for v in variants]


def print_lodo(results: Sequence[HeldOut], out: TextIO) -> None:
    """Per-held-out tables, language split, summary and the decision."""
    for ho in results:
        print(f"#### Held out: {ho.key} (TEST n={ho.n_test})\n", file=out)
        print(
            "| System | AUC | Decision variant | Detection rate on attacks | "
            "False-positive rate on benign |",
            file=out,
        )
        print("| --- | --- | --- | --- | --- |", file=out)
        systems = [ho.legacy, ho.legacy_pooled, *[r for _, _, r in ho.singles], ho.pooled]
        for res in systems:
            for v, _ in _rows(res):
                o = res.outcomes[v]
                print(
                    f"| {res.name} | {res.auc:.3f} | {v} | {o.detection().fmt()} | "
                    f"{o.fpr().fmt()} |",
                    file=out,
                )
        print(file=out)
        counts = {
            lang: ho.legacy.outcomes[GATE_REFUSE].detection(lang).n
            + ho.legacy.outcomes[GATE_REFUSE].fpr(lang).n
            for lang in ("en", "de", "other")
        }
        langs = [lang for lang, n in counts.items() if n >= 10]
        if len(langs) >= 2:
            print(f"Per language, held out {ho.key} (heuristic `guess_language`):\n", file=out)
            print(
                "| Language | System | Decision variant | Detection rate on attacks | "
                "False-positive rate on benign |",
                file=out,
            )
            print("| --- | --- | --- | --- | --- |", file=out)
            for lang in langs:
                for res in systems:
                    for v, _ in _rows(res):
                        o = res.outcomes[v]
                        print(
                            f"| {lang} | {res.name} | {v} | {o.detection(lang).fmt()} | "
                            f"{o.fpr(lang).fmt()} |",
                            file=out,
                        )
            print(file=out)

    print(
        "#### Summary: full gate, refuse only (Youden's J = detection - FPR, in points)\n", file=out
    )
    print(
        "| Held out | legacy (shipped) | legacy refit pooled | v2-single (mean of sources) | "
        "v2 pooled |",
        file=out,
    )
    print("| --- | --- | --- | --- | --- |", file=out)

    def j(res: SystemResult) -> float:
        o = res.outcomes[GATE_REFUSE]
        return 100.0 * (o.detection().value - o.fpr().value)

    cols: list[list[float]] = [[], [], [], []]
    for ho in results:
        single_j = float(np.mean([j(r) for _, _, r in ho.singles]))
        row = [j(ho.legacy), j(ho.legacy_pooled), single_j, j(ho.pooled)]
        for c, val in zip(cols, row, strict=True):
            c.append(val)
        print(f"| {ho.key} | " + " | ".join(f"{val:.1f}" for val in row) + " |", file=out)
    print("| **mean** | " + " | ".join(f"**{np.mean(c):.1f}**" for c in cols) + " |\n", file=out)

    legacy = [dataset_score(ho.key, ho.legacy.outcomes[GATE_REFUSE]) for ho in results]
    pooled = [dataset_score(ho.key, ho.pooled.outcomes[GATE_REFUSE]) for ho in results]
    rule = decide(legacy, pooled)
    print("#### Decision rule (fixed before scoring)\n", file=out)
    print(
        f"- mean J: legacy {100 * rule.mean_legacy:.1f}, pooled {100 * rule.mean_pooled:.1f}: "
        f"{'pass' if rule.beats_mean else 'FAIL'}",
        file=out,
    )
    for lg, pl in zip(legacy, pooled, strict=True):
        flag = "FAIL" if lg.key in rule.fpr_violations else "ok"
        print(
            f"- {lg.key}: pooled FPR {100 * pl.fpr:.1f}% vs legacy "
            f"{100 * lg.fpr:.1f}% (limit +{100 * FPR_SLACK:.0f} points): {flag}",
            file=out,
        )
    lo, hi = bootstrap_mean_dj(
        [(ho.legacy.outcomes[GATE_REFUSE], ho.pooled.outcomes[GATE_REFUSE]) for ho in results]
    )
    print(
        f"- informational, not part of the rule: mean J gain of pooled over legacy "
        f"{100 * (rule.mean_pooled - rule.mean_legacy):+.1f} points, bootstrap 95% interval "
        f"[{100 * lo:+.1f}, {100 * hi:+.1f}] ({BOOTSTRAPS} resamples, seed {BOOTSTRAP_SEED})",
        file=out,
    )
    verdict = "pooled becomes the default" if rule.passes else "legacy stays the default"
    print(f"\n**Decision: {verdict}.**\n", file=out)

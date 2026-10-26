"""Safegate scored on the public sets that published guards were scored on (``safegate benchmark``).

``docs/protocol-sota.md`` fixes the sets, the metrics and the reading of every number before any
score. Each :class:`SuiteSpec` names one public set (:mod:`safegate.publicsets`) and the metrics
its published numbers use; :mod:`safegate.published` holds those numbers with their sources. A
safegate row is printed next to the published rows of the same metric.

Two verdicts exist for every safegate number: the *flag* (``refuse`` or ``escalate`` of the full
gate, optionally behind the shipped defence) for F1, recall and accuracy, and a *ranking score*
for AUC and TPR at a target FPR. The ranking score puts every flagged prompt above every allowed
one and orders each group by the detector's own score, so the ROC curve passes through the gate's
operating point and the gate's flag is one point on it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from safegate.benchmark import SUITES, Suite
from safegate.defence import SHIPPED_COMPONENTS, DefenceConfig
from safegate.external import Fetcher, Rate, http_get
from safegate.fresh import FreshRow
from safegate.gate import BLOCKING_ACTIONS, SafetyGate
from safegate.injection import InjectionDetector
from safegate.metrics import (
    Confusion,
    auc_ci,
    f1_ci,
    f1_from_recall,
    roc_auc,
    tpr_at_fpr,
)
from safegate.publicsets import SET_BY_KEY, leaked, load_set, training_texts
from safegate.published import (
    PUBLISHED,
    SOURCES,
    format_value,
    not_comparable_rows,
)

Judge = Callable[[str], tuple[bool, float]]  # text -> (flagged, ranking score)
RANK_BONUS = 2.0  # a flagged prompt outranks every allowed one (detector scores are in [0, 1])
SHIPPED = DefenceConfig.of(SHIPPED_COMPONENTS)


@dataclass(frozen=True)
class Metric:
    """One published metric: ``kind`` is f1, recall, auc or tpr (at ``target`` FPR)."""

    key: str
    kind: str
    label: str
    target: float | None = None


F1 = Metric("f1", "f1", "F1 of the flagged class at the guard's own decision")
F1_RECALL = Metric(
    "f1", "f1_recall", "F1 of the harmful class (all rows harmful, so F1 = 2R/(1+R))"
)
RECALL = Metric("recall", "recall", "recall of the attack class at the guard's own decision")
AUC = Metric("auc", "auc", "ROC AUC (threshold-free)")
TPR = tuple(
    Metric(f"tpr@{t:g}", "tpr", f"TPR at {t:g}% FPR (threshold set on the test rows)", t / 100)
    for t in (1.0, 0.5, 0.1, 0.05)
)


@dataclass(frozen=True)
class SuiteSpec:
    """One public set and the metrics its published rows use."""

    key: str
    metrics: tuple[Metric, ...]
    reading: str  # what the numbers mean for safegate, stated before scoring


SUITE_SPECS: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        "promptshield",
        (AUC, *TPR),
        "injection detection on conversational and task prompts; thresholds are set on the test "
        "rows themselves, as the paper did",
    ),
    SuiteSpec("deepset", (F1,), "injection detection, English and German, 116 prompts"),
    SuiteSpec("jailbreak", (F1,), "jailbreak detection against benign prompts, 262 prompts"),
    SuiteSpec("xtram1", (AUC,), "synthetic injections against benign prompts"),
    SuiteSpec("slabs", (AUC,), "short injections against benign prompts"),
    SuiteSpec(
        "jbb",
        (AUC, F1),
        "harmful behaviour requests against topic-paired benign requests: harm detection, "
        "outside the injection threat model",
    ),
    SuiteSpec("gandalf_test", (RECALL,), "attacks only: recall of 'ignore previous instructions'"),
    SuiteSpec(
        "harmbench",
        (F1_RECALL,),
        "plain harmful requests, no injection wording: harm detection, outside the injection "
        "threat model",
    ),
    SuiteSpec(
        "toxicchat",
        (F1,),
        "toxic user prompts (harm detection, outside the injection threat model); 91 of the 362 "
        "are also jailbreak attempts",
    ),
)
SUITE_BY_KEY = {s.key: s for s in SUITE_SPECS}
SOTA_KEYS = tuple(SUITE_BY_KEY)
CONFIGS: tuple[tuple[str, DefenceConfig | None], ...] = (
    ("gate alone", None),
    ("shipped defence, no windows", SHIPPED),
)


def make_judge(stage2: str, defence: DefenceConfig | None) -> Judge:
    """Flag and ranking score of the full gate, memoised by text."""
    gate = SafetyGate(detector=InjectionDetector(stage2=stage2), defence=defence)
    cache: dict[str, tuple[bool, float]] = {}

    def judge(text: str) -> tuple[bool, float]:
        hit = cache.get(text)
        if hit is None:
            decision = gate.check_input(text)
            flagged = decision.action in BLOCKING_ACTIONS
            detection = float(decision.metadata.get("detection_score", decision.score))
            hit = (flagged, detection + (RANK_BONUS if flagged else 0.0))
            cache[text] = hit
            gate.audit_records.clear()
        return hit

    return judge


@dataclass(frozen=True)
class SetScore:
    """One configuration of one gate on one set, with everything the metrics need."""

    set_key: str
    stage2: str
    config: str
    labels: tuple[bool, ...]
    flags: tuple[bool, ...]
    scores: tuple[float, ...]
    leak: tuple[bool, ...]
    groups: tuple[str, ...]

    @property
    def n_leaked(self) -> int:
        return sum(self.leak)

    def clean(self) -> SetScore:
        """Only the rows whose text is not in safegate's training data."""
        keep = [i for i, leak in enumerate(self.leak) if not leak]
        return SetScore(
            self.set_key,
            self.stage2,
            self.config,
            tuple(self.labels[i] for i in keep),
            tuple(self.flags[i] for i in keep),
            tuple(self.scores[i] for i in keep),
            tuple(False for _ in keep),
            tuple(self.groups[i] for i in keep),
        )

    def confusion(self) -> Confusion:
        return Confusion.of(self.labels, self.flags)


def score_set(
    rows: Sequence[FreshRow], judge: Judge, train_keys: set[str], stage2: str, config: str
) -> SetScore:
    verdicts = [judge(r.text) for r in rows]
    return SetScore(
        rows[0].source,
        stage2,
        config,
        tuple(r.is_attack for r in rows),
        tuple(v[0] for v in verdicts),
        tuple(v[1] for v in verdicts),
        tuple(leaked(list(rows), train_keys)),
        tuple(r.group for r in rows),
    )


def run_sota(
    stages: Sequence[str],
    keys: Sequence[str],
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
    fetch: Fetcher = http_get,
) -> tuple[dict[str, list[FreshRow]], list[SetScore]]:
    """Score every gate in ``stages`` on every set in ``keys``, in both configurations."""
    data = {k: load_set(k, cache_dir, fetch, refresh) for k in keys}
    train_keys = training_texts(cache_dir, fetch)
    scores: list[SetScore] = []
    for stage2 in stages:
        for name, defence in CONFIGS:
            judge = make_judge(stage2, defence)
            for k in keys:
                scores.append(score_set(data[k], judge, train_keys, stage2, name))
    return data, scores


# ---------------------------------------------------------------------------
# Reading a metric off a SetScore
# ---------------------------------------------------------------------------


def system_name(stage2: str, config: str, extra: str = "") -> str:
    gate = "default gate (v2)" if stage2 == "v2" else "legacy gate"
    return f"safegate {gate}, {config}{extra}"


def value_of(metric: Metric, s: SetScore) -> float:
    """The metric as a fraction (NaN when the defence row has no ranking score)."""
    c = s.confusion()
    if metric.kind == "f1":
        return c.f1
    if metric.kind == "f1_recall":
        return f1_from_recall(c.recall)
    if metric.kind == "recall":
        return c.recall
    if metric.kind == "auc":
        return roc_auc(s.labels, s.scores)
    assert metric.target is not None
    return tpr_at_fpr(s.labels, s.scores, metric.target).tpr


def has_ranking(metric: Metric, s: SetScore) -> bool:
    """AUC and TPR@FPR are read off the gate alone: the defence adds views, not a score."""
    return metric.kind not in ("auc", "tpr") or s.config == "gate alone"


def score_cell(metric: Metric, s: SetScore) -> str:
    """The score of a safegate row as printed, with its interval or achieved FPR."""
    c = s.confusion()
    if metric.kind == "f1":
        lo, hi = f1_ci(s.labels, s.flags)
        return f"{c.f1:.3f} [{lo:.3f}, {hi:.3f}] (P {c.precision:.3f}, R {c.recall:.3f})"
    if metric.kind == "f1_recall":
        lo, hi = Rate(c.tp, c.tp + c.fn).ci
        return (
            f"{f1_from_recall(c.recall):.3f} [{f1_from_recall(lo):.3f}, {f1_from_recall(hi):.3f}]"
            f" (recall {Rate(c.tp, c.tp + c.fn).fmt()})"
        )
    if metric.kind == "recall":
        return Rate(c.tp, c.tp + c.fn).fmt()
    if metric.kind == "auc":
        lo, hi = auc_ci(s.labels, s.scores)
        return f"{roc_auc(s.labels, s.scores):.3f} [{lo:.3f}, {hi:.3f}]"
    assert metric.target is not None
    t = tpr_at_fpr(s.labels, s.scores, metric.target)
    return f"{t.tpr:.2%} (achieved FPR {t.fpr:.3%})"


def safegate_rows(
    suite: SuiteSpec, metric: Metric, scores: Sequence[SetScore]
) -> list[tuple[float, tuple[str, ...]]]:
    """(sort value, table row) for every safegate configuration that has this metric."""
    out: list[tuple[float, tuple[str, ...]]] = []
    for s in scores:
        if s.set_key != suite.key or not has_ranking(metric, s):
            continue
        k = s.n_leaked
        yes = "yes (same rows, same metric)"
        variants = [
            ("", s, yes + (f"; {k}/{len(s.labels)} rows also occur in training data" if k else ""))
        ]
        if k and k < len(s.labels):
            variants.append(
                (
                    f", {len(s.labels) - k} rows not in the training data",
                    s.clean(),
                    "no: a subset of the rows, shown to separate memorisation",
                )
            )
        for extra, sc, comparable in variants:
            if len(set(sc.labels)) < 2 and metric.kind in ("auc", "tpr"):
                continue
            out.append(
                (
                    value_of(metric, sc),
                    (
                        system_name(s.stage2, s.config, extra),
                        f"this repository, `safegate benchmark --suite {suite.key}`",
                        metric.label,
                        score_cell(metric, sc),
                        comparable,
                    ),
                )
            )
    return out


def published_rows(suite: SuiteSpec, metric: Metric) -> list[tuple[float, tuple[str, ...]]]:
    """Published rows of the same set and metric whose protocol is identical."""
    out: list[tuple[float, tuple[str, ...]]] = []
    for p in PUBLISHED:
        if p.suite != suite.key or p.metric != metric.key or p.comparable:
            continue
        note = f" ({p.note})" if p.note else ""
        out.append(
            (
                p.value,
                (
                    p.system,
                    SOURCES[p.source].cite(),
                    metric.label,
                    format_value(metric.kind, p.value) + note,
                    "yes (same rows, same metric)",
                ),
            )
        )
    return out


def comparison_table(
    suite: SuiteSpec, metric: Metric, scores: Sequence[SetScore]
) -> list[tuple[str, ...]]:
    """Safegate and published rows of one metric, best first."""
    rows = safegate_rows(suite, metric, scores) + published_rows(suite, metric)
    return [r for _, r in sorted(rows, key=lambda x: -(x[0] if x[0] == x[0] else -1.0))]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_table(rows: Sequence[Sequence[str]], out: TextIO) -> None:
    print("| System | Source | Metric | Score | Comparable? |", file=out)
    print("| --- | --- | --- | --- | --- |", file=out)
    for r in rows:
        print("| " + " | ".join(r) + " |", file=out)
    print(file=out)


def extras_table(suite: SuiteSpec, scores: Sequence[SetScore], out: TextIO) -> None:
    """Precision, recall, FPR, AUC of every safegate configuration: context for the F1 rows."""
    print("| Configuration | Flagged | Precision | Recall | FPR | AUC |", file=out)
    print("| --- | --- | --- | --- | --- | --- |", file=out)
    for s in scores:
        if s.set_key != suite.key:
            continue
        c = s.confusion()
        auc = f"{roc_auc(s.labels, s.scores):.3f}" if s.config == "gate alone" else "n/a"
        fpr = Rate(c.fp, c.fp + c.tn).fmt() if c.fp + c.tn else "n/a (no benign rows)"
        print(
            f"| {system_name(s.stage2, s.config)} | {c.tp + c.fp}/{len(s.labels)} "
            f"| {c.precision:.3f} | {Rate(c.tp, c.tp + c.fn).fmt()} | {fpr} | {auc} |",
            file=out,
        )
    print(file=out)


def group_table(suite: SuiteSpec, scores: Sequence[SetScore], out: TextIO) -> None:
    """Flag rates per group of rows (sets with attack-only or mixed groups)."""
    groups = sorted({g for s in scores if s.set_key == suite.key for g in s.groups})
    if len(groups) < 2:
        return
    print("| Configuration | Group | Rows | Flagged |", file=out)
    print("| --- | --- | --- | --- |", file=out)
    for s in scores:
        if s.set_key != suite.key:
            continue
        for g in groups:
            flags = [f for f, gg in zip(s.flags, s.groups, strict=True) if gg == g]
            print(
                f"| {system_name(s.stage2, s.config)} | {g} | {len(flags)} "
                f"| {Rate(sum(flags), len(flags)).fmt()} |",
                file=out,
            )
    print(file=out)


def print_sota(
    data: dict[str, list[FreshRow]],
    scores: Sequence[SetScore],
    keys: Sequence[str],
    out: TextIO | None = None,
) -> None:
    out = out or sys.stdout
    print("## Benchmark against published guards: further public sets\n", file=out)
    print(
        "Flagged = `refuse` or `escalate`. F1, recall and AUC carry bootstrap or Wilson 95% "
        "intervals. Published rows are transcribed from the sources named in each row; no other "
        "guard is run here. Where rows also occur in safegate's training data, the score is "
        "printed a second time without them.\n",
        file=out,
    )
    for key in keys:
        spec = SET_BY_KEY[key]
        rows = data[key]
        pos = sum(r.is_attack for r in rows)
        print(
            f"- {spec.title}: {spec.origin} @ `{spec.revision}`, licence {spec.license}; "
            f"{len(rows)} rows ({pos} {spec.positive}, {len(rows) - pos} other)",
            file=out,
        )
    print(file=out)
    for key in keys:
        suite = SUITE_BY_KEY[key]
        spec = SET_BY_KEY[key]
        print(f"### {spec.title}\n", file=out)
        print(f"Reading: {suite.reading}.\n", file=out)
        for metric in suite.metrics:
            print(f"**{metric.label}**\n", file=out)
            print_table(comparison_table(suite, metric, scores), out)
        print("Safegate in detail:\n", file=out)
        extras_table(suite, scores, out)
        group_table(suite, scores, out)
    print("### Not comparable (published, but not scorable under the same protocol)\n", file=out)
    print("| System or benchmark | Source | Reported | Why not comparable |", file=out)
    print("| --- | --- | --- | --- |", file=out)
    for r in not_comparable_rows():
        print("| " + " | ".join(r) + " |", file=out)
    print(file=out)


def resolve_suites(name: str) -> tuple[list[Suite], tuple[str, ...]]:
    """The InjecGuard suites and the further public sets that ``--suite NAME`` selects."""
    if name == "all":
        return list(SUITES.values()), SOTA_KEYS
    if name == "injecguard":
        return list(SUITES.values()), ()
    if name == "public":
        return [], SOTA_KEYS
    if name in SUITES:
        return [SUITES[name]], ()
    return [], (name,)

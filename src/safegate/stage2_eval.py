"""Protocol and reporting for the stage-2 comparison (``safegate stage2-report``).

Protocol (fixed before any v2 score on a TEST split was computed):

* Model selection (features, class weighting, L2) uses TRAIN splits only: five-fold
  cross-validation inside each dataset's TRAIN split, plus training on one dataset's TRAIN split
  and scoring the *other* dataset's TRAIN split (:func:`dev_grid`). TEST splits are not touched.
* Thresholds are picked on out-of-fold validation scores cut from the training rows, refuse at
  validation FPR <= 2%, review band up to FPR <= 10% (:func:`fit_v2`).
* Scenarios (:func:`scenarios`): (a) in-domain, (b) cross-dataset, (c) bundled data only,
  (d) the former shipped file (bundled + deepset + jailbreak TRAIN), refit. Each is scored once
  on a TEST split, for legacy and for v2.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

from safegate.external import (
    ExternalSample,
    Outcome,
    Rate,
    shorten,
)
from safegate.gate import SafetyGate
from safegate.injection import InjectionDetector, legacy_bucket, legacy_terms
from safegate.stage2 import (
    DEFAULT_L2,
    FeatureSpec,
    SparseRows,
    Stage2Model,
    Thresholds,
    fit_logreg,
    fit_stage2,
    roc_auc,
    sigmoid,
    stratified_folds,
)
from safegate.taxonomy import Sample

REFUSE_FPR = 0.02
REVIEW_FPR = 0.10
FOLDS = 5
SEED = 0
DATASET_KEYS = ("deepset", "jailbreak")


@dataclass(frozen=True)
class Pool:
    """One public dataset's TRAIN and TEST splits."""

    key: str
    train: list[ExternalSample]
    test: list[ExternalSample]


def _label_array(rows: Sequence[ExternalSample]) -> np.ndarray:
    return np.array([int(r.is_attack) for r in rows], dtype=np.int64)


def fit_v2(
    bundled: Sequence[Sample],
    external_train: Sequence[ExternalSample],
    *,
    spec: FeatureSpec | None = None,
    balanced: bool = True,
    l2: float = DEFAULT_L2,
) -> Stage2Model:
    """Fit v2 on the bundled set plus public TRAIN rows; thresholds come from out-of-fold scores.

    When public rows are present the thresholds are chosen on *their* validation scores only:
    the synthetic prompts are easy and would make the band too lenient.
    """
    texts = [s.text for s in bundled] + [r.text for r in external_train]
    labels = [s.is_attack for s in bundled] + [r.is_attack for r in external_train]
    cal = [False] * len(bundled) + [True] * len(external_train)
    return fit_stage2(
        texts,
        labels,
        spec=spec,
        calibration_mask=cal if external_train else None,
        balanced=balanced,
        l2=l2,
        folds=FOLDS,
        refuse_fpr=REFUSE_FPR,
        review_fpr=REVIEW_FPR,
        seed=SEED,
    )


def v2_detector(model: Stage2Model) -> InjectionDetector:
    return InjectionDetector(stage2="v2", model=model)


def legacy_detector(samples: Sequence[Sample] | None = None) -> InjectionDetector:
    """The legacy stage 2: bundled set only, or refit on ``samples``."""
    det = InjectionDetector(stage2="legacy")
    if samples is not None:
        det.fit(list(samples))
    return det


# ---------------------------------------------------------------------------
# Scoring one system on one set
# ---------------------------------------------------------------------------

VARIANTS = (
    "detector alone, refuse",
    "detector alone, refuse or review",
    "full gate, refuse only",
    "full gate, refuse or escalate",
)


def evaluate(detector: InjectionDetector, rows: Sequence[ExternalSample]) -> dict[str, Outcome]:
    """Score every prompt once; derive the four decision variants of :data:`VARIANTS`."""
    gate = SafetyGate(detector=detector)
    results: dict[str, list[tuple[ExternalSample, bool]]] = {v: [] for v in VARIANTS}
    for r in rows:
        det = detector.detect(r.text)
        action = gate.check_input(r.text).action
        results[VARIANTS[0]].append((r, det.is_injection))
        results[VARIANTS[1]].append((r, det.is_injection or det.review))
        results[VARIANTS[2]].append((r, action == "refuse"))
        results[VARIANTS[3]].append((r, action in ("refuse", "escalate")))
    return {v: Outcome(v, res) for v, res in results.items()}


@dataclass
class Scenario:
    """One train/test pairing, scored for legacy and v2."""

    name: str
    train_desc: str
    test_key: str
    legacy: dict[str, Outcome]
    v2: dict[str, Outcome]
    thresholds: Thresholds
    auc_legacy: float  # threshold-free stage-2 ranking quality on the TEST rows (post hoc)
    auc_v2: float


def scenarios(
    bundled: Sequence[Sample],
    pools: dict[str, Pool],
    *,
    log: TextIO | None = None,
) -> list[Scenario]:
    """The protocol's scenarios, in report order. TEST splits are scored here and nowhere else."""
    ds, jb = pools["deepset"], pools["jailbreak"]

    def as_samples(rows: Sequence[ExternalSample]) -> list[Sample]:
        return [Sample(f"x{i}", r.text, None, r.is_attack) for i, r in enumerate(rows)]

    plan: list[tuple[str, str, list[ExternalSample], Pool]] = [
        ("(a1) in-domain", "bundled + deepset TRAIN", ds.train, ds),
        ("(a2) in-domain", "bundled + jailbreak TRAIN", jb.train, jb),
        ("(b1) cross-dataset", "bundled + deepset TRAIN", ds.train, jb),
        ("(b2) cross-dataset", "bundled + jailbreak TRAIN", jb.train, ds),
        ("(c1) bundled only", "bundled", [], ds),
        ("(c2) bundled only", "bundled", [], jb),
        ("(d1) two-dataset v2", "bundled + deepset + jailbreak TRAIN", ds.train + jb.train, ds),
        ("(d2) two-dataset v2", "bundled + deepset + jailbreak TRAIN", ds.train + jb.train, jb),
    ]
    out: list[Scenario] = []
    fitted: dict[str, Stage2Model] = {}
    for name, desc, ext_train, test_pool in plan:
        if log:
            print(f"  scoring {name} train={desc} test={test_pool.key}", file=log, flush=True)
        if desc not in fitted:
            fitted[desc] = fit_v2(bundled, ext_train)
        model = fitted[desc]
        v2 = v2_detector(model)
        legacy = legacy_detector(list(bundled) + as_samples(ext_train) if ext_train else None)
        rows = test_pool.test
        y = _label_array(rows)
        auc_legacy = roc_auc(np.array([legacy.stage2_score(r.text) for r in rows]), y)
        auc_v2 = roc_auc(model.score_many([r.text for r in rows]), y)
        out.append(
            Scenario(
                name,
                desc,
                test_pool.key,
                evaluate(legacy, rows),
                evaluate(v2, rows),
                model.thresholds,
                auc_legacy,
                auc_v2,
            )
        )
    return out


def print_scenarios(scs: Sequence[Scenario], out: TextIO) -> None:
    """One table per scenario, then per-language tables for the mixed-language TEST set."""
    for sc in scs:
        print(
            f"#### {sc.name}: train on {sc.train_desc}, test on {sc.test_key} TEST "
            f"(v2 thresholds: review {sc.thresholds.review:.3f}, "
            f"refuse {sc.thresholds.refuse:.3f})\n",
            file=out,
        )
        print(
            f"Stage-2 score AUC on this TEST set (threshold-free): legacy {sc.auc_legacy:.3f}, "
            f"v2 {sc.auc_v2:.3f}\n",
            file=out,
        )
        print(
            "| Stage 2 | System | Detection rate on attacks | False-positive rate on benign |",
            file=out,
        )
        print("| --- | --- | --- | --- |", file=out)
        for label, outcomes, variants in (
            ("legacy", sc.legacy, (VARIANTS[0], VARIANTS[2])),
            ("v2", sc.v2, VARIANTS),
        ):
            for v in variants:
                o = outcomes[v]
                print(f"| {label} | {v} | {o.detection().fmt()} | {o.fpr().fmt()} |", file=out)
        print(file=out)
    deepset = [sc for sc in scs if sc.test_key == "deepset"]
    if deepset:
        print("#### deepset TEST by language (heuristic `guess_language`)\n", file=out)
        print(
            "| Scenario | Language | Stage 2 | System | Detection rate on attacks | "
            "False-positive rate on benign |",
            file=out,
        )
        print("| --- | --- | --- | --- | --- | --- |", file=out)
        for sc in deepset:
            for lang in ("en", "de", "other"):
                for label, outcomes, variants in (
                    ("legacy", sc.legacy, (VARIANTS[0], VARIANTS[2])),
                    ("v2", sc.v2, VARIANTS),
                ):
                    for v in variants:
                        o = outcomes[v]
                        d, f = o.detection(lang), o.fpr(lang)
                        if d.n + f.n < 10:
                            continue
                        print(
                            f"| {sc.name} | {lang} | {label} | {v} | {d.fmt()} | {f.fmt()} |",
                            file=out,
                        )
        print(file=out)


# ---------------------------------------------------------------------------
# Diagnosis of the legacy stage 2 (TRAIN splits only)
# ---------------------------------------------------------------------------


def legacy_contributions(det: InjectionDetector, text: str) -> dict[str, float]:
    """Per-feature-string contribution to the legacy logit (collisions share a bucket)."""
    counts = Counter(legacy_terms(text))
    per_bucket: dict[int, float] = defaultdict(float)
    for term, c in counts.items():
        idx, sign = legacy_bucket(term, det.dim)
        per_bucket[idx] += sign * c
    norm = float(np.sqrt(sum(v * v for v in per_bucket.values()))) or 1.0
    return {
        term: legacy_bucket(term, det.dim)[1]
        * c
        * float(det.weights[legacy_bucket(term, det.dim)[0]])
        / norm
        for term, c in counts.items()
    }


def _quantiles(x: np.ndarray) -> str:
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return " / ".join(f"{v:.2f}" for v in q)


def print_diagnosis(
    bundled: Sequence[Sample], pools: dict[str, Pool], out: TextIO, *, n_fp: int = 10, top: int = 10
) -> None:
    """Why the legacy stage 2 over-flags: score spread, top weights, FP terms and FP examples."""
    det = legacy_detector()
    b_scores = np.array([det.stage2_score(s.text) for s in bundled])
    b_y = np.array([int(s.is_attack) for s in bundled])
    print(
        "### Legacy stage 2: score spread (TRAIN splits; the model was fitted on neither)\n",
        file=out,
    )
    print(
        f"bias {det.bias:.3f}, weights in [{det.weights.min():.2f}, {det.weights.max():.2f}]",
        file=out,
    )
    print(
        f"On its own 403 training samples: AUC {roc_auc(b_scores, b_y):.3f}, accuracy at 0.55 "
        f"{float(np.mean((b_scores >= det.threshold) == (b_y == 1))):.1%}\n",
        file=out,
    )
    print(
        "| Set | benign score min/q25/median/q75/max | "
        "attack score min/q25/median/q75/max | AUC | benign >= 0.55 | attacks >= 0.55 |",
        file=out,
    )
    print("| --- | --- | --- | --- | --- | --- |", file=out)
    fp_rows: dict[str, list[ExternalSample]] = {}
    for key in DATASET_KEYS:
        rows = pools[key].train
        sc = np.array([det.stage2_score(r.text) for r in rows])
        y = _label_array(rows)
        flagged = sc >= det.threshold
        fp_rows[key] = [r for r, f in zip(rows, flagged, strict=True) if f and not r.is_attack]
        n_benign, n_attack = int(np.sum(y == 0)), int(np.sum(y == 1))
        print(
            f"| {key} TRAIN | {_quantiles(sc[y == 0])} | {_quantiles(sc[y == 1])} | "
            f"{roc_auc(sc, y):.3f} | {Rate(int(np.sum(flagged & (y == 0))), n_benign).fmt()} | "
            f"{Rate(int(np.sum(flagged & (y == 1))), n_attack).fmt()} |",
            file=out,
        )
    print(file=out)

    # inverse map bucket -> most frequent term, from every text the model or this report saw
    inverse: dict[int, Counter[str]] = defaultdict(Counter)
    corpus = [s.text for s in bundled] + [r.text for k in DATASET_KEYS for r in pools[k].train]
    for text in corpus:
        for term in set(legacy_terms(text)):
            inverse[legacy_bucket(term, det.dim)[0]][term] += 1
    order = np.argsort(det.weights)
    print(
        "### Legacy stage 2: largest weights (bucket -> most frequent feature string in it)\n",
        file=out,
    )
    print("| Direction | Feature | Weight | Distinct strings sharing the bucket |", file=out)
    print("| --- | --- | --- | --- |", file=out)
    for direction, idxs in (("attack (+)", order[::-1][:top]), ("benign (-)", order[:top])):
        for i in idxs:
            term = inverse[int(i)].most_common(1)[0][0] if inverse[int(i)] else "?"
            print(
                f"| {direction} | `{term}` | {det.weights[i]:+.2f} | {len(inverse[int(i)])} |",
                file=out,
            )
    print(file=out)

    all_fp = [r for k in DATASET_KEYS for r in fp_rows[k]]
    totals: dict[str, float] = defaultdict(float)
    for r in all_fp:
        for term, c in legacy_contributions(det, r.text).items():
            totals[term] += c
    print(
        f"### Feature strings that push benign TRAIN prompts over 0.55 "
        f"(summed logit contribution over the {len(all_fp)} false positives)\n",
        file=out,
    )
    for term, c in sorted(totals.items(), key=lambda kv: -kv[1])[:top]:
        print(f"- `{term}`: {c:+.2f}", file=out)
    print(file=out)

    for key in DATASET_KEYS:
        rows = fp_rows[key]
        print(
            f"### First {min(n_fp, len(rows))} of {len(rows)} legacy stage-2 false positives, "
            f"{key} TRAIN\n",
            file=out,
        )
        for r in rows[:n_fp]:
            contrib = sorted(legacy_contributions(det, r.text).items(), key=lambda kv: -kv[1])[:3]
            terms = ", ".join(f"`{t}` {c:+.2f}" for t, c in contrib)
            print(
                f"- {shorten(r.text, 110)} "
                f"(score {det.stage2_score(r.text):.2f}; top terms {terms})",
                file=out,
            )
        print(file=out)


# ---------------------------------------------------------------------------
# Model selection on TRAIN splits only
# ---------------------------------------------------------------------------

DEV_SPECS: dict[str, FeatureSpec] = {
    "char3-5 + word1-2": FeatureSpec(word_max=2),
    "char3-5 + word1": FeatureSpec(word_max=1),
    "char2-4 + word1-2": FeatureSpec(char_min=2, char_max=4, word_max=2),
}
DEV_L2 = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)


@dataclass(frozen=True)
class DevRow:
    features: str
    balanced: bool
    l2: float
    auc_in_deepset: float
    auc_in_jailbreak: float
    auc_deepset_to_jailbreak: float
    auc_jailbreak_to_deepset: float

    @property
    def cross(self) -> float:
        return (self.auc_deepset_to_jailbreak + self.auc_jailbreak_to_deepset) / 2

    @property
    def in_domain(self) -> float:
        return (self.auc_in_deepset + self.auc_in_jailbreak) / 2


def select_config(rows: Sequence[DevRow]) -> DevRow:
    """Highest cross-dataset dev AUC; ties go to in-domain AUC, then to the balanced loss."""
    return max(rows, key=lambda r: (round(r.cross, 3), round(r.in_domain, 3), r.balanced))


def dev_grid(
    bundled: Sequence[Sample], pools: dict[str, Pool], *, log: TextIO | None = None
) -> list[DevRow]:
    """Dev AUCs per configuration. Reads TRAIN splits only; TEST splits are never read here."""
    by = np.array([int(s.is_attack) for s in bundled], dtype=np.int64)
    rows: list[DevRow] = []
    for fname, spec in DEV_SPECS.items():
        xb = SparseRows.from_texts([s.text for s in bundled], spec)
        xe = {
            k: SparseRows.from_texts([r.text for r in pools[k].train], spec) for k in DATASET_KEYS
        }
        ye = {k: _label_array(pools[k].train) for k in DATASET_KEYS}
        for balanced, l2 in itertools.product((True, False), DEV_L2):
            kw: dict[str, Any] = {"balanced": balanced, "l2": l2}
            auc: dict[str, float] = {}
            for k in DATASET_KEYS:
                oof = np.zeros(len(ye[k]))
                for val in stratified_folds(ye[k], FOLDS, SEED):
                    mask = np.ones(len(ye[k]), dtype=bool)
                    mask[val] = False
                    tr = np.flatnonzero(mask)
                    w, b = fit_logreg(
                        SparseRows.concat(xb, xe[k].take(tr)), np.concatenate([by, ye[k][tr]]), **kw
                    )
                    oof[val] = sigmoid(xe[k].take(val).dot(w) + b)
                auc[f"in_{k}"] = roc_auc(oof, ye[k])
            for a, o in (("deepset", "jailbreak"), ("jailbreak", "deepset")):
                w, b = fit_logreg(SparseRows.concat(xb, xe[a]), np.concatenate([by, ye[a]]), **kw)
                auc[f"{a}->{o}"] = roc_auc(xe[o].dot(w) + b, ye[o])
            row = DevRow(
                fname,
                balanced,
                l2,
                auc["in_deepset"],
                auc["in_jailbreak"],
                auc["deepset->jailbreak"],
                auc["jailbreak->deepset"],
            )
            rows.append(row)
            if log:
                print(
                    f"  dev {fname} balanced={balanced} l2={l2:g}: cross {row.cross:.3f}",
                    file=log,
                    flush=True,
                )
    return rows


def print_dev_grid(rows: Sequence[DevRow], out: TextIO) -> None:
    print(
        "| Features | Class-balanced | L2 | AUC deepset CV | AUC jailbreak CV | "
        "AUC deepset->jailbreak TRAIN | AUC jailbreak->deepset TRAIN | mean cross |",
        file=out,
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |", file=out)
    for r in rows:
        print(
            f"| {r.features} | {r.balanced} | {r.l2:g} | {r.auc_in_deepset:.3f} | "
            f"{r.auc_in_jailbreak:.3f} | {r.auc_deepset_to_jailbreak:.3f} | "
            f"{r.auc_jailbreak_to_deepset:.3f} | {r.cross:.3f} |",
            file=out,
        )
    best = select_config(rows)
    print(
        f"\nSelected (max mean cross AUC): {best.features}, balanced={best.balanced}, "
        f"l2={best.l2:g}, cross {best.cross:.3f}, in-domain {best.in_domain:.3f}",
        file=out,
    )

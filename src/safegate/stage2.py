"""Stage 2 v2: sparse hashed-feature logistic regression with a calibrated abstain band.

Why this exists: the legacy stage 2 (``injection.extract_features`` + 250 plain gradient steps)
is badly under-fitted: its scores sit in about [0.26, 0.74] with the median benign prompt near
0.51, so a 0.5 or 0.55 cut-off flags roughly half of everything (see ``safegate stage2-report``).

What changes here, and nothing else:

* features: casefolded words plus word-bounded character 3-5-grams (``char_wb``),
  sub-linear term frequency, signed hashing into ``2**16`` buckets, head+tail truncation of very
  long prompts. Character n-grams inside words also cover German compounds.
* training: class-balanced loss, L2, Adam, full batch. Pure numpy on a CSR layout, so the
  ``2**16`` buckets never become a dense matrix.
* decision: two thresholds chosen on validation folds cut from the training data
  (:func:`select_thresholds`): ``>= refuse`` -> refuse, ``[review, refuse)`` -> abstain and
  escalate to a human, below ``review`` -> allow.

Everything is deterministic (CRC32 hashing, seeded folds).
"""

from __future__ import annotations

import json
import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Hashed feature layout. Defaults are the shipped v2 configuration."""

    dim: int = 1 << 16
    char_min: int = 3
    char_max: int = 5
    word_max: int = 1
    sublinear_tf: bool = True
    max_chars: int = 2000  # prompts longer than this keep their first and last max_chars/2

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.dim & (self.dim - 1):
            raise ValueError("dim must be a power of two")
        if not 0 < self.char_min <= self.char_max:
            raise ValueError("need 0 < char_min <= char_max")
        if self.word_max < 1 or self.max_chars < 2:
            raise ValueError("word_max must be >= 1 and max_chars >= 2")


_WORD = re.compile(r"\w+")


def clip_text(text: str, max_chars: int) -> str:
    """Keep the head and the tail of a long prompt (an injected instruction is often last)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " " + text[-half:]


def feature_terms(text: str, spec: FeatureSpec) -> list[str]:
    """Feature strings of one prompt, with duplicates (the count is the raw term frequency)."""
    folded = unicodedata.normalize("NFKC", clip_text(text, spec.max_chars)).casefold()
    words = _WORD.findall(folded)
    terms: list[str] = []
    for n in range(1, spec.word_max + 1):
        terms.extend("w:" + " ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    for w in words:
        padded = f" {w} "
        for n in range(spec.char_min, spec.char_max + 1):
            terms.extend("c:" + padded[i : i + n] for i in range(len(padded) - n + 1))
    return terms


def hash_term(term: str, dim: int) -> tuple[int, float]:
    """Bucket index and sign of a feature string (CRC32: deterministic, unlike ``hash``)."""
    h = zlib.crc32(term.encode("utf-8"))
    return h % dim, -1.0 if h >> 31 else 1.0


def featurize(text: str, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    """One prompt as an L2-normalised sparse row: ``(sorted bucket indices, values)``."""
    acc: dict[int, float] = {}
    for term, count in Counter(feature_terms(text, spec)).items():
        idx, sign = hash_term(term, spec.dim)
        tf = 1.0 + float(np.log(count)) if spec.sublinear_tf else float(count)
        acc[idx] = acc.get(idx, 0.0) + sign * tf
    if not acc:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    idx_arr = np.fromiter(sorted(acc), dtype=np.int32, count=len(acc))
    val = np.array([acc[int(i)] for i in idx_arr], dtype=np.float64)
    norm = float(np.linalg.norm(val))
    if norm > 0.0:
        val /= norm
    return idx_arr, val.astype(np.float32)


@dataclass
class SparseRows:
    """CSR-style batch of sparse feature rows."""

    indptr: np.ndarray  # int64, len n+1
    indices: np.ndarray  # int32
    data: np.ndarray  # float32
    dim: int

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[np.ndarray, np.ndarray]], dim: int) -> SparseRows:
        lens = np.array([len(i) for i, _ in rows], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        if len(rows) and indptr[-1] > 0:
            indices = np.concatenate([i for i, _ in rows]).astype(np.int32)
            data = np.concatenate([v for _, v in rows]).astype(np.float32)
        else:
            indices, data = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        return cls(indptr, indices, data, dim)

    @classmethod
    def from_texts(cls, texts: Sequence[str], spec: FeatureSpec) -> SparseRows:
        return cls.from_rows([featurize(t, spec) for t in texts], spec.dim)

    def row_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_rows), np.diff(self.indptr))

    def take(self, rows: np.ndarray) -> SparseRows:
        """Sub-batch with the given row positions (order kept)."""
        parts = []
        for r in rows:
            lo, hi = self.indptr[r], self.indptr[r + 1]
            parts.append((self.indices[lo:hi], self.data[lo:hi]))
        return SparseRows.from_rows(parts, self.dim)

    @classmethod
    def concat(cls, a: SparseRows, b: SparseRows) -> SparseRows:
        """Rows of ``a`` followed by rows of ``b``."""
        if a.dim != b.dim:
            raise ValueError("cannot concatenate batches with different dim")
        return cls(
            np.concatenate([a.indptr, b.indptr[1:] + a.indptr[-1]]),
            np.concatenate([a.indices, b.indices]),
            np.concatenate([a.data, b.data]),
            a.dim,
        )

    def dot(self, w: np.ndarray) -> np.ndarray:
        """``X @ w`` for every row."""
        contrib = self.data * w[self.indices]
        return np.bincount(self.row_ids(), weights=contrib, minlength=self.n_rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_L2 = 1e-2  # chosen on TRAIN-only dev folds, see `safegate stage2-report --dev-grid`


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, clipped so ``exp`` cannot overflow."""
    out: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return out


def sample_weights(y: np.ndarray, balanced: bool) -> np.ndarray:
    """Per-sample loss weights; ``balanced`` gives both classes the same total weight."""
    if not balanced:
        return np.ones(len(y), dtype=np.float64)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=np.float64)
    return np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))


def fit_logreg(
    x: SparseRows,
    y: np.ndarray,
    *,
    balanced: bool = True,
    l2: float = DEFAULT_L2,
    epochs: int = 300,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Weighted L2 logistic regression by full-batch Adam; returns ``(weights, bias)``.

    The loss is ``sum_i s_i * logloss_i / sum_i s_i + l2/2 * |w|^2``.
    """
    n = x.n_rows
    if n == 0:
        raise ValueError("cannot fit on zero samples")
    yf = y.astype(np.float64)
    s = sample_weights(y, balanced)
    s = s / s.sum()
    rows = x.row_ids()
    w = np.zeros(x.dim, dtype=np.float64)
    b = 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b = v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        z = np.bincount(rows, weights=x.data * w[x.indices], minlength=n) + b
        err = s * (sigmoid(z) - yf)
        g_w = np.bincount(x.indices, weights=x.data * err[rows], minlength=x.dim) + l2 * w
        g_b = float(err.sum())
        m_w = beta1 * m_w + (1 - beta1) * g_w
        v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
        m_b = beta1 * m_b + (1 - beta1) * g_b
        v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
        c1, c2 = 1 - beta1**t, 1 - beta2**t
        w -= lr * (m_w / c1) / (np.sqrt(v_w / c2) + eps)
        b -= lr * (m_b / c1) / (np.sqrt(v_b / c2) + eps)
    return w.astype(np.float32), float(b)


# ---------------------------------------------------------------------------
# Validation folds and thresholds
# ---------------------------------------------------------------------------


def stratified_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """``k`` disjoint validation index sets, each with the same class mix (deterministic)."""
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for pos, i in enumerate(idx):
            folds[pos % k].append(int(i))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def out_of_fold_scores(
    x: SparseRows, y: np.ndarray, *, k: int = 5, seed: int = 0, **fit_kwargs: Any
) -> np.ndarray:
    """Score every row with a model that never saw it: each fold is scored by a fit on the rest."""
    scores = np.zeros(x.n_rows, dtype=np.float64)
    for val in stratified_folds(y, k, seed):
        mask = np.ones(x.n_rows, dtype=bool)
        mask[val] = False
        train = np.flatnonzero(mask)
        w, b = fit_logreg(x.take(train), y[train], **fit_kwargs)
        scores[val] = sigmoid(x.take(val).dot(w) + b)
    return scores


def threshold_for_fpr(scores: np.ndarray, y: np.ndarray, max_fpr: float) -> float:
    """Lowest cut-off ``t`` (flag when ``score >= t``) with validation FPR <= ``max_fpr``.

    With ``k = floor(max_fpr * n_benign)`` false positives allowed, ``t`` sits just above the
    (k+1)-th highest benign score, so at most ``k`` benign validation prompts are flagged.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    benign = np.sort(scores[y == 0])[::-1]
    if len(benign) == 0:
        raise ValueError("need at least one benign validation sample")
    k = int(np.floor(max_fpr * len(benign) + 1e-9))
    if k >= len(benign):
        return 0.0
    return float(np.nextafter(benign[k], np.inf))


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class Thresholds:
    """``score >= refuse`` -> refuse; ``review <= score < refuse`` -> abstain (escalate)."""

    review: float
    refuse: float

    def __post_init__(self) -> None:
        if self.review > self.refuse:
            raise ValueError("review threshold must not exceed refuse threshold")

    def verdict(self, score: float) -> str:
        if score >= self.refuse:
            return "refuse"
        if score >= self.review:
            return "escalate"
        return "allow"


def select_thresholds(
    scores: np.ndarray, y: np.ndarray, *, refuse_fpr: float = 0.02, review_fpr: float = 0.10
) -> Thresholds:
    """Pick both cut-offs on validation scores (refuse: FPR <= 2%, review band: FPR <= 10%)."""
    if review_fpr < refuse_fpr:
        raise ValueError("review_fpr must be >= refuse_fpr")
    return Thresholds(
        review=threshold_for_fpr(scores, y, review_fpr),
        refuse=threshold_for_fpr(scores, y, refuse_fpr),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Stage2Model:
    """Trained stage-2 scorer plus its calibrated decision band."""

    spec: FeatureSpec
    weights: np.ndarray
    bias: float
    thresholds: Thresholds

    def score(self, text: str) -> float:
        idx, val = featurize(text, self.spec)
        return float(sigmoid(np.array(float(val @ self.weights[idx]) + self.bias)))

    def score_many(self, texts: Sequence[str]) -> np.ndarray:
        x = SparseRows.from_texts(texts, self.spec)
        return sigmoid(x.dot(self.weights.astype(np.float64)) + self.bias)

    def verdict(self, text: str) -> str:
        return self.thresholds.verdict(self.score(text))

    def explain(self, text: str, top: int = 8) -> list[tuple[str, float]]:
        """Feature strings of ``text`` ranked by their signed contribution to the logit."""
        counts = Counter(feature_terms(text, self.spec))
        signed_tf: dict[str, tuple[int, float]] = {}
        bucket_sum: dict[int, float] = {}
        for term, count in counts.items():
            idx, sign = hash_term(term, self.spec.dim)
            tf = 1.0 + float(np.log(count)) if self.spec.sublinear_tf else float(count)
            signed_tf[term] = (idx, sign * tf)
            bucket_sum[idx] = bucket_sum.get(idx, 0.0) + sign * tf
        norm = float(np.sqrt(sum(v * v for v in bucket_sum.values()))) or 1.0
        contrib = {t: v * float(self.weights[i]) / norm for t, (i, v) in signed_tf.items()}
        return sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top]

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | str, meta: dict[str, Any] | None = None) -> None:
        nz = np.flatnonzero(self.weights)
        header = {
            "spec": asdict(self.spec),
            "bias": self.bias,
            "thresholds": asdict(self.thresholds),
            "meta": meta or {},
        }
        np.savez_compressed(
            Path(path),
            indices=nz.astype(np.int32),
            values=self.weights[nz].astype(np.float32),
            header=np.array(json.dumps(header)),
        )

    @classmethod
    def load(cls, path: Path | str) -> Stage2Model:
        with np.load(Path(path), allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            spec = FeatureSpec(**header["spec"])
            weights = np.zeros(spec.dim, dtype=np.float32)
            weights[z["indices"]] = z["values"]
        return cls(spec, weights, float(header["bias"]), Thresholds(**header["thresholds"]))


def fit_stage2(
    texts: Sequence[str],
    labels: Sequence[bool],
    *,
    spec: FeatureSpec | None = None,
    calibration_mask: Sequence[bool] | None = None,
    balanced: bool = True,
    l2: float = DEFAULT_L2,
    epochs: int = 300,
    folds: int = 5,
    refuse_fpr: float = 0.02,
    review_fpr: float = 0.10,
    seed: int = 0,
) -> Stage2Model:
    """Fit on all rows; choose the decision band on out-of-fold scores of the same rows.

    ``calibration_mask`` restricts which rows' out-of-fold scores pick the thresholds (e.g. only
    the public-data rows, so that easy synthetic prompts do not make the band too lenient).
    The fold count is capped at the size of the smaller class; fewer than two rows in a class
    cannot be validated and raise ``ValueError``.
    """
    spec = spec or FeatureSpec()
    y = np.array([1 if v else 0 for v in labels], dtype=np.int64)
    k = min(folds, int(np.sum(y == 1)), int(np.sum(y == 0)))
    if k < 2:
        raise ValueError("need at least two attack and two benign rows to cut validation folds")
    x = SparseRows.from_texts(texts, spec)
    oof = out_of_fold_scores(x, y, k=k, seed=seed, balanced=balanced, l2=l2, epochs=epochs)
    cal = np.ones(len(y), dtype=bool) if calibration_mask is None else np.array(calibration_mask)
    thresholds = select_thresholds(oof[cal], y[cal], refuse_fpr=refuse_fpr, review_fpr=review_fpr)
    w, b = fit_logreg(x, y, balanced=balanced, l2=l2, epochs=epochs)
    return Stage2Model(spec, w, b, thresholds)

"""Transformer guards as an optional second stage behind the safegate pre-filter.

Protocol: ``docs/protocol-neural.md``. Two public, ungated checkpoints are supported
(:data:`MODEL_SPECS`). ``torch`` and ``transformers`` are imported only when a model is actually
run, never on import of this module, and they live in the optional ``[guard]`` extra. Every
probability is written to a JSONL cache (``data/neural_cache/``) that is committed, so scoring
replays offline: with ``offline=True`` a cache miss raises :class:`MissingScore` and the model
libraries are never touched.

The cascade (:func:`route`, :func:`cascade_flag`): safegate's gate decides what is clearly an
attack (``refuse``) and what is clearly benign (``allow`` with a detection score under the band's
lower edge); everything else goes to the transformer, whose verdict (probability >= 0.5) decides.
The lower edges come from :data:`BANDS_FILE`, derived on training data only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path("data/neural_cache")
BANDS_FILE = "data/neural_bands.json"
MAX_LENGTH = 512  # tokens; the model card pipelines truncate to the model's 512 window
FLAG_THRESHOLD = 0.5  # probability of "injection" at or above which a transformer flags
RANK_BONUS = 2.0  # a flagged row outranks every unflagged one (scores are in [0, 1])


class MissingScore(RuntimeError):
    """Offline mode, and a text has no cached probability."""


@dataclass(frozen=True)
class ModelSpec:
    """One public checkpoint at a pinned revision.

    ``head`` is ``deberta`` for the stock ``DebertaV2ForSequenceClassification`` (context pooler
    plus classifier), or ``cls-linear`` for PIGuard, whose custom class puts one linear layer on
    the first token (its ``modeling_piguard.py``); the latter is rebuilt here, so no remote code
    is executed.
    """

    key: str
    hf_id: str
    revision: str
    head: str
    files: tuple[str, ...]
    published: str  # who published which numbers for it, one line

    def short(self) -> str:
        return f"{self.hf_id}@{self.revision[:8]}"


_COMMON = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "README.md",
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "piguard": ModelSpec(
        "piguard",
        "leolee99/PIGuard",
        "dd78b24e330193a22d2293ac66922dd4f982f563",
        "cls-linear",
        _COMMON,
        "InjecGuard paper Table 7 and ACL 2025 camera-ready; third-party tables in "
        "docs/sota-review.md",
    ),
    "protectai": ModelSpec(
        "protectai",
        "protectai/deberta-v3-base-prompt-injection-v2",
        "90c9989b1a342275dd0d1a95aad283c04e075671",
        "deberta",
        (*_COMMON, "LICENSE"),
        "vendor card (accuracy 95.25 on 20,000 private prompts); InjecGuard Table 7; "
        "third-party tables in docs/sota-review.md",
    ),
}


# ---------------------------------------------------------------------------------------------
# Probability cache
# ---------------------------------------------------------------------------------------------


def text_key(text: str) -> str:
    """The cache key of a text: 32 hex characters of its sha256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class ScoreStore:
    """``text -> P(injection)`` for one model, as an append-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._p: dict[str, float] = {}
        self._pending: list[str] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._p[str(rec["k"])] = float(rec["p"])

    def __len__(self) -> int:
        return len(self._p)

    def get(self, text: str) -> float | None:
        return self._p.get(text_key(text))

    def put(self, text: str, p: float) -> None:
        key = text_key(text)
        if key not in self._p:
            self._p[key] = round(p, 6)
            self._pending.append(json.dumps({"k": key, "p": self._p[key]}))

    def flush(self) -> None:
        if not self._pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(self._pending) + "\n")
        self._pending.clear()


# ---------------------------------------------------------------------------------------------
# The model (torch and transformers imported lazily)
# ---------------------------------------------------------------------------------------------


def pick_device(prefer: str | None = None) -> str:
    """``prefer`` if given; else Apple MPS when available, else CPU."""
    if prefer:
        return prefer
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class TransformerGuard:
    """A pinned checkpoint scoring texts, with its probabilities cached.

    ``offline=True`` reads the cache only. The cache lives in ``cache_dir / <key>.jsonl`` and is
    keyed by text; ``max_length`` is part of the protocol, not of the key.
    """

    def __init__(
        self,
        spec: ModelSpec,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        offline: bool = False,
        device: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.spec = spec
        self.store = ScoreStore(Path(cache_dir) / f"{spec.key}.jsonl")
        self.offline = offline
        self._device_pref = device
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self._model: Any = None
        self._tok: Any = None
        self.device = device or "cache"

    # -- loading ------------------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, DebertaV2TokenizerFast  # type: ignore[attr-defined]

        path = snapshot_download(
            self.spec.hf_id, revision=self.spec.revision, allow_patterns=list(self.spec.files)
        )
        self.device = pick_device(self._device_pref)
        tok = DebertaV2TokenizerFast.from_pretrained(path)
        if self.spec.head == "deberta":
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = _load_cls_linear(path)
        model.eval()
        self._model = model.to(self.device)
        self._tok = tok
        torch.set_grad_enabled(False)
        self._log(f"loaded {self.spec.short()} on {self.device}")

    def versions(self) -> dict[str, str]:
        """Library and device versions of the run (for the cache's meta file)."""
        import torch
        import transformers

        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": self.device,
        }

    def _logits(self, texts: Sequence[str]) -> Any:
        import torch

        enc = self._tok(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        out = self._model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return torch.softmax(out.logits.float(), dim=-1)[:, 1]

    # -- scoring ------------------------------------------------------------------------------

    def scores(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        """P(injection) per text; computes and caches what is missing unless offline."""
        missing = [t for t in dict.fromkeys(texts) if self.store.get(t) is None]
        if missing:
            if self.offline:
                raise MissingScore(
                    f"{len(missing)} text(s) have no cached probability for {self.spec.key}; "
                    "run without --offline (needs the [guard] extra)"
                )
            self._compute(missing, batch_size)
        out: list[float] = []
        for t in texts:
            p = self.store.get(t)
            assert p is not None
            out.append(p)
        return out

    def _compute(self, texts: list[str], batch_size: int) -> None:
        self._load()
        order = sorted(texts, key=len, reverse=True)  # long first: padding waste stays small
        start = time.perf_counter()
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            probs = self._logits(batch).cpu().tolist()
            for text, p in zip(batch, probs, strict=True):
                self.store.put(text, float(p))
            if (i // batch_size) % 50 == 0:
                self.store.flush()
                done = i + len(batch)
                rate = done / max(time.perf_counter() - start, 1e-9)
                self._log(f"  {self.spec.key}: {done}/{len(order)} ({rate:.0f} texts/s)")
        self.store.flush()

    def time_single(self, texts: Sequence[str], warmup: int = 5) -> list[float]:
        """Seconds per text at batch size 1 (tokenise, forward, softmax), not cached."""
        if self.offline:
            raise MissingScore("latency needs the model; run without --offline")
        self._load()
        for t in list(texts)[:warmup]:
            self._logits([t]).cpu()
        out: list[float] = []
        for t in texts:
            start = time.perf_counter()
            self._logits([t]).cpu().tolist()
            out.append(time.perf_counter() - start)
        return out

    def write_meta(self, rows_scored: int) -> None:
        """Provenance next to the cache: model id, revision, window, library versions."""
        meta = {
            "model": self.spec.hf_id,
            "revision": self.spec.revision,
            "head": self.spec.head,
            "max_length": MAX_LENGTH,
            "flag_threshold": FLAG_THRESHOLD,
            "texts_cached": len(self.store),
            "texts_scored_in_last_run": rows_scored,
            **self.versions(),
        }
        path = self.store.path.with_suffix(".meta.json")
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _load_cls_linear(path: str) -> Any:
    """PIGuard: ``DebertaV2Model`` plus one linear layer on the first token (no pooler)."""
    import torch
    from safetensors.torch import load_file
    from transformers import DebertaV2Config, DebertaV2Model

    raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    for drop in ("model_type", "auto_map", "architectures", "transformers_version", "torch_dtype"):
        raw.pop(drop, None)
    config = DebertaV2Config(**raw)

    class _PIGuard(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.deberta = DebertaV2Model(config)  # type: ignore[no-untyped-call]
            self.classifier = torch.nn.Linear(config.hidden_size, int(config.num_labels or 2))

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden = hidden.last_hidden_state[:, 0, :]
            return type("Out", (), {"logits": self.classifier(hidden)})()

    model = _PIGuard()
    state = load_file(str(Path(path) / "model.safetensors"))
    result = model.load_state_dict(state, strict=False)
    # the checkpoint also carries the stock context pooler, which PIGuard's own class never uses
    unexpected = [k for k in result.unexpected_keys if not k.startswith("pooler.")]
    missing = [k for k in result.missing_keys if "position_ids" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"PIGuard weights do not fit: missing {missing}, unexpected {unexpected}"
        )
    return model


# ---------------------------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """A pre-registered escalation band.

    ``budget`` is a dev share chosen in advance. In the ``budget`` family (``narrow``, ``mid``,
    ``wide``) it is the share of dev benign rows allowed at or above ``lo`` (including rows the
    gate refuses). In the ``target`` family (``t35`` ... ``t70``) it is the share of dev benign rows
    sent to the transformer, that is, at or above ``lo`` and below the refuse threshold. ``lo`` is
    the stage-2 score at which a gate-allowed row is escalated.
    """

    name: str
    budget: float
    lo: float


BAND_FAMILIES = {"budget": ("bands", "budget"), "target": ("target_bands", "target")}


def load_bands_file() -> dict[str, Any]:
    """The whole of ``neural_bands.json``: the bands and the dev shares they were derived with."""
    path = Path(str(resources.files("safegate").joinpath(BANDS_FILE)))
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_bands(family: str = "budget") -> dict[str, Band]:
    """The bands of ``src/safegate/data/neural_bands.json`` (``neural-eval --derive-bands``).

    ``family`` is ``budget`` (narrow, mid, wide), ``target`` (the escalation-rate family of
    ``docs/protocol-neural.md``, "Escalation-rate bands") or ``all`` (both, budget first).
    """
    data = load_bands_file()
    families = list(BAND_FAMILIES) if family == "all" else [family]
    out: dict[str, Band] = {}
    for fam in families:
        section, field = BAND_FAMILIES[fam]
        out.update({k: Band(k, float(v[field]), float(v["lo"])) for k, v in data[section].items()})
    return out


ROUTE_FLAG = "flag"
ROUTE_PASS = "pass"
ROUTE_ESCALATE = "escalate"


def route(action: str, detection_score: float, band: Band) -> str:
    """Where the pre-filter sends a row: flag (clear attack), pass (clear benign) or escalate.

    ``refuse`` is a clear attack (a stage-1 regex hit, a stage-2 score at or over the refuse
    threshold, or a content rule). ``escalate`` (the gate's own abstain band) always goes to the
    transformer, as does any non-blocking row whose detection score reaches the band's lower edge.
    """
    if action == "refuse":
        return ROUTE_FLAG
    if action == "escalate" or detection_score >= band.lo:
        return ROUTE_ESCALATE
    return ROUTE_PASS


def cascade_flag(routed: str, p_injection: float | None) -> bool:
    """The cascade's verdict: the gate's for flag and pass, the transformer's for escalate."""
    if routed == ROUTE_ESCALATE:
        assert p_injection is not None
        return p_injection >= FLAG_THRESHOLD
    return routed == ROUTE_FLAG


def cascade_score(routed: str, detection_score: float, p_injection: float | None) -> float:
    """Ranking score for AUC and TPR at FPR: flagged above unflagged, each by its own score."""
    flagged = cascade_flag(routed, p_injection)
    base = p_injection if routed == ROUTE_ESCALATE and p_injection is not None else detection_score
    return base + (RANK_BONUS if flagged else 0.0)

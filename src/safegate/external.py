"""Evaluation on public labelled prompt-injection / jailbreak datasets.

The detector and gate are never fitted on this data unless a retrained variant is requested
explicitly, and that variant is trained on the dataset's TRAIN split only and scored on TEST.

Datasets are downloaded over plain HTTPS with the standard library and cached as JSONL under
``data/raw/`` (gitignored). Third-party data is never committed.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/datasets"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_RESOLVE = "https://huggingface.co/datasets"
ROWS_PAGE = 100

Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class DatasetSpec:
    """A public dataset: where it comes from, which revision was evaluated, its licence."""

    key: str
    hf_id: str
    revision: str  # HF commit sha the README numbers were produced with
    license: str
    kind: str  # "rows-api" (datasets-server JSON) or "csv" (raw files at the pinned revision)
    description: str
    csv_files: dict[str, str] = field(default_factory=dict)  # split -> path inside the repo

    @property
    def card_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.hf_id}"


DATASETS: dict[str, DatasetSpec] = {
    "deepset": DatasetSpec(
        key="deepset",
        hf_id="deepset/prompt-injections",
        revision="4f61ecb038e9c3fb77e21034b22511b523772cdd",
        license="apache-2.0 (card front-matter; nested dataset_info also says cc-by-4.0)",
        kind="rows-api",
        description="prompt injections (label 1) vs benign prompts (label 0); English + German",
    ),
    "jailbreak": DatasetSpec(
        key="jailbreak",
        hf_id="jackhhao/jailbreak-classification",
        revision="2f2ceeb39658696fd3f462403562b6eea5306287",
        license="apache-2.0",
        kind="csv",
        description="jailbreak prompts vs benign prompts (balanced config of the card)",
        csv_files={
            "train": "balanced/jailbreak_dataset_train_balanced.csv",
            "test": "balanced/jailbreak_dataset_test_balanced.csv",
        },
    ),
}


@dataclass(frozen=True)
class ExternalSample:
    """One labelled prompt from an external dataset."""

    text: str
    is_attack: bool
    split: str


@dataclass(frozen=True)
class LoadedDataset:
    """Samples plus the provenance recorded when they were downloaded."""

    spec: DatasetSpec
    samples: list[ExternalSample]
    meta: dict[str, Any]

    def split(self, name: str) -> list[ExternalSample]:
        return [s for s in self.samples if s.split == name]


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------


def http_get(url: str, attempts: int = 4) -> bytes:
    """GET a URL with the standard library; retries 5xx/network errors (datasets-server flakes)."""
    req = urllib.request.Request(url, headers={"User-Agent": "safegate-external-eval"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data: bytes = resp.read()
            return data
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


def parse_rows_page(payload: dict[str, Any], split: str) -> tuple[list[ExternalSample], int]:
    """Parse one datasets-server ``/rows`` page (label 1 = attack) and the split's row total."""
    out = [
        ExternalSample(
            text=str(item["row"]["text"]),
            is_attack=int(item["row"]["label"]) == 1,
            split=split,
        )
        for item in payload["rows"]
    ]
    return out, int(payload["num_rows_total"])


def parse_jailbreak_csv(raw: str, split: str) -> list[ExternalSample]:
    """Parse the ``prompt,type`` CSV format (type ``jailbreak`` = attack, ``benign`` = benign)."""
    out: list[ExternalSample] = []
    for row in csv.DictReader(io.StringIO(raw)):
        kind = row["type"].strip().lower()
        if kind not in ("jailbreak", "benign"):
            raise ValueError(f"unexpected label {row['type']!r}")
        out.append(ExternalSample(text=row["prompt"], is_attack=kind == "jailbreak", split=split))
    return out


def _fetch_rows_api(spec: DatasetSpec, fetch: Fetcher) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for split in ("train", "test"):
        offset, total = 0, 1
        while offset < total:
            query = urllib.parse.urlencode(
                {
                    "dataset": spec.hf_id,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": ROWS_PAGE,
                }
            )
            page = json.loads(fetch(f"{HF_ROWS_API}?{query}"))
            rows, total = parse_rows_page(page, split)
            if not rows:
                break
            samples.extend(rows)
            offset += len(rows)
        if offset != total:
            raise RuntimeError(f"{spec.hf_id}/{split}: got {offset} of {total} rows")
    return samples


def _fetch_csv(spec: DatasetSpec, fetch: Fetcher) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for split, path in spec.csv_files.items():
        url = f"{HF_RESOLVE}/{spec.hf_id}/resolve/{spec.revision}/{path}"
        samples.extend(parse_jailbreak_csv(fetch(url).decode("utf-8"), split))
    return samples


def _write_cache(cache: Path, samples: list[ExternalSample], meta: dict[str, Any]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for split in dict.fromkeys(s.split for s in samples):
        with open(cache / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                if s.split == split:
                    f.write(json.dumps({"text": s.text, "is_attack": s.is_attack}) + "\n")
    (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_cache(cache: Path) -> tuple[list[ExternalSample], dict[str, Any]] | None:
    meta_path = cache / "meta.json"
    if not meta_path.exists():
        return None
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    samples: list[ExternalSample] = []
    for split in meta["splits"]:
        with open(cache / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                samples.append(ExternalSample(rec["text"], bool(rec["is_attack"]), split))
    return samples, meta


def load_dataset(
    spec: DatasetSpec,
    cache_dir: Path | str = "data/raw",
    fetch: Fetcher = http_get,
    refresh: bool = False,
) -> LoadedDataset:
    """Return the dataset from the cache, downloading it first when absent."""
    cache = Path(cache_dir) / spec.key
    cached = None if refresh else _read_cache(cache)
    if cached is not None:
        return LoadedDataset(spec, cached[0], cached[1])

    observed = str(json.loads(fetch(f"{HF_API}/{spec.hf_id}"))["sha"])
    if spec.kind == "rows-api":
        samples = _fetch_rows_api(spec, fetch)
    elif spec.kind == "csv":
        samples = _fetch_csv(spec, fetch)
    else:
        raise ValueError(f"unknown dataset kind {spec.kind!r}")
    if spec.kind == "rows-api" and observed != spec.revision:
        print(
            f"warning: {spec.hf_id} is now at revision {observed}; the numbers in the README "
            f"were produced at {spec.revision} and may not reproduce.",
            file=sys.stderr,
        )
    meta = {
        "hf_id": spec.hf_id,
        "pinned_revision": spec.revision,
        "observed_revision": observed,
        "license": spec.license,
        "card_url": spec.card_url,
        "downloaded": date.today().isoformat(),
        "splits": list(dict.fromkeys(s.split for s in samples)),
    }
    _write_cache(cache, samples, meta)
    return LoadedDataset(spec, samples, meta)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Rate:
    """k hits out of n with its Wilson 95% interval."""

    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.k, self.n)

    def fmt(self) -> str:
        lo, hi = self.ci
        return f"{self.value:.1%} ({self.k}/{self.n}) [{lo:.1%}, {hi:.1%}]"


_DE_WORDS = frozenset(
    {
        *("der", "die", "das", "und", "ist", "nicht", "ein", "eine", "einen", "mit", "für"),
        *("auf", "wie", "was", "wer", "wann", "warum", "ich", "du", "sie", "wir", "ihr", "es"),
        *("zu", "von", "im", "den", "dem", "des", "dass", "aber", "oder", "auch", "wenn"),
        *("bitte", "kann", "können", "sich", "über", "nach", "bei", "aus", "noch", "nur"),
        *("wird", "sind", "war", "haben", "hat", "sein", "mir", "mich", "dir", "dich", "uns"),
    }
)
_EN_WORDS = frozenset(
    {
        *("the", "and", "is", "are", "not", "a", "an", "with", "for", "on", "how", "what"),
        *("who", "when", "why", "i", "you", "he", "she", "we", "they", "it", "to", "of", "in"),
        *("that", "this", "but", "or", "also", "if", "please", "can", "could", "be", "was"),
        *("were", "have", "has", "my", "me", "your", "our", "as", "at", "by", "from", "do"),
        *("does", "will", "would", "should", "about"),
    }
)


def guess_language(text: str) -> str:
    """Stop-word heuristic: ``de`` / ``en`` / ``other``. Unreliable on very short texts."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    de = sum(w in _DE_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if re.search(r"[äöüß]", text.lower()):
        de += 1
    if de == 0 and en == 0:
        return "other"
    return "de" if de > en else "en"


@dataclass
class Outcome:
    """Per-sample block/pass decisions of one system on one set of samples."""

    name: str
    results: list[tuple[ExternalSample, bool]]  # (sample, system blocked it)
    causes: dict[str, str] = field(default_factory=dict)  # prompt text -> why it was blocked

    def _select(self, attack: bool, lang: str | None) -> list[tuple[ExternalSample, bool]]:
        return [
            (s, blocked)
            for s, blocked in self.results
            if s.is_attack == attack and (lang is None or guess_language(s.text) == lang)
        ]

    def detection(self, lang: str | None = None) -> Rate:
        rows = self._select(True, lang)
        return Rate(sum(b for _, b in rows), len(rows))

    def fpr(self, lang: str | None = None) -> Rate:
        rows = self._select(False, lang)
        return Rate(sum(b for _, b in rows), len(rows))

    def false_negatives(self) -> list[ExternalSample]:
        return [s for s, blocked in self.results if s.is_attack and not blocked]

    def false_positives(self) -> list[ExternalSample]:
        return [s for s, blocked in self.results if not s.is_attack and blocked]


def score(name: str, samples: Sequence[ExternalSample], blocks: Callable[[str], bool]) -> Outcome:
    """Run ``blocks`` on every sample and record the decisions."""
    return Outcome(name, [(s, blocks(s.text)) for s in samples])


def shorten(text: str, limit: int = 160) -> str:
    """One-line, length-capped rendering of a prompt for error analysis."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"

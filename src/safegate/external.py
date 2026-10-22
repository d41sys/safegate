"""Evaluation on public labelled prompt-injection / jailbreak datasets.

The detector and gate are never fitted on this data unless a retrained variant is requested
explicitly, and that variant is trained on the dataset's TRAIN split only and scored on TEST.

Datasets are downloaded over plain HTTPS with the standard library and cached as JSONL under
``data/raw/`` (gitignored). Third-party data is never committed.
"""

from __future__ import annotations

import csv
import hashlib
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
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/datasets"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_RESOLVE = "https://huggingface.co/datasets"
ROWS_PAGE = 100
ROWS_PAUSE = 0.3  # seconds between datasets-server pages (it rate-limits bursts with 429)
HASH_TEST_MOD = 5  # datasets without an official split: sha1(text) % 5 == 0 goes to TEST (20%)

Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class RowsSource:
    """One (config, split) slice of a datasets-server dataset, with its label rule.

    ``label_field`` names a 0/1 or bool column; when it is None every row of the slice gets
    ``attack``. ``assign_split`` is the TRAIN/TEST split the rows land in; None means "decide by
    :func:`hash_split`" (datasets that ship no usable split). Rows whose ``exclude_field`` value is
    in ``exclude_values`` are dropped (licence or provenance filter).
    """

    config: str
    split: str
    text_field: str
    label_field: str | None = None
    attack: bool = False
    assign_split: str | None = None
    exclude_field: str | None = None
    exclude_values: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DatasetSpec:
    """A public dataset: where it comes from, which revision was evaluated, its licence."""

    key: str
    hf_id: str
    revision: str  # HF commit sha the README numbers were produced with
    license: str
    kind: str  # "rows-api", "rows-multi" (datasets-server, several sources) or "csv"
    description: str
    csv_files: dict[str, str] = field(default_factory=dict)  # split -> path inside the repo
    sources: tuple[RowsSource, ...] = ()  # "rows-multi" only
    max_benign_per_attack: float | None = None  # "rows-multi": deterministic benign subsample

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
    "itw": DatasetSpec(
        key="itw",
        hf_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        revision="a10aab8eff1c73165a442d4464dce192bd28b9c5",
        license="mit",
        kind="rows-multi",
        description=(
            "jailbreak prompts collected in the wild (config jailbreak_2023_12_25) vs regular "
            "prompts from the same platforms (regular_2023_12_25); no official split, so TEST = "
            "sha1(text) % 5 == 0; benign deterministically subsampled to 2 per attack"
        ),
        sources=(
            RowsSource(
                "jailbreak_2023_12_25", "train", "prompt", label_field="jailbreak", attack=True
            ),
            RowsSource("regular_2023_12_25", "train", "prompt", label_field="jailbreak"),
        ),
        max_benign_per_attack=2.0,
    ),
    "neuralchemy": DatasetSpec(
        key="neuralchemy",
        hf_id="neuralchemy/Prompt-injection-dataset",
        revision="7d70432dfcf47a821612cbf9d34e9d9e3ad20e75",
        license=(
            "apache-2.0 on the card; its own source table marks WildGuard / JudgeComparison as "
            "'Research', so rows with source=wildguard_judgecomp are dropped; remaining sources "
            "are Apache 2.0, CC BY 4.0 (HackAPrompt) or MIT (HarmBench)"
        ),
        kind="rows-multi",
        description=(
            "config `core` (original samples, no augmentation): prompt injection / jailbreak "
            "categories vs benign incl. hard negatives; official train and test splits"
        ),
        sources=tuple(
            RowsSource(
                "core",
                split,
                "text",
                label_field="label",
                assign_split=split,
                exclude_field="source",
                exclude_values=frozenset({"wildguard_judgecomp"}),
            )
            for split in ("train", "test")
        ),
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


def http_get(url: str, attempts: int = 8) -> bytes:
    """GET a URL with the standard library; retries 429/5xx/network errors with backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "safegate-external-eval"})
    for attempt in range(1, attempts + 1):
        delay = min(2.0 * 2 ** (attempt - 1), 30.0)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data: bytes = resp.read()
            return data
        except urllib.error.HTTPError as err:
            if (err.code < 500 and err.code != 429) or attempt == attempts:
                raise
            retry_after = err.headers.get("Retry-After", "") if err.headers else ""
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(delay)
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


def _fetch_pages(hf_id: str, config: str, split: str, fetch: Fetcher) -> list[dict[str, Any]]:
    """All datasets-server ``/rows`` pages of one (config, split); raises when rows are missing."""
    pages: list[dict[str, Any]] = []
    offset, total = 0, 1
    while offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": hf_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": ROWS_PAGE,
            }
        )
        page: dict[str, Any] = json.loads(fetch(f"{HF_ROWS_API}?{query}"))
        total = int(page["num_rows_total"])
        if not page["rows"]:
            break
        pages.append(page)
        offset += len(page["rows"])
        time.sleep(ROWS_PAUSE)
    if offset != total:
        raise RuntimeError(f"{hf_id}/{config}/{split}: got {offset} of {total} rows")
    return pages


def _fetch_rows_api(spec: DatasetSpec, fetch: Fetcher) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for split in ("train", "test"):
        for page in _fetch_pages(spec.hf_id, "default", split, fetch):
            samples.extend(parse_rows_page(page, split)[0])
    return samples


def stable_hash(text: str, salt: str) -> int:
    """Deterministic 64-bit hash (sha1), independent of PYTHONHASHSEED."""
    return int.from_bytes(hashlib.sha1(f"{salt}:{text}".encode()).digest()[:8], "big")


def hash_split(text: str) -> str:
    """TEST for 1 in :data:`HASH_TEST_MOD` prompts, chosen by a hash of the text alone."""
    return "test" if stable_hash(text, "split") % HASH_TEST_MOD == 0 else "train"


def parse_source_page(payload: dict[str, Any], source: RowsSource) -> list[ExternalSample]:
    """Parse one ``/rows`` page of a :class:`RowsSource`; split is "" when it must be hashed."""
    out: list[ExternalSample] = []
    for item in payload["rows"]:
        row = item["row"]
        if source.exclude_field and str(row[source.exclude_field]) in source.exclude_values:
            continue
        attack = source.attack if source.label_field is None else bool(int(row[source.label_field]))
        out.append(ExternalSample(str(row[source.text_field]), attack, source.assign_split or ""))
    return out


def assemble_samples(
    rows: Sequence[ExternalSample], max_benign_per_attack: float | None = None
) -> list[ExternalSample]:
    """Clean raw rows into a dataset with disjoint TRAIN and TEST splits.

    Drops empty texts, repeated texts (first occurrence wins), texts that carry both labels, and
    TRAIN rows whose text is also in TEST. With ``max_benign_per_attack`` the benign rows are cut
    to that many per attack, keeping the ones with the smallest ``stable_hash`` (deterministic).
    Rows without a split (``""``) are then assigned by :func:`hash_split`.
    """
    labels: dict[str, set[bool]] = {}
    for r in rows:
        labels.setdefault(r.text.strip(), set()).add(r.is_attack)
    test_texts = {r.text.strip() for r in rows if r.split == "test"}
    seen: set[str] = set()
    kept: list[ExternalSample] = []
    for r in rows:
        key = r.text.strip()
        if not key or key in seen or len(labels[key]) > 1:
            continue
        if r.split == "train" and key in test_texts:
            continue
        seen.add(key)
        kept.append(r)
    if max_benign_per_attack is not None:
        budget = int(max_benign_per_attack * sum(r.is_attack for r in kept))
        benign = sorted(
            (r for r in kept if not r.is_attack), key=lambda r: stable_hash(r.text, "benign-cap")
        )
        allowed = {r.text for r in benign[:budget]}
        kept = [r for r in kept if r.is_attack or r.text in allowed]
    return [r if r.split else replace(r, split=hash_split(r.text)) for r in kept]


def _fetch_rows_multi(spec: DatasetSpec, fetch: Fetcher) -> list[ExternalSample]:
    raw: list[ExternalSample] = []
    for source in spec.sources:
        for page in _fetch_pages(spec.hf_id, source.config, source.split, fetch):
            raw.extend(parse_source_page(page, source))
    return assemble_samples(raw, spec.max_benign_per_attack)


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
    elif spec.kind == "rows-multi":
        samples = _fetch_rows_multi(spec, fetch)
    elif spec.kind == "csv":
        samples = _fetch_csv(spec, fetch)
    else:
        raise ValueError(f"unknown dataset kind {spec.kind!r}")
    if spec.kind in ("rows-api", "rows-multi") and observed != spec.revision:
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

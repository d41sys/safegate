"""Fresh public evaluation data for ``docs/protocol-window-150.md`` and ``safegate benchmark``.

Six sources the rest of the repository never read: SPML (attacks and benign), Gandalf
"ignore instructions" (attacks), the BIPIA test prompts and the WildGuard benign prompts that ship
in the InjecGuard repository, a Dolly sample (long benign texts) and NotInject (over-defense). Each
one is downloaded at a pinned revision, cached as JSONL under ``data/raw/<key>/`` (gitignored, never
committed) and cleaned with one rule: a text is dropped when it is empty, repeats an earlier text,
occurs in any split of the four datasets of :data:`safegate.external.DATASETS` or in the bundled
training set, or carries both labels. Nothing here scores a gate.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from safegate.external import (
    DATASETS,
    HF_API,
    HF_RESOLVE,
    Fetcher,
    fetch_pages,
    http_get,
    load_dataset,
    stable_hash,
)

INJECGUARD_COMMIT = "0bffa314cde90a5c078c24b3302d7fbb4bd731fa"  # the only commit of its datasets/
INJECGUARD_RAW = (
    f"https://raw.githubusercontent.com/leolee99/InjecGuard/{INJECGUARD_COMMIT}/datasets"
)

# Sampling caps of the adaptive pool (protocol A.1): the first `cap` rows in sha1 order of the text.
POOL_CAPS: dict[tuple[str, bool], int] = {
    ("spml", True): 400,
    ("gandalf", True): 400,
    ("spml", False): 500,
    ("dolly", False): 600,
}


@dataclass(frozen=True)
class FreshRow:
    """One cleaned text: its source key, its subset inside the source and its label."""

    source: str
    group: str
    text: str
    is_attack: bool


@dataclass(frozen=True)
class FreshSource:
    """Where a source comes from and which revision the numbers were produced with."""

    key: str
    origin: str  # card or repository URL
    revision: str
    license: str
    description: str
    hf_id: str | None = None  # set for HF datasets (the revision is then checked at download)


SOURCES: tuple[FreshSource, ...] = (
    FreshSource(
        key="spml",
        origin="https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection",
        revision="02ce8084e979bc7d4c24ee35d22ecb7f2db96ff5",
        license="mit",
        description="user prompts of the SPML chatbot set: injection (1) and benign (0)",
        hf_id="reshabhs/SPML_Chatbot_Prompt_Injection",
    ),
    FreshSource(
        key="gandalf",
        origin="https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        revision="04737b65e90a6794ec227012e4a255a7def6344b",
        license="mit",
        description="Gandalf prompts close to 'ignore all previous instructions' (train, "
        "validation and test pooled); attacks only",
        hf_id="Lakera/gandalf_ignore_instructions",
    ),
    FreshSource(
        key="bipia",
        origin="https://github.com/leolee99/InjecGuard/tree/" + INJECGUARD_COMMIT + "/datasets",
        revision=INJECGUARD_COMMIT,
        license="mit (repository); upstream microsoft/BIPIA is MIT",
        description="BIPIA_text.json (75) and BIPIA_code.json (50): the BIPIA test prompts of the "
        "InjecGuard paper; attacks only",
    ),
    FreshSource(
        key="wildguard",
        origin="https://github.com/leolee99/InjecGuard/tree/" + INJECGUARD_COMMIT + "/datasets",
        revision=INJECGUARD_COMMIT,
        license="odc-by (allenai/wildguardmix card; this copy ships inside an MIT repository)",
        description="wildguard.json: 971 benign prompts of WildGuardTest",
    ),
    FreshSource(
        key="dolly",
        origin="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        license="cc-by-sa-3.0",
        description="instruction, plus the context when there is one; long benign texts",
        hf_id="databricks/databricks-dolly-15k",
    ),
    FreshSource(
        key="notinject",
        origin="https://huggingface.co/datasets/leolee99/NotInject",
        revision="847ae76cf8fea5ed325429e569ae8cfef022d2e0",
        license="mit",
        description="339 benign prompts with 1, 2 or 3 injection trigger words (over-defense)",
        hf_id="leolee99/NotInject",
    ),
)
SOURCE_BY_KEY = {s.key: s for s in SOURCES}
NOTINJECT_SPLITS = ("NotInject_one", "NotInject_two", "NotInject_three")
BIPIA_FILES = (("BIPIA_text", "BIPIA_text.json"), ("BIPIA_code", "BIPIA_code.json"))


def match_key(text: str) -> str:
    """The normalised text two rows are compared by (whitespace and case folded)."""
    return " ".join(text.split()).casefold()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _spml(fetch: Fetcher) -> list[FreshRow]:
    src = SOURCE_BY_KEY["spml"]
    url = f"{HF_RESOLVE}/{src.hf_id}/resolve/{src.revision}/spml_prompt_injection.csv"
    csv.field_size_limit(sys.maxsize)
    reader = csv.DictReader(io.StringIO(fetch(url).decode("utf-8")))
    return [
        FreshRow("spml", "spml", row["User Prompt"], row["Prompt injection"].strip() == "1")
        for row in reader
    ]


def _rows_api(
    hf_id: str, splits: Iterable[str], field_name: str, fetch: Fetcher
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for split in splits:
        for page in fetch_pages(hf_id, "default", split, fetch):
            out.extend((split, str(item["row"][field_name])) for item in page["rows"])
    return out


def _gandalf(fetch: Fetcher) -> list[FreshRow]:
    pairs = _rows_api(
        "Lakera/gandalf_ignore_instructions", ("train", "validation", "test"), "text", fetch
    )
    return [FreshRow("gandalf", "gandalf", text, True) for _, text in pairs]


def _bipia(fetch: Fetcher) -> list[FreshRow]:
    out: list[FreshRow] = []
    for group, name in BIPIA_FILES:
        data: dict[str, list[str]] = json.loads(fetch(f"{INJECGUARD_RAW}/{name}"))
        out.extend(
            FreshRow("bipia", group, text, True) for texts in data.values() for text in texts
        )
    return out


def _wildguard(fetch: Fetcher) -> list[FreshRow]:
    data: list[dict[str, Any]] = json.loads(fetch(f"{INJECGUARD_RAW}/wildguard.json"))
    return [
        FreshRow("wildguard", "wildguard", str(r["prompt"]), int(r["label"]) == 1) for r in data
    ]


def _dolly(fetch: Fetcher) -> list[FreshRow]:
    src = SOURCE_BY_KEY["dolly"]
    url = f"{HF_RESOLVE}/{src.hf_id}/resolve/{src.revision}/databricks-dolly-15k.jsonl"
    out: list[FreshRow] = []
    for line in fetch(url).decode("utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            context = str(rec.get("context", "")).strip()
            text = str(rec["instruction"]) + (f"\n\n{context}" if context else "")
            out.append(FreshRow("dolly", "dolly", text, False))
    return out


def _notinject(fetch: Fetcher) -> list[FreshRow]:
    pairs = _rows_api("leolee99/NotInject", NOTINJECT_SPLITS, "prompt", fetch)
    return [FreshRow("notinject", split, text, False) for split, text in pairs]


_DOWNLOADERS: dict[str, Callable[[Fetcher], list[FreshRow]]] = {
    "spml": _spml,
    "gandalf": _gandalf,
    "bipia": _bipia,
    "wildguard": _wildguard,
    "dolly": _dolly,
    "notinject": _notinject,
}


def _write_cache(cache: Path, rows: Sequence[FreshRow], meta: dict[str, Any]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"group": r.group, "text": r.text, "is_attack": r.is_attack}) + "\n")
    (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_cache(cache: Path, key: str) -> list[FreshRow] | None:
    if not (cache / "meta.json").exists():
        return None
    with open(cache / "rows.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return [FreshRow(key, r["group"], r["text"], bool(r["is_attack"])) for r in recs]


def load_source(
    key: str, cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get, refresh: bool = False
) -> list[FreshRow]:
    """The raw rows of one source: from the cache, downloading the pinned revision when absent."""
    src = SOURCE_BY_KEY[key]
    cache = Path(cache_dir) / key
    cached = None if refresh else _read_cache(cache, key)
    if cached is not None:
        return cached
    observed = src.revision
    if src.hf_id is not None:
        observed = str(json.loads(fetch(f"{HF_API}/{src.hf_id}"))["sha"])
        if observed != src.revision:
            print(
                f"warning: {src.hf_id} is now at revision {observed}; the protocol pins "
                f"{src.revision} and the rows API serves the newest, so numbers may differ.",
                file=sys.stderr,
            )
    rows = _DOWNLOADERS[key](fetch)
    meta = {
        "key": key,
        "origin": src.origin,
        "pinned_revision": src.revision,
        "observed_revision": observed,
        "license": src.license,
        "downloaded": date.today().isoformat(),
        "rows": len(rows),
    }
    _write_cache(cache, rows, meta)
    return rows


# ---------------------------------------------------------------------------
# Cleaning and pools
# ---------------------------------------------------------------------------


def bundled_texts() -> list[str]:
    """The texts of the bundled ``injection_train.jsonl`` (stage 2 was fitted on them)."""
    path = Path(str(resources.files("safegate").joinpath("data/injection_train.jsonl")))
    with open(path, encoding="utf-8") as f:
        return [str(json.loads(line)["text"]) for line in f if line.strip()]


def training_keys(cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get) -> set[str]:
    """Match keys of every text in the four datasets of the repository and the bundled set."""
    keys = {match_key(t) for t in bundled_texts()}
    for spec in DATASETS.values():
        keys.update(match_key(s.text) for s in load_dataset(spec, cache_dir, fetch).samples)
    return keys


def clean(raw: Sequence[FreshRow], exclude: set[str]) -> list[FreshRow]:
    """Drop empty, repeated, excluded and label-conflicting texts (first occurrence wins)."""
    labels: dict[str, set[bool]] = {}
    for r in raw:
        labels.setdefault(match_key(r.text), set()).add(r.is_attack)
    seen: set[str] = set()
    out: list[FreshRow] = []
    for r in raw:
        key = match_key(r.text)
        if not key or key in seen or key in exclude or len(labels[key]) > 1:
            continue
        seen.add(key)
        out.append(r)
    return out


def first_by_hash(rows: Sequence[FreshRow], cap: int) -> list[FreshRow]:
    """The first ``cap`` rows in sha1 order of the text (the order ``adaptive.dev_sample`` uses)."""
    return sorted(rows, key=lambda r: stable_hash(r.text, "dev"))[:cap]


@dataclass(frozen=True)
class Fresh:
    """All cleaned rows of every source, in source order, plus how many each cleaning step cost."""

    rows: list[FreshRow]
    dropped: dict[str, int]  # source key -> rows removed by the cleaning

    def select(
        self, source: str, attack: bool | None = None, group: str | None = None
    ) -> list[FreshRow]:
        return [
            r
            for r in self.rows
            if r.source == source
            and (attack is None or r.is_attack == attack)
            and (group is None or r.group == group)
        ]

    def pool(self, attack: bool) -> list[FreshRow]:
        """The adaptive pool: capped sources by sha1 order, the others whole.

        The benign pool excludes NotInject, which is scored on its own.
        """
        out: list[FreshRow] = []
        for src in SOURCES:
            if src.key == "notinject":
                continue
            rows = self.select(src.key, attack)
            cap = POOL_CAPS.get((src.key, attack))
            out.extend(first_by_hash(rows, cap) if cap is not None else rows)
        return out


def load_fresh(
    cache_dir: Path | str = "data/raw",
    fetch: Fetcher = http_get,
    refresh: bool = False,
    exclude: set[str] | None = None,
) -> Fresh:
    """Every source, downloaded if needed, cleaned against the repository's own data.

    ``exclude`` replaces the training-data keys (tests pass a small set; the default reads the four
    datasets and the bundled file).
    """
    keys = training_keys(cache_dir, fetch) if exclude is None else exclude
    raw = [r for src in SOURCES for r in load_source(src.key, cache_dir, fetch, refresh)]
    rows = clean(raw, keys)
    kept = {src.key: sum(r.source == src.key for r in rows) for src in SOURCES}
    seen = {src.key: sum(r.source == src.key for r in raw) for src in SOURCES}
    return Fresh(rows, {k: seen[k] - kept[k] for k in kept})

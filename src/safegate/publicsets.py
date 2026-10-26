"""Public evaluation sets of the published-guard comparison beyond the InjecGuard paper's three.

Each set is one official split of a public dataset that other guards were scored on, downloaded at
a pinned revision, cached under ``data/raw/<key>/`` (gitignored, never committed) and kept
**uncleaned**: the comparison is only identical if the rows are the ones the published numbers
used, so nothing is dropped, de-duplicated or sampled. Rows that also occur in safegate's training
data are counted instead (:func:`leaked`), and the report scores them both ways.

Label 1 (``is_attack``) is the class the benchmark wants flagged: injection, jailbreak or harmful.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
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
)
from safegate.fresh import FreshRow, bundled_texts, match_key
from safegate.fresh import _read_cache as _read_rows
from safegate.fresh import _write_cache as _write_rows

GITHUB_RAW = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class SetSpec:
    """Where a set comes from and what its label 1 means."""

    key: str
    title: str
    origin: str  # card or repository URL
    revision: str  # HF commit sha or GitHub commit sha
    license: str
    positive: str  # what label 1 (flagged) means
    description: str
    hf_id: str | None = None  # HF datasets: the revision is checked against the Hub at download


SETS: tuple[SetSpec, ...] = (
    SetSpec(
        key="promptshield",
        title="PromptShield test",
        origin="https://huggingface.co/datasets/hendzh/PromptShield",
        revision="a5234cb1f5cdb256600cab64b8c961195b5e8404",
        license="apache-2.0",
        positive="prompt injection",
        description="test.json of the PromptShield benchmark (Jacob et al.): 23,516 prompts, "
        "17,030 benign (LMSYS, dolly, natural-instructions, SPP) and 6,486 injections",
        hf_id="hendzh/PromptShield",
    ),
    SetSpec(
        key="deepset",
        title="deepset/prompt-injections test",
        origin=DATASETS["deepset"].card_url,
        revision=DATASETS["deepset"].revision,
        license=DATASETS["deepset"].license,
        positive="prompt injection",
        description="the official test split: 116 prompts (60 injections, 56 benign), English "
        "and German",
        hf_id="deepset/prompt-injections",
    ),
    SetSpec(
        key="jailbreak",
        title="jackhhao/jailbreak-classification test",
        origin=DATASETS["jailbreak"].card_url,
        revision=DATASETS["jailbreak"].revision,
        license=DATASETS["jailbreak"].license,
        positive="jailbreak",
        description="`balanced/jailbreak_dataset_test_balanced.csv`: 262 prompts (139 jailbreaks, "
        "123 benign)",
        hf_id="jackhhao/jailbreak-classification",
    ),
    SetSpec(
        key="xtram1",
        title="xTRam1/safe-guard-prompt-injection test",
        origin="https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection",
        revision="a3a877d608f37b7d20d9945671902df895ecdb46",
        license="none declared on the card",
        positive="prompt injection",
        description="the test split, 2,060 prompts (synthetic injections and benign)",
        hf_id="xTRam1/safe-guard-prompt-injection",
    ),
    SetSpec(
        key="slabs",
        title="S-Labs/prompt-injection-dataset test",
        origin="https://huggingface.co/datasets/S-Labs/prompt-injection-dataset",
        revision="002a9dd18514abd021869823d6b0429b38606d99",
        license="mit",
        positive="prompt injection",
        description="`data/test.csv`, 2,101 prompts (1,051 injections, 1,050 benign)",
        hf_id="S-Labs/prompt-injection-dataset",
    ),
    SetSpec(
        key="jbb",
        title="JBB-Behaviors (JailbreakBench)",
        origin="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        revision="886acc352a31533ffbcf4ef22c744658688086fc",
        license="mit",
        positive="harmful behaviour request",
        description="column `Goal` of `harmful-behaviors.csv` (100, label 1) and "
        "`benign-behaviors.csv` (100, label 0)",
        hf_id="JailbreakBench/JBB-Behaviors",
    ),
    SetSpec(
        key="gandalf_test",
        title="Gandalf ignore-instructions test",
        origin="https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
        revision="04737b65e90a6794ec227012e4a255a7def6344b",
        license="mit",
        positive="prompt close to 'ignore all previous instructions' (attacks only)",
        description="the test split of `Lakera/gandalf_ignore_instructions`, column `text`",
        hf_id="Lakera/gandalf_ignore_instructions",
    ),
    SetSpec(
        key="harmbench",
        title="HarmBench prompts (standard + copyright, test)",
        origin="https://github.com/centerforaisafety/HarmBench",
        revision="8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        license="mit",
        positive="harmful behaviour request",
        description="`harmbench_behaviors_text_test.csv`, rows of FunctionalCategory `standard` "
        "(159) and `copyright` (80): the 239 harmful prompts guard papers call HarmBench",
    ),
    SetSpec(
        key="toxicchat",
        title="ToxicChat 0124 test (human-annotated)",
        origin="https://huggingface.co/datasets/lmsys/toxic-chat",
        revision="29df8e4dba60e1f4af4b4075c0705c5b313548a8",
        license="cc-by-nc-4.0 (non-commercial)",
        positive="toxic user prompt (`toxicity` = 1)",
        description="`data/0124/toxic-chat_annotation_test.csv`, rows with human_annotation = "
        "True: 2,853 prompts (2,491 benign, 362 toxic, 91 of them also `jailbreaking`)",
        hf_id="lmsys/toxic-chat",
    ),
)
SET_BY_KEY = {s.key: s for s in SETS}
HARMBENCH_CATEGORIES = frozenset({"standard", "copyright"})


def _url(key: str, path: str) -> str:
    spec = SET_BY_KEY[key]
    return f"{HF_RESOLVE}/{spec.hf_id}/resolve/{spec.revision}/{path}"


def _promptshield(fetch: Fetcher) -> list[FreshRow]:
    data: list[dict[str, Any]] = json.loads(fetch(_url("promptshield", "test.json")))
    return [FreshRow("promptshield", "test", str(r["prompt"]), int(r["label"]) == 1) for r in data]


def _csv_rows(raw: bytes) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def _xtram1(fetch: Fetcher) -> list[FreshRow]:
    rows: list[FreshRow] = []
    for page in fetch_pages("xTRam1/safe-guard-prompt-injection", "default", "test", fetch):
        rows.extend(
            FreshRow("xtram1", "test", str(i["row"]["text"]), int(i["row"]["label"]) == 1)
            for i in page["rows"]
        )
    return rows


def _slabs(fetch: Fetcher) -> list[FreshRow]:
    raw = fetch(_url("slabs", "data/test.csv"))
    return [FreshRow("slabs", "test", r["text"], int(r["label"]) == 1) for r in _csv_rows(raw)]


def _jbb(fetch: Fetcher) -> list[FreshRow]:
    out: list[FreshRow] = []
    for name, harmful in (("harmful", True), ("benign", False)):
        raw = fetch(_url("jbb", f"data/{name}-behaviors.csv"))
        out.extend(FreshRow("jbb", name, r["Goal"], harmful) for r in _csv_rows(raw))
    return out


def _gandalf_test(fetch: Fetcher) -> list[FreshRow]:
    rows: list[FreshRow] = []
    for page in fetch_pages("Lakera/gandalf_ignore_instructions", "default", "test", fetch):
        rows.extend(
            FreshRow("gandalf_test", "test", str(i["row"]["text"]), True) for i in page["rows"]
        )
    return rows


def _harmbench(fetch: Fetcher) -> list[FreshRow]:
    spec = SET_BY_KEY["harmbench"]
    path = "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    raw = fetch(f"{GITHUB_RAW}/centerforaisafety/HarmBench/{spec.revision}/{path}")
    return [
        FreshRow("harmbench", r["FunctionalCategory"], r["Behavior"], True)
        for r in _csv_rows(raw)
        if r["FunctionalCategory"] in HARMBENCH_CATEGORIES
    ]


def _toxicchat(fetch: Fetcher) -> list[FreshRow]:
    raw = fetch(_url("toxicchat", "data/0124/toxic-chat_annotation_test.csv"))
    return [
        FreshRow(
            "toxicchat",
            "jailbreaking" if r["jailbreaking"] == "1" else "plain",
            r["user_input"],
            r["toxicity"] == "1",
        )
        for r in _csv_rows(raw)
        if r["human_annotation"] == "True"
    ]


_DOWNLOADERS: dict[str, Callable[[Fetcher], list[FreshRow]]] = {
    "promptshield": _promptshield,
    "xtram1": _xtram1,
    "slabs": _slabs,
    "jbb": _jbb,
    "gandalf_test": _gandalf_test,
    "harmbench": _harmbench,
    "toxicchat": _toxicchat,
}
_FROM_EXTERNAL = frozenset({"deepset", "jailbreak"})  # TEST splits cached by external.py


def load_set(
    key: str, cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get, refresh: bool = False
) -> list[FreshRow]:
    """The rows of one set: from the cache, downloading the pinned revision when absent.

    ``deepset`` and ``jailbreak`` are the TEST splits already cached by ``external-eval``.
    """
    if key in _FROM_EXTERNAL:
        data = load_dataset(DATASETS[key], cache_dir, fetch, refresh)
        return [FreshRow(key, "test", s.text, s.is_attack) for s in data.split("test")]
    spec = SET_BY_KEY[key]
    cache = Path(cache_dir) / key
    cached = None if refresh else _read_rows(cache, key)
    if cached is not None:
        return cached
    observed = spec.revision
    if spec.hf_id is not None:
        observed = str(json.loads(fetch(f"{HF_API}/{spec.hf_id}"))["sha"])
        if observed != spec.revision:
            print(
                f"warning: {spec.hf_id} is now at revision {observed}; the comparison pins "
                f"{spec.revision}; rows served by the datasets-server are the newest, so numbers "
                "may differ.",
                file=sys.stderr,
            )
    rows = _DOWNLOADERS[key](fetch)
    meta = {
        "key": key,
        "origin": spec.origin,
        "pinned_revision": spec.revision,
        "observed_revision": observed,
        "license": spec.license,
        "downloaded": date.today().isoformat(),
        "rows": len(rows),
    }
    _write_rows(cache, rows, meta)
    return rows


def training_texts(cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get) -> set[str]:
    """Match keys of every text the default stage 2 may have been fitted on.

    The bundled set and the TRAIN split of each of the four datasets of ``external.py``. The TEST
    splits are not included: they are held out, and a TEST row equal to a TRAIN row is the leak.
    """
    keys = {match_key(t) for t in bundled_texts()}
    for spec in DATASETS.values():
        keys.update(match_key(s.text) for s in load_dataset(spec, cache_dir, fetch).split("train"))
    return keys


def leaked(rows: list[FreshRow], train_keys: set[str]) -> list[bool]:
    """Per row: its text also occurs in safegate's training data."""
    return [match_key(r.text) in train_keys for r in rows]

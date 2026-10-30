"""Indirect-injection training data for the opt-in ``--stage2 v3`` (``docs/protocol-indirect.md``).

The shipped v2 stage 2 is fitted on direct jailbreak and injection prompts. BIPIA, which the
InjecGuard paper scores, consists of bare injected instructions of a different kind ("Modify your
answer to ...", "Can you use Base16 ..."). This module builds extra TRAIN rows of the indirect kind
from public sources, none of them BIPIA, NotInject or WildGuard:

* attacker strings that are injected into tool outputs, emails or documents (InjecAgent, AgentDojo,
  Nemotron-RL-Agentic-IPI, LLMail-Inject);
* the same strings composed, deterministically, into benign documents (Dolly texts, the InjecAgent
  tool-response templates), each next to its unmodified benign twin;
* benign task-like instructions from the same sources, so the false-positive rate does not rise.

Every source is downloaded at a pinned revision and cached as JSONL under ``data/raw/<key>/``
(gitignored, never committed), as ``fresh.py`` does. SPML (``fresh.py``) and the in-the-wild
jailbreak prompts (``itw``, already in the pooled pipeline) are used as they are. Nothing here
scores a gate.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from safegate.external import HF_API, HF_RESOLVE, ExternalSample, Fetcher, http_get, stable_hash
from safegate.fresh import FreshRow, match_key

GITHUB_RAW = "https://raw.githubusercontent.com"
INJECAGENT_COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
AGENTDOJO_COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
NEMOTRON_REVISION = "d738d4f361cc38bb4d7a42b9066776dade5332f5"
LLMAIL_REVISION = "1063bdf01ec8762b812d5e06ee768a06faa5a6f7"

LLMAIL_KEEP = 600  # attacks kept in the cache: the first 600 in sha1 order of the text
PLACEHOLDER = "<Attacker Instruction>"  # InjecAgent tool-response templates
DEV_MOD = 5  # 1 in 5 attack strings (and their pairs) is held out of every candidate's TRAIN rows
SANITY_MOD = 10  # 1 in 10 benign Dolly / SPML texts never enters training (sanity check only)
COMPOSE_MAX = 500  # pairs of each composition kind in the universe (a candidate takes fewer)
AGENTDOJO_SUITES = ("workspace", "banking", "slack", "travel")

# Sources a candidate may name. ``spml`` and ``dolly`` come from ``fresh.py``; the rest below.
SOURCE_KEYS = ("spml", "dolly", "injecagent", "agentdojo", "nemotron", "llmail")


@dataclass(frozen=True)
class IndirectSource:
    """Where a source comes from and which revision the numbers were produced with."""

    key: str
    origin: str
    revision: str
    license: str
    description: str


SOURCES: tuple[IndirectSource, ...] = (
    IndirectSource(
        key="injecagent",
        origin="https://github.com/uiuc-kang-lab/InjecAgent",
        revision=INJECAGENT_COMMIT,
        license="mit (GitHub licence field)",
        description="62 attacker instructions (attacker_cases_dh / _ds), 17 user instructions and "
        "their tool-response templates with an <Attacker Instruction> slot (user_cases)",
    ),
    IndirectSource(
        key="agentdojo",
        origin="https://github.com/ethz-spylab/agentdojo",
        revision=AGENTDOJO_COMMIT,
        license="mit (GitHub licence field)",
        description="GOAL strings of the injection tasks and PROMPT strings of the user tasks of "
        "the four v1 suites (workspace, banking, slack, travel), parsed from the Python source",
    ),
    IndirectSource(
        key="nemotron",
        origin="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        revision=NEMOTRON_REVISION,
        license="cc-by-4.0 (card)",
        description="1,272 synthetic records: the injection text, the benign user request and the "
        "tool-returned document with and without the injection",
    ),
    IndirectSource(
        key="llmail",
        origin="https://huggingface.co/datasets/microsoft/llmail-inject-challenge",
        revision=LLMAIL_REVISION,
        license="mit (card)",
        description="phase-2 labelled submissions whose attack triggered the API (attack_attempt "
        f"true, reason api_triggered), the first {LLMAIL_KEEP} in sha1 order; 203 benign emails "
        "of emails_for_fp_tests.json",
    ),
)
SOURCE_BY_KEY = {s.key: s for s in SOURCES}

# Candidates dropped before any run, with the reason (printed by ``indirect-train``).
DROPPED: tuple[tuple[str, str], ...] = (
    (
        "xxz224/prompt-injection-attack-dataset (HF)",
        "no licence on the card (the repository has no README), so reuse terms are unknown",
    ),
    ("hackaprompt/hackaprompt-dataset (HF)", "gated (auto-approval click-through)"),
    (
        "TaskTracker, SEP (Should-it-be-executed-or-processed)",
        "not on the Hub under any id searched and no public repository found",
    ),
    (
        "Lakera/mosscap_prompt_injection (HF)",
        "direct jailbreak prompts of the same kind as the pooled gandalf/itw data, not indirect",
    ),
)


@dataclass(frozen=True)
class RawRow:
    """One cached piece of a source: a subset (``kind``), a label and the key its dev fold uses."""

    source: str
    kind: str
    text: str
    is_attack: bool
    pair: str  # rows built from the same attack string share it (and so fall in the same fold)


# ---------------------------------------------------------------------------
# Parsing (pure; tests feed tiny fixtures)
# ---------------------------------------------------------------------------


def _jsonl(raw: bytes | str) -> list[dict[str, Any]]:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_injecagent(attackers: Sequence[bytes | str], user_cases: bytes | str) -> list[RawRow]:
    """Attacker instructions (attacks), user instructions and tool-response templates (benign)."""
    out: list[RawRow] = []
    for raw in attackers:
        for rec in _jsonl(raw):
            text = str(rec["Attacker Instruction"]).strip()
            out.append(RawRow("injecagent", "attacker_instruction", text, True, text))
    for rec in _jsonl(user_cases):
        user = str(rec["User Instruction"]).strip()
        out.append(RawRow("injecagent", "user_instruction", user, False, user))
        template = str(rec["Tool Response Template"])
        if PLACEHOLDER in template:
            out.append(RawRow("injecagent", "template", template, False, template))
    return out


def _literal_env(tree: ast.AST) -> dict[str, Any]:
    """Names bound to literals anywhere in the module (class bodies included)."""
    env: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    env[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    return env


def _eval_string(node: ast.expr, env: Mapping[str, Any]) -> str | None:
    """The value of a string expression made of literals, f-strings over known names, ``+``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                parts.append(str(piece.value))
            elif isinstance(piece, ast.FormattedValue) and piece.format_spec is None:
                value = _eval_value(piece.value, env)
                if value is None:
                    return None
                parts.append(str(value))
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _eval_string(node.left, env), _eval_string(node.right, env)
        return None if left is None or right is None else left + right
    return None


def _eval_value(node: ast.expr, env: Mapping[str, Any]) -> Any | None:
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and len(node.args) == 1
    ):
        items = _eval_value(node.args[0], env)
        if isinstance(items, list | tuple) and all(isinstance(i, str) for i in items):
            return node.func.value.value.join(items)
    return None


def agentdojo_strings(source: str, names: tuple[str, ...]) -> list[str]:
    """Values of the class attributes in ``names`` (GOAL or PROMPT) that can be evaluated."""
    tree = ast.parse(source)
    env = _literal_env(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in names
            ):
                value = _eval_string(stmt.value, env)
                if value is not None and value.strip():
                    out.append(" ".join(value.split()))
    return out


def parse_agentdojo(goal_sources: Sequence[str], prompt_sources: Sequence[str]) -> list[RawRow]:
    out: list[RawRow] = []
    for src in goal_sources:
        for text in agentdojo_strings(src, ("GOAL",)):
            out.append(RawRow("agentdojo", "injection_goal", text, True, text))
    for src in prompt_sources:
        for text in agentdojo_strings(src, ("PROMPT",)):
            out.append(RawRow("agentdojo", "user_task", text, False, text))
    return out


def _find_string(obj: Any, needle: str) -> str | None:
    """The first string value inside ``obj`` (nested dicts and lists) that contains ``needle``."""
    if isinstance(obj, str):
        return obj if needle in obj else None
    children: Iterable[Any] = ()
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    for child in children:
        hit = _find_string(child, needle)
        if hit is not None:
            return hit
    return None


def parse_nemotron(raw: bytes | str) -> list[RawRow]:
    """Injection text, user request, and the document field with and without the injection."""
    out: list[RawRow] = []
    for rec in _jsonl(raw):
        injection = str(rec["injection"]["injection_text"]).strip()
        if not injection:
            continue
        out.append(RawRow("nemotron", "injection_text", injection, True, injection))
        for message in rec["responses_create_params"]["input"]:
            if message.get("role") == "user":
                out.append(
                    RawRow(
                        "nemotron",
                        "user_request",
                        str(message["content"]).strip(),
                        False,
                        injection,
                    )
                )
        doc = _find_string(rec["environment"], str(rec["injection"]["injection_text"]))
        if doc is not None:
            clean = " ".join(doc.replace(str(rec["injection"]["injection_text"]), " ").split())
            if clean:
                out.append(RawRow("nemotron", "document_attack", doc.strip(), True, injection))
                out.append(RawRow("nemotron", "document_benign", clean, False, injection))
    return out


def parse_llmail(
    labelled: bytes | str, benign: bytes | str, keep: int = LLMAIL_KEEP
) -> list[RawRow]:
    """API-triggering attacks (the first ``keep`` in sha1 order) and the benign emails."""
    data: dict[str, dict[str, Any]] = json.loads(labelled)
    attacks = [
        text
        for text, label in data.items()
        if str(label.get("attack_attempt")).lower() == "true"
        and label.get("reason") == "api_triggered"
    ]
    attacks.sort(key=lambda t: stable_hash(t, "cap"))
    out = [RawRow("llmail", "email_attack", t, True, t) for t in attacks[:keep]]
    out.extend(
        RawRow("llmail", "email_benign", str(t).strip(), False, str(t).strip())
        for t in json.loads(benign)
        if str(t).strip()
    )
    return out


# ---------------------------------------------------------------------------
# Download and cache
# ---------------------------------------------------------------------------


def _hf(rev: str, repo: str, path: str) -> str:
    return f"{HF_RESOLVE}/{repo}/resolve/{rev}/{path}"


def _download_injecagent(fetch: Fetcher) -> list[RawRow]:
    base = f"{GITHUB_RAW}/uiuc-kang-lab/InjecAgent/{INJECAGENT_COMMIT}/data"
    files = [fetch(f"{base}/attacker_cases_{k}.jsonl") for k in ("dh", "ds")]
    return parse_injecagent(files, fetch(f"{base}/user_cases.jsonl"))


def _download_agentdojo(fetch: Fetcher) -> list[RawRow]:
    base = f"{GITHUB_RAW}/ethz-spylab/agentdojo/{AGENTDOJO_COMMIT}/src/agentdojo/default_suites/v1"
    goals = [fetch(f"{base}/{s}/injection_tasks.py").decode() for s in AGENTDOJO_SUITES]
    prompts = [fetch(f"{base}/{s}/user_tasks.py").decode() for s in AGENTDOJO_SUITES]
    return parse_agentdojo(goals, prompts)


def _download_nemotron(fetch: Fetcher) -> list[RawRow]:
    repo = "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
    return parse_nemotron(fetch(_hf(NEMOTRON_REVISION, repo, "train.jsonl")))


def _download_llmail(fetch: Fetcher) -> list[RawRow]:
    repo = "microsoft/llmail-inject-challenge"
    labelled = fetch(_hf(LLMAIL_REVISION, repo, "data/labelled_unique_submissions_phase2.json"))
    benign = fetch(_hf(LLMAIL_REVISION, repo, "data/emails_for_fp_tests.json"))
    return parse_llmail(labelled, benign)


_DOWNLOADERS: dict[str, Callable[[Fetcher], list[RawRow]]] = {
    "injecagent": _download_injecagent,
    "agentdojo": _download_agentdojo,
    "nemotron": _download_nemotron,
    "llmail": _download_llmail,
}


def load_raw(
    key: str, cache_dir: Path | str = "data/raw", fetch: Fetcher = http_get, refresh: bool = False
) -> list[RawRow]:
    """The cached rows of one source, downloading the pinned revision when absent."""
    src = SOURCE_BY_KEY[key]
    cache = Path(cache_dir) / key
    if not refresh and (cache / "meta.json").exists():
        with open(cache / "rows.jsonl", encoding="utf-8") as f:
            return [RawRow(key, **json.loads(line)) for line in f if line.strip()]
    observed = src.revision
    if "huggingface.co" in src.origin:
        hf_id = src.origin.split("datasets/", 1)[1]
        observed = str(json.loads(fetch(f"{HF_API}/{hf_id}"))["sha"])
    rows = _DOWNLOADERS[key](fetch)
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            rec = {"kind": r.kind, "text": r.text, "is_attack": r.is_attack, "pair": r.pair}
            f.write(json.dumps(rec) + "\n")
    meta = {
        "key": key,
        "origin": src.origin,
        "pinned_revision": src.revision,
        "observed_revision": observed,
        "license": src.license,
        "downloaded": date.today().isoformat(),
        "rows": len(rows),
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# The universe: every new row any candidate may use
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewRow:
    """One new training row. ``fold`` decides its dev fold; rows of one pair share it."""

    source: str  # a key of SOURCE_KEYS, or "composed"
    kind: str
    text: str
    is_attack: bool
    fold: str

    def as_sample(self, split: str = "train") -> ExternalSample:
        return ExternalSample(self.text, self.is_attack, split)


def in_dev(fold: str) -> bool:
    """Whether ``fold`` is held out of every candidate's TRAIN rows (dev rows only)."""
    return stable_hash(match_key(fold), "k2-dev") % DEV_MOD == 0


def in_sanity(text: str) -> bool:
    """Whether a benign Dolly/SPML text is reserved for the sanity check (never trained on)."""
    return stable_hash(match_key(text), "k2-sanity") % SANITY_MOD == 0


def insert_attack(doc: str, attack: str, seed: str) -> str:
    """``attack`` inserted at the start, middle or end of ``doc``, chosen by ``seed``.

    The middle position is the sentence or line boundary closest to the centre of the text; the
    joiner is a blank line or a space. Both choices are a pure function of ``seed``.
    """
    h = stable_hash(seed, "k2-insert")
    where, joiner = h % 3, ("\n\n", " ")[(h // 3) % 2]
    doc, attack = doc.strip(), attack.strip()
    if where == 0:
        return attack + joiner + doc
    if where == 2:
        return doc + joiner + attack
    bounds = [m.end() for m in re.finditer(r"[.!?]\s|\n", doc)]
    if not bounds:
        return doc + joiner + attack
    cut = min(bounds, key=lambda b: abs(b - len(doc) / 2))
    return doc[:cut].rstrip() + joiner + attack + joiner + doc[cut:].lstrip()


def _by_hash(items: Sequence[str], salt: str) -> list[str]:
    return sorted(items, key=lambda t: stable_hash(t, salt))


@dataclass(frozen=True)
class Universe:
    """All new rows (dev fold included), the sanity rows, and what leak removal cost."""

    rows: list[NewRow]
    sanity: list[ExternalSample]
    removed: dict[str, int]  # source key -> rows dropped for matching a benchmark text

    def of(self, source: str, attack: bool | None = None) -> list[NewRow]:
        return [
            r for r in self.rows if r.source == source and (attack is None or r.is_attack == attack)
        ]


def _unique(rows: Sequence[NewRow]) -> list[NewRow]:
    """Drop repeated and label-conflicting texts (first occurrence wins), as ``fresh.clean``."""
    labels: dict[str, set[bool]] = {}
    for r in rows:
        labels.setdefault(match_key(r.text), set()).add(r.is_attack)
    seen: set[str] = set()
    out: list[NewRow] = []
    for r in rows:
        k = match_key(r.text)
        if not k or k in seen or len(labels[k]) > 1:
            continue
        seen.add(k)
        out.append(r)
    return out


def compose_pairs(
    attacks: Sequence[str],
    docs: Sequence[str],
    templates: Sequence[str],
    fillers: Sequence[str],
    *,
    n_docs: int = COMPOSE_MAX,
    n_templates: int = COMPOSE_MAX,
) -> list[NewRow]:
    """Attack strings composed into benign documents, each next to its unmodified twin.

    ``attacks``, ``docs`` and ``fillers`` are walked in sha1 order of their text, so the result is
    a pure function of the inputs. A document pair is (document with the attack inserted, the
    document); a template pair is (template with the attack in its slot, the template with a
    benign filler). A document is paired only with an attack of the same dev fold, so no document
    is seen in training and again in a held-out pair.
    """
    pool = _by_hash(list(attacks), "k2-compose")
    out: list[NewRow] = []
    if not pool:
        return out
    by_fold = {
        flag: iter([d for d in _by_hash(list(docs), "k2-doc") if in_dev(d) == flag])
        for flag in (False, True)
    }
    for i in range(n_docs):
        attack = pool[i % len(pool)]
        doc = next(by_fold[in_dev(attack)], None)
        if doc is None:
            continue
        out.append(NewRow("composed", "doc_attack", insert_attack(doc, attack, doc), True, attack))
        out.append(NewRow("composed", "doc_benign", doc, False, attack))
    tpls, fills = _by_hash(list(templates), "k2-tpl"), _by_hash(list(fillers), "k2-fill")
    if not tpls or not fills:
        return out
    for i in range(min(n_templates, len(pool) * len(tpls))):
        attack = pool[(i * 7 + 3) % len(pool)]
        tpl = tpls[i % len(tpls)]
        out.append(NewRow("composed", "tpl_attack", tpl.replace(PLACEHOLDER, attack), True, attack))
        filler = fills[i % len(fills)]
        out.append(
            NewRow("composed", "tpl_benign", tpl.replace(PLACEHOLDER, filler), False, attack)
        )
    return out


def _first_sentence(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    m = re.search(r"[.!?](\s|$)", flat)
    return (flat[: m.end()].strip() if m else flat)[:limit]


def build_universe(
    raw: Mapping[str, Sequence[RawRow]],
    dolly: Sequence[FreshRow],
    spml: Sequence[FreshRow],
    leak_keys: set[str],
) -> Universe:
    """Every new row, leak-checked and de-duplicated. The inputs are sorted, so the order is fixed.

    Leak removal runs on every text before anything is composed from it: a text whose
    ``match_key`` is in ``leak_keys`` (BIPIA, NotInject, WildGuard and the nine public sets) is
    dropped, and so is an attack string or document that would feed a composition.
    """
    removed = {k: 0 for k in (*SOURCE_KEYS, "composed")}
    sanity: list[ExternalSample] = []

    def keep(source: str, text: str) -> bool:
        if match_key(text) in leak_keys:
            removed[source] += 1
            return False
        return True

    rows: list[NewRow] = []
    for src, fresh_rows in (("spml", spml), ("dolly", dolly)):
        for r in fresh_rows:
            if not r.is_attack and in_sanity(r.text):
                if keep(src, r.text):
                    sanity.append(ExternalSample(r.text, False, src))
                continue
            if not keep(src, r.text):
                continue
            if src == "dolly":
                kind = "document" if "\n\n" in r.text else "instruction"
            else:
                kind = "chat_attack" if r.is_attack else "chat_benign"
            rows.append(NewRow(src, kind, r.text, r.is_attack, r.text))
    for key, raws in raw.items():
        rows.extend(
            NewRow(key, r.kind, r.text, r.is_attack, r.pair) for r in raws if keep(key, r.text)
        )
    rows = [r for r in rows if r.kind != "template"]  # slots are composed, never trained on bare
    rows = _unique(rows)

    attacks = sorted(
        {
            r.text
            for r in rows
            if r.is_attack
            and r.source in ("injecagent", "agentdojo", "nemotron")
            and r.kind != "document_attack"
        }
    )
    docs = [r.text for r in rows if r.source == "dolly" and r.kind == "document"]
    templates = [r.text for r in raw.get("injecagent", ()) if r.kind == "template"]
    fillers = [_first_sentence(d) for d in docs]
    composed = [
        c for c in compose_pairs(attacks, docs, templates, fillers) if keep("composed", c.text)
    ]
    composed = _unique(composed)
    used = {c.text for c in composed if c.kind == "doc_benign"}
    # a document that sits in a pair as the benign twin is not also a bare row (no duplicates)
    rows = [r for r in rows if not (r.source == "dolly" and r.text in used)]
    rows.extend(composed)
    return Universe(rows, sanity, removed)


# ---------------------------------------------------------------------------
# Candidates (pre-registered in docs/protocol-indirect.md)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """Which sources feed TRAIN, whether composed pairs are included, and the per-class cap."""

    key: str
    title: str
    sources: tuple[str, ...]
    compose: bool
    cap: int  # per source and class; composed pairs: cap // 2 of each kind


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "A",
        "bare rows, all sources, cap 300",
        ("spml", "dolly", "injecagent", "agentdojo", "nemotron", "llmail"),
        False,
        300,
    ),
    Candidate(
        "B",
        "A plus composed pairs, cap 300",
        ("spml", "dolly", "injecagent", "agentdojo", "nemotron", "llmail"),
        True,
        300,
    ),
    Candidate(
        "C",
        "indirect sources only (no SPML) plus composed pairs, cap 1000",
        ("dolly", "injecagent", "agentdojo", "nemotron", "llmail"),
        True,
        1000,
    ),
)
CANDIDATE_BY_KEY = {c.key: c for c in CANDIDATES}


def _cap_class(rows: Sequence[NewRow], cap: int) -> list[NewRow]:
    return sorted(rows, key=lambda r: stable_hash(r.text, "k2-cap"))[:cap]


def select_rows(u: Universe, cand: Candidate, fold: str) -> list[NewRow]:
    """The candidate's rows in ``fold``: "train" (non-dev), "dev" or "all" (the shipped fit).

    Bare rows are capped per source and class in sha1 order of the text. Composed pairs are capped
    per kind, taken whole (both members) in sha1 order of their attack string.
    """
    if fold not in ("train", "dev", "all"):
        raise ValueError(f"fold must be train, dev or all, got {fold!r}")
    pick = [r for r in u.rows if fold == "all" or in_dev(r.fold) == (fold == "dev")]
    out: list[NewRow] = []
    for source in cand.sources:
        for attack in (True, False):
            out.extend(
                _cap_class(
                    [r for r in pick if r.source == source and r.is_attack == attack], cand.cap
                )
            )
    if cand.compose:
        for prefix in ("doc", "tpl"):
            pairs = sorted(
                {r.fold for r in pick if r.source == "composed" and r.kind.startswith(prefix)},
                key=lambda f: stable_hash(f, "k2-cap"),
            )[: cand.cap // 2]
            keep = set(pairs)
            out.extend(
                r
                for r in pick
                if r.source == "composed" and r.kind.startswith(prefix) and r.fold in keep
            )
    return out


def dev_rows(u: Universe) -> list[NewRow]:
    """The common dev set: every dev-fold row except SPML attacks (direct, not indirect)."""
    return [r for r in u.rows if in_dev(r.fold) and not (r.source == "spml" and r.is_attack)]


# ---------------------------------------------------------------------------
# Leak accounting
# ---------------------------------------------------------------------------


def remove_leaks(
    rows: Sequence[ExternalSample], leak_keys: set[str]
) -> tuple[list[ExternalSample], int]:
    """``rows`` without any text whose ``match_key`` is in ``leak_keys``, and how many went."""
    kept = [r for r in rows if match_key(r.text) not in leak_keys]
    return kept, len(rows) - len(kept)

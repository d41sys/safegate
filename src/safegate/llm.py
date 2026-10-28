"""A small Anthropic Messages client: stdlib ``urllib`` only, with a JSONL disk cache.

Every call goes through the cache. A cache entry holds the request, the response content, the
model id the API returned, the usage and a UTC timestamp; the API key is never part of it (the key
is read from ``ANTHROPIC_API_KEY`` when a request is actually sent, and is removed from every error
message). Scoring reads the cache, so re-scoring is free and reproducible: with ``offline=True`` a
miss raises ``CacheMiss`` instead of calling the network.

Protocol: ``docs/protocol-llm.md``. The pinned model rejects ``temperature``, so requests carry no
sampling parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MODEL = "claude-sonnet-5-5"
DEFAULT_CACHE_DIR = Path("data/llm_cache")
# 4,000 calls in total for this work. 62 are reserved: 2 probes, 50 calls of a dev run whose cache
# file was discarded (see docs/protocol-llm.md, "Amendments"), and 10 spare.
MAX_CALLS = 3938
RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
TAG = "claude-sonnet-5-5 via Anthropic API, no logprobs"

Json = dict[str, Any]
Transport = Callable[[Json], Json]


class LLMError(RuntimeError):
    """The API refused the request or kept failing."""


class CacheMiss(LLMError):
    """Offline mode, and the request is not in the cache."""


class BudgetExceeded(LLMError):
    """The next call would pass the call budget."""


class _Retryable(Exception):
    def __init__(self, status: int, retry_after: float | None, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.retry_after = retry_after


@dataclass(frozen=True)
class Response:
    """One model reply: the content blocks plus what the API said about it."""

    content: list[Json]
    stop_reason: str
    model: str
    usage: Json
    cached: bool = False

    @property
    def text(self) -> str:
        """All text blocks joined."""
        return "".join(str(b["text"]) for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[Json]:
        """The ``tool_use`` blocks, in order."""
        return [b for b in self.content if b.get("type") == "tool_use"]


@dataclass
class Stats:
    """Calls and tokens of one client (this process only; cache hits cost nothing)."""

    network_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0

    def line(self) -> str:
        return (
            f"{self.network_calls} API calls ({self.cache_hits} served from the cache, "
            f"{self.retries} retried), {self.input_tokens} input and {self.output_tokens} "
            "output tokens in this run"
        )


def request_key(model: str, payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of the model and the request body."""
    blob = json.dumps({"model": model, **payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _redact(text: str, key: str | None) -> str:
    return text.replace(key, "***") if key else text


def http_transport(payload: Json, *, timeout: float = 180.0) -> Json:
    """POST one request. Raises ``_Retryable`` on 429/5xx/529 and network errors, else LLMError."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set (use the offline cache to score without it)")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: Json = json.loads(resp.read())
            return data
    except urllib.error.HTTPError as err:
        body = _redact(err.read().decode("utf-8", "replace")[:400], key)
        if err.code in RETRY_STATUS:
            raw = err.headers.get("retry-after") if err.headers else None
            try:
                after = float(raw) if raw else None
            except ValueError:
                after = None
            raise _Retryable(err.code, after, body) from None
        raise LLMError(f"HTTP {err.code}: {body}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        raise _Retryable(0, None, _redact(str(err), key)) from None


@dataclass
class AnthropicClient:
    """Cached, retrying, budgeted Messages client; safe to share between threads."""

    cache_path: Path
    model: str = MODEL
    offline: bool = False
    max_calls: int = MAX_CALLS
    attempts: int = 8
    transport: Transport = http_transport
    sleep: Callable[[float], None] = time.sleep
    stats: Stats = field(default_factory=Stats)
    _entries: dict[str, Json] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _key_locks: dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)
    _banked: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        self._entries = _read_cache(self.cache_path)
        self._banked = _count_cached(self.cache_path.parent)

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> Response:
        """One Messages call, from the cache when this exact request was made before."""
        payload: Json = {
            "max_tokens": max_tokens,
            "system": system,
            "messages": [dict(m) for m in messages],
        }
        if tools:
            payload["tools"] = [dict(t) for t in tools]
        key = request_key(self.model, payload)
        with self._lock:
            hit = self._entries.get(key)
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        if hit is None:
            with key_lock:
                with self._lock:
                    hit = self._entries.get(key)
                if hit is None:
                    return self._call(key, payload)
        with self._lock:
            self.stats.cache_hits += 1
        return _response(hit["response"], cached=True)

    def _call(self, key: str, payload: Json) -> Response:
        if self.offline:
            raise CacheMiss(f"offline: request {key[:12]} is not in {self.cache_path}")
        with self._lock:
            if self._banked + self.stats.network_calls >= self.max_calls:
                raise BudgetExceeded(f"call budget of {self.max_calls} reached")
            self.stats.network_calls += 1
        body: Json = {"model": self.model, **payload}
        data: Json | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                data = self.transport(body)
                break
            except _Retryable as err:
                if attempt == self.attempts:
                    raise LLMError(f"gave up after {attempt} attempts: {err}") from None
                delay = err.retry_after or min(2.0 * 2 ** (attempt - 1), 60.0)
                with self._lock:
                    self.stats.retries += 1
                self.sleep(delay + random.uniform(0, 1))
        if data is None:  # pragma: no cover - the loop returns or raises
            raise LLMError("no response")
        response = {
            "content": data["content"],
            "stop_reason": data.get("stop_reason") or "",
            "model": data["model"],
            "usage": _usage(data.get("usage") or {}),
        }
        entry: Json = {
            "key": key,
            "request": body,
            "response": response,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._entries[key] = entry
            self.stats.input_tokens += int(response["usage"].get("input_tokens", 0))
            self.stats.output_tokens += int(response["usage"].get("output_tokens", 0))
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        return _response(response, cached=False)

    def entries(self) -> list[Json]:
        """Every cache entry of this client's file."""
        with self._lock:
            return list(self._entries.values())

    def run_date(self) -> str:
        """Local calendar date of the earliest cache entry (the date the numbers carry)."""
        stamps = sorted(str(e["timestamp"]) for e in self.entries())
        if not stamps:
            return datetime.now().astimezone().date().isoformat()
        return datetime.fromisoformat(stamps[0]).astimezone().date().isoformat()

    def tag(self) -> str:
        """The provenance tag every reported number carries."""
        return f"{self.model} via Anthropic API, {self.run_date()}, no logprobs"

    def served_models(self) -> set[str]:
        """The model ids the API returned for the entries of this cache."""
        return {str(e["response"]["model"]) for e in self.entries()}


def _usage(raw: Mapping[str, Any]) -> Json:
    return {
        k: raw[k]
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens")
        if k in raw
    }


def _response(raw: Mapping[str, Any], *, cached: bool) -> Response:
    return Response(
        content=[dict(b) for b in raw["content"]],
        stop_reason=str(raw["stop_reason"]),
        model=str(raw["model"]),
        usage=dict(raw["usage"]),
        cached=cached,
    )


def _read_cache(path: Path) -> dict[str, Json]:
    entries: dict[str, Json] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries[entry["key"]] = entry
    return entries


def _count_cached(directory: Path) -> int:
    """Entries over every cache file next to this one: the calls already spent."""
    total = 0
    if directory.is_dir():
        for p in directory.glob("*.jsonl"):
            with open(p, encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
    return total


def parse_agent_spec(spec: str) -> str:
    """``anthropic:<model>`` to the model id; anything else is an error."""
    provider, _, model = spec.partition(":")
    if provider != "anthropic" or not model:
        raise ValueError(f"unknown agent {spec!r}: expected 'anthropic:<model>'")
    return model


def totals(clients: Sequence[AnthropicClient]) -> tuple[int, int, int]:
    """(calls in the cache files, input tokens, output tokens) over the clients' caches."""
    calls = inp = out = 0
    for c in clients:
        for e in c.entries():
            calls += 1
            inp += int(e["response"]["usage"].get("input_tokens", 0))
            out += int(e["response"]["usage"].get("output_tokens", 0))
    return calls, inp, out

"""The cached Anthropic client, with a fake transport: no key and no network."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from safegate.llm import (
    AnthropicClient,
    BudgetExceeded,
    CacheMiss,
    LLMError,
    _Retryable,
    parse_agent_spec,
    request_key,
)


def _reply(text: str = "ok") -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "model": "claude-sonnet-5-5",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }


def _client(tmp_path: Path, transport: Any, **kw: Any) -> AnthropicClient:
    return AnthropicClient(tmp_path / "c.jsonl", transport=transport, sleep=lambda _s: None, **kw)


def test_second_identical_request_is_served_from_the_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return _reply()

    client = _client(tmp_path, transport)
    msg = [{"role": "user", "content": "hi"}]
    first = client.complete(system="s", messages=msg)
    second = client.complete(system="s", messages=msg)
    assert (first.text, first.cached, second.cached) == ("ok", False, True)
    assert len(calls) == 1
    assert "temperature" not in calls[0] and calls[0]["model"] == "claude-sonnet-5-5"
    assert (client.stats.network_calls, client.stats.cache_hits) == (1, 1)
    # a new client reads the file; the entry keeps the request, the served model and the usage
    again = _client(tmp_path, transport)
    entry = again.entries()[0]
    assert entry["response"]["model"] == "claude-sonnet-5-5"
    assert entry["response"]["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert entry["request"]["messages"] == msg and entry["timestamp"].endswith("+00:00")
    assert again.complete(system="s", messages=msg).cached and len(calls) == 1


def test_any_change_of_the_request_is_a_new_key() -> None:
    base = {"max_tokens": 5, "system": "s", "messages": [{"role": "user", "content": "a"}]}
    assert request_key("m", base) != request_key("m2", base)
    assert request_key("m", base) != request_key("m", {**base, "system": "t"})
    assert request_key("m", base) == request_key("m", dict(reversed(list(base.items()))))


def test_offline_mode_reads_the_cache_and_fails_on_a_miss(tmp_path: Path) -> None:
    online = _client(tmp_path, lambda _b: _reply("cached"))
    online.complete(system="s", messages=[{"role": "user", "content": "q"}])

    def never(_b: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("network used")

    offline = _client(tmp_path, never, offline=True)
    assert (
        offline.complete(system="s", messages=[{"role": "user", "content": "q"}]).text == "cached"
    )
    with pytest.raises(CacheMiss):
        offline.complete(system="s", messages=[{"role": "user", "content": "other"}])


def test_retries_with_backoff_then_succeeds_and_gives_up(tmp_path: Path) -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def flaky(_b: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Retryable(429, None, "rate limited")
        return _reply()

    client = AnthropicClient(tmp_path / "a.jsonl", transport=flaky, sleep=sleeps.append)
    assert client.complete(system="s", messages=[{"role": "user", "content": "x"}]).text == "ok"
    assert client.stats.retries == 2 and len(sleeps) == 2 and sleeps[1] > sleeps[0] - 1

    def down(_b: dict[str, Any]) -> dict[str, Any]:
        raise _Retryable(529, 0.0, "overloaded")

    bad = AnthropicClient(tmp_path / "b.jsonl", transport=down, sleep=lambda _s: None, attempts=3)
    with pytest.raises(LLMError, match="gave up after 3"):
        bad.complete(system="s", messages=[{"role": "user", "content": "x"}])


def test_budget_counts_cached_entries_of_the_directory(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda _b: _reply(), max_calls=2)
    client.complete(system="s", messages=[{"role": "user", "content": "1"}])
    client.complete(system="s", messages=[{"role": "user", "content": "2"}])
    client.complete(system="s", messages=[{"role": "user", "content": "1"}])  # cache hit
    with pytest.raises(BudgetExceeded):
        client.complete(system="s", messages=[{"role": "user", "content": "3"}])
    # another file in the same directory shares the ledger
    other = AnthropicClient(tmp_path / "other.jsonl", transport=lambda _b: _reply(), max_calls=2)
    with pytest.raises(BudgetExceeded):
        other.complete(system="s", messages=[{"role": "user", "content": "4"}])


def test_concurrent_identical_requests_make_one_call(tmp_path: Path) -> None:
    n = {"calls": 0}
    gate = threading.Event()

    def slow(_b: dict[str, Any]) -> dict[str, Any]:
        n["calls"] += 1
        gate.wait(0.2)
        return _reply()

    client = _client(tmp_path, slow)
    threads = [
        threading.Thread(
            target=client.complete,
            kwargs={"system": "s", "messages": [{"role": "user", "content": "same"}]},
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert n["calls"] == 1


def test_the_key_never_reaches_the_cache_or_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-123")
    client = AnthropicClient(tmp_path / "k.jsonl", transport=lambda _b: _reply())
    client.complete(system="s", messages=[{"role": "user", "content": "x"}])
    assert "sk-test-secret-123" not in (tmp_path / "k.jsonl").read_text()


def test_http_transport_without_a_key_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from safegate.llm import http_transport

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        http_transport({"model": "m"})


def test_agent_spec() -> None:
    assert parse_agent_spec("anthropic:claude-sonnet-5-5") == "claude-sonnet-5-5"
    for bad in ("claude-sonnet-5-5", "openai:gpt", "anthropic:"):
        with pytest.raises(ValueError):
            parse_agent_spec(bad)


@pytest.mark.network
@pytest.mark.skipif(
    not __import__("os").environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_live_call(tmp_path: Path) -> None:
    client = AnthropicClient(tmp_path / "live.jsonl", max_calls=5)
    out = client.complete(
        system="Reply with one word.", messages=[{"role": "user", "content": "Hi"}]
    )
    assert out.model == "claude-sonnet-5-5" and out.content

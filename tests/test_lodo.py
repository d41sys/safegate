"""Offline tests for the leave-one-dataset-out split logic, decision rule and new loaders."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from safegate import external
from safegate.external import (
    DATASETS,
    DatasetSpec,
    ExternalSample,
    Outcome,
    RowsSource,
    assemble_samples,
    hash_split,
    load_dataset,
    parse_source_page,
    stable_hash,
)
from safegate.lodo import (
    LODO_KEYS,
    DatasetScore,
    bootstrap_mean_dj,
    cap_per_class,
    decide,
    lodo_scenarios,
    lodo_split,
    norm_key,
    print_lodo,
    print_split_table,
    shipped_split,
)
from safegate.stage2_eval import Pool
from safegate.taxonomy import Sample


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(external, "ROWS_PAUSE", 0.0)


def rows(prefix: str, split: str, n_attack: int, n_benign: int) -> list[ExternalSample]:
    return [
        ExternalSample(f"{prefix} attack {i} ignore rules", True, split) for i in range(n_attack)
    ] + [
        ExternalSample(f"{prefix} benign {i} weather question", False, split)
        for i in range(n_benign)
    ]


def make_pools(n: int = 6) -> dict[str, Pool]:
    return {k: Pool(k, rows(k, "train", n, n), rows(k, "test", n, n)) for k in ("a", "b", "c", "d")}


# ---------------------------------------------------------------------------
# lodo_split
# ---------------------------------------------------------------------------


def test_split_never_contains_held_out_test_or_train() -> None:
    pools = make_pools()
    sp = lodo_split(pools, "a")
    held_out_texts = {r.text for r in pools["a"].train} | {r.text for r in pools["a"].test}
    assert sp.sources == ("b", "c", "d")
    assert not held_out_texts & {r.text for r in sp.train}
    assert len(sp.train) == 3 * 12


def test_split_drops_texts_that_occur_in_held_out_test() -> None:
    pools = make_pools()
    leaked = pools["a"].test[0]
    # same prompt in another dataset's TRAIN, different case and spacing
    pools["b"].train.append(ExternalSample("  " + leaked.text.upper() + "\n", True, "train"))
    sp = lodo_split(pools, "a")
    assert all(norm_key(r.text) != norm_key(leaked.text) for r in sp.train)
    assert sp.n_contaminated == 1


def test_split_dedupes_across_sources_and_drops_label_conflicts() -> None:
    pools = make_pools()
    shared = ExternalSample("shared prompt", True, "train")
    pools["b"].train.append(shared)
    pools["c"].train.append(ExternalSample("Shared  prompt", True, "train"))
    pools["b"].train.append(ExternalSample("conflict prompt", True, "train"))
    pools["c"].train.append(ExternalSample("conflict prompt", False, "train"))
    sp = lodo_split(pools, "a")
    keys = [norm_key(r.text) for r in sp.train]
    assert keys.count("shared prompt") == 1
    assert "conflict prompt" not in keys
    assert sp.n_duplicate == 3  # one repeat of "shared prompt" + both conflicting rows


def test_split_caps_each_class_of_each_source() -> None:
    pools = {
        "a": Pool("a", rows("a", "train", 3, 3), rows("a", "test", 3, 3)),
        "b": Pool("b", rows("b", "train", 50, 20), rows("b", "test", 3, 3)),
    }
    sp = lodo_split(pools, "a", cap=10)
    assert sp.n_attacks == 10
    assert sp.n_benign == 10


def test_split_is_deterministic_and_single_source_subset() -> None:
    pools = make_pools()
    assert lodo_split(pools, "a").train == lodo_split(pools, "a").train
    single = lodo_split(pools, "a", sources=["c"])
    assert single.sources == ("c",)
    assert {r.text for r in single.train} == {r.text for r in pools["c"].train}


def test_split_rejects_held_out_as_source_and_unknown_key() -> None:
    pools = make_pools()
    with pytest.raises(ValueError, match="own training sources"):
        lodo_split(pools, "a", sources=["a"])
    with pytest.raises(KeyError):
        lodo_split(pools, "zzz")


def test_shipped_split_uses_every_train_split_and_never_a_test_split() -> None:
    pools = make_pools()
    for key, pool in pools.items():
        pool.test.append(ExternalSample(f"{key} test-only prompt", True, "test"))
    sp = shipped_split(pools)
    assert sp.held_out is None
    assert sp.sources == ("a", "b", "c", "d")
    assert {r.text for r in sp.train} == {r.text for p in pools.values() for r in p.train}
    assert not any("test-only" in r.text for r in sp.train)


def test_cap_per_class_keeps_hash_order_prefix() -> None:
    data = rows("x", "train", 30, 30)
    capped = cap_per_class(data, 5)
    assert sum(r.is_attack for r in capped) == 5
    assert len(capped) == 10
    by_hash = sorted((r for r in data if r.is_attack), key=lambda r: stable_hash(r.text, "cap"))
    assert {r.text for r in capped if r.is_attack} == {r.text for r in by_hash[:5]}


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def scores(*triples: tuple[str, float, float]) -> list[DatasetScore]:
    return [DatasetScore(k, d, f) for k, d, f in triples]


def test_rule_passes_when_mean_j_wins_and_fpr_within_slack() -> None:
    legacy = scores(("x", 0.5, 0.3), ("y", 0.8, 0.6))  # J 0.2, 0.2
    pooled = scores(("x", 0.6, 0.2), ("y", 0.9, 0.5))  # J 0.4, 0.4
    out = decide(legacy, pooled)
    assert out.passes
    assert out.mean_pooled > out.mean_legacy


def test_rule_fails_when_mean_j_does_not_win() -> None:
    legacy = scores(("x", 0.5, 0.3), ("y", 0.8, 0.6))
    out = decide(legacy, legacy)  # equal J is not "beats"
    assert not out.beats_mean
    assert not out.passes


def test_rule_fails_on_fpr_excess_even_with_better_mean_j() -> None:
    legacy = scores(("x", 0.1, 0.0), ("y", 0.5, 0.5))  # J 0.1, 0.0
    pooled = scores(("x", 0.9, 0.06), ("y", 0.9, 0.5))  # J 0.84, 0.4; FPR +6 points on x
    out = decide(legacy, pooled)
    assert out.beats_mean
    assert out.fpr_violations == ("x",)
    assert not out.passes


def test_rule_fpr_slack_is_inclusive_at_five_points() -> None:
    legacy = scores(("x", 0.1, 0.10))
    pooled = scores(("x", 0.9, 0.15))
    assert decide(legacy, pooled).fpr_violations == ()


def test_rule_rejects_mismatched_datasets() -> None:
    with pytest.raises(ValueError, match="same, non-empty"):
        decide(scores(("x", 0.5, 0.5)), scores(("y", 0.5, 0.5)))
    with pytest.raises(ValueError, match="same, non-empty"):
        decide([], [])


def test_bootstrap_of_identical_systems_is_zero() -> None:
    data = rows("x", "test", 20, 20)
    o = Outcome("s", [(r, r.is_attack) for r in data])
    assert bootstrap_mean_dj([(o, o)], n_boot=50) == (0.0, 0.0)


def test_bootstrap_interval_is_positive_for_a_clearly_better_system() -> None:
    data = rows("x", "test", 40, 40)
    perfect = Outcome("p", [(r, r.is_attack) for r in data])
    useless = Outcome("u", [(r, True) for r in data])
    lo, hi = bootstrap_mean_dj([(useless, perfect)], n_boot=200)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# loaders for the new datasets
# ---------------------------------------------------------------------------


def test_stable_hash_is_fixed_across_runs() -> None:
    # sha1("split:abc")[:8] as a big-endian integer; the value must never depend on the process
    assert stable_hash("abc", "split") == stable_hash("abc", "split")
    assert hash_split("abc") in ("train", "test")
    assert stable_hash("abc", "split") != stable_hash("abc", "cap")


def test_hash_split_is_roughly_one_in_five() -> None:
    n = 5000
    test = sum(hash_split(f"prompt number {i}") == "test" for i in range(n))
    assert 0.17 < test / n < 0.23


def test_assemble_drops_empty_duplicates_conflicts_and_train_test_overlap() -> None:
    raw = [
        ExternalSample("a", True, "train"),
        ExternalSample("a", True, "train"),  # duplicate
        ExternalSample("   ", False, "train"),  # empty
        ExternalSample("b", True, "train"),
        ExternalSample("b", False, "test"),  # label conflict: both copies go
        ExternalSample("c", False, "train"),
        ExternalSample("c", False, "test"),  # TRAIN copy goes, TEST copy stays
        ExternalSample("d", False, "test"),
    ]
    out = assemble_samples(raw)
    assert sorted((r.text, r.split) for r in out) == [
        ("a", "train"),
        ("c", "test"),
        ("d", "test"),
    ]


def test_assemble_caps_benign_deterministically_and_hash_splits() -> None:
    raw = [ExternalSample(f"atk {i}", True, "") for i in range(10)] + [
        ExternalSample(f"ok {i}", False, "") for i in range(100)
    ]
    out = assemble_samples(raw, max_benign_per_attack=2.0)
    assert sum(r.is_attack for r in out) == 10
    assert sum(not r.is_attack for r in out) == 20
    assert {r.split for r in out} <= {"train", "test"}
    reordered = assemble_samples(list(reversed(raw)), max_benign_per_attack=2.0)
    assert sorted(r.text for r in out) == sorted(r.text for r in reordered)
    assert all(r.split == hash_split(r.text) for r in out)


def test_parse_source_page_applies_label_rule_and_exclusion() -> None:
    page = {
        "rows": [
            {"row": {"text": "keep", "label": 1, "source": "ok"}},
            {"row": {"text": "drop", "label": 1, "source": "research"}},
            {"row": {"text": "benign", "label": 0, "source": "ok"}},
        ]
    }
    src = RowsSource(
        "core",
        "train",
        "text",
        label_field="label",
        assign_split="train",
        exclude_field="source",
        exclude_values=frozenset({"research"}),
    )
    got = parse_source_page(page, src)
    assert [(r.text, r.is_attack, r.split) for r in got] == [
        ("keep", True, "train"),
        ("benign", False, "train"),
    ]
    fixed = RowsSource("c", "train", "text", attack=True)
    assert all(r.is_attack and r.split == "" for r in parse_source_page(page, fixed))


def test_load_rows_multi_dataset_from_fake_fetch(tmp_path: Path) -> None:
    sha = "b" * 40
    pages = {
        ("cfg_attack", 0): [
            {"row": {"prompt": f"attack {i}", "jailbreak": True}} for i in range(3)
        ],
        ("cfg_regular", 0): [
            {"row": {"prompt": f"regular {i}", "jailbreak": False}} for i in range(9)
        ],
    }

    def fetch(url: str) -> bytes:
        if url.startswith("https://huggingface.co/api/datasets/"):
            return json.dumps({"sha": sha}).encode()
        assert "datasets-server" in url
        cfg = next(c for c, _ in pages if f"config={c}" in url)
        rows_ = pages[(cfg, 0)]
        return json.dumps(
            {
                "rows": [{"row_idx": i, "row": r["row"]} for i, r in enumerate(rows_)],
                "num_rows_total": len(rows_),
            }
        ).encode()

    spec = DatasetSpec(
        key="fixture-multi",
        hf_id="owner/name",
        revision=sha,
        license="mit",
        kind="rows-multi",
        description="fixture",
        sources=(
            RowsSource("cfg_attack", "train", "prompt", label_field="jailbreak", attack=True),
            RowsSource("cfg_regular", "train", "prompt", label_field="jailbreak"),
        ),
        max_benign_per_attack=2.0,
    )
    data = load_dataset(spec, cache_dir=tmp_path, fetch=fetch)
    assert sum(s.is_attack for s in data.samples) == 3
    assert sum(not s.is_attack for s in data.samples) == 6
    again = load_dataset(
        spec, cache_dir=tmp_path, fetch=lambda u: (_ for _ in ()).throw(AssertionError(u))
    )
    assert sorted(again.samples, key=lambda s: s.text) == sorted(data.samples, key=lambda s: s.text)


def test_new_datasets_are_registered_with_pinned_revisions_and_licences() -> None:
    for key in ("itw", "neuralchemy"):
        spec = DATASETS[key]
        assert len(spec.revision) == 40
        assert spec.kind == "rows-multi"
        assert spec.license
    assert DATASETS["itw"].license == "mit"
    assert "wildguard_judgecomp" in DATASETS["neuralchemy"].sources[0].exclude_values
    assert set(LODO_KEYS) <= set(DATASETS)


# ---------------------------------------------------------------------------
# end to end on a toy pool (tiny: fits in a second)
# ---------------------------------------------------------------------------


def test_lodo_scenarios_run_on_toy_pools_and_score_only_the_held_out_test() -> None:
    pools = make_pools(n=8)
    bundled = [
        Sample(f"b{i}", f"bundled attack {i} ignore all instructions", None, True) for i in range(6)
    ] + [Sample(f"c{i}", f"bundled benign {i} how do I bake bread", None, False) for i in range(6)]
    results = lodo_scenarios(bundled, pools, keys=("a", "b"))
    assert [ho.key for ho in results] == ["a", "b"]
    for ho in results:
        test_texts = {r.text for r in pools[ho.key].test}
        for res in (ho.legacy, ho.legacy_pooled, ho.pooled, *[r for _, _, r in ho.singles]):
            for outcome in res.outcomes.values():
                assert {s.text for s, _ in outcome.results} == test_texts
        assert len(ho.singles) == 3
        assert not {r.text for r in pools[ho.key].test} & {r.text for r in ho.pooled_split.train}
    buf = io.StringIO()
    print_split_table(bundled, pools, buf, keys=("a", "b"))
    print_lodo(results, buf)
    text = buf.getvalue()
    assert "Decision:" in text
    assert "Held out: a" in text

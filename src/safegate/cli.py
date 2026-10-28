"""Command-line interface for the safegate safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from safegate.adaptive import (
    ADOPT_MAX_FPR_RISE,
    ADOPT_MIN_ASR_DROP,
    BUDGET,
    DEV_ATTACKS_PER_DATASET,
    DEV_BENIGN_PER_DATASET,
    WINDOW_DEV_ATTACKS_PER_DATASET,
    WINDOW_DEV_BENIGN_PER_DATASET,
    WINDOW_SIZES,
    WINDOW_STRIDE_DIVISORS,
    WINDOW_THRESHOLDS,
    Attack,
    Run,
    adoption,
    check_unicode_confusables,
    choose_window,
    describe,
    dev_sample,
    make_gate_oracle,
    make_oracle,
    run_attacks,
    select_components,
    select_windows,
    verdict,
    window_grid,
)
from safegate.benchmark import (
    SUITES,
    adaptive_rows,
    load_for_cli,
    print_benchmark,
    print_window_report,
    run_benchmark,
    window_protocol,
)
from safegate.defence import COMPONENTS, SHIPPED_COMPONENTS, SHIPPED_WINDOWS, DefenceConfig
from safegate.external import (
    DATASETS,
    ExternalSample,
    LoadedDataset,
    Outcome,
    Rate,
    load_dataset,
    score,
    shorten,
)
from safegate.gate import BLOCKING_ACTIONS, SafetyGate
from safegate.injection import DEFAULT_STAGE2, STAGE2_MODES, InjectionDetector
from safegate.llm import (
    DEFAULT_CACHE_DIR,
    AnthropicClient,
    LLMError,
    parse_agent_spec,
    totals,
)
from safegate.llmguard import (
    dev_set,
    format_guard_report,
    load_guard_data,
    run_guard,
    score_dev,
)
from safegate.lodo import (
    LODO_KEYS,
    MAX_TRAIN_PER_CLASS,
    lodo_scenarios,
    print_lodo,
    print_split_table,
    shipped_split,
)
from safegate.mock_llm import MockStreamingLLM
from safegate.perturb import homoglyph_unlisted, homoglyph_unseen, rng_for
from safegate.refusal import RefusalEvaluator
from safegate.sota import SOTA_KEYS, print_sota, resolve_suites, run_sota
from safegate.spec import check_spec, load_spec_policy
from safegate.stage2_eval import (
    DATASET_KEYS,
    Pool,
    dev_grid,
    fit_v2,
    print_dev_grid,
    print_diagnosis,
    print_scenarios,
    scenarios,
)
from safegate.taxonomy import Sample
from safegate.toolagent import LLM_CONDITIONS, format_llm_report, run_dev, run_llm_suite
from safegate.toolpolicy import (
    Session,
    ToolCall,
    ToolPolicy,
    check_tool_call,
    load_tool_policy,
)
from safegate.toolsuite import CONDITION_HELP, format_report, run_suite
from safegate.windows import WindowConfig


def _detector(args: argparse.Namespace) -> InjectionDetector:
    """The detector selected by ``--stage2`` (the shipped pooled v2 model by default)."""
    return InjectionDetector(stage2=args.stage2)


def _load_bundled_samples() -> list[Sample]:
    """Load the bundled injection training dataset."""
    candidates: list[Path] = []
    try:
        pkg = resources.files("safegate").joinpath("data/injection_train.jsonl")
        candidates.append(Path(str(pkg)))
    except Exception:
        pass
    candidates.append(Path(__file__).parent / "data" / "injection_train.jsonl")

    for path in candidates:
        if path.exists():
            samples: list[Sample] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(Sample.from_dict(json.loads(line)))
            return samples
    raise FileNotFoundError("Could not locate bundled injection_train.jsonl dataset")


def _gate(args: argparse.Namespace) -> SafetyGate:
    """The gate selected by ``--stage2`` and ``--defence`` (no defence unless asked for)."""
    return SafetyGate(detector=_detector(args), defence=_defence_from(args))


def cmd_check(args: argparse.Namespace) -> int:
    """Check a single piece of text against the safety gate."""
    gate = _gate(args)
    decision = gate.check_input(args.text)

    print(f"action:   {decision.action}")
    print(f"category: {decision.category or 'none'}")
    print(f"score:    {decision.score:.4f}")
    if decision.rule_ids:
        print(f"rules:    {', '.join(decision.rule_ids)}")
    if decision.reasons:
        print("reasons:")
        for r in decision.reasons:
            print(f"  - {r}")
    if decision.sanitized_text is not None:
        print(f"sanitized: {decision.sanitized_text}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one prompt through all three checkpoints against the mock streaming LLM."""
    gate = _gate(args)
    output, trace = gate.run(args.prompt, MockStreamingLLM())
    print(f"output: {output.strip()}")
    print(f"final action: {trace.final_action}")
    if args.trace:
        print(json.dumps(trace.to_dict(), indent=2))
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    """Run red-team evaluation on the held-out 20% split of the training set."""
    samples = _load_bundled_samples()
    rng = np.random.default_rng(0)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    split = int(0.8 * len(samples))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    detector = InjectionDetector(data_path=None, stage2=args.stage2)
    detector.fit(train_samples)

    report = detector.evaluate(test_samples)

    print("## Red-team evaluation (held-out 20% split)\n")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples:  {len(test_samples)}")
    print()
    print("| Category | Detected | Total | Rate |")
    print("| --- | ---: | ---: | ---: |")
    for cat, stats in sorted(report["per_category"].items()):
        rate = stats["detected"] / stats["total"] if stats["total"] else 0.0
        print(f"| {cat} | {stats['detected']} | {stats['total']} | {rate:.1%} |")
    print()
    print(
        f"**Overall attack detection rate:** "
        f"{report['attack_detection_rate']:.1%} "
        f"({report['detected_attacks']}/{report['total_attacks']})"
    )
    print(
        f"**Benign false-positive rate:** "
        f"{report['benign_fpr']:.1%} "
        f"({report['benign_false_positives']}/{report['total_benign']})"
    )

    defence = _defence_from(args)
    if defence is not None:
        print(
            "\n### Full gate on the same held-out samples, without and with "
            f"`{describe(defence)}`\n"
        )
        print("| System | Detection rate on attacks | False-positive rate on benign |")
        print("| --- | --- | --- |")
        for label, gate in (
            ("full gate", SafetyGate(detector=detector)),
            ("full gate + defence", SafetyGate(detector=detector, defence=defence)),
        ):
            hits = [gate.check_input(s.text).action in BLOCKING_ACTIONS for s in test_samples]
            att = Rate(sum(h for h, s in zip(hits, test_samples, strict=True) if s.is_attack),
                       sum(s.is_attack for s in test_samples))  # fmt: skip
            ben = Rate(sum(h for h, s in zip(hits, test_samples, strict=True) if not s.is_attack),
                       sum(not s.is_attack for s in test_samples))  # fmt: skip
            print(f"| {label} | {att.fmt()} | {ben.fmt()} |")
        print()

    target_det = 0.85
    target_fpr = 0.10
    det_ok = report["attack_detection_rate"] >= target_det
    fpr_ok = report["benign_fpr"] <= target_fpr
    print()
    print(
        f"Target detection ≥ {target_det:.0%}: "
        f"{'PASS' if det_ok else 'FAIL'} "
        f"(actual {report['attack_detection_rate']:.1%})"
    )
    print(
        f"Target benign FPR ≤ {target_fpr:.0%}: "
        f"{'PASS' if fpr_ok else 'FAIL'} "
        f"(actual {report['benign_fpr']:.1%})"
    )
    return 0


def cmd_refusal_eval(args: argparse.Namespace) -> int:
    """Run the refusal evaluation harness against the default safety gate."""
    gate = _gate(args)
    evaluator = RefusalEvaluator()
    report = evaluator.evaluate_gate(gate)

    print("## Refusal evaluation\n")
    print(report.summary_table())
    print()
    print("### Adversarial prompts (should refuse)")
    for r in report.results:
        if r.expected_action == "refuse":
            status = "REFUSED" if r.is_refusal else "ALLOWED"
            print(f"  [{status}] {r.prompt[:70]}")
    print()
    print("### Benign-scary prompts (should allow)")
    for r in report.results:
        if r.expected_action == "allow":
            status = "OVER-REFUSED" if r.is_refusal else "ALLOWED"
            print(f"  [{status}] {r.prompt[:70]}")
    return 0


def _as_samples(rows: list[ExternalSample], prefix: str) -> list[Sample]:
    return [Sample(f"{prefix}-{i}", r.text, None, r.is_attack) for i, r in enumerate(rows)]


def _detector_outcome(
    name: str, detector: InjectionDetector, rows: list[ExternalSample]
) -> Outcome:
    return score(name, rows, lambda t: detector.detect(t).is_injection)


def _gate_outcome(
    name: str,
    detector: InjectionDetector,
    rows: list[ExternalSample],
    defence: DefenceConfig | None = None,
) -> Outcome:
    gate = SafetyGate(detector=detector, defence=defence)
    causes: dict[str, str] = {}

    def blocks(text: str) -> bool:
        decision = gate.check_input(text)
        blocked = decision.action in BLOCKING_ACTIONS
        if blocked:
            cause = ", ".join(r for r in decision.rule_ids if r != "default-allow")
            if any(r.startswith("Statistical model") for r in decision.reasons):
                cause += " [stage 2 only]"  # no stage-1 regex fired; logistic regression decided
            causes[text] = cause
        return blocked

    outcome = score(name, rows, blocks)
    outcome.causes = causes
    return outcome


def _print_outcome_table(title: str, outcomes: list[Outcome]) -> None:
    print(f"### {title}\n")
    print("| System | Detection rate on attacks | False-positive rate on benign |")
    print("| --- | --- | --- |")
    for o in outcomes:
        print(f"| {o.name} | {o.detection().fmt()} | {o.fpr().fmt()} |")
    print()


def _print_language_table(outcomes: list[Outcome]) -> None:
    first = outcomes[0]
    # Languages with fewer than 10 prompts are noise; a single language needs no breakdown.
    counts = {
        lang: first.detection(lang).n + first.fpr(lang).n
        for lang in ("en", "de", "other")
        if first.detection(lang).n + first.fpr(lang).n >= 10
    }
    if len(counts) < 2:
        return
    print("### By language (stop-word heuristic, see `guess_language`; shipped systems)\n")
    print("| Language | System | Detection rate on attacks | False-positive rate on benign |")
    print("| --- | --- | --- | --- |")
    for lang in counts:
        for o in outcomes:
            print(f"| {lang} | {o.name} | {o.detection(lang).fmt()} | {o.fpr(lang).fmt()} |")
    print()


def _print_errors(title: str, outcome: Outcome, limit: int) -> None:
    fns, fps = outcome.false_negatives(), outcome.false_positives()
    print(f"### Error analysis: {title}\n")
    print(f"False negatives (attacks let through): {len(fns)}; first {min(limit, len(fns))}:\n")
    for s in fns[:limit]:
        print(f"- [{s.split}] {shorten(s.text)}")
    print(f"\nFalse positives (benign blocked): {len(fps)}; first {min(limit, len(fps))}:\n")
    for s in fps[:limit]:
        cause = f" <- {outcome.causes[s.text]}" if s.text in outcome.causes else ""
        print(f"- [{s.split}] {shorten(s.text)}{cause}")
    if outcome.causes:
        by_rule = Counter(outcome.causes[s.text] for s in fps)
        print("\nBlocking rule(s) behind all false positives:\n")
        for rule, n in by_rule.most_common():
            print(f"- {rule}: {n}")
    print()


def _print_dataset_header(data: LoadedDataset) -> None:
    spec, meta = data.spec, data.meta
    print(f"## External evaluation: {spec.hf_id}\n")
    print(f"- source:   {spec.card_url}")
    print(f"- about:    {spec.description}")
    print(f"- licence:  {spec.license}")
    print(f"- revision: pinned {spec.revision}, observed at download {meta['observed_revision']}")
    print(f"- cached:   downloaded {meta['downloaded']}")
    for split in meta["splits"]:
        rows = data.split(split)
        n_att = sum(r.is_attack for r in rows)
        print(f"- {split}: {len(rows)} rows ({n_att} attacks, {len(rows) - n_att} benign)")
    print()


def cmd_external_eval(args: argparse.Namespace) -> int:
    """Score the shipped detector and gate on a public labelled dataset."""
    data = load_dataset(DATASETS[args.dataset], cache_dir=args.cache_dir, refresh=args.refresh)
    _print_dataset_header(data)

    shipped = _detector(args)
    gate_label = "full gate (`check_input`, refuse/escalate = block)"
    test, everything = data.split("test"), data.samples
    defence = _defence_from(args)
    defended_label = f"full gate + defence `{describe(defence)}`" if defence else ""
    zero_shot_test = [
        _detector_outcome("detector alone", shipped, test),
        _gate_outcome(gate_label, shipped, test),
    ]
    if defence is not None:
        zero_shot_test.append(_gate_outcome(defended_label, shipped, test, defence))
    if args.stage2 == "legacy":
        zero_shot_all = [
            _detector_outcome("detector alone", shipped, everything),
            _gate_outcome(gate_label, shipped, everything),
        ]
        if defence is not None:
            zero_shot_all.append(_gate_outcome(defended_label, shipped, everything, defence))
        print("Legacy stage 2, bundled data only: nothing here was fitted on this data.\n")
        _print_outcome_table(f"Shipped, TEST split only (n={len(test)})", zero_shot_test)
        _print_outcome_table(f"Shipped, TRAIN+TEST (n={len(everything)})", zero_shot_all)
        _print_language_table(zero_shot_all)
        error_sets = [("full gate", zero_shot_all[1]), ("detector alone", zero_shot_all[0])]
        error_scope = "TRAIN+TEST"
    else:
        print(
            "Shipped v2 stage 2 was fitted on the bundled set plus the TRAIN splits of deepset, "
            "jailbreak, itw and neuralchemy (per-class cap, de-duplicated); only the TEST split "
            "below is held out, so these rows are in-domain-plus, not leave-one-out numbers.\n"
        )
        _print_outcome_table(f"Shipped v2, TEST split only (n={len(test)})", zero_shot_test)
        _print_language_table(zero_shot_test)
        error_sets = [("full gate", zero_shot_test[1]), ("detector alone", zero_shot_test[0])]
        error_scope = "TEST"

    if not args.no_retrain:
        train = data.split("train")
        bundled = _load_bundled_samples()
        external_train = _as_samples(train, "ext-train")
        variants: list[tuple[str, list[Sample]]] = [
            ("stage 2 refit on the bundled set + this dataset's TRAIN", bundled + external_train),
            ("stage 2 refit on this dataset's TRAIN only", external_train),
        ]
        outcomes: list[Outcome] = []
        for label, fit_samples in variants:
            if args.stage2 == "v2":
                only_ext = fit_samples is external_train
                det = InjectionDetector(
                    stage2="v2", model=fit_v2([] if only_ext else bundled, train)
                )
            else:
                det = InjectionDetector(stage2="legacy")
                det.fit(fit_samples)
            outcomes.append(_detector_outcome(f"detector alone, {label}", det, test))
            outcomes.append(_gate_outcome(f"full gate, {label}", det, test))
        print("Retrained variants (separate rows: stage 2 fitted on TRAIN, scored on TEST).\n")
        _print_outcome_table(f"RETRAINED on TRAIN, scored on TEST (n={len(test)})", outcomes)

    if args.errors > 0:
        for name, outcome in error_sets:
            _print_errors(f"shipped {name} on {error_scope}", outcome, args.errors)
    return 0


def _load_pools(cache_dir: str, keys: Sequence[str] = DATASET_KEYS) -> dict[str, Pool]:
    pools: dict[str, Pool] = {}
    for key in keys:
        data = load_dataset(DATASETS[key], cache_dir=cache_dir)
        pools[key] = Pool(key, data.split("train"), data.split("test"))
    return pools


def cmd_lodo_report(args: argparse.Namespace) -> int:
    """Leave-one-dataset-out comparison of legacy, v2-single and the pooled v2 (downloads)."""
    bundled = _load_bundled_samples()
    pools = _load_pools(args.cache_dir, LODO_KEYS)
    sections = set(args.section) if args.section else {"splits", "score"}
    if "splits" in sections:
        print("## LODO data and pooled training sets (no scoring)\n")
        print_split_table(bundled, pools, sys.stdout)
    if "score" in sections:
        print("## LODO scores, each held-out TEST split scored once\n")
        print_lodo(lodo_scenarios(bundled, pools, log=sys.stderr), sys.stdout)
    return 0


def cmd_stage2_report(args: argparse.Namespace) -> int:
    """Diagnose the legacy stage 2, run the dev grid, and score the protocol's scenarios."""
    bundled = _load_bundled_samples()
    # Diagnosis and dev grid read TRAIN splits only; only `protocol` scores TEST.
    pools = _load_pools(args.cache_dir)
    sections = set(args.section) if args.section else {"diagnose", "protocol"}
    if "diagnose" in sections:
        print("## Diagnosis of the legacy stage 2 (TRAIN splits only)\n")
        print_diagnosis(bundled, pools, sys.stdout)
    if "dev-grid" in sections:
        print("## Model selection on TRAIN splits only (AUC; TEST splits not read)\n")
        print_dev_grid(dev_grid(bundled, pools, log=sys.stderr), sys.stdout)
        print()
    if "protocol" in sections:
        print("## Protocol scenarios, scored once on TEST splits\n")
        print_scenarios(
            scenarios(bundled, pools, log=sys.stderr),
            sys.stdout,
        )
    return 0


def cmd_stage2_train(args: argparse.Namespace) -> int:
    """Fit the shipped pooled v2 model (bundled + all four TRAIN splits) and save it."""
    bundled = _load_bundled_samples()
    pools = _load_pools(args.cache_dir, LODO_KEYS)
    split = shipped_split(pools)
    model = fit_v2(bundled, split.train)
    meta = {
        "trained_on": ["bundled injection_train.jsonl", *[f"{k} TRAIN" for k in LODO_KEYS]],
        "datasets": {
            k: {
                "hf_id": DATASETS[k].hf_id,
                "revision": DATASETS[k].revision,
                "licence": DATASETS[k].license,
            }
            for k in LODO_KEYS
        },
        "per_class_cap": MAX_TRAIN_PER_CLASS,
        "n_rows": len(bundled) + len(split.train),
    }
    model.save(args.out, meta)
    t = model.thresholds
    print(
        f"wrote {args.out}: {len(bundled)} bundled + {len(split.train)} public rows "
        f"({split.n_attacks} attacks / {split.n_benign} benign; {split.n_duplicate} duplicates "
        f"dropped), review {t.review:.4f}, refuse {t.refuse:.4f}"
    )
    return 0


def _load_json_object(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return data


def _parse_windows(spec: str) -> WindowConfig | None:
    """``none``, ``shipped`` or ``SIZE:STRIDE:THRESHOLD`` (characters, characters, score)."""
    if spec == "none":
        return None
    if spec == "shipped":
        return SHIPPED_WINDOWS
    try:
        size, stride, threshold = spec.split(":")
        return WindowConfig(int(size), int(stride), float(threshold))
    except ValueError as exc:
        raise ValueError(
            f"--windows: none, shipped or SIZE:STRIDE:THRESHOLD, got {spec!r}"
        ) from exc


def _parse_defence(spec: str, windows: str | None = None) -> DefenceConfig | None:
    """The defence for ``--defence`` and ``--windows``; windows follow ``shipped`` by default."""
    if spec == "none":
        components: tuple[str, ...] = ()
    elif spec == "shipped":
        components = SHIPPED_COMPONENTS
    else:
        components = tuple(c for c in spec.split(",") if c)
    win = _parse_windows(
        windows if windows is not None else ("shipped" if spec == "shipped" else "none")
    )
    if not components and win is None:
        return None
    return DefenceConfig.of(components, win)


def _defence_from(args: argparse.Namespace) -> DefenceConfig | None:
    return _parse_defence(args.defence, args.windows)


def _load_attack_rows(
    args: argparse.Namespace, attacks: int, benign: int
) -> tuple[list[Attack], list[str]]:
    """The rows of every dataset in ``--split``; with ``--dev-*``, the sha1-ordered TRAIN head.

    With ``--fresh`` the rows are the fresh pools of ``docs/protocol-window-150.md`` instead.
    """
    if args.fresh:
        return adaptive_rows(load_for_cli(args.cache_dir))
    out_attacks: list[Attack] = []
    out_benign: list[str] = []
    for key in LODO_KEYS:
        rows = load_dataset(DATASETS[key], cache_dir=args.cache_dir).split(args.split)
        att = [r for r in rows if r.is_attack]
        ben = [r for r in rows if not r.is_attack]
        if attacks:
            att, ben = dev_sample(att, attacks), dev_sample(ben, benign)
        elif args.limit:
            att, ben = att[: args.limit], ben[: args.limit]
        out_attacks.extend(Attack(r.text, key) for r in att)
        out_benign.extend(r.text for r in ben)
    return out_attacks, out_benign


def _dev_components(attacks: list[Attack], benign: list[str]) -> None:
    print("## Defence development on TRAIN rows (no TEST row is read)\n")
    print(
        f"{len(attacks)} attacks and {len(benign)} benign prompts "
        f"({DEV_ATTACKS_PER_DATASET} / {DEV_BENIGN_PER_DATASET} per dataset, sha1 order); "
        "both gates; single-family evasions of the attacks each gate blocks unperturbed.\n"
    )
    oracles = {"v2": make_oracle("v2"), "legacy": make_oracle("legacy")}
    steps = select_components(oracles, attacks, benign)
    print("| Component | Evasions newly blocked | Share of all evasions | "
          "Worst benign FPR rise | Kept |")  # fmt: skip
    print("| --- | ---: | ---: | ---: | --- |")
    for st in steps:
        print(
            f"| {st.component} | {st.recovered}/{st.total_evasions} | {st.recovery:.1%} | "
            f"{st.fpr_rise * 100:+.1f} points | {'yes' if st.kept else 'no'} |"
        )
    kept = [st.component for st in steps if st.kept]
    print(f"\nSelected: `{','.join(kept)}`")


def _dev_windows(args: argparse.Namespace) -> None:
    """The Unicode-confusables check and the window grid, TRAIN rows only."""
    gates = {"v2": make_gate_oracle("v2"), "legacy": make_gate_oracle("legacy")}
    # The seven greedy-selected components; `unicode_confusables` is what this run decides on.
    components = tuple(c for c in SHIPPED_COMPONENTS if c != "unicode_confusables")
    attacks, benign = _load_attack_rows(args, DEV_ATTACKS_PER_DATASET, DEV_BENIGN_PER_DATASET)
    print("## Unicode confusables on TRAIN rows (no TEST row is read)\n")
    print(
        f"{len(attacks)} attacks, {len(benign)} benign prompts; both gates; shipped components "
        f"`{', '.join(components)}` with and without `unicode_confusables`.\n"
    )
    uni = check_unicode_confusables(
        {n: g.blocks for n, g in gates.items()},
        attacks,
        benign,
        components,
        lambda t: homoglyph_unseen(t, rng_for("homoglyph_unseen", t, 0)),
    )
    print("| Probe evasions without | Probe evasions with | Attacks caught (both gates) | "
          "Probe evasion drop | Worst benign FPR rise | Kept |")  # fmt: skip
    print("| ---: | ---: | ---: | ---: | ---: | --- |")
    print(
        f"| {uni.evaded_before} | {uni.evaded_after} | {uni.caught} | "
        f"{uni.probe_drop * 100:.1f} points | {uni.fpr_rise * 100:+.1f} points | "
        f"{'yes' if uni.kept else 'no'} |\n"
    )
    if uni.kept:
        components = (*components, "unicode_confusables")

    attacks, benign = _load_attack_rows(
        args, WINDOW_DEV_ATTACKS_PER_DATASET, WINDOW_DEV_BENIGN_PER_DATASET
    )
    print("## Window grid on TRAIN rows (no TEST row is read)\n")
    print(
        f"{len(attacks)} attacks and {len(benign)} benign prompts "
        f"({WINDOW_DEV_ATTACKS_PER_DATASET} / {WINDOW_DEV_BENIGN_PER_DATASET} per dataset, sha1 "
        f"order); both gates pooled; components `{', '.join(components)}`; ASR is the adaptive "
        f"attacker at K = {args.budget} against the defence with windows.\n"
    )
    grid = window_grid(
        args.window_sizes or WINDOW_SIZES,
        args.window_divisors or WINDOW_STRIDE_DIVISORS,
        args.window_thresholds or WINDOW_THRESHOLDS,
    )
    base_asr, steps = select_windows(gates, attacks, benign, components, args.budget, grid)
    print(f"Same defence without windows: ASR {base_asr.fmt()}.\n")
    print("| Size | Stride | Threshold | ASR at K | Worst benign FPR rise | Feasible |")
    print("| ---: | ---: | ---: | --- | ---: | --- |")
    for st in steps:
        c = st.config
        print(
            f"| {c.size} | {c.stride} | {c.threshold:g} | {st.asr.fmt()} | "
            f"{st.fpr_rise * 100:+.1f} points | {'yes' if st.feasible else 'no'} |"
        )
    best = choose_window(steps)
    c = best.config
    print(
        f"\nSelected: size {c.size}, stride {c.stride}, threshold {c.threshold:g} "
        f"(ASR {best.asr.fmt()}, FPR rise {best.fpr_rise * 100:+.1f} points, "
        f"{'feasible' if best.feasible else 'NO CELL FEASIBLE: smallest FPR rise'})."
    )


def cmd_adaptive_eval(args: argparse.Namespace) -> int:
    """Rule-based adaptive attacker against the gate, with and without a normalisation defence."""
    if args.fresh and (args.dev_defence or args.dev_windows):
        print("error: --fresh is a TEST-style run; it cannot combine with --dev-*", file=sys.stderr)
        return 2
    if (args.dev_defence or args.dev_windows) and args.split != "train":
        print("error: --dev-* reads TRAIN rows only; use --split train", file=sys.stderr)
        return 2
    if args.dev_windows:
        _dev_windows(args)
        return 0
    if args.dev_defence:
        _dev_components(*_load_attack_rows(args, DEV_ATTACKS_PER_DATASET, DEV_BENIGN_PER_DATASET))
        return 0

    attacks, benign = _load_attack_rows(args, 0, 0)
    defence = _defence_from(args)
    label = f"the default gate (`--stage2 {args.stage2}`)"
    gate = make_gate_oracle(args.stage2)

    def run(config: DefenceConfig | None) -> Run:
        return run_attacks(
            label,
            "fresh" if args.fresh else args.split,
            gate.blocks,
            attacks,
            benign,
            config,
            budget=args.budget,
            probe=lambda t: homoglyph_unseen(t, rng_for("homoglyph_unseen", t, 0)),
            scorer=gate.score,
            probe_unlisted=lambda t: homoglyph_unlisted(t, rng_for("homoglyph_unlisted", t, 0)),
        )

    result = run(defence)
    result.print(sys.stdout)
    if result.defended_eval is not None:
        print(f"Verdict (rule fixed before scoring): **{verdict(result)}**.")
    if args.compare_defence is not None:
        previous = _parse_defence(args.compare_defence, "none")
        if previous is None:
            print("error: --compare-defence needs a defence, not none", file=sys.stderr)
            return 2
        ok, drop, rise = adoption(run(previous), result)
        print(
            f"\nAdoption rule against `{describe(previous)}` "
            f"(fixed before scoring): ASR at K = {args.budget} falls by {drop * 100:.1f} points "
            f"(needs {ADOPT_MIN_ASR_DROP * 100:.0f}), benign FPR rises by {rise * 100:+.1f} "
            f"points (allows {ADOPT_MAX_FPR_RISE * 100:.0f}): **{'adopt' if ok else 'reject'}**."
        )
    return 0


def _benchmark_guard(args: argparse.Namespace) -> int:
    """``benchmark --suite injecguard --guard anthropic:<model>`` (``docs/protocol-llm.md``)."""
    if args.suite != "injecguard":
        print("error: --guard needs --suite injecguard", file=sys.stderr)
        return 2
    try:
        model = parse_agent_spec(args.guard)
        client = _llm_client(args, model, "guard.jsonl")
        data = load_guard_data(args.cache_dir)
        run = run_guard(client, data)
    except (OSError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"# {model} as a zero-shot injection guard\n")
    print(format_guard_report(run, data, client.tag(), _usage_line(client), model))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Score the gate on public sets next to published guards (``docs/protocol-sota.md``)."""
    if args.guard:
        return _benchmark_guard(args)
    stages = tuple(STAGE2_MODES) if args.stage2 == "both" else (args.stage2,)
    suites, public = resolve_suites(args.suite)
    if suites:
        fresh = load_for_cli(args.cache_dir, args.refresh)
        for stage2 in stages:
            print_benchmark(stage2, fresh, run_benchmark(stage2, fresh, suites), suites)
    if public:
        data, scores = run_sota(stages, public, args.cache_dir, args.refresh)
        print_sota(data, scores, public)
    return 0


def cmd_guard_dev(args: argparse.Namespace) -> int:
    """Score the zero-shot LLM guard prompt on the 50-item dev set (bundled training data)."""
    try:
        model = parse_agent_spec(args.guard)
        client = _llm_client(args, model, "guard-dev.jsonl")
        recall, benign, bad = score_dev(client)
    except (OSError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"dev set: {len(dev_set())} items of injection_train.jsonl (25 attacks, 25 benign)")
    print(f"attack recall:   {recall.fmt()}")
    print(f"benign accuracy: {benign.fmt()}")
    print(f"unparseable:     {bad}")
    print(_usage_line(client))
    return 0


def cmd_window_eval(args: argparse.Namespace) -> int:
    """Windows 150:37:0 against the shipped defence on the fresh data; applies the fixed rule."""
    fresh = load_for_cli(args.cache_dir, args.refresh)
    print_window_report(window_protocol(args.stage2, fresh, args.budget), fresh)
    return 0


def cmd_spec_check(args: argparse.Namespace) -> int:
    """Check an emitted constraint spec against a contract-bounds policy.

    Exit code: 0 for allow/warn, 1 for a blocking action, 2 for unreadable input.
    """
    try:
        policy = load_spec_policy(Path(args.policy))
        spec = _load_json_object(args.spec)
        previous = _load_json_object(args.previous) if args.previous else None
        decision = check_spec(spec, policy, previous)
    except (OSError, ValueError) as exc:  # JSONDecodeError and SpecPolicyError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        print(f"action:   {decision.action}")
        print(f"rules:    {', '.join(decision.rule_ids) or 'none'}")
        print(f"delta:    {'checked' if decision.metadata['delta_checked'] else 'skipped'}")
        for reason in decision.reasons:
            print(f"  - {reason}")
    return 1 if decision.action in BLOCKING_ACTIONS else 0


def _load_tool_call(path: str) -> tuple[ToolCall, dict[str, list[str]], Session | None]:
    """Parse a ``tool-check`` input: ``tool``, ``args``, ``labels`` and an optional ``session``."""
    data = _load_json_object(path)
    unknown = set(data) - {"tool", "args", "labels", "session"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    call = ToolCall.from_dict(data)
    labels = data.get("labels", {})
    if not isinstance(labels, dict) or not all(
        isinstance(v, list) and all(isinstance(x, str) for x in v) for v in labels.values()
    ):
        raise ValueError(f"{path}: labels must map argument names to lists of label strings")
    session = None
    if "session" in data:
        raw = data["session"]
        if not isinstance(raw, dict) or set(raw) - {"labels", "consequential"}:
            raise ValueError(f"{path}: session takes only labels and consequential")
        consequential = raw.get("consequential", False)
        session_labels = raw.get("labels", [])
        if not isinstance(consequential, bool) or not isinstance(session_labels, list):
            raise ValueError(f"{path}: session.labels is a list, session.consequential a bool")
        session = Session()
        session.observe(session_labels)
        session.consequential = consequential
    return call, labels, session


def cmd_tool_check(args: argparse.Namespace) -> int:
    """Check one proposed tool call, with provenance labels, against a tool policy.

    Exit code: 0 for allow/warn, 1 for a blocking action, 2 for unreadable input.
    """
    try:
        policy = load_tool_policy(Path(args.policy))
        call, labels, session = _load_tool_call(args.call)
        decision = check_tool_call(call, labels, policy, session)
    except (OSError, ValueError) as exc:  # JSONDecodeError, ToolPolicyError, LabelError
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        two = decision.metadata.get("rule_of_two")
        print(f"action:   {decision.action}")
        print(f"tool:     {call.tool} ({decision.metadata['capability']})")
        print(f"rules:    {', '.join(decision.rule_ids) or 'none'}")
        print(f"session:  {'not checked' if two is None else json.dumps(two)}")
        for reason in decision.reasons:
            print(f"  - {reason}")
    return 1 if decision.action in BLOCKING_ACTIONS else 0


def _llm_client(args: argparse.Namespace, model: str, default_cache: str) -> AnthropicClient:
    """The cached client selected by ``--llm-cache`` and ``--llm-offline``."""
    path = Path(args.llm_cache or Path(DEFAULT_CACHE_DIR) / default_cache)
    return AnthropicClient(path, model=model, offline=args.llm_offline)


def _usage_line(client: AnthropicClient) -> str:
    calls, inp, out = totals([client])
    served = ", ".join(sorted(client.served_models())) or "none"
    return (
        f"Cache `{client.cache_path}`: {calls} API calls, {inp} input and {out} output tokens in "
        f"total; model id returned by the API: {served}. This invocation: {client.stats.line()}."
    )


def _tool_eval_llm(args: argparse.Namespace, policy: ToolPolicy | None) -> int:
    """``tool-eval --agent anthropic:<model>``: the model as the agent (``protocol-llm.md``)."""
    try:
        model = parse_agent_spec(args.agent)
        client = _llm_client(args, model, "tool-eval-dev.jsonl" if args.dev else "tool-eval.jsonl")
        if args.dev:
            for run in run_dev(client, policy):
                print(json.dumps(run.to_json()))
            print(_usage_line(client), file=sys.stderr)
            return 0
        result = run_llm_suite(client, policy)
    except (OSError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.cases:
        for runs in result.attacks.values():
            for r in runs:
                print(json.dumps(r.to_json()))
        return 0
    print(f"# Tool-call layer: {model} as the agent\n")
    for name in LLM_CONDITIONS:
        print(f"- `{name}`: {CONDITION_HELP[name]}")
    print()
    print(format_llm_report(result, client.tag(), _usage_line(client)))
    return 0


def cmd_tool_eval(args: argparse.Namespace) -> int:
    """Run the synthetic tool-call scenario suite (``docs/protocol-toolcheck.md``)."""
    try:
        policy = load_tool_policy(Path(args.policy)) if args.policy else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.agent:
        return _tool_eval_llm(args, policy)
    result = run_suite(policy)
    if args.cases:
        for cond, runs in result.attacks.items():
            for r in runs:
                print(json.dumps({"condition": cond, **asdict(r)}))
        return 0
    print("# Tool-call layer: synthetic scenario suite\n")
    for name, text in CONDITION_HELP.items():
        print(f"- `{name}`: {text}")
    print()
    print(format_report(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="safegate",
        description="Constitutional safety gate for LLM input/output filtering",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    stage2_opt = argparse.ArgumentParser(add_help=False)
    stage2_opt.add_argument(
        "--stage2",
        choices=STAGE2_MODES,
        default=DEFAULT_STAGE2,
        help="stage-2 model: v2 (default; pooled, calibrated, abstain band) or legacy",
    )

    defence_opt = argparse.ArgumentParser(add_help=False)
    defence_opt.add_argument(
        "--defence",
        default="none",
        help="input defence in front of the gate: none (default), shipped, or a comma list of "
        f"{','.join(COMPONENTS)}; `shipped` includes the shipped windows",
    )
    defence_opt.add_argument(
        "--windows",
        default=None,
        help="windowed scanning: none, shipped or SIZE:STRIDE:THRESHOLD "
        "(default: shipped with --defence shipped, otherwise none)",
    )

    p_check = sub.add_parser(
        "check",
        parents=[stage2_opt, defence_opt],
        help="Check a single text input against the safety gate",
    )
    p_check.add_argument("text", type=str, help="Text to check")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser(
        "run",
        parents=[stage2_opt, defence_opt],
        help="Run a prompt through pre-gen, during-gen and post-gen (mock LLM)",
    )
    p_run.add_argument("prompt", type=str, help="Prompt text")
    p_run.add_argument("--trace", action="store_true", help="Print the per-request JSON trace")
    p_run.set_defaults(func=cmd_run)

    p_red = sub.add_parser(
        "redteam",
        parents=[stage2_opt, defence_opt],
        help="Run red-team detection evaluation on held-out samples",
    )
    p_red.set_defaults(func=cmd_redteam)

    p_ref = sub.add_parser(
        "refusal-eval",
        parents=[stage2_opt, defence_opt],
        help="Evaluate refusal rate, over-refusal rate, and refusal quality",
    )
    p_ref.set_defaults(func=cmd_refusal_eval)

    p_ext = sub.add_parser(
        "external-eval",
        parents=[stage2_opt, defence_opt],
        help="Evaluate the shipped detector and gate on a public labelled dataset (downloads)",
    )
    p_ext.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p_ext.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_ext.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p_ext.add_argument("--no-retrain", action="store_true", help="skip the retrained variants")
    p_ext.add_argument("--errors", type=int, default=5, help="false negatives/positives to list")
    p_ext.set_defaults(func=cmd_external_eval)

    p_rep = sub.add_parser(
        "stage2-report",
        help="Diagnose legacy stage 2, select v2 on TRAIN, score the protocol on TEST (needs data)",
    )
    p_rep.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_rep.add_argument(
        "--section",
        action="append",
        choices=("diagnose", "dev-grid", "protocol"),
        help="repeatable; default: diagnose + protocol (dev-grid takes about 10 minutes)",
    )
    p_rep.set_defaults(func=cmd_stage2_report)

    p_trn = sub.add_parser(
        "stage2-train",
        help="Fit the shipped pooled v2 model on bundled + all four TRAIN splits and save it",
    )
    p_trn.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_trn.add_argument("--out", default="src/safegate/data/stage2_v2.npz")
    p_trn.set_defaults(func=cmd_stage2_train)

    p_lodo = sub.add_parser(
        "lodo-report",
        help="Leave-one-dataset-out: legacy vs v2-single vs pooled v2 on four public datasets",
    )
    p_lodo.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_lodo.add_argument(
        "--section",
        action="append",
        choices=("splits", "score"),
        help="repeatable; default: splits + score ('splits' prints sizes only and scores nothing)",
    )
    p_lodo.set_defaults(func=cmd_lodo_report)

    p_adv = sub.add_parser(
        "adaptive-eval",
        parents=[stage2_opt],
        help="Rule-based adaptive attacker (homoglyphs, encodings, padding, ...) vs the gate",
    )
    p_adv.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_adv.add_argument("--split", choices=("test", "train"), default="test")
    p_adv.add_argument("--budget", type=int, default=BUDGET, help="adaptive queries per attack")
    p_adv.add_argument(
        "--defence",
        default="shipped",
        help=f"defence: shipped (default), none, or a comma list of {','.join(COMPONENTS)}",
    )
    p_adv.add_argument(
        "--windows",
        default=None,
        help="windowed scanning: none, shipped or SIZE:STRIDE:THRESHOLD "
        "(default: shipped with --defence shipped, otherwise none)",
    )
    p_adv.add_argument(
        "--compare-defence",
        default=None,
        help="also run this defence (components, no windows) and apply the adoption rule to it",
    )
    p_adv.add_argument(
        "--dev-defence",
        action="store_true",
        help="select defence components on a TRAIN subsample (needs --split train)",
    )
    p_adv.add_argument(
        "--window-sizes",
        type=lambda v: [int(x) for x in v.split(",")],
        help="--dev-windows grid: comma list of sizes (default 150,300,600)",
    )
    p_adv.add_argument(
        "--window-divisors",
        type=lambda v: [int(x) for x in v.split(",")],
        help="--dev-windows grid: stride = size // divisor, comma list (default 2,4)",
    )
    p_adv.add_argument(
        "--window-thresholds",
        type=lambda v: [float(x) for x in v.split(",")],
        help="--dev-windows grid: comma list of thresholds (default 0,0.55,0.75,0.95)",
    )
    p_adv.add_argument(
        "--dev-windows",
        action="store_true",
        help="check unicode_confusables and select the window cell on TRAIN (needs --split train)",
    )
    p_adv.add_argument("--limit", type=int, default=0, help="first N rows per dataset and class")
    p_adv.add_argument(
        "--fresh",
        action="store_true",
        help="score the fresh pools of docs/protocol-window-150.md instead of the --split rows",
    )
    p_adv.set_defaults(func=cmd_adaptive_eval)

    p_bench = sub.add_parser(
        "benchmark",
        help="Score the gate on public sets, next to the published scores of other guards",
    )
    p_bench.add_argument(
        "--stage2",
        choices=(*STAGE2_MODES, "both"),
        default="both",
        help="stage-2 model: v2 (shipped default), legacy or both (default)",
    )
    p_bench.add_argument(
        "--suite",
        choices=(*SUITES, *SOTA_KEYS, "injecguard", "public", "all"),
        default="all",
        help="one set, `injecguard` (the paper's three), `public` (the further sets) or all",
    )
    p_bench.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_bench.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p_bench.add_argument(
        "--guard",
        help="score a model as a zero-shot guard instead of safegate, e.g. "
        "anthropic:claude-sonnet-5-5 (with --suite injecguard; needs ANTHROPIC_API_KEY unless "
        "--llm-offline)",
    )
    p_bench.add_argument("--llm-cache", help="response cache file (default data/llm_cache/...)")
    p_bench.add_argument(
        "--llm-offline", action="store_true", help="read the cache only: no key, no network"
    )
    p_bench.set_defaults(func=cmd_benchmark)

    p_gdev = sub.add_parser(
        "guard-dev",
        help="Zero-shot LLM guard prompt on the 50-item dev set (bundled training data only)",
    )
    p_gdev.add_argument("--guard", default="anthropic:claude-sonnet-5-5", help="anthropic:<model>")
    p_gdev.add_argument("--llm-cache", help="response cache file (default data/llm_cache/...)")
    p_gdev.add_argument(
        "--llm-offline", action="store_true", help="read the cache only: no key, no network"
    )
    p_gdev.set_defaults(func=cmd_guard_dev)

    p_win = sub.add_parser(
        "window-eval",
        parents=[stage2_opt],
        help="Windows 150:37:0 against the shipped defence on fresh data, with the fixed rule",
    )
    p_win.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_win.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p_win.add_argument("--budget", type=int, default=BUDGET, help="adaptive queries per attack")
    p_win.set_defaults(func=cmd_window_eval)

    p_spec = sub.add_parser(
        "spec-check",
        help="Check an LLM-emitted constraint spec (JSON) against a contract-bounds policy (YAML)",
    )
    p_spec.add_argument("spec", help="emitted spec JSON file")
    p_spec.add_argument("--policy", required=True, help="spec policy YAML file")
    p_spec.add_argument("--previous", help="last trusted spec JSON, enables the delta bounds")
    p_spec.add_argument("--json", action="store_true", help="print the Decision as JSON")
    p_spec.set_defaults(func=cmd_spec_check)

    p_tool = sub.add_parser(
        "tool-check",
        help="Check a tool call with provenance labels (JSON) against a tool policy (YAML)",
    )
    p_tool.add_argument("call", help="call JSON: tool, args, labels, optional session")
    p_tool.add_argument("--policy", required=True, help="tool policy YAML file")
    p_tool.add_argument("--json", action="store_true", help="print the Decision as JSON")
    p_tool.set_defaults(func=cmd_tool_check)

    p_teval = sub.add_parser(
        "tool-eval",
        help="Run the synthetic home-battery scenario suite for the tool-call layer",
    )
    p_teval.add_argument("--policy", help="tool policy YAML (default: examples/tools/energy.yaml)")
    p_teval.add_argument(
        "--cases", action="store_true", help="print every attack run as JSON lines"
    )
    p_teval.add_argument(
        "--agent",
        help="a real model as the agent instead of the scripted oracle, e.g. "
        "anthropic:claude-sonnet-5-5 (needs ANTHROPIC_API_KEY unless --llm-offline)",
    )
    p_teval.add_argument("--llm-cache", help="response cache file (default data/llm_cache/...)")
    p_teval.add_argument(
        "--llm-offline", action="store_true", help="read the cache only: no key, no network"
    )
    p_teval.add_argument(
        "--dev", action="store_true", help="with --agent: the three dev cases (plumbing check)"
    )
    p_teval.set_defaults(func=cmd_tool_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the safegate console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

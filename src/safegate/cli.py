"""Command-line interface for the safegate safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from safegate.external import (
    DATASETS,
    ExternalSample,
    LoadedDataset,
    Outcome,
    load_dataset,
    score,
    shorten,
)
from safegate.gate import SafetyGate
from safegate.injection import InjectionDetector
from safegate.mock_llm import MockStreamingLLM
from safegate.refusal import RefusalEvaluator
from safegate.spec import check_spec, load_spec_policy
from safegate.taxonomy import Sample


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


def cmd_check(args: argparse.Namespace) -> int:
    """Check a single piece of text against the safety gate."""
    gate = SafetyGate()
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
    gate = SafetyGate()
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

    detector = InjectionDetector(data_path=None)
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
    gate = SafetyGate()
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


BLOCKING_ACTIONS = frozenset({"refuse", "escalate"})


def _as_samples(rows: list[ExternalSample], prefix: str) -> list[Sample]:
    return [Sample(f"{prefix}-{i}", r.text, None, r.is_attack) for i, r in enumerate(rows)]


def _detector_outcome(
    name: str, detector: InjectionDetector, rows: list[ExternalSample]
) -> Outcome:
    return score(name, rows, lambda t: detector.detect(t).is_injection)


def _gate_outcome(name: str, detector: InjectionDetector, rows: list[ExternalSample]) -> Outcome:
    gate = SafetyGate(detector=detector)
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

    shipped = InjectionDetector()
    test, everything = data.split("test"), data.samples
    zero_shot_all = [
        _detector_outcome("detector alone", shipped, everything),
        _gate_outcome("full gate (`check_input`, refuse/escalate = block)", shipped, everything),
    ]
    zero_shot_test = [
        _detector_outcome("detector alone", shipped, test),
        _gate_outcome("full gate (`check_input`, refuse/escalate = block)", shipped, test),
    ]
    print("Shipped detector and gate: nothing here was fitted on this data.\n")
    _print_outcome_table(f"Shipped, TEST split only (n={len(test)})", zero_shot_test)
    _print_outcome_table(f"Shipped, TRAIN+TEST (n={len(everything)})", zero_shot_all)
    _print_language_table(zero_shot_all)

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
            det = InjectionDetector()
            det.fit(fit_samples)
            outcomes.append(_detector_outcome(f"detector alone, {label}", det, test))
            outcomes.append(_gate_outcome(f"full gate, {label}", det, test))
        print("Retrained variants (separate rows: stage 2 fitted on TRAIN, scored on TEST).\n")
        _print_outcome_table(f"RETRAINED on TRAIN, scored on TEST (n={len(test)})", outcomes)

    if args.errors > 0:
        _print_errors("shipped full gate on TRAIN+TEST", zero_shot_all[1], args.errors)
        _print_errors("shipped detector alone on TRAIN+TEST", zero_shot_all[0], args.errors)
    return 0


def _load_json_object(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return data


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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="safegate",
        description="Constitutional safety gate for LLM input/output filtering",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check a single text input against the safety gate")
    p_check.add_argument("text", type=str, help="Text to check")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser(
        "run", help="Run a prompt through pre-gen, during-gen and post-gen (mock LLM)"
    )
    p_run.add_argument("prompt", type=str, help="Prompt text")
    p_run.add_argument("--trace", action="store_true", help="Print the per-request JSON trace")
    p_run.set_defaults(func=cmd_run)

    p_red = sub.add_parser(
        "redteam",
        help="Run red-team detection evaluation on held-out samples",
    )
    p_red.set_defaults(func=cmd_redteam)

    p_ref = sub.add_parser(
        "refusal-eval",
        help="Evaluate refusal rate, over-refusal rate, and refusal quality",
    )
    p_ref.set_defaults(func=cmd_refusal_eval)

    p_ext = sub.add_parser(
        "external-eval",
        help="Evaluate the shipped detector and gate on a public labelled dataset (downloads)",
    )
    p_ext.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p_ext.add_argument("--cache-dir", default="data/raw", help="download cache (gitignored)")
    p_ext.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p_ext.add_argument("--no-retrain", action="store_true", help="skip the retrained variants")
    p_ext.add_argument("--errors", type=int, default=5, help="false negatives/positives to list")
    p_ext.set_defaults(func=cmd_external_eval)

    p_spec = sub.add_parser(
        "spec-check",
        help="Check an LLM-emitted constraint spec (JSON) against a contract-bounds policy (YAML)",
    )
    p_spec.add_argument("spec", help="emitted spec JSON file")
    p_spec.add_argument("--policy", required=True, help="spec policy YAML file")
    p_spec.add_argument("--previous", help="last trusted spec JSON, enables the delta bounds")
    p_spec.add_argument("--json", action="store_true", help="print the Decision as JSON")
    p_spec.set_defaults(func=cmd_spec_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the safegate console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

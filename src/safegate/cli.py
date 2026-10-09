"""Command-line interface for the safegate safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from safegate.gate import SafetyGate
from safegate.injection import InjectionDetector
from safegate.refusal import RefusalEvaluator
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the safegate console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

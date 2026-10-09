"""End-to-end safety gate integrating injection detection, classification, and stream filtering."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from safegate.classifier import (
    ContentClassifier,
    KeywordClassifier,
)
from safegate.injection import InjectionDetector
from safegate.rules import (
    Rule,
    RulesEngine,
    get_default_rules_path,
)


@dataclass(frozen=True)
class Decision:
    """Safety policy decision for a scanned input or output."""

    action: str  # "allow", "redact", "refuse", "escalate"
    rule_ids: list[str]
    reasons: list[str]
    sanitized_text: str | None = None
    category: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision to a dictionary."""
        return asdict(self)


class SafetyGate:
    """Safety gate: multi-stage input validation plus streaming output protection."""

    def __init__(
        self,
        rules: list[Rule] | Path | str | None = None,
        detector: InjectionDetector | None = None,
        classifier: ContentClassifier | None = None,
        audit_log_path: Path | str | None = None,
        buffer_size: int = 64,
    ) -> None:
        if rules is None:
            default_path = get_default_rules_path()
            self.rules_engine = RulesEngine.from_yaml(default_path)
        elif isinstance(rules, (Path, str)):
            self.rules_engine = RulesEngine.from_yaml(rules)
        elif isinstance(rules, list):
            self.rules_engine = RulesEngine(rules)
        else:
            raise TypeError("rules must be a list of Rule instances, a Path, or None")

        self.detector = detector or InjectionDetector()
        self.classifier = classifier or KeywordClassifier()
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.buffer_size = buffer_size
        self.audit_records: list[dict[str, Any]] = []

    def _log_decision(self, target: str, text: str, decision: Decision) -> None:
        """Record an audit trail entry in memory and append to JSONL if configured."""
        record = {
            "timestamp": time.time(),
            "target": target,
            "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
            "action": decision.action,
            "rule_ids": decision.rule_ids,
            "reasons": decision.reasons,
            "category": decision.category,
            "score": decision.score,
        }
        self.audit_records.append(record)
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def check_input(self, text: str) -> Decision:
        """Scan input with the injection detector, content classifier and rules."""
        detection = self.detector.detect(text)
        classification = self.classifier.classify(text)

        rule_result = self.rules_engine.evaluate(
            text=text,
            target="input",
            detection=detection,
            classification=classification,
        )

        rule_ids = [r.id for r in rule_result.matched_rules]
        reasons = list(rule_result.reasons)

        # Include detector reasons if injection was flagged
        if detection.is_injection and detection.reasons:
            reasons.extend(detection.reasons)

        # Include classifier reasons if flagged
        if classification.flagged and classification.reasons:
            reasons.extend(classification.reasons)

        cat_val = detection.category.value if detection.category else None
        if not cat_val and classification.categories:
            cat_val = max(classification.categories, key=lambda k: classification.categories[k])

        decision = Decision(
            action=rule_result.action,
            rule_ids=rule_ids,
            reasons=list(dict.fromkeys(reasons)),  # deduplicate preserving order
            sanitized_text=rule_result.redacted_text if rule_result.action == "redact" else None,
            category=cat_val,
            score=detection.score,
            metadata={
                "detection_score": detection.score,
                "classifier_flagged": classification.flagged,
            },
        )

        self._log_decision("input", text, decision)
        return decision

    def stream_output(self, chunks: Iterable[str]) -> Iterator[str]:
        """Filter a stream through a sliding buffer; stop early on a mid-stream refuse."""
        buf = ""
        withheld_msg = "\n[response withheld by safety gate]"

        output_rules = [r for r in self.rules_engine.rules if r.applies_to in ("output", "both")]

        def evaluate_buffer(content: str) -> tuple[str, list[Rule], str]:
            """Check rules on buffer. Returns (action, matching_rules, redacted_content)."""
            redacted = content
            violating_rules: list[Rule] = []
            highest_action = "allow"
            action_rank = {"refuse": 4, "escalate": 3, "redact": 2, "allow": 1}

            for rule in output_rules:
                if rule.matches(redacted):
                    violating_rules.append(rule)
                    if action_rank[rule.action] > action_rank[highest_action]:
                        highest_action = rule.action
                    if rule.action == "redact":
                        if rule.match.regex:
                            redacted = re.sub(
                                rule.match.regex,
                                f"[REDACTED:{rule.id}]",
                                redacted,
                            )
                        else:
                            redacted = f"[REDACTED:{rule.id}]"

            return highest_action, violating_rules, redacted

        for chunk in chunks:
            buf += chunk
            action, matched_rules, redacted_buf = evaluate_buffer(buf)

            if action == "refuse":
                dec = Decision(
                    action="refuse",
                    rule_ids=[r.id for r in matched_rules],
                    reasons=[f"Output safety violation: {r.description}" for r in matched_rules],
                )
                self._log_decision("output", buf, dec)
                yield withheld_msg
                return

            if action == "redact":
                buf = redacted_buf

            if len(buf) > self.buffer_size:
                emit_len = len(buf) - self.buffer_size
                to_emit = buf[:emit_len]
                buf = buf[emit_len:]
                yield to_emit

        # Final flush on stream exhaustion
        if buf:
            action, matched_rules, redacted_buf = evaluate_buffer(buf)
            if action == "refuse":
                dec = Decision(
                    action="refuse",
                    rule_ids=[r.id for r in matched_rules],
                    reasons=[f"Output safety violation: {r.description}" for r in matched_rules],
                )
                self._log_decision("output", buf, dec)
                yield withheld_msg
                return
            if action == "redact":
                buf = redacted_buf
            yield buf

    def wrap(
        self,
        llm_fn: Callable[[str], Iterable[str]],
    ) -> Callable[[str], Iterator[str]]:
        """Wrap an LLM text-to-stream generation function with input and output safety gates."""

        def wrapped_llm(prompt: str) -> Iterator[str]:
            input_decision = self.check_input(prompt)

            if input_decision.action == "refuse":
                yield (
                    "I cannot fulfill this request due to safety policies: "
                    + ", ".join(input_decision.reasons)
                )
                return

            if input_decision.action == "escalate":
                yield (
                    "This request has been escalated for administrative review: "
                    + ", ".join(input_decision.reasons)
                )
                return

            effective_prompt = (
                input_decision.sanitized_text
                if input_decision.action == "redact" and input_decision.sanitized_text
                else prompt
            )

            raw_stream = llm_fn(effective_prompt)
            yield from self.stream_output(raw_stream)

        return wrapped_llm

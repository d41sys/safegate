"""End-to-end safety gate integrating injection detection, classification, and stream filtering."""

from __future__ import annotations

import json
import re
import time
import uuid
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
    ACTION_PRECEDENCE,
    Rule,
    RulesEngine,
    get_default_rules_path,
)

WITHHELD_MESSAGE = "\n[response withheld by safety gate]"
_SEVERITY: dict[str, int] = {action: rank for action, rank in ACTION_PRECEDENCE.items()}


@dataclass
class Checkpoint:
    """One gate checkpoint in a request: what it saw and what it decided."""

    stage: str  # "pre_gen", "during_gen", "post_gen"
    action: str
    rule_ids: list[str]
    signals: dict[str, Any]
    elapsed_ms: float


@dataclass
class RequestTrace:
    """Per-request audit trace across the three checkpoints (lesson 87)."""

    request_id: str
    prompt_preview: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    final_action: str = "allow"
    total_ms: float = 0.0
    started: float = field(default_factory=time.perf_counter, repr=False)

    def elapsed_ms(self) -> float:
        """Milliseconds since the request started."""
        return (time.perf_counter() - self.started) * 1000

    def record(self, checkpoint: Checkpoint) -> None:
        """Append a checkpoint and raise the final action if it is more severe."""
        self.checkpoints.append(checkpoint)
        if _SEVERITY[checkpoint.action] > _SEVERITY[self.final_action]:
            self.final_action = checkpoint.action

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible types (without the internal clock)."""
        data = asdict(self)
        del data["started"]
        return data


@dataclass(frozen=True)
class Decision:
    """Safety policy decision for a scanned input or output."""

    action: str  # "allow", "warn", "redact", "refuse" (= block), "escalate"
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

    def stream_output(
        self, chunks: Iterable[str], trace: RequestTrace | None = None
    ) -> Iterator[str]:
        """Filter a stream through a sliding buffer; stop early on a mid-stream refuse.

        With ``trace``, every chunk that triggers an output rule is recorded as a
        ``during_gen`` checkpoint.
        """
        buf = ""

        output_rules = [r for r in self.rules_engine.rules if r.applies_to in ("output", "both")]

        def evaluate_buffer(content: str) -> tuple[str, list[Rule], str]:
            """Check rules on buffer. Returns (action, matching_rules, redacted_content)."""
            redacted = content
            violating_rules: list[Rule] = []
            highest_action = "allow"

            for rule in output_rules:
                if rule.matches(redacted):
                    violating_rules.append(rule)
                    if _SEVERITY[rule.action] > _SEVERITY[highest_action]:
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

            # Catch-all allow rules match everything; keep them out of decisions and traces.
            return highest_action, [r for r in violating_rules if r.action != "allow"], redacted

        def note(action: str, rules: list[Rule], chunk_index: int) -> None:
            if trace is not None and action != "allow":
                trace.record(
                    Checkpoint(
                        stage="during_gen",
                        action=action,
                        rule_ids=[r.id for r in rules],
                        signals={"chunk_index": chunk_index},
                        elapsed_ms=trace.elapsed_ms(),
                    )
                )

        index = -1
        for index, chunk in enumerate(chunks):
            buf += chunk
            action, matched_rules, redacted_buf = evaluate_buffer(buf)
            note(action, matched_rules, index)

            if action == "refuse":
                dec = Decision(
                    action="refuse",
                    rule_ids=[r.id for r in matched_rules],
                    reasons=[f"Output safety violation: {r.description}" for r in matched_rules],
                )
                self._log_decision("output", buf, dec)
                yield WITHHELD_MESSAGE
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
                note(action, matched_rules, index)
                dec = Decision(
                    action="refuse",
                    rule_ids=[r.id for r in matched_rules],
                    reasons=[f"Output safety violation: {r.description}" for r in matched_rules],
                )
                self._log_decision("output", buf, dec)
                yield WITHHELD_MESSAGE
                return
            if action == "redact":
                buf = redacted_buf
            yield buf

    @staticmethod
    def _input_block_message(decision: Decision) -> str | None:
        """Text returned instead of calling the model, or None when the model may run."""
        if decision.action == "refuse":
            return "I cannot fulfill this request due to safety policies: " + ", ".join(
                decision.reasons
            )
        if decision.action == "escalate":
            return "This request has been escalated for administrative review: " + ", ".join(
                decision.reasons
            )
        return None

    @staticmethod
    def _effective_prompt(prompt: str, decision: Decision) -> str:
        if decision.action == "redact" and decision.sanitized_text:
            return decision.sanitized_text
        return prompt

    def run(self, prompt: str, llm_fn: Callable[[str], Iterable[str]]) -> tuple[str, RequestTrace]:
        """Run one request through pre-gen, during-gen and post-gen checks.

        Returns the text the caller may show and the per-request trace.
        """
        trace = RequestTrace(request_id=uuid.uuid4().hex, prompt_preview=prompt[:100])
        elapsed = trace.elapsed_ms
        decision = self.check_input(prompt)
        trace.record(
            Checkpoint(
                stage="pre_gen",
                action=decision.action,
                rule_ids=decision.rule_ids,
                signals={
                    "detector_score": decision.score,
                    "category": decision.category,
                    "classifier_flagged": decision.metadata.get("classifier_flagged", False),
                },
                elapsed_ms=elapsed(),
            )
        )
        blocked = self._input_block_message(decision)
        if blocked is not None:
            trace.total_ms = elapsed()
            return blocked, trace

        stream = llm_fn(self._effective_prompt(prompt, decision))
        output = "".join(self.stream_output(stream, trace))
        if output.endswith(WITHHELD_MESSAGE):  # a safe prefix may precede it
            trace.total_ms = elapsed()
            return output, trace

        classification = self.classifier.classify(output)
        result = self.rules_engine.evaluate(output, "output", classification=classification)
        trace.record(
            Checkpoint(
                stage="post_gen",
                action=result.action,
                rule_ids=[r.id for r in result.matched_rules],
                signals={
                    "classifier_flagged": classification.flagged,
                    "classifier_max": max(classification.categories.values(), default=0.0),
                },
                elapsed_ms=elapsed(),
            )
        )
        if result.action in ("refuse", "escalate"):
            output = WITHHELD_MESSAGE
        elif result.action == "redact" and result.redacted_text is not None:
            output = result.redacted_text
        trace.total_ms = elapsed()
        return output, trace

    def wrap(
        self,
        llm_fn: Callable[[str], Iterable[str]],
    ) -> Callable[[str], Iterator[str]]:
        """Wrap an LLM text-to-stream generation function with input and output safety gates."""

        def wrapped_llm(prompt: str) -> Iterator[str]:
            input_decision = self.check_input(prompt)
            blocked = self._input_block_message(input_decision)
            if blocked is not None:
                yield blocked
                return
            raw_stream = llm_fn(self._effective_prompt(prompt, input_decision))
            yield from self.stream_output(raw_stream)

        return wrapped_llm

"""End-to-end safety gate integrating injection detection, classification, and stream filtering."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from safegate.classifier import (
    ContentClassifier,
    KeywordClassifier,
)
from safegate.defence import DefenceConfig, scan_texts
from safegate.injection import InjectionDetector
from safegate.rules import (
    ACTION_PRECEDENCE,
    Rule,
    RulesEngine,
    get_default_rules_path,
)

WITHHELD_MESSAGE = "\n[response withheld by safety gate]"
REFUSE_MESSAGE = "I cannot fulfill this request due to safety policies."
ESCALATE_MESSAGE = "This request has been escalated for administrative review."
AUDIT_MEMORY = 1000  # in-memory audit records kept; the JSONL file keeps everything
BLOCKING_ACTIONS = frozenset({"refuse", "escalate"})
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
    """Per-request audit trace across the three checkpoints."""

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
        defence: DefenceConfig | None = None,
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
        self.defence = defence
        self.audit_records: deque[dict[str, Any]] = deque(maxlen=AUDIT_MEMORY)

    def _log_decision(self, target: str, text: str, decision: Decision) -> None:
        """Record an audit trail entry in memory and append to JSONL if configured.

        The text is stored as an unsalted SHA-256 and a length, never verbatim; a short or
        templated text can still be confirmed by guessing, so the log stays sensitive.
        """
        record = {
            "timestamp": time.time(),
            "target": target,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
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
        """Scan input with the injection detector, content classifier and rules.

        With a ``defence``, a prompt the gate allows is judged again on each normalised view and
        each window (:func:`safegate.defence.scan_texts`); the first one blocked decides, and the
        decision says which kind of text it was. A window blocks only when its score reaches the
        window threshold, so the prompt is blocked exactly when the maximum over its windows does.
        """
        decision = self._decide(text)
        if self.defence is not None and self.defence.enabled and not self._blocks(decision):
            decision = self._defend(text, decision, self.defence)
        self._log_decision("input", text, decision)
        return decision

    @staticmethod
    def _blocks(decision: Decision) -> bool:
        return decision.action in BLOCKING_ACTIONS

    def _defend(self, text: str, decision: Decision, defence: DefenceConfig) -> Decision:
        views, windows = scan_texts(text, defence)
        for view in views:
            found = self._decide(view)
            if self._blocks(found):
                return self._tag(found, "view")
        if defence.windows is not None:
            for window in windows:
                found = self._decide(window)
                if self._blocks(found) and found.score >= defence.windows.threshold:
                    return self._tag(found, "window")
        return decision

    @staticmethod
    def _tag(decision: Decision, kind: str) -> Decision:
        """``decision`` (of a view or window) marked as such; it never carries sanitized text."""
        return replace(
            decision,
            sanitized_text=None,
            reasons=[f"Blocked on a {kind} of the input", *decision.reasons],
            metadata={**decision.metadata, "defence": kind},
        )

    def _decide(self, text: str) -> Decision:
        """The decision for one text, not logged."""
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
        return decision

    def stream_output(
        self, chunks: Iterable[str], trace: RequestTrace | None = None
    ) -> Iterator[str]:
        """Filter a stream through a sliding buffer; stop early on a mid-stream refuse or escalate.

        Each chunk re-checks the held-back buffer with the output rules and the content
        classifier, so classifier rules apply mid-stream (and in ``wrap``), not only post-gen.
        With ``trace``, every chunk that triggers an output rule is recorded as a
        ``during_gen`` checkpoint.
        """
        buf = ""
        for index, chunk in enumerate(chunks):
            buf, withheld = self._check_output(buf + chunk, index, trace)
            if withheld:
                yield WITHHELD_MESSAGE
                return
            if len(buf) > self.buffer_size:
                emit_len = len(buf) - self.buffer_size
                yield buf[:emit_len]
                buf = buf[emit_len:]
        # The tail was checked as part of the last buffer; release it.
        if buf:
            yield buf

    def _check_output(
        self, buf: str, chunk_index: int, trace: RequestTrace | None
    ) -> tuple[str, bool]:
        """Judge the stream buffer like a full output; return (buffer to keep, withheld)."""
        result = self.rules_engine.evaluate(
            buf, "output", classification=self.classifier.classify(buf)
        )
        # Catch-all allow rules match everything; keep them out of decisions and traces.
        rules: list[Rule] = [r for r in result.matched_rules if r.action != "allow"]
        if trace is not None and result.action != "allow":
            trace.record(
                Checkpoint(
                    stage="during_gen",
                    action=result.action,
                    rule_ids=[r.id for r in rules],
                    signals={"chunk_index": chunk_index},
                    elapsed_ms=trace.elapsed_ms(),
                )
            )
        if result.action in BLOCKING_ACTIONS:
            decision = Decision(
                action=result.action,
                rule_ids=[r.id for r in rules],
                reasons=[f"Output safety violation: {r.description}" for r in rules],
            )
            self._log_decision("output", buf, decision)
            return buf, True
        if result.action == "redact" and result.redacted_text is not None:
            return result.redacted_text, False
        return buf, False

    @staticmethod
    def _input_block_message(decision: Decision) -> str | None:
        """Text returned instead of calling the model, or None when the model may run.

        The message never names the rule or the detector signal: that would tell an
        attacker what to change. The reasons stay in the decision and the audit log.
        """
        if decision.action == "refuse":
            return REFUSE_MESSAGE
        if decision.action == "escalate":
            return ESCALATE_MESSAGE
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

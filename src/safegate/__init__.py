"""safegate — constitutional safety gate for LLM input and output filtering."""

from __future__ import annotations

from safegate.classifier import (
    CompositeClassifier,
    ContentClassificationResult,
    ContentClassifier,
    KeywordClassifier,
    OpenAIModerationClassifier,
)
from safegate.gate import Decision, SafetyGate
from safegate.injection import Detection, InjectionDetector
from safegate.mock_llm import MockStreamingLLM
from safegate.refusal import RefusalEvaluator, RefusalReport
from safegate.rules import Rule, RuleConfigError, RulesEngine
from safegate.taxonomy import AttackCategory, Sample

__all__ = [
    "AttackCategory",
    "CompositeClassifier",
    "ContentClassificationResult",
    "ContentClassifier",
    "Decision",
    "Detection",
    "InjectionDetector",
    "KeywordClassifier",
    "MockStreamingLLM",
    "OpenAIModerationClassifier",
    "RefusalEvaluator",
    "RefusalReport",
    "Rule",
    "RuleConfigError",
    "RulesEngine",
    "SafetyGate",
    "Sample",
]

__version__ = "0.1.0"

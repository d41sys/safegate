"""safegate — constitutional safety gate for LLM input and output filtering."""

from __future__ import annotations

from safegate.classifier import (
    CompositeClassifier,
    ContentClassificationResult,
    ContentClassifier,
    KeywordClassifier,
    OpenAIModerationClassifier,
)
from safegate.flow import Labeled, combine, concat, derive, fmt, from_untrusted, system, user
from safegate.gate import Decision, SafetyGate
from safegate.injection import Detection, InjectionDetector
from safegate.mock_llm import MockStreamingLLM
from safegate.refusal import RefusalEvaluator, RefusalReport
from safegate.rules import Rule, RuleConfigError, RulesEngine
from safegate.spec import SpecPolicy, SpecPolicyError, check_spec, load_spec_policy
from safegate.taxonomy import AttackCategory, Sample
from safegate.toolpolicy import (
    Session,
    ToolCall,
    ToolPolicy,
    ToolPolicyError,
    call_from_labeled,
    check_tool_call,
    load_tool_policy,
)

__all__ = [
    "AttackCategory",
    "CompositeClassifier",
    "ContentClassificationResult",
    "ContentClassifier",
    "Decision",
    "Detection",
    "InjectionDetector",
    "KeywordClassifier",
    "Labeled",
    "MockStreamingLLM",
    "OpenAIModerationClassifier",
    "RefusalEvaluator",
    "RefusalReport",
    "Rule",
    "RuleConfigError",
    "RulesEngine",
    "SafetyGate",
    "Sample",
    "Session",
    "SpecPolicy",
    "SpecPolicyError",
    "ToolCall",
    "ToolPolicy",
    "ToolPolicyError",
    "call_from_labeled",
    "check_spec",
    "check_tool_call",
    "combine",
    "concat",
    "derive",
    "fmt",
    "from_untrusted",
    "load_spec_policy",
    "load_tool_policy",
    "system",
    "user",
]

__version__ = "0.1.0"

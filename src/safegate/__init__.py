"""safegate — constitutional safety gate for LLM input and output filtering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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
    check_egress_text,
    check_tool_call,
    load_tool_policy,
    redact_egress_text,
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
    "check_egress_text",
    "check_spec",
    "check_tool_call",
    "combine",
    "concat",
    "derive",
    "fmt",
    "from_untrusted",
    "load_spec_policy",
    "load_tool_policy",
    "redact_egress_text",
    "system",
    "user",
]

try:
    __version__ = _pkg_version("safegate")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

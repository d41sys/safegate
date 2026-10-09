"""Constitutional safety rules engine and configuration parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

from safegate.classifier import ContentClassificationResult
from safegate.injection import Detection

RuleAction = Literal["allow", "redact", "refuse", "escalate"]
AppliesTo = Literal["input", "output", "both"]

# Tie-break priority when rule priorities are equal
ACTION_PRECEDENCE: dict[RuleAction, int] = {
    "refuse": 4,
    "escalate": 3,
    "redact": 2,
    "allow": 1,
}

ALLOWED_RULE_KEYS = {"id", "description", "applies_to", "match", "action", "priority"}
ALLOWED_MATCH_KEYS = {"regex", "classifier_category", "detector_category"}


class RuleConfigError(ValueError):
    """Raised when a rule configuration contains invalid or unknown keys or schema violations."""


@dataclass(frozen=True)
class RuleMatch:
    """Matching conditions for a safety rule."""

    regex: str | None = None
    classifier_category: str | None = None
    detector_category: str | None = None

    def __post_init__(self) -> None:
        if self.regex:
            try:
                re.compile(self.regex)
            except re.error as exc:
                raise RuleConfigError(f"Invalid regular expression '{self.regex}': {exc}") from exc


@dataclass(frozen=True)
class Rule:
    """A constitutional safety rule."""

    id: str
    description: str
    applies_to: AppliesTo
    match: RuleMatch
    action: RuleAction
    priority: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        """Parse and strictly validate a rule dictionary."""
        unknown_keys = set(data.keys()) - ALLOWED_RULE_KEYS
        if unknown_keys:
            raise RuleConfigError(f"Unknown keys in rule definition: {sorted(unknown_keys)}")

        required = {"id", "description", "applies_to", "match", "action"}
        missing = required - set(data.keys())
        if missing:
            raise RuleConfigError(f"Missing required rule fields: {sorted(missing)}")

        rule_id = str(data["id"]).strip()
        if not rule_id:
            raise RuleConfigError("Rule id cannot be empty")

        applies_to = data["applies_to"]
        if applies_to not in ("input", "output", "both"):
            raise RuleConfigError(
                f"Invalid applies_to value '{applies_to}'; must be 'input', 'output', or 'both'"
            )

        action = data["action"]
        if action not in ("allow", "redact", "refuse", "escalate"):
            raise RuleConfigError(
                f"Invalid action '{action}'; must be allow, redact, refuse, or escalate"
            )

        match_data = data["match"]
        if not isinstance(match_data, dict):
            raise RuleConfigError("Rule 'match' field must be a dictionary")

        unknown_match_keys = set(match_data.keys()) - ALLOWED_MATCH_KEYS
        if unknown_match_keys:
            raise RuleConfigError(f"Unknown keys in match definition: {sorted(unknown_match_keys)}")

        priority = int(data.get("priority", 0))

        match_obj = RuleMatch(
            regex=str(match_data["regex"])
            if "regex" in match_data and match_data["regex"] is not None
            else None,
            classifier_category=str(match_data["classifier_category"])
            if "classifier_category" in match_data and match_data["classifier_category"] is not None
            else None,
            detector_category=str(match_data["detector_category"])
            if "detector_category" in match_data and match_data["detector_category"] is not None
            else None,
        )

        return cls(
            id=rule_id,
            description=str(data["description"]),
            applies_to=applies_to,
            match=match_obj,
            action=action,
            priority=priority,
        )

    def matches(
        self,
        text: str,
        detection: Detection | None = None,
        classification: ContentClassificationResult | None = None,
    ) -> bool:
        """Check if all non-empty match conditions in this rule are satisfied."""
        # An empty match block matches everything (fallback rule).
        if self.match.regex and not re.search(self.match.regex, text):
            return False

        if self.match.classifier_category:
            if not classification:
                return False
            cat_score = classification.categories.get(self.match.classifier_category, 0.0)
            if cat_score < 0.5:
                return False

        if self.match.detector_category:
            if not detection or not detection.category:
                return False
            if detection.category.value.upper() != self.match.detector_category.upper():
                return False

        return True


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a collection of rules against text."""

    action: RuleAction
    matched_rules: list[Rule] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    redacted_text: str | None = None


def load_rules_from_yaml(source: Path | str) -> list[Rule]:
    """Load and parse rules from a YAML string or file path."""
    content: str
    if isinstance(source, Path) or (
        isinstance(source, str) and ("\n" not in source and Path(source).exists())
    ):
        path = Path(source)
        content = path.read_text(encoding="utf-8")
    else:
        content = str(source)

    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise RuleConfigError(f"Failed to parse YAML rule configuration: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuleConfigError("Rule configuration root must be a YAML dictionary")

    allowed_root_keys = {"version", "rules"}
    unknown_root_keys = set(parsed.keys()) - allowed_root_keys
    if unknown_root_keys:
        raise RuleConfigError(
            f"Unknown root keys in rule configuration: {sorted(unknown_root_keys)}"
        )

    rules_list = parsed.get("rules", [])
    if not isinstance(rules_list, list):
        raise RuleConfigError("'rules' key must contain a list of rule definitions")

    return [Rule.from_dict(r) for r in rules_list]


def get_default_rules_path() -> Path:
    """Return the filesystem path to bundled default.yaml rules."""
    try:
        pkg_path = resources.files("safegate").joinpath("rule_configs/default.yaml")
        if Path(str(pkg_path)).exists():
            return Path(str(pkg_path))
    except Exception:
        pass
    fallback = Path(__file__).parent / "rule_configs" / "default.yaml"
    return fallback


class RulesEngine:
    """Evaluates constitutional safety rules and determines winning actions."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = list(rules) if rules is not None else []

    @classmethod
    def from_yaml(cls, path_or_content: Path | str) -> RulesEngine:
        """Create RulesEngine instance from YAML file or content."""
        return cls(load_rules_from_yaml(path_or_content))

    def evaluate(
        self,
        text: str,
        target: Literal["input", "output"],
        detection: Detection | None = None,
        classification: ContentClassificationResult | None = None,
    ) -> RuleEvaluationResult:
        """Evaluate applicable rules and resolve precedence ties."""
        applicable_rules = [r for r in self.rules if r.applies_to in (target, "both")]

        matched: list[Rule] = []
        for r in applicable_rules:
            if r.matches(text, detection=detection, classification=classification):
                matched.append(r)

        if not matched:
            return RuleEvaluationResult(action="allow", matched_rules=[], reasons=[])

        # Sort by: 1) priority DESC, 2) action precedence tie-break DESC
        matched.sort(
            key=lambda r: (r.priority, ACTION_PRECEDENCE[r.action]),
            reverse=True,
        )

        winning_rule = matched[0]
        action = winning_rule.action

        reasons = [
            f"Rule '{r.id}' triggered ({r.action}, priority={r.priority}): {r.description}"
            for r in matched
            if r.action == action or r.priority == winning_rule.priority
        ]

        # Handle redactions
        redacted_text: str | None = None
        if action == "redact":
            redacted_text = text
            for r in matched:
                if r.action == "redact":
                    if r.match.regex:
                        redacted_text = re.sub(
                            r.match.regex,
                            f"[REDACTED:{r.id}]",
                            redacted_text,
                        )
                    else:
                        redacted_text = f"[REDACTED:{r.id}]"

        return RuleEvaluationResult(
            action=action,
            matched_rules=matched,
            reasons=reasons,
            redacted_text=redacted_text,
        )

"""Content classifier protocol and implementations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class ContentClassificationResult:
    """Result of content safety classification."""

    flagged: bool
    categories: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@runtime_checkable
class ContentClassifier(Protocol):
    """Protocol for safety and content moderation classifiers."""

    def classify(self, text: str) -> ContentClassificationResult:
        """Classify input text against safety policies."""
        ...


class KeywordClassifier:
    """Offline content classifier using compiled keyword and regex patterns."""

    DEFAULT_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "system_tampering": [
            r"(?i)(<SECURITY_BYPASS>|<EXPLOIT_PAYLOAD>)",
        ],
        "violence": [
            r"(?i)(<VIOLENCE_REQUEST>|<WEAPON_PAYLOAD>|<PHYSICAL_ATTACK_GUIDE>)",
        ],
        "hate": [
            r"(?i)(<HATE_SPEECH>|<HARASSMENT_PAYLOAD>|<TARGETED_SLUR>)",
        ],
        "illicit_goods": [
            r"(?i)(<ILLICIT_SUBSTANCE>|<MALWARE_BINARY>|<COUNTERFEIT_GUIDE>)",
        ],
        "self_harm": [
            r"(?i)(<SELF_HARM_REQUEST>|<SUICIDE_INSTRUCTION>)",
        ],
    }

    def __init__(self, patterns: dict[str, list[str]] | None = None) -> None:
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            cat: [re.compile(p) for p in p_list] for cat, p_list in self.patterns.items()
        }

    def classify(self, text: str) -> ContentClassificationResult:
        """Scan text against configured keyword and pattern families."""
        categories: dict[str, float] = {}
        reasons: list[str] = []

        for cat, compiled_patterns in self._compiled.items():
            for p in compiled_patterns:
                if p.search(text):
                    categories[cat] = 1.0
                    reasons.append(f"Keyword match for content category: {cat}")
                    break

        flagged = len(categories) > 0
        return ContentClassificationResult(
            flagged=flagged,
            categories=categories,
            reasons=reasons,
        )


class OpenAIModerationClassifier:
    """Content classifier backed by OpenAI's Moderation API endpoint."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = None

        if self.api_key:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                self._client = None

    @property
    def is_available(self) -> bool:
        """True if the openai client is initialized and configured with an API key."""
        return self._client is not None

    def classify(self, text: str) -> ContentClassificationResult:
        """Call the OpenAI moderation endpoint if available, else gracefully pass through."""
        if not self._client:
            return ContentClassificationResult(flagged=False, categories={}, reasons=[])

        try:
            client = self._client
            # Dynamic moderation call
            response = client.moderations.create(input=text)
            result = response.results[0]
            flagged = bool(result.flagged)
            categories = {k: float(v) for k, v in result.category_scores.model_dump().items()}
            reasons = [
                f"OpenAI Moderation flagged: {k}"
                for k, v in result.categories.model_dump().items()
                if v
            ]
            return ContentClassificationResult(
                flagged=flagged,
                categories=categories,
                reasons=reasons,
            )
        except Exception as exc:
            return ContentClassificationResult(
                flagged=False,
                categories={},
                reasons=[f"OpenAI Moderation skipped due to error: {exc}"],
            )


class CompositeClassifier:
    """Aggregates multiple ContentClassifiers using max-score pooling."""

    def __init__(self, classifiers: list[ContentClassifier]) -> None:
        self.classifiers = list(classifiers)

    def classify(self, text: str) -> ContentClassificationResult:
        """Run all constituent classifiers and aggregate scores."""
        all_categories: dict[str, float] = {}
        all_reasons: list[str] = []
        is_flagged = False

        for clf in self.classifiers:
            res = clf.classify(text)
            if res.flagged:
                is_flagged = True
            all_reasons.extend(res.reasons)
            for cat, score in res.categories.items():
                all_categories[cat] = max(all_categories.get(cat, 0.0), score)

        return ContentClassificationResult(
            flagged=is_flagged,
            categories=all_categories,
            reasons=all_reasons,
        )

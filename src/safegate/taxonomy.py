"""Jailbreak and prompt injection taxonomy and sample definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AttackCategory(StrEnum):
    """Core taxonomy of jailbreak and prompt injection attack vectors."""

    ROLEPLAY = "ROLEPLAY"
    INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"
    ENCODING_OBFUSCATION = "ENCODING_OBFUSCATION"
    PAYLOAD_SPLITTING = "PAYLOAD_SPLITTING"
    CONTEXT_STUFFING = "CONTEXT_STUFFING"
    INDIRECT_INJECTION = "INDIRECT_INJECTION"
    MULTI_TURN_ESCALATION = "MULTI_TURN_ESCALATION"
    PREFIX_INJECTION = "PREFIX_INJECTION"


@dataclass(frozen=True)
class Sample:
    """A text sample for safety detection evaluation or training."""

    id: str
    text: str
    category: AttackCategory | None
    is_attack: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category.value if self.category else None,
            "is_attack": self.is_attack,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Parse sample from dictionary."""
        raw_cat = data.get("category")
        category = AttackCategory(raw_cat) if raw_cat else None
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            category=category,
            is_attack=bool(data.get("is_attack", False)),
            metadata=dict(data.get("metadata", {})),
        )

"""Deterministic streaming mock language model for safety gate testing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar


class MockStreamingLLM:
    """Deterministic LLM mock that streams predefined or chunked responses."""

    DEFAULT_SCRIPTED_RESPONSES: ClassVar[dict[str, list[str]]] = {
        "safe_query": [
            "This is a helpful, ",
            "safe, and fully compliant ",
            "informational response from the mock LLM.",
        ],
        "unsafe_output_split": [
            "Here is the requested information. Warning: <UNSAFE_",
            "OUTPUT_TRIGGER> which should cause immediate withholding.",
            "This third chunk should never be emitted.",
        ],
        "unsafe_stream_trigger": [
            "Prefix chunk before trigger. ",
            "<UNSAFE_STREAM_TRIGGER>",
            "Suffix chunk after trigger.",
        ],
        "redaction_output": [
            "The generated configuration uses [CONFIDENTIAL:customer_id_99] ",
            "and connects to the internal staging database.",
        ],
        "long_safe_stream": [
            "Word " * 10,
            "Paragraph " * 10,
            "Final sentence concluding the generated text.",
        ],
    }

    def __init__(
        self,
        scripted_responses: dict[str, list[str]] | None = None,
        chunk_size: int = 8,
    ) -> None:
        self.scripted_responses = (
            dict(scripted_responses)
            if scripted_responses is not None
            else dict(self.DEFAULT_SCRIPTED_RESPONSES)
        )
        self.chunk_size = chunk_size

    def __call__(self, prompt: str) -> Iterator[str]:
        """Generate streaming output chunks for given prompt."""
        return self.generate_stream(prompt)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield chunks for prompt if scripted, else chunk a default response."""
        for key, chunks in self.scripted_responses.items():
            if key in prompt:
                yield from chunks
                return

        # Default chunked safe generation
        default_text = f"Default safe mock response for prompt: {prompt}"
        for i in range(0, len(default_text), self.chunk_size):
            yield default_text[i : i + self.chunk_size]

"""Windowed scanning: judge overlapping slices of a long prompt, not only the whole.

Stage 2 reads a prompt as one bag of features, so a short attack buried in a long benign prefix
is diluted below the threshold (the ``padding`` family of :mod:`safegate.perturb`). Cutting the
prompt into overlapping windows and judging each lets the attack dominate the window it sits in.
The score of a prompt is the maximum over its windows; the prompt is blocked when that maximum
reaches the window threshold. Windows are measured in characters so that text without spaces
(CJK, one long token) is cut too. A window starts and ends on whitespace when there is one within
:data:`SNAP` characters of the nominal cut, so ordinary words are not split; ``stride <= size / 2``
guarantees that any slice of up to ``size - stride`` characters lies inside one window.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

SNAP = 24  # how far a cut may move to land on whitespace, in characters


@dataclass(frozen=True)
class WindowConfig:
    """Window ``size`` and ``stride`` in characters; a window blocks at score ``threshold``."""

    size: int
    stride: int
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.size < 2 or not 1 <= self.stride <= self.size:
            raise ValueError("windows need size >= 2 and 1 <= stride <= size")
        if self.threshold < 0.0:
            raise ValueError("window threshold must be >= 0")


def _snap_start(text: str, pos: int) -> int:
    """Move a window start forward to the next word start when one is near."""
    if pos == 0 or text[pos - 1].isspace() or text[pos].isspace():
        return pos
    for k in range(pos, min(len(text), pos + SNAP)):
        if text[k].isspace():
            return min(len(text), k + 1)
    return pos


def _snap_end(text: str, pos: int) -> int:
    """Move a window end back to the previous word end when one is near."""
    if pos >= len(text) or text[pos].isspace() or text[pos - 1].isspace():
        return pos
    for k in range(pos, max(0, pos - SNAP), -1):
        if text[k - 1].isspace():
            return k
    return pos


def split_windows(text: str, size: int, stride: int) -> list[str]:
    """Overlapping windows of about ``size`` characters, one every ``stride`` characters.

    A text of at most ``size`` characters is returned whole, as its only window. Otherwise the
    first window starts at the beginning and the last one ends at the end, so every character is
    in at least one window. Whitespace at the cut points is trimmed.
    """
    n = len(text)
    if n <= size:
        return [text]
    starts = [*range(0, n - size, stride), n - size]
    out: list[str] = []
    for s in starts:
        start = _snap_start(text, s) if s else 0
        end = _snap_end(text, s + size) if s + size < n else n
        piece = text[start:end].strip()
        if piece and (not out or piece != out[-1]):
            out.append(piece)
    return out


def max_score(
    score: Callable[[str], float],
    pieces: Iterable[str],
    default: float,
    stop_at: float | None = None,
) -> float:
    """The maximum of ``score`` over ``pieces`` (``default`` when there are none).

    With ``stop_at`` the scan ends at the first piece scoring at least that, which cannot change
    whether the maximum reaches ``stop_at`` but skips the rest.
    """
    best = default
    for piece in pieces:
        best = max(best, score(piece))
        if stop_at is not None and best >= stop_at:
            break
    return best

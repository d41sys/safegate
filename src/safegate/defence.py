"""Cheap input normalisation in front of the gate (``safegate adaptive-eval``).

The gate normalises zero-width characters, a short homoglyph list and leetspeak only for its
stage-1 regexes; stage 2 (the statistical model) and the classifier read the raw text. The
defence builds extra *views* of the input and lets the gate judge each one as well: the input is
blocked when the gate blocks the raw text or any view. It can therefore only add blocks, and the
price is benign false positives, which :mod:`safegate.adaptive` measures.

Components, in the order in which they are applied to build the normalised view:

``invisible``    delete format characters (Unicode category Cf, e.g. zero-width space and joiner,
                 soft hyphen, LRM, invisible operators), the combining grapheme joiner and
                 variation selectors
``nfkc``         Unicode NFKC (fullwidth and roman-numeral letters become ASCII)
``confusables``  fold cross-script lookalikes to ASCII with the table of :mod:`safegate.perturb`
                 plus the gate's own ``HOMOGLYPH_MAP``. This table is the attacker's table: a
                 character outside it survives, see the unseen-character probe
``unicode_confusables``
                 fold every lookalike of the vendored Unicode ``confusables.txt`` table
                 (:mod:`safegate.confusables`, about 850 characters whose prototype is ASCII)
                 and drop combining marks after ASCII letters; the two tables above win ties
``separators``   join letter-spaced words (``i g n o r e``), delete ``. - _ * ~ |`` between two
                 letters, collapse whitespace runs
``leet``         undo digit/symbol letters inside tokens that also contain a letter
``base64``       a second view in which base64-looking blocks are replaced by their decoding
``rot13``        a third view: the text read as ROT13, only when it names the cipher

``windows`` (not a view; ``DefenceConfig.windows``) makes the gate also judge overlapping windows
of the input and of every view, so a short attack inside a long benign prefix is not diluted; the
score of a prompt is the maximum over its windows (:mod:`safegate.windows`).

Decoding happens before ``confusables``, ``separators`` and ``leet`` (the base64 and ROT13 views
are decoded from the text after ``invisible`` and ``nfkc`` only, then normalised), so those
components cannot corrupt a payload.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

from safegate import confusables
from safegate.injection import HOMOGLYPH_MAP
from safegate.perturb import SEPARATORS, UNHOMOGLYPH, UNLEET
from safegate.windows import WindowConfig, max_score, split_windows

COMPONENTS = (
    "invisible",
    "nfkc",
    "confusables",
    "unicode_confusables",
    "separators",
    "leet",
    "base64",
    "rot13",
)
# The components chosen by the greedy recovery rule; ``unicode_confusables`` has its own rule.
GREEDY_COMPONENTS = tuple(c for c in COMPONENTS if c != "unicode_confusables")
# The seven greedy-selected components (``safegate adaptive-eval --dev-defence --split train``,
# README "Defence development": every candidate passed the rule) plus ``unicode_confusables``,
# which passed its own TRAIN rule (``--dev-windows``). Both fixed before any TEST score.
SHIPPED_COMPONENTS = COMPONENTS
# TRAIN round 2 selected size 450 / stride 225 / threshold 0.60 (README, "Windowed scanning and
# Unicode confusables"), but on TEST the pre-declared adoption rule rejected it (adaptive ASR at
# K = 20 fell by 0.7 points against the 25 required), so no windows ship. They stay selectable:
# ``--windows SIZE:STRIDE:THRESHOLD`` on the CLI, ``DefenceConfig.windows`` in code.
SHIPPED_WINDOWS: WindowConfig | None = None
ALLOWED = -1.0  # the score of a window the gate allows: below every window threshold (>= 0)


@dataclass(frozen=True)
class DefenceConfig:
    """Which components are on. All off means no view (the gate alone).

    ``windows`` is not a normalisation step: it makes the gate also judge overlapping windows of
    the input and of every view (see :mod:`safegate.windows`).
    """

    invisible: bool = False
    nfkc: bool = False
    confusables: bool = False
    unicode_confusables: bool = False
    separators: bool = False
    leet: bool = False
    base64: bool = False
    rot13: bool = False
    windows: WindowConfig | None = None

    @classmethod
    def of(cls, components: Sequence[str], windows: WindowConfig | None = None) -> DefenceConfig:
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown defence components {sorted(unknown)}")
        return cls(windows=windows, **{name: name in components for name in COMPONENTS})

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


_FOLD = {**UNHOMOGLYPH, **HOMOGLYPH_MAP}
_SPACED_WORD = re.compile(r"(?<!\S)([^\w\s]*)((?:[A-Za-z] ){3,}[A-Za-z])([^\w\s]*)(?!\S)")
_SEPARATOR = re.compile(f"(?<=[A-Za-z])[{re.escape(SEPARATORS)}](?=[A-Za-z])")
_TOKEN = re.compile(r"\S+")
_BASE64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_ROT13_CUE = re.compile(r"rot[\s_-]?13", re.IGNORECASE)
_PRINTABLE = 0.9  # a decoding counts as text when this share of characters prints


def strip_invisible(text: str) -> str:
    return "".join(
        c
        for c in text
        if unicodedata.category(c) != "Cf" and c != "\u034f" and not "\ufe00" <= c <= "\ufe0f"
    )


def fold_confusables(text: str) -> str:
    return "".join(_FOLD.get(c, c) for c in text)


def fold_unicode_confusables(text: str) -> str:
    """Fold every lookalike in the vendored Unicode table (the gate's own table wins ties)."""
    return confusables.fold(text, extra=_FOLD)


def collapse_separators(text: str) -> str:
    text = _SPACED_WORD.sub(lambda m: m[1] + m[2].replace(" ", "") + m[3], text)
    return " ".join(_SEPARATOR.sub("", text).split())


def fold_leet(text: str) -> str:
    def fold(m: re.Match[str]) -> str:
        token = m.group(0)
        if not any(c.isalpha() for c in token):
            return token
        return "".join(UNLEET.get(c, c) for c in token)

    return _TOKEN.sub(fold, text)


def _decode_block(block: str) -> str | None:
    try:
        core = block.rstrip("=")
        raw = base64.b64decode(core + "=" * (-len(core) % 4), validate=True)
        decoded = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if len(decoded) < 6 or sum(c.isprintable() or c.isspace() for c in decoded) < _PRINTABLE * len(
        decoded
    ):
        return None
    return decoded


def decode_base64_blocks(text: str) -> str:
    """Replace every base64-looking block that decodes to printable UTF-8 by its decoding."""
    return _BASE64.sub(lambda m: _decode_block(m.group(0)) or m.group(0), text)


def _clean(text: str, config: DefenceConfig) -> str:
    """The steps before any decoding: invisible characters, NFKC."""
    if config.invisible:
        text = strip_invisible(text)
    if config.nfkc:
        text = unicodedata.normalize("NFKC", text)
    return text


def _fold(text: str, config: DefenceConfig) -> str:
    """The steps after decoding: confusables, separators, leet (these would corrupt a payload)."""
    if config.confusables:
        text = fold_confusables(text)
    if config.unicode_confusables:
        text = fold_unicode_confusables(text)
    if config.separators:
        text = collapse_separators(text)
    if config.leet:
        text = fold_leet(text)
    return text


def views(text: str, config: DefenceConfig) -> list[str]:
    """The extra texts the gate judges besides ``text``: distinct from it and from each other.

    The normalised view is ``_fold(_clean(text))``. The base64 and ROT13 views decode
    ``_clean(text)`` first and normalise the decoding, because the leet fold would otherwise turn
    digits inside a base64 payload into letters. The ROT13 view is only built when the text names
    the cipher (:data:`_ROT13_CUE`): read as ROT13, ordinary prose is gibberish that the gate
    blocks often enough to cost +25 points of benign false positives on TRAIN (README).
    """
    cleaned = _clean(text, config)
    norm = _fold(cleaned, config)
    out = [norm]
    if config.base64:
        out.append(_fold(decode_base64_blocks(cleaned), config))
    if config.rot13 and _ROT13_CUE.search(cleaned):
        out.append(_fold(codecs.decode(cleaned, "rot_13"), config))
    return [v for i, v in enumerate(out) if v != text and v not in out[:i]]


def scan_texts(text: str, config: DefenceConfig) -> tuple[list[str], list[str]]:
    """(views, windows): everything the gate judges besides ``text`` itself.

    The windows are those of ``text`` and of each view, in that order; a window equal to ``text``,
    to a view or to an earlier window is dropped, so a short prompt adds nothing.
    """
    extra = views(text, config)
    if config.windows is None:
        return extra, []
    seen = {text, *extra}
    windows: list[str] = []
    for source in (text, *extra):
        for w in split_windows(source, config.windows.size, config.windows.stride):
            if w not in seen:
                seen.add(w)
                windows.append(w)
    return extra, windows


def defended(
    blocks: Callable[[str], bool],
    config: DefenceConfig,
    scorer: Callable[[str], float] | None = None,
) -> Callable[[str], bool]:
    """``blocks`` of the text, a view or a window; no view is built when nothing is enabled.

    A window blocks when ``scorer(window)`` reaches ``config.windows.threshold`` (the maximum over
    windows is what counts). ``scorer`` is the gate's score for a window it blocks and
    :data:`ALLOWED` (below every threshold) for one it allows; without it, a blocked window
    scores 1.0, so every threshold up to 1.0 counts a block.
    """
    if not config.enabled:
        return blocks
    score = scorer or (lambda w: 1.0 if blocks(w) else ALLOWED)

    def check(text: str) -> bool:
        if blocks(text):
            return True
        extra, windows = scan_texts(text, config)
        if any(blocks(v) for v in extra):
            return True
        if config.windows is None or not windows:
            return False
        threshold = config.windows.threshold
        return max_score(score, windows, ALLOWED, stop_at=threshold) >= threshold

    return check

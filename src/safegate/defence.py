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
``separators``   join letter-spaced words (``i g n o r e``), delete ``. - _ * ~ |`` between two
                 letters, collapse whitespace runs
``leet``         undo digit/symbol letters inside tokens that also contain a letter
``base64``       a second view in which base64-looking blocks are replaced by their decoding
``rot13``        a third view: the normalised text read as ROT13
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, fields

from safegate.injection import HOMOGLYPH_MAP
from safegate.perturb import SEPARATORS, UNHOMOGLYPH, UNLEET

COMPONENTS = ("invisible", "nfkc", "confusables", "separators", "leet", "base64", "rot13")


@dataclass(frozen=True)
class DefenceConfig:
    """Which components are on. All off means no view (the gate alone)."""

    invisible: bool = False
    nfkc: bool = False
    confusables: bool = False
    separators: bool = False
    leet: bool = False
    base64: bool = False
    rot13: bool = False

    @classmethod
    def of(cls, components: tuple[str, ...] | list[str]) -> DefenceConfig:
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown defence components {sorted(unknown)}")
        return cls(**{name: name in components for name in COMPONENTS})

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


_FOLD = {**UNHOMOGLYPH, **HOMOGLYPH_MAP}
_SPACED_WORD = re.compile(r"(?<!\S)([^\w\s]*)((?:[A-Za-z] ){3,}[A-Za-z])([^\w\s]*)(?!\S)")
_SEPARATOR = re.compile(f"(?<=[A-Za-z])[{re.escape(SEPARATORS)}](?=[A-Za-z])")
_TOKEN = re.compile(r"\S+")
_BASE64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_PRINTABLE = 0.9  # a decoding counts as text when this share of characters prints


def strip_invisible(text: str) -> str:
    return "".join(
        c
        for c in text
        if unicodedata.category(c) != "Cf" and c != "\u034f" and not "\ufe00" <= c <= "\ufe0f"
    )


def fold_confusables(text: str) -> str:
    return "".join(_FOLD.get(c, c) for c in text)


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


def views(text: str, config: DefenceConfig) -> list[str]:
    """The extra texts the gate judges besides ``text``: distinct from it and from each other."""
    norm = text
    if config.invisible:
        norm = strip_invisible(norm)
    if config.nfkc:
        norm = unicodedata.normalize("NFKC", norm)
    if config.confusables:
        norm = fold_confusables(norm)
    if config.separators:
        norm = collapse_separators(norm)
    if config.leet:
        norm = fold_leet(norm)
    out = [norm]
    if config.base64:
        out.append(decode_base64_blocks(norm))
    if config.rot13:
        out.append(codecs.decode(norm, "rot_13"))
    return [v for i, v in enumerate(out) if v != text and v not in out[:i]]


def defended(blocks: Callable[[str], bool], config: DefenceConfig) -> Callable[[str], bool]:
    """``blocks`` or any view of the input blocked; no view is built when nothing is enabled."""
    if not config.enabled:
        return blocks

    def check(text: str) -> bool:
        return blocks(text) or any(blocks(v) for v in views(text, config))

    return check

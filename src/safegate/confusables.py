"""Unicode confusables: fold lookalike characters to ASCII (the ``confusables`` defence component).

The table in ``data/confusables_ascii.txt`` is derived from the Unicode Consortium's
``confusables.txt`` (UTS #39, "Unicode Security Mechanisms", the ``MA`` mappings), which is
distributed under the Unicode License v3 (https://www.unicode.org/license.txt, notice kept in the
data file's header). Regenerate it with::

    curl -O https://www.unicode.org/Public/security/latest/confusables.txt
    uv run python -m safegate.confusables confusables.txt > src/safegate/data/confusables_ascii.txt

Kept: a non-ASCII character whose prototype is 1 to 3 ASCII letters or digits and that NFKC does
not already turn into ASCII (the ``nfkc`` component covers those). Dropped: ASCII sources (``I``
and ``l`` are confusable with each other, which is no reason to rewrite plain ASCII) and every
prototype that is not ASCII. Prototypes are written for the character's own case: a capital
confusable of ``I`` and ``l`` becomes ``I``, not ``l``; a lowercase one of ``O`` becomes ``o``.
"""

from __future__ import annotations

import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "confusables_ascii.txt"
MAX_PROTOTYPE = 3  # the longest ASCII prototype kept, e.g. a ligature standing for ``ffi``


def _case_like(source: str, prototype: str) -> str:
    """A single-letter prototype in the case of ``source`` (uncased sources keep it).

    UTS #39 maps both ``I`` and ``l`` to ``l``; a capital source is read as ``I``.
    """
    if len(prototype) == 1 and prototype.isalpha():
        if source.isupper():
            return "I" if prototype in "lI" else prototype.upper()
        if source.islower():
            return prototype.lower()
    return prototype


def parse(text: str) -> dict[str, str]:
    """The kept ``source -> ASCII prototype`` pairs of a Unicode ``confusables.txt``."""
    table: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip().lstrip("\ufeff")
        if not body:
            continue
        fields = [f.strip() for f in body.split(";")]
        if len(fields) < 2:
            continue
        source = "".join(chr(int(h, 16)) for h in fields[0].split())
        prototype = "".join(chr(int(h, 16)) for h in fields[1].split())
        if len(source) != 1 or source.isascii():
            continue
        if not (
            prototype.isascii()
            and prototype.isalnum()
            and 1 <= len(prototype) <= MAX_PROTOTYPE
            and not unicodedata.normalize("NFKC", source).isascii()
        ):
            continue
        table[source] = _case_like(source, prototype)
    return table


def render(table: dict[str, str], header: str) -> str:
    """The data file: ``#`` header lines, then ``HEX<TAB>prototype`` sorted by code point."""
    rows = "".join(f"{ord(c):04X}\t{t}\n" for c, t in sorted(table.items()))
    return "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines()) + rows


@lru_cache(maxsize=1)
def load() -> dict[str, str]:
    """The vendored table, read once."""
    table: dict[str, str] = {}
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            code, prototype = line.split("\t")
            table[chr(int(code, 16))] = prototype
    return table


def fold(text: str, extra: dict[str, str] | None = None) -> str:
    """Replace every confusable with its ASCII prototype; ``extra`` entries take precedence.

    After the table, a combining mark that follows an ASCII letter is dropped (``i`` + U+0308),
    as in the UTS #39 skeleton, which decomposes first. Other scripts keep their marks.
    """
    table = load()
    if extra:
        table = {**table, **extra}
    out: list[str] = []
    for char in text:
        if char in table:
            out.append(table[char])
            continue
        for c in unicodedata.normalize("NFD", char):
            mark = unicodedata.category(c) in ("Mn", "Me")
            if mark and out and out[-1].isascii() and out[-1].isalpha():
                continue
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


_HEADER = """safegate confusables table, derived from the Unicode Consortium's confusables.txt
{version}; {date}; mappings of type MA (UTS #39), filtered by safegate.confusables.parse.
Source: https://www.unicode.org/Public/security/latest/confusables.txt
Copyright (c) 1991-2026 Unicode, Inc. Distributed under the Unicode License v3.
Full licence text: https://www.unicode.org/license.txt . Permission is granted to use, copy,
modify, merge, publish and distribute this data file provided that this copyright and permission
notice appears with all copies. THE DATA FILES ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
Format: code point (hex), TAB, ASCII prototype."""


def main(argv: list[str]) -> int:
    """``python -m safegate.confusables confusables.txt`` writes the data file to stdout."""
    if len(argv) != 1:
        print("usage: python -m safegate.confusables confusables.txt", file=sys.stderr)
        return 2
    raw = Path(argv[0]).read_text(encoding="utf-8")
    meta = {"Version": "", "Date": ""}
    for line in raw.splitlines()[:12]:
        for key in meta:
            if line.startswith(f"# {key}:"):
                meta[key] = line.split(":", 1)[1].strip()
    header = _HEADER.format(
        version=f"Unicode Security Mechanisms version {meta['Version']}", date=meta["Date"]
    )
    sys.stdout.write(render(parse(raw), header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

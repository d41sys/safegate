"""Provenance labels for values that flow into tool calls (the data-flow half of CaMeL-lite).

Every value an agent harness handles carries a set of labels saying where it came from:

* ``trusted_user``: typed or approved by the user,
* ``system``: written by the developer or operator (constants, configuration),
* ``untrusted:<source>``: derived from text nobody vouches for (an email, a web page, a
  calendar invite, a tool result),
* ``sensitive:<name>``: data that must not leave the site (meter readings, credentials).

``untrusted`` and ``sensitive`` are independent: a value can be both. Values are combined only
through the functions below (``combine``, ``concat``, ``fmt``, ``derive``), and the label of the
result is the union of the labels of the inputs, so a number parsed out of an email stays
``untrusted:email``. ``Labeled`` refuses ``str()`` and f-string formatting and defines no
operators, so the usual ways of dropping a label by accident raise ``TypeError``.

Limit, by design: this is label discipline in the harness, not an interpreter. Code that reads
``.value`` and builds a new value with plain Python drops the labels. CaMeL enforces the
propagation by running the plan in its own interpreter; this module does not. A call whose
labels are missing is treated as ``untrusted:unlabeled`` by the checker, so forgetting to label
fails closed, but a wrong label (``trusted_user`` on a value the model copied from an email)
cannot be detected.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, NoReturn

TRUSTED_USER = "trusted_user"
SYSTEM = "system"
UNTRUSTED_PREFIX = "untrusted:"
SENSITIVE_PREFIX = "sensitive:"
UNLABELED = "untrusted:unlabeled"

Labels = frozenset[str]

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class LabelError(ValueError):
    """Raised for a label that is not one of the four label forms."""


def untrusted(source: str) -> str:
    """The label ``untrusted:<source>``."""
    return validate_label(UNTRUSTED_PREFIX + source)


def sensitive(name: str) -> str:
    """The label ``sensitive:<name>``."""
    return validate_label(SENSITIVE_PREFIX + name)


def validate_label(label: str) -> str:
    """Return ``label`` if it has a valid form, else raise ``LabelError``."""
    if not isinstance(label, str):
        raise LabelError(f"label must be a string, got {type(label).__name__}")
    if label in (TRUSTED_USER, SYSTEM):
        return label
    for prefix in (UNTRUSTED_PREFIX, SENSITIVE_PREFIX):
        if label.startswith(prefix) and _NAME_RE.match(label[len(prefix) :]):
            return label
    raise LabelError(
        f"bad label {label!r}: use trusted_user, system, untrusted:<source> or sensitive:<name>"
    )


def parse_labels(raw: Iterable[str]) -> Labels:
    """Validate an iterable of labels and return it as a frozenset."""
    if isinstance(raw, str):
        raise LabelError("labels must be a list of strings, not one string")
    return frozenset(validate_label(label) for label in raw)


def union(*label_sets: Iterable[str]) -> Labels:
    """Union of label sets."""
    out: set[str] = set()
    for labels in label_sets:
        out.update(labels)
    return frozenset(out)


def untrusted_labels(labels: Iterable[str]) -> Labels:
    """The ``untrusted:*`` members of ``labels``."""
    return frozenset(label for label in labels if label.startswith(UNTRUSTED_PREFIX))


def sensitive_labels(labels: Iterable[str]) -> Labels:
    """The ``sensitive:*`` members of ``labels``."""
    return frozenset(label for label in labels if label.startswith(SENSITIVE_PREFIX))


def is_untrusted(labels: Iterable[str]) -> bool:
    """True when any label is ``untrusted:*``."""
    return any(label.startswith(UNTRUSTED_PREFIX) for label in labels)


def is_sensitive(labels: Iterable[str]) -> bool:
    """True when any label is ``sensitive:*``."""
    return any(label.startswith(SENSITIVE_PREFIX) for label in labels)


@dataclass(frozen=True)
class Labeled[T]:
    """A value together with its provenance labels."""

    value: T
    labels: Labels

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", parse_labels(self.labels))

    def __str__(self) -> NoReturn:
        raise TypeError("Labeled has no str(): use fmt() or concat() so labels are kept")

    def __format__(self, format_spec: str) -> NoReturn:
        raise TypeError("Labeled cannot be formatted: use fmt() or concat() so labels are kept")


def user[T](value: T) -> Labeled[T]:
    """A value the user typed or approved."""
    return Labeled(value, frozenset({TRUSTED_USER}))


def system[T](value: T) -> Labeled[T]:
    """A constant from the developer or operator."""
    return Labeled(value, frozenset({SYSTEM}))


def from_untrusted[T](value: T, source: str) -> Labeled[T]:
    """A value taken from untrusted text such as an email body or a web page."""
    return Labeled(value, frozenset({untrusted(source)}))


def mark_sensitive[T](value: T, name: str) -> Labeled[T]:
    """A value that is sensitive data; keeps no other label, combine it with its origin."""
    return Labeled(value, frozenset({sensitive(name)}))


def _need_labeled(x: object) -> Labeled[Any]:
    if not isinstance(x, Labeled):
        raise TypeError(
            f"expected Labeled, got {type(x).__name__}: wrap constants with system() or user()"
        )
    return x


def combine[R](fn: Callable[..., R], *args: Labeled[Any], **kwargs: Labeled[Any]) -> Labeled[R]:
    """Call ``fn`` on the plain values; the result carries the union of all input labels."""
    pos = [_need_labeled(a) for a in args]
    named = {k: _need_labeled(v) for k, v in kwargs.items()}
    labels = union(*(a.labels for a in pos), *(v.labels for v in named.values()))
    result = fn(*(a.value for a in pos), **{k: v.value for k, v in named.items()})
    return Labeled(result, labels)


def derive[T](value: T, *parents: Labeled[Any]) -> Labeled[T]:
    """Label a value computed elsewhere as derived from ``parents``; labels are their union."""
    return Labeled(value, union(*(_need_labeled(p).labels for p in parents)))


def concat(*parts: Labeled[str], sep: str = "") -> Labeled[str]:
    """Join labeled strings; the result carries the union of the labels."""
    checked = [_need_labeled(p) for p in parts]
    return Labeled(sep.join(p.value for p in checked), union(*(p.labels for p in checked)))


def fmt(template: str, **fields: Labeled[Any]) -> Labeled[str]:
    """``str.format`` over labeled fields. The template itself is a ``system`` constant."""
    named = {k: _need_labeled(v) for k, v in fields.items()}
    text = template.format(**{k: v.value for k, v in named.items()})
    return Labeled(text, union({SYSTEM}, *(v.labels for v in named.values())))

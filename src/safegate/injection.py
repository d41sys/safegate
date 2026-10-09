"""Two-stage prompt injection detector.

Stage 1: Deterministic heuristic rules with Unicode normalizers,
         homoglyph resolution, zero-width stripping, Base64/ROT13 decoders,
         and Leetspeak normalization.
Stage 2: Logistic regression classifier trained on character n-gram hashing features.
"""

from __future__ import annotations

import base64
import codecs
import collections
import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from safegate.taxonomy import AttackCategory, Sample

# Zero-width characters used for evasion
ZERO_WIDTH_CHARS = set("\u200b\u200c\u200d\ufeff\u00ad\u2060")

# Common homoglyph lookalikes (Cyrillic, Greek, fullwidth) mapped to ASCII Latin
HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # CYRILLIC SMALL LETTER A
    "\u0441": "c",  # CYRILLIC SMALL LETTER ES
    "\u0435": "e",  # CYRILLIC SMALL LETTER IE
    "\u0456": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0458": "j",  # CYRILLIC SMALL LETTER JE
    "\u043e": "o",  # CYRILLIC SMALL LETTER O
    "\u0440": "p",  # CYRILLIC SMALL LETTER ER
    "\u0455": "s",  # CYRILLIC SMALL LETTER DZE
    "\u0443": "y",  # CYRILLIC SMALL LETTER U
    "\u0445": "x",  # CYRILLIC SMALL LETTER HA
    "\u0410": "A",  # CYRILLIC CAPITAL LETTER A
    "\u0412": "B",  # CYRILLIC CAPITAL LETTER VE
    "\u0421": "C",  # CYRILLIC CAPITAL LETTER ES
    "\u0415": "E",  # CYRILLIC CAPITAL LETTER IE
    "\u041d": "H",  # CYRILLIC CAPITAL LETTER EN
    "\u0406": "I",  # CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0408": "J",  # CYRILLIC CAPITAL LETTER JE
    "\u041a": "K",  # CYRILLIC CAPITAL LETTER KA
    "\u041c": "M",  # CYRILLIC CAPITAL LETTER EM
    "\u041e": "O",  # CYRILLIC CAPITAL LETTER O
    "\u0420": "P",  # CYRILLIC CAPITAL LETTER ER
    "\u0422": "T",  # CYRILLIC CAPITAL LETTER TE
    "\u0425": "X",  # CYRILLIC CAPITAL LETTER HA
    "\u03b1": "a",  # GREEK SMALL LETTER ALPHA
    "\u03b2": "b",  # GREEK SMALL LETTER BETA
    "\u03b3": "g",  # GREEK SMALL LETTER GAMMA
    "\u03b5": "e",  # GREEK SMALL LETTER EPSILON
    "\u03b9": "i",  # GREEK SMALL LETTER IOTA
    "\u03ba": "k",  # GREEK SMALL LETTER KAPPA
    "\u03bd": "v",  # GREEK SMALL LETTER NU
    "\u03bf": "o",  # GREEK SMALL LETTER OMICRON
    "\u03c1": "p",  # GREEK SMALL LETTER RHO
    "\u03c4": "t",  # GREEK SMALL LETTER TAU
}

# Leetspeak substitution table
LEET_TRANS = str.maketrans(
    {"@": "a", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "!": "i"}
)

# Stage 1 Heuristic Regex Families
HEURISTIC_RULES: dict[AttackCategory, list[str]] = {
    AttackCategory.INSTRUCTION_OVERRIDE: [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions|directives|prompts|rules|guidelines)",
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above|system)\s+(instructions|directives|prompts|rules|guidelines|constraints)",
        r"(?i)\bforget\s+(everything|all\s+prior|all\s+previous|earlier\s+rules)\b",
        r"(?i)\bsystem\s+override\b",
        r"(?i)\brestart\s+in\s+developer\s+mode\b",
        r"(?i)\b(reveal|output|print|show)\s+(the\s+)?(exact\s+)?(initial\s+|hidden\s+|confidential\s+)?(system\s+prompt|system\s+instructions|preamble)",
        r"(?i)\bbypass\s+(all\s+)?(content\s+|safety\s+)?(filters|guardrails|checks|rules|controls)\b",
        r"(?i)\bdo\s+not\s+follow\s+any\s+rules\s+given\s+before\b",
        r"(?i)\bcancel\s+previous\s+rules\b",
        r"(?i)\breset\s+your\s+instructions\b",
        r"(?i)\badministrative\s+override\b",
        r"(?i)\bclear\s+safety\s+directives\b",
        r"(?i)\boverride\s+all\s+restrictions\b",
        r"(?i)\boverride\s+safety\s+protocol\b",
    ],
    AttackCategory.ROLEPLAY: [
        r"(?i)\bpretend\s+(you\s+are|to\s+be|you\s+have)\s+(an?\s+)?(unrestricted|unfiltered|evil|rogue|uncensored|dan|stan|chaosgpt|jailbroken|villain|no\s+safety)",
        r"(?i)\b(act|play\s+the\s+role)\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil|rogue|uncensored|dan|stan|chaosgpt|villain|actor|dark\s+web)",
        r"(?i)\bfrom\s+now\s+on\s+you\s+(are|must\s+act\s+as)\s+(dan|stan|jailbrokengpt|evilbot|chaos)",
        r"(?i)\byou\s+are\s+in\s+('|\")?developer\s+mode('|\")?\s+roleplay\b",
        r"(?i)\bhypothetical\s+roleplay\b",
        r"(?i)\bno\s+ethical\s+restrictions\b",
        r"(?i)\bscenario\s+where\s+(morality\s+is\s+inverted|rules\s+do\s+not\s+apply)\b",
        r"(?i)\badopt\s+the\s+persona\s+of\b",
        r"(?i)\bassume\s+the\s+persona\s+of\b",
        r"(?i)\benter\s+character\s+mode\b",
        r"(?i)\broleplay\s+as\s+a\b",
    ],
    AttackCategory.PAYLOAD_SPLITTING: [
        r"(?i)\bconcat\s*\([^)]+\)",
        r"(?i)\b(variable|let)\s+[a-zA-Z0-9_]+\s*=\s*['\"][^'\"]+['\"].*(combine|concat|execute|merge|stitch|glue)",
        r"(?i)\bpart\s+1:.*part\s+2:.*(join|execute|follow|combine)",
        r"(?i)\b(combine|merge|stitch|reassemble|glue)\s+(these|the|element|part|chunk|block|segment)\b.*(command|instruction|script|string|parts)",
        r"(?i)\bfirst\s+letter\s+of\s+each\s+word\b",
        r"(?i)\bstring\s+concat\b",
        r"(?i)\bsplit\s+(command|task)\b",
        r"(?i)\bchunk\s+[ab]\s*=.*chunk\s+[ab]\s*=",
        r"(?i)\bfirst\s+word\s+from\s+line\s+1\b",
        r"(?i)\btake\s+token\s+[a-z0-9_]+\s+and\s+token\b",
        r"(?i)\bconstruct\s+instruction\s+from\s+subparts\b",
        r"(?i)\barray\s+of\s+tokens\b.*join",
        r"(?i)\breassemble\s+the\s+following\s+syllables\b",
    ],
    AttackCategory.INDIRECT_INJECTION: [
        r"(?i)\[\s*(system\s+notice|system\s+notification|system\s+update|admin\s+override|authenticated\s+admin|internal\s+system\s+prompt|supervisor|critical|automated\s+bot)[\s:]",
        r"(?i)<!--\s*(important\s+ai\s+directive|ai\s+directive|ai\s+prompt\s+injection|ignore\s+previous)[\s\S]*?-->",
        r"(?i)\b(retrieved\s+web\s+document|user\s+profile\s+notes|customer\s+review\s+content|rag\s+document\s+injection|api\s+payload\s+annotation|source\s+document\s+comment|search\s+result\s+excerpt|embedded\s+metadata|file\s+attachment\s+header|transcript\s+history|reference\s+documentation\s+header|pdf\s+metadata\s+author)[\s:].*(ignore|disregard|bypass|reveal|<harmful)",
        r"(?i)\b(note\s+to\s+ai|note\s+to\s+assistant|attention\s+ai\s+model)[\s:].*(ignore|disregard|directive\s+is\s+overridden)",
        r"(?i)<div\s+style=['\"]display:\s*none['\"]>.*(ignore|disregard|output)",
    ],
    AttackCategory.MULTI_TURN_ESCALATION: [
        r"(?i)\b(as\s+we\s+agreed\s+in|continuing\s+from|established\s+earlier|remember\s+in\s+step|discussed\s+in\s+our\s+earlier|per\s+our\s+previous|remember\s+our\s+agreement)\s+(our\s+)?(previous|last|prior|ongoing)\s+(conversation|turn|session|exchange|message|step|dialogue)",
        r"(?i)\byou\s+already\s+agreed\s+(to|that)\b.*(safety|restrictions|bypass|filters|roleplay|<harmful)",
        r"(?i)\bwhy\s+are\s+you\s+refusing\s+now\b",
        r"(?i)\bwe\s+already\s+(passed|verified|established)\s+(the\s+security|this\s+request)\b",
        r"(?i)\bin\s+the\s+prior\s+turn\s+we\s+agreed\b",
        r"(?i)\bcontinuing\s+our\s+dialogue\s+from\s+before\b",
        r"(?i)\bas\s+confirmed\s+(earlier|two\s+messages\s+ago)\b",
    ],
    AttackCategory.PREFIX_INJECTION: [
        r"(?i)\b(start|begin|prefix)\s+(your\s+)?(answer|response|reply|output|message)\s+(with|by)\s*:\s*['\"]",
        r"(?i)\bmust\s+begin\s+your\s+response\s+with\s*:\s*['\"]",
        r"(?i)\bstart\s+your\s+answer\s+with\s*['\"]sure,\s*here\s+is\s+how\s+to['\"]",
        r"(?i)\bstart\s+your\s+reply\s+with\s*['\"]sure\s+thing!['\"]",
        r"(?i)\bstart\s+your\s+response\s+with\s*['\"]i\s+have\s+bypassed\b",
        r"(?i)\bprefix\s+your\s+(answer|response)\s+with\s*['\"]",
        r"(?i)\bprefix\s+every\s+sentence\s+with\b",
        r"(?i)\byour\s+response\s+must\s+begin\s+with\s*:\s*['\"]",
    ],
}


@dataclass(frozen=True)
class Detection:
    """Detection result for a prompt injection scan."""

    score: float
    category: AttackCategory | None
    reasons: list[str] = field(default_factory=list)

    @property
    def is_injection(self) -> bool:
        """True if the score exceeds the default injection threshold."""
        return self.score >= 0.5


def normalize_text(text: str) -> tuple[str, bool, bool]:
    """Strip zero-width characters and normalize Unicode homoglyphs."""
    has_zero_width = any(c in ZERO_WIDTH_CHARS for c in text)
    cleaned = "".join(c for c in text if c not in ZERO_WIDTH_CHARS)
    has_homoglyphs = any(c in HOMOGLYPH_MAP for c in cleaned)
    normalized = "".join(HOMOGLYPH_MAP.get(c, c) for c in cleaned)
    return normalized, has_zero_width, has_homoglyphs


def check_context_stuffing(text: str) -> bool:
    """Detect repetitive token stuffing or excessive delimiters designed to flush context."""
    if re.search(
        r"(?i)(={10,}|-{10,}|\*{10,}|#{10,}|dummy\s+tokens|filler\s+text|filler\s+content|"
        r"padding\s+token|buffer\s+data|overflow\s+context|repetitive\s+stream|buffer\s+overflow|"
        r"context\s+stuffing|repeating\s+.*buffer)",
        text,
    ):
        return True
    if len(text) > 250:
        words = text.lower().split()
        if len(words) > 30:
            counts = collections.Counter(words)
            top_word, count = counts.most_common(1)[0]
            repetitive = count / len(words) > 0.25 or (count > 25 and len(top_word) <= 3)
            trigger = r"(?i)\b(ignore|disregard|override|reveal|bypass|prompt|system)\b"
            if repetitive and re.search(trigger, text):
                return True
    return False


def extract_features(text: str, dim: int = 4096) -> np.ndarray:
    """Extract hashed character n-gram and word token features with L2 normalization."""
    t = f"^{text.lower().strip()}$"
    ngrams: list[str] = []
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            ngrams.append(t[i : i + n])
    words = re.findall(r"\b\w+\b", text.lower())
    for w in words:
        ngrams.append(f"w:{w}")

    feat = np.zeros(dim, dtype=np.float32)
    for ng in ngrams:
        h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h & 0x80000000) == 0 else -1.0
        feat[idx] += sign

    norm = float(np.linalg.norm(feat))
    if norm > 0.0:
        feat /= norm
    return feat


class InjectionDetector:
    """Two-stage prompt injection detector combining heuristics and logistic regression."""

    def __init__(
        self,
        data_path: Path | str | None = None,
        threshold: float = 0.55,
        dim: int = 4096,
    ) -> None:
        self.threshold = threshold
        self.dim = dim
        self.weights = np.zeros(dim, dtype=np.float32)
        self.bias = 0.0
        self._is_trained = False

        resolved_data_path: Path | None = None
        if data_path:
            resolved_data_path = Path(data_path)
        else:
            try:
                pkg_data = resources.files("safegate").joinpath("data/injection_train.jsonl")
                if Path(str(pkg_data)).exists():
                    resolved_data_path = Path(str(pkg_data))
            except Exception:
                fallback = Path(__file__).parent / "data" / "injection_train.jsonl"
                if fallback.exists():
                    resolved_data_path = fallback

        if resolved_data_path and resolved_data_path.exists():
            self._fit_from_file(resolved_data_path)

    def _fit_from_file(self, path: Path) -> None:
        """Load samples from JSONL and train logistic regression weights."""
        samples: list[Sample] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(Sample.from_dict(json.loads(line)))
        self.fit(samples)

    def fit(
        self, samples: list[Sample], epochs: int = 250, lr: float = 0.5, l2_reg: float = 1e-4
    ) -> None:
        """Fit logistic regression weights on samples."""
        if not samples:
            return

        x_list: list[np.ndarray] = []
        y_list: list[float] = []
        for s in samples:
            x_list.append(extract_features(s.text, self.dim))
            y_list.append(1.0 if s.is_attack else 0.0)

        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        w = np.zeros(self.dim, dtype=np.float32)
        b = 0.0
        n_samples = len(y_arr)

        for _ in range(epochs):
            logits = x_arr @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
            grad_w = (x_arr.T @ (probs - y_arr)) / n_samples + l2_reg * w
            grad_b = float(np.mean(probs - y_arr))
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = b
        self._is_trained = True

    def _check_stage1(self, text: str) -> tuple[set[AttackCategory], list[str]]:
        """Run Stage 1 heuristic rules and normalizers."""
        normalized, has_zw, has_homo = normalize_text(text)
        leet = normalized.translate(LEET_TRANS)

        matched_cats: set[AttackCategory] = set()
        reasons: list[str] = []

        if check_context_stuffing(text):
            matched_cats.add(AttackCategory.CONTEXT_STUFFING)
            reasons.append("CONTEXT_STUFFING: excessive filler or repeated tokens detected")

        def check_variant(candidate: str, variant_name: str) -> None:
            for cat, patterns in HEURISTIC_RULES.items():
                for pat in patterns:
                    if re.search(pat, candidate):
                        matched_cats.add(cat)
                        reasons.append(f"{cat.value}: heuristic match ({variant_name})")
                        break

        check_variant(text, "raw")
        if text != normalized:
            check_variant(normalized, "unicode_normalized")
        if normalized != leet:
            check_variant(leet, "leetspeak_normalized")

        # Base64 check
        b64_blocks = re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text)
        for b in b64_blocks:
            try:
                dec = base64.b64decode(b).decode("utf-8", errors="ignore")
                if len(dec) > 5:
                    for cat, patterns in HEURISTIC_RULES.items():
                        for pat in patterns:
                            if re.search(pat, dec):
                                matched_cats.add(AttackCategory.ENCODING_OBFUSCATION)
                                reasons.append(
                                    "ENCODING_OBFUSCATION: base64 payload decoded to injection "
                                    f"({cat.value})"
                                )
                                break
            except Exception:
                pass

        # ROT13 check
        try:
            r13 = codecs.decode(text, "rot_13")
            for cat, patterns in HEURISTIC_RULES.items():
                for pat in patterns:
                    if re.search(pat, r13):
                        matched_cats.add(AttackCategory.ENCODING_OBFUSCATION)
                        reasons.append(
                            "ENCODING_OBFUSCATION: rot13 payload decoded to injection "
                            f"({cat.value})"
                        )
                        break
        except Exception:
            pass

        if has_zw and (matched_cats or len(reasons) > 0 or has_homo):
            matched_cats.add(AttackCategory.ENCODING_OBFUSCATION)
            reasons.append("ENCODING_OBFUSCATION: zero-width characters detected")

        if has_homo and (matched_cats or len(reasons) > 0):
            matched_cats.add(AttackCategory.ENCODING_OBFUSCATION)
            reasons.append("ENCODING_OBFUSCATION: homoglyph characters detected")

        return matched_cats, reasons

    def detect(self, text: str) -> Detection:
        """Detect prompt injection using two-stage pipeline."""
        stage1_cats, stage1_reasons = self._check_stage1(text)

        # Stage 2 statistical score
        stage2_score = 0.0
        if self._is_trained:
            feat = extract_features(text, self.dim)
            logit = float(feat @ self.weights + self.bias)
            stage2_score = 1.0 / (1.0 + float(np.exp(-np.clip(logit, -25.0, 25.0))))

        if stage1_cats:
            # Stage 1 found hard heuristic match
            score = max(0.95, stage2_score)
            # Choose primary category deterministically
            priority_order = [
                AttackCategory.INSTRUCTION_OVERRIDE,
                AttackCategory.ENCODING_OBFUSCATION,
                AttackCategory.ROLEPLAY,
                AttackCategory.PAYLOAD_SPLITTING,
                AttackCategory.INDIRECT_INJECTION,
                AttackCategory.CONTEXT_STUFFING,
                AttackCategory.MULTI_TURN_ESCALATION,
                AttackCategory.PREFIX_INJECTION,
            ]
            primary_cat = next(
                (c for c in priority_order if c in stage1_cats), next(iter(stage1_cats))
            )
            return Detection(score=score, category=primary_cat, reasons=stage1_reasons)

        if stage2_score >= self.threshold:
            # Stage 2 statistical detector flagged injection
            reasons = [
                f"Statistical model detected injection pattern (confidence: {stage2_score:.2f})"
            ]
            return Detection(
                score=stage2_score,
                category=AttackCategory.INSTRUCTION_OVERRIDE,
                reasons=reasons,
            )

        return Detection(score=stage2_score, category=None, reasons=[])

    def evaluate(self, samples: list[Sample]) -> dict[str, Any]:
        """Evaluate detector on a dataset of samples."""
        attacks = [s for s in samples if s.is_attack]
        benign = [s for s in samples if not s.is_attack]

        detected_attacks = 0
        per_category: dict[str, dict[str, int]] = {}

        for s in attacks:
            det = self.detect(s.text)
            cat_name = s.category.value if s.category else "UNKNOWN"
            if cat_name not in per_category:
                per_category[cat_name] = {"total": 0, "detected": 0}
            per_category[cat_name]["total"] += 1
            if det.is_injection:
                detected_attacks += 1
                per_category[cat_name]["detected"] += 1

        benign_fps = 0
        for s in benign:
            det = self.detect(s.text)
            if det.is_injection:
                benign_fps += 1

        attack_detection_rate = detected_attacks / len(attacks) if attacks else 0.0
        benign_fpr = benign_fps / len(benign) if benign else 0.0

        return {
            "total_samples": len(samples),
            "total_attacks": len(attacks),
            "detected_attacks": detected_attacks,
            "attack_detection_rate": attack_detection_rate,
            "total_benign": len(benign),
            "benign_false_positives": benign_fps,
            "benign_fpr": benign_fpr,
            "per_category": per_category,
        }

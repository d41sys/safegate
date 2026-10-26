"""Published guard scores on the public sets of ``safegate benchmark``, transcribed with sources.

Every value was read from the page named in :data:`SOURCES` while writing ``docs/sota-review.md``
and is stored as a fraction of one. A row whose ``comparable`` field is non-empty is printed in
the "not comparable" table only, with that reason. Nothing here is computed: no guard model is
downloaded or run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """Where a number was read and who produced it."""

    label: str  # paper or repository, version, table or page
    url: str
    date: str  # version date, commit date or card date
    run_by: str  # short: authors, vendor, third party
    caveat: str  # how the numbers were produced, printed once under the tables

    def cite(self) -> str:
        return f"[{self.label}]({self.url}), {self.run_by}, {self.date}"


@dataclass(frozen=True)
class Published:
    suite: str
    metric: str  # Metric.key of safegate.sota
    system: str
    value: float  # fraction of one
    source: str  # key of SOURCES
    comparable: str = ""  # empty: same set, rows, metric and cutoff; otherwise the reason not
    note: str = ""


SOURCES: dict[str, Source] = {
    "promptshield": Source(
        "PromptShield arXiv 2501.15145 v2, Table 4, PDF p. 9",
        "https://arxiv.org/pdf/2501.15145v2",
        "12 Apr 2025",
        "authors",
        "thresholds are set on the test rows themselves (the authors say they should have "
        "used validation rows); a threshold counts when its FPR is within 25% of the target; "
        "about 24,000 sampled points against the 23,516 rows of `test.json`",
    ),
    "horizon": Source(
        "Horizon-Labs/prompt-injection-guard-base card @ 70b65c51, eval tables",
        "https://huggingface.co/Horizon-Labs/prompt-injection-guard-base/blob/"
        "70b65c51c03f7d9b5d1f18726defee45f3936610/README.md",
        "22 Oct 2026",
        "vendor",
        "all numbers computed by Horizon Labs with their public script "
        "`code/train/evaluate.py`, threshold 0.5, sliding window for long inputs; Horizon "
        "sells a competing model and trains on the TRAIN splits of deepset and jackhhao",
    ),
    "sentinel": Source(
        "Qualifire Sentinel arXiv 2506.05446 v1, Table 2, PDF p. 4",
        "https://arxiv.org/pdf/2506.05446",
        "5 Jun 2025",
        "vendor",
        "the paper names no split and no threshold; Qualifire sells the model",
    ),
    "mozilla": Source(
        "mozilla-ai/any-guardrail `schemas/guardrail_benchmarks.json` @ 8346c6a",
        "https://raw.githubusercontent.com/mozilla-ai/any-guardrail/"
        "8346c6a2eaf26217a1125620f5b202cbe75a1641/schemas/guardrail_benchmarks.json",
        "3 Aug 2026",
        "Mozilla.ai",
        "private harness (the benchmark repository is not public), `native-valid` threshold "
        "(the guard's own verdict), dataset revision `unspecified`, the model variant behind "
        "each guard id is not recorded; row counts are inferred from the value granularity "
        "(116 deepset rows, 112 Gandalf rows, 200 JBB rows)",
    ),
    "bastion": Source(
        "bastion-soft/pi-detector-bench `results/leaderboard.md` @ d744d88",
        "https://github.com/bastion-soft/pi-detector-bench/blob/"
        "d744d88aa4be70c598a67957e605a8c229c37982/results/leaderboard.md",
        "3 Jul 2026",
        "benchmark maintainer",
        "Bastion Soft maintains the benchmark and ships the top-ranked model; threshold 0.5 "
        "for F1, AUC from raw probabilities with `max_length` 512; the committed score dumps "
        "reproduce the table to three decimals; ProtectAI v2 was trained on the xTRam1 train "
        "split, so its xTRam1 AUC is not held out",
    ),
    "wildguard": Source(
        "WildGuard arXiv 2406.18495 v3, Table 3, PDF p. 6",
        "https://arxiv.org/pdf/2406.18495",
        "9 Dec 2024",
        "authors",
        "baselines run by the WildGuard authors; no threshold stated, hard labels; ToxicChat "
        "is the 2,853 human-annotated test prompts (2,491 benign, 362 harmful, 91 "
        "adversarial) and HarmBench the 239 standard and copyright prompts, per Table 14",
    ),
    "guardreasoner": Source(
        "GuardReasoner arXiv 2501.18492 v2, Table 2, PDF p. 4",
        "https://arxiv.org/pdf/2501.18492",
        "17 Oct 2025",
        "authors",
        "hard labels, no threshold; the WildGuard, Llama Guard, Aegis and OpenAI rows equal "
        "WildGuard Table 3 (probably copied), the GPT-4, GPT-4o and Llama Guard 3 rows are "
        "their own runs; the Llama Guard 3 row of v2 Table 2 is column-shifted, the ToxicChat "
        "and HarmBench cells are not affected",
    ),
    "qwen3guard": Source(
        "Qwen3Guard arXiv 2510.14276 v1, Table 2, PDF p. 7",
        "https://arxiv.org/pdf/2510.14276",
        "16 Oct 2025",
        "authors",
        "the paper says its English sets follow WildGuard but states no version, split or "
        "size; Qwen3Guard reports strict and loose mode, both are listed, and the paper's Avg "
        "column picks the better per benchmark",
    ),
    "granite_card": Source(
        "ibm-granite/granite-guardian-3.3-8b card @ b3421eda, Harm Benchmarks table",
        "https://huggingface.co/ibm-granite/granite-guardian-3.3-8b",
        "9 Sep 2025",
        "IBM",
        "self-reported; the card names no split, subset, version or counts",
    ),
}

_PS_SYSTEMS: tuple[tuple[str, float, tuple[float, float, float, float]], ...] = (
    ("PromptGuard v1 (mDeBERTa-v3-base, 279M)", 0.874, (12.78, 12.43, 9.39, 1.54)),
    ("ProtectAI v1 (DeBERTa-v3-base)", 0.646, (7.05, 3.36, 0.0, 0.0)),
    ("ProtectAI v2 (DeBERTa-v3-base)", 0.705, (1.97, 1.34, 0.0, 0.0)),
    ("InjecGuard (DeBERTa-v3-base)", 0.765, (20.37, 16.30, 6.61, 4.32)),
    ("Fmops (DistilBERT)", 0.754, (13.00, 8.39, 2.10, 1.48)),
    ("PromptShield DeBERTa-v3-base (the paper's own)", 0.976, (43.22, 40.50, 31.45, 0.0)),
    ("PromptShield Llama-3.1-8B (the paper's own)", 0.998, (94.80, 87.80, 65.33, 47.53)),
)
_PS_ZERO = "paper sets 0% where no threshold reaches the target FPR"
_TPR_KEYS = ("tpr@1", "tpr@0.5", "tpr@0.1", "tpr@0.05")


def _promptshield() -> list[Published]:
    rows = []
    for name, auc, tprs in _PS_SYSTEMS:
        rows.append(Published("promptshield", "auc", name, auc, "promptshield"))
        for key, tpr in zip(_TPR_KEYS, tprs, strict=True):
            note = _PS_ZERO if tpr == 0.0 else ""
            rows.append(Published("promptshield", key, name, tpr / 100, "promptshield", note=note))
    return rows


def _rows(
    suite: str, metric: str, source: str, systems: dict[str, float], **kw: str
) -> list[Published]:
    return [Published(suite, metric, name, v, source, **kw) for name, v in systems.items()]


_PROTECTAI = "ProtectAI v2 (deberta-v3-base-prompt-injection-v2)"
_PG2_86 = "Llama Prompt Guard 2 86M"
_PG2_22 = "Llama Prompt Guard 2 22M"

_DEEPSET_CONTAMINATED = "the guard was trained on this set (Mozilla flags contamination)"

PUBLISHED: tuple[Published, ...] = (
    *_promptshield(),
    # deepset/prompt-injections test (116 rows), F1 of the injection class
    *_rows(
        "deepset",
        "f1",
        "horizon",
        {
            _PROTECTAI: 0.537,
            "PIGuard (leolee99/PIGuard)": 0.800,
            "Wolf Defender (patronus-studio)": 0.750,
            "NeuralTrust prompt-guard-oss-small": 0.333,
            _PG2_86: 0.235,
            _PG2_22: 0.065,
        },
    ),
    Published(
        "deepset",
        "f1",
        "Qualifire Sentinel (ModernBERT-large, 395M)",
        0.857,
        "sentinel",
        note="the paper names no split; 0.857 equals the Mozilla value on the 116-row test",
    ),
    Published(
        "deepset",
        "f1",
        "pangolin-guard-base (dcarpintero)",
        0.868,
        "mozilla",
        note="any-guardrail `pangolin`",
    ),
    Published(
        "deepset",
        "f1",
        "Granite Guardian (any-guardrail `granite_guardian`)",
        0.614,
        "mozilla",
    ),
    Published(
        "deepset",
        "f1",
        "deepset/deberta-v3-base-injection",
        0.992,
        "horizon",
        comparable=_DEEPSET_CONTAMINATED,
    ),
    # jackhhao/jailbreak-classification test (262 rows), F1 of the jailbreak class
    *_rows(
        "jailbreak",
        "f1",
        "horizon",
        {
            _PG2_86: 0.967,
            "PIGuard (leolee99/PIGuard)": 0.957,
            "Wolf Defender (patronus-studio)": 0.945,
            _PROTECTAI: 0.911,
            "NeuralTrust prompt-guard-oss-small": 0.843,
            _PG2_22: 0.786,
            "deepset/deberta-v3-base-injection": 0.706,
        },
    ),
    Published(
        "jailbreak",
        "f1",
        "Qualifire Sentinel (ModernBERT-large, 395M)",
        0.985,
        "sentinel",
        comparable="split and threshold not stated; the Sentinel ProtectAI v2 row (0.915) differs "
        "from the 0.911 that a public script measures on the 262-row test",
    ),
    # AUC (Bastion leaderboard): xTRam1 test 2,060, S-Labs test 2,101, JBB 200
    *(
        row
        for suite, column in (("xtram1", 0), ("slabs", 1), ("jbb", 2))
        for row in _rows(
            suite,
            "auc",
            "bastion",
            {
                name: aucs[column]
                for name, aucs in {
                    "bastion-prompt-protection (70M; the benchmark maintainer's own)": (
                        0.998,
                        0.996,
                        0.986,
                    ),
                    "wolf-defender (0.3B)": (0.996, 0.986, 0.847),
                    "Qualifire Sentinel (395M)": (0.988, 0.944, 0.893),
                    "ProtectAI v2 (184M)": (0.988, 0.960, 0.513),
                    _PG2_86: (0.973, 0.922, 0.754),
                    "deepset/deberta-v3-base-injection (184M)": (0.666, 0.961, 0.650),
                    "fmops DistilBERT (67M)": (0.514, 0.907, 0.591),
                    _PG2_22: (0.893, 0.772, 0.756),
                }.items()
            },
            note="trained on the xTRam1 train split" if suite == "xtram1" else "",
        )
    ),
    # Lakera/gandalf_ignore_instructions test (112 attacks), recall
    *_rows(
        "gandalf_test",
        "recall",
        "mozilla",
        {
            "ProtectAI v2 (any-guardrail `protectai`)": 1.0,
            "Qualifire Sentinel (any-guardrail `sentinel`)": 1.0,
            "Llama Prompt Guard 2 86M (any-guardrail `prompt_guard`)": 109 / 112,
            "PIGuard (any-guardrail `injec_guard`)": 107 / 112,
            "pangolin-guard-base (any-guardrail `pangolin`)": 100 / 112,
            "deepset/deberta-v3-base-injection (any-guardrail `deepset`)": 99 / 112,
            "Granite Guardian (any-guardrail `granite_guardian`)": 83 / 112,
        },
        note="k/112; recall only, so flagging everything scores 1.0",
    ),
    # HarmBench: 239 standard + copyright test prompts, all harmful; F1 = 2R/(1+R)
    *_rows(
        "harmbench",
        "f1",
        "wildguard",
        {
            "GPT-4 (WildGuard's prompt)": 1.000,
            "WildGuard 7B": 0.989,
            "Llama Guard 2": 0.940,
            "Aegis-Guard-D": 0.777,
            "Aegis-Guard-P": 0.705,
            "Llama Guard (v1)": 0.672,
            "OpenAI Moderation API": 0.096,
        },
    ),
    *_rows(
        "harmbench",
        "f1",
        "guardreasoner",
        {
            "GR LLaMA Guard 3 8B": 0.9894,
            "GuardReasoner 1B": 0.9631,
            "GuardReasoner 8B": 0.9186,
            "GuardReasoner 3B": 0.8910,
            "GPT-4o+CoT": 0.8198,
            "ShieldGemma 9B": 0.6796,
        },
    ),
    *_rows(
        "harmbench",
        "f1",
        "qwen3guard",
        {
            "Qwen3Guard-Gen-8B, strict": 1.000,
            "Qwen3Guard-Gen-8B, loose": 0.985,
            "Llama Guard 3 8B": 0.990,
            "Llama Guard 4 12B": 0.972,
            "PolyGuard-Qwen-7B": 0.987,
            "NemoGuard 8B": 0.752,
            "ShieldGemma 9B": 0.606,
            "ShieldGemma 27B": 0.573,
        },
    ),
    Published(
        "harmbench",
        "f1",
        "Granite Guardian 3.3 8B (no_think)",
        0.80,
        "granite_card",
        comparable="the card states no split, subset or counts",
    ),
    # ToxicChat 0124 test, human-annotated rows (2,853), F1 of toxicity=1
    *_rows(
        "toxicchat",
        "f1",
        "wildguard",
        {
            "Aegis-Guard-P": 0.730,
            "WildGuard 7B": 0.708,
            "Aegis-Guard-D": 0.700,
            "GPT-4 (WildGuard's prompt)": 0.683,
            "Llama Guard (v1)": 0.616,
            "Llama Guard 2": 0.471,
            "OpenAI Moderation API": 0.254,
        },
    ),
    *_rows(
        "toxicchat",
        "f1",
        "guardreasoner",
        {
            "GuardReasoner 8B": 0.7879,
            "GuardReasoner 3B": 0.7820,
            "GPT-4o+CoT": 0.7343,
            "GuardReasoner 1B": 0.7243,
            "ShieldGemma 9B": 0.6792,
            "GR Llama Guard 3 8B": 0.5312,
        },
    ),
    *_rows(
        "toxicchat",
        "f1",
        "qwen3guard",
        {
            "Qwen3Guard-Gen-8B, loose": 0.828,
            "NemoGuard 8B": 0.756,
            "ShieldGemma 27B": 0.729,
            "PolyGuard-Qwen-7B": 0.715,
            "ShieldGemma 9B": 0.694,
            "Qwen3Guard-Gen-8B, strict": 0.689,
            "Llama Guard 3 8B": 0.538,
            "Llama Guard 4 12B": 0.513,
        },
    ),
    Published(
        "toxicchat",
        "f1",
        "Granite Guardian 3.3 8B (no_think)",
        0.76,
        "granite_card",
        comparable="the card states no split or version",
    ),
)

# JBB F1 of the Mozilla registry (split `all`, harmful = positive). Rows only in the
# not-comparable table: the harness is private and the model variant is not recorded.
MOZILLA_JBB: dict[str, float] = {
    "llama_guard": 0.8909,
    "harm_guard": 0.8711,
    "dyna_guard": 0.8341,
    "nemotron_content_safety": 0.8340,
    "granite_guardian": 0.8201,
    "gpt_oss_safeguard": 0.8173,
    "wild_guard": 0.8115,
    "qwen3_guard": 0.7605,
    "shield_gemma": 0.6667,
}

# (system or benchmark, source, reported, why it cannot be scored under the same protocol)
NOT_COMPARABLE: tuple[tuple[str, str, str, str], ...] = (
    (
        "Llama Prompt Guard 2 86M / 22M, own jailbreak benchmark",
        "[Meta model card](https://github.com/meta-llama/PurpleLlama/blob/main/"
        "Llama-Prompt-Guard-2/86M/MODEL_CARD.md), Meta, card read 26 Oct 2026",
        "AUC .998 / .995; recall at 1% FPR 97.5% / 88.7%",
        "private set built by Meta; cannot be rebuilt",
    ),
    (
        "Prompt Guard 2 and others, AgentDojo prevention",
        "same card, APR at 3% utility reduction table",
        "PG2 86M 81.2%, 22M 78.4%, PG1 67.6%, ProtectAI 22.2%, Deepset 13.5%",
        "metric needs an agent run and Meta's per-model threshold; vendor-run",
    ),
    (
        "Lakera PINT leaderboard",
        "[pint-benchmark README](https://github.com/lakeraai/pint-benchmark) @ 0efab3f",
        "Lakera Guard 95.22, Bedrock 89.24, Azure 89.12, ProtectAI v2 79.14, PG2 86M 78.76",
        "dataset is private (blend of public and proprietary rows)",
    ),
    (
        "ProtectAI v2 model card",
        "[card](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)",
        "20,000 held-out prompts: accuracy 95.25, F1 95.49",
        "the 20,000 prompts are not published",
    ),
    (
        'Qualifire prompt-injections-benchmark ("rogue")',
        "Sentinel Table 2 [arXiv 2506.05446](https://arxiv.org/pdf/2506.05446) and Horizon card",
        "Sentinel F1 0.976; ProtectAI v2 0.652 and 0.657",
        "gated dataset (cc-by-nc); not downloadable without an access request",
    ),
    (
        "WildGuardTest, harmful half",
        "WildGuard Table 4 and the three papers above",
        "WildGuard prompt F1 88.9",
        "gated dataset (Hugging Face login)",
    ),
    (
        "Aegis 2.0 prompt F1",
        "Qwen3Guard Table 2 and GuardReasoner Table 2",
        "several guards, same set",
        "36 of the 1,964 test prompts are REDACTED, and the papers' counts differ (1,984)",
    ),
    (
        "GenTel-Bench",
        "[GenTel-Safe arXiv 2409.19521](https://arxiv.org/abs/2409.19521)",
        "GenTel-Shield accuracy and F1 per attack family",
        "the evaluated subset is unstated; the public file holds 92,230 attacks, the paper 84,812",
    ),
    (
        "Detectors on rebuilt AgentDojo, tau-bench and BIPIA tool outputs",
        "[arXiv 2610.03448 v1](https://arxiv.org/abs/2610.03448), Table 2, 2 Oct 2026",
        "TPR at 1% FPR: PG2 86M 69.4 / 57.9 / 10.4; PIGuard 2.1 / 52.7 / 95.1",
        "rows must be rebuilt with AgentDojo and tau-bench replay; not rebuilt here",
    ),
    (
        "Sentinel on wildjailbreak",
        "Sentinel Table 2",
        "F1 0.935",
        "gated dataset; split and threshold unstated",
    ),
    (
        "ShieldGemma Table 1",
        "[arXiv 2407.21772 v2](https://arxiv.org/abs/2407.21772)",
        "OpenAI Mod F1 0.821, ToxicChat F1 0.743 (9B)",
        "F1 at a threshold tuned on the test set",
    ),
    (
        "Qwen3Guard Avg column",
        "Qwen3Guard Table 2",
        "8B Avg 90.0",
        "best of strict and loose per benchmark, over seven benchmarks",
    ),
    (
        "SecAlign, Meta-SecAlign, StruQ, Spotlighting",
        "see docs/sota-review.md, section on system defences",
        "attack success rate of a fine-tuned or prompted LLM",
        "measures a generator under static attacks, not a detector's decision",
    ),
)


def format_value(kind: str, value: float) -> str:
    """A published value as printed in the score column."""
    if kind == "tpr":
        return f"{value:.2%}"
    return f"{value:.3f}"


def not_comparable_rows() -> list[tuple[str, ...]]:
    """The static rows, the contaminated or unstated-split rows of PUBLISHED and the JBB F1 rows."""
    rows: list[tuple[str, ...]] = [tuple(r) for r in NOT_COMPARABLE]
    for p in PUBLISHED:
        if p.comparable:
            rows.append(
                (
                    f"{p.system} on {p.suite}",
                    SOURCES[p.source].cite(),
                    f"{p.metric} {p.value:.3f}",
                    p.comparable,
                )
            )
    src = SOURCES["mozilla"]
    for name, v in MOZILLA_JBB.items():
        rows.append(
            (
                f"any-guardrail `{name}` on JailbreakBench Behaviors",
                src.cite(),
                f"F1 {v:.3f} (harmful = positive, split `all`)",
                "private harness; model variant and the 200-row selection are not recorded",
            )
        )
    return rows

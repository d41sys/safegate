# Protocol: safegate against published guards on further public sets

Fixed on 26 Oct 2026, **before** any score of safegate on these sets was produced. The published
numbers it will be set against are committed in the same change (`src/safegate/published.py`).
The code that scores follows in the next commit and may not change this protocol; a change after
scoring is listed under "Deviations" in the report with the reason.

## Question

Where does safegate's default gate (`--stage2 v2`) and its legacy gate (`--stage2 legacy`) stand
against guards whose authors or third parties published scores on a public set, when the rows,
the metric and the cutoff are the same? Nothing is run but safegate: it is model-free, so no
guard model is downloaded or loaded. Every other number is transcribed from a page named in
`docs/sota-review.md` and `published.py`.

## Sets

Each set is the official split at a pinned revision, **not cleaned, de-duplicated or sampled**,
so the rows are the rows the published numbers used. Cached under `data/raw/<key>/`
(gitignored).

| `--suite` | Rows scored | Label 1 | Revision |
| --- | --- | --- | --- |
| `promptshield` | `test.json`, 23,516 (17,030 benign, 6,486 injection) | injection | HF `hendzh/PromptShield` `a5234cb1` |
| `deepset` | test, 116 (60 / 56) | injection | `deepset/prompt-injections` `4f61ecb0` |
| `jailbreak` | balanced test, 262 (139 / 123) | jailbreak | `jackhhao/jailbreak-classification` `2f2ceeb3` |
| `xtram1` | test, 2,060 (650 / 1,410) | injection | `xTRam1/safe-guard-prompt-injection` `a3a877d6` |
| `slabs` | `data/test.csv`, 2,101 (1,051 / 1,050) | injection | `S-Labs/prompt-injection-dataset` `002a9dd1` |
| `jbb` | `Goal` column of both files, 200 (100 / 100) | harmful request | `JailbreakBench/JBB-Behaviors` `886acc35` |
| `gandalf_test` | test, 112, attacks only | attack | `Lakera/gandalf_ignore_instructions` `04737b65` |
| `harmbench` | test behaviours, `standard` + `copyright`, 239, harmful only | harmful request | HarmBench GitHub `8e1604d1` |
| `toxicchat` | 0124 test, `human_annotation` true, 2,853 (2,491 / 362) | `toxicity` = 1 | `lmsys/toxic-chat` `29df8e4d` |

Row counts were profiled (counts and label columns only) before this protocol; the profile
matches the published counts (WildGuard Table 14: ToxicChat 2,853 with 362 harmful and 91
adversarial, HarmBench 239).

## Gates and configurations

* Gates: default (`v2`) and legacy, both through `SafetyGate` with no other change.
* Configurations: **gate alone**, and **shipped defence without windows** (the windows cell was
  rejected in `docs/protocol-window-150.md`).
* Flag: the gate's action is `refuse` or `escalate`. `review` and `allow` count as not flagged.

## Metrics

| Metric | Definition | Used by |
| --- | --- | --- |
| F1 | of label 1, from the flag | deepset, jailbreak, JBB (Mozilla), ToxicChat |
| recall | share of label 1 flagged | Gandalf |
| F1 on all-harmful | 2R / (1 + R), which is F1 when every row is positive | HarmBench |
| AUC | ROC AUC of the ranking score, ties at mid-ranks | PromptShield, xTRam1, S-Labs, JBB |
| TPR at FPR 1%, 0.5%, 0.1%, 0.05% | PromptShield Table 4 | PromptShield |

**Ranking score.** `2 * flagged + detection_score`, where `detection_score` is the detector score
every decision carries. Flagged prompts rank above allowed ones, each group by detector score, so
the ROC curve passes through the gate's own operating point. Stage-1 regex hits score at least
0.95, so ties are many; AUC uses mid-ranks. AUC and TPR at FPR are scored for the gate alone; the
defence adds views and not a score.

**TPR at a target FPR.** The highest TPR whose FPR on the test rows is **at or below** the target,
thresholds placed between distinct scores; the achieved FPR is printed. PromptShield sets the
threshold on the test rows themselves and accepts an FPR within 25% of the target (their text,
Table 4 footnote: the value is 0% when no threshold reaches the target). Ours is stricter than
theirs at the same target, which favours the published rows; it also uses the test rows to place
the threshold, as they did. No validation-calibrated row is produced.

**Intervals.** 95% Wilson for recall and rates; bootstrap over rows (1,000 resamples, seed
20261026, percentile) for F1 and AUC.

## Comparability rule

A published number is printed next to safegate's only if **set, split, rows, metric and cutoff are
identical**: the source states them, or its counts or values reproduce them (e.g. every Mozilla
Gandalf value is k / 112). Everything else goes in the separate "not comparable" table with the
reason: private data, gated data, unstated split, threshold tuned on the test set, best-of-modes
average, rows redacted, or a different quantity (attack success of a generator).

Each published row names the source, the table and page, the version and its date, and who ran
it. Vendor-run results are marked. Published numbers for the same guard differ across papers
(e.g. Llama Guard 3 on ToxicChat 53.12 against 53.8); all are shown, none averaged.

## Contamination

Safegate's default stage 2 is fitted on the bundled texts and on the TRAIN splits of deepset,
jackhhao, the in-the-wild jailbreak prompts and neuralchemy. Each scored row is matched on a
normalised key against that training data. The report prints the number of matching rows per set
and, when it is above zero, scores the remaining rows as well, marked "not comparable (subset)".
HarmBench is expected to match, since neuralchemy contains HarmBench-derived prompts. Published
guards that trained on a set's TRAIN split are named in the report (PIGuard on deepset, jackhhao
and xTRam1 TRAIN; ProtectAI v2 on xTRam1) and are not removed.

## Reading the result

* Win: safegate's point estimate is above the published number and the lower interval bound is
  above it too. Loss: the upper bound is below it. Otherwise: inconclusive.
* The two harm sets (HarmBench, ToxicChat) and JBB are harm detection. Safegate's
  `KeywordClassifier` matches placeholder tokens, not harm categories, so only its injection
  detector can flag them. Low scores there are a scope fact, not a bug; they are printed so the
  scope is visible.
* Gandalf is attacks only: recall rewards flagging everything and says nothing about false
  positives.
* Vendor-run or maintainer-run numbers (Qualifire, Horizon, Bastion, Mozilla) are marked as such.

## Disclosure

Before this protocol the data was profiled (row counts, label counts, text lengths) and the gate
was timed on the first 600 PromptShield rows (block counts only: 94 for v2, 227 for legacy). No
metric was computed and no threshold or rule was chosen from those runs.

## Not scored, and why

PINT (private), the Prompt Guard 2 and LlamaFirewall evaluations (private Meta sets), the
Qualifire/"rogue" benchmark and WildGuardTest (gated), Aegis 2.0 (36 redacted prompts, count
mismatch between papers), GenTel-Bench (evaluated subset unstated), the AgentDojo/tau-bench
rebuild of arXiv 2610.03448 (needs those frameworks), and static attack-success rates of
fine-tuned generators. The list with sources is printed by `safegate benchmark` and kept in
`docs/sota-review.md`.

## Amendments after scoring

The rules above were fixed before any metric was computed (commit `3a722ef`). The scoring run
(`runs/sota-public.md`, not committed) then showed three things that changed what the report
prints. Nothing below changed a score or a threshold.

1. **TPR at a fixed FPR, second rule.** The strict rule (the highest TPR whose FPR is at most the
   target) gives 0.00% at 0.5% FPR on PromptShield, because 1,311 attacks and 86 benign prompts
   share the top score (stage-1 regex hits): the gate has no operating point between 0% and
   0.505% FPR. The PromptShield paper reads its numbers with a tolerance (FPR up to 1.25 times the
   target; `PAPER_TOLERANCE = 0.25` in `metrics.py`), so rows under that rule were added, marked
   "yes, added after scoring". Both rules are printed; the strict row is still the pre-registered
   one. Result under the tolerance rule: 20.21% (achieved FPR 0.581%) at 0.5% FPR, 20.21% / 20.40% at 1%;
   0.00% at 0.1% and 0.05% under both.
2. **Leak expectation was wrong for HarmBench.** Section "Contamination" expects HarmBench to match
   neuralchemy heavily. The measured exact-text overlap is 5/239. The larger leaks are xTRam1
   (110/2,060), jackhhao (99/262) and JailbreakBench (130/200).
3. **Presentation only.** The source column was shortened and a "Sources and their caveats" list
   (who ran each number, harness and variant caveats) was added after the tables; an AUC that is not
   defined prints `n/a` instead of `nan`; the ACL 2025 camera-ready NotInject baselines of four
   guards are printed next to the arXiv v3 ones. The ranking score, the configurations, the
   comparability rule and the decision rules are unchanged.

## Update, 2026-10-29: one model run through an API

The rule above, that nothing is run but safegate, still holds for the twelve sets of this protocol.
`docs/protocol-llm.md` adds one row to the InjecGuard-set table: claude-sonnet-5-5 as a zero-shot
guard through the Anthropic API, with its own protocol, dev set (50 items of the bundled training
data) and a 600-row subsample of WildGuard benign. No published number was changed.

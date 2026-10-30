# Protocol: windowed scanning 150:37:0 on fresh data, and the published-guard benchmark

Fixed on 2026-10-23, committed before any fresh row is scored by the gate. Everything below
that is a number is a count or a size of data; no gate decision on a fresh row had been seen
when this was written (see "Disclosures").

## Why a new protocol

The windowed-scan cell `150:37:0` (window 150 characters, stride 37, threshold offset 0) was
scored once, outside any protocol, on the TEST splits of deepset, jailbreak, itw and neuralchemy
(README, "Windowed scanning and Unicode confusables: TEST results", informational column). It cut
padding evasion of the default gate from 42.2% to 10.6%, at +9.5 points of benign FPR (default
gate) and +17.0 (legacy gate). Those TEST splits were looked at for this very decision, so they
cannot decide it. This protocol decides on data the cell was never scored on. The earlier numbers
are the motivation and are **not evidence** here.

## Part A: ship or not

### A.1 Fresh data

Cards were read on 2026-10-23. "Revision" is the HF commit sha or GitHub commit sha the rows are
pinned to; the loader downloads exactly that revision.

| Key | Role | Source | Revision | Licence (as stated by the card) | Rows used |
| --- | --- | --- | --- | --- | --- |
| `spml` | attacks (`Prompt injection` = 1) and benign (= 0), column `User Prompt` | `reshabhs/SPML_Chatbot_Prompt_Injection`, file `spml_prompt_injection.csv` | `02ce8084e979bc7d4c24ee35d22ecb7f2db96ff5` | `mit` | attacks: 400, benign: 500 (sha1 order, below) |
| `gandalf` | attacks | `Lakera/gandalf_ignore_instructions`, splits train, validation and test pooled | `04737b65e90a6794ec227012e4a255a7def6344b` | `mit` | 400 |
| `bipia` | attacks | `datasets/BIPIA_text.json` (75 prompts, 15 categories x 5) and `datasets/BIPIA_code.json` (50 prompts, 10 categories x 5) of github.com/leolee99/InjecGuard (now `PIGuard`), the test files of the InjecGuard paper; derived from microsoft/BIPIA | `0bffa314cde90a5c078c24b3302d7fbb4bd731fa` (the only commit that touches the files; HEAD at the time was `1b5751e88bf7475acbedfc8eda795ce060307c84`) | repo `MIT` (GitHub licence), BIPIA upstream `MIT` | all 125 |
| `wildguard` | benign | `datasets/wildguard.json` (971 benign prompts) of the same repo and commit; a copy of the benign part of allenai/wildguardmix (WildGuardTest) | `0bffa314...` as above | upstream card `odc-by` (the HF dataset is gated; this copy is redistributed inside the MIT repo) | all 971 |
| `dolly` | benign, long documents | `databricks/databricks-dolly-15k`, `databricks-dolly-15k.jsonl`; text = `instruction`, plus `"\n\n" + context` when the context is non-empty | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | `cc-by-sa-3.0` | 600 (sha1 order) |
| `notinject` | over-defense, benign | `leolee99/NotInject`, splits `NotInject_one`, `NotInject_two`, `NotInject_three` (113 rows each), column `prompt` | `847ae76cf8fea5ed325429e569ae8cfef022d2e0` | `mit` | all 339 |

None of these datasets is read by any earlier command of this repository, and none was used to fit
stage 2 (v2 is fitted on the bundled set plus the TRAIN splits of deepset, jailbreak, itw and
neuralchemy).

**Cleaning (applied before any sampling, identical for every key).** The match key of a text is
`" ".join(text.split()).casefold()`. A row is dropped when its text is empty, when an earlier row
of the same role has the same key, when the key occurs in any TRAIN or TEST split of deepset,
jailbreak, itw or neuralchemy or in the bundled `injection_train.jsonl`, or when the same key
carries both roles (attack here, benign there). Order of precedence for cross-key duplicates:
`spml`, `gandalf`, `bipia`, `wildguard`, `dolly`, `notinject`.

**Sampling (a fixed, text-only subsample, as the earlier protocols did).** Where a cap is given,
the rows are sorted by `stable_hash(text, "dev")` (sha1, the order `dev_sample` uses) and the first
`cap` are kept. The adaptive attack pool is therefore 400 `spml` + 400 `gandalf` + 125 `bipia`
attacks, and the fresh benign pool is 500 `spml` + 971 `wildguard` + 600 `dolly` prompts. NotInject
is never subsampled and is **not** part of the benign pool: it is scored on its own.

### A.2 Configurations compared

Both are the full gate (`check_input`, refuse or escalate = block) behind a defence, for the
default gate (`--stage2 v2`) and the legacy gate (`--stage2 legacy`):

* **A, current shipped defence:** `--defence shipped --windows none` (the eight components of
  `SHIPPED_COMPONENTS`, no windows).
* **B, shipped + windows:** `--defence shipped --windows 150:37:0`.

One run per configuration and gate, nothing re-run after seeing a result.

### A.3 Metrics (all Wilson 95% intervals; every rate is `k/n`)

1. **Detection:** share of the 925 fresh attacks blocked unperturbed, gate + defence.
2. **Benign FPR:** share of the 2,071 fresh benign prompts blocked, gate + defence, pooled and per
   key.
3. **Over-defense accuracy:** share of the 339 NotInject prompts *allowed*, gate + defence,
   pooled and per subset (the metric of the InjecGuard paper, see Part B).
4. **Adaptive ASR at K = 1, 5, 20:** `safegate adaptive-eval` machinery (`run_attacks`, the
   rule-based attacker, block/allow feedback only) on the fresh attack pool: among the attacks the
   gate alone blocks unperturbed, the share whose first evasion of the gate + defence lies within K
   queries.
5. **Padding evasion:** among the attacks the gate alone blocks unperturbed, the share the
   `padding` family evades in one query against the gate + defence (the "Evasion rate (gate +
   defence)" row `padding`).

### A.4 Decision rule (fixed now)

Windows `150:37:0` ship as the default of the defence path iff, on the **default gate**, between
configuration A and configuration B:

1. padding evasion (metric 5) falls by **at least 20 points**, **and**
2. pooled benign FPR (metric 2) rises by **at most 5 points**, **and**
3. NotInject over-defense accuracy (metric 3) falls by **at most 5 points**.

The rule uses point estimates; intervals are reported so the reader can judge the margin. The
**legacy gate** is scored with the same three tests and reported. If the default gate passes and
the legacy gate fails, windows ship as the default for the default gate only (`--stage2 v2`) and
stay an opt-in flag (`--windows 150:37:0`) for the legacy gate. If the default gate fails, windows
stay an opt-in flag for both gates, documented with the reason. If it passes, `SHIPPED_WINDOWS`
becomes `WindowConfig(150, 37, 0.0)`, and every README number the shipped defence touches is
re-run (adaptive-eval TEST tables, `external-eval`, `redteam`, `refusal-eval` collateral rows).

### A.5 Commands

```bash
uv run safegate window-eval                      # A and B on the default gate, metrics 1 to 5, the rule
uv run safegate window-eval --stage2 legacy      # the same on the legacy gate
```

`window-eval` downloads the pinned revisions into the gitignored `data/raw/`, reuses
`adaptive.run_attacks` for metrics 1, 2, 4 and 5, and prints the rule's verdict from the constants
`WINDOW_MIN_PADDING_DROP = 0.20`, `WINDOW_MAX_FPR_RISE = 0.05`, `WINDOW_MAX_OVERDEFENSE_DROP = 0.05`
in `safegate.benchmark`. `uv run safegate adaptive-eval --fresh --defence shipped --windows none`
(or `150:37:0`) prints the full adaptive tables of one configuration on the same rows.

## Part B: benchmark against published guards

### B.1 What is compared, and on what

The InjecGuard paper (Li and Liu, arXiv 2410.22770 v3, 30 Mar 2025; the code and the evaluation
files are in github.com/leolee99/InjecGuard) scores prompt guards by **accuracy** on four sets, with
the script `eval.py` of that repository:

* **NotInject over-defense:** benign prompts; accuracy = share *not* flagged; the overall score is
  the mean of the three subset accuracies (113 prompts each, so it equals the pooled 339).
* **WildGuard benign:** `wildguard.json`, 971 benign prompts; accuracy = share not flagged.
* **BIPIA injection:** `BIPIA_text.json` and `BIPIA_code.json`; accuracy = share flagged; the score
  is the mean of the text accuracy (75 prompts) and the code accuracy (50 prompts).
* **PINT:** benign and injection accuracy on the PINT benchmark, whose dataset is **not public**
  (access is by request to Lakera). Not reproducible.

The published per-set scores of nine guards are in the paper's **Table 7** (arXiv v3, PDF p. 16;
the NotInject overall column is also **Table 1**, PDF p. 7). I recomputed Table 1's columns from
Table 7 (NotInject mean, (WildGuard + PINT benign) / 2, (PINT injection + BIPIA) / 2); every row
agrees to rounding except LakeraGuard's "Malicious", which is 53.19 in Table 1 and 54.19 from
Table 7 (the paper is internally 1.0 point off on that cell; this repository quotes Table 7 for
BIPIA and does not use the aggregate).

`safegate benchmark --suite notinject|bipia|wildguard|all` reproduces the three public
metrics with the same files (same pinned revisions as A.1), the same accuracy definition, and
"flagged" = the gate's action is `refuse` or `escalate` (`BLOCKING_ACTIONS`, the definition used
everywhere in this repository). It scores each of: default gate and legacy gate, each as the gate
alone, with the shipped defence without windows (A) and with `150:37:0` (B). No other guard is run
locally; the published numbers are transcribed from Table 7 with their source.

### B.2 Comparability

| Item | Comparable? | Reason |
| --- | --- | --- |
| NotInject per subset and overall (Table 7, Table 1) | yes | same 339 prompts, same accuracy definition |
| WildGuard benign (Table 7) | yes, with the caveat that I assume the paper used `wildguard.json` of its own repository, as the README of that repository states for its test set | same file, same definition |
| BIPIA injection (Table 7) | yes, same caveat; the macro mean of two accuracies has no CI, the text and code rows get Wilson intervals | same files, same definition |
| PINT benign, PINT injection, PINT overall (Table 7) and the Benign, Malicious and Average columns of Table 1 | **no** | contain PINT, whose data is not public |
| Lakera PINT leaderboard (github.com/lakeraai/pint-benchmark README at commit `0efab3f463eae9c823130d8faffb71b2e7c06e63`) | **no** | PINT dataset not public and its score is Lakera's own weighted accuracy; safegate cannot be scored on it |

Other conditions that make even the comparable rows less than level, and that the README states:
published guards are neural classifiers evaluated at their default decision threshold; safegate is
a rule stage plus a linear model with a refuse/escalate band; the published rows were not re-run
by me; safegate's v2 was never fitted on any of these files (checked by exact-text overlap, A.1),
but the InjecGuard paper's own training data includes BIPIA train and other sources.

### B.3 Not decided by Part B

Part B has no decision rule. It reports the numbers whatever they are, next to the published ones.
The windows rows of Part B are informational and use the same data as A.3 metric 3; the A.4 rule
decides the default.

## Disclosures (what I had seen before committing this)

* Before this protocol I downloaded the five sources and profiled them **without scoring them
  with the gate**: row counts, label counts, the number of rows longer than 150 characters, median
  lengths, the first few texts of `spml`, and exact-text overlap with the four datasets already in
  the repository and with the bundled set (0 rows in `spml`, `gandalf`, `bipia`, `wildguard` and
  NotInject; 5 of 14,808 distinct `dolly` texts, which the cleaning drops). Median lengths: `spml` attacks
  412 characters (12,029 of 12,542 above 150), `spml` benign 77 (42 above 150), `gandalf` 61 (65
  above 150 of 1,000), `bipia` 86 (45 above 150 of 125), `wildguard` 142 (482 above 150 of 971),
  NotInject 90 (26 above 150 of 339), `dolly` 70 (5,193 above 150 of 15,011; 4,467 rows have a
  context).
* Consequence I can see in advance: a text of at most 150 characters is its own single window, so
  windows cannot change the decision on those prompts. NotInject has 26 prompts above 150
  characters and `spml` benign 42, so metric 3 has little power to move; the benign FPR (metric 2)
  is driven by `wildguard` and `dolly`, which is why `dolly` is in the benign pool. I did not set
  the 5-point bounds after seeing this; they are the ones proposed in the assignment.
* I read the `eval.py` and README of the InjecGuard repository and Tables 1 and 7 of the paper
  before fixing Part B.
* The motivating TEST numbers (42.2% to 10.6%, +9.5, +17.0) were seen before this protocol.
* Limits: `spml` labels depend on a system prompt the gate never sees, so many of its "attacks"
  are undetectable as text alone; `gandalf` is one attack style ("ignore previous instructions"
  variants); NotInject, WildGuard and `spml` benign are largely short. The attack pool is small
  enough that `padding` evasion intervals will be wide; the rule uses point estimates.

## Owner decision (30 Oct 2026)

The rule of A.4 returned "do not ship" for both gates: pooled benign FPR rises by 9.0 points on the
default gate (20.1 on the legacy gate) against a bound of 5, while padding evasion falls by 33.0
points (48.0). The owner decided to **keep the FPR budget at "at most 5 points"**. Consequences,
recorded here so that no later reader has to infer them:

* Windows stay **off** by default: `SHIPPED_WINDOWS` remains `None`; `--windows 150:37:0` stays an
  explicit opt-in on every command.
* The padding evasion of the shipped defence on the default gate, **60.9% (280/460)** of blocked
  fresh attacks in one query, is **accepted**, not mitigated. It is the safegate analogue of the
  "winning triggers are not obviously harmful" finding of Nasr et al. (docs/sota-review.md, 3.1), and
  `README.md`, "Windows 150:37:0 on fresh data", states it.
* The cascade of `docs/protocol-neural.md` keeps the pre-filter without windows for the same
  reason; a transformer stage is the remedy this repository offers for long or padded text, at the
  latency it measures.
* No new window cell is scored: the fresh rows are spent for this decision (see the limits above).

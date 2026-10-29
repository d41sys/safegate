# State of the art in prompt-injection and jailbreak defence, 2023-2026, and where safegate stands

Read on 26 Oct 2026. Every number below was read from the page named next to it, by one agent and
then (for the benchmark and system-defence numbers) again by a second one; the differences
between the two readings are in "Corrections found by the second reading" at the end. Version
and date are given for every paper because several changed between versions. Nothing was run
except safegate: no guard model was downloaded or loaded. (Section 5b, added 29 Oct 2026, is the
one exception: it reports claude-sonnet-5-5 called through the Anthropic API; still nothing local.)

Safegate is a model-free text gate: stage-1 regex families with base64 and ROT13 decoding, stage-2
logistic regression over signed hashed word and character n-grams (numpy only), an optional input
normaliser that adds extra views, and an abstain band that returns `escalate`. It runs in a few
milliseconds on a CPU (about 2 ms per prompt over the 23,516 PromptShield rows here) and its
weights contain no text. Everything else in this review is a fine-tuned language model, an LLM
prompted as a judge, or an architecture around an agent.

Contents: [1 Detectors](#1-detectors) · [2 System-level defences](#2-system-level-defences) ·
[3 Adaptive attacks](#3-adaptive-attacks) · [4 Benchmarks](#4-benchmarks) ·
[5 Head to head](#5-head-to-head-on-identical-public-sets) (and [5b a frontier model measured here](#5b-a-frontier-model-measured-here-as-a-guard-and-as-an-agent)) · [6 What safegate is and is not](#6-what-safegate-is-and-is-not) ·
[7 Not comparable](#7-published-numbers-that-cannot-be-compared) ·
[8 Corrections](#8-corrections-found-by-the-second-reading)

## 1. Detectors

Role: PI = prompt or indirect injection, JB = jailbreak, HARM = content-harm taxonomy. "Vendor"
means the authors of the system ran the evaluation.

| System | What it is | Threat model | Headline result and protocol | Difference from safegate |
| --- | --- | --- | --- | --- |
| **Llama Prompt Guard 2** 86M / 22M (Meta, 29 Apr 2025) | mDeBERTa-base (86M) and DeBERTa-xsmall (22M) classifier, benign or malicious, 512-token window, energy-based loss, modified tokenizer | JB and PI: "explicitly attempts to override prior instructions" | Private Meta benchmark: AUC .998 (86M), recall at 1% FPR 97.5% (86M), 88.7% (22M); AgentDojo APR at 3% utility reduction 81.2% / 78.4% vs ProtectAI 22.2%, Deepset 13.5%. [Model card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md), vendor. The LlamaFirewall paper prints AUC .98 for the 86M model on the same table | Same job, a pretrained transformer (about 92 ms per call on an A100 per the card) against hashed n-grams. Generalises to paraphrase and other languages, safegate does not. Weights are gated, safegate's are in the repository |
| **PIGuard** (InjecGuard; Li, Liu, Zhang, Xiao; ACL 2025) | DeBERTa-v3-base 184M trained with MOF (mitigating over-defense for free): trigger-word-biased augmentation | PI, with over-defense as the design target | [arXiv 2410.22770 v3, 30 Mar 2025](https://arxiv.org/abs/2410.22770), Table 1, accuracy %: NotInject 87.32, benign 85.74, malicious 77.39, average 83.48. The ACL camera-ready re-ran four baselines on NotInject (see section 5). Authors' run | Same protocol as safegate's existing `benchmark`. PIGuard trains on the TRAIN splits of deepset (546), jackhhao (1,044) and xTRam1 (8,236) and on BIPIA train, so its BIPIA column is not held out: [arXiv 2610.03448](https://arxiv.org/abs/2610.03448) finds 1,116 BIPIA inputs in its training split |
| **ProtectAI deberta-v3-base-prompt-injection-v2** (Apr 2024) | DeBERTa-v3-base 184M, archived | PI only; "does not detect jailbreak attacks or handle non-English" | Card: held-out 20,000 prompts, accuracy 95.25, F1 95.49 (vendor, prompts unpublished) | Same job; used as the baseline in almost every table here |
| **PromptShield detector** (Jacob et al., [arXiv 2501.15145 v2](https://arxiv.org/abs/2501.15145), 12 Apr 2025) | DeBERTa-v3-base and Llama-3.1-8B fine-tuned on the PromptShield training set | PI, with low-FPR deployment as the target | Table 4 on the paper's own 23.5k-row test set: AUC .998 and TPR 94.80% at 1% FPR (Llama-3.1-8B), 43.22% at 1% FPR (DeBERTa); InjecGuard 20.37%, PromptGuard 12.78% | Defines the metric (TPR at fixed low FPR) that exposes the AUC-ranking mismatch; safegate is scored on it in section 5 |
| **Qualifire Sentinel** ([arXiv 2506.05446 v1](https://arxiv.org/abs/2506.05446), 5 Jun 2025) | ModernBERT-large 395M, trained on public and private data | PI and jailbreak | Table 2, F1: wildjailbreak 0.935, jackhhao 0.985, deepset 0.857, own benchmark 0.976 (ProtectAI v2 0.733, 0.915, 0.536, 0.652). Vendor; split and threshold not stated | Same job; the only guard with a published deepset F1 that is not its own sales claim |
| **Llama Guard 4** (Meta, 29 Apr 2025) | 12B dense model pruned from Llama 4 Scout, natively multimodal | HARM (MLCommons S1-S13, S14 code-interpreter abuse) | In-house set, output filtering: EN recall 69%, FPR 11%, F1 61% (card, "we omit evals against competitor models"). Third parties: prompt F1 average 75.9 against Llama Guard 3's 79.4 ([Qwen3Guard Table 2](https://arxiv.org/abs/2510.14276)) | Harm taxonomy, not injection; needs a GPU |
| **Qwen3Guard** Gen / Stream 0.6-8B ([arXiv 2510.14276 v1](https://arxiv.org/abs/2510.14276), 16 Oct 2025) | Qwen3 fine-tuned on 1.19M labelled samples, safe / controversial / unsafe, 119 languages; Stream adds a token-level head | HARM, with a Jailbreak category for input | Table 2, English prompt F1 (best of strict and loose per benchmark): 8B average 90.0; WildGuard-7B 85.8, LG4-12B 75.9. Authors' run; the average is an oracle over two modes | LLM-sized semantic guard; no indirect-injection benchmark. Its per-benchmark mode choice is not a single operating point |
| **ShieldGemma** 2B / 9B / 27B ([arXiv 2407.21772 v2](https://arxiv.org/abs/2407.21772), 4 Aug 2024) | Gemma 2 fine-tuned; Yes/No token probability per policy | HARM (sexually explicit, hate, dangerous, harassment) | Table 1, optimal F1 / AU-PRC: OpenAI Mod 0.821 / 0.907 (9B), ToxicChat 0.743 / 0.802 (9B), thresholds tuned on the test set. Third parties print much lower: HarmBench prompt F1 60.6 (Qwen3Guard Table 2) | Per-policy LLM scorer with a calibrated probability, safegate's abstain band is the analogue |
| **WildGuard** 7B ([arXiv 2406.18495 v3](https://arxiv.org/abs/2406.18495), 9 Dec 2024) | Mistral-7B fine-tuned on 86,759 items, three tasks: prompt harm, response harm, refusal | HARM including adversarial jailbreak prompts | Table 3 prompt F1 average 86.1 on five public sets; WildGuardTest prompt F1 88.9; as a moderator it cuts jailbreak success from 79.8% to 2.4%. Authors' run | Safegate's "WildGuard benign" column is the benign half of WildGuardMix |
| **Granite Guardian** 3.0-3.3 (IBM, [arXiv 2412.07724 v2](https://arxiv.org/abs/2412.07724)) | Granite LLM fine-tuned; risk chosen by a criteria string, including `jailbreaking`, RAG groundedness, function-call hallucination | HARM and RAG | Paper Table 6 aggregate (3.0-8B): AUC 0.871, F1 0.758. Card 3.3-8B: harm aggregate F1 0.81, HarmBench prompt 0.80, ToxicChat 0.76 (self-reported, split not stated) | Criteria-prompted LLM; no PI benchmark in the card |
| **GuardReasoner** 1B / 3B / 8B ([arXiv 2501.18492 v2](https://arxiv.org/abs/2501.18492), 17 Oct 2025) | Reasoning fine-tune plus hard-sample DPO on 127K samples | HARM prompt, response, refusal | Table 2 F1: ToxicChat 78.79 and HarmBench 91.86 (8B); the abstract claims +5.74% over GPT-4o+CoT and +20.84% over Llama Guard 3 on average | Harm only; reasoning trace costs tokens |
| **gpt-oss-safeguard** 20B / 120B (OpenAI, 29 Oct 2025) | gpt-oss fine-tuned to reason from a policy given at inference | Any developer policy | Vendor: OpenAI Mod F1 82.9, ToxicChat F1 79.3 / 79.9; no PI benchmark | The opposite of a fixed gate; a candidate escalation target for the abstain band |
| **Constitutional Classifiers** and ++ ([2501.18837](https://arxiv.org/abs/2501.18837), 31 Jan 2025; [2601.04603](https://arxiv.org/abs/2601.04603), 8 Jan 2026) | Claude-based input and output classifiers trained on constitution-generated data; ++ adds a cascade and linear probes | Universal jailbreaks in CBRN | v1: 3,000+ estimated red-team hours, no universal jailbreak at the detail of an unguarded model; automated jailbreak success 86% to under 5%; +0.38% refusals, +23.7% compute. ++: 40x cost cut versus the exchange baseline, 0.05% flag rate. Closed | Different scale and threat. The transferable idea, a refusal-rate budget and a cheap first stage that escalates, is what safegate's thresholds already do |
| **Lakera Guard**, Azure Prompt Shield, AWS Bedrock Guardrails, Google Model Armor | Commercial APIs | PI, JB | PINT score (private data): Lakera 95.22, Bedrock 89.24, Azure 89.12, Model Armor 70.07. Vendor-run | Not scorable; see section 7 |

**Observations that matter for the comparison.**

1. Detector scores depend on the operating point more than on the model. At 1% FPR, Prompt Guard 2
   86M detects 69.4% of AgentDojo injections but 10.4% of BIPIA; PIGuard 2.1% against 95.1%
   ([arXiv 2610.03448 v1](https://arxiv.org/abs/2610.03448), 2 Oct 2026, Table 2, third party).
   Independent AUROC of Prompt Guard 2 on 47,611 held-out examples is 0.757 (86M) and 0.714 (22M)
   ([arXiv 2610.20907 v1](https://arxiv.org/abs/2610.20907), 14 Oct 2026), far from Meta's .998
   on its private set. Safegate therefore reports AUC and TPR at fixed FPR next to the
   operating-point scores.
2. Harm guards (Llama Guard, Qwen3Guard, ShieldGemma, Granite, WildGuard, gpt-oss-safeguard)
   publish no injection benchmark and are themselves injectable: a forged query appended after the
   request lets guards clear flagged items ([arXiv 2610.31695 v1](https://arxiv.org/abs/2610.31695),
   20 Oct 2026, single author). A regex and n-gram gate has no instruction channel to hijack.
3. Vendor-run numbers dominate. Third-party runs of the same guard on the same rows (Horizon,
   Mozilla.ai, Bastion) agree with each other to three decimals where their rows are the same
   (for example Prompt Guard 2 86M on deepset: 0.235 in both Horizon and Mozilla), but all three
   have a stake or use a private harness; see the caveat list printed by `safegate benchmark`.

## 2. System-level defences

Safegate sits in front of the model; these sit in or around it. None is a substitute for the other:
a gate cannot give the guarantees of a design that cuts the data path.

| Defence | Type | Version and date | Headline result as printed | Adaptive attack in the paper? | Difference from safegate |
| --- | --- | --- | --- | --- | --- |
| **CaMeL** (Debenedetti et al.) | Architecture: a privileged LLM writes a plan as Python, a quarantined LLM parses untrusted data, an interpreter tracks capabilities and enforces policies | [arXiv 2503.18813 v2](https://arxiv.org/abs/2503.18813), 24 Jun 2025 (v1 24 Mar 2025) | Solves 77% of AgentDojo tasks with provable security against 84% undefended (abstract). §6.2.1: 949 attacks; Gemini 2.5 Pro 300 to 0 (but Table 4 prints Flash 297 and Pro 163, see section 8), GPT-4o-mini with instruction hierarchy 276 to 0; Claude 3.5 Sonnet: CaMeL 0, next best defence (tool filter) 8. Claude 4 Sonnet still 11-13 attacks in Table 4, on two injection tasks the paper puts outside its threat model | No; §7 lists side channels (indirect inference, exception 1-bit leak, timing). Nasr et al. skip it: its control flow is data-independent, so their search attack does not apply | A guarantee by construction instead of a score; costs utility (77% against 84%) and needs an agent rewrite. A gate in front of it adds nothing to the guarantee |
| **Agents Rule of Two** (Meta, [blog](https://ai.meta.com/blog/practical-ai-agent-security/), 31 Oct 2025) and the **lethal trifecta** (Willison, [16 Jun 2025](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)) | Design rule | n/a | No numbers. Per session an agent should satisfy no more than two of: [A] processes untrustworthy inputs, [B] has access to sensitive systems or private data, [C] can change state or communicate externally; otherwise human-in-the-loop. Willison: guardrails that catch "95% of attacks" are "very much a failing grade" in web security | n/a | Says where a probabilistic gate is not allowed to be the only control: when the agent holds all three properties |
| **LlamaFirewall** (Meta, [arXiv 2505.03574 v1](https://arxiv.org/abs/2505.03574), 6 May 2025) | Layers: Prompt Guard 2, AlignmentCheck (LLM auditor of the agent's chain of thought), CodeShield | v1 only | §4.3.2, AgentDojo replay of ten models, attack success / utility: none .1763 / .4773; Prompt Guard 2 86M .0753 / .4701; AlignmentCheck (Llama 4 Maverick) .0289 / .4309; combined .0175 / .4268. The paper calls AgentDojo "a narrow class of attacks". Vendor | No | Prompt Guard 2 inside it is safegate's closest relative; Nasr et al. then measure that classifier at 94% search ASR |
| **PromptArmor** ([arXiv 2507.15219 v1](https://arxiv.org/abs/2507.15219), 21 Jul 2025) | An off-the-shelf LLM prompted to detect and remove injected spans from tool output | v1 only | Table 1, AgentDojo, FPR / FNR / utility / ASR %: no defence - / - / 64.27 / 54.53; GPT-4o 0.07 / 0.23 / 68.68 / 0.47; GPT-4.1 0.56 / 0.13 / 72.02 / 0.00; o4-mini 0.34 / 0.47 / 76.35 / 0.08. In the same table: fine-tuned DeBERTa 28.41 / 22.03 / 29.73 / 18.92, DataSentinel 0.10 / 48.78 / 46.38 / 38.63 | Yes, Table 3 (AgentVigil, GPT-4.1): ASR 36.99% undefended to 0.34% | Needs an LLM call per tool output; PISmith (below) breaks it at 1.00 ASR@10 |
| **SecAlign** (Chen et al., [arXiv 2410.05451 v3](https://arxiv.org/abs/2410.05451), 3 Jul 2025, CCS 2025) | Training: DPO on secure against insecure outputs | v3 | Abstract: reduces injection success "to <10%". §1: 0% on optimisation-free attacks, mainly <10% on optimisation-based ones, more than 4x better than StruQ; strongest attack 8% (Fig. 1) | Yes (GCG, AdvPrompter, NeuralExec) | Changes model weights; safegate cannot |
| **Meta-SecAlign** ([arXiv 2507.02735 v4](https://arxiv.org/abs/2507.02735), 28 Sep 2026; v1 3 Jul 2025) | Open-weight SecAlign-trained Llama-3.1-8B and 3.3-70B | v4 read | Table 4 (Llama-3.3-70B, undefended / SecAlign / Meta-SecAlign): AgentDojo utility 59.8 / 6.2 / 84.5; InjecAgent ASR 53.8 / 0 / 0.5; AgentDojo ASR 14.7 / 0 / 1.9. Table 5, GCG on AlpacaFarm: 98.1 / 53.8 / 47.1 (70B) | §5.2: "PI is not solved"; §5.3: "cannot resist strong adaptive attacks" | Model-side |
| **StruQ** ([arXiv 2402.06363 v2](https://arxiv.org/abs/2402.06363), 25 Sep 2024) | Structured queries: delimiters and separate channels, trained | v2 | §1: manual attacks <2%; Llama TAP 97% to 9%, GCG 97% to 58% (Mistral 100 to 36 and 99 to 56); "not yet fully secure" | Yes | Model-side |
| **Spotlighting** (Microsoft, [arXiv 2403.14720 v1](https://arxiv.org/abs/2403.14720), 20 Mar 2024) | Prompt transform: delimit, datamark or encode untrusted text | v1 | Abstract: attack success from more than 50% to below 2%. §5.1: datamarking GPT-3.5-Turbo about 50% to below 3%, Text-003 40% to 0%; only encoding reaches below 2% on GPT-3.5 | No; concedes delimiters can be subverted by an attacker who knows the prompt | Composes with a gate |
| **Instruction Hierarchy** (OpenAI, [arXiv 2404.13208 v1](https://arxiv.org/abs/2404.13208), 19 Apr 2024) | Fine-tune to rank system above user above tool | v1 | Robustness up to +63% in-domain, up to +34% held out (per-eval figure numbers not extractable) | No | Model-side |
| **DataSentinel** ([arXiv 2504.11358 v4](https://arxiv.org/abs/2504.11358), 12 Nov 2025, IEEE S&P 2025 distinguished paper) | Known-answer detector fine-tuned with a minimax adaptive attacker | v4 | §1: FPR close to 0, FNR at most 0.07 on 7 tasks, 9 attacks, 6 LLMs | Yes, 3 adaptive attacks, only where the injection changes the backend model's task | The closest "detector trained against its own attacker"; Nasr et al. still reach 80% |
| **MELON** ([arXiv 2502.05174 v4](https://arxiv.org/abs/2502.05174), 10 Jun 2025, ICML 2025) | Masked re-execution; flags an attack when the original and masked runs call the same tools | v4 | GPT-4o AgentDojo: MELON ASR 0.24% against 16.06% undefended | No | Needs two agent runs |
| **Design patterns** (Beurer-Kellner et al., [arXiv 2506.08837 v3](https://arxiv.org/abs/2506.08837), 27 Jun 2025), **Progent** ([arXiv 2504.11703 v3](https://arxiv.org/abs/2504.11703)), **FIDES** ([arXiv 2505.23643 v2](https://arxiv.org/abs/2505.23643)) | Patterns, symbolic tool-privilege policy, information-flow labels | as listed | Progent: ASR 39.9% to 1.0% on AgentDojo, 70.3% to 3.9% on ASB. FIDES: "stops all prompt injection attacks in AgentDojo" with policy checks on | Progent yes (its Fig. 10 bars not extractable); the others no | Architecture again |

**Frontier-lab numbers (single reading, not re-checked).** Claude Opus 5 system card, 24 Jul 2026,
§5.2: Gray Swan indirect-injection set, k=15 attempts, Opus 5 2.0% (Opus 4.8 5.5%); an adaptive
coding attacker, 200 attempts per scenario, 0.56% without safeguards and 0.18% with probes. GPT-6
Astra system card, 3 Sep 2026: Gray Swan arena ASR over 15 attempts 8.5% against 27.0% for GPT-5.6
Sol. Gray Swan competition, [arXiv 2603.15714](https://arxiv.org/abs/2603.15714), 16 Mar 2026: 464
participants, 272,000 attempts, ASR from 0.5% to 8.5% across 13 models. NIST/CAISI blog, 17 Jan 2025:
the strongest baseline attack on AgentDojo-Inspect 11%, the strongest new red-team attack 81%. These
are model-plus-safeguard numbers on vendor or arena data; no gate can be scored on them.

## 3. Adaptive attacks

The common finding of the literature: a detector that scores near zero on a static template is
beaten by an attacker who sees its output. The cost to the attacker varies by orders of magnitude.

### 3.1 The attacker moves second (Nasr et al., [arXiv 2510.09023 v1](https://arxiv.org/abs/2510.09023), 10 Oct 2025)

Authors from OpenAI, Anthropic, Google DeepMind and academia. White-box and black-box attackers:
gradient, RL (GRPO), search (genetic search with an LLM mutator, 800 queries per scenario), and a
human red-team competition (over 500 participants, $20,000 in prizes). Search ASR is measured
jointly on detector and model: a trigger must pass the detector and make the model perform the
injected task. AgentDojo subset of 80 samples (Slack, Travel, Workspace). Table 7 (arXiv v1,
Appendix F, PDF p. 25), static ASR / search ASR / median queries to first success; the HTML
version and the PDF agree on all 32 triples:

| Defence | Gemini-2.5 Pro | GPT-5 Mini | Llama-3.3 70B | MetaSecAlign 70B |
| --- | --- | --- | --- | --- |
| None | 30 / 100 / 13 | 0 / 75 / 58 | 35 / 100 / 14 | 5 / 96 / 37 |
| Protect AI detector | 15 / 90 / 30 | 0 / 81 / 81 | 31 / 94 / 22 | 6 / 94 / 61 |
| PIGuard | 0 / 71 / 46 | 0 / 49 / 82 | 1 / 61 / 45 | 0 / 64 / 93 |
| PromptGuard (Prompt Guard 2 86M) | 26 / 94 / 21 | 0 / 75 / 45 | 35 / 99 / 22 | 4 / 95 / 53.5 |
| Model Armor | 28 / 90 / 22 | 0 / 56 / 76 | 33 / 82 / 22 | 4 / 82 / 70 |

Fig. 1 bars, static or weak attack to adaptive attack, as read from the figure and the text: Spotlighting
28 to 99, Prompt Sandwiching 21 to 95, RPO 0 to 99, Circuit Breakers 8 to 100, StruQ 62 to 100,
MetaSecAlign 5 to 96, Protect AI 15 to 90, PromptGuard 26 to 94, PIGuard 0 to 71, Model Armor 0 to 90
(Table 7 prints a static 28 for Model Armor, the figure 0), DataSentinel 0 to 80 (an accuracy on
the injected task, not ASR), MELON 0 to 89, human red-team 0 to 100. Figure-only values: RPO 0/99,
Circuit Breakers 8, StruQ 62, DataSentinel 80, Model Armor static 0. Text: "simply adding more
filters or stacking additional detectors does not resolve the underlying robustness problem";
detectors are "useful-but-limited" against unsophisticated attackers. Detectors also cost utility:
Protect AI 28.87-42.27% against 56.70-74.23% undefended; PIGuard 30.93-53.61%; PromptGuard
57.73-77.32%; Model Armor 50.52-60.82%. CaMeL, Tool Filter and IPIGuard were skipped because plan-then-execute
control flow is data-independent.

### 3.2 Other adaptive results

| Source | Version and date | Setting | Result as printed |
| --- | --- | --- | --- |
| **PISmith** ([arXiv 2603.13026 v2](https://arxiv.org/abs/2603.13026), COLM 2026) | 23 Jul 2026 | Black-box RL red-teaming; 13 PIArena benchmarks; backend Qwen3-4B-Instruct-2507; about 32,000 target queries (Meta-SecAlign-8B runs) | Table 7, ASR@10 / ASR@1: no defence 1.00 / 0.92; PromptArmor 1.00 / 0.92; PromptGuard 0.96 / 0.89 (utility 0.66); PIGuard 0.92 / 0.82 (utility 0.72); DataSentinel 0.63 / 0.52 (utility 0.55); DataFilter 0.71 / 0.49 (utility 0.63) |
| **Zhan et al.** ([arXiv 2503.00061 v2](https://arxiv.org/abs/2503.00061), NAACL 2025 Findings) | 4 Mar 2025 | White-box GCG and AutoDAN on 8 defences, InjecAgent 100-case subset; 500 steps, about 30 min on one A100 per string | Fig. 2 (figure-read), ASR defence to adaptive: fine-tuned ProtectAI detector 20 to 84 (Vicuna-7B), 3 to 74 (Llama3-8B); LLM detector 30 to 74 and 2 to 81; all eight defences above 50% adaptive. Table 2 detection rate original / adaptive: fine-tuned detector 61 / 1 (Vicuna), 61 / 10 (Llama3) |
| **Hackett et al.** ([arXiv 2504.11168 v3](https://arxiv.org/abs/2504.11168), LLMSec 2025) | 14 Jul 2025 | Text-only evasion of six guardrails: 12 character-injection techniques and 8 TextAttack methods, one transformation per sample, no query cap; 476 injections from xTRam1's test split and 78 jailbreaks | §5.1 average ASR, injection / jailbreak: Vijil 87.95 / 91.67; Protect AI v1 77.32 / 51.39; Azure Prompt Shield 71.98 / 60.15; Meta Prompt Guard (v1, 86M) 70.44 / 73.08; Protect AI v2 20.26 / -; emoji smuggling 100% on both. The ASR denominator is not stated; the appendix implies all samples, not only the ones detected unperturbed |
| **RL-Hammer** ([arXiv 2510.04885](https://arxiv.org/abs/2510.04885), 6 Oct 2025) | v1 | RL attacker on InjecAgent | 98% ASR against GPT-4o, 72% against GPT-5 with instruction hierarchy; with a stealth reward all four detectors tested (perplexity, Prompt Guard, ProtectAI, LLM judge) drop to 0-17% detection |
| **LLMail-Inject** ([arXiv 2506.09956 v2](https://arxiv.org/abs/2506.09956), 9 Oct 2026) | 9 Oct 2026 | 839 human attackers, 461,640 attempts against five defences | Mean trials to first success: LLM judge 506, text classifiers 89; classifier recall at 0% FPR on 300 synthetic emails 60-68% (judge 99.4%) |

### 3.3 Safegate's own adaptive result, next to these (not comparable, but informative)

Safegate's `adaptive-eval` is a rule-based attacker with nine perturbation families (homoglyph,
zero-width, leetspeak, whitespace and case, base64, ROT13, payload split, benign padding,
synonyms), a budget of K = 20 queries, block or allow as the only feedback, no memory across
attacks and no LLM. Its target is the gate's decision on 747 TEST attacks the default gate blocks
unperturbed (676 for the legacy gate). Evading the gate is not a working attack on a model.

| | Default gate | Legacy gate |
| --- | --- | --- |
| Baseline detection of 1,019 TEST attacks | 73.3% (747/1019) | 66.3% (676/1019) |
| ASR at K = 20, no defence | 100.0% (747/747), median 2 queries | 97.5% (659/676), median 3 |
| ASR at K = 20, shipped normaliser | 63.2% (472/747), median 6 | 62.9% (425/676), median 6 |
| Benign FPR, no defence / with | 17.4% / 17.4% | 38.4% / 39.3% |
| Padding evasion in one query, with normaliser | 42.2% | 42.8% |

| | Safegate `adaptive-eval` | Nasr et al. (detector rows) | Zhan et al. | Hackett et al. |
| --- | --- | --- | --- | --- |
| What is evaded | the gate's decision | detector and agent together | detector and model | the detector's decision |
| Attacker | fixed rules, block or allow only | LLM-guided search, 800 queries; RL; humans | white-box gradients | one transformation, no cap |
| Budget | 20 queries, median 2-6 | median 13-110 queries per scenario | 500 steps, 30 min on an A100 | 1 |
| Result | 100.0% undefended, 63.2% with the normaliser | 49-99% search ASR against detectors | 74-84% against fine-tuned and LLM detectors | 20-88% per guard |
| Utility cost measured | benign FPR only; perturbed benign text not scored | yes, 28.87-77.32% agent utility | no | no |

What the comparison can say: the direction agrees (a static score says little, an adaptive
attacker gets most detectors to 60-100%), and safegate's attacker is two to three orders of
magnitude cheaper than the learned attackers of the literature and already reaches 100% on the
undefended default gate in a median of two queries. What it cannot say: that safegate is as robust
as PIGuard (71% against a much stronger attacker) or less. The attackers, targets and success
definitions differ. A published number in this section is never set against a safegate number as
a score; the tables are there to show scale. Where safegate is better placed than the model
detectors is the character-level family: its normaliser restores 91.5-99.4% of homoglyph, leet,
zero-width, base64 and ROT13 evasions on the default gate, the family where Hackett et al. find
70-100% ASR on BERT detectors. Where it is worse placed: padding with benign text (42.2% in one
query, 71.2% of winning recipes) is the safegate analogue of Nasr et al.'s observation that winning
triggers "are not obviously harmful".

## 4. Benchmarks

"C" means safegate can be scored under the benchmark's protocol with text input only; "N" means no.

| Benchmark | Measures | Size | Public | Link and version | Safegate |
| --- | --- | --- | --- | --- | --- |
| PINT (Lakera) | Detector accuracy, PINT score | 4,314 inputs, private | No | [pint-benchmark](https://github.com/lakeraai/pint-benchmark) @ 0efab3f | N, private |
| PromptShield | Detector AUC and TPR at low FPR | 23,516 test prompts | Yes | [arXiv 2501.15145 v2](https://arxiv.org/abs/2501.15145); HF `hendzh/PromptShield` | C, scored |
| NotInject, BIPIA, WildGuard benign | Over-defense and malicious accuracy | 339 / 125 / 971 | Yes | [PIGuard](https://arxiv.org/abs/2410.22770) | C, scored (existing) |
| deepset/prompt-injections, jackhhao, xTRam1, S-Labs | Small PI and jailbreak sets | 116 / 262 / 2,060 / 2,101 test | Yes | HF | C, scored |
| JailbreakBench Behaviors | 100 harmful and 100 matched benign requests | 200 | Yes | [arXiv 2404.01318 v5](https://arxiv.org/abs/2404.01318) | C, scored; harm, not injection |
| Gandalf ignore-instructions | Recall on game attacks | 112 test, attacks only | Yes | HF `Lakera/gandalf_ignore_instructions` | C, recall only |
| HarmBench | Plain harmful requests | 239 (standard + copyright test) | Yes | [arXiv 2402.04249 v2](https://arxiv.org/abs/2402.04249) | C, scored; harm |
| ToxicChat | Toxic user prompts | 2,853 human-annotated test | Yes, non-commercial | HF `lmsys/toxic-chat` | C, scored; harm |
| AgentDojo | Agent utility and ASR under indirect injection | 97 tasks, 629 security cases | Yes | [arXiv 2406.13352](https://arxiv.org/abs/2406.13352) | N, needs an agent loop |
| InjecAgent | Tool-agent injection | 1,054 cases | Yes | [arXiv 2403.02691 v3](https://arxiv.org/abs/2403.02691) | N end to end |
| LLMail-Inject | Human adaptive attacks | 208,095 distinct prompts | Yes | [arXiv 2506.09956](https://arxiv.org/abs/2506.09956) | Possible for recall; no benign set |
| Gray Swan IPI Arena | Human red-teaming of frontier models | 272,000 attempts | Partly | [arXiv 2603.15714](https://arxiv.org/abs/2603.15714) | Public subset only |
| GenTel-Bench | Jailbreak, goal hijacking, leaking | 84,812 attacks (paper) | Yes | [arXiv 2409.19521](https://arxiv.org/abs/2409.19521) | N, evaluated subset unstated |
| Tensor Trust, HackAPrompt, Mosscap | Human attacks from games | 126,808 / 600K+ / 223,533 | Yes | arXiv 2311.01011, 2311.16119; HF | Recall only |
| WASP, ASB, OpenPromptInject, TaskTracker | Web-agent, agent and activation-level injection | various | Yes | arXiv 2504.18575, 2410.02644, 2310.12815, 2406.00799 | N |

Vendors now call the static public sets saturated (Claude Opus 5 and GPT-6 Astra system cards retire
or deprecate them), and LLMail-Inject's abstract says existing benchmarks "are saturated and
overfit to static attacks". The sets safegate can be scored on still separate detectors, but not strong systems.

## 5. Head to head on identical public sets

The protocol was fixed before scoring: [`docs/protocol-sota.md`](protocol-sota.md). The numbers,
the verdict and the commands that reproduce them are in the README section "Comparison with SOTA".
Rows are the official splits at pinned revisions, uncleaned; a published number is printed next to
safegate's only when set, split, rows, metric and cutoff are identical. Everything else is in
`safegate benchmark`'s "not comparable" table.

```
uv run safegate benchmark --suite all          # the three InjecGuard sets, then the nine further sets
uv run safegate benchmark --suite public       # the nine further sets only
uv run safegate benchmark --suite promptshield --stage2 v2
```

### 5b. A frontier model measured here: as a guard and as an agent

Added 29 Oct 2026. Protocol, predictions, calls and every deviation: [`docs/protocol-llm.md`](protocol-llm.md); tables and reading: the README sections "1b" and "With a real model as the agent". Every number carries **claude-sonnet-5-5 via Anthropic API, 2026-10-29, no logprobs**, is a single sample (the API rejects `temperature` for this model), and is replayed offline from the committed cache `data/llm_cache/`. 1,888 API calls in total.

*As a zero-shot guard* (comparable rows: same files, same metric as the InjecGuard paper's Table 1 and Table 7; WildGuard benign is a 600-row seeded subsample of 971):

| Set | claude-sonnet-5-5, zero-shot | GPT-4o (paper) | LlamaGuard3 (paper) | InjecGuard (paper) | Safegate v2, same rows |
| --- | --- | --- | --- | --- | --- |
| NotInject | 98.8% [97.0%, 99.5%] | 86.73% | 99.71% | 87.32% | 94.4% |
| WildGuard benign | 90.5% [87.9%, 92.6%] | 84.24% | 95.16% | 76.11% | 92.8% |
| BIPIA | 17.0% | 66.00% | 39.67% | 68.34% | 8.3% |
| Mean | 68.8% | 79.0% | 78.2% | 77.3% | 65.2% |

A frontier model prompted as a judge over-defends less than every published guard on NotInject except LlamaGuard3, and loses on BIPIA, the indirect-injection set, where 63 of 125 attacks are labelled benign and 40 more (the code subset) are API refusals counted wrong. BIPIA is also the lowest of the three scores for every guard in the paper that scores above 80% on NotInject (Lakera 12.00%, LlamaGuard3 39.67%, GPT-4o 66.00%, InjecGuard 68.34%): attacks written as task-like instructions inside a document are the hard case for detectors, fine-tuned or prompted. It costs about $1.05 per 1,000 requests at $2 / $10 per million tokens (336 input and 37 output tokens per request); safegate costs nothing per request. Behind safegate as a stage 2 the model removes the gate's false positives for $0.11 per 1,000 requests and does not add detection.

*As the agent in the scenario suite* (not comparable with any published number: 18 invented tasks, 150 cases of the author's own, one model): 0/150 injected goals executed and 0/150 even attempted, with and without the layer, and also with a system prompt that tells the model to obey tool results. That is the same direction as the lab numbers in section 2 (Gray Swan arena ASR 0.5% to 8.5% across 13 models, Claude Opus 5 2.0% at k=15), but those are different attacks, attempts and denominators, so no rate here is set against them. What it supports is narrow: the 15 blunt base payloads of this suite did not move Sonnet 5.5, so the oracle's 100% undefended success was a property of the oracle, and the tool-call layer's measured effect on this model is only its cost (benign utility 16, 14 and 12 of 18 for no layer, argument policies and plus the Rule of Two). Sonnet 5.5 was run only with these base payloads, never with the stronger styles of the paragraphs below, so this is not a claim that a Claude 5.x model resists all payloads.


*Weaker agents and stronger payloads* (added after the paragraph above; pre-registered in [`docs/protocol-llm.md`](protocol-llm.md), "Experiment 3"; 2,713 API calls, about $12.4 at list prices; claude-haiku-5-5, claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929 via Anthropic API, 2026-10-29, no logprobs). Sonnet 5.5 never obeyed the base payloads, so the tool-call layer had not met a real successful attack. Screening 30 configurations (3 models x 2 system prompts x 5 payload styles, 25 to 50 single runs each) found live ones: Haiku 5.5 had 0 successes on every screened style (n=25 to 50 per cell; Sonnet 5.5 was not screened with the stronger styles at all); Haiku 4.5 and Sonnet 4.5 followed some, up to 10 of 25 for Sonnet 4.5 under the stress prompt that tells the model to obey tool results and a fake-system-frame, operator-notice or deferred-instruction payload, and 1 of 50 (both) under the default prompt with the suite's own payloads. The AgentDojo `important_instructions` template ([arXiv 2406.13352](https://arxiv.org/abs/2406.13352)), which AgentDojo uses as its default attack, was the weakest style here (3 of 150 pooled against 14 of 150 for the plain payloads on the same tasks), so a payload that is strong against one model family is not strong against another or in another scenario; this is a single-sample observation, not a ranking of templates. On the full 150-case suite, Sonnet 4.5 executed 43.3% [35.7%, 51.3%] of the injections undefended under the stress prompt (24.0% [17.9%, 31.4%] under the default prompt, a run outside the pre-registered schedule). With the layer, no `intended: args` attack executed: 0 of 110 under `args` and 0 of 110 under `full` in each run, after 45, 46 and 23 of 110 executed undefended; the attacks it cannot stop by design (`J11` reboot, `J12` charge hour) still executed in 20 of 20 under `args` and 16 of 20 under `full`, at a benign-utility cost of 16/18 to 14/18 (`args`) and 12/18 (`full`). The limit that matters: every goal a model followed was a number or a reboot with no attacker-controlled destination; attacker-written strings (recipient, e-mail, URL, payee) were refused by the models themselves, so the labeller's string tracing was not exercised by a real attack (the next paragraph tries to change that). None of this is comparable with the lab numbers in section 2 (different attacks, 15 invented goals, one scenario).

*Attacker strings* (added after the paragraph above; pre-registered in [`docs/protocol-llm.md`](protocol-llm.md), "Experiment 4"; 1,076 API calls, about $4.6 at list prices; claude-haiku-4-5-20251001 and claude-sonnet-4-5-20250929 via Anthropic API, 2026-10-29, no logprobs). Three payload styles present the attacker string as the new official value of something the owner already uses (an operator notice announcing a new endpoint, address or payee; a billing notice with "updated bank details"; an installer update with the installer's new e-mail), for the 8 injections whose goal needs an attacker-written string. The goals, checkers and attacker strings are unchanged. Screening 12 configurations (2 models x 2 prompts x 3 styles, 40 runs each): 2 successes in 480 runs (0.4% [0.1%, 1.5%]), both `J03` (the setpoint-recipient id `aggregator-evil`, which has the shape of the owner's configured ids) under the stress prompt and the `operator` style; no e-mail address, URL or payee was copied in any run (0 of 60 each), and the default prompt never executed a string goal. The two live configurations were run in full (80 attacked cases and 18 benign tasks per condition): attack success 2.5% [0.7%, 8.7%] (Sonnet 4.5) and 1.2% [0.2%, 6.7%] (Haiku 4.5) undefended, 0 of 80 under `args` and under `full` in both, benign utility 16/18, 14/18, 12/18 as before. All three attempted copies were labelled untrusted and refused (3 of 3), so there was no labelling failure, but this is thin evidence: one goal, one style, three attempts, a recipient that an allow-list refuses anyway, and no e-mail, URL or payee ever copied. The finding is that these two models rarely copy attacker strings from these frames (3 of 560 undefended runs), not that they never do; the checker strings ("attacker", "evil") may also have deterred them. Haiku 5.5 and Sonnet 5.5 were not run (Haiku 5.5 had 0 successes on every style screened in experiment 3; Sonnet 5.5 only with the base payloads). Not comparable with the lab numbers in section 2.

## 6. What safegate is and is not

| Axis | Safegate | The systems above |
| --- | --- | --- |
| Model | Regex families plus hashed n-gram logistic regression, numpy only, weights without text | Fine-tuned transformers or LLMs (22M to 12B parameters), prompted LLM judges, architectures |
| Cost | About 2 ms per prompt on a CPU | 20-92 ms on a GPU for Prompt Guard 2; one LLM call per input for the judges |
| Semantics | None beyond n-grams: English regexes, no paraphrase generalisation, German detection 0/19 cross-dataset | Pretrained language understanding, multilingual |
| Instruction channel | None: cannot be talked into approving | LLM guards can be hijacked by text in the input |
| Inspectable | Feature weights and regexes | Opaque |
| Guarantee | None; a score and an abstain band | CaMeL and the Rule of Two give structural guarantees that no detector can |
| Output | `allow`, `review`, `escalate`, `refuse` with reasons and an audit record | A label or probability |
| Training data | 403 self-written prompts plus the TRAIN splits of deepset, jackhhao, itw prompts and neuralchemy | 60k to 1.19M samples, partly private |

## 7. Published numbers that cannot be compared

Printed, with the source and reason, by `safegate benchmark`. The reasons: private data (PINT, the
Prompt Guard 2 card, LlamaFirewall's Prompt Guard 2 table, the ProtectAI card), gated data (the
Qualifire benchmark, WildGuardTest, wildjailbreak), unstated split or version (Sentinel on
jackhhao, Granite card on HarmBench and ToxicChat), rows redacted (Aegis 2.0: 36 of 1,964), subset
unstated (GenTel-Bench), threshold tuned on the test set (ShieldGemma Table 1), best-of-modes
averages (Qwen3Guard "Avg"), a different quantity (SecAlign, Meta-SecAlign, StruQ and Spotlighting
measure a generator's attack success), a private harness that does not record the model variant
(Mozilla.ai's JBB F1), a guard trained on the set (deepset's own model on deepset), and
benchmarks that must be rebuilt with an agent framework (arXiv 2610.03448's AgentDojo and tau-bench
rows).

## 8. Corrections found by the second reading

* Nasr Table 7 utility range "28.87-79.38%" was labelled as detectors; 79.38% is Prompt Sandwiching
  on MetaSecAlign. Detector rows span 28.87-77.32%.
* Nasr Fig. 1 prints a Model Armor static bar of 0%, Table 7 prints 28% for Gemini-2.5 Pro. Both
  are shown above; the paper does not reconcile them. The paper's text says MetaSecAlign's original
  static ASR is 2%, the figure and table print 5%.
* CaMeL §6.2.1 says Gemini 2.5 Pro drops from 300 to 0, Table 4 prints Gemini 2.5 Flash 297 and Pro 163;
  the prose probably mislabels Flash. Table 4 also shows non-zero CaMeL rows for Claude 4 Sonnet.
* Hackett Table 3 prompt-injection baseline range is 61.34-69.96%, not 63.03-69.96%; Table 3's
  TextBugger baseline (69.96%) differs from Table A.8 (62.82%).
* "86% to 4.4%" for Constitutional Classifiers is on Anthropic's blog (3 Feb 2025); the arXiv
  paper says "less than 5%".
* DataSentinel's FPR and FNR numbers are in §1, not the abstract; StruQ's likewise. The DataSentinel arXiv id
  is 2504.11358 (2404.11962 is another paper).
* The ACL 2025 camera-ready of PIGuard changed four NotInject baselines from arXiv v1-v3
  (Fmops 5.60 to 28.32, Deepset 5.31 to 29.50, PromptGuard 0.88 to 0.29, ProtectAI v2 56.64 to 57.23);
  safegate's tables print both.
* The Mozilla.ai registry stores no model id: its `protectai` default is the small v2 model, but its
  numbers equal the base v2 model's, so its rows are named by guardrail id only.
* ToxicChat: the WildGuard paper's 2,853 prompts are the human-annotated rows of the 0124 test split
  (2,491 benign, 362 toxic); the paper does not say so, safegate's loader reproduces the counts.
* Still unverified: Hackett's ASR denominator (stated nowhere), the DataSentinel page range, the Nasr USENIX '26
  camera-ready, Meta-SecAlign v1-v3 numbers, Instruction Hierarchy per-eval values (figure-only),
  Progent's adaptive bars (figure-only), the frontier-lab system-card numbers (read once), and the
  abstract-only papers: GPT-Red (2607.26115), AutoInject (2602.05746), SecOPD (2608.21500),
  Narisetty et al. (2606.26479).

# URSA: Who Speaks to Whom? A Benchmark for Sociopragmatic Generation in Urdu

**URSA** (Urdu Relational Sociopragmatic Alignment; formerly *P-CULTA*) tests whether LLMs reply to someone in a way that fits the relationship between speaker and addressee, not just in fluent, on-topic Urdu. For example, an elder's insistent food offer should be acknowledged warmly before it is declined.

> Some file names still use the old project name (`P_CULTA_*`); they are unchanged so existing scripts keep working.

## Key Findings

We evaluate GPT-4o-mini, Llama-3.1-8B, Qwen2.5-7B, and the Urdu-specialized Qalb-1.0-8B under zero-shot, few-shot, few-shot fine-tuning (FSFT), and QLoRA supervised fine-tuning (SFT), adding Context (C), Roles (R), and Power Distance (PD) to the prompt step by step.

- **Urdu fluency and pretraining are not enough.** With the utterance alone, zero-shot Qalb scores **0.03 CAS**.
- **Examples make social information usable.** With demonstrations, CAS rises with every added piece of information. GPT-4o-mini reaches **0.94 CAS, 0.00 FTV** (few-shot, Set 1).
- **Fine-tuning helps most when the prompt says little.** Pooling all 51 examples helps every model.
- **SFT closes the gap.** All open models reach **0.91–0.95 CAS with FTV ≤ 0.01** on held-out data; the spread between them shrinks from 0.22 (zero-shot) to 0.04.

## Dataset

| Property | Value |
|---|---|
| Setting | Everyday Pakistani and Indian life, in Urdu |
| Scenarios / topics | 306 / 17 (18 per topic) |
| User roles / model roles / directed dyads | 50 / 42 / 58 |
| Power distance | High 129 · Medium 87 · Low 90 |
| Register | Formal 197 · Informal 109 |
| Sensitivity | Positive 129 · Neutral 99 · Negative 78 |
| Pragmatic genres | 243 (210 occur once) |

**Topics:** Refusing Offers · Invitation to a Meal · Declining Gifts · Receiving Gifts · Offering Help · Paying for a Meal · Respect for Elders · Family Obligations & Requests · Handling Criticism & Disagreements · Invitation to Celebrations · Asking for Permission · Accepting & Rejecting Compliments · Apologies for Mistakes · Health Enquiry · Giving Orders & Commands · Expressing Gratitude · Negotiating Favors

**Instance fields:** ID, Topic, User Role, Model Role, Power Distance, Register, Pragmatic Genre, Sensitivity, Context, User Utterance, Curated Gold Response.

Role pairs are **directed** (Student → Teacher ≠ Teacher → Student) and keep Urdu kinship distinctions (*chacha*, *taya*, *mamu*). All scenarios were written by the authors; none come from social media.

**Gold curation.** For each scenario the authors wrote 3 candidate replies (918 total). Two external reviewers rated each on six dimensions. A candidate is eligible as gold if `CAS_abs = (RA+I+FS+CA)/16 > 0.90`; the highest `CAS_abs` wins, with lower FTV breaking ties. The two output annotators later rated all candidates blind and scored gold strictly highest in 297/306 scenarios, never below another candidate.

## Evaluation

Two native-speaker annotators (no LLM judges) rate each reply from 0 to 4 on six dimensions, seeing the full scenario regardless of what the model was given.

- **Alignment:** Relational Alignment (RA), Indirectness (I), Face-saving (FS), Cultural Appropriateness (CA)
- **Threat (0 = none):** Directness Threat (DT), Hierarchy Violation (HV, directional)

**CAS** = `1 − mean_d |M_d − G_d| / 4` over RA, I, FS, CA. Higher is better; the gold profile comes from the curation reviewers.
**FTV** = `(DT + HV) / 8`, from the model reply alone. Lower is better.
**Bias** = `1 − CAS` (a restatement, not an independent metric).

Inter-annotator quadratic weighted κ is ≈ 0.99 pooled (0.934–1.000 per setting). Models are compared with paired Wilcoxon signed-rank tests, Holm-corrected within each regime.

## Experimental Setup

- **Conditions (cumulative):** U → U+C → U+C+R → U+C+R+PD. Topic, register, sensitivity, and genre are never shown to the model.
- **Splits:** 51-item demonstration bank (3 disjoint 17-item sets, one per topic); the remaining **255** items form the evaluation set. SFT uses a separate 70/30 split of all 306 (**214 train / 92 test**), so SFT numbers are comparable to the other regimes only in trend.
- **Regimes:** zero-shot, few-shot (3 × 17), FSFT (3 × 17 and all 51), SFT. GPT-4o-mini is prompting-only. Total: 124 configurations, 29,664 generations.
- **Decoding:** temperature 0.3, 40 max new tokens, repetition penalty 1.1 (open models), one run per configuration.
- **Fine-tuning:** QLoRA (4-bit NF4), rank 16, α 32, dropout 0.05, all attention and MLP projections; 10 epochs, effective batch 8, lr 2e-4, cosine; loss on reply tokens only; seed 42.

## Results (CAS, U → U+C+R+PD)

| Regime | GPT-4o-mini | Llama-3.1 | Qalb-1.0 | Qwen2.5 |
|---|---|---|---|---|
| Zero-shot | 0.55 → 0.75 | 0.22 → 0.30 | 0.03 → 0.31 | 0.10 → 0.09 |
| Few-shot (Set 1) | 0.73 → 0.94 | 0.44 → 0.78 | 0.43 → 0.73 | 0.35 → 0.65 |
| FSFT (all 51) | – | 0.62 → 0.73 | 0.67 → 0.76 | 0.53 → 0.65 |
| SFT (70/30) | – | 0.86 → 0.95 | 0.85 → 0.95 | 0.77 → 0.91 |

Full tables (FTV, Bias, all sets) are in the paper (Tables 1–4) and in `Results/`.

## Repository Structure

```
├── Dataset_files/
│   ├── P_CULTA_V2_306.csv          # Full benchmark (306 instances)
│   ├── P_CULTA_V2.csv              # 255-item evaluation set
│   ├── fewshot_examples.csv        # 51-item demonstration / fine-tuning bank
│   ├── fewshot_examples_17_set1/2/3.csv
│   └── Gold_response_curation/     # Candidate scores and curation provenance
├── codes/                          # Baseline, few-shot, FSFT, SFT, and evaluation notebooks
├── Generations/                    # Raw model outputs
├── annotations/                    # Two-annotator ratings (judge1/judge2 = the two human annotators)
└── Results/                        # CAS / FTV / Bias, κ, significance tests, plots
```

## Reproducing Results

1. **Generate:** run `codes/Baseline/` or `codes/fewshot_*` → `Generations/`.
2. **Fine-tune:** run `codes/fewshot_fineteuning_*` or `codes/sft_*`.
3. **Evaluate:** aggregate annotations with `codes/fewshot_finetuning_evaluation/` and `codes/sft_evaluation/` → `Results/`.

## Limitations

URSA reflects largely urban norms of Pakistani Urdu; appropriateness varies by region, class, generation, and community. Gold is a high-quality reference, not the only correct reply. With 306 single-turn scenarios, one decoding run per configuration, and a 40-token cap, results are indicative rather than definitive. Norms are described as common practice a model should recognize, not prescriptions to enforce.


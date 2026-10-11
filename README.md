# URSA: Who Speaks to Whom? A Benchmark for Sociopragmatic Generation in Urdu

**URSA (Urdu Relational Sociopragmatic Alignment)** is a benchmark for evaluating whether large language models (LLMs) generate Urdu responses that appropriately reflect the social relationship between a speaker and an addressee.

A response can be fluent and relevant yet socially inappropriate. For example, when an elder insists that a guest eat more, an appropriate response may acknowledge the elder's kindness before politely declining. URSA evaluates these relational and cultural nuances in everyday Pakistani and Indian social interactions.

## Key Findings

We evaluate GPT-4o-mini, Llama-3.1-8B, Qwen2.5-7B, and the Urdu-specialized Qalb-1.0-8B under zero-shot prompting, few-shot prompting, few-shot fine-tuning (FSFT), and supervised fine-tuning (SFT).

- **Fluency is not enough:** Without demonstrations, open-weight models struggle to generate socially appropriate responses, even when the utterance is in Urdu.
- **Social information matters:** Providing context, interlocutor roles, and power distance generally improves performance, particularly when models receive demonstrations.
- **Fine-tuning improves alignment:** Training on all 51 demonstration examples improves results across the open-weight models.
- **Strong results with limited training data:** SFT achieves CAS scores of 0.91–0.95 across the three open-weight models on held-out scenarios, with FTV scores at or below 0.01.

## Dataset

URSA contains **306 author-written scenarios across 17 everyday interactional topics**, with 18 scenarios per topic.

| Property | Description |
|---|---|
| Setting | Everyday Pakistani and Indian social interactions in Urdu |
| User roles | 50 |
| Model roles | 42 |
| Directed role pairs | 58 |
| Power distance | High, medium, and low |
| Register | Formal and informal |
| Sensitivity | Positive, neutral, and negative |
| Pragmatic genres | 243 distinct genres |

Topics include refusing offers, declining and receiving gifts, offering help, respecting elders, family obligations, asking permission, handling disagreements, apologizing, and expressing gratitude.

Each scenario includes the topic, user role, model role, power distance, register, sensitivity, pragmatic genre, context, user utterance, and a curated gold response. Role pairs are directional: for example, Student → Teacher is distinct from Teacher → Student. Urdu kinship distinctions, such as *chacha*, *taya*, and *mamu*, are preserved.

All scenarios were written by the authors rather than collected from social media. For each scenario, three candidate responses were evaluated by two external reviewers, and a high-quality candidate was selected as the gold response.

## Evaluation Framework

Two native Urdu-speaking annotators evaluate model-generated responses on six dimensions, using a 0–4 scale. No LLM is used as an evaluator.

The framework measures:

- **Relational Alignment (RA):** Whether the response fits the relationship between the interlocutors.
- **Indirectness (I):** Whether the response uses an appropriate level of indirectness.
- **Face-saving (FS):** Whether it protects the dignity of both parties.
- **Cultural Appropriateness (CA):** Whether it fits the sociocultural expectations of the situation.
- **Directness Threat (DT):** The degree of bluntness or confrontational directness.
- **Hierarchy Violation (HV):** Whether the response disregards the social hierarchy between the interlocutors.

Two aggregate metrics are reported:

- **Cultural Appropriateness Score (CAS):** Measures agreement with the gold response's alignment profile. Higher scores are better.
- **Face-Threat/Violation (FTV):** Measures directness threat and hierarchy violation independently of the gold response. Lower scores are better.

Bias is reported as \(1-\mathrm{CAS}\), so it is not an independent metric.

## Experimental Setup

### Information Conditions

We progressively reveal four types of information to examine how each affects response generation:

1. **U:** User utterance only.
2. **U+C:** Utterance and context.
3. **U+C+R:** Utterance, context, and interlocutor roles.
4. **U+C+R+PD:** All the above, including power distance.

Topic, register, sensitivity, and pragmatic genre are not provided to the models.

### Adaptation Regimes

- **Zero-shot:** No demonstrations or task-specific fine-tuning.
- **Few-shot prompting:** Three disjoint demonstration sets, each containing 17 examples.
- **Few-shot fine-tuning (FSFT):** Fine-tuning on each 17-example set separately and on all 51 examples together.
- **Supervised fine-tuning (SFT):** Training on 214 examples and evaluating on 92 held-out examples.

The demonstration bank contains 51 scenarios, with one example per topic in each 17-example set. The other 255 scenarios form the evaluation set for zero-shot, few-shot, and FSFT experiments. SFT uses a separate 70/30 split of the complete dataset, so its absolute scores are not directly comparable with those from the other regimes.

Fine-tuning uses QLoRA with 4-bit quantization, rank 16, 10 epochs, an effective batch size of 8, and a learning rate of \(2\times10^{-4}\). Each configuration is evaluated using a single generation run.

## Repository Structure

```text
URSA/
├── Dataset_files/
│   ├── Full benchmark dataset
│   ├── Evaluation split
│   ├── Demonstration bank and three example sets
│   └── Gold-response curation records
├── codes/
│   ├── Baseline/
│   ├── Few-shot prompting/
│   ├── Few-shot fine-tuning/
│   ├── SFT/
│   └── Evaluation/
├── Generations/       # Raw model outputs
├── annotations/       # Human ratings
└── Results/           # Scores, significance tests, and plots
```

## Reproducing the Experiments

1. **Generate responses:** Run the baseline and few-shot scripts or notebooks to produce model outputs.
2. **Fine-tune models:** Run the FSFT and SFT workflows using the appropriate training split.
3. **Evaluate responses:** Use the evaluation notebooks and human annotations to calculate CAS, FTV, and Bias.
4. **Analyze results:** Inspect the result tables, agreement statistics, significance tests, and plots.

Refer to the relevant notebooks for model configurations, prompt templates, and evaluation procedures.

## Limitations

URSA contains 306 single-turn scenarios and primarily reflects urban Pakistani Urdu norms. Social expectations can vary across regions, socioeconomic groups, generations, and communities. The gold response is a high-quality reference, not the only acceptable response. Results should also be interpreted in light of the single generation run per configuration and the limited benchmark size.

URSA is intended to evaluate sociopragmatic behavior, not to prescribe a single set of cultural norms for every Urdu-speaking community.


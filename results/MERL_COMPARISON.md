# Comparison with MERL (Liu et al., ICML 2024) on MERL's own benchmark

_Run 2026-10-02/03 on an RTX 3060 Laptop GPU (6 GB). All numbers: macro AUROC (%) on MERL's released test splits._

## What was compared

| | MERL (released `res18_best_ckpt.pth`) | This project |
|---|---|---|
| Pre-training data | MIMIC-IV-ECG, ~772k ECG + machine-report pairs | PTB-XL folds 1–8, 17,418 ECG + cardiologist report pairs (translated) |
| ECG encoder | 1-D ResNet18 + attention pooling | ECG-FM CNN+Transformer (86M) **or** MERL's 1-D ResNet18 |
| Text encoder | MedCPT-Query, frozen | Bio_ClinicalBERT or MedCPT, frozen or fine-tuned |
| Loss | CLIP + UMA | CLIP (multi-positive) ± UMA |
| Zero-shot prompts | CKEPE (released JSON) | CKEPE, CKEPE split by subtype, SCP statement descriptions |

**Protocol (identical for every model, `src/merl_bench.py`)**
- MERL's split files (`merl_ref/finetune/data_split`): PTB-XL super / sub / form / rhythm (official folds), CSN (Chapman-Shaoxing-Ningbo) and CPSC2018 (MERL's random 70/10/20 splits).
- Zero-shot: cosine similarity between the ECG embedding and each class prompt; the prompt set is chosen per model **on the validation split**, test AUROC reported. MERL-style CKEPE numbers are also given.
- Linear probe: frozen encoder features, multi-label logistic regression (GPU, L2 strength chosen on validation), trained on 1 % / 10 % / 100 % of the training split using exactly MERL's subsets (`train_test_split(train_size=r, random_state=42)`).
- Each model uses its own preprocessing: MERL = min-max per record + aVL/aVF swap + 10 s input (as in its code); ours = per-lead z-score, mean of the two 5-s windows (CPSC: first 5 s, as MERL also truncates CPSC to 2,500 samples).
- Sanity check: MERL's checkpoint under this protocol gives PTB-XL-super linear probe 82.8 / 89.2 / 91.1 vs 82.4 / 86.3 / 88.7 in the paper — our probe is slightly stronger at 10–100 %, so **compare models within this table, not against paper numbers**.

**Important caveat.** Our models are pre-trained on PTB-XL training folds, so the four PTB-XL tasks are *in-domain* for us and *out-of-domain* for MERL. **CSN and CPSC2018 are out-of-domain for every model** and are the fair comparison.


## Headline (averages over the six tasks)

| Model | Pre-training pairs | Zero-shot | Linear probe 1 % | 10 % | 100 % |
|---|---|---|---|---|---|
| MERL, released checkpoint | ~772k (MIMIC-IV-ECG) | **77.4** | 74.5 | 84.4 | **88.9** |
| **A6: ResNet18 + MedCPT fine-tuned** (2 seeds) | 17k (PTB-XL) | **77.0** (76.9 / 77.1) | 74.6 (73.8 / 75.4) | 83.7 | 88.3 |
| A1: ECG-FM + MedCPT fine-tuned | 17k | 75.2 | **75.9** | **84.9** | 88.8 |
| A0: last week's configuration (ECG-FM + Bio_ClinicalBERT) | 17k | 72.8 | 73.5 | 84.4 | 88.1 |
| Random init (ECG-FM, no pre-training) | – | – | 65.9 | 76.1 | 82.9 |

**The MERL-inspired changes raise our zero-shot average from 72.8 to 77.0 (+4.2), to within 0.4 of MERL, with ~45× less pre-training data; the A1 variant matches or beats MERL's linear probe at 1–10 % labels.**

### Out-of-domain only (CSN + CPSC2018 — no model saw these datasets)

| Model | CSN zero-shot | CSN LP 1 / 10 / 100 % | CPSC zero-shot | CPSC LP 1 / 10 / 100 % |
|---|---|---|---|---|
| MERL | **75.7** | 72.7 / **87.6** / **93.6** | **83.9** | 77.4 / 89.6 / **93.3** |
| A6 seed 0 / seed 1 | 73.2 / 72.9 | **72.8** / 82.9 / 91.6 · 71.8 / 82.7 / 90.9 | 75.2 / 76.0 | 79.5 / **89.8** / 92.4 · **81.4** / **89.8** / 92.6 |
| A1 | 70.8 | 70.9 / 85.7 / 90.7 | 74.3 | 79.4 / 89.2 / 92.4 |
| A0 | 71.2 | 70.4 / 84.0 / 89.7 | 69.9 | 79.6 / 86.5 / 90.3 |

Honest reading: **out of domain, MERL is still clearly better at zero-shot** (CSN +2.5–2.8, CPSC +8–9) **and at 10–100 % linear probing on CSN**; with few labels (1 %) our model is on par or better. Part of our 6-task parity comes from the PTB-XL tasks, which are in-domain for us. Closing the out-of-domain zero-shot gap is what pre-training on MIMIC-IV-ECG (open access) in the lab should test.

## Ablations (one change at a time vs A0, 5 epochs each)

| Change (MERL idea) | Zero-shot avg | LP 1 % avg | Verdict |
|---|---|---|---|
| Bio_ClinicalBERT → **MedCPT**, fine-tuned (A1) | 72.8 → **75.2** | 73.5 → **75.9** | **keep** — largest single gain, as in MERL's Table 4 |
| MedCPT **frozen** (A2, MERL's code) | 72.5 | 75.1 | drop — zero-shot on CSN/CPSC falls 3–8 points vs fine-tuned |
| Bio_ClinicalBERT **frozen** (A3) | 71.1 | 73.4 | drop — confirms: with 17k pairs the text encoder must be fine-tuned |
| + **UMA** loss (A4) | 72.2 | 74.4 | not adopted — helps linear probe on several tasks (rhythm 1 %: 64 → 72) but slightly hurts zero-shot |
| ECG-FM Transformer → **ResNet18** (A5) | **75.7** | 73.1 | **keep for zero-shot** — reproduces MERL's "CNN > Transformer" (Table 8); 4M vs 86M params, faster to train and evaluate |
| ResNet18 + MedCPT (A6) | **76.9 / 77.1** | 73.8 / 75.4 | **best zero-shot → v3 notebook default** |
| A1 trained 15 epochs (F2) | 72.3 | 73.9 | worse — validation zero-shot peaks at epoch 2–4, then declines (over-fits PTB-XL report style) |

Prompt sets: MERL's **CKEPE prompts were selected on validation for most tasks and models**; our SCP-description prompts win on PTB-XL-super for some models.

## What to tell your professor

1. Re-implemented MERL's evaluation exactly (their splits, their checkpoint, their prompts) and reproduced their PTB-XL linear-probe numbers (82.8 vs 82.4 at 1 %).
2. Adopting MERL's text encoder (MedCPT) and ECG backbone (ResNet18) — but **fine-tuning** the text encoder, unlike MERL — closes most of the gap: 6-task zero-shot 77.0 vs 77.4 with 45× less data.
3. MERL's other two ingredients did not transfer at this scale: frozen text encoder (hurts) and UMA (helps linear probe, not zero-shot).
4. Remaining gap is out-of-domain zero-shot (CPSC −8 to −9, CSN −2.5) → next step: pre-train on MIMIC-IV-ECG in the lab and select checkpoints on validation zero-shot averaged over several datasets (single-task selection picked a bad epoch in F2).

## Full tables

## Zero-shot macro AUROC (test). Prompt set chosen per model on validation, MERL-style CKEPE in brackets
| model | ptbxl_super | ptbxl_sub | ptbxl_form | ptbxl_rhythm | csn | cpsc | mean |
|---|---|---|---|---|---|---|---|
| A0_base_bcb_ft | 78.6 [76.1] | 77.1 [77.1] | 62.6 [62.6] | 77.2 [77.2] | 71.2 [71.2] | 69.9 [66.1] | 72.8 |
| A1_medcpt_ft | 83.0 [83.0] | 77.8 [77.8] | 67.9 [67.9] | 77.4 [74.9] | 70.8 [70.8] | 74.3 [72.0] | 75.2 |
| A2_medcpt_frozen | 83.1 [83.1] | 76.4 [76.4] | 67.4 [67.4] | 74.6 [72.1] | 67.3 [67.3] | 66.1 [65.9] | 72.5 |
| A3_bcb_frozen | 80.9 [80.9] | 76.6 [76.6] | 62.1 [62.1] | 72.2 [72.2] | 68.8 [68.8] | 65.9 [64.7] | 71.1 |
| A4_bcb_ft_uma | 77.4 [77.4] | 78.0 [78.0] | 62.8 [62.8] | 75.9 [75.9] | 69.2 [69.2] | 69.9 [67.4] | 72.2 |
| A5_resnet18_bcb_ft | 83.2 [83.2] | 80.3 [80.3] | 64.3 [64.3] | 82.0 [82.0] | 72.8 [72.8] | 71.8 [69.5] | 75.7 |
| A6_resnet18_medcpt_ft | 83.2 [82.8] | 78.9 [78.9] | 69.9 [69.9] | 80.8 [80.8] | 73.2 [73.2] | 75.2 [73.6] | 76.9 |
| A6s1_resnet18_medcpt_ft_seed1 | 83.5 [83.5] | 79.7 [79.7] | 70.2 [70.2] | 80.5 [80.5] | 72.9 [72.9] | 76.0 [74.3] | 77.1 |
| F2_medcpt_ft_15ep | 81.3 [80.7] | 75.1 [74.5] | 68.4 [68.4] | 73.4 [73.4] | 64.7 [64.7] | 70.7 [68.2] | 72.3 |
| MERL_released_res18 | 77.3 [77.3] | 78.2 [78.2] | 67.5 [67.5] | 81.6 [81.6] | 75.7 [75.7] | 83.9 [83.9] | 77.4 |
| ours_F1_best_epoch4 | 81.6 [77.8] | 78.1 [78.1] | 61.6 [61.6] | 75.5 [75.5] | 71.3 [71.3] | 70.7 [67.7] | 73.1 |
| random_init_ecgfm | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Linear probe macro AUROC (test), 1% / 10% / 100% of the training split
| model | ptbxl_super | ptbxl_sub | ptbxl_form | ptbxl_rhythm | csn | cpsc |
|---|---|---|---|---|---|---|
| A0_base_bcb_ft | 86.2 / 88.9 / 89.8 | 74.2 / 81.3 / 84.2 | 66.2 / 78.3 / 84.0 | 64.3 / 87.4 / 90.2 | 70.4 / 84.0 / 89.7 | 79.6 / 86.5 / 90.3 |
| A1_medcpt_ft | 86.6 / 89.5 / 90.3 | 75.1 / 82.2 / 85.2 | 71.4 / 76.8 / 84.4 | 72.1 / 85.8 / 89.9 | 70.9 / 85.7 / 90.7 | 79.4 / 89.2 / 92.4 |
| A2_medcpt_frozen | 86.4 / 88.6 / 89.4 | 74.7 / 81.5 / 83.7 | 67.8 / 73.0 / 82.1 | 71.7 / 84.3 / 87.8 | 70.6 / 83.4 / 89.1 | 79.8 / 86.6 / 90.8 |
| A3_bcb_frozen | 87.7 / 89.2 / 90.0 | 74.6 / 82.0 / 85.3 | 68.6 / 71.7 / 84.4 | 65.3 / 84.4 / 89.6 | 68.2 / 83.5 / 90.7 | 76.2 / 86.9 / 90.3 |
| A4_bcb_ft_uma | 86.3 / 89.2 / 90.2 | 75.5 / 81.4 / 86.3 | 67.9 / 78.2 / 84.3 | 71.9 / 86.9 / 91.1 | 67.8 / 83.3 / 90.9 | 76.9 / 86.9 / 91.1 |
| A5_resnet18_bcb_ft | 85.1 / 88.5 / 89.7 | 73.0 / 80.6 / 82.0 | 69.8 / 78.8 / 83.1 | 61.4 / 84.2 / 86.5 | 71.0 / 83.1 / 90.8 | 78.6 / 88.7 / 92.1 |
| A6_resnet18_medcpt_ft | 84.9 / 88.7 / 90.0 | 74.3 / 80.5 / 83.9 | 68.7 / 77.3 / 83.8 | 62.8 / 85.6 / 88.0 | 72.8 / 82.9 / 91.6 | 79.5 / 89.8 / 92.4 |
| A6s1_resnet18_medcpt_ft_seed1 | 83.8 / 88.6 / 90.2 | 73.3 / 80.6 / 84.2 | 71.2 / 74.3 / 84.5 | 70.8 / 83.4 / 86.7 | 71.8 / 82.7 / 90.9 | 81.4 / 89.8 / 92.6 |
| F2_medcpt_ft_15ep | 85.8 / 88.5 / 89.4 | 75.2 / 81.5 / 86.1 | 67.3 / 73.0 / 82.9 | 67.6 / 83.4 / 88.5 | 71.2 / 82.7 / 88.8 | 76.3 / 85.7 / 90.1 |
| MERL_released_res18 | 82.8 / 89.2 / 91.1 | 74.1 / 80.1 / 84.3 | 62.7 / 74.1 / 80.7 | 77.4 / 85.9 / 90.3 | 72.7 / 87.6 / 93.6 | 77.4 / 89.6 / 93.3 |
| ours_F1_best_epoch4 | 86.3 / 88.6 / 89.7 | 74.6 / 81.1 / 85.0 | 66.4 / 75.7 / 82.2 | 68.8 / 88.5 / 90.0 | 70.5 / 83.7 / 88.9 | 76.7 / 87.9 / 91.5 |
| random_init_ecgfm | 75.0 / 81.9 / 86.0 | 68.6 / 75.1 / 81.0 | 58.2 / 64.9 / 73.1 | 62.8 / 77.4 / 84.4 | 64.9 / 77.1 / 86.4 | 65.9 / 80.2 / 86.4 |
| *MERL paper, Table 1* | 82.4 / 86.3 / 88.7 | 64.9 / 80.6 / 84.7 | 58.3 / 72.4 / 79.7 | 53.3 / 82.9 / 88.3 | 66.6 / 82.7 / 88.0 | 70.3 / 85.3 / 90.6 |
| *Random init, paper Table 1* | 70.5 / 77.1 / 81.6 | 55.8 / 67.6 / 77.9 | 55.8 / 62.5 / 73.0 | 46.3 / 62.4 / 79.3 | 47.2 / 63.2 / 73.1 | 55.0 / 71.5 / 78.3 |

## Averages over the tasks each model was evaluated on (n = number of tasks)
| model | n | zero-shot (best prompt set on val) | zero-shot (CKEPE) | LP 1% | LP 10% | LP 100% |
|---|---|---|---|---|---|---|
| A0_base_bcb_ft | 6 | 72.8 | 71.7 | 73.5 | 84.4 | 88.1 |
| A1_medcpt_ft | 6 | 75.2 | 74.4 | 75.9 | 84.9 | 88.8 |
| A2_medcpt_frozen | 6 | 72.5 | 72.0 | 75.1 | 82.9 | 87.2 |
| A3_bcb_frozen | 6 | 71.1 | 70.9 | 73.4 | 82.9 | 88.4 |
| A4_bcb_ft_uma | 6 | 72.2 | 71.8 | 74.4 | 84.3 | 89.0 |
| A5_resnet18_bcb_ft | 6 | 75.7 | 75.3 | 73.1 | 84.0 | 87.4 |
| A6_resnet18_medcpt_ft | 6 | 76.9 | 76.5 | 73.8 | 84.1 | 88.3 |
| A6s1_resnet18_medcpt_ft_seed1 | 6 | 77.1 | 76.8 | 75.4 | 83.2 | 88.2 |
| F2_medcpt_ft_15ep | 6 | 72.3 | 71.7 | 73.9 | 82.5 | 87.6 |
| MERL_released_res18 | 6 | 77.4 | 77.4 | 74.5 | 84.4 | 88.9 |
| ours_F1_best_epoch4 | 6 | 73.1 | 72.0 | 73.9 | 84.2 | 87.9 |
| random_init_ecgfm | 6 | n/a | n/a | 65.9 | 76.1 | 82.9 |
| *MERL paper (zero-shot = CKEPE, 6 tasks)* | 6 | 75.2 | 75.2 | 66.0 | 81.7 | 86.6 |


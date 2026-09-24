# Overnight validation of the ECG ↔ Text CLIP pipeline (PTB-XL)

_Run on 2026-09-24 on a laptop RTX 3060 (6 GB) with bf16 + gradient checkpointing, batch 64.
All numbers are on the **official PTB-XL test fold (fold 10, n = 2,198)**; model/prompt choices were made on fold 9 only._

## 1. Bugs found in the v1 notebook (these change the conclusions of the Colab run)

| # | Bug | Effect |
|---|---|---|
| 1 | Superclass labels looked for a column `diagnostic_superclass` that does not exist in `scp_statements.csv` (it is `diagnostic_class`) and silently fell back to a hand-written map that sent rhythm codes (`SR`, `STACH`, `SBRAD`) to NORM and `AFIB` to CD. | Label counts were NORM 12,756 / STTC 717 instead of the official NORM 9,514 / STTC 5,235. The "zero-shot 52 % / linear-probe 62 %" results were measured against wrong labels. |
| 2 | `classification_report(labels, preds, target_names=LABEL_ORDER)` assigns names to the labels in **alphabetical** order (CD, HYP, MI, NORM, STTC), not in `LABEL_ORDER`. | Every per-class row was mislabelled (the row printed "CD, F1 0.70, support 1306" was actually **NORM**). |
| 3 | Text builder dropped every SCP statement with likelihood < 50. PTB-XL stores all rhythm/form statements with likelihood 0. | "sinus rhythm", "atrial fibrillation", "sinus tachycardia"… never appeared in the training text. |
| 4 | `SEX_MAP = {0: 'female', 1: 'male'}`; PTB-XL uses 0 = male, 1 = female. | Every text stated the wrong sex. |
| 5 | Evaluation took a random 5-s crop of each test ECG. | Test numbers changed from run to run. Now: average of both 5-s windows (deterministic). |
| 6 | Temperature LR = 2e-5. | τ stayed at 0.0694–0.0700 for the whole run (it never learned). |
| 7 | No random-init baseline. | A **completely untrained** encoder already reaches 0.861 linear-probe AUROC; any claim must be measured against this. |
| 8 | `ECG_PRETRAINED` warm start loaded with `strict=False`, but `Pretrain.ipynb` names its modules differently (`feature_extractor`, `transformer_layers`, `final_norm`…). | A CMSC checkpoint would have loaded **zero** weights without any error. v2 maps the keys and loads with `strict=True`. |

## 2. What was changed (all in `ECG_Multimodal.ipynb` v2)

- Correct multi-label superclass targets; standard PTB-XL metric (**macro AUROC**) + single-label accuracy with correct class names.
- **Text source = translated cardiologist reports** (`data/ptbxl_reports_en.csv`, see §5). Option `TEXT_SOURCE='scp'` kept.
- **Multi-positive contrastive loss**: a batch of 64 contains ~26 pairs of ECGs with an *identical* report ("sinus rhythm. normal ecg."). Plain CLIP treats those as negatives. (Ablation R4: with report text this makes no measurable difference — kept because it is the correct objective, not claimed as a gain.)
- LRs: ECG encoder 1e-4 (trained from scratch), BERT 2e-5, projection heads 2e-4, temperature 1e-3; linear warm-up + cosine.
- Random 5-s crop anywhere in the 10 s for training; deterministic 2-window average for evaluation.
- Signal cache (`.npy`, built once in ~2–6 min) → no `wfdb` reads during training.
- bf16 autocast, optional gradient checkpointing (`GRAD_CKPT=True` for < 12 GB GPUs).
- `best.pt` = weights only (0.8 GB); `last.pt` = weights + optimizer (2.3 GB) for resuming. v1 saved two 2.3 GB files to Drive every epoch.
- Linear probe with 1 % / 10 % / 100 % of the training labels + random-init baseline row.

## 3. Results (5 epochs each unless stated)

Zero-shot = no labelled ECG used; score = cosine similarity to the SCP statement descriptions of each superclass
("max" = best-matching statement; selected on validation, see §4).

| Run | Text | Loss | LR (ECG / BERT) | Zero-shot AUROC (SCP prompts) | Zero-shot AUROC (hand-written prompts) | Linear probe 1 % | 10 % | 100 % |
|---|---|---|---|---|---|---|---|---|
| Random init | – | – | – | 0.516 | 0.523 | 0.764 | 0.830 | 0.861 |
| R0 = v1 notebook settings | SCP (v1 builder) | CLIP | 2e-5 / 2e-6 | **0.799** | 0.683 | 0.862 | 0.878 | 0.885 |
| R1 | SCP (all statements) | multi-pos | 1e-4 / 2e-5 | 0.704 (0.779 at best epoch) | 0.668 | **0.882** | **0.896** | **0.905** |
| **R2 (chosen)** | **reports (EN)** | multi-pos | 1e-4 / 2e-5 | 0.787 | **0.741** | 0.872 | 0.888 | 0.900 |
| R3 | reports (EN) | multi-pos | 1e-4 / **2e-6** | 0.730 | 0.746 | 0.874 | 0.889 | 0.898 |
| R4 | reports (EN) | **plain CLIP** | 1e-4 / 2e-5 | 0.783 | 0.735 | 0.873 | 0.887 | 0.898 |
| R2, seed 1 (replicate) | reports (EN) | multi-pos | 1e-4 / 2e-5 | 0.775 | 0.734 | 0.874 | 0.891 | 0.901 |
| **F1 = R2 for 15 epochs** (last epoch) | reports (EN) | multi-pos | 1e-4 / 2e-5 | 0.780 | 0.744 | **0.874** | **0.890** | **0.902** |
| **F1, checkpoint selected on val zero-shot (epoch 4)** | reports (EN) | multi-pos | 1e-4 / 2e-5 | **0.814** | **0.772** | – | – | – |

### F1: what longer training buys (same config as R2)

| Epoch | Zero-shot AUROC (test) | Linear probe 1 % / 10 % / 100 % (test) | ECG→report R@5 (val) |
|---|---|---|---|
| 5 | 0.784 | 0.869 / 0.888 / 0.898 | 0.165 |
| 10 | 0.773 | 0.873 / 0.890 / 0.902 | 0.235 |
| 15 | 0.780 | 0.874 / 0.890 / 0.902 | 0.261 |

Validation zero-shot fluctuates 0.77–0.82 from epoch to epoch with no trend (peak 0.825 at epoch 4); training loss keeps falling (3.45 → 1.27).
**The ECG representation plateaus after ~10–15 epochs on PTB-XL alone; only cross-modal retrieval keeps improving.**
→ Recommended lab run: **15 epochs, select the checkpoint on validation zero-shot AUROC** (the v2 notebook does this). Selecting F1's best-validation checkpoint (val 0.825) gives **test zero-shot AUROC 0.814** (NORM 0.898, MI 0.810, STTC 0.840, CD 0.802, HYP 0.718) — the best zero-shot result of the night, above R0's label-leaked 0.799.  The next real gain will come from more ECG–report pairs (e.g. MIMIC-IV-ECG, ~800k pairs), not from more epochs.

![](figures/fig_linear_probe_and_curves.png)
![](figures/fig_zero_shot_per_class.png)

### Run-to-run noise (R2 seed 0 vs seed 1)

Zero-shot AUROC 0.787 vs 0.775 (**±0.006**); linear probe within **0.003** at every label fraction.
So, at this scale:

| Comparison | Difference | Verdict |
|---|---|---|
| CLIP pre-training vs random init (linear probe 1 %) | +0.11 | real |
| v2 fixes vs v1 settings (linear probe, R2 vs R0) | +0.010 to +0.015 | real |
| Label-derived text vs reports (linear probe, R1 vs R2) | +0.005 to +0.010 | real but small |
| BERT LR 2e-6 vs 2e-5 (zero-shot, R3 vs R2) | −0.057 | real |
| Reports vs label-derived text with natural-language prompts (R2 vs R0) | +0.05 | real |
| Multi-positive vs plain CLIP (R2 vs R4) | +0.004 | **noise** |
| 15 vs 5 epochs (linear probe, F1 vs R2) | +0.002 | **noise**; retrieval +0.07 is real |

### How to read this

- **Pre-training works, and helps most when labels are scarce.** With 1 % of the labels (~170 ECGs) the CLIP encoder beats the random encoder by ≈ +0.11 AUROC; with 100 % the gap shrinks to ≈ +0.04. That is the expected signature of useful self-/weakly-supervised pre-training and is the headline result to report.
- **R0's 0.799 zero-shot is not a fair zero-shot result**: its training text is generated from the same SCP labels that define the evaluation classes (label leakage). R2 learns from free-text reports written by cardiologists and reaches 0.78 at epoch 5 (0.787 / 0.775 over two seeds); with checkpoint selection on validation, F1 reaches **0.814** — this is the honest headline number.
- **R1 learns the best ECG features** (linear probe) because its text is a clean, complete label description — again, label-derived supervision. Its zero-shot peaked at epoch 2 and then degraded.
- **R2 vs R3**: lowering BERT's LR does not change the ECG features (linear probe identical) but hurts zero-shot (0.787 → 0.730) — BERT has to adapt to the reports' clinical-shorthand style.
- Retrieval (validation, ECG → its own report among all unique reports): R0 R@5 = 0.11, R1 0.32, R2 0.19.

## 4. Zero-shot prompt study (post-hoc, chosen on validation)

| Prompt set | R0 | R1 | R2 | F1 |
|---|---|---|---|---|
| SCP descriptions, **max** over statements | **0.799** | **0.779** | **0.779** | **0.814** |
| SCP descriptions, mean | 0.779 | 0.740 | 0.748 | – |
| v1 hand-written templates | 0.700 | 0.703 | 0.748 | 0.772 |
| SCP descriptions + "sinus rhythm." | 0.667 | 0.720 | 0.572 | – |

(best-epoch checkpoints; test AUROC). Adding shared context like "sinus rhythm." to every prompt collapses HYP to ≈ 0.35, because the shared phrase dominates the similarity. **Use the plain statement descriptions with max-aggregation** (what the notebook does).

## 5. Report translation (`src/translate_reports.py` → `data/ptbxl_reports_en.csv`)

PTB-XL reports are 67 % German, 30 % English, 3 % Swedish, with encoding artefacts ("vÄnster") and boilerplate
("4.46 unbestätigter bericht" = unconfirmed report). Pipeline: clean → **German cardiology glossary** (~90 rules, e.g.
"überdrehter linkstyp" → "left axis deviation", "p-sinistrocardiale" → "left atrial enlargement",
"alter unbest." → "age undetermined") → MarianMT (Helsinki-NLP opus-mt-de-en / sv-en, CPU) only for the phrases the glossary does not cover.
Without the glossary MarianMT produced e.g. "overturned left type" for left axis deviation. Result: 9,111 unique English reports.
Known leftovers: 9 reports with word-order variants ("linkstyp überdreht" → "left axis. overturned"), rare OCR-like typos.

## 6. Limitations / honest caveats for the thesis

- 5-epoch runs on 17k ECGs; one replicate seed only. Zero-shot differences below ~0.015 and linear-probe differences below ~0.005 should be treated as noise.
- Zero-shot classes are the 5 coarse superclasses; the SCP-description prompts are the label names of the diagnostic statements (standard "class-name prompt" zero-shot, as in CLIP), not hand-tuned per model.
- The ECG encoder is trained from scratch; warm-starting it from the CMSC checkpoint of `Pretrain.ipynb` (`ECG_PRETRAINED=...`, keys mapped by `map_cmsc_checkpoint`, verified to load strictly and reproduce the Pretrain encoder's outputs) is the obvious next experiment.
- Machine-translated reports contain some errors (see §5).

## 7. Reproducing these numbers

Every run is one line of [`src/experiments.txt`](../src/experiments.txt) (`python src/ecgclip.py train <line>`);
`runs/<name>/results.jsonl` holds each evaluation, `runs/<name>/log.txt` the training log,
`runs/posthoc_*.txt` the prompt study. `python src/summarize.py --curves` rebuilds the table in §3,
`python src/make_figures.py` the figures. The random-init baseline is the `init` evaluation of R0 (`--eval_init 1`).
The best checkpoint (`F1`, epoch 4, 0.8 GB) is not in the repository because of GitHub's file-size limit.

# ECG ↔ Text Contrastive Learning on PTB-XL

A CLIP-style multimodal model that aligns **12-lead ECG signals** with **clinical text** (cardiologist reports),
so that ECGs can be classified from plain-English descriptions (zero-shot), retrieved by text query, and used as
pre-trained features for label-efficient downstream classification.

- **ECG encoder**: 1-D ResNet18 (4 M params, from MERL; default) or an ECG-FM-style CNN + 12-layer Transformer (86 M), trained from scratch.
- **Text encoder**: [MedCPT-Query](https://huggingface.co/ncbi/MedCPT-Query-Encoder) (default) or [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), fine-tuned.
- **Objective**: symmetric InfoNCE with a learnable temperature in a shared 512-d space (multi-positive variant for identical reports).
- **Data**: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) — 21,799 ECGs (500 Hz, 10 s), official folds (1–8 train / 9 val / 10 test).
- **Text supervision**: the cardiologists' free-text reports, cleaned and translated to English (they are 67 % German, 3 % Swedish) — see [`data/`](data).

## Comparison with MERL (ICML 2024) on MERL's own benchmark — new

Six tasks (PTB-XL super / sub / form / rhythm, CSN, CPSC2018), MERL's splits and released checkpoint, identical protocol
([`src/merl_bench.py`](src/merl_bench.py)); macro AUROC averaged over the six tasks:

| Model | Pre-training pairs | Zero-shot | Linear probe 1 % | 10 % | 100 % |
|---|---|---|---|---|---|
| MERL (released checkpoint) | ~772k MIMIC-IV-ECG | **77.4** | 74.5 | 84.4 | **88.9** |
| **This work, v3 (ResNet18 + MedCPT fine-tuned), 2 seeds** | 17k PTB-XL | **77.0** | 74.6 | 83.7 | 88.3 |
| This work, v2 configuration | 17k PTB-XL | 72.8 | 73.5 | 84.4 | 88.1 |
| Untrained encoder | – | – | 65.9 | 76.1 | 82.9 |

MERL-inspired changes (MedCPT text encoder, ResNet18 backbone) raise zero-shot by +4.2 to within 0.4 of MERL with ~45× less data.
Out of domain (CSN, CPSC2018) MERL keeps a clear zero-shot lead — see **[`results/MERL_COMPARISON.md`](results/MERL_COMPARISON.md)**
for the full tables, ablations (frozen text encoder, UMA, backbone, epochs) and caveats.

## Next: pre-training on MIMIC-IV-ECG (lab)

Pipeline for the same ~800k ECG–report pairs MERL used (open access): resumable parallel download, preparation straight
from the zip (patient-level split, leads reordered by name, MERL's report cleaning), memory-mapped training with
checkpoint selection on MERL-validation zero-shot, resume, and evaluation — tested end to end on a 3k-record subset.
Step-by-step guide: **[`docs/MIMIC_LAB.md`](docs/MIMIC_LAB.md)** · one script: [`lab/mimic_pipeline.ps1`](lab/mimic_pipeline.ps1).

## v2 results (PTB-XL test fold, macro AUROC over the 5 diagnostic superclasses)

| Model | Zero-shot (no labelled ECGs) | Linear probe, 1 % labels | 10 % | 100 % |
|---|---|---|---|---|
| Untrained encoder (baseline) | 0.516 | 0.764 | 0.830 | 0.861 |
| **This work, 15 epochs** | **0.814**¹ | **0.874** | **0.890** | **0.902** |

¹ checkpoint selected on validation zero-shot AUROC; last-epoch checkpoint: 0.780.

- Pre-training helps most when labels are scarce: **+0.11 AUROC with 1 % of the labels** (~170 ECGs).
- The model is trained on free-text reports, not on text generated from the labels, so the zero-shot number is not inflated by label leakage.
- Full ablations, run-to-run noise, the zero-shot prompt study and all caveats: **[`results/RESULTS.md`](results/RESULTS.md)**.

![](results/figures/fig_linear_probe_and_curves.png)
![](results/figures/fig_zero_shot_per_class.png)

## Pipeline

```
12-lead ECG, 500 Hz, 10 s ─► z-score per lead ─► random 5-s crop (train) / mean of both 5-s windows (eval)
        │
        ▼
  CNN (4 blocks, 256 ch) ─► 12-layer Transformer (768-d, conv. relative positions) ─► mean-pool ─► MLP ─► L2 ─┐
                                                                                                           ├─► InfoNCE (τ learnable)
  Report text ─► Bio_ClinicalBERT ─► [CLS] ─► MLP ─► L2 ─────────────────────────────────────────────────────┘

Evaluation
  zero-shot     cos(ECG, text of each superclass' SCP statements), max over statements → macro AUROC
  linear probe  frozen 768-d ECG features + logistic regression, 1 % / 10 % / 100 % of training labels
  retrieval     ECG → its own report among all unique reports (Recall@1/5)
```

## Repository layout

| Path | Contents |
|---|---|
| [`notebooks/ECG_Multimodal.ipynb`](notebooks/ECG_Multimodal.ipynb) | **Main pipeline** (Colab / lab GPU): data, labels, text, signal cache, model, training, zero-shot, linear probe (+ random-init baseline), t-SNE, retrieval |
| [`notebooks/Pretrain_CMSC.ipynb`](notebooks/Pretrain_CMSC.ipynb) | ECG-only contrastive pre-training (CMSC); its checkpoint can warm-start the ECG encoder (`ECG_PRETRAINED`) |
| [`src/ecgclip.py`](src/ecgclip.py) | Script version used for every experiment in `results/` (same model code / state-dict keys as the notebook) |
| [`src/experiments.txt`](src/experiments.txt) | Exact command-line arguments of every run |
| [`src/translate_reports.py`](src/translate_reports.py) | Report cleaning + German cardiology glossary + MarianMT translation |
| [`src/posthoc_eval.py`](src/posthoc_eval.py), [`src/summarize.py`](src/summarize.py), [`src/make_figures.py`](src/make_figures.py) | Prompt study, results table, figures |
| [`src/download_ptbxl.py`](src/download_ptbxl.py) | Parallel download of PTB-XL (500 Hz records) from PhysioNet's AWS open-data mirror |
| [`src/merl_bench.py`](src/merl_bench.py), [`src/experiments_merl.txt`](src/experiments_merl.txt) | MERL benchmark (zero-shot + linear probe on 6 tasks) and the exact runs of the MERL comparison |
| [`src/download_cpsc2018.py`](src/download_cpsc2018.py), [`src/merl_resnet1d.py`](src/merl_resnet1d.py) | CPSC2018 download; MERL's ResNet18 (MIT) |
| [`merl_ref/`](merl_ref) | MERL's splits, CKEPE prompts and reference code (MIT licence) |
| [`src/download_mimic_ecg.py`](src/download_mimic_ecg.py), [`src/prepare_mimic.py`](src/prepare_mimic.py), [`lab/`](lab), [`docs/MIMIC_LAB.md`](docs/MIMIC_LAB.md) | MIMIC-IV-ECG download / preparation, lab script and guide (`ecgclip.py --dataset mimic`) |
| [`data/ptbxl_reports_en.csv`](data/ptbxl_reports_en.csv) | English translation of every PTB-XL report |
| [`results/`](results) | Write-up, figures, per-run logs (`results.jsonl` = every evaluation, `log.txt` = training log) |

## Quick start

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # GPU build first (CPU-only otherwise on Windows)
pip install -r requirements.txt          # exact versions used for results/ (full snapshot: requirements-lock.txt)
python src/download_ptbxl.py            # or download the zip from PhysioNet into data/
```

**Notebook (Colab or lab GPU)**: open `notebooks/ECG_Multimodal.ipynb`, set `PTB_DIR`, `SAVE_DIR`, `CACHE_DIR`
`REPORTS_EN_CSV` and `CKEPE_JSON` in the configuration cell, run all. Defaults are the best configuration on MERL's
benchmark (ResNet18 + MedCPT fine-tuned, reports as text, 5 epochs, LR 2e-4 ECG / 2e-5 text / 2e-4 heads / 1e-3
temperature, batch 64, bf16). Switches: `ECG_ARCH` (`resnet18` / `fm`), `TEXT_MODEL_NAME`, `FREEZE_TEXT`, `USE_UMA`.
On GPUs with < 12 GB set `GRAD_CKPT = True`.

**Scripts** (paths configurable through `PTB_DIR`, `REPORTS_EN_CSV`, `ECG_CACHE`, `RUNS_DIR`):

```bash
python src/ecgclip.py cache                                    # build the 2.6 GB signal cache once
python src/ecgclip.py train --name F1_report_15ep --text report --multipos 1 \
    --lr_ecg 1e-4 --lr_text 2e-5 --lr_proj 2e-4 --lr_temp 1e-3 --epochs 15 --ckpt 1 --full_every 5
```

Hardware used for `results/`: one RTX 3060 Laptop GPU (6 GB), bf16 + gradient checkpointing, ≈ 6 min / epoch.

## Data and licences

- PTB-XL is © its authors, released under **CC BY 4.0** by PhysioNet. `data/ptbxl_reports_en.csv` is a derived work
  (cleaned + machine-translated `report` field) and is shared under the same licence. Please cite:
  Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154 (2020);
  and Goldberger, A. et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23) (2000).
- Machine translations contain some errors (documented in `results/RESULTS.md` §5); they are not clinical-grade.

## References

- McKeen, K. et al. *ECG-FM: An Open Electrocardiogram Foundation Model.* arXiv:2408.05178 (2024) — ECG encoder architecture and CMSC pre-training.
- Yang, K. et al. *ECG-LM: Understanding Electrocardiogram with a Large Language Model.* Health Data Science (2025), doi:10.34133/hds.0221.
- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- Alsentzer, E. et al. *Publicly Available Clinical BERT Embeddings.* Clinical NLP Workshop 2019.
- Liu, C. et al. *Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement (MERL).* ICML 2024 — benchmark, splits, CKEPE prompts, ResNet18 backbone.
- Jin, Q. et al. *MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs.* Bioinformatics 39(11) (2023).

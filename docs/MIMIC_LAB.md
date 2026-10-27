# Pre-training on MIMIC-IV-ECG in the lab

Goal: pre-train the v3 model (1-D ResNet18 + MedCPT, CLIP loss) on **MIMIC-IV-ECG** (~800k ECG + machine-report pairs,
the data MERL used) instead of PTB-XL, then evaluate on MERL's benchmark — where PTB-XL, CSN and CPSC2018 are now all
**out-of-domain**, making the comparison with MERL fully fair.

Everything runs through [`lab/mimic_pipeline.ps1`](../lab/mimic_pipeline.ps1) (Windows PowerShell, conda env `ecgmm`).
Each step can be stopped and re-run safely.

## 0. Requirements

| | |
|---|---|
| GPU | RTX 5090 (32 GB). Batch 256 without gradient checkpointing; on smaller GPUs add `--ckpt 1` or lower `--bs` |
| Disk | **~140 GB free** on a fast SSD: 36.3 GB zip + ~93 GB prepared signal cache (+6 GB benchmark caches). The zip can be deleted after `prepare` |
| RAM | 16 GB is enough — the 93 GB cache is read as a memory map |
| Data already in the lab | PTB-XL and the CSN ("arrhythmia study") folder under `ECG LM\` (set `$Root` / the paths at the top of the script if they differ) |
| Software | Python 3.10, versions pinned in `requirements.txt` (`-Step setup` installs them, torch 2.11.0 + CUDA 12.8) |
| Licence | MIMIC-IV-ECG is **open access** (ODbL), no credentialing needed |

```powershell
conda activate ecgmm
cd "C:\Users\eq26\Desktop\ECG LM"
git clone https://github.com/equintero75/ecg-text-clip
cd ecg-text-clip
.\lab\mimic_pipeline.ps1 -Step setup
```
If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process Bypass`.

## 1. Download (`-Step download`) — start it the evening before

```powershell
.\lab\mimic_pipeline.ps1 -Step download -Data D:\mimic
```
Downloads the 36.3 GB zip from PhysioNet in 64 MB chunks over 24 parallel connections, then CPSC2018 (0.9 GB).
PhysioNet throughput varies a lot: on a home connection we measured **0.25 MB/s total** (24 connections) on
2026-10-03 — that would be ~40 h — while on other days a single connection gave 128 KB/s. Look at the `MB/s` printed
after the first few chunks and divide 36,300 MB by it; a university network is usually much faster. If it stops
(network, reboot), run the same command again: finished chunks are kept.

## 2. Prepare (`-Step prepare`)

```powershell
.\lab\mimic_pipeline.ps1 -Step prepare -Data D:\mimic
```
Reads every record **directly from the zip** (no 90 GB extraction) with 8 processes and writes `D:\mimic\prepared\`:

- `sig.npy` — float16 `[N, 12, 5000]`, 10 s @ 500 Hz, **leads reordered by name to the PTB-XL order** (MIMIC stores
  I, II, III, aVR, **aVF, aVL**, …), z-scored per lead (identical preprocessing to the PTB-XL pipeline).
- `meta.csv` — `subject_id, study_id, split, bad, nan_frac, text`. Text = the machine statements `report_0…17`
  joined and lower-cased (MERL's cleaning); studies with < 3 words are dropped (as in MERL).
  **Split is by patient** (2 % of subjects for validation) — MERL split randomly by record.
- `stats.json` — counts and exclusions.

Measured on a 3,010-record test subset: reading from the zip processed ~1,000 records/s (folder: ~120/s), and the
zip-based output was byte-identical to the folder-based one. Full dataset: expect **~15–60 min** plus ~3 min for the
report text. Report cleaning on the full metadata keeps **772,735** of 800,035 studies (MERL: 771,693); the most common
report, "sinus rhythm. normal ecg.", covers 12 % of studies — so the multi-positive loss (on by default) matters here.

## 3. Benchmark caches (`-Step bench_cache`)

Builds the raw-signal caches for MERL's six tasks (PTB-XL super/sub/form/rhythm, CSN, CPSC2018). They are used **during
training** to pick the best checkpoint (mean validation zero-shot AUROC over the six tasks — the protocol MERL uses;
selecting on PTB-XL alone picked a bad epoch in our F2 run) and afterwards for the final evaluation.

## 4. Smoke test (`-Step smoke`) — 5 minutes

200 steps on 2 % of MIMIC. Check in the output: memory (`mem … GB`), speed (`… s/it`), and that the `EVAL` line contains
`val_merl_zs`. Estimate the epoch time as `s/it × (≈ 770k / batch)`.

## 5. Train (`-Step train`)

```powershell
.\lab\mimic_pipeline.ps1 -Step train -Data D:\mimic -Epochs 10 -Batch 256
```
= `src/ecgclip.py train --dataset mimic --text_model ncbi/MedCPT-Query-Encoder --ecg_arch resnet18 --lr_ecg 2e-4
--bs 256 --epochs 10 --select_on merl --resume 1`.

- After every epoch: validation loss / retrieval on the MIMIC validation patients + **MERL-validation zero-shot**;
  the best epoch is saved as `runs\<name>\best.pt`, the full state (model + optimiser) as `last.pt`.
- **Interrupted? Run the same command again** — it resumes from `last.pt` (start of the next epoch).
- Log: `runs\<name>\log.txt` (step lines every 50 steps, one `EVAL` json per epoch).

Measured on the 6 GB laptop GPU (ResNet18 + MedCPT, batch 64, gradient checkpointing): 0.55 s/step. On the 5090 with
batch 256 and no checkpointing, expect very roughly 0.3–0.6 s/step → ~3,000 steps per epoch → **~15–30 min per epoch**
plus a few minutes of validation; 10 epochs ≈ 3–6 h. Use the smoke test's `s/it` for a real estimate.

## 6. Evaluate (`-Step eval`)

```powershell
.\lab\mimic_pipeline.ps1 -Step eval
```
Runs `src/merl_bench.py` on `best.pt` (zero-shot + 1/10/100 % linear probe, 6 tasks) and prints the table next to the
existing rows (MERL's released checkpoint, our PTB-XL-pretrained models). Copy `results\merl_bench\<name>.json` into the
repo and update `results/MERL_COMPARISON.md`.

## What to expect / what to report

- Same data scale as MERL → the honest test of whether **fine-tuning MedCPT** (our change) beats MERL's frozen MedCPT
  out-of-domain, which PTB-XL-only pre-training could not show (CPSC zero-shot gap −8 to −9).
- Report both zero-shot (CKEPE prompts, and the prompt set chosen on validation) and the linear probe.
- Ablations worth a second run if time allows: `--ecg_arch fm` (86M Transformer — may benefit from 45× more data),
  `--freeze_text 1` (MERL's exact setting), `--uma 1`.

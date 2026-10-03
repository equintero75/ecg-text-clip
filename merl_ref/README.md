# Files from MERL (third-party, MIT licence)

Copied from <https://github.com/cheliu-computation/MERL-ICML2024> (Liu et al., *Zero-Shot ECG Classification with
Multimodal Learning and Test-time Clinical Knowledge Enhancement*, ICML 2024). See `LICENSE` (MIT, © 2024 Che Liu).

| Path | Used for |
|---|---|
| `finetune/data_split/` | MERL's train/val/test splits for PTB-XL (super/sub/form/rhythm), CSN and CPSC2018 — used by `src/merl_bench.py` |
| `zeroshot/CKEPE_prompt.json` | MERL's released CKEPE zero-shot prompts |
| `utils/resnet1d.py`, `utils/vit1d.py`, `utils/utils_builder.py`, `finetune/models/` | reference model code (the ResNet18 is also copied to `src/merl_resnet1d.py`) |

MERL's pretrained checkpoint (`ckpt/res18_best_ckpt.pth`, 457 MB) is **not** committed. Download it from the link in
MERL's README (Google Drive folder) into `merl_ref/ckpt/` to reproduce the MERL rows of `results/MERL_COMPARISON.md`.

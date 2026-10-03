# MIMIC-IV-ECG pre-training pipeline for the lab machine (Windows, RTX 5090, conda env "ecgmm").
# Run step by step:   .\lab\mimic_pipeline.ps1 -Step download   (then prepare, bench_cache, train, eval)
# Every step can be re-run: download resumes, prepare/bench_cache skip finished work, train resumes from last.pt.
param(
    [Parameter(Mandatory = $true)][ValidateSet('setup', 'download', 'prepare', 'bench_cache', 'smoke', 'train', 'eval')]
    [string]$Step,
    [string]$Root     = 'C:\Users\eq26\Desktop\ECG LM',            # base folder of the lab project
    [string]$Data     = 'D:\mimic',                                # >= 140 GB free: 36 GB zip + 93 GB prepared cache
    [string]$RunName  = 'M1_mimic_resnet18_medcpt',
    [int]$Epochs      = 10,
    [int]$Batch       = 256,
    [int]$Workers     = 8
)
$ErrorActionPreference = 'Stop'
$Repo = Join-Path $Root 'ecg-text-clip'
$Zip  = Join-Path $Data 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
$Prep = Join-Path $Data 'prepared'

# paths used by the scripts (see README); PTB-XL / CSN are the folders already used by the lab notebooks
$env:PTB_DIR   = Join-Path $Root 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
$env:CSN_DIR   = Join-Path $Root 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
$env:CPSC_DIR  = Join-Path $Root 'cpsc_2018'
$env:RUNS_DIR  = Join-Path $Root 'runs'
$env:ECG_CACHE = Join-Path $Root 'cache'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
Set-Location $Repo

switch ($Step) {
    'setup' {
        # once: clone the repo next to the data and install the extra packages into ecgmm
        if (-not (Test-Path $Repo)) { git clone https://github.com/equintero75/ecg-text-clip $Repo }
        python -m pip install -r (Join-Path $Repo 'requirements.txt')
    }
    'download' {
        # ~36 GB from PhysioNet with 24 parallel, resumable connections. Speed depends on PhysioNet
        # (measured 0.5-3 MB/s) -> plan for an overnight run; just re-run the command if it stops.
        python src\download_mimic_ecg.py zip --out $Data --workers 24
        python src\download_cpsc2018.py 16          # CPSC2018 for the benchmark (~0.9 GB), skips existing files
    }
    'prepare' {
        # reads the zip directly (no extraction), writes sig.npy (~93 GB) + meta.csv
        python src\prepare_mimic.py --src $Zip --out $Prep --workers $Workers
    }
    'bench_cache' {
        # signal caches for MERL's benchmark (PTB-XL, CSN, CPSC2018) -> used for checkpoint selection and evaluation
        python src\merl_bench.py cache
        python src\ecgclip.py cache                  # PTB-XL training cache (only needed for PTB-XL runs)
    }
    'smoke' {
        # 200 steps on 2 % of MIMIC: checks memory, speed (s/it) and the MERL-val selection before the long run
        python src\ecgclip.py train --name "${RunName}_smoke" --dataset mimic --mimic_dir $Prep `
            --text report --text_model ncbi/MedCPT-Query-Encoder --ecg_arch resnet18 --lr_ecg 2e-4 `
            --bs $Batch --num_workers $Workers --epochs 1 --max_steps 200 --train_frac 0.02
    }
    'train' {
        python src\ecgclip.py train --name $RunName --dataset mimic --mimic_dir $Prep `
            --text report --text_model ncbi/MedCPT-Query-Encoder --ecg_arch resnet18 --lr_ecg 2e-4 `
            --bs $Batch --num_workers $Workers --epochs $Epochs --select_on merl --resume 1
    }
    'eval' {
        # full benchmark (zero-shot + 1/10/100 % linear probe on 6 tasks); PTB-XL is now OUT-of-domain too
        python src\merl_bench.py eval --model "ours:$($env:RUNS_DIR)\$RunName" --tag $RunName
        python src\merl_bench.py table
    }
}

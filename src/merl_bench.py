"""
MERL benchmark (Liu et al., ICML 2024) with MERL's own splits (merl_ref/finetune/data_split):
PTB-XL super / sub / form / rhythm, CSN (Chapman-Shaoxing-Ningbo) and CPSC2018.

For every model:  zero-shot macro AUROC (several prompt sets)  +  linear probe macro AUROC with 1 / 10 / 100 % of the
training split (same subsets as MERL: train_test_split(train_size=r, random_state=42)).

Usage:
  python src/merl_bench.py cache                       # build raw-signal caches once
  python src/merl_bench.py eval --model ours:<run_dir> [--tag NAME]
  python src/merl_bench.py eval --model merl:<res18_best_ckpt.pth>
  python src/merl_bench.py eval --model random
  python src/merl_bench.py table                       # markdown table of everything evaluated
"""
import argparse, ast, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecgclip as E  # noqa: E402

REPO_ROOT = E.REPO_ROOT
SPLITS = os.path.join(REPO_ROOT, 'merl_ref', 'finetune', 'data_split')
CKEPE = os.path.join(REPO_ROOT, 'merl_ref', 'zeroshot', 'CKEPE_prompt.json')
CSN_DIR = os.environ.get('CSN_DIR', os.path.join(REPO_ROOT, 'data', 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'))
CPSC_DIR = os.environ.get('CPSC_DIR', os.path.join(REPO_ROOT, 'data', 'cpsc_2018'))
BENCH_CACHE = os.environ.get('BENCH_CACHE', os.path.join(REPO_ROOT, 'cache', 'bench'))
OUT = os.environ.get('BENCH_OUT', os.path.join(REPO_ROOT, 'results', 'merl_bench'))
DEVICE = 'cuda'

TASKS = {  # name: (split prefix, first label column, dataset, valid samples at 500 Hz)
    'ptbxl_super': ('ptbxl/super_class/ptbxl_super_class', 6, 'ptbxl', 5000),
    'ptbxl_sub': ('ptbxl/sub_class/ptbxl_sub_class', 6, 'ptbxl', 5000),
    'ptbxl_form': ('ptbxl/form/ptbxl_form', 6, 'ptbxl', 5000),
    'ptbxl_rhythm': ('ptbxl/rhythm/ptbxl_rhythm', 6, 'ptbxl', 5000),
    'csn': ('chapman/chapman', 3, 'csn', 5000),
    'cpsc': ('icbeb/icbeb', 7, 'cpsc', 2500),   # MERL keeps the first 2,500 samples of CPSC (zero-padded to 5,000)
}
# MERL's paper, Table 1 (linear probe, %) - for reference rows in the table
MERL_PAPER_LP = {'ptbxl_super': (82.39, 86.27, 88.67), 'ptbxl_sub': (64.90, 80.56, 84.72), 'ptbxl_form': (58.26, 72.43, 79.65),
                 'ptbxl_rhythm': (53.33, 82.88, 88.34), 'cpsc': (70.33, 85.32, 90.57), 'csn': (66.60, 82.74, 87.95)}
RANDOM_PAPER_LP = {'ptbxl_super': (70.45, 77.09, 81.61), 'ptbxl_sub': (55.82, 67.60, 77.91), 'ptbxl_form': (55.82, 62.54, 73.00),
                   'ptbxl_rhythm': (46.26, 62.36, 79.29), 'cpsc': (54.96, 71.47, 78.33), 'csn': (47.22, 63.17, 73.13)}


# ───────────────────────────── data ─────────────────────────────
def split_csv(task, part):
    p, c0, ds, _ = TASKS[task]
    df = pd.read_csv(os.path.join(SPLITS, p + f'_{part}.csv'))
    return df, list(df.columns[c0:])


def build_caches():
    import wfdb
    os.makedirs(BENCH_CACHE, exist_ok=True)
    # PTB-XL: raw (mV) for every record, in ptbxl_database order (indexed by ecg_id)
    f = os.path.join(BENCH_CACHE, 'ptbxl_raw.npy')
    if not os.path.exists(f):
        df, _ = E.load_meta()
        arr = np.lib.format.open_memmap(f + '.tmp.npy', mode='w+', dtype=np.float16, shape=(len(df), 12, 5000))
        for i, fn in enumerate(df['filename_hr']):
            arr[i] = np.nan_to_num(wfdb.rdsamp(os.path.join(E.PTB_DIR, fn))[0].T[:, :5000])
            if i % 5000 == 0: print('ptbxl', i, flush=True)
        arr.flush(); del arr; os.replace(f + '.tmp.npy', f)
        np.save(os.path.join(BENCH_CACHE, 'ptbxl_ids.npy'), df.index.values)
    # CSN and CPSC: every record referenced by MERL's splits
    for ds, prefix in [('csn', 'chapman/chapman'), ('cpsc', 'icbeb/icbeb')]:
        f = os.path.join(BENCH_CACHE, f'{ds}_raw.npy')
        if os.path.exists(f):
            continue
        keys = pd.concat([pd.read_csv(os.path.join(SPLITS, prefix + f'_{p}.csv')) for p in ['train', 'val', 'test']])
        keys = keys['ecg_path' if ds == 'csn' else 'filename'].astype(str).drop_duplicates().tolist()
        if ds == 'cpsc':
            loc = {}
            for g in range(1, 8):
                gd = os.path.join(CPSC_DIR, f'g{g}')
                if os.path.isdir(gd):
                    for fn in os.listdir(gd):
                        if fn.endswith('.hea'):
                            loc[fn[:-4]] = os.path.join(gd, fn[:-4])
            missing = [k for k in keys if k not in loc]
            if missing:
                print(f'cpsc: {len(missing)} records missing, skipping CPSC cache', flush=True)
                continue
        arr = np.lib.format.open_memmap(f + '.tmp.npy', mode='w+', dtype=np.float16, shape=(len(keys), 12, 5000))
        for i, k in enumerate(keys):
            path = os.path.join(CSN_DIR, k.lstrip('/').replace('chapman/', '', 1))[:-4] if ds == 'csn' else loc[k]
            if ds == 'csn':   # read the .mat directly like MERL (some CSN headers have malformed dates that break wfdb)
                from scipy.io import loadmat
                x = np.nan_to_num(loadmat(path + '.mat')['val'].astype(np.float32))
            else:
                x = np.nan_to_num(wfdb.rdsamp(path)[0].T)
            if ds == 'cpsc':
                x = np.pad(x[:, :2500], ((0, 0), (0, 2500)))
            arr[i, :, :min(5000, x.shape[1])] = x[:, :5000]
            if i % 5000 == 0: print(ds, i, flush=True)
        arr.flush(); del arr; os.replace(f + '.tmp.npy', f)
        json.dump(keys, open(os.path.join(BENCH_CACHE, f'{ds}_keys.json'), 'w'))
    print('BENCH_CACHE_DONE', flush=True)


class TaskData:
    def __init__(self, task):
        self.task = task
        _, _, ds, self.valid = TASKS[task]
        self.parts = {}
        if ds == 'ptbxl':
            raw = np.load(os.path.join(BENCH_CACHE, 'ptbxl_raw.npy'), mmap_mode='r')
            pos = {int(e): i for i, e in enumerate(np.load(os.path.join(BENCH_CACHE, 'ptbxl_ids.npy')))}
            key = lambda df: [pos[int(e)] for e in df['ecg_id']]
        else:
            raw = np.load(os.path.join(BENCH_CACHE, f'{ds}_raw.npy'), mmap_mode='r')
            pos = {k: i for i, k in enumerate(json.load(open(os.path.join(BENCH_CACHE, f'{ds}_keys.json'))))}
            col = 'ecg_path' if ds == 'csn' else 'filename'
            key = lambda df: [pos[str(k)] for k in df[col]]
        for part in ['train', 'val', 'test']:
            df, labels = split_csv(task, part)
            self.labels = labels
            self.parts[part] = (key(df), df[labels].values.astype(np.float32))
        self.raw = raw


# ───────────────────────────── prompts ─────────────────────────────
def prompt_sets(task, labels):
    d = json.load(open(CKEPE, encoding='utf-8'))
    sets = {'ckepe': {l: [d[l].lower()] for l in labels},                          # MERL's protocol (one string / class)
            'ckepe_split': {l: [x.strip().lower() + '.' for x in d[l].split(',') if x.strip()] for l in labels}}
    if task.startswith('ptbxl'):
        _, scp = E.load_meta()
        norm = lambda c: c.strip('_')
        if task == 'ptbxl_super':
            g = scp[scp['diagnostic'] == 1].groupby('diagnostic_class')['description']
            sets['scp'] = {l: [x + '.' for x in g.get_group(l)] for l in labels}
        elif task == 'ptbxl_sub':
            dg = scp[scp['diagnostic'] == 1]
            sets['scp'] = {l: [x + '.' for x in dg[dg['diagnostic_subclass'].map(norm) == l]['description']] for l in labels}
        else:
            idx = {norm(c): c for c in scp.index}
            sets['scp'] = {l: [scp.loc[idx[l], 'description'] + '.'] for l in labels}
    return sets


# ───────────────────────────── models ─────────────────────────────
def zscore(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


class OursAdapter:
    """ECG-FM/ResNet CLIP model trained with ecgclip.py (z-score per lead, mean of 5-s windows)."""
    def __init__(self, run_dir=None, random_init=False, seed=0):
        from transformers import AutoTokenizer
        args = {'text_model': 'emilyalsentzer/Bio_ClinicalBERT', 'ecg_arch': 'fm', 'uma': 0}
        if run_dir:
            args.update(json.loads(open(os.path.join(run_dir, 'log.txt')).readline()[len('ARGS '):]))
        torch.manual_seed(seed)
        enc = E.ResNetECGEncoder() if args.get('ecg_arch') == 'resnet18' else E.ECGEncoder()
        self.model = E.ECGTextCLIP(enc, E.ClinicalTextEncoder(args['text_model']), uma=bool(args.get('uma', 0))).to(DEVICE).eval()
        if run_dir and not random_init:
            sd = torch.load(os.path.join(run_dir, 'best.pt'), map_location=DEVICE, weights_only=True)
            print(self.model.load_state_dict(sd, strict=True))
        self.tok = AutoTokenizer.from_pretrained(args['text_model'])
        self.zero_shot = not random_init

    @torch.no_grad()
    def embed(self, raw, idx, valid, bs=128):
        feats, projs = [], []
        for i in range(0, len(idx), bs):
            x = zscore(np.asarray(raw[np.sort(idx[i:i + bs])][np.argsort(np.argsort(idx[i:i + bs]))], dtype=np.float32)[:, :, :valid])
            x = torch.from_numpy(x).to(DEVICE)
            wins = [x[..., :2500], x[..., 2500:5000]] if valid >= 5000 else [x[..., :2500]]
            n = x.size(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = self.model.ecg_encoder(torch.cat(wins, 0)).float()
            p = self.model.ecg_proj(f)
            feats.append(sum(f[k * n:(k + 1) * n] for k in range(len(wins))).cpu() / len(wins))
            projs.append(F.normalize(sum(p[k * n:(k + 1) * n] for k in range(len(wins))), dim=-1).cpu())
        return torch.cat(feats).numpy(), torch.cat(projs)

    @torch.no_grad()
    def text(self, texts):
        t = self.tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return F.normalize(self.model.encode_text(**t).float(), dim=-1).cpu()


class MerlAdapter:
    """MERL released ResNet18 checkpoint, with MERL's own preprocessing (min-max per record, aVL/aVF swap, 5,000 samples)."""
    def __init__(self, ckpt):
        from transformers import AutoModel, AutoTokenizer
        from merl_resnet1d import ResNet18
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        self.enc = ResNet18().to(DEVICE).eval()
        self.enc.load_state_dict({k[len('ecg_encoder.'):]: v for k, v in sd.items() if k.startswith('ecg_encoder.')})
        self.downconv = nn.Conv1d(512, 256, 1).to(DEVICE).eval()
        self.downconv.load_state_dict({k[len('downconv.'):]: v for k, v in sd.items() if k.startswith('downconv.')})
        self.pos = sd['att_pool_head.positional_embedding'].to(DEVICE)
        self.cls = sd['att_pool_head.cls_token'].to(DEVICE)
        self.mhsa = nn.MultiheadAttention(256, 4, batch_first=True).to(DEVICE).eval()
        self.mhsa.load_state_dict({k[len('att_pool_head.mhsa.'):]: v for k, v in sd.items() if k.startswith('att_pool_head.mhsa.')})
        self.c_proj = nn.Linear(256, 256).to(DEVICE).eval()
        self.c_proj.load_state_dict({k[len('att_pool_head.c_proj.'):]: v for k, v in sd.items() if k.startswith('att_pool_head.c_proj.')})
        self.lm = AutoModel.from_pretrained('ncbi/MedCPT-Query-Encoder').to(DEVICE).eval()
        print('MedCPT', self.lm.load_state_dict({k[len('lm_model.'):]: v for k, v in sd.items() if k.startswith('lm_model.')}, strict=False))
        self.tok = AutoTokenizer.from_pretrained('ncbi/MedCPT-Query-Encoder')
        self.proj_t = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 256)).to(DEVICE).eval()
        self.proj_t.load_state_dict({k[len('proj_t.'):]: v for k, v in sd.items() if k.startswith('proj_t.')})
        self.zero_shot = True

    @torch.no_grad()
    def embed(self, raw, idx, valid, bs=128):
        feats, projs = [], []
        for i in range(0, len(idx), bs):
            x = np.asarray(raw[np.sort(idx[i:i + bs])][np.argsort(np.argsort(idx[i:i + bs]))], dtype=np.float32)
            mn = x.min(axis=(1, 2), keepdims=True); mx = x.max(axis=(1, 2), keepdims=True)
            x = (x - mn) / (mx - mn + 1e-8)
            x[:, [4, 5]] = x[:, [5, 4]]                   # MIMIC lead order (aVF, aVL)
            x = torch.from_numpy(x).to(DEVICE)
            h = self.enc(x)                               # [B, 512, 313]
            feats.append(h.mean(-1).cpu())                # MERL linear probe: avg-pooled 512-d feature
            z = self.downconv(h).permute(0, 2, 1)         # attention pooling head (as in MERL ECGCLIP.ext_ecg_emb)
            c = (self.cls + self.pos[:, :1]).expand(z.size(0), -1, -1)
            q = torch.cat([c, z], 1) + self.pos
            o, _ = self.mhsa(q[:, :1], q, q)
            projs.append(F.normalize(self.c_proj(o).squeeze(1), dim=-1).cpu())
        return torch.cat(feats).numpy(), torch.cat(projs)

    @torch.no_grad()
    def text(self, texts):
        t = self.tok([s.lower() for s in texts], return_tensors='pt', padding=True, truncation=True, max_length=256).to(DEVICE)
        return F.normalize(self.proj_t(self.lm(**t).pooler_output), dim=-1).cpu()


# ───────────────────────────── metrics ─────────────────────────────
def macro_auc(Y, S):
    aucs = [roc_auc_score(Y[:, k], S[:, k]) for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]
    return float(np.mean(aucs)) * 100


def linear_probe(Xtr, Ytr, Xva, Yva, Xte, Yte, steps=400):
    """Multi-label logistic regression on GPU (full batch, Adam); L2 strength chosen on the validation split."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    T = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(DEVICE)
    xtr, xva, xte = T(Xtr), T(Xva), T(Xte)
    ytr = torch.from_numpy(Ytr).to(DEVICE)
    best = None
    for wd in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        lin = nn.Linear(xtr.size(1), ytr.size(1)).to(DEVICE)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lin(xtr), ytr) + wd * lin.weight.pow(2).sum() / xtr.size(0)
            loss.backward(); opt.step()
        with torch.no_grad():
            va = macro_auc(Yva, lin(xva).cpu().numpy())
            te = macro_auc(Yte, lin(xte).cpu().numpy())
        if best is None or va > best[0]:
            best = (va, te, wd)
    return best


def evaluate(adapter, tag, tasks):
    os.makedirs(OUT, exist_ok=True)
    out_f = os.path.join(OUT, f'{tag}.json')
    res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    for task in tasks:
        if task in res:
            continue
        try:
            D = TaskData(task)
        except FileNotFoundError as ex:
            print(f'[{tag}] skip {task}: {ex}', flush=True); continue
        t0 = time.time()
        emb = {p: adapter.embed(D.raw, np.asarray(idx), D.valid) for p, (idx, _) in D.parts.items()}
        Y = {p: y for p, (_, y) in D.parts.items()}
        r = {'n_test': int(len(Y['test'])), 'n_classes': len(D.labels)}
        if adapter.zero_shot:
            for name, prompts in prompt_sets(task, D.labels).items():
                temb = {l: adapter.text(prompts[l]) for l in D.labels}
                for agg in (['max', 'mean'] if name != 'ckepe' else ['mean']):
                    def scores(P):
                        return torch.stack([(P @ temb[l].T).max(1).values if agg == 'max'
                                            else P @ F.normalize(temb[l].mean(0), dim=-1) for l in D.labels], 1).numpy()
                    r[f'zs_{name}_{agg}'] = {'val': macro_auc(Y['val'], scores(emb['val'][1])),
                                             'test': macro_auc(Y['test'], scores(emb['test'][1]))}
        Xtr = emb['train'][0]
        for ratio in [1, 10, 100]:
            sel = np.arange(len(Xtr)) if ratio == 100 else train_test_split(np.arange(len(Xtr)), train_size=ratio / 100, random_state=42)[0]
            va, te, wd = linear_probe(Xtr[sel], Y['train'][sel], emb['val'][0], Y['val'], emb['test'][0], Y['test'])
            r[f'lp_{ratio}'] = {'val': va, 'test': te, 'wd': wd}
        r['sec'] = round(time.time() - t0, 1)
        res[task] = r
        json.dump(res, open(out_f, 'w'), indent=1)
        zs = {k: round(v['test'], 2) for k, v in r.items() if k.startswith('zs_')}
        print(f'[{tag}] {task}: LP {r["lp_1"]["test"]:.2f}/{r["lp_10"]["test"]:.2f}/{r["lp_100"]["test"]:.2f} ZS {zs} ({r["sec"]}s)', flush=True)
        torch.cuda.empty_cache()
    print(f'BENCH_DONE {tag}', flush=True)


def table():
    order = list(TASKS)
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.json'):
            rows.append((f[:-5], json.load(open(os.path.join(OUT, f)))))
    lines = ['## Zero-shot macro AUROC (test). Prompt set chosen per model on validation, MERL-style CKEPE in brackets',
             '| model | ' + ' | '.join(order) + ' | mean |', '|---' * (len(order) + 2) + '|']
    for tag, r in rows:
        cells, vals = [], []
        for t in order:
            if t not in r or not any(k.startswith('zs_') for k in r[t]):
                cells.append('–'); continue
            zs = {k: v for k, v in r[t].items() if k.startswith('zs_')}
            best = max(zs, key=lambda k: zs[k]['val'])
            vals.append(zs[best]['test'])
            cells.append(f"{zs[best]['test']:.1f} [{zs['zs_ckepe_mean']['test']:.1f}]")
        lines.append(f'| {tag} | ' + ' | '.join(cells) + f" | {np.mean(vals):.1f} |" if vals else f'| {tag} | ' + ' | '.join(cells) + ' | – |')
    lines += ['', '## Linear probe macro AUROC (test), 1% / 10% / 100% of the training split',
              '| model | ' + ' | '.join(order) + ' |', '|---' * (len(order) + 1) + '|']
    for tag, r in rows:
        lines.append(f'| {tag} | ' + ' | '.join(
            (f"{r[t]['lp_1']['test']:.1f} / {r[t]['lp_10']['test']:.1f} / {r[t]['lp_100']['test']:.1f}" if t in r else '–') for t in order) + ' |')
    lines.append('| *MERL paper, Table 1* | ' + ' | '.join(' / '.join(f'{v:.1f}' for v in MERL_PAPER_LP[t]) for t in order) + ' |')
    lines.append('| *Random init, paper Table 1* | ' + ' | '.join(' / '.join(f'{v:.1f}' for v in RANDOM_PAPER_LP[t]) for t in order) + ' |')
    lines += ['', '## Averages over the tasks each model was evaluated on (n = number of tasks)',
              '| model | n | zero-shot (best prompt set on val) | zero-shot (CKEPE) | LP 1% | LP 10% | LP 100% |', '|---|---|---|---|---|---|---|']
    for tag, r in rows:
        ts = [t for t in order if t in r]
        zs = [r[t][max((k for k in r[t] if k.startswith('zs_')), key=lambda k: r[t][k]['val'])]['test']
              for t in ts if any(k.startswith('zs_') for k in r[t])]
        ck = [r[t]['zs_ckepe_mean']['test'] for t in ts if 'zs_ckepe_mean' in r[t]]
        f = lambda v: f'{np.mean(v):.1f}' if v else 'n/a'
        lines.append(f'| {tag} | {len(ts)} | {f(zs)} | {f(ck)} | ' + ' | '.join(f([r[t][f"lp_{k}"]["test"] for t in ts]) for k in (1, 10, 100)) + ' |')
    lines.append('| *MERL paper (zero-shot = CKEPE, 6 tasks)* | 6 | 75.2 | 75.2 | ' + ' | '.join(
        f'{np.mean([MERL_PAPER_LP[t][i] for t in order]):.1f}' for i in range(3)) + ' |')
    txt = '\n'.join(lines).replace('–', 'n/a')
    open(os.path.join(OUT, 'TABLE.md'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cache', 'eval', 'table'])
    ap.add_argument('--model', default='random')
    ap.add_argument('--tag', default='')
    ap.add_argument('--tasks', default=','.join(TASKS))
    a = ap.parse_args()
    if a.cmd == 'cache':
        build_caches()
    elif a.cmd == 'table':
        table()
    else:
        kind, _, path = a.model.partition(':')
        if kind == 'merl':
            ad = MerlAdapter(path)
        elif kind == 'ours':
            ad = OursAdapter(path)
        else:
            ad = OursAdapter(None, random_init=True)
        evaluate(ad, a.tag or (os.path.basename(path.rstrip('/\\')) if path else 'random_init'), a.tasks.split(','))

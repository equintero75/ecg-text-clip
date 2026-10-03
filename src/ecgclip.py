"""
ECG-Text CLIP experiment runner (mirrors ECG_Multimodal.ipynb, with fixes + ablation flags).

Usage:
  python ecgclip.py cache                    # build signal cache once
  python ecgclip.py train --name exp0 [flags]
"""
import argparse, ast, json, math, os, sys, time, random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTB_DIR = os.environ.get('PTB_DIR', os.path.join(REPO_ROOT, 'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'))
CACHE = os.environ.get('ECG_CACHE', os.path.join(REPO_ROOT, 'cache'))
RUNS = os.environ.get('RUNS_DIR', os.path.join(REPO_ROOT, 'runs'))
SUPERS = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
DEVICE = 'cuda'


# ───────────────────────────── data ─────────────────────────────
def load_meta():
    df = pd.read_csv(os.path.join(PTB_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(PTB_DIR, 'scp_statements.csv'), index_col=0)
    return df, scp


def build_cache():
    import wfdb
    os.makedirs(CACHE, exist_ok=True)
    df, _ = load_meta()
    n = len(df)
    arr = np.lib.format.open_memmap(os.path.join(CACHE, 'sig500.npy'), mode='w+',
                                    dtype=np.float16, shape=(n, 12, 5000))
    bad = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, fn in enumerate(df['filename_hr']):
        try:
            sig, meta = wfdb.rdsamp(os.path.join(PTB_DIR, fn))
        except Exception as ex:
            bad[i] = True
            continue
        x = sig.T.astype(np.float32)
        if not np.isfinite(x).all():
            bad[i] = True
            x = np.nan_to_num(x)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
        arr[i] = np.clip(x, -50, 50).astype(np.float16)
        if i % 2000 == 0:
            print(f'cache {i}/{n} {time.time()-t0:.0f}s', flush=True)
    arr.flush()
    np.save(os.path.join(CACHE, 'bad.npy'), bad)
    np.save(os.path.join(CACHE, 'ecg_ids.npy'), df.index.values)
    print('cache done', bad.sum(), 'non-finite records', flush=True)


def superclass_labels(df, scp):
    """Standard PTB-XL superdiagnostic multi-label targets (diagnostic statements only)."""
    diag = scp[scp['diagnostic'] == 1]
    code2cls = diag['diagnostic_class'].to_dict()
    Y = np.zeros((len(df), len(SUPERS)), dtype=np.float32)
    for i, codes in enumerate(df['scp_codes']):
        for c in codes:
            if c in code2cls:
                Y[i, SUPERS.index(code2cls[c])] = 1
    return Y


def build_texts(df, scp, version):
    if version == 'report':
        # cardiologist free-text report, cleaned and machine-translated to English (translate_reports.py)
        rep = pd.read_csv(os.environ.get('REPORTS_EN_CSV', os.path.join(REPO_ROOT, 'data', 'ptbxl_reports_en.csv')), index_col='ecg_id').reindex(df.index)
        return [t if isinstance(t, str) and t.strip() else 'electrocardiogram.' for t in rep['report_en']]
    desc = scp['description'].to_dict()
    is_diag = scp['diagnostic'].fillna(0).to_dict()
    is_form = scp['form'].fillna(0).to_dict()
    is_rhy = scp['rhythm'].fillna(0).to_dict()
    sexmap = {0: 'male', 1: 'female'}  # PTB-XL: 0 = male, 1 = female
    texts = []
    for _, row in df.iterrows():
        codes = row['scp_codes']
        if version == 'v0':
            # replica of the original notebook text builder
            sc = sorted(codes.items(), key=lambda x: x[1], reverse=True)
            f = [desc[c] for c, v in sc if v >= 50 and c in desc]
            parts = []
            if f:
                parts.append('. '.join(f) + '.')
            elif sc:
                parts.append('. '.join(desc.get(c, c) for c, _ in sc) + '.')
            demo = []
            if pd.notna(row['age']):
                demo.append(f"{int(row['age'])} year old")
            if pd.notna(row['sex']):
                demo.append({0: 'female', 1: 'male'}.get(int(row['sex']), 'unknown'))
            if demo:
                parts.append('Patient: ' + ' '.join(demo) + '.')
            texts.append(' '.join(parts) or 'Electrocardiogram recording.')
        else:
            # v1: every statement (diagnostic by likelihood, then form, then rhythm)
            d = sorted([c for c in codes if is_diag.get(c)], key=lambda c: -codes[c])
            fm = [c for c in codes if is_form.get(c) and not is_diag.get(c)]
            rh = [c for c in codes if is_rhy.get(c)]
            phrases = [desc[c] for c in d + fm + rh if c in desc]
            t = '. '.join(dict.fromkeys(phrases)) + '.' if phrases else 'electrocardiogram.'
            if version == 'v1demo':
                demo = []
                if pd.notna(row['age']) and row['age'] < 300:
                    demo.append(f"{int(row['age'])} year old")
                if pd.notna(row['sex']):
                    demo.append(sexmap.get(int(row['sex']), ''))
                t = t + ' ' + ' '.join(demo) + ' patient.'
            texts.append(t)
    return texts


class ECGSet(torch.utils.data.Dataset):
    def __init__(self, sig, idx, train):
        self.sig, self.idx, self.train = sig, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.sig[j]
        if self.train:
            s = random.randint(0, 2500)
            return torch.from_numpy(np.ascontiguousarray(x[:, s:s + 2500])).float(), i
        return torch.from_numpy(np.ascontiguousarray(x)).float(), i  # full 10 s; split at eval


# ───────────────────────────── model (same state_dict keys as notebook) ─────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, channels=256, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ECGFeatureExtractor(nn.Module):
    def __init__(self, in_channels=12, hidden=256):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, 2, 2)
        self.norm = nn.GroupNorm(1, hidden)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(ConvBlock(hidden), ConvBlock(hidden), ConvBlock(hidden))

    def forward(self, x):
        return self.blocks(self.act(self.norm(self.proj(x))))


class RelativePositionalConv(nn.Module):
    def __init__(self, embed_dim=768, pos_dim=128, kernel=128, groups=16):
        super().__init__()
        self.in_ = nn.Linear(embed_dim, pos_dim)
        self.conv = nn.Conv1d(pos_dim, pos_dim, kernel, padding=kernel // 2, groups=groups)
        self.act = nn.GELU()
        self.out_ = nn.Linear(pos_dim, embed_dim)

    def forward(self, x):
        p = self.in_(x).transpose(1, 2)
        p = self.conv(p)[..., :x.size(1)]
        p = self.act(p).transpose(1, 2)
        return x + self.out_(p)


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        return x + self.drop2(self.ff(self.norm2(x)))


class ECGEncoder(nn.Module):
    def __init__(self, in_ch=12, cnn_dim=256, embed_dim=768, heads=12, ff_dim=3072, n_layers=12, dropout=0.1):
        super().__init__()
        self.cnn = ECGFeatureExtractor(in_ch, cnn_dim)
        self.proj = nn.Linear(cnn_dim, embed_dim)
        self.pos = RelativePositionalConv(embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads, ff_dim, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.grad_ckpt = False
        self.out_dim = embed_dim

    def forward(self, x):
        z = self.cnn(x).transpose(1, 2)
        c = self.pos(self.proj(z))
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                c = torch.utils.checkpoint.checkpoint(layer, c, use_reentrant=False)
            else:
                c = layer(c)
        return self.norm(c).mean(1)


class ResNetECGEncoder(nn.Module):
    """MERL's 1-D ResNet18 (see merl_resnet1d.py) + global average pooling -> [B, 512]."""
    def __init__(self):
        super().__init__()
        from merl_resnet1d import ResNet18
        self.net = ResNet18()
        self.out_dim = 512
        self.grad_ckpt = False

    def forward(self, x):
        return self.net(x).mean(dim=-1)


class ClinicalTextEncoder(nn.Module):
    def __init__(self, name, freeze_layers=0):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(name)
        self.embed_dim = self.bert.config.hidden_size
        if freeze_layers > 0:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            for l in self.bert.encoder.layer[:freeze_layers]:
                for p in l.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ECGTextCLIP(nn.Module):
    def __init__(self, ecg_encoder, text_encoder, proj_dim=512, init_temp=0.07, uma=False):
        super().__init__()
        self.ecg_encoder, self.text_encoder = ecg_encoder, text_encoder
        self.ecg_proj = ProjectionHead(getattr(ecg_encoder, 'out_dim', 768), proj_dim)
        if uma:
            # MERL UMA (code version): two bias-free linear maps of the pooled ECG feature, each followed by dropout 0.1
            d = getattr(ecg_encoder, 'out_dim', 768)
            self.uma1 = nn.Linear(d, 256, bias=False)
            self.uma2 = nn.Linear(d, 256, bias=False)
            self.uma_drop = nn.Dropout(0.1)
        self.text_proj = ProjectionHead(text_encoder.embed_dim, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def encode_ecg(self, ecg):
        return self.ecg_proj(self.ecg_encoder(ecg))

    def encode_text(self, input_ids, attention_mask, **kwargs):
        return self.text_proj(self.text_encoder(input_ids, attention_mask))

    def uma_views(self, feat):
        return (F.normalize(self.uma_drop(self.uma1(feat)), dim=-1),
                F.normalize(self.uma_drop(self.uma2(feat)), dim=-1))


def clip_loss(e, t, temp, text_ids=None):
    logits = (e @ t.T) / temp
    if text_ids is None:
        lab = torch.arange(e.size(0), device=e.device)
        return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
    # multi-positive: samples that share the same report text are positives, not negatives
    pos = (text_ids[:, None] == text_ids[None, :]).float()
    tgt = pos / pos.sum(1, keepdim=True)
    l1 = -(tgt * F.log_softmax(logits, 1)).sum(1).mean()
    l2 = -(tgt * F.log_softmax(logits.T, 1)).sum(1).mean()
    return (l1 + l2) / 2


# ───────────────────────────── eval ─────────────────────────────
@torch.no_grad()
def embed_ecg(model, sig, idx, bs=128):
    """Deterministic: average of the two non-overlapping 5 s windows. Returns (encoder feats, projected)."""
    model.eval()
    feats, projs = [], []
    loader = torch.utils.data.DataLoader(ECGSet(sig, idx, train=False), batch_size=bs, shuffle=False, num_workers=0)
    for x, _ in loader:
        x = x.to(DEVICE)
        xs = torch.cat([x[..., :2500], x[..., 2500:]], 0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.ecg_encoder(xs).float()
        p = model.ecg_proj(f)
        n = x.size(0)
        feats.append(((f[:n] + f[n:]) / 2).cpu())
        projs.append(F.normalize(p[:n] + p[n:], dim=-1).cpu())
    return torch.cat(feats).numpy(), torch.cat(projs)


@torch.no_grad()
def embed_texts(model, tok, texts, bs=256):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model.encode_text(**t).float().cpu())
    return F.normalize(torch.cat(out), dim=-1)


ORIG_TEMPLATES = {
    'NORM': ['Normal sinus rhythm. No significant abnormalities.', 'Normal ECG. Regular heart rate and rhythm.'],
    'MI': ['Myocardial infarction. ST elevation with Q waves.', 'Acute myocardial infarction. Ischemic changes on ECG.'],
    'STTC': ['ST-T wave changes. Non-specific repolarization abnormality.', 'T-wave inversion. ST depression.'],
    'CD': ['Conduction disturbance. Bundle branch block.', 'Left bundle branch block. Prolonged QRS duration.'],
    'HYP': ['Ventricular hypertrophy. Left ventricular hypertrophy with high voltage.', 'Right ventricular hypertrophy. Axis deviation.'],
}


def scp_prompts(scp):
    diag = scp[scp['diagnostic'] == 1]
    return {s: [d + '.' for d in diag[diag['diagnostic_class'] == s]['description']] for s in SUPERS}


def ckepe_prompts(split=False, classes=SUPERS):
    """MERL's released CKEPE prompts (one comma-separated subtype list per class)."""
    d = json.load(open(os.path.join(REPO_ROOT, 'merl_ref', 'zeroshot', 'CKEPE_prompt.json'), encoding='utf-8'))
    if split:
        return {c: [x.strip().lower() + '.' for x in d[c].split(',') if x.strip()] for c in classes}
    return {c: [d[c].lower()] for c in classes}


def zero_shot_scores(model, tok, ecg_proj, prompts, agg='mean'):
    cols = []
    for s in SUPERS:
        te = embed_texts(model, tok, prompts[s])            # [k, D]
        sims = ecg_proj @ te.T                               # [N, k]
        if agg == 'mean':
            c = F.normalize(te.mean(0), dim=-1)
            cols.append(ecg_proj @ c)
        else:
            cols.append(sims.max(1).values)
    return torch.stack(cols, 1).numpy()


def macro_auc(Y, S):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(Y[:, k], S[:, k]) for k in range(Y.shape[1])]
    return float(np.mean(aucs)), [round(float(a), 4) for a in aucs]


def single_label_acc(Y, S):
    """Accuracy / macro-F1 on records with exactly one superclass (comparable to the original notebook metric)."""
    from sklearn.metrics import accuracy_score, f1_score
    m = Y.sum(1) == 1
    y = Y[m].argmax(1)
    p = S[m].argmax(1)
    return float(accuracy_score(y, p)), float(f1_score(y, p, average='macro')), int(m.sum())


def linear_probe(Xtr, Ytr, Xva, Yva, Xte, Yte, fracs=(0.01, 0.1, 1.0), seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    res = {}
    for frac in fracs:
        n = max(int(len(Xtr) * frac), 50)
        sel = rng.choice(len(Xtr), n, replace=False)
        sc = StandardScaler().fit(Xtr[sel])
        a, b, c = sc.transform(Xtr[sel]), sc.transform(Xva), sc.transform(Xte)
        best = None
        for C in [0.001, 0.01, 0.1, 1.0]:
            va, te = [], []
            for k in range(len(SUPERS)):
                if Ytr[sel, k].min() == Ytr[sel, k].max():
                    va.append(0.5); te.append(0.5); continue
                clf = LogisticRegression(C=C, max_iter=2000).fit(a, Ytr[sel, k])
                va.append(roc_auc_score(Yva[:, k], clf.decision_function(b)))
                te.append(roc_auc_score(Yte[:, k], clf.decision_function(c)))
            if best is None or np.mean(va) > best[0]:
                best = (np.mean(va), np.mean(te), C, te)
        res[str(frac)] = {'val_auc': round(float(best[0]), 4), 'test_auc': round(float(best[1]), 4),
                          'C': best[2], 'per_class': [round(float(x), 4) for x in best[3]]}
    return res


# ───────────────────────────── train ─────────────────────────────
def main_train(a):
    if a.exec_cmd:
        import subprocess
        subprocess.run(a.exec_cmd, shell=True, check=False, cwd=HERE)
        return
    if a.posthoc:
        import subprocess
        for run in a.posthoc.split(','):
            subprocess.run([sys.executable, os.path.join(HERE, 'posthoc_eval.py'), run], check=False)
        return
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    out_dir = os.path.join(RUNS, a.name); os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, 'log.txt'), 'a')

    def P(*s):
        msg = ' '.join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    P('ARGS', json.dumps(vars(a)))
    df, scp = load_meta()
    sig = np.load(os.path.join(CACHE, 'sig500.npy'), mmap_mode='r')
    if a.ram:
        sig = np.ascontiguousarray(sig)  # ~2.6 GB in RAM
    bad = np.load(os.path.join(CACHE, 'bad.npy'))
    Y = superclass_labels(df, scp)
    texts = build_texts(df, scp, a.text)
    text_ids = pd.factorize(pd.Series(texts))[0]
    P('unique texts', text_ids.max() + 1, '| example:', texts[0], '||', texts[10])

    fold = df['strat_fold'].values
    tr = np.where((fold <= 8) & ~bad)[0]
    va = np.where((fold == 9) & ~bad)[0]
    te = np.where((fold == 10) & ~bad)[0]
    if a.train_frac < 1:
        tr = np.random.RandomState(0).choice(tr, int(len(tr) * a.train_frac), replace=False)
    lab = Y.sum(1) > 0
    P(f'train {len(tr)} val {len(va)} test {len(te)} | label counts (all):', dict(zip(SUPERS, Y.sum(0).astype(int))))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.text_model)
    enc = tok(texts, padding=False, truncation=True, max_length=a.max_len)['input_ids']
    P('text token len: mean', np.mean([len(x) for x in enc]).round(1), 'max', max(len(x) for x in enc))

    ecg_enc = ResNetECGEncoder() if a.ecg_arch == 'resnet18' else ECGEncoder(n_layers=a.ecg_layers)
    ecg_enc.grad_ckpt = a.ckpt
    if a.ecg_init:
        sd = torch.load(a.ecg_init, map_location='cpu')
        P('loaded ecg init', ecg_enc.load_state_dict(sd, strict=False))
    txt_enc = ClinicalTextEncoder(a.text_model, freeze_layers=a.freeze_text_layers)
    if a.ckpt:
        txt_enc.bert.gradient_checkpointing_enable()
    if a.freeze_text:
        for p in txt_enc.parameters():
            p.requires_grad = False
    model = ECGTextCLIP(ecg_enc, txt_enc, init_temp=a.temp, uma=bool(a.uma)).to(DEVICE)
    if not a.learn_temp:
        model.log_temp.requires_grad = False
    P('trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    groups = [
        {'params': [p for p in model.ecg_encoder.parameters()], 'base_lr': a.lr_ecg},
        {'params': [p for p in model.text_encoder.parameters() if p.requires_grad], 'base_lr': a.lr_text},
        {'params': list(model.ecg_proj.parameters()) + list(model.text_proj.parameters())
                   + ([model.uma1.weight, model.uma2.weight] if a.uma else []), 'base_lr': a.lr_proj},
        {'params': [model.log_temp], 'base_lr': a.lr_temp, 'weight_decay': 0.0},
    ]
    groups = [g for g in groups if len(g['params']) > 0 and any(p.requires_grad for p in g['params'])]
    for g in groups:
        g['lr'] = g['base_lr']
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)

    loader = torch.utils.data.DataLoader(ECGSet(sig, tr, train=True), batch_size=a.bs, shuffle=True,
                                         num_workers=0, drop_last=True)
    steps_per_epoch = len(loader)
    total = a.epochs * steps_per_epoch
    warm = int(a.warmup * steps_per_epoch)

    def set_lr(step):
        if step < warm:
            s = step / max(1, warm)
        else:
            s = 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups:
            g['lr'] = g['base_lr'] * max(s, a.min_lr_frac)

    def batch_text(ii):
        ids = [enc[tr[i]] for i in ii]
        L = max(len(x) for x in ids)
        inp = torch.zeros(len(ids), L, dtype=torch.long)
        att = torch.zeros(len(ids), L, dtype=torch.long)
        for k, x in enumerate(ids):
            inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
        return inp.to(DEVICE), att.to(DEVICE)

    def evaluate(tag, full=False):
        t0 = time.time()
        r = {'tag': tag}
        # val contrastive loss (deterministic first window, per-text multi-positive and plain)
        fva, pva = embed_ecg(model, sig, va)
        tva = embed_texts(model, tok, [texts[i] for i in va])
        with torch.no_grad():
            vl = []
            for i in range(0, len(va) - a.bs + 1, a.bs):
                vl.append(clip_loss(pva[i:i + a.bs], tva[i:i + a.bs], model.temperature.detach().cpu()).item())
        r['val_clip_loss'] = round(float(np.mean(vl)), 4)
        # ECG->text retrieval over unique val texts: is the true report ranked top-1 / top-5?
        uniq, inv = np.unique([texts[i] for i in va], return_inverse=True)
        tu = embed_texts(model, tok, list(uniq))
        rank = (pva @ tu.T).argsort(1, descending=True)
        r['val_e2t_R@1'] = round(float((rank[:, 0].numpy() == inv).mean()), 4)
        r['val_e2t_R@5'] = round(float((rank[:, :5].numpy() == inv[:, None]).any(1).mean()), 4)
        # zero-shot on val (cheap) and test
        for split, idx, proj in [('val', va, pva)] + ([('test', te, None)] if full else []):
            if proj is None:
                fte, proj = embed_ecg(model, sig, idx)
            Ys = Y[idx]
            for pname, prompts in [('orig', ORIG_TEMPLATES), ('scp', scp_prompts(scp)),
                                   ('ckepe', ckepe_prompts(split=False)), ('ckepesplit', ckepe_prompts(split=True))]:
                for agg in (['mean', 'max'] if pname in ('scp', 'ckepesplit') else ['mean']):
                    S = zero_shot_scores(model, tok, proj, prompts, agg)
                    auc, per = macro_auc(Ys, S)
                    acc, f1, n = single_label_acc(Ys, S)
                    key = f'{split}_zs_{pname}_{agg}'
                    r[key + '_auc'] = round(auc, 4)
                    r[key + '_acc1'] = round(acc, 4)
                    r[key + '_f1'] = round(f1, 4)
                    if split == 'test':
                        r[key + '_per_class_auc'] = per
        if full:
            ftr, _ = embed_ecg(model, sig, tr if len(tr) <= 20000 else tr[:20000])
            r['linear_probe'] = linear_probe(ftr, Y[tr], fva, Y[va], fte, Y[te])
        r['eval_sec'] = round(time.time() - t0, 1)
        torch.cuda.empty_cache()
        P('EVAL', json.dumps(r))
        with open(os.path.join(out_dir, 'results.jsonl'), 'a') as f:
            f.write(json.dumps(r) + '\n')
        model.train()
        return r

    if a.eval_init:
        evaluate('init', full=True)

    step = 0
    best = -1
    t_start = time.time()
    for ep in range(a.epochs):
        model.train()
        t0 = time.time(); losses = []; skipped = 0
        for x, ii in loader:
            set_lr(step)
            x = x.to(DEVICE, non_blocking=True)
            inp, att = batch_text(ii.numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat = model.ecg_encoder(x)
                e = model.ecg_proj(feat)
                t = model.encode_text(inp, att)
            e, t = e.float(), t.float()
            tid = torch.from_numpy(text_ids[tr[ii.numpy()]]).to(DEVICE) if a.multipos else None
            loss = clip_loss(e, t, model.temperature, tid)
            if a.uma:
                u1, u2 = model.uma_views(feat.float())
                loss = loss + a.uma_w * clip_loss(u1, u2, torch.tensor(0.07, device=DEVICE))
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            losses.append(loss.item()); step += 1
            if step % 50 == 0:
                P(f'ep {ep+1} step {step}/{total} loss {np.mean(losses[-50:]):.4f} temp {model.temperature.item():.4f} '
                  f'gn {gn.item():.2f} lr_ecg {opt.param_groups[0]["lr"]:.2e} {(time.time()-t0)/len(losses):.2f}s/it '
                  f'mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
            if a.max_steps and step >= a.max_steps:
                break
        P(f'EPOCH {ep+1} train_loss {np.mean(losses):.4f} skipped {skipped} time {(time.time()-t0)/60:.1f}min')
        full = (ep + 1 == a.epochs) or (a.full_every and (ep + 1) % a.full_every == 0)
        r = evaluate(f'ep{ep+1}', full=full)
        score = r['val_zs_scp_max_auc']
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
        if a.max_steps and step >= a.max_steps:
            break
    P(f'DONE total {(time.time()-t_start)/60:.1f}min best val_zs_scp_mean_auc {best:.4f}')
    torch.save(model.ecg_encoder.state_dict(), os.path.join(out_dir, 'ecg_encoder_last.pt'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--name', default='dbg')
    ap.add_argument('--text', default='v1', choices=['v0', 'v1', 'v1demo', 'report'])
    ap.add_argument('--text_model', default='emilyalsentzer/Bio_ClinicalBERT')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr_ecg', type=float, default=1e-4)
    ap.add_argument('--lr_text', type=float, default=2e-5)
    ap.add_argument('--lr_proj', type=float, default=2e-4)
    ap.add_argument('--lr_temp', type=float, default=1e-3)
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--learn_temp', type=int, default=1)
    ap.add_argument('--multipos', type=int, default=1)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warmup', type=float, default=1.0)
    ap.add_argument('--min_lr_frac', type=float, default=0.02)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--freeze_text', type=int, default=0)
    ap.add_argument('--freeze_text_layers', type=int, default=0)
    ap.add_argument('--ecg_layers', type=int, default=12)
    ap.add_argument('--ecg_init', default='')
    ap.add_argument('--ckpt', type=int, default=0)
    ap.add_argument('--ram', type=int, default=1)
    ap.add_argument('--train_frac', type=float, default=1.0)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--full_every', type=int, default=0)
    ap.add_argument('--eval_init', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--posthoc', default='')
    ap.add_argument('--ecg_arch', default='fm', choices=['fm', 'resnet18'])
    ap.add_argument('--uma', type=int, default=0)
    ap.add_argument('--uma_w', type=float, default=1.0)
    ap.add_argument('--exec_cmd', default='')
    a = ap.parse_args()
    if a.cmd == 'cache':
        build_cache()
    else:
        main_train(a)

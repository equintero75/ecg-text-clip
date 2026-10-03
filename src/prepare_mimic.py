"""
Prepare MIMIC-IV-ECG for ECG-text pre-training.

  python src/prepare_mimic.py --src D:/mimic/mimic-iv-ecg-...-1.0.zip --out D:/mimic_prepared            # lab (reads the zip)
  python src/prepare_mimic.py --src data/mimic_subset --out data/mimic_subset_prepared \
         --records data/mimic_subset/record_list_subset.csv                                                # local test

Outputs (in --out):
  sig.npy    float16 [N, 12, 5000]: 10 s at 500 Hz, leads in PTB-XL order (I, II, III, aVR, aVL, aVF, V1-V6),
             z-scored per lead -- the same preprocessing as the PTB-XL cache, so models transfer directly.
             N ~ 772k on the full dataset -> ~93 GB (read as a memory map during training, never loaded into RAM).
  meta.csv   idx, subject_id, study_id, split (train/val, by SUBJECT), bad, nan_frac, text
  stats.json counts and the exclusion breakdown

Text = MIMIC machine statements report_0..report_17 joined with '. ', lower-cased (MERL's cleaning);
studies with an empty report or fewer than 3 words are excluded (as in MERL).
Signals: MIMIC stores the leads as I, II, III, aVR, aVF, aVL, ... -> reordered by NAME from each header.
Missing samples (-32768) are linearly interpolated; records with > 10 % missing or an all-flat signal are flagged bad.
"""
import argparse, json, os, re, sys, time, zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
N_SAMP = 5000
_zip, _prefix, _src, _out = None, '', '', ''


def clean_report(row, cols):
    parts = [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()]
    r = '. '.join(parts)
    r = r.replace('EKG', 'ECG').replace('ekg', 'ecg')
    r = r.strip('*** ').strip(' ***').strip('***').strip('=-').strip('=')
    r = re.sub(r'\s+', ' ', r.replace('\n', ' ')).strip().lower()
    r = re.sub(r'\.(\s*\.)+', '.', r)          # "sinus rhythm.. normal ecg." -> "sinus rhythm. normal ecg."
    if r and not r.endswith('.'):
        r += '.'
    return r


def _init(src, prefix, out):
    global _zip, _prefix, _src, _out
    _src, _prefix, _out = src, prefix, out
    _zip = zipfile.ZipFile(src) if src.lower().endswith('.zip') else None


def _read(rel):
    if _zip is not None:
        return _zip.read(_prefix + rel)
    with open(os.path.join(_src, rel.replace('/', os.sep)), 'rb') as f:
        return f.read()


def parse_record(path):
    """Returns (signal [12, 5000] float32 in mV, PTB-XL lead order, NaN where missing)."""
    hea = _read(path + '.hea').decode().splitlines()
    n_sig, fs, n_samp = int(hea[0].split()[1]), int(float(hea[0].split()[2])), int(hea[0].split()[3])
    assert fs == 500, f'fs={fs}'
    names, gains, bases = [], [], []
    for line in hea[1:1 + n_sig]:
        tok = line.split()
        assert tok[1] == '16', f'format {tok[1]}'
        g = tok[2].split('/')[0]
        base = 0
        if '(' in g:
            g, base = g.split('(')[0], int(g.split('(')[1].rstrip(')'))
        names.append(tok[-1]); gains.append(float(g) or 200.0); bases.append(base)
    raw = np.frombuffer(_read(path + '.dat'), dtype='<i2')[:n_samp * n_sig].reshape(n_samp, n_sig)
    x = (raw.astype(np.float32) - np.array(bases, np.float32)) / np.array(gains, np.float32)
    x[raw == -32768] = np.nan
    pos = {n.upper(): i for i, n in enumerate(names)}
    out = np.full((12, N_SAMP), np.nan, np.float32)
    for j, lead in enumerate(LEADS):
        if lead.upper() in pos:
            sig = x[:N_SAMP, pos[lead.upper()]]
            out[j, :len(sig)] = sig
    return out


def preprocess(x):
    """Interpolate missing samples, z-score per lead, clip; returns (float16 array, nan_frac, bad)."""
    nan = ~np.isfinite(x)
    nan_frac = float(nan.mean())
    if nan_frac > 0.10:
        return None, nan_frac, True
    if nan.any():
        t = np.arange(x.shape[1])
        for j in range(12):
            m = nan[j]
            if m.all():
                x[j] = 0.0
            elif m.any():
                x[j, m] = np.interp(t[m], t[~m], x[j, ~m])
    sd = x.std(1, keepdims=True)
    if float(sd.max()) < 1e-6:
        return None, nan_frac, True
    x = (x - x.mean(1, keepdims=True)) / (sd + 1e-8)
    return np.clip(x, -50, 50).astype(np.float16), nan_frac, False


def _work(chunk):
    sig = np.load(os.path.join(_out, 'sig.npy'), mmap_mode='r+')
    res = []
    for i, path in chunk:
        try:
            y, nf, bad = preprocess(parse_record(path))
            if y is not None:
                sig[i] = y
            res.append((i, nf, bad))
        except Exception as e:
            res.append((i, 1.0, True))
    sig.flush()
    return res


def main(a):
    os.makedirs(a.out, exist_ok=True)
    meta_dir = a.meta or (os.path.dirname(a.src) if a.src.lower().endswith('.zip') else a.src)
    prefix = ''
    if a.src.lower().endswith('.zip'):
        with zipfile.ZipFile(a.src) as z:
            names = z.namelist()
            rl_name = [n for n in names if n.endswith('record_list.csv')][0]
            prefix = rl_name[:-len('record_list.csv')]
            if not os.path.exists(os.path.join(meta_dir, 'record_list.csv')):
                for f in ['record_list.csv', 'machine_measurements.csv']:
                    open(os.path.join(a.out, f), 'wb').write(z.read(prefix + f))
                meta_dir = a.out
        print('zip prefix:', prefix, flush=True)

    rl = pd.read_csv(a.records or os.path.join(meta_dir, 'record_list.csv'))
    mm = pd.read_csv(os.path.join(meta_dir, 'machine_measurements.csv'), low_memory=False)
    cols = [f'report_{k}' for k in range(18) if f'report_{k}' in mm.columns]
    mm['text'] = mm.apply(lambda r: clean_report(r, cols), axis=1)
    df = rl.merge(mm[['study_id', 'text']], on='study_id', how='left')
    n0 = len(df)
    df['text'] = df['text'].fillna('')
    df = df[df['text'].str.split().str.len() >= 3].reset_index(drop=True)
    if a.limit:
        df = df.iloc[:a.limit]
    subj = df['subject_id'].drop_duplicates().sample(frac=1.0, random_state=0).values
    val_subj = set(subj[:max(1, int(len(subj) * a.val_frac))])
    df['split'] = np.where(df['subject_id'].isin(val_subj), 'val', 'train')
    df['idx'] = np.arange(len(df))
    print(f'{n0} records -> {len(df)} with a report of >= 3 words | {len(subj)} subjects, {len(val_subj)} in val', flush=True)

    sig_path = os.path.join(a.out, 'sig.npy')
    if not os.path.exists(sig_path) or a.overwrite:
        np.lib.format.open_memmap(sig_path, mode='w+', dtype=np.float16, shape=(len(df), 12, N_SAMP)).flush()
    jobs = list(zip(df['idx'].tolist(), df['path'].tolist()))
    chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
    nan_frac = np.zeros(len(df), np.float32); bad = np.zeros(len(df), bool)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.src, prefix, a.out)) as pool:
        for k, res in enumerate(pool.imap_unordered(_work, chunks)):
            for i, nf, b in res:
                nan_frac[i], bad[i] = nf, b
            if k % 20 == 0:
                done = (k + 1) * 500
                rate = done / max(time.time() - t0, 1e-6)
                print(f'  {min(done, len(df))}/{len(df)} records | {rate:.0f}/s | ETA {(len(df) - done) / max(rate, 1e-6) / 60:.1f} min', flush=True)
    df['nan_frac'] = nan_frac; df['bad'] = bad
    df[['idx', 'subject_id', 'study_id', 'split', 'bad', 'nan_frac', 'text']].to_csv(os.path.join(a.out, 'meta.csv'), index=False)
    stats = {'records_in_record_list': int(n0), 'records_with_report': int(len(df)), 'bad_signals': int(bad.sum()),
             'train': int(((df.split == 'train') & ~df.bad).sum()), 'val': int(((df.split == 'val') & ~df.bad).sum()),
             'unique_texts': int(df['text'].nunique()), 'minutes': round((time.time() - t0) / 60, 1)}
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)
    print('PREPARE_DONE', json.dumps(stats), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='MIMIC-IV-ECG zip, or a folder containing files/ (and the csvs)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--meta', default='', help='folder with record_list.csv + machine_measurements.csv (default: next to --src)')
    ap.add_argument('--records', default='', help='optional subset of record_list.csv')
    ap.add_argument('--val_frac', type=float, default=0.02)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', type=int, default=0)
    main(ap.parse_args())

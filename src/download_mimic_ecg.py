"""
Download MIMIC-IV-ECG v1.0 (open access, ODbL) from PhysioNet.

  python src/download_mimic_ecg.py zip    --out D:/mimic          # full 36.3 GB zip, parallel resumable byte ranges (lab)
  python src/download_mimic_ecg.py subset --out data/mimic_subset --n 3000   # metadata + N records (testing)

The zip is downloaded in fixed-size chunks with many connections (PhysioNet serves ~0.1-0.5 MB/s per connection) and can
be resumed: re-run the same command and only missing chunks are fetched. prepare_mimic.py reads records directly from the
zip, so it does not need to be extracted (saves ~90 GB).
"""
import argparse, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://physionet.org/files/mimic-iv-ecg/1.0/'
ZIP_URL = 'https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/'
ZIP_NAME = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'
UA = {'User-Agent': 'Wget/1.21'}


def fetch(url, rng=None, timeout=120):
    h = dict(UA)
    if rng:
        h['Range'] = f'bytes={rng[0]}-{rng[1]}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def remote_size(url):
    req = urllib.request.Request(url, headers=dict(UA, Range='bytes=0-0'))
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Range'].split('/')[-1])


def ranged_download(url, dst, workers=24, chunk=64 << 20):
    """Parallel, resumable download of one large file into dst (chunks kept in dst.parts/ until complete)."""
    size = remote_size(url)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f'{os.path.basename(dst)} already complete ({size/1e9:.2f} GB)', flush=True)
        return
    parts = dst + '.parts'
    os.makedirs(parts, exist_ok=True)
    jobs = [(i, s, min(s + chunk, size) - 1) for i, s in enumerate(range(0, size, chunk))]

    def get(job):
        i, a, b = job
        p = os.path.join(parts, f'{i:06d}')
        if os.path.exists(p) and os.path.getsize(p) == b - a + 1:
            return 0
        for attempt in range(8):
            try:
                data = fetch(url, (a, b), timeout=300)
                if len(data) != b - a + 1:
                    raise IOError(f'short read {len(data)}')
                open(p + '.tmp', 'wb').write(data)
                os.replace(p + '.tmp', p)
                return len(data)
            except Exception as e:
                time.sleep(5 * (attempt + 1))
        print(f'FAILED chunk {i} (re-run to resume)', flush=True)
        return 0

    t0, done = time.time(), 0
    print(f'{os.path.basename(dst)}: {size/1e9:.2f} GB in {len(jobs)} chunks, {workers} connections', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for k, n in enumerate(ex.map(get, jobs)):
            done += n
            if k % 20 == 0:
                el = time.time() - t0
                print(f'  {k+1}/{len(jobs)} chunks | {done/1e9:.2f} GB this session | {done/1e6/max(el,1):.1f} MB/s', flush=True)
    missing = [j for j in jobs if not os.path.exists(os.path.join(parts, f'{j[0]:06d}'))]
    if missing:
        print(f'INCOMPLETE: {len(missing)} chunks missing - re-run the same command to resume', flush=True)
        sys.exit(1)
    with open(dst + '.tmp', 'wb') as out:
        for i, _, _ in jobs:
            out.write(open(os.path.join(parts, f'{i:06d}'), 'rb').read())
    os.replace(dst + '.tmp', dst)
    for i, _, _ in jobs:
        os.remove(os.path.join(parts, f'{i:06d}'))
    os.rmdir(parts)
    print(f'DONE {dst} ({size/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min', flush=True)


def download_subset(out, n, workers):
    os.makedirs(out, exist_ok=True)
    for f in ['record_list.csv', 'machine_measurements.csv', 'machine_measurements_data_dictionary.csv', 'LICENSE.txt']:
        ranged_download(BASE + f, os.path.join(out, f), workers=8, chunk=8 << 20)
    import pandas as pd
    rl = pd.read_csv(os.path.join(out, 'record_list.csv'))
    # deterministic sample of whole subjects (keeps all of a subject's ECGs together)
    subj = rl['subject_id'].drop_duplicates().sample(frac=1.0, random_state=0)
    keep, total = [], 0
    for s in subj:
        k = rl.index[rl['subject_id'] == s] if total < n else []
        keep += list(k); total += len(k)
        if total >= n:
            break
    rl = rl.loc[keep]
    rl.to_csv(os.path.join(out, 'record_list_subset.csv'), index=False)
    files = [p + ext for p in rl['path'] for ext in ('.hea', '.dat')]
    print(f'subset: {len(rl)} records from {rl.subject_id.nunique()} subjects -> {len(files)} files', flush=True)

    def get(rel):
        dst = os.path.join(out, rel.replace('/', os.sep))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(6):
            try:
                data = fetch(BASE + rel)
                open(dst + '.tmp', 'wb').write(data); os.replace(dst + '.tmp', dst)
                return len(data)
            except Exception:
                time.sleep(3 * (attempt + 1))
        print('FAILED', rel, flush=True)
        return 0

    t0, tot = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for i, b in enumerate(ex.map(get, files)):
            tot += b
            if i % 1000 == 0:
                print(f'  {i}/{len(files)} files {tot/1e6:.0f} MB {tot/1e6/max(1, time.time()-t0):.2f} MB/s', flush=True)
    print(f'SUBSET_DONE {len(rl)} records in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['zip', 'subset'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    if a.mode == 'zip':
        os.makedirs(a.out, exist_ok=True)
        ranged_download(ZIP_URL, os.path.join(a.out, ZIP_NAME), workers=a.workers)
    else:
        download_subset(a.out, a.n, a.workers)

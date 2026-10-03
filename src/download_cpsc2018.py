"""Parallel download of CPSC2018 (PhysioNet/CinC Challenge 2021 copy, 6,877 records, 500 Hz) used by MERL's benchmark."""
import os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://physionet.org/files/challenge-2021/1.0.3/training/cpsc_2018/'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.environ.get('CPSC_DIR', os.path.join(REPO_ROOT, 'data', 'cpsc_2018'))


def get(rel):
    path = os.path.join(DST, rel.replace('/', os.sep))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(6):
        try:
            try:
                with urllib.request.urlopen(BASE + rel, timeout=90) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                # PhysioNet's RECORDS lists A1000, A2000, ... in g<N> but the files live in g<N+1>
                if e.code != 404:
                    raise
                g, name = rel.split('/')
                with urllib.request.urlopen(f'{BASE}g{int(g[1:]) + 1}/{name}', timeout=90) as r:
                    data = r.read()
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
            return len(data)
        except Exception:
            time.sleep(3 * (attempt + 1))
    print('FAILED', rel, flush=True)
    return 0


files = []
for g in range(1, 8):
    rec = urllib.request.urlopen(f'{BASE}g{g}/RECORDS', timeout=60).read().decode().split()
    files += [f'g{g}/{r}.hea' for r in rec] + [f'g{g}/{r}.mat' for r in rec]
print('files', len(files), flush=True)
t0, total = time.time(), 0
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 24) as ex:
    for i, n in enumerate(ex.map(get, files)):
        total += n
        if i % 1000 == 0:
            print(f'{i}/{len(files)} {total/1e6:.0f}MB {total/1e6/max(1, time.time()-t0):.2f}MB/s', flush=True)
print(f'CPSC_DONE {total/1e6:.0f}MB {(time.time()-t0)/60:.1f}min', flush=True)

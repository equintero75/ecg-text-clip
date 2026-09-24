"""Parallel download of PTB-XL 1.0.3 (500 Hz records + metadata) from PhysioNet's AWS open-data mirror."""
import csv, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
DST = os.environ.get('PTB_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'))


def get(rel):
    path = os.path.join(DST, rel.replace('/', os.sep))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                data = r.read()
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
            return len(data)
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    print('FAILED', rel, flush=True)
    return 0


for meta in ['ptbxl_database.csv', 'scp_statements.csv', 'LICENSE.txt', 'SHA256SUMS.txt']:
    get(meta)

with open(os.path.join(DST, 'ptbxl_database.csv'), newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
files = []
for r in rows:
    files += [r['filename_hr'] + '.hea', r['filename_hr'] + '.dat']
print('files to fetch', len(files), flush=True)

t0 = time.time(); total = 0
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 48) as ex:
    for i, n in enumerate(ex.map(get, files)):
        total += n
        if i % 4000 == 0:
            print(f'{i}/{len(files)} {total/1e6:.0f}MB {total/1e6/max(1,time.time()-t0):.1f}MB/s', flush=True)
print(f'DL_DONE {total/1e6:.0f}MB in {(time.time()-t0)/60:.1f}min', flush=True)

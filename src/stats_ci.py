"""
Bootstrap confidence intervals and paired tests for the MERL-benchmark results.

Reads the per-ECG test predictions saved by merl_bench.py (results/merl_bench/preds/<tag>/<task>.npz).
Runs that differ only by seed are grouped automatically: tag "S_A6_s0", "S_A6_s1", ... -> group "S_A6".

  python src/stats_ci.py                       # all groups, default comparisons, B = 2000
  python src/stats_ci.py --compare S_A6:MERL_released_res18 S_A6:S_A7 --B 5000

Method (test-set bootstrap, paired, seed-averaged):
  * For every task, B resamples of the test ECGs are drawn ONCE and shared by all models (-> paired comparisons).
  * For each resample, macro AUROC is computed for every seed of a configuration and averaged over seeds.
  * Point estimate = seed-mean AUROC on the full test set; 95 % CI = 2.5 / 97.5 percentiles over resamples.
  * Task averages (all 6 tasks; out-of-domain = CSN + CPSC) average the per-task replicates.
  * Paired difference A - B: percentiles of the replicate differences; two-sided p = 2 * min(P(diff <= 0), P(diff >= 0)).
  * Seed SD = standard deviation across seeds of the full-test AUROC (training randomness, reported separately).
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('BENCH_OUT', os.path.join(os.path.dirname(HERE), 'results', 'merl_bench'))
TASKS = ['ptbxl_super', 'ptbxl_sub', 'ptbxl_form', 'ptbxl_rhythm', 'csn', 'cpsc']
METRICS = ['zs_best', 'zs_ckepe', 'lp_1', 'lp_10', 'lp_100']
AGGS = {'mean6': TASKS, 'ood': ['csn', 'cpsc'], 'ptbxl': TASKS[:4]}


def macro_auc(y, s):
    """Macro AUROC over classes with both labels present (rank formula; ties only between identical samples)."""
    n = len(y)
    order = np.argsort(s, axis=0)                     # one sort; ranks scattered back (no second argsort)
    r = np.empty(s.shape, dtype=np.float64)
    np.put_along_axis(r, order, np.arange(1, n + 1, dtype=np.float64)[:, None], axis=0)
    P = y.sum(0).astype(np.float64)
    N = n - P
    ok = (P > 0) & (N > 0)
    auc = ((r * y).sum(0) - P * (P + 1) / 2) / np.where(ok, P * N, 1.0)
    return float(auc[ok].mean()) * 100 if ok.any() else np.nan


def load():
    runs = {}
    for d in sorted(glob.glob(os.path.join(OUT, 'preds', '*'))):
        tag = os.path.basename(d)
        runs[tag] = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in glob.glob(os.path.join(d, '*.npz'))}
    groups = {}
    for tag in runs:
        groups.setdefault(re.sub(r'_s\d+$', '', tag), []).append(tag)
    return runs, groups


def _job(args):
    """Full-test AUROC and B bootstrap AUROCs for one (run, task, metric). Resamples depend only on (seed, task),
    so every model sees exactly the same resamples -> paired comparisons."""
    key, y, s, B, seed, task = args
    n = len(y)
    idx = np.random.RandomState(seed + TASKS.index(task)).randint(0, n, size=(B, n))
    return key, macro_auc(y, s), np.array([macro_auc(y[b], s[b]) for b in idx], dtype=np.float32)


def main(a):
    sys.stdout.reconfigure(encoding="utf-8")
    from multiprocessing import Pool
    runs, groups = load()
    jobs = [((tag, t, m), p['y'], p[m], a.B, a.seed, t) for tag in runs for t, p in runs[tag].items() for m in METRICS if m in p]
    res = {}
    with Pool(a.workers) as pool:
        for k, (key, full, reps) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            res[key] = (full, reps)
            if k % 25 == 0:
                print(f'  {k + 1}/{len(jobs)} bootstrap jobs', flush=True)

    point, rep, seed_sd = {}, {}, {}
    for g, tags in sorted(groups.items()):
        point[g], rep[g], seed_sd[g] = {}, {}, {}
        for m in METRICS:
            point[g][m], rep[g][m], per_seed = {}, {}, {}
            for t in TASKS:
                if not all((tg, t, m) in res for tg in tags):     # every seed must have this task/metric
                    continue
                full = [res[(tg, t, m)][0] for tg in tags]
                point[g][m][t] = float(np.mean(full))
                rep[g][m][t] = np.nanmean(np.stack([res[(tg, t, m)][1] for tg in tags]), axis=0)
                per_seed[t] = full
            for agg, ts in AGGS.items():
                if all(t in point[g][m] for t in ts):
                    point[g][m][agg] = float(np.mean([point[g][m][t] for t in ts]))
                    rep[g][m][agg] = np.mean([rep[g][m][t] for t in ts], axis=0)
                    seeds = np.array([[per_seed[t][k] for t in ts] for k in range(len(tags))]).mean(1)
                    seed_sd[g][f'{m}:{agg}'] = float(seeds.std(ddof=1)) if len(tags) > 1 else np.nan

    def ci(x):
        return np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)

    lines = [f'# Bootstrap confidence intervals ({a.B} test-set resamples, seed-averaged)', '',
             'Macro AUROC (%), point estimate [95 % CI]. *n* = seeds. Seed SD = spread across seeds of the 6-task mean.', '']
    for m, title in [('zs_best', 'Zero-shot (prompt set chosen on validation)'), ('zs_ckepe', 'Zero-shot (MERL CKEPE prompts)'),
                     ('lp_1', 'Linear probe, 1 % labels'), ('lp_10', 'Linear probe, 10 % labels'), ('lp_100', 'Linear probe, 100 % labels')]:
        gs = [g for g in point if 'mean6' in point[g][m]]
        if not gs:
            continue
        lines += [f'## {title}', '', '| config | n | ' + ' | '.join(TASKS) + ' | **6-task mean** | out-of-domain (CSN+CPSC) | seed SD |',
                  '|---' * (len(TASKS) + 5) + '|']
        for g in sorted(gs, key=lambda g: -point[g][m]['mean6']):
            cell = lambda k: f'{point[g][m][k]:.1f} [{ci(rep[g][m][k])[0]:.1f}, {ci(rep[g][m][k])[1]:.1f}]'
            sd = seed_sd[g].get(f'{m}:mean6', np.nan)
            lines.append(f'| {g} | {len(groups[g])} | ' + ' | '.join(f'{point[g][m][t]:.1f}' for t in TASKS)
                         + f' | **{cell("mean6")}** | {cell("ood")} | {"" if np.isnan(sd) else f"{sd:.2f}"} |')
        lines.append('')

    comps = a.compare or [c for c in ['S_A6:MERL_released_res18', 'S_A6:S_A7', 'S_A6:S_A5', 'S_A6:S_A1', 'S_A1:S_A0',
                                      'S_A6:random_init_resnet18', 'S_A6:S_A0'] if all(x in point for x in c.split(':'))]
    if comps:
        lines += ['## Paired comparisons (A − B, same bootstrap resamples)', '',
                  '| A − B | metric | 6-task mean: diff [95 % CI], p | out-of-domain: diff [95 % CI], p |', '|---|---|---|---|']
        for c in comps:
            A, B = c.split(':')
            for m in METRICS:
                cells = []
                for agg in ['mean6', 'ood']:
                    if agg in point[A][m] and agg in point[B][m]:
                        d = rep[A][m][agg] - rep[B][m][agg]
                        p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))
                        lo, hi = ci(d)
                        star = ' **' if p < 0.05 else ''
                        cells.append(f'{point[A][m][agg] - point[B][m][agg]:+.1f} [{lo:+.1f}, {hi:+.1f}], p={p:.3f}{star}')
                    else:
                        cells.append('n/a')
                if any(c != 'n/a' for c in cells):
                    lines.append(f'| {A} − {B} | {m} | ' + ' | '.join(cells) + ' |')
        lines += ['', '** p < 0.05 (two-sided, uncorrected). With several comparisons, treat p values near 0.05 with caution.', '']
    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'STATS.md'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump({'point': point, 'seed_sd': seed_sd, 'B': a.B, 'groups': groups},
              open(os.path.join(OUT, 'stats.json'), 'w'), indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compare', nargs='*', default=None, help='pairs GROUP_A:GROUP_B')
    main(ap.parse_args())

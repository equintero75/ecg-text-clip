import json, os, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.environ.get('RUNS_PARENT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
rows = []
for d in sorted(glob.glob(os.path.join(HERE, 'runs', '*'))):
    f = os.path.join(d, 'results.jsonl')
    if not os.path.exists(f) or 'aborted' in d or 'smoke' in d:
        continue
    R = [json.loads(l) for l in open(f)]
    for r in R:
        if 'linear_probe' not in r:
            continue
        lp = r['linear_probe']
        rows.append({
            'run': os.path.basename(d), 'tag': r['tag'],
            'best_val_zs': max(x['val_zs_scp_max_auc'] for x in R if x['tag'] != 'init') if r['tag'] != 'init' else r['val_zs_scp_max_auc'],
            'ZS_orig': r['test_zs_orig_mean_auc'], 'ZS_scp_mean': r['test_zs_scp_mean_auc'], 'ZS_scp_max': r['test_zs_scp_max_auc'],
            'acc1_scp_max': r['test_zs_scp_max_acc1'], 'LP1%': lp['0.01']['test_auc'], 'LP10%': lp['0.1']['test_auc'],
            'LP100%': lp['1.0']['test_auc'], 'R@1': r['val_e2t_R@1'], 'R@5': r['val_e2t_R@5'],
        })
cols = ['run', 'tag', 'ZS_orig', 'ZS_scp_mean', 'ZS_scp_max', 'acc1_scp_max', 'LP1%', 'LP10%', 'LP100%', 'R@1', 'R@5']
print('| ' + ' | '.join(cols) + ' |')
print('|' + '---|' * len(cols))
for r in rows:
    print('| ' + ' | '.join(str(r[c]) if not isinstance(r[c], float) else f'{r[c]:.3f}' for c in cols) + ' |')
if '--curves' in sys.argv:
    for d in sorted(glob.glob(os.path.join(HERE, 'runs', '*'))):
        f = os.path.join(d, 'results.jsonl')
        if not os.path.exists(f) or 'aborted' in d or 'smoke' in d:
            continue
        print(os.path.basename(d), [(json.loads(l)['tag'], json.loads(l)['val_zs_scp_max_auc']) for l in open(f)])

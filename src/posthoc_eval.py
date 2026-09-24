"""Post-hoc zero-shot prompt study on a trained run: prompt formats selected on VAL, reported on TEST."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecgclip import *

run = sys.argv[1]
d = os.path.join(RUNS, run)
args = json.loads(open(os.path.join(d, 'log.txt')).readline()[len('ARGS '):])
df, scp = load_meta()
sig = np.ascontiguousarray(np.load(os.path.join(CACHE, 'sig500.npy'), mmap_mode='r'))
Y = superclass_labels(df, scp)
fold = df['strat_fold'].values
va, te = np.where(fold == 9)[0], np.where(fold == 10)[0]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args['text_model'])
model = ECGTextCLIP(ECGEncoder(), ClinicalTextEncoder(args['text_model'])).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(d, 'best.pt'), map_location=DEVICE))
_, pva = embed_ecg(model, sig, va)
_, pte = embed_ecg(model, sig, te)

diag = scp[scp['diagnostic'] == 1]
base = {s: list(diag[diag['diagnostic_class'] == s]['description']) for s in SUPERS}
sets = {
    'desc': {s: [x + '.' for x in v] for s, v in base.items()},
    'desc+sinus': {s: [x + '. sinus rhythm.' for x in v] for s, v in base.items()},
    'desc+sinus+ecg': {s: [x + '. sinus rhythm.' if s == 'NORM' else x + '. abnormal ecg. sinus rhythm.' for x in v] for s, v in base.items()},
    'orig': ORIG_TEMPLATES,
}
res = {}
for name, prompts in sets.items():
    for agg in ['mean', 'max']:
        sv = zero_shot_scores(model, tok, pva, prompts, agg)
        st = zero_shot_scores(model, tok, pte, prompts, agg)
        res[f'{name}/{agg}'] = (macro_auc(Y[va], sv)[0], macro_auc(Y[te], st)[0], macro_auc(Y[te], st)[1])
best = max(res, key=lambda k: res[k][0])
for k, (v, t, per) in sorted(res.items(), key=lambda x: -x[1][0]):
    print(f'{run:12s} {k:22s} val {v:.4f} test {t:.4f} per-class {[round(p, 3) for p in per]}' + ('  <- selected on val' if k == best else ''))
with open(os.path.join(d, 'posthoc.json'), 'w') as f:
    json.dump({k: {'val': v, 'test': t, 'per_class': per} for k, (v, t, per) in res.items()} | {'selected': best}, f, indent=1)

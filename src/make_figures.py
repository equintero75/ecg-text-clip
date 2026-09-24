import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUT = os.path.join(HERE, 'figures')


def load(run):
    return [json.loads(l) for l in open(os.path.join(HERE, 'runs', run, 'results.jsonl'))]


def train_loss(run):
    return [float(l.split()[3]) for l in open(os.path.join(HERE, 'runs', run, 'log.txt')) if l.startswith('EPOCH')]


# ── Figure 1: linear probe vs label fraction ──────────────────────────────
runs = [('R_randinit', 'Random init (no pre-training)', 'grey'),
        ('R0_orig', 'R0: v1 notebook settings', 'tab:orange'),
        ('R2_report', 'R2: reports + fixes (5 ep)', 'tab:blue'),
        ('F1_report_15ep', 'F1: reports + fixes (15 ep)', 'tab:green')]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
fr = ['0.01', '0.1', '1.0']
for run, label, col in runs:
    r = load(run)[-1]
    ax[0].plot([1, 10, 100], [r['linear_probe'][f]['test_auc'] for f in fr], 'o-', label=label, color=col)
ax[0].set_xscale('log'); ax[0].set_xticks([1, 10, 100]); ax[0].set_xticklabels(['1%', '10%', '100%'])
ax[0].set_xlabel('Labelled training ECGs used by the linear probe'); ax[0].set_ylabel('Test macro AUROC')
ax[0].set_title('Linear probe on frozen ECG encoder (PTB-XL superclasses)'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

# ── Figure 1b: F1 curves ──────────────────────────────────────────────────
R = load('F1_report_15ep')
ep = list(range(1, len(R) + 1))
ax2 = ax[1]
ax2.plot(ep, [r['val_zs_scp_max_auc'] for r in R], 'o-', label='Val zero-shot macro AUROC', color='tab:green')
ax2.plot(ep, [r['val_e2t_R@5'] for r in R], 's-', label='Val ECG→report Recall@5', color='tab:purple')
ax2.set_xlabel('Epoch'); ax2.set_ylim(0, 1); ax2.grid(alpha=.3)
ax3 = ax2.twinx()
ax3.plot(ep, train_loss('F1_report_15ep'), '--', color='tab:red', label='Train loss')
ax3.set_ylabel('Train loss', color='tab:red')
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right'); ax2.set_title('F1 (15 epochs): training curves')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_linear_probe_and_curves.png'), dpi=150)
print('saved fig 1')

# ── Figure 2: zero-shot per class, R0 vs F1 ───────────────────────────────
SUP = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.26
for k, (run, label, col) in enumerate([('R0_orig', 'R0 (label-derived text)', 'tab:orange'),
                                        ('R2_report', 'R2 (reports, 5 ep)', 'tab:blue'),
                                        ('F1_report_15ep', 'F1 (reports, 15 ep)', 'tab:green')]):
    per = load(run)[-1]['test_zs_scp_max_per_class_auc']
    ax.bar([i + (k - 1) * w for i in range(5)], per, w, label=label, color=col)
ax.axhline(0.5, color='grey', ls=':', lw=1)
ax.set_xticks(range(5)); ax.set_xticklabels(SUP); ax.set_ylim(0.4, 1); ax.set_ylabel('Test AUROC')
ax.set_title('Zero-shot AUROC per superclass (no labelled ECGs used)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_zero_shot_per_class.png'), dpi=150)
print('saved fig 2')

# %%
import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
# %%
# 第 2 步：每个样本先跟自己的 None 比
# long_df = pd.read_csv("./metadata/long_table_base.csv")
# %%
INPUT_PATH = './metadata/long_table_base.csv'
METRICS = ['ACC', 'NMI', 'ARI']

COND_NAME = {
    (1, 1, 1): 'K-CL-ML',
    (1, 0, 1): 'K-CL',
    (1, 1, 0): 'K-ML',
    (1, 0, 0): 'K',
    (0, 1, 1): 'CL-ML',
    (0, 0, 1): 'CL',
    (0, 1, 0): 'ML',
    (0, 0, 0): 'None',
}
# %%
df = pd.read_csv(INPUT_PATH)
df['condition'] = [COND_NAME[(k, ml, cl)] for k, ml, cl in zip(df['K'], df['ML'], df['CL'])] #!
df
# %%
none_df = df[(df['K'] == 0) & (df['ML'] == 0) & (df['CL'] == 0)][
    ['sample', 'seed', 'ACC', 'NMI', 'ARI']
].rename(columns={m: 'none_' + m for m in METRICS})
none_df.shape[0] # 120
# %%
cond_df = df[~((df['K'] == 0) & (df['ML'] == 0) & (df['CL'] == 0))].copy()
cond_df.shape[0] # 840
# %%
cond_df = cond_df.merge(none_df, on=['sample', 'seed'], how='left', validate='many_to_one')
# %%
for m in METRICS:
    cond_df['delta_' + m] = cond_df[m] - cond_df['none_' + m]
cond_df
# %%
delta_by_seed = cond_df[
    ['donor', 'pair', 'sample', 'seed', 'condition', 'K', 'ML', 'CL'] + METRICS + ['none_' + m for m in METRICS] + ['delta_' + m for m in METRICS]
].sort_values(['donor', 'pair', 'sample', 'K', 'ML', 'CL', 'seed']).reset_index(drop=True)
delta_by_seed
# %%
delta_by_seed.to_csv('./metadata/delta_by_seed.csv', index=False)
# %%
assert len(delta_by_seed) == 12 * 7 * 10
assert delta_by_seed[['delta_' + m for m in METRICS]].notna().all().all()
assert set(delta_by_seed['condition']) == set(COND_NAME.values()) - {'None'}
print(delta_by_seed.shape)
print(delta_by_seed.head())
# %%
# by groups
group_keys = ['donor', 'pair', 'sample', 'condition', 'K', 'ML', 'CL']
agg_dict = {'n_seeds': ('seed', 'nunique')}
# %%
for m in METRICS:
    agg_dict['delta_' + m + '_mean'] = ('delta_' + m, 'mean')
    agg_dict['delta_' + m + '_std'] = ('delta_' + m, 'std')
# %%
delta_by_sample = cond_df.groupby(group_keys, as_index=False).agg(**agg_dict)
# %%
for m in METRICS:
    delta_by_sample['delta_' + m + '_se'] = (
        delta_by_sample['delta_' + m + '_std'] / np.sqrt(delta_by_sample['n_seeds'])
    )
    tcrit = t_dist.ppf(0.975, delta_by_sample['n_seeds'] - 1)
    delta_by_sample['delta_' + m + '_ci95_low'] = (
        delta_by_sample['delta_' + m + '_mean'] - tcrit * delta_by_sample['delta_' + m + '_se']
    )
    delta_by_sample['delta_' + m + '_ci95_high'] = (
        delta_by_sample['delta_' + m + '_mean'] + tcrit * delta_by_sample['delta_' + m + '_se']
    )
# %%
delta_by_sample = delta_by_sample.sort_values(
    ['donor', 'pair', 'sample', 'K', 'ML', 'CL']
).reset_index(drop=True)
delta_by_sample.to_csv('./metadata/delta_by_sample.csv', index=False)

assert len(delta_by_sample) == 12 * 7
assert (delta_by_sample['n_seeds'] == 10).all()
print(delta_by_sample.shape)
print(delta_by_sample.head())
# %%
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# %%
OUT_DIR = './figures/delta_prelim'
os.makedirs(OUT_DIR, exist_ok=True)

COND_ORDER = ['K', 'ML', 'CL', 'K-ML', 'K-CL', 'CL-ML', 'K-CL-ML']
KEY_CONDS = ['K-CL-ML', 'ML', 'K']
# %%
sample_meta = (delta_by_sample[['donor', 'pair', 'sample']]
               .drop_duplicates()
               .sort_values(['donor', 'pair', 'sample'])
               .reset_index(drop=True))
sample_meta
# %%
SAMPLE_ORDER = sample_meta['sample'].tolist()
SAMPLE_LABELS = ['%s-p%d-%s' % (r.donor, r.pair, r.sample)
                 for r in sample_meta.itertuples(index=False)]
DONOR_ORDER = sample_meta['donor'].drop_duplicates().tolist()
DONOR_COLORS = dict(zip(DONOR_ORDER, sns.color_palette('Set2', n_colors=len(DONOR_ORDER))))
PAIR_COLORS = {1: '#1f77b4', 2: '#d62728'}
# %%
sns.set_theme(style='whitegrid', context='notebook')
# %%
# 7 个条件下 seed 级 delta 分布 + 12 个样本均值
fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 4.2 * len(METRICS)), sharex=True)
for i, (ax, m) in enumerate(zip(axes, METRICS)):
    sns.boxplot(data=delta_by_seed, x='condition', y='delta_' + m,
                order=COND_ORDER, color='0.9', width=0.6,
                showfliers=False, ax=ax)
    sns.stripplot(data=delta_by_sample, x='condition', y='delta_' + m + '_mean',
                  order=COND_ORDER, hue='donor', hue_order=DONOR_ORDER,
                  palette=DONOR_COLORS, dodge=False, jitter=0.08,
                  size=4.5, alpha=0.9, edgecolor='white', linewidth=0.3, ax=ax)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_title('%s: seed-level delta distribution + sample means' % m)
    ax.set_xlabel('')
    ax.set_ylabel('delta vs None')
axes[-1].set_xticklabels(COND_ORDER, rotation=30, ha='right')
fig.legend(handles, labels, title='donor', loc='upper right', bbox_to_anchor=(1.0, 0.98))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_delta_distribution_by_condition.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
# %%
# 样本 × 条件热图，ACC/NMI/ARI 三张
fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6.5))
for i, (ax, m) in enumerate(zip(axes, METRICS)):
    pivot = (delta_by_sample
             .pivot(index='sample', columns='condition', values='delta_' + m + '_mean')
             .reindex(index=SAMPLE_ORDER, columns=COND_ORDER))
    lim = float(np.nanmax(np.abs(pivot.values)))
    sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0, vmin=-lim, vmax=lim,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                cbar_kws={'label': 'delta %s' % m},
                yticklabels=SAMPLE_LABELS if i == 0 else False)
    ax.set_title('%s: mean delta vs None' % m)
    ax.set_xlabel('')
    ax.set_ylabel('sample' if i == 0 else '')
    ax.set_xticklabels(COND_ORDER, rotation=35, ha='right')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_sample_by_condition_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
# %%
# fig3_donor_ACC.png, fig3_donor_NMI.png, fig3_donor_ARI.png：按供体分面，样本点 + 供体均值线
for m in METRICS:
    fig, axes = plt.subplots(1, len(DONOR_ORDER), figsize=(15, 4.5), sharey=True)
    for j, (ax, donor) in enumerate(zip(axes, DONOR_ORDER)):
        sub = delta_by_sample[delta_by_sample['donor'] == donor]
        samples = (sub[['pair', 'sample']].drop_duplicates()
                   .sort_values(['pair', 'sample'])['sample'].tolist())
        offsets = np.linspace(-0.18, 0.18, len(samples))
        for k, sample in enumerate(samples):
            g = sub[sub['sample'] == sample].set_index('condition').reindex(COND_ORDER)
            pair = int(g['pair'].iloc[0])
            ax.scatter(np.arange(len(COND_ORDER)) + offsets[k],
                       g['delta_' + m + '_mean'].values,
                       color=PAIR_COLORS.get(pair, '0.5'), s=30, alpha=0.9)
        mean_vals = sub.groupby('condition')['delta_' + m + '_mean'].mean().reindex(COND_ORDER)
        ax.plot(np.arange(len(COND_ORDER)), mean_vals.values,
                color='black', marker='o', lw=1.5, label='donor mean')
        ax.axhline(0, color='grey', lw=1, ls='--')
        ax.set_xticks(np.arange(len(COND_ORDER)))
        ax.set_xticklabels(COND_ORDER, rotation=35, ha='right')
        ax.set_title('%s (n=%d samples)' % (donor, len(samples)))
        ax.set_xlabel('')
        ax.set_ylabel('delta %s vs None' % m if j == 0 else '')

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=PAIR_COLORS[1], markersize=7, label='pair 1'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=PAIR_COLORS[2], markersize=7, label='pair 2'),
        plt.Line2D([0], [0], color='black', marker='o', label='donor mean'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.0, 0.98))
    fig.suptitle('%s: sample effects by donor' % m)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_donor_%s.png' % m),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
# %%
# fig4_seed_stability_ARI.png：12 个样本的 seed 稳定性，展示 K-CL-ML、ML、K
seed_pivot = (delta_by_seed[delta_by_seed['condition'].isin(KEY_CONDS)]
              .pivot_table(index=['sample', 'seed'], columns='condition', values='delta_ARI')
              .reset_index())

fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
axes = axes.ravel()
for ax, sample in zip(axes, SAMPLE_ORDER):
    g = seed_pivot[seed_pivot['sample'] == sample].sort_values('seed')
    for cond in KEY_CONDS:
        ax.plot(g['seed'], g[cond], marker='o', ms=3, lw=1, label=cond)
    ax.axhline(0, color='grey', lw=1, ls='--')
    ax.set_title(SAMPLE_LABELS[SAMPLE_ORDER.index(sample)], fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

axes[0].legend(fontsize=8)
for ax in axes[-4:]:
    ax.set_xlabel('seed')
for ax in axes[::4]:
    ax.set_ylabel('delta ARI')
fig.suptitle('Seed-level stability for key conditions (ARI)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_seed_stability_ARI.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# %%
# fig5_none_baseline.png：每个样本的 None 基线（seed 散点 + 均值 ± SD）
none_base = (delta_by_seed[['donor', 'pair', 'sample', 'seed',
                            'none_ACC', 'none_NMI', 'none_ARI']]
             .drop_duplicates(['sample', 'seed']))
fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 3.2 * len(METRICS)), sharex=True)
rng = np.random.RandomState(0)
for ax, m in zip(axes, METRICS):
    for i, sample in enumerate(SAMPLE_ORDER):
        g = none_base[none_base['sample'] == sample]
        donor = g['donor'].iloc[0]
        color = DONOR_COLORS[donor]
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g['none_' + m],
                   s=10, color=color, alpha=0.35, edgecolor='none')
        ax.errorbar(i, g['none_' + m].mean(), yerr=g['none_' + m].std(),
                    fmt='o', color=color, ms=6, capsize=3, lw=1.5)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel('None %s' % m)
    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color='grey', lw=0.8, ls=':')

axes[-1].set_xticks(np.arange(len(SAMPLE_ORDER)))
axes[-1].set_xticklabels(SAMPLE_LABELS, rotation=35, ha='right')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=DONOR_COLORS[d], markersize=7, label=d)
                  for d in DONOR_ORDER]
fig.legend(handles=legend_handles, title='donor', loc='upper right')
fig.suptitle('None baseline by sample')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_none_baseline.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
# %%
summary_rows = []
for m in METRICS:
    for cond in COND_ORDER:
        vals = delta_by_sample.loc[delta_by_sample['condition'] == cond,
                                   'delta_' + m + '_mean']
        summary_rows.append({
            'metric': m,
            'condition': cond,
            'n_samples': len(vals),
            'mean_of_sample_means': vals.mean(),
            'positive_samples': int((vals > 0).sum()),
            'min_sample_mean': vals.min(),
            'max_sample_mean': vals.max(),
        })

delta_summary = pd.DataFrame(summary_rows)
delta_summary.to_csv('./metadata/delta_summary_by_condition.csv', index=False)
print(delta_summary)
# %%
# 到目前为止，初步可见的模式：
# K 单独开启在 12 个样本上基本都是正的，较稳定。
# ML 的效应明显依赖样本/供体：Br5595 普遍较强，Br5292 的 151509/151510 接近 0 或为负。
# CL 总体较弱，多数接近 0，个别样本为负。
# 同一 pair 的两张切片趋势较接近（如 151507/151508、151669/151670），但 pair 之间可以明显不同。
# seed 级曲线大多比较平稳，说明样本间差异不是 seed 随机波动造成的。
# %%
# 对 K(GAT) / ML / CL 的 2x2x2 控制实验结果做析因分析
# 输入：metadata/<run_dir>/ 下 8 个条件各一个 metrics_*.csv（列: seed,ACC,NMI,ARI,kNN_ACC,MoranI）
# 输出：控制台统计表 + figures/ 下的交互作用图
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats as sstats

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, 'metadata/try_151507_5rep')
FIG_DIR = os.path.join(HERE, 'figures')

METRICS = ['ACC', 'NMI', 'ARI', 'kNN_ACC', 'MoranI']
# 按指标分层：表达/聚类类 vs 空间一致性类
EXPR_METRICS = ['ACC', 'NMI', 'ARI']
SPATIAL_METRICS = ['kNN_ACC', 'MoranI']

# 条件名 -> 文件名（与 02_visualize 的命名保持一致）
CONDITIONS = {
    'K+ML+CL': 'metrics_K-CL-ML_151507.csv',
    'K+ML':    'metrics_K-ML_151507.csv',
    'K+CL':    'metrics_K-CL_151507.csv',
    'K':       'metrics_K_151507.csv',
    'ML+CL':   'metrics_ML-CL_151507.csv',
    'ML':      'metrics_ML_151507.csv',
    'CL':      'metrics_CL_151507.csv',
    'None':    'metrics__151507.csv',
}


def decode_factors(name):
    return {'K': int('K' in name), 'ML': int('ML' in name), 'CL': int('CL' in name)}


def load_all(run_dir=RUN_DIR):
    frames = []
    for name, fname in CONDITIONS.items():
        df = pd.read_csv(os.path.join(run_dir, fname))
        df['cond'] = name
        for k, v in decode_factors(name).items():
            df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


long_df = load_all()
print('=== 每个条件的均值（n=%d/条件）===' % long_df.groupby('cond').size().iloc[0])
print(long_df.groupby('cond')[METRICS].mean().round(3))

# %%
# ---- 1. 主效应：每个因子在 level=1 与 level=0 之间的均值差 ----
def main_effects(df, metrics=METRICS):
    rows = {}
    for m in metrics:
        rows[m] = {
            'K':  df[df.K == 1][m].mean() - df[df.K == 0][m].mean(),
            'ML': df[df.ML == 1][m].mean() - df[df.ML == 0][m].mean(),
            'CL': df[df.CL == 1][m].mean() - df[df.CL == 0][m].mean(),
        }
    return pd.DataFrame(rows).T


print('\n=== 主效应 (mean|on - mean|off) ===')
print(main_effects(long_df).round(3))

# %%
# ---- 2. 三因素析因 ANOVA：主效应 + 交互效应 + 偏 eta^2 ----
def bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    return multipletests(pvals, method='fdr_bh')[1]


def three_way_anova(df, metric):
    model = smf.ols(f'{metric} ~ K * ML * CL', data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)  # 平衡设计下 typ=2 与 typ=3 等价
    ss_res = aov.loc['Residual', 'sum_sq']
    aov['partial_eta_sq'] = aov['sum_sq'] / (aov['sum_sq'] + ss_res)
    aov['metric'] = metric
    return aov


anova_all = pd.concat([three_way_anova(long_df, m) for m in METRICS])
anova_all['p_BH'] = np.nan
_mask = anova_all['PR(>F)'].notna()
anova_all.loc[_mask, 'p_BH'] = bh(anova_all.loc[_mask, 'PR(>F)'])
print('\n=== 三因素 ANOVA (typ=2, BH 校正) ===')
print(anova_all[['metric', 'F', 'PR(>F)', 'p_BH', 'partial_eta_sq']].round(4))

# %%
# ---- 3. 事后/简单效应：固定 K 水平后，ML 或 CL 的效应 ----
def simple_effect(df, factor, level, metric):
    sub = df[df['K'] == level] if factor != 'K' else df[df['K'] == level]
    other = sub[sub[factor] == 1][metric]
    base = sub[sub[factor] == 0][metric]
    diff = other.mean() - base.mean()
    t, p = sstats.ttest_ind(other, base, equal_var=True)
    return diff, p


rows = []
for m in METRICS:
    for f, lev, tag in [('ML', 1, 'ML|K=1'), ('ML', 0, 'ML|K=0'),
                        ('CL', 1, 'CL|K=1'), ('CL', 0, 'CL|K=0'),
                        ('K', 1, 'K|ML=1'), ('K', 0, 'K|ML=0')]:
        if f == 'K':
            sub = long_df[long_df['ML'] == lev]
            diff = sub[sub.K == 1][m].mean() - sub[sub.K == 0][m].mean()
            t, p = sstats.ttest_ind(sub[sub.K == 1][m], sub[sub.K == 0][m], equal_var=True)
        else:
            diff, p = simple_effect(long_df, f, lev, m)
        rows.append(dict(metric=m, effect=tag, diff=diff, p=p))
simple = pd.DataFrame(rows)
simple['p_BH'] = bh(simple['p'])
print('\n=== 简单效应（固定另一因子后）===')
print(simple.round(4))

# %%
# ---- 4. 交互作用图：x=K on/off，线=ML，分面=CL ----
os.makedirs(FIG_DIR, exist_ok=True)
for m in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cl in zip(axes, [0, 1]):
        sub = long_df[long_df.CL == cl]
        for ml in [0, 1]:
            s = sub[sub.ML == ml].groupby('K')[m].agg(['mean', 'std'])
            ax.errorbar([0, 1], s['mean'], yerr=s['std'], marker='o', capsize=4, label='ML=%d' % ml)
        ax.set_title('CL=%d' % cl)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['K off', 'K on'])
        ax.set_xlabel('GAT(K)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel(m)
    fig.suptitle('Interaction plot: %s' % m)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'interaction_%s.png' % m), dpi=150)
    plt.close(fig)

# %%
# ---- 5. 结论速览：显著的项（BH 后 p<0.05）----
sig = anova_all[anova_all['p_BH'] < 0.05].copy()
print('\n=== ANOVA 中 BH 校正后仍显著的项 ===')
print(sig[['metric', 'F', 'p_BH', 'partial_eta_sq']].round(4))
print('\n=== 各指标的表达式/空间分层均值 ===')
print('表达/聚类类:', main_effects(long_df, EXPR_METRICS).round(3).to_dict())
print('空间一致性类:', main_effects(long_df, SPATIAL_METRICS).round(3).to_dict())

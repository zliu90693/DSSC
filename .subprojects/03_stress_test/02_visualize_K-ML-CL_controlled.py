# %%
from functions.visualize_DSSC_pred import *
# %%
DATASETS = {
    '151507': ('../../data/sample_151507_anno.h5', '../../src/out_151507'),
    # 'osmFISH': ('../../data/osmFISH_cortex.h5', '../../src/out_osmFISH'),
}
# %%
# K = 20, ML use，CL use
# seed: 200
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(201, 206))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_K-CL-ML_%s.csv' % name, index=False)
# %%
# K = 20, ML use，CL not use
# seed: 300
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(301, 306))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_K-ML_%s.csv' % name, index=False)
# %%
# K = 20, ML not use，CL use
# seed: 400
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(401, 406))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_K-CL_%s.csv' % name, index=False)
# %%
# K = 20, ML not use，CL not use
# seed: 500
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(501, 506))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_K_%s.csv' % name, index=False)
# %%
# K = 0, ML use，CL use
# seed: 600
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(601, 606))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_ML-CL_%s.csv' % name, index=False)
# %%
# K = 0, ML use，CL not use
# seed: 700
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(701, 706))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_ML_%s.csv' % name, index=False)
# %%
# K = 0, ML not use，CL use
# seed: 800
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(801, 806))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_CL_%s.csv' % name, index=False)
# %%
# K = 0, ML not use，CL not use
# seed: 900
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(901, 906))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics__%s.csv' % name, index=False)
# %%
METRICS = ['ACC', 'NMI', 'ARI', 'kNN_ACC', 'MoranI']
METRIC_FILES = {
    'K-CL-ML (seed: 200)': './metadata/metrics_K-CL-ML_151507.csv',
    'K-ML (seed: 300)': './metadata/metrics_K-ML_151507.csv',
    'K-CL (seed: 400)': './metadata/metrics_K-CL_151507.csv',
    'K (seed: 500)': './metadata/metrics_K_151507.csv',
    'ML-CL (seed: 600)': './metadata/metrics_ML-CL_151507.csv',
    'ML (seed: 700)': './metadata/metrics_ML_151507.csv',
    'CL (seed: 800)': './metadata/metrics_CL_151507.csv',
    'None (seed: 900)': './metadata/metrics__151507.csv'
    # 'osmFISH': './metadata/metrics_osmFISH.csv',
}
# %%
for name, path in METRIC_FILES.items():
    fig, ax = plot_metrics_bar(path, name, metrics=METRICS)
    figpath = f"./figures/{path.split('/')[-1].split('.')[0]}.png"
    fig.savefig(figpath, dpi=150)
plt.show()
# %%
# 单图
fig, ax = plot_metrics_grouped(
    METRIC_FILES, METRICS,
    savepath='./figures/metrics_K-ML-CL_controlled_all8.png')
plt.show()
# %%

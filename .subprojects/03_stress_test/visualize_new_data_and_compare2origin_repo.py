# %%
from functions.visualize_DSSC_pred import *
# %%
DATASETS = {
    'K-CL-ML_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K20_ML1_CL1'),
    'K-CL_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K20_ML0_CL1'),
    'K-ML_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K20_ML1_CL0'),
    'K_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K20_ML0_CL0'),

    'CL-ML_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K0_ML1_CL1'),
    'CL_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K0_ML0_CL1'),
    'ML_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K0_ML1_CL0'),
    '_151507': ('../../data/spatialLIBD/sample_151507_anno.h5', '../../src/out_151507/K0_ML0_CL0'),
    # 'osmFISH': ('../../data/osmFISH_cortex.h5', '../../src/out_osmFISH'),
}
# %%
for name, (h5_path, save_dir) in DATASETS.items():
    df = evaluate_dataset(h5_path, save_dir, seeds=range(201, 206))
    print('=== %s ===' % name)
    display(df.set_index('seed'))
    print('mean +/- std')
    display(summarize_runs(df))
    df.to_csv('./metadata/metrics_%s.csv' % name, index=False)
# %%
METRICS = ['ACC', 'NMI', 'ARI', 'kNN_ACC', 'MoranI']
METRIC_FILES = {
    'K-CL-ML_151507': './metadata/metrics_K-CL-ML_151507.csv',
    'K-CL_151507': './metadata/metrics_K-CL_151507.csv',
    'K-ML_151507': './metadata/metrics_K-ML_151507.csv',
    'K_151507': './metadata/metrics_K_151507.csv',

    'CL-ML_151507': './metadata/metrics_CL-ML_151507.csv',
    'CL_151507': './metadata/metrics_CL_151507.csv',
    'ML_151507': './metadata/metrics_ML_151507.csv',
    '_151507': './metadata/metrics__151507.csv'
    # 'osmFISH': './metadata/metrics_osmFISH.csv',
}
# %%
for name, path in METRIC_FILES.items():
    fig, ax = plot_metrics_bar(path, name, metrics=METRICS)
    # figpath = f"./figures/{path.split('/')[-1].split('.')[0]}.png"
    # fig.savefig(figpath, dpi=150)
plt.show()
# %%
fig, ax = plot_metrics_grouped(
    METRIC_FILES, METRICS,
    savepath='./figures/metrics_K-ML-CL_controlled_all8_new_data.png')
plt.show()
# %%

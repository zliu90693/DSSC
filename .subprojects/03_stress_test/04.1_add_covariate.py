# 以 long_df 为基础，加入可能导致异质性的技术协变量
# %%
import pandas as pd
import h5py
import numpy as np
import glob
import os
import re
# %%
long_df = pd.read_csv("./metadata/long_table_base.csv")
# %%
SAMPLES = ['151507', '151508', '151509', '151510', 
           '151669', '151670', '151671', '151672',
           '151673', '151674', '151675', '151676']
SAMPLES_ADATA = [h5py.File(f"../../data/spatialLIBD/sample_{sample}_anno.h5", 'r') for sample in SAMPLES]
# %%
CONSTRAINT_DIR = '../../example_constraints'
LAYER_ORDER = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
# %%
def constraint_coverage(path, n_spots):
    idx = np.unique(np.loadtxt(path, dtype=int).reshape(-1) - 1) # .reshape(-1)：将两列放进一个array里
    if idx.size == 0:
        return 0.0
    if idx.min() < 0 or idx.max() >= n_spots:
        raise ValueError('constraint index out of range: %s' % path)
    return len(idx) / n_spots
# %%
# %%
cov_rows = []
for sample, h5 in zip(SAMPLES, SAMPLES_ADATA):
    y_all = np.array(h5['Y']).astype(str)
    x_all = np.array(h5['X'])
    genes = np.array(h5['Gene']).astype(str)

    keep = y_all != 'NA'
    y = y_all[keep]
    x = x_all[keep]
    n_spots = int(keep.sum())

    total_counts = x.sum(axis=1) # shape: 4221
    detected_genes = (x > 0).sum(axis=1) # shape: 4221
    mt_idx = np.where(np.char.startswith(np.char.upper(genes), 'MT-'))[0]
    mt_counts = x[:, mt_idx].sum(axis=1) # shape: 4221

    row = {
        'sample': int(sample),
        'n_spots': n_spots,
        'median_counts': float(np.median(total_counts)), # 深度中位数
        'median_genes': float(np.median(detected_genes)), # 检测到基因中位数
        'mito_frac': float(np.mean(mt_counts / np.maximum(total_counts, 1))), # 线粒体深度占总深度比例
        'na_frac': float((~keep).sum() / len(keep)), # len(keep)仍然是4226，keep.sum()是4221
        'n_layers': int(len(np.unique(y))),
    }

    layer_counts = pd.Series(y).value_counts()
    unknown = set(layer_counts.index) - set(LAYER_ORDER)
    if unknown:
        raise ValueError('unknown layer labels: %s' % sorted(unknown))
    for layer in LAYER_ORDER:
        row['prop_' + layer] = float(layer_counts.get(layer, 0) / n_spots)

    ml_path = os.path.join(CONSTRAINT_DIR, 'sample_%s_mlFromMarks_test.txt' % sample)
    cl_path = os.path.join(CONSTRAINT_DIR, 'sample_%s_clFromMarks_test.txt' % sample)
    row['ml_coverage'] = float(constraint_coverage(ml_path, n_spots))
    row['cl_coverage'] = float(constraint_coverage(cl_path, n_spots))
    cov_rows.append(row)
# %%
cov_columns = (['sample', 'n_spots', 'median_counts', 'median_genes', 'mito_frac', 'na_frac', 'n_layers']
               + ['prop_' + layer for layer in LAYER_ORDER]
               + ['ml_coverage', 'cl_coverage'])
sample_covariates = pd.DataFrame(cov_rows)[cov_columns]
sample_covariates.to_csv('./metadata/sample_covariates.csv', index=False)
# %%
print(sample_covariates.shape)
# print(sample_covariates)

# %%
for h5 in SAMPLES_ADATA:
    h5.close()
# %%
# %%
merged = long_df.merge(sample_covariates, on='sample', how='left')
print(merged.shape)
print(merged['sample'].equals(long_df['sample']))
# %%
merged.to_csv("./metadata/merged.csv", index=False)
# %%

# %%
import pandas as pd
# %%
# donor: Br5292; Br5595; Br8100
# pair: 1, 2; 1, 2; 1, 2
# sample: 151507, 151508, 151509, 151510;
#          151669, 151670, 151671, 151672;
#          151673, 151674, 151675, 151676
# %%
# columns: sample, donor, pair, section_in_pair, seed, K, ML, CL, ACC, NMI, ARI, kNN_ACC, MoranI

# %%
import glob
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_ROOT = os.path.join(BASE_DIR, 'metadata')

# 供体 -> 样本（每个供体两对空间重复，每对两张切片）
DONORS = {
    'Br5292': ['151507', '151508', '151509', '151510'],
    'Br5595': ['151669', '151670', '151671', '151672'],
    'Br8100': ['151673', '151674', '151675', '151676'],
}

# 样本 -> donor / pair：每个供体内前两张为 pair 1，后两张为 pair 2
SAMPLE_META = {}
for donor, samples in DONORS.items():
    for i, sample in enumerate(samples):
        SAMPLE_META[sample] = {'donor': donor, 'pair': i // 2 + 1}

# 组合名 -> K / ML / CL 三因素的开(1)关(0)状态
COND_FLAGS = {
    'K-CL-ML': {'K': 1, 'ML': 1, 'CL': 1},
    'K-CL':    {'K': 1, 'ML': 0, 'CL': 1},
    'K-ML':    {'K': 1, 'ML': 1, 'CL': 0},
    'K':       {'K': 1, 'ML': 0, 'CL': 0},
    'CL-ML':   {'K': 0, 'ML': 1, 'CL': 1},
    'CL':      {'K': 0, 'ML': 0, 'CL': 1},
    'ML':      {'K': 0, 'ML': 1, 'CL': 0},
    'None':    {'K': 0, 'ML': 0, 'CL': 0},
}

METRICS = ['ACC', 'NMI', 'ARI', 'kNN_ACC', 'MoranI']

# %%
# 读取 metadata/<sample>/metrics_<cond>_<sample>.csv，合并为长表
frames = []
for path in sorted(glob.glob(os.path.join(META_ROOT, '151*', 'metrics_*.csv'))):
    name = os.path.basename(path)
    m = re.fullmatch(r'metrics_(.+)_(\d+)\.csv', name)
    if m is None:
        continue
    cond, sample = m.group(1), m.group(2)
    if sample not in SAMPLE_META:
        continue
    if cond not in COND_FLAGS:
        print('!! unknown condition: %s (skip %s)' % (cond, path))
        continue

    df = pd.read_csv(path)
    df.insert(0, 'donor', SAMPLE_META[sample]['donor'])
    df.insert(1, 'pair', SAMPLE_META[sample]['pair'])
    df.insert(2, 'sample', sample)
    for flag, value in COND_FLAGS[cond].items():
        df[flag] = value
    frames.append(df)

LONG_COLUMNS = ['donor', 'pair', 'sample', 'seed', 'K', 'ML', 'CL'] + METRICS
long_df = pd.concat(frames, ignore_index=True)
long_df = long_df[LONG_COLUMNS]
long_df = long_df.sort_values(
    ['donor', 'pair', 'sample', 'K', 'ML', 'CL', 'seed']
).reset_index(drop=True)

# %%
# 保存基础长表
out_path = os.path.join(META_ROOT, 'long_table.csv')
long_df.to_csv(out_path, index=False)
print('saved:', out_path)
print('shape:', long_df.shape)
print(long_df.head())

# %%
# 完整性检查：12 samples x 8 conditions x 10 seeds = 960 rows
counts = long_df.groupby('sample').size()
print('\nrows per sample:')
print(counts)
assert len(long_df) == 12 * 8 * 10, 'expected 960 rows, got %d' % len(long_df)
assert set(counts.index) == set(SAMPLE_META), 'unexpected sample set'
assert (counts == 8 * 10).all(), counts[counts != 8 * 10]
assert set(long_df['seed']) == set(range(200, 210)), 'unexpected seeds'
assert long_df[METRICS].notna().all().all(), 'missing metric values'
print('\nall checks passed')
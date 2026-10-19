# %%
from functions.visualize_DSSC_pred import *
import os

# %%
# 12 个样本；每个样本 8 个组合，输出目录 ../../src/out_*/子目录
SAMPLES = ['151507', '151508', '151509', '151510',
           '151669', '151670', '151671', '151672',
           '151673', '151674', '151675', '151676']

# 与 151507 现状一致（原用 201–205）；新数据实际生成了 200–209，可改为 range(200, 210) 用满 10 个重复
SEEDS = range(200, 210)

METRICS = ['ACC', 'NMI', 'ARI', 'kNN_ACC', 'MoranI']

# 组合名 -> 输出子目录（对应 run 脚本的 K{K}_ML{ML}_CL{CL}）
COND_SUBDIR = {
    'K-CL-ML': 'K20_ML1_CL1',
    'K-CL':    'K20_ML0_CL1',
    'K-ML':    'K20_ML1_CL0',
    'K':       'K20_ML0_CL0',
    'CL-ML':   'K0_ML1_CL1',
    'CL':      'K0_ML0_CL1',
    'ML':      'K0_ML1_CL0',
    'None':        'K0_ML0_CL0'
}

try:
    display
except NameError:
    display = print

# %%
for sample in SAMPLES:
    h5_path = '../../data/spatialLIBD/sample_%s_anno.h5' % sample
    out_root = '../../src/out_%s' % sample
    meta_dir = './metadata/%s' % sample
    fig_dir = './figures/%s' % sample
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    metric_files = {}
    for cond, sub in COND_SUBDIR.items():
        name = '%s_%s' % (cond, sample)
        save_dir = os.path.join(out_root, sub)
        csv_path = os.path.join(meta_dir, 'metrics_%s.csv' % name)
        if not os.path.isdir(save_dir):
            print('!! missing dir: %s (skip)' % save_dir)
            continue
        df = evaluate_dataset(h5_path, save_dir, seeds=SEEDS)
        print('=== %s ===' % name)
        display(df.set_index('seed'))
        print('mean +/- std')
        display(summarize_runs(df))
        df.to_csv(csv_path, index=False)
        metric_files[name] = csv_path

    # 8 个组合合并到一张图
    if metric_files:
        fig, ax = plot_metrics_grouped(
            metric_files, METRICS,
            savepath=os.path.join(fig_dir, 'metrics_K-ML-CL_controlled_all8_new_data.png'))
        plt.show()
        plt.close(fig)
# %%

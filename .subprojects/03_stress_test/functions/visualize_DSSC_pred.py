import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy import spatial
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../src')) 
#! ↑ 注意，在 Python 中，os.path.abspath('../../../src') 是以当前进程的工作目录（即上一级目录中的 Jupyter 所在的目录）为基准来计算相对路径的
# 不以 visualize_DSSC_pred.py 文件本身的位置为基准
# 因此是 '../../src'，不用 '../../../src'
from utils import eval_cluster, knn_ACC, Iscore_label

def load_origin_reference(h5_path):
    with h5py.File(h5_path, 'r') as f:
        y = np.array(f['Y'])
        pos = np.array(f['Pos'])
    if pos.shape[1] != 2:
        pos = pos.T
    keep = np.where(y.astype(str) != 'NA')[0]
    y_true = pd.factorize(y[keep].astype(str))[0]
    pos = pos[keep].astype(float)
    return y_true, pos


def build_spatial_neighbor_graphs(pos, k0=20):
    A0 = kneighbors_graph(pos, k0, mode='connectivity', metric='euclidean',
                          include_self=False, n_jobs=-1).toarray()
    dist = spatial.distance_matrix(pos, pos)
    p_ = [np.argpartition(dist[i], k0)[:k0] for i in range(dist.shape[0])]
    return A0, p_

def evaluate_prediction(y_true, y_pred, p_, A0):
    acc, nmi, ari = eval_cluster(y_true, y_pred)
    return {
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari,
        'kNN_ACC': knn_ACC(p_, y_pred),
        'MoranI': Iscore_label(y_pred + 1.0, A0),
    }


def evaluate_dataset(h5_path, save_dir, seeds=range(1001, 1006), pred_prefix='pred.csv_'):
    y_true, pos = load_origin_reference(h5_path)
    A0, p_ = build_spatial_neighbor_graphs(pos)
    rows = []
    for s in seeds:
        y_pred = np.loadtxt(os.path.join(save_dir, pred_prefix + str(s))).astype(int)
        assert len(y_pred) == len(y_true), 'seed %s length %d != %d' % (s, len(y_pred), len(y_true))
        rows.append(dict(seed=s, **evaluate_prediction(y_true, y_pred, p_, A0)))
    return pd.DataFrame(rows)


def plot_metrics_bar(csv_path, title, metrics, ylim=(0, 1)):
    df = pd.read_csv(csv_path)
    means = [df[m].mean() for m in metrics]
    stds = [df[m].std() for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=stds, capsize=8, color='steelblue')
    for xi, mean, std in zip(x, means, stds):
        ax.text(xi, mean + std, '%.3f' % mean, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('score')
    ax.set_ylim(*ylim)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def summarize_runs(df):
    return df.drop(columns='seed').agg(['mean', 'std']).T

def plot_metrics_grouped(metric_files, metrics, ylim=(0, 1),
                         colors=None, figsize=(14, 5), savepath=None):
    names = list(metric_files.keys())
    n = len(names)
    means = np.zeros((n, len(metrics)))
    stds = np.zeros((n, len(metrics)))
    for i, path in enumerate(metric_files.values()):
        df = pd.read_csv(path)
        means[i] = df[metrics].mean().values
        stds[i] = df[metrics].std().values

    x = np.arange(len(metrics))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=figsize)
    colors = colors or plt.get_cmap('tab10').colors
    for i, name in enumerate(names):
        ax.bar(x + (i - (n - 1) / 2) * width, means[i], width,
               yerr=stds[i], capsize=3, color=colors[i % len(colors)], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('score')
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax
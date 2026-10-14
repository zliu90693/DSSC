# 实际运行 DSSC 前，检查输入数据
# %%
# import scanpy as sc
import h5py
import numpy as np
# %%
osm_cortex = h5py.File("../data/osmFISH_cortex.h5")
# %%
print(osm_cortex.keys()) # ['Genes', 'Pos', 'Region', 'X', 'Y']
# %%
print(osm_cortex["X"].shape) # cell * gene 矩阵，(4839, 33)
print(osm_cortex["Y"].shape) # 细胞类型注释，整数编码形式(4839,)
print(osm_cortex["Region"].shape) # 细胞类型注释，真实标签形式 (4839,)
print(osm_cortex["Pos"].shape) # 各细胞的空间信息，(4839, 2)
print(osm_cortex["Genes"].shape) # 基因数，(33,)
# %%
for k in osm_cortex.keys():
    d = osm_cortex[k]
    print("-"*40)
    print(k, d.shape, d.dtype)
    print(d[:5])                      # 前几个值
    print("Unique Value:", np.unique(d[:]))  # 分类/标签类数据看唯一值
# %%
sample_151507_anno = h5py.File("../data/sample_151507_anno.h5")
print(sample_151507_anno.keys()) # ['Del', 'Gene', 'Loc', 'Pos', 'X', 'Y']
# %%
for k in sample_151507_anno.keys():
    d = sample_151507_anno[k]
    print("-"*40)
    print(k, d.shape, d.dtype)
    print(d[:5])
    print("Unique Value:", np.unique(d[:]))
# %%
#! make_links_from_Markers.R 的功能：sample_151507_anno.h5 -> make_links_from_Markers.R -> Must-Link & Cannot-Link
'''
论文真正重要的设计是同时把两类 spatial prior 纳入 clustering: 
空间信息进入 graph neural network, 
而 marker genes 的空间表达模式进一步产生 cell-to-cell constraints; 
同时 expression representation 本身通过 denoising autoencoder/ZINB 等进行建模。
'''
# %%
# 20261014，下载了新 spatialLIBD 数据集后的测试：
# repo 中附带的旧数据
sample_151507_anno = h5py.File("../data/sample_151507_anno.h5", 'r')
print(sample_151507_anno.keys()) # ['Del', 'Gene', 'Loc', 'Pos', 'X', 'Y']
# %%
print(sample_151507_anno['Del'].shape) # (161,)
print(sample_151507_anno['Gene'].shape) # (24402,)
print(sample_151507_anno['Loc'].shape) # (2, 4226)
print(sample_151507_anno['Pos'].shape) # (2, 4226)
print(sample_151507_anno['X'].shape) # (4226, 24402)
print(sample_151507_anno['Y'].shape) # (4226,)
# %%
# 在 .subprojects/02_replicate/get_other_data.ipynb 中下载的新数据
#! 检查新数据，与旧数据对比，确保 get_other_data.ipynb 中的 h5 保存未出错误：
sample_151507_anno_new = h5py.File("../data/spatialLIBD/sample_151507_anno.h5", 'r')
print(sample_151507_anno_new.keys()) # ['Gene', 'Pos', 'X', 'Y']
# %%
print(sample_151507_anno_new['Gene'].shape) # (33538,)
print(sample_151507_anno_new['Pos'].shape) # (2, 4226)
print(sample_151507_anno_new['X'].shape) # (4226, 33538)
print(sample_151507_anno_new['Y'].shape) # (4226,)
# Del 在 DSSC 中从未使用过，spot 的筛选只靠 Y == "NA"（run_DSSC.py:65）和预处理阈值，与 Del 无关，因此新数据可用
# 相较于旧数据，新数据的形状发生了一定变化 (4226, 33538)
# %%

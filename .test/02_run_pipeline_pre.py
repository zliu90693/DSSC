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

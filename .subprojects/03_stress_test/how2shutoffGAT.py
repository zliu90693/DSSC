# %%
import h5py
import numpy as np
# %%
data_mat = h5py.File("../../data/sample_151507_anno.h5", 'r')
x = np.array(data_mat['X'])
pos = np.array(data_mat['Pos'])
if pos.shape[1] != 2: 
    pos = pos.T
y = np.array(data_mat['Y'])
data_mat.close()
# %%
import pandas as pd
import scanpy as sc
# %%
print(f'需剔除细胞数: {(y.astype(np.str) == "NA").sum()}') # 需剔除的细胞数：5
# 2 剔除 NA
f = np.where(y.astype(np.str) != "NA")[0]
y = y[f]
x = x[f,:]
pos = pos[f,:]
pos = pos.astype(np.float)
# 3 获取簇数
n_clusters = np.shape(np.unique(y))[0] # 共有几簇？
print(f'簇数: {n_clusters}')
# 4 构建 Anndata 对象
obs = pd.DataFrame()
obs['cell_labels'] = pd.Categorical(y, categories=np.unique(y), ordered=False)
obs['batch'] = pd.Categorical(np.repeat(1, x.shape[0]), categories=[1], ordered=False)
obs['pos_x'] = pos[:,0]
obs['pos_y'] = pos[:,1]
adata = sc.AnnData(X=x, obs=obs)
# %%
import sys, pathlib
sys.path.append(str(pathlib.Path("../../src").resolve()))
from preprocess import preprocessing_rna
# %%
adata, count_X = preprocessing_rna(adata=adata, n_top_features=2000) 
# %%
pos_ = np.zeros([adata.n_obs, 2])
pos_[:,0] = adata.obs["pos_x"]
pos_[:,1] = adata.obs["pos_y"]
y = np.array(adata.obs["cell_labels"].cat.codes)
# %%
pos_.shape
# %%
# %%
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from preprocess import norm_adj
from scipy import spatial
# %%
A = kneighbors_graph(pos_, 20, mode="connectivity", metric="euclidean", include_self=True, n_jobs=-1)
A = A.toarray()    
A = sparse.csr_matrix(A)
A_n = norm_adj(A)
# %%
print(A.sum()) # 84420
# %%
A_self = sparse.identity(pos_.shape[0], format="csr")
A_self
# %%
print(A_self.sum()) # 4221
# %%
#? 如果args.knn等于0，那么它和单位阵A_n_self等价吗？
A_knn0 = kneighbors_graph(pos_, 0, mode="connectivity", metric="euclidean", include_self=True, n_jobs=-1)
A_knn0.sum()
#! 不行，如果传入0，kneighbors_graph会直接报错
# %%
#? 对单位阵进行norm_adj会发生什么？
A_self_n = norm_adj(sparse.csr_matrix(A_self.toarray()))
print(A_self_n.sum()) # 4221.0
#! 依旧是单位阵
# %%

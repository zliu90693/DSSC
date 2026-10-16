# %%
import numpy as np
# %%
anchors_ml = np.loadtxt("../../example_constraints/sample_151507_mlFromMarks_test.txt").astype(np.int)
anchors_ml
# %%
ml_ind1 = anchors_ml[:,0] - 1
ml_ind2 = anchors_ml[:,1] - 1 
ml_ind1 # 这是 ML 数据中的第一列，共两列
# %%
# 进入 fit
ml_pool1 = ml_ind1
ml_pool2 = ml_ind2
# %%
n_ml = 4000 # n_ml 是从 ml_pool1 池子里取出的采样数，默认 4000
# %%
print(ml_pool1.shape[0]) # 26203
# %%
k = min(n_ml, ml_pool1.shape[0])
print(k) # 4000
# %%
inds = np.random.choice(ml_pool1.shape[0], k, replace=False)
ml_ind1 = ml_pool1[inds]
ml_ind2 = ml_pool2[inds]
# %%
ml_ind1.shape # (4000,)
# %%

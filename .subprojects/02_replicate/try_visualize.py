# %%
import h5py, numpy as np, matplotlib.pyplot as plt
# %%
# data = h5py.File('../../data/sample_151507_anno.h5')
# # %%
# x, y, pos = np.array(data['X']), np.array(data['Y']), np.array(data['Pos'])
# if pos.shape[1] != 2: pos = pos.T 
# # %%
# f = np.where(y.astype(str) != 'NA')[0]
# y, pos = y[f], pos[f].astype(float)
# # %%
# pred = np.loadtxt('../../src/out_151507/pred.csv_5')
# assert len(pred) == len(pos) 
# # %%
# plt.scatter(pos[:,0], pos[:,1], c=pred, s=5, cmap='tab20')
# %%
def visualize_(
        h5_path: str, 
        pred_path: str
    ) -> None:
    data = h5py.File(h5_path, 'r') # 指定 'r'，否则运行 run_DSSC.py 时会导致 HDF5 文件锁冲突
    x, y, pos = np.array(data['X']), np.array(data['Y']), np.array(data['Pos'])
    if pos.shape[1] != 2: pos = pos.T 
    f = np.where(y.astype(str) != 'NA')[0]
    y, pos = y[f], pos[f].astype(float)
    pred = np.loadtxt(pred_path)
    assert len(pred) == len(pos) 
    plt.scatter(pos[:,0], pos[:,1], c=pred, s=5, cmap='tab20')
# %%
visualize_(
    '../../data/sample_151507_anno.h5',
    '../../src/out_151507/pred.csv_5'
)
# %%
visualize_(
    '../../data/sample_151507_anno.h5',
    '../../src/out_151507/pred.csv_1'
)
# %%
visualize_(
    '../../data/sample_151507_anno.h5',
    '../../src/out_151507/pred.csv_2'
)
# %%
visualize_(
    '../../data/osmFISH_cortex.h5',
    '../../src/out_osmFISH/pred.csv_5'
)
# %%

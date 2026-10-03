# DSSC
A model-based constrained deep learning clustering approach for spatial-resolved single-cell data  
![Model structure](./src/fig1_structure.png)  

# Dependencies in Python  
```bash
conda env create -f .env/dssc_4090.yml
```
Note that due to differences in GPU and CUDA core versions, the dependency versions used here differ from those in [the original repository](https://github.com/xianglin226/DSSC). Please refer to `.test/01_test_env.ipynb` for details on environment setup and troubleshooting.

All experiments of DSSC in this study are conducted on Nvidia 4090 (24GB) GPU.

#The input data should be in h5 format with:  
(1) "X" - count matrix  
(2) "Y" - true labels (if available)  
(3) "Pos" - spatial coordinate  
(4) "Genes" - feature names (Use to build constraints)  

# Dependencies in R
R 4.1.0  

Seurat 4.2.0  

cccd 1.5  

rhdf5 2.38.1  

ggplot2 3.3.6  

# Run DSSC 
1) Build constraints (See make_links_from_Markers.R)

```bash

```

2) Run DSSC (See run_DSSC.sh, then run.sh)  

```bash
cd src
chmod +x run.sh
export HDF5_USE_FILE_LOCKING=FALSE # If the project is located on a network drive or shared drive that does not support POSIX locks, explicitly set the environment variable.
./run.sh
```
Note: An error occurred during execution, preventing the proper processing of `out_osmFISH` (see `.logs/warn_err_all.log` for details). Investigation traced the issue to the `pos = pos.T` operation (originally on line 56, now line 57) in `run_DSSC.py`; this has now been fixed.

# Cite this work  
Lin, X., Gao, L., Whitener, N., Ahmed, A., & Wei, Z. (2022). A model-based constrained deep learning clustering approach for spatially resolved single-cell data. Genome Research, 32(10), 1906-1917.

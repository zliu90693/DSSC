# DSSC
A model-based constrained deep learning clustering approach for spatial-resolved single-cell data  
![Model structure](https://github.com/xianglin226/DSSC/blob/master/src/fig1_structure.png?raw=true)  

# Dependencies in Python  
```bash
conda env create -f .env/dssc_4090.yml
```
Note that due to differences in GPU and CUDA core versions, the dependency versions used here differ from those in the original repository. Please refer to `.test/test_env.ipynb` for details on environment setup and troubleshooting.

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
2) Run DSSC (See run_DSSC.sh, then run.sh)  

# Cite this work  
Lin, X., Gao, L., Whitener, N., Ahmed, A., & Wei, Z. (2022). A model-based constrained deep learning clustering approach for spatially resolved single-cell data. Genome Research, 32(10), 1906-1917.

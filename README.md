# SGMP
This is the code repository for paper "Representation Learning on Spatial Networks".
Zheng Zhang, Liang Zhao, Representation Learning on Spatial Networks, **NeurIPS 2021**

# Examples of running the models:

**1. training the SGMP model on BACE dataset:
- bash run_SGMP_BACE.sh

**2. training the SGMP model with sampling spanning trees on BACE dataset:
- bash run_SGMP_st_BACE.sh

**3. training the PointNet benchmark model on BACE dataset:
- bash run_PointNet_BACE.sh

**4. training the SGMP model on QM9 dataset for target 0 (mu):
- bash run_SGMP_QM9.sh

**5. training the SGMP model with sampling spanning trees on QM9 dataset for target 0 (mu):
- bash run_SGMP_st_QM9.sh

**6. training the PointNet benchmark model on QM9 dataset for target 0 (mu):
- bash run_PointNet_QM9.sh
 
**7. generate synthetic dataset
- cd data
- python build_synthetic_data.py

** The hyper-parameters can be modified in the bash scripts

** HCP brain network data can be acquired from https://www.humanconnectome.org/ since we cannot re-distribute it.

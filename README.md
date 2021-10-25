#This is the code for paper "Representation Learning on Spatial Networks"

#Examples of running the models:

#for training the SGMP model on BACE dataset:
%run bash run_SGMP_BACE.sh

#for training the SGMP model with sampling spanning trees on BACE dataset:
%run bash run_SGMP_st_BACE.sh

#for training the PointNet benchmark model on BACE dataset:
%run bash run_PointNet_BACE.sh

#for training the SGMP model on QM9 dataset for target 0 (mu):
%run bash run_SGMP_QM9.sh

#for training the SGMP model with sampling spanning trees on QM9 dataset for target 0 (mu):
%run bash run_SGMP_st_QM9.sh

#for training the PointNet benchmark model on QM9 dataset for target 0 (mu):
%run bash run_PointNet_QM9.sh
 
#for generate synthetic dataset
%run cd data
%run python build_synthetic_data.py

#All the parameters can be modified in the scripts
#The HCP brain network data can be acquired from https://www.humanconnectome.org/ since we cannot re-distribute it.

import torch
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import torch
import pandas as pd

def load_brain_data(data_dir='./data', structure='sc', target='ReadEng_Unadj', threshold=1e5, random_seed=12345):
    np.random.seed(random_seed)
    coor_PATH = os.path.join(data_dir, 'Brain', 'ROI_coordinates')
    graph_PATH = os.path.join(data_dir, 'Brain', 'SC_FC_dataset_0905')
    df = pd.read_csv(os.path.join(graph_PATH, 'targets.csv'))
    
    with open(os.path.join(coor_PATH, 'coordinates_mean.npy'), 'rb') as f:
        coordinates = np.load(f)

    task_list = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
    
    age_dict = {'22-25':0, '26-30':1, '31-35':2, '36+':3}
    
    dataset = []
    subjects_list = []
    for task in task_list:
        inputFile = 'SC_'+task+'_correlationFC.npz'
        data = np.load(os.path.join(graph_PATH, inputFile))
        if structure == 'fc':
            struc = data['fc'] if task != 'RESTINGSTATE' else data['rawFC']
        elif structure == 'sc':
            struc = data['sc'] if task != 'RESTINGSTATE' else data['rawSC']
        subjects = data['subjects']
        for i in np.random.permutation(len(subjects)):
            subject = subjects[i]
            if not subject in subjects_list:
                y_temp = df.loc[df['Subject']==subject, target].values[0]
                y0 = age_dict[y_temp]
                    
                item = Data(
                    x=torch.arange(68, dtype=torch.long),
                    pos=torch.tensor(coordinates, dtype=torch.float),
                    edge_index=torch.tensor(np.concatenate(np.nonzero((struc[i]>threshold)|(struc[i]<-threshold))).reshape(2,-1), dtype=torch.long),
                    edge_attr=torch.tensor(struc[i][(struc[i]>threshold)|(struc[i]<-threshold)].reshape(-1), dtype=torch.float),
                    y=torch.tensor([[y0]], dtype=torch.float),
                    name_struc=structure,
                )
                dataset.append(item)
                subjects_list.append(subject)
            
    return dataset
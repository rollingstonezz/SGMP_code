import numpy as np
import os
from tqdm import tqdm
from math import pi as PI
import time

import ase
import torch
import pickle as pkl
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, to_networkx

import networkx as nx
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.distance_measures import diameter as G_diameter
from networkx.algorithms.distance_measures import radius as G_radius
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix

from collections import Counter
import matplotlib.pyplot as plt

def S_eccentricity(pos):
    dis_matrix = distance_matrix(pos, pos)
    eccentricity = dis_matrix.max(axis=1)
    return eccentricity
def S_diameter(pos):
    eccentricity = S_eccentricity(pos)
    return np.max(eccentricity)
def S_radius(pos):
    eccentricity = S_eccentricity(pos)
    return np.min(eccentricity)

def build_synthetic_data(n, m, dim, D, p, L):   
    x = np.zeros(n, dtype=int)
    pos = np.random.random((n, dim)) * D # Random Given the position
    k_max=1 # initialize k_max
    k_list = [1, 1]
    edge_index = [[0],[1]]
    for i in range(2,n):
        # P_spatial
        distance = np.power(np.sum((pos[:i] - pos[i])**p, axis=-1), 1/p)
        P_spatial = np.exp(-distance/L)
        # P_graph
        P_graph = (np.array(k_list)) / (k_max)
        # P total
        P = P_spatial * P_graph

        # random sample
        flag = np.random.random(size=(i,)) < P
        #print(P)
        neighbors = np.nonzero(flag)[0]
        
        
        for j in neighbors:
            edge_index[0].append(j)
            edge_index[1].append(i)
            k_list[j] += 1
        k_list.append(len(neighbors))
        k_max = max(k_list)

    undirected_edge_index = [edge_index[0] + edge_index[1], edge_index[1] + edge_index[0]]
    x = torch.zeros((len(pos), 1), dtype=torch.float)
    pos=torch.tensor(pos, dtype=torch.float)
    edge_index=torch.tensor(undirected_edge_index, dtype=torch.long)
    edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=n)
    x, pos = x[mask],  pos[mask] 
    edge_attr = torch.zeros(len(edge_index[0]), dtype=torch.float)
    row, col = edge_index.numpy()
    distances = torch.norm(pos[row]-pos[col],dim=-1).numpy()
    edges = [(row[i], col[i], distances[i]) for i in range(len(row))]
    # build networkx graph
    temp = Data(x=x, edge_index=edge_index)
    G = to_networkx(temp)
    for row, col, weight in edges:
        G[row][col]['weight'] = weight
        
    y=torch.tensor([[average_clustering(G), S_diameter(pos), S_radius(pos), L]], dtype=torch.float)
    
    data = Data(
        x=x, 
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    return data, np.array(k_list)[mask.numpy()]

N = 20 # number of generated graphs
n=20
m=3
dim = 3 # spatial dimension 
D = 5.0 # the space limit for all nodes (rectangular space)
p = 2

synthetic_dataset = []
counter_all = Counter()

for num in tqdm(range(N)):
    for n in [15, 20, 25, 30]:
        for m in [2]:
            for D in np.arange(1.0, 10.1, 1.0):
                for L in [D*0.75, D*1.0, D*1.25, D*1.5]:
                        data, k_list = build_synthetic_data(n, m, dim, D, p, L)
                        counter = Counter(k_list)
                        counter_all += counter
                        synthetic_dataset.append(data)
                        
print('Number of graphs', len(synthetic_dataset))

if not os.path.exists('./synthetic'):
    os.makedirs('./synthetic')
with open('./synthetic/synthetic.pkl','wb') as file:
    pkl.dump(synthetic_dataset, file)
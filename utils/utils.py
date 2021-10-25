import numpy as np

import networkx as nx
from networkx.utils import UnionFind

from typing import Optional
import torch
from torch import Tensor

from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Optional

def random_spanning_tree(edge_index):
    r=torch.randperm(len(edge_index[0]), device=edge_index.device)
    row, col = edge_index[:,r].numpy()
    subtrees = UnionFind()
    spanning_edges = []
    i = 0

    while i < len(row):
        if subtrees[row[i]] != subtrees[col[i]]:
            subtrees.union(row[i], col[i])
            spanning_edges.append([row[i], col[i]])
        i += 1
    return spanning_edges

def scipy_spanning_tree(edge_index, num_nodes, num_edges):
    row, col = edge_index.numpy()
    cgraph = csr_matrix((np.random.random(num_edges) + 1, (row, col)), shape=(num_nodes, num_nodes))
    Tcsr = minimum_spanning_tree(cgraph)
    tree_row, tree_col = Tcsr.nonzero()
    spanning_edges = np.concatenate([[tree_row], [tree_col]]).T
    return spanning_edges
    
def build_spanning_tree_edge(edge_index, algo='union', num_nodes=None, num_edges=None):
    # spanning_edges
    if algo=='union':
        spanning_edges = random_spanning_tree(edge_index)
    elif algo=='scipy':
        spanning_edges = scipy_spanning_tree(edge_index, num_nodes, num_edges)
        
    spanning_edges = torch.tensor(spanning_edges, dtype=torch.long, device=edge_index.device).T
    spanning_edges_undirected = torch.stack(
        [
            torch.cat([spanning_edges[0], spanning_edges[1]]),
            torch.cat([spanning_edges[1], spanning_edges[0]]),
        ]
    )
    return spanning_edges_undirected


def add_self_loops(edge_index, edge_attr: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    N = num_nodes if num_nodes is not None else len(torch.unique(edge_index))

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        assert edge_attr.size(0) == edge_index.size(1)
        loop_attr = edge_attr.new_full((N, edge_attr.size(1)), fill_value)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_attr

def triplets(edge_index, num_nodes):
    row, col = edge_index  # i->j

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[col]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = row.repeat_interleave(num_triplets)
    idx_j = col.repeat_interleave(num_triplets)
    edx_1st = value.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    edx_2nd = adj_t_row.storage.value()
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  # Remove go back triplets.
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  # Remove repeat self loop triplets.
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  # Remove self-loop neighbors
    mask = ~(mask1 | mask2 | mask3) # 0 -> 0 -> 0 or # 0 -> 1 -> 2
    idx_i, idx_j, idx_k, edx_1st, edx_2nd = idx_i[mask], idx_j[mask], idx_k[mask], edx_1st[mask], edx_2nd[mask]
    
    # count real number of triplets
    num_triplets_real = torch.cumsum(num_triplets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_triplets, dim=0)-1]

    return torch.stack([idx_i, idx_j, idx_k]), num_triplets_real.to(torch.long), edx_1st, edx_2nd

def fourthplets(edge_index, triplets, num_nodes, edge_attr_index_1st, edge_attr_index_2nd):
    row, col = edge_index  # i->j
    i, j, k = triplets  # i->j->k

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[k]
    num_fourthlets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (i->j->k->p) for fourthlets.
    idx_i = i.repeat_interleave(num_fourthlets)
    idx_j = j.repeat_interleave(num_fourthlets)
    idx_k = k.repeat_interleave(num_fourthlets)
    edge_attr_index_1st = edge_attr_index_1st.repeat_interleave(num_fourthlets)
    edge_attr_index_2nd = edge_attr_index_2nd.repeat_interleave(num_fourthlets)
    idx_p = adj_t_row.storage.col()
    edge_attr_index_3rd = adj_t_row.storage.value()
    mask1 = (idx_i != idx_p) & (idx_j != idx_k) & (idx_j != idx_p) & (idx_k != idx_p) # 0 -> 1 -> 2 -> 3
    mask2 = (idx_i == idx_p) & (idx_j == idx_p) & (idx_k == idx_p) # 0 -> 0 -> 0 -> 0
    mask = mask1 | mask2
    
    idx_i, idx_j, idx_k, idx_p = idx_i[mask], idx_j[mask], idx_k[mask], idx_p[mask]
    edge_attr_index_1st, edge_attr_index_2nd, edge_attr_index_3rd = edge_attr_index_1st[mask], edge_attr_index_2nd[mask], edge_attr_index_3rd[mask]
    
    # count real number of fourthlets
    num_fourthlets_real = torch.cumsum(num_fourthlets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_fourthlets, dim=0)-1]
    
    return torch.stack([idx_i, idx_j, idx_k, idx_p]), num_fourthlets_real.to(torch.long), edge_attr_index_1st, edge_attr_index_2nd, edge_attr_index_3rd 

def find_higher_order_neighbors(edge_index, num_of_nodes, order=1):    
    edge_index_1st = edge_index
    
    if order==1:
        return edge_index_1st
    elif order==2:
        # find 2nd & 3rd neighbors
        edge_index_2nd, num_2nd_neighbors, edge_attr_index_1st, edge_attr_index_2nd = triplets(edge_index_1st, num_of_nodes)
        return edge_index_1st, edge_index_2nd, num_2nd_neighbors, edge_attr_index_1st, edge_attr_index_2nd
    elif order==3:
        # find 2nd & 3rd neighbors
        edge_index_2nd, num_2nd_neighbors, edge_attr_index_1st, edge_attr_index_2nd = triplets(edge_index_1st, num_of_nodes)
        edge_index_3rd, num_3rd_neighbors, edge_attr_index_1st, edge_attr_index_2nd, edge_attr_index_3rd = fourthplets(edge_index_1st, edge_index_2nd, num_of_nodes, edge_attr_index_1st, edge_attr_index_2nd)
        return edge_index_1st, edge_index_2nd, edge_index_3rd, num_2nd_neighbors, num_3rd_neighbors, edge_attr_index_1st, edge_attr_index_2nd, edge_attr_index_3rd
    else:
        raise NotImplementedError('We currently only support up to 3rd neighbors')
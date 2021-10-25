import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_scatter import scatter

def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))


def point_pair_features(pos_i: Tensor, pos_j: Tensor, normal_i: Tensor,
                        normal_j: Tensor) -> Tensor:
    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=1),
        get_angle(normal_i, pseudo),
        get_angle(normal_j, pseudo),
        get_angle(normal_i, normal_j)
    ], dim=1)

class PPFConv(MessagePassing):
    def __init__(self, local_nn, global_nn, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PPFConv, self).__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
    
    def forward(self, x, pos, edge_index):  # yapf: disable
        """"""
        epsilon = 1e-12 # for numerical stable
        normal = pos / (pos.norm(dim=-1).reshape(-1,1)+epsilon)
        out = self.propagate(edge_index, x=x, pos=pos, normal=normal)
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


    def message(self, x_j, pos_i, pos_j,
                normal_i, normal_j) -> Tensor:
        msg = point_pair_features(pos_i, pos_j, normal_i, normal_j)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)
    
class PPFNet(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', num_layers=3):
        super(PPFNet, self).__init__()
        self.readout = readout    
        self.node_lin = Sequential(
            Linear(input_channels_node, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.embedding = Embedding(100, hidden_channels)
        self.num_layers = num_layers
        self.local_nn = ModuleList()
        self.global_nn = ModuleList()
        for _ in range(self.num_layers):
            block = Sequential(
                Linear(hidden_channels+4, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.local_nn.append(block)
        for _ in range(self.num_layers):
            block = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.global_nn.append(block)
            
        self.convs = ModuleList()
        for i in range(self.num_layers):
            conv = PPFConv(self.local_nn[i], self.global_nn[i])
            self.convs.append(conv)
            
        self.lin1 = Linear(hidden_channels, hidden_channels//2)
        self.lin2 = Linear(hidden_channels//2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        for nn in self.local_nn:
            torch.nn.init.xavier_uniform_(nn[0].weight)
            nn[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(nn[2].weight)
            nn[2].bias.data.fill_(0)
        for nn in self.global_nn:
            torch.nn.init.xavier_uniform_(nn[0].weight)
            nn[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(nn[2].weight)
            nn[2].bias.data.fill_(0)
            
        torch.nn.init.xavier_uniform_(self.node_lin[0].weight)
        self.node_lin[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.node_lin[2].weight)
        self.node_lin[2].bias.data.fill_(0)

    def forward(self, x, pos, edge_index, batch):
        if x.dim() == 1:
            x = self.embedding(x)
        else:
            x = self.node_lin(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, pos, edge_index)
            x = x.relu()
        
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = scatter(x, batch, dim=0, reduce=self.readout)   
        
        return x    
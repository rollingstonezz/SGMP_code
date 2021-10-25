import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter
    
class PointConv(MessagePassing):
    def __init__(self, local_nn, global_nn, aggr='add'):

        super(PointConv, self).__init__(aggr=aggr)
        self.local_nn = local_nn
        self.global_nn = global_nn
        
    def forward(self, x, pos, edge_index):
        """"""            
        out = self.propagate(edge_index, x=x, pos=pos)
        return self.global_nn(out)

    def message(self, x_i, x_j, pos_i, pos_j):
        temp = torch.cat([x_i, x_j, pos_j], dim=-1)
        return self.local_nn(temp)

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.input_channels, self.output_channels,
                                           self.dim)
    
class PointNet(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', num_layers=3):
        super(PointNet, self).__init__()
        self.readout = readout    
        self.node_lin = Sequential(
            Linear(input_channels_node, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        self.local_nn = ModuleList()
        self.global_nn = ModuleList()
        for _ in range(self.num_layers):
            block = Sequential(
                Linear(hidden_channels*2+3, hidden_channels),
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
            conv = PointConv(self.local_nn[i], self.global_nn[i])
            self.convs.append(conv)
            
        self.embedding = Embedding(100, hidden_channels)
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
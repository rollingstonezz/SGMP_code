########################################################
###  modified from the version by pytorch-geometric  ###
########################################################
import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter
    
class GatedGraphConv(MessagePassing):
    def __init__(self, output_channels: int, num_layers: int, aggr: str = 'add', **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.output_channels = output_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, output_channels, output_channels))
        self.rnn = torch.nn.GRUCell(output_channels, output_channels)
        
        self.edge_weight = Linear(output_channels, 1)

        self.reset_parameters()    
        
    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.edge_weight.reset_parameters()
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.edge_weight.weight)
        self.edge_weight.bias.data.fill_(0)
        
    def forward(self,x, pos, edge_index):
        """"""
        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=None, size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else self.edge_weight(edge_weight) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
    
class GatedNet(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', num_layers=3):
        super(GatedNet, self).__init__()
        self.readout = readout    
        self.node_lin = Sequential(
            Linear(input_channels_node+3, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        self.convs = ModuleList()
        for i in range(self.num_layers):
            conv = GatedGraphConv(hidden_channels, num_layers=num_layers)
            self.convs.append(conv)
        
        self.lin1 = Linear(hidden_channels, hidden_channels//2)
        self.lin2 = Linear(hidden_channels//2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):            
        torch.nn.init.xavier_uniform_(self.node_lin[0].weight)
        self.node_lin[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.node_lin[2].weight)
        self.node_lin[2].bias.data.fill_(0)


    def forward(self, x, pos, edge_index, batch):
        x = torch.cat([x, pos], dim=1)
        x = self.node_lin(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, pos, edge_index)
            x = x.relu()
        
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = scatter(x, batch, dim=0, reduce=self.readout)   
        
        return x    
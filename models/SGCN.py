# benchmark sgcn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax

class SConv(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians):
        super(SConv, self).__init__(aggr='mean')
        self.mlp1 = Sequential(
            Linear(hidden_channels*2, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mlp2 = Sequential(
            Linear(hidden_channels*2, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mlp3 = Sequential(
            Linear(3, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )


    def forward(self, x, pos, edge_index):
        h = self.propagate(edge_index, x=pos, h=x)
        x = torch.cat([h, x], dim=1)
        x = self.mlp1(x)
        return x

    def message(self, x_i, x_j, h_j):
        dist = (x_j-x_i)
        spatial = self.mlp3(dist)
        temp = torch.cat([h_j, spatial], dim=1)
        return self.mlp2(temp)
    
class SGCN(torch.nn.Module):

    def __init__(self,input_channels_node=1, hidden_channels=128, output_channels=1, readout='add', num_layers=3):
        super(SGCN, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        
        self.readout = readout
        self.node_lin = Sequential(
            Linear(input_channels_node, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        self.interactions = ModuleList()
        for _ in range(num_layers):
            block = SConv(hidden_channels, num_gaussians=hidden_channels)
            self.interactions.append(block)
            
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):            
        torch.nn.init.xavier_uniform_(self.node_lin[0].weight)
        self.node_lin[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.node_lin[2].weight)
        self.node_lin[2].bias.data.fill_(0)
        
    def forward(self, x, pos, edge_index, batch):
        
        x = self.node_lin(x)
        for block in self.interactions:
            x = block(x, pos, edge_index)
            x = x.relu()
            
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        out = scatter(x, batch, dim=0, reduce=self.readout)        
    
        return out  
    
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
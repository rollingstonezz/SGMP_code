########################################################
###  modified from the version by pytorch-geometric  ###
########################################################
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
    
class GATConv(MessagePassing): 
    def __init__(self, input_channels_node: int, output_channels: int, heads: int = 1, negative_slope: float = 0.1,  **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.input_channels_node = input_channels_node
        self.output_channels = output_channels
        self.heads = heads
        self.negative_slope = negative_slope
                
        self.lin_l = Linear(input_channels_node, heads * output_channels, bias=False)
        self.lin_r = self.lin_l

        self.att_l = Parameter(torch.Tensor(1, heads, output_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, output_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_l.weight)
        torch.nn.init.xavier_uniform_(self.att_l)
        torch.nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, pos, edge_index):
        H, C = self.heads, self.output_channels

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin_l(x).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r))
        out = out.mean(dim=1)
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.input_channels_node, self.output_channels, self.heads)
    
class GATNet(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', num_layers=3):
        super(GATNet, self).__init__()
        self.readout = readout    
        self.node_lin = Sequential(
            Linear(input_channels_node+3, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        self.convs = ModuleList()
        for i in range(self.num_layers):
            conv = GATConv(hidden_channels, hidden_channels, hidden_channels)
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
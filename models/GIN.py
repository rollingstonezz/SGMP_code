########################################################
###  modified from the version by pytorch-geometric  ###
########################################################
import torch
from torch import Tensor
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
    
class GINConv(MessagePassing): 
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], pos: Tensor, edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j 

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
    
class GINNet(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', eps=0., num_layers=3):
        super(GINNet, self).__init__()
        self.readout = readout    
        self.node_lin = Sequential(
            Linear(input_channels_node+3, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.num_layers = num_layers
        
        self.mlp = ModuleList()
        for _ in range(self.num_layers):
            block = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.mlp.append(block)
            
        self.convs = ModuleList()
        for i in range(self.num_layers):
            conv = GINConv(nn=self.mlp[i], eps=eps)
            self.convs.append(conv)
            
        self.lin1 = Linear(hidden_channels, hidden_channels//2)
        self.lin2 = Linear(hidden_channels//2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        for nn in self.mlp:
            torch.nn.init.xavier_uniform_(nn[0].weight)
            nn[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(nn[2].weight)
            nn[2].bias.data.fill_(0)
            
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
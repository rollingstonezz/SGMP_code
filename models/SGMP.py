from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch import Tensor
from torch_scatter import scatter

def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SGMP(torch.nn.Module):

    def __init__(self, input_channels_node=None, hidden_channels=128, 
                 output_channels=1, num_interactions=3,
                 num_gaussians=(50,6,12), cutoff=10.0,
                 readout='add'):
        super(SGMP, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        
        self.input_channels_node = input_channels_node
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.readout = readout
        # the gaussian expansion here is used to help quicker converge
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians[0])
        self.theta_expansion = GaussianSmearing(0.0, PI, num_gaussians[1])
        self.phi_expansion = GaussianSmearing(0.0, 2*PI, num_gaussians[2])
        self.embedding = Sequential(
            Linear(input_channels_node, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = SPNN(hidden_channels, num_gaussians, self.distance_expansion, self.theta_expansion, self.phi_expansion, input_channels=hidden_channels)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = torch.nn.ReLU()
        self.lin2 = Linear(hidden_channels // 2, output_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        for block in self.interactions:
            block.reset_parameters()
            
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        
    def forward(self, x, pos, batch, edge_index_3rd=None):
        x = self.embedding(x)
        
        distances = {}
        thetas = {}
        phis = {}
        i, j, k, p = edge_index_3rd
        i_to_j_dis = (pos[j] - pos[i]).norm(p=2, dim=1)
        k_to_j_dis = (pos[k] - pos[j]).norm(p=2, dim=1)
        p_to_j_dis = (pos[p] - pos[j]).norm(p=2, dim=1)
        distances[1] = i_to_j_dis
        distances[2] = k_to_j_dis
        distances[3] = p_to_j_dis
        theta_ijk = get_angle(pos[j] - pos[i], pos[k] - pos[j])
        theta_ijp = get_angle(pos[j] - pos[i], pos[p] - pos[j])
        thetas[1] = theta_ijk
        thetas[2] = theta_ijp

        v1 = torch.cross(pos[j] - pos[i], pos[k] - pos[j], dim=1)
        v2 = torch.cross(pos[j] - pos[i], pos[p] - pos[j], dim=1)
        phi_ijkp = get_angle(v1, v2)
        vn = torch.cross(v1, v2, dim=1)
        flag = torch.sign((vn * (pos[j]- pos[i])).sum(dim=1))
        phis[1] = phi_ijkp * flag
            
        for interaction in self.interactions:
            x = x + interaction(
                x,
                distances,
                thetas,
                phis,
                edge_index_3rd,
               )
            
        
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=x.device)
            
        x = scatter(x, batch, dim=0, reduce=self.readout)  
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        
        return x    
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_interactions})')
    

class SPNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, distance_expansion, theta_expansion, phi_expansion, input_channels=None, readout='add'):
        super(SPNN, self).__init__()
        
        self.readout = readout
        self.distance_expansion = distance_expansion
        self.theta_expansion = theta_expansion
        self.phi_expansion = phi_expansion
        
        self.mlps_dist = ModuleList()
        mlp_1st = Sequential(
                Linear(num_gaussians[0], hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
        mlp_2nd = Sequential(
                Linear(num_gaussians[0]+num_gaussians[1], hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            ) 
        mlp_3rd = Sequential(
                Linear(num_gaussians[0]+num_gaussians[1]+num_gaussians[2], hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
        self.mlps_dist.append(mlp_1st)
        self.mlps_dist.append(mlp_2nd)
        self.mlps_dist.append(mlp_3rd)
            
        self.combine = Sequential(
                Linear(hidden_channels*7, hidden_channels*4),
                torch.nn.ReLU(),
                Linear(hidden_channels*4, hidden_channels*2),
                torch.nn.ReLU(),
                Linear(hidden_channels*2, hidden_channels),
            )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(3):
            torch.nn.init.xavier_uniform_(self.mlps_dist[i][0].weight)
            self.mlps_dist[i][0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.combine[0].weight)
        self.combine[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.combine[2].weight)
        self.combine[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.combine[4].weight)
        self.combine[4].bias.data.fill_(0)
            
    def forward(self,
            x, 
            distances,
            thetas,
            phis,
            edge_index_3rd=None,
           ):
        i, j, k, p = edge_index_3rd

        node_attr_0st = x[i]
        node_attr_1st = x[j]
        node_attr_2nd = x[k]
        node_attr_3rd = x[p]
        geo_encoding_1st = self.mlps_dist[0](
            self.distance_expansion(distances[1])
         )
        geo_encoding_2nd = self.mlps_dist[1](
            torch.cat([self.distance_expansion(distances[2]), self.theta_expansion(thetas[1])], dim=-1)
         )
        geo_encoding_3rd = self.mlps_dist[2](
            torch.cat([self.distance_expansion(distances[3]), self.theta_expansion(thetas[2]), self.phi_expansion(phis[1])], dim=-1)
         )

        concatenated_vector = torch.cat(
            [
                node_attr_0st,
                node_attr_1st,
                node_attr_2nd,
                node_attr_3rd,
                geo_encoding_1st, 
                geo_encoding_2nd, 
                geo_encoding_3rd, 
            ],
            dim=-1
        )            
        x = self.combine(concatenated_vector)

        # aggregate
        x = scatter(x, i, dim=0, reduce=self.readout)
        
        return x
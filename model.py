# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import ModuleList

###############################################################################
#                         STAN: Spatio-Temporal Attention Network           #
#-----------------------------------------------------------------------------
# Key Fix: WeightedSumConv now handles empty edge_index to prevent returning
# integers. If adjacency is empty, we return x directly.
###############################################################################

class WeightedSumConv(MessagePassing):
    """
    A message-passing layer that multiplies source node features by learned 
    attention weights and sums the results at the target node.
    """
    def __init__(self):
        super(WeightedSumConv, self).__init__(aggr='add')  # sum aggregation

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: [N, d], node features
        :param edge_index: [2, E], adjacency (src, dst)
        :param edge_attr: [E, 1], attention scores
        :return: [N, d], updated node features
        """
        # Key fix: If no edges, just return the input x to avoid empty aggregator
        if edge_index.size(1) == 0:
            return x

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        :param x_j: [E, d], features of source nodes
        :param edge_attr: [E, 1], attention weights
        :return: [E, d], weighted messages
        """
        return x_j * edge_attr

    def update(self, aggr_out):
        """
        :param aggr_out: [N, d], aggregated output from neighbors
        :return: [N, d], updated node features
        """
        return aggr_out

class GATLayer(nn.Module):
    """
    Single-head Graph Attention layer.
    - Applies a linear transform
    - Computes attention coefficients
    - Aggregates neighbor features weighted by attention
    """
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.conv = WeightedSumConv()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weights of the layer.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
        if self.attn_fc.bias is not None:
            nn.init.constant_(self.attn_fc.bias, 0)

    def forward(self, adj, h):
        """
        :param adj: [2, E] edge_index
        :param h: [N, in_dim] node features
        :return: [N, out_dim], updated features
        """
        z = self.fc(h)  # [N, out_dim]
        src = adj[0]    # [E]
        dst = adj[1]    # [E]

        # Compute attention
        att_feat = torch.cat((z[src], z[dst]), dim=-1)  # [E, 2*out_dim]
        att_edge = F.leaky_relu(self.attn_fc(att_feat)) # [E, 1]
        # Extra step: apply sigmoid or clamp to handle extreme attention values
        att_edge = torch.sigmoid(att_edge)

        # Weighted sum
        output = self.conv(z, adj, att_edge)  # [N, out_dim]
        return output

class MultiHeadGATLayer(nn.Module):
    """
    Wraps multiple single-head GAT layers into one multi-head representation.
    """
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = ModuleList([GATLayer(in_dim, out_dim) for _ in range(num_heads)])
        self.merge = merge

    def forward(self, adj, h):
        head_outs = [attn_head(adj, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=-1)  # [N, out_dim * num_heads]
        else:
            return torch.mean(torch.stack(head_outs), dim=0)

class STAN(nn.Module):
    """
    Spatio-Temporal Attention Network (STAN).
    - GAT -> GAT -> GRU
    - Data-driven + Physical (SIR) heads
    """
    def __init__(self, 
                 num_nodes, 
                 num_features, 
                 num_timesteps_input, 
                 num_timesteps_output, 
                 population=1e10, 
                 gat_dim1=32, 
                 gat_dim2=32, 
                 gru_dim=32, 
                 num_heads=1, 
                 device='cpu'):
        super(STAN, self).__init__()
        self.n_nodes = num_nodes
        self.nfeat = num_features // num_timesteps_input
        self.history = num_timesteps_input
        self.horizon = num_timesteps_output
        self.pop = population
        
        # 1) GAT layers
        self.layer1 = MultiHeadGATLayer(self.nfeat*self.history, gat_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(gat_dim1*num_heads, gat_dim2, 1)

        # 2) GRU
        self.gru = nn.GRU(gat_dim2, gru_dim, batch_first=True)

        # 3) Data-driven outputs
        self.nn_res_I = nn.Linear(gru_dim+2, self.horizon)
        self.nn_res_R = nn.Linear(gru_dim+2, self.horizon)

        # 4) Physical (SIR) outputs
        self.nn_res_sir = nn.Linear(gru_dim+2, 2)

        self.device = device

    def forward(self, X, adj, states, N=None):
        """
        :param X: [B, T, nLoc, F]
        :param adj: [2, E]
        :param states: [B*nLoc, 2] (I, R)
        :param N: [nLoc, 1] population
        :return: (predictions, phy_predictions)
                 - predictions: [B*nLoc, horizon, 2] (I_pred, R_pred)
                 - phy_predictions: [B*nLoc, horizon, 2] (I_phy, R_phy)
        """
        B, T, nLoc, F = X.size()

        # Last time step differences
        last_diff_I = X[:, -1, :, 1]  # [B, nLoc]
        last_diff_R = X[:, -1, :, 2]  # [B, nLoc]

        # Flatten for GAT: [B*nLoc, T*F]
        X = X.permute(0, 2, 1, 3).contiguous()  
        X = X.view(B*nLoc, T*F)

        # GAT layers
        cur_h = self.layer1(adj, X)
        cur_h = cur_h.float().to(self.device)
        elu = nn.ELU()
        cur_h = elu(cur_h)

        cur_h = self.layer2(adj, cur_h)
        cur_h = cur_h.float().to(self.device)
        cur_h = elu(cur_h)

        # GRU
        cur_h = cur_h.unsqueeze(1)  # [B*nLoc, 1, gat_dim2]
        h_out, _ = self.gru(cur_h)  # [B*nLoc, 1, gru_dim]
        h_out = h_out.squeeze(1)    # [B*nLoc, gru_dim]

        # Concat daily deltas
        hc = torch.cat([
            h_out,
            last_diff_I.view(B*nLoc, 1),
            last_diff_R.view(B*nLoc, 1)
        ], dim=1)  # [B*nLoc, gru_dim + 2]

        # Data-driven predictions
        pred_I = self.nn_res_I(hc).unsqueeze(-1)  # [B*nLoc, horizon, 1]
        pred_R = self.nn_res_R(hc).unsqueeze(-1)  # [B*nLoc, horizon, 1]

        # Physical SIR predictions
        alpha_beta = self.nn_res_sir(hc)          # [B*nLoc, 2]
        alpha = torch.sigmoid(alpha_beta[:, 0]).unsqueeze(1)  # [B*nLoc,1]
        beta  = torch.sigmoid(alpha_beta[:, 1]).unsqueeze(1)  # [B*nLoc,1]

        # Handle population
        if N is None:
            # Use global pop if not provided
            N = torch.tensor([self.pop], dtype=torch.float32, device=X.device).repeat(B*nLoc,1)  # [B*nLoc,1]
        else:
            if isinstance(N, torch.Tensor):
                if N.dim() == 2 and N.size(0) == nLoc:
                    N = N.repeat(B,1)  # [B*nLoc,1]
                else:
                    raise ValueError(f"Population tensor N has unexpected shape: {N.shape}")
            else:
                # Scalar
                N = torch.tensor([N], dtype=torch.float32, device=X.device).repeat(B*nLoc,1)  # [B*nLoc,1]

        # Physical Model Forward
        phy_I_list = []
        phy_R_list = []

        last_I = states[:,0]
        last_R = states[:,1]

        for i in range(self.horizon):
            last_S = N.squeeze(-1) - last_I - last_R
            dI = alpha.squeeze(1)*last_I*(last_S/N.squeeze(-1)) - beta.squeeze(1)*last_I
            dR = beta.squeeze(1)*last_I

            phy_I_list.append(dI)
            phy_R_list.append(dR)

            # Euler integration
            last_I = last_I + dI.detach()
            last_R = last_R + dR.detach()

        phy_I = torch.stack(phy_I_list, dim=1).unsqueeze(-1)  # [B*nLoc, horizon,1]
        phy_R = torch.stack(phy_R_list, dim=1).unsqueeze(-1)  # [B*nLoc, horizon,1]

        return (
            torch.cat([pred_I, pred_R], dim=-1),
            torch.cat([phy_I, phy_R], dim=-1)
        )

    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

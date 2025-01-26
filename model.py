# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class WeightedSumConv(MessagePassing):
    def __init__(self):
        super(WeightedSumConv, self).__init__(aggr='add')  # 'add' means sum aggregation
    
    def forward(self, x, edge_index, edge_attr):
        # Trigger message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return x_j * edge_attr  # message = source node features * edge weight
    
    def update(self, aggr_out):
        return aggr_out  # aggregated output


class GATLayer(nn.Module):
    """
    Single-head GAT layer that expects:
      - adj: shape [2, E] (the global edge_index)
      - h:   shape [N, in_dim]
    Converts edge_index into a sparse adjacency matrix internally.
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
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, adj, h):
        """
        :param adj:  [2, E] edge_index
        :param h:    [N, in_dim]
        :return:     [N, out_dim]
        """
        z = self.fc(h)  # [N, out_dim]

        # Compute attention on edges
        src = adj[0]
        dst = adj[1]
        att_feat = torch.cat((z[src], z[dst]), dim=-1)  # [E, 2*out_dim]
        att_edge = F.leaky_relu(self.attn_fc(att_feat))  # [E, 1]
        # Removed .squeeze(-1) to maintain shape [E, 1]

        # Weighted sum
        output = self.conv(z, adj, att_edge)
        return output


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, adj, h):
        head_outs = [attn_head(adj, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=-1)  # [N, out_dim * num_heads]
        else:
            # If merge='mean', for example
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)


class STAN(nn.Module):
    """
    Spatio-Temporal Attention Network (STAN)
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
        self.nfeat = num_features // num_timesteps_input  # Adjust feature dimension
        self.history = num_timesteps_input
        self.horizon = num_timesteps_output
        self.pop = population
        
        self.layer1 = MultiHeadGATLayer(self.nfeat * self.history, gat_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(gat_dim1 * num_heads, gat_dim2, 1)

        self.gru = nn.GRU(gat_dim2, gru_dim, batch_first=True)

        self.nn_res_I = nn.Linear(gru_dim+2, self.horizon)
        self.nn_res_R = nn.Linear(gru_dim+2, self.horizon)
        self.nn_res_sir = nn.Linear(gru_dim+2, 2)

        self.gru_dim = gru_dim
        self.device = device

    def forward(self, X, adj, states, N=None):
        """
        :param X:      [B, T, num_nodes, num_features]
        :param adj:    [2, E] edge_index
        :param states: [B*num_nodes, 2]
        :param N:      [num_nodes, 1] population
        """
        B, T, nLoc, F = X.size()
        # Extract last differences from the final time step
        last_diff_I = X[:, -1, :, 1]  # [B, nLoc]
        last_diff_R = X[:, -1, :, 2]  # [B, nLoc]

        # Flatten X from [B, T, nLoc, F] => [B*nLoc, T*F]
        X = X.permute(0, 2, 1, 3).contiguous()      # [B, nLoc, T, F]
        X = X.view(B * nLoc, T * F)               # [B*nLoc, T*F]

        # GAT layers
        cur_h = self.layer1(adj, X)               # [B*nLoc, gat_dim1 * num_heads]
        cur_h = F.elu(cur_h)
        cur_h = self.layer2(adj, cur_h)           # [B*nLoc, gat_dim2]
        cur_h = F.elu(cur_h)

        # Prepare for GRU: [B*nLoc, 1, gat_dim2]
        cur_h = cur_h.unsqueeze(1)
        h_out, _ = self.gru(cur_h)                # [B*nLoc, 1, gru_dim]
        h_out = h_out.squeeze(1)                   # [B*nLoc, gru_dim]

        # Concatenate with last differences
        hc = torch.cat([
            h_out,
            last_diff_I.view(B * nLoc, 1),
            last_diff_R.view(B * nLoc, 1)
        ], dim=1)  # [B*nLoc, gru_dim + 2]

        # Predictions (data-driven)
        pred_I = self.nn_res_I(hc).unsqueeze(-1)  # [B*nLoc, horizon, 1]
        pred_R = self.nn_res_R(hc).unsqueeze(-1)  # [B*nLoc, horizon, 1]

        # Physical predictions
        alpha_beta = self.nn_res_sir(hc)          # [B*nLoc, 2]
        alpha = torch.sigmoid(alpha_beta[:, 0]).unsqueeze(1)  # [B*nLoc, 1]
        beta = torch.sigmoid(alpha_beta[:, 1]).unsqueeze(1)   # [B*nLoc, 1]

        if N is None:
            # Fallback to global population
            N = self.pop
        else:
            # Ensure N is replicated for the batch
            if isinstance(N, torch.Tensor) and N.dim() == 2 and N.size(0) == nLoc:
                N = N.repeat(B, 1)                # [B*nLoc, 1]
            elif not isinstance(N, torch.Tensor):
                # Make it a tensor and replicate
                N = torch.tensor([N], dtype=torch.float32, device=X.device)
                N = N.repeat(B * nLoc, 1)

        # Initialize lists for physical predictions
        phy_I = []
        phy_R = []

        # Initialize last_I and last_R
        last_I = states[:, 0]                       # [B*nLoc]
        last_R = states[:, 1]                       # [B*nLoc]

        for i in range(self.horizon):
            last_S = N.squeeze(-1) - last_I - last_R

            dI = alpha.squeeze(1) * last_I * (last_S / N.squeeze(-1)) - beta.squeeze(1) * last_I
            dR = beta.squeeze(1) * last_I

            phy_I.append(dI)
            phy_R.append(dR)

            last_I = last_I + dI.detach()
            last_R = last_R + dR.detach()

        phy_I = torch.stack(phy_I, dim=1).unsqueeze(-1)  # [B*nLoc, horizon, 1]
        phy_R = torch.stack(phy_R, dim=1).unsqueeze(-1)  # [B*nLoc, horizon, 1]

        return (
            torch.cat([pred_I, pred_R], dim=-1),   # [B*nLoc, horizon, 2]
            torch.cat([phy_I, phy_R], dim=-1)      # [B*nLoc, horizon, 2]
        )

    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

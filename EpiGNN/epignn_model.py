# EpiGNN/epignn_model.py

#!/usr/bin/env python3
"""
epignn_model.py
---------------
Contains the EpiGNN model, Graph Learner, GAT layer classes, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGATLayer(nn.Module):
    """
    A Multi-Head Graph Attention Layer as described in the GAT paper.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h, adj):
        """
        Forward pass for the Multi-Head GAT layer.

        Parameters:
        - h: (batch_size, num_nodes, in_features)
        - adj: (batch_size, num_nodes, num_nodes)

        Returns:
        - h_prime: (batch_size, num_nodes, num_heads * out_features)
        """
        batch_size, num_nodes, _ = h.size()
        Wh = self.W(h)  # (batch_size, num_nodes, num_heads * out_features)
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, self.out_features)  # (batch_size, num_nodes, num_heads, out_features)
        Wh = Wh.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_nodes, out_features)

        # Compute attention scores
        a_input = torch.cat([Wh.unsqueeze(3).repeat(1, 1, 1, num_nodes, 1),
                             Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)], dim=-1)  # (batch_size, num_heads, num_nodes, num_nodes, 2*out_features)
        # Compute e_ij for each head
        e = self.leakyrelu((a_input * self.a.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum(dim=-1))  # (batch_size, num_heads, num_nodes, num_nodes)

        # Masked attention: set e_ij = -inf if no edge
        e = e.masked_fill(adj.unsqueeze(1) == 0, float("-inf"))

        # Softmax to get attention coefficients
        attention = torch.softmax(e, dim=-1)  # (batch_size, num_heads, num_nodes, num_nodes)
        attention = self.dropout(attention)

        # Compute the attention-weighted sum of node features
        h_prime = torch.matmul(attention, Wh)  # (batch_size, num_heads, num_nodes, out_features)

        # Concatenate all heads
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_nodes, num_heads, out_features)
        h_prime = h_prime.view(batch_size, num_nodes, self.num_heads * self.out_features)  # (batch_size, num_nodes, num_heads * out_features)

        return h_prime


class GraphLearner(nn.Module):
    """
    Learns an adjacency matrix from node embeddings.
    """
    def __init__(self, hidden_dim, tanhalpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, emb):
        """
        Forward pass to learn adjacency matrix.

        Parameters:
        - emb: (batch_size, num_nodes, hidden_dim)

        Returns:
        - adj: (batch_size, num_nodes, num_nodes)
        """
        n1 = torch.tanh(self.alpha * self.linear1(emb))  # (batch_size, num_nodes, hidden_dim)
        n2 = torch.tanh(self.alpha * self.linear2(emb))  # (batch_size, num_nodes, hidden_dim)

        adj = (torch.bmm(n1, n2.transpose(1, 2)) - torch.bmm(n2, n1.transpose(1, 2))) * self.alpha  # (batch_size, num_nodes, num_nodes)
        adj = torch.relu(torch.tanh(adj))  # Ensure non-negative

        # Add self-connections
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)  # (batch_size, num_nodes, num_nodes)
        adj = adj + eye

        # Clamp to ensure numerical stability
        adj = torch.clamp(adj, min=1e-6, max=1.0)

        return adj


class ConvBranch(nn.Module):
    """
    A convolutional branch with optional pooling.
    """
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor=2, hidP=1, isPool=True):
        super(ConvBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

        # Initialize weights
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        """
        Forward pass for the convolutional branch.

        Parameters:
        - x: (batch_size, in_channels, T, m)

        Returns:
        - x: (batch_size, out_channels * hidP, m)
        """
        x = self.conv(x)  # (batch_size, out_channels, T_out, m)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, "pooling"):
            x = self.pooling(x)  # (batch_size, out_channels, hidP, m)
        bs = x.size(0)
        x = x.view(bs, -1, x.size(-1))  # (batch_size, out_channels * hidP, m)
        return self.activate(x)


class RegionAwareConv(nn.Module):
    """
    Combines local, periodic, and global convolution branches for spatiotemporal features.
    """
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.conv_l1 = ConvBranch(m, nfeat, k, kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m, nfeat, k, kernel_size=5, dilation_factor=1, hidP=hidP)
        self.conv_p1 = ConvBranch(m, nfeat, k, kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m, nfeat, k, kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_g  = ConvBranch(m, nfeat, k, kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        Forward pass for RegionAwareConv.

        Parameters:
        - x: (batch_size, num_features, T, m)

        Returns:
        - x: (batch_size, m, out_features)
        """
        xl1 = self.conv_l1(x)  # (batch_size, k * hidP, m)
        xl2 = self.conv_l2(x)  # (batch_size, k * hidP, m)
        x_local = torch.cat([xl1, xl2], dim=1)  # (batch_size, 2 * k * hidP, m)

        xp1 = self.conv_p1(x)  # (batch_size, k * hidP, m)
        xp2 = self.conv_p2(x)  # (batch_size, k * hidP, m)
        x_period = torch.cat([xp1, xp2], dim=1)  # (batch_size, 2 * k * hidP, m)

        xg = self.conv_g(x)  # (batch_size, k, m)

        xcat = torch.cat([x_local, x_period, xg], dim=1)  # (batch_size, 4 * k * hidP + k, m)
        return self.activate(xcat).permute(0, 2, 1)  # (batch_size, m, 4 * k * hidP + k)


class EpiGNN(nn.Module):
    """
    EpiGNN model with optional GAT layers, GraphLearner, etc.
    """
    def __init__(
        self,
        num_nodes,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        k=8,
        hidA=32,
        hidR=40,
        hidP=1,
        n_layer=3,
        num_heads=4,
        dropout=0.5,
        device="cpu"
    ):
        super(EpiGNN, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        self.backbone = RegionAwareConv(nfeat=num_features, P=num_timesteps_input, m=num_nodes, k=k, hidP=hidP)

        self.WQ = nn.Linear(hidR := hidR, hidA)
        self.WK = nn.Linear(hidR, hidA)
        self.t_enc = nn.Linear(1, hidR)
        self.s_enc = nn.Linear(1, hidR)

        self.d_gate = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        self.graphGen = GraphLearner(hidR)

        self.GATLayers = nn.ModuleList([
            MultiHeadGATLayer(hidR, hidR // num_heads, num_heads=num_heads, dropout=dropout)
            for _ in range(n_layer)
        ])

        gat_output_dim = (hidR // num_heads) * num_heads * n_layer
        self.output = nn.Linear(gat_output_dim + hidR, num_timesteps_output)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, adjacency_type="static"):
        """
        Forward pass for EpiGNN.

        Parameters:
        - X: (batch_size, T_in, m, feats)
        - adj: (batch_size, m, m)
        - adjacency_type: "static", "dynamic", or "hybrid"

        Returns:
        - res: (batch_size, T_out, m)
        """
        # Reshape X for backbone: (batch_size, feats, T_in, m)
        X_reshaped = X.permute(0, 3, 1, 2)
        temp_emb = self.backbone(X_reshaped)  # (batch_size, m, out_features)

        query = self.WQ(temp_emb)  # (batch_size, m, hidA)
        key = self.WK(temp_emb)    # (batch_size, m, hidA)

        attn = torch.bmm(query, key.transpose(1, 2))  # (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)  # (batch_size, m, 1)
        t_enc = self.t_enc(attn)  # (batch_size, m, hidR)

        d = torch.sum(adj, dim=2).unsqueeze(2)  # (batch_size, m, 1)
        s_enc = self.s_enc(d)  # (batch_size, m, hidR)

        feat_emb = temp_emb + t_enc + s_enc  # (batch_size, m, hidR)
        learned_adj = self.graphGen(feat_emb)  # (batch_size, m, m)

        # Combine adjacency matrices based on type
        if adjacency_type == "static":
            combined_adj = adj
        elif adjacency_type == "dynamic":
            combined_adj = learned_adj
        elif adjacency_type == "hybrid":
            combined_adj = adj + learned_adj
        else:
            raise ValueError("Invalid adjacency_type: static|dynamic|hybrid")

        laplace_adj = getLaplaceMat(X.size(0), self.num_nodes, combined_adj)

        node_state = feat_emb  # (batch_size, m, hidR)
        gat_outputs = []
        for gat in self.GATLayers:
            node_state = gat(node_state, laplace_adj)  # (batch_size, m, num_heads * out_features)
            node_state = F.elu(node_state)
            gat_outputs.append(node_state)

        gat_cat = torch.cat(gat_outputs, dim=-1)  # (batch_size, m, num_heads * out_features * n_layer)
        node_state_all = torch.cat([gat_cat, feat_emb], dim=-1)  # (batch_size, m, combined_features)
        res = self.output(node_state_all)  # (batch_size, m, T_out)
        return res.transpose(1, 2)  # (batch_size, T_out, m)


def getLaplaceMat(bs, m, adj):
    """
    Computes the Laplacian matrix.

    Parameters:
    - bs (int): Batch size.
    - m (int): Number of nodes.
    - adj (torch.Tensor): Adjacency matrix (batch_size, m, m).

    Returns:
    - laplace (torch.Tensor): Normalized Laplacian matrix (batch_size, m, m).
    """
    eye = torch.eye(m, device=adj.device).unsqueeze(0).repeat(bs, 1, 1)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)  # (batch_size, m)
    deg_inv = 1.0 / (deg + 1e-12)  # Avoid division by zero
    deg_inv_mat = eye * deg_inv.unsqueeze(2)  # (batch_size, m, m)
    laplace = torch.bmm(deg_inv_mat, adj_bin)  # (batch_size, m, m)
    return laplace

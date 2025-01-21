# EpiGNN/epignn_model_variant.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from EpiGNN.epignn_model import getLaplaceMat  # or wherever getLaplaceMat is defined

class GraphConvLayer(nn.Module):
    """
    Basic Graph Convolutional Layer with ELU activation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()  # Using ELU activation
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1.0 / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        """
        feature: (batch_size, m, in_features)
        adj: (batch_size, m, m)
        """
        support = torch.matmul(feature, self.weight)   # (batch_size, m, out_features)
        output = torch.bmm(adj, support)               # (batch_size, m, out_features)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

class GraphLearner(nn.Module):
    """
    Learns adjacency matrix via attention-like node embeddings.
    """
    def __init__(self, hidden_dim, tanhalpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        """
        embedding: (batch_size, m, hidR)
        """
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))
        adj = (torch.bmm(nodevec1, nodevec2.transpose(1, 2))
               - torch.bmm(nodevec2, nodevec1.transpose(1, 2)))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    """
    Single branch for RegionAwareConv (Conv2D + optional pooling).
    """
    def __init__(self,
                 m: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation_factor: int = 2,
                 hidP: int = 1,
                 isPool: bool = True):
        super(ConvBranch, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation_factor, 1)
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, in_channels, T, m)
        """
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))  # (batch_size, out_channels*hidP, m)
        return self.activate(x)

class RegionAwareConv(nn.Module):
    """
    Combines local, period, and global convolution branches for spatiotemporal features.
    """
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        # Local Convs
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=1, hidP=hidP)
        # Period Convs
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        # Global Conv
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                 kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, num_features, T, m)
        """
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)

        x_global = self.conv_g(x)

        x_cat = torch.cat([x_local, x_period, x_global], dim=1)
        return self.activate(x_cat).permute(0, 2, 1)  # (batch_size, m, combined_features)

class EpiGNNVariant(nn.Module):
    """
    EpiGNN Variant using GraphConvLayer instead of GAT layers, 
    with adjacency types: static, dynamic, or hybrid.
    """
    def __init__(
        self,
        num_nodes,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        k=8,
        hidA=32,  # used for Q/K transformations
        hidR=40,
        hidP=1,
        n_layer=1,
        dropout=0.5,
        device='cpu'
    ):
        super(EpiGNNVariant, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k = k
        self.n = n_layer
        self.dropout_layer = nn.Dropout(dropout)

        # RegionAwareConv
        self.backbone = RegionAwareConv(
            nfeat=num_features, P=num_timesteps_input,
            m=num_nodes, k=k, hidP=hidP
        )

        # Q/K transformations
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # Gating param for adjacency
        self.d_gate = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner for dynamic adjacency
        self.graphGen = GraphLearner(self.hidR)

        # GNN blocks
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
            for _ in range(self.n)
        ])

        # Final projection
        # We'll concatenate the outputs from each GNN block + original embedding
        self.output = nn.Linear(self.hidR * self.n + self.hidR, num_timesteps_output)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, adjacency_type='hybrid'):
        """
        X: (batch_size, T_in, num_nodes, num_features)
        adj: (batch_size, num_nodes, num_nodes)
        adjacency_type: 'static', 'dynamic', or 'hybrid'
        """
        # 1) Reshape X for the RegionAwareConv
        #    (batch_size, num_features, T_in, num_nodes)
        X_reshaped = X.permute(0, 3, 1, 2)
        temp_emb = self.backbone(X_reshaped)  # (batch_size, num_nodes, hidR)

        # 2) Q/K transformations for attention
        query = self.dropout_layer(self.WQ(temp_emb))
        key   = self.dropout_layer(self.WK(temp_emb))

        attn = torch.bmm(query, key.transpose(1, 2))
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)
        t_enc = self.dropout_layer(self.t_enc(attn))

        # 3) Local transmission risk
        d = torch.sum(adj, dim=2).unsqueeze(2)  # (batch_size, num_nodes, 1)
        s_enc = self.dropout_layer(self.s_enc(d))

        # 4) Combine embeddings
        feat_emb = temp_emb + t_enc + s_enc  # (batch_size, num_nodes, hidR)

        # 5) Learned adjacency
        learned_adj = self.graphGen(feat_emb)

        # 6) Combine adjacency based on adjacency_type
        if adjacency_type == 'static':
            combined_adj = adj
        elif adjacency_type == 'dynamic':
            combined_adj = learned_adj
        elif adjacency_type == 'hybrid':
            # Example gating approach:
            d_mat = torch.sum(adj, dim=1, keepdim=True) * torch.sum(adj, dim=2, keepdim=True)
            d_mat = torch.sigmoid(self.d_gate * d_mat)
            spatial_adj = d_mat * adj
            combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)
        else:
            raise ValueError("Invalid adjacency_type. Must be 'static', 'dynamic', or 'hybrid'.")

        # 7) Laplacian-like adjacency
        from EpiGNN.epignn_model import getLaplaceMat  # or your local utility
        laplace_adj = getLaplaceMat(X.size(0), self.num_nodes, combined_adj)

        # 8) Pass through GNN blocks
        node_state = feat_emb
        node_states_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)  # (batch_size, num_nodes, hidR)
            node_state = F.elu(node_state)
            node_states_list.append(node_state)

        # 9) Concatenate all GNN outputs + original embedding
        node_state_cat = torch.cat(node_states_list, dim=-1)  # (batch_size, num_nodes, hidR * n)
        node_state_all = torch.cat([node_state_cat, feat_emb], dim=-1)  # (batch_size, num_nodes, hidR * n + hidR)

        # 10) Final projection => (batch_size, T_out, num_nodes)
        res = self.output(node_state_all)  # (batch_size, num_nodes, T_out)
        return res.transpose(1, 2)  # (batch_size, T_out, num_nodes)

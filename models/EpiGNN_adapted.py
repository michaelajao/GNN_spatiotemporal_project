# EpiGNN_adapted.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing

from .base import BaseModel


def getLaplaceMat(batch_size, m, adj):
    """
    Computes the Laplacian matrix for the graph convolution.

    Args:
        batch_size (int): Number of graphs in the batch.
        m (int): Number of nodes in each graph.
        adj (torch.Tensor): Adjacency matrix of shape (batch_size, m, m).

    Returns:
        torch.Tensor: Laplacian matrix of shape (batch_size, m, m).
    """
    i_mat = torch.eye(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    o_mat = torch.ones(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    adj = torch.where(adj > 0, o_mat, adj)

    d_mat_in = torch.sum(adj, dim=1)
    d_mat_out = torch.sum(adj, dim=2)
    d_mat = d_mat_out.unsqueeze(2) + 1e-12
    d_mat = torch.pow(d_mat, -1)
    d_mat = i_mat * d_mat

    laplace_mat = torch.bmm(d_mat, adj)
    return laplace_mat


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        """
        Learns the adjacency matrix based on node embeddings.

        Args:
            embedding (torch.Tensor): Node embeddings of shape (batch_size, num_nodes, hidden_dim).

        Returns:
            torch.Tensor: Learned adjacency matrix of shape (batch_size, num_nodes, num_nodes).
        """
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)

        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2)) - torch.bmm(nodevec2, nodevec1.transpose(1, 2))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj


class ConvBranch(nn.Module):
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor=2, hidP=1, isPool=True):
        super(ConvBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, x.size(-1))
        x = self.activate(x)
        return x


class RegionAwareConv(nn.Module):
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=5, dilation_factor=1, hidP=hidP)
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        Applies multiple convolution branches to extract local, periodic, and global features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, P, m).

        Returns:
            torch.Tensor: Output tensor after convolution and activation, shape (batch_size, k*4, m).
        """
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)

        x_global = self.conv_g(x)

        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x


class EpiGNN(BaseModel):
    """
    Adapted Epidemiological Graph Neural Network (EpiGNN) for Hospitalization Prediction.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph (e.g., 7 NHS regions).
    num_features : int
        Number of features per node per timestep.
    num_timesteps_input : int
        Number of timesteps considered for each input sample.
    num_timesteps_output : int
        Number of output timesteps to predict.
    k : int, optional
        Number of local neighborhoods to consider in the graph learning layer. Default: 8.
    hidA : int, optional
        Dimension of attention in the model. Default: 64.
    hidR : int, optional
        Dimension of hidden layers in the recurrent neural network part. Default: 40.
    hidP : int, optional
        Dimension of positional encoding in the model. Default: 1.
    n_layer : int, optional
        Number of layers in the graph neural network. Default: 2.
    dropout : float, optional
        Dropout rate for regularization during training to prevent overfitting. Default: 0.5.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_timesteps_output, num_nodes), representing the predicted ICU bed usage for each node over future timesteps.
    """
    def __init__(self, 
                num_nodes, 
                num_features, 
                num_timesteps_input,
                num_timesteps_output, 
                k=8, 
                hidA=64, 
                hidR=40, 
                hidP=1, 
                n_layer=2, 
                dropout=0.5, 
                device='cpu'):
        super(EpiGNN, self).__init__()
        self.device = device
        self.nfeat = num_features
        self.m = num_nodes
        self.w = num_timesteps_input
        self.droprate = dropout
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k = k
        self.n = n_layer
        self.dropout = nn.Dropout(self.droprate)

        # Feature embedding
        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # Global transmission risk encoding
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # Local transmission risk encoding
        self.s_enc = nn.Linear(1, self.hidR)

        # External resources (if any, optional)
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for _ in range(self.n)])

        # Prediction layer
        self.output = nn.Linear(self.hidR * 2, num_timesteps_output)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # Best practice
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, states=None, dynamic_adj=None, index=None):
        """
        Forward pass of the adapted EpiGNN model for Hospitalization Prediction.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor with shape (batch_size, num_timesteps_input, num_nodes, num_features).
        adj : torch.Tensor
            Static adjacency matrix with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            Current state variables tensor (if applicable). Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix (if applicable). Default: None.
        index : torch.Tensor, optional
            Indices for external resources (if applicable). Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_timesteps_output, num_nodes),
            representing the predicted ICU bed usage for each node over future timesteps.
        """
        adj = adj.bool().float()
        batch_size = X.size(0)  # batch_size, T, N, F

        # Step 1: Use multi-scale convolution to extract feature embedding (RegionAwareConv)
        temp_emb = self.backbone(X)  # Shape: (batch_size, hidR, m)

        # Step 2: Generate global transmission risk encoding
        query = self.WQ(temp_emb)  # Shape: (batch_size, m, hidA)
        query = self.dropout(query)
        key = self.WK(temp_emb)    # Shape: (batch_size, m, hidA)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))  # Shape: (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)  # Normalize
        attn = torch.sum(attn, dim=-1).unsqueeze(2)      # Shape: (batch_size, m, 1)
        t_enc = self.t_enc(attn)                         # Shape: (batch_size, m, hidR)
        t_enc = self.dropout(t_enc)

        # Step 3: Generate local transmission risk encoding
        d = torch.sum(adj, dim=1).unsqueeze(1)            # Shape: (batch_size, 1, m)
        s_enc = self.s_enc(d)                             # Shape: (batch_size, m, hidR)
        s_enc = self.dropout(s_enc)

        # Step 4: Three embedding fusion
        feat_emb = temp_emb + t_enc + s_enc                # Shape: (batch_size, m, hidR)

        # Step 5: Region-Aware Graph Learner
        # Load external resource if available (optional)
        if self.external_parameter is not None and index is not None:
            extra_adj_list = []
            zeros_mt = torch.zeros((self.m, self.m)).to(adj.device)
            for i in range(batch_size):
                offset = 20
                if i - offset >= 0:
                    idx = i - offset
                    extra_adj_list.append(self.external_parameter[index[i], :, :].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.cat(extra_adj_list, dim=0)  # Shape: (batch_size, m, m)
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
        else:
            external_info = 0

        # Apply Graph Learner to generate a graph
        d_mat = torch.mm(torch.sum(adj, dim=1), torch.sum(adj, dim=1).transpose(0, 1))  # Shape: (batch_size, m, m)
        d_mat = torch.mul(self.d_gate, d_mat)                                        # Shape: (batch_size, m, m)
        d_mat = torch.sigmoid(d_mat)                                                 # Shape: (batch_size, m, m)
        spatial_adj = torch.mul(d_mat, adj)                                          # Shape: (batch_size, m, m)
        learned_adj = self.graphGen(feat_emb)                                        # Shape: (batch_size, m, m)

        # If additional information, fuse
        if external_info != 0:
            adj = learned_adj + spatial_adj + external_info
        else:
            adj = learned_adj + spatial_adj

        # Get Laplace adjacency matrix
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)

        # Step 6: Graph Convolution Network
        node_state = feat_emb                                       # Shape: (batch_size, m, hidR)
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)           # Shape: (batch_size, m, hidR)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)
        
        # Concatenate node states from all GNN layers
        node_state = torch.cat(node_state_list, dim=-1)           # Shape: (batch_size, m, hidR * n_layer)

        # Concatenate initial features and GNN outputs
        node_state = torch.cat([node_state, feat_emb], dim=-1)    # Shape: (batch_size, m, hidR * n_layer + hidR)

        # Step 7: Prediction
        res = self.output(node_state)                              # Shape: (batch_size, m, num_timesteps_output)
        res = res.transpose(1, 2)                                   # Shape: (batch_size, num_timesteps_output, m)

        return res  # Predicted covidOccupiedMVBeds

    def initialize(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

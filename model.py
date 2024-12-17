# FILE: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class STAN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        """
        Spatio-Temporal Attention Network (STAN) Model using PyTorch Geometric.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim1 (int): Hidden dimension for the first GAT layer.
            hidden_dim2 (int): Hidden dimension for the second GAT layer.
            gru_dim (int): Dimension of the GRU hidden state.
            num_heads (int): Number of attention heads.
            pred_window (int): Prediction window size.
            device (torch.device): Device to run computations on.
        """
        super(STAN, self).__init__()
        self.device = device

        # GAT Layers
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim1, heads=num_heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(in_channels=hidden_dim1 * num_heads, out_channels=hidden_dim2, heads=1, concat=True, dropout=0.6)

        # GRU Cell
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)

        # Prediction Layers
        self.nn_res_I = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_sir = nn.Linear(gru_dim + 2, 2)

    def forward(self, data, dynamic, cI, cR, N, I, R, dI, dR, h=None):
        """
        Forward pass for the STAN model.

        Args:
            data (torch_geometric.data.Batch): Batched graph data containing edge_index and batch.
            dynamic (Tensor): Dynamic features [batch_size * num_loc, history_window * n_feat].
            cI (Tensor): Cumulative infected cases [batch_size * num_loc].
            cR (Tensor): Cumulative recovered cases [batch_size * num_loc].
            N (Tensor): Total population [batch_size * num_loc, 1].
            I (Tensor): Current infected cases [batch_size * num_loc].
            R (Tensor): Current recovered cases [batch_size * num_loc].
            dI (Tensor): Change in infected cases [batch_size * num_loc].
            dR (Tensor): Change in recovered cases [batch_size * num_loc].
            h (Tensor, optional): Hidden state for GRU. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 
                - Predicted I [batch_size * num_loc, pred_window]
                - Predicted R [batch_size * num_loc, pred_window]
                - Physical I predictions [batch_size * num_loc, pred_window]
                - Physical R predictions [batch_size * num_loc, pred_window]
                - Updated hidden state [batch_size * num_loc, gru_dim]
        """
        # Pass through GAT layers
        x = dynamic  # [batch_size * num_loc, in_dim]
        x = self.gat1(x, data.edge_index)  # [batch_size * num_loc, hidden_dim1 * num_heads]
        x = F.elu(x)
        x = self.gat2(x, data.edge_index)  # [batch_size * num_loc, hidden_dim2]
        x = F.elu(x)

        # GRU Cell
        if h is None:
            h = torch.zeros(x.size(0), self.gru.hidden_size).to(self.device)
        
        h = self.gru(x, h)  # [batch_size * num_loc, gru_dim]

        # Concatenate GRU hidden state with cI and cR per node
        hc = torch.cat((h, cI.unsqueeze(1), cR.unsqueeze(1)), dim=1)  # [batch_size * num_loc, gru_dim + 2]

        # Predict I and R
        pred_I = self.nn_res_I(hc)  # [batch_size * num_loc, pred_window]
        pred_R = self.nn_res_R(hc)  # [batch_size * num_loc, pred_window]

        # Predict alpha and beta for SIR model
        pred_res = self.nn_res_sir(hc)  # [batch_size * num_loc, 2]
        alpha = torch.sigmoid(pred_res[:, 0])  # [batch_size * num_loc]
        beta = torch.sigmoid(pred_res[:, 1])   # [batch_size * num_loc]

        # Physical model predictions
        # S_graph = N - I - R (per node)
        S_graph = N.squeeze(1) - I - R  # [batch_size * num_loc]

        phy_I = []
        phy_R = []

        for _ in range(pred_window):
            dI_val = alpha * I * (S_graph / N.squeeze(1)) - beta * I  # [batch_size * num_loc]
            dR_val = beta * I  # [batch_size * num_loc]

            phy_I.append(dI_val)
            phy_R.append(dR_val)

            # Update I and R for next step
            I = I + dI_val.detach()
            R = R + dR_val.detach()

        phy_I = torch.stack(phy_I, dim=1)  # [batch_size * num_loc, pred_window]
        phy_R = torch.stack(phy_R, dim=1)  # [batch_size * num_loc, pred_window]

        # Return all predictions
        return pred_I, pred_R, phy_I, phy_R, h

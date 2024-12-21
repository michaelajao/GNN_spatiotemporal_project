import os 
import random
import math
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.io as pio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import Parameter
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# Set default plot style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# -----------------------------------
# 1. Seed & Device Configuration
# -----------------------------------
RANDOM_SEED = 123

def seed_torch(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------
# 2. Hyperparameters
# -----------------------------------
num_nodes = 7
num_features = 5   # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
num_timesteps_input = 14
num_timesteps_output = 7
k = 8
hidA = 32
hidR = 40
hidP = 1
n_layer = 4  # Number of GNN layers
dropout = 0.5
learning_rate = 1e-3
num_epochs = 1000
batch_size = 32
threshold_distance = 300  # km threshold for adjacency
early_stopping_patience = 10

# -----------------------------------
# 3. Reference Coordinates Correction
# -----------------------------------
REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}

# -----------------------------------
# 4. Data Loading and Preprocessing
# -----------------------------------
def load_and_correct_data(data, reference_coordinates):
    # Assign correct geographic coordinates
    for region, coords in reference_coordinates.items():
        data.loc[data['areaName'] == region, ['latitude', 'longitude']] = coords

    # Apply 7-day rolling mean to specified features
    rolling_features = ['new_confirmed', 'new_deceased', 'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']
    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )
    
    # Fill any leftover NaNs
    data[rolling_features] = data[rolling_features].fillna(0)
    
    # Sort data chronologically by region
    data.sort_values(['areaName', 'date'], inplace=True)

    return data

class NHSRegionDataset(Dataset):
    """
    Creates a dataset of shape:
      X: (num_timesteps_input, num_nodes, num_features)
      Y: (num_timesteps_output, num_nodes)
    The DataLoader will add the batch dimension automatically.
    """
    def __init__(self, data, num_timesteps_input, num_timesteps_output, scaler=None):
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        self.regions = self.data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        # We'll focus on these features
        self.features = ['new_confirmed', 'new_deceased', 'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']

        # Pivot to create a time-series structure: index=date, columns=region_idx
        # Each cell is an array of len(features).
        # The shape will be (num_dates, num_nodes, num_features) after we reshape.
        self.pivot = self.data.pivot(index='date', columns='region_idx', values=self.features)
        
        # Forward fill missing by date, then fill any leftover with 0
        self.pivot.ffill(inplace=True)
        self.pivot.fillna(0, inplace=True)
        
        # Convert pivot to NumPy array
        # pivot.values shape: (num_dates, num_nodes * num_features)
        self.num_features = len(self.features)
        self.num_dates = self.pivot.shape[0]
        
        # Reshape to (num_dates, num_nodes, num_features)
        self.feature_array = self.pivot.values.reshape(
            self.num_dates, self.num_nodes, self.num_features
        )

        # Check for population consistency
        populations = self.data.groupby('areaName')['population'].unique()
        inconsistent_pop = populations[populations.apply(len) > 1]
        if not inconsistent_pop.empty:
            raise ValueError(f"Inconsistent population values in regions: {inconsistent_pop.index.tolist()}")
        
        # Scaling
        if scaler is not None:
            self.scaler = scaler
            # Reshape for scaler: (num_dates*num_nodes, num_features)
            self.feature_array = self.scaler.transform(self.feature_array.reshape(-1, self.num_features))
            # Reshape back to (num_dates, num_nodes, num_features)
            self.feature_array = self.feature_array.reshape(self.num_dates, self.num_nodes, self.num_features)
        else:
            self.scaler = None
        
    def __len__(self):
        # For each valid starting index, we can produce an (X, Y) pair
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx):
        """
        X shape => (num_timesteps_input, num_nodes, num_features)
        Y shape => (num_timesteps_output, num_nodes)
        """
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # (T_in, m, F)
        # For Y, we pick only the 5th feature => covidOccupiedMVBeds => index=4
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def compute_geographic_adjacency(regions, latitudes, longitudes, threshold=threshold_distance):
    """
    Creates a binary adjacency matrix (num_nodes x num_nodes) 
    based on geographic distance threshold using haversine.
    """
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in km
        return c * r

    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 1
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
    return torch.tensor(adj_matrix, dtype=torch.float32)

def getLaplaceMat(batch_size, m, adj):
    """
    Compute the Laplacian-like matrix for GCN from adjacency.
    """
    i_mat = torch.eye(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    # Convert adjacency to "1" for edges > 0
    adj_bin = (adj > 0).float()  
    # Degree matrix
    deg = torch.sum(adj_bin, dim=2)  # shape: (batch_size, m)
    deg_inv = 1.0 / (deg + 1e-12)
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)  # shape: (batch_size, m, m)
    
    laplace_mat = torch.bmm(deg_inv_mat, adj_bin)
    return laplace_mat

# -----------------------------------
# 5. Model Definition
# -----------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1.0 / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        # feature: (batch_size, m, in_features)
        # adj: (batch_size, m, m)
        support = torch.matmul(feature, self.weight)  # (batch_size, m, out_features)
        output = torch.bmm(adj, support)             # (batch_size, m, out_features)
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
        embedding: (batch_size, m, hidR)
        Generate a learned adjacency via an attention-like mechanism.
        """
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))

        # Symmetrical adjacency
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2)) - \
              torch.bmm(nodevec2, nodevec1.transpose(1, 2))

        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    """
    A single branch of the RegionAwareConv that does a 2D convolution 
    and optional pooling over the time dimension.
    """
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor=2, hidP=1, isPool=True):
        super(ConvBranch, self).__init__()
        # Conv2d with kernel_size=(kernel_size,1) and dilation=(dilation_factor,1)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
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
        x: (batch_size, in_channels, seq_len, m)
        """
        x = self.conv(x)         # => shape: (batch_size, out_channels, new_seq_len, m)
        x = self.batchnorm(x)
        
        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)  # => shape: (batch_size, out_channels, hidP, m)
        
        # Flatten out_channels * hidP along dim=1
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))  # => (batch_size, out_channels*hidP, m)

        return self.activate(x)

class RegionAwareConv(nn.Module):
    """
    The 'backbone' that extracts local, period, and global features 
    from the time dimension for each node.
    """
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        # local convs: kernel_size=3 or 5 with dilation=1
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=5, dilation_factor=1, hidP=hidP)
        # period convs: kernel_size=3 or 5 with dilation>1
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        # global conv: kernel_size=P, no pooling
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k, kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, nfeat, P, m)
        """
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)    # => (batch_size, 2*k*hidP, m)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)   # => (batch_size, 2*k*hidP, m)

        x_global = self.conv_g(x)  # => (batch_size, k, m)

        # final cat => (batch_size, something, m)
        x = torch.cat([x_local, x_period, x_global], dim=1)
        return self.activate(x).permute(0, 2, 1)  # => (batch_size, m, (something))

class EpiGNN(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 num_features, 
                 num_timesteps_input,
                 num_timesteps_output, 
                 k=8, 
                 hidA=32, 
                 hidR=40,   # updated
                 hidP=1, 
                 n_layer=1, 
                 dropout=0.5, 
                 device='cpu'):
        super(EpiGNN, self).__init__()
        self.device = device
        self.m = num_nodes
        self.w = num_timesteps_input
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k = k
        self.n = n_layer
        self.dropout_layer = nn.Dropout(dropout)

        # The backbone
        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # Some linear layers
        self.WQ = nn.Linear(self.hidR, self.hidA)  
        self.WK = nn.Linear(self.hidR, self.hidA)  
        self.t_enc = nn.Linear(1, self.hidR)       
        self.s_enc = nn.Linear(1, self.hidR)       

        # Removed external_parameter for simplicity

        # Gating parameter
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner
        self.graphGen = GraphLearner(self.hidR)

        # GNN block(s)
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for _ in range(self.n)
        ])

        # Final output: we concat (node_state from each layer + original feat_emb) => total dimension = hidR*(n_layer) + hidR
        self.output = nn.Linear(self.hidR*(self.n) + self.hidR, num_timesteps_output)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, states=None, dynamic_adj=None, index=None):
        """
        X: (batch_size, T, m, F)
        adj: (batch_size, m, m)
        """
        # Permute input to match RegionAwareConv requirement => (batch_size, F, T, m)
        X_reshaped = X.permute(0, 3, 1, 2)  # => (batch_size, F, T, m)
        
        # Pass through the backbone => returns shape (batch_size, m, hidR)
        temp_emb = self.backbone(X_reshaped)  # (batch_size, m, hidR)

        # Apply dropout to Q/K transformations
        query = self.dropout_layer(self.WQ(temp_emb))  # (batch_size, m, hidA=32)
        key   = self.dropout_layer(self.WK(temp_emb))  # (batch_size, m, hidA=32)

        attn = torch.bmm(query, key.transpose(1, 2))  # => (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)  # => (batch_size, m, 1)
        t_enc = self.dropout_layer(self.t_enc(attn))  # => (batch_size, m, hidR)

        # Local transmission risk encoding
        # adj: (batch_size, m, m). Summation over dim=1 => node degrees
        d = torch.sum(adj, dim=1).unsqueeze(2)  # (batch_size, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))  # => (batch_size, m, hidR)

        # Combine
        feat_emb = temp_emb + t_enc + s_enc  # => (batch_size, m, hidR)

        # Learned adjacency
        d_mat = torch.sum(adj, dim=1, keepdim=True) * torch.sum(adj, dim=2, keepdim=True)
        d_mat = torch.sigmoid(self.d_gate * d_mat)
        spatial_adj = d_mat * adj  # => (batch_size, m, m)

        learned_adj = self.graphGen(feat_emb)  # => (batch_size, m, m)

        # Combine adjacencies and clamp to [0, 1]
        combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)
        laplace_adj = getLaplaceMat(X.size(0), self.m, combined_adj)

        # GNN layers
        node_state = feat_emb  # shape: (batch_size, m, hidR)
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = self.dropout_layer(layer(node_state, laplace_adj))  # (batch_size, m, hidR)
            node_state_list.append(node_state)

        # Concat GNN outputs + original feat_emb
        node_state_all = torch.cat(node_state_list, dim=-1)  # => (batch_size, m, hidR*n)
        node_state_all = torch.cat([node_state_all, feat_emb], dim=-1)  # => (batch_size, m, hidR*n + hidR)

        # Final output => (batch_size, m, num_timesteps_output)
        res = self.output(node_state_all)  # => (batch_size, m, 7)
        return res.transpose(1, 2)        # => (batch_size, 7, m)

# -----------------------------------
# 6. Data Loading and Normalization
# -----------------------------------
csv_path = "../data/merged_nhs_covid_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

data = pd.read_csv(csv_path, parse_dates=['date'])
data = load_and_correct_data(data, REFERENCE_COORDINATES)

# Create initial dataset without scaling
initial_dataset = NHSRegionDataset(data, num_timesteps_input=num_timesteps_input, num_timesteps_output=num_timesteps_output, scaler=None)
print(f"Total samples in initial dataset: {len(initial_dataset)}")

# Chronological split
total_len = len(initial_dataset)
train_size = int(0.7 * total_len)
val_size   = int(0.15 * total_len)
test_size  = total_len - train_size - val_size

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, total_len))

# Initialize and fit scaler on training data
scaler = StandardScaler()

# Extract all training features and fit scaler
train_features = []
for i in range(train_size):
    X, _ = initial_dataset[i]
    train_features.append(X.numpy())
train_features = np.concatenate(train_features, axis=0).reshape(-1, num_features)
scaler.fit(train_features)

# Now create the scaled dataset
scaled_dataset = NHSRegionDataset(data, num_timesteps_input=num_timesteps_input, num_timesteps_output=num_timesteps_output, scaler=scaler)
print(f"Total samples in scaled dataset: {len(scaled_dataset)}")

# Create subsets based on chronological indices
train_subset = Subset(scaled_dataset, train_indices)
val_subset = Subset(scaled_dataset, val_indices)
test_subset = Subset(scaled_dataset, test_indices)

print(f"Training samples:   {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples:       {len(test_subset)}")

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False, drop_last=False)

# -----------------------------------
# 7. Compute Geographic Adjacency
# -----------------------------------
regions = scaled_dataset.regions.tolist()
latitudes = [data[data['areaName'] == region]['latitude'].iloc[0] for region in regions]
longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

adj = compute_geographic_adjacency(regions, latitudes, longitudes).to(device)
print("Adjacency matrix:")
print(adj.cpu().numpy())

# Plot adjacency as a geographic graph
adj_np = adj.cpu().numpy()
G = nx.from_numpy_array(adj_np)
mapping = {i: region for i, region in enumerate(regions)}
G = nx.relabel_nodes(G, mapping)
pos = {region: (longitudes[i], latitudes[i]) for i, region in enumerate(regions)}

plt.figure(figsize=(12, 10))
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color='lightblue', edge_color='gray', font_size=12, font_weight='bold')
plt.title('Geographic Adjacency Graph')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('off')
plt.tight_layout()
plt.savefig('geographic_adjacency_graph.png', dpi=300)
plt.show()

# -----------------------------------
# 8. Model Initialization
# -----------------------------------
model = EpiGNN(
    num_nodes=num_nodes,
    num_features=num_features,
    num_timesteps_input=num_timesteps_input,
    num_timesteps_output=num_timesteps_output,
    k=k,
    hidA=hidA,
    hidR=hidR,
    hidP=hidP,
    n_layer=n_layer,
    dropout=dropout,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.MSELoss()

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# -----------------------------------
# 9. Training Loop
# -----------------------------------
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()

        # Expand adjacency to batch size
        batch_size_current = batch_X.size(0)
        batch_adj = adj.unsqueeze(0).repeat(batch_size_current, 1, 1)

        pred = model(batch_X, batch_adj)
        loss = criterion(pred, batch_Y)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    all_val_preds = []
    all_val_actuals = []

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            batch_size_current = batch_X.size(0)
            batch_adj = adj.unsqueeze(0).repeat(batch_size_current, 1, 1)

            pred = model(batch_X, batch_adj)
            vloss = criterion(pred, batch_Y)
            epoch_val_loss += vloss.item()

            all_val_preds.append(pred.cpu().numpy())
            all_val_actuals.append(batch_Y.cpu().numpy())

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Step scheduler
    scheduler.step(avg_val_loss)

    # Evaluate R2 (per node)
    all_val_preds = np.concatenate(all_val_preds, axis=0)     # => shape: (num_samples, T_out, m)
    all_val_actuals = np.concatenate(all_val_actuals, axis=0) # => shape: (num_samples, T_out, m)
    
    # Flatten to (num_samples*T_out, m) to feed r2_score
    preds_2d = all_val_preds.reshape(-1, num_nodes)
    actuals_2d = all_val_actuals.reshape(-1, num_nodes)
    r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')
    
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val R² (per node): {r2_vals}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_epignn_adapted_model.pth')
        print("Model checkpoint saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Plot train vs val losses
plt.figure(figsize=(12,8))
sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label='Train Loss', color='blue')
sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_loss_curves.png', dpi=300)
plt.show()

# -----------------------------------
# 10. Test Evaluation
# -----------------------------------
# Load best model
model.load_state_dict(torch.load('best_epignn_adapted_model.pth', map_location=device))
model.eval()

test_loss = 0.0
all_preds = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        batch_size_current = batch_X.size(0)
        batch_adj = adj.unsqueeze(0).repeat(batch_size_current, 1, 1)

        pred = model(batch_X, batch_adj)
        loss = criterion(pred, batch_Y)
        test_loss += loss.item()

        all_preds.append(pred.cpu())
        all_actuals.append(batch_Y.cpu())

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss (MSE): {avg_test_loss:.4f}")

# Concatenate predictions and actual values
all_preds = torch.cat(all_preds, dim=0)       # => shape: (N, T_out, m)
all_actuals = torch.cat(all_actuals, dim=0)   # => shape: (N, T_out, m)

# Inverse transform only the 'covidOccupiedMVBeds' feature
if scaled_dataset.scaler is not None:
    # Get the scaling parameters for the 5th feature
    scale_covid = scaled_dataset.scaler.scale_[4]
    mean_covid = scaled_dataset.scaler.mean_[4]
    
    all_preds_np = all_preds.numpy() * scale_covid + mean_covid
    all_actuals_np = all_actuals.numpy() * scale_covid + mean_covid
else:
    all_preds_np = all_preds.numpy()
    all_actuals_np = all_actuals.numpy()

# Flatten to 2D
preds_flat = all_preds_np.reshape(-1, num_nodes)
actuals_flat = all_actuals_np.reshape(-1, num_nodes)

# Metrics
mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
r2_per_node = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

# Print MAE and R² for each region
for idx, region in enumerate(regions):
    print(f"Region: {region}, MAE: {mae_per_node[idx]:.4f}, R²: {r2_per_node[idx]:.4f}")

# -----------------------------------
# 11. Enhanced Visualization
# -----------------------------------
# Print MAE and R² for each region
for idx, region in enumerate(regions):
    print(f"Region: {region}, MAE: {mae_per_node[idx]:.4f}, R²: {r2_per_node[idx]:.4f}")

# -------------------------------
# 11.1. Mapping Predictions to Dates
# -------------------------------
# Extract unique sorted dates
unique_dates = data['date'].sort_values().unique()

# Determine the start index for the test set
test_start_idx = train_size + val_size

# Initialize lists to store forecasted dates
forecast_dates = []

# Number of samples in the test set
num_test_samples = len(test_subset)

# For each test sample, assign forecasted dates
for i in range(num_test_samples):
    # The input window ends at 'test_start_idx + i + num_timesteps_input - 1'
    # The prediction starts at 'test_start_idx + i + num_timesteps_input'
    pred_start_idx = test_start_idx + i + num_timesteps_input
    pred_end_idx = pred_start_idx + num_timesteps_output
    # Ensure indices do not exceed the total number of dates
    if pred_end_idx > len(unique_dates):
        pred_end_idx = len(unique_dates)
    # Assign forecasted dates for this sample
    sample_forecast_dates = unique_dates[pred_start_idx:pred_end_idx]
    # If there are fewer dates than 'num_timesteps_output', pad with the last available date
    if len(sample_forecast_dates) < num_timesteps_output:
        last_date = unique_dates[-1]
        sample_forecast_dates = np.append(sample_forecast_dates, [last_date]*(num_timesteps_output - len(sample_forecast_dates)))
    forecast_dates.extend(sample_forecast_dates)

# Create a DataFrame for predictions
preds_df = pd.DataFrame(all_preds_np.reshape(-1, num_nodes), columns=regions)
preds_df['Date'] = forecast_dates

# Create a DataFrame for actuals
actuals_df = pd.DataFrame(all_actuals_np.reshape(-1, num_nodes), columns=regions)
actuals_df['Date'] = forecast_dates

# Aggregate predictions by averaging for each date
agg_preds_df = preds_df.groupby('Date').mean().reset_index()

# Since actuals are unique per date, take the first occurrence
agg_actuals_df = actuals_df.groupby('Date').first().reset_index()

# Merge actual and predicted data
merged_df = pd.merge(agg_preds_df, agg_actuals_df, on='Date', suffixes=('_Predicted', '_Actual'))

# Ensure 'Date' is datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# -------------------------------
# 11.2. Plotting Actual vs Predicted Time Series for Each Region
# -------------------------------
# Function to create and save time series plots
def plot_time_series(region, df):
    plt.figure(figsize=(14, 7))
    
    # Plot Actual Values
    sns.lineplot(x='Date', y=f'{region}_Actual', data=df, label='Actual', color='blue', marker='o')
    
    # Plot Predicted Values
    sns.lineplot(x='Date', y=f'{region}_Predicted', data=df, label='Predicted', color='red', linestyle='--', marker='x')
    
    # Formatting the x-axis for better readability
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.title(f'Actual vs Predicted COVID Occupied MV Beds for {region}')
    plt.xlabel('Date')
    plt.ylabel('COVID Occupied MV Beds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{region.replace(" ", "_")}.png', dpi=300)
    plt.show()

# Create time series plots for each region
for region in regions:
    plot_time_series(region, merged_df)

# -------------------------------
# 11.3. Additional Relevant Visualizations
# -------------------------------

# a. Error Distribution Histogram for Each Region with KDE and Annotations
def plot_error_distribution(region, df):
    errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.title(f'Prediction Error Distribution for {region}')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Calculate statistics
    mean_error = errors.mean()
    median_error = errors.median()
    
    # Add vertical lines for mean and median
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}')
    plt.axvline(median_error, color='green', linestyle='dotted', linewidth=1, label=f'Median: {median_error:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'error_distribution_{region.replace(" ", "_")}.png', dpi=300)
    plt.show()

# Plot error distributions
for region in regions:
    plot_error_distribution(region, merged_df)

# b. Cumulative Error Over Time for Each Region with Shaded Areas
def plot_cumulative_error(region, df):
    errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
    cumulative_errors = errors.cumsum()
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='Date', y=cumulative_errors, data=df, label='Cumulative Error', color='green')
    plt.fill_between(df['Date'], cumulative_errors, color='green', alpha=0.1)
    plt.title(f'Cumulative Prediction Error Over Time for {region}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'cumulative_error_{region.replace(" ", "_")}.png', dpi=300)
    plt.show()

# Plot cumulative errors
for region in regions:
    plot_cumulative_error(region, merged_df)

# 4. Heatmap of Prediction Errors Over Time for Each Region
for node_idx, region in enumerate(regions):
    errors = all_preds_np[:, :, node_idx] - all_actuals_np[:, :, node_idx]
    plt.figure(figsize=(14, 6))
    sns.heatmap(errors, cmap='coolwarm', annot=False, cbar=True)
    plt.title(f'Heatmap of Prediction Errors for {region}')
    plt.xlabel('Timestep Output')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.show()


# d. Boxplot of Prediction Errors for Each Region with Swarm Overlay
def plot_error_boxplot(df):
    error_data = []
    for region in regions:
        errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
        error_data.append(pd.Series(errors, name=region))
    
    error_df = pd.concat(error_data, axis=1)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=error_df, palette='Set2')
    sns.swarmplot(data=error_df, color=".25")
    plt.title('Boxplot of Prediction Errors for Each Region')
    plt.xlabel('Region')
    plt.ylabel('Prediction Error (Predicted - Actual)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('boxplot_prediction_errors.png', dpi=300)
    plt.show()

# Plot boxplot with swarm
plot_error_boxplot(merged_df)

# e. Scatter Plot of Actual vs Predicted Values for Each Region with Regression Line
def plot_scatter_actual_vs_predicted(region, df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[f'{region}_Actual'], y=df[f'{region}_Predicted'], color='teal', alpha=0.6, edgecolor=None)
    sns.regplot(x=df[f'{region}_Actual'], y=df[f'{region}_Predicted'], scatter=False, color='red', label='Regression Line')
    plt.title(f'Actual vs Predicted COVID Occupied MV Beds for {region}')
    plt.xlabel('Actual COVID Occupied MV Beds')
    plt.ylabel('Predicted COVID Occupied MV Beds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scatter_actual_vs_predicted_{region.replace(" ", "_")}.png', dpi=300)
    plt.show()

# Plot scatter plots with regression lines
for region in regions:
    plot_scatter_actual_vs_predicted(region, merged_df)

# -----------------------------------
# 12. Save Final Model
# -----------------------------------
torch.save(model.state_dict(), 'epignn_adapted_model_final.pth')
print("Final model saved as 'epignn_adapted_model_final.pth'.")

# -----------------------------------
# 13. Visualize the Learned Adjacency for One Sample
# -----------------------------------
def plot_learned_adjacency(model, adj, regions):
    example_X, _ = next(iter(test_loader))
    example_X = example_X.to(device)
    
    with torch.no_grad():
        # Replicate how we get temp_emb inside the model.
        X_reshaped = example_X.permute(0, 3, 1, 2)  # (batch_size, F, T, m)
        emb_for_adj = model.backbone(X_reshaped)    # => (batch_size, m, hidR)
        # Evaluate GraphLearner on the first item in the batch
        single_emb = emb_for_adj[0].unsqueeze(0)    # => (1, m, hidR)
        learned_adj = model.graphGen(single_emb)    # => (1, m, m)

        # Combine adjacencies and clamp to [0, 1]
        # Ensure 'adj' is batched
        batch_adj = adj.unsqueeze(0)  # (1, m, m)
        d_mat = torch.sum(batch_adj, dim=1, keepdim=True) * torch.sum(batch_adj, dim=2, keepdim=True)
        d_mat = torch.sigmoid(model.d_gate * d_mat)
        spatial_adj = d_mat * batch_adj  # => (1, m, m)
        combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)
        learned_adj_np = combined_adj.squeeze(0).cpu().numpy()

    # Plot the adjacency matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(learned_adj_np, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=regions, yticklabels=regions)
    plt.title('Learned Adjacency Matrix (Test Sample)')
    plt.xlabel('Regions')
    plt.ylabel('Regions')
    plt.tight_layout()
    plt.savefig('learned_adjacency_matrix_test_sample.png', dpi=300)
    plt.show()

# Plot the learned adjacency matrix
plot_learned_adjacency(model, adj, regions)

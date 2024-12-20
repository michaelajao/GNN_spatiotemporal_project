import os
import random
import math
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

########################################
# Seed & Device Configuration
########################################
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

########################################
# Hyperparameters
########################################
num_nodes = 7
num_features = 5  # Features: [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
num_timesteps_input = 14
num_timesteps_output = 7
k = 8
hidA = 32
hidR = 40  # Updated from 20 to 40
hidP = 1
n_layer = 4  # Reduced from 2
dropout = 0.5
learning_rate = 1e-3  # Adjusted
num_epochs = 1000
batch_size = 32
threshold_distance = 300  # km threshold for adjacency

########################################
# Reference Coordinates Correction
########################################
REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}

########################################
# Data Loading and Preprocessing
########################################
def load_and_correct_data(data, reference_coordinates):
    # Assign correct geographic coordinates
    for region, coords in reference_coordinates.items():
        data.loc[data['areaName'] == region, ['latitude', 'longitude']] = coords

    print("Unique latitudes:", data['latitude'].unique())
    print("Unique longitudes:", data['longitude'].unique())

    # Apply 7-day rolling mean to specified features
    rolling_features = ['new_confirmed', 'new_deceased', 'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']
    data[rolling_features] = data.groupby('areaName')[rolling_features].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Handle any missing values after rolling
    data[rolling_features] = data[rolling_features].fillna(0)
    
    # Sort data chronologically by region
    data.sort_values(['areaName', 'date'], inplace=True)

    return data

class NHSRegionDataset(Dataset):
    def __init__(self, data, num_timesteps_input, num_timesteps_output, transform=None, scaler=None):
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.transform = transform

        self.regions = self.data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        self.features = ['new_confirmed', 'new_deceased', 'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']

        # Pivot for time-series structure
        self.pivot = self.data.pivot(index='date', columns='region_idx', values=self.features)
        self.pivot.ffill(inplace=True)
        self.pivot.fillna(0, inplace=True)

        self.feature_array = self.pivot.values
        self.num_features = len(self.features)
        self.num_dates = self.feature_array.shape[0]
        self.feature_array = self.feature_array.reshape(self.num_dates, self.num_nodes, self.num_features)

        # Validate population consistency
        populations = self.data.groupby('areaName')['population'].unique()
        inconsistent_pop = populations[populations.apply(len) > 1]
        if not inconsistent_pop.empty:
            raise ValueError(f"Inconsistent population values in regions: {inconsistent_pop.index.tolist()}")

        # Remove normalization steps
        # if scaler:
        #     self.scaler = scaler
        #     self.feature_array = self.scaler.transform(self.feature_array.reshape(-1, self.num_features))
        #     self.feature_array = self.feature_array.reshape(self.num_dates, self.num_nodes, self.num_features)
        # else:
        #     self.scaler = None

    def __len__(self):
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx):
        X = self.feature_array[idx:idx + self.num_timesteps_input]  # (num_timesteps_input, num_nodes, num_features)
        Y = self.feature_array[idx + self.num_timesteps_input: idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def compute_geographic_adjacency(regions, latitudes, longitudes, threshold=threshold_distance):
    # Compute adjacency based on haversine distances
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in km
        return c * r

    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes))
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
    # Compute Laplacian matrix for GCN
    i_mat = torch.eye(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    o_mat = torch.ones(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    adj = torch.where(adj > 0, o_mat, adj)

    d_mat_out = torch.sum(adj, dim=2)
    d_mat = d_mat_out.unsqueeze(2) + 1e-12
    d_mat = torch.pow(d_mat, -1)
    d_mat = i_mat * d_mat

    laplace_mat = torch.bmm(d_mat, adj)
    return laplace_mat

########################################
# Model Definition
########################################
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
        # embedding: (batch_size, m, hidR)
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))

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
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)
        batch_size = x.size(0)
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
        # x: (batch_size, num_features, P, m)
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)

        x_global = self.conv_g(x)
        x = torch.cat([x_local, x_period, x_global], dim=1)
        x = self.activate(x)
        return x

class EpiGNN(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 num_features, 
                 num_timesteps_input,
                 num_timesteps_output, 
                 k=8, 
                 hidA=32, 
                 hidR=40,    # Updated hidR to 40
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

        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        self.WQ = nn.Linear(self.hidR, self.hidA)  # nn.Linear(40, 32)
        self.WK = nn.Linear(self.hidR, self.hidA)  # nn.Linear(40, 32)
        self.t_enc = nn.Linear(1, self.hidR)       # nn.Linear(1, 40)

        self.s_enc = nn.Linear(1, self.hidR)       # nn.Linear(1, 40)

        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.external_parameter)

        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)
        self.graphGen = GraphLearner(self.hidR)    # Update GraphLearner input size
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for _ in range(self.n)])

        # hidR * (n_layer +1): each GNN block outputs hidR, plus the original feat_emb
        self.output = nn.Linear(self.hidR * (self.n + 1), num_timesteps_output)  # nn.Linear(80, 7)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, states=None, dynamic_adj=None, index=None):
        # X: (batch_size, T, m, F)
        adj = adj.bool().float()
        batch_size = X.size(0)

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, self.m, self.m)

        # Backbone expects (batch_size, F, T, m)
        X_reshaped = X.permute(0, 3, 1, 2)
        temp_emb = self.backbone(X_reshaped)  # (batch_size, hidR, m)
        temp_emb = temp_emb.permute(0, 2, 1)  # (batch_size, m, hidR)

        # Global transmission risk encoding
        query = self.dropout_layer(self.WQ(temp_emb))  # (batch_size, m, hidA=32)

        key = self.dropout_layer(self.WK(temp_emb))    # (batch_size, m, hidA=32)

        attn = torch.bmm(query, key.transpose(1, 2))   # (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)   # (batch_size, m, 1)
        t_enc = self.dropout_layer(self.t_enc(attn))   # (batch_size, m, hidR=40)

        # Local transmission risk encoding
        d = torch.sum(adj, dim=1).unsqueeze(2)  # (batch_size, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))  # (batch_size, m, hidR=40)

        # Fusion
        feat_emb = temp_emb + t_enc + s_enc  # (batch_size, m, hidR=40)

        # External information (optional)
        if self.external_parameter is not None and index is not None:
            batch_ext = []
            zeros_mt = torch.zeros((self.m, self.m)).to(adj.device)
            for i in range(batch_size):
                offset = 20
                if i - offset >= 0:
                    idx = i - offset
                    batch_ext.append(self.external_parameter[index[i], :, :].unsqueeze(0))
                else:
                    batch_ext.append(zeros_mt.unsqueeze(0))
            extra_info = torch.cat(batch_ext, dim=0)
            external_info = F.relu(torch.mul(self.external_parameter, extra_info))
        else:
            external_info = 0

        # Graph learning
        d_mat = torch.bmm(torch.sum(adj, dim=1).unsqueeze(2), torch.sum(adj, dim=1).unsqueeze(1))  # (batch_size, m, m)
        d_mat = torch.sigmoid(torch.mul(self.d_gate, d_mat))
        spatial_adj = torch.mul(d_mat, adj)
        learned_adj = self.graphGen(feat_emb)  # (batch_size, m, m)

        if external_info != 0:
            adj = learned_adj + spatial_adj + external_info
        else:
            adj = learned_adj + spatial_adj

        laplace_adj = getLaplaceMat(batch_size, self.m, adj)

        # GNN layers
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = self.dropout_layer(layer(node_state, laplace_adj))
            node_state_list.append(node_state)

        node_state = torch.cat(node_state_list, dim=-1)  # (batch_size, m, hidR * n_layer=40)
        node_state = torch.cat([node_state, feat_emb], dim=-1)  # (batch_size, m, 80)

        # Final prediction
        res = self.output(node_state)  # (batch_size, m, 7)
        res = res.transpose(1, 2)    # (batch_size, 7, m)
        return res

########################################
# Data Loading and Normalization
########################################
csv_path = '../data/merged_nhs_covid_data.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

data = pd.read_csv(csv_path, parse_dates=['date'])
data = load_and_correct_data(data, REFERENCE_COORDINATES)

# Initialize dataset without normalization
dataset = NHSRegionDataset(data, num_timesteps_input=num_timesteps_input, num_timesteps_output=num_timesteps_output)
print(f"Total samples in dataset: {len(dataset)}")

# Split the dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Remove normalization steps
# scaler = StandardScaler()
# train_features = train_dataset.dataset.feature_array[:train_size].reshape(-1, num_features)
# scaler.fit(train_features)

# # Apply scaler to all splits
# for split in [train_dataset, val_dataset, test_dataset]:
#     split.dataset.feature_array = scaler.transform(split.dataset.feature_array.reshape(-1, num_features)).reshape(split.dataset.feature_array.shape)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

########################################
# Compute Geographic Adjacency
########################################
regions = dataset.regions.tolist()
latitudes = [data[data['areaName'] == region]['latitude'].iloc[0] for region in regions]
longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

adj = compute_geographic_adjacency(regions, latitudes, longitudes).to(device)
print("Adjacency matrix:")
print(adj.cpu().numpy())

# Plotting the geographic adjacency graph
import networkx as nx

adj_np = adj.cpu().numpy()

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(adj_np)

# Assign region names as node labels
mapping = {i: region for i, region in enumerate(regions)}
G = nx.relabel_nodes(G, mapping)

# Set positions based on geographic coordinates
pos = {region: (longitudes[i], latitudes[i]) for i, region in enumerate(regions)}

# Plot the graph
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray')
plt.title('Geographic Adjacency Graph')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

########################################
# Model Initialization
########################################
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

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight_decay
criterion = nn.MSELoss()

########################################
# Training Loop
########################################
best_val_loss = float('inf')
early_stopping_patience = 10
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()

        batch_size_current = batch_X.size(0)
        batch_adj = adj.unsqueeze(0).repeat(batch_size_current, 1, 1)

        pred = model(batch_X, batch_adj)
        loss = criterion(pred, batch_Y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0
    all_val_preds = []
    all_val_actuals = []
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            batch_size_current = batch_X.size(0)
            batch_adj = adj.unsqueeze(0).repeat(batch_size_current, 1, 1)

            pred = model(batch_X, batch_adj)
            loss = criterion(pred, batch_Y)
            epoch_val_loss += loss.item()
            
            all_val_preds.append(pred.cpu().numpy())
            all_val_actuals.append(batch_Y.cpu().numpy())

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Concatenate the lists into arrays
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_actuals = np.concatenate(all_val_actuals, axis=0)

    # Reshape arrays to (total_samples, num_nodes)
    all_val_preds = all_val_preds.reshape(-1, num_nodes)
    all_val_actuals = all_val_actuals.reshape(-1, num_nodes)
    
    # Calculate R² for validation
    r2 = r2_score(all_val_actuals, all_val_preds, multioutput='raw_values')
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val R²: {r2}")

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

# Plot training and validation loss
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

########################################
# Load Best Model and Evaluate on Test Set
########################################
# Addressing FutureWarning by setting weights_only=True
model.load_state_dict(torch.load('best_epignn_adapted_model.pth', map_location=device), strict=True)
model.eval()

test_loss = 0
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
print(f"Test Loss: {avg_test_loss:.4f}")

# Concatenate predictions and actual values
all_preds = torch.cat(all_preds, dim=0)
all_actuals = torch.cat(all_actuals, dim=0)

# Reshape tensors for inverse transformation
all_preds_np = all_preds.cpu().numpy()
all_actuals_np = all_actuals.cpu().numpy()

# Proceed to plotting using the inverse-transformed data
num_plots = 3
for i in range(min(num_plots, all_preds_np.shape[0])):
    sample_pred = all_preds_np[i]
    sample_actual = all_actuals_np[i]

    plt.figure(figsize=(12, 8))
    for node_idx, region in enumerate(regions):
        plt.plot(range(num_timesteps_output), sample_actual[:, node_idx], label=f'Actual - {region}')
        plt.plot(range(num_timesteps_output), sample_pred[:, node_idx], '--', label=f'Predicted - {region}')

    plt.xlabel('Future Timestep')
    plt.ylabel('COVID Occupied MV Beds')
    plt.title(f'Sample {i+1}: Predictions vs Actual (Original Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Save final model
torch.save(model.state_dict(), 'epignn_adapted_model_final.pth')
print("Final model saved as 'epignn_adapted_model_final.pth'.")

# Compute additional metrics (MAE, R²)
preds_flat = all_preds.view(-1, num_nodes).numpy()
actuals_flat = all_actuals.view(-1, num_nodes).numpy()

mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
r2_per_node = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

for idx, region in enumerate(regions):
    print(f"Region: {region}, MAE: {mae_per_node[idx]:.4f}, R2 Score: {r2_per_node[idx]:.4f}")

########################################
# Visualizing the Learned Adjacency Matrix
########################################
# Select a single sample for visualization
example_X, _ = next(iter(test_loader))
example_X = example_X.to(device)
single_sample_X = example_X[0].unsqueeze(0)  # Shape: (1, T, m, F)

with torch.no_grad():
    # Pass through backbone
    emb_for_adj = model.backbone(single_sample_X.permute(0,3,1,2))  # (1, hidR, m)
    emb_for_adj = emb_for_adj.permute(0, 2, 1)  # (1, m, hidR)
    
    # Pass through GraphLearner
    learned_adj = model.graphGen(emb_for_adj)  # (1, m, m)
    learned_adj = learned_adj.cpu().numpy()[0]  # (m, m)

# Plotting
plt.figure(figsize=(8,6))
sns.heatmap(learned_adj, annot=True, fmt=".2f", cmap='viridis', xticklabels=regions, yticklabels=regions)
plt.title('Learned Adjacency Matrix (Test Sample)')
plt.xlabel('Regions')
plt.ylabel('Regions')
plt.show()

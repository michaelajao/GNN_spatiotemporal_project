# FILE: train.py

import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset
from tqdm import tqdm
from epiweeks import Week
from haversine import haversine
# from utils import date_today, gravity_law_commute_dist  # Ensure these are defined in utils.py
# from model import STAN  # Import STAN from model.py
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Ensure reproducibility
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

# Device configuration
device = torch.device('cpu')
print(f'Using device: {device}')

# Set environment variables
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

# read data
data = pd.read_csv('./data/merged_nhs_covid_data.csv')
data

# Load and merge data
raw_data = pickle.load(open('./data/state_covid_data.pickle','rb'))
raw_data.to_csv('./data/state_covid_data.csv', index=False)
pop_data = pd.read_csv('./data/uszips.csv')
pop_data = pop_data.groupby('state_name').agg({
    'population':'sum',
    'density':'mean',
    'lat':'mean',
    'lng':'mean'
}).reset_index()
raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

# Generate location similarity based on Gravity Law
loc_list = list(raw_data['state'].unique())
loc_dist_map = {}

for each_loc in loc_list:
    loc_dist_map[each_loc] = {}
    for each_loc2 in loc_list:
        lat1 = raw_data[raw_data['state'] == each_loc]['lat'].unique()[0]
        lng1 = raw_data[raw_data['state'] == each_loc]['lng'].unique()[0]
        pop1 = raw_data[raw_data['state'] == each_loc]['population'].unique()[0]

        lat2 = raw_data[raw_data['state'] == each_loc2]['lat'].unique()[0]
        lng2 = raw_data[raw_data['state'] == each_loc2]['lng'].unique()[0]
        pop2 = raw_data[raw_data['state'] == each_loc2]['population'].unique()[0]

        loc_dist_map[each_loc][each_loc2] = gravity_law_commute_dist(
            lat1, lng1, pop1,
            lat2, lng2, pop2,
            r=0.5
        )

num_locations = len(loc_list)
print(f"Number of unique locations: {num_locations}")

# Convert loc_dist_map to a DataFrame for visualization
loc_dist_df = pd.DataFrame(loc_dist_map).fillna(0)

# Create a heatmap of location similarities
plt.figure(figsize=(15, 12))
sns.heatmap(loc_dist_df, cmap='viridis', linewidths=.5)
plt.title('Location Similarity Based on Gravity Law')
plt.xlabel('Location')
plt.ylabel('Location')
plt.show()

# Generate Adjacency Map based on distance threshold
dist_threshold = 18

for each_loc in loc_dist_map:
    loc_dist_map[each_loc] = {k: v for k, v in sorted(loc_dist_map[each_loc].items(), key=lambda item: item[1], reverse=True)}

adj_map = {}
for each_loc in loc_dist_map:
    adj_map[each_loc] = []
    for i, (each_loc2, distance) in enumerate(loc_dist_map[each_loc].items()):
        if distance > dist_threshold:
            if i < 3:  # Keep top 3 connections
                adj_map[each_loc].append(each_loc2)
            else:
                break
        else:
            if i < 1:  # Keep top 1 connection if distance below threshold
                adj_map[each_loc].append(each_loc2)
            else:
                break

# Create edge indices for PyTorch Geometric
edge_index = [[], []]
state_to_index = {state: idx for idx, state in enumerate(loc_list)}
for state, neighbors in adj_map.items():
    for neighbor in neighbors:
        edge_index[0].append(state_to_index[state])
        edge_index[1].append(state_to_index[neighbor])

edge_index = torch.tensor(edge_index, dtype=torch.long)

# Add self-loops
self_loops = torch.arange(0, num_locations, dtype=torch.long).unsqueeze(0).repeat(2,1)
edge_index = torch.cat([edge_index, self_loops], dim=1)

# Create a PyTorch Geometric Data object
data = Data(edge_index=edge_index, num_nodes=num_locations).to(device)

# Visualize the adjacency graph
G_nx = nx.DiGraph()
# Add nodes
for state in adj_map.keys():
    G_nx.add_node(state)

# Add edges
for state, neighbors in adj_map.items():
    for neighbor in neighbors:
        G_nx.add_edge(state, neighbor)

# Plot the adjacency graph
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G_nx, seed=RANDOM_SEED)  # For consistent layout
nx.draw(
    G_nx,
    pos,
    with_labels=True,
    node_size=3000,
    node_color='skyblue',
    font_size=10,
    font_weight='bold',
    edge_color='gray'
)
plt.title('Adjacency Map of States Based on Gravity Law')
plt.show()

# Preprocess features
active_cases = []
confirmed_cases = []
new_cases = []
death_cases = []
static_feat = []

for each_loc in loc_list:
    active = raw_data[raw_data['state'] == each_loc]['active'].values
    confirmed = raw_data[raw_data['state'] == each_loc]['confirmed'].values
    new = raw_data[raw_data['state'] == each_loc]['new_cases'].values
    deaths = raw_data[raw_data['state'] == each_loc]['deaths'].values
    static = raw_data[raw_data['state'] == each_loc][['population','density','lng','lat']].values
    active_cases.append(active)
    confirmed_cases.append(confirmed)
    new_cases.append(new)
    death_cases.append(deaths)
    static_feat.append(static)

active_cases = np.array(active_cases)  # [num_loc, timestep]
confirmed_cases = np.array(confirmed_cases)  # [num_loc, timestep]
death_cases = np.array(death_cases)  # [num_loc, timestep]
new_cases = np.array(new_cases)  # [num_loc, timestep]
static_feat = np.array(static_feat)[:, 0, :]  # [num_loc, 4]
recovered_cases = confirmed_cases - active_cases - death_cases  # [num_loc, timestep]
susceptible_cases = np.expand_dims(static_feat[:, 0], -1) - active_cases - recovered_cases  # [num_loc, timestep]

# Compute differences for dynamic features
dI = np.concatenate((np.zeros((active_cases.shape[0],1), dtype=np.float32), np.diff(active_cases, axis=1)), axis=-1)  # [num_loc, timestep]
dR = np.concatenate((np.zeros((recovered_cases.shape[0],1), dtype=np.float32), np.diff(recovered_cases, axis=1)), axis=-1)  # [num_loc, timestep]
dS = np.concatenate((np.zeros((susceptible_cases.shape[0],1), dtype=np.float32), np.diff(susceptible_cases, axis=1)), axis=-1)  # [num_loc, timestep]

# Build normalizer
normalizer = {'S':{}, 'I':{}, 'R':{}, 'dS':{}, 'dI':{}, 'dR':{}}

for i, each_loc in enumerate(loc_list):
    normalizer['S'][each_loc] = (np.mean(susceptible_cases[i]), np.std(susceptible_cases[i]) + 1e-5)
    normalizer['I'][each_loc] = (np.mean(active_cases[i]), np.std(active_cases[i]) + 1e-5)
    normalizer['R'][each_loc] = (np.mean(recovered_cases[i]), np.std(recovered_cases[i]) + 1e-5)
    normalizer['dI'][each_loc] = (np.mean(dI[i]), np.std(dI[i]) + 1e-5)
    normalizer['dR'][each_loc] = (np.mean(dR[i]), np.std(dR[i]) + 1e-5)
    normalizer['dS'][each_loc] = (np.mean(dS[i]), np.std(dS[i]) + 1e-5)

# Prepare data for training, validation, and testing
def prepare_data(data, sum_I, sum_R, history_window=5, pred_window=15, slide_step=5):
    """
    Prepares data for training, validation, and testing.

    Args:
        data (np.ndarray): Dynamic features [num_loc, timestep, n_feat].
        sum_I (np.ndarray): Cumulative infected cases [num_loc, timestep].
        sum_R (np.ndarray): Cumulative recovered cases [num_loc, timestep].
        history_window (int, optional): Number of historical time steps. Defaults to 5.
        pred_window (int, optional): Number of prediction time steps. Defaults to 15.
        slide_step (int, optional): Step size for sliding window. Defaults to 5.

    Returns:
        Tuple: Prepared features and targets.
    """
    # Data shape: n_loc, timestep, n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]
    n_feat = data.shape[2]

    x = []
    y_I = []
    y_R = []
    last_I = []
    last_R = []
    concat_I = []
    concat_R = []
    for i in range(0, timestep - history_window - pred_window +1, slide_step):
        x.append(data[:, i:i + history_window, :].reshape((n_loc, history_window * n_feat)))  # [num_loc, history_window * n_feat]

        concat_I.append(sum_I[:, i + history_window -1])  # [num_loc,]
        concat_R.append(sum_R[:, i + history_window -1])  # [num_loc,]
        last_I.append(sum_I[:, i + history_window -1])  # [num_loc,]
        last_R.append(sum_R[:, i + history_window -1])  # [num_loc,]

        y_I.append(sum_I[:, i + history_window:i + history_window + pred_window])  # [num_loc, pred_window]
        y_R.append(sum_R[:, i + history_window:i + history_window + pred_window])  # [num_loc, pred_window]

    x = np.array(x, dtype=np.float32)  # [num_samples, num_loc, history_window * n_feat]
    last_I = np.array(last_I, dtype=np.float32)  # [num_samples, num_loc]
    last_R = np.array(last_R, dtype=np.float32)  # [num_samples, num_loc]
    concat_I = np.array(concat_I, dtype=np.float32)  # [num_samples, num_loc]
    concat_R = np.array(concat_R, dtype=np.float32)  # [num_samples, num_loc]
    y_I = np.array(y_I, dtype=np.float32)  # [num_samples, num_loc, pred_window]
    y_R = np.array(y_R, dtype=np.float32)  # [num_samples, num_loc, pred_window]
    return x, last_I, last_R, concat_I, concat_R, y_I, y_R

valid_window = 25
test_window = 25

history_window = 6
pred_window = 15
slide_step = 5

normalize = True

dynamic_feat = np.concatenate(
    (
        np.expand_dims(dI, axis=-1),
        np.expand_dims(dR, axis=-1),
        np.expand_dims(dS, axis=-1)
    ),
    axis=-1
)  # [num_loc, timestep, 3]

# Normalize dynamic features
if normalize:
    for i, each_loc in enumerate(loc_list):
        dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer['dI'][each_loc][0]) / normalizer['dI'][each_loc][1]
        dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer['dR'][each_loc][0]) / normalizer['dR'][each_loc][1]
        dynamic_feat[i, :, 2] = (dynamic_feat[i, :, 2] - normalizer['dS'][each_loc][0]) / normalizer['dS'][each_loc][1]

# Convert normalization parameters to torch tensors and move to device
dI_mean = torch.tensor([normalizer['dI'][loc][0] for loc in loc_list], dtype=torch.float32).to(device)
dI_std = torch.tensor([normalizer['dI'][loc][1] for loc in loc_list], dtype=torch.float32).to(device)
dR_mean = torch.tensor([normalizer['dR'][loc][0] for loc in loc_list], dtype=torch.float32).to(device)
dR_std = torch.tensor([normalizer['dR'][loc][1] for loc in loc_list], dtype=torch.float32).to(device)

# Split data into train, validation, and test sets
train_feat = dynamic_feat[:, :-valid_window - test_window, :]  # [num_loc, train_timestep, 3]
val_feat = dynamic_feat[:, -valid_window - test_window:-test_window, :]  # [num_loc, val_timestep, 3]
test_feat = dynamic_feat[:, -test_window:, :]  # [num_loc, test_timestep, 3]

train_sum_I = active_cases[:, :-valid_window - test_window]  # [num_loc, train_timestep]
train_sum_R = recovered_cases[:, :-valid_window - test_window]  # [num_loc, train_timestep]

val_sum_I = active_cases[:, -valid_window - test_window:-test_window]  # [num_loc, val_timestep]
val_sum_R = recovered_cases[:, -valid_window - test_window:-test_window]  # [num_loc, val_timestep]

test_sum_I = active_cases[:, -test_window:]  # [num_loc, test_timestep]
test_sum_R = recovered_cases[:, -test_window:]  # [num_loc, test_timestep]

train_x, train_I, train_R, train_cI, train_cR, train_yI, train_yR = prepare_data(
    train_feat,
    train_sum_I,
    train_sum_R,
    history_window,
    pred_window,
    slide_step
)
val_x, val_I, val_R, val_cI, val_cR, val_yI, val_yR = prepare_data(
    val_feat,
    val_sum_I,
    val_sum_R,
    history_window,
    pred_window,
    slide_step
)
test_x, test_I, test_R, test_cI, test_cR, test_yI, test_yR = prepare_data(
    test_feat,
    test_sum_I,
    test_sum_R,
    history_window,
    pred_window,
    slide_step
)

# Initialize the STAN model
in_dim = 3 * history_window
hidden_dim1 = 32
hidden_dim2 = 32
gru_dim = 32
num_heads = 1
pred_window = pred_window
device = device

model = STAN(
    num_nodes=num_locations,
    num_features=in_dim,
    num_timesteps_input=history_window,
    num_timesteps_output=pred_window,
    population=1e10,  # Adjust if a different population is needed
    gat_dim1=hidden_dim1,
    gat_dim2=hidden_dim2,
    gru_dim=gru_dim,
    num_heads=num_heads,
    device=device
).to(device)

# Define optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

# Prepare N as [num_loc,1], already done above
N = torch.tensor(static_feat[:, 0], dtype=torch.float32).to(device).unsqueeze(-1)  # [num_loc, 1]

# Training parameters
all_loss = []
file_name = './save/stan.pth'
min_loss = 1e10

epoch_count = 50 if normalize else 300
scale = 0.1

# Create save directory if it doesn't exist
os.makedirs('./save/', exist_ok=True)

# Define a simple dataset class for batching
class SIRDataset(Dataset):
    def __init__(self, x, cI, cR, N, I, R, dI, dR, yI, yR):
        self.x = x  # [num_samples, num_loc, history_window * n_feat]
        self.cI = cI  # [num_samples, num_loc]
        self.cR = cR
        self.N = N  # [num_loc, 1]
        self.I = I  # [num_samples, num_loc]
        self.R = R
        self.dI = dI  # [num_samples, num_loc]
        self.dR = dR
        self.yI = yI  # [num_samples, num_loc, pred_window]
        self.yR = yR

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        node_features = torch.tensor(self.x[idx], dtype=torch.float32)  # Convert to tensor
        cI = torch.tensor(self.cI[idx], dtype=torch.float32)            # Convert to tensor
        cR = torch.tensor(self.cR[idx], dtype=torch.float32)            # Convert to tensor
        I = torch.tensor(self.I[idx], dtype=torch.float32)              # Convert to tensor
        R = torch.tensor(self.R[idx], dtype=torch.float32)              # Convert to tensor
        dI = torch.tensor(self.dI[idx], dtype=torch.float32)            # Convert to tensor
        dR = torch.tensor(self.dR[idx], dtype=torch.float32)            # Convert to tensor
        yI = torch.tensor(self.yI[idx], dtype=torch.float32)            # Convert to tensor
        yR = torch.tensor(self.yR[idx], dtype=torch.float32)            # Convert to tensor

        # Create a Data object for each sample
        data_sample = Data(
            x=node_features.clone().detach(),  # [num_loc, history_window * n_feat]
            edge_index=data.edge_index.clone()
        )
        data_sample.cI = cI.clone().detach()
        data_sample.cR = cR.clone().detach()
        data_sample.N = self.N.clone().detach()  # Convert and clone
        data_sample.I = I.clone().detach()
        data_sample.R = R.clone().detach()
        data_sample.dI = dI.clone().detach()
        data_sample.dR = dR.clone().detach()
        data_sample.yI = yI.clone().detach()
        data_sample.yR = yR.clone().detach()

        return data_sample

# Create datasets
train_dataset = SIRDataset(train_x, train_cI, train_cR, N, train_I, train_R, train_x[:, :, 0], train_x[:, :, 1], train_yI, train_yR)
val_dataset = SIRDataset(val_x, val_cI, val_cR, N, val_I, val_R, val_x[:, :, 0], val_x[:, :, 1], val_yI, val_yR)
test_dataset = SIRDataset(test_x, test_cI, test_cR, N, test_I, test_R, test_x[:, :, 0], test_x[:, :, 1], test_yI, test_yR)

# Create PyTorch Geometric DataLoaders
batch_size = 32  # Adjust based on your memory constraints
train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in tqdm(range(epoch_count), desc='Training Epochs'):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move batch to device
        batch = batch.to(device)

        # Extract tensors
        dynamic = batch.x  # [batch_size * num_loc, history_window * n_feat]
        adj = data.edge_index  # Adjacency information from train.py
        states = torch.stack([batch.I, batch.R], dim=-1)  # Combine I and R for states
        N_batch = batch.N  # [num_loc, 1]
        yI = batch.yI  # [batch_size * num_loc, pred_window]
        yR = batch.yR  # [batch_size * num_loc, pred_window]

        # Pass through the model
        predictions, phy_predictions = model(
            X=dynamic,       # [batch_size * num_loc, history_window * n_feat]
            adj=edge_index,  # [2, num_edges]
            states=states,   # [batch_size * num_loc, 2]
            N=N_batch        # [num_loc, 1]
        )

        # Normalize physical predictions if required
        if normalize:
            # Expand normalization tensors to match phy_I and phy_R dimensions
            dI_std_expanded = dI_std.unsqueeze(1).repeat(1, pred_window)
            dI_mean_expanded = dI_mean.unsqueeze(1).repeat(1, pred_window)
            dR_std_expanded = dR_std.unsqueeze(1).repeat(1, pred_window)
            dR_mean_expanded = dR_mean.unsqueeze(1).repeat(1, pred_window)

            phy_predictions[:, :, 0] = (phy_predictions[:, :, 0] * dI_std_expanded) + dI_mean_expanded  # [batch_size * num_loc, pred_window]
            phy_predictions[:, :, 1] = (phy_predictions[:, :, 1] * dR_std_expanded) + dR_mean_expanded  # [batch_size * num_loc, pred_window]

        # Compute loss
        loss = (
            criterion(predictions[:, :, 0], yI) +
            criterion(predictions[:, :, 1], yR) +
            scale * criterion(phy_predictions[:, :, 0], yI) +
            scale * criterion(phy_predictions[:, :, 1], yR)
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    all_loss.append(avg_epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = batch.to(device)

            # Extract tensors
            dynamic = batch.x  # [batch_size * num_loc, history_window * n_feat]
            adj = data.edge_index  # Adjacency information from train.py
            states = torch.stack([batch.I, batch.R], dim=-1)  # Combine I and R for states
            N_batch = batch.N  # [num_loc, 1]
            yI = batch.yI  # [batch_size * num_loc, pred_window]
            yR = batch.yR  # [batch_size * num_loc, pred_window]

            # Pass through the model
            predictions, phy_predictions = model(
                X=dynamic,       # [batch_size * num_loc, history_window * n_feat]
                adj=edge_index,  # [2, num_edges]
                states=states,   # [batch_size * num_loc, 2]
                N=N_batch        # [num_loc, 1]
            )

            # Normalize physical predictions if required
            if normalize:
                # Expand normalization tensors to match phy_I and phy_R dimensions
                dI_std_expanded = dI_std.unsqueeze(1).repeat(1, pred_window)
                dI_mean_expanded = dI_mean.unsqueeze(1).repeat(1, pred_window)
                dR_std_expanded = dR_std.unsqueeze(1).repeat(1, pred_window)
                dR_mean_expanded = dR_mean.unsqueeze(1).repeat(1, pred_window)

                phy_predictions[:, :, 0] = (phy_predictions[:, :, 0] * dI_std_expanded) + dI_mean_expanded  # [batch_size * num_loc, pred_window]
                phy_predictions[:, :, 1] = (phy_predictions[:, :, 1] * dR_std_expanded) + dR_mean_expanded  # [batch_size * num_loc, pred_window]

            # Compute loss
            loss = (
                criterion(predictions[:, :, 0], yI) +
                criterion(predictions[:, :, 1], yR) +
                scale * criterion(phy_predictions[:, :, 0], yI) +
                scale * criterion(phy_predictions[:, :, 1], yR)
            )

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Save the model if validation loss has decreased
    if avg_val_loss < min_loss:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, file_name)
        min_loss = avg_val_loss

    # Print loss every epoch
    print(f"Epoch {epoch+1}/{epoch_count}, Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(all_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Load the best model
checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()

# Make predictions with the test set
test_pred_I = []
test_pred_R = []
test_phy_I = []
test_phy_R = []

with torch.no_grad():
    for batch in test_loader:
        # Move batch to device
        batch = batch.to(device)

        # Extract tensors
        dynamic = batch.x  # [batch_size * num_loc, history_window * n_feat]
        adj = data.edge_index  # Adjacency information from train.py
        states = torch.stack([batch.I, batch.R], dim=-1)  # Combine I and R for states
        N_batch = batch.N  # [num_loc, 1]
        yI = batch.yI  # [batch_size * num_loc, pred_window]
        yR = batch.yR  # [batch_size * num_loc, pred_window]

        # Pass through the model
        predictions, phy_predictions = model(
            X=dynamic,       # [batch_size * num_loc, history_window * n_feat]
            adj=edge_index,  # [2, num_edges]
            states=states,   # [batch_size * num_loc, 2]
            N=N_batch        # [num_loc, 1]
        )

        # Normalize physical predictions if required
        if normalize:
            # Expand normalization tensors to match phy_I and phy_R dimensions
            dI_std_expanded = dI_std.unsqueeze(1).repeat(1, pred_window)
            dI_mean_expanded = dI_mean.unsqueeze(1).repeat(1, pred_window)
            dR_std_expanded = dR_std.unsqueeze(1).repeat(1, pred_window)
            dR_mean_expanded = dR_mean.unsqueeze(1).repeat(1, pred_window)

            phy_predictions[:, :, 0] = (phy_predictions[:, :, 0] * dI_std_expanded) + dI_mean_expanded  # [batch_size * num_loc, pred_window]
            phy_predictions[:, :, 1] = (phy_predictions[:, :, 1] * dR_std_expanded) + dR_mean_expanded  # [batch_size * num_loc, pred_window]

        # Cumulatively sum predictions
        pred_I_batch = predictions[:, :, 0].cumsum(dim=1) + batch.I.unsqueeze(1)  # [batch_size * num_loc, pred_window]
        pred_R_batch = predictions[:, :, 1].cumsum(dim=1) + batch.R.unsqueeze(1)  # [batch_size * num_loc, pred_window]

        test_pred_I.append(pred_I_batch.cpu().numpy())
        test_pred_R.append(pred_R_batch.cpu().numpy())
        test_phy_I.append(phy_predictions[:, :, 0].cpu().numpy())
        test_phy_R.append(phy_predictions[:, :, 1].cpu().numpy())

# Concatenate all batches
test_pred_I = np.concatenate(test_pred_I, axis=0)  # [num_samples * num_loc, pred_window]
test_pred_R = np.concatenate(test_pred_R, axis=0)
test_phy_I = np.concatenate(test_phy_I, axis=0)
test_phy_R = np.concatenate(test_phy_R, axis=0)

# Get real y values
def get_real_y(data, history_window=5, pred_window=15, slide_step=5):
    """
    Retrieves real target values.

    Args:
        data (np.ndarray): Dynamic features [num_loc, timestep, n_feat].
        history_window (int, optional): Number of historical time steps. Defaults to 5.
        pred_window (int, optional): Number of prediction time steps. Defaults to 15.
        slide_step (int, optional): Step size for sliding window. Defaults to 5.

    Returns:
        np.ndarray: Real target values [num_samples * num_loc, pred_window].
    """
    # Data shape: n_loc, timestep, n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]
    n_feat = data.shape[2]

    y = []
    for i in range(0, timestep - history_window - pred_window +1, slide_step):
        y.append(data[:, i + history_window:i + history_window + pred_window])

    y = np.array(y, dtype=np.float32).reshape(-1, y[0].shape[0], y[0].shape[1])  # [num_samples, num_loc, pred_window]
    return y.reshape(-1, pred_window)  # [num_samples * num_loc, pred_window]

I_true = get_real_y(active_cases[:], history_window, pred_window, slide_step)

# Plot predictions vs true values for Active Cases for each location
for loc_idx, loc_name in enumerate(loc_list):
    plt.figure(figsize=(12, 6))
    plt.plot(I_true[:, loc_idx, :].flatten(), c='r', label='Ground truth')
    plt.plot(test_pred_I[:, loc_idx, :].flatten(), c='b', label='Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Active Cases')
    plt.title(f'Active Cases: True vs Predicted for {loc_name}')
    plt.legend()
    plt.show()

# Optionally, save predictions to a file
# np.save('./save/pred_I.npy', test_pred_I)
# np.save('./save/I_true.npy', I_true)

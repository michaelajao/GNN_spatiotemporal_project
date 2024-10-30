import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.nn import GCNConv
from haversine import haversine
from tqdm.auto import tqdm
from datetime import timedelta
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler

# =======================
# Step 1: Set Parameters
# =======================

WINDOW_SIZES = [3, 7]  # Time windows to experiment with
START_DATE = '2020-06-01'
END_DATE = '2021-12-01'
FEATURE_COLUMNS = ['confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']
TARGET_COLUMN = 'hospitalization'  # Forecasting hospitalization
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# =======================
# Step 2: Load and Preprocess Data
# =======================

# 2.1 Load COVID-19 Data
covid_data_path = '../../data/processed/processed_covid_data.pickle'  # Update the path accordingly
with open(covid_data_path, 'rb') as f:
    raw_data = pickle.load(f)

# 2.2 Load Population Data
pop_data_path = '../../data/uszips.csv'  # Update the path accordingly
pop_data = pd.read_csv(pop_data_path)

# 2.3 Aggregate Population Data by State
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

# 2.4 Merge COVID Data with Population Data
raw_data = pd.merge(
    raw_data,
    pop_data,
    how='inner',
    left_on='state',
    right_on='state_name'
)

# 2.5 Convert 'date_today' Column to Datetime
raw_data['date_today'] = pd.to_datetime(raw_data['date_today'])

# 2.6 Handle Missing Values in 'hospitalization' Column
raw_data['hospitalization'] = raw_data['hospitalization'].ffill()
raw_data['hospitalization'] = raw_data['hospitalization'].fillna(0)

# 2.7 Filter Data Between START_DATE and END_DATE
raw_data = raw_data[
    (raw_data['date_today'] >= pd.to_datetime(START_DATE)) &
    (raw_data['date_today'] <= pd.to_datetime(END_DATE))
].reset_index(drop=True)

# 2.8 Map Dates to Weeks and Aggregate Data
raw_data['week'] = raw_data['date_today'].dt.to_period('W').apply(lambda r: r.start_time)
raw_data = raw_data.groupby(['week', 'state']).agg({
    'confirmed': 'sum',
    'deaths': 'sum',
    'recovered': 'sum',
    'active': 'sum',
    'hospitalization': 'sum',
    'new_cases': 'sum',
    'population': 'first',
    'density': 'first',
    'lat': 'first',
    'lng': 'first'
}).reset_index()

# Sort data
raw_data = raw_data.sort_values(by=['week', 'state']).reset_index(drop=True)

# =======================
# Step 3: Data Normalization
# =======================

# Initialize scalers
scalers = {}
for feature in FEATURE_COLUMNS:
    scalers[feature] = MinMaxScaler()
    raw_data[feature] = scalers[feature].fit_transform(raw_data[[feature]])

# Normalize the target
scaler_target = MinMaxScaler()
raw_data[TARGET_COLUMN] = scaler_target.fit_transform(raw_data[[TARGET_COLUMN]])

# =======================
# Step 4: Prepare Time Series Data
# =======================

class COVIDDataset(InMemoryDataset):
    def __init__(self, root, raw_data, window_size, transform=None, pre_transform=None):
        self.raw_data = raw_data
        self.window_size = window_size
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return [f'covid_dataset_w{self.window_size}.pt']
    
    def process(self):
        data_list = []
        
        states = self.raw_data['state'].unique()
        state_to_id = {state: idx for idx, state in enumerate(states)}
        
        # Get latitudes and longitudes for each state
        state_info = self.raw_data.groupby('state').agg({
            'lat': 'first',
            'lng': 'first'
        }).reset_index()
        state_info['node_id'] = state_info['state'].map(state_to_id)
        state_info = state_info.sort_values('node_id').reset_index(drop=True)
        latitudes = state_info['lat'].values
        longitudes = state_info['lng'].values
        
        num_states = len(states)
        distance_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(i+1, num_states):
                distance = haversine((latitudes[i], longitudes[i]), (latitudes[j], longitudes[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Define edges based on K nearest neighbors
        K_NEIGHBORS = 5
        neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='precomputed')
        neighbors.fit(distance_matrix)
        distances, indices = neighbors.kneighbors(distance_matrix)
        edge_index = []
        for i in range(num_states):
            for j in indices[i, 1:]:  # Skip the first neighbor (itself)
                edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Make the graph undirected
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        # Edge attributes
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)
        
        # Process data for each week
        all_weeks = self.raw_data['week'].unique()
        all_weeks = np.sort(all_weeks)

        for i in tqdm(range(self.window_size, len(all_weeks)), desc='Processing weeks'):
            weeks_window = all_weeks[i - self.window_size:i]
            target_week = all_weeks[i]
            
            window_data = self.raw_data[self.raw_data['week'].isin(weeks_window)]
            target_data = self.raw_data[self.raw_data['week'] == target_week]
            
            x_list = []
            y_list = []
            for idx in range(num_states):
                state = state_info.loc[idx, 'state']
                state_window_data = window_data[window_data['state'] == state]
                if len(state_window_data) == self.window_size:
                    x_seq = state_window_data[FEATURE_COLUMNS].values
                    x_list.append(x_seq)
                    y_value = target_data[target_data['state'] == state][TARGET_COLUMN].values
                    if len(y_value) == 1:
                        y_list.append(y_value[0])
                    else:
                        y_list.append(np.nan)
                else:
                    x_list.append(None)
                    y_list.append(np.nan)
            
            valid_indices = [idx for idx in range(num_states) if x_list[idx] is not None and not np.isnan(y_list[idx])]
            if len(valid_indices) == 0:
                continue  # Skip if no valid states
            
            x = []
            y = []
            for idx in valid_indices:
                x.append(x_list[idx])
                y.append(y_list[idx])
            
            x = torch.tensor(x, dtype=torch.float)  # [num_nodes, window_size, num_features]
            y = torch.tensor(y, dtype=torch.float)  # [num_nodes]
            
            idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
            mask = np.isin(edge_index[0], valid_indices) & np.isin(edge_index[1], valid_indices)
            edge_index_filtered = edge_index[:, mask]
            # Remap node indices
            edge_index_filtered = torch.tensor([[idx_map[idx.item()] for idx in edge_index_filtered[0]],
                                                [idx_map[idx.item()] for idx in edge_index_filtered[1]]],
                                               dtype=torch.long)
            edge_attr_filtered = edge_attr[mask]
            
            data = Data(
                x=x,
                y=y,
                edge_index=edge_index_filtered,
                edge_attr=edge_attr_filtered,
                week=target_week
            )
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# =======================
# Step 5: Create Datasets and Data Loaders
# =======================

for window_size in WINDOW_SIZES:
    dataset = COVIDDataset(root=f'../../data/covid_dataset_w{window_size}', raw_data=raw_data, window_size=window_size)
    
    num_samples = len(dataset)
    train_end = int(num_samples * TRAIN_RATIO)
    val_end = int(num_samples * (TRAIN_RATIO + VAL_RATIO))
    
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]
    
    # Create DataLoaders with batch_size=1
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # =======================
    # Step 6: Define the Model
    # =======================
    
    class GraphLearner(nn.Module):
        def __init__(self, in_dim, hid_dim):
            super(GraphLearner, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim)
            )
        
        def forward(self, x):
            h = self.fc(x)
            scores = torch.mm(h, h.t())
            adj = F.relu(scores)
            return adj

    class COVIDPredictor(nn.Module):
        def __init__(self, num_features, window_size, hidden_dim, output_dim=1):
            super(COVIDPredictor, self).__init__()
            self.window_size = window_size
            self.num_features = num_features
            self.hidden_dim = hidden_dim
            
            # Temporal encoding
            self.temporal_encoder = nn.GRU(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
            
            # Graph Learner
            self.graph_learner = GraphLearner(hidden_dim, hidden_dim)
            
            # GNN layers
            self.gnn1 = GCNConv(hidden_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
            
            # Output layer
            self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, data):
            x = data.x  # [num_nodes, window_size, num_features]
            num_nodes = x.size(0)
            # Temporal encoding
            _, h = self.temporal_encoder(x)
            h = h.squeeze(0)  # [num_nodes, hidden_dim]
            
            # Learn adjacency matrix
            adj = self.graph_learner(h)  # [num_nodes, num_nodes]
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_weight = adj[edge_index[0], edge_index[1]]
            
            # GNN layers
            h = F.relu(self.gnn1(h, edge_index, edge_weight))
            h = F.relu(self.gnn2(h, edge_index, edge_weight))
            
            # Output layer
            out = self.fc_out(h).squeeze(-1)  # [num_nodes]
            return out

    # Initialize model, optimizer, and loss function
    num_features = len(FEATURE_COLUMNS)
    hidden_dim = 64
    model = COVIDPredictor(num_features=num_features, window_size=window_size, hidden_dim=hidden_dim)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # =======================
    # Step 7: Training Loop
    # =======================

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to('cpu')
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Window Size {window_size} | Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to('cpu')
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # =======================
    # Step 8: Testing
    # =======================

    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to('cpu')
            out = model(data)
            loss = criterion(out, data.y)
            test_loss += loss.item()
            all_preds.append(out.numpy())
            all_targets.append(data.y.numpy())
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Denormalize predictions and targets
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    preds_denorm = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_denorm = scaler_target.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Compute evaluation metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(targets_denorm, preds_denorm)
    rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
    print(f'Window Size {window_size} | Test MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    


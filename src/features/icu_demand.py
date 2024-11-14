# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import pickle

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Scikit-learn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set_style('whitegrid')

# Evaluation Metrics
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_ccc(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    numerator = 2 * cor * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

# Data Loading and Preprocessing
df = pd.read_pickle('../../data/processed/processed_covid_data.pickle')
df['date_today'] = pd.to_datetime(df['date_today'])
df.fillna(0, inplace=True)

# Correct ICU Bed Demand Calculation
icu_admission_rate = 0.05
df['icu_bed_demand'] = (df['confirmed'] * icu_admission_rate).round()

# Weekly Aggregation
df_weekly = df.set_index('date_today').groupby('state').resample('W').sum().reset_index()

# Feature Engineering
df_weekly['mortality_rate'] = df_weekly['deaths'] / df_weekly['confirmed']
df_weekly['recovery_rate'] = df_weekly['recovered'] / df_weekly['confirmed']
df_weekly.replace([np.inf, -np.inf], 0, inplace=True)
df_weekly.fillna(0, inplace=True)

lag_features = ['confirmed', 'deaths', 'recovered', 'active', 'icu_bed_demand']
lag_periods = [1, 2, 3]

for lag in lag_periods:
    for feature in lag_features:
        df_weekly[f'{feature}_lag_{lag}'] = df_weekly.groupby('state')[feature].shift(lag)

df_weekly.fillna(0, inplace=True)

features = [
    'confirmed', 'deaths', 'recovered', 'active',
    'mortality_rate', 'recovery_rate',
    'confirmed_lag_1', 'confirmed_lag_2', 'confirmed_lag_3',
    'deaths_lag_1', 'deaths_lag_2', 'deaths_lag_3',
    'recovered_lag_1', 'recovered_lag_2', 'recovered_lag_3',
    'active_lag_1', 'active_lag_2', 'active_lag_3',
    'icu_bed_demand_lag_1', 'icu_bed_demand_lag_2', 'icu_bed_demand_lag_3'
]
target = 'icu_bed_demand'

# Prepare Data for Each State
states = df_weekly['state'].unique()
state_results = {}

for state in states:
    print(f"\nProcessing state: {state}")
    state_df = df_weekly[df_weekly['state'] == state].reset_index(drop=True)

    if len(state_df) < 30:
        print(f"Not enough data for {state}, skipping.")
        continue

    tscv = TimeSeriesSplit(n_splits=5)
    X = state_df[features].values
    y = state_df[target].values.reshape(-1, 1)
    dates = state_df['date_today'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(np.hstack((X, y)))
    X_scaled, y_scaled = X_scaled[:, :-1], X_scaled[:, -1]

    fold = 0
    fold_metrics = []
    for train_index, test_index in tscv.split(X_scaled):
        fold += 1
        print(f"Fold {fold}/5")
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]
        dates_train, dates_test = dates[train_index], dates[test_index]

        seq_length = 3

        def create_sequences(X, y, seq_length):
            xs, ys = [], []
            for i in range(seq_length, len(X)):
                xs.append(X[i-seq_length:i])
                ys.append(y[i])
            return np.array(xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Insufficient data after sequencing for {state} in fold {fold}, skipping.")
            continue

        X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
        y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(device)

        class TimeSeriesDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        batch_size = 16
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Model Definition
        input_size = X_train_seq.shape[2]
        hidden_size = 64
        num_layers = 2
        output_size = 1
        dropout_rate = 0.2

        class LSTMModel(nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    dropout=dropout_rate, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
                c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training Loop with Early Stopping
        patience = 5
        min_val_loss = np.inf
        epochs_no_improve = 0
        best_model_state = None
        num_epochs = 50

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

            if avg_train_loss < min_val_loss:
                min_val_loss = avg_train_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    model.load_state_dict(best_model_state)
                    break

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_seq).cpu().numpy().flatten()
        y_test_seq_cpu = y_test_seq.cpu().numpy().flatten()

        inv_combined = scaler.inverse_transform(
            np.hstack((X_test_seq.cpu().numpy().reshape(-1, input_size), predictions.reshape(-1, 1)))
        )
        inv_predictions = inv_combined[:, -1]

        inv_y_test = scaler.inverse_transform(
            np.hstack((X_test_seq.cpu().numpy().reshape(-1, input_size), y_test_seq_cpu.reshape(-1, 1)))
        )[:, -1]

        rmse = calculate_rmse(inv_y_test, inv_predictions)
        mae = calculate_mae(inv_y_test, inv_predictions)
        r2 = calculate_r2(inv_y_test, inv_predictions)
        ccc = calculate_ccc(inv_y_test, inv_predictions)

        fold_metrics.append({
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CCC': ccc,
            'true_values': inv_y_test,
            'predictions': inv_predictions,
            'dates': dates_test[seq_length:]
        })

        print(f"Fold {fold} Metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, CCC: {ccc:.2f}")

    if fold_metrics:
        avg_rmse = np.mean([m['RMSE'] for m in fold_metrics])
        avg_mae = np.mean([m['MAE'] for m in fold_metrics])
        avg_r2 = np.mean([m['R2'] for m in fold_metrics])
        avg_ccc = np.mean([m['CCC'] for m in fold_metrics])

        print(f"Average Metrics for {state} - RMSE: {avg_rmse:.2f}, MAE: {avg_mae:.2f}, "
              f"R2: {avg_r2:.2f}, CCC: {avg_ccc:.2f}")

        state_results[state] = {
            'metrics': {
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'R2': avg_r2,
                'CCC': avg_ccc
            },
            'fold_metrics': fold_metrics
        }

# Visualization
selected_state = 'New York'
if selected_state in state_results:
    last_fold = state_results[selected_state]['fold_metrics'][-1]
    dates = last_fold['dates']
    true_values = last_fold['true_values']
    predictions = last_fold['predictions']

    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_values, label='Actual ICU Bed Demand', color='black', linewidth=2)
    plt.plot(dates, predictions, label='Predicted ICU Bed Demand', linestyle='--', color='blue')
    plt.title(f'Actual vs Predicted ICU Bed Demand in {selected_state}')
    plt.xlabel('Date')
    plt.ylabel('ICU Bed Demand')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'icu_bed_demand_predictions_{selected_state}.png', dpi=300)
    plt.show()

    # Residual Plot
    residuals = true_values - predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates, residuals, label='Residuals', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Residuals Over Time in {selected_state}')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'residuals_{selected_state}.png', dpi=300)
    plt.show()
else:
    print(f"No results available for {selected_state}.")

# Summary of Results
all_states_metrics = {
    'RMSE': np.mean([state_results[s]['metrics']['RMSE'] for s in state_results]),
    'MAE': np.mean([state_results[s]['metrics']['MAE'] for s in state_results]),
    'R2': np.mean([state_results[s]['metrics']['R2'] for s in state_results]),
    'CCC': np.mean([state_results[s]['metrics']['CCC'] for s in state_results]),
}

print("\nOverall Average Metrics Across All States:")
print(f"RMSE: {all_states_metrics['RMSE']:.2f}")
print(f"MAE: {all_states_metrics['MAE']:.2f}")
print(f"R2: {all_states_metrics['R2']:.2f}")
print(f"CCC: {all_states_metrics['CCC']:.2f}")

# Save results
with open('state_results.pkl', 'wb') as f:
    pickle.dump(state_results, f)

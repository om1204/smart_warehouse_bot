# part4_networks/rnn_model.py
# RNN anomaly detector per §6.2.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AnomalyRNN(nn.Module):
    def __init__(self):
        super(AnomalyRNN, self).__init__()
        # Input features: 3 (e.g., vibration, current, temperature)
        self.lstm = nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, 3)
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_out = lstm_out[:, -1, :]
        out = self.linear(last_out)
        return self.sigmoid(out)


def generate_synthetic_sequences(n: int = 500, seq_len: int = 10, seed: int = 42):
    """
    Generate synthetic data for RNN training.
    Features: [vibration, current, temperature]
    Normal data: low variance, centered around normal operating values.
    Anomaly data: spikes or drifts.
    Returns: X (numpy array), y (numpy array)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = []
    y = []

    for _ in range(n):
        is_anomaly = np.random.rand() > 0.5

        # Base normal values
        vib = np.random.normal(0.5, 0.1, seq_len)
        cur = np.random.normal(1.0, 0.2, seq_len)
        temp = np.random.normal(30.0, 2.0, seq_len)

        if is_anomaly:
            # Inject anomaly: sudden spike in one or more features towards the end
            spike_idx = np.random.randint(seq_len // 2, seq_len)
            anomaly_type = np.random.choice([0, 1, 2])
            if anomaly_type == 0:
                vib[spike_idx:] += np.random.normal(2.0,
                                                    0.5, seq_len - spike_idx)
            elif anomaly_type == 1:
                cur[spike_idx:] += np.random.normal(3.0,
                                                    0.5, seq_len - spike_idx)
            else:
                temp[spike_idx:] += np.random.normal(
                    15.0, 5.0, seq_len - spike_idx)
            y.append(1.0)
        else:
            y.append(0.0)

        # Stack features
        seq = np.column_stack([vib, cur, temp])
        X.append(seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def train_rnn() -> AnomalyRNN:
    """
    Train the RNN model.
    """
    X, y = generate_synthetic_sequences()

    # Simple split
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = AnomalyRNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model


def rnn_predict_anomaly(window: list[dict], model: AnomalyRNN = None) -> bool:
    """
    Predict if a given window of telemetry data constitutes an anomaly.
    window: List of dicts, each with 'vibration', 'current', 'temperature'.
    """
    if model is None:
        model = train_rnn()

    model.eval()

    # Extract features
    seq = []
    for d in window:
        seq.append([
            float(d.get('vibration', 0.5)),
            float(d.get('current', 1.0)),
            float(d.get('temperature', 30.0))
        ])

    x = torch.tensor([seq], dtype=torch.float32)
    with torch.no_grad():
        prob = model(x).item()

    return prob > 0.5

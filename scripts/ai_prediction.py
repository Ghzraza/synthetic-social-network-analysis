# ai_prediction.py
# Author: Ghazal Raza
# Purpose: Functions to train and predict temporal network evolution using RNN

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx

class NetworkDataset(Dataset):
    """
    PyTorch dataset for temporal network adjacency matrices.
    """
    def __init__(self, evolution):
        self.data = []
        for t in range(len(evolution)-1):
            X = nx.to_numpy_array(evolution[t])
            y = nx.to_numpy_array(evolution[t+1])
            self.data.append((X,y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X,y = self.data[idx]
        return torch.tensor(X,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

class SimpleRNN(nn.Module):
    """
    Simple RNN model to predict next network state.
    """
    def __init__(self, num_nodes, hidden_size=64):
        super(SimpleRNN,self).__init__()
        self.rnn = nn.RNN(input_size=num_nodes, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        out,_ = self.rnn(x)
        out = self.fc(out[:,-1,:])
        return out

def train_rnn(evolution, num_nodes=20, epochs=20, lr=0.01):
    """
    Train RNN on network evolution data.

    Args:
        evolution (list): List of networkx.Graph over time.
        num_nodes (int): Number of nodes.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        model (nn.Module): Trained RNN model.
    """
    dataset = NetworkDataset(evolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SimpleRNN(num_nodes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            X_rnn = X.unsqueeze(1)  # batch, seq_len=1, features
            output = model(X_rnn)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}, Loss={total_loss/len(loader):.4f}")
    return model

def predict_next_network(model, last_network):
    """
    Predict next network adjacency matrix using trained RNN.

    Args:
        model (nn.Module): Trained RNN model.
        last_network (networkx.Graph): Last network in evolution.

    Returns:
        predicted_adj (ndarray): Predicted adjacency matrix.
    """
    X_last = torch.tensor(nx.to_numpy_array(last_network), dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    predicted_adj = model(X_last).detach().numpy()
    return predicted_adj

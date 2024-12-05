# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import GPy
import numpy as np

# Define the device at the top of the file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_init_std, log_std_init_mean, log_std_init_std):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(out_features, in_features) * weight_init_std)
        self.log_std = nn.Parameter(torch.randn(out_features, in_features) * log_std_init_std + log_std_init_mean)

    def forward(self, x):
        weights = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        return F.linear(x, weights)

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, n_features, hidden_layers, weight_init_std, log_std_init_mean, log_std_init_std, log_std_clamp):
        super().__init__()
        layers = []
        prev_size = n_features
        for size in hidden_layers:
            layers.extend([BayesianLinear(prev_size, size, weight_init_std, log_std_init_mean, log_std_init_std), nn.ReLU()])
            prev_size = size
        layers.append(BayesianLinear(prev_size, 2, weight_init_std, log_std_init_mean, log_std_init_std))
        self.layers = nn.Sequential(*layers)
        self.log_std_clamp = log_std_clamp

    def forward(self, x):
        output = self.layers(x)
        return torch.distributions.Normal(output[:, 0], torch.exp(output[:, 1].clamp(*self.log_std_clamp)))

def NLL_loss_bnn(targets, distribution):
    return -distribution.log_prob(targets).mean()

def train_bnn(model, train_loader, epochs, learning_rate, grad_clip_norm):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            distribution = model(data)
            loss = NLL_loss_bnn(target, distribution)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {total_loss / len(train_loader):.4f}')

def predict_bnn(model, input_data, n_samples=100):
    model.eval()
    with torch.no_grad():
        samples = torch.stack([model(input_data).sample() for _ in range(n_samples)])
    return samples.mean(0), samples.std(0)

# Ensemble Model
class NeuralNetwork(nn.Module):
    def __init__(self, n_input, *neurons):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_neurons = n_input
        for n in neurons:
            layers.extend([nn.Linear(prev_neurons, n), nn.ReLU()])
            prev_neurons = n
        layers.append(nn.Linear(prev_neurons, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train_ensemble(model, optimizer, train_loader, train_loader_shuffled, epochs):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader_shuffled:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Make sure labels are 2D
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}')

def predict_ensemble(models, data_loader):
    for model in models:
        model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            batch_predictions = [model(inputs).cpu().numpy() for model in models]
            predictions.append(np.array(batch_predictions))
            actuals.append(labels.cpu().numpy())  # Move labels to CPU before converting to numpy

    predictions = np.concatenate(predictions, axis=1)
    means = np.mean(predictions, axis=0).ravel()
    stds = np.std(predictions, axis=0).ravel()
    actuals = np.concatenate(actuals)

    return means, stds, actuals

# Gaussian Process Regression
def train_gpr(xtrain, ytrain, kernel_variance, kernel_lengthscale, white_kernel_variance, max_iterations):
    kernel = GPy.kern.Matern32(input_dim=xtrain.shape[1], variance=kernel_variance, lengthscale=kernel_lengthscale)
    kernel += GPy.kern.White(xtrain.shape[1], variance=white_kernel_variance)
    model = GPy.models.GPRegression(xtrain, ytrain, kernel)
    model.optimize(max_iters=max_iterations)
    return model

def predict_gpr(model, xtest):
    mean, variance = model.predict(xtest)
    std = np.sqrt(variance)
    return mean, std

# Monte Carlo Dropout
class DropoutModel(nn.Module):
    def __init__(self, n_bits, n_1, n_2, n_3, dropout_rate):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(n_bits, n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.fc3 = nn.Linear(n_2, n_3)
        self.fc4 = nn.Linear(n_3, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, apply_dropout=True):
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if apply_dropout else x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def train_mcd(model, train_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.2f}')

def predict_mcd(model, data_loader, T):
    model.eval()
    all_predictions = []
    
    for _ in range(T):
        predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs, apply_dropout=True)
                predictions.append(outputs.cpu().numpy())
        all_predictions.append(np.concatenate(predictions))
    
    all_predictions = np.array(all_predictions)
    mean_pred = np.mean(all_predictions, axis=0).squeeze()
    std_pred = np.std(all_predictions, axis=0).squeeze()
    
    return mean_pred, std_pred

# Mean-Variance Estimation
class MVE_Model(nn.Module):
    def __init__(self, input_shape, hidden_layers):
        super(MVE_Model, self).__init__()
        layers = []
        prev_size = input_shape
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_layers[-1], 2)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        mean = x[:, 0]
        std_dev = F.softplus(x[:, 1]) + 1e-6
        return mean, std_dev

def NLL_loss(targets, mean, std):
    variance = std**2
    nll_first_term = torch.log(2 * torch.pi * variance) / 2
    nll_second_term = ((targets - mean)**2) / (2 * variance)
    return torch.mean(nll_first_term + nll_second_term)

def train_mve(model, train_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            mean, std = model(data)
            loss = NLL_loss(target, mean, std)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')

def predict_mve(model, loader):
    model.eval()
    means, std_devs = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            mean, std_dev = model(data)
            means.append(mean.cpu().numpy())
            std_devs.append(std_dev.cpu().numpy())
    return np.concatenate(means), np.concatenate(std_devs)
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import GPy
import numpy as np

# Define the device at the top of the file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Bayesian Neural Network Model ----------------------
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

# ---------------------- Gaussian Process Regression Model ----------------------
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

# ---------------------- Monte Carlo Dropout Model ----------------------
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

# ---------------------- Mean-Variance Estimation Model ----------------------
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

# ---------------------- BNN-MCMC Model ----------------------
class BayesianNN(nn.Module):
    def __init__(self, n_features, HYPERPARAMETERS):
        super().__init__()
        h1, h2, h3 = HYPERPARAMETERS['hidden_layers']
        self.fc1 = nn.Linear(n_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_mean = nn.Linear(h3, 1)
        self.fc_log_std = nn.Linear(h3, 1)
        self.log_std_clamp = HYPERPARAMETERS['log_std_clamp']
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, HYPERPARAMETERS['weight_init_std'])
                nn.init.zeros_(m.bias)
        
        # Initialize log_std layer with different parameters
        nn.init.normal_(self.fc_log_std.weight, 0, HYPERPARAMETERS['log_std_init_std'])
        nn.init.constant_(self.fc_log_std.bias, HYPERPARAMETERS['log_std_init_mean'])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x).squeeze(-1)
        log_std = self.fc_log_std(x).squeeze(-1)
        log_std = torch.clamp(log_std, *self.log_std_clamp)
        return mean, log_std
    
    def get_params(self):
        # Get all parameters as a single flattened vector
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)
    
    def set_params(self, flat_params):
        # Set all model parameters from a single flattened vector
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.size()))
            offset += numel
        return self
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def nll_loss(mean, log_std, target):
    # Negative log likelihood loss
    variance = torch.exp(2 * log_std)
    return torch.mean(0.5 * torch.log(2 * torch.pi * variance) + 0.5 * ((target - mean) ** 2) / variance)

class HamiltonianMonteCarlo:
    def __init__(self, model, X, y, step_size=0.001, num_steps=10, prior_std=1.0):
        self.model = model
        # Explicitly get the device and print it for debugging
        model_device = next(model.parameters()).device
        print(f"Model device in HMC init: {model_device}")
        
        # Explicitly set self.device
        self.device = model_device
        print(f"Setting HMC device to: {self.device}")
        
        # Move inputs to the correct device
        self.X = X.to(self.device)
        self.y = y.to(self.device)
        
        # Store other parameters
        self.step_size = step_size
        self.num_steps = num_steps
        self.prior_std = prior_std
        self.mass_matrix_diag = None
        
        print(f"HamiltonianMonteCarlo initialized with device: {self.device}")
    
    def initialize_mass_matrix(self, params):
        params = params.to(self.device)
        self.mass_matrix_diag = torch.ones_like(params, device=self.device)
        small_const = 1e-6
        self.mass_matrix_diag = 1.0 / (params.abs() + small_const).clamp(min=0.1, max=100.0)
        self.mass_matrix_diag = self.mass_matrix_diag.to(self.device)
    
    def potential_energy(self, params):
        # Make sure params are on the correct device
        params = params.to(self.device)
        self.model.set_params(params)
        mean, log_std = self.model(self.X)
        loss = nll_loss(mean, log_std, self.y)
        prior = 0.5 * torch.sum(params**2) / (self.prior_std**2)
        return loss + prior
    
    def kinetic_energy(self, momentum):
        # Make sure momentum is on the correct device
        momentum = momentum.to(self.device)
        mass_matrix = self.mass_matrix_diag.to(self.device)
        return 0.5 * torch.sum(momentum**2 / mass_matrix)
    
    def compute_gradients(self, params):
        # Make sure params are on the correct device
        params = params.to(self.device)
        params_copy = params.clone().detach().requires_grad_(True)
        U = self.potential_energy(params_copy)
        U.backward()
        return params_copy.grad.clone().detach().to(self.device)
    
    def leapfrog_step(self, position, momentum, step_size):
        # Make sure inputs are on the correct device
        position = position.to(self.device)
        momentum = momentum.to(self.device)
        
        # Ensure mass_matrix_diag is on the correct device
        mass_matrix = self.mass_matrix_diag.to(self.device)
        
        grad_U = self.compute_gradients(position)
        momentum = momentum - 0.5 * step_size * grad_U
        position = position + step_size * momentum / mass_matrix
        grad_U = self.compute_gradients(position)
        momentum = momentum - 0.5 * step_size * grad_U
        return position.to(self.device), momentum.to(self.device)
    
    def sample(self, current_position):
        # Make sure current_position is on the correct device
        current_position = current_position.to(self.device)
        
        if self.mass_matrix_diag is None:
            self.initialize_mass_matrix(current_position)
            
        # Ensure mass_matrix_diag is on the correct device
        self.mass_matrix_diag = self.mass_matrix_diag.to(self.device)
        
        position = current_position.clone().detach().to(self.device)
        momentum = torch.randn_like(position, device=self.device) * torch.sqrt(self.mass_matrix_diag.to(self.device))
        
        current_U = self.potential_energy(position)
        current_K = self.kinetic_energy(momentum)
        current_H = current_U + current_K
        
        new_position = position.clone().to(self.device)
        new_momentum = momentum.clone().to(self.device)
        
        try:
            for i in range(self.num_steps):
                new_position, new_momentum = self.leapfrog_step(new_position, new_momentum, self.step_size)
                
            new_momentum = -new_momentum
            
            new_U = self.potential_energy(new_position)
            new_K = self.kinetic_energy(new_momentum)
            new_H = new_U + new_K
            
            if torch.isnan(new_H) or torch.isinf(new_H):
                print("Warning: Numerical instability detected in HMC.")
                return position.to(self.device), False
            
            acceptance_ratio = torch.exp(current_H - new_H)
            accept_probability = torch.min(torch.tensor(1.0, device=self.device), acceptance_ratio)
            
            if torch.rand(1, device=self.device) < accept_probability:
                return new_position.to(self.device), True
            else:
                return position.to(self.device), False
        except Exception as e:
            print(f"HMC sampling error: {e}")
            return position.to(self.device), False

# ---------------------- Prediction Function ----------------------
def predict_with_uncertainty(model, x_tensor, params_samples=None, n_samples=475):
    model.eval()  # Set model to evaluation mode
    device = x_tensor.device
    
    if params_samples is not None and len(params_samples) >= 10:
        # Use HMC samples for prediction
        print(f"Using {len(params_samples)} HMC samples for prediction")
        
        # First ensure all params_samples are on the same device
        for i in range(len(params_samples)):
            if params_samples[i].device != device:
                params_samples[i] = params_samples[i].to(device)
        
        means, stds = [], []
        for i, params in enumerate(params_samples):
            # Double-check device
            if params.device != device:
                params = params.to(device)
            
            model.set_params(params)
            with torch.no_grad():
                mean, log_std = model(x_tensor)
                means.append(mean)
                stds.append(torch.exp(log_std))
        
        # Stack tensors - ensure on correct device
        means = torch.stack(means, dim=0)
        stds = torch.stack(stds, dim=0)
        
        if means.device != device:
            means = means.to(device)
        if stds.device != device:
            stds = stds.to(device)
        
        # Calculate final predictions and uncertainties
        mean_pred = means.mean(dim=0)
        std_pred = torch.sqrt((means.var(dim=0) + stds.mean(dim=0)**2))
        
    else:
        # Fall back to MC dropout if not enough HMC samples
        print("Using MC dropout for predictions")
        model.train()  # Enable dropout
        means, stds = [], []
        
        for _ in range(n_samples):
            with torch.no_grad():
                mean, log_std = model(x_tensor)
                means.append(mean)
                stds.append(torch.exp(log_std))
        
        # Stack tensors - ensure on correct device
        means = torch.stack(means, dim=0)
        stds = torch.stack(stds, dim=0)
        
        if means.device != device:
            means = means.to(device)
        if stds.device != device:
            stds = stds.to(device)
        
        # Calculate final predictions and uncertainties
        mean_pred = means.mean(dim=0)
        std_pred = torch.sqrt((means.var(dim=0) + stds.mean(dim=0)**2))
    
    # Final device check before returning
    if mean_pred.device != device:
        mean_pred = mean_pred.to(device)
    if std_pred.device != device:
        std_pred = std_pred.to(device)
        
    return mean_pred, std_pred

# ---------------------- Quantile Regression Model ----------------------
# Define the Quantile Regression model
class QR_Model(nn.Module):
    def __init__(self, input_shape, quantiles=[0.1, 0.5, 0.9], HIDDEN_LAYERS=[1024, 256, 128]):
        super(QR_Model, self).__init__()
        self.quantiles = quantiles
        self.HIDDEN_LAYERS = HIDDEN_LAYERS
        
        layers = []
        prev_size = input_shape
        for size in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(HIDDEN_LAYERS[-1], len(quantiles))
        
    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x

# Quantile loss function
def quantile_loss(preds, target, quantiles):
    # preds shape: (batch_size, num_quantiles)
    # target shape: (batch_size)
    assert preds.size(1) == len(quantiles), "Expected quantile predictions shape to match number of quantiles"
    
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max(q * errors, (q - 1) * errors))
    
    # Mean over all quantiles and batch
    return torch.mean(torch.stack(losses))

# Function to extract predictions for a dataset
def predictions_QR(model, data_loader):
    # Evaluate the model
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    # Convert lists to numpy arrays
    predictions = np.vstack(predictions)
    actuals = np.concatenate(actuals)
    
    lower = predictions[:, 0]
    mean = predictions[:, 1]
    upper = predictions[:, 2]
    
    std = np.abs((upper - lower) / 2)
    
    return mean, std
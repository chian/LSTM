import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn

# === CONFIGURATION: Set all key hyperparameters here ===
latent_dim = 8         # Bottleneck size for BetaVAE
beta = 4.0             # Beta for BetaVAE
input_dim = 15         # Number of features per timepoint (update if needed)
hidden_dim = 64        # LSTM hidden size
num_layers = 1         # LSTM layers
dropout = 0.1          # LSTM dropout
batch_size = 64        # Training batch size
num_epochs = 100       # Number of epochs for VAE/LSTM
anneal_epochs = 20     # KL annealing epochs
lr = 1e-5              # Learning rate
seq_len = 14           # LSTM input sequence length
# =======================================================

# 1. Utility functions to save/load BetaVAE weights

def save_beta_vae(model, path):
    """Save BetaVAE weights to a file."""
    torch.save(model.state_dict(), path)

def load_beta_vae(model_class, path, *args, **kwargs):
    """Load BetaVAE weights from a file. Returns the loaded model."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

# 2. Extract BetaVAE embeddings (mu) from a dataset

def extract_embeddings(model, data_tensor, device):
    """
    Args:
        model: trained BetaVAE
        data_tensor: torch.Tensor, shape (n_samples, input_dim)
        device: torch.device
    Returns:
        embeddings: torch.Tensor, shape (n_samples, latent_dim)
    """
    model.eval()
    data_tensor = data_tensor.to(device)
    with torch.no_grad():
        mu, _ = model.encode(data_tensor)
    return mu.cpu()

# 3. Build (embedding sequence, next embedding) pairs for LSTM training

def build_lstm_embedding_dataset(embeddings, seq_len=14):
    """
    Args:
        embeddings: torch.Tensor, shape (n_timepoints, latent_dim)
        seq_len: int, length of input sequence
    Returns:
        X: torch.Tensor, shape (n_samples, seq_len, latent_dim)
        y: torch.Tensor, shape (n_samples, latent_dim)
    """
    n_timepoints, latent_dim = embeddings.shape
    X = []
    y = []
    for i in range(n_timepoints - seq_len):
        X.append(embeddings[i:i+seq_len])
        y.append(embeddings[i+seq_len])
    X = torch.stack(X) if X else torch.empty(0, seq_len, latent_dim)
    y = torch.stack(y) if y else torch.empty(0, latent_dim)
    return X, y

# 4. (Stub) for LSTM training using these pairs

class SimpleLSTM(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Use last output
        return out

def train_lstm_on_embeddings(X, y, latent_dim, hidden_dim=64, num_layers=1, dropout=0.1, lr=1e-3, batch_size=32, num_epochs=100, device='cpu'):
    """
    Train an LSTM to predict next-step embeddings.
    Args:
        X: torch.Tensor, shape (n_samples, seq_len, latent_dim)
        y: torch.Tensor, shape (n_samples, latent_dim)
        latent_dim: int
        hidden_dim, num_layers, dropout, lr, batch_size, num_epochs: LSTM/optimizer params
        device: torch.device or str
    Returns:
        model: trained LSTM
        loss_history: list of float
    """
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleLSTM(latent_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    return model, loss_history


def prepare_lstm_data_from_profiles(model, profiles_tensor, device, seq_len=14):
    """
    Given a batch of microbial profiles (torch.Tensor), extract embeddings using the trained BetaVAE encoder,
    then build LSTM training sequences and targets.

    Args:
        model: trained BetaVAE
        profiles_tensor: torch.Tensor, shape (n_samples, input_dim)
        device: torch.device
        seq_len: int, length of input sequence for LSTM
    Returns:
        X: torch.Tensor, shape (n_samples - seq_len, seq_len, latent_dim)
        y: torch.Tensor, shape (n_samples - seq_len, latent_dim)
    """
    embeddings = extract_embeddings(model, profiles_tensor, device)  # (n_samples, latent_dim)
    X, y = build_lstm_embedding_dataset(embeddings, seq_len=seq_len)
    return X, y 


def prepare_lstm_data_from_trajectories(model, trajectories, device, seq_len=14):
    """
    Given a list of trajectories (each a torch.Tensor of shape [n_timepoints, input_dim]),
    extract embeddings using the trained BetaVAE encoder, build LSTM training sequences and targets for each,
    and concatenate all (X, y) pairs across trajectories.

    Args:
        model: trained BetaVAE
        trajectories: list of torch.Tensor, each shape (n_timepoints, input_dim)
        device: torch.device
        seq_len: int, length of input sequence for LSTM
    Returns:
        X_all: torch.Tensor, shape (total_samples, seq_len, latent_dim)
        y_all: torch.Tensor, shape (total_samples, latent_dim)
    """
    X_list = []
    y_list = []
    for traj in trajectories:
        embeddings = extract_embeddings(model, traj, device)  # (n_timepoints, latent_dim)
        X, y = build_lstm_embedding_dataset(embeddings, seq_len=seq_len)
        if X.shape[0] > 0:
            X_list.append(X)
            y_list.append(y)
    if X_list:
        X_all = torch.cat(X_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
    else:
        latent_dim = extract_embeddings(model, trajectories[0], device).shape[1]
        X_all = torch.empty(0, seq_len, latent_dim)
        y_all = torch.empty(0, latent_dim)
    return X_all, y_all 

import matplotlib.pyplot as plt
import numpy as np

def decode_embeddings(decoder, embeddings, device):
    """
    Decode a batch of embeddings using the BetaVAE decoder.
    Args:
        decoder: BetaVAE.decoder (nn.Module)
        embeddings: torch.Tensor, shape (n_samples, latent_dim)
        device: torch.device
    Returns:
        decoded: torch.Tensor, shape (n_samples, input_dim)
    """
    decoder.eval()
    with torch.no_grad():
        decoded = decoder(embeddings.to(device)).cpu()
    return decoded


def decode_and_plot_lstm_predictions(beta_vae, lstm_model, X, true_profiles, device, feature_names=None, num_features_to_plot=4):
    """
    Given a batch of input embedding sequences (X), use the LSTM to predict next embeddings,
    decode them with the BetaVAE decoder, and plot vs. true profiles.
    Also compute and print MSE.
    Args:
        beta_vae: trained BetaVAE
        lstm_model: trained LSTM
        X: torch.Tensor, shape (n_samples, seq_len, latent_dim)
        true_profiles: torch.Tensor, shape (n_samples, input_dim)
        device: torch.device
        feature_names: list of str, optional
        num_features_to_plot: int, number of features/species to plot
    Returns:
        decoded_preds: torch.Tensor, shape (n_samples, input_dim)
        mse: float
    """
    lstm_model.eval()
    with torch.no_grad():
        pred_embeddings = lstm_model(X.to(device))  # (n_samples, latent_dim)
    decoded_preds = decode_embeddings(beta_vae.decoder, pred_embeddings, device)  # (n_samples, input_dim)
    mse = ((decoded_preds - true_profiles.cpu()) ** 2).mean().item()
    print(f"Validation MSE (decoded): {mse:.6f}")
    # Plotting
    n_plot = min(num_features_to_plot, decoded_preds.shape[1])
    plt.figure(figsize=(10, 6))
    for i in range(n_plot):
        plt.plot(true_profiles[:, i].cpu().numpy(), label=f"True {feature_names[i] if feature_names else i}", lw=2)
        plt.plot(decoded_preds[:, i].cpu().numpy(), label=f"Pred {feature_names[i] if feature_names else i}", ls=':', lw=2)
    plt.xlabel('Sample')
    plt.ylabel('Abundance')
    plt.title('LSTM Decoded Predictions vs. True Profiles')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return decoded_preds, mse 


def run_beta_vae_experiment(
    latent_dim=8,
    beta=4.0,
    input_dim=15,
    hidden_dim=64,
    num_layers=1,
    batch_size=64,
    num_epochs=100,
    anneal_epochs=20,
    lr=1e-5,
    testing_traj_ind=0,
    save_path=None
):
    import pickle
    from beta_vae import BetaVAE, train_beta_vae
    import os
    from beta_vae import total_correlation_metric, latent_traversal_plot
    import data_parsing
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # 1. Load all training and testing sequences
    data_path = os.path.join('data', 'train_test_sequences.pickle')
    with open(data_path, 'rb') as f:
        all_training_sequences, all_testing_sequences = pickle.load(f)

    # 2. Use data_parsing.setup_testing to split into train/val
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainingData, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(
        all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)

    # 3. Prepare training DataLoader (flatten to single timepoints for VAE)
    train_profiles = trainingData.tensors[0].reshape(-1, input_dim).float()
    dataset = TensorDataset(train_profiles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Validation tensor (flatten validation input sequence)
    val_tensor = tensor_true_test_in.reshape(-1, input_dim).float()

    # 5. Instantiate BetaVAE
    vae = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    # 6. Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # 7. Train BetaVAE with validation
    vae, tc_history, best_tc, best_epoch = train_beta_vae(vae, dataloader, optimizer, device, epochs=num_epochs, anneal_epochs=anneal_epochs, tc_interval=10, val_tensor=val_tensor)

    # 8. Do not save trained model in evaluation mode
    return vae, tc_history, best_tc, best_epoch

if __name__ == "__main__":
    from itertools import product
    # Define hyperparameter grid
    latent_dims = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    betas = [1.0, 4.0, 10.0]
    hidden_dims = [16, 32, 64]
    num_layers_list = [1, 2, 3, 4]
    lrs = [1e-4, 1e-5, 1e-6]
    batch_sizes = [32, 64]
    num_epochs = 50
    anneal_epochs = 10
    input_dim = 15
    results = []
    for latent_dim, beta, hidden_dim, num_layers, lr, batch_size in product(latent_dims, betas, hidden_dims, num_layers_list, lrs, batch_sizes):
        print(f"\n=== Running BetaVAE: latent_dim={latent_dim}, beta={beta}, hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr}, batch_size={batch_size} ===")
        vae, tc_history, best_tc, best_epoch = run_beta_vae_experiment(
            latent_dim=latent_dim,
            beta=beta,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            num_epochs=num_epochs,
            anneal_epochs=anneal_epochs,
            lr=lr,
            save_path=None
        )
        print(f"Best TC for this run: {best_tc:.4f} at epoch {best_epoch}")
        results.append({
            'latent_dim': latent_dim,
            'beta': beta,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lr': lr,
            'batch_size': batch_size,
            'tc': best_tc,
            'best_epoch': best_epoch,
            'tc_history': tc_history
        })
    print("\nAll runs complete. Results (sorted by best TC):")
    for r in sorted(results, key=lambda x: x['tc']):
        print(r)
    best_result = min(results, key=lambda x: x['tc'])
    print(f"\nBest overall: TC={best_result['tc']:.4f} at epoch {best_result['best_epoch']} with hyperparameters: {best_result}") 
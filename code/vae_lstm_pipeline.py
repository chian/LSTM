import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

def train_lstm_on_embeddings(*args, **kwargs):
    """Stub for LSTM training on embedding sequences (to be implemented)."""
    pass 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, beta=4.0, hidden_dim=64):
        super(BetaVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # For normalized/relative abundance data
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld, recon_loss, kld

def train_beta_vae(model, dataloader, optimizer, device, epochs=100, anneal_epochs=20):
    """
    Train BetaVAE with monotonic KL annealing.
    Args:
        model: BetaVAE instance
        dataloader: DataLoader
        optimizer: torch optimizer
        device: torch.device
        epochs: int, total number of epochs
        anneal_epochs: int, number of epochs to linearly ramp beta from 0 to model.beta
    """
    model.train()
    base_beta = model.beta  # Save configured beta
    for epoch in range(epochs):
        # Monotonic linear annealing schedule
        if anneal_epochs > 0:
            model.beta = min(base_beta, base_beta * (epoch + 1) / anneal_epochs)
        else:
            model.beta = base_beta
        total_loss, total_recon, total_kld = 0, 0, 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, z = model(x)
            loss, recon_loss, kld = model.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
        print(f"Epoch {epoch+1}: Loss={total_loss:.2f} Recon={total_recon:.2f} KLD={total_kld:.2f} Beta={model.beta:.4f}")
    # Restore model.beta to base_beta after training
    model.beta = base_beta
    return model

# Placeholder for disentanglement metrics (to be implemented as needed)
def compute_mig(latent_codes, factors):
    """Compute Mutual Information Gap (MIG) given latent codes and ground-truth factors."""
    pass

def compute_sap(latent_codes, factors):
    """Compute Separated Attribute Predictability (SAP) given latent codes and ground-truth factors."""
    pass

def compute_dci(latent_codes, factors):
    """Compute DCI Disentanglement metric given latent codes and ground-truth factors."""
    pass 
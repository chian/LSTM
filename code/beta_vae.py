import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, beta=4.0, hidden_dim=64, num_layers=2):
        super(BetaVAE, self).__init__(); self.input_dim = input_dim; self.latent_dim = latent_dim; self.beta = beta; self.hidden_dim = hidden_dim; self.num_layers = num_layers
        # Automated shrinking for encoder
        encoder_layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            next_dim = int(hidden_dim * ((latent_dim / hidden_dim) ** (i / (num_layers - 1)))) if num_layers > 1 else hidden_dim
            encoder_layers.append(nn.Linear(prev_dim, next_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = next_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        # Automated expanding for decoder
        decoder_layers = []
        prev_dim = latent_dim
        for i in range(num_layers):
            next_dim = int(hidden_dim * ((input_dim / hidden_dim) ** (i / (num_layers - 1)))) if num_layers > 1 else hidden_dim
            decoder_layers.append(nn.Linear(prev_dim, next_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = next_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

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

def total_correlation_metric(mu, n_bins=20):
    """
    Estimate total correlation (TC) of latent codes (mu) using histogram binning.
    Args:
        mu: torch.Tensor, shape (n_samples, latent_dim)
        n_bins: int, number of bins for histogram
    Returns:
        tc: float, estimated total correlation
    """
    import numpy as np
    mu_np = mu.detach().cpu().numpy()
    n_samples, latent_dim = mu_np.shape
    tc = 0.0
    # Compute joint histogram
    joint_hist, _ = np.histogramdd(mu_np, bins=n_bins)
    joint_prob = joint_hist / np.sum(joint_hist)
    joint_prob = joint_prob[joint_prob > 0]
    joint_entropy = -np.sum(joint_prob * np.log(joint_prob + 1e-12))
    # Compute marginal entropies
    marginal_entropies = []
    for d in range(latent_dim):
        hist, _ = np.histogram(mu_np[:, d], bins=n_bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        marginal_entropy = -np.sum(prob * np.log(prob + 1e-12))
        marginal_entropies.append(marginal_entropy)
    sum_marginal = np.sum(marginal_entropies)
    tc = sum_marginal - joint_entropy
    return tc

def latent_traversal_plot(beta_vae, input_profile, device, n_steps=7, std_range=3.0, feature_names=None):
    """
    Plot reconstructions as each latent is traversed across +/- std_range*std, holding others fixed.
    Args:
        beta_vae: trained BetaVAE
        input_profile: torch.Tensor, shape (input_dim,)
        device: torch.device
        n_steps: int, number of steps per latent
        std_range: float, range in std devs to traverse
        feature_names: list of str, optional
    """
    beta_vae.eval()
    import matplotlib.pyplot as plt
    import numpy as np
    plt.ion()
    with torch.no_grad():
        mu, logvar = beta_vae.encode(input_profile.unsqueeze(0).to(device))
        mu = mu[0].cpu().numpy()
        std = np.sqrt(np.exp(logvar[0].cpu().numpy()))
        latent_dim = mu.shape[0]
        fig, axes = plt.subplots(latent_dim, n_steps, figsize=(2.5*n_steps, 2*latent_dim))
        for d in range(latent_dim):
            vals = np.linspace(mu[d] - std_range*std[d], mu[d] + std_range*std[d], n_steps)
            for i, v in enumerate(vals):
                z = mu.copy()
                z[d] = v
                z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
                recon = beta_vae.decode(z_tensor).cpu().numpy()[0]
                ax = axes[d, i] if latent_dim > 1 else axes[i]
                ax.plot(recon, lw=2)
                if feature_names is not None:
                    ax.set_xticks(np.arange(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha='right')
                ax.set_title(f"z{d}={v:.2f}")
                if i == 0:
                    ax.set_ylabel(f"Latent {d}")
        plt.tight_layout()
        plt.show(block=False)

def train_beta_vae(model, dataloader, optimizer, device, epochs=100, anneal_epochs=20, tc_interval=10, val_tensor=None):
    """
    Train BetaVAE with monotonic KL annealing and print TC at intervals. Also print validation loss if val_tensor is provided.
    At each interval, if val_tensor is provided, show a latent traversal plot for the first validation sample.
    Args:
        model: BetaVAE instance
        dataloader: DataLoader
        optimizer: torch optimizer
        device: torch.device
        epochs: int, total number of epochs
        anneal_epochs: int, number of epochs to linearly ramp beta from 0 to model.beta
        tc_interval: int, how often (in epochs) to print total correlation
        val_tensor: torch.Tensor or None, shape (n_val_samples, input_dim)
    Returns:
        model: trained BetaVAE
        tc_history: list of (epoch, TC) tuples
        best_tc: float (lowest TC)
        best_epoch: int (epoch with lowest TC)
    """
    model.train()
    base_beta = model.beta  # Save configured beta
    tc_history = []
    best_tc = float('inf')
    best_epoch = -1
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
        # Compute TC and validation loss at intervals
        if (epoch+1) % tc_interval == 0 or epoch == 0 or epoch == epochs-1:
            # Get all mu for the dataset
            model.eval()
            all_mu = []
            with torch.no_grad():
                for batch in dataloader:
                    x = batch[0].to(device)
                    mu, _ = model.encode(x)
                    all_mu.append(mu.cpu())
            all_mu = torch.cat(all_mu, dim=0)
            tc = total_correlation_metric(all_mu)
            tc_history.append((epoch+1, tc))
            if tc < best_tc:
                best_tc = tc
                best_epoch = epoch+1
            val_str = ""
            if val_tensor is not None:
                with torch.no_grad():
                    x_val = val_tensor.to(device)
                    recon_x_val, mu_val, logvar_val, z_val = model(x_val)
                    val_loss, val_recon, val_kld = model.loss_function(recon_x_val, x_val, mu_val, logvar_val)
                val_str = f" | ValLoss={val_loss.item():.2f} ValRecon={val_recon.item():.2f} ValKLD={val_kld.item():.2f}"
            print(f"Epoch {epoch+1}: Loss={total_loss:.2f} Recon={total_recon:.2f} KLD={total_kld:.2f} Beta={model.beta:.4f} TC={tc:.4f}{val_str}")
            # Show latent traversal plot for first validation sample
            if val_tensor is not None:
                sample = val_tensor[0].to(device)
                print(f"Latent traversal plot for validation sample (epoch {epoch+1})")
                latent_traversal_plot(model, sample, device)
            model.train()
        else:
            print(f"Epoch {epoch+1}: Loss={total_loss:.2f} Recon={total_recon:.2f} KLD={total_kld:.2f} Beta={model.beta:.4f}")
    # Restore model.beta to base_beta after training
    model.beta = base_beta
    return model, tc_history, best_tc, best_epoch

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
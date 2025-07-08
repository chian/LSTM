import numpy as np
import torch
from itertools import combinations
import random

# Utility: compositional mean across timepoints (for baseline sequence)
def compositional_mean_baseline(input_seq):
    """
    Compute the compositional mean across timepoints for a sequence.
    Args:
        input_seq: torch.Tensor, shape (timesteps, features), each row sums to 1
    Returns:
        baseline: torch.Tensor, shape (timesteps, features), each row is the compositional mean
    """
    # Compute mean in the simplex (compositional mean)
    # Geometric mean, then renormalize
    log_x = torch.log(input_seq + 1e-12)  # avoid log(0)
    mean_log = log_x.mean(dim=0)
    gmean = torch.exp(mean_log)
    gmean = gmean / gmean.sum()  # renormalize to sum to 1
    baseline = gmean.unsqueeze(0).repeat(input_seq.shape[0], 1)
    return baseline

# Utility: ablate a feature with renormalization for compositional data
def ablate_feature_compositional(x, feature_idx, baseline_value):
    """
    Ablate a single feature in a compositional vector, renormalizing the rest.
    Args:
        x: torch.Tensor, shape (features,), sums to 1
        feature_idx: int, index to ablate
        baseline_value: float, value to set for ablated feature
    Returns:
        x_new: torch.Tensor, shape (features,), sums to 1
    """
    x_new = x.clone()
    other_sum = x.sum() - x[feature_idx]
    if other_sum < 1e-8:
        # All mass is in feature_idx, just set to baseline
        x_new[feature_idx] = baseline_value
        return x_new
    scale = (1 - baseline_value) / other_sum
    for j in range(len(x)):
        if j == feature_idx:
            x_new[j] = baseline_value
        else:
            x_new[j] = x[j] * scale
    return x_new


def shapley_feature_attribution(model, input_seq, baseline, target_idx, device=None, perturb_func=None):
    """
    Compute Shapley value-based feature attribution for a single input sequence, separating self-effect and neighbor effects.
    Args:
        model: PyTorch model (should be in eval mode)
        input_seq: torch.Tensor, shape (timesteps, features)
        baseline: torch.Tensor, same shape as input_seq, used for masking
        target_idx: int, index of the target species (self)
        device: torch.device, optional
        perturb_func: function(input_seq, feature_idx) -> perturbed_seq, optional
    Returns:
        self_effect: float (average Shapley value for target's own history)
        neighbor_effects: dict {neighbor_idx: shapley_value}
        max_neighbor_effect: float (max Shapley value among neighbors)
    """
    model.eval()
    if device is not None:
        input_seq = input_seq.to(device)
        baseline = baseline.to(device)
    num_features = input_seq.shape[1]
    feature_indices = list(range(num_features))
    shapley_values = np.zeros(num_features)
    n_layers = model.num_layers
    batch_size = 1  # single sequence
    for i in feature_indices:
        contrib = 0.0
        count = 0
        for k in range(num_features):
            for S in combinations([j for j in feature_indices if j != i], k):
                mask = torch.ones(num_features, dtype=torch.bool)
                mask[list(S)] = False
                x_S = input_seq.clone()
                x_S[:, mask] = baseline[:, mask]
                x_Si = x_S.clone()
                x_Si[:, i] = input_seq[:, i]
                if perturb_func is not None:
                    x_S = perturb_func(input_seq, S)
                    x_Si = perturb_func(input_seq, tuple(list(S)+[i]))
                with torch.no_grad():
                    # Initialize hidden state before each call
                    model.hidden = (
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                    )
                    out_S = model(x_S.unsqueeze(0))
                    model.hidden = (
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                    )
                    out_Si = model(x_Si.unsqueeze(0))
                v_S = out_S[0, target_idx].item()
                v_Si = out_Si[0, target_idx].item()
                contrib += (v_Si - v_S)
                count += 1
        shapley_values[i] = contrib / count if count > 0 else 0.0
    # Separate self and neighbor effects
    self_effect = shapley_values[target_idx]
    neighbor_effects = {i: shapley_values[i] for i in feature_indices if i != target_idx}
    max_neighbor_effect = max(neighbor_effects.values()) if neighbor_effects else 0.0
    return self_effect, neighbor_effects, max_neighbor_effect


def shapley_species_timepoint_matrix(
    model, input_seq, baseline, device=None, n_samples=1000, agg_func=np.mean, seed=None
):
    """
    Approximate Shapley values for all (species, timepoint) pairs using random sampling.
    Args:
        model: PyTorch model (should be in eval mode)
        input_seq: torch.Tensor, shape (timesteps, features)
        baseline: torch.Tensor, same shape as input_seq, used for masking
        device: torch.device, optional
        n_samples: int, number of random subset samples per variable
        agg_func: function, how to aggregate output (e.g., np.mean, np.sum)
        seed: int, random seed for reproducibility
    Returns:
        shapley_matrix: np.ndarray, shape (features, timesteps)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    model.eval()
    if device is not None:
        input_seq = input_seq.to(device)
        baseline = baseline.to(device)
    timesteps, features = input_seq.shape
    variables = [(f, t) for f in range(features) for t in range(timesteps)]
    shapley_matrix = np.zeros((features, timesteps))
    n_layers = model.num_layers
    batch_size = 1
    for var_idx, (f, t) in enumerate(variables):
        contribs = []
        for _ in range(n_samples):
            # Randomly sample a subset S of variables not including (f, t)
            other_vars = [(fi, ti) for fi in range(features) for ti in range(timesteps) if (fi, ti) != (f, t)]
            k = np.random.randint(0, len(other_vars)+1)
            S = set(random.sample(other_vars, k))
            # Build mask for S (True = keep, False = mask)
            mask = torch.zeros((timesteps, features), dtype=torch.bool)
            for (fi, ti) in S:
                mask[ti, fi] = True
            # S without (f, t): mask (f, t) as well
            mask_S = mask.clone()
            # S with (f, t): add (f, t)
            mask_Si = mask.clone()
            mask_Si[t, f] = True
            # Build input tensors
            x_S = baseline.clone()
            x_S[mask_S] = input_seq[mask_S]
            x_Si = baseline.clone()
            x_Si[mask_Si] = input_seq[mask_Si]
            with torch.no_grad():
                model.hidden = (
                    torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                    torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                )
                out_S = model(x_S.unsqueeze(0))
                model.hidden = (
                    torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                    torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                )
                out_Si = model(x_Si.unsqueeze(0))
            # Aggregate output (e.g., mean over output species)
            v_S = agg_func(out_S[0].cpu().numpy())
            v_Si = agg_func(out_Si[0].cpu().numpy())
            contribs.append(v_Si - v_S)
        shapley_matrix[f, t] = np.mean(contribs)
    return shapley_matrix


def shapley_species_timepoint_to_output_tensor(
    model, input_seq, baseline, device=None, n_samples=1000, seed=None
):
    """
    Approximate Shapley values for each (input_species, input_timepoint, output_species) for the next timepoint prediction.
    Args:
        model: PyTorch model (should be in eval mode)
        input_seq: torch.Tensor, shape (timesteps, features)
        baseline: torch.Tensor, same shape as input_seq, used for masking
        device: torch.device, optional
        n_samples: int, number of random subset samples per variable
        seed: int, random seed for reproducibility
    Returns:
        shapley_tensor: np.ndarray, shape (features, timesteps, features)
            (input_species, input_timepoint, output_species)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    model.eval()
    if device is not None:
        input_seq = input_seq.to(device)
        baseline = baseline.to(device)
    timesteps, features = input_seq.shape
    variables = [(f, t) for f in range(features) for t in range(timesteps)]
    shapley_tensor = np.zeros((features, timesteps, features))
    n_layers = model.num_layers
    batch_size = 1
    for f in range(features):
        for t in range(timesteps):
            contribs = [[] for _ in range(features)]  # one list per output species
            for _ in range(n_samples):
                # Randomly sample a subset S of variables not including (f, t)
                other_vars = [(fi, ti) for fi in range(features) for ti in range(timesteps) if (fi, ti) != (f, t)]
                k = np.random.randint(0, len(other_vars)+1)
                S = set(random.sample(other_vars, k))
                # Build mask for S (True = keep, False = mask)
                mask = torch.zeros((timesteps, features), dtype=torch.bool)
                for (fi, ti) in S:
                    mask[ti, fi] = True
                # S without (f, t): mask (f, t) as well
                mask_S = mask.clone()
                # S with (f, t): add (f, t)
                mask_Si = mask.clone()
                mask_Si[t, f] = True
                # Build input tensors
                x_S = baseline.clone()
                x_S[mask_S] = input_seq[mask_S]
                x_Si = baseline.clone()
                x_Si[mask_Si] = input_seq[mask_Si]
                with torch.no_grad():
                    model.hidden = (
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                    )
                    out_S = model(x_S.unsqueeze(0))
                    model.hidden = (
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device),
                        torch.zeros(n_layers, batch_size, model.hidden_layer_size).to(input_seq.device)
                    )
                    out_Si = model(x_Si.unsqueeze(0))
                # out_S and out_Si are shape (1, features) for next timepoint prediction
                v_S = out_S[0].cpu().numpy()
                v_Si = out_Si[0].cpu().numpy()
                for out_f in range(features):
                    contribs[out_f].append(v_Si[out_f] - v_S[out_f])
            for out_f in range(features):
                shapley_tensor[f, t, out_f] = np.mean(contribs[out_f])
    return shapley_tensor 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from torch.utils.data import Dataset
import math
import collections
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset, ConcatDataset
import matplotlib.cm as cm
import pickle
import data_parsing
import lstm_funcs
import msda
import shapley_attribution
import seaborn as sns
from shapley_attribution import compositional_mean_baseline, ablate_feature_compositional
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
from itertools import combinations
import random

# This is just the main script if you want to run the model one-off or something
# 

# species names and device setup
blastT_labels = ['L. crispatus',
 'L. iners',
 'L. gasseri',
 'L. jensenii',
 'L. other',
 'Gardnerella spp',
 'Streptococcus spp', 
 'Atopobium spp',
 'Prevotella spp',
 'Bergeyella spp',
 'Corynebacterium spp',
 'Finegoldia spp',
 'Sneathia spp',
 'Dialister spp',
 'Other']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

#Loading training data
with open('data/train_test_sequences.pickle','rb') as f:
    all_training_sequences, all_testing_sequences = pickle.load(f)

# Pick testing traj and remove training ones w overlap to avoid overfit
# Also do scaling and tensor stuff (see data_parsing)
testing_traj_ind = 130
batch_size = 32
trainingData2, test_in, tensor_true_test_in, tensor_true_test_data = data_parsing.setup_testing(all_training_sequences, all_testing_sequences, testing_traj_ind, device=device)


ensemble_size = 1 #Number of restarts
num_channels = 15
num_timesteps = 14
train_window = 14

total_tloss = 0
ensemble_shapley_matrices = []
ensemble_self_effects = []
ensemble_max_neighbor_effects = []
for z in range(ensemble_size):
    
    hparamdict = {'hsize': 20,
   'layers': 1,
   'batch_size': 32,
   'num_epochs': 1000,
   'lr': 0.01,
   'dropout': 0.1}

    hsize = hparamdict['hsize']
    layers = hparamdict['layers']
    LR = hparamdict['lr']
    batch_size = hparamdict['batch_size']
    num_epochs = hparamdict['num_epochs']
    dropout = hparamdict['dropout']
    
    print('Hyperparameters:', hparamdict)
    
    train_dataloader = DataLoader(trainingData2, batch_size=batch_size, shuffle=True)
    best_prediction, bpred_epoch, best_model, test_loss = lstm_funcs.full_train(num_channels,hsize,layers,dropout,batch_size,LR,
                                         num_epochs,train_dataloader,test_in,
                                         tensor_true_test_data, device=device)
    total_tloss += test_loss
    print(f'Run: {z+1}    Test Loss: {test_loss}    Restart: {z+1}    Best Epoch: {bpred_epoch}')
    if z == ensemble_size:
        avg_tloss = total_tloss/(z+1)
        print('Average Test Loss:', avg_tloss)

    # --- PREDICTION PLOT ---
    base_data_color = ['black', 'red', 'orange', 'green', 'blue', 'gold', 'purple', 'pink', 'grey', 'turquoise', 'cyan', 'crimson', 'indigo', 'olive', 'saddlebrown']
    pred_color = base_data_color
    best_fit_idx = list(range(4))
    full_test_data = torch.cat((torch.clone(tensor_true_test_in),torch.clone(tensor_true_test_data)))
    lstm_funcs.plot_best_fit(full_test_data, best_prediction, pred_color, base_data_color, best_fit_idx, blastT_labels, False, None, full_test_data, run_number=z+1)

    # --- SHAPLEY ATTRIBUTION ---
    # Baseline for timepoint attribution: compositional mean
    baseline = compositional_mean_baseline(tensor_true_test_in.cpu())

    # Compute mean_vec once for the perturb function
    mean_vec = compositional_mean_baseline(tensor_true_test_in.cpu())[0]
    # Perturb function for feature ablation: ablate each feature at all timepoints
    def feature_ablation_perturb(input_seq, ablated_features, mean_vec):
        ablated_seq = input_seq.clone()
        for t in range(input_seq.shape[0]):
            for f in ablated_features:
                ablated_seq[t] = ablate_feature_compositional(ablated_seq[t], f, mean_vec[f])
        return ablated_seq

    def compute_shapley_for_species(target_idx):
        perturbed_inputs_S = []
        perturbed_inputs_Si = []
        feature_indices = list(range(len(blastT_labels)))
        for k in range(len(blastT_labels)):
            for S in combinations([j for j in feature_indices if j != target_idx], k):
                x_S = feature_ablation_perturb(tensor_true_test_in.cpu(), S, mean_vec)
                x_Si = feature_ablation_perturb(tensor_true_test_in.cpu(), tuple(list(S)+[target_idx]), mean_vec)
                perturbed_inputs_S.append(x_S)
                perturbed_inputs_Si.append(x_Si)
        batch_S = torch.stack(perturbed_inputs_S)
        batch_Si = torch.stack(perturbed_inputs_Si)
        with torch.no_grad():
            # Set up hidden state for batch
            n_layers = best_model.num_layers
            batch_size = batch_S.size(0)
            best_model.hidden = (
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_S.device),
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_S.device)
            )
            out_S = best_model(batch_S)
            best_model.hidden = (
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_Si.device),
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_Si.device)
            )
            out_Si = best_model(batch_Si)
        # Calculate Shapley values for this target_idx
        shapley_values = (out_Si[:, target_idx] - out_S[:, target_idx]).cpu().numpy()
        self_effect = np.mean(shapley_values)
        neighbor_effects = {i: np.mean(shapley_values) for i in feature_indices if i != target_idx}
        max_neighbor_effect = max(neighbor_effects.values()) if neighbor_effects else 0.0
        return self_effect, neighbor_effects, max_neighbor_effect

    results = Parallel(n_jobs=4)(
        delayed(compute_shapley_for_species)(target_idx) for target_idx in range(len(blastT_labels))
    )
    # Initialize arrays before unpacking results
    shapley_matrix = np.zeros((len(blastT_labels), len(blastT_labels)))
    self_effects = np.zeros(len(blastT_labels))
    max_neighbor_effects = np.zeros(len(blastT_labels))
    # Unpack results into the original arrays
    for target_idx, (self_effect, neighbor_effects, max_neighbor_effect) in enumerate(results):
        print(f'--- Shapley attribution for {blastT_labels[target_idx]} ---')
        print(f'Self-effect (own history): {self_effect:.4f}')
        print(f'Max neighbor effect: {max_neighbor_effect:.4f}')
        print('Neighbor effects:')
        for idx, val in neighbor_effects.items():
            print(f'  {blastT_labels[idx]}: {val:.4f}')
            shapley_matrix[target_idx, idx] = val
        shapley_matrix[target_idx, target_idx] = self_effect
        self_effects[target_idx] = self_effect
        max_neighbor_effects[target_idx] = max_neighbor_effect
    ensemble_shapley_matrices.append(shapley_matrix)
    ensemble_self_effects.append(self_effects)
    ensemble_max_neighbor_effects.append(max_neighbor_effects)
    # --- HEATMAP PLOT ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(shapley_matrix, xticklabels=blastT_labels, yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=True, fmt=".2f")
    plt.xlabel('Neighbor Species')
    plt.ylabel('Target Species')
    plt.title(f'Shapley Attribution Heatmap (Neighbor Effects) - Run {z+1}')
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)

# --- AVERAGE ACROSS ENSEMBLES ---
if ensemble_shapley_matrices:
    avg_shapley_matrix = np.mean(ensemble_shapley_matrices, axis=0)
    avg_self_effects = np.mean(ensemble_self_effects, axis=0)
    avg_max_neighbor_effects = np.mean(ensemble_max_neighbor_effects, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_shapley_matrix, xticklabels=blastT_labels, yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=True, fmt=".2f")
    plt.xlabel('Neighbor Species')
    plt.ylabel('Target Species')
    plt.title('Average Shapley Attribution Heatmap (Neighbor Effects) - All Runs')
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    print('\n=== Average Self-effect (own history) across all runs ===')
    for idx, val in enumerate(avg_self_effects):
        print(f'{blastT_labels[idx]}: {val:.4f}')
    print('\n=== Average Max Neighbor Effect across all runs ===')
    for idx, val in enumerate(avg_max_neighbor_effects):
        print(f'{blastT_labels[idx]}: {val:.4f}')

    # --- SPECIES-TIMEPOINT SHAPLEY HEATMAP ---
    print('\nComputing approximate Shapley values for all (species, timepoint) pairs...')
    # Use the best model and first test input for demonstration
    # For timepoint ablation, baseline is compositional mean
    def compute_shapley_for_species_timepoint(f, t, n_samples=100, seed=42):
        if seed is not None:
            random.seed(seed + f * 1000 + t)
            np.random.seed(seed + f * 1000 + t)
        perturbed_inputs_S = []
        perturbed_inputs_Si = []
        timesteps, features = tensor_true_test_in.cpu().shape
        variables = [(fi, ti) for fi in range(features) for ti in range(timesteps) if (fi, ti) != (f, t)]
        for _ in range(n_samples):
            k = np.random.randint(0, len(variables)+1)
            S = set(random.sample(variables, k))
            mask = torch.zeros((timesteps, features), dtype=torch.bool)
            for (fi, ti) in S:
                mask[ti, fi] = True
            mask_S = mask.clone()
            mask_Si = mask.clone()
            mask_Si[t, f] = True
            x_S = baseline.clone()
            x_S[mask_S] = tensor_true_test_in.cpu()[mask_S]
            x_Si = baseline.clone()
            x_Si[mask_Si] = tensor_true_test_in.cpu()[mask_Si]
            perturbed_inputs_S.append(x_S)
            perturbed_inputs_Si.append(x_Si)
        batch_S = torch.stack(perturbed_inputs_S)
        batch_Si = torch.stack(perturbed_inputs_Si)
        with torch.no_grad():
            n_layers = best_model.num_layers
            batch_size = batch_S.size(0)
            best_model.hidden = (
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_S.device),
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_S.device)
            )
            out_S = best_model(batch_S)
            best_model.hidden = (
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_Si.device),
                torch.zeros(n_layers, batch_size, best_model.hidden_layer_size).to(batch_Si.device)
            )
            out_Si = best_model(batch_Si)
        v_S = out_S[:, :].cpu().numpy()
        v_Si = out_Si[:, :].cpu().numpy()
        contribs = np.mean(v_Si - v_S, axis=0)  # mean over batch, per output species
        # For the heatmap, aggregate as in original code (mean over output species)
        return np.mean(contribs)

    timesteps, features = tensor_true_test_in.cpu().shape
    results = Parallel(n_jobs=4)(
        delayed(compute_shapley_for_species_timepoint)(f, t, n_samples=100, seed=42)
        for f in range(features) for t in range(timesteps)
    )
    shapley_matrix = np.array(results).reshape((features, timesteps))
    plt.figure(figsize=(14, 8))
    sns.heatmap(shapley_matrix, xticklabels=[f'T{t+1}' for t in range(shapley_matrix.shape[1])], yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=False)
    plt.xlabel('Timepoint')
    plt.ylabel('Species')
    plt.title('Shapley Attribution Heatmap (Species × Timepoint)')
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    # Marginal sums/averages
    species_marginal = shapley_matrix.mean(axis=1)
    timepoint_marginal = shapley_matrix.mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(blastT_labels, species_marginal)
    plt.ylabel('Mean Shapley Value')
    plt.title('Mean Shapley Attribution per Species (Averaged over Timepoints)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)
    plt.figure(figsize=(10, 4))
    plt.bar([f'T{t+1}' for t in range(shapley_matrix.shape[1])], timepoint_marginal)
    plt.ylabel('Mean Shapley Value')
    plt.title('Mean Shapley Attribution per Timepoint (Averaged over Species)')
    plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)

# === PDF OUTPUT FOR ALL PLOTS ===
with PdfPages('shapley_results.pdf') as pdf:
    # --- PREDICTION PLOT ---
    base_data_color = ['black', 'red', 'orange', 'green', 'blue', 'gold', 'purple', 'pink', 'grey', 'turquoise', 'cyan', 'crimson', 'indigo', 'olive', 'saddlebrown']
    pred_color = base_data_color
    best_fit_idx = list(range(4))
    full_test_data = torch.cat((torch.clone(tensor_true_test_in),torch.clone(tensor_true_test_data)))
    lstm_funcs.plot_best_fit(full_test_data, best_prediction, pred_color, base_data_color, best_fit_idx, blastT_labels, False, None, full_test_data, run_number=z+1)
    pdf.savefig()  # Save prediction plot
    plt.close()

    # --- SHAPLEY ATTRIBUTION HEATMAP ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(shapley_matrix, xticklabels=blastT_labels, yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=True, fmt=".2f")
    plt.xlabel('Neighbor Species')
    plt.ylabel('Target Species')
    plt.title(f'Shapley Attribution Heatmap (Neighbor Effects) - Run {z+1}')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- AVERAGE ACROSS ENSEMBLES ---
    if ensemble_shapley_matrices:
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_shapley_matrix, xticklabels=blastT_labels, yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=True, fmt=".2f")
        plt.xlabel('Neighbor Species')
        plt.ylabel('Target Species')
        plt.title('Average Shapley Attribution Heatmap (Neighbor Effects) - All Runs')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- SPECIES-TIMEPOINT SHAPLEY HEATMAP ---
        plt.figure(figsize=(14, 8))
        sns.heatmap(shapley_matrix, xticklabels=[f'T{t+1}' for t in range(shapley_matrix.shape[1])], yticklabels=blastT_labels, center=0, cmap='coolwarm', annot=False)
        plt.xlabel('Timepoint')
        plt.ylabel('Species')
        plt.title('Shapley Attribution Heatmap (Species × Timepoint)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Marginal sums/averages
        plt.figure(figsize=(10, 4))
        plt.bar(blastT_labels, species_marginal)
        plt.ylabel('Mean Shapley Value')
        plt.title('Mean Shapley Attribution per Species (Averaged over Timepoints)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.bar([f'T{t+1}' for t in range(shapley_matrix.shape[1])], timepoint_marginal)
        plt.ylabel('Mean Shapley Value')
        plt.title('Mean Shapley Attribution per Timepoint (Averaged over Species)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

# Comment out interactive plot display
# plt.ioff()
# plt.show()
# plt.pause(0.1)
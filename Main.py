# Unknown = Unknown
# 1 = Illicit
# 2 = Licit
#Code for paper reproduce-ablility
seeded_run = True
parameter_tuning = True
num_epochs = 200
#Detecting system (pc or mac)
import platform
pc = platform.system()

import os
if pc == 'Darwin':
    os.chdir("/Users/lambertusvanzyl/Desktop/Reproduce-ability_paper")
    #/Users/lambertusvanzyl/Desktop/Reproduce-ability_paper
else:
    os.chdir("/Users/Lambertus/Desktop/Reproduce-ability_paper")
    #"C:\Users\Lambertus\Desktop\Reproduce-ability_paper"
    
#%% Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx

from functools import partial

#Importing PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool

#Importing sklearn libraries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import sys

#%% Importing custom libraries
from reading_data import readfiles
from pre_processing import elliptic_pre_processing, create_data_object, create_elliptic_masks
from Models import GCN, GAT, GIN
from helper_functions import apply_node_mask_and_remap, train_gnn

#%% Setting seed
if seeded_run == True:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
#%%Getting data ready for models
#Reading in data
features_df, classes_df, edgelist_df = readfiles(pc)
classes_df, edgelist_df, features_df, known_nodes = elliptic_pre_processing(classes_df, edgelist_df, features_df)
# Create data object
data = create_data_object(features_df, edgelist_df, classes_df)
#Create mask for data
train_mask, val_mask_backdated, test_mask_backdated, train_perf_eval, val_perf_eval, test_perf_eval = create_elliptic_masks(features_df, edgelist_df, known_nodes)

#%% Testing if model runs
#Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(num_node_features = data.num_features, num_classes = 2, hidden_units = 64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
#Getting data variables for each phase
train_data = apply_node_mask_and_remap(data, train_mask, features_df)
val_data = apply_node_mask_and_remap(data, val_mask_backdated, features_df)
test_data = apply_node_mask_and_remap(data, test_mask_backdated, features_df)

#%%
#Testing
temp = data.y.cpu().numpy()[:161603]
out = temp[val_perf_eval]
print(f'temp: {temp}\nout: {out}')
#%%
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)
#%%
#metrics, best_f1_model_wts = train_gnn(num_epochs=200, data=train_data, model=model, optimizer=optimizer, criterion=criterion, train_mask=train_mask, train_perf_eval=train_perf_eval, val_data=val_data, val_perf_eval=val_perf_eval)


# %% Optuna
import optuna

if parameter_tuning == True:
    from Optuna import run_optuna
    best_params, best_value = run_optuna(
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=num_epochs,
        train_mask=train_mask,
        train_perf_eval=train_perf_eval,
        val_perf_eval=val_perf_eval,
        n_trials=100)
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    
#%% Testing model
testing = True
if testing == True:
    # Set model and optimizer parameters from Optuna best_params
    hidden_units = best_params.get('hidden_units', 64)
    lr = best_params.get('learning_rate', 0.05)
    weight_decay = best_params.get('weight_decay', 5e-4)
    #num_heads = best_params.get('num_heads', 1)

    # Re-initialize model and optimizer with best parameters
    model = GCN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Extracted Optuna parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best Optuna objective value: {best_value}")

    # Now you can run validation/testing using the model with best parameters
    # metrics, best_f1_model_wts = train_gnn(...)
    
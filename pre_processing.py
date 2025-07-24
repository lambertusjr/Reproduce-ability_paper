def elliptic_pre_processing(classes_df, edgelist_df, features_df):
    #Geting known nodes
    known_nodes = classes_df != 'unknown'
    known_nodes = known_nodes['class'].values
    #Remap class labels
    class_mapping = {
        'unknown': -1,
        '1': 0,
        '2': 1
    }
    classes_df['class'] = classes_df['class'].map(class_mapping)
    classes_df['class'] = classes_df['class'].astype(int) 
    return classes_df, edgelist_df, features_df, known_nodes
import torch
from torch_geometric.data import Data
def create_data_object(features_df, edgelist_df, classes_df, node_mask = None, edge_mask = None):
    #Converting dataframes to tensors
    features_tensor = torch.tensor(features_df.drop(columns=['txId']).values, dtype=torch.float)
    edgelist_tensor = torch.tensor(edgelist_df.values.T, dtype=torch.long)
    classes_tensor = torch.tensor(classes_df['class'].values, dtype=torch.int)
    
    data = Data(x = features_tensor, edge_index=edgelist_tensor, y=classes_tensor)
    
    return data
import numpy as np
import pandas as pd
def create_elliptic_masks(features_df, edgelist_df, known_nodes):
    #Getting unique time steps
    time_steps = features_df['time_step'].unique()
    #Creating masks depending on time steps
    #Training 0-30
    #Validation 31-40
    #Testing 41-49
    train_mask = (features_df['time_step'] <= 30) & (features_df['time_step'] >= 1)
    val_mask = (features_df['time_step'] <= 40) & (features_df['time_step'] >= 31)
    test_mask = (features_df['time_step'] <= 49) & (features_df['time_step'] >= 41)
    
    #Creating masks that contain previous time steps to maximise data usage
    val_mask_backdated = val_mask | train_mask
    test_mask_backdated = test_mask | val_mask_backdated
    train_size = train_mask.sum()
    val_size = val_mask.sum()
    test_size = test_mask.sum()
    
    #Creating masks used for performance evaluation
    train_perf_eval = train_mask[:train_mask.sum()] & known_nodes[:train_mask.sum()]
    #Creating validation performance evaluation mask
    val_begin_empty = np.zeros((train_mask.sum()), dtype=bool)
    known_nodes_val_only = known_nodes[(123287):(train_mask.sum() + val_size)]
    val_perf_eval = np.concatenate((val_begin_empty, known_nodes_val_only))
    
    
    #Creating test performance evaluation mask
    test_begin_empty = np.zeros(val_mask_backdated.sum(),dtype=bool)
    known_nodes_test_only = known_nodes[(val_mask_backdated.sum() +1):]
    test_perf_eval = np.concatenate((test_begin_empty, known_nodes_test_only))
    
    #Converting series to numpy arrays
    train_mask = train_mask.to_numpy()
    train_perf_eval = train_perf_eval.to_numpy()
    val_mask_backdated = val_mask_backdated.to_numpy()
    test_mask_backdated = test_mask_backdated.to_numpy()
    
    return train_mask, val_mask_backdated, test_mask_backdated, train_perf_eval, val_perf_eval, test_perf_eval
    
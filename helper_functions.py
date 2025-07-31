import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch_geometric
from torch_geometric.data import Data
from debugging import compare_1d_tensors

def apply_node_mask_and_remap(data: Data, node_mask: torch.Tensor, e_txs_feat) -> Data:
    # Convert node features to NumPy array (excluding 'txId')
    x_np = e_txs_feat.drop(columns=['txId']).to_numpy()
    y_np = data.y.cpu().numpy()
    ids = e_txs_feat['txId'].to_numpy()

    # Apply node mask
    node_mask_np = node_mask.numpy() if isinstance(node_mask, torch.Tensor) else node_mask
    x_masked = x_np[node_mask_np]
    y_masked = y_np[node_mask_np]
    masked_ids = ids[node_mask_np]

    # Build remapping: old ID -> new ID
    max_id = np.max(ids) + 1
    remap_array = -np.ones(max_id, dtype=int)
    remap_array[masked_ids] = np.arange(len(masked_ids))

    # Remap edge indices
    edge_index_np = data.edge_index.cpu().numpy()
    src, dst = edge_index_np[0], edge_index_np[1]
    src_remapped = remap_array[src]
    dst_remapped = remap_array[dst]

    # Filter out edges connected to masked-out nodes
    valid_edge_mask = (src_remapped != -1) & (dst_remapped != -1)
    edge_index_filtered = np.stack([src_remapped[valid_edge_mask], dst_remapped[valid_edge_mask]])

    # Convert to tensors
    x_tensor = torch.tensor(x_masked, dtype=torch.float)
    y_tensor = torch.tensor(y_masked, dtype=torch.long)
    edge_index_tensor = torch.tensor(edge_index_filtered, dtype=torch.long)
    edge_attr_tensor = torch.ones(edge_index_tensor.shape[1], 1)

    # Build new data object
    new_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)
    return new_data

def train_and_val_gnn(num_epochs, data, model, optimizer, criterion, train_mask, train_perf_eval, val_data, val_perf_eval):
    """
    
    """
    best_f1 = 0.0
    best_f1_model_wts = None
    # Creating dictionaries to store metrics
    metrics = {
        'train': {
            'acc': [], 'prec': [], 'prec_weighted': [], 'rec': [],
            'rec_weighted': [], 'f1': [], 'f1_weighted': [], 'auc': []
        },
        'val': {
            'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []
        }
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_perf_eval], data.y[train_perf_eval])
        loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        y_train_pred = out[train_perf_eval].argmax(dim=1)
        y_train_true = data.y[train_perf_eval]
        train_acc = (y_train_pred == y_train_true).sum().item() / len(y_train_true)
        train_prec = precision_score(y_train_true.cpu(), y_train_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        train_prec_weighted = precision_score(y_train_true.cpu(), y_train_pred.cpu(), average='weighted', zero_division=0)
        train_rec = recall_score(y_train_true.cpu(), y_train_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        train_rec_weighted = recall_score(y_train_true.cpu(), y_train_pred.cpu(), average='weighted', zero_division=0)
        train_f1 = f1_score(y_train_true.cpu(), y_train_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        train_f1_weighted = f1_score(y_train_true.cpu(), y_train_pred.cpu(), average='weighted', zero_division=0)
        train_auc = sklearn.metrics.roc_auc_score(y_train_true.cpu(), out[train_perf_eval][:, 1].detach().cpu())
        
        metrics['train']['acc'].append(train_acc)
        metrics['train']['prec'].append(train_prec)
        metrics['train']['prec_weighted'].append(train_prec_weighted)
        metrics['train']['rec'].append(train_rec)
        metrics['train']['rec_weighted'].append(train_rec_weighted)
        metrics['train']['f1'].append(train_f1)
        metrics['train']['f1_weighted'].append(train_f1_weighted)
        metrics['train']['auc'].append(train_auc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            y_val_pred = val_out[val_perf_eval].argmax(dim=1)
            y_val_true = val_data.y[val_perf_eval]
            
            val_acc = (y_val_pred == y_val_true).sum().item() / len(y_val_true)
            val_prec = precision_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            val_rec = recall_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            val_f1 = f1_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            val_auc = sklearn.metrics.roc_auc_score(y_val_true.cpu(), val_out[val_perf_eval][:, 1].cpu())
            
            metrics['val']['acc'].append(val_acc)
            metrics['val']['prec'].append(val_prec)
            metrics['val']['rec'].append(val_rec)
            metrics['val']['f1'].append(val_f1)
            metrics['val']['auc'].append(val_auc)
            
            #Save the best model based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_f1_model_wts = deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Precision (Weighted)', 'Recall', 'Recall (Weighted)', 'F1-Score', 'F1-Score (Weighted)', 'AUC'],
                'Train': [train_acc, train_prec, train_prec_weighted, train_rec, train_rec_weighted, train_f1, train_f1_weighted, train_auc],
                'Validation': [val_acc, val_prec, 'N/A', val_rec, 'N/A', val_f1, 'N/A', val_auc]
            })
            #print(df.to_string())

    return metrics, best_f1_model_wts

def evaluate(model, val_data, val_perf_eval):
    model.eval()
    with torch.no_grad():
        out = model(val_data)
        y_val_pred = out[val_perf_eval].argmax(dim=1)
        y_val_true = val_data.y[val_perf_eval]
        
        val_acc = (y_val_pred == y_val_true).sum().item() / len(y_val_true)
        val_prec = precision_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        val_rec = recall_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        val_f1 = f1_score(y_val_true.cpu(), y_val_pred.cpu(), pos_label=0, average='binary', zero_division=0)
        val_auc = sklearn.metrics.roc_auc_score(y_val_true.cpu(), out[val_perf_eval][:, 1].cpu())
        
        return val_acc, val_prec, val_rec, val_f1, val_auc

from Models import GCN, GAT, GIN
def testing_GNN(num_epochs, data, model, optimizer, criterion, train_mask, train_perf_eval, val_data, val_perf_eval, test_data, test_perf_eval):
    #This function is used to validate performance results
    avg_results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    for i in range(30):
        #Training and validation
        metrics, best_f1_model_wts = train_and_val_gnn(num_epochs, data, model, optimizer, criterion, train_mask, train_perf_eval, val_data, val_perf_eval)
        #Getting results from validation
        avg_results['acc'].append(metrics['val']['acc'][-1])
        avg_results['prec'].append(metrics['val']['prec'][-1])
        avg_results['rec'].append(metrics['val']['rec'][-1])
        avg_results['f1'].append(metrics['val']['f1'][-1])
        avg_results['auc'].append(metrics['val']['auc'][-1])
        
        # Load the best model weights
        model.load_state_dict(best_f1_model_wts)
        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_out = model(test_data)
            y_test_pred = test_out[test_perf_eval].argmax(dim=1)
            y_test_true = test_data.y[test_perf_eval]
            
            #compare_1d_tensors(y_test_pred, y_test_true, precision=4)
            
            test_acc = (y_test_pred == y_test_true).sum().item() / len(y_test_true)
            test_prec = precision_score(y_test_true.cpu(), y_test_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            test_rec = recall_score(y_test_true.cpu(), y_test_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            test_f1 = f1_score(y_test_true.cpu(), y_test_pred.cpu(), pos_label=0, average='binary', zero_division=0)
            test_auc = sklearn.metrics.roc_auc_score(y_test_true.cpu(), test_out[test_perf_eval][:, 1].cpu())
            
            print(f"Test Results - Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, F1-Score: {test_f1}, AUC: {test_auc}")
    return avg_results
    #Beginning training
    
    
    


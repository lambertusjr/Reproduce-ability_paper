import optuna
import numpy as np
import torch
from functools import partial

from helper_functions import train_gnn, evaluate
from Models import GCN, GAT, GIN

def objective(trial, train_data, device, num_epochs, train_mask, train_perf_eval, val_data, val_perf_eval):
    #Setting hyperparameters for optuna runs
    hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 48, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    alpha = trial.suggest_float("alpha", 0.2, 0.7)
    gamma = trial.suggest_float("gamma", 2.0, 5.0)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    model = GAT(num_node_features=train_data.num_features, num_classes=2, hidden_units=hidden_units, num_heads=num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    train_data = train_data.to(device)
    model = model.to(device)
    print(f"\n=== Trial {trial.number} ===")
    
    metrics, best_f1_model_wts = train_gnn(num_epochs, train_data, model, optimizer, criterion, train_mask, train_perf_eval, val_data, val_perf_eval)
    
    model.load_state_dict(best_f1_model_wts)
    
    val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_data, val_perf_eval)
    
    return val_f1

def clear_log_file(output_file = "optuna_log.txt"):
    """
    Clear the contents of the log file.

    Parameters
    ----------
    output_file : str, optional
        The path to the log file to be cleared. Default is "optuna_log.txt".
    """
    with open(output_file, 'w') as f:
        pass 
    print(f"{output_file} has been cleared.")
    
    
def run_optuna(train_data, device, num_epochs, train_mask, train_perf_eval, val_data, val_perf_eval, n_trials=100):
    study = optuna.create_study(direction="maximize", study_name="GNN Hyperparameter Optimization")
    study.optimize(partial(objective, train_data=train_data, device=device, num_epochs=num_epochs,train_mask=train_mask, train_perf_eval=train_perf_eval, val_data=val_data, val_perf_eval=val_perf_eval), n_trials=n_trials)
    best_params = study.best_params
    best_value = study.best_value
    output_file = "optuna_log.txt"
    with open(output_file, "a") as f:
        f.write("GCN model: \n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    return best_params, best_value
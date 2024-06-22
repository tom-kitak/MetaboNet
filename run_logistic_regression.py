import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    auc,
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)
import logistic_regression_model as log_reg_model
import importlib
import datetime
import json
import os
import time
import random
import copy
import argparse

importlib.reload(log_reg_model)

parser = argparse.ArgumentParser(description="Set config parameters")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
args = parser.parse_args()

# Configuration
config = {
    "max_num_epochs": 100,
    "random_state": args.seed,
    "validation_size": 0.2,
    "test_size": 0.2,
    "batch_size": 32,
    "n_folds": 10,
    "data_path": "/trinity/home/r103868/data",  # /storage/scratch/groshchupkin/Tom_dataset
    "select_after_epoch": 10,
    "n_averages_model": 10,
    "seed_runs_folder": "results_test1",
}
config["experiment_name"] = f"logreg_seed_{config['random_state']}"
# Hyperparameter grid
hyperparameter_grid = {
    "learning_rate": [0.001, 0.0001],
    "l1_value": [0.1, 0.01, 0.001, 0],
    "positive_class_weight": [8, 12, 16],
    "scheduler": [
        {
            "type": "StepLR",  # Fixed scheduler
            "params": {"step_size": 30, "gamma": 0.1},
        },
        {
            "type": "ReduceLROnPlateau",  # Plateau Scheduler
            "params": {"mode": "max", "factor": 0.1, "patience": 20},
        },
        {
            "type": "CosineAnnealingLR",  # Cosine Annealing Scheduler
            "params": {"T_max": 100},
        },
    ],
}
# Flags
plot = False
prints = True

print("config:", config)

# Record the start time
start_time = time.time()
interesting_info = dict()

# Device configuration for GPU/CPU usage
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    interesting_info["device"] = (
        f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})"
    )
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU)")
    interesting_info["device"] = f"Using device: {device} (CPU)"

# Set seeds for reproducibility
np.random.seed(config["random_state"])
torch.manual_seed(config["random_state"])
random.seed(config["random_state"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["random_state"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading
data = pd.read_csv(config["data_path"] + "/dataset.csv")
X = data.iloc[:, 3:].values
y = data["depression_label"].values
original_indices = np.arange(len(data))  # Save the original indices

# Splitting the data into training, validation, and test sets (60-20-20 split)
X_train_val, X_test, y_train_val, y_test, _, test_indices = train_test_split(
    X,
    y,
    original_indices,
    test_size=config["test_size"],
    stratify=y,
    random_state=config["random_state"],
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=config["validation_size"] / (1 - config["test_size"]),
    stratify=y_train_val,
    random_state=config["random_state"],
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])


def create_scheduler(optimizer, scheduler_config):
    scheduler_type = scheduler_config["type"]
    scheduler_params = scheduler_config["params"]
    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def average_model_states(states):
    avg_state = {}
    # Iterate through each key in the state dictionary
    for key in states[0].keys():
        # Stack all the tensors for this key along a new dimension and compute the mean
        stacked_tensors = torch.stack([s[key].float() for s in states], dim=0)
        avg_state[key] = torch.mean(stacked_tensors, dim=0)
    return avg_state


# Model training and hyperparameter optimization
def train_and_evaluate(
    fn_model,
    fn_train_loader,
    fn_fold_val_loader,
    fn_criterion,
    fn_optimizer,
    fn_scheduler,
    fn_max_num_epochs,
    device,
    return_averaged_model=False,
):
    fn_train_losses = []
    fn_val_losses = []
    fn_val_mcc = []

    # Store top 10
    top_scores = []
    if return_averaged_model:
        top_models = []

    for epoch in range(fn_max_num_epochs):
        fn_model.train()
        epoch_train_loss = 0.0

        for inputs, labels in fn_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            fn_optimizer.zero_grad()
            outputs = fn_model(inputs)
            loss = fn_criterion(outputs, labels)
            loss += fn_model.l1_regularization()  # Add L1 regularization to the loss
            epoch_train_loss += loss.item() * inputs.size(0)
            loss.backward()
            fn_optimizer.step()

        fn_train_losses.append(epoch_train_loss / len(fn_train_loader.dataset))

        fn_model.eval()
        epoch_val_loss = 0.0
        epoch_val_preds, epoch_val_labels = [], []
        with torch.no_grad():
            for inputs, labels in fn_fold_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = fn_model(inputs)
                loss = fn_criterion(outputs, labels)
                loss += fn_model.l1_regularization()
                epoch_val_loss += loss.item() * inputs.size(0)
                epoch_val_preds.extend(outputs.detach().cpu().numpy().flatten())
                epoch_val_labels.extend(labels.detach().cpu().numpy().flatten())

        fn_val_losses.append(epoch_val_loss / len(fn_fold_val_loader.dataset))

        epoch_val_mcc = matthews_corrcoef(epoch_val_labels, np.round(epoch_val_preds))
        fn_val_mcc.append(epoch_val_mcc)

        if epoch > config["select_after_epoch"]:
            top_scores.append(epoch_val_mcc)
            top_scores = sorted(top_scores, reverse=True)[: config["n_averages_model"]]
            if return_averaged_model:
                top_models.append((epoch_val_mcc, copy.deepcopy(fn_model.state_dict())))
                top_models = sorted(top_models, key=lambda x: x[0], reverse=True)[
                    : config["n_averages_model"]
                ]

        # Step the scheduler (if not ReduceLROnPlateau)
        if isinstance(fn_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            fn_scheduler.step(epoch_val_mcc)
        else:
            fn_scheduler.step()

    if plot:
        # Plotting loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(fn_train_losses)), fn_train_losses, label="Training Loss")
        plt.plot(range(len(fn_val_losses)), fn_val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.show()

        # Plotting MCC curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(fn_val_mcc)), fn_val_mcc)
        plt.xlabel("Epoch")
        plt.ylabel("MCC score")
        plt.title("Training and Validation MCC Score Over Epochs")
        plt.show()

    # Get average MCC
    top_n_average_mcc = np.mean(top_scores)
    if prints:
        print(f"    Top 10 average MCC: {top_n_average_mcc}")

    if return_averaged_model:
        # Average the top 10 models
        averaged_model_state = average_model_states([state for _, state in top_models])
        fn_model.load_state_dict(averaged_model_state)
        return fn_model, top_n_average_mcc
    else:
        return None, top_n_average_mcc


# K-Fold Cross Validation with hyperparameter optimization
print("K-Fold Cross Validation with hyperparameter optimization")

best_hyperparameters = None
best_folds_val_mcc = -1.0

kf = StratifiedKFold(
    n_splits=config["n_folds"], shuffle=True, random_state=config["random_state"]
)

for params in ParameterGrid(hyperparameter_grid):
    if prints:
        print(f"Current testing hyperparameters: {params}")

    folds_avg_val_mcc = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        if prints:
            print(f"  Cross-validation Fold {fold+1}")

        fold_train_subset = Subset(
            TensorDataset(X_train_tensor, y_train_tensor), train_idx
        )
        fold_val_subset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_idx)
        fold_train_loader = DataLoader(
            fold_train_subset, batch_size=config["batch_size"], shuffle=True
        )
        fold_val_loader = DataLoader(
            fold_val_subset, batch_size=config["batch_size"], shuffle=False
        )

        fold_model = log_reg_model.LogisticRegressionModel(
            X.shape[1], l1_value=params["l1_value"], device=device
        )
        fold_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([params["positive_class_weight"]]).to(device)
        )
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=params["learning_rate"])
        fold_scheduler = create_scheduler(fold_optimizer, params["scheduler"])
        fold_model.to(device)

        _, fold_val_mcc = train_and_evaluate(
            fold_model,
            fold_train_loader,
            fold_val_loader,
            fold_criterion,
            fold_optimizer,
            fold_scheduler,
            config["max_num_epochs"],
            device,
            return_averaged_model=False,
        )
        folds_avg_val_mcc += fold_val_mcc

    folds_avg_val_mcc /= config["n_folds"]
    if prints:
        print(f"  Avg Folds MCC: {folds_avg_val_mcc}")

    if folds_avg_val_mcc > best_folds_val_mcc:
        best_folds_val_mcc = folds_avg_val_mcc
        best_hyperparameters = params

print(f"Best cross-validation hyperparameters: {best_hyperparameters}")
print(f"Best cross-validation MCC: {best_folds_val_mcc}")

# Final Model Training on the training set only with the best hyperparameters and Validation with the chosen model
print("Final Model Training on the training set only with the best hyperparameters")

tuned_hyperparameters_model = log_reg_model.LogisticRegressionModel(
    X.shape[1], l1_value=best_hyperparameters["l1_value"], device=device
)

final_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([best_hyperparameters["positive_class_weight"]]).to(device)
)
final_optimizer = optim.Adam(
    tuned_hyperparameters_model.parameters(), lr=best_hyperparameters["learning_rate"]
)
final_scheduler = create_scheduler(final_optimizer, best_hyperparameters["scheduler"])
tuned_hyperparameters_model.to(device)

final_model, _ = train_and_evaluate(
    tuned_hyperparameters_model,
    train_loader,
    val_loader,
    final_criterion,
    final_optimizer,
    final_scheduler,
    config["max_num_epochs"],
    device,
    return_averaged_model=True,
)

# Testing the final model
test_preds, test_labels = [], []
final_model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = final_model(inputs)
        test_preds.extend(outputs.detach().cpu().numpy().flatten())
        test_labels.extend(labels.detach().cpu().numpy().flatten())

final_test_preds = np.round(test_preds)

# Collect test classification information
test_classification_information = {
    "original_index": test_indices,
    "true_class": test_labels,
    "predicted_class": final_test_preds,
    "predicted_prob": test_preds,
}

test_conf_matrix = confusion_matrix(test_labels, final_test_preds)
test_f1 = f1_score(test_labels, final_test_preds)
test_roc_auc = roc_auc_score(test_labels, test_preds)

precision, recall, _ = precision_recall_curve(test_labels, test_preds)
test_auc_pr = auc(recall, precision)

test_mcc = matthews_corrcoef(test_labels, final_test_preds)

# Model Interpretation and Analysis
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("Test F1 Score:", test_f1)
print("Test ROC-AUC:", test_roc_auc)
print("Test AUC-PR:", test_auc_pr)
print("Test MCC:", test_mcc)
print("Test Classification Report:")
print(classification_report(test_labels, final_test_preds))

# Saving results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
seed_runs_folder = config["data_path"] + f"/results/{config['seed_runs_folder']}"
os.makedirs(seed_runs_folder, exist_ok=True)
results_dir = seed_runs_folder + f"/{config['experiment_name']}"
os.makedirs(results_dir, exist_ok=True)

np.save(os.path.join(results_dir, "test_labels.npy"), np.array(test_labels))
np.save(os.path.join(results_dir, "test_preds.npy"), np.array(test_preds))

# Save the configuration
config_filename = os.path.join(results_dir, "config.json")
with open(config_filename, "w") as f:
    json.dump(config, f)
print(f"Configuration saved to {config_filename}")

# Save the hyperparameter grid
hyperparameter_grid_filename = os.path.join(results_dir, "hyperparameter_grid.json")
with open(hyperparameter_grid_filename, "w") as f:
    json.dump(hyperparameter_grid, f)
print(f"Hyperparameter grid saved to {hyperparameter_grid_filename}")

# Save the best hyperparameters
best_hyperparameters_filename = os.path.join(results_dir, "best_hyperparameters.json")
with open(best_hyperparameters_filename, "w") as f:
    json.dump(best_hyperparameters, f)
print(f"Best hyperparameters saved to {best_hyperparameters_filename}")

# Save the best model weights
final_model_weights_filename = os.path.join(results_dir, "final_model_weights.pt")
torch.save(final_model.state_dict(), final_model_weights_filename)
print(f"Best model weights saved to {final_model_weights_filename}")

# Save test results
test_dataset_results = {
    "confusion_matrix": test_conf_matrix.tolist(),
    "f1_score": test_f1,
    "roc_auc": test_roc_auc,
    "auc_pr": test_auc_pr,
    "mcc": test_mcc,
    "classification_report": classification_report(
        test_labels, final_test_preds, output_dict=True
    ),
}
test_dataset_filename = os.path.join(results_dir, "test_dataset_results.json")
with open(test_dataset_filename, "w") as f:
    json.dump(test_dataset_results, f)
print(f"Test results saved to {test_dataset_filename}")

class_info_test_path = os.path.join(results_dir, "classification_info_test.csv")
test_classification_information_df = pd.DataFrame(test_classification_information)
test_classification_information_df.to_csv(class_info_test_path, index=False)
print(f"Classification info saved to {class_info_test_path}")

# Record the end time
end_time = time.time()
duration = end_time - start_time

# Convert duration to minutes and seconds
minutes = int(duration // 60)
seconds = duration % 60

print(f"Time taken: {minutes} minutes and {seconds:.2f} seconds")

interesting_info["time taken"] = f"{minutes} min {seconds:.2f} sec"

interesting_info_filename = os.path.join(results_dir, "interesting_info.json")
with open(interesting_info_filename, "w") as f:
    json.dump(interesting_info, f)
print(f"Interesting info saved to {interesting_info_filename}")

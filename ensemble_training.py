import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_architecture import CrystalToxicityTransformer
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from hyperparameter_tuning import GraphDataset, collate_fn
# Initialize logger
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCH = 3000
class EnsembleModel(nn.Module):
    """
    Ensemble of CrystalToxicityTransformer models
    Averages predictions from multiple models
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, graph_data, exp_features):
        """Average predictions from all models"""
        predictions = []
        for model in self.models:
            pred = model(graph_data, exp_features)
            predictions.append(pred)
        return torch.stack(predictions).mean(dim=0)


def train_single_model(model, train_loader, val_loader, epochs=EPOCH, patience=300,
                       learning_rate=0.001, weight_decay=1e-4):
    """
    Train a single model with early stopping
    Returns trained model and validation loss
    """
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=50, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=0.5)

    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batched_graph, X_batch, y_batch in train_loader:
            batched_graph = batched_graph.to(DEVICE)
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(batched_graph, X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batched_graph, X_batch, y_batch in val_loader:
                batched_graph = batched_graph.to(DEVICE)
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(batched_graph, X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)

        # Update scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_model)
    return model, best_loss


def train_ensemble_parallel(study, graphs_train, X_train, y_train, graphs_val, X_val, y_val,
                            n_models=5, num_workers=4):
    """
    Train an ensemble of models in parallel using top hyperparameters
    Returns the ensemble model
    """
    # Select top hyperparameter configurations
    top_trials = sorted(study.trials, key=lambda t: t.value)[:n_models]
    configs = [trial.params for trial in top_trials]

    # Create datasets
    train_dataset = GraphDataset(graphs_train, X_train, y_train)
    val_dataset = GraphDataset(graphs_val, X_val, y_val)

    # Create models
    models = []
    for config in configs:
        # Extract only model architecture parameters
        model_params = {
            'exp_feature_dim': X_train.shape[1],
            'gnn_hidden_dim': config.get('gnn_hidden_dim', 256),
            'transformer_dim': config.get('transformer_dim', 256),
            'nhead': config.get('nhead', 8),
            'n_layers': config.get('n_layers', 3),
            'dropout_rate': config.get('dropout_rate', 0.2)
        }

        # Extract optimizer parameters separately
        optimizer_params = {
            'learning_rate': config.get('learning_rate', 0.001),
            'weight_decay': config.get('weight_decay', 1e-4)
        }

        model = CrystalToxicityTransformer(**model_params)
        models.append((model, optimizer_params))

    # Train models in parallel
    trained_models = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, (model, optimizer_params) in enumerate(models):
            logger.info(f"Submitting model {i + 1} for training with params: {model_params}")

            # Create new data loaders for each model
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=4, pin_memory=True
            )

            # Submit training task with optimizer parameters
            future = executor.submit(
                train_single_model,
                model=model.to(DEVICE),
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCH,
                patience=300,
                learning_rate=optimizer_params['learning_rate'],
                weight_decay=optimizer_params['weight_decay']
            )
            futures.append(future)

        # Collect results
        for i, future in enumerate(as_completed(futures)):
            model, val_loss = future.result()
            trained_models.append(model.cpu())  # Move to CPU to save GPU memory
            logger.info(f"Model {i + 1} training completed with val loss: {val_loss:.4f}")

    # Create ensemble
    ensemble = EnsembleModel(trained_models).to(DEVICE)
    return ensemble

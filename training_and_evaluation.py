import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from torch.utils.data import DataLoader
#from graph_construction import Batch  # Import from your graph_construction.py
from hyperparameter_tuning import hyperparameter_tuning
from hyperparameter_tuning import GraphDataset, collate_fn

# Initialize logger
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batched_graph, X_batch, y_batch in test_loader:
            batched_graph = batched_graph.to(DEVICE)
            X_batch = X_batch.to(DEVICE)
            
            preds = model(batched_graph, X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": all_preds,
        "targets": all_targets
    }

def train_full_pipeline(graphs_train, X_train, y_train, graphs_val, X_val, y_val, 
                        graphs_test, X_test, y_test, n_trials=30, n_models=5):
    """
    Complete training pipeline:
    1. Hyperparameter tuning
    2. Ensemble training
    3. Final evaluation
    """
    # Hyperparameter tuning
    study = hyperparameter_tuning(
        graphs_train, X_train, y_train,
        graphs_val, X_val, y_val,
        n_trials=n_trials
    )
    
    # Ensemble training
    from ensemble_training import train_ensemble_parallel
    ensemble = train_ensemble_parallel(
        study, graphs_train, X_train, y_train,
        graphs_val, X_val, y_val, n_models=n_models
    )
    
    # Prepare test loader
    test_dataset = GraphDataset(graphs_test, X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Evaluate ensemble
    results = evaluate_model(ensemble, test_loader)
    
    logger.info(f"Test Results:")
    logger.info(f"MAE: {results['mae']:.4f}")
    logger.info(f"RMSE: {results['rmse']:.4f}")
    logger.info(f"R²: {results['r2']:.4f}")
    
    return ensemble, results

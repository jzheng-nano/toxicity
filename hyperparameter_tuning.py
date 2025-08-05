import optuna
import torch
import logging
import numpy as np
import gc
from tqdm import tqdm
from torch.cuda import amp
from optuna.visualization import plot_optimization_history, plot_intermediate_values
from model_architecture import CrystalToxicityTransformer
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Initialize logger
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # Increased batch size for better GPU utilization
NUM_WORKERS = 4  # More workers for faster data loading
PIN_MEMORY = True  # Enable pinned memory for faster data transfer


class GraphDataset(Dataset):
    """Optimized dataset for graph data with caching"""

    def __init__(self, graphs, X, y):
        self.graphs = graphs
        self.X = np.array(X, dtype=np.float32)  # Use float32 for memory efficiency
        self.y = np.array(y, dtype=np.float32)
        self.cached_data = [None] * len(graphs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.cached_data[idx] is None:
            graph = self.graphs[idx]
            # Precompute and cache graph properties
            self.cached_data[idx] = {
                'num_nodes': graph.x.size(0),
                'num_edges': graph.edge_index.size(1),
                'x_dtype': graph.x.dtype,
                'edge_attr_dtype': graph.edge_attr.dtype
            }
        return self.graphs[idx], self.X[idx], self.y[idx]


def collate_fn(batch):
    """Optimized collate function with GPU preloading"""
    graphs, X, y = zip(*batch)

    # Batch graphs efficiently
    batched_graph = Batch.from_data_list(graphs)

    # Convert to tensors with GPU preloading
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).pin_memory()
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).pin_memory()

    return batched_graph, X_tensor, y_tensor


class Objective:
    """Optimized objective class with resource monitoring"""

    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = amp.GradScaler()  # For mixed precision training

    def __call__(self, trial):
        # Log trial start
        trial_id = trial.number
        logger.info(f"Starting trial {trial_id}")

        # Sample hyperparameters
        gnn_hidden_dim = trial.suggest_categorical("gnn_hidden_dim", [128, 256, 512])
        transformer_dim = trial.suggest_categorical("transformer_dim", [128, 256, 512])

        # Use a fixed set of nhead options for all trials
        nhead_options = [4, 8, 16, 32]
        nhead = trial.suggest_categorical("nhead", nhead_options)

        # Check if transformer_dim is divisible by nhead
        if transformer_dim % nhead != 0:
            # Skip this trial as it's invalid
            logger.info(
                f"Skipping trial {trial_id}: transformer_dim({transformer_dim}) not divisible by nhead({nhead})")
            return float('inf')  # Return a high loss to indicate bad trial

        # Sample other parameters
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Get feature dimension
        _, X_sample, _ = next(iter(self.train_loader))
        exp_feature_dim = X_sample.size(1)

        # Initialize model with mixed precision support
        model = CrystalToxicityTransformer(
            exp_feature_dim=exp_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            transformer_dim=transformer_dim,
            nhead=nhead,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        ).to(DEVICE)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-6
        )
        criterion = torch.nn.HuberLoss(delta=0.5)

        # Training loop with progress bar
        best_val_loss = float('inf')
        progress_bar = tqdm(total=150, desc=f"Trial {trial_id}", position=trial_id)

        for epoch in range(150):  # Increased epochs for better convergence
            # Training phase
            model.train()
            train_loss = 0.0
            for batched_graph, X_batch, y_batch in self.train_loader:
                # Move data to GPU with async transfer
                batched_graph = batched_graph.to(DEVICE, non_blocking=True)
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                # Mixed precision training with updated API
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(batched_graph, X_batch)
                    loss = criterion(preds, y_batch)

                # Scale loss and backprop
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item() * len(y_batch)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batched_graph, X_batch, y_batch in self.val_loader:
                    batched_graph = batched_graph.to(DEVICE, non_blocking=True)
                    X_batch = X_batch.to(DEVICE, non_blocking=True)
                    y_batch = y_batch.to(DEVICE, non_blocking=True)

                    # Mixed precision validation with updated API
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        preds = model(batched_graph, X_batch)
                        loss = criterion(preds, y_batch)

                    val_loss += loss.item() * len(y_batch)

            # Calculate average losses
            train_loss /= len(self.train_loader.dataset)
            val_loss /= len(self.val_loader.dataset)

            # Update scheduler
            scheduler.step(val_loss)

            # Report intermediate result
            trial.report(val_loss, epoch)

            # Update progress bar
            progress_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            progress_bar.update(1)

            # Pruning based on intermediate results
            if trial.should_prune():
                progress_bar.close()
                logger.info(f"Trial {trial_id} pruned at epoch {epoch}")
                raise optuna.TrialPruned()

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        progress_bar.close()
        logger.info(f"Trial {trial_id} completed with best val_loss: {best_val_loss:.4f}")
        return best_val_loss


def hyperparameter_tuning(graphs_train, X_train, y_train, graphs_val, X_val, y_val,
                          n_trials=1, timeout=14400):  # 4 hours timeout
    """Enhanced hyperparameter tuning with full resource utilization"""
    # Create datasets
    train_dataset = GraphDataset(graphs_train, X_train, y_train)
    val_dataset = GraphDataset(graphs_val, X_val, y_val)

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, persistent_workers=True
    )

    # Create objective
    objective_fn = Objective(train_loader, val_loader)

    # Define study with enhanced settings
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=20, reduction_factor=3
        )
    )

    # Run optimization with progress tracking
    with tqdm(total=n_trials, desc="Hyperparameter Optimization") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"best_val": f"{study.best_value:.4f}"})

        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            callbacks=[callback],
            n_jobs=1  # We'll handle parallelism at a higher level
        )

    # Generate optimization visualizations
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image("optimization_history.png")

        fig2 = plot_intermediate_values(study)
        fig2.write_image("intermediate_values.png")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    logger.info("=" * 50)
    logger.info("Hyperparameter Tuning Summary")
    logger.info("=" * 50)
    logger.info(f"Best trial value: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

    # Print top 5 trials
    logger.info("\nTop 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value)
    for i, trial in enumerate(sorted_trials[:5]):
        logger.info(f"#{i + 1}: Value={trial.value:.4f}, Params={trial.params}")

    return study

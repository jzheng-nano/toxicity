import os
import time
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import psutil
# Import modules
from data_preprocessing import load_csv_data, apply_feature_engineering, preprocess_features, download_structure_data
from graph_construction import structure_to_graph
from training_and_evaluation import train_full_pipeline
from visualization import generate_all_plots

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MATERIALS_FILE = 'materials_name.csv'
EXPERIMENT_FILE = 'data_cleaned1B.csv'

def main():
    """Main pipeline execution"""
    logger.info("Starting Crystal Toxicity Prediction Pipeline")
    start_time = time.time()
    try:
        # Step 1: Load and preprocess data
        logger.info("Loading and preprocessing data...")
        merged_df = load_csv_data(MATERIALS_FILE, EXPERIMENT_FILE)
        merged_df = apply_feature_engineering(merged_df)
        X_experimental, y, cat_processor = preprocess_features(merged_df)
        
        # Step 2: Download crystal structures
        logger.info("Downloading crystal structures...")
        unique_mp_ids = merged_df['mp_id'].unique()
        crystal_data_cache = download_structure_data(unique_mp_ids, graph_method='crystalnn')
        crystal_graphs = [crystal_data_cache['graphs'].get(mp_id) for mp_id in merged_df['mp_id']]
        
        # Step 3: Data splitting
        logger.info("Splitting data...")
        indices = np.arange(len(y))
        X_temp, X_test, graphs_temp, graphs_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X_experimental, crystal_graphs, y, indices, test_size=0.15, random_state=42
        )
        X_train, X_val, graphs_train, graphs_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_temp, graphs_temp, y_temp, idx_temp, test_size=0.15, random_state=42
        )
        
        # Step 4: Train ensemble model
        logger.info(f"Memory usage before training: {psutil.virtual_memory().percent}%")

        # Start GPU memory monitoring
        gpu_before = torch.cuda.memory_allocated()

        logger.info("Starting model training...")
        ensemble, results = train_full_pipeline(
            graphs_train, X_train, y_train,
            graphs_val, X_val, y_val,
            graphs_test, X_test, y_test,
            n_trials=200,  # Increased number of trials
            n_models=5
        )

        # Log resource usage
        gpu_after = torch.cuda.memory_allocated()
        logger.info(f"GPU memory used: {(gpu_after - gpu_before) / 1e9:.2f} GB")
        logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Peak CPU memory: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        # Step 5: Generate visualizations
        logger.info("Generating visualizations...")
        generate_all_plots(
            results['targets'], results['predictions'],
            output_dir="visualizations",
            model_name="ensemble_model"
        )
        
        # Save ensemble model
        torch.save(ensemble.state_dict(), "toxicity_ensemble_model.pth")
        logger.info("Ensemble model saved")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # For CUDA compatibility
    main()

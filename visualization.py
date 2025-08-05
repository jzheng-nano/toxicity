import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def plot_true_vs_predicted(y_true, y_pred, output_dir, model_name="model"):
    """Plot true vs predicted values"""
    plot_path = os.path.join(output_dir, f"{model_name}_true_vs_pred.png")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with density coloring
    sns.kdeplot(x=y_true, y=y_pred, fill=True, thresh=0, levels=100, cmap="viridis")
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"True vs Predicted plot saved at {plot_path}")

def plot_residuals(y_true, y_pred, output_dir, model_name="model"):
    """Plot prediction residuals"""
    residuals = y_true - y_pred
    plot_path = os.path.join(output_dir, f"{model_name}_residuals.png")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="blue", bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Prediction Residual Distribution')
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Residual plot saved at {plot_path}")

def plot_error_distribution(y_true, y_pred, output_dir, model_name="model"):
    """Plot absolute error distribution"""
    errors = np.abs(y_true - y_pred)
    plot_path = os.path.join(output_dir, f"{model_name}_error_dist.png")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=errors)
    plt.xlabel('Absolute Error')
    plt.title('Prediction Error Distribution')
    plt.grid(True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Error distribution plot saved at {plot_path}")

def generate_all_plots(y_true, y_pred, output_dir, model_name="model"):
    """Generate all evaluation plots"""
    os.makedirs(output_dir, exist_ok=True)
    plot_true_vs_predicted(y_true, y_pred, output_dir, model_name)
    plot_residuals(y_true, y_pred, output_dir, model_name)
    plot_error_distribution(y_true, y_pred, output_dir, model_name)
    logger.info(f"All visualizations saved in {output_dir}")
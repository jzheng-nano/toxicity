import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from graph_construction import structure_to_graph, create_dummy_graph
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Constants
MATERIALS_API_KEY = "JBkm9BAbwqy33jXXfIsRhz8C0R2v9gkI"
CRYSTAL_DATA_CACHE = "crystal_data_enhanced.pkl"
CONFIG_FILE = "model_config_enhanced.pkl"


def load_csv_data(materials_file, experiment_file):
    """
    Load CSV data from the materials and experimental data files.
    """
    if not os.path.exists(materials_file) or not os.path.exists(experiment_file):
        raise FileNotFoundError("One or more required files are missing.")

    logger.info("Loading CSV data...")
    materials_df = pd.read_csv(materials_file)
    exp_df = pd.read_csv(experiment_file)

    logger.info("Merging materials and experimental data...")
    merged_df = pd.merge(exp_df, materials_df, on='Materials', how='left')

    required_columns = [
        'mp_id', 'Cell_Viability', 'Metal_Valence_State', 'Shape',
        'Coatings', 'Cell_line_name', 'Assay', 'Size', 'Time', 'Concentration'
    ]
    for col in required_columns:
        if col not in merged_df.columns:
            raise ValueError(f"Missing required column: {col}")

    return merged_df


def apply_feature_engineering(df):
    """
    Apply feature engineering to the dataset, including log transformations and feature interactions.
    """
    logger.info("Applying feature engineering...")

    # Log transformations to avoid extremely small constants
    df['Log_Concentration'] = np.log10(df['Concentration'] + 1e-8)
    df['Log_Time'] = np.log10(df['Time'] + 1e-8)
    df['Log_Size'] = np.log10(df['Size'] + 1e-8)

    # Feature interactions
    df['Size_Time'] = df['Log_Size'] * df['Log_Time']
    df['Concentration_Time'] = df['Log_Concentration'] * df['Log_Time']
    df['Size_Concentration'] = df['Log_Size'] * df['Log_Concentration']
    df['Size_Concentration_Time'] = (
            df['Log_Size'] * df['Log_Concentration'] * df['Log_Time']
    )

    return df


def preprocess_features(df):
    """
    Preprocess categorical and numerical features, and return the combined feature matrix.
    """
    logger.info("Preprocessing features...")

    categorical_features = ['Shape', 'Coatings', 'Cell_line_name', 'Assay']
    numerical_features = [
        'Log_Size', 'Log_Time', 'Log_Concentration',
        'Size_Time', 'Concentration_Time',
        'Size_Concentration', 'Size_Concentration_Time'
    ]

    # Polynomial feature generation
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    num_poly = poly.fit_transform(df[numerical_features])
    num_poly_df = pd.DataFrame(num_poly, columns=poly.get_feature_names_out(numerical_features))

    # One-hot encode categorical features
    cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = cat_processor.fit_transform(df[categorical_features])

    # Combine numerical and categorical features
    X_experimental = np.hstack([num_poly_df.values, X_cat])
    y = df['Cell_Viability'].values

    logger.info("Saving preprocessing configuration...")
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump({
            'cat_processor': cat_processor,
            'poly': poly,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        }, f)

    return X_experimental, y, cat_processor


def download_structure_data(unique_mp_ids, graph_method):
    """
    Download and cache crystal structure data from the Materials Project API.

    Args:
        unique_mp_ids: List of material project IDs.
        graph_method: Method for graph construction ('crystalnn', 'voronoi', etc.).

    Returns:
        A dictionary containing structures and graphs for the materials.
    """
    # Check if cache exists
    if os.path.exists(CRYSTAL_DATA_CACHE):
        logger.info(f"Loading cached crystal data from {CRYSTAL_DATA_CACHE}...")
        with open(CRYSTAL_DATA_CACHE, 'rb') as f:
            return pickle.load(f)

    logger.info("Downloading crystal structure data...")
    crystal_data_cache = {'structures': {}, 'graphs': {}}
    failed_ids = []

    with MPRester(MATERIALS_API_KEY) as mpr:
        for mp_id in tqdm(unique_mp_ids, desc="Downloading structures"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    structure = mpr.get_structure_by_material_id(mp_id)
                    structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

                graph = structure_to_graph(structure, graph_method)
                graph.mp_id = mp_id  # Add unique identifier
                crystal_data_cache['structures'][mp_id] = structure
                crystal_data_cache['graphs'][mp_id] = graph
            except Exception as e:
                logger.warning(f"Failed to download {mp_id}: {str(e)}")
                failed_ids.append(mp_id)
                dummy = create_dummy_graph(mp_id)
                crystal_data_cache['graphs'][mp_id] = dummy

    # Save cache
    with open(CRYSTAL_DATA_CACHE, 'wb') as f:
        pickle.dump(crystal_data_cache, f)
        logger.info(f"Crystal data saved to cache: {CRYSTAL_DATA_CACHE}")

    if failed_ids:
        logger.warning(f"Failed to download structures for IDs: {', '.join(failed_ids)}")

    return crystal_data_cache
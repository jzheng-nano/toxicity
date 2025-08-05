import math
import torch
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from torch_geometric.data import Data
import logging
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

def create_dummy_graph(mp_id="dummy"):
    """Create placeholder graph for missing structures"""
    dummy_features = torch.tensor([[1.0] + [0.0] * 15])
    return Data(
        x=dummy_features,
        edge_index=torch.tensor([[0, 0]], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor([[0.0]], dtype=torch.float),
        mp_id=mp_id,
        batch=torch.tensor([0], dtype=torch.long)
    )

def get_atomic_features(species):
    """Generate atomic features from periodic table properties"""
    try:
        element = Element(species.symbol)
        features = [
            element.Z, element.X, element.atomic_mass,
            element.group, element.row, element.atomic_radius,
            element.average_ionic_radius, element.average_anionic_radius,
            element.average_cationic_radius, element.ionization_energy,
            element.electron_affinity
        ]
        
        # Periodic features
        period_sin = math.sin(2 * math.pi * element.row / 9)
        period_cos = math.cos(2 * math.pi * element.row / 9)
        group_sin = math.sin(2 * math.pi * element.group / 18)
        group_cos = math.cos(2 * math.pi * element.group / 18)
        
        # Valence electrons
        nvalence = element.nvalence if hasattr(element, 'nvalence') else element.group
        features.append(nvalence)
        features.extend([period_sin, period_cos, group_sin, group_cos])
        
        return features
    except Exception as e:
        logger.warning(f"Failed to compute features for {species}: {e}")
        return [0.0] * 16

def structure_to_graph(structure, method='crystalnn'):
    """Convert crystal structure to graph representation"""
    try:
        # Node features
        atomic_features = [get_atomic_features(site.specie) for site in structure]
        x = torch.tensor(atomic_features, dtype=torch.float)
        
        # Normalize features
        for i in range(x.size(1)):
            col = x[:, i]
            if col.max() - col.min() > 1e-6:
                x[:, i] = (col - col.min()) / (col.max() - col.min())
            else:
                x[:, i] = 0.0
        
        # Construct edges
        edge_index, edge_attr = construct_edges(structure, method)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mp_id=getattr(structure, 'material_id', 'unknown'),
            batch=torch.zeros(x.size(0), dtype=torch.long)  # Temporary batch
        )
        return data
    except Exception as e:
        logger.warning(f"Failed to convert structure: {e}")
        return create_dummy_graph()

def construct_edges(structure, method):
    """Construct edges using specified method"""
    edge_index = []
    edge_attr = []
    cutoff = 8.0
    
    for _ in range(3):  # Retry with increasing cutoff
        try:
            if method == 'voronoi':
                vnn = VoronoiNN(tol=0.5, cutoff=cutoff)
                for i in range(len(structure)):
                    neighbors = vnn.get_nn_info(structure, i)
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        if i != j and [i, j] not in edge_index and [j, i] not in edge_index:
                            edge_index.append([i, j])
                            edge_attr.append([neighbor['weight']])
            elif method == 'crystalnn':
                cnn = CrystalNN()
                for i in range(len(structure)):
                    neighbors = cnn.get_nn_info(structure, i)
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        if i != j and [i, j] not in edge_index and [j, i] not in edge_index:
                            edge_index.append([i, j])
                            edge_attr.append([neighbor['weight']])
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if edge_index:
                break
            cutoff += 3.0
        except Exception as e:
            logger.warning(f"Edge construction failed: {str(e)}")
            cutoff += 3.0
    
    # Fallback if no edges
    if not edge_index:
        logger.warning("Using fully connected graph")
        num_nodes = len(structure)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                distance = structure[i].distance(structure[j])
                edge_index.append([i, j])
                edge_attr.append([distance])
    
    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_attr, dtype=torch.float)
    )
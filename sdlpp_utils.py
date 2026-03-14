"""
SDLPP Utilities
Helper functions for loading and using SDLPP results
"""

import numpy as np
import pickle
from pathlib import Path


def load_sdlpp_results(dataset_name='lost', results_dir='./results'):
    """Load SDLPP results from saved files.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset (default: 'lost')
    results_dir : str or Path
        Base results directory (default: './results')
    """
    results_dir = Path(results_dir) / dataset_name
    with open(results_dir / 'sdlpp_results.pkl', 'rb') as f:
        return pickle.load(f)


def load_projected_data(dataset_name='lost', results_dir='./results'):
    """Load projected data.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset (default: 'lost')
    """
    results_dir = Path(results_dir) / dataset_name
    return np.load(results_dir / 'lower_data.npy')


def load_projection_matrix(dataset_name='lost', results_dir='./results'):
    """Load projection matrix.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset (default: 'lost')
    """
    results_dir = Path(results_dir) / dataset_name
    return np.load(results_dir / 'projection_matrix.npy')


def load_from_npz(dataset_name='lost', results_dir='./results'):
    """Load results from NPZ file.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset (default: 'lost')
    """
    results_dir = Path(results_dir) / dataset_name
    npz = np.load(results_dir / 'sdlpp_results.npz')
    return npz['lower_data'], npz['P']


def project_new_data(X_new, P, scaler=None):
    """Project new data using learned projection matrix.
    
    Parameters:
    -----------
    X_new : ndarray
        New data (n_samples, n_features)
    P : ndarray
        Projection matrix (n_features, target_d)
    scaler : StandardScaler, optional
        Fitted scaler for normalization
        
    Returns:
    --------
    X_proj : ndarray
        Projected data (n_samples, target_d)
    """
    if scaler is not None:
        X_new = scaler.transform(X_new)
    return X_new @ P

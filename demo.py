"""SDLPP Demo - Dimensionality Reduction with Partial Labels"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import pickle
import argparse
from sdlpp import sdlpp


def save_results(lower_data, P, Y_history, para, scaler, labels, output_dir, formats):
    """
    Save results in multiple formats.
    
    Parameters:
    -----------
    labels : dict
        Dictionary containing 'target' and 'partial_target'
    formats : list
        List of formats: ['pkl', 'npy', 'npz', 'csv', 'mat']
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save as pickle (complete results)
    if 'pkl' in formats:
        results = {
            'lower_data': lower_data,
            'P': P,
            'Y_history': Y_history,
            'hyperparameters': para,
            'scaler': scaler,
            'labels': labels
        }
        pkl_file = output_dir / 'sdlpp_results.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
        saved_files.append(str(pkl_file))
    
    # Save as numpy arrays
    if 'npy' in formats:
        np.save(output_dir / 'lower_data.npy', lower_data)
        np.save(output_dir / 'projection_matrix.npy', P)
        if labels.get('target') is not None:
            np.save(output_dir / 'target.npy', labels['target'])
        if labels.get('partial_target') is not None:
            np.save(output_dir / 'partial_target.npy', labels['partial_target'])
        saved_files.extend([
            str(output_dir / 'lower_data.npy'),
            str(output_dir / 'projection_matrix.npy')
        ])
        if labels.get('target') is not None:
            saved_files.append(str(output_dir / 'target.npy'))
        if labels.get('partial_target') is not None:
            saved_files.append(str(output_dir / 'partial_target.npy'))
    
    # Save as single npz file
    if 'npz' in formats:
        npz_data = {
            'lower_data': lower_data,
            'P': P
        }
        if labels.get('target') is not None:
            npz_data['target'] = labels['target']
        if labels.get('partial_target') is not None:
            npz_data['partial_target'] = labels['partial_target']
        np.savez(output_dir / 'sdlpp_results.npz', **npz_data)
        saved_files.append(str(output_dir / 'sdlpp_results.npz'))
    
    # Save as CSV (projected data)
    if 'csv' in formats:
        np.savetxt(output_dir / 'lower_data.csv', lower_data, delimiter=',')
        np.savetxt(output_dir / 'projection_matrix.csv', P, delimiter=',')
        saved_files.extend([
            str(output_dir / 'lower_data.csv'),
            str(output_dir / 'projection_matrix.csv')
        ])
        if labels.get('target') is not None:
            np.savetxt(output_dir / 'target.csv', labels['target'], delimiter=',')
            saved_files.append(str(output_dir / 'target.csv'))
        if labels.get('partial_target') is not None:
            np.savetxt(output_dir / 'partial_target.csv', labels['partial_target'], delimiter=',')
            saved_files.append(str(output_dir / 'partial_target.csv'))
    
    # Save as MATLAB .mat file
    if 'mat' in formats:
        mat_data = {
            'lower_data': lower_data,
            'P': P
        }
        if labels.get('target') is not None:
            mat_data['target'] = labels['target']
        if labels.get('partial_target') is not None:
            mat_data['partial_target'] = labels['partial_target']
        mat_file = output_dir / 'sdlpp_results.mat'
        sio.savemat(str(mat_file), mat_data)
        saved_files.append(str(mat_file))
    
    return saved_files


def main(dataset_name='lost', data_path=None, save_formats=['pkl', 'npy'],
         use_sample_reliability=False, use_class_balance=False):
    """
    Run SDLPP on dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset (used for output directory)
    data_path : str
        Path to data file. If None, uses datasets/{dataset_name}.mat
    save_formats : list
        Formats to save results: ['pkl', 'npy', 'npz', 'csv', 'mat']
    """
    # Load data
    if data_path is None:
        data_path = Path(__file__).parent / 'datasets' / f'{dataset_name}.mat'
    else:
        data_path = Path(data_path)
    
    mat_data = sio.loadmat(str(data_path))
    data = mat_data['data']
    partial_target = mat_data['partial_target']
    
    # Load true labels if available
    target = mat_data.get('target', None)
    
    # Store labels
    labels = {
        'partial_target': partial_target.toarray() if hasattr(partial_target, 'toarray') else partial_target,
        'target': target.toarray() if hasattr(target, 'toarray') else target if target is not None else None
    }
    
    
    # Normalize data (zscore)
    scaler_mean = np.mean(data, axis=0)
    scaler_std = np.std(data, axis=0, ddof=1)
    scaler_std[scaler_std == 0] = 1
    data = (data - scaler_mean) / scaler_std

    class ZscoreScaler:
        def __init__(self, mean, std): self.mean_, self.scale_ = mean, std
        def transform(self, x): return (x - self.mean_) / self.scale_
    scaler = ZscoreScaler(scaler_mean, scaler_std)
    
    # Hyperparameters
    para = {
        'T': 100,
        'target_d': 13,
        'k': 8,
        'miu': 0.1,
        'thr': 0.95,
        'use_sample_reliability': use_sample_reliability,
        'use_class_balance': use_class_balance,
        'imbalance_alpha': 0.5,
        'imbalance_eps': 1e-8
    }
    
    # Run SDLPP
    lower_data, P, Y_history = sdlpp(data, partial_target, para)
    
    # Save results
    output_dir = Path(__file__).parent / 'results' / dataset_name
    saved_files = save_results(lower_data, P, Y_history, para, scaler, labels, output_dir, save_formats)
    
    return lower_data, P


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDLPP Dimensionality Reduction')
    parser.add_argument('--dataset', type=str, default='lost',
                        help='Dataset name (default: lost)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data file (optional)')
    parser.add_argument('--formats', type=str, nargs='+', 
                        default=['pkl', 'npy', 'csv'],
                        help='Save formats: pkl, npy, npz, csv, mat (default: pkl npy csv)')
    parser.add_argument('--sample-reliability', action='store_true',
                        help='Enable sample reliability weighting')
    parser.add_argument('--class-balance', action='store_true',
                        help='Enable class balance weighting')
    
    args = parser.parse_args()
    
    main(dataset_name=args.dataset, 
         data_path=args.data_path,
         save_formats=args.formats,
         use_sample_reliability=args.sample_reliability,
         use_class_balance=args.class_balance)

import logging
import cloudpickle
from scipy import stats
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
from wbfm.utils.external.utils_pymc import (build_baseline_priors, 
                                            build_sigmoid_term_pca, 
                                            build_curvature_term, 
                                            build_final_likelihood)
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
from pathlib import Path
import os
import matplotlib.pyplot as plt




def save_cv_results(neuron_name, df_cv_compare, cv_results_dict, output_dir, dataset_name='all'):
    """
    Save grouped cross-validation results (without full traces, which are too large).
    
    Saves:
    - df_cv_compare: model comparison DataFrame
    - cv_results_dict: fold-level results and statistics for each model
    - Visualization: LOO comparison plot
    
    Parameters
    ----------
    neuron_name : str
        Name of the neuron
    df_cv_compare : pd.DataFrame
        Output from az.compare() with LOO metrics
    cv_results_dict : dict
        Mapping of model names to cv_results
    output_dir : str
        Directory to save outputs
    dataset_name : str
        Dataset identifier for filename
    """
    if dataset_name == 'all':
        output_fname_base = f'{neuron_name}_cv'
    else:
        output_fname_base = f'{neuron_name}_{dataset_name}_cv'
    
    # Save the comparison DataFrame
    fname_compare = os.path.join(output_dir, f'{output_fname_base}_compare.h5')
    df_cv_compare.to_hdf(fname_compare, key='cv_compare')
    print(f"Saved CV comparison to {fname_compare}")
    
    # Save fold-level CV results as pickle (preserves structure better than HDF5)
    fname_results = os.path.join(output_dir, f'{output_fname_base}_results.pkl')
    with open(fname_results, 'wb') as buffer:
        cloudpickle.dump(cv_results_dict, buffer)
    print(f"Saved CV fold results to {fname_results}")
    
    # Save a summary of fold results to HDF5 for easy inspection
    fold_summary_data = {}
    for model_name, cv_results in cv_results_dict.items():
        # Check if this is grouped CV results (with fold_results) or temporal split results (without)
        if 'fold_results' in cv_results:
            # Grouped CV format: has fold_results list
            fold_df = pd.DataFrame(cv_results['fold_results'])
            # Drop the large test_ll_samples array for summary
            fold_summary = fold_df[['fold', 'group_id', 'test_ll', 'train_ll', 'test_size', 'train_size']].copy()
            fold_summary['model'] = model_name
            fold_summary_data[model_name] = fold_summary
        else:
            # Temporal split format: single train/test split per model
            fold_summary = pd.DataFrame({
                'fold': [0],
                'group_id': ['temporal_split'],
                'test_ll': [cv_results['test_ll']],
                'train_ll': [cv_results['train_ll']],
                'test_size': [cv_results['test_size']],
                'train_size': [cv_results['train_size']],
                'model': [model_name],
            })
            fold_summary_data[model_name] = fold_summary
    
    if fold_summary_data:
        fold_summary_full = pd.concat(fold_summary_data.values(), ignore_index=True)
        fname_summary = os.path.join(output_dir, f'{output_fname_base}_fold_summary.h5')
        fold_summary_full.to_hdf(fname_summary, key='fold_summary', mode='w')
        print(f"Saved fold summary to {fname_summary}")
    
    # Save comparison plot
    try:
        az.plot_compare(df_cv_compare, insample_dev=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{output_fname_base}_comparison.png'), dpi=150)
        plt.close()
        print(f"Saved comparison plot")
    except Exception as e:
        print(f"Warning: could not save comparison plot: {e}")
    
    print(f"Saved all CV results for {neuron_name} in {output_dir}")



def leave_one_trial_out_cv_from_posterior(Xy, all_traces, neuron_name):
    """
    Do approximate leave-one-trial-out cross-validation by setting the trial-level parameters to their prior, but not refitting the population-level parameters.
    """

    ## Get raw data
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
    curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
    residual_mode='pca_global'
    dataset_name = "all"

    # First pack into a single dataframe to drop nan, then unpack
    df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                            curvature_terms=curvature_terms_to_use, residual_mode=residual_mode)

    pca_modes = df_model[['x_pca0', 'x_pca1']].values
    y = df_model['y'].values
    curvature = df_model[curvature_terms_to_use].values

    ## Rebuild traces, substituting per-trial information with population information

    dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
    coords = {'dataset_name': dataset_name_values}
    dims = 'dataset_name'
    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

    idata = all_traces[neuron_name]

    # 1. Re-instantiate the exact model structure
    with pm.Model(coords=coords) as marginal_model:
        # Use pm.Data to pass in the original inputs
        # 'y' must be the original 50k length vector
        # 'pca_modes' and 'curvature' must be the original inputs
        
        intercept, sigma = build_baseline_priors()
        sigmoid_term, _ = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term = build_curvature_term(curvature, 
                                            curvature_terms_to_use=curvature_terms_to_use, 
                                            **dim_opt)

        mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
        
        # This creates the 'y' log_likelihood group ArviZ needs
        likelihood = build_final_likelihood(mu, sigma, y)

    # 2. Prepare the "Prior-Swapped" InferenceData
    trial_params = ["zscore_pca0_amplitude", "zscore_pca1_amplitude", "zscore_intercept"]
    idata_marginal = idata.copy()

    # Remove the existing likelihood so compute_log_likelihood has a clean slate
    if "log_likelihood" in idata_marginal:
        del idata_marginal.log_likelihood
        
    for param in trial_params:
        if param in idata_marginal.posterior:
            # Get shape: (chain, draw, 36)
            post_shape = idata_marginal.posterior[param].shape
            # Replace the 'spiky' posterior with standard normal noise (the prior)
            idata_marginal.posterior[param] = (("chain", "draw", "trial_dim"), 
                                            np.random.normal(0, 1, size=post_shape))
    
    # 3. Compute the Pointwise Log-Likelihood
    with marginal_model:
        # This runs the 'mu' math using posterior fixed effects + random prior z-scores
        pm.compute_log_likelihood(idata_marginal)#, **opt)

    # 1. Get the pointwise log-likelihood from the marginalized idata
    # Shape is (chain, draw, time_dim)
    pointwise_ll = idata_marginal.log_likelihood["y"]

    # 2. Add the trial mapping as a coordinate to the xarray object
    # This 'labels' every one of the 50k points with its trial ID
    # Example: trial_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2...]
    idx = Xy[neuron_name].dropna().index
    dataset_vector = Xy.loc[idx, 'dataset_name'].reset_index(drop=True)
    trial_mapping = xr.DataArray(dataset_vector, dims=['y_dim_0'])

    pointwise_ll = pointwise_ll.assign_coords(trial_id=("y_dim_0", np.asarray(trial_mapping)))

    # 3. Use xarray's vectorized groupby to sum everything into 36 buckets
    # This is the 'Leave-One-Trial-Out' summation step
    trial_ll = pointwise_ll.groupby("trial_id").sum(dim="y_dim_0")

    # 4. Rebuild the InferenceData for ArviZ
    # The trial_ll now has shape (chain, draw, 36)
    loto_idata = az.InferenceData(
        posterior=idata_marginal.posterior,  # Re-attach the posterior samples
        log_likelihood=xr.Dataset({"y": trial_ll.rename({"trial_id": "trial"})}),
    )

    loto_result = az.loo(loto_idata, pointwise=True)

    return loto_result


def compute_curvature_term_numpy(curvature, eigenworm1_coefficient, eigenworm2_coefficient,
                                  additional_coefficients=None, curvature_terms_to_use=None):
    """
    Compute curvature term from numpy arrays (matching build_curvature_term logic).

    Parameters
    ----------
    curvature : ndarray, shape (n, n_terms)
        Curvature features
    eigenworm1_coefficient : ndarray or scalar
        Coefficient for eigenworm 1
    eigenworm2_coefficient : ndarray or scalar
        Coefficient for eigenworm 2
    additional_coefficients : dict, optional
        Dict mapping coefficient names to their values for additional terms
    curvature_terms_to_use : list, optional
        List of curvature term names

    Returns
    -------
    curvature_term : ndarray
        Computed curvature term
    """
    curvature_term = eigenworm1_coefficient * curvature[:, 0] + eigenworm2_coefficient * curvature[:, 1]

    if additional_coefficients is not None and len(additional_coefficients) > 0:
        for i, coef_val in enumerate(additional_coefficients.values()):
            curvature_term = curvature_term + coef_val * curvature[:, i+2]

    return curvature_term


def compute_studentt_logp(mu, sigma, y, nu=5):
    """Compute log-likelihood under StudentT distribution using scipy."""
    return np.sum(stats.t.logpdf(y, df=nu, loc=mu, scale=sigma))


def compute_sigmoid_term_pca_numpy(x_pca_modes, pca0_amplitude, pca1_amplitude, inflection_point):
    """
    Compute sigmoid term from numpy arrays (matching build_sigmoid_term_pca logic).

    Parameters
    ----------
    x_pca_modes : ndarray, shape (n, 2)
        PCA modes
    pca0_amplitude : ndarray or scalar
        PCA0 amplitude value(s)
    pca1_amplitude : ndarray or scalar
        PCA1 amplitude value(s)
    inflection_point : scalar
        Inflection point of sigmoid

    Returns
    -------
    sigmoid_term : ndarray
        Computed sigmoid term
    """
    pca_term = pca0_amplitude * x_pca_modes[:, 0] + pca1_amplitude * x_pca_modes[:, 1]
    sigmoid_term = 1.0 / (1.0 + np.exp(-(pca_term - inflection_point)))
    return sigmoid_term


def grouped_cv_refitting(Xy, neuron_name, dataset_name='all', residual_mode='pca_global',
                         use_additional_eigenworms=True, group_by='dataset_name', n_splits=6, 
                         model_type='hierarchical_pca', DEBUG=False):
    """
    Perform grouped cross-validation by refitting the model for each fold.
    
    Uses LeaveOneGroupOut to split data by trial/group, trains on all other groups,
    and evaluates on the held-out group.
    
    Parameters
    ----------
    Xy : pd.DataFrame
        Full dataset
    neuron_name : str
        Name of the neuron to model
    dataset_name : str
        Which dataset(s) to use ('all', specific dataset name, etc.)
    residual_mode : str
        Residual/preprocessing mode for data
    use_additional_eigenworms : bool
        Whether to include eigenworms 2 and 3
    group_by : str
        Column name to use for grouping (default: 'dataset_name')
    n_splits : int
        Number of CV folds (default: 6)
    model_type : str
        Which model to fit: 'null', 'nonhierarchical', or 'hierarchical_pca' (default)
    DEBUG : bool
        If True, runs with fewer samples for testing
    
    Returns
    -------
    cv_results : dict
        Dictionary containing:
        - 'fold_results': list of results per fold
        - 'test_scores': array of test log-likelihoods per fold
        - 'train_scores': array of training log-likelihoods per fold
        - 'group_ids': array of group IDs that were left out
        - 'neuron_name': neuron being analyzed
    """
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
    if use_additional_eigenworms:
        curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
    
    # Get data for this neuron
    df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                               curvature_terms=curvature_terms_to_use, 
                                               residual_mode=residual_mode)
    
    if df_model.shape[0] == 0:
        print(f"No valid data for {neuron_name}")
        return None
    
    # Get group labels
    if group_by not in df_model.columns:
        print(f"Group column '{group_by}' not found in dataframe")
        return None
    
    groups = df_model[group_by].values
    X_pca = df_model[['x_pca0', 'x_pca1']].values
    X_curvature = df_model[curvature_terms_to_use].values
    y = df_model['y'].values
    
    # Setup factorization for hierarchical model
    dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
    coords = {'dataset_name': dataset_name_values}
    dims = 'dataset_name'
    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)
    
    # Initialize cross-validator
    cv = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    test_scores = []
    train_scores = []
    group_ids = []
    
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(y, groups=groups)), total=n_splits,
                                                 desc=f"CV for {neuron_name} ({model_type})"):
        
        # Split data
        X_pca_train, X_pca_test = X_pca[train_idx], X_pca[test_idx]
        X_curv_train, X_curv_test = X_curvature[train_idx], X_curvature[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        dataset_idx_train = dataset_name_idx[train_idx]
        dataset_idx_test = dataset_name_idx[test_idx]
        
        # Recompute dataset indices for training fold (starting from 0)
        dataset_idx_train_fold, dataset_names_train = pd.Series(dataset_idx_train).factorize()
        coords_fold = {'dataset_name': dataset_names_train}
        dim_opt_fold = dict(dims='dataset_name', dataset_name_idx=dataset_idx_train_fold)
        
        # Build and fit model on training data
        with pm.Model(coords=coords_fold) as cv_model:
            # Use pm.Data for inputs so we can swap them later
            X_pca_data = pm.Data('X_pca_data', X_pca_train, mutable=True)
            X_curv_data = pm.Data('X_curv_data', X_curv_train, mutable=True)
            y_data = pm.Data('y_data', y_train, mutable=True)
            
            intercept, sigma = build_baseline_priors(**dim_opt_fold)
            
            if model_type == 'null':
                # Just intercept, no other terms
                mu_train = pm.Deterministic('mu_train', intercept)
            elif model_type == 'nonhierarchical':
                # Curvature but no sigmoid
                curvature_term = build_curvature_term(X_curv_data, 
                                                     curvature_terms_to_use=curvature_terms_to_use,
                                                     **dim_opt_fold)
                mu_train = pm.Deterministic('mu_train', intercept + curvature_term)
            elif model_type == 'hierarchical_pca':
                # Sigmoid times curvature (full model)
                sigmoid_term, _ = build_sigmoid_term_pca(X_pca_data, **dim_opt_fold)
                curvature_term = build_curvature_term(X_curv_data, 
                                                     curvature_terms_to_use=curvature_terms_to_use,
                                                     **dim_opt_fold)
                mu_train = pm.Deterministic('mu_train', intercept + sigmoid_term * curvature_term)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            likelihood_train = build_final_likelihood(mu_train, sigma, y_data)
        
        # Sample from training model
        with cv_model:
            opt = dict(draws=1000, tune=1000, random_seed=42, target_accept=0.96)
            if DEBUG:
                opt['draws'] = 10
                opt['tune'] = 10
            
            trace = pm.sample(**opt, chains=4, return_inferencedata=True, 
                            idata_kwargs={"log_likelihood": True}, progressbar=False)
        
        # Get training log-likelihood
        train_ll = trace.log_likelihood["y"].sum(dim="y_dim_0").mean().values
        train_scores.append(train_ll)
        
        # Evaluate on test data by manually computing from posterior samples
        print("  Evaluating on test data...")
        posterior_samples = trace.posterior
        test_lls = []
        
        # Map test dataset indices to the fold's coordinate system
        dataset_idx_test_fold = pd.Categorical(dataset_idx_test, categories=dataset_names_train, ordered=True).codes
        
        for chain_idx in range(posterior_samples.sizes['chain']):
            for draw_idx in range(posterior_samples.sizes['draw']):
                # Extract intercept
                intercept_sample = posterior_samples['intercept'].isel(chain=chain_idx, draw=draw_idx).values
                if len(intercept_sample.shape) > 0:  # Per-dataset
                    intercept_sample = intercept_sample[dataset_idx_test_fold]
                
                # Extract sigma
                sigma_sample = posterior_samples['sigma'].isel(chain=chain_idx, draw=draw_idx).values
                if len(sigma_sample.shape) > 0:  # Per-dataset
                    sigma_sample = sigma_sample[dataset_idx_test_fold]
                
                if model_type == 'null':
                    # No additional terms; just intercept
                    mu_test = intercept_sample
                elif model_type == 'nonhierarchical':
                    # Curvature term but no sigmoid
                    eigenworm1_coef = posterior_samples['eigenworm1_coefficient'].isel(chain=chain_idx, draw=draw_idx).values
                    eigenworm2_coef = posterior_samples['eigenworm2_coefficient'].isel(chain=chain_idx, draw=draw_idx).values
                    
                    if len(eigenworm1_coef.shape) > 0:  # Per-dataset
                        eigenworm1_coef = eigenworm1_coef[dataset_idx_test_fold]
                        eigenworm2_coef = eigenworm2_coef[dataset_idx_test_fold]
                    
                    # Collect additional coefficients
                    additional_coefs = {}
                    for i, col_name in enumerate(curvature_terms_to_use[2:]):
                        if col_name.startswith('eigenworm'):
                            coef_name = f'eigenworm{int(col_name[-1])+1}_coefficient'
                        else:
                            coef_name = f'{col_name}_coefficient'
                        
                        if coef_name in posterior_samples:
                            coef = posterior_samples[coef_name].isel(chain=chain_idx, draw=draw_idx).values
                            additional_coefs[coef_name] = coef
                    
                    curvature_term = compute_curvature_term_numpy(X_curv_test, eigenworm1_coef, eigenworm2_coef, 
                                                                 additional_coefs, curvature_terms_to_use)
                    mu_test = intercept_sample + curvature_term
                elif model_type == 'hierarchical_pca':
                    # Sigmoid times curvature (full model)
                    # Extract PCA amplitudes and compute sigmoid term using helper function
                    inflection_point = posterior_samples['inflection_point'].isel(chain=chain_idx, draw=draw_idx).values
                    if 'pca0_amplitude' in posterior_samples:
                        # Hierarchical per-dataset amplitudes
                        pca0_amp = posterior_samples['pca0_amplitude'].isel(chain=chain_idx, draw=draw_idx).values[dataset_idx_test_fold]
                        pca1_amp = posterior_samples['pca1_amplitude'].isel(chain=chain_idx, draw=draw_idx).values[dataset_idx_test_fold]
                    else:
                        # Scalar amplitudes
                        pca_amp = posterior_samples['pca_amplitude'].isel(chain=chain_idx, draw=draw_idx).values
                        pca0_amp = pca_amp[0]
                        pca1_amp = pca_amp[1]
                    
                    sigmoid_term = compute_sigmoid_term_pca_numpy(X_pca_test, pca0_amp, pca1_amp, inflection_point)
                    
                    # Extract curvature coefficients and compute term using helper function
                    eigenworm1_coef = posterior_samples['eigenworm1_coefficient'].isel(chain=chain_idx, draw=draw_idx).values
                    eigenworm2_coef = posterior_samples['eigenworm2_coefficient'].isel(chain=chain_idx, draw=draw_idx).values
                    
                    if len(eigenworm1_coef.shape) > 0:  # Per-dataset
                        eigenworm1_coef = eigenworm1_coef[dataset_idx_test_fold]
                        eigenworm2_coef = eigenworm2_coef[dataset_idx_test_fold]
                    
                    # Collect additional coefficients
                    additional_coefs = {}
                    for i, col_name in enumerate(curvature_terms_to_use[2:]):
                        if col_name.startswith('eigenworm'):
                            coef_name = f'eigenworm{int(col_name[-1])+1}_coefficient'
                        else:
                            coef_name = f'{col_name}_coefficient'
                        
                        if coef_name in posterior_samples:
                            coef = posterior_samples[coef_name].isel(chain=chain_idx, draw=draw_idx).values
                            additional_coefs[coef_name] = coef
                    
                    curvature_term = compute_curvature_term_numpy(X_curv_test, eigenworm1_coef, eigenworm2_coef, 
                                                                 additional_coefs, curvature_terms_to_use)
                    
                    # Compute mu and log-likelihood
                    mu_test = intercept_sample + sigmoid_term * curvature_term
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                test_ll = compute_studentt_logp(mu_test, sigma_sample, y_test, nu=100)
                test_lls.append(test_ll)
        
        # Convert per-sample test log-likelihoods into array with shape (n_chain, n_draw)
        test_lls = np.asarray(test_lls)
        n_chain = posterior_samples.sizes['chain']
        n_draw = posterior_samples.sizes['draw']
        # If number of samples matches, reshape; otherwise leave as 1D
        if test_lls.size == n_chain * n_draw:
            test_lls_samples = test_lls.reshape(n_chain, n_draw)
        else:
            test_lls_samples = test_lls
        
        # Mean across posterior samples for reporting
        test_ll_mean = np.mean(test_lls_samples) if test_lls_samples.size else np.nan
        test_scores.append(test_ll_mean)
        group_ids.append(groups[test_idx[0]])
        
        fold_results.append({
            'fold': fold_idx,
            'group_id': groups[test_idx[0]],
            'test_ll': test_ll_mean,
            'train_ll': train_ll,
            'test_size': len(test_idx),
            'train_size': len(train_idx),
            'test_ll_samples': test_lls_samples,
        })
        
        print(f"  Train LL: {train_ll:.4f}, Test LL: {test_ll_mean:.4f}")
    
    cv_results = {
        'fold_results': fold_results,
        'test_scores': np.array(test_scores),
        'train_scores': np.array(train_scores),
        'group_ids': np.array(group_ids),
        'neuron_name': neuron_name,
        'mean_test_ll': np.mean(test_scores),
        'std_test_ll': np.std(test_scores),
    }
    
    return cv_results


def temporal_train_test_split(Xy, neuron_name, dataset_name='all', residual_mode='pca_global',
                              use_additional_eigenworms=True, train_frac=2/3, 
                              model_type='hierarchical_pca', DEBUG=False):
    """
    Perform temporal train-test split by holding out the middle third of the time series.
    
    Splits data temporally: trains on first and last thirds, tests on middle third.
    This allows learning trial-specific parameters for all trials.
    
    Parameters
    ----------
    Xy : pd.DataFrame
        Full dataset
    neuron_name : str
        Name of the neuron to model
    dataset_name : str
        Which dataset(s) to use ('all', specific dataset name, etc.)
    residual_mode : str
        Residual/preprocessing mode for data
    use_additional_eigenworms : bool
        Whether to include eigenworms 2 and 3
    train_frac : float
        Fraction of data to use for training (default: 2/3)
    model_type : str
        Which model to fit: 'null', 'nonhierarchical', or 'hierarchical_pca' (default)
    DEBUG : bool
        If True, runs with fewer samples for testing
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'test_ll': test log-likelihood
        - 'train_ll': training log-likelihood
        - 'test_size': number of test samples
        - 'train_size': number of training samples
        - 'neuron_name': neuron being analyzed
        - 'trace': posterior trace object
        - 'model': PyMC model object
    """
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
    if use_additional_eigenworms:
        curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
    
    # Get data for this neuron
    df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                               curvature_terms=curvature_terms_to_use, 
                                               residual_mode=residual_mode)
    
    if df_model.shape[0] == 0:
        print(f"No valid data for {neuron_name}")
        return None
    
    # Sort by time to ensure temporal ordering
    if 'time' in df_model.columns:
        df_model = df_model.sort_values(['dataset_name', 'time'])
    
    # Make sure that the index of df_model is contiguous
    df_model = df_model.reset_index()
    
    # Create temporal split: middle third for test, outer thirds for train
    # Do this per trial (dataset_name) so each trial contributes balanced data
    def split_trial_temporal(group, train_frac=2/3):
        """Split a trial's indices into train/test temporally."""
        n = len(group)
        test_size = int(n * (1 - train_frac))
        train_size_each_side = (n - test_size) // 2
        
        indices = group.index.values
        train = np.concatenate([
            indices[:train_size_each_side],
            indices[train_size_each_side + test_size:]
        ])
        test = indices[train_size_each_side:train_size_each_side + test_size]
        return train, test
    
    # Apply split to each trial and collect indices
    train_idx_list = []
    test_idx_list = []
    for _, group in df_model.groupby('dataset_name'):
        train, test = split_trial_temporal(group, train_frac)
        train_idx_list.append(train)
        test_idx_list.append(test)
    
    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)
    
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # Extract features
    X_pca = df_model[['x_pca0', 'x_pca1']].values
    X_curvature = df_model[curvature_terms_to_use].values
    y = df_model['y'].values
    
    # Setup factorization for hierarchical model
    dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
    coords = {'dataset_name': dataset_name_values}
    dims = 'dataset_name'
    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)
    
    # Split data
    X_pca_train, X_pca_test = X_pca[train_idx], X_pca[test_idx]
    X_curv_train, X_curv_test = X_curvature[train_idx], X_curvature[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    dataset_idx_train = dataset_name_idx[train_idx]
    dataset_idx_test = dataset_name_idx[test_idx]
    
    # Build and fit model
    with pm.Model(coords=coords) as model:
        # Use pm.Data for inputs so we can swap them later
        X_pca_data = pm.Data('X_pca_data', X_pca_train, mutable=True)
        X_curv_data = pm.Data('X_curv_data', X_curv_train, mutable=True)
        y_data = pm.Data('y_data', y_train, mutable=True)
        dataset_idx_data = pm.Data('dataset_idx', dataset_idx_train, mutable=True)
        
        # Update dim_opt to use the mutable dataset index
        dim_opt_mutable = dict(dims=dims, dataset_name_idx=dataset_idx_data)
        
        intercept, sigma = build_baseline_priors(**dim_opt_mutable)
        
        if model_type == 'null':
            # Just intercept, no other terms
            mu = pm.Deterministic('mu', intercept)
        elif model_type == 'nonhierarchical':
            # Curvature but no sigmoid
            curvature_term = build_curvature_term(X_curv_data, 
                                                 curvature_terms_to_use=curvature_terms_to_use,
                                                 **dim_opt_mutable)
            mu = pm.Deterministic('mu', intercept + curvature_term)
        elif model_type == 'hierarchical_pca':
            # Sigmoid times curvature (full model)
            sigmoid_term, _ = build_sigmoid_term_pca(X_pca_data, **dim_opt_mutable)
            curvature_term = build_curvature_term(X_curv_data, 
                                                 curvature_terms_to_use=curvature_terms_to_use,
                                                 **dim_opt_mutable)
            mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        likelihood = build_final_likelihood(mu, sigma, y_data)
    
    # Sample from model
    with model:
        opt = dict(draws=1000, tune=1000, random_seed=42, target_accept=0.96)
        if DEBUG:
            opt['draws'] = 10
            opt['tune'] = 10
        
        trace = pm.sample(**opt, chains=4, return_inferencedata=True, 
                        idata_kwargs={"log_likelihood": True}, progressbar=True)
    
    # Get training log-likelihood
    train_ll = trace.log_likelihood["y"].sum(dim="y_dim_0").mean().values
    print(f"Training log-likelihood: {train_ll:.4f}")
    
    # Evaluate on test data by swapping in test data
    print("Evaluating on test data...")
    with model:
        pm.set_data({
            'X_pca_data': X_pca_test,
            'X_curv_data': X_curv_test,
            'y_data': y_test,
            'dataset_idx': dataset_idx_test
        })
        
        # Remove the training log_likelihood group to avoid conflict when computing test log_likelihood
        del trace.log_likelihood
        
        # Compute test log-likelihood
        test_ll_idata = pm.compute_log_likelihood(trace, progressbar=True)
        test_ll = test_ll_idata.log_likelihood["y"].sum(dim="y_dim_0").mean().values
    
    print(f"Test log-likelihood: {test_ll:.4f}")
    
    results = {
        'test_ll': float(test_ll),
        'train_ll': float(train_ll),
        'test_size': len(test_idx),
        'train_size': len(train_idx),
        'neuron_name': neuron_name,
        'model_type': model_type,
        'trace': trace,
        'model': model,
        'train_idx': train_idx,
        'test_idx': test_idx,
    }
    
    return results


def cv_results_to_arviz(cv_results):
    """
    Convert grouped cross-validation results to an ArviZ-compatible InferenceData object.

    The resulting `InferenceData` contains a `posterior` group (fold-level summary
    treated as chain/draw singletons) and a `log_likelihood` group with pointwise
    fold log-likelihoods so that `az.loo()` and `az.plot_khat()` can be used.

    Parameters
    ----------
    cv_results : dict
        Output from `grouped_cv_refitting`

    Returns
    -------
    idata : arviz.InferenceData
        InferenceData object with `posterior` and `log_likelihood` groups.
    """
    fold_results = cv_results['fold_results']
    fold_df = pd.DataFrame(fold_results)

    # Posterior-like dataset: fold-wise summary (treated as length-1 chain/draw)
    posterior_ds = xr.Dataset(
        {
            'test_ll': (['fold'], fold_df['test_ll'].values),
            'train_ll': (['fold'], fold_df['train_ll'].values),
        },
        coords={'fold': fold_df['fold'].values},
        attrs={
            'neuron_name': cv_results.get('neuron_name', ''),
            'mean_test_ll': float(cv_results.get('mean_test_ll', np.nan)),
            'std_test_ll': float(cv_results.get('std_test_ll', np.nan)),
        },
    )

    # Expand to have singleton chain/draw dims so arviz accepts it
    posterior_ds = posterior_ds.expand_dims(chain=[0], draw=[0])

    # Build log-likelihood group with both sum and mean-per-observation values
    # so users can choose normalization. We prefer to include per-posterior-sample
    # values when available so ArviZ/PSIS can operate (shape: chain x draw x folds).
    n_folds = len(fold_df)
    test_size = fold_df['test_size'].values

    # If `test_ll_samples` was recorded per fold, assemble into array
    if 'test_ll_samples' in fold_df.columns and not fold_df['test_ll_samples'].isnull().all():
        # Each entry should be an array of shape (n_chain, n_draw) or a flat array
        samples_list = []
        for v in fold_df['test_ll_samples'].values:
            arr = np.asarray(v)
            if arr.ndim == 1:
                # Single flat list of samples; attempt to reshape later
                samples_list.append(arr)
            elif arr.ndim == 2:
                # (n_chain, n_draw)
                samples_list.append(arr)
            else:
                # Unexpected dims, flatten
                samples_list.append(arr.reshape(-1))

        # Try to stack into shape (n_folds, n_chain, n_draw) if possible
        try:
            stacked = np.stack(samples_list, axis=-1)  # will be (n_chain, n_draw, n_folds) or (n_samples, n_folds)
        except Exception:
            # Fallback: convert to object array then to list-of-lists
            stacked = np.array([np.asarray(x) for x in samples_list], dtype=object)

        # Normalize shapes to (n_chain, n_draw, n_folds)
        if isinstance(stacked, np.ndarray) and stacked.dtype != object:
            if stacked.ndim == 3:
                # If stack produced (n_chain, n_draw, n_folds) or (n_samples, n_chain, n_draw)
                if stacked.shape[0] == n_folds:
                    # (n_folds, n_chain, n_draw) -> transpose
                    samples_array = np.transpose(stacked, (1, 2, 0))
                elif stacked.shape[-1] == n_folds:
                    samples_array = stacked
                else:
                    # Unexpected ordering; reshape conservatively
                    samples_array = stacked.reshape(stacked.shape[0], stacked.shape[1], n_folds)
            elif stacked.ndim == 2:
                # (n_samples, n_folds) -> treat as (1, n_samples, n_folds)
                samples_array = stacked.reshape(1, stacked.shape[0], stacked.shape[1])
            else:
                # Fallback to singleton
                samples_array = stacked.reshape(1, 1, n_folds)
        else:
            # Could not stack numerically; fallback to singleton summary
            y_sum = fold_df['test_ll'].values
            y_mean = np.array([s / (sz if sz > 0 else 1) for s, sz in zip(y_sum, test_size)])
            ll_sum_array = y_sum.reshape(1, 1, n_folds)
            ll_mean_array = y_mean.reshape(1, 1, n_folds)
            log_likelihood_ds = xr.Dataset(
                {
                    'y_sum': (['chain', 'draw', 'y_dim_0'], ll_sum_array),
                    'y_mean': (['chain', 'draw', 'y_dim_0'], ll_mean_array),
                    'test_size': (['y_dim_0'], test_size),
                },
                coords={'chain': [0], 'draw': [0], 'y_dim_0': np.arange(n_folds)},
            )
            idata = az.InferenceData(posterior=posterior_ds, log_likelihood=log_likelihood_ds)
            return idata

        # At this point samples_array should be numeric with shape (chain, draw, n_folds)
        samples_array = np.asarray(samples_array)

        # Compute per-fold sums (already sums over observations) and per-observation means
        ll_sum_array = samples_array
        # Broadcast test_size to divide across folds
        test_size_broadcast = test_size.reshape(1, 1, n_folds)
        ll_mean_array = ll_sum_array / (test_size_broadcast.astype(float) + (test_size_broadcast == 0))

        coords_ll = {
            'chain': np.arange(samples_array.shape[0]),
            'draw': np.arange(samples_array.shape[1]),
            'y_dim_0': np.arange(n_folds),
        }

        log_likelihood_ds = xr.Dataset(
            {
                'y_sum': (['chain', 'draw', 'y_dim_0'], ll_sum_array),
                'y_mean': (['chain', 'draw', 'y_dim_0'], ll_mean_array),
                'test_size': (['y_dim_0'], test_size),
            },
            coords=coords_ll,
        )
    else:
        # Fall back to singleton chain/draw as before
        test_size = fold_df['test_size'].values
        y_sum = fold_df['test_ll'].values  # currently stored as mean over posterior samples of sum-ll
        # per-observation mean (normalize by test size) — avoid division by zero
        y_mean = np.array([s / (sz if sz > 0 else 1) for s, sz in zip(y_sum, test_size)])

        ll_sum_array = y_sum.reshape(1, 1, n_folds)
        ll_mean_array = y_mean.reshape(1, 1, n_folds)

        log_likelihood_ds = xr.Dataset(
            {
                'y_sum': (['chain', 'draw', 'y_dim_0'], ll_sum_array),
                'y_mean': (['chain', 'draw', 'y_dim_0'], ll_mean_array),
                'test_size': (['y_dim_0'], test_size),
            },
            coords={'chain': [0], 'draw': [0], 'y_dim_0': np.arange(n_folds)},
        )

    idata = az.InferenceData(posterior=posterior_ds, log_likelihood=log_likelihood_ds)
    return idata


def compute_cv_loo(cv_results, normalize=True, pointwise=True):
    """
    Build an ArviZ `InferenceData` from CV results and compute LOO.

    Parameters
    ----------
    cv_results : dict
        Output from `grouped_cv_refitting`.
    normalize : bool
        If True, use per-observation mean log-likelihood (`y_mean`) for LOO so
        folds with different sizes are comparable. If False, use the raw sums
        (`y_sum`).
    pointwise : bool
        Passed to `az.loo()`.

    Returns
    -------
    idata, loo
        The constructed `InferenceData` and the computed `loo` object.
    """
    idata = cv_results_to_arviz(cv_results)

    # Select which variable in log_likelihood to use for LOO
    if normalize:
        # Move y_mean into the expected 'y' variable for arviz
        ll = xr.Dataset({'y': idata.log_likelihood['y_mean']})
    else:
        ll = xr.Dataset({'y': idata.log_likelihood['y_sum']})

    idata_for_loo = az.InferenceData(posterior=idata.posterior, log_likelihood=ll)
    loo = az.loo(idata_for_loo, pointwise=pointwise)
    return idata_for_loo, loo


def grouped_cv_compare(Xy, neuron_name, dataset_name='all', residual_mode='pca_global',
                       use_additional_eigenworms=True, models_to_compare=None, DEBUG=False):
    """
    Perform grouped cross-validation on multiple models and return az.compare-compatible output.

    Parameters
    ----------
    Xy : pd.DataFrame
        Full dataset
    neuron_name : str
        Name of the neuron to model
    dataset_name : str
        Which dataset(s) to use ('all', specific dataset name, etc.)
    residual_mode : str
        Residual/preprocessing mode for data
    use_additional_eigenworms : bool
        Whether to include eigenworms 2 and 3
    models_to_compare : list of str, optional
        Which models to compare. Default: ['null', 'nonhierarchical', 'hierarchical_pca']
    DEBUG : bool
        If True, runs with fewer samples for testing

    Returns
    -------
    df_cv_compare : pd.DataFrame
        az.compare()-compatible DataFrame with LOO and model comparison metrics
    cv_results_dict : dict
        Dictionary mapping model names to their cv_results
    """
    if models_to_compare is None:
        models_to_compare = ['null', 'nonhierarchical', 'hierarchical_pca']

    # Dictionary to store LOO results for each model
    loo_results = {}
    cv_results_dict = {}

    for model_name in models_to_compare:
        print(f"\n{'='*60}")
        print(f"Running grouped CV for {model_name} model on {neuron_name}")
        print(f"{'='*60}")

        # Run grouped CV for this model
        cv_results = grouped_cv_refitting(
            Xy, neuron_name,
            dataset_name=dataset_name,
            residual_mode=residual_mode,
            use_additional_eigenworms=use_additional_eigenworms,
            group_by='dataset_name',
            n_splits=6,
            model_type=model_name,
            DEBUG=DEBUG
        )

        if cv_results is None:
            print(f"Skipping {model_name} for {neuron_name} (no CV results)")
            continue

        cv_results_dict[model_name] = cv_results

        # Compute LOO from CV results
        idata, loo = compute_cv_loo(cv_results, normalize=True, pointwise=True)
        loo_results[model_name] = loo

        print(f"LOO for {model_name}: {loo.elpd_loo:.2f}")

    if not loo_results:
        print(f"No LOO results computed for {neuron_name}")
        return None, cv_results_dict

    # Use az.compare to produce a standardized comparison DataFrame
    df_cv_compare = az.compare(loo_results, ic='loo')

    return df_cv_compare, cv_results_dict


def temporal_split_compare(Xy, neuron_name, dataset_name='all', residual_mode='pca_global',
                                  use_additional_eigenworms=True, models_to_compare=None,
                                  train_frac=2/3, DEBUG=False):
    """
    Simplified comparison using only test set performance (no LOO).
    
    This is faster and more straightforward than the LOO-based comparison,
    and directly uses the held-out test set performance.
    
    Parameters
    ----------
    Same as temporal_split_compare
    
    Returns
    -------
    df_compare : pd.DataFrame
        Comparison DataFrame with test/train performance metrics
    results_dict : dict
        Dictionary mapping model names to their temporal split results
    """
    if models_to_compare is None:
        models_to_compare = ['null', 'nonhierarchical', 'hierarchical_pca']
    
    results_dict = {}
    comparison_data = []
    
    for model_name in models_to_compare:
        print(f"\n{'='*60}")
        print(f"Running temporal split for {model_name} model on {neuron_name}")
        print(f"{'='*60}")
        
        # Run temporal split for this model
        results = temporal_train_test_split(
            Xy, neuron_name,
            dataset_name=dataset_name,
            residual_mode=residual_mode,
            use_additional_eigenworms=use_additional_eigenworms,
            train_frac=train_frac,
            model_type=model_name,
            DEBUG=DEBUG
        )
        
        if results is None:
            print(f"Skipping {model_name} for {neuron_name} (no results)")
            continue
        
        results_dict[model_name] = results
        
        comparison_data.append({
            'model': model_name,
            'test_ll': results['test_ll'],
            'train_ll': results['train_ll'],
            'test_ll_per_obs': results['test_ll'] / results['test_size'],
            'train_ll_per_obs': results['train_ll'] / results['train_size'],
            'generalization_gap': (results['train_ll'] / results['train_size']) - 
                                 (results['test_ll'] / results['test_size']),
            'test_size': results['test_size'],
            'train_size': results['train_size'],
        })
        
        print(f"Test LL/obs for {model_name}: {results['test_ll'] / results['test_size']:.4f}")
    
    if not comparison_data:
        print(f"No results for {neuron_name}")
        return None, results_dict
    
    df_compare = pd.DataFrame(comparison_data).set_index('model')
    
    # Add ranking based on test performance
    df_compare['rank'] = df_compare['test_ll_per_obs'].rank(ascending=False).astype(int)
    df_compare = df_compare.sort_values('test_ll_per_obs', ascending=False)
    
    # Add relative performance vs best model
    best_test_ll = df_compare['test_ll_per_obs'].max()
    df_compare['test_ll_diff'] = df_compare['test_ll_per_obs'] - best_test_ll

    # Add names to match az.compare format
    df_compare.index.name = 'model'
    df_compare['elpd_loo'] = df_compare['test_ll_per_obs'] * df_compare['test_size']  # undo normalization for compatibility
    df_compare['p_loo'] = np.nan  # not computed here
    df_compare['loo_scale'] = 'test_ll_per_obs'
    
    return df_compare, results_dict


def main_cv_comparison(neuron_name=None, do_gfp=False, dataset_name='all', skip_if_exists=True, 
                       residual_mode='pca_global', use_additional_eigenworms=True, DEBUG=False):
    """
    Run grouped cross-validation model comparison for a neuron and save results.
    
    Similar to main() but uses grouped_cv_refitting on multiple models instead of 
    fitting each model once. Much faster and suitable for quick CV-based model selection.
    
    Parameters
    ----------
    neuron_name : str, optional
        Name of neuron to analyze. Defaults to 'VB02'
    do_gfp : bool
        Whether to use GFP data
    dataset_name : str
        'all', 'loop', or specific dataset name
    skip_if_exists : bool
        Skip if output already exists
    residual_mode : str
        Residual/preprocessing mode
    use_additional_eigenworms : bool
        Include eigenworms 2 and 3
    DEBUG : bool
        Run with fewer samples for testing
    """
    if DEBUG:
        skip_if_exists = False
        neuron_name = 'VB02'
    if neuron_name is None:
        neuron_name = 'VB02'

    print(f"Running grouped CV comparison for {neuron_name} with do_gfp={do_gfp} and residual_mode={residual_mode}")

    data_dir = get_hierarchical_modeling_dir(do_gfp)
    fname = os.path.join(data_dir, 'data.h5')
    if not os.path.exists(fname):
        logging.warning(f"Could not find data file {fname}, trying backup")
        fname = os.path.join(data_dir, 'data_backup.h5')
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Could not find data file {fname}")
    Xy = pd.read_hdf(fname)
    print(f"Loaded data from {fname}")

    if dataset_name == 'loop':
        # Loop over all datasets
        for dataset_name in Xy['dataset_name'].unique():
            print(f"Running {neuron_name} for {dataset_name}")
            if dataset_name == 'loop':
                continue
            main_cv_comparison(neuron_name, do_gfp=do_gfp, dataset_name=dataset_name, 
                             skip_if_exists=skip_if_exists, residual_mode=residual_mode,
                             use_additional_eigenworms=use_additional_eigenworms, DEBUG=DEBUG)
        return

    if dataset_name == 'all':
        output_dir = os.path.join(data_dir, 'output_cv')
    else:
        output_dir = os.path.join(data_dir, 'output_cv_single_dataset')
    if DEBUG:
        output_dir = f"{output_dir}_debug"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check if it already exists
    if skip_if_exists and os.path.exists(os.path.join(output_dir, f'{neuron_name}_cv_compare.h5')):
        print(f"Skipping {neuron_name} because CV results already exist")
        return

    # Run grouped CV comparison
    df_cv_compare, cv_results_dict = temporal_split_compare(
        Xy, neuron_name,
        dataset_name=dataset_name,
        residual_mode=residual_mode,
        use_additional_eigenworms=use_additional_eigenworms,
        models_to_compare=['null', 'nonhierarchical', 'hierarchical_pca'],
        DEBUG=DEBUG
    )

    if df_cv_compare is None:
        print(f"Skipping {neuron_name} because there is no valid data")
        return

    save_cv_results(neuron_name, df_cv_compare, cv_results_dict, output_dir, dataset_name)


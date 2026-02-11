import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import cloudpickle
from matplotlib import pyplot as plt
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron


def initialize_hierarchical_model_data(Xy, neuron_name, dataset_name='all', residual_mode='pca_global',
                                       use_additional_eigenworms=True, use_additional_behaviors=False, verbose=0):
    """
    Initialize data for hierarchical Bayesian models.
    
    Handles:
    - Loading PCA modes and setting up curvature terms
    - Factorizing dataset indices if using hierarchical structure
    - Creating model coordinates and dimension options
    
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
        Include eigenworms 2 and 3
    use_additional_behaviors : bool
        Include additional behavior terms (speed, self_collision, etc.)
    
    Returns
    -------
    dict with keys:
        - 'df_model': DataFrame with all model data
        - 'pca_modes': array of shape (n_timepoints, 2)
        - 'curvature_terms_to_use': list of curvature term names
        - 'coords': dict of PyMC coordinates
        - 'dims': dimension name ('dataset_name' or None)
        - 'dataset_name_idx': array of dataset indices or None
        - 'dim_opt': dict with dims and dataset_name_idx for unpacking into functions
    """
    # Build curvature terms list
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
    if use_additional_eigenworms:
        curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
    if use_additional_behaviors:
        curvature_terms_to_use.extend(['speed', 'self_collision'])
    
    # Load data for this neuron
    df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                               curvature_terms=curvature_terms_to_use, 
                                               residual_mode=residual_mode, verbose=verbose)
    
    # Extract PCA modes
    pca_modes = df_model[['x_pca0', 'x_pca1']].values
    
    # Initialize hierarchical structure based on dataset_name
    if dataset_name == 'all':
        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'
    else:
        coords = {}
        dims, dataset_name_idx = None, None
    
    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)
    
    return {
        'df_model': df_model,
        'pca_modes': pca_modes,
        'curvature_terms_to_use': curvature_terms_to_use,
        'coords': coords,
        'dims': dims,
        'dataset_name_idx': dataset_name_idx,
        'dim_opt': dim_opt,
    }


def fit_multiple_models(Xy, neuron_name, dataset_name='2022-11-23_worm8', residual_mode='pca_global',
                        sample_posterior=True, sample_large_variables=False, use_additional_behaviors=False,
                        use_additional_eigenworms=True,
                        dryrun=False, DEBUG=False) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Fit multiple models to the same data, to be used for model comparison

    Parameters
    ----------
    Xy
    neuron_name

    Returns
    -------

    """
    rng = 424242
    
    # Initialize model data using helper function
    try:
        model_data = initialize_hierarchical_model_data(
            Xy, neuron_name, 
            dataset_name=dataset_name,
            residual_mode=residual_mode,
            use_additional_eigenworms=use_additional_eigenworms,
            use_additional_behaviors=use_additional_behaviors
        )
    except KeyError as e:
        print(f"Skipping {neuron_name} because there is no valid data (KeyError: {e})")
        return None, None, None
    
    if model_data['df_model'].shape[0] == 0:
        print(f"Skipping {neuron_name} because there is no valid data (shape is 0)")
        return None, None, None
    
    # Extract relevant data
    pca_modes = model_data['pca_modes']
    curvature = model_data['df_model'][model_data['curvature_terms_to_use']].values
    y = model_data['df_model']['y'].values
    coords = model_data['coords']
    dim_opt = model_data['dim_opt']
    curvature_terms_to_use = model_data['curvature_terms_to_use']

    baseline_opt = dict(vary_intercept_per_trial=False, vary_intercept=False, vary_sigma_per_dataset=True)

    with pm.Model(coords=coords) as null_model:
        # Just do a flat line (intercept)
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        likelihood = build_final_likelihood(sigma, y, intercept=intercept, curvature_term=None, sigmoid_term=None)

    with pm.Model(coords=coords) as nonhierarchical_model:
        # Curvature, but no sigmoid
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        curvature_term, gamma = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        likelihood = build_final_likelihood(sigma, y, intercept=intercept, curvature_term=curvature_term, sigmoid_term=None,
                                            gamma=gamma)

    with pm.Model(coords=coords) as hierarchical_pca_model:
        # Curvature multiplied by sigmoid
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        sigmoid_term, beta = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term, gamma = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        # Diagnostic: how correlated are the terms?
        # pm.Deterministic('correlation_sigmoid_curvature', pt._orr(sigmoid_term, curvature_term))

        # Helper variable: total amplitude of eigenworms12 after modulation by curvature_term, to help interpret the overall effect size of the hierarchy
        modulated_eigenworm12_amplitude = pm.Deterministic('modulated_eigenworm12_amplitude', gamma * beta)

        likelihood = build_final_likelihood(sigma, y, intercept=intercept, sigmoid_term=sigmoid_term, curvature_term=curvature_term,
                                            gamma=gamma, beta=beta)

    coords.update({'time': np.arange(len(y))})

    all_models = {'hierarchical_pca': hierarchical_pca_model,
                  'null': null_model,
                  'nonhierarchical': nonhierarchical_model}
    all_traces = {}
    if dryrun:
        return pd.DataFrame(), all_traces, all_models
    # base_names_to_sample = {'y', 'sigmoid_term', 'curvature_term', 'phase_shift', 'sigmoid_slope'}
    for name, model in all_models.items():
        with model:
            opt = dict(draws=1500, tune=3000, random_seed=rng, target_accept=0.98)
            if DEBUG:
                opt['draws'] = 10
                opt['tune'] = 10

            trace = pm.sample(**opt,
                              chains=10, return_inferencedata=True, idata_kwargs={"log_likelihood": True},
                              cores=10)
            if sample_posterior:
                if sample_large_variables:
                    posterior_keys = list(trace.posterior.keys())
                    # Keep only those that have 'term' in it, because those are the time series
                    posterior_keys = [key for key in posterior_keys if 'term' in key]
                    posterior_keys.extend(['y', 'mu'])
                else:
                    posterior_keys = ['y']
                print(f"Sampling posterior predictive for {name}: {posterior_keys}")
                trace.extend(pm.sample_posterior_predictive(trace, random_seed=rng, progressbar=False,
                                                            var_names=posterior_keys))

            all_traces[name] = trace

    # Compute model comparisons
    all_loo = {}
    for name, trace in all_traces.items():
        loo = az.loo(trace)
        all_loo[name] = loo
    df_compare = az.compare(all_loo)

    return df_compare, all_traces, all_models


def build_baseline_priors(dims=None, dataset_name_idx=None, 
                          vary_sigma_per_dataset=True,
                          vary_intercept_per_trial=False, vary_intercept=False):
    # Note that with dr/r50 input data, the median is subtracted out so this is nearly centered already
    if not vary_intercept:
        # Mean is subtracted per dataset, so it should be fine
        intercept = None

    if dims is None:
        if vary_intercept:
            intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

    else:
        if vary_intercept_per_trial:
            # Include hyperprior
            hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
            hyper_intercept_sigma = pm.Exponential('hyper_intercept_sigma', lam=1)
            zscore_intercept = pm.Normal('zscore_intercept', mu=0, sigma=1, dims=dims)
            intercept = pm.Deterministic('intercept', hyper_intercept + zscore_intercept*hyper_intercept_sigma)[dataset_name_idx]
        elif vary_intercept:
            # i.e. same as if no dims were passed
            intercept = pm.Normal('intercept', mu=0, sigma=1)

        # Also or alternatively vary sigma per dataset; simpler because we don't have to zscore it
        if vary_sigma_per_dataset:
            sigma = pm.HalfNormal("sigma", sigma=1.0, dims=dims)[dataset_name_idx]
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

    return intercept, sigma


def build_final_likelihood(sigma, y, nu=5, intercept=None, sigmoid_term=None, curvature_term=None, gamma=None, beta=None):
    """
    Build the final likelihood for the model.
    
    Parameters
    ----------
    mu : array-like
        Mean of the Student-t distribution
    sigma : array-like
        Scale of the Student-t distribution
    y : array-like
        Observed data
    nu : float
        Degrees of freedom for Student-t (default: 5)
    prob_flip_sigma : float or None
        If None, use standard StudentT likelihood with mu.
        If a float between 0 and 1, marginalize over sign by using a mixture
        of mu and -mu, where prob_flip_sigma is the probability of the negative
        component (default: None)
    
    Returns
    -------
    pm.Distribution
        The likelihood distribution
    """
    # Build mu by combining non-None components
    # Initialize as pytensor tensors to ensure compatibility with pm.Deterministic
    mu_val = pt.as_tensor(0.0)
    if intercept is not None:
        mu_val = intercept
    else:
        # Initialize intercept to zero if not provided
        intercept = pt.as_tensor(0.0)
    
    # First, behavior only
    if curvature_term is not None:
        mu_val = mu_val + gamma * curvature_term

    # Full model (add nonlinearity)
    if sigmoid_term is not None and curvature_term is not None:
        mu_val = mu_val + beta * sigmoid_term * curvature_term

    # For backwards compatibility
    mu = pm.Deterministic('mu', mu_val)

    return pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)


def build_sigmoid_term_pca(x_pca_modes, dims=None, dataset_name_idx=None):
    """
    Build sigmoid term from PCA modes with positive amplitude constraints.
    
    Returns
    -------
    sigmoid_term : pm.Deterministic
        Sigmoid transformation of PCA term
    """

    # PCA modes and coefficients
    if dims is None:
        hyper_pca_amplitude, hyper_pca_sigma = np.array([0, 0]), np.array([1, 1])
        zscore_pca_amplitude = pm.Normal('zscore_pca_amplitude', mu=0, sigma=1, dims=dims)
        pca_amplitude = pm.Deterministic('pca_amplitude',
                                         hyper_pca_amplitude + zscore_pca_amplitude*hyper_pca_sigma)
        pca_term = pm.Deterministic('pca_term', pm.math.dot(x_pca_modes, pca_amplitude))
        beta = pm.Normal('beta', mu=0, sigma=1)
        inflection = pm.Normal('inflection', 0.0, 1.0)
    else:
        # Tight prior on inflection in normalized units, and make hierarchical
        inf_mu = pm.Normal('inflection_mu', 0.0, 1.0)
        inf_sigma = pm.HalfNormal('inflection_sigma', 0.5)
        z_inf = pm.Normal('z_inflection', 0.0, 1.0, dims=dims)
        inflection = pm.Deterministic('inflection', inf_mu + inf_sigma * z_inf)[dataset_name_idx]
        
        try:
            n_modes = int(x_pca_modes.shape[1])
        except TypeError:
            # Fallback: convert to symbolic int tensor
            n_modes = x_pca_modes.shape[1].eval()

        # Force only the first mode to be positive, because that's the one that we've anchored across datasets
        log_hyper = pm.Normal(f"log_hyper_pca0", mu=0.0, sigma=0.3)
        log_sigma = pm.HalfNormal(f"log_sigma_pca0", sigma=0.2)
        z = pm.Normal(f"z_pca0", 0.0, 1.0, dims=dims)

        log_amp = pm.Deterministic(
            f"log_pca0_amplitude",
            log_hyper + z * log_sigma
        )

        amp = pm.Deterministic(
            f"pca0_amplitude",
            pm.math.exp(log_amp)
        )

        pca_amplitudes = [amp]
        
        # Other modes are unconstrained
        for k in range(1, n_modes):
            amp = pm.Normal(f"pca{k}_amplitude", mu=0, sigma=0.5, dims=dims)
            pca_amplitudes.append(amp)

        pca_term = sum(
            pca_amplitudes[k][dataset_name_idx] * x_pca_modes[:, k]
            for k in range(n_modes)
        )
        
        # Hierarchical beta
        hyper_beta = pm.Normal('hyper_beta', mu=0, sigma=0.5)
        hyper_beta_sigma = pm.HalfNormal('hyper_beta_sigma', sigma=0.2)
        z_beta = pm.Normal('z_beta', mu=0, sigma=1, dims=dims)
        beta = pm.Deterministic('beta', hyper_beta + z_beta*hyper_beta_sigma)[dataset_name_idx]

    # Put it together to create the per-time-point Sigmoid term

    # Preprocessing point: separate scale into a separate variable, to reduce the chance that the model will saturate early
    pca_centered = pca_term - pm.math.mean(pca_term)
    pca_scaled = pca_centered / pt.std(pca_centered)

    k = pm.HalfNormal('sigmoid_slope', sigma=1.0)  # or pm.LogNormal('sigmoid_slope', 0.0, 0.5)
    sigmoid_term = pm.Deterministic('sigmoid_term', pt.sigmoid(k * pca_scaled - inflection))

    # standardize s for the interaction term too
    # s_centered = s_raw - pm.math.mean(s_raw)
    # s_std = s_centered / pt.std(s_centered)
    # sigmoid_term = pm.Deterministic('sigmoid_term', s_std)

    # Standardize to allow interpretation of coefficient (beta) as the expected change in sigmoid_term for a 1 unit change in curvature_term
    # sigmoid_term = sigmoid_term - pm.math.mean(sigmoid_term)
    
    return sigmoid_term, beta


def build_curvature_term(curvature, curvature_terms_to_use=None, dims=None, dataset_name_idx=None,

                         DEBUG=False):
    if curvature_terms_to_use is None:
        assert curvature.shape[1] == 4, f"Default curvature terms are for 4 eigenworms, found {curvature.shape[1]}"
        curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    if DEBUG:
        print(f"Using curvature terms {curvature_terms_to_use}")
    # Alternative: sample directly from the phase shift and amplitude, then convert into coefficients
    # This assumes that eigenworms 1 and 2 are approximately a sine and cosine wave, and puts it into polar coordinates
    phase_shift = pm.VonMises('phase_shift', mu=0.0, kappa=1e-3)  # Essentially uniform prior over phase shifts (0 gives error)
    if dims is None:
        hyper_log_amplitude, hyper_log_sigma = 0, 1
    else:
        # Hyperprior for overall amplitude
        hyper_log_amplitude = pm.Normal('log_amplitude_mu', mu=0, sigma=0.5)
        hyper_log_sigma = pm.HalfNormal('log_amplitude_sigma', sigma=0.5)
    zscore_log_amplitude = pm.Normal('zscore_log_amplitude', mu=0, sigma=1, dims=dims)
    log_amplitude = pm.Deterministic('log_amplitude', hyper_log_amplitude + zscore_log_amplitude*hyper_log_sigma)
    gamma = pm.Deterministic('gamma', pm.math.exp(log_amplitude))[dataset_name_idx] if dims is not None else pm.Deterministic('gamma', pm.math.exp(log_amplitude))

    # There is a positive and negative solution, so choose the positive one for the first term
    eigenworm1_coefficient = pm.Deterministic('eigenworm1_coefficient', pm.math.cos(phase_shift))
    eigenworm2_coefficient = pm.Deterministic('eigenworm2_coefficient', - pm.math.sin(phase_shift))
    # The rest are not part of the sine/cosine pair, but we aren't sure how many there are
    additional_column_dict = {}
    if len(curvature_terms_to_use) > 2:
        for col_name in curvature_terms_to_use[2:]:
            # None of these terms are modulated per dataset
            # If the column name is like "eigenworm3", the coefficient name is "eigenworm4_coefficient"
            # Because we want to start at 1, not 0
            if col_name.startswith('eigenworm'):
                coef_name = f'eigenworm{int(col_name[-1])+1}_coefficient'
            else:
                coef_name = f'{col_name}_coefficient'
            if DEBUG:
                print(f"Adding {coef_name} to the model")
            additional_column_dict[coef_name] = pm.Normal(coef_name, mu=0, sigma=1, dims=None)

    if dims is None:
        all_cols = [eigenworm1_coefficient, eigenworm2_coefficient]
        all_cols.extend(list(additional_column_dict.values()))  # Don't need to worry about the order
        coefficients_vec = pm.Deterministic('coefficients_vec', pm.math.stack(all_cols))
        curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    else:
        # Multiply them separately, but do not subindex by dataset for other terms
        curvature_term = pm.Deterministic('curvature_term',
                                          eigenworm1_coefficient * curvature[:, 0] +
                                          eigenworm2_coefficient * curvature[:, 1] +
                                          pt.sum(pt.stack([coef * curvature[:, i+2] for i, coef in enumerate(additional_column_dict.values())]), axis=0)
                                          )

    return curvature_term, gamma


def main_full_models(neuron_name=None, dataset_name='all', skip_if_exists=True, residual_mode='pca_global',
                     use_additional_eigenworms=True, keep_large_vars=False, DEBUG=False, **dataset_kwargs):
    """
    Fit full posterior models for multiple model structures and compute LOO.
    
    This is the original main() function. Runs for hardcoded data location for a single neuron.
    Saves all the information in the same directory as the data, in the 'output' subdirectory.

    Commonly used with:
        dataset_name = 'all' to run the neuron for all datasets at once
        dataset_name = 'loop' to run that neuron for each dataset seperately

    Returns
    -------

    """
    if DEBUG:
        skip_if_exists = False
        neuron_name = 'VB02'  # I know this exists even in GFP
    if neuron_name is None:
        neuron_name = 'VB02'

    print(f"Running all 3 bayesian models for {neuron_name} with dataset_kwargs={dataset_kwargs} and residual_mode={residual_mode} and DEBUG={DEBUG}")

    data_dir = get_hierarchical_modeling_dir(**dataset_kwargs)
    fname = os.path.join(data_dir, 'data.h5')
    if not os.path.exists(fname):
        # Try to read from backup
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
                # Recursion error
                continue
            main_full_models(neuron_name, dataset_name=dataset_name, skip_if_exists=skip_if_exists,
                           residual_mode=residual_mode, use_additional_eigenworms=use_additional_eigenworms, 
                           keep_large_vars=keep_large_vars, DEBUG=DEBUG, **dataset_kwargs)
        return

    if dataset_name == 'all':
        output_dir = os.path.join(data_dir, f'output_{residual_mode}')
    else:
        output_dir = os.path.join(data_dir, 'output_single_dataset')
    if DEBUG:
        output_dir = f"{output_dir}_debug"
    Path(output_dir).mkdir(exist_ok=True)
    # Check if it already exists
    if skip_if_exists and os.path.exists(os.path.join(output_dir, f'{neuron_name}_loo.h5')):
        print(f"Skipping {neuron_name} because it already exists")
        return

    # Fit models
    df_compare, all_traces, all_models = fit_multiple_models(Xy, neuron_name, dataset_name=dataset_name,
                                                             residual_mode=residual_mode, sample_large_variables=keep_large_vars,
                                                             use_additional_eigenworms=use_additional_eigenworms,
                                                             DEBUG=DEBUG)

    if df_compare is None:
        print(f"Skipping {neuron_name} because there is no valid data")
        return

    save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir, keep_large_vars=keep_large_vars)


def drop_large_variables_from_idata(idata, large_vars_to_drop=None, verbose=0):
    """
    Drop large deterministic variables from an ArviZ InferenceData object.
    
    Removes variables from posterior, posterior_predictive, and log_likelihood groups
    to reduce file size. Useful for traces with large deterministic outputs like
    ~20k observation arrays (curvature_term, mu, sigmoid_term, etc.).
    
    Parameters
    ----------
    idata : arviz.InferenceData
        The inference data object to modify
    large_vars_to_drop : list of str, optional
        Variable names to drop. Default: ['curvature_term', 'mu', 'sigmoid_term', 'pca_term', 'y']
    
    Returns
    -------
    idata_cleaned : arviz.InferenceData
        Modified copy with large variables removed
    """
    if large_vars_to_drop is None:
        large_vars_to_drop = ['curvature_term', 'mu', 'sigmoid_term', 'pca_term', 'y']
    if verbose >= 1:
        print(f"Dropping large variables: {large_vars_to_drop}")
    
    idata_cleaned = idata.copy()
    vars_to_drop = []
    pp_vars_to_drop = []
    ll_vars_to_drop = []
    
    # Drop from posterior
    if hasattr(idata_cleaned, 'posterior') and idata_cleaned.posterior is not None:
        # Debugging: rename intercept
        # print("DEBUGGING DROP LARGE VARS")
        # print(idata_cleaned.posterior)
        # idata_cleaned.posterior = idata_cleaned.posterior.rename(
        #     {"intercept": "_intercept"}
        # )
        # print(idata_cleaned.posterior)
        vars_to_drop = [v for v in large_vars_to_drop if v in idata_cleaned.posterior.data_vars]
        if vars_to_drop:
            idata_cleaned.posterior = idata_cleaned.posterior.drop_vars(vars_to_drop)

            # idata_cleaned.posterior = idata_cleaned.posterior.reset_encoding()
            if verbose >= 1:
                print(f"Dropped from posterior: {vars_to_drop}")
    
    # Drop from posterior_predictive if it exists
    if hasattr(idata_cleaned, 'posterior_predictive') and idata_cleaned.posterior_predictive is not None:
        pp_vars_to_drop = [v for v in large_vars_to_drop if v in idata_cleaned.posterior_predictive.data_vars]
        if pp_vars_to_drop:
            idata_cleaned.posterior_predictive = idata_cleaned.posterior_predictive.drop_vars(pp_vars_to_drop)
            # idata_cleaned.posterior_predictive = idata_cleaned.posterior_predictive.reset_encoding()
            if verbose >= 1:
                print(f"Dropped from posterior_predictive: {pp_vars_to_drop}")
    
    # Drop from log_likelihood if it exists
    if hasattr(idata_cleaned, 'log_likelihood') and idata_cleaned.log_likelihood is not None:
        ll_vars_to_drop = [v for v in large_vars_to_drop if v in idata_cleaned.log_likelihood.data_vars]
        if ll_vars_to_drop:
            idata_cleaned.log_likelihood = idata_cleaned.log_likelihood.drop_vars(ll_vars_to_drop)
            # idata_cleaned.log_likelihood = idata_cleaned.log_likelihood.reset_encoding()
            if verbose >= 1:
                print(f"Dropped from log_likelihood: {ll_vars_to_drop}")
    
    # Try to clear potentially stale metadata
    # Reconstruct InferenceData with only non-None groups
    # idata_groups = {}
    # for group_name in ['posterior', 'posterior_predictive', 'log_likelihood', 'observed_data', 'constant_data', 'sample_stats']:
    #     if hasattr(idata_cleaned, group_name) and getattr(idata_cleaned, group_name) is not None:
    #         idata_groups[group_name] = getattr(idata_cleaned, group_name)
    
    # idata_cleaned = az.InferenceData(**idata_groups)
    modified_flag = len(vars_to_drop) + len(pp_vars_to_drop) + len(ll_vars_to_drop) > 0
    return idata_cleaned, modified_flag


def save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir, keep_large_vars=False,
                           verbose=0):
    # Save objects, possibly dropping large variables from traces
    if dataset_name == 'all':
        output_fname_base = f'{neuron_name}'

        # arviz has a specific function for traces
        for model_name, traces in all_traces.items():
            # Optionally drop large variables (deterministic outputs with ~20k observations) to reduce file size
            if not keep_large_vars:
                if verbose >= 1:
                    print(f"Processing {model_name}...")
                traces_to_save, _ = drop_large_variables_from_idata(traces)
            else:
                traces_to_save = traces
            
            az.to_netcdf(traces_to_save, os.path.join(output_dir, f'{output_fname_base}_{model_name}_trace.nc'))
    else:
        output_fname_base = f'{neuron_name}_{dataset_name}'
    # Also save the model
    # See https://discourse.pymc.io/t/how-save-pymc-v5-models/13022
    model_fname = os.path.join(output_dir, f'{output_fname_base}_model.cloud_pkl')
    with open(model_fname, 'wb') as buffer:
        cloudpickle.dump(all_models, buffer)
    # Only save for the all dataset version
    fname = os.path.join(output_dir, f'{output_fname_base}_loo.h5')
    df_compare.to_hdf(fname, key='df_with_missing')
    # Save plots
    az.plot_compare(df_compare, insample_dev=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_fname_base}_model_comparison.png'))
    plt.close()
    print(f"Saved all objects for {neuron_name} in {output_dir}")



def reconstruct_model_term_from_trace(idata, neuron_name, Xy=None, dataset_name='all', residual_mode='pca_global',
                                      use_additional_eigenworms=True, var_names=None, DEBUG=False):
    """
    Reconstruct model variables from saved posterior samples using PyMC.
    
    Builds a PyMC model and uses pm.compute_deterministics to evaluate specified
    variables for all posterior samples. Can reconstruct deterministics (sigmoid_term,
    curvature_term, mu, etc.) or posterior predictive samples (y).
    Uses the same coords as the original model.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        Saved posterior with model parameters in the posterior group. 
        Should have coords matching the original model.
    neuron_name : str
        Name of the neuron to reconstruct
    Xy : pd.DataFrame, optional
        Full dataset. If None, will be loaded from hardcoded location using get_hierarchical_modeling_dir
    dataset_name : str
        Which dataset(s) to use ('all', specific dataset name, etc.)
    residual_mode : str
        Residual/preprocessing mode for data (default: 'pca_global')
    use_additional_eigenworms : bool
        Whether the model used eigenworms 2 and 3 (default: True)
    var_names : list of str, optional
        Variables to reconstruct. Default: ['sigmoid_term']
        Can include deterministics like 'sigmoid_term', 'curvature_term', 'mu', 'y', etc.
    
    Returns
    -------
    idata : arviz.InferenceData
        InferenceData with computed variables added to appropriate groups
    
    Examples
    --------
    >>> # Reconstruct sigmoid term
    >>> idata_recon = reconstruct_model_term_from_trace(idata, neuron_name='VB02', Xy=Xy)
    >>> quantiles = idata_recon.posterior['sigmoid_term'].quantile([0.05, 0.5, 0.95], dim=['chain', 'draw'])
    
    >>> # Reconstruct multiple variables
    >>> idata_recon = reconstruct_model_term_from_trace(idata, neuron_name='VB02', Xy=Xy,
    ...                                                    var_names=['sigmoid_term', 'curvature_term', 'mu'])
    
    >>> # Reconstruct final model output
    >>> idata_recon = reconstruct_model_term_from_trace(idata, neuron_name='VB02', Xy=Xy,
    ...                                                    var_names=['y'])
    """
    if var_names is None:
        var_names = ['sigmoid_term']
    
    # Load data if not provided
    if Xy is None:
        data_dir = get_hierarchical_modeling_dir(do_gfp=False)
        fname = os.path.join(data_dir, 'data.h5')
        if not os.path.exists(fname):
            logging.warning(f"Could not find data file {fname}, trying backup")
            fname = os.path.join(data_dir, 'data_backup.h5')
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Could not find data file {fname}")
        Xy = pd.read_hdf(fname)
    
    # Initialize model data using helper function
    model_data = initialize_hierarchical_model_data(
        Xy, neuron_name, 
        dataset_name=dataset_name,
        residual_mode=residual_mode,
        use_additional_eigenworms=use_additional_eigenworms,
        verbose=0 if not DEBUG else 1
    )
    
    # Build a fresh model with the same coords as the original
    with pm.Model(coords=model_data['coords']) as recon_model:
        x_pca_data = pm.Data('x_pca_data', model_data['pca_modes'])
        curvature_data = pm.Data('curvature_data', model_data['df_model'][model_data['curvature_terms_to_use']].values)
        y_data = pm.Data('y_data', model_data['df_model']['y'].values)
        
        if DEBUG:
            print(f"Reconstructing variables {var_names}")
            print("PCA modes shape:", model_data['pca_modes'].shape)
        
        # Build model components based on which variables are requested
        sigmoid_term_deterministic, beta = None, None
        if any(var in var_names for var in ['sigmoid_term', 'pca_term']):
            sigmoid_term_deterministic, beta = build_sigmoid_term_pca(
                x_pca_data, 
                **model_data['dim_opt']
            )
        
        curvature_term, gamma = None, None
        if any(var in var_names for var in ['curvature_term', 'mu', 'y']):
            curvature_term, gamma, _ = build_curvature_term(
                curvature_data, 
                curvature_terms_to_use=model_data['curvature_terms_to_use'],
                **model_data['dim_opt']
            )
        
        likelihood = None
        if 'mu' in var_names or 'y' in var_names:
            intercept, sigma = build_baseline_priors(**model_data['dim_opt'])
            
            # Build the sigmoid term value to use
            sigmoid_term_val = sigmoid_term_deterministic if sigmoid_term_deterministic is not None else None
            
            # Build likelihood for y and/or mu (likelihood creates mu as a deterministic)
            likelihood = build_final_likelihood(sigma, y_data, intercept=intercept, 
                                               sigmoid_term=sigmoid_term_val, 
                                               curvature_term=curvature_term,
                                               gamma=gamma, beta=beta)
    
    # Use PyMC's compute_deterministics to evaluate deterministics for all posterior samples
    deterministic_vars = [v for v in var_names if v not in ['y']]
    
    if deterministic_vars:
        with recon_model:
            idata = pm.compute_deterministics(
                idata, 
                var_names=deterministic_vars,
                progressbar=False,
                merge_dataset=True
            )
    
    # For posterior predictive samples (like 'y'), use sample_posterior_predictive
    if 'y' in var_names:
        with recon_model:
            idata.extend(pm.sample_posterior_predictive(
                idata,
                var_names=['y'],
                random_seed=42,
                progressbar=False
            ))
    
    return idata, model_data


def sample_prior_predictive_for_neuron(neuron_name, Xy=None, dataset_name='all', residual_mode='pca_global',
                                       use_additional_eigenworms=True, var_names=None, 
                                       num_samples=1000, random_seed=42, DEBUG=False):
    """
    Sample from prior predictive distribution for model variables.
    
    Builds a PyMC model and samples from the joint prior (priors only, no data conditioning).
    Useful for prior predictive checks to assess if priors are reasonable.
    
    Parameters
    ----------
    neuron_name : str
        Name of the neuron to sample for
    Xy : pd.DataFrame, optional
        Full dataset. If None, will be loaded from hardcoded location using get_hierarchical_modeling_dir
    dataset_name : str
        Which dataset(s) to use ('all', specific dataset name, etc.)
    residual_mode : str
        Residual/preprocessing mode for data (default: 'pca_global')
    use_additional_eigenworms : bool
        Whether the model used eigenworms 2 and 3 (default: True)
    var_names : list of str, optional
        Variables to sample. Default: ['y']
        Can include 'sigmoid_term', 'curvature_term', 'mu', 'y', etc.
    num_samples : int
        Number of prior samples to draw (default: 1000)
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    idata : arviz.InferenceData
        InferenceData with prior samples in 'prior' and 'prior_predictive' groups
    
    Examples
    --------
    >>> # Sample prior predictive for final output
    >>> idata_prior = sample_prior_predictive_for_neuron(neuron_name='VB02', Xy=Xy)
    
    >>> # Sample deterministics from prior
    >>> idata_prior = sample_prior_predictive_for_neuron(neuron_name='VB02', Xy=Xy,
    ...                                                   var_names=['sigmoid_term', 'curvature_term', 'y'])
    
    >>> # More samples for thorough prior predictive checks
    >>> idata_prior = sample_prior_predictive_for_neuron(neuron_name='VB02', Xy=Xy, num_samples=5000)
    """
    # if var_names is None:
    #     var_names = ['y']
    
    # Load data if not provided
    if Xy is None:
        data_dir = get_hierarchical_modeling_dir(do_gfp=False)
        fname = os.path.join(data_dir, 'data.h5')
        if not os.path.exists(fname):
            logging.warning(f"Could not find data file {fname}, trying backup")
            fname = os.path.join(data_dir, 'data_backup.h5')
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Could not find data file {fname}")
        Xy = pd.read_hdf(fname)
    
    # Initialize model data using helper function
    model_data = initialize_hierarchical_model_data(
        Xy, neuron_name, 
        dataset_name=dataset_name,
        residual_mode=residual_mode,
        use_additional_eigenworms=use_additional_eigenworms,
        verbose=0 if not DEBUG else 1
    )

    baseline_opt = dict(vary_intercept_per_trial=False, vary_intercept=False)
    
    # Build a fresh model with the same coords as the original
    with pm.Model(coords=model_data['coords']) as prior_model:
        x_pca_data = pm.Data('x_pca_data', model_data['pca_modes'])
        curvature_data = pm.Data('curvature_data', model_data['df_model'][model_data['curvature_terms_to_use']].values)
        y_data = pm.Data('y_data', model_data['df_model']['y'].values)
        
        if DEBUG:
            print(f"Sampling prior predictive for variables {var_names}")
            print("PCA modes shape:", model_data['pca_modes'].shape)
        
        # Always build the full hierarchical model, regardless of var_names
        # var_names is only used for filtering what to sample
        sigmoid_term_deterministic, beta = build_sigmoid_term_pca(
            x_pca_data, 
            **model_data['dim_opt']
        )
        
        curvature_term, gamma = build_curvature_term(
            curvature_data, 
            curvature_terms_to_use=model_data['curvature_terms_to_use'],
            **model_data['dim_opt']
        )
        
        intercept, sigma = build_baseline_priors(**model_data['dim_opt'], **baseline_opt)
        
        # Build the full likelihood with all terms
        likelihood = build_final_likelihood(sigma, y_data, intercept=intercept, 
                                           sigmoid_term=sigmoid_term_deterministic, 
                                           curvature_term=curvature_term,
                                           gamma=gamma, beta=beta)
    
    # Sample from prior predictive
    with prior_model:
        idata = pm.sample_prior_predictive(
            random_seed=random_seed,
            var_names=var_names,
            draws=num_samples,
        )
    
    return idata, model_data



##
## Model type 2: hierarchical ttests for triggered averages
##

@dataclass
class ExamplePymcPlotter:
    """
    Uses the same data as the bayesian simulation, but with custom parameters (for plotting purposes)

    An alternative is to do prior predictive simulations, but then there is no control over the exact parameters
    """

    Xy: pd.DataFrame
    neuron_name: str
    dataset_name: str = 'all'
    residual_mode: str = 'pca_global'
    curvature_terms_to_use: list = field(default_factory=lambda: ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3'])

    def __post_init__(self):
        self.df = get_dataframe_for_single_neuron(self.Xy, self.neuron_name, self.curvature_terms_to_use,
                                                  dataset_name=self.dataset_name, residual_mode=self.residual_mode)

    def model_radial_coordinates(self, eigenworm12_amplitude: float=0, eigenworm12_phase:float=0,
                                 eigenworm34_amplitudes=None,
                                 pca_amplitudes=None, inflection_point: float=0, intercept=0):
        """
        Evaluate a simulated model with the given parameters, using radial coordinates for eigenworms12

        Assumes 4 eigenworms are used

        Returns
        -------

        """
        if eigenworm34_amplitudes is None:
            eigenworm34_amplitudes = [0, 0]
        if pca_amplitudes is None:
            pca_amplitudes = [0, 0]

        # Build the curvature (behavior) term
        eig1 = eigenworm12_amplitude * np.cos(eigenworm12_phase)
        eig2 = eigenworm12_amplitude * np.sin(eigenworm12_phase)
        coefficients_vec = np.array([eig1, eig2, *eigenworm34_amplitudes])
        curvature = self.df[self.curvature_terms_to_use].values

        curvature_term = curvature @ coefficients_vec

        # Build the pca (sigmoid) term
        pca_modes = self.df[['x_pca0', 'x_pca1']].values
        pca_term = pca_modes @ pca_amplitudes

        x = pca_term - inflection_point
        sigmoid_term = 1.0 / (1.0 + np.exp(-x))

        # Combine
        mu = intercept + sigmoid_term * curvature_term
        df = pd.DataFrame({'y': mu, 'sigmoid_term': sigmoid_term, 'curvature_term': curvature_term})

        return df


def pt_corr(x, y):
    x_centered = x - pt.mean(x)
    y_centered = y - pt.mean(y)

    cov = pt.mean(x_centered * y_centered)
    std_x = pt.sqrt(pt.mean(x_centered**2))
    std_y = pt.sqrt(pt.mean(y_centered**2))

    return cov / (std_x * std_y)


if __name__ == '__main__':
    # Get neuron name from argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_name', '-n', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--residual_mode', type=str, default='pca_global')
    # Boolean
    parser.add_argument('--avb_hiscl', action='store_true')
    parser.add_argument('--gfp', action='store_true')
    parser.add_argument('--simple_eigenworms', action='store_true')
    parser.add_argument('--keep_large_vars', action='store_true', help='Keep large deterministic variables (curvature_term, mu, etc.) in saved traces')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cv_comparison', action='store_true', help='Run grouped CV comparison instead of full model fitting')

    args = parser.parse_args()

    residual_mode = args.residual_mode
    if residual_mode == 'None':
        residual_mode = None

    if args.cv_comparison:
        from wbfm.utils.external.utils_pymc_cv import main_cv_comparison
        main_cv_comparison(neuron_name=args.neuron_name, gfp=args.gfp, 
                           residual_mode=residual_mode,
                           use_additional_eigenworms=not args.simple_eigenworms, DEBUG=args.debug)
    else:
        main_full_models(neuron_name=args.neuron_name, 
                         residual_mode=residual_mode,
                         use_additional_eigenworms=not args.simple_eigenworms, keep_large_vars=args.keep_large_vars, DEBUG=args.debug,
                         gfp=args.gfp, avb_hiscl=args.avb_hiscl)

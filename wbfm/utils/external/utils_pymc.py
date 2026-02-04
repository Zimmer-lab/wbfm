import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
import arviz as az
import cloudpickle
from matplotlib import pyplot as plt
from scipy import stats
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
from statsmodels.stats.multitest import multipletests


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
                        sample_posterior=True, use_additional_behaviors=False,
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

    baseline_opt = dict(vary_intercept_per_trial=False, vary_intercept=False)

    with pm.Model(coords=coords) as null_model:
        # Just do a flat line (intercept)
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        likelihood = build_final_likelihood(sigma, y, intercept=intercept, curvature_term=None, sigmoid_term=None)

    with pm.Model(coords=coords) as nonhierarchical_model:
        # Curvature, but no sigmoid
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        likelihood = build_final_likelihood(sigma, y, intercept=intercept, curvature_term=curvature_term, sigmoid_term=None)

    with pm.Model(coords=coords) as hierarchical_pca_model:
        # Curvature multiplied by sigmoid
        intercept, sigma = build_baseline_priors(**dim_opt, **baseline_opt)
        sigmoid_term, prob_flip_sign = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        likelihood = build_final_likelihood(sigma, y, intercept=intercept, sigmoid_term=sigmoid_term, curvature_term=curvature_term, prob_flip_sign=prob_flip_sign)

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
            opt = dict(draws=1000, tune=1000, random_seed=rng, target_accept=0.96)
            if DEBUG:
                opt['draws'] = 10
                opt['tune'] = 10

            trace = pm.sample(**opt,
                              chains=10, return_inferencedata=True, idata_kwargs={"log_likelihood": True},
                              cores=10)
            if sample_posterior:
                posterior_keys = list(trace.posterior.keys())
                # var_names = base_names_to_sample.intersection(posterior_keys)
                # Keep only those that have 'term' in it, because those are the time series
                posterior_keys = [key for key in posterior_keys if 'term' in key]
                posterior_keys.extend(['y', 'mu'])
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
                          vary_intercept_per_trial=False, vary_intercept=False):
    # Note that with dr/r50 input data, the median is subtracted out so this is nearly centered already
    if not vary_intercept:
        # Mean is subtracted per dataset, so it should be fine
        intercept = None

    if dims is None:
        if vary_intercept:
            intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfCauchy("sigma", beta=0.5)

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
        sigma = pm.HalfNormal("sigma", sigma=1.0, dims=dims)[dataset_name_idx]

    return intercept, sigma


def build_final_likelihood(sigma, y, nu=5, 
                           intercept=None, sigmoid_term=None, curvature_term=None, prob_flip_sign=None):
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
    mu_flipped_val = pt.as_tensor(0.0)
    if intercept is not None:
        mu_val = intercept
        mu_flipped_val = intercept
    else:
        # Initialize intercept to zero if not provided
        intercept = pt.as_tensor(0.0)
    if sigmoid_term is not None and curvature_term is not None:
        mu_val = mu_val + sigmoid_term * curvature_term
        if prob_flip_sign is not None:
            mu_flipped_val = intercept + (1 - sigmoid_term) * curvature_term
    elif curvature_term is not None:
        mu_val = mu_val + curvature_term
        mu_flipped_val = mu_flipped_val + curvature_term
    
    mu = pm.Deterministic('mu', mu_val)

    if prob_flip_sign is None:
        return pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)
    else:
        # Marginalize over sign: mixture of mu and -mu
        # marginalize_sign is the weight for the positive component
        mu_flipped = pm.Deterministic('mu_flipped', mu_flipped_val)

        return pm.Mixture('y', w=[1.0 - prob_flip_sign, prob_flip_sign], 
                         comp_dists=[pm.StudentT.dist(mu=mu, sigma=sigma, nu=nu),
                                    pm.StudentT.dist(mu=mu_flipped, sigma=sigma, nu=nu)],
                         observed=y)


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
        if any(var in var_names for var in ['sigmoid_term', 'pca_term']):
            sigmoid_term_deterministic, prob_flip_sign = build_sigmoid_term_pca(
                x_pca_data, 
                **model_data['dim_opt']
            )
        
        if any(var in var_names for var in ['curvature_term', 'mu', 'y']):
            curvature_term = build_curvature_term(
                curvature_data, 
                curvature_terms_to_use=model_data['curvature_terms_to_use'],
                **model_data['dim_opt']
            )
        
        if 'mu' in var_names or 'y' in var_names:
            intercept, sigma = build_baseline_priors(**model_data['dim_opt'])
            
            # Build the sigmoid term value to use
            sigmoid_term_val = sigmoid_term_deterministic if 'sigmoid_term' in var_names or 'mu' in var_names else 1
            
            # Build likelihood for y and/or mu (likelihood creates mu as a deterministic)
            # Get prob_flip_sign from earlier if it was created
            pfs = prob_flip_sign if 'sigmoid_term' in var_names or 'mu' in var_names else None
            likelihood = build_final_likelihood(sigma, y_data, intercept=intercept, 
                                               sigmoid_term=sigmoid_term_val, 
                                               curvature_term=curvature_term,
                                               prob_flip_sign=pfs)
    
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
        sigmoid_term_deterministic, prob_flip_sign = build_sigmoid_term_pca(
            x_pca_data, 
            **model_data['dim_opt']
        )
        
        curvature_term = build_curvature_term(
            curvature_data, 
            curvature_terms_to_use=model_data['curvature_terms_to_use'],
            **model_data['dim_opt']
        )
        
        intercept, sigma = build_baseline_priors(**model_data['dim_opt'], **baseline_opt)
        
        # Build the full likelihood with all terms
        likelihood = build_final_likelihood(sigma, y_data, intercept=intercept, 
                                           sigmoid_term=sigmoid_term_deterministic, 
                                           curvature_term=curvature_term,
                                           prob_flip_sign=prob_flip_sign)
    
    # Sample from prior predictive
    with prior_model:
        idata = pm.sample_prior_predictive(
            random_seed=random_seed,
            var_names=var_names,
            draws=num_samples,
        )
    
    return idata, model_data


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


def build_sigmoid_term(x, force_positive_slope=True):
    # NOT USED
    # Sigmoid (hierarchy) term
    if force_positive_slope:
        log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
        sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))
    else:
        sigmoid_slope = pm.Normal('sigmoid_slope', mu=0, sigma=1)
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)
    # Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(sigmoid_slope * (x - inflection_point)))
    return sigmoid_term


def build_sigmoid_term_pca(x_pca_modes, force_positive_slope=True, dims=None, dataset_name_idx=None):
    """
    Build sigmoid term from PCA modes with positive amplitude constraints.
    
    Returns
    -------
    sigmoid_term : pm.Deterministic
        Sigmoid transformation of PCA term
    prob_flip_sign : pm.Distribution or None
        Probability to flip sign (used in marginalized likelihood).
        Returned when dims is not None, otherwise None.
    """
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=5)

    # PCA modes and coefficients
    if dims is None:
        hyper_pca_amplitude, hyper_pca_sigma = np.array([0, 0]), np.array([1, 1])
        zscore_pca_amplitude = pm.Normal('zscore_pca_amplitude', mu=0, sigma=1, dims=dims)
        pca_amplitude = pm.Deterministic('pca_amplitude',
                                         hyper_pca_amplitude + zscore_pca_amplitude*hyper_pca_sigma)
        pca_term = pm.Deterministic('pca_term', pm.math.dot(x_pca_modes, pca_amplitude))
        prob_flip_sign = None
    else:
        try:
            n_modes = int(x_pca_modes.shape[1])
        except TypeError:
            # Fallback: convert to symbolic int tensor
            n_modes = x_pca_modes.shape[1].eval()

        # Force only the first mode to be positive, because that's the one that we've anchored across datasets
        log_hyper = pm.Normal(f"log_hyper_pca0", 0.0, 1.0)
        log_sigma = pm.HalfNormal(f"log_sigma_pca0", 0.2)
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
            amp = pm.Normal(f"pca{k}_amplitude", mu=0, sigma=1, dims=dims)
            pca_amplitudes.append(amp)

        pca_term = sum(
            pca_amplitudes[k][dataset_name_idx] * x_pca_modes[:, k]
            for k in range(n_modes)
        )

        # Probability to flip sign (marginalize over sign in likelihood)
        prob_flip_sign = pm.Beta('prob_flip_sign', alpha=1, beta=1)

    # Put it together Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(pca_term - inflection_point))
    
    return sigmoid_term, prob_flip_sign


def build_curvature_term(curvature, curvature_terms_to_use=None, dims=None, dataset_name_idx=None,
                         DEBUG=False):
    if curvature_terms_to_use is None:
        assert curvature.shape[1] == 4, f"Default curvature terms are for 4 eigenworms, found {curvature.shape[1]}"
        curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    if DEBUG:
        print(f"Using curvature terms {curvature_terms_to_use}")
    # Alternative: sample directly from the phase shift and amplitude, then convert into coefficients
    # This assumes that eigenworms 1 and 2 are approximately a sine and cosine wave, and puts it into polar coordinates
    phase_shift = pm.Uniform('phase_shift', lower=-np.pi, upper=np.pi, transform=pm.distributions.transforms.circular)
    if dims is None:
        hyper_log_amplitude, hyper_log_sigma = 0, 1
    else:
        # Hyperprior
        hyper_log_amplitude = pm.Normal('log_amplitude_mu', mu=0, sigma=0.5)
        hyper_log_sigma = pm.HalfNormal('log_amplitude_sigma', sigma=0.5)
    zscore_log_amplitude = pm.Normal('zscore_log_amplitude', mu=0, sigma=1, dims=dims)
    log_amplitude = pm.Deterministic('log_amplitude', hyper_log_amplitude + zscore_log_amplitude*hyper_log_sigma)
    amplitude = pm.Deterministic('amplitude', pm.math.exp(log_amplitude))
    # There is a positive and negative solution, so choose the positive one for the first term
    eigenworm1_coefficient = pm.Deterministic('eigenworm1_coefficient', amplitude * pm.math.cos(phase_shift))
    eigenworm2_coefficient = pm.Deterministic('eigenworm2_coefficient', -amplitude * pm.math.sin(phase_shift))
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
                                          eigenworm1_coefficient[dataset_name_idx] * curvature[:, 0] +
                                          eigenworm2_coefficient[dataset_name_idx] * curvature[:, 1] +
                                          pt.sum(pt.stack([coef * curvature[:, i+2] for i, coef in enumerate(additional_column_dict.values())]), axis=0)
                                          )
    return curvature_term


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
                                                             residual_mode=residual_mode,
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


def _load_all_traces(foldername, single_neuron=None):
    from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
    fnames = neurons_with_confident_ids()
    if single_neuron is not None:
        fnames = [single_neuron]
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    print(f"Loaded {len(all_traces)} out of {len(fnames)} neurons from {foldername}")
    return all_traces


def do_bayesian_ttests(suffix = '_pca_global', single_neuron=None, do_gfp=False,
                       recalculate_sigmoid=True,
                       all_traces=None, Xy=None, verbose=0):
    from wbfm.utils.general.utils_hardcoded import role_of_neuron_dict

    parent_folder = get_hierarchical_modeling_dir(gfp=do_gfp)
                
    foldername = os.path.join(parent_folder, f'output{suffix}')
    if all_traces is None:
        all_traces = _load_all_traces(foldername, single_neuron=single_neuron)

    if Xy is None:
        Xy = pd.read_hdf(os.path.join(parent_folder, 'data.h5'))

    # foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
    # all_traces_gfp = load_all_traces(foldername)

    cols_to_plot = ['Hierarchy only', 'Hierarchical Behavior']

    # Just plot all
    if single_neuron is None:
        neurons_to_plot = set(all_traces.keys())
    else:
        neurons_to_plot = [single_neuron]

    var_names = ['prob_flip_sign',
                #'hyper_pca0_amplitude', "hyper_pca1_amplitude",  
                'pca0_amplitude', 'pca1_amplitude',
                'log_amplitude_mu', 'amplitude',
                'phase_shift', 
                'eigenworm3_coefficient', 'eigenworm4_coefficient'
    ]
    # Reference values for ttests; prob_flip_sign is special
    var_names_ref = [0.5, 0.01, 0.01]  # pca amplitudes are strictly positive, so use a ROPE
    var_names_ref.extend([0.0] * (len(var_names) - 3))
    var_names_ref = xr.DataArray(
        var_names_ref,
        dims=["variable"],
        coords={"variable": var_names}
    )

    var_names2 = ["sigmoid_term"]

    def _convert_0d_xarray(array, suffix=''):
        return pd.DataFrame({
            f"{name}{suffix}": [da.item()]
            for name, da in array.data_vars.items()
        })

    all_dfs = {}
    for n in tqdm(neurons_to_plot):
        # Original set of variables
        posterior = az.extract(all_traces[n], group='posterior', var_names=var_names, filter_vars='like')
        median_ds = posterior[var_names].median()
        median_df = _convert_0d_xarray(median_ds)
        all_dfs[n] = [median_df.T]

        # Also store the variances, with suffix '_var'
        var_ds = posterior[var_names].var()
        var_df = _convert_0d_xarray(var_ds, suffix='_var')
        all_dfs[n].append(var_df.T)

        # Also calculate the bayesian p-value vs. 0 for all columns
        prob_gt_zero = (posterior[var_names] > var_names_ref).mean()
        if verbose >= 1:
            print(f"Neuron {n} prob_gt_zero:\n{prob_gt_zero}")
        # Get the smaller one (or vector) and multiply by 2
        p_two_sided = 2 * xr.where(
            prob_gt_zero <= 0.5,
            prob_gt_zero,
            1 - prob_gt_zero
        )
        _df = _convert_0d_xarray(p_two_sided, suffix="_p_value")
        all_dfs[n].append(_df.T)

        # Recalculate the sigmoid term
        if recalculate_sigmoid:
            try:
                idata = all_traces[n]
                idata = reconstruct_model_term_from_trace(idata, n, Xy)
                
                # Variables with specific postprocessing
                dat = az.extract(idata, group='posterior', var_names=var_names2, filter_vars='like')
                summary = xr.Dataset(
                    {
                        "sigmoid_term": dat.median(),
                        "sigmoid_term_quantile": dat.quantile(0.8),
                        "sigmoid_term_variance": dat.var(),
                    }
                )
                all_dfs[n].extend([_convert_0d_xarray(summary).T])
            except AttributeError:
                logging.warning(f"Could not reconstruct sigmoid term for neuron {n}, skipping")
        
        all_dfs[n] = pd.concat(all_dfs[n])
    
    # Add final columns
    df_params = pd.concat(all_dfs, axis=1).T.droplevel(1)
    df_params['dataset_type'] = 'residual'

    # df_params['muscle_position'] = muscle_position
    # df_params.loc['RID', 'muscle_position'] = np.nan  # RID is strange

    # _df = df_to_plot_with_var[df_to_plot_with_var['datatype']=='Freely Moving (GCaMP, residual)'].copy()
    # _df.index = _df['neuron_name']
    # df_params['Hierarchy Score'] = _df['Hierarchy Score']
    # df_params['Relative Hierarchy Score'] = _df['Relative Hierarchy Score']

    df_params['Neuron Type'] = list(pd.Series(df_params.index).map(role_of_neuron_dict()))

    # Multiple comparison correction
    for col in df_params.columns:
        if '_p_value' in col:
            sig_flag, p_value_corrected, *_ = multipletests(df_params[col].values.squeeze(), method='fdr_bh', alpha=0.05)
            df_params[f'{col}_corrected'] = p_value_corrected
            df_params[f'{col}_is_significant'] = sig_flag

    # Get radial term: combination of raw curvature amplitude and median of the sigmoid term
    if recalculate_sigmoid and 'sigmoid_term_quantile' in df_params.columns:
        # df_params['r'] = np.exp(df_params['log_amplitude_mu']) * df_params['sigmoid_term_quantile'] 
        df_params['r'] = df_params['amplitude'] * df_params['sigmoid_term_quantile']
    else:
        logging.warning("sigmoid_term_quantile not found in df_params; using amplitude only for 'r'") 
        df_params['r'] = df_params['amplitude']
    if 'Relative Hierarchy Score' in df_params.columns:
        df_params['size'] = df_params['Relative Hierarchy Score'] + 1  # Add a minimum size
    else:
        df_params['size'] = 1.0  # Default size

    df_params['Neuron Type'] = pd.Series(df_params.index).map(role_of_neuron_dict(include_ventral_dorsal=True)).values

    # r = df_params['log_amplitude_mu']
    df_params['text'] = np.array(df_params.index)
    df_params['text_complete'] = np.array(df_params.index)
    if 'r' in df_params.columns:
        df_params.loc[df_params['r'] < 0.1, 'text'] = ''

    # Basic printing
    if verbose >= 1:
        print("Corrected p-values for pca0_amplitude:")
        print(df_params[['hyper_pca0_amplitude_p_value_corrected', 'hyper_pca0_amplitude_p_value_is_significant']].sort_values('hyper_pca0_amplitude_p_value_corrected'))

    return df_params


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

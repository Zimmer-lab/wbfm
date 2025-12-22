import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import arviz as az
import cloudpickle
from matplotlib import pyplot as plt
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir, get_triggered_average_modeling_dir, \
    get_triggered_average_dataframe_fname
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron


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
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
    if use_additional_eigenworms:
        curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
    if use_additional_behaviors:
        # curvature_terms_to_use = curvature_terms_to_use[:2]
        curvature_terms_to_use.extend([#'dorsal_only_body_curvature', 'dorsal_only_head_curvature',
                                       #    'ventral_only_body_curvature', 'ventral_only_head_curvature',
                                           'speed',
                                           'self_collision'])
    # First pack into a single dataframe to drop nan, then unpack
    try:
        df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                                   curvature_terms=curvature_terms_to_use, residual_mode=residual_mode)
    except KeyError as e:
        print(f"Skipping {neuron_name} because there is no valid data (KeyError: {e})")
        return None, None, None
    pca_modes = df_model[['x_pca0', 'x_pca1']].values
    y = df_model['y'].values
    curvature = df_model[curvature_terms_to_use].values

    if df_model.shape[0] == 0:
        print(f"Skipping {neuron_name} because there is no valid data (shape is 0)")
        return None, None, None

    if dataset_name == 'all':
        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'
    else:
        coords = {}
        dims, dataset_name_idx = None, None

    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

    with pm.Model(coords=coords) as null_model:
        # Just do a flat line (intercept)
        intercept, sigma = build_baseline_priors()#**dim_opt)
        mu = pm.Deterministic('mu', intercept)
        likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as nonhierarchical_model:
        # Curvature, but no sigmoid
        intercept, sigma = build_baseline_priors()#**dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        mu = pm.Deterministic('mu', intercept + curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as hierarchical_pca_model:
        # Curvature multiplied by sigmoid
        intercept, sigma = build_baseline_priors()#**dim_opt)
        sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

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
                              chains=4, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
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


def build_baseline_priors(dims=None, dataset_name_idx=None):
    if dims is None:
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        sigma = pm.HalfCauchy("sigma", beta=0.02)

    else:
        # Include hyperprior
        hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
        hyper_intercept_sigma = pm.Exponential('hyper_intercept_sigma', lam=1)
        zscore_intercept = pm.Normal('zscore_intercept', mu=0, sigma=1, dims=dims)
        intercept = pm.Deterministic('intercept', hyper_intercept + zscore_intercept*hyper_intercept_sigma)[dataset_name_idx]

        # Also vary sigma per dataset; simpler because we don't have to zscore it
        sigma = pm.HalfCauchy("sigma", beta=0.02, dims=dims)[dataset_name_idx]

    return intercept, sigma


def build_final_likelihood(mu, sigma, y, nu=100):
    return pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)


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
    # Sigmoid (hierarchy) term
    # if force_positive_slope:
    #     log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
    #     sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))
    # else:
    #     sigmoid_slope = pm.Normal('sigmoid_slope', mu=0, sigma=1)
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)

    # PCA modes and coefficients
    if dims is None:
        hyper_pca_amplitude, hyper_pca_sigma = np.array([0, 0]), np.array([1, 1])
        zscore_pca_amplitude = pm.Normal('zscore_pca_amplitude', mu=0, sigma=1, dims=dims)
        pca_amplitude = pm.Deterministic('pca_amplitude',
                                         hyper_pca_amplitude + zscore_pca_amplitude*hyper_pca_sigma)
        pca_term = pm.Deterministic('pca_term', pm.math.dot(x_pca_modes, pca_amplitude))
    else:
        # Hyperprior
        hyper_pca0_amplitude = pm.Normal('hyper_pca0_amplitude', mu=0, sigma=1)
        hyper_pca0_sigma = pm.Exponential('hyper_pca0_sigma', lam=1)
        hyper_pca1_amplitude = pm.Normal('hyper_pca1_amplitude', mu=0, sigma=1)
        hyper_pca1_sigma = pm.Exponential('hyper_pca1_sigma', lam=1)
        zscore_pca0_amplitude = pm.Normal('zscore_pca0_amplitude', mu=0, sigma=1, dims=dims)
        zscore_pca1_amplitude = pm.Normal('zscore_pca1_amplitude', mu=0, sigma=1, dims=dims)

        pca0_amplitude = pm.Deterministic('pca0_amplitude',
                                          hyper_pca0_amplitude + zscore_pca0_amplitude*hyper_pca0_sigma)
        pca1_amplitude = pm.Deterministic('pca1_amplitude',
                                          hyper_pca1_amplitude + zscore_pca1_amplitude*hyper_pca1_sigma)
        # Multiply them separately
        pca_term = pm.Deterministic('pca_term',
                                    pca0_amplitude[dataset_name_idx] * x_pca_modes[:, 0] +
                                    pca1_amplitude[dataset_name_idx] * x_pca_modes[:, 1])

    # Put it together Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(pca_term - inflection_point))
    return sigmoid_term


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
        hyper_log_amplitude = pm.Normal('log_amplitude_mu', mu=0, sigma=1)
        hyper_log_sigma = pm.Exponential('log_amplitude_sigma', lam=1)
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
            additional_column_dict[coef_name] = pm.Normal(coef_name, mu=0, sigma=0.5, dims=None)
    # eigenworm3_coefficient = pm.Normal('eigenworm3_coefficient', mu=0, sigma=0.5, dims=None)
    # eigenworm4_coefficient = pm.Normal('eigenworm4_coefficient', mu=0, sigma=0.5, dims=None)

    if dims is None:
        all_cols = [eigenworm1_coefficient, eigenworm2_coefficient]#, eigenworm3_coefficient, eigenworm4_coefficient]
        all_cols.extend(list(additional_column_dict.values()))  # Don't need to worry about the order
        coefficients_vec = pm.Deterministic('coefficients_vec', pm.math.stack(all_cols))
        curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    else:
        # Multiply them separately, but do not subindex by dataset for other terms
        curvature_term = pm.Deterministic('curvature_term',
                                          eigenworm1_coefficient[dataset_name_idx] * curvature[:, 0] +
                                          eigenworm2_coefficient[dataset_name_idx] * curvature[:, 1] +
                                          np.sum([coef * curvature[:, i+2] for i, coef
                                                  in enumerate(additional_column_dict.values())])
                                          )
    return curvature_term


def build_drift_term_gp(n, lengthscale=None, dims=None, dataset_name_idx=None):
    # Drift term (gaussian process)
    if lengthscale is None:
        lengthscale = n / 3.0
    eta = 2.0
    cov = eta ** 2 * pm.gp.cov.ExpQuad(1, lengthscale)
    # Add white noise to stabilise
    cov += pm.gp.cov.WhiteNoise(1e-6)
    # Actual gp, then make it a function
    gp = pm.gp.Latent(cov_func=cov)  # VERY slow
    X = np.linspace(0, n, n)[:, None]  # The inputs to the GP must be arranged as a column vector
    drift_term = gp.prior("f", X=X)

    return drift_term


def build_drift_term(dims=None, dataset_name_idx=None):
    # Drift term (random walk); needs 'time' in the dims
    # std of random walk
    sigma_alpha = pm.Exponential("sigma_alpha", 1.0)

    drift_term = pm.GaussianRandomWalk(
        "alpha", sigma=sigma_alpha, init_dist=pm.Normal.dist(0, 1), dims="time"
    )

    return drift_term


def leave_one_trial_out_cv_with_model(Xy, all_traces, neuron_name):

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
        sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
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


def main(neuron_name=None, do_gfp=False, dataset_name='all', skip_if_exists=True, residual_mode='pca_global',
         use_additional_eigenworms=True, DEBUG=False):
    """
    Runs for hardcoded data location for a single neuron

    Saves all the information in the same directory as the data, in the 'output' subdirectory

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

    print(f"Running all 3 bayesian models for {neuron_name} with do_gfp={do_gfp} and residual_mode={residual_mode} and DEBUG={DEBUG}")

    data_dir = get_hierarchical_modeling_dir(do_gfp)
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
            main(neuron_name, do_gfp=do_gfp, dataset_name=dataset_name, skip_if_exists=skip_if_exists,
                 residual_mode=residual_mode)
        return

    if dataset_name == 'all':
        output_dir = os.path.join(data_dir, 'output')
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

    save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir)


def save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir):
    # Save objects
    if dataset_name == 'all':
        output_fname_base = f'{neuron_name}'

        # arviz has a specific function for traces
        for model_name, traces in all_traces.items():
            az.to_netcdf(traces, os.path.join(output_dir, f'{output_fname_base}_{model_name}_trace.nc'))
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

def do_hierarchical_ttest(neuron_name, do_immob=False, do_mutant=False, do_downshift=False, do_hiscl=False,
                          trigger_type='raw_rev', skip_if_exists=True):
    """
    Designed to be used with a dataframe generated via get_df_triggered_from_trigger_type_all_traces_as_df

    Returns
    -------

    """

    # Load the data
    fname, _ = get_triggered_average_dataframe_fname(trigger_type, do_downshift, do_hiscl, do_immob, do_mutant)

    Xy = pd.read_hdf(fname)

    # Get information for this neuron
    df = Xy[neuron_name].melt(ignore_index=False).reset_index().dropna()

    # Add columns needed for the hierarchical model
    df['before'] = df['index'] < 0  # Add a column for before/after the event
    y = df['value'].values

    # First index: condition (before and after)
    # Need a cross product column of everything we will index: dataset, trial, and condition
    join_str = '--'
    df['dataset_trial_condition'] = df.dataset_name + join_str + df.trial_idx.astype(str) + join_str + df.before.astype(str)
    df['dataset_condition'] = df.dataset_name + join_str + df.before.astype(str)
    df['dataset_trial'] = df.dataset_name + join_str + df.trial_idx.astype(str)

    # First: just the values
    dataset_values = df["dataset_name"].unique()
    condition_values = df["before"].unique()
    trial_values = df["trial_idx"].unique()

    full_cross_product_strings = df['dataset_trial_condition'].unique()
    dataset_trial_cross_product_strings = df['dataset_trial'].unique()
    dataset_condition_cross_product_strings = df['dataset_condition'].unique()

    # Top index: from condition to the next level (dataset)
    # dataset2condition_names = [v.split(join_str)[2] for v in full_cross_product_strings]
    dataset2condition_names = [v.split(join_str)[1] for v in dataset_condition_cross_product_strings]
    dataset2condition_idx = [list(condition_values.astype(str)).index(name) for name in dataset2condition_names]

    # Middle index: from dataset to next level (trial)
    # We need to get the keys that will get here, i.e. that correspond to one condition

    # Middle index: from trial to the parent dataset AND condition
    trial2dataset_names = [join_str.join(np.array(v.split(join_str))[[0, 2]]) for v in full_cross_product_strings]
    # trial2dataset_idx = [list(dataset_values).index(name) for name in trial2dataset_names]
    trial2dataset_idx = [list(dataset_condition_cross_product_strings).index(name) for name in trial2dataset_names]

    # Bottom index: from the entire dataset to the parent combination idx, which is not the raw trial idx
    time2trial_idx = [list(full_cross_product_strings).index(name) for name in df['dataset_trial_condition']]

    coords = {
        "dataset_name": dataset_values,
        "condition_values": condition_values,
        "trial_values": trial_values,
        "dataset2condition_names": dataset2condition_names,
        "trial2dataset_names": trial2dataset_names,
        "dataset_condition_cross_product_strings": dataset_condition_cross_product_strings,
        "full_cross_product_strings": full_cross_product_strings
    }

    # Define the model
    with pm.Model(coords=coords) as hierarchical_model:

        alpha_condition = pm.Normal('alpha_condition', mu=0, sigma=1, dims='condition_values')
        sigma_condition = pm.HalfCauchy('sigma_condition', beta=0.5, dims='condition_values')

        # This is the thing that will be compared
        diff_of_means = pm.Deterministic("Difference of means", alpha_condition[1] - alpha_condition[0])

        # Hyperpriors for dataset-level random effects

        # Dataset-level random effects
        alpha_dataset = pm.Normal('alpha_dataset', mu=alpha_condition[dataset2condition_idx],
                                  sigma=sigma_condition[dataset2condition_idx],
                                  dims='dataset_condition_cross_product_strings')

        # Hyperpriors for trial-level random effects
        sigma_alpha_trial = pm.HalfNormal('sigma_alpha_trial', sigma=1)
        alpha_trial = pm.Normal('alpha_trial', mu=alpha_dataset[trial2dataset_idx], sigma=sigma_alpha_trial,
                                dims='full_cross_product_strings')

        # Model error term
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of the outcome
        # Indexing maps from datapoint (len of vector) to the dimensions of beta_trial
        mu = alpha_trial[time2trial_idx]

        # Likelihood (observed data)
        # y_obs = pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=100, observed=y)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sample from the model
    with hierarchical_model:
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=True, target_accept=0.999, cores=10)

    # Plot some diagnostics
    az.plot_posterior(
        trace,
        var_names=["Difference of means"],
        ref_val=0,
    )

    # Save
    output_dir = get_triggered_average_modeling_dir()  # Same as the input
    print(f"Saving all objects for {neuron_name} in {output_dir}")
    base_fname = f'hierarchical_ttest-{neuron_name}-immob{do_immob}-mutant{do_mutant}.png'
    plt.savefig(os.path.join(output_dir, base_fname))
    plt.close()

    # Save the trace
    base_fname = base_fname.replace('.png', '.zarr')
    trace.to_zarr(os.path.join(output_dir, base_fname))


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


if __name__ == '__main__':
    # Get neuron name from argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_name', '-n', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--residual_mode', type=str, default='pca_global')
    # Boolean
    parser.add_argument('--do_gfp', action='store_true')
    parser.add_argument('--simple_eigenworms', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    residual_mode = args.residual_mode
    if residual_mode == 'None':
        residual_mode = None

    main(neuron_name=args.neuron_name, do_gfp=args.do_gfp, residual_mode=residual_mode,
         use_additional_eigenworms=not args.simple_eigenworms, DEBUG=args.debug)

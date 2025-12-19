import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import arviz as az
from tqdm.auto import tqdm

import xarray as xr
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir


def plot_ts(idata, y='y', y_hat='y', num_samples=100, title='', confidence_interval=0.68, to_show=True,
            color_yhat=None, color_obs=None):
    """
    My implementation of the plot_ts function from ArviZ, which doesn't work for me.

    Uses plotly

    Parameters
    ----------
    idata
    y
    y_hat
    num_samples
    title
    confidence_interval - float, between 0 and 1; default is equal to the standard deviation of a gaussian

    Returns
    -------

    """
    if isinstance(idata, dict):
        for key in idata:
            plot_ts(idata[key], y=y, y_hat=y_hat, num_samples=num_samples, title=key, to_show=to_show)
        return

    if color_yhat is None:
        color_yhat = px.colors.qualitative.D3[1]
    if color_obs is None:
        color_obs = px.colors.qualitative.D3[0]

    # Build everything in a big dataframe, which will be used to plot everything
    y_obs = idata.observed_data.y

    # Get a sample from the large matrix of predictions
    # Original shape: (num_chains, num_draws, num_timesteps)
    # Target shape: (num_samples, num_timesteps)
    # First flatten the first two dimensions
    try:
        y_hat = idata.posterior_predictive.y
    except AttributeError:
        y_hat = idata.prior_predictive.y
    y_hat = np.array(y_hat).reshape(-1, y_hat.shape[2])
    # Take a random sample from both num_chains and num_draws
    idx = np.random.choice(y_hat.shape[0], num_samples)
    y_hat = y_hat[idx]

    # Combine everything into a single dataframe
    df = pd.DataFrame(y_hat.T)
    df['observed'] = y_obs

    # Plot
    fig = go.Figure()
    # Average of the samples
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].mean(axis=1), mode='lines', name='y_hat',
                             line=dict(color=color_yhat, width=2), opacity=0.75))
    # Confidence interval
    lower_bound = df.iloc[:, :-1].quantile(0.5 - confidence_interval / 2, axis=1)
    upper_bound = df.iloc[:, :-1].quantile(0.5 + confidence_interval / 2, axis=1)
    fig.add_trace(go.Scatter(x=df.index, y=lower_bound, mode='lines', name='y_hat',
                             line=dict(color=color_yhat, width=1), showlegend=False,
                             opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=upper_bound, mode='lines', name='y_hat',
                             line=dict(color=color_yhat, width=1), fill='tonexty',
                             showlegend=False, opacity=0.5))

    # Observed
    fig.add_trace(go.Scatter(x=df.index, y=df['observed'], mode='lines', name='observed',
                             line=dict(color=color_obs, width=2)))

    fig.update_layout(title=title)

    if to_show:
        fig.show()
    return fig


def plot_model_elements(idata, y_list=None, to_show=True, include_observed=True):
    """

    Parameters
    ----------
    idata
    y_list

    Returns
    -------

    """
    if y_list is None:
        y_list = ['y', 'sigmoid_term', 'curvature_term', 'mu']
    all_pred = {}
    for y_name in y_list:
        if 'posterior_predictive' in idata:
            y_pred = np.mean(np.mean(idata.posterior_predictive[y_name], axis=0), axis=0)
        elif y_name in idata.prior_predictive:
            y_pred = np.mean(np.mean(idata.prior_predictive[y_name], axis=0), axis=0)
        elif y_name in idata.prior:
            y_pred = np.mean(np.mean(idata.prior[y_name], axis=0), axis=0)
        else:
            raise ValueError(f"Could not find {y_name} in idata")
        all_pred[y_name] = y_pred
    df_pred = pd.DataFrame(all_pred)
    if include_observed:
        try:
            # Also get the observed data
            y_obs = idata.observed_data.y
            df_pred['observed'] = y_obs
        except AttributeError:
            pass
        # Move 'observed' to first position
        # df_pred = df_pred[['observed'] + [col for col in df_pred.columns if col != 'observed']]

    df_pred.rename(columns={'y': 'model_prediction', 'observed': 'data'}, inplace=True)

    # category_orders = ['observed']
    # category_orders.extend(y_list)
    fig = px.line(df_pred)#, category_orders=category_orders)
    fig.update_yaxes(title_text='dR/R50')
    fig.update_xaxes(title_text='Time (frames)')
    if to_show:
        fig.show()
    return fig


def load_from_disk_and_plot(trace_fname, model_substring='hierarchical', check_if_exists=True,
                            update_original_trace=True, verbose=1):
    """
    Loads a trace from disk, build the posterior predictive and plot it

    Returns
    -------

    """
    # Get the neuron name from the filename
    # Assume it is like */output/{neuron_name}_{model_name}_trace.nc
    neuron_name = Path(trace_fname).name.split('_')[0]
    if 'neuron' in neuron_name:
        if verbose >= 2:
            print(f"Detected non-ided neuron, skipping {trace_fname}")
        return None

    out_fname = Path(trace_fname).parent.parent.joinpath('plots').joinpath(f'{neuron_name}_posterior_predictive.html')
    if check_if_exists and os.path.exists(out_fname):
        if verbose >= 1:
            print(f"File {out_fname} already exists. Skipping")
        return None
    Path(out_fname).parent.mkdir(parents=True, exist_ok=True)

    if model_substring is not None:
        if model_substring not in trace_fname:
            if verbose >= 2:
                print(f"Skipping {trace_fname} because it is not a {model_substring} model")
            return None

    # Load the trace
    trace = az.from_netcdf(trace_fname)

    # Check if gcamp or gfp
    if Path(trace_fname).is_relative_to(get_hierarchical_modeling_dir()):
        is_gfp = False
    elif Path(trace_fname).is_relative_to(get_hierarchical_modeling_dir(gfp=True)):
        is_gfp = True
    else:
        raise ValueError(f"Could not find the original data file for {neuron_name}")

    trace.extend(sample_posterior_predictive(neuron_name, trace, is_gfp))
    if update_original_trace:
        if verbose >= 1:
            print(f"Updating original trace {trace_fname}")
        az.to_netcdf(trace, trace_fname)

    # Plot
    fig = plot_model_elements(trace, to_show=False)

    # Save
    fig.write_html(str(out_fname))

    return fig


def sample_posterior_predictive(neuron_name, trace, is_gfp=False):
    if 'posterior_predictive' not in trace:
        print(f"Posterior predictive not found for neuron {neuron_name}. Rebuilding the model")
        data_fname = os.path.join(get_hierarchical_modeling_dir(gfp=is_gfp), 'data.h5')

        Xy = pd.read_hdf(data_fname)
        # Rebuild the model, which is required to build the posterior predictive
        from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
        curvature_terms = ['eigenworm0', 'eigenworm1', 'eigenworm2']
        df_model = get_dataframe_for_single_neuron(Xy, neuron_name, curvature_terms=curvature_terms)
        x = df_model['x'].values
        y = df_model['y'].values
        curvature = df_model[curvature_terms].values

        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'

        dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)
        import pymc as pm
        with pm.Model(coords=coords) as hierarchical_model:
            # Full model
            from wbfm.utils.external.utils_pymc import build_baseline_priors, build_sigmoid_term, build_curvature_term, \
                build_final_likelihood
            intercept, sigma = build_baseline_priors(**dim_opt)
            sigmoid_term = build_sigmoid_term(x)
            curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms, **dim_opt)

            mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
            likelihood = build_final_likelihood(mu, sigma, y)

            # Sample
            trace = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term',
                                                                     'curvature_term', 'mu'])
    return trace


def leave_one_trial_out_cv(all_traces, Xy):
    """
    Performs leave-one-trial-out cross-validation given posterior distributions and the original indexing

    Returns
    -------

    """

    loto_results = {}
    for neuron_name, idata in tqdm(all_traces.items()):

        # 1. Get the log-likelihood
        # Dimensions: (chain, draw, obs_id)
        log_lik = idata.log_likelihood['y']

        # 2. Define your trial mapping
        # This should be a 1D array/Series of length obs_id, 
        # where each entry is the trial ID for that observation.
        # Example: trial_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2...]
        idx = Xy[neuron_name].dropna().index
        dataset_vector = Xy.loc[idx, 'dataset_name'].reset_index(drop=True)
        trial_mapping = xr.DataArray(dataset_vector, dims=['y_dim_0'])

        # 3. Perform Grouped Summation
        # This handles the unequal trial lengths automatically.
        # It sums the log-probs of all points within a trial.
        # Resulting dims: (chain, draw, trial_id)
        log_lik_grouped = log_lik.groupby(trial_mapping).sum(dim='y_dim_0')

        log_lik_grouped.name = "y" 
        idata_loto = az.InferenceData(
            posterior=idata.posterior, 
            log_likelihood=log_lik_grouped.to_dataset()
        )

        loto_results[neuron_name] = az.loo(idata_loto)

    return loto_results

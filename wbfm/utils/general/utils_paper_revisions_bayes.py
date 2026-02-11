from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pymc import reconstruct_model_term_from_trace
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
import xarray as xr
import os
import logging

from wbfm.utils.general.utils_paper import calculate_bayesian_model_categories, package_bayesian_df_for_plot


def _load_all_traces(foldername, single_neuron=None):
    from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
    fnames = neurons_with_confident_ids()
    if single_neuron is not None:
        fnames = [single_neuron]
    all_traces = {}
    for neuron in tqdm(fnames, desc=f"Loading traces from {foldername}"):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    print(f"Loaded {len(all_traces)} out of {len(fnames)} neurons from {foldername}")
    return all_traces


def _load_model_comparison_dfs(foldername, single_neuron=None):
    from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
    fnames = neurons_with_confident_ids()
    if single_neuron is not None:
        fnames = [single_neuron]
    _all_dfs = {}
    for filename in tqdm(Path(foldername).iterdir(), desc='Loading model comparison DataFrames'):
        if filename.name.endswith('.h5') and 'single' not in filename.name:
            neuron_name = '_'.join(filename.name.split('_')[:-1])
            _all_dfs[neuron_name] = pd.read_hdf(filename)
    df = pd.concat(_all_dfs).reset_index(names=['neuron_name', 'model_type'])
    print(f"Loaded {len(_all_dfs)} out of {len(fnames)} neurons from {foldername}")
    return df


def calculate_paper_model_comparisons(suffix='_pca_global', single_neuron=None, Xy=None, Xy_gfp=None,
                                      y_column = 'Relative Hierarchy Score', x_column = 'Behavior Score', remove_names_of_ns=True,
                                      verbose=0):

    # Load raw arviz results
    foldername = os.path.join(get_hierarchical_modeling_dir(), f'output{suffix}')
    foldername_gfp = os.path.join(get_hierarchical_modeling_dir(gfp=True), f'output{suffix}')

    df = _load_model_comparison_dfs(foldername, single_neuron=single_neuron)
    df_gfp = _load_model_comparison_dfs(foldername_gfp, single_neuron=single_neuron)

    if Xy is None:
        Xy = pd.read_hdf(os.path.join(get_hierarchical_modeling_dir(), 'data.h5'))
    if Xy_gfp is None:
        Xy_gfp = pd.read_hdf(os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5'))

    # Package for plotting
    df_to_plot_gcamp = package_bayesian_df_for_plot(df, df_normalization=Xy, 
                                                # min_num_datapoints=10000
                                               ).assign(datatype='Freely Moving (GCaMP, residual)')
    df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, df_normalization=Xy_gfp, 
                                                # min_num_datapoints=5000
                                                ).assign(datatype='Freely Moving (GFP, residual)')
    
    df_to_plot = pd.concat([df_to_plot_gcamp, df_to_plot_gfp])
    df_to_plot['Dataset Type'] = df_to_plot['datatype']
    df_to_plot['Size'] = 1

    # Calculate categories and text for plotting first panel
    df_to_plot, _df, text, y_max_gfp, x_max_gfp = calculate_bayesian_model_categories(x_column, y_column, df_to_plot_gfp, df_to_plot_gcamp, 
                                                                                      remove_names_of_ns=remove_names_of_ns)

    df_to_plot.reset_index(inplace=True, drop=True)  # Already set as a column (neuron_name)

    return df_to_plot, _df, text, y_max_gfp, x_max_gfp


def calculate_bayesian_ttests(suffix = '_pca_global', single_neuron=None, do_gfp=False,
                              recalculate_sigmoid=True, var_names=None,
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

    # Just plot all
    if single_neuron is None:
        neurons_to_plot = set(all_traces.keys())
    else:
        neurons_to_plot = [single_neuron]

    if var_names is None:
        var_names = ['log_hyper_pca0', 'pca1_amplitude',
                    'hyper_beta', 
                    'log_amplitude_mu',
                    'phase_shift',
                    'eigenworm3_coefficient', 'eigenworm4_coefficient'
        ]
    # Reference values for ttests; prob_flip_sign is special
    var_names_ref = [0.01]  # pca amplitudes are strictly positive, so use a ROPE
    var_names_ref.extend([0.0] * (len(var_names) - len(var_names_ref)))
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
    for n in tqdm(neurons_to_plot, desc="Processing neurons for Bayesian t-tests"):
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

    df_params['Neuron Type'] = list(pd.Series(df_params.index).map(role_of_neuron_dict()))

    # Multiple comparison correction
    for col in df_params.columns:
        if '_p_value' in col:
            sig_flag, p_value_corrected, *_ = multipletests(df_params[col].values.squeeze(), method='fdr_bh', alpha=0.05)
            df_params[f'{col}_corrected'] = p_value_corrected
            df_params[f'{col}_is_significant'] = sig_flag

    # Get radial term: combination of raw curvature amplitude and median of the sigmoid term
    if recalculate_sigmoid and 'sigmoid_term_quantile' in df_params.columns:
        df_params['r'] = np.exp(df_params['log_amplitude_mu']) * df_params['sigmoid_term_quantile']
        # df_params['r'] = df_params['amplitude'] * df_params['sigmoid_term_quantile']
    else:
        logging.warning("sigmoid_term_quantile not found in df_params; using amplitude only for 'r'")
        df_params['r'] = np.exp(df_params['log_amplitude_mu'])
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

    if do_gfp:
        df_params = df_params.assign(datatype='Freely Moving (GFP, residual)')
    else:
        df_params = df_params.assign(datatype='Freely Moving (GCaMP, residual)')

    df_params.reset_index(inplace=True, names=['neuron_name'])

    return df_params


def calculate_all_bayesian_model_data(suffix='_pca_global', single_neuron=None):
    """
    Use above functions to calculate all data needed for the Bayesian model comparison and t-tests

    """

    df_to_plot, _df, text, y_max_gfp, x_max_gfp = calculate_paper_model_comparisons(suffix=suffix, single_neuron=single_neuron)

    df_params_gcamp = calculate_bayesian_ttests(suffix=suffix, single_neuron=single_neuron, do_gfp=False,
                                                recalculate_sigmoid=False, var_names=None,
                                                all_traces=None, Xy=None, verbose=0)
    
    # Combine the model comparison data with the parameter data for plotting
    df_params = df_params_gcamp.merge(df_to_plot, on=['datatype', 'neuron_name'], how='left')

    return df_params, y_max_gfp, x_max_gfp
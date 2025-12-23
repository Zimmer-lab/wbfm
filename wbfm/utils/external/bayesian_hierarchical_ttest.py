from wbfm.utils.general.utils_hardcoded import get_triggered_average_dataframe_fname, get_triggered_average_modeling_dir


import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import os


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
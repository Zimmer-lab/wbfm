import os

from wbfm.utils.general.utils_hardcoded import load_paper_datasets, get_hierarchical_modeling_dir
from wbfm.utils.visualization.multiproject_wrappers import build_cca_time_series_from_multiple_projects, build_trace_time_series_from_multiple_projects, \
    build_behavior_time_series_from_multiple_projects, build_cross_dataset_eigenworms, \
    build_pca_time_series_from_multiple_projects


def export_data_for_hierarchical_model(suffix='', skip_if_exists=True, delete_if_exists=False):
    """
    Loads the relevant projects, and exports both behavior and traces to a single .h5 file

    Returns
    -------

    """
    # Check if file exists
    data_dir = get_hierarchical_modeling_dir(suffix=suffix)
    output_fname = os.path.join(data_dir, 'data.h5')
    print(f"Exporting data to {output_fname} with suffix {suffix}")
    if os.path.exists(output_fname):
        if skip_if_exists:
            print(f"File {output_fname} already exists, skipping")
            return
        elif delete_if_exists:
            print(f"File {output_fname} already exists, deleting")
            os.remove(output_fname)
        else:
            raise FileExistsError(f"File {output_fname} already exists; set delete_if_exists=True to overwrite"
                                  f" or skip_if_exists=True to skip")

    # Load projects from the suffix
    all_projects = load_paper_datasets(suffix)
    do_immobilized = 'immob' in suffix

    # Get individual data elements
    df_all_traces = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_traces.sort_values(['dataset_name', 'local_time'], inplace=True)

    if not do_immobilized:
        behavior_names = ['curvature_vb02', #'curvature_5', 'curvature_10', 'curvature_15', 'curvature_20',
                          'speed', 'ventral_only_head_curvature', 'dorsal_only_head_curvature',
                          'ventral_only_body_curvature', 'dorsal_only_body_curvature', 'self_collision',
                          'head_signed_curvature', 'summed_curvature',
                          'worm_nose_peak_frequency', 'worm_head_peak_frequency', 'worm_body_peak_frequency']
        # Also include the entire kymograph
        behavior_names += [f'curvature_{i}' for i in range(1, 100)]
        df_all_behavior = build_behavior_time_series_from_multiple_projects(all_projects, behavior_names=behavior_names)
        df_all_behavior.sort_values(['dataset_name', 'local_time'], inplace=True)

        # Add discrete behaviors, and save as int
        discrete_behavior_names = ['rev', 'ventral_turn', 'dorsal_turn', 'pause', 'self_collision']
        df_beh_binary = build_behavior_time_series_from_multiple_projects(all_projects, behavior_names=discrete_behavior_names)
        for name in discrete_behavior_names:
            df_beh_binary[name] = df_beh_binary[name].astype(int)
        df_all_behavior = df_all_behavior.merge(df_beh_binary, on=['dataset_name', 'local_time'], how='inner')

        # Recalculate multi-dataset eigenworms
        df_eigenworms = build_cross_dataset_eigenworms(all_projects)

        # Also calculate CCA modes as done in the paper
        df_all_cca = build_cca_time_series_from_multiple_projects(all_projects, use_paper_options=True)[0]
        df_all_cca.rename(columns={i: f'pca_{i}' for i in range(4)}, inplace=True)

    # Get pca modes
    df_all_pca = build_pca_time_series_from_multiple_projects(all_projects, use_paper_options=True)
    df_all_pca.rename(columns={i: f'pca_{i}' for i in range(4)}, inplace=True)

    # Get manifold in two ways: pc1 subtraction and pc1 and 2 subtraction (original)
    df_all_manifold = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True,
                                                                     interpolate_nan=True, residual_mode='pca_global')
    df_all_manifold.sort_values(['dataset_name', 'local_time'], inplace=True)
    # Different calculation of the manifold
    df_all_manifold1 = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True,
                                                                      interpolate_nan=True,
                                                                      residual_mode='pca_global_1')
    df_all_manifold1.sort_values(['dataset_name', 'local_time'], inplace=True)

    # Align and export
    # Remake local time columns to just be integers
    df_all_traces['local_time'] = df_all_traces.groupby('dataset_name').cumcount()
    df_all_manifold['local_time'] = df_all_manifold.groupby('dataset_name').cumcount()
    df_all_manifold1['local_time'] = df_all_manifold1.groupby('dataset_name').cumcount()
    if not do_immobilized:
        df_all_behavior['local_time'] = df_all_behavior.groupby('dataset_name').cumcount()
        df_eigenworms['local_time'] = df_eigenworms.groupby('dataset_name').cumcount()
    df_all_pca['local_time'] = df_all_pca.groupby('dataset_name').cumcount()
    # Include all neurons
    df_all = df_all_traces.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                                 suffixes=('', '_manifold'))
    df_all = df_all.merge(df_all_manifold, on=['dataset_name', 'local_time'], how='inner',
                          suffixes=('', '_manifold1'))
    if not do_immobilized:
        df_all = df_all.merge(df_all_behavior, on=['dataset_name', 'local_time'], how='inner')
        df_all = df_all.merge(df_eigenworms, on=['dataset_name', 'local_time'], how='inner')
        df_all = df_all.merge(df_all_cca, on=['dataset_name', 'local_time'], how='inner')
    df_all = df_all.merge(df_all_pca, on=['dataset_name', 'local_time'], how='inner')

    # Export
    df_all.to_hdf(output_fname, key='df_with_missing')
    print(f"Exported to {output_fname}")

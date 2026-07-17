import os

import pandas as pd
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


def export_data_for_oded_lab():
    """
    Uses the same data as export_data_for_hierarchical_model, but removes the manifold and pca columns

    Returns
    -------

    """
    # Load generated data
    data_dir = get_hierarchical_modeling_dir()
    fname = os.path.join(data_dir, 'data.h5')
    df_all = pd.read_hdf(fname)

    # Remove manifold and pca columns
    cols_to_remove = [col for col in df_all.columns if 'manifold' in col or 'pca' in col or "'" in col or 'eigenworm' in col or '/' in col]
    cols_to_remove += ['curvature_vb02', 'AVABL']
    df_all.drop(columns=cols_to_remove, inplace=True)
    df_all = pd.DataFrame(df_all)

    # Remove columns with too few values
    threshold = 4000  # More than 2 datasets
    df_all = df_all.dropna(thresh=threshold, axis=1)

    # Export
    fname = os.path.join(data_dir, 'data_oded_lab.h5')
    df_all.to_hdf(fname, key='df_with_missing')

    return df_all


def calculate_bundle_net_export(project_data, output_dir=None):
    """
    Calculates a trace and behavior dataframe, designed for use with the bundle net paper

    Parameters
    ----------
    project_data

    Returns
    -------

    """

    # Get filtered traces with manual IDs
    df_traces = project_data.calc_default_traces(use_paper_options=True)

    # Get behavior annotations, but only the main ones and simplify
    worm: WormFullVideoPosture = project_data.worm_posture_class
    df_beh_raw = worm.beh_annotation(fluorescence_fps=True, reset_index=True)
    df_beh = BehaviorCodes.convert_to_simple_states_vector(df_beh_raw)
    df_beh = df_beh.apply(lambda x: x.name)  # Save simple strings
    df_beh.index = df_traces.index

    df_curvature = worm.curvature(fluorescence_fps=True, reset_index=True)
    df_curvature.index = df_traces.index

    if output_dir is not None:
        traces_fname = os.path.join(output_dir, 'traces.csv')
        df_traces.to_csv(traces_fname)
        # df_traces.to_hdf(traces_fname, key='df_with_missing', mode='w')

        beh_fname = os.path.join(output_dir, 'behavior.csv')
        df_beh.to_csv(beh_fname)

        curvature_fname = os.path.join(output_dir, 'curvature.csv')
        df_curvature.to_csv(curvature_fname)

    return df_traces, df_beh


if __name__ == "__main__":
    export_data_for_oded_lab()

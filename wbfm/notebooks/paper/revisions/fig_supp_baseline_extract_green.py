"""
Step 1 of the baseline vs. oscillation control (Reviewer 2, detection effects): export raw green traces.

Run in the `wbfm` conda env (project loading needs it; the analysis notebook fig_supp_baseline.ipynb runs in
`pymc_311` for arviz):
    cd wbfm/notebooks/paper/revisions
    /home/charles/anaconda3/envs/wbfm/bin/python -I fig_supp_baseline_extract_green.py

Source and preprocessing are identical to Supplementary Fig. S4f ("Mean Green Brightness",
../supplement/fig_S4_gfp.py): build_trace_time_series_from_multiple_projects(..., use_paper_options=True,
channel_mode='green', min_nonnan=0.7). With use_paper_options=True this loads the cached paper_traces_green.h5
of each project (paper_trace_settings() with channel_mode='green'). Note: the current dispatcher only supports
the green channel with interpolate_nan=True (the cached file), so this is passed explicitly; the S4 call without it
now raises a ValueError.

Datasets: the same 36 freely moving GCaMP datasets as the hierarchical-modeling data.h5
(load_paper_datasets(['gcamp', 'hannah_O2_fm']); fig_S4_gfp.py itself used only the 23 'gcamp' datasets).

Outputs (in fig_supp_baseline/):
    green_traces.h5   tall dataframe: dataset_name, local_time (integer frame index, as in data.h5),
                      local_time_physical (s), one column per (manually IDed) neuron
    volume_traces.h5  segmented volume (voxels, 'area') per neuron and frame, raw; same keys and neuron names
    ratio_traces.h5   same, for the ratio traces used in data.h5 (dr_over_r_50, no interpolation); only used to check
                      the frame alignment
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects

OUT_DIR = Path(__file__).resolve().parent / 'fig_supp_baseline'
OUT_DIR.mkdir(exist_ok=True)
SKIP_EXISTING = True  # set False to re-export the green and ratio traces


def _to_frame_index(df):
    df = df.sort_values(['dataset_name', 'local_time']).reset_index(drop=True)
    df = df.rename(columns={'local_time': 'local_time_physical'})
    df.insert(1, 'local_time', df.groupby('dataset_name').cumcount())
    return df


if __name__ == '__main__':
    all_projects = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
    print(f"Loaded {len(all_projects)} projects")

    if not SKIP_EXISTING or not (OUT_DIR / 'green_traces.h5').exists():
        df_green = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True,
                                                                  channel_mode='green', min_nonnan=0.7,
                                                                  interpolate_nan=True)
        df_green = _to_frame_index(df_green)
        df_green.to_hdf(OUT_DIR / 'green_traces.h5', key='df_with_missing')
        print(f"Saved green traces: {df_green.shape}")

    if not SKIP_EXISTING or not (OUT_DIR / 'ratio_traces.h5').exists():
        # Same call as for the trace columns of data.h5 (see export_data_for_hierarchical_model)
        df_ratio = build_trace_time_series_from_multiple_projects(all_projects, use_paper_options=True)
        df_ratio = _to_frame_index(df_ratio)
        df_ratio.to_hdf(OUT_DIR / 'ratio_traces.h5', key='df_with_missing')
        print(f"Saved ratio traces: {df_ratio.shape}")

    # Segmented volume (voxels) per neuron and frame. The paper green trace is calculation_mode='integration', i.e. the
    # SUM over the neuron's voxels minus background_per_pixel * volume (then bleach-corrected), so a large dim neuron
    # reads as bright. Dividing by this volume gives the mean intensity per voxel above background.
    # Raw values (no filtering/interpolation), same frames and the same manual-ID renaming as the paper traces.
    all_dfs = {}
    for dataset_name, p in all_projects.items():
        df_vol = p.calc_raw_traces(tuple(p.neuron_names), channel_mode='green', calculation_mode='volume').copy()
        df_vol = df_vol.iloc[:p.num_frames, :]
        mapping = p.neuron_name_to_manual_id_mapping(confidence_threshold=0, only_include_confident_labels=True)
        df_vol = df_vol.rename(columns=mapping)
        df_vol.index = np.arange(len(df_vol))
        all_dfs[dataset_name] = df_vol
    df_vol = pd.concat(all_dfs).reset_index(names=['dataset_name', 'local_time'])
    df_vol = df_vol.sort_values(['dataset_name', 'local_time']).reset_index(drop=True)
    df_vol.to_hdf(OUT_DIR / 'volume_traces.h5', key='df_with_missing')
    print(f"Saved volume traces: {df_vol.shape}")

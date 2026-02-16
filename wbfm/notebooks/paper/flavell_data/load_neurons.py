#!/usr/bin/env python3
"""Load traces with neurons as columns, dataset_name for grouping."""

import json
import glob
import pandas as pd
import numpy as np

try:
    from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
    HAS_WBFM = True
except ImportError:
    HAS_WBFM = False

DEFAULT_BEHAVIOR_COLUMNS = ['velocity', 'angular_velocity', 'head_curvature', 'body_curvature', 'pumping']


def load_neurons(glob_pattern="*.json", behavior_columns=None, verbose=True, include_hierarchical=False):
    """Load neuron traces from JSON files.

    Args:
        glob_pattern: Glob pattern to match JSON files (default: "*.json")
        behavior_columns: List of behavior column names to include (default: DEFAULT_BEHAVIOR_COLUMNS)
        verbose: Whether to print loading progress (default: True)
        include_hierarchical: Whether to also load hierarchical modeling data from wbfm (default: False)

    Returns:
        DataFrame with dataset_name, local_time, behavior columns, and neuron columns
    """
    if behavior_columns is None:
        behavior_columns = DEFAULT_BEHAVIOR_COLUMNS

    json_files = sorted(glob.glob(glob_pattern))
    if verbose:
        print(f"Found {len(json_files)} JSON files")

    all_dfs = []

    for fpath in json_files:
        fname = fpath.replace(".json", "")
        
        with open(fpath, "r") as f:
            data = json.load(f)
        
        traces = data["trace_original"]
        labeled = data.get("labeled", {})
        n_timepoints = len(traces[0])
        
        trace_dict = {}
        for i, trace in enumerate(traces):
            if str(i + 1) in labeled:
                col_name = labeled[str(i + 1)].get("label")
            else:
                col_name = f"neuron_{i+1:03d}"
            
            trace_dict[col_name] = trace
        
        df = pd.DataFrame(trace_dict)
        
        for col in behavior_columns:
            if col in data:
                df[col] = data[col]
        
        df.insert(0, 'local_time', np.arange(n_timepoints))
        df.insert(0, 'dataset_name', fname)
        
        all_dfs.append(df)
        if verbose:
            print(f"  Loaded {fname}: {n_timepoints} timepoints, {len(trace_dict)} neurons, {len(df.columns) - 2} total columns")

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.assign(source='flavell')

    if include_hierarchical:
        if not HAS_WBFM:
            raise ImportError("wbfm package not available")
        data_dir = get_hierarchical_modeling_dir()
        h5_path = f"{data_dir}/data.h5"
        if verbose:
            print(f"\nLoading hierarchical modeling data from {h5_path}")
        hm_df = pd.read_hdf(h5_path, "df_with_missing")
        if verbose:
            print(f"  Loaded hierarchical: {hm_df.shape[0]} timepoints x {hm_df.shape[1]} columns")
        hm_df = hm_df.assign(source='zimmer')
        result = pd.concat([result, hm_df], ignore_index=True)

    if verbose:
        print(f"\nCreated dataframe: {result.shape[0]} timepoints x {result.shape[1]} columns")
        print(f"Columns: {list(result.columns[:5])} ... ({len(result.columns)} total)")
        print(f"\nExample groupby:")
        print(f"  result.groupby('dataset_name').apply(lambda x: x['neuron_001'].dropna().mean())")

    return result


if __name__ == "__main__":
    result = load_neurons()

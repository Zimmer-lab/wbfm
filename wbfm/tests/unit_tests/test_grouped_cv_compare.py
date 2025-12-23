#!/usr/bin/env python
"""
Quick test of grouped_cv_compare functionality.
"""
import pandas as pd
import os
from wbfm.utils.external.utils_pymc import grouped_cv_compare
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir

# Load data
data_dir = get_hierarchical_modeling_dir(gfp=False)
fname = os.path.join(data_dir, 'data.h5')
if not os.path.exists(fname):
    fname = os.path.join(data_dir, 'data_backup.h5')

print(f"Loading data from {fname}")
Xy = pd.read_hdf(fname)

# Test on a single neuron
neuron_name = 'VB02'
print(f"\nTesting grouped_cv_compare on {neuron_name}")
print(f"Dataset shape: {Xy.shape}")

# Run CV comparison for all three models
df_cv_compare, cv_results_dict = grouped_cv_compare(
    Xy, 
    neuron_name,
    dataset_name='all',
    residual_mode='pca_global',
    use_additional_eigenworms=True,
    models_to_compare=['null', 'nonhierarchical', 'hierarchical_pca'],
    DEBUG=True  # Set to True for quick debug run
)

if df_cv_compare is not None:
    print("\n" + "="*60)
    print("CV Comparison Results (az.compare format):")
    print("="*60)
    print(df_cv_compare)
    print("\nColumns:", df_cv_compare.columns.tolist())
    print("\nModel ranking (by ELPD-LOO):")
    for i, model in enumerate(df_cv_compare.index):
        print(f"  {i+1}. {model}: elpd_loo={df_cv_compare.loc[model, 'elpd_loo']:.2f}")
else:
    print("No results returned from grouped_cv_compare")

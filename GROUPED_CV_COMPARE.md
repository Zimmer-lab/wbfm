# Grouped Cross-Validation for Multiple Models

## Overview
Added `grouped_cv_compare()` function that runs grouped CV on multiple models (null, nonhierarchical, hierarchical_pca) and returns results in `az.compare()`-compatible format.

## Key Changes

### 1. Updated `grouped_cv_refitting()`
- **Added parameter**: `model_type` (str) — which model to fit ('null', 'nonhierarchical', 'hierarchical_pca')
- **Updated docstring** to document new parameter
- **Conditional model building**: 
  - `null`: Just intercept
  - `nonhierarchical`: Intercept + curvature term
  - `hierarchical_pca`: Intercept + sigmoid(PCA) × curvature (full model)
- **Branched posterior evaluation**: Each model type has appropriate extraction logic for its parameters

### 2. New Function: `grouped_cv_compare()`
**Purpose**: Run CV on multiple models and produce az.compare-compatible output
- **Inputs**: Xy, neuron_name, dataset_name, residual_mode, use_additional_eigenworms, models_to_compare, DEBUG
- **Outputs**: 
  - `df_cv_compare`: DataFrame from `az.compare()` with LOO metrics (elpd_loo, p_loo, elpd_diff, weight, etc.)
  - `cv_results_dict`: Dict mapping model names → cv_results for further analysis
- **Workflow**:
  1. Loop over each model in `models_to_compare`
  2. Call `grouped_cv_refitting()` with `model_type` set
  3. Compute LOO via `compute_cv_loo(normalize=True)`
  4. Collect all LOO results and call `az.compare()`
  5. Return standardized comparison DataFrame

## Usage

```python
from wbfm.utils.external.utils_pymc import grouped_cv_compare
import pandas as pd

# Load data
Xy = pd.read_hdf('path/to/data.h5')

# Compare all three models for a neuron
df_cv_compare, cv_results_dict = grouped_cv_compare(
    Xy, 
    neuron_name='VB02',
    dataset_name='all',
    models_to_compare=['null', 'nonhierarchical', 'hierarchical_pca'],
    DEBUG=False
)

# Results in az.compare format
print(df_cv_compare)  # Ranked by ELPD-LOO, includes p_loo, weight, etc.

# Access individual model CV results if needed
print(cv_results_dict['hierarchical_pca'])
```

## Output Format

The returned DataFrame matches `az.compare()` output:
```
                      rank  elpd_loo  p_loo  elpd_diff  weight  ...
hierarchical_pca         0    123.45   5.67       0.00    0.95
nonhierarchical          1    110.22   3.21      13.23    0.05
null                     2     50.00   1.00      73.45    0.00
```

Models are ranked by ELPD-LOO (higher is better). Weights sum to 1.0 and represent Bayesian model averaging weights.

## Implementation Details

- Uses per-posterior-sample log-likelihoods (shape: chain × draw × fold) so PSIS operates correctly
- Normalizes per fold by dividing by test set size for fair comparison across folds
- Gracefully skips models with failed CV runs
- Prints progress for each model as it runs

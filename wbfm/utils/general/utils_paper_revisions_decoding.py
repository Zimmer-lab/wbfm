"""
Reviewer request (Fig. 3, Bayesian model): "How does the GFP data perform for the analyses in Fig. 3? E.g. since you
can not obtain a true manifold, can you decode headswings from GFP data? And does the same pattern appear when
annotating fwd/backward from behavior, rather than PC1?"

The Bayesian model goes behavior -> neuron. Here we go the opposite direction: decode head swings from the whole
recorded population of each dataset, for GCaMP and GFP, using either raw traces or residuals after removing:
    - the global PCA manifold (2 modes; exactly the residual used in Fig. 3)
    - 2 CCA modes fit to continuous behaviors (NOTE: these behaviors include eigenworms 0-3, so this residual removes
      head-swing-aligned variance by construction; it is a conservative test, not a leaky one)
    - discrete, behaviorally-annotated states (rev, ventral/dorsal turn, pause, self-collision); this is the
      "annotate fwd/backward from behavior rather than PC1" version

See DECODING_METHODS below for the full methods text.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from wbfm.utils.general.utils_hardcoded import default_discrete_behaviors

RESIDUAL_MODES = {
    'Raw': None,
    'Residual (PCA)': 'pca_global',
    'Residual (CCA, continuous)': 'cca_continuous',
    'Residual (discrete behaviors)': 'discrete',
}
HEAD_SWING_TARGETS = ('eigenworm0', 'eigenworm1')
ALPHAS = np.logspace(-1, 5, 13)

DECODING_METHODS = """
Head-swing decoding from whole-brain activity.
Data. We used the same per-dataset tables as for the hierarchical Bayesian model (Fig. 3): {n_gcamp} freely moving
GCaMP datasets and {n_gfp} freely moving GFP datasets, sampled at ~3.47 volumes/s. Head swings were quantified as the
first two eigenworms (eigenworm 1 and 2, i.e. the dorso-ventral head-bending oscillation and its quadrature
component), each z-scored per dataset.

Neural features. For each dataset, all neurons (identified and unidentified) with <= {max_nan_pct}% missing values
were used (median {n_neurons_gcamp} neurons per GCaMP and {n_neurons_gfp} per GFP dataset); remaining gaps were
linearly interpolated. Four versions of the activity were decoded from:
 (1) Raw: the traces used throughout the paper.
 (2) Residual (PCA): each trace minus its reconstruction from the first two principal components of that dataset
     (the "manifold"), identical to the residual modeled in Fig. 3.
 (3) Residual (CCA, continuous): each trace minus its linear regression on the first two neural CCA modes, where CCA
     was fit between neural activity (first 3 PCs) and continuous behaviors (body speed, curvature measures and
     eigenworms 1-4). Because these behaviors include the eigenworms, this removes head-swing-aligned variance by
     construction and is therefore a conservative control.
 (4) Residual (discrete behaviors): each trace minus its linear regression on binary, behaviorally-annotated
     states (reversal, ventral turn, dorsal turn, pause, self-collision). Unlike (2), forward/backward state is taken
     from behavior annotation rather than PC1.
Regressions in (3) and (4) were fit separately per neuron within each dataset, since unidentified neurons cannot be
matched across datasets. None of the residualizations use the eigenworms directly except (3), as noted.

Decoder. For each dataset separately, eigenworm 1 and 2 were jointly predicted from the instantaneous (lag 0)
population activity with ridge regression, using all time points (no restriction to forward or reversal states).
Performance was assessed with blocked {n_folds}-fold cross-validation: the recording was split into {n_folds}
contiguous time blocks; each block was held out in turn, and the {gap} volumes (~{gap_s:.1f} s) on either side of the
held-out block were excluded from training to limit leakage through temporal autocorrelation. Within each training
set, features were z-scored using training statistics only, and the ridge penalty was selected from {n_alphas}
log-spaced values (10^{a_min:.0f}-10^{a_max:.0f}) by an inner blocked {n_folds}-fold cross-validation (with the same
gap) on the training data only. Decoding performance is the coefficient of determination (R^2) of the concatenated
held-out predictions, averaged over the two eigenworms.

Null distribution. To control for the autocorrelation of both signals, the analysis was repeated {n_shuffles} times
with the eigenworm time series circularly shifted relative to the neural data by a random offset between 10% and 90%
of the recording length. A dataset was considered to significantly decode head swings if its R^2 exceeded the
(1 - 0.05) quantile of its own null (one-sided permutation p-value, (1 + #null >= observed) / (1 + n_shuffles)).

Alignment control. Repeating the decoding with the neural data shifted by -8 to +12 volumes relative to behavior gave
maximal R^2 at lag 0 for both GCaMP and GFP, so instantaneous (lag 0) decoding was used for all analyses.

Group comparisons. Per-dataset R^2 values were compared between GCaMP and GFP with a two-sided Mann-Whitney U test,
separately for each type of activity.
"""


def _z_score(x):
    return (x - x.mean()) / x.std()


def _r2(y_true, y_pred):
    """R2 per column on all held-out predictions (concatenated across folds)"""
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - sse / sst


def blocked_folds(n, n_folds=5, gap=10):
    """Contiguous test blocks; training excludes `gap` samples on each side of the test block"""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    idx = np.arange(n)
    for i in range(n_folds):
        test = idx[edges[i]:edges[i + 1]]
        train = idx[(idx < edges[i] - gap) | (idx >= edges[i + 1] + gap)]
        yield train, test


def _ridge_predict_all_alphas(X_train, Y_train, X_test, alphas=ALPHAS):
    """Standardize on train, fit ridge for all alphas via one SVD; returns predictions (n_alphas, n_test, n_targets)"""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd == 0] = 1
    Xtr, Xte = (X_train - mu) / sd, (X_test - mu) / sd
    y_mu = Y_train.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ (Y_train - y_mu)
    XteV = Xte @ Vt.T
    return np.stack([XteV @ ((s / (s ** 2 + a))[:, None] * UtY) + y_mu for a in alphas])


def nested_cv_ridge_r2(X, Y, n_folds=5, gap=10, alphas=ALPHAS):
    """Outer blocked CV for evaluation, inner blocked CV (training data only) to choose the ridge penalty"""
    y_pred = np.zeros_like(Y)
    chosen_alphas = []
    for train, test in blocked_folds(len(Y), n_folds, gap):
        # Inner CV on the training set only (indices relative to the training set)
        Xtr, Ytr = X[train], Y[train]
        inner_sse = np.zeros(len(alphas))
        for itr, ite in blocked_folds(len(Ytr), n_folds, gap):
            pred = _ridge_predict_all_alphas(Xtr[itr], Ytr[itr], Xtr[ite], alphas)
            inner_sse += ((pred - Ytr[ite][None]) ** 2).sum(axis=(1, 2))
        best = int(np.argmin(inner_sse))
        chosen_alphas.append(alphas[best])
        y_pred[test] = _ridge_predict_all_alphas(Xtr, Ytr, X[test], alphas[best:best + 1])[0]
    return _r2(Y, y_pred), chosen_alphas


def population_residuals(Xy_dataset, neuron_cols, residual_mode):
    """Raw or residual traces for all neurons of one dataset (see module docstring)"""
    X = Xy_dataset[neuron_cols]
    if residual_mode is None:
        return X
    if residual_mode == 'pca_global':
        return X - Xy_dataset[[f'{c}_manifold' for c in neuron_cols]].values
    if residual_mode == 'cca_continuous':
        Z = Xy_dataset[['CCA_neural_mode_1', 'CCA_neural_mode_2']].values
    elif residual_mode == 'discrete':
        Z = Xy_dataset[default_discrete_behaviors()].values.astype(float)
    else:
        raise NotImplementedError(residual_mode)
    out = X.copy()
    for c in neuron_cols:
        y = X[c].values
        m = ~np.isnan(y)
        out.loc[m, c] = y[m] - LinearRegression().fit(Z[m], y[m]).predict(Z[m])
    return out


def neuron_columns(Xy):
    """All neural trace columns (identified and unidentified), excluding manifold and behavior columns"""
    cols = list(Xy.columns)
    beh_cols = set(cols[cols.index('curvature_vb02'):])
    return [c for c in cols if c not in beh_cols and c not in ('dataset_name', 'local_time')
            and not c.endswith('_manifold') and not c.endswith('_manifold1')]


def get_dataset_features(df_dataset, neuron_cols_all, residual_mode, max_nan_fraction=0.05):
    df_dataset = df_dataset.sort_values('local_time')
    nan_frac = df_dataset[neuron_cols_all].isna().mean()
    neuron_cols = list(nan_frac[nan_frac <= max_nan_fraction].index)
    X = population_residuals(df_dataset, neuron_cols, residual_mode)
    X = X.interpolate(limit_direction='both').values
    Y = np.column_stack([_z_score(df_dataset[t].values) for t in HEAD_SWING_TARGETS])
    return X, Y


def decode_one_dataset(X, Y, n_shuffles=100, n_folds=5, gap=10, seed=0):
    rng = np.random.default_rng(seed)
    r2, alphas = nested_cv_ridge_r2(X, Y, n_folds, gap)
    n = len(Y)
    shifts = rng.integers(int(0.1 * n), int(0.9 * n), size=n_shuffles)
    null = np.array([nested_cv_ridge_r2(X, np.roll(Y, s, axis=0), n_folds, gap)[0].mean() for s in shifts])
    out = dict(r2=r2.mean(), **{f'r2_{t}': v for t, v in zip(HEAD_SWING_TARGETS, r2)},
               r2_null_mean=null.mean(), r2_null_95=np.quantile(null, 0.95),
               p_shift=(1 + np.sum(null >= r2.mean())) / (1 + n_shuffles),
               median_alpha=np.median(alphas), n_neurons=X.shape[1], n_samples=n)
    return out


def run_population_decoding(Xy_dict, n_shuffles=100, n_folds=5, gap=10, max_nan_fraction=0.05, n_jobs=-1):
    """
    Xy_dict: datatype name -> Xy dataframe, e.g. {'GCaMP': Xy, 'GFP': Xy_gfp}

    Returns a long dataframe with one row per (datatype, residual mode, dataset)
    """
    all_dfs = []
    for datatype, Xy in Xy_dict.items():
        cols = neuron_columns(Xy)
        for mode_name, residual_mode in RESIDUAL_MODES.items():
            print(f"Decoding head swings: {datatype}, {mode_name}")
            groups = list(Xy.groupby('dataset_name'))

            def _run(i, dataset_name, df_dataset):
                X, Y = get_dataset_features(df_dataset, cols, residual_mode, max_nan_fraction)
                res = decode_one_dataset(X, Y, n_shuffles=n_shuffles, n_folds=n_folds, gap=gap, seed=i)
                return dict(dataset_name=dataset_name, **res)

            rows = Parallel(n_jobs=n_jobs)(delayed(_run)(i, n, d) for i, (n, d) in enumerate(groups))
            all_dfs.append(pd.DataFrame(rows).assign(datatype=datatype, residual_mode=mode_name))
    return pd.concat(all_dfs, ignore_index=True)

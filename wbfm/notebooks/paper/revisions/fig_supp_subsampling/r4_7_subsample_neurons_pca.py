"""R4.7: robustness of the PCA manifold and Fig 3 PC1-weight categories to neuron subsampling.

Usage: python -I step2_analysis.py cache.pkl out_dir
"""
import sys, os, pickle, time
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/charles/Current_work/repos/dlc_for_wbfm')
from wbfm.utils.external.utils_pandas import combine_columns_with_suffix
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
from wbfm.utils.general.utils_paper import calc_p_values_for_pca_weights

cache_fname, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
with open(cache_fname, 'rb') as f:
    cache = pickle.load(f)

FRACS_MANIFOLD = [0.5, 0.6, 0.7, 0.8, 0.9]
N_REP_MANIFOLD = 50
FRACS_CAT = [0.5, 0.7, 0.9]
N_REP_CAT = 20
rng = np.random.default_rng(0)


def svd_centered(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, U, S, Vt


def dim_stats(S):
    lam = S ** 2
    frac = np.cumsum(lam) / lam.sum()
    return dict(var_top2=frac[1], pr=lam.sum() ** 2 / (lam ** 2).sum(),
                n50=int(np.searchsorted(frac, 0.5) + 1), n80=int(np.searchsorted(frac, 0.8) + 1))


def is_identified(name):
    return 'neuron' not in name


# ---------------------------------------------------------------- Part 1: manifold (Bayesian input) + dimensionality
rows = []
sanity = []
for (dtype, name), d in cache.items():
    X = d['X']
    Xv = X.values.astype(float)
    N = Xv.shape[1]
    Xc, U, S, Vt = svd_centered(Xv)
    U2 = U[:, :2]
    # Sanity: my PCA == paper calc_pca_modes (uncombined, used for the Bayesian manifold)
    sanity.append(dict(dtype=dtype, name=name, N=N,
                       c_pc1_vs_paper=abs(np.corrcoef(U[:, 0], d['m_raw'].iloc[:, 0])[0, 1]),
                       c_pc2_vs_paper=abs(np.corrcoef(U[:, 1], d['m_raw'].iloc[:, 1])[0, 1])))
    ids = [i for i, c in enumerate(X.columns) if is_identified(c)]
    R_full = Xc[:, ids] - U2 @ (U2.T @ Xc[:, ids])
    rows.append(dict(dtype=dtype, name=name, N=N, frac=1.0, rep=0, c_pc1=1, c_pc2=1, subspace_min=1, subspace_mean=1,
                     resid_corr_median=1, **dim_stats(S)))
    for frac in FRACS_MANIFOLD:
        k = int(round(frac * N))
        for rep in range(N_REP_MANIFOLD):
            sub = np.sort(rng.choice(N, size=k, replace=False))
            _, Us, Ss, _ = svd_centered(Xv[:, sub])
            Us2 = Us[:, :2]
            cos = np.linalg.svd(Us2.T @ U2, compute_uv=False)
            R_sub = Xc[:, ids] - Us2 @ (Us2.T @ Xc[:, ids])
            a = R_sub - R_sub.mean(0); b = R_full - R_full.mean(0)
            rc = (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
            rows.append(dict(dtype=dtype, name=name, N=N, frac=frac, rep=rep,
                             c_pc1=abs(np.corrcoef(Us[:, 0], U[:, 0])[0, 1]),
                             c_pc2=abs(np.corrcoef(Us[:, 1], U[:, 1])[0, 1]),
                             subspace_min=cos.min(), subspace_mean=cos.mean(),
                             resid_corr_median=np.nanmedian(rc), **dim_stats(Ss)))
    print(dtype, name, N, len(ids), flush=True)
df_man = pd.DataFrame(rows)
df_man.to_csv(os.path.join(out_dir, 'manifold_subsampling.csv'), index=False)
pd.DataFrame(sanity).to_csv(os.path.join(out_dir, 'sanity_pca_vs_paper.csv'), index=False)
print(pd.DataFrame(sanity).describe(), flush=True)


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
            rc = [np.corrcoef(R_sub[:, j], R_full[:, j])[0, 1] for j in range(len(ids))]
            rows.append(dict(dtype=dtype, name=name, N=N, frac=frac, rep=rep,
                             c_pc1=abs(np.corrcoef(Us[:, 0], U[:, 0])[0, 1]),
                             c_pc2=abs(np.corrcoef(Us[:, 1], U[:, 1])[0, 1]),
                             subspace_min=cos.min(), subspace_mean=cos.mean(),
                             resid_corr_median=np.nanmedian(rc), **dim_stats(Ss)))
df_man = pd.DataFrame(rows)
df_man.to_csv(os.path.join(out_dir, 'manifold_subsampling.csv'), index=False)
pd.DataFrame(sanity).to_csv(os.path.join(out_dir, 'sanity_pca_vs_paper.csv'), index=False)
print(pd.DataFrame(sanity).describe(), flush=True)

# ---------------------------------------------------------------- Part 2: Fig 3 PC1-weight categories
confident = set(neurons_with_confident_ids(combine_left_right=True))


def comb(X):
    return combine_columns_with_suffix(X, suffixes=['L', 'R'], how='mean')


full_info = {}
for key, d in cache.items():
    Xcomb = comb(d['X'])
    Xc, U, S, Vt = svd_centered(Xcomb.values.astype(float))
    w = pd.Series(Vt[0], index=Xcomb.columns)
    ref = d['w_comb'].iloc[:, 0].reindex(w.index)
    sign = np.sign(np.corrcoef(w.values, ref.values)[0, 1])
    full_info[key] = dict(Xcomb=Xcomb, Xc=Xc, u1=U[:, 0] * sign, w=w * sign,
                          c_vs_paper=abs(np.corrcoef(w.values, ref.values)[0, 1]))
print('Fig3 full weights vs paper: min |corr| =', min(v['c_vs_paper'] for v in full_info.values()), flush=True)


def to_df(weights_by_key, dtype):
    df = pd.DataFrame({k[1]: v for k, v in weights_by_key.items() if k[0] == dtype}).T
    df = df.loc[:, df.columns.isin(confident)]
    df = df.loc[:, ~df.columns.str.contains('neuron')]
    return df


# sign correction as in calc_pca_weights_for_all_projects (top weight positive), computed once on full data
full_w = {k: v['w'] for k, v in full_info.items()}
sign_vec = {}
for dtype in ['gcamp', 'immob']:
    df = to_df(full_w, dtype)
    s = np.sign(df.iloc[:, df.abs().sum().argmax()])
    sign_vec[dtype] = s


def categories(weights_by_key, seed):
    np.random.seed(seed)  # scipy permutation test uses the global RNG
    dfs = {}
    for dtype in ['gcamp', 'immob']:
        df = to_df(weights_by_key, dtype)
        df = df.mul(sign_vec[dtype].reindex(df.index), axis='index')
        dfs[dtype] = df.reindex(df.median().sort_values(ascending=False).index, axis=1)
    df_both, _, _ = calc_p_values_for_pca_weights(dfs['gcamp'], dfs['immob'])
    return df_both.groupby('neuron_name')['Result_simple'].first()


cat_rows = []
t0 = time.time()
for seed in range(5):
    c = categories(full_w, seed)
    cat_rows.append(pd.DataFrame(dict(variant='full', frac=1.0, rep=seed, neuron=c.index, cat=c.values)))
print('full categories x5 done', time.time() - t0, flush=True)

for frac in FRACS_CAT:
    for rep in range(N_REP_CAT):
        w_drop, w_proj = {}, {}
        for key, d in cache.items():
            X = d['X']
            N = X.shape[1]
            sub = np.sort(rng.choice(N, size=int(round(frac * N)), replace=False))
            Xs = comb(X.iloc[:, sub])
            _, Us, Ss, Vts = svd_centered(Xs.values.astype(float))
            fi = full_info[key]
            sign = np.sign(np.corrcoef(Us[:, 0], fi['u1'])[0, 1])
            u1 = Us[:, 0] * sign
            # (a) dropped neurons are missing; weights only for sampled neurons
            w_drop[key] = pd.Series(Vts[0] * sign, index=Xs.columns)
            # (b) manifold from the subset; every neuron's weight = projection onto the subset PC1 (unit norm, as PCA loadings)
            wp = fi['Xc'].T @ u1
            w_proj[key] = pd.Series(wp / np.linalg.norm(wp), index=fi['Xcomb'].columns)
        for variant, wd in [('drop', w_drop), ('project', w_proj)]:
            c = categories(wd, 1000 + rep)
            cat_rows.append(pd.DataFrame(dict(variant=variant, frac=frac, rep=rep, neuron=c.index, cat=c.values)))
        print(frac, rep, time.time() - t0, flush=True)
df_cat = pd.concat(cat_rows)
df_cat.to_csv(os.path.join(out_dir, 'categories_subsampling.csv'), index=False)
print('done')

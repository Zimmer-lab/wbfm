"""R4.7: paper's intrinsic-dimensionality estimators (fig_2_intrinsic_dimensionality.ipynb) vs neuron sampling fraction.

Same estimators and defaults as the paper (skdim CorrInt, FisherS, MADA, MOM, TLE), applied to
calc_default_traces(use_paper_options=True, interpolate_nan=True) restricted to random subsets of neurons.
Also recomputes the PCA-based measures for all three data types.

Usage: python -I part2_skdim.py skdim_path out_dir cache1.pkl [cache2.pkl ...]
"""
import os, sys, pickle, warnings, zlib
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, sys.argv[1])
import skdim

out_dir = sys.argv[2]
CACHE = {}
for fname in sys.argv[3:]:
    with open(fname, 'rb') as f:
        CACHE.update({k: v['X'] for k, v in pickle.load(f).items()})

METHODS = [skdim.id.CorrInt, skdim.id.FisherS, skdim.id.MADA, skdim.id.MOM, skdim.id.TLE]
FRACS = [0.5, 0.6, 0.7, 0.8, 0.9]
N_REP = 20


def pca_stats(Xv):
    Xc = Xv - Xv.mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    cum = np.cumsum(lam) / lam.sum()
    return dict(pca_var_top2=cum[1], pca_pr=lam.sum() ** 2 / (lam ** 2).sum(),
                pca_n50=int(np.searchsorted(cum, 0.5) + 1), pca_n80=int(np.searchsorted(cum, 0.8) + 1))


def run(task):
    key, frac, rep = task
    warnings.filterwarnings('ignore')
    X = CACHE[key]
    N = X.shape[1]
    if frac == 1.0:
        sub = np.arange(N)
    else:
        rng = np.random.default_rng(zlib.crc32(f"{key}_{frac}_{rep}".encode()))
        sub = np.sort(rng.choice(N, size=int(round(frac * N)), replace=False))
    Xs = X.iloc[:, sub]
    row = dict(dtype=key[0], name=key[1], N=N, n_sampled=len(sub), frac=frac, rep=rep)
    for m in METHODS:
        try:
            row[m.__name__] = float(m().fit(Xs).dimension_)
        except ValueError:  # as in the paper notebook
            row[m.__name__] = np.nan
    row.update(pca_stats(Xs.values.astype(float)))
    return row


if __name__ == '__main__':
    tasks = [(k, 1.0, 0) for k in CACHE] + [(k, f, r) for k in CACHE for f in FRACS for r in range(N_REP)]
    with Pool(40) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run, tasks, chunksize=4)):
            rows.append(row)
            if i % 500 == 0:
                print(i, len(tasks), flush=True)
    df = pd.DataFrame(rows)
    df['ID_median_across_methods'] = df[[m.__name__ for m in METHODS]].median(axis=1)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'dimensionality_subsampling.csv'), index=False)
    print('done', len(df))

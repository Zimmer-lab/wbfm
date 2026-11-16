"""Cache paper traces + reference PCA outputs for the R4.7 subsampling check."""
import sys, pickle
sys.path.insert(0, '/home/charles/Current_work/repos/dlc_for_wbfm')
from wbfm.utils.general.utils_hardcoded import load_paper_datasets

out = {}
for dtype, key in [('gcamp', ['gcamp', 'hannah_O2_fm']), ('immob', 'immob')]:
    projs = load_paper_datasets(key)
    for name, p in projs.items():
        X = p.calc_default_traces(use_paper_options=True, interpolate_nan=True, rename_neurons_using_manual_ids=True)
        # Reference outputs exactly as used in the paper
        w_comb, m_comb, ev_comb, _ = p.calc_pca_modes(use_paper_options=True, combine_left_right=True)
        w_raw, m_raw, ev_raw, _ = p.calc_pca_modes(use_paper_options=True, n_components=2)
        out[(dtype, name)] = dict(X=X, w_comb=w_comb, m_comb=m_comb, ev_comb=ev_comb, m_raw=m_raw, w_raw=w_raw)
        print(dtype, name, X.shape, flush=True)

with open(sys.argv[1], 'wb') as f:
    pickle.dump(out, f)
print('done', len(out))

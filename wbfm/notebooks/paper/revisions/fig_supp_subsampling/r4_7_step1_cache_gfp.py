import sys, pickle
sys.path.insert(0, '/home/charles/Current_work/repos/dlc_for_wbfm')
from wbfm.utils.general.utils_hardcoded import load_paper_datasets
out = {}
for name, p in load_paper_datasets('gfp').items():
    X = p.calc_default_traces(use_paper_options=True, interpolate_nan=True)
    out[('gfp', name)] = dict(X=X)
    print('gfp', name, X.shape, flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
print('done', len(out))

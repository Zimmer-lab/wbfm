# %% [markdown]
# # Baseline vs. oscillation control (Reviewer 2: detection effects)
#
# Question: could oscillations be harder to detect on a low baseline of indicator activity, so that the
# "hierarchically gated" neurons of Fig. 5 only *appear* not to oscillate in their off state?
#
# Per neuron x dataset x action state (forward / backward crawling, from the behavior annotation in data.h5):
# 1. Baseline: median raw GREEN intensity (same source as Supplementary Fig. S4f; exported by
#    `fig_supp_baseline_extract_green.py`, which must be run first in the `wbfm` env).
# 2. Oscillation amplitude from the hierarchical Bayesian model: gamma_j + beta_j * <s_j>_state, where
#    s_j = sigmoid(A1_j P1 + A2_j P2 - kappa_j) is recomputed from posterior medians and the PCA modes in data.h5.
# 3. Dynamic-range position: percentile of the off-state median within the on-state green distribution.
#
# Run in the `pymc_311` env (arviz/xarray).

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path(os.getcwd())
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
from wbfm.utils.general.utils_paper import apply_figure_settings

OUT_DIR = HERE / 'fig_supp_baseline'
OUT_DIR.mkdir(exist_ok=True)
TRACE_DIR = Path(get_hierarchical_modeling_dir()) / 'output_pca_global'

MIN_FRAMES_PER_STATE = 100   # per neuron x dataset x state (~30 s at the volume rate)
N_DRAWS_CHECK = 300          # posterior draws used for the draw-wise check of the amplitude
RNG = np.random.default_rng(42)

# %% [markdown]
# ## Neuron groups (from the Fig. 5 notebook's df_params, column Category_new)

# %%
df_params = pd.read_hdf(HERE / 'bayesian_alt_manifold' / 'df_params.h5')  # saved by fig_bayesian_ALT_CONDA.ipynb
df_params = df_params[df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)'].set_index('neuron_name')

GROUP_NAMES = {'Hierarchy > GFP and PC1 is significant': 'Gated',
               'Behavior is above GFP': 'Always oscillating',
               'Not above artifacts': 'Not oscillating'}
group_of_neuron = df_params['Category_new'].map(GROUP_NAMES)

expected = {
    'Gated': {'RMDVL', 'RMED', 'RMDDL', 'SMDDR', 'RMEV', 'RMDDR', 'VB03', 'VB02', 'DB01'},
    'Always oscillating': {'RMDVR', 'URADR', 'SMDDL', 'URXL', 'DA01', 'RIAR', 'SMDVL', 'SIADR', 'DB02', 'DD01'},
}
for g, names in expected.items():
    found = set(group_of_neuron[group_of_neuron == g].index)
    assert found == names, f"{g}: df_params gives {sorted(found)}, expected {sorted(names)}"
print(group_of_neuron.value_counts())

# Colors as in Fig. 5b
cmap = px.colors.qualitative.D3
GROUP_COLORS = {'Gated': cmap[0], 'Always oscillating': cmap[2], 'Not oscillating': cmap[7]}
GROUP_ORDER = ['Gated', 'Always oscillating', 'Not oscillating']

# %% [markdown]
# ## Data: behavior / PCA modes (data.h5) and raw green traces (projects)

# %%
Xy = pd.read_hdf(os.path.join(get_hierarchical_modeling_dir(), 'data.h5'))
df_green = pd.read_hdf(OUT_DIR / 'green_traces.h5')    # paper green trace: SUM over the neuron's voxels (minus background)
df_vol = pd.read_hdf(OUT_DIR / 'volume_traces.h5')     # segmented volume (voxels) per neuron and frame
df_ratio = pd.read_hdf(OUT_DIR / 'ratio_traces.h5')
key = ['dataset_name', 'local_time']

# Brightness = mean intensity per voxel above background = integrated green / segmented volume (per frame).
# Otherwise a large dim neuron would read as bright. Frames without a volume (tracking gaps, where the paper trace is
# interpolated) use the median volume of that neuron in that dataset.
assert (df_vol.groupby('dataset_name').size() == df_green.groupby('dataset_name').size()).all(), "Frame counts differ"
green_neurons = [c for c in df_green.columns if c not in key + ['local_time_physical']]
_g = df_green.set_index(key)[green_neurons]
_v = df_vol.set_index(key).reindex(index=_g.index, columns=green_neurons)
_v = _v.where(_v > 0)
missing_vol = [n for n in green_neurons if _v[n].isna().all()]
print(f"Neurons without any volume values: {len(missing_vol)} {missing_vol[:10]}")
print(f"Fraction of frames with a volume, over neurons with a green trace: {_v.notna().values[_g.notna().values].mean():.3f}")
_v = _v.groupby(level='dataset_name').transform(lambda c: c.fillna(c.median()))
df_green_sum = df_green
df_green_mean = (_g / _v).reset_index()

# Action states: forward / backward crawling, excluding turns and pauses (from the behavior annotation)
no_turn_pause = (Xy['ventral_turn'] == 0) & (Xy['dorsal_turn'] == 0) & (Xy['pause'] == 0)
Xy['state'] = np.select([(Xy['rev'] == 0) & no_turn_pause, (Xy['rev'] == 1) & no_turn_pause],
                        ['forward', 'backward'], default='excluded')
print(Xy['state'].value_counts())

# %% [markdown]
# ### Alignment check between data.h5 and the re-loaded project traces

# %%
n_xy = Xy.groupby('dataset_name').size()
n_green = df_green.groupby('dataset_name').size()
align = pd.DataFrame({'n_frames_data_h5': n_xy, 'n_frames_green': n_green})
assert set(n_xy.index) == set(n_green.index), "Dataset names differ between data.h5 and the projects"
print(f"{len(n_xy)} datasets; frames data.h5 <= green in all: {(align.n_frames_data_h5 <= align.n_frames_green).all()}")
print(align[align.n_frames_data_h5 != align.n_frames_green])

# (a) The ratio trace columns of data.h5 should be identical to the re-loaded paper ratio traces at the same keys
_m = Xy[key + ['local_time_physical']].merge(df_ratio, on=key, how='left', suffixes=('', '_proj'))
neurons_common = [n for n in group_of_neuron.index if n in Xy.columns and n in df_ratio.columns]
max_abs_diff = {}
for n in neurons_common:
    a, b = Xy[n].values, _m[n].values
    ok = ~np.isnan(a) & ~np.isnan(b)
    max_abs_diff[n] = np.nanmax(np.abs(a[ok] - b[ok])) if ok.any() else np.nan
print(f"Max |data.h5 - reloaded ratio| over {len(neurons_common)} neurons: {np.nanmax(list(max_abs_diff.values())):.2e}")
# (b) physical time stamps agree
dt = np.nanmax(np.abs(_m['local_time_physical'] - _m['local_time_physical_proj']))
print(f"Max |physical time difference|: {dt:.2e} s")

# (c) Green vs. data.h5 ratio: cross-correlation peaks at lag 0 (green rises with calcium)
_g = Xy[key].merge(df_green, on=key, how='left')
lags = np.arange(-5, 6)
best_lag = []
for n in ['VB02', 'DB01', 'DD01', 'AVAL', 'AVBL', 'RMDVL', 'SMDDL']:
    for ds, idx in Xy.groupby('dataset_name').groups.items():
        r, g = Xy.loc[idx, n].values, _g.loc[idx, n].values if n in _g else None
        if g is None or np.isnan(g).all() or np.isnan(r).mean() > 0.5:
            continue
        cc = [pd.Series(r).corr(pd.Series(g).shift(l)) for l in lags]
        best_lag.append(dict(neuron=n, dataset=ds, best_lag=lags[int(np.nanargmax(cc))], r0=cc[5]))
best_lag = pd.DataFrame(best_lag)
print(f"Green-vs-ratio best lag (frames) over {len(best_lag)} neuron/dataset pairs:")
print(best_lag['best_lag'].value_counts().sort_index())
print(f"Median correlation at lag 0: {best_lag['r0'].median():.2f}")
align_summary = dict(n_datasets=len(n_xy), max_ratio_diff=np.nanmax(list(max_abs_diff.values())), max_time_diff=dt,
                     frac_best_lag_0=(best_lag['best_lag'] == 0).mean(), median_r0=best_lag['r0'].median())

# %% [markdown]
# ## Per neuron x dataset x state: green baseline and Bayesian amplitude

# %%
CURV = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def per_dataset_state_table(neuron):
    """Green baseline, gate and amplitude per dataset and state for one neuron"""
    fname = TRACE_DIR / f'{neuron}_hierarchical_pca_trace.nc'
    post = xr.open_dataset(fname, group='posterior')
    # Same frames and z-scoring as used to fit the model
    df_model = get_dataframe_for_single_neuron(Xy, neuron, curvature_terms=CURV, residual_mode='pca_global',
                                               verbose=0)
    assert len(df_model) == post.sizes['pca_term_dim_0'], f"{neuron}: frame count differs from the fit"
    ds_idx, ds_names = df_model['dataset_name'].factorize()
    assert list(ds_names) == list(post['dataset_name'].values), f"{neuron}: dataset order differs from the fit"

    P = df_model[['x_pca0', 'x_pca1']].values
    A = post['pca_amplitudes_normalized'].values            # chain, draw, 2, n_ds
    kappa = post['inflection'].values                       # chain, draw, n_ds
    beta = post['beta'].values
    gamma = post['gamma'].values
    flat = lambda v: v.reshape((-1,) + v.shape[2:])
    A, kappa, beta, gamma = flat(A), flat(kappa), flat(beta), flat(gamma)
    post.close()

    # Primary: posterior medians of the per-dataset parameters
    A_med, kappa_med = np.median(A, axis=0), np.median(kappa, axis=0)
    beta_med, gamma_med = np.median(beta, axis=0), np.median(gamma, axis=0)
    s_med = _sigmoid(A_med[0, ds_idx] * P[:, 0] + A_med[1, ds_idx] * P[:, 1] - kappa_med[ds_idx])

    # Check: same quantity computed per posterior draw, then median
    draws = RNG.choice(A.shape[0], size=N_DRAWS_CHECK, replace=False)
    s_draws = _sigmoid(A[draws][:, 0, ds_idx] * P[:, 0] + A[draws][:, 1, ds_idx] * P[:, 1] - kappa[draws][:, ds_idx])

    states = Xy.loc[df_model.index, 'state'].values
    local_time = Xy.loc[df_model.index, 'local_time'].values
    green_lookup = df_green_mean.set_index(key)[neuron] if neuron in df_green_mean.columns else None
    sum_lookup = df_green_sum.set_index(key)[neuron] if neuron in df_green_sum.columns else None

    rows = []
    for j, ds in enumerate(ds_names):
        in_ds = ds_idx == j
        _idx = pd.MultiIndex.from_arrays([np.repeat(ds, in_ds.sum()), local_time[in_ds]])
        g_all = green_lookup.reindex(_idx).values if green_lookup is not None else np.full(in_ds.sum(), np.nan)
        gsum_all = sum_lookup.reindex(_idx).values if sum_lookup is not None else np.full(in_ds.sum(), np.nan)
        st_ds = states[in_ds]
        for state in ['forward', 'backward']:
            m = st_ds == state
            m_full = np.zeros(len(df_model), bool)
            m_full[np.where(in_ds)[0][m]] = True
            g = g_all[m]
            amp_draws = gamma[draws, j] + beta[draws, j] * s_draws[:, m_full].mean(axis=1)
            rows.append(dict(
                neuron=neuron, dataset_name=ds, state=state, n_frames=int(m.sum()),
                n_green_valid=int(np.isfinite(g).sum()),
                green_median=np.nanmedian(g) if np.isfinite(g).any() else np.nan,
                green_values=g[np.isfinite(g)],
                green_sum_median=np.nanmedian(gsum_all[m]) if np.isfinite(gsum_all[m]).any() else np.nan,
                gate_mean=s_med[m_full].mean() if m.any() else np.nan,
                gamma=gamma_med[j], beta=beta_med[j],
                amplitude=gamma_med[j] + beta_med[j] * s_med[m_full].mean() if m.any() else np.nan,
                amplitude_drawwise=np.median(amp_draws) if m.any() else np.nan,
                amplitude_full_open=gamma_med[j] + beta_med[j],
            ))
    return rows


all_rows = []
for neuron in tqdm(group_of_neuron.index):
    if not (TRACE_DIR / f'{neuron}_hierarchical_pca_trace.nc').exists():
        print(f"No trace for {neuron}")
        continue
    all_rows.extend(per_dataset_state_table(neuron))
df_long = pd.DataFrame(all_rows)
df_long['group'] = df_long['neuron'].map(group_of_neuron)
print(df_long.shape)

# %% [markdown]
# ### Which state is "on" for the gated neurons? (sign of the PC1 gating coefficient vs. empirical gate)

# %%
gate_by_state = df_long.groupby(['neuron', 'state'])['gate_mean'].median().unstack()
on_state = pd.DataFrame({
    'group': group_of_neuron,
    'hyper_pca0': df_params['hyper_pca0'],
    'gate_forward': gate_by_state['forward'],
    'gate_backward': gate_by_state['backward'],
})
on_state['on_state_from_sign'] = np.where(on_state['hyper_pca0'] < 0, 'forward', 'backward')  # PC1 is higher in reversals
on_state['on_state_empirical'] = np.where(on_state['gate_forward'] > on_state['gate_backward'], 'forward', 'backward')
pc1_rev = Xy.groupby('dataset_name').apply(lambda d: d.loc[d.rev == 1, 'pca_0'].mean() - d.loc[d.rev == 0, 'pca_0'].mean())
print(f"PC1 (pca_0) higher during reversals in {(pc1_rev > 0).sum()}/{len(pc1_rev)} datasets")
print(on_state[on_state.group == 'Gated'])
assert (on_state.loc[on_state.group == 'Gated', 'on_state_from_sign'] ==
        on_state.loc[on_state.group == 'Gated', 'on_state_empirical']).all()

# %% [markdown]
# ### Frame threshold, on/off assignment, dynamic-range percentile

# %%
ok = (df_long['n_frames'] >= MIN_FRAMES_PER_STATE) & (df_long['n_green_valid'] >= MIN_FRAMES_PER_STATE)
# Keep only neuron x dataset pairs where BOTH states pass the threshold (paired design)
pair_ok = ok.groupby([df_long['neuron'], df_long['dataset_name']]).transform('all')
n_excluded_pairs = (~pair_ok).groupby([df_long['neuron'], df_long['dataset_name']]).first().sum()
df = df_long[pair_ok].copy()
print(f"Excluded {n_excluded_pairs} neuron x dataset pairs with < {MIN_FRAMES_PER_STATE} frames in a state")

# Class-level medians across datasets
cls = df.groupby(['neuron', 'state']).agg(green=('green_median', 'median'), green_sum=('green_sum_median', 'median'),
                                          amplitude=('amplitude', 'median'),
                                          amplitude_drawwise=('amplitude_drawwise', 'median'),
                                          gate=('gate_mean', 'median'), n_datasets=('dataset_name', 'nunique'))
cls = cls.reset_index()
cls['group'] = cls['neuron'].map(group_of_neuron)

# On/off: gated -> empirical gate; others -> off = state with the lower class-level median green
lower_green_state = cls.pivot(index='neuron', columns='state', values='green').idxmin(axis=1)
on_off = {}
for n in cls['neuron'].unique():
    if group_of_neuron[n] == 'Gated':
        on = on_state.loc[n, 'on_state_empirical']
    else:
        on = 'backward' if lower_green_state[n] == 'forward' else 'forward'
    on_off[n] = on
df['on_state'] = df['neuron'].map(on_off)
df['role'] = np.where(df['state'] == df['on_state'], 'on', 'off')
cls['on_state'] = cls['neuron'].map(on_off)
cls['role'] = np.where(cls['state'] == cls['on_state'], 'on', 'off')

# Measure 3: percentile of the off-state median within the on-state green distribution (same neuron/dataset)
pct_rows = []
for (n, ds), d in df.groupby(['neuron', 'dataset_name']):
    on_vals = d.loc[d.role == 'on', 'green_values'].iloc[0]
    off_med = d.loc[d.role == 'off', 'green_median'].iloc[0]
    pct_rows.append(dict(neuron=n, dataset_name=ds, group=group_of_neuron[n],
                         off_percentile_in_on=stats.percentileofscore(on_vals, off_med, kind='mean'),
                         on_p5=np.percentile(on_vals, 5), on_p95=np.percentile(on_vals, 95), off_median=off_med))
df_pct = pd.DataFrame(pct_rows)
cls_pct = df_pct.groupby('neuron').agg(off_percentile_in_on=('off_percentile_in_on', 'median'),
                                       frac_datasets_off_within_on_5_95=(
                                           'off_median', lambda x: np.nan)).reset_index()
_within = (df_pct['off_median'] >= df_pct['on_p5']) & (df_pct['off_median'] <= df_pct['on_p95'])
cls_pct['frac_datasets_off_within_on_5_95'] = cls_pct['neuron'].map(_within.groupby(df_pct['neuron']).mean())
cls_pct['group'] = cls_pct['neuron'].map(group_of_neuron)

# %% [markdown]
# ### Consistency with Fig. 5d (radial summary exp(m_gamma) + exp(m_beta))

# %%
check = df.groupby('neuron').agg(full_open_median=('amplitude_full_open', 'median'),
                                 gamma_median=('gamma', 'median'))
check['fig5d_r_gated'] = np.exp(df_params['log_hyper_beta']) + np.exp(df_params['log_amplitude_mu'])
check['exp_m_gamma'] = np.exp(df_params['log_amplitude_mu'])
# Radius actually plotted in Fig. 5d for 'Behavior is above GFP' neurons (notebook cell: exp(m_gamma) + 0.5*exp(m_gamma))
check['fig5d_r_plotted'] = np.where(check.index.map(group_of_neuron) == 'Always oscillating', 1.5 * check['exp_m_gamma'], check['fig5d_r_gated'])
check['group'] = check.index.map(group_of_neuron)
amp_wide = cls.pivot(index='neuron', columns='role', values='amplitude')
check['amp_on'], check['amp_off'] = amp_wide['on'], amp_wide['off']
check['ratio_full_open_vs_fig5d'] = check['full_open_median'] / check['fig5d_r_gated']
_dw = cls.pivot(index='neuron', columns='role', values='amplitude_drawwise')
check['max_rel_diff_drawwise'] = (np.abs(_dw - amp_wide) / amp_wide).max(axis=1)
print(check[check.group != 'Not oscillating'].round(3).sort_values('group'))
print(f"Spearman(full-open per-dataset median, Fig. 5d r) over gated+always: "
      f"{stats.spearmanr(*check.loc[check.group != 'Not oscillating', ['full_open_median', 'fig5d_r_gated']].values.T)}")

# %% [markdown]
# ## Statistics

# %%
def rank_biserial_paired(diff):
    diff = np.asarray(diff)
    diff = diff[diff != 0]
    r = stats.rankdata(np.abs(diff))
    return (r[diff > 0].sum() - r[diff < 0].sum()) / r.sum()


def rank_biserial_mwu(x, y):
    u = stats.mannwhitneyu(x, y, alternative='two-sided').statistic
    return 2 * u / (len(x) * len(y)) - 1  # >0: x tends to be larger


def hier_boot_median(df_in, value_col, n_boot=5000, unit='neuron', sub='dataset_name', rng=None):
    """Hierarchical bootstrap: resample neurons, then datasets within each neuron; statistic = median of the
    neuron-level medians"""
    rng = np.random.default_rng(rng)
    groups = {k: v[value_col].values for k, v in df_in.groupby(unit)}
    keys = list(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        ks = rng.choice(keys, size=len(keys), replace=True)
        out[b] = np.median([np.median(rng.choice(groups[k], size=len(groups[k]), replace=True)) for k in ks])
    return out


stats_rows = []
wide = cls.pivot(index='neuron', columns='role', values='green')
wide['group'] = wide.index.map(group_of_neuron)

# ---- Q1: gated neurons, off vs. on (paired across neuron classes)
g = wide[wide.group == 'Gated']
diff = g['off'] - g['on']
w = stats.wilcoxon(g['off'], g['on'])
stats_rows.append(dict(question='Q1', comparison='Gated: off vs. on (paired, class medians)', n=f"{len(g)} neurons",
                       median_a=g['off'].median(), median_b=g['on'].median(), median_diff=diff.median(),
                       test='Wilcoxon signed-rank', statistic=w.statistic, p=w.pvalue,
                       effect_size=rank_biserial_paired(diff), effect_size_name='matched-pairs rank-biserial'))
# Per-dataset paired differences, hierarchical bootstrap
d_pairs = df[df.group == 'Gated'].pivot_table(index=['neuron', 'dataset_name'], columns='role',
                                              values='green_median').reset_index()
d_pairs['diff'] = d_pairs['off'] - d_pairs['on']
boot = hier_boot_median(d_pairs, 'diff', rng=1)
stats_rows.append(dict(question='Q1', comparison='Gated: off - on (per dataset), hierarchical bootstrap',
                       n=f"{d_pairs.neuron.nunique()} neurons, {len(d_pairs)} neuron x dataset pairs",
                       median_diff=np.median(d_pairs.groupby('neuron')['diff'].median()),
                       ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                       p=2 * min((boot <= 0).mean(), (boot >= 0).mean()), test='hierarchical bootstrap (neurons, datasets)'))
# Per-neuron Wilcoxon across datasets (descriptive)
per_neuron_q1 = []
for n, d in d_pairs.groupby('neuron'):
    w_n = stats.wilcoxon(d['off'], d['on'])
    per_neuron_q1.append(dict(neuron=n, n_datasets=len(d), median_off=d['off'].median(), median_on=d['on'].median(),
                              median_diff=d['diff'].median(), frac_datasets_off_lower=(d['diff'] < 0).mean(),
                              p_wilcoxon=w_n.pvalue))
per_neuron_q1 = pd.DataFrame(per_neuron_q1)
from statsmodels.stats.multitest import multipletests
per_neuron_q1['p_fdr'] = multipletests(per_neuron_q1['p_wilcoxon'], method='fdr_bh')[1]
print(per_neuron_q1.round(3))

# Mixed model: green ~ role + neuron (fixed) with dataset random intercept
import statsmodels.formula.api as smf


def mixed_paired(d, formula):
    """Paired design (same neuron x dataset in both conditions): neuron as fixed effect (in formula), random intercepts
    for dataset and for each neuron x dataset pair (statsmodels variance components are nested within groups)"""
    return smf.mixedlm(formula, d, groups=d['dataset_name'], re_formula='1',
                       vc_formula={'pair': '0 + C(neuron)'}).fit()


def mixed_crossed(d, formula):
    """Between neuron groups: crossed random intercepts for dataset and neuron (single group + variance components)"""
    d = d.assign(_one=1)
    return smf.mixedlm(formula, d, groups=d['_one'],
                       vc_formula={'dataset': '0 + C(dataset_name)', 'neuron': '0 + C(neuron)'}).fit()

_mm = df[df.group == 'Gated'][['neuron', 'dataset_name', 'role', 'green_median']].copy()
mm = mixed_paired(_mm, 'green_median ~ C(role, Treatment("on")) + C(neuron)')
coef = 'C(role, Treatment("on"))[T.off]'
stats_rows.append(dict(question='Q1', comparison='Gated: off - on, mixed model (neuron fixed; dataset and neuron x dataset random)',
                       n=f"{_mm.neuron.nunique()} neurons, {_mm.dataset_name.nunique()} datasets, {len(_mm)} obs",
                       median_diff=mm.params[coef], ci_low=mm.conf_int().loc[coef, 0], ci_high=mm.conf_int().loc[coef, 1],
                       p=mm.pvalues[coef], test='linear mixed model (statsmodels MixedLM)'))

# ---- Q2: gated off-state vs. baselines at which always-oscillating neurons oscillate
gated_off = wide.loc[wide.group == 'Gated', 'off']
gated_on = wide.loc[wide.group == 'Gated', 'on']
alw = cls[cls.group == 'Always oscillating'].pivot(index='neuron', columns='state', values='green')
nonosc = cls[cls.group == 'Not oscillating'].pivot(index='neuron', columns='state', values='green')
comparisons_q2 = {
    'Gated off vs. always-osc. forward': (gated_off, alw['forward']),
    'Gated off vs. always-osc. backward': (gated_off, alw['backward']),
    'Gated off vs. always-osc. lower state': (gated_off, alw.min(axis=1)),
    'Gated on vs. always-osc. forward': (gated_on, alw['forward']),
    'Gated off vs. not-osc. forward': (gated_off, nonosc['forward']),
    'Gated off vs. not-osc. backward': (gated_off, nonosc['backward']),
}
for name, (a, b) in comparisons_q2.items():
    mw = stats.mannwhitneyu(a, b, alternative='two-sided')
    stats_rows.append(dict(question='Q2', comparison=name + ' (class medians)', n=f"{len(a)} vs {len(b)} neurons",
                           median_a=a.median(), median_b=b.median(), median_diff=a.median() - b.median(),
                           test='Mann-Whitney U', statistic=mw.statistic, p=mw.pvalue,
                           effect_size=rank_biserial_mwu(a, b), effect_size_name='rank-biserial (>0: first larger)'))
# Fraction of always-oscillating class baselines (either state) below the gated off-state median
alw_all = pd.concat([alw['forward'], alw['backward']])
frac_alw_below = (alw_all < gated_off.median()).mean()

# Q2 check: hierarchical bootstrap of the difference of group medians, and mixed model with dataset random effect
_a = df[(df.group == 'Gated') & (df.role == 'off')]
_b = df[(df.group == 'Always oscillating')]
rng = np.random.default_rng(2)
boot_a = hier_boot_median(_a, 'green_median', rng=rng)
boot_b_f = hier_boot_median(_b[_b.state == 'forward'], 'green_median', rng=rng)
boot_b_b = hier_boot_median(_b[_b.state == 'backward'], 'green_median', rng=rng)
for nm, bb in [('forward', boot_b_f), ('backward', boot_b_b)]:
    dd = boot_a - bb
    stats_rows.append(dict(question='Q2', comparison=f'Gated off - always-osc. {nm}, hierarchical bootstrap',
                           n=f"{_a.neuron.nunique()} vs {_b.neuron.nunique()} neurons",
                           median_diff=np.median(dd), ci_low=np.percentile(dd, 2.5), ci_high=np.percentile(dd, 97.5),
                           p=2 * min((dd <= 0).mean(), (dd >= 0).mean()), test='hierarchical bootstrap (neurons, datasets)'))
_mm2 = pd.concat([_a.assign(cmp='gated_off'), _b.assign(cmp='always_' + _b.state)])[
    ['neuron', 'dataset_name', 'cmp', 'green_median']]
mm2 = mixed_crossed(_mm2, 'green_median ~ C(cmp, Treatment("gated_off"))')
for lvl in ['always_forward', 'always_backward']:
    c = f'C(cmp, Treatment("gated_off"))[T.{lvl}]'
    stats_rows.append(dict(question='Q2', comparison=f'Always-osc. {lvl.split("_")[1]} - gated off, mixed model '
                                                     f'(crossed random intercepts: dataset, neuron)',
                           n=f"{len(_mm2)} obs", median_diff=mm2.params[c], ci_low=mm2.conf_int().loc[c, 0],
                           ci_high=mm2.conf_int().loc[c, 1], p=mm2.pvalues[c], test='linear mixed model'))

# ---- Q3: dynamic-range position
for grp in GROUP_ORDER:
    v = cls_pct.loc[cls_pct.group == grp, 'off_percentile_in_on']
    stats_rows.append(dict(question='Q3', comparison=f'{grp}: off-state median percentile within on-state range',
                           n=f"{len(v)} neurons", median_a=v.median(), ci_low=v.quantile(0.25), ci_high=v.quantile(0.75),
                           test='descriptive (median, IQR across neuron classes)'))
a = cls_pct.loc[cls_pct.group == 'Gated', 'off_percentile_in_on']
for grp in ['Always oscillating', 'Not oscillating']:
    b = cls_pct.loc[cls_pct.group == grp, 'off_percentile_in_on']
    mw = stats.mannwhitneyu(a, b, alternative='two-sided')
    stats_rows.append(dict(question='Q3', comparison=f'Percentile: gated vs. {grp.lower()}', n=f"{len(a)} vs {len(b)} neurons",
                           median_a=a.median(), median_b=b.median(), test='Mann-Whitney U', statistic=mw.statistic,
                           p=mw.pvalue, effect_size=rank_biserial_mwu(a, b), effect_size_name='rank-biserial'))

# ---- Amplitude: does the model-based amplitude actually drop in the off state?
amp = cls.pivot(index='neuron', columns='role', values='amplitude')
amp['group'] = amp.index.map(group_of_neuron)
for grp in GROUP_ORDER:
    s = amp[amp.group == grp]
    w = stats.wilcoxon(s['on'], s['off'])
    stats_rows.append(dict(question='Amplitude', comparison=f'{grp}: Bayesian amplitude on vs. off', n=f"{len(s)} neurons",
                           median_a=s['on'].median(), median_b=s['off'].median(), median_diff=(s['on'] - s['off']).median(),
                           test='Wilcoxon signed-rank', statistic=w.statistic, p=w.pvalue,
                           effect_size=rank_biserial_paired(s['on'] - s['off']), effect_size_name='matched-pairs rank-biserial'))
# Across all gated+always classes and states: is amplitude related to baseline?
_c = cls[cls.group != 'Not oscillating']
rho = stats.spearmanr(_c['green'], _c['amplitude'])
stats_rows.append(dict(question='Amplitude', comparison='Spearman(baseline, amplitude), gated+always, both states',
                       n=f"{len(_c)} neuron-states", statistic=rho.statistic, p=rho.pvalue, test='Spearman'))

# Gate values (the sigmoid has a fixed slope in z-scored PC units, so it does not fully close)
gate_role = cls.pivot(index='neuron', columns='role', values='gate')
gate_role['group'] = gate_role.index.map(group_of_neuron)
for grp in GROUP_ORDER:
    s_ = gate_role[gate_role.group == grp]
    stats_rows.append(dict(question='Gate', comparison=f'{grp}: mean gate s_j, on vs. off state', n=f"{len(s_)} neurons",
                           median_a=s_['on'].median(), median_b=s_['off'].median(),
                           median_diff=(s_['on'] - s_['off']).median(), test='descriptive (median across neurons)'))
_g = amp[amp.group == 'Gated']
stats_rows.append(dict(question='Amplitude', comparison='Gated: off/on amplitude ratio (median across neurons)',
                       n=f"{len(_g)} neurons", median_a=(_g['off'] / _g['on']).median(),
                       ci_low=(_g['off'] / _g['on']).min(), ci_high=(_g['off'] / _g['on']).max(),
                       test='descriptive (median, min-max)'))
df_stats = pd.DataFrame(stats_rows)
pd.set_option('display.width', 250)
print(df_stats.drop(columns=[c for c in ['effect_size_name'] if c in df_stats]).round(4).to_string())
print(f"Fraction of always-oscillating class baselines (both states) below the gated off-state median: {frac_alw_below:.2f}")

# %% [markdown]
# ## Boxplot statistics (panel b): neuron level and dataset level
#
# Boxes: gated neurons in their encoding / non-encoding state; ungated and non-encoding neurons in forward / backward.
# Comparisons: paired within each group (encoding vs. non-encoding; forward vs. backward), and the non-encoding state of
# gated neurons vs. each other box. BH-FDR within each level, as in the other paper figures.
# - Neuron level: value = median across datasets; Wilcoxon signed-rank (paired) / Mann-Whitney U.
# - Dataset level: value = per neuron x dataset median; linear mixed models
#   (paired: neuron fixed, random intercepts for dataset and neuron x dataset; between groups: crossed random
#   intercepts for dataset and neuron).

# %%
BOX_LABELS = ['Gated: encoding state', 'Gated: non-encoding state', 'Ungated: forward', 'Ungated: backward',
              'No encoding: forward', 'No encoding: backward']
BOX_GROUP = ['Gated', 'Gated', 'Always oscillating', 'Always oscillating', 'Not oscillating', 'Not oscillating']
LEGEND_NAME = {'Gated': 'Gated encoding', 'Always oscillating': 'Ungated encoding', 'Not oscillating': 'No encoding'}
BOX_COMPARISONS = [(0, 1), (2, 3), (4, 5), (1, 2), (1, 3), (1, 4), (1, 5)]


def _box_index(group, role, state):
    if group == 'Gated':
        return 0 if role == 'on' else 1
    return {'Always oscillating': 2, 'Not oscillating': 4}[group] + (0 if state == 'forward' else 1)


df['box'] = [_box_index(g, r, st) for g, r, st in zip(df['group'], df['role'], df['state'])]
cls['box'] = [_box_index(g, r, st) for g, r, st in zip(cls['group'], cls['role'], cls['state'])]
from statsmodels.stats.multitest import multipletests


def box_statistics(value_col_cls='green', value_col_ds='green_median'):
    rows = []
    for i, j in BOX_COMPARISONS:
        paired = BOX_GROUP[i] == BOX_GROUP[j]
        a_c = cls[cls.box == i].set_index('neuron')[value_col_cls]
        b_c = cls[cls.box == j].set_index('neuron')[value_col_cls]
        if paired:
            b_c = b_c.reindex(a_c.index)
            t = stats.wilcoxon(a_c, b_c)
            test_c, eff = 'Wilcoxon signed-rank', rank_biserial_paired(b_c - a_c)
        else:
            t = stats.mannwhitneyu(b_c, a_c, alternative='two-sided')
            test_c, eff = 'Mann-Whitney U', rank_biserial_mwu(b_c, a_c)
        # Dataset level: mixed model, coefficient = box j - box i
        d = df[df.box.isin([i, j])][['neuron', 'dataset_name', 'box', value_col_ds]].rename(columns={value_col_ds: 'v'})
        d['is_j'] = (d['box'] == j).astype(int)
        m = mixed_paired(d, 'v ~ is_j + C(neuron)') if paired else mixed_crossed(d, 'v ~ is_j')
        ci = m.conf_int().loc['is_j']
        rows.append(dict(box_a=BOX_LABELS[i], box_b=BOX_LABELS[j], paired=paired,
                         n_neurons=f"{len(a_c)}" if paired else f"{len(a_c)} vs {len(b_c)}",
                         median_a=a_c.median(), median_b=b_c.median(), median_diff_b_minus_a=(b_c - a_c).median() if paired else b_c.median() - a_c.median(),
                         neuron_test=test_c, neuron_stat=t.statistic, neuron_p=t.pvalue, neuron_effect_rank_biserial_b_vs_a=eff,
                         n_obs_dataset_level=len(d), n_datasets=d['dataset_name'].nunique(),
                         mixed_coef_b_minus_a=m.params['is_j'], mixed_ci_low=ci[0], mixed_ci_high=ci[1], mixed_p=m.pvalues['is_j']))
    out = pd.DataFrame(rows)
    out['neuron_p_fdr'] = multipletests(out['neuron_p'], method='fdr_bh')[1]
    out['mixed_p_fdr'] = multipletests(out['mixed_p'], method='fdr_bh')[1]
    return out


box_stats = box_statistics('green', 'green_median')
box_stats_sum = box_statistics('green_sum', 'green_sum_median')  # for comparison: summed (integrated) brightness
pd.set_option('display.width', 250)
cols_show = ['box_a', 'box_b', 'n_neurons', 'median_a', 'median_b', 'neuron_p', 'neuron_p_fdr',
             'neuron_effect_rank_biserial_b_vs_a', 'mixed_coef_b_minus_a', 'mixed_p', 'mixed_p_fdr']
print("Mean brightness per voxel:")
print(box_stats[cols_show].round(4).to_string())
print("Summed brightness (paper green trace), for comparison:")
print(box_stats_sum[cols_show].round(4).to_string())
_sv = cls.dropna(subset=['green', 'green_sum'])
print(f"Spearman(summed, mean-per-voxel) across neuron-states: {stats.spearmanr(_sv['green_sum'], _sv['green']).statistic:.2f}")

# %% [markdown]
# ## Save tables

# %%
per_neuron = cls.pivot(index='neuron', columns='state', values=['green', 'green_sum', 'amplitude', 'gate', 'n_datasets'])
per_neuron.columns = [f'{a}_{b}' for a, b in per_neuron.columns]
per_neuron['group'] = per_neuron.index.map(group_of_neuron)
per_neuron['on_state'] = per_neuron.index.map(on_off)
per_neuron['green_on'] = wide['on']
per_neuron['green_off'] = wide['off']
per_neuron['amplitude_on'] = amp['on']
per_neuron['amplitude_off'] = amp['off']
per_neuron['off_percentile_in_on'] = cls_pct.set_index('neuron')['off_percentile_in_on']
per_neuron['frac_datasets_off_within_on_5_95'] = cls_pct.set_index('neuron')['frac_datasets_off_within_on_5_95']
per_neuron['hyper_pca0'] = df_params['hyper_pca0']
per_neuron['fig5d_r'] = check['fig5d_r_gated']
per_neuron = per_neuron.sort_values(['group', 'neuron'])
per_neuron.round(4).to_csv(OUT_DIR / 'per_neuron_table.csv')
df.drop(columns=['green_values']).round(5).to_csv(OUT_DIR / 'per_neuron_dataset_state_table.csv', index=False)
df_pct.round(3).to_csv(OUT_DIR / 'dynamic_range_percentile_per_dataset.csv', index=False)
df_stats.to_csv(OUT_DIR / 'stats_summary.csv', index=False)
box_stats.to_csv(OUT_DIR / 'boxplot_stats.csv', index=False)
box_stats_sum.to_csv(OUT_DIR / 'boxplot_stats_summed_brightness.csv', index=False)
per_neuron_q1.round(4).to_csv(OUT_DIR / 'q1_per_neuron_wilcoxon.csv', index=False)
pd.Series(align_summary).to_csv(OUT_DIR / 'alignment_check.csv')
n_frames_summary = df.groupby(['group', 'state'])['n_frames'].describe()
n_frames_summary.to_csv(OUT_DIR / 'n_frames_summary.csv')
print(n_frames_summary)

# %% [markdown]
# ## Figure panels

# %%
def _save(fig, name, width_factor, height_factor):
    apply_figure_settings(fig, width_factor=width_factor, height_factor=height_factor)
    fig.write_image(OUT_DIR / f'{name}.png', scale=3)
    fig.write_image(OUT_DIR / f'{name}.svg')
    fig.write_html(OUT_DIR / f'{name}.html')


STATE_SYMBOL = {'forward': 'circle', 'backward': 'triangle-down'}
HIGHLIGHT = ['VB02', 'DB01', 'DD01']
# Panels a and b show the same class-level baselines, so they share the y-axis
Y_RANGE = [0, cls['green'].max() * 1.4]
Y_TITLE = 'Baseline (mean green<br>per voxel, a.u.)'


def _pval(comparison):
    return df_stats.loc[df_stats['comparison'] == comparison, 'p'].iloc[0]


def _add_bracket(fig, i0, i1, p, k, y0_domain, y_step=0.08):
    """Paper-style significance bracket (stars / ns, add_p_value_annotation) between box traces i0 and i1.
    Brackets are stacked (k) from y0_domain upwards, in axis-domain units, inside the plotting area."""
    add_p_value_annotation(fig, array_columns=[[i0, i1]], precalculated_p_values={None: p}, show_ns=True,
                           height_mode='all_same', annotation_y_shift=y0_domain - 1.01 + y_step * k)
    fig.layout.annotations[-1].update(yanchor='bottom', yshift=1)  # put the stars/ns text on top of the bracket line


# ---- a: baseline vs. Bayesian amplitude, per neuron class and state
fig = go.Figure()
for grp in GROUP_ORDER[::-1]:
    sub = cls[cls.group == grp]
    for n, d in sub.groupby('neuron'):
        d = d.set_index('state').loc[['forward', 'backward']]
        fig.add_trace(go.Scatter(x=d['amplitude'], y=d['green'], mode='lines', showlegend=False, hoverinfo='skip',
                                 line=dict(color=GROUP_COLORS[grp], width=1), opacity=0.6 if grp != 'Not oscillating' else 0.35))
    for st in ['forward', 'backward']:
        d = sub[sub.state == st]
        fig.add_trace(go.Scatter(x=d['amplitude'], y=d['green'], mode='markers', name=f'{grp}, {st}',
                                 text=d['neuron'], hovertemplate='%{text}<br>amp %{x:.3f}<br>green %{y:.0f}',
                                 marker=dict(color=GROUP_COLORS[grp], symbol=STATE_SYMBOL[st], size=8,
                                             line=dict(width=1, color='black'))))
for n in HIGHLIGHT:
    d = cls[(cls.neuron == n) & (cls.role == 'on')].iloc[0]
    fig.add_annotation(x=np.log10(d['amplitude']), y=d['green'], text=n, showarrow=False, yshift=12, font=dict(size=10))
fig.update_xaxes(title='Oscillation amplitude (Bayesian model)', type='log',
                 tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext=['0.01', '0.02', '0.05', '0.1', '0.2', '0.5'])
fig.update_yaxes(title=Y_TITLE, range=Y_RANGE, tickformat='~s')
fig.update_layout(legend=dict(yanchor='top', y=1.0, xanchor='left', x=1.02, title='', font=dict(size=10)))
_save(fig, 'a_baseline_vs_amplitude', 0.6, 0.3)

# %%
# ---- b: boxplots of the green baseline (mean brightness per voxel)
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
TICKS = ['Encoding', 'Non-encoding', 'Forward', 'Backward', 'Forward', 'Backward']
BOX_COLORS = [GROUP_COLORS[g] for g in BOX_GROUP]


def make_baseline_boxplot(level='neuron', show_pair_lines=True):
    """level='neuron': one point per neuron (median across datasets), Wilcoxon / Mann-Whitney;
    level='dataset': one point per neuron x dataset, linear mixed models"""
    if level == 'neuron':
        data, val, unit = cls, 'green', ['neuron']
        p_values = box_stats['neuron_p_fdr'].values
    else:
        data, val, unit = df, 'green_median', ['neuron', 'dataset_name']
        p_values = box_stats['mixed_p_fdr'].values
    fig = go.Figure()
    shown = set()
    for b in range(6):  # box traces first: add_p_value_annotation indexes them by trace number
        d = data[data.box == b]
        g = BOX_GROUP[b]
        # (y as lists: with plotly 6, fig.to_dict() base64-encodes numpy arrays, which add_p_value_annotation can't read)
        fig.add_trace(go.Box(x=[b] * len(d), y=d[val].tolist(), name=LEGEND_NAME[g], legendgroup=g,
                             showlegend=g not in shown, marker_color=BOX_COLORS[b], boxpoints='all',
                             jitter=0 if show_pair_lines else 0.5,
                             pointpos=0, marker=dict(size=4 if level == 'neuron' else 2.5,
                                                     opacity=1 if level == 'neuron' else 0.5),
                             line=dict(width=1.5), text=d['neuron'].tolist()))
        shown.add(g)
    if show_pair_lines:
        for b0 in [0, 2, 4]:  # one line trace per group, segments separated by None
            w = data[data.box.isin([b0, b0 + 1])].pivot_table(index=unit, columns='box', values=val).dropna()
            xs, ys = [], []
            for _, r in w.iterrows():
                xs += [b0, b0 + 1, None]
                ys += [r[b0], r[b0 + 1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', showlegend=False, hoverinfo='skip',
                                     line=dict(color='gray', width=0.7 if level == 'neuron' else 0.4),
                                     opacity=0.8 if level == 'neuron' else 0.25))
    # Brackets: paired comparisons on the first level, then the non-encoding state of gated neurons vs. the rest
    levels = {(0, 1): 0, (2, 3): 0, (4, 5): 0, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4}
    for (i, j), p in zip(BOX_COMPARISONS, p_values):
        add_p_value_annotation(fig, array_columns=[[i, j]], precalculated_p_values={None: p}, show_ns=True,
                               height_mode='all_same', annotation_y_shift=0.6 - 1.01 + 0.075 * levels[(i, j)])
        fig.layout.annotations[-1].update(yanchor='bottom', yshift=1, font=dict(size=11))
    ymax = data[val].max()
    fig.update_yaxes(title='Baseline (mean green<br>per voxel, a.u.)', range=[0, ymax / 0.56], tickformat='~s')
    fig.update_xaxes(title='', tickvals=list(range(6)), ticktext=TICKS, range=[-0.6, 5.6], tickangle=-40)
    fig.update_layout(legend=dict(yanchor='top', y=1.0, xanchor='left', x=1.01, title='', font=dict(size=10)))
    return fig


fig = make_baseline_boxplot('neuron')
_save(fig, 'b_baseline_boxplots', 0.6, 0.3)
fig = make_baseline_boxplot('dataset')
_save(fig, 'b_baseline_boxplots_datasets', 0.6, 0.3)
fig = make_baseline_boxplot('dataset', show_pair_lines=False)
_save(fig, 'b_baseline_boxplots_datasets_nolines', 0.6, 0.3)

# %%
# ---- c: dynamic-range percentile by group
fig = go.Figure()
for grp in GROUP_ORDER:
    d = cls_pct[cls_pct.group == grp]
    fig.add_trace(go.Box(x=[grp.replace(' ', '<br>', 1)] * len(d), y=d['off_percentile_in_on'].tolist(), marker_color=GROUP_COLORS[grp],
                         boxpoints='all', jitter=0.4, pointpos=0, marker=dict(size=5), text=d['neuron'],
                         showlegend=False, line=dict(width=1.5)))
for n in HIGHLIGHT:
    d = cls_pct[cls_pct.neuron == n].iloc[0]
    fig.add_annotation(x=d['group'].replace(' ', '<br>', 1), y=d['off_percentile_in_on'], text=n, showarrow=True,
                       arrowhead=0, ax=40, ay=0, font=dict(size=10))
fig.add_hrect(y0=20, y1=50, line_width=0, fillcolor='gray', opacity=0.12)
_add_bracket(fig, 0, 1, _pval('Percentile: gated vs. always oscillating'), 0, 0.77, y_step=0.09)
_add_bracket(fig, 0, 2, _pval('Percentile: gated vs. not oscillating'), 1, 0.77, y_step=0.09)
fig.update_yaxes(title='Off-state baseline:<br>percentile of on-state values', range=[0, 120], tickvals=[0, 20, 40, 60, 80, 100])
fig.update_xaxes(title='')
_save(fig, 'c_dynamic_range_percentile', 0.4, 0.3)
print("Done")

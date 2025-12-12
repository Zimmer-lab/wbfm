import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import ttest_ind, norm

# ======================================
# Configuration parameters (edit as needed)
# ======================================
F_LOW = 0.007    # Hz, lower bound of band [^1]
F_HIGH = 0.033   # Hz, upper bound of band [^1]
THRESHOLD_FRACTION = 0.2  # per-neuron band fraction threshold (Figure S2B metric) [^2]
QUIET_THRESHOLD = 20.0    # % of neurons above THRESHOLD_FRACTION to classify "quiet" [^1]
ALPHA_BH = 0.05           # Benjamini–Hochberg FDR level [^2]

# ==================================
# Helper functions: spectral metrics
# ==================================

def compute_delta_ff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ΔF/F per neuron (column): ΔF/F = (F - mean(F)) / mean(F) [^2].
    If df already contains ΔF/F, you can skip this step.
    """
    mean_vals = df.mean(axis=0)
    return (df - mean_vals) / mean_vals

def band_power_fraction_per_neuron(x: np.ndarray, d: float,
                                   f_low: float = F_LOW, f_high: float = F_HIGH) -> float:
    """
    Fraction of power in [f_low, f_high] Hz for a single neuron trace x using normalized PSD [^2].
    """
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2
    total_power = psd.sum()
    if total_power == 0:
        return 0.0
    psd /= total_power

    freqs = np.fft.rfftfreq(len(x), d=d)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    return float(psd[band_mask].sum())

def recording_metrics(df: pd.DataFrame, d: float,
                      f_low: float = F_LOW, f_high: float = F_HIGH,
                      threshold: float = THRESHOLD_FRACTION,
                      apply_delta_ff: bool = False) -> Tuple[float, float, np.ndarray]:
    """
    For one recording (df: T x N_neurons), compute:
      - per-neuron fraction of power in [f_low, f_high],
      - average fraction across neurons (for sorting/summary),
      - percentage of neurons with fraction > threshold (Figure S2B metric) [^2].
    Optionally apply ΔF/F per neuron [^2].
    """
    df_proc = compute_delta_ff(df) if apply_delta_ff else df

    fractions = []
    for col in df_proc.columns:
        x = df_proc[col].values
        frac = band_power_fraction_per_neuron(x, d, f_low, f_high)
        fractions.append(frac)

    fractions = np.array(fractions, dtype=float)
    avg_fraction = float(fractions.mean()) if len(fractions) > 0 else np.nan
    pct_neurons_above = 100.0 * (np.sum(fractions > threshold) / len(fractions)) if len(fractions) > 0 else np.nan
    return avg_fraction, pct_neurons_above, fractions

def spectral_edge_50(df: pd.DataFrame, d: float, apply_delta_ff: bool = False
                    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    50% spectral edge using CDF of averaged power spectra across neurons [^1]:
      - squared FFT per neuron,
      - average across neurons,
      - normalize,
      - CDF over frequency bins,
      - frequency at which CDF crosses 0.5.
    Returns: f50, freqs, psd_avg_norm, cdf
    """
    df_proc = compute_delta_ff(df) if apply_delta_ff else df

    psd_list = []
    n_time = df_proc.shape[0]
    for col in df_proc.columns:
        x = df_proc[col].values
        X = np.fft.rfft(x)
        psd_list.append(np.abs(X) ** 2)

    if len(psd_list) == 0:
        return np.nan, None, None, None

    psd_avg = np.mean(np.vstack(psd_list), axis=0)
    total_power = psd_avg.sum()
    if total_power == 0:
        return np.nan, None, None, None

    psd_avg_norm = psd_avg / total_power
    freqs = np.fft.rfftfreq(n_time, d=d)
    cdf = np.cumsum(psd_avg_norm)
    idx = int(np.searchsorted(cdf, 0.5))
    f50 = float(freqs[idx]) if idx < len(freqs) else float(freqs[-1])
    return f50, freqs, psd_avg_norm, cdf

# ============================================
# Statistical tests: Welch t-test and BH (FDR)
# ============================================

def welch_one_sided_ttest(groupA: np.ndarray, groupB: np.ndarray, alternative: str = 'greater'):
    """
    One-sided Welch’s t-test (equal_var=False) using SciPy [^2].
    alternative='greater' tests H1: mean(A) > mean(B).
    """
    return ttest_ind(groupA, groupB, equal_var=False, alternative=alternative)

def benjamini_hochberg(pvals: List[float], alpha: float = ALPHA_BH):
    """
    Benjamini–Hochberg step-up FDR correction on p-values [^2].
    Returns adjusted p-values and a boolean array indicating which tests pass at FDR alpha.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.arange(1, m + 1)
    adj = np.empty(m, dtype=float)
    adj[order] = np.minimum.accumulate((pvals[order][::-1] * m / ranked[::-1]))[::-1]
    crit = (ranked / m) * alpha
    passed = pvals[order] <= crit
    passed_ordered = np.zeros(m, dtype=bool)
    passed_ordered[order] = passed
    return adj, passed_ordered

# ==========================================
# Two-proportion z-test for quiet recordings
# ==========================================

def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test (pooled proportion) with SciPy’s normal CDF for two-sided p-value [^2].
    Returns (z-statistic, two-sided p-value). If denom=0 (identical proportions of 0 or 1), returns (nan, nan).
    """
    p_pool = (x1 + x2) / (n1 + n2)
    denom = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return np.nan, np.nan
    p1 = x1 / n1
    p2 = x2 / n2
    z = (p1 - p2) / denom
    pval_two_sided = 2 * norm.sf(np.abs(z))
    return float(z), float(pval_two_sided)

# ==========================
# Aggregation and metrics
# ==========================

def compute_all_metrics(
    all_dfs: Dict[Tuple[str, str], List[pd.DataFrame]],
    sampling_intervals_all: Dict[Tuple[str, str], List[float]],
    apply_delta_ff: bool = False
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Compute per-recording metrics for each condition:
      - avg_fraction (mean neuron band fraction),
      - pct_neurons_above (Figure S2B metric),
      - fractions per neuron,
      - f50 spectral edge and PSD/CDF (Figure S2A).
    Returns dict keyed by (condition_name, laser_wavelength) -> list of recording metrics dicts.
    """
    results = {}
    for key, df_list in all_dfs.items():
        d_list = sampling_intervals_all[key]
        assert len(df_list) == len(d_list), f"Sampling intervals length mismatch for key {key}"
        metrics_list = []
        for df, d in zip(df_list, d_list):
            avg_fraction, pct_neurons_above, fractions = recording_metrics(
                df, d, F_LOW, F_HIGH, THRESHOLD_FRACTION, apply_delta_ff=apply_delta_ff
            )
            f50, freqs, psd_avg_norm, cdf = spectral_edge_50(df, d, apply_delta_ff=apply_delta_ff)
            metrics_list.append({
                'avg_fraction': avg_fraction,            # used to sort recordings (Figure S1-like) [^3]
                'pct_neurons_above': pct_neurons_above,  # Figure S2B metric [^2]
                'fractions': fractions,
                'f50': f50,                              # 50% spectral edge (Figure S2A) [^1]
                'freqs': freqs,
                'psd_avg_norm': psd_avg_norm,            # normalized averaged PSD (Figure S2A top) [^1]
                'cdf': cdf                                # CDF (Figure S2A bottom) [^1]
            })
        results[key] = metrics_list
    return results

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Helper for consistent wavelength colors across figures
def make_wavelength_color_map(keys_or_results):
    """
    Build a consistent color map for wavelengths seen in either a dict keyed by (condition, wavelength)
    or a results dict of the same structure.
    """
    if isinstance(keys_or_results, dict):
        wavelengths = sorted({k[1] for k in keys_or_results.keys()})
    else:
        wavelengths = sorted(keys_or_results)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    return {w: palette[i % len(palette)] for i, w in enumerate(wavelengths)}


def plot_s2a_plotly_simple(results):
    """
    Simplified Figure S2A:
      - Top: normalized PSD averaged across recordings per condition
      - Bottom: corresponding CDF averaged across recordings per condition
      - Color by wavelength; label traces with "condition | wavelength"
      - Vertical dotted lines at F_LOW and F_HIGH; zoom x-axis to [0, 2*F_HIGH]

    Input
    -----
    results: dict[(condition, wavelength)] -> list of per-recording dicts with keys:
        'freqs', 'psd_avg_norm', 'cdf'
      (as returned by compute_all_metrics)

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    color_map = make_wavelength_color_map(results)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Normalized PSD", "CDF"),
                        vertical_spacing=0.1)

    # Build averaged curves per condition (within identical frequency-grid groups)
    for (cond, wavelength), recs in results.items():
        # Group recordings by frequency-grid length
        by_len = {}
        for r in recs:
            freqs, psd, cdf = r['freqs'], r['psd_avg_norm'], r['cdf']
            if freqs is None or psd is None or cdf is None:
                continue
            L = len(freqs)
            by_len.setdefault(L, {'freqs': [], 'psd': [], 'cdf': []})
            by_len[L]['freqs'].append(freqs)
            by_len[L]['psd'].append(psd)
            by_len[L]['cdf'].append(cdf)

        # Plot averaged traces (one per frequency-grid group)
        for L, grp in by_len.items():
            freqs_ref = grp['freqs'][0]
            psd_mean = np.mean(np.vstack(grp['psd']), axis=0)
            cdf_mean = np.mean(np.vstack(grp['cdf']), axis=0)
            label = f"{cond} | {wavelength}"
            color = color_map[wavelength]

            fig.add_trace(
                go.Scatter(x=freqs_ref, y=psd_mean, mode='lines',
                           name=label, legendgroup=wavelength, line=dict(color=color)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=freqs_ref, y=cdf_mean, mode='lines',
                           name=label, legendgroup=wavelength, showlegend=False,
                           line=dict(color=color)),
                row=2, col=1
            )

    # Vertical dotted lines at band edges [^1]
    for row in [1, 2]:
        fig.add_vline(x=F_LOW, line=dict(color='black', dash='dot', width=1), row=row, col=1)
        fig.add_vline(x=F_HIGH, line=dict(color='black', dash='dot', width=1), row=row, col=1)

    # Axes and layout
    fig.update_yaxes(title_text="Normalized PSD", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 2 * F_HIGH], row=2, col=1)
    fig.update_layout(title="Figure S2A: Averaged normalized PSDs and CDFs per condition [^2]",
                      height=600)
    return fig


def plot_s2b_plotly_simple(results):
    """
    Simplified Figure S2B:
      - Boxplots of % of neurons with band fraction in [F_LOW, F_HIGH] > THRESHOLD_FRACTION per recording
      - x-axis: condition, color: wavelength
      - Matches the metric definition and usage in the supplemental methods and S2B panel [^2]

    Input
    -----
    results: dict[(condition, wavelength)] -> list of per-recording dicts with keys:
        'pct_neurons_above'

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    color_map = make_wavelength_color_map(results)
    # Build data per wavelength
    waves = sorted(color_map.keys())
    data = {w: {'x': [], 'y': []} for w in waves}
    for (cond, wavelength), recs in results.items():
        ys = [r['pct_neurons_above'] for r in recs if not np.isnan(r['pct_neurons_above'])]
        xs = [cond] * len(ys)
        data[wavelength]['x'].extend(xs)
        data[wavelength]['y'].extend(ys)

    fig = go.Figure()
    for w in waves:
        fig.add_trace(go.Box(
            x=data[w]['x'],
            y=data[w]['y'],
            name=str(w),
            marker_color=color_map[w]
        ))
    fig.update_layout(
        title=f"Figure S2B: % neurons with fraction in [{F_LOW}, {F_HIGH}] Hz > {THRESHOLD_FRACTION} [^2]",
        xaxis_title="Condition",
        yaxis_title="% of neurons > threshold",
        boxmode='group',
        height=500
    )
    return fig


def plot_s2c_plotly_simple(quiet_counts, totals):
    """
    Simplified Figure S2C:
      - Bar plot of proportion of quiet recordings per condition
      - x-axis: condition, color: wavelength (derived from condition labels of the form 'Condition | Wavelength')

    Inputs
    ------
    quiet_counts: dict["Condition | Wavelength"] -> int
    totals: dict["Condition | Wavelength"] -> int

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    # Derive wavelength mapping from labels "Condition | Wavelength"
    labels = list(quiet_counts.keys())
    parsed = [(lab.split('|')[0].strip(), lab.split('|')[1].strip()) for lab in labels]
    wavelengths = sorted({w for (_, w) in parsed})
    color_map = make_wavelength_color_map({('cond', w): None for w in wavelengths})

    # Collect bars per wavelength
    waves = wavelengths
    data = {w: {'x': [], 'y': []} for w in waves}
    for lab in labels:
        cond = lab.split('|')[0].strip()
        w = lab.split('|')[1].strip()
        prop = (quiet_counts[lab] / totals[lab]) if totals[lab] > 0 else np.nan
        data[w]['x'].append(cond)
        data[w]['y'].append(prop)

    fig = go.Figure()
    for w in waves:
        fig.add_trace(go.Bar(
            x=data[w]['x'],
            y=data[w]['y'],
            name=str(w),
            marker_color=color_map[w]
        ))
    fig.update_layout(
        title="Figure S2C: Proportion of quiet recordings per condition [^2]",
        xaxis_title="Condition",
        yaxis_title="Proportion of quiet recordings",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )
    return fig

# ==========================
# Plotly: Figure S2A-like
# ==========================

def plot_s2a_plotly(results: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> go.Figure:
    """
    Plot Figure S2A-like panels: normalized PSD (top) and CDF (bottom) per condition [^1].
    For each condition, averages curves across recordings that share identical frequency grids.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Normalized PSD", "CDF"),
                        vertical_spacing=0.1)

    for (cond_name, wavelength), recs in results.items():
        # Collect curves by frequency-grid length
        groups_by_len = {}
        for r in recs:
            freqs, psd, cdf = r['freqs'], r['psd_avg_norm'], r['cdf']
            if freqs is None or psd is None or cdf is None:
                continue
            L = len(freqs)
            groups_by_len.setdefault(L, {'freqs': [], 'psd': [], 'cdf': []})
            groups_by_len[L]['freqs'].append(freqs)
            groups_by_len[L]['psd'].append(psd)
            groups_by_len[L]['cdf'].append(cdf)

        # Plot per-frequency-grid average
        for L, grp in groups_by_len.items():
            freqs_ref = grp['freqs'][0]
            psd_mean = np.mean(np.vstack(grp['psd']), axis=0)
            cdf_mean = np.mean(np.vstack(grp['cdf']), axis=0)
            label = f"{cond_name} | {wavelength}"

            fig.add_trace(
                go.Scatter(x=freqs_ref, y=psd_mean, mode='lines', name=label, legendgroup=label),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=freqs_ref, y=cdf_mean, mode='lines', name=label, legendgroup=label, showlegend=False),
                row=2, col=1
            )

    fig.update_yaxes(title_text="Normalized PSD", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_layout(title="Figure S2A-like: Averaged normalized PSDs and CDFs per condition [^1]",
                      height=600)
    return fig

# ==========================
# Plotly: Figure S2B-like
# ==========================

def prepare_s2b_data(results: Dict[Tuple[str, str], List[Dict[str, Any]]]):
    """
    Prepare data for S2B: percentage of neurons with band fraction > 0.2 per recording [^2].
    Returns lists per condition: cond_labels, data_values, means, stds.
    """
    cond_labels = []
    data_values = []
    means = []
    stds = []
    for (cond_name, wavelength), recs in results.items():
        vals = [r['pct_neurons_above'] for r in recs if not np.isnan(r['pct_neurons_above'])]
        cond_labels.append(f"{cond_name} | {wavelength}")
        data_values.append(vals)
        means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else np.nan)
    return cond_labels, data_values, means, stds

def plot_s2b_plotly(cond_labels: List[str], data_values: List[List[float]],
                    means: List[float], stds: List[float]) -> go.Figure:
    """
    Plot Figure S2B-like with Plotly:
      - Boxplots (Q1–Q3),
      - Overlay mean with error bars showing standard deviation (to reflect whiskers as SD per description) [^1].
    """
    fig = go.Figure()

    for i, (label, vals) in enumerate(zip(cond_labels, data_values)):
        fig.add_trace(go.Box(
            y=vals,
            name=label,
            boxmean=False,  # we'll overlay mean explicitly
            showlegend=False
        ))

    # Overlay means with SD error bars
    fig.add_trace(go.Scatter(
        x=cond_labels,
        y=means,
        mode='markers',
        name='Mean ± SD',
        error_y=dict(type='data', array=stds, visible=True),
        marker=dict(color='gray', size=8)
    ))

    fig.update_layout(
        title=f"Figure S2B-like: % neurons with fraction in [{F_LOW}, {F_HIGH}] Hz > {THRESHOLD_FRACTION} [^2]",
        yaxis_title="% of neurons > threshold",
        height=500
    )
    return fig

def compute_s2b_stats_plotly(cond_labels: List[str], data_values: List[List[float]],
                             alpha: float = ALPHA_BH, alternative: str = 'greater'):
    """
    Compute one-sided Welch’s t-tests between adjacent condition pairs and BH correction [^2].
    Returns dict with stats and a list of annotation strings.
    """
    comparisons = []
    for i in range(len(cond_labels) - 1):
        comparisons.append((cond_labels[i], cond_labels[i+1]))

    pvals, stats, labels = [], [], []
    for condA, condB in comparisons:
        idxA = cond_labels.index(condA)
        idxB = cond_labels.index(condB)
        groupA = np.array(data_values[idxA], dtype=float)
        groupB = np.array(data_values[idxB], dtype=float)
        res = ttest_ind(groupA, groupB, equal_var=False, alternative=alternative)  # [^2]
        pvals.append(res.pvalue)
        stats.append(res.statistic)
        labels.append((condA, condB))

    adj_pvals, passed = benjamini_hochberg(pvals, alpha=alpha)  # [^2]

    out = {}
    annotations = []
    for i, lab in enumerate(labels):
        out[lab] = {
            't_stat': stats[i],
            'pval_raw': pvals[i],
            'pval_adj': adj_pvals[i],
            'passed_fdr': bool(passed[i])
        }
        annotations.append(f"{lab[0]} > {lab[1]} | BH p={adj_pvals[i]:.3g}")
    return out, annotations

# ==========================
# Plotly: Figure S2C-like
# ==========================

def classify_quiet(results: Dict[Tuple[str, str], List[Dict[str, Any]]],
                   quiet_threshold: float = QUIET_THRESHOLD):
    """
    Classify recordings as 'quiet' based on % of neurons above threshold (S2B metric) [^1].
    Returns quiet_counts and totals per condition label.
    """
    quiet_counts = {}
    totals = {}
    for (cond_name, wavelength), recs in results.items():
        label = f"{cond_name} | {wavelength}"
        vals = [r['pct_neurons_above'] for r in recs if not np.isnan(r['pct_neurons_above'])]
        quiet_counts[label] = int(np.sum(np.array(vals) < quiet_threshold))
        totals[label] = len(vals)
    return quiet_counts, totals

def plot_s2c_plotly(quiet_counts: Dict[str, int], totals: Dict[str, int]) -> go.Figure:
    """
    Plot Figure S2C-like: bar plot of proportion of quiet recordings per condition [^1].
    """
    conds = list(quiet_counts.keys())
    props = [quiet_counts[c] / totals[c] if totals[c] > 0 else np.nan for c in conds]

    fig = go.Figure(go.Bar(x=conds, y=props))
    fig.update_layout(
        title="Figure S2C-like: Proportion of quiet recordings per condition [^1]",
        yaxis_title="Proportion of quiet recordings",
        height=400
    )
    return fig, conds, props

def compute_s2c_stats(quiet_counts: Dict[str, int], totals: Dict[str, int]):
    """
    Two-proportion z-tests for adjacent condition pairs (quiet proportions) [^2].
    Returns dict mapping (condA, condB) to z-stat and two-sided p-value.
    """
    conds = list(quiet_counts.keys())
    comparisons = []
    for i in range(len(conds) - 1):
        comparisons.append((conds[i], conds[i+1]))

    out = {}
    for condA, condB in comparisons:
        x1, n1 = quiet_counts[condA], totals[condA]
        x2, n2 = quiet_counts[condB], totals[condB]
        z, pval = two_proportion_ztest(x1, n1, x2, n2)  # may return (nan, nan) for degenerate cases [^2]
        out[(condA, condB)] = {'z_stat': z, 'pval_two_sided': pval}
    return out


import numpy as np
import plotly.graph_objects as go

def compute_fig1_metrics(
    all_dfs,
    sampling_intervals_all,
    apply_delta_ff=False,
    comparisons=None,           # list of pairs: [ ((condA, λA), (condB, λB)), ... ]
    alternative='greater',      # H1: mean(condA) > mean(condB) [^2]
    alpha=ALPHA_BH              # BH FDR level [^2]
):
    """
    Compute Figure 1E and 1F metrics per recording and aggregate by condition.

    Metrics per condition:
      - Fig 1E: Average fraction of power in [F_LOW, F_HIGH] Hz across neurons for each recording [^1].
      - Fig 1F: Frequency below which 50% of the averaged spectral power resides for each recording [^1].

    Also:
      - Identifies the median recording (by Fig 1E metric) per condition (to mark with a black square).
      - Optionally performs one-sided Welch’s t-tests with Benjamini–Hochberg correction across specified condition pairs [^2].

    Parameters
    ----------
    all_dfs : dict[(condition_name, laser_wavelength) -> list[pd.DataFrame]]
        Each DataFrame is T x N_neurons for a single recording.
    sampling_intervals_all : dict[(condition_name, laser_wavelength) -> list[float]]
        Sampling intervals (seconds) aligned to each recording [^1].
    apply_delta_ff : bool
        If True, apply ΔF/F per neuron before spectral analysis [^2].
    comparisons : list[tuple[(cond_name_A, λ_A), (cond_name_B, λ_B)]] or None
        Optional condition-pair comparisons for stats. H1: mean(A) > mean(B) [^2].
    alternative : {'greater', 'less', 'two-sided'}
        Alternative for Welch’s t-test (default 'greater') [^2].
    alpha : float
        FDR level for Benjamini–Hochberg correction (default ALPHA_BH) [^2].

    Returns
    -------
    out : dict
      - 'per_condition': dict[(cond_name, λ)] -> {
            'avg_fractions': list[float],   # per recording (Fig 1E) [^1]
            'f50': list[float],             # per recording (Fig 1F) [^1]
            'median_idx': int or None       # index of median recording by avg_fraction
        }
      - 'stats': dict (only if comparisons is not None) with entries:
            ((condA, λA), (condB, λB)) -> {
                't_stat': float,
                'pval_raw': float,
                'pval_adj': float,
                'passed_fdr': bool,
                'percent_change_mean': float  # 100 * (meanA - meanB)/meanB
            }
    """
    per_condition = {}

    # Compute per-recording metrics for each condition
    for key, df_list in all_dfs.items():
        d_list = sampling_intervals_all[key]
        assert len(df_list) == len(d_list), f"Sampling intervals length mismatch for key {key}"

        avg_fracs = []
        f50_list = []
        for df, d in zip(df_list, d_list):
            # Figure 1E metric: average across neurons of per-neuron band fraction [^1]
            avg_fraction, _, _ = recording_metrics(
                df, d, F_LOW, F_HIGH, THRESHOLD_FRACTION, apply_delta_ff=apply_delta_ff
            )
            # Figure 1F metric: 50% spectral edge from averaged spectrum CDF [^1]
            f50, _, _, _ = spectral_edge_50(df, d, apply_delta_ff=apply_delta_ff)
            avg_fracs.append(avg_fraction)
            f50_list.append(f50)

        # Median recording index by average fraction (for black square) [^1]
        valid_vals = [(i, v) for i, v in enumerate(avg_fracs) if not np.isnan(v)]
        if len(valid_vals) > 0:
            sorted_idx = sorted(valid_vals, key=lambda iv: iv[1])
            median_pos = len(sorted_idx) // 2
            median_idx = sorted_idx[median_pos][0]
        else:
            median_idx = None

        per_condition[key] = {
            'avg_fractions': avg_fracs,
            'f50': f50_list,
            'median_idx': median_idx
        }

    # Optional stats across specified comparisons (one-sided Welch + BH) [^2]
    stats_out = {}
    if comparisons is not None and len(comparisons) > 0:
        pvals = []
        tstats = []
        pct_changes = []
        pairs = []
        for (condA, condB) in comparisons:
            arrA = np.array(per_condition[condA]['f50'], dtype=float)
            arrB = np.array(per_condition[condB]['f50'], dtype=float)
            res = welch_one_sided_ttest(arrA, arrB, alternative=alternative)  # SciPy Welch [^2]
            pvals.append(res.pvalue)
            tstats.append(res.statistic)
            pairs.append((condA, condB))
            meanA = np.nanmean(arrA)
            meanB = np.nanmean(arrB)
            pct_change = 100.0 * (meanA - meanB) / meanB if np.isfinite(meanA) and np.isfinite(meanB) and meanB != 0 else np.nan
            pct_changes.append(pct_change)

        adj_pvals, passed = benjamini_hochberg(pvals, alpha=alpha)  # BH correction [^2]

        for i, pair in enumerate(pairs):
            stats_out[pair] = {
                't_stat': tstats[i],
                'pval_raw': pvals[i],
                'pval_adj': adj_pvals[i],
                'passed_fdr': bool(passed[i]),
                'percent_change_mean': pct_changes[i]
            }

    return {'per_condition': per_condition, 'stats': stats_out if len(stats_out) > 0 else None}


def plot_fig1E_F_plotly_simple(
    per_condition
):
    """
    Simplified Plotly boxplots for Figure 1E and 1F:
      - x-axis: condition
      - color: wavelength
      - regular boxplots (no median markers or mean/SD overlays)

    Inputs
    ------
    per_condition : dict[(condition, wavelength)] -> {
        'avg_fractions': list[float],  # Figure 1E metric [^1]
        'f50': list[float]             # Figure 1F metric [^1]
    }

    Returns
    -------
    figures : dict
      - 'Fig1E': plotly.graph_objects.Figure
      - 'Fig1F': plotly.graph_objects.Figure
    """
    # Collect all wavelengths to assign consistent colors
    wavelengths = sorted({w for (_, w) in per_condition.keys()})
    # Simple color palette mapping for wavelengths
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wavelengths)}

    # Prepare traces per wavelength for Fig 1E
    fig1e = go.Figure()
    # Build data per wavelength: x=condition, y=avg_fraction
    data_1e = {w: {'x': [], 'y': []} for w in wavelengths}
    for (cond, w), vals in per_condition.items():
        ys = [v for v in vals['avg_fractions'] if not np.isnan(v)]
        xs = [cond] * len(ys)
        data_1e[w]['x'].extend(xs)
        data_1e[w]['y'].extend(ys)

    for w in wavelengths:
        fig1e.add_trace(go.Box(
            x=data_1e[w]['x'],
            y=data_1e[w]['y'],
            name=str(w),
            marker_color=color_map[w]
        ))
    fig1e.update_layout(
        title=f"Figure 1E: Average fraction of power in [{F_LOW}, {F_HIGH}] Hz per recording [^1]",
        xaxis_title="Condition",
        yaxis_title="Average fraction in band",
        boxmode='group',
        height=500
    )

    # Prepare traces per wavelength for Fig 1F
    fig1f = go.Figure()
    data_1f = {w: {'x': [], 'y': []} for w in wavelengths}
    for (cond, w), vals in per_condition.items():
        ys = [v for v in vals['f50'] if not np.isnan(v)]
        xs = [cond] * len(ys)
        data_1f[w]['x'].extend(xs)
        data_1f[w]['y'].extend(ys)

    for w in wavelengths:
        fig1f.add_trace(go.Box(
            x=data_1f[w]['x'],
            y=data_1f[w]['y'],
            name=str(w),
            marker_color=color_map[w]
        ))
    fig1f.update_layout(
        title="Figure 1F: Frequency below which 50% of average spectral power resides [^1]",
        xaxis_title="Condition",
        yaxis_title="50% spectral edge (Hz)",
        boxmode='group',
        height=500
    )

    return {'Fig1E': fig1e, 'Fig1F': fig1f}


# ==========================
# Driver function
# ==========================

def reproduce_figures_plotly(
    all_dfs: Dict[Tuple[str, str], List[pd.DataFrame]],
    sampling_intervals_all: Dict[Tuple[str, str], List[float]],
    apply_delta_ff: bool = False,
    quiet_threshold: float = QUIET_THRESHOLD
):
    """
    End-to-end reproduction:
      - Compute metrics,
      - Plot S2A (normalized PSD & CDF),
      - Plot S2B (percentage above threshold) + stats,
      - Plot S2C (quiet proportions) + stats.
    """
    # Compute metrics per recording
    results = compute_all_metrics(all_dfs, sampling_intervals_all, apply_delta_ff=apply_delta_ff)

    # Figure S2A-like
    fig_s2a = plot_s2a_plotly_simple(results)

    # Figure S2B-like
    # cond_labels, data_values, means, stds = prepare_s2b_data(results)
    fig_s2b = plot_s2b_plotly_simple(results)

    # S2B stats and annotations (adjacent pairs by default)
    # s2b_stats, annotations = compute_s2b_stats_plotly(cond_labels, data_values, alpha=ALPHA_BH, alternative='greater')

    # Add annotations to the S2B figure (top area)
    # if len(annotations) > 0:
    #     # Place annotations above the highest y value
    #     all_vals = np.concatenate([np.array(v) for v in data_values if len(v) > 0]) if len(data_values) > 0 else np.array([])
    #     ymax = float(np.max(all_vals)) if all_vals.size > 0 else 100.0
    #     for i, ann in enumerate(annotations):
    #         fig_s2b.add_annotation(x=cond_labels[min(i+1, len(cond_labels)-1)], y=ymax,
    #                                text=ann, showarrow=False, yshift=10)

    # Figure S2C-like
    quiet_counts, totals = classify_quiet(results, quiet_threshold=quiet_threshold)
    fig_s2c, conds_s2c, props_s2c = plot_s2c_plotly_simple(quiet_counts, totals)

    # S2C stats (adjacent pairs)
    s2c_stats = compute_s2c_stats(quiet_counts, totals)

    # Print statistics summaries
    s2b_stats = dict()
    print("S2B Welch’s one-sided t-tests with BH correction [^2]:")
    for k, v in s2b_stats.items():
        condA, condB = k
        print(f"{condA} > {condB}: t={v['t_stat']:.3f}, p_raw={v['pval_raw']:.3g}, p_BH={v['pval_adj']:.3g}, pass_FDR={v['passed_fdr']}")

    print("\nS2C two-proportion z-tests [^2]:")
    for k, v in s2c_stats.items():
        condA, condB = k
        print(f"{condA} vs {condB}: z={v['z_stat']}, p_two_sided={v['pval_two_sided']}")

    return {
        'results': results,
        'figures': {'S2A': fig_s2a, 'S2B': fig_s2b, 'S2C': fig_s2c},
        'stats': {'S2B': s2b_stats, 'S2C': s2c_stats}
    }
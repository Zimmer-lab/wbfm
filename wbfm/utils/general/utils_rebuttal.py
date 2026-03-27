
# ======================================
# Configuration parameters (edit as needed)
# ======================================
"""
SUPPLEMENTAL FIGURES S2A, S2B, S2C: DUAL-MODE VISUALIZATION (March 2026 REFACTOR)
=================================================================================

ARCHITECTURE OVERVIEW:
This module generates publication-quality supplement figures with two visualization modes:

1. CLASSIC MODE (use_mean_and_shading=False):
   - S2A: Stacked subplots (original layout)
   - S2B: Grouped boxplots (percent neurons above threshold)
   - S2C: Bar plots (quiet proportions)

2. MEAN+SHADING MODE (use_mean_and_shading=True):
   - S2A: 2 rows × 4 cols grid (PSD/CDF rows, conditions 0-3 columns)
   - S2B: Line plots with mean ± shading per wavelength
   - S2C: Line plots with mean ± shading per wavelength

KEY DESIGN DECISIONS:
--------------------

DATA FORMAT (Critical for Understanding):
- Data preparation functions (prepare_s2a_psd_data_for_mean_and_shading, etc.)
  intentionally produce ONE ROW PER FREQUENCY PER RECORDING PER GROUP.
- NO pre-averaging in data prep. This separation of concerns prevents bugs.
- Aggregation happens downstream in plotly_plot_mean_and_shading:
  * Groups by (condition, wavelength)
  * Computes mean/std ACROSS RECORDINGS for each group
  
Example data structure after prepare_s2a_psd_data_for_mean_and_shading:
  frequency  psd         group
  0.008      0.5         0 | Green       <- freq bin from recording 1, cond 0, wavelength Green
  0.008      0.6         0 | Green       <- freq bin from recording 2, cond 0, wavelength Green
  0.008      0.3         1 | Green       <- freq bin from recording 1, cond 1, wavelength Green
  (continues for all freq bins, all recordings, all groups...)

S2A SPECIAL ARCHITECTURE (2×4 GRID):
When use_mean_and_shading=True:
  - Creates 2 rows (PSD, CDF) × 4 cols (Condition 0, 1, 2, 3)
  - CONDITION MUST be in [0,1,2,3] for grid layout to work
  - Each subplot shows all wavelengths for that condition (color-coded)
  - plotly_plot_mean_and_shading computes separate mean/std per wavelength within condition
  - Band-edge lines (F_LOW, F_HIGH) shown on all subplots
  - X-axis uniform across all subplots: [0, 2*F_HIGH]

COLOR MAPPING STRATEGY:
- make_wavelength_color_map(results) returns {wavelength -> color} palette
- Type consistency critical: use original wavelength objects from results dict
- Do NOT convert wavelengths to strings before building color map
- Build group_cmap directly from results.items() to maintain type alignment

SPECTRUM CONFIGURATION:
- Frequency band: [F_LOW=0.007, F_HIGH=0.033] Hz
- X-axis limit: [0, 2*F_HIGH] Hz
- Post-processing: percentile normalization per recording
- CDF computed via complementary empirical CDF (1 - standard CDF)

METRICS COMPUTED (compute_all_metrics output):
- PSD: Power Spectral Density (L/period, normalized per recording)
- CDF: Cumulative Distribution Function of activity levels
- S2B: Percent neurons with frequency band activity > THRESHOLD_FRACTION
- S2C: Percent recordings with >QUIET_THRESHOLD neurons above threshold

WORKFLOW:
1. reproduce_figures_plotly: Main entry point
2. compute_all_metrics: Extract PSD/CDF/thresholds per recording
3. Conditional logic based on use_mean_and_shading flag:
   IF False: Use original plot_s2a_plotly_simple, etc. (stacked layout)
   IF True: 
     a. prepare_s2a_psd_data_for_mean_and_shading -> DataFrame
     b. prepare_s2a_cdf_data_for_mean_and_shading -> DataFrame
     c. plot_s2a_mean_and_shading -> 2×4 grid figure
     (Similar pattern for S2B, S2C)

ERROR PREVENTION NOTES:
- Ensure results dict has condition keys in [0,1,2,3] (not strings like "0")
- Don't aggregate data before prepare_s2a_* (causes group mixing)
- Check wavelength type consistency in color map building
- Use direct results dict keys for cmap, not derived values
"""
from collections import defaultdict
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from tqdm.auto import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.external.utils_plotly import combine_plotly_figures, extract_shapes_as_figure, plotly_plot_mean_and_shading
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, options_for_ethogram
from wbfm.utils.general.utils_paper import apply_figure_settings, split_time_series_with_laser_switches
from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.visualization.plot_traces import make_summary_heatmap_and_subplots


F_LOW = 0.007    # Hz, lower bound of band [^1]
F_HIGH = 0.033   # Hz, upper bound of band [^1]
THRESHOLD_FRACTION = 0.2  # per-neuron band fraction threshold (Figure S2B metric) [^2]
QUIET_THRESHOLD = 20.0    # % of neurons above THRESHOLD_FRACTION to classify "quiet" [^1]
ALPHA_BH = 0.05           # Benjamini–Hochberg FDR level [^2]

# ==================================
# Helper functions: spectral metrics
# ==================================

def determine_common_fft_length(
    all_dfs: Dict[Tuple[str, str], List[pd.DataFrame]],
    strategy: str = 'max'
) -> int:
    """
    Determine a common FFT length to ensure all datasets use the same frequency grid.
    
    This function finds the optimal time-series length for zero-padding so that:
    - All recordings are padded to the same length before FFT
    - Frequency resolution is uniform across all conditions
    - Fine frequency details are preserved
    
    Parameters
    ----------
    all_dfs : dict[(condition, wavelength)] -> list[pd.DataFrame]
        All recordings for all conditions
    strategy : {'max', 'median'}, default='max'
        'max': Use longest recording length (preserves fine resolution)
        'median': Use median length (balances resolution with padding overhead)
    
    Returns
    -------
    target_fft_length : int
        The standardized FFT length to use for all recordings
    """
    all_lengths = []
    for key, df_list in all_dfs.items():
        for df in df_list:
            all_lengths.append(df.shape[0])
    
    if len(all_lengths) == 0:
        raise ValueError("No recordings found in all_dfs")
    
    if strategy == 'max':
        return int(np.max(all_lengths))
    elif strategy == 'median':
        return int(np.median(all_lengths))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def pad_signal_for_fft(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Zero-pad signal to target length (appends zeros at end).
    
    Parameters
    ----------
    x : np.ndarray
        Original signal
    target_length : int
        Desired length (must be >= len(x))
    
    Returns
    -------
    x_padded : np.ndarray
        Zero-padded signal of length target_length
    """
    if len(x) > target_length:
        raise ValueError(f"Signal length {len(x)} exceeds target {target_length}")
    if len(x) == target_length:
        return x.copy()
    return np.pad(x, (0, target_length - len(x)), mode='constant', constant_values=0)


def compute_delta_ff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ΔF/F per neuron (column): ΔF/F = (F - mean(F)) / mean(F) [^2].
    If df already contains ΔF/F, you can skip this step.
    """
    mean_vals = df.mean(axis=0)
    if (mean_vals == 0).any():
        raise ValueError("compute_delta_ff: one or more columns have zero mean, causing division by zero")
    return (df - mean_vals) / mean_vals

def band_power_fraction_per_neuron(x: np.ndarray, d: float,
                                   f_low: float = F_LOW, f_high: float = F_HIGH,
                                   target_fft_length: int = None) -> float:
    """
    Fraction of power in [f_low, f_high] Hz for a single neuron trace x using normalized PSD [^2].
    
    Parameters
    ----------
    x : np.ndarray
        Single-neuron time series
    d : float
        Sampling interval (seconds)
    f_low, f_high : float
        Frequency band bounds (Hz)
    target_fft_length : int, optional
        If provided, zero-pad signal to this length before FFT.
        Ensures all recordings use the same frequency grid.
        If None, uses signal length as-is.
    
    Returns
    -------
    float
        Normalized power fraction in band [f_low, f_high]
    """
    x_fft = pad_signal_for_fft(x, target_fft_length) if target_fft_length is not None else x
    X = np.fft.rfft(x_fft)
    psd = np.abs(X) ** 2
    total_power = psd.sum()
    if total_power == 0:
        return 0.0
    psd /= total_power

    freqs = np.fft.rfftfreq(len(x_fft), d=d)
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    return float(psd[band_mask].sum())

def recording_metrics(df: pd.DataFrame, d: float,
                      f_low: float = F_LOW, f_high: float = F_HIGH,
                      threshold: float = THRESHOLD_FRACTION,
                      apply_delta_ff: bool = False,
                      target_fft_length: int = None) -> Tuple[float, float, np.ndarray]:
    """
    For one recording (df: T x N_neurons), compute:
      - per-neuron fraction of power in [f_low, f_high],
      - average fraction across neurons (for sorting/summary),
      - percentage of neurons with fraction > threshold (Figure S2B metric) [^2].
    Optionally apply ΔF/F per neuron [^2].
    
    Parameters
    ----------
    df : pd.DataFrame
        Recording data (T x N_neurons)
    d : float
        Sampling interval (seconds)
    f_low, f_high : float
        Frequency band bounds
    threshold : float
        Threshold for counting neurons above band fraction
    apply_delta_ff : bool
        Apply ΔF/F normalization
    target_fft_length : int, optional
        FFT length for standardizing frequency grid across recordings
    
    Returns
    -------
    avg_fraction : float
        Average band fraction across neurons
    pct_neurons_above : float
        Percentage of neurons exceeding threshold
    fractions : np.ndarray
        Per-neuron band fractions
    """
    df_proc = compute_delta_ff(df) if apply_delta_ff else df

    fractions = []
    for col in df_proc.columns:
        x = df_proc[col].values
        frac = band_power_fraction_per_neuron(x, d, f_low, f_high, target_fft_length=target_fft_length)
        fractions.append(frac)

    fractions = np.array(fractions, dtype=float)
    avg_fraction = float(fractions.mean()) if len(fractions) > 0 else np.nan
    pct_neurons_above = 100.0 * (np.sum(fractions > threshold) / len(fractions)) if len(fractions) > 0 else np.nan
    return avg_fraction, pct_neurons_above, fractions

def spectral_edge_50(df: pd.DataFrame, d: float, apply_delta_ff: bool = False,
                    target_fft_length: int = None
                    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    50% spectral edge using CDF of averaged power spectra across neurons [^1]:
      - squared FFT per neuron,
      - average across neurons,
      - normalize,
      - CDF over frequency bins,
      - frequency at which CDF crosses 0.5.
    
    Parameters
    ----------
    df : pd.DataFrame
        Recording data (T x N_neurons)
    d : float
        Sampling interval (seconds)
    apply_delta_ff : bool
        Apply ΔF/F normalization
    target_fft_length : int, optional
        If provided, zero-pad signals to this length before FFT.
        Ensures consistent frequency grid across recordings.
        If None, uses signal length as-is.
    
    Returns
    -------
    f50 : float
        Frequency at 50% spectral power
    freqs : np.ndarray
        Frequency grid (Hz)
    psd_avg_norm : np.ndarray
        Normalized averaged PSD
    cdf : np.ndarray
        Cumulative distribution function of power
    """
    df_proc = compute_delta_ff(df) if apply_delta_ff else df

    psd_list = []
    n_time = df_proc.shape[0]
    for col in df_proc.columns:
        x = df_proc[col].values
        x_fft = pad_signal_for_fft(x, target_fft_length) if target_fft_length is not None else x
        X = np.fft.rfft(x_fft)
        psd_list.append(np.abs(X) ** 2)

    if len(psd_list) == 0:
        return np.nan, None, None, None

    psd_avg = np.mean(np.vstack(psd_list), axis=0)
    total_power = psd_avg.sum()
    if total_power == 0:
        return np.nan, None, None, None

    psd_avg_norm = psd_avg / total_power
    n_fft = target_fft_length if target_fft_length is not None else n_time
    freqs = np.fft.rfftfreq(n_fft, d=d)
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
    apply_delta_ff: bool = False,
    fft_length_strategy: str = 'max'
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Compute per-recording metrics for each condition:
      - avg_fraction (mean neuron band fraction),
      - pct_neurons_above (Figure S2B metric),
      - fractions per neuron,
      - f50 spectral edge and PSD/CDF (Figure S2A).
    
    CRITICAL: All recordings are zero-padded to the same FFT length to ensure
    consistent frequency grid across all datasets.
    
    Parameters
    ----------
    all_dfs : dict[(condition, wavelength)] -> list[pd.DataFrame]
        Recording data for all conditions
    sampling_intervals_all : dict[(condition, wavelength)] -> list[float]
        Sampling intervals (seconds) for each recording
    apply_delta_ff : bool, default=False
        Apply ΔF/F normalization
    fft_length_strategy : {'max', 'median'}, default='max'
        Strategy for determining common FFT length:
        - 'max': Use longest recording (finest frequency resolution)
        - 'median': Use median recording length (balance resolution vs padding)
    
    Returns
    -------
    dict[(condition, wavelength)] -> list[dict]
        Per-recording metrics with keys: 'avg_fraction', 'pct_neurons_above',
        'fractions', 'f50', 'freqs', 'psd_avg_norm', 'cdf'
    """
    # Determine common FFT length across all recordings
    target_fft_length = determine_common_fft_length(all_dfs, strategy=fft_length_strategy)
    
    results = {}
    for key, df_list in all_dfs.items():
        d_list = sampling_intervals_all[key]
        assert len(df_list) == len(d_list), f"Sampling intervals length mismatch for key {key}"
        metrics_list = []
        for df, d in zip(df_list, d_list):
            avg_fraction, pct_neurons_above, fractions = recording_metrics(
                df, d, F_LOW, F_HIGH, THRESHOLD_FRACTION, apply_delta_ff=apply_delta_ff,
                target_fft_length=target_fft_length
            )
            f50, freqs, psd_avg_norm, cdf = spectral_edge_50(
                df, d, apply_delta_ff=apply_delta_ff, target_fft_length=target_fft_length
            )
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
                        subplot_titles=("Normalized<br>PSD", "CDF"),
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
    fig.update_yaxes(title_text="Normalized<br>PSD", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 2 * F_HIGH], row=2, col=1)
    fig.update_layout(title="",
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

    # Build flat DataFrame for px.box
    rows = []
    for (cond, wavelength), recs in results.items():
        for r in recs:
            if not np.isnan(r['pct_neurons_above']):
                rows.append({
                    'condition': cond,
                    'wavelength': str(wavelength),
                    'pct_neurons_above': r['pct_neurons_above']
                })
    df = pd.DataFrame(rows)

    fig = px.box(
        df,
        x='condition',
        y='pct_neurons_above',
        color='wavelength',
        color_discrete_map={str(w): c for w, c in color_map.items()},
        points='all',                   # show all individual points
        title=f"Figure S2B: % neurons with fraction in [{F_LOW}, {F_HIGH}] Hz > {THRESHOLD_FRACTION} [^2]",
        labels={
            'condition': 'Condition',
            'pct_neurons_above': '% of neurons > threshold',
            'wavelength': 'Wavelength'
        },
        height=500,
    )

    fig.update_layout(boxmode='group')

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


def compute_fig1_metrics(
    all_dfs,
    sampling_intervals_all,
    apply_delta_ff=False,
    fft_length_strategy='max',
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
      - Optionally performs one-sided Welch's t-tests with Benjamini–Hochberg correction across specified condition pairs [^2].

    Parameters
    ----------
    all_dfs : dict[(condition_name, laser_wavelength) -> list[pd.DataFrame]]
        Each DataFrame is T x N_neurons for a single recording.
    sampling_intervals_all : dict[(condition_name, laser_wavelength) -> list[float]]
        Sampling intervals (seconds) aligned to each recording [^1].
    apply_delta_ff : bool
        If True, apply ΔF/F per neuron before spectral analysis [^2].
    fft_length_strategy : {'max', 'median'}, default='max'
        Strategy for determining common FFT length across all recordings [^1]
    comparisons : list[tuple[(cond_name_A, λ_A), (cond_name_B, λ_B)]] or None
        Optional condition-pair comparisons for stats on Fig 1F (f50) metric. H1: mean(condA) > mean(condB) [^2].
    alternative : {'greater', 'less', 'two-sided'}
        Alternative for Welch's t-test (default 'greater') [^2].
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
    results = compute_all_metrics(all_dfs, sampling_intervals_all, apply_delta_ff=apply_delta_ff,
                                  fft_length_strategy=fft_length_strategy)

    per_condition = {}
    for key, metrics_list in results.items():
        avg_fracs = [m['avg_fraction'] for m in metrics_list]
        f50_list = [m['f50'] for m in metrics_list]

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

    stats_out = {}
    if comparisons is not None and len(comparisons) > 0:
        pvals = []
        tstats = []
        pct_changes = []
        pairs = []
        for (condA, condB) in comparisons:
            arrA = np.array(per_condition[condA]['f50'], dtype=float)
            arrB = np.array(per_condition[condB]['f50'], dtype=float)
            res = welch_one_sided_ttest(arrA, arrB, alternative=alternative)
            pvals.append(res.pvalue)
            tstats.append(res.statistic)
            pairs.append((condA, condB))
            meanA = np.nanmean(arrA)
            meanB = np.nanmean(arrB)
            pct_change = 100.0 * (meanA - meanB) / meanB if np.isfinite(meanA) and np.isfinite(meanB) and meanB != 0 else np.nan
            pct_changes.append(pct_change)

        adj_pvals, passed = benjamini_hochberg(pvals, alpha=alpha)

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
    per_condition, category_orders=None
):
    """
    Simplified Plotly boxplots for Figure 1E and 1F using plotly.express:
      - x-axis: condition
      - color: wavelength
      - individual data points plotted as jittered dots overlaid on boxes

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
    wavelengths = ({w for (_, w) in per_condition.keys()})
    # Simple color palette mapping for wavelengths
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wavelengths)}
    condition_name_mapping = {0: "First", 1: "Middle", 2: "Last", 3: "Leifer conditions"}

    # Prepare data for Fig 1E as a list of dicts
    data_1e = []
    for (cond, w), vals in per_condition.items():
        for v in vals['avg_fractions']:
            if not np.isnan(v):
                data_1e.append({
                    'Condition': condition_name_mapping[cond],
                    'Wavelength': w,
                    'Value': v
                })

    df_1e = pd.DataFrame(data_1e)
    fig1e = px.box(df_1e, x='Condition', y='Value', color='Wavelength',
                   color_discrete_map=color_map, points='all', category_orders=category_orders)
    fig1e.update_layout(
        title="",
        xaxis_title="Condition",
        yaxis_title=f"Average fraction of power<br>in [{F_LOW}, {F_HIGH}] Hz",
        boxmode='group',
        height=500
    )

    # Prepare data for Fig 1F as a list of dicts
    data_1f = []
    for (cond, w), vals in per_condition.items():
        for v in vals['f50']:
            if not np.isnan(v):
                data_1f.append({
                    'Condition': condition_name_mapping[cond],
                    'Wavelength': w,
                    'Value': v
                })

    df_1f = pd.DataFrame(data_1f)
    fig1f = px.box(df_1f, x='Condition', y='Value', color='Wavelength',
                   color_discrete_map=color_map, points='all',
                   category_orders=category_orders)
    fig1f.update_layout(
        title="",
        xaxis_title="Condition",
        yaxis_title="Median of the<br>mean Power Spectrum (Hz)",
        boxmode='group',
        height=500
    )

    return {'Fig1E': fig1e, 'Fig1F': fig1f}


def prepare_s2a_spectral_data_for_mean_and_shading(results, value_column):
    """
    Prepare S2A spectral data (PSD or CDF) for plotly_plot_mean_and_shading.
    One row per frequency point per recording; NO pre-averaging within groups.
    plotly_plot_mean_and_shading computes mean/std per group downstream.
    
    Parameters
    ----------
    results : dict[(condition, wavelength)] -> list[dict]
        Output from compute_all_metrics. Each dict has 'freqs' and value_column.
        Condition must be in [0,1,2,3] for plot_s2a_mean_and_shading grid layout.
    value_column : str
        Name of the column to extract ('psd_avg_norm' or 'cdf').
    
    Returns
    -------
    df : pd.DataFrame
        Columns: ['frequency', value_column, 'group']
        Format: "condition_id | wavelength_id"
        Rows = sum(num_recordings * num_frequencies)
    """
    data_list = []
    for (cond, wavelength), recs in results.items():
        group_id = f"{cond} | {wavelength}"
        
        by_len = {}
        for r in recs:
            freqs, values = r['freqs'], r[value_column]
            if freqs is None or values is None:
                continue
            L = len(freqs)
            by_len.setdefault(L, {'freqs': [], 'values': []})
            by_len[L]['freqs'].append(freqs)
            by_len[L]['values'].append(values)
        
        for L, grp in by_len.items():
            freqs_ref = grp['freqs'][0]
            for values_array in grp['values']:
                for freq, val in zip(freqs_ref, values_array):
                    data_list.append({
                        'frequency': float(freq),
                        value_column: float(val),
                        'group': group_id
                    })
    return pd.DataFrame(data_list)


def prepare_s2a_psd_data_for_mean_and_shading(results):
    return prepare_s2a_spectral_data_for_mean_and_shading(results, 'psd_avg_norm')


def prepare_s2a_cdf_data_for_mean_and_shading(results):
    return prepare_s2a_spectral_data_for_mean_and_shading(results, 'cdf')


def axis_ref(row_idx, col_idx, n_cols):
    """Plotly axis helper function"""
    linear_idx = (row_idx - 1) * n_cols + col_idx
    suffix = '' if linear_idx == 1 else str(linear_idx)
    return f'x{suffix}', f'y{suffix}'
    

def plot_s2a_mean_and_shading(results, shade_style='std', cmap=None, DEBUG=False):
    """
    Plot S2A spectral data (PSD and CDF) with mean lines and confidence shading.
    ARCHITECTURE: 2 rows × 4 cols grid (row 1=PSD, row 2=CDF; cols=conditions 0-3).
    Each subplot shows all wavelengths for that condition (color-coded by wavelength).
    
    WORKFLOW:
    1. Prepare flat DataFrames (one row per freq/recording/group)
    2. Create 2×4 subplot grid
    3. For each condition (0-3):
       - Filter data by condition
       - Plot PSD and CDF separately, add traces to grid
       - plotly_plot_mean_and_shading computes mean/std per wavelength within condition
    4. Add band-edge lines (F_LOW, F_HIGH) to all subplots
    5. Set x-axis to [0, 2*F_HIGH] consistently
    
    Parameters
    ----------
    results : dict[(condition, wavelength)] -> list[dict]
        From compute_all_metrics. Condition MUST be in [0,1,2,3] for grid layout.
    shade_style : {'std', 'quantile'}, default='std'
        'std' = mean±1σ; 'quantile' = IQR
    cmap : dict, optional
        Color map {group_id -> color}. If None, uses make_wavelength_color_map.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        2×4 subplot grid (1400×800px). Each line is (condition, wavelength) pair.
        Shading shows variability across recordings within that group.
    """
    if DEBUG:
        print("DEBUG: Starting plot_s2a_mean_and_shading with results keys:", list(results.keys())[:5])
    # Prepare data
    df_psd = prepare_s2a_psd_data_for_mean_and_shading(results)
    df_cdf = prepare_s2a_cdf_data_for_mean_and_shading(results)
    
    # Build color map for groups if not provided
    if cmap is None:
        wavelength_cmap = make_wavelength_color_map(results)
        # Create a group-based color map directly from results
        group_cmap = {}
        for (cond, wavelength), _ in results.items():
            group_id = f"{cond} | {wavelength}"
            group_cmap[group_id] = wavelength_cmap[wavelength]
    else:
        group_cmap = cmap
    
    n_cols = 4

    # Create 2x4 subplot grid
    fig = make_subplots(
        rows=2, cols=n_cols,
        subplot_titles=[f"Condition {i}" for i in range(4)] + [""] * 4,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Iterate over conditions (0, 1, 2, 3)
    for col_idx, cond in enumerate([0, 1, 2, 3], start=1):
        # Filter data for this condition
        df_psd_cond = df_psd[df_psd['group'].str.contains(f"^{cond} \\|", regex=True)]
        df_cdf_cond = df_cdf[df_cdf['group'].str.contains(f"^{cond} \\|", regex=True)]

        if DEBUG:
            print(f"DEBUG: Condition {cond} - PSD rows: {len(df_psd_cond)}, CDF rows: {len(df_cdf_cond)}")
        
        # Plot PSD (row 1)
        if len(df_psd_cond) > 0:
            fig_psd_cond = plotly_plot_mean_and_shading(
                df_psd_cond,
                x='frequency',
                y='psd_avg_norm',
                color='group',
                line_name='Mean',
                shade_style=shade_style,
                cmap=group_cmap,
                DEBUG=DEBUG
            )
            # Add traces from individual figure to subplot
            for trace in fig_psd_cond.data:
                xref, yref = axis_ref(row_idx=1, col_idx=col_idx, n_cols=n_cols)
                trace.update(xaxis=xref, yaxis=yref)
                fig.add_trace(trace, row=1, col=col_idx)
            
        
        # Plot CDF (row 2)
        if len(df_cdf_cond) > 0:
            fig_cdf_cond = plotly_plot_mean_and_shading(
                df_cdf_cond,
                x='frequency',
                y='cdf',
                color='group',
                line_name='Mean',
                shade_style=shade_style,
                cmap=group_cmap,
                DEBUG=DEBUG
            )
            for trace in fig_cdf_cond.data:
                xref, yref = axis_ref(row_idx=2, col_idx=col_idx, n_cols=n_cols)
                trace.update(xaxis=xref, yaxis=yref)
                fig.add_trace(trace, row=2, col=col_idx)
    
    # Update axes
    fig.update_yaxes(title_text="Normalized<br>PSD", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2)
    
    # Add vertical lines at band edges for all subplots
    for col_idx in range(1, 5):
        fig.add_vline(x=F_LOW, line=dict(color='black', dash='dot', width=1), 
                     row=1, col=col_idx)
        fig.add_vline(x=F_HIGH, line=dict(color='black', dash='dot', width=1), 
                     row=1, col=col_idx)
        fig.add_vline(x=F_LOW, line=dict(color='black', dash='dot', width=1), 
                     row=2, col=col_idx)
        fig.add_vline(x=F_HIGH, line=dict(color='black', dash='dot', width=1), 
                     row=2, col=col_idx)
    
    # Limit x-axis to 2*F_HIGH for all subplots
    for col_idx in range(1, 5):
        fig.update_xaxes(range=[0, 2 * F_HIGH], row=1, col=col_idx)
        fig.update_xaxes(range=[0, 2 * F_HIGH], row=2, col=col_idx)
    
    fig.update_layout(
        title="",
        height=800,
        width=1400
    )
    
    return fig


def prepare_s2b_data_for_mean_and_shading(results):
    """
    Flatten S2B metrics (% neurons above threshold) for plotly_plot_mean_and_shading.
    
    Converts results dict to DataFrame with one row per recording per (condition, wavelength).
    NO pre-aggregation; plotly_plot_mean_and_shading computes mean/std per wavelength.
    
    Parameters
    ----------
    results : dict[(condition, wavelength)] -> list[dict]
        From compute_all_metrics. Each dict has 'pct_neurons_above' key.
    
    Returns
    -------
    df_s2b : pd.DataFrame
        Columns: ['condition', 'wavelength', 'pct_neurons_above']
        Index: one row per recording per (condition, wavelength) pair
    """
    data_list = []
    for (cond, wavelength), recs in results.items():
        for rec in recs:
            if not np.isnan(rec['pct_neurons_above']):
                data_list.append({
                    'condition': cond,
                    'wavelength': str(wavelength),
                    'pct_neurons_above': rec['pct_neurons_above']
                })
    return pd.DataFrame(data_list)


def prepare_s2c_data_for_mean_and_shading(quiet_counts, totals):
    """
    Flatten S2C metrics (proportion quiet recordings) for plotly_plot_mean_and_shading.
    
    Converts quiet_counts and totals dicts to DataFrame with one row per group
    (condition, wavelength). Returns the proportion for each group.
    
    Note: Unlike S2A/S2B prepare functions which return per-recording data for
    downstream aggregation, this function receives pre-aggregated counts and
    returns one row per group with the computed proportion.
    
    Parameters
    ----------
    quiet_counts : dict[label_str] -> int
        Count of quiet recordings per group, label format "condition | wavelength"
    totals : dict[label_str] -> int
        Total recordings per group, label format "condition | wavelength"
    
    Returns
    -------
    df_s2c : pd.DataFrame
        Columns: ['condition', 'wavelength', 'prop_quiet']
        One row per group with calculated proportion
    """
    data_list = []
    for label in quiet_counts.keys():
        cond = label.split('|')[0].strip()
        wavelength = label.split('|')[1].strip()
        prop = (quiet_counts[label] / totals[label]) if totals[label] > 0 else np.nan
        if not np.isnan(prop):
            data_list.append({
                'condition': cond,
                'wavelength': str(wavelength),
                'prop_quiet': prop
            })
    return pd.DataFrame(data_list)


def plot_s2b_mean_and_shading(df_s2b, shade_style='std', cmap=None):
    """
    Plot S2B data (% neurons above threshold) with mean lines and confidence shading.
    
    Plots each wavelength as a separate line across conditions, with shading showing
    variability across recordings for that wavelength within each condition.
    
    Parameters
    ----------
    df_s2b : pd.DataFrame
        Output from prepare_s2b_data_for_mean_and_shading.
        Columns: ['condition', 'wavelength', 'pct_neurons_above']
        One row per recording per (condition, wavelength) pair.
    shade_style : {'std', 'quantile'}, default='std'
        'std' = mean±1σ; 'quantile' = IQR (25th-75th percentile)
    cmap : dict, optional
        Color map {wavelength_str -> color}. If None, uses make_wavelength_color_map.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Single figure with wavelength traces and confidence regions.
        X-axis: conditions; Y-axis: % neurons above threshold
    """
    if cmap is None:
        cmap = make_wavelength_color_map({('cond', w): None for w in df_s2b['wavelength'].unique()})
        # Convert to format expected by plotly_plot_mean_and_shading
        cmap = {str(w): color for w, color in cmap.items()}
    
    fig = plotly_plot_mean_and_shading(
        df_s2b, 
        x='condition', 
        y='pct_neurons_above', 
        color='wavelength',
        line_name='Mean',
        shade_style=shade_style,
        cmap=cmap
    )
    fig.update_layout(
        title=f"Figure S2B: % neurons with fraction in [{F_LOW}, {F_HIGH}] Hz > {THRESHOLD_FRACTION} [^2]",
        xaxis_title="Condition",
        yaxis_title="% of neurons > threshold",
        height=500
    )
    return fig


def plot_s2c_mean_and_shading(df_s2c, shade_style='std', cmap=None):
    """
    Plot S2C data (proportion quiet recordings) with mean lines and confidence shading.
    
    Plots each wavelength as a separate line across conditions, with shading showing
    variability across recordings for that wavelength within each condition.
    
    Parameters
    ----------
    df_s2c : pd.DataFrame
        Output from prepare_s2c_data_for_mean_and_shading.
        Columns: ['condition', 'wavelength', 'prop_quiet']
        One row per recording per (condition, wavelength) pair.
    shade_style : {'std', 'quantile'}, default='std'
        'std' = mean±1σ; 'quantile' = IQR (25th-75th percentile)
    cmap : dict, optional
        Color map {wavelength_str -> color}. If None, uses make_wavelength_color_map.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Single figure with wavelength traces and confidence regions.
        X-axis: conditions; Y-axis: proportion of quiet recordings [0, 1]
    """
    if cmap is None:
        cmap = make_wavelength_color_map({('cond', w): None for w in df_s2c['wavelength'].unique()})
        # Convert to format expected by plotly_plot_mean_and_shading
        cmap = {str(w): color for w, color in cmap.items()}
    
    fig = plotly_plot_mean_and_shading(
        df_s2c, 
        x='condition', 
        y='prop_quiet', 
        color='wavelength',
        line_name='Mean',
        shade_style=shade_style,
        cmap=cmap
    )
    fig.update_layout(
        title=f"Figure S2C: Proportion of quiet recordings (quiet_threshold={QUIET_THRESHOLD}%)",
        xaxis_title="Condition",
        yaxis_title="Proportion of quiet recordings",
        height=500,
        yaxis_range=[0, 1]
    )
    return fig


# ==========================
# Driver function
# ==========================

def reproduce_figures_plotly(
    all_dfs: Dict[Tuple[str, str], List[pd.DataFrame]],
    sampling_intervals_all: Dict[Tuple[str, str], List[float]],
    apply_delta_ff: bool = False,
    quiet_threshold: float = QUIET_THRESHOLD,
    use_mean_and_shading: bool = False,
    shade_style: str = 'std',
    cmap: dict = None,
    fft_length_strategy: str = 'max',
    DEBUG=False
):
    """
    Generate S2A/S2B/S2C supplement figures with optional dual-mode visualization.
    
    MODES:
    - default (use_mean_and_shading=False): Original stacked boxplots/bars
    - mean+shading (use_mean_and_shading=True): Mean lines with confidence regions
    
    S2A SPECIAL CASE: 2×4 GRID LAYOUT
    When use_mean_and_shading=True, S2A displays 2 rows × 4 cols:
    - Rows: Row 1=PSD (power spectral density), Row 2=CDF (cumulative distribution)
    - Cols: Condition 0, 1, 2, 3 (side-by-side for easy comparison)
    Each subplot shows all wavelengths color-coded with mean + shading per wavelength.
    
    CRITICAL DATA FORMAT DECISION:
    Data prep functions (prepare_s2a_*) DO NOT pre-average.
    They produce one row per (frequency, recording, group).
    Aggregation happens downstream in plotly_plot_mean_and_shading:
    - Groups by (condition, wavelength)
    - Computes mean/std across recordings for that group
    This keeps concerns separated and avoids cross-condition averaging bugs.
    
    Parameters
    ----------
    all_dfs : dict[(condition, wavelength)] -> list[pd.DataFrame]
        Recording trace data from compute_all_metrics
    sampling_intervals_all : dict[(condition, wavelength)] -> list[float]
        Recording durations (seconds)
    apply_delta_ff : bool, default=False
        Apply ΔF/F normalization per neuron
    quiet_threshold : float, default=QUIET_THRESHOLD
        Activity threshold for quiet classification
    use_mean_and_shading : bool, default=False
        If True: S2A uses 2×4 grid; S2B/S2C use line plots
        If False: Original stacked plots
    shade_style : {'std', 'quantile'}, default='std'
        'std' = mean±1σ; 'quantile' = IQR (25th-75th percentile)
    cmap : dict, optional
        Color map {wavelength -> color}. If None, uses make_wavelength_color_map.
    fft_length_strategy : {'max', 'median'}, default='max'
        Strategy for determining common FFT length across recordings:
        - 'max': Use longest recording (finest frequency resolution)
        - 'median': Use median recording length
    
    Returns
    -------
    dict with keys:
        'results' : dict[(condition, wavelength)] -> list[dict], metrics per recording
        'figures' : dict with 'S2A', 'S2B', 'S2C' plotly figures
        'stats' : dict with statistical test results
    """
    if DEBUG:
        print("DEBUG: Starting figure reproduction with the following parameters:")
        print(f"apply_delta_ff={apply_delta_ff}, quiet_threshold={quiet_threshold}, use_mean_and_shading={use_mean_and_shading}, shade_style={shade_style}")
        print(f"fft_length_strategy={fft_length_strategy}")
        print(f"Number of conditions: {len(all_dfs)}")
        for key in all_dfs.keys():
            print(f"Condition {key}: {len(all_dfs[key])} recordings")
    # Compute metrics per recording with standardized FFT length
    results = compute_all_metrics(all_dfs, sampling_intervals_all, apply_delta_ff=apply_delta_ff,
                                  fft_length_strategy=fft_length_strategy)

    # Figure S2A-like
    if use_mean_and_shading:
        fig_s2a = plot_s2a_mean_and_shading(results, shade_style=shade_style, cmap=cmap, DEBUG=DEBUG)
    else:
        fig_s2a = plot_s2a_plotly_simple(results)

    # Figure S2B-like
    fig_s2b = plot_s2b_plotly_simple(results)

    # Figure S2C-like
    quiet_counts, totals = classify_quiet(results, quiet_threshold=quiet_threshold)
    if use_mean_and_shading:
        df_s2c = prepare_s2c_data_for_mean_and_shading(quiet_counts, totals)
        fig_s2c = plot_s2c_mean_and_shading(df_s2c, shade_style=shade_style, cmap=cmap)
    else:
        fig_s2c = plot_s2c_plotly_simple(quiet_counts, totals)

    # S2C stats (adjacent pairs)
    s2c_stats = compute_s2c_stats(quiet_counts, totals)

    s2b_stats = dict()

    return {
        'results': results,
        'figures': {'S2A': fig_s2a, 'S2B': fig_s2b, 'S2C': fig_s2c},
        'stats': {'S2B': s2b_stats, 'S2C': s2c_stats}
    }


def rebuttal_trace_opt():
    return {'use_paper_options': False, 'interpolate_nan': True, 
            'rename_neurons_using_manual_ids': True, 'manual_id_confidence_threshold': 0,
            'high_pass_bleach_correct': True, 'remove_invalid_neurons': True}


def load_laser_switch_experiments_as_subprojects(all_projects_505_488_505, all_projects_488_505_488, 
                                                 immob=True, dataset_as_outer_key=False, DEBUG=False):
    """Uses manually annotated or automatically detected laser switch times to split the 505-488-505 and 488-505-488 projects into sub-projects for each laser wavelength segment. Returns a dict of sub-projects keyed by (segment_index, laser_wavelength)."""
    all_sub_projects = defaultdict(dict)
    manual_split_annotation = manual_annotation_of_dataset_splits(immob)

    laser_wavelengths = [505, 488, 505]
    for name, p in all_projects_505_488_505.items():
        # For each project, split it into 3 and then append to the appropriate list
        
        starts_stops = manual_split_annotation.get(p.shortened_name, None)
        if starts_stops is None:
            starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
        all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

        for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
            all_sub_projects[(i, laser_wavelength)][seg.shortened_name] = seg
        
        if DEBUG:
            print(f"Only returning one project: {name} split into segments with laser wavelengths: {laser_wavelengths}")
            for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
                print(f"  Segment {i}: Laser {laser_wavelength} nm, Name: {seg.shortened_name}, Time range: {starts_stops[i]}")
            break
    
    laser_wavelengths = [488, 505, 488]
    for name, p in all_projects_488_505_488.items():
        # For each project, split it into 3 and then append to the appropriate list

        starts_stops = manual_split_annotation.get(p.shortened_name, None)
        if starts_stops is None:
            starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
        all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
        
        for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
            all_sub_projects[(i, laser_wavelength)][seg.shortened_name] = seg

        if DEBUG:
            print(f"Only returning one project: {name} split into segments with laser wavelengths: {laser_wavelengths}")
            for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
                print(f"  Segment {i}: Laser {laser_wavelength} nm, Name: {seg.shortened_name}, Time range: {starts_stops[i]}")
            break
    
    if dataset_as_outer_key:
        # Switch to dataset as outer key: {dataset_name: {(segment_index, laser_wavelength): {project_name: project}}}
        all_sub_projects_by_dataset = defaultdict(dict)
        for (segment_index, laser_wavelength), projects in all_sub_projects.items():
            for project_name, project in projects.items():
                all_sub_projects_by_dataset[project_name][(segment_index, laser_wavelength)] = project
        return all_sub_projects_by_dataset
    else:
        return all_sub_projects


def manual_annotation_of_dataset_splits(immob):
    if immob:
        manual_split_annotation = {'2025-09-15_17-12_505_6min_488_6min_561_6min_worm2-2025-09-15':
                                    [[0, 774], [776, 1527], [1529, 2269]],
                                '2025-09-15_16-47_505_6min_488_6min_561_6min_worm1-2025-09-15':
                                    [[0, 777], [778, 1525], [1527, 2269]],
                                '2025-09-15_17-38_505_6min_488_6min_561_6min_worm3-2025-09-15':
                                    [[0, 774], [776, 1526], [1528, 2269]],
                                '2025-09-15_15-55_488_6min_505_6min_488_6min_worm5-2025-09-15':
                                    [[0, 776], [778, 1527], [1530, 2269]],
                                '2025-09-15_15-30_488_6min_505_6min_488_6min_worm4-2025-09-15':
                                    [[0, 774], [781, 1540], [1542, 2269]],
                                '2025-10-02_14-11_505_6min_488_6min_561_6min_worm1_100uW-2025-10-02':
                                    [[0, 773], [783, 1520], [1531, 2269]],
                                '2025-09-23_14-15_505_6min_488_6min_561_6min_worm1-2025-09-23':
                                    [[0, 769], [781, 1521], [1532, 2269]],
                                '2025-09-23_12-18_488_6min_505_6min_488_6min_worm1-2025-09-23':  # Laser was off for a long time in the middle, so removing that part
                                    [[0, 770], [822, 1527], [1541, 2269]],
                                '2025-09-23_14-42_488_6min_505_6min_488_6min_worm2-2025-09-23':
                                    [[0, 772], [783, 1523], [1533, 2269]],
                                '2025-09-23_15-10_505_6min_488_6min_561_6min_worm2-2025-09-23':
                                    [[0, 776], [787, 1519], [1530, 2269]],
                                    # Inactive
                                '2025-10-02_14-35_505_6min_488_6min_505_6min_worm2_100uW-2025-10-02':
                                    [[0, 769], [782, 1519], [1531, 2269]],
                                '2025-10-02_14-58_505_6min_488_6min_505_6min_worm3_100uW-2025-10-02':
                                    [[0, 787], [801, 1527], [1538, 2269]],
                                '2025-09-15_13-40_488_6min_505_6min_488_6min_worm1-2025-09-15': #Giant artifact (just removing)????????
                                    [[0, 787], [801, 1527], [1538, 2269]],
                                '2025-09-15_14-47_488_6min_505_6min_488_6min_worm3-2025-09-15':
                                    [[0, 774], [779, 1527], [1529, 2269]],
                                '2025-10-02_13-46_488_6min_505_6min_488_6min_worm2_100uW-2025-10-02':
                                    [[0, 770], [781, 1520], [1531, 2269]],
                                '2025-09-15_16-21_488_6min_505_6min_488_6min_worm6-2025-09-15':
                                    [[0, 778], [780, 1525], [1528, 2269]],
                                '2025-09-15_14-10_488_6min_505_6min_488_6min_worm2-2025-09-15':
                                    [[0, 771], [778, 1521], [1531, 2269]],
                                '2025-10-02_10-41_488_6min_505_6min_488_6min_worm1_100uW-2025-10-02':
                                    [[0, 769], [787, 1520], [1532, 2269]],
                                }
    else:
        manual_split_annotation = {'2025-11-20_12-56_shifts_505_488_505_worm3-2025-11-20':
                                [[0, 797], [813, 1660], [1675, 2499]],
                            '2025-11-20_11-35_shifts_505_488_505_worm1-2025-11-20':
                                [[0, 835], [852, 1668], [1685, 2499]],
                            '2025-11-20_12-21_shifts_505_488_505_worm2-2025-11-20':
                                [[0, 799], [818, 1680], [1696, 2499]],
                            '2025-11-20_11-52_shifts_488_505_488_worm2-2025-11-20':
                                [[0, 839], [859, 1668], [1685, 2499]],
                            '2025-11-20_11-08_shifts_488_505_488_worm1-2025-11-20':
                                [[0, 870], [892, 1725], [1742, 2499]],
                            '2025-11-20_12-39_shifts_488_505_488_worm3-2025-11-20':
                                [[0, 811], [829, 1645], [1662, 2499]],
                            }
                            
    return manual_split_annotation


def make_heatmap_stack(these_heatmaps: dict, these_ethograms: dict, output_folder=None, prefix='', DEBUG=False):
        
    n = len(these_heatmaps)
    base_row_heights = np.array([0.85, 0.1, 0.05])
    base_row_heights = list(base_row_heights / base_row_heights.sum())
    
    all_row_heights = list(np.array(n*base_row_heights) / n)
    
    subplot_opt = dict(rows=len(all_row_heights), row_heights=all_row_heights, 
                    vertical_spacing=0.0
                    )
    
    all_figs = []
    for k in these_heatmaps.keys():
        all_figs.append(these_heatmaps[k])
        all_figs.append(extract_shapes_as_figure(these_ethograms[k], only_include_shapes_with_yref='y'))
        all_figs.append(go.Figure())  # Dummy empty figure
    
    fig = combine_plotly_figures(all_figs, horizontal=False, custom_subplot_opt=subplot_opt, hide_interior_xlabels=True,
                                force_yref_paper=False)
    
    fig.update_yaxes(title="", showticklabels=False, overwrite=True)
    fig.update_xaxes(title="", showticklabels=False, overwrite=True)
    fig.update_xaxes(title="Seconds", showticklabels=True, row=len(all_row_heights), overwrite=True)
    
    apply_figure_settings(fig=fig, width_factor=0.15, height_factor=1.0)
    
    if output_folder is not None:
        fname = os.path.join(output_folder, f'stacked_heatmaps_with_ethograms-{prefix}.png')
        fig.write_image(fname, scale=3)

    return fig


def add_vline_based_on_splits(_fig, vps, splits):
    for s in splits[:-1]:
        opt = dict(x=s[1]/vps, line_width=3, line_color='black')#, line_dash='dash')
        _fig.add_vline(**opt, y0=0, y1=1)


def make_heatmap(dat, splits=None, vps=None):
    x_for_plots_volumes = dat.columns
    heatmap = go.Heatmap(y=dat.index, z=dat, x=x_for_plots_volumes,
                         zmin=-0.25, zmax=1.25, colorscale='jet', xaxis="x", yaxis="y",
                         coloraxis='coloraxis1')
    
    fig = go.Figure()
    fig.add_trace(heatmap)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False, autosize=False, #**plotly_opt,
                       coloraxis=dict(colorscale="jet"))
    
    fig.update_coloraxes(cmin=-0.25, cmax=0.75, colorbar=dict(
        # thickness=10,
        # title=dict(text=r'ΔR / R₅₀', **font_dict)
        # title=dict(text=r'$\frac{\Delta R}{R_{50}}$', **font_dict)
    ))

    if splits is not None and vps is not None:
        add_vline_based_on_splits(fig, vps, splits)

    return fig


def make_ethogram(df_beh, splits=None, vps=None, use_alternate_cmap=False):
    ethogram_cmap_opt = dict()
    
    ethogram_opt = options_for_ethogram(df_beh, **ethogram_cmap_opt, include_turns=False,
                                        to_extend_short_states=False, use_alternate_cmap=use_alternate_cmap)
    fig_beh = go.Figure()
    fig_beh.update_layout(shapes=[opt for opt in ethogram_opt])

    if splits is not None and vps is not None:
        add_vline_based_on_splits(fig_beh, vps, splits)

    return fig_beh


def get_data_from_subprojects(these_subprojects, all_splits):
    
    these_traces = []
    these_beh = []

    # Get individual datasets
    for (i_seg, wavelength), seg in tqdm(these_subprojects.items(), total=len(these_subprojects), leave=False):
        fig1, fig2, results = make_summary_heatmap_and_subplots(seg, trace_opt=rebuttal_trace_opt(), 
                                                       to_save=False, to_show=False, include_speed_subplot=False,
                                                       base_height=[0.25, 0.2], base_width=1.0, output_folder=None)
        
        name = seg.shortened_name  # Note that this is the same as the parent project
        offset = all_splits[name][i_seg][0]
        
        _df = results['neural_activity'].T.reset_index(drop=True).copy()
        _df.index += offset
        these_traces.append(_df.T)

        _df = results['beh_vec'].reset_index(drop=True).copy()
        _df.index += offset
        these_beh.append(_df)

    # Combine and fix indices (with gaps)
    df_traces = pd.concat(these_traces, axis=1).copy()
    
    vps = seg.physical_unit_conversion.volumes_per_second  # Same for all segs
    df_traces = fill_missing_indices_with_nan(df_traces.T)[0]
    expected_index = df_traces.index
    df_traces.index /= vps
    df_traces = df_traces.T

    min_nonnan = 0.75
    min_nonnan = int(min_nonnan * df_traces.shape[1])
    df_traces = df_traces.dropna(thresh=min_nonnan)

    df_beh = pd.concat(these_beh)
    df_beh = fill_missing_indices_with_nan(df_beh, expected_index=expected_index)[0].fillna(BehaviorCodes.UNKNOWN)
    df_beh.index /= vps

    # Also map indices to wavelengths
    df_laser = df_beh.copy().reset_index(drop=True)
    df_laser[:] = np.nan
    for (i_seg, wavelength), seg in these_subprojects.items():
        idx = these_beh[i_seg].index
        df_laser.iloc[idx] = wavelength

    return df_traces, df_beh, df_laser, vps


def get_data_from_dict_of_subprojects(all_subprojects, all_splits, prefix=None, DEBUG=False):
    
    all_heatmaps = defaultdict(dict)
    all_ethograms = defaultdict(dict)
    all_df_traces = defaultdict(dict)
    all_df_beh = defaultdict(dict)
    all_df_laser = defaultdict(dict)

    for i, (name, these_subprojects) in tqdm(enumerate(all_subprojects.items()), total=len(all_subprojects)):
        df_traces, df_beh, df_laser, vps = get_data_from_subprojects(these_subprojects, all_splits)
        if prefix is None:
            # Dynamically determine the prefix based on the wavelength of the first segment... not great but consistent with other functions
            prefix = list(these_subprojects.keys())[0][1]
        
        if DEBUG:
            print(f"Processed {name} with prefix {prefix}: df_traces shape={df_traces.shape}, df_beh shape={df_beh.shape}, df_laser shape={df_laser.shape}")
            print(f"Number of behavior states in df_beh: {(df_beh[0].map(lambda x: x.value).diff() > 0).sum()}")

        all_df_traces[prefix][name] = df_traces
        all_df_beh[prefix][name] = df_beh
        all_df_laser[prefix][name] = df_laser
        all_heatmaps[prefix][name] = make_heatmap(df_traces, all_splits[name], vps)
        all_ethograms[prefix][name] = make_ethogram(df_beh, all_splits[name], vps, use_alternate_cmap=True)

    return all_heatmaps, all_ethograms, all_df_traces, all_df_beh, all_df_laser

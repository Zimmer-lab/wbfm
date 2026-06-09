# import
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from wbfm.utils.projects.finished_project_data import load_all_projects_in_folder
from wbfm.utils.general.utils_paper import split_time_series_with_laser_switches
from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_pandas import melt_nested_dict

manual_split_annotation = {'2025-09-15_17-12_505_6min_488_6min_561_6min_worm2-2025-09-15':
                              [[0, 774], [776, 1527], [1529, 2269]],
                           '2025-09-15_16-47_505_6min_488_6min_561_6min_worm1-2025-09-15':
                              [[0, 777], [778, 1525], [1527, 2269]],
                           '2025-09-15_17-38_505_6min_488_6min_561_6min_worm3-2025-09-15':
                              [[0, 774], [776, 1526], [1528, 2269]],
                           '2025-09-15_15-55_488_6min_505_6min_488_6min_worm5-2025-09-15':
                              [[0, 776], [778, 1402], [1404, 2269]],
                           '2025-09-15_15-30_488_6min_505_6min_488_6min_worm4-2025-09-15':
                              [[0, 774], [781, 1540], [1542, 2269]]}

def calc_durations_dataframe(all_projects, states=None,use_manual_annotation=False,remove_idx_of_tracking_failures=True):
    if states is None:
        states = [BehaviorCodes.FWD, BehaviorCodes.REV]
    # Note that the signed_speed_angular does not have a well-defined sign across datasets

    all_durations = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        try:
            for state in states:
                ind_class = p.worm_posture_class.calc_triggered_average_indices(state=state, min_duration=0, use_manual_annotation=use_manual_annotation,remove_idx_of_tracking_failures=remove_idx_of_tracking_failures)
                all_durations[str(state)][name] = ind_class.all_state_durations(include_censored=False)[0]
        except ValueError:
            continue
    df_durations = melt_nested_dict(all_durations, all_same_lengths=False)

    return df_durations

def analyze_laser_swap_datasets(
    datasets,
    laser_wavelengths=(488, 505, 488),
    calc_durations_dataframe=None,
    calc_speed_dataframe=None,
    rev_dur_manual_annot=False,
    show_progress=True,
):
    """
    Analyze datasets collected during laser wavelength switching experiments.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to project objects.
    laser_wavelengths : sequence of int, optional
        Laser wavelengths corresponding to each segment.
    calc_durations_dataframe : callable
        Function that computes duration statistics for a segment.
    calc_speed_dataframe : callable
        Function that computes speed statistics for a segment.
    rev_dur_manual_annot: bool, optional
        Whether to use manual annotation for reversal durations. If False, will use automatic annotation.

    show_progress : bool, optional
        Whether to display a tqdm progress bar.

    Returns
    -------
    tuple
        (df_reversals, df_speed, df_duration)
    """
    if calc_durations_dataframe is None:
        raise ValueError("calc_durations_dataframe must be provided.")

    if calc_speed_dataframe is None:
        raise ValueError("calc_speed_dataframe must be provided.")

    remove_idx_of_tracking_failures = True
    if rev_dur_manual_annot == True:
        print("Using manual annotation for reversal durations.")
        remove_idx_of_tracking_failures=False

    reversals = []
    speed_dfs = []
    duration_dfs = []

    iterator = tqdm(datasets.items()) if show_progress else datasets.items()

    for name, project in iterator:
        starts_stops = manual_split_annotation.get(project.shortened_name, None)
        print(f"manually annotated starts and stops for {project.shortened_name}: {starts_stops}")
        if starts_stops is None:
            starts_stops = split_time_series_with_laser_switches(project.green_traces, brightness_threshold=605e3)
            print(f"automatically detected starts and stops for {project.shortened_name}: {starts_stops}")
        all_segments = split_project_data_in_time(project, starts_stops, verbose=0)
        if not len(all_segments) == 3:
            print(f"Expected 3 segments for {project.shortened_name} but found {len(all_segments)}. Check the detected starts and stops: {starts_stops}")
            split_time_series_with_laser_switches(project.green_traces, brightness_threshold=2e4, DEBUG=True)
            raise ValueError

        segments = split_project_data_in_time(
            project,
            starts_stops,
            verbose=False,
        )

        for position, (wavelength, segment) in enumerate(
            zip(laser_wavelengths, segments)
        ):
            raw_num_rev = len(
                segment.worm_posture_class
                .get_starts_and_ends_of_reversals(use_manual_annotation=rev_dur_manual_annot,remove_idx_of_tracking_failures=remove_idx_of_tracking_failures)[0]
            )

            raw_pause_length = (
                segment.worm_posture_class
                .calc_behavior_from_alias("pause")
                .sum()
                / project.physical_unit_conversion.volumes_per_second
            )

            num_minutes = (
                project.num_frames
                / project.physical_unit_conversion.volumes_per_second
                / 60
            )

            reversals.append({
                "dataset_name": name,
                "position": position,
                "Laser wavelength": wavelength,
                "raw_num_rev": raw_num_rev,
                "raw_pause_length": raw_pause_length,
                "num_minutes": num_minutes,
                "rev_per_minute": raw_num_rev / num_minutes,
                "pause_per_minute": raw_pause_length / num_minutes,
            })

            duration_df = calc_durations_dataframe(
                {segment.shortened_name: segment},
            use_manual_annotation=rev_dur_manual_annot,
            remove_idx_of_tracking_failures=remove_idx_of_tracking_failures)

            duration_df["BehaviorCodes.FWD"] /= project.physical_unit_conversion.volumes_per_second
            duration_df["BehaviorCodes.REV"] /= project.physical_unit_conversion.volumes_per_second
            duration_df["position"] = position
            duration_df["Laser wavelength"] = wavelength
            duration_dfs.append(duration_df)

            speed_df = calc_speed_dataframe(
                {segment.shortened_name: segment}
            )
            speed_df["position"] = position
            speed_df["Laser wavelength"] = wavelength
            speed_dfs.append(speed_df)

    df_reversals = pd.DataFrame(reversals)
    df_speed = pd.concat(speed_dfs, ignore_index=True)
    df_duration = pd.concat(duration_dfs, ignore_index=True)

    return df_reversals, df_speed, df_duration

import itertools
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests


def pairwise_ks_test_annotation(
    df,
    value_column,
    group_column="Laser wavelength",
    correction_method="fdr_bh",
    significance_marker="*",
):
    """
    Perform pairwise Kolmogorov-Smirnov tests between groups and
    generate Plotly-ready annotation text.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    value_column : str
        Column containing the values to compare.
    group_column : str, optional
        Column defining the groups.
    correction_method : str, optional
        Multiple-testing correction method passed to multipletests().
    significance_marker : str, optional
        Symbol appended to significant comparisons.

    Returns
    -------
    results_df : pandas.DataFrame
        Statistical test results.
    annotation_text : str
        HTML-formatted annotation string.
    """
    groups = df[group_column].dropna().unique()

    comparisons = []
    raw_pvals = []

    for group1, group2 in itertools.combinations(groups, 2):
        data1 = df.loc[
            df[group_column] == group1,
            value_column,
        ].dropna()

        data2 = df.loc[
            df[group_column] == group2,
            value_column,
        ].dropna()

        statistic, p_value = ks_2samp(data1, data2)

        comparisons.append((group1, group2))
        raw_pvals.append(p_value)

    reject, corrected_pvals, _, _ = multipletests(
        raw_pvals,
        method=correction_method,
    )

    results = []
    annotation_lines = []

    for (group1, group2), raw_p, corr_p, is_significant in zip(
        comparisons,
        raw_pvals,
        corrected_pvals,
        reject,
    ):
        results.append({
            "group_1": group1,
            "group_2": group2,
            "raw_p_value": raw_p,
            "corrected_p_value": corr_p,
            "significant": is_significant,
        })

        line = f"{group1} vs {group2}: p={corr_p:.3e}"
        if is_significant:
            line += f" {significance_marker}"
        annotation_lines.append(line)

    results_df = pd.DataFrame(results)
    annotation_text = "<br>".join(annotation_lines)

    return results_df, annotation_text

# load datasets

datasets_488_swap = load_all_projects_in_folder(r'Z:\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\freely_moving_505\488_505_488')
datasets_505_swap = load_all_projects_in_folder(r'Z:\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\freely_moving_505\505_488_505')
no_light_datasets = load_all_projects_in_folder(r"Z:\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\freely_moving_505\not_light_control")


df_reversals_488, df_speed_488, df_duration_488 = analyze_laser_swap_datasets(datasets_488_swap,
                                                                              (488,505,488),
                                                                              calc_durations_dataframe,
                                                                              calc_speed_dataframe,
                                                                              rev_dur_manual_annot=False)
df_reversals_505, df_speed_505, df_duration_505 = analyze_laser_swap_datasets(datasets_505_swap,
                                                                              (505,488,505),
                                                                              calc_durations_dataframe,
                                                                              calc_speed_dataframe,
                                                                              rev_dur_manual_annot=False)

no_light_duration_df = calc_durations_dataframe(no_light_datasets,remove_idx_of_tracking_failures=True,use_manual_annotation=True)
no_light_duration_df['BehaviorCodes.FWD'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
no_light_duration_df['BehaviorCodes.REV'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
no_light_duration_df['Laser wavelength'] = 'no light'
no_light_duration_df['position'] = 3
df_duration_merged = pd.concat([df_duration_505,df_duration_488,no_light_duration_df])

### plotting and stats
beh_duration_name  = "BehaviorCodes.REV"


groups = df_duration_merged['Laser wavelength'].unique()

# collect p-values
comparisons = []
pvals = []

for g1, g2 in itertools.combinations(groups, 2):
    d1 = df_duration_merged[df_duration_merged['Laser wavelength'] == g1][beh_duration_name]
    d2 = df_duration_merged[df_duration_merged['Laser wavelength'] == g2][beh_duration_name]

    stat, p = ks_2samp(d1, d2)

    comparisons.append((g1, g2))
    pvals.append(p)

# multiple testing correction (e.g., Benjamini-Hochberg)
reject, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')

annotation_text = ""

for (g1, g2), p_corr, rej in zip(comparisons, pvals_corr, reject):
    annotation_text += f"{g1} vs {g2}: p={p_corr:.3}"
    if rej:
        annotation_text += " *"
    annotation_text += "<br>"

import plotly.express as px
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map

fig = px.histogram(
    df_duration_merged,
    color='Laser wavelength',
    x=beh_duration_name,
    histnorm="percent",
    color_discrete_map=plotly_paper_color_discrete_map(),
    category_orders={'Index name': ['min 0-6', 'min 6-12', 'min 12-18']},
    opacity=0.8,
    barmode="overlay",
    nbins=100,
    title="Histogram of reversal duration"
)

# 1: Add outline to bars
fig.update_traces(
    marker=dict(
        line=dict(
            width=1,
            color='black'
        )
    )
)


# 2: Axis titles + larger fonts
fig.update_layout(
    xaxis_title="Reversal duration (sec)",
    yaxis_title="Percent events",
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24)
)
fig.update_layout(
    legend=dict(
        font=dict(size=22),          # legend entries
        title_font=dict(size=24)     # legend title (optional but recommended)
    )
)

# 3: add stats
fig.add_annotation(
    text=annotation_text,
    xref="paper", yref="paper",
    x=1.02, y=1,
    showarrow=False,
    align="left",
    font=dict(size=18),
    bordercolor="black",
    borderwidth=1
)

import os
import plotly.express as px
fig.write_html(os.path.join(r"C:\Users\Itamar\Desktop", "505", f"{beh_duration_name}_duration_histogram_plot.html"))

df_avg_duration = (
    df_duration_merged
    .groupby(['position','dataset_name', 'Laser wavelength'])[beh_duration_name]
    .mean()
    .reset_index()
)
results_df, annotation_text = pairwise_ks_test_annotation(df_avg_duration,
                            beh_duration_name, group_column='Laser wavelength')

df_avg_duration['Index name'] = df_avg_duration['position'].map({0: 'min 0-4', 1:'min 4-8', 2:'min 8-12', 3:'Constant'})

import os
import plotly.express as px
### plotting and stats
beh_duration_name  = "BehaviorCodes.FWD"
df_duration_merged['Index name'] = df_duration_merged['position'].map({0: 'min 0-6', 1:'min 6-12', 2:'min 12-18', 3:'Constant'})
fig = px.box(df_duration_merged, color='Laser wavelength', y=beh_duration_name, points='all', x='Index name',
      color_discrete_map=plotly_paper_color_discrete_map())
fig.update_xaxes(type="category")
# 2: Axis titles + larger fonts
fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Reversal Duration (sec)" if beh_duration_name == "BehaviorCodes.REV" else "Forward Duration (sec)",
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24)
)
fig.update_layout(
    legend=dict(
        font=dict(size=22),          # legend entries
        title_font=dict(size=24)     # legend title (optional but recommended)
    )
)

def pairwise_group_stats(
    df,
    value_column,
    primary_group="Index Name",
    secondary_group="Laser wavelength",
    correction_method="fdr_bh",
    test_func=ks_2samp,
):
    """
    Perform pairwise statistical comparisons between levels of
    `secondary_group` within each level of `primary_group`.

    Example
    -------
    For each condition (e.g. 'min 0-4'), compare:
        488 vs 505
        488 vs no light
        505 vs no light

    Returns
    -------
    stats_df : pandas.DataFrame
        One row per comparison.
    """
    results = []

    for primary_value in df[primary_group].dropna().unique():
        subset = df[df[primary_group] == primary_value]

        groups = subset[secondary_group].dropna().unique()

        for g1, g2 in itertools.combinations(groups, 2):
            x1 = subset.loc[
                subset[secondary_group] == g1,
                value_column,
            ].dropna()

            x2 = subset.loc[
                subset[secondary_group] == g2,
                value_column,
            ].dropna()

            if len(x1) == 0 or len(x2) == 0:
                continue

            stat, p = test_func(x1, x2)

            results.append({
                primary_group: primary_value,
                "group_1": g1,
                "group_2": g2,
                "statistic": stat,
                "p_raw": p,
            })

    stats_df = pd.DataFrame(results)

    if len(stats_df) == 0:
        return stats_df

    reject, p_corr, _, _ = multipletests(
        stats_df["p_raw"],
        method=correction_method,
    )

    stats_df["p_corrected"] = p_corr
    stats_df["significant"] = reject
    stats_df["significance"] = stats_df["p_corrected"].apply(
        pvalue_to_stars
    )

    return stats_df


def pvalue_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def add_stat_annotations(
    fig,
    stats_df,
    x_order,
    hue_order,
    y_lookup,
    primary_group="Index name",
    secondary_group="Laser wavelength",
    y_padding=0.05,
    line_height=0.08,
):
    """
    Correct significance annotations for Plotly grouped boxplots.

    Important:
    Plotly categorical axes do NOT accept fractional x-values.
    We must use xshift in pixels instead.
    """

    # Pixel offsets for each hue
    pixel_offsets = {
        hue: offset
        for hue, offset in zip(
            hue_order,
            [-80, 0, 80][:len(hue_order)]
        )
    }

    for condition in x_order:
        subset = (
            stats_df.loc[
                stats_df[primary_group] == condition
            ]
            .reset_index(drop=True)
        )

        if subset.empty:
            continue

        base_y = y_lookup[condition]

        for i, row in subset.iterrows():
            g1 = row["group_1"]
            g2 = row["group_2"]

            y = base_y * (1 + y_padding + i * line_height)

            xshift1 = pixel_offsets[g1]
            xshift2 = pixel_offsets[g2]

            # Left vertical
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                xsizemode="pixel",
                ysizemode="scaled",
                xanchor=condition,
                x0=xshift1,
                x1=xshift1,
                y0=y * 0.98,
                y1=y,
                line=dict(width=1.5),
            )

            # Horizontal
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                xsizemode="pixel",
                ysizemode="scaled",
                xanchor=condition,
                x0=xshift1,
                x1=xshift2,
                y0=y,
                y1=y,
                line=dict(width=1.5),
            )

            # Right vertical
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                xsizemode="pixel",
                ysizemode="scaled",
                xanchor=condition,
                x0=xshift2,
                x1=xshift2,
                y0=y * 0.98,
                y1=y,
                line=dict(width=1.5),
            )

            fig.add_annotation(
                x=condition,
                y=y,
                xref="x",
                yref="y",
                xshift=(xshift1 + xshift2) / 2,
                text=row["significance"],
                showarrow=False,
                yshift=10,
                font=dict(size=14),
            )

    return fig

stats_df = pairwise_group_stats(df_avg_duration,beh_duration_name, primary_group='Index name', secondary_group='Laser wavelength')
"""
fig = add_stat_annotations(
    fig=fig,
    stats_df=stats_df,
    primary_group="Index name",          # <-- correct
    secondary_group="Laser wavelength",  # <-- correct
    x_order=[
        "min 0-4",
        "min 4-8",
        "min 8-12",
        "Constant",
    ],
    hue_order=[488, 505, "no light"],    # match your actual datatype
    y_lookup={
        "min 0-4": df_avg_duration.loc[
            df_avg_duration["Index name"] == "min 0-4",
            beh_duration_name
        ].max(),
        "min 4-8": df_avg_duration.loc[
            df_avg_duration["Index name"] == "min 4-8",
            beh_duration_name
        ].max(),
        "min 8-12": df_avg_duration.loc[
            df_avg_duration["Index name"] == "min 8-12",
            beh_duration_name
        ].max(),
        "Constant": df_avg_duration.loc[
            df_avg_duration["Index name"] == "Constant",
            beh_duration_name
        ].max(),
    },
    y_padding=0.10,
    line_height=0.08,
)
"""
# fig.update_xaxes(type="category")
fig.write_html(os.path.join(r"C:\Users\Itamar\Desktop","505",f"{beh_duration_name}_duration_plot.html"))


### IMMOBILIZED ###


datasets_488_immob_leifer = load_all_projects_in_folder(r'\\samba.lisc.univie.ac.at\scratch\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\488_leifer_only')
datasets_505_immob_leifer = load_all_projects_in_folder(r'\\samba.lisc.univie.ac.at\scratch\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\505_leifer_only')
datasets_505_immob_leifer.pop('2026-02-17_20-43_2026-02-17_11-51_immob_10per_bead_05umtet_250uW505_840uW561_worm8-2026-02-17')  # remove sleeping worm dataset

immob_488_rev_duration = calc_durations_dataframe(datasets_488_immob_leifer,use_manual_annotation=True)
immob_505_rev_duration = calc_durations_dataframe(datasets_505_immob_leifer,use_manual_annotation=True)
immob_488_rev_duration['BehaviorCodes.FWD'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
immob_488_rev_duration['BehaviorCodes.REV'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
immob_488_rev_duration['Laser wavelength'] = '488 immob'
immob_505_rev_duration['BehaviorCodes.FWD'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
immob_505_rev_duration['BehaviorCodes.REV'] /= no_light_datasets[list(no_light_datasets.keys())[0]].physical_unit_conversion.volumes_per_second
immob_505_rev_duration['Laser wavelength'] = '505 immob'
df_duration_merged = pd.concat([immob_488_rev_duration,immob_505_rev_duration])
df_avg_duration = (
    df_duration_merged
    .groupby(['dataset_name', 'Laser wavelength'])[beh_duration_name]
    .mean()
    .reset_index())

results_df, annotation_text = pairwise_ks_test_annotation(df_avg_duration,
                            beh_duration_name, group_column='Laser wavelength')


import os
import plotly.express as px
### plotting and stats
### REVERSAL - PULLED ###
beh_duration_name  = "BehaviorCodes.REV"
fig = px.box(df_duration_merged, color='Laser wavelength', y=beh_duration_name, points='all', x='Laser wavelength',
      color_discrete_map=plotly_paper_color_discrete_map())
fig.update_xaxes(type="category")
# 2: Axis titles + larger fonts
fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Reversal Duration (sec)" if beh_duration_name == "BehaviorCodes.REV" else "Forward Duration (sec)",
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24)
)
fig.update_layout(
    legend=dict(
        font=dict(size=22),          # legend entries
        title_font=dict(size=24)     # legend title (optional but recommended)
    )
)

fig.write_html(os.path.join(r"C:\Users\Itamar\Desktop","505",f"{beh_duration_name}_immob_duration_plot.html"))

### REVERSAL - PULLED ###

beh_duration_name  = "BehaviorCodes.REV"
fig = px.box(df_avg_duration, color='Laser wavelength', y=beh_duration_name, points='all', x='Laser wavelength',
      color_discrete_map=plotly_paper_color_discrete_map())
fig.update_xaxes(type="category")
# 2: Axis titles + larger fonts
fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Reversal Duration (sec)" if beh_duration_name == "BehaviorCodes.REV" else "Forward Duration (sec)",
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24)
)
fig.update_layout(
    legend=dict(
        font=dict(size=22),          # legend entries
        title_font=dict(size=24)     # legend title (optional but recommended)
    )
)

fig.write_html(os.path.join(r"C:\Users\Itamar\Desktop","505",f"{beh_duration_name}_immob_duration_plot_individual.html"))

### SWAP LIGHT - PULLED ###
datasets_488_immob_swap = load_all_projects_in_folder(r'\\samba.lisc.univie.ac.at\scratch\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\immobilized_505\488_505_488')
datasets_505_immob_swap = load_all_projects_in_folder(r'\\samba.lisc.univie.ac.at\scratch\neurobiology\zimmer\ItamarLev\WBFM\WBFM_projects\immobilized_505\505_488_505')

# an empty function to pass to analyze_laser_swap_datasets since we only care about the durations here and not the speed or reversals per minute
def dummy_calc_speed_dataframe(all_projects):
    return pd.DataFrame()

_, _, df_duration_immob_488 = analyze_laser_swap_datasets(datasets_488_immob_swap,
                                                                              (488,505,488),
                                                                              calc_durations_dataframe,
                                                                              dummy_calc_speed_dataframe,
                                                                              rev_dur_manual_annot=True)
_, _, df_duration_immob_505 = analyze_laser_swap_datasets(datasets_505_immob_swap,
                                                                              (505,488,505),
                                                                              calc_durations_dataframe,
                                                                              dummy_calc_speed_dataframe,
                                                                              rev_dur_manual_annot=False)
df_duration_merged = pd.concat([df_duration_immob_488,df_duration_immob_505])
df_duration_merged['Index name'] = df_duration_merged['position'].map({0: 'min 0-6', 1:'min 7-12', 2:'min 13-18'})


beh_duration_name  = "BehaviorCodes.REV"

df_avg_duration = (
    df_duration_merged
    .groupby(['position','dataset_name', 'Laser wavelength'])[beh_duration_name]
    .mean()
    .reset_index())
df_avg_duration['Index name'] = df_avg_duration['position'].map({0: 'min 0-6', 1:'min 7-12', 2:'min 13-18'})

import os
import plotly.express as px
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map

fig = px.box(df_duration_merged, color='Laser wavelength', y=beh_duration_name, points='all', x='Laser wavelength',
      color_discrete_map=plotly_paper_color_discrete_map())
fig.update_xaxes(type="category")
# 2: Axis titles + larger fonts
fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Reversal Duration (sec)" if beh_duration_name == "BehaviorCodes.REV" else "Forward Duration (sec)",
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24)
)
fig.update_layout(
    legend=dict(
        font=dict(size=22),          # legend entries
        title_font=dict(size=24)     # legend title (optional but recommended)
    )
)

fig.write_html(os.path.join(r"C:\Users\Itamar\Desktop","505",f"{beh_duration_name}_immob_swap_duration_plot_pulled.html"))

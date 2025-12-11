import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import melt_nested_dict
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


def apply_function_to_pages(video_fname,
                            func,
                            num_pages=None):
    """
    Applies 'func' to each xy plane of an ome tiff, looping over pages

    This loop goes over pages, and makes sense when the metadata is corrupted

    """
    dat = []

    with tifffile.TiffFile(video_fname, multifile=False) as tif:
        if num_pages is None:
            num_pages = len(tif.pages)
        for i, page in enumerate(tif.pages):

            if i >= num_pages:
                break

            if i % 100 == 0:
                print(f'Page {i}/{num_pages}')
            dat.append(func(page.asarray()))

    return np.array(dat)


##
## Behavior summary statistics
##

from collections import defaultdict


def calc_net_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()
    xy0 = df.loc[0, :][i_seg]
    xy1 = df.iloc[-1, :][i_seg]

    return np.linalg.norm(xy0 - xy1)


def calc_cumulative_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()[i_seg]
    dist = np.sqrt((df['X'] - df['X'].shift()) ** 2 + (df['Y'] - df['Y'].shift()) ** 2)
    line_integral = np.nansum(dist)

    return line_integral


def calc_displacement_dataframes(all_projects):
    all_displacements = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        all_displacements['net'][name] = calc_net_displacement(p)
        all_displacements['cumulative'][name] = calc_cumulative_displacement(p)
    df_displacement_gcamp = pd.DataFrame(all_displacements)

    return df_displacement_gcamp


def calc_speed_vector(p, speed_type):
    # Units: mm/2
    worm = p.worm_posture_class
    return worm.calc_behavior_from_alias(speed_type)


def calc_speed_dataframe(all_projects):
    # Note that the signed_speed_angular only has a well-defined sign across datasets if the ventral side is annotated
    speed_types = ['abs_stage_speed', 'middle_body_speed', 'signed_middle_body_speed',
                   'worm_speed_average_all_segments', 'signed_speed_angular']

    all_speeds = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        try:
            for speed_type in speed_types:
                all_speeds[speed_type][name] = calc_speed_vector(p, speed_type)
        except ValueError:
            continue
    df_speed = melt_nested_dict(all_speeds)

    return df_speed#, all_speeds


def calc_durations_dataframe(all_projects, states=None):
    if states is None:
        states = [BehaviorCodes.FWD, BehaviorCodes.REV]
    # Note that the signed_speed_angular does not have a well-defined sign across datasets

    all_durations = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        try:
            for state in states:
                ind_class = p.worm_posture_class.calc_triggered_average_indices(state=state, min_duration=0)
                all_durations[str(state)][name] = ind_class.all_state_durations(include_censored=False)[0]
        except ValueError:
            continue
    df_durations = melt_nested_dict(all_durations, all_same_lengths=False)

    return df_durations


def calc_onset_frequency_dataframe(all_projects, states=None):
    """
    Like calc_durations_dataframe, but calculates the number of onsets per minute

    This does not have the same interaction with censoring as calc_durations_dataframe, and is arguably more
    correct

    Parameters
    ----------
    all_projects

    Returns
    -------

    """
    if states is None:
        states = [BehaviorCodes.FWD, BehaviorCodes.REV]

    all_frequencies = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        try:
            for state in states:
                ind_class = p.worm_posture_class.calc_triggered_average_indices(state=state, min_duration=0)
                num_events = len(ind_class.idx_onsets)
                # Most are 8 minutes, but some may be cut shorter
                duration_in_minutes = p.num_frames / p.physical_unit_conversion.volumes_per_second / 60
                all_frequencies[str(state)][name] = num_events / duration_in_minutes
        except ValueError:
            continue
    df_durations = melt_nested_dict(all_frequencies, all_same_lengths=False)

    return df_durations


def calc_turn_amplitude_dataframe(all_projects):
    final_ventral_dict = {}
    final_dorsal_dict = {}

    for name, p in tqdm(all_projects.items()):

        worm = p.worm_posture_class
        y_curvature = worm.calc_behavior_from_alias('head_signed_curvature')

        ventral_peaks, ventral_peak_times, _ = worm.get_peaks_post_reversal(y_curvature, num_points_after_reversal=20)
        dorsal_peaks, dorsal_peak_times, all_rev_ends = worm.get_peaks_post_reversal(-y_curvature, num_points_after_reversal=20)

        # Keep the positive peak if it is a ventral turn at that time, or negative peak if it is dorsal
        # ... actually I don't think Ulises' annotations are that frame-accurate, so I will take whichever peak is closer to the end of the reversal, i.e. the first body bend
        ventral_to_keep = []
        dorsal_to_keep = []
        for vp, vt, dp, dt, end in zip(ventral_peaks, ventral_peak_times, dorsal_peaks, dorsal_peak_times, all_rev_ends):
            # Alternate: take the one with the higher amplitude
            # if np.abs(vp) > np.abs(dp):
            #     ventral_to_keep.append(vp)
            # else:
            #     dorsal_to_keep.append(dp)
            
            # Skip if the event is exactly at the end
    #         vt_at_edge = vt == end
    #         vt_early = vt < dt
    #         dt_at_edge = dt == end
            
            if vt < dt and vt > end:
                ventral_to_keep.append(vp)
            elif dt > end:
                dorsal_to_keep.append(-dp)
            else:
                pass
        final_ventral_dict[name] = ventral_to_keep
        final_dorsal_dict[name] = dorsal_to_keep

        # For now, ignore the dataset they came from
    df_ventral = pd.DataFrame(np.concatenate(list(final_ventral_dict.values())))
    df_ventral['Turn Direction'] = 'Ventral'
    df_dorsal = pd.DataFrame(np.concatenate(list(final_dorsal_dict.values())))
    df_dorsal['Turn Direction'] = 'Dorsal'

    df_turns = pd.concat([df_ventral, df_dorsal])
    df_turns.columns = ['Amplitude', 'Turn Direction']
    df_turns['Amplitude'] *= 1000  # Change to 1/mm

    return df_turns

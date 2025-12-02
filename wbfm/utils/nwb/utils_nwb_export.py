import logging
import os
import re
from pathlib import Path
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet
from skimage.segmentation import watershed

from dask import compute, delayed
import dask.array as da
import numpy as np
import scipy
# from hdmf_zarr import NWBZarrIO
from matplotlib import pyplot as plt
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from hdmf.common import DynamicTable, VectorData
from pynwb.behavior import BehavioralTimeSeries, SpatialSeries, Position
from pynwb.image import ImageSeries
from pynwb.ophys import ImageSegmentation, PlaneSegmentation, RoiResponseSeries, Fluorescence, DfOverF
from hdmf.data_utils import GenericDataChunkIterator
from dateutil import tz
import pandas as pd
from datetime import datetime
from hdmf.backends.hdf5.h5_utils import H5DataIO
# ndx_mulitchannel_volume is the novel NWB extension for multichannel optophysiology in C. elegans
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, \
    MultiChannelVolume, MultiChannelVolumeSeries, SegmentationLabels
from skimage.measure import regionprops
from tifffile import tifffile
from tqdm.auto import tqdm
from wbfm.utils.external.utils_pandas import convert_binary_columns_to_one_hot, fill_missing_indices_with_nan

import itertools
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.external.utils_neuron_names import int2name_neuron
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.general.utils_filenames import get_sequential_filename


def create_vol_seg_centers(name, description, ImagingVolume, positions,
                           labels=None, reference_images=None) -> PlaneSegmentation:
    """
    Use this function to create volume segmentation where each ROI is coordinates
    for a single neuron center in XYZ space.

    Positions should be a 2d array of size (N,3) where N is the number of neurons and
    3 refers to the XYZ coordinates of the neuron in that order.

    Labels should be an array of cellIDs in the same order as the neuron positions.

    From: https://github.com/focolab/NWBelegans/blob/main/NWB_convert.py#L817
    """

    vs = PlaneSegmentation(
        name=name,
        description=description,
        imaging_plane=ImagingVolume,
        reference_images=reference_images
    )

    for i in range(positions.shape[0]):
        voxel_mask = []
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]

        voxel_mask.append([np.uint(x), np.uint(y), np.uint(z), 1])  # add weight of 1 to each ROI

        vs.add_roi(voxel_mask=voxel_mask)

    if labels is None:
        labels = [''] * positions.shape[0]
    else:
        vs.add_column(
            name='ID_labels',
            description='ROI ID labels',
            data=labels.astype(str),
        )
    return vs


def nwb_using_project_data(project_data: ProjectData, include_image_data=True, output_folder=None,
                           enforce_nonoverlapping_behaviors=False, DEBUG=False):
    """
    Convert a ProjectData class to an NWB h5 file, optionally including all raw image data.

    Following: https://github.com/focolab/NWB/blob/main/NWB_tutorial.ipynb

    Parameters
    ----------
    project_data

    Returns
    -------

    """
    try:
        cfg_nwb = project_data.project_config.get_nwb_config()
    except PermissionError:
        logging.warning(f"You do not have permissions for project {project_data.shortened_name} to save NWB files.")
        cfg_nwb = None

    if DEBUG:
        logging.warning("DEBUG mode; will not save final output (this is a dry run)")
        output_folder = None
    elif output_folder is None:
        if cfg_nwb is not None:
            # Save within the project_data folder
            output_folder = cfg_nwb.absolute_subfolder
        else:
            raise PermissionError(f"Either project permissions or output folder is required to save NWB files.")

    output_fname = os.path.join(output_folder, project_data.shortened_name)
    if not include_image_data:
        output_fname = f'{output_fname}_no_image_data'
    output_fname = f'{output_fname}.nwb'

    # Unpack variables from project_data
    # Everything in the zimmer lab is produced with a time stamp saved in the filename of the raw data folder
    # Like this: 2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH
    # So, we will try to parse it:
    try:
        raw_dir = project_data.raw_data_dir
        year_month_day, hour_minute, strain, subject_id, _ = raw_dir.split('_')
        year, month, day = year_month_day.split('-')
        hour, minute = hour_minute.split('-')
        session_start_time = datetime(int(year), int(month), int(day), int(hour), int(minute), 0, tzinfo=tz.gettz("Europe/Vienna"))
    except:
        session_start_time = datetime(2022, 11, 27, 21, 41, 10, tzinfo=tz.gettz("Europe/Vienna"))
        # Convert the datetime to a string that can be used as a default subject_id
        subject_id = session_start_time.strftime("%Y%m%d-%H-%M-%S")
        strain = None

    # Unpack metadata (no matter what the stage of the project is)
    print("Calculating metadata...")
    raw_data_cfg = project_data.project_config.get_raw_data_config()
    if strain is None and raw_data_cfg.has_valid_self_path:
        strain = raw_data_cfg.config.get('strain', 'unknown')
    physical_units_class = project_data.physical_unit_conversion

    flag = check_all_needed_data_for_step(project_data.project_config, 5, raise_error=False,
                                          verbose=0)
    if not flag:
        logging.info("Project data is incomplete, will save only the raw data")
        nwbfile, fname = nwb_only_raw_data(project_data, session_start_time, subject_id, strain,
                                           physical_units_class, output_folder)
        return nwbfile, fname

    # Unpack traces and locations
    print("Calculating traces...")
    gce_quant_red = project_data.red_traces.swaplevel(i=0, j=1, axis=1).copy()
    gce_quant_green = project_data.green_traces.swaplevel(i=0, j=1, axis=1).copy()
    gce_quant_ratio = gce_quant_red.copy()

    gce_quant_dict = {'red': gce_quant_red, 'green': gce_quant_green, 'ratio': gce_quant_ratio}
    # Rename to use manual ids, if they exist
    id_mapping = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0,
                                                               only_include_confident_labels=True)
    for key in gce_quant_dict.keys():
        gce_quant_dict[key].rename(columns=id_mapping, inplace=True, level=1)
    # Store just the background subtracting red or green traces, because we don't want to store the object volume
    # Use the exact options in the paper
    trace_opt = dict(interpolate_nan=True)
    df_traces_red = project_data.calc_paper_traces(channel_mode='red', **trace_opt)
    # Here we have a subset of columns, so we need to keep only the proper set
    kept_columns = df_traces_red.columns
    gce_quant_red.loc[:, ('intensity_image', kept_columns)] = df_traces_red.values
    df_traces_green = project_data.calc_paper_traces(channel_mode='green', **trace_opt)
    gce_quant_green.loc[:, ('intensity_image', kept_columns)] = df_traces_green.values

    df_traces_ratio = project_data.calc_paper_traces(channel_mode='dr_over_r_50', **trace_opt)
    gce_quant_ratio.loc[:, ('intensity_image', kept_columns)] = df_traces_ratio.values

    # Then drop any other columns that are not in the kept columns
    gce_quant_red = gce_quant_red.loc[:, (slice(None), kept_columns)]
    gce_quant_green = gce_quant_green.loc[:, (slice(None), kept_columns)]
    gce_quant_ratio = gce_quant_ratio.loc[:, (slice(None), kept_columns)]
    gce_quant_dict = {'red': gce_quant_red, 'green': gce_quant_green, 'ratio': gce_quant_ratio}

    # Unpack videos
    if include_image_data:
        calcium_video_dict = {'red': project_data.red_data, 'green': project_data.green_data}
        segmentation_video = project_data.segmentation
    else:
        calcium_video_dict = {'red': None, 'green': None}
        segmentation_video = None

    # Unpack behavior video and time seriesdata
    video_class = project_data.worm_posture_class
    if video_class.has_full_kymograph:
        if include_image_data:
            behavior_video = video_class.raw_behavior_video
        else:
            behavior_video = None
        behavior_time_series_dict = {}
        behavior_time_series_names = ['angular_velocity', 'head_curvature', 'body_curvature', 'reversal_events',
                                      'velocity',
                                      'ventral_only_body_curvature', 'dorsal_only_body_curvature',
                                      'ventral_only_head_curvature', 'dorsal_only_head_curvature']
        behavior_time_series_dict['continuous_behaviors'] = video_class.calc_behavior_from_alias(behavior_time_series_names,
                                                                                                 reset_index=False)
        hilbert_outputs = ['hilbert_phase', 'hilbert_amplitude', 'hilbert_frequency', 'hilbert_carrier']
        for name in hilbert_outputs:
            behavior_time_series_dict[name] = video_class.calc_behavior_from_alias(name, reset_index=False)
        # behavior_time_series_dict = video_class.calc_behavior_from_alias(behavior_time_series_names)
        # Also add some more basic time series data
        behavior_time_series_dict['kymograph'] = video_class.curvature(fluorescence_fps=True, reset_index=False)
        behavior_time_series_dict['stage_position'] = video_class.stage_position(fluorescence_fps=True, reset_index=False)
        behavior_time_series_dict['eigenworms'] = video_class.eigenworms(fluorescence_fps=True, reset_index=False)
        # Also add a dataframe of the discrete behaviors
        from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
        discrete_time_series_names = BehaviorCodes.default_state_hierarchy(use_strings=True,
                                                                           include_self_collision=True)
        df_discrete = video_class.calc_behavior_from_alias(discrete_time_series_names, include_slowing=True,
                                                           reset_index=False)
        if enforce_nonoverlapping_behaviors:
            idx = behavior_time_series_dict['continuous_behaviors']['velocity'].index
            df_discrete = convert_binary_columns_to_one_hot(pd.DataFrame(df_discrete, index=idx),
                                                            discrete_time_series_names)
        behavior_time_series_dict['discrete_states'] = df_discrete

    else:
        print("No behavior data found")
        behavior_video, behavior_time_series_dict = None, None

    # Unpack centroids and segmentation-to-final-id mapping
    df_tracking = project_data.final_tracks

    nwb_file, fname = nwb_with_traces_from_components(calcium_video_dict, segmentation_video, gce_quant_dict,
                                                      session_start_time, subject_id, strain, physical_units_class,
                                                      behavior_video, behavior_time_series_dict, df_tracking,
                                                      output_fname, include_image_data)
    # Update in the project config
    if cfg_nwb is not None:
        cfg_nwb.config['nwb_filename'] = fname
        cfg_nwb.update_self_on_disk()

    return nwb_file, fname


def nwb_from_matlab_tracker(matlab_fname, output_folder=None):
    """
    Convert a matlab tracker file to an NWB h5 file.

    Parameters
    ----------
    matlab_fname

    Returns
    -------

    """
    if output_folder is None:
        logging.warning("No output folder specified, will not save final output (this is a dry run)")

    import mat73
    mat = mat73.loadmat(matlab_fname)

    # Unpack variables from matlab file
    session_start_time = mat['added']['dateAdded'][0]
    # This is a string like '26-Jan-2019 10:35:22'; convert to datetime
    session_start_time = datetime.strptime(session_start_time, '%d-%b-%Y %H:%M:%S')
    # Convert the datetime to a string that can be used as a subject_id
    subject_id = session_start_time.strftime("%Y%m%d-%H-%M-%S")

    # Define a regular expression pattern to match "ZIM" followed by numbers
    pattern = r'ZIM(\d+)'
    match = re.search(pattern, matlab_fname)
    if match:
        # Extract the matched substring
        strain = match.group(0)
    else:
        print(f"Pattern 'ZIM' not found in the input string.")
        raise NotImplementedError

    # Unpack traces
    id_names = mat["ID1"]
    raw_colnames = [f"neuron_{i:03d}" for i in range(len(id_names))]
    # colnames = [dummy if ID is None else ID for dummy, ID in zip(raw_colnames, id_names)]
    colnames = raw_colnames
    gce_quant = pd.DataFrame(mat["deltaFOverF"], columns=colnames)
    # Add a new level to the columns to specify the type of data (here, just 'intensity_image')
    gce_quant.columns = pd.MultiIndex.from_product([['intensity_image'], gce_quant.columns])
    gce_dict = gce_quant.to_dict()
    # Also needs to have the additional columns that my freely moving projects do:
    #   ['x', 'y', 'z', 'intensity_image', 'label', 'index']
    # But actually build a dictionary and then convert to dataframe to avoid fragmentation warnings
    n = len(gce_quant)
    t_vec = list(gce_quant.index)
    for name in raw_colnames:
        # The label column has to be correct, i.e. each neuron should have a label such as 'neuron_001' -> 1
        idx = int(name.split('_')[1])
        gce_dict[('label', name)] = {t: idx for t in t_vec}
        # TODO: proper x, y, z, index
        gce_dict[('x', name)] = {t: 1 for t in t_vec}
        gce_dict[('y', name)] = {t: 1 for t in t_vec}
        gce_dict[('z', name)] = {t: 1 for t in t_vec}
        gce_dict[('index', name)] = {t: t for t in t_vec}
    gce_quant = pd.DataFrame(gce_dict)

    # Unpack video
    video_dict = {'red': None}

    # TODO: FIX
    raise NotImplementedError
    # nwb_file = nwb_with_traces_from_components(video_dict, gce_quant, session_start_time, subject_id, strain,
    #                                            physical_units_class=None, output_folder=output_folder)
    # return nwb_file


def nwb_with_traces_from_components(calcium_video_dict, segmentation_video, gce_quant_dict, session_start_time, subject_id, strain,
                                    physical_units_class, behavior_video, behavior_time_series_dict, df_tracking,
                                    output_fname, include_image_data):
    # Initialize and populate the NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain, subject_id)

    # Get the overall metadata of the imaging device
    device = _zimmer_microscope_device(nwbfile)
    if physical_units_class is None:
        # STUB FOR IMMOBILIZED (which has very messy metadata
        logging.warning("No physical units class provided, using default values")
        grid_spacing = [0.4, 0.4, 0.2]
        rate = 2.0  # Volumes per second
    else:
        grid_spacing = physical_units_class.grid_spacing
        rate = physical_units_class.volumes_per_second
    CalcImagingVolume, order_optical_channels = build_optical_channel_objects(device, grid_spacing, list(calcium_video_dict.keys()))
    nwbfile.add_imaging_plane(CalcImagingVolume)

    # Add the traces and tracking data
    nwbfile = convert_traces_and_tracking_to_nwb(
        nwbfile, segmentation_video, gce_quant_dict, CalcImagingVolume, physical_units_class, device=device
    )
    # Finish: metadata
    nwbfile.processing['CalciumActivity'].add(order_optical_channels)

    # Add the video data (optional)
    if include_image_data:
        convert_calcium_videos_to_nwb(nwbfile, calcium_video_dict, device, CalcImagingVolume, rate)
        CalciumSegSeries = convert_segmentation_video_to_nwb(CalcImagingVolume, device, segmentation_video, physical_units_class=physical_units_class)
        nwbfile.processing['CalciumActivity'].add(CalciumSegSeries)

    # Add the behavior video and time series, if they exist
    if behavior_video is not None and include_image_data:
        nwbfile = convert_behavior_video_to_nwb(nwbfile, behavior_video, fps=physical_units_class.frames_per_second)
    if behavior_time_series_dict is not None:
        nwbfile = convert_behavior_series_to_nwb(nwbfile, behavior_time_series_dict)
    else:
        print("No behavior time series data found, skipping...")
    
    # Add centroids (output of the tracking), if it exists
    if df_tracking is not None:
        position, dt = df_to_nwb_tracking(df_tracking)
        
        if position is not None:
            nwbfile.processing['CalciumActivity'].add(position)
        else:
            raise ValueError(f"Tracking dataframe was found (shape: {df_tracking.shape}), but could not convert it into centroids")

        if include_image_data:
            # This table is not useful (and perhaps misleading) without the corresponding segmentation
            nwbfile.processing['CalciumActivity'].add(dt)

    if output_fname:
        logging.info(f"Saving NWB file to {output_fname}")
        fname = get_sequential_filename(output_fname)
        with NWBHDF5IO(fname, mode='w') as io:
        # with NWBZarrIO(path=fname, mode="w") as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")

    return nwbfile, fname


def nwb_only_raw_data(project_data, session_start_time, subject_id, strain, physical_units_class,
                      output_folder=None):
    """
    Convert a ProjectData class to an NWB h5 file, but only include the raw data.

    Parameters
    ----------
    project_data

    Returns
    -------

    """
    # Unpack videos from project_data
    p = project_data.project_config.get_preprocessing_class()
    raw_video_dict = {'red': p.open_raw_data_as_4d_dask(red_not_green=True),
                      'green': p.open_raw_data_as_4d_dask(red_not_green=False)}

    # NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain, subject_id)

    device = _zimmer_microscope_device(nwbfile)
    rate = physical_units_class.volumes_per_second
    convert_calcium_videos_to_nwb(nwbfile, raw_video_dict, device, rate, raw_videos=True)

    # Create a stub processing module, because I'm not sure where else the CalcOptChanRefs should go
    calcium_imaging_module = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data (STUB)'
    )
    calcium_imaging_module.add(CalcOptChanRefs)

    # Save the NWB file
    fname = None
    if output_folder:
        fname = os.path.join(output_folder, subject_id + '.nwb')
        logging.info(f"Saving NWB file to {fname}")
        with NWBHDF5IO(fname, mode='w') as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")

    return nwbfile, fname


def initialize_nwb_file(session_start_time, strain, subject_id):
    print("Initializing nwb file...")
    nwbfile = NWBFile(
        session_description='Add a description for the experiment/session. Can be just long form text',
        # Can use any identity marker that is specific to an individual trial. We use date-time to specify trials
        identifier=session_start_time.strftime("%Y%m%d-%H-%M-%S"),
        # Specify date and time of trial. Datetime entries are in order Year, Month, Day, Hour, Minute, Second. Not all entries are necessary
        session_start_time=session_start_time,
        lab='Zimmer lab',
        institution='University of Vienna',
        related_publications=''
    )
    nwbfile.subject = CElegansSubject(
        # This is the same as the NWBFile identifier for us, but does not have to be. It should just identify the subject for this trial uniquely.
        subject_id=subject_id,
        # Age is optional but should be specified in ISO 8601 duration format similarly to what is shown here for growth_stage_time
        # age = pd.Timedelta(hours=2, minutes=30).isoformat(),
        # Date of birth is a required field but if you do not know or if it's not relevant, you can just use the current date or the date of the experiment
        date_of_birth=session_start_time,
        # Specify growth stage of worm - should be one of two-fold, three-fold, L1-L4, YA, OA, dauer, post-dauer L4, post-dauer YA, post-dauer OA
        growth_stage='YA',
        # Optional: specify time in current growth stage
        # growth_stage_time=pd.Timedelta(hours=2, minutes=30).isoformat(),
        # Specify temperature at which animal was cultivated
        cultivation_temp=20.,
        description="free form text description, can include whatever you want here",
        # Currently using the ontobee species link until NWB adds support for C. elegans
        species="http://purl.obolibrary.org/obo/NCBITaxon_6239",
        # Currently just using O for other until support added for other gender specifications
        sex="O",
        strain=strain
    )
    return nwbfile


def laser_properties(channel_str='red'):
    if channel_str == 'red':
        # RED
        emission_lambda = 617.
        emission_delta = 73.
        excitation_lambda = 561.
        laser_tuple = ("mScarlet",)
    elif channel_str == 'green':
        # GREEN
        emission_lambda = 525.
        emission_delta = 50.
        excitation_lambda = 488.
        laser_tuple = ("GFP-GCaMP",)
    else:
        raise ValueError(f"Unknown channel string: {channel_str}")
    laser_description = f'GFP/GCaMP channel, f{excitation_lambda} excitation, {emission_lambda}/{emission_delta}m emission'
    laser_tuple = laser_tuple + (
    f"Chroma ET {emission_lambda}/{emission_delta}", f"{excitation_lambda}-{emission_lambda}-{emission_delta}m")
    return emission_lambda, emission_delta, excitation_lambda, laser_description, laser_tuple


def convert_calcium_videos_to_nwb(nwbfile, video_dict: dict, device, CalcImagingVolume, rate, raw_videos=False):
    print("Initializing imaging channels...")
    # Convert a dictionary of video data into a single multi-channel numpy array
    # With proper metadata
    # Stack the dask arrays lazily along a new channel axis
    video_list = list(video_dict.values())
    video_data = da.stack(video_list, axis=-1)
    # Reshape to be TXYZC from TZXYC
    video_data = video_data.transpose(0, 2, 3, 1, 4)
    chunk_shape = list(video_data.shape[1:-1])  # Remove time point and channel
    chunk_shape.append(1)  # Add the channel
    chunk_shape.insert(0, 1)  # Add the time point

    # The DataChunkIterator wraps the data generator function and will stitch together the chunks as it iteratively reads over the full file
    if video_data is not None:
        data = CustomDataChunkIterator(
            array=video_data,
            # this will be the max shape of the final image. Can leave blank or set as the size of your full data if you know that ahead of time
            # maxshape=None,
            # buffer_size=10,
            chunk_shape=tuple(chunk_shape)
        )
        wrapped_data = H5DataIO(data=data, compression="gzip", compression_opts=4)
    else:
        wrapped_data = H5DataIO(np.zeros((1, 1, 1, 1), dtype=np.uint8), compression='gzip')

    calcium_image_series = MultiChannelVolumeSeries(
        name="CalciumImageSeries" if not raw_videos else "RawCalciumImageSeries",
        description="Raw GCaMP series images",
        comments="GFP-GCaMP channel is the GCaMP signal, mScarlet is the reference signal",
        data=wrapped_data,
        device=device,
        unit="Voxel gray counts",
        # scan_line_rate=None,  # TODO: what is this?
        resolution=1.,
        #smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=rate,
        imaging_volume=CalcImagingVolume,
        # dimension=None, #  Gives a warning; what should this be?
    )

    nwbfile.add_acquisition(calcium_image_series)


def build_optical_channel_objects(device, grid_spacing, video_keys: list):
    # The loop below takes the list of channels and converts it into a list of OpticalChannelPlus objects which hold the metadata
    # for the optical channels used in the experiment
    CalcChannels = []
    for key in video_keys:
        laser_tuple = laser_properties(key)[-1]
        CalcChannels.append(laser_tuple)
    CalcOptChannels = []
    CalcOptChanRefData = []
    for fluor, des, wave in CalcChannels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        OptChan = OpticalChannelPlus(
            name=fluor,
            description=des,
            excitation_lambda=excite,
            excitation_range=[excite - 1.5, excite + 1.5],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid
        )
        CalcOptChannels.append(OptChan)
        CalcOptChanRefData.append(wave)
    # This object just contains references to the order of channels because OptChannels does not preserve ordering
    order_optical_channels = OpticalChannelReferences(
        name='order_optical_channels',
        channels=CalcOptChanRefData
    )

    CalcImagingVolume = ImagingVolume(
        name='CalciumImVol',
        description='Imaging plane used to acquire calcium imaging data',
        optical_channel_plus=CalcOptChannels,
        order_optical_channels=order_optical_channels,
        device=device,
        location='Worm head',
        grid_spacing=grid_spacing,
        grid_spacing_unit='um',
        reference_frame='Worm head'
    )
    return CalcImagingVolume, order_optical_channels


def _iter_volumes(video_data):
    """
    Expects a 5d input (TXYZC), yields a 4d image: XYZC

    In other words, no transposing is done here

    Note: only actually interates over the first dimension; data could be fewer (e.g. no channel)

    """
    if video_data is None:
        return None

    # We iterate through all of the timepoints and yield each timepoint back to the DataChunkIterator
    for i in range(video_data.shape[0]):
        # Make sure array ends up as the correct dtype coming out of this function (the dtype that your data was collected as)
        yield video_data[i]
    return


def convert_traces_and_tracking_to_nwb(nwbfile, segmentation_video, gce_quant_dict, CalcImagingVolume,
                                       physical_units_class, device, DEBUG=False):
    print("Converting traces and tracking to nwb format...")
    gce_quant_red = convert_tracking_dataframe_to_nwb_format(gce_quant_dict['red'], DEBUG)
    gce_quant_green = convert_tracking_dataframe_to_nwb_format(gce_quant_dict['green'], DEBUG)
    gce_quant_ratio = convert_tracking_dataframe_to_nwb_format(gce_quant_dict['ratio'], DEBUG)

    rate = physical_units_class.volumes_per_second

    # Extract the blobs (with time series) from red and green
    blobquant_red, blobquant_green, blobquant_ratio = None, None, None
    for idx in tqdm(gce_quant_red['blob_ix'].unique(), leave=False, disable=not DEBUG, desc="Extracting segmentation ids..."):
        blob_red = gce_quant_red[gce_quant_red['blob_ix'] == idx]
        blobquant_red = _add_blob(blob_red, blobquant_red)

        blob_green = gce_quant_green[gce_quant_green['blob_ix'] == idx]
        blobquant_green = _add_blob(blob_green, blobquant_green)

        blob_ratio = gce_quant_ratio[gce_quant_ratio['blob_ix'] == idx]
        blobquant_ratio = _add_blob(blob_ratio, blobquant_ratio)

    print("Extracting segmentation coordinates...")
    volsegs = []
    for t in tqdm(range(blobquant_red.shape[1]), leave=False, disable=not DEBUG):
        blobs = np.squeeze(blobquant_red[:, t, 0:3])
        IDs = np.squeeze(blobquant_red[:, t, 4])
        labels = IDs.astype(str)
        labels = np.where(labels != 'nan', labels, '')

        vsname = 'Seg_tpoint_' + str(t)
        description = 'Neuron segmentation for time point ' + str(t) + ' in calcium image series'
        volseg = create_vol_seg_centers(vsname, description, CalcImagingVolume, blobs, labels=labels)

        volsegs.append(volseg)


    CalcImSegCoords = ImageSegmentation(
        name='CalciumSeriesSegmentationCoords',
        plane_segmentations=volsegs,
        # reference_images=one_p_series,  # optional
    )

    # Do not repeat ids for all time points, just save t=0
    calc_IDs = np.squeeze(blobquant_red[:, 0, 4])
    calc_labels = calc_IDs.astype(str)
    calc_labels = np.where(['neuron' not in n for n in calc_labels], calc_labels, '')
    Calclabels = SegmentationLabels(
        name='NeuronIDs',
        labels=calc_labels,
        description='Calcium segmentation labels with xyz coordinates',
        ImageSegmentation=CalcImSegCoords,
        # MCVSeriesSegmentation=CalcImSeg,
    )

    rt_region = volsegs[0].create_roi_table_region(
        description='Segmented neurons associated with calcium image series. This rt_region uses the location of the neurons at the first time point',
        region=list(np.arange(blobquant_red.shape[0]))
    )

    # Take only gce quantification column and transpose so time is in the first dimension
    gce_data_red = np.transpose(blobquant_red[:, :, 3]).astype(float)
    gce_data_green = np.transpose(blobquant_green[:, :, 3]).astype(float)
    gce_data_ratio = np.transpose(blobquant_ratio[:, :, 3]).astype(float)

    # Traces: Red (reference)
    RefRoiResponse = RoiResponseSeries(
        #See https://pynwb.readthedocs.io/en/stable/pynwb.ophys.html#pynwb.ophys.RoiResponseSeries for additional key word argument options
        name='ReferenceCalciumImResponseSeries',
        description='Fluorescence for reference channel in calcium imaging',
        data=gce_data_red,  #first dimension should represent time and second dimension should represent ROIs
        rois=rt_region,
        unit='integrated image intensity',  #the unit of measurement for the data input here
        rate=rate
    )

    RefFluor = Fluorescence(
        name='ReferenceFluorescence',
        roi_response_series=RefRoiResponse
    )

    # If you have raw fluorescence values rather than DFoF use the Fluorescence object instead of the DfOverF object to save your RoiResponseSeries
    SignalRoiResponse = RoiResponseSeries(
        # See https://pynwb.readthedocs.io/en/stable/pynwb.ophys.html#pynwb.ophys.RoiResponseSeries for additional key word argument options
        name='SignalCalciumImResponseSeries',
        description='Green fluorescence activity for calcium imaging data',
        data=gce_data_green,
        rois=rt_region,
        unit='integrated image intensity',
        rate=rate,
    )

    SignalFluor = Fluorescence(
        name='SignalFluorescence',
        roi_response_series=SignalRoiResponse
    )

    # Final "ratio" values (dr/r50)
    RatioRoiResponse = RoiResponseSeries(
        # See https://pynwb.readthedocs.io/en/stable/pynwb.ophys.html#pynwb.ophys.RoiResponseSeries for additional key word argument options
        name='SignalCalciumImResponseSeries',
        description='dR/R50 fluorescence activity for calcium imaging data',
        data=gce_data_ratio,
        rois=rt_region,
        unit='integrated image intensity',
        rate=rate,
    )

    RatioFluor = DfOverF(  # Change to Fluorescence if using raw fluorescence
        name='SignalDFoF',
        roi_response_series=RatioRoiResponse
    )

    # Add data under the processed module
    calcium_imaging_module = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )

    # Finish: segmentation
    calcium_imaging_module.add(Calclabels)
    calcium_imaging_module.add(CalcImSegCoords)
    # Finish: signal time series
    calcium_imaging_module.add(SignalFluor)
    calcium_imaging_module.add(RefFluor)
    calcium_imaging_module.add(RatioFluor)

    return nwbfile


def convert_segmentation_video_to_nwb(CalcImagingVolume, device, segmentation_video, physical_units_class):
    rate = physical_units_class.volumes_per_second
    # Convert segmentation video from TZXY to TXYZ
    segmentation_video = np.transpose(segmentation_video, [0, 2, 3, 1])
    chunk_shape = list(segmentation_video.shape[1:])  # One time point
    chunk_shape.insert(0, 1)  # Add the time point
    # Build a generator (like the raw data) but for the segmentation data
    data = CustomDataChunkIterator(
        array=segmentation_video,
        # this will be the max shape of the final image. Can leave blank or set as the size of your full data if you know that ahead of time
        # maxshape=None,
        # buffer_size=10,
        chunk_shape=tuple(chunk_shape)
    )
    wrapped_data = H5DataIO(data=data, compression="gzip", compression_opts=4)
    CalciumSegSeries = MultiChannelVolumeSeries(
        name="CalciumSeriesSegmentation",
        description="Series of indexed masks associated with calcium segmentation",
        comments="Include here whether ROIs are tracked across frames or any other comments",
        data=wrapped_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        # dimension=None, #  Gives a warning; what should this be?,
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    )
    return CalciumSegSeries


def convert_behavior_video_to_nwb(nwbfile, behavior_video, fps):
    print("Converting behavior to nwb format...")
    # Behavior is already TXY
    chunk_shape = list(behavior_video.shape)

    # Build a generator (like the raw data) but for the behavior data
    data = CustomDataChunkIterator(
        array=behavior_video,
        chunk_shape=tuple(chunk_shape)
    )
    wrapped_data = H5DataIO(data=data, compression="gzip", compression_opts=4)

    behavior_video_series = ImageSeries(
        name="BrightFieldNIR",
        description="Behavioral image in near-infrared light",
        data=wrapped_data,
        unit="seconds",
        rate=fps
    )

    # Create new processing module for the behavior video
    behavior_module = nwbfile.create_processing_module(name="BF_NIR",
                                                       description="Behavioral image in near-infrared light")
    behavior_module.add(behavior_video_series)

    return nwbfile


def convert_behavior_series_to_nwb(nwbfile, behavior_time_series_dict):
    print("Converting behavior time series to nwb format...")
    behavior_module = nwbfile.create_processing_module(name="Behavior",
                                                       description="Behavioral time series")

    for name, time_series in behavior_time_series_dict.items():
        unit = 'seconds'
        if isinstance(time_series, np.ndarray):
            raise NotImplementedError
            # data = time_series
            # timestamps = np.arange(len(data))
            # unit = 'frames'
        elif isinstance(time_series, dict):
            keys = list(time_series.keys())
            timestamps = time_series[keys[0]].index.values
            _time_series_columns = time_series
        else:
            # Assume pandas dataframe
            # data = time_series.values
            timestamps = time_series.index.values
            # each column should be stored as a sub time series
            _time_series_columns = time_series.to_dict(orient='list')
        time_series_dict = {}
        for colname, coldata in _time_series_columns.items():
            coldata = coldata.values if isinstance(coldata, pd.Series) else coldata
            # Convert column name to string, but be careful if it is an integer
            # Specifically if it is <10, it should be padded with zeros to maintain sorting
            if isinstance(colname, int):
                colname = f"{colname:02d}"
            elif isinstance(colname, str):
                pass
            else:
                colname = str(colname)
            time_series_dict[colname] = TimeSeries(name=colname, data=coldata, timestamps=timestamps, unit=unit)
        # This nested requires nested indexing in the final object...
        # _time_series_obj = TimeSeries(name=name, data=data, timestamps=timestamps, unit=unit)
        behavior_module.add(BehavioralTimeSeries(name=name, time_series=time_series_dict))

    return nwbfile


def _add_blob(blob, blobquant):
    blobarr = np.asarray(blob[['X', 'Y', 'Z', 'gce_quant', 'ID']])
    blobarr = blobarr[np.newaxis, :, :]
    if blobquant is None:
        blobquant = blobarr
    else:
        blobquant = np.vstack((blobquant, blobarr))
    return blobquant


def convert_tracking_dataframe_to_nwb_format(gce_quant_raw, DEBUG=False):
    """
    Converts my tracking dataframe to the format that the NWB tutorial expects, i.e. the Kato lab standard

    Parameters
    ----------
    gce_quant
    DEBUG

    Returns
    -------

    """
    gce_quant = gce_quant_raw.copy()
    # Copy the label column to be blob_ix, but need to manually create the multiindex because it is multiple columns
    new_columns = pd.MultiIndex.from_tuples([('blob_ix', c[1]) for c in gce_quant_raw[['label']].columns])
    gce_quant[new_columns] = gce_quant['label'].copy()
    gce_quant.loc[:, ('blob_ix', slice(None))] = gce_quant.loc[:, ('blob_ix', slice(None))].fillna(
        method='bfill').fillna(method='ffill')
    # Expects a long single-level dataframe
    gce_quant = gce_quant.stack(level=1, dropna=False).reset_index(level=1, drop=True)  # .dropna(how='all')
    gce_quant.reset_index(inplace=True)
    # Replace NaN with 0's, because it has to be int in the end
    gce_quant = gce_quant.fillna(0)
    # Rename columns to be the format of this file
    gce_quant = gce_quant.rename(
        columns={'x': 'X', 'y': 'Y', 'z': 'Z', 'intensity_image': 'gce_quant', 'label': 'ID', 'index': 'T',
                 'blob_ix': 'blob_ix'})
    if DEBUG:
        print(len(gce_quant['blob_ix'].unique()))  # Count the number of unique blobs in this file
        print(len(gce_quant['T'].unique()))  # Count the number of unique time points in this file
    gce_quant = gce_quant[['X', 'Y', 'Z', 'gce_quant', 'ID', 'T', 'blob_ix']]  # Reorder columns to order we want

    # Make sure the ID column the string, corresponding to the initial column name
    # First build mapping from label to column name, using the mode (all labels should be the same, unless they don't exist)
    label_mapping = gce_quant_raw['label'].mode().T.to_dict()[0]
    label_mapping = {v: k for k, v in label_mapping.items()}
    gce_quant['ID'] = gce_quant['ID'].apply(lambda x: label_mapping[x] if x != 0 else '')

    return gce_quant


def convert_nwb_to_trace_dataframe(nwbfile):
    """
    Convert an NWB file to a dataframe of traces

    Parameters
    ----------
    nwbfile

    Returns
    -------

    """
    # Rois are the same for both channels
    activity = nwbfile.processing['CalciumActivity']
    try:
        rois = activity['CalciumSeriesSegmentationCoords'].plane_segmentations
    except KeyError:
        # Flavell style
        # This is just one time point, we need the full list... I think it doesn't exist
        # rois = np.array(activity['CalciumSeriesSegmentation'].plane_segmentations['Aligned_neuron_coordinates']['voxel_mask'])
        rois = None

    all_dfs = {}
    for channel in ['Signal', 'Reference']:
        # Extract the information as long vectors
        try:
            if f'{channel}DFoF' in activity.data_interfaces:
                red = activity[f'{channel}DFoF'][f'{channel}CalciumImResponseSeries'].data
            elif f'{channel}Fluorescence' in activity.data_interfaces:
                red = activity[f'{channel}Fluorescence'][f'{channel}CalciumImResponseSeries'].data
            elif f'{channel}RawFluor' in activity.data_interfaces:
                red = activity[f'{channel}RawFluor'][f'{channel}CalciumImResponseSeries'].data
            else:
                logging.debug(f"Failed to extract traces data for channel {channel}")
                continue
        except KeyError:
            logging.debug(f"Failed to extract traces data for channel {channel}")
            continue

        if rois is not None:
            _all_dfs = {}
            for name, r in tqdm(rois.items()):
                rois_list = r.voxel_mask.data[:]
                # This loop keeps the ids
                df = pd.DataFrame()
                df[['x', 'y', 'z', 'weight']] = np.array([r.astype(int) for r_inner in rois_list for r in r_inner]).reshape(-1,
                                                                                                                            4)
                _all_dfs[int(name.split('_')[-1])] = df
            df_traces = pd.concat(_all_dfs, axis=1)
            # Add labels
            for n in df_traces.columns.get_level_values(0).unique():
                df_traces.loc[:, (n, 'label')] = np.array(df_traces.index) + 1
            id_mapping = {n: int2name_neuron(n + 1) for n in df_traces.index}
            df_traces = df_traces.rename(index=id_mapping)
            df_traces = df_traces.unstack().swaplevel(0, 2).unstack().T.replace(0, np.nan).copy()

            # Add traces (column name is from opencv)
            red = pd.DataFrame(red)
            red = red.rename(columns=id_mapping)
            red = pd.concat({'intensity_image': red}, axis=1).swaplevel(0, 1, axis=1)
            df_traces = df_traces.join(red).sort_index()

        else:
            df_traces = pd.DataFrame(red)
            # Make the columns into the correct format: multiindex, with the first level being the neuron name
            # and the second being the features: ('label' for the int label, 'intensity_image' for the trace)

            # First make the labels, with the same number of rows as the traces (repeated label)
            df_labels = pd.DataFrame(index=df_traces.index, columns=df_traces.columns)
            df_labels.loc[:, :] = np.array(df_traces.columns.get_level_values(0)).reshape(1, -1)
            # Build columns: strings from the first level of the original columns
            col_names = [int2name_neuron(n+1) for n in df_traces.columns]
            df_labels.columns = pd.MultiIndex.from_product([col_names, ['label']])
            df_traces.columns = pd.MultiIndex.from_product([col_names, ['intensity_image']])
            # Combine the two dataframes using a multiindex
            df_traces = pd.concat([df_labels, df_traces], axis=1)

        all_dfs[channel] = df_traces

    return all_dfs


def nwb_from_pedro_format(folder_name: str, output_folder=None):
    """
    Pedro's manually organized format, which contains:
    1. ome.tif for the neuropal stacks
    2. .xlsx for the neuropal ID, with two sheets separated for head and tail
    3. .txt for fiji-exported positions of the ID'ed neurons. This is XY; Z is in the .xlsx file
    4. Optional: ome.tif for the head (if traces are calculated)
    5. Optional: ome.tif for the tail (if traces are calculated)
    6. .mat for metadata and the traces (if traces are calculated)

    Parameters
    ----------
    folder_name

    Returns
    -------

    """
    # Use file names and combine several files into useable format (not nwb yet)
    raw_project_files, df, session_start_time, strain_id, subject_id, fps, traces = \
        unpack_pedro_project(folder_name)

    # Load the neuropal stack (image)
    fname = [f for f in raw_project_files if ('NeuroPAL' in f) and f.endswith('.ome.tif')][0]
    fname = os.path.join(folder_name, fname)
    with tifffile.TiffFile(fname) as f:
        neuropal_stacks = f.asarray()
    # Want to have the dimensions be XYZC
    neuropal_stacks = neuropal_stacks.transpose((3, 2, 0, 1))

    # Pack everything as a NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain_id, subject_id)

    # Same as the one used for the freely moving experiments
    device = _zimmer_microscope_device(nwbfile)

    # Add neuropal stacks
    nwbfile = add_neuropal_stacks_to_nwb(nwbfile, device, neuropal_stacks)

    # segmentation and IDs
    NeuroPALImSeg = convert_segmentation_to_nwb(nwbfile, df)

    # Add certain things as separate processing modules
    finalize_nwb_processing_modules(NeuroPALImSeg, nwbfile)

    if output_folder:
        fname = os.path.join(output_folder, subject_id + '.nwb')
        logging.info(f"Saving NWB file to {fname}")
        with NWBHDF5IO(fname, mode='w') as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")

    return nwbfile


def finalize_nwb_processing_modules(NeuroPALImSeg, nwbfile):
    # First, unpack the objects
    calcium_image_series = None
    OpticalChannelRefs = nwbfile.imaging_planes['NeuroPALImVol'].order_optical_channels
    ImSeg = None
    SignalFluor = None
    CalcOptChanRefs = None

    # we add our raw NeuroPAL image to the acquisition module of the base NWB file
    if calcium_image_series is not None:
        nwbfile.add_acquisition(calcium_image_series)
    # we create a processing module for our neuroPAL data
    neuroPAL_module = nwbfile.create_processing_module(
        name='NeuroPAL',
        description='NeuroPAL image metadata and segmentation'
    )
    neuroPAL_module.add(NeuroPALImSeg)
    # neuroPAL_module.add(Seglabels) #optional, include if defining labels in separate SegmentationLabels object
    neuroPAL_module.add(OpticalChannelRefs)

    # we create a processing module for our calcium imaging data
    if ImSeg is not None:
        ophys = nwbfile.create_processing_module(
            name='CalciumActivity',
            description='Calcium time series metadata, segmentation, and fluorescence data'
        )
        ophys.add(ImSeg)
        # ophys.add(CalciumSegSeries) # comment out above line and uncomment this one if using indexed mask approach
        # ophys.add(FirstFrameSeg) # uncomment if using indexed mask approach
        ophys.add(SignalFluor)
        ophys.add(CalcOptChanRefs)
        # ophys.add(RefFluor)
        # ophys.add(ProcFluor)

    return nwbfile


def _zimmer_microscope_device(nwbfile):
    device = nwbfile.create_device(
        name="Spinning disk confocal",
        description="Zeiss Observer.Z1 Inverted Microscope with Yokogawa CSU-X1, "
                    "Zeiss LD LCI Plan-Apochromat 40x WI objective 1.2 NA",
        manufacturer="Zeiss"
    )
    return device


def unpack_pedro_project(folder_name):
    # Unpack the folder name into parts, which will form the basis of the input and output file names
    folder_dirname = Path(folder_name).name
    folder_parts = folder_dirname.split('_')
    raw_project_files = os.listdir(folder_name)
    raw_project_files = [f for f in raw_project_files if not f.startswith('.')]
    # Unpack these parts into metadata for the NWB file
    date_str = folder_parts[0]
    day = int(date_str[:2])
    month = int(date_str[2:4])
    year = int("20" + date_str[4:])
    session_start_time = datetime(year, month, day)
    strain_id = folder_parts[1]
    # Subject is date and then an integer for which worm on that day
    # Get this from the folder name, e.g. <date>_<strain>_worm<id>
    match = re.search(r'worm(\d+)', folder_dirname)
    if match:
        worm_number = int(match.group(1))
    else:
        print(f"Pattern 'worm' not found in the input string.")
        raise NotImplementedError
    subject_id = f"{session_start_time.strftime('%Y%m%d')}-{worm_number:02d}"
    # First load the ID and position data
    # File should be .xlsx and contain "NeuroPAL" in the name
    fname = [f for f in raw_project_files if ('NeuroPAL' in f) and f.endswith('.xlsx')][0]
    fname = os.path.join(folder_name, fname)
    sheet_name_base = f"{folder_parts[0]}_{folder_parts[2]}"
    all_dfs_excel = []
    for suffix in ['head', 'tail']:
        sheet_name = f"{sheet_name_base}_{suffix}"
        try:
            df = pd.read_excel(fname, sheet_name=sheet_name)
            df.loc[~(df['neuron ID'].isnull()), 'body_part'] = suffix
            all_dfs_excel.append(df)
        except ValueError as e:
            print("Did not find sheet", sheet_name, "in file", fname, " this is probably not a problem")
    df = all_dfs_excel[0].copy()
    if len(all_dfs_excel) > 1:
        df.update(all_dfs_excel[1])
    # Now load the XY position data
    fname = [f for f in raw_project_files if ('NeuroPAL' in f) and f.endswith('.txt')][0]
    fname = os.path.join(folder_name, fname)
    df_xy = pd.read_csv(fname, sep='\t', header=None)
    df_xy.columns = ['X', 'Y']
    df = pd.concat([df_xy, df], axis=1)

    # Add column for the z position, which is contained in the comments
    def _convert_comment_to_z(entry):
        if isinstance(entry, str):
            return int(entry.split(' ')[1])
        else:
            return np.nan

    df['Z'] = df['comments'].apply(_convert_comment_to_z)

    # Also load the .mat file to get the frames per second
    # NOT NEEDED IF NO TRACES

    fname = [f for f in raw_project_files if ('MainStruct' in f) and f.endswith('.mat')]
    if len(fname) > 0:
        fname = fname[0]
        fname = os.path.join(folder_name, fname)
        mat = scipy.io.loadmat(fname, simplify_cells=True)

        # Get the core mat dict, which is the only key in this object without __
        mat = mat[[k for k in mat.keys() if '__' not in k][0]]
        fps = mat['fps']

        traces = mat['traces']
    else:
        print(f"No traces found in {folder_name}, setting fps and traces to dummy values")
        fps = 0
        traces = []

    return raw_project_files, df, session_start_time, strain_id, subject_id, fps, traces


def add_neuropal_stacks_to_nwb(nwbfile, device, neuropal_stacks):
    """
    Add a neuropal stack to an existing NWB file.

    Parameters
    ----------
    nwbfile
    device
    neuropal_stacks

    Returns
    -------

    """

    # First, the metadata (hardcoded)

    # Channels is a list of tuples where each tuple contains the fluorophore used, the specific emission filter used, and a short description
    # structured as "excitation wavelength - emission filter center point- width of emission filter in nm"
    # Make sure this list is in the same order as the channels in your data
    channels = [("mNeptune 2.5", "Chroma ET 647/57", "561-647-57m"),
                ("Tag RFP-T", "Chroma ET 586/20", "561-586-20m"),
                ("CyOFP1", "BrightLine HC 617/73", "488-617-73m"),  # excited with blue, observe in red
                ("mTagBFP2", "BrightLine HC 447/60", "405-447-60m"),  # UV excited, observe in blue
                ("GFP-GCaMP", "BrightLine HC 525/50", "488-525-50m"),
                # ("CyOFP1-high filter", "Chroma ET 700/75", "488-700-75m"),
                #("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")
                ]
    # We also have mScarlet, which is not normally in neuropal

    OptChannels = []
    OptChanRefData = []
    # The loop below takes the list of channels and converts it into a list of OpticalChannelPlus objects which hold the metadata
    # for the optical channels used in the experiment
    for fluor, des, wave in channels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        OptChan = OpticalChannelPlus(
            name=fluor,
            description=des,
            excitation_lambda=excite,
            excitation_range=[excite - 1.5, excite + 1.5],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid
        )

        OptChannels.append(OptChan)
        OptChanRefData.append(wave)

    # This object just contains references to the order of channels because OptChannels does not preserve ordering by itself
    OpticalChannelRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=OptChanRefData
    )

    ImagingVol = ImagingVolume(
        name='NeuroPALImVol',
        # Add connections to the OptChannels and OpticalChannelRefs objects
        optical_channel_plus=OptChannels,
        order_optical_channels=OpticalChannelRefs,
        # Free form description of what is being imaged in this volume
        description='NeuroPAL image of C. elegans brain',
        # Reference the device created earlier that was used to acquire this data
        device=device,
        # Specifies where in the C. elegans body the image is being taken of
        location="Head and Tail",
        # Specifies the voxel spacing in x, y, z respectively. The values specified should be how many micrometers of physical
        # distance are covered by a single pixel in each dimension
        # TODO: grid spacing
        grid_spacing=[0.4, 0.4, 0.2],
        grid_spacing_unit='micrometers',
        # Origin coords, origin coords unit, and reference frames are carry over fields from other model organisms where you
        # are likely only looking at a small portion of the brain. These fields are unfortunately required but feel free to put
        # whatever feels right here
        origin_coords=[0, 0, 0],
        origin_coords_unit="micrometers",
        reference_frame="Worm head"
    )

    nwbfile.add_imaging_plane(ImagingVol)  # add this ImagingVol to the nwbfile

    # Then, the data
    data = neuropal_stacks
    RGBW_channels = [0, 1, 2, 3]

    Image = MultiChannelVolume(
        name='NeuroPALImageRaw',
        # This is the same OpticalChannelRefs used in the associated Imaging Volume
        order_optical_channels=OpticalChannelRefs,
        description='free form description of image',
        # Specifies which channels in the image are associated with the RGBW channels - should be a list of channel indices as shown above
        RGBW_channels=RGBW_channels,
        # This is the raw data numpy array that we loaded above
        data=H5DataIO(data=data, compression=True),
        # This is a reference to the Imaging Volume object we defined previously
        imaging_volume=ImagingVol
    )

    nwbfile.add_acquisition(Image)

    return nwbfile


def convert_segmentation_to_nwb(nwbfile, df):
    """
    Uses an unpacked matlab file to add segmentation and IDs to create an ImageSegmentation object.

    Will be added to the NWB file as a processing module.

    Parameters
    ----------
    nwbfile
    df

    Returns
    -------

    """
    # Unpack the imaging volume from the main object
    ImagingVol = nwbfile.imaging_planes['NeuroPALImVol']

    vs = PlaneSegmentation(
        name='NeuroPALNeurons',
        description='Neuron centers for multichannel volumetric image. Weight set at 1 for all voxels. Labels refers to cell ID of segmented neurons',
        #Reference the same ImagingVolume that your image was taken with
        imaging_plane=ImagingVol,
    )

    # Use 'blobs' to follow the tutorial
    blobs = df.copy()
    IDs = blobs['neuron ID']
    labels = IDs.replace(np.nan, '', regex=True)
    labels = list(np.asarray(labels))

    valid_ids = []
    for i, row in blobs.iterrows():
        voxel_mask = []
        x = row['X']
        y = row['Y']
        z = row['Z']
        weight = 1

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            # These are just extra rows in the excel file that don't have any data
            continue

        voxel_mask.append([np.uint(x), np.uint(y), np.uint(z), weight])
        vs.add_roi(voxel_mask=voxel_mask)
        valid_ids.append(i)

    # Add ID's of valid neurons
    labels = [labels[i] for i in valid_ids]
    vs.add_column(
        name='ID_labels',
        description='ROI ID labels',
        data=labels,
        index=True,
    )

    NeuroPALImSeg = ImageSegmentation(
        name='NeuroPALSegmentation',
    )
    NeuroPALImSeg.add_plane_segmentation(vs)

    return NeuroPALImSeg


def plot_image_and_blobs(nwbfile):
    # Unpack
    with NWBHDF5IO(nwbfile, mode='r', load_namespaces=True) as io:
        read_nwbfile = io.read()
        try:
            seg = read_nwbfile.processing['CalciumActivity']['CalciumSeriesSegmentation']['Seg_tpoint_0'].voxel_mask[:]
        except TypeError:
            seg = read_nwbfile.processing['CalciumActivity']['SegmentationVol0']['Seg_tpoint_0'].voxel_mask[:]
        # print(seg)
        image = read_nwbfile.acquisition['CalciumImageSeries'].data[0, ...]

    # Build variables for plotting
    blobs = pd.DataFrame.from_records(seg, columns=['X', 'Y', 'Z', 'weight'])
    blobs = blobs.drop(['weight'], axis=1)
    blobs = blobs.replace('nan', np.nan, regex=True)
    # print(blobs)

    # blobs['x'] = np.round(blobs['x'] / 0.4)
    # blobs['Y'] = np.round(blobs['y'] / 0.4)
    # print(proc_image.shape)

    RGB = image[:, :, :, :-1]

    Zmax = np.max(RGB, axis=2)
    Ymax = np.max(RGB, axis=1)

    plt.figure(figsize=(10, 10))

    plt.imshow(np.transpose(Zmax, [1, 0, 2]))
    plt.scatter(blobs['x'], blobs['y'], s=5, alpha=0.5, color='white')
    plt.xlim((0, Zmax.shape[0]))
    plt.ylim((0, Zmax.shape[1]))
    plt.gca().set_aspect('equal')

    plt.show()

    plt.figure()

    plt.imshow(np.transpose(Ymax, [1, 0, 2]))
    plt.scatter(blobs['x'], blobs['z'], s=5, alpha=0.5, color='white')
    plt.xlim((0, Ymax.shape[0]))
    plt.ylim((0, Ymax.shape[1]))
    plt.gca().set_aspect('equal')

    plt.show()


def plot_image_and_segmentation(nwbfile):
    # Unpack
    with NWBHDF5IO(nwbfile, mode='r', load_namespaces=True) as io:
        read_nwbfile = io.read()
        # Only works if the full segmentation is saved
        seg = read_nwbfile.processing['CalciumActivity']['CalciumSeriesSegmentation'].data[0, ...]
        # print(seg)
        image = read_nwbfile.acquisition['CalciumImageSeries'].data[0, ...]

    # Build variables for plotting
    RGB = image[:, :, :, :-1]

    Zmax = np.max(RGB, axis=2)
    seg_zmax = np.max(seg, axis=2)

    Ymax = np.max(RGB, axis=1)
    seg_ymax = np.max(seg, axis=1)

    plt.figure(figsize=(10, 10))

    plt.imshow(np.transpose(Zmax, [1, 0, 2]))
    plt.imshow(np.transpose(seg_zmax, [1, 0]), alpha=0.5, cmap='tab20')
    # plt.scatter(blobs['x'], blobs['y'], s=5, alpha=0.5, color='white')
    plt.xlim((0, Zmax.shape[0]))
    plt.ylim((0, Zmax.shape[1]))
    plt.gca().set_aspect('equal')

    plt.show()

    plt.figure()

    plt.imshow(np.transpose(Ymax, [1, 0, 2]))
    plt.imshow(np.transpose(seg_ymax, [1, 0]), alpha=0.5, cmap='tab20')
    plt.xlim((0, Ymax.shape[0]))
    plt.ylim((0, Ymax.shape[1]))
    plt.gca().set_aspect('equal')

    plt.show()


class CustomDataChunkIterator(GenericDataChunkIterator):
    """
    Needed because the non-abstract default DataChunkIterator doesn't allow chunk_shape specification
    See: https://hdmf.readthedocs.io/en/stable/hdmf.data_utils.html#hdmf.data_utils.DataChunkIterator

    Code copied from tutorial: https://hdmf.readthedocs.io/en/stable/tutorials/plot_generic_data_chunk_tutorial.html

    I think I don't need to define a __next__ method because numpy + chunk_size takes care of it
    """
    def __init__(self, array: np.ndarray, **kwargs):
        self.array = array
        super().__init__(**kwargs)

    def _get_data(self, selection):
        return np.array(self.array[selection])

    def _get_maxshape(self):
        return self.array.shape

    def _get_dtype(self):
        return self.array.dtype


def df_to_nwb_tracking(df, timestamps=None, reference_frame="unknown", unit="pixels", centroids_are_tracked=True):
    """
    Converts a MultiIndex DataFrame to NWB SpatialSeries + object-ID table.

    df: pandas DataFrame indexed by, e.g., time and object_id, with cols x,y,z (optional: also raw_segmentation_id)
    timestamps: array of timestamps aligned to df.index (first level should be time)

    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected df.columns to be a MultiIndex of (neuron_id, coord)")

    neuron_ids = df.columns.get_level_values(0).unique()
    coord_names = ['x', 'y', 'z']  # or whatever coordinates you have
    if not all(c in df.columns.get_level_values(1) for c in coord_names):
        coord_names = None
    if timestamps is None:
        timestamps = np.arange(len(df.index))

    if centroids_are_tracked:
        # Build DynamicTable of neurons with correct raw segmentation IDs for all time points
        dt = DynamicTable(name="NeuronSegmentationID", description="Segmentation IDs per neuron", id=timestamps)
        for nid in tqdm(neuron_ids, desc="Adding neuron IDs to DynamicTable"):
            seg_ids = df.loc[:, (nid, 'raw_segmentation_id')].values
            dt.add_column(name=str(nid), description=f"Raw Seg ID for {nid}", data=seg_ids)
        position_name = "NeuronCentroids"
    else:
        dt = None
        position_name = "NeuronCentroidsUntracked"

    if coord_names is not None:
        position = Position(name=position_name)
        for neuron in tqdm(neuron_ids, desc="Adding centroids to Position"):
            neuron_df = df[neuron]
            data = neuron_df[coord_names].to_numpy()

            ss = SpatialSeries(
                name=neuron,
                data=data,
                unit=unit,
                reference_frame=reference_frame,
                timestamps=timestamps
            )
            position.add_spatial_series(ss)
    else:
        position = None

    return position, dt


def load_per_neuron_position(nwbfile_module):
    data_dict = {}
    timestamps = None

    for series in nwbfile_module.spatial_series.values():
        data = series.data[:]
        neuron = series.name
        coords = ['x', 'y', 'z'][:data.shape[1]]
        for i, coord in enumerate(coords):
            data_dict[(neuron, coord)] = data[:, i]
        if timestamps is None:
            timestamps = series.timestamps[:]

    df = pd.DataFrame(data_dict, index=timestamps)

    # Currently the Leifer data is stored just as xyz positions in a very tall dataframe, and it should be converted to a multiindex
    if df.shape[1] == 3:
        logging.warning("Detected single-series (tall) format for centroids, converting to multiindex format")
        # Reshape from (objects per time)*(time) x 3 and seg IDs to (time) x (objects per time) x 4

        # Drop the improperly formed multiindex column level
        df.columns = df.columns.droplevel(0)

        # The segmentation id (track id) is stored in the "control" field of the series (there is only one series in this case)
        assert len(nwbfile_module.spatial_series) == 1, "Expected only one spatial series"
        series = list(nwbfile_module.spatial_series.values())[0]
        raw_neuron_ind_in_list = series.control[:]
        # Pivot the values, multiindex columns = (object_id, variable)
        df['raw_neuron_ind_in_list'] = raw_neuron_ind_in_list
        df['raw_segmentation_id'] = df['raw_neuron_ind_in_list'] + 1
        df['neuron_name'] = df['raw_segmentation_id'].apply(lambda x: int2name_neuron(x))
        df = df.reset_index(names='time').pivot(index="time", columns="neuron_name", values=["x", "y", "z", "raw_segmentation_id", "raw_neuron_ind_in_list"])

        # Reorder to (neuron_name, z/x/y)
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        df, _ = fill_missing_indices_with_nan(df, expected_max_t=int(df.index.max()))
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df


def compute_centroids_parallel(seg_dask, intensity_dask=None):
    """
    Compute centroids for each label in each timepoint using dask.delayed
    
    Args:
        seg_dask: dask array (T, X, Y, Z), integer labels
        intensity_dask (optional): dask array (T, X, Y, Z), intensity values
    Returns:
        centroids: dict {time: {raw_segmentation_index: (x, y, z)}}
    """
    def process_timepoint(seg, intensity, t):
        props = regionprops(seg.astype(int), intensity_image=intensity)
        centroids_t = {}
        for prop in props:
            if intensity is not None:
                centroid = tuple(float(c) for c in prop.weighted_centroid)
            else:
                centroid = tuple(float(c) for c in prop.centroid)
            label = int(prop.label)
            centroids_t[label] = centroid
        return t, centroids_t

    tasks = []
    n_timepoints = seg_dask.shape[0]
    for t in range(n_timepoints):
        seg = seg_dask[t]
        if intensity_dask is not None:
            intensity = intensity_dask[t]
        else:
            intensity = None
        tasks.append(delayed(process_timepoint)(seg, intensity, t))

    results = compute(*tasks, scheduler='threads')  # or 'processes'
    # Assemble into dict
    centroids = {t: centroids_t for t, centroids_t in results}
    return centroids


def add_centroid_data_to_df_tracking(seg_dask, df_tracking, df_tracking_offset=0):
    """df_tracking should have a column that matches the raw segmentation id in each time point to the true or tracked ids"""

    # This doesn't have centroid information, so add it in
    centroids_dict = compute_centroids_parallel(seg_dask)
    # Based on the segmentation ids in tracking_df, create the xyz columns that should be added
    all_neurons = list(df_tracking.columns.get_level_values(0).unique())
    # A dict with 3 entries per neuron: (neuron, x), (neuron, y), (neuron, z) -> (respective array)
    coord_names = ['x', 'y', 'z']
    all_keys = itertools.product(all_neurons, coord_names)
    def _init_nan_numpy():
        _array = np.empty(np.max(df_tracking.index.values) + 1)  # Should not be the shape of df_tracking, which might have empty rows
        _array[:] = np.nan
        return _array
    mapped_centroids_dict = {k: _init_nan_numpy() for k in all_keys}
    
    # Note that df_tracking is 1-indexed, so the index will have to be fixed later
    for t, these_centroids in tqdm(centroids_dict.items(), desc="Mapping centroids to segmentation"):
        if t + df_tracking_offset not in df_tracking.index:
            logging.warning(f"Time {t + df_tracking_offset} not found in df_tracking index, skipping")
            continue
        # For each neuron, get the raw_segmentation_id at this time point and map it to the centroid coordinates
        for neuron in all_neurons:
            try:
                raw_seg = df_tracking.loc[t+df_tracking_offset, (neuron, 'raw_segmentation_id')]
            except KeyError:
                logging.warning(f"Tracking DataFrame does not contain entry for neuron {neuron} at time {t+df_tracking_offset}")
                continue
            if np.isnan(raw_seg):
                continue

            try:
                this_centroid = these_centroids[int(raw_seg)]
            except KeyError:
                logging.error(f"Ground truth was annotated as {int(raw_seg)} at t={t}, but it doesn't exist in the image... this probably means the ground truth has an off-by-one error")
            for _name, _c in zip(coord_names, this_centroid):
                mapped_centroids_dict[(neuron, _name)][t] = _c
    # Convert to dataframe, then combine with original tracking dataframe
    df_centroids = pd.DataFrame(mapped_centroids_dict)
    # assert df_centroids.index.equals(df_tracking.index), f"Centroid DataFrame index {df_centroids.index} does not match tracking DataFrame index {df_tracking.index}"
    df_tracking = pd.concat([df_centroids, df_tracking], axis=1)

    return df_tracking


def calc_simple_segmentation_mapping_table(df_tracking):
    """Create nwb table with the ids, assuming the column names correspond to the raw segmentation ids"""
    timestamps = np.arange(len(df_tracking.index))
    neuron_ids = df_tracking.columns.get_level_values(0).unique()
    dt = DynamicTable(name="NeuronSegmentationID", description="Segmentation IDs per neuron", id=timestamps)
    for nid in tqdm(neuron_ids, desc="Adding neuron IDs to DynamicTable"):
        # Create a vector with the id (i.e. column name) for each time point with a valid centroid, otherwise NaN
        seg_ids = np.zeros(len(timestamps), dtype=float)
        seg_ids[:] = name2int_neuron_and_tracklet(nid)  # Convert the neuron name to an integer ID
        # Remove any invalid time points, assuming all xyz coordinates are either present or nan
        valid_times = timestamps[df_tracking.loc[:, (nid, 'x')].notna()]
        seg_ids[~np.isin(timestamps, valid_times)] = np.nan
        # Add the column to the DynamicTable
        dt.add_column(name=str(nid), description=f"Raw Seg ID for {nid}", data=seg_ids)
    return dt


def add_segmentation_ids_given_video_segmentation(nwb_file, DEBUG=False):
    """
    In the case where the final neuron centroids are saved in the nwb file, but the table mapping those ids to neuron names is not,
    this function will add the centroid ids to the tracking dataframe.

    This assumes that any segmentation will be generated from the centroids themselves, and retain the same ids as the centroids.
    """
    # Load the NWB file
    with NWBHDF5IO(nwb_file, mode='r+') as nwb_io:
        nwb_obj = nwb_io.read()

        activity = nwb_obj.processing['CalciumActivity']
        # Check if the segmentation ids are already present
        if 'NeuronSegmentationID' in activity.data_interfaces:
            logging.info(f"Segmentation IDs already present in {nwb_file}, skipping addition.")
            return nwb_obj

        # Get the final centroid ids
        df_tracking = load_per_neuron_position(activity['NeuronCentroids'])
        if DEBUG:
            print(f"Loaded tracking DataFrame with shape {df_tracking.shape}: {df_tracking.head()}")

        dt = calc_simple_segmentation_mapping_table(df_tracking)
        nwb_obj.processing['CalciumActivity'].add(dt)

        # Save the NWB file
        nwb_io.write(nwb_obj)
        
    logging.info(f"Added segmentation IDs to NWB file {nwb_file}.")
    return nwb_obj


def dask_stack_volumes(volume_iter):
    """Stack a generator of volumes into a dask array along time."""
    return da.stack(volume_iter, axis=0)


def segment_from_centroids_using_watershed(centroids, video, compactness=0.5, dtype=np.uint16, noise_threshold=3, DEBUG=False):

    if len(video.shape) == 5:
        video = video[..., 0]  # Just take the red channel
    T, X, Y, Z = video.shape

    client = get_client()
    video_future = client.scatter(video, broadcast=True)
    
    # Stack results
    segmented_video = dask_stack_volumes([da.from_delayed(_iter_segment_video(video_future, centroids[t], t, noise_threshold, dtype, compactness), shape=(X, Y, Z), dtype=dtype) for t in range(T)])

    return segmented_video

@delayed
def _iter_segment_video(video, frame_centroids, t, noise_threshold, dtype, compactness):
    """Segment a single timepoint using watershed"""
    video_frame = video[t]
    X, Y, Z = video_frame.shape
    
    # Create markers as a full-size volume from centroids
    markers = np.zeros_like(video_frame, dtype=np.int32)
    
    for i, (x, y, z) in enumerate(frame_centroids):
        # Convert to integer coordinates and ensure they're within bounds
        if np.isnan(x):
            continue
        x_int, y_int, z_int = int(round(x)), int(round(y)), int(round(z))
        
        if (0 <= z_int < Z and 0 <= y_int < Y and 0 <= x_int < X):
            # Labels start from 1, and assume the centroids are in the correct order
            markers[x_int, y_int, z_int] = i + 1
            # Also make sure that this exact point is not 0 in the video
            if video_frame[x_int, y_int, z_int] <= noise_threshold:
                video_frame[x_int, y_int, z_int] = noise_threshold + 1
                logging.warning(f"Very dim centroid ({i+1}) found for time {t}; setting coordinates to threshold level ({noise_threshold+1}) ")
    
    # If no valid markers, return empty segmentation
    if markers.max() == 0:
        return np.zeros_like(video_frame, dtype=dtype)
    
    try:
        # Apply distance transform to the volume
        distance = ndi.distance_transform_edt(video_frame)
    
        # Apply watershed segmentation
        segmentation = watershed(
            -distance,
            markers, 
            compactness=compactness,
            mask=video_frame > noise_threshold,
            watershed_line=False
        )
    except Exception as e:
        logging.warning(f"Watershed failed for frame {t} with {len(markers)} markers: {e}")
        return np.zeros_like(video_frame, dtype=dtype)

    # # Remap segmentation so each region gets the marker label that seeded it (watershed skips missing indices)  
    # unique_labels = np.unique(segmentation)
    # remapped = np.zeros_like(segmentation, dtype=dtype)
    # for label in unique_labels:
    #     if label == 0:
    #         continue  # background
    #     # Find which marker generated this region
    #     mask = (segmentation == label)
    #     marker_labels = markers[mask]
    #     marker_labels = marker_labels[marker_labels > 0]
    #     if len(marker_labels) > 0:
    #     # Assign the most common marker label to the region
    #         new_label = np.bincount(marker_labels).argmax()
    #     remapped[mask] = new_label
    # segmentation = remapped
    # yield da.from_array(segmentation.astype(dtype))
    return segmentation.astype(dtype)
            
        # except Exception as e:
        #     logging.warning(f"Watershed failed for frame {frame_idx}: {e}")
        #     return markers.astype(dtype)


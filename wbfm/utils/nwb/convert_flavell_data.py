import os
import nrrd
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import OpticalChannel
from ndx_multichannel_volume import MultiChannelVolumeSeries
from datetime import datetime
from hdmf.backends.hdf5.h5_utils import H5DataIO
import glob
import argparse
from wbfm.utils.nwb.utils_nwb_export import CustomDataChunkIterator, build_optical_channel_objects, _zimmer_microscope_device, add_centroid_data_to_df_tracking, df_to_nwb_tracking
import dask.array as da
from tqdm import tqdm
import json
import pandas as pd


def iter_volumes(base_dir, n_start, n_timepoints, channel=None, segmentation=False):
    """Yield 3D volumes for a given channel. If a file is missing, yield an array of zeros with the correct shape."""
    zero_shape = None
    for t in tqdm(range(n_start, n_timepoints), leave=False, desc=f"Iterating through volumes from video: channel={channel}, segmentation={segmentation}"):
        pattern = get_flavell_timepoint_pattern(base_dir, t, channel, segmentation)
        matches = glob.glob(pattern)
        if matches:
            path = matches[0]
            if os.path.exists(path):
                data, _ = nrrd.read(path)
                if zero_shape is None:
                    zero_shape = data.shape
                yield data
                continue
        # If file is missing, yield zeros
        if zero_shape is not None:
            yield np.zeros(zero_shape, dtype=np.float32)
        else:
            # Try to infer shape from the first available file in the directory
            fallback_pattern = get_flavell_channel_pattern(base_dir, channel, segmentation)
            fallback_matches = glob.glob(fallback_pattern)
            if fallback_matches:
                fallback_data, _ = nrrd.read(fallback_matches[0])
                zero_shape = fallback_data.shape
                yield np.zeros(zero_shape, dtype=np.float32)
            else:
                raise RuntimeError(f"Cannot determine shape for channel {channel} at time {t}. No files found.")


def get_flavell_channel_pattern(base_dir, channel=0, segmentation=False):
    """Get the glob pattern for a specific channel in NRRD_cropped."""
    if not segmentation:
        return f'{base_dir}/NRRD_cropped/*_ch{channel}.nrrd'
    else:
        return f'{base_dir}/img_roi_watershed/*.nrrd'


def get_flavell_timepoint_pattern(base_dir, t, channel=0, segmentation=False):
    """Get the glob pattern for a specific timepoint and channel in NRRD_cropped."""
    if not segmentation:
        t_str = f"{t:04d}"
        return f'{base_dir}/NRRD_cropped/*_t{t_str}_ch{channel}.nrrd'
    else:
        return f'{base_dir}/img_roi_watershed/{t}.nrrd'


def get_flavell_tracking_file(base_dir):
    """Get the glob pattern for tracking data in NRRD_cropped."""
    tracking_pattern = f'{base_dir}/*inv_map.json'
    tracking_files = glob.glob(tracking_pattern)
    if not tracking_files:
        raise FileNotFoundError(f"No tracking files found in {base_dir} with pattern {tracking_pattern}")
    if len(tracking_files) > 1:
        raise RuntimeError(f"Multiple tracking files found in {base_dir}: {tracking_files}. Please ensure only one exists.")    
    return tracking_files[0]


def convert_flavell_tracking_to_df(base_dir, tracking_fraction_threshold=0.5, DEBUG=False):
    """Convert Flavell tracking data to DataFrame format."""
    tracking_file = get_flavell_tracking_file(base_dir)
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
        # Build DataFrame: rows=time, columns=UIDs (i.e. neuron names), values=segmentation id (of the raw segmentation)
        df = pd.DataFrame.from_dict(tracking_data, orient='index').T
        if DEBUG:
            print(f"Tracking data loaded from {tracking_file} with shape {df.shape}")
            print(df.head())
        # Flatten any list values in the DataFrame; Some UIDs may have multiple segmentation IDs... technically should be the union of the two
        def _process_elements(x):
            if isinstance(x, list):
                return x[0] if len(x) == 1 else np.nan
            elif isinstance(x, (int, float)):
                # assert np.isnan(x) # This is the only case where we expect a single value
                return x
            else:
                raise TypeError(f"Unexpected type {type(x)} in tracking data: {x}; expected list")
        df = df.applymap(_process_elements)
        if DEBUG:
            print(f"After processing of list data, has shape {df.shape} with df.count()={df.count()}")
            print(df.head())
        # Remove objects with too few tracking points
        df = df.loc[:, df.notna().sum() >= tracking_fraction_threshold * len(df)]
        if DEBUG:
            print(f"After removing low tracking objects, has shape {df.shape} with df.count()={df.count()}")
            print(df.head())

        # Convert datatypes
        if DEBUG:
            print(f"Converting index from")
        df.index.name = 'time'
        # Translate from 1-based to 0-based indexing
        df.index = [int(i) - 1 for i in df.index]
        df.sort_index(inplace=True)
        # Needed for NWB compatibility
        df.columns = df.columns.astype(str)

        # Add MultiIndex column
        df.columns = pd.MultiIndex.from_product([df.columns, ['raw_segmentation_id']])
        if DEBUG:
            print(f"Final DataFrame has shape {df.shape} with df.count()={df.count()}")
            print(df.head())
    return df, tracking_file


def find_min_max_timepoint(base_dir, channel=None, segmentation=False):
    """Find the minimum and maximum timepoint index for a given channel in NRRD_cropped."""
    pattern = get_flavell_channel_pattern(base_dir, channel, segmentation)
    matches = glob.glob(pattern)
    min_t = None
    max_t = -1
    for path in matches:
        # Extract tXXXX from filename if it is a volume, otherwise use the filename directly
        basename = os.path.basename(path)
        if segmentation:
            # Just the basename without the extension
            parts = basename.split('.')
            if len(parts) > 1 and parts[0].isdigit():
                t = int(parts[0])
            else:
                continue
        else:
            parts = basename.split('_')
            for part in parts:
                if part.startswith('t') and part[1:5].isdigit():
                    t = int(part[1:5])
                    break
            else:
                continue
        if min_t is None or t < min_t:
            min_t = t
        if t > max_t:
            max_t = t
    return min_t, max_t


def count_valid_volumes(base_dir, channel):
    """Count valid volumes for a channel."""
    pattern = f'{base_dir}/NRRD_cropped/*_ch{channel}.nrrd'
    matches = glob.glob(pattern)
    return len(matches)


def count_valid_segmentations(base_dir, n_timepoints):
    count = 0
    for t in range(n_timepoints):
        path = f'{base_dir}/img_roi_watershed/{t}.nrrd'
        if os.path.exists(path):
            count += 1
    return count


def dask_stack_volumes(volume_iter, frame_shape):
    """Stack a generator of volumes into a dask array."""
    # Each block is a single volume (3D), stacked along axis=0 (time)
    return da.stack(volume_iter, axis=0)


def create_nwb_file_only_images(session_description, identifier, session_start_time, device_name, imaging_rate):
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        lab='Flavell lab',
        institution='MIT'
    )
    device = nwbfile.create_device(name=device_name)
    opt_ch_green = OpticalChannel('GFP', 'green channel', 488.0)
    opt_ch_red = OpticalChannel('RFP', 'red channel', 561.0)
    imaging_plane = nwbfile.create_imaging_plane(
        name='ImagingPlane',
        optical_channel=[opt_ch_green, opt_ch_red],
        description='Dummy imaging plane',
        device=device,
        excitation_lambda=488.0,
        imaging_rate=imaging_rate,
        indicator='GFP/RFP',
        location='unknown'
    )
    return nwbfile, imaging_plane


def convert_flavell_to_nwb(
    base_dir,
    output_path,
    session_description='Dummy Flavell data conversion',
    identifier='flavell_dummy',
    device_name='Microscope',
    imaging_rate=1.0,
    DEBUG=False
):
    session_start_time = datetime.now()
    nwbfile, imaging_plane = create_nwb_file_only_images(
        session_description, identifier, session_start_time, device_name, imaging_rate
    )

    # Count valid frames for each channel
    if DEBUG:
        start_frame, end_frame = 1, 20
        print(f"DEBUG mode: limiting to frames {start_frame} to {end_frame}")
    else:
        min_green, n_green = find_min_max_timepoint(base_dir, 1, False)
        min_red, n_red = find_min_max_timepoint(base_dir, 2, False)
        min_seg, n_seg = find_min_max_timepoint(base_dir, segmentation=True)
        # We know the last frame here actually exists, so add 1
        end_frame = max(n_green, n_red, n_seg) + 1
        start_frame = min(min_green, min_red, min_seg)
        if end_frame == 0:
            raise RuntimeError("No valid frames found for all channels.")

    # Use the first valid green volume to get shape
    green_gen = iter_volumes(base_dir, start_frame, end_frame, 1)
    try:
        first_green = next(green_gen)
    except StopIteration:
        raise RuntimeError("No green channel volumes found. Check your input data and n_frames value.")
    frame_shape = first_green.shape

    # Build dask arrays for each channel
    green_dask = dask_stack_volumes(iter_volumes(base_dir, start_frame, end_frame, 1), frame_shape)
    red_dask = dask_stack_volumes(iter_volumes(base_dir, start_frame, end_frame, 2), frame_shape)
    seg_dask = dask_stack_volumes(iter_volumes(base_dir, start_frame, end_frame, segmentation=True), frame_shape)
    
    print(f"Found red video with shape {red_dask.shape} and green video with shape {green_dask.shape}")
    print(f"Found segmentation data with shape {seg_dask.shape}")
    
    # Make single multi-channel data series
    # Flavell data is already TXYZ, so stack to make a 5D TXYZC array
    red_green_dask = da.stack([red_dask, green_dask], axis=-1)

    # Ensure chunk_video matches the number of dimensions in red_green_dask
    chunk_video = (1,) + red_green_dask.shape[1:-1] + (1,)
    print(f"Creating NWB file with chunk size {chunk_video} and size {red_green_dask.shape} for green/red data")
    green_red_data = H5DataIO(
        data=CustomDataChunkIterator(array=red_green_dask, chunk_shape=chunk_video, display_progress=True),
        compression="gzip"
    )

    chunk_seg = (1,) + frame_shape  # chunk along time only
    print(f"Segmentations will be stored with chunk size {chunk_seg} and size {seg_dask.shape}")
    seg_data = H5DataIO(
        data=CustomDataChunkIterator(array=seg_dask, chunk_shape=chunk_seg, display_progress=True),
        compression="gzip"
    )

    # Build metadata objects
    grid_spacing = (0.3, 0.3, 0.3)  # Flavell data is isotropic
    device = _zimmer_microscope_device(nwbfile)
    CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
    # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
    nwbfile.add_imaging_plane(CalcImagingVolume)

    nwbfile.add_acquisition(MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        description="Series of calcium imaging data",
        comments="Calcium imaging data from Flavell lab",
        data=green_red_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        dimension=frame_shape,  # Gives a warning; what should this be?
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=imaging_rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    ))
    # Add segmentation under the processed module
    calcium_imaging_module = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )
    calcium_imaging_module.add(MultiChannelVolumeSeries(
        name="CalciumSeriesSegmentationUntracked",
        description="Series of indexed masks associated with calcium segmentation",
        comments="Segmentation masks for calcium imaging data from Flavell lab",
        data=seg_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        dimension=frame_shape,  # Gives a warning; what should this be?
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=imaging_rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    ))

    # Add tracking information
    df_tracking, tracking_file = convert_flavell_tracking_to_df(base_dir, DEBUG=DEBUG)
    if df_tracking.empty:
        raise RuntimeError(f"Tracking file did not produce a valid dataframe; check formatting of {tracking_file}")
    
    df_tracking = add_centroid_data_to_df_tracking(seg_dask, df_tracking)
    
    position, dt = df_to_nwb_tracking(df_tracking)
    if position is not None:
        calcium_imaging_module.add(position)
    else:
        raise ValueError("Position should be created")

    calcium_imaging_module.add(dt)

    with NWBHDF5IO(output_path, 'w') as io:
        io.write(nwbfile)
    print(f"Done. NWB file written to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Flavell data to NWB format.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing input data, including: NRRD_cropped and img_roi_watershed folders and' \
        '.json files with tracking data')
    parser.add_argument('--output_path', type=str, required=False, help='Output NWB file path')
    parser.add_argument('--session_description', type=str, default='Flavell Lab Data', help='Session description')
    parser.add_argument('--identifier', type=str, default='flavell_001', help='NWB file identifier')
    parser.add_argument('--device_name', type=str, default='FlavellMicroscope', help='Device name')
    parser.add_argument('--imaging_rate', type=float, default=1.0, help='Imaging rate (Hz)')
    parser.add_argument('--debug', action='store_true', help='If set, only convert the first 10 time points')

    args = parser.parse_args()

    # If the output path is not an absolute path, make it absolute by joining with the base_dir
    if args.output_path is None:
        suffix = '_DEBUG' if args.debug else ''
        args.output_path = os.path.join(args.base_dir, f'flavell_data{suffix}.nwb')  
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(args.base_dir, args.output_path)

    convert_flavell_to_nwb(
        base_dir=args.base_dir,
        output_path=args.output_path,
        session_description=args.session_description,
        identifier=args.identifier,
        device_name=args.device_name,
        imaging_rate=args.imaging_rate,
        DEBUG=args.debug
    )

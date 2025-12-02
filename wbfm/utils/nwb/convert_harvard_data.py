import h5py
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.behavior import Position, SpatialSeries
from ndx_multichannel_volume import MultiChannelVolumeSeries
from wbfm.utils.nwb.utils_nwb_export import build_optical_channel_objects, calc_simple_segmentation_mapping_table, load_per_neuron_position,
CustomDataChunkIterator, dask_stack_volumes, segment_from_centroids_using_watershed
from pynwb import TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal
import dask.array as da
from pathlib import Path
import argparse
from dask import delayed
import logging
from scipy import ndimage as ndi
from tqdm.auto import tqdm
import time
from dask.distributed import Client, get_client

def iter_frames(h5_file, n_timepoints, frame_shape):
    for i in range(n_timepoints):
        data = h5_file[str(i)]["frame"][...]
        yield da.from_array(data, chunks=frame_shape).transpose((1, 2, 3, 0))  # Lazy loading

@delayed
def load_frame(h5_path, i):
    with h5py.File(h5_path, "r") as h5_file:
        data = h5_file[str(i)]["frame"][...]
        return np.array(data).transpose((1, 2, 3, 0))  # Indirect lazy loading via dask.delayed


def convert_harvard_to_nwb(input_path, 
                           output_path=None,
                           session_description="Harvard",
                           identifier="Harvard",
                           device_name="Harvard",
                           imaging_rate=10.0,
                           eager_segmentation_mode=False,
                           DEBUG=False):

    start_time = time.time()

    # === USER PARAMETERS ===
    experiment_name = input_path.split("/")[-1].split(".")[0]
    if output_path is None:
        output_path = Path(input_path).with_suffix('.nwb')

    with h5py.File(input_path, "r") as f:

        # === Create NWBFile ===
        nwbfile = NWBFile(
            session_description=session_description,
            identifier=identifier,
            session_start_time=datetime.now(tz=tzlocal()),
            experimenter="Harvard Guy "+str(experiment_name),
            lab="Some Harvard Lab",
            institution="Harvard",
        )

        # === Add Subject ===
        nwbfile.subject = Subject(
            subject_id="subject "+experiment_name,
            species="Caenorhabditis elegans",
            description="Tiny worm",
        )

        # === Add Fluorescence Data ===
        ci_int = f["ci_int"][:]  # shape (97, 1331, 12)
        # Unclear if this is just a list of pixels, but it seems to be... so just take the mean
        ci_mean = ci_int.mean(axis=2)  # shape (97, 1331) → average over 12 pixels

        # Transpose to (time, neurons)
        ci_mean = ci_mean.T  # shape (1331, 97)

        flu_ts = TimeSeries(
            name="mean_fluorescence",
            data=ci_mean,
            unit="a.u.",
            rate=imaging_rate,
            description="Mean calcium intensity per neuron, averaged over ROI pixels"
        )
        nwbfile.add_acquisition(flu_ts)

        # === Add 3D Tracking Data ===
        calcium_imaging_module = nwbfile.create_processing_module(
            name='CalciumActivity',
            description='Calcium time series metadata, segmentation, and fluorescence data'
        )
        points = f["points"][:]  # shape (1331, 98, 3)
        if DEBUG:
            points = points[:10, ...]
        position_module = Position(name="NeuronCentroids")

        for neuron_idx in tqdm(range(points.shape[1]), desc="Formatting neuron positions..."):
            neuron_trace = points[:, neuron_idx, :]  # (1331, 3)

            position_module.add_spatial_series(SpatialSeries(
                name=f"neuron_{(neuron_idx+1):03d}",
                data=neuron_trace,
                unit="micrometers",
                reference_frame="Lab frame",
                description=f"3D position of neuron {neuron_idx}",
                timestamps=np.arange(neuron_trace.shape[0])
            ))

        calcium_imaging_module.add(position_module)

        # Also add the mapping to segmentation ID (segmentation is done below)
        df_tracking = load_per_neuron_position(position_module)

        dt = calc_simple_segmentation_mapping_table(df_tracking)
        calcium_imaging_module.add(dt)

        # Tranpose channel to be last
        print(f"Detected video with frame shape: {f['0/frame'].shape}")
        frame_shape = f["0/frame"].shape
        frame_shape = frame_shape[1:] + (frame_shape[0], )

        nn_keys = []
        for key in f.keys():
            if key.isdigit():
                nn_keys.append(int(key))
        num_frames = np.array(nn_keys).max() + 1    
        if DEBUG:
            num_frames = 10
        series_shape = (num_frames, ) + frame_shape
        
        # Build metadata objects
        grid_spacing = (0.45, 0.45, 1.75)
        device = nwbfile.create_device(name=device_name)
        CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
        # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
        nwbfile.add_imaging_plane(CalcImagingVolume)

        imvol_dask = dask_stack_volumes([da.from_delayed(load_frame(input_path, i), shape=frame_shape, dtype=np.uint16) for i in range(num_frames)])
        imvol_dask = imvol_dask.compute()
        # Check for nan and inf
        if np.isnan(imvol_dask).any() or np.isinf(imvol_dask).any():
            raise ValueError("Input data contains NaN or Inf values")

        chunk_video = (1,) + imvol_dask.shape[1:-1] + (1,)
        video_data = H5DataIO(
            data=CustomDataChunkIterator(array=imvol_dask, chunk_shape=chunk_video, display_progress=True),
            compression="gzip"
        )

        nwbfile.add_acquisition(MultiChannelVolumeSeries(
            name="CalciumImageSeries",
            description="Series of calcium imaging data",
            comments="Calcium imaging data from Harvard lab",
            data=video_data,  # data here should be series of indexed masks
            # Elements below can be kept the same as the CalciumImageSeries defined above
            device=device,
            unit="Voxel gray counts",
            scan_line_rate=2995.,
            dimension=series_shape,  
            resolution=1.,
            # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
            rate=imaging_rate,  # sampling rate in hz
            imaging_volume=CalcImagingVolume,
        ))

        # Calculate segmentation using simple watershed
        seg_dask = segment_from_centroids_using_watershed(points, imvol_dask, DEBUG=DEBUG)
        if eager_segmentation_mode:
            print(f"Eager segmentation mode enabled; computing segmentation in memory; estimated size: {seg_dask.nbytes / (1024**3):.2f} GB")
            seg_dask = seg_dask.compute()
            # Check for nan or inf
            if np.isnan(seg_dask).any() or np.isinf(seg_dask).any():
                raise ValueError("Segmentation data contains NaN or Inf values")

        chunk_seg = (1,) + frame_shape[:-1]  # chunk along time only
        print(f"Segmentations will be stored with chunk size {chunk_seg} and size {seg_dask.shape}")
        seg_data = H5DataIO(
            data=CustomDataChunkIterator(array=seg_dask, chunk_shape=chunk_seg, display_progress=True),
            compression="gzip"
        )
        calcium_imaging_module.add(MultiChannelVolumeSeries(
            name="CalciumSeriesSegmentation",
            description="Series of indexed masks associated with calcium segmentation",
            comments="Segmentation masks for calcium imaging data from Harvard lab, generated via watershed",
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

        # === Write NWB file ===
        with NWBHDF5IO(output_path, "w") as io:
            io.write(nwbfile)

    end_time = time.time()
    print(f"NWB file written to {output_path} (total time: {end_time - start_time:.2f} seconds)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Harvard data to NWB format.")
    parser.add_argument('--input_path', type=str, required=True, help='Full dataset as an .h5 file')
    parser.add_argument('--output_path', type=str, required=False, help='Output NWB file path')
    parser.add_argument('--session_description', type=str, default='Harvard Lab Data', help='Session description')
    parser.add_argument('--identifier', type=str, default='samuel_001', help='NWB file identifier')
    parser.add_argument('--device_name', type=str, default='HarvardMicroscope', help='Device name')
    parser.add_argument('--imaging_rate', type=float, default=10.0, help='Imaging rate (Hz)')
    parser.add_argument('--eager_segmentation_mode', action='store_true', help='Instead of lazy segmentation, compute segmentation eagerly in memory (may be much faster)')
    parser.add_argument('--use_processes', action='store_true', help='Instead of threads for dask, use processes (may be much faster for watershed step)')
    parser.add_argument('--debug', action='store_true', help='If set, only convert the first 10 time points')

    args = parser.parse_args()
    # Start Dask dashboard for visualization
    client = Client(processes=args.use_processes, n_workers= 4)# if args.eager_segmentation_mode else 16)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    print("If running on a remote computer, you may need to set up SSH port forwarding to access the dashboard in your browser.")
    print("For example, run the following command on your local machine:")
    print("  ssh -N -L 8787:localhost:8787 <your-remote-username>@<remote-host>")
    print("Then open http://localhost:8787 in your local web browser.")

    convert_harvard_to_nwb(
        input_path=args.input_path,
        output_path=args.output_path,
        session_description=args.session_description,
        identifier=args.identifier,
        device_name=args.device_name,
        imaging_rate=args.imaging_rate,
        eager_segmentation_mode=args.eager_segmentation_mode,
        DEBUG=args.debug
    )


import time
import h5py
import numpy as np
import dask.array as da
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.behavior import Position, SpatialSeries
from ndx_multichannel_volume import MultiChannelVolumeSeries
from datetime import datetime
from dateutil.tz import tzlocal
from tqdm.auto import tqdm
from wbfm.utils.nwb.utils_nwb_export import (build_optical_channel_objects,
                                             calc_simple_segmentation_mapping_table,
                                             load_per_neuron_position,
                                             CustomDataChunkIterator,
                                             dask_stack_volumes,
                                             segment_from_centroids_using_watershed)
from pynwb import TimeSeries
from wbfm.utils.general.utils_filenames import add_name_suffix
from pathlib import Path


def add_segmentation_to_nwb(input_path, 
                            output_path=None,
                            session_description="Leifer",
                            identifier="Leifer",
                            device_name="Leifer",
                            imaging_rate=6.0,
                            eager_segmentation_mode=False,
                            DEBUG=False):
    """Reads a nwb file and adds segmentation data based on 3D tracking data (centroids)."""

    start_time = time.time()

    # === USER PARAMETERS ===
    if output_path is None:
        output_path = Path(input_path).with_suffix('.nwb')
        output_path = add_name_suffix(output_path, '_with_segmentation')

    with NWBHDF5IO(input_path, mode='r+', load_namespaces=True) as nwb_io:
        if isinstance(nwb_io, NWBFile):
            print('NWB file loaded successfully')
            nwbfile = nwb_io
        else:
            nwbfile = nwb_io.read()

        # Also add the mapping to segmentation ID (segmentation is done below)
        calcium_imaging_module = nwbfile.processing['CalciumActivity']
        position_module = calcium_imaging_module['NeuronCentroids']

        # Load centroid data
        df_tracking = load_per_neuron_position(position_module)
        # Reshape centroid from from multi-index columns (neurons; rows are time) to 3D array of shape (time, neurons, 3)
        points = df_tracking.loc[:, (slice(None), ('z', 'x', 'y'))].to_numpy().reshape((len(df_tracking), len(df_tracking.columns.levels[0]), 3))

        # dt = calc_simple_segmentation_mapping_table(df_tracking)
        # calcium_imaging_module.add(dt)

        # Build metadata objects

        CalcImagingVolume = nwbfile.acquisition['CalciumImageSeries'].imaging_volume
        # print(CalcImagingVolume)
        # for name, acq in CalcImagingVolume.__dict__.items():
        #     print(f"- {name}: {type(acq).__name__}")
        #     if hasattr(acq, 'data'):
        #         try:
        #             shape = acq.data.shape
        #             print(f"  Shape: {shape}")
        #         except AttributeError:
        #             print("  Shape: Not available")
        # return
        # grid_spacing = (0.3, 0.3, 1.5)  # TODO
        # device = nwbfile.create_device(name=device_name)
        # CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
        # # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
        # nwbfile.add_imaging_plane(CalcImagingVolume)

        # Load data which will be used for segmentation
        dat = nwbfile.acquisition['CalciumImageSeries'].data
        chunks = (1, ) + dat.shape[1:-1] + (1,)

        imvol_dask = da.from_array(dat, chunks=chunks)[..., 0]
        imvol_dask = imvol_dask.compute()
        # Check for nan and inf
        if np.isnan(imvol_dask).any() or np.isinf(imvol_dask).any():
            raise ValueError("Input data contains NaN or Inf values")

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
        io.write(nwbfile)

    end_time = time.time()
    print(f"NWB file written to {output_path} (total time: {end_time - start_time:.2f} seconds)")
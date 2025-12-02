import os
import zarr
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.external.utils_zarr import zip_raw_data_zarr
import dask.array as da
import argparse


def unpack_nwb_to_project_structure(project_dir):
    """
    Unpack an NWB file into the expected on-disk project structure for the pipeline.
    
    Usage:
        unpack_nwb_to_project_structure("/path/to/project", "/path/to/project/nwb/yourfile.nwb")

    Parameters
    ----------
    project_dir : str or Path
        Path to the root of the project (should contain nwb/ with the .nwb file).
    nwb_path : str or Path
        Path to the .nwb file.
    """
    # Load the project directory and try find the NWB file
    project_data = ProjectData.load_final_project_data(project_dir, allow_hybrid_loading=True)
    cfg = project_data.project_config
    nwb_cfg = cfg.get_nwb_config()
    nwb_path = nwb_cfg.resolve_relative_path_from_config("nwb_filename")
    if nwb_path is None or not os.path.exists(nwb_path):
        raise FileNotFoundError(f"Expected NWB file at {nwb_path}; otherwise, please provide the nwb_path argument.")

    # Write preprocessed videos as zarr
    if project_data.red_data is not None and project_data.green_data is not None:
        project_data.logger.info("Writing preprocessed videos as zarr")
        # Ensure the preprocessed directory exists
        preproc_cfg = cfg.get_preprocessing_config()
        preproc_dir = preproc_cfg.absolute_subfolder
        if not os.path.exists(preproc_dir):
            raise FileNotFoundError(f"Expected preprocessed directory at {preproc_dir}")
        # These don't have a default name, so make one
        red_zarr_path = os.path.join(preproc_dir, "preprocessed_red.zarr")
        green_zarr_path = os.path.join(preproc_dir, "preprocessed_green.zarr")
        # Ensure the directories exist
        os.makedirs(red_zarr_path, exist_ok=True)
        os.makedirs(green_zarr_path, exist_ok=True)
        # Check to see if these are dask arrays; if so, convert to zarr using dask's save functionality
        chunks = (1,) + project_data.red_data.shape[1:]  # Assuming the first dimension is time or frames
        if isinstance(project_data.red_data, da.Array):
            project_data.logger.info("Data is a dask array; saving as zarr using dask's save functionality.")
            project_data.red_data.to_zarr(red_zarr_path, overwrite=True, compute=True)
            project_data.green_data.to_zarr(green_zarr_path, overwrite=True, compute=True)
        else:
            project_data.logger.info("Data is not a dask array; saving as zarr using zarr's save_array.")   
            zarr.save_array(red_zarr_path, project_data.red_data, chunks=chunks)
            zarr.save_array(green_zarr_path, project_data.green_data, chunks=chunks)
        # Then zip these folders
        red_zarr_zip_path = zip_raw_data_zarr(red_zarr_path)
        green_zarr_zip_path = zip_raw_data_zarr(green_zarr_path)
        # Update the config file with these paths; this is actually the main config
        preproc_cfg.config['preprocessed_red_fname'] = str(preproc_cfg.unresolve_absolute_path(red_zarr_zip_path))
        preproc_cfg.config['preprocessed_green_fname'] = str(preproc_cfg.unresolve_absolute_path(green_zarr_zip_path))
        preproc_cfg.update_self_on_disk()
    else:
        project_data.logger.info("No preprocessed video data found in the NWB file.")

    # Write segmentation as zarr
    if project_data.raw_segmentation is not None:
        segment_cfg = cfg.get_segmentation_config()
        seg_dir = segment_cfg.absolute_subfolder
        if not os.path.exists(seg_dir):
            raise FileNotFoundError(f"Expected segmentation directory at {seg_dir}")
        
        seg_zarr_path = segment_cfg.resolve_relative_path_from_config("output_masks")
        os.makedirs(os.path.dirname(seg_zarr_path), exist_ok=True)
        # Save the raw segmentation as zarr, but check for dask
        if isinstance(project_data.raw_segmentation, da.Array):
            project_data.logger.info("Raw segmentation is a dask array; saving as zarr using dask's save functionality.")
            da.to_zarr(project_data.raw_segmentation, seg_zarr_path, overwrite=True)
        else:
            zarr.save_array(seg_zarr_path, project_data.raw_segmentation, chunks=(1,)+project_data.segmentation.shape[1:])

        # Update the config with the segmentation path
        segment_cfg.config['segmentation_fname'] = str(seg_zarr_path)
        segment_cfg.update_self_on_disk()
    else:
        project_data.logger.info("No raw segmentation data found in the NWB file.")

    # Write final segmentation, if available
    if project_data.segmentation is not None:
        traces_cfg = cfg.get_traces_config()
        final_seg_dir = traces_cfg.absolute_subfolder
        if not os.path.exists(final_seg_dir):
            raise FileNotFoundError(f"Expected final segmentation directory at {final_seg_dir}")
        reindexed_masks_path = traces_cfg.resolve_relative_path_from_config("reindexed_masks")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(reindexed_masks_path), exist_ok=True)
        # Save the reindexed masks as zarr
        if isinstance(project_data.segmentation, da.Array):
            project_data.logger.info("Reindexed segmentation is a dask array; saving as zarr using dask's save functionality.")
            da.to_zarr(project_data.segmentation, reindexed_masks_path, overwrite=True)
        else:
            zarr.save_array(reindexed_masks_path, project_data.segmentation, chunks=(1,)+project_data.segmentation.shape[1:])
        reindexed_masks_path_zip = zip_raw_data_zarr(reindexed_masks_path)
        # Update the config with the reindexed masks path
        traces_cfg.config['reindexed_masks'] = str(cfg.unresolve_absolute_path(reindexed_masks_path_zip))
        traces_cfg.update_self_on_disk()
    else:
        project_data.logger.info("No final segmentation data found in the NWB file.")

    # # Write traces as h5
    # traces_dir = Path(project_dir) / "3-traces"
    # traces_dir.mkdir(exist_ok=True)
    # red_traces_path = traces_dir / "red_traces.h5"
    # green_traces_path = traces_dir / "green_traces.h5"
    # project_data.red_traces.to_hdf(red_traces_path, key="traces")
    # project_data.green_traces.to_hdf(green_traces_path, key="traces")

    # # Write segmentation metadata if available
    # if project_data.segmentation_metadata is not None:
    #     meta_path = seg_dir / "segmentation_metadata.pkl"
    #     project_data.segmentation_metadata.save(meta_path)

    # Reload the project data to ensure all paths are updated
    project_data.logger.info("NWB unpacking complete. See project structure below for details.")
    project_data = ProjectData.load_final_project_data_from_config(project_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Flavell data to NWB format.")
    parser.add_argument('--project_dir', type=str, required=True, help='Base directory containing project')
    parser.add_argument('--nwb_path', type=str, required=False, help='NWB file path')
    parser.add_argument('--debug', action='store_true', help='')

    args = parser.parse_args()

    unpack_nwb_to_project_structure(
        project_dir=args.project_dir,
        nwb_path=args.nwb_path,
    )

import concurrent
import logging
import os
from os import path as osp
from pathlib import Path
from shutil import copytree

import numpy as np
import tifffile
import zarr

from wbfm.utils.external.utils_zarr import zip_raw_data_zarr
from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings, \
    preprocess_all_frames_using_config, background_subtract_single_channel
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from wbfm.utils.projects.utils_filenames import get_sequential_filename, resolve_mounted_path_in_current_os, \
    add_name_suffix, get_location_of_new_project_defaults, get_bigtiff_fname_from_folder, \
    get_both_bigtiff_fnames_from_parent_folder
from wbfm.utils.projects.utils_project import get_project_name, edit_config, safe_cd


def build_project_structure_from_config(_config: dict, logger: logging.Logger) -> None:
    parent_folder = _config['project_dir']
    rel_dir_name = get_project_name(_config)

    # Build copied folder structure
    abs_dir_name = osp.join(parent_folder, rel_dir_name)
    abs_dir_name = get_sequential_filename(abs_dir_name)
    logger.info(f"Building new project at: {abs_dir_name}")

    # If the user just passed the parent raw data folder, then convert that into green and red
    parent_data_folder = _config.get('parent_data_folder', None)
    green_bigtiff_fname, red_bigtiff_fname = \
        _config.get('green_bigtiff_fname', None), _config.get('red_bigtiff_fname', None)
    if parent_data_folder is not None:
        green_bigtiff_fname, red_bigtiff_fname = get_both_bigtiff_fnames_from_parent_folder(parent_data_folder)

    if green_bigtiff_fname is None and _config.get('green_bigtiff_fname', None) is None:
        search_failed = True
    elif red_bigtiff_fname is None and _config.get('red_bigtiff_fname', None) is None:
        search_failed = True
    else:
        search_failed = False

    if search_failed:
        logging.warning(f"Failed to find bigtiff files in folder {parent_data_folder}")
        raise FileNotFoundError("Must pass either a) bigtiff data file directly, or b) proper parent folder")
    else:
        _config['red_bigtiff_fname'] = red_bigtiff_fname
        _config['green_bigtiff_fname'] = green_bigtiff_fname

    # Uses the pip installed package location
    src = get_location_of_new_project_defaults()
    copytree(src, abs_dir_name)

    # Update the copied project config with the new dest folder
    dest_fname = 'project_config.yaml'
    project_fname = osp.join(abs_dir_name, dest_fname)
    project_fname = Path(project_fname).resolve()

    edit_config(str(project_fname), _config)

    # Also update the snakemake file with the project directory
    snakemake_fname = osp.join(abs_dir_name, 'snakemake', 'config.yaml')
    snakemake_updates = {'project_dir': abs_dir_name}
    edit_config(snakemake_fname, snakemake_updates)


def write_data_subset_using_config(cfg: ModularProjectConfig,
                                   out_fname: str = None,
                                   video_fname: str = None,
                                   tiff_not_zarr: bool = True,
                                   pad_to_align_with_original: bool = False,
                                   save_fname_in_red_not_green: bool = None,
                                   use_preprocessed_data: bool = False,
                                   preprocessing_settings: PreprocessingSettings = None,
                                   which_channel: str = None,
                                   DEBUG: bool = False) -> None:
    """Takes the original giant .btf file from and writes the subset of the data as zarr or tiff"""

    out_fname, preprocessing_settings, project_dir, bigtiff_start_volume, verbose, video_fname = _unpack_config_for_data_subset(
        cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr, use_preprocessed_data,
        video_fname)

    with safe_cd(project_dir):
        preprocessed_dat, _ = preprocess_all_frames_using_config(DEBUG, cfg, verbose, video_fname,
                                                                 preprocessing_settings, None, which_channel,
                                                                 out_fname)

    if not pad_to_align_with_original and bigtiff_start_volume > 0:
        # i.e. remove the unpreprocessed data, creating an offset between the bigtiff and the zarr
        preprocessed_dat = preprocessed_dat[bigtiff_start_volume:, ...]
        # Resave the video; otherwise the old data isn't actually removed
        chunks = (1, ) + preprocessed_dat.shape[1:]
        zarr.save_array(out_fname, preprocessed_dat, chunks=chunks)
        cfg.logger.info(f"Removing {bigtiff_start_volume} unprocessed volumes")
    cfg.logger.info(f"Writing array of size: {preprocessed_dat.shape}")

    if tiff_not_zarr:
        # Have to add a color channel to make format: TZCYX
        # Imagej seems to expect this weird format
        out_dat = np.expand_dims(preprocessed_dat, 2).astype('uint16')
        tifffile.imwrite(out_fname, out_dat, imagej=True, metadata={'axes': 'TZCYX'})

    # Save this name in the config file itself
    if save_fname_in_red_not_green is not None:
        if save_fname_in_red_not_green:
            edits = {'preprocessed_red': out_fname}
        else:
            edits = {'preprocessed_green': out_fname}
        cfg.config.update(edits)
        cfg.update_self_on_disk()


def _unpack_config_for_data_subset(cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr,
                                   use_preprocessed_data, video_fname):
    verbose = cfg.config['verbose']
    project_dir = cfg.project_dir
    # preprocessing_fname = os.path.join('1-segmentation', 'preprocessing_config.yaml')
    if use_preprocessed_data:
        preprocessing_settings = None
        if verbose >= 1:
            print("Reusing already preprocessed data")
    elif preprocessing_settings is None:
        preprocessing_settings = PreprocessingSettings.load_from_config(cfg)
        # preprocessing_fname = cfg.config['preprocessing_config']
        # preprocessing_settings = PreprocessingSettings.load_from_yaml(preprocessing_fname)
    if out_fname is None:
        if tiff_not_zarr:
            out_fname = os.path.join(project_dir, "data_subset.tiff")
        else:
            out_fname = os.path.join(project_dir, "data_subset.zarr")
    else:
        out_fname = os.path.join(project_dir, out_fname)
    if video_fname is None:
        if save_fname_in_red_not_green:
            if not use_preprocessed_data:
                video_fname = cfg.config['red_bigtiff_fname']
            else:
                video_fname = cfg.resolve_relative_path_from_config('preprocessed_red')
        else:
            if not use_preprocessed_data:
                video_fname = cfg.config['green_bigtiff_fname']
            else:
                video_fname = cfg.resolve_relative_path_from_config('preprocessed_green')
        video_fname = resolve_mounted_path_in_current_os(video_fname, verbose=0)
    start_volume = cfg.config['dataset_params'].get('bigtiff_start_volume', None)
    if start_volume is None:
        logging.warning("Did not find bigtiff_start_volume; is this an old style project?")
        logging.warning("Using start volume of 0. If this is fine, then no changes are needed")
        start_volume = 0
        cfg.config['dataset_params']['bigtiff_start_volume'] = 0  # Will be written to disk later
    return out_fname, preprocessing_settings, project_dir, start_volume, verbose, video_fname


def crop_zarr_using_config(cfg: ModularProjectConfig):

    fields = ['preprocessed_red', 'preprocessed_green']
    to_crop = [cfg.config[f] for f in fields]
    start_volume = cfg.config['dataset_params']['start_volume']
    num_frames = cfg.config['dataset_params']['num_frames']
    end_volume = start_volume + num_frames

    new_fnames = []
    for fname in to_crop:
        this_vid = zarr.open(fname)
        new_vid = this_vid[start_volume:end_volume, ...]
        new_fname = add_name_suffix(fname, f'-num_frames{num_frames}')
        new_fnames.append(new_fname)
        logging.info(f"Saving original file {fname} with new name {new_fname}")

        zarr.save_array(new_fname, new_vid, chunks=this_vid.chunks)

    # Also update config file
    for field, name in zip(fields, new_fnames):
        cfg.config[field] = str(name)
    cfg.config['dataset_params']['start_volume'] = 0
    cfg.config['dataset_params']['bigtiff_start_volume'] = start_volume

    cfg.update_self_on_disk()


def zip_zarr_using_config(project_cfg: ModularProjectConfig):
    project_cfg.logger.info("Zipping zarr data (both channels)")
    out_fname_red_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_red'], verbose=1)
    out_fname_green_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_green'], verbose=1)

    project_cfg.config['preprocessed_red'] = str(project_cfg.unresolve_absolute_path(out_fname_red_7z))
    project_cfg.config['preprocessed_green'] = str(project_cfg.unresolve_absolute_path(out_fname_green_7z))
    project_cfg.update_self_on_disk()


def subtract_background_using_config(cfg: ModularProjectConfig, do_preprocessing=True, DEBUG=False):
    """
    Read a video of the background and the otherwise fully preprocessed data, and simply subtract

    NOTE: if z-alignment (rotation) is used, then this can cause some artifacts
    """

    preprocessing_settings = PreprocessingSettings.load_from_config(cfg)
    num_slices = preprocessing_settings.raw_number_of_planes
    num_frames = 50  # TODO: is this constant?
    if DEBUG:
        num_frames = 2

    opt = dict(num_frames=num_frames, num_slices=num_slices, preprocessing_settings=preprocessing_settings,
               DEBUG=DEBUG)
    if not do_preprocessing:
        opt['preprocessing_settings'] = None
    raw_fname_red = cfg.config[f'preprocessed_red']
    background_fname_red = cfg.config[f'red_background_fname']
    raw_fname_green = cfg.config[f'preprocessed_green']
    background_fname_green = cfg.config[f'green_background_fname']

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        red_fname_subtracted = ex.submit(background_subtract_single_channel, raw_fname_red, background_fname_red,
                                         **opt).result()
        green_fname_subtracted = ex.submit(background_subtract_single_channel, raw_fname_green, background_fname_green,
                                           **opt).result()
    cfg.config['preprocessed_red'] = str(red_fname_subtracted)
    cfg.config['preprocessed_green'] = str(green_fname_subtracted)

    zip_zarr_using_config(cfg)

import concurrent
import logging
import os
import shutil
from os import path as osp
from pathlib import Path
from shutil import copytree
import numpy as np
import tifffile
import zarr
from PIL import Image, UnidentifiedImageError
from wbfm.utils.external.utils_zarr import zip_raw_data_zarr
from wbfm.utils.general.preprocessing.bounding_boxes import generate_legacy_bbox_fname, \
    calculate_bounding_boxes_from_cfg_and_save
from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings, \
    preprocess_all_frames_using_config, background_subtract_single_channel
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.general.utils_filenames import error_if_dot_in_name

from wbfm.utils.general.utils_filenames import get_sequential_filename, add_name_suffix, \
    get_location_of_new_project_defaults, get_both_bigtiff_fnames_from_parent_folder, \
    get_ndtiff_fnames_from_parent_folder, generate_output_data_names
from wbfm.utils.projects.utils_project import get_relative_project_name, safe_cd, update_project_config_path, \
    update_snakemake_config_path, update_nwb_config_path


def build_project_structure_from_config(config: dict, logger: logging.Logger = None) -> None:
    """
    Builds a project from passed user data, which determines:
        The location and name of the new project
        The raw data files it points to

    By default, the folder name of the project will read the date of the original data, and include that as a prefix

    Parameters
    ----------
    config: dict with the following:
        Raw data information, meaning either of the following:
            parent_data_folder - the folder containing the raw data
            green_bigtiff_fname AND red_bigtiff_fname - the individual channel bigtiff files
        Project location, i.e.:
            project_dir - the parent folder where the new project will be created
    logger

    Returns
    -------

    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # If the user just passed the parent raw data folder, then convert that into green and red
    parent_data_folder = config.get('parent_data_folder', None)
    green_fname, red_fname = \
        config.get('green_bigtiff_fname', None), config.get('red_bigtiff_fname', None)
    # First try for the new format: ndtiff
    is_btf = False
    if parent_data_folder is not None:
        green_fname, red_fname = get_ndtiff_fnames_from_parent_folder(parent_data_folder)
        search_failed = _check_if_search_succeeded(config, green_fname, red_fname)

        if search_failed:
            green_fname, red_fname = get_both_bigtiff_fnames_from_parent_folder(parent_data_folder)
            is_btf = True
    search_failed = _check_if_search_succeeded(config, green_fname, red_fname)

    if search_failed:
        logging.warning(f"Failed to find raw files in folder {parent_data_folder}")
        raise FileNotFoundError("Must pass either a) bigtiff data file directly, or "
                                "b) proper parent folder with bigtiffs or ndtiffs in it.")
    else:
        # Convert to absolute paths if not already
        if not osp.isabs(green_fname):
            green_fname = osp.join(parent_data_folder, green_fname)
        if not osp.isabs(red_fname):
            red_fname = osp.join(parent_data_folder, red_fname)

        if is_btf:
            logging.warning("Found bigtiff files, which are deprecated.")
            config['green_bigtiff_fname'] = green_fname
            config['red_bigtiff_fname'] = red_fname
        else:
            config['red_fname'] = red_fname
            config['green_fname'] = green_fname

    # Check to make sure there are no '.' characters
    for key in ['red_fname', 'green_fname', 'red_bigtiff_fname', 'green_bigtiff_fname']:
        error_if_dot_in_name(config.get(key, ''))

    # Build the full project name using the date the data was taken
    basename = Path(red_fname).name.split('_')[0]
    project_config_updates = config

    project_fname, _ = build_project_structure(project_config_updates, basename)

    # Copy simple raw data files to the project; for now just stage_position
    from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
    stage_position_filename = WormFullVideoPosture.find_stage_position_in_folder(parent_data_folder)
    if stage_position_filename is not None:
        project_config = ModularProjectConfig(project_fname)
        beh_folder = project_config.get_behavior_config().absolute_subfolder
        # Copy just this file
        shutil.copy(stage_position_filename, osp.join(beh_folder, Path(stage_position_filename).name))

    # If there is a neuropal dataset to add, do so
    if 'neuropal_path' in config:
        neuropal_path = config['neuropal_path']
        copy_data = config.get('copy_neuropal_data', True)
        from wbfm.utils.projects.utils_neuropal import add_neuropal_to_project
        add_neuropal_to_project(project_fname, neuropal_path, copy_data=copy_data)

    return project_fname


def get_absolute_project_name(basename, project_config_updates):
    parent_folder = project_config_updates['project_dir']
    experimenter = project_config_updates.get('experimenter', '')
    task = project_config_updates.get('task_name', '')
    rel_new_project_name = get_relative_project_name(basename, experimenter=experimenter, task=task)
    abs_new_project_name = osp.join(parent_folder, rel_new_project_name)
    abs_new_project_name = get_sequential_filename(abs_new_project_name)
    abs_new_project_name = str(Path(abs_new_project_name).resolve())
    return abs_new_project_name


def build_project_structure(project_config_updates, basename=None):
    """project_config_updates must at least have project_dir as a string"""
    project_folder_abs = get_absolute_project_name(basename, project_config_updates)
    # Uses the pip installed package location
    src = get_location_of_new_project_defaults()
    copytree(src, project_folder_abs)
    # Update the copied project config with the new dest folder
    project_fname = update_project_config_path(project_folder_abs, project_config_updates)
    # Also update the snakemake file with the project directory
    update_snakemake_config_path(project_folder_abs)
    return project_fname, project_folder_abs


def build_project_structure_from_nwb_file(config, nwb_file, copy_nwb_file=False):
    """
    This mostly just copies the empty project structure and then copies (or moves) the nwb into it
    """
    project_fname, project_folder_abs = build_project_structure(config, basename=str(Path(nwb_file).stem))

    # Move or copy the nwb file
    target_nwb_filename_rel = os.path.join('nwb', Path(nwb_file).name)
    target_nwb_filename_abs = os.path.join(project_folder_abs, target_nwb_filename_rel)
    # For some reason shutil.move gives a later error, so just copy
    shutil.copy(nwb_file, target_nwb_filename_abs)
    if not copy_nwb_file:
        os.remove(nwb_file)

    # Update the config file
    target_nwb_filename_in_config = target_nwb_filename_rel if copy_nwb_file else target_nwb_filename_abs
    update_nwb_config_path(project_folder_abs, target_nwb_filename_in_config)

    return project_fname


def _check_if_search_succeeded(_config, green_fname, red_fname):
    if green_fname is None and _config.get('green_bigtiff_fname', None) is None \
            and _config.get('green_fname', None) is None:
        search_failed = True
    elif red_fname is None and _config.get('red_bigtiff_fname', None) is None \
            and _config.get('red_fname', None) is None:
        search_failed = True
    else:
        search_failed = False
    return search_failed


def calculate_number_of_volumes_from_tiff_file(num_raw_slices, red_bigtiff_fname):
    logging.warning("Detecting number of total frames in the video, may take ~30 seconds."
                    " Note: this should not be needed for ndtiff videos, only bigtiffs (deprecated).")
    try:
        full_video = Image.open(red_bigtiff_fname)
        num_2d_frames = full_video.n_frames
    except UnidentifiedImageError:
        full_video = tifffile.TiffFile(red_bigtiff_fname)
        num_2d_frames = len(full_video.pages)
    num_volumes = num_2d_frames / num_raw_slices
    # This should be an integer
    if num_volumes % 1 != 0:
        logging.warning(f"Number of frames {num_2d_frames} is not an integer multiple of num_slices "
                        f"{num_raw_slices}... this may be a problem. "
                        f"Continuing by removing the fractional volume")
    return num_volumes


def write_data_subset_using_config(cfg: ModularProjectConfig,
                                   out_fname: str = None,
                                   tiff_not_zarr: bool = False,
                                   pad_to_align_with_original: bool = False,
                                   save_fname_in_red_not_green: bool = None,
                                   use_preprocessed_data: bool = False,
                                   preprocessing_settings: PreprocessingSettings = None,
                                   which_channel: str = None,
                                   DEBUG: bool = False) -> None:
    """
    Takes the original giant .btf file from and writes the subset of the data (or full dataset) as zarr or tiff

    Parameters
    ----------
    cfg: config class
    out_fname: output filename. Should end in .zarr for zarr
    tiff_not_zarr: flag for output format
    pad_to_align_with_original: flag for behavior if bigtiff_start_volume > 0, i.e. frames are removed at the beginning
    save_fname_in_red_not_green: where to save the out_fname in the config file
    use_preprocessed_data: flag for using already preprocessed data
    preprocessing_settings: class with preprocessing settings. Can be loaded from cfg
    which_channel: green or red
    DEBUG

    Returns
    -------

    """

    out_fname, preprocessing_settings, project_dir, bigtiff_start_volume, verbose = _unpack_config_for_data_subset(
        cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr, use_preprocessed_data)

    with safe_cd(project_dir):
        preprocessed_dat = preprocess_all_frames_using_config(cfg, preprocessing_settings, None,
                                                              which_channel, out_fname, verbose, DEBUG)

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
            edits = {'preprocessed_red_fname': out_fname}
        else:
            edits = {'preprocessed_green_fname': out_fname}
        preprocessing_settings.cfg_preprocessing.config.update(edits)
        preprocessing_settings.cfg_preprocessing.update_self_on_disk()


def _unpack_config_for_data_subset(cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr,
                                   use_preprocessed_data):
    verbose = cfg.config['verbose']
    project_dir = cfg.project_dir
    # preprocessing_fname = os.path.join('1-segmentation', 'preprocessing_config.yaml')
    if use_preprocessed_data:
        preprocessing_settings = None
        if verbose >= 1:
            print("Reusing already preprocessed data")
    elif preprocessing_settings is None:
        preprocessing_settings = cfg.get_preprocessing_class()
        # preprocessing_fname = cfg.config['preprocessing_config']
        # preprocessing_settings = PreprocessingSettings.load_from_yaml(preprocessing_fname)
    if out_fname is None:
        if tiff_not_zarr:
            out_fname = os.path.join(project_dir, "data_subset.tiff")
        else:
            out_fname = os.path.join(project_dir, "data_subset.zarr")
    else:
        out_fname = os.path.join(project_dir, out_fname)
    params = cfg.config.get('deprecated_dataset_params', {})
    start_volume = params.get('bigtiff_start_volume', None)
    if start_volume is None:
        start_volume = 0
        if 'deprecated_dataset_params' not in cfg.config:
            cfg.config['deprecated_dataset_params'] = {}
        cfg.config['deprecated_dataset_params']['bigtiff_start_volume'] = 0  # Will be written to disk later
    else:
        logging.warning("Found a start_volume, but this is deprecated. Attempting to continue, but may not work.")
    return out_fname, preprocessing_settings, project_dir, start_volume, verbose


def crop_zarr_using_config(cfg: ModularProjectConfig):

    preprocessing_class = cfg.get_preprocessing_class()
    to_crop = [preprocessing_class.get_path_to_preprocessed_data(red_not_green=True),
               preprocessing_class.get_path_to_preprocessed_data(red_not_green=False)]
    params = cfg.config.get('deprecated_dataset_params', {})
    start_volume = params.get('start_volume', 0)
    num_frames = params.get('num_frames', None)
    if num_frames is None:
        raise ValueError("Must pass number of frames to crop")
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
    if 'deprecated_dataset_params' not in cfg.config:
        cfg.config['deprecated_dataset_params'] = {}
    cfg.config['deprecated_dataset_params']['start_volume'] = 0
    cfg.config['deprecated_dataset_params']['bigtiff_start_volume'] = start_volume

    cfg.update_self_on_disk()


def zip_zarr_using_config(preprocessing_class: PreprocessingSettings):
    preprocessing_class.cfg_preprocessing.logger.info("Zipping zarr data (both channels)")
    out_fname_red_7z = zip_raw_data_zarr(preprocessing_class.get_path_to_preprocessed_data(red_not_green=True),
                                         verbose=1)
    out_fname_green_7z = zip_raw_data_zarr(preprocessing_class.get_path_to_preprocessed_data(red_not_green=False),
                                           verbose=1)

    cfg = preprocessing_class.cfg_preprocessing
    cfg.config['preprocessed_red_fname'] = str(cfg.unresolve_absolute_path(out_fname_red_7z))
    cfg.config['preprocessed_green_fname'] = str(cfg.unresolve_absolute_path(out_fname_green_7z))
    cfg.update_self_on_disk()


def subtract_background_using_config(cfg: ModularProjectConfig, do_preprocessing=True, DEBUG=False):
    """
    Read a video of the background and the otherwise fully preprocessed data, and simply subtract

    NOTE: if z-alignment (rotation) is used, then this can cause some artifacts
    """

    preprocessing_settings = cfg.get_preprocessing_class()
    num_slices = preprocessing_settings.raw_number_of_planes
    num_frames = 50
    if DEBUG:
        num_frames = 2

    opt = dict(num_frames=num_frames, num_slices=num_slices, preprocessing_settings=preprocessing_settings,
               DEBUG=DEBUG)
    if not do_preprocessing:
        opt['preprocessing_settings'] = None
    raw_fname_red = preprocessing_settings.get_path_to_preprocessed_data(red_not_green=True)
    background_fname_red = preprocessing_settings.cfg_preprocessing.config[f'background_fname_red']
    raw_fname_green = preprocessing_settings.get_path_to_preprocessed_data(red_not_green=False)
    background_fname_green = preprocessing_settings.cfg_preprocessing.config[f'background_fname_green']

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        red_fname_subtracted = ex.submit(background_subtract_single_channel, raw_fname_red, background_fname_red,
                                         **opt).result()
        green_fname_subtracted = ex.submit(background_subtract_single_channel, raw_fname_green, background_fname_green,
                                           **opt).result()
    preprocessing_settings.cfg_preprocessing.config['preprocessed_red_fname'] = str(red_fname_subtracted)
    preprocessing_settings.cfg_preprocessing.config['preprocessed_green_fname'] = str(green_fname_subtracted)

    zip_zarr_using_config(preprocessing_settings)


def calculate_total_number_of_frames_from_bigtiff(cfg):
    cfg.logger.warning("Deprecated function: calculate_total_number_of_frames_from_bigtiff")
    num_frames = cfg.num_frames
    if num_frames is None:
        # Check the number of total frames in the video, and update the parameter
        # Note: requires correct value of num_slices
        num_raw_slices = cfg.num_slices
        red_bigtiff_fname = cfg.config['red_bigtiff_fname']
        num_volumes = calculate_number_of_volumes_from_tiff_file(num_raw_slices, red_bigtiff_fname)
        num_frames = int(num_volumes)
        cfg.logger.debug(f"Calculated number of frames: {num_frames}")
        if 'deprecated_dataset_params' not in cfg.config:
            cfg.config['deprecated_dataset_params'] = {}
        cfg.config['deprecated_dataset_params']['num_frames'] = num_frames
        cfg.update_self_on_disk()


def preprocess_fluorescence_data(cfg, to_zip_zarr_using_7z, DEBUG=False):
    # Load the project, to make sure even if raw data is in a weird format it works
    from wbfm.utils.projects.finished_project_data import ProjectData

    project_data = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True)
    cfg = project_data.project_config

    options = {'tiff_not_zarr': False,
               'pad_to_align_with_original': False,
               'use_preprocessed_data': False,
               'DEBUG': DEBUG}
    logger = cfg.logger
    project_dir = cfg.project_dir
    cfg.config['project_dir'] = project_dir
    preprocessing_cfg = cfg.get_preprocessing_config()
    red_output_fname, green_output_fname = generate_output_data_names(cfg)
    # Open the raw data using the config file directly
    _, is_btf = cfg.get_raw_data_fname(red_not_green=True)
    if is_btf:
        calculate_total_number_of_frames_from_bigtiff(cfg)
    bbox_fname = preprocessing_cfg.config.get('bounding_boxes_fname', None)
    if bbox_fname is None:
        generate_legacy_bbox_fname(project_dir)
    with safe_cd(project_dir):
        preprocessing_settings = cfg.get_preprocessing_class()

        # Very first: calculate the alignment between the red and green channels (camera misalignment)
        preprocessing_settings.calculate_warp_mat(cfg)
        green_name = Path(green_output_fname)
        fname = green_name.parent / (green_name.stem + "_camera_alignment.pickle")
        preprocessing_settings.path_to_camera_alignment_matrix = fname

        # Second: within-stack alignment using the red channel, which will be saved to disk
        options['out_fname'] = red_output_fname
        options['save_fname_in_red_not_green'] = True
        # Location: same as the preprocessed red channel (possibly not the bigtiff)
        red_name = Path(options['out_fname'])
        fname = red_name.parent / (red_name.stem + "_preprocessed.pickle")
        preprocessing_settings.path_to_previous_warp_matrices = fname

        if Path(options['out_fname']).exists() and fname.exists():
            logger.info("Preprocessed red already exists; skipping to green")
        else:
            logger.info("Preprocessing red...")
            preprocessing_settings.do_mirroring = False
            assert preprocessing_settings.to_save_warp_matrices
            write_data_subset_using_config(cfg, preprocessing_settings=preprocessing_settings,
                                           which_channel='red', **options)

        # Third the green channel will read the warp matrices per-volume (step 2) and between cameras (step 1)
        logger.info("Preprocessing green...")
        options['out_fname'] = green_output_fname
        options['save_fname_in_red_not_green'] = False
        preprocessing_settings.to_use_previous_warp_matrices = True
        if cfg.config['dataset_params']['red_and_green_mirrored']:
            preprocessing_settings.do_mirroring = True
        write_data_subset_using_config(cfg, preprocessing_settings=preprocessing_settings,
                                       which_channel='green', **options)

        # Save the warp matrices (camera and per-volume) to disk if needed further
        preprocessing_settings.save_all_warp_matrices()

        # Also saving bounding boxes for future segmentation (speeds up and dramatically reduces false positives)
        Path(bbox_fname).parent.mkdir(parents=True, exist_ok=True)
        calculate_bounding_boxes_from_cfg_and_save(cfg, bbox_fname, red_not_green=True)

        bbox_fname = preprocessing_cfg.unresolve_absolute_path(bbox_fname)
        preprocessing_cfg.config['bounding_boxes_fname'] = bbox_fname
        preprocessing_cfg.update_self_on_disk()
    if to_zip_zarr_using_7z:
        # Reload the config file to make sure all filenames are correct
        preprocessing_settings = cfg.get_preprocessing_class()
        zip_zarr_using_config(preprocessing_settings)
    logger.info("Finished.")

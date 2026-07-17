import os
from pathlib import Path

from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.external.utils_yaml import load_config


def segment_local_data_subset(project_config, out_fname=None):
    """
    Segments a dataset that has been copied locally; assumed to be named 'data_subset.tif'

    Applies NO preprocessing; assumes that is done with the subset already

    See also: write_data_subset_from_config
    """
    from wbfm.utils.segmentation.util.utils_pipeline import _segment_full_video_3d, _segment_full_video_2d

    cfg = load_config(project_config)
    project_dir = Path(project_config).parent

    with safe_cd(project_dir):
        segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])

    if out_fname is None:
        out_fname = "masks_subset.zarr"
    mask_fname = os.path.join("1-segmentation", out_fname)
    metadata_fname = os.path.join("1-segmentation", "metadata_subset.pickle")

    video_path = "data_subset.tiff"
    verbose = cfg['verbose']
    num_slices = cfg['dataset_params']['num_slices']
    num_frames = cfg['dataset_params']['num_frames']
    preprocessing_settings = None
    frame_list = list(range(num_frames))
    metadata = {}

    model_type = segment_cfg['segmentation_type']
    if model_type == '3d':
        stardist_model_name = "charlie_3d"
        with safe_cd(project_dir):
            _segment_full_video_3d(cfg, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                                   preprocessing_settings, stardist_model_name, verbose, video_path)
    else:
        stardist_model_name = segment_cfg['segmentation_params']['stardist_model_name']
        opt_postprocessing = segment_cfg['postprocessing_params']
        with safe_cd(project_dir):
            _segment_full_video_2d(cfg, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                                   opt_postprocessing, preprocessing_settings, stardist_model_name, verbose, video_path)

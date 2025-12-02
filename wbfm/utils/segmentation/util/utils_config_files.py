import logging
import pickle

from wbfm.utils.general.utils_filenames import pickle_load_binary

from wbfm.utils.segmentation.util.utils_paths import get_output_fnames


def _unpack_config_file(preprocessing_cfg, segment_cfg, project_cfg, DEBUG):
    # Initializing variables
    if DEBUG:
        num_frames = 1
    video_path = project_cfg.get_preprocessing_class().get_path_to_preprocessed_data(red_not_green=True)
    # Generate new filenames if they are not set
    mask_fname = segment_cfg.config['output_masks']
    metadata_fname = segment_cfg.config['output_metadata']
    output_dir = segment_cfg.config['output_folder']
    mask_fname, metadata_fname = get_output_fnames(video_path, output_dir, mask_fname, metadata_fname)
    metadata_fname = segment_cfg.unresolve_absolute_path(metadata_fname)
    mask_fname = segment_cfg.unresolve_absolute_path(mask_fname)
    # Save settings
    segment_cfg.config['output_masks'] = mask_fname
    segment_cfg.config['output_metadata'] = metadata_fname
    verbose = project_cfg.config['verbose']
    stardist_model_name = segment_cfg.config['segmentation_params']['stardist_model_name']
    zero_out_borders = segment_cfg.config['segmentation_params']['zero_out_borders']
    # Preprocessing information
    bbox_fname = preprocessing_cfg.resolve_relative_path_from_config('bounding_boxes_fname')
    if bbox_fname is not None:
        all_bounding_boxes = pickle_load_binary(bbox_fname)
        project_cfg.logger.info(f"Found bounding boxes at: {bbox_fname}")
    else:
        all_bounding_boxes = None
        project_cfg.logger.warning(f"Did not find bounding boxes at: {bbox_fname},"
                                   f"a large number of false positive segmentations might be generated.")
    sum_red_and_green_channels = segment_cfg.config['segmentation_params'].get('sum_red_and_green_channels', False)
    segment_on_green_channel = project_cfg.config['dataset_params'].get('segment_and_track_on_green_channel', False)
    if sum_red_and_green_channels:
        if segment_on_green_channel:
            project_cfg.logger.warning("segment_on_green_channel and sum_red_and_green_channels are both true; "
                                       "ignoring sum_red_and_green_channels")
            sum_red_and_green_channels = False
        else:
            project_cfg.logger.warning("Summing red and green channels for segmentation; does not affect metadata.")

    return (mask_fname, metadata_fname, stardist_model_name, verbose, video_path,
            zero_out_borders, all_bounding_boxes, sum_red_and_green_channels, segment_on_green_channel)

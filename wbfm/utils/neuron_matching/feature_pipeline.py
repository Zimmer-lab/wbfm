import concurrent.futures
import numpy as np
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons
from tqdm.auto import tqdm

from barlow_track.utils.utils_tracking import get_target_size_from_args
from wbfm.utils.neuron_matching.class_frame_pair import FramePair, calc_FramePair_from_FeatureSpaceTemplates, calc_FramePair_from_Frames, \
    FramePairOptions
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame, \
    build_reference_frame_encoding
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


##
## Full traces function
##

def match_all_adjacent_frames(all_frame_dict, start_volume, end_volume, frame_pair_options: FramePairOptions, use_tracker_class=False):
    all_frame_pairs = {}
    frame_range = range(start_volume, end_volume - 1)
    for i_frame in tqdm(frame_range):
        key = (i_frame, i_frame + 1)
        frame0, frame1 = all_frame_dict[key[0]], all_frame_dict[key[1]]
        if use_tracker_class:
            this_pair = calc_FramePair_from_FeatureSpaceTemplates(frame0, frame1, frame_pair_options=frame_pair_options)
        else:
            this_pair = calc_FramePair_from_Frames(frame0, frame1, frame_pair_options=frame_pair_options)
        all_frame_pairs[key] = this_pair
    return all_frame_pairs


def calculate_frame_objects_full_video(video_data, frame_range, video_fname,
                                       z_depth_neuron_encoding=None, encoder_opt=None, max_workers=8,
                                       preprocessing_settings=None, 
                                       use_barlow_network=False, project_data=None,
                                       logger=None, **kwargs):
    # Get initial volume; settings are same for all
    vol_shape = video_data[0, ...].shape
    all_detected_neurons = project_data.segmentation_metadata

    if use_barlow_network:
        # Load the network
        from barlow_track.utils.barlow import load_barlow_model
        network_path = encoder_opt.get('network_path', None)
        del encoder_opt['network_path']
        gpu, model, args = load_barlow_model(network_path)
        encoder_opt['gpu'] = gpu
        encoder_opt['model'] = model

        # Load the neuron-crop generator
        from barlow_track.utils.barlow import NeuronImageWithGTDataset
        target_sz = get_target_size_from_args(args)
        num_frames = project_data.num_frames
        dataset = NeuronImageWithGTDataset(project_data, num_frames, target_sz, include_untracked=True)
        encoder_opt['dataset'] = dataset

    # Make sure the preprocessing is ready (if prior steps are from nwb it may not be)
    if not preprocessing_settings.alpha_is_ready:
        preprocessing_settings.calculate_alpha_from_data(video_data)

    def _build_frame(frame_ind: int) -> ReferenceFrame:
        metadata = {'frame_ind': frame_ind,
                    'vol_shape': vol_shape,
                    'video_fname': video_fname,
                    'z_depth': z_depth_neuron_encoding,
                    'alpha_red': preprocessing_settings.alpha_red,
                    '_raw_data': np.array(video_data[frame_ind, ...])}
        f = build_reference_frame_encoding(metadata=metadata, all_detected_neurons=all_detected_neurons,
                                           encoder_opt=encoder_opt, use_barlow_network=use_barlow_network)
        return f

    # Build all frames initially, then match
    all_frame_dict = dict()
    # logger.info(f"Calculating Frame objects for frames: {frame_range[0]} to {frame_range[-1]}")
    with tqdm(total=len(frame_range)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_build_frame, i): i for i in frame_range}
            for future in concurrent.futures.as_completed(futures):
                i_frame = futures[future]
                all_frame_dict[i_frame] = future.result()
                pbar.update(1)
    return all_frame_dict

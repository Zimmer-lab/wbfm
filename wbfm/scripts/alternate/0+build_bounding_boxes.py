"""
"""

# main function
import os

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None,
              DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    bounding_box_fname = os.path.join(cfg.project_dir, '1-segmentation', 'bounding_boxes.pickle')
    segment_cfg = cfg.get_segmentation_config()

    num_frames = cfg.config['dataset_params']['num_frames']


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    video_fname = _config['cfg'].config['preprocessed_red']
    bbox_fname = _config['bounding_box_fname']
    num_frames = _config['num_frames']
    calculate_bounding_boxes_from_fnames(video_fname, bbox_fname, num_frames)

    segment_cfg = _config['segment_cfg']
    bbox_fname = segment_cfg.unresolve_absolute_path(bbox_fname)
    segment_cfg.config['bbox_fname'] = bbox_fname
    segment_cfg.update_self_on_disk()

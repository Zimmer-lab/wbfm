"""
"""

# main function
from multiprocessing import allow_connection_pickling
import os

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_cfg_and_save
from wbfm.utils.projects.finished_project_data import ProjectData
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
    bounding_box_fname = os.path.join(cfg.project_dir, 'dat', 'bounding_boxes.pickle')
    segment_cfg = cfg.get_segmentation_config()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    cfg = _config['cfg']

    bbox_fname = _config['bounding_box_fname']
    calculate_bounding_boxes_from_cfg_and_save(cfg, bbox_fname)

    segment_cfg = _config['segment_cfg']
    bbox_fname = segment_cfg.unresolve_absolute_path(bbox_fname)
    segment_cfg.config['bbox_fname'] = bbox_fname
    segment_cfg.update_self_on_disk()

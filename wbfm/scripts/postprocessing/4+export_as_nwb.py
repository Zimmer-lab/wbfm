"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""

import os
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_project import safe_cd
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.segmentation.util.utils_pipeline import segment_video_using_config_2d, segment_video_using_config_3d

from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, output_folder=None, include_image_data=True, DEBUG=False)

from wbfm.utils.nwb.utils_nwb_export import nwb_using_project_data


@ex.config
def cfg(project_path, DEBUG):
    pass


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project = ProjectData.load_final_project_data(_config['project_path'], allow_hybrid_loading=True)
    nwb_using_project_data(project, include_image_data=_config['include_image_data'], output_folder=_config['output_folder'])

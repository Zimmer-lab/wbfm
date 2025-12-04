"""
"""

# main function
import os

# Experiment tracking
import sacred

from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_neuropal import add_neuropal_to_project, segment_neuropal_from_project
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, raw_neuropal_path=None, copy_data=True, also_segment=True, subsample_in_z=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_path = _config['project_path']
    raw_neuropal_path = _config['raw_neuropal_path']
    copy_data = _config['copy_data']

    add_neuropal_to_project(project_path, raw_neuropal_path, copy_data=copy_data)

    if _config['also_segment']:
        subsample_in_z = _config['subsample_in_z']
        segment_neuropal_from_project(project_path, subsample_in_z)
        
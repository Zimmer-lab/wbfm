"""
"""

# main function
import os

# Experiment tracking
import sacred

from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.nwb.convert_leifer_data import add_segmentation_to_nwb
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.projects.utils_neuropal import add_neuropal_to_project
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(nwb_path=None, DEBUG=False)


@ex.config
def cfg(nwb_path, DEBUG):
    # Manually load yaml files

    pass
    # if not DEBUG:
    #     using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    nwb_path = _config['nwb_path']
    add_segmentation_to_nwb(nwb_path, DEBUG=_config['DEBUG'])


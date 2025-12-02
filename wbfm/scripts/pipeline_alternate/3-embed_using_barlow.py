"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred
from barlow_track.utils.track_using_barlow import embed_using_barlow_from_config
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, model_fname=None, results_subfolder=None, allow_hybrid_loading=True, use_projection_space=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    # check_all_needed_data_for_step(cfg, 2)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_cfg = _config['cfg']
    model_fname = _config['model_fname']
    results_subfolder = _config['results_subfolder']

    embed_using_barlow_from_config(project_cfg, model_fname, results_subfolder, 
                                   allow_hybrid_loading=_config['allow_hybrid_loading'],
                                   use_projection_space=_config['use_projection_space'],
                                   DEBUG=DEBUG)

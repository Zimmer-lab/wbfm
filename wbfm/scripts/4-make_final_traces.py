"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.pipeline.traces import full_step_4_make_traces_from_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, allow_only_global_tracker=False, match_using_indices=True, allow_hybrid_loading=True, DEBUG=False)


@ex.config
def cfg(project_path, allow_only_global_tracker, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_4.log')

    # check_all_needed_data_for_step(cfg, 4, training_data_required=False)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_cfg = _config['cfg']
    allow_only_global_tracker = _config['allow_only_global_tracker']

    full_step_4_make_traces_from_config(project_cfg, allow_only_global_tracker, 
                                        allow_hybrid_loading=_config['allow_hybrid_loading'], 
                                        match_using_indices=_config['match_using_indices'],
                                        DEBUG=DEBUG)

"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred

from wbfm.pipeline.traces import full_step_4_make_traces_from_config
from wbfm.pipeline.tracking import match_tracks_and_tracklets_using_config
from wbfm.pipeline.tracklets import consolidate_tracklets_using_config
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
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_after_manual_annotation.log')
    # check_all_needed_data_for_step(cfg, 4, training_data_required=False)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    cfg = _config['cfg']

    # Only do an update of the manually annotated tracklets, not the entire matching pipeline
    track_config = cfg.get_tracking_config()
    track_config.config['final_3d_postprocessing']['only_use_previous_matches'] = True
    track_config.update_self_on_disk()

    # 3b
    match_tracks_and_tracklets_using_config(cfg, DEBUG=DEBUG)

    training_cfg = cfg.get_training_config()
    z_threshold = training_cfg.config['pairwise_matching_params']['z_threshold']
    consolidate_tracklets_using_config(cfg, z_threshold=z_threshold, DEBUG=DEBUG)

    # 4
    full_step_4_make_traces_from_config(cfg, DEBUG)

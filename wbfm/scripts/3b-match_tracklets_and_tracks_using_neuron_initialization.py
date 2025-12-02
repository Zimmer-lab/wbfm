"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.pipeline.tracklets import consolidate_tracklets_using_config
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.pipeline.tracking import match_tracks_and_tracklets_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_3b.log')

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run, _log):
    sacred.commands.print_config(_run)
    cfg = _config['cfg']

    DEBUG = _config['DEBUG']
    match_tracks_and_tracklets_using_config(cfg, DEBUG=DEBUG)

    training_cfg = cfg.get_training_config()
    z_threshold = training_cfg.config['pairwise_matching_params'].get('z_threshold', 2.5)
    consolidate_tracklets_using_config(cfg, z_threshold=z_threshold, DEBUG=DEBUG)

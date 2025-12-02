# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.pipeline.tracking import track_using_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb

from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step

cgitb.enable(format='text')
from wbfm.utils.projects.utils_project import safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, out_fname=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_3a.log')
    project_dir = cfg.project_dir

    check_all_needed_data_for_step(cfg, 2)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_cfg = _config['cfg']
    project_dir = _config['project_dir']

    with safe_cd(project_dir):
        tracklet_cfg = project_cfg.get_training_config()
        use_barlow_network = tracklet_cfg.config['tracker_params'].get('use_barlow_network', False)

        track_using_using_config(project_cfg, use_superglue_tracker=not use_barlow_network, DEBUG=_config['DEBUG'])

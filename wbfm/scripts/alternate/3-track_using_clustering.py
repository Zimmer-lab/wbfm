# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.nn_utils.fdnc_predict import track_using_fdnc_from_config


cgitb.enable(format='text')
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    check_all_needed_data_for_step(cfg, 3)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_cfg = _config['cfg']

    track_using_fdnc_from_config(project_cfg, _config['DEBUG'])

"""
main
"""

# Experiment tracking
import logging
import os

import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.visualization.plot_traces import make_default_summary_plots_using_config

from wbfm.pipeline.traces import extract_traces_using_config
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    traces_cfg = cfg.get_traces_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    # Set environment variables to (try to) deal with rare blosc decompression errors
    os.environ["BLOSC_NOLOCK"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"

    DEBUG = _config['DEBUG']
    project_cfg = _config['cfg']

    with safe_cd(_config['project_dir']):
        # Reads masks from disk, and writes traces
        project_data = ProjectData.load_final_project_data(project_cfg, allow_hybrid_loading=True)
        extract_traces_using_config(project_data, name_mode='neuron', DEBUG=DEBUG)

        # By default make some visualizations
        # Note: reloads the project data
        logging.info("Making default grid plots")
        project_data = ProjectData.load_final_project_data(project_cfg, allow_hybrid_loading=True)
        make_grid_plot_from_project(project_data, channel_mode='all', calculation_mode='integration')

        # By default make some visualizations
        project_data = ProjectData.load_final_project_data(project_cfg, allow_hybrid_loading=True)
        make_default_summary_plots_using_config(project_data)

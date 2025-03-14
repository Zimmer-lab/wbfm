"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.plot_traces import make_default_triggered_average_plots, \
    make_pirouette_split_triggered_average_plots, make_summary_hilbert_triggered_average_grid_plot, \
    make_fwd_and_turn_triggered_average_plots

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None)


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    # Load the project to speed up the trace calculations
    project_data = ProjectData.load_final_project_data_from_config(_config['project_path'])

    # Reversal and forward, and two in one
    make_default_triggered_average_plots(project_data)
    make_fwd_and_turn_triggered_average_plots(project_data, turn_state=BehaviorCodes.VENTRAL_TURN)
    make_fwd_and_turn_triggered_average_plots(project_data, turn_state=BehaviorCodes.DORSAL_TURN)

    # Hilbert phase
    make_summary_hilbert_triggered_average_grid_plot(project_data)
    # make_summary_hilbert_triggered_average_grid_plot(project_data, return_fast_scale_separation=True)
    make_summary_hilbert_triggered_average_grid_plot(project_data, residual_mode='pca', interpolate_nan=True)
    make_summary_hilbert_triggered_average_grid_plot(project_data, residual_mode='pca_global', interpolate_nan=True)

    # make_pirouette_split_triggered_average_plots(project_data)

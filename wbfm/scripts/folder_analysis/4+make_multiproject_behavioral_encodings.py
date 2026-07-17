"""
main
"""
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
# Experiment tracking
import sacred
from sacred import Experiment
from sklearn.model_selection import KFold

from wbfm.utils.projects.finished_project_data import load_all_projects_in_folder
# main function
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToUnivariateEncoding
from wbfm.utils.visualization.multiproject_wrappers import MultiProjectBehaviorPlotterWithBehavior
from wbfm.utils.general.utils_hardcoded import load_paper_datasets, get_summary_visualization_dir

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, cv_options='all')


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    folder_path = _config['project_path']
    if folder_path.lower() in ["gcamp", "gfp"]:
        # Save in common folder, but make a subfolder
        all_projects = load_paper_datasets(folder_path)
        output_folder = Path(get_summary_visualization_dir())
        output_folder = output_folder.joinpath(Path(folder_path).name)
        output_folder.mkdir(exist_ok=True)
    else:
        all_projects = load_all_projects_in_folder(folder_path)
        output_folder = Path(folder_path)

    behavior_plotter = MultiProjectBehaviorPlotterWithBehavior(all_projects=all_projects,
                                                               class_constructor=NeuronToUnivariateEncoding)
    plot_opt = dict(df_name='ratio', to_save=True, saving_folder=output_folder)

    # See _unpack_data_from_name for valid values
    y_train_values = ['signed_speed', 'abs_speed', 'signed_speed_smoothed', 'signed_speed_angular']
    # y_train_values = ['signed_speed', 'abs_speed', 'summed_curvature', 'pirouette', 'signed_speed_smoothed']

    # Test different options for cross validation
    # all_cv_types = [partial(KFold, n_splits=3), TimeSeriesSplit, RollingOriginForwardValidation]
    # all_cv_names = ["KFold", "TimeSeriesSplit", "RollingOriginForwardValidation"]
    all_cv_types = [partial(KFold, n_splits=3), partial(KFold, n_splits=10, shuffle=True)]
    # all_cv_names = ["KFold", "KFold_shuffle"]

    # New: residuals!
    # all_cv_names = ["KFold_residual", "KFold_shuffle_residual"]
    # behavior_plotter.set_for_all_classes({'use_residual_traces': True})

    if _config['cv_options'] == 'all':
        cv_to_check = all_cv_types
        cv_names = all_cv_names
    else:
        raise NotImplementedError

    for name, cv in zip(cv_names, cv_to_check):
        this_output_folder = Path(f"{str(output_folder)}_{name}")
        this_output_folder.mkdir(exist_ok=True)
        print(f"Saving all output in folder: {this_output_folder}")
        behavior_plotter.set_for_all_classes({'cv_factory': cv})
        plot_opt['saving_folder'] = this_output_folder

        for y_train_name in y_train_values:
            behavior_plotter.paired_boxplot_overall_multi_dataset('ratio', y_train=y_train_name)
            fname = this_output_folder.joinpath(f"encoding_{y_train_name}.png")
            plt.savefig(fname)

            plot_opt['y_train'] = y_train_name
            behavior_plotter.plot_model_prediction(use_multineuron=True, **plot_opt)
            plt.close('all')
            behavior_plotter.plot_model_prediction(use_multineuron=False, **plot_opt)
            plt.close('all')
            behavior_plotter.plot_sorted_correlations(**plot_opt)
            plt.close('all')

        print(f"Finished! Check output in folder: {this_output_folder}")

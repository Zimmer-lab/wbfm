
# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.utils_project import make_project_like

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, target_directory=None, steps_to_keep=None, DEBUG=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    target_directory = _config['target_directory']
    project_path = _config['project_path']
    steps_to_keep = _config['steps_to_keep']

    make_project_like(project_path, target_directory=target_directory, steps_to_keep=steps_to_keep, verbose=1)

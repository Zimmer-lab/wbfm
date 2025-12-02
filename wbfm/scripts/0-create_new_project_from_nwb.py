import os

import sacred
from sacred import Experiment
from sacred import SETTINGS
import cgitb
from wbfm.utils.external.custom_errors import MissingAnalysisError

cgitb.enable(format='text')
from wbfm.pipeline.project_initialization import build_project_structure_from_nwb_file
from wbfm.utils.nwb.utils_nwb_unpack import unpack_nwb_to_project_structure
from wbfm.utils.segmentation.util.utils_metadata import recalculate_metadata_from_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
ex.add_config(project_dir=None, nwb_file=None, copy_nwb_file=True, unpack_nwb=True)

@ex.config
def cfg(project_dir, nwb_file, copy_nwb_file):
    pass


@ex.automain
def main(_config, _run, _log):
    """
    Example:
    python 0a-create_new_project_from_nwb.py with
        nwb_file='/path/to/your/data.nwb'
        project_dir='/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms'

    See also wbfm/scripts/0a-create_new_project.py
    """
    sacred.commands.print_config(_run)

    project_fname = build_project_structure_from_nwb_file(_config, _config['nwb_file'], _config['copy_nwb_file'])

    if _config['unpack_nwb']:
        unpack_nwb_to_project_structure(project_fname)
        try:
            recalculate_metadata_from_config(project_fname, name_mode='neuron', allow_hybrid_loading=True)
        except MissingAnalysisError:
            _log.warning("Could not recalculate metadata after unpacking NWB file; this may be because segmentation has not yet been run.")

    
    print(f"Successfully created new project at: {project_fname}")

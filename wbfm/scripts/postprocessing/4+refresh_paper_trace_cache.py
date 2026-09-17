"""
Refresh stale paper-trace disk caches (.cache/paper_traces*.h5).

Background: the NWB exporter renames raw-trace neurons using the CURRENT manual
annotation (neuron_name_to_manual_id_mapping), but the paper traces themselves are
loaded from disk cache. If the annotation was updated after the cache was built
(e.g. neuron_019 -> AUAR with certainty 0), the cached columns keep the old names
and the export fails with KeyError('neuron_019'). In that case the underlying
.h5 files are self-consistent (red/green raw traces and red/green/ratio paper
traces each match); only the cache is stale relative to the annotation.

This script reports cache vs. annotation modification times, deletes the stale
caches, and recalculates them (same recalculation as 4+export_paper_traces.py).
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.pipeline.traces import calc_paper_traces_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, check_only=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    check_all_needed_data_for_step(cfg, 4, training_data_required=False)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


def _report_staleness(project_data):
    """Print mtimes of the manual annotation vs. the paper-trace caches."""
    import os
    annotation_fname = getattr(project_data, 'df_manual_tracking_fname', None)
    cache_fnames = project_data.data_cacher.list_of_paper_trace_methods(return_filenames=True)
    if annotation_fname is not None and os.path.exists(annotation_fname):
        print(f"Manual annotation: {annotation_fname} (mtime {os.path.getmtime(annotation_fname)})")
    else:
        print(f"Manual annotation not found (got {annotation_fname}); cannot check staleness")
        return
    stale = set(project_data.data_cacher.warn_if_caches_stale())
    for fname in cache_fnames:
        if fname is not None and os.path.exists(fname):
            marker = "STALE" if fname in stale else "ok"
            print(f"  [{marker}] {fname} (mtime {os.path.getmtime(fname)})")
        else:
            print(f"  [missing] {fname}")


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    check_only = _config['check_only']
    project_cfg = _config['cfg']
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    _report_staleness(project_data)

    # Delete stale caches (dry run if check_only or DEBUG)
    project_data.data_cacher.clear_disk_cache(dry_run=(check_only or DEBUG))

    if check_only or DEBUG:
        print("Check-only mode; not recalculating. Rerun with check_only=False to refresh.")
        return

    calc_paper_traces_using_config(project_data, DEBUG)

import logging
import os.path as osp
from tqdm.auto import tqdm

from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.external.custom_errors import AnalysisOutOfOrderError, IncompleteConfigFileError
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd


def _check_and_print(all_to_check: list, description: str, verbose: int):
    all_exist = all(map(osp.exists, all_to_check))
    if verbose >= 1:
        if all_exist:
            if verbose >= 2:
                print(f"Found all files ({description})")
        else:
            print(f"Did not find some necessary files: {all_to_check}")
    return all_exist


def check_all_needed_data_for_step(project_config: ModularProjectConfig,
                                   step_index: int,
                                   raise_error=True,
                                   training_data_required=True,
                                   verbose=1):
    if project_config is None or not project_config.has_valid_self_path:
        logging.warning("No project config provided; cannot check data")
        if raise_error:
            raise IncompleteConfigFileError("No project config provided; cannot check data")
        else:
            return True
    flag = True
    if step_index > 0:
        flag = check_preprocessed_data(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Preprocessing')
    if step_index > 1:
        flag = check_segmentation(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Segmentation')
    if step_index > 2:
        if training_data_required:
            flag = check_training_final(project_config, verbose)
        else:
            flag = True
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Training data')
    if step_index > 3:
        flag = check_tracking(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Tracking')
    if step_index > 4:
        flag = check_traces(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Traces')
    return flag


def check_preprocessed_data(project_config: ModularProjectConfig, verbose=0):

    try:
        p = project_config.get_preprocessing_class()
        all_to_check = [
            p.get_path_to_preprocessed_data(red_not_green=True),
            p.get_path_to_preprocessed_data(red_not_green=False)
        ]
        all_exist = _check_and_print(all_to_check, 'preprocessed data', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_segmentation(project_config: ModularProjectConfig, verbose=0):
    cfg_segment = project_config.get_segmentation_config()

    try:
        all_to_check = [
            cfg_segment.resolve_relative_path_from_config('output_masks'),
            cfg_segment.resolve_relative_path_from_config('output_metadata')
        ]
        all_exist = _check_and_print(all_to_check, 'segmentation', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_raw(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

    try:
        with safe_cd(cfg_training.project_dir):
            training_folder = '2-training_data'
            file_names = ['clust_df_dat.pickle', 'frame_dat.pickle', 'match_dat.pickle']
            all_to_check = map(lambda file: osp.join(training_folder, 'raw', file), file_names)
            all_exist = _check_and_print(all_to_check, 'raw training data', verbose)
            return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_only_tracklets(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

    try:
        all_to_check = [
            cfg_training.resolve_relative_path_from_config('df_3d_tracklets'),
        ]
        all_exist = _check_and_print(all_to_check, 'all tracklets', verbose)
        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_final(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

    try:
        all_to_check = [
            cfg_training.resolve_relative_path_from_config('df_3d_tracklets'),
            cfg_training.resolve_relative_path_from_config('df_training_3d_tracks'),
            cfg_training.resolve_relative_path_from_config('reindexed_masks'),
            cfg_training.resolve_relative_path_from_config('reindexed_metadata')
        ]
        all_exist = _check_and_print(all_to_check, 'final training data', verbose)
        return all_exist
    except (AssertionError, TypeError):
        return False


def check_tracking(project_config: ModularProjectConfig, verbose=0):
    tracking_cfg = project_config.get_tracking_config()

    try:
        all_to_check = [tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')]
        all_exist = _check_and_print(all_to_check, 'tracking', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_traces(project_config: ModularProjectConfig, verbose=0):
    try:
        with safe_cd(project_config.project_dir):
            traces_cfg = project_config.get_traces_config()
            file_names = ['all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            # file_names = ['reindexed_masks.zarr.zip', 'all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            make_full_name = lambda file: traces_cfg.resolve_relative_path(file, prepend_subfolder=True)
            all_to_check = list(map(make_full_name, file_names))
            all_exist = _check_and_print(all_to_check, 'traces', verbose)

            return all_exist
    except (AssertionError, TypeError):
        return False


def check_traces_and_segmentation_sanity(project_data, max_frames_to_sample: int = 20,
                                         check_traces: bool = True,
                                         check_segmentation: bool = True) -> list:
    """
    Return a list of user-facing problem strings; empty if the project looks sane.

    Designed to be called by GUIs (trace explorer, progress GUI) so that users get a
    clear error instead of a blank/missing trace display when segmentation failed.

    Parameters
    ----------
    project_data
        Duck-typed ProjectData-like object (only attributes are used, to avoid imports)
    max_frames_to_sample
        Number of frames to sample evenly when counting segmented objects
    check_traces
        If True, report when the traces table is empty/missing
    check_segmentation
        If True, inspect segmentation metadata for failed segmentation (no objects,
        or too many objects)

    Checks:
    - No traces found (empty/missing traces dataframe)
    - No objects found in segmentation metadata
    - Too many objects per frame (segmentation likely failed, e.g. unsuited dataset or
      parameters): object count saturates the configured max_number_of_objects, or is
      implausibly high even when no cap is configured
    - Most frames have zero detected objects (nearly-empty segmentation)
    """
    import numpy as np

    problems = []

    # --- No traces ---------------------------------------------------------
    if check_traces:
        red_traces = getattr(project_data, 'red_traces', None)
        try:
            n_traces = 0 if red_traces is None else int(red_traces.shape[1])
        except Exception:
            n_traces = 0
        if n_traces == 0:
            problems.append(
                "No traces found: the traces table is empty (or missing). "
                "This usually means segmentation or tracking failed, or the wrong project was loaded. "
                "Check that earlier pipeline steps completed and that segmentation parameters "
                "match this dataset."
            )

    # --- Object counts from segmentation metadata --------------------------
    if not check_segmentation:
        return problems

    seg_meta = getattr(project_data, 'segmentation_metadata', None)
    if seg_meta is None:
        return problems

    try:
        frames = list(seg_meta.which_frames)
    except Exception:
        return problems
    if not frames:
        problems.append(
            "No objects found in segmentation metadata: no frames contain detected objects. "
            "Segmentation likely failed; check the dataset and segmentation parameters "
            "(e.g. thresholds, max_number_of_objects) and re-run segmentation."
        )
        return problems

    # Sample frames evenly for speed
    if len(frames) > max_frames_to_sample:
        sample_idx = np.linspace(0, len(frames) - 1, max_frames_to_sample).astype(int)
        sampled_frames = [frames[i] for i in sample_idx]
    else:
        sampled_frames = frames

    counts = []
    for t in sampled_frames:
        try:
            counts.append(len(seg_meta.detect_neurons_from_file(int(t))))
        except Exception:
            continue
    if not counts:
        return problems

    max_count = int(max(counts))
    median_count = float(np.median(counts))

    # Configured cap, if available
    max_allowed = None
    try:
        cfg = getattr(project_data, 'project_config', None)
        seg_cfg = cfg.get_segmentation_config()
        max_allowed = seg_cfg.config.get('postprocessing_params', {}).get('max_number_of_objects')
    except Exception:
        max_allowed = None

    if max_count == 0:
        problems.append(
            "No objects found in segmentation for any sampled frame. "
            "Segmentation likely failed; check the dataset and segmentation parameters "
            "and re-run segmentation."
        )
    elif median_count == 0:
        problems.append(
            f"Segmentation found objects in only a few frames (max {max_count} objects in a "
            "sampled frame, but the median sampled frame is empty). "
            "Tracking/traces will be unreliable; consider re-running segmentation."
        )
    elif max_allowed is not None and max_count >= int(max_allowed):
        problems.append(
            f"Too many objects identified (segmentation likely failed): found up to {max_count} "
            f"objects per frame, reaching the configured max_number_of_objects={max_allowed}. "
            "This often indicates an unsuited dataset or imperfect segmentation parameters; "
            "adjust segmentation parameters and re-run segmentation before trusting traces."
        )
    elif max_allowed is None and max_count > 500:
        # Heuristic fallback when no cap is configured (a worm has ~100-200 neurons)
        problems.append(
            f"Too many objects identified (segmentation likely failed): found up to {max_count} "
            "objects per frame, which is implausibly high for a worm. "
            "This often indicates an unsuited dataset or imperfect segmentation parameters; "
            "adjust segmentation parameters and re-run segmentation before trusting traces."
        )

    return problems


def check_zarr_file_integrity(project_config: ModularProjectConfig, verbose=0):
    p = project_config.get_preprocessing_class()
    fnames = [p.get_path_to_preprocessed_data(red_not_green=True),
              p.get_path_to_preprocessed_data(red_not_green=False)]

    for fname in fnames:
        logging.info(f"Checking integrity of {fname}")
        z = zarr_reader_folder_or_zipstore(fname)

        for frame in tqdm(z, leave=False):
            tmp = frame.shape


def print_sacred_log(project_config: str) -> None:
    from sacred.observers import TinyDbReader
    project_config = ModularProjectConfig(project_config)

    reader = TinyDbReader(project_config.get_log_dir())
    results = reader.fetch_report(indices=-1)

    try:
        print(results[0])
    except KeyError:
        print("Key error in the log; this means a step is in progress or the log is corrupted")


def get_project_status(project_config: ModularProjectConfig, verbose=2):
    """
    Returns the index of the last step that was completed

    Parameters
    ----------
    project_config
    verbose

    Returns
    -------

    """
    opt = dict(project_config=project_config, training_data_required=False, raise_error=False)

    project_config.logger.info("Determining status of project...")
    i_step = 1
    for i_step in tqdm([1, 2, 3, 4, 5]):
        passed = check_all_needed_data_for_step(step_index=i_step, **opt)
        if not passed:
            if verbose >= 1:
                project_config.logger.info(f"==============================")
                project_config.logger.info(f"Next pipeline step required: {i_step-1}")
                project_config.logger.info(f"==============================")
            break
    else:
        if verbose >= 1:
            project_config.logger.info("All steps of project are complete; manual annotation can begin")

    # Return last completed step
    return i_step - 1

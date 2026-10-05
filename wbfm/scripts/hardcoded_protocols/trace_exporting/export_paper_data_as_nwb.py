"""
Export paper traces in NWB format.

Backends (--backend):
  - 'serial' (default): one dataset at a time. Always safe.
  - 'processes': one worker process per export. Safe: workers only share
    plain strings and each loads its own project.
  - 'threads': UNSAFE with the current export code and likely to produce
    corrupt outputs or crash. The export path calls os.chdir()
    (wbfm.utils.projects.utils_project.cd_to_dir), which is process-global,
    and this libhdf5 build is not threadsafe (read corruption, VOL errors,
    segfaults). Both failure modes were observed in practice (16 workers ->
    instant segfault; 4 workers -> silent data-read failures). This backend
    exists only for comparison and prints a warning before proceeding.

Task enumeration is fast: only project *paths* are loaded
(load_paper_datasets(..., only_load_paths=True)), and each task loads just
its own full project from its config path.

Two ways to run:
  1. One job, serial or N worker processes:
       python export_paper_data_as_nwb.py
       python export_paper_data_as_nwb.py --include_image_data --backend processes --num_workers 4
  2. SLURM job array, one dataset per task (see submit_export_array.py):
       python export_paper_data_as_nwb.py --include_image_data --backend serial --only_index $SLURM_ARRAY_TASK_ID
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.nwb.utils_nwb_export import get_nwb_export_fname_from_parts, nwb_using_project_data
from wbfm.utils.projects.finished_project_data import ProjectData


def get_parent_dir(include_image_data: bool) -> str:
    parent_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/nwb'
    return os.path.join(parent_dir, 'with_images' if include_image_data else 'no_images')


def build_tasks(all_suffixes, parent_dir, verbose=True):
    """
    Fast task enumeration: load project *paths* only, not full projects.

    Returns a deterministic list of (suffix, name, config_path, this_folder),
    all plain strings. config_path points at the project's project_config.yaml.
    """
    tasks = []
    loader = tqdm(all_suffixes, desc='Loading dataset paths') if verbose else all_suffixes
    for suffix in loader:
        subfolder_name = f'exported_data_{suffix}'
        this_folder = os.path.join(parent_dir, subfolder_name)
        Path(this_folder).mkdir(exist_ok=True)
        path_dict = load_paper_datasets(suffix, only_load_paths=True)
        for name in sorted(path_dict):
            tasks.append((suffix, name, str(path_dict[name]), this_folder))
    return tasks


def export_one_task(task):
    """
    Export a single project.

    `task` is a tuple of plain strings/bools only:
        (suffix, name, config_path, this_folder, include_image_data, skip_if_exists)
    so it is picklable under any multiprocessing start method. The full
    project is loaded here, from its config path.

    Returns (name, status, message) with status in {'exported', 'skipped', 'error'}.
    Thread-safety note: only safe in separate processes (or serially); see
    the module docstring for why threads corrupt outputs.
    """
    _, name, config_path, this_folder, include_image_data, skip_if_exists = task
    # Skip check first, from the name alone: no project loading needed, and the
    # filename cannot drift from the writer's (same helper builds both).
    output_fname = get_nwb_export_fname_from_parts(name, this_folder, include_image_data)
    if skip_if_exists and os.path.exists(output_fname):
        return name, 'skipped', f'Skipping {output_fname} because it already exists'

    try:
        project = ProjectData.load_final_project_data(config_path, verbose=0)
    except Exception as e:
        return name, 'error', f'Error loading {name}: {e}'

    try:
        print(f'Exporting {name} to {this_folder}', flush=True)
        nwb_using_project_data(project, include_image_data=include_image_data, output_folder=this_folder)
    except Exception as e:
        return name, 'error', f'Error exporting {name}: {e}'
    return name, 'exported', None


def report_and_exit(results):
    n_exported = sum(1 for _, s in results if s == 'exported')
    n_skipped = sum(1 for _, s in results if s == 'skipped')
    failed = sorted(name for name, s in results if s == 'error')
    print(f'Exported {n_exported}, skipped {n_skipped}, failed {len(failed)} of {len(results)} projects',
          flush=True)
    if failed:
        print('Failed projects:', flush=True)
        for name in failed:
            print(f'  {name}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(
        description='Export traces in nwb format',
        epilog='''
Examples:
  # Export all default suffixes (gfp, '', mutant, immob), serially
  python export_paper_data_as_nwb.py

  # Export specific suffixes only
  python export_paper_data_as_nwb.py --suffixes gfp mutant

  # Include image data in exports
  python export_paper_data_as_nwb.py --include_image_data

  # Parallel export with 4 worker processes and image data
  python export_paper_data_as_nwb.py --include_image_data --backend processes --num_workers 4

  # SLURM array: exactly one dataset, the one at this index (see submit_export_array.py)
  python export_paper_data_as_nwb.py --include_image_data --backend serial --only_index $SLURM_ARRAY_TASK_ID
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--include_image_data', action='store_true', help='Whether to include image data in the export')
    parser.add_argument('--delete_existing', action='store_true',
                        help='Whether to delete existing export files before exporting (default skips them)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: export a single project, then stop')
    parser.add_argument('--suffixes', nargs='+', default=['gfp', '', 'mutant', 'immob'],
                        help='Dataset suffixes to export')
    parser.add_argument('--backend', choices=['serial', 'processes', 'threads'], default='serial',
                        help="'serial' (default, safe) exports one dataset at a time. "
                             "'processes' runs each export in its own process. "
                             "'threads' shares one process and is UNSAFE with the current export code "
                             "(os.chdir races, non-threadsafe HDF5: expect corrupt outputs or segfaults).")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (processes backend only). '
                             'Note: each worker holds a full project in memory; with --include_image_data '
                             'keep this modest (2-4).')
    parser.add_argument('--only_index', type=int, default=None,
                        help='Export only the single task at this index of the task list and exit. '
                             'Intended for SLURM job arrays. Out-of-range indexes exit 0 (no-op) so the '
                             'array size may exceed the task count.')
    args = parser.parse_args()

    DEBUG = args.debug
    include_image_data = args.include_image_data
    skip_if_exists = not args.delete_existing

    tasks = build_tasks(args.suffixes, get_parent_dir(include_image_data))
    print(f'{len(tasks)} project(s) to export', flush=True)

    full_tasks = [(suffix, name, config_path, folder, include_image_data, skip_if_exists)
                  for suffix, name, config_path, folder in tasks]

    if DEBUG or args.only_index is not None:
        # Single-project modes: debug exports the first task, array mode the indexed one.
        idx = 0 if DEBUG else args.only_index
        if not 0 <= idx < len(full_tasks):
            if DEBUG:
                print('No tasks; nothing to do.', flush=True)
                raise SystemExit(1)
            print(f'--only_index {idx} out of range for {len(full_tasks)} tasks; nothing to do.',
                  flush=True)
            raise SystemExit(0)
        name, status, message = export_one_task(full_tasks[idx])
        if message is not None:
            print(message, flush=True)
        if DEBUG:
            print(f'Exported {name} to {full_tasks[idx][3]}, breaking', flush=True)
        raise SystemExit(0 if status in ('exported', 'skipped') else 1)

    if args.backend == 'serial' or args.num_workers <= 1:
        results = []
        for task in tqdm(full_tasks, desc='Exporting'):
            print("=" * 50, flush=True)
            name, status, message = export_one_task(task)
            if message is not None:
                print(message, flush=True)
            results.append((name, status))
        report_and_exit(results)

    elif args.backend == 'threads':
        print('WARNING: threaded export is UNSAFE with the current export code and will '
              'likely produce corrupt outputs or crash: os.chdir() is process-global '
              '(relative-path races) and this libhdf5 build is not threadsafe '
              '(read corruption, VOL errors, segfaults). Prefer --backend processes, '
              '--backend serial, or a SLURM job array. Continuing anyway in 5 seconds...',
              flush=True)
        import time
        time.sleep(5)
        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers, thread_name_prefix='nwb-export') as executor:
            futures = {executor.submit(export_one_task, t): t[1] for t in full_tasks}
            for future in tqdm(as_completed(futures), total=len(full_tasks), desc='Exporting'):
                name, status, message = future.result()
                if message is not None:
                    print(message, flush=True)
                results.append((name, status))
        report_and_exit(results)

    else:  # processes
        ctx_name = 'fork' if hasattr(os, 'fork') else None
        ctx = mp.get_context(ctx_name) if ctx_name else mp.get_context()
        print(f'Exporting with {args.num_workers} worker processes (start method: {ctx._name})', flush=True)
        results = []
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(export_one_task, t): t[1] for t in full_tasks}
            for future in tqdm(as_completed(futures), total=len(full_tasks), desc='Exporting'):
                name, status, message = future.result()
                if message is not None:
                    print(message, flush=True)
                results.append((name, status))
        report_and_exit(results)

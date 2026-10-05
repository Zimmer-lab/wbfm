"""
Parallel NWB export dispatcher (process-based).

Unlike the threaded dispatcher in export_paper_data_as_nwb.py, this script runs
each export in its own worker PROCESS. That matters because the export code as
written is not thread-safe:

  - wbfm.utils.projects.utils_project.cd_to_dir() calls os.chdir(), which is
    process-global. With threads, one worker can change the working directory
    while another resolves a relative path (e.g. '3-tracking/...'), producing
    bogus "No such file or directory" failures.
  - The libhdf5 build in use is not threadsafe. Concurrent HDF5 reads/writes
    from multiple threads corrupt reads ("Problems reading the array data",
    VOL-connector errors) and can segfault the whole interpreter (observed
    with 16 threads: instant core dump, zero outputs).

Observed consequences of threaded exports: 16 workers -> immediate
segmentation fault; 4 workers -> 3 silent data-read failures in the first
batch. Do not be surprised by corrupt outputs if you use threads anyway; the
'threads' backend below exists only for comparison and prints a warning.

Task enumeration is fast: only project *paths* are loaded
(load_paper_datasets(..., only_load_paths=True)), and each worker loads just
its own full project from its config path. Nothing unpicklable is passed
between processes, so this works with either the 'fork' or 'spawn' start
method.

Two ways to run:
  1. One job, N worker processes:
       python export_paper_data_as_nwb_parallel.py --include_image_data --num_workers 4
  2. SLURM job array, one dataset per task (see submit_export_array.py):
       python submit_export_array.py --include_image_data --max_concurrent 4
"""
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.nwb.utils_nwb_export import get_nwb_export_fname, nwb_using_project_data
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
    Export a single project. Runs in a worker process (or thread).

    `task` is a tuple of plain strings/bools only:
        (suffix, name, config_path, this_folder, include_image_data, skip_if_exists)
    so it is picklable under any multiprocessing start method. The full
    project is loaded here, inside the worker, from its config path.

    Returns (name, status, message) with status in {'exported', 'skipped', 'error'}.
    """
    _, name, config_path, this_folder, include_image_data, skip_if_exists = task
    try:
        project = ProjectData.load_final_project_data(config_path, verbose=0)
    except Exception as e:
        return name, 'error', f'Error loading {name}: {e}'

    output_fname = get_nwb_export_fname(project, this_folder, include_image_data)
    if skip_if_exists and os.path.exists(output_fname):
        return name, 'skipped', f'Skipping {output_fname} because it already exists'

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
    parser = argparse.ArgumentParser(
        description='Export paper traces in NWB format using parallel worker processes',
        epilog='''
Examples:
  # One job, 4 worker processes, no image data
  python export_paper_data_as_nwb_parallel.py --num_workers 4

  # With image data (large files)
  python export_paper_data_as_nwb_parallel.py --include_image_data --num_workers 2

  # SLURM array: exactly one dataset, the one at this index (see submit_export_array.py)
  python export_paper_data_as_nwb_parallel.py --include_image_data --backend serial --only_index $SLURM_ARRAY_TASK_ID
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--include_image_data', action='store_true', help='Whether to include image data in the export')
    parser.add_argument('--delete_existing', action='store_true',
                        help='Whether to delete existing export files before exporting (default skips them)')
    parser.add_argument('--suffixes', nargs='+', default=['gfp', '', 'mutant', 'immob'],
                        help='Dataset suffixes to export')
    parser.add_argument('--backend', choices=['processes', 'threads', 'serial'], default='processes',
                        help="'processes' (default, safe) runs each export in its own process. "
                             "'threads' shares one process and is UNSAFE with the current export code "
                             "(os.chdir races, non-threadsafe HDF5: expect corrupt outputs or segfaults). "
                             "'serial' exports one dataset at a time.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (ignored for serial and --only_index). '
                             'Note: each worker holds a full project in memory; with --include_image_data '
                             'keep this modest (2-4).')
    parser.add_argument('--only_index', type=int, default=None,
                        help='Export only the single task at this index of the task list and exit. '
                             'Intended for SLURM job arrays. Out-of-range indexes exit 0 (no-op) so the '
                             'array size may exceed the task count.')
    args = parser.parse_args()

    include_image_data = args.include_image_data
    skip_if_exists = not args.delete_existing

    tasks = build_tasks(args.suffixes, get_parent_dir(include_image_data))
    print(f'{len(tasks)} project(s) to export', flush=True)

    if args.only_index is not None:
        # Single-dataset mode for SLURM arrays: run in-process, serially.
        if not 0 <= args.only_index < len(tasks):
            print(f'--only_index {args.only_index} out of range for {len(tasks)} tasks; nothing to do.',
                  flush=True)
            raise SystemExit(0)
        suffix, name, config_path, this_folder = tasks[args.only_index]
        _, status, message = export_one_task(
            (suffix, name, config_path, this_folder, include_image_data, skip_if_exists))
        if message is not None:
            print(message, flush=True)
        raise SystemExit(0 if status in ('exported', 'skipped') else 1)

    full_tasks = [(suffix, name, config_path, folder, include_image_data, skip_if_exists)
                  for suffix, name, config_path, folder in tasks]

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
              '(read corruption, VOL errors, segfaults). Prefer --backend processes '
              'or a SLURM job array. Continuing anyway in 5 seconds...', flush=True)
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

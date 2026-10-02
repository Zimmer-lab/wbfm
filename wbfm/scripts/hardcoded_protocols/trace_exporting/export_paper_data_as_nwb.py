import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm
import argparse

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.nwb.utils_nwb_export import get_nwb_export_fname, nwb_using_project_data


def export_single_project(name, project, this_folder, include_image_data, skip_if_exists):
    """
    Export one project to NWB. Thread-safe: each task touches only its own
    project object and writes its own output file.

    Returns (name, status, message) with status in {'exported', 'skipped', 'error'}.
    """
    output_fname = get_nwb_export_fname(project, this_folder, include_image_data)
    if skip_if_exists and os.path.exists(output_fname):
        return name, 'skipped', f'Skipping {output_fname} because it already exists'

    try:
        print(f'Exporting {name} to {this_folder}')
        nwb_using_project_data(project, include_image_data=include_image_data, output_folder=this_folder)
    except Exception as e:
        return name, 'error', f'Error exporting {name}: {e}'
    return name, 'exported', None


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(
        description='Export traces in nwb format',
        epilog='''
Examples:
  # Export all default suffixes (gfp, '', mutant, immob)
  python export_paper_data_as_nwb.py

  # Export specific suffixes only
  python export_paper_data_as_nwb.py --suffixes gfp mutant

  # Include image data in exports
  python export_paper_data_as_nwb.py --include_image_data

  # Custom suffixes with images and debug mode
  python export_paper_data_as_nwb.py --suffixes gfp "" mutant --include_image_data --debug

  # Parallel export with 8 worker threads
  python export_paper_data_as_nwb.py --include_image_data --num_workers 8
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--include_image_data', action='store_true', help='Whether to include image data in the export')
    parser.add_argument('--delete_existing', action='store_true', help='Whether to delete existing export files before exporting (default skips them)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--suffixes', nargs='+', default=['gfp', '', 'mutant', 'immob'], help='Dataset suffixes to export')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for parallel export. Tasks are light and mostly IO-bound, '
                             'so threads (not processes) are used. Use 1 for sequential export. '
                             'Note: each worker holds one project in memory, so do not oversubscribe on '
                             '--include_image_data runs.')
    args = parser.parse_args()

    DEBUG = args.debug
    include_image_data = args.include_image_data
    skip_if_exists = not args.delete_existing
    num_workers = 1 if DEBUG else max(1, args.num_workers)

    # Export to hardcoded locations
    parent_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/nwb'
    if include_image_data:
        parent_dir = os.path.join(parent_dir, 'with_images')
    else:
        parent_dir = os.path.join(parent_dir, 'no_images')
    all_suffixes = args.suffixes

    # Build the full task list first; folder creation and project loading stay serial
    tasks = []
    for suffix in tqdm(all_suffixes, desc='Loading datasets'):
        subfolder_name = f'exported_data_{suffix}'
        this_folder = os.path.join(parent_dir, subfolder_name)
        Path(this_folder).mkdir(exist_ok=True)
        all_projects = load_paper_datasets(suffix)
        for name, project in all_projects.items():
            tasks.append((name, project, this_folder))

    print(f'Exporting {len(tasks)} projects using {num_workers} worker thread(s)')

    if DEBUG:
        # Sequential debug mode: export a single project, then stop
        name, project, this_folder = tasks[0]
        _, status, message = export_single_project(name, project, this_folder, include_image_data, skip_if_exists)
        if message is not None:
            print(message)
        print(f'Exported {name} to {this_folder}, breaking')

    elif num_workers == 1:
        for name, project, this_folder in tqdm(tasks, desc='Exporting'):
            print("=" * 50)
            _, status, message = export_single_project(name, project, this_folder, include_image_data, skip_if_exists)
            if message is not None:
                print(message)
            if status == 'error':
                continue

    else:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='nwb-export') as executor:
            future_to_name = {
                executor.submit(export_single_project, name, project, this_folder,
                                include_image_data, skip_if_exists): name
                for name, project, this_folder in tasks
            }
            for future in tqdm(as_completed(future_to_name), total=len(tasks), desc='Exporting'):
                name, status, message = future.result()
                if message is not None:
                    print(message)
                results.append((name, status))

        # Summary (in the threaded branch only; the serial branches print inline)
        n_exported = sum(1 for _, s in results if s == 'exported')
        n_skipped = sum(1 for _, s in results if s == 'skipped')
        failed = sorted(name for name, s in results if s == 'error')
        print(f'Exported {n_exported}, skipped {n_skipped}, failed {len(failed)} of {len(results)} projects')
        if failed:
            print('Failed projects:')
            for name in failed:
                print(f'  {name}')
            raise SystemExit(1)
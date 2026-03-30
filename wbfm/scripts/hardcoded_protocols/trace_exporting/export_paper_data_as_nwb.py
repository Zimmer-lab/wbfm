import os
from pathlib import Path

from tqdm.auto import tqdm
import argparse

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.nwb.utils_nwb_export import nwb_using_project_data

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
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--include_image_data', action='store_true', help='Whether to include image data in the export')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--suffixes', nargs='+', default=['gfp', '', 'mutant', 'immob'], help='Dataset suffixes to export')
    args = parser.parse_args()

    DEBUG = args.debug
    include_image_data = args.include_image_data

    # Export to hardcoded locations
    parent_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/nwb'
    if include_image_data:
        parent_dir = os.path.join(parent_dir, 'with_images')
    else:
        parent_dir = os.path.join(parent_dir, 'no_images')
    all_suffixes = args.suffixes

    for suffix in tqdm(all_suffixes):
        subfolder_name = f'exported_data_{suffix}'
        this_folder = os.path.join(parent_dir, subfolder_name)
        Path(this_folder).mkdir(exist_ok=True)
        all_projects = load_paper_datasets(suffix)

        for name, project in all_projects.items():

            # Skip if file exists
            # if args.skip_if_exists and project.exported_data_path.exists():
            #     print(f'Skipping {project.exported_data_path}')
            #     continue

            # Export data
            try:
                print("=" * 50)
                print(f'Exporting {name} to {this_folder}')
                nwb_using_project_data(project, include_image_data=include_image_data, output_folder=this_folder)
            except Exception as e:
                print(f'Error exporting {name}: {e}')
                continue

            if DEBUG:
                print(f'Exported {name} to {this_folder}, breaking')
                break

import os
from pathlib import Path

from tqdm.auto import tqdm
import argparse

from wbfm.utils.general.hardcoded_paths import load_paper_datasets
from wbfm.utils.nwb.utils_nwb_export import nwb_using_project_data

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Export traces in nwb format')
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    DEBUG = args.debug

    # Export to hardcoded locations
    parent_dir = '/lisc/user/fieseler/zimmer/fieseler/paper/nwb'
    all_suffixes = ['gfp', '', 'mutant']  # don't include immob

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
                nwb_using_project_data(project, include_image_data=False, output_folder=this_folder)
            except Exception as e:
                print(f'Error exporting {name}: {e}')
                continue

            if DEBUG:
                print(f'Exported {name} to {this_folder}, breaking')
                break

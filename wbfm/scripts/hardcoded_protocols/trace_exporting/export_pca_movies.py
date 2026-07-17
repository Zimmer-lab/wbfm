import os
from pathlib import Path

from tqdm.auto import tqdm
import argparse

from wbfm.utils.general.utils_hardcoded import load_paper_datasets
from wbfm.utils.visualization.utils_export_videos import save_video_of_heatmap_and_pca_with_behavior

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Export movies of pca phase space, behavior, and heatmap')
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--include_slowing', action='store_true', help='Include slowing in the video')
    parser.add_argument('--suffixes', '-s', nargs='+', default=['gfp', 'gcamp', 'mutant'], help='Suffixes to include in the export')
    args = parser.parse_args()

    include_slowing = args.include_slowing
    DEBUG = args.debug

    # Export to hardcoded locations
    parent_dir = '/lisc/user/fieseler/zimmer/fieseler/paper/pca_movies'
    all_suffixes = args.suffixes  # don't include immob

    for suffix in tqdm(all_suffixes):
        subfolder_name = f'movies_{suffix}'
        this_folder = os.path.join(parent_dir, subfolder_name)
        Path(this_folder).mkdir(exist_ok=True)
        all_projects = load_paper_datasets(suffix)

        for name, project in all_projects.items():
            Path(parent_dir).mkdir(exist_ok=True)
            output_fname = os.path.join(this_folder, f'{project.shortened_name}.mp4')

            # Skip if file exists
            # if args.skip_if_exists and project.exported_data_path.exists():
            #     print(f'Skipping {project.exported_data_path}')
            #     continue

            # Export data
            save_video_of_heatmap_and_pca_with_behavior(project, output_fname=output_fname,
                                                        include_slowing=include_slowing, DEBUG=DEBUG)

            if DEBUG:
                print(f'Exported {name} to {this_folder}, breaking')
                break

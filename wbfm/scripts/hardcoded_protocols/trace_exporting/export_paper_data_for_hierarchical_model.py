from tqdm.auto import tqdm
import argparse
from wbfm.utils.traces.utils_hierarchical_model_data_export import export_data_for_hierarchical_model

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Export data for hierarchical modeling')
    parser.add_argument('--skip_if_exists', action='store_true', default=False, help='Skip if file exists')
    parser.add_argument('--delete_if_exists', action='store_true', default=False, help='Delete if file exists')
    args = parser.parse_args()

    # Do gfp first because it's faster, so sometimes I can start other pipelines more quickly
    all_suffixes = ['gfp', 'immob', '', 'immob_mutant_o2', 'immob_o2', 'immob_o2_hiscl', 'mutant']
    for suffix in tqdm(all_suffixes):
        export_data_for_hierarchical_model(suffix=suffix,
                                           skip_if_exists=args.skip_if_exists, delete_if_exists=args.delete_if_exists)

import argparse
from ast import arg
import logging
import os
import re
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True  # if you want to preserve quoted strings


def extract_sweep_and_trial(path):
    """
    Given a full path to the snakemake config file, extract:
    - SWEEPTYPE from 'SWEEPTYPE_sweep' folder
    - trial_N from 'PROJECTNAMEtrial_N' folder

    Example path:
        ~/zimmer/fieseler/barlow_track_paper/analyzed_projects/zimmer/augmentation_sweep/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_formattrial_0/snakemake/snakemake_config.yaml
    Format:
        /path/to/LAB/SWEEPTYPE_sweep/PROJECTNAMEtrial_N/snakemake/snakemake_config.yaml

    """
    parts = path.split(os.sep)
    try:
        lab_type = parts[-5]  # LAB
        sweep_dir = parts[-4]  # SWEEPTYPE_sweep
        project_dir = parts[-3]  # PROJECTNAMEtrial_N

        sweep_type = sweep_dir.replace('_sweep', '')
        if 'trial_' not in project_dir:
            raise ValueError("PROJECTNAMEtrial_N format not found in path")
        else:
            project_parts = project_dir.split('trial_')
            if len(project_parts) < 2:
                raise ValueError("PROJECTNAMEtrial_N format not found in path")
            trial_name = 'trial_' + project_parts[-1]

        if not sweep_type or not trial_name or not lab_type:
            raise ValueError("Could not extract SWEEPTYPE or trial")

        return sweep_type, trial_name, lab_type

    except Exception as e:
        print(f"[!] Skipping '{path}': {e}")
        return None, None


def update_config_file(config_path, networks_parent_dir, dry_run=False):
    sweep_type, trial_name, lab_type = extract_sweep_and_trial(config_path)
    model_dir = f"{networks_parent_dir}/{sweep_type}_{lab_type}"
    if not os.path.exists(model_dir):
        # Some have 'sweep' in the folder name
        model_dir = f"{networks_parent_dir}/{sweep_type}_sweep_{lab_type}"

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' does not exist")
    else:
        model_path = os.path.join(model_dir, trial_name, 'resnet50.pth')
        
    if not os.path.exists(model_path):
        print("="*20)
        logging.warning(f"Model not found at {model_path}; skipping {config_path}. Setting barlow_model_path to null (analysis should fail later)")
        model_path = None

    try:
        with open(config_path, 'r') as f:
            data = yaml.load(f)

        # Update or insert keys
        data['use_barlow_tracker'] = True
        data['barlow_model_path'] = model_path

        if not dry_run:
            with open(config_path, 'w') as f:
                yaml.dump(data, f)
        else:
            print()
            print(f"[DRY RUN] Would update '{config_path}' with:")
            print(f"  use_barlow_tracker: True")
            print(f"  barlow_model_path: {model_path}")
            return

        print(f"[✓] Updated: {config_path}")

    except Exception as e:
        print(f"[!] Error updating '{config_path}': {e}")


def find_and_update_configs(root_dir, networks_parent_dir, dry_run=False):
    for dirpath, _, filenames in os.walk(root_dir):
        if 'snakemake_config.yaml' in filenames:
            full_path = os.path.join(dirpath, 'snakemake_config.yaml')
            update_config_file(full_path, networks_parent_dir, dry_run=dry_run)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Update snakemake_config.yaml files with Barlow network settings.")
    args.add_argument('--project_dir', type=str, help="Root directory to start searching from.")
    args.add_argument('--networks_parent_dir', type=str, help="Parent directory where trained Barlow networks are stored.")
    args.add_argument('--dry_run', action='store_true', help="If set, will only print changes without writing to files.")
    args = args.parse_args()

    find_and_update_configs(args.project_dir, args.networks_parent_dir, dry_run=args.dry_run)

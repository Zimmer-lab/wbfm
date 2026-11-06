import argparse
import shutil
from os import path as osp
from pathlib import Path

from tqdm.auto import tqdm

from wbfm.utils.projects.project_config_classes import ModularProjectConfig


def iter_project_configs(target):
    target = Path(target)
    if target.is_file() and target.name == "project_config.yaml":
        yield target
        return
    for sub in sorted(target.iterdir()):
        if not sub.is_dir():
            continue
        fname = sub / "project_config.yaml"
        if fname.exists():
            yield fname


def restore_one(project_fname, overwrite=False, dry_run=False):
    project_fname = str(project_fname)
    cfg = ModularProjectConfig(project_fname, log_to_file=False)
    try:
        cfg.get_local_raw_data_config_filename()
        if not overwrite:
            return "exists"
    except FileNotFoundError:
        pass
    remote = cfg.get_remote_raw_data_config_filename()
    beh_folder = cfg.get_behavior_config().absolute_subfolder
    new_fname = osp.join(beh_folder, "raw_data_config.yaml")
    if dry_run:
        return f"would copy {remote} -> {new_fname}"
    Path(beh_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(remote, new_fname)
    return f"copied {remote} -> {new_fname}"


def main():
    parser = argparse.ArgumentParser(description="Copy missing behavior/raw_data_config.yaml from the raw data folder config.yaml, reusing the same lookup as project initialization")
    parser.add_argument("target", help="project_config.yaml, single project folder, or parent folder of projects")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing local raw_data_config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="only report what would be done")
    args = parser.parse_args()

    for fname in tqdm(list(iter_project_configs(args.target))):
        try:
            result = restore_one(fname, overwrite=args.overwrite, dry_run=args.dry_run)
            print(f"{fname.parent.name}: {result}")
        except FileNotFoundError as e:
            print(f"{fname.parent.name}: skipped, no raw data config found ({e})")
        except PermissionError as e:
            print(f"{fname.parent.name}: skipped, permission error ({e})")


if __name__ == "__main__":
    main()

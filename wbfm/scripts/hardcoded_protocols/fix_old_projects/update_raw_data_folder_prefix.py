"""Point projects at a new raw-data location (field-aware).

Scope: ONLY the top-level project_config.yaml of each project, and ONLY
these known raw-data fields (not every key containing the prefix, and not
subfolder configs like preprocessing/behavior yaml, which use relative
paths):
  - parent_data_folder: prefix-swap, must exist afterwards.
  - red/green_bigtiff_fname: prefix-swap if the new .btf exists; otherwise
    clear the field and set red/green_fname to the new channel folder
    (folder-style, is_btf=False). Clearing is required because
    get_raw_data_fname() prefers a non-empty bigtiff field without checking
    that it exists.
  - red/green_fname (if already present): prefix-swap, must exist afterwards.

A project is skipped unless every updated path checks out (or --force).
Writing is done via wbfm.utils.external.utils_yaml.edit_config.

Example:
    old: /lisc/data/scratch/neurobiology/zimmer/brenner/WBFM_raw
    new: /lisc/data/scratch/neurobiology/zimmer/zeillinger/WBFM_raw
"""

import argparse
from pathlib import Path

from wbfm.utils.external.utils_yaml import edit_config, load_config


def iter_project_configs(target):
    target = Path(target)
    if target.is_file() and target.name == "project_config.yaml":
        yield target
        return
    if target.is_dir() and (target / "project_config.yaml").exists():
        yield target / "project_config.yaml"
        return
    for sub in sorted(target.iterdir()):
        if not sub.is_dir():
            continue
        fname = sub / "project_config.yaml"
        if fname.exists():
            yield fname


def _swap(path, old_prefix, new_prefix):
    if not path or not str(path).startswith(old_prefix):
        return None
    return str(path).replace(old_prefix, new_prefix, 1)


def remap_one(project_fname, old_prefix, new_prefix, dry_run=False, force=False):
    cfg = load_config(str(project_fname))
    edits = {}  # field -> new value
    problems = []

    parent = cfg.get("parent_data_folder")
    new_parent = _swap(parent, old_prefix, new_prefix)
    if new_parent is not None:
        if Path(new_parent).exists():
            edits["parent_data_folder"] = new_parent
        else:
            problems.append(f"new parent_data_folder missing: {new_parent}")

    for color in ("red", "green"):
        btf_field, folder_field = f"{color}_bigtiff_fname", f"{color}_fname"
        btf = cfg.get(btf_field)
        new_btf = _swap(btf, old_prefix, new_prefix)
        if new_btf is not None:
            if Path(new_btf).exists():
                edits[btf_field] = new_btf
            else:
                # Fall back to folder-style reference of the new channel folder
                channel_folder = str(Path(new_btf).parent)
                if Path(channel_folder).exists():
                    edits[btf_field] = None
                    edits[folder_field] = channel_folder
                else:
                    problems.append(f"neither {new_btf} nor {channel_folder} exist")
        else:
            folder = cfg.get(folder_field)
            new_folder = _swap(folder, old_prefix, new_prefix)
            if new_folder is not None:
                if Path(new_folder).exists():
                    edits[folder_field] = new_folder
                else:
                    problems.append(f"new {folder_field} missing: {new_folder}")

    if not edits and not problems:
        return "unchanged (no stale prefix found)"
    if problems and not force:
        return f"SKIPPED, {'; '.join(problems)}"
    if dry_run:
        summary = ", ".join(f"{field} -> {new}" for field, new in edits.items())
        return f"would update {len(edits)} field(s): {summary}"
    edit_config(str(project_fname), edits)
    summary = ", ".join(f"{field} -> {new}" for field, new in edits.items())
    extra = f"; WARNING: {'; '.join(problems)}" if problems else ""
    return f"updated {len(edits)} field(s): {summary}{extra}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="project_config.yaml, single project folder, or parent folder of projects")
    parser.add_argument("--old-prefix", required=True, help="stale raw-data path prefix to replace")
    parser.add_argument("--new-prefix", required=True, help="new raw-data path prefix")
    parser.add_argument("--dry_run", action="store_true", help="only report what would be done")
    parser.add_argument("--force", action="store_true", help="write even if some new paths do not exist")
    args = parser.parse_args()

    for fname in iter_project_configs(args.target):
        try:
            result = remap_one(fname, old_prefix=args.old_prefix, new_prefix=args.new_prefix,
                               dry_run=args.dry_run, force=args.force)
            print(f"{fname.parent.name}: {result}")
        except PermissionError as e:
            print(f"{fname.parent.name}: skipped, permission error ({e})")


if __name__ == "__main__":
    main()

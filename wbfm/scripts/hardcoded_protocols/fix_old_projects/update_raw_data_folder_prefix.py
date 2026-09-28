"""Point projects at a new raw-data location (field-aware).

Dumb prefix swap, nothing else: no existence checks, no fallbacks.

Default scope: ONLY the top-level project_config.yaml of each project, and
ONLY these known raw-data fields:
  - parent_data_folder, red/green_bigtiff_fname, red/green_fname.
Writing is done via wbfm.utils.external.utils_yaml.edit_config.

With --sweep-all-yaml, additionally replaces the prefix in EVERY string
value of EVERY *.yaml/*.yml file under the project folder. Skips *.zarr*
directories.

Example:
    python update_raw_data_folder_prefix.py --target /lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/brenner/analyze/freely_moving_mutant  --old-prefix /lisc/data/scratch/neurobiology/zimmer/brenner/WBFM_raw --new-prefix /lisc/data/scratch/neurobiology/zimmer/zeillinger/WBFM_raw --dryrun
"""

import argparse
import os
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


def _short(path, old_prefix, new_prefix):
    """Abbreviate a path using the <OLD>/<NEW> legend printed by main()."""
    s = str(path)
    if s.startswith(old_prefix):
        return "<OLD>" + s[len(old_prefix):]
    if s.startswith(new_prefix):
        return "<NEW>" + s[len(new_prefix):]
    return s


def _format_changes(edits, olds, old_prefix, new_prefix, verb):
    """Multi-line, one block per field:
        field:
          - <OLD>/...
          + <NEW>/...
    """
    lines = [f"{verb} {len(edits)} field(s):"]
    for field, new in edits.items():
        old = olds.get(field)
        lines.append(f"    {field}:")
        if old is not None:
            lines.append(f"      - {_short(old, old_prefix, new_prefix)}")
        if new is None:
            lines.append("      + (cleared)")
        else:
            lines.append(f"      + {_short(new, old_prefix, new_prefix)}")
    return "\n".join(lines)


def remap_one(project_fname, old_prefix, new_prefix, dryrun=False):
    cfg = load_config(str(project_fname))
    edits = {}  # field -> new value
    olds = {}  # field -> old value (for display)

    for field in ("parent_data_folder", "red_bigtiff_fname", "green_bigtiff_fname",
                    "red_fname", "green_fname"):
        new_value = _swap(cfg.get(field), old_prefix, new_prefix)
        if new_value is not None:
            edits[field] = new_value
            olds[field] = cfg.get(field)

    if not edits:
        return "unchanged (no stale prefix found)"
    if dryrun:
        return _format_changes(edits, olds, old_prefix, new_prefix, "would update")
    edit_config(str(project_fname), edits)
    return _format_changes(edits, olds, old_prefix, new_prefix, "updated")


def _swap_in_obj(obj, old_prefix, new_prefix, stats):
    """Recursively prefix-swap every string in dicts/lists. stats: [n_swapped]."""
    if isinstance(obj, dict):
        return {k: _swap_in_obj(v, old_prefix, new_prefix, stats) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_swap_in_obj(v, old_prefix, new_prefix, stats) for v in obj]
    if isinstance(obj, str) and old_prefix in obj:
        stats[0] += 1
        return obj.replace(old_prefix, new_prefix)
    return obj


def iter_yaml_files(project_dir):
    project_dir = Path(project_dir)
    for root, dirnames, filenames in os.walk(project_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.endswith((".zarr", ".zarr.zip")))
        for name in sorted(filenames):
            if name.endswith((".yaml", ".yml")):
                fname = Path(root) / name
                # project_config.yaml is handled field-aware by remap_one, not here
                if fname == project_dir / "project_config.yaml":
                    continue
                yield fname


def sweep_all_yaml(project_dir, old_prefix, new_prefix, dryrun=False):
    """Plain prefix-swap of every string value in every yaml file. Returns report string."""
    touched, total_swaps = [], 0
    for fname in iter_yaml_files(project_dir):
        try:
            text = fname.read_text()
        except (PermissionError, UnicodeDecodeError):
            continue
        if old_prefix not in text:
            continue
        cfg = load_config(str(fname))
        if not isinstance(cfg, dict):
            continue
        stats = [0]
        new_cfg = _swap_in_obj(cfg, old_prefix, new_prefix, stats)
        if stats[0] == 0:
            continue
        total_swaps += stats[0]
        if not dryrun:
            edit_config(str(fname), new_cfg)
        touched.append(f"{fname.relative_to(project_dir)} ({stats[0]})")
    if not touched:
        return "sweep: no other yaml files contained the prefix"
    lines = [f"sweep: {'would update' if dryrun else 'updated'} {total_swaps} value(s) in {len(touched)} file(s):"]
    lines.extend(f"    {entry}" for entry in touched)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="project_config.yaml, single project folder, or parent folder of projects")
    parser.add_argument("--old-prefix", required=True, help="stale raw-data path prefix to replace")
    parser.add_argument("--new-prefix", required=True, help="new raw-data path prefix")
    parser.add_argument("--dryrun", action="store_true", help="only report what would be done")
    parser.add_argument("--sweep-all-yaml", action="store_true",
                        help="also prefix-swap every string value in every *.yaml/*.yml under the project folder")
    args = parser.parse_args()

    print(f"<OLD> = {args.old_prefix}")
    print(f"<NEW> = {args.new_prefix}")
    for fname in iter_project_configs(args.target):
        print(f"=== {fname.parent.name} ===")
        try:
            result = remap_one(fname, old_prefix=args.old_prefix, new_prefix=args.new_prefix,
                               dryrun=args.dryrun)
            print(result)
            if args.sweep_all_yaml:
                sweep = sweep_all_yaml(fname.parent, old_prefix=args.old_prefix, new_prefix=args.new_prefix,
                                       dryrun=args.dryrun)
                print(sweep)
        except PermissionError as e:
            print(f"skipped, permission error ({e})")


if __name__ == "__main__":
    main()

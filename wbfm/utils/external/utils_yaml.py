import os
import pathlib
import typing
from os import path as osp
from pathlib import Path

from ruamel.yaml import YAML


def edit_config(config_fname: typing.Union[str, pathlib.Path], edits: dict, allow_new_creation=True, DEBUG: bool = False) -> dict:
    """Generic overwriting, based on DLC. Will create new file if one isn't found"""

    if DEBUG:
        print(f"Editing config file at: {config_fname}")
    if Path(config_fname).exists():
        cfg = load_config(config_fname)
    else:
        if allow_new_creation:
            cfg = {}
            print(f"Config file not found, creating new one")
        else:
            raise FileNotFoundError(config_fname)

    if DEBUG:
        print(f"Initial config: {cfg}")
        print(f"Edits: {edits}")

    for k, v in edits.items():
        cfg[k] = v

    with open(config_fname, "w") as f:
        YAML().dump(cfg, f)

    return cfg


def load_config(config_fname: typing.Union[str, pathlib.Path]) -> dict:
    if not osp.exists(config_fname):
        # Try to append "project_config.yaml" to the end
        config_fname_appended = osp.join(config_fname, 'project_config.yaml')
        if not osp.exists(config_fname_appended):
            raise FileNotFoundError(f"{config_fname} not found from current directory {os.getcwd()}!")
        config_fname = config_fname_appended

    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    return cfg


def recursive_dict_update(base_dict, update_dict):
    """
    Recursively update dict `base_dict` with dict `update_dict`.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
            recursive_dict_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

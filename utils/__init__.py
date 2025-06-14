import os
import re
import sys
from typing import Any, Dict, List

import tomli_w

# 根据Python版本选择合适的TOML库
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def list_dir(path: str) -> List[str]:
    validate_dir(path)

    def sort_key(name: str):
        parts = re.split(r'(\d+)', name)
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(os.listdir(path), key=sort_key)


def validate_dir(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path {path} is not a directory.")


def validate_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"Path {path} is not a file.")


def apply_changes(a: dict, b: dict) -> None:
    for key, subdict in b.items():
        if key not in a:
            a[key] = subdict
        else:
            for subkey, value in subdict.items():
                a[key][subkey] = value


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)

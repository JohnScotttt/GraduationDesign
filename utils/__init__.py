import os
import re
import sys
from typing import Any, Dict, List


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


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config

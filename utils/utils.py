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
    """
    List the contents of a directory, sorted alphabetically for letters and numerically for numbers.

    Args:
        path (str): The directory path to list.

    Returns:
        List[str]: Sorted list of directory contents.
    """
    validate_dir(path)

    def sort_key(name: str):
        parts = re.split(r'(\d+)', name)
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(os.listdir(path), key=sort_key)


def validate_dir(path: str) -> None:
    """
    Validate if the given path is a directory and exists.
    Args:
        path (str): The path to validate.
    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path {path} is not a directory.")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from the TOML file.
    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config

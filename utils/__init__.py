import os
import re
from typing import List


def _listdir(path: str) -> List[str]:
    """
    List the contents of a directory, sorted alphabetically for letters and numerically for numbers.

    Args:
        path (str): The directory path to list.

    Returns:
        List[str]: Sorted list of directory contents.
    """
    _validate_directory(path)

    def sort_key(name: str):
        parts = re.split(r'(\d+)', name)
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(os.listdir(path), key=sort_key)


def _validate_directory(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path {path} is not a directory.")
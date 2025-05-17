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


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config

def _params_to_dict(params_obj):
    """将params对象转换为字典格式"""
    result = {}
    for field in params_obj.__dataclass_fields__:
        value = getattr(params_obj, field)
        if hasattr(value, '__dataclass_fields__'):
            result[field] = _params_to_dict(value)
        else:
            result[field] = value
    return result

def save_config(config: Any, config_path: str) -> None:
    """保存配置到TOML文件"""
    if hasattr(config, '__dataclass_fields__'):
        config = _params_to_dict(config)
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)

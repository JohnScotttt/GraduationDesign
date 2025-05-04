import os
from typing import List, Tuple

import pandas as pd

from utils import list_dir, validate_dir


def generate_datalist(dataset_path: str, dataset_name: str = 'LOL', output_dir: str = './') -> None:
    """
    Generate a .csv file containing the pairs of high and low image paths.

    Args:
        dataset_path (str): Path to the dataset.
        dataset_name (str): Name of the dataset if public, otherwise 'private' or empty.
        output_dir (str): Directory to save the output files.
    """
    if dataset_name == 'LOL':
        _LOL(dataset_path, output_dir)
    elif dataset_name == 'LOLv2':
        _LOLv2(dataset_path, output_dir)
    elif dataset_name == 'private':
        _private(dataset_path, output_dir)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _get_image_pairs(low_dir: str, high_dir: str) -> List[Tuple[str, str]]:
    validate_dir(low_dir)
    validate_dir(high_dir)

    low_list = list_dir(low_dir)
    high_list = list_dir(high_dir)

    return [(os.path.join(low_dir, path), os.path.join(high_dir, path))
            for path in low_list if path.endswith('.png') and path in high_list]


def _LOL(dataset_path: str, output_dir: str) -> None:
    validate_dir(dataset_path)

    train_list = _get_image_pairs(os.path.join(dataset_path, 'our485', 'low'),
                                  os.path.join(dataset_path, 'our485', 'high'))

    eval_list = _get_image_pairs(os.path.join(dataset_path, 'eval15', 'low'),
                                 os.path.join(dataset_path, 'eval15', 'high'))

    os.makedirs(output_dir, exist_ok=True)

    _write_csv(os.path.join(output_dir, 'train.tsv'), train_list)
    _write_csv(os.path.join(output_dir, 'eval.tsv'), eval_list)


def _LOLv2(dataset_path: str, output_dir: str) -> None:
    validate_dir(dataset_path)

    train_list = _get_image_pairs(os.path.join(dataset_path, 'Train', 'Low'),
                                  os.path.join(dataset_path, 'Train', 'Normal'))

    test_list = _get_image_pairs(os.path.join(dataset_path, 'Test', 'Low'),
                                 os.path.join(dataset_path, 'Test', 'Normal'))

    os.makedirs(output_dir, exist_ok=True)

    _write_csv(os.path.join(output_dir, 'train.tsv'), train_list)
    _write_csv(os.path.join(output_dir, 'test.tsv'), test_list)


def _private(dataset_path: str, output_dir: str) -> None:
    # _validate_directory(dataset_path)

    raise NotImplementedError(
        "The private dataset preparation is not implemented yet."
    )


def _write_csv(file_path: str, data_list: List[Tuple[str, str]]) -> None:
    if os.path.exists(file_path):
        overwrite = input(
            f"\033[33m {file_path} already exists. Do you want to overwrite it? (y/[n]): \033[0m").strip().lower()
        if overwrite != 'y':
            return

    df = pd.DataFrame(data_list, columns=['low_path', 'high_path'])
    df.to_csv(file_path, sep='\t', index=False, header=False)

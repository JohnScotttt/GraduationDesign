import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    """Dataset for low-light image enhancement"""

    def __init__(self, tsv_file):
        """
        Args:
            tsv_file (string): Path to the TSV file containing pairs of low-light and ground truth image paths
        """
        # Read TSV file without header
        self.data_pairs = pd.read_csv(tsv_file, sep='\t', header=None, names=[
                                      'low_light', 'ground_truth'])

        # Check if files exist
        self.data_pairs = self.data_pairs[
            self.data_pairs.apply(lambda x: os.path.exists(
                x['low_light']) and os.path.exists(x['ground_truth']), axis=1)
        ]

        # Reset index after filtering
        self.data_pairs = self.data_pairs.reset_index(drop=True)

        # Initialize tensor conversion
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image paths and convert to string
        low_light_path = str(self.data_pairs.iloc[idx]['low_light'])
        gt_path = str(self.data_pairs.iloc[idx]['ground_truth'])

        # Load images
        low_light_img = Image.open(low_light_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # Convert to tensors [0, 1]
        low_light_tensor = self.to_tensor(low_light_img)
        gt_tensor = self.to_tensor(gt_img)

        return low_light_tensor, gt_tensor


def get_dataloader(tsv_file, batch_size=1, shuffle=True, num_workers=1):
    """Create data loader for training or evaluation
    Args:
        tsv_file (string): Path to the TSV file
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of subprocesses for data loading
    Returns:
        dataloader: PyTorch DataLoader object
    """
    dataset = LowLightDataset(tsv_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Usage example
if __name__ == '__main__':
    # Create training data loader
    train_loader = get_dataloader(
        tsv_file='path/to/train.tsv',
        batch_size=4,
        shuffle=True
    )

    # Create validation data loader
    val_loader = get_dataloader(
        tsv_file='path/to/val.tsv',
        batch_size=1,
        shuffle=False
    )

    # Test data loading
    for low_light, ground_truth in train_loader:
        print(f"Low-light shape: {low_light.shape}")
        print(f"Ground-truth shape: {ground_truth.shape}")
        print(f"Value range: [{low_light.min():.3f}, {low_light.max():.3f}]")

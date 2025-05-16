import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop


class LowLightDataset(Dataset):
    """Dataset for low-light image enhancement"""

    def __init__(self, tsv_file, patch_size=None, preload=True):
        """
        Args:
            tsv_file (string): Path to the TSV file containing pairs of low-light and ground truth image paths
            patch_size (int, tuple, or None): Size of the patch to crop from the images. None means no cropping.
            preload (bool): Whether to preload all images into memory
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

        # Set patch size
        self.patch_size = patch_size
        
        # Preload images if requested
        self.preload = preload
        self.preloaded_data = []
        if preload:
            print("Preloading images into memory...")
            for idx in range(len(self.data_pairs)):
                low_light_path = str(self.data_pairs.iloc[idx]['low_light'])
                gt_path = str(self.data_pairs.iloc[idx]['ground_truth'])
                
                low_light_img = Image.open(low_light_path).convert('RGB')
                gt_img = Image.open(gt_path).convert('RGB')
                
                low_light_tensor = self.to_tensor(low_light_img)
                gt_tensor = self.to_tensor(gt_img)
                
                self.preloaded_data.append((low_light_tensor, gt_tensor))
            print("Preloading completed!")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            low_light_tensor, gt_tensor = self.preloaded_data[idx]
        else:
            # Get image paths and convert to string
            low_light_path = str(self.data_pairs.iloc[idx]['low_light'])
            gt_path = str(self.data_pairs.iloc[idx]['ground_truth'])

            # Load images
            low_light_img = Image.open(low_light_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')

            # Convert to tensors [0, 1]
            low_light_tensor = self.to_tensor(low_light_img)
            gt_tensor = self.to_tensor(gt_img)

        # Decide crop size and whether to crop
        if self.patch_size is not None:
            if isinstance(self.patch_size, int):
                crop_h = crop_w = self.patch_size
            elif isinstance(self.patch_size, (tuple, list)) and len(self.patch_size) == 2:
                crop_h, crop_w = self.patch_size
            else:
                raise ValueError("patch_size must be int, tuple of (h, w), or None")
            _, h, w = low_light_tensor.shape
            if h < crop_h or w < crop_w:
                raise ValueError(f"Image size ({h},{w}) is smaller than patch size ({crop_h},{crop_w})")
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            low_light_tensor = crop(low_light_tensor, top, left, crop_h, crop_w)
            gt_tensor = crop(gt_tensor, top, left, crop_h, crop_w)
        # else: do not crop
        return low_light_tensor, gt_tensor


def get_dataloader(tsv_file, batch_size=1, shuffle=True, num_workers=1, patch_size=None, preload=True):
    """Create data loader for training or evaluation
    Args:
        tsv_file (string): Path to the TSV file
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of subprocesses for data loading
        patch_size (int, tuple, or None): Size of the patch to crop from the images. None means no cropping.
        preload (bool): Whether to preload all images into memory
    Returns:
        dataloader: PyTorch DataLoader object
    """
    dataset = LowLightDataset(tsv_file, patch_size=patch_size, preload=preload)

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
        shuffle=True,
        patch_size=224
    )

    # Create validation data loader
    val_loader = get_dataloader(
        tsv_file='path/to/val.tsv',
        batch_size=1,
        shuffle=False,
        patch_size=224
    )

    # Test data loading
    for low_light, ground_truth in train_loader:
        print(f"Low-light shape: {low_light.shape}")
        print(f"Ground-truth shape: {ground_truth.shape}")
        print(f"Value range: [{low_light.min():.3f}, {low_light.max():.3f}]")

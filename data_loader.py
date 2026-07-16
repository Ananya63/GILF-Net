# ============================================================
# Data Loading & Preprocessing
# ============================================================
# Builds stratified train/val/test splits (70/20/10) for an
# ImageFolder-style dataset (e.g. the PBC WBC dataset), with:
#   - class-balanced sampling for training (WeightedRandomSampler)
#   - standard ImageNet normalization
#   - basic augmentation on the training split only
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit


def build_transforms(image_size=256):
    """Returns (train_transform, val_test_transform, unnormalized_transform)."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=25, translate=(0.25, 0.25), scale=(0.75, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    # Used only to read labels for the stratified split (no normalization needed)
    unnormalized_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transform, val_test_transform, unnormalized_transform


def count_per_class(subset):
    counts = Counter()
    for _, y in subset:
        counts[int(y)] += 1
    return counts


def get_dataloaders(data_dir, batch_size=16, image_size=256,
                     val_test_size=0.3, test_fraction_of_temp=1 / 3,
                     random_state=42, num_workers=2, save_split_csv=None):
    """
    Builds stratified 70/20/10 train/val/test DataLoaders from an
    ImageFolder-compatible directory.

    Args:
        data_dir: path to the root folder, organized as one subfolder per class.
        batch_size: batch size for all three loaders.
        image_size: images are resized to (image_size, image_size).
        val_test_size: fraction of data held out for val+test combined (default 0.3 -> 70/30 split).
        test_fraction_of_temp: fraction of the held-out 30% used for the test set
            (default 1/3 -> final split is 70/20/10).
        random_state: seed for reproducible splits.
        num_workers: DataLoader worker processes.
        save_split_csv: optional path to save a CSV summary of the class-wise split.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform, val_test_transform, unnormalized_transform = build_transforms(image_size)

    # Load once (unnormalized) purely to extract labels for stratification
    base_dataset = datasets.ImageFolder(root=data_dir, transform=unnormalized_transform)
    labels = np.array(base_dataset.targets)
    class_names = base_dataset.classes

    print("Class mapping:", base_dataset.class_to_idx)
    print("Total images:", len(base_dataset))

    # Stratified split: first carve off val+test, then split that into val/test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_size, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))

    temp_labels = labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction_of_temp, random_state=random_state)
    val_sub_idx, test_sub_idx = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # Re-instantiate with the correct transform per split
    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_idx)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=val_test_transform), val_idx)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=val_test_transform), test_idx)

    # Class-balanced sampling for training only
    train_targets = labels[train_idx]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Reporting
    print("\nClass names:", class_names)
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    num_classes = len(class_names)
    train_counts = np.bincount(labels[train_idx], minlength=num_classes)
    val_counts = np.bincount(labels[val_idx], minlength=num_classes)
    test_counts = np.bincount(labels[test_idx], minlength=num_classes)
    total_counts = train_counts + val_counts + test_counts

    df_split = pd.DataFrame({
        "Class": class_names,
        "Train": train_counts,
        "Validation": val_counts,
        "Test": test_counts,
        "Total": total_counts
    })
    print("\nClass-wise Data Split:\n", df_split)

    if save_split_csv:
        df_split.to_csv(save_split_csv, index=False)
        print(f"Saved split summary to {save_split_csv}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Example usage - update DATA_DIR to your local dataset path
    DATA_DIR = os.environ.get("WBC_DATA_DIR", "./data/PBC_dataset_normal_DIB")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=16,
        save_split_csv="classwise_split.csv"
    )

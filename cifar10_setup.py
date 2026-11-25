"""Reproducible CIFAR-10 data setup for model development and final evaluation."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Numeric characters from M26CSA001, kept in their original order.
SEED = 26001
DATA_DIR = "data"
BATCH_SIZE = 128
NUM_WORKERS = 0
NUM_CLASSES = 10
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
SPLIT_FILE = Path(DATA_DIR) / "split_seed_26001.npz"

_NORMALIZE = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
)


class PixelScramble:
    """Apply one fixed spatial permutation while preserving each RGB pixel."""

    def __init__(self, seed=SEED):
        self.order = torch.randperm(32 * 32, generator=torch.Generator().manual_seed(seed))

    def __call__(self, image):
        return image.reshape(3, -1)[:, self.order].reshape(3, 32, 32)


def make_transform(condition="clean", normalize=True):
    operations = []
    if condition == "grayscale":
        operations.append(transforms.Grayscale(num_output_channels=3))
    elif condition not in ("clean", "scrambled"):
        raise ValueError(f"Unknown CIFAR-10 condition: {condition}")

    operations.append(transforms.ToTensor())
    if condition == "scrambled":
        operations.append(PixelScramble())
    if normalize:
        operations.append(_NORMALIZE)
    return transforms.Compose(operations)


def seed_everything(seed=SEED):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_split_indices(targets, split_file=SPLIT_FILE):
    """Create and save one deterministic, stratified split of CIFAR-10 train."""
    indices = np.arange(len(targets))
    train_indices, validation_indices = train_test_split(
        indices,
        train_size=TRAIN_SIZE,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=np.asarray(targets),
    )
    counts = np.bincount(np.asarray(targets)[train_indices], minlength=NUM_CLASSES)
    validation_counts = np.bincount(
        np.asarray(targets)[validation_indices], minlength=NUM_CLASSES
    )
    if len(train_indices) != TRAIN_SIZE or len(validation_indices) != VALIDATION_SIZE:
        raise RuntimeError("CIFAR-10 train/validation split has incorrect sizes.")
    if not np.all(counts == 4_500) or not np.all(validation_counts == 500):
        raise RuntimeError("CIFAR-10 split is not balanced across all classes.")

    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_file,
        train_indices=train_indices,
        validation_indices=validation_indices,
        seed=np.asarray(SEED),
    )
    return train_indices, validation_indices


def make_train_validation_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    download=True,
    condition="clean",
    normalize=True,
):
    """Return stratified 45,000-image train and 5,000-image validation loaders."""
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=make_transform(condition, normalize),
    )

    train_indices, validation_indices = make_split_indices(train_dataset.targets)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        Subset(train_dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, validation_loader


def make_test_loader(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    download=True,
    condition="clean",
    normalize=True,
):
    """Create the official test loader; use only after model choices are fixed."""
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=make_transform(condition, normalize),
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

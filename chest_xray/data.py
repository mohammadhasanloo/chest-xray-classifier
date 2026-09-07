"""Loading and splitting the chest X-ray images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

IMAGE_SIZE = (128, 128)
SPLIT_DIRECTORIES = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})
RANDOM_STATE = 0


def class_names(root: Path) -> list[str]:
    """Class folder names in a fixed order.

    Sorted rather than taken from directory listing order, which is undefined:
    label 0 must mean the same class on every machine.
    """
    train_root = root / SPLIT_DIRECTORIES[0]
    names = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not names:
        raise FileNotFoundError(f"no class directories under {train_root}")
    return names


def load_images(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read every image under every split into one array, with integer labels.

    The published dataset ships a validation split of only sixteen images, too
    few to select on, so all three directories are pooled here and re-split below.
    """
    root = Path(root)
    names = class_names(root)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for split in SPLIT_DIRECTORIES:
        for label, name in enumerate(names):
            directory = root / split / name
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    raise OSError(f"could not read {path}")
                images.append(cv2.cvtColor(cv2.resize(image, IMAGE_SIZE), cv2.COLOR_BGR2RGB))
                labels.append(label)

    if not images:
        raise FileNotFoundError(f"no images found under {root}")
    # Kept as uint8 in [0, 255]: EfficientNet normalises its own input.
    return np.asarray(images, dtype=np.uint8), np.asarray(labels), names


def stratified_splits(
    images: np.ndarray, labels: np.ndarray, test_size: float = 0.4
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split 60/20/20, preserving the class ratio in every split."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_index, holdout_index = next(splitter.split(images, labels))

    holdout_images, holdout_labels = images[holdout_index], labels[holdout_index]
    halver = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    test_index, val_index = next(halver.split(holdout_images, holdout_labels))

    return {
        "train": (images[train_index], labels[train_index]),
        "validation": (holdout_images[val_index], holdout_labels[val_index]),
        "test": (holdout_images[test_index], holdout_labels[test_index]),
    }

"""Tests for splitting, thresholding and model wiring."""

from __future__ import annotations

import numpy as np
import pytest

from chest_xray.data import IMAGE_SIZE, class_names, load_images, stratified_splits
from chest_xray.evaluate import best_threshold, evaluate

CLASSES = ("NORMAL", "PNEUMONIA")


@pytest.fixture
def dataset_root(tmp_path):
    """A miniature stand-in: <split>/<class>/<image>.jpeg, imbalanced like the real one."""
    import cv2

    rng = np.random.default_rng(0)
    counts = {"train": (12, 32), "val": (2, 4), "test": (4, 10)}
    for split, per_class in counts.items():
        for label, name in enumerate(CLASSES):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for n in range(per_class[label]):
                image = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
                cv2.imwrite(str(directory / f"{n}.jpeg"), image)
    return tmp_path


def test_class_names_are_sorted_not_filesystem_ordered(dataset_root):
    """os.listdir order is undefined, so label 0 must not depend on it."""
    assert class_names(dataset_root) == sorted(CLASSES)


def test_images_load_as_uint8_in_full_range(dataset_root):
    """EfficientNet rescales internally, so images must stay in [0, 255]."""
    images, labels, names = load_images(dataset_root)
    assert images.dtype == np.uint8
    assert images.shape == (64, *IMAGE_SIZE, 3)
    assert images.max() > 1
    assert labels.shape == (64,)
    assert names == sorted(CLASSES)


def test_splits_are_disjoint_and_keep_the_class_ratio(dataset_root):
    images, labels, _ = load_images(dataset_root)
    splits = stratified_splits(images, labels)

    sizes = {name: len(y) for name, (_, y) in splits.items()}
    assert sum(sizes.values()) == len(labels)

    overall = labels.mean()
    for _, y in splits.values():
        assert abs(y.mean() - overall) < 0.1


def test_best_threshold_beats_the_default_on_imbalanced_scores():
    """A well-separated but shifted score distribution should not be cut at 0.5."""
    y_true = np.array([0] * 20 + [1] * 80)
    probabilities = np.concatenate(
        [np.linspace(0.05, 0.25, 20), np.linspace(0.30, 0.95, 80)]
    )
    threshold = best_threshold(y_true, probabilities)
    assert 0.25 <= threshold <= 0.35

    tuned = evaluate(y_true, probabilities, list(CLASSES))
    default = evaluate(y_true, probabilities, list(CLASSES), threshold=0.5)
    assert tuned.accuracy >= default.accuracy
    assert tuned.auc == pytest.approx(1.0)


def test_evaluation_confusion_matrix_is_two_by_two():
    y_true = np.array([0, 0, 1, 1])
    result = evaluate(y_true, np.array([0.1, 0.2, 0.8, 0.9]), list(CLASSES), threshold=0.5)
    assert result.matrix.shape == (2, 2)
    assert result.accuracy == 1.0


def test_model_maps_a_batch_of_images_to_probabilities():
    """Built without pretrained weights so the test needs no download."""
    from chest_xray.model import build_model

    model = build_model(weights=None)
    images = np.random.default_rng(4).integers(0, 256, (2, *IMAGE_SIZE, 3)).astype("float32")
    probabilities = model.predict(images, verbose=0)

    assert probabilities.shape == (2, 1)
    assert 0.0 <= probabilities.min() and probabilities.max() <= 1.0


def test_augmentation_preserves_shape_and_changes_pixels():
    from chest_xray.model import augmentation_pipeline

    images = np.random.default_rng(5).integers(0, 256, (2, *IMAGE_SIZE, 3)).astype("float32")
    augmented = np.asarray(augmentation_pipeline()(images, training=True))
    assert augmented.shape == images.shape
    assert not np.allclose(augmented, images)

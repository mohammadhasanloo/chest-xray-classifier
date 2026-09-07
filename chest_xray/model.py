"""EfficientNetB2 with a small classification head."""

from __future__ import annotations

import keras
from keras import layers

from chest_xray.data import IMAGE_SIZE

INPUT_SHAPE = (*IMAGE_SIZE, 3)
FROZEN_LAYERS = 74


def build_model(
    frozen_layers: int = FROZEN_LAYERS,
    learning_rate: float = 1e-5,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Binary classifier over 128x128 RGB chest X-rays.

    Input is expected in [0, 255], not [0, 1]. ``keras.applications.EfficientNet``
    carries its own rescaling and normalisation layers, so images divided by 255
    beforehand would be normalised twice and land in a range the pretrained
    weights were never fitted for.

    ``weights`` is exposed so the architecture can be built and tested without
    pulling the pretrained checkpoint.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    backbone = keras.applications.EfficientNetB2(
        include_top=False, weights=weights, input_shape=INPUT_SHAPE
    )
    for layer in backbone.layers[:frozen_layers]:
        layer.trainable = False

    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability")(x)

    model = keras.Model(inputs, outputs, name="chest_xray_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def augmentation_pipeline() -> keras.Sequential:
    """Geometric augmentation only.

    No colour jitter and no vertical flip: an upside-down chest X-ray is not a
    radiograph anyone will ever be asked to read, and brightness shifts change
    exactly the tissue contrast the model is supposed to be reading.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(30 / 360),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomZoom(0.2),
        ],
        name="augmentation",
    )

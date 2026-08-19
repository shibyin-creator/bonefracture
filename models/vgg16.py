"""VGG-16 backbone and softmax classifier (IEEE Access 2025, Fig. 6).

Frozen ImageNet convolutional stack + Flatten + Dense(ReLU) + Dropout +
Dense(ReLU) + Softmax. Compiled with Adam (lr=5e-4) and categorical
cross-entropy.
"""

from __future__ import annotations

from config import Config


def build_vgg16_softmax(num_classes: int | None = None, image_size: tuple[int, int] = Config.IMAGE_SIZE):
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    num_classes = num_classes or len(Config.MORPHOLOGY_CLASSES)
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = Flatten(name="flatten")(base.output)
    x = Dense(512, activation="relu", name="fc1")(x)
    x = Dropout(0.5, name="dropout")(x)
    x = Dense(256, activation="relu", name="fc2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base.input, outputs=outputs, name="vgg16_fracture_softmax")
    model.compile(
        optimizer=Adam(learning_rate=Config.VGG_LEARNING_RATE),
        loss=Config.LOSS_NAME,
        metrics=["accuracy"],
    )
    return model


def build_vgg16_feature_extractor(pooling: str = "avg"):
    from tensorflow.keras.applications import VGG16

    return VGG16(weights="imagenet", include_top=False, pooling=pooling)

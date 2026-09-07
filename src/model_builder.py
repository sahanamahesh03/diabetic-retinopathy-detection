"""
model_builder.py
-----------------
Builds the CNN model using EfficientNetB0 with transfer learning.

Architecture:
- Base: EfficientNetB0 pretrained on ImageNet (feature extractor)
- Custom head: GlobalAveragePooling -> Dropout -> Dense(5, softmax)

We provide two functions:
- build_model(): builds the model with the base FROZEN (fast initial training)
- unfreeze_model(): unfreezes the top layers of the base for fine-tuning
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

import config


def build_model():
    """
    Builds the full model: EfficientNetB0 base (frozen) + custom
    classification head for our 5 DR classes.
    """
    # Load EfficientNetB0 pretrained on ImageNet.
    # include_top=False: we exclude its original 1000-class classification
    # layer, since we need our own 5-class output instead.
    # input_shape: matches our IMAGE_SIZE (224, 224) with 3 color channels.
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    )

    # Freeze the base model - its pretrained weights won't be updated
    # during initial training. We only train our new custom layers.
    base_model.trainable = False

    # Build the full model by stacking our custom head on top of the base
    inputs = layers.Input(shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS))

    # Pass input through the frozen EfficientNetB0 base.
    # training=False ensures batch-norm layers inside the base behave in
    # inference mode, which is correct/recommended when the base is frozen.
    x = base_model(inputs, training=False)

    # GlobalAveragePooling2D: condenses the base model's output feature
    # maps (a 3D tensor) into a single 1D vector per image, by averaging.
    # This is much lighter than Flatten() and works well with EfficientNet.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout: randomly "turns off" 30% of neurons during training.
    # This is a regularization technique that helps prevent overfitting -
    # especially important since our dataset (2563 training images) is
    # small relative to the model's capacity.
    x = layers.Dropout(0.3)(x)

    # Final classification layer: one output neuron per class (5 total),
    # with softmax activation so outputs form a probability distribution
    # (all 5 values sum to 1.0 - e.g. [0.7, 0.1, 0.1, 0.05, 0.05]).
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Compile the model: defines how it learns.
    # - optimizer: Adam, a reliable general-purpose choice
    # - loss: sparse_categorical_crossentropy, correct choice since our
    #   labels are plain integers (0-4), not one-hot encoded vectors
    # - metrics: track accuracy during training for a quick sanity check
    #   (we'll compute more detailed metrics like F1/ROC-AUC separately
    #   in evaluate.py)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_model(model, base_model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top N layers of the base model for fine-tuning,
    EXCEPT BatchNormalization layers, which stay frozen.

    Why: BatchNorm layers track running statistics from training data.
    With our small batch size (16, due to CPU constraints), unfreezing
    them causes unstable, noisy statistic updates that corrupt the
    pretrained features rather than gently adapting them - a well-known
    issue when fine-tuning EfficientNet with small batches.
    """
    base_model.trainable = True

    # Freeze everything except the last `num_layers_to_unfreeze` layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Even among the unfrozen layers, keep BatchNorm layers frozen
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

    # Re-compile with a much lower learning rate. This is critical -
    # fine-tuning pretrained weights with a normal learning rate would
    # destroy the useful patterns they already learned ("catastrophic
    # forgetting"). A small learning rate makes only gentle adjustments.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Quick test: build the model and print its architecture summary
    print("Building model...")
    model, base_model = build_model()

    print("\nModel summary:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"Trainable parameters (base frozen): {trainable_params:,}")

    print("\nmodel_builder.py test completed successfully!")
"""
augmentation.py
----------------
Data augmentation pipeline. Applies random transformations (flips, rotations,
zoom, brightness/contrast changes) to training images on the fly.

Why this matters for us specifically:
1. Our dataset is small (2563 training images) - augmentation artificially
   increases variety, helping the model generalize instead of memorizing.
2. Our classes are imbalanced (e.g. only 193 "Severe" examples) - augmentation
   helps the model see more varied versions of these rare examples too.

Note: augmentation is applied ONLY to training data, never to validation
or test data - we want to evaluate on real, unaltered images.
"""

import tensorflow as tf
import dr_config as config


def augment_image(image, label):
    """
    Applies a random combination of augmentations to a single image.
    Each transformation has some randomness, so the same image looks
    slightly different every time the model sees it during training.

    Note: images are in [0, 255] range here (EfficientNetB0 handles its
    own internal rescaling), so augmentation ranges are scaled accordingly.
    """

    # Randomly flip the image horizontally (left-right).
    image = tf.image.random_flip_left_right(image)

    # Randomly flip vertically too.
    image = tf.image.random_flip_up_down(image)

    # Randomly rotate the image by 0, 90, 180, or 270 degrees.
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Randomly adjust brightness slightly (+/- 20% of the 0-255 range).
    image = tf.image.random_brightness(image, max_delta=51.0)

    # Randomly adjust contrast slightly (multiplicative, no range change needed).
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # After brightness/contrast adjustments, pixel values might go
    # slightly outside the valid [0, 255] range - clip them back in.
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label


def build_augmented_dataset(df, batch_size=None):
    """
    Builds a training dataset pipeline WITH augmentation applied.
    This mirrors build_dataset() from data_loader.py, but adds the
    augmentation step after loading each image.
    """
    from data_loader import _load_and_preprocess_image  # reuse existing loader

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    filepaths = df["filepath"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=config.RANDOM_SEED)

    # Step 1: load and resize the image (same as before)
    ds = ds.map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Step 2: apply random augmentation on top
    ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


if __name__ == "__main__":
    # Quick test: load a few training images, augment them, verify shapes
    from data_loader import load_labels, split_data

    print("Loading and splitting data...")
    df = load_labels()
    train_df, val_df, test_df = split_data(df)

    print("\nBuilding augmented training dataset...")
    train_ds = build_augmented_dataset(train_df)

    print("Fetching one augmented batch to verify shapes...")
    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        print(f"Pixel value range: [{tf.reduce_min(images).numpy():.3f}, {tf.reduce_max(images).numpy():.3f}]")

    print("\naugmentation.py test completed successfully!")

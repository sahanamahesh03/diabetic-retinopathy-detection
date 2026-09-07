"""
data_loader.py
---------------
Handles reading the dataset CSV, splitting it into train/validation/test
sets, and building TensorFlow data pipelines that load and preprocess
images in batches (memory-efficient - doesn't load all images at once).
"""

import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import dr_config as config  # our centralized settings file


def load_labels():
    """
    Reads train.csv and returns a DataFrame with two columns:
    - filepath: full path to each image file
    - label: the diagnosis class (0-4)
    """
    df = pd.read_csv(config.TRAIN_CSV_PATH)

    # train.csv has 'id_code' (filename without extension) and 'diagnosis'.
    # We build the full image path by adding the folder + '.png' extension.
    df["filepath"] = df["id_code"].apply(
        lambda x: os.path.join(config.TRAIN_IMAGES_DIR, x + ".png")
    )
    df["label"] = df["diagnosis"]

    return df[["filepath", "label"]]


def split_data(df):
    """
    Splits the dataframe into train/validation/test sets.
    Uses 'stratify' so each split has the same class proportions as the
    full dataset - critical given our class imbalance (e.g. Severe DR
    is rare, so we must ensure it appears in all three splits fairly).
    """
    # First split: separate out the training set
    train_df, temp_df = train_test_split(
        df,
        train_size=config.TRAIN_SPLIT,
        stratify=df["label"],
        random_state=config.RANDOM_SEED
    )

    # Second split: divide the remainder into validation and test
    # (temp_df is 30% of data; we split that into 15%/15% of the original,
    # which is 50/50 of this remaining chunk)
    relative_val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED
    )

    # Save these splits to CSV so we can reuse the exact same split later
    # (for evaluation, Grad-CAM, etc.) without re-randomizing
    train_df.to_csv(os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.SPLITS_DIR, "test.csv"), index=False)

    return train_df, val_df, test_df


def _load_and_preprocess_image(filepath, label):
    """
    Loads a single image from disk, resizes it, and normalizes pixel values.
    This function is applied to every image automatically as the pipeline runs.
    """
    # Read the raw file bytes from disk
    image = tf.io.read_file(filepath)

    # Decode PNG bytes into an actual image tensor (3 channels = RGB)
    image = tf.image.decode_png(image, channels=config.IMAGE_CHANNELS)

    # Resize every image to the same fixed size (224x224) the model expects
    image = tf.image.resize(image, config.IMAGE_SIZE)

    # NOTE: We do NOT divide by 255 here. EfficientNetB0 has its own
    # built-in preprocessing/rescaling layer and expects raw pixel
    # values in the [0, 255] range as input.

    return image, label


def build_dataset(df, batch_size=None, shuffle=False):
    """
    Converts a DataFrame (filepath, label columns) into a tf.data.Dataset
    that loads and preprocesses images on the fly, in batches.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    filepaths = df["filepath"].values
    labels = df["label"].values

    # Create a dataset of (filepath, label) pairs
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        # Shuffle the order of images each epoch - important for training
        # so the model doesn't learn any accidental ordering patterns
        ds = ds.shuffle(buffer_size=len(df), seed=config.RANDOM_SEED)

    # Apply our image loading/preprocessing function to every item.
    # num_parallel_calls lets TensorFlow load multiple images at once
    # using available CPU cores, speeding up the pipeline.
    ds = ds.map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Group images into batches
    ds = ds.batch(batch_size)

    # Prefetch: prepares the next batch while the current one is being
    # used for training, reducing idle time
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


if __name__ == "__main__":
    # Quick test: run this file directly to verify everything works
    print("Loading labels...")
    df = load_labels()
    print(f"Total labeled images: {len(df)}")

    print("\nSplitting into train/val/test...")
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\nBuilding train dataset pipeline...")
    train_ds = build_dataset(train_df, shuffle=True)

    print("Fetching one batch to verify shapes...")
    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")  # (batch_size, 224, 224, 3)
        print(f"Batch label shape: {labels.shape}")   # (batch_size,)

    print("\ndata_loader.py test completed successfully!")

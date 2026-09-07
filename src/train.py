"""
train.py
---------
Trains the diabetic retinopathy classification model in two phases:
1. Train the custom head with the EfficientNetB0 base frozen.
2. Fine-tune by unfreezing the top layers of the base at a low learning rate.

Handles class imbalance using computed class weights.
Uses callbacks to save the best model, stop early if no improvement,
and reduce learning rate on plateau.
"""

import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import dr_config as config
from data_loader import load_labels, split_data, build_dataset
from augmentation import build_augmented_dataset
from model_builder import build_model, unfreeze_model


def get_class_weights(train_df):
    """
    Computes class weights to counter our class imbalance.
    Classes with fewer examples (e.g. Severe DR, 193 images) get a
    higher weight, so mistakes on them are penalized more during training.
    """
    labels = train_df["label"].values
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weight_dict = dict(zip(classes, weights))

    print("Computed class weights:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls} ({config.CLASS_NAMES[cls]}): weight = {weight:.3f}")

    return class_weight_dict


def get_callbacks(phase_name):
    """
    Builds the list of training callbacks.
    phase_name is used to name the saved checkpoint file distinctly
    for phase 1 (frozen) vs phase 2 (fine-tuning).
    """
    checkpoint_path = os.path.join(
        config.SAVED_MODELS_DIR, f"best_model_{phase_name}.keras"
    )

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks


def main():
    print("=" * 60)
    print("STEP 1: Loading and splitting data")
    print("=" * 60)
    df = load_labels()
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n" + "=" * 60)
    print("STEP 2: Computing class weights")
    print("=" * 60)
    class_weights = get_class_weights(train_df)

    print("\n" + "=" * 60)
    print("STEP 3: Building data pipelines")
    print("=" * 60)
    train_ds = build_augmented_dataset(train_df)
    val_ds = build_dataset(val_df, shuffle=False)

    print("\n" + "=" * 60)
    print("STEP 4: Building model (base frozen)")
    print("=" * 60)
    model, base_model = build_model()
    model.summary()

    print("\n" + "=" * 60)
    print("PHASE 1: Training custom head (base frozen)")
    print("=" * 60)
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1_frozen")
    )

    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    model = unfreeze_model(model, base_model, num_layers_to_unfreeze=20)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2_finetuned")
    )

    print("\n" + "=" * 60)
    print("STEP 5: Saving final model")
    print("=" * 60)
    final_model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

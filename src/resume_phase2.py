"""
resume_phase2.py
------------------
Loads the Phase 1 checkpoint (base frozen, already trained) and runs
ONLY Phase 2 fine-tuning with the corrected unfreeze_model() function
(BatchNorm layers kept frozen). Saves results under new filenames so
we don't confuse them with the earlier failed run.
"""

import os
import tensorflow as tf

import dr_config as config
from data_loader import load_labels, split_data, build_dataset
from augmentation import build_augmented_dataset
from model_builder import unfreeze_model
from train import get_class_weights, get_callbacks


def main():
    print("=" * 60)
    print("STEP 1: Loading and splitting data (same split as before)")
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
    print("STEP 4: Loading Phase 1 checkpoint")
    print("=" * 60)
    phase1_path = os.path.join(config.SAVED_MODELS_DIR, "best_model_phase1_frozen.keras")
    model = tf.keras.models.load_model(phase1_path)
    print(f"Loaded model from: {phase1_path}")

    # Find the EfficientNetB0 layer inside the loaded model so we can
    # pass it to unfreeze_model() just like in the original train.py
    base_model = model.get_layer("efficientnetb0")

    print("\n" + "=" * 60)
    print("STEP 5: Applying CORRECTED unfreeze (BatchNorm stays frozen)")
    print("=" * 60)
    model = unfreeze_model(model, base_model, num_layers_to_unfreeze=20)
    model.summary()

    print("\n" + "=" * 60)
    print("PHASE 2 (CORRECTED): Fine-tuning")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2_finetuned_v2")
    )

    print("\n" + "=" * 60)
    print("STEP 6: Saving final model")
    print("=" * 60)
    final_model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model_v2.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nPhase 2 (corrected) training completed successfully!")


if __name__ == "__main__":
    main()

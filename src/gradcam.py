"""
gradcam.py
-----------
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) to
visualize which regions of a retinal image most influenced the model's
prediction. This provides explainability - showing the model isn't a
"black box" but focuses on clinically relevant regions.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config


def load_and_prepare_image(filepath):
    """
    Loads a single image and prepares it for the model, matching the
    exact preprocessing used during training (resize, no manual scaling -
    EfficientNetB0 handles that internally).
    Returns both the preprocessed array (for the model) and the original
    resized image (for display purposes).
    """
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=config.IMAGE_CHANNELS)
    image = tf.image.resize(image, config.IMAGE_SIZE)

    # Keep a copy for display (as a plain numpy array, uint8 0-255)
    display_image = image.numpy().astype(np.uint8)

    # Add a batch dimension: model expects shape (1, 224, 224, 3), not (224, 224, 3)
    model_input = tf.expand_dims(image, axis=0)

    return model_input, display_image


def make_gradcam_heatmap(model, image_array, last_conv_layer_name="top_conv"):
    """
    Generates the raw Grad-CAM heatmap for a given image.

    last_conv_layer_name: EfficientNetB0's final convolutional layer,
    where feature maps are most semantically rich (best for Grad-CAM).
    """
    # Get a reference to the EfficientNetB0 sub-model inside our full model
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Build a helper model that outputs BOTH:
    # 1. The last conv layer's feature maps
    # 2. The base model's final output (before our custom head)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(image_array)

        # Pass base_model's output through the REST of our custom head
        # (GlobalAveragePooling -> Dropout -> Dense) to get final predictions.
        # We manually replicate these layers since grad_model only goes
        # up to the base model's output.
        x = model.get_layer("global_average_pooling2d")(base_outputs)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense")(x)

        # Get the predicted class (highest probability) and its score
        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    # Compute gradients of the predicted class score with respect to
    # the last conv layer's feature maps - this tells us how much each
    # feature map "matters" for this specific prediction
    grads = tape.gradient(class_score, conv_outputs)

    # Average gradients across width and height - this gives one
    # importance weight per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its importance, then sum them all up
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: we only care about features that POSITIVELY influenced the
    # prediction (negative influence isn't useful for visualization)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1] range for consistent visualization
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(predicted_class.numpy()), predictions.numpy()[0]


def overlay_heatmap(heatmap, display_image, alpha=0.4):
    """
    Resizes the heatmap to match the original image size, applies a
    color map (jet: blue=low, red=high importance), and blends it with
    the original image.
    """
     # Resize heatmap from its small native size (e.g. 7x7) up to 224x224.
    # Using bicubic interpolation (instead of default) gives smooth
    # gradients instead of hard blocky edges.
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], config.IMAGE_SIZE, method="bicubic"
    ).numpy().squeeze()

    # Clip because bicubic interpolation can slightly overshoot [0,1]
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    # Apply a light Gaussian blur for additional smoothness
    from scipy.ndimage import gaussian_filter
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=8)

    # Convert heatmap values [0,1] into RGB colors using the 'jet' colormap
    jet = plt.colormaps["jet"]
    jet_colors = jet(heatmap_resized)[:, :, :3]  # drop alpha channel
    jet_heatmap = (jet_colors * 255).astype(np.uint8)

    # Blend original image and colored heatmap
    overlayed = (jet_heatmap * alpha + display_image * (1 - alpha)).astype(np.uint8)

    return overlayed


def generate_gradcam_for_image(model, filepath, save_path):
    """
    Full pipeline for one image: load, predict, generate heatmap,
    overlay, and save a side-by-side comparison figure.
    """
    model_input, display_image = load_and_prepare_image(filepath)
    heatmap, predicted_class, probs = make_gradcam_heatmap(model, model_input)
    overlayed = overlay_heatmap(heatmap, display_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlayed)
    predicted_label = config.CLASS_NAMES[predicted_class]
    confidence = probs[predicted_class] * 100
    axes[2].set_title(f"Overlay\nPredicted: {predicted_label} ({confidence:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return predicted_class, probs


def main():
    print("Loading trained model...")
    model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model.keras")
    model = tf.keras.models.load_model(model_path)

    print("Loading test set...")
    test_df = pd.read_csv(os.path.join(config.SPLITS_DIR, "test.csv"))

    # Pick a handful of sample images to visualize - one per class if
    # possible, so we can see Grad-CAM behavior across all severity levels
    print("\nSelecting sample images (one per class where available)...")
    samples = []
    for class_idx in range(config.NUM_CLASSES):
        class_samples = test_df[test_df["label"] == class_idx]
        if len(class_samples) > 0:
            samples.append(class_samples.iloc[0])

    print(f"Generating Grad-CAM visualizations for {len(samples)} sample images...")
    for i, row in enumerate(samples):
        filepath = row["filepath"]
        true_label = config.CLASS_NAMES[row["label"]]
        save_path = os.path.join(
            config.GRADCAM_DIR, f"gradcam_sample_{i}_true_{true_label}.png"
        )

        predicted_class, probs = generate_gradcam_for_image(model, filepath, save_path)
        predicted_label = config.CLASS_NAMES[predicted_class]

        print(f"  Image {i+1}: True={true_label} | Predicted={predicted_label} | Saved to {save_path}")

    print("\nGrad-CAM generation completed successfully!")


if __name__ == "__main__":
    main()
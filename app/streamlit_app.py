"""
streamlit_app.py
------------------
A simple web app for diabetic retinopathy prediction with Grad-CAM
explainability. Upload a retinal image, get the predicted severity
class, confidence score, and a visual heatmap of what the model focused on.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import dr_config as config


@st.cache_resource
def load_trained_model():
    model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model.keras")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize(config.IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    display_image = image_array.astype(np.uint8)
    model_input = np.expand_dims(image_array, axis=0)
    return model_input, display_image


def make_gradcam_heatmap(model, image_array, last_conv_layer_name="top_conv"):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(image_array)
        x = model.get_layer("global_average_pooling2d")(base_outputs)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense")(x)

        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(predicted_class.numpy()), predictions.numpy()[0]


def overlay_heatmap(heatmap, display_image, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], config.IMAGE_SIZE, method="bicubic"
    ).numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=8)

    jet = plt.colormaps["jet"]
    jet_colors = jet(heatmap_resized)[:, :, :3]
    jet_heatmap = (jet_colors * 255).astype(np.uint8)

    overlayed = (jet_heatmap * alpha + display_image * (1 - alpha)).astype(np.uint8)
    return overlayed


st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")

st.title("Diabetic Retinopathy Detection with Grad-CAM")
st.write(
    "Upload a retinal fundus image to get a diabetic retinopathy "
    "severity prediction, along with a Grad-CAM visualization showing "
    "which regions of the image most influenced the model's decision."
)

uploaded_file = st.file_uploader(
    "Upload a retinal image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    with st.spinner("Loading model and running prediction..."):
        model = load_trained_model()
        model_input, display_image = preprocess_image(pil_image)
        heatmap, predicted_class, probs = make_gradcam_heatmap(model, model_input)
        overlayed = overlay_heatmap(heatmap, display_image)

    predicted_label = config.CLASS_NAMES[predicted_class]
    confidence = probs[predicted_class] * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Uploaded Image")
        st.image(display_image, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        jet_display = plt.colormaps["jet"]
        heatmap_colored = (jet_display(heatmap)[:, :, :3] * 255).astype(np.uint8)
        st.image(heatmap_colored, use_container_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlayed, use_container_width=True)

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.1f}%")

    st.subheader("Class Probabilities")
    prob_dict = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)}
    st.bar_chart(prob_dict)

    st.info(
        "This tool is for educational/demonstration purposes as part of "
        "a final-year academic project, and is NOT a substitute for "
        "professional medical diagnosis."
    )
else:
    st.info("Please upload a retinal image to get started.")

"""
streamlit_app.py
------------------
A simple web app for diabetic retinopathy prediction with Grad-CAM
explainability. Upload a retinal image, get the predicted severity
class, confidence score, a visual heatmap of what the model focused on,
plain-language explanation, nearby ophthalmologist search, and a
downloadable PDF report.
"""

import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from fpdf import FPDF

import dr_config as config


DR_EXPLANATIONS = {
    "No_DR": "No signs of diabetic retinopathy detected. Continue with regular eye checkups as recommended by your doctor.",
    "Mild": "Early-stage diabetic retinopathy. Small areas of damage (microaneurysms) are present in the retina's blood vessels. Regular monitoring is advised.",
    "Moderate": "Moderate diabetic retinopathy. More blood vessels are affected, which may impact vision over time. Follow-up with an eye specialist is recommended.",
    "Severe": "Severe diabetic retinopathy. Significant blood vessel damage has occurred, raising the risk of vision loss. Prompt medical attention is strongly advised.",
    "Proliferative": "Proliferative diabetic retinopathy - the most advanced stage. Abnormal new blood vessels are growing, which can cause serious vision complications. Immediate medical care is critical."
}

FAQ_RESPONSES = {
    ("what is diabetic retinopathy", "what is dr", "diabetic retinopathy"): 
        "Diabetic retinopathy is an eye condition caused by damage to blood vessels in the retina due to high blood sugar levels over time. It's a common complication of diabetes and can lead to vision loss if untreated.",
    ("what is grad-cam", "gradcam", "what is the heatmap", "heatmap mean"):
        "Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that shows which parts of an image most influenced the AI model's prediction. Red/yellow areas mean the model focused there most; blue/green areas had less influence.",
    ("what should i do", "next steps", "what now", "what do i do"):
        "If the result shows Mild or Moderate, schedule a checkup with an eye specialist. If it shows Severe or Proliferative, please consult an ophthalmologist as soon as possible. You can use the ophthalmologist finder above to locate nearby specialists.",
    ("how accurate", "accuracy", "how good is this model", "reliable"):
        "This model was tested on unseen data and achieved about 74% overall accuracy and a 0.94 ROC-AUC score. However, this is a student project for educational purposes and should not replace a professional medical diagnosis.",
    ("what are the stages", "severity levels", "classes", "categories"):
        "There are 5 stages: No DR (no signs), Mild, Moderate, Severe, and Proliferative (most advanced). Each stage reflects increasing damage to the retina's blood vessels.",
    ("who made this", "who built this", "about this project"):
        "This tool is a final-year academic project using a CNN (EfficientNetB0) with Grad-CAM explainability to detect diabetic retinopathy from retinal images.",
}

def get_faq_response(user_question):
    question_lower = user_question.lower()
    for keywords, response in FAQ_RESPONSES.items():
        if any(keyword in question_lower for keyword in keywords):
            return response
    return "I'm not sure about that one. Try asking about diabetic retinopathy, the Grad-CAM heatmap, next steps, model accuracy, or the severity stages."


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


def is_likely_retinal_image(pil_image):
    """
    Stricter heuristic check: retinal fundus images have a near-black
    background with a distinctly circular, warm-toned region in the center.
    Checks that the image corners are very dark (near black) while the
    center is warm-toned and significantly brighter.
    """
    img = pil_image.convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)

    # Check the four corners - real fundus images have solid black corners
    corner_size = 10
    corners = np.concatenate([
        arr[:corner_size, :corner_size, :].flatten(),
        arr[:corner_size, -corner_size:, :].flatten(),
        arr[-corner_size:, :corner_size, :].flatten(),
        arr[-corner_size:, -corner_size:, :].flatten()
    ])
    corner_brightness = corners.mean()

    # Center region color and brightness
    center = arr[35:65, 35:65, :]
    center_r = center[:, :, 0].mean()
    center_b = center[:, :, 2].mean()
    center_brightness = center.mean()

    corners_are_dark = corner_brightness < 30
    center_is_warm = center_r > center_b
    center_is_bright_enough = center_brightness > 40
    significant_contrast = (center_brightness - corner_brightness) > 30

    return corners_are_dark and center_is_warm and center_is_bright_enough and significant_contrast
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

    if not is_likely_retinal_image(pil_image):
        st.error(
            "This doesn't look like a retinal fundus image. "
            "Please upload a clear retinal scan for accurate results."
        )
        st.image(pil_image, caption="Uploaded image", width=300)
        st.stop()

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

    st.caption(
        "Heatmap color guide: 🔵 Blue/Green = low influence on the prediction  |  "
        "🟡 Yellow = moderate influence  |  🔴 Red = high influence. "
        "Red and yellow regions show where the model focused most when making its decision."
    )

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.1f}%")
    st.write(f"**What this means:** {DR_EXPLANATIONS[predicted_label]}")

    if predicted_label in ["Severe", "Proliferative"]:
        st.error(
            "This result suggests a serious condition. "
            "Please consult an ophthalmologist as soon as possible for a proper diagnosis and care."
        )
    elif predicted_label in ["Mild", "Moderate"]:
        st.warning(
            "This result suggests early to moderate signs of diabetic retinopathy. "
            "It's recommended to schedule a checkup with an eye care professional."
        )

    if predicted_label != "No_DR":
        st.subheader("Find an Ophthalmologist Near You")
        city = st.text_input("Enter your city or area:")
        if city:
            maps_query = f"ophthalmologist near {city}".replace(" ", "+")
            maps_url = f"https://www.google.com/maps/search/{maps_query}"
            st.markdown(f"[Search ophthalmologists near {city} on Google Maps]({maps_url})")

    st.subheader("Class Probabilities")
    prob_dict = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)}
    st.bar_chart(prob_dict)

    def generate_pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Diabetic Retinopathy Detection Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Predicted Class: {predicted_label}", ln=True)
        pdf.cell(0, 10, f"Confidence: {confidence:.1f}%", ln=True)
        pdf.ln(5)

        pdf.multi_cell(0, 8, f"Explanation: {DR_EXPLANATIONS[predicted_label]}")
        pdf.ln(5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_overlay:
            Image.fromarray(overlayed).save(tmp_overlay.name)
            pdf.image(tmp_overlay.name, x=10, w=90)

        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 6, "This tool is for educational/demonstration purposes as part of a final-year academic project, and is NOT a substitute for professional medical diagnosis.")

        return pdf.output(dest="S")

    pdf_bytes = generate_pdf_report()
    st.download_button(
        label="Download PDF Report",
        data=bytes(pdf_bytes),
        file_name=f"DR_report_{predicted_label}.pdf",
        mime="application/pdf"
    )

    st.caption(
        "This tool is for educational/demonstration purposes as part of a "
        "final-year academic project, and is NOT a substitute for professional medical diagnosis."
    )
else:
    st.info("Please upload a retinal image to get started.")

st.markdown("---")
st.subheader("Ask a Question")
st.write("Have a question about diabetic retinopathy or this tool? Ask below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Type your question here:", key="faq_input")

if user_question:
    answer = get_faq_response(user_question)
    st.session_state.chat_history.append((user_question, answer))

for question, answer in reversed(st.session_state.chat_history):
    st.write(f"**You:** {question}")
    st.write(f"**Bot:** {answer}")
    st.write("")

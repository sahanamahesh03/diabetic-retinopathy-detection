"""
streamlit_app.py
------------------
Diabetic Retinopathy Detection with Grad-CAM explainability.
Three-step flow: Home -> Upload -> Results, using session-state page navigation.
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
    "Proliferative": "Proliferative diabetic retinopathy, the most advanced stage. Abnormal new blood vessels are growing, which can cause serious vision complications. Immediate medical care is critical."
}

SEVERITY_COLORS = {
    "No_DR": "#1B5E20",
    "Mild": "#B7791F",
    "Moderate": "#C2410C",
    "Severe": "#B91C1C",
    "Proliferative": "#7F1D1D"
}

FAQ_RESPONSES = {
    ("what is diabetic retinopathy", "what is dr", "diabetic retinopathy"):
        "Diabetic retinopathy is an eye condition caused by damage to blood vessels in the retina due to high blood sugar levels over time.",
    ("what is grad-cam", "gradcam", "what is the heatmap", "heatmap mean"):
        "Grad-CAM shows which parts of an image most influenced the model's prediction. Red/yellow areas had high influence; blue/green areas had low influence.",
    ("what should i do", "next steps", "what now", "what do i do"):
        "If the result shows Mild or Moderate, schedule a checkup with an eye specialist. If it shows Severe or Proliferative, consult an ophthalmologist as soon as possible.",
    ("how accurate", "accuracy", "how good is this model", "reliable"):
        "This model achieved about 74% overall accuracy and a 0.94 ROC-AUC score on unseen test data. This is a student project and should not replace professional diagnosis.",
    ("what are the stages", "severity levels", "classes", "categories"):
        "There are 5 stages: No DR, Mild, Moderate, Severe, and Proliferative (most advanced).",
    ("who made this", "who built this", "about this project"):
        "This is a final-year academic project using a CNN (EfficientNetB0) with Grad-CAM explainability.",
}

def get_faq_response(user_question):
    question_lower = user_question.lower()
    for keywords, response in FAQ_RESPONSES.items():
        if any(keyword in question_lower for keyword in keywords):
            return response
    return "Try asking about diabetic retinopathy, the heatmap, next steps, accuracy, or the severity stages."


@st.cache_resource
def load_trained_model():
    model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model.keras")
    return tf.keras.models.load_model(model_path)


def is_likely_retinal_image(pil_image):
    img = pil_image.convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)
    corner_size = 10
    corners = np.concatenate([
        arr[:corner_size, :corner_size, :].flatten(),
        arr[:corner_size, -corner_size:, :].flatten(),
        arr[-corner_size:, :corner_size, :].flatten(),
        arr[-corner_size:, -corner_size:, :].flatten()
    ])
    corner_brightness = corners.mean()
    center = arr[35:65, 35:65, :]
    center_r = center[:, :, 0].mean()
    center_b = center[:, :, 2].mean()
    center_brightness = center.mean()
    return (corner_brightness < 60 and center_r > center_b and
            center_brightness > 30 and (center_brightness - corner_brightness) > 15)


def preprocess_image(pil_image):
    image = pil_image.convert("RGB").resize(config.IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    display_image = image_array.astype(np.uint8)
    model_input = np.expand_dims(image_array, axis=0)
    return model_input, display_image


def make_gradcam_heatmap(model, image_array, last_conv_layer_name="top_conv"):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(inputs=base_model.input, outputs=[last_conv_layer.output, base_model.output])
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
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], config.IMAGE_SIZE, method="bicubic").numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=8)
    jet = plt.colormaps["jet"]
    jet_colors = jet(heatmap_resized)[:, :, :3]
    jet_heatmap = (jet_colors * 255).astype(np.uint8)
    return (jet_heatmap * alpha + display_image * (1 - alpha)).astype(np.uint8)


def generate_pdf_report(predicted_label, confidence, overlayed):
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
    pdf.multi_cell(0, 6, "Educational/demonstration project - not a substitute for professional medical diagnosis.")
    return pdf.output(dest="S")


st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
        background-color: #F2F5F8 !important;
    }
    [data-testid="stAppViewContainer"] * {
        color: #1B2430 !important;
    }
    .hero {
        background-color: #12283B; padding: 2.2rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .hero h1, .hero p, .hero * { color: #FFFFFF !important; }
    .page-header {
        background-color: #12283B; padding: 1.6rem 2rem; border-radius: 10px; margin-bottom: 1.3rem;
    }
    .page-header h1, .page-header p, .page-header * { color: #FFFFFF !important; }
    .card {
        background-color: #FFFFFF; padding: 1.4rem 1.6rem; border-radius: 10px;
        border: 1px solid #D7DEE5; margin-bottom: 1rem;
    }
    .badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 6px;
        font-weight: 700; font-size: 0.95rem;
    }
    .badge, .badge * { color: #FFFFFF !important; }
    .stButton>button {
        background-color: #1B5E7A !important; border-radius: 8px;
        padding: 0.55rem 1.3rem; border: none; font-weight: 600;
    }
    .stButton>button, .stButton>button * { color: #FFFFFF !important; }
    .stButton>button:hover { background-color: #12283B !important; }
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1>Diabetic Retinopathy Detection</h1>
        <p>An AI-based screening assistant that analyzes retinal fundus images to detect signs of
        diabetic retinopathy, with a visual explanation of the model's decision.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Step 1</h3><p>Upload a retinal fundus image.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>Step 2</h3><p>The model analyzes the image and predicts severity.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>Step 3</h3><p>View the result with a visual explanation and report.</p></div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Start Detection"):
        st.session_state.page = "upload"
        st.rerun()

    st.caption("Educational/demonstration project as part of a final-year academic project. Not a substitute for professional medical diagnosis.")


# ============================================================
# UPLOAD PAGE
# ============================================================
elif st.session_state.page == "upload":
    st.markdown("""
    <div class="page-header">
        <h1>Upload Retinal Image</h1>
        <p>Upload a clear retinal fundus photo in JPG or PNG format.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Preview", width=350)

        if st.button("Analyze Image"):
            if not is_likely_retinal_image(pil_image):
                st.error("This doesn't look like a retinal fundus image. Please upload a clear retinal scan.")
            else:
                with st.spinner("Analyzing image..."):
                    model = load_trained_model()
                    model_input, display_image = preprocess_image(pil_image)
                    heatmap, predicted_class, probs = make_gradcam_heatmap(model, model_input)
                    overlayed = overlay_heatmap(heatmap, display_image)

                predicted_label = config.CLASS_NAMES[predicted_class]
                confidence = probs[predicted_class] * 100

                st.session_state.result = {
                    "display_image": display_image,
                    "heatmap": heatmap,
                    "overlayed": overlayed,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "probs": probs,
                }
                st.session_state.page = "results"
                st.rerun()

    st.write("")
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================
elif st.session_state.page == "results":
    r = st.session_state.result
    if r is None:
        st.warning("No result to show yet.")
        if st.button("Go to Upload"):
            st.session_state.page = "upload"
            st.rerun()
    else:
        st.markdown("""
        <div class="page-header">
            <h1>Detection Result</h1>
            <p>Prediction, explainability, and recommended next steps.</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Uploaded Image")
            st.image(r["display_image"], use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Heatmap")
            jet_display = plt.colormaps["jet"]
            heatmap_colored = (jet_display(r["heatmap"])[:, :, :3] * 255).astype(np.uint8)
            st.image(heatmap_colored, use_container_width=True)
        with col3:
            st.subheader("Overlay")
            st.image(r["overlayed"], use_container_width=True)

        st.caption("Heatmap guide: blue/green = low influence, yellow = moderate, red = high influence on the prediction.")

        predicted_label = r["predicted_label"]
        confidence = r["confidence"]
        color = SEVERITY_COLORS[predicted_label]

        st.markdown(f"""
        <div class="card">
            <span class="badge" style="background-color:{color};">{predicted_label}</span>
            <span style="margin-left:12px; font-size:1.1rem;"><b>Confidence: </b>{confidence:.1f}%</span>
            <p style="margin-top:12px;">{DR_EXPLANATIONS[predicted_label]}</p>
        </div>
        """, unsafe_allow_html=True)

        if predicted_label in ["Severe", "Proliferative"]:
            st.error("This result suggests a serious condition. Please consult an ophthalmologist as soon as possible.")
        elif predicted_label in ["Mild", "Moderate"]:
            st.warning("This result suggests early to moderate signs of diabetic retinopathy. Schedule a checkup with an eye care professional.")

        if predicted_label != "No_DR":
            st.subheader("Find an Ophthalmologist Near You")
            city = st.text_input("Enter your city or area:")
            if city:
                maps_query = f"ophthalmologist near {city}".replace(" ", "+")
                st.markdown(f"[Search ophthalmologists near {city} on Google Maps](https://www.google.com/maps/search/{maps_query})")

        st.subheader("Class Probabilities")
        prob_dict = {config.CLASS_NAMES[i]: float(r["probs"][i]) for i in range(config.NUM_CLASSES)}
        st.bar_chart(prob_dict)

        pdf_bytes = generate_pdf_report(predicted_label, confidence, r["overlayed"])
        st.download_button("Download PDF Report", data=bytes(pdf_bytes), file_name=f"DR_report_{predicted_label}.pdf", mime="application/pdf")

        st.markdown("---")
        st.subheader("Ask a Question")
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
        if st.button("Analyze Another Image"):
            st.session_state.result = None
            st.session_state.page = "upload"
            st.rerun()

        st.caption("Educational/demonstration project as part of a final-year academic project. Not a substitute for professional medical diagnosis.")

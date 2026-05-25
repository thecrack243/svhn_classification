import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# COMPACT CUSTOM CSS
# =========================
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
    .icon { font-family: 'Font Awesome 6 Free'; font-weight: 900; }
    </style>
    <style>
    /* Eliminate default padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 95rem;
    }
    /* Tighten all margins */
    h1 { margin-bottom: 0.2rem !important; font-size: 2rem !important; }
    h2 { margin-bottom: 0.3rem !important; font-size: 1.3rem !important; }
    h3 { margin-bottom: 0.3rem !important; font-size: 1.1rem !important; }
    p { margin-bottom: 0.3rem !important; }
    /* Reduce spacing between elements */
    .stMarkdown { margin-bottom: 0.5rem !important; }
    /* Compact buttons */
    .stButton>button {
        padding: 0.4rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    /* Tight metric cards */
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    /* Progress bars compact */
    .stProgress > div > div > div > div {
        height: 12px;
    }
    /* Reduce chart margins */
    .stBarChart { margin-top: -1rem !important; }
    /* Canvas container tight */
    .canvas-container iframe {
        border-radius: 8px;
    }
    /* Horizontal rule spacing */
    hr { margin: 0.8rem 0 !important; }
    /* Hide empty space in columns */
    .row-widget.stButton { margin-top: 0.2rem; }
    </style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.title("Digit Recognizer")
st.caption("Deep Learning Digit Classification using CNN and SVHN Dataset")

st.markdown("---")

# =========================
# MODEL LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    possible_paths = [
        "../models/svhn_cnn.h5", "./models/svhn_cnn.h5",
        "models/svhn_cnn.h5", "svhn_cnn.h5",
        "../models/svhn_cnn.keras", "./models/svhn_cnn.keras",
    ]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if model_path is None:
        st.error("Model not found in `models/` directory"); st.stop()
    tf.get_logger().setLevel('ERROR')
    return tf.keras.models.load_model(model_path)

model = load_model()

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image_data):
    if image_data is None or image_data[:, :, :3].sum() == 0:
        return None
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("RGB")
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================
# SIDEBAR - APP INFO & SETTINGS
# =========================
with st.sidebar:
    st.markdown('<h2><i class="fa-solid fa-circle-info"></i> About</h2>', unsafe_allow_html=True)
    st.markdown("""
    **Digit Recognizer** uses a Convolutional Neural Network (CNN) 
    trained on the [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) 
    to classify handwritten digits (0-9).
    
    ### How to use:
    1. Draw a digit in the canvas
    2. Click **Predict Digit**
    3. View results and confidence scores
    
    ### Model Details:
    - **Architecture**: CNN
    - **Input**: 32×32 RGB images
    - **Classes**: 10 (digits 0-9)
    - **Dataset**: Street View House Numbers
    """)
    
    st.markdown("---")
    st.caption("v1.0 | © 2026 Emmanuel Ilunga")
    
    # Clear cache button for debugging
    if st.button(" Clear Model Cache", type="secondary"):
        st.cache_resource.clear()
        st.rerun()

# =========================
# MAIN LAYOUT - 3 COLUMNS
# =========================
left_col, mid_col, right_col = st.columns([1.2, 1, 1.2], gap="small")

# =========================
# LEFT → CANVAS + BUTTONS
# =========================
with left_col:
    st.markdown('<h2><i class="fa-solid fa-pen-to-square"></i> Draw</h2>', unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )
    
    # Buttons side by side, no gaps
    predict_btn = st.button("Predict Digit")

# =========================
# MIDDLE → METRICS (vertical stack)
# =========================
with mid_col:
    st.markdown('<h2><i class="fa-solid fa-bullseye"></i> Result</h2>', unsafe_allow_html=True)
    
    # Placeholder state
    if not predict_btn:
        st.metric("Digit", "—")
        st.metric("Confidence", "—")
        st.markdown("**Top 3:**")
        c1, c2, c3 = st.columns(3)
        c1.metric("#1", "—"); c2.metric("#2", "—"); c3.metric("#3", "—")
    
    # Prediction state
    elif predict_btn and canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
        with st.spinner("Thinking..."):
            img = preprocess_image(canvas_result.image_data)
            if img is not None:
                prediction = model.predict(img, verbose=0)[0]
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                
                # Big metrics stacked
                st.metric("Digit", predicted_class, 
                         delta="" if confidence > 0.8 else None)
                st.metric("Confidence", f"{confidence*100:.1f}%",
                         delta_color="normal" if confidence > 0.8 else "inverse")
                
                # Top 3 horizontal
                st.markdown("**Top 3:**")
                top3 = np.argsort(prediction)[-3:][::-1]
                c1, c2, c3 = st.columns(3)
                for col, digit, prob in zip([c1, c2, c3], top3, prediction[top3]):
                    col.metric(f"{digit}", f"{prob*100:.0f}%")
                
                # Status indicator
                if confidence > 0.8:
                    st.success(" ✅ High confidence")
                elif confidence > 0.5:
                    st.warning(" ⚠️ Uncertain")
                else:
                    st.error(" ❌ Low confidence")
            else:
                st.error("Invalid image")
    else:
        st.warning("Draw first!")

# =========================
# RIGHT → PROBABILITY CHART
# =========================
with right_col:
    st.markdown('<h2><i class="fa-solid fa-chart-simple"></i> Probabilities</h2>', unsafe_allow_html=True)
    
    if predict_btn and 'prediction' in locals():
        probs = prediction
        
        # Compact probability bars
        for i in range(10):
            prob_pct = probs[i] * 100
            is_predicted = (i == predicted_class)
            color = " " if is_predicted else "  "
            bar_col, text_col = st.columns([4, 1], gap="small")
            with bar_col:
                st.progress(float(probs[i]), text=f"{color} {i}")
            with text_col:
                st.markdown(f"**{prob_pct:.1f}%**", unsafe_allow_html=True)
    else:
        # Empty state chart
        empty_data = {"Digit": list(range(10)), "Probability": [0.1]*10}
        st.bar_chart(empty_data, x="Digit", y="Probability", 
                    color="#cccccc", height=180)
        st.caption("Probability distribution will appear here")


# =========================
# FOOTER (single line)
# =========================
st.markdown("---")
st.caption("© 2026 Emmanuel Ilunga | Deep Learning Project | Streamlit + TensorFlow")
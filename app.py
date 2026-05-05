"""
Pepper / Chili Disease Detector — Streamlit Web App

Upload a pepper leaf image → get disease prediction →
confidence score → farmer-friendly explanation and action plan.
"""

import os
import json
import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌶️ Pepper Disease Detector",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES    = ['Bacterial Spot', 'Healthy']
MODEL_PATH     = 'models/pepper_model.pth'
INFO_PATH      = 'disease_info.json'
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
DEVICE         = torch.device('cpu')   # Streamlit Cloud doesn't have GPU; CPU is fine for inference

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    /* Disease card (red for disease, green for healthy) */
    .disease-card {
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #e74c3c;
        background-color: #fdf0ee;
        margin: 10px 0;
    }
    .healthy-card {
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #27ae60;
        background-color: #edfaf1;
        margin: 10px 0;
    }
    /* Metric boxes */
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    /* Action item */
    .action-item {
        background: #fff3cd;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #ffc107;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 16px;
        font-size: 0.85em;
        border-top: 1px solid #dee2e6;
        margin-top: 30px;
    }
    /* Confidence bar label */
    .conf-label {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """
    Load EfficientNet-B0 with trained weights.
    @st.cache_resource ensures the model is loaded once and reused across
    all user interactions (avoids reloading 20MB on every upload).
    """
    # Build the same architecture used during training
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Load only the model weights (not optimizer state)
        model.load_state_dict(checkpoint['model_state'])
        st.sidebar.success(f"✅ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    else:
        st.sidebar.warning("⚠️ Model file not found. Using untrained weights for demo.")

    model.eval()   # Set to inference mode (disables dropout, uses running BN stats)
    model.to(DEVICE)
    return model


@st.cache_data
def load_disease_info():
    """
    Load the cached disease descriptions JSON.
    @st.cache_data caches the return value so the file is read only once.
    """
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            return json.load(f)
    else:
        # Fallback minimal info if JSON not found
        return {
            "Bacterial Spot": {
                "description": "Bacterial spot disease detected on pepper leaf.",
                "symptoms": ["Dark spots with yellow halo", "Leaf drop"],
                "cause": "Xanthomonas campestris pv. vesicatoria",
                "recommended_actions": ["Apply copper bactericide", "Remove infected leaves"],
                "severity": "High",
                "urgency": "🔴 Act immediately"
            },
            "Healthy": {
                "description": "Plant appears healthy. No disease detected.",
                "symptoms": ["No symptoms"],
                "cause": "None",
                "recommended_actions": ["Continue regular care"],
                "severity": "None",
                "urgency": "🟢 No action needed"
            }
        }

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalized torch tensor ready for the model.

    Steps:
    1. Convert to RGB (handles RGBA, grayscale, etc.)
    2. Resize to 256×256
    3. Center-crop to 224×224 (EfficientNet-B0 input size)
    4. Convert to tensor (float32, values in [0,1])
    5. Normalize with ImageNet mean/std

    Returns:
        Tensor of shape (1, 3, 224, 224) — batch dimension added.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    image_rgb = pil_image.convert('RGB')   # Ensure 3-channel RGB
    tensor    = transform(image_rgb)        # Shape: (3, 224, 224)
    return tensor.unsqueeze(0)             # Add batch dim: (1, 3, 224, 224)


def predict(model, image_tensor: torch.Tensor):
    """
    Run inference on a preprocessed image tensor.

    Returns:
        predicted_class (int): Index of predicted class (0 or 1).
        class_name (str): Human-readable class name.
        confidence (float): Probability of predicted class (0–1).
        all_probs (list[float]): Probabilities for all classes.
    """
    with torch.no_grad():           # No gradients needed at inference
        image_tensor = image_tensor.to(DEVICE)
        logits = model(image_tensor)          # Raw scores: shape (1, 2)
        probs  = F.softmax(logits, dim=1)     # Convert to probabilities (sum to 1)

    all_probs       = probs[0].cpu().numpy().tolist()
    predicted_class = int(np.argmax(all_probs))
    confidence      = all_probs[predicted_class]
    class_name      = CLASS_NAMES[predicted_class]

    return predicted_class, class_name, confidence, all_probs

# ─────────────────────────────────────────────────────────────────────────────
# Plotly confidence bar chart
# ─────────────────────────────────────────────────────────────────────────────

def make_confidence_chart(probs: list, class_names: list):
    """
    Create an interactive horizontal bar chart of class probabilities.

    Args:
        probs: List of float probabilities (one per class).
        class_names: List of class label strings.
    Returns:
        Plotly Figure object.
    """
    colors = ['#e74c3c' if name == 'Bacterial Spot' else '#27ae60'
              for name in class_names]

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],   # Convert to percentage
        y=class_names,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(size=14, color='black')
    ))

    fig.update_layout(
        title=dict(text="Prediction Confidence", font=dict(size=16)),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 115],            # Leave space for labels
            ticksuffix="%",
            showgrid=True,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(tickfont=dict(size=14)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/"
             "Pepper_Capsicum_annuum.jpg/320px-Pepper_Capsicum_annuum.jpg",
             caption="Capsicum annuum", use_column_width=True)

    st.markdown("### 🌶️ About This App")
    st.info(
        "This app detects **Bacterial Spot disease** in pepper / chili leaves using "
        "a fine-tuned **EfficientNet-B0** model trained on the **PlantVillage** dataset."
    )

    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a **clear photo** of a pepper leaf
    2. Ensure the **leaf fills** most of the frame
    3. Use **good lighting** (natural daylight preferred)
    4. Avoid blurry or obstructed images
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🌶️ Pepper / Chili Disease Detector</h1>
    <p style="font-size:1.1em; margin:0;">
        Upload a leaf photo → Instant AI diagnosis → Farmer-friendly action plan
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model and disease info at startup
# ─────────────────────────────────────────────────────────────────────────────

model        = load_model()
disease_info = load_disease_info()

# ─────────────────────────────────────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📸 Upload Pepper Leaf Image")

col_upload, col_info = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Accepted formats: JPG, JPEG, PNG, BMP, WEBP. Max file size: 200 MB."
    )

with col_info:
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - 📷 Use a camera resolution ≥ 1MP
    - 🌿 Focus on a **single leaf** if possible
    - ☀️ Capture in **natural daylight**
    - 📐 Hold the phone **parallel** to the leaf
    - 🔍 Include **both sides** of a lesion if visible
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction and results display
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    # Read and display image
    image_bytes = uploaded_file.read()
    pil_image   = Image.open(io.BytesIO(image_bytes))

    st.markdown("---")
    st.markdown("## 🔍 Analysis Results")

    # Layout: image | results
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("### 📷 Uploaded Image")
        st.image(pil_image, caption=f"File: {uploaded_file.name}",
                 use_column_width=True)

        # Show image metadata
        w, h = pil_image.size
        st.caption(f"Resolution: {w}×{h} px | Format: {pil_image.format or 'Unknown'} | "
                   f"Mode: {pil_image.mode}")

    with col_res:
        # ── Run model inference ──
        with st.spinner("🤖 Analyzing leaf... Please wait..."):
            img_tensor = preprocess_image(pil_image)
            pred_class, class_name, confidence, all_probs = predict(model, img_tensor)

        # ── Prediction badge ──
        if class_name == "Bacterial Spot":
            st.markdown(
                f'<div style="background:#e74c3c; color:white; padding:16px; '
                f'border-radius:10px; text-align:center; font-size:1.4em; '
                f'font-weight:bold; margin-bottom:16px;">'
                f'🔴 {class_name} Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background:#27ae60; color:white; padding:16px; '
                f'border-radius:10px; text-align:center; font-size:1.4em; '
                f'font-weight:bold; margin-bottom:16px;">'
                f'🟢 {class_name}</div>',
                unsafe_allow_html=True
            )

        # ── Confidence metrics ──
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric(
                label="Confidence",
                value=f"{confidence*100:.1f}%",
                delta="High" if confidence > 0.85 else "Moderate" if confidence > 0.65 else "Low"
            )
        with col_c2:
            st.metric(
                label="Prediction",
                value=class_name,
                delta="⚠️ Diseased" if class_name == "Bacterial Spot" else "✅ Healthy"
            )

        # ── Confidence bar chart ──
        st.markdown("### 📊 Confidence Scores")
        fig = make_confidence_chart(all_probs, CLASS_NAMES)
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Disease information section (full width)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    info = disease_info.get(class_name, {})

    card_class = "disease-card" if class_name == "Bacterial Spot" else "healthy-card"

    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.markdown(f"## {'🦠' if class_name == 'Bacterial Spot' else '✅'} {class_name}")
    st.markdown(f"**Urgency:** {info.get('urgency', 'N/A')}")
    st.markdown(f"**Severity:** {info.get('severity', 'N/A')}")
    st.markdown(f"**Cause:** *{info.get('cause', 'Unknown')}*")
    st.markdown(f"\n{info.get('description', '')}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Two-column layout for symptoms and actions
    col_sym, col_act = st.columns(2)

    with col_sym:
        st.markdown("### 🔬 Symptoms to Look For")
        symptoms = info.get('symptoms', [])
        for symptom in symptoms:
            st.markdown(f"- {symptom}")

    with col_act:
        st.markdown("### 🌿 Recommended Actions")
        actions = info.get('recommended_actions', [])
        for i, action in enumerate(actions, 1):
            st.markdown(
                f'<div class="action-item">'
                f'<b>Step {i}:</b> {action}'
                f'</div>',
                unsafe_allow_html=True
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Additional information expanders
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")

    with st.expander("📚 About Bacterial Spot Disease", expanded=False):
        st.markdown("""
        ### Bacterial Spot in Pepper Plants

        **Pathogen:** *Xanthomonas campestris* pv. *vesicatoria* (also known as
        *Xanthomonas euvesicatoria* in newer taxonomy).

        **Disease Cycle:**
        1. Bacteria survive in infected plant debris and soil
        2. Spread by splashing rain, irrigation water, and wind-driven moisture
        3. Enter the plant through natural openings (stomata) or wounds
        4. Optimum temperature for infection: 24–30°C with high humidity

        **Economic Impact:**
        - Causes 20–30% yield loss in severe outbreaks
        - Affects both pepper fruit and foliage
        - Major problem in tropical and subtropical regions including India

        **Preventive Measures:**
        - Use certified disease-free seeds
        - Apply 0.1% mercuric chloride or hot water treatment (50°C, 25 min) to seeds
        - Maintain proper plant spacing for air circulation
        - Spray copper hydroxide (Kocide) preventively during wet weather
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # Download result as text report
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Diagnosis Report")

    report_text = f"""
PEPPER LEAF DISEASE DIAGNOSIS REPORT
=====================================
Generated by: Pepper Disease Detector
File analyzed: {uploaded_file.name}

DIAGNOSIS
---------
Prediction:   {class_name}
Confidence:   {confidence*100:.2f}%
Urgency:      {info.get('urgency', 'N/A')}
Severity:     {info.get('severity', 'N/A')}
Cause:        {info.get('cause', 'N/A')}

DESCRIPTION
-----------
{info.get('description', 'N/A')}

SYMPTOMS
--------
{chr(10).join(f'  - {s}' for s in info.get('symptoms', []))}

RECOMMENDED ACTIONS
-------------------
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(info.get('recommended_actions', [])))}

CLASS PROBABILITIES
-------------------
{chr(10).join(f'  {CLASS_NAMES[i]}: {all_probs[i]*100:.2f}%' for i in range(len(CLASS_NAMES)))}

MODEL INFORMATION
-----------------
  Architecture: EfficientNet-B0
  Training Dataset: PlantVillage (pepper subset)
  Classes: Bacterial Spot, Healthy

DISCLAIMER
----------
This tool is for educational purposes only. Consult a qualified
agronomist for professional diagnosis and treatment advice.
"""

    st.download_button(
        label="📄 Download Report (.txt)",
        data=report_text,
        file_name=f"diagnosis_{uploaded_file.name.split('.')[0]}.txt",
        mime="text/plain"
    )

else:
    # ── No image uploaded yet — show instructions ──
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#edfaf1; padding:20px; border-radius:12px; text-align:center;">
            <h2>📤</h2>
            <h4>Upload Image</h4>
            <p>Upload a photo of your pepper plant leaf using the uploader above.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#e8f4fd; padding:20px; border-radius:12px; text-align:center;">
            <h2>🤖</h2>
            <h4>AI Analysis</h4>
            <p>Our EfficientNet-B0 model instantly classifies the leaf as healthy or diseased.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#fff3cd; padding:20px; border-radius:12px; text-align:center;">
            <h2>🌿</h2>
            <h4>Get Action Plan</h4>
            <p>Receive a farmer-friendly explanation and step-by-step treatment advice.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 What This App Can Detect")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        <div style="background:#fdf0ee; padding:20px; border-radius:12px;
                    border-left:5px solid #e74c3c;">
            <h4>🔴 Bacterial Spot</h4>
            <p><em>Xanthomonas campestris pv. vesicatoria</em></p>
            <ul>
                <li>Dark brown spots with yellow halo</li>
                <li>Water-soaked lesions on leaves</li>
                <li>Premature leaf drop</li>
                <li>Fruit lesions with rough surface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div style="background:#edfaf1; padding:20px; border-radius:12px;
                    border-left:5px solid #27ae60;">
            <h4>🟢 Healthy</h4>
            <p><em>No disease detected</em></p>
            <ul>
                <li>Uniform green leaf color</li>
                <li>No spots or lesions</li>
                <li>Normal leaf texture</li>
                <li>Vigorous growth pattern</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

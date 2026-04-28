"""
app.py
------
Interface Streamlit — Détection et Classification de Maladies des Plantes
Modèles disponibles : Random Forest | SVM | MobileNetV2 (Grad-CAM)
"""

import os
import tempfile
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

from src.utils.agricultural_advice import get_advice
from predict import (load_models, get_predictions_dl,
                     get_predictions_ml, blend_gradcam)

# ---------------------------------------------------------------------------
# Configuration page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        border-left: 4px solid #22c55e;
    }
    .metric-card.warn { border-left-color: #f59e0b; }
    .metric-card.danger { border-left-color: #ef4444; }
    .prob-bar-label { font-size: 0.85rem; color: #94a3b8; }
    .section-title { color: #22c55e; font-weight: 700; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chargement des modèles (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modèles IA...")
def load_cached_models():
    return load_models()

rf_data, svm_data, dl_model, classes = load_cached_models()

# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/leaf.png", width=64)
    st.title("🌿 Plant Disease AI")
    st.markdown("---")

    st.subheader("⚙️ Modèle IA")
    mode = st.radio(
        "Choisir l'architecture :",
        ["🧠 Deep Learning (MobileNetV2)",
         "🌲 Machine Learning (Random Forest)",
         "📐 Machine Learning (SVM)"],
        index=0
    )

    st.markdown("---")
    st.subheader("📊 Modèles chargés")
    st.markdown(f"{'✅' if rf_data  else '❌'} Random Forest")
    st.markdown(f"{'✅' if svm_data else '❌'} SVM (RBF)")
    st.markdown(f"{'✅' if dl_model else '❌'} MobileNetV2")

    st.markdown("---")
    st.subheader("ℹ️ Description")
    if "Deep" in mode:
        st.info("Réseau convolutif pré-entraîné (ImageNet) adapté au "
                "diagnostic foliaire. Explicabilité via **Grad-CAM**.")
    elif "Random" in mode:
        st.info("Forêt aléatoire entraînée sur des features Couleur + "
                "Texture GLCM + Forme (Hu Moments).")
    else:
        st.info("SVM à noyau RBF, entraîné sur les mêmes features que RF. "
                "Généralement plus précis sur des données normalisées.")

    st.markdown("---")
    st.caption("Projet académique — Détection de Maladies des Plantes")

# ---------------------------------------------------------------------------
# Corps principal
# ---------------------------------------------------------------------------
st.title("🌿 Détection et Classification Automatique de Maladies des Plantes")
st.markdown("Chargez une image de feuille de **Maïs**, **Pomme de terre** "
            "ou **Tomate** pour obtenir un diagnostic et des conseils agricoles.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "📁 Uploader une image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Sauvegarder temporairement
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    tfile  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    tfile.close()
    temp_path = tfile.name

    col_img, col_seg = st.columns(2)
    with col_img:
        st.subheader("📷 Image originale")
        st.image(temp_path, use_container_width=True)

    run_btn = st.button("🚀 Lancer la Détection", type="primary",
                        use_container_width=True)

    if run_btn:
        # -------------------------------------------------------------------
        # DEEP LEARNING
        # -------------------------------------------------------------------
        if "Deep" in mode:
            if dl_model is None:
                st.error("❌ Modèle DL introuvable. Lancez `train_dl.py` d'abord.")
            else:
                with st.spinner("Réseau de neurones en cours d'analyse..."):
                    top3, heatmap, segmented = get_predictions_dl(
                        dl_model, classes, temp_path)

                col_res, col_cam = st.columns(2)

                with col_res:
                    st.subheader("🔬 Résultats (MobileNetV2)")
                    for i, (cls_name, prob) in enumerate(top3):
                        short = cls_name.replace("(maize)", "🌽").replace(
                            "Tomato", "🍅").replace("Potato", "🥔")
                        st.markdown(f"**#{i+1}** — {short}")
                        st.progress(int(min(prob, 100)),
                                    text=f"{prob:.1f}%")

                    predicted_class = top3[0][0]
                    advice = get_advice(predicted_class)
                    severity = "danger" if ("Late_blight" in predicted_class
                                            or "Septoria" in predicted_class) else \
                               "warn"   if "healthy" not in predicted_class else ""
                    st.markdown("---")
                    st.markdown(f"<div class='metric-card {severity}'>"
                                f"<b>💡 Conseil Agricole</b><br>{advice}</div>",
                                unsafe_allow_html=True)

                with col_seg:
                    st.subheader("🌡️ Explicabilité — Grad-CAM")
                    st.caption("Les zones chaudes indiquent les régions ayant le plus contribué à la prédiction du modèle.")
                    
                    colored_hm, gradcam_img = blend_gradcam(temp_path, heatmap)
                    
                    col_hm1, col_hm2 = st.columns(2)
                    with col_hm1:
                        st.image(colored_hm, use_container_width=True, caption="Heatmap pure")
                    with col_hm2:
                        st.image(gradcam_img, use_container_width=True, caption="Overlay final")

                    st.subheader("🍃 Segmentation (vision classique)")
                    if segmented is not None:
                        st.image(segmented, use_container_width=True,
                                 caption="Feuille isolée (masque HSV)")

        # -------------------------------------------------------------------
        # MACHINE LEARNING — RF ou SVM
        # -------------------------------------------------------------------
        else:
            model_data = rf_data if "Random" in mode else svm_data
            model_label = "Random Forest" if "Random" in mode else "SVM (RBF)"

            if model_data is None:
                st.error(f"❌ Modèle {model_label} introuvable. "
                         "Lancez `train_ml.py` d'abord.")
            else:
                with st.spinner(f"Segmentation + extraction features ({model_label})..."):
                    top3, segmented = get_predictions_ml(model_data, temp_path)

                col_res, col_seg = st.columns(2)

                with col_res:
                    st.subheader(f"🔬 Résultats ({model_label})")
                    for i, (cls_name, prob) in enumerate(top3):
                        short = cls_name.replace("(maize)", "🌽").replace(
                            "Tomato", "🍅").replace("Potato", "🥔")
                        st.markdown(f"**#{i+1}** — {short}")
                        st.progress(int(min(prob, 100)),
                                    text=f"{prob:.1f}%")

                    predicted_class = top3[0][0]
                    advice = get_advice(predicted_class)
                    severity = "danger" if ("Late_blight" in predicted_class
                                            or "Septoria" in predicted_class) else \
                               "warn"   if "healthy" not in predicted_class else ""
                    st.markdown("---")
                    st.markdown(f"<div class='metric-card {severity}'>"
                                f"<b>💡 Conseil Agricole</b><br>{advice}</div>",
                                unsafe_allow_html=True)

                with col_seg:
                    st.subheader("🍃 Segmentation (vision classique)")
                    st.caption("Masque HSV double (vert + brun) appliqué à la feuille")
                    if segmented is not None:
                        st.image(segmented, use_container_width=True,
                                 caption="Feuille isolée du fond")

    # Nettoyage fichier temporaire
    try:
        os.unlink(temp_path)
    except Exception:
        pass

else:
    # Placeholder
    st.info("👆 Uploadez une image de feuille pour commencer le diagnostic.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Modèles disponibles", "3")
    col2.metric("Classes détectées", "12")
    col3.metric("Méthodes d'explicabilité", "Grad-CAM + Segmentation HSV")

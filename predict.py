"""
predict.py
----------
Module d'inférence unifié pour les trois modèles :
  - Random Forest  (RF)
  - SVM            (SVM)
  - MobileNetV2    (DL) + Grad-CAM

Interface publique :
  load_models()                          → (rf, svm, dl_model, classes)
  get_predictions_ml(model, scaler, classes, img_path, model_name)
  get_predictions_dl(dl_model, classes, img_path) → (top3, heatmap, segmented)
  blend_gradcam(img_path, heatmap)       → np.ndarray RGB
"""

import cv2
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from src.vision.segmentation import segment_leaf
from src.features.extractors import extract_features

MODEL_DIR = Path("./models")


# ===========================================================================
# Chargement des modèles
# ===========================================================================
def load_models():
    """
    Charge les modèles ML (RF, SVM) et DL disponibles dans models/.

    Retourne
    --------
    rf_data  : dict | None  → {"model", "scaler", "classes"}
    svm_data : dict | None  → {"model", "scaler", "classes"}
    dl_model : tf.keras.Model | None
    classes  : list[str]    → liste de référence des 12 classes
    """
    rf_data  = None
    svm_data = None
    dl_model = None
    classes  = None

    rf_path  = MODEL_DIR / "rf_model.pkl"
    svm_path = MODEL_DIR / "svm_model.pkl"
    dl_path  = MODEL_DIR / "mobilenetv2_plants.keras"

    if rf_path.exists():
        rf_data = joblib.load(rf_path)
        classes = rf_data["classes"]
        print(f"✔ RF chargé  ({len(classes)} classes)")

    if svm_path.exists():
        svm_data = joblib.load(svm_path)
        if classes is None:
            classes = svm_data["classes"]
        print(f"✔ SVM chargé ({len(svm_data['classes'])} classes)")

    if dl_path.exists():
        dl_model = tf.keras.models.load_model(str(dl_path), compile=False)
        print(f"✔ DL chargé  (MobileNetV2)")

    if classes is None:
        # Fallback : ordre lexicographique standard du projet
        classes = sorted([
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___healthy",
        ])

    return rf_data, svm_data, dl_model, classes


# ===========================================================================
# Grad-CAM — implémentation robuste par nom de couche
# ===========================================================================
def make_gradcam_heatmap(img_array: np.ndarray,
                         model: tf.keras.Model) -> np.ndarray:
    """
    Grad-CAM : carte d'activation pour la classe prédite.

    Stratégie : construire un grad_model qui retourne
      (sortie de la dernière couche conv de MobileNetV2, sortie finale du modèle)
    puis calculer les gradients via GradientTape.

    La couche cible est 'out_relu' de MobileNetV2 (dernière activation
    avant GlobalAveragePooling, shape (None, 7, 7, 1280)).
    """
    TARGET_LAYER = "out_relu"   # Dernière activation spatiale de MobileNetV2

    try:
        # Trouver le sous-modèle MobileNetV2 (pas le Sequential data_augmentation)
        base_model = next(
            l for l in model.layers
            if isinstance(l, tf.keras.Model)
            and not isinstance(l, tf.keras.Sequential)
            and "mobilenet" in l.name.lower()
        )

        # Sous-modèle qui expose la feature map cible et la sortie globale
        last_conv_output = base_model.output
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_output, model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            top_class = tf.argmax(predictions[0])
            class_score = predictions[:, top_class]

        # Gradients de la classe gagnante par rapport à la feature map
        grads = tape.gradient(class_score, conv_outputs)
        # Pondération spatiale (Global Average Pooling des gradients)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # Combinaison pondérée
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs[0]), axis=-1
        )
        # ReLU + normalisation [0, 1]
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    except Exception as exc:
        print(f"⚠ Grad-CAM échoué : {exc}")
        return np.zeros((7, 7), dtype=np.float32)


def blend_gradcam(img_path: str, heatmap: np.ndarray) -> tuple:
    """
    Superpose la heatmap Grad-CAM sur l'image originale.
    Retourne (colored_heatmap, superimposed).
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Redimensionner la heatmap (7,7) → (224,224)
    heatmap_up = cv2.resize(heatmap.astype(np.float32), (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_up)

    # Colormap JET via OpenCV
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Fusion α-blending : 70% original + 30% heatmap pour plus de clarté
    superimposed = cv2.addWeighted(img, 0.7, colored, 0.3, 0)
    return colored, superimposed


# ===========================================================================
# Inférence DL
# ===========================================================================
def get_predictions_dl(dl_model: tf.keras.Model,
                       classes: list,
                       img_path: str) -> tuple:
    """
    Prédit la maladie avec MobileNetV2 et génère la heatmap Grad-CAM.

    Retourne
    --------
    top3      : list[(class_name, prob_pct)]
    heatmap   : np.ndarray  (7, 7)
    segmented : np.ndarray  (H, W, 3) — feuille segmentée (visuel)
    """
    # --- Chargement image ---
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)           # (224, 224, 3)
    img_batch = tf.expand_dims(img_array, axis=0)           # (1, 224, 224, 3)

    # --- Prédiction ---
    preds = dl_model.predict(img_batch, verbose=0)[0]       # (num_classes,)
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(classes[i], float(preds[i]) * 100) for i in top3_idx]

    # --- Grad-CAM ---
    heatmap = make_gradcam_heatmap(img_batch, dl_model)

    # --- Segmentation pour affichage visuel ---
    try:
        _, _, segmented = segment_leaf(img_path)
    except Exception:
        segmented = np.array(img)

    return top3, heatmap, segmented


# ===========================================================================
# Inférence ML (RF ou SVM)
# ===========================================================================
def get_predictions_ml(model_data: dict,
                       img_path: str) -> tuple:
    """
    Prédit la maladie avec un modèle ML classique (RF ou SVM).

    Paramètres
    ----------
    model_data : dict  {"model", "scaler", "classes"}

    Retourne
    --------
    top3      : list[(class_name, prob_pct)]
    segmented : np.ndarray (H, W, 3) — feuille segmentée
    """
    clf     = model_data["model"]
    scaler  = model_data["scaler"]
    classes = model_data["classes"]

    img_rgb, mask, segmented = segment_leaf(str(img_path))
    features = extract_features(img_rgb, mask)
    features_scaled = scaler.transform([features])

    probs = clf.predict_proba(features_scaled)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(classes[i], float(probs[i]) * 100) for i in top3_idx]

    return top3, segmented

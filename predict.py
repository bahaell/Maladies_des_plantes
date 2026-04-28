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
def _find_last_conv_output(model: tf.keras.Model):
    """Retourne la couche 'out_relu' (objet Layer) dans la hiérarchie du modèle."""
    # Essai 1 : chercher directement dans les couches du modèle
    for layer in reversed(model.layers):
        if layer.name == "out_relu":
            return layer          # retourne la couche, pas .output
    # Essai 2 : chercher dans les sous-modèles (MobileNetV2 emboîtté)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if sublayer.name == "out_relu":
                    return sublayer    # retourne la couche
    # Essai 3 : sous-modèle MobileNetV2 directement
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer          # retourne le sous-modèle entier
    return None


def make_gradcam_heatmap(img_array: np.ndarray,
                         model: tf.keras.Model) -> np.ndarray:
    """
    Grad-CAM robuste pour Keras 3 / TF 2.x.

    Approche :
      1. Trouver la dernière couche convolutive spatiale (out_relu, 7×7×1280)
      2. Construire un grad_model à deux sorties
      3. Approche one-hot différentiable pour la sélection de classe
      4. GradientTape watchant img_tensor AVANT le forward pass
    """
    try:
        last_conv_layer = _find_last_conv_output(model)
        if last_conv_layer is None:
            print("⚠ Grad-CAM : couche out_relu introuvable")
            return np.zeros((7, 7), dtype=np.float32)

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]   # .output ici
        )

        img_tensor = tf.Variable(
            tf.cast(img_array, tf.float32), trainable=False
        )

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            # One-hot différentiable sans conversion Python int (graph-safe)
            top_class   = tf.argmax(predictions[0])           # tensor int64
            n_classes   = predictions.shape[-1]
            one_hot     = tf.one_hot(top_class, n_classes)    # (12,)
            one_hot     = tf.expand_dims(one_hot, 0)          # (1, 12)
            class_score = tf.reduce_sum(predictions * one_hot)  # scalaire

        # d(class_score) / d(conv_outputs)
        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            print("⚠ Grad-CAM : gradient None → fallback sur gradient/input")
            return np.zeros((7, 7), dtype=np.float32)

        # Pooling spatial des gradients (Grad-CAM weighting)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (1280,)
        # Pondération des feature maps
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs[0]), axis=-1      # (7, 7)
        )
        heatmap = tf.nn.relu(heatmap).numpy()

        # Debug
        print(f"  Grad-CAM → max={heatmap.max():.4f}  mean={heatmap.mean():.4f}"
              f"  classe={top_class}")

        if heatmap.max() > 1e-8:
            heatmap = heatmap / heatmap.max()
        else:
            # Gradient nul : utiliser les activations brutes comme fallback
            print("  ⚠ Gradients nuls → fallback sur activations brutes")
            raw = conv_outputs[0].numpy()
            heatmap = np.mean(raw, axis=-1)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    except Exception as exc:
        print(f"⚠ Grad-CAM exception : {exc}")
        return np.zeros((7, 7), dtype=np.float32)


def blend_gradcam(img_path: str, heatmap: np.ndarray) -> tuple:
    """
    Superpose la heatmap Grad-CAM sur l'image originale.
    Retourne (colored_heatmap, superimposed).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Image introuvable ou illisible : {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Redimensionner la heatmap (7,7) → (224,224) — INTER_CUBIC pour meilleur rendu
    heatmap_up = cv2.resize(heatmap.astype(np.float32), (224, 224),
                            interpolation=cv2.INTER_CUBIC)
    # Lissage Gaussien pour éliminer le côté pixelisé de la heatmap 7×7
    heatmap_up = cv2.GaussianBlur(heatmap_up, (9, 9), 0)
    # Renormaliser après blur
    if heatmap_up.max() > 0:
        heatmap_up = heatmap_up / heatmap_up.max()
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

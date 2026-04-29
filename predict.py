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


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===========================================================================
# Grad-CAM — pipeline à 3 méthodes en cascade
# ===========================================================================

def _clean_heatmap(heatmap):
    """Supprime les artefacts sur les bords (padding) et normalise."""
    if heatmap is None or heatmap.max() < 1e-8:
        return None
    # Supprimer les bords extrêmes qui capturent souvent du bruit de convolution (heatmap 7x7)
    heatmap[0, :] = 0; heatmap[-1, :] = 0
    heatmap[:, 0] = 0; heatmap[:, -1] = 0
    
    if heatmap.max() > 0:
        return (heatmap / heatmap.max()).astype(np.float32)
    return None

def _gradcam_method1(model, img_tensor, n_classes):
    """Méthode 1 : Grad-CAM classique via out_relu."""
    try:
        # Crucial : MobileNetV2 nécessite des pixels en [-1, 1]
        preprocessed_img = preprocess_input(tf.identity(img_tensor))

        last_conv_layer = None
        for layer in model.layers:
            if layer.name == "out_relu":
                last_conv_layer = layer
                break
            if isinstance(layer, tf.keras.Model):
                for sublayer in layer.layers:
                    if sublayer.name == "out_relu":
                        last_conv_layer = sublayer
                        break
            if last_conv_layer:
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            tape.watch(preprocessed_img)
            conv_out, preds = grad_model(preprocessed_img, training=False)
            top_class = tf.argmax(preds[0])
            one_hot   = tf.expand_dims(tf.one_hot(top_class, n_classes), 0)
            loss      = tf.reduce_sum(preds * one_hot)

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(
            tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
        ).numpy()

        return _clean_heatmap(heatmap)

    except Exception as e:
        print(f"  Méthode 1 échouée : {type(e).__name__}")
        return None


def _gradcam_method2(model, img_tensor, n_classes):
    """Méthode 2 : Grad-CAM via sous-modèle MobileNetV2 entier."""
    try:
        mobilenet_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
                mobilenet_layer = layer
                break

        if mobilenet_layer is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[mobilenet_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_out, preds = grad_model(img_tensor, training=False)
            top_class = tf.argmax(preds[0])
            one_hot   = tf.expand_dims(tf.one_hot(top_class, n_classes), 0)
            loss      = tf.reduce_sum(preds * one_hot)

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(
            tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
        ).numpy()

        if heatmap.max() > 1e-8:
            return (heatmap / heatmap.max()).astype(np.float32)
        return None

    except Exception as e:
        print(f"  Méthode 2 échouée : {type(e).__name__}")
        return None


def _gradcam_method3_saliency(model, img_tensor, n_classes):
    """Méthode 3 (fallback) : Input Gradient Saliency — toujours fonctionnel."""
    img_var = tf.Variable(img_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds    = model(img_var, training=False)
        top_class = tf.argmax(preds[0])
        one_hot  = tf.expand_dims(tf.one_hot(top_class, n_classes), 0)
        loss     = tf.reduce_sum(preds * one_hot)

    grads   = tape.gradient(loss, img_var)           # (1, 224, 224, 3)
    saliency = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()  # (224, 224)
    if saliency.max() > 1e-10:
        saliency /= saliency.max()
    # Réduire à 7×7 pour le pipeline identique
    saliency_small = cv2.resize(saliency, (7, 7), interpolation=cv2.INTER_AREA)
    return saliency_small.astype(np.float32)


def make_gradcam_heatmap(img_array: np.ndarray,
                         model: tf.keras.Model) -> np.ndarray:
    """
    Génère une heatmap d'activation en cascade :
      1. Grad-CAM via out_relu (MobileNetV2 standard)
      2. Grad-CAM via sous-modèle MobileNetV2 entier
      3. Input Gradient Saliency (toujours fonctionnel)
    """
    n_classes  = model.output_shape[-1]
    img_tensor = tf.cast(img_array, tf.float32)

    # --- Essai 1 ---
    heatmap = _gradcam_method1(model, img_tensor, n_classes)
    if heatmap is not None:
        print(f"  Grad-CAM méthode 1 ✅  max={heatmap.max():.4f}")
        return heatmap

    # --- Essai 2 ---
    heatmap = _gradcam_method2(model, img_tensor, n_classes)
    if heatmap is not None:
        print(f"  Grad-CAM méthode 2 ✅  max={heatmap.max():.4f}")
        return heatmap

    # --- Essai 3 (fallback saliency) ---
    print("  ⚠ Grad-CAM méthodes 1&2 échouées → Input Gradient Saliency")
    heatmap = _gradcam_method3_saliency(model, img_tensor, n_classes)
    print(f"  Saliency méthode 3 ✅  max={heatmap.max():.4f}")
    return heatmap




def blend_gradcam(img_path: str, heatmap: np.ndarray) -> tuple:
    """
    Superpose la heatmap Grad-CAM sur l'image originale.
    Utilise la segmentation pour nettoyer le fond.
    """
    from src.vision.segmentation import segment_leaf
    
    # 1. Charger l'image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Image introuvable : {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_224 = cv2.resize(img_rgb, (224, 224))

    # 2. Obtenir le masque de la feuille (notre segmentation HSV)
    # segment_leaf retourne (orig, mask, seg)
    _, mask, _ = segment_leaf(img_path)
    mask_224 = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask_bool = (mask_224 > 0).astype(np.float32)

    # 3. Préparer la heatmap
    heatmap_up = cv2.resize(heatmap.astype(np.float32), (224, 224),
                            interpolation=cv2.INTER_CUBIC)
    
    # --- NETTOYAGE : On multiplie par le masque pour effacer le fond ---
    heatmap_up = heatmap_up * mask_bool
    
    # 4. Lissage et normalisation
    heatmap_up = cv2.GaussianBlur(heatmap_up, (11, 11), 0)
    if heatmap_up.max() > 0:
        heatmap_up = heatmap_up / heatmap_up.max()
    
    heatmap_uint8 = np.uint8(255 * heatmap_up)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # 5. Fusion α-blending
    superimposed = cv2.addWeighted(img_224, 0.7, colored, 0.3, 0)
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

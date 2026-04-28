"""
benchmark.py
------------
Comparaison complète des trois modèles sur le Test Set :
  - Random Forest  (ML)
  - SVM RBF        (ML)
  - MobileNetV2    (DL)

Métriques : Accuracy, Précision, Rappel, F1-score (weighted),
            Temps d'entraînement, Temps d'inférence

Sorties :
  benchmark_results.csv   — tableau complet
  benchmark_chart.png     — graphe comparatif
"""

import sys
import os
# Garantir que la racine du projet est dans le PYTHONPATH (robuste Windows)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from src.vision.segmentation import segment_leaf
from src.features.extractors import extract_features

PROCESSED_DIR = Path("./data/processed")
MODEL_DIR     = Path("./models")
VALID_EXT     = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_ml_test_features(file_paths, labels_dl, classes_dl):
    """
    Extrait les features ML sur les fichiers du test set DL.
    Retourne X_test, y_true (noms de classes string).
    """
    X, y = [], []
    errors = 0
    for fp, lbl_idx in zip(file_paths, labels_dl):
        class_name = classes_dl[lbl_idx]
        try:
            img_rgb, mask, _ = segment_leaf(str(fp))
            feat = extract_features(img_rgb, mask)
            X.append(feat)
            y.append(class_name)
        except Exception:
            errors += 1
    if errors:
        print(f"  ⚠ {errors} image(s) ignorée(s)")
    return np.array(X, dtype=np.float32), np.array(y)


def _metrics(name, y_true, y_pred, train_time, infer_time):
    return {
        "Modèle":               name,
        "Accuracy (%)":         round(accuracy_score(y_true, y_pred) * 100, 2),
        "Précision (%)":        round(precision_score(y_true, y_pred,
                                                      average="weighted",
                                                      zero_division=0) * 100, 2),
        "Rappel (%)":           round(recall_score(y_true, y_pred,
                                                   average="weighted",
                                                   zero_division=0) * 100, 2),
        "F1-score (%)":         round(f1_score(y_true, y_pred,
                                               average="weighted",
                                               zero_division=0) * 100, 2),
        "Entraînement (s)":     round(train_time, 1),
        "Inférence Test (s)":   round(infer_time, 2),
    }


def _plot_benchmark(df: pd.DataFrame) -> None:
    metrics = ["Accuracy (%)", "Précision (%)", "Rappel (%)", "F1-score (%)"]
    models  = df["Modèle"].tolist()
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = df[df["Modèle"] == model][metrics].values.flatten()
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.87)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Benchmark — Random Forest vs SVM vs MobileNetV2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_chart.png", dpi=150)
    plt.close()
    print("  ✔ Graphe sauvegardé : benchmark_chart.png")


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------
def run_benchmark() -> None:
    print("=" * 65)
    print("   BENCHMARK COMPLET — ML vs DL")
    print("=" * 65)

    rf_path  = MODEL_DIR / "rf_model.pkl"
    svm_path = MODEL_DIR / "svm_model.pkl"
    dl_path  = MODEL_DIR / "mobilenetv2_plants.keras"

    missing = [p for p in [rf_path, svm_path, dl_path] if not p.exists()]
    if missing:
        print(f"⛔ Modèles manquants : {[m.name for m in missing]}")
        print("   Lancez train_ml.py et train_dl.py d'abord.")
        return

    # -----------------------------------------------------------------------
    # Charger le Dataset Test (référence commune)
    # -----------------------------------------------------------------------
    print("\n[+] Chargement du Test Set...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        image_size=(224, 224),
        batch_size=32,
        label_mode="int",
        shuffle=False
    )
    classes_dl    = test_ds.class_names
    y_true_dl_idx = np.concatenate([y.numpy() for _, y in test_ds])
    y_true_dl_str = np.array([classes_dl[i] for i in y_true_dl_idx])
    file_paths    = test_ds.file_paths

    results = []

    # -----------------------------------------------------------------------
    # 1. RANDOM FOREST
    # -----------------------------------------------------------------------
    print("\n[1/3] Random Forest...")
    rf_data    = joblib.load(rf_path)
    rf_model   = rf_data["model"]
    rf_scaler  = rf_data["scaler"]
    rf_metrics = rf_data.get("metrics", {})
    rf_train_t = rf_metrics.get("train_time", 0)

    print("      Extraction features sur le test set...")
    X_test_ml, y_true_ml = _load_ml_test_features(
        file_paths, y_true_dl_idx, classes_dl)
    X_test_ml_s = rf_scaler.transform(X_test_ml)

    t0 = time.time()
    y_pred_rf = rf_model.predict(X_test_ml_s)
    rf_infer_t = time.time() - t0

    results.append(_metrics("Random Forest", y_true_ml, y_pred_rf,
                             rf_train_t, rf_infer_t))

    # -----------------------------------------------------------------------
    # 2. SVM
    # -----------------------------------------------------------------------
    print("\n[2/3] SVM (RBF)...")
    svm_data    = joblib.load(svm_path)
    svm_model   = svm_data["model"]
    svm_scaler  = svm_data["scaler"]
    svm_metrics = svm_data.get("metrics", {})
    svm_train_t = svm_metrics.get("train_time", 0)

    X_test_svm_s = svm_scaler.transform(X_test_ml)  # Même features que RF

    t0 = time.time()
    y_pred_svm = svm_model.predict(X_test_svm_s)
    svm_infer_t = time.time() - t0

    results.append(_metrics("SVM (RBF)", y_true_ml, y_pred_svm,
                             svm_train_t, svm_infer_t))

    # -----------------------------------------------------------------------
    # 3. DEEP LEARNING (MobileNetV2)
    # -----------------------------------------------------------------------
    print("\n[3/3] MobileNetV2...")
    dl_model   = tf.keras.models.load_model(str(dl_path), compile=False)
    test_ds_pf = test_ds.prefetch(tf.data.AUTOTUNE)

    t0 = time.time()
    y_pred_probs = dl_model.predict(test_ds_pf, verbose=1)
    dl_infer_t   = time.time() - t0

    y_pred_dl = np.argmax(y_pred_probs, axis=1)
    y_pred_dl_str = np.array([classes_dl[i] for i in y_pred_dl])

    results.append(_metrics("MobileNetV2 (DL)", y_true_dl_str,
                             y_pred_dl_str, 0, dl_infer_t))

    # -----------------------------------------------------------------------
    # Tableau final
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)

    print(f"\n{'='*65}")
    print("  RÉSULTAT FINAL")
    print(f"{'='*65}")
    print(df.to_string(index=False))

    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8")
    print("\n  ✔ Sauvegardé : benchmark_results.csv")

    _plot_benchmark(df)


if __name__ == "__main__":
    run_benchmark()

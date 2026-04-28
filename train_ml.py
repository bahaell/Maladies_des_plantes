"""
train_ml.py
-----------
Entraînement des modèles Machine Learning classiques :
  - Random Forest (RF)
  - Support Vector Machine (SVM / kernel RBF)

Pipeline :
  data/processed/train → segmentation → extraction features → StandardScaler
  → RandomForest + SVM → évaluation → sauvegarde modèles

Sorties :
  models/rf_model.pkl   — RandomForest + classes
  models/svm_model.pkl  — SVM + StandardScaler + classes
  confusion_matrix_rf.png
  confusion_matrix_svm.png
"""

import os
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score,
                             precision_score, recall_score)

from src.vision.segmentation import segment_leaf
from src.features.extractors import extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("./data/processed")
MODEL_DIR     = Path("./models")
VALID_EXT     = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Chargement et extraction de features
# ---------------------------------------------------------------------------
def load_features(split_name: str) -> tuple:
    """
    Parcourt data/processed/<split_name>, applique la segmentation et
    extrait le vecteur de features pour chaque image.

    Retourne
    --------
    X      : np.ndarray  (N, D)
    y      : np.ndarray  (N,)  labels string
    classes: list[str]   liste des noms de classes (ordre lexicographique)
    """
    X, y = [], []
    split_dir = PROCESSED_DIR / split_name
    classes   = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    print(f"\n[+] Extraction des features — {split_name.upper()} "
          f"({len(classes)} classes)...")

    errors = 0
    for cls in classes:
        cls_dir = split_dir / cls
        files   = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXT]
        print(f"  → {cls:<55} {len(files):>5} images", end="", flush=True)

        ok = 0
        for img_path in files:
            try:
                img_rgb, mask, _ = segment_leaf(str(img_path))
                features = extract_features(img_rgb, mask)
                X.append(features)
                y.append(cls)
                ok += 1
            except Exception as e:
                errors += 1

        print(f"  ✔ {ok} ok")

    if errors:
        print(f"\n  ⚠ {errors} image(s) ignorée(s) (corrompues ou illisibles).")

    return np.array(X, dtype=np.float32), np.array(y), classes


# ---------------------------------------------------------------------------
# Matrice de confusion
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, classes: list, filename: str,
                          title: str) -> None:
    """Génère, affiche et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("Vraie classe")
    ax.set_xlabel("Classe prédite")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✔ Matrice sauvegardée : {filename}")


# ---------------------------------------------------------------------------
# Métriques complètes
# ---------------------------------------------------------------------------
def print_metrics(name: str, y_true, y_pred, classes: list,
                  train_time: float, infer_time: float) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    r   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS — {name}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Précision  : {p*100:.2f}%")
    print(f"  Rappel     : {r*100:.2f}%")
    print(f"  F1-score   : {f1*100:.2f}%")
    print(f"  Entraîn.   : {train_time:.1f}s")
    print(f"  Inférence  : {infer_time:.1f}s (dataset test complet)")
    print(f"\n{classification_report(y_true, y_pred, target_names=classes, zero_division=0)}")

    return {"model": name, "accuracy": acc, "precision": p,
            "recall": r, "f1": f1,
            "train_time": train_time, "infer_time": infer_time}


# ---------------------------------------------------------------------------
# Entraînement principal
# ---------------------------------------------------------------------------
def train_ml() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Chargement des données (TOUT le dataset, sans limite artificielle)
    # -----------------------------------------------------------------------
    X_train, y_train, classes = load_features("train")
    X_test,  y_test,  _       = load_features("test")

    print(f"\n  Dimensions vecteur features : {X_train.shape[1]}")
    print(f"  Taille Train : {len(X_train)} | Test : {len(X_test)}")

    # -----------------------------------------------------------------------
    # StandardScaler (nécessaire pour SVM, bénéfique pour RF aussi)
    # -----------------------------------------------------------------------
    print("\n[+] Normalisation StandardScaler...")
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # 1. RANDOM FOREST
    # -----------------------------------------------------------------------
    print("\n[+] Entraînement RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    y_pred_rf = rf.predict(X_test_s)
    rf_infer_time = time.time() - t0

    metrics_rf = print_metrics("Random Forest", y_test, y_pred_rf, classes,
                               rf_train_time, rf_infer_time)
    save_confusion_matrix(y_test, y_pred_rf, classes,
                          "confusion_matrix_rf.png",
                          "Matrice de Confusion — Random Forest")

    # Sauvegarde RF
    rf_path = MODEL_DIR / "rf_model.pkl"
    joblib.dump({"model": rf, "scaler": scaler, "classes": classes,
                 "metrics": metrics_rf}, rf_path)
    print(f"  ✔ Modèle RF sauvegardé : {rf_path}")

    # -----------------------------------------------------------------------
    # 2. SVM (kernel RBF, C=10)
    # -----------------------------------------------------------------------
    print("\n[+] Entraînement SVM (kernel RBF)...")
    t0 = time.time()
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,   # Pour predict_proba → Top-3
        random_state=42
    )
    svm.fit(X_train_s, y_train)
    svm_train_time = time.time() - t0

    t0 = time.time()
    y_pred_svm = svm.predict(X_test_s)
    svm_infer_time = time.time() - t0

    metrics_svm = print_metrics("SVM (RBF)", y_test, y_pred_svm, classes,
                                svm_train_time, svm_infer_time)
    save_confusion_matrix(y_test, y_pred_svm, classes,
                          "confusion_matrix_svm.png",
                          "Matrice de Confusion — SVM (RBF)")

    # Sauvegarde SVM
    svm_path = MODEL_DIR / "svm_model.pkl"
    joblib.dump({"model": svm, "scaler": scaler, "classes": classes,
                 "metrics": metrics_svm}, svm_path)
    print(f"  ✔ Modèle SVM sauvegardé : {svm_path}")

    # -----------------------------------------------------------------------
    # Comparaison rapide RF vs SVM
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  COMPARAISON RAPIDE ML")
    print(f"{'='*60}")
    print(f"  {'Modèle':<20} {'Accuracy':>10} {'F1-score':>10} {'Temps entr.':>12}")
    print(f"  {'-'*55}")
    for m in [metrics_rf, metrics_svm]:
        print(f"  {m['model']:<20} {m['accuracy']*100:>9.2f}% "
              f"{m['f1']*100:>9.2f}% {m['train_time']:>11.1f}s")


if __name__ == "__main__":
    train_ml()

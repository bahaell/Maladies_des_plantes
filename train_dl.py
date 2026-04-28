"""
train_dl.py
-----------
Entraînement MobileNetV2 par Transfer Learning en deux phases :
  Phase 1 — Feature Extraction  : base_model gelé, head entraîné (10 epochs)
  Phase 2 — Fine-tuning          : 30 dernières couches dégelées, lr réduit (10 epochs)

Sorties :
  models/mobilenetv2_plants.keras   — meilleur modèle complet
  training_history.png              — courbes accuracy/loss par epoch
  confusion_matrix_dl.png           — matrice de confusion sur le test set
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization,
                                     RandomFlip, RandomRotation, RandomZoom)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR  = Path("./data/processed")
MODEL_DIR      = Path("./models")
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS_PHASE1  = 10   # Feature extraction
EPOCHS_PHASE2  = 10   # Fine-tuning
FINE_TUNE_AT   = 100  # Geler les couches avant cet index dans MobileNetV2


# ---------------------------------------------------------------------------
# Calcul des class_weights (avant prefetch)
# ---------------------------------------------------------------------------
def compute_weights(dataset) -> dict:
    """Calcule class_weight='balanced' depuis un tf.data.Dataset."""
    y_all = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_all),
        y=y_all
    )
    return dict(enumerate(weights))


# ---------------------------------------------------------------------------
# Graphe des courbes d'apprentissage
# ---------------------------------------------------------------------------
def plot_history(history_p1, history_p2) -> None:
    """Sauvegarde les courbes accuracy et loss des deux phases."""
    # Concaténer les métriques
    acc  = history_p1.history["accuracy"]      + history_p2.history["accuracy"]
    val  = history_p1.history["val_accuracy"]  + history_p2.history["val_accuracy"]
    loss = history_p1.history["loss"]          + history_p2.history["loss"]
    vloss= history_p1.history["val_loss"]      + history_p2.history["val_loss"]
    ep   = range(1, len(acc) + 1)
    sep  = len(history_p1.history["accuracy"])  # séparation phase 1 / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, metric in zip(
            axes,
            [acc, loss],
            [val, vloss],
            ["Accuracy", "Loss"]):
        ax.plot(ep, train_vals, label=f"Train {metric}", color="steelblue")
        ax.plot(ep, val_vals,   label=f"Val   {metric}", color="tomato")
        ax.axvline(sep, color="gray", linestyle="--", label="Début fine-tuning")
        ax.set_title(f"{metric} par epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("MobileNetV2 — Courbes d'apprentissage (Phase 1 + Fine-tuning)", y=1.02)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ Courbes sauvegardées : training_history.png")


# ---------------------------------------------------------------------------
# Matrice de confusion DL
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, classes, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Matrice de Confusion — MobileNetV2 (Transfer Learning)")
    ax.set_ylabel("Vraie classe")
    ax.set_xlabel("Classe prédite")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✔ Matrice sauvegardée : {filename}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def train_deep_learning() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    AUTOTUNE = tf.data.AUTOTUNE

    # -----------------------------------------------------------------------
    # Chargement des datasets
    # -----------------------------------------------------------------------
    print("[+] Chargement des datasets (Train / Val / Test)...")

    def load_ds(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            PROCESSED_DIR / split,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=shuffle,
            seed=42
        )

    train_ds = load_ds("train", shuffle=True)
    val_ds   = load_ds("val",   shuffle=False)
    test_ds  = load_ds("test",  shuffle=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  → {num_classes} classes : {class_names}")

    # Class weights (avant prefetch)
    class_weight = compute_weights(train_ds)
    print(f"  → Class weights calculés ({num_classes} entrées)")

    # Optimisation pipeline
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds_eval = test_ds.prefetch(AUTOTUNE)

    # -----------------------------------------------------------------------
    # Construction du modèle
    # -----------------------------------------------------------------------
    print("\n[+] Construction du modèle MobileNetV2...")

    # Data augmentation (appliquée uniquement en entraînement)
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.15),
        RandomZoom(0.15),
    ], name="data_augmentation")

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Phase 1 : tout gelé

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = data_augmentation(inputs)
    x = preprocess_input(x)                        # [-1, 1] pour MobileNetV2
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="MobileNetV2_PlantDisease")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print(f"  → Paramètres entraînables : "
          f"{sum(w.numpy().size for w in model.trainable_weights):,}")

    best_model_path = MODEL_DIR / "mobilenetv2_plants.keras"
    callbacks_p1 = [
        ModelCheckpoint(str(best_model_path), save_best_only=True,
                        monitor="val_accuracy", mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=4,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                          min_lr=1e-6, verbose=1),
    ]

    # -----------------------------------------------------------------------
    # Phase 1 — Feature Extraction
    # -----------------------------------------------------------------------
    print(f"\n[+] Phase 1 — Feature Extraction ({EPOCHS_PHASE1} epochs max)...")
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weight,
        callbacks=callbacks_p1
    )

    # -----------------------------------------------------------------------
    # Phase 2 — Fine-tuning (dégeler les 30 dernières couches)
    # -----------------------------------------------------------------------
    print(f"\n[+] Phase 2 — Fine-tuning (couches {FINE_TUNE_AT}+ dégelées)...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  → {trainable_count} couches dégelées dans MobileNetV2")

    # Recompiler avec un lr très faible pour ne pas détruire les features
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks_p2 = [
        ModelCheckpoint(str(best_model_path), save_best_only=True,
                        monitor="val_accuracy", mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                          min_lr=1e-8, verbose=1),
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weight,
        callbacks=callbacks_p2
    )

    # -----------------------------------------------------------------------
    # Courbes d'apprentissage
    # -----------------------------------------------------------------------
    plot_history(history_p1, history_p2)

    # -----------------------------------------------------------------------
    # Évaluation finale sur le Test set
    # -----------------------------------------------------------------------
    print("\n[+] Évaluation finale sur le Test Set...")
    # Recharger le meilleur modèle sauvegardé
    best_model = tf.keras.models.load_model(str(best_model_path), compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    loss, accuracy = best_model.evaluate(test_ds_eval, verbose=1)
    print(f"\n  Test Accuracy  : {accuracy*100:.2f}%")
    print(f"  Test Loss      : {loss:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred_probs = best_model.predict(test_ds_eval, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    save_confusion_matrix(y_true, y_pred, class_names, "confusion_matrix_dl.png")
    print(f"\n✅ Modèle sauvegardé : {best_model_path}")


if __name__ == "__main__":
    train_deep_learning()

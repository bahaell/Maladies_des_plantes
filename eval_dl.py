import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = Path("./data/processed")
MODEL_DIR = Path("./models")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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

print("\n[+] Évaluation finale sur le Test Set (Reprise)...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    PROCESSED_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)
class_names = test_ds.class_names
test_ds_eval = test_ds.prefetch(tf.data.AUTOTUNE)

best_model_path = MODEL_DIR / "mobilenetv2_plants.keras"
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
print(f"\n✅ Terminé.")

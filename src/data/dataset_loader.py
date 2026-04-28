"""
dataset_loader.py
-----------------
Intègre les 12 classes cibles depuis les archives ZIP déjà téléchargées :
  - Maïs  ×4  (depuis corn-or-maize dataset)
  - Pomme de terre ×3 (depuis PlantVillage)
  - Tomate ×5 dont Septoria_leaf_spot (depuis PlantVillage)
"""

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RAW_ZIP_DIR   = Path("./data/zip")
FINAL_DATA_DIR = Path("./data/raw")

# ---------------------------------------------------------------------------
# 12 classes cibles (noms standardisés utilisés dans tout le projet)
# ---------------------------------------------------------------------------
TARGET_CLASSES = [
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
]

# ---------------------------------------------------------------------------
# Mapping : dossier source (dans ZIP) → nom standardisé cible
# ---------------------------------------------------------------------------
PLANTVILLAGE_MAPPING = {
    # Pomme de terre
    "Potato___Early_blight":        "Potato___Early_blight",
    "Potato___Late_blight":         "Potato___Late_blight",
    "Potato___healthy":             "Potato___healthy",
    # Tomate
    "Tomato_Early_blight":          "Tomato___Early_blight",
    "Tomato___Early_blight":        "Tomato___Early_blight",
    "Tomato_Late_blight":           "Tomato___Late_blight",
    "Tomato___Late_blight":         "Tomato___Late_blight",
    "Tomato_Leaf_Mold":             "Tomato___Leaf_Mold",
    "Tomato___Leaf_Mold":           "Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot":    "Tomato___Septoria_leaf_spot",
    "Tomato___Septoria_leaf_spot":  "Tomato___Septoria_leaf_spot",
    "Tomato_healthy":               "Tomato___healthy",
    "Tomato___healthy":             "Tomato___healthy",
}

CORN_MAPPING = {
    "Healthy":       "Corn_(maize)___healthy",
    "Common_Rust":   "Corn_(maize)___Common_rust_",
    "Blight":        "Corn_(maize)___Northern_Leaf_Blight",
    "Gray_Leaf_Spot":"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
}


def _copy_class(src: Path, dst_name: str) -> int:
    """Copie récursivement src → FINAL_DATA_DIR/dst_name. Retourne le nb d'images."""
    dst = FINAL_DATA_DIR / dst_name
    if dst.exists():
        count = len(list(dst.glob("*.*")))
        print(f"  ✔ Déjà présent : {dst_name} ({count} images)")
        return count
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            shutil.copy2(f, dst / f.name)
            copied += 1
    print(f"  ✔ Copié        : {dst_name} ({copied} images)")
    return copied


def prepare_data() -> None:
    """Pipeline principal de préparation des données."""
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PRÉPARATION DES DONNÉES — 12 CLASSES")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Recherche des dossiers PlantVillage dans les archives ZIP
    # -----------------------------------------------------------------------
    print("\n[1/2] Intégration des classes PlantVillage (Pomme de terre + Tomate)...")
    pv_roots = []
    for root, dirs, _ in os.walk(str(RAW_ZIP_DIR)):
        for d in dirs:
            if d in PLANTVILLAGE_MAPPING:
                pv_roots.append((Path(root) / d, PLANTVILLAGE_MAPPING[d]))

    # Dédupliquer (plusieurs niveaux de dossier dans le ZIP)
    already_done = set()
    for src_path, dst_name in pv_roots:
        if dst_name not in already_done:
            _copy_class(src_path, dst_name)
            already_done.add(dst_name)

    # -----------------------------------------------------------------------
    # 2. Intégration des classes Maïs
    # -----------------------------------------------------------------------
    print("\n[2/2] Intégration des classes Maïs...")
    for root, dirs, _ in os.walk(str(RAW_ZIP_DIR)):
        for d in dirs:
            for key, dst_name in CORN_MAPPING.items():
                if d.lower() == key.lower() and dst_name not in already_done:
                    _copy_class(Path(root) / d, dst_name)
                    already_done.add(dst_name)

    # -----------------------------------------------------------------------
    # Validation finale
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VALIDATION FINALE")
    print("=" * 60)
    missing = []
    for cls in TARGET_CLASSES:
        cls_dir = FINAL_DATA_DIR / cls
        if cls_dir.exists():
            n = len(list(cls_dir.glob("*.*")))
            print(f"  ✔ {cls:<55} {n:>5} images")
        else:
            print(f"  ✗ MANQUANT : {cls}")
            missing.append(cls)

    if missing:
        print(f"\n⚠ {len(missing)} classe(s) manquante(s) — vérifiez les archives ZIP.")
    else:
        print(f"\n✅ 12/12 classes présentes. Données prêtes pour le split.")


if __name__ == "__main__":
    prepare_data()

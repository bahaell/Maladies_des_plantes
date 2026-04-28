"""
splitter.py
-----------
Divise le dataset brut (data/raw) en Train / Val / Test de façon stratifiée,
déterministe et sans fuite de données.
Ratios : 70% / 15% / 15%
"""

import os
import shutil
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR       = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def analyze_class_balance() -> dict:
    """Analyse et affiche la distribution des images par classe."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Le dossier {RAW_DIR} n'existe pas. "
                                "Lancez dataset_loader.py d'abord.")

    class_counts = {}
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir():
            imgs = [f for f in d.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
            class_counts[d.name] = len(imgs)

    # Graphe
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(class_counts)), list(class_counts.values()), color="steelblue")
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(list(class_counts.keys()), rotation=45, ha="right", fontsize=8)
    ax.set_title("Distribution des images par classe (avant split)")
    ax.set_ylabel("Nombre d'images")
    plt.tight_layout()
    plt.savefig("class_balance.png", dpi=150)
    plt.close()

    print("\n--- Distribution des classes ---")
    total = sum(class_counts.values())
    for cls, count in class_counts.items():
        bar = "█" * (count // 50)
        print(f"  {cls:<55} {count:>5}  {bar}")
    print(f"\n  TOTAL : {total} images | {len(class_counts)} classes")
    return class_counts


def _clean_processed() -> None:
    """Supprime et recrée le dossier processed pour éviter toute fuite de données."""
    if PROCESSED_DIR.exists():
        print("⚠ Nettoyage de data/processed (éviter fuite train→test)...")
        shutil.rmtree(PROCESSED_DIR)
    for split in ("train", "val", "test"):
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)


def split_dataset(force_redo: bool = False) -> None:
    """
    Effectue le split stratifié 70/15/15.

    Paramètres
    ----------
    force_redo : bool
        Si True, supprime l'ancien split même s'il existe.
    """
    random.seed(RANDOM_SEED)

    # Vérification idempotente
    if PROCESSED_DIR.exists() and not force_redo:
        existing = [d.name for d in (PROCESSED_DIR / "train").iterdir()
                    if d.is_dir()] if (PROCESSED_DIR / "train").exists() else []
        if len(existing) >= 12:
            print(f"✔ Split déjà effectué ({len(existing)} classes). "
                  "Utilisez force_redo=True pour refaire.")
            return

    _clean_processed()
    class_counts = analyze_class_balance()

    print(f"\n--- Split 70% Train / 15% Val / 15% Test (seed={RANDOM_SEED}) ---")
    summary = {}
    for cls, total in class_counts.items():
        src_dir = RAW_DIR / cls
        files = [f.name for f in src_dir.iterdir()
                 if f.suffix.lower() in VALID_EXTENSIONS]
        random.shuffle(files)

        n_train = int(total * TRAIN_RATIO)
        n_val   = int(total * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }
        summary[cls] = {k: len(v) for k, v in splits.items()}

        for split_name, split_files in splits.items():
            dest_dir = PROCESSED_DIR / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for fname in split_files:
                shutil.copy2(src_dir / fname, dest_dir / fname)

        print(f"  {cls:<55} Train:{splits['train'].__len__():>4} "
              f"| Val:{splits['val'].__len__():>4} "
              f"| Test:{splits['test'].__len__():>4}")

    # Validation croisée : vérifier que train/val/test ont les mêmes classes
    for split in ("train", "val", "test"):
        classes_in_split = sorted(d.name for d in (PROCESSED_DIR / split).iterdir()
                                  if d.is_dir())
        if classes_in_split != sorted(class_counts.keys()):
            raise RuntimeError(f"Le split '{split}' n'a pas toutes les classes !")

    print(f"\n✅ Split terminé. {len(class_counts)} classes × 3 splits.")


if __name__ == "__main__":
    split_dataset(force_redo=False)

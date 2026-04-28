"""
segmentation.py
---------------
Prétraitement et segmentation de la feuille :

  Étape 1 — Filtrage du bruit (Filtre Gaussien 5×5)
    → Supprime le bruit de capteur et les artéfacts de compression JPEG
    → Le noyau 5×5 est un compromis : assez grand pour lisser le bruit de
       capteur (~1-2 px), assez petit pour préserver les contours des taches
       (Early/Late blight ont des bords nets qu'on veut garder).

  Étape 2 — Redimensionnement à 224×224
    → Taille standard imposée par MobileNetV2 (Transfer Learning ImageNet)
    → Identique pour ML (extracteurs) et DL (réseau convolutif)

  Étape 3 — Conversion BGR → HSV
    → L'espace HSV sépare la teinte (H) de la luminosité (V).
    → Beaucoup plus robuste au changement d'éclairage que RGB/BGR :
       une feuille verte reste à H≈60° quelle que soit l'intensité lumineuse.
    → Permet un masquage par plage de teinte simple et efficace.

  Étape 4 — Double masque HSV (vert + brun) + morphologie
    → Masque vert (H: 22–95) : zones saines / légèrement malades
    → Masque brun (H: 5–22)  : zones nécrosées (late blight, early blight)
    → Opérations OPEN (élimine les pixels isolés) + CLOSE (bouche les trous)
       avec un noyau elliptique 7×7.
"""

import cv2
import numpy as np

IMG_SIZE = 224  # Taille cible imposée par MobileNetV2


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applique le pipeline de prétraitement standard :
      1. Redimensionnement à 224×224
      2. Filtre Gaussien 5×5 (débruitage)

    Retourne l'image BGR prétraitée.
    """
    # Étape 1 — Redimensionnement
    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)

    # Étape 2 — Filtre Gaussien (noyau 5×5, σ auto)
    # Choix du 5×5 : suffisant pour supprimer le bruit JPEG/capteur
    # sans flouter les contours des taches foliaires (> 3 px en général)
    blurred = cv2.GaussianBlur(resized, (5, 5), sigmaX=0)
    return blurred


def segment_leaf(image_path: str):
    """
    Segmente la feuille depuis son arrière-plan par double masque HSV.

    Paramètres
    ----------
    image_path : str
        Chemin absolu ou relatif vers l'image.

    Retourne
    --------
    image_rgb   : np.ndarray  (H, W, 3) uint8 — image originale en RGB
    mask        : np.ndarray  (H, W)    uint8 — masque binaire (0/255)
    segmented   : np.ndarray  (H, W, 3) uint8 — image avec fond noir
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    # ── Prétraitement ──────────────────────────────────────────────────────
    # 1. Redimensionnement + Débruitage Gaussien (noyau 5×5)
    image_bgr = preprocess_image(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # Masque 1 — Vert / Jaune-vert (feuilles normales et légèrement malades)
    # H : 22–95, S : 30+, V : 30+
    # ------------------------------------------------------------------
    lower_green = np.array([22,  30,  30])
    upper_green = np.array([95, 255, 255])
    mask_green  = cv2.inRange(hsv, lower_green, upper_green)

    # ------------------------------------------------------------------
    # Masque 2 — Brun / Orange (nécrose avancée, taches sombres)
    # H : 5–22, S : 50+, V : 40+
    # ------------------------------------------------------------------
    lower_brown = np.array([ 5,  50,  40])
    upper_brown = np.array([22, 255, 200])
    mask_brown  = cv2.inRange(hsv, lower_brown, upper_brown)

    # Fusion des deux masques
    mask = cv2.bitwise_or(mask_green, mask_brown)

    # ------------------------------------------------------------------
    # Nettoyage morphologique (suppression du bruit + comblement de trous)
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # ------------------------------------------------------------------
    # Application du masque
    # ------------------------------------------------------------------
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return image_rgb, mask, segmented


if __name__ == "__main__":
    print("Module segmentation prêt. Utilisez segment_leaf(path).")

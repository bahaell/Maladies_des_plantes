"""
segmentation.py
---------------
Isole la feuille de son arrière-plan par seuillage HSV double :
  - Masque vert  : feuilles saines / légèrement malades
  - Masque brun  : feuilles très malades (nécrose avancée)
Résultat : image segmentée + masque binaire propre.
"""

import cv2
import numpy as np


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

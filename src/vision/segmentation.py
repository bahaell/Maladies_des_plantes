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


# ===========================================================================
# Détection de contours — Sobel
# ===========================================================================
def detect_edges_sobel(image_rgb: np.ndarray) -> np.ndarray:
    """
    Détection de contours par gradient de Sobel.

    Applique les opérateurs Sobel en X et Y sur l'image en niveaux de gris,
    puis calcule le gradient total : G = √(Gx² + Gy²).

    Paramètres
    ----------
    image_rgb : np.ndarray  (H, W, 3) uint8 en RGB

    Retourne
    --------
    edges : np.ndarray  (H, W) uint8, contours en blanc sur fond noir
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Gradient horizontal (détecte les bords verticaux)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    # Gradient vertical (détecte les bords horizontaux)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    # Magnitude du gradient
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalisation sur [0, 255]
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-10))
    return magnitude


# ===========================================================================
# Détection de contours — Canny
# ===========================================================================
def detect_edges_canny(image_rgb: np.ndarray,
                       low_threshold: int = 50,
                       high_threshold: int = 150) -> np.ndarray:
    """
    Détection de contours par algorithme de Canny (double seuillage).

    Canny améliore Sobel en supprimant les faux contours grâce à :
      1. Lissage Gaussien interne
      2. Non-maxima suppression (amincissement des contours)
      3. Double seuillage par hystérésis (low=50, high=150)

    Paramètres
    ----------
    image_rgb      : np.ndarray (H, W, 3) uint8 en RGB
    low_threshold  : seuil bas de l'hystérésis  (défaut=50)
    high_threshold : seuil haut de l'hystérésis (défaut=150)

    Retourne
    --------
    edges : np.ndarray (H, W) uint8, contours binaires
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


# ===========================================================================
# Segmentation par Seuillage d'Otsu
# ===========================================================================
def segment_otsu(image_rgb: np.ndarray) -> tuple:
    """
    Segmentation par seuillage automatique d'Otsu sur l'image en niveaux de gris.

    L'algorithme d'Otsu cherche le seuil T qui minimise la variance
    intra-classe (ou maximise la variance inter-classes). Il est optimal
    pour les histogrammes bimodaux (fond clair vs feuille sombre ou l'inverse).

    Paramètres
    ----------
    image_rgb : np.ndarray (H, W, 3) uint8

    Retourne
    --------
    thresh_val : int          — valeur du seuil calculé automatiquement
    mask_otsu  : np.ndarray   — masque binaire (H, W) uint8
    segmented  : np.ndarray   — image segmentée (H, W, 3) uint8
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh_val, mask_otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # OTSU calcule T automatiquement
    )
    # Nettoyage morphologique léger
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_otsu)
    return int(thresh_val), mask_otsu, segmented


# ===========================================================================
# Segmentation par Clustering K-Means
# ===========================================================================
def segment_kmeans(image_rgb: np.ndarray, k: int = 3) -> tuple:
    """
    Segmentation par clustering K-Means dans l'espace RGB.

    K-Means partitionne les pixels en K clusters selon leur distance
    Euclidienne en RGB. Chaque cluster correspond à une région homogène.
    On identifie le cluster "feuille" comme celui dont la moyenne G est
    la plus élevée (feuilles = dominante verte).

    Paramètres
    ----------
    image_rgb : np.ndarray (H, W, 3) uint8
    k         : int — nombre de clusters (défaut=3 : fond, feuille, tache)

    Retourne
    --------
    labels    : np.ndarray (H, W) int   — label de cluster par pixel
    centers   : np.ndarray (K, 3) float — couleur moyenne de chaque cluster
    segmented : np.ndarray (H, W, 3)    — image avec fond noir
    """
    h, w = image_rgb.shape[:2]
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels_flat, centers = cv2.kmeans(
        pixels, k, None, criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS  # Initialisation K-Means++ (plus stable)
    )

    labels = labels_flat.reshape(h, w)
    centers = np.uint8(centers)

    # Identifier le cluster "feuille" : le plus vert (canal G le plus élevé)
    leaf_cluster = int(np.argmax(centers[:, 1]))  # canal G = index 1
    mask_kmeans = np.uint8((labels == leaf_cluster) * 255)
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_kmeans)

    return labels, centers, segmented


if __name__ == "__main__":
    print("Module segmentation prêt. Utilisez segment_leaf(path).")


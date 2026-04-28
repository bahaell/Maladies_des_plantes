"""
extractors.py
-------------
Extraction de caractéristiques visuelles à quatre niveaux :
  1. Couleur RGB — Histogramme RGB 1D (8 bins × 3 canaux = 24 dimensions)
  2. Couleur HSV — Histogramme HSV 3D (8×8×8 = 512 dimensions)
  3. Texture     — GLCM Haralick via scikit-image (20 dimensions)
  4. Forme       — Moments de Hu + métriques de contour (10 dimensions)

Vecteur final : 566 dimensions
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ---------------------------------------------------------------------------
# 0. COULEUR — Histogramme RGB (comparaison avec HSV)
# ---------------------------------------------------------------------------
def extract_rgb_histogram(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogramme RGB 1D calculé sur chaque canal séparément.
    Bins : 8 par canal → 3 × 8 = 24 valeurs normalisées.

    Utilisé pour la comparaison visuelle RGB vs HSV dans le notebook.
    """
    features = []
    for channel in range(3):  # R, G, B
        hist = cv2.calcHist([image_rgb], [channel], mask, [8], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features.append(hist.flatten())
    return np.concatenate(features)  # 24 valeurs


# ---------------------------------------------------------------------------
# 1. COULEUR — Histogramme HSV normalisé
# ---------------------------------------------------------------------------
def extract_color_histogram(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogramme HSV 3D calculé uniquement sur les pixels de la feuille.
    Bins : H=8, S=8, V=8 → 512 valeurs normalisées.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()  # 512 valeurs


# ---------------------------------------------------------------------------
# 2. TEXTURE — GLCM Haralick (scikit-image)
# ---------------------------------------------------------------------------
def extract_glcm_texture(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Matrice de Co-occurrence des Niveaux de Gris (GLCM).
    Propriétés extraites : contraste, dissimilarité, homogénéité,
                           énergie, corrélation — sur 4 angles.
    → 20 valeurs (5 propriétés × 4 angles).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Appliquer le masque : remplacer les pixels hors-feuille par 0
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Redimensionner pour accélérer le calcul GLCM
    resized = cv2.resize(masked_gray, (64, 64))

    # Réduire à 64 niveaux pour un GLCM plus représentatif
    resized = (resized // 4).astype(np.uint8)

    distances  = [1]
    angles     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    glcm = graycomatrix(resized, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)

    features = []
    for prop in properties:
        values = graycoprops(glcm, prop).flatten()  # 1 dist × 4 angles → 4 valeurs
        features.extend(values)

    return np.array(features, dtype=np.float32)  # 20 valeurs


# ---------------------------------------------------------------------------
# 3. FORME — Moments de Hu + Métriques de contour
# ---------------------------------------------------------------------------
def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """
    Caractéristiques de forme extraites depuis le masque binaire :
      - 7 moments de Hu (invariants à rotation/scale/translation)
      - Ratio aire feuille / aire image
      - Circularité  = 4π × Aire / Périmètre²
      - Compacité    = Périmètre² / Aire
    → 10 valeurs.
    """
    moments = cv2.moments(mask)
    hu      = cv2.HuMoments(moments).flatten()

    # Log-transform pour stabiliser les ordres de grandeur
    hu_log = np.array([-np.copysign(np.log10(abs(h) + 1e-10), h) for h in hu],
                      dtype=np.float32)

    # Contours pour périmètre / aire
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt       = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        img_area  = mask.shape[0] * mask.shape[1]

        leaf_ratio   = area / img_area if img_area > 0 else 0.0
        circularity  = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        compactness  = (perimeter ** 2 / area) if area > 0 else 0.0
    else:
        leaf_ratio = circularity = compactness = 0.0

    shape_metrics = np.array([leaf_ratio, circularity, compactness], dtype=np.float32)
    return np.concatenate([hu_log, shape_metrics])  # 10 valeurs


# ---------------------------------------------------------------------------
# Vecteur final unifié
# ---------------------------------------------------------------------------
def extract_features(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Construit le vecteur de caractéristiques complet :
      - Couleur  : 512 valeurs  (histogramme HSV)
      - Texture  :  20 valeurs  (GLCM Haralick)
      - Forme    :  10 valeurs  (Hu Moments + métriques)
    Total : 566 dimensions.
    """
    rgb     = extract_rgb_histogram(image_rgb, mask)      #  24
    color   = extract_color_histogram(image_rgb, mask)    # 512
    texture = extract_glcm_texture(image_rgb, mask)       #  20
    shape   = extract_shape_features(mask)                #  10

    return np.concatenate([rgb, color, texture, shape]).astype(np.float32)


# 📋 Rapport Technique — Étape 3 : Extraction de Caractéristiques (Features)

> **Projet :** Détection et Classification Automatique de Maladies des Plantes  
> **Section :** Extraction de Caractéristiques Visuelles  
> **Fichier principal :** `src/features/extractors.py`

---

## Pourquoi Extraire des Caractéristiques ?

Les classifieurs ML classiques (Random Forest, SVM) ne peuvent pas traiter directement les pixels bruts d'une image 224×224 (soit 224×224×3 = **150 528 valeurs**). Ce vecteur est trop haut-dimensionnel, redondant et non-invariant (une simple rotation change toutes les valeurs).

L'extraction de features transforme l'image en un vecteur **compact, informatif et invariant** qui capture l'essence visuelle de la maladie :

- **Compact** : 566 valeurs au lieu de 150 528 → gain d'un facteur 266×
- **Informatif** : chaque valeur a une signification physique (couleur, texture, forme)
- **Invariant** : les moments de Hu sont invariants à la rotation, translation et mise à l'échelle

---

## Vue d'Ensemble du Vecteur Final

```
Image segmentée (224×224×3)
         │
         ├── Histogramme RGB  → 24 valeurs
         ├── Histogramme HSV  → 512 valeurs
         ├── GLCM Haralick    → 20 valeurs
         └── Forme + Hu       → 10 valeurs
         │
         ▼
  Vecteur de 566 dimensions
         │
         ▼
  StandardScaler → Normalisation μ=0, σ=1
         │
         ▼
  Classifieur RF / SVM
```

---

## Feature 1 — Histogramme RGB (24 dimensions)

### Principe

L'histogramme RGB mesure la **distribution des intensités** sur chaque canal de couleur (Rouge, Vert, Bleu) séparément.

Avec **8 bins par canal**, on divise la plage [0, 255] en 8 intervalles de 32 niveaux chacun. On compte les pixels dans chaque intervalle.

### Pourquoi 8 bins ?

| Nombre de bins | Avantages | Inconvénients |
|----------------|-----------|---------------|
| 4 bins | Très compact | Perd trop de détails |
| **8 bins** | **Bon compromis** | **24 valeurs par couleur** |
| 16 bins | Plus précis | 48 valeurs, redondant avec HSV |
| 256 bins | Maximum de précision | Trop de bruit, overfitting |

### Calcul sur le masque uniquement

```python
hist = cv2.calcHist([image_rgb], [channel], mask, [8], [0, 256])
```

Le paramètre `mask` (masque binaire de la feuille) garantit que seuls les pixels de la feuille sont comptés. Le fond noir (0,0,0) est exclu.

### Comparaison RGB vs HSV

| Critère | RGB | HSV |
|---------|-----|-----|
| Robustesse à l'éclairage | Faible (R, G, B varient) | Forte (H reste stable) |
| Interprétabilité | Intuitive | Moins intuitive |
| Usage principal | Baseline de comparaison | Production (classifieur) |

L'histogramme RGB est utile pour la **comparaison visuelle** dans le notebook. L'histogramme HSV est celui utilisé dans le pipeline de production car il est plus discriminant.

---

## Feature 2 — Histogramme HSV 3D (512 dimensions)

### Principe

Contrairement à l'histogramme RGB qui traite les canaux séparément, l'histogramme HSV 3D considère les **combinaisons de (H, S, V)** simultanément. Chaque bin correspond à une région dans l'espace HSV.

Avec **8 bins par canal** : 8 × 8 × 8 = **512 bins** au total.

### Pourquoi un histogramme 3D et pas 1D ?

Un histogramme 1D sur chaque canal HSV (3 × 8 = 24 valeurs) ignorerait les corrélations entre canaux. Par exemple, un pixel avec H=60° (vert), S=200 (très saturé), V=100 (sombre) correspond à une zone nécrosée sombre, différente d'un pixel H=60°, S=50, V=200 (vert clair). L'histogramme 3D capture ces corrélations.

### Normalisation min-max

```python
cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
```

La normalisation ramène toutes les valeurs dans [0, 1]. Sans elle, les images avec plus de pixels dans la feuille (feuilles grandes) auraient des histogrammes avec des valeurs absolues plus élevées, biaisant le classifieur.

---

## Feature 3 — Texture GLCM (Gray Level Co-occurrence Matrix) — 20 dimensions

### Principe

La GLCM mesure la **fréquence à laquelle deux pixels, séparés d'une distance d dans une direction θ, ont des intensités données**. Elle capture la texture de la surface foliaire.

Formellement, pour une image I, la GLCM est définie par :
```
P(i, j, d, θ) = nombre de paires (p, q) telles que
                I(p) = i, I(q) = j
                et q est à distance d de p dans la direction θ
```

### Propriétés extraites (5 × 4 angles = 20 valeurs)

| Propriété | Description | Élevée si |
|-----------|-------------|-----------|
| **Contraste** | Variation locale d'intensité | Texture rugueuse, beaucoup de taches |
| **Dissimilarité** | Somme |i-j| pondérée | Grandes variations de gris |
| **Homogénéité** | Concentration sur la diagonale | Texture uniforme (feuille saine) |
| **Énergie** | Uniformité de la distribution | Texture répétitive |
| **Corrélation** | Dépendance linéaire voisins | Texture orientée (veines) |

### Paramètres choisis

```python
distances  = [1]                              # Distance de 1 pixel (voisinage immédiat)
angles     = [0, π/4, π/2, 3π/4]             # 4 orientations : 0°, 45°, 90°, 135°
levels     = 64                               # 64 niveaux de gris (réduit de 256 pour la vitesse)
symmetric  = True                             # P(i,j) = P(j,i) → matrice symétrique
normed     = True                             # Probabilités (somme = 1)
```

**Distance = 1** : capture la co-occurrence des pixels directement adjacents, ce qui correspond aux structures microscopiques des maladies (sporulation, nécrose cellulaire).

**64 niveaux de gris** (au lieu de 256) : réduction de la quantification pour réduire la mémoire et le temps de calcul sans perdre d'information discriminante. La GLCM 256×256 est 16× plus grande que la GLCM 64×64.

**4 angles (0°, 45°, 90°, 135°)** : rendent l'analyse **isotrope** (indépendante de l'orientation). Les taches de maladies n'ont pas d'orientation préférentielle.

### Application sur une image réduite (64×64)

```python
resized = cv2.resize(masked_gray, (64, 64))
```

Le redimensionnement à 64×64 réduit le temps de calcul de la GLCM de **224²/64² = 12× environ**, sans perte significative d'information texturale (les patterns de texture sont visibles à cette résolution).

---

## Feature 4 — Forme (10 dimensions)

### 4.1 Moments de Hu (7 valeurs)

Les **Moments de Hu** (1962) sont 7 valeurs invariantes par rapport à :
- **Translation** (déplacement de la feuille)
- **Rotation** (orientation de la feuille)
- **Mise à l'échelle** (distance de la caméra)

Ils sont calculés à partir des moments d'ordre 2 et 3 de la distribution des pixels du masque.

Leur transformation logarithmique est appliquée pour gérer les ordres de grandeur très différents :
```python
hu_log = -np.copysign(np.log10(abs(h) + 1e-10), h)
```

### 4.2 Ratio Aire (1 valeur)

```
Ratio = Aire contour principal / Aire totale image
```

Mesure la **proportion de la feuille** dans l'image. Une grande feuille saine → ratio élevé. Une feuille très malade et nécrosée → ratio plus faible (les zones mortes sont souvent exclues du masque vert).

### 4.3 Circularité (1 valeur)

```
Circularité = 4π × Aire / Périmètre²
```

- Valeur = 1.0 pour un cercle parfait
- Valeur = 0 pour une forme très allongée ou déchiquetée

Les feuilles saines ont une forme régulière (circularité ~0.3-0.6). Les feuilles très malades sont souvent déformées (circularité plus faible).

### 4.4 Compacité (1 valeur)

```
Compacité = Périmètre² / Aire
```

Inverse normalisé de la circularité. Élevée pour les formes complexes (beaucoup de bords denticulés, taches qui découpent le contour).

---

## Normalisation — StandardScaler

Avant d'être passé aux classifieurs, le vecteur de 566 dimensions est **normalisé** par le `StandardScaler` de scikit-learn :

```
x_normalisé = (x - μ) / σ
```

- **μ** = moyenne calculée sur le set d'entraînement uniquement
- **σ** = écart-type calculé sur le set d'entraînement uniquement

### Pourquoi normaliser ?

| Sans normalisation | Avec normalisation |
|-------------------|-------------------|
| L'histogramme HSV (512 valeurs, plage [0,1]) domine | Toutes les features contribuent équitablement |
| Le SVM est sensible à l'échelle → très mauvaises performances | SVM optimal sur données centrées-réduites |
| RF moins affecté mais tout de même amélioré | RF légèrement amélioré |

### Pourquoi StandardScaler et pas MinMaxScaler ?

`StandardScaler` est robuste aux **outliers** (pixels aberrants → histogramme avec un bin à valeur extrême). `MinMaxScaler` est très sensible aux outliers : un seul bin élevé compresse tous les autres vers 0.

---

## Résumé du Vecteur Final (566 dimensions)

| Feature | Dimensions | Méthode | Invariant à |
|---------|------------|---------|-------------|
| Histogramme RGB | 24 | `cv2.calcHist` 1D | — |
| Histogramme HSV 3D | 512 | `cv2.calcHist` 3D | Éclairage (canal H) |
| GLCM Haralick | 20 | `skimage.graycomatrix` | Orientation (4 angles) |
| Moments de Hu | 7 | `cv2.HuMoments` | Rotation, translation, scale |
| Métriques de forme | 3 | `cv2.contourArea/arcLength` | Taille absolue |
| **TOTAL** | **566** | | |

---

*Document généré dans le cadre du projet académique — Classification des Maladies des Plantes*

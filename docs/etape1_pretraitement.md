# 📋 Rapport Technique — Étape 1 : Prétraitement des Images

> **Projet :** Détection et Classification Automatique de Maladies des Plantes  
> **Section :** Pipeline de Prétraitement  
> **Fichier principal :** `src/vision/segmentation.py`

---

## Pourquoi le Prétraitement est-il Indispensable ?

Le prétraitement est la **première étape critique** du pipeline. Des images brutes présentent systématiquement plusieurs problèmes qui dégradent les performances des algorithmes d'apprentissage :

- **Tailles hétérogènes :** PlantVillage contient des images allant de 256×256 à plus de 1024×768 pixels. Un réseau de neurones ou un extracteur de features attend une taille fixe.
- **Bruit de capteur et artéfacts JPEG :** Chaque image compressée en JPEG présente une quantification par blocs de 8×8 pixels. Ce bruit haute-fréquence pollue la matrice GLCM (texture) et perturbe le seuillage HSV.
- **Variabilité de l'éclairage :** Une même feuille verte photographiée à l'ombre ou en plein soleil aura des valeurs RGB totalement différentes. Le modèle ne doit pas confondre l'éclairage avec une maladie.

Notre pipeline de prétraitement résout ces trois problèmes de manière séquentielle.

---

## Vue d'Ensemble du Pipeline

```
Image brute (taille variable, RGB/BGR)
         │
         ▼
  ┌──────────────────────────┐
  │  Étape 1 : Resize 224×224│  → Taille fixe pour ML/DL
  └──────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Étape 2 : Filtre Gauss. │  → Suppression du bruit JPEG/capteur
  └──────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Étape 3 : RGB → HSV     │  → Robustesse à l'éclairage
  └──────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Étape 4 : Split 70/15/15│  → Partitionnement déterministe
  └──────────────────────────┘
         │
         ▼
  Image prête pour Segmentation, ML et DL
```

---

## Étape 1.1 — Redimensionnement des Images (224 × 224)

### Explication

Toutes les images du dataset sont redimensionnées à **224 × 224 pixels** avant tout traitement. Cette étape est la première à être appliquée.

### Pourquoi 224 × 224 ?

Cette taille n'est pas arbitraire. Elle est imposée par l'architecture **MobileNetV2**, qui a été pré-entraîné sur le dataset ImageNet avec des entrées de taille `224 × 224 × 3`.

En Transfer Learning, pour réutiliser les poids d'ImageNet directement (ce qui est notre stratégie en Phase 1), les images doivent avoir exactement cette taille. Choisir une taille différente obligerait à redimensionner en interne ou à perdre la compatibilité avec les couches convolutives.

De plus, utiliser **la même taille en ML et en DL** garantit que les vecteurs de features (histogramme HSV, GLCM) sont calculés sur des images cohérentes entre elles, ce qui rend le benchmark ML vs DL équitable.

### Pourquoi l'interpolation `INTER_AREA` ?

OpenCV propose plusieurs méthodes d'interpolation. Notre choix est `cv2.INTER_AREA` :

| Méthode | Description | Cas d'usage idéal |
|---------|-------------|-------------------|
| `INTER_NEAREST` | Voisin le plus proche | Masques binaires, très rapide |
| `INTER_LINEAR` | Bilinéaire (défaut OpenCV) | Agrandissement (upscaling) |
| `INTER_CUBIC` | Bicubique | Agrandissement haute qualité |
| **`INTER_AREA`** | **Décimation par moyennage** | **Réduction (downscaling) ← Notre cas** |

Les images PlantVillage font en général **256×256 à 1024×768 pixels**. On effectue donc une **réduction**. `INTER_AREA` calcule la valeur de chaque pixel de destination en faisant la moyenne des pixels sources correspondants, ce qui évite l'aliasing (crénelage) et conserve mieux les détails que `INTER_LINEAR` pour la réduction.

### Code

```python
# src/vision/segmentation.py
IMG_SIZE = 224  # Taille cible imposée par MobileNetV2

def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    ...
```

### Impact Visuel

L'image est ramenée à un format standard. L'information visuelle (taches, couleurs, textures) est conservée. Les petites différences de dimensions disparaissent.

---

## Étape 1.2 — Filtrage du Bruit (Filtre Gaussien 5 × 5)

### Explication

Après le redimensionnement, on applique un **filtre de lissage Gaussien** pour supprimer le bruit haute-fréquence présent dans les images.

### Le Problème : d'où vient le bruit ?

Deux sources de bruit affectent nos images :

1. **Bruit de capteur photographique** : Chaque pixel d'un capteur CMOS/CCD mesure un nombre de photons, mais avec un bruit aléatoire quantique (distribution de Poisson, assimilable à un bruit Gaussien). Ce bruit est de 1 à 3 pixels de rayon.

2. **Artéfacts de compression JPEG** : Le format JPEG compresse les images par blocs de **8×8 pixels** (Transformée en Cosinus Discrète). À des taux de compression moyens ou forts, des transitions abruptes (blocs) apparaissent aux frontières. Ces blocs créent de fausses discontinuités qui faussent la matrice GLCM (qui mesure justement la co-occurrence des niveaux de gris).

### La Solution : Filtre Gaussien

Le filtre Gaussien lisse l'image en remplaçant chaque pixel par une **moyenne pondérée** de ses voisins. Les poids suivent une distribution Gaussienne (plus élevés au centre, décroissants vers les bords), ce qui donne un résultat plus naturel qu'un filtre moyen uniforme.

### Pourquoi un noyau **5 × 5** et pas autre chose ?

Le choix du noyau est un **compromis signal/bruit** critique :

| Noyau | Bruit supprimé | Contours préservés | Verdict |
|-------|---------------|-------------------|---------|
| 3×3 | Faible | Très bien préservés | Lissage insuffisant |
| **5×5** | **Bon** | **Bien préservés** | **✅ Choix optimal** |
| 7×7 | Très fort | Dégradés (flou visible) | Trop agressif |
| 11×11 | Excessif | Détruits | Inacceptable |

Les taches de maladies (Early blight, Septoria) ont des **contours nets de plus de 3 pixels** de rayon. Le noyau 5×5 (rayon = 2px) est assez petit pour ne pas les flouter, mais assez grand pour couvrir le bruit de capteur (~1-2px).

### Pourquoi σ = 0 (automatique) ?

En OpenCV, `sigmaX=0` déclenche le calcul automatique : `σ = 0.3 × ((ksize-1) × 0.5 - 1) + 0.8`.

Pour un noyau 5×5 : `σ = 0.3 × (2.5 - 1) + 0.8 = 0.3 × 1.5 + 0.8 = **1.25**`

Cette valeur de **σ ≈ 1.25** est la valeur recommandée dans la littérature de traitement d'images médicales pour un lissage conservateur (ex: Gonzalez & Woods, *Digital Image Processing*). On l'utilise car elle est :
- Suffisante pour supprimer le bruit quantique
- Trop faible pour flouter les structures biologiques (taches, veines)

### Pourquoi pas un Filtre Médian ?

Le filtre médian est excellent pour le bruit impulsionnel (bruit sel & poivre : pixels aléatoirement noirs ou blancs). Mais notre bruit est de nature **Gaussienne** (photons, quantification DCT). Pour le bruit Gaussien, le filtre Gaussien est mathématiquement optimal. De plus, le filtre médian est **3× plus lent** sur CPU.

### Code

```python
# src/vision/segmentation.py
blurred = cv2.GaussianBlur(resized, (5, 5), sigmaX=0)
```

---

## Étape 1.3 — Conversion BGR → HSV

### Explication

Après le débruitage, l'image est convertie de l'espace **BGR** (format interne OpenCV) vers l'espace **HSV** (Teinte / Saturation / Valeur).

### Qu'est-ce que l'espace HSV ?

HSV est un espace colorimétrique **cylindrique** qui décompose la couleur en 3 composantes indépendantes :

| Composante | Plage (OpenCV) | Description |
|------------|---------------|-------------|
| **H** (Hue / Teinte) | 0–180° | La couleur pure (rouge, vert, bleu...) |
| **S** (Saturation) | 0–255 | L'intensité/pureté de la couleur |
| **V** (Value / Valeur) | 0–255 | La luminosité |

> **Note OpenCV :** En OpenCV, H est compris entre 0 et 180 (pas 360°) pour tenir sur un `uint8`. Chaque degré vaut 2° réels.

### Pourquoi HSV et pas RGB ?

En RGB, les trois canaux sont **corrélés** avec l'éclairage. Si on augmente la luminosité d'une feuille verte, R, G **et** B augmentent tous les trois simultanément. Il est impossible de définir un seuil fixe sur RGB pour isoler "le vert".

En HSV, la **Teinte (H) est invariante à la luminosité**. Une feuille verte a toujours H ≈ 60–90°, qu'elle soit photographiée à l'ombre ou en plein soleil. Seul V change avec la luminosité.

```
RGB → dépend de l'éclairage → Seuillage impossible
HSV → H stable à l'éclairage → Seuillage par plage de teinte fiable
```

### Mapping HSV des pathologies de notre dataset

| Maladie | Teinte H (approx.) | Couleur observée |
|---------|-------------------|-----------------|
| Feuille saine | 60–95° | Vert |
| Early blight (début) | 35–60° | Jaune-vert |
| Early blight (avancé) | 5–22° | Brun/orange |
| Late blight | 0–10° et faible S | Brun très sombre → noir |
| Common rust (rouille) | 20–30° | Orange vif |
| Septoria | 5–20° | Beige/brun clair |

Ce tableau justifie nos deux masques HSV (§ Segmentation, étape suivante) :
- **Masque vert** : H ∈ [22°, 95°] → capture les zones saines et semi-saines
- **Masque brun** : H ∈ [5°, 22°] → capture les zones nécrosées

### Code

```python
# src/vision/segmentation.py
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
```

> **Note importante :** OpenCV charge les images en **BGR** (Blue-Green-Red) par défaut, contrairement à Matplotlib qui attend du **RGB**. La conversion `BGR→RGB` est faite pour l'affichage Streamlit et Matplotlib. La conversion `BGR→HSV` est faite pour la segmentation.

---

## Étape 1.4 — Partitionnement Stratifié du Dataset (70 / 15 / 15)

### Explication

Le dataset de **13 563 images** est divisé en trois sous-ensembles non chevauchants : Train, Validation et Test.

### Les Ratios : 70% / 15% / 15%

| Sous-ensemble | Ratio | Rôle |
|---------------|-------|------|
| **Train** | 70% (9 490 images) | Apprentissage des paramètres des modèles |
| **Validation** | 15% (2 029 images) | Réglage des hyperparamètres, Early Stopping |
| **Test** | 15% (2 044 images) | Évaluation finale non biaisée |

Le ratio **70/15/15** est un standard académique largement adopté pour les datasets de taille moyenne (5 000–100 000 images). Un ratio 80/10/10 serait envisageable pour de plus grands datasets.

### Pourquoi "Stratifié" ?

Le partitionnement **stratifié** garantit que chaque sous-ensemble contient **la même proportion** de chaque classe. Sans stratification, le hasard pourrait surrepésenter une classe dans Train (fuite de données déguisée) ou sous-représenter une classe dans Test (évaluation biaisée).

```
Tomato___Septoria_leaf_spot : 1771 images
  → Train : 1239  (70%)
  → Val   :  265  (15%)
  → Test  :  267  (15%)
```

### Graine Aléatoire seed=42

La graine `random.seed(42)` fixe le générateur pseudo-aléatoire de Python. Cela rend le split **parfaitement reproductible** : deux exécutions du script sur le même dataset produisent exactement les mêmes fichiers dans train/val/test.

La valeur 42 est une convention de facto dans la communauté ML (issue de *The Hitchhiker's Guide to the Galaxy*). N'importe quelle valeur entière fixe aurait le même effet.

### Nettoyage Préalable : Prévention de la Fuite de Données

```python
def _clean_processed() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)  # Suppression complète
    for split in ("train", "val", "test"):
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)
```

Cette suppression préalable est **non négociable**. Sans elle, une deuxième exécution du script pourrait ajouter de nouvelles images sans supprimer les anciennes, créant un chevauchement entre train et test (**data leakage**). Ce biais ferait artificiellement monter l'accuracy de test.

### Code

```python
# src/data/splitter.py
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(RANDOM_SEED)
files = [f.name for f in src_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
random.shuffle(files)

n_train = int(total * TRAIN_RATIO)
n_val   = int(total * VAL_RATIO)
splits = {
    "train": files[:n_train],
    "val":   files[n_train:n_train + n_val],
    "test":  files[n_train + n_val:]
}
```

---

## Résumé des Choix Techniques

| Paramètre | Valeur choisie | Alternatives rejetées | Raison du choix |
|-----------|---------------|----------------------|-----------------|
| Taille image | **224×224** | 128×128, 256×256 | Compatibilité MobileNetV2 (ImageNet standard) |
| Interpolation resize | **INTER_AREA** | INTER_LINEAR, INTER_CUBIC | Optimal pour la réduction d'image |
| Filtre débruitage | **Gaussien 5×5** | Médian, Bilatéral, Aucun | Bruit Gaussien (JPEG/capteur), préserve contours |
| Sigma Gaussien | **σ=0 (auto=1.25)** | σ=1, σ=2 | Valeur littérature pour lissage conservateur |
| Espace de couleur | **HSV** | RGB, LAB, YUV | H invariant à la luminosité → seuillage fiable |
| Split | **70/15/15** | 80/10/10, 60/20/20 | Standard académique pour datasets ~10 000 images |
| Seed | **42** | Toute autre valeur fixe | Convention ML, valeur de référence |

---

*Document généré dans le cadre du projet académique — Classification des Maladies des Plantes*

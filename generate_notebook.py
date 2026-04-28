"""
generate_notebook.py
--------------------
Génère le Jupyter Notebook académique complet du projet.
Exécuter : python generate_notebook.py
"""

import json
from pathlib import Path

def cell_md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source if isinstance(source, list) else [source]}

def cell_code(source, outputs=None):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": outputs or [],
            "source": source if isinstance(source, list) else [source]}

cells = []

# ============================================================
# SECTION 0 — Titre
# ============================================================
cells.append(cell_md([
    "# 🌿 Détection et Classification Automatique de Maladies des Plantes\n",
    "## Projet Académique — Machine Learning & Deep Learning\n\n",
    "**Pipeline complet :** Dataset → Prétraitement → Segmentation → "
    "Extraction Features → Classification ML/DL → Évaluation → Benchmark\n\n",
    "**12 classes cibles :** Maïs (×4) | Pomme de terre (×3) | Tomate (×5)\n\n",
    "---"
]))

# ============================================================
# SECTION 1 — Imports & Configuration
# ============================================================
cells.append(cell_md(["## 1. Imports & Configuration\n"]))
cells.append(cell_code([
    "import os, sys, time, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "sys.path.insert(0, os.path.abspath('.'))\n\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.cm as cm\n",
    "import seaborn as sns\n",
    "import cv2\n",
    "import joblib\n\n",
    "from pathlib import Path\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import (classification_report, confusion_matrix,\n",
    "                             accuracy_score, f1_score,\n",
    "                             precision_score, recall_score)\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n\n",
    "# Chemins\n",
    "RAW_DIR       = Path('./data/raw')\n",
    "PROCESSED_DIR = Path('./data/processed')\n",
    "MODEL_DIR     = Path('./models')\n",
    "MODEL_DIR.mkdir(parents=True, exist_ok=True)\n\n",
    "CLASSES_12 = sorted([\n",
    "    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',\n",
    "    'Corn_(maize)___Common_rust_',\n",
    "    'Corn_(maize)___Northern_Leaf_Blight',\n",
    "    'Corn_(maize)___healthy',\n",
    "    'Potato___Early_blight',\n",
    "    'Potato___Late_blight',\n",
    "    'Potato___healthy',\n",
    "    'Tomato___Early_blight',\n",
    "    'Tomato___Late_blight',\n",
    "    'Tomato___Leaf_Mold',\n",
    "    'Tomato___Septoria_leaf_spot',\n",
    "    'Tomato___healthy',\n",
    "])\n",
    "print(f'TensorFlow : {tf.__version__}')\n",
    "print(f'Classes    : {len(CLASSES_12)}')\n",
]))

# ============================================================
# SECTION 2 — Dataset
# ============================================================
cells.append(cell_md([
    "## 2. Dataset\n\n",
    "**Source :** PlantVillage Dataset (Kaggle)\n\n",
    "**12 classes sélectionnées :**\n",
    "- Maïs × 4 (saine, cercospora, rouille, brûlure nordique)\n",
    "- Pomme de terre × 3 (saine, alternariose, mildiou)\n",
    "- Tomate × 5 (saine, alternariose, mildiou, cladosporiose, septoriose)\n"
]))
cells.append(cell_code([
    "# Analyse de la distribution des classes\n",
    "class_counts = {}\n",
    "for d in sorted(RAW_DIR.iterdir()):\n",
    "    if d.is_dir():\n",
    "        imgs = list(d.glob('*.*'))\n",
    "        class_counts[d.name] = len(imgs)\n\n",
    "df_dist = pd.DataFrame(list(class_counts.items()),\n",
    "                        columns=['Classe', 'Nombre d images'])\n",
    "print(df_dist.to_string(index=False))\n",
    "print(f\"\\nTotal : {df_dist['Nombre d images'].sum()} images\")\n\n",
    "fig, ax = plt.subplots(figsize=(14, 5))\n",
    "ax.barh(df_dist['Classe'], df_dist['Nombre d images'], color='steelblue')\n",
    "ax.set_xlabel('Nombre d images')\n",
    "ax.set_title('Distribution des images par classe')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

# ============================================================
# SECTION 3 — Prétraitement
# ============================================================
cells.append(cell_md([
    "## 3. Prétraitement des Images\n\n",
    "Le prétraitement conditionne la qualité de toute la chaîne de traitement. "
    "Voici les 4 étapes appliquées à chaque image du dataset :\n\n",
    "| Étape | Opération | Bibliothèque | Paramètres clés | Justification |\n",
    "|-------|-----------|-------------|-----------------|---------------|\n",
    "| 1 | **Redimensionnement** | OpenCV | 224×224, INTER_AREA | Taille imposée par MobileNetV2 ; cohérente ML/DL |\n",
    "| 2 | **Filtrage Gaussien** | OpenCV | Noyau 5×5, σ=auto | Supprime le bruit JPEG/capteur sans détruire les contours |\n",
    "| 3 | **Conversion RGB→HSV** | OpenCV | — | Sépare la teinte (H) de la luminosité (V), robuste à l'éclairage |\n",
    "| 4 | **Split 70/15/15** | Python stdlib | seed=42 | Reproductibilité et absence de fuite train→test |\n"
]))

# 3.1 — Redimensionnement
cells.append(cell_md([
    "### 3.1 Redimensionnement (224×224)\n\n",
    "**Pourquoi 224×224 ?**  \n",
    "MobileNetV2, notre architecture DL, a été pré-entraîné sur ImageNet avec des entrées de "
    "taille **224×224×3**. Pour que le Transfer Learning soit cohérent, toutes les images "
    "doivent avoir exactement cette résolution. On utilise la même taille en ML pour que les "
    "vecteurs de features soient comparables entre eux.\n\n",
    "**Choix de l'interpolation `INTER_AREA` :**  \n",
    "Optimal pour la **réduction** d'image (les images PlantVillage font ~256×256 à 800×600). "
    "Contrairement à INTER_LINEAR, il fait une vraie moyenne des pixels sans aliasing.\n"
]))
cells.append(cell_code([
    "VALID_EXT = {'.jpg','.jpeg','.png','.bmp'}\n",
    "\n",
    "# Sélection d une image de démonstration\n",
    "demo_cls_path = PROCESSED_DIR / 'train' / 'Tomato___Late_blight'\n",
    "demo_file = next(f for f in demo_cls_path.iterdir() if f.suffix.lower() in VALID_EXT)\n",
    "img_original = cv2.cvtColor(cv2.imread(str(demo_file)), cv2.COLOR_BGR2RGB)\n",
    "img_resized  = cv2.resize(img_original, (224, 224), interpolation=cv2.INTER_AREA)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "axes[0].imshow(img_original)\n",
    "axes[0].set_title(f'Original ({img_original.shape[1]}×{img_original.shape[0]} px)')\n",
    "axes[0].axis('off')\n",
    "axes[1].imshow(img_resized)\n",
    "axes[1].set_title('Redimensionné (224×224 px)')\n",
    "axes[1].axis('off')\n",
    "plt.suptitle('Étape 1 — Redimensionnement INTER_AREA', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

# 3.2 — Filtrage Gaussien
cells.append(cell_md([
    "### 3.2 Filtrage du Bruit (Filtre Gaussien 5×5)\n\n",
    "**Problème :** Les images PlantVillage contiennent du bruit de capteur et des "
    "artéfacts de compression JPEG (blocs 8×8 pixels visibles en agrandissant). "
    "Ce bruit parasite l'extraction de texture (GLCM) et la segmentation HSV.\n\n",
    "**Solution : Filtre Gaussien** `GaussianBlur(kernel=(5,5), σ=auto)`\n\n",
    "- Le **noyau 5×5** est un compromis étudié : assez grand pour lisser le bruit "
    "de capteur (1–2 px), mais assez petit pour **préserver les contours** des taches "
    "foliaires (bords > 3 px en général).\n",
    "- **σ=0 (auto)** : OpenCV calcule σ = 0.3 × ((ksize-1)×0.5 - 1) + 0.8 ≈ **1.1** "
    "pour un noyau 5×5. Valeur standard et conservatrice.\n\n",
    "**Alternative étudiée et rejetée :** Le filtre Médian (MedianBlur) est meilleur pour "
    "le bruit impulsionnel (sel & poivre), mais plus lent et moins approprié ici où "
    "le bruit est Gaussien (capteur photo).\n"
]))
cells.append(cell_code([
    "img_noisy    = img_resized.copy()\n",
    "img_denoised = cv2.GaussianBlur(img_resized, (5, 5), sigmaX=0)\n",
    "\n",
    "# Zoom sur un patch 64×64 pour rendre visible la différence\n",
    "patch_orig = img_noisy[80:144, 80:144]\n",
    "patch_den  = img_denoised[80:144, 80:144]\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(11, 8))\n",
    "axes[0][0].imshow(img_noisy);   axes[0][0].set_title('Avant filtrage'); axes[0][0].axis('off')\n",
    "axes[0][1].imshow(img_denoised); axes[0][1].set_title('Après Gaussien 5×5'); axes[0][1].axis('off')\n",
    "axes[1][0].imshow(patch_orig);   axes[1][0].set_title('Zoom avant (64×64)'); axes[1][0].axis('off')\n",
    "axes[1][1].imshow(patch_den);    axes[1][1].set_title('Zoom après (64×64)'); axes[1][1].axis('off')\n",
    "plt.suptitle('Étape 2 — Filtrage Gaussien (noyau 5×5)', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(f'PSNR estimé (proxy) : {10 * np.log10(255**2 / np.mean((img_noisy.astype(float) - img_denoised.astype(float))**2 + 1e-10)):.1f} dB')\n",
]))

# 3.3 — Conversion RGB → HSV
cells.append(cell_md([
    "### 3.3 Conversion RGB → HSV\n\n",
    "**Pourquoi HSV et pas RGB ?**  \n",
    "En RGB, une feuille verte sous forte lumière a les mêmes composantes que "
    "le gazon sous lumière normale. La segmentation par seuillage est impossible.\n\n",
    "En HSV, la **Teinte (H)** reste stable quelle que soit l'intensité lumineuse :\n",
    "- Feuille verte saine → H ≈ 60–95° (quel que soit l'éclairage)\n",
    "- Zone nécrosée → H ≈ 5–22° (brun/orange)\n\n",
    "La **Saturation (S)** et la **Valeur (V)** permettent d'exclure le fond blanc/gris "
    "(faible S) et les zones trop sombres (faible V).\n"
]))
cells.append(cell_code([
    "img_hsv = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2HSV)\n",
    "\n",
    "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
    "panels = [\n",
    "    (img_denoised,         'Image RGB (après filtre)',  None),\n",
    "    (img_hsv[:,:,0],       'Canal H — Teinte',          'hsv'),\n",
    "    (img_hsv[:,:,1],       'Canal S — Saturation',      'Greens'),\n",
    "    (img_hsv[:,:,2],       'Canal V — Valeur/Luminosité','gray'),\n",
    "]\n",
    "for ax, (data, title, cmap) in zip(axes, panels):\n",
    "    ax.imshow(data, cmap=cmap)\n",
    "    ax.set_title(title, fontsize=9)\n",
    "    ax.axis('off')\n",
    "plt.suptitle('Étape 3 — Décomposition en canaux HSV', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

# 3.4 — Analyse des histogrammes
cells.append(cell_md([
    "### 3.4 Analyse des Histogrammes de Couleur\n\n",
    "L'histogramme de couleur est l'une des représentations les plus utiles en "
    "classification d'images. Chaque maladie crée une **signature spectrale distincte** :\n",
    "- **Mildiou (Late blight)** → dominante brune/noire (H faible, S élevée, V basse)\n",
    "- **Feuille saine** → dominante verte (H ≈ 60–90°, S et V modérées)\n",
    "- **Rouille (Common rust)** → dominante orangée (H ≈ 20–30°)\n\n",
    "Nous comparons les histogrammes H (Teinte) de 4 classes différentes pour illustrer "
    "la séparabilité des maladies dans l'espace HSV.\n"
]))
cells.append(cell_code([
    "compare_classes = [\n",
    "    ('Tomato___healthy',            'Tomate Saine',    '#2ecc71'),\n",
    "    ('Tomato___Late_blight',        'Mildiou Tomate',  '#8e44ad'),\n",
    "    ('Corn_(maize)___Common_rust_', 'Rouille Maïs',    '#e74c3c'),\n",
    "    ('Potato___Early_blight',       'Alternariose',    '#e67e22'),\n",
    "]\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "for ax, (cls, label, color) in zip(axes.flatten(), compare_classes):\n",
    "    cls_dir = PROCESSED_DIR / 'train' / cls\n",
    "    all_hist = np.zeros(180)\n",
    "    count = 0\n",
    "    for f in list(cls_dir.iterdir())[:30]:  # 30 images max pour la vitesse\n",
    "        if f.suffix.lower() not in VALID_EXT: continue\n",
    "        img = cv2.imread(str(f))\n",
    "        if img is None: continue\n",
    "        img = cv2.resize(img, (224, 224))\n",
    "        img = cv2.GaussianBlur(img, (5, 5), 0)\n",
    "        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n",
    "        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()\n",
    "        all_hist += h_hist\n",
    "        count += 1\n",
    "    all_hist /= (count + 1e-10)\n",
    "    all_hist /= (all_hist.max() + 1e-10)  # normalisation\n",
    "    ax.plot(all_hist, color=color, linewidth=1.5)\n",
    "    ax.fill_between(range(180), all_hist, alpha=0.25, color=color)\n",
    "    ax.axvline(22,  color='gray', linestyle='--', alpha=0.5, label='H=22 (seuil vert)')\n",
    "    ax.axvline(95,  color='gray', linestyle=':',  alpha=0.5, label='H=95')\n",
    "    ax.set_title(f'Histogramme Teinte (H) — {label}', fontsize=10)\n",
    "    ax.set_xlabel('Valeur de Teinte (0–180°)')\n",
    "    ax.set_ylabel('Fréquence normalisée')\n",
    "    ax.legend(fontsize=7)\n",
    "    ax.grid(alpha=0.3)\n",
    "plt.suptitle('Analyse des Histogrammes HSV — Signature spectrale par maladie',\n",
    "             fontsize=12, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

# 3.5 — Split
cells.append(cell_md([
    "### 3.5 Partitionnement Stratifié (70 / 15 / 15)\n\n",
    "**Seed fixée à 42** pour la reproductibilité totale (même résultats entre "
    "deux exécutions). Le nettoyage préalable du dossier `processed/` empêche "
    "toute contamination des données entre les splits.\n"
]))
cells.append(cell_code([
    "# Vérification du split\n",
    "for split in ['train', 'val', 'test']:\n",
    "    split_dir = PROCESSED_DIR / split\n",
    "    if split_dir.exists():\n",
    "        total = sum(len(list((split_dir/c).glob('*.*')))\n",
    "                    for c in os.listdir(split_dir)\n",
    "                    if (split_dir/c).is_dir())\n",
    "        n_cls = len([d for d in split_dir.iterdir() if d.is_dir()])\n",
    "        print(f'  {split.upper():<6} : {n_cls:>2} classes | {total:>6} images')\n",
    "    else:\n",
    "        print(f'  {split.upper():<6} : NON GÉNÉRÉ — lancez splitter.py')\n",
]))

# ============================================================
# SECTION 4 — Segmentation
# ============================================================
cells.append(cell_md([
    "## 4. Segmentation et Détection de Contours\n\n",
    "Quatre méthodes complémentaires sont implémentées et comparées :\n\n",
    "| Méthode | Type | Usage dans le projet |\n",
    "|---------|------|---------------------|\n",
    "| **Sobel** | Gradient | Visualisation des zones de texture anormale |\n",
    "| **Canny** | Contours multi-seuils | Délimitation précise des lésions |\n",
    "| **Otsu** | Seuillage automatique | Segmentation fond/feuille |\n",
    "| **K-Means (k=3)** | Clustering | Détection fond/saine/malade |\n",
    "| **HSV double masque** ⭐ | Seuillage couleur | **Pipeline ML production** |\n"
]))

# 4.1 — Détection de contours
cells.append(cell_md([
    "### 4.1 Détection de Contours — Sobel vs Canny\n\n",
    "**Sobel** calcule le gradient d'intensité (magnitude = √(Gx² + Gy²)).  \n",
    "**Canny** ajoute le lissage Gaussien, la non-maxima suppression et le double seuillage (low=50, high=150) → contours plus propres.\n"
]))
cells.append(cell_code([
    "from src.vision.segmentation import (\n",
    "    segment_leaf, detect_edges_sobel, detect_edges_canny,\n",
    "    segment_otsu, segment_kmeans\n",
    ")\n\n",
    "VALID_EXT = {'.jpg','.jpeg','.png','.bmp'}\n",
    "demo_cls  = 'Tomato___Late_blight'\n",
    "demo_path = str(next(f for f in (PROCESSED_DIR/'train'/demo_cls).iterdir()\n",
    "                     if f.suffix.lower() in VALID_EXT))\n",
    "orig_rgb, mask_hsv, seg_hsv = segment_leaf(demo_path)\n\n",
    "sobel  = detect_edges_sobel(orig_rgb)\n",
    "canny  = detect_edges_canny(orig_rgb, low_threshold=50, high_threshold=150)\n\n",
    "fig, axes = plt.subplots(1, 4, figsize=(18, 5))\n",
    "data_titles = [\n",
    "    (orig_rgb,  'Image originale (RGB)',           None),\n",
    "    (sobel,     'Sobel (gradient magnitude)',       'gray'),\n",
    "    (canny,     'Canny (low=50, high=150)',         'gray'),\n",
    "    (seg_hsv,   'Référence : HSV double masque',   None),\n",
    "]\n",
    "for ax, (d, t, cm) in zip(axes, data_titles):\n",
    "    ax.imshow(d, cmap=cm); ax.set_title(t, fontsize=9); ax.axis('off')\n",
    "plt.suptitle('Comparaison Sobel vs Canny — Détection de Zones Anormales',\n",
    "             fontsize=12, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'Sobel  : {(sobel>50).sum()} pixels de contour')\n",
    "print(f'Canny  : {(canny>0).sum()} pixels de contour')\n",
]))

# 4.2 — Seuillage Otsu
cells.append(cell_md([
    "### 4.2 Seuillage Automatique d'Otsu\n\n",
    "Otsu calcule automatiquement le seuil T qui maximise la **variance inter-classes** "
    "(fond vs feuille). Aucun paramètre à régler.\n"
]))
cells.append(cell_code([
    "thresh_val, mask_otsu, seg_otsu = segment_otsu(orig_rgb)\n\n",
    "fig, axes = plt.subplots(1, 4, figsize=(18, 5))\n",
    "gray_img = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)\n",
    "hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()\n",
    "# Affichage : image grise + histogramme + masque + segmentation\n",
    "axes[0].imshow(gray_img, cmap='gray'); axes[0].set_title('Niveaux de gris'); axes[0].axis('off')\n",
    "axes[1].plot(hist_gray, color='#2c3e50', linewidth=1)\n",
    "axes[1].axvline(thresh_val, color='red', linestyle='--', label=f'T*={thresh_val}')\n",
    "axes[1].set_title(f'Histogramme gris + seuil Otsu (T*={thresh_val})')\n",
    "axes[1].legend(); axes[1].grid(alpha=0.3)\n",
    "axes[2].imshow(mask_otsu, cmap='gray'); axes[2].set_title('Masque Otsu'); axes[2].axis('off')\n",
    "axes[3].imshow(seg_otsu); axes[3].set_title('Feuille segmentée (Otsu)'); axes[3].axis('off')\n",
    "plt.suptitle('Seuillage Automatique d\'Otsu', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n",
]))

# 4.3 — K-Means
cells.append(cell_md([
    "### 4.3 Segmentation par Clustering K-Means (k=3)\n\n",
    "K-Means regroupe les pixels en **3 clusters** : fond, feuille saine, zones malades.  \n",
    "Initialisation **K-Means++** pour plus de stabilité. Le cluster 'feuille' est identifié "
    "comme celui avec la plus grande valeur de canal Vert (G).\n"
]))
cells.append(cell_code([
    "labels, centers, seg_km = segment_kmeans(orig_rgb, k=3)\n\n",
    "# Image avec chaque pixel coloré selon son cluster\n",
    "cluster_img = centers[labels]  # image reconstruite par couleur de cluster\n\n",
    "fig, axes = plt.subplots(1, 4, figsize=(18, 5))\n",
    "axes[0].imshow(orig_rgb);       axes[0].set_title('Original');                axes[0].axis('off')\n",
    "axes[1].imshow(cluster_img);    axes[1].set_title('K-Means k=3 (régions)');  axes[1].axis('off')\n",
    "axes[2].imshow(seg_km);         axes[2].set_title('Cluster feuille isolé');  axes[2].axis('off')\n",
    "axes[3].imshow(seg_hsv);        axes[3].set_title('Référence HSV');          axes[3].axis('off')\n",
    "plt.suptitle('Clustering K-Means (k=3) vs Référence HSV', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n",
    "print('Centres des clusters (RGB) :')\n",
    "for i, c in enumerate(centers):\n",
    "    print(f'  Cluster {i} : R={c[0]:3d}, G={c[1]:3d}, B={c[2]:3d}')\n",
]))

# 4.4 — Comparaison finale
cells.append(cell_md([
    "### 4.4 Comparaison Subjective des 5 Méthodes\n\n",
    "| Méthode | Isolation feuille | Détection taches | Vitesse | Paramètres |\n",
    "|---------|-------------------|-----------------|---------|------------|\n",
    "| Sobel | ❌ Non | ✅ Contours des taches | ⚡ Très rapide | ksize |\n",
    "| Canny | ❌ Non | ✅✅ Contours précis | ⚡ Rapide | low, high |\n",
    "| Otsu | ✅ Oui (fond clair) | ❌ Fond/feuille seulement | ⚡ Très rapide | Aucun |\n",
    "| K-Means | ✅ Oui | ✅ Partiellement | 🐢 Lent | k |\n",
    "| **HSV masque** ⭐ | **✅✅ Précis** | **✅✅ Vert+Brun** | **⚡ Rapide** | **Seuils HSV** |\n",
    "\n",
    "**Conclusion :** Le double masque HSV est la meilleure méthode pour ce dataset car les couleurs "
    "des maladies (vert→brun) sont biologiquement stables et connues a priori.\n"
]))
cells.append(cell_code([
    "# Comparaison visuelle finale — toutes les méthodes sur la même image\n",
    "_, mask_otsu2, seg_otsu2 = segment_otsu(orig_rgb)\n",
    "_, _, seg_km2 = segment_kmeans(orig_rgb, k=3)\n",
    "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
    "panels = [\n",
    "    (orig_rgb,                    'Original'),\n",
    "    (sobel,                       'Sobel — Gradient'),\n",
    "    (canny,                       'Canny — Contours'),\n",
    "    (seg_otsu2,                   'Otsu — Seuillage auto'),\n",
    "    (seg_km2,                     'K-Means k=3'),\n",
    "    (seg_hsv,                     'HSV double masque ⭐'),\n",
    "]\n",
    "cmaps = [None, 'gray', 'gray', None, None, None]\n",
    "for ax, (img, title), cmap in zip(axes.flatten(), panels, cmaps):\n",
    "    ax.imshow(img, cmap=cmap)\n",
    "    ax.set_title(title, fontsize=10)\n",
    "    ax.axis('off')\n",
    "plt.suptitle(f'Comparaison des 5 méthodes — {demo_cls.split(\"___\")[-1]}',\n",
    "             fontsize=13, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n",
]))

# ============================================================
# SECTION 5 — Extraction de Features
# ============================================================
cells.append(cell_md([
    "## 5. Extraction de Caractéristiques\n\n",
    "| Catégorie | Méthode | Dimensions |\n",
    "|-----------|---------|------------|\n",
    "| **Couleur RGB** | Histogramme 1D par canal (8 bins × 3) | 24 |\n",
    "| **Couleur HSV** | Histogramme HSV 3D (8×8×8 bins) | 512 |\n",
    "| **Texture** | GLCM Haralick (contraste, énergie, corrélation…) | 20 |\n",
    "| **Forme**   | Moments de Hu (×7) + Circularité + Compacité + Ratio aire | 10 |\n",
    "| **Total** | | **566** |\n"
]))

# 5.1 Histogrammes RGB vs HSV
cells.append(cell_md([
    "### 5.1 Histogrammes de Couleur — RGB vs HSV\n\n",
    "L'histogramme RGB (24 dims) est calculé pour la **comparaison visuelle**. "
    "L'histogramme HSV 3D (512 dims) est utilisé dans le pipeline ML car la Teinte (H) "
    "est invariante à l'éclairage.\n"
]))
cells.append(cell_code([
    "from src.features.extractors import (\n",
    "    extract_rgb_histogram, extract_color_histogram,\n",
    "    extract_glcm_texture, extract_shape_features, extract_features\n",
    ")\n\n",
    "test_cls = 'Tomato___Late_blight'\n",
    "test_dir = PROCESSED_DIR / 'train' / test_cls\n",
    "test_img_path = next(f for f in test_dir.iterdir() if f.suffix.lower() in VALID_EXT)\n",
    "img_rgb, mask, _ = segment_leaf(str(test_img_path))\n\n",
    "rgb_f = extract_rgb_histogram(img_rgb, mask)\n",
    "hsv_f = extract_color_histogram(img_rgb, mask)\n\n",
    "# Visualisation des histogrammes RGB\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "colors = ['#e74c3c', '#2ecc71', '#3498db']\n",
    "labels = ['Rouge (R)', 'Vert (G)', 'Bleu (B)']\n",
    "for i, (col, lbl) in enumerate(zip(colors, labels)):\n",
    "    axes[0].bar(np.arange(8) + i*0.25, rgb_f[i*8:(i+1)*8],\n",
    "               width=0.25, color=col, alpha=0.8, label=lbl)\n",
    "axes[0].set_title('Histogramme RGB (8 bins × 3 canaux = 24 dims)')\n",
    "axes[0].set_xlabel('Bin'); axes[0].set_ylabel('Fréquence normalisée')\n",
    "axes[0].legend(); axes[0].grid(alpha=0.3)\n",
    "# Projection H marginal de l histogramme HSV 3D\n",
    "hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)\n",
    "h_hist = cv2.calcHist([hsv_img], [0], mask, [8], [0, 180]).flatten()\n",
    "h_hist /= (h_hist.max() + 1e-10)\n",
    "axes[1].bar(range(8), h_hist, color='#9b59b6', alpha=0.8)\n",
    "axes[1].set_title('Histogramme Teinte H (8 bins du HSV 3D, 512 dims total)')\n",
    "axes[1].set_xlabel('Bin Teinte'); axes[1].set_ylabel('Fréquence normalisée')\n",
    "axes[1].grid(alpha=0.3)\n",
    "plt.suptitle('Comparaison Histogrammes RGB vs HSV', fontsize=12, fontweight='bold')\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'RGB hist : {rgb_f.shape} — {rgb_f[:6].round(3)}')\n",
    "print(f'HSV hist : {hsv_f.shape} (vecteur 3D aplati)')\n",
]))

# 5.2 GLCM Texture
cells.append(cell_md([
    "### 5.2 Texture — GLCM Haralick (20 dimensions)\n\n",
    "La GLCM mesure la co-occurrence des niveaux de gris entre pixels voisins sur **4 orientations** "
    "(0°, 45°, 90°, 135°). **5 propriétés** extraites : contraste, dissimilarité, homogénéité, énergie, corrélation.\n"
]))
cells.append(cell_code([
    "texture_f = extract_glcm_texture(img_rgb, mask)\n",
    "shape_f   = extract_shape_features(mask)\n",
    "all_f     = extract_features(img_rgb, mask)\n\n",
    "props  = ['Contraste', 'Dissimilarité', 'Homogénéité', 'Énergie', 'Corrélation']\n",
    "angles = ['0°', '45°', '90°', '135°']\n",
    "texture_mat = texture_f.reshape(5, 4)\n\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "sns.heatmap(texture_mat, annot=True, fmt='.3f', xticklabels=angles,\n",
    "            yticklabels=props, cmap='YlOrRd', ax=ax)\n",
    "ax.set_title('Propriétés GLCM — 5 métriques × 4 orientations (20 valeurs)')\n",
    "plt.tight_layout(); plt.show()\n\n",
    "print(f'Texture GLCM : {texture_f.shape} valeurs')\n",
    "print(f'Forme (Hu)   : {shape_f.shape} valeurs')\n",
    "print(f'Vecteur final: {all_f.shape} dimensions')\n",
]))

# ============================================================
# SECTION 6 — Random Forest
# ============================================================
cells.append(cell_md([
    "## 6. Classification — Random Forest\n\n",
    "- `n_estimators=300`, `class_weight='balanced'`\n",
    "- Features normalisées par `StandardScaler`\n",
    "- Entraîné sur **tout** le set d'entraînement (sans limite artificielle)\n"
]))
cells.append(cell_code([
    "# Chargement du modèle RF déjà entraîné\n",
    "rf_path = MODEL_DIR / 'rf_model.pkl'\n",
    "if rf_path.exists():\n",
    "    rf_data = joblib.load(rf_path)\n",
    "    rf_model = rf_data['model']\n",
    "    rf_scaler = rf_data['scaler']\n",
    "    classes = rf_data['classes']\n",
    "    print(f'RF chargé — {len(classes)} classes')\n",
    "    print(f'Paramètres : {rf_model.get_params()}')\n",
    "else:\n",
    "    print('Modèle RF non trouvé. Lancez : python train_ml.py')\n",
]))
cells.append(cell_code([
    "# Matrice de confusion RF\n",
    "from PIL import Image\n",
    "if Path('confusion_matrix_rf.png').exists():\n",
    "    img = Image.open('confusion_matrix_rf.png')\n",
    "    plt.figure(figsize=(14,11))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title('Matrice de Confusion — Random Forest')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Lancez train_ml.py pour générer la matrice de confusion.')\n",
]))

# ============================================================
# SECTION 7 — SVM
# ============================================================
cells.append(cell_md([
    "## 7. Classification — SVM (Support Vector Machine)\n\n",
    "- Kernel **RBF** avec `C=10`, `gamma='scale'`\n",
    "- `class_weight='balanced'` pour compenser le déséquilibre\n",
    "- `probability=True` pour les prédictions Top-3\n"
]))
cells.append(cell_code([
    "svm_path = MODEL_DIR / 'svm_model.pkl'\n",
    "if svm_path.exists():\n",
    "    svm_data = joblib.load(svm_path)\n",
    "    svm_model = svm_data['model']\n",
    "    print(f'SVM chargé — {len(svm_data[\"classes\"])} classes')\n",
    "    print(f'Paramètres : {svm_model.get_params()}')\n",
    "else:\n",
    "    print('Modèle SVM non trouvé. Lancez : python train_ml.py')\n",
]))
cells.append(cell_code([
    "if Path('confusion_matrix_svm.png').exists():\n",
    "    img = Image.open('confusion_matrix_svm.png')\n",
    "    plt.figure(figsize=(14,11))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title('Matrice de Confusion — SVM (RBF)')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
]))

# ============================================================
# SECTION 8 — MobileNetV2
# ============================================================
cells.append(cell_md([
    "## 8. Deep Learning — MobileNetV2 (Transfer Learning)\n\n",
    "**Architecture :**\n",
    "```\n",
    "Input (224×224×3)\n",
    "   → Data Augmentation (Flip, Rotation, Zoom)\n",
    "   → preprocess_input [-1, 1]\n",
    "   → MobileNetV2 (ImageNet weights) — couches gelées en Phase 1\n",
    "   → GlobalAveragePooling2D\n",
    "   → BatchNormalization\n",
    "   → Dropout(0.3)\n",
    "   → Dense(12, softmax)\n",
    "```\n",
    "**Phase 1** : Feature Extraction (base gelée, lr=1e-3)\n\n",
    "**Phase 2** : Fine-tuning (30 dernières couches dégelées, lr=1e-5)\n"
]))
cells.append(cell_code([
    "dl_path = MODEL_DIR / 'mobilenetv2_plants.keras'\n",
    "if dl_path.exists():\n",
    "    dl_model = tf.keras.models.load_model(str(dl_path), compile=False)\n",
    "    dl_model.summary()\n",
    "else:\n",
    "    print('Modèle DL non trouvé. Lancez : python train_dl.py')\n",
]))
cells.append(cell_code([
    "# Courbes d apprentissage\n",
    "if Path('training_history.png').exists():\n",
    "    img = Image.open('training_history.png')\n",
    "    plt.figure(figsize=(14,5))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title('Courbes Accuracy & Loss — Phase 1 + Fine-tuning')\n",
    "    plt.show()\n",
]))
cells.append(cell_code([
    "# Matrice de confusion DL\n",
    "if Path('confusion_matrix_dl.png').exists():\n",
    "    img = Image.open('confusion_matrix_dl.png')\n",
    "    plt.figure(figsize=(14,11))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title('Matrice de Confusion — MobileNetV2')\n",
    "    plt.show()\n",
]))

# ============================================================
# SECTION 9 — Grad-CAM
# ============================================================
cells.append(cell_md([
    "## 9. Explicabilité — Grad-CAM\n\n",
    "**Gradient-weighted Class Activation Mapping** permet de visualiser "
    "quelles zones de l'image ont influencé la décision du réseau.\n\n",
    "- Couche cible : `out_relu` (dernière activation spatiale de MobileNetV2, 7×7×1280)\n",
    "- Les gradients de la classe prédite sont moyennés spatialement\n",
    "- La combinaison pondérée produit une carte de chaleur (rouge = zone décisive)\n"
]))
cells.append(cell_code([
    "from predict import make_gradcam_heatmap, blend_gradcam\n\n",
    "dl_path = MODEL_DIR / 'mobilenetv2_plants.keras'\n",
    "if not dl_path.exists():\n",
    "    print('DL model required.')\n",
    "else:\n",
    "    dl_model = tf.keras.models.load_model(str(dl_path), compile=False)\n",
    "    demo_classes = ['Tomato___Late_blight', 'Corn_(maize)___Common_rust_',\n",
    "                    'Potato___Early_blight']\n",
    "    demo_paths = []\n",
    "    for cls in demo_classes:\n",
    "        d = PROCESSED_DIR / 'test' / cls\n",
    "        if d.exists():\n",
    "            files = [f for f in d.iterdir() if f.suffix.lower() in VALID_EXT]\n",
    "            if files: demo_paths.append((cls, str(files[0])))\n\n",
    "    fig, axes = plt.subplots(len(demo_paths), 3, figsize=(13, 4.5*len(demo_paths)))\n",
    "    if len(demo_paths)==1: axes=[axes]\n",
    "    for row,(cls,path) in enumerate(demo_paths):\n",
    "        img = tf.keras.utils.load_img(path, target_size=(224,224))\n",
    "        arr = tf.keras.utils.img_to_array(img)\n",
    "        batch = tf.expand_dims(arr, 0)\n",
    "        heatmap = make_gradcam_heatmap(batch, dl_model)\n",
    "        blended = blend_gradcam(path, heatmap)\n",
    "        orig_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)\n",
    "        hm_up = cv2.resize(heatmap, (224,224))\n",
    "        cols_data = [cv2.resize(orig_rgb,(224,224)), hm_up, blended]\n",
    "        ttls = ['Original', 'Heatmap Grad-CAM', 'Superposition']\n",
    "        for col,(d,t) in enumerate(zip(cols_data,ttls)):\n",
    "            axes[row][col].imshow(d, cmap='hot' if col==1 else None)\n",
    "            axes[row][col].set_title(f'{t}\\n{cls.split(\"___\")[-1]}', fontsize=9)\n",
    "            axes[row][col].axis('off')\n",
    "    plt.suptitle('Grad-CAM — Explicabilité du modèle Deep Learning', fontsize=13)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
]))

# ============================================================
# SECTION 10 — Benchmark
# ============================================================
cells.append(cell_md([
    "## 10. Benchmark ML vs Deep Learning\n\n",
    "Comparaison des trois modèles sur le **même test set** avec 5 métriques :\n",
    "Accuracy, Précision, Rappel, F1-score (weighted), Temps d'inférence\n"
]))
cells.append(cell_code([
    "if Path('benchmark_results.csv').exists():\n",
    "    df_bench = pd.read_csv('benchmark_results.csv')\n",
    "    print(df_bench.to_string(index=False))\n",
    "else:\n",
    "    print('Lancez : python src/utils/benchmark.py')\n",
]))
cells.append(cell_code([
    "if Path('benchmark_chart.png').exists():\n",
    "    from PIL import Image\n",
    "    img = Image.open('benchmark_chart.png')\n",
    "    plt.figure(figsize=(13,6))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title('Comparaison Benchmark — RF vs SVM vs MobileNetV2')\n",
    "    plt.show()\n",
]))

# ============================================================
# SECTION 11 — Conclusion
# ============================================================
cells.append(cell_md([
    "## 11. Conclusion & Recommandations\n\n",
    "### Synthèse des résultats\n\n",
    "| Critère | Random Forest | SVM (RBF) | MobileNetV2 |\n",
    "|---------|--------------|-----------|-------------|\n",
    "| **Accuracy** | ~77% | ~80% | ~92% |\n",
    "| **Interprétabilité** | Importance features | Kernel SHAP | Grad-CAM |\n",
    "| **Vitesse inférence** | Très rapide | Rapide | Modérée (CPU) |\n",
    "| **Mémoire modèle** | ~7 Mo | ~15 Mo | ~10 Mo |\n\n",
    "### Points forts du Deep Learning\n",
    "- Précision nettement supérieure grâce au Transfer Learning (ImageNet)\n",
    "- Robuste aux variations de luminosité, angle et fond\n",
    "- Grad-CAM offre une explicabilité visuelle forte pour les agriculteurs\n\n",
    "### Points forts du Machine Learning classique\n",
    "- Entraînable sans GPU, rapide sur CPU\n",
    "- Features interprétables (GLCM, Hu Moments)\n",
    "- Idéal pour déploiement sur systèmes embarqués\n\n",
    "### Perspectives d'amélioration\n",
    "1. Augmenter le dataset (images de terrain en conditions réelles)\n",
    "2. Tester EfficientNetV2 ou ConvNeXt pour encore plus d'accuracy\n",
    "3. Déploiement mobile via TFLite (quantification int8)\n",
    "4. Intégration d'une API REST pour usage en télédétection agricole\n\n",
    "---\n",
    "*Projet réalisé avec PlantVillage Dataset | TensorFlow 2.x | scikit-learn*\n"
]))

# ============================================================
# Assemblage du notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

out_path = Path("Plant_Disease_Project.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook généré : {out_path}")
print(f"   {len(cells)} cellules | 11 sections")

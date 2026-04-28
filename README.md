# 🌿 Détection et Classification Automatique de Maladies des Plantes

Ce projet académique complet implémente un système intelligent capable de diagnostiquer les maladies des plantes à partir d'images foliaires. Il couvre un pipeline de bout en bout : du nettoyage du jeu de données (Dataset) jusqu'au déploiement d'une application web interactive, en passant par l'extraction de caractéristiques complexes et l'entraînement de modèles d'Intelligence Artificielle de pointe.

![Aperçu de l'interface](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)

---

## 🔬 Objectifs du Projet

- **Architecture Robuste :** Implémenter une structure de code professionnelle (Modulaire, orientée objet, PEP 8).
- **Vision par Ordinateur :** Isoler la feuille de son arrière-plan (Segmentation par double masque HSV).
- **Machine Learning (ML) :** Extraire des descripteurs mathématiques (Couleur HSV, Texture GLCM, Forme/Moments de Hu) et classifier avec Random Forest et SVM.
- **Deep Learning (DL) :** Faire du Transfer Learning avec l'architecture `MobileNetV2`.
- **Explicabilité (XAI) :** Implémenter l'algorithme `Grad-CAM` pour comprendre quelles zones de l'image le réseau a analysées.

---

## 🗂️ Les 12 Classes Analysées

Le projet couvre 12 classes (maladies et plants sains) à travers 3 espèces majeures :

1. **Maïs (Corn) 🌽 :** Cercospora leaf spot, Common rust, Northern Leaf Blight, Saine.
2. **Pomme de terre (Potato) 🥔 :** Early blight (Alternariose), Late blight (Mildiou), Saine.
3. **Tomate (Tomato) 🍅 :** Early blight, Late blight, Leaf Mold, Septoria leaf spot, Saine.

*Taille totale du dataset analysé : +13 500 images.*

---

## 🏆 Résultats du Benchmark

Tous les modèles ont été comparés rigoureusement sur le même set de test (15% du dataset global) :

| Modèle | Accuracy | Précision | F1-Score | Extraction de Features |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 89.04 % | 89.10 % | 88.79 % | Histogramme HSV + GLCM + Hu Moments (542 dims) |
| **SVM (Noyau RBF)** | 87.33 % | 87.57 % | 87.37 % | Histogramme HSV + GLCM + Hu Moments (542 dims) |
| **MobileNetV2 (DL)** | **93.59 %** | **93.85 %** | **93.57 %** | Réseau convolutif profond pré-entraîné (ImageNet) |

---

## 🚀 Installation & Lancement

1. **Cloner le répertoire :**
   ```bash
   git clone https://github.com/bahaell/Maladies_des_plantes.git
   cd Maladies_des_plantes
   ```

2. **Créer un environnement virtuel (recommandé) :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'Application Web (Streamlit) :**
   ```bash
   streamlit run app.py
   ```
   > 🌐 L'application s'ouvrira automatiquement sur http://localhost:8501. Vous pourrez uploader vos propres images, générer des Grad-CAM et obtenir des conseils agronomiques.

---

## 📂 Structure du Répertoire

```text
Maladies_des_plantes/
├── app.py                       # Interface utilisateur principale (Streamlit)
├── predict.py                   # Fonctions d'inférence (Inférences et Grad-CAM)
├── train_dl.py                  # Script d'entraînement pour le Deep Learning
├── train_ml.py                  # Script d'entraînement pour RF et SVM
├── Plant_Disease_Project.ipynb  # Jupyter Notebook pour soutenance académique
├── src/
│   ├── data/                    # Chargement, tri et split des images brutes
│   ├── features/                # Extraction de texture (GLCM), couleur et forme
│   ├── vision/                  # Algorithmes classiques (Segmentation HSV)
│   └── utils/                   # Benchmarks et base de connaissances agricole
├── models/                      # Emplacement des modèles sauvegardés (ignorés sur git)
└── requirements.txt             # Liste des paquets Python
```

---

*Projet réalisé dans un cadre académique.*

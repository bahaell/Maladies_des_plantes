# 📋 Rapport Technique — Étape 5 : Évaluation de la Classification

> **Projet :** Détection et Classification Automatique de Maladies des Plantes  
> **Section :** Métriques d'évaluation ML/DL  
> **Fichiers :** `train_ml.py`, `eval_dl.py`, `src/utils/benchmark.py`

---

## Pourquoi Évaluer sur un Test Set Indépendant ?

L'évaluation sur les données **d'entraînement** donne toujours un résultat optimiste (le modèle a "mémorisé" ces données). Pour mesurer la capacité de **généralisation** — c'est-à-dire les performances sur des images jamais vues — il faut un **test set indépendant** qui n'a jamais participé à l'entraînement ni au choix des hyperparamètres.

Notre test set : **2 044 images** (15% du dataset), strictement séparé depuis le début du projet.

---

## Les Métriques Utilisées

### 1. Accuracy (Taux de Classification Correct)

```
Accuracy = (Vrais Positifs + Vrais Négatifs) / Total
         = Nombre de prédictions correctes / Total prédictions
```

**Interprétation :** Proportion globale d'images correctement classifiées.

**Limite :** Trompeuse sur un dataset déséquilibré. Si `Potato___healthy` représente 1% du dataset, un modèle qui l'ignore systématiquement aura encore 99% d'accuracy.

**C'est pourquoi nous utilisons aussi Précision, Rappel et F1.**

---

### 2. Précision (Precision)

```
Précision_classe_i = VP_i / (VP_i + FP_i)
                   = Vrais Positifs / (Vrais Positifs + Faux Positifs)
```

**Interprétation :** Parmi toutes les images **prédites comme classe i**, quelle proportion l'est vraiment ?

**Exemple concret :** Si le modèle prédit "Tomato Late blight" 200 fois, et que 160 sont correctes → Précision = 160/200 = **80%**.

**Quand maximiser la Précision ?** Quand le coût d'une **fausse alarme** est élevé. Ex: un agriculteur qui traite une plante saine par erreur gaspille des pesticides.

---

### 3. Rappel (Recall / Sensibilité)

```
Rappel_classe_i = VP_i / (VP_i + FN_i)
                = Vrais Positifs / (Vrais Positifs + Faux Négatifs)
```

**Interprétation :** Parmi toutes les images qui **appartiennent vraiment à la classe i**, quelle proportion le modèle a-t-il correctement identifiée ?

**Exemple concret :** S'il y a 287 images de Late blight dans le test set, et que le modèle en identifie correctement 261 → Rappel = 261/287 = **91%**.

**Quand maximiser le Rappel ?** Quand le coût d'un **cas manqué** est élevé. Ex: manquer une maladie infectious peut détruire une récolte entière. **Dans notre contexte agricole, le Rappel est la métrique prioritaire.**

---

### 4. F1-Score

```
F1 = 2 × (Précision × Rappel) / (Précision + Rappel)
```

**Interprétation :** Moyenne harmonique de la Précision et du Rappel. Pénalise les modèles qui excellent sur l'une mais sont mauvais sur l'autre.

Pour un dataset déséquilibré, on utilise le **F1 weighted** (pondéré par le nombre d'images de chaque classe), qui est plus représentatif que le F1 macro (traite toutes les classes également).

---

## Résultats Complets par Modèle

### Random Forest — Test Set (2 044 images)

**Accuracy : 89.04% | F1-score weighted : 88.79%**

| Classe | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn — Cercospora | 0.73 | 0.72 | 0.72 | 87 |
| Corn — Common rust | 0.95 | 0.93 | 0.94 | 197 |
| Corn — Northern Blight | 0.86 | 0.82 | 0.84 | 173 |
| Corn — Healthy | 0.99 | 0.98 | 0.98 | 175 |
| Potato — Early blight | 0.96 | 0.96 | 0.96 | 150 |
| Potato — Late blight | 0.88 | 0.90 | 0.89 | 150 |
| Potato — Healthy | 0.91 | 0.88 | 0.89 | 24 |
| Tomato — Early blight | 0.88 | 0.79 | 0.83 | 150 |
| Tomato — Late blight | 0.76 | 0.85 | 0.80 | 287 |
| Tomato — Leaf Mold | 0.88 | 0.87 | 0.87 | 144 |
| Tomato — Septoria | 0.87 | 0.96 | 0.91 | 267 |
| Tomato — Healthy | 0.97 | 1.00 | 0.99 | 240 |

---

### SVM (RBF) — Test Set (2 044 images)

**Accuracy : 87.33% | F1-score weighted : 87.37%**

| Classe | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn — Cercospora | 0.73 | 0.70 | 0.72 | 87 |
| Corn — Common rust | 0.95 | 0.92 | 0.94 | 197 |
| Corn — Northern Blight | 0.84 | 0.79 | 0.81 | 173 |
| Corn — Healthy | 0.98 | 0.97 | 0.97 | 175 |
| Potato — Early blight | 0.96 | 0.95 | 0.96 | 150 |
| Potato — Late blight | 0.86 | 0.89 | 0.87 | 150 |
| Potato — Healthy | 0.92 | 0.92 | 0.92 | 24 |
| Tomato — Early blight | 0.86 | 0.77 | 0.81 | 150 |
| Tomato — Late blight | 0.72 | 0.81 | 0.76 | 287 |
| Tomato — Leaf Mold | 0.86 | 0.83 | 0.85 | 144 |
| Tomato — Septoria | 0.89 | 0.87 | 0.88 | 267 |
| Tomato — Healthy | 0.97 | 0.99 | 0.98 | 240 |

---

### MobileNetV2 (DL) — Test Set (2 044 images)

**Accuracy : 93.59% | F1-score weighted : 93.57%**

| Classe | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn — Cercospora | 0.75 | 0.87 | 0.81 | 87 |
| Corn — Common rust | 0.97 | 0.98 | 0.98 | 197 |
| Corn — Northern Blight | 0.94 | 0.86 | 0.90 | 173 |
| Corn — Healthy | 1.00 | 1.00 | 1.00 | 175 |
| Potato — Early blight | 0.97 | 0.98 | 0.97 | 150 |
| Potato — Late blight | 0.88 | 0.91 | 0.90 | 150 |
| Potato — Healthy | 0.69 | 0.92 | 0.79 | 24 |
| Tomato — Early blight | 0.93 | 0.76 | 0.84 | 150 |
| Tomato — Late blight | 0.97 | 0.91 | 0.94 | 287 |
| Tomato — Leaf Mold | 0.93 | 0.97 | 0.95 | 144 |
| Tomato — Septoria | 0.94 | 0.97 | 0.96 | 267 |
| Tomato — Healthy | 0.94 | 1.00 | 0.97 | 240 |

---

## Analyse des Confusions Récurrentes

### Classe difficile : Corn___Cercospora (F1 ≈ 0.72–0.81)

C'est la classe la plus difficile pour **tous les modèles**, y compris MobileNetV2. 

**Raison :** Les taches de Cercospora sont visuellement très similaires à la Northern Leaf Blight (forme allongée, couleur grise). La différence est subtile (taille des taches, présence de halos jaunes) et difficile à capturer avec des histogrammes de couleur.

**Solution envisageable :** Augmenter les données (data augmentation aggressive) ou utiliser EfficientNet qui capte mieux les détails fins.

### Classe facile : Corn___Healthy / Tomato___Healthy (F1 ≈ 0.97–1.00)

Les feuilles saines ont une couleur verte uniforme, une texture homogène et une forme régulière. Ces propriétés créent une **signature HSV unique** très facilement discriminable par tous les modèles.

### Confusion principale ML : Tomato Late blight ↔ Tomato Early blight

La confusion entre ces deux classes explique les F1 plus faibles sur Tomato___Early_blight (0.81–0.84 en ML). Les deux maladies créent des taches brunes mais avec des patterns différents que les histogrammes de couleur ne distinguent pas assez finement. MobileNetV2, grâce aux features convolutives spatiales, les distingue mieux (F1=0.84 vs 0.94).

---

## Matrice de Confusion — Comment la Lire ?

Les matrices de confusion sont sauvegardées dans :
- `confusion_matrix_rf.png`
- `confusion_matrix_svm.png`
- `confusion_matrix_dl.png`

**Lecture :** Chaque cellule (i, j) indique le nombre d'images de la **vraie classe i** qui ont été **prédites comme classe j**.
- La **diagonale principale** (cases vertes) = prédictions correctes
- Les **cases hors diagonale** = erreurs de classification

Une matrice parfaite aurait toutes les images sur la diagonale.

---

*Document généré dans le cadre du projet académique — Classification des Maladies des Plantes*

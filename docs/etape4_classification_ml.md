# 📋 Rapport Technique — Étape 4 : Classification par Machine Learning

> **Projet :** Détection et Classification Automatique de Maladies des Plantes  
> **Section :** Pipeline ML (Random Forest + SVM)  
> **Fichier principal :** `train_ml.py` | `src/features/extractors.py`

---

## Vue d'Ensemble du Pipeline ML

```
Images brutes (data/raw)
        │
        ▼
  Splitter 70/15/15 (seed=42)
        │
   ┌────┴────┐
   │         │
 Train      Test
 (9 490)   (2 044)
   │
   ▼
 Segmentation HSV → Masque binaire
        │
        ▼
 Extraction Features (566 dims)
   RGB hist + HSV hist + GLCM + Hu
        │
        ▼
 StandardScaler (fit sur Train uniquement)
        │
      ┌─┴─┐
      │   │
     RF   SVM
      │   │
      └─┬─┘
        ▼
 Évaluation sur Test Set
 (Accuracy, Precision, Recall, F1)
```

---

## Étape 4.1 — Division du Dataset (Train / Val / Test)

### Configuration choisie : 70% / 15% / 15%

La division du dataset est la première décision critique. Elle conditionne la validité de toutes les métriques obtenues.

| Sous-ensemble | Taille | Rôle |
|---------------|--------|------|
| **Train** | 9 490 images | Apprentissage des paramètres des modèles |
| **Val** | 2 029 images | Réglage des hyperparamètres (EarlyStopping pour DL) |
| **Test** | 2 044 images | Évaluation finale, **jamais vue pendant l'entraînement** |

### Résultats de distribution par classe

| Classe | Train | Val | Test |
|--------|-------|-----|------|
| Corn — Cercospora | 401 | 86 | 87 |
| Corn — Common rust | 914 | 195 | 197 |
| Corn — Northern Blight | 802 | 171 | 173 |
| Corn — Healthy | 813 | 174 | 175 |
| Potato — Early blight | 700 | 150 | 150 |
| Potato — Late blight | 700 | 150 | 150 |
| Potato — Healthy | 106 | 22 | 24 |
| Tomato — Early blight | 700 | 150 | 150 |
| Tomato — Late blight | 1 336 | 286 | 287 |
| Tomato — Leaf Mold | 666 | 142 | 144 |
| Tomato — Septoria | 1 239 | 265 | 267 |
| Tomato — Healthy | 1 113 | 238 | 240 |
| **TOTAL** | **9 490** | **2 029** | **2 044** |

### Pourquoi le Split Stratifié est obligatoire ici ?

Le dataset est **fortement déséquilibré** : de 24 images (Potato___healthy dans le test) à 1 336 images (Tomato___Late_blight dans le train). Un split aléatoire non-stratifié risquerait de :
- Mettre toutes les images de `Potato___healthy` dans Train → le Test n'a aucun exemple de cette classe = métriques faussées.
- Surreprésenter les grandes classes dans Train → biais d'apprentissage.

Le split stratifié garantit que **chaque classe est représentée dans la même proportion** dans les trois ensembles.

---

## Étape 4.2 — Vecteur d'Entrée du Modèle ML (566 dimensions)

Avant de passer au modèle, chaque image est transformée en un vecteur numérique. Le pipeline complet pour une image est :

```python
# 1. Segmentation
image_rgb, mask, segmented = segment_leaf(image_path)

# 2. Extraction des 4 types de features
features = extract_features(image_rgb, mask)  # → np.array de 566 valeurs

# 3. Normalisation
features_scaled = scaler.transform([features])  # → valeurs centrées-réduites

# 4. Prédiction
prediction = model.predict(features_scaled)
```

### Les 566 dimensions du vecteur

| Feature | Dimensions | Capture quoi ? |
|---------|------------|----------------|
| Histogramme RGB | 24 | Distribution des couleurs brutes |
| Histogramme HSV 3D | 512 | Distribution teinte/saturation/luminosité |
| GLCM (5 props × 4 angles) | 20 | Texture : rugosité, homogénéité, motifs |
| Moments de Hu (×7) + métriques | 10 | Forme globale de la zone malade |

Ce vecteur encode l'information visuelle de manière **compacte, interprétable et invariante**.

---

## Étape 4.3 — Modèle 1 : Random Forest (RF)

### Principe

Random Forest est un ensemble de **N arbres de décision** entraînés sur des sous-échantillons aléatoires du dataset (bagging) et avec des sous-ensembles aléatoires de features à chaque nœud (feature randomness).

La prédiction finale est le **vote majoritaire** des N arbres. Cette double source d'aléatoire réduit fortement la variance (overfitting) par rapport à un seul arbre de décision.

### Hyperparamètres choisis

```python
RandomForestClassifier(
    n_estimators    = 300,      # Nombre d'arbres
    class_weight    = 'balanced',  # Correction du déséquilibre
    n_jobs          = -1,       # Parallélisation sur tous les CPU
    random_state    = 42        # Reproductibilité
)
```

#### Pourquoi `n_estimators = 300` ?

La performance d'un RF augmente avec le nombre d'arbres jusqu'à un **plateau de saturation**. Des études empiriques montrent que ce plateau est atteint entre 100 et 500 arbres pour des datasets de ~10 000 images avec ~500 features. 300 est un compromis entre performance maximale et temps de calcul raisonnable (5.5 secondes sur CPU).

| n_estimators | Performance | Temps calcul |
|-------------|-------------|-------------|
| 50 | -2 à -3% accuracy | ~1s |
| 100 | -0.5% accuracy | ~2s |
| **300** | **Optimal** | **~5.5s** |
| 500 | +0.1% (non significatif) | ~9s |

#### Pourquoi `class_weight = 'balanced'` ?

`Potato___healthy` ne représente que **152 images** sur 13 563 (1.1% du dataset), contre 1 909 pour `Tomato___Late_blight` (14.1%). Sans correction, le RF aurait tendance à ignorer les classes minoritaires.

`class_weight='balanced'` calcule automatiquement un poids inverse à la fréquence de chaque classe :
```
poids_classe = n_total / (n_classes × n_images_classe)
```

Cela force le modèle à **pénaliser davantage** les erreurs sur les classes rares.

### Résultats Random Forest

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **89.04%** |
| **Précision** (weighted) | 89.10% |
| **Rappel** (weighted) | 89.04% |
| **F1-score** (weighted) | 88.79% |
| Temps d'entraînement | 5.5s |
| Temps d'inférence (test set) | 0.09s |

#### Meilleures et pires classes

| Classe | F1-score | Observation |
|--------|----------|-------------|
| Tomato___healthy | 0.99 | Très discriminable (vert uniforme) |
| Corn___Common_rust | 0.94 | Couleur orange très spécifique |
| Tomato___Late_blight | 0.88 | Bonne malgré la variabilité |
| Corn___Cercospora | **0.72** | Similaire visuellement à Septoria |
| Tomato___Early_blight | **0.83** | Confusion avec Late blight |

---

## Étape 4.4 — Modèle 2 : SVM à Noyau RBF

### Principe

Le SVM (Support Vector Machine) cherche l'**hyperplan de séparation optimal** qui maximise la **marge** entre les classes les plus proches. Pour des données non-linéairement séparables, le **noyau RBF** (Radial Basis Function) projette implicitement les données dans un espace de dimension infinie où elles deviennent séparables.

```
K(x, y) = exp(-γ × ||x - y||²)
```

### Pourquoi le Noyau RBF ?

Nos features (histogrammes, GLCM) ne sont **pas linéairement séparables**. Un SVM linéaire obtiendrait ~70-75% d'accuracy. Le noyau RBF capte les **relations non-linéaires** entre les features couleur/texture/forme, ce qui est essentiel pour distinguer des maladies visuellement proches.

| Noyau | Accuracy estimée | Usage recommandé |
|-------|-----------------|-----------------|
| Linéaire | ~75% | Features linéairement séparables |
| Polynomial | ~82% | Relations polynomiales |
| **RBF** | **87.33%** | **Relations complexes non-linéaires** ← Notre cas |
| Sigmoid | ~78% | Peu utilisé en pratique |

### Hyperparamètres choisis

```python
SVC(
    kernel        = 'rbf',
    C             = 10,           # Paramètre de régularisation
    gamma         = 'scale',      # γ = 1 / (n_features × Var(X))
    class_weight  = 'balanced',   # Correction du déséquilibre
    probability   = True,         # Activer les probabilités (Top-3 Streamlit)
    random_state  = 42
)
```

#### Pourquoi `C = 10` ?

`C` est le paramètre de régularisation qui contrôle le compromis **marge large / erreurs de classification**.

- **C faible (ex: 0.1)** : grande marge, tolère plus d'erreurs → underfitting
- **C élevé (ex: 1000)** : petite marge, fit précis sur train → overfitting
- **C = 10** : valeur standard recommandée dans la littérature pour les données déjà normalisées (StandardScaler). Validé par grid search implicite sur 5 valeurs testées empiriquement.

#### Pourquoi `gamma = 'scale'` ?

`gamma` contrôle l'influence de chaque point d'entraînement dans le noyau RBF.

`gamma='scale'` calcule automatiquement :
```
γ = 1 / (n_features × Var(X_train)) = 1 / (566 × σ²)
```

Cette formule adapte γ au nombre de features et à leur variance réelle, évitant le sur- ou sous-lissage. `gamma='auto'` (ancienne option) utilise `1/n_features` sans tenir compte de la variance, ce qui est moins précis.

#### Pourquoi `probability = True` ?

Cette option active l'estimation de probabilité par la méthode de **Platt scaling** (calibration isotonique). Elle est nécessaire pour afficher dans l'application Streamlit les **probabilités Top-3** par classe.
**Inconvénient** : ralentit l'entraînement de ~3× (validation croisée 5-fold interne). C'est pour cela que le SVM prend **44.6s** contre 5.5s pour le RF.

### Résultats SVM (RBF)

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **87.33%** |
| **Précision** (weighted) | 87.57% |
| **Rappel** (weighted) | 87.33% |
| **F1-score** (weighted) | 87.37% |
| Temps d'entraînement | 44.6s |
| Temps d'inférence (test set) | 5.4s |

---

## Étape 4.5 — Normalisation StandardScaler

**Rôle critique du scaler :** Le SVM est particulièrement sensible à l'échelle des features.

Sans normalisation :
- Les 512 valeurs de l'histogramme HSV (plage [0, 1]) dominent numériquement
- Les Moments de Hu (plage [1e-10, 10]) ont une contribution négligeable
- Résultat : SVM sous-optimal (~75% accuracy)

Avec `StandardScaler` :
```
x_normalisé = (x - μ_train) / σ_train
```

Toutes les features ont **moyenne = 0** et **écart-type = 1** → contributions équilibrées.

> ⚠️ **Le scaler est fitté uniquement sur les données d'entraînement** (`fit_transform(X_train)`) et appliqué aux données de test (`transform(X_test)`). Utiliser `fit_transform` sur le test set causerait une **fuite de données** (le modèle "verrait" la distribution des données de test).

---

## Étape 4.6 — Sauvegarde des Modèles

Les modèles sont sauvegardés avec `joblib` dans un dictionnaire contenant :

```python
{
    "model":   <instance du classifieur>,
    "scaler":  <instance du StandardScaler ajusté>,
    "classes": <liste des 12 classes dans l'ordre>,
    "train_time": <temps d'entraînement en secondes>,
    "accuracy": <accuracy sur le test set>
}
```

La sauvegarde du **scaler** avec le modèle est essentielle : pour prédire une nouvelle image, il faut appliquer exactement la même normalisation que lors de l'entraînement.

---

## Comparaison RF vs SVM

| Critère | Random Forest | SVM (RBF) |
|---------|--------------|-----------|
| **Accuracy** | **89.04%** | 87.33% |
| **F1-score** | **88.79%** | 87.37% |
| Temps entraînement | **5.5s** | 44.6s |
| Temps inférence | **0.09s** | 5.4s |
| Mémoire modèle | ~7 Mo | ~15 Mo |
| Interprétabilité | Feature importance | Difficile (black box) |
| Robustesse au bruit | Élevée (bagging) | Modérée |
| Sensible à l'échelle | Non | Oui (scaler requis) |

### Conclusion

**Random Forest est le meilleur modèle ML classique** pour ce projet :
- 89.04% d'accuracy, soit +1.7% de plus que SVM
- 8× plus rapide à l'entraînement
- 60× plus rapide à l'inférence
- Plus facile à interpréter grâce aux importances de features

Les deux modèles sont toutefois largement surpassés par **MobileNetV2 (93.59%)**, qui bénéficie d'un apprentissage de représentation automatique via le Transfer Learning.

---

*Document généré dans le cadre du projet académique — Classification des Maladies des Plantes*

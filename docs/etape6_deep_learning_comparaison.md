# 📋 Rapport Technique — Étape 6 : Extension Deep Learning & Étude Comparative

> **Projet :** Détection et Classification Automatique de Maladies des Plantes  
> **Section :** MobileNetV2 (Transfer Learning) + Benchmark ML vs DL  
> **Fichiers :** `train_dl.py`, `src/utils/benchmark.py`, `predict.py`

---

## Pourquoi Passer au Deep Learning ?

Les modèles ML classiques (RF, SVM) ont une limite fondamentale : ils dépendent de **features artisanales** (histogrammes, GLCM, Hu Moments) conçues manuellement. Ces features peuvent manquer des patterns subtils que l'œil humain perçoit mais que les formules mathématiques ne capturent pas.

Le Deep Learning (CNN) **apprend automatiquement** les features pertinentes directement depuis les pixels bruts, sans intervention humaine. Avec suffisamment de données et de puissance de calcul, il surpasse systématiquement le ML classique en vision par ordinateur.

---

## Architecture Choisie : MobileNetV2 (Transfer Learning)

### Pourquoi MobileNetV2 et pas un CNN from scratch ?

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **CNN from scratch** | Totalement adaptable | Nécessite 100k+ images, GPU puissant, semaines d'entraînement |
| **Transfer Learning MobileNetV2** ✅ | Connaissances ImageNet, entraînable en heures sur CPU | Moins flexible |
| ResNet50 | Plus précis | 4× plus lourd (~25M params) |
| EfficientNetB0 | Légèrement meilleur | Plus récent, moins documenté |

Avec seulement **~13 500 images** et un entraînement sur **CPU**, un CNN from scratch ne convergerait pas. Le Transfer Learning réutilise les **3,4 millions de paramètres** de MobileNetV2 déjà entraînés sur 1,2 million d'images ImageNet.

### Architecture Complète

```
Input : 224 × 224 × 3 (RGB)
    │
    ├── Data Augmentation (entraînement seulement)
    │     RandomFlip(horizontal)
    │     RandomRotation(±10%)
    │     RandomZoom(±20%)
    │
    ├── preprocess_input(x) → normalise [-1, 1]
    │     (spécifique MobileNetV2, pas [0,1] ou zscore)
    │
    ├── MobileNetV2 (ImageNet weights)
    │     154 couches convolutives
    │     Dépthwise Separable Convolutions
    │     Bottlenecks résiduels (inverted residuals)
    │     Sortie : Feature Map 7 × 7 × 1 280
    │
    ├── GlobalAveragePooling2D → 1 280 valeurs
    │
    ├── BatchNormalization
    │
    ├── Dropout(0.3)
    │
    └── Dense(12, activation='softmax') → 12 probabilités
```

### Pourquoi `preprocess_input` de MobileNetV2 ?

MobileNetV2 a été entraîné avec des pixels normalisés dans **[-1, 1]** (pas [0, 1] ni z-score). Utiliser une autre normalisation briserait la compatibilité avec les poids ImageNet et dégraderait les performances de ~5-8%.

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# x_normalized = (x / 127.5) - 1  ← formule interne
```

### Pourquoi `GlobalAveragePooling2D` et pas `Flatten` ?

`Flatten` de la feature map 7×7×1280 produirait un vecteur de **62 720 valeurs** → risque d'overfitting élevé avec seulement 9 490 images de train.

`GlobalAveragePooling2D` calcule la **moyenne spatiale** de chaque des 1 280 canaux → vecteur de **1 280 valeurs**. Moins paramétrique, plus régularisé, et meilleurs résultats empiriques.

---

## Stratégie d'Entraînement : Fine-Tuning en 2 Phases

### Phase 1 — Feature Extraction (couches gelées)

```python
base_model.trainable = False  # Geler toutes les couches MobileNetV2
optimizer = Adam(learning_rate=1e-3)
epochs = 10 (avec EarlyStopping patience=5)
```

**Objectif :** Entraîner uniquement la tête de classification (Dense 12) à partir des features génériques extraites par MobileNetV2.

**Pourquoi geler ?** Les couches de bas niveau de MobileNetV2 (bords, textures, couleurs) sont déjà optimales pour ImageNet et fonctionnent aussi bien pour les maladies végétales. Les dégeler au début causerait une catastrophic forgetting.

### Phase 2 — Fine-tuning des 30 Dernières Couches

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  # Geler tout sauf les 30 dernières couches

optimizer = Adam(learning_rate=1e-5)  # lr 100× plus faible
epochs = 10 supplémentaires (avec EarlyStopping)
```

**Objectif :** Adapter les features de haut niveau de MobileNetV2 (formes complexes, textures spécifiques aux maladies) au domaine agricole.

**Pourquoi seulement les 30 dernières couches ?** Les premières couches détectent des features génériques (lignes, textures simples) valables pour tous les domaines. Les dernières couches détectent des concepts spécifiques au domaine (races de chiens, objets ImageNet). Ce sont ces dernières qu'on adapte au domaine agricole.

**Pourquoi `learning_rate = 1e-5` ?** Un lr trop élevé (1e-3) en Phase 2 détruirait les poids pré-entraînés (catastrophic forgetting). `1e-5` permet des ajustements subtils et progressifs.

---

## Callbacks de Régularisation

```python
EarlyStopping(
    monitor  = 'val_accuracy',
    patience = 5,        # Arrêter si pas d'amélioration pendant 5 epochs
    restore_best_weights = True  # Remettre les meilleurs poids
)

ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.3,      # Réduire lr × 0.3 en cas de plateau
    patience = 3,        # Attendre 3 epochs avant de réduire
    min_lr   = 1e-7
)

ModelCheckpoint(
    save_best_only = True,  # Sauvegarder seulement le meilleur modèle
    monitor = 'val_accuracy'
)
```

### Résultats de l'Entraînement DL

**Fin de Phase 2 (meilleur epoch) :**
- Val Accuracy : **93.40%** (epoch 10/10)
- Val Loss : 0.1740

**Évaluation finale sur Test Set :**

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | **93.59%** |
| **Test Loss** | 0.1867 |
| **Précision** (weighted) | 93.85% |
| **Rappel** (weighted) | 93.59% |
| **F1-score** (weighted) | 93.57% |
| Temps entraînement total | ~90 min (CPU) |
| Temps d'inférence (test set) | 34.6s (2 044 images) |

---

## Explicabilité — Grad-CAM

Le Deep Learning est souvent reproché d'être une "boîte noire". Pour le rendre interprétable, nous avons implémenté **Grad-CAM** (Gradient-weighted Class Activation Mapping).

### Principe

1. Faire une passe avant (forward pass) et obtenir la prédiction
2. Calculer les gradients de la classe prédite par rapport aux **activations de la dernière couche convolutive** (`out_relu`, sortie 7×7×1280 de MobileNetV2)
3. Faire la moyenne des gradients spatialement → **poids d'importance** par canal
4. Pondérer les activations par ces poids → **carte d'activation 7×7**
5. Appliquer ReLU (ne garder que les contributions positives)
6. Redimensionner à 224×224 et superposer sur l'image originale

```python
# src/predict.py
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    conv_outputs, predictions = grad_model(img_tensor, training=False)
    tape.watch(conv_outputs)
    class_score = predictions[:, tf.argmax(predictions[0])]

grads = tape.gradient(class_score, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
heatmap = tf.nn.relu(heatmap).numpy()
```

### Interprétation

- **Zones rouges/chaudes** : régions qui ont le plus influencé la décision du modèle
- **Zones bleues/froides** : régions sans influence

Pour les maladies foliaires, Grad-CAM met en évidence les **taches nécrotiques**, les **bords de lésions** et les **sporulations**, confirmant que le modèle analyse bien les bons indices visuels.

---

## Étude Comparative Complète : ML vs Deep Learning

### Résultats du Benchmark (Test Set : 2 044 images)

| Modèle | Accuracy | Précision | Rappel | F1-score | Entraînement | Inférence/image |
|--------|----------|-----------|--------|----------|-------------|-----------------|
| Random Forest | 89.04% | 89.10% | 89.04% | 88.79% | 5.5s | 0.04ms |
| SVM (RBF) | 87.33% | 87.57% | 87.33% | 87.37% | 44.6s | 2.6ms |
| **MobileNetV2** | **93.59%** | **93.85%** | **93.59%** | **93.57%** | ~90min | 17ms |

### Analyse par Axe

#### 1. Précision (Accuracy / F1-score)

MobileNetV2 surpasse les modèles ML de **+4.55%** en accuracy. Cette différence, significative en pratique agricole, s'explique par la richesse de la représentation apprise :
- RF/SVM voient un vecteur de 566 valeurs (features artisanales)
- MobileNetV2 traite directement les pixels via 3,4M de paramètres spatiaux

#### 2. Vitesse d'Entraînement

| Modèle | Vitesse | Justification |
|--------|---------|---------------|
| Random Forest | ⚡ 5.5s | Parallèle, arbres indépendants |
| SVM | 44.6s | Optimisation quadratique + Platt scaling |
| MobileNetV2 | ~90 min | 3,4M paramètres, 10 epochs × 2 phases |

**Pour une mise en production rapide**, RF est clairement supérieur.

#### 3. Vitesse d'Inférence

| Modèle | ms / image | Adapté à |
|--------|-----------|---------|
| Random Forest | 0.04ms | Systèmes embarqués, Raspberry Pi |
| SVM | 2.6ms | Applications web légères |
| MobileNetV2 | 17ms | Applications bureau, serveurs |

Le nom "Mobile" de MobileNetV2 vient de son architecture optimisée pour les **appareils mobiles** (smartphones). Sur GPU, l'inférence descendrait à <1ms.

#### 4. Interprétabilité

| Modèle | Méthode | Niveau de détail |
|--------|---------|-----------------|
| Random Forest | Feature Importance | Quelles features (GLCM vs Couleur) sont les plus utiles |
| SVM | SHAP values (non implémenté) | Contribution par feature pour chaque prédiction |
| **MobileNetV2** | **Grad-CAM** | **Localisation spatiale des zones décisives sur l'image** |

MobileNetV2 offre l'explicabilité la plus riche grâce à Grad-CAM, ce qui est crucial pour convaincre les agriculteurs d'adopter l'outil.

#### 5. Besoins en Ressources

| Modèle | RAM | Taille modèle | GPU requis |
|--------|-----|--------------|-----------|
| Random Forest | ~500 Mo | ~7 Mo | Non |
| SVM | ~800 Mo | ~15 Mo | Non |
| MobileNetV2 | ~2 Go | ~14 Mo | Optionnel |

---

## Conclusion et Recommandations

### Pour la soutenance

| Scénario | Recommandation | Justification |
|----------|---------------|---------------|
| **Précision maximale** | MobileNetV2 (93.59%) | Transfer Learning ImageNet |
| **Déploiement embarqué** | Random Forest (89.04%) | Ultra-rapide, légère RAM |
| **Interprétabilité max** | MobileNetV2 + Grad-CAM | Visualisation spatiale |
| **Entraînement rapide** | Random Forest (5.5s) | Réentraîner rapidement sur nouvelles données |

### Points Forts de notre Approche Hybride

Notre système n'impose pas un seul modèle — l'interface Streamlit permet à l'utilisateur de **choisir le modèle** selon son contexte :
- Inspection rapide sur terrain → Random Forest
- Diagnostic précis avec explicabilité → MobileNetV2

Cette flexibilité est un véritable atout académique et pratique.

---

*Document généré dans le cadre du projet académique — Classification des Maladies des Plantes*

"""
agricultural_advice.py
----------------------
Base de connaissances agronomiques pour les 12 classes du projet.
"""

ADVICE_DICT = {
    # ── Maïs ────────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "🌽 Tache grise / Cercospora\n"
        "• Appliquer un fongicide à base d'azoxystrobine dès les premières taches.\n"
        "• Privilégier les hybrides tolérants lors du prochain semis.\n"
        "• Éviter l'excès d'humidité foliaire (irrigation le matin)."
    ),
    "Corn_(maize)___Common_rust_": (
        "🌽 Rouille commune\n"
        "• Traiter avec un fongicide triazole (tébuconazole) si >5% de feuilles atteintes.\n"
        "• Éviter les semis tardifs favorisant l'humidité.\n"
        "• Surveiller les conditions météo (température 16–25°C = risque maximum)."
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "🌽 Brûlure nordique des feuilles\n"
        "• Rotation des cultures sur 1–2 ans pour éliminer les résidus infectés.\n"
        "• Appliquer un fongicide protecteur (mancozèbe) dès les stades V6–V8.\n"
        "• Enfouir les résidus de récolte pour limiter l'inoculum."
    ),
    "Corn_(maize)___healthy": (
        "🌽 Maïs sain ✅\n"
        "• Plant en bonne santé. Maintenir la fertilisation azotée recommandée.\n"
        "• Surveiller régulièrement pour détecter précocement toute apparition de maladie."
    ),

    # ── Pomme de terre ───────────────────────────────────────────────────────
    "Potato___Early_blight": (
        "🥔 Alternariose (Early blight)\n"
        "• Retirer les feuilles basales infectées et les détruire hors du champ.\n"
        "• Appliquer du chlorothalonil ou du mancozèbe toutes les 7–10 jours.\n"
        "• Assurer une bonne nutrition en potassium pour renforcer la résistance."
    ),
    "Potato___Late_blight": (
        "🥔 Mildiou (Late blight) ⚠ URGENT\n"
        "• Maladie destructrice — agir dans les 24h après détection.\n"
        "• Traitement systémique : métalaxyl-M + mancozèbe ou cymoxanil + fongicide de contact.\n"
        "• Arracher et brûler les plants trop atteints. Ne pas composter.\n"
        "• Surveiller météo : risque maximal entre 10–24°C avec humidité >90%."
    ),
    "Potato___healthy": (
        "🥔 Pomme de terre saine ✅\n"
        "• Plant en bonne santé. Contrôler l'humidité du sol et éviter l'excès d'irrigation.\n"
        "• Traitement préventif en période à risque mildiou (fongicide à base de cuivre)."
    ),

    # ── Tomate ───────────────────────────────────────────────────────────────
    "Tomato___Early_blight": (
        "🍅 Alternariose tomate (Early blight)\n"
        "• Pailler le sol pour éviter les éclaboussures de terre sur le feuillage.\n"
        "• Pulvériser de la bouillie bordelaise (cuivre) toutes les 10–14 jours.\n"
        "• Supprimer les feuilles inférieures touchées pour limiter la progression."
    ),
    "Tomato___Late_blight": (
        "🍅 Mildiou tomate (Late blight) ⚠ URGENT\n"
        "• Arracher immédiatement les plants fortement atteints et détruire hors serre.\n"
        "• Appliquer un fongicide systémique à base de cuivre ou métalaxyl.\n"
        "• Aérer la serre / serre tunnel pour réduire l'humidité résiduelle."
    ),
    "Tomato___Leaf_Mold": (
        "🍅 Cladosporiose (Leaf Mold)\n"
        "• Améliorer la circulation d'air : tailler les gourmands, espacer les plants.\n"
        "• Éviter d'arroser le feuillage — arrosage au pied uniquement.\n"
        "• Traitement fongicide : chlorothalonil ou bifonazole en préventif."
    ),
    "Tomato___Septoria_leaf_spot": (
        "🍅 Septoriose (Septoria leaf spot)\n"
        "• Supprimer les feuilles présentant les petites taches circulaires blanches.\n"
        "• Appliquer un fongicide cuivrique ou à base de manèbe dès les premiers symptômes.\n"
        "• Respecter la rotation culturale (éviter tomate au même emplacement 2 ans de suite)."
    ),
    "Tomato___healthy": (
        "🍅 Tomate saine ✅\n"
        "• Plant en bonne santé. Retirer les gourmands pour une meilleure aération.\n"
        "• Maintenir un arrosage régulier au pied pour éviter les fluctuations hydriques."
    ),
}


def get_advice(class_name: str) -> str:
    """Retourne le conseil agricole pour une classe donnée."""
    return ADVICE_DICT.get(
        class_name,
        "⚠ Classe non reconnue. Consultez un expert agricole de votre région."
    )


# Liste des 12 classes officielles du projet
TARGET_CLASSES = sorted(ADVICE_DICT.keys())

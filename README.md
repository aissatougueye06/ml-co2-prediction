# Prédiction des émissions de CO₂ — Machine Learning

Application des méthodes de la Machine Learning Specialization (DeepLearning.AI) au jeu de données ADEME des véhicules commercialisés en France.

## Notebooks

- **`01-regression-simple.ipynb`** — régression linéaire à une variable (masse → CO₂), sur les motorisations diesel et essence. Coût, descente de gradient, normalisation.
- **`02-regression-multiple.ipynb`** — régression multiple (masse, puissance, boîte de vitesses). Détection de deux fuites de données, dédoublonnage des variantes d'homologation, feature engineering, diagnostic des résidus et domaine de validité.
- **`03-classification.ipynb`** — le véhicule dépasse-t-il le seuil du malus écologique (131 g/km) ? Deux approches comparées sur les diesels et les essences : classer directement (régression logistique), ou prédire le CO₂ par régression multiple puis appliquer le seuil. Pipelines scikit-learn (`ColumnTransformer`, one-hot, terme quadratique, normalisation), découpage commun stratifié, métriques de classification, analyse des erreurs et robustesse sur plusieurs découpages.

Les fonctions de chargement et de préparation des données (`charger`, `preparer`) sont dans `preparation.py`.

## Ce que donnent les modèles

### Prédire le CO₂ (notebook 02)

Sur les essences, l'erreur moyenne est de **15,5 g/km sur le jeu de test**, contre 42,9 g/km pour une prédiction constante — l'erreur est divisée par 2,8. Elle reste très inégale selon le segment : 11,5 g/km sous 75 kW, 31,7 g/km au-delà de 300 kW. Ces chiffres sont mesurés sur un découpage unique et favorable ; le rapport au modèle naïf, lui, est stable d'un tirage à l'autre.

Trois résultats qui ne se voyaient pas dans l'analyse exploratoire :

- Neuf colonnes sur 26 sont dérivées de la cible. L'une d'elles (`puiss_admin`) est invisible dans une matrice de corrélation parce que la relation est non linéaire.
- Une ligne du fichier n'est pas un véhicule mais une variante d'homologation : 65 % des lignes essence sont des réplicats.
- La puissance explique 76 % de la variance chez l'essence contre 12 % chez le diesel. Deux populations, deux modèles.

### Prédire le malus (notebook 03)

Les deux approches se valent : **F1 de 87 à 89 %** selon le découpage, pour une accuracy d'environ 85 % contre 57 % en prédisant toujours la classe majoritaire. L'écart entre elles est plus petit que la variation d'un découpage à l'autre.

Trois résultats :

- Elles se trompent différemment, et de façon reproductible : le classifieur a la meilleure précision (moins de malus annoncés à tort), la régression le meilleur rappel (moins de malus manqués), sur les trois découpages testés. Le choix relève du coût métier, pas de la performance.
- Les erreurs se concentrent à la frontière : environ 41 % d'erreur à moins de 5 g/km du seuil, environ 3 % au-delà de 20 g/km. Les 85 % globaux masquent ces deux régimes.
- La motorisation n'explique pas les erreurs : l'écart diesel/essence change de signe d'un découpage à l'autre.

Les limites sont nommées dans la conclusion des notebooks 02 et 03.

## Données

`data/vehicules.csv` — ADEME 2014, ~20 900 versions de véhicules, versionné dans le dépôt pour que les résultats restent reproductibles. L'étude porte sur les motorisations thermiques pures (`GO`, `ES`), soit 19 609 lignes ; hybrides, gaz et électriques sortent du périmètre.

Source : [data.gouv.fr](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france),
Licence Ouverte / Etalab.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Stack

Python 3.11+ · pandas · numpy · matplotlib · scikit-learn

## Voir aussi

- [analyse-co2-vehicules](https://github.com/aissatougueye06/analyse-co2-vehicules) — l'analyse exploratoire préalable
- [ml-notes](https://github.com/aissatougueye06/ml-notes) — le carnet technique alimenté au fil de ces projets

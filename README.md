# Prédiction des émissions de CO₂ — Machine Learning


Application des méthodes de la Machine Learning Specialization (DeepLearning.AI) au jeu de données ADEME des véhicules commercialisés en France.


## Notebooks

- **`01-regression-simple.ipynb`** — régression linéaire à une variable (masse → CO₂), sur les motorisations diesel et essence. Coût, descente de gradient, normalisation.
- **`02-regression-multiple.ipynb`** — régression multiple (masse, puissance, boîte de vitesses). Détection de deux fuites de données, dédoublonnage des variantes d'homologation, feature engineering, diagnostic des résidus et domaine de validité.
- **`03-classification.ipynb`** — le véhicule dépasse-t-il le seuil du malus écologique (131 g/km) ? Deux approches comparées sur les diesels et les essences : classer directement (régression logistique), ou prédire le CO₂ par régression multiple puis appliquer le seuil. Pipelines scikit-learn (`ColumnTransformer`, one-hot, terme quadratique, normalisation), découpage commun stratifié, métriques de classification, analyse des erreurs et robustesse sur plusieurs découpages.
- **`04-arbres-boosting.ipynb`** — arbre de décision, forêt aléatoire et boosting (HistGradientBoosting, XGBoost) sur les essences, comparés à la régression utilisée dans `co2-api`. Découpage par configuration (`GroupKFold`, `GroupShuffleSplit`) pour neutraliser les véhicules jumeaux, recherche d'hyperparamètres, comparaison appariée pli par pli, erreur par tranche de puissance et limites d'extrapolation des arbres.

Les fonctions de chargement et de préparation des données (`charger`, `preparer`) sont dans `preparation.py` (notebooks 01 à 03) et `prepa_arbres_boosting.py` (notebook 04).


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

### Arbres et boosting (notebook 04)

Sur des configurations de véhicules jamais vues, le boosting (HistGradientBoosting) ramène l'erreur moyenne de **14,5 à 8,6 g/km sur le jeu de test**, par rapport à la régression de `co2-api`, avec les trois mêmes variables. XGBoost fait jeu égal ; la forêt aléatoire est un peu en retrait (11,1 contre 10,6 g/km en validation croisée, écart confirmé pli par pli). Ces chiffres ne se comparent pas directement à ceux du notebook 02 : le découpage par configuration est plus exigeant.

Trois résultats :

- Un découpage aléatoire surestimait les arbres : 56 % des véhicules de test avaient un « jumeau » (même masse, même puissance, même boîte) à l'entraînement, et un arbre qui apprend par cœur en profitait. Découper par configuration fait passer l'arbre seul de 9,1 à 12,7 g/km, la régression seulement de 15,6 à 16,1.
- L'hypothèse avancée dans le notebook 02 est réfutée : les fortes erreurs sur les véhicules puissants venaient de la forme du modèle linéaire, pas de l'absence de la carrosserie et de l'aérodynamique. En erreur relative, le boosting est stable sur tout le parc (6 à 7 %), alors que la régression se dégrade aux deux extrémités.
- Les arbres n'extrapolent pas : au-delà de la puissance maximale vue à l'entraînement, leur prédiction devient plate. L'API devra signaler les véhicules hors du domaine d'entraînement.

Les limites sont nommées dans la conclusion des notebooks 02, 03 et 04.


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

Python 3.11+ · pandas · numpy · matplotlib · scikit-learn · XGBoost


## Voir aussi

- [analyse-co2-vehicules](https://github.com/aissatougueye06/analyse-co2-vehicules) — l'analyse exploratoire préalable
- [co2-api](https://github.com/aissatougueye06/co2-api) — l'API de prédiction (FastAPI) qui sert le modèle choisi dans le notebook 04
- [ml-notes](https://github.com/aissatougueye06/ml-notes) — le carnet technique alimenté au fil de mes projets

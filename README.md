# Prédiction des émissions de CO₂ — Machine Learning

Application des méthodes de la Machine Learning Specialization (DeepLearning.AI) au jeu de données ADEME des véhicules commercialisés en France.

## Notebooks
- `01-regression-simple.ipynb` — régression linéaire à une variable (masse → CO₂), sur les motorisations diesel et essence
...

## Données
`data/vehicules.csv` — ADEME 2014, ~20 900 versions de véhicules, versionné dans le dépôt pour que les résultats restent reproductibles. Source: [data.gouv.fr](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france), Licence Ouverte / Etalab.

## Installation
```bash
pip install -r requirements.txt
jupyter lab
```

## Stack
Python 3.10+ · pandas · numpy · matplotlib · scikit-learn

Analyse exploratoire préalable : [analyse-co2-vehicules](https://github.com/aissatougueye06/analyse-co2-vehicules)

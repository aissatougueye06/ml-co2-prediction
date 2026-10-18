"""Préparation des données ADEME pour le notebook 03.

Chaîne en trois étapes : `charger` lit le CSV brut, `lignes_exploitables`
filtre les motorisations retenues et les lignes complètes, `preparer`
dédoublonne pour obtenir un véhicule par ligne.
"""

from pathlib import Path

import pandas as pd

# Clés d'identification d'un véhicule réel. La masse est incluse volontairement :
# les finitions d'un même modèle ont des masses différentes, et cette variation
# est une information, pas un doublon.
CLES = ["lib_mrq_doss", "lib_mod_doss", "masse_ordma_min", "puiss_max", "co2_mixte"]

# Colonnes conservées dans le jeu de travail.
GARDEES = [
    "lib_mrq_doss",
    "lib_mod_doss",
    "typ_boite_nb_rapp",
    "masse_ordma_min",
    "puiss_max",
    "co2_mixte",
    "energ",
]

# Colonnes sans lesquelles une ligne est inutilisable.
ESSENTIELLES = ["masse_ordma_min", "puiss_max", "co2_mixte", "typ_boite_nb_rapp"]

# Colonnes dérivées de la cible. Référence documentaire uniquement : l'exclusion
# est déjà faite en positif par `GARDEES`, qui ne retient aucune d'entre elles.
FUITES = [
    "conso_urb_93",
    "conso_exurb",
    "conso_mixte",
    "puiss_admin",
    "co_typ_1",
    "hc",
    "nox",
    "hcnox",
    "ptcl",
]

# Motorisations conservées dans le jeu de travail : diesel et essence.
ENERGIES = ["GO", "ES"]


def charger(chemin: str | Path) -> pd.DataFrame:
    """Lit le CSV ADEME et nettoie les espaces de fin des colonnes texte.

    Le fichier est un export administratif français : séparateur point-virgule,
    encodage latin-1, et des espaces de fin invisibles sur les colonnes à
    format fixe (`"GO "` au lieu de `"GO"`).
    """
    df = pd.read_csv(chemin, sep=";", encoding="latin-1")
    texte = df.select_dtypes(include="str").columns
    df[texte] = df[texte].apply(lambda s: s.str.strip())
    return df


def lignes_exploitables(df: pd.DataFrame) -> pd.DataFrame:
    """Lignes de motorisation 'GO' et 'ES' dont les colonnes essentielles sont renseignées.

    Étape intermédiaire : les variantes d'homologation ne sont pas encore
    dédoublonnées, donc une ligne n'est pas encore un véhicule.
    """

    return df[df["energ"].isin(ENERGIES)][GARDEES].dropna(subset=ESSENTIELLES)


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre sur deux motorisations ('GO' et 'ES'), supprime les NaN et les doublons d'homologation.

    Une ligne du fichier est une variante d'homologation, pas un véhicule : un
    modèle déclaré en 18 versions pèserait 18 fois plus lourd dans l'entraînement.
    Le dédoublonnage sur `CLES` ramène le fichier à des véhicules réels.
    """

    return lignes_exploitables(df).drop_duplicates(subset=CLES)

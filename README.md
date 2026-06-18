# Flight Delays MLOps

Projet MLOps autour de la prediction et du suivi des retards de vols.

## Objectifs

- Structurer un pipeline de donnees reproductible.
- Entrainer et evaluer des modeles de prediction.
- Versionner les donnees et les artefacts avec DVC.
- Preparer une base claire pour les tests, l'orchestration et le deploiement.

## Arborescence

```text
.
├── artifacts/          # Modeles, metriques et sorties generees
├── data/               # Donnees du projet
│   ├── external/       # Donnees issues de sources externes
│   ├── interim/        # Donnees transformees intermediaires
│   ├── processed/      # Donnees pretes pour l'entrainement
│   └── raw/            # Donnees brutes
├── docs/               # Documentation projet
├── notebooks/          # Analyses exploratoires
├── src/                # Code source Python
│   ├── data/           # Chargement et preparation des donnees
│   ├── features/       # Feature engineering
│   ├── models/         # Entrainement et prediction
│   └── utils/          # Fonctions utilitaires
└── tests/              # Tests automatises
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

## Execution

Les scripts seront ajoutes progressivement dans `src/`.

Exemple attendu a terme :

```bash
python -m src.data.make_dataset
python -m src.models.train_model
```

## Versionnement des donnees

DVC est prevu pour versionner les donnees et artefacts lourds.

```bash
dvc init
dvc add data/raw/<fichier>
git add data/raw/<fichier>.dvc .gitignore
git commit -m "Track raw data with DVC"
```

## Tests

Les tests seront places dans `tests/`.

```bash
pytest
```

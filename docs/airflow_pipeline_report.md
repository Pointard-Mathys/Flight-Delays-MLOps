# Pipeline orchestre avec Airflow

## DAG

Le pipeline est orchestre par le DAG `flight_delay_ml_pipeline`, defini dans `airflow/dags/flight_delay_pipeline_dag.py`.

Ordre des taches :

1. `clean_previous_outputs`
2. `check_raw_data_exists`
3. `force_failure_for_demo`
4. `check_raw_data_format`
5. `prepare_data`
6. `check_processed_data_format`
7. `train_model`
8. `check_model_artifact`
9. `evaluate_model`
10. `check_metrics_artifact`
11. `save_model`

## Controles integres

Le DAG contient plusieurs controles explicites :

- `check_raw_data_exists` verifie que `data/raw/flights.csv` existe et n'est pas vide.
- `check_raw_data_format` verifie les colonnes attendues du fichier brut.
- `check_processed_data_format` verifie que `train.csv` et `test.csv` existent, ne sont pas vides et respectent le schema attendu.
- `check_model_artifact` verifie que `artifacts/model.joblib` existe et n'est pas vide.
- `check_metrics_artifact` verifie que `artifacts/metrics.txt` existe et n'est pas vide.

## Gestion d'echec

La tache `clean_previous_outputs` supprime les sorties d'anciens runs avant de relancer le pipeline. Ainsi, un ancien modele ou un ancien fichier de metriques ne peut pas masquer un echec du run courant.

Airflow gere ensuite les dependances : si une tache echoue, les taches dependantes sont marquees comme non executees/skipped et ne consomment pas des donnees ou artefacts invalides.

Pour produire un run echoue de demonstration sans modifier les donnees, declencher le DAG avec la configuration suivante :

```json
{
  "force_failure": true
}
```

La tache `force_failure_for_demo` echouera volontairement, et les taches suivantes ne seront pas executees.

## Execution

Demarrer Airflow :

```bash
docker compose up airflow
```

Interface Airflow :

- URL : `http://localhost:8080`
- Identifiant : `admin`
- Mot de passe : `admin`

Pour un run reussi, declencher `flight_delay_ml_pipeline` sans configuration particuliere.

Pour un run echoue, declencher `flight_delay_ml_pipeline` avec la configuration JSON de demonstration ci-dessus.

## Captures d'ecran

Run reussi :

![Run Airflow reussi](screenshots/airflow_success_run.png)

Run echoue :

![Run Airflow echoue](screenshots/airflow_failed_run.png)

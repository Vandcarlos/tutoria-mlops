import sys, os
sys.path.append("/opt/airflow")  # garante acesso ao src
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.public_dataset.extract import extract
from src.public_dataset.transform import transform
from src.airflow_tasks.public_dataset.load_task import load_task

DATASET_KAGGLE_HUB = os.getenv("DATASET_KAGGLE_HUB")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

def start_task(**ctx):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run = mlflow.start_run(run_name=f"etl-{ctx['logical_date']}")

    mlflow.set_tags({
        "dag_id": ctx["dag"].dag_id,
        "execution_date": str(ctx["logical_date"]),
        "env": os.getenv("ENV", "local"),
    })

    print("tracking_uri:", mlflow.get_tracking_uri())
    print("artifact_uri:", run.info.artifact_uri)

    return run.info.run_id

with DAG(
    dag_id="etl_public_dataset_dag",
    description="Pipeline ETL de dataset público",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["etl", "public"],
    default_args={"retries": 1}
) as dag:
    t0 = PythonOperator(task_id="start", python_callable=start_task)

    t1 = PythonOperator(task_id="extract",
                        python_callable=extract,
                        op_kwargs={
                            "run_id": "{{ ti.xcom_pull(task_ids='start') }}",
                            "dataset_kaggle_hub": DATASET_KAGGLE_HUB
                        })

    t2 = PythonOperator(task_id="transform",
                        python_callable=transform,
                        op_kwargs={
                            "run_id": "{{ ti.xcom_pull(task_ids='start') }}"
                        })

    t3 = PythonOperator(task_id="load",
                        python_callable=load_task,
                        op_kwargs={
                            "run_id": "{{ ti.xcom_pull(task_ids='start') }}"
                        })

    t0 >> t1 >> t2 >> t3

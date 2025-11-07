from airflow.providers.postgres.hooks.postgres import PostgresHook
from src.public_dataset.load import load

TARGET_TABLE = "dataset_etl"

def load_task(run_id: str):
    hook = PostgresHook(postgres_conn_id="PG_DWH")
    load(run_id=run_id, target_table=TARGET_TABLE, engine_url=hook.get_uri())

import mlflow
import pandas as pd
import src.parquet_utils as parquet_utils

from contextlib import contextmanager

@contextmanager
def use_run(run_id: str | None, experiment="etl-public", run_name=None):
    created_here = False

    if run_id:
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name or "ad-hoc")
        created_here = True

    try:
        yield mlflow
    finally:
        if created_here and mlflow.active_run(): mlflow.end_run()

def log_artifact(df: pd.DataFrame, name: str, artifact_path: str = "data"):
    local_path = parquet_utils.save_data_frame(df=df, name=name)
    mlflow.log_artifact(local_path, artifact_path=artifact_path)
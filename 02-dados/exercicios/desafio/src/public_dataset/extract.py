import kagglehub
import os
import pandas as pd
import mlflow

import src.mlflow_utils as mlflow_utils

def extract(run_id: str | None, dataset_kaggle_hub: str) -> pd.DataFrame:
    with mlflow_utils.use_run(run_id=run_id):
        mlflow.log_param("dataset", dataset_kaggle_hub)

        print(f"🔹 Extract kaggle dataset {dataset_kaggle_hub}")
        dataset_path = kagglehub.dataset_download(dataset_kaggle_hub)
        print(f"dataset path: {dataset_path}", f"dataset files: {os.listdir(dataset_path)}")

        dataset_csv_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
        print(f"dataset csv path: {dataset_csv_path}")

        df_raw = pd.read_csv(dataset_csv_path)
        mlflow_utils.log_artifact(df=df_raw, name="raw")

    return df_raw

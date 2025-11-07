import os
import pandas as pd

AIRFLOW_DATA_DIR = os.getenv("AIRFLOW_DATA_DIR", "../data")

def _build_path(name: str) -> str:
    os.makedirs(AIRFLOW_DATA_DIR, exist_ok=True)  # cria se não existir
    return os.path.join(AIRFLOW_DATA_DIR, f"{name}.parquet")

def save_data_frame(df: pd.DataFrame, name: str) -> str:
    local_path = _build_path(name=name)
    df.to_parquet(local_path, index=False)
    return local_path

def load_data_frame(name: str) -> pd.DataFrame:
    local_path = _build_path(name=name)
    return pd.read_parquet(local_path)

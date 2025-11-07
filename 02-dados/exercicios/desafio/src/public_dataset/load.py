import src.mlflow_utils as mlflow_utils
import src.parquet_utils as parquet_utils
import mlflow

from sqlalchemy import create_engine

def load(run_id: str, target_table: str, engine_url: str, schema: str = "public", chunksize: int = 1000):
    with mlflow_utils.use_run(run_id=run_id):
        engine = create_engine(engine_url)

        df = parquet_utils.load_data_frame(name="clear")

        with engine.begin():
            df.to_sql(
                target_table,
                con=engine,
                schema=schema,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=chunksize,
            )

        mlflow.log_metric("rows_loaded", len(df))

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
from mlflow.tracking import MlflowClient
c = MlflowClient()
for mv in c.search_model_versions('name="Modelo_Risco_Defasagem"'):
    print("version:", mv.version, "run_id:", mv.run_id, "source:", mv.source)

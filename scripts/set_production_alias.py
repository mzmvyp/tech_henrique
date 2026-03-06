"""Define o alias 'production' para a versão do modelo que tem artifacts em mlruns."""
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
from mlflow.tracking import MlflowClient
c = MlflowClient()
# Versão 5 tem source m-dff594087fcb407a9fa2c20cd933c4bd que existe em mlruns
c.set_registered_model_alias("Modelo_Risco_Defasagem", "production", "5")
print("Alias 'production' definido para versão 5.")

from mlflow.pyfunc import PyFuncModel
from utils.tfrecord_utils import load_tfrecord_dataset
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
import numpy as np
import pandas as pd


def load_mlflow_model(run_id: str, tracking_uri: str = "https://dagshub.com/felytz/waveled.mlflow"):
    """
    Carga un modelo almacenado en MLflow usando su run_id desde DagsHub.

    Parámetros:
    - run_id (str): ID del run de MLflow.
    - tracking_uri (str): URI del servidor de seguimiento (por defecto: DagsHub).

    Retorna:
    - model: modelo cargado como objeto PyFunc.
    """
    # Establece la URI del servidor de tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Verifica si el run existe
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except mlflow.exceptions.MlflowException as e:
        raise ValueError(f"No se encontró el run con ID '{run_id}': {e}")

    # Ruta al modelo 
    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo desde {model_uri}: {e}")

    return model






def predict_music_mlflow(model_uri, tfrecord_path, seq_length=10, batch_size=1):
    """
    Extrae embeddings de audio desde TFRecord y usa un modelo MLflow para predecir.
    
    Args:
        model_uri (str): URI del modelo MLflow
        tfrecord_path (str): Ruta al archivo TFRecord
        seq_length (int): Longitud máxima de la secuencia
        batch_size (int): Tamaño de batch para la predicción

    Returns:
        list: Predicciones del modelo para cada ejemplo
    """
    model = mlflow.pyfunc.load_model(model_uri)
    dataset = load_tfrecord_dataset(tfrecord_path, seq_length=seq_length, batch_size=batch_size)

    predictions = []
    for batch_embeddings in dataset:
        # batch_embeddings: shape (batch_size, seq_length, 128)
        batch_np = batch_embeddings.numpy().reshape(batch_embeddings.shape[0], -1)  # flatten

        try:
            preds = model.predict(batch_np)
        except Exception:
            preds = model.predict(pd.DataFrame(batch_np))

        predictions.extend(preds)

    return predictions

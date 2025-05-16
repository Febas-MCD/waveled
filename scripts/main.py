from utils.inference_utils import predict_music_mlflow

preds = predict_music_mlflow(
    model_uri="runs:/e0f0e4cad21b49c1b5624580b5cc9460/model",
    tfrecord_path="C:/Users/Sebastian/Documents/GitHub/waveled/data/processed/output/salida.tfrecord",
    seq_length=10,
    batch_size=4
)

print(preds)

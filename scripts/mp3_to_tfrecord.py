import os
import sys
import librosa
import numpy as np
import tensorflow as tf

def wav_to_mel_embedding(wav_path, sr=22050, n_mels=128, seq_length=10, hop_length=512):
    """
    Lee un archivo wav/mp3 y extrae un mel espectrograma normalizado y recortado/paddeado.
    Retorna un ndarray de shape [seq_length, n_mels], dtype uint8.
    """
    # Cargar audio
    audio, _ = librosa.load(wav_path, sr=sr)
    
    # Extraer mel espectrograma
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convertir a dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalizar entre 0 y 255 (uint8)
    mel_norm = 255 * (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_uint8 = mel_norm.astype(np.uint8).T  # Transponer para tener [frames, n_mels]

    # Recortar o paddear a seq_length
    if mel_uint8.shape[0] > seq_length:
        mel_uint8 = mel_uint8[:seq_length]
    else:
        pad_width = seq_length - mel_uint8.shape[0]
        mel_uint8 = np.pad(mel_uint8, ((0, pad_width), (0,0)), mode='constant', constant_values=0)
    
    return mel_uint8





def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in values])

def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[v])) for v in values])





def create_tfrecord(wav_path, tfrecord_path, video_id="audio_0", labels=[1], seq_length=10):
    """
    Crea un TFRecord en formato SequenceExample con el mel espectrograma extraído.
    
    Args:
        wav_path: ruta al archivo wav/mp3
        tfrecord_path: ruta donde guardar el TFRecord
        video_id: string ID del audio
        labels: lista de ints con etiquetas (ejemplo: [1] para música)
        seq_length: longitud fija de secuencia
    
    El TFRecord tendrá:
    - Contexto con "video_id" (string) y "labels" (int64 list)
    - Secuencia "audio_embedding" con seq_length vectores uint8 de tamaño 128 bytes cada uno.
    """
    mel_uint8 = wav_to_mel_embedding(wav_path, seq_length=seq_length)  # shape (seq_length, 128), uint8

    # Convertir cada vector de 128 uint8 a bytes
    audio_embedding_bytes = [mel_uint8[i].tobytes() for i in range(seq_length)]

    context = tf.train.Features(feature={
        "video_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id.encode('utf-8')])),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        "audio_embedding": tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[b])) for b in audio_embedding_bytes
        ])
    })

    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        writer.write(example.SerializeToString())
    print(f"TFRecord creado en {tfrecord_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python mp3_a_tfrecord.py entrada.mp3 salida.tfrecord")
        sys.exit(1)
    
    entrada = sys.argv[1]
    salida = sys.argv[2]

    create_tfrecord(entrada, salida, video_id=os.path.basename(entrada), labels=[1], seq_length=10)


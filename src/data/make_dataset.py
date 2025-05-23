import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from src.paths import *

def load_audio_set_data():
    """Load and process AudioSet ontology and labels"""
    # Load ontology
    with open(os.path.join(RAW_DIR, 'ontology.json'), "r", encoding="utf-8") as f:
        data = json.load(f)
    df_ontology = pd.DataFrame(data)

    # Create music labels
    keywords_column_name = ["music", "musical", "song", "instrument", "singing"]
    keywords_column_description = ["music", "musical", "song", "singing"]

    pattern_column_name = "|".join(keywords_column_name)
    pattern_column_description = "|".join(keywords_column_description)

    name_contains = df_ontology["name"].str.lower().str.contains(pattern_column_name)
    description_contains = df_ontology["description"].str.lower().str.contains(pattern_column_description, na=False)

    df_ontology["is_music"] = (name_contains | description_contains).astype(int)

    # Load class labels
    df_class_labels_indices = pd.read_csv(os.path.join(RAW_DIR, 'class_labels_indices.csv'))

    # Merge ontology and labels
    df_ontology_labels = pd.merge(
        df_class_labels_indices, 
        df_ontology, 
        left_on='mid', 
        right_on='id', 
        how='left'
    )
    df_ontology_labels = df_ontology_labels.drop(columns=['mid', 'display_name'])
    id_labels_dict = df_ontology_labels.set_index('index')['id'].to_dict()
    df_ontology_labels.set_index('index', inplace=True)

    # Get music IDs
    music_ids = set(df_ontology_labels[df_ontology_labels["is_music"] == 1]["id"].astype(str))
    
    return music_ids, id_labels_dict

@tf.function(reduce_retracing=True)
def parse_music_example(example_proto, music_ids, id_labels_dict, seq_length=10):
    """Parse TFRecord example."""
    # Convert dict to constant tensor
    id_labels_tensor = tf.constant(list(id_labels_dict.values()))

    # Define features
    context_features = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64)
    }
    sequence_features = {
        "audio_embedding": tf.io.FixedLenSequenceFeature([], tf.string)
    }

    # Parse the example
    context, sequences = tf.io.parse_single_sequence_example(
        example_proto, 
        context_features=context_features, 
        sequence_features=sequence_features
    )

    # Process audio embeddings
    audio_embeddings = tf.io.decode_raw(sequences['audio_embedding'], tf.uint8)
    audio_embeddings = tf.reshape(audio_embeddings, [-1, 128])
    audio_embeddings = (tf.cast(audio_embeddings, tf.float32) - 127.5) / 127.5
    audio_embeddings = audio_embeddings[:seq_length]
    padding = [[0, seq_length - tf.shape(audio_embeddings)[0]], [0, 0]]
    audio_embeddings = tf.pad(audio_embeddings, padding)
    audio_embeddings.set_shape([seq_length, 128])

    # Process labels
    labels = tf.sparse.to_dense(context['labels'])
    id_labels = tf.gather(id_labels_tensor, labels)
    
    # Check if any label matches music_ids
    is_music = tf.reduce_any(tf.equal(tf.expand_dims(id_labels, -1), music_ids))
    
    return audio_embeddings, tf.cast(is_music, tf.float32)

def create_dataset(tfrecord_dir, music_ids, id_labels_dict, batch_size=32, seq_length=10, is_training=True):
    """Create TF dataset pipeline."""
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
        
    # Convert music_ids to tensor
    music_ids_tensor = tf.constant([str(id) for id in music_ids], dtype=tf.string)
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    
    # Parse examples
    parse_fn = lambda x: parse_music_example(x, music_ids_tensor, id_labels_dict, seq_length)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def dataset_to_numpy(dataset):
    """Convert TF Dataset to numpy arrays"""
    X, y = [], []
    for audio_emb, label in dataset.unbatch():
        X.append(audio_emb.numpy().flatten())  # Flatten to [seq_length * 128]
        y.append(label.numpy())
    return np.array(X), np.array(y)

def enhance_linear_svc(model):
    """Verifies if LinearSVC is already calibrated, wraps if needed"""
    if isinstance(model, LinearSVC):
        print("⚠️ Found uncalibrated LinearSVC - adding calibration (this may not work well)")
        return CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    return model
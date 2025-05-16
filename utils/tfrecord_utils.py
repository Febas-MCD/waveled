import tensorflow as tf

def parse_audio_embedding_only(example_proto, seq_length=10):
    sequence_features = {
        "audio_embedding": tf.io.FixedLenSequenceFeature([], tf.string)
    }
    context, sequences = tf.io.parse_single_sequence_example(
        example_proto, context_features={}, sequence_features=sequence_features
    )
    
    audio_embeddings = tf.io.decode_raw(sequences['audio_embedding'], tf.uint8)
    audio_embeddings = tf.reshape(audio_embeddings, [-1, 128])
    audio_embeddings = (tf.cast(audio_embeddings, tf.float32) - 127.5) / 127.5
    audio_embeddings = audio_embeddings[:seq_length]
    padding = [[0, seq_length - tf.shape(audio_embeddings)[0]], [0, 0]]
    audio_embeddings = tf.pad(audio_embeddings, padding)
    audio_embeddings.set_shape([seq_length, 128])
    
    return audio_embeddings


def load_tfrecord_dataset(tfrecord_path, seq_length=10, batch_size=1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(lambda x: parse_audio_embedding_only(x, seq_length))
    dataset = parsed_dataset.batch(batch_size)
    return dataset


import librosa
import resampy
import numpy as np
import math
import tensorflow as tf
from src.paths import *
import tensorflow_hub as hub

# Load VGGish
try:
    vggish_model = hub.load(VGGISH_MODEL_URL)
    print("✅ VGGish model loaded successfully")
except Exception as e:
    print(f"❌ Error loading VGGish model: {str(e)}")
    vggish_model = None

def load_audio(mp3_path):
    """Load audio file using librosa"""
    try:
        y, sr = librosa.load(mp3_path, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            y = resampy.resample(y, sr, SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=SAMPLE_RATE)
        return y, duration
    except Exception as e:
        print(f"❌ Error loading audio file {mp3_path}: {str(e)}")
        return None, 0

def split_audio(audio_samples, segment_duration=SEGMENT_DURATION):
    """Split audio into segments"""
    samples_per_segment = int(SAMPLE_RATE * segment_duration)
    num_segments = math.ceil(len(audio_samples) / samples_per_segment)
    
    segments = []
    for i in range(int(num_segments)):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio_samples[start:end]
        
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)), 
                            mode='constant')
        
        segments.append(segment)
    
    return segments, num_segments

def extract_vggish_embeddings(audio_segment):
    """Extract VGGish embeddings from audio segment"""
    try:
        audio_tensor = tf.convert_to_tensor(audio_segment, dtype=tf.float32)
        embeddings = vggish_model(audio_tensor).numpy()
        
        if embeddings.shape[0] < NUM_SEGMENTS_PER_CHUNK:
            pad_shape = [(0, NUM_SEGMENTS_PER_CHUNK - embeddings.shape[0]), (0, 0)]
            embeddings = np.pad(embeddings, pad_shape, mode='constant')
        else:
            embeddings = embeddings[:NUM_SEGMENTS_PER_CHUNK]
        
        embeddings = (embeddings - 127.5) / 127.5
        return embeddings
    except Exception as e:
        print(f"❌ Error extracting embeddings: {str(e)}")
        return np.zeros((NUM_SEGMENTS_PER_CHUNK, EMBEDDING_SIZE))

def predict_with_model(model, features):
    """Make predictions adapting to each model's capabilities"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
            return pred, proba
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(features)[0]
            pred = model.predict(features)[0]
            proba = 1 / (1 + np.exp(-decision))
            return pred, proba
        else:
            pred = model.predict(features)[0]
            return pred, 1.0 if pred == 1 else 0.0
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return 0, 0.0
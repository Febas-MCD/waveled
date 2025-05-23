import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Data subdirectories
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')

# Model objects directory
MODEL_OBJECTS_DIR = os.path.join(MODELS_DIR, 'objects')
os.makedirs(MODEL_OBJECTS_DIR, exist_ok=True)

# Test files directory
TEST_FILES_DIR = os.path.join(BASE_DIR, 'to_test')
os.makedirs(TEST_FILES_DIR, exist_ok=True)

# Results file
RESULTS_FILE = os.path.join(REPORTS_DIR, 'resultados.csv')

# AudioSet specific paths
def get_audio_set_paths(split):
    """Get paths for AudioSet files based on split"""
    splits = {
        'bal_train': 'balanced_train_segments.csv',
        'unbal_train': 'unbalanced_train_segments.csv',
        'eval': 'eval_segments.csv'
    }
    
    return {
        'raw': os.path.join(RAW_DIR, split, splits[split]),
        'interim': os.path.join(INTERIM_DIR, split, splits[split]),
        'processed': os.path.join(PROCESSED_DIR, split, splits[split]),
        'tfrecord_dir': os.path.join(RAW_DIR, split)
    }

# Other constants
SAMPLE_RATE = 16000  # Hz
SEGMENT_DURATION = 10.0  # seconds
VGGISH_HOP_SIZE = 0.96  # seconds
EMBEDDING_SIZE = 128
NUM_SEGMENTS_PER_CHUNK = int(SEGMENT_DURATION / VGGISH_HOP_SIZE)
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
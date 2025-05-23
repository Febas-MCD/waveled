import os
import joblib
import librosa
import resampy
import numpy as np
import pandas as pd
from src.paths import *
from src.models.predict_utils import *
import tensorflow_hub as hub

def load_local_models():
    """Load locally saved models from the new directory structure"""
    models = {}
    # List all model directories (LinearSVC_PCA, LogisticRegression_PCA, etc.)
    for model_dir in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir)
        if os.path.isdir(model_path):
            # Look for model.joblib in each model directory
            model_file = os.path.join(model_path, 'model.joblib')
            if os.path.exists(model_file):
                models[model_dir] = joblib.load(model_file)
    
    # Load preprocessing objects from the objects directory
    scaler = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'pca.joblib'))
    scaler_pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler_pca.joblib'))
    
    return models, scaler, pca, scaler_pca

def predict_audio_files(models, scaler, pca, scaler_pca):
    """Predict music content for all files in test directory"""
    results = []
    
    for audio_file in os.listdir(TEST_FILES_DIR):
        if not audio_file.lower().endswith(('.mp3', '.wav')):
            continue
            
        file_path = os.path.join(TEST_FILES_DIR, audio_file)
        predictions = {
            'archivo': audio_file,
            'prediccion total': 'Not music'  # Default value
        }
        
        try:
            # Load and process audio
            y, sr = librosa.load(file_path, sr=None, mono=True)
            if sr != SAMPLE_RATE:
                y = resampy.resample(y, sr, SAMPLE_RATE)
            
            segments, num_segments = split_audio(y)
            
            # Initialize model tracking
            model_stats = {model_name: {'probas': [], 'preds': []} 
                         for model_name in models.keys()}
            
            # Process each segment
            for segment in segments:
                embeddings = extract_vggish_embeddings(segment)
                flattened = embeddings.flatten().reshape(1, -1)
                scaled = scaler.transform(flattened)
                
                for model_name, model in models.items():
                    use_pca = 'PCA' in model_name
                    features = scaler_pca.transform(pca.transform(scaled)) if use_pca else scaled
                    
                    pred, proba = predict_with_model(model, features)
                    model_stats[model_name]['preds'].append(pred)
                    model_stats[model_name]['probas'].append(proba)
            
            # Calculate statistics for each model
            for model_name in models.keys():
                # Calculate mean probability
                mean_proba = np.mean(model_stats[model_name]['probas']) * 100
                predictions[f'{model_name} % music'] = f"{mean_proba:.1f}%"
                
                # Calculate percentage of segments classified as music
                music_segments = sum(model_stats[model_name]['preds'])
                segment_percentage = (music_segments / num_segments) * 100
                predictions[f'{model_name} % segments'] = f"{segment_percentage:.1f}%"
            
            # Determine final prediction
            total_music_segments = sum(
                sum(stats['preds']) for stats in model_stats.values()
            )
            total_segments = num_segments * len(models)
            
            if (total_music_segments / total_segments) > 0.5:
                predictions['prediccion total'] = 'Music'
            
            results.append(predictions)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            results.append(predictions)  # Add with default values
    
    # Save results
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols = ['archivo', 'prediccion total']
    model_names = list(models.keys())
    for model in model_names:
        cols.extend([f'{model} % music', f'{model} % segments'])
    
    df = df[cols]
    df.to_csv(RESULTS_FILE, index=False)
    return df

if __name__ == "__main__":
    models, scaler, pca, scaler_pca = load_local_models()
    results = predict_audio_files(models, scaler, pca, scaler_pca)
    print(results)
import os
import mlflow
import dagshub
import joblib
from mlflow.tracking import MlflowClient
from src.paths import *
from src.models.predict_model import predict_audio_files
import tensorflow as tf
import tensorflow_hub as hub

# Initialize VGGish model
vggish_model = hub.load(VGGISH_MODEL_URL)

def load_latest_models():
    """Load latest models from DagsHub with proper error handling"""
    try:
        # Initialize DagsHub connection
        dagshub.init(repo_owner='Febas-MCD', repo_name='waveled', mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/Febas-MCD/waveled.mlflow")
        
        client = MlflowClient()
        models = {}
        
        # Get all registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("⚠️ No registered models found in the model registry")
            return None, None, None, None

        for model in registered_models:
            try:
                # Get the latest version (using new alias system instead of stages)
                latest_version = client.get_model_version(
                    name=model.name,
                    version=model.latest_versions[0].version
                )
                
                # Load model using run_id instead of stage/alias
                model_uri = f"models:/{model.name}/{latest_version.version}"
                
                if model.name == 'XGBoost':
                    models[model.name] = mlflow.xgboost.load_model(model_uri)
                else:
                    models[model.name] = mlflow.sklearn.load_model(model_uri)
                
                print(f"✅ Successfully loaded model: {model.name}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model.name}: {str(e)}")
                continue

        # Load preprocessing objects from local objects directory
        try:
            scaler = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler.joblib'))
            pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'pca.joblib'))
            scaler_pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler_pca.joblib'))
            print("✅ Successfully loaded preprocessing objects")
            return models, scaler, pca, scaler_pca
            
        except Exception as e:
            print(f"❌ Error loading preprocessing objects from {MODEL_OBJECTS_DIR}: {str(e)}")
            return None, None, None, None

    except Exception as e:
        print(f"⚠️ Failed to initialize DagsHub connection: {str(e)}")
        return None, None, None, None

def load_local_models():
    """Load models and preprocessing objects from local directory structure"""
    models = {}
    model_dirs = {
        'LinearSVC_PCA': os.path.join(MODELS_DIR, 'LinearSVC_PCA', 'model.joblib'),
        'LogisticRegression_PCA': os.path.join(MODELS_DIR, 'LogisticRegression_PCA', 'model.joblib'),
        'RandomForest': os.path.join(MODELS_DIR, 'RandomForest', 'model.joblib'),
        'XGBoost': os.path.join(MODELS_DIR, 'XGBoost', 'model.joblib')
    }
    
    try:
        for model_name, model_path in model_dirs.items():
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"✅ Successfully loaded local model: {model_name}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
        
        # Load preprocessing objects
        scaler = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler.joblib'))
        pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'pca.joblib'))
        scaler_pca = joblib.load(os.path.join(MODEL_OBJECTS_DIR, 'scaler_pca.joblib'))
        print("✅ Successfully loaded local preprocessing objects")
        return models, scaler, pca, scaler_pca
        
    except Exception as e:
        print(f"❌ Error loading local models: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    # First try to load from DagsHub
    print("Attempting to load models from DagsHub...")
    models, scaler, pca, scaler_pca = load_latest_models()
    
    # Fallback to local models if DagsHub fails
    if not models:
        print("\nFalling back to local models...")
        models, scaler, pca, scaler_pca = load_local_models()
    
    if models:
        print("\nSuccessfully loaded models. Starting predictions...")
        results = predict_audio_files(models, scaler, pca, scaler_pca)
        print(results)
    else:
        print("\nFailed to load any models. Please check your configuration.")
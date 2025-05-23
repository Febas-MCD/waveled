import os
import mlflow
import dagshub
import joblib
from mlflow.tracking import MlflowClient
from src.paths import MODELS_DIR

def upload_models_to_dagshub():
    """Upload locally trained models to DagsHub"""
    try:
        dagshub.init(repo_owner='Febas-MCD', repo_name='waveled', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/Febas-MCD/waveled.mlflow")
    except Exception as e:
        print(f"⚠️ Failed to connect to DagsHub: {str(e)}")
        return
    
    client = MlflowClient()
    
    # Upload each model from its subdirectory
    for model_dir in os.listdir(MODELS_DIR):
        model_subdir = os.path.join(MODELS_DIR, model_dir)
        if os.path.isdir(model_subdir):
            model_path = os.path.join(model_subdir, "model.joblib")
            
            if os.path.exists(model_path):
                model_name = model_dir  # Use directory name as model name
                model = joblib.load(model_path)
                
                with mlflow.start_run(run_name=f"{model_name}_upload"):
                    if "XGBoost" in model_name:
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path=model_name,
                            registered_model_name=model_name
                        )
                    else:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=model_name,
                            registered_model_name=model_name
                        )
                    
                    # Set as champion
                    latest_version = client.get_latest_versions(model_name)[0].version
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="Champion",
                        version=latest_version
                    )
                    
                    print(f"✅ Successfully uploaded {model_name}")
            else:
                print(f"⚠️ No model.joblib found in {model_subdir}")

if __name__ == "__main__":
    upload_models_to_dagshub()
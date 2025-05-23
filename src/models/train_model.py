import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from src.paths import *
from src.data.make_dataset import load_audio_set_data, create_dataset, dataset_to_numpy
from src.features.build_features import create_preprocessing_pipeline
from src.models.train_utils import enhance_linear_svc, train_model_with_gridsearch

def get_model_configurations(y_train):
    """Return configurations for all models to train"""
    return [
        {
            "model": LogisticRegression(class_weight='balanced', random_state=10, max_iter=1000),
            "name": "LogisticRegression_PCA",
            "params": {
                'C': [0.01, 0.1, 1, 10], 
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            "use_pca": True,
            "calibrate": False
        },
        {
            "model": LinearSVC(class_weight='balanced', random_state=10, dual=False, max_iter=10000),
            "name": "LinearSVC_PCA",
            "params": {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'loss': ['squared_hinge']
            },
            "use_pca": True,
            "calibrate": True 
        },
        {
            "model": RandomForestClassifier(class_weight='balanced', random_state=10),
            "name": "RandomForest",
            "params": {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            "use_pca": False,
            "calibrate": False
        },
        {
            "model": XGBClassifier(
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                random_state=10,
                eval_metric='logloss'
            ),
            "name": "XGBoost",
            "params": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6],
                'subsample': [0.8, 1.0],
            },
            "use_pca": False,
            "calibrate": False
        }
    ]

def evaluate_model(model, X_val, y_val):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_val)
    return {
        "f1_score": f1_score(y_val, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_val, y_pred)
    }

def save_model_artifacts(model, model_name, metrics, params, use_pca):
    """Save all model artifacts locally"""
    # Create model directory
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "metrics": metrics,
        "parameters": params,
        "features": "PCA" if use_pca else "Original"
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model artifacts saved to {model_dir}")

def train_and_save_model(model_config, X_train, y_train, X_val, y_val, X_val_pca, X_val_scaled):
    """Train and save a single model locally"""
    model_name = model_config["name"]
    print(f"\n🚀 Training {model_name}...")
    
    # Train model
    best_model, best_params = train_model_with_gridsearch(
        model=model_config["model"],
        params=model_config["params"],
        X_train=X_train,
        y_train=y_train,
        calibrate=model_config["calibrate"]
    )
    
    # Special handling for XGBoost
    if model_name == "XGBoost":
        xgb_final = XGBClassifier(
            **best_params,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=10,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_model = xgb_final
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    
    # Select appropriate validation set
    val_data = X_val_pca if model_config["use_pca"] else X_val_scaled
    
    # Evaluate
    metrics = evaluate_model(best_model, val_data, y_val)
    
    # Save
    save_model_artifacts(
        model=best_model,
        model_name=model_name,
        metrics=metrics,
        params=best_params,
        use_pca=model_config["use_pca"]
    )
    
    print(f"✅ {model_name} - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return best_model

def main():
    # Load and prepare data
    print("📊 Loading and preparing data...")
    music_ids, id_labels_dict = load_audio_set_data()
    full_dataset = create_dataset(
        tfrecord_dir=get_audio_set_paths('bal_train')['tfrecord_dir'],
        music_ids=music_ids,
        batch_size=32,
        id_labels_dict=id_labels_dict
    )
    
    # Split dataset
    dataset_size = sum(1 for _ in full_dataset)
    val_size = int(0.2 * dataset_size)
    train_ds = full_dataset.skip(val_size)
    val_ds = full_dataset.take(val_size)
    
    # Convert to numpy arrays
    X_train, y_train = dataset_to_numpy(train_ds)
    X_val, y_val = dataset_to_numpy(val_ds)
    
    # Create preprocessing pipeline
    print("⚙️ Creating preprocessing pipeline...")
    scaler, pca, scaler_pca = create_preprocessing_pipeline(X_train)
    
    # Apply preprocessing
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_train_pca = scaler_pca.transform(X_train_pca)
    X_val_pca = scaler_pca.transform(X_val_pca)
    
    # Get model configurations
    model_configs = get_model_configurations(y_train)
    
    # Train models
    trained_models = {}
    for config in model_configs:
        # Select appropriate training data
        train_data = X_train_pca if config["use_pca"] else X_train_scaled
        
        trained_model = train_and_save_model(
            model_config=config,
            X_train=train_data,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_val_pca=X_val_pca,
            X_val_scaled=X_val_scaled
        )
        trained_models[config["name"]] = trained_model
    
    print("\n🎉 All models trained and saved locally!")
    print(f"📁 Models saved in: {MODELS_DIR}")
    return trained_models

if __name__ == "__main__":
    main()
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.paths import MODEL_OBJECTS_DIR
import joblib

def create_preprocessing_pipeline(X_train):
    """Create and save preprocessing pipeline"""
    # Normalization
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Normalization for PCA space
    scaler_pca = StandardScaler().fit(X_train_pca)
    X_train_pca = scaler_pca.transform(X_train_pca)
    
    # Save preprocessing objects
    joblib.dump(scaler, os.path.join(MODEL_OBJECTS_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(MODEL_OBJECTS_DIR, 'pca.joblib'))
    joblib.dump(scaler_pca, os.path.join(MODEL_OBJECTS_DIR, 'scaler_pca.joblib'))
    
    return scaler, pca, scaler_pca

def apply_preprocessing(X, scaler, pca=None, scaler_pca=None):
    """Apply preprocessing pipeline"""
    scaled = scaler.transform(X)
    
    if pca and scaler_pca:
        scaled = pca.transform(scaled)
        scaled = scaler_pca.transform(scaled)
    
    return scaled
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import warnings

def enhance_linear_svc(model):
    """Verifies if LinearSVC is already calibrated, wraps if needed"""
    if isinstance(model, LinearSVC):
        print("⚠️ Found uncalibrated LinearSVC - adding calibration (this may not work well)")
        return CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    return model

def train_model_with_gridsearch(model, params, X_train, y_train, calibrate=False):
    """Train model with GridSearchCV"""
    if calibrate:
        print("⚙️ Setting up calibration pipeline...")
        # First find best LinearSVC parameters
        svc_grid = GridSearchCV(
            model,
            params,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        svc_grid.fit(X_train, y_train)
        
        # Then calibrate the best model
        calibrated_model = CalibratedClassifierCV(
            svc_grid.best_estimator_,
            method='sigmoid',
            cv='prefit'
        )
        calibrated_model.fit(X_train, y_train)
        return calibrated_model, svc_grid.best_params_
    else:
        # Standard GridSearchCV for other models
        grid = GridSearchCV(
            model,
            params,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_
"""
Script 05: Model Training

This script trains the ordinal classification model:
- Uses LightGBM for multi-class ordinal classification
- Implements class weighting for imbalanced data
- Performs cross-validation
- Includes hyperparameter tuning with Optuna
- Saves the trained model

Ordinal Classification Approach:
Since fire size classes have a natural order (Small < Medium < Large),
we use ordinal-aware training with cumulative link model concepts.

Usage:
    python scripts/05_train_model.py [--tune]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    TRAIN_PARQUET,
    TEST_PARQUET,
    MODELS_DIR,
    TARGET_COLUMN,
    TARGET_CLASS_NAMES,
    LIGHTGBM_PARAMS,
    OPTUNA_SEARCH_SPACE,
    N_OPTUNA_TRIALS,
    N_FOLDS,
    RANDOM_STATE,
    USE_CLASS_WEIGHTS,
    PRIMARY_METRIC
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test: {len(test_df):,} rows")
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for modeling."""
    exclude_cols = [
        TARGET_COLUMN,
        'NWCG_REPORTING_AGENCY', 'STAT_CAUSE_DESCR', 'STATE', 'OWNER_DESCR',
        'COUNTY'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Prepare features and targets for training."""
    print("\nPreparing data...")
    
    feature_cols = get_feature_columns(train_df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COLUMN].values
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature columns: {feature_cols}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def compute_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute sample weights for class imbalance."""
    print("\nComputing class weights...")
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    
    weight_dict = dict(zip(classes, class_weights))
    print(f"  Class weights: {weight_dict}")
    
    # Create sample weights array
    sample_weights = np.array([weight_dict[y] for y in y_train])
    
    return sample_weights


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Evaluate model predictions."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred, weights='linear')  # Linear weights for ordinal
    }
    
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics


def cross_validate(X: np.ndarray, y: np.ndarray, params: dict, 
                   sample_weights: np.ndarray = None) -> tuple[float, float]:
    """Perform cross-validation and return mean and std of primary metric."""
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        if sample_weights is not None:
            weights_fold = sample_weights[train_idx]
        else:
            weights_fold = None
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train, weight=weights_fold)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 500),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict
        y_pred = model.predict(X_fold_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # Score
        score = f1_score(y_fold_val, y_pred_class, average='macro')
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, 
              sample_weights: np.ndarray) -> float:
    """Optuna objective function for hyperparameter tuning."""
    
    params = LIGHTGBM_PARAMS.copy()
    
    # Sample hyperparameters
    params['n_estimators'] = trial.suggest_int('n_estimators', *OPTUNA_SEARCH_SPACE['n_estimators'])
    params['max_depth'] = trial.suggest_int('max_depth', *OPTUNA_SEARCH_SPACE['max_depth'])
    params['learning_rate'] = trial.suggest_float('learning_rate', *OPTUNA_SEARCH_SPACE['learning_rate'], log=True)
    params['num_leaves'] = trial.suggest_int('num_leaves', *OPTUNA_SEARCH_SPACE['num_leaves'])
    params['min_child_samples'] = trial.suggest_int('min_child_samples', *OPTUNA_SEARCH_SPACE['min_child_samples'])
    params['subsample'] = trial.suggest_float('subsample', *OPTUNA_SEARCH_SPACE['subsample'])
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', *OPTUNA_SEARCH_SPACE['colsample_bytree'])
    params['reg_alpha'] = trial.suggest_float('reg_alpha', *OPTUNA_SEARCH_SPACE['reg_alpha'])
    params['reg_lambda'] = trial.suggest_float('reg_lambda', *OPTUNA_SEARCH_SPACE['reg_lambda'])
    
    # Cross-validate
    mean_score, _ = cross_validate(X, y, params, sample_weights)
    
    return mean_score


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, 
                         sample_weights: np.ndarray) -> dict:
    """Tune hyperparameters using Optuna."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    print(f"\nRunning {N_OPTUNA_TRIALS} Optuna trials...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, sample_weights),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True
    )
    
    print(f"\nBest trial:")
    print(f"  Value (macro F1): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    # Merge best params with base params
    best_params = LIGHTGBM_PARAMS.copy()
    best_params.update(study.best_trial.params)
    
    best_params_path = MODELS_DIR / 'best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f)
    print(f"  Best params saved: {best_params_path}")

    return best_params


def train_final_model(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      params: dict, sample_weights: np.ndarray,
                      feature_names: list) -> lgb.Booster:
    """Train final model on full training data."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    # Split training data for validation to avoid using test set
    print("Splitting training data for validation (10%)...")
    if sample_weights is not None:
        X_train_split, X_val_split, y_train_split, y_val_split, weights_train, weights_val = train_test_split(
            X_train, y_train, sample_weights,
            test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
        )
        weights_train = None
        weights_val = None
    
    # Create datasets
    train_data = lgb.Dataset(X_train_split, label=y_train_split, weight=weights_train,
                             feature_name=feature_names)
    val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data, weight=weights_val,
                           feature_name=feature_names)
    
    # Train
    print("\nTraining...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params.get('n_estimators', 2000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluate
    print("\n" + "-"*40)
    
    # Train predictions (on the split used for training)
    y_train_pred = np.argmax(model.predict(X_train_split), axis=1)
    evaluate_model(y_train_split, y_train_pred, "Train (Split)")
    
    # Test predictions (on the held-out test set)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")
    
    # Classification report
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=TARGET_CLASS_NAMES))
    
    return model, test_metrics


def save_model(model: lgb.Booster, params: dict, feature_names: list, metrics: dict) -> None:
    """Save trained model and metadata."""
    print("\nSaving model...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save LightGBM model
    model_path = MODELS_DIR / 'wildfire_model.txt'
    model.save_model(str(model_path))
    print(f"  Model: {model_path}")
    
    # Save metadata
    metadata = {
        'params': params,
        'feature_names': feature_names,
        'metrics': metrics,
        'target_classes': TARGET_CLASS_NAMES
    }
    metadata_path = MODELS_DIR / 'model_metadata.joblib'
    joblib.dump(metadata, metadata_path)
    print(f"  Metadata: {metadata_path}")


def main():
    """Main training pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train wildfire classification model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, feature_cols = prepare_data(train_df, test_df)
    
    # Compute class weights
    sample_weights = None
    if USE_CLASS_WEIGHTS:
        sample_weights = compute_weights(y_train)
    
    # Get parameters
    if args.tune:
        params = tune_hyperparameters(X_train, y_train, sample_weights)
    else:
        best_params_path = MODELS_DIR / 'best_params.json'
        if best_params_path.exists():
            # Load saved best params
            with open(best_params_path, 'r') as f:
                tuned_params = json.load(f)
            params = LIGHTGBM_PARAMS.copy()
            params.update(tuned_params)
            print(f"Loaded best params from {best_params_path}")
        else:
            # Fallback to defaults
            params = LIGHTGBM_PARAMS.copy()
            params['n_estimators'] = 500
            params['max_depth'] = 8
            params['learning_rate'] = 0.05
            params['num_leaves'] = 64
            params['min_child_samples'] = 50
            params['subsample'] = 0.8
            params['colsample_bytree'] = 0.8
            print("No saved params found; using defaults")
    
    # Train final model
    model, metrics = train_final_model(
        X_train, y_train, X_test, y_test,
        params, sample_weights, feature_cols
    )
    
    # Save model
    save_model(model, params, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

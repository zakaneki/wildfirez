"""
Script 06: Model Evaluation

This script performs comprehensive evaluation of the trained model:
- Confusion matrix visualization
- Per-class metrics analysis
- Ordinal-specific metrics (linear weighted kappa)
- SHAP feature importance analysis
- Error analysis

Usage:
    python scripts/06_evaluate.py
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    TEST_PARQUET,
    TRAIN_PARQUET,
    MODELS_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    TARGET_CLASS_NAMES
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_model_and_data() -> tuple:
    """Load trained model, metadata, and test data."""
    print("Loading model and data...")
    
    # Load model
    model_path = MODELS_DIR / 'wildfire_model.txt'
    model = lgb.Booster(model_file=str(model_path))
    print(f"  Model: {model_path}")
    
    # Load metadata
    metadata_path = MODELS_DIR / 'model_metadata.joblib'
    metadata = joblib.load(metadata_path)
    print(f"  Metadata: {metadata_path}")
    
    # Load test data
    test_df = pd.read_parquet(TEST_PARQUET)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"  Test data: {len(test_df):,} rows")
    
    return model, metadata, train_df, test_df


def prepare_data(df: pd.DataFrame, feature_names: list) -> tuple:
    """Prepare features and target from dataframe."""
    X = df[feature_names].values
    y = df[TARGET_COLUMN].values
    return X, y


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute comprehensive metrics."""
    
    metrics = {
        # Standard metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
        
        # Ordinal-specific: Linear weighted Cohen's Kappa
        # Penalizes predictions farther from true class
        'cohen_kappa_linear': cohen_kappa_score(y_true, y_pred, weights='linear'),
        'cohen_kappa_quadratic': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        
        # Per-class metrics
        'per_class_precision': precision_score(y_true, y_pred, average=None),
        'per_class_recall': recall_score(y_true, y_pred, average=None),
        'per_class_f1': f1_score(y_true, y_pred, average=None)
    }
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
    
    print("\nOrdinal Metrics (penalize distance from true class):")
    print(f"  Cohen's Kappa (Linear):    {metrics['cohen_kappa_linear']:.4f}")
    print(f"  Cohen's Kappa (Quadratic): {metrics['cohen_kappa_quadratic']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(TARGET_CLASS_NAMES):
        print(f"  {name:<10} {metrics['per_class_precision'][i]:>10.4f} "
              f"{metrics['per_class_recall'][i]:>10.4f} {metrics['per_class_f1'][i]:>10.4f}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Plot and save confusion matrix."""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=TARGET_CLASS_NAMES, yticklabels=TARGET_CLASS_NAMES)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Normalized (percentages)
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', ax=ax2,
                xticklabels=TARGET_CLASS_NAMES, yticklabels=TARGET_CLASS_NAMES)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Plot classification metrics as bar chart."""
    print("\nGenerating classification report plot...")
    
    report = classification_report(y_true, y_pred, target_names=TARGET_CLASS_NAMES, output_dict=True)
    
    # Convert to DataFrame
    df_report = pd.DataFrame(report).T
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df_report = df_report[['precision', 'recall', 'f1-score']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(TARGET_CLASS_NAMES))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_report['precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, df_report['recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, df_report['f1-score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Fire Size Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_CLASS_NAMES)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_importance(model: lgb.Booster, X: np.ndarray, 
                         feature_names: list, save_path: Path,
                         max_display: int = 20) -> None:
    """Generate SHAP feature importance plots."""
    print("\nGenerating SHAP analysis...")
    print(f"  X shape: {X.shape}")
    print(f"  Number of feature names: {len(feature_names)}")
    
    # Use a sample for SHAP (faster computation)
    sample_size = min(5000, len(X))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP values is a list of arrays (one per class for multiclass)
    # Average absolute SHAP values across all classes
    if isinstance(shap_values, list):
        # If it's a list of arrays, each array is (samples, features)
        # We want the mean absolute value for each feature across all samples and all classes
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Handle case where shap_values is a single array (samples, features * classes)
        # or (samples, features)
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # If we have a multiple of features, it's likely multiclass flattened
        num_feats = len(feature_names)
        if mean_shap.size > num_feats and mean_shap.size % num_feats == 0:
            n_classes = mean_shap.size // num_feats
            print(f"  Aggregating SHAP values for {n_classes} classes...")
            mean_shap = mean_shap.reshape(n_classes, num_feats).mean(axis=0)
    
    # Ensure mean_shap is 1D
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.flatten()
    
    print(f"  Mean SHAP shape: {mean_shap.shape}")

    # Handle mismatch between feature_names and mean_shap length
    if len(feature_names) != mean_shap.size:
        print(f"  WARNING: Feature names ({len(feature_names)}) != SHAP values ({mean_shap.size})")
        # Trim to match
        n = min(len(feature_names), mean_shap.size)
        feature_names = feature_names[:n]
        mean_shap = mean_shap[:n]
        print(f"  Trimmed to {n} features")

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=True)
    
    # Plot 1: Feature Importance Bar Chart
    plt.figure(figsize=(10, 8))
    top_features = importance_df.tail(max_display)
    plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {max_display} Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved importance plot: {save_path}")
    
    # Plot 2: SHAP Summary Plot (Large Fires)
    # Extract SHAP values for Large fire class (class index 2)
    shap_values_large = None
    num_feats = len(feature_names)
    
    if isinstance(shap_values, list) and len(shap_values) > 2:
        # Already a list of arrays per class
        shap_values_large = shap_values[2]
    elif isinstance(shap_values, np.ndarray):
        # Single array - need to reshape if it's (samples, features * classes)
        if shap_values.shape[1] == num_feats * 3:
            # Reshape from (samples, features*classes) to (samples, classes, features)
            # Then extract class 2 (Large fires)
            reshaped = shap_values.reshape(shap_values.shape[0], 3, num_feats)
            shap_values_large = reshaped[:, 2, :]  # Class 2 = Large
            print(f"  Extracted Large fire SHAP values: {shap_values_large.shape}")
        elif shap_values.shape[1] == num_feats:
            # Binary or single output - use as-is
            shap_values_large = shap_values
    
    if shap_values_large is not None:
        summary_path = save_path.parent / f"{save_path.stem}_summary{save_path.suffix}"
        plt.figure(figsize=(10, 8))
        try:
            print("  Generating SHAP summary plot...")
            shap.summary_plot(shap_values_large, X_sample, feature_names=feature_names,
                             max_display=max_display, show=False)
            plt.title('SHAP Summary: Large Fire Class', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"  Saved summary plot: {summary_path}")
        except Exception as e:
            print(f"  Could not generate summary plot: {e}")
        plt.close()
    else:
        print("  Skipping summary plot (could not extract Large class SHAP values)")
    
    # Print top features
    print("\n  Top 10 Most Important Features:")
    for _, row in importance_df.tail(10).iloc[::-1].iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def analyze_errors(test_df: pd.DataFrame, y_true: np.ndarray, 
                   y_pred: np.ndarray, save_path: Path) -> None:
    """Analyze misclassifications."""
    print("\nAnalyzing misclassifications...")
    
    # Add predictions to dataframe
    test_df = test_df.copy()
    test_df['predicted'] = y_pred
    test_df['correct'] = y_true == y_pred
    
    errors = test_df[~test_df['correct']]
    
    print(f"\n  Total errors: {len(errors):,} ({len(errors)/len(test_df)*100:.1f}%)")
    
    # Error types
    print("\n  Error Distribution:")
    for true_class in range(3):
        for pred_class in range(3):
            if true_class != pred_class:
                count = ((y_true == true_class) & (y_pred == pred_class)).sum()
                if count > 0:
                    pct = count / len(errors) * 100
                    true_name = TARGET_CLASS_NAMES[true_class]
                    pred_name = TARGET_CLASS_NAMES[pred_class]
                    print(f"    {true_name} → {pred_name}: {count:,} ({pct:.1f}%)")
    
    # Adjacent vs non-adjacent errors (important for ordinal)
    adjacent_errors = 0
    non_adjacent_errors = 0
    
    for true_class, pred_class in zip(y_true[y_true != y_pred], y_pred[y_true != y_pred]):
        if abs(true_class - pred_class) == 1:
            adjacent_errors += 1
        else:
            non_adjacent_errors += 1
    
    print(f"\n  Ordinal Error Analysis:")
    print(f"    Adjacent errors (off by 1):     {adjacent_errors:,} ({adjacent_errors/len(errors)*100:.1f}%)")
    print(f"    Non-adjacent errors (off by 2): {non_adjacent_errors:,} ({non_adjacent_errors/len(errors)*100:.1f}%)")


def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray, save_path: Path) -> None:
    """Plot prediction probability distributions."""
    print("\nGenerating prediction distribution plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, class_name) in enumerate(zip(axes, TARGET_CLASS_NAMES)):
        # Get probabilities for this class
        proba = y_proba[:, i]
        
        # Split by actual class
        for true_class in range(3):
            mask = y_true == true_class
            ax.hist(proba[mask], bins=50, alpha=0.5, 
                   label=f'Actual: {TARGET_CLASS_NAMES[true_class]}', density=True)
        
        ax.set_xlabel(f'P({class_name})')
        ax.set_ylabel('Density')
        ax.set_title(f'Predicted Probability: {class_name}', fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, metadata, train_df, test_df = load_model_and_data()
    feature_names = metadata['feature_names']
    
    # Prepare data
    X_test, y_test = prepare_data(test_df, feature_names)
    X_train, y_train = prepare_data(train_df, feature_names)
    
    # Make predictions
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics)
    
    # Generate plots
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / 'confusion_matrix.png')
    plot_classification_report(y_test, y_pred, FIGURES_DIR / 'classification_metrics.png')
    plot_prediction_distribution(y_test, y_pred, y_proba, FIGURES_DIR / 'prediction_distribution.png')
    
    # SHAP analysis
    importance_df = plot_shap_importance(model, X_test, feature_names, 
                                         FIGURES_DIR / 'shap_importance.png')
    
    # Error analysis
    analyze_errors(test_df, y_test, y_pred, FIGURES_DIR / 'error_analysis.png')
    
    # Save importance rankings
    importance_df.to_csv(FIGURES_DIR / 'feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("✓ Evaluation Complete!")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

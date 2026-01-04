"""
Configuration settings for the Wildfire Size Classification project.

Target: Ordinal classification of fire size into 3 classes:
    - 0 (Small): Classes A + B (0 - 9.9 acres)
    - 1 (Medium): Classes C + D (10 - 299 acres)  
    - 2 (Large): Classes E + F + G (300+ acres)
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# SQLite database path (adjust filename if different)
SQLITE_DB_PATH = PROJECT_ROOT / "FPA_FOD_20170508.sqlite"

# Output paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Processed data files
RAW_PARQUET = PROCESSED_DATA_DIR / "fires_raw.parquet"
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "fires_processed.parquet"
FEATURES_PARQUET = PROCESSED_DATA_DIR / "fires_features.parquet"
TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DATA_DIR / "test.parquet"

# =============================================================================
# TARGET VARIABLE CONFIGURATION
# =============================================================================

# Original fire size classes mapping to ordinal target
# A (0-0.25 acres), B (0.26-9.9 acres) -> 0 (Small)
# C (10-99.9 acres), D (100-299 acres) -> 1 (Medium)
# E (300-999 acres), F (1000-4999 acres), G (5000+ acres) -> 2 (Large)

FIRE_SIZE_CLASS_MAPPING = {
    'A': 0, 'B': 0,  # Small
    'C': 1, 'D': 1,  # Medium
    'E': 2, 'F': 2, 'G': 2  # Large
}

TARGET_CLASS_NAMES = ['Small', 'Medium', 'Large']
TARGET_COLUMN = 'fire_size_ordinal'
ORIGINAL_TARGET_COLUMN = 'FIRE_SIZE_CLASS'

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Columns to drop (IDs, redundant info, text fields)
COLUMNS_TO_DROP = [
    'FOD_ID', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM',
    'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME',
    'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME',
    'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID',
    'FIRE_CODE', 'FIRE_NAME', 
    'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME',
    'MTBS_ID', 'MTBS_FIRE_NAME', 'COMPLEX_NAME',
    'DISCOVERY_DATE', 'DISCOVERY_TIME',
    'CONT_DATE', 'CONT_DOY', 'CONT_TIME',
    'FIPS_CODE', 'FIPS_NAME',
    'FIRE_SIZE',  # Don't use actual size as feature - it's what we're predicting
    'FIRE_SIZE_CLASS',  # Original target
    'Shape'  # Geometry column if present
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    'NWCG_REPORTING_AGENCY',
    'STAT_CAUSE_DESCR',
    'STATE',
    'OWNER_DESCR'
]

# Numerical features (after feature engineering)
NUMERICAL_FEATURES = [
    'LATITUDE',
    'LONGITUDE',
    'DISCOVERY_DOY',
    'FIRE_YEAR'
]

# Temporal features to create
TEMPORAL_FEATURES = [
    'month',
    'season',
    'day_of_week',
    'is_weekend'
]

# Geospatial features to create
GEOSPATIAL_FEATURES = [
    'lat_bin',
    'lon_bin', 
    'geo_cluster',
    'lat_lon_interaction'
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.2

# Cross-validation folds
N_FOLDS = 5

# Class weights for imbalanced data (will be computed dynamically)
USE_CLASS_WEIGHTS = True

# LightGBM base parameters for ordinal classification
LIGHTGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Optuna hyperparameter search space
OPTUNA_SEARCH_SPACE = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3),
    'num_leaves': (20, 150),
    'min_child_samples': (10, 100),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 1.0)
}

# Number of Optuna trials
N_OPTUNA_TRIALS = 50

# =============================================================================
# GEOSPATIAL CLUSTERING CONFIGURATION
# =============================================================================

# Number of clusters for geographic regions
N_GEO_CLUSTERS = 20

# Latitude/Longitude binning
LAT_BINS = 10
LON_BINS = 10

# =============================================================================
# EVALUATION METRICS
# =============================================================================

# Primary metric for model selection
PRIMARY_METRIC = 'macro_f1'

# All metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'balanced_accuracy', 
    'macro_f1',
    'weighted_f1',
    'cohen_kappa',
    'macro_precision',
    'macro_recall'
]

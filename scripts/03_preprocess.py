"""
Script 03: Data Preprocessing

This script preprocesses the raw wildfire data:
- Creates ordinal target variable (3 classes: Small, Medium, Large)
- Drops irrelevant columns (IDs, text fields, redundant info)
- Handles missing values
- Encodes categorical variables
- Splits data into train/test sets (stratified)

Usage:
    python scripts/03_preprocess.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    RAW_PARQUET,
    PROCESSED_PARQUET,
    TRAIN_PARQUET,
    TEST_PARQUET,
    PROCESSED_DATA_DIR,
    FIRE_SIZE_CLASS_MAPPING,
    TARGET_CLASS_NAMES,
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE
)


def load_data() -> pd.DataFrame:
    """Load the raw parquet data."""
    print("Loading raw data...")
    df = pd.read_parquet(RAW_PARQUET)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create ordinal target variable from FIRE_SIZE_CLASS."""
    print("\nCreating ordinal target variable...")
    
    # Map original classes to ordinal (0, 1, 2)
    df[TARGET_COLUMN] = df['FIRE_SIZE_CLASS'].map(FIRE_SIZE_CLASS_MAPPING)
    
    # Check for unmapped values
    unmapped = df[TARGET_COLUMN].isna().sum()
    if unmapped > 0:
        print(f"  Warning: {unmapped} records could not be mapped. Dropping...")
        df = df.dropna(subset=[TARGET_COLUMN])
    
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    
    # Print distribution
    print("\n  Target Variable Distribution:")
    for val in sorted(df[TARGET_COLUMN].unique()):
        count = (df[TARGET_COLUMN] == val).sum()
        pct = count / len(df) * 100
        print(f"    {val} ({TARGET_CLASS_NAMES[val]}): {count:,} ({pct:.2f}%)")
    
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not useful for prediction."""
    print("\nDropping irrelevant columns...")
    
    # Get columns that exist in the dataframe
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    
    print(f"  Dropping {len(cols_to_drop)} columns:")
    for col in cols_to_drop[:10]:
        print(f"    - {col}")
    if len(cols_to_drop) > 10:
        print(f"    ... and {len(cols_to_drop) - 10} more")
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"  Remaining columns: {len(df.columns)}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    print("\nHandling missing values...")
    
    initial_rows = len(df)
    
    # Check missing in essential columns
    essential_cols = ['LATITUDE', 'LONGITUDE', 'FIRE_YEAR', 'DISCOVERY_DOY', TARGET_COLUMN]
    for col in essential_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing values")
    
    # Drop rows with missing essential values
    df = df.dropna(subset=[c for c in essential_cols if c in df.columns])
    
    # For categorical features, fill with 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna('Unknown')
                print(f"  {col}: Filled {missing} missing with 'Unknown'")
    
    rows_dropped = initial_rows - len(df)
    print(f"\n  Rows dropped due to missing essential values: {rows_dropped:,}")
    print(f"  Remaining rows: {len(df):,}")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features using Label Encoding."""
    print("\nEncoding categorical features...")
    
    encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
            n_categories = len(le.classes_)
            print(f"  {col}: {n_categories} categories")
    
    return df, encoders


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select features for modeling."""
    print("\nSelecting features for modeling...")
    
    # Features to keep
    feature_cols = [
        # Numerical
        'LATITUDE', 'LONGITUDE', 'FIRE_YEAR', 'DISCOVERY_DOY',
        # Encoded categorical
        'NWCG_REPORTING_AGENCY_encoded',
        'STAT_CAUSE_DESCR_encoded',
        'STATE_encoded',
        'OWNER_DESCR_encoded',
        # Target
        TARGET_COLUMN
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Also keep original categorical columns for reference
    original_cats = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    
    all_cols = available_cols + original_cats
    all_cols = list(dict.fromkeys(all_cols))  # Remove duplicates, preserve order
    
    df = df[all_cols]
    
    print(f"  Selected {len(available_cols)} feature columns + target")
    print(f"  Final columns: {list(df.columns)}")
    
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    print("\nSplitting data into train/test sets...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )
    
    print(f"  Train set: {len(train_df):,} rows ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  Test set: {len(test_df):,} rows ({100*TEST_SIZE:.0f}%)")
    
    # Verify stratification
    print("\n  Target distribution in splits:")
    for name, data in [('Train', train_df), ('Test', test_df)]:
        dist = data[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
        dist_str = ", ".join([f"{TARGET_CLASS_NAMES[i]}: {v:.1f}%" for i, v in dist.items()])
        print(f"    {name}: {dist_str}")
    
    return train_df, test_df


def save_data(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save processed data to parquet files."""
    print("\nSaving processed data...")
    
    # Create directory if needed
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full processed data
    df.to_parquet(PROCESSED_PARQUET, index=False)
    print(f"  Full processed data: {PROCESSED_PARQUET}")
    
    # Save train/test splits
    train_df.to_parquet(TRAIN_PARQUET, index=False)
    print(f"  Train data: {TRAIN_PARQUET}")
    
    test_df.to_parquet(TEST_PARQUET, index=False)
    print(f"  Test data: {TEST_PARQUET}")


def print_summary(df: pd.DataFrame) -> None:
    """Print preprocessing summary."""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts().to_string())
    
    print(f"\nFeature statistics:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != TARGET_COLUMN:
            print(f"  {col}:")
            print(f"    Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"    Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create target variable
    df = create_target_variable(df)
    
    # Drop irrelevant columns
    df = drop_irrelevant_columns(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Select features
    df = select_features(df)
    
    # Split data
    train_df, test_df = split_data(df)
    
    # Save data
    save_data(df, train_df, test_df)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "="*60)
    print("✓ Preprocessing Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

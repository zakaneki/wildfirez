"""
Script 04: Feature Engineering

This script creates additional features for the model:
- Temporal features (month, season, day of week)
- Geospatial features (lat/lon bins, clustering, interactions)
- Coordinate transformations

Usage:
    python scripts/04_feature_engineering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    TRAIN_PARQUET,
    TEST_PARQUET,
    FEATURES_PARQUET,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    N_GEO_CLUSTERS,
    LAT_BINS,
    LON_BINS,
    RANDOM_STATE
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test: {len(test_df):,} rows")
    return train_df, test_df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from DISCOVERY_DOY."""
    print("\nCreating temporal features...")
    
    # Convert day of year to datetime for feature extraction
    # Using a non-leap year as reference
    reference_year = 2001
    df['temp_date'] = pd.to_datetime(
        df['DISCOVERY_DOY'].astype(int).astype(str) + f'-{reference_year}', 
        format='%j-%Y',
        errors='coerce'
    )
    
    # Handle invalid dates
    invalid_dates = df['temp_date'].isna().sum()
    if invalid_dates > 0:
        print(f"  Warning: {invalid_dates} invalid day of year values")
        # Fill with median day
        median_doy = df['DISCOVERY_DOY'].median()
        df.loc[df['temp_date'].isna(), 'temp_date'] = pd.to_datetime(
            f'{int(median_doy)}-{reference_year}', format='%j-%Y'
        )
    
    # Extract features
    df['month'] = df['temp_date'].dt.month
    df['day_of_week'] = df['temp_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['month'].apply(lambda m: 
        1 if m in [12, 1, 2] else
        2 if m in [3, 4, 5] else
        3 if m in [6, 7, 8] else 4
    )
    
    # Fire season indicator (peak fire months: June-October)
    df['is_fire_season'] = df['month'].isin([6, 7, 8, 9, 10]).astype(int)
    
    # Drop temporary date column
    df = df.drop(columns=['temp_date'])
    
    print("  Created: month, day_of_week, is_weekend, season, is_fire_season")
    
    return df


def create_geospatial_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Create geospatial features from coordinates."""
    print("\nCreating geospatial features...")
    
    # 1. Latitude/Longitude bins
    print("  Creating coordinate bins...")
    
    # Define bin edges based on continental US bounds
    lat_min, lat_max = 24.0, 50.0
    lon_min, lon_max = -125.0, -66.0
    
    lat_edges = np.linspace(lat_min, lat_max, LAT_BINS + 1)
    lon_edges = np.linspace(lon_min, lon_max, LON_BINS + 1)
    
    for df in [train_df, test_df]:
        df['lat_bin'] = pd.cut(df['LATITUDE'], bins=lat_edges, labels=False, include_lowest=True)
        df['lon_bin'] = pd.cut(df['LONGITUDE'], bins=lon_edges, labels=False, include_lowest=True)
        
        # Fill NaN bins (locations outside continental US) with nearest bin
        df['lat_bin'] = df['lat_bin'].fillna(df['lat_bin'].median()).astype(int)
        df['lon_bin'] = df['lon_bin'].fillna(df['lon_bin'].median()).astype(int)
    
    # 2. Geographic clustering using K-Means
    print(f"  Fitting K-Means clustering (k={N_GEO_CLUSTERS})...")
    
    # Prepare coordinates for clustering
    train_coords = train_df[['LATITUDE', 'LONGITUDE']].values
    test_coords = test_df[['LATITUDE', 'LONGITUDE']].values
    
    # Scale coordinates
    scaler = StandardScaler()
    train_coords_scaled = scaler.fit_transform(train_coords)
    test_coords_scaled = scaler.transform(test_coords)
    
    # Fit K-Means on train data
    kmeans = KMeans(n_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    train_df['geo_cluster'] = kmeans.fit_predict(train_coords_scaled)
    test_df['geo_cluster'] = kmeans.predict(test_coords_scaled)
    
    print(f"  Cluster distribution (train):")
    cluster_dist = train_df['geo_cluster'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        pct = count / len(train_df) * 100
        if pct >= 3:  # Only show clusters with >= 3%
            print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")
    
    # 3. Coordinate interactions
    print("  Creating coordinate interactions...")
    
    for df in [train_df, test_df]:
        # Quadratic terms (captures non-linear patterns)
        df['lat_squared'] = df['LATITUDE'] ** 2
        df['lon_squared'] = df['LONGITUDE'] ** 2
        df['lat_lon_interaction'] = df['LATITUDE'] * df['LONGITUDE']
        
        # Distance from geographic center of continental US
        # Approximate center: 39.8°N, 98.6°W
        center_lat, center_lon = 39.8, -98.6
        df['dist_from_center'] = np.sqrt(
            (df['LATITUDE'] - center_lat) ** 2 + 
            (df['LONGITUDE'] - center_lon) ** 2
        )
    
    print("  Created: lat_bin, lon_bin, geo_cluster, lat_squared, lon_squared, lat_lon_interaction, dist_from_center")
    
    return train_df, test_df, kmeans


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical encoding for periodic features."""
    print("\nCreating cyclical features...")
    
    # Cyclical encoding for month (captures January-December continuity)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding for day of year
    df['doy_sin'] = np.sin(2 * np.pi * df['DISCOVERY_DOY'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['DISCOVERY_DOY'] / 365)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print("  Created: month_sin/cos, doy_sin/cos, dow_sin/cos")
    
    return df


def create_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create year-based features."""
    print("\nCreating year features...")
    
    # Normalized year (0-1 scale for 1992-2015)
    min_year, max_year = 1992, 2015
    df['year_normalized'] = (df['FIRE_YEAR'] - min_year) / (max_year - min_year)
    
    # Years since start
    df['years_since_1992'] = df['FIRE_YEAR'] - min_year
    
    print("  Created: year_normalized, years_since_1992")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for modeling."""
    # Exclude target, original categorical text columns, and intermediate columns
    exclude_cols = [
        TARGET_COLUMN,
        'NWCG_REPORTING_AGENCY', 'STAT_CAUSE_DESCR', 'STATE', 'OWNER_DESCR',
        'COUNTY'  # If present
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save feature-engineered data."""
    print("\nSaving feature-engineered data...")
    
    # Overwrite train/test files with new features
    train_df.to_parquet(TRAIN_PARQUET, index=False)
    test_df.to_parquet(TEST_PARQUET, index=False)
    
    print(f"  Train data: {TRAIN_PARQUET}")
    print(f"  Test data: {TEST_PARQUET}")
    
    # Also save combined for reference
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_parquet(FEATURES_PARQUET, index=False)
    print(f"  Combined data: {FEATURES_PARQUET}")


def print_summary(train_df: pd.DataFrame) -> None:
    """Print feature engineering summary."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    feature_cols = get_feature_columns(train_df)
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("\nFeature list:")
    
    # Group features by type
    temporal = [c for c in feature_cols if c in ['month', 'day_of_week', 'is_weekend', 'season', 'is_fire_season', 
                                                   'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'dow_sin', 'dow_cos']]
    geospatial = [c for c in feature_cols if c in ['lat_bin', 'lon_bin', 'geo_cluster', 'lat_squared', 'lon_squared',
                                                    'lat_lon_interaction', 'dist_from_center', 'LATITUDE', 'LONGITUDE']]
    year_feats = [c for c in feature_cols if c in ['FIRE_YEAR', 'year_normalized', 'years_since_1992', 'DISCOVERY_DOY']]
    encoded = [c for c in feature_cols if c.endswith('_encoded')]
    
    print(f"\n  Temporal ({len(temporal)}): {temporal}")
    print(f"\n  Geospatial ({len(geospatial)}): {geospatial}")
    print(f"\n  Year-based ({len(year_feats)}): {year_feats}")
    print(f"\n  Encoded categorical ({len(encoded)}): {encoded}")


def main():
    """Main feature engineering pipeline."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Create temporal features
    train_df = create_temporal_features(train_df)
    test_df = create_temporal_features(test_df)
    
    # Create geospatial features
    train_df, test_df, kmeans = create_geospatial_features(train_df, test_df)
    
    # Create cyclical features
    train_df = create_cyclical_features(train_df)
    test_df = create_cyclical_features(test_df)
    
    # Create year features
    train_df = create_year_features(train_df)
    test_df = create_year_features(test_df)
    
    # Save data
    save_data(train_df, test_df)
    
    # Print summary
    print_summary(train_df)
    
    print("\n" + "="*60)
    print("✓ Feature Engineering Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

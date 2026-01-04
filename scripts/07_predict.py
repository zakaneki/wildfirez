"""
Script 07: Prediction Pipeline

This script provides inference capabilities:
- Load trained model
- Preprocess new data
- Generate predictions with probabilities
- Can be used as a module or standalone script

Usage:
    # Single prediction
    python scripts/07_predict.py --lat 34.05 --lon -118.24 --state CA --cause "Debris Burning" --month 7
    
    # Batch prediction from CSV
    python scripts/07_predict.py --input new_fires.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    MODELS_DIR,
    TARGET_CLASS_NAMES,
    FIRE_SIZE_CLASS_MAPPING,
    CATEGORICAL_FEATURES,
    N_GEO_CLUSTERS,
    LAT_BINS,
    LON_BINS
)


class WildfirePredictor:
    """Wildfire size class predictor."""
    
    def __init__(self, model_dir: Path = MODELS_DIR):
        """Initialize predictor with trained model."""
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.encoders = {}
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and metadata."""
        model_path = self.model_dir / 'wildfire_model.txt'
        metadata_path = self.model_dir / 'model_metadata.joblib'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
        
        self.model = lgb.Booster(model_file=str(model_path))
        self.metadata = joblib.load(metadata_path)
        self.feature_names = self.metadata['feature_names']
        
        print(f"Loaded model with {len(self.feature_names)} features")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction."""
        df = df.copy()
        
        # Ensure required columns exist
        required = ['LATITUDE', 'LONGITUDE', 'FIRE_YEAR', 'DISCOVERY_DOY']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Temporal features
        reference_year = 2001
        df['temp_date'] = pd.to_datetime(
            df['DISCOVERY_DOY'].astype(int).astype(str) + f'-{reference_year}',
            format='%j-%Y',
            errors='coerce'
        )
        
        df['month'] = df['temp_date'].dt.month
        df['day_of_week'] = df['temp_date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['season'] = df['month'].apply(lambda m:
            1 if m in [12, 1, 2] else
            2 if m in [3, 4, 5] else
            3 if m in [6, 7, 8] else 4
        )
        df['is_fire_season'] = df['month'].isin([6, 7, 8, 9, 10]).astype(int)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['DISCOVERY_DOY'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['DISCOVERY_DOY'] / 365)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Year features
        min_year, max_year = 1992, 2015
        df['year_normalized'] = (df['FIRE_YEAR'] - min_year) / (max_year - min_year)
        df['years_since_1992'] = df['FIRE_YEAR'] - min_year
        
        # Geospatial features
        lat_min, lat_max = 24.0, 50.0
        lon_min, lon_max = -125.0, -66.0
        lat_edges = np.linspace(lat_min, lat_max, LAT_BINS + 1)
        lon_edges = np.linspace(lon_min, lon_max, LON_BINS + 1)
        
        df['lat_bin'] = pd.cut(df['LATITUDE'], bins=lat_edges, labels=False, include_lowest=True)
        df['lon_bin'] = pd.cut(df['LONGITUDE'], bins=lon_edges, labels=False, include_lowest=True)
        df['lat_bin'] = df['lat_bin'].fillna(5).astype(int)
        df['lon_bin'] = df['lon_bin'].fillna(5).astype(int)
        
        # Coordinate features
        df['lat_squared'] = df['LATITUDE'] ** 2
        df['lon_squared'] = df['LONGITUDE'] ** 2
        df['lat_lon_interaction'] = df['LATITUDE'] * df['LONGITUDE']
        
        center_lat, center_lon = 39.8, -98.6
        df['dist_from_center'] = np.sqrt(
            (df['LATITUDE'] - center_lat) ** 2 +
            (df['LONGITUDE'] - center_lon) ** 2
        )
        
        # Placeholder for geo_cluster (would need kmeans model)
        df['geo_cluster'] = 0
        
        # Drop temporary columns
        df = df.drop(columns=['temp_date'], errors='ignore')
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        # Simple label encoding for inference
        # In production, would need to use same encoders as training
        for col in CATEGORICAL_FEATURES:
            encoded_col = f'{col}_encoded'
            if col in df.columns:
                # Simple hash-based encoding as fallback
                df[encoded_col] = df[col].astype(str).apply(lambda x: hash(x) % 100)
            else:
                df[encoded_col] = 0
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data for prediction."""
        df = self._create_features(df)
        df = self._encode_categoricals(df)
        
        # Select and order features to match training
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features (filled with 0): {missing_features}")
            for f in missing_features:
                df[f] = 0
        
        X = df[self.feature_names].values
        return X
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for input data."""
        X = self.preprocess(df)
        
        # Get probabilities
        proba = self.model.predict(X)
        pred_class = np.argmax(proba, axis=1)
        
        # Create results dataframe
        results = df.copy()
        results['predicted_class'] = pred_class
        results['predicted_label'] = [TARGET_CLASS_NAMES[c] for c in pred_class]
        results['prob_small'] = proba[:, 0]
        results['prob_medium'] = proba[:, 1]
        results['prob_large'] = proba[:, 2]
        results['confidence'] = np.max(proba, axis=1)
        
        return results
    
    def predict_single(self, latitude: float, longitude: float,
                       fire_year: int, discovery_doy: int,
                       state: str = 'Unknown',
                       cause: str = 'Unknown',
                       agency: str = 'Unknown',
                       owner: str = 'Unknown') -> dict:
        """Predict for a single fire event."""
        
        df = pd.DataFrame([{
            'LATITUDE': latitude,
            'LONGITUDE': longitude,
            'FIRE_YEAR': fire_year,
            'DISCOVERY_DOY': discovery_doy,
            'STATE': state,
            'STAT_CAUSE_DESCR': cause,
            'NWCG_REPORTING_AGENCY': agency,
            'OWNER_DESCR': owner
        }])
        
        result = self.predict(df).iloc[0]
        
        return {
            'predicted_class': int(result['predicted_class']),
            'predicted_label': result['predicted_label'],
            'probabilities': {
                'Small': float(result['prob_small']),
                'Medium': float(result['prob_medium']),
                'Large': float(result['prob_large'])
            },
            'confidence': float(result['confidence'])
        }


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description='Wildfire size prediction')
    
    # Single prediction arguments
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--year', type=int, default=2015, help='Fire year')
    parser.add_argument('--doy', type=int, default=200, help='Day of year')
    parser.add_argument('--state', type=str, default='Unknown', help='State code')
    parser.add_argument('--cause', type=str, default='Unknown', help='Fire cause')
    
    # Batch prediction arguments
    parser.add_argument('--input', type=str, help='Input CSV file for batch prediction')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WildfirePredictor()
    
    if args.input:
        # Batch prediction
        print(f"\nProcessing batch predictions from: {args.input}")
        df = pd.read_csv(args.input)
        results = predictor.predict(df)
        
        output_path = args.output or 'predictions.csv'
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
    elif args.lat is not None and args.lon is not None:
        # Single prediction
        print("\n" + "="*60)
        print("SINGLE FIRE PREDICTION")
        print("="*60)
        
        result = predictor.predict_single(
            latitude=args.lat,
            longitude=args.lon,
            fire_year=args.year,
            discovery_doy=args.doy,
            state=args.state,
            cause=args.cause
        )
        
        print(f"\nInput:")
        print(f"  Location: ({args.lat}, {args.lon})")
        print(f"  Year: {args.year}, Day of Year: {args.doy}")
        print(f"  State: {args.state}, Cause: {args.cause}")
        
        print(f"\nPrediction:")
        print(f"  Class: {result['predicted_class']} ({result['predicted_label']})")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {label:>6}: {prob:>6.1%} {bar}")
        
    else:
        # Demo prediction
        print("\n" + "="*60)
        print("DEMO PREDICTION")
        print("="*60)
        
        # Example: Summer fire in California
        result = predictor.predict_single(
            latitude=34.05,
            longitude=-118.24,
            fire_year=2015,
            discovery_doy=200,  # Mid-July
            state='CA',
            cause='Debris Burning'
        )
        
        print("\nExample: Summer fire in Los Angeles area")
        print(f"  Predicted: {result['predicted_label']} (confidence: {result['confidence']:.1%})")
        print(f"  Probabilities: Small={result['probabilities']['Small']:.1%}, "
              f"Medium={result['probabilities']['Medium']:.1%}, "
              f"Large={result['probabilities']['Large']:.1%}")
        
        print("\nUsage:")
        print("  Single: python 07_predict.py --lat 34.05 --lon -118.24 --state CA --cause 'Lightning'")
        print("  Batch:  python 07_predict.py --input fires.csv --output predictions.csv")


if __name__ == "__main__":
    main()

# Wildfire Size Classification Project

Predicting wildfire size classes using machine learning on the FPA FOD (Fire Program Analysis Fire-Occurrence Database) containing 1.88 million US wildfire records from 1992-2015.

## Project Overview

This project builds an **ordinal classification model** to predict fire size categories:
- **Small** (0-9.9 acres): Original classes A + B
- **Medium** (10-299 acres): Original classes C + D  
- **Large** (300+ acres): Original classes E + F + G

### Key Features
- **Ordinal-aware classification**: Leverages the natural ordering of fire size classes
- **Geospatial features**: Coordinate clustering, regional binning, distance metrics
- **Temporal features**: Cyclical encoding of month/day, fire season indicators
- **Class imbalance handling**: Balanced class weights for rare large fire events
- **Interpretable results**: SHAP feature importance analysis

## Project Structure

```
wildfires/
├── config/
│   ├── __init__.py            # Package init
│   └── config.py              # Configuration settings
├── data/
│   └── processed/             # Processed parquet files (train/test splits)
├── models/                    # Saved model artifacts
│   ├── best_params.json       # Tuned hyperparameters
│   ├── model_metadata.joblib  # Feature names and metrics
│   └── wildfire_model.txt     # Trained LightGBM model
├── reports/
│   └── figures/               # Visualizations and metrics
├── scripts/
│   ├── 01_extract_data.py     # Extract SQLite → Parquet
│   ├── 02_eda.py              # Exploratory data analysis
│   ├── 03_preprocess.py       # Data preprocessing
│   ├── 04_feature_engineering.py  # Feature creation
│   ├── 05_train_model.py      # Model training
│   ├── 06_evaluate.py         # Model evaluation
│   └── 07_predict.py          # Prediction pipeline
├── run_pipeline.py            # Run full or partial pipeline
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore rules
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+
- SQLite database file (`FPA_FOD_20170508.sqlite`)

### Installation

1. Clone/download the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the SQLite database file in the project root

### Running the Pipeline

**Using the pipeline runner (recommended):**

```bash
# Run full pipeline
python run_pipeline.py

# Skip EDA step
python run_pipeline.py --skip-eda

# Run with hyperparameter tuning
python run_pipeline.py --tune

# Resume from a specific step (1-7)
python run_pipeline.py --from-step 5
```

**Or execute scripts individually:**

```bash
# 1. Extract data from SQLite
python scripts/01_extract_data.py

# 2. Exploratory data analysis (generates plots)
python scripts/02_eda.py

# 3. Preprocess data
python scripts/03_preprocess.py

# 4. Feature engineering
python scripts/04_feature_engineering.py

# 5. Train model (add --tune for hyperparameter tuning)
python scripts/05_train_model.py
# python scripts/05_train_model.py --tune  # With Optuna tuning

# 6. Evaluate model
python scripts/06_evaluate.py

# 7. Make predictions
python scripts/07_predict.py --lat 34.05 --lon -118.24 --state CA --cause "Lightning"
```

## Model Details

### Features Used
- **Temporal**: Month, day of week, season, fire season indicator (cyclically encoded)
- **Geospatial**: Lat/lon coordinates, regional clusters (K-means), coordinate bins
- **Categorical**: State, fire cause, reporting agency, land owner
- **Year**: Fire year, years since 1992

### Algorithm
- **LightGBM** gradient boosting for multi-class classification
- Class weights to handle imbalanced data (~90% small fires)
- Linear weighted Cohen's Kappa for ordinal evaluation

### Expected Performance
- Balanced Accuracy: ~65-75%
- Macro F1 Score: ~0.45-0.55
- Large fire detection is challenging due to class imbalance

## Evaluation Metrics

For ordinal classification, we prioritize:
- **Macro F1**: Equal importance to all classes
- **Balanced Accuracy**: Accounts for class imbalance
- **Linear Weighted Kappa**: Penalizes predictions far from true class

## Output Files

After running the pipeline:
- `data/processed/`: Parquet files for train/test splits
- `models/wildfire_model.txt`: Trained LightGBM model
- `models/model_metadata.joblib`: Feature names and metrics
- `reports/figures/`: Visualizations (confusion matrix, SHAP plots, etc.)

## Data Source

**Fire Program Analysis Fire-Occurrence Database (FPA FOD)**
- 1.88 million geo-referenced wildfire records
- Period: 1992-2015
- 140 million acres burned
- Source: US federal, state, and local fire organizations

## License

This project uses publicly available government data.

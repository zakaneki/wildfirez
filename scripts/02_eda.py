"""
Script 02: Exploratory Data Analysis (EDA)

This script performs comprehensive EDA on the wildfire dataset:
- Class distribution analysis (original 7 classes and grouped 3 classes)
- Geographic distribution of fires
- Temporal patterns (yearly, monthly, seasonal)
- Missing value analysis
- Feature correlations

Generates visualization plots saved to reports/figures/

Usage:
    python scripts/02_eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    RAW_PARQUET,
    FIGURES_DIR,
    FIRE_SIZE_CLASS_MAPPING,
    TARGET_CLASS_NAMES
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data() -> pd.DataFrame:
    """Load the raw parquet data."""
    print("Loading data...")
    df = pd.read_parquet(RAW_PARQUET)
    print(f"  Loaded {len(df):,} records")
    return df


def analyze_class_distribution(df: pd.DataFrame) -> None:
    """Analyze and visualize fire size class distribution."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Original 7 classes
    print("\nOriginal Fire Size Classes:")
    original_dist = df['FIRE_SIZE_CLASS'].value_counts().sort_index()
    for cls, count in original_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {cls}: {count:>10,} ({pct:>6.2f}%)")
    
    # Grouped 3 classes
    df['fire_size_grouped'] = df['FIRE_SIZE_CLASS'].map(FIRE_SIZE_CLASS_MAPPING)
    
    print("\nGrouped Classes (Target Variable):")
    grouped_dist = df['fire_size_grouped'].value_counts().sort_index()
    for cls_idx, count in grouped_dist.items():
        pct = count / len(df) * 100
        cls_name = TARGET_CLASS_NAMES[cls_idx]
        print(f"  {cls_idx} ({cls_name:>6}): {count:>10,} ({pct:>6.2f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original distribution
    colors_orig = sns.color_palette("YlOrRd", 7)
    ax1 = axes[0]
    original_dist.plot(kind='bar', ax=ax1, color=colors_orig, edgecolor='black')
    ax1.set_title('Original Fire Size Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fire Size Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # Add percentage labels
    for i, (idx, val) in enumerate(original_dist.items()):
        pct = val / len(df) * 100
        ax1.annotate(f'{pct:.1f}%', (i, val), ha='center', va='bottom', fontsize=9)
    
    # Grouped distribution
    colors_grouped = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    ax2 = axes[1]
    grouped_dist.plot(kind='bar', ax=ax2, color=colors_grouped, edgecolor='black')
    ax2.set_title('Grouped Fire Size Distribution (Target)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fire Size Category')
    ax2.set_ylabel('Count')
    ax2.set_xticklabels(TARGET_CLASS_NAMES, rotation=0)
    
    # Add percentage labels
    for i, (idx, val) in enumerate(grouped_dist.items()):
        pct = val / len(df) * 100
        ax2.annotate(f'{pct:.1f}%', (i, val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: class_distribution.png")


def analyze_geographic_distribution(df: pd.DataFrame) -> None:
    """Analyze and visualize geographic distribution of fires."""
    print("\n" + "="*60)
    print("GEOGRAPHIC DISTRIBUTION")
    print("="*60)
    
    # Top states
    print("\nTop 15 States by Fire Count:")
    state_dist = df['STATE'].value_counts().head(15)
    for state, count in state_dist.items():
        pct = count / len(df) * 100
        print(f"  {state}: {count:>10,} ({pct:>5.1f}%)")
    
    # Fire locations scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # All fires (sampled for performance)
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    ax1 = axes[0]
    scatter = ax1.scatter(
        df_sample['LONGITUDE'], 
        df_sample['LATITUDE'],
        c=df_sample['FIRE_SIZE_CLASS'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}),
        cmap='YlOrRd',
        alpha=0.3,
        s=1
    )
    ax1.set_title(f'Fire Locations (n={sample_size:,} sample)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim(-130, -65)
    ax1.set_ylim(24, 50)
    plt.colorbar(scatter, ax=ax1, label='Fire Size Class (A=0 to G=6)')
    
    # Large fires only (E, F, G)
    df_large = df[df['FIRE_SIZE_CLASS'].isin(['E', 'F', 'G'])]
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        df_large['LONGITUDE'],
        df_large['LATITUDE'],
        c=df_large['FIRE_SIZE_CLASS'].map({'E': 0, 'F': 1, 'G': 2}),
        cmap='Reds',
        alpha=0.5,
        s=5
    )
    ax2.set_title(f'Large Fires Only (E/F/G, n={len(df_large):,})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim(-130, -65)
    ax2.set_ylim(24, 50)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'geographic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: geographic_distribution.png")


def analyze_temporal_patterns(df: pd.DataFrame) -> None:
    """Analyze temporal patterns in the data."""
    print("\n" + "="*60)
    print("TEMPORAL PATTERNS")
    print("="*60)
    
    # Convert discovery day of year to month
    df['month'] = pd.to_datetime(df['DISCOVERY_DOY'], format='%j').dt.month
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Yearly trend
    ax1 = axes[0, 0]
    yearly = df.groupby('FIRE_YEAR').size()
    yearly.plot(kind='line', ax=ax1, marker='o', linewidth=2, markersize=4)
    ax1.set_title('Fires per Year', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Fires')
    ax1.grid(True, alpha=0.3)
    
    # Monthly distribution
    ax2 = axes[0, 1]
    monthly = df.groupby('month').size()
    monthly.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
    ax2.set_title('Fires by Month', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Fires')
    ax2.tick_params(axis='x', rotation=0)
    
    # Large fires by month
    ax3 = axes[1, 0]
    df['fire_size_grouped'] = df['FIRE_SIZE_CLASS'].map(FIRE_SIZE_CLASS_MAPPING)
    monthly_by_class = df.groupby(['month', 'fire_size_grouped']).size().unstack(fill_value=0)
    monthly_by_class.columns = TARGET_CLASS_NAMES
    monthly_by_class.plot(kind='bar', ax=ax3, width=0.8, 
                          color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax3.set_title('Fire Size Category by Month', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Number of Fires')
    ax3.tick_params(axis='x', rotation=0)
    ax3.legend(title='Size Category')
    
    # Fire causes
    ax4 = axes[1, 1]
    cause_dist = df['STAT_CAUSE_DESCR'].value_counts().head(10)
    cause_dist.plot(kind='barh', ax=ax4, color='steelblue', edgecolor='black')
    ax4.set_title('Top 10 Fire Causes', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Fires')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: temporal_patterns.png")
    
    # Print monthly stats
    print("\nFires by Month:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in monthly.items():
        pct = count / len(df) * 100
        print(f"  {month_names[month-1]}: {count:>10,} ({pct:>5.1f}%)")


def analyze_missing_values(df: pd.DataFrame) -> None:
    """Analyze missing values in the dataset."""
    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    print(f"\nColumns with missing values: {len(missing_df)}")
    print("\nTop 20 columns with missing values:")
    for col, row in missing_df.head(20).iterrows():
        print(f"  {col}: {row['Missing Count']:,} ({row['Missing %']:.1f}%)")
    
    # Visualize
    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        missing_df.head(20)['Missing %'].plot(
            kind='barh', ax=ax, color='salmon', edgecolor='black'
        )
        ax.set_title('Missing Values by Column (Top 20)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Missing %')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: missing_values.png")


def analyze_cause_by_size(df: pd.DataFrame) -> None:
    """Analyze fire causes by fire size category."""
    print("\n" + "="*60)
    print("FIRE CAUSE BY SIZE ANALYSIS")
    print("="*60)
    
    df['fire_size_grouped'] = df['FIRE_SIZE_CLASS'].map(FIRE_SIZE_CLASS_MAPPING)
    
    # Cross-tabulation
    cause_size = pd.crosstab(
        df['STAT_CAUSE_DESCR'], 
        df['fire_size_grouped'],
        normalize='index'
    ) * 100
    cause_size.columns = TARGET_CLASS_NAMES
    
    print("\nFire Cause Distribution by Size Category (% of each cause):")
    print(cause_size.round(1).to_string())
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    cause_size.plot(kind='barh', ax=ax, stacked=True, 
                    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white')
    ax.set_title('Fire Size Distribution by Cause', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage')
    ax.legend(title='Size Category', loc='lower right')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cause_by_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: cause_by_size.png")


def analyze_owner_distribution(df: pd.DataFrame) -> None:
    """Analyze land owner distribution."""
    print("\n" + "="*60)
    print("LAND OWNER ANALYSIS")
    print("="*60)
    
    owner_dist = df['OWNER_DESCR'].value_counts()
    print("\nFires by Land Owner:")
    for owner, count in owner_dist.head(10).items():
        pct = count / len(df) * 100
        print(f"  {owner}: {count:,} ({pct:.1f}%)")


def main():
    """Main EDA pipeline."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Run analyses
    analyze_class_distribution(df)
    analyze_geographic_distribution(df)
    analyze_temporal_patterns(df)
    analyze_missing_values(df)
    analyze_cause_by_size(df)
    analyze_owner_distribution(df)
    
    print("\n" + "="*60)
    print("✓ EDA Complete!")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

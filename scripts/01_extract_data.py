"""
Script 01: Extract Data from SQLite Database

This script connects to the FPA FOD SQLite database, extracts the Fires table,
and saves it as a Parquet file for faster subsequent processing.

Usage:
    python scripts/01_extract_data.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import (
    SQLITE_DB_PATH,
    PROCESSED_DATA_DIR,
    RAW_PARQUET
)


def connect_to_database(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Please ensure the SQLite file is in the project root."
        )
    
    print(f"Connecting to database: {db_path}")
    return sqlite3.connect(db_path)


def get_table_info(conn: sqlite3.Connection) -> None:
    """Print information about tables in the database."""
    cursor = conn.cursor()
    
    # Get list of user tables (skip SpatiaLite system tables)
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%' 
        AND name NOT LIKE 'spatial%' 
        AND name NOT LIKE 'virt%'
        AND name NOT LIKE 'view%'
        AND name NOT LIKE 'geometry%'
    """)
    tables = cursor.fetchall()
    
    print("\n" + "="*60)
    print("DATABASE TABLES")
    print("="*60)
    
    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count:,} rows")
        except Exception as e:
            print(f"  {table_name}: (could not read - {str(e)[:30]})")
    
    print("="*60 + "\n")


def extract_fires_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract the Fires table from the database."""
    print("Extracting Fires table...")
    
    query = "SELECT * FROM Fires"
    df = pd.read_sql_query(query, conn)
    
    print(f"  Loaded {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return df


def save_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to Parquet format."""
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to Parquet: {output_path}")
    df.to_parquet(output_path, index=False, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"  File size: {file_size_mb:.1f} MB")


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the extracted data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nDate Range: {df['FIRE_YEAR'].min()} - {df['FIRE_YEAR'].max()}")
    
    print("\nFire Size Class Distribution:")
    size_dist = df['FIRE_SIZE_CLASS'].value_counts().sort_index()
    for cls, count in size_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {cls}: {count:>10,} ({pct:>5.1f}%)")
    
    print("\nTop 10 States by Fire Count:")
    state_dist = df['STATE'].value_counts().head(10)
    for state, count in state_dist.items():
        print(f"  {state}: {count:,}")
    
    print("\nTop Causes:")
    cause_dist = df['STAT_CAUSE_DESCR'].value_counts().head(5)
    for cause, count in cause_dist.items():
        pct = count / len(df) * 100
        print(f"  {cause}: {count:,} ({pct:.1f}%)")
    
    print("\nMissing Values (top 10 columns):")
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    for col, count in missing.items():
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    
    print("="*60 + "\n")


def main():
    """Main extraction pipeline."""
    print("\n" + "="*60)
    print("WILDFIRE DATA EXTRACTION")
    print("="*60 + "\n")
    
    # Connect to database
    conn = connect_to_database(SQLITE_DB_PATH)
    
    try:
        # Show database info
        get_table_info(conn)
        
        # Extract Fires table
        df = extract_fires_table(conn)
        
        # Print summary
        print_data_summary(df)
        
        # Save to Parquet
        save_to_parquet(df, RAW_PARQUET)
        
        print("\n✓ Data extraction complete!")
        print(f"  Output: {RAW_PARQUET}")
        
    finally:
        conn.close()
        print("  Database connection closed.")


if __name__ == "__main__":
    main()

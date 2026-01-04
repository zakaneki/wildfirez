"""
Main Pipeline Runner

Runs the complete ML pipeline from data extraction to evaluation.

Usage:
    python run_pipeline.py           # Run full pipeline
    python run_pipeline.py --skip-eda  # Skip EDA step
    python run_pipeline.py --tune    # Include hyperparameter tuning
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, extra_args: list = None) -> bool:
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / "scripts" / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run wildfire ML pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--from-step", type=int, default=1, help="Start from step number (1-7)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("WILDFIRE SIZE CLASSIFICATION PIPELINE")
    print("="*60)
    
    steps = [
        ("01_extract_data.py", []),
        ("02_eda.py", []),
        ("03_preprocess.py", []),
        ("04_feature_engineering.py", []),
        ("05_train_model.py", ["--tune"] if args.tune else []),
        ("06_evaluate.py", []),
    ]
    
    for i, (script, extra_args) in enumerate(steps, 1):
        if i < args.from_step:
            print(f"\nSkipping step {i}: {script}")
            continue
            
        if args.skip_eda and "eda" in script:
            print(f"\nSkipping EDA step: {script}")
            continue
        
        success = run_script(script, extra_args)
        if not success:
            print(f"\nPipeline failed at step {i}: {script}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutputs:")
    print("  - Model: models/wildfire_model.txt")
    print("  - Figures: reports/figures/")
    print("  - Data: data/processed/")
    print("\nNext steps:")
    print("  - Review figures in reports/figures/")
    print("  - Make predictions: python scripts/07_predict.py --lat 34.05 --lon -118.24")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

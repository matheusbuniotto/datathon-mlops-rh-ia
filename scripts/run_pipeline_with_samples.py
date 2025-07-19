#!/usr/bin/env python3
"""
Pipeline runner using sample data for demo/testing purposes.

This script runs the ML pipeline using small sample data files instead of
the full dataset, making it suitable for demos, testing, and development.
"""

import os
import shutil
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.pipeline_run_all import main as run_full_pipeline  # noqa: E402


def setup_sample_data():
    """
    Copy sample data files to temporary location for pipeline processing.
    """
    print("🔧 Setting up sample data for pipeline...")
    
    sample_files = {
        "data/raw/sample_applicants.json": "data/raw/applicants.json",
        "data/raw/sample_vagas.json": "data/raw/vagas.json", 
        "data/raw/sample_prospects.json": "data/raw/prospects.json"
    }
    
    # Create backup of original files if they exist
    backups = {}
    for src, dst in sample_files.items():
        if os.path.exists(dst):
            backup_path = f"{dst}.backup"
            shutil.copy2(dst, backup_path)
            backups[dst] = backup_path
            print(f"   📦 Backed up {dst} to {backup_path}")
        
        # Copy sample to main location
        shutil.copy2(src, dst)
        print(f"   📁 Using sample data: {src} -> {dst}")
    
    return backups


def restore_original_data(backups):
    """
    Restore original data files from backups.
    """
    print("🔄 Restoring original data files...")
    
    for original_path, backup_path in backups.items():
        if os.path.exists(backup_path):
            shutil.move(backup_path, original_path)
            print(f"   ✅ Restored {original_path}")


def main():
    """
    Run the pipeline with sample data.
    """
    print("🚀 RecrutaIA Rank - Sample Data Pipeline")
    print("=" * 50)
    
    # Check if sample files exist
    sample_files = [
        "data/raw/sample_applicants.json",
        "data/raw/sample_vagas.json", 
        "data/raw/sample_prospects.json"
    ]
    
    missing_samples = [f for f in sample_files if not os.path.exists(f)]
    if missing_samples:
        print("❌ Missing sample data files:")
        for f in missing_samples:
            print(f"   - {f}")
        print("\nPlease ensure sample data files are available.")
        return 1
    
    backups = {}
    try:
        # Setup sample data
        backups = setup_sample_data()
        
        # Run the pipeline
        print("\n🔄 Running ML pipeline with sample data...")
        run_full_pipeline()
        
        print("\n✅ Sample pipeline completed successfully!")
        print("\nℹ️  Note: This used sample data (100 total records).")
        print("For production, use the full dataset and run: uv run app/pipeline_run_all.py")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1
        
    finally:
        # Always restore original data
        if backups:
            restore_original_data(backups)
    
    return 0


if __name__ == "__main__":
    exit(main())
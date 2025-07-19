#!/usr/bin/env python3
"""
Data Download Script for RecrutaIA Rank

Downloads the real production data from GitHub Releases if not present locally.
Automatically called by the pipeline when data files are missing.
"""

import os
import urllib.request
import sys
from pathlib import Path

# Configuration
GITHUB_REPO = "matheusbuniotto/datathon-mlops-rh-ia"
RELEASE_TAG = "v1.0-data"

# Data files configuration
DATA_FILES = {
    "applicants.json": {
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/applicants.json",
        "size_mb": 194,
        "description": "Candidate profiles and applications"
    },
    "vagas.json": {
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/vagas.json", 
        "size_mb": 37,
        "description": "Job positions and requirements"
    },
    "prospects.json": {
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/prospects.json",
        "size_mb": 21,
        "description": "Additional prospect data"
    }
}

def check_files_exist():
    """Check if all required data files exist."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    
    missing_files = []
    for filename in DATA_FILES.keys():
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    return missing_files

def download_file(filename, config):
    """Download a single file with progress indication."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / filename
    url = config["url"]
    
    print(f"📥 Downloading {filename} ({config['size_mb']}MB)")
    print(f"   {config['description']}")
    print(f"   From: {url}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r   Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, file_path, progress_hook)
        print(f"\n✅ {filename} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        if file_path.exists():
            file_path.unlink()  # Remove partial file
        return False

def download_all_data():
    """Download all missing data files."""
    # Check if release exists (basic check)
    print(f"📡 Configured to download from: {GITHUB_REPO}/releases/{RELEASE_TAG}")
    
    missing_files = check_files_exist()
    
    if not missing_files:
        print("✅ All data files already present")
        return True
    
    print("🚀 RecrutaIA Rank - Data Download")
    print("=" * 50)
    print(f"Missing files: {', '.join(missing_files)}")
    print(f"Total download size: ~{sum(DATA_FILES[f]['size_mb'] for f in missing_files)}MB")
    print()
    
    success_count = 0
    for filename in missing_files:
        if download_file(filename, DATA_FILES[filename]):
            success_count += 1
        print()  # Add spacing between downloads
    
    if success_count == len(missing_files):
        print("🎉 All data files downloaded successfully!")
        print("📊 You can now run the full pipeline with real data.")
        return True
    else:
        print(f"⚠️  Downloaded {success_count}/{len(missing_files)} files")
        print("🔧 Please check your internet connection and GitHub release.")
        return False

def main():
    """Main download function."""
    try:
        return 0 if download_all_data() else 1
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
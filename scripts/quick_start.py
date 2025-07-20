#!/usr/bin/env python3
"""
Quick Start Script for RecrutaIA Rank

This script sets up everything needed to run the demo with sample data:
1. Installs dependencies
2. Runs pipeline with sample data
3. Sets up monitoring
4. Provides next steps

Perfect for new users who just pulled the repository.
"""

import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_uv_available():
    """Check if uv is available."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    """
    Quick start setup process.
    """
    print("🚀 RecrutaIA Rank - Quick Start Setup")
    print("=" * 50)
    print("This script will set up a working demo.")
    print()
    print("📊 Data Options:")
    print("   1. Sample data (100 records) - Fast demo, limited ML performance")
    print("   2. Real data (download from GitHub) - Full dataset, best performance")
    print()

    while True:
        choice = input(
            "Choose data option (1 for sample, 2 for real, or Enter for sample): "
        ).strip()
        if choice in ["", "1"]:
            use_sample_data = True
            break
        elif choice == "2":
            use_sample_data = False
            break
        else:
            print("Please enter 1, 2, or press Enter")

    print()

    # Check if uv is available
    if not check_uv_available():
        print("❌ uv is not installed or not in PATH")
        print(
            "💡 Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
        print("   Or use pip instead: pip install -r requirements-dev.txt")
        return 1

    # Step 1: Install dependencies
    if not run_command("uv sync", "Installing dependencies"):
        print("💡 If uv fails, try: pip install -r requirements-dev.txt")
        return 1

    # Step 2: Install package in development mode
    if not run_command("uv pip install -e .", "Installing package in development mode"):
        return 1

    # Step 3: Run pipeline
    if use_sample_data:
        print("\n📊 Running ML pipeline with sample data...")
        pipeline_cmd = "uv run scripts/run_pipeline_with_samples.py"
        pipeline_desc = "Running sample data pipeline"
    else:
        print("\n📊 Running ML pipeline with real data...")
        pipeline_cmd = "uv run app/pipeline_run_all.py"
        pipeline_desc = "Running full data pipeline (this may take longer)"

    if not run_command(pipeline_cmd, pipeline_desc):
        print("💡 Pipeline failed - you may need to create directories manually")
        print(
            "   Try: mkdir -p data/processed data/embeddings data/model_input data/final"
        )
        if not use_sample_data:
            print("   Or check if data download was configured correctly")
        return 1

    # Step 4: Train model
    model_desc = (
        "Training model with sample data"
        if use_sample_data
        else "Training model with real data"
    )
    if not run_command("uv run app/model/train_ranker.py", model_desc):
        print("💡 Model training failed - check if pipeline completed successfully")
        return 1

    # Step 5: Setup monitoring
    if not run_command(
        "uv run scripts/setup_monitoring.py", "Setting up monitoring components"
    ):
        print("⚠️  Monitoring setup failed - API will work but with limited monitoring")

    print("\n" + "=" * 50)
    print("✅ Quick start setup completed!")
    print("\n🚀 Next steps:")
    print("   1. Start the API:")
    print("      uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("   2. Test the API:")
    print("      curl http://localhost:8000/health")
    print('      curl "http://localhost:8000/v1/list-vagas"')
    print('      curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"')
    print()
    print("   3. Start monitoring (optional):")
    print("      docker-compose up --build")
    print("      Then visit:")
    print("      - Grafana: http://localhost:3000 (no login required)")
    print("      - Prometheus: http://localhost:9090")
    print()
    if use_sample_data:
        print("📝 Note: This setup uses sample data (100 records) for demo purposes.")
        print("   For production, run again and choose option 2 for real data.")
    else:
        print("📝 Note: This setup uses the full production dataset.")
        print("   You now have a complete ML system with real data!")

    return 0


if __name__ == "__main__":
    exit(main())

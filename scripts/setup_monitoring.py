#!/usr/bin/env python3
"""
Setup script for monitoring components.

This script generates the reference profile required for data drift monitoring.
It's automatically run when the API starts if the profile doesn't exist.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_monitoring():
    """
    Set up monitoring components required for the API.
    """
    print("🔧 Setting up monitoring components...")

    # Check if reference profile exists
    profile_path = project_root / "data" / "monitoring" / "reference_profile.json"

    if profile_path.exists():
        print("✅ Reference profile already exists")
        return True

    print("📊 Generating reference profile for data drift monitoring...")
    print("   This may take a few minutes on first run...")

    try:
        # Import and run the data drift profile creation
        from app.monitoring.data_drift import create_reference_profile

        create_reference_profile()
        print("✅ Reference profile created successfully")
        return True

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install dependencies first: uv sync")
        return False

    except FileNotFoundError as e:
        print(f"❌ Missing training data: {e}")
        print("💡 Please run the data pipeline first:")
        print("   uv run scripts/run_pipeline_with_samples.py")
        print("   OR uv run app/pipeline_run_all.py")
        return False

    except Exception as e:
        print(f"❌ Error creating reference profile: {e}")
        return False


def main():
    """
    Main setup function.
    """
    print("🚀 RecrutaIA Rank - Monitoring Setup")
    print("=" * 50)

    success = setup_monitoring()

    if success:
        print("\n✅ Monitoring setup completed!")
        print("🔄 You can now start the API and monitoring stack.")
        return 0
    else:
        print("\n❌ Monitoring setup failed!")
        print("🔧 Please check the error messages above and try again.")
        return 1


if __name__ == "__main__":
    exit(main())

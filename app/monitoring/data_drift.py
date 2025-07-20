import pandas as pd
import json
from loguru import logger
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data/model_input/X_train.npz")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "app/model/features_used_in_training.txt")
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data/monitoring/reference_profile.json")


def get_feature_names_from_txt(file_path: str) -> list:
    """Extracts feature names from a text file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    features = []
    for line in lines:
        if line.startswith("  - "):
            features.append(line.strip("  - \n"))
    return features


def create_reference_profile():
    """
    Creates a data profile from the training data to be used as a reference for drift detection.
    """
    logger.info("Starting reference profile creation...")

    # Load feature names
    feature_names = get_feature_names_from_txt(FEATURES_PATH)

    # Load training data
    # Note: X_train i s a sparse matrix
    from scipy.sparse import load_npz

    X_train_sparse = load_npz(TRAIN_DATA_PATH)

    # Convert to DataFrame for easier profiling
    df_train = pd.DataFrame.sparse.from_spmatrix(X_train_sparse, columns=feature_names)

    profile = {"numerical": {}, "categorical": {}}

    for feature in feature_names:
        # Simple heuristic to differentiate numerical from categorical from one-hot encoded features
        if df_train[feature].nunique() > 2 and (
            df_train[feature].dtype == "float" or df_train[feature].dtype == "int"
        ):
            # Treat as numerical
            profile["numerical"][feature] = {
                "mean": df_train[feature].mean(),
                "std": df_train[feature].std(),
            }
        else:
            # categorical
            profile["categorical"][feature] = (
                df_train[feature].value_counts(normalize=True).to_dict()
            )

    # Ensure the directory for the profile exists
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)

    # Save profile to JSON
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=4)

    logger.success(f"Reference profile created and saved to {PROFILE_PATH}")


if __name__ == "__main__":
    create_reference_profile()

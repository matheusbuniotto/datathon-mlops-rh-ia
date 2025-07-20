from setuptools import setup, find_packages

setup(
    name="datathon-mlops-rh-ia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "loguru",
        "fastapi",
        "uvicorn[standard]",
    ],
)

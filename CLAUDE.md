# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RecrutaIA Rank is an end-to-end MLOps pipeline for ranking job candidates using machine learning. It's built for the Datathon MLOps RH IA challenge and implements a production-ready candidate ranking system with monitoring and evaluation capabilities.

## Development Commands

### Environment Setup
```bash
# Install dependencies (recommended: use uv)
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_loader.py
pytest tests/test_ranking_preparation.py
```

### Model Training and Evaluation
```bash
# Train ranking model with hyperparameter tuning
uv run app/model/train_ranker_tuning.py dev

# Train model with fixed parameters
python app/model/train_ranker.py

# Evaluate trained model
python app/model/evaluate_ranker.py
```

### Data Pipeline
```bash
# Run complete data pipeline (JSON → Parquet → Embeddings → Ranking Dataset)
python app/pipeline_run_all.py

# Run individual pipeline stage
python app/pipeline.py
```

### API Development
```bash
# Run API locally for development
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

# Check API health
curl http://localhost:8000/health

# Example prediction request
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
```

### Docker and Production
```bash
# Build and run all services (API + monitoring)
docker-compose up --build

# Build API service only
docker-compose build api

# Run in detached mode
docker-compose up -d
```

### Code Quality
```bash
# Run linting (Ruff is configured in pyproject.toml)
ruff check .

# Format code
ruff format .
```

## Architecture Overview

The system follows a staged ML pipeline architecture:

### Core Components

1. **Data Pipeline (`app/pipeline.py`)**: Orchestrates the complete data processing flow
   - Raw JSON data → Parquet conversion
   - SQL-based data merging via DuckDB
   - Embedding generation using sentence-transformers
   - Ranking dataset preparation

2. **ML Pipeline Stages (`app/stages/`)**:
   - `embeddings_stage.py`: Generates semantic embeddings for job descriptions and candidate profiles
   - `ranking_preparation_stage.py`: Creates training data for ranking model with relevance targets
   - `feature_engineering_stage.py`: Feature engineering and preprocessing
   - `data_split_stage.py`: Data splitting for training/validation/test

3. **Model Training (`app/model/`)**:
   - `train_ranker.py`: LightGBM ranking model training
   - `train_ranker_tuning.py`: Hyperparameter optimization with Optuna
   - `evaluate_ranker.py`: Model evaluation with ranking metrics (NDCG, MAP)

4. **API Service (`services/api/`)**:
   - FastAPI-based REST API
   - Real-time candidate ranking predictions
   - Prometheus metrics integration
   - Health monitoring

5. **Monitoring Stack (`services/monitoring/`)**:
   - Prometheus for metrics collection
   - Grafana for visualization and dashboards
   - Custom business metrics and data drift monitoring

### Data Flow

```
Raw Data (JSON) → Data Pipeline → Embeddings → Feature Engineering → Model Training → API Deployment
                                     ↓
                              Monitoring & Evaluation
```

### Key Data Artifacts

- `data/processed/merged.parquet`: Unified recruitment data
- `data/embeddings/combined_embeddings.parquet`: Semantic embeddings for all entities
- `data/model_input/`: Preprocessed features ready for model training
- `models/lgbm_ranker.pkl`: Trained LightGBM ranking model

### API Endpoints

- `GET /health`: Service health check
- `GET /v1/recommend_ranked?vaga_id={id}&top_n={n}`: Get ranked candidate recommendations
- `GET /metrics`: Prometheus metrics endpoint

### Monitoring

- **Prometheus**: `http://localhost:9090` (metrics collection)
- **Grafana**: `http://localhost:3000` (admin/admin) (dashboards and visualization)
- Custom metrics for API performance and model predictions

## Important Notes

- The project uses LightGBM for ranking with group-based learning-to-rank
- Embeddings are generated using sentence-transformers (multilingual models)
- All services run in Docker containers for consistent deployment
- DuckDB is used for efficient data processing and SQL operations
- Model evaluation focuses on ranking-specific metrics (NDCG, MAP)
- The system includes data drift monitoring capabilities

## Development Tips

- Use `uv` for faster dependency management when available
- The notebooks in `notebooks/` are for exploration and may need cleanup
- Model artifacts are saved in both `app/model/` and `models/` directories
- All major pipeline stages have corresponding test files
- Ruff is configured to exclude Jupyter notebooks from linting
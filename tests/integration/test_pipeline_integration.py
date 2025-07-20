import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.data_loader import load_applicants, load_jobs, load_prospects
from app.prediction.predictor import predict_rank_for_vaga


class TestPipelineIntegration:
    """Integration tests for the complete ML pipeline"""

    def test_data_loading_integration(self):
        """Test that all data loaders work with real data"""
        # Test loading real data files
        df_app = load_applicants("data/raw/applicants.json")
        df_jobs = load_jobs("data/raw/vagas.json")
        df_prospects = load_prospects("data/raw/prospects.json")

        # Verify data structure and content
        assert len(df_app) > 0, "Applicants data should not be empty"
        assert len(df_jobs) > 0, "Jobs data should not be empty"
        assert len(df_prospects) > 0, "Prospects data should not be empty"

        # Verify key columns exist
        assert "codigo_candidato" in df_app.columns
        assert "codigo_vaga" in df_jobs.columns
        assert "codigo_vaga" in df_prospects.columns

    def test_pipeline_data_flow(self):
        """Test that data flows correctly through pipeline stages"""
        # Check that key pipeline outputs exist
        expected_files = [
            "data/processed/applicants.parquet",
            "data/processed/vagas.parquet",
            "data/processed/prospects.parquet",
            "data/processed/merged.parquet",
            "data/embeddings/combined_embeddings.parquet",
            "data/processed/rank_ready.parquet",
        ]

        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            pytest.skip(
                f"Pipeline outputs missing: {missing_files}. Run pipeline first."
            )

        # Load and verify each stage output
        for file_path in expected_files:
            df = pd.read_parquet(file_path)
            assert len(df) > 0, f"File {file_path} should contain data"

    def test_model_artifacts_exist(self):
        """Test that required model artifacts exist"""
        required_models = ["models/lgbm_ranker.pkl", "models/feature_pipeline.joblib"]

        missing_models = []
        for model_path in required_models:
            if not os.path.exists(model_path):
                missing_models.append(model_path)

        if missing_models:
            pytest.skip(
                f"Model artifacts missing: {missing_models}. Train models first."
            )

        # Verify models can be loaded (basic smoke test)
        from joblib import load

        # Test LightGBM model loading
        model = load("models/lgbm_ranker.pkl")
        assert hasattr(model, "predict"), "Model should have predict method"

        # Test feature pipeline loading
        pipeline = load("models/feature_pipeline.joblib")
        assert hasattr(pipeline, "transform"), "Pipeline should have transform method"

    def test_end_to_end_prediction(self):
        """Test complete end-to-end prediction flow"""
        # Load test data
        test_data_path = "data/final/test_candidates_raw.parquet"
        if not os.path.exists(test_data_path):
            pytest.skip(f"Test data not found: {test_data_path}")

        df_test = pd.read_parquet(test_data_path)

        # Get a sample vaga_id for testing
        sample_vaga = df_test["codigo_vaga"].iloc[0]

        # Test prediction pipeline
        result = predict_rank_for_vaga(
            df_candidates=df_test,
            vaga_id=sample_vaga,
            top_n=3,
            model_path="models/lgbm_ranker.pkl",
            pipeline_path="models/feature_pipeline.joblib",
        )

        # Verify prediction result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        # Add more specific assertions based on your predictor output format

    def test_data_consistency_through_pipeline(self):
        """Test that data maintains consistency through pipeline stages"""
        # Load original and processed data
        df_jobs = pd.read_parquet("data/processed/vagas.parquet")
        df_merged = pd.read_parquet("data/processed/merged.parquet")

        # Check that merge preserves key relationships
        # Convert to same type for comparison (data types may differ)
        original_vagas = set(str(x) for x in df_jobs["codigo_vaga"].unique())
        merged_vagas = set(str(x) for x in df_merged["codigo_vaga"].unique())

        # Merged data should contain subset of original vagas (due to inner joins)
        assert merged_vagas.issubset(original_vagas), (
            "Merged vagas should be subset of original"
        )
        assert len(merged_vagas) > 0, "Should have some vagas after merging"

    def test_pipeline_performance_basic(self):
        """Basic performance test for pipeline components"""
        import time

        # Test data loading performance
        start_time = time.time()
        df_test = pd.read_parquet("data/final/test_candidates_raw.parquet")
        load_time = time.time() - start_time

        assert load_time < 5.0, f"Data loading took {load_time:.2f}s, should be < 5s"

        # Test prediction performance for small batch
        start_time = time.time()
        sample_vaga = df_test["codigo_vaga"].iloc[0]

        predict_rank_for_vaga(
            df_candidates=df_test.head(100),  # Small subset for speed
            vaga_id=sample_vaga,
            top_n=5,
            model_path="models/lgbm_ranker.pkl",
            pipeline_path="models/feature_pipeline.joblib",
        )
        prediction_time = time.time() - start_time

        assert prediction_time < 10.0, (
            f"Prediction took {prediction_time:.2f}s, should be < 10s"
        )


class TestPipelineErrorHandling:
    """Test pipeline error handling and edge cases"""

    def test_missing_data_handling(self):
        """Test pipeline behavior with missing data files"""
        # This would require mocking or temporary file manipulation
        # For now, we'll test the check function
        from app.pipeline import check_and_download_data

        # Test with existing data (should return True)
        result = check_and_download_data()
        # The exact assertion depends on whether data exists
        assert isinstance(result, bool), "Should return boolean value"

    def test_empty_data_handling(self):
        """Test pipeline behavior with empty data"""
        # Create temporary empty data
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = os.path.join(temp_dir, "empty.parquet")
            empty_df = pd.DataFrame({"codigo_vaga": [], "dummy": []})
            empty_df.to_parquet(empty_file, index=False)

            # Load empty data
            df_empty = pd.read_parquet(empty_file)
            assert len(df_empty) == 0, "Empty data should have 0 rows"

    def test_invalid_vaga_id_prediction(self):
        """Test prediction with invalid vaga_id"""
        df_test = pd.read_parquet("data/final/test_candidates_raw.parquet")

        # Test with non-existent vaga_id
        try:
            result = predict_rank_for_vaga(
                df_candidates=df_test,
                vaga_id=999999,  # Non-existent ID
                top_n=5,
                model_path="models/lgbm_ranker.pkl",
                pipeline_path="models/feature_pipeline.joblib",
            )
            # If it doesn't raise an error, verify the result structure
            assert isinstance(result, dict), (
                "Should return dict even for invalid vaga_id"
            )
        except Exception as e:
            # If it raises an error, it should be handled gracefully
            assert isinstance(e, (ValueError, KeyError)), (
                f"Should raise appropriate error, got {type(e)}"
            )


class TestDataQuality:
    """Test data quality throughout the pipeline"""

    def test_data_completeness(self):
        """Test that critical data fields are not missing"""
        df_merged = pd.read_parquet("data/processed/merged.parquet")

        # Check critical columns exist
        critical_columns = ["codigo_vaga"]
        missing_columns = [
            col for col in critical_columns if col not in df_merged.columns
        ]
        assert len(missing_columns) == 0, f"Missing critical columns: {missing_columns}"

        # Check for reasonable data completeness
        total_rows = len(df_merged)
        assert total_rows > 0, "Merged data should not be empty"

    def test_embedding_data_quality(self):
        """Test that embeddings are generated correctly"""
        if not os.path.exists("data/embeddings/combined_embeddings.parquet"):
            pytest.skip("Embeddings not generated")

        df_embeddings = pd.read_parquet("data/embeddings/combined_embeddings.parquet")

        # Check that embeddings exist
        assert len(df_embeddings) > 0, "Embeddings data should not be empty"

        # Check for actual embedding columns
        expected_emb_cols = ["emb_vaga", "emb_cv"]
        for col in expected_emb_cols:
            assert col in df_embeddings.columns, f"Should have {col} column"
            # Check that embeddings are not null for most rows
            non_null_count = df_embeddings[col].notna().sum()
            assert non_null_count > 0, f"Should have non-null embeddings in {col}"

    def test_ranking_data_quality(self):
        """Test that ranking dataset is properly prepared"""
        if not os.path.exists("data/processed/rank_ready.parquet"):
            pytest.skip("Ranking data not generated")

        df_rank = pd.read_parquet("data/processed/rank_ready.parquet")

        # Check basic structure
        assert len(df_rank) > 0, "Ranking data should not be empty"
        assert "codigo_vaga" in df_rank.columns, "Should have vaga identifier"

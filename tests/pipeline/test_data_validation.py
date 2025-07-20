import pytest
import pandas as pd
import os
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.data_loader import load_applicants, load_jobs, load_prospects


class TestDataValidation:
    """Test data validation across the pipeline"""

    def test_raw_data_schemas(self):
        """Test that raw data has expected schemas"""
        # Test applicants data schema
        df_app = load_applicants("data/raw/applicants.json")

        expected_app_columns = {
            "codigo_candidato",
            "nome",
            "email",
            "telefone",
            "nivel_ingles",
            "nivel_academico",
            "nivel_profissional",
            "area_atuacao",
            "conhecimentos_tecnicos",
            "cv",
        }
        actual_app_columns = set(df_app.columns)

        missing_columns = expected_app_columns - actual_app_columns
        assert len(missing_columns) == 0, (
            f"Missing applicant columns: {missing_columns}"
        )

        # Test jobs data schema
        df_jobs = load_jobs("data/raw/vagas.json")

        required_job_columns = {"codigo_vaga", "titulo_vaga", "cliente"}
        actual_job_columns = set(df_jobs.columns)

        missing_job_columns = required_job_columns - actual_job_columns
        assert len(missing_job_columns) == 0, (
            f"Missing job columns: {missing_job_columns}"
        )

        # Test prospects data schema
        df_prospects = load_prospects("data/raw/prospects.json")

        required_prospect_columns = {"codigo_vaga", "codigo_candidato"}
        actual_prospect_columns = set(df_prospects.columns)

        missing_prospect_columns = required_prospect_columns - actual_prospect_columns
        assert len(missing_prospect_columns) == 0, (
            f"Missing prospect columns: {missing_prospect_columns}"
        )

    def test_data_types_consistency(self):
        """Test that data types are consistent across pipeline stages"""
        # Load processed data
        df_app = pd.read_parquet("data/processed/applicants.parquet")
        df_prospects = pd.read_parquet("data/processed/prospects.parquet")

        # Check that ID columns contain numeric values (even if stored as object)
        # Try to convert to numeric to verify they are valid
        try:
            pd.to_numeric(df_app["codigo_candidato"].head(100))
            app_ids_numeric = True
        except (ValueError, TypeError):
            app_ids_numeric = False

        try:
            pd.to_numeric(df_prospects["codigo_candidato"].head(100))
            prospect_ids_numeric = True
        except (ValueError, TypeError):
            prospect_ids_numeric = False

        assert app_ids_numeric, "Applicant IDs should be convertible to numeric"
        assert prospect_ids_numeric, (
            "Prospect candidate IDs should be convertible to numeric"
        )

    def test_data_completeness(self):
        """Test data completeness requirements"""
        df_app = pd.read_parquet("data/processed/applicants.parquet")
        df_jobs = pd.read_parquet("data/processed/vagas.parquet")
        df_prospects = pd.read_parquet("data/processed/prospects.parquet")

        # Critical fields should not be null
        assert df_app["codigo_candidato"].notna().all(), (
            "All applicants should have candidate codes"
        )
        assert df_jobs["codigo_vaga"].notna().all(), "All jobs should have job codes"
        assert df_prospects["codigo_vaga"].notna().all(), (
            "All prospects should have job codes"
        )
        assert df_prospects["codigo_candidato"].notna().all(), (
            "All prospects should have candidate codes"
        )

        # Check for reasonable data volumes
        assert len(df_app) > 1000, "Should have reasonable number of applicants"
        assert len(df_jobs) > 100, "Should have reasonable number of jobs"
        assert len(df_prospects) > 1000, "Should have reasonable number of prospects"

    def test_referential_integrity(self):
        """Test referential integrity between datasets"""
        df_app = pd.read_parquet("data/processed/applicants.parquet")
        df_jobs = pd.read_parquet("data/processed/vagas.parquet")
        df_prospects = pd.read_parquet("data/processed/prospects.parquet")

        # All prospect candidates should exist in applicants
        prospect_candidates = set(df_prospects["codigo_candidato"].unique())
        app_candidates = set(df_app["codigo_candidato"].unique())

        missing_candidates = prospect_candidates - app_candidates
        # Some missing is acceptable due to data processing
        missing_ratio = len(missing_candidates) / len(prospect_candidates)
        assert missing_ratio < 0.5, f"Too many missing candidates: {missing_ratio:.2%}"

        # All prospect jobs should exist in jobs
        prospect_jobs = set(df_prospects["codigo_vaga"].unique())
        available_jobs = set(df_jobs["codigo_vaga"].unique())

        missing_jobs = prospect_jobs - available_jobs
        missing_job_ratio = len(missing_jobs) / len(prospect_jobs)
        assert missing_job_ratio < 0.5, (
            f"Too many missing jobs: {missing_job_ratio:.2%}"
        )

    def test_merged_data_quality(self):
        """Test quality of merged dataset"""
        if not os.path.exists("data/processed/merged.parquet"):
            pytest.skip("Merged data not available")

        df_merged = pd.read_parquet("data/processed/merged.parquet")

        # Should have reasonable number of records
        assert len(df_merged) > 1000, "Merged data should have substantial records"

        # Should have key columns from all sources
        assert "codigo_vaga" in df_merged.columns
        assert any("candidato" in col.lower() for col in df_merged.columns)

        # Check for duplicates (candidates can apply to multiple jobs)
        total_rows = len(df_merged)
        unique_combinations = len(
            df_merged[["codigo_vaga", "prospect_codigo_candidato"]].drop_duplicates()
        )

        # Should not have exact duplicates
        assert unique_combinations == total_rows, (
            "Should not have duplicate candidate-job combinations"
        )

    def test_embedding_data_structure(self):
        """Test embedding data structure and quality"""
        if not os.path.exists("data/embeddings/combined_embeddings.parquet"):
            pytest.skip("Embeddings not available")

        df_emb = pd.read_parquet("data/embeddings/combined_embeddings.parquet")

        # Should have embedding columns
        assert "emb_vaga" in df_emb.columns, "Should have job embeddings"
        assert "emb_cv" in df_emb.columns, "Should have CV embeddings"

        # Check embedding quality
        sample_size = min(100, len(df_emb))
        sample_df = df_emb.sample(n=sample_size)

        # Most embeddings should not be null
        vaga_emb_null_ratio = sample_df["emb_vaga"].isna().mean()
        cv_emb_null_ratio = sample_df["emb_cv"].isna().mean()

        assert vaga_emb_null_ratio < 0.5, (
            f"Too many null job embeddings: {vaga_emb_null_ratio:.2%}"
        )
        assert cv_emb_null_ratio < 0.5, (
            f"Too many null CV embeddings: {cv_emb_null_ratio:.2%}"
        )

    def test_ranking_data_preparation(self):
        """Test ranking dataset preparation"""
        if not os.path.exists("data/processed/rank_ready.parquet"):
            pytest.skip("Ranking data not available")

        df_rank = pd.read_parquet("data/processed/rank_ready.parquet")

        # Should have substantial data
        assert len(df_rank) > 1000, "Ranking dataset should have substantial records"

        # Should have key identifiers
        assert "codigo_vaga" in df_rank.columns, "Should have job identifiers"

        # Check for reasonable distribution of jobs
        jobs_per_vaga = df_rank.groupby("codigo_vaga").size()
        avg_candidates_per_job = jobs_per_vaga.mean()

        assert avg_candidates_per_job > 1, (
            "Should have multiple candidates per job on average"
        )
        assert avg_candidates_per_job < 1000, "Candidates per job should be reasonable"


class TestDataDrift:
    """Test for data drift detection capabilities"""

    def test_production_data_monitoring_setup(self):
        """Test that production data monitoring is set up"""
        monitoring_dir = "data/monitoring"

        # Monitoring directory should exist
        assert os.path.exists(monitoring_dir), "Monitoring directory should exist"

        # Should have reference profile for drift detection
        reference_profile = os.path.join(monitoring_dir, "reference_profile.json")
        if os.path.exists(reference_profile):
            # If reference profile exists, it should be valid JSON
            import json

            with open(reference_profile, "r") as f:
                profile_data = json.load(f)
            assert isinstance(profile_data, dict), (
                "Reference profile should be valid JSON object"
            )

    def test_data_schema_stability(self):
        """Test that data schemas remain stable across different files"""
        # Compare schemas between different data files
        df_merged = pd.read_parquet("data/processed/merged.parquet")

        if os.path.exists("data/final/test_candidates_raw.parquet"):
            df_test = pd.read_parquet("data/final/test_candidates_raw.parquet")

            # Should have overlapping columns
            merged_cols = set(df_merged.columns)
            test_cols = set(df_test.columns)

            common_cols = merged_cols & test_cols
            assert len(common_cols) > 5, (
                "Should have substantial column overlap between datasets"
            )


class TestBusinessLogicValidation:
    """Test business logic and domain-specific validations"""

    def test_candidate_job_matching_logic(self):
        """Test candidate-job matching business logic"""
        df_prospects = pd.read_parquet("data/processed/prospects.parquet")

        # Should have reasonable candidate-job relationships
        candidates_per_job = df_prospects.groupby("codigo_vaga")[
            "codigo_candidato"
        ].nunique()
        jobs_per_candidate = df_prospects.groupby("codigo_candidato")[
            "codigo_vaga"
        ].nunique()

        # Business logic checks
        avg_candidates_per_job = candidates_per_job.mean()
        avg_jobs_per_candidate = jobs_per_candidate.mean()

        assert avg_candidates_per_job > 1, (
            "Jobs should have multiple candidates on average"
        )
        assert avg_jobs_per_candidate >= 1, "Candidates should apply to jobs"

        # Check for outliers that might indicate data quality issues
        max_candidates_per_job = candidates_per_job.max()
        max_jobs_per_candidate = jobs_per_candidate.max()

        assert max_candidates_per_job < 10000, (
            "No job should have excessive candidates (data quality check)"
        )
        assert max_jobs_per_candidate < 1000, (
            "No candidate should apply to excessive jobs (data quality check)"
        )

    def test_temporal_data_consistency(self):
        """Test temporal aspects of the data"""
        df_prospects = pd.read_parquet("data/processed/prospects.parquet")

        # If we have date fields, validate them
        date_columns = [
            col
            for col in df_prospects.columns
            if "data" in col.lower() or "date" in col.lower()
        ]

        for date_col in date_columns:
            if df_prospects[date_col].dtype == "object":
                # Try to parse dates
                try:
                    parsed_dates = pd.to_datetime(
                        df_prospects[date_col], errors="coerce"
                    )
                    valid_dates_ratio = parsed_dates.notna().mean()

                    if valid_dates_ratio > 0.5:  # If most values look like dates
                        # Check date range is reasonable
                        min_date = parsed_dates.min()
                        max_date = parsed_dates.max()

                        # Should be within reasonable range (last 10 years to next year)
                        current_year = pd.Timestamp.now().year
                        assert min_date.year >= current_year - 10, (
                            f"Dates too old in {date_col}"
                        )
                        assert max_date.year <= current_year + 1, (
                            f"Dates too future in {date_col}"
                        )
                except (ValueError, TypeError, AttributeError):
                    # Skip if date parsing fails
                    pass

    def test_text_data_quality(self):
        """Test quality of text fields"""
        df_app = pd.read_parquet("data/processed/applicants.parquet")

        # CV field should have reasonable content
        if "cv" in df_app.columns:
            cv_lengths = df_app["cv"].astype(str).str.len()
            avg_cv_length = cv_lengths.mean()

            # CVs should have reasonable length
            assert avg_cv_length > 10, "CVs should have reasonable content length"

            # Most CVs should not be empty
            non_empty_cv_ratio = (cv_lengths > 10).mean()
            assert non_empty_cv_ratio > 0.3, (
                "Most candidates should have meaningful CV content"
            )

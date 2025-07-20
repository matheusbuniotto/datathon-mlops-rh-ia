from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint"""

    def test_health_endpoint_returns_ok(self):
        """Test that health endpoint returns 200 and correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestListVagasEndpoint:
    """Test the list-vagas endpoint"""

    def test_list_vagas_success(self):
        """Test successful list-vagas call with real data"""
        response = client.get("/v1/list-vagas")
        assert response.status_code == 200

        data = response.json()
        assert "total_vagas" in data
        assert "vaga_ids" in data
        assert isinstance(data["total_vagas"], int)
        assert isinstance(data["vaga_ids"], list)
        assert data["total_vagas"] > 0
        assert len(data["vaga_ids"]) == data["total_vagas"]

    @patch("pandas.read_parquet")
    def test_list_vagas_file_not_found(self, mock_read_parquet):
        """Test list-vagas returns 404 when data file is missing"""
        mock_read_parquet.side_effect = FileNotFoundError("File not found")

        response = client.get("/v1/list-vagas")
        assert response.status_code == 404
        assert "Arquivo de dados não encontrado" in response.json()["detail"]

    @patch("pandas.read_parquet")
    def test_list_vagas_general_error(self, mock_read_parquet):
        """Test list-vagas returns 500 on general errors"""
        mock_read_parquet.side_effect = Exception("Database error")

        response = client.get("/v1/list-vagas")
        assert response.status_code == 500
        assert "Erro ao listar vagas" in response.json()["detail"]


class TestRecommendRankedEndpoint:
    """Test the recommend_ranked endpoint"""

    def test_recommend_ranked_success(self):
        """Test successful recommendation with real data"""
        # Use a known vaga_id from our test data
        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=3")
        assert response.status_code == 200

        data = response.json()
        # Verify response structure based on predictor output
        assert isinstance(data, dict)
        # The exact structure depends on predict_rank_for_vaga output

    def test_recommend_ranked_with_different_top_n(self):
        """Test recommendation with different top_n values"""
        for top_n in [1, 5, 10]:
            response = client.get(f"/v1/recommend_ranked?vaga_id=1650&top_n={top_n}")
            assert response.status_code == 200

    def test_recommend_ranked_missing_vaga_id(self):
        """Test that missing vaga_id parameter returns 422"""
        response = client.get("/v1/recommend_ranked?top_n=5")
        assert response.status_code == 422

    def test_recommend_ranked_invalid_vaga_id_type(self):
        """Test that non-integer vaga_id returns 422"""
        response = client.get("/v1/recommend_ranked?vaga_id=invalid&top_n=5")
        assert response.status_code == 422

    def test_recommend_ranked_invalid_top_n_type(self):
        """Test that non-integer top_n returns 422"""
        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=invalid")
        assert response.status_code == 422

    def test_recommend_ranked_default_top_n(self):
        """Test that top_n defaults to 5 when not provided"""
        response = client.get("/v1/recommend_ranked?vaga_id=1650")
        assert response.status_code == 200

    @patch("pandas.read_parquet")
    def test_recommend_ranked_file_not_found(self, mock_read_parquet):
        """Test recommend_ranked returns 404 when files are missing"""
        mock_read_parquet.side_effect = FileNotFoundError("File not found")

        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=5")
        assert response.status_code == 404
        assert "Arquivos necessários não encontrados" in response.json()["detail"]

    @patch("services.api.routes.predict_rank_for_vaga")
    def test_recommend_ranked_prediction_error(self, mock_predict):
        """Test recommend_ranked returns 500 on prediction errors"""
        mock_predict.side_effect = Exception("Model prediction failed")

        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=5")
        assert response.status_code == 500
        assert "Erro ao processar requisição" in response.json()["detail"]


class TestAPIMetrics:
    """Test that API metrics endpoints work"""

    def test_metrics_endpoint_accessible(self):
        """Test that /metrics endpoint is accessible"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should contain Prometheus metrics format
        assert "api_requests_total" in response.text


class TestAPIIntegration:
    """Integration tests using real data flow"""

    def test_full_workflow_integration(self):
        """Test complete workflow: list vagas -> get recommendation"""
        # First get available vagas
        vagas_response = client.get("/v1/list-vagas")
        assert vagas_response.status_code == 200

        vagas_data = vagas_response.json()
        available_vagas = vagas_data["vaga_ids"]
        assert len(available_vagas) > 0

        # Use first available vaga for recommendation
        test_vaga = available_vagas[0]
        rec_response = client.get(f"/v1/recommend_ranked?vaga_id={test_vaga}&top_n=3")
        assert rec_response.status_code == 200

    def test_edge_case_large_top_n(self):
        """Test behavior with very large top_n values"""
        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=10000")
        # Should still work, just return available candidates
        assert response.status_code == 200

    def test_edge_case_zero_top_n(self):
        """Test behavior with zero top_n"""
        response = client.get("/v1/recommend_ranked?vaga_id=1650&top_n=0")
        # Should work (empty result or handle gracefully)
        assert response.status_code == 200

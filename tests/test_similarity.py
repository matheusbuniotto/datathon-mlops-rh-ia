import pandas as pd
import pytest
from app.similarity import recommend_candidates_for_vaga


@pytest.fixture
def sample_data():
    # Simula DF com embeddings simples
    return pd.DataFrame({
        "codigo_vaga": [1]*3,
        "codigo_candidato": [101, 102, 103],
        "nome_candidato": ["Ana", "Bruno", "Clara"],
        "cv": ["engenheiro de dados"]*3,
        "applicants_codigo_candidato": [101, 102, 103],
        "emb_vaga": [[0.1]*384]*3,
        "emb_cv": [[0.1]*384, [0.2]*384, [0.3]*384],
        "nivel_profissional": ["Pleno", "Sênior", "Júnior"],
        "emb_vaga_areas_atuacao": [[0.1]*384]*3,
        "emb_candidato_area_atuacao": [[0.1]*384, [0.2]*384, [0.3]*384],  
        "candidato_area_atuacao": ["Dados", "Infra", "Dados"],        
        "email": ["ana@gmail.com", "bruno@hotmail.com", "clara@uol.com"],
        "similaridade_vaga_cv": [0.9, 0.8, 0.7],
        "similaridade_area": [0.85, 0.75, 0.65],
        "similaridade_combinada": [0.875, 0.775, 0.675],
        "similaridade": [0.9, 0.8, 0.7]
    })


def test_recommendation_top_1(sample_data):
    result = recommend_candidates_for_vaga(sample_data, codigo_vaga=1, top_n=1)
    assert not result.empty
    assert result.iloc[0]["applicants_codigo_candidato"] in [101, 102, 103]
    assert "similaridade_combinada" in result.columns


def test_invalid_vaga_raises(sample_data):
    with pytest.raises(ValueError):
        recommend_candidates_for_vaga(sample_data, codigo_vaga=999, top_n=1)



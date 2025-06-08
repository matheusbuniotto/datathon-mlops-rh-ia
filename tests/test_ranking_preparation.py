import pandas as pd
from app.stages.ranking_preparation_stage import prepare_ranking_dataset

def test_prepare_ranking_dataset():
    df = pd.DataFrame({
        "situacao_candidato": [
            "Contratado pela Decision",
            "Desistiu",
            "Entrevista com Cliente",
            None
        ]
    })

    df_out = prepare_ranking_dataset(df)
    assert "target_rank" in df_out.columns
    assert df_out["target_rank"].tolist() == [5, 0, 3, -1]

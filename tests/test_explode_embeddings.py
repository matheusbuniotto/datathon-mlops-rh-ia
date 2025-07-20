import numpy as np
import pandas as pd
from app.utils.embedding_utils import explode_embeddings


def test_explode_embeddings():
    df = pd.DataFrame(
        {
            "emb_vaga": [np.array([1, 2]), np.array([3, 4])],
            "emb_cv": [np.array([5, 6]), np.array([7, 8])],
            "emb_vaga_areas_atuacao": [np.array([9, 10]), np.array([11, 12])],
            "emb_candidato_area_atuacao": [np.array([13, 14]), np.array([15, 16])],
        }
    )

    df_out = explode_embeddings(df)
    assert df_out.shape[1] == 8
    assert "emb_cv_1" in df_out.columns

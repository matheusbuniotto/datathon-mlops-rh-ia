import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


def recommend_candidates_for_vaga(df: pd.DataFrame, codigo_vaga: int, top_n: int = 7) -> pd.DataFrame:
    """
    Calcula a similaridade entre uma vaga e todos os candidatos e retorna os top_n recomendados.
    """

    logger.info(f"Buscando vaga com id {codigo_vaga}")
    vaga_row = df[df["codigo_vaga"] == codigo_vaga]

    if vaga_row.empty:
        logger.warning(f"Vaga {codigo_vaga} não encontrada.")
        raise ValueError(f"Vaga {codigo_vaga} não encontrada.")

    emb_vaga = np.vstack(vaga_row["emb_vaga"].values) # type: ignore
    emb_cv = np.vstack(df["emb_cv"].values) # type: ignore
    emb_vaga_atuacao = np.vstack(vaga_row["emb_vaga_areas_atuacao"].values) # type: ignore
    emb_candidato_atuacao = np.vstack(df["emb_candidato_area_atuacao"].values) # type: ignore

    logger.debug(f"Calculando similaridade entre {codigo_vaga} e {len(emb_cv)} candidatos.")
    similarities = cosine_similarity(emb_vaga, emb_cv)[0]
    similarities_area = cosine_similarity(emb_vaga_atuacao, emb_candidato_atuacao)[0]

    df_result = df.copy()
    df_result["similaridade_vaga_cv"] = similarities
    df_result["similaridade_area"] = similarities_area

    # Combina as similaridades com pesos
    peso_cv = 0.9
    peso_area = 0.1
    df_result["similaridade_combinada"] = (
        peso_cv * df_result["similaridade_vaga_cv"] + peso_area * df_result["similaridade_area"]
    )

    # Ordena pela similaridade combinada
    df_result = df_result.sort_values(by="similaridade_combinada", ascending=False)
    df_result = df_result.drop_duplicates(subset=["email"])
    df_top = df_result.head(top_n)

    logger.success(f"Top-{top_n} candidatos recomendados para vaga {codigo_vaga}.")

    return df_top[[
        
        "codigo_vaga", "applicants_codigo_candidato", "nome_candidato", "email", "nivel_profissional","vaga_areas_atuacao",
        "candidato_area_atuacao", "similaridade_vaga_cv", "cv", "similaridade_area", "similaridade_combinada"

    ]]

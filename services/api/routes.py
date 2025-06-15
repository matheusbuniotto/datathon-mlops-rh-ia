# services/api/routes.py
from fastapi import APIRouter, Query
from services.api.model_loader import load_model
import pandas as pd
import duckdb
from loguru import logger
from joblib import load
from app.utils.embedding_utils import explode_embeddings
import numpy as np

router = APIRouter()

@router.get("/recommend_ranked")
def recommend_ranked(vaga_id: int = Query(...), top_n: int = Query(5)):
    logger.info(f"[API] Requisição recebida: vaga_id={vaga_id}, top_n={top_n}")

    try:
        df = pd.read_parquet("data/final/test_candidates_raw.parquet")
    except FileNotFoundError:
        logger.error("[API] Arquivo test_candidates_raw.parquet não encontrado.")
        return {"erro": "Arquivo de candidatos não encontrado."}

    df = df[df["codigo_vaga"] == vaga_id]

    if df.empty:
        logger.warning(f"[API] Nenhum candidato encontrado para a vaga {vaga_id}.")
        return {"vaga_id": vaga_id, "candidatos": []}

    # Carrega modelo e pipeline
    model = load_model()
    pipe = load("models/feature_pipeline.joblib")

    # Explode os embeddings
    df = explode_embeddings(df)

    # Garante consistência com as features usadas no treino
    X = pipe.transform(df)
    scores = model.predict(X)

    df["score"] = scores
    df_ranked = df.sort_values("score", ascending=False).head(top_n)

    candidatos = df_ranked[["codigo", "nome", "email", "score"]].to_dict(orient="records")
    return {"vaga_id": vaga_id, "candidatos": candidatos}

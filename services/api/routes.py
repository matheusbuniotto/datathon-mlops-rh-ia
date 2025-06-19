# services/api/routes.py
from fastapi import APIRouter, Query, HTTPException
from services.api.model_loader import load_model
import pandas as pd
from loguru import logger
from app.prediction.predictor import predict_rank_for_vaga

router = APIRouter()

@router.get("/recommend_ranked")
def recommend_ranked(vaga_id: int = Query(...), top_n: int = Query(5)):
    logger.info(f"[API] Requisição recebida: vaga_id={vaga_id}, top_n={top_n}")

    try:
        # Carrega os dados
        df = pd.read_parquet("data/final/test_candidates_raw.parquet")
        
        # Usa a função de predição
        resultado = predict_rank_for_vaga(
            df_candidates=df,
            vaga_id=vaga_id,
            top_n=top_n,
            model_path="models/lgbm_ranker.pkl",
            pipeline_path="models/feature_pipeline.joblib"
        )
        
        return resultado

    except FileNotFoundError:
        logger.error("[API] Arquivo de dados ou modelo não encontrado.")
        raise HTTPException(
            status_code=404,
            detail="Arquivos necessários não encontrados."
        )
    except Exception as e:
        logger.error(f"[API] Erro ao processar requisição: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar requisição: {str(e)}"
        )

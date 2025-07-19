# services/api/routes.py
from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from loguru import logger
from app.prediction.predictor import predict_rank_for_vaga
import os

router = APIRouter()

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

@router.get("/list-vagas")
def list_vagas():
    """
    Returns a list of all available vaga_ids that can be used in the recommend_ranked endpoint.
    """
    logger.info("[API] Requisição recebida para listagem de vagas")
    
    try:
        # Use absolute paths based on project root
        data_path = os.path.join(PROJECT_ROOT, "data/final/test_candidates_raw.parquet")
        
        logger.info(f"Loading data from: {data_path}")
        
        # Load the data
        df = pd.read_parquet(data_path)
        
        # Get unique vaga IDs
        unique_vagas = sorted(df['codigo_vaga'].unique().tolist())
        
        logger.success(f"[API] Found {len(unique_vagas)} unique vagas")
        
        return {
            "total_vagas": len(unique_vagas),
            "vaga_ids": unique_vagas
        }
        
    except FileNotFoundError as e:
        logger.error(f"[API] Arquivo não encontrado: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo de dados não encontrado: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[API] Erro ao listar vagas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar vagas: {str(e)}"
        )

@router.get("/recommend_ranked")
def recommend_ranked(vaga_id: int = Query(...), top_n: int = Query(5)):
    logger.info(f"[API] Requisição recebida: vaga_id={vaga_id}, top_n={top_n}")

    try:
        # Use absolute paths based on project root
        data_path = os.path.join(PROJECT_ROOT, "data/final/test_candidates_raw.parquet")
        model_path = os.path.join(PROJECT_ROOT, "models/lgbm_ranker.pkl")
        pipeline_path = os.path.join(PROJECT_ROOT, "models/feature_pipeline.joblib")

        logger.info(f"Loading data from: {data_path}")
        
        # Carrega os dados
        df = pd.read_parquet(data_path)
        
        # Log production data for monitoring
        monitoring_data_path = os.path.join(PROJECT_ROOT, "data/monitoring")
        os.makedirs(monitoring_data_path, exist_ok=True)
        df.to_parquet(os.path.join(monitoring_data_path, "production_data.parquet"))
        
        # Usa a função de predição
        resultado = predict_rank_for_vaga(
            df_candidates=df,
            vaga_id=vaga_id,
            top_n=top_n,
            model_path=model_path,
            pipeline_path=pipeline_path
        )
        
        return resultado

    except FileNotFoundError as e:
        logger.error(f"[API] Arquivo não encontrado: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise HTTPException(
            status_code=404,
            detail=f"Arquivos necessários não encontrados. Path tentado: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[API] Erro ao processar requisição: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar requisição: {str(e)}"
        )

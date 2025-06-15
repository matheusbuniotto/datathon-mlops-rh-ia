from joblib import load
from functools import lru_cache
from loguru import logger

MODEL_PATH_PROD = "models/lgbm_ranker.pkl"

@lru_cache()
def load_model():
    logger.info("[Model] Carregando modelo LightGBM rankeador...")
    model = load(MODEL_PATH_PROD)
    logger.success("[Model] Modelo carregado com sucesso.")
    return model

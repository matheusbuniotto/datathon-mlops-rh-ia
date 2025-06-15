from joblib import dump
from app.stages.feature_engineering_stage import apply_feature_pipeline
from app.stages.data_split_stage import split_dataset_by_vaga
import pandas as pd
import os
from loguru import logger

def save_dataset_for_prediction():
    df = pd.read_parquet("data/processed/rank_ready.parquet")
    df_train, df_val, df_test = split_dataset_by_vaga(df)

    # Aplica pipeline e salva pipeline treinado
    logger.info("[SavePipeline] Gerando e salvando pipeline de features...")
    X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe = apply_feature_pipeline(
        df_train, df_val, df_test
    )

    os.makedirs("models", exist_ok=True)
    dump(pipe, "models/feature_pipeline.joblib")
    logger.success("[SavePipeline] Pipeline salvo em models/feature_pipeline.joblib")

    # Salva também o dataset de teste final, com as features brutas + embeddings explodidos
    df_test.to_parquet("data/final/test_candidates_raw.parquet", index=False)
    logger.success("[SavePipeline] df_test salvo em data/final/test_candidates_raw.parquet")

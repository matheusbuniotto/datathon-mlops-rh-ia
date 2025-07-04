import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from joblib import load
from loguru import logger
from typing import Dict, Any
from app.utils.embedding_utils import explode_embeddings
from app.stages.feature_engineering_stage import calculate_area_similarity
from app.utils.strings_clean_utils import clean_area_atuacao
import json
from scipy.stats import ks_2samp, chi2_contingency
from prometheus_client import Gauge

# Prometheus metrics for data drift
DATA_DRIFT_GAUGE = Gauge(
    "data_drift_p_value",
    "P-value for data drift detection tests",
    ["feature", "test_type"]
)

def detect_data_drift(df_new: pd.DataFrame, reference_profile_path: str):
    """
    Detects data drift by comparing the new data with a reference profile.
    """
    logger.info("Checking for data drift...")
    with open(reference_profile_path, 'r') as f:
        reference_profile = json.load(f)

    for feature, profile in reference_profile["numerical"].items():
        if feature in df_new.columns:
            p_value = ks_2samp(df_new[feature], np.random.normal(profile["mean"], profile["std"], len(df_new[feature])))[1]
            DATA_DRIFT_GAUGE.labels(feature=feature, test_type="ks").set(p_value)
            if p_value < 0.05:
                logger.warning(f"Data drift detected for numerical feature '{feature}' (p-value: {p_value:.4f})")

    for feature, profile in reference_profile["categorical"].items():
        if feature in df_new.columns:
            # Create contingency table
            observed_counts = df_new[feature].value_counts().reindex(profile.keys(), fill_value=0)
            expected_counts = pd.Series(profile) * len(df_new)
            contingency_table = pd.DataFrame([observed_counts, expected_counts]).T
            
            # Chi-squared test
            _, p_value, _, _ = chi2_contingency(contingency_table)
            DATA_DRIFT_GAUGE.labels(feature=feature, test_type="chi2").set(p_value)
            if p_value < 0.05:
                logger.warning(f"Data drift detected for categorical feature '{feature}' (p-value: {p_value:.4f})")

def predict_rank_for_vaga(df_candidates: pd.DataFrame, 
                         vaga_id: int, 
                         top_n: int = 5, 
                         model_path: str = "models/lgbm_ranker.pkl",
                         pipeline_path: str = "models/feature_pipeline.joblib") -> Dict[str, Any]:
    """
    Faz o predict e retorna o ranking de candidatos para uma vaga específica.
    
    Args:
        df_candidates: DataFrame com os candidatos a serem ranqueados
        vaga_id: ID da vaga para filtrar os candidatos
        top_n: Número de candidatos a retornar no ranking
        model_path: Caminho para o modelo treinado
        pipeline_path: Caminho para o pipeline de features
        
    Returns:
        Dicionário com o ID da vaga e a lista de candidatos ranqueados
    """
    try:
        # Filtra candidatos para a vaga específica
        df_vaga = df_candidates[df_candidates["codigo_vaga"] == vaga_id].copy()
        
        if df_vaga.empty:
            logger.warning(f"Nenhum candidato encontrado para a vaga {vaga_id}.")
            return {"vaga_id": vaga_id, "candidatos": []}
        
        # Carrega modelo e pipeline
        logger.info("Carregando modelo e pipeline...")
        model = load(model_path)
        pipe = load(pipeline_path)
        
        # Pré-processamento dos dados
        logger.info("Pré-processando os dados...")
        
        # 1. Expande embeddings
        df_vaga = explode_embeddings(df_vaga)
        
        # 2. Preenche valores faltantes
        for col in df_vaga.columns:
            if pd.api.types.is_numeric_dtype(df_vaga[col]):
                df_vaga[col] = df_vaga[col].fillna(-999)
            else:
                df_vaga[col] = df_vaga[col].fillna("Indefinido")
        
        # 3. Limpa áreas de atuação
        df_vaga['candidato_area_atuacao'] = clean_area_atuacao(df_vaga, 'candidato_area_atuacao')
        df_vaga['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_vaga, 'vaga_areas_atuacao')
        
        # 4. Calcula similaridade de área
        df_vaga = calculate_area_similarity(df_vaga)

        # 5. Aplica transformações do pipeline
        logger.info("Aplicando transformações do pipeline...")
        X_processed = pipe.transform(df_vaga)
        
        # Data Drift Detection
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        reference_profile_path = os.path.join(project_root, "data/monitoring/reference_profile.json")
        detect_data_drift(pd.DataFrame(X_processed, columns=pipe.get_feature_names_out()), reference_profile_path)

        # 6. Faz predições
        logger.info("Fazendo predições...")
        predictions = model.predict(X_processed)
        
        # 7. Cria DataFrame com resultados
        results_df = pd.DataFrame({
            'nome_candidato': df_vaga['nome_candidato'],
            'score': np.round(predictions, 3)
        })

        # Ordena por score decrescente e atribui rank (1 para maior score)
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
        results_df['rank'] = results_df.index + 1

        # 8. Pega top N candidatos
        top_candidates = results_df.head(top_n).to_dict('records')
        logger.success(f"Ranking gerado com sucesso para vaga {vaga_id}")
        return {
            "vaga_id": vaga_id,
            "candidatos": top_candidates
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar ranking para vaga {vaga_id}: {str(e)}")
        raise e

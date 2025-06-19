import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from joblib import load
from loguru import logger
from typing import List, Dict, Any
from app.utils.embedding_utils import explode_embeddings
from app.stages.feature_engineering_stage import calculate_area_similarity
from app.utils.strings_clean_utils import clean_area_atuacao

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
        
        # 6. Faz predições
        logger.info("Fazendo predições...")
        predictions = model.predict(X_processed)
        
        # 7. Cria DataFrame com resultados
        results_df = pd.DataFrame({
            'nome_candidato': df_vaga['nome_candidato'],
            'score': np.round(predictions, 3),
            'rank': predictions.argsort() + 1 
        })
        
        # 8. Ordena por score e pega top N candidatos
        top_candidates = (results_df.sort_values('score', ascending=False)
                        .head(top_n)
                        .to_dict('records'))
        
        logger.success(f"Ranking gerado com sucesso para vaga {vaga_id}")
        return {
            "vaga_id": vaga_id,
            "candidatos": top_candidates
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar ranking para vaga {vaga_id}: {str(e)}")
        raise e

# Add this at the bottom for testing:
if __name__ == "__main__":
    # Test the predictor
    data_path = "data/final/test_candidates_raw.parquet"
    test_df = pd.read_parquet(data_path)
    vaga_id_teste = test_df['codigo_vaga'].iloc[0]
    
    resultado = predict_rank_for_vaga(
        df_candidates=test_df,
        vaga_id=vaga_id_teste,
    )
    print(resultado)
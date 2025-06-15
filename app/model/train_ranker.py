import lightgbm as lgb
import numpy as np
from scipy import sparse
import os
from joblib import dump
from loguru import logger
import json

DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"
MODEL_PATH_PROD = "models/lgbm_ranker.pkl"
PIPELINE_PATH = "data/model_input/preprocessing_pipeline.pkl"


def load_feature_names(path):
    """
    Carrega os nomes das features salvos durante o preprocessing
    """
    try:
        # Tentar carregar do arquivo JSON primeiro (mais legível)
        json_path = os.path.join(path, "feature_names.json")
        if os.path.exists(json_path):
            logger.info("[LightGBM] Carregando nomes das features do arquivo JSON...")
            with open(json_path, "r", encoding='utf-8') as f:
                feature_names = json.load(f)
            logger.success(f"[LightGBM] {len(feature_names)} nomes de features carregados do JSON")
            return feature_names
        
        # Fallback para arquivo numpy
        npy_path = os.path.join(path, "feature_names.npy")
        if os.path.exists(npy_path):
            logger.info("[LightGBM] Carregando nomes das features do arquivo numpy...")
            feature_names = np.load(npy_path, allow_pickle=True).tolist()
            logger.success(f"[LightGBM] {len(feature_names)} nomes de features carregados do numpy")
            return feature_names
            
        logger.warning("[LightGBM] Nenhum arquivo de nomes de features encontrado")
        return None
        
    except Exception as e:
        logger.error(f"[LightGBM] Erro ao carregar nomes das features: {e}")
        return None


def log_feature_info(X_train, feature_names=None):
    """
    Loga informações detalhadas sobre as features sendo usadas no treinamento
    """
    logger.info("[LightGBM] === INFORMAÇÕES DAS FEATURES ===")
    logger.info(f"[LightGBM] Total de features: {X_train.shape[1]}")
    logger.info(f"[LightGBM] Total de amostras: {X_train.shape[0]}")
    logger.info(f"[LightGBM] Tipo da matriz: {type(X_train)}")
    logger.info(f"[LightGBM] Esparsidade: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")
    
    if feature_names and len(feature_names) == X_train.shape[1]:
        logger.info(f"[LightGBM] Nomes das features ({len(feature_names)}):")
        
        # Agrupar features por tipo para melhor visualização
        ordinal_features = []
        categorical_features = []
        numerical_features = []
        
        for feature in feature_names:
            if any(col in feature for col in ["nivel_profissional", "nivel_academico", "nivel_ingles"]):
                ordinal_features.append(feature)
            elif feature in ["area_similarity", "vaga_sap"]:
                numerical_features.append(feature)
            else:
                categorical_features.append(feature)
        
        if ordinal_features:
            logger.info(f"[LightGBM] 📊 Features Ordinais ({len(ordinal_features)}):")
            for feat in ordinal_features:
                logger.info(f"[LightGBM]   - {feat}")
        
        if numerical_features:
            logger.info(f"[LightGBM] 🔢 Features Numéricas ({len(numerical_features)}):")
            for feat in numerical_features:
                logger.info(f"[LightGBM]   - {feat}")
            
        if categorical_features:
            logger.info(f"[LightGBM] 🏷️  Features Categóricas ({len(categorical_features)}):")
            
            # Agrupar por prefixo (nome da coluna original)
            cat_groups = {}
            for feat in categorical_features:
                prefix = feat.split('_')[0] if '_' in feat else feat
                if prefix not in cat_groups:
                    cat_groups[prefix] = []
                cat_groups[prefix].append(feat)
            
            for prefix, features in cat_groups.items():
                logger.info(f"[LightGBM]   {prefix} ({len(features)} valores):")
                if len(features) <= 5:
                    for feat in features:
                        logger.info(f"[LightGBM]     - {feat}")
                else:
                    for feat in features[:3]:
                        logger.info(f"[LightGBM]     - {feat}")
                    logger.info(f"[LightGBM]     ... e mais {len(features) - 3} valores")
                    
        # Salvar lista completa de features em arquivo separado para referência
        features_log_path = "app/model/features_used_in_training.txt"
        try:
            with open(features_log_path, "w", encoding='utf-8') as f:
                f.write("=== FEATURES UTILIZADAS NO TREINAMENTO ===\n\n")
                f.write(f"Total: {len(feature_names)} features\n")
                f.write(f"Data de treinamento: {pd.Timestamp.now()}\n\n")
                
                f.write("FEATURES ORDINAIS:\n")
                for feat in ordinal_features:
                    f.write(f"  - {feat}\n")
                
                f.write("\nFEATURES NUMÉRICAS:\n")
                for feat in numerical_features:
                    f.write(f"  - {feat}\n")
                
                f.write("\nFEATURES CATEGÓRICAS:\n")
                for feat in categorical_features:
                    f.write(f"  - {feat}\n")
                    
            logger.info(f"[LightGBM] Lista completa de features salva em {features_log_path}")
            
        except Exception as e:
            logger.warning(f"[LightGBM] Não foi possível salvar lista de features: {e}")
                
    else:
        if feature_names is None:
            logger.warning("[LightGBM] ⚠️  Nomes das features não disponíveis")
        else:
            logger.error(f"[LightGBM] ❌ Incompatibilidade: {len(feature_names)} nomes vs {X_train.shape[1]} features")
    
    logger.info("[LightGBM] =====================================")


def load_inputs(path):
    logger.info(f"[LightGBM] Carregando dados de entrada de {path}")
    X_train = sparse.load_npz(os.path.join(path, "X_train.npz"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    group_train = np.load(os.path.join(path, "group_train.npy"))
    
    logger.info(f"[LightGBM] Dados carregados: X_train{X_train.shape}, y_train{y_train.shape}, group_train{group_train.shape}")
    logger.info(f"[LightGBM] Primeiros 10 grupos: {group_train[:10]}")
    logger.info(f"[LightGBM] Total de grupos: {len(group_train)}, Total de amostras: {group_train.sum()}")
    
    return X_train, y_train, group_train


def get_optimized_params():
    """Parâmetros otimizados - substitua pelos seus melhores parâmetros aqui"""
    
    # Se você tiver o arquivo de melhores parâmetros, carregue assim:
    best_params_path = "app/model/best_params.json"
    if os.path.exists(best_params_path):
        logger.info("[LightGBM] Carregando parâmetros otimizados...")
        with open(best_params_path, 'r') as f:
            return json.load(f)
    
    # Caso contrário, use estes parâmetros melhorados:
    logger.info("[LightGBM] Usando parâmetros melhorados padrão...")
    return {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],
        "boosting_type": "gbdt",
        "num_leaves": 120,           # ⬆️ Aumentado de 50
        "learning_rate": 0.1,        # ⬆️ Aumentado de 0.05
        "feature_fraction": 0.9,     # 🆕 Reduz overfitting
        "bagging_fraction": 0.8,     # 🆕 Adiciona randomização  
        "bagging_freq": 5,           # 🆕 Frequência do bagging
        "min_data_in_leaf": 50,      # 🆕 Controla complexidade
        "lambda_l1": 0.1,            # 🆕 Regularização L1
        "lambda_l2": 0.1,            # 🆕 Regularização L2
        "verbosity": -1,
        "ndcg_eval_at": [3, 5],      
        "map_eval_at": [5, 10],     
        "random_state": 1993,
        "verbose": 1
    }


def train_lgbm_ranker(X_train, y_train, group_train):
    logger.info("[LightGBM] Iniciando treinamento do modelo rankeador...")

    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    
    # Usa parâmetros otimizados
    params = get_optimized_params()
    
    # Treina com mais rounds (early stopping automático se tiver validação)
    model = lgb.train(params, train_set, num_boost_round=300)  # ⬆️ Aumentado de 100
    
    logger.success("[LightGBM] Modelo treinado com sucesso.")
    return model, params


if __name__ == "__main__":
    # Importar pandas apenas se necessário para timestamp
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[LightGBM] Pandas não disponível para timestamp")
        pd = None
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("app/model", exist_ok=True)

    # Carregar dados
    X_train, y_train, group_train = load_inputs(DATA_PATH)
    
    # Carregar nomes das features
    feature_names = load_feature_names(DATA_PATH)
    
    # Logar informações das features
    log_feature_info(X_train, feature_names)
    
    # Treinar modelo
    model, params_used = train_lgbm_ranker(X_train, y_train, group_train)

    # Salva modelo
    dump(model, MODEL_PATH)
    logger.info(f"[LightGBM] Modelo salvo em {MODEL_PATH}")
    
    dump(model, MODEL_PATH_PROD)
    logger.info(f"[LightGBM] Modelo PROD  salvo em {MODEL_PATH_PROD}")
    
    # Salva parâmetros usados para referência
    with open("app/model/params_used_last_train.json", "w") as f:
        json.dump(params_used, f, indent=2)
    logger.info("[LightGBM] Parâmetros salvos em app/model/params_used_last_train.json")
    
    # Salvar nomes das features no diretório do modelo também para fácil acesso
    if feature_names:
        with open("app/model/feature_names.json", "w", encoding='utf-8') as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
        logger.info("[LightGBM] Nomes das features salvos em app/model/feature_names.json")
        
        # Log final com resumo
        logger.info("[LightGBM] === RESUMO FINAL DO TREINAMENTO ===")
        logger.info(f"[LightGBM] ✅ Modelo treinado com {len(feature_names)} features")
        logger.info(f"[LightGBM] ✅ {X_train.shape[0]} amostras de treinamento")
        logger.info(f"[LightGBM] ✅ {len(group_train)} grupos (vagas)")
        logger.info("[LightGBM] ==========================================")
    else:
        logger.warning("[LightGBM] ⚠️  Modelo treinado, mas nomes das features não disponíveis")
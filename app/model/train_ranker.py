import lightgbm as lgb
import numpy as np
from scipy import sparse
import os
from joblib import dump
from loguru import logger
import json

DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"


def load_inputs(path):
    logger.info(f"[LightGBM] Carregando dados de entrada de {path}")
    X_train = sparse.load_npz(os.path.join(path, "X_train.npz"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    group_train = np.load(os.path.join(path, "group_train.npy"))
    print(group_train[:10], group_train.sum(), len(y_train))
    logger.info(f"[LightGBM] Dados carregados: {X_train.shape}, {y_train.shape}, {group_train.shape}")
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
    os.makedirs("models", exist_ok=True)
    os.makedirs("app/model", exist_ok=True)

    X_train, y_train, group_train = load_inputs(DATA_PATH)
    model, params_used = train_lgbm_ranker(X_train, y_train, group_train)

    # Salva modelo
    dump(model, MODEL_PATH)
    logger.info(f"[LightGBM] Modelo salvo em {MODEL_PATH}")
    
    # Salva parâmetros usados para referência
    with open("app/model/params_used_last_train.json", "w") as f:
        json.dump(params_used, f, indent=2)
    logger.info("[LightGBM] Parâmetros salvos em app/model/params_used.json")
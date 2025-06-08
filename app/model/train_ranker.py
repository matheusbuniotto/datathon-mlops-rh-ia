import lightgbm as lgb
import numpy as np
from scipy import sparse
import os
from joblib import dump
from loguru import logger

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


def train_lgbm_ranker(X_train, y_train, group_train):
    logger.info("[LightGBM] Iniciando treinamento do modelo rankeador...")

    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)

    params = {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],
        "boosting_type": "gbdt",
        "num_leaves": 50,
        "learning_rate": 0.05,
        "verbosity": -1,
        "ndcg_eval_at": [3, 5],
        "map_eval_at": [5],
        "random_state": 1993,
        "verbose": 1
    }

    model = lgb.train(params, train_set, num_boost_round=100)
    logger.success("[LightGBM] Modelo treinado com sucesso.")
    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    X_train, y_train, group_train = load_inputs(DATA_PATH)
    model = train_lgbm_ranker(X_train, y_train, group_train)

    dump(model, MODEL_PATH)
    logger.info(f"[LightGBM] Modelo salvo em {MODEL_PATH}")

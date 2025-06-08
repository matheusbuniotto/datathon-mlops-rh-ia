import numpy as np
from joblib import load
from scipy import sparse
from sklearn.metrics import ndcg_score, average_precision_score
from loguru import logger
import os

DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"

def load_inputs():
    X_val = sparse.load_npz(os.path.join(DATA_PATH, "X_val.npz"))
    y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))
    group_val = np.load(os.path.join(DATA_PATH, "group_val.npy"))
    return X_val, y_val, group_val


def evaluate_model(model, X_val, y_val, group_val, k=5, relevance_threshold=0):
    logger.info("[Eval] Avaliando modelo LightGBM no conjunto de validação...")

    y_pred = model.predict(X_val)

    scores_ndcg = []
    scores_map = []

    offset = 0
    for size in group_val:
        y_true_group = y_val[offset:offset+size]
        y_pred_group = y_pred[offset:offset+size]

        if len(np.unique(y_true_group)) > 1:
            # NDCG funciona com múltiplos níveis de relevância
            scores_ndcg.append(ndcg_score([y_true_group], [y_pred_group], k=k))
            
            # Para MAP, binarizamos os rótulos (relevante vs não relevante)
            y_true_binary = (y_true_group > relevance_threshold).astype(int)
            
            # Só calcula MAP se houver pelo menos um item relevante
            if np.sum(y_true_binary) > 0:
                scores_map.append(average_precision_score(y_true_binary, y_pred_group))
        
        offset += size

    logger.success(f"[Eval] NDCG@{k}: {np.mean(scores_ndcg):.4f}")
    logger.success(f"[Eval] MAP@{k}:  {np.mean(scores_map):.4f}")

    return np.mean(scores_ndcg), np.mean(scores_map)


if __name__ == "__main__":
    model = load(MODEL_PATH)
    X_test, y_test, group_test= load_inputs()
    evaluate_model(model, X_test, y_test, group_test, k=5, relevance_threshold=0.2)
    logger.info("[Eval] Avaliação concluída.")
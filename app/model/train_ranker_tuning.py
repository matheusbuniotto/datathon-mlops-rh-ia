import lightgbm as lgb
import numpy as np
from scipy import sparse
import os
from joblib import dump
from loguru import logger
import optuna
from sklearn.model_selection import ParameterGrid
import json

DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"
STUDY_PATH = "app/model/optuna_study.pkl"


def load_inputs(path, split):
    """Carrega dados de treino, validação ou teste"""
    logger.info(f"[LightGBM] Carregando dados de {split} de {path}")
    X = sparse.load_npz(os.path.join(path, f"X_{split}.npz"))
    y = np.load(os.path.join(path, f"y_{split}.npy"))
    group = np.load(os.path.join(path, f"group_{split}.npy"))
    logger.info(f"[LightGBM] Dados {split} carregados: {X.shape}, {y.shape}, {group.shape}")
    return X, y, group


def train_lgbm_ranker_with_validation(X_train, y_train, group_train, 
                                    X_val, y_val, group_val, params):
    """Treina modelo com validação e early stopping"""
    
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)

    model = lgb.train(
        params, 
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'val'],
        num_boost_round=1000,  # Número alto, vai parar early
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=25)
        ]
    )
    
    return model


def objective(trial, X_train, y_train, group_train, X_val, y_val, group_val):
    """Função objetivo para Optuna"""
    
    params = {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "ndcg_eval_at": [3, 5, 10],
        "map_eval_at": [5, 10],
        "random_state": 1993,
        
        # Hiperparâmetros a otimizar
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        valid_names=['val'],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(0)  # Silencioso
        ]
    )
    
    # Retorna NDCG@5 do conjunto de validação
    return model.best_score['val']['ndcg@5']


def hyperparameter_tuning(X_train, y_train, group_train, X_val, y_val, group_val, n_trials=50):
    """Otimização de hiperparâmetros com Optuna"""
    
    logger.info(f"[Optuna] Iniciando otimização com {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, group_train, X_val, y_val, group_val),
        n_trials=n_trials
    )
    
    logger.success(f"[Optuna] Melhor NDCG@5: {study.best_value:.4f}")
    logger.info(f"[Optuna] Melhores parâmetros: {study.best_params}")
    
    return study.best_params


def grid_search_simple(X_train, y_train, group_train, X_val, y_val, group_val):
    """Grid search simples para comparação rápida"""
    
    param_grid = {
        'num_leaves': [30, 50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'feature_fraction': [0.8, 0.9, 1.0],
    }
    
    best_score = 0
    best_params = None
    
    logger.info("[Grid Search] Iniciando busca em grade...")
    
    for params in ParameterGrid(param_grid):
        base_params = {
            "objective": "lambdarank",
            "metric": ["ndcg", "map"],
            "boosting_type": "gbdt",
            "verbosity": -1,
            "ndcg_eval_at": [5],
            "random_state": 1993,
        }
        base_params.update(params)
        
        train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
        
        model = lgb.train(
            base_params,
            train_set,
            valid_sets=[val_set],
            valid_names=['val'],
            num_boost_round=200,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        score = model.best_score['val']['ndcg@5']
        logger.info(f"[Grid Search] Params: {params} -> NDCG@5: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = base_params
    
    logger.success(f"[Grid Search] Melhor NDCG@5: {best_score:.4f}")
    return best_params


def train_final_model(X_train, y_train, group_train, X_val, y_val, group_val, 
                     X_test, y_test, group_test, best_params):
    """Treina modelo final com melhores parâmetros"""
    
    logger.info("[Final Model] Treinando modelo final...")
    
    # Combina train + val para treino final
    X_combined = sparse.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    group_combined = np.concatenate([group_train, group_val])
    
    train_set = lgb.Dataset(X_combined, label=y_combined, group=group_combined)
    test_set = lgb.Dataset(X_test, label=y_test, group=group_test, reference=train_set)
    
    model = lgb.train(
        best_params,
        train_set,
        valid_sets=[test_set],
        valid_names=['test'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=25)
        ]
    )
    
    logger.success("[Final Model] Modelo final treinado!")
    return model


if __name__ == "__main__":
    os.makedirs("app/model", exist_ok=True)

    # Carrega todos os dados
    X_train, y_train, group_train = load_inputs(DATA_PATH, "train")
    X_val, y_val, group_val = load_inputs(DATA_PATH, "val")
    X_test, y_test, group_test = load_inputs(DATA_PATH, "test")

    # M<etodo de otimização
    optimization_method = "optuna" 
    
    if optimization_method == "optuna":
        # Otimização com Optuna (mais sofisticada)
        best_params = hyperparameter_tuning(
            X_train, y_train, group_train, 
            X_val, y_val, group_val, 
            n_trials=30
        )
        
    elif optimization_method == "grid_search":
        # Grid search simples (mais rápida)
        best_params = grid_search_simple(
            X_train, y_train, group_train,
            X_val, y_val, group_val
        )
        
    else:
        # Parâmetros padrão melhorados
        best_params = {
            "objective": "lambdarank",
            "metric": ["ndcg", "map"],
            "boosting_type": "gbdt",
            "num_leaves": 100,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbosity": -1,
            "ndcg_eval_at": [3, 5, 10],
            "map_eval_at": [5, 10],
            "random_state": 1993,
        }

    # Treina modelo final
    final_model = train_final_model(
        X_train, y_train, group_train,
        X_val, y_val, group_val,
        X_test, y_test, group_test,
        best_params
    )

    # Salva modelo e parâmetros
    dump(final_model, MODEL_PATH)
    
    with open("app/model/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    logger.success(f"[Final] Modelo salvo em {MODEL_PATH}")
    logger.info("[Final] Parâmetros salvos em app/model/best_params.json")
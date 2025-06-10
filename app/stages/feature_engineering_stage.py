import pandas as pd
import numpy as np
from scipy import sparse
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from joblib import dump
from loguru import logger
from fuzzywuzzy import fuzz
from app.utils.strings_clean_utils import clean_area_atuacao
from app.constants import (
    NIVEL_PROFISSIONAL_ORDER,
    NIVEL_ACADEMICO_ORDER,
    NIVEL_INGLES_ORDER
)

def get_preprocessing_pipeline():
    # Define columns
    ordinais = {
        "nivel_profissional": NIVEL_PROFISSIONAL_ORDER,
        "nivel_profissional_vaga": NIVEL_PROFISSIONAL_ORDER,
        "nivel_academico": NIVEL_ACADEMICO_ORDER,
        "nivel_ingles": NIVEL_INGLES_ORDER,
        "nivel_ingles_vaga": NIVEL_INGLES_ORDER
    }
    categoricas = ["cliente", "recrutador", "vaga_areas_atuacao_clean"]
    numericas = ["vaga_sap", "area_similarity"]
    
    # Add both embedding columns
    emb_cv_cols = [f"emb_cv_{i}" for i in range(384)]
    emb_vaga_cols = [f"emb_vaga_{i}" for i in range(384)]

    # Check scikit-learn version for OneHotEncoder sparsity parameter
    from sklearn import __version__
    sklearn_version = tuple(map(int, __version__.split('.')[:2]))
    ohe_sparse_param = {'sparse_output': True} if sklearn_version >= (1, 2) else {'sparse': True}

    # Pipelines
    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=[ordinais[col] for col in ordinais],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1))
    ])
    
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=100, **ohe_sparse_param))
    ])
    
    # Separate pipelines for each embedding type
    emb_cv_pipeline = Pipeline([
        ("pca", PCA(n_components=75))
    ])
    
    emb_vaga_pipeline = Pipeline([
        ("pca", PCA(n_components=75))
    ])

    transformers = [
        ("ordinal", ordinal_pipeline, list(ordinais.keys())),
        ("categorical", categorical_pipeline, categoricas),
        ("emb_cv", emb_cv_pipeline, emb_cv_cols),
        ("emb_vaga", emb_vaga_pipeline, emb_vaga_cols),
        ("passthrough", "passthrough", numericas)
    ]

    return ColumnTransformer(transformers=transformers)

def get_feature_names_from_pipeline(pipeline, input_df):
    try:
        feature_names = []
        for name, transformer, columns in pipeline.transformers_:
            if name == "ordinal":
                feature_names.extend(columns)
            elif name == "categorical":
                encoder = transformer.named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(columns)
                    feature_names.extend(cat_features.tolist())
                else:
                    for i, col in enumerate(columns):
                        if hasattr(encoder, 'categories_') and i < len(encoder.categories_):
                            for cat in encoder.categories_[i]:
                                feature_names.append(f"{col}_{cat}")
                        else:
                            feature_names.append(f"{col}_encoded")
            elif name in ["emb_cv", "emb_vaga"]:
                n_components = transformer.named_steps['pca'].n_components
                prefix = "emb_cv" if name == "emb_cv" else "emb_vaga"
                feature_names.extend([f"{prefix}_pca_{i}" for i in range(n_components)])
            elif name == "passthrough":
                feature_names.extend(columns)
        
        logger.info(f"[Features] Extracted {len(feature_names)} feature names from pipeline")
        return feature_names
    except Exception as e:
        logger.error(f"[Features] Error extracting feature names: {e}")
        return None

def calculate_area_similarity(df):
    df = df.copy()
    def fuzzy_similarity(row):
        candidato_area = str(row['candidato_area_atuacao']).lower()
        vaga_area = str(row['vaga_areas_atuacao']).lower()
        if candidato_area in ['nan', 'indefinido', ''] or vaga_area in ['nan', 'indefinido', '']:
            return 0
        return fuzz.ratio(candidato_area, vaga_area)
    
    df['area_similarity'] = df.apply(fuzzy_similarity, axis=1)
    return df

def save_model_input(X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, feature_names=None, path="data/model_input"):
    os.makedirs(path, exist_ok=True)
    if not sparse.issparse(X_train):
        X_train = sparse.csr_matrix(X_train)
    if not sparse.issparse(X_val):
        X_val = sparse.csr_matrix(X_val)
    if not sparse.issparse(X_test):
        X_test = sparse.csr_matrix(X_test)

    sparse.save_npz(f"{path}/X_train.npz", X_train)
    sparse.save_npz(f"{path}/X_val.npz", X_val)
    sparse.save_npz(f"{path}/X_test.npz", X_test)
    np.save(f"{path}/y_train.npy", y_train)
    np.save(f"{path}/y_val.npy", y_val)
    np.save(f"{path}/y_test.npy", y_test)
    np.save(f"{path}/group_train.npy", group_train)
    np.save(f"{path}/group_val.npy", group_val)
    np.save(f"{path}/group_test.npy", group_test)
    
    if feature_names is not None:
        np.save(f"{path}/feature_names.npy", np.array(feature_names, dtype=object))
        import json
        with open(f"{path}/feature_names.json", "w", encoding='utf-8') as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
        logger.info(f"[Features] Saved {len(feature_names)} feature names to {path}")
    else:
        logger.warning("[Features] Feature names not available to save")

def apply_feature_pipeline(df_train, df_val, df_test):
    logger.info("[Features] Applying encoding transformations...")
    from app.utils.embedding_utils import explode_embeddings

    logger.info("[Features] Expanding embeddings...")
    df_train = explode_embeddings(df_train)
    df_val = explode_embeddings(df_val)
    df_test = explode_embeddings(df_test)
    logger.success("[Features] Embeddings expanded successfully.")

    def fill_missing(df):
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna("Indefinido")
        return df

    df_train = fill_missing(df_train)
    df_val = fill_missing(df_val)
    df_test = fill_missing(df_test)

    logger.info("[Features] Cleaning area of expertise columns...")
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    
    df_train['candidato_area_atuacao'] = clean_area_atuacao(df_train, 'candidato_area_atuacao')
    df_val['candidato_area_atuacao'] = clean_area_atuacao(df_val, 'candidato_area_atuacao')
    df_test['candidato_area_atuacao'] = clean_area_atuacao(df_test, 'candidato_area_atuacao')
    
    df_train['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_train, 'vaga_areas_atuacao')
    df_val['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_val, 'vaga_areas_atuacao')
    df_test['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_test, 'vaga_areas_atuacao')
    logger.success("[Features] Area of expertise columns cleaned successfully.")
    
    logger.info("[Features] Calculating area similarity...")
    df_train = calculate_area_similarity(df_train)
    df_val = calculate_area_similarity(df_val)
    df_test = calculate_area_similarity(df_test)
    logger.success("[Features] Area similarity calculated successfully.")

    y_train = df_train["target_rank"]
    y_val = df_val["target_rank"]
    y_test = df_test["target_rank"]

    group_train = df_train.groupby("codigo_vaga").size().values
    group_val = df_val.groupby("codigo_vaga").size().values
    group_test = df_test.groupby("codigo_vaga").size().values

    pipe = get_preprocessing_pipeline()
    logger.info("[Features] Fitting transformation pipeline...")
    X_train = pipe.fit_transform(df_train)
    X_val = pipe.transform(df_val)
    X_test = pipe.transform(df_test)

    logger.info("[Features] Extracting feature names...")
    feature_names = get_feature_names_from_pipeline(pipe, df_train)
    
    pipeline_path = "data/model_input/preprocessing_pipeline.pkl"
    dump(pipe, pipeline_path)
    logger.info(f"[Features] Pipeline saved to {pipeline_path}")

    logger.success("[Features] Feature pipeline applied successfully.")
    
    if feature_names:
        logger.info(f"[Features] === FEATURE SUMMARY ===")
        logger.info(f"[Features] Total features: {len(feature_names)}")
        logger.info(f"[Features] Matrix shape: {X_train.shape}")
        
        ordinal_features = [f for f in feature_names if any(nivel in f for nivel in ["nivel_profissional", "nivel_academico", "nivel_ingles"])]
        numerical_features = [f for f in feature_names if f in ["vaga_sap", "area_similarity"] or f.startswith("emb_cv_pca_")]
        categorical_features = [f for f in feature_names if f not in ordinal_features and f not in numerical_features]
        
        logger.info(f"[Features] Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
        logger.info(f"[Features] Numerical Features ({len(numerical_features)}): {numerical_features[:10] + ['...'] if len(numerical_features) > 10 else numerical_features}")
        logger.info(f"[Features] Categorical Features: {len(categorical_features)}")
        
        if categorical_features and len(categorical_features) <= 10:
            logger.info(f"[Features] Categorical: {categorical_features}")
        elif categorical_features:
            logger.info(f"[Features] First 10 Categorical: {categorical_features[:10]}")
            logger.info(f"[Features] ... and {len(categorical_features) - 10} more categorical features")
        
        logger.info("[Features] =====================================")
    
    logger.info("[Features] Saving model input...")
    save_model_input(X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, feature_names)
    logger.success("[Features] Model input saved successfully.")
    logger.info("[Features] Feature pipeline completed.")
    return X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe
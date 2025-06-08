import pandas as pd
import numpy as np
from scipy import sparse
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from loguru import logger
from app.constants import (
    NIVEL_PROFISSIONAL_ORDER,
    NIVEL_ACADEMICO_ORDER,
    NIVEL_INGLES_ORDER
)


def get_preprocessing_pipeline():
    # Define colunas
    ordinais = {
        "nivel_profissional": NIVEL_PROFISSIONAL_ORDER,
        "nivel_profissional_vaga": NIVEL_PROFISSIONAL_ORDER,
        "nivel_academico": NIVEL_ACADEMICO_ORDER,
        "nivel_ingles": NIVEL_INGLES_ORDER,
        "nivel_ingles_vaga": NIVEL_INGLES_ORDER
    }

    categoricas = ["cliente", "recrutador"]
    numericas = ["vaga_sap"]

    # Pipelines
    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=[ordinais[col] for col in ordinais],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1))
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=500))
    ])

    transformers = [
        ("ordinal", ordinal_pipeline, list(ordinais.keys())),
        ("categorical", categorical_pipeline, categoricas),
        ("passthrough", "passthrough", numericas)
    ]

    return ColumnTransformer(transformers=transformers)

def save_model_input(X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, path="data/model_input"):
    os.makedirs(path, exist_ok=True)

    sparse.save_npz(f"{path}/X_train.npz", X_train)
    sparse.save_npz(f"{path}/X_val.npz", X_val)
    sparse.save_npz(f"{path}/X_test.npz", X_test)

    np.save(f"{path}/y_train.npy", y_train)
    np.save(f"{path}/y_val.npy", y_val)
    np.save(f"{path}/y_test.npy", y_test)

    np.save(f"{path}/group_train.npy", group_train)
    np.save(f"{path}/group_val.npy", group_val)
    np.save(f"{path}/group_test.npy", group_test)
    
def apply_feature_pipeline(df_train, df_val, df_test):
    logger.info("[Features] Aplicando transformações de encoding...")
    
    from app.utils.embedding_utils import explode_embeddings

    logger.info("[Features] Expandindo embeddings...")
    df_train = explode_embeddings(df_train)
    df_val = explode_embeddings(df_val)
    df_test = explode_embeddings(df_test)
    logger.success("[Features] Embeddings expandidos com sucesso.")

    # Substituir missing values de acordo com o tipo de dados
    def fill_missing(df):
        # Fazer uma cópia para evitar SettingWithCopyWarning
        df = df.copy()
        
        # Para cada coluna, preencher NA de acordo com o tipo
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna("Indefinido")
        return df

    df_train = fill_missing(df_train)
    df_val = fill_missing(df_val)
    df_test = fill_missing(df_test)

    y_train = df_train["target_rank"]
    y_val = df_val["target_rank"]
    y_test = df_test["target_rank"]

    group_train = df_train.groupby("codigo_vaga").size().values
    group_val = df_val.groupby("codigo_vaga").size().values
    group_test = df_test.groupby("codigo_vaga").size().values

    pipe = get_preprocessing_pipeline()
    X_train = pipe.fit_transform(df_train)
    X_val = pipe.transform(df_val)
    X_test = pipe.transform(df_test)

    logger.success("[Features] Pipeline de features aplicado com sucesso.")
    
    logger.info("[Features] Salvando input do modelo...")
    save_model_input(X_train, y_train, group_train,
                 X_val, y_val, group_val,
                 X_test, y_test, group_test)

    logger.success("[Features] Input do modelo salvo com sucesso.")
    logger.info("[Features] Pipeline de features concluído.")
    return X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe

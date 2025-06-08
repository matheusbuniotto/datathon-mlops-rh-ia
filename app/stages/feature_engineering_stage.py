import pandas as pd
import numpy as np
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
    numericas = ["vaga_sap", "similaridade_vaga_cv", "similaridade_area", "similaridade_combinada"]

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


def apply_feature_pipeline(df_train, df_val, df_test):
    logger.info("[Features] Aplicando transformações de encoding...")

    # Get columns from pipeline definition
    ordinais = [
        "nivel_profissional",
        "nivel_profissional_vaga",
        "nivel_academico",
        "nivel_ingles",
        "nivel_ingles_vaga"
    ]
    categoricas = ["cliente", "recrutador", "estado"]

    def fillna_cats(df):
        for col in ordinais + categoricas:
            if col in df.columns:
                df[col] = df[col].fillna("Indefinido")
        return df

    df_train = fillna_cats(df_train)
    df_val = fillna_cats(df_val)
    df_test = fillna_cats(df_test)

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
    return X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe

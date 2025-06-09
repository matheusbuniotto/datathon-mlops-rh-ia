import pandas as pd
import numpy as np
from scipy import sparse
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    # Define colunas
    ordinais = {
        "nivel_profissional": NIVEL_PROFISSIONAL_ORDER,
        "nivel_profissional_vaga": NIVEL_PROFISSIONAL_ORDER,
        "nivel_academico": NIVEL_ACADEMICO_ORDER,
        "nivel_ingles": NIVEL_INGLES_ORDER,
        "nivel_ingles_vaga": NIVEL_INGLES_ORDER
    }

    # Atualizar colunas categóricas - remover candidato_area_atuacao e adicionar vaga_areas_atuacao_clean
    categoricas = ["cliente", "recrutador", "vaga_areas_atuacao_clean"]
    numericas = ["vaga_sap", "area_similarity"] 

    # Pipelines
    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=[ordinais[col] for col in ordinais],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1))
    ])

    # Pipeline categórico com min_frequency=100 para vaga_areas_atuacao_clean
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=100))
    ])

    transformers = [
        ("ordinal", ordinal_pipeline, list(ordinais.keys())),
        ("categorical", categorical_pipeline, categoricas),
        ("passthrough", "passthrough", numericas)
    ]

    return ColumnTransformer(transformers=transformers)


def get_feature_names_from_pipeline(pipeline, input_df):
    """
    Extrai os nomes das features após a transformação pelo pipeline
    """
    try:
        feature_names = []
        
        # Para cada transformer no pipeline
        for name, transformer, columns in pipeline.transformers_:
            if name == "ordinal":
                # Features ordinais mantêm os nomes originais
                feature_names.extend(columns)
                
            elif name == "categorical":
                # Features categóricas são expandidas com one-hot encoding
                # Primeiro, precisamos ajustar o transformer com os dados
                encoder = transformer.named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(columns)
                    feature_names.extend(cat_features.tolist())
                else:
                    # Fallback: usar as categorias do encoder
                    for i, col in enumerate(columns):
                        if hasattr(encoder, 'categories_') and i < len(encoder.categories_):
                            for cat in encoder.categories_[i]:
                                feature_names.append(f"{col}_{cat}")
                        else:
                            feature_names.append(f"{col}_encoded")
                            
            elif name == "passthrough":
                # Features numéricas passam direto
                feature_names.extend(columns)
        
        logger.info(f"[Features] Extraídos {len(feature_names)} nomes de features do pipeline")
        return feature_names
        
    except Exception as e:
        logger.error(f"[Features] Erro ao extrair nomes das features: {e}")
        return None


def calculate_area_similarity(df):
    """
    Calcula a similaridade fuzzy entre candidato_area_atuacao e vaga_areas_atuacao
    usando .lower() para ambas as colunas
    """
    # Criar uma cópia do DataFrame para evitar fragmentação
    df = df.copy()
    
    def fuzzy_similarity(row):
        candidato_area = str(row['candidato_area_atuacao']).lower()
        vaga_area = str(row['vaga_areas_atuacao']).lower()
        
        # Se alguma das áreas for vazia ou indefinida, retorna 0
        if candidato_area in ['nan', 'indefinido', ''] or vaga_area in ['nan', 'indefinido', '']:
            return 0
        
        # Calcula a similaridade usando fuzzywuzzy
        return fuzz.ratio(candidato_area, vaga_area)
    
    df['area_similarity'] = df.apply(fuzzy_similarity, axis=1)
    return df


def save_model_input(X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, feature_names=None, path="data/model_input"):
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
    
    # Salvar nomes das features se disponíveis
    if feature_names is not None:
        # Salvar como array numpy
        np.save(f"{path}/feature_names.npy", np.array(feature_names, dtype=object))
        
        # Salvar como JSON para fácil leitura
        import json
        with open(f"{path}/feature_names.json", "w", encoding='utf-8') as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Features] Salvos {len(feature_names)} nomes de features em {path}")
    else:
        logger.warning("[Features] Nomes das features não disponíveis para salvar")


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

    # Aplicar limpeza de strings das colunas de área de atuação
    logger.info("[Features] Limpando colunas de área de atuação...")
    
    # Fazer uma cópia dos DataFrames para evitar fragmentação
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    
    # Limpar candidato_area_atuacao
    df_train['candidato_area_atuacao'] = clean_area_atuacao(df_train, 'candidato_area_atuacao')
    df_val['candidato_area_atuacao'] = clean_area_atuacao(df_val, 'candidato_area_atuacao')
    df_test['candidato_area_atuacao'] = clean_area_atuacao(df_test, 'candidato_area_atuacao')
    
    # Limpar vaga_areas_atuacao e criar coluna limpa para encoding
    df_train['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_train, 'vaga_areas_atuacao')
    df_val['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_val, 'vaga_areas_atuacao')
    df_test['vaga_areas_atuacao_clean'] = clean_area_atuacao(df_test, 'vaga_areas_atuacao')
    
    logger.success("[Features] Colunas de área de atuação limpas com sucesso.")
    
    # Calcular similaridade fuzzy entre as áreas
    logger.info("[Features] Calculando similaridade entre áreas de atuação...")
    df_train = calculate_area_similarity(df_train)
    df_val = calculate_area_similarity(df_val)
    df_test = calculate_area_similarity(df_test)
    logger.success("[Features] Similaridade entre áreas calculada com sucesso.")

    y_train = df_train["target_rank"]
    y_val = df_val["target_rank"]
    y_test = df_test["target_rank"]

    group_train = df_train.groupby("codigo_vaga").size().values
    group_val = df_val.groupby("codigo_vaga").size().values
    group_test = df_test.groupby("codigo_vaga").size().values

    # Criar e ajustar o pipeline
    pipe = get_preprocessing_pipeline()
    logger.info("[Features] Ajustando pipeline de transformação...")
    X_train = pipe.fit_transform(df_train)
    X_val = pipe.transform(df_val)
    X_test = pipe.transform(df_test)

    # Extrair nomes das features após o ajuste
    logger.info("[Features] Extraindo nomes das features...")
    feature_names = get_feature_names_from_pipeline(pipe, df_train)
    
    # Salvar o pipeline para uso posterior
    pipeline_path = "data/model_input/preprocessing_pipeline.pkl"
    dump(pipe, pipeline_path)
    logger.info(f"[Features] Pipeline salvo em {pipeline_path}")

    logger.success("[Features] Pipeline de features aplicado com sucesso.")
    
    # Log das informações das features
    if feature_names:
        logger.info(f"[Features] === RESUMO DAS FEATURES ===")
        logger.info(f"[Features] Total de features: {len(feature_names)}")
        logger.info(f"[Features] Shape da matriz: {X_train.shape}")
        
        # Categorizar features para melhor visualização
        ordinal_features = [f for f in feature_names if any(nivel in f for nivel in ["nivel_profissional", "nivel_academico", "nivel_ingles"])]
        numerical_features = [f for f in feature_names if f in ["vaga_sap", "area_similarity"]]
        categorical_features = [f for f in feature_names if f not in ordinal_features and f not in numerical_features]
        
        logger.info(f"[Features] Features Ordinais ({len(ordinal_features)}): {ordinal_features}")
        logger.info(f"[Features] Features Numéricas ({len(numerical_features)}): {numerical_features}")
        logger.info(f"[Features] Features Categóricas: {len(categorical_features)}")
        
        # Mostrar algumas categóricas como exemplo
        if categorical_features:
            if len(categorical_features) <= 10:
                logger.info(f"[Features] Categóricas: {categorical_features}")
            else:
                logger.info(f"[Features] Primeiras 10 categóricas: {categorical_features[:10]}")
                logger.info(f"[Features] ... e mais {len(categorical_features) - 10} features categóricas")
        
        logger.info("[Features] =====================================")
    
    logger.info("[Features] Salvando input do modelo...")
    save_model_input(X_train, y_train, group_train,
                     X_val, y_val, group_val,
                     X_test, y_test, group_test,
                     feature_names=feature_names)

    logger.success("[Features] Input do modelo salvo com sucesso.")
    logger.info("[Features] Pipeline de features concluído.")
    return X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe
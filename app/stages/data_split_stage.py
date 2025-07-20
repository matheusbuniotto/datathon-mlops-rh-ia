import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from loguru import logger


def split_dataset_by_vaga(
    df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=1993
):
    logger.info("[Split] Iniciando divisão do dataset por vaga...")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["codigo_vaga"]))

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    # Divindo treino =  treino + validação
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state
    )
    train_idx, val_idx = next(gss_val.split(df_train, groups=df_train["codigo_vaga"]))

    df_train_final = df_train.iloc[train_idx]
    df_val_final = df_train.iloc[val_idx]

    logger.success(
        f"[Split] Feito! Tamanhos: train={len(df_train_final)}, val={len(df_val_final)}, test={len(df_test)}"
    )

    return (
        df_train_final.reset_index(drop=True),
        df_val_final.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )

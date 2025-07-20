import pandas as pd
from loguru import logger

from app.constants import SITUACAO_TO_SCORE


def prepare_ranking_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"[Ranking] Colunas disponíveis no DataFrame: {df.columns.tolist()}")
    logger.info(
        "[Ranking] Gerando score de relevância com base na situação do candidato..."
    )

    df = df.copy()
    df["target_rank"] = (
        df["situacao_candidato"].map(SITUACAO_TO_SCORE).fillna(0).astype(int)
    )

    logger.success(
        f"[Ranking] Dataset preparado com coluna target_rank. Distribuição:\n{df['target_rank'].value_counts()}"
    )
    return df

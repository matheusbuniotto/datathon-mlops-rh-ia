import pandas as pd
from loguru import logger

SITUACAO_TO_SCORE = {
    "Contratado pela Decision": 5,
    "Contratado como Hunting": 5,
    "Proposta Aceita": 4,
    "Encaminhar Proposta": 4,
    "Documentação CLT/PJ/Cooperado": 4,
    "Aprovado": 3,
    "Entrevista com Cliente": 3,
    "Entrevista Técnica": 2,
    "Em avaliação pelo RH": 2,
    "Encaminhado ao Requisitante": 1,
    "Inscrito": 1,
    "Prospect": 0,
    "Não Aprovado pelo RH": 0,
    "Não Aprovado pelo Requisitante": 0,
    "Não Aprovado pelo Cliente": 0,
    "Desistiu": 0,
    "Desistiu da Contratação": 0,
    "Sem interesse nesta vaga": 0,
    "Recusado": 0,
    None: -1
}


def prepare_ranking_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"[Ranking] Colunas disponíveis no DataFrame: {df.columns.tolist()}")
    logger.info("[Ranking] Gerando score de relevância com base na situação do candidato...")

    df = df.copy()
    df["target_rank"] = df["situacao_candidato"].map(SITUACAO_TO_SCORE).fillna(0).astype(int)

    logger.success(f"[Ranking] Dataset preparado com coluna target_rank. Distribuição:\n{df['target_rank'].value_counts()}")
    return df

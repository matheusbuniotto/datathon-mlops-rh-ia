import argparse
import json 

import pandas as pd
from app.similarity import recommend_candidates_for_vaga
from loguru import logger

EMBEDDINGS_PATH = "data/embeddings/combined_embeddings.parquet"


def main():
    parser = argparse.ArgumentParser(description="Sistema de recomendação de candidatos.")
    parser.add_argument("--codigo_vaga", type=int, required=True, help="ID da vaga para recomendação")
    parser.add_argument("--top_n", type=int, default=5, help="Número de candidatos recomendados (padrão = 5)")
    args = parser.parse_args()

    logger.info(f"Carregando embeddings de {EMBEDDINGS_PATH}")
    df = pd.read_parquet(EMBEDDINGS_PATH)

    logger.info(f"Recomendando candidatos para codigo_vaga={args.codigo_vaga}")
    try:
        top_candidatos = recommend_candidates_for_vaga(df, args.codigo_vaga, args.top_n)
        # Remove duplicatas pelo email
        top_candidatos = top_candidatos.drop_duplicates(subset=["email"])
        logger.info(f"{len(top_candidatos)} candidatos recomendados para a vaga {args.codigo_vaga}")
        payload = top_candidatos[[
            "nome_candidato", "email", "similaridade_combinada", "similaridade_vaga_cv", "similaridade_area"
        ]].to_dict(orient="records")
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        logger.debug(f"Payload de recomendação (JSON): {payload_json}")
        return payload
    except ValueError as e:
        logger.error(str(e))
        return []


if __name__ == "__main__":
    main()

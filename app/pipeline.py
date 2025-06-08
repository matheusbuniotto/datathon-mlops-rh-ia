import os
import duckdb
import pandas as pd

from app.data_loader import load_applicants, load_jobs, load_prospects
from app.stages.embeddings_stage import generate_and_save_embeddings
from app.stages.ranking_preparation_stage import prepare_ranking_dataset
from loguru import logger


DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
SQL_PATH = "data/sql/merge_recrutamento.sql"
EMBEDDING_PATH_SAVE = "data/embeddings"

def run_pipeline():
    print("[INFO] Carregando dados JSON...")
    df_app = load_applicants(os.path.join(DATA_RAW, "applicants.json"))
    df_vagas = load_jobs(os.path.join(DATA_RAW, "vagas.json"))
    df_prospects = load_prospects(os.path.join(DATA_RAW, "prospects.json"))

    print("[INFO] Salvando arquivos Parquet...")
    df_app.to_parquet(os.path.join(DATA_PROCESSED, "applicants.parquet"), index=False)
    df_vagas.to_parquet(os.path.join(DATA_PROCESSED, "vagas.parquet"), index=False)
    df_prospects.to_parquet(os.path.join(DATA_PROCESSED, "prospects.parquet"), index=False)

    print("[INFO] Executando SQL para unificação...")
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_query = f.read()

    db = duckdb.connect(database=":memory:")
    df_merged = db.execute(sql_query).fetchdf()

    print("[INFO] Salvando tabela unificada.")
    df_merged.to_parquet(os.path.join(DATA_PROCESSED, "merged.parquet"), index=False)

    # Nova etapa: embeddings
    embeddings_file = os.path.join(EMBEDDING_PATH_SAVE, "combined_embeddings.parquet")
    if not os.path.exists(embeddings_file):
        generate_and_save_embeddings(
            input_path=os.path.join(DATA_PROCESSED, "merged.parquet"),
            output_path=EMBEDDING_PATH_SAVE
        )
    else:
        print(f"[INFO] Embeddings já existem em {embeddings_file}, pulando geração.")

     # Nova etapa: preparação para rankeamento
    df_ = pd.read_parquet(os.path.join(EMBEDDING_PATH_SAVE, "combined_embeddings.parquet"))
    df_rank = prepare_ranking_dataset(df_)
    df_rank.to_parquet("data/processed/rank_ready.parquet", index=False)

    logger.success("[Pipeline] Dataset com target de rankeamento salvo em data/processed/rank_ready.parquet")
    
    print("[SUCCESS] Pipeline executado com sucesso. Arquivo salvo em data/processed/merged.parquet")
    return df_merged
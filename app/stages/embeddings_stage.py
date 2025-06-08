import os
import pandas as pd
from app.embeddings import load_encoder, encode_texts


def generate_and_save_embeddings(input_path: str, output_path: str) -> pd.DataFrame:
    print("[INFO] Lendo merged.parquet...")
    df = pd.read_parquet(input_path)

    print("[INFO] Concatenando textos...")
    df["texto_vaga"] = df["titulo_vaga"].fillna('') + "\n" + df["principais_atividades"].fillna('') + "\n" + df["competencias"].fillna('')
    df["texto_cv"] = df["cv"].fillna('')
    df["texto_vaga_areas_atuacao"] = df["vaga_areas_atuacao"].fillna('')
    df["texto_cv_areas_atuacao"] = df["candidato_area_atuacao"].fillna('')

    encoder = load_encoder("sentence-transformers/all-MiniLM-L6-v2")

    emb_vaga = encode_texts(encoder, df["texto_vaga"].tolist())
    emb_cv = encode_texts(encoder, df["texto_cv"].tolist())
    emb_vaga_atuacao = encode_texts(encoder, df["texto_vaga_areas_atuacao"].tolist())
    emb_candidato_atuacao = encode_texts(encoder, df["texto_cv_areas_atuacao"].tolist())
    
    df["emb_vaga"] = list(emb_vaga)
    df["emb_cv"] = list(emb_cv)
    df["emb_vaga_areas_atuacao"]  = list(emb_vaga_atuacao)
    df["emb_candidato_area_atuacao"] = list(emb_candidato_atuacao)

    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, "combined_embeddings.parquet")
    df.to_parquet(full_path, index=False)

    print(f"[SUCCESS] Embeddings gerados e salvos em {full_path}")
    return df

import os
import pandas as pd
from app.embeddings import load_encoder, encode_texts


def generate_and_save_embeddings(input_path: str, output_path: str) -> pd.DataFrame:
    print("[INFO] Lendo merged.parquet...")
    df = pd.read_parquet(input_path)

    print("[INFO] Concatenando textos...")
    df["texto_vaga"] = (
        df["titulo_vaga"].fillna("")
        + "\n"
        + df["principais_atividades"].fillna("")
        + "\n"
        + df["competencias"].fillna("")
    )
    df["texto_cv"] = df["cv"].fillna("")

    encoder = load_encoder("sentence-transformers/all-MiniLM-L6-v2")

    emb_vaga = encode_texts(encoder, df["texto_vaga"].tolist())
    emb_cv = encode_texts(encoder, df["texto_cv"].tolist())

    df["emb_vaga"] = list(emb_vaga)
    df["emb_cv"] = list(emb_cv)

    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, "combined_embeddings.parquet")
    df.to_parquet(full_path, index=False)

    print(f"[SUCCESS] Embeddings gerados e salvos em {full_path}")
    return df

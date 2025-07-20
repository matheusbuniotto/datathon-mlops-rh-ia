from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"  #'paraphrase-multilingual-MiniLM-L12-v2'
)


def load_encoder(model_name: str = MODEL) -> SentenceTransformer:
    print("[INFO] Carregando modelo de embeddings...")
    return SentenceTransformer(model_name)


def encode_texts(
    model: SentenceTransformer, texts: List[str], batch_size: int = 48
) -> np.ndarray:
    print(f"[INFO] Gerando embeddings para {len(texts)} textos...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings

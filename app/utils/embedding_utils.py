import pandas as pd


def explode_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def explode_column(df, colname, prefix):
        arr = df[colname].tolist()
        arr_df = pd.DataFrame(arr, index=df.index)
        arr_df.columns = [f"{prefix}_{i}" for i in range(arr_df.shape[1])]
        return arr_df

    emb_vaga_df = explode_column(df, "emb_vaga", "emb_vaga")
    emb_cv_df = explode_column(df, "emb_cv", "emb_cv")

    df = df.drop(
        columns=[
            "emb_vaga",
            "emb_cv",
        ]
    )

    return pd.concat([df, emb_vaga_df, emb_cv_df], axis=1)

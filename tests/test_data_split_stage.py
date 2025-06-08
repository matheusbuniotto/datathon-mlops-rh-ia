import pandas as pd
from app.stages.data_split_stage import split_dataset_by_vaga


def test_split_dataset():
    df = pd.DataFrame({
        "codigo_vaga": [1]*3 + [2]*3 + [3]*4 + [4]*5,
        "codigo_candidato": list(range(15)),
        "target_rank": [5, 0, 1, 0, 0, 1, 2, 3, 4, 1, 5, 0, 1, 0, 0]
    })
    df_train, df_val, df_test = split_dataset_by_vaga(df, test_size=0.2, val_size=0.2)
    
    assert set(df_train["codigo_vaga"]).isdisjoint(df_val["codigo_vaga"])
    assert set(df_train["codigo_vaga"]).isdisjoint(df_test["codigo_vaga"])
    assert set(df_val["codigo_vaga"]).isdisjoint(df_test["codigo_vaga"])
    assert len(df_train) + len(df_val) + len(df_test) == len(df)

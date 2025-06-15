from app.pipeline import run_pipeline
from app.stages.feature_engineering_stage import apply_feature_pipeline
from app.model.save_pipeline import save_dataset_for_prediction
import pandas as pd

def run_model_pipeline():
    print("[INFO] Carregando dataset para rankeamento...")
    df = pd.read_parquet("data/processed/rank_ready.parquet")

    print("[INFO] Dividindo dataset em treino, validação e teste...")
    from app.stages.data_split_stage import split_dataset_by_vaga
    df_train, df_val, df_test = split_dataset_by_vaga(df)

    print("[INFO] Aplicando pipeline de feature engineering...")
    X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test, pipe = apply_feature_pipeline(
        df_train, df_val, df_test
    )

    print("[SUCCESS] Pipeline de feature engineering concluído e input do modelo salvo.")

if __name__ == "__main__":
    run_pipeline()
    run_model_pipeline()
    save_dataset_for_prediction()

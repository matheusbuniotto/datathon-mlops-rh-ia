import numpy as np
from joblib import load
from scipy import sparse
from sklearn.metrics import ndcg_score, average_precision_score
from loguru import logger
import os
import json

DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"


def load_inputs(dataset_type):
    """Carrega dados do conjunto especificado"""
    logger.info(f"[Eval] Carregando dados do conjunto: {dataset_type}")
    X = sparse.load_npz(os.path.join(DATA_PATH, f"X_{dataset_type}.npz"))
    y = np.load(os.path.join(DATA_PATH, f"y_{dataset_type}.npy"))
    group = np.load(os.path.join(DATA_PATH, f"group_{dataset_type}.npy"))
    logger.info(
        f"[Eval] Dados {dataset_type} carregados: {X.shape}, {y.shape}, {group.shape}"
    )
    return X, y, group


def evaluate_model(model, X, y, group, dataset_name, k=5, relevance_threshold=0):
    """Avalia modelo em um conjunto específico"""
    logger.info(f"[Eval] Avaliando modelo no conjunto: {dataset_name}")

    y_pred = model.predict(X)

    scores_ndcg = []
    scores_map = []
    total_groups = 0
    valid_groups = 0

    offset = 0
    for size in group:
        y_true_group = y[offset : offset + size]
        y_pred_group = y_pred[offset : offset + size]
        total_groups += 1

        if len(np.unique(y_true_group)) > 1:
            valid_groups += 1

            # NDCG funciona com múltiplos níveis de relevância
            scores_ndcg.append(ndcg_score([y_true_group], [y_pred_group], k=k))

            # Para MAP, binarizamos os rótulos (relevante vs não relevante)
            y_true_binary = (y_true_group > relevance_threshold).astype(int)

            # Só calcula MAP se houver pelo menos um item relevante
            if np.sum(y_true_binary) > 0:
                scores_map.append(average_precision_score(y_true_binary, y_pred_group))

        offset += size

    # Calcula métricas
    ndcg_mean = np.mean(scores_ndcg) if scores_ndcg else 0.0
    map_mean = np.mean(scores_map) if scores_map else 0.0

    logger.success(f"[Eval {dataset_name.upper()}] NDCG@{k}: {ndcg_mean:.4f}")
    logger.success(f"[Eval {dataset_name.upper()}] MAP@{k}:  {map_mean:.4f}")
    logger.info(
        f"[Eval {dataset_name.upper()}] Grupos válidos: {valid_groups}/{total_groups}"
    )

    return {
        "dataset": dataset_name,
        "ndcg": ndcg_mean,
        "map": map_mean,
        "total_groups": total_groups,
        "valid_groups": valid_groups,
        "k": k,
        "relevance_threshold": relevance_threshold,
    }


def comprehensive_evaluation(model_path, k=5, relevance_threshold=0):
    """
    Avaliação completa seguindo melhores práticas:

    1. VAL: Para desenvolvimento e debugging (pode usar quantas vezes quiser)
    2. TEST: Para avaliação final apenas (use UMA VEZ no final)
    """

    logger.info("=" * 60)
    logger.info("🎯 AVALIAÇÃO SEGUINDO MELHORES PRÁTICAS")
    logger.info("=" * 60)

    # Carrega modelo
    model = load(model_path)
    logger.success(f"[Eval] Modelo carregado de: {model_path}")

    results = {}

    # 1. AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO
    logger.info("\n" + "=" * 40)
    logger.info("📊 FASE 1: AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
    logger.info("💡 Use esta métrica para desenvolvimento e tuning")
    logger.info("=" * 40)

    try:
        X_val, y_val, group_val = load_inputs("val")
        results["validation"] = evaluate_model(
            model, X_val, y_val, group_val, "validation", k, relevance_threshold
        )
    except FileNotFoundError:
        logger.warning("[Eval] Conjunto de validação não encontrado!")
        results["validation"] = None

    # 2. AVALIAÇÃO NO CONJUNTO DE TESTE
    logger.info("\n" + "=" * 40)
    logger.info("🔒 FASE 2: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    logger.info("⚠️  ATENÇÃO: Use apenas UMA VEZ para avaliação final!")
    logger.info("=" * 40)

    try:
        X_test, y_test, group_test = load_inputs("test")
        results["test"] = evaluate_model(
            model, X_test, y_test, group_test, "test", k, relevance_threshold
        )
    except FileNotFoundError:
        logger.error("[Eval] Conjunto de teste não encontrado!")
        results["test"] = None

    # 3. RELATÓRIO FINAL
    logger.info("\n" + "=" * 50)
    logger.info("📋 RELATÓRIO FINAL DE AVALIAÇÃO")
    logger.info("=" * 50)

    if results["validation"]:
        val_ndcg = results["validation"]["ndcg"]
        logger.info("🔧 VALIDAÇÃO (para desenvolvimento):")
        logger.info(f"   NDCG@{k}: {val_ndcg:.4f}")
        logger.info(f"   MAP@{k}:  {results['validation']['map']:.4f}")

    if results["test"]:
        test_ndcg = results["test"]["ndcg"]
        logger.info("🎯 TESTE (performance final):")
        logger.info(f"   NDCG@{k}: {test_ndcg:.4f}")
        logger.info(f"   MAP@{k}:  {results['test']['map']:.4f}")

        # Análise de overfitting
        if results["validation"]:
            diff = val_ndcg - test_ndcg
            if abs(diff) < 0.02:
                logger.success(f"✅ Modelo bem generalizado (diff: {diff:+.4f})")
            elif diff > 0.02:
                logger.warning(f"⚠️  Possível overfitting (val-test: {diff:+.4f})")
            else:
                logger.info(f"🤔 Teste melhor que validação (diff: {diff:+.4f})")

    # Salva resultados
    results_path = "app/model/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"📁 Resultados salvos em: {results_path}")

    return results


def development_evaluation(model_path, k=5, relevance_threshold=0):
    """
    Avaliação APENAS no conjunto de validação
    Use esta função durante o desenvolvimento/tuning
    """
    logger.info("🔧 AVALIAÇÃO DE DESENVOLVIMENTO (apenas validação)")

    model = load(model_path)
    X_val, y_val, group_val = load_inputs("val")

    return evaluate_model(
        model, X_val, y_val, group_val, "validation", k, relevance_threshold
    )


def final_test_evaluation(model_path, k=5, relevance_threshold=0):
    """
    Avaliação APENAS no conjunto de teste
    Use esta função UMA VEZ para avaliação final
    """
    logger.warning("🔒 AVALIAÇÃO FINAL - USE APENAS UMA VEZ!")
    logger.warning("⚠️  Esta é sua avaliação definitiva do modelo!")

    model = load(model_path)
    X_test, y_test, group_test = load_inputs("test")

    return evaluate_model(
        model, X_test, y_test, group_test, "test", k, relevance_threshold
    )


if __name__ == "__main__":
    import sys

    # Verifica argumentos da linha de comando
    evaluation_type = sys.argv[1] if len(sys.argv) > 1 else "comprehensive"

    if evaluation_type == "dev":
        # Para desenvolvimento - pode usar quantas vezes quiser
        logger.info("🔧 Executando avaliação de desenvolvimento...")
        result = development_evaluation(MODEL_PATH, k=5, relevance_threshold=0)

    elif evaluation_type == "final":
        # Para avaliação final - use apenas uma vez!
        logger.warning("🔒 Executando avaliação final...")
        response = input("⚠️  TEM CERTEZA? Esta é a avaliação final! (yes/no): ")
        if response.lower() == "yes":
            result = final_test_evaluation(MODEL_PATH, k=5, relevance_threshold=0.2)
        else:
            logger.info("Avaliação cancelada.")
            exit()

    else:
        # Avaliação completa (padrão)
        logger.info("📊 Executando avaliação completa...")
        results = comprehensive_evaluation(MODEL_PATH, k=5, relevance_threshold=0.2)

    logger.success("✅ Avaliação concluída!")

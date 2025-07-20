import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from scipy import sparse
from sklearn.metrics import ndcg_score, average_precision_score
import os
import json
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# Configuração de estilo
plt.style.use("default")
sns.set_palette("husl")

# Caminhos dos seus arquivos
DATA_PATH = "data/model_input/"
MODEL_PATH = "app/model/lgbm_ranker.pkl"
RESULTS_PATH = "app/model/evaluation_results.json"


class RankingAnalyzer:
    """Classe para análise completa dos resultados de ranking"""

    def __init__(self, model_path=MODEL_PATH, data_path=DATA_PATH):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.results = {}

    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            self.model = load(self.model_path)
            logger.success(f"✅ Modelo carregado: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False

    def load_inputs(self, dataset_type):
        """Carrega dados do conjunto especificado"""
        try:
            X = sparse.load_npz(os.path.join(self.data_path, f"X_{dataset_type}.npz"))
            y = np.load(os.path.join(self.data_path, f"y_{dataset_type}.npy"))
            group = np.load(os.path.join(self.data_path, f"group_{dataset_type}.npy"))
            logger.info(
                f"📁 Dados {dataset_type} carregados: X{X.shape}, y{y.shape}, group{group.shape}"
            )
            return X, y, group
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {dataset_type}: {e}")
            return None, None, None

    def analyze_predictions_vs_labels(self, y_true, y_pred, dataset_name="Dataset"):
        """Análise detalhada das predições vs labels verdadeiros"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Análise de Predições - {dataset_name}", fontsize=16, fontweight="bold"
        )

        # 1. Distribuição dos scores preditos
        axes[0, 0].hist(y_pred, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Distribuição dos Scores Preditos")
        axes[0, 0].set_xlabel("Score Predito")
        axes[0, 0].set_ylabel("Frequência")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Distribuição dos labels verdadeiros
        axes[0, 1].hist(
            y_true,
            bins=np.arange(y_true.min(), y_true.max() + 2) - 0.5,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 1].set_title("Distribuição dos Labels Verdadeiros")
        axes[0, 1].set_xlabel("Relevância Verdadeira")
        axes[0, 1].set_ylabel("Frequência")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Scatter plot predito vs verdadeiro
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5, s=10, color="purple")
        axes[1, 0].plot(
            [y_true.min(), y_true.max()], [y_pred.min(), y_pred.max()], "r--", lw=2
        )
        axes[1, 0].set_xlabel("Relevância Verdadeira")
        axes[1, 0].set_ylabel("Score Predito")
        axes[1, 0].set_title("Predições vs Labels Verdadeiros")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Box plot por nível de relevância
        df_analysis = pd.DataFrame(
            {"true_relevance": y_true, "predicted_score": y_pred}
        )

        unique_relevances = sorted(df_analysis["true_relevance"].unique())
        box_data = [
            df_analysis[df_analysis["true_relevance"] == rel]["predicted_score"].values
            for rel in unique_relevances
        ]

        axes[1, 1].boxplot(box_data, labels=unique_relevances)
        axes[1, 1].set_title("Distribuição de Scores por Nível de Relevância")
        axes[1, 1].set_xlabel("Nível de Relevância")
        axes[1, 1].set_ylabel("Score Predito")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Estatísticas por grupo de relevância
        print(f"\n📊 === ESTATÍSTICAS POR NÍVEL DE RELEVÂNCIA - {dataset_name} ===")
        stats = (
            df_analysis.groupby("true_relevance")["predicted_score"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        print(stats)

        return df_analysis

    def detailed_group_analysis(self, X, y, group, dataset_name="Dataset"):
        """Análise detalhada por vaga (queries)"""

        if self.model is None:
            logger.error("❌ Modelo não carregado!")
            return

        logger.info(f"🔍 Analisando {len(group)} vaga em {dataset_name}...")

        y_pred = self.model.predict(X)

        group_metrics = []
        offset = 0

        for i, size in enumerate(group):
            y_true_group = y[offset : offset + size]
            y_pred_group = y_pred[offset : offset + size]

            # Calcula métricas apenas se há variabilidade nos labels
            if len(np.unique(y_true_group)) > 1:
                ndcg_5 = ndcg_score([y_true_group], [y_pred_group], k=5)
                ndcg_10 = ndcg_score([y_true_group], [y_pred_group], k=10)

                # MAP para labels binarizados
                y_true_binary = (y_true_group > 0).astype(int)
                if np.sum(y_true_binary) > 0:
                    map_score = average_precision_score(y_true_binary, y_pred_group)
                else:
                    map_score = 0.0

                group_metrics.append(
                    {
                        "group_id": i,
                        "group_size": size,
                        "ndcg_5": ndcg_5,
                        "ndcg_10": ndcg_10,
                        "map_score": map_score,
                        "relevance_levels": len(np.unique(y_true_group)),
                        "max_relevance": y_true_group.max(),
                        "mean_prediction": y_pred_group.mean(),
                        "std_prediction": y_pred_group.std(),
                    }
                )

            offset += size

        # Converte para DataFrame
        group_df = pd.DataFrame(group_metrics)

        # Visualizações
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Análise por vagas (Queries) - {dataset_name}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Distribuição de NDCG@5
        axes[0, 0].hist(
            group_df["ndcg_5"], bins=30, alpha=0.7, color="lightblue", edgecolor="black"
        )
        axes[0, 0].set_title("Distribuição NDCG@5 por Grupo")
        axes[0, 0].set_xlabel("NDCG@5")
        axes[0, 0].set_ylabel("Frequência")
        axes[0, 0].axvline(
            group_df["ndcg_5"].mean(),
            color="red",
            linestyle="--",
            label=f"Média: {group_df['ndcg_5'].mean():.3f}",
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Distribuição de MAP
        axes[0, 1].hist(
            group_df["map_score"],
            bins=30,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].set_title("Distribuição MAP por Grupo")
        axes[0, 1].set_xlabel("MAP Score")
        axes[0, 1].set_ylabel("Frequência")
        axes[0, 1].axvline(
            group_df["map_score"].mean(),
            color="red",
            linestyle="--",
            label=f"Média: {group_df['map_score'].mean():.3f}",
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. NDCG vs Tamanho do Grupo
        axes[0, 2].scatter(
            group_df["group_size"], group_df["ndcg_5"], alpha=0.6, color="purple"
        )
        axes[0, 2].set_title("NDCG@5 vs Tamanho do Grupo")
        axes[0, 2].set_xlabel("Tamanho do Grupo")
        axes[0, 2].set_ylabel("NDCG@5")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Box plot NDCG por Max Relevance
        relevance_groups = group_df.groupby("max_relevance")["ndcg_5"].apply(list)
        axes[1, 0].boxplot(relevance_groups.values, labels=relevance_groups.index)
        axes[1, 0].set_title("NDCG@5 por Máxima Relevância")
        axes[1, 0].set_xlabel("Máxima Relevância no Grupo")
        axes[1, 0].set_ylabel("NDCG@5")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Correlação entre métricas
        axes[1, 1].scatter(
            group_df["ndcg_5"], group_df["map_score"], alpha=0.6, color="orange"
        )
        axes[1, 1].set_title("Correlação NDCG@5 vs MAP")
        axes[1, 1].set_xlabel("NDCG@5")
        axes[1, 1].set_ylabel("MAP Score")

        # Calcular correlação
        correlation = group_df["ndcg_5"].corr(group_df["map_score"])
        axes[1, 1].text(
            0.05,
            0.95,
            f"Correlação: {correlation:.3f}",
            transform=axes[1, 1].transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Top 10 piores vagas
        worst_groups = group_df.nsmallest(10, "ndcg_5")
        axes[1, 2].bar(
            range(len(worst_groups)), worst_groups["ndcg_5"], color="red", alpha=0.7
        )
        axes[1, 2].set_title("Top 10 vagas com Menor NDCG@5")
        axes[1, 2].set_xlabel("Grupo (ID)")
        axes[1, 2].set_ylabel("NDCG@5")
        axes[1, 2].set_xticks(range(len(worst_groups)))
        axes[1, 2].set_xticklabels(worst_groups["group_id"], rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Estatísticas resumo
        print(f"\n📈 === ESTATÍSTICAS POR vagas - {dataset_name} ===")
        print(f"Total de vagas analisados: {len(group_df)}")
        print(
            f"NDCG@5 médio: {group_df['ndcg_5'].mean():.4f} ± {group_df['ndcg_5'].std():.4f}"
        )
        print(
            f"MAP médio: {group_df['map_score'].mean():.4f} ± {group_df['map_score'].std():.4f}"
        )
        print(
            f"vagas com NDCG@5 < 0.5: {len(group_df[group_df['ndcg_5'] < 0.5])} ({len(group_df[group_df['ndcg_5'] < 0.5]) / len(group_df) * 100:.1f}%)"
        )
        print(f"Tamanho médio dos vagas: {group_df['group_size'].mean():.1f}")

        return group_df

    def load_and_compare_results(self):
        """Carrega resultados do arquivo JSON e faz comparação"""

        if not os.path.exists(RESULTS_PATH):
            logger.warning(
                "⚠️ Arquivo de resultados não encontrado. Execute o script de avaliação primeiro!"
            )
            return None

        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)

        # Extrai métricas
        datasets = []
        ndcg_scores = []
        map_scores = []

        for dataset, metrics in results.items():
            if metrics:  # Se não for None
                datasets.append(dataset.title())
                ndcg_scores.append(metrics["ndcg"])
                map_scores.append(metrics["map"])

        if not datasets:
            logger.warning("⚠️ Nenhum resultado válido encontrado!")
            return None

        # Gráfico de comparação
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # NDCG Comparison
        bars1 = ax1.bar(
            datasets, ndcg_scores, color=["lightblue", "lightcoral"], alpha=0.8
        )
        ax1.set_title("Comparação NDCG@5", fontsize=14, fontweight="bold")
        ax1.set_ylabel("NDCG Score")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Adicionar valores nas barras
        for bar, value in zip(bars1, ndcg_scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # MAP Comparison
        bars2 = ax2.bar(datasets, map_scores, color=["lightgreen", "orange"], alpha=0.8)
        ax2.set_title("Comparação MAP@5", fontsize=14, fontweight="bold")
        ax2.set_ylabel("MAP Score")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Adicionar valores nas barras
        for bar, value in zip(bars2, map_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

        # Análise de overfitting
        if len(datasets) >= 2:
            val_ndcg = ndcg_scores[0] if "Validation" in datasets else None
            test_ndcg = ndcg_scores[1] if "Test" in datasets else ndcg_scores[0]

            if val_ndcg and test_ndcg:
                diff = val_ndcg - test_ndcg
                print("\n🎯 === ANÁLISE DE OVERFITTING ===")
                print(f"NDCG Validação: {val_ndcg:.4f}")
                print(f"NDCG Teste: {test_ndcg:.4f}")
                print(f"Diferença: {diff:+.4f}")

                if abs(diff) < 0.02:
                    print("✅ Modelo bem generalizado!")
                elif diff > 0.02:
                    print("⚠️ Possível overfitting detectado!")
                else:
                    print("🤔 Teste melhor que validação (incomum)")

        return results

    def analyze_feature_importance(self):
        """Analisa importância das features do modelo"""

        if self.model is None:
            if not self.load_model():
                return None

        try:
            # LightGBM feature importance
            importance = self.model.feature_importance(importance_type="gain")
            feature_names = [f"feature_{i}" for i in range(len(importance))]

            # Criar DataFrame
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)

            # Plot das top features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)

            bars = plt.barh(
                range(len(top_features)), top_features["importance"], color="skyblue"
            )
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("Feature Importance (Gain)")
            plt.title(
                "Top 20 Features Mais Importantes", fontsize=14, fontweight="bold"
            )
            plt.gca().invert_yaxis()

            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, top_features["importance"])):
                plt.text(
                    bar.get_width() + max(top_features["importance"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    ha="left",
                    va="center",
                )

            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.show()

            print("📊 Top 10 Features Mais Importantes:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.0f}")

            return importance_df

        except Exception as e:
            logger.error(f"❌ Erro ao analisar feature importance: {e}")
            return None

    def run_complete_analysis(self):
        """Executa análise completa dos resultados"""

        print("🚀 === ANÁLISE COMPLETA DE RESULTADOS LEARN TO RANK ===\n")

        # 1. Carregar modelo
        if not self.load_model():
            return

        # 2. Análise dos resultados salvos
        print("📊 1. COMPARAÇÃO DE RESULTADOS ENTRE CONJUNTOS")
        print("=" * 50)
        self.load_and_compare_results()

        # 3. Análise de feature importance
        print("\n🎯 2. ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
        print("=" * 50)
        self.analyze_feature_importance()

        # 4. Análise detalhada no conjunto de validação
        print("\n🔍 3. ANÁLISE DETALHADA - CONJUNTO DE VALIDAÇÃO")
        print("=" * 50)
        X_val, y_val, group_val = self.load_inputs("val")
        if X_val is not None:
            y_pred_val = self.model.predict(X_val)
            self.analyze_predictions_vs_labels(y_val, y_pred_val, "Validação")
            self.detailed_group_analysis(X_val, y_val, group_val, "Validação")


# === EXECUÇÃO PRINCIPAL ===
def main():
    """Função principal para executar a análise"""

    # Criar analisador
    analyzer = RankingAnalyzer()

    # Executar análise completa
    analyzer.run_complete_analysis()


# === FUNÇÕES DE CONVENIÊNCIA ===
def quick_comparison():
    """Comparação rápida dos resultados"""
    analyzer = RankingAnalyzer()
    return analyzer.load_and_compare_results()


def analyze_validation_only():
    """Análise apenas no conjunto de validação"""
    analyzer = RankingAnalyzer()
    if analyzer.load_model():
        X_val, y_val, group_val = analyzer.load_inputs("val")
        if X_val is not None:
            y_pred = analyzer.model.predict(X_val)
            analyzer.analyze_predictions_vs_labels(y_val, y_pred, "Validação")
            return analyzer.detailed_group_analysis(
                X_val, y_val, group_val, "Validação"
            )


def show_feature_importance():
    """Mostra apenas a importância das features"""
    analyzer = RankingAnalyzer()
    return analyzer.analyze_feature_importance()


if __name__ == "__main__":
    # Executar análise completa
    main()
    analyze_validation_only()

    print("\n" + "=" * 60)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("💡 Para análises específicas, use:")
    print("   • quick_comparison() - comparação rápida")
    print("   • analyze_validation_only() - análise detalhada validação")
    print("   • show_feature_importance() - importância features")
    print("=" * 60)

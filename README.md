# RecrutaIA Rank - Sistema de Rankeamento de Candidatos

## Sobre o Projeto

Este projeto é um pipeline MLOps completo para classificação de candidatos para vagas de emprego usando aprendizado de máquina e práticas modernas de MLOps. Desenvolvido como projeto de conclusão do curso de **Machine Learning Engineering da FIAP**, demonstra uma abordagem prática e profissional para implementar, monitorar e avaliar modelos de classificação em produção.

## Principais Características

- **API de Machine Learning**: API REST usando FastAPI para servir predições de classificação de candidatos em tempo real
- **Pipeline MLOps Completo**: Pré-processamento de dados, engenharia de features, treinamento automatizado e avaliação usando LightGBM
- **Monitoramento Profissional**: Stack completa com Prometheus e Grafana para observabilidade, métricas de negócio e detecção de data drift
- **Containerização**: Arquitetura cloud-native com Docker e Docker Compose para deploy consistente
- **Embeddings Semânticos**: Utiliza sentence-transformers para análise inteligente de perfis e descrições de vagas
- **Avaliação Robusta**: Métricas específicas para ranking (NDCG, MAP) com análise detalhada de performance
- **Reprodutibilidade**: Ambiente completamente versionado e dependências controladas

## 🚀 Demo Rápida: Sistema Completo em 1 Comando

**Quer ver o sistema funcionando? Execute um comando e tenha uma solução MLOps completa rodando:**

```bash
# Clone e inicie tudo
git clone https://github.com/matheusbuniotto/datathon-mlops-rh-ia.git
cd datathon-mlops-rh-ia
docker-compose up --build
```

**🎯 O que você obtém instantaneamente:**
- ✅ **API de ML em produção** com modelos treinados → `http://localhost:8000`
- ✅ **Dashboard de monitoramento** profissional → `http://localhost:3000`  
- ✅ **Coleta de métricas** em tempo real → `http://localhost:9090`
- ✅ **Zero configuração** - ambiente pronto para demonstração

**🧪 Teste a API:**
```bash
# Verificação de saúde
curl http://localhost:8000/health

# Obter posições de trabalho disponíveis
curl "http://localhost:8000/v1/list-vagas"

# Obter candidatos ranqueados
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
```

**📊 Explore o Dashboard:**
- Métricas de performance da API e latência
- Análise de drift de dados e qualidade do modelo
- Dashboards profissionais com insights de negócio

## 🛠️ Desenvolvimento: Explorando o Pipeline Completo

**Interessado em entender como tudo funciona? Aqui está o workflow completo de desenvolvimento:**

### 1. Configuração do Ambiente
```bash
# Instale dependências (recomendado: use uv)
uv sync

# Instale o pacote em modo desenvolvimento
uv pip install -e .
```

### 2. Configuração Interativa com Opções de Dados
```bash
# Configuração interativa - escolha dados de amostra ou reais
uv run scripts/quick_start.py

# Siga as instruções para selecionar:
# 1. Dados de amostra (100 registros) - Demo rápida
# 2. Dados reais (download automático) - Performance completa
```

### 3. Pipeline e Treinamento Manual
```bash
# Baixar dados de produção reais (se necessário)
uv run scripts/download_data.py

# Executar pipeline completo de dados
uv run app/pipeline_run_all.py

# Treinar modelo com ajuste de hiperparâmetros
uv run app/model/train_ranker_tuning.py dev

# Avaliar performance do modelo
uv run app/model/evaluate_ranker.py

# Executar API local
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Ferramentas de Desenvolvimento
```bash
# Executar testes
pytest

# Qualidade do código
uv run ruff check .
uv run ruff format .
```

### 📊 Informações sobre Dados

**Dados de Amostra (incluídos no repositório):**
- `sample_applicants.json` (50 candidatos, 212KB)
- `sample_vagas.json` (20 posições, 44KB)  
- `sample_prospects.json` (30 prospects, 36KB)

**Dados de Produção (baixados automaticamente do GitHub Releases):**
- `applicants.json` (194MB) - Base completa de candidatos
- `vagas.json` (37MB) - Posições de trabalho completas
- `prospects.json` (21MB) - Todos os dados de prospects

## Principais Endpoints

- `GET /health`: Verificação de saúde do serviço
- `GET /v1/recommend_ranked?vaga_id={id}&top_n={n}`: Obter recomendações de candidatos ranqueados para uma vaga específica
- `GET /v1/list-vagas`: Retorna todos os IDs de vagas disponíveis para usar no endpoint de recomendação
- `GET /metrics`: Endpoint de métricas do Prometheus

## Visão Geral da Arquitetura

O sistema segue uma arquitetura de pipeline ML em estágios:

### Componentes Principais

1. **Pipeline de Dados (`app/pipeline.py`)**: Orquestra o fluxo completo de processamento de dados
   - Conversão de dados JSON brutos → Parquet
   - Fusão de dados baseada em SQL via DuckDB
   - Geração de embeddings usando sentence-transformers
   - Preparação do dataset de classificação

2. **Estágios do Pipeline ML (`app/stages/`)**:
   - `embeddings_stage.py`: Gera embeddings semânticos para descrições de vagas e perfis de candidatos
   - `ranking_preparation_stage.py`: Cria dados de treinamento para modelo de classificação com alvos de relevância
   - `feature_engineering_stage.py`: Engenharia de features e pré-processamento
   - `data_split_stage.py`: Divisão de dados para treinamento/validação/teste

3. **Treinamento do Modelo (`app/model/`)**:
   - `train_ranker.py`: Treinamento do modelo de classificação LightGBM
   - `train_ranker_tuning.py`: Otimização de hiperparâmetros com Optuna
   - `evaluate_ranker.py`: Avaliação do modelo com métricas de classificação (NDCG, MAP)

4. **Serviço de API (`services/api/`)**:
   - API REST baseada em FastAPI
   - Predições de classificação de candidatos em tempo real
   - Integração com métricas do Prometheus
   - Monitoramento de saúde

5. **Stack de Monitoramento (`services/monitoring/`)**:
   - Prometheus para coleta de métricas
   - Grafana para visualização e dashboards
   - Métricas de negócio personalizadas e monitoramento de drift de dados

### Fluxo de Dados

```
Dados Brutos (JSON) → Pipeline de Dados → Embeddings → Engenharia de Features → Treinamento do Modelo → Implantação da API
                                             ↓
                                      Monitoramento & Avaliação
```

### Artefatos de Dados Principais

- `data/processed/merged.parquet`: Dados de recrutamento unificados
- `data/embeddings/combined_embeddings.parquet`: Embeddings semânticos para todas as entidades
- `data/model_input/`: Features pré-processadas prontas para treinamento do modelo
- `models/lgbm_ranker.pkl`: Modelo LightGBM de classificação treinado

## Monitoramento

- **Prometheus**: `http://localhost:9090` (coleta de métricas)
- **Grafana**: `http://localhost:3000` (admin/admin) (dashboards e visualização)
- Métricas personalizadas para performance da API e predições do modelo

## Exemplos de Chamadas à API

```bash
# Obter lista de todos os IDs de vagas disponíveis
curl "http://localhost:8000/v1/list-vagas"

# Obter recomendações de candidatos ranqueados para vagas específicas
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=6647&top_n=10"
```

## Requisitos

- Docker & Docker Compose
- Python 3.11+ (para desenvolvimento local)
- Veja `requirements-dev.txt` para dependências

## Detalhes Técnicos

- **Framework ML**: LightGBM para classificação com learning-to-rank baseado em grupos
- **Embeddings**: Gerados usando sentence-transformers (modelos multilíngues)
- **Processamento de Dados**: DuckDB para operações SQL eficientes e processamento de dados
- **Framework de API**: FastAPI com integração de métricas do Prometheus
- **Containerização**: Todos os serviços executam em containers Docker para implantação consistente
- **Avaliação do Modelo**: Métricas específicas de classificação (NDCG, MAP)
- **Monitoramento**: Capacidades de monitoramento de drift de dados em tempo real

## Contexto Acadêmico

Este projeto foi desenvolvido como **Projeto de Conclusão** do curso de **Machine Learning Engineering** da **FIAP** e implementa conceitos fundamentais de MLOps:

### Conceitos Aplicados
- **Versionamento de Modelos**: Controle completo de versões de dados, código e modelos
- **CI/CD para ML**: Pipeline automatizado de integração e deploy
- **Monitoramento em Produção**: Observabilidade completa com métricas customizadas
- **Reproducibilidade**: Ambientes containerizados e dependências fixas  
- **Data Drift Detection**: Monitoramento de qualidade e mudanças nos dados
- **Model Serving**: API profissional para servir modelos em produção

### Arquitetura Enterprise
O sistema demonstra uma arquitetura robusta seguindo boas práticas da indústria, adequada para ambientes corporativos e soluções escaláveis.

## Dicas de Desenvolvimento

- Use `uv` para gerenciamento mais rápido de dependências quando disponível
- Os notebooks em `notebooks/` são para exploração e podem precisar de limpeza
- Artefatos do modelo são salvos nos diretórios `app/model/` e `models/`
- Todos os estágios principais do pipeline têm arquivos de teste correspondentes
- Ruff está configurado para excluir notebooks Jupyter do linting

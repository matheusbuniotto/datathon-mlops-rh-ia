# RecrutaIA Rank - Datathon MLOps RH IA

## Visão Geral

Este projeto é um pipeline MLOps completo para classificação de candidatos para vagas de emprego usando aprendizado de máquina e práticas modernas de MLOps. Foi desenvolvido para o desafio Datathon MLOps RH IA e demonstra uma abordagem pronta para produção para implementar, monitorar e avaliar modelos de classificação.

## Principais Características

- **FastAPI**: API REST para servir predições de classificação de candidatos
- **Pipeline de Machine Learning**: Pré-processamento de dados, engenharia de features, treinamento de modelo e predição usando LightGBM e scikit-learn
- **Monitoramento**: Prometheus e Grafana integrados para monitoramento em tempo real da API e modelo
- **Dockerizado**: Todos os serviços (API, Prometheus, Grafana) executam em containers para fácil implantação
- **Notebooks**: Para exploração de dados, verificação de embeddings e testes com dados mock
- **Avaliação**: Scripts e ferramentas para avaliação robusta do modelo (NDCG, MAP, análise de grupos)
- **Reprodutibilidade**: Todas as dependências são fixas e rastreadas para ambientes consistentes

## Início Rápido (Novos Usuários)

**Acabou de baixar o repositório? Tenha uma demo funcionando em 5 minutos:**

```bash
# Configuração com um comando usando dados de amostra
uv run scripts/quick_start.py

# Siga as instruções exibidas para iniciar API e monitoramento
```

Isso configura tudo necessário para uma demo funcionando com dados de amostra (100 registros).

## Como Executar

### Docker (Recomendado para Pull & Run)

**Perfeito para: "Acabei de baixar o repositório e quero tudo funcionando"**

```bash
# Um comando - inicia API + Stack de Monitoramento
docker-compose up --build

# Só isso! Tudo estará disponível em:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000 (sem necessidade de login)
# - Prometheus: http://localhost:9090
```

**O que isso te dá:**
- ✅ **API pronta para uso** com modelos treinados
- ✅ **Dados de amostra** pré-carregados para demos
- ✅ **Stack completa de monitoramento** (Grafana + Prometheus)
- ✅ **Zero configuração local** necessária

**Teste a API:**
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/v1/list-vagas"
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
```

### Configuração de Desenvolvimento

1. **Configuração do Ambiente**
   ```bash
   # Instale dependências (recomendado: use uv)
   pip install -r requirements-dev.txt
   
   # Instale o pacote em modo de desenvolvimento
   pip install -e .
   ```

2. **Execute a API Localmente**
   ```bash
   uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Executar Testes**
   ```bash
   # Execute todos os testes
   pytest
   
   # Execute arquivos de teste específicos
   pytest tests/test_data_loader.py
   pytest tests/test_ranking_preparation.py
   ```

4. **Treinamento e Avaliação do Modelo**
   ```bash
   # Treine o modelo de classificação com otimização de hiperparâmetros
   uv run app/model/train_ranker_tuning.py dev
   
   # Treine o modelo com parâmetros fixos
   python app/model/train_ranker.py
   
   # Avalie o modelo treinado
   python app/model/evaluate_ranker.py
   ```

5. **Pipeline de Dados**
   ```bash
   # Execute o pipeline completo de dados (JSON → Parquet → Embeddings → Dataset de Classificação)
   python app/pipeline_run_all.py
   
   # Execute estágio individual do pipeline
   python app/pipeline.py
   ```

6. **Qualidade de Código**
   ```bash
   # Execute linting (Ruff está configurado no pyproject.toml)
   ruff check .
   
   # Formate o código
   ruff format .
   ```

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

## Dicas de Desenvolvimento

- Use `uv` para gerenciamento mais rápido de dependências quando disponível
- Os notebooks em `notebooks/` são para exploração e podem precisar de limpeza
- Artefatos do modelo são salvos nos diretórios `app/model/` e `models/`
- Todos os estágios principais do pipeline têm arquivos de teste correspondentes
- Ruff está configurado para excluir notebooks Jupyter do linting

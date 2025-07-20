# RecrutaIA Rank - Sistema de Rankeamento de Candidatos

## Resumo do Projeto

Este projeto apresenta um pipeline MLOps completo para ranqueamento de candidatos a vagas de emprego, integrando técnicas modernas de machine learning e práticas robustas de engenharia de MLOps. Desenvolvido como Trabalho de Conclusão do curso de **Pós-graduação em Machine Learning Engineering da FIAP**, o sistema exemplifica a implementação, monitoramento e avaliação de modelos de machine learning em ambiente produtivo.

A solução utiliza um modelo de **Learning-to-Rank** com LightGBM para classificar candidatos conforme sua compatibilidade com vagas específicas, fundamentando-se em dados históricos de contratações e desempenho em processos seletivos anteriores. 

### TLDR - Execução Rápida

- **Suba tudo com o comando:**  
    ```bash
    docker compose up --build
    ```
    - API: [http://localhost:8000](http://localhost:8000)
        - **Listar Vagas**: [http://localhost:8000/v1/list-vagas](http://localhost:8000/v1/list-vagas)
        - **Recomendar Candidatos**: [http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5](http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5)
    - Grafana: [http://localhost:3000](http://localhost:3000)
    - Prometheus: [http://localhost:9090](http://localhost:9090)

- **Principais scripts:**
    - API REST: `services/api/main.py`
    - Pipeline ETL: `app/pipeline.py`
    - Treinamento: `app/model/train_ranker_tuning.py`
    - Avaliação: `app/model/evaluate_ranker.py`

- **Para desenvolvimento local:**
    ```bash
    uv sync && uv pip install -e .
    uv run app/pipeline_run_all.py
    uv run app/model/train_ranker_tuning.py dev
    uv run app/model/evaluate_ranker.py
    uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

## Sobre o modelo utilizado

O sistema implementa um modelo de **Learning-to-Rank** usando LightGBM para ordenar candidatos por compatibilidade com vagas específicas, baseado no histórico de contratações e progressão em processos seletivos.

### Algoritmo e Arquitetura
- **LightGBM Ranker**: Gradient Boosting otimizado para ranking com objective `lambdarank`
- **Grouping Strategy**: Candidatos agrupados por `codigo_vaga` para ranking interno
- **Target Engineering**: Labels de relevância baseadas em outcomes históricos de contratação

### Features Principais
- **Embeddings Semânticos**: Representações vetoriais de CVs e descrições de vagas via sentence-transformers
- **Similarity Scores**: Cosine similarity entre embeddings de candidato e vaga
- **Features Categóricas**: Nível acadêmico, profissional, área de atuação, nível de inglês
- **Features Textuais**: Análise de compatibilidade textual entre requisitos e perfil

### Tratamento de Features
- **Preprocessing Pipeline**: StandardScaler para features numéricas, LabelEncoder para categóricas
- **Feature Engineering**: PCA para redução dimensional dos embeddings (preservando 95% da variância)
- **Missing Values**: Imputação com valores padrão específicos por tipo de feature
- **Text Processing**: Limpeza e normalização de campos textuais antes da geração de embeddings

### Target e Pesos de Ranking
- **Relevance Labels**: 
  - `Maior relevância`: Candidato contratado
  - `Média relevância`: Avançou para etapas finais
  - `Baixa relevância`: Participou do processo
  - `Não relevante`: Não selecionado
- **Group Weights**: Balanceamento por vaga para evitar bias em vagas com muitos candidatos
- **Class Weights**: Ajuste automático para classes desbalanceadas no dataset (contratados)

### Métricas de Avaliação
- **NDCG@k**: Normalized Discounted Cumulative Gain para avaliar qualidade do ranking
- **MAP**: Mean Average Precision para medir precisão das recomendações
- **Group-wise Metrics**: Avaliação específica por vaga para detectar variações de performance

### Hiperparâmetros Otimizados
- **Learning Rate**: 0.1 (otimizado via Optuna)
- **Num Leaves**: 31-127 (ajustado para evitar overfitting)
- **Max Depth**: 6-10 (balanceamento complexidade/generalização)
- **Regularização**: L1/L2 regularization para controle de overfitting

## Características Principais

### Componentes e desenvolvimento
- **API REST de Machine Learning**: Interface FastAPI para servir predições de classificação de candidatos em tempo real
- **Suite de Testes Abrangente**: 40 testes automatizados cobrindo API, integração e validação de dados com 78% de cobertura de código
- **Qualidade de Código**: Linting automatizado e formatação consistente usando Ruff
- **Arquitetura Modular**: Separação clara de responsabilidades entre pipeline de dados, treinamento e serving

### Pipeline MLOps
- **Processamento de Dados**: Sistema automatizado de pré-processamento, engenharia de features e preparação de datasets usando Duckdb e Python
- **Treinamento Automatizado**: Implementação LightGBM com otimização de hiperparâmetros via Optuna
- **Embeddings Semânticos**: Utilização de sentence-transformers para análise comparativa de perfis profissionais e descrições de vagas
- **Avaliação Robusta**: Métricas específicas para ranking (NDCG, MAP) com análise de performance

### Monitoramento e Observabilidade
- **Stack de Monitoramento**: Implementação Prometheus e Grafana para observabilidade completa
- **Métricas de Negócio**: Monitoramento de performance da API e qualidade das predições
- **Detecção de Data Drift**: Sistema automatizado de monitoramento de mudanças nos dados de produção
- **Dashboards**: Visualizações em tempo real via grafana

### Infraestrutura e Deploy
- **Containerização**: Arquitetura implementada com Docker e Docker COmpose
- **Reprodutibilidade**: Ambiente completamente versionado com gestão controlada de dependências
- **Preparação CI/CD**: Estrutura desenvolvida para integração contínua e deploy automatizado

## Demo do Sistema

### Execução Completa em Docker

O sistema pode ser executado integralmente através do docker compose, proporcionando um ambiente isolado e reproduzível:

```bash
git clone https://github.com/matheusbuniotto/datathon-mlops-rh-ia.git
cd datathon-mlops-rh-ia
docker compose up --build
```

### Componentes
- **API de Machine Learning**: Interface REST disponível em `http://localhost:8000`
- **Sistema de Monitoramento**: Dashboard Grafana acessível via `http://localhost:3000`
- **Coleta de Métricas**: Prometheus configurado em `http://localhost:9090`
- **Ambiente Zero-Config**: Sistema completamente configurado sem necessidade de setup manual

### Validação da API
Para validação e utilização da API como teste, por padrão é utilizada uma fração do dataset usado como teste.

```bash
# Verificação de saúde do sistema
curl http://localhost:8000/health

# Listagem de vagas disponíveis (uma fração do dataset de testes)
curl "http://localhost:8000/v1/list-vagas"

# Execução de predições de ranking
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
```

### Funcionalidades do Dashboard
- Métricas de performance e latência da API
- Análise de data drift e qualidade do modelo
- Dashboards executivos com insights de negócio

## Desenvolvimento e Configuração

### Configuração do Ambiente de Desenvolvimento

Para análise detalhada e desenvolvimento local, o sistema oferece configuração completa do ambiente:

### 1. Configuração do Ambiente
```bash
# Instalação de dependências utilizando uv
uv sync

# Instalação do pacote em modo desenvolvimento
uv pip install -e .
```

### 2. Configuração de Dados
```bash
# Configuração interativa para seleção de datasets
uv run scripts/quick_start.py

# Opções disponíveis:
# 1. Dados de amostra (100 registros) - Demonstração rápida do fluxo
# 2. Dados completos (download automático) - Análise completa
```

### 3. Execução do Pipeline e Treinamento
```bash
# Download de dados de produção (quando necessário)
uv run scripts/download_data.py

# Execução do pipeline completo de processamento
uv run app/pipeline_run_all.py

# Treinamento com otimização de hiperparâmetros
uv run app/model/train_ranker_tuning.py dev #o parametro dev usa dataset de validação.

# Avaliação de performance do modelo
uv run app/model/evaluate_ranker.py

# Execução da API em ambiente local
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Validação e Controle de Qualidade
```bash
# Execução da suite completa de testes (40 testes)
uv run pytest

# Execução de testes por módulo específico
uv run pytest tests/api/              # Testes de API (16 testes)
uv run pytest tests/integration/     # Testes de integração (12 testes)
uv run pytest tests/pipeline/        # Validação de dados (12 testes)

# Geração de relatório de cobertura
uv run pytest --cov=services.api --cov=app --cov-report=term-missing

# Verificação de qualidade de código
uv run ruff check .    # Análise estática
uv run ruff format .   # Formatação automática
```

### Especificação dos Datasets

**Dados de Amostra (incluídos no repositório):**
- `sample_applicants.json` - 50 candidatos (212KB)
- `sample_vagas.json` - 20 posições (44KB)  
- `sample_prospects.json` - 30 prospects (36KB)

**Dados de Produção (obtidos via GitHub Releases):**
- `applicants.json` - Base completa de candidatos (194MB)
- `vagas.json` - Posições de trabalho completas (37MB)
- `prospects.json` - Dados completos de prospects (21MB)

## Estrutura de Testes e Validação

O projeto implementa uma suite de testes abrangente seguindo padrões da indústria para garantir qualidade e confiabilidade em ambiente de produção.

### Arquitetura de Testes

```
tests/
├── api/                    # Testes de API (16 testes)
│   └── test_endpoints.py   # Validação completa de endpoints REST
├── integration/            # Testes de Integração (12 testes)
│   └── test_pipeline_integration.py  # Fluxo end-to-end do pipeline
└── pipeline/               # Validação de Dados (12 testes)
    └── test_data_validation.py      # Qualidade e consistência dos dados
```

### Cobertura de Testes Detalhada

#### Testes de API (`tests/api/`)
- **Endpoints Funcionais**: Validação de `/health`, `/list-vagas`, `/recommend_ranked`
- **Casos de Erro**: Tratamento de arquivos ausentes, parâmetros inválidos, falhas de predição
- **Validação de Entrada**: Tipos de dados, parâmetros obrigatórios, valores limites
- **Integração Real**: Testes com dados reais de produção
- **Métricas**: Verificação de endpoints do Prometheus

**Resultados**: 16/16 testes aprovados com 78% de cobertura do módulo API

#### Testes de Integração (`tests/integration/`)
- **Pipeline End-to-End**: Validação do fluxo completo de dados
- **Carregamento de Dados**: Verificação de integridade dos loaders
- **Artefatos de Modelo**: Validação de modelos treinados e pipelines
- **Predição Completa**: Teste do fluxo completo de predição
- **Performance**: Benchmarks básicos de tempo de resposta
- **Tratamento de Erros**: Validação de cenários de falha

**Resultados**: 12/12 testes aprovados com validação completa do pipeline

#### Validação de Dados (`tests/pipeline/`)
- **Schemas de Dados**: Verificação de estrutura e tipos de dados
- **Integridade Referencial**: Consistência entre datasets relacionados
- **Qualidade de Dados**: Detecção de valores ausentes e inconsistências
- **Regras de Negócio**: Validação de lógica específica do domínio
- **Monitoramento de Drift**: Verificação da configuração de monitoramento
- **Dados Temporais**: Validação de consistência temporal

**Resultados**: 12/12 testes aprovados com validação robusta de qualidade

### Métricas de Qualidade

- **40 testes totais** executados com 100% de aprovação
- **78% cobertura de código** no módulo crítico da API
- **Tempo de execução**: ~26 segundos para suite completa
- **Validação com dados reais**: 4,273 candidatos, 1,053 vagas
- **Automação completa**: Integração com pytest e relatórios de cobertura

### Comandos de Validação

```bash
# Validação rápida - testes críticos
uv run pytest tests/api/test_endpoints.py::TestHealthEndpoint -v

# Validação completa com relatório de cobertura
uv run pytest tests/ --cov=services.api --cov=app --cov-report=html

# Testes de performance
uv run pytest tests/integration/ -k "performance" -v
```

## Principais Endpoints

- `GET /health`: Verificação de saúde do serviço
- `GET /v1/recommend_ranked?vaga_id={id}&top_n={n}`: Obter recomendações de candidatos ranqueados para uma vaga específica
- `GET /v1/list-vagas`: Retorna todos os IDs de vagas disponíveis para usar no endpoint de recomendação
- `GET /metrics`: Endpoint de métricas do Prometheus

## Arquitetura do Sistema

### Visão Geral Técnica

O sistema implementa uma arquitetura MLOps moderna seguindo padrões da indústria, com separação clara de responsabilidades e foco em escalabilidade e manutenibilidade.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dados Brutos  │ -> │ Pipeline de ML   │ -> │  Serving API    │
│  (JSON/Parquet) │    │ (Treinamento)    │    │   (FastAPI)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Processamento  │    │   Validação &    │    │  Monitoramento  │
│   & Features    │    │     Testes       │    │  & Observability│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Componentes Detalhados

#### 1. Pipeline de Dados (`app/pipeline.py`)
**Responsabilidade**: Orquestração completa do fluxo de processamento de dados

- **Ingestão**: Conversão automática JSON → Parquet com validação de schema
- **Transformação**: Operações SQL complexas via DuckDB para performance otimizada  
- **Enriquecimento**: Geração de embeddings semânticos usando sentence-transformers
- **Preparação**: Criação de datasets prontos para treinamento de modelos

**Stack**: DuckDB, Pandas, Sentence-Transformers

#### 2. Estágios Modulares do ML (`app/stages/`)
**Responsabilidade**: Processamento especializado por etapa do pipeline

- **`embeddings_stage.py`**: Geração de representações vetoriais para texto de vagas/CVs
- **`ranking_preparation_stage.py`**: Construção de targets de relevância para learning-to-rank
- **`feature_engineering_stage.py`**: Criação e transformação de features preditivas
- **`data_split_stage.py`**: Divisão estratificada para treino/validação/teste

**Benefícios**: Modularidade, testabilidade, reutilização

#### 3. Treinamento e Avaliação (`app/model/`)
**Responsabilidade**: Desenvolvimento e validação de modelos ML

- **`train_ranker.py`**: Treinamento LightGBM com configurações otimizadas
- **`train_ranker_tuning.py`**: Otimização automática de hiperparâmetros via Optuna
- **`evaluate_ranker.py`**: Avaliação rigorosa com métricas específicas de ranking (NDCG, MAP)

**Algoritmo**: LightGBM Ranker com learning-to-rank baseado em grupos

#### 4. API de Serving (`services/api/`)
**Responsabilidade**: Disponibilização de modelos em produção

- **FastAPI Framework**: API REST moderna com documentação automática
- **Predições Real-time**: Endpoint otimizado para baixa latência
- **Monitoramento Integrado**: Métricas automáticas via Prometheus
- **Validação de Input**: Sanitização e validação rigorosa de parâmetros

**Performance**: < 2s tempo de resposta para predições

#### 5. Observabilidade (`services/monitoring/`)
**Responsabilidade**: Monitoramento e alertas em produção

- **Prometheus**: Coleta automática de métricas de sistema e negócio
- **Grafana**: Dashboards profissionais para stakeholders técnicos e de negócio
- **Data Drift Detection**: Monitoramento proativo de mudanças nos dados
- **Alertas**: Configuração de thresholds para degradação de performance

#### 6. Validação e Testes (`tests/`)
**Responsabilidade**: Garantia de qualidade e confiabilidade

- **Testes de API**: Validação completa de endpoints com casos reais
- **Testes de Integração**: Verificação end-to-end do pipeline
- **Validação de Dados**: Checks automáticos de qualidade e consistência
- **Cobertura**: 78% no módulo crítico da API

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

## Trabalho de Conclusão de Curso

### Contexto Acadêmico

Este projeto foi desenvolvido como Trabalho de Conclusão do curso de Pós-graduação em Machine Learning Engineering da FIAP, representando uma implementação prática e profissional de conceitos fundamentais de MLOps em um cenário real de classificação de candidatos.

### Objetivos do Trabalho

#### Objetivo Principal
Desenvolver um sistema MLOps completo para classificação de candidatos a vagas de emprego, demonstrando a aplicação prática de conceitos de engenharia de machine learning em ambiente de produção.

#### Objetivos Específicos
1. **Implementar Pipeline MLOps**: Desenvolver pipeline automatizado desde ingestão de dados até serving de modelos
2. **Garantir Qualidade**: Implementar suite de testes abrangente com alta cobertura
3. **Monitoramento Produtivo**: Criar sistema de observabilidade com métricas de negócio
4. **Reprodutibilidade**: Garantir ambiente completamente versionado e replicável
5. **Escalabilidade**: Arquitetura cloud-native preparada para crescimento

### Conceitos MLOps Implementados

#### DevOps para ML
- **Versionamento de Artefatos**: Controle de versões para dados, código e modelos
- **Containerização**: Arquitetura baseada em Docker para deploy consistente
- **Automação**: Pipeline automatizado de processamento e treinamento
- **Infraestrutura como Código**: Configuração versionada do ambiente

#### Validação e Qualidade
- **Testing Strategy**: 40+ testes cobrindo API, integração e validação de dados
- **Continuous Integration**: Estrutura preparada para CI/CD
- **Code Quality**: Linting automatizado e padrões de código
- **Data Quality**: Validação automática de esquemas e integridade

#### Monitoramento e Observabilidade
- **Model Monitoring**: Tracking de performance e degradação de modelos
- **Data Drift Detection**: Detecção automática de mudanças nos dados
- **Business Metrics**: Métricas específicas do domínio de recrutamento
- **Alerting**: Sistema de alertas para anomalias

#### Model Serving
- **API REST**: Endpoint profissional para servir predições
- **Low Latency**: Otimização para tempo de resposta < 2s
- **Scalability**: Arquitetura preparada para alta demanda
- **Health Checks**: Monitoramento de saúde em tempo real

### Tecnologias e Ferramentas

#### Stack de Machine Learning
- **LightGBM**: Algoritmo de gradient boosting para ranking
- **Sentence-Transformers**: Embeddings semânticos multilíngues
- **Optuna**: Otimização automática de hiperparâmetros
- **Scikit-learn**: Pipeline de preprocessing e métricas

#### Stack de Engenharia
- **FastAPI**: Framework moderno para APIs REST
- **Docker**: Containerização e orquestração
- **Prometheus + Grafana**: Stack completa de monitoramento
- **DuckDB**: Processamento analítico de dados

#### Stack de Qualidade
- **Pytest**: Framework de testes com alta cobertura
- **Ruff**: Linting e formatação de código
- **UV**: Gerenciamento rápido de dependências Python

### Resultados Alcançados

#### Métricas Técnicas
- **40 testes implementados** com 100% de aprovação
- **78% cobertura de código** no módulo crítico da API
- **< 2s tempo de resposta** para predições em produção
- **4,273 candidatos** e **1,053 vagas** processados nos testes

#### Entregáveis
- Sistema MLOps completo funcionando em containers
- API REST documentada e testada
- Pipeline de dados automatizado
- Suite de testes profissional
- Stack de monitoramento configurada
- Documentação técnica detalhada

### Contribuições e Diferenciais

1. **Arquitetura Enterprise**: Sistema desenvolvido seguindo padrões da indústria
2. **Testing-First**: Implementação com foco em qualidade desde o início
3. **Real-World Data**: Validação com dados reais de recrutamento
4. **Production-Ready**: Sistema preparado para ambiente produtivo
5. **Observabilidade Completa**: Monitoramento end-to-end implementado

### Aplicabilidade Profissional

Este trabalho demonstra competências essenciais para Machine Learning Engineers em empresas modernas:

- **Pensamento Sistêmico**: Visão completa do ciclo de vida de ML
- **Qualidade de Software**: Implementação de testes e boas práticas
- **Observabilidade**: Monitoramento proativo de sistemas em produção
- **Escalabilidade**: Arquitetura preparada para crescimento
- **Reprodutibilidade**: Garantia de replicabilidade científica

O sistema resultante representa uma implementação profissional que pode ser diretamente aplicada em cenários corporativos reais de classificação e recomendação.

## Estrutura do Projeto

```
datathon-mlops-rh-ia/
├── app/                          # Core da aplicação ML
│   ├── model/                    # Módulos de treinamento e avaliação
│   │   ├── train_ranker.py       # Treinamento LightGBM
│   │   ├── train_ranker_tuning.py# Otimização de hiperparâmetros
│   │   └── evaluate_ranker.py    # Avaliação de modelos
│   ├── stages/                   # Estágios do pipeline ML
│   │   ├── embeddings_stage.py   # Geração de embeddings
│   │   ├── ranking_preparation_stage.py
│   │   └── feature_engineering_stage.py
│   ├── prediction/               # Módulo de predição
│   │   └── predictor.py          # Lógica de predição em produção
│   ├── monitoring/               # Monitoramento e observabilidade
│   ├── utils/                    # Utilitários compartilhados
│   ├── data_loader.py            # Carregamento de dados
│   └── pipeline.py               # Orquestrador principal
├── services/                     # Serviços de produção
│   ├── api/                      # API REST FastAPI
│   │   ├── main.py               # Aplicação principal
│   │   ├── routes.py             # Definição de endpoints
│   │   └── model_loader.py       # Carregamento de modelos
│   └── monitoring/               # Stack de monitoramento
│       ├── prometheus/           # Configuração Prometheus
│       └── grafana/              # Dashboards Grafana
├── tests/                        # Suite de testes profissional
│   ├── api/                      # Testes de API (16 testes)
│   │   └── test_endpoints.py     # Validação completa de endpoints
│   ├── integration/              # Testes de integração (12 testes)
│   │   └── test_pipeline_integration.py
│   └── pipeline/                 # Validação de dados (12 testes)
│       └── test_data_validation.py
├── data/                         # Estrutura de dados
│   ├── raw/                      # Dados brutos (JSON)
│   ├── processed/                # Dados processados (Parquet)
│   ├── embeddings/               # Embeddings gerados
│   ├── model_input/              # Features para treinamento
│   ├── final/                    # Datasets finais
│   └── monitoring/               # Dados de monitoramento
├── models/                       # Artefatos de modelos treinados
├── notebooks/                    # Análise exploratória
├── scripts/                      # Scripts auxiliares
├── docker-compose.yml            # Orquestração de containers
├── pyproject.toml                # Configuração do projeto
├── requirements-dev.txt          # Dependências de desenvolvimento
└── CLAUDE.md                     # Instruções para desenvolvimento
```

### Fluxo de Trabalho Recomendado

#### Para Avaliação Rápida
```bash
# 1. Demonstração completa (1 comando)
docker-compose up --build

# 2. Verificação de funcionamento
curl http://localhost:8000/health
curl "http://localhost:8000/v1/list-vagas"
```

#### Para Análise Técnica
```bash
# 1. Execução de testes completos
uv run pytest tests/ --cov=services.api --cov=app

# 2. Verificação de qualidade do código
uv run ruff check .

# 3. Exploração da arquitetura
ls -la app/ services/ tests/
```

#### Para Desenvolvimento
```bash
# 1. Configuração do ambiente
uv sync && uv pip install -e .

# 2. Execução do pipeline
uv run app/pipeline_run_all.py

# 3. Treinamento do modelo
uv run app/model/train_ranker_tuning.py dev

# 4. Avaliação dos resultados
uv run app/model/evaluate_ranker.py
```

## Requisitos e Compatibilidade

### Requisitos Mínimos
- **Docker & Docker Compose** (para demonstração rápida)
- **Python 3.11+** (para desenvolvimento local)
- **8GB RAM** (recomendado para processamento completo)
- **2GB espaço em disco** (para dados e modelos)

### Compatibilidade
- **Linux** (Ubuntu 20.04+, CentOS 7+)
- **macOS** (Intel e Apple Silicon)
- **Windows** (WSL2 recomendado)
- **Cloud Platforms** (AWS, GCP, Azure)

### Dependências Principais
- **LightGBM 4.6+** - Algoritmo de ML principal
- **FastAPI 0.115+** - Framework da API
- **Sentence-Transformers 4.1+** - Embeddings semânticos
- **Prometheus/Grafana** - Stack de monitoramento
- **Pytest 8.4+** - Framework de testes

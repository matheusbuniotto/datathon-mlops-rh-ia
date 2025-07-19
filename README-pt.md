# Datathon MLOps RH IA

## Arquitetura

```
Usuário
   |
   v
[API FastAPI] ---> /metrics ---> [Prometheus] ---> [Grafana]
   |
   v
[Modelo de ML]
```

## Como Usar Docker

```bash
# Construa todos os serviços (necessário apenas se alterar código ou dependências)
docker-compose build

# Inicie todos os serviços
docker-compose up

# Reinicie apenas o serviço da API (sem rebuild)
docker-compose restart api
```

## Principais Endpoints

- `/health`: Verificação de saúde da API
- `/v1/recommend_ranked`: Recomendações de candidatos ranqueados para uma vaga específica
- `/v1/list-vagas`: Retorna todos os IDs de vagas disponíveis para usar no endpoint de recomendação

## Exemplos de Chamadas à API

```bash
# Obter lista de todos os IDs de vagas disponíveis
curl "http://localhost:8000/v1/list-vagas"

# Obter recomendações de candidatos ranqueados para vagas específicas
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=1650&top_n=5"
curl "http://localhost:8000/v1/recommend_ranked?vaga_id=6647&top_n=10"
```

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

## Exemplo de Chamada à API

```bash
curl "http://localhost:8000/predict"
```

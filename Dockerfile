FROM python:3.11-slim
WORKDIR /app

# Atualiza e instala deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements espeifico do docker
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copia todo o resto da aplicação
COPY . .

# Instala os pacotes
RUN pip install -e .

# Cria os diretórios necessários
RUN mkdir -p data/monitoring data/processed data/embeddings data/model_input data/final

# Expoe a porta
EXPOSE 8000

# Executa o entry point da api
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
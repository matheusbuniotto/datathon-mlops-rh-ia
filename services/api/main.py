from fastapi import FastAPI
from services.api.routes import router as recommend_router
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

app = FastAPI(title="RecrutaIA Rank API", version="1.0")

# Custom metric for model predictions
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_type"]
)

# FIX: Instrumentator fora 
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

@app.get("/health")
def health_check():
    logger.info("[API] Health check OK")
    return {"status": "ok"}

@app.get("/predict")
def predict():
    # ... your prediction logic ...
    MODEL_PREDICTIONS.labels(model_type="ranking").inc()
    return {"result": "ok"}

# Include recommendation routes
debug_prefix = "/v1"
app.include_router(recommend_router, prefix=debug_prefix)

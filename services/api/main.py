from fastapi import FastAPI, Request
from services.api.routes import router as recommend_router
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time
import threading
from pathlib import Path
from app.monitoring.monitor import monitoring_job

app = FastAPI(title="RecrutaIA Rank API", version="1.0")

MODEL_PREDICTIONS = Counter(
    "model_predictions_total", "Total number of model predictions", ["model_type"]
)

API_REQUESTS_TOTAL = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint"]
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

# Instrumentator exposes metrics at /metrics
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting RecrutaIA Rank API...")

    # Check and setup monitoring components
    project_root = Path(__file__).parent.parent.parent
    profile_path = project_root / "data" / "monitoring" / "reference_profile.json"

    if not profile_path.exists():
        logger.warning("⚠️  Reference profile not found - monitoring will be limited")
        logger.info(
            "💡 To enable full monitoring, run: uv run scripts/setup_monitoring.py"
        )
        logger.info("   Or run the data pipeline first to generate training data")
    else:
        logger.info("✅ Reference profile found - full monitoring enabled")

    logger.info("Starting monitoring job in background...")
    monitoring_thread = threading.Thread(target=monitoring_job, daemon=True)
    monitoring_thread.start()


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint).inc()
    API_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(
        duration
    )
    return response


@app.get("/health")
def health_check():
    logger.info("[API] Health check OK")
    return {"status": "ok"}


debug_prefix = "/v1"
app.include_router(recommend_router, prefix=debug_prefix)

from fastapi import FastAPI
from services.api.routes import router as recommend_router  
from loguru import logger

app = FastAPI(title="RecrutaIA Rank API", version="1.0")

@app.get("/health")
def health_check():
    logger.info("[API] Health check OK")
    return {"status": "ok"}

# Include recommendation routes
debug_prefix = "/v1"
app.include_router(recommend_router, prefix=debug_prefix)  # Uncomment this

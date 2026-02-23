from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(title="EdgeCare Triage - ML APIs")

# ✅ CORS: required for browser fetch from https://edgecare.onrender.com
# Fixes: OPTIONS 405 (preflight), CORS blocked requests for /ml/analyze-image, /ml/analyze-symptoms
ALLOWED_ORIGINS = [
    "https://edgecare.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],   # ✅ must include OPTIONS
    allow_headers=["*"],   # ✅ must include Content-Type, Authorization, etc.
)

# Your ML routes
app.include_router(router, prefix="/ml")

# Optional: quick root ping (useful for Render health checks)
@app.get("/")
def root():
    return {"status": "ok", "service": "edgecare-ml"}

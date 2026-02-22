from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="EdgeCare Triage - ML APIs")
app.include_router(router, prefix="/ml")

from fastapi import APIRouter, UploadFile, File
from app.schemas.symptoms import SymptomInput
from app.schemas.fusion import FusionInput
from app.ml.image_model import analyze_image
from app.ml.symptom_model import analyze_symptoms
from app.ml.fusion import fuse_results
from app.ml.report import generate_report
from app.ml.metrics import model_metrics
from app.ml.health import health_check
from app.schemas.report import ReportInput

router = APIRouter()

@router.post("/analyze-image")
async def analyze_image_api(file: UploadFile = File(...)):
    return analyze_image(file)

@router.post("/analyze-symptoms")
def analyze_symptoms_api(data: SymptomInput):
    return analyze_symptoms(data)

@router.post("/fuse-results")
def fuse_api(data: FusionInput):
    return fuse_results(data)


@router.post("/generate-report")
def report_api(payload: ReportInput):
    return generate_report(payload.dict())

@router.get("/model-metrics")
def metrics_api():
    return model_metrics()

@router.get("/health")
def health_api():
    return health_check()

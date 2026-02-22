from pydantic import BaseModel

class ReportInput(BaseModel):
    image_result: dict
    symptom_result: dict
    fusion_result: dict
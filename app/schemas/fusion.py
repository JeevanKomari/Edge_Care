from pydantic import BaseModel, Field

class FusionInput(BaseModel):
    """
    Input schema for fusion engine.
    Contains normalized severity scores generated
    by image-based and symptom-based ML models.
    """

    image_severity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized severity score from image classification model (0 to 1)"
    )

    symptom_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized risk score from symptom classification model (0 to 1)"
    )

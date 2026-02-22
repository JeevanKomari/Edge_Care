from pydantic import BaseModel, Field

class SymptomInput(BaseModel):
    """
    Input schema for symptom-based risk analysis.
    Data is collected through a guided patient questionnaire.
    """

    itching: bool = Field(
        ...,
        description="Indicates whether the patient is experiencing itching"
    )

    fever: bool = Field(
        ...,
        description="Indicates whether the patient has fever"
    )

    pain_level: int = Field(
        ...,
        ge=0,
        le=10,
        description="Patient-reported pain level on a scale of 0 to 10"
    )

    duration_days: int = Field(
        ...,
        ge=0,
        le=30,
        description="Duration of symptoms in days"
    )

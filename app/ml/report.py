import uuid
from datetime import datetime


def calculate_confidence_level(final_score: float) -> str:
    """
    Determines report confidence based on fused severity score.
    """
    if final_score is None:
        return "Unknown"
    elif final_score < 0.3:
        return "Low"
    elif final_score < 0.6:
        return "Medium"
    else:
        return "High"


def generate_report(payload):
    """
    Generates a structured, decision-ready triage report
    for physician review, based on ML inference results.
    """

    image_result = payload.get("image_result", {})
    symptom_result = payload.get("symptom_result", {})
    fusion_result = payload.get("fusion_result", {})

    final_score = fusion_result.get("final_severity_score")
    confidence_level = calculate_confidence_level(final_score)

    return {
        "report_id": f"TRIAGE-{uuid.uuid4().hex[:6].upper()}",
        "generated_at": datetime.utcnow().isoformat(),

        "ai_assessment_summary": {
            "final_severity_level": fusion_result.get("final_severity_level"),
            "final_severity_score": final_score,
            "recommended_action": fusion_result.get("recommended_action")
        },

        "model_outputs": {
            "image_analysis": {
                "predicted_class": image_result.get("predicted_class"),
                "confidence": image_result.get("confidence"),
                "all_probabilities": image_result.get("all_probabilities")
            },
            "symptom_analysis": {
                "risk_score": symptom_result.get("symptom_risk_score"),
                "severity_level": symptom_result.get("severity_level")
            }
        },

        "clinical_note_for_physician": (
            "This triage report is generated using AI-based analysis of images "
            "and self-reported symptoms. It is intended to support, not replace, "
            "professional clinical judgment."
        ),

        "confidence_level": confidence_level
    }

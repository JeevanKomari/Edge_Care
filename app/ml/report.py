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
    disease_result = image_result.get("disease", {}) if isinstance(image_result, dict) else {}

    final_score = fusion_result.get("final_severity_score")
    confidence_level = calculate_confidence_level(final_score)
    condition_name = (
        disease_result.get("display_name")
        or disease_result.get("predicted_class_display")
        or disease_result.get("predicted_class")
        or image_result.get("display_name")
        or image_result.get("predicted_class")
    )
    condition_confidence = disease_result.get("confidence")
    condition_source = disease_result.get("source")

    return {
        "report_id": f"TRIAGE-{uuid.uuid4().hex[:6].upper()}",
        "generated_at": datetime.utcnow().isoformat(),
        "condition": condition_name,
        "condition_prediction": condition_name,
        "gemini_condition": condition_name if condition_source == "gemini" else None,
        "confidence": condition_confidence,
        "condition_source": condition_source,

        "ai_assessment_summary": {
            "final_severity_level": fusion_result.get("final_severity_level"),
            "final_severity_score": final_score,
            "recommended_action": fusion_result.get("recommended_action"),
            "condition": condition_name,
        },

        "model_outputs": {
            "image_analysis": {
                "predicted_class": image_result.get("predicted_class"),
                "confidence": image_result.get("confidence"),
                "all_probabilities": image_result.get("all_probabilities"),
                "disease": disease_result,
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

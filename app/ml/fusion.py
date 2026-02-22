def fuse_results(data):
    """
    Safe fusion logic combining image severity and symptom risk
    with uncertainty handling and medical escalation rules.
    """

    image_score = data.image_severity_score
    symptom_score = data.symptom_risk_score

    # Base weighted fusion
    final_score = round((0.6 * image_score) + (0.4 * symptom_score), 2)

    # 🚨 SAFETY RULE 1: High image severity should NOT be downgraded
    if image_score >= 0.6:
        final_score = max(final_score, image_score)

    # 🚨 SAFETY RULE 2: Uncertain predictions (low confidence zone)
    if 0.3 <= image_score <= 0.45 and abs(image_score - symptom_score) < 0.1:
        return {
            "final_severity_score": final_score,
            "final_severity_level": "Uncertain",
            "recommended_action": "Severity unclear. Manual medical review recommended."
        }

    # Final severity classification
    if final_score < 0.2:
        level = "Normal"
        action = "No medical intervention required. Continue normal hygiene and monitoring."

    elif final_score < 0.4:
        level = "Mild"
        action = "Home care and observation advised. Consult a doctor if symptoms persist."

    elif final_score < 0.7:
        level = "Moderate"
        action = "Doctor consultation recommended within 24–48 hours."

    else:
        level = "Severe"
        action = "Immediate medical attention required. Please consult a healthcare professional urgently."

    return {
        "final_severity_score": final_score,
        "final_severity_level": level,
        "recommended_action": action
    }

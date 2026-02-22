import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# Lightweight symptom-based ML model (Logistic Regression)
# -------------------------------------------------

# Sample training data (used for demonstration & validation)
# Features: [itching, fever, pain_level, duration_days]
X = np.array([
    [1, 0, 2, 3],
    [0, 1, 4, 5],
    [1, 1, 3, 7],
    [0, 0, 1, 1]
])

# Labels: 0 = low risk, 1 = high risk
y = [0, 1, 1, 0]

model = LogisticRegression()
model.fit(X, y)


def analyze_symptoms(data):
    """
    Analyzes structured symptom data using a logistic regression model
    to estimate health risk and severity level.
    """

    features = np.array([[
        int(data.itching),
        int(data.fever),
        data.pain_level,
        data.duration_days
    ]])

    # Predict probability of high-risk condition
    risk_score = float(model.predict_proba(features)[0][1])
    risk_score = round(risk_score, 2)

    # Severity classification aligned with project definition
    if risk_score < 0.2:
        severity = "Normal"
    elif risk_score < 0.4:
        severity = "Mild"
    elif risk_score < 0.7:
        severity = "Moderate"
    else:
        severity = "Severe"

    return {
        "symptom_risk_score": risk_score,
        "severity_level": severity
    }

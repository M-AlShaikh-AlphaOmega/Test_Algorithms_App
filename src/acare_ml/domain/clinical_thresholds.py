CLINICAL_DECISION_THRESHOLD = 0.5

SEVERITY_THRESHOLDS = {
    "mild": (0.3, 0.5),
    "moderate": (0.5, 0.7),
    "severe": (0.7, 1.0),
}


def classify_severity(probability: float) -> str:
    for severity, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= probability < high:
            return severity
    return "unknown"


ACCEPTABLE_SENSITIVITY = 0.85
ACCEPTABLE_SPECIFICITY = 0.80

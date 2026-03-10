from typing import Dict, Optional, Tuple, List

# Canonical long labels and their short display names
LABEL_SHORT_MAP: Dict[str, str] = {
    "Acne and Rosacea Photos": "Acne / Rosacea",
    "Cellulitis Impetigo and other Bacterial Infections": "Bacterial Infection",
    "Eczema Photos": "Eczema",
    "Light Diseases and Disorders of Pigmentation": "Pigmentation Disorder",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma / Nevi",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses and other Benign Tumors": "Benign Tumor",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal Infection",
    "Urticaria Hives": "Hives (Urticaria)",
    "Warts Molluscum and other Viral Infections": "Viral Infection",
}

# Lower-case synonym → canonical long label
SYNONYM_MAP: Dict[str, str] = {
    "acne": "Acne and Rosacea Photos",
    "rosacea": "Acne and Rosacea Photos",
    "cellulitis": "Cellulitis Impetigo and other Bacterial Infections",
    "impetigo": "Cellulitis Impetigo and other Bacterial Infections",
    "bacterial infection": "Cellulitis Impetigo and other Bacterial Infections",
    "eczema": "Eczema Photos",
    "dermatitis": "Eczema Photos",
    "atopic dermatitis": "Eczema Photos",
    "pigmentation": "Light Diseases and Disorders of Pigmentation",
    "vitiligo": "Light Diseases and Disorders of Pigmentation",
    "melasma": "Light Diseases and Disorders of Pigmentation",
    "melanoma": "Melanoma Skin Cancer Nevi and Moles",
    "mole": "Melanoma Skin Cancer Nevi and Moles",
    "nevi": "Melanoma Skin Cancer Nevi and Moles",
    "nevus": "Melanoma Skin Cancer Nevi and Moles",
    "psoriasis": "Psoriasis pictures Lichen Planus and related diseases",
    "lichen planus": "Psoriasis pictures Lichen Planus and related diseases",
    "seborrheic keratosis": "Seborrheic Keratoses and other Benign Tumors",
    "benign tumor": "Seborrheic Keratoses and other Benign Tumors",
    "benign lesion": "Seborrheic Keratoses and other Benign Tumors",
    "tinea": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "ringworm": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "candidiasis": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "fungal infection": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "urticaria": "Urticaria Hives",
    "hives": "Urticaria Hives",
    "wart": "Warts Molluscum and other Viral Infections",
    "warts": "Warts Molluscum and other Viral Infections",
    "molluscum": "Warts Molluscum and other Viral Infections",
    "viral infection": "Warts Molluscum and other Viral Infections",
}


def normalize_condition(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Map a raw condition label (model/Gemini) to canonical & display names."""
    if not label:
        return None, None
    raw = label.strip()
    lower = raw.lower()
    canonical = None

    # Exact canonical match
    for long_name in LABEL_SHORT_MAP.keys():
        if lower == long_name.lower():
            canonical = long_name
            break

    # Synonym match
    if canonical is None:
        canonical = SYNONYM_MAP.get(lower)

    # Fallback: try startswith/contains for robustness
    if canonical is None:
        for key, value in SYNONYM_MAP.items():
            if key in lower:
                canonical = value
                break

    display = LABEL_SHORT_MAP.get(canonical) if canonical else None
    return canonical, display


def normalize_predictions(preds: List[Dict]) -> List[Dict]:
    """Attach canonical/display names to prediction dicts."""
    normed = []
    for pred in preds:
        label = pred.get("label")
        canonical, display = normalize_condition(label)
        new_pred = dict(pred)
        if canonical:
            new_pred["canonical_label"] = canonical
        if display:
            new_pred["display_name"] = display
        normed.append(new_pred)
    return normed

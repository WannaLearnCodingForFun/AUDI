"""Map Kaggle PV panel defect classes to AUDI condition labels."""

from __future__ import annotations

from typing import Dict, Tuple

# Kaggle class → (condition_description, defect category)
PV_CLASS_MAP: Dict[str, Tuple[str, str]] = {
    "Clean": ("Normal", "none"),
    "Dusty": ("Soiling", "soiling"),
    "Bird-drop": ("Surface contamination", "contamination"),
    "Electrical-damage": ("Electrical defect", "electrical"),
    "Physical-Damage": ("Physical defect", "physical"),
    "Physical-damage": ("Physical defect", "physical"),
    "Snow-Covered": ("Surface obstruction", "obstruction"),
    "Snow-covered": ("Surface obstruction", "obstruction"),
}

PV_CLASSES = list(PV_CLASS_MAP.keys())


def map_pv_class(source_class: str) -> Dict[str, str]:
    """Return condition and source metadata for a Kaggle class label."""
    condition, defect = PV_CLASS_MAP.get(source_class, ("Unknown", "unknown"))
    return {
        "condition_description": condition,
        "source_class": source_class,
        "defect_category": defect,
    }


def normalize_class_name(name: str) -> str:
    """Normalize directory/filename class labels to canonical form."""
    normalized = name.strip().replace("_", "-").replace(" ", "-")
    aliases = {
        "birddrop": "Bird-drop",
        "bird-drop": "Bird-drop",
        "electricaldamage": "Electrical-damage",
        "electrical-damage": "Electrical-damage",
        "physicaldamage": "Physical-damage",
        "physical-damage": "Physical-damage",
        "snowcovered": "Snow-covered",
        "snow-covered": "Snow-covered",
        "clean": "Clean",
        "dusty": "Dusty",
    }
    key = normalized.lower()
    for alias, canonical in aliases.items():
        if key == alias.lower():
            return canonical
    for canonical in PV_CLASSES:
        if key == canonical.lower():
            return canonical
    return name

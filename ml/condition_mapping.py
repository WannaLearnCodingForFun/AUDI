"""Explicit source-class → unified condition mapping (Phase 1).

Each rule maps a dataset source and a folder pattern (regex on the image's
directory path relative to dataset/raw/<source>/) to
(asset_type, unified_condition, defect_type, is_defect). Sources with
annotations (masks, VOC XML) use a named `handler` that the prepare script
implements. Anything that matches no rule is reported in STATS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

UNIFIED_CONDITIONS: Dict[str, List[str]] = {
    "solar_panel": ["normal", "soiling", "bird_drop", "electrical_damage", "physical_damage", "snow_cover"],
    "building": ["normal", "crack"],
    "utility_pole": ["normal", "insulator_defect"],
    "machinery": [
        "normal", "crack", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches", "corrosion", "anomaly",
    ],
}

# Classes below datasets.min_images_per_class are merged into this parent (same asset_type)
# or dropped when no parent is listed.
PARENT_CLASS: Dict[str, Dict[str, str]] = {
    "machinery": {
        "crack": "anomaly",
        "inclusion": "anomaly",
        "patches": "anomaly",
        "pitted_surface": "anomaly",
        "rolled_in_scale": "anomaly",
        "scratches": "anomaly",
    },
    "solar_panel": {"bird_drop": "soiling", "snow_cover": "soiling"},
}


@dataclass(frozen=True)
class Rule:
    source: str
    pattern: str                 # regex matched (case-insensitive) against the relative directory
    asset_type: str
    condition: Optional[str]     # None for handler-driven rules
    defect_type: Optional[str]
    is_defect: bool
    handler: Optional[str] = None
    note: str = ""
    # handler-driven rules: annotation class id/name → (condition, defect_type, is_defect)
    class_map: Optional[Dict[str, Tuple[str, Optional[str], bool]]] = field(default=None, hash=False, compare=False)

    def matches(self, rel_dir: str) -> bool:
        return re.search(self.pattern, rel_dir, flags=re.IGNORECASE) is not None


def _pv(source: str, prefix: str = r"(^|/)") -> List[Rule]:
    table = [
        ("Clean", "normal", None, False),
        ("Dusty", "soiling", "soiling", True),
        ("Bird-drop", "bird_drop", "bird_drop", True),
        ("Electrical-damage", "electrical_damage", "electrical_damage", True),
        ("Physical-Damage", "physical_damage", "physical_damage", True),
        ("Snow-Covered", "snow_cover", "snow_cover", True),
    ]
    return [Rule(source, prefix + re.escape(folder) + r"(/|$)", "solar_panel", cond, dt, d) for folder, cond, dt, d in table]


RULES: List[Rule] = [
    *_pv("pv_defect"),
    *_pv("solar_clean_faulty"),
    Rule("crack_concrete", r"(^|/)Positive$", "building", "crack", "crack", True),
    Rule("crack_concrete", r"(^|/)Negative$", "building", "normal", None, False),
    Rule("sdnet2018", r"(^|/)Non-cracked$", "building", "normal", None, False),
    Rule("sdnet2018", r"(^|/)Cracked$", "building", "crack", "crack", True),
    Rule(
        "uav_crack", r"(^|/)image$", "building", None, None, False, handler="mask_tiles",
        note="segmentation masks in sibling masks/: tiles without crack pixels → normal, tiles above the crack fraction → crack",
    ),
    Rule("uav_crack", r"(^|/)masks$", "building", None, None, False, handler="skip", note="masks consumed by mask_tiles"),
    Rule(
        "cplid_insulator", r"(^|/)(Normal|Defective)_Insulators/images$", "utility_pole", None, None, False,
        handler="voc_insulator",
        note="VOC boxes: each insulator box is cropped; a box containing a 'defect' box centre → insulator_defect "
             "(defective images are synthetic composites per the dataset README)",
        class_map={"insulator": ("normal", None, False), "defect": ("insulator_defect", "insulator_defect", True)},
    ),
    *[
        Rule("powerline_parts", rf"(^|/)\d+_{name}$", "utility_pole", "normal", None, False)
        for name in ("disk_insulator", "Pin_insulator", "dropout", "tf_bush", "pole", "cross_arm")
    ],
    Rule("powerline_parts", r"(^|/)\d+_defective_disk_insulator$", "utility_pole", "insulator_defect", "defective_disk_insulator", True),
    Rule("powerline_parts", r"(^|/)\d+_defective_pin_insulator$", "utility_pole", "insulator_defect", "defective_pin_insulator", True),
    Rule(
        "powerline_parts", r"(^|/)\d+_defe[fc]tive_dropout$", "utility_pole", "insulator_defect", "defective_dropout", True,
        note="dropout cut-out (porcelain insulator body); upstream folder is misspelt 'defeftive'",
    ),
    Rule(
        "powerline_parts", r"(^|/)\d+_defective_tf_bush$", "utility_pole", "insulator_defect", "defective_tf_bush", True,
        note="transformer bushing is an insulator component",
    ),
    Rule("neu_steel", r"/images/crazing$", "machinery", "crack", "crazing", True, note="crazing = network of fine surface cracks"),
    Rule("neu_steel", r"/images/inclusion$", "machinery", "inclusion", "inclusion", True),
    Rule("neu_steel", r"/images/patches$", "machinery", "patches", "patches", True),
    Rule("neu_steel", r"/images/pitted_surface$", "machinery", "pitted_surface", "pitted_surface", True),
    Rule("neu_steel", r"/images/rolled-in_scale$", "machinery", "rolled_in_scale", "rolled_in_scale", True),
    Rule("neu_steel", r"/images/scratches$", "machinery", "scratches", "scratches", True),
    Rule(
        "corrosion_detect", r"(^|/)images$", "machinery", None, None, False, handler="yolo_labels",
        note="YOLO boxes in sibling labels/; images with empty label files are skipped (unlabelled, not verified clean)",
        class_map={"1": ("corrosion", "corrosion", True)},
    ),
    Rule(
        "corrosion_rustguard", r"(^|/)(train|valid|test)/images$", "machinery", None, None, False, handler="yolo_labels",
        note="Roboflow polygons → bounding boxes; classes Metal / Metal corroido / Tornillo sin corrosion",
        class_map={
            "0": ("normal", None, False),
            "1": ("corrosion", "corrosion", True),
            "2": ("normal", None, False),
        },
    ),
    Rule("mvtec_ad", r"^[a-z_]+/(train|test)/good$", "machinery", "normal", None, False),
    Rule("mvtec_ad", r"^[a-z_]+/test/scratch[a-z_]*$", "machinery", "scratches", "scratches", True),
    Rule("mvtec_ad", r"^[a-z_]+/test/(?!good$)[a-z_]+$", "machinery", "anomaly", None, True,
         note="all other MVTec defect folders; defect_type is the folder name"),
]


def rules_for(source: str) -> List[Rule]:
    return [r for r in RULES if r.source == source]


def match_rule(source: str, rel_dir: str) -> Optional[Rule]:
    for rule in rules_for(source):
        if rule.matches(rel_dir):
            return rule
    return None


def is_defect_condition(condition: str) -> bool:
    return condition != "normal"

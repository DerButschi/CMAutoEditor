from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

DEFAULT_PROFILE_CONFIGS = {
    "black_sea": "default_osm_config_cmbs.json",
    "cold_war": "default_osm_config_cmcw.json",
    "fortress_italy": "default_osm_config_cmfi.json",
    "shock_force_2": "default_osm_config_cmsf2.json",
}

BUILDING_OUTLINE_PROCESSES = {
    "type_from_residential_building_outline",
    "type_from_church_outline",
    "type_from_barn_outline",
    "type_from_barn_outlines",
}


def test_default_osm_config_labels_match_profile_menus() -> None:
    for profile, config_name in DEFAULT_PROFILE_CONFIGS.items():
        menu_labels = _profile_menu_labels(profile)
        config = json.loads((REPO_ROOT / config_name).read_text(encoding="utf-8"))

        invalid_labels = []
        for entry_name, entry in config.items():
            if not isinstance(entry, dict) or _is_building_outline_entry(entry):
                continue
            for cm_type in entry.get("cm_types", ()):
                if not isinstance(cm_type, dict) or cm_type.get("dummy") is True:
                    continue
                for field in ("menu", "cat1", "cat2", "direction"):
                    value = cm_type.get(field)
                    if isinstance(value, str) and value not in menu_labels:
                        invalid_labels.append((entry_name, field, value))

        assert invalid_labels == []


def _profile_menu_labels(profile: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / "profiles" / profile / "menu.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MENU_DICT" and isinstance(node.value, ast.Dict):
                    return {
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
    return set()


def _is_building_outline_entry(entry: dict) -> bool:
    return any(process in BUILDING_OUTLINE_PROCESSES for process in entry.get("process", ()))

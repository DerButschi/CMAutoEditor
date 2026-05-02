from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppResources:
    app_root: Path
    executable_root: Path
    data_cache_path: Path
    config_dir: Path
    dll_dir: Path


_DLL_HANDLES: list[object] = []
_PREPARED_DLL_DIRS: set[Path] = set()


def resolve_resources() -> AppResources:
    if getattr(sys, "frozen", False):
        return _resolve_packaged_resources()
    return _resolve_source_resources()


def prepare_runtime_environment(resources: AppResources) -> None:
    dll_dir = resources.dll_dir.resolve()
    if dll_dir in _PREPARED_DLL_DIRS:
        return

    _prepend_path_once(dll_dir)
    if os.name == "nt" and dll_dir.exists() and hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    _PREPARED_DLL_DIRS.add(dll_dir)


def find_default_osm_configs(resources: AppResources) -> list[Path]:
    if not resources.config_dir.exists():
        return []
    return sorted(
        path
        for path in resources.config_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def _resolve_source_resources() -> AppResources:
    app_root = Path(__file__).resolve().parents[1]
    executable_root = Path.cwd().resolve()
    return AppResources(
        app_root=app_root,
        executable_root=executable_root,
        data_cache_path=executable_root / "data_cache",
        config_dir=executable_root,
        dll_dir=app_root / "dll",
    )


def _resolve_packaged_resources() -> AppResources:
    executable_root = Path(sys.executable).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", executable_root)).resolve()
    app_root = bundle_root / "cm_terrain_extractor_app"
    if not app_root.exists():
        app_root = Path(__file__).resolve().parents[1]

    bundle_dll_dir = bundle_root / "dll"
    executable_dll_dir = executable_root / "dll"
    dll_dir = bundle_dll_dir if bundle_dll_dir.exists() else executable_dll_dir

    return AppResources(
        app_root=app_root,
        executable_root=executable_root,
        data_cache_path=executable_root / "data_cache",
        config_dir=executable_root,
        dll_dir=dll_dir,
    )


def _prepend_path_once(path: Path) -> None:
    path_str = str(path)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized_entries = {str(Path(entry).resolve()) for entry in path_entries if entry}
    if path_str in normalized_entries:
        return
    os.environ["PATH"] = os.pathsep.join([path_str, *path_entries]) if path_entries else path_str

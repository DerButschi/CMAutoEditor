from __future__ import annotations

import os
import sys
from pathlib import Path


def _clear_packaged_flags(monkeypatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_resolve_resources_source_mode_uses_cwd_for_configs_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    from cm_terrain_extractor_app.app_core.resources import resolve_resources

    _clear_packaged_flags(monkeypatch)
    monkeypatch.chdir(tmp_path)

    resources = resolve_resources()

    assert resources.executable_root == tmp_path
    assert resources.config_dir == tmp_path
    assert resources.data_cache_path == tmp_path / "data_cache"
    assert resources.app_root.name == "cm_terrain_extractor_app"
    assert resources.dll_dir == resources.app_root / "dll"


def test_resolve_resources_packaged_mode_uses_exe_root_and_bundle_app(
    tmp_path: Path, monkeypatch
) -> None:
    from cm_terrain_extractor_app.app_core.resources import resolve_resources

    bundle_root = tmp_path / "bundle"
    executable_root = tmp_path / "dist"
    app_root = bundle_root / "cm_terrain_extractor_app"
    dll_dir = bundle_root / "dll"
    app_root.mkdir(parents=True)
    dll_dir.mkdir()
    executable_root.mkdir()
    executable = executable_root / "cm_terrain_extractor_app.exe"
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    resources = resolve_resources()

    assert resources.app_root == app_root
    assert resources.executable_root == executable_root
    assert resources.config_dir == executable_root
    assert resources.data_cache_path == executable_root / "data_cache"
    assert resources.dll_dir == dll_dir


def test_find_default_osm_configs_returns_sorted_json_files(tmp_path: Path) -> None:
    from cm_terrain_extractor_app.app_core.resources import (
        AppResources,
        find_default_osm_configs,
    )

    (tmp_path / "z_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("{}", encoding="utf-8")
    (tmp_path / "nested.json").mkdir()
    resources = AppResources(
        app_root=tmp_path / "app",
        executable_root=tmp_path,
        data_cache_path=tmp_path / "data_cache",
        config_dir=tmp_path,
        dll_dir=tmp_path / "dll",
    )

    assert find_default_osm_configs(resources) == [
        tmp_path / "a_config.json",
        tmp_path / "z_config.json",
    ]


def test_prepare_runtime_environment_adds_dll_path_once(tmp_path: Path, monkeypatch) -> None:
    from cm_terrain_extractor_app.app_core import resources as resources_module
    from cm_terrain_extractor_app.app_core.resources import (
        AppResources,
        prepare_runtime_environment,
    )

    dll_dir = tmp_path / "dll"
    dll_dir.mkdir()
    added_dll_dirs = []

    class FakeDllHandle:
        pass

    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "existing")]))
    monkeypatch.setattr(resources_module.os, "name", "nt")
    monkeypatch.setattr(
        resources_module.os,
        "add_dll_directory",
        lambda path: added_dll_dirs.append(path) or FakeDllHandle(),
        raising=False,
    )
    resources = AppResources(
        app_root=tmp_path / "app",
        executable_root=tmp_path,
        data_cache_path=tmp_path / "data_cache",
        config_dir=tmp_path,
        dll_dir=dll_dir,
    )

    prepare_runtime_environment(resources)
    prepare_runtime_environment(resources)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries.count(str(dll_dir)) == 1
    assert path_entries[0] == str(dll_dir)
    assert added_dll_dirs == [str(dll_dir)]

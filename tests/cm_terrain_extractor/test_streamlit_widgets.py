from __future__ import annotations


def test_format_cache_size_label_summarizes_nested_cache_size(tmp_path) -> None:
    from cm_terrain_extractor_app.streamlit_ui.widgets import format_cache_size_label

    cache_dir = tmp_path / "cache"
    nested = cache_dir / "nested"
    nested.mkdir(parents=True)
    (cache_dir / "a.bin").write_bytes(b"a" * 512)
    (nested / "b.bin").write_bytes(b"b" * 1536)

    assert format_cache_size_label(cache_dir) == "2.0 KB"


def test_format_cache_size_label_handles_missing_cache_dir(tmp_path) -> None:
    from cm_terrain_extractor_app.streamlit_ui.widgets import format_cache_size_label

    assert format_cache_size_label(tmp_path / "missing") == "0.0 KB"

from __future__ import annotations

import pandas as pd

from cm_terrain_extractor_app.app_core.state import AppState


class FakeUpload:
    name = "buildings_ringmauer.geojson"

    def __init__(self, data: bytes) -> None:
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


def test_uploaded_osm_file_signature_uses_name_size_and_content() -> None:
    from cm_terrain_extractor_app.streamlit_ui.sidebar import _uploaded_osm_file_signature

    first = _uploaded_osm_file_signature(filename="a.geojson", data=b"same")

    assert first == _uploaded_osm_file_signature(filename="a.geojson", data=b"same")
    assert first != _uploaded_osm_file_signature(filename="b.geojson", data=b"same")
    assert first != _uploaded_osm_file_signature(filename="a.geojson", data=b"different")


def test_handle_uploaded_osm_file_ignores_unchanged_uploader_rerun(monkeypatch) -> None:
    from cm_terrain_extractor_app.streamlit_ui import sidebar

    load_calls = []
    bbox = object()

    def fake_load_osm_data_from_uploaded_bytes(*, data: bytes, filename: str | None = None) -> dict:
        load_calls.append((data, filename))
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(
        sidebar,
        "load_osm_data_from_uploaded_bytes",
        fake_load_osm_data_from_uploaded_bytes,
    )
    monkeypatch.setattr(sidebar, "_get_bounding_box", lambda osm_data, resources: bbox)

    state = AppState(
        bbox_object=object(),
        selected_area_valid=True,
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
    )
    upload = FakeUpload(b'{"type":"FeatureCollection","features":[]}')

    sidebar._handle_uploaded_osm_file(state, object(), upload)

    assert len(load_calls) == 1
    assert state.osm_output is None

    state.osm_output = pd.DataFrame({"category": ["road"]})
    state.osm_geometries = {"roads": []}

    sidebar._handle_uploaded_osm_file(state, object(), upload)

    assert len(load_calls) == 1
    assert state.osm_output is not None
    assert state.osm_geometries == {"roads": []}

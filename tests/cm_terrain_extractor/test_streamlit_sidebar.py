from __future__ import annotations

import pandas as pd

from cm_terrain_extractor_app.app_core.state import AppState


class FakeUpload:
    def __init__(self, data: bytes, name: str = "buildings_ringmauer.geojson") -> None:
        self.data = data
        self.name = name

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


def test_handle_uploaded_bbox_file_updates_selected_bbox_once(monkeypatch) -> None:
    from cm_terrain_extractor_app.streamlit_ui import sidebar

    bbox_df = pd.DataFrame(
        {
            "x": [7.0, 7.1, 7.1, 7.0],
            "y": [51.0, 51.0, 51.1, 51.1],
        }
    )
    bbox = object()
    update_calls = []
    rerun_calls = []

    def fake_bbox_from_bbox_dataframe(*, dataframe: pd.DataFrame, resources: object) -> object:
        pd.testing.assert_frame_equal(dataframe, bbox_df)
        assert resources == "resources"
        return bbox

    monkeypatch.setattr(sidebar, "_bbox_from_bbox_dataframe", fake_bbox_from_bbox_dataframe)
    monkeypatch.setattr(
        sidebar,
        "update_state_from_bbox",
        lambda state_arg, bbox_arg: update_calls.append((state_arg, bbox_arg)),
    )
    monkeypatch.setattr(sidebar.st, "rerun", lambda: rerun_calls.append(True))

    state = AppState()
    upload = FakeUpload(bbox_df.to_csv(index=False).encode("utf-8"), "bbox.csv")

    sidebar._handle_uploaded_bbox_file(state, "resources", upload)
    sidebar._handle_uploaded_bbox_file(state, "resources", upload)

    assert update_calls == [(state, bbox)]
    assert rerun_calls == [True]
    assert state.bbox_uploaded_file_signature is not None

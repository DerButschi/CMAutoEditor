def test_drawing_to_lon_lat_points_extracts_first_coordinate_ring() -> None:
    from cm_terrain_extractor_app.map_view.drawing import drawing_to_lon_lat_points

    assert drawing_to_lon_lat_points(_polygon_drawing()) == [
        (7.10, 51.20),
        (7.30, 51.20),
        (7.30, 51.35),
        (7.10, 51.35),
        (7.10, 51.20),
    ]


def test_extract_last_active_drawing_handles_missing_payload() -> None:
    from cm_terrain_extractor_app.map_view.drawing import extract_last_active_drawing

    drawing = _polygon_drawing()

    assert extract_last_active_drawing({"last_active_drawing": drawing}) is drawing
    assert extract_last_active_drawing({"last_active_drawing": None}) is None
    assert extract_last_active_drawing({}) is None


def test_drawing_to_bounding_box_preserves_lon_lat_polygon_order() -> None:
    from cm_terrain_extractor_app.map_view.drawing import drawing_to_bounding_box

    bbox = drawing_to_bounding_box(_polygon_drawing())

    assert list(bbox.polygon_wgs84.exterior.coords) == [
        (7.10, 51.20),
        (7.30, 51.20),
        (7.30, 51.35),
        (7.10, 51.35),
        (7.10, 51.20),
    ]


def _polygon_drawing() -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [7.10, 51.20],
                    [7.30, 51.20],
                    [7.30, 51.35],
                    [7.10, 51.35],
                    [7.10, 51.20],
                ],
                [
                    [7.15, 51.25],
                    [7.20, 51.25],
                    [7.20, 51.30],
                    [7.15, 51.30],
                    [7.15, 51.25],
                ],
            ],
        },
    }

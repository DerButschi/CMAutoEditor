def test_drawing_to_lon_lat_points_extracts_first_coordinate_ring() -> None:
    from cm_terrain_extractor_app.map_view.drawing import drawing_to_lon_lat_points

    drawing = {
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

    assert drawing_to_lon_lat_points(drawing) == [
        (7.10, 51.20),
        (7.30, 51.20),
        (7.30, 51.35),
        (7.10, 51.35),
        (7.10, 51.20),
    ]

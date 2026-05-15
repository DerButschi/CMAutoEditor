from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring
from shapely.strtree import STRtree
from terrain_extraction.osm_extraction.models import (
    CMType,
    FeatureRecord,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
)


@dataclass(frozen=True, slots=True)
class _SourceLine:
    feature: FeatureRecord
    geometry: LineString

    @property
    def metadata_key(self) -> tuple[str, str, int]:
        return self.feature.config_name, self.feature.process.value, self.feature.priority


@dataclass(slots=True)
class _PointCluster:
    point_ids: list[int]
    point: Point
    snapped: bool


class NetworkTopologyBuilder:
    def __init__(self, *, clip_geometry: BaseGeometry | None = None, snap_tolerance_m: float = 1.0) -> None:
        if snap_tolerance_m < 0:
            raise ValueError("snap_tolerance_m must be non-negative")
        self.clip_geometry = clip_geometry
        self.snap_tolerance_m = snap_tolerance_m

    def build(self, features: tuple[FeatureRecord, ...]) -> TopologyGraph:
        source_lines, clipped_lines = self._normalize_features(features)
        if not source_lines:
            return TopologyGraph(diagnostics=self._diagnostics(0, 0, clipped_lines, 0, 0, 0))

        split_points = self._collect_split_points(source_lines)
        clusters = self._cluster_points(split_points)
        nodes, point_to_node = self._build_nodes(clusters)
        edges = self._build_edges(source_lines, nodes, point_to_node, split_points)
        collapsed_edges, collapsed_nodes, collapsed_count = self._collapse_degree_two_chains(nodes, edges)

        diagnostics = self._diagnostics(
            source_line_count=len(source_lines),
            normalized_line_count=len(source_lines),
            clipped_line_count=clipped_lines,
            intersection_count=sum(
                1
                for point_id, _point in enumerate(split_points)
                if self._point_roles[point_id] == "intersection"
            ),
            snapped_point_count=sum(1 for cluster in clusters if cluster.snapped),
            collapsed_count=collapsed_count,
        )
        return TopologyGraph(nodes=collapsed_nodes, edges=collapsed_edges, diagnostics=diagnostics)

    def _normalize_features(self, features: tuple[FeatureRecord, ...]) -> tuple[list[_SourceLine], int]:
        source_lines: list[_SourceLine] = []
        clipped_count = 0
        for feature in features:
            for line in _extract_lines(feature.geometry):
                clipped = line
                if self.clip_geometry is not None:
                    if not line.intersects(self.clip_geometry):
                        continue
                    clipped = line.intersection(self.clip_geometry)
                    if not line.equals_exact(clipped, tolerance=0.001):
                        clipped_count += 1
                for clipped_line in _extract_lines(clipped):
                    if clipped_line.length > 0:
                        source_lines.append(_SourceLine(feature=feature, geometry=clipped_line))
        return source_lines, clipped_count

    def _collect_split_points(self, source_lines: list[_SourceLine]) -> list[Point]:
        split_points: list[Point] = []
        self._line_point_ids: dict[int, set[int]] = defaultdict(set)
        self._point_roles: dict[int, str] = {}
        for line_idx, source_line in enumerate(source_lines):
            coords = list(source_line.geometry.coords)
            for coord in (coords[0], coords[-1]):
                point_id = len(split_points)
                split_points.append(Point(coord))
                self._line_point_ids[line_idx].add(point_id)
                self._point_roles[point_id] = "endpoint"

        tree = STRtree([source_line.geometry for source_line in source_lines])
        for line_idx, source_line in enumerate(source_lines):
            for candidate_idx in _query_indices(tree, source_line.geometry, source_lines):
                if candidate_idx <= line_idx:
                    continue
                intersection = source_line.geometry.intersection(source_lines[candidate_idx].geometry)
                for point in _extract_points(intersection):
                    point_id = len(split_points)
                    split_points.append(point)
                    self._line_point_ids[line_idx].add(point_id)
                    self._line_point_ids[candidate_idx].add(point_id)
                    self._point_roles[point_id] = "intersection"
        return split_points

    def _cluster_points(self, points: list[Point]) -> list[_PointCluster]:
        parent = list(range(len(points)))

        def find(point_id: int) -> int:
            while parent[point_id] != point_id:
                parent[point_id] = parent[parent[point_id]]
                point_id = parent[point_id]
            return point_id

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        if self.snap_tolerance_m > 0 and points:
            tree = STRtree(points)
            for point_id, point in enumerate(points):
                search_area = point.buffer(self.snap_tolerance_m)
                for candidate_id in _query_point_indices(tree, search_area, points):
                    if candidate_id <= point_id:
                        continue
                    if point.distance(points[candidate_id]) <= self.snap_tolerance_m:
                        union(point_id, candidate_id)

        grouped: dict[int, list[int]] = defaultdict(list)
        for point_id in range(len(points)):
            grouped[find(point_id)].append(point_id)

        clusters = []
        for point_ids in grouped.values():
            x = sum(points[point_id].x for point_id in point_ids) / len(point_ids)
            y = sum(points[point_id].y for point_id in point_ids) / len(point_ids)
            first_point = points[point_ids[0]]
            snapped = any(not first_point.equals_exact(points[point_id], tolerance=0.001) for point_id in point_ids[1:])
            clusters.append(_PointCluster(point_ids=point_ids, point=Point(x, y), snapped=snapped))
        return clusters

    def _build_nodes(
        self,
        clusters: list[_PointCluster],
    ) -> tuple[tuple[TopologyNode, ...], dict[int, int]]:
        nodes = []
        point_to_node = {}
        for node_id, cluster in enumerate(sorted(clusters, key=lambda cluster: (cluster.point.x, cluster.point.y))):
            for point_id in cluster.point_ids:
                point_to_node[point_id] = node_id
            nodes.append(
                TopologyNode(
                    node_id=node_id,
                    point=cluster.point,
                    source_point_count=len(cluster.point_ids),
                    diagnostics={"snapped": cluster.snapped},
                )
            )
        return tuple(nodes), point_to_node

    def _build_edges(
        self,
        source_lines: list[_SourceLine],
        nodes: tuple[TopologyNode, ...],
        point_to_node: dict[int, int],
        split_points: list[Point],
    ) -> tuple[TopologyEdge, ...]:
        edges = []
        for line_idx, source_line in enumerate(source_lines):
            ordered_entries = self._ordered_line_node_entries(
                source_line.geometry,
                self._line_point_ids[line_idx],
                nodes,
                point_to_node,
                split_points,
            )
            for (start_distance, start_node_id), (end_distance, end_node_id) in zip(
                ordered_entries,
                ordered_entries[1:],
                strict=False,
            ):
                if start_node_id == end_node_id:
                    continue
                start = nodes[start_node_id].point
                end = nodes[end_node_id].point
                edges.append(
                    TopologyEdge(
                        edge_id=len(edges),
                        start_node_id=start_node_id,
                        end_node_id=end_node_id,
                        geometry=_line_substring_with_node_endpoints(
                            source_line.geometry,
                            start_distance,
                            end_distance,
                            start,
                            end,
                        ),
                        feature_ids=(source_line.feature.feature_id,),
                        source_indices=(source_line.feature.source_index,),
                        config_name=source_line.feature.config_name,
                        process=source_line.feature.process,
                        priority=source_line.feature.priority,
                        diagnostics={"source_length_m": source_line.geometry.length},
                        cm_type=source_line.feature.cm_type,
                    )
                )
        return tuple(edges)

    def _ordered_line_node_entries(
        self,
        line: LineString,
        point_ids: set[int],
        nodes: tuple[TopologyNode, ...],
        point_to_node: dict[int, int],
        split_points: list[Point],
    ) -> tuple[tuple[float, int], ...]:
        entries = []
        seen = set()
        for point_id in point_ids:
            node_id = point_to_node[point_id]
            if node_id in seen:
                continue
            seen.add(node_id)
            entries.append((line.project(split_points[point_id]), node_id))
        entries.sort(key=lambda entry: entry[0])

        if len(entries) >= 2:
            return tuple(entries)

        endpoint_entries = []
        for coord in (line.coords[0], line.coords[-1]):
            point = Point(coord)
            node_id = min(nodes, key=lambda node: node.point.distance(point)).node_id
            endpoint_entries.append((line.project(point), node_id))
        return tuple(endpoint_entries)

    def _collapse_degree_two_chains(
        self,
        nodes: tuple[TopologyNode, ...],
        edges: tuple[TopologyEdge, ...],
    ) -> tuple[tuple[TopologyEdge, ...], tuple[TopologyNode, ...], int]:
        adjacency: dict[int, list[TopologyEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.start_node_id].append(edge)
            adjacency[edge.end_node_id].append(edge)

        metadata_by_node = {
            node_id: {self._edge_metadata(edge) for edge in incident_edges}
            for node_id, incident_edges in adjacency.items()
        }
        collapsible_nodes = {
            node_id
            for node_id, incident_edges in adjacency.items()
            if len(incident_edges) == 2
            and len(metadata_by_node[node_id]) == 1
            and not nodes[node_id].diagnostics.get("snapped", False)
        }
        if not collapsible_nodes:
            return edges, nodes, 0

        visited_edges: set[int] = set()
        new_edges = []
        for edge in edges:
            if edge.edge_id in visited_edges:
                continue
            chain = self._walk_chain(edge, adjacency, collapsible_nodes, visited_edges)
            new_edges.append(self._merge_chain(len(new_edges), chain))

        used_node_ids = {edge.start_node_id for edge in new_edges} | {edge.end_node_id for edge in new_edges}
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(used_node_ids))}
        remapped_nodes = tuple(
            TopologyNode(
                node_id=node_id_mapping[node.node_id],
                point=node.point,
                source_point_count=node.source_point_count,
                diagnostics=node.diagnostics,
            )
            for node in nodes
            if node.node_id in node_id_mapping
        )
        remapped_edges = tuple(
            TopologyEdge(
                edge_id=edge.edge_id,
                start_node_id=node_id_mapping[edge.start_node_id],
                end_node_id=node_id_mapping[edge.end_node_id],
                geometry=edge.geometry,
                feature_ids=edge.feature_ids,
                source_indices=edge.source_indices,
                config_name=edge.config_name,
                process=edge.process,
                priority=edge.priority,
                diagnostics=edge.diagnostics,
                cm_type=edge.cm_type,
            )
            for edge in new_edges
        )
        return remapped_edges, remapped_nodes, len(collapsible_nodes)

    def _walk_chain(
        self,
        start_edge: TopologyEdge,
        adjacency: dict[int, list[TopologyEdge]],
        collapsible_nodes: set[int],
        visited_edges: set[int],
    ) -> list[TopologyEdge]:
        chain = [start_edge]
        visited_edges.add(start_edge.edge_id)
        for direction in ("start", "end"):
            current_node_id = start_edge.start_node_id if direction == "start" else start_edge.end_node_id
            while current_node_id in collapsible_nodes:
                next_edges = [edge for edge in adjacency[current_node_id] if edge.edge_id not in visited_edges]
                if len(next_edges) != 1 or self._edge_metadata(next_edges[0]) != self._edge_metadata(start_edge):
                    break
                next_edge = next_edges[0]
                visited_edges.add(next_edge.edge_id)
                if direction == "start":
                    chain.insert(0, next_edge)
                else:
                    chain.append(next_edge)
                current_node_id = _other_node_id(next_edge, current_node_id)
        return chain

    def _merge_chain(
        self,
        edge_id: int,
        chain: list[TopologyEdge],
    ) -> TopologyEdge:
        node_ids = [chain[0].start_node_id, chain[0].end_node_id]
        for edge in chain[1:]:
            if edge.start_node_id == node_ids[-1]:
                node_ids.append(edge.end_node_id)
            elif edge.end_node_id == node_ids[-1]:
                node_ids.append(edge.start_node_id)
            elif edge.start_node_id == node_ids[0]:
                node_ids.insert(0, edge.end_node_id)
            else:
                node_ids.insert(0, edge.start_node_id)

        coords = _chain_coords(chain, node_ids)
        feature_ids = tuple(dict.fromkeys(feature_id for edge in chain for feature_id in edge.feature_ids))
        source_indices = tuple(dict.fromkeys(source_index for edge in chain for source_index in edge.source_indices))
        first_edge = chain[0]
        return TopologyEdge(
            edge_id=edge_id,
            start_node_id=node_ids[0],
            end_node_id=node_ids[-1],
            geometry=LineString(coords),
            feature_ids=feature_ids,
            source_indices=source_indices,
            config_name=first_edge.config_name,
            process=first_edge.process,
            priority=first_edge.priority,
            diagnostics={"collapsed_edge_count": len(chain)},
            cm_type=first_edge.cm_type,
        )

    def _edge_metadata(self, edge: TopologyEdge) -> tuple[str, str, int, tuple[str, str, str | None, str | int | None]]:
        return edge.config_name, edge.process.value, edge.priority, _cm_type_key(edge.cm_type)

    def _diagnostics(
        self,
        source_line_count: int,
        normalized_line_count: int,
        clipped_line_count: int,
        intersection_count: int,
        snapped_point_count: int,
        collapsed_count: int,
    ) -> dict[str, Any]:
        return {
            "source_lines": source_line_count,
            "normalized_lines": normalized_line_count,
            "clipped_lines": clipped_line_count,
            "intersection_points": intersection_count,
            "snapped_points": snapped_point_count,
            "collapsed_degree_two_nodes": collapsed_count,
            "snap_tolerance_m": self.snap_tolerance_m,
        }


def _extract_lines(geometry: BaseGeometry) -> tuple[LineString, ...]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, LineString):
        return _line_segments_if_closed(geometry)
    if isinstance(geometry, MultiLineString):
        return tuple(line for part in geometry.geoms for line in _extract_lines(part))
    if isinstance(geometry, Polygon):
        boundaries = [geometry.exterior, *geometry.interiors]
        return tuple(
            LineString([start, end])
            for boundary in boundaries
            for start, end in zip(boundary.coords, boundary.coords[1:], strict=False)
        )
    if isinstance(geometry, GeometryCollection):
        return tuple(line for part in geometry.geoms for line in _extract_lines(part))
    return ()


def _line_substring_with_node_endpoints(
    line: LineString,
    start_distance: float,
    end_distance: float,
    start: Point,
    end: Point,
) -> LineString:
    segment = substring(line, start_distance, end_distance)
    if isinstance(segment, Point):
        return LineString([(start.x, start.y), (end.x, end.y)])
    coords = list(segment.coords)
    if len(coords) < 2:
        coords = [(start.x, start.y), (end.x, end.y)]
    else:
        coords[0] = (start.x, start.y)
        coords[-1] = (end.x, end.y)
    return LineString(coords)


def _chain_coords(chain: list[TopologyEdge], node_ids: list[int]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for edge_index, edge in enumerate(chain):
        expected_start = node_ids[edge_index]
        if edge.start_node_id == expected_start:
            edge_coords = list(edge.geometry.coords)
        else:
            edge_coords = list(reversed(edge.geometry.coords))
        if coords and coords[-1] == edge_coords[0]:
            coords.extend(edge_coords[1:])
        else:
            coords.extend(edge_coords)
    return coords


def _line_segments_if_closed(line: LineString) -> tuple[LineString, ...]:
    coords = list(line.coords)
    if len(coords) < 2:
        return ()
    if coords[0] != coords[-1]:
        return (line,)
    return tuple(LineString([start, end]) for start, end in zip(coords, coords[1:], strict=False) if start != end)


def _extract_points(geometry: BaseGeometry) -> tuple[Point, ...]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Point):
        return (geometry,)
    if isinstance(geometry, MultiPoint):
        return tuple(geometry.geoms)
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return (Point(coords[0]), Point(coords[-1]))
    if isinstance(geometry, GeometryCollection):
        return tuple(point for part in geometry.geoms for point in _extract_points(part))
    return ()


def _query_indices(tree: STRtree, geometry: BaseGeometry, source_lines: list[_SourceLine]) -> tuple[int, ...]:
    results = tree.query(geometry)
    if len(results) == 0:
        return ()
    first = results[0]
    if isinstance(first, Integral):
        return tuple(int(result) for result in results)
    indices = []
    for result in results:
        for idx, source_line in enumerate(source_lines):
            if result.equals_exact(source_line.geometry, tolerance=0.0):
                indices.append(idx)
                break
    return tuple(indices)


def _query_point_indices(tree: STRtree, geometry: BaseGeometry, points: list[Point]) -> tuple[int, ...]:
    results = tree.query(geometry)
    if len(results) == 0:
        return ()
    first = results[0]
    if isinstance(first, Integral):
        return tuple(int(result) for result in results)
    indices = []
    for result in results:
        for idx, point in enumerate(points):
            if result.equals_exact(point, tolerance=0.0):
                indices.append(idx)
                break
    return tuple(indices)


def _other_node_id(edge: TopologyEdge, node_id: int) -> int:
    if edge.start_node_id == node_id:
        return edge.end_node_id
    return edge.start_node_id


def _cm_type_key(cm_type: CMType | None) -> tuple[str, str, str | None, str | int | None]:
    if cm_type is None:
        return ("", "", None, None)
    return cm_type.menu, cm_type.cat1, cm_type.cat2, cm_type.direction

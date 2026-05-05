from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.config_schema import ExtractionConfig, extract_tags
from terrain_extraction.osm_extraction.models import FeatureRecord


@dataclass(frozen=True, slots=True)
class FeatureMatcher:
    config: ExtractionConfig

    def match_features(self, features: Iterable[Any]) -> tuple[FeatureRecord, ...]:
        records: list[FeatureRecord] = []
        for source_index, feature in enumerate(features):
            properties = _feature_properties(feature)
            tags = extract_tags(properties)
            feature_id = _feature_id(feature, properties)
            matching_entries = self.config.matching_entries(tags, feature_id)
            if not matching_entries:
                continue

            geometry = _feature_geometry(feature)
            if geometry is None:
                continue

            for entry in matching_entries:
                for process in entry.processes:
                    records.append(
                        FeatureRecord(
                            feature_id=feature_id,
                            source_index=source_index,
                            config_name=entry.name,
                            process=process,
                            priority=entry.priority,
                            geometry=geometry,
                            source_tags=tags,
                            source_properties=properties,
                        )
                    )

        return tuple(records)


def _feature_properties(feature: Any) -> Mapping[str, Any]:
    if isinstance(feature, Mapping):
        properties = feature.get("properties", {})
    else:
        properties = getattr(feature, "properties", {})
    if properties is None:
        return MappingProxyType({})
    return MappingProxyType(dict(properties))


def _feature_id(feature: Any, properties: Mapping[str, Any]) -> Any:
    if "id" in properties:
        return properties["id"]
    if isinstance(feature, Mapping):
        return feature.get("id")
    return getattr(feature, "id", None)


def _feature_geometry(feature: Any) -> BaseGeometry | None:
    if isinstance(feature, Mapping):
        geometry = feature.get("geometry")
    else:
        geometry = getattr(feature, "geometry", None)
    if isinstance(geometry, BaseGeometry):
        return geometry
    if geometry is None:
        return None
    try:
        return shape(geometry)
    except Exception:
        return None

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from terrain_extraction.osm_extraction.models import CMType, ProcessKind


class ConfigValidationError(ValueError):
    """Raised when an OSM extraction config cannot be compiled safely."""


RoadValidationMode = Literal["strict", "warn"]

_DEFAULT_MAX_CUTSET_CYCLE_RANK = 2
_DEFAULT_MAX_CUTSET_VERTICES = 4
_DEFAULT_MAX_CUTSET_CANDIDATE_PRODUCT_LOG10 = 5.0
_DEFAULT_TINY_EXACT_MAX_CELLS = 12
_DEFAULT_TINY_EXACT_CANDIDATE_PRODUCT_LOG10 = 5.0


@dataclass(frozen=True, slots=True)
class TileAssignmentSolverConfig:
    max_cutset_cycle_rank: int = _DEFAULT_MAX_CUTSET_CYCLE_RANK
    max_cutset_vertices: int = _DEFAULT_MAX_CUTSET_VERTICES
    max_cutset_candidate_product_log10: float = _DEFAULT_MAX_CUTSET_CANDIDATE_PRODUCT_LOG10
    tiny_exact_max_cells: int = _DEFAULT_TINY_EXACT_MAX_CELLS
    tiny_exact_candidate_product_log10: float = _DEFAULT_TINY_EXACT_CANDIDATE_PRODUCT_LOG10


@dataclass(frozen=True, slots=True)
class CMTypeMatch:
    cm_type: CMType | None
    cm_type_index: int | None
    first_matching_tag_index: int | None
    matched_by_tags: bool


@dataclass(frozen=True, slots=True)
class TagSelector:
    pairs: tuple[tuple[str, Any], ...]
    require_all: bool = False

    @classmethod
    def from_raw(cls, raw: object, *, field_name: str, require_all: bool = False) -> TagSelector:
        return cls(_normalize_tag_pairs(raw, field_name), require_all=require_all)

    def matches(self, tags: Mapping[str, Any]) -> bool:
        if not self.pairs:
            return True
        pair_matches = (tags.get(key) == value for key, value in self.pairs)
        if self.require_all:
            return all(pair_matches)
        return any(pair_matches)


@dataclass(frozen=True, slots=True)
class ConfigEntry:
    name: str
    active: bool
    priority: int
    tag_selector: TagSelector
    required_tags: TagSelector
    excluded_tags: TagSelector
    allowed_ids: frozenset[Any] | None
    excluded_ids: frozenset[Any]
    processes: tuple[ProcessKind, ...]
    legacy_processes: tuple[str, ...]
    cm_types: tuple[CMType, ...]
    raw_cm_types: tuple[Mapping[str, Any], ...]
    modifiers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifiers", MappingProxyType(dict(self.modifiers)))
        object.__setattr__(
            self,
            "raw_cm_types",
            tuple(MappingProxyType(dict(cm_type)) for cm_type in self.raw_cm_types),
        )

    def matches_tags(self, tags: Mapping[str, Any], feature_id: Any = None) -> bool:
        if not self.active:
            return False
        if not self.tag_selector.matches(tags):
            return False
        if not self.required_tags.matches(tags):
            return False
        if self.excluded_tags.pairs and self.excluded_tags.matches(tags):
            return False
        if feature_id in self.excluded_ids:
            return False
        return not (self.allowed_ids is not None and feature_id not in self.allowed_ids)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    entries: tuple[ConfigEntry, ...]
    seed: int | None = None
    road_validation_mode: RoadValidationMode = "warn"
    tile_assignment_solver: TileAssignmentSolverConfig = field(default_factory=TileAssignmentSolverConfig)
    feature_flags: Mapping[str, bool] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigValidationError(f"Duplicate OSM config entries: {duplicates}")
        object.__setattr__(self, "feature_flags", MappingProxyType(dict(self.feature_flags)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @classmethod
    def from_path(cls, path: str | Path, *, seed: int | None = None) -> ExtractionConfig:
        with open(path, encoding="utf-8") as config_file:
            return cls.from_mapping(json.load(config_file), seed=seed)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any], *, seed: int | None = None) -> ExtractionConfig:
        entries = [
            _compile_entry(name, raw_entry)
            for name, raw_entry in raw_config.items()
            if _is_config_entry(raw_entry)
        ]
        return cls(
            entries=tuple(entries),
            seed=seed,
            road_validation_mode=_road_validation_mode(raw_config.get("road_validation_mode", "warn")),
            tile_assignment_solver=_tile_assignment_solver_config(raw_config.get("tile_assignment_solver", {})),
        )

    def entry_by_name(self, name: str) -> ConfigEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def matching_entries(self, tags: Mapping[str, Any], feature_id: Any = None) -> tuple[ConfigEntry, ...]:
        return tuple(entry for entry in self.entries if entry.matches_tags(tags, feature_id))


def extract_tags(properties: Mapping[str, Any]) -> Mapping[str, Any]:
    tags = properties.get("tags")
    if isinstance(tags, Mapping):
        return MappingProxyType(dict(tags))
    return MappingProxyType(dict(properties))


def match_cm_type(cm_types: Iterable[Mapping[str, Any]], tags: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for cm_type in cm_types:
        selector = TagSelector.from_raw(cm_type.get("tags", ()), field_name="cm_types.tags")
        if selector.matches(tags):
            return cm_type
    return None


def matched_or_first_cm_type(entry: ConfigEntry, tags: Mapping[str, Any]) -> CMType | None:
    return matched_or_first_cm_type_match(entry, tags).cm_type


def matched_or_first_cm_type_match(entry: ConfigEntry, tags: Mapping[str, Any]) -> CMTypeMatch:
    for index, raw_cm_type in enumerate(entry.raw_cm_types):
        matches, tag_index = _cm_type_tags_match(raw_cm_type.get("tags", ()), tags)
        if matches:
            cm_type = entry.cm_types[index]
            return CMTypeMatch(
                cm_type=None if cm_type.modifiers.get("dummy") is True else cm_type,
                cm_type_index=index,
                first_matching_tag_index=tag_index,
                matched_by_tags=True,
            )
    if not entry.cm_types:
        return CMTypeMatch(None, None, None, False)
    cm_type = entry.cm_types[0]
    return CMTypeMatch(
        cm_type=None if cm_type.modifiers.get("dummy") is True else cm_type,
        cm_type_index=0,
        first_matching_tag_index=None,
        matched_by_tags=False,
    )


def _cm_type_tags_match(raw_tags: object, tags: Mapping[str, Any]) -> tuple[bool, int | None]:
    pairs = _normalize_tag_pairs(raw_tags, "cm_types.tags")
    if not pairs:
        return True, None
    for index, (key, value) in enumerate(pairs):
        if tags.get(key) == value:
            return True, index
    return False, None


def _compile_entry(name: str, raw_entry: Mapping[str, Any]) -> ConfigEntry:
    raw_processes = _normalize_string_list(raw_entry.get("process"), name, "process")
    processes = tuple(_process_kind(process, name) for process in raw_processes)
    raw_cm_types = _normalize_mapping_list(raw_entry.get("cm_types", ()), name, "cm_types")
    modifiers = raw_entry.get("modifiers", {})
    if not isinstance(modifiers, Mapping):
        raise ConfigValidationError(f"{name}.modifiers must be an object")

    return ConfigEntry(
        name=name,
        active=bool(raw_entry.get("active", True)),
        priority=int(raw_entry.get("priority", 0)),
        tag_selector=TagSelector.from_raw(raw_entry.get("tags", ()), field_name=f"{name}.tags"),
        required_tags=TagSelector.from_raw(
            raw_entry.get("required_tags", ()),
            field_name=f"{name}.required_tags",
            require_all=True,
        ),
        excluded_tags=TagSelector.from_raw(
            raw_entry.get("exclude_tags", ()),
            field_name=f"{name}.exclude_tags",
        ),
        allowed_ids=_optional_id_set(raw_entry.get("allowed_ids")),
        excluded_ids=frozenset(raw_entry.get("exclude_ids", ())),
        processes=processes,
        legacy_processes=tuple(raw_processes),
        cm_types=tuple(_compile_cm_type(raw_cm_type, name) for raw_cm_type in raw_cm_types),
        raw_cm_types=tuple(raw_cm_types),
        modifiers=modifiers,
    )


def _compile_cm_type(raw_cm_type: Mapping[str, Any], entry_name: str) -> CMType:
    if raw_cm_type.get("dummy") is True:
        menu = raw_cm_type.get("menu", "-1")
        cat1 = raw_cm_type.get("cat1", "-1")
    else:
        try:
            menu = raw_cm_type["menu"]
            cat1 = raw_cm_type["cat1"]
        except KeyError as exc:
            raise ConfigValidationError(
                f"{entry_name}.cm_types entries require menu and cat1"
            ) from exc

    return CMType(
        menu=str(menu),
        cat1=str(cat1),
        cat2=None if "cat2" not in raw_cm_type else str(raw_cm_type["cat2"]),
        direction=raw_cm_type.get("direction"),
        tile_id=raw_cm_type.get("id"),
        modifiers={
            key: value
            for key, value in raw_cm_type.items()
            if key not in {"menu", "cat1", "cat2", "direction", "id", "tags"}
        },
    )


def _process_kind(process: str, entry_name: str) -> ProcessKind:
    try:
        return ProcessKind.from_legacy(process)
    except ValueError as exc:
        raise ConfigValidationError(f"{entry_name}.process contains unknown process {process!r}") from exc


def _normalize_tag_pairs(raw: object, field_name: str) -> tuple[tuple[str, Any], ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ConfigValidationError(f"{field_name} must be a list of [key, value] pairs")

    pairs = []
    for pair in raw:
        if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2:
            raise ConfigValidationError(f"{field_name} must contain [key, value] pairs")
        pairs.append((str(pair[0]), pair[1]))
    return tuple(pairs)


def _normalize_string_list(raw: object, entry_name: str, field_name: str) -> tuple[str, ...]:
    if raw is None:
        raise ConfigValidationError(f"{entry_name}.{field_name} is required")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ConfigValidationError(f"{entry_name}.{field_name} must be a list")
    return tuple(str(item) for item in raw)


def _normalize_mapping_list(
    raw: object,
    entry_name: str,
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ConfigValidationError(f"{entry_name}.{field_name} must be a list")
    entries = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ConfigValidationError(f"{entry_name}.{field_name} entries must be objects")
        entries.append(item)
    return tuple(entries)


def _optional_id_set(raw: object) -> frozenset[Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ConfigValidationError("allowed_ids must be a list")
    return frozenset(raw)


def _road_validation_mode(raw: object) -> RoadValidationMode:
    if isinstance(raw, str) and raw in {"strict", "warn"}:
        return raw
    raise ConfigValidationError("road_validation_mode must be 'strict' or 'warn'")


def _tile_assignment_solver_config(raw: object) -> TileAssignmentSolverConfig:
    if raw in (None, {}):
        return TileAssignmentSolverConfig()
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("tile_assignment_solver must be an object")
    return TileAssignmentSolverConfig(
        max_cutset_cycle_rank=_nonnegative_int(
            raw.get("max_cutset_cycle_rank", _DEFAULT_MAX_CUTSET_CYCLE_RANK),
            "tile_assignment_solver.max_cutset_cycle_rank",
        ),
        max_cutset_vertices=_nonnegative_int(
            raw.get("max_cutset_vertices", _DEFAULT_MAX_CUTSET_VERTICES),
            "tile_assignment_solver.max_cutset_vertices",
        ),
        max_cutset_candidate_product_log10=_nonnegative_float(
            raw.get(
                "max_cutset_candidate_product_log10",
                _DEFAULT_MAX_CUTSET_CANDIDATE_PRODUCT_LOG10,
            ),
            "tile_assignment_solver.max_cutset_candidate_product_log10",
        ),
        tiny_exact_max_cells=_nonnegative_int(
            raw.get("tiny_exact_max_cells", _DEFAULT_TINY_EXACT_MAX_CELLS),
            "tile_assignment_solver.tiny_exact_max_cells",
        ),
        tiny_exact_candidate_product_log10=_nonnegative_float(
            raw.get(
                "tiny_exact_candidate_product_log10",
                _DEFAULT_TINY_EXACT_CANDIDATE_PRODUCT_LOG10,
            ),
            "tile_assignment_solver.tiny_exact_candidate_product_log10",
        ),
    )


def _nonnegative_int(raw: object, field_name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} must be a non-negative integer") from exc
    if value < 0:
        raise ConfigValidationError(f"{field_name} must be a non-negative integer")
    return value


def _nonnegative_float(raw: object, field_name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} must be a non-negative number") from exc
    if value < 0:
        raise ConfigValidationError(f"{field_name} must be a non-negative number")
    return value


def _is_config_entry(raw_entry: object) -> bool:
    return isinstance(raw_entry, Mapping) and "process" in raw_entry

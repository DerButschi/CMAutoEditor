from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    timings: Mapping[str, float | None] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "quality", MappingProxyType(dict(self.quality)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timings": dict(self.timings),
            "counts": dict(self.counts),
            "quality": dict(self.quality),
            "diagnostics": dict(self.diagnostics),
        }

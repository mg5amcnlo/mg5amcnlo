# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import cast, get_type_hints

from .errors import ConfigurationError
from .models import RunConfig


@dataclass(frozen=True, slots=True)
class ConfigField:
    """One schema-v1 leaf shared by TOML and command-line resolution."""

    path: str
    kind: str
    nullable: bool = False
    choices: tuple[object, ...] = ()
    dynamic_kind: str | None = None
    required: bool = False
    default: object | None = None


def _immutable_default(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_default(entry) for key, entry in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_default(entry) for entry in value)
    return value


def _build_registry(
    cls: type[object], prefix: str = ""
) -> tuple[dict[str, ConfigField], set[str]]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    registry: dict[str, ConfigField] = {}
    sections: set[str] = set()
    hints = get_type_hints(cls)
    for item in fields(cls):
        path = f"{prefix}.{item.name}" if prefix else item.name
        if item.metadata.get("config_section"):
            sections.add(path)
            child_registry, child_sections = _build_registry(hints[item.name], path)
            registry.update(child_registry)
            sections.update(child_sections)
            continue

        raw_spec = item.metadata.get("config")
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"configuration field {path!r} has no registry metadata")

        required = item.default is MISSING and item.default_factory is MISSING
        if required:
            default: object | None = None
        elif item.default is not MISSING:
            default = item.default
        else:
            default_factory = cast(Callable[[], object], item.default_factory)
            default = default_factory()
        registry[path] = ConfigField(
            path=path,
            kind=str(raw_spec["kind"]),
            nullable=bool(raw_spec.get("nullable", False)),
            choices=tuple(raw_spec.get("choices", ())),
            dynamic_kind=(
                str(raw_spec["dynamic_kind"])
                if raw_spec.get("dynamic_kind") is not None
                else None
            ),
            required=required,
            default=_immutable_default(default),
        )
    return registry, sections


_REGISTRY, _SECTIONS = _build_registry(RunConfig)
FIELD_REGISTRY: Mapping[str, ConfigField] = MappingProxyType(_REGISTRY)
CONFIG_SECTIONS: tuple[str, ...] = tuple(sorted(_SECTIONS))


def get_config_field(path: str) -> ConfigField:
    """Return the exact or schema-approved dynamic field for a dotted path."""

    if path in FIELD_REGISTRY:
        return FIELD_REGISTRY[path]
    for parent_path, item in FIELD_REGISTRY.items():
        prefix = f"{parent_path}."
        if item.dynamic_kind is not None and path.startswith(prefix):
            key = path[len(prefix) :]
            if key and "." not in key:
                return ConfigField(
                    path=path,
                    kind=item.dynamic_kind,
                    nullable=False,
                    required=False,
                )
    raise ConfigurationError(f"unknown configuration field {path!r}")


def default_values() -> dict[str, object]:
    """Return a fresh flat mapping of every schema default."""

    return {
        path: _copy_default(item.default)
        for path, item in FIELD_REGISTRY.items()
        if not item.required
    }


def _copy_default(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _copy_default(entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_default(entry) for entry in value)
    return value


__all__ = [
    "CONFIG_SECTIONS",
    "FIELD_REGISTRY",
    "ConfigField",
    "default_values",
    "get_config_field",
]

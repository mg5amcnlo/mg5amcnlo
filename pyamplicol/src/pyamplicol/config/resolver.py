# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import difflib
import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, TypeVar, cast, get_type_hints

from .errors import ConfigurationError
from .models import Action, ProcessEntry, RunConfig
from .registry import CONFIG_SECTIONS, FIELD_REGISTRY, ConfigField, default_values

_DataclassT = TypeVar("_DataclassT")


def _extend_errors(errors: list[str], error: ConfigurationError) -> None:
    errors.extend(error.messages)


def _raise_errors(errors: Sequence[str]) -> None:
    if errors:
        raise ConfigurationError(errors)


def _freeze_public(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_public(entry) for key, entry in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public(entry) for entry in value)
    return value


@dataclass(frozen=True, slots=True)
class ConfigOverride:
    path: str
    value: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_public(self.value))


@dataclass(frozen=True, slots=True)
class ClampRequest:
    path: str
    effective: object
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ConfigurationError("a clamp reason must not be empty")
        object.__setattr__(self, "effective", _freeze_public(self.effective))


@dataclass(frozen=True, slots=True)
class ConfigClamp:
    path: str
    requested: object
    effective: object
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", _freeze_public(self.requested))
        object.__setattr__(self, "effective", _freeze_public(self.effective))


@dataclass(frozen=True, slots=True)
class ConfigResolution:
    requested: RunConfig
    effective: RunConfig
    clamps: tuple[ConfigClamp, ...] = ()

    @property
    def was_clamped(self) -> bool:
        return bool(self.clamps)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clamps", tuple(self.clamps))


def _field_for_path(path: str) -> tuple[ConfigField, str | None]:
    exact = FIELD_REGISTRY.get(path)
    if exact is not None:
        return exact, None
    for parent, item in FIELD_REGISTRY.items():
        prefix = f"{parent}."
        if item.dynamic_kind is not None and path.startswith(prefix):
            dynamic_key = path[len(prefix) :]
            if dynamic_key and "." not in dynamic_key:
                return (
                    ConfigField(path=path, kind=item.dynamic_kind),
                    dynamic_key,
                )
    _unknown_field(path)


def _unknown_field(path: str) -> NoReturn:
    raise ConfigurationError(_unknown_field_message(path))


def _unknown_field_message(path: str) -> str:
    candidates = list(FIELD_REGISTRY)
    suggestion = difflib.get_close_matches(path, candidates, n=1, cutoff=0.55)
    suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
    return f"unknown configuration field {path!r}{suffix}"


def _is_section(path: str) -> bool:
    return path in CONFIG_SECTIONS


def _flatten_mapping(
    value: Mapping[str, object],
    *,
    prefix: str = "",
    _errors: list[str] | None = None,
) -> dict[str, object]:
    owns_errors = _errors is None
    errors = [] if _errors is None else _errors
    result: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            errors.append("configuration keys must be non-empty strings")
            continue
        path = f"{prefix}.{key}" if prefix else key
        if path in FIELD_REGISTRY:
            result[path] = entry
            continue
        if _is_section(path):
            if not isinstance(entry, Mapping):
                errors.append(f"configuration section {path!r} must be a table")
                continue
            result.update(_flatten_mapping(entry, prefix=path, _errors=errors))
            continue
        # Dotted dedicated inputs may address one dynamic map element directly.
        try:
            _spec, dynamic_key = _field_for_path(path)
        except ConfigurationError as exc:
            _extend_errors(errors, exc)
            continue
        if dynamic_key is None:
            errors.append(_unknown_field_message(path))
            continue
        result[path] = entry
    if owns_errors:
        _raise_errors(errors)
    return result


def _normalize_dedicated(
    value: Mapping[str, object], *, _errors: list[str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    nested: dict[str, object] = {}
    for key, entry in value.items():
        if "." in key:
            result[key] = entry
        else:
            nested[key] = entry
    if nested:
        result.update(_flatten_mapping(nested, _errors=_errors))
    return result


def _resolve_path(value: object, base_dir: Path, path: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigurationError(f"{path} must be a path")
    candidate = Path(os.fspath(value)).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def _coerce_process_entries(value: object, path: str) -> tuple[ProcessEntry, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{path} must be a list of process entry tables")

    entries: list[ProcessEntry] = []
    errors: list[str] = []
    allowed_fields = ("expression", "name")
    for index, raw_entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        if isinstance(raw_entry, ProcessEntry):
            entries.append(raw_entry)
            continue
        if not isinstance(raw_entry, Mapping):
            errors.append(f"{entry_path} must be a table")
            continue

        for key in raw_entry:
            if not isinstance(key, str) or not key:
                errors.append(f"{entry_path} keys must be non-empty strings")
                continue
            if key not in allowed_fields:
                suggestion = difflib.get_close_matches(
                    key, allowed_fields, n=1, cutoff=0.55
                )
                suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
                errors.append(
                    f"unknown configuration field {entry_path}.{key!s}{suffix}"
                )

        try:
            entries.append(
                ProcessEntry(
                    expression=cast(str, raw_entry.get("expression")),
                    name=cast(str | None, raw_entry.get("name")),
                )
            )
        except ConfigurationError as exc:
            errors.extend(f"{entry_path}: {message}" for message in exc.messages)

    _raise_errors(errors)
    return tuple(entries)


def _coerce(
    item: ConfigField,
    value: object,
    *,
    base_dir: Path,
) -> object:
    if value is None:
        if item.nullable:
            return None
        raise ConfigurationError(f"{item.path} may not be null")

    kind = item.kind
    result: object
    if kind == "str":
        if not isinstance(value, str):
            raise ConfigurationError(f"{item.path} must be a string")
        if item.path == "model.source":
            from .._internal.sm_heft import is_sm_heft_alias

            if value in {"built-in-sm", "builtin_sm"} or is_sm_heft_alias(value):
                result = value
            else:
                result = str(_resolve_path(value, base_dir, item.path))
        else:
            result = value
    elif kind == "path":
        result = _resolve_path(value, base_dir, item.path)
    elif kind == "bool":
        if not isinstance(value, bool):
            raise ConfigurationError(f"{item.path} must be true or false")
        result = value
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{item.path} must be an integer")
        result = value
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{item.path} must be a number")
        result = float(value)
    elif kind == "auto_int":
        if value == "auto":
            result = value
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{item.path} must be 'auto' or an integer")
        else:
            result = value
    elif kind == "auto_bool":
        if value == "auto" or isinstance(value, bool):
            result = value
        else:
            raise ConfigurationError(f"{item.path} must be 'auto', true, or false")
    elif kind == "list_str":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigurationError(f"{item.path} must be a list of strings")
        if not all(isinstance(entry, str) for entry in value):
            raise ConfigurationError(f"{item.path} must be a list of strings")
        result = tuple(cast(Sequence[str], value))
    elif kind == "list_int":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigurationError(f"{item.path} must be a list of integers")
        if any(
            isinstance(entry, bool) or not isinstance(entry, int) for entry in value
        ):
            raise ConfigurationError(f"{item.path} must be a list of integers")
        result = tuple(cast(Sequence[int], value))
    elif kind == "process_entries":
        result = _coerce_process_entries(value, item.path)
    elif kind in ("map_list_str", "map_int"):
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{item.path} must be a table")
        dynamic_kind = "list_str" if kind == "map_list_str" else "int"
        converted: dict[str, object] = {}
        for key, entry in value.items():
            if not isinstance(key, str) or not key:
                raise ConfigurationError(f"{item.path} keys must be strings")
            converted[key] = _coerce(
                ConfigField(path=f"{item.path}.{key}", kind=dynamic_kind),
                entry,
                base_dir=base_dir,
            )
        result = converted
    else:
        raise AssertionError(f"unsupported configuration kind {kind!r}")

    if item.choices and result not in item.choices:
        allowed = ", ".join(repr(choice) for choice in item.choices)
        raise ConfigurationError(
            f"{item.path} must be one of {allowed}; got {result!r}"
        )
    return result


def parse_override(
    expression: str,
    *,
    base_dir: os.PathLike[str] | str | None = None,
) -> ConfigOverride:
    """Parse one schema-aware ``PATH=VALUE`` command-line override."""

    if "=" not in expression:
        raise ConfigurationError(
            f"override {expression!r} must use dotted.path=value syntax"
        )
    raw_path, raw_value = expression.split("=", 1)
    path = raw_path.strip()
    if not path:
        raise ConfigurationError("override path must not be empty")
    item, _dynamic_key = _field_for_path(path)
    value_text = raw_value.strip()
    if not value_text:
        raise ConfigurationError(f"override {path!r} has no value")

    # A choice spelled "none" (process.analyze_process_groups) is not null.
    if value_text.lower() in ("null", "none") and value_text not in item.choices:
        parsed: object = None
    else:
        try:
            parsed = tomllib.loads(f"value = {value_text}\n")["value"]
        except tomllib.TOMLDecodeError as exc:
            if item.kind not in ("str", "path", "auto_int", "auto_bool"):
                raise ConfigurationError(
                    f"invalid value for override {path!r}: {value_text!r}"
                ) from exc
            parsed = value_text

    root = Path.cwd() if base_dir is None else Path(base_dir)
    root = root.expanduser().resolve(strict=False)
    return ConfigOverride(path, _coerce(item, parsed, base_dir=root))


def _apply_value(values: dict[str, object], path: str, value: object) -> None:
    _item, dynamic_key = _field_for_path(path)
    if dynamic_key is None:
        values[path] = value
        return
    parent_path = path.rsplit(".", 1)[0]
    current = values.get(parent_path, {})
    if not isinstance(current, Mapping):
        raise ConfigurationError(f"{parent_path} must be a table")
    updated = dict(current)
    updated[dynamic_key] = value
    values[parent_path] = updated


def _coerce_flat(
    raw: Mapping[str, object],
    *,
    base_dir: Path,
    _errors: list[str] | None = None,
) -> dict[str, object]:
    owns_errors = _errors is None
    errors = [] if _errors is None else _errors
    result: dict[str, object] = {}
    for path, value in raw.items():
        try:
            item, _dynamic_key = _field_for_path(path)
            result[path] = _coerce(item, value, base_dir=base_dir)
        except ConfigurationError as exc:
            _extend_errors(errors, exc)
    if owns_errors:
        _raise_errors(errors)
    return result


def _build_dataclass(
    cls: type[_DataclassT], prefix: str, values: Mapping[str, object]
) -> _DataclassT:
    kwargs: dict[str, object] = {}
    errors: list[str] = []
    hints = get_type_hints(cls)
    for item in fields(cls):  # type: ignore[arg-type]
        path = f"{prefix}.{item.name}" if prefix else item.name
        if item.metadata.get("config_section"):
            try:
                kwargs[item.name] = _build_dataclass(hints[item.name], path, values)
            except ConfigurationError as exc:
                _extend_errors(errors, exc)
        else:
            if path not in values:
                errors.append(f"missing required configuration field {path!r}")
            else:
                kwargs[item.name] = values[path]
    _raise_errors(errors)
    return cls(**kwargs)


def _make_run_config(values: Mapping[str, object]) -> RunConfig:
    return _build_dataclass(RunConfig, "", values)


def _read_card(path: Path) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            parsed = tomllib.load(stream)
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read configuration card {path}: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {path}: {exc}") from exc
    return cast(Mapping[str, object], parsed)


def resolve_config(
    card: os.PathLike[str] | str | Mapping[str, object] | None = None,
    *,
    action: Action | None = None,
    dedicated: Mapping[str, object] | None = None,
    overrides: Iterable[str | ConfigOverride] = (),
    clamps: Iterable[ClampRequest] = (),
    base_dir: os.PathLike[str] | str | None = None,
) -> ConfigResolution:
    """Resolve schema-v1 configuration in its normative precedence order."""

    if isinstance(card, Mapping):
        raw_card = card
        root = Path.cwd() if base_dir is None else Path(base_dir)
        root = root.expanduser().resolve(strict=False)
    elif card is not None:
        card_path = Path(os.fspath(card)).expanduser().resolve(strict=False)
        raw_card = _read_card(card_path)
        root = card_path.parent
    else:
        raw_card = {}
        root = Path.cwd() if base_dir is None else Path(base_dir)
        root = root.expanduser().resolve(strict=False)

    values = default_values()
    card_errors: list[str] = []
    card_fields = _flatten_mapping(raw_card, _errors=card_errors)
    card_flat = _coerce_flat(card_fields, base_dir=root, _errors=card_errors)
    _raise_errors(card_errors)
    for path, value in card_flat.items():
        _apply_value(values, path, value)

    if action is not None:
        action_item = FIELD_REGISTRY["action"]
        _apply_value(values, "action", _coerce(action_item, action, base_dir=root))

    if dedicated:
        dedicated_errors: list[str] = []
        dedicated_fields = _normalize_dedicated(dedicated, _errors=dedicated_errors)
        dedicated_flat = _coerce_flat(
            dedicated_fields, base_dir=root, _errors=dedicated_errors
        )
        _raise_errors(dedicated_errors)
        for path, value in dedicated_flat.items():
            _apply_value(values, path, value)

    override_errors: list[str] = []
    for raw_override in overrides:
        try:
            parsed_override = (
                raw_override
                if isinstance(raw_override, ConfigOverride)
                else parse_override(raw_override, base_dir=root)
            )
            item, _dynamic_key = _field_for_path(parsed_override.path)
            normalized = _coerce(item, parsed_override.value, base_dir=root)
            _apply_value(values, parsed_override.path, normalized)
        except ConfigurationError as exc:
            _extend_errors(override_errors, exc)
    _raise_errors(override_errors)

    requested = _make_run_config(values)

    effective_values = dict(values)
    records: list[ConfigClamp] = []
    clamp_errors: list[str] = []
    for clamp in clamps:
        try:
            item, dynamic_key = _field_for_path(clamp.path)
            effective_value = _coerce(item, clamp.effective, base_dir=root)
            if dynamic_key is None:
                requested_value = effective_values[clamp.path]
            else:
                parent = effective_values[clamp.path.rsplit(".", 1)[0]]
                if not isinstance(parent, Mapping) or dynamic_key not in parent:
                    raise ConfigurationError(
                        f"cannot clamp missing configuration field {clamp.path!r}"
                    )
                requested_value = parent[dynamic_key]
            _apply_value(effective_values, clamp.path, effective_value)
            records.append(
                ConfigClamp(
                    path=clamp.path,
                    requested=requested_value,
                    effective=effective_value,
                    reason=clamp.reason,
                )
            )
        except ConfigurationError as exc:
            _extend_errors(clamp_errors, exc)
    _raise_errors(clamp_errors)

    effective = _make_run_config(effective_values)
    effective = replace(effective, color=effective.resolved_color)
    if effective.evaluator.jit.compress == "auto":
        effective = replace(
            effective,
            evaluator=replace(
                effective.evaluator,
                jit=replace(
                    effective.evaluator.jit,
                    compress=effective.evaluator.resolved_jit_compress,
                ),
            ),
        )
    return ConfigResolution(
        requested=requested,
        effective=effective,
        clamps=tuple(records),
    )


def load_config(
    card: os.PathLike[str] | str,
    *,
    dedicated: Mapping[str, object] | None = None,
    overrides: Iterable[str | ConfigOverride] = (),
    clamps: Iterable[ClampRequest] = (),
) -> RunConfig:
    """Load a card and return its effective immutable configuration."""

    return resolve_config(
        card,
        dedicated=dedicated,
        overrides=overrides,
        clamps=clamps,
    ).effective


def config_to_dict(config: object) -> dict[str, object]:
    """Convert a typed config to a stable JSON/TOML-friendly dictionary."""

    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a configuration dataclass")
    return cast(dict[str, object], _to_plain(config))


def config_to_toml(config: object) -> str:
    """Serialize one typed configuration as canonical reloadable TOML."""

    try:
        import tomli_w
    except ImportError as exc:
        raise ConfigurationError(
            "TOML serialization requires the pyAmpliCol runtime dependency 'tomli-w'"
        ) from exc
    payload = cast(dict[str, Any], _without_none(config_to_dict(config)))
    return tomli_w.dumps(payload)


def resolution_to_dict(resolution: ConfigResolution) -> dict[str, object]:
    """Return the requested/effective configuration and every resource clamp."""

    if not isinstance(resolution, ConfigResolution):
        raise TypeError("resolution_to_dict expects a ConfigResolution")
    return {
        "requested": config_to_dict(resolution.requested),
        "effective": config_to_dict(resolution.effective),
        "adjustments": [
            {
                "path": clamp.path,
                "requested": _to_plain(clamp.requested),
                "effective": _to_plain(clamp.effective),
                "reason": clamp.reason,
            }
            for clamp in resolution.clamps
        ],
    }


def _to_plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_plain(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(entry) for entry in value]
    return value


def _without_none(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_none(entry)
            for key, entry in value.items()
            if entry is not None
        }
    if isinstance(value, list):
        return [_without_none(entry) for entry in value]
    return value


__all__ = [
    "ClampRequest",
    "ConfigClamp",
    "ConfigOverride",
    "ConfigResolution",
    "config_to_dict",
    "config_to_toml",
    "load_config",
    "parse_override",
    "resolution_to_dict",
    "resolve_config",
]

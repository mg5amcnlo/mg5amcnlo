# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import pyamplicol as _pyamplicol
from pyamplicol.config import EvaluatorConfig

from .errors import ModelError

if TYPE_CHECKING:
    from pyamplicol.models.loading import CompiledModel

ModelSourceKind: TypeAlias = Literal[
    "built-in-sm",
    "built-in-sm-heft",
    "ufo",
    "json",
    "compiled",
    "prepared",
]
_PROCESS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ModelSource:
    """A built-in, packaged HEFT, UFO, JSON, or precompiled model input.

    Use :meth:`from_config` for a resolved run card or :meth:`from_path` for an
    external model. Calling :meth:`compile` produces the canonical serialized
    model consumed by process generation.
    """

    kind: ModelSourceKind
    path: Path | None = None
    restriction: Path | str | None = None
    simplify: bool = True

    def __post_init__(self) -> None:
        if self.kind not in (
            "built-in-sm",
            "built-in-sm-heft",
            "ufo",
            "json",
            "compiled",
            "prepared",
        ):
            raise ModelError(f"unsupported model source kind {self.kind!r}")
        if not isinstance(self.simplify, bool):
            raise ModelError("model simplify must be a boolean")
        if self.path is not None:
            try:
                resolved_path = (
                    Path(os.fspath(self.path)).expanduser().resolve(strict=False)
                )
            except TypeError as exc:
                raise ModelError("model path must be path-like or null") from exc
            object.__setattr__(self, "path", resolved_path)
        if self.restriction is not None:
            if isinstance(self.restriction, str):
                if not self.restriction or any(
                    character.isspace() for character in self.restriction
                ):
                    raise ModelError(
                        "model restriction names must be non-empty and contain no "
                        "whitespace"
                    )
            else:
                try:
                    resolved_restriction = (
                        Path(os.fspath(self.restriction))
                        .expanduser()
                        .resolve(strict=False)
                    )
                except TypeError as exc:
                    raise ModelError(
                        "model restriction must be a name, path-like, or null"
                    ) from exc
                object.__setattr__(self, "restriction", resolved_restriction)
        if self.kind in {"built-in-sm", "built-in-sm-heft"}:
            if self.path is not None:
                raise ModelError("built-in models have no public source path")
        elif self.kind in ("ufo", "json") and (
            self.path is None or not self.path.is_absolute()
        ):
            raise ModelError("external model paths must be absolute")
        elif self.path is not None and not self.path.is_absolute():
            raise ModelError("compiled model paths must be absolute")
        if isinstance(self.restriction, Path) and not self.restriction.is_absolute():
            raise ModelError("model restriction paths must be absolute")

    @classmethod
    def built_in_sm(cls) -> ModelSource:
        return cls(kind="built-in-sm")

    @classmethod
    def built_in_sm_heft(cls) -> ModelSource:
        """Select the packaged Standard Model with scalar HEFT interactions."""

        return cls(kind="built-in-sm-heft")

    @classmethod
    def from_config(cls, config: object) -> ModelSource:
        """Create a source from the typed ``[model]`` run-card section.

        Restriction selectors such as ``default``, ``none``, or a loader-defined
        name remain names. Explicit restriction files are resolved relative to
        the selected UFO/JSON model by :meth:`from_path`.
        """

        from pyamplicol.config import ModelConfig

        if not isinstance(config, ModelConfig):
            raise TypeError("model source config must be a ModelConfig")
        if config.source == "built-in-sm":
            if config.restriction not in (None, "default"):
                raise ModelError(
                    "model restrictions can only be applied to external models"
                )
            if not config.simplify:
                raise ModelError(
                    "simplification cannot be disabled for the built-in Standard Model"
                )
            return cls.built_in_sm()
        from pyamplicol._internal.sm_heft import is_sm_heft_alias

        if is_sm_heft_alias(config.source):
            if config.restriction not in (None, "default"):
                raise ModelError(
                    "model restrictions cannot be applied to built-in-sm-heft"
                )
            if not config.simplify:
                raise ModelError(
                    "simplification cannot be disabled for built-in-sm-heft"
                )
            return cls.built_in_sm_heft()
        return cls.from_path(
            config.source,
            restriction=config.restriction,
            simplify=config.simplify,
        )

    @classmethod
    def from_path(
        cls,
        path: os.PathLike[str] | str,
        restriction: os.PathLike[str] | str | None = None,
        simplify: bool = True,
    ) -> ModelSource:
        """Resolve an external source kind without importing Symbolica."""

        source = Path(os.fspath(path)).expanduser().resolve(strict=False)
        if not source.exists():
            raise ModelError(f"model source does not exist: {source}")
        if source.is_dir():
            kind: ModelSourceKind = "ufo"
        elif source.is_file() and source.name.lower().endswith(".pyamplicol-model"):
            kind = "prepared"
        elif source.is_file() and source.suffix.lower() == ".json":
            lowered_name = source.name.lower()
            kind = (
                "compiled"
                if lowered_name.endswith(
                    (".pyamplicol-model.json", ".compiled-model.json")
                )
                else "json"
            )
        elif source.is_file():
            kind = "compiled"
        else:
            raise ModelError(f"model source is not a file or directory: {source}")

        restriction_value: Path | str | None = None
        if restriction is not None:
            raw_restriction = os.fspath(restriction)
            if not isinstance(raw_restriction, str):
                raise ModelError(
                    "model restriction must resolve to a text path or name"
                )
            restriction_path = Path(raw_restriction).expanduser()
            if not restriction_path.is_absolute():
                base = source if source.is_dir() else source.parent
                restriction_path = base / restriction_path
            restriction_path = restriction_path.resolve(strict=False)
            if restriction_path.is_file():
                restriction_value = restriction_path
            elif not isinstance(restriction, str) or _restriction_looks_like_path(
                raw_restriction
            ):
                raise ModelError(
                    f"model restriction file does not exist: {restriction_path}"
                )
            else:
                restriction_value = raw_restriction
        return cls(
            kind=kind,
            path=source,
            restriction=restriction_value,
            simplify=simplify,
        )

    def compile(
        self,
        *,
        cache_dir: os.PathLike[str] | str | None = None,
        use_cache: bool = True,
        require_supported: bool = True,
        prepared_output: os.PathLike[str] | str | None = None,
        evaluator: EvaluatorConfig | None = None,
    ) -> _pyamplicol.CompiledModel:
        """Compile or load this source and return its compiled-model handle.

        By default this only creates the canonical, portable model IR in memory.
        Supplying both ``prepared_output`` and ``evaluator`` additionally writes
        one self-contained ``.pyamplicol-model`` bundle containing exactly the
        requested evaluator backend pack. The returned handle is loaded from that
        bundle and can be passed directly to eager, recurrence, or on-the-fly
        generation.
        """

        if (prepared_output is None) != (evaluator is None):
            raise ModelError(
                "prepared model compilation requires both prepared_output and "
                "an EvaluatorConfig"
            )
        if evaluator is not None and not isinstance(evaluator, EvaluatorConfig):
            raise ModelError("evaluator must be an EvaluatorConfig or null")

        resolved_cache: Path | None = None
        if cache_dir is not None:
            try:
                resolved_cache = (
                    Path(os.fspath(cache_dir)).expanduser().resolve(strict=False)
                )
            except TypeError as exc:
                raise ModelError(
                    "model cache directory must be path-like or null"
                ) from exc
        resolved_prepared_output: Path | None = None
        if prepared_output is not None:
            try:
                resolved_prepared_output = (
                    Path(os.fspath(prepared_output)).expanduser().resolve(strict=False)
                )
            except TypeError as exc:
                raise ModelError(
                    "prepared model output must be path-like or null"
                ) from exc
            if not resolved_prepared_output.name.lower().endswith(".pyamplicol-model"):
                raise ModelError(
                    "prepared model output must end with '.pyamplicol-model'"
                )
        try:
            payload = _compile_model_source_payload(
                self,
                cache_dir=resolved_cache,
                use_cache=use_cache,
                require_supported=require_supported,
            )
            if resolved_prepared_output is not None:
                assert evaluator is not None
                from pyamplicol.models.prepared_compile import prepare_model_bundle

                prepared = prepare_model_bundle(
                    payload,
                    resolved_prepared_output,
                    evaluator=evaluator,
                )
                from pyamplicol.models.loading import load_compiled_model

                payload = load_compiled_model(prepared.output)
            from .models import _compiled_model_from_payload

            return _compiled_model_from_payload(payload)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc


def _compile_model_source_payload(
    source: ModelSource,
    *,
    cache_dir: Path | None,
    use_cache: bool,
    require_supported: bool,
) -> CompiledModel:
    """Compile one normalized source through the shared private boundary."""

    if source.kind == "built-in-sm-heft":
        from pyamplicol._internal.sm_heft import compile_sm_heft_source

        return compile_sm_heft_source(
            cache_dir=cache_dir,
            use_cache=use_cache,
            require_supported=require_supported,
        )
    if source.kind == "built-in-sm":
        source_input: str | Path = source.kind
    elif source.path is not None:
        source_input = source.path
    else:
        raise ModelError("an in-memory compiled model source cannot be reloaded")
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        source_input,
        restriction=(
            "default" if source.restriction is None else os.fspath(source.restriction)
        ),
        simplify=source.simplify,
        cache_dir=cache_dir,
        use_cache=use_cache,
        require_supported=require_supported,
    )


def _restriction_looks_like_path(value: str) -> bool:
    path = Path(value)
    return (
        path.is_absolute()
        or path.parent != Path(".")
        or path.name.startswith("restrict_")
        or path.suffix.lower() in {".dat", ".json"}
    )


def _validate_delimiters(expression: str) -> None:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in expression:
        if character in "([{":
            stack.append(character)
        elif character in pairs and (not stack or stack.pop() != pairs[character]):
            raise ValueError("process expression has unbalanced delimiters")
    if stack:
        raise ValueError("process expression has unbalanced delimiters")


def _default_process_name(expression: str) -> str:
    readable = expression.replace("~", "bar")
    readable = readable.replace(">", "_to_")
    readable = re.sub(r"[^A-Za-z0-9]+", "_", readable).strip("_").lower()
    if not readable or not readable[0].isalpha():
        readable = f"process_{readable}" if readable else "process"
    if len(readable) <= 72:
        return readable
    digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:61]}_{digest}"


@dataclass(frozen=True, slots=True)
class ProcessRequest:
    """One concrete or multiparticle process expression and its artifact name."""

    expression: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression:
            raise ValueError("process expression must be a non-empty string")
        if not isinstance(self.name, str) or not _PROCESS_NAME.fullmatch(self.name):
            raise ValueError(
                "process name must start with a letter and contain only "
                "letters, digits, '.', '_', or '-'"
            )

    @classmethod
    def parse(cls, expression: str, *, name: str | None = None) -> ProcessRequest:
        """Normalize an expression and derive a stable readable name if omitted."""

        if not isinstance(expression, str):
            raise TypeError("process expression must be a string")
        if "\n" in expression or "\r" in expression:
            raise ValueError("one ProcessRequest may contain only one line")
        normalized = " ".join(expression.split())
        if not normalized:
            raise ValueError("process expression must not be empty")
        _validate_delimiters(normalized)
        if ">" not in normalized:
            raise ValueError("process expression must contain '>'")
        for segment in normalized.split(","):
            if ">" not in segment:
                continue
            if segment.count(">") != 1:
                raise ValueError("each process clause must contain exactly one arrow")
            sides = segment.split(">")
            if any(not side.strip() for side in sides):
                raise ValueError("each process arrow must have particles on both sides")
        selected_name = _default_process_name(normalized) if name is None else name
        return cls(expression=normalized, name=selected_name)


@dataclass(frozen=True, slots=True)
class ProcessAlias:
    """A public process alias backed by a generated concrete process.

    ``particle_permutation`` maps each stored representative external index to
    its public alias index. Incoming and outgoing legs may each be reordered,
    but a leg may not cross the process arrow. Runtime selectors and metadata
    are remapped to the alias automatically.
    """

    name: str
    process_name: str
    particle_permutation: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not _PROCESS_NAME.fullmatch(self.name):
            raise ValueError(f"invalid process alias name {self.name!r}")
        if not _PROCESS_NAME.fullmatch(self.process_name):
            raise ValueError(f"invalid aliased process name {self.process_name!r}")
        permutation = tuple(self.particle_permutation)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in permutation
        ):
            raise ValueError("particle_permutation must contain non-negative integers")
        if permutation and sorted(permutation) != list(range(len(permutation))):
            raise ValueError("particle_permutation must be a complete permutation")
        object.__setattr__(self, "particle_permutation", permutation)


@dataclass(frozen=True, slots=True)
class ProcessSet:
    """A named collection of generation requests and optional crossing aliases."""

    requests: tuple[ProcessRequest, ...]
    aliases: tuple[ProcessAlias, ...] = ()

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        aliases = tuple(self.aliases)
        if not requests:
            raise ValueError("a ProcessSet must contain at least one request")
        if not all(isinstance(request, ProcessRequest) for request in requests):
            raise TypeError("ProcessSet.requests must contain ProcessRequest objects")
        if not all(isinstance(alias, ProcessAlias) for alias in aliases):
            raise TypeError("ProcessSet.aliases must contain ProcessAlias objects")
        request_names = tuple(request.name for request in requests)
        if len(set(request_names)) != len(request_names):
            raise ValueError("ProcessSet request names must be unique")
        alias_names = tuple(alias.name for alias in aliases)
        if len(set(alias_names)) != len(alias_names):
            raise ValueError("ProcessSet alias names must be unique")
        if set(alias_names).intersection(request_names):
            raise ValueError("process alias names may not shadow request names")
        unknown_targets = {
            alias.process_name
            for alias in aliases
            if alias.process_name not in request_names
        }
        if unknown_targets:
            raise ValueError(
                "process aliases refer to unknown requests: "
                + ", ".join(sorted(unknown_targets))
            )
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "aliases", aliases)

    @classmethod
    def from_expressions(
        cls,
        expressions: Iterable[str],
        *,
        names: Sequence[str] = (),
        aliases: Iterable[ProcessAlias] = (),
    ) -> ProcessSet:
        """Build a process set from expressions using optional aligned names."""

        expression_tuple = tuple(expressions)
        name_tuple = tuple(names)
        if name_tuple and len(name_tuple) != len(expression_tuple):
            raise ValueError("process names must be empty or aligned with expressions")
        requests = tuple(
            ProcessRequest.parse(
                expression,
                name=name_tuple[index] if name_tuple else None,
            )
            for index, expression in enumerate(expression_tuple)
        )
        return cls(requests=requests, aliases=tuple(aliases))


__all__ = [
    "ModelSource",
    "ModelSourceKind",
    "ProcessAlias",
    "ProcessRequest",
    "ProcessSet",
]

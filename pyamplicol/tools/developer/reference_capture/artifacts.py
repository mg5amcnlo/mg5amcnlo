# SPDX-License-Identifier: 0BSD
"""Artifact generation, runtime extraction, and fixture assembly."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

from .common import (
    ANALYTIC_CERTIFIED_DIGITS,
    ANALYTIC_EVIDENCE_SET_ID,
    FORTRAN_CERTIFIED_DIGITS,
    FORTRAN_EVIDENCE_SET_ID,
    MODEL_ASSETS,
    OBSERVATION_PRECISION,
    ROOT,
    WATCHDOG_GB,
    ArtifactCaptureSpec,
    CaptureError,
    CapturePoint,
    DependencySnapshot,
    ProcessCaptureSpec,
    RuntimeSnapshot,
    SourceSnapshot,
    as_mapping,
    as_sequence,
    canonical_decimal,
    decimal_digits_to_bits,
    developer_module,
)
from .points import build_reference_points, stable_external_leg_ids

SM_ONE_BODY_PROCESSES = (
    ProcessCaptureSpec(
        "sm_ddbar_z",
        "d d~ > z",
        "model:builtin-sm",
        ("0", "0", "91.188"),
        "exact-2to1",
    ),
    ProcessCaptureSpec(
        "sm_udbar_wplus",
        "u d~ > w+",
        "model:builtin-sm",
        ("0", "0", "80.41900244575616"),
        "exact-2to1",
    ),
)
SM_TWO_BODY_PROCESSES = (
    ProcessCaptureSpec(
        "sm_ddbar_zg",
        "d d~ > z g",
        "model:builtin-sm",
        ("0", "0", "91.188", "0"),
        "seeded-two-body",
        (104729, 104759, 104761),
    ),
    ProcessCaptureSpec(
        "sm_ddbar_ee",
        "d d~ > e- e+",
        "model:builtin-sm",
        ("0", "0", "0", "0"),
        "seeded-two-body",
        (196613, 196643, 196657),
    ),
    ProcessCaptureSpec(
        "sm_ddbar_uubar",
        "d d~ > u u~",
        "model:builtin-sm",
        ("0", "0", "0", "0"),
        "seeded-two-body",
        (221603, 221621, 221623),
    ),
    ProcessCaptureSpec(
        "sm_ddbar_ddbar",
        "d d~ > d d~",
        "model:builtin-sm",
        ("0", "0", "0", "0"),
        "seeded-two-body",
        (234803, 234811, 234817),
    ),
    ProcessCaptureSpec(
        "sm_gg_gg",
        "g g > g g",
        "model:builtin-sm",
        ("0", "0", "0", "0"),
        "seeded-two-body",
        (246613, 246631, 246637),
    ),
    ProcessCaptureSpec(
        "sm_gg_ttbar",
        "g g > t t~",
        "model:builtin-sm",
        ("0", "0", "173", "173"),
        "seeded-two-body",
        (271603, 271619, 271637),
    ),
)
SM_THREE_BODY_PROCESSES = (
    ProcessCaptureSpec(
        "sm_ddbar_zgg",
        "d d~ > z g g",
        "model:builtin-sm",
        ("0", "0", "91.188", "0", "0"),
        "one-massive-two-massless",
        (130363, 130367, 130369),
    ),
)
SCALAR_PROCESS = ProcessCaptureSpec(
    "scalars_2to2",
    "scalar_0 scalar_0 > scalar_0 scalar_0",
    "model:scalars-json",
    ("0", "0", "0", "0"),
    "seeded-two-body",
    (155921, 155933, 155947),
)
SCALAR_GRAVITY_PROCESS = ProcessCaptureSpec(
    "scalar_gravity_2to2",
    "scalar_0 scalar_0 > graviton graviton",
    "model:scalar-gravity-json",
    ("0", "0", "0", "0"),
    "seeded-two-body",
    (181081, 181087, 181123),
)


def artifact_capture_specs(artifact_root: Path) -> tuple[ArtifactCaptureSpec, ...]:
    """Return the fixed named process ladder captured under ``artifact_root``."""

    del artifact_root
    scalar_source = MODEL_ASSETS / "scalars" / "scalars.json"
    gravity_source = MODEL_ASSETS / "scalar_gravity" / "scalar_gravity.json"
    return (
        *(
            ArtifactCaptureSpec(
                id=f"builtin-sm-one-body-{accuracy}",
                directory_name=f"builtin-sm-one-body-{accuracy}",
                model_id="model:builtin-sm",
                model_source=None,
                color_accuracy=cast(Literal["lc", "nlc", "full"], accuracy),
                processes=SM_ONE_BODY_PROCESSES,
            )
            for accuracy in ("lc", "nlc", "full")
        ),
        *(
            ArtifactCaptureSpec(
                id=f"builtin-sm-two-body-{accuracy}",
                directory_name=f"builtin-sm-two-body-{accuracy}",
                model_id="model:builtin-sm",
                model_source=None,
                color_accuracy=cast(Literal["lc", "nlc", "full"], accuracy),
                processes=SM_TWO_BODY_PROCESSES,
            )
            for accuracy in ("lc", "nlc", "full")
        ),
        ArtifactCaptureSpec(
            id="scalars-json-lc",
            directory_name="scalars-json-lc",
            model_id="model:scalars-json",
            model_source=scalar_source,
            color_accuracy="lc",
            processes=(SCALAR_PROCESS,),
        ),
        ArtifactCaptureSpec(
            id="scalar-gravity-json-lc",
            directory_name="scalar-gravity-json-lc",
            model_id="model:scalar-gravity-json",
            model_source=gravity_source,
            color_accuracy="lc",
            processes=(SCALAR_GRAVITY_PROCESS,),
        ),
        *(
            ArtifactCaptureSpec(
                id=f"builtin-sm-three-body-{accuracy}",
                directory_name=f"builtin-sm-three-body-{accuracy}",
                model_id="model:builtin-sm",
                model_source=None,
                color_accuracy=cast(Literal["lc", "nlc", "full"], accuracy),
                processes=SM_THREE_BODY_PROCESSES,
            )
            for accuracy in ("lc", "nlc", "full")
        ),
    )


def _api_types() -> tuple[Any, Any, Any, Any, Any]:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    api = importlib.import_module("pyamplicol")
    config = importlib.import_module("pyamplicol.config")
    return (
        api.Generator,
        api.ModelSource,
        api.ProcessRequest,
        api.ProcessSet,
        config,
    )


def materialize_artifacts(
    specs: Sequence[ArtifactCaptureSpec],
    artifact_root: Path,
    mode: Literal["generate", "reuse"],
) -> dict[str, Path]:
    """Generate without replacement or explicitly validate retained locations."""

    if mode not in {"generate", "reuse"}:
        raise CaptureError(f"unsupported artifact mode {mode!r}")
    paths = {spec.id: artifact_root / spec.directory_name for spec in specs}
    if mode == "reuse":
        missing = [
            path for path in paths.values() if not (path / "artifact.json").is_file()
        ]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise CaptureError(
                "explicit artifact reuse requested, but schema-v3 manifests are "
                f"missing from: {rendered}"
            )
        return paths

    Generator, ModelSource, ProcessRequest, ProcessSet, config_module = _api_types()
    artifact_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        destination = paths[spec.id]
        if destination.exists():
            raise CaptureError(
                f"artifact destination already exists: {destination}; use "
                "--reuse-artifacts to retain and reuse it, or choose a new root"
            )
        process_set = ProcessSet(
            tuple(
                ProcessRequest.parse(process.expression, name=process.id)
                for process in spec.processes
            )
        )
        run_config = config_module.RunConfig(
            action="generate",
            color=config_module.ColorConfig(accuracy=spec.color_accuracy),
            generation=config_module.GenerationConfig(
                validation=config_module.GenerationValidationConfig(samples=10)
            ),
        )
        model = (
            None
            if spec.model_source is None
            else ModelSource.from_path(spec.model_source)
        )
        try:
            result = Generator(run_config).generate(
                process_set,
                destination,
                model=model,
                mode="error",
            )
        except Exception as error:
            raise CaptureError(
                f"schema-v3 generation failed for {spec.id}; any completed artifacts "
                f"remain under {artifact_root}: {error}"
            ) from error
        if result.schema_version != 3 or result.output != destination.resolve():
            raise CaptureError(
                f"generation returned invalid result metadata for {spec.id}"
            )
    return paths


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read artifact JSON {path}: {error}") from error
    return as_mapping(value, str(path))


def _load_manifest(path: Path) -> Any:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    artifacts = importlib.import_module("pyamplicol.artifacts")
    try:
        return artifacts.load_manifest(path, verify_payloads=True)
    except Exception as error:
        raise CaptureError(
            f"invalid retained schema-v3 artifact {path}: {error}"
        ) from error


def _load_runtime(path: Path, process_id: str) -> Any:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    api = importlib.import_module("pyamplicol")
    try:
        return api.Runtime.load(path, process=process_id)
    except Exception as error:
        raise CaptureError(
            f"cannot load runtime for {process_id!r} from {path}: {error}"
        ) from error


def _payload_record(manifest: Any, role: str, process_id: str | None) -> Any:
    matches = tuple(
        record
        for record in manifest.payloads
        if record.role == role and record.process_id == process_id
    )
    if len(matches) != 1:
        scope = "model" if process_id is None else f"process {process_id}"
        raise CaptureError(
            f"artifact {manifest.root} must contain exactly one {role} payload for "
            f"{scope}; found {len(matches)}"
        )
    return matches[0]


def _artifact_process(manifest: Any, process_id: str) -> Mapping[str, object]:
    matches = tuple(
        process for process in manifest.processes if process["id"] == process_id
    )
    if len(matches) != 1:
        raise CaptureError(
            f"artifact {manifest.root} does not contain exactly one process "
            f"{process_id}"
        )
    return cast(Mapping[str, object], matches[0])


def _json_number(value: object, where: str) -> str:
    if isinstance(value, (Decimal, float, int)) and not isinstance(value, bool):
        return canonical_decimal(value)
    raise CaptureError(f"{where} is not a finite JSON number")


def _compiled_model_contract_sha256(compiled: Mapping[str, object]) -> str:
    semantic_payload = {
        key: value
        for key, value in compiled.items()
        if key not in {"conversion_seconds", "phase_timings"}
    }
    encoded = _canonical_json_bytes(semantic_payload)
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, (Decimal, float)):
        return canonical_decimal(value).encode("ascii")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True).encode("ascii")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CaptureError("canonical compiled-model JSON has a non-string key")
        members = (
            json.dumps(key, ensure_ascii=True).encode("ascii")
            + b":"
            + _canonical_json_bytes(value[key])
            for key in sorted(cast(Mapping[str, object], value))
        )
        return b"{" + b",".join(members) + b"}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return b"[" + b",".join(_canonical_json_bytes(item) for item in value) + b"]"
    raise CaptureError(
        f"canonical compiled-model JSON contains unsupported {type(value).__name__}"
    )


def _model_payload(
    spec: ArtifactCaptureSpec,
    manifest: Any,
    dependencies: DependencySnapshot,
    compiled: Mapping[str, object],
) -> dict[str, object]:
    if compiled.get("schema_version") != manifest.model["compiled_schema_version"]:
        raise CaptureError(f"compiled-model schema mismatch in {manifest.root}")
    source = as_mapping(compiled.get("source"), "compiled-model.source")
    if source.get("digest") != manifest.model["content_sha256"]:
        raise CaptureError(f"compiled-model source digest mismatch in {manifest.root}")
    defaults = as_mapping(
        compiled.get("parameter_defaults"), "compiled-model.parameter_defaults"
    )
    parameter_defaults: dict[str, object] = {}
    for name, raw_value in sorted(defaults.items()):
        components = as_sequence(raw_value, f"parameter_defaults.{name}")
        if len(components) != 2:
            raise CaptureError(f"model parameter {name!r} is not complex-valued")
        parameter_defaults[str(name)] = {
            "real": _json_number(components[0], f"parameter_defaults.{name}.real"),
            "imag": _json_number(components[1], f"parameter_defaults.{name}.imag"),
        }

    lock = dependencies.release_lock
    symbolica = as_mapping(lock.get("symbolica"), "release-lock.symbolica")
    loader = as_mapping(lock.get("ufo_model_loader"), "release-lock.ufo_model_loader")
    producer = as_mapping(compiled.get("producer"), "compiled-model.producer")
    if producer.get("symbolica") != symbolica["python_version"]:
        raise CaptureError(
            "compiled model was not produced by release-locked Symbolica"
        )
    if producer.get("ufo_model_loader") != loader["required_version"]:
        raise CaptureError(
            "compiled model was not produced by release-locked ufo-model-loader"
        )
    artifact_symbolica = tuple(
        dependency
        for dependency in manifest.dependencies
        if str(dependency["name"]).lower() == "symbolica"
    )
    if len(artifact_symbolica) != 1 or (
        artifact_symbolica[0]["version"] != symbolica["python_version"]
    ):
        raise CaptureError("artifact dependency metadata disagrees with release lock")

    return {
        "id": spec.model_id,
        "name": str(manifest.model["name"]),
        "source_kind": str(manifest.model["source_kind"]),
        "content_sha256": str(manifest.model["content_sha256"]),
        "compiled_model_sha256": _compiled_model_contract_sha256(compiled),
        "compiled_schema_version": int(manifest.model["compiled_schema_version"]),
        "restriction": manifest.model.get("restriction"),
        "dependency_ids": [
            "dependency:pyamplicol-candidate",
            "dependency:release-lock",
            "dependency:install-state",
            "dependency:symbolica",
            "dependency:symjit",
            "dependency:ufo-model-loader",
        ],
        "parameter_defaults": parameter_defaults,
    }


def _external_model_metadata(
    process_id: str,
    pdgs: Sequence[int],
    compiled: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[Decimal, ...]]:
    ir = as_mapping(compiled.get("ir"), "compiled-model.ir")
    raw_particles = as_sequence(ir.get("particles"), "compiled-model.ir.particles")
    particle_by_pdg: dict[int, Mapping[str, object]] = {}
    for index, raw_particle in enumerate(raw_particles):
        particle = as_mapping(raw_particle, f"compiled-model.ir.particles[{index}]")
        pdg = particle.get("pdg_code")
        if isinstance(pdg, bool) or not isinstance(pdg, int):
            raise CaptureError("compiled-model particle PDG is not an integer")
        if pdg in particle_by_pdg:
            raise CaptureError(f"compiled-model contains duplicate particle PDG {pdg}")
        particle_by_pdg[pdg] = particle
    defaults = as_mapping(
        compiled.get("parameter_defaults"), "compiled-model.parameter_defaults"
    )
    spins: list[int] = []
    colors: list[int] = []
    masses: list[Decimal] = []
    for pdg in pdgs:
        external_particle = particle_by_pdg.get(pdg)
        if external_particle is None:
            raise CaptureError(f"{process_id} PDG {pdg} is absent from compiled model")
        if external_particle.get("propagating") is not True:
            raise CaptureError(f"{process_id} PDG {pdg} is not a propagating particle")
        spin = external_particle.get("spin")
        color = external_particle.get("color")
        if isinstance(spin, bool) or not isinstance(spin, int) or spin < 1:
            raise CaptureError(f"{process_id} PDG {pdg} has invalid UFO spin")
        if isinstance(color, bool) or not isinstance(color, int):
            raise CaptureError(
                f"{process_id} PDG {pdg} has invalid color representation"
            )
        mass_name = external_particle.get("mass")
        if not isinstance(mass_name, str) or not mass_name:
            raise CaptureError(f"{process_id} PDG {pdg} has invalid mass metadata")
        if mass_name == "ZERO":
            mass = Decimal(0)
        else:
            components = as_sequence(
                defaults.get(mass_name),
                f"compiled-model.parameter_defaults.{mass_name}",
            )
            if len(components) != 2:
                raise CaptureError(
                    f"mass parameter {mass_name!r} is not complex-valued"
                )
            mass = Decimal(str(components[0]))
            imaginary = Decimal(str(components[1]))
            if not mass.is_finite() or mass < 0 or imaginary != 0:
                raise CaptureError(
                    f"mass parameter {mass_name!r} is not finite, real, and "
                    "non-negative"
                )
        spins.append(spin)
        colors.append(color)
        masses.append(mass)
    return tuple(spins), tuple(colors), tuple(masses)


def _physical_helicity_domain(spin: int, mass: Decimal) -> tuple[int, ...]:
    if spin == 1:
        return (0,)
    if spin % 2 == 0:
        return tuple(range(-(spin - 1), spin, 2))
    physical_spin = (spin - 1) // 2
    if mass == 0:
        return (-physical_spin, physical_spin)
    return tuple(range(-physical_spin, physical_spin + 1))


def _process_payload(
    process_spec: ProcessCaptureSpec,
    manifest_process: Mapping[str, object],
    physics: Any,
    compiled: Mapping[str, object],
) -> dict[str, object]:
    particles = tuple(sorted(physics.external_particles, key=lambda item: item.index))
    indices = tuple(particle.index for particle in particles)
    if indices != tuple(range(len(particles))):
        raise CaptureError(f"{process_spec.id} external-particle indices are not dense")
    pdgs = tuple(particle.pdg_id for particle in particles)
    manifest_pdg_values = as_sequence(
        manifest_process.get("external_pdgs"),
        f"{process_spec.id} manifest external_pdgs",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in manifest_pdg_values
    ):
        raise CaptureError(f"{process_spec.id} manifest PDGs are not integers")
    manifest_pdgs = tuple(cast(int, value) for value in manifest_pdg_values)
    if pdgs != manifest_pdgs:
        raise CaptureError(f"{process_spec.id} runtime and manifest PDG order differs")
    if physics.process != process_spec.expression or (
        manifest_process["expression"] != process_spec.expression
    ):
        raise CaptureError(
            f"{process_spec.id} process expression is not the named input"
        )
    if physics.helicity_coverage != "complete":
        raise CaptureError(
            f"{process_spec.id} lacks complete runtime helicity coverage"
        )
    spins, colors, masses = _external_model_metadata(process_spec.id, pdgs, compiled)
    domains = tuple(
        _physical_helicity_domain(spin, mass)
        for spin, mass in zip(spins, masses, strict=True)
    )
    complete_axis = set(product(*domains))
    actual_axis = {tuple(helicity.values) for helicity in physics.helicities}
    if actual_axis != complete_axis:
        raise CaptureError(
            f"{process_spec.id} runtime axis does not span its per-leg domains"
        )
    return {
        "id": process_spec.id,
        "expression": process_spec.expression,
        "external_pdgs": list(pdgs),
        "external_labels": [particle.label for particle in particles],
        "external_leg_ids": list(stable_external_leg_ids(particles)),
        "external_spins": list(spins),
        "external_colors": list(colors),
        "external_masses": [canonical_decimal(mass) for mass in masses],
        "external_helicity_domains": [list(domain) for domain in domains],
        "initial_state_count": sum(
            particle.state == "incoming" for particle in particles
        ),
        "alias_of": None,
        "final_state_permutation": None,
    }


def _axis_payload(physics: Any) -> dict[str, object]:
    helicities = [
        {
            "id": helicity.id,
            "index": helicity.index,
            "values": list(helicity.values),
            "computed": helicity.computed,
            "structural_zero": helicity.structural_zero,
            "representative_id": helicity.representative_id,
            "coefficient": canonical_decimal(helicity.coefficient),
        }
        for helicity in physics.helicities
    ]
    if physics.color_accuracy == "lc":
        colors = [
            {
                "kind": "lc-flow",
                "id": color.id,
                "index": color.index,
                "word": list(color.word),
                "computed": color.computed,
                "representative_id": color.representative_id,
                "coefficient": canonical_decimal(color.coefficient),
            }
            for color in physics.color_flows
        ]
    else:
        colors = [
            {
                "kind": "contracted-color",
                "id": color.id,
                "index": color.index,
                "description": color.description,
            }
            for color in physics.contracted_color_components
        ]
    return {"helicities": helicities, "colors": colors}


def _normalization_payload(
    physics_payload: Mapping[str, object],
    color_accuracy: str,
) -> dict[str, object]:
    extensions = as_mapping(physics_payload.get("extensions"), "physics.extensions")
    normalization = as_mapping(
        extensions.get("normalization"), "physics.extensions.normalization"
    )
    if normalization.get("color_accuracy") != color_accuracy:
        raise CaptureError("runtime normalization color accuracy is inconsistent")
    coupling_flag = normalization.get("couplings_in_stage_evaluators")
    if not isinstance(coupling_flag, bool):
        raise CaptureError(
            "normalization.couplings_in_stage_evaluators must be boolean"
        )
    return {
        "average_factor": _json_number(
            normalization.get("average_factor"), "normalization.average_factor"
        ),
        "color_factor": _json_number(
            normalization.get("color_factor"), "normalization.color_factor"
        ),
        "identical_factor": _json_number(
            normalization.get("identical_factor"), "normalization.identical_factor"
        ),
        "global_coupling_factor": _json_number(
            normalization.get("global_coupling_factor"),
            "normalization.global_coupling_factor",
        ),
        "quark_line_partner_factor": _json_number(
            normalization.get("quark_line_partner_factor", 1),
            "normalization.quark_line_partner_factor",
        ),
        "couplings_in_stage_evaluators": coupling_flag,
    }


def _topology_payload(
    execution_payload: Mapping[str, object], physics: Any
) -> dict[str, object]:
    summary = as_mapping(execution_payload.get("dag_summary"), "execution.dag_summary")
    required = ("current_count", "interaction_count", "amplitude_root_count")
    if any(
        isinstance(summary.get(name), bool) or not isinstance(summary.get(name), int)
        for name in required
    ):
        raise CaptureError(
            "execution DAG summary does not contain integer topology counts"
        )
    return {
        "currents": int(cast(int, summary["current_count"])),
        "interactions": int(cast(int, summary["interaction_count"])),
        "roots": int(cast(int, summary["amplitude_root_count"])),
        "reduction_groups": len(physics.reduction.groups),
    }


def _reduction_payload(physics: Any, *, lc: bool) -> dict[str, object]:
    groups = [
        {
            "id": str(group.id),
            "representative_helicity_id": str(group.representative_helicity_id),
            "representative_color_id": str(group.representative_color_id),
            "physical_helicity_ids": [
                str(identifier) for identifier in group.physical_helicity_ids
            ],
            "physical_color_ids": [
                str(identifier) for identifier in group.physical_color_ids
            ],
        }
        for group in physics.reduction.groups
    ]
    if not groups:
        raise CaptureError("runtime physics metadata has no reduction groups")
    return {
        "kind": physics.reduction.kind,
        "cell_semantics": (
            "sum-all-contributing-groups" if lc else "fully-contracted-color"
        ),
        "groups": groups,
        "plan_sha256": "0" * 64,
    }


def _exact_decimal_sum(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    if not all(value.is_finite() for value in values):
        raise CaptureError("runtime returned a non-finite resolved observation")

    def finite_exponent(value: Decimal) -> int:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            raise CaptureError("finite Decimal has a non-integral exponent")
        return exponent

    common_exponent = min(finite_exponent(value) for value in values)
    total = 0
    for value in values:
        sign, digits, _ = value.as_tuple()
        exponent = finite_exponent(value)
        coefficient = int("".join(str(digit) for digit in digits) or "0")
        if sign:
            coefficient = -coefficient
        total += coefficient * 10 ** (exponent - common_exponent)
    if total == 0:
        return Decimal(0)
    digits = tuple(int(character) for character in str(abs(total)))
    return Decimal((int(total < 0), digits, common_exponent))


def _resolved_observations(
    runtime: Any,
    physics: Any,
    points: Sequence[CapturePoint],
    evidence_ids: Sequence[str],
    certified_digits: int,
) -> list[dict[str, object]]:
    momenta = tuple(point.momenta for point in points)
    try:
        resolved = runtime.evaluate_resolved(momenta, precision=OBSERVATION_PRECISION)
    except Exception as error:
        raise CaptureError(
            f"p{OBSERVATION_PRECISION} resolved evaluation failed for "
            f"{physics.process_id}: {error}"
        ) from error
    if resolved.helicity_ids != physics.helicity_ids or (
        resolved.color_ids != physics.color_ids
    ):
        raise CaptureError(
            f"resolved output axes differ from runtime physics for {physics.process_id}"
        )
    if len(resolved.values) != len(points):
        raise CaptureError(f"runtime omitted points for {physics.process_id}")
    observations: list[dict[str, object]] = []
    for point, point_values, evidence_id in zip(
        points, resolved.values, evidence_ids, strict=True
    ):
        rows: list[list[str]] = []
        flat: list[Decimal] = []
        for helicity, raw_row in zip(physics.helicities, point_values, strict=True):
            row: list[str] = []
            for raw_value in raw_row:
                if not isinstance(raw_value, Decimal) or not raw_value.is_finite():
                    raise CaptureError(
                        "p80 runtime returned a non-Decimal cell for "
                        f"{physics.process_id}"
                    )
                if helicity.structural_zero and raw_value != 0:
                    raise CaptureError(
                        f"runtime violated structural zero {helicity.id} at {point.id}"
                    )
                flat.append(raw_value)
                row.append(canonical_decimal(raw_value))
            rows.append(row)
        total = _exact_decimal_sum(flat)
        observations.append(
            {
                "point_id": point.id,
                "arithmetic_precision_bits": decimal_digits_to_bits(
                    OBSERVATION_PRECISION
                ),
                "round_trip_decimal_digits": OBSERVATION_PRECISION,
                "certified_decimal_digits": certified_digits,
                "values": rows,
                "total": canonical_decimal(total),
                "evidence_refs": [evidence_id],
            }
        )
    return observations


def _case_payload(
    spec: ArtifactCaptureSpec,
    process_spec: ProcessCaptureSpec,
    manifest: Any,
    manifest_process: Mapping[str, object],
    runtime: Any,
    points: Sequence[CapturePoint],
) -> dict[str, object]:
    physics = runtime.physics
    physics_record = _payload_record(manifest, "runtime-physics", process_spec.id)
    execution_record = _payload_record(manifest, "evaluator-manifest", process_spec.id)
    physics_payload = _read_json(manifest.root / physics_record.path)
    execution_payload = _read_json(manifest.root / execution_record.path)
    if physics_payload.get("process_id") != process_spec.id or (
        execution_payload.get("process") != process_spec.expression
    ):
        raise CaptureError(f"artifact payload identity mismatch for {process_spec.id}")
    if physics.color_accuracy != spec.color_accuracy or (
        manifest_process["color_accuracy"] != spec.color_accuracy
    ):
        raise CaptureError(f"artifact color accuracy mismatch for {process_spec.id}")
    axes = _axis_payload(physics)
    color_count = len(cast(Sequence[object], axes["colors"]))
    capabilities = set(physics.selector_capabilities)
    if "helicity" not in capabilities:
        raise CaptureError(
            f"runtime does not expose helicity selection: {process_spec.id}"
        )
    case_id = f"case:{process_spec.id}:{spec.color_accuracy}"
    evidence_ids = tuple(
        (
            f"evidence:fortran:{case_id}:{point.id}"
            if spec.model_id == "model:builtin-sm"
            else f"evidence:analytic:{case_id}:{point.id}"
        )
        for point in points
    )
    lc = spec.color_accuracy == "lc"
    certified_digits = (
        FORTRAN_CERTIFIED_DIGITS
        if spec.model_id == "model:builtin-sm"
        else ANALYTIC_CERTIFIED_DIGITS
    )
    return {
        "id": case_id,
        "case_kind": "substantive",
        "model_id": spec.model_id,
        "process_id": process_spec.id,
        "color_accuracy": spec.color_accuracy,
        "point_policy": (
            "degenerate-2to1"
            if process_spec.point_policy == "exact-2to1"
            else "standard"
        ),
        "point_ids": [point.id for point in points],
        "coverage": {
            "helicities": physics.helicity_coverage,
            "color": physics.color_coverage,
            "color_kind": physics.color_kind,
            "helicity_count": len(physics.helicities),
            "color_component_count": color_count,
            "structural_zero_helicity_count": physics.structural_zero_helicity_count,
        },
        "selectors": {
            "helicity": "helicity" in capabilities,
            "color_flow": "color_flow" in capabilities,
            "omitted_helicity": "all-components",
            "omitted_color": "all-components" if lc else "contracted-component",
        },
        "normalization": _normalization_payload(physics_payload, spec.color_accuracy),
        "topology": _topology_payload(execution_payload, physics),
        "artifact_physics_sha256": physics_record.sha256,
        "artifact_execution_sha256": execution_record.sha256,
        "physics_case_sha256": "0" * 64,
        "axes": axes,
        "reduction": _reduction_payload(physics, lc=lc),
        "observations": _resolved_observations(
            runtime,
            physics,
            points,
            evidence_ids,
            certified_digits,
        ),
    }


def _validated_capture_masses(
    process_spec: ProcessCaptureSpec,
    process_payload: Mapping[str, object],
) -> tuple[Decimal, ...]:
    try:
        expected = tuple(
            Decimal(value) for value in process_spec.expected_external_masses
        )
        actual = tuple(
            Decimal(str(value))
            for value in cast(
                Sequence[object], process_payload["external_masses"]
            )
        )
    except (ArithmeticError, KeyError) as error:
        raise CaptureError(
            f"invalid mass contract for {process_spec.expression!r}"
        ) from error
    if any(
        canonical_decimal(value) != declared
        for value, declared in zip(
            expected,
            process_spec.expected_external_masses,
            strict=True,
        )
    ):
        raise CaptureError(
            f"declared masses for {process_spec.expression!r} are not canonical"
        )
    if actual != expected:
        raise CaptureError(
            f"artifact masses {actual} for {process_spec.expression!r} differ "
            f"from the independent capture contract {expected}"
        )
    return expected


def assemble_fixture(
    specs: Sequence[ArtifactCaptureSpec],
    artifact_paths: Mapping[str, Path],
    source: SourceSnapshot,
    runtime_snapshot: RuntimeSnapshot,
    dependencies: DependencySnapshot,
    *,
    captured_at: str,
    capture_command: Sequence[str],
) -> dict[str, object]:
    """Assemble the fixture exclusively from validated artifacts and runtimes."""

    process_payloads: dict[str, dict[str, object]] = {}
    point_payloads: dict[str, tuple[CapturePoint, ...]] = {}
    model_payloads: dict[str, dict[str, object]] = {}
    cases: list[dict[str, object]] = []
    for spec in specs:
        path = artifact_paths.get(spec.id)
        if path is None:
            raise CaptureError(f"no artifact path was supplied for {spec.id}")
        manifest = _load_manifest(path)
        producer = as_mapping(manifest.producer, f"artifact {path} producer")
        if producer.get("version") != runtime_snapshot.version:
            raise CaptureError(
                f"artifact {path} was produced by pyamplicol "
                f"{producer.get('version')!r}, not the validated capture candidate "
                f"{runtime_snapshot.version!r}"
            )
        expected = {process.id: process.expression for process in spec.processes}
        actual = {
            str(process["id"]): str(process["expression"])
            for process in manifest.processes
        }
        if actual != expected:
            raise CaptureError(
                f"artifact {path} does not contain the required named process set"
            )
        compiled_record = _payload_record(manifest, "compiled-model", None)
        compiled = _read_json(manifest.root / compiled_record.path)
        model_payload = _model_payload(spec, manifest, dependencies, compiled)
        previous_model = model_payloads.setdefault(spec.model_id, model_payload)
        if previous_model != model_payload:
            raise CaptureError(
                f"retained artifacts disagree on model provenance for {spec.model_id}"
            )
        for process_spec in spec.processes:
            manifest_process = _artifact_process(manifest, process_spec.id)
            runtime = _load_runtime(path, process_spec.id)
            process_payload = _process_payload(
                process_spec, manifest_process, runtime.physics, compiled
            )
            previous_process = process_payloads.setdefault(
                process_spec.id, process_payload
            )
            if previous_process != process_payload:
                raise CaptureError(
                    "retained artifacts disagree on process metadata for "
                    f"{process_spec.id}"
                )
            points = point_payloads.setdefault(
                process_spec.id,
                build_reference_points(
                    process_spec,
                    _validated_capture_masses(process_spec, process_payload),
                ),
            )
            cases.append(
                _case_payload(
                    spec,
                    process_spec,
                    manifest,
                    manifest_process,
                    runtime,
                    points,
                )
            )

    process_order = tuple(
        dict.fromkeys(process.id for spec in specs for process in spec.processes)
    )
    model_order = tuple(dict.fromkeys(spec.model_id for spec in specs))
    fixture: dict[str, object] = {
        "fixture_schema_version": 2,
        "kind": "pyamplicol-reference-physics",
        "fixture_id": "fixture:compact-reference-ladder-v2",
        "provenance": {
            "source_repository": source.repository_uri,
            "source_revision": source.revision,
            "source_tree_sha256": source.tree_sha256,
            "captured_at": captured_at,
            "capture_command": list(capture_command),
            "working_tree_clean": True,
            "memory_watchdog_gb": WATCHDOG_GB,
        },
        "dependencies": list(dependencies.payloads),
        "models": [model_payloads[model_id] for model_id in model_order],
        "processes": [process_payloads[process_id] for process_id in process_order],
        "points": [
            point.as_payload()
            for process_id in process_order
            for point in point_payloads[process_id]
        ],
        "cases": cases,
        "evidence_sets": [
            FORTRAN_EVIDENCE_SET_ID,
            ANALYTIC_EVIDENCE_SET_ID,
        ],
    }
    reference = developer_module("reference_fixture_v2")
    for raw_case in cast(list[dict[str, object]], fixture["cases"]):
        reduction = as_mapping(raw_case["reduction"], "case.reduction")
        cast(dict[str, object], reduction)["plan_sha256"] = (
            reference.reduction_plan_sha256(reduction)
        )
        raw_case["physics_case_sha256"] = reference.physics_case_sha256(
            fixture,
            str(raw_case["id"]),
        )
    return fixture

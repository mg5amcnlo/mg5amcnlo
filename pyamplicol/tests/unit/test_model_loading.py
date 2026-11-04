# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import pyamplicol
from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models._physics_ir import ContractionIR
from pyamplicol.models.contracts import (
    DEFAULT_FEYNMAN_PROPAGATOR_SOURCE,
    MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    CompiledClosureContractionRecord,
    CompiledCouplingRecord,
    CompiledDirectContractionRecord,
    CompiledModelIR,
    CompiledParticleRecord,
)
from pyamplicol.models.loading import (
    ModelCompileOptions,
    _classify_external_propagators,
    _load_external_model,
    _loader_restriction_name,
    _model_compiler_source_paths,
    _require_ufo_model_loader,
    _sanitized_model_environment,
    _source_digest,
)


class _SerializedModel:
    propagators: tuple[object, ...] = ()

    def wrap_indices_in_lorentz_structures(self) -> None:
        pass

    def to_json(self, _look: object) -> str:
        return '{"name":"synthetic","propagators":[]}'


@pytest.mark.parametrize(
    "source",
    (
        "x+1",
        "x^(1/2)",
        "sqrt(x)",
        "2*UFO::{}::aS^(1/2)*𝜋^(1/2)",
        "x^(1/3)",
    ),
)
def test_ufo_sqrt_normalization_is_one_shot_and_idempotent(source: str) -> None:
    from symbolica import Expression
    from ufo_model_loader import symbolica_processing

    expression = Expression.parse(source)
    expected = expression.to_canonical_string()

    normalized = symbolica_processing.replace_from_sqrt(expression)
    normalized_again = symbolica_processing.replace_from_sqrt(normalized)

    normalized_text = normalized.to_canonical_string()
    assert "sqrt(" not in normalized_text
    if source != "sqrt(x)":
        assert normalized_text == expected
    assert normalized_again.to_canonical_string() == normalized_text


def test_ufo_standard_cmath_normalization_needs_no_runtime_patch() -> None:
    from ufo_model_loader import model, symbolica_processing

    original_processing_parse = symbolica_processing.parse_python_expression_safe
    original_model_parse = model.parse_python_expression_safe

    with _sanitized_model_environment():
        assert model.parse_python_expression_safe is (
            symbolica_processing.parse_python_expression_safe
        )
        normalized = symbolica_processing.parse_python_expression_safe(
            "cmath.sin(x) + cmath.cos(x) + cmath.asin(y) + cmath.acos(y)"
        ).to_canonical_string()
        assert "cmath" not in normalized
        assert all(f"{name}(" in normalized for name in ("sin", "cos", "asin", "acos"))

    assert (
        symbolica_processing.parse_python_expression_safe is original_processing_parse
    )
    assert model.parse_python_expression_safe is original_model_parse

    try:
        with _sanitized_model_environment():
            raise RuntimeError("scope exit")
    except RuntimeError as error:
        assert str(error) == "scope exit"
    assert (
        symbolica_processing.parse_python_expression_safe is original_processing_parse
    )
    assert model.parse_python_expression_safe is original_model_parse


@pytest.mark.parametrize("version", ["1.0.0", "1.0.1", "1.1.0", "1.0.0+local"])
def test_ufo_loader_accepts_supported_releases(monkeypatch, version: str) -> None:
    monkeypatch.setattr(
        "pyamplicol.models.loading._distribution_version", lambda *_: version
    )
    _require_ufo_model_loader()


@pytest.mark.parametrize("version", ["not installed", "0.1.8", "0.2.0", "1.0.0rc1"])
def test_ufo_loader_rejects_unsupported_versions_before_loading(
    monkeypatch, version: str
) -> None:
    monkeypatch.setattr(
        "pyamplicol.models.loading._distribution_version", lambda *_: version
    )
    with pytest.raises(RuntimeError, match=r"1\.0\.0 or newer"):
        _load_external_model(Path("unused.json"), options=ModelCompileOptions())


def test_all_packaged_ufo_sm_parameter_expressions_normalize_once() -> None:
    from ufo_model_loader import symbolica_processing

    source = (
        Path(pyamplicol.__file__).parent
        / "assets"
        / "models"
        / "json"
        / "sm"
        / "sm.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    expressions = [
        parameter["expression"]
        for parameter in payload["parameters"]
        if parameter["expression"] is not None
    ]

    with _sanitized_model_environment():
        normalized = [
            symbolica_processing.parse_python_expression_safe(expression)
            for expression in expressions
        ]

    assert len(normalized) == len(expressions)
    assert all(expression.to_canonical_string() for expression in normalized)


def test_model_loading_sanitizes_historical_scalar_environment(
    monkeypatch,
) -> None:
    from ufo_model_loader import commands

    monkeypatch.setenv("UFO_SCALARS_MODEL_N_SCALARS", "91")
    monkeypatch.setenv("UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS", "3,4")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    seen: dict[str, str | bool | None] = {}

    def fake_load_model(*_args, **_kwargs):
        seen["scalars"] = os.environ.get("UFO_SCALARS_MODEL_N_SCALARS")
        seen["gravity"] = os.environ.get("UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS")
        seen["dont_write_bytecode"] = sys.dont_write_bytecode
        return _SerializedModel(), {"lam": 1.0 + 2.0j}

    monkeypatch.setattr(commands, "load_model", fake_load_model)
    payload = _load_external_model(
        Path("synthetic.json"),
        options=ModelCompileOptions(),
    )

    assert seen == {
        "scalars": None,
        "gravity": None,
        "dont_write_bytecode": True,
    }
    assert sys.dont_write_bytecode is previous_dont_write_bytecode
    assert os.environ["UFO_SCALARS_MODEL_N_SCALARS"] == "91"
    assert os.environ["UFO_GRAVITY_MODEL_N_POINT_INTERACTIONS"] == "3,4"
    assert payload["parameter_card"] == {"lam": [1.0, 2.0]}


def test_external_propagator_classification_uses_expressions_not_names() -> None:
    from ufo_model_loader.commands import load_model

    source = (
        Path(pyamplicol.__file__).parent
        / "assets"
        / "models"
        / "json"
        / "scalars"
        / "scalars.json"
    )
    with _sanitized_model_environment():
        model, _parameter_card = load_model(
            str(source),
            None,
            True,
            wrap_indices_in_lorentz_structures=False,
        )
    propagator = model.propagators[0]
    propagator.name = "renamed_without_a_default_suffix"

    assert _classify_external_propagators(model)[propagator.particle.name] == (
        DEFAULT_FEYNMAN_PROPAGATOR_SOURCE
    )

    propagator.name = "custom_propFeynman"
    propagator.numerator += 1

    assert _classify_external_propagators(model)[propagator.particle.name] == (
        MODEL_SUPPLIED_PROPAGATOR_SOURCE
    )


def test_restriction_does_not_reclassify_loader_default_propagators() -> None:
    from ufo_model_loader.commands import load_model

    source = (
        Path(pyamplicol.__file__).parent
        / "assets"
        / "models"
        / "json"
        / "sm"
        / "sm.json"
    )
    with _sanitized_model_environment():
        model, _parameter_card = load_model(
            str(source),
            None,
            True,
            wrap_indices_in_lorentz_structures=False,
        )

    assert set(_classify_external_propagators(model).values()) == {
        DEFAULT_FEYNMAN_PROPAGATOR_SOURCE
    }


def test_runtime_domain_symbols_do_not_redefine_symbolica_attributes() -> None:
    _sym._ensure_symbolica()
    complex_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "complex",
    )
    real_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "real",
    )
    imaginary_symbol = symbols.runtime_domain(
        "synthetic",
        "shared_parameter",
        "imaginary",
    )

    assert complex_symbol != real_symbol
    assert complex_symbol != imaginary_symbol
    assert real_symbol.is_real()
    assert (-1j * imaginary_symbol).is_real()


def test_model_symbol_registry_preserves_identity_and_is_collision_resistant() -> None:
    _sym._ensure_symbolica()
    model_symbols = symbols.model("sm")

    assert model_symbols.namespace == "model_sm"
    assert model_symbols.symbol("MZ") == model_symbols.expression("UFO::MZ")
    assert model_symbols.symbol("MZ") != model_symbols.symbol("mz")
    assert "UFO::" not in model_symbols.expression_string("UFO::MZ + UFO::aS")
    assert symbols.model("SM-v1").namespace != symbols.model("SM v1").namespace


def test_runtime_model_parameter_uses_the_declared_symbol_owner() -> None:
    _sym._ensure_symbolica()

    assert symbols.runtime_model_parameter("sm", "MZ") == symbols.model("sm").symbol(
        "MZ"
    )
    assert symbols.runtime_model_parameter(
        "sm", "derived_coupling_81"
    ) == symbols.derived_coupling("sm", 81)


def test_compiled_model_rejects_process_global_ufo_scalar_symbols() -> None:
    with pytest.raises(ValueError, match="process-global UFO symbol"):
        CompiledModelIR(
            name="synthetic",
            orders=(),
            parameters=(),
            particles=(),
            couplings=(
                CompiledCouplingRecord(
                    name="GC_1",
                    expression="UFO::aS",
                    resolved_expression="UFO::aS",
                    value=(1.0, 0.0),
                    orders=(),
                ),
            ),
            propagators=(),
            vertex_terms=(),
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
        )


def test_compiled_model_rejects_non_involutive_particle_identity() -> None:
    particle = CompiledParticleRecord(
        name="psi",
        antiname="psi_bar",
        pdg_code=700_001,
        spin=2,
        color=3,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )

    with pytest.raises(ValueError, match="absent antiparticle"):
        CompiledModelIR(
            name="invalid-particle-pair",
            orders=(),
            parameters=(),
            particles=(particle,),
            couplings=(),
            propagators=(),
            vertex_terms=(),
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
        )


def test_compiled_particle_role_metadata_round_trips() -> None:
    particle = CompiledParticleRecord(
        name="x",
        antiname="x",
        pdg_code=700_010,
        spin=3,
        color=8,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )
    model = CompiledModelIR(
        name="role-round-trip",
        orders=(),
        parameters=(),
        particles=(particle,),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )

    restored = CompiledModelIR.from_dict(model.to_dict()).particles[0]

    assert restored.statistics == "boson"
    assert restored.wavefunction_family == "vector"
    assert restored.color_role == "adjoint"
    assert restored.self_conjugate is True
    assert restored.source_orientation == "self-conjugate"


def _compiled_scalar_contraction_model() -> CompiledModelIR:
    particle = CompiledParticleRecord(
        name="phi",
        antiname="phi_bar",
        pdg_code=700_020,
        spin=1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=1.0,
        quantum_numbers=(("electric_charge", "1"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )
    antiparticle = CompiledParticleRecord(
        name="phi_bar",
        antiname="phi",
        pdg_code=-700_020,
        spin=1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=-1.0,
        quantum_numbers=(("electric_charge", "-1"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )
    scalar = ContractionIR(
        name="scalar",
        left_basis="scalar",
        right_basis="scalar",
        coefficients=((1.0, 0.0),),
    )
    return CompiledModelIR(
        name="contraction-round-trip",
        orders=(),
        parameters=(),
        particles=(particle, antiparticle),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(
            CompiledDirectContractionRecord(
                left_particle="phi",
                left_chirality=0,
                right_particle="phi_bar",
                right_chirality=0,
                contraction_ir=scalar,
            ),
        ),
        closure_contractions=(
            CompiledClosureContractionRecord(
                particle="phi",
                chirality=0,
                contraction_ir=scalar,
            ),
        ),
    )


def test_compiled_contraction_records_round_trip_strictly() -> None:
    model = _compiled_scalar_contraction_model()

    restored = CompiledModelIR.from_dict(model.to_dict())

    assert restored.direct_contractions == model.direct_contractions
    assert restored.closure_contractions == model.closure_contractions

    missing = model.to_dict()
    del missing["direct_contractions"]
    with pytest.raises(
        ValueError,
        match="missing required field 'direct_contractions'",
    ):
        CompiledModelIR.from_dict(missing)

    unknown = model.to_dict()
    direct = unknown["direct_contractions"]
    assert isinstance(direct, list)
    assert isinstance(direct[0], dict)
    direct[0]["inferred_from_dimension"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        CompiledModelIR.from_dict(unknown)


def test_compiled_contraction_records_reject_duplicate_and_missing_selectors() -> None:
    model = _compiled_scalar_contraction_model()
    duplicate = model.to_dict()
    direct = duplicate["direct_contractions"]
    assert isinstance(direct, list)
    direct.append(deepcopy(direct[0]))
    with pytest.raises(ValueError, match="duplicate direct contraction selector"):
        CompiledModelIR.from_dict(duplicate)

    missing_particle = model.to_dict()
    direct = missing_particle["direct_contractions"]
    assert isinstance(direct, list)
    assert isinstance(direct[0], dict)
    direct[0]["right_particle"] = "absent"
    with pytest.raises(ValueError, match="absent particle 'absent'"):
        CompiledModelIR.from_dict(missing_particle)


def test_compiled_contraction_records_reject_invalid_physics_contracts() -> None:
    model = _compiled_scalar_contraction_model()

    not_antiparticles = model.to_dict()
    direct = not_antiparticles["direct_contractions"]
    assert isinstance(direct, list)
    assert isinstance(direct[0], dict)
    direct[0]["right_particle"] = "phi"
    with pytest.raises(ValueError, match="are not an antiparticle pair"):
        CompiledModelIR.from_dict(not_antiparticles)

    wrong_dimension = model.to_dict()
    direct = wrong_dimension["direct_contractions"]
    assert isinstance(direct, list)
    assert isinstance(direct[0], dict)
    contraction = direct[0]["contraction_ir"]
    assert isinstance(contraction, dict)
    contraction["coefficients"] = [[1.0, 0.0], [1.0, 0.0]]
    with pytest.raises(
        ValueError,
        match="2 coefficients for current dimensions 1 and 1",
    ):
        CompiledModelIR.from_dict(wrong_dimension)

    wrong_chirality = model.to_dict()
    direct = wrong_chirality["direct_contractions"]
    assert isinstance(direct, list)
    assert isinstance(direct[0], dict)
    contraction = direct[0]["contraction_ir"]
    assert isinstance(contraction, dict)
    contraction["chirality_relation"] = "opposite"
    with pytest.raises(ValueError, match="violates opposite chirality relation"):
        CompiledModelIR.from_dict(wrong_chirality)

    non_scalar_closure = model.to_dict()
    closure = non_scalar_closure["closure_contractions"]
    assert isinstance(closure, list)
    assert isinstance(closure[0], dict)
    contraction = closure[0]["contraction_ir"]
    assert isinstance(contraction, dict)
    contraction["name"] = "not-a-scalar-projection"
    with pytest.raises(ValueError, match="one-component scalar projection"):
        CompiledModelIR.from_dict(non_scalar_closure)


def test_model_cache_inputs_are_content_based(tmp_path: Path) -> None:
    left = tmp_path / "left" / "model"
    right = tmp_path / "right" / "model"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    for root in (left, right):
        (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "particles.py").write_text("PARTICLES = []\n", encoding="utf-8")

    assert _source_digest("ufo", left) == _source_digest("ufo", right)

    first = tmp_path / "first restriction.dat"
    second = tmp_path / "second restriction.dat"
    first.write_text("BLOCK MASS\n", encoding="utf-8")
    second.write_text("BLOCK MASS\n", encoding="utf-8")
    first_payload = ModelCompileOptions(restriction=str(first)).canonical_payload()
    second_payload = ModelCompileOptions(restriction=str(second)).canonical_payload()
    assert first_payload == second_payload
    assert str(tmp_path) not in repr(first_payload)


def test_restriction_paths_are_adapted_to_loader_names(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    ufo_restriction = ufo / "restrict_no_widths.dat"
    ufo_restriction.write_text("BLOCK DECAY\n", encoding="ascii")
    serialized = tmp_path / "sm.json"
    serialized.write_text("{}", encoding="ascii")
    json_restriction = tmp_path / "restrict_no_widths.json"
    json_restriction.write_text("{}", encoding="ascii")

    assert _loader_restriction_name(ufo, "default") is None
    assert _loader_restriction_name(ufo, "none") == "full"
    assert _loader_restriction_name(ufo, str(ufo_restriction)) == "no_widths"
    assert _loader_restriction_name(serialized, str(json_restriction)) == "no_widths"


def test_restriction_paths_must_follow_loader_layout(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    misplaced = tmp_path / "restrict_default.dat"
    misplaced.write_text("BLOCK MASS\n", encoding="ascii")
    malformed = ufo / "default.dat"
    malformed.write_text("BLOCK MASS\n", encoding="ascii")

    with pytest.raises(ValueError, match="stored next to"):
        _loader_restriction_name(ufo, str(misplaced))
    with pytest.raises(ValueError, match="must be named"):
        _loader_restriction_name(ufo, str(malformed))


def test_model_compiler_fingerprint_covers_all_lowering_sources() -> None:
    package_root = Path(pyamplicol.__file__).resolve().parent
    source_paths = _model_compiler_source_paths()
    relative_paths = {
        path.relative_to(package_root).as_posix() for path in source_paths
    }

    model_sources = {
        f"models/{path.name}" for path in (package_root / "models").glob("*.py")
    }
    assert model_sources <= relative_paths
    assert {
        "_internal/physics/parameters.py",
        "_internal/physics/symbols.py",
        "_internal/physics/types.py",
        "processes/core_syntax.py",
    } <= relative_paths


def test_model_cache_identity_includes_symbolica_serialization_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pyamplicol.models import loading

    source = tmp_path / "model.json"
    source.write_text("{}", encoding="ascii")
    cache = tmp_path / "cache"
    _digest, first_fingerprint, first_path = loading._compilation_cache_identity(
        "json",
        source,
        options=ModelCompileOptions(),
        cache_dir=cache,
    )

    monkeypatch.setattr(
        loading,
        "SYMBOLICA_SERIALIZATION_ABI",
        "symbolica-test-serialization-v2",
    )
    _digest, second_fingerprint, second_path = loading._compilation_cache_identity(
        "json",
        source,
        options=ModelCompileOptions(),
        cache_dir=cache,
    )

    assert (
        first_fingerprint["symbolica_serialization_abi"]
        != (second_fingerprint["symbolica_serialization_abi"])
    )
    assert first_path != second_path

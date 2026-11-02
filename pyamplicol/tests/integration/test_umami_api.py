# SPDX-License-Identifier: 0BSD
"""Bounded native-provider checks; reuse the already staged Rusticol SDK."""

from __future__ import annotations

import ctypes as ct
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)


def _unavailable(reason):
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _config_executable():
    configured = os.environ.get("RUSTICOL_CONFIG")
    candidates = (
        configured,
        str(Path(sys.executable).parent / "rusticol-config"),
        shutil.which("rusticol-config"),
    )
    for candidate in candidates:
        if candidate and (shutil.which(candidate) or Path(candidate).is_file()):
            return candidate
    _unavailable(
        "UMAMI native integration requires the staged rusticol-config executable"
    )


def _library(path):
    library = ct.CDLL(str(path))
    library.umami_initialize.argtypes = (ct.POINTER(ct.c_void_p), ct.c_char_p)
    library.umami_free.argtypes = (ct.c_void_p,)
    library.umami_get_meta.argtypes = (ct.c_int, ct.c_void_p)
    for name in ("supported_inputs", "required_inputs", "supported_outputs"):
        getattr(library, "umami_" + name).argtypes = (
            ct.POINTER(ct.POINTER(ct.c_bool)),
            ct.POINTER(ct.c_int),
        )
    library.umami_get_parameter.argtypes = (
        ct.c_void_p,
        ct.c_char_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    )
    library.umami_set_parameter.argtypes = (
        ct.c_void_p,
        ct.c_char_p,
        ct.c_double,
        ct.c_double,
    )
    library.umami_matrix_element.argtypes = (
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_void_p),
        ct.c_size_t,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_void_p),
    )
    return library


@dataclass
class Provider:
    artifact: Path
    build: Path
    data: dict
    library: object
    structural_keys: list[dict]

    @contextmanager
    def handle(self):
        value = ct.c_void_p()
        assert (
            self.library.umami_initialize(ct.byref(value), os.fsencode(self.artifact))
            == 0
        )
        assert value.value is not None
        try:
            yield value
        finally:
            assert self.library.umami_free(value) == 0


@pytest.fixture(
    scope="module",
    params=(
        pytest.param(("lc", False), id="lc"),
        pytest.param(("lc5", False), id="lc-identical-ggg"),
        pytest.param(("lcq", False), id="lc-identical-quarks"),
        pytest.param(("full", False), id="full"),
        pytest.param(("nlc", False), id="nlc"),
        pytest.param(("lc", True), id="restricted-helicity"),
        pytest.param(("ufo", False), id="ufo-flavours"),
    ),
)
def providers(request, tmp_path_factory):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    config_executable = _config_executable()
    accuracy, restricted = request.param
    is_ufo = accuracy == "ufo"
    is_ggg = accuracy == "lc5"
    is_qqg = accuracy == "lcq"
    if is_ufo:
        accuracy = "full"
    elif is_ggg or is_qqg:
        accuracy = "lc"
    root = tmp_path_factory.mktemp("umami-" + accuracy)
    artifact, build = root / "artifact with spaces", root / "build with spaces"
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            selected_source_helicities={"1": -1, "2": 1, "3": -1, "4": 1}
            if restricted
            else {},
        ),
        color=ColorConfig(accuracy=accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=True,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    # The massive colour singlet forces a separate fixed-mass provider, while
    # retaining quark colour flow and generic UFO parameter declarations.
    include_quark = not (is_ggg or is_qqg) and (
        accuracy == "full" or (accuracy == "lc" and not restricted)
    )
    expressions = (
        ("g g > g g g",)
        if is_ggg
        else ("d d > d d g",)
        if is_qqg
        else ("g g > g g", "d d~ > z g")
        if include_quark
        else ("g g > g g",)
    )
    names = (
        ("ggg",)
        if is_ggg
        else ("dd_ddg",)
        if is_qqg
        else ("gg", "dd_zg")
        if include_quark
        else ("gg",)
    )
    from pyamplicol.generation import umami_semantics

    original_keys = umami_semantics.recurrence_structural_keys
    observed_keys = []

    def record_keys(**kwargs):
        keys = original_keys(**kwargs)
        observed_keys.append(keys)
        return keys

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(umami_semantics, "recurrence_structural_keys", record_keys)
        if is_ufo:
            from tests.integration.umami_fixtures import equivalent_scalar_model

            model, requests = equivalent_scalar_model(root / "ufo-source")
            prepared = model.compile(
                cache_dir=root / "model-cache",
                prepared_output=root / "equivalent-scalars.pyamplicol-model",
                evaluator=config.evaluator,
            )
            Generator(config).generate(
                requests,
                artifact,
                model=prepared,
            )
        else:
            with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
                Generator(config).generate(
                    ProcessSet.from_expressions(expressions, names=names),
                    artifact,
                    model=ModelSource.from_path(prepared_model),
                )
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(artifact / "API/umami"),
            f"BUILD_DIR={build}",
            f"RUSTICOL_CONFIG={config_executable}",
        ],
        env=dict(
            os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src")
        ),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root_metadata = json.loads((artifact / "API/umami/metadata.json").read_text())
    paths = sorted((artifact / "API/umami/providers").glob("*/metadata.json"))
    documents = (
        [json.loads(path.read_text()) for path in paths] if paths else [root_metadata]
    )
    assert len(documents) == (1 if is_ufo else len(expressions))
    if is_ufo:
        assert documents[0]["provider"]["supports_alpha_s"]
        assert all(
            process["alpha_s_parameter"] == "aS"
            for process in documents[0]["runtime_processes"]
        )
    if restricted:
        assert (
            documents[0]["runtime_processes"][0]["coverage"]["helicities"] == "selected"
        )
        assert len(documents[0]["runtime_processes"][0]["helicity_ids"]) == 1
    if is_ggg:
        data = documents[0]
        assert len(data["channels"]) == 4
        assert data["grouping"]["physical_contributions"] == 24
        assert [channel["phasespace_order"] for channel in data["channels"]] == [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 3, 4],
            [0, 2, 3, 1, 4],
            [0, 2, 3, 4, 1],
        ]
        for channel in data["channels"]:
            (entry,) = channel["processes"]
            assert len(entry["members"]) == 6
            assert entry["matrix_elements"] == [{"pdg_ids": 0, "factor": 6.0}]
    if is_qqg:
        grouping_metadata = documents[0]["grouping"]
        assert (
            grouping_metadata["exported_contributions"]
            < grouping_metadata["physical_contributions"]
        )
    return [
        Provider(
            artifact,
            build,
            data,
            _library(build / data["provider"]["library"]),
            observed_keys,
        )
        for data in documents
    ]


def _entry(provider, channel=0, flavour=0):
    return provider.data["channels"][channel]["processes"][flavour]["runtime"]


def _runtime_point(provider, process_id):
    lines = (provider.artifact / "API/validation_points.dat").read_text().splitlines()
    fields = next(line.split() for line in lines[1:] if line.split()[0] == process_id)
    components = tuple(float(value) for value in fields[2:])
    assert len(components) == 4 * int(fields[1])
    return tuple(
        components[index : index + 4] for index in range(0, len(components), 4)
    )


def _point(provider, entry):
    point = _runtime_point(provider, entry["process_id"])
    # The provider maps physical slots to representative runtime slots.
    physical = [None] * len(point)
    for runtime_slot, physical_slot in enumerate(entry["permutation"]):
        physical[physical_slot] = point[runtime_slot]
    return tuple(physical)


def _member_input(representative, member, point):
    # Compose the two physical-to-representative maps without changing the
    # physical labelling on which a consumer applies cuts or flavour weights.
    mapped = [None] * len(point)
    for j, slot in enumerate(representative["permutation"]):
        mapped[slot] = point[member["runtime"]["permutation"][j]]
    return tuple(mapped)


def _expected(provider, entry, point, parameters=None):
    runtime = Runtime.load(provider.artifact, process=entry["process_id"])
    if parameters:
        runtime.set_model_parameters(parameters)
    runtime_point = tuple(point[index] for index in entry["permutation"])
    selected = None if entry["color_id"] is None else (entry["color_id"],)
    return (
        float(complex(runtime.evaluate((runtime_point,), color_flows=selected)[0]).real)
        * entry["factor"]
    )


def _batch(
    provider,
    handle,
    points,
    *,
    channel=0,
    flavour=0,
    alpha_s=None,
    offset=0,
    count=None,
    helicity=None,
):
    stride, n = len(points), len(points[0])
    momenta = (ct.c_double * (4 * n * stride))(
        *(
            float(points[event][leg][component])
            for component in range(4)
            for leg in range(n)
            for event in range(stride)
        )
    )
    inputs = {
        0: momenta,
        2: (ct.c_int * stride)(*[flavour] * stride),
        4: (ct.c_double * stride)(*[0.5] * stride),
        8: (ct.c_uint * stride)(*[channel] * stride),
    }
    if alpha_s is not None:
        inputs[1] = (ct.c_double * stride)(*alpha_s)
    if helicity is not None:
        inputs[6] = (ct.c_int * stride)(*[helicity] * stride)
    outputs = {
        0: (ct.c_double * stride)(*[-123.0] * stride),
        3: (ct.c_int * stride)(*[-123] * stride),
    }
    if provider.data["provider"]["color_accuracy"] == "lc":
        outputs[2] = (ct.c_int * stride)(*[-123] * stride)
    status = provider.library.umami_matrix_element(
        handle,
        stride - offset if count is None else count,
        stride,
        offset,
        len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(
            *(ct.cast(value, ct.c_void_p) for value in inputs.values())
        ),
        len(outputs),
        (ct.c_int * len(outputs))(*outputs),
        (ct.c_void_p * len(outputs))(
            *(ct.cast(value, ct.c_void_p) for value in outputs.values())
        ),
    )
    return status, {key: list(value) for key, value in outputs.items()}


def _parameter(provider, handle, name):
    real, imaginary = ct.c_double(), ct.c_double()
    assert (
        provider.library.umami_get_parameter(
            handle, name.encode(), ct.byref(real), ct.byref(imaginary)
        )
        == 0
    )
    return complex(real.value, imaginary.value)


def test_metadata_capabilities_and_hidden_native_symbols(providers):
    for provider in providers:
        with pytest.raises(AttributeError):
            _ = provider.library.rusticol_runtime_load
        count = ct.c_int()
        assert provider.library.umami_get_meta(1, ct.byref(count)) == 0
        assert count.value == provider.data["provider"]["particle_count"]
        masses = (ct.c_double * count.value)()
        assert provider.library.umami_get_meta(5, masses) == 0
        assert list(masses) == provider.data["provider"]["masses"]
        assert provider.library.umami_get_meta(2, ct.byref(count)) == 6
        has_color = provider.data["provider"]["color_accuracy"] == "lc"
        assert provider.library.umami_get_meta(4, ct.byref(count)) == (
            0 if has_color else 6
        )
        if has_color and provider.data["runtime_processes"][0]["id"] == "dd_zg":
            for channel in provider.data["channels"]:
                for entry in channel["processes"]:
                    quark, antiquark, boson, gluon = provider.data["color_flows"][
                        entry["color_flows"]
                    ]
                    assert quark[1] == antiquark[0] == 0
                    assert boson == [0, 0]
                    assert quark[0] == gluon[0] > 0
                    assert antiquark[1] == gluon[1] > 0
                    assert gluon[0] != gluon[1]
        for operation in ("supported_inputs", "required_inputs", "supported_outputs"):
            values, length = ct.POINTER(ct.c_bool)(), ct.c_int()
            assert (
                getattr(provider.library, "umami_" + operation)(
                    ct.byref(values), ct.byref(length)
                )
                == 0
            )
            flags = list(values[: length.value])
            if operation == "required_inputs":
                assert flags == [True] + [False] * (len(flags) - 1)
            elif operation == "supported_inputs":
                assert flags[1] == provider.data["provider"]["supports_alpha_s"]
                assert flags[6] and not flags[3] and not flags[5] and not flags[7]
                assert not any(flags[9:])
            else:
                assert flags == [True, False, has_color, True] + [False] * (
                    len(flags) - 4
                )


def test_all_exported_flavours_match_native_and_preserve_stride(providers):
    for provider in providers:
        with provider.handle() as handle:
            for channel, description in enumerate(provider.data["channels"]):
                for flavour in range(len(description["processes"])):
                    entry = _entry(provider, channel, flavour)
                    point = _point(provider, entry)
                    expected = _expected(provider, entry, point)
                    status, outputs = _batch(
                        provider,
                        handle,
                        (point,) * 3,
                        channel=channel,
                        flavour=flavour,
                        offset=1,
                        count=1,
                    )
                    assert status == 0
                    assert outputs[0] == pytest.approx(
                        [-123.0, expected, -123.0], rel=2e-11, abs=1e-15
                    )
                    if 2 in outputs:
                        assert outputs[2] == [-123, entry["color_index"], -123]
                    assert outputs[3][::2] == [-123, -123]
                    assert 0 <= outputs[3][1] < len(provider.data["helicities"])


def test_grouped_members_match_physical_flows_at_asymmetric_point(providers):
    for provider in providers:
        reconstructed, complete = {}, {}
        cut_native, cut_umami = 0.0, 0.0
        cut_decisions = set()
        with provider.handle() as handle:
            for channel, description in enumerate(provider.data["channels"]):
                for flavour, exported in enumerate(description["processes"]):
                    representative = exported["runtime"]
                    for member in exported["members"]:
                        runtime = Runtime.load(
                            provider.artifact, process=member["process_id"]
                        )
                        point = (
                            (
                                (500, 0, 0, 500),
                                (500, 0, 0, -500),
                                (500, 300, 0, 400),
                                (500, -300, 0, -400),
                            )
                            if member["process_id"] == "gg"
                            else _runtime_point(provider, member["process_id"])
                        )
                        selection = (
                            None
                            if member["color_id"] is None
                            else (member["color_id"],)
                        )
                        expected = float(
                            complex(
                                runtime.evaluate((point,), color_flows=selection)[0]
                            ).real
                        )
                        process_id = member["process_id"]
                        if process_id not in complete:
                            complete[process_id] = float(
                                complex(runtime.evaluate((point,))[0]).real
                            )
                        # Both maps name the same representative runtime slots.
                        # Test each member at its own physical momenta; do not
                        # confuse integrated multiplicity with pointwise sum.
                        mapped = _member_input(representative, member, point)
                        status, outputs = _batch(
                            provider,
                            handle,
                            (mapped,),
                            channel=channel,
                            flavour=flavour,
                        )
                        assert status == 0
                        assert outputs[0][0] == pytest.approx(
                            expected, rel=2e-11, abs=1e-15
                        )
                        reconstructed[process_id] = (
                            reconstructed.get(process_id, 0.0)
                            + outputs[0][0] * member["factor"]
                        )
                        if process_id == "gg":
                            # An explicitly labelled-leg cut is not invariant
                            # under exchanging the two final-state gluons.
                            samples = (point, (*point[:2], point[3], point[2]))
                        elif process_id in ("flavour_one", "flavour_two"):
                            samples = (point,)
                        else:
                            continue
                        for sample in samples:
                            accepted = (
                                sample[2][3] > 0
                                if process_id == "gg"
                                else member["pdgs"][2] == 1001
                            )
                            cut_decisions.add(accepted)
                            status, selected = _batch(
                                provider,
                                handle,
                                (_member_input(representative, member, sample),),
                                channel=channel,
                                flavour=flavour,
                            )
                            assert status == 0
                            physical = float(
                                complex(
                                    runtime.evaluate((sample,), color_flows=selection)[
                                        0
                                    ]
                                ).real
                            )
                            weight = member["factor"] if accepted else 0.0
                            cut_native += weight * physical
                            cut_umami += weight * selected[0][0]
        # The member loop restores each labelled physical flow, including
        # cross-flavour compaction; an orbit factor is not a pointwise sum.
        assert reconstructed == pytest.approx(complete, rel=2e-11, abs=1e-15)
        if cut_decisions:
            assert cut_decisions == {False, True}
            assert cut_native > 0
            assert cut_umami == pytest.approx(cut_native, rel=2e-11, abs=1e-15)


def test_parameters_event_override_and_independent_concurrent_handles(providers):
    for provider in providers:
        entry = _entry(provider)
        description = provider.data["runtime_processes"][entry["process_index"]]
        supports_alpha_s = provider.data["provider"]["supports_alpha_s"]
        name = (
            description["alpha_s_parameter"]
            if supports_alpha_s
            else next(
                parameter["name"]
                for parameter in description["parameters"]
                if parameter["mutable"]
                and parameter["default_real"] > 0
                and not parameter.get("is_complex", False)
            )
        )
        point = _point(provider, entry)
        with provider.handle() as first, provider.handle() as second:
            original = _parameter(provider, first, name)
            assert original.imag == 0 and original.real > 0
            changed = original.real * 1.2
            assert (
                provider.library.umami_set_parameter(first, name.encode(), changed, 0.0)
                == 0
            )
            assert _parameter(provider, first, name) == changed
            assert _parameter(provider, second, name) == original
            if entry["process_id"] == "flavour_one":
                assert _parameter(provider, first, "G") == pytest.approx(
                    math.sqrt(4 * math.pi * changed)
                )
                # Recheck both physical UFO flavours after mutation, including
                # the one whose computation is provided by its representative.
                checked = set()
                for channel, description in enumerate(provider.data["channels"]):
                    for flavour, exported in enumerate(description["processes"]):
                        for member in exported["members"]:
                            physical = _runtime_point(provider, member["process_id"])
                            native = Runtime.load(
                                provider.artifact, process=member["process_id"]
                            )
                            native.set_model_parameters({name: changed})
                            status, result = _batch(
                                provider,
                                first,
                                (_member_input(exported["runtime"], member, physical),),
                                channel=channel,
                                flavour=flavour,
                            )
                            assert status == 0
                            assert result[0][0] == pytest.approx(
                                float(complex(native.evaluate((physical,))[0]).real),
                                rel=2e-11,
                            )
                            checked.add(member["process_id"])
                assert checked == {"flavour_one", "flavour_two"}
            event_values = (original.real * 0.8, original.real * 1.1)
            status, outputs = _batch(
                provider, first, (point,) * 2, alpha_s=event_values
            )
            if supports_alpha_s:
                assert status == 0
                expected = [
                    _expected(provider, entry, point, {name: value})
                    for value in event_values
                ]
                assert outputs[0] == pytest.approx(expected, rel=2e-11)
                if entry["process_id"] == "flavour_one":
                    assert outputs[0][0] / outputs[0][1] == pytest.approx(
                        (event_values[0] / event_values[1]) ** 2, rel=2e-11
                    )
            else:
                assert status == 3  # No generic alpha_s name may be invented.
            assert _parameter(provider, first, name) == changed
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_batch, provider, handle, (point,))
                    for handle in (first, second)
                ]
                values = [future.result() for future in futures]
            for (status, outputs), value in zip(
                values, (changed, original.real), strict=True
            ):
                assert status == 0
                assert outputs[0][0] == pytest.approx(
                    _expected(provider, entry, point, {name: value}), rel=2e-11
                )
            assert (
                provider.library.umami_set_parameter(first, b"does-not-exist", 1.0, 0.0)
                == 1
            )


def test_fixed_helicities_and_sampled_helicity(providers):
    for provider in providers:
        with provider.handle() as handle:
            entry = _entry(provider)
            exported = provider.data["channels"][0]["processes"][0]["runtime"]
            point = _point(provider, entry)
            status, outputs = _batch(provider, handle, (point,))
            assert status == 0
            total, label = outputs[0][0], outputs[3][0]
            values = {}
            for index in exported["helicity_export"]:
                status, fixed = _batch(provider, handle, (point,), helicity=index)
                assert status == 0 and fixed[3] == [index]
                values[index] = fixed[0][0]
            assert sum(values.values()) == pytest.approx(total, rel=1e-12)
            # the label of the random number 0.5 in the runtime helicity order
            cumulative, expected = 0.0, None
            for index in exported["helicity_export"]:
                if values[index] <= 0.0:
                    continue
                expected, cumulative = index, cumulative + values[index]
                if cumulative > 0.5 * total:
                    break
            assert label == expected


def test_error_contract_and_zero_count(providers):
    for provider in providers:
        invalid = ct.c_void_p(123)
        assert (
            provider.library.umami_initialize(ct.byref(invalid), b"relative/artifact")
            == 1
        )
        assert invalid.value is None
        with provider.handle() as handle:
            evaluate = provider.library.umami_matrix_element
            assert evaluate(handle, 0, 0, 0, 0, None, None, 0, None, None) == 0
            assert evaluate(handle, 1, 1, 0, 0, None, None, 0, None, None) == 7
            assert evaluate(handle, 1, 1, 1, 0, None, None, 0, None, None) == 1
            selector = ct.c_int(0)
            pointer = (ct.c_void_p * 1)(ct.cast(ct.byref(selector), ct.c_void_p))
            assert (
                evaluate(handle, 0, 0, 0, 1, (ct.c_int * 1)(7), pointer, 0, None, None)
                == 3
            )
            assert (
                evaluate(handle, 0, 0, 0, 0, None, None, 1, (ct.c_int * 1)(1), pointer)
                == 4
            )
            point = _point(provider, _entry(provider))
            assert _batch(provider, handle, (point,), channel=999)[0] == 1
            assert _batch(provider, handle, (point,), flavour=-1)[0] == 1


def test_rejects_other_valid_artifact_with_identical_external_labels(
    providers, tmp_path
):
    provider = providers[0]
    if not (
        provider.data["provider"]["color_accuracy"] == "lc"
        and provider.data["provider"]["helicity_count"] > 1
        and provider.data["runtime_processes"][0]["id"] == "gg"
    ):
        return
    # A genuinely generated artifact with a different native layout, not a
    # damaged manifest or a change of SDK metadata alone. Particle/helicity
    # labels and colour accuracy by themselves cannot protect compiled tables.
    other = tmp_path / "different native artifact"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="all-flow-union"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as model:
        Generator(config).generate(
            ProcessSet.from_expressions(("g g > g g",), names=("gg",)),
            other,
            model=ModelSource.from_path(model),
        )
    manifest = json.loads((other / "artifact.json").read_text())
    assert manifest["artifact_id"] != provider.data["provider"]["artifact_id"]
    original = Runtime.load(provider.artifact, process="gg")
    alternate = Runtime.load(other, process="gg")
    assert alternate.physics.helicity_ids == original.physics.helicity_ids
    point = _runtime_point(provider, "gg")
    assert alternate.evaluate((point,)) == pytest.approx(original.evaluate((point,)))
    invalid = ct.c_void_p(123)
    assert provider.library.umami_initialize(ct.byref(invalid), os.fsencode(other)) == 1
    assert invalid.value is None


def _assert_relative_make_run(provider, *, makefile=None):
    sdk = provider.artifact / "API/umami"
    command = ["make", "-C", str(sdk)]
    if makefile is not None:
        command.extend(("-f", str(makefile)))
    command.extend(
        (
            "run",
            f"BUILD_DIR={os.path.relpath(provider.build, sdk)}",
            f"RUSTICOL_CONFIG={_config_executable()}",
        )
    )
    completed = subprocess.run(
        command,
        env=dict(
            os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src")
        ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(
        next(line for line in completed.stdout.splitlines() if line.startswith("{"))
    )
    entry = _entry(provider)
    assert result["matrix_element"] == pytest.approx(
        _expected(provider, entry, _point(provider, entry)), rel=2e-11
    )


def test_linked_driver_survives_relocation(providers, tmp_path):
    provider = providers[0]
    _assert_relative_make_run(provider)
    relocated = tmp_path / "relocated files with spaces"
    artifact, build = relocated / "artifact", relocated / "build"
    shutil.copytree(provider.artifact, artifact)
    build.mkdir(parents=True)
    library_name = provider.data["provider"]["library"]
    driver_name = (
        "umami_driver"
        if library_name == "libumami.so"
        else "umami_driver_" + provider.data["provider"]["id"]
    )
    shutil.copy2(provider.build / library_name, build / library_name)
    shutil.copy2(provider.build / driver_name, build / driver_name)
    completed = subprocess.run(
        [str(build / driver_name), str(artifact)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    entry = _entry(provider)
    assert result["matrix_element"] == pytest.approx(
        _expected(provider, entry, _point(provider, entry)), rel=2e-11
    )


def test_real_generation_exercises_structural_proofs(providers):
    for provider in providers:
        if (
            provider.data["provider"]["color_accuracy"] == "lc"
            and provider.data["provider"]["helicity_count"] > 1
        ):
            assert provider.structural_keys and any(provider.structural_keys), (
                "ordinary gg must exercise structural proofs"
            )
        if provider.data["runtime_processes"][0]["id"] == "flavour_one":
            assert len(provider.structural_keys) == 2
            assert provider.structural_keys[0]
            assert provider.structural_keys[0] == provider.structural_keys[1]
            entries = [
                entry
                for channel in provider.data["channels"]
                for entry in channel["processes"]
            ]
            assert len(entries) == 1
            assert all(
                entry["runtime"]["process_id"] == "flavour_one" for entry in entries
            )
            members = [member for entry in entries for member in entry["members"]]
            assert members[0]["pdgs"] != members[1]["pdgs"]


@pytest.fixture(scope="module")
def flavor_blind(tmp_path_factory):
    """``p p > j j`` with --flavor-blind and, as reference, every flavour.

    The built-in expansion has no ``g q > q g``: the explicit ones add members
    whose relation exchanges the incoming legs.
    """
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    config_executable = _config_executable()
    root = tmp_path_factory.mktemp("umami-flavor-blind")
    artifacts = {}
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for blind in (True, False):
            config = RunConfig(
                action="generate",
                process=ProcessConfig(flavor_scheme=2, flavor_blind=blind),
                color=ColorConfig(accuracy="lc"),
                generation=GenerationConfig(
                    workers=1,
                    emit_api_bundle=True,
                    relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
                    validation=GenerationValidationConfig(
                        enabled=False, post_build_validation=False
                    ),
                ),
                evaluator=EvaluatorConfig(
                    execution_mode="recurrence",
                    optimization=EvaluatorOptimizationConfig(cores=1),
                    jit=JITConfig(optimization_level=2),
                ),
            )
            artifacts[blind] = root / ("blind" if blind else "reference")
            Generator(config).generate(
                ProcessSet.from_expressions(("p p > j j", "g u > u g", "g u~ > u~ g")),
                artifacts[blind],
                model=ModelSource.from_path(prepared_model),
            )
    build = root / "build"
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(artifacts[True] / "API/umami"),
            f"BUILD_DIR={build}",
            f"RUSTICOL_CONFIG={config_executable}",
        ],
        env=dict(
            os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src")
        ),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return artifacts[True], artifacts[False], build


def _canonical_tags(flow):
    names = {}
    return tuple(
        tuple(names.setdefault(tag, len(names) + 1) if tag else 0 for tag in pair)
        for pair in flow
    )


def test_flavor_blind_members_match_separately_generated_processes(flavor_blind):
    from pyamplicol.artifacts.umami import FLAVOR_BLIND_ASSUMPTION
    from pyamplicol.artifacts.umami_reweight import load_umami_metadata

    blind, reference, build = flavor_blind
    groups = json.loads((blind / "process-groups.json").read_text())
    assumed = {
        member["process_id"]
        for group in groups["groups"]
        for member in group["members"]
        if not member["generated"]
    }
    assert assumed
    members = [member for group in groups["groups"] for member in group["members"]]
    assert any(member["conjugated"] for member in members)
    assert any(member["permutation"][:2] == [1, 0] for member in members)
    # The same physical flow of a separately generated process, by its LHE tags.
    reference_colors = {
        (
            member["process_id"],
            _canonical_tags(data["color_flows"][member["color_flows"]]),
        ): member["color_id"]
        for data in load_umami_metadata(reference)
        for channel in data["channels"]
        for entry in channel["processes"]
        for member in entry["members"]
    }
    # An asymmetric massless point in each member's physical leg order.
    sine, cosine = math.sqrt(1 - 0.31**2), 0.31
    direction = (sine * math.cos(0.7), sine * math.sin(0.7), cosine)
    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (500.0, *(500.0 * v for v in direction)),
        (500.0, *(-500.0 * v for v in direction)),
    )
    natives, checked = {}, []
    for data in load_umami_metadata(blind):
        provider = Provider(
            blind, build, data, _library(build / data["provider"]["library"]), []
        )
        generated = {process["id"] for process in data["runtime_processes"]}
        with provider.handle() as handle:
            for channel, description in enumerate(data["channels"]):
                for flavour, exported in enumerate(description["processes"]):
                    for member in exported["members"]:
                        process_id = member["process_id"]
                        if process_id in generated:
                            continue
                        assert member["runtime"]["process_id"] != process_id
                        assert (
                            FLAVOR_BLIND_ASSUMPTION in data["grouping"]["assumptions"]
                        )
                        color = reference_colors[
                            (
                                process_id,
                                _canonical_tags(
                                    data["color_flows"][member["color_flows"]]
                                ),
                            )
                        ]
                        if process_id not in natives:
                            natives[process_id] = Runtime.load(
                                reference, process=process_id
                            )
                        expected = float(
                            complex(
                                natives[process_id].evaluate(
                                    (point,), color_flows=(color,)
                                )[0]
                            ).real
                        )
                        status, outputs = _batch(
                            provider,
                            handle,
                            (_member_input(exported["runtime"], member, point),),
                            channel=channel,
                            flavour=flavour,
                        )
                        assert status == 0
                        assert outputs[0][0] == pytest.approx(
                            expected, rel=1e-12, abs=1e-15
                        )
                        checked.append((process_id, color))
    # Every physical flow of every assumed member is exported exactly once.
    assert sorted(checked) == sorted(
        (process_id, color)
        for (process_id, _), color in reference_colors.items()
        if process_id in assumed
    )

# SPDX-License-Identifier: 0BSD
"""Small physics and public-SDK parity checks for both FFT colour bases."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
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
from pyamplicol.models.builtin.validation import generic_validation_point
from tests.integration.test_on_the_fly_runtime_capabilities import (
    _require_native_on_the_fly,
)
from tests.integration.test_runtime_selector_api_parity import (
    _compile_probe,
    _NativeSdk,
    _source_environment,
)
from tests.integration.test_runtime_selector_api_parity import (
    native_sdk as _native_sdk_fixture,
)

native_sdk = _native_sdk_fixture

_LANES = ("recurrence", "on-the-fly")
_BASES = ("trace", "adjoint")
_FOUR_GLUON_POINTS = (
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 300, 0, 400), (500, -300, 0, -400)),
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 400, 300, 0), (500, -400, -300, 0)),
)


@pytest.fixture(scope="module")
def prepared_sm() -> Iterator[ModelSource]:
    _require_native_on_the_fly()
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as path:
        yield ModelSource.from_path(path)


def _config(
    lane: str,
    basis: str,
    *,
    accuracy: str = "full",
    selected_helicities: tuple[int, ...] = (),
    reference_color_order: tuple[int, ...] | None = None,
) -> RunConfig:
    return RunConfig(
        action="generate",
        process=ProcessConfig(
            reference_color_order=reference_color_order or (),
            selected_source_helicities={
                str(index): value
                for index, value in enumerate(selected_helicities, start=1)
            },
        ),
        color=ColorConfig(
            accuracy=accuracy,
            contraction="symmetric-group-fft",
            fft_basis=basis,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=lane,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _assert_fft_degree(artifact: Path, expected: int) -> None:
    execution_paths = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_paths) == 1
    execution = json.loads(execution_paths[0].read_text(encoding="utf-8"))
    color = execution["runtime_metadata"]["color_contraction"]
    assert color["storage"] == "convolution-kernels"
    assert color["factorization"]["kind"] == "symmetric-group-fourier"
    assert color["factorization"]["rank"] == expected
    assert color["fft_provenance"]["degree"] == expected


@pytest.mark.parametrize("lane", _LANES)
def test_automatic_fft_default_matches_forced_fft_and_direct(
    tmp_path: Path, prepared_sm: ModelSource, lane: str
) -> None:
    config = _config(
        lane,
        "adjoint",
        selected_helicities=(-1, 1, -1, 1) if lane == "recurrence" else (),
    )
    values = {}
    for method in ("auto", "symmetric-group-fft", "direct"):
        artifact = tmp_path / method
        Generator(
            replace(config, color=ColorConfig(accuracy="full", contraction=method))
        ).generate("g g > g g", artifact, model=prepared_sm)
        if method != "direct":
            _assert_fft_degree(artifact, 2)
        runtime = Runtime.load(artifact)
        values[method] = tuple(
            complex(value) for value in runtime.evaluate(_FOUR_GLUON_POINTS)
        )
        del runtime
    assert any(abs(value) > 0 for value in values["auto"])
    assert values["auto"] == pytest.approx(values["symmetric-group-fft"], rel=2e-12)
    assert values["auto"] == pytest.approx(values["direct"], rel=2e-11)


@pytest.fixture(scope="module", params=_LANES)
def four_gluon_artifacts(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    prepared_sm: ModelSource,
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp(f"adjoint-fft-{request.param}")
    artifacts = {}
    for basis in _BASES:
        artifact = root / basis
        Generator(
            _config(
                request.param,
                basis,
                reference_color_order=(3, 2, 4, 1) if basis == "adjoint" else None,
            )
        ).generate("g g > g g", artifact, model=prepared_sm)
        _assert_fft_degree(artifact, 3 if basis == "trace" else 2)
        artifacts[basis] = artifact
    return artifacts


def test_adjoint_fft_matches_trace_for_each_helicity_and_the_total(
    four_gluon_artifacts: dict[str, Path],
    request: pytest.FixtureRequest,
) -> None:
    runtimes = {
        basis: Runtime.load(path) for basis, path in four_gluon_artifacts.items()
    }
    try:
        resolved = {
            basis: runtime.evaluate_resolved(_FOUR_GLUON_POINTS)
            for basis, runtime in runtimes.items()
        }
        assert resolved["adjoint"].helicity_ids == resolved["trace"].helicity_ids
        assert len(resolved["trace"].helicity_ids) == 16
        flattened = {
            basis: tuple(
                complex(value)
                for point in result.values
                for helicity in point
                for value in helicity
            )
            for basis, result in resolved.items()
        }
        assert flattened["adjoint"] == pytest.approx(
            flattened["trace"], rel=2e-11, abs=1e-18
        )
        assert any(abs(value) > 0 for value in flattened["trace"])
        totals = {
            basis: tuple(
                complex(value) for value in runtime.evaluate(_FOUR_GLUON_POINTS)
            )
            for basis, runtime in runtimes.items()
        }
        assert totals["adjoint"] == pytest.approx(totals["trace"], rel=2e-11)
        for basis, result in resolved.items():
            assert tuple(complex(value) for value in result.total()) == pytest.approx(
                totals[basis], rel=2e-12
            )
        if request.node.callspec.params["four_gluon_artifacts"] == "recurrence":
            exact = {
                basis: tuple(
                    complex(value)
                    for value in runtime.evaluate(_FOUR_GLUON_POINTS, precision=32)
                )
                for basis, runtime in runtimes.items()
            }
            assert exact["adjoint"] == pytest.approx(exact["trace"], rel=2e-12)

        # Coupling normalization is shared by the two bases and remains mutable.
        for runtime in runtimes.values():
            runtime.set_model_parameters({"normalization.alpha_s_me_check": 0.13})
        assert runtimes["adjoint"].evaluate(_FOUR_GLUON_POINTS) == pytest.approx(
            runtimes["trace"].evaluate(_FOUR_GLUON_POINTS), rel=2e-11
        )
    finally:
        runtimes.clear()


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize(
    "process,accuracy,helicities",
    (
        ("g g > g g g", "full", (-1, 1, -1, 1, -1)),
        ("g g > g g g g", "nlc", (-1, 1, -1, 1, -1, 1)),
    ),
)
def test_adjoint_fft_matches_trace_for_selected_helicity_and_nlc(
    tmp_path: Path,
    prepared_sm: ModelSource,
    lane: str,
    process: str,
    accuracy: str,
    helicities: tuple[int, ...],
) -> None:
    points = tuple(
        tuple(
            tuple(float(component) for component in leg.momentum)
            for leg in generic_validation_point(process, seed=seed)
        )
        for seed in (101, 203)
    )
    helicity_id = "h:" + ",".join(f"{value:+d}" for value in helicities)
    values = {}
    for basis in _BASES:
        artifact = tmp_path / basis
        # Recurrence specializes source states at generation; OTF keeps its
        # complete source domain and selects exactly the same states at runtime.
        Generator(
            _config(
                lane,
                basis,
                accuracy=accuracy,
                selected_helicities=helicities if lane == "recurrence" else (),
            )
        ).generate(process, artifact, model=prepared_sm)
        _assert_fft_degree(artifact, len(helicities) - (1 if basis == "trace" else 2))
        runtime = Runtime.load(artifact)
        try:
            values[basis] = tuple(
                complex(value)
                for value in runtime.evaluate(points, helicities=(helicity_id,))
            )
        finally:
            del runtime
    assert any(abs(value) > 0 for value in values["trace"])
    assert values["adjoint"] == pytest.approx(values["trace"], rel=2e-10, abs=1e-18)


_SDK_PROBES = {
    "c": r"""
        #include <rusticol.h>
        #include <stdio.h>
        #include <stdlib.h>
        static void check(int status) {
            if (status) {
                char error[2048]; size_t needed;
                rusticol_last_error_message(error, sizeof error, &needed);
                fprintf(stderr, "%s\n", error); exit(1);
            }
        }
        int main(int argc, char **argv) {
            if (argc != 34) return 2;
            double momenta[32], values[2];
            for (size_t i=0; i<32; ++i) momenta[i]=strtod(argv[i+2], NULL);
            RusticolRuntimeHandle *runtime=NULL;
            check(rusticol_runtime_load(argv[1], NULL, NULL, &runtime));
            check(rusticol_runtime_evaluate_f64(runtime, momenta, 32, 2, values, 2));
            for (size_t i=0; i<2; ++i) printf("VALUE %zu %.17g\n", i, values[i]);
            check(rusticol_runtime_free(runtime)); return 0;
        }
    """,
    "cpp": r"""
        #include <rusticol.hpp>
        #include <cstdlib>
        #include <iomanip>
        #include <iostream>
        int main(int argc, char **argv) {
            if (argc != 34) return 2;
            std::vector<double> momenta;
            for (int i=2; i<argc; ++i) momenta.push_back(std::strtod(argv[i], nullptr));
            rusticol::Runtime runtime(argv[1]);
            auto values=runtime.evaluate(momenta, 2);
            for (std::size_t i=0; i<values.size(); ++i)
                std::cout << "VALUE " << i << " "
                          << std::setprecision(17) << values[i] << "\n";
        }
    """,
    "fortran": r"""
        program adjoint_fft_parity
          use, intrinsic :: iso_c_binding
          use rusticol
          implicit none
          type(rusticol_runtime) :: runtime
          character(len=4096) :: artifact, arg
          real(c_double) :: momenta(32)
          real(c_double), allocatable :: values(:)
          integer :: i
          if (command_argument_count() /= 33) stop 2
          call get_command_argument(1, artifact)
          do i=1,32
            call get_command_argument(i+1, arg)
            read(arg, *) momenta(i)
          end do
          call runtime%load(trim(artifact))
          call runtime%evaluate(momenta, 2_c_size_t, values)
          do i=1,2
            write(*,'(A,I0,1X,ES25.17E3)') 'VALUE ', i-1, values(i)
          end do
          call runtime%close()
        end program adjoint_fft_parity
    """,
    "rust": r"""
        #[allow(dead_code)]
        mod rusticol { include!(env!("RUSTICOL_RUST_SOURCE")); }
        use rusticol::Runtime;
        fn main() -> Result<(), Box<dyn std::error::Error>> {
            let args: Vec<_> = std::env::args().collect();
            if args.len() != 34 { return Err("wrong argument count".into()); }
            let momenta: Vec<f64> = args[2..].iter()
                .map(|s| s.parse()).collect::<Result<_,_>>()?;
            let mut runtime = Runtime::load(&args[1], None, None)?;
            for (i, value) in runtime.evaluate_f64(&momenta, 2)?.iter().enumerate() {
                println!("VALUE {i} {value:.17e}");
            }
            Ok(())
        }
    """,
}


@pytest.fixture(scope="module", params=tuple(_SDK_PROBES))
def sdk_driver(
    request: pytest.FixtureRequest,
    native_sdk: _NativeSdk,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return _compile_probe(
        request.param,
        _SDK_PROBES[request.param],
        native_sdk,
        tmp_path_factory.mktemp(f"adjoint-fft-sdk-{request.param}"),
    )


def test_adjoint_fft_artifacts_use_the_same_native_sdk_evaluation_api(
    sdk_driver: Path,
    four_gluon_artifacts: dict[str, Path],
) -> None:
    runtime = Runtime.load(four_gluon_artifacts["trace"])
    try:
        expected = tuple(
            float(complex(value).real) for value in runtime.evaluate(_FOUR_GLUON_POINTS)
        )
    finally:
        del runtime
    for artifact in four_gluon_artifacts.values():
        result = subprocess.run(
            [
                str(sdk_driver),
                str(artifact),
                *(
                    str(value)
                    for point in _FOUR_GLUON_POINTS
                    for leg in point
                    for value in leg
                ),
            ],
            env=_source_environment(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        measured = {
            int(parts[1]): float(parts[2])
            for line in result.stdout.splitlines()
            if (parts := line.split()) and parts[0] == "VALUE"
        }
        assert set(measured) == {0, 1}
        assert tuple(measured[index] for index in range(2)) == pytest.approx(
            expected, rel=2e-11
        )

# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501 -- native probes have their own source formatting.
"""Completed OTF caches round-trip through every SDK without rebuilding queries."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessRequest, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)
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

_POINTS = (
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 300, 0, 400), (500, -300, 0, -400)),
    ((500, 0, 0, 500), (500, 0, 0, -500), (500, 400, 300, 0), (500, -400, -300, 0)),
)


@pytest.fixture(scope="module", params=("lc", "full"))
def otf_output(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
):
    _require_native_on_the_fly()
    artifact = tmp_path_factory.mktemp(f"otf-cache-{request.param}") / "process"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=request.param),
        generation=GenerationConfig(
            workers=1,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="on-the-fly",
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            ProcessRequest.parse("g g > g g", name="gg_gg"),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )
    expected = Runtime.load(artifact).evaluate(_POINTS)
    return artifact, tuple(float(complex(value).real) for value in expected)


_PROBES = {
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
            if (argc != 35) return 2;
            double momenta[32], before, after[2];
            for (size_t i=0; i<32; ++i) momenta[i]=strtod(argv[i+3], NULL);
            RusticolRuntimeHandle *runtime=NULL;
            check(rusticol_runtime_load(argv[1], NULL, NULL, &runtime));
            const char invalid_utf8[]={(char)0xff, 0};
            if (rusticol_runtime_save(runtime, NULL) != RUSTICOL_STATUS_INVALID_ARGUMENT ||
                rusticol_runtime_load_cache(runtime, NULL) != RUSTICOL_STATUS_INVALID_ARGUMENT ||
                rusticol_runtime_save(runtime, invalid_utf8) != RUSTICOL_STATUS_INVALID_ARGUMENT ||
                rusticol_runtime_load_cache(runtime, invalid_utf8) != RUSTICOL_STATUS_INVALID_ARGUMENT) return 5;
            RusticolWarmUpResult cold={0};
            check(rusticol_runtime_warm_up_f64_with_cores(runtime, momenta, 16, NULL, 0, NULL, 0, 2, NULL, NULL, &cold));
            if (!cold.first_evaluation_completed) return 6;
            check(rusticol_runtime_evaluate_f64(runtime, momenta, 16, 1, &before, 1));
            check(rusticol_runtime_save(runtime, argv[2]));
            check(rusticol_runtime_free(runtime));
            check(rusticol_runtime_load(argv[1], NULL, NULL, &runtime));
            check(rusticol_runtime_load_cache(runtime, argv[2]));
            RusticolWarmUpResult warm={0};
            check(rusticol_runtime_warm_up_f64(runtime, momenta, 16, NULL, 0, NULL, 0, NULL, NULL, &warm));
            if (!warm.already_warm || warm.warmed_query_count) return 3;
            check(rusticol_runtime_evaluate_f64(runtime, momenta, 32, 2, after, 2));
            printf("BEFORE %.17g\n", before);
            for (size_t i=0; i<2; ++i) printf("VALUE %zu %.17g\n", i, after[i]);
            check(rusticol_runtime_free(runtime)); return 0;
        }
    """,
    "cpp": r"""
        #include <rusticol.hpp>
        #include <cstdlib>
        #include <iomanip>
        #include <iostream>
        int main(int argc, char **argv) {
            if (argc != 35) return 2;
            std::vector<double> momenta;
            for (int i=3; i<argc; ++i) momenta.push_back(std::strtod(argv[i], nullptr));
            std::vector<double> point(momenta.begin(), momenta.begin()+16);
            double before;
            {
                rusticol::Runtime runtime(argv[1]);
                auto cold=runtime.warm_up(point, {}, {}, {}, 2);
                if (!cold.first_evaluation_completed) return 6;
                before=runtime.evaluate(point, 1).at(0);
                runtime.save(argv[2]);
                const std::string invalid_path=std::string(argv[2])+std::string("\0suffix", 7);
                bool rejected_save=false, rejected_load=false;
                try { runtime.save(invalid_path); } catch (const std::invalid_argument &) { rejected_save=true; }
                try { runtime.load_cache(invalid_path); } catch (const std::invalid_argument &) { rejected_load=true; }
                if (!rejected_save || !rejected_load) return 5;
            }
            rusticol::Runtime restored(argv[1]);
            restored.load_cache(argv[2]);
            auto warm=restored.warm_up(point);
            if (!warm.already_warm || warm.warmed_query_count) return 3;
            auto after=restored.evaluate(momenta, 2);
            std::cout << "BEFORE " << std::setprecision(17) << before << "\n";
            for (std::size_t i=0; i<after.size(); ++i)
                std::cout << "VALUE " << i << " " << std::setprecision(17) << after[i] << "\n";
        }
    """,
    "fortran": r"""
        program cache_roundtrip
          use, intrinsic :: iso_c_binding
          use rusticol
          implicit none
          type(rusticol_runtime) :: runtime
          type(rusticol_warm_up_result) :: warm
          character(len=4096) :: artifact, cache, arg
          real(c_double) :: momenta(32), before
          real(c_double), allocatable :: values(:)
          integer(c_int) :: status
          integer :: i
          if (command_argument_count() /= 34) stop 2
          call get_command_argument(1, artifact)
          call get_command_argument(2, cache)
          do i=1,32
            call get_command_argument(i+2, arg)
            read(arg, *) momenta(i)
          end do
          call runtime%load(trim(artifact))
          call runtime%warm_up(momenta(:16), warm, n_cores=0, ierr=status)
          if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 6
          call runtime%warm_up(momenta(:16), warm, n_cores=-1, ierr=status)
          if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 6
          call runtime%warm_up(momenta(:16), warm, n_cores=2)
          if (warm%first_evaluation_completed == 0) stop 6
          call runtime%evaluate(momenta(:16), 1_c_size_t, values)
          before=values(1)
          call runtime%save(trim(cache))
          call runtime%save(trim(cache)//achar(0)//'suffix', ierr=status)
          if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 5
          call runtime%load_cache(trim(cache)//achar(0)//'suffix', ierr=status)
          if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 5
          call runtime%close()
          call runtime%load(trim(artifact))
          call runtime%load_cache(trim(cache))
          call runtime%warm_up(momenta(:16), warm)
          if (warm%already_warm == 0 .or. warm%warmed_query_count /= 0) stop 3
          call runtime%evaluate(momenta, 2_c_size_t, values)
          write(*,'(A,ES25.17E3)') 'BEFORE ', before
          do i=1,2
            write(*,'(A,I0,1X,ES25.17E3)') 'VALUE ', i-1, values(i)
          end do
          call runtime%close()
        end program cache_roundtrip
    """,
    "rust": r"""
        #[allow(dead_code)]
        mod rusticol { include!(env!("RUSTICOL_RUST_SOURCE")); }
        use rusticol::{Runtime, Selectors};
        fn main() -> Result<(), Box<dyn std::error::Error>> {
            let args: Vec<_> = std::env::args().collect();
            if args.len() != 35 { return Err("wrong argument count".into()); }
            let momenta: Vec<f64> = args[3..].iter().map(|s| s.parse()).collect::<Result<_,_>>()?;
            let before = {
                let mut runtime = Runtime::load(&args[1], None, None)?;
                assert!(runtime.warm_up(&momenta[..16], &Selectors::default(), Some(0), None).is_err());
                let cold = runtime.warm_up_f64(&momenta[..16], &Selectors::default(), Some(2), None)?;
                assert!(cold.first_evaluation_completed);
                let value = runtime.evaluate_f64(&momenta[..16], 1)?[0];
                runtime.save(&args[2])?;
                let invalid_path = format!("{}\0suffix", args[2]);
                assert!(runtime.save(&invalid_path).is_err());
                assert!(runtime.load_cache(&invalid_path).is_err());
                value
            };
            let mut restored = Runtime::load(&args[1], None, None)?;
            restored.load_cache(&args[2])?;
            let warm = restored.warm_up(&momenta[..16], &Selectors::default(), None, None)?;
            assert!(warm.already_warm);
            assert_eq!(warm.warmed_query_count, 0);
            let after = restored.evaluate_f64(&momenta, 2)?;
            println!("BEFORE {before:.17e}");
            for (i, value) in after.iter().enumerate() { println!("VALUE {i} {value:.17e}"); }
            Ok(())
        }
    """,
}


@pytest.fixture(scope="module", params=tuple(_PROBES))
def cache_driver(
    request: pytest.FixtureRequest,
    native_sdk: _NativeSdk,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return _compile_probe(
        request.param,
        _PROBES[request.param],
        native_sdk,
        tmp_path_factory.mktemp(f"cache-sdk-{request.param}"),
    )


def test_completed_otf_cache_roundtrips_across_native_sdk_and_python(
    cache_driver: Path,
    otf_output,
    tmp_path: Path,
) -> None:
    artifact, expected = otf_output
    cache = tmp_path / "completed-cache.bin"
    command = [
        str(cache_driver),
        str(artifact),
        str(cache),
        *(str(value) for point in _POINTS for leg in point for value in leg),
    ]
    result = subprocess.run(
        command, env=_source_environment(), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = {
        int(parts[1]): float(parts[2])
        for line in result.stdout.splitlines()
        if (parts := line.split()) and parts[0] == "VALUE"
    }
    assert set(measured) == {0, 1}
    assert [measured[i] for i in range(2)] == pytest.approx(expected, rel=2e-12)
    before = next(
        float(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("BEFORE ")
    )
    assert before == pytest.approx(expected[0], rel=2e-12)
    assert cache.stat().st_size > 0
    restored = Runtime.load(artifact)
    restored.load_cache(cache)
    warm = restored.warm_up(_POINTS[:1])
    assert warm.already_warm
    assert warm.warmed_query_count == 0
    values = restored.evaluate(_POINTS)
    assert tuple(float(complex(value).real) for value in values) == pytest.approx(
        expected, rel=2e-12
    )

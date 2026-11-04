# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501 -- embedded native probe sources have their own formatting.
"""One generated correlator example, evaluated independently through every SDK."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatorConfig,
    EmitGluon,
    Generator,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from tests.integration.test_runtime_selector_api_parity import (
    _compile_probe,
    _NativeSdk,
    _source_environment,
    _unavailable,
)
from tests.integration.test_runtime_selector_api_parity import (
    native_sdk as _native_sdk_fixture,
)

native_sdk = _native_sdk_fixture

_ROOT = Path(__file__).resolve().parents[2]
_POINTS = (
    (
        (400, 0, 0, 400),
        (400, 0, 0, -400),
        (300, 300, 0, 0),
        (250, -150, 200, 0),
        (250, -150, -200, 0),
    ),
    (
        (400, 0, 0, 400),
        (400, 0, 0, -400),
        (300, 0, 300, 0),
        (250, 200, -150, 0),
        (250, -200, -150, 0),
    ),
)


@pytest.fixture(scope="module", params=("lc", "nlc", "full"))
def correlated_output(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    artifact = tmp_path_factory.mktemp(f"correlated-sdk-{request.param}") / "process"
    alpha = (EmitGluon(2, -3), EmitGluon(-3, -1), EmitGluon(-3, -2))
    beta = (EmitGluon(1, -1), EmitGluon(3, -2), EmitGluon(4, -3))
    # An ordered, non-Hermitian NNLO insertion, multiplied by C_F on the
    # independent hard leg 4, gives a genuinely complex N3LO test below.
    ordered_bra = (EmitGluon(2, -1), EmitGluon(1, -2), EmitGluon(4, -3))
    ordered_ket = (EmitGluon(3, -2), EmitGluon(1, -1), EmitGluon(4, -3))
    declarations = CorrelatorConfig(
        color_correlations=(
            ColorCorrelator("cascade-interference", alpha, beta),
            ColorCorrelator("ordered", ordered_bra, ordered_ket),
            ColorCorrelator("ordered-reverse", ordered_ket, ordered_bra),
        ),
        spin_correlations=((5,),),
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=request.param),
        generation=GenerationConfig(workers=1),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(("d d~ > d d~ g",), names=("dd_ddg",)),
        artifact,
        model=ModelSource.built_in_sm(),
        correlators=declarations,
    )
    runtime = Runtime.load(artifact, process="dd_ddg")
    oracle = runtime.evaluate_correlated_many(
        _POINTS,
        {
            "born": CorrelatedRequest("born", {}),
            "interference": CorrelatedRequest("cascade-interference", {}),
            "spin": CorrelatedRequest("cascade-interference", {5: (0, 0, 0, 1)}),
        },
        precision=50,
    )
    return (
        artifact,
        tuple(tuple(complex(value) for value in series) for series in oracle.values()),
        request.param,
    )


@pytest.fixture(scope="module", params=("c", "cpp", "fortran", "rust"))
def correlated_driver(
    request: pytest.FixtureRequest,
    native_sdk: _NativeSdk,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    extension = {"c": "c", "cpp": "cpp", "fortran": "f90", "rust": "rs"}[request.param]
    source = (_ROOT / "examples" / "native" / f"correlated.{extension}").read_text()
    return _compile_probe(
        request.param,
        source,
        native_sdk,
        tmp_path_factory.mktemp(f"correlated-sdk-{request.param}"),
    )


def test_correlated_sdk_matches_python_oracle(
    correlated_driver: Path, correlated_output
) -> None:
    artifact, oracle, _ = correlated_output
    result = subprocess.run(
        [str(correlated_driver), str(artifact), "dd_ddg"],
        capture_output=True,
        text=True,
        env=_source_environment(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measured = {}
    for line in result.stdout.splitlines():
        if line.startswith("VALUE "):
            _, request, point, real, imaginary = line.split()
            key = (int(request), int(point))
            assert key not in measured
            measured[key] = complex(float(real), float(imaginary))
    assert set(measured) == {
        (request, point) for request in range(3) for point in range(2)
    }
    for request, series in enumerate(oracle):
        for point, expected in enumerate(series):
            assert measured[request, point] == pytest.approx(
                expected, rel=2e-10, abs=1e-15
            )


# These small input-driven probes exercise the same physics requests through
# each public wrapper. Numerical assertions stay in Python, avoiding four
# independent copies of the physics conventions and tolerances.
_PHYSICS_PROBES = {
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
            if (argc != 4) return 2;
            FILE *input = fopen(argv[3], "r");
            size_t points, legs, count;
            if (!input || fscanf(input, "%zu %zu %zu", &points, &legs, &count) != 3) return 2;
            size_t length = 4 * points * legs;
            double *momenta = calloc(length, sizeof(double));
            for (size_t i=0; i<length; ++i) if (fscanf(input, "%lf", momenta+i) != 1) return 2;
            RusticolCorrelatedRequest *requests = calloc(count, sizeof(*requests));
            for (size_t r=0; r<count; ++r) {
                char *id = calloc(128, 1); size_t spins;
                if (fscanf(input, "%127s %zu", id, &spins) != 2) return 2;
                RusticolSpinCorrelationVector *vectors = calloc(spins, sizeof(*vectors));
                for (size_t s=0; s<spins; ++s) {
                    size_t leg; double *values = calloc(8 * points, sizeof(double));
                    if (fscanf(input, "%zu", &leg) != 1) return 2;
                    for (size_t i=0; i<8*points; ++i) if (fscanf(input, "%lf", values+i) != 1) return 2;
                    vectors[s] = (RusticolSpinCorrelationVector){leg, points, values, 8*points};
                }
                requests[r] = (RusticolCorrelatedRequest){id, vectors, spins, 0};
            }
            fclose(input);
            RusticolRuntimeHandle *runtime = NULL;
            check(rusticol_runtime_load(argv[1], argv[2], NULL, &runtime));
            double *before=calloc(points,sizeof(double)), *after=calloc(points,sizeof(double));
            double *values=calloc(2*points*count,sizeof(double));
            check(rusticol_runtime_evaluate_f64(runtime,momenta,length,points,before,points));
            check(rusticol_runtime_evaluate_correlated_many_f64(runtime,momenta,length,points,requests,count,NULL,0,values,2*points*count));
            check(rusticol_runtime_evaluate_f64(runtime,momenta,length,points,after,points));
            for (size_t p=0; p<points; ++p) if(before[p]!=after[p]) return 1;
            for(size_t r=0;r<count;++r) for(size_t p=0;p<points;++p)
                printf("VALUE %zu %zu %.17g %.17g\n",r,p,values[2*(r*points+p)],values[2*(r*points+p)+1]);
            for(size_t r=0;r<count;++r) {
                for(size_t s=0;s<requests[r].spin_vector_count;++s) free((void*)requests[r].spin_vectors[s].components);
                free((void*)requests[r].spin_vectors); free((void*)requests[r].color_correlation);
            }
            free(requests);free(values);free(before);free(after);free(momenta);
            check(rusticol_runtime_free(runtime)); return 0;
        }
    """,
    "cpp": r"""
        #include <rusticol.hpp>
        #include <fstream>
        #include <iomanip>
        #include <iostream>
        int main(int argc,char **argv) {
            if(argc!=4) return 2;
            try {
                std::ifstream input(argv[3]); size_t points,legs,count;
                input>>points>>legs>>count;
                std::vector<double> momenta(4*points*legs);
                for(auto &value:momenta) input>>value;
                std::vector<rusticol::CorrelatedRequest> requests;
                for(size_t r=0;r<count;++r) {
                    std::string id;size_t spins;input>>id>>spins;
                    std::vector<rusticol::SpinCorrelationVector> vectors;
                    for(size_t s=0;s<spins;++s) {
                        rusticol::SpinCorrelationVector vector;input>>vector.leg;
                        vector.components.resize(points);
                        for(auto &point:vector.components) for(auto &value:point) {
                            double re,im;input>>re>>im;value={re,im};
                        }
                        vectors.push_back(std::move(vector));
                    }
                    requests.push_back({id,vectors});
                }
                if(!input) return 2;
                rusticol::Runtime runtime(argv[1],argv[2]);
                const auto before=runtime.evaluate(momenta,points);
                const auto values=runtime.evaluate_correlated_many(momenta,points,requests);
                if(before!=runtime.evaluate(momenta,points)) return 1;
                std::cout<<std::setprecision(17);
                for(size_t r=0;r<count;++r) for(size_t p=0;p<points;++p)
                    std::cout<<"VALUE "<<r<<' '<<p<<' '<<values(r,p).real()<<' '<<values(r,p).imag()<<'\n';
            } catch(const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
        }
    """,
    "fortran": r"""
        program physics_probe
          use, intrinsic :: iso_c_binding
          use rusticol
          implicit none
          type(rusticol_runtime) :: runtime
          type(rusticol_correlated_request), allocatable :: requests(:)
          character(len=4096) :: artifact, key, filename
          character(len=128) :: identifier
          integer :: unit, points, legs, count, spins, r, p, s, mu
          real(c_double), allocatable :: momenta(:), before(:), after(:)
          complex(c_double_complex), allocatable :: values(:,:)
          real(c_double) :: re, im
          call get_command_argument(1,artifact)
          call get_command_argument(2,key)
          call get_command_argument(3,filename)
          open(newunit=unit,file=trim(filename),status="old",action="read")
          read(unit,*) points,legs,count
          allocate(momenta(4*points*legs),requests(count))
          read(unit,*) momenta
          do r=1,count
            read(unit,*) identifier,spins
            requests(r)%color_correlation=trim(identifier)
            allocate(requests(r)%spin_vectors(spins))
            do s=1,spins
              read(unit,*) requests(r)%spin_vectors(s)%leg
              allocate(requests(r)%spin_vectors(s)%components(4,points))
              do p=1,points
                do mu=1,4
                  read(unit,*) re,im
                  requests(r)%spin_vectors(s)%components(mu,p)=cmplx(re,im,kind=c_double)
                end do
              end do
            end do
          end do
          close(unit)
          call runtime%load(trim(artifact),process_key=trim(key))
          call runtime%evaluate(momenta,int(points,c_size_t),before)
          call runtime%evaluate_correlated_many(momenta,int(points,c_size_t),requests,values)
          call runtime%evaluate(momenta,int(points,c_size_t),after)
          if(any(before/=after)) error stop "ordinary result changed"
          do r=1,count
            do p=1,points
              write(*,'(A,1X,I0,1X,I0,2(1X,ES25.17E3))') "VALUE",r-1,p-1,real(values(p,r),c_double),aimag(values(p,r))
            end do
          end do
          call runtime%close()
        end program physics_probe
    """,
    "rust": r"""
        #[allow(dead_code)]
        mod rusticol { include!(env!("RUSTICOL_RUST_SOURCE")); }
        use rusticol::{Complex64,CorrelatedRequest,Runtime,SpinCorrelationVector};
        fn main()->Result<(),Box<dyn std::error::Error>> {
            let args=std::env::args().collect::<Vec<_>>();
            let data=std::fs::read_to_string(&args[3])?;
            let mut tokens=data.split_whitespace();
            let points:usize=tokens.next().unwrap().parse()?;
            let legs:usize=tokens.next().unwrap().parse()?;
            let count:usize=tokens.next().unwrap().parse()?;
            let mut momenta=Vec::new();
            for _ in 0..4*points*legs { momenta.push(tokens.next().unwrap().parse::<f64>()?); }
            let mut requests=Vec::new();
            for _ in 0..count {
                let id=tokens.next().unwrap().to_owned();
                let spins:usize=tokens.next().unwrap().parse()?;
                let mut vectors=Vec::new();
                for _ in 0..spins {
                    let leg=tokens.next().unwrap().parse()?;
                    let mut components=vec![[Complex64::default();4];points];
                    for point in &mut components { for value in point {
                        *value=Complex64::new(tokens.next().unwrap().parse()?,tokens.next().unwrap().parse()?);
                    } }
                    vectors.push(SpinCorrelationVector{leg,components});
                }
                requests.push(CorrelatedRequest{color_correlation:id,spin_vectors:Some(vectors)});
            }
            let mut runtime=Runtime::load(&args[1],Some(&args[2]),None)?;
            let before=runtime.evaluate_f64(&momenta,points)?;
            let values=runtime.evaluate_correlated_many_f64(&momenta,points,&requests,&[])?;
            assert_eq!(before,runtime.evaluate_f64(&momenta,points)?);
            for r in 0..count {for p in 0..points {
                let value=values.get(r,p).unwrap();
                println!("VALUE {r} {p} {:.17e} {:.17e}",value.re,value.im);
            }}
            Ok(())
        }
    """,
}


@pytest.fixture(scope="module")
def correlated_physics_output(tmp_path_factory):
    from tests.integration.test_correlators import _configuration

    output = tmp_path_factory.mktemp("correlated-sdk-physics") / "process"
    cascade = (EmitGluon(1, -3), EmitGluon(-3, -1), EmitGluon(-3, -2))
    independent = (EmitGluon(2, -1), EmitGluon(3, -2), EmitGluon(4, -3))
    triple = tuple(EmitGluon(1, -i) for i in range(1, 4))
    Generator(_configuration()).generate(
        ProcessSet.from_expressions(("g g > g g",), names=("gg",)),
        output,
        model=ModelSource.built_in_sm(),
        correlators=CorrelatorConfig(
            color_correlations=(
                *(ColorCorrelator.dipole(f"T1{leg}", 1, leg) for leg in range(1, 5)),
                ColorCorrelator("triple-self", triple, triple),
                ColorCorrelator("forward", cascade, independent),
                ColorCorrelator("reverse", independent, cascade),
            ),
            spin_correlations=((3,), (4,), (3, 4)),
        ),
    )
    points = (
        ((500, 0, 0, 500), (500, 0, 0, -500), (500, 300, 0, 400), (500, -300, 0, -400)),
        ((500, 0, 0, 500), (500, 0, 0, -500), (500, 400, 0, 300), (500, -400, 0, -300)),
    )
    transverse = (
        ((0, 0.8, 0, -0.6), (0, 0.6, 0, -0.8)),
        ((0, 0, 1, 0),) * 2,
    )
    requests = {"born": CorrelatedRequest("born", {})}
    requests.update(
        {f"T1{leg}": CorrelatedRequest(f"T1{leg}", {}) for leg in range(1, 5)}
    )
    requests["triple-self"] = CorrelatedRequest("triple-self", {})
    for leg in (3, 4):
        for pol, vectors in enumerate(transverse):
            requests[f"pol{leg}-{pol}"] = CorrelatedRequest("born", {leg: vectors})
        requests[f"ward{leg}"] = CorrelatedRequest(
            "born", {leg: tuple(point[leg - 1] for point in points)}
        )
    for a in range(2):
        for b in range(2):
            requests[f"joint-{a}{b}"] = CorrelatedRequest(
                "born", {3: transverse[a], 4: transverse[b]}
            )
    requests["joint-ward"] = CorrelatedRequest(
        "born", {leg: tuple(point[leg - 1] for point in points) for leg in (3, 4)}
    )
    requests["mixed-ward"] = CorrelatedRequest(
        "T13", {3: tuple(point[2] for point in points), 4: transverse[0]}
    )
    complex_vectors = {
        leg: tuple(
            tuple(complex(x, y) for x, y in zip(first, second, strict=True))
            for first, second in zip(transverse[0], transverse[1], strict=True)
        )
        for leg in (3, 4)
    }
    requests["complex-forward"] = CorrelatedRequest("forward", complex_vectors)
    requests["complex-reverse"] = CorrelatedRequest("reverse", complex_vectors)
    runtime = Runtime.load(output, process="gg")
    oracle = runtime.evaluate_correlated_many(points, requests, precision=45)
    return output, points, requests, oracle


def _write_physics_input(path, points, requests):
    lines = [f"{len(points)} {len(points[0])} {len(requests)}"]
    lines.append(
        " ".join(str(value) for point in points for leg in point for value in leg)
    )
    for request in requests.values():
        vectors = request.spin_vectors or {}
        lines.append(f"{request.color_correlation} {len(vectors)}")
        for leg, batch in vectors.items():
            lines.append(str(leg))
            for vector in batch:
                for component in vector:
                    number = complex(component)
                    lines.append(f"{number.real:.17g} {number.imag:.17g}")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("language", ("c", "cpp", "fortran", "rust"))
def test_all_native_sdks_colour_coherence_and_joint_spin_physics(
    language, native_sdk, correlated_physics_output, tmp_path
):
    output, points, requests, oracle = correlated_physics_output
    binary = _compile_probe(language, _PHYSICS_PROBES[language], native_sdk, tmp_path)
    input_path = tmp_path / "requests.txt"
    _write_physics_input(input_path, points, requests)
    run = subprocess.run(
        [str(binary), str(output), "gg", str(input_path)],
        capture_output=True,
        text=True,
        env=_source_environment(),
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    labels = tuple(requests)
    measured = {}
    for line in run.stdout.splitlines():
        if line.startswith("VALUE "):
            _, request, point, real, imag = line.split()
            key = (labels[int(request)], int(point))
            assert key not in measured
            measured[key] = complex(float(real), float(imag))
    assert set(measured) == {(label, point) for label in labels for point in range(2)}
    for point in range(2):
        born = measured["born", point]
        assert born.real > 0
        assert abs(born.imag) < born.real * 1e-12
        assert measured["T11", point] == pytest.approx(3 * born, rel=2e-10)
        assert (
            abs(sum(measured[f"T1{leg}", point] for leg in range(1, 5)))
            < abs(born) * 2e-10
        )
        assert measured["triple-self", point] == pytest.approx(27 * born, rel=2e-10)
        for leg in (3, 4):
            assert sum(
                measured[f"pol{leg}-{pol}", point] for pol in range(2)
            ) == pytest.approx(born, rel=2e-10)
            assert abs(measured[f"ward{leg}", point]) < abs(born) * 1e-15 * 1e6
        assert sum(
            measured[f"joint-{a}{b}", point] for a in range(2) for b in range(2)
        ) == pytest.approx(born, rel=2e-10)
        assert abs(measured["joint-ward", point]) < abs(born) * 1e-15 * 1e12
        assert abs(measured["mixed-ward", point]) < abs(born) * 1e-15 * 1e6
        assert measured["complex-reverse", point] == pytest.approx(
            measured["complex-forward", point].conjugate(),
            rel=2e-10,
            abs=abs(born) * 1e-12,
        )
        for label in labels:
            value = measured[label, point]
            assert math.isfinite(value.real) and math.isfinite(value.imag)
            if "ward" not in label:
                assert value == pytest.approx(
                    complex(oracle[label][point]), rel=2e-10, abs=abs(born) * 1e-12
                )


@pytest.fixture(scope="module")
def directed_complex_requests(correlated_output):
    output, _, accuracy = correlated_output
    runtime = Runtime.load(output, process="dd_ddg")
    vectors = {
        5: ((1 + 2j, 3 - 4j, 5 + 7j, -2 + 3j), (2 - 1j, -4 + 3j, 7 - 5j, 3 + 2j))
    }
    requests = {
        "forward": CorrelatedRequest("ordered", vectors),
        "reverse": CorrelatedRequest("ordered-reverse", vectors),
    }
    oracle = runtime.evaluate_correlated_many(_POINTS, requests, precision=45)
    return output, requests, oracle, accuracy


@pytest.mark.parametrize("language", ("c", "cpp", "fortran", "rust"))
def test_native_directed_complex_insertion_and_reverse(
    language, native_sdk, directed_complex_requests, tmp_path
):
    output, requests, oracle, accuracy = directed_complex_requests
    # A genuinely nonzero imaginary part prevents a real-only reducer from
    # passing the conjugacy check vacuously.
    if accuracy == "lc":
        # This ordering starts beyond leading colour; LC must not promote it.
        assert all(complex(value) == 0 for value in oracle["forward"])
    else:
        assert any(
            abs(complex(value).imag) > abs(complex(value)) * 1e-5
            for value in oracle["forward"]
        )
    binary = _compile_probe(language, _PHYSICS_PROBES[language], native_sdk, tmp_path)
    input_path = tmp_path / "directed-requests.txt"
    _write_physics_input(input_path, _POINTS, requests)
    run = subprocess.run(
        [str(binary), str(output), "dd_ddg", str(input_path)],
        capture_output=True,
        text=True,
        env=_source_environment(),
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    measured = {}
    for line in run.stdout.splitlines():
        if line.startswith("VALUE "):
            _, request, point, real, imag = line.split()
            measured[int(request), int(point)] = complex(float(real), float(imag))
    assert set(measured) == {
        (request, point) for request in range(2) for point in range(2)
    }
    for point in range(2):
        for request, label in enumerate(requests):
            assert measured[request, point] == pytest.approx(
                complex(oracle[label][point]), rel=2e-10
            )
        assert measured[1, point] == pytest.approx(
            measured[0, point].conjugate(), rel=2e-10
        )

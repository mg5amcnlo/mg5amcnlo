# SPDX-License-Identifier: 0BSD
"""Cheap study checks: no generation, native evaluations, or Symbolica session."""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/NLO_limits/nlo_limit_stability_tests.py"
)
spec = importlib.util.spec_from_file_location("nlo_limit_study", SCRIPT)
assert spec is not None and spec.loader is not None
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def value(real, imag=0):
    return SimpleNamespace(real=Decimal(real), imag=Decimal(imag))


@pytest.mark.parametrize("resolved", (False, True))
def test_cached_born_requires_current_per_process_declarations(tmp_path, resolved):
    expected = study.declarations().to_json_dict()
    for name, expression in study.EXPRESSIONS.items():
        directory = tmp_path / name
        (directory / "config").mkdir(parents=True)
        (directory / "artifact.json").write_text(json.dumps({
            "artifact_id": name, "processes": [{"expression": expression}],
        }))
        (directory / "config/effective.toml").write_text(
            '[color]\naccuracy="full"\ncontraction="direct"\n'
            '[evaluator]\nexecution_mode="compiled"\n'
            '[generation.relation_discovery]\n'
            f'mode="{study.STUDY["relation_discovery"][name]}"\n'
        )
        if name == "born":
            (directory / "correlators.json").write_text(json.dumps({
                "declarations": expected,
                "processes": {"born": {"declarations": expected} if resolved else {}},
            }))
    if resolved:
        paths, identities = study.generate_artifacts(tmp_path)
        assert paths["born"] == tmp_path / "born"
        assert identities["born"] == "born"
    else:
        with pytest.raises(ValueError, match="fresh artifact directory"):
            study.generate_artifacts(tmp_path)


@pytest.mark.parametrize("limit", ("collinear", "soft"))
@pytest.mark.parametrize("exponent", (1, 4, 15, 100))
def test_source_is_onshell_conserving_and_maps_to_fixed_born(limit, exponent):
    source = study.make_point(limit, exponent, 300)
    for key, item in source["checks"].items():
        if key == "positive_energies":
            assert item
        else:
            assert Decimal(item) < Decimal("1e-175"), key
    with localcontext() as context:
        context.prec = 300
        born = [[Decimal(v) for v in p] for p in source["born"]]
        e = [Decimal(v) for v in source["spin"]]
        assert abs(study.dot(e, e) + 1) < Decimal("1e-290")
        assert abs(study.dot(e, born[2])) < Decimal("1e-285")
        assert abs(study.dot(e, born[3])) < Decimal("1e-285")
        if limit == "collinear":
            expected = (Decimal(1000) * Decimal(source["delta"])) ** 2
            expected /= Decimal("0.6") * Decimal("0.4")
            assert abs(Decimal(source["s_ij"]) / expected - 1) < Decimal("1e-90")


def test_soft_direction_is_fixed_and_wide_angle():
    with localcontext() as context:
        context.prec = 110
        first, second = (study.make_point("soft", x, 100) for x in (2, 10))
        q = [Decimal(v) / Decimal(first["delta"]) for v in first["real"][3]]
        r = [Decimal(v) / Decimal(second["delta"]) for v in second["real"][3]]
        assert max(abs(a - b) for a, b in zip(q, r, strict=True)) < Decimal("1e-90")
        assert abs(q[0] - 500) < Decimal("1e-90")
        for raw in first["born"]:
            p = [Decimal(v) for v in raw]
            assert study.dot(p, q) / (p[0] * q[0]) > Decimal("0.1")


def test_binary64_operations_do_not_inherit_decimal_precision():
    with localcontext() as context:
        context.prec = 120
        study.arithmetic_smoke("double")
        d = study.Binary64
        assert d(1) + d("1e-20") == 1
        assert (d(1) / d(3)) == Decimal.from_float(1.0 / 3.0)
        assert d(2).sqrt() == Decimal.from_float(2.0**0.5)
        assert isinstance(sum((d(1), d(2))), d)
        assert isinstance(-d(1), d)
        assert isinstance(+d(1), d)


def test_local_spin_counterterm_reduces_to_average_only_for_half_density():
    source = study.make_point("collinear", 3, 100)
    with localcontext() as context:
        context.prec = 100
        components = {"born": value(1), "spin": value("0.5")}
        actual = study.counterterm(source, components, Decimal, Decimal(1))
        z, sij = Decimal(source["z"]), Decimal(source["s_ij"])
        expected = 8 * Decimal(study.ALPHA_S) / sij * 6
        expected *= z / (1 - z) + (1 - z) / z + z * (1 - z)
        expected *= 6  # Remove the public Born's 1/3!.
        assert abs(actual / expected - 1) < Decimal("1e-90")
        components["spin"] = value("0.3")
        assert study.counterterm(source, components, Decimal, Decimal(1)) != expected


def test_soft_sign_pair_count_and_scaling():
    components = {"born": value("1.333333333333333333333333333333333")}
    components.update({f"T{a}{b}": value(-1) for a, b in study.DIPOLES})
    with localcontext() as context:
        context.prec = 100
        counterterms = [
            study.counterterm(
                study.make_point("soft", x, 110), components, Decimal, Decimal(1)
            )
            for x in (2, 3)
        ]
        assert counterterms[0] > 0
        assert abs(counterterms[1] / counterterms[0] - 100) < Decimal("1e-90")
        source = study.make_point("soft", 2, 110)
        p = [[Decimal(v) for v in vector] for vector in source["born"]]
        q = [Decimal(v) for v in source["real"][3]]
        ordered = sum(
            study.dot(a, b) / (study.dot(a, q) * study.dot(b, q))
            for i, a in enumerate(p)
            for j, b in enumerate(p)
            if i != j
        )
        assert abs(
            counterterms[0] / (4 * Decimal(study.ALPHA_S) * 6 * ordered) - 1
        ) < Decimal("1e-90")


def test_many_requests_include_physical_born_all_dipoles_and_vector_batch():
    sources = [study.make_point("collinear", x, 80) for x in (2, 3, 4)]
    requests = study.correlation_requests(sources, diagnostics=True)
    assert requests["born"].spin_vectors == {}
    assert requests["T13"].color_correlation == "T13"
    assert len(requests["spin"].spin_vectors[3]) == len(sources)
    assert {f"T{a}{b}" for a, b in study.DIPOLES} <= requests.keys()
    assert {"basis1", "basis2", "ward", "T33"} <= requests.keys()


def test_real_batch_bisection_is_exception_only():
    class FakeRuntime:
        def __init__(self, bad=False):
            self.calls = []
            self.bad = bad

        def evaluate(self, points, precision):
            self.calls.append(points)
            if self.bad and any(p[0][0] == 2 for p in points):
                raise ArithmeticError("singular endpoint")
            return [complex(p[0][0]) for p in points]

    sources = [{"real": [[str(i), "0", "0", "0"]]} for i in (1, 2, 3)]
    runtime = FakeRuntime()
    values, diagnostic = study.evaluate_real_batch(
        runtime, sources, "double", 16, study.Binary64
    )
    assert values == [1, 2, 3]
    assert len(runtime.calls) == 1 and not diagnostic["diagnostic_bisection"]
    runtime = FakeRuntime(bad=True)
    values, diagnostic = study.evaluate_real_batch(
        runtime, sources, "double", 16, study.Binary64
    )
    assert values == [1, None, 3]
    assert diagnostic["diagnostic_bisection"] and 1 in diagnostic["errors"]


def test_stability_metric_does_not_invent_references():
    record = {"format": "double", "status": "ok", "F": "1.0000000001"}
    reference = {"precision": 1200, "status": "ok", "F": "1"}
    result = study.compare_records(record, reference)
    assert Decimal(result["stable_digits"]) == 10
    assert 0 < result["stable_fraction"] < 1
    assert study.compare_records(record, None) == {"reference_status": "missing"}
    assert "stable_fraction" not in study.compare_records(
        record, {**reference, "F": "0"}
    )
    assert (
        study.compare_records({**record, "status": "failed"}, reference)[
            "stable_fraction"
        ]
        == 0
    )


def test_checkpoint_and_render_parser_are_independent_of_execution(tmp_path):
    path = tmp_path / "results.json"
    report = {"study": study.STUDY, "records": {}}
    study.checkpoint(path, report)
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    args = study.parser().parse_args(["--render", "--output", str(path)])
    assert args.render and args.memory_limit_gib == 10.0


def test_binary64_adapter_is_scoped_and_sets_retained_stage_precision():
    from pyamplicol.runtime import _color_topology_exact, correlations, symbolica_exact

    original_decimal = symbolica_exact.Decimal
    original_working = symbolica_exact._working_precision
    original_topology_zero = _color_topology_exact._ZERO
    seen = []

    class Stage:
        def evaluate_complex(self, values):
            seen.append(values)
            return [[complex(0.1)]]

        def evaluate_complex_with_prec(self, values, precision):
            raise AssertionError("binary64 must not use arbitrary Float dispatch")

    evaluator = symbolica_exact._ExactEvaluator(input_len=1, evaluator=Stage())
    with study.arithmetic("double", 16):
        assert symbolica_exact._working_precision(1000) == 16
        assert correlations.Decimal is study.Binary64
        assert _color_topology_exact.Decimal is study.Binary64
        assert isinstance(_color_topology_exact._ZERO, study.Binary64)
        assert isinstance(_color_topology_exact._ONE, study.Binary64)
        assert isinstance(_color_topology_exact._TWO, study.Binary64)
        output = evaluator.evaluate(((study.Binary64("0.1"), study.Binary64(0)),), 16)
        assert seen[0] == [complex(0.1)]
        assert isinstance(output[0][0], study.Binary64)
    assert symbolica_exact.Decimal is original_decimal
    assert symbolica_exact._working_precision is original_working
    assert correlations.Decimal is Decimal
    assert _color_topology_exact.Decimal is Decimal
    assert _color_topology_exact._ZERO is original_topology_zero


def test_evaluation_requests_whole_point_batch_once(monkeypatch):
    import pyamplicol

    calls = []
    loads = []

    class FakeRuntime:
        @classmethod
        def load(cls, path, model_parameters):
            loads.append((path, model_parameters))
            return cls()

        def evaluate_correlated_many(self, points, requests, precision):
            calls.append((points, requests))
            return {
                name: tuple(value(1 if name == "born" else "0.3") for _ in points)
                for name in requests
            }

        def evaluate(self, points, precision):
            return [1j * 0 + 1 for _ in points]

    @contextmanager
    def fake_arithmetic(name, precision):
        with localcontext() as context:
            context.prec = 60
            yield Decimal

    monkeypatch.setattr(pyamplicol, "Runtime", FakeRuntime)
    monkeypatch.setattr(study, "arithmetic", fake_arithmetic)
    monkeypatch.setattr(study, "target_pi", lambda scalar, precision: Decimal(1))
    sources = [study.make_point("collinear", x, 80) for x in (2, 3, 4)]
    records = study.evaluate_batch(
        {"real": "real", "born": "born"}, sources, "double", 16
    )
    assert loads == [
        ("real", {"alpha_s": float(study.ALPHA_S)}),
        ("born", {"alpha_s": float(study.ALPHA_S)}),
    ]
    assert len(calls) == 1 and len(calls[0][0]) == 3
    assert len(calls[0][1]) == 12
    assert [record["exponent"] for record in records] == [2, 3, 4]
    assert records[2]["correlations"]["T13"] == {"real": "0.3", "imag": "0"}
    assert records[2]["batch"]["public_correlated_many_calls"] == 1
    assert records[2]["batch"]["public_real_calls"] == 1
    assert records[2]["batch"]["point_count"] == 3


def test_resume_preserves_failed_measurements_unless_retry_is_explicit():
    report = {
        "records": {
            "collinear:double:1": {"status": "ok"},
            "collinear:double:2": {"status": "failed"},
        }
    }
    assert study.pending_exponents(report, "collinear", "double", [1, 2, 3]) == [3]
    assert study.pending_exponents(
        report, "collinear", "double", [1, 2, 3], retry_failed=True
    ) == [2, 3]


@pytest.mark.parametrize(
    ("oracle_status", "target_status"),
    (("failed", "ok"), ("missing", "ok"), ("ok", "failed"), ("ok", "ok")),
)
def test_scan_requires_requested_oracles_but_partial_render_is_available(
    tmp_path, monkeypatch, oracle_status, target_status
):
    output = tmp_path / "results.json"
    study.checkpoint(output, {
        "study": study.STUDY,
        "oracle_precision": 1200,
        "records": {
            f"{limit}:oracle:{exponent}": {
                "limit": limit, "format": "oracle", "exponent": exponent,
                "status": "failed",
            }
            for limit, exponent in (("soft", 2), ("collinear", 99))
        },
    })
    rendered = []

    def evaluate(paths, sources, format_name, precision, **kwargs):
        status = oracle_status if format_name == "oracle" else target_status
        if status == "missing":
            return []
        return [{
            "limit": source["limit"], "exponent": source["exponent"],
            "format": format_name, "precision": precision, "status": status,
            "F": "1", "F_over_R": "0.001",
        } for source in sources]

    monkeypatch.setattr(study, "generate_artifacts", lambda path: ({}, {}))
    monkeypatch.setattr(study, "evaluate_batch", evaluate)
    monkeypatch.setattr(study, "render", lambda report, path: rendered.append(report))
    arguments = [
        "--_guard-child", "--formats", "arbitrary-1000", "--limits", "collinear",
        "--exponents", "2", "--output", str(output),
    ]
    if oracle_status == "ok":
        # One requested point suffices; historical failed references are irrelevant.
        assert study.main(arguments) == 0
        assert len(rendered) == 1
    else:
        with pytest.raises(ValueError, match="oracle measurements: collinear:oracle:2"):
            study.main(arguments)
        assert not rendered
    retained = json.loads(output.read_text())
    target = retained["records"]["collinear:arbitrary-1000:2"]
    assert target["status"] == target_status
    if oracle_status == "failed":
        assert retained["records"]["collinear:oracle:2"]["status"] == "failed"
    if oracle_status == "ok" and target_status == "failed":
        assert target["stable_fraction"] == 0
    # An explicit redraw remains available even when no valid reference exists.
    rendered.clear()
    assert study.main(["--render", "--output", str(output)]) == 0
    assert rendered == [retained]


def test_prepare_only_allows_smaller_oracle_without_starting_arithmetic(monkeypatch):
    calls = []

    def prepare(directory):
        calls.append(directory)
        return {}, {"real": "a", "born": "b"}

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only must not evaluate arithmetic or amplitudes")

    monkeypatch.setattr(study, "generate_artifacts", prepare)
    monkeypatch.setattr(study, "arithmetic_smoke", forbidden)
    monkeypatch.setattr(study, "evaluate_batch", forbidden)
    assert (
        study.main(
            [
                "--prepare-only",
                "--_guard-child",
                "--formats",
                "double",
                "double-double",
                "--oracle-precision",
                "160",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="exceed every selected target"):
        study.main(["--oracle-precision", "160"])

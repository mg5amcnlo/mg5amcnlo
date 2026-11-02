#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Fixed-azimuth collinear and wide-angle soft NLO subtraction stability.

Run --smoke first. No paper-directory imports or amplitude formulae are used.
The low-precision adapters are confined to this serial study process; they
do not change pyAmpliCol's ordinary precision policy. See README.md for the
physics definition and why the unaveraged remainder need not be finite.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from decimal import Decimal, localcontext
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[2]
FORMATS = {
    "double": (16, 53 * math.log10(2)),
    "double-double": (32, 31.0),
    "arbitrary-1000": (1000, 1000.0),
}
EXPRESSIONS = {"real": "g g > g g g g", "born": "g g > g g g"}
DIPOLES = tuple(combinations(range(1, 6), 2))
ALPHA_S = "0.125"
STUDY = {
    "version": 1,
    "real": EXPRESSIONS["real"],
    "born": EXPRESSIONS["born"],
    "sqrt_s": "1000",
    "alpha_s": ALPHA_S,
    "z_collinear": "3/5",
    "azimuth_cos_sin": ["3/5", "4/5"],
    "normalization": "initial averaged; final symmetry removed: R*24, B*6",
    "difference": "F=C-R, without azimuth averaging",
    "soft_kernel_momenta": "mapped Born; crossed colour charges; no extra signs",
    "relation_discovery": {"real": "certified-reuse", "born": "off"},
}


def transport(value):
    """Decimal transport must not make Symbolica infer a smaller precision."""
    value = Decimal(value)
    with localcontext() as context:
        context.prec = max(80, len(value.as_tuple().digits))
        return Decimal(format(value, ".79E"))


class Binary64(Decimal):
    """Decimal-compatible transport; every scalar operation rounds to f64."""

    @classmethod
    def operation(cls, operation, *values):
        args = [float(value) for value in values]
        if operation == "identity":
            result = args[0]
        elif operation == "add":
            result = args[0] + args[1]
        elif operation == "sub":
            result = args[0] - args[1]
        elif operation == "mul":
            result = args[0] * args[1]
        elif operation == "div":
            result = args[0] / args[1]
        elif operation == "sqrt":
            result = math.sqrt(args[0])
        else:
            raise ValueError(f"unknown arithmetic operation: {operation}")
        if not math.isfinite(result):
            raise ArithmeticError(f"non-finite binary64 {operation}")
        return Decimal.__new__(cls, Decimal.from_float(result))

    def __new__(cls, value=0):
        if isinstance(value, cls):
            return value
        return cls.operation("identity", Decimal(value))

    @classmethod
    def from_float(cls, value):
        return cls(Decimal.from_float(value))

    def __add__(self, other):
        return self.operation("add", self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.operation("sub", self, other)

    def __rsub__(self, other):
        return self.operation("sub", other, self)

    def __mul__(self, other):
        return self.operation("mul", self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.operation("div", self, other)

    def __rtruediv__(self, other):
        return self.operation("div", other, self)

    def __neg__(self):
        return Decimal.__new__(type(self), Decimal.copy_negate(self))

    def __abs__(self):
        return Decimal.__new__(type(self), Decimal.copy_abs(self))

    def __pos__(self):
        return self

    def __pow__(self, exponent, modulo=None):
        if modulo is not None or int(exponent) != exponent:
            raise TypeError("study arithmetic supports integer powers only")
        exponent = int(exponent)
        if exponent < 0:
            return type(self)(1) / self ** (-exponent)
        result, factor = type(self)(1), self
        while exponent:
            if exponent & 1:
                result *= factor
            exponent >>= 1
            if exponent:
                factor *= factor
        return result

    def sqrt(self, context=None):
        return self.operation("sqrt", self)


class DoubleDouble(Binary64):
    """Actual Symbolica DoubleFloat arithmetic, with 31-digit transport."""

    evaluators: ClassVar[dict] = {}

    @classmethod
    def operation(cls, operation, *values):
        if operation not in cls.evaluators:
            from symbolica import E, S

            expression, names = {
                "identity": ("x", ["x"]),
                "add": ("x+y", ["x", "y"]),
                "sub": ("x-y", ["x", "y"]),
                "mul": ("x*y", ["x", "y"]),
                "div": ("x/y", ["x", "y"]),
                "sqrt": ("sqrt(x)", ["x"]),
            }[operation]
            cls.evaluators[operation] = E(expression).evaluator(
                [S(name) for name in names]
            )
        result = cls.evaluators[operation].evaluate_with_prec(
            [transport(value) for value in values], 32
        )[0]
        if not result.is_finite():
            raise ArithmeticError(f"non-finite double-double {operation}")
        return Decimal.__new__(cls, result)


@contextmanager
def arithmetic(format_name, precision):
    """Study-local, restored patches: sources, kernels, coefficients, reduction.

    Explicit modules only, all imported before patching. This context must not
    be used concurrently with another evaluator; the command is serial and
    runs under its own guarded subprocess. Fresh runtimes are loaded inside it.
    """
    modules = [
        importlib.import_module(f"pyamplicol.runtime.{name}")
        for name in (
            "_color_weight_exact",
            "_normalization_exact",
            "symbolica_exact",
            "correlated_exact",
            "correlations",
            "_color_topology_exact",
        )
    ]
    normalization = modules[1]
    scalar = {"double": Binary64, "double-double": DoubleDouble}.get(
        format_name, Decimal
    )
    patches = []
    normalization._pi.cache_clear()

    def working_precision(requested):
        return precision

    def upcast(value, requested):
        return transport(value)

    def study_pi(requested):
        return target_pi(scalar, precision)

    def unsupported_parameter_derivation(self, values, requested):
        raise RuntimeError(
            "This study's low-precision adapters require built-in SM kernels "
            "without a derived model-parameter expression evaluator"
        )

    try:
        if scalar is not Decimal:
            for module in modules:
                for name, value in list(vars(module).items()):
                    replacement = None
                    if value is Decimal:
                        replacement = scalar
                    elif type(value) is Decimal:
                        replacement = scalar(value)
                    elif name == "_working_precision":
                        replacement = working_precision
                    elif name == "_upcast_decimal":
                        replacement = upcast
                    if replacement is not None:
                        patches.append((module, name, value))
                        setattr(module, name, replacement)
            patches.append((normalization, "_pi", normalization._pi))
            normalization._pi = lru_cache(maxsize=1)(study_pi)
            exact = modules[2]
            expression_class = exact._ExactExpressionEvaluator
            patches.append((expression_class, "evaluate", expression_class.evaluate))
            expression_class.evaluate = unsupported_parameter_derivation
            if scalar is Binary64:
                evaluator_class = exact._ExactEvaluator
                original_stage = evaluator_class._evaluate_prepared

                def binary64_stage(self, values, requested):
                    if self.evaluator is None:
                        return original_stage(self, values, requested)
                    # with_prec(...,16) uses arbitrary Float, NOT binary64.
                    outputs = self.evaluator.evaluate_complex(
                        [complex(float(real), float(imag)) for real, imag in values]
                    )[0]
                    return tuple(
                        (Binary64(float(value.real)), Binary64(float(value.imag)))
                        for value in outputs
                    )

                patches.append((evaluator_class, "_evaluate_prepared", original_stage))
                evaluator_class._evaluate_prepared = binary64_stage
        with localcontext() as context:
            context.prec = precision + 8 if scalar is Decimal else 80
            yield scalar
    finally:
        for module, name, value in reversed(patches):
            setattr(module, name, value)
        normalization._pi.cache_clear()


def arithmetic_smoke(format_name, *, kernels=False):
    """Reject f64/MP mislabeled as DD before generating any study process."""
    scalar = {"double": Binary64, "double-double": DoubleDouble}[format_name]
    tiny = "1e-20"
    observed = (scalar(1) + scalar(tiny)) - scalar(1)
    if format_name == "double":
        assert observed == 0
    else:
        assert abs(Decimal(observed) / Decimal(tiny) - 1) < Decimal("1e-10")
        assert (scalar(1) + scalar("1e-40")) - scalar(1) == 0
    result = {"format": format_name, "one_plus_1e_20_minus_one": str(observed)}
    if kernels:
        from symbolica import E, S

        from pyamplicol.runtime.symbolica_exact import _ExactEvaluator

        evaluator = E("x+y").evaluator([S("x"), S("y")])
        precision = FORMATS[format_name][0]
        exact = _ExactEvaluator(input_len=2, evaluator=evaluator)
        with arithmetic(format_name, precision):
            observed_kernel = Decimal(
                exact.evaluate(
                    ((scalar(1), scalar(0)), (scalar(tiny), scalar(0))), precision
                )[0][0]
            )
            halfway = exact.evaluate(
                ((scalar(1), scalar(0)), (scalar(2.0**-53), scalar(0))), precision
            )[0][0]
            if format_name == "double":
                assert halfway == 1, "complex stage is not binary64 ties-to-even"
            else:
                assert halfway > 1, "complex stage lost the DoubleFloat half-ulp"
            result["complex_kernel_one_plus_half_ulp"] = str(halfway)
        with localcontext() as context:
            context.prec = 80
            residual = observed_kernel - 1
            if format_name == "double":
                assert residual == 0, "complex stage is not binary64"
            else:
                assert abs(residual / Decimal(tiny) - 1) < Decimal("1e-10"), (
                    "complex stage lost the DoubleFloat correction"
                )
        result["complex_kernel_one_plus_1e_20_minus_one"] = str(residual)
    return result


def dot(a, b):
    return a[0] * b[0] - sum(x * y for x, y in zip(a[1:], b[1:], strict=True))


def linear(*terms):
    return [sum(scale * vector[mu] for scale, vector in terms) for mu in range(4)]


def born_geometry():
    """The paper's rational generic 1 TeV Born point and transverse basis."""
    d, q = Decimal, Decimal(1000)
    n = (d(2) / 3, d(1) / 3, d(2) / 3)
    ta = (d(2) / 3, -d(2) / 3, -d(1) / 3)
    tb = (d(1) / 3, d(2) / 3, -d(2) / 3)
    incoming = [[q / 2, d(0), d(0), q / 2], [q / 2, d(0), d(0), -q / 2]]
    parent = [4 * q / 9] + [4 * q * a / 9 for a in n]
    spectator = [5 * q / 26] + [
        q * (-3 * a / 26 + 2 * b / 13) for a, b in zip(n, tb, strict=True)
    ]
    untouched = [85 * q / 234] + [
        q * (-77 * a / 234 - 2 * b / 13) for a, b in zip(n, tb, strict=True)
    ]
    e1 = [d(0), *ta]
    e2 = [d(1) / 2] + [a / 2 + b for a, b in zip(n, tb, strict=True)]
    return [*incoming, parent, spectator, untouched], e1, e2


def constraint_checks(point):
    q = Decimal(1000)
    return {
        "mass_shell_over_s": str(max(abs(dot(p, p)) for p in point) / q**2),
        "conservation_over_Q": str(
            max(
                abs(point[0][mu] + point[1][mu] - sum(p[mu] for p in point[2:]))
                for mu in range(4)
            )
            / q
        ),
        "positive_energies": all(p[0] > 0 for p in point),
    }


def forward_map(point):
    pi, pj, pk = point[2:5]
    ij, ik, jk = dot(pi, pj), dot(pi, pk), dot(pj, pk)
    y = ij / (ij + ik + jk)
    z = ik / (ik + jk)
    mapped = [
        *point[:2],
        linear((1, pi), (1, pj), (-y / (1 - y), pk)),
        linear((1 / (1 - y), pk)),
        point[5],
    ]
    return mapped, y, z, 2 * ij


def make_point(limit, exponent, digits=1280):
    """One high-precision source point; no target arithmetic is used here."""
    if limit not in {"collinear", "soft"} or exponent < 1 or digits < 16:
        raise ValueError("invalid limit, exponent, or source precision")
    with localcontext() as context:
        context.prec = max(digits + 80, 2 * exponent + 80)
        born, e1, e2 = born_geometry()
        parent, spectator = born[2:4]
        delta, q = Decimal(10) ** -exponent, Decimal(1000)
        e = linear((Decimal(3) / 5, e1), (Decimal(4) / 5, e2))
        if limit == "collinear":
            z = Decimal(3) / 5
            kt = delta * q
            transverse = linear((kt, e))
            y = kt**2 / (2 * z * (1 - z) * dot(parent, spectator))
        else:
            r = [q / 2] + [q * a / 2 for a in e1[1:]]
            alpha = dot(r, spectator) / dot(parent, spectator)
            beta = dot(r, parent) / dot(parent, spectator)
            r_perp = linear((1, r), (-alpha, parent), (-beta, spectator))
            z = 1 - delta * alpha
            y = delta * beta / z
            transverse = linear((-delta, r_perp))
        if not (0 < z < 1 and 0 < y < 1):
            raise ValueError("trajectory outside the physical inverse-map domain")
        pi = linear((z, parent), (y * (1 - z), spectator), (1, transverse))
        pj = linear((1 - z, parent), (y * z, spectator), (-1, transverse))
        pk = linear((1 - y, spectator))
        real = [*born[:2], pi, pj, pk, born[4]]
        mapped, actual_y, actual_z, sij = forward_map(real)
        checks = constraint_checks(real)
        checks["mapped_born_over_Q"] = str(
            max(
                abs(a - b)
                for p, r in zip(mapped, born, strict=True)
                for a, b in zip(p, r, strict=True)
            )
            / q
        )
        checks["y_error"] = str(abs(actual_y - y))
        checks["z_error"] = str(abs(actual_z - z))
        with localcontext() as output:
            output.prec = digits

            def strings(vectors):
                return [[str(+value) for value in p] for p in vectors]

            return {
                "limit": limit,
                "exponent": exponent,
                "delta": str(delta),
                "real": strings(real),
                "born": strings(born),
                "spin": strings([e])[0],
                "basis": strings([e1, e2]),
                "s_ij": str(+sij),
                "z": str(+z),
                "y": str(+y),
                "checks": checks,
            }


@lru_cache(maxsize=8)
def target_pi(scalar, precision):
    if scalar is Binary64:
        return scalar(math.pi)
    from symbolica import E

    # Pi is a constant input, rounded once into the target arithmetic. Asking
    # Expression.evaluate(...,32) would perform arbitrary, not DD, arithmetic.
    working = 80 if scalar is DoubleDouble else precision + 8
    return scalar(E("pi").evaluate({}, decimal_digit_precision=working)[0])


def counterterm(source, values, scalar, pi):
    """All inputs include public normalization; remove only final factorials."""
    b = scalar(6) * scalar(values["born"].real)
    prefactor = scalar(8) * pi * scalar(ALPHA_S)
    if source["limit"] == "collinear":
        z, sij = scalar(source["z"]), scalar(source["s_ij"])
        spin = scalar(6) * scalar(values["spin"].real)
        a = z / (scalar(1) - z) + (scalar(1) - z) / z
        return (
            prefactor
            / sij
            * scalar(6)
            * (a * b + scalar(2) * z * (scalar(1) - z) * spin)
        )
    born = [[scalar(v) for v in p] for p in source["born"]]
    soft = [scalar(v) for v in source["real"][3]]
    total = scalar(0)
    for a, b in DIPOLES:
        pa, pb = born[a - 1], born[b - 1]
        weight = dot(pa, pb) / (dot(pa, soft) * dot(pb, soft))
        total += weight * scalar(6) * scalar(values[f"T{a}{b}"].real)
    return -prefactor * total


def declarations():
    from pyamplicol import ColorCorrelator, CorrelatorConfig

    return CorrelatorConfig(
        color_correlations=tuple(
            ColorCorrelator.dipole(f"T{a}{b}", a, b) for a, b in (*DIPOLES, (3, 3))
        ),
        spin_correlations=((3,),),
    )


def correlation_requests(sources, diagnostics=False):
    from pyamplicol import CorrelatedRequest

    requests = {
        "born": CorrelatedRequest("born", spin_vectors={}),
        "spin": CorrelatedRequest(
            "born", spin_vectors={3: [source["spin"] for source in sources]}
        ),
        **{
            f"T{a}{b}": CorrelatedRequest(f"T{a}{b}", spin_vectors={})
            for a, b in DIPOLES
        },
    }
    if diagnostics:
        requests["T33"] = CorrelatedRequest("T33", spin_vectors={})
        for name, key in (("basis1", 0), ("basis2", 1)):
            requests[name] = CorrelatedRequest(
                "born", spin_vectors={3: [source["basis"][key] for source in sources]}
            )
        requests["ward"] = CorrelatedRequest(
            "born", spin_vectors={3: [source["born"][2] for source in sources]}
        )
    return requests


def generate_artifacts(directory):
    """One standard generation per missing output; never overwrite an artifact."""
    import tomllib

    from pyamplicol import Generator, ModelSource
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        RunConfig,
    )

    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full", contraction="direct"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )
    paths = {name: directory / name for name in EXPRESSIONS}
    identities = {}
    for name, path in paths.items():
        if not (path / "artifact.json").exists():
            print(f"Generating {name}: {EXPRESSIONS[name]}", flush=True)
            Generator(config).generate(
                EXPRESSIONS[name],
                path,
                model=ModelSource.built_in_sm(),
                correlators=declarations() if name == "born" else None,
            )
        manifest = json.loads((path / "artifact.json").read_text())
        effective = tomllib.loads((path / "config" / "effective.toml").read_text())
        if (
            len(manifest["processes"]) != 1
            or manifest["processes"][0]["expression"] != EXPRESSIONS[name]
            or effective["color"]["accuracy"] != "full"
            or effective["color"]["contraction"] != "direct"
            or effective["evaluator"]["execution_mode"] != "compiled"
            or effective["generation"]["relation_discovery"]["mode"]
            != STUDY["relation_discovery"][name]
        ):
            raise ValueError(
                f"Incompatible study artifact {path}; choose a fresh directory"
            )
        if name == "born":
            catalogue = json.loads((path / "correlators.json").read_text())
            expected = declarations().to_json_dict()
            if (
                catalogue["declarations"] != expected
                or not catalogue.get("processes")
                or any(
                    process.get("declarations") != expected
                    for process in catalogue["processes"].values()
                )
            ):
                raise ValueError(
                    f"Incompatible correlation declarations in {path}; "
                    "choose a fresh artifact directory"
                )
        identities[name] = manifest["artifact_id"]
    return paths, identities


def evaluate_real_batch(runtime, sources, format_name, precision, scalar):
    """Normal path: one batch. Bisect only after a genuine batch exception."""
    points = [source["real"] for source in sources]
    if format_name == "double":
        points = [[[float(v) for v in p] for p in point] for point in points]
    calls, errors = 0, {}

    def run(batch, start):
        nonlocal calls
        calls += 1
        try:
            if format_name == "double":
                values = runtime.evaluate(batch, precision=16)
                result = [scalar(complex(value).real) for value in values]
            else:
                # Sum resolved DD values in DD, not an exact Decimal helper.
                resolved = runtime.evaluate_resolved(batch, precision=precision)
                result = [
                    sum((scalar(v) for row in point for v in row), scalar(0))
                    for point in resolved.values
                ]
            if len(result) != len(batch):
                raise ValueError("real evaluator returned an inconsistent batch size")
            for index, value in enumerate(result):
                if not value.is_finite() or value <= 0:
                    errors[start + index] = "non-finite or nonpositive real result"
                    result[index] = None
            return result
        except Exception as exc:
            if len(batch) == 1:
                errors[start] = f"{type(exc).__name__}: {exc}"
                return [None]
            middle = len(batch) // 2
            return run(batch[:middle], start) + run(batch[middle:], start + middle)

    values = run(points, 0)
    return values, {"calls": calls, "diagnostic_bisection": calls > 1, "errors": errors}


def validate_correlations(values, precision):
    """Dimensionless checks; no high-multiplicity real ME is needed here."""
    with localcontext() as context:
        context.prec = max(40, precision + 8)
        b = Decimal(values["born"].real)
        if b <= 0:
            raise AssertionError("Born result must be positive")
        tolerance = Decimal(10) ** (-min(20, max(8, precision - 6)))
        checks = {
            "completeness": abs(
                (Decimal(values["basis1"].real) + Decimal(values["basis2"].real)) / b
                - 1
            ),
            "ward": abs(Decimal(values["ward"].real) / (b * Decimal(1000) ** 2)),
            "casimir": abs(Decimal(values["T33"].real) / b - 3),
            "imaginary": max(abs(Decimal(v.imag)) for v in values.values()) / b,
        }
        for leg in range(1, 6):
            others = [
                tuple(sorted((leg, other))) for other in range(1, 6) if other != leg
            ]
            checks[f"coherence_{leg}"] = abs(
                sum((Decimal(values[f"T{a}{b}"].real) for a, b in others), 3 * b) / b
            )
        if any(value > tolerance for value in checks.values()):
            raise AssertionError(f"Born correlation checks failed: {checks}")
        anisotropy = abs(Decimal(values["spin"].real) / b - Decimal("0.5"))
        if anisotropy < Decimal("1e-8"):
            raise AssertionError("chosen Born/vector does not resolve spin anisotropy")
        return {
            **{key: str(value) for key, value in checks.items()},
            "spin_anisotropy": str(anisotropy),
        }


def evaluate_batch(paths, sources, format_name, precision, *, diagnostics=False):
    from pyamplicol import Runtime

    with arithmetic(format_name, precision) as scalar:
        started = time.perf_counter()
        real = Runtime.load(
            paths["real"],
            model_parameters={
                "alpha_s": float(ALPHA_S),
            },
        )
        born = Runtime.load(
            paths["born"],
            model_parameters={
                "alpha_s": float(ALPHA_S),
            },
        )
        requests = correlation_requests(sources, diagnostics=diagnostics)
        loaded = time.perf_counter()
        # One public call: every requested colour/spin contraction, every point.
        results = born.evaluate_correlated_many(
            [source["born"] for source in sources],
            requests,
            precision=precision,
        )
        correlated_done = time.perf_counter()
        # Public named access: results["T13"][point_index].real / .imag.
        real_values, real_diagnostics = evaluate_real_batch(
            real,
            sources,
            format_name,
            precision,
            scalar,
        )
        real_done = time.perf_counter()
        pi = target_pi(scalar, precision)
        records = []
        for index, (source, real_value) in enumerate(
            zip(sources, real_values, strict=True)
        ):
            components = {name: values[index] for name, values in results.items()}
            record = {
                "limit": source["limit"],
                "exponent": source["exponent"],
                "format": format_name,
                "precision": precision,
                "delta": source["delta"],
                "source_checks": source["checks"],
                "correlations": {
                    name: {"real": str(value.real), "imag": str(value.imag)}
                    for name, value in components.items()
                },
                "real_batch": real_diagnostics,
            }
            if diagnostics:
                record["correlation_checks"] = validate_correlations(
                    components, precision
                )
            try:
                if real_value is None:
                    raise ArithmeticError(real_diagnostics["errors"][index])
                r = scalar(24) * real_value
                c = counterterm(source, components, scalar, pi)
                f = c - r
                if not all(value.is_finite() for value in (r, c, f)):
                    raise ArithmeticError("non-finite subtraction")
                record.update(
                    {
                        "status": "ok",
                        "R": str(r),
                        "C": str(c),
                        "F": str(f),
                        "C_over_R": str(c / r),
                        "F_over_R": str(f / r),
                    }
                )
                if source["limit"] == "collinear":
                    z, sij = scalar(source["z"]), scalar(source["s_ij"])
                    pgg = scalar(6) * (z / (1 - z) + (1 - z) / z + z * (1 - z))
                    averaged = scalar(8) * pi * scalar(ALPHA_S) / sij * pgg
                    averaged *= scalar(6) * scalar(components["born"].real)
                    record["spin_averaged_control_over_R"] = str((averaged - r) / r)
            except (ArithmeticError, ValueError) as exc:
                record.update({"status": "failed", "error": str(exc)})
            records.append(record)
        batch = {
            "point_count": len(sources),
            "request_count": len(requests),
            "public_correlated_many_calls": 1,
            "public_real_calls": real_diagnostics["calls"],
            "runtime_load_seconds": loaded - started,
            "correlations_seconds": correlated_done - loaded,
            "real_seconds": real_done - correlated_done,
            "combination_seconds": time.perf_counter() - real_done,
        }
        for record in records:
            record["batch"] = batch
        # Adapter-specific caches cannot survive into the next format.
        del born, real
        return records


def record_key(record):
    return f"{record['limit']}:{record['format']}:{record['exponent']}"


def pending_exponents(report, limit, format_name, exponents, *, retry_failed=False):
    """A measured failure is data, not an implicit request for endless retries."""
    return [
        x
        for x in exponents
        if f"{limit}:{format_name}:{x}" not in report["records"]
        or (
            retry_failed
            and report["records"][f"{limit}:{format_name}:{x}"]["status"] != "ok"
        )
    ]


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compare_records(record, reference):
    if reference is None or reference.get("status") != "ok":
        return {"reference_status": "missing"}
    if record.get("status") != "ok":
        return {"reference_status": "ok", "stable_digits": "0", "stable_fraction": 0.0}
    with localcontext() as context:
        context.prec = int(reference["precision"]) + 30
        expected, observed = Decimal(reference["F"]), Decimal(record["F"])
        if expected == 0:
            return {"reference_status": "zero-remainder"}
        error = abs((observed - expected) / expected)
        tracked = Decimal(str(FORMATS[record["format"]][1]))
        digits = (
            tracked if error == 0 else min(tracked, max(Decimal(0), -error.log10()))
        )
        return {
            "reference_status": "ok",
            "epsilon_F": str(error),
            "stable_digits": str(digits),
            "stable_fraction": float(digits / tracked),
        }


def update_comparisons(report):
    for record in report["records"].values():
        if record["format"] != "oracle":
            reference = report["records"].get(
                f"{record['limit']}:oracle:{record['exponent']}"
            )
            record.update(compare_records(record, reference))


def checkpoint(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def validate_factorization(report):
    """Only the well-resolved oracle determines the physics convergence test."""
    for limit in ("collinear", "soft"):
        points = sorted(
            (
                r
                for r in report["records"].values()
                if r["limit"] == limit
                and r["format"] == "oracle"
                and r["status"] == "ok"
            ),
            key=lambda r: r["exponent"],
        )
        if len(points) < 2:
            continue
        errors = [abs(Decimal(point["F_over_R"])) for point in points]
        if errors[-1] >= errors[0] / 3 or errors[-1] >= Decimal("0.01"):
            raise AssertionError(f"{limit} counterterm does not converge: {errors}")


def validate_requested_oracles(report, limits, exponents):
    """A completed scan needs references; partial render-only requests do not."""
    missing = [
        key
        for limit in limits
        for exponent in exponents
        if report["records"].get(key := f"{limit}:oracle:{exponent}", {}).get("status")
        != "ok"
    ]
    if missing:
        raise ValueError(
            "Missing or failed requested oracle measurements: "
            + ", ".join(missing)
            + ". Checkpoints are retained; use --retry-failed to retry failures "
            "or --render to redraw available results."
        )


def render(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "double": ("Double precision", "#cc6600", "--", "o"),
        "double-double": ("DoubleFloat (31 digits)", "#2767ad", "-.", "s"),
        "arbitrary-1000": ("1000 decimal digits", "#23834a", "-", "D"),
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(10.5, 4), sharey=True, layout="constrained"
    )
    for axis, limit in zip(axes, ("collinear", "soft"), strict=True):
        for name in FORMATS:
            points = sorted(
                (
                    r
                    for r in report["records"].values()
                    if r["limit"] == limit
                    and r["format"] == name
                    and "stable_fraction" in r
                    and 1 <= r["exponent"] <= 15
                ),
                key=lambda r: r["exponent"],
            )
            if points:
                label, color, linestyle, marker = styles[name]
                axis.plot(
                    [p["exponent"] for p in points],
                    [p["stable_fraction"] for p in points],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                    marker=marker,
                    markerfacecolor="white",
                    markersize=3.4,
                    linewidth=1.5,
                    markeredgewidth=0.8,
                )
        axis.set(
            title={"collinear": "Fixed-azimuth collinear", "soft": "Wide-angle soft"}[
                limit
            ],
            xlabel=r"$-\log_{10}\delta$",
            xlim=(0.8, 15.2),
            ylim=(-0.012, 1.03),
        )
        axis.set_xticks(range(1, 16, 2))
        axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axis.grid(color="0.86", linewidth=0.45)
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel("Stable-digit fraction")
    axes[1].legend(
        loc="center right",
        bbox_to_anchor=(0.985, 0.54),
        frameon=False,
        fontsize=8.5,
        handlelength=2.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(figure)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--artifact-dir", type=Path, default=Path(".artifacts/nlo-limits")
    )
    result.add_argument(
        "--output", type=Path, default=Path(".artifacts/nlo-limits/results.json")
    )
    result.add_argument(
        "--formats", nargs="+", choices=tuple(FORMATS), default=list(FORMATS)
    )
    result.add_argument(
        "--limits",
        nargs="+",
        choices=("collinear", "soft"),
        default=["collinear", "soft"],
    )
    result.add_argument("--exponents", nargs="+", type=int, default=list(range(1, 16)))
    result.add_argument("--oracle-precision", type=int, default=1200)
    result.add_argument("--memory-limit-gib", type=float, default=10.0)
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate/reuse process artifacts without evaluating the scan",
    )
    result.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry measured failures; default preserves them",
    )
    result.add_argument(
        "--smoke", action="store_true", help="Only x=2,3,4; check physics identities"
    )
    result.add_argument(
        "--self-test", action="store_true", help="Kinematics only; no Symbolica"
    )
    result.add_argument(
        "--render",
        action="store_true",
        help="Render retained JSON only; no evaluations",
    )
    result.add_argument(
        "--plot", type=Path, help="PDF/PNG output stem; defaults next to results"
    )
    result.add_argument("--_guard-child", action="store_true", help=argparse.SUPPRESS)
    return result


def self_test():
    for limit in ("collinear", "soft"):
        for exponent in (1, 4, 15, 100):
            source = make_point(limit, exponent, digits=300)
            for key, value in source["checks"].items():
                if key == "positive_energies":
                    assert value
                else:
                    assert Decimal(value) < Decimal("1e-175"), (limit, exponent, key)
    arithmetic_smoke("double")
    print("Kinematics and binary64 arithmetic checks passed.")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    if args.self_test:
        self_test()
        return 0
    if args.render:
        render(
            json.loads(args.output.read_text()),
            args.plot or args.output.with_suffix(""),
        )
        return 0
    if (
        args.oracle_precision <= max(FORMATS[name][0] for name in args.formats)
        or min(args.exponents) < 1
    ):
        raise ValueError(
            "oracle precision must exceed every selected target; "
            "exponents must be positive"
        )
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0:
        raise ValueError("a positive finite process-tree memory guard is required")
    if not args._guard_child:
        watchdog = ROOT / "tools" / "ci" / "memory_watchdog.py"
        return subprocess.call(
            [
                sys.executable,
                str(watchdog),
                "--limit-gib",
                str(args.memory_limit_gib),
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--_guard-child",
            ]
        )
    if args.prepare_only:
        started = time.perf_counter()
        _, identities = generate_artifacts(args.artifact_dir)
        print(
            json.dumps(
                {
                    "artifacts": identities,
                    "git_revision": git_revision(),
                    "preparation_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        return 0
    arithmetic_checks = []
    for name in args.formats:
        if name in ("double", "double-double"):
            check = arithmetic_smoke(name, kernels=True)
            arithmetic_checks.append(check)
            print(json.dumps(check), flush=True)
    exponents = [2, 3, 4] if args.smoke else sorted(set(args.exponents))
    report = {"study": STUDY, "oracle_precision": args.oracle_precision, "records": {}}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if (
            report["study"] != STUDY
            or report["oracle_precision"] != args.oracle_precision
        ):
            raise ValueError(
                "Existing report has a different study definition; choose --output"
            )
    report["arithmetic_checks"] = arithmetic_checks
    report["git_revision"] = git_revision()
    prepared = time.perf_counter()
    paths, identities = generate_artifacts(args.artifact_dir)
    report["preparation_seconds"] = time.perf_counter() - prepared
    if report.get("artifacts", identities) != identities:
        raise ValueError(
            "Report belongs to different process artifacts; choose --output"
        )
    report["artifacts"] = identities
    for name in ["oracle", *args.formats]:
        precision = args.oracle_precision if name == "oracle" else FORMATS[name][0]
        for limit in args.limits:
            missing = pending_exponents(
                report, limit, name, exponents, retry_failed=args.retry_failed
            )
            if not missing:
                continue
            print(f"Evaluating {limit}, {name}: x={missing}", flush=True)
            sources = [
                make_point(limit, x, digits=args.oracle_precision + 80) for x in missing
            ]
            records = evaluate_batch(
                paths, sources, name, precision, diagnostics=args.smoke
            )
            for record in records:
                record["git_revision"] = report["git_revision"]
                report["records"][record_key(record)] = record
            update_comparisons(report)
            checkpoint(args.output, report)
    validate_requested_oracles(report, args.limits, exponents)
    validate_factorization(report)
    render(report, args.plot or args.output.with_suffix(""))
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

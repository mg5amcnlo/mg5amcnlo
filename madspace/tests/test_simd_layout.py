"""Results that must not depend on how the CPU runtime lays out tensors for the
SIMD kernels.

The vectorized kernels load and store the lanes of a vector contiguously, so
they only run if every batched argument has batch stride one or is a literal
stored padded to the vector size; everything else (row-major inputs from numpy
or torch, broadcast tensors from fb.full, large literal arrays) is handled by
the scalar kernels. Inputs marked single in the instruction set have their own
kernel type. These tests run in every SIMD mode (tests/run_simd_modes.py) and
compare against numpy, so all of these paths have to agree. They also cover the
constant folding rules that keep broadcast tensors away from the kernels.

Except for the gradient tests, which also cover kernel code shared with the GPU,
these are CPU runtime or graph building tests and are marked cpu_only.
"""

import numpy as np
import pytest
import torch
from madspace_testing import new_context
from pytest import approx

import madspace as ms
from madspace.torch import FunctionModule

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(4321)

# not a multiple of any SIMD vector size, so the scalar tail runs as well
BATCH_SIZE = 1003


def build(input_types, output_types, body):
    fb = ms.FunctionBuilder(ms.NamedTypes(input_types), ms.NamedTypes(output_types))
    outputs = body(fb, *(fb.input(i) for i in range(len(input_types))))
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    for i, output in enumerate(outputs):
        fb.output(i, output)
    return fb.function()


def call(function, *args):
    outputs = ms.FunctionRuntime(function)(*args)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [np.asarray(output) for output in outputs]


def instruction_names(function):
    return [instr.instruction.name for instr in function.instructions]


# constant folding


@pytest.mark.cpu_only
def test_sub_with_constant_first_is_not_folded():
    func = build(
        [("x", ms.batch_float)],
        [("y", ms.batch_float)],
        lambda fb, x: fb.sub(0.0, x),
    )
    assert instruction_names(func) == ["sub"]
    x = rng.normal(size=BATCH_SIZE)
    (y,) = call(func, x)
    assert y == approx(-x)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "op, const_first, const",
    [
        ("add", False, 0.0),
        ("add", True, 0.0),
        ("sub", False, 0.0),
        ("mul", False, 1.0),
        ("mul", True, 1.0),
    ],
)
def test_identities_are_folded(op, const_first, const):
    def body(fb, x):
        args = (const, x) if const_first else (x, const)
        return getattr(fb, op)(*args)

    func = build([("x", ms.batch_float)], [("y", ms.batch_float)], body)
    assert instruction_names(func) == []


@pytest.mark.cpu_only
def test_stack_of_literals_is_folded():
    def body(fb, x):
        stacked = fb.stack([1.0, 2.0, 3.0])
        assert stacked.literal_value == ([3], [1.0, 2.0, 3.0])
        return fb.mul(x, stacked)

    func = build(
        [("x", ms.batch_float_array(3))], [("y", ms.batch_float_array(3))], body
    )
    assert instruction_names(func) == ["mul_float64"]


@pytest.mark.cpu_only
def test_single_flavor_integrand_has_no_constant_gathers(tmp_path):
    """With a single flavor the flavor index is a constant tensor: the flavor
    tables (remap, factors, mirror) must be resolved when building the graph
    instead of gathering from them, and a weight that is one by construction
    must not be multiplied."""
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = tmp_path / "alphas.info"
    alphas_file.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n"
    )
    running_coupling = ms.RunningCoupling(ms.AlphaSGrid(str(alphas_file)))
    matrix_element = ms.MatrixElement(
        0xBADCAFE,
        5,
        ms.Integrand.matrix_element_inputs,
        ms.Integrand.matrix_element_outputs,
        1,
        True,
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=1000.0,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
    )
    for mirror in (False, True):
        mapping = ms.PhaseSpaceMapping(
            [0.0] * 5, 1000.0, leptonic=True, mirror_beams=mirror
        )
        integrand = ms.Integrand(
            mapping,
            [diff_xs],
            running_coupling=running_coupling,
            energy_scale=ms.EnergyScale(5, 91.188),
            channel_indices=[0],
            flavor_remap=[0],
            flavor_factors=[2.0],
            flavor_mirror=[mirror],
        )
        func = integrand.function()
        names = instruction_names(func)
        assert "gather" not in names
        assert "gather_int" not in names
        sources = {}
        for instr in func.instructions:
            for output in instr.outputs:
                sources[output.local_index] = instr.instruction.name
        for instr in func.instructions:
            if instr.instruction.name == "mul_float64":
                assert all(
                    sources.get(arg.local_index) != "full" for arg in instr.inputs
                )


# memory layouts


@pytest.mark.cpu_only
@pytest.mark.parametrize("order", ["C", "F"])
def test_row_major_and_column_major_inputs(order):
    """Column-major (batch stride one) inputs run the vectorized kernels,
    row-major ones the scalar kernels."""
    func = build(
        [("x", ms.batch_float_array(3)), ("y", ms.batch_float_array(3))],
        [("z", ms.batch_float_array(3))],
        lambda fb, x, y: fb.add(fb.mul(x, y), fb.sqrt(fb.square(y))),
    )
    x = np.asarray(rng.normal(size=(BATCH_SIZE, 3)), order=order)
    y = np.asarray(rng.normal(size=(BATCH_SIZE, 3)), order=order)
    (z,) = call(func, x, y)
    assert z == approx(x * y + np.abs(y))


@pytest.mark.cpu_only
@pytest.mark.parametrize("size", [3, 8, 100])
def test_literal_arrays(size):
    """Small literal arrays are stored padded to the vector size, large ones
    (more than 256 stored elements) are not and fall back to the scalar
    kernels."""
    values = list(rng.normal(size=size))
    func = build(
        [("x", ms.batch_float_array(size))],
        [("y", ms.batch_float_array(size))],
        lambda fb, x: fb.add(fb.mul(x, fb.stack(values)), fb.stack([1.5] * size)),
    )
    x = np.asfortranarray(rng.normal(size=(BATCH_SIZE, size)))
    (y,) = call(func, x)
    assert y == approx(x * np.array(values) + 1.5)


@pytest.mark.cpu_only
def test_literal_output_is_not_padded():
    func = build(
        [("x", ms.batch_float)],
        [("y", ms.batch_float_array(3))],
        lambda fb, x: fb.stack([1.0, 2.0, 3.0]),
    )
    (y,) = call(func, rng.normal(size=BATCH_SIZE))
    assert y.reshape(-1, 3)[0] == approx([1.0, 2.0, 3.0])
    # the padded storage would give the inner dimension a stride of the vector size
    assert y.strides[-1] == y.itemsize


@pytest.mark.cpu_only
def test_broadcast_full_input():
    """fb.full creates a batch-sized tensor with batch stride zero."""

    def body(fb, x):
        full = fb.full([2.5, fb.batch_size([x])])
        return fb.mul(x, full)

    func = build([("x", ms.batch_float)], [("y", ms.batch_float)], body)
    x = rng.normal(size=BATCH_SIZE)
    (y,) = call(func, x)
    assert y == approx(2.5 * x)


@pytest.mark.cpu_only
def test_cut_all_and_cut_any_with_scalar_bounds():
    func = build(
        [("obs", ms.batch_float_array(4))],
        [("all", ms.batch_float), ("any", ms.batch_float)],
        lambda fb, obs: (fb.cut_all(obs, -2.0, 2.0), fb.cut_any(obs, -2.0, 2.0)),
    )
    obs = np.asfortranarray(rng.uniform(-2.5, 2.5, size=(BATCH_SIZE, 4)))
    w_all, w_any = call(func, obs)
    inside = (obs >= -2.0) & (obs <= 2.0)
    assert w_all == approx(np.all(inside, axis=1).astype(float))
    assert w_any == approx(np.any(inside, axis=1).astype(float))


# single inputs


@pytest.mark.cpu_only
def test_single_input_rejects_batched_argument():
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("r", ms.batch_float)]),
        ms.NamedTypes([("s", ms.batch_float), ("gs", ms.batch_float)]),
    )
    r = fb.input(0)
    with pytest.raises(ValueError, match="cannot have batch dimension"):
        fb.breit_wigner_invariant(r, r, 1.0, 0.0, 100.0)


@pytest.mark.cpu_only
def test_single_inputs():
    """Breit-Wigner sampling with literal (single) mass and width, compared
    between its forward and inverse direction."""
    mass, width, s_min, s_max = 91.0, 2.5, 50.0**2, 150.0**2

    def body(fb, r):
        s, gs = fb.breit_wigner_invariant(r, mass, width, s_min, s_max)
        r_out, gs_inv = fb.breit_wigner_invariant_inverse(s, mass, width, s_min, s_max)
        return r_out, gs, gs_inv

    func = build(
        [("r", ms.batch_float)],
        [("r_out", ms.batch_float), ("gs", ms.batch_float), ("gs_inv", ms.batch_float)],
        body,
    )
    r = rng.uniform(size=BATCH_SIZE)
    r_out, gs, gs_inv = call(func, r)
    assert r_out == approx(r)
    assert gs * gs_inv == approx(np.ones(BATCH_SIZE))


# gradients


def test_global_probability_gradient_is_summed_over_events():
    """The gradient of a broadcast (global) probability vector is the sum of the
    contributions of all events."""
    ctx = new_context()
    ctx.define_global("test_simd_layout_probs", ms.DataType.float64, [4], True)
    ctx.get_global("test_simd_layout_probs").torch()[:] = torch.tensor(
        [[0.1, 0.4, 0.3, 0.2]]
    )
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("index", ms.batch_int)]),
        ms.NamedTypes([("r", ms.batch_float), ("det", ms.batch_float)]),
    )
    probs = fb.get_global("test_simd_layout_probs", ms.DataType.float64, [4])
    r, det = fb.sample_discrete_probs_inverse(fb.input(0), probs)
    fb.output(0, r)
    fb.output(1, det)
    module = FunctionModule(fb.function(), ctx)
    param = module.global_params["test_simd_layout_probs"]

    index = torch.tensor(rng.integers(0, 4, size=BATCH_SIZE), dtype=torch.int32)
    weights = torch.tensor(rng.uniform(size=BATCH_SIZE))

    def loss():
        return (module(index)[1] * weights).sum()

    loss().backward()
    grad = param.grad.clone()

    eps = 1e-6
    grad_fd = torch.zeros(4, dtype=torch.float64)
    with torch.no_grad():
        for i in range(4):
            param[0, i] += eps
            plus = loss()
            param[0, i] -= 2 * eps
            minus = loss()
            param[0, i] += eps
            grad_fd[i] = (plus - minus) / (2 * eps)
    assert grad[0].cpu().numpy() == approx(grad_fd.cpu().numpy(), rel=1e-5)


def test_broadcast_gradient_through_stack():
    """stack of a broadcast (global) and a batched value: the gradient of the
    global accumulates the whole batch."""
    ctx = new_context()
    ctx.define_global("test_simd_layout_scalar", ms.DataType.float64, [], True)
    ctx.get_global("test_simd_layout_scalar").torch()[:] = 1.5
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("x", ms.batch_float)]),
        ms.NamedTypes([("y", ms.batch_float_array(2))]),
    )
    g = fb.get_global("test_simd_layout_scalar", ms.DataType.float64, [])
    fb.output(0, fb.stack([g, fb.input(0)]))
    module = FunctionModule(fb.function(), ctx)
    param = module.global_params["test_simd_layout_scalar"]

    x = torch.tensor(rng.normal(size=BATCH_SIZE))
    weights = torch.tensor(rng.uniform(size=(BATCH_SIZE, 2)))
    (module(x) * weights).sum().backward()
    assert param.grad.item() == approx(weights[:, 0].sum().item())

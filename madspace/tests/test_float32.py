"""Single-precision NN instructions: forward and backward against torch."""

import pytest
import torch
from madspace_testing import DEVICE, default_context, new_context

import madspace as ms
from madspace.torch import FunctionModule


@pytest.fixture
def gen():
    """A fresh local generator per test, so results don't depend on test order."""
    return torch.Generator().manual_seed(1234)


def _draw(draw, *args, requires_grad=False, **kwargs):
    """Random numbers from the CPU generator, moved to the device of the tests."""
    values = draw(*args, device="cpu", **kwargs).to(DEVICE)
    return values.requires_grad_(requires_grad)


def _randn(*args, **kwargs):
    return _draw(torch.randn, *args, **kwargs)


def _rand(*args, **kwargs):
    return _draw(torch.rand, *args, **kwargs)


N = 103
DIMS_IN = 7
DIMS_OUT = 5

ACTIVATIONS = {
    "relu": torch.relu,
    "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, 0.01),
    "elu": torch.nn.functional.elu,
    "gelu": torch.nn.functional.gelu,
    "sigmoid": torch.sigmoid,
    "softplus": torch.nn.functional.softplus,
    "scaled_tanh": lambda x: 4 * torch.tanh(x / 4),
}

DTYPES = {
    "float32": (ms.DataType.float32, torch.float32, 1e-5),
    "float64": (ms.DataType.float64, torch.float64, 1e-12),
}


def batch_type(dtype, shape):
    return ms.Type(dtype, ms.batch_size, shape)


@pytest.mark.parametrize("precision", DTYPES.keys())
@pytest.mark.parametrize("name", ACTIVATIONS.keys())
def test_activation(name, precision, gen):
    ms_dtype, torch_dtype, tol = DTYPES[precision]
    io_type = batch_type(ms_dtype, [DIMS_IN])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type)]), ms.NamedTypes([("out", io_type)])
    )
    fb.output(0, getattr(fb, name)(fb.input(0)))
    module = FunctionModule(fb.function(), default_context())

    x = (3 * _randn((N, DIMS_IN), dtype=torch_dtype, generator=gen)).requires_grad_()
    out = module(x)
    assert out.dtype == torch_dtype
    out_grad = _randn(out.shape, dtype=out.dtype, generator=gen)
    (out * out_grad).sum().backward()

    x_ref = x.detach().clone().requires_grad_()
    out_ref = ACTIVATIONS[name](x_ref)
    (out_ref * out_grad).sum().backward()
    torch.testing.assert_close(out, out_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("precision", DTYPES.keys())
def test_matmul(precision, gen):
    ms_dtype, torch_dtype, tol = DTYPES[precision]
    fb = ms.FunctionBuilder(
        ms.NamedTypes(
            [
                ("x", batch_type(ms_dtype, [DIMS_IN])),
                ("w", ms.Type(ms_dtype, ms.BatchSize.one, [DIMS_OUT, DIMS_IN])),
                ("b", ms.Type(ms_dtype, ms.BatchSize.one, [DIMS_OUT])),
            ]
        ),
        ms.NamedTypes([("y", batch_type(ms_dtype, [DIMS_OUT]))]),
    )
    fb.output(0, fb.matmul(fb.input(0), fb.input(1), fb.input(2)))
    module = FunctionModule(fb.function(), default_context())

    x = _randn((N, DIMS_IN), dtype=torch_dtype, generator=gen, requires_grad=True)
    # single inputs are passed without batch dimension
    w = _randn(
        (DIMS_OUT, DIMS_IN), dtype=torch_dtype, generator=gen, requires_grad=True
    )
    b = _randn((DIMS_OUT,), dtype=torch_dtype, generator=gen, requires_grad=True)
    y = module(x, w, b)
    assert y.dtype == torch_dtype
    y_grad = _randn(y.shape, dtype=y.dtype, generator=gen)
    (y * y_grad).sum().backward()

    refs = [t.detach().clone().requires_grad_() for t in (x, w, b)]
    y_ref = refs[0] @ refs[1].T + refs[2]
    (y_ref * y_grad).sum().backward()
    torch.testing.assert_close(y, y_ref, rtol=tol, atol=tol)
    for t, t_ref in zip((x, w, b), refs):
        torch.testing.assert_close(t.grad, t_ref.grad, rtol=tol, atol=tol)


def test_convert_roundtrip(gen):
    in_type = batch_type(ms.DataType.float64, [3])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", in_type)]),
        ms.NamedTypes(
            [("single", batch_type(ms.DataType.float32, [3])), ("out", in_type)]
        ),
    )
    single = fb.convert_float32(fb.input(0))
    fb.output(0, single)
    fb.output(1, fb.convert_float64(fb.relu(single)))
    module = FunctionModule(fb.function(), default_context())

    x = _randn((N, 3), dtype=torch.float64, generator=gen, requires_grad=True)
    single_out, out = module(x)
    assert single_out.dtype == torch.float32
    assert out.dtype == torch.float64
    torch.testing.assert_close(single_out, x.detach().float())
    torch.testing.assert_close(out, torch.relu(x.detach().float()).double())
    out.sum().backward()
    torch.testing.assert_close(x.grad, (x.detach() > 0).double())


def test_literal_converted_for_float32(gen):
    io_type = batch_type(ms.DataType.float32, [1])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type)]), ms.NamedTypes([("out", io_type)])
    )
    # float64 literals are converted when the instruction dispatches to float32
    fb.output(0, fb.matmul(fb.input(0), ms.Value([2.0], [1, 1]), ms.Value([1.0])))
    module = FunctionModule(fb.function(), default_context())
    x = _randn((N, 1), dtype=torch.float32, generator=gen)
    torch.testing.assert_close(module(x), 2 * x + 1)


def test_flow_float32_subnets(tmp_path, gen):
    """A flow with float32 subnetworks matches the float64 one with the same
    (rounded) weights; the Jacobian and the output stay in float64."""
    # a private context, so that no globals are left in the default one
    ctx = new_context()
    flows = {
        dtype: ms.Flow(
            4, prefix=f"fp32_test_{name}", subnet_precision=dtype, invert_spline=False
        )
        for name, dtype in [("f32", ms.DataType.float32), ("f64", ms.DataType.float64)]
    }
    for flow in flows.values():
        flow.initialize_globals(ctx, 42)
    # randomize all weights (the last layers start at zero) and round to float32
    names_f64 = [n for n in ctx.global_names() if n.startswith("fp32_test_f64")]
    for name in names_f64:
        weights = ctx.get_global(name).torch()
        weights[:] = (
            0.3 * _randn(weights.shape, generator=gen, dtype=torch.float32).double()
        )
        ctx.get_global(name.replace("f64", "f32")).torch()[:] = weights.float()

    # outputs and gradients with respect to the input and the float32 weights
    r = _rand((N, 4), dtype=torch.float64, generator=gen)
    outputs, grads = {}, {}
    for dtype, flow in flows.items():
        module = FunctionModule(flow.forward_function(), ctx)
        r_grad = r.clone().requires_grad_()
        out, det = module(r_grad)
        assert out.dtype == det.dtype == torch.float64
        (out.sum() + det.log().sum()).backward()
        weight = next(p for n, p in module.global_params.items() if "weight" in n)
        assert weight.grad.dtype == weight.dtype
        outputs[dtype] = (out.detach(), det.detach())
        grads[dtype] = (r_grad.grad, weight.grad.double())
    for o32, o64 in zip(outputs[ms.DataType.float32], outputs[ms.DataType.float64]):
        torch.testing.assert_close(o32, o64, rtol=1e-3, atol=1e-4)
    for g32, g64 in zip(grads[ms.DataType.float32], grads[ms.DataType.float64]):
        torch.testing.assert_close(g32, g64, rtol=5e-3, atol=1e-4)

    # float32 globals survive saving and loading
    ctx.save_globals(str(tmp_path / "globals.tar"))
    loaded = new_context()
    loaded.load_globals(str(tmp_path / "globals.tar"))
    for name in names_f64:
        name32 = name.replace("f64", "f32")
        global32 = loaded.get_global(name32).torch()
        assert global32.dtype == torch.float32
        torch.testing.assert_close(global32, ctx.get_global(name32).torch())


def test_structural_ops_backward(gen):
    """stack, unstack, cat, batch_cat and batch_split propagate float32 gradients."""
    in_type = batch_type(ms.DataType.float32, [3])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", in_type)]),
        ms.NamedTypes([("out", batch_type(ms.DataType.float32, [12]))]),
    )
    x = fb.input(0)
    a, b = fb.unstack(fb.stack([x, fb.relu(x)]))
    c = fb.cat([a, b])
    combined, counts = fb.batch_cat([c, c])
    fb.output(0, fb.cat(fb.batch_split(combined, counts)))
    module = FunctionModule(fb.function(), default_context())

    x = _randn((N, 3), dtype=torch.float32, generator=gen, requires_grad=True)
    out = module(x)
    out_grad = _randn(out.shape, dtype=out.dtype, generator=gen)
    (out * out_grad).sum().backward()

    x_ref = x.detach().clone().requires_grad_()
    c_ref = torch.cat([x_ref, torch.relu(x_ref)], dim=1)
    out_ref = torch.cat([c_ref, c_ref], dim=1)
    (out_ref * out_grad).sum().backward()
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.parametrize("precision", DTYPES.keys())
@pytest.mark.parametrize("op", ["stack", "cat", "batch_cat"])
def test_structural_ops_backward_repeated_input(op, precision):
    """An input passed several times to one structural op gets every gradient.

    On the CPU each gradient add is a separate thread pool job, so adds into the
    same gradient tensor used to race and lose updates now and then. The batch is
    large and the input repeated many times so that racing adds overlap.
    """
    ms_dtype, torch_dtype, _ = DTYPES[precision]
    repeats = 16
    in_type = batch_type(ms_dtype, [3])
    out_shape = {"stack": [repeats, 3], "cat": [3 * repeats], "batch_cat": [3]}[op]
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", in_type)]),
        ms.NamedTypes([("out", batch_type(ms_dtype, out_shape))]),
    )
    x = fb.input(0)
    if op == "stack":
        out = fb.stack([x] * repeats)
    elif op == "cat":
        out = fb.cat([x] * repeats)
    else:
        out, _ = fb.batch_cat([x] * repeats)
    fb.output(0, out)
    module = FunctionModule(fb.function(), default_context())

    x = torch.zeros((100_000, 3), dtype=torch_dtype, requires_grad=True)
    for _ in range(5):
        x.grad = None
        module(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, repeats), rtol=0, atol=0)


def test_random_int_gather(gen):
    """random_int returns int32 indices, as used when sampling the replay buffer."""
    io_type = batch_type(ms.DataType.float64, [])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type)]), ms.NamedTypes([("out", io_type)])
    )
    size = fb.batch_size([fb.input(0)])
    fb.output(0, fb.batch_gather(fb.random_int(size, size), fb.input(0)))
    module = FunctionModule(fb.function(), default_context())
    x = _rand((N,), dtype=torch.float64, generator=gen)
    out = module(x)
    assert out.shape == x.shape
    assert torch.isin(out, x).all()

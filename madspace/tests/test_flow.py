import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from madspace_testing import default_context, generator, rand, randn
from pytest import approx

import madspace as ms
from madspace.torch import FunctionModule


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_block_gradient(inverse):
    n_bins = 10
    n_dims = 4
    n_cond = (3 * n_bins + 1) * n_dims
    cond_type = ms.batch_float_array(n_cond)
    io_type = ms.batch_float_array(n_dims)

    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type), ("cond", cond_type)]),
        ms.NamedTypes([("out", io_type), ("jac", ms.batch_float)]),
    )
    widths_unnorm, heights_unnorm, derivatives = fb.rqs_reshape(
        fb.input(1), ms.Value(n_bins)
    )
    widths = fb.softmax(widths_unnorm)
    heights = fb.softmax(heights_unnorm)
    rqs_condition = fb.rqs_find_bin(
        fb.input(0),
        heights if inverse else widths,
        widths if inverse else heights,
        derivatives,
    )
    out, det = (
        fb.rqs_inverse(fb.input(0), rqs_condition)
        if inverse
        else fb.rqs_forward(fb.input(0), rqs_condition)
    )
    fb.output(0, out)
    fb.output(1, fb.reduce_product(det))
    func = fb.function()
    module = FunctionModule(func, default_context())

    n_points = 200
    g = generator(3210)
    r_in = rand((n_points, n_dims), generator=g, requires_grad=True)
    r_cond = randn((n_points, n_cond), generator=g, requires_grad=True)
    torch.autograd.gradcheck(module, [r_in, r_cond], raise_exception=True, rtol=1e-4)


def test_one_dimensional_flow_is_refused():
    # a coupling block needs two non-empty halves, so a flow over a single random
    # number (2 -> 1 phase space) would have no block at all
    for input_dim in [0, 1]:
        with pytest.raises(ValueError, match="at least 2"):
            ms.Flow(input_dim=input_dim)


@pytest.mark.parametrize("input_dim", [2, 3, 5])
def test_flow_initialize_from_vegas(input_dim):
    context = ms.Context(ms.cpu_device())
    vegas = ms.VegasMapping(input_dim, 10, "vegas")
    vegas.initialize_globals(context)
    flow = ms.Flow(input_dim=input_dim, prefix="flow", bin_count=10)
    flow.initialize_from_vegas(context, vegas.grid_name(), 42)
    assert flow.input_dim() == input_dim

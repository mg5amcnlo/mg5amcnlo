import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest import approx

import madspace as ms
from madspace.torch import FunctionModule

torch.set_default_dtype(torch.float64)
torch.manual_seed(3210)


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
    module = FunctionModule(func)

    n_points = 200
    r_in = torch.rand((n_points, n_dims), requires_grad=True)
    r_cond = torch.randn((n_points, n_cond), requires_grad=True)
    torch.autograd.gradcheck(module, [r_in, r_cond], raise_exception=True, rtol=1e-4)

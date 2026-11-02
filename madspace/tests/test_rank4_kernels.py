"""An elementwise kernel on a rank-4 tensor, batch x 2 x 3 x 4.

The GPU kernel launcher used to compute only the first slice of the last dimension
of such a tensor, leaving the other elements of the output unset.
"""

import torch
from madspace_testing import default_context, generator, randn

import madspace as ms


def test_rank4_add(n=4096):
    rank4 = ms.Type(ms.DataType.float64, ms.batch_size, [2, 3, 4])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("x", rank4)]), ms.NamedTypes([("y", rank4)])
    )
    x = fb.input(0)
    fb.output(0, fb.add(x, x))
    a = randn((n, 2, 3, 4), generator=generator(23))
    (y,) = ms.FunctionRuntime(fb.function(), default_context()).call([a])
    assert torch.equal(torch.from_dlpack(y), 2 * a)

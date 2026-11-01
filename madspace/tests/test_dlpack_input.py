"""numpy inputs that numpy refuses to export through the unversioned DLPack protocol.

numpy < 2.1 has only that protocol, which cannot mark a tensor read-only, so it raises a
BufferError for a read-only array. numpy 1.x also makes every array it imports with
from_dlpack read-only, so there an output of madspace could not be passed back in. The
GPU CI runs with the numpy 1.26 of the cluster modules, where these failed.
"""

import numpy as np
from pytest import approx

import madspace as ms


def double():
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("x", ms.batch_float)]), ms.NamedTypes([("y", ms.batch_float)])
    )
    fb.output(0, fb.add(fb.input(0), fb.input(0)))
    return ms.FunctionRuntime(fb.function())


def test_read_only_array_is_accepted():
    x = np.arange(5.0)
    x.setflags(write=False)
    assert np.asarray(double()(x)) == approx(2 * np.arange(5.0))
    assert not x.flags.writeable


def test_output_can_be_passed_back_in():
    runtime = double()
    y = runtime(np.arange(5.0))
    assert np.asarray(runtime(y)) == approx(4 * np.arange(5.0))

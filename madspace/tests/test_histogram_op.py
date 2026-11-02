"""The histogram instruction: n_bins interior bins plus an underflow and an overflow bin.

Its outputs hold n_bins + 2 values, in that order: underflow, the interior bins, overflow.
The GPU runtime reduces the events to these n_bins + 2 keys (and used to do so into a
scratch buffer of n_bins, summing the squared weights of the wrong events in each bin);
the GPU CI runs these tests on a GPU via MADSPACE_TEST_DEVICE, where the large histogram
is compared with the CPU runtime.
"""

import torch
from madspace_testing import default_context, generator, rand, randn

import madspace as ms

N_BINS = 5


def histogram_function(n_bins):
    out_type = ms.Type(ms.DataType.float64, ms.BatchSize.one, [n_bins + 2])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(name, ms.batch_float) for name in ("x", "w", "lo", "hi")]),
        ms.NamedTypes([("values", out_type), ("square_values", out_type)]),
    )
    values, square_values = fb.histogram(*(fb.input(i) for i in range(4)), n_bins)
    fb.output(0, values)
    fb.output(1, square_values)
    return fb.function()


def run_histogram(x, w, lo, hi):
    runtime = ms.FunctionRuntime(histogram_function(N_BINS), default_context())
    return [torch.from_dlpack(t)[0] for t in runtime.call([x, w, lo, hi])]


def test_histogram_underflow_bins_overflow():
    x = torch.tensor([-1.0, 0.1, 0.5, 0.95, 1.5, 2.0, 0.3])
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    n = len(x)
    values, square_values = run_histogram(x, w, torch.zeros(n), torch.ones(n))
    # underflow: -1; bins of width 0.2: 0.1, 0.3, 0.5, -, 0.95; overflow: 1.5 and 2
    expected = torch.tensor([1.0, 2.0, 7.0, 3.0, 0.0, 4.0, 5.0 + 6.0])
    torch.testing.assert_close(values, expected, rtol=1e-7, atol=0)
    torch.testing.assert_close(
        square_values,
        torch.tensor([1.0, 4.0, 49.0, 9.0, 0.0, 16.0, 25.0 + 36.0]),
        rtol=1e-7,
        atol=0,
    )


def test_histogram_range_per_event():
    """min and max are per-event inputs: the same value falls in different bins."""
    x = torch.full((3,), 0.5)
    w = torch.ones(3)
    values, _ = run_histogram(
        x, w, torch.tensor([0.0, 0.4, 1.0]), torch.tensor([1.0, 1.4, 2.0])
    )
    # 0.5 in [0, 1): bin 2 (index 3); in [0.4, 1.4): bin 0 (index 1); below [1, 2): underflow
    torch.testing.assert_close(
        values, torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), rtol=1e-7, atol=0
    )


def test_large_histogram_matches_cpu_runtime(n=100000, n_bins=40):
    """Many events in many bins on the test device, against the CPU runtime."""
    function = histogram_function(n_bins)
    g = generator(17)
    inputs = [
        -0.5 + 2.0 * rand(n, generator=g),
        randn(n, generator=g),
        torch.zeros(n),
        torch.ones(n),
    ]
    device = [
        torch.from_dlpack(t).cpu()
        for t in ms.FunctionRuntime(function, default_context()).call(inputs)
    ]
    cpu = [
        torch.from_dlpack(t)
        for t in ms.FunctionRuntime(function, ms.default_context()).call(
            [t.cpu() for t in inputs]
        )
    ]
    for d, c in zip(device, cpu):
        rel_diff = (d - c).abs() / c.abs().clamp(min=1e-300)
        assert rel_diff.max().item() < 1e-9
    underflow, overflow = cpu[0][0][0].item(), cpu[0][0][-1].item()
    assert underflow != 0 and overflow != 0

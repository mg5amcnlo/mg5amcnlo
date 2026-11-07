"""Discrete sampling where the option probabilities can be exactly zero.

The probabilities of the flavour sampler are parton densities (times an
active-flavour mask), and modern sets return exactly 0.0 below the c/b
thresholds and at large x, so drawing a discrete option whose density is zero
is routine rather than pathological. kernel_sample_discrete_probs handles
that by construction instead of via a sentinel: an option with probability
zero can never be selected, and if every option in a row is zero there is no
real choice to make, so the sampler deterministically returns index 0 with
probability 1 (det == 1), same as any other certain draw.
"""

import torch
from madspace_testing import default_context

import madspace as ms

N_OPTIONS = 4

# row 0: a healthy prior
# row 1: every option's density exactly zero
# row 2: a single non-zero option
PRIOR = torch.tensor(
    [
        [0.3, 0.2, 0.4, 0.1],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7],
    ]
)
RANDOM = torch.tensor([0.42, 0.42, 0.42])
UNUSED_INDEX = torch.zeros(len(RANDOM), dtype=torch.int32)
ALL_ZERO_ROW = 1


_prefix_count = 0


def sampler_and_context():
    # the default context is shared between tests, so every sampler needs its
    # own name for the probability global
    global _prefix_count
    _prefix_count += 1
    ctx = default_context()
    sampler = ms.DiscreteSampler([N_OPTIONS], f"test_zero_prob{_prefix_count}_", [0])
    sampler.initialize_globals(ctx)
    return sampler, ctx


def run(builder_fn, output_names, inputs, output_types=None):
    sampler, ctx = sampler_and_context()
    types = output_types or {}
    fb = ms.FunctionBuilder(
        ms.NamedTypes(
            [
                ("r", ms.batch_float),
                ("prior", ms.batch_float_array(N_OPTIONS)),
                ("index_in", ms.batch_int),
            ]
        ),
        ms.NamedTypes(
            [(name, types.get(name, ms.batch_float)) for name in output_names]
        ),
    )
    builder_fn(fb, sampler)
    runtime = ms.FunctionRuntime(fb.function(), ctx)
    return dict(zip(output_names, (torch.from_dlpack(t) for t in runtime.call(inputs))))


def test_forward_det_is_always_at_least_one():
    """det is 1/p, so it is >= 1. The all-zero row is no exception: it gets
    index 0 and probability 1, so det == 1 there too.
    """

    def build(fb, sampler):
        result = sampler.build_forward(fb, [fb.input(0)], [fb.input(1)])
        fb.output(0, result["det"])
        fb.output(1, result[0])

    out = run(
        build,
        ["det", "index"],
        [RANDOM, PRIOR, UNUSED_INDEX],
        output_types={"index": ms.batch_int},
    )
    assert torch.all(out["det"] >= 1.0)
    assert out["det"][ALL_ZERO_ROW] == 1.0
    assert out["index"][ALL_ZERO_ROW] == 0


def test_forward_selects_the_expected_bucket():
    """r selects the option whose cumulative-probability bucket contains it.
    For [0.3, 0.2, 0.4, 0.1] the bucket edges are [0.3, 0.5, 0.9, 1.0].
    """
    prior = torch.tensor([0.3, 0.2, 0.4, 0.1]).repeat(5, 1)
    random = torch.tensor([0.0, 0.29, 0.31, 0.89, 0.99])
    expected = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32)

    def build(fb, sampler):
        fb.output(0, sampler.build_forward(fb, [fb.input(0)], [fb.input(1)])[0])

    out = run(
        build,
        ["index"],
        [random, prior, UNUSED_INDEX[: len(random)]],
        output_types={"index": ms.batch_int},
    )
    assert torch.equal(out["index"], expected)


def test_forward_never_selects_a_zero_probability_option():
    """Only one option carries any density; every draw of r must land on it,
    including r = 0, which sits on the edge of the zero-width buckets.
    """
    prior = torch.tensor([0.0, 0.0, 0.0, 0.7]).repeat(5, 1)
    random = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999999])

    def build(fb, sampler):
        fb.output(0, sampler.build_forward(fb, [fb.input(0)], [fb.input(1)])[0])

    out = run(
        build,
        ["index"],
        [random, prior, UNUSED_INDEX[: len(random)]],
        output_types={"index": ms.batch_int},
    )
    assert torch.all(out["index"] == 3)


def test_forward_and_inverse_dets_are_reciprocal():
    """build_forward's det for the option it draws is 1/p; build_inverse's
    det for that same option is p, so the product is 1, also on the all-zero
    row where both dets are 1.
    """

    def build(fb, sampler):
        prior_in = fb.input(1)
        forward = sampler.build_forward(fb, [fb.input(0)], [prior_in])
        inverse = sampler.build_inverse(fb, [forward[0]], [prior_in])
        fb.output(0, forward["det"])
        fb.output(1, inverse["det"])

    out = run(build, ["det_fwd", "det_inv"], [RANDOM, PRIOR, UNUSED_INDEX])
    product = out["det_fwd"] * out["det_inv"]
    assert torch.allclose(product, torch.ones_like(product))


def test_inverse_all_zero_prior_returns_the_canonical_draw():
    """When every option's density is zero there is nothing to invert:
    kernel_sample_discrete_probs_inverse returns the canonical r = 0.5 and
    det = 1 no matter which index is asked for, matching the forward
    sampler's index-0/probability-1 convention for the same row.
    """
    prior = torch.zeros((3, N_OPTIONS))
    index_in = torch.tensor([0, 1, 3], dtype=torch.int32)

    def build(fb, sampler):
        inverse = sampler.build_inverse(fb, [fb.input(2)], [fb.input(1)])
        fb.output(0, inverse[0])
        fb.output(1, inverse["det"])

    out = run(build, ["r", "det"], [torch.zeros(3), prior, index_in])
    assert torch.all(out["r"] == 0.5)
    assert torch.all(out["det"] == 1.0)

"""The me_frame boost applied to the momenta before the matrix element call.

Same definition as madevent's boost_to_frame (Template/LO/SubProcesses/genps.f):
the momenta of the particles listed in me_frame are summed up and every external
momentum is boosted into the rest frame of that sum. For a collision, except for
the incoming system itself, that rest frame is reached by a boost along the beam,
to zero rapidity of the sum (boost_to_zero_rapidity), and then a transverse one
(boost_to_frame), so that it does not depend on the frame along the beam the
momenta come in.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand

import madspace as ms

COUNT = 1000
N_PARTICLES = 5


@pytest.fixture
def momenta():
    """Physical momenta: two massless incoming along z with different energies
    (so the lab frame is boosted with respect to the partonic CM), and three
    massive outgoing balancing them."""
    g = generator(1234)
    e1 = 100.0 + 900.0 * rand(COUNT, generator=g)
    e2 = 100.0 + 900.0 * rand(COUNT, generator=g)
    zeros = torch.zeros(COUNT)
    p = torch.zeros((COUNT, N_PARTICLES, 4))
    p[:, 0] = torch.stack([e1, zeros, zeros, e1], dim=1)
    p[:, 1] = torch.stack([e2, zeros, zeros, -e2], dim=1)
    for i in range(2, N_PARTICLES - 1):
        px, py, pz = -40.0 + 80.0 * rand((3, COUNT), generator=g)
        mass = 20.0 + 10.0 * i
        p[:, i] = torch.stack(
            [torch.sqrt(mass**2 + px**2 + py**2 + pz**2), px, py, pz], dim=1
        )
    # last particle balances the rest, its mass is whatever comes out
    p[:, N_PARTICLES - 1] = p[:, 0] + p[:, 1] - p[:, 2:-1].sum(dim=1)
    # little headroom at low e1 + e2: fail here, not as a kernel bug
    assert (p[:, N_PARTICLES - 1, 0] > 0).all(), "balancing leg has no energy"
    total = p[:, :2].sum(dim=1)
    assert (total[:, 0] ** 2 - (total[:, 1:] ** 2).sum(dim=1) > 0).all()
    return p


def boost_momenta(momenta, frame):
    """Reference implementation: boost every momentum into the rest frame of the
    sum of the particles in ``frame`` (one-based indices). Textbook Lorentz
    boost with velocity beta = p_boost/E_boost, written out independently of the
    kernel under test."""
    p_boost = momenta[:, [i - 1 for i in frame]].sum(dim=1)
    mass = torch.sqrt(p_boost[:, 0] ** 2 - (p_boost[:, 1:] ** 2).sum(dim=1))
    gamma = (p_boost[:, 0] / mass)[:, None]
    beta = (p_boost[:, 1:] / p_boost[:, [0]])[:, None, :]  # (count, 1, 3)
    beta2 = (beta**2).sum(dim=2)  # (count, 1)
    energy = momenta[:, :, 0]  # (count, n)
    beta_dot_p = (momenta[:, :, 1:] * beta).sum(dim=2)  # (count, n)

    out = torch.empty_like(momenta)
    out[:, :, 0] = gamma * (energy - beta_dot_p)
    coeff = (
        torch.where(
            beta2 > 0,
            (gamma - 1) * beta_dot_p / torch.where(beta2 > 0, beta2, 1.0),
            0.0,
        )
        - gamma * energy
    )
    out[:, :, 1:] = momenta[:, :, 1:] + coeff[:, :, None] * beta
    return out


def run_boost(momenta, frame_mask, along_beam=False):
    """boost_to_frame, after boost_to_zero_rapidity with the same mask if
    along_beam -- the two steps MatrixElement applies for a collision"""
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("p", ms.batch_four_vec_array(N_PARTICLES))]),
        ms.NamedTypes([("p_frame", ms.batch_four_vec_array(N_PARTICLES))]),
    )
    p = fb.input(0)
    if along_beam:
        p = fb.boost_to_zero_rapidity(p, ms.Value(frame_mask))
    fb.output(0, fb.boost_to_frame(p, ms.Value(frame_mask)))
    runtime = ms.FunctionRuntime(fb.function())
    return torch.from_dlpack(runtime.call([momenta])[0])


@pytest.mark.parametrize("frame", [[1, 2], [3], [4], [3, 4], [3, 4, 5]], ids=str)
def test_boost_matches_reference(momenta, frame):
    mask = [1 if i + 1 in frame else 0 for i in range(N_PARTICLES)]
    result = run_boost(momenta, mask)
    torch.testing.assert_close(
        result, boost_momenta(momenta, frame), rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize("frame", [[1, 2], [3], [3, 4]], ids=str)
def test_boost_preserves_invariants(momenta, frame):
    mask = [1 if i + 1 in frame else 0 for i in range(N_PARTICLES)]
    result = run_boost(momenta, mask)

    def m2(p):
        return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(dim=-1)

    torch.testing.assert_close(m2(result), m2(momenta), rtol=1e-8, atol=1e-6)
    torch.testing.assert_close(
        m2(result.sum(dim=1)), m2(momenta.sum(dim=1)), rtol=1e-10, atol=0.0
    )


def test_selected_sum_is_at_rest(momenta):
    """The defining property of the frame: the summed three-momentum vanishes."""
    for frame in ([1, 2], [3], [3, 4], [3, 4, 5]):
        mask = [1 if i + 1 in frame else 0 for i in range(N_PARTICLES)]
        result = run_boost(momenta, mask)
        p_sum = result[:, [i - 1 for i in frame]].sum(dim=1)
        scale = result[:, :, 0].abs().amax(dim=1)
        torch.testing.assert_close(
            p_sum[:, 1:] / scale[:, None],
            torch.zeros_like(p_sum[:, 1:]),
            rtol=1e-7,
            atol=1e-12,
        )


def test_single_particle_frame_is_exactly_at_rest(momenta):
    """With one particle defining the frame, its three-momentum must be exactly
    zero, not just small: vxxxxx branches on pp == 0 and otherwise builds the
    polarisation vectors of a massive vector out of the rounding noise."""
    mask = [0, 0, 1, 0, 0]
    result = run_boost(momenta, mask)
    assert torch.all(result[:, 2, 1:] == 0.0)


def test_incoming_frame_undoes_the_beam_boost(momenta):
    """me_frame = [1, 2] is the partonic centre of mass: the two incoming
    momenta come out back to back with equal energies."""
    result = run_boost(momenta, [1, 1, 0, 0, 0])
    torch.testing.assert_close(result[:, 0, 0], result[:, 1, 0], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(result[:, 0, 3], -result[:, 1, 3], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        result[:, 0, 1:3], torch.zeros_like(result[:, 0, 1:3]), rtol=1e-7, atol=1e-9
    )


def boost_z(momenta, rapidity):
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out = momenta.clone()
    out[..., 0] = momenta[..., 0] * ch + momenta[..., 3] * sh
    out[..., 3] = momenta[..., 3] * ch + momenta[..., 0] * sh
    return out


def test_frame_does_not_depend_on_the_frame_the_momenta_arrive_in(momenta):
    """The two-step boost madspace applies to a collision -- first along the
    beam to zero rapidity of the selection, then into its rest frame -- has to
    give the same momenta whatever longitudinal boost the input carries: lab,
    partonic CM, or the frames aMC@NLO evaluates its real emission in. A single
    boost straight to the target does not: boosts along different directions do
    not compose into the boost between the end frames, and the leftover Wigner
    rotation turns the spin axis of a single particle held at rest."""
    for frame in ([0, 0, 1, 0, 0], [0, 0, 1, 1, 0]):
        reference = run_boost(momenta, frame, along_beam=True)
        scale = reference[:, :, 0].abs().amax(dim=1)[:, None, None]
        for rapidity in (-1.3, 0.7):
            shifted = run_boost(boost_z(momenta, rapidity), frame, along_beam=True)
            torch.testing.assert_close(
                shifted / scale, reference / scale, rtol=0, atol=1e-10
            )
    # the single boost does depend on it
    frame = [0, 0, 1, 0, 0]
    single = run_boost(momenta, frame)
    shifted = run_boost(boost_z(momenta, 0.7), frame)
    assert ((shifted - single).abs() / scale).amax() > 1e-2


def test_single_particle_frame_is_collins_soper(momenta):
    """For a single particle the frame reached along the beam is the
    Collins-Soper frame: its z axis bisects beam 1 and minus beam 2."""
    result = run_boost(momenta, [0, 0, 1, 0, 0], along_beam=True)
    assert torch.all(result[:, 2, 1:] == 0.0)
    n1 = result[:, 0, 1:] / result[:, 0, 1:].norm(dim=1, keepdim=True)
    n2 = -result[:, 1, 1:] / result[:, 1, 1:].norm(dim=1, keepdim=True)
    bisector = n1 + n2
    torch.testing.assert_close(
        bisector[:, :2], torch.zeros_like(bisector[:, :2]), rtol=0, atol=1e-12
    )
    assert torch.all(bisector[:, 2] > 0)


def test_zero_rapidity_boost_is_along_the_beam(momenta):
    """boost_to_zero_rapidity leaves transverse momenta alone and brings the
    selection to zero longitudinal momentum."""
    p = momenta.clone()
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("p", ms.batch_four_vec_array(N_PARTICLES))]),
        ms.NamedTypes([("p_frame", ms.batch_four_vec_array(N_PARTICLES))]),
    )
    fb.output(0, fb.boost_to_zero_rapidity(fb.input(0), ms.Value([0, 0, 1, 1, 0])))
    result = torch.from_dlpack(ms.FunctionRuntime(fb.function()).call([p])[0])
    torch.testing.assert_close(result[:, :, 1:3], p[:, :, 1:3], rtol=0, atol=0)
    scale = result[:, :, 0].abs().amax(dim=1)
    torch.testing.assert_close(
        result[:, 2:4, 3].sum(dim=1) / scale,
        torch.zeros_like(scale),
        rtol=0,
        atol=1e-12,
    )


def test_boost_with_an_empty_frame_is_the_identity():
    """Nothing selected means nothing defines a frame. Not reachable through
    the run card, but boost_to_frame is a public instruction: it must pass the
    momenta through rather than boost by a null vector."""
    p = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0, 100.0],
                [50.0, 1.0, 2.0, -49.0],
                [90.0, -1.0, -2.0, 89.0],
                [60.0, 0.0, 0.0, 60.0],
                [70.0, 3.0, 0.0, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(run_boost(p, [0] * N_PARTICLES), p, rtol=0, atol=0)


def test_matrix_element_rejects_out_of_range_frame():
    with pytest.raises(ValueError, match="me_frame"):
        ms.MatrixElement(
            matrix_element_index=0,
            particle_count=4,
            inputs=[ms.MatrixElement.momenta_in],
            outputs=[ms.MatrixElement.matrix_element_out],
            me_frame=[1, 5],
        )


def matrix_element(**kwargs):
    return ms.MatrixElement(
        matrix_element_index=0,
        particle_count=4,
        inputs=[ms.MatrixElement.momenta_in],
        outputs=[ms.MatrixElement.matrix_element_out],
        **kwargs,
    )


def matrix_element_graph(me_frame, incoming_count=2):
    """The compute graph MatrixElement builds, as text. Needs no matrix-element
    library: only the graph is built, never run."""
    me = ms.MatrixElement(
        matrix_element_index=0,
        particle_count=4,
        inputs=[ms.MatrixElement.momenta_in],
        outputs=[ms.MatrixElement.matrix_element_out],
        me_frame=me_frame,
        incoming_count=incoming_count,
    )
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("p", ms.batch_four_vec_array(4))]),
        ms.NamedTypes([("me", ms.batch_float)]),
    )
    fb.output(0, me.build_function(fb, [fb.input(0)])[0])
    return str(fb.function())


def test_matrix_element_emits_the_boost():
    """The number of boost_to_frame calls in front of the matrix element."""
    # no frame asked for: the momenta go in untouched
    assert matrix_element_graph([]).count("boost_to_frame") == 0
    # the rest frame of the incoming system is one boost away from the lab
    assert matrix_element_graph([1, 2]).count("boost_to_frame") == 1
    assert matrix_element_graph([1, 2]).count("boost_to_zero_rapidity") == 0
    assert matrix_element_graph([1], incoming_count=1).count("boost_to_frame") == 1
    # any other frame of a collision is reached along the beam first, never
    # straight from the lab
    for frame in ([3], [3, 4]):
        graph = matrix_element_graph(frame)
        assert graph.count("boost_to_zero_rapidity") == 1
        assert graph.count("boost_to_frame") == 1
    # and a decay through the rest frame of the decaying particle
    graph = matrix_element_graph([2], incoming_count=1)
    assert graph.count("boost_to_zero_rapidity") == 0
    assert graph.count("boost_to_frame") == 2


def test_matrix_element_frame_mask():
    """me_frame = [1, 2] is the rest frame of the incoming system itself, so it
    is reached with a single boost and needs no reference frame."""
    me = matrix_element(me_frame=[1, 2])
    assert list(me.frame_mask()) == [1, 1, 0, 0]
    assert list(me.reference_mask()) == []

    me_no_frame = matrix_element()
    assert list(me_no_frame.frame_mask()) == []
    assert list(me_no_frame.reference_mask()) == []


def test_matrix_element_reference_frame():
    """A decay reaches any other frame from the rest frame of the decaying
    particle; a collision has no reference frame, it goes along the beam."""
    me = matrix_element(me_frame=[3])
    assert list(me.frame_mask()) == [0, 0, 1, 0]
    assert list(me.reference_mask()) == []

    decay = matrix_element(me_frame=[2], incoming_count=1)
    assert list(decay.frame_mask()) == [0, 1, 0, 0]
    assert list(decay.reference_mask()) == [1, 0, 0, 0]
    # the decaying particle's rest frame is the reference frame itself
    assert list(matrix_element(me_frame=[1], incoming_count=1).reference_mask()) == []

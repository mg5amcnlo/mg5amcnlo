"""The ``adaptive_prob`` returned by an Integrand in MadNIS training mode is the
density the event was sampled with, q, for both settings of
``drop_cuts_and_rescale``.

The MadNIS loss and the replay-buffer unweighter take it as q. The product of the
Jacobians of the adaptive mappings is 1/q, so it has to be inverted for both
settings (with ``drop_cuts_and_rescale`` also rescaled by N / N_acc); returning
the Jacobian itself would make training with ``drop_zero_integrands = false``
use 1/q. Checked against the density from IntegrandProbability at the same
points, with a randomized flow so that q is not constant.
"""

import pytest
import torch
from madspace_testing import new_context

import madspace as ms

CM_ENERGY = 1000.0
N = 1000

pytestmark = pytest.mark.cpu_only


def build_integrand(context, drop_cuts_and_rescale, tmp_path):
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = tmp_path / "alphas.info"
    alphas_file.write_text(
        "AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\n"
        "AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\n"
    )
    alphas_grid = ms.AlphaSGrid(str(alphas_file))
    alphas_grid.initialize_globals(context)
    pids = [11, -11, 1, -1]
    # a pt cut is sampled within by the mapping, a rapidity cut drops events
    obs = ms.Observable(pids, ms.Observable.obs_eta, [[1]])
    mapping = ms.PhaseSpaceMapping(
        [0.0] * 4,
        CM_ENERGY,
        leptonic=True,
        cuts=ms.Cuts([ms.CutItem(obs, max=0.5)]),
    )
    # 0xBADCAFE: the built-in flat test matrix element, no library needed
    matrix_element = ms.MatrixElement(
        0xBADCAFE,
        4,
        ms.Integrand.matrix_element_inputs,
        ms.Integrand.matrix_element_outputs,
        1,
        True,
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
    )
    flow = ms.Flow(input_dim=mapping.random_dim(), prefix="test.flow")
    flow.initialize_globals(context, 1)
    cwnet = ms.ChannelWeightNetwork(
        channel_count=1, particle_count=4, prefix="test.cwnet"
    )
    cwnet.initialize_globals(context, 1)
    return ms.Integrand(
        mapping,
        [diff_xs],
        adaptive_map=flow,
        running_coupling=ms.RunningCoupling(alphas_grid),
        energy_scale=ms.EnergyScale(4, 91.188),
        chan_weight_net=cwnet,
        madnis_training=True,
        drop_cuts_and_rescale=drop_cuts_and_rescale,
        channel_indices=[0],
    )


@pytest.mark.parametrize("drop_cuts_and_rescale", [False, True])
def test_adaptive_prob_is_the_sampling_density(drop_cuts_and_rescale, tmp_path):
    context = new_context()
    integrand = build_integrand(context, drop_cuts_and_rescale, tmp_path)
    prob_function = ms.IntegrandProbability(integrand).function()
    # the initial flow is the identity, randomize it so that q is not constant
    gen = torch.Generator().manual_seed(2)
    for name, _ in prob_function.globals:
        param = context.get_global(name).torch()
        param += 0.3 * torch.randn(param.shape, generator=gen, dtype=param.dtype)

    outputs = [
        torch.from_dlpack(t)
        for t in ms.FunctionRuntime(integrand.function(), context).call(
            [torch.tensor([N], dtype=torch.int64)]
        )
    ]
    # outputs: full_weight, weight, latent, adaptive_prob, channel_index,
    # channel_weights, cwnet_input, channel_index_in_group
    weight, latent, adaptive_prob = outputs[1], outputs[2], outputs[3]
    chan_index_in_group = outputs[7]
    (density,) = (
        torch.from_dlpack(t)
        for t in ms.FunctionRuntime(prob_function, context).call(
            [latent, chan_index_in_group]
        )
    )
    assert density.std() > 0.1 * density.mean()

    count_acc = int((weight != 0).sum())
    assert 0 < count_acc < N
    if drop_cuts_and_rescale:
        # the events failing the cuts are dropped, q is rescaled to the rest
        assert len(weight) == count_acc
        expected = density * N / count_acc
    else:
        assert len(weight) == N
        expected = density
    torch.testing.assert_close(adaptive_prob, expected, rtol=1e-10, atol=0.0)

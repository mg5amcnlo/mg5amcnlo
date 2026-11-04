"""apply_mlm_hardest_shower_scale: the MLM shower starts from the hardest scale of
the event, not from the collider energy the clustering gives a leg it booked no
scale onto. Only particles that are not light partons move; a light parton at
the collider energy keeps it (Pythia8's do-not-match flag)."""

import pytest

import madspace as ms

SQRT_S = 13000.0
IN, OUT, RES = (
    ms.LHEParticle.status_incoming,
    ms.LHEParticle.status_outgoing,
    ms.LHEParticle.status_intermediate_resonance,
)


def event(scalup, particles):
    """particles: (pdg_id, status, pt_clust)"""
    lhe = ms.LHEEvent(scale=scalup)
    parts = []
    for pdg_id, status, scale in particles:
        p = ms.LHEParticle(pdg_id=pdg_id, status_code=status)
        p.cluster_scale = scale
        parts.append(p)
    lhe.particles = parts
    lhe.has_cluster_scales = True
    return lhe


def scales(lhe):
    return [p.cluster_scale for p in lhe.particles]


def test_leptons_start_at_mu_f():
    # u g > W(> e+ ve) d: the leptons were at sqrt(s), the jet has its own scale
    lhe = event(83.0, [(2, IN, 0.), (21, IN, 0.), (24, RES, 0.), (-11, OUT, SQRT_S),
                       (12, OUT, SQRT_S), (1, OUT, 25.0)])
    ms.apply_mlm_hardest_shower_scale(lhe, SQRT_S, 4)
    assert scales(lhe) == [0., 0., 0., 83.0, 83.0, 25.0]
    assert lhe.scale == 83.0
    assert 'pt_clust_4="83.00000"' in lhe.format()


def test_a_jet_harder_than_mu_f_sets_the_scale():
    lhe = event(83.0, [(2, IN, 0.), (21, IN, 0.), (-11, OUT, SQRT_S), (12, OUT, SQRT_S),
                       (1, OUT, 150.0)])
    ms.apply_mlm_hardest_shower_scale(lhe, SQRT_S, 4)
    assert scales(lhe)[2:] == [150.0, 150.0, 150.0]


def test_a_core_light_parton_keeps_the_flag():
    # the gluon in the core is not matched by Pythia8 only at sqrt(s), and its
    # flag does not count towards the hardest scale
    lhe = event(90.0, [(2, IN, 0.), (-2, IN, 0.), (-11, OUT, SQRT_S), (12, OUT, SQRT_S),
                       (21, OUT, SQRT_S), (1, OUT, 30.0)])
    ms.apply_mlm_hardest_shower_scale(lhe, SQRT_S, 4)
    assert scales(lhe)[2:] == [90.0, 90.0, SQRT_S, 30.0]


@pytest.mark.parametrize("max_jet_flavor, b_scale", [(4, 210.0), (5, SQRT_S)])
def test_heavy_quarks_follow_max_jet_flavor(max_jet_flavor, b_scale):
    lhe = event(210.0, [(21, IN, 0.), (21, IN, 0.), (6, OUT, SQRT_S), (-6, OUT, SQRT_S),
                        (5, OUT, SQRT_S), (21, OUT, 60.0)])
    ms.apply_mlm_hardest_shower_scale(lhe, SQRT_S, max_jet_flavor)
    assert scales(lhe)[2:] == [210.0, 210.0, b_scale, 60.0]


def test_events_without_cluster_scales_are_left_alone():
    lhe = event(83.0, [(2, IN, 0.), (-1, IN, 0.), (-11, OUT, SQRT_S), (12, OUT, SQRT_S)])
    lhe.has_cluster_scales = False
    ms.apply_mlm_hardest_shower_scale(lhe, SQRT_S, 4)
    assert scales(lhe)[2:] == [SQRT_S, SQRT_S]


def test_off_by_default():
    config = ms.GeneratorConfig()
    assert config.mlm_hardest_shower_scale is False
    assert config.mlm_max_jet_flavor == 4

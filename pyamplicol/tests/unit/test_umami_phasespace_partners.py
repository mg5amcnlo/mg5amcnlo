# SPDX-License-Identifier: 0BSD
"""The phase-space channels that integrate one leading-colour order together:
where its colour singlets attach to the colour lines, which non-cyclic order
its lines take, and which s-channel resonances it maps out."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from types import SimpleNamespace

from pyamplicol.artifacts.umami import (
    _MAX_PARTNERS,
    _configurations,
    _line_blocks,
    _phasespace_partners,
)
from pyamplicol.generation.s_channel_resonances import s_channel_resonances
from pyamplicol.models.base import Particle
from tests.unit.test_umami_metadata import export, process

# all-outgoing colour representations
Q, QBAR, G, S = 3, -3, 8, 1


def test_line_blocks_follow_the_word():
    roles = [QBAR, Q, G, Q, QBAR, G]
    assert _line_blocks([1, 2, 0, 3, 5, 4], roles) == [[1, 2, 0], [3, 5, 4]]
    # a gluon trace has no open lines
    assert _line_blocks([0, 1, 2], [G, G, G]) is None


def test_singlets_follow_the_antiquark_in_every_order():
    # q q~ > e+ e- g, the word q g q~ crossed: slots 1 (q~ in) ... 0 (q in)
    roles = [QBAR, Q, S, S, G]
    partners = _phasespace_partners([1, 4, 0], [2, 3], roles, [()])
    assert [order for order, _ in partners] == [[1, 4, 0, 2, 3], [1, 4, 0, 3, 2]]
    assert all(configuration == () for _, configuration in partners)


def test_singlets_are_distributed_over_the_lines():
    # two lines and one singlet: after the first or the second antiquark
    roles = [Q, QBAR, Q, QBAR, S]
    partners = _phasespace_partners([0, 1, 2, 3], [4], roles, [()])
    assert [order for order, _ in partners] == [[0, 1, 2, 3, 4], [0, 1, 4, 2, 3]]


def test_three_lines_take_both_cyclic_orders():
    roles = [Q, QBAR] * 3
    partners = _phasespace_partners(list(range(6)), [], roles, [()])
    assert [order for order, _ in partners] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 4, 5, 2, 3],
    ]


def test_a_resonance_keeps_its_decay_products_together():
    roles = [QBAR, Q, S, S, G]
    z = (23, frozenset({2, 3}))
    configurations, _ = _configurations([z])
    partners = _phasespace_partners([1, 4, 0], [2, 3], roles, configurations)
    # the non-resonant placements and the Z, with its legs as one unit
    assert partners == [
        ([1, 4, 0, 2, 3], ()),
        ([1, 4, 0, 3, 2], ()),
        ([1, 4, 0, 2, 3], ((23, frozenset({2, 3})),)),
    ]


def test_a_colour_singlet_decaying_to_a_complete_line():
    roles = [Q, QBAR, Q, QBAR, G]
    configurations, _ = _configurations([(23, frozenset({2, 3}))])
    # the Z decays to the line 2 3 of this order, but not to legs of two lines
    assert len(_phasespace_partners([0, 1, 2, 3], [], roles, configurations)) == 2
    assert len(_phasespace_partners([0, 3, 2, 1], [], roles, configurations)) == 1


def test_overlapping_resonances_are_alternatives():
    # e+ e- e+ ve: a Z from either positron, with the W from the other
    z1, z2 = (23, frozenset({2, 4})), (23, frozenset({3, 4}))
    w1, w2 = (24, frozenset({2, 5})), (24, frozenset({3, 5}))
    configurations, complete = _configurations([z1, z2, w1, w2])
    assert complete
    assert configurations == [(), (z1, w2), (z2, w1), (z1,), (z2,), (w1,), (w2,)]


def test_nested_resonances_stay_contiguous():
    roles = [QBAR, Q, S, S, S, S]
    h = (25, frozenset({2, 3, 4, 5}))
    z1, z2 = (23, frozenset({2, 4})), (23, frozenset({3, 5}))
    configurations, _ = _configurations([h, z1, z2])
    partners = _phasespace_partners([1, 0], [2, 3, 4, 5], roles, configurations)
    resonant = [order for order, configuration in partners if len(configuration) == 3]
    assert resonant == [[1, 0, 2, 4, 3, 5]]


def test_every_compatible_set_of_resonances_is_a_configuration():
    # H > Z Z > 4l: each resonance alone and every nested or disjoint set, except
    # the H alone, a four-body decay that the phase-space mapping cannot build
    h = (25, frozenset({2, 3, 4, 5}))
    z1, z2 = (23, frozenset({2, 4})), (23, frozenset({3, 5}))
    configurations, complete = _configurations([h, z1, z2])
    assert complete
    assert configurations == [
        (),
        (h, z1, z2),
        (h, z1),
        (h, z2),
        (z1, z2),
        (z1,),
        (z2,),
    ]


def test_many_resonances_are_capped_not_rejected():
    # 24 pairwise disjoint resonances have 2^24 configurations
    resonances = [(23, frozenset({2 * i, 2 * i + 1})) for i in range(24)]
    started = time.perf_counter()
    configurations, complete = _configurations(resonances, limit=100)
    assert time.perf_counter() - started < 1
    assert not complete
    assert len(configurations) == 100
    assert configurations[0] == ()
    assert [len(c) for c in configurations[1:]] == sorted(
        (len(c) for c in configurations[1:]), reverse=True
    )
    assert _configurations(resonances, limit=100) == (configurations, complete)


def test_partners_are_truncated_deterministically():
    # eight singlets: 8! orders per configuration, enumerated only up to the cap
    roles = [QBAR, Q] + [S] * 8
    singlets = list(range(2, 10))
    resonances = [(23, frozenset({2 * i, 2 * i + 1})) for i in range(1, 5)]
    configurations, _ = _configurations(resonances)
    started = time.perf_counter()
    partners = _phasespace_partners([1, 0], singlets, roles, configurations)
    assert time.perf_counter() - started < 0.5
    assert len(partners) <= _MAX_PARTNERS
    assert partners[0] == ([1, 0, *singlets], ())
    # every configuration keeps its first placements, the largest ones first
    assert [c for c in dict.fromkeys(c for _, c in partners)] == configurations
    assert partners == _phasespace_partners([1, 0], singlets, roles, configurations)


def test_unmapped_singlet_resonances_are_dropped_with_one_warning(caplog):
    # q q~ > s s g: an s decaying to an s and the gluon, and a coloured one
    p = process(pdgs=(991, -991, 995, 995, 990), words=((2, 5, 1),))
    p.physics["extensions"]["s_channel_resonances"] = [
        {"pdg": 23, "labels": [3, 4], "color": 1},
        {"pdg": 25, "labels": [3, 5], "color": 1},
        {"pdg": 25, "labels": [4, 5], "color": 1},
        {"pdg": 6, "labels": [4, 5], "color": 3},
    ]
    with caplog.at_level(logging.WARNING, logger="pyamplicol.artifacts.umami"):
        (data,) = export(p)
    (warning,) = caplog.records
    assert "25 > 995 990, 25 > 995 990" in warning.getMessage()
    # one channel samples the non-resonant map, one the Z, each a partner of
    # the other
    assert sorted(
        tuple(r["pdg"] for r in c["phasespace_channels"][0]["resonances"])
        for c in data["channels"]
    ) == [(), (23,)]
    for channel in data["channels"]:
        for entry in channel["processes"]:
            assert entry["multichannels"][0] == 0
            assert {
                tuple(r["pdg"] for r in channel["phasespace_channels"][i]["resonances"])
                for i in entry["multichannels"]
            } == {(), (23,)}
    # a coloured resonance is not followed; the launch checks it
    for channel in data["channels"]:
        assert {(r["pdg"], r["color"]) for r in channel["colored_resonances"]} == {
            (6, 3)
        }


def test_singlet_resonance_inside_a_gluon_trace_is_dropped_with_a_warning(caplog):
    p = process()
    p.physics["extensions"]["s_channel_resonances"] = [
        {"pdg": 25, "labels": [3, 4], "color": 1}
    ]
    with caplog.at_level(logging.WARNING, logger="pyamplicol.artifacts.umami"):
        (data,) = export(p)
    (warning,) = caplog.records
    assert "25 > 990 990" in warning.getMessage()
    (channel,) = data["channels"]
    assert all(not ps["resonances"] for ps in channel["phasespace_channels"])


def test_line_resonance_of_some_colour_orders_is_not_warned(caplog):
    p = process(pdgs=(991, -991, 992, -992), words=((2, 1, 3, 4), (2, 4, 3, 1)))
    p.physics["extensions"]["s_channel_resonances"] = [
        {"pdg": 23, "labels": [3, 4], "color": 1}
    ]
    sectors = (
        {
            "id": 0,
            "kind": "open-lines",
            "word_labels": [2, 1, 3, 4],
            "open_color_lines": [
                {
                    "fundamental_label": 2,
                    "adjoint_labels": [],
                    "antifundamental_label": 1,
                },
                {
                    "fundamental_label": 3,
                    "adjoint_labels": [],
                    "antifundamental_label": 4,
                },
            ],
        },
        {
            "id": 1,
            "kind": "open-lines",
            "word_labels": [2, 4, 3, 1],
            "open_color_lines": [
                {
                    "fundamental_label": 2,
                    "adjoint_labels": [],
                    "antifundamental_label": 4,
                },
                {
                    "fundamental_label": 3,
                    "adjoint_labels": [],
                    "antifundamental_label": 1,
                },
            ],
        },
    )
    with caplog.at_level(logging.WARNING, logger="pyamplicol.artifacts.umami"):
        (data,) = export(replace(p, color_sectors=sectors))
    assert not caplog.records
    resonant = [
        c["phasespace_channels"][0]
        for c in data["channels"]
        if c["phasespace_channels"][0]["resonances"]
    ]
    assert [r["pdg"] for ps in resonant for r in ps["resonances"]] == [23]


class _Model:
    """f f~ > l- l+ through any vector of ``particles``."""

    def __init__(self, particles, parameters):
        self.particles = {p.pdg: p for p in particles}
        self.parameters = parameters

    def vertices_accepting(self, left, right, color_accuracy):
        if {left, right} != {11, -11}:
            return []
        return [
            SimpleNamespace(particles=(left, right, pdg))
            for pdg in self.particles
            if abs(pdg) != 11
        ]

    def skip_duplicate_vertex_orientation(self, vertex):
        return False

    def vertex_coupling_orders(self, vertex):
        return (("QED", 1),)

    def particle(self, pdg):
        return self.particles[pdg]

    def anti_particle(self, pdg):
        return self.particles[pdg].anti_pdg

    def charge(self, pdg):
        return self.particles[pdg].charge

    def color_rep(self, pdg):
        return self.particles[pdg].color_rep

    def runtime_mass_parameter_name(self, pdg):
        return self.parameters.get((pdg, "mass"))

    def runtime_width_parameter_name(self, pdg):
        return self.parameters.get((pdg, "width"))


def test_candidates_are_kept_unless_their_mass_or_width_is_zero_in_the_model():
    def vector(pdg, mass, width):
        return Particle(pdg, pdg, 2, 4, 1, mass, width)

    particles = [
        Particle(11, -11, 1, 4, 1, 50.0, 0.0, -1.0),
        Particle(-11, 11, 1, 4, 1, 50.0, 0.0, 1.0),
        vector(22, 0.0, 0.0),  # massless
        vector(23, 91.0, 2.5),
        vector(97, 60.0, 1.0),  # below the threshold of the default masses
        vector(98, 200.0, 0.0),  # no width in the model
        vector(99, 200.0, 0.0),  # zero width on the parameter card
    ]
    parameters = {(11, "mass"): "ML", (-11, "mass"): "ML", (99, "width"): "WX"}
    parameters |= {(pdg, kind): kind for pdg in (23, 97, 98, 99) for kind in ("mass",)}
    parameters |= {(pdg, "width"): "W" for pdg in (23, 97)}

    def leg(label, pdg, outgoing, side):
        return SimpleNamespace(
            label=label, pdg=pdg, outgoing_pdg=outgoing, is_final=side == "final"
        )

    ir = SimpleNamespace(
        color_accuracy="lc",
        legs=(
            leg(1, 11, -11, "initial"),
            leg(2, -11, 11, "initial"),
            leg(3, 11, 11, "final"),
            leg(4, -11, -11, "final"),
        ),
    )
    found = s_channel_resonances(ir, _Model(particles, parameters))
    assert found == [{"pdg": pdg, "labels": [3, 4], "color": 1} for pdg in (23, 97, 99)]


def test_a_resonance_decays_to_at_most_three_daughters():
    # a Z of four legs that has no nested resonance would be a four-body decay,
    # which the phase-space mapping cannot build; with one nested it has three
    z_4l = (23, frozenset({2, 3, 4, 5}))
    z_ee = (23, frozenset({2, 3}))
    z_mm = (23, frozenset({4, 5}))
    configurations, _ = _configurations([z_4l])
    assert configurations == [()]
    configurations, _ = _configurations([z_4l, z_ee, z_mm])
    assert (z_4l,) not in configurations
    assert (z_4l, z_ee) in configurations and (z_ee, z_mm) in configurations
    z_3l = (23, frozenset({2, 3, 4}))
    assert _configurations([z_3l])[0] == [(), (z_3l,)]


def test_every_partner_channel_samples_a_copy_of_its_own():
    """A contribution is copied into each phase-space channel integrating it,
    relabelled so that the channel is canonical: one copy per partner, which
    samples its own map first, the physical members on the first copy only."""
    # q q~ > s s g with a Z between the singlets: two placements and the Z
    p = process(pdgs=(991, -991, 995, 995, 990), words=((2, 5, 1),))
    p.physics["extensions"]["s_channel_resonances"] = [
        {"pdg": 23, "labels": [3, 4], "color": 1}
    ]
    (data,) = export(p)
    copies = [e for c in data["channels"] for e in c["processes"]]
    primaries = [e for e in copies if e["members"]]
    assert len(primaries) == 1
    assert len(copies) == len(primaries[0]["multichannels"]) == 3
    for channel in data["channels"]:
        assert channel["phasespace_order"] == (
            channel["phasespace_channels"][0]["phasespace_order"]
        )
        for entry in channel["processes"]:
            assert entry["multichannels"][0] == 0
            assert len(set(entry["multichannels"])) == 3
    # the identical singlets swapped: two copies in the non-resonant channel
    assert sorted(len(c["processes"]) for c in data["channels"]) == [1, 2]

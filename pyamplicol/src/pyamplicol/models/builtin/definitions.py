# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from ..base import (
    Particle,
    Vertex,
)


class BuiltinSMDefinitionMixin:
    def _build_particles(self) -> list[Particle]:
        particles: list[Particle] = []
        for i in range(1, 6):
            if i % 2 == 0:
                particles.append(
                    Particle(
                        i,
                        -i,
                        2,
                        4,
                        3,
                        charge=2.0 / 3.0,
                        weak_isospin=(0.5, 0.0),
                        weak_hypercharge=(1.0 / 3.0, 4.0 / 3.0),
                    )
                )
            else:
                particles.append(
                    Particle(
                        i,
                        -i,
                        2,
                        4,
                        3,
                        charge=-1.0 / 3.0,
                        weak_isospin=(-0.5, 0.0),
                        weak_hypercharge=(1.0 / 3.0, -2.0 / 3.0),
                    )
                )
        particles.extend(
            [
                Particle(
                    6,
                    -6,
                    2,
                    4,
                    3,
                    mass=173.0,
                    width=1.491500,
                    charge=2.0 / 3.0,
                    weak_isospin=(0.5, 0.0),
                    weak_hypercharge=(1.0 / 3.0, 4.0 / 3.0),
                ),
                Particle(21, 21, 2, 4, 8),
                Particle(99, 99, -1, 4, 1),
                Particle(-21, -21, -1, 6, 8),
                Particle(22, 22, 2, 4, 1),
                Particle(23, 23, 3, 4, 1, mass=91.188, width=2.441404),
                Particle(-23, -23, -1, 6, 1),
                Particle(
                    24,
                    -24,
                    3,
                    4,
                    1,
                    mass=80.419002445756163,
                    width=2.0476,
                    charge=1.0,
                    weak_isospin=(1.0, 1.0),
                ),
                Particle(
                    25,
                    25,
                    1,
                    1,
                    1,
                    mass=125.0,
                    width=0.0063823389999999999,
                    weak_isospin=(-0.5, -0.5),
                    weak_hypercharge=(1.0, 1.0),
                ),
                Particle(125, 125, -1, 1, 1),
                Particle(126, 126, -1, 1, 1),
                Particle(127, 127, -1, 1, 1),
                Particle(26, -26, -1, 6, 1, charge=1.0, weak_isospin=(1.0, 1.0)),
            ]
        )
        for i in range(1, 4):
            charged = 11 + (2 * i - 2)
            neutrino = 12 + (2 * i - 2)
            particles.append(
                Particle(
                    charged,
                    -charged,
                    2,
                    4,
                    1,
                    charge=-1.0,
                    weak_isospin=(-0.5, 0.0),
                    weak_hypercharge=(-1.0, -2.0),
                )
            )
            particles.append(
                Particle(
                    neutrino,
                    -neutrino,
                    2,
                    4,
                    1,
                    weak_isospin=(0.5, 0.0),
                    weak_hypercharge=(-1.0, 0.0),
                )
            )
        return particles

    def _build_vertices(self) -> list[Vertex]:
        vertices: list[Vertex] = [
            Vertex(0, (21, 21, 21)),
            Vertex(1, (21, 21, -21)),
            Vertex(2, (-21, 21, 21)),
            Vertex(3, (21, -21, 21)),
        ]
        self._extend_quark_gluon_vertices(vertices)
        self._extend_electroweak_gauge_vertices(vertices)
        self._extend_higgs_vertices(vertices)
        self._extend_lepton_vertices(vertices)
        return vertices

    def _extend_quark_gluon_vertices(self, vertices: list[Vertex]) -> None:
        for i in range(1, 7):
            vertices.extend(
                [
                    Vertex(4, (21, i, i)),
                    Vertex(5, (21, -i, -i)),
                    Vertex(6, (i, 21, i)),
                    Vertex(7, (-i, 21, -i)),
                    Vertex(9, (-i, i, 21)),
                    Vertex(8, (i, -i, 99), (1.0 / 3.0, 0.0)),
                    Vertex(4, (99, i, i)),
                    Vertex(5, (99, -i, -i)),
                    Vertex(6, (i, 99, i)),
                    Vertex(7, (-i, 99, -i)),
                    Vertex(10, (i, 22, i), self.photon_fermion_coupling(i)),
                    Vertex(11, (-i, 22, -i), self.photon_fermion_coupling(-i)),
                    Vertex(10, (i, 23, i), self.z_fermion_coupling(i)),
                    Vertex(11, (-i, 23, -i), self.z_fermion_coupling(-i)),
                ]
            )
            if i % 2 == 0:
                vertices.append(
                    Vertex(10, (i, -24, i - 1), (self.charged_current_coupling(), 0.0))
                )
                vertices.append(
                    Vertex(11, (-i, 24, -i + 1), (self.charged_current_coupling(), 0.0))
                )
            else:
                vertices.append(
                    Vertex(10, (i, 24, i + 1), (self.charged_current_coupling(), 0.0))
                )
                vertices.append(
                    Vertex(
                        11, (-i, -24, -i - 1), (self.charged_current_coupling(), 0.0)
                    )
                )

    def _extend_electroweak_gauge_vertices(self, vertices: list[Vertex]) -> None:
        ngc = self.neutral_gauge_coupling()
        wc = self.weak_coupling()
        vertices.extend(
            [
                Vertex(12, (24, -24, 23), (-ngc, 0.0)),
                Vertex(12, (-24, 24, 23), (ngc, 0.0)),
                Vertex(12, (24, -24, 22), (-self.charge(24), 0.0)),
                Vertex(12, (-24, 24, 22), (self.charge(24), 0.0)),
                Vertex(12, (24, 22, 24), (self.charge(24), 0.0)),
                Vertex(12, (-24, 22, -24), (self.charge(-24), 0.0)),
                Vertex(12, (22, 24, 24), (-self.charge(24), 0.0)),
                Vertex(12, (22, -24, -24), (-self.charge(-24), 0.0)),
                Vertex(12, (24, 23, 24), (ngc, 0.0)),
                Vertex(12, (-24, 23, -24), (-ngc, 0.0)),
                Vertex(12, (23, 24, 24), (-ngc, 0.0)),
                Vertex(12, (23, -24, -24), (ngc, 0.0)),
                Vertex(13, (24, -24, -23), (wc, 0.0)),
                Vertex(13, (-24, 24, -23), (-wc, 0.0)),
                Vertex(14, (-23, 24, 24), (wc, 0.0)),
                Vertex(14, (-23, -24, -24), (-wc, 0.0)),
                Vertex(15, (24, -23, 24), (-wc, 0.0)),
                Vertex(15, (-24, -23, -24), (wc, 0.0)),
                Vertex(13, (24, 22, 26), (self.charge(24), 0.0)),
                Vertex(13, (-24, 22, -26), (-self.charge(-24), 0.0)),
                Vertex(13, (22, 24, 26), (-self.charge(24), 0.0)),
                Vertex(13, (22, -24, -26), (self.charge(-24), 0.0)),
                Vertex(13, (24, 23, 26), (ngc, 0.0)),
                Vertex(13, (-24, 23, -26), (ngc, 0.0)),
                Vertex(13, (23, 24, 26), (-ngc, 0.0)),
                Vertex(13, (23, -24, -26), (-ngc, 0.0)),
                Vertex(14, (26, 22, 24), (self.charge(26), 0.0)),
                Vertex(14, (-26, 22, -24), (-self.charge(-26), 0.0)),
                Vertex(14, (26, -24, 22), (-self.charge(26), 0.0)),
                Vertex(14, (-26, 24, 22), (self.charge(-26), 0.0)),
                Vertex(14, (26, 23, 24), (ngc, 0.0)),
                Vertex(14, (-26, 23, -24), (ngc, 0.0)),
                Vertex(14, (26, -24, 23), (-ngc, 0.0)),
                Vertex(14, (-26, 24, 23), (-ngc, 0.0)),
                Vertex(15, (22, 26, 24), (-self.charge(26), 0.0)),
                Vertex(15, (22, -26, -24), (self.charge(-26), 0.0)),
                Vertex(15, (24, -26, 22), (self.charge(24), 0.0)),
                Vertex(15, (-24, 26, 22), (-self.charge(-24), 0.0)),
                Vertex(15, (23, 26, 24), (-ngc, 0.0)),
                Vertex(15, (23, -26, -24), (-ngc, 0.0)),
                Vertex(15, (24, -26, 23), (ngc, 0.0)),
                Vertex(15, (-24, 26, 23), (ngc, 0.0)),
            ]
        )

    def _extend_higgs_vertices(self, vertices: list[Vertex]) -> None:
        wc = self.weak_coupling()
        wc2 = wc**2
        cw2 = self.cos_weak**2
        for i in range(1, 7):
            if self.mass(i) == 0.0:
                continue
            coupling = self.mass(i) * wc / (self.mass(24) * 2.0)
            vertices.append(Vertex(16, (i, 25, i), (coupling, 0.0)))
            vertices.append(Vertex(16, (-i, 25, -i), (coupling, 0.0)))
        vertices.extend(
            [
                Vertex(17, (24, -24, 25), (self.mass(24) * wc, 0.0)),
                Vertex(17, (-24, 24, 25), (self.mass(24) * wc, 0.0)),
                Vertex(18, (25, 24, 24), (self.mass(24) * wc, 0.0)),
                Vertex(18, (25, -24, -24), (self.mass(24) * wc, 0.0)),
                Vertex(19, (24, 25, 24), (self.mass(24) * wc, 0.0)),
                Vertex(19, (-24, 25, -24), (self.mass(24) * wc, 0.0)),
                Vertex(
                    17,
                    (23, 23, 25),
                    (self.mass(23) * self.weak_coupling_over_cosine(), 0.0),
                ),
                Vertex(
                    18,
                    (25, 23, 23),
                    (self.mass(23) * self.weak_coupling_over_cosine(), 0.0),
                ),
                Vertex(
                    19,
                    (23, 25, 23),
                    (self.mass(23) * self.weak_coupling_over_cosine(), 0.0),
                ),
                Vertex(
                    20,
                    (25, 25, 25),
                    ((-3.0 / 2.0) * wc * (self.mass(25) ** 2 / self.mass(24)), 0.0),
                ),
                Vertex(17, (24, -24, 127), (0.5 * wc2, 0.0)),
                Vertex(17, (-24, 24, 127), (0.5 * wc2, 0.0)),
                Vertex(17, (23, 23, 127), (0.5 * wc2 / cw2, 0.0)),
                Vertex(20, (125, 25, 25), (1.0, 0.0)),
                Vertex(20, (25, 125, 25), (1.0, 0.0)),
                Vertex(
                    20,
                    (25, 25, 125),
                    ((-3.0 / 4.0) * wc2 * self.mass(25) ** 2 / self.mass(24) ** 2, 0.0),
                ),
                Vertex(20, (127, 25, 25), (1.0, 0.0)),
                Vertex(20, (25, 127, 25), (1.0, 0.0)),
                Vertex(20, (25, 25, 126), (1.0, -10.0)),
                Vertex(18, (126, 23, 23), (-0.5 * wc2 / cw2, 0.0)),
                Vertex(19, (23, 126, 23), (-0.5 * wc2 / cw2, 0.0)),
                Vertex(18, (126, 24, 24), (-0.5 * wc2, 0.0)),
                Vertex(18, (126, -24, -24), (-0.5 * wc2, 0.0)),
                Vertex(19, (24, 126, 24), (-0.5 * wc2, 0.0)),
                Vertex(19, (-24, 126, -24), (-0.5 * wc2, 0.0)),
            ]
        )

    def _extend_lepton_vertices(self, vertices: list[Vertex]) -> None:
        for i in range(1, 4):
            charged = 11 + (2 * i - 2)
            neutrino = 12 + (2 * i - 2)
            vertices.extend(
                [
                    Vertex(
                        21,
                        (charged, -charged, 22),
                        self.photon_fermion_coupling(charged),
                    ),
                    Vertex(
                        22,
                        (-charged, charged, 22),
                        self.photon_fermion_coupling(-charged),
                    ),
                    Vertex(
                        21, (charged, -charged, 23), self.z_fermion_coupling(charged)
                    ),
                    Vertex(
                        22, (-charged, charged, 23), self.z_fermion_coupling(-charged)
                    ),
                    Vertex(
                        21,
                        (charged, -neutrino, -24),
                        (self.charged_current_coupling(), 0.0),
                    ),
                    Vertex(
                        22,
                        (-charged, neutrino, 24),
                        (self.charged_current_coupling(), 0.0),
                    ),
                    Vertex(
                        10,
                        (charged, 22, charged),
                        self.photon_fermion_coupling(charged),
                    ),
                    Vertex(
                        11,
                        (-charged, 22, -charged),
                        self.photon_fermion_coupling(-charged),
                    ),
                    Vertex(
                        23,
                        (22, charged, charged),
                        self.photon_fermion_coupling(charged),
                    ),
                    Vertex(
                        24,
                        (22, -charged, -charged),
                        self.photon_fermion_coupling(-charged),
                    ),
                    Vertex(
                        10, (charged, 23, charged), self.z_fermion_coupling(charged)
                    ),
                    Vertex(
                        11, (-charged, 23, -charged), self.z_fermion_coupling(-charged)
                    ),
                    Vertex(
                        23, (23, charged, charged), self.z_fermion_coupling(charged)
                    ),
                    Vertex(
                        24, (23, -charged, -charged), self.z_fermion_coupling(-charged)
                    ),
                ]
            )

################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for madgraph/various/hepmc_parser.py"""

from __future__ import absolute_import
import os
import tempfile
import unittest

import madgraph.various.hepmc_parser as hepmc_parser

HEPMC_Particle = hepmc_parser.HEPMC_Particle
HEPMC_Vertex = hepmc_parser.HEPMC_Vertex
HEPMC_Event = hepmc_parser.HEPMC_Event
HEPMC_EventFile = hepmc_parser.HEPMC_EventFile

# ---------------------------------------------------------------------------
# Minimal HepMC2 file content used across several tests
# ---------------------------------------------------------------------------
_HEPMC2_SINGLE_EVENT = """\
HepMC::Version 2.06.09
HepMC::IO_GenEvent-START_EVENT_LISTING
E 249 -1 -1.0000000000000000e+00 -1.0000000000000000e+00 -1.0000000000000000e+00 0 0 462 1 2 0 1 8.2247251000000005e-22
N 1 "0000"
U GEV MM
C 0.117500 0.007546
F 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0 0
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6.5000000000000000e+03 6.5000000000000000e+03 9.3827200000000005e-01 4 0 0 0 0
P 2 2212 0 0 -6.5000000000000000e+03 6.5000000000000000e+03 9.3827200000000005e-01 4 0 0 0 0
V -2 0 0 0 0 0 0 2 0
P 3 1 0 0 3.0332529367341937e+01 3.0332529367341937e+01 0 3 0 0 -1 1 2 501
P 4 -1 0 0 -3.0332529367341937e+01 3.0332529367341937e+01 0 3 0 0 -2 1 1 501
HepMC::IO_GenEvent-END_EVENT_LISTING
"""

_HEPMC2_TWO_EVENTS = """\
HepMC::Version 2.06.09
HepMC::IO_GenEvent-START_EVENT_LISTING
E 1 -1 -1.0 0.118 0.007 0 -1 2 1 2 0 1 1.0
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0
P 2 2212 0 0 -6500 6500 0.938 4 0 0 0 0
E 2 -1 -1.0 0.118 0.007 0 -1 2 1 2 0 1 2.0
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0
P 2 2212 0 0 -6500 6500 0.938 4 0 0 0 0
HepMC::IO_GenEvent-END_EVENT_LISTING
"""


class TestHEPMCParticle(unittest.TestCase):
    """Tests for HEPMC_Particle."""

    # Canonical P-line from the HepMC2 format:
    # P barcode pdg px py pz E mass status pol_theta pol_phi vtx_bar nb_flows [flow_pairs]
    _PLINE = "P 3 -2 0 0 3.0332529367341937e+01 3.0332529367341937e+01 0 21 0 0 -3 1 2 501"

    def test_default_construction(self):
        """HEPMC_Particle() with no arguments uses zero defaults."""
        p = HEPMC_Particle()
        self.assertEqual(p.barcode, 0)
        self.assertEqual(p.pdg, 0)
        self.assertEqual(p.px, 0)
        self.assertEqual(p.status, 0)
        self.assertEqual(p.nb_flow_list, 0)
        self.assertEqual(p.flows, [])

    def test_parse_barcode(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.barcode, 3)

    def test_parse_pdg(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.pdg, -2)

    def test_parse_momenta(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertAlmostEqual(p.px, 0.0)
        self.assertAlmostEqual(p.py, 0.0)
        self.assertAlmostEqual(p.pz, 30.332529367341937)
        self.assertAlmostEqual(p.E, 30.332529367341937)

    def test_parse_mass_and_status(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertAlmostEqual(p.mass, 0.0)
        self.assertEqual(p.status, 21)

    def test_parse_polarisation(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertAlmostEqual(p.polarization_theta, 0.0)
        self.assertAlmostEqual(p.polarization_phi, 0.0)

    def test_parse_vertex_barcode(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.vertex_barcode, -3)

    def test_parse_flows(self):
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.nb_flow_list, 1)
        self.assertEqual(p.flows, [(2, 501)])

    def test_parse_no_flows(self):
        line = "P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0"
        p = HEPMC_Particle(line)
        self.assertEqual(p.nb_flow_list, 0)
        self.assertEqual(p.flows, [])

    def test_pdg_code_property(self):
        """pdg_code and pid are aliases for pdg."""
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.pdg_code, p.pdg)

    def test_helicity_property(self):
        """helicity always returns 9 (convention for unknown)."""
        p = HEPMC_Particle(self._PLINE)
        self.assertEqual(p.helicity, 9)

    def test_str_roundtrip_barcode_and_pdg(self):
        """str() output contains the barcode and PDG id."""
        p = HEPMC_Particle(self._PLINE)
        s = str(p)
        parts = s.split()
        self.assertEqual(parts[0], 'P')
        self.assertEqual(int(parts[1]), p.barcode)
        self.assertEqual(int(parts[2]), p.pdg)

    def test_multiple_flows(self):
        """Particle with two colour flows is parsed correctly."""
        line = "P 5 21 0 0 50 50 0 21 0 0 -3 2 1 501 2 502"
        p = HEPMC_Particle(line)
        self.assertEqual(p.nb_flow_list, 2)
        self.assertEqual(p.flows, [(1, 501), (2, 502)])


class TestHEPMCVertex(unittest.TestCase):
    """Tests for HEPMC_Vertex."""

    _VLINE = "V -8 0 0 0 0 0 0 2 0"

    def test_default_construction(self):
        v = HEPMC_Vertex()
        self.assertEqual(v.barcode, 0)
        self.assertEqual(v.nb_outgoing, 0)
        self.assertEqual(v.incoming, [])
        self.assertEqual(v.outcoming, [])

    def test_parse_barcode(self):
        v = HEPMC_Vertex(self._VLINE)
        self.assertEqual(v.barcode, -8)

    def test_parse_nb_outgoing(self):
        v = HEPMC_Vertex(self._VLINE)
        self.assertEqual(v.nb_outgoing, 2)

    def test_parse_nb_orphan(self):
        v = HEPMC_Vertex(self._VLINE)
        self.assertEqual(v.nb_orphan, 0)

    def test_parse_position(self):
        v = HEPMC_Vertex(self._VLINE)
        self.assertAlmostEqual(v.x, 0.0)
        self.assertAlmostEqual(v.y, 0.0)
        self.assertAlmostEqual(v.z, 0.0)
        self.assertAlmostEqual(v.ctau, 0.0)

    def test_parse_no_weights(self):
        v = HEPMC_Vertex(self._VLINE)
        self.assertEqual(v.nb_weight, 0)
        self.assertEqual(v.weights, [])

    def test_add_incoming_and_outcoming(self):
        """add_incoming / add_outcoming append to the respective lists."""
        v = HEPMC_Vertex()
        p1 = HEPMC_Particle()
        p2 = HEPMC_Particle()
        v.add_incoming(p1)
        v.add_outcoming(p2)
        self.assertIn(p1, v.incoming)
        self.assertIn(p2, v.outcoming)

    def test_vertex_stored_in_event(self):
        """When event is provided, the vertex registers itself."""
        evt = HEPMC_Event()
        v = HEPMC_Vertex(self._VLINE, event=evt)
        self.assertIn(-8, evt.vertex)
        self.assertIs(evt.vertex[-8], v)


class TestHEPMCEvent(unittest.TestCase):
    """Tests for HEPMC_Event parsing."""

    # Full E-line example
    _ELINE = ("E 249 -1 -1.0000000000000000e+00 -1.0000000000000000e+00 "
              "-1.0000000000000000e+00 0 0 462 1 2 0 1 8.2247251000000005e-22")

    def test_default_construction(self):
        evt = HEPMC_Event()
        self.assertEqual(evt.event_id, 0)
        self.assertEqual(evt.nb_weight, 0)
        self.assertEqual(evt.weights, [])
        self.assertEqual(evt.particles, {})
        self.assertEqual(evt.vertex, {})

    def test_parse_e_line_event_id(self):
        evt = HEPMC_Event()
        evt.parse_E(self._ELINE)
        self.assertEqual(evt.event_id, 249)

    def test_parse_e_line_coupling_constants(self):
        evt = HEPMC_Event()
        evt.parse_E(self._ELINE)
        self.assertAlmostEqual(evt.alphas, -1.0)
        self.assertAlmostEqual(evt.alphaew, -1.0)

    def test_parse_e_line_single_weight(self):
        evt = HEPMC_Event()
        evt.parse_E(self._ELINE)
        self.assertEqual(evt.nb_weight, 1)
        self.assertAlmostEqual(evt.weights[0], 8.2247251e-22)

    def test_parse_e_line_multiple_weights(self):
        eline = "E 1 0 100.0 0.118 0.0073 0 -1 2 1 2 0 2 0.5 1.5"
        evt = HEPMC_Event()
        evt.parse_E(eline)
        self.assertEqual(evt.nb_weight, 2)
        self.assertAlmostEqual(evt.weights[0], 0.5)
        self.assertAlmostEqual(evt.weights[1], 1.5)

    def test_parse_e_line_zero_weights(self):
        eline = "E 1 0 100.0 0.118 0.0073 0 -1 2 1 2 0 0"
        evt = HEPMC_Event()
        evt.parse_E(eline)
        self.assertEqual(evt.nb_weight, 0)
        self.assertEqual(evt.weights, [])

    def test_wgt_property_with_weight(self):
        evt = HEPMC_Event()
        evt.parse_E(self._ELINE)
        self.assertAlmostEqual(evt.wgt, 8.2247251e-22)

    def test_wgt_property_no_weight(self):
        """wgt returns 0 when there are no weights."""
        evt = HEPMC_Event()
        self.assertEqual(evt.wgt, 0.0)

    def test_wgt_setter(self):
        """Setting wgt creates exactly one weight entry."""
        evt = HEPMC_Event()
        evt.wgt = 3.14
        self.assertEqual(evt.nb_weight, 1)
        self.assertAlmostEqual(evt.weights[0], 3.14)
        self.assertAlmostEqual(evt.wgt, 3.14)

    def test_wgt_setter_overwrite(self):
        """Setting wgt twice keeps only the latest value."""
        evt = HEPMC_Event()
        evt.wgt = 1.0
        evt.wgt = 2.0
        self.assertAlmostEqual(evt.wgt, 2.0)

    def test_parse_auxiliary_lines(self):
        """N/U/C/H/F lines are stored verbatim."""
        evt = HEPMC_Event()
        evt.parse_N('N 1 "weights"')
        evt.parse_U('U GEV MM')
        evt.parse_C('C 0.118 0.007')
        evt.parse_H('H some header')
        evt.parse_F('F 0 0 0 0 0 0 0 0 0')
        self.assertIn('N', evt.N)
        self.assertIn('U', evt.U)
        self.assertIn('C', evt.C)

    def test_full_event_particle_count(self):
        """Parsing a full event text gives the expected number of particles."""
        text = """\
E 249 -1 -1.0 -1.0 -1.0 0 0 462 1 2 0 1 8.22e-22
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0
P 2 2212 0 0 -6500 6500 0.938 4 0 0 0 0
V -2 0 0 0 0 0 0 2 0
P 3 1 0 0 30 30 0 3 0 0 -1 1 2 501
P 4 -1 0 0 -30 30 0 3 0 0 -2 1 1 501
"""
        evt = HEPMC_Event(text)
        self.assertEqual(len(evt.particles), 4)

    def test_full_event_vertex_count(self):
        text = """\
E 249 -1 -1.0 -1.0 -1.0 0 0 462 1 2 0 1 8.22e-22
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0
P 2 2212 0 0 -6500 6500 0.938 4 0 0 0 0
V -2 0 0 0 0 0 0 2 0
P 3 1 0 0 30 30 0 3 0 0 -1 1 2 501
P 4 -1 0 0 -30 30 0 3 0 0 -2 1 1 501
"""
        evt = HEPMC_Event(text)
        self.assertEqual(len(evt.vertex), 2)

    def test_full_event_event_id(self):
        text = ("E 249 -1 -1.0 -1.0 -1.0 0 0 462 1 2 0 1 8.22e-22\n"
                "V -1 0 0 0 0 0 0 0 0\n")
        evt = HEPMC_Event(text)
        self.assertEqual(evt.event_id, 249)

    def test_add_particle(self):
        evt = HEPMC_Event()
        p = HEPMC_Particle()
        p.barcode = 42
        evt.add_particle(p)
        self.assertIn(42, evt.particles)
        self.assertIs(evt.particles[42], p)

    def test_add_vertex(self):
        evt = HEPMC_Event()
        v = HEPMC_Vertex()
        v.barcode = -5
        evt.add_vertex(v)
        self.assertIn(-5, evt.vertex)

    def test_iteration(self):
        """Iterating over an event yields its particles."""
        text = """\
E 1 -1 -1.0 -1.0 -1.0 0 0 1 1 2 0 1 1.0
V -1 0 0 0 0 0 2 2 0
P 1 2212 0 0 6500 6500 0.938 4 0 0 0 0
P 2 2212 0 0 -6500 6500 0.938 4 0 0 0 0
"""
        evt = HEPMC_Event(text)
        particles = list(evt)
        self.assertEqual(len(particles), 2)
        self.assertTrue(all(isinstance(p, HEPMC_Particle) for p in particles))


class TestHEPMCEventFile(unittest.TestCase):
    """Tests for HEPMC_EventFile iteration and file handling."""

    def _write_tmp(self, content):
        """Write *content* to a named temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix='.hepmc')
        os.close(fd)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_single_event_iteration(self):
        """A file with one event yields exactly one HEPMC_Event."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            events = list(HEPMC_EventFile(path))
            self.assertEqual(len(events), 1)
        finally:
            os.unlink(path)

    def test_single_event_content(self):
        """The parsed event has the correct id and weight."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            evt = next(iter(HEPMC_EventFile(path)))
            self.assertEqual(evt.event_id, 249)
            self.assertAlmostEqual(evt.wgt, 8.2247251e-22)
        finally:
            os.unlink(path)

    def test_two_events_iteration(self):
        """A file with two events yields two HEPMC_Event objects."""
        path = self._write_tmp(_HEPMC2_TWO_EVENTS)
        try:
            events = list(HEPMC_EventFile(path))
            self.assertEqual(len(events), 2)
        finally:
            os.unlink(path)

    def test_two_events_order_and_ids(self):
        """Events are returned in file order with correct ids."""
        path = self._write_tmp(_HEPMC2_TWO_EVENTS)
        try:
            events = list(HEPMC_EventFile(path))
            self.assertEqual(events[0].event_id, 1)
            self.assertEqual(events[1].event_id, 2)
        finally:
            os.unlink(path)

    def test_two_events_weights(self):
        """Each event has the weight recorded in its E-line."""
        path = self._write_tmp(_HEPMC2_TWO_EVENTS)
        try:
            events = list(HEPMC_EventFile(path))
            self.assertAlmostEqual(events[0].wgt, 1.0)
            self.assertAlmostEqual(events[1].wgt, 2.0)
        finally:
            os.unlink(path)

    def test_file_header_captured(self):
        """The file header (before START_EVENT_LISTING) is captured."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            ef = HEPMC_EventFile(path)
            self.assertIn('HepMC', ef.header)
        finally:
            os.unlink(path)

    def test_is_iterable(self):
        """HEPMC_EventFile implements the iterator protocol."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            ef = HEPMC_EventFile(path)
            self.assertIs(iter(ef), ef)
        finally:
            os.unlink(path)

    def test_stop_iteration_at_end(self):
        """Iteration raises StopIteration after the last event."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            ef = HEPMC_EventFile(path)
            next(ef)
            with self.assertRaises(StopIteration):
                next(ef)
        finally:
            os.unlink(path)

    def test_name_property(self):
        """name property returns the underlying file path."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            ef = HEPMC_EventFile(path)
            self.assertEqual(ef.name, path)
        finally:
            os.unlink(path)

    def test_closed_property_open(self):
        """closed is False while the file is open."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            ef = HEPMC_EventFile(path)
            self.assertFalse(ef.closed)
        finally:
            os.unlink(path)

    def test_particles_in_single_event(self):
        """The single event in the test file contains 4 particles."""
        path = self._write_tmp(_HEPMC2_SINGLE_EVENT)
        try:
            evt = next(iter(HEPMC_EventFile(path)))
            self.assertEqual(len(evt.particles), 4)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()

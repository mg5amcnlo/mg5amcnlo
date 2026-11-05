################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Which output formats a loop-induced ([noborn=...]) process may be sent to.

A loop-induced process does not reach the exporters the way a [virt=...] one
does.  master_interface borrows the MadLoop interface only long enough to
validate the model, then switches *back* to 'MadGraph' and calls
create_loop_induced -- so the process is exported by the ordinary tree-level
output machinery (madgraph_interface.do_output, then ExportV4Factory with
output_type='default', or ExportCPPFactory) even though the amplitude is a
LoopAmplitude and the matrix element a LoopHelasMatrixElement.

Every format used to hand that loop matrix element to a tree-level exporter and
die deep inside it -- 'output standalone_fortran' with an IndexError in
write_check_sa, 'output matrix' with "wavefunction_rank has not been computed",
'output mg7' with a KeyError on the first loop leg, which the mg7 exporter's
edge-name map does not contain.  Two things were wrong:

  * `HelasMatrixElement._flavor_enumeration_context` counted *every* motherless
    wavefunction as an external leg.  A LoopHelasMatrixElement's L-cut
    wavefunctions are motherless too, so `get_external_flavors()` returned
    tuples of length nexternal+2 -- which is what write_check_sa tripped over;
  * the exporter itself was the tree-level one.  'standalone_fortran' is now
    routed to the MadLoop standalone exporter (LOOP_INDUCED_FORMATS), joining
    'madevent' which has always had the LoopInducedExporterME* exporters.

Mind the vocabulary: the format with the MadLoop backend is the *Fortran*
standalone, which is spelled 'standalone_fortran' since the export formats were
renamed.  Plain 'standalone' now names the MadMatrix (C++) output, which has no
loop backend and must be refused like any other.

The formats with no MadLoop backend at all still cannot serve the process, and
refuse it up front pointing at [sqrvirt=...], which stays in the MadLoop
interface and yields the very same matrix element.

The [sqrvirt=]/[virt=] tests at the end are important to keep green: the
refusal keys off the amplitude being a LoopAmplitude, and those amplitudes are
LoopAmplitudes too -- they are spared only because they never reach these two
factories.  If that ever stops being true, the escape hatch the error message
recommends would be refused along with everything else.
"""

from __future__ import absolute_import

import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_exporters as loop_exporters
import madgraph.loop.loop_helas_objects as loop_helas_objects

from madgraph import InvalidCmd, MG5DIR

pjoin = os.path.join

# The cheapest loop-induced process there is: no tree-level diagram exists for
# g g > h in loop_sm, so [noborn=QCD] gives the quark-loop amplitude alone.
LOOP_INDUCED_PROCESS = 'g g > h [noborn=QCD]'
# A 2 -> 2 one, to check the leg count is not accidentally right.
LOOP_INDUCED_PROCESS_2TO2 = 'g g > z z [noborn=QCD]'
# The same matrix element, generated the way the refusal message recommends.
SQRVIRT_PROCESS = 'g g > h [sqrvirt=QCD]'
# An ordinary virtual correction, for good measure.
VIRTUAL_PROCESS = 'u u~ > d d~ [virt=QCD]'



def get_interface(process=LOOP_INDUCED_PROCESS, model='loop_sm'):
    """A fresh MasterCmd with 'process' generated, ready to be output.

    Not cached: 'output' populates _curr_matrix_elements, and a second output
    of a different format on the same interface would reuse them.
    """

    interface = MGCmd.MasterCmd()
    interface.no_notification()
    if model:
        interface.exec_cmd('import model %s' % model, printcmd=False,
                           precmd=True)
    interface.exec_cmd('generate %s' % process, printcmd=False, precmd=True)
    return interface


# building these is the slow part and they are only read, so cache them
_matrix_elements = {}


def get_matrix_element(process):
    """The LoopHelasMatrixElement of a loop-induced process."""

    if process not in _matrix_elements:
        # compute_loop_nc as group_subprocs builds them: building a loop ME
        # without it leaves colour data that a later output picks up (a
        # pre-existing cross-talk, reproducible on a clean tree)
        _matrix_elements[process] = loop_helas_objects.LoopHelasProcess(
            get_interface(process)._curr_amps, optimized_output=True,
            compute_loop_nc=True).get_matrix_elements()[0]
    return _matrix_elements[process]


#===============================================================================
# TestLoopInducedExternalFlavors
#===============================================================================
class TestLoopInducedExternalFlavors(unittest.TestCase):
    """get_external_flavors() must describe the external legs only.

    The L-cut wavefunctions of a loop matrix element are motherless like the
    external ones and carry number_external = nexternal+1 / nexternal+2, so they
    used to be counted as two extra external legs.
    """

    def test_loop_induced_external_flavors(self):
        """g g > h [noborn=QCD] has 3 external legs, not 5."""

        me = get_matrix_element(LOOP_INDUCED_PROCESS)
        self.assertEqual(me.get_nexternal_ninitial(), (3, 2))
        flavors, pdgs = me.get_external_flavors(return_pdgs=True)
        self.assertEqual([tuple(f) for f in flavors], [(1, 1, 1)])
        self.assertEqual([list(p) for p in pdgs], [[21, 21, 25]])

    def test_loop_induced_external_flavors_4legs(self):
        """Same for a 2 -> 2 loop-induced process."""

        me = get_matrix_element(LOOP_INDUCED_PROCESS_2TO2)
        self.assertEqual(me.get_nexternal_ninitial(), (4, 2))
        flavors, pdgs = me.get_external_flavors(return_pdgs=True)
        self.assertEqual([tuple(f) for f in flavors], [(1, 1, 1, 1)])
        self.assertEqual([list(p) for p in pdgs], [[21, 21, 23, 23]])


#===============================================================================
# TestLoopInducedOutput
#===============================================================================
class TestLoopInducedOutput(unittest.TestCase):
    """The output formats a loop-induced process may and may not be sent to."""

    def setUp(self):
        # 'output' chdir's into the directory it writes; come back before it is
        # removed, or the next MasterCmd() cannot even call os.getcwd()
        os.chdir(MG5DIR)
        self.tmpdir = tempfile.mkdtemp(prefix='loop_induced_output')

    def tearDown(self):
        os.chdir(MG5DIR)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    #===========================================================================
    # helpers
    #===========================================================================
    def get_exporter(self, interface, format='standalone_fortran'):
        """The exporter ExportV4Factory picks for 'format'."""

        interface._export_format = format
        interface._export_dir = pjoin(self.tmpdir, 'factory')
        return export_v4.ExportV4Factory(interface, True,
                                         group_subprocesses=False)

    def assert_output_refused(self, format):
        """'output <format>' must raise InvalidCmd, and say what to do instead.

        Anything that is not an InvalidCmd -- an IndexError, a KeyError, a
        MadGraph5Error from inside a tree-level exporter -- means the loop
        matrix element reached an exporter that cannot write it.
        """

        interface = get_interface()
        out_dir = pjoin(self.tmpdir, format.replace(' ', '_'))
        try:
            interface.exec_cmd('output %s %s -f' % (format, out_dir),
                               printcmd=False, precmd=True)
        except InvalidCmd as error:
            self.assertTrue('sqrvirt' in str(error),
                            "'output %s' refused %s without pointing at "
                            "[sqrvirt=]: %s" % (format, LOOP_INDUCED_PROCESS,
                                                error))
            return
        except Exception as error:
            raise AssertionError(
                "'output %s' crashed on %s with %s: %s"
                % (format, LOOP_INDUCED_PROCESS, type(error).__name__, error))
        raise AssertionError(
            "'output %s' silently accepted %s; no exporter for that format can "
            "write a LoopHelasMatrixElement" % (format, LOOP_INDUCED_PROCESS))

    def assert_output_succeeds(self, process, format='standalone_fortran'):
        """'output <format>' must go through, and write a MadLoop directory."""

        interface = get_interface(process)
        out_dir = pjoin(self.tmpdir, 'ok')
        interface.exec_cmd('output %s %s -f' % (format, out_dir),
                           printcmd=False, precmd=True)
        # A MadLoop output, not a tree-level one: this file only exists when a
        # loop exporter ran.
        self.assertTrue(
            os.path.exists(pjoin(out_dir, 'Cards', 'MadLoopParams.dat')),
            '%s did not produce a MadLoop output' % process)

    #===========================================================================
    # 'standalone_fortran' is served by the MadLoop standalone exporter
    #===========================================================================
    def test_standalone_factory_uses_the_madloop_exporter(self):
        """ExportV4Factory must not hand a loop-induced process to the
        tree-level standalone exporter."""

        interface = get_interface()
        self.assertTrue(isinstance(interface._curr_amps[0],
                                   loop_diagram_generation.LoopAmplitude),
                        "%s did not produce a LoopAmplitude"
                        % LOOP_INDUCED_PROCESS)
        self.assertIsInstance(self.get_exporter(interface),
                              loop_exporters.LoopProcessExporterFortranSA)

    def test_standalone_factory_unchanged_for_tree(self):
        """A tree process must still get the tree-level exporter."""

        interface = get_interface('g g > t t~', model='sm')
        exporter = self.get_exporter(interface)
        self.assertIsInstance(exporter, export_v4.ProcessExporterFortranSA)
        self.assertFalse(isinstance(
            exporter, loop_exporters.LoopProcessExporterFortranSA))

    def test_output_standalone_fortran_accepts_loop_induced(self):
        """'output standalone_fortran' used to die with an IndexError in
        write_check_sa; it now writes a MadLoop standalone directory."""

        self.assert_output_succeeds(LOOP_INDUCED_PROCESS)

    def test_no_model_imported(self):
        """validate_model is called before create_loop_induced's check_add, so
        it has to bootstrap the model itself."""

        interface = MGCmd.MasterCmd()
        interface.no_notification()
        self.assertFalse(interface._curr_model)
        interface.exec_cmd('generate %s' % LOOP_INDUCED_PROCESS,
                           printcmd=False, precmd=True)
        self.assertEqual(interface._curr_model.get('name'), 'loop_sm')

    #===========================================================================
    # ... the formats with no MadLoop backend refuse it
    #===========================================================================
    def test_loop_induced_formats_are_matched_exactly(self):
        """The allow-list must never be tested with startswith.

        'standalone' is a prefix of the format that does have a loop backend
        ('standalone_fortran') and it names the MadMatrix output, which does
        not -- so a startswith test would get this exactly backwards.
        """

        for format in ['standalone', 'standalone_mg7', 'standalone_msP',
                       'standalone_msF', 'standalone_rw']:
            self.assertNotIn(format, export_v4.LOOP_INDUCED_FORMATS)

    def test_output_matrix_refuses_loop_induced(self):
        """'output matrix' shares the standalone branch of the factory; it used
        to die with "wavefunction_rank has not been computed"."""

        self.assert_output_refused('matrix')

    def test_output_mg7_refuses_loop_induced(self):
        """'output mg7' -- the default format -- used to die with a KeyError on
        the first loop leg.  mg7/madmatrix has no MadLoop backend at all."""

        self.assert_output_refused('mg7')

    def test_output_standalone_refuses_loop_induced(self):
        """The prefix trap: plain 'standalone' is the MadMatrix output.

        It must not ride on the 'standalone_fortran' entry of the allow-list.
        While it did, it reached the mg7 exporter and died there with a
        KeyError on the first loop leg -- the very crash the guard in
        ExportCPPFactory exists to prevent.
        """

        self.assert_output_refused('standalone')

    def test_output_standalone_msP_refuses_loop_induced(self):
        """Same for the MadSpin standalone variants."""

        self.assert_output_refused('standalone_msP')

    def assert_refusal_keeps_directory(self, line):
        """A refused 'output' must not delete the directory it would write to.

        This is why do_output checks ahead of its own rmtree: the factories
        refuse too, but they only run once the directory is already gone.
        """

        out_dir = pjoin(self.tmpdir, 'existing')
        os.mkdir(out_dir)
        sentinel = pjoin(out_dir, 'sentinel.txt')
        open(sentinel, 'w').write('do not delete me\n')

        interface = get_interface()
        self.assertRaises(InvalidCmd, interface.exec_cmd, line % out_dir)
        self.assertTrue(os.path.exists(sentinel),
                        '%s wiped the existing directory' % (line % out_dir))

    def test_refusal_leaves_an_existing_directory_alone(self):
        """The refused format is the one asked for."""

        self.assert_refusal_keeps_directory('output mg7 %s -f')

    def test_me_exporter_refusal_leaves_an_existing_directory_alone(self):
        """--me_exporter= writes into the same directory, so it has to be
        refused by the same early check; testing _export_format alone let it
        fall through to the factory, which runs after the rmtree."""

        self.assert_refusal_keeps_directory(
                                'output madevent %s --me_exporter=mg7 -f')

    #===========================================================================
    # ... and the routes that already worked must keep working
    #===========================================================================
    def test_output_madevent_still_accepts_loop_induced(self):
        """madevent is the format loop-induced processes are *for*.

        It has the LoopInducedExporterMEGroup / ...MENoGroup exporters, so it
        must go straight through the refusal above.
        """

        interface = get_interface()
        out_dir = pjoin(self.tmpdir, 'me')
        interface.exec_cmd('output madevent %s -f' % out_dir,
                           printcmd=False, precmd=True)
        self.assertTrue(
            os.path.isdir(pjoin(out_dir, 'SubProcesses', 'MadLoop5_resources')),
            'madevent did not produce a loop-induced output')

    def test_sqrvirt_standalone_output_still_works(self):
        """The escape hatch the refusal message recommends must actually work.

        [sqrvirt=] gives a LoopAmplitude just like [noborn=] does; it is spared
        the refusal only because master_interface keeps it in the MadLoop
        interface, which passes output_type='madloop'.
        """

        self.assert_output_succeeds(SQRVIRT_PROCESS)

    def test_virt_standalone_output_still_works(self):
        """Same for an ordinary [virt=] process."""

        self.assert_output_succeeds(VIRTUAL_PROCESS)

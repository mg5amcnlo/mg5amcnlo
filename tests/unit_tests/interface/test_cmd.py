##############################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
""" Basic test of the command interface """

from __future__ import absolute_import
import unittest
import madgraph
import madgraph.interface.master_interface as cmd
import madgraph.core.base_objects as base_objects
import MadSpin.interface_madspin as ms_cmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.various.misc as misc
import math
import os
import logging

import tests.parallel_tests.test_aloha as test_aloha

import tempfile
pjoin = os.path.join
class TestValidCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    def setUp(self):
        if not hasattr(self, 'cmd'):
            TestValidCmd.cmd = cmd.MasterCmd()
            TestValidCmd.cmd.no_notification(
            )
        self.debugging = False
        if self.debugging:
            self.path = pjoin(MG5DIR, "tmp_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='acc_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 

    
    def wrong(self,*opt):
        self.assertRaises(madgraph.MadGraph5Error, *opt)
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
    
    def test_shell_and_continuation_line(self):
        """ check that the cmd line interpret shell and ; correctly """
        
        #Those tests are important for this type of launch: 
        # cd DIR; ./bin/generate_events 
        try:
            os.remove('/tmp/tmp_file')
        except:
            pass
        
        self.do('! cd /tmp; touch tmp_file')
        self.assertTrue(os.path.exists('/tmp/tmp_file'))
        
        try:
            os.remove('/tmp/tmp_file')
        except:
            pass
        self.do(' ! cd /tmp; touch tmp_file')
        self.assertTrue(os.path.exists('/tmp/tmp_file'))
    
    def test_cleaning_history(self):
        """check that the cleaning of the history command works as expected"""
        
        # Test the call present inside do_generate        
        history="""set cluster_queue 2
        import model mssm
        generate p p > go go 
        add process p p > go go j
        set gauge Feynman
        check p p > go go
        output standalone
        display particles
        generate p p > go go"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history[:] = history
        self.cmd.history.clean(remove_bef_last='generate', keep_switch=True,
                     allow_for_removal= ['generate', 'add process', 'output'])

        goal = """set cluster_queue 2
        import model mssm
        set gauge Feynman
        generate p p > go go"""
        goal = [l.strip() for l in  goal.split('\n')]

        self.assertEqual(self.cmd.history, goal)
        
        # Test the call present in do_import model
        history="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        add process p p > go go j
        set gauge Feynman
        check p p > go go
        output standalone
        display particles
        generate p p > go go
        import heft"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history[:] = history        
        
        self.cmd.history.clean(remove_bef_last='import', keep_switch=True,
                        allow_for_removal=['generate', 'add process', 'output'])

        # Test the call present in do_import model
        goal="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        import model mssm --modelname
        set gauge Feynman
        import heft""" 

        goal = [l.strip() for l in  goal.split('\n')]

        self.assertEqual(self.cmd.history, goal)
        
        
        # Test the call present in do_output
        history="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        output standalone
        launch
        output"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history[:] = history         
        
        self.cmd.history.clean(allow_for_removal = ['output'], keep_switch=True,
                           remove_bef_last='output')

        goal="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        output"""
        
        goal = [l.strip() for l in  goal.split('\n')]
        self.assertEqual(self.cmd.history, goal)
    
    def test_InvalidCmd(self):
        """test that the Invalid Command are dealt with correctly"""
        
        master = cmd.MasterCmd()
        master.no_notification()
        self.assertRaises(master.InvalidCmd, master.do_generate,('aa'))
        try:
            master.run_cmd('aa')
        except Exception as error:
            print(error)
            self.assertTrue(False, 'error are not treated correctly')
        
        # Madspin
        master = ms_cmd.MadSpinInterface()
        master.no_notification()
        self.assertRaises(Exception, master.do_define,('aa'))
        
        with misc.MuteLogger(['fatalerror'], [40],['/tmp/fatalerror.log'], keep=False):
            try:
                master.run_cmd('define aa')
            except Exception as error:
                self.assertTrue(False, 'error are not treated correctly: %s' % error)
            text = open('/tmp/fatalerror.log').read()
            self.assertNotIn('{', text)
            self.assertIn('MS_debug', text)

    def test_help_category(self):
        """Check that no help category are introduced by mistake.
           If this test fails, this is due to a un-expected ':' in a command of
           the cmd interface.
        """
        
        category = set()
        categories_nb = {}
        for interface_class in cmd.MasterCmd.__mro__:
            valid_command = [c for c in dir(interface_class) if c.startswith('do_')]
            name = interface_class.__name__
            if name in ['CmdExtended', 'CmdShell', 'Cmd']:
                continue
            for command in valid_command:
                obj = getattr(interface_class, command)
                if obj.__doc__ and ':' in obj.__doc__:
                    cat = obj.__doc__.split(':',1)[0]
                    category.add(cat)
                    if cat in categories_nb:
                        categories_nb[cat] += 1
                    else:
                        categories_nb[cat] = 1

        target = set(['Not in help', 'Main commands', 'Documented commands'])
        self.assertEqual(target, category)
        self.assertEqual(categories_nb['Not in help'], 29)
    
    @test_aloha.set_global()
    def test_check_import_model(self):    
    
        cmd = self.cmd
        cmd.do_import('sm')
        target ={}
        for obj in cmd._curr_model.get('lorentz'):
            target[str(obj)] = str(obj.structure)
        cmd.do_import('MSSM_SLHA2')
        cmd.do_generate('p p > t t~')
        cmd.do_output(self.run_dir)
        cmd.do_import('sm')
        for obj in cmd._curr_model.get('lorentz'):
            self.assertEqual(target[str(obj)], obj.structure)

        self.assertEqual(cmd._curr_model.get('name'), 'sm')

        import models as ufomodels
        ufomodel = ufomodels.load_model(cmd._curr_model.get('name'))
        for key in target:
            try:
                to_check = getattr(ufomodel.lorentz, key).structure
            except:
                continue
            else:
                self.assertEqual(to_check, target[key])

    @test_aloha.set_global()
    def test_check_generate(self):
        """check if generate format are correctly supported"""
    
        cmd = self.cmd
        cmd.do_import('sm')
        
        # valid syntax
        cmd.check_process_format('e+ e- > e+ e-')
        cmd.check_process_format('e+ e- > mu+ mu- QED=0')
        cmd.check_process_format('e+ e- > mu+ ta- / x $y @1')
        cmd.check_process_format('e+ e- > mu+ ta- $ x /y @1')
        cmd.check_process_format('e+ e- > mu+ ta- $ x /y, (e+ > e-, e-> ta) @1')
        cmd.check_process_format('e+ e- > Z{L}, Z > mu+ mu- @1')
        cmd.check_process_format('e+ e- > Z{0}, Z > mu+ mu- @1')
        cmd.check_process_format('e+{L} e- > mu+{L} mu-{R} @1')
        cmd.check_process_format('e+ e- > t{L} t~ Z{L}, t > mu+ mu- @1')
        cmd.check_process_format('g g > Z Z [ noborn=QCD] @1')
        cmd.check_process_format('u u~ > 2w+ 2j')
        cmd.check_process_format('u u~ > 2w+{0} 2j')
        cmd.check_process_format,'u u~ > e+{L} vl [QED]'
        
        # unvalid syntax
        self.wrong(cmd.check_process_format, ' e+ e-')
        self.wrong(cmd.check_process_format, ' e+ e- > e+ e-,')
        self.wrong(cmd.check_process_format, ' e+ e- > > e+ e-')
        self.wrong(cmd.check_process_format, ' e+ e- > j / g > e+ e-')        
        self.wrong(cmd.check_process_format, ' e+ e- > j $ g > e+  e-')         
        self.wrong(cmd.check_process_format, ' e+ > j / g > e+ > e-')        
        self.wrong(cmd.check_process_format, ' e+ > j $ g > e+ > e-')
        self.wrong(cmd.check_process_format, ' e+ > e+, (e+ > e- / z, e- > top')   
        self.wrong(cmd.check_process_format, 'e+ > ')
        self.wrong(cmd.check_process_format, 'e+ >')
        self.wrong(cmd.check_process_format, 'e+ e- > Z{L} > mu+ mu-')
        self.wrong(cmd.check_process_format, 'e+ e- > Z > mu+ mu- / W+{L}')
        self.wrong(cmd.check_process_format, 'e+ e- > Z > mu+ mu- $ W+{L}')
        self.wrong(cmd.check_process_format, 'u u~ > t{L} t~ [QCD]')
        self.wrong(cmd.check_process_format, 'u u~ > W+{L} vl [ QED QCD]')
        self.wrong(cmd.check_process_format,'u u~ > w+{L} [QCD]')
        self.wrong(cmd.check_process_format,'u u~ > e+{L} vl [QED]')
        
    @test_aloha.set_global()
    def test_output_default(self):
        """check that if a export_dir is define before an output
           a new one is propose"""
           
        cmd = self.cmd
        cmd._export_dir = 'tmp'
        cmd._curr_amps = 'dummy'
        cmd._curr_model = {'name':'WHY'}
        cmd.check_output([])
        
        self.assertNotEqual('tmp', cmd._export_dir)
        
    @test_aloha.set_global()
    def test_simple_generate(self):
        """check that simple syntax goes trough and return expected process"""
           
        cmd = self.cmd
        self.do('import model sm')
        self.do('generate 2p > 2j')
        self.assertTrue(cmd._curr_amps)
        proc = cmd._curr_amps[0].get('process').get('legs')
        self.assertEqual(len(proc), 4)
        
    @test_aloha.set_global()
    def test_generate_polarised(self):
        """check that simple syntax goes trough and return expected process"""
           
        cmd = self.cmd
        self.do('import model sm')
        self.do('define v = z a')
        self.do('generate v{0} v{0} > w+ w-')
        self.assertTrue(cmd._curr_amps)
        self.assertEqual(len(cmd._curr_amps), 1)
        proc = cmd._curr_amps[0].get('process').get('legs')
        self.assertEqual(len(proc), 4)

        self.do('generate v v > w+ w-')
        self.assertEqual(len(cmd._curr_amps), 3)

        try:
            self.do('generate v a{0} > w+ w-')
        except madgraph.core.diagram_generation.NoDiagramException:
            pass # a{0} should crash since a is massless
        else:
            raise Exception("photon should not generate diagram when Longitudinally polarised.") 
        
        self.do('generate v{0T} v{0} > w+ w-')
        self.assertEqual(len(cmd._curr_amps), 2)

    def test_consider_axial_is_not_a_madgraph_option(self):
        """'consider_axial' is deliberately NOT a global MG5 option: whether
        the axial state may be written out is an *output* question
        ('output standalone --allow_axial') and, for MadSpin, a MadSpin card
        option. See check_axial_output()."""
        cmd = self.cmd
        self.assertNotIn('consider_axial', cmd.options_madgraph)
        self.assertNotIn('consider_axial', cmd.options)
        self.assertNotIn('consider_axial', cmd._set_options)
        self.assertRaises(madgraph.InvalidCmd,
                          cmd.exec_cmd, 'set consider_axial True')
        # ... it lives in the MadSpin card instead
        import MadSpin.interface_madspin as ms_interface
        self.assertIn('consider_axial', ms_interface.MadSpinOptions())

    @test_aloha.set_global()
    def test_generate_axial_polarisation(self):
        """{A} on a FINAL-STATE particle needs the off-shell '*'. Generation
        itself is then unconditional; it is the OUTPUT that has to be asked
        for it explicitly with '--allow_axial'."""
        import madgraph.core.helas_objects as helas_objects

        cmd = self.cmd
        self.do('import model sm')
        try:
            # no star: '{A}' is not a polarisation of an on-shell particle
            self.assertRaises(madgraph.InvalidCmd,
                              self.do, 'generate p p > z{A} h')
            try:
                self.do('generate p p > z{A} h')
            except madgraph.InvalidCmd as error:
                self.assertIn('off-shell', str(error))

            # initial state is refused (MadSpin's decay-side ME would want
            # it one day; it is not wired up)
            self.assertRaises(madgraph.InvalidCmd,
                              self.do, 'generate z{A}* z > w+ w-')

            # the propagator syntax is unchanged
            self.do('generate t > w+{A} b, w+ > ta+ vt')
            self.assertTrue(cmd._curr_amps)

            # with the star, generation goes through with no option at all
            self.do('generate p p > z{A}* h')
            self.assertTrue(cmd._curr_amps)
            legs = cmd._curr_amps[0].get('process').get('legs')
            axial = [l for l in legs if l.get('polarization') == [99]]
            self.assertEqual(len(axial), 1)
            self.assertTrue(axial[0].get('state'))
            self.assertTrue(axial[0].get('offshell'))

            # and the matrix element now builds: '{A}' has an external
            # wavefunction (VXXXXX with nhel = 4).
            me = helas_objects.HelasMultiProcess(
                cmd._curr_amps).get('matrix_elements')[0]

            # the NHEL table carries 4 (the HELAS scalar-polarisation
            # convention), NEVER the raw Leg-level 99
            hel_matrix = list(me.get_helicity_matrix())
            values = set(h for row in hel_matrix for h in row)
            self.assertIn(4, values)
            self.assertNotIn(99, values)

            # get_helicity_per_particle -- what fills the ALLOW_HEL rows of
            # GET_DENSITY -- must agree with it, otherwise GET_DENSITY's outer
            # loop matches nothing and every density comes back zero
            per_part = me.get_helicity_per_particle()
            for i, choices in enumerate(per_part):
                self.assertEqual(set(choices),
                                 set(row[i] for row in hel_matrix))
            self.assertNotIn(99, [h for c in per_part for h in c])
        finally:
            cmd.exec_cmd('generate p p > t t~')

    def test_polarization_to_helicities(self):
        """the one place where a polarization brace becomes a helicity."""
        import madgraph.core.base_objects as base_objects
        # physical entries pass through
        self.assertEqual(
            base_objects.polarization_to_helicities([1, -1, 0]), [1, -1, 0])
        # '{A}' (99) becomes the HELAS scalar polarisation nhel = 4
        self.assertEqual(base_objects.polarization_to_helicities([99]), [4])
        self.assertEqual(
            base_objects.polarization_to_helicities([1, -1, 0, 99]),
            [1, -1, 0, 4])
        # order is preserved: GET_DENSITY takes ALLOW_HEL(1) from the first
        # entry and matches it against the NHEL table
        self.assertEqual(
            base_objects.polarization_to_helicities([99, 1, -1, 0]),
            [4, 1, -1, 0])

    @test_aloha.set_global()
    def test_axial_output_needs_allow_axial(self):
        """'{A}' on a final state is only written out when the output line
        asks for it."""
        cmd = self.cmd
        self.do('import model sm')
        try:
            self.do('generate t > w+{A}* b')
            legs = cmd.collect_axial_external_legs(cmd._curr_amps)
            self.assertEqual(len(legs), 1)

            # no flag -> refused, and the message names the flag
            try:
                cmd.check_output(['standalone', '/tmp/should_not_exist', '-f'])
            except madgraph.InvalidCmd as error:
                self.assertIn('--allow_axial', str(error))
            else:
                self.fail('output accepted {A} without --allow_axial')

            # flag, but a format with no off-shell external leg support
            try:
                cmd.check_output(['standalone_cpp', '/tmp/should_not_exist',
                                  '-f', '--allow_axial'])
            except madgraph.InvalidCmd as error:
                self.assertIn('standalone', str(error))
            else:
                self.fail('standalone_cpp accepted an off-shell {A} leg')

            # flag + fortran standalone -> accepted
            cmd.check_output(['standalone', '/tmp/should_not_exist', '-f',
                              '--allow_axial'])

            # a '{A}' that is a propagator is not an external leg and needs
            # nothing
            self.do('generate t > w+{A} b, w+ > ta+ vt')
            self.assertEqual(cmd.collect_axial_external_legs(cmd._curr_amps),
                             [])
            cmd.check_output(['standalone', '/tmp/should_not_exist', '-f'])
        finally:
            cmd.exec_cmd('generate p p > t t~')

    @test_aloha.set_global()
    def test_generate_propagator_only_polarisation(self):
        """{G},{H},{Q},{W},{S} name a piece of the propagator *numerator* of a
        massive vector; there is no external wavefunction for them. They stay
        valid on a leg that is decayed further and are refused everywhere
        else."""
        import madgraph.core.helas_objects as helas_objects

        cmd = self.cmd
        self.do('import model sm')
        tags = ['G', 'H', 'Q', 'W', 'S']
        try:
            for tag in tags:
                # --- refused on a genuine final state ------------------
                for proc in ('generate p p > z{%s} h' % tag,
                             # the off-shell '*' does not rescue them either
                             'generate p p > z{%s}* h' % tag):
                    self.assertRaises(madgraph.InvalidCmd, self.do, proc)
                    try:
                        self.do(proc)
                    except madgraph.InvalidCmd as error:
                        self.assertIn('{%s}' % tag, str(error))
                        self.assertIn('propagator', str(error))
                        self.assertIn('decayed further', str(error))
                        self.assertIn('final-state particle', str(error))

                # --- refused on an initial state ----------------------
                self.assertRaises(madgraph.InvalidCmd,
                                  self.do, 'generate z{%s} z > w+ w-' % tag)
                try:
                    self.do('generate z{%s} z > w+ w-' % tag)
                except madgraph.InvalidCmd as error:
                    self.assertIn('initial-state particle', str(error))

                # --- still fine as a propagator -----------------------
                self.do('generate t > w+{%s} b, w+ > ta+ vt' % tag)
                self.assertTrue(cmd._curr_amps)

            # the combined '{0S}' brace (pol=[0,9], propagator form P1LS) is
            # covered by the same rule through its '9' entry
            self.assertRaises(madgraph.InvalidCmd,
                              self.do, 'generate p p > z{0S} h')
            self.do('generate t > w+{0S} b, w+ > ta+ vt')
            self.assertTrue(cmd._curr_amps)

            # the walk recurses into the decay chains: here the '{G}' sits one
            # level down, on a w+ that is itself never decayed
            self.assertRaises(
                madgraph.InvalidCmd, self.do,
                'generate p p > t t~, t > w+{G} b, t~ > w- b~')

            # the guard is duplicated one layer down, for the direct-API path
            # that does not go through the command interface at all
            legs = base_objects.LegList([
                base_objects.Leg({'id': 2, 'state': False, 'number': 1}),
                base_objects.Leg({'id': -2, 'state': False, 'number': 2}),
                base_objects.Leg({'id': 23, 'state': True, 'number': 3,
                                  'polarization': [4]}),
                base_objects.Leg({'id': 25, 'state': True, 'number': 4}),
                ])
            try:
                helas_objects.HelasWavefunction(legs[2], 0,
                                                cmd._curr_model)
            except madgraph.InvalidCmd as error:
                self.assertIn('{G}', str(error))
                self.assertIn('propagator', str(error))
            else:
                self.fail('HelasWavefunction accepted a {G} external leg')
        finally:
            cmd.exec_cmd('generate p p > t t~')

    def test_propagator_polarisation_round_trip(self):
        """nice_string()/input_string() used to print the raw integer ({4},
        {99}, ...), which the parser rejects with "polarization are between -3
        and 3". They must print the brace letter instead."""
        cmd = self.cmd
        self.do('import model sm')
        try:
            self.do('generate p p > z{A}* h')
            proc = cmd._curr_amps[0].get('process')
            self.assertIn('z{A}*', proc.nice_string(prefix=False))
            self.assertIn('z{A}*', proc.input_string())
            self.assertIn('z{A}*', proc.base_string())
            # and the printed string is accepted back by the parser
            self.do('generate %s' % proc.input_string())
            self.assertTrue(cmd._curr_amps)

            # the propagator braces print their letter too, including the
            # combined '{0S}' which must not grow a comma (a ',' inside the
            # brace also breaks the decay-chain split)
            for tag, expected in (('G', 'w+{G}'), ('H', 'w+{H}'),
                                  ('Q', 'w+{Q}'), ('W', 'w+{W}'),
                                  ('S', 'w+{S}'), ('0S', 'w+{0S}')):
                self.do('generate t > w+{%s} b, w+ > ta+ vt' % tag)
                proc = cmd._curr_amps[0].get('amplitudes')[0].get('process')
                self.assertIn(expected, proc.nice_string(prefix=False))
                self.assertIn(expected, proc.input_string())
                self.assertNotIn(',', proc.input_string())
        finally:
            cmd.exec_cmd('generate p p > t t~')



class TestAxialWavefunction(unittest.TestCase):
    """The external wavefunction of the axial polarisation '{A}'.

    HELAS has always called that state nhel = 4. It is eps_A^mu = p^mu/vmass,
    and for the off-shell external leg '{A}' needs, vmass is sqrt(p.p) -- so
    eps_A is the unit vector p^mu/sqrt(p.p) that completes the three physical
    polarisations into an orthonormal tetrad."""

    def test_python_wavefunction(self):
        """the reference implementation: aloha/template_files/wavefunctions.py"""
        import aloha.template_files.wavefunctions as wavefunctions

        Q = 60.
        p = [0., 21., -33.5, 44.7]
        p[0] = math.sqrt(p[1]**2 + p[2]**2 + p[3]**2 + Q**2)
        metric = [1., -1., -1., -1.]

        def dot(a, b):
            return sum(metric[i] * a[i] * b[i] for i in range(4))

        eps = {}
        for nhel in (-1, 0, 1, 4):
            wf = wavefunctions.vxxxxx(p, Q, nhel, 1)
            eps[nhel] = [complex(wf[i]) for i in (2, 3, 4, 5)]

        # the axial state IS p^mu / sqrt(p.p)
        for i in range(4):
            self.assertAlmostEqual(eps[4][i].real, p[i] / Q, 10)
            self.assertAlmostEqual(eps[4][i].imag, 0., 12)

        # ... a unit TIMELIKE vector, where the physical ones are spacelike
        self.assertAlmostEqual(
            dot(eps[4], [c.conjugate() for c in eps[4]]).real, 1., 10)
        for nhel in (-1, 0, 1):
            self.assertAlmostEqual(
                dot(eps[nhel], [c.conjugate() for c in eps[nhel]]).real, -1., 10)
            # ... and orthogonal to it (the physical states are transverse
            # to p, the axial one is along p)
            self.assertAlmostEqual(
                abs(dot(eps[nhel], [c.conjugate() for c in eps[4]])), 0., 10)

    def test_fortran_and_cpp_templates_have_a_live_branch(self):
        """The nhel = 4 branch used to sit commented out inside a disabled
        '#ifdef HELAS_CHECK' -- and with the OLD component indices, writing
        over vc(1:2) which now hold the momentum. Guard both."""
        import madgraph

        files = {'aloha_functions.f': ('vc(3)', 'vc(4)', 'vc(5)', 'vc(6)'),
                 # the loop convention puts the momentum in vc(1:4)
                 'aloha_functions_loop.f': ('vc(5)', 'vc(6)', 'vc(7)', 'vc(8)'),
                 }
        for name, components in files.items():
            path = os.path.join(madgraph.MG5DIR, 'aloha', 'template_files', name)
            text = open(path).read()
            start = text.index('      subroutine vxxxxx')
            end = text.index('\n      subroutine ', start + 10)
            body = text[start:end]
            live = [l for l in body.split('\n')
                    if not l.startswith('c') and 'nhel.eq.4' in l]
            self.assertTrue(live, '%s has no live nhel = 4 branch' % name)
            branch = body[body.index(live[0]):]
            branch = branch[:branch.index('endif')]
            for comp in components:
                self.assertIn(comp + ' = ', branch,
                              '%s: nhel = 4 does not fill %s' % (name, comp))

        path = os.path.join(madgraph.MG5DIR, 'aloha', 'template_files',
                            'vxxxxx.cc')
        text = open(path).read()
        self.assertIn('nhel == 4', text)
        for comp in ('vc[2]', 'vc[3]', 'vc[4]', 'vc[5]'):
            self.assertIn(comp, text)

    def test_offshell_leg_is_called_with_the_virtuality(self):
        """Both external-wavefunction writers must feed sqrt(p.p), not the
        pole mass, to a leg carrying the off-shell '*'. That is what makes the
        axial state p^mu/sqrt(p.p)."""
        import madgraph.core.base_objects as base_objects
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.helas_call_writers as helas_call_writers
        import madgraph.interface.master_interface as master

        mgcmd = master.MasterCmd()
        mgcmd.exec_cmd('import model sm')
        mgcmd.exec_cmd('generate t > w+{A}* b')
        me = helas_objects.HelasMultiProcess(
            mgcmd._curr_amps).get('matrix_elements')[0]
        wf = [w for w in me.get_external_wavefunctions()
              if w.get('polarization') == [99]][0]

        fortran = helas_call_writers.FortranUFOHelasCallWriter(
            mgcmd._curr_model).get_wavefunction_call(wf)
        self.assertIn('SQRT(P(0,2)**2', fortran)
        self.assertNotIn('MDL_MW', fortran)

        python = helas_call_writers.PythonUFOHelasCallWriter(
            mgcmd._curr_model).get_wavefunction_call(wf)
        self.assertIn('p[1][0]**2', python)
        self.assertNotIn('MW', python)


class TestExtendedCmd(unittest.TestCase):
    """test the extension of cmd interface"""
    
    
    def test_the_exit_from_child_cmd(self):
        """ """
        main = ext_cmd.Cmd()
        child = ext_cmd.Cmd()
        main.define_child_cmd_interface(child, interface=False)
        self.assertEqual(main.child, child)
        self.assertEqual(child.mother, main)        
        
        ret = main.do_quit('')
        self.assertEqual(ret, None)
        self.assertEqual(main.child, None)
        ret = main.do_quit('')
        self.assertEqual(ret, True)
        
    def test_the_exit_from_child_cmd2(self):
        """ """
        main = ext_cmd.Cmd()
        child = ext_cmd.Cmd()
        main.define_child_cmd_interface(child, interface=False)
        self.assertEqual(main.child, child)
        self.assertEqual(child.mother, main)        
        
        ret = child.do_quit('')
        self.assertEqual(ret, True)
        self.assertEqual(main.child, None)
        #ret = main.do_quit('')
        #self.assertEqual(ret, True)        

class TestMadSpinFCT_in_interface(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    def setUp(self):
        if not hasattr(self, 'cmd'):
            TestMadSpinFCT_in_interface.cmd = cmd.MasterCmd()
            TestMadSpinFCT_in_interface.cmd.exec_cmd('import model sm')
            
            
    def test_get_final_part(self):
        """ """
        
        output = self.cmd.get_final_part(' p p > e+ e-')
        self.assertEqual(output, set([-11, 11]))

        output = self.cmd.get_final_part(' p p > e+ e- QED=2')
        self.assertEqual(output, set([-11, 11]))
        
        output = self.cmd.get_final_part(' p p > z > e+ e-')
        self.assertEqual(output, set([-11, 11]))        
          
        output = self.cmd.get_final_part(' p p > z > e+ e- / a')
        self.assertEqual(output, set([-11, 11]))

        output = self.cmd.get_final_part(' p p > z > e+ e- [QCD]')
        self.assertEqual(output, set([-11, 11]))
        
        output = self.cmd.get_final_part(' p p > z > e+ e- [ QCD ]')
        self.assertEqual(output, set([-11, 11]))
        
        output = self.cmd.get_final_part(' p p > z > e+ e- [ all = QCD ]')
        self.assertEqual(output, set([-11, 11]))
        
        output = self.cmd.get_final_part(' p p > z > l+ l- [ all = QCD ]')
        self.assertEqual(output, set([-11, 11, -13, 13]))
        
        output = self.cmd.get_final_part(' p p > z j, z > l+ l- [ all = QCD ]')
        self.assertEqual(output, set([-11, 11, -13, 13, 1, 2, 3, 4, 21, -1, -2,-3,-4]))
        
        output = self.cmd.get_final_part(' p p > t t~ [ all = QCD ] , (t > b z, z > l+ l-) ')
        self.assertEqual(output, set([-11, 11, -13, 13, -6, 5])) 
        
        output = self.cmd.get_final_part('p p > 2Z')
        self.assertEqual(output, set([23]))        
        
        output = self.cmd.get_final_part('p p > Z{L} j')
        self.assertEqual(output, set([1, 2, 3, 4, -1, 21, -4, -3, -2, 23]))         

        output = self.cmd.get_final_part('p p > Z{L} j, Z > e+ e-')
        self.assertEqual(output, set([1, 2, 3, 4, -1, 21, -4, -3, -2, 11, -11])) 
        
        output = self.cmd.get_final_part('p p > 2Z{L} ')
        self.assertEqual(output, set([23]))         

        output = self.cmd.get_final_part('p p > 2Z{L} j, Z > e+ e-')
        self.assertEqual(output, set([1, 2, 3, 4, -1, 21, -4, -3, -2, 11, -11]))         


class TestModel_interface(unittest.TestCase):
    """ check if the ValidCmd works correctly """


    def test_startfromalpha0_attribute(self):
        """ check that the startfromalpha0 attribute is correctly set in the model """

        self.cmd = cmd.MasterCmd()
        self.cmd.exec_cmd('import model sm')
        self.assertFalse(self.cmd._curr_model.get('startfromalpha0'))

        self.cmd.exec_cmd('import model loop_qcd_qed_sm_a0')
        self.assertTrue(self.cmd._curr_model.get('startfromalpha0'))

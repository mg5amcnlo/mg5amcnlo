##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
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
""" Basic test of the command interface """

from __future__ import absolute_import
import unittest
import madgraph
import madgraph.interface.master_interface as cmd
import madgraph.core.base_objects as base_objects
import MadSpin.interface_madspin as ms_cmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.various.misc as misc
import os
import logging

import tests.parallel_tests.test_aloha as test_aloha

import tempfile
pjoin = os.path.join
MG5DIR = madgraph.MG5DIR


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

        import madgraph.models as ufomodels
        ufomodel = ufomodels.load_model(cmd._curr_model.get('name'))
        for key in target:
            try:
                to_check = getattr(ufomodel.lorentz, key).structure
            except:
                continue
            else:
                self.assertEqual(to_check, target[key])

    @test_aloha.set_global()
    def test_polarisation_nlo_regimes(self):
        """A polarised NLO process is checked differently in each regime.

        The three do not share a hazard, so they must not share a check:

          - 'virt' is standalone MadLoop: the user supplies the phase-space
            point and so picks the frame themselves;
          - loop-induced ('noborn', 'sqrvirt') has no Born to subtract and no
            counterterm that has to sit in the same frame, and 'noborn' is
            boosted by the same LO boost_to_frame as a tree process, so
            nothing NLO-specific applies;
          - the subtracted modes thread the frame through Born, real, FKS
            counterterms, virtual and the MC-counterterm azimuth by hand, and
            only the QCD path is validated. Colour is restricted there by two
            separate rules:
              * FINAL state: only MASSLESS coloured particles are refused. A
                massive coloured emitter was measured (see the p p > t t~
                [QCD] closure study), a massless one has not been -- and for a
                massless one the real emission could not carry the born's
                polarization onto the FKS emitter anyway (g -> q q~ changes
                the identity of leg j).
              * INITIAL state: ANY coloured particle is refused, whatever its
                mass. The initial-state splitting runs backwards,
                g -> q(-> born) q~, so the polarized quark is an internal line
                of the real and no external leg can carry the projection.
                A 1 -> N decay is exempt: fks_base.find_reals never splits the
                initial state of a decay process.

        The two gates used to disagree: the first admitted noborn/sqrvirt and
        the second refused them again for being massive, so the first clause
        was dead and loop-induced polarisation was refused outright.
        """
        cmd = self.cmd
        cmd.do_import('sm')

        # loop-induced: no restriction at all, massive or not, any order
        cmd.check_process_format('g g > z{0} z{0} [noborn=QCD]')
        cmd.check_process_format('g g > z{0} z{0} [sqrvirt=QCD]')
        cmd.check_process_format('g g > z{0} z{0} [noborn=QED]')
        # standalone MadLoop: likewise
        cmd.check_process_format('u u~ > w+{0} w-{0} [virt=QED]')
        # subtracted, QCD: the boost is threaded through
        cmd.check_process_format('p p > z{0} j [QCD]')
        cmd.check_process_format('p p > z{0} j [real=QCD]')

        # subtracted, but outside the reach of the validated boost
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'p p > z{0} j [QED]')
        # no frame at all in this mode
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'p p > z{0} j [tree=QCD]')
        # A coloured polarised particle is also an FKS emitter. For a MASSIVE
        # one that was measured -- p p > t t~ [QCD], all four top-helicity
        # combinations, closure at +0.03 sigma, check_poles 20/20 and test_ME
        # clean on the FKS configurations whose emitter IS the polarised top
        # (docs/nlo_polarisation_massive_colour.md) -- so it is allowed.
        cmd.check_process_format('u u~ > t{L} t~ [QCD]')
        cmd.check_process_format('p p > t{+} t~{-} [QCD]')
        cmd.check_process_format('p p > t{-} t~{+} [QCD]')
        cmd.check_process_format('g g > t{+} t~{+} [real=QCD]')
        cmd.check_process_format('u u~ > t{R} t~{L} [LOonly=QCD]')
        # ... a MASSLESS coloured one is not: still untested, so still refused.
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'g{+} g > t t~ [QCD]')
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'u{+} u~ > t t~ [QCD]')

        # A coloured particle in the INITIAL state is refused whatever its
        # mass: the initial-state splitting is read backwards,
        # g -> q(-> born) q~, so the polarized quark is an internal line of the
        # real emission and there is no external leg to carry the projection.
        # The massless-only rule above does not catch this, which is why it is
        # a rule of its own: b and t are massive in the default sm, and
        # b{+} b~ > h [QCD] used to reach generation and die there with a raw
        # fks_common.FKSProcessError traceback.
        for proc in ['b{+} b~ > h [QCD]',
                     't{+} t~ > z [QCD]',
                     'b~{+} b > h [real=QCD]',
                     'g{+} g > t t~ [LOonly=QCD]']:
            self.assertRaises(cmd.InvalidCmd, cmd.check_process_format, proc)
        # control: the same colliders, with the polarization on a colourless
        # leg or on a coloured FINAL-state one -- accepted
        cmd.check_process_format('b b~ > h{0} [QCD]')
        cmd.check_process_format('b b~ > t{+} t~ [QCD]')
        # the two colour rules give distinguishable messages
        try:
            cmd.check_process_format('t{+} t~ > z [QCD]')
            self.fail('an initial-state coloured polarized leg must be refused')
        except cmd.InvalidCmd as error:
            self.assertIn('INITIAL', str(error))
        try:
            cmd.check_process_format('p p > u{+} u~ [QCD]')
            self.fail('a massless coloured polarized leg must be refused')
        except cmd.InvalidCmd as error:
            self.assertIn('massless color charged', str(error))
            self.assertNotIn('INITIAL', str(error))

        # A 1 -> N DECAY is exempt. Its "initial state" is the decaying
        # particle, and fks_base.find_reals skips initial-state splittings for
        # decay processes ("no splittings for initial states in decay
        # processes"), so that leg is a spectator and keeps its polarization.
        cmd.check_process_format('t{+} > w+ b QED=1 [QCD]')
        cmd.check_process_format('h > b{+} b~ QED=1 [QCD]')
        cmd.check_process_format('t{+} > w+ b{-} QED=1 [QCD]')

        # The multiparticle case is what proves the constituent expansion
        # works: p and j are refused through their gluon / massless quarks.
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'p{+} p > t t~ [QCD]')
        self.assertRaises(cmd.InvalidCmd,
                          cmd.check_process_format, 'p p > j{+} j [QCD]')
        # ... and the refusal is not keyed on the gluon: a multiparticle with
        # no gluon in it is refused too, through its massless quarks.
        # (This does NOT show the walk reaches every constituent. MG5 orders
        # multiparticle members gluon first, then by |pdg| ascending, so
        # 'qlight = u u~ d d~' is stored as [2, 1, -2, -1] and the refusal
        # fires on the FIRST member. In the SM one cannot build a
        # multiparticle whose first member is massive-coloured and a later one
        # massless-coloured, so "reaches every constituent" is untestable
        # here.)
        # do_define mutates the interface, and self.cmd is a class attribute
        # shared by every test in this class, so use a private instance.
        own = self.cmd.__class__()
        own.no_notification()
        own.do_import('sm')
        own.do_define('qlight = u u~ d d~')
        self.assertRaises(own.InvalidCmd,
                          own.check_process_format,
                          'p p > qlight{+} qlight [QCD]')
        # An UPPERCASE multiparticle label: do_define lowercases the key it
        # stores, so the walk has to look it up lowercased. It used to fall
        # through to get_particle('qq') -> None and crash with an
        # AttributeError instead of accepting the (massive, coloured) members.
        own.do_define('QQ = t t~')
        own.check_process_format('p p > QQ{+} QQ [QCD]')
        # An unknown particle name is not this check's business: it must not
        # crash here, the process parser reports it.
        own.check_process_format('p p > nosuchparticle{+} t~ [QCD]')
        # ... but colour is no restriction where there is no subtraction
        cmd.check_process_format('g g > t{L} t~ [noborn=QCD]')
        cmd.check_process_format('g{+} g > t t~ [noborn=QCD]')

    @test_aloha.set_global()
    def test_polarisation_no_duplicate_helicity(self):
        """A polarization restriction must name each helicity at most once.

        '{++}' used to be accepted and silently gave a factor two: the
        helicity matrix is itertools.product over the raw list -- so [1,1]
        yields ncomb=8 rows for 4 distinct helicity assignments -- while the
        denominator factor is built from the same restriction as '{+}'.  The
        overlap can also be hidden behind a multi-valued label: 'T' expands to
        (+1,-1), so '{+T}', '{RT}', '{-T}' and '{LT}' repeat a helicity too.
        Both families are refused; the message has to name the repeated
        helicity and what each label expanded to, since '{+T}' does not look
        like a duplicate until you know what 'T' covers.
        """
        cmd = self.cmd
        cmd.do_import('sm')

        # accepted: every spelling below names each helicity exactly once
        for accepted, expected in [('+', [1]), ('-', [-1]), ('0', [0]),
                                   ('T', [1, -1]), ('L', [-1]), ('R', [1]),
                                   ('A', [99]), ('S', [9]),
                                   ('+-', [1, -1]), ('-+', [-1, 1]),
                                   ('0T', [0, 1, -1]), ('T0', [1, -1, 0]),
                                   ('LR', [-1, 1]), ('0+', [0, 1]),
                                   ('0S', [0, 9]), ('GH', [4, 5]),
                                   ('GQW', [4, 6, 7]), ('0-', [0, -1]),
                                   # '+0' is the signed spelling of helicity
                                   # 0, so '{+0+}' is 0 and +1 -- not a
                                   # duplicate, however it reads
                                   ('+0', [0]), ('+0+', [0, 1]),
                                   ('+2', [2])]:
            procdef = cmd.extract_process('p p > w+{%s}' % accepted)
            self.assertEqual(procdef['legs'][-1]['polarization'], expected,
                             'polarization {%s} should be %s' % (accepted,
                                                                 expected))

        # refused: a helicity named twice, literally or after expansion
        for refused in ['++', '--', '+-+', 'TT', 'LL', 'RR', '00', 'GG', 'AA',
                        '+T', 'RT', '-T', 'LT', 'TL', 'TR', 'T+', 'T-',
                        'R+', 'L-', '0T0']:
            self.assertRaises(cmd.InvalidCmd, cmd.extract_process,
                              'p p > w+{%s}' % refused)

        # the message names the repeated helicity and both labels
        try:
            cmd.extract_process('p p > w+{+T}')
        except cmd.InvalidCmd as error:
            msg = str(error)
            self.assertIn('+1 (right)', msg)
            self.assertIn('-1 (left)', msg)
            self.assertIn('selects', msg)
            self.assertIn('already selected', msg)
        else:
            raise Exception('{+T} repeats helicity +1 and must be refused')

        # the same parse loop serves LO: the double-count is not NLO-specific
        self.assertRaises(cmd.InvalidCmd, cmd.extract_process,
                          'p p > w+{++} [real=QCD]')
        # ... and it reaches every leg of a decay chain, which parses each
        # piece through the same extract_process
        self.assertRaises(cmd.InvalidCmd, cmd.extract_decay_chain_process,
                          'e+ e- > z{T}, z > mu+{++} mu-')

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
        cmd.check_process_format('u u~ > w+{L} [QCD]')
        cmd.check_process_format('u u~ > z{0} g [QCD]')
        cmd.check_process_format('u u~ > z{0} g [real=QCD]')
        cmd.check_process_format('u u~ > z{0} g [LOonly=QCD]')
        # standalone MadLoop: the user supplies the momenta, so the frame is
        # theirs and the perturbation orders do not matter
        cmd.check_process_format('u u~ > z{0} g [virt=QCD]')
        cmd.check_process_format('u u~ > z{0} g [virt=QED QCD]')
        # a MASSIVE coloured particle may be polarised in the subtracted
        # regime: measured on p p > t t~ [QCD], see
        # docs/nlo_polarisation_massive_colour.md
        cmd.check_process_format('u u~ > t{L} t~ [QCD]')
        cmd.check_process_format('p p > t{+} t~{-} [QCD]')

        # invalid syntax
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
        # a MASSLESS coloured particle stays refused in the subtracted regime,
        # multiparticles included -- p and j carry a gluon
        self.wrong(cmd.check_process_format, 'g{+} g > t t~ [QCD]')
        self.wrong(cmd.check_process_format, 'u{+} u~ > t t~ [QCD]')
        self.wrong(cmd.check_process_format, 'p{+} p > t t~ [QCD]')
        self.wrong(cmd.check_process_format, 'p p > j{+} j [QCD]')
        # massive colourless polarization at NLO QCD is supported since the
        # me_frame boost reaches the virtual; mixed and pure QED are not
        self.wrong(cmd.check_process_format, 'u u~ > W+{L} vl [ QED QCD]')
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
        """check that simple syntax goes through and return expected process"""
           
        cmd = self.cmd
        self.do('import model sm')
        self.do('generate 2p > 2j')
        self.assertTrue(cmd._curr_amps)
        proc = cmd._curr_amps[0].get('process').get('legs')
        self.assertEqual(len(proc), 4)
        
    @test_aloha.set_global()
    def test_generate_polarised(self):
        """check that simple syntax goes through and return expected process"""
           
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
                proc = 'generate p p > z{%s} h' % tag
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

    @test_aloha.set_global()
    def test_propagator_polarisation_round_trip(self):
        """nice_string()/input_string()/base_string() used to print the raw
        integer ({4}, {99}, ...), which the parser rejects with "polarization
        are between -3 and 3" -- so a printed process line could not be read
        back. They must print the brace letter instead."""
        cmd = self.cmd
        self.do('import model sm')
        try:
            # the propagator braces print their letter, including the
            # combined '{0S}' which must not grow a comma (a ',' inside the
            # brace also breaks the decay-chain split on ',')
            for tag, expected in (('G', 'w+{G}'), ('H', 'w+{H}'),
                                  ('Q', 'w+{Q}'), ('W', 'w+{W}'),
                                  ('S', 'w+{S}'), ('A', 'w+{A}'),
                                  ('0S', 'w+{0S}')):
                self.do('generate t > w+{%s} b, w+ > ta+ vt' % tag)
                proc = cmd._curr_amps[0].get('amplitudes')[0].get('process')
                self.assertIn(expected, proc.nice_string(prefix=False))
                self.assertIn(expected, proc.input_string())
                self.assertIn(expected, proc.base_string())
                self.assertNotIn(',', proc.input_string())
                # and the printed string is accepted back by the parser.
                # 'proc' is the core amplitude, so re-attach the decay that
                # makes the brace legal in the first place.
                self.do('generate %s, w+ > ta+ vt' % proc.input_string())
                self.assertTrue(cmd._curr_amps)
        finally:
            cmd.exec_cmd('generate p p > t t~')


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

class TestHepToolsInstallTarget(unittest.TestCase):
    """'install <tool>' must record its paths in this installation's own
    configuration, unless HEPTools genuinely lives somewhere shared. Writing
    installation-specific absolute paths into the per-user file is what made
    one MadGraph download PDF sets into another one's HEPTools (issue #94)."""

    def target(self, heptools_install_dir):
        import madgraph.interface.madgraph_interface as mg_cmd
        return mg_cmd.MadGraphCmd.heptools_install_target(heptools_install_dir)

    def test_default_stays_in_the_installation(self):
        """The default './HEPTools' is inside MG5DIR, so its paths are private
        to this installation -- this is the branch that used to be dead."""
        for value in ['./HEPTools', None, '', os.path.join(MG5DIR, 'HEPTools')]:
            prefix, config_file = self.target(value)
            self.assertEqual(prefix, os.path.join(MG5DIR, 'HEPTools'))
            self.assertEqual(config_file, '')

    def test_external_prefix_uses_the_user_config(self):
        prefix, config_file = self.target(tempfile.gettempdir())
        self.assertEqual(prefix, os.path.realpath(tempfile.gettempdir()))
        self.assertEqual(config_file, misc.user_config_file())
        self.assertNotIn('.mg5', config_file)


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


class _MassOnlyModel(dict):
    """The little of a model that the mass advice reads.

    is_massive() wants the particles and the value of each mass parameter,
    and reads 'parameter_dict' off the model as a dict (Model is one), so
    this is a dict too.
    """

    def __init__(self, masses, name='fake'):
        """masses: {pdg: (mass parameter name, value)}"""

        dict.__init__(self)
        self['parameter_dict'] = dict(masses.values())
        self['name'] = name
        self['modelpath'] = '/models/%s' % name
        self._particles = dict((pdg, {'mass': name})
                               for pdg, (name, _) in masses.items())

    def get(self, key):
        if key == 'particle_dict':
            return self._particles
        return dict.get(self, key)


class NeglectedMassAdviceTest(unittest.TestCase):
    """What `import model` says about a light fermion mass the model kept.

    A massive u, d, s, e or mu costs diagrams and helicity configurations for
    a mass which is negligible anyway, and customize_model removes it. A
    massive b or tau is the 4F scheme and a deliberate choice, so the same
    message there would be noise on the most common import there is.
    """

    B_AND_TAU = {5: ('MB', 4.7), 15: ('MTA', 1.777)}

    def advice_for(self, masses):
        """What advise_neglected_masses logs for a model with those masses."""

        import madgraph.interface.madgraph_interface as mg_interface

        class _Interface(object):
            NEGLECTED_MASSES = mg_interface.MadGraphCmd.NEGLECTED_MASSES

        interface = _Interface()
        interface._advised_masses_for = None
        interface._curr_model = (_MassOnlyModel(masses)
                                if masses is not None else None)

        captured = []

        class _Handler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        # the interface logs to 'cmdprint', which is what the user sees
        logger = logging.getLogger('cmdprint')
        saved = (logger.handlers, logger.propagate, logger.level)
        logger.handlers = [_Handler()]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            mg_interface.MadGraphCmd.advise_neglected_masses(interface)
        finally:
            (logger.handlers, logger.propagate, logger.level) = saved
        return captured

    def test_a_massive_up_quark_is_reported(self):
        masses = dict(self.B_AND_TAU)
        masses[2] = ('MU', 2.55e-3)
        message = self.advice_for(masses)
        self.assertEqual(len(message), 1)
        self.assertIn('for u.', message[0])
        self.assertIn('customize_model', message[0])
        self.assertIn("'flavour scheme'", message[0])
        self.assertNotIn('massive leptons', message[0])

    def test_massive_light_leptons_are_reported(self):
        masses = dict(self.B_AND_TAU)
        masses[11] = ('MEL', 5.11e-4)
        masses[13] = ('MMU', 0.106)
        message = self.advice_for(masses)
        self.assertEqual(len(message), 1)
        self.assertIn('for e, mu.', message[0])
        self.assertIn("'nb of massive leptons'", message[0])
        self.assertNotIn('flavour scheme', message[0])

    def test_both_kinds_name_both_options(self):
        masses = dict(self.B_AND_TAU)
        masses[3] = ('MS', 0.101)
        masses[13] = ('MMU', 0.106)
        message = self.advice_for(masses)
        self.assertIn('for s, mu.', message[0])
        self.assertIn("options 'flavour scheme' and 'nb of massive leptons'",
                      message[0])

    def test_the_4f_scheme_says_nothing(self):
        """b and tau massive, everything else massless: that is `sm`."""

        self.assertEqual(self.advice_for(self.B_AND_TAU), [])

    def test_a_mass_hardcoded_to_zero_says_nothing(self):
        """A UFO whose u mass is ZERO cannot be restricted, and is already
        what the advice would ask for."""

        masses = dict(self.B_AND_TAU)
        masses[2] = ('ZERO', 0.0)
        self.assertEqual(self.advice_for(masses), [])

    def test_no_model_at_all_is_harmless(self):
        self.assertEqual(self.advice_for(None), [])

    def test_it_is_said_once_per_model(self):
        """MG7 re-imports the model behind the user's back -- `set gauge`,
        `set complex_mass_scheme`, and `check gauge` four times over -- and
        each one comes through do_import."""

        import madgraph.interface.madgraph_interface as mg_interface

        class _Interface(object):
            NEGLECTED_MASSES = mg_interface.MadGraphCmd.NEGLECTED_MASSES
            _advised_masses_for = None

        interface = _Interface()
        masses = dict(self.B_AND_TAU)
        masses[2] = ('MU', 2.55e-3)
        interface._curr_model = _MassOnlyModel(masses)

        said = []

        class _Handler(logging.Handler):
            def emit(self, record):
                said.append(record.msg)

        logger = logging.getLogger('cmdprint')
        saved = (logger.handlers, logger.propagate, logger.level)
        logger.handlers = [_Handler()]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            for _ in range(5):
                mg_interface.MadGraphCmd.advise_neglected_masses(interface)
            # ... and a different model is a new thing to say
            interface._curr_model = _MassOnlyModel(masses, name='other')
            mg_interface.MadGraphCmd.advise_neglected_masses(interface)
        finally:
            (logger.handlers, logger.propagate, logger.level) = saved
        self.assertEqual(len(said), 2, said)


class CheckDisplayWithoutProcessTest(unittest.TestCase):
    """'display processes' before anything is generated has to say so, not die.

    _fks_multi_proc only exists once an NLO process has been generated, and
    check_display read it unconditionally.
    """

    def setUp(self):
        import madgraph.interface.master_interface as cmd
        self.cmd = cmd.MasterCmd()
        self.cmd.do_import('model sm')

    def test_it_raises_invalid_cmd(self):
        self.assertFalse(hasattr(self.cmd, '_fks_multi_proc'))
        for what in ['processes', 'diagrams', 'diagrams_text']:
            try:
                self.cmd.check_display([what])
            except AttributeError as error:
                self.fail('check_display(%r) raised %s' % (what, error))
            except Exception as error:
                self.assertTrue('No process generated' in str(error), what)
            else:
                self.fail('check_display(%r) accepted an empty session' % what)

    def test_it_accepts_a_generated_process(self):
        self.cmd.do_generate('e+ e- > mu+ mu-')
        self.cmd.check_display(['processes'])   # must not raise


class RequiredSChannelErrorTest(unittest.TestCase):
    """A bad required s-channel has to be reported for what it is.

    extract_process asks extract_particle_ids for the required s-channels with
    crash_on_duplication, and used to turn *any* InvalidCmd coming back into a
    message about the '> A A >' syntax -- so an unknown particle name in that
    position was reported as a syntax that the user had not used.
    """

    @classmethod
    def setUpClass(cls):
        cls.cmd = cmd.MasterCmd()
        cls.cmd.do_import('model sm')

    def error_of(self, process):
        try:
            self.cmd.extract_process(process)
        except madgraph.InvalidCmd as error:
            return str(error)
        self.fail('%r was accepted' % process)

    def test_an_unknown_particle_says_so(self):
        message = self.error_of('b b~ > w+ w- | h+ h- > ta+ vt ta- vt~')
        self.assertIn('h+', message)
        self.assertNotIn('A A', message)

    def test_a_repeated_one_still_reports_the_syntax(self):
        message = self.error_of('p p > z z > e+ e- mu+ mu-')
        self.assertIn('A A', message)

    def test_a_good_one_is_accepted(self):
        self.cmd.extract_process('p p > z | a > e+ e-')   # must not raise


class DisplayInertCouplingOrderTest(unittest.TestCase):
    """A coupling order the model declares but which no interaction carries
    is inert: it can not be used to constrain a process. That happens as soon
    as a restriction card removes every vertex of that order -- SMEFTatNLO
    keeps NP in its hierarchy while its default restriction drops all the NP
    vertices, so 'generate p p > t t~ NP<=2' is rejected while 'display
    coupling_order' used to advertise NP like any other order.
    """

    @classmethod
    def setUpClass(cls):
        cls.cmd = cmd.MasterCmd()
        cls.cmd.do_import('model sm')

    def setUp(self):
        self.hierarchy = self.cmd._curr_model['order_hierarchy']

    def tearDown(self):
        self.cmd._curr_model['order_hierarchy'] = self.hierarchy

    def display(self):
        import contextlib, io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.cmd.do_display('coupling_order')
        return output.getvalue()

    def test_an_order_carried_by_no_interaction_is_flagged(self):
        self.cmd._curr_model['order_hierarchy'] = \
                                      {'NP': 1, 'QCD': 2, 'QED': 4}
        text = self.display()
        self.assertIn('NP : weight = 1 [inert]', text)
        self.assertIn('QCD : weight = 2\n', text)
        self.assertIn('QED : weight = 4\n', text)
        # and the flag is explained
        self.assertIn('no interaction of this model carries that order', text)

    def test_a_model_with_no_inert_order_is_left_alone(self):
        text = self.display()
        self.assertNotIn('inert', text)
        self.assertIn('QCD : weight = 1\n', text)
        self.assertIn('QED : weight = 2\n', text)

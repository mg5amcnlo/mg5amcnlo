##############################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
""" Basic test of the command interface """

from __future__ import absolute_import
import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.madevent_interface as mecmd
import madgraph.interface.common_run_interface as runcmd
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import os
try:
    import readline
except:
    readline = None

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class FakeInterface(object):
    
    do_define_transfer_fct = ''
    complete_define_transfer_fct = ''
    help_define_transfer_fct = ''
    
    def __init__(self, me_dir):
        self.me_dir = me_dir
        self.inputfile = None
    

class TestEditCardCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    def setUp(self):
        """ """
    
        if os.path.exists('/tmp/edit_card'):
            os.system('rm -rf /tmp/edit_card')
        os.system('mkdir /tmp/edit_card;mkdir /tmp/edit_card/Cards;')
        template_path = pjoin(root_path, '..', 'Template')
        card= 'param_card'
        files.cp(pjoin(root_path, 'input_files/restrict_sm.dat'), '/tmp/edit_card/Cards/%s.dat' % card)
        files.cp(pjoin(root_path, 'input_files/restrict_sm.dat'), '/tmp/edit_card/Cards/%s_default.dat' % card)
                
        card = 'run_card'
        import madgraph.various.banner as banner_mod
        card = banner_mod.RunCardLO()
        card.write('/tmp/edit_card/Cards/run_card.dat')
        card.write('/tmp/edit_card/Cards/run_card_default.dat')


        card = 'MadWeight_card'
        files.cp(pjoin(template_path, 'MadWeight/Cards/%s.dat' % card), '/tmp/edit_card/Cards')
        files.cp(pjoin(template_path, 'MadWeight/Cards/%s.dat' % card), '/tmp/edit_card/Cards/%s_default.dat' % card)
        card = 'shower_card'
        import madgraph.various.shower_card as shower_mod
        card = shower_mod.ShowerCard()
        card.write("/tmp/edit_card/Cards/shower_card.dat", template=pjoin(template_path, 'NLO/Cards/shower_card.dat'))
        card.write("/tmp/edit_card/Cards/shower_card_default.dat", template=pjoin(template_path, 'NLO/Cards/shower_card.dat'))
        
        #MadLoop Card
        files.cp(pjoin(template_path, 'loop_material/StandAlone/Cards/MadLoopParams.dat'), '/tmp/edit_card/Cards')
        
        #Pythia8 
        files.cp(pjoin(template_path, 'LO/Cards/pythia8_card_default.dat'), '/tmp/edit_card/Cards/pythia8_card_default.dat')
        files.cp(pjoin(template_path, 'LO/Cards/pythia8_card_default.dat'), '/tmp/edit_card/Cards/pythia8_card.dat')
        
        fakemother = FakeInterface('/tmp/edit_card/')
        self.cmd = runcmd.AskforEditCard('', cards=['run_card.dat', 'param_card.dat', 'madweight_card.dat', 'shower_card.dat',
                                                    'pythia8_card.dat'],
                                        mode='auto', mother_interface=fakemother)
        
        #Reweight
        files.cp(pjoin(template_path, 'Common/Cards/density_card_default.dat'), '/tmp/edit_card/Cards/density_card_default.dat') #create density_card_default.dat
        files.cp(pjoin(template_path, 'Common/Cards/density_card_default.dat'), '/tmp/edit_card/Cards/reweight_card.dat') #create reweight_card.dat
        
        fakemother = FakeInterface('/tmp/edit_card/')
        self.rwcmd = runcmd.AskforEditCard('', cards=['run_card.dat', 'param_card.dat', 'reweight_card.dat'], mother_interface=fakemother)

    def get_completion(self, text):
        if readline:
            readline.__doc__ = 'libedit'
        data = text.split()
        fct = getattr(self.cmd, 'complete_%s' % (data[0]))
        line = ' '.join(data[1:])
        
        return fct("", text, len(text),len(text))
        
    #just changed the class from which one gets the attributes
    def get_completion_reweight(self, text):
        if readline:
            readline.__doc__ = 'libedit'
        data = text.split()
        fct = getattr(self.rwcmd, 'complete_%s' % (data[0]))
        line = ' '.join(data[1:])
        
        return fct("", text, len(text),len(text))
        
        
    def test_autocompletion(self):
        """"""
        
        # First Level (set)
        first_level = self.get_completion('set')
        self.assertIn('MadWeight_card', first_level)
        self.assertIn('param_card', first_level)
        self.assertIn('run_card', first_level)
        self.assertIn('bjet_is_jet', first_level)
        self.assertNotIn('13', first_level)
        self.assertIn('etaj', first_level)
        self.assertIn('mass', first_level)
        self.assertIn('width', first_level)
        self.assertIn('decay', first_level)
        self.assertIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertIn('wt', first_level)
        self.assertIn('iregimode', first_level)
        self.assertIn('MadLoop_card', first_level)
        
        # MadWeight completion -------------------------------------------------
        # set MadWeight_card
        first_level = self.get_completion('set MadWeight_card')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertIn('bjet_is_jet', first_level)
        self.assertNotIn('13', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('iregimode', first_level)
         
        # set MadWeight_card mw_perm
        first_level = self.get_completion('set MadWeight_card mw_perm')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertIn('bjet_is_jet', first_level)
        self.assertNotIn('13', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('iregimode', first_level)
        
        # set MadWeight_card mw_perm bjet_is_jet
        first_level = self.get_completion('set MadWeight_card mw_per bjet_is_jet')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('13', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('iregimode', first_level)

        # set MadWeight_card mw_parameter
        first_level = self.get_completion('set MadWeight_card mw_parameter')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertIn('13', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('iregimode', first_level)
        
        
        # MadEvent completion -------------------------------------------------
        first_level = self.get_completion('set param_card ')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertIn('mass', first_level)
        self.assertIn('width', first_level)
        self.assertIn('decay', first_level)
        self.assertIn('wt', first_level)
        self.assertIn('mh', first_level)
        self.assertIn('as', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('iregimode', first_level)

        first_level = self.get_completion('set param_card wolfenstein')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('iregimode', first_level)

        first_level = self.get_completion('set param_card wolfenstein etaws')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('Auto', first_level)
        self.assertNotIn('iregimode', first_level)

        first_level = self.get_completion('set param_card decay')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertIn('6', first_level)
        self.assertNotIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('Auto', first_level)
        self.assertNotIn('iregimode', first_level)
  
        first_level = self.get_completion('set param_card width')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertIn('6', first_level)
        self.assertNotIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('Auto', first_level)
        self.assertNotIn('iregimode', first_level)
        
        first_level = self.get_completion('set param_card width wt')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertIn('Auto', first_level)
        self.assertIn('Auto@NLO', first_level)
        self.assertNotIn('iregimode', first_level)
        
        first_level = self.get_completion('set param_card width 6')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('1', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('etaws', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertIn('Auto', first_level)
        self.assertIn('Auto@NLO', first_level)
        self.assertNotIn('iregimode', first_level)
        
        # Run_card completion -------------------------------------------------
        first_level = self.get_completion('set run_card ')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('as', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('Auto@NLO', first_level)
        self.assertNotIn('iregimode', first_level)
          
        first_level = self.get_completion('set run_card htjmax')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('as', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('iregimode', first_level)
        
        # ML_card completion ---------------------------------------------------               
        first_level = self.get_completion('set MadLoop_card')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('as', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertIn('iregimode', first_level)
        
        first_level = self.get_completion('set MadLoop_card iregimode')
        self.assertNotIn('MadWeight_card', first_level)
        self.assertNotIn('param_card', first_level)
        self.assertNotIn('run_card', first_level)
        self.assertNotIn('MadLoop_card', first_level)
        self.assertNotIn('bjet_is_jet', first_level)
        self.assertNotIn('6', first_level)
        self.assertNotIn('etaj', first_level)
        self.assertNotIn('mass', first_level)
        self.assertNotIn('width', first_level)
        self.assertNotIn('decay', first_level)
        self.assertNotIn('wt', first_level)
        self.assertNotIn('mh', first_level)
        self.assertNotIn('as', first_level)
        self.assertNotIn('mw_parameter', first_level)
        self.assertIn('default', first_level)
        self.assertNotIn('iregimode', first_level)
        
        # First Level for density module (set) ----------
        first_level = self.get_completion_reweight('set')
        self.assertIn('reweight_card', first_level)
        self.assertIn('helicity_direction', first_level)
        self.assertIn('particle_in_density_matrix', first_level)
        self.assertIn('boost_choice', first_level)
        self.assertIn('order_helicities', first_level)
        self.assertIn('axis_referential', first_level)
        self.assertIn('symmetrise_initial_state', first_level)
        self.assertIn('matrix_normalisation', first_level)
        self.assertNotIn('default', first_level)
        self.assertNotIn('[6]', first_level)

        # Second Level for density module (set reweight_card) ---------------
        second_level = self.get_completion_reweight('set reweight_card')
        self.assertIn('default', second_level)
        self.assertIn('helicity_direction', second_level)
        self.assertIn('particle_in_density_matrix', second_level)
        self.assertIn('boost_choice', second_level)
        self.assertIn('order_helicities', second_level)
        self.assertIn('axis_referential', second_level)
        self.assertIn('symmetrise_initial_state', second_level)
        self.assertIn('matrix_normalisation', second_level)
        self.assertNotIn('[6]', second_level)
        
    def test_modif_param_card(self):
        """ """
        
        param = self.cmd.param_card
        wt = param['decay'].get((6,)).value
        self.cmd.do_set('width wt 100')
        self.assertEqual(param['decay'].get((6,)).value, 100)
        self.cmd.do_set('wt default')        
        self.assertEqual(param['decay'].get((6,)).value, wt)
        self.cmd.do_set('param_card width 6 Auto')        
        self.assertEqual(param['decay'].get((6,)).value, 'Auto') 
        self.cmd.do_set('param_card width 6 Auto@NLO')        
        self.assertEqual(param['decay'].get((6,)).value, 'Auto@NLO')          
        self.cmd.do_set('param_card width 6 auto@nLo')        
        self.assertEqual(param['decay'].get((6,)).value, 'Auto@NLO')  
        # the options of compute_widths, after the auto
        self.cmd.do_set('param_card width 6 auto --nlo --body_decay=3')
        self.assertEqual(param['decay'].get((6,)).value, 'Auto --nlo --body_decay=3')
        self.cmd.do_set('width 6 AUTO --body_decay=2')
        self.assertEqual(param['decay'].get((6,)).value, 'Auto --body_decay=2')
        self.cmd.do_set('width 6 auto --nlo')
        self.assertEqual(param['decay'].get((6,)).value, 'Auto@NLO')
        # an option a width does not have: nothing changes
        self.cmd.do_set('width 6 auto --foo=1')
        self.assertEqual(param['decay'].get((6,)).value, 'Auto@NLO')
        self.cmd.do_set('wt default')
        self.assertEqual(param['decay'].get((6,)).value, wt)

    def test_invalid_set_warns_interactively(self):
        """an invalid set command leaves the cards alone without raising"""
        param = self.cmd.param_card
        mass = param['mass'].get((6,)).value
        for line in ('foo_bar 3', 'mt abc', 'param_card mass 999 10',
                     'run_card ptj', 'mass all abc', ''):
            self.cmd.do_set(line)
        self.assertEqual(param['mass'].get((6,)).value, mass)

    def test_invalid_set_is_an_error_in_scripted_runs(self):
        """a launch script must stop on a set command it cannot apply"""
        self.cmd.mother_interface.inputfile = iter([])
        param = self.cmd.param_card
        mass = param['mass'].get((6,)).value
        for line in ('foo_bar 3', 'mt abc', 'param_card mass 999 10',
                     'run_card ptj', 'mass all abc', 'width 6 auto --foo=1',
                     'madweight_card', 'events'):
            self.assertRaises(madgraph.InvalidCmd, self.cmd.do_set, line)
        self.assertEqual(param['mass'].get((6,)).value, mass)
        # valid commands are unaffected
        self.cmd.do_set('ptj 50')
        self.assertEqual(self.cmd.run_card['ptj'], 50)
        self.cmd.do_set('mt 180')
        self.assertEqual(param['mass'].get((6,)).value, 180)

    def test_set_with_energy_unit(self):
        """a value with an energy unit is converted to GeV for an energy (a
        mass, a width, an energy entry of the run_card) and refused for
        anything else: a warning interactively, an error in a scripted run"""
        param = self.cmd.param_card
        self.cmd.do_set('mt 1 TeV')
        self.assertEqual(param['mass'].get((6,)).value, 1000.)
        self.cmd.do_set('mass 6 172.5 GeV')
        self.assertEqual(param['mass'].get((6,)).value, 172.5)
        self.cmd.do_set('wt 1500 MeV')
        self.assertEqual(param['decay'].get((6,)).value, 1.5)
        self.cmd.do_set('ptj 0.1 TeV')
        self.assertEqual(self.cmd.run_card['ptj'], 100.)
        self.cmd.do_set('ebeam1 6.8TeV')
        self.assertEqual(self.cmd.run_card['ebeam1'], 6800.)

        ymt = param['yukawa'].get((6,)).value
        etaj = self.cmd.run_card['etaj']
        nevents = self.cmd.run_card['nevents']
        lines = ('ymt 1 TeV', 'yukawa 6 1 TeV', 'etaj 2 TeV', 'nevents 1 TeV')
        for line in lines:
            self.cmd.do_set(line)
        self.cmd.mother_interface.inputfile = iter([])
        for line in lines:
            self.assertRaises(madgraph.InvalidCmd, self.cmd.do_set, line)
        self.assertEqual(param['yukawa'].get((6,)).value, ymt)
        self.assertEqual(self.cmd.run_card['etaj'], etaj)
        self.assertEqual(self.cmd.run_card['nevents'], nevents)

    def test_energy_shortcut_with_unit(self):
        """an energy shortcut takes its value with a unit (a unit-less lhc/lcc
        value is in TeV, the others in GeV); a real extra argument is refused"""
        run_card = self.cmd.run_card
        for line, ebeam in (('lhc 500 GeV', 250.), ('lhc 13.6TeV', 6800.),
                            ('lhc 13', 6500.), ('ebeam 6.8 TeV', 6800.),
                            ('lep 200 GeV', 100.)):
            self.cmd.do_set(line)
            self.assertEqual((run_card['ebeam1'], run_card['ebeam2']), (ebeam, ebeam), line)
        self.cmd.do_set('fixed_scale 0.1 TeV')
        self.assertEqual(run_card['scale'], 100.)
        self.cmd.mother_interface.inputfile = iter([])
        for line in ('lhc 13 14', 'cm_velocity 0.5 GeV'):
            self.assertRaises(madgraph.InvalidCmd, self.cmd.do_set, line)
        self.assertEqual(run_card['ebeam1'], 100.)

    def test_rejected_param_value_changes_nothing(self):
        """a value refused for a whole block, a block scale or a name mapped
        to several entries is refused once, and leaves the param_card (and
        its "modified" flag) alone"""
        param = self.cmd.param_card
        masses = dict((k, p.value) for k, p in param['mass'].param_dict.items())
        self.cmd.modified_card.discard('param')
        with self.assertLogs('madgraph.stdout', level='WARNING') as logs:
            self.cmd.do_set('mass all abc')
            self.cmd.do_set('mass scale abc')
        self.assertEqual(len(logs.output), 2, logs.output)
        self.assertNotIn('param', self.cmd.modified_card)
        self.assertEqual(dict((k, p.value) for k, p in param['mass'].param_dict.items()),
                         masses)

        # a name standing for a mass and a yukawa: a unit fits only the mass,
        # so neither entry changes and nothing claims that they did
        self.cmd.pname2block['mt'] = [('mass', (6,)), ('yukawa', (6,))]
        ymt = param['yukawa'].get((6,)).value
        with self.assertLogs('madgraph.stdout', level='WARNING') as logs:
            self.cmd.do_set('mt 1 TeV')
        self.assertEqual(len(logs.output), 1, logs.output)
        self.assertIn('can not carry a unit', logs.output[0])
        self.assertEqual(param['mass'].get((6,)).value, masses[(6,)])
        self.assertEqual(param['yukawa'].get((6,)).value, ymt)
        self.assertNotIn('param', self.cmd.modified_card)
        self.cmd.do_set('mt 180')
        self.assertEqual((param['mass'].get((6,)).value, param['yukawa'].get((6,)).value),
                         (180., 180.))

    def test_modif_run_card(self):
        """ """

        run = self.cmd.run_card
        ptj = run['ptj']
        self.cmd.do_set('ptj 100')
        self.assertEqual(run['ptj'], 100)
        self.cmd.do_set('run_card ptj default')        
        self.assertEqual(run['ptj'], ptj)
        
        run.list_parameter['ptj'] = float
        self.cmd.do_set('ptj 100, 300')
        self.assertEqual(run['ptj'], [100, 300])
        self.cmd.do_set('ptj 100  200.1')
        self.assertEqual(run['ptj'], [100, 200.1])
        self.cmd.do_set('ptj 100  200.1, 3e3')
        self.assertEqual(run['ptj'], [100, 200.1, 3000])

    def test_dynamical_scale_choice_ht(self):
        """set dynamical_scale_choice HT/n on the LO, NLO and mg7 cards"""
        import shutil
        import tempfile
        import madgraph.various.banner as banner_mod
        tpl = pjoin(root_path, '..', 'madgraph', 'iolibs', 'template_files',
                    'mg7', 'run_card.toml')

        def editor(card, name, template=None):
            d = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, d)
            os.mkdir(pjoin(d, 'Cards'))
            paths = {}
            for key, f in (('run', name), ('run_default', name.replace('.', '_default.'))):
                paths[key] = pjoin(d, 'Cards', f)
                if template:
                    card.write(paths[key], template=template)
                else:
                    card.write(paths[key])
            return runcmd.AskforEditCard('', cards=paths, mode='auto',
                                         mother_interface=FakeInterface(d))

        # (value, H_T rather than H_T/2, factor)
        expected = [('HT/4', False, 0.5), ('ht/8', False, 0.25),
                    ('HT/3', False, 2/3.), ('HT/2.5', False, 0.8),
                    ('HT/2', False, 1.0), ('HT', True, 1.0)]

        cmd = editor(banner_mod.RunCardMG7(), 'run_card.toml', tpl)
        for value, full, factor in expected:
            cmd.do_set('dynamical_scale_choice %s' % value)
            beam = cmd.run_card['beam']
            self.assertEqual(beam['dynamical_scale_choice'],
                             'transverse_mass' if full else 'half_transverse_mass')
            self.assertAlmostEqual(beam['scale_factor'], factor)
        # fixed scales are left alone (a warning says they win)
        cmd.do_set('beam.fixed_ren_scale True')
        cmd.do_set('beam.dynamical_scale_choice HT/4')
        self.assertTrue(cmd.run_card['beam']['fixed_ren_scale'])
        self.assertEqual(cmd.run_card['beam']['scale_factor'], 0.5)

        cmd = editor(banner_mod.RunCardLO(), 'run_card.dat')
        for value, full, factor in expected:
            cmd.do_set('dynamical_scale_choice %s' % value)
            self.assertEqual(cmd.run_card['dynamical_scale_choice'], 2 if full else 3)
            self.assertAlmostEqual(cmd.run_card['scalefact'], factor)
        # HT/2 means H_T/2: an earlier HT/4 factor does not survive it
        cmd.do_set('dynamical_scale_choice HT/4')
        cmd.do_set('run_card dynamical_scale_choice HT/2')
        self.assertEqual(cmd.run_card['scalefact'], 1.0)
        # the other value shortcuts are unchanged
        cmd.do_set('dynamical_scale_choice et')
        self.assertEqual(cmd.run_card['dynamical_scale_choice'], 1)
        # a bare "set HT/4" is not a command
        cmd.do_set('HT/4')
        self.assertEqual(cmd.run_card['dynamical_scale_choice'], 1)
        self.assertEqual(cmd.run_card['scalefact'], 1.0)

        cmd = editor(banner_mod.RunCardNLO(), 'run_card.dat')
        for value, full, factor in expected:
            cmd.do_set('dynamical_scale_choice %s' % value)
            self.assertEqual(cmd.run_card['dynamical_scale_choice'], [2 if full else 3])
            self.assertAlmostEqual(cmd.run_card['mur_over_ref'], factor)
            self.assertAlmostEqual(cmd.run_card['muf_over_ref'], factor)

    def test_modif_ML_card(self):

        ML = self.cmd.MLcard 

        iregimode = self.cmd.MLcard['IREGIMODE']
        
        #check that nothing change if type is not correct
        self.cmd.do_set('IREGIMODE True') #should do nothrin
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertNotIn('iregimode', ML.user_set)
        
        #check that we change it correctly when an input is given
        self.cmd.do_set('IREGIMODE %s' % (iregimode+1))
        self.assertEqual(iregimode+1, self.cmd.MLcard['IREGIMODE'])
        self.assertIn('iregimode', ML.user_set)

        #check that we change it correctly when going back to default
        self.cmd.do_set('IREGIMODE default')
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertNotIn('iregimode', ML.user_set)
        
        # check that the full change is ok
        self.cmd.do_set('madloop_card IREGIMODE %s' % (iregimode+1))
        self.assertEqual(iregimode+1, self.cmd.MLcard['IREGIMODE'])
        self.assertIn('iregimode', ML.user_set)
        
        IREGIRECY = self.cmd.MLcard['IREGIRECY'] 
        self.cmd.do_set('madloop_card IREGIRECY F')
        self.assertEqual(False, self.cmd.MLcard['iregirecy'])
        self.assertIs(self.cmd.MLcard['iregirecy'], self.cmd.MLcard['IREGIRECY'])
        
        self.cmd.do_set('madloop_card default')             
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertNotIn('iregimode', self.cmd.MLcard.user_set)
        self.assertEqual(IREGIRECY, self.cmd.MLcard['IREGIRECY'])
        self.assertNotIn('iregimode', self.cmd.MLcard.user_set)
        self.assertNotIn('iregirecy', self.cmd.MLcard.user_set)
        
    def test_modif_madweight_card(self):
        """ """        
        
        mw = self.cmd.mw_card
        nb_event_by_node = mw['mw_run']['nb_event_by_node']
        self.cmd.do_set('nb_event_by_node 53')
        self.assertEqual(mw['mw_run']['nb_event_by_node'], '53')
        self.cmd.do_set('MadWeight_card nb_event_by_node default')
        self.assertEqual(mw['mw_run']['nb_event_by_node'], nb_event_by_node)
        
        # check that we can add block
        self.cmd.do_set('MadWeight_card mw_select E True')
        self.assertEqual(mw['mw_select']['e'].lower(), 'true')
        # check that we can add more than one value + add item
        self.cmd.do_set('mw_select F 1 2 3')
        self.assertEqual(mw['mw_select']['f'], ['1', '2','3'])

        
    def test_modif_shower_card(self):
        """ """
        
        shower = self.cmd.shower_card
        nevents = shower['nevents']
        self.cmd.do_set('shower_card nevents 199')
        self.assertEqual(shower['nevents'], 199)
        self.cmd.do_set('shower_card nevents default')        
        self.assertEqual(shower['nevents'], -1)
        self.cmd.do_set('mup_stable true')
        self.assertEqual(shower['mup_stable'], True)
        self.cmd.do_set('mup_stable F')
        self.assertEqual(shower['mup_stable'], False)
        self.cmd.do_set('mup_stable .true.')
        self.assertEqual(shower['mup_stable'], True)
        self.cmd.do_set('analyse a.o b.o C.o d.o')
        self.assertEqual(shower['analyse'], 'a.o b.o C.o d.o')
        self.cmd.do_set('shower_card analyse a.o b.o c.o f.o')
        self.assertEqual(shower['analyse'], 'a.o b.o c.o f.o')
        self.cmd.do_set('extrapaths extra/1 ex/t/ra2 EXtra3')
        self.assertEqual(shower['extrapaths'], 'extra/1 ex/t/ra2 EXtra3')
        self.cmd.do_set('includepaths extra/1 ex/t/ra2 EXtra3')
        self.assertEqual(shower['includepaths'], 'extra/1 ex/t/ra2 EXtra3')
        self.cmd.do_set('extralibs lib1 liB2 lIB3')
        self.assertEqual(shower['extralibs'], 'lib1 liB2 lIB3')
        self.cmd.do_set('pythia8_options {"tune_ee": 11, "tune_pp":4}')
        self.assertEqual(shower['pythia8_options'], {'tune_ee': '11', 'tune_pp': '4'})

        # finally reload the default shower card
        # and check that, when any variable is set, the others correspond to the new shower_card
        # (Bug 1708113)
        self.cmd.copy_file('/tmp/edit_card/Cards/shower_card_default.dat')
        # check that we have the default
        shower = self.cmd.shower_card
        self.assertEqual(shower['nevents'], -1)
        self.cmd.do_set('shower_card nevents 150')
        self.assertEqual(shower['analyse'], '')

    def test_modif_pythia8_card(self):
        """ """        
        
        py8 = self.cmd.PY8Card
        qcut = py8['JetMatching:qCut']
        self.cmd.do_set('jetmatching:qcut 53')
        self.assertEqual(py8['JetMatching:qCut'], 53.)
        
        self.cmd.copy_file('/tmp/edit_card/Cards/pythia8_card_default.dat')
        self.assertNotEqual(py8['JetMatching:qCut'], 53.)
        self.assertEqual(py8['JetMatching:qCut'], qcut)
        
        self.cmd.do_set('JetMatching:nJetMax 2')
        self.assertEqual(py8['JetMatching:qCut'], qcut)
        self.assertEqual(py8['JetMatching:nJetMax'], 2)
        
    def test_modif_reweight_card(self):
        """we try every set command for the density module"""
        
        #testing each parameter of reweight_card
        self.rwcmd.do_set('helicity_direction [6]')
        self.assertEqual(self.rwcmd.reweight_card['helicity_direction'], '[6]')

        self.rwcmd.do_set('particle_in_density_matrix [6, -6] pt [0, 0]')
        self.assertEqual(self.rwcmd.reweight_card['particle_in_density_matrix'], '[6, -6] pt [0, 0]')

        self.rwcmd.do_set('boost_choice [24, -24]')
        self.assertEqual(self.rwcmd.reweight_card['boost_choice'], '[24, -24]')

        self.rwcmd.do_set('order_helicities [-1, -1, -1, 1, 1, -1, 1, 1]')
        self.assertEqual(self.rwcmd.reweight_card['order_helicities'], [-1, -1, -1, 1, 1, -1, 1, 1])

        self.rwcmd.do_set('axis_referential [2, 3]')
        self.assertEqual(self.rwcmd.reweight_card['axis_referential'], [2, 3])

        self.rwcmd.do_set('symmetrise_initial_state True')
        self.assertEqual(self.rwcmd.reweight_card['symmetrise_initial_state'], True)

        self.rwcmd.do_set('matrix_normalisation False')
        self.assertEqual(self.rwcmd.reweight_card['matrix_normalisation'], False)

        #testing the reset the reweight_card
        self.rwcmd.do_set('reweight_card default')
        self.assertEqual(self.rwcmd.reweight_card['helicity_direction'], '[0]')
        self.assertEqual(self.rwcmd.reweight_card['particle_in_density_matrix'], '[6, -6]')
        self.assertEqual(self.rwcmd.reweight_card['boost_choice'], '[0]')
        self.assertEqual(self.rwcmd.reweight_card['order_helicities'], [0])
        self.assertEqual(self.rwcmd.reweight_card['axis_referential'], [0])
        self.assertEqual(self.rwcmd.reweight_card['symmetrise_initial_state'], False)
        self.assertEqual(self.rwcmd.reweight_card['matrix_normalisation'], True)
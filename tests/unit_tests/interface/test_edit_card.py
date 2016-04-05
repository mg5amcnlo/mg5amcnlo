##############################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
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

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.madevent_interface as mecmd
import madgraph.interface.common_run_interface as runcmd
import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import os
import readline


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
        files.cp(pjoin(template_path, 'NLO/Cards/%s.dat' % card), '/tmp/edit_card/Cards')
        files.cp(pjoin(template_path, 'NLO/Cards/%s.dat' % card), '/tmp/edit_card/Cards/%s_default.dat' % card)
        
        #MadLoop Card
        files.cp(pjoin(template_path, 'loop_material/StandAlone/Cards/MadLoopParams.dat'), '/tmp/edit_card/Cards')
        
        fakemother = FakeInterface('/tmp/edit_card/')
        self.cmd = runcmd.AskforEditCard('', cards=['run_card.dat', 'param_card.dat', 'madweight_card.dat', 'shower_card.dat'],
                                        mode='auto', mother_interface=fakemother)

    def get_completion(self, text):
        readline.__doc__ = 'libedit'
        data = text.split()
        fct = getattr(self.cmd, 'complete_%s' % (data[0]))
        line = ' '.join(data[1:])
        
        return fct("", text, len(text),len(text))
        
        
        
        
    def test_autocompletion(self):
        """"""
        
        # First Level (set)
        first_level = self.get_completion('set')
        self.assertTrue('MadWeight_card' in first_level)
        self.assertTrue('param_card' in first_level)
        self.assertTrue('run_card' in first_level)         
        self.assertTrue('bjet_is_jet' in first_level)          
        self.assertTrue('13' not in first_level)          
        self.assertTrue('etaj' in first_level)
        self.assertTrue('mass' in first_level)
        self.assertTrue('width' in first_level)
        self.assertTrue('decay' in first_level)
        self.assertTrue('mw_parameter'  in first_level) 
        self.assertTrue('default'  not in first_level)        
        self.assertTrue('wt' in first_level)
        self.assertTrue('iregimode' in first_level)
        self.assertTrue('MadLoop_card' in first_level)
        
        # MadWeight completion -------------------------------------------------
        # set MadWeight_card
        first_level = self.get_completion('set MadWeight_card')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level) 
        self.assertTrue('MadLoop_card' not in first_level)        
        self.assertTrue('bjet_is_jet' in first_level)          
        self.assertTrue('13' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('mw_parameter'  in first_level)        
        self.assertTrue('default'  in first_level) 
        self.assertTrue('iregimode' not in first_level)
         
        # set MadWeight_card mw_perm
        first_level = self.get_completion('set MadWeight_card mw_perm')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' in first_level)          
        self.assertTrue('13' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level) 
        self.assertTrue('default'  not in first_level)   
        self.assertTrue('iregimode' not in first_level)
        
        # set MadWeight_card mw_perm bjet_is_jet
        first_level = self.get_completion('set MadWeight_card mw_per bjet_is_jet')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level) 
        self.assertTrue('MadLoop_card' not in first_level)        
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('13' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level) 
        self.assertTrue('default'  in first_level) 
        self.assertTrue('iregimode' not in first_level)

        # set MadWeight_card mw_parameter
        first_level = self.get_completion('set MadWeight_card mw_parameter')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('13'  in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level) 
        self.assertTrue('default'  not in first_level)
        self.assertTrue('iregimode' not in first_level) 
        
        
        # MadEvent completion -------------------------------------------------
        first_level = self.get_completion('set param_card ')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' in first_level)
        self.assertTrue('width' in first_level)
        self.assertTrue('decay' in first_level)
        self.assertTrue('wt' in first_level)
        self.assertTrue('mh' in first_level)
        self.assertTrue('as' in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'  in first_level)
        self.assertTrue('iregimode' not in first_level) 

        first_level = self.get_completion('set param_card wolfenstein')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('1' in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default' not in first_level)
        self.assertTrue('iregimode' not in first_level)

        first_level = self.get_completion('set param_card wolfenstein etaws')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('1' not in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'  in first_level)
        self.assertTrue('Auto' not in first_level)
        self.assertTrue('iregimode' not in first_level)        

        first_level = self.get_completion('set param_card decay')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6'  in first_level)          
        self.assertTrue('1' not in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt'  in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default' not in first_level)
        self.assertTrue('Auto' not in first_level)
        self.assertTrue('iregimode' not in first_level)
  
        first_level = self.get_completion('set param_card width')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6'  in first_level)          
        self.assertTrue('1' not in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt'  in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default' not in first_level)
        self.assertTrue('Auto' not in first_level)
        self.assertTrue('iregimode' not in first_level)
        
        first_level = self.get_completion('set param_card width wt')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6'  not in first_level)          
        self.assertTrue('1' not in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt'  not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default' in first_level)
        self.assertTrue('Auto' in first_level)
        self.assertTrue('Auto@NLO' in first_level)
        self.assertTrue('iregimode' not in first_level)
        
        first_level = self.get_completion('set param_card width 6')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6'  not in first_level)          
        self.assertTrue('1' not in first_level) 
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt'  not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('etaws'  not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default' in first_level)
        self.assertTrue('Auto' in first_level)
        self.assertTrue('Auto@NLO' in first_level)
        self.assertTrue('iregimode' not in first_level)       
        
        # Run_card completion -------------------------------------------------
        first_level = self.get_completion('set run_card ')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('etaj'  in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('as' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'  in first_level)
        self.assertTrue('Auto@NLO' not in first_level)
        self.assertTrue('iregimode' not in first_level) 
          
        first_level = self.get_completion('set run_card htjmax')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('as' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'  in first_level)
        self.assertTrue('iregimode' not in first_level)
        
        # ML_card completion ---------------------------------------------------               
        first_level = self.get_completion('set MadLoop_card')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level) 
        self.assertTrue('MadLoop_card' not in first_level)        
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('as' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'  in first_level)
        self.assertTrue('iregimode' in first_level)        
        
        first_level = self.get_completion('set MadLoop_card iregimode')
        self.assertTrue('MadWeight_card'  not in first_level)
        self.assertTrue('param_card'  not in first_level)
        self.assertTrue('run_card' not in first_level)
        self.assertTrue('MadLoop_card' not in first_level)         
        self.assertTrue('bjet_is_jet' not in first_level)          
        self.assertTrue('6' not in first_level)          
        self.assertTrue('etaj' not in first_level)
        self.assertTrue('mass' not in first_level)
        self.assertTrue('width' not in first_level)
        self.assertTrue('decay' not in first_level)
        self.assertTrue('wt' not in first_level)
        self.assertTrue('mh' not in first_level)
        self.assertTrue('as' not in first_level)
        self.assertTrue('mw_parameter'  not in first_level)        
        self.assertTrue('default'   in first_level)
        self.assertTrue('iregimode' not in first_level)        
        
        
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
        
    def test_modif_run_card(self):
        """ """

        run = self.cmd.run_card
        ptj = run['ptj']
        self.cmd.do_set('ptj 100')
        self.assertEqual(run['ptj'], 100)
        self.cmd.do_set('run_card ptj default')        
        self.assertEqual(run['ptj'], ptj)
        
        run.list_parameter.add('ptj')
        self.cmd.do_set('ptj 100, 300')
        self.assertEqual(run['ptj'], [100, 300])
        self.cmd.do_set('ptj 100  200.1')
        self.assertEqual(run['ptj'], [100, 200.1])
        self.cmd.do_set('ptj 100  200.1, 3e3')
        self.assertEqual(run['ptj'], [100, 200.1, 3000])

    def test_modif_ML_card(self):

        ML = self.cmd.MLcard 

        iregimode = self.cmd.MLcard['IREGIMODE']
        
        #check that nothing change if type is not correct
        self.cmd.do_set('IREGIMODE True') #should do nothrin
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertTrue('iregimode' not in ML.user_set)
        
        #check that we change it correctly when an input is given
        self.cmd.do_set('IREGIMODE %s' % (iregimode+1))
        self.assertEqual(iregimode+1, self.cmd.MLcard['IREGIMODE'])
        self.assertTrue('iregimode' in ML.user_set)        

        #check that we change it correctly when going back to default
        self.cmd.do_set('IREGIMODE default')
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertTrue('iregimode'  not in ML.user_set)
        
        # check that the full change is ok
        self.cmd.do_set('madloop_card IREGIMODE %s' % (iregimode+1))
        self.assertEqual(iregimode+1, self.cmd.MLcard['IREGIMODE'])
        self.assertTrue('iregimode' in ML.user_set) 
        
        IREGIRECY = self.cmd.MLcard['IREGIRECY'] 
        self.cmd.do_set('madloop_card IREGIRECY F')
        self.assertEqual(False, self.cmd.MLcard['iregirecy'])
        self.assertTrue(self.cmd.MLcard['iregirecy'] is self.cmd.MLcard['IREGIRECY'])
        
        self.cmd.do_set('madloop_card default')             
        self.assertEqual(iregimode, self.cmd.MLcard['IREGIMODE'])
        self.assertTrue('iregimode'  not in self.cmd.MLcard.user_set)                       
        self.assertEqual(IREGIRECY, self.cmd.MLcard['IREGIRECY'])
        self.assertTrue('iregimode'  not in self.cmd.MLcard.user_set)
        self.assertTrue('iregirecy'  not in self.cmd.MLcard.user_set)
        
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
        

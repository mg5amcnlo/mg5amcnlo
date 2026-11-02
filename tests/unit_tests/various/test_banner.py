################################################################################
#
# Copyright (c) 2012 The MadGraph7 Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import unittest
import json
import shutil
import tempfile
import madgraph.various.banner as bannermod
import madgraph.various.misc as misc
import os
import madgraph.models as models
import io
import re
import sys
from madgraph import MG5DIR



_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

pjoin = os.path.join


class TestBanner(unittest.TestCase):
    """ A class to test the banner functionality """
    
    
    def test_banner(self):

        #try to instantiate a banner with no argument
        mybanner = bannermod.Banner()
        self.assertTrue(hasattr, (mybanner, "lhe_version"))
        
        #check that you can instantiate a banner from a banner object
        secondbanner = bannermod.Banner(mybanner)
        
        # check that all attribute are common
        self.assertEqual(mybanner.__dict__, secondbanner.__dict__)
        
        # check that the two are different and independent
        self.assertNotEqual(id(secondbanner), id(mybanner))
        mybanner.test = True
        self.assertFalse(hasattr(secondbanner, "test"))
        
        #adding card to the banner
        mybanner.add_text('param_card', 
                          open(pjoin(_file_path,'..', 'input_files', 'param_card_0.dat')).read())

        mybanner.add_text('run_card', open(pjoin(_file_path, '..', 'input_files', 'run_card_ee.dat')).read())
        self.assertIn('slha', mybanner)
        
        #check that the banner can be written        
        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        mybanner.write(fsock)

        #charge a card
        mybanner.charge_card('param_card')
        self.assertTrue(hasattr(mybanner, 'param_card'))
        self.assertIsInstance(mybanner.param_card, models.check_param_card.ParamCard)
        self.assertIn('mass', mybanner.param_card)
        

        # access element of the card
        self.assertRaises(KeyError, mybanner.get, 'param_card', 'mt')
        self.assertEqual(mybanner.get('param_card', 'mass', 6).value, 175.0)
        self.assertEqual(mybanner.get('run_card', 'lpp1'), 0)
        


class TestConfigFileCase(unittest.TestCase):
    """ A class to test the TestConfig functionality """
    # a lot of the functionality is actually already tested in the child
    # TESTMadLoopParam and are not repeated here.
     
    def setUp(self):
        
        self.config = bannermod.ConfigFile()
        self.config.add_param('lower', 1)
        self.config.add_param('UPPER', 1)
        assert self.config.__dict__
   
    def test_sum_object(self):
        """ check for the case handling only #more test in TESTMadLoopParam """
        
        self.assertEqual(self.config.lower_to_case, {"lower":"lower", "upper":"UPPER"})

        # add a dictionary
        a = {'lower2':2, 'UPPER2':2, 'Mixed':2} 
        config2 = self.config + a
        
        #ensure that config is not change
        self.assertEqual(len(self.config),2)
        self.assertEqual(self.config.lower_to_case, {"lower":"lower", "upper":"UPPER"})

        self.assertEqual(type(config2), bannermod.ConfigFile)
        self.assertFalse(dict.__contains__(config2, 'UPPER2'))
        self.assertIn('UPPER2', config2)
        
        # from a dictionary add a config file
        config3 = a + self.config
        self.assertTrue(not hasattr(config3, 'lower_to_dict'))
        self.assertEqual(type(config3), dict)
        self.assertTrue(dict.__contains__(config3, 'UPPER2'))
        self.assertTrue(config3.__contains__('UPPER2'))        
        self.assertTrue(dict.__contains__(config3, 'UPPER'))
        self.assertTrue(config3.__contains__('UPPER'))
          
    def test_handling_list_of_values(self):
        """check that the read/write of a list of value works"""
        
        # add a parameter which can be a list
        self.config.add_param("list", [1])
        self.assertEqual(self.config['list'], [1])
        # try to write info in it via the string
        self.config['list'] = "1,2, 3, 4 , 5"

        self.assertEqual(self.config['list'],[1,2,3,4,5])
        self.config['list'] = [1.0,2,3+0j]
        self.assertEqual(self.config['list'],[1,2,3])


        
        # check that it fail for invalid input:
        self.assertRaises(Exception, self.config.__setitem__, 'list', [1,'a'])
        self.assertRaises(Exception, self.config.add_param, "list2", [1, 2.0])
        #self.assertRaises(Exception, self.config.add_param, 'list3', ['a'])
        
        #check that we can go back to non list format:
        self.config['list'] = '-2'
        self.assertEqual(self.config['list'], [-2])
        
        #check that space only format works as well
        self.config['list'] = "1 2 3 4e1"
        self.assertEqual(self.config['list'],[1,2,3,40])
        
        #check that space + command format works as well
        self.config['list'] = " 1 2, 3, 5d1 "
        self.assertEqual(self.config['list'],[1,2,3,50])        
        
        self.config['list'] = (1,2,3,'4')
        self.assertEqual(self.config['list'],[1,2,3,4]) 
        self.config['list'] = set((1,'2',3,'4'))
        self.assertEqual(set(self.config['list']),set([1,2,3,4])) 
        
        self.assertRaises(Exception, self.config.__setitem__, 'list', {1:2,3:4},raiseerror=True)
        

        # add a parameter which can be a list of string
        self.config.add_param("list_s", ['1'])
        self.assertEqual(self.config['list_s'], ['1'])
        self.config['list_s'] = " 1 2, 3, 5d1 "
        self.assertEqual(self.config['list_s'],['1','2','3', '5d1'])
        self.config['list_s'] = r" 1\ 2, 3, 5d1 "
        self.assertEqual(self.config['list_s'],[r'1\ 2','3', '5d1']) 

        self.config['list_s'] = "['--pdf=central', '--mur=1,2,3']"
        self.assertEqual(self.config['list_s'],['--pdf=central', '--mur=1,2,3']) 
        self.config['list_s'] = "[--pdf='central', --mur='1,2,3']"
        self.assertEqual(self.config['list_s'],['--pdf=\'central\'', '--mur=\'1,2,3\''])         
        
        # Fail to have the correct behavior for that one. Should be ok in general               
        #self.config['list_s'] = " 1\\ 2, 3, 5d1 "        
        #self.assertEqual(self.config['list_s'],['1\\', '2','3', '5d1'])

        # check status with allowed (auto filtering) and correct check
        self.config.add_param("list_a", [1], allowed=[0,1,2])
        self.assertEqual(self.config['list_a'], [1])
        self.config['list_a'] = "1 , 2"
        self.assertEqual(self.config['list_a'],[1,2])
        self.config['list_a'] = ["1"]
        self.assertEqual(self.config['list_a'],[1])
        self.config['list_a'] = "1,2,3"
        self.assertEqual(self.config['list_a'],[1,2]) # dropping not valid entry
        self.config['list_a'] = "3,4"
        self.assertEqual(self.config['list_a'],[1,2]) #default is to keep previous value


    def test_handling_dict_of_values(self):
        """check that the read/write of a list of value works"""
        
        # add a parameter which can be a list
        self.config.add_param("dict", {'__type__':1.0})
        self.assertEqual(self.config['dict'], {})
        self.assertFalse(self.config['dict'])
        self.assertEqual(dict.__getitem__(self.config,'dict'), {})
         
        # try to write info in it via the string
        self.config['dict'] = "1,2"
        self.assertEqual(self.config['dict'],{'1':2.0})
        self.config['dict'] = "3,4"
        self.assertEqual(self.config['dict'],{'1':2.0, '3': 4.0})
        self.config['dict'] = "5 6"
        self.assertEqual(self.config['dict'],{'1':2.0, '3': 4.0, '5':6.0})
        self.config['dict'] = "7:8"
        self.assertEqual(self.config['dict'],{'1':2.0, '3': 4.0, '5':6.0, '7':8.0 })
        self.config['dict'] = "7: 9.2"
        self.assertEqual(self.config['dict'],{'1':2.0, '3': 4.0, '5':6.0, '7':9.2 })        
        
        
        self.config['dict'] = "{5:6,'7':8}"
        self.assertEqual(self.config['dict'],{'5':6.0, '7': 8.0})        
        
        self.config['dict'] = {'5':6,'3':4+0j}
        self.assertEqual(self.config['dict'],{'5':6.0, '3': 4.0})           
        
        self.assertRaises(Exception, self.config.__setitem__, 'dict', [1,2,3])
        self.assertRaises(Exception, self.config.__setitem__, 'dict', {'test':'test'})
        self.assertRaises(Exception, self.config.__setitem__, 'dict', "22")

        self.config['dict'] = " {'TimeShower:QEDshowerByQ':0, 'TimeShower:QEDshowerByL':1.0}"
        self.assertEqual(self.config['dict'],{'TimeShower:QEDshowerByQ':0.0, 'TimeShower:QEDshowerByL': 1.0})
        
    def test_integer_handling(self):

        self.config.add_param("int", 1)
        self.config['int'] = '30*2'
        self.assertEqual(self.config['int'] ,30*2)
        
        self.config['int'] = 3.0
        self.assertEqual(self.config['int'] ,3)
         
        self.config['int'] = '3k'
        self.assertEqual(self.config['int'] ,3000)
        
        self.config['int'] = '3M'
        self.assertEqual(self.config['int'] ,3000000)                        

        self.config['int'] = '4d1'
        self.assertEqual(self.config['int'] ,40) 

        self.config['int'] = '30/2'
        self.assertEqual(self.config['int'] , 15)

    def test_float_handling(self):

        self.config.add_param("int", 1.0)
        self.config['int'] = '30*2'
        self.assertEqual(self.config['int'] ,30*2)
        
        self.config['int'] = 3.0
        self.assertEqual(self.config['int'] ,3)
         
        self.config['int'] = '3k'
        self.assertEqual(self.config['int'] ,3000)
        
        self.config['int'] = '3M'
        self.assertEqual(self.config['int'] ,3000000)                        

        self.config['int'] = '4d1'
        self.assertEqual(self.config['int'] ,40) 

        self.config['int'] = '30/4'
        self.assertEqual(self.config['int'] , 15/2.)

    def test_auto_handling(self):
        """check that any parameter can be set on auto and recover"""
        
        self.config['lower'] = 'auto'
        self.assertEqual(self.config['lower'],'auto')
        self.assertEqual(dict.__getitem__(self.config,'lower'),1)
        self.assertIn('lower', self.config.auto_set)
        self.assertNotIn('lower', self.config.user_set)
        
        self.config['lower'] = 2 
        self.assertEqual(self.config['lower'], 2)
        self.assertEqual(dict.__getitem__(self.config,'lower'),2)
        
        self.config.add_param('test', [1,2])
        self.config['test'] = 'auto'
        self.assertEqual(self.config['test'],'auto')
        self.assertEqual(dict.__getitem__(self.config,'test'),[1,2])
        
        self.assertRaises(Exception, self.config.__setitem__, 'test', 'onestring')
        self.config['test'] = '3,4'
        self.assertEqual(self.config['test'], [3,4])
        self.assertEqual(dict.__getitem__(self.config,'test'), [3,4])                
        
        self.config.set('test', ['1',5.0], user=True)
        self.config.set('test', 'auto', changeifuserset=False)
        self.assertEqual(self.config['test'], [1,5])
        self.assertEqual(dict.__getitem__(self.config,'test'), [1,5])
        
        self.config.set('test', 'auto', user=True)
        self.assertEqual(self.config['test'],'auto')
        self.assertEqual(dict.__getitem__(self.config,'test'), [1,5])
        
        for key, value in self.config.items():
            if key == 'test':
                self.assertEqual(value, 'auto')
                break
        else:
            self.assertFalse(True, 'wrong key when looping over key')
        
        
    def test_system_only(self):
        """test that the user can not modify a parameter system only"""
        
        self.config.add_param('test', [1,2], system=True)
        
        self.config['test'] = [3,4]
        self.assertEqual(self.config['test'], [3,4])
        
        self.config.set('test', '1 4', user=True)
        self.assertEqual(self.config['test'], [3,4])               
        
        self.config.set('test', '1 4', user=False)
        self.assertEqual(self.config['test'], [1,4])         

    def test_config_iadd(self):
        
        self.config['lower'] +=1
        self.assertTrue(self.config['lower'],2)
        
        #check that postscript are correctly called
        self.config.control = False

        #Note that this is a bit hacky since this is not a normal class function
        # but this does the job
        def f( value, *args, **opts):
            self.config.control=True
            
        self.config.post_set_lower = f
        self.config['lower'] +=1
        self.assertTrue(self.config['lower'],3)
        self.assertTrue(self.config.control)
      
      
    def test_for_loop(self):
        """ check correct handling of case"""
    
        keys = []
        for key in self.config:
            keys.append(key)
        self.assertEqual(set(keys), set(self.config.keys()))
        self.assertNotIn('upper', keys)
        self.assertIn('UPPER', keys)

    def test_guess_type(self):
        """check the guess_type_from_value(value) static function"""

        fct = bannermod.ConfigFile.guess_type_from_value
        self.assertEqual(fct("1.0"), "float")
        self.assertEqual(fct("1"), "int")
        self.assertEqual(fct("35"), "int")
        self.assertEqual(fct("35."), "float")
        self.assertEqual(fct("True"), "bool")
        self.assertEqual(fct("T"), "str")
        self.assertEqual(fct("auto"), "str")
        self.assertEqual(fct("my_name"), "str")
        self.assertEqual(fct("import tensorflow; sleep(10)"), "str")

        self.assertEqual(fct(1.0), "float")

        self.assertEqual(fct("[1,2]"), "list")
        self.assertEqual(fct("{1:2}"), "dict")


#    def test_in(self):
#        """actually tested in sum_object"""
#       
#    def test_update(self):
#        """actually tested in sum_object"""


class TestMadAnalysis5Card(unittest.TestCase):
    """ A class to test the MadAnalysis5 card IO functionality """

    def setUp(self):
        pass
    
    def test_MadAnalysis5Card(self):
        """ Basic check that the read-in write-out of MadAnalysis5 works as
        expected."""
        
        MG5aMCtag = bannermod.MadAnalysis5Card._MG5aMC_escape_tag
        
        input = io.StringIO(
"""%(MG5aMCtag)s inputs = *.hepmc *.stdhep
%(MG5aMCtag)s stdout_lvl=20
%(MG5aMCtag)s reconstruction_name = reco1
%(MG5aMCtag)s reco_output = lhe
First command of a reco1
Second command of a reco1
%(MG5aMCtag)s reconstruction_name = reco2
%(MG5aMCtag)s reco_output = root
First command of a reco2
Second command of a reco2
%(MG5aMCtag)s analysis_name = FirstAnalysis
%(MG5aMCtag)s set_reconstructions = ['reco1', 'reco2']
First command of a first analysis
#Second command of a first analysis
etc...
%(MG5aMCtag)s analysis_name = MyNewAnalysis
%(MG5aMCtag)s set_reconstructions = ['reco1']
First command of a new analysis
#Second command of a new analysis
etc...
%(MG5aMCtag)s reconstruction_name = recoA
%(MG5aMCtag)s reco_output = lhe
First command of a recoA
Second command of a recoA
etc...
%(MG5aMCtag)s recasting_commands
First command of recasting
#Second command of recasting
etc...
%(MG5aMCtag)s recasting_card
First command of recasting
#Second command of recasting
etc...
%(MG5aMCtag)s analysis_name = YetANewAnalysis
%(MG5aMCtag)s set_reconstructions = ['reco1', 'recoA']
First command of yet a new analysis
Second command of yet a new analysis
etc...
%(MG5aMCtag)s reconstruction_name = recoB
%(MG5aMCtag)s reco_output = root
First command of a recoB
Second command of a recoB
etc..."""%{'MG5aMCtag':MG5aMCtag})
        
        myMA5Card = bannermod.MadAnalysis5Card(input)
        input.seek(0)
        output = io.StringIO()
        myMA5Card.write(output)
        output.seek(0)
        self.assertEqual(myMA5Card,bannermod.MadAnalysis5Card(output))
        output.seek(0)
        output_target = input.getvalue().split('\n')
        output_target = [l for l in output_target if not l.startswith('#')]
        self.assertEqual(output.getvalue(),'\n'.join(output_target))
        
class TestPythia8Card(unittest.TestCase):
    """ A class to test the Pythia8 card IO functionality """
   
    def setUp(self):
        self.basic_PY8_template = open(pjoin(MG5DIR,'Template','LO','Cards',
                                         'pythia8_card_default.dat'),'r').read()
        
    def test_PY8Card_basic(self):
        """ Basic consistency check of a read-write of the default card."""
        
        pythia8_card_out = bannermod.PY8Card()
        out = io.StringIO()
        pythia8_card_out.write(out,self.basic_PY8_template)
        #       misc.sprint('WRITTEN:',out.getvalue())
        
        pythia8_card_read = bannermod.PY8Card()
        # Rewind
        out.seek(0)
        pythia8_card_read.read(out)       
        self.assertEqual(pythia8_card_out,pythia8_card_read)
        
        return
        
        # Below are some debug lines, comment the above return to run them
        # ========== 
        # Keep the following if you want to print out all parameters with
        # print_only_visible=False
        pythia8_card_read.system_set = set([k.lower() for k in 
                                                      pythia8_card_read.keys()])
        for subrunID in pythia8_card_read.subruns.keys():
            pythia8_card_read.subruns[subrunID].system_set = \
              set([k.lower() for k in pythia8_card_read.subruns[subrunID].keys()])
        # ==========
              
        out = io.StringIO()
        pythia8_card_read.write(out,self.basic_PY8_template)       
        misc.sprint('READ:',out.getvalue())
        out = io.StringIO()
        pythia8_card_read.write(out,self.basic_PY8_template,print_only_visible=True)       
        misc.sprint('Only visible:',out.getvalue())

    def test_PY8Card_with_subruns(self):
        """ Basic consistency check of a read-write of the default card."""
       
        default_PY8Card = bannermod.PY8Card(self.basic_PY8_template)

        template_with_subruns = self.basic_PY8_template + \
"""
Main:subrun=0
! My Run 0
blabla=2
Main:numberOfEvents      = 0
Main:subrun=7
! My Run 7
Main:numberOfEvents      = 73
Beams:LHEF='events_miaou.lhe.gz'
Main:subrun=12
! My other Run 
Main:numberOfEvents      = 120
bloublou=kramoisi
Beams:LHEF='events_ouaf.lhe.gz'
"""
        modified_PY8Card = bannermod.PY8Card(template_with_subruns)
        
        # Add the corresponding features to the default PY8 card
        default_PY8Card.subruns[0].add_param('blabla','2')
        default_PY8Card.subruns[0]['Main:numberOfEvents']=0
        PY8SubRun7 = bannermod.PY8SubRun(subrun_id=7)
        PY8SubRun7['Beams:LHEF']='events_miaou.lhe.gz'
        PY8SubRun7['Main:numberOfEvents']=73
        default_PY8Card.add_subrun(PY8SubRun7)
        PY8SubRun12 = bannermod.PY8SubRun(subrun_id=12)
        PY8SubRun12['Beams:LHEF']='events_ouaf.lhe.gz'
        PY8SubRun12['Main:numberOfEvents']=120
        PY8SubRun12.add_param('bloublou','kramoisi')
        default_PY8Card.add_subrun(PY8SubRun12)
        self.assertEqual(default_PY8Card, modified_PY8Card)

        # Now write the card
        out = io.StringIO()
        modified_PY8Card.write(out,self.basic_PY8_template)
        out.seek(0)
        read_PY8Card=bannermod.PY8Card(out)
        self.assertEqual(modified_PY8Card, read_PY8Card)

        # Now write the card, and write all parameters, including hidden ones.
        # We force that by setting them 'system_set'
        modified_PY8Card.system_set = set([k.lower() for k in 
                                                      modified_PY8Card.keys()])
        for subrunID in modified_PY8Card.subruns.keys():
            modified_PY8Card.subruns[subrunID].system_set = \
              set([k.lower() for k in modified_PY8Card.subruns[subrunID].keys()])
        out = io.StringIO()
        modified_PY8Card.write(out,self.basic_PY8_template)
        out.seek(0)        
        read_PY8Card=bannermod.PY8Card(out)
        self.assertEqual(modified_PY8Card, read_PY8Card)


class TestRivetCard(unittest.TestCase):
    """ Test the RivetCard functionality """

    def get_weight_name(self, run_opts={}, py8_opts={}, weight_name=None):
        run_card = bannermod.RunCardLO()
        for key, value in run_opts.items():
            run_card[key] = value
        py8_card = bannermod.PY8Card()
        for key, value in py8_opts.items():
            py8_card[key] = value
        rivet_card = bannermod.RivetCard()
        if weight_name is not None:
            rivet_card['weight_name'] = weight_name
        rivet_card.setWeightName(runcard=run_card, py8card=py8_card)
        return rivet_card['weight_name']

    def test_setWeightName(self):
        """ check that the weight name matches the HepMC weight name written
            by MG5aMC_PY8_interface for the central merging scale """

        # no merging
        self.assertEqual(self.get_weight_name({'ickkw': 0}), 'None')

        # MLM: user-defined and automatic (1.5*xqcut) qCut
        self.assertEqual(self.get_weight_name({'ickkw': 1},
                                              {'JetMatching:qCut': 30.0}),
                         'Weight_MERGING=30.000')
        self.assertEqual(self.get_weight_name({'ickkw': 1, 'xqcut': 20.0},
                                              {'JetMatching:qCut': -1.0}),
                         'Weight_MERGING=30.000')

        # CKKW-L: user-defined and automatic (ptlund/ktdurham) TMS
        self.assertEqual(self.get_weight_name({'ickkw': 0, 'ptlund': 15.0},
                                              {'Merging:TMS': 25.5}),
                         'Weight_MERGING=25.500')
        self.assertEqual(self.get_weight_name({'ickkw': 0, 'ptlund': 15.0},
                                              {'Merging:TMS': -1.0}),
                         'Weight_MERGING=15.000')
        self.assertEqual(self.get_weight_name({'ickkw': 0, 'ktdurham': 40.0},
                                              {'Merging:TMS': -1.0}),
                         'Weight_MERGING=40.000')

        # NLO run_card (no ktdurham/ptlund/xqcut entries): no crash
        for ickkw in [0, 3]:
            run_card = bannermod.RunCardNLO()
            run_card['ickkw'] = ickkw
            rivet_card = bannermod.RivetCard()
            rivet_card.setWeightName(runcard=run_card,
                                     py8card=bannermod.PY8Card())
            self.assertEqual(rivet_card['weight_name'], 'None')

        # user-defined weight name is kept
        self.assertEqual(self.get_weight_name({'ickkw': 1},
                                              {'JetMatching:qCut': 30.0},
                                              weight_name='MUR=1_MUF=1'),
                         'MUR=1_MUF=1')


def wrap_like_older_versions(text):
    """the proc card as ProcCard.write used to write it: every line cut at 70
    characters, wherever that falls, with a '\\' continuation"""
    out = []
    for line in text.split('\n'):
        while len(line) > 70:
            out.append(line[:70] + '\\')
            line = line[70:]
        out.append(line)
    return '\n'.join(out)


class TestProcCardWrappedLines(unittest.TestCase):
    """Older versions of ProcCard.write wrapped lines at 70 characters
    wherever that falls, so a command-line option can be split across two
    lines of proc_card_mg5.dat (old outputs, banners of old event files).
    Options must be read back through ProcCard, never by grepping the file.
    The writer now keeps every command on one line."""

    prefix = 'output standalone_fortran '
    options = ' --density=3,4 -f'

    def write_and_reread(self, lines, wrap=True):
        """write the card (wrapped as older versions did if wrap) and read
        it back"""
        card = bannermod.ProcCard()
        for line in lines:
            card.append(line)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pjoin(tmpdir, 'proc_card_mg5.dat')
            card.write(path)
            with open(path) as stream:
                raw = stream.read()
            if wrap:
                raw = wrap_like_older_versions(raw)
                with open(path, 'w') as stream:
                    stream.write(raw)
            return raw, bannermod.ProcCard(path)

    def test_write_keeps_long_lines_whole(self):
        output = self.prefix + '/' + 'd' * 80 + self.options
        generate = ('generate p p > t t~, (t > w+ b, w+ > e+ ve), '
                    '(t~ > w- b~, w- > mu- vm~) @0 QED=2 QCD=2')
        raw, card = self.write_and_reread([generate, output], wrap=False)
        lines = raw.split('\n')
        self.assertIn(output, lines)
        self.assertIn(generate, lines)
        self.assertFalse(any(line.endswith('\\') for line in lines))
        self.assertEqual(card.get_output_options(), {'density': '3,4'})

    def test_density_option_survives_every_wrap_position(self):
        # slide the 70-character cut through the whole option, value included
        start = 70 - len(self.prefix) - len(self.options) - 1
        split_in_flag = 0
        for path_length in range(start, 70 - len(self.prefix)):
            output_dir = '/' + 'd' * (path_length - 1)
            output = self.prefix + output_dir + self.options
            raw, card = self.write_and_reread(
                ['import model loop_sm',
                 'generate g g > z* g [sqrvirt=QCD]',
                 output])
            if not any('--density' in line for line in raw.split('\n')):
                split_in_flag += 1
            self.assertIn(output, card)
            self.assertEqual(card.get_output_options(), {'density': '3,4'},
                             msg='wrapped as:\n%s' % raw)
        # the regression case is in the sweep: '-\' then '-density=3,4 -f'
        self.assertGreater(split_in_flag, 0)

    def test_regression_layout(self):
        # the layout seen in a worktree, cut between the two dashes:
        # '--density' starts at column 69
        output_dir = '/' + 'd' * (69 - len(self.prefix) - 3) + 's'
        raw, card = self.write_and_reread(
            ['generate g g > z* g [sqrvirt=QCD]',
             self.prefix + output_dir + self.options])
        self.assertIn('s -\\\n-density=3,4 -f\n', raw)
        self.assertEqual(card.get_output_options(), {'density': '3,4'})

    def test_last_output_options(self):
        _, card = self.write_and_reread(
            ['generate e+ e- > mu+ mu-',
             'output standalone /tmp/a --prefix=int --hel_recycling=False'],
            wrap=False)
        self.assertEqual(card.get_output_options(),
                         {'prefix': 'int', 'hel_recycling': 'False'})

        _, card = self.write_and_reread(
            ['generate e+ e- > mu+ mu-', 'output standalone /tmp/a -f'])
        self.assertEqual(card.get_output_options(), {})

        _, card = self.write_and_reread(['generate e+ e- > mu+ mu-'])
        self.assertEqual(card.get_output_options(), {})

    def test_wrapped_modelname(self):
        # '-modelname' starts at column 66: cut as '-mode\' / 'lname'
        model = '/' + 'm' * (65 - len('import model ') - 1) + ' -modelname'
        raw, card = self.write_and_reread(['import model %s' % model])
        self.assertNotIn('-modelname', raw)
        self.assertIn('-modelname', card.get('full_model_line'))


import re
import shutil
class TestRunCardNLOMeFrame(unittest.TestCase):
    """me_frame is only meaningful for the whole initial state or for
    final-state particles; a partial initial state must be refused at parse
    time rather than reaching the generated Fortran."""

    def _card(self, me_frame):
        run_card = bannermod.RunCardNLO()
        run_card.set('me_frame', me_frame, user=True)
        return run_card

    def test_a_single_beam_is_refused(self):
        # The regression: this used to pass check_validity because only a
        # *mix* of initial and final legs was tested. A massless beam then
        # died in get_me_frame_boost with an opaque 'not timelike' stop, and
        # a massive one silently built the frame the guard means to refuse.
        for me_frame in ([1], [2]):
            with self.assertRaises(bannermod.InvalidRunCard):
                self._card(me_frame).check_validity()

    def test_both_beams_are_the_partonic_cm_and_allowed(self):
        self._card([1, 2]).check_validity()

    def test_final_state_only_is_allowed(self):
        self._card([3, 4]).check_validity()
        self._card([3]).check_validity()

    def test_mixing_initial_and_final_is_refused(self):
        for me_frame in ([1, 3], [1, 2, 3]):
            with self.assertRaises(bannermod.InvalidRunCard):
                self._card(me_frame).check_validity()


class TestRunCard(unittest.TestCase):
    """ A class to test the TestConfig functionality """
    # a lot of the functionality is actually already tested in the child
    # TESTMadLoopParam and are not repeated here.
    

    def setUp(self):
        self.debugging = getattr(unittest, 'debug', False)
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
            #if os.path.exists(self.tmpdir):
            #    shutil.rmtree(self.tmpdir)
            #os.mkdir(self.tmpdir)
        else:
            if os.path.exists(pjoin(MG5DIR, 'TEST_AMC')):
                shutil.rmtree(pjoin(MG5DIR, 'TEST_AMC'))
            os.mkdir(pjoin(MG5DIR, 'TEST_AMC'))
            self.tmpdir = pjoin(MG5DIR, 'TEST_AMC')
            
    def tearDown(self):
        if not self.debugging:
            shutil.rmtree(self.tmpdir)
        
    def test_basic(self):
        """ """
        
        # check the class factory works        
        run_card = bannermod.RunCard()
        self.assertIsInstance(run_card, bannermod.RunCard)
        self.assertIsInstance(run_card, bannermod.RunCardLO)
        self.assertNotIsInstance(run_card, bannermod.RunCardNLO)
        
        path = pjoin(_file_path, '..', 'input_files', 'run_card_matching.dat')
        run_card = bannermod.RunCard(path)
        self.assertIsInstance(run_card, bannermod.RunCard)
        self.assertIsInstance(run_card, bannermod.RunCardLO)
        self.assertNotIsInstance(run_card, bannermod.RunCardNLO)
        
        path = pjoin(_file_path,'..', 'input_files', 'run_card_nlo.dat')
        run_card = bannermod.RunCard(path)
        self.assertIsInstance(run_card, bannermod.RunCard)
        self.assertIsInstance(run_card, bannermod.RunCardNLO)
        self.assertNotIsInstance(run_card, bannermod.RunCardLO)
        
        #check the copy
        run_card2 = bannermod.RunCard(run_card)
        self.assertIsInstance(run_card, bannermod.RunCard)
        self.assertIsInstance(run_card, bannermod.RunCardNLO)
        self.assertNotIsInstance(run_card, bannermod.RunCardLO)
        #check all list/dict are define
        self.assertTrue(hasattr(run_card2, 'user_set'))
        self.assertTrue(hasattr(run_card2, 'hidden_param'))
        self.assertTrue(hasattr(run_card2, 'includepath')) 
        self.assertTrue(hasattr(run_card2, 'fortran_name'))
        self.assertFalse(hasattr(run_card2, 'default'))
        self.assertTrue(hasattr(run_card2, 'cuts_parameter'))   
              

    def test_default(self):
      
        run_card = bannermod.RunCard()
#        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        fsock = open(pjoin(self.tmpdir,'run_card_test'),'w')
        run_card.write(fsock)
        fsock.close()
        run_card2 = bannermod.RunCard(fsock.name)
      
        for key in run_card:
            if key in ['pdlabel1', 'pdlabel2']:
                continue
            self.assertEqual(run_card[key], run_card2[key], '%s element does not match %s, %s' %(key, run_card[key], run_card2[key]))
      
        run_card = bannermod.RunCardNLO()
#        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        fsock = open(pjoin(self.tmpdir,'run_card_test2'),'w')
        run_card.write(fsock)
        fsock.close()
        #card should be identical if we do not run the consistency post-processing
        run_card2 = bannermod.RunCard(fsock.name, consistency=False)
        for key in run_card:
            self.assertEqual(run_card[key], run_card2[key], 'not equal entry for %s" %s!=%s' %(key,run_card[key], run_card2[key]))  
            
        #but default can be updated otherwise
        run_card3 = bannermod.RunCard(fsock.name)
        has_difference = False
        has_userset = False
        for key in run_card:
            key = key.lower()
            if run_card[key] != run_card3[key]:
                has_difference = True
                self.assertIn(key.lower(), run_card.hidden_param)
                self.assertNotIn(key.lower, run_card3.user_set)
            if key in run_card3.user_set:
                has_userset=True   
                self.assertNotIn(key, run_card.user_set)
        self.assertTrue(has_difference)
        self.assertTrue(has_userset)
        
        #write run_card3 and check that nothing is changed
#        fsock2 = tempfile.NamedTemporaryFile(mode = 'w')
        fsock2 = open(pjoin(self.tmpdir,'run_card_test3'),'w')
        run_card3.write(fsock2)
        fsock2.close()

        text1 = open(fsock.name).read()
        text2 = open(fsock2.name).read()
        self.assertFalse("$RUNNING" in text1)
        self.assertFalse("$RUNNING" in text2)
        text1 = text1.replace('\n\n\n', '\n')
        text1 = text1.replace('\n \n', '\n')
        text2 = text2.replace('\n \n', '\n')
        self.assertEqual(text1, text2)

    def test_check_valid_LO(self):
        """ensure that some handling are done correctly"""

        run_card = bannermod.RunCardLO()
        run_card['dsqrt_q2fact1'] = 10
        run_card['dsqrt_q2fact2'] = 20

        # check that if fixed_fac_scale is on False and lpp1=1 fixed_fac_scale1 is False but if lpp=2 then  fixed_fac_scale1 is True
        run_card.set('fixed_fac_scale', False, user=True)
        run_card.set('lpp1', 2, user=True)
        run_card.check_validity()
        self.assertEqual(run_card['fixed_fac_scale1'], True)
        self.assertEqual(run_card['fixed_fac_scale2'], False)


        run_card.set('lpp1', 1, user=True)
        run_card.set('pdlabel', 'none', user=True)
        with self.assertRaises(bannermod.InvalidRunCard):
            run_card.check_validity()
        run_card.set('pdlabel', 'nn23lo1', user=True)
        self.assertEqual(run_card['pdlabel'], 'nn23lo1')
        run_card.check_validity()
        self.assertEqual(run_card['fixed_fac_scale1'], False)
        self.assertEqual(run_card['fixed_fac_scale2'], False)

        # check that for elastic a a collision we do  not force to use fixed_fac_scale1/2
        run_card.set('lpp1', 2, user=True)
        run_card.set('lpp2', 2, user=True)
        #with self.assertRaises(bannermod.InvalidRunCard):
        run_card.check_validity()
        run_card.set('fixed_fac_scale1', False, user=True)
        run_card.set('fixed_fac_scale2', False, user=True)
        run_card.check_validity()  # no crashing anymore
        self.assertEqual(run_card['fixed_fac_scale1'], False)
        self.assertEqual(run_card['fixed_fac_scale2'], False)

    def test_guess_entry_fromname(self):
        """ check that the function guess_entry_fromname works as expected
        """

        run_card = bannermod.RunCardLO()
        fct = run_card.guess_entry_fromname
        input = ("STR_INCLUDE_PDF", "True ")
        expected = ("str", "INCLUDE_PDF", {})
        self.assertEqual(fct(*input), expected)

        input = ("INCLUDE_PDF", "True ")
        expected = ("bool", "INCLUDE_PDF", {})
        self.assertEqual(fct(*input), expected)

        # note MIN is case sensitive
        input = ("min_PDF", "1.45")
        expected = ("float", "min_PDF", {'cut':True})
        self.assertEqual(fct(*input), expected)

        input = ("MIN_PDF", "12345")
        expected = ("int", "MIN_PDF", {})
        self.assertEqual(fct(*input), expected)

        input = ("test_list", "[1,2,3,4,5]")
        expected = ("list", "test_list", {'typelist': int})
        self.assertEqual(fct(*input), expected)

        input = ("test_data", "[1,2,3,4,5]")
        expected = ("list", "test_data",  {'typelist': int})
        self.assertEqual(fct(*input), expected)

        input = ("test_data<cut=True><include=False><fortran_name=input_2>", "[1,2,3,4,5]")
        expected = ("list", "test_data",  
        {'typelist': int, 'cut':True, 'include':False , 'fortran_name':'input_2', 'autodef':False})
        self.assertEqual(fct(*input), expected)

        input = ("list_float_data", "[1,2,3,4,5]")
        expected = ("list", "data",  {'typelist': float})
        self.assertEqual(fct(*input), expected)

        input = ("list_data", "[1,2,3,4,5]")
        expected = ("list", "data",  {'typelist': int})
        self.assertEqual(fct(*input), expected)

        input = ("data", "{'__type__':1.0, 'value':3.0}")
        expected = ("dict", "data", {'autodef': False, 'include': False})
        self.assertEqual(fct(*input), expected)

        input = ("data", "1,2,3")
        expected = ("list", "data", {'typelist': int})
        self.assertEqual(fct(*input), expected)

    def test_add_unknown_entry(self):
        """check that one can added complex structure via unknown entry functionality with the smart detection.
        
           note that the smart detection is done via guess_entry_fromname which is tested by
           test_guess_entry_fromname. So this test is mainly to test that the output function of that function is correctly
           linked to add_param as done in add_unknown_entry
        """

        run_card = bannermod.RunCardLO()
        fct = run_card.add_unknown_entry

        # simple one 
        input = ("STR_INCLUDE_PDF", "True ", False)
        fct(*input)
        # check value and that parameter is hidden by default and in autodef
        name = "INCLUDE_PDF" 
        self.assertEqual(run_card[name], "True")
        self.assertIn(name.lower(), run_card.hidden_param)
        self.assertIn(name.lower(), run_card.definition_path[True])

        # complex case: list + metadata
        input = ("test_data<cut=True><include=False><fortran_name=input_2>", "[1,2,3,4,5]", False)
        fct(*input)
        # check value and that parameter is hidden by default and in autodef
        name = "test_data"
        self.assertEqual(run_card[name], [1,2,3,4,5]) # this check that list are correctly formatted
        self.assertIn(name.lower(), run_card.hidden_param)
        self.assertNotIn(name.lower(), run_card.definition_path[True]) # since include is False
        # check that metadata was passed correctly
        self.assertNotIn(name, run_card.includepath[True])
        self.assertIn(name, run_card.cuts_parameter)
        self.assertIn(name, run_card.fortran_name)
        self.assertEqual(run_card.fortran_name[name], "input_2")
        self.assertIn(name, run_card.list_parameter)
        self.assertEqual(run_card.list_parameter[name], int)


        # complex case: dictionary 
        input = ("test_dict", "{'__type__':1.0, '6':3.0}", False)
        fct(*input)
        # check value and that parameter is hidden by default and in autodef
        name = "test_dict"
        self.assertEqual(run_card[name], {'__type__':1.0, '6':3.0}) # this check that list are correctly formatted
        self.assertIn(name.lower(), run_card.hidden_param)
        # default for dict is not to include in Fortran
        self.assertNotIn(name.lower(), run_card.definition_path[True]) 
        self.assertNotIn(name, run_card.includepath[True])

        # check that one can overwrite hidden
        input = ("max_data<hidden=False>", "3.0", False)
        fct(*input)
        name = "max_data"
        self.assertEqual(run_card[name], 3.0)
        self.assertNotIn(name.lower(), run_card.hidden_param)

        # check that one can overwrite autodef
        input = ("max_data2<autodef=False>", "3", False)
        fct(*input)
        name = "max_data2"
        self.assertEqual(run_card[name], 3.0)
        self.assertNotIn(name.lower(), run_card.definition_path[True])
        self.assertIn(name.lower(), run_card.hidden_param)

        # check that one can overwrite include to False but autodef to True
        # check that one can overwrite autodef
        input = ("data3<autodef=True><include=False>", "True", False)
        fct(*input)
        name = "data3"
        self.assertEqual(run_card[name], 1.0)
        self.assertIn(name.lower(), run_card.definition_path[True])
        self.assertIn(name.lower(), run_card.hidden_param)
        self.assertNotIn(name, run_card.includepath[True])

    def test_add_definition(self):
        """ check the functionality that add an entry to an include file.
            check also that the common block is added automatically
        """

        run_card = bannermod.RunCardLO()
        run_card.add_unknown_entry("STR_INCLUDE_PDF", "True ", unknow_warning=False)
        f = io.StringIO()
        f.write("c .   this is a comment to test feature of missing end line ")
        run_card.write_autodef(None,output_file=f)
        self.assertIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("C     START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C     STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf", f.getvalue()) #no automatic formatting due to iostring for unittest

        # adding a second in place
        run_card.add_unknown_entry("BOOL_INCLUDE_PDF2", "True ", unknow_warning=False)
        run_card.write_autodef(None,output_file=f)
        self.assertIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertIn("C     START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C     STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        # order of the two variable within the common block is not important
        if "COMMON/USER_CUSTOM_RUN/include_pdf," in f.getvalue():
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf,include_pdf2", f.getvalue())
        else:
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf2,include_pdf", f.getvalue())

        # reset, keep one , remove one and add a new one (keep same stream)
        run_card = bannermod.RunCardLO()
        run_card.add_unknown_entry("BOOL_INCLUDE_PDF2", "True ", unknow_warning=False)
        run_card.add_unknown_entry("test_list", "[1,2,3,4,5]", unknow_warning=False)
        run_card.write_autodef(None,output_file=f)
        self.assertNotIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertIn("INTEGER TEST_LIST(0:5)", f.getvalue())
        # check common block part
        self.assertIn("C     START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C     STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        if "COMMON/USER_CUSTOM_RUN/include_pdf2," in f.getvalue():
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf2,test_list", f.getvalue())
        else:
            self.assertIn("COMMON/USER_CUSTOM_RUN/test_list,include_pdf2", f.getvalue())

        #change list size
        run_card["test_list"] = [1,2,3,4,5,6,7]
        run_card.write_autodef(None,output_file=f)
        self.assertNotIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertNotIn("INTEGER TEST_LIST(0:5)", f.getvalue())
        self.assertIn("INTEGER TEST_LIST(0:7)", f.getvalue())
        # check common block part
        self.assertIn("C     START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C     STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        if "COMMON/USER_CUSTOM_RUN/include_pdf2," in f.getvalue():
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf2,test_list", f.getvalue())
        else:
            self.assertIn("COMMON/USER_CUSTOM_RUN/test_list,include_pdf2", f.getvalue())

        #check that cleaning is occurring correctly
        run_card = bannermod.RunCardLO()
        run_card.write_autodef(None,output_file=f)
        self.assertNotIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertNotIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertNotIn("INTEGER TEST_LIST(0:5)", f.getvalue())
        self.assertNotIn("INTEGER TEST_LIST(0:7)", f.getvalue())
        # check common block part
        self.assertNotIn("C     START USER COMMON BLOCK", f.getvalue())
        self.assertNotIn("C     STOP USER COMMON BLOCK", f.getvalue())
        self.assertNotIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())

    def test_autodef_nomissmatch(self):
        """
        check that the code detects LO/NLO cards mismatch and crash correctly in that case
        """
        
        LO = bannermod.RunCardLO()
        NLO = bannermod.RunCardNLO()
        flo = io.StringIO()
        fnlo = io.StringIO()
        LO.write(flo)
        NLO.write(fnlo)
        loinput = flo.getvalue().split('\n')
        nloinput = fnlo.getvalue().split('\n')
        
        # check that LO card  can not be used for NLO run
        self.assertRaises(bannermod.InvalidRunCard, NLO.read, loinput)

        # check that NLO card  can not be used for LO run    
        self.assertRaises(bannermod.InvalidRunCard, LO.read, nloinput)


    def test_custom_fcts(self):
        """check that the functionality to replace user_define function is 
        working as expected"""

        custom_contents1 = """
      subroutine get_dummy_x1(sjac, X1, R, pbeam1, pbeam2, stot, shat)
      implicit none
      include 'maxparticles.inc'
      include 'run.inc'
c      include 'genps.inc'
      double precision sjac ! jacobian. should be updated not reinit
      double precision X1   ! bjorken X. output
      double precision R    ! random value after grid transfrormation. between 0 and 1
      double precision pbeam1(0:3) ! momentum of the first beam (input and/or output)
      double precision pbeam2(0:3) ! momentum of the second beam (input and/or output)
      double precision stot        ! total energy  (input and /or output)
      double precision shat        ! output

c     global variable to set (or not)
      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap

      set_cm_rap=.true. ! then cm_rap will be set as .5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
                         ! ebeam(1) and ebeam(2) are defined here thanks to 'run.inc'
      shat = x1**2*ebeam(1)*ebeam(2)
      CHECK1
      return
      end
     """   
        custom_contents2 = """
     logical  function dummy_boostframe()
      implicit none
c
c
      dummy_boostframe = .true.
      CHECK2
      return
      end
        """
        custom_contents = custom_contents1 + custom_contents2

        # prepare simplify setup
        os.mkdir(pjoin(self.tmpdir,'SubProcesses'))
        import madgraph.iolibs.files as files
        files.cp(pjoin(MG5DIR,'Template','LO','SubProcesses','dummy_fct.f'), pjoin(self.tmpdir,'SubProcesses'))
        open(pjoin(self.tmpdir, 'custom'),'w').write(custom_contents)
        
        #launch the function
        LO = bannermod.RunCardLO()
        LO.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)
        
        #test the functionality
        #check that .orig is indeed created
        self.assertTrue(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))

        #check that new function have been written
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertIn('CHECK1', new_text)
        self.assertIn('CHECK2', new_text)
        self.assertIn("DUMMY_CUTS=.TRUE.", new_text)
        self.assertIn("SHAT = X(1)*X(2)*EBEAM(1)*EBEAM(2)", new_text)
        self.assertIn("LOGICAL  FUNCTION DUMMY_BOOSTFRAME()", new_text)

        # launch the function with only one edited
        open(pjoin(self.tmpdir, 'custom'),'w').write(custom_contents1)
        LO.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)

        #test the functionality
        #check that .orig is still created
        self.assertTrue(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))
        orig = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')).read()
        self.assertNotIn('CHECK1', orig)
        self.assertNotIn('CHECK2', orig)

        #check that new function have been written
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertIn('CHECK1', new_text)
        self.assertNotIn('CHECK2', new_text)
        self.assertIn("DUMMY_CUTS=.TRUE.", new_text)
        self.assertIn("SHAT = X(1)*X(2)*EBEAM(1)*EBEAM(2)", new_text)
        self.assertIn("LOGICAL  FUNCTION DUMMY_BOOSTFRAME()", new_text)

        # check that cleaning works
        LO.edit_dummy_fct_from_file([], self.tmpdir)
        self.assertFalse(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertEqual(orig, new_text)
        self.assertNotIn('CHECK1', new_text)
        self.assertNotIn('CHECK2', new_text)


    def test_custom_fcts_uppercase(self):
        """fortran is case insensitive: a custom_fcts file written in upper case
           (which is the idiomatic f77 style) has to be accepted, has to be
           written in the correct file and has to replace --not duplicate--
           the original routine."""

        custom_contents = """
      LOGICAL FUNCTION DUMMY_CUTS(P)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL)
      DUMMY_CUTS = .TRUE.
      CHECKUP
      RETURN
      END

      SUBROUTINE USER_UPPER_FCT()
      IMPLICIT NONE
      CHECKUSER
      RETURN
      END
        """

        # prepare simplify setup
        os.mkdir(pjoin(self.tmpdir,'SubProcesses'))
        import madgraph.iolibs.files as files
        files.cp(pjoin(MG5DIR,'Template','LO','SubProcesses','dummy_fct.f'), pjoin(self.tmpdir,'SubProcesses'))
        open(pjoin(self.tmpdir, 'custom'),'w').write(custom_contents)

        LO = bannermod.RunCardLO()
        # this used to raise InvalidRunCard since the lookup was case sensitive
        LO.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)

        # the correct file is the one which has been patched
        self.assertTrue(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertIn('CHECKUP', new_text)
        self.assertIn('CHECKUSER', new_text)

        # the original dummy_cuts has to be removed, not duplicated
        # (otherwise the fortran compiler complains about a duplicated symbol)
        self.assertEqual(1, len(re.findall(r'FUNCTION\s+DUMMY_CUTS', new_text, re.I)))
        # the routine we did not overwrite is still there
        self.assertIn('GET_DUMMY_X1', new_text)

        # and cleaning still works
        LO.edit_dummy_fct_from_file([], self.tmpdir)
        self.assertFalse(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertNotIn('CHECKUP', new_text)
        self.assertNotIn('CHECKUSER', new_text)

    def test_custom_fcts_unknown_fct(self):
        """a function which is not allowed to be overwritten has to raise an
           error which actually names that function"""

        custom_contents = """
      LOGICAL FUNCTION NOT_A_DUMMY_FCT(P)
      IMPLICIT NONE
      NOT_A_DUMMY_FCT = .TRUE.
      RETURN
      END
        """

        os.mkdir(pjoin(self.tmpdir,'SubProcesses'))
        import madgraph.iolibs.files as files
        files.cp(pjoin(MG5DIR,'Template','LO','SubProcesses','dummy_fct.f'), pjoin(self.tmpdir,'SubProcesses'))
        open(pjoin(self.tmpdir, 'custom'),'w').write(custom_contents)

        for card in [bannermod.RunCardLO(), bannermod.RunCardNLO()]:
            try:
                card.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)
            except bannermod.InvalidRunCard as error:
                # the name of the offending function has to be in the message
                self.assertIn('NOT_A_DUMMY_FCT', str(error))
                self.assertNotIn('%s', str(error))
            else:
                self.fail('InvalidRunCard should have been raised')


    # the custom dynamical scale advertised in the FAQ (answers.launchpad.net/mg5amcnlo/+faq/3325)
    custom_scale = """
      double precision function user_dynamical_scale(P)
      implicit none
      include 'nexternal.inc'
      double precision P(0:3, nexternal)
      include 'run.inc'
      character*80 temp_scale_id
      common/ctemp_scale_id/temp_scale_id
      double precision dot, pt
      double precision xm2
      xm2 = dot(P(0,3),P(0,3))
      user_dynamical_scale = sqrt(xm2 + 0.5d0*(pt(P(0,3))**2 + pt(P(0,4))**2))
      temp_scale_id = 'CHECKSCALE'
      return
      end
        """

    def test_custom_fcts_vector_inc_lo(self):
        """a (pre 3.6) LO custom function including run.inc needs vector.inc to
           be added since run.inc dimensions arrays with VECSIZE_MEMMAX"""

        os.mkdir(pjoin(self.tmpdir,'SubProcesses'))
        import madgraph.iolibs.files as files
        files.cp(pjoin(MG5DIR,'Template','LO','SubProcesses','dummy_fct.f'), pjoin(self.tmpdir,'SubProcesses'))
        open(pjoin(self.tmpdir, 'custom'),'w').write(self.custom_scale)

        LO = bannermod.RunCardLO()
        LO.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)

        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertIn('CHECKSCALE', new_text)
        fct = new_text[new_text.index('USER_DYNAMICAL_SCALE'):]
        self.assertIn("INCLUDE 'vector.inc'", fct)
        # and it has to be included *before* run.inc
        self.assertLess(fct.index("INCLUDE 'vector.inc'"), fct.index("INCLUDE 'run.inc'"))

    def test_custom_fcts_no_vector_inc_nlo(self):
        """vector.inc does not exist in a NLO output (and run.inc does not need
           it there): it must not be added to the user function.
           see bug #2147417"""

        os.mkdir(pjoin(self.tmpdir,'SubProcesses'))
        import madgraph.iolibs.files as files
        files.cp(pjoin(MG5DIR,'Template','NLO','SubProcesses','dummy_fct.f'), pjoin(self.tmpdir,'SubProcesses'))
        open(pjoin(self.tmpdir, 'custom'),'w').write(self.custom_scale)

        NLO = bannermod.RunCardNLO()
        NLO.edit_dummy_fct_from_file([pjoin(self.tmpdir, 'custom')], self.tmpdir)

        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        # the function is correctly written ...
        self.assertIn('CHECKSCALE', new_text)
        self.assertIn('USER_DYNAMICAL_SCALE', new_text)
        # ... but without any include of vector.inc (which does not exist at NLO)
        self.assertNotIn('vector.inc', new_text.lower())

        # cleaning still works
        NLO.edit_dummy_fct_from_file([], self.tmpdir)
        self.assertFalse(os.path.exists(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f.orig')))
        new_text = open(pjoin(self.tmpdir,'SubProcesses','dummy_fct.f')).read()
        self.assertNotIn('CHECKSCALE', new_text)

    def test_retro_compatible_mode_selection(self):
        """the guard on the shipped file: a fix is only applied if the original
           file does use the corresponding include itself"""

        # the static method itself is unchanged when explicitly asked for the fix
        lines = ["      double precision function user_dynamical_scale(P)",
                 "      implicit none",
                 "      include 'run.inc'",
                 "      end"]
        self.assertIn("       include 'vector.inc'",
                      bannermod.RunCard.retro_compatible_custom_fct(lines, mode=['vector.inc']))
        # but an empty mode disables every fix
        self.assertEqual(lines,
                      bannermod.RunCard.retro_compatible_custom_fct(lines, mode=[]))

        # LO opts-in for the vector.inc fix, NLO does not
        self.assertIn('vector.inc', bannermod.RunCardLO.retro_compatible_modes)
        self.assertNotIn('vector.inc', bannermod.RunCardNLO.retro_compatible_modes)

    def test_pdlabel_block(self):
        """ check that pdlabel handling is done correctly
            this include that check_validity works as expected for such parameter too """

        run_card = bannermod.RunCardLO()

        # setting e- p collision
        run_card['lpp1'] = 3
        run_card['lpp2'] = 1
        run_card.check_validity()
        # check that pdlabel is set correctly
        # previously PDF was nn23lo1 (lhaid 230000) with this reference value
        # 'mixed'/'nn23lo1': the proton beam now defaults to LHAPDF
        # (NNPDF40_lo_as_01180), so the pair is eva + lhapdf and the combined
        # label is the one that links the right library.
        self.assertEqual(run_card['pdlabel'], 'lhapdf')
        self.assertEqual(run_card['pdlabel1'], 'eva') # since automatically set to eva if lpp=3/4 and pdlabel is lhapdf/nnpdf
        self.assertEqual(run_card['pdlabel2'], 'lhapdf')
        run_card.set('pdlabel', 'lhapdf', user=True) 
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'lhapdf') #important for linking the correct library
        self.assertEqual(run_card['pdlabel1'], 'eva') # since automatically set to eva if lpp=3/4 and pdlabel is lhapdf/nnpdf
        self.assertEqual(run_card['pdlabel2'], 'lhapdf')
        # check that pdlabel is set on the proton pdf
        self.assertEqual(run_card['pdlabel'], run_card['pdlabel2'])

        # setting p p collision
        run_card = bannermod.RunCardLO()
        run_card['lpp1'] = 1
        run_card['lpp2'] = 1
        run_card.check_validity()
        # check that pdlabel is set on the proton pdf
        self.assertEqual(run_card['pdlabel'], run_card['pdlabel2'])
        self.assertEqual(run_card['pdlabel'], run_card['pdlabel1'])
        # should now allow asymmetric pdlabel here
        run_card.set('pdlabel1','lhapdf', user=True) 
        # 'nn23lo1' rather than the invalid 'nnpdf23lo1' used before: that one
        # was rejected and left pdlabel2 at its default, which only differed
        # from 'lhapdf' while the default was nn23lo1 itself.
        run_card.set('pdlabel2', 'nn23lo1', user=True) 
        with self.assertRaises(bannermod.InvalidRunCard):
            run_card.check_validity()
        run_card.set('pdlabel2', 'lhapdf', user=True) 
        run_card.check_validity()

        # setting mu+ mu- collision
        run_card = bannermod.RunCardLO()
        run_card['lpp1'] = 0
        run_card['lpp2'] = 0
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'none')
        # one EVA mode
        run_card['lpp2'] = 4
        run_card.set('pdlabel2', 'eva', user=True)
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'eva')
        self.assertEqual(run_card['pdlabel1'], 'none')
        self.assertEqual(run_card['pdlabel2'], 'eva')
        # one EVA mode
        run_card['lpp1'] = 3
        run_card['lpp2'] = 0
        run_card.set('pdlabel1', 'iww', user=True)
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'iww')
        self.assertEqual(run_card['pdlabel2'], 'none')
        self.assertEqual(run_card['pdlabel1'], 'iww')
        
        
        # double EVA mode
        run_card = bannermod.RunCardLO()
        run_card['lpp1'] = 4
        run_card['lpp2'] = 4
        run_card.set('pdlabel', 'eva', user=True)
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'eva')
        self.assertEqual(run_card['pdlabel1'], 'eva')
        self.assertEqual(run_card['pdlabel2'], 'eva') 

        #check that random PDF can not be assigned
        run_card.set('pdlabel', 'xxx', user=True)
        self.assertNotEqual(run_card['pdlabel'], 'xxx')

        # dressed electron check list of valid dressed pdf is working
        self.assertEqual(len(run_card.allowed_lep_densities), 2)
        self.assertEqual(len(run_card.allowed_lep_densities[(-11,11)]), 7)
        self.assertEqual(len(run_card.allowed_lep_densities[(-13,13)]), 1)

        # Dressed lepton
        run_card = bannermod.RunCardLO()
        run_card['lpp1'] = 4
        run_card['lpp2'] = 4

        run_card.set('pdlabel', 'isronlyll', user=True)
        run_card.check_validity()
        self.assertEqual(run_card['pdlabel'], 'isronlyll') # at python kept dedicated value
        self.assertEqual(run_card['pdlabel1'], 'isronlyll')
        self.assertEqual(run_card['pdlabel2'], 'isronlyll')
        # check that at fortran pdlabel is passed to generic value "dressed"
        # but that individual values are kept
        f = io.StringIO()
        run_card.write_include_file(None,output_file=f)
        self.assertIn("pdlabel = 'dressed'", f.getvalue())
        self.assertIn("pdsublabel(1) = 'isronlyll'", f.getvalue())
        self.assertIn("pdsublabel(2) = 'isronlyll", f.getvalue())
        # to be 100% that wrong name is not passed to fortran
        self.assertNotIn("pdlabel = 'isronlyll'", f.getvalue())

    def test_fixed_fac_scale_block(self):

        run_card = bannermod.RunCardLO()
        run_card['dsqrt_q2fact1'] = 10
        run_card['dsqrt_q2fact2'] = 20
        self.assertNotIn('fixed_fact_scale', run_card.display_block)
        self.assertTrue(bannermod.fixedfacscale.status(run_card))


        run_card.set('fixed_fac_scale', True, user=True)
        self.assertEqual(run_card['fixed_fac_scale1'], True)
        self.assertEqual(run_card['fixed_fac_scale2'], True)
        self.assertIn('fixed_fac_scale', run_card.user_set)
        self.assertNotIn('fixed_fac_scale1', run_card.user_set)
        self.assertNotIn('fixed_fac_scale2', run_card.user_set)
        self.assertTrue(bannermod.fixedfacscale.status(run_card))
        f = io.StringIO()
        run_card.write(output_file=f)
        self.assertIn("True = fixed_fac_scale ", f.getvalue())


        f = io.StringIO()
        run_card.write_include_file(None,output_file=f)
        self.assertIn("fixed_fac_scale1 = .true.", f.getvalue())
        self.assertIn("fixed_fac_scale2 = .true.", f.getvalue())



        run_card.set('fixed_fac_scale1', False, user=True)
        self.assertFalse(bannermod.fixedfacscale.status(run_card))
        self.assertNotIn('fixed_fac_scale', run_card.user_set)
        self.assertNotIn('fixed_fac_scale2', run_card.user_set)    
        self.assertNotIn('fixed_fact_scale', run_card.display_block)

        f = io.StringIO()
        run_card.write(output_file=f)
        self.assertNotIn("True = fixed_fac_scale ", f.getvalue())
        self.assertIn("False = fixed_fac_scale1", f.getvalue())
        self.assertIn("True = fixed_fac_scale2", f.getvalue())

        # check that if  'fixed_fact_scale' is set within display_block
        # the mode is to keep the value of fixed_fac_scale1/2 within the card even if
        # both are identical
        run_card.display_block.append('fixed_fact_scale')
        self.assertFalse(bannermod.fixedfacscale.status(run_card))
        run_card.set('fixed_fac_scale', True, user=True)
        self.assertFalse(bannermod.fixedfacscale.status(run_card)) 

        f = io.StringIO()
        run_card.write_include_file(None,output_file=f)
        self.assertIn("fixed_fac_scale1 = .true.", f.getvalue())
        self.assertIn("fixed_fac_scale2 = .true.", f.getvalue())

        f = io.StringIO()
        run_card.write(output_file=f)
        self.assertNotIn("True = fixed_fac_scale ", f.getvalue())
        self.assertIn("True = fixed_fac_scale1", f.getvalue())
        self.assertIn("True = fixed_fac_scale2", f.getvalue())


    def test_negative_iseed(self):
        """Check that a negative iseed is preserved on disk across runs but
        exported as its absolute value to the Fortran include file. This is
        verified for both LO and NLO run cards, and for the
        `reset_iseed_in_run_card` helper used at run time.
        """
        import madgraph.interface.common_run_interface as common_run

        for run_card_class in (bannermod.RunCardLO, bannermod.RunCardNLO):
            # 1. write_include_file must export abs(iseed)
            run_card = run_card_class()
            run_card.set('iseed', -42, user=True)
            f = io.StringIO()
            run_card.write_include_file(None, output_file=f)
            content = f.getvalue()
            self.assertIn("iseed = 42", content)
            self.assertNotIn("iseed = -42", content)

            # positive value is unchanged
            run_card = run_card_class()
            run_card.set('iseed', 7, user=True)
            f = io.StringIO()
            run_card.write_include_file(None, output_file=f)
            self.assertIn("iseed = 7", content := f.getvalue())
            self.assertNotIn("iseed = -7", content)

            # 2. reset_iseed_in_run_card preserves negative iseed on disk
            #    but resets a positive iseed to 0
            me_dir = tempfile.mkdtemp(prefix='amc_iseed_')
            os.mkdir(pjoin(me_dir, 'Cards'))
            try:
                # negative case: must NOT be reset to 0
                run_card = run_card_class()
                run_card.set('iseed', -42, user=True)
                run_card.write(pjoin(me_dir, 'Cards', 'run_card.dat'))

                class FakeCmd:
                    pass
                fake = FakeCmd()
                fake.run_card = run_card
                fake.me_dir = me_dir

                common_run.CommonRunCmd.reset_iseed_in_run_card(fake)
                self.assertEqual(run_card['iseed'], -42)
                # also check the on-disk value
                reloaded = bannermod.RunCard(pjoin(me_dir, 'Cards', 'run_card.dat'))
                self.assertEqual(reloaded['iseed'], -42)

                # positive case: must be reset to 0
                run_card = run_card_class()
                run_card.set('iseed', 7, user=True)
                run_card.write(pjoin(me_dir, 'Cards', 'run_card.dat'))
                fake.run_card = run_card
                common_run.CommonRunCmd.reset_iseed_in_run_card(fake)
                self.assertEqual(run_card['iseed'], 0)
                reloaded = bannermod.RunCard(pjoin(me_dir, 'Cards', 'run_card.dat'))
                self.assertEqual(reloaded['iseed'], 0)

                # zero case: nothing happens, stays at zero
                run_card = run_card_class()
                run_card.set('iseed', 0, user=True)
                run_card.write(pjoin(me_dir, 'Cards', 'run_card.dat'))
                fake.run_card = run_card
                common_run.CommonRunCmd.reset_iseed_in_run_card(fake)
                self.assertEqual(run_card['iseed'], 0)
            finally:
                import shutil
                shutil.rmtree(me_dir)


class TestRunCardMG7(unittest.TestCase):
    """Test the TOML run_card (RunCardMG7) used by the mg7/madnis mode."""

    def test_gridpack_pigz_setting(self):
        import io
        import tomllib
        for value in ("auto", True, False):
            card = bannermod.RunCardMG7()
            card.set("run.use_pigz", value)
            output = io.StringIO()
            card.write_gridpack_card(output)
            parsed = tomllib.loads(output.getvalue())
            self.assertEqual(parsed["run"]["use_pigz"], value)
            full_output = io.StringIO()
            card.write(full_output)
            self.assertEqual(
                tomllib.loads(full_output.getvalue())["run"]["use_pigz"], value)

    def test_gzip_configuration(self):
        card = bannermod.RunCardMG7()
        card.set("run.use_pigz", "auto")
        card.set("run.cpu_thread_pool_size", -1)
        self.assertEqual(card.gzip_configuration(),
                         {'use_pigz': None, 'nb_core': None})
        card.set("run.use_pigz", True)
        card.set("run.cpu_thread_pool_size", 4)
        self.assertEqual(card.gzip_configuration(),
                         {'use_pigz': True, 'nb_core': 4})
        card.set("run.use_pigz", False)
        self.assertEqual(card.gzip_configuration()['use_pigz'], False)

    template = pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files',
                     'mg7', 'run_card.toml')

    def test_runcard_dispatches_to_mg7_for_toml(self):
        """banner.RunCard(...) returns a RunCardMG7 for TOML input"""
        rc = bannermod.RunCardMG7()
        out = io.StringIO()
        rc.write(out, template=self.template)
        content = out.getvalue()
        # dispatch from string content
        self.assertIsInstance(bannermod.RunCard(content), bannermod.RunCardMG7)
        # dispatch from a .toml path
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
        tmp.write(content)
        tmp.close()
        try:
            self.assertIsInstance(bannermod.RunCard(tmp.name), bannermod.RunCardMG7)
        finally:
            os.remove(tmp.name)

    def test_mlm_jet_radius_defaults_to_madevents_d(self):
        """beam.jet_radius is madevent's hidden run_card d (common/to_dj/D),
        which is 1 there and in FxFx; hidden in the TOML card too, so it is only
        written once it differs from that, and reads back as 1 otherwise"""
        import tomllib
        legacy = bannermod.RunCardLO()
        self.assertEqual(legacy['d'], 1.0)
        rc = bannermod.RunCardMG7()
        self.assertEqual(rc['beam']['jet_radius'], legacy['d'])
        out = io.StringIO()
        rc.write(out, template=self.template)
        self.assertNotIn('jet_radius', tomllib.loads(out.getvalue())['beam'])
        self.assertEqual(
            bannermod.RunCardMG7(out.getvalue())['beam']['jet_radius'], 1.0)
        rc.set('beam.jet_radius', 0.4, user=True)
        out = io.StringIO()
        rc.write(out, template=self.template)
        self.assertEqual(tomllib.loads(out.getvalue())['beam']['jet_radius'], 0.4)

    def test_hidden_mlm_options_survive_a_rewrite(self):
        """The MLM clustering options are hidden: absent from the default card,
        and written at the end of [beam] once set, also when set without
        user=True as from_LO does, so a rewrite of the card keeps them"""
        import tomllib
        hidden = ['jet_radius', 'jet_scale_scheme', 'scale_scheme',
                  'parton_line_scheme', 'alphas_reweighting', 'pdf_reweighting',
                  'clustering_mean_pdf_ladder', 'clustering_measure',
                  'clustering_history']
        rc = bannermod.RunCardMG7()
        out = io.StringIO()
        rc.write(out, template=self.template)
        beam = tomllib.loads(out.getvalue())['beam']
        self.assertEqual([k for k in hidden if k in beam], [])
        self.assertIn('max_jet_flavor', beam)
        rc.set('beam.scale_scheme', 'madevent')
        rc.set('beam.clustering_history', 'diagram', user=True)
        out = io.StringIO()
        rc.write(out, template=self.template)
        reread = bannermod.RunCardMG7(out.getvalue())
        self.assertEqual(reread['beam']['scale_scheme'], 'madevent')
        self.assertEqual(reread['beam']['clustering_history'], 'diagram')
        # and a second rewrite keeps them
        out2 = io.StringIO()
        reread.write(out2, template=self.template)
        self.assertEqual(out2.getvalue(), out.getvalue())

    def test_numeric_suffix_like_legacy(self):
        """integer/float k/M suffixes are parsed as in the legacy RunCard"""
        rc = bannermod.RunCardMG7()
        rc.set('generation.events', '10k', user=True)
        self.assertEqual(rc['generation']['events'], 10000)
        rc.set('generation.events', '2M', user=True)
        self.assertEqual(rc['generation']['events'], 2000000)

    def test_energy_units(self):
        """energy units are converted to GeV for energy parameters"""
        rc = bannermod.RunCardMG7()
        rc.set('beam.e_cm', '13 TeV', user=True)
        self.assertEqual(rc['beam']['e_cm'], 13000.0)
        rc.set('beam.ren_scale', '500GeV', user=True)
        self.assertEqual(rc['beam']['ren_scale'], 500.0)
        # parse_energy leaves unit-less / non-energy strings untouched
        self.assertEqual(rc.parse_energy('30'), '30')
        self.assertEqual(rc.parse_energy('1 TeV'), 1000.0)
        self.assertAlmostEqual(rc.parse_energy('500 MeV'), 0.5)

    def test_set_cut_with_units(self):
        """cut bounds are settable and converted to GeV; dimensionless ok"""
        rc = bannermod.RunCardMG7()
        rc.set_cut('jet-pt.min', '1 TeV')
        self.assertEqual(rc['cuts']['jet-pt'], {'min': 1000.0})
        rc.set_cut('lepton-pt.min', '500 MeV')
        self.assertAlmostEqual(rc['cuts']['lepton-pt']['min'], 0.5)
        rc.set_cut('jet-eta_abs.max', '4')   # dimensionless, no unit
        self.assertEqual(rc['cuts']['jet-eta_abs'], {'max': 4.0})
        self.assertTrue(rc.is_cut_name('jet-pt.min'))
        self.assertTrue(rc.is_cut_name('cuts.jet-pt.max'))
        self.assertFalse(rc.is_cut_name('generation.events'))

    def test_set_cut_rejects_an_unknown_bound(self):
        """a typo in the bound used to create a cut named 'jet-pt.mni'"""
        rc = bannermod.RunCardMG7()
        before = dict(rc['cuts'])
        self.assertRaises(bannermod.InvalidRunCard, rc.set_cut, 'jet-pt.mni', 20)
        self.assertRaises(bannermod.InvalidRunCard, rc.set_cut, 'cuts.jet-pt.minimum', 20)
        self.assertEqual(dict(rc['cuts']), before)
        rc.set_cut('jet-pt', 25)      # no bound: the minimum
        self.assertEqual(rc['cuts']['jet-pt']['min'], 25.0)

    def test_custom_parameter_keeps_its_case(self):
        """a custom function reads its parameters by their exact name"""
        rc = bannermod.RunCardMG7()
        rc.read_data({'custom_functions': {'cut_parameters': {'ptMin': 10}}})
        rc.set('custom_functions.cut_parameters.ptMin', 30, user=True)
        rc.set('Custom_Functions.CUT_parameters.etaMax', '2.5', user=True)
        self.assertEqual(rc.custom_parameters('cut'), {'ptMin': 30.0, 'etaMax': 2.5})

    def test_energy_units_only_for_energies(self):
        """only the entries declared energy=True (and the [cuts] on an energy
        observable) take a value with a unit; any other number refuses one"""
        lo = bannermod.RunCardLO()
        lo.set('ptj', '1 TeV', user=True)
        self.assertEqual(lo['ptj'], 1000.)
        lo['pt_min_pdg'] = {6: '500 GeV'}
        self.assertEqual(lo['pt_min_pdg'], {6: 500.})
        for name, value in (('etaj', '2 TeV'), ('nevents', '1 TeV'), ('drjj', '1 MeV')):
            self.assertRaises(bannermod.InvalidCardEdition, lo.set, name, value, user=True)
        self.assertRaises(bannermod.InvalidCardEdition, lo.__setitem__,
                          'eta_min_pdg', {6: '1 TeV'})
        lo.set('run_tag', '13TeV', user=True)   # a string keeps its unit
        self.assertEqual(lo['run_tag'], '13TeV')
        nlo = bannermod.RunCardNLO()
        nlo.set('mur_ref_fixed', '0.1 TeV', user=True)
        self.assertEqual(nlo['mur_ref_fixed'], 100.)
        self.assertRaises(bannermod.InvalidCardEdition, nlo.set, 'jetradius', '1 GeV')

        rc = bannermod.RunCardMG7()
        rc.set('beam.min_scale', '1 TeV', user=True)
        self.assertEqual(rc['beam.min_scale'], 1000.)
        self.assertRaises(bannermod.InvalidCardEdition, rc.set, 'beam.jet_radius', '1 GeV')
        self.assertRaises(bannermod.InvalidCardEdition, rc.set, 'events', '1 TeV')
        before = dict(rc['cuts'])
        for name, value in (('jet-eta_abs.max', '4 TeV'), ('jet-lepton-delta_r.min', '1 MeV')):
            self.assertRaises(bannermod.InvalidRunCard, rc.set_cut, name, value)
        for name, value in (('etaj', '2 TeV'), ('drjj', '1 MeV')):
            self.assertRaises(bannermod.InvalidRunCard, rc.set_lo_cut, name, value)
        self.assertRaises(bannermod.InvalidRunCard, rc.set_lo_param,
                          'eta_max_pdg', '{6: 2 TeV}')
        self.assertEqual(dict(rc['cuts']), before)
        self.assertEqual(rc.set_cut('sqrt_s.min', '2*mz', {'mz': 91.188})[2], 182.376)
        self.assertEqual(rc.set_lo_cut('ptj', '1 TeV')[2], 1000.)

    def test_set_ambiguous_key_honours_raiseerror(self):
        """a bare key present in several sections is an error when asked for"""
        rc = bannermod.RunCardMG7()
        rc.set('enable', True, user=True)          # warning only
        self.assertRaises(bannermod.InvalidCardEdition, rc.set, 'enable', True,
                          user=True, raiseerror=True)
        self.assertRaises(bannermod.InvalidCardEdition, rc.set, 'pdf', 'NNPDF',
                          user=True, raiseerror=True)

    def test_set_lo_cut(self):
        """madevent cut names land in [cuts] as from_LO maps them, the "no
        cut" sentinels removing the bound; [histograms] is left alone"""
        rc = bannermod.RunCardMG7()
        rc.dynamic_sections['histograms']['sqrt_s'] = \
            {'min': 0., 'max': 2000., 'bin_count': 50}
        self.assertEqual(rc.set_lo_cut('mmll', '200'),
                         ('lepton-sfos_pair_mass', 'min', 200.0))
        self.assertEqual(rc['cuts']['lepton-sfos_pair_mass'], {'min': 200.0})
        rc.set_lo_cut('MMLLMAX', '1 TeV')
        self.assertEqual(rc['cuts']['lepton-sfos_pair_mass'],
                         {'min': 200.0, 'max': 1000.0})
        # minimum 0 / maximum -1 mean "no cut"
        self.assertEqual(rc.set_lo_cut('mmll', 0),
                         ('lepton-sfos_pair_mass', 'min', None))
        self.assertEqual(rc['cuts']['lepton-sfos_pair_mass'], {'max': 1000.0})
        rc.set_lo_cut('mmllmax', -1)
        self.assertNotIn('lepton-sfos_pair_mass', rc['cuts'])
        # a maximum of 0 is a real cut, as in from_LO
        rc.set_lo_cut('ptjmax', 0)
        self.assertEqual(rc['cuts']['jet-pt'], {'min': 20.0, 'max': 0.0})
        rc.set_lo_cut('etaj', -1)
        self.assertNotIn('jet-eta_abs', rc['cuts'])
        # sqrt_s is a cut and a histogram under the same key
        rc.set_lo_cut('dsqrt_shat', '500')
        self.assertEqual(rc['cuts']['sqrt_s'], {'min': 500.0})
        self.assertEqual(rc['histograms']['sqrt_s'],
                         {'min': 0., 'max': 2000., 'bin_count': 50})
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_cut, 'ptj', 'abc')

    def test_set_lo_cut_composite(self):
        """the ordered/summed cuts from_LO converts are settable too, and
        bring the group they are written on when the card lacks it"""
        rc = bannermod.RunCardMG7()
        multiparticles = rc['multiparticles']
        self.assertNotIn('parton', multiparticles)
        self.assertNotIn('alllepton', multiparticles)
        rc.set_lo_cut('ptj1min', 50)
        rc.set_lo_cut('ht3min', 200)
        self.assertEqual(rc['cuts']['jet_1-pt'], {'min': 50.0})
        self.assertEqual(rc['cuts']['jet_1-jet_2-jet_3-pt-sum'], {'min': 200.0})
        self.assertNotIn('parton', multiparticles)
        rc.set_lo_cut('ihtmin', 100)
        self.assertEqual(rc['cuts']['parton-pt-sum'], {'min': 100.0})
        self.assertEqual(sorted(multiparticles['parton']),
                         [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21])
        rc.set_lo_cut('ptllmin', 30)
        self.assertEqual(rc['cuts']['alllepton-sum-pt'], {'min': 30.0})
        self.assertEqual(list(multiparticles['alllepton']),
                         list(multiparticles['lepton']) + list(multiparticles['missing']))
        # a group the user defined is kept
        multiparticles['alllepton'] = [11, -11]
        rc.set_lo_cut('mmnl', 80)
        self.assertEqual(multiparticles['alllepton'], [11, -11])
        # "no cut" removes the cut, not the group
        rc.set_lo_cut('ihtmin', 0)
        self.assertNotIn('parton-pt-sum', rc['cuts'])
        self.assertIn('parton', multiparticles)

    def test_set_lo_param_value_translation(self):
        """madevent codes become the mg7 names; mg7 values are left alone"""
        rc = bannermod.RunCardMG7()
        self.assertEqual(rc.set_lo_param('dynamical_scale_choice', '1'),
                         ({'beam.dynamical_scale_choice': 'transverse_energy'}, []))
        rc.set_lo_param('dynamical_scale_choice', 'shat')
        self.assertEqual(rc['beam.dynamical_scale_choice'], 'partonic_energy')
        # no mg7 equivalent: a warning, the value is kept
        changes, warnings = rc.set_lo_param('dynamical_scale_choice', '0')
        self.assertEqual(changes, {})
        self.assertIn('user-defined', warnings[0])
        self.assertEqual(rc['beam.dynamical_scale_choice'], 'partonic_energy')
        self.assertIsNone(rc.set_lo_param('dynamical_scale_choice', 'transverse_mass'))
        self.assertIsNone(rc.set_lo_param('dynamical_scale_choice', 'HT/4'))
        rc.set_lo_param('sde_strategy', '2')
        self.assertEqual(rc['phasespace.sde_strategy'], 'denominators')
        rc.set_lo_param('sde_strategy', '1')
        self.assertEqual(rc['phasespace.sde_strategy'], 'diagrams')
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'sde_strategy', '3')
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param,
                          'dynamical_scale_choice', '99')
        self.assertEqual(rc['beam.dynamical_scale_choice'], 'partonic_energy')
        self.assertIsNone(rc.set_lo_param('sde_strategy', 'denominators'))
        rc.set_lo_param('maxjetflavor', '5')
        self.assertEqual(sorted(rc['multiparticles']['jet']),
                         [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21])
        rc.set_lo_param('fixed_fac_scale', 'T')
        self.assertIs(rc['beam.fixed_fact_scale'], True)
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'maxjetflavor', '9')
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'fixed_fac_scale', 'abc')
        # the beam settings are set_lo_beam_param's
        for name in rc.lo_beam_params:
            self.assertNotIn(name, rc._LO_SET_PARAMS)
        self.assertEqual(rc.lo_set_warnings(), [])

    def test_set_lo_param_fixed_fac_scale_per_beam(self):
        """fixed_fac_scale1/2 combine within one question; different values,
        which the one mg7 switch cannot hold, are reported once, at the end"""
        rc = bannermod.RunCardMG7()
        rc.set_lo_param('fixed_fac_scale1', 'T')
        self.assertIs(rc['beam.fixed_fact_scale'], True)
        self.assertEqual(rc.set_lo_param('fixed_fac_scale2', 'T')[1], [])
        self.assertEqual(rc.lo_set_warnings(), [])
        rc.set_lo_param('fixed_fac_scale1', 'F')
        self.assertIs(rc['beam.fixed_fact_scale'], False)
        self.assertIn('fixed_fac_scale1 = False and fixed_fac_scale2 = True',
                      rc.lo_set_warnings()[0])
        self.assertEqual(rc.lo_set_warnings(), [])    # said once

    def test_set_lo_param_pdg_cuts(self):
        """{pdg: value} cuts as written in a run_card.dat"""
        rc = bannermod.RunCardMG7()
        rc.set_lo_param('pt_min_pdg', '{6: 100, 25: 50}')
        self.assertEqual(rc['cuts']['pdg6-pt'], {'min': 100.0})
        self.assertEqual(rc['cuts']['pdg25-pt'], {'min': 50.0})
        self.assertEqual(rc['multiparticles']['pdg6'], [6, -6])
        rc.set_lo_param('eta_max_pdg', '{6: 2.5}')
        self.assertEqual(rc['cuts']['pdg6-eta_abs'], {'max': 2.5})
        changes, _ = rc.set_lo_param('pt_min_pdg', '{6: 0}')
        self.assertEqual(changes, {'cuts.pdg6-pt': None})
        self.assertNotIn('pdg6-pt', rc['cuts'])
        # X X~ pairs only, whichever of the two is set first
        rc.set_lo_param('mxx_only_part_antipart', '{6: True}')
        rc.set_lo_param('mxx_min_pdg', '{6: 500}')
        self.assertEqual(rc['cuts']['pdg6-sfos_pair_mass'], {'min': 500.0})
        rc.set_lo_param('mxx_only_part_antipart', "{'default': False, 6: False}")
        self.assertNotIn('pdg6-sfos_pair_mass', rc['cuts'])
        self.assertEqual(rc['cuts']['pdg6-pair_mass'], {'min': 500.0})
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'pt_min_pdg', '{6: abc}')
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'pt_min_pdg', '100')

    def test_set_lo_param_pdg_cut_forms(self):
        """the {pdg: value} parameters take the forms the run_card.dat editor
        takes, a single entry updating only that pdg"""
        rc = bannermod.RunCardMG7()
        rc.set_lo_param('pt_min_pdg', '6 100')
        rc.set_lo_param('pt_min_pdg', '25:50')
        rc.set_lo_param('pt_min_pdg', "{'23': 30}")
        self.assertEqual(rc['cuts']['pdg6-pt'], {'min': 100.0})
        self.assertEqual(rc['cuts']['pdg25-pt'], {'min': 50.0})
        self.assertEqual(rc['cuts']['pdg23-pt'], {'min': 30.0})
        self.assertRaises(bannermod.InvalidCmd, rc.set_lo_param, 'pt_min_pdg', '{top: 100}')

    def test_set_lo_param_pair_mass_follows_the_card(self):
        """mxx_min_pdg updates the pair-mass cut the card already has, and
        mg7 never ends up with two pair-mass minima on one pdg"""
        rc = bannermod.RunCardMG7()
        cuts = rc.dynamic_sections['cuts']
        # as converted from a run_card.dat with mxx_only_part_antipart {6: True}
        cuts['pdg6-sfos_pair_mass'] = {'min': 250.0}
        rc.set_lo_param('mxx_min_pdg', '{6: 600}')
        self.assertEqual(cuts['pdg6-sfos_pair_mass'], {'min': 600.0})
        self.assertNotIn('pdg6-pair_mass', cuts)
        # an explicit choice wins, and the other kind of pair loses its minimum
        rc.set_lo_param('mxx_only_part_antipart', '{6: False}')
        self.assertEqual(cuts['pdg6-pair_mass'], {'min': 600.0})
        self.assertNotIn('pdg6-sfos_pair_mass', cuts)
        cuts['pdg6-sfos_pair_mass'] = {'min': 100.0, 'max': 900.0}
        rc.set_lo_param('mxx_min_pdg', '{6: 700}')
        self.assertEqual(cuts['pdg6-pair_mass'], {'min': 700.0})
        self.assertEqual(cuts['pdg6-sfos_pair_mass'], {'max': 900.0})
        # renaming onto an existing cut merges the bounds, the tighter wins
        rc.set_lo_param('mxx_only_part_antipart', '{6: True}')
        self.assertEqual(cuts['pdg6-sfos_pair_mass'], {'max': 900.0, 'min': 700.0})
        self.assertNotIn('pdg6-pair_mass', cuts)
        # the choice is kept with the card, not with the question
        rc.lo_set_warnings()
        rc.set_lo_param('mxx_min_pdg', '{25: 300}')
        self.assertNotIn('pdg25-sfos_pair_mass', cuts)
        rc.set_lo_param('mxx_only_part_antipart', '{25: True}')
        rc.lo_set_warnings()
        rc.set_lo_param('mxx_min_pdg', '{25: 400}')
        self.assertEqual(cuts['pdg25-sfos_pair_mass'], {'min': 400.0})
        self.assertNotIn('pdg25-pair_mass', cuts)

    def test_set_lo_param_matches_from_LO(self):
        """"set" lands where the run_card.dat conversion puts the same values"""
        lo = bannermod.RunCardLO()
        values = {'dynamical_scale_choice': 1, 'sde_strategy': 2,
                  'maxjetflavor': 5, 'fixed_fac_scale': True}
        for name, value in values.items():
            lo.set(name, value, user=True)
        converted, _ = bannermod.RunCardMG7.from_LO(lo, warn=False)
        rc = bannermod.RunCardMG7()
        for name, value in values.items():
            self.assertIsNotNone(rc.set_lo_param(name, str(value)), name)
        for key in ('beam.dynamical_scale_choice', 'beam.fixed_fact_scale',
                    'phasespace.sde_strategy'):
            self.assertEqual(rc[key], converted[key], key)
        self.assertEqual(rc['multiparticles']['jet'], converted['multiparticles']['jet'])
        self.assertEqual(rc.lo_set_warnings(), [])

    def test_evaluate_math_and_masses(self):
        """values support arithmetic and mass references"""
        rc = bannermod.RunCardMG7()
        masses = {'mz': 91.0, 'mh': 125.0}
        self.assertEqual(rc.evaluate('91.0*2', masses), 182.0)
        self.assertEqual(rc.evaluate('mz/2', masses), 45.5)
        self.assertEqual(rc.evaluate('2*mh', masses), 250.0)
        self.assertEqual(rc.evaluate('mz+9', masses), 100.0)
        self.assertEqual(rc.evaluate('1 TeV', masses), 1000.0)
        # non-evaluable values are returned unchanged (k/M, strings, bools)
        self.assertEqual(rc.evaluate('10k', masses), '10k')
        self.assertEqual(rc.evaluate('default', masses), 'default')
        self.assertEqual(rc.evaluate('True', masses), 'True')

    def test_collider_shortcuts(self):
        """lhc/lep shortcuts set e_cm (GeV) and leptonic"""
        rc = bannermod.RunCardMG7()
        rc.set_collider('lhc', '13')            # unit-less -> TeV
        self.assertEqual(rc['beam']['e_cm'], 13000.0)
        self.assertIs(rc['beam']['leptonic'], False)
        rc.set_collider('lhc', '13 TeV')        # explicit unit
        self.assertEqual(rc['beam']['e_cm'], 13000.0)
        rc.set_collider('lep', '250')           # unit-less -> GeV
        self.assertEqual(rc['beam']['e_cm'], 250.0)
        self.assertIs(rc['beam']['leptonic'], True)

    def test_per_beam_energy_and_pdf(self):
        """ebeam1/ebeam2 and pdf1/pdf2 are per beam; e_cm and pdf are derived
        from them and setting either sets both beams"""
        rc = bannermod.RunCardMG7()
        rc.set('beam.ebeam1', '7 TeV', user=True)
        rc.set('beam.ebeam2', 4000, user=True)
        self.assertEqual(rc['beam']['ebeam1'], 7000.0)
        self.assertAlmostEqual(rc['beam']['e_cm'], 2 * (7000.0 * 4000.0) ** 0.5)
        self.assertEqual(rc['ebeam2'], 4000.0)  # legacy run_card.dat name
        rc.set('e_cm', '14 TeV', user=True)
        self.assertEqual((rc['beam']['ebeam1'], rc['beam']['ebeam2']), (7000.0, 7000.0))
        rc.set('beam.pdf', 'CT18LO', user=True)
        self.assertEqual((rc['beam']['pdf1'], rc['beam']['pdf2']), ('CT18LO', 'CT18LO'))
        self.assertEqual(rc['beam']['pdf'], 'CT18LO')
        rc.set('beam.pdf2', 'nNNPDF30_nlo_as_0118_A208_Z82', user=True)
        self.assertRaises(KeyError, lambda: rc['beam']['pdf'])
        # a bare "pdf" could be the beam or the systematics one: left untouched
        rc.set('pdf', 'NNPDF31_lo_as_0118', user=True)
        self.assertEqual(rc['beam']['pdf1'], 'CT18LO')
        self.assertEqual(rc['systematics']['pdf'], ['errorset'])
        with self.assertRaises(bannermod.InvalidRunCard):
            rc.set('beam.e_cm', 'scan:[7000, 13000]', user=True)
        rc.set('beam.ebeam2', 0, user=True)
        self.assertRaises(bannermod.InvalidRunCard, rc.check_validity)

    def test_set_lo_beam_param(self):
        """madevent beam settings (set ebeam/lpp/pdlabel/lhaid at the launch
        question) go to the per-beam entries"""
        rc = bannermod.RunCardMG7()
        beams = lambda key: (rc['beam'][key + '1'], rc['beam'][key + '2'])
        rc.set_lo_beam_param('ebeam', '6.8 TeV')
        self.assertEqual(beams('ebeam'), (6800.0, 6800.0))
        rc.set_lo_beam_param('ebeam2', 'mz*40', masses={'mz': 91.188})
        self.assertEqual(beams('ebeam'), (6800.0, 3647.52))
        rc.set_lo_beam_param('pdlabel', 'nn23lo1')
        self.assertEqual(beams('pdf'), ('NNPDF23_lo_as_0130_qed',) * 2)
        rc.set_lo_beam_param('pdlabel1', 'cteq6l1')
        self.assertEqual(beams('pdf'), ('cteq6l1', 'NNPDF23_lo_as_0130_qed'))
        # the set is chosen by lhaid: 'lhapdf' alone changes nothing
        rc.set_lo_beam_param('pdlabel', 'lhapdf')
        self.assertEqual(beams('pdf'), ('cteq6l1', 'NNPDF23_lo_as_0130_qed'))
        # an id outside the built-in table, through the LHAPDF index
        index_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(index_dir, 'pdfsets.index'), 'w') as fsock:
                fsock.write('244800 NNPDF23_nlo_as_0119_qed 5\n')
            rc.set_lo_beam_param('lhaid', '244800', data_paths=[index_dir])
        finally:
            shutil.rmtree(index_dir)
        self.assertEqual(beams('pdf'), ('NNPDF23_nlo_as_0119_qed',) * 2)
        for name, value in (('lhaid', '999999'), ('pdlabel2', 'isronlyll'),
                            ('pdlabel2', 'none'), ('lpp1', '2'), ('lpp', 'p')):
            self.assertRaises(bannermod.InvalidRunCard,
                              rc.set_lo_beam_param, name, value)
        self.assertEqual(beams('pdf'), ('NNPDF23_nlo_as_0119_qed',) * 2)
        # one lepton/hadron switch for both beams
        rc.set_lo_beam_param('lpp', '0')
        self.assertTrue(rc['beam']['leptonic'])
        rc.set_lo_beam_param('lpp1', '1')
        self.assertFalse(rc['beam']['leptonic'])
        rc.set_lo_beam_param('pdlabel', 'none')
        self.assertTrue(rc['beam']['leptonic'])

    def test_read_card_with_e_cm_and_pdf(self):
        """a card written before the per-beam keys still reads"""
        rc = bannermod.RunCardMG7()
        out = io.StringIO()
        rc.write(out, template=self.template)
        text = out.getvalue()
        text = text.replace('ebeam1 = 6500.0\nebeam2 = 6500.0', 'e_cm = 1000.0')
        text = re.sub(r'pdf1 = .*\npdf2 = .*', 'pdf = "NNPDF23_lo_as_0130_qed"', text)
        self.assertNotIn('ebeam1 =', text)
        self.assertNotIn('pdf1 =', text)
        rc2 = bannermod.RunCardMG7(text)
        self.assertEqual((rc2['beam']['ebeam1'], rc2['beam']['ebeam2']), (500.0, 500.0))
        self.assertEqual(rc2['beam']['pdf2'], 'NNPDF23_lo_as_0130_qed')
        out = io.StringIO()
        rc2.write(out, template=self.template)
        self.assertIn('ebeam2 = 500.0', out.getvalue())
        self.assertNotIn('e_cm =', out.getvalue())

    def test_fixed_scale_and_remove_cuts(self):
        """fixed_scale sets all scales; remove_all_cut clears cuts"""
        rc = bannermod.RunCardMG7()
        rc.set_fixed_scale('mz', {'mz': 91.0})
        self.assertIs(rc['beam']['fixed_ren_scale'], True)
        self.assertIs(rc['beam']['fixed_fact_scale'], True)
        self.assertEqual(rc['beam']['ren_scale'], 91.0)
        self.assertEqual(rc['beam']['fact_scale1'], 91.0)
        self.assertEqual(rc['beam']['fact_scale2'], 91.0)
        self.assertTrue(rc['cuts'])
        rc.remove_all_cut()
        self.assertEqual(dict(rc['cuts']), {})

    def test_run_card_scan_expansion(self):
        """scan:[...] / correlated scanN:[...] values expand via RunCardIterator"""
        rc = bannermod.RunCardMG7()
        out = io.StringIO()
        rc.write(out, template=self.template)
        text = out.getvalue()
        # the collision energy is scanned through both beams, moving together
        text = text.replace('ebeam1 = 6500.0', 'ebeam1 = "scan2:[3500, 6500]"')
        text = text.replace('ebeam2 = 6500.0', 'ebeam2 = "scan2:[3500, 6500]"')
        text = text.replace('ren_scale = 91.188', 'ren_scale = "scan1:[91.0, 172.0]"')
        text = text.replace('fact_scale1 = 91.188', 'fact_scale1 = "scan1:[45.5, 86.0]"')
        # valid TOML, and detected as a scan
        import tomllib
        tomllib.loads(text)
        it = bannermod.RunCardIterator(text)
        points = [(c['beam']['e_cm'], c['beam']['ren_scale'], c['beam']['fact_scale1'])
                  for c in it]
        # e_cm (2 values) x correlated scan-1 group (2 values) = 4 points
        self.assertEqual(len(points), 4)
        for e_cm, ren, fac in points:
            self.assertIn(e_cm, (7000.0, 13000.0))
            # ren_scale and fact_scale1 share scan-id 1 -> always move together
            self.assertIn((ren, fac), [(91.0, 45.5), (172.0, 86.0)])

    def test_run_card_scan_summary_json(self):
        """RunCardIterator.write_summary also writes a json summary"""
        tmpdir = tempfile.mkdtemp()
        try:
            events = pjoin(tmpdir, 'Events')
            for run in ['run_01', 'run_02']:
                os.makedirs(pjoin(events, run))
            it = bannermod.RunCardIterator.__new__(bannermod.RunCardIterator)
            it.param_order = ['run_card#ebeam1']
            it.cross = [
                {'run_name': 'run_01', 'bench': [6500.], 'cross(pb)': 2.0, 'error(pb)': 0.1},
                {'run_name': 'run_02', 'bench': [7000.], 'exception': ValueError('boom')},
            ]
            path = pjoin(events, 'scan_run_01.txt')
            it.write_summary(path)
            with open(pjoin(events, 'scan_run_01.json')) as fsock:
                data = json.load(fsock)
            self.assertEqual(data['scan_parameters'], [{'id': 'run_card#ebeam1'}])
            self.assertEqual(data['points'][0]['parameters'], {'run_card#ebeam1': 6500.})
            self.assertEqual(data['points'][0]['results'],
                             {'cross(pb)': 2.0, 'error(pb)': 0.1})
            # a point that crashed keeps its error message and has no result
            self.assertEqual(data['points'][1]['exception'], 'boom')
            self.assertEqual(data['points'][1]['results'],
                             {'cross(pb)': None, 'error(pb)': None})
        finally:
            shutil.rmtree(tmpdir)

    def test_from_LO_conversion(self):
        """RunCardMG7.from_LO ports the supported LO settings and reports the rest"""
        lo = bannermod.RunCardLO()
        lo['ebeam1'] = 6500
        lo['ebeam2'] = 6500
        lo['nevents'] = 25000
        lo['ptj'] = 30
        lo['etaj'] = 4.5
        lo['mmjj'] = 200
        lo['mmll'] = 50
        lo['cut_decays'] = True
        lo['dynamical_scale_choice'] = 3
        lo['scalefact'] = 0.5
        lo['SDE_strategy'] = 2
        lo['maxjetflavor'] = 5
        lo['xqcut'] = 20          # merging -> not supported
        lo['polbeam1'] = 80       # polarization -> not supported
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        # ported
        self.assertEqual(mg7['beam']['e_cm'], 13000.0)
        self.assertEqual(mg7['beam']['ebeam1'], 6500.0)
        self.assertEqual(mg7['generation']['events'], 25000)
        self.assertEqual(mg7['beam']['dynamical_scale_choice'], 'half_transverse_mass')
        self.assertEqual(mg7['beam']['scale_factor'], 0.5)
        self.assertEqual(mg7['phasespace']['sde_strategy'], 'denominators')
        self.assertIn(5, mg7['multiparticles']['jet'])
        self.assertEqual(mg7['cuts']['jet-pt'], {'min': 30.0})
        self.assertEqual(mg7['cuts']['jet-eta_abs'], {'max': 4.5})
        # a jet-pair mass, not the mass of each jet
        self.assertEqual(mg7['cuts']['jet-pair_mass'], {'min': 200.0})
        self.assertNotIn('jet-mass', mg7['cuts'])
        # same-flavour opposite-sign lepton pairs only, as LO mmll
        self.assertEqual(mg7['cuts']['lepton-sfos_pair_mass'], {'min': 50.0})
        self.assertTrue(mg7['phasespace']['cut_decays'])
        # MadEvent's default: decay products are not cut
        self.assertFalse(bannermod.RunCardMG7()['phasespace']['cut_decays'])
        # reported as not transferable
        joined = ' '.join(dropped)
        self.assertIn('xqcut', joined)
        self.assertIn('polbeam1', joined)
        # the produced card is valid TOML
        import io, tomllib
        buf = io.StringIO()
        mg7.write(buf, template=self.template)
        tomllib.loads(buf.getvalue())

    def test_legacy_keys_report_mlm_merging(self):
        """The madevent shower interface picks MLM jet matching from ickkw
        and sets JetMatching:qCut from xqcut and nQmatch from maxjetflavor.
        An mg7 MLM sample has to report them, or Pythia8 showers it with no
        matching and double counts the jets."""
        card = bannermod.RunCardMG7()
        # not merged: the plain shower path, as before
        self.assertEqual(card['ickkw'], 0)
        self.assertEqual(card['xqcut'], 0.0)
        self.assertEqual(card['maxjetflavor'], 5)
        card.set('beam.dynamical_scale_choice', 'mlm')
        card.set('phasespace.xqcut', 30.0)
        card.set('beam.max_jet_flavor', 4)
        self.assertTrue(card.is_mlm_merged())
        self.assertEqual(card['ickkw'], 1)
        self.assertEqual(card['xqcut'], 30.0)
        self.assertEqual(card['maxjetflavor'], 4)
        self.assertEqual(card['ktdurham'], -1.0)
        # no merging cut, or no beams to merge on: not a merged sample
        card.set('phasespace.xqcut', 0.0)
        self.assertEqual(card['ickkw'], 0)
        card.set('phasespace.xqcut', 30.0)
        card.set('beam.leptonic', True)
        self.assertEqual(card['ickkw'], 0)
        self.assertEqual(card['xqcut'], 0.0)

    def test_set_maxjetflavor_sets_the_mlm_jet_flavours(self):
        rc = bannermod.RunCardMG7()
        rc.set_lo_param('maxjetflavor', '5')
        self.assertEqual(rc['beam']['max_jet_flavor'], 5)

    def test_from_LO_conversion_of_mlm_merging(self):
        """ickkw = 1 with xqcut is madevent's MLM: mg7's "mlm" scale choice
        with madevent's scale scheme, the same xqcut and jet flavours"""
        lo = bannermod.RunCardLO()
        lo['ickkw'] = 1
        lo['xqcut'] = 20
        lo['maxjetflavor'] = 5
        lo['auto_ptj_mjj'] = False
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertEqual(mg7['beam']['dynamical_scale_choice'], 'mlm')
        self.assertEqual(mg7['beam']['scale_scheme'], 'madevent')
        self.assertEqual(mg7['phasespace']['xqcut'], 20.0)
        self.assertFalse(mg7['phasespace']['auto_ptj_mjj'])
        self.assertEqual(mg7['beam']['max_jet_flavor'], 5)
        joined = ' '.join(dropped)
        self.assertNotIn('ickkw', joined)
        self.assertNotIn('xqcut', joined)
        self.assertEqual(mg7['ickkw'], 1)
        # without a merging cut there is nothing to merge
        lo['xqcut'] = 0
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertNotEqual(mg7['beam']['dynamical_scale_choice'], 'mlm')
        self.assertIn('ickkw=1', ' '.join(dropped))

    def test_from_LO_conversion_per_beam(self):
        """asymmetric beam energies and PDFs are kept per beam"""
        lo = bannermod.RunCardLO()
        lo['ebeam1'] = 7000
        lo['ebeam2'] = 4000
        lo['pdlabel1'] = 'nn23lo1'
        lo['pdlabel2'] = 'cteq6l1'
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertEqual((mg7['beam']['ebeam1'], mg7['beam']['ebeam2']), (7000.0, 4000.0))
        self.assertAlmostEqual(mg7['beam']['e_cm'], 2 * (7000.0 * 4000.0) ** 0.5)
        self.assertEqual((mg7['beam']['pdf1'], mg7['beam']['pdf2']),
                         ('NNPDF23_lo_as_0130_qed', 'cteq6l1'))
        self.assertNotIn('pdlabel', ' '.join(dropped))

    def test_from_LO_conversion_of_composite_cuts(self):
        """ordered, HT, summed-momentum, energy and per-pdg LO cuts are ported"""
        lo = bannermod.RunCardLO()
        lo['ptj1min'] = 100
        lo['ptj2min'] = 50
        lo['xptj'] = 120          # lands on the same bound as ptj1min: tighter wins
        lo['htjmin'] = 300
        lo['ihtmin'] = 400
        lo['ht2min'] = 150
        lo['ptllmin'] = 30
        lo['mmnl'] = 60
        lo['mmnlmax'] = 500
        lo['etajmin'] = 1.0
        lo['ej'] = 50
        lo['pt_min_pdg'] = {6: 100}
        lo['eta_max_pdg'] = {6: 2.5}
        lo['mxx_min_pdg'] = {6: 250, 25: 100}
        lo['mxx_only_part_antipart'] = {'default': False, 6: True}
        lo['cutuse'] = 1          # not representable
        lo['xetamin'] = 2         # not representable
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        cuts = mg7['cuts']
        self.assertEqual(cuts['jet_1-pt'], {'min': 120.0})
        self.assertEqual(cuts['jet_2-pt'], {'min': 50.0})
        self.assertEqual(cuts['jet-pt-sum'], {'min': 300.0})
        self.assertEqual(cuts['parton-pt-sum'], {'min': 400.0})
        self.assertEqual(cuts['jet_1-jet_2-pt-sum'], {'min': 150.0})
        self.assertEqual(cuts['alllepton-sum-pt'], {'min': 30.0})
        self.assertEqual(cuts['alllepton-sum-mass'], {'min': 60.0, 'max': 500.0})
        self.assertEqual(cuts['jet-eta_abs'], {'max': 5.0, 'min': 1.0})
        self.assertEqual(cuts['jet-e'], {'min': 50.0})
        # one group [pdg, -pdg] per particle; X X~ pairs only where asked for
        self.assertEqual(cuts['pdg6-pt'], {'min': 100.0})
        self.assertEqual(cuts['pdg6-eta_abs'], {'max': 2.5})
        self.assertEqual(cuts['pdg6-sfos_pair_mass'], {'min': 250.0})
        self.assertEqual(cuts['pdg25-pair_mass'], {'min': 100.0})
        groups = mg7['multiparticles']
        self.assertEqual(groups['pdg6'], [6, -6])
        self.assertEqual(groups['alllepton'], groups['lepton'] + groups['missing'])
        self.assertIn(5, groups['parton'])
        joined = ' '.join(dropped)
        self.assertIn('cutuse', joined)
        self.assertIn('xetamin', joined)
        self.assertNotIn('ptj1min', joined)
        # no cut means no extra group
        plain, _ = bannermod.RunCardMG7.from_LO(bannermod.RunCardLO(), warn=False)
        self.assertNotIn('alllepton', plain['multiparticles'])
        self.assertNotIn('parton', plain['multiparticles'])

    def test_from_LO_pdg_cuts_read_from_a_run_card(self):
        """the per-pdg cuts of a run_card.dat file, which is read with string
        pdg keys ({'6': 100.0}), are converted too"""
        lo = bannermod.RunCardLO()
        lo.set('pt_min_pdg', '{6: 100}', user=True)
        lo.set('mxx_min_pdg', '{6: 250, 25: 100}', user=True)
        lo.set('mxx_only_part_antipart', '{6: True}', user=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'run_card.dat')
            lo.write(path)
            lo = bannermod.RunCardLO(path)
        self.assertEqual(lo['pt_min_pdg'], {'6': 100.0})
        mg7, _ = bannermod.RunCardMG7.from_LO(lo, warn=False)
        cuts = mg7['cuts']
        self.assertEqual(cuts['pdg6-pt'], {'min': 100.0})
        self.assertEqual(cuts['pdg6-sfos_pair_mass'], {'min': 250.0})
        self.assertEqual(cuts['pdg25-pair_mass'], {'min': 100.0})
        self.assertEqual(mg7['multiparticles']['pdg25'], [25, -25])

    def test_from_LO_me_frame(self):
        """me_frame is carried over, the LO default included: madevent's [1,2]
        is the partonic c.m., mg7's own default [] the lab frame"""
        lo = bannermod.RunCardLO()
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertEqual(mg7['run']['me_frame'], [1, 2])
        lo.set('me_frame', [3, 4], user=True)
        lo.update_system_parameter_for_include()     # frame_id 24
        mg7, dropped = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertEqual(mg7['run']['me_frame'], [3, 4])
        self.assertNotIn('frame', ' '.join(dropped))
        # no leg selected: madevent stays in the partonic c.m.
        lo.set('me_frame', [], user=True)
        mg7, _ = bannermod.RunCardMG7.from_LO(lo, warn=False)
        self.assertEqual(mg7['run']['me_frame'], [1, 2])

    def test_int_with_operator_is_not_silently_zero(self):
        """'ht/4' used to parse as 0, i.e. dynamical_scale_choice = user hook"""
        fmt = bannermod.ConfigFile.format_variable
        self.assertEqual(fmt('10/2', int), 5)
        self.assertEqual(fmt('2*3', int), 6)
        for bad in ('ht/4', 'foo*2', '4/0'):
            self.assertRaises(bannermod.InvalidCmd, fmt, bad, int)
        lo = bannermod.RunCardLO()
        self.assertRaises(bannermod.InvalidCmd, lo.__setitem__, 'dynamical_scale_choice', 'ht/4')
        self.assertEqual(lo['dynamical_scale_choice'], -1)

    def test_scale_factor(self):
        """[beam] scale_factor: default, legacy view, and the fixed-scale warning"""
        rc = bannermod.RunCardMG7()
        self.assertEqual(rc['beam']['scale_factor'], 1.0)
        # legacy 'scalefact' view: the factor, or 1.0 when both scales are fixed
        rc['beam']['scale_factor'] = 0.25
        self.assertEqual(rc['scalefact'], 0.25)
        rc['beam']['fixed_ren_scale'] = True
        rc['beam']['fixed_fact_scale'] = True
        self.assertEqual(rc['scalefact'], 1.0)
        # ... and that combination warns rather than silently doing nothing
        with self.assertLogs('madevent.cards', level='WARNING') as cm:
            rc.check_validity()
        self.assertIn('scale_factor', ' '.join(cm.output))
        # a non-positive factor is refused
        rc['beam']['fixed_ren_scale'] = False
        rc['beam']['fixed_fact_scale'] = False
        rc['beam']['scale_factor'] = 0.0
        self.assertRaises(bannermod.InvalidRunCard, rc.check_validity)

    def test_removed_cpu_mode_refused_for_a_run_reported_from_a_banner(self):
        """A cpu_mode that no longer exists (a card from before the backend
        renaming) is a hard error for a card that will be run. Read back from
        an event file's <MG7RunCard> it is only reported and the default kept,
        so that the tools reading the sample do not abort on it."""
        out = io.StringIO()
        bannermod.RunCardMG7().write(out, template=self.template)
        text, count = re.subn(r'(?m)^cpu_mode\s*=.*$', 'cpu_mode = "cpu_128b"',
                              out.getvalue())
        self.assertEqual(count, 1)

        self.assertRaises(bannermod.InvalidRunCard,
                          bannermod.RunCardMG7, text, consistency=False)

        with self.assertLogs('madevent.cards', level='WARNING') as cm:
            rc = bannermod.RunCardMG7(text, consistency=False, from_banner=True)
        self.assertIn("cpu_mode='cpu_128b'", ' '.join(cm.output))
        self.assertEqual(rc['run']['cpu_mode'], 'auto')
        # the leniency ends with the read
        self.assertRaises(bannermod.InvalidRunCard,
                          rc.__setitem__, 'run.cpu_mode', 'cpu_128b')

        mybanner = bannermod.Banner()
        mybanner['mg7runcard'] = text
        with self.assertLogs('madevent.cards', level='WARNING'):
            self.assertEqual(mybanner.get('run_card', 'nevents'),
                             rc['generation']['events'])

    def test_event_norm_of_an_mg7_sample(self):
        """mg7 weights average to sigma. event_norm is only a legacy-compat
        key of RunCardMG7, invisible to `in`, which made the reweighting log
        sum the weights: its new cross section came out N_events too large.
        get_event_norm sees it, for the card read back from a banner too."""
        rc = bannermod.RunCardMG7()
        self.assertNotIn('event_norm', rc)
        self.assertEqual(rc.get_event_norm(), 'average')

        out = io.StringIO()
        rc.write(out, template=self.template)
        mybanner = bannermod.Banner()
        mybanner['mg7runcard'] = out.getvalue()
        self.assertEqual(mybanner.charge_card('run_card').get_event_norm(),
                         'average')

        # legacy cards: their own value, 'sum' when they have none
        lo = bannermod.RunCardLO()
        self.assertEqual(lo.get_event_norm(), 'average')
        lo['event_norm'] = 'sum'
        self.assertEqual(lo.get_event_norm(), 'sum')
        dict.__delitem__(lo, 'event_norm')
        self.assertEqual(lo.get_event_norm(), 'sum')

    def test_defaults_and_section_access(self):
        """default values are accessible through nested-section views"""
        rc = bannermod.RunCardMG7()
        # fixed (typed) parameters, via card["section"]["key"]
        self.assertEqual(rc['generation']['events'], 100000)
        self.assertEqual(rc['beam']['e_cm'], 13000.0)
        self.assertIs(rc['vegas']['enable'], True)
        self.assertEqual(rc['madnis']['enable'], 'auto')
        self.assertEqual(rc['madnis']['nn_float_precision'], 'float32')
        self.assertEqual(rc['phasespace']['mode'], 'multichannel')
        # the "enable" key colliding between [vegas] and [madnis] is stored
        # independently, so editing one does not affect the other
        rc['madnis']['enable'] = True
        self.assertIs(rc['vegas']['enable'], True)
        self.assertIs(rc['madnis']['enable'], True)
        # dynamic sections are plain nested dicts
        self.assertEqual(rc['multiparticles']['jet'],
                         [1, 2, 3, 4, -1, -2, -3, -4, 21])
        self.assertEqual(rc['cuts']['jet-pt'], {'min': 20.0})
        self.assertEqual(rc['cuts'].get('order_by', 'pt'), 'pt')

    def test_write_is_valid_toml(self):
        """write() produces valid TOML identical to the python defaults"""
        import tomllib
        rc = bannermod.RunCardMG7()
        out = io.StringIO()
        rc.write(out, template=self.template)
        data = tomllib.loads(out.getvalue())
        self.assertEqual(data['run']['output_format'], 'lhe_npy')
        self.assertEqual(data['beam']['ebeam1'], 6500.0)
        self.assertEqual(data['beam']['ebeam2'], 6500.0)
        self.assertEqual(data['beam']['pdf1'], data['beam']['pdf2'])
        self.assertNotIn('e_cm', data['beam'])
        self.assertIs(data['vegas']['enable'], True)
        self.assertEqual(data['multiparticles']['photon'], [22])
        self.assertEqual(data['cuts']['sqrt_s'], {'min': 0.0})
        self.assertEqual(data['madnis']['adam_eps'], 1e-8)
        self.assertEqual(data['madnis']['nn_float_precision'], 'float32')

    def test_round_trip(self):
        """write() then read() preserves every value"""
        rc = bannermod.RunCardMG7()
        rc['generation']['events'] = 250
        rc['phasespace']['mode'] = 'flat'
        rc['madnis']['nn_float_precision'] = 'float64'
        out = io.StringIO()
        rc.write(out, template=self.template)
        rc2 = bannermod.RunCardMG7(out.getvalue())
        self.assertEqual(rc2['generation']['events'], 250)
        self.assertEqual(rc2['phasespace']['mode'], 'flat')
        self.assertEqual(rc2['madnis']['nn_float_precision'], 'float64')
        self.assertEqual(rc2['multiparticles']['lepton'], rc['multiparticles']['lepton'])
        self.assertEqual(rc2['cuts']['jet-delta_r'], {'min': 0.4})

    def test_allowed_value(self):
        """allowed-value enforcement preserves the previous value on bad input"""
        rc = bannermod.RunCardMG7()
        rc.set('phasespace.mode', 'not_a_mode', raiseerror=False)
        self.assertEqual(rc['phasespace']['mode'], 'multichannel')
        rc.set('phasespace.mode', 'flat', raiseerror=False)
        self.assertEqual(rc['phasespace']['mode'], 'flat')

    def test_create_default_for_process(self):
        """process dependent defaults: lepton collider drops jet cuts"""
        class PC(dict):
            def __init__(self):
                super().__init__(ninitial=2, loop_induced=False, colored_pdgs=[])

        def proc(initial):
            legs = [{'state': False, 'id': i} for i in initial]
            legs.append({'state': True, 'id': 11})
            return [{'legs': legs}]

        rc = bannermod.RunCardMG7()
        rc.create_default_for_process(PC(), '', [proc([11, -11])])
        self.assertIs(rc['beam']['leptonic'], True)
        self.assertEqual(rc['beam']['e_cm'], 1000.0)
        self.assertFalse([k for k in rc['cuts'] if k.startswith('jet')])


class TestPhotonPDFDefault(unittest.TestCase):
    """issue #311: the NNPDF4.0 default sets have no photon, so a process with
    a photon taken from the proton, or with EW corrections at NLO, defaults to
    the QED set of the same family."""

    model = None

    @classmethod
    def setUpClass(cls):
        import madgraph.models.import_ufo as import_ufo
        cls.model = import_ufo.import_model('sm')

    def process(self, initial, final=(11, -11), nlo=False):
        import madgraph.core.base_objects as base_objects
        import madgraph.fks.fks_common as fks_common
        Leg = fks_common.FKSLeg if nlo else base_objects.Leg
        legs = [Leg({'id': i, 'state': False}) for i in initial]
        legs += [Leg({'id': i, 'state': True}) for i in final]
        return base_objects.Process({'legs': base_objects.LegList(legs),
                                     'model': self.model})

    def lo_card(self, initial, **user):
        rc = bannermod.RunCardLO()
        for key, value in user.items():
            rc.set(key, value, user=True)
        pc = bannermod.ProcCharacteristic()
        pc['ninitial'] = 2
        rc.create_default_for_process(pc, '', [[self.process(initial)]])
        return rc

    def nlo_card(self, initial, splitting_types=('QCD',), **user):
        rc = bannermod.RunCardNLO()
        for key, value in user.items():
            rc.set(key, value, user=True)
        pc = bannermod.ProcCharacteristic()
        pc['ninitial'] = 2
        pc['splitting_types'] = list(splitting_types)
        rc.create_default_for_process(pc, '', [self.process(initial, nlo=True)])
        return rc

    def test_lo(self):
        self.assertEqual(self.lo_card([2, -2])['lhaid'], 331900)
        self.assertEqual(self.lo_card([22, 22])['lhaid'], 335900)
        self.assertEqual(self.lo_card([2, 22])['lhaid'], 335900)
        # the user choice wins
        self.assertEqual(self.lo_card([22, 22], lhaid=331900)['lhaid'], 331900)

    def test_nlo(self):
        # plain sm has a massive b: 4-flavour set
        self.assertEqual(self.nlo_card([2, -2])['lhaid'], [334700])
        # EW corrections or a photon in the initial state: QED set
        self.assertEqual(self.nlo_card([2, -2], ('QCD', 'QED'))['lhaid'], [335900])
        self.assertEqual(self.nlo_card([22, 2])['lhaid'], [335900])
        # lepton collider: no proton PDF, nothing changes
        rc = self.nlo_card([11, -11], ('QED',))
        self.assertEqual(rc['lpp1'], 0)
        self.assertEqual(rc['lhaid'], [334700])
        # the user choice wins
        self.assertEqual(self.nlo_card([2, -2], ('QED',), lhaid=[331700])['lhaid'],
                         [331700])

    def test_mg7(self):
        def card(initial):
            rc = bannermod.RunCardMG7()
            pc = bannermod.ProcCharacteristic()
            pc['ninitial'] = 2
            rc.create_default_for_process(pc, '', [[self.process(initial)]])
            return rc
        self.assertEqual(card([2, -2])['beam']['pdf1'], 'NNPDF40_lo_as_01180')
        rc = card([22, 22])
        self.assertEqual(rc['beam']['pdf1'], 'NNPDF40_nlo_as_01180_qed')
        self.assertEqual(rc['beam']['pdf2'], 'NNPDF40_nlo_as_01180_qed')
        self.assertIs(rc['beam']['leptonic'], False)


class FakeParticle(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class FakeModel(dict):
    """The little of a model the [histograms] defaults look at."""

    NAMES = {1: 'd', 2: 'u', 5: 'b', 6: 't', 11: 'e-', 13: 'mu-', 21: 'g',
             22: 'a', 23: 'z', 24: 'w+', 25: 'h'}

    def __init__(self, masses=None):
        super().__init__(parameter_dict=masses or {'MT': 173.0})

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def get_particle(self, pdg):
        pdg = abs(pdg)
        return FakeParticle(name=self.NAMES.get(pdg, 'x%d' % pdg),
                            self_antipart=pdg in (21, 22, 23, 25),
                            mass='MT' if pdg == 6 else 'ZERO')


def mg7_proc(initial, final, model=None):
    """One process, in the shape create_default_for_process is handed."""
    legs = [{'state': False, 'id': i} for i in initial]
    legs += [{'state': True, 'id': i} for i in final]
    return {'legs': legs, 'legs_with_decays': [], 'model': model or FakeModel()}


class TestRunCardMG7AutoMLM(unittest.TestCase):
    """A process generated with several jet multiplicities is set up as an MLM
    merged sample at output time, as madevent's ickkw = 1 is."""

    class PC(dict):
        def __init__(self, ninitial=2):
            super().__init__(ninitial=ninitial, loop_induced=False,
                             colored_pdgs=[])

    def build(self, processes):
        rc = bannermod.RunCardMG7()
        rc.create_default_for_process(self.PC(), '', processes)
        return rc

    def test_extra_jets_turn_mlm_on(self):
        rc = self.build([[mg7_proc([2, -1], [-11, 12])],
                         [mg7_proc([2, -1], [-11, 12, 21])]])
        self.assertEqual(rc['beam']['dynamical_scale_choice'], 'mlm')
        self.assertEqual(rc['phasespace']['xqcut'], 30.0)
        self.assertTrue(rc.is_mlm_merged())
        self.assertEqual(rc['ickkw'], 1)
        # drjj = drjl = 0, as madevent
        self.assertNotIn('jet-delta_r', rc.dynamic_sections['cuts'])
        self.assertNotIn('jet-lepton-delta_r', rc.dynamic_sections['cuts'])
        self.assertIn('lepton-delta_r', rc.dynamic_sections['cuts'])

    def test_an_extra_photon_is_no_merging(self):
        rc = self.build([[mg7_proc([2, -1], [-11, 12])],
                         [mg7_proc([2, -1], [-11, 12, 22])]])
        self.assertNotEqual(rc['beam']['dynamical_scale_choice'], 'mlm')
        self.assertEqual(rc['phasespace']['xqcut'], 0.0)
        self.assertIn('jet-delta_r', rc.dynamic_sections['cuts'])

    def test_one_multiplicity_is_no_merging(self):
        rc = self.build([[mg7_proc([2, -1], [-11, 12, 21])]])
        self.assertFalse(rc.is_mlm_merged())

    def test_lepton_collider_is_no_merging(self):
        rc = self.build([[mg7_proc([11, -11], [2, -2])],
                         [mg7_proc([11, -11], [2, -2, 21])]])
        self.assertFalse(rc.is_mlm_merged())
        self.assertEqual(rc['phasespace']['xqcut'], 0.0)


class TestRunCardMG7Histograms(unittest.TestCase):
    """[histograms]: the default plots written at output time."""

    class PC(dict):
        def __init__(self, ninitial=2):
            super().__init__(ninitial=ninitial, loop_induced=False,
                             colored_pdgs=[])

    def build(self, processes, ninitial=2):
        rc = bannermod.RunCardMG7()
        rc.create_default_for_process(self.PC(ninitial), '', processes)
        return rc

    def test_one_group_per_particle(self):
        """p p > t t~: the tops get a [multiparticles] group of their own"""
        rc = self.build([[mg7_proc([21, 21], [6, -6])]])
        self.assertEqual(rc['multiparticles']['t'], [6, -6])
        keys = list(rc['histograms'])
        self.assertEqual(keys, ['t_1-pt', 't_2-pt', 't_1-eta', 't_2-eta',
                                't_1-t_2-pair_mass', 'sqrt_s', 'weight'])
        self.assertEqual(rc['histograms']['t_1-pt']['min'], 0.)
        self.assertGreater(rc['histograms']['t_1-pt']['max'], 0.)
        self.assertEqual(rc['histograms']['t_1-eta']['min'],
                         -rc['histograms']['t_1-eta']['max'])

    def test_predefined_group_when_flavours_are_merged(self):
        """a merged-flavour leg (81/82/83) is histogrammed as its group

        The run time only ever sees the representative of a merged group, so
        asking for "e" in a `p p > l+ l-` run would select nothing.
        """
        rc = self.build([[mg7_proc([81, -81], [82, -82])]])
        self.assertEqual([k for k in rc['histograms'] if k.endswith('-pt')],
                         ['lepton_1-pt', 'lepton_2-pt'])
        self.assertNotIn('e', rc['multiparticles'])

    def test_predefined_group_when_several_members(self):
        """p p > j j is one 'jet' group, not one group per flavour"""
        rc = self.build([[mg7_proc([21, 21], [21, 21]),
                          mg7_proc([21, 21], [1, -1])]])
        self.assertEqual([k for k in rc['histograms'] if k.endswith('-pt')],
                         ['jet_1-pt', 'jet_2-pt'])

    def test_group_that_is_a_single_particle(self):
        """b and a have a group of their own already (bottom, photon)"""
        rc = self.build([[mg7_proc([21, 21], [5, -5])]])
        self.assertEqual([k for k in rc['histograms'] if k.endswith('-pt')],
                         ['bottom_1-pt', 'bottom_2-pt'])

    def test_multiplicity_is_the_smallest_over_processes(self):
        """an index has to exist in every subprocess, or it is an error there"""
        rc = self.build([[mg7_proc([21, 21], [6, -6, 21]),
                          mg7_proc([21, 1], [6, -6, 1])]])
        keys = [k for k in rc['histograms'] if k.endswith('-pt')]
        # two tops always, one jet always (never two)
        self.assertEqual(keys, ['t_1-pt', 't_2-pt', 'jet-pt'])

    def test_decays_are_resolved(self):
        """p p > z, z > e+ e- plots the leptons, not the z"""
        proc = mg7_proc([81, -81], [23])
        proc['legs_with_decays'] = [{'state': False, 'id': 81},
                                    {'state': False, 'id': -81},
                                    {'state': True, 'id': 82},
                                    {'state': True, 'id': -82}]
        rc = self.build([[proc]])
        self.assertEqual([k for k in rc['histograms'] if k.endswith('-pt')],
                         ['lepton_1-pt', 'lepton_2-pt'])

    def test_decay_process_uses_the_decaying_mass(self):
        """a 1 -> N width has no collider energy to scale the ranges with"""
        rc = self.build([[mg7_proc([6], [5, 24])]], ninitial=1)
        self.assertEqual(rc['histograms']['bottom-pt']['max'], 100.)
        self.assertEqual(rc['histograms']['bottom-w-pair_mass']['max'], 200.)
        # madspace's sqrt_s is that of two beams
        self.assertNotIn('sqrt_s', rc['histograms'])

    def test_decaying_mass_without_parameter_dict(self):
        """only a ModelReader has a 'parameter_dict'

        At output time (MadSpin's mg7 decays) the model can be a plain Model
        or a LoopModel: its LoopModel.get('parameter_dict') used to raise a
        PhysicsObjectError, and a Model gave 0. MW is an internal parameter
        of sm-full, so it has to be evaluated from the external ones.
        """
        import madgraph.models.import_ufo as import_ufo
        import madgraph.core.base_objects as base_objects
        import madgraph.loop.loop_base_objects as loop_base_objects

        full = import_ufo.import_model('sm-full')
        for cls in (base_objects.Model, loop_base_objects.LoopModel):
            model = cls()
            for key in model:
                if key in full:
                    model[key] = full[key]
            self.assertNotIn('parameter_dict', model)
            proc = mg7_proc([24], [-11, 12], model=model)
            mass = bannermod.RunCardMG7._decaying_mass([[proc]])
            self.assertAlmostEqual(mass, 80.419, places=3)
            # evaluated on the side: the model itself is not touched
            self.assertIsNone(model.get_parameter('mdl_MW').value)

    def test_every_pair_gets_an_invariant_mass(self):
        rc = self.build([[mg7_proc([21, 21], [6, -6, 25])]])
        pairs = [k for k in rc['histograms'] if k.endswith('-pair_mass')]
        self.assertEqual(pairs, ['t_1-t_2-pair_mass', 't_1-h-pair_mass',
                                 't_2-h-pair_mass'])

    def test_hwu_output_is_off_by_default(self):
        """the HwU file is a second copy of what info.json already has"""
        rc = self.build([[mg7_proc([21, 21], [6, -6])]])
        self.assertIs(rc['run']['write_hwu'], False)
        out = io.StringIO()
        rc['run']['write_hwu'] = True
        rc.write(out)
        self.assertIn('write_hwu = true', out.getvalue())
        self.assertIs(bannermod.RunCardMG7(out.getvalue())['run']['write_hwu'],
                      True)

    def test_weight_distribution(self):
        """'weight' is the reserved key for the event weight itself

        It is binned in units of the unit weight, so the range does not
        depend on the process (an unweighted sample is a spike at 1).
        """
        rc = self.build([[mg7_proc([21, 21], [6, -6])]])
        weight = rc['histograms']['weight']
        self.assertEqual(weight['min'], 0.)
        self.assertGreater(weight['max'], 1.)

    def test_weight_distribution_of_an_interference(self):
        """a squared-order constraint can give negative weights, at -1: the
        range is symmetric so that they are not lost in the underflow"""
        proc = mg7_proc([21, 21], [6, -6])
        proc['squared_orders'] = {'HIG': 1}
        rc = self.build([[proc]])
        weight = rc['histograms']['weight']
        self.assertEqual(weight['min'], -weight['max'])
        self.assertGreater(weight['max'], 1.)

    def test_round_trip(self):
        """the generated section survives write() + read()"""
        rc = self.build([[mg7_proc([21, 21], [6, -6])]])
        out = io.StringIO()
        rc.write(out)
        back = bannermod.RunCardMG7(out.getvalue())
        self.assertEqual(dict(back['histograms']), dict(rc['histograms']))
        self.assertEqual(back['multiparticles']['t'], [6, -6])

    def test_remove_and_restore(self):
        """set histograms OFF / default"""
        rc = self.build([[mg7_proc([21, 21], [6, -6])]])
        default = io.StringIO()
        rc.write(default)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml',
                                         delete=False) as tmp:
            tmp.write(default.getvalue())
            path = tmp.name
        try:
            rc.remove_all_histograms()
            self.assertEqual(dict(rc['histograms']), {})
            self.assertTrue(rc.restore_default_histograms(path))
            self.assertIn('t_1-pt', rc['histograms'])
            self.assertFalse(rc.restore_default_histograms(path + '.missing'))
        finally:
            os.remove(path)

    def test_no_process_no_histograms(self):
        """no crash (and nothing written) without usable process information"""
        rc = bannermod.RunCardMG7()
        rc.create_default_for_process(self.PC(), '', [])
        self.assertEqual(dict(rc['histograms']), {})


MadLoopParam = bannermod.MadLoopParam
class TestMadLoopParam(unittest.TestCase):
    """ A class to test the MadLoopParam functionality """
    
    
    def test_initMadLoopParam(self):
        """check that we can initialize a file"""
        
        #1. create the object without argument and the default file
        param1 = MadLoopParam()
        param2 = MadLoopParam(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))
        
        #2. check that they are all equivalent
        self.assertEqual(param2.user_set, set())
        self.assertEqual(param1.user_set, set())
        for key, value1 in param1.items():
            self.assertEqual(value1, param2[key])
        
        #3. check that all the Default value in the file MadLoopParams.dat
        #   are coherent with the default in python
        
        fsock = open(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))
        previous_line = ["", ""]
        for line in fsock:
            if previous_line[0].startswith('#'):
                name = previous_line[0][1:].strip()
                self.assertIn('default', line.lower())
                value = line.split('::')[1].strip()
                param2[name] = value # do this such that the formatting is done
                self.assertEqual(param1[name], param2[name])
                self.assertTrue(previous_line[1].startswith('!'))
            previous_line = [previous_line[1], line]
            
    def test_modifparameter(self):
        """ test that we can modify the parameter and that the formatting is applied
        correctly """

        #1. create the object without argument
        param1 = MadLoopParam()

        to_test = {"MLReductionLib": {'correct': ['1|2', ' 1|2 '],
                                      'wrong':[1/2, 0.3, True],
                                      'target': ['1|2', '1|2']},
                   "IREGIMODE": {'correct' : [1.0, 2, 3, -1, '1.0', '2', '-3', '-3.0'],
                                 'wrong' : ['1.5', '-1.5', 1.5, -3.4, True, 'starwars'],
                                 'target': [1,2,3,-1,1,2,-3,-3]
                                  },
                   "IREGIRECY": {'correct' : [True, False, 0, 1, '0', '1',
                                                '.true.', '.false.','T', 
                                                  'F', 'true', 'false', 'True \n'],
                                 'wrong' : ['a', [], 5, 66, {}, None, -1],
                                 "target": [True, False, False, True, False, True, 
                                            True, False,True, False,True,False, True]},
                   "CTStabThres": {'correct': [1.0, 1e-3, 1+0j, 1,"1d-3", "1e-3"],
                                   'wrong': [True, 'hello'],
                                   'target': [1.0,1e-3, 1.0, 1.0, 1e-3, 1e-3]}
                   }

        import madgraph.various.misc as misc
        for name, data in to_test.items():
            for i,value in enumerate(data['correct']):
                param1[name] = value
                self.assertEqual(param1[name],  data['target'][i])
                self.assertNotIn(name.lower(), param1.user_set)
                self.assertEqual(type(data['target'][i]), type(param1[name]))
            for value in data['wrong']:
                self.assertRaises(Exception, param1.__setitem__, (name, value))
                
    def test_writeMLparam(self):
        """check that the writing is correct"""
        
        param1 = MadLoopParam(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))
        
        textio = io.StringIO()
        param1.write(textio)
        text=textio.getvalue()
        
        #read the data.
        param2=MadLoopParam(text)
        
        #check that they are correct
        for key, value in param1.items():
            self.assertEqual(value, param2[key])
            self.assertIn(key.lower(), param2.user_set)
            
    def test_sum_object(self):
        
        param1 = MadLoopParam(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))


        new = {'test': 1, 'value': 'data----------------------------------'}

        ########################################################################
        # 1. simple sum all key different
        ########################################################################        
        param2 = param1 + new

        self.assertIsInstance(param2, MadLoopParam)
        self.assertIsInstance(param2, dict)
        self.assertNotEqual(id(param1), id(param2))
        
        #check that they are correct
        for key, value in param1.items():
            self.assertEqual(value, param2[key])
            self.assertNotIn(key.lower(), param2.user_set)
        for key, value in new.items():
            self.assertEqual(value, param2[key])
            self.assertNotIn(key.lower(), param2.user_set)
        self.assertNotIn('test', param1)
                   
        
        
        ########################################################################
        # 2. add same key in both term
        ########################################################################
        new = {'test': 1, 'value': 'data', 'CTLoopLibrary':1}
        param2 = param1 + new
        #check that they are correct
        for key, value in param1.items():
            if key != 'CTLoopLibrary':
                self.assertEqual(value, param2[key])
                self.assertNotIn(key.lower(), param2.user_set)
                     
        for key, value in new.items():
            self.assertEqual(value, param2[key])
            self.assertNotIn(key.lower(), param2.user_set)
            
            
        ########################################################################
        # 3. reverse order
        ########################################################################
        param2 = new + param1   
        
        #check sanity
        self.assertNotIsInstance(param2, MadLoopParam)
        self.assertIsInstance(param2, dict)
        self.assertNotEqual(id(new), id(param2))
        self.assertNotEqual(id(param1), id(param2))
        
        #check that value are correct
        for key, value in param1.items():
                self.assertEqual(value, param2[key])        
        for key, value in new.items():
            if key != 'CTLoopLibrary':
                self.assertEqual(value, param2[key])


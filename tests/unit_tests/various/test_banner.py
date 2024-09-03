################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import unittest
import tempfile
import madgraph.various.banner as bannermod
import madgraph.various.misc as misc
import os
import models
import six
StringIO = six
import sys
from madgraph import MG5DIR



_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

pjoin = os.path.join


class TestBanner(unittest.TestCase):
    """ A class to test the banner functionality """
    
    
    def test_banner(self):

        #try to instansiate a banner with no argument
        mybanner = bannermod.Banner()
        self.assertTrue(hasattr, (mybanner, "lhe_version"))
        
        #check that you can instantiate a banner from a banner object
        secondbanner = bannermod.Banner(mybanner)
        
        # check that all attribute are common
        self.assertEqual(mybanner.__dict__, secondbanner.__dict__)
        
        # check that the two are different and independant
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
    # a lot of the functionality are actually already tested in the child
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
        self.config['list_s'] = " 1\ 2, 3, 5d1 "
        self.assertEqual(self.config['list_s'],['1\ 2','3', '5d1']) 

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

        #Note that this is a bit hacky since this is not a normall class fct
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
        
        input = StringIO.StringIO(
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
        output = StringIO.StringIO()
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
        out = StringIO.StringIO()
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
              
        out = StringIO.StringIO()
        pythia8_card_read.write(out,self.basic_PY8_template)       
        misc.sprint('READ:',out.getvalue())
        out = StringIO.StringIO()
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
        out = StringIO.StringIO()
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
        out = StringIO.StringIO()
        modified_PY8Card.write(out,self.basic_PY8_template)
        out.seek(0)        
        read_PY8Card=bannermod.PY8Card(out)
        self.assertEqual(modified_PY8Card, read_PY8Card)



import shutil
class TestRunCard(unittest.TestCase):
    """ A class to test the TestConfig functionality """
    # a lot of the funtionality are actually already tested in the child
    # TESTMadLoopParam and are not repeated here.
    

    def setUp(self):
        self.debugging = unittest.debug
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
            if key == 'hel_recycling' and six.PY2:
                continue 
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

        # check that for elastisc a a collision we do  not force to use fixed_fac_scale1/2
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
           test_guess_entry_fromname. So this test is mainly to test that the output function of that function is corrrectly
           linked to add_param as done in add_unknown_entry
        """

        run_card = bannermod.RunCardLO()
        fct = run_card.add_unknown_entry

        # simple one 
        input = ("STR_INCLUDE_PDF", "True ")
        fct(*input)
        # check value and that parameter is hidden by default and in autodef
        name = "INCLUDE_PDF" 
        self.assertEqual(run_card[name], "True")
        self.assertIn(name.lower(), run_card.hidden_param)
        self.assertIn(name.lower(), run_card.definition_path[True])

        # complex case: list + metadata
        input = ("test_data<cut=True><include=False><fortran_name=input_2>", "[1,2,3,4,5]")
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
        input = ("test_dict", "{'__type__':1.0, '6':3.0}")
        fct(*input)
        # check value and that parameter is hidden by default and in autodef
        name = "test_dict"
        self.assertEqual(run_card[name], {'__type__':1.0, '6':3.0}) # this check that list are correctly formatted
        self.assertIn(name.lower(), run_card.hidden_param)
        # default for dict is not to include in Fortran
        self.assertNotIn(name.lower(), run_card.definition_path[True]) 
        self.assertNotIn(name, run_card.includepath[True])

        # check that one can overwritte hidden 
        input = ("max_data<hidden=False>", "3.0")
        fct(*input)
        name = "max_data"
        self.assertEqual(run_card[name], 3.0)
        self.assertNotIn(name.lower(), run_card.hidden_param)

        # check that one can overwritte autodef
        input = ("max_data2<autodef=False>", "3")
        fct(*input)
        name = "max_data2"
        self.assertEqual(run_card[name], 3.0)
        self.assertNotIn(name.lower(), run_card.definition_path[True])
        self.assertIn(name.lower(), run_card.hidden_param)

        # check that one can overwritte include to False but autodef to True
        # check that one can overwritte autodef
        input = ("data3<autodef=True><include=False>", "True")
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
        run_card.add_unknown_entry("STR_INCLUDE_PDF", "True ")
        f = StringIO.StringIO()
        f.write("c .   this is a comment to test feature of missing end line ")
        run_card.write_autodef(None,output_file=f)
        self.assertIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("C START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf", f.getvalue()) #no automatic formatting due to iostring for unittest

        # adding a second in place
        run_card.add_unknown_entry("BOOL_INCLUDE_PDF2", "True ")
        run_card.write_autodef(None,output_file=f)
        self.assertIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertIn("C START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        # order of the two variable within the common block is not important
        if "COMMON/USER_CUSTOM_RUN/include_pdf," in f.getvalue():
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf,include_pdf2", f.getvalue())
        else:
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf2,include_pdf", f.getvalue())

        # reset, keep one , remove one and add a new one (keep same stream)
        run_card = bannermod.RunCardLO()
        run_card.add_unknown_entry("BOOL_INCLUDE_PDF2", "True ")
        run_card.add_unknown_entry("test_list", "[1,2,3,4,5]")
        run_card.write_autodef(None,output_file=f)
        self.assertNotIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertIn("INTEGER TEST_LIST(0:5)", f.getvalue())
        # check common block part
        self.assertIn("C START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C STOP USER COMMON BLOCK", f.getvalue())
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
        self.assertIn("C START USER COMMON BLOCK", f.getvalue())
        self.assertIn("C STOP USER COMMON BLOCK", f.getvalue())
        self.assertIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())
        if "COMMON/USER_CUSTOM_RUN/include_pdf2," in f.getvalue():
            self.assertIn("COMMON/USER_CUSTOM_RUN/include_pdf2,test_list", f.getvalue())
        else:
            self.assertIn("COMMON/USER_CUSTOM_RUN/test_list,include_pdf2", f.getvalue())

        #check that cleaning is occuring correctly 
        run_card = bannermod.RunCardLO()
        run_card.write_autodef(None,output_file=f)
        self.assertNotIn("CHARACTER INCLUDE_PDF(0:100)", f.getvalue())
        self.assertNotIn("LOGICAL INCLUDE_PDF2", f.getvalue())
        self.assertNotIn("INTEGER TEST_LIST(0:5)", f.getvalue())
        self.assertNotIn("INTEGER TEST_LIST(0:7)", f.getvalue())
        # check common block part
        self.assertNotIn("C START USER COMMON BLOCK", f.getvalue())
        self.assertNotIn("C STOP USER COMMON BLOCK", f.getvalue())
        self.assertNotIn("COMMON/USER_CUSTOM_RUN/", f.getvalue())

    def test_autodef_nomissmatch(self):
        """
        check that the code detects LO/NLO cards missmatch and crash correctly in that case
        """
        
        LO = bannermod.RunCardLO()
        NLO = bannermod.RunCardNLO()
        flo = StringIO.StringIO()
        fnlo = StringIO.StringIO()
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

        # launch the function with only one editted
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


    def test_pdlabel_block(self):
        """ check that pdlabel handling is done correctly
            this include that check_validity works as expected for such parameter too """

        run_card = bannermod.RunCardLO()

        # setting e- p collision
        run_card['lpp1'] = 3
        run_card['lpp2'] = 1
        run_card.check_validity()
        # check that pdlabel is set correctly
        self.assertEqual(run_card['pdlabel'], 'mixed')
        self.assertEqual(run_card['pdlabel1'], 'eva') # since automatically set to eva if lpp=3/4 and pdlabel is lhapdf/nnpdf
        self.assertEqual(run_card['pdlabel2'], 'nn23lo1')
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
        # should now allow assymetric pdlabel here
        run_card.set('pdlabel1','lhapdf', user=True) 
        run_card.set('pdlabel2', 'nnpdf23lo1', user=True) 
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
        self.assertEqual(len(run_card.allowed_lep_densities), 1)
        self.assertEqual(len(run_card.allowed_lep_densities[(-11,11)]), 6)

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
        # but that invidual value are kept 
        f = StringIO.StringIO()
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

        run_card.set('fixed_fac_scale', True, user=True)
        #self.assertNotIn('fixed_fact_scale', run_card.display_block)
        self.assertEqual(run_card['fixed_fac_scale2'], False) #check that this is default value

        run_card.set('fixed_fac_scale1', False, user=True)
        #self.assertIn('fixed_fact_scale', run_card.display_)
        self.assertEqual(run_card['fixed_fac_scale2'], True)
        self.assertNotIn('fixed_fac_scale', run_card.user_set)
        self.assertNotIn('fixed_fac_scale2', run_card.user_set)    

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
        """ test that we can modify the parameter and that the formating is applied 
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
        """check that the writting is correct"""
        
        param1 = MadLoopParam(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))
        
        textio = StringIO.StringIO()
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


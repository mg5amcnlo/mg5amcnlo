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

import unittest
import tempfile
import madgraph.various.banner as bannermod
import madgraph.various.misc as misc
import os
import models
import StringIO
from madgraph import MG5DIR

import StringIO

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
        self.assertTrue(mybanner.has_key('slha'))
        
        #check that the banner can be written        
        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        mybanner.write(fsock)

        #charge a card
        mybanner.charge_card('param_card')
        self.assertTrue(hasattr(mybanner, 'param_card'))
        self.assertTrue(isinstance(mybanner.param_card, models.check_param_card.ParamCard))
        self.assertTrue('mass' in mybanner.param_card)
        

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
        self.assertTrue('UPPER2' in config2)
        
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
        
        self.assertRaises(Exception, self.config.__setitem__, 'list', {1:2,3:4})
        

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

    def test_handling_dict_of_values(self):
        """check that the read/write of a list of value works"""
        
        # add a parameter which can be a list
        self.config.add_param("dict", {'__type__':1.0})
        self.assertEqual(self.config['dict'], {})
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

    def test_auto_handling(self):
        """check that any parameter can be set on auto and recover"""
        
        self.config['lower'] = 'auto'
        self.assertEqual(self.config['lower'],'auto')
        self.assertEqual(dict.__getitem__(self.config,'lower'),1)
        self.assertTrue('lower' in self.config.auto_set)
        self.assertFalse('lower' in self.config.user_set)
        
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

      
    def test_for_loop(self):
        """ check correct handling of case"""
    
        keys = []
        for key in self.config:
            keys.append(key)
        self.assertEqual(set(keys), set(self.config.keys()))
        self.assertTrue('upper' not in keys)
        self.assertTrue('UPPER' in keys)
    
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

class TestRunCard(unittest.TestCase):
    """ A class to test the TestConfig functionality """
    # a lot of the funtionality are actually already tested in the child
    # TESTMadLoopParam and are not repeated here.
     
        
    def test_basic(self):
        """ """
        
        # check the class factory works        
        run_card = bannermod.RunCard()
        self.assertTrue(isinstance(run_card, bannermod.RunCard))
        self.assertTrue(isinstance(run_card, bannermod.RunCardLO))
        self.assertFalse(isinstance(run_card, bannermod.RunCardNLO))
        
        path = pjoin(_file_path, '..', 'input_files', 'run_card_matching.dat')
        run_card = bannermod.RunCard(path)
        self.assertTrue(isinstance(run_card, bannermod.RunCard))
        self.assertTrue(isinstance(run_card, bannermod.RunCardLO))
        self.assertFalse(isinstance(run_card, bannermod.RunCardNLO))        
        
        path = pjoin(_file_path,'..', 'input_files', 'run_card_nlo.dat')
        run_card = bannermod.RunCard(path)
        self.assertTrue(isinstance(run_card, bannermod.RunCard))
        self.assertTrue(isinstance(run_card, bannermod.RunCardNLO))
        self.assertFalse(isinstance(run_card, bannermod.RunCardLO))         
        
        #check the copy
        run_card2 = bannermod.RunCard(run_card)
        self.assertTrue(isinstance(run_card, bannermod.RunCard))
        self.assertTrue(isinstance(run_card, bannermod.RunCardNLO))
        self.assertFalse(isinstance(run_card, bannermod.RunCardLO))         
        #check all list/dict are define
        self.assertTrue(hasattr(run_card2, 'user_set'))
        self.assertTrue(hasattr(run_card2, 'hidden_param'))
        self.assertTrue(hasattr(run_card2, 'includepath')) 
        self.assertTrue(hasattr(run_card2, 'fortran_name'))
        self.assertFalse(hasattr(run_card2, 'default'))
        self.assertTrue(hasattr(run_card2, 'cuts_parameter'))   
              
  
    def test_default(self):
      
        run_card = bannermod.RunCard()
        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        run_card.write(fsock)
      
        run_card2 = bannermod.RunCard(fsock.name)
      
        for key in run_card:
            self.assertEqual(run_card[key], run_card2[key])
      
        run_card = bannermod.RunCardNLO()
        fsock = tempfile.NamedTemporaryFile(mode = 'w')
        run_card.write(fsock)
      
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
                self.assertTrue(key.lower() in run_card.hidden_param) 
                self.assertTrue(key.lower not in run_card3.user_set)
            if key in run_card3.user_set:
                has_userset=True   
                self.assertFalse(key in run_card.user_set)            
        self.assertTrue(has_difference)
        self.assertTrue(has_userset)
        
        #write run_card3 and check that nothing is changed
        fsock2 = tempfile.NamedTemporaryFile(mode = 'w')
        run_card3.write(fsock2)
        
        self.assertEqual(open(fsock.name).read(), open(fsock2.name).read())
            

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
                self.assertTrue(name.lower() not in param1.user_set)
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
            self.assertTrue(key.lower() in param2.user_set)
            
    def test_sum_object(self):
        
        param1 = MadLoopParam(pjoin(MG5DIR,"Template", "loop_material","StandAlone",
                                      "Cards","MadLoopParams.dat"))


        new = {'test': 1, 'value': 'data----------------------------------'}

        ########################################################################
        # 1. simple sum all key different
        ########################################################################        
        param2 = param1 + new

        self.assertTrue(isinstance(param2, MadLoopParam))
        self.assertTrue(isinstance(param2, dict))
        self.assertNotEqual(id(param1), id(param2))
        
        #check that they are correct
        for key, value in param1.items():
            self.assertEqual(value, param2[key])
            self.assertFalse(key.lower() in param2.user_set)        
        for key, value in new.items():
            self.assertEqual(value, param2[key])
            self.assertFalse(key.lower() in param2.user_set)  
        self.assertTrue('test' not in param1)
                   
        
        
        ########################################################################
        # 2. add same key in both term
        ########################################################################
        new = {'test': 1, 'value': 'data', 'CTLoopLibrary':1}
        param2 = param1 + new
        #check that they are correct
        for key, value in param1.items():
            if key != 'CTLoopLibrary':
                self.assertEqual(value, param2[key])
                self.assertFalse(key.lower() in param2.user_set)   
                     
        for key, value in new.items():
            self.assertEqual(value, param2[key])
            self.assertFalse(key.lower() in param2.user_set)   
            
            
        ########################################################################
        # 3. reverse order
        ########################################################################
        param2 = new + param1   
        
        #check sanity
        self.assertFalse(isinstance(param2, MadLoopParam))
        self.assertTrue(isinstance(param2, dict))
        self.assertNotEqual(id(new), id(param2))
        self.assertNotEqual(id(param1), id(param2))
        
        #check that value are correct
        for key, value in param1.items():
                self.assertEqual(value, param2[key])        
        for key, value in new.items():
            if key != 'CTLoopLibrary':
                self.assertEqual(value, param2[key])


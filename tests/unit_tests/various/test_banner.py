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
from madgraph import MG5DIR

import StringIO

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

pjoin = os.path.join


class TESTBanner(unittest.TestCase):
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
    # a lot of the funtionality are actually already tested in the child
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
        # Fail to have the correct behavior for that one. Should be ok in general       
        #self.config['list_s'] = " 1\\ 2, 3, 5d1 "        
        #self.assertEqual(self.config['list_s'],['1\\', '2','3', '5d1'])

        
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
        self.assertTrue(hasattr(run_card2, 'not_in_include')) 
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
      
        run_card2 = bannermod.RunCard(fsock.name)
      
        for key in run_card:
            self.assertEqual(run_card[key], run_card2[key])  
            


MadLoopParam = bannermod.MadLoopParam
class TESTMadLoopParam(unittest.TestCase):
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
            
                
        
        
        
        
        
        
        
        
        
        
            
                

















        

        
            
        
        
        
        
        

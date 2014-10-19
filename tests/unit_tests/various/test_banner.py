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
        self.assertEqual(mybanner.get('run_card', 'lpp1'), '0')
        

MadLoopParam = bannermod.MadLoopParam
class TESTMadLoopParam(unittest.TestCase):
    """ A class to test the banner functionality """
    
    
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
        
        
        
        
        
        
        
        
        
            
                

















        

        
            
        
        
        
        
        

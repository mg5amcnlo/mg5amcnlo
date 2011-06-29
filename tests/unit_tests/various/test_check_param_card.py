################################################################################
#
# Copyright (c) 2011 The MadGraph Development team and Contributors
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
from __future__ import division
import random
import StringIO
import os
import sys
import tests.unit_tests as unittest

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

import madgraph.core.base_objects as base_objects

import models.import_ufo as import_ufo

sys.path.append('%s/../../Template/bin/internal' % _file_path)
import check_param_card as writter



class TestParamCardRule(unittest.TestCase):
    """ Test the ParamCardRule Object"""
    
    def setUp(self):
        """test"""
        self.main = writter.ParamCardRule()
    
    def test_read(self):
        """Check if we can read a file"""
        
        self.main.load_rule(os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_rule_sm.dat'))
        
        self.assertEqual(2, len(self.main.zero))
        self.assertEqual(self.main.zero,[('Mass', [1], ''), ('Mass', [2], '')])
        self.assertEqual(self.main.one,[('CKM', [1, 1], ''), ('CKM', [2, 2], '')])
        self.assertEqual(self.main.identical,[('Mass', [1], [2], '')])
        
    def test_write(self):
        """Check if we can write a file"""
        self.main.add_zero('mass',[1])
        self.main.add_zero('mass',[2])
        self.main.add_one('mass',[3,2])
        self.main.add_identical('mass',[1],[2])
        fsock = StringIO.StringIO()
        self.main.write_file(fsock)
        out = fsock.getvalue()
        
        target = """<file>######################################################################
## VALIDITY RULE FOR THE PARAM_CARD   ####
######################################################################
<zero>
     mass 1 # 
     mass 2 # 
</zero>
<one>
     mass 3    2 # 
</one>
<identical>
     mass 1 : 2 # 
</identical>
<constraint>
</constraint>
</file>"""

        self.assertEqual(out.split('\n'), target.split('\n'))
        
    def test_read_write_param_card(self):
        """Test that we can write a param_card from the dict info"""
        
        dict = self.main.read_param_card(os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat'))
        
        solution = {'yukawa': {'[4]': (0.0, 'ymc'),
                               '[5]': (0.0, 'ymb'), 
                               '[6]': (164.5, 'ymt'), 
                               '[15]': (1.777, 'ymtau')}, 
                    'ckmblock': {'[1]': (0.0, 'cabi')}, 
                    'sminputs': {'[3]': (0.118, 'as'), 
                                 '[1]': (132.507, 'aewm1'), 
                                 '[2]': (1.16639e-05, 'gf')}, 
                    'mass': {'[13]': (0.0, 'mm'), 
                             '[23]': (91.188, 'mz'), 
                             '[15]': (1.777, 'mta'), 
                             '[11]': (0.0, 'me'), 
                             '[6]': (174.3, 'mt'), 
                             '[25]': (91.188, 'mh'), 
                             '[4]': (0.0, 'mc'), 
                             '[5]': (0.0, 'mb'), 
                             '[2]': (0.0, 'mu'), 
                             '[3]': (0.0, 'ms'), 
                             '[1]': (0.0, 'md')}, 
                    'decay': {'[6]': (0.0, 'wt'), 
                              '[25]': (2.441404, 'wh'), 
                              '[24]': (3.0, 'ww'), 
                              '[23]': (2.441404, 'wz')}}
        
        self.assertEqual(dict, solution)
        fsock = StringIO.StringIO()
        self.main.write_param_card(fsock, dict)
        output = fsock.getvalue()
        dict = self.main.read_param_card([l+'\n' for l in output.split('\n')])
        self.assertEqual(dict, solution)
 
    
    def test_load_with_restrict_model(self):
        """ check that the rule are correctly set for a restriction """
        
        # Load a model and a given restriction file
        sm_path = import_ufo.find_ufo_path('sm')
        base_model = import_ufo.import_full_model(sm_path)
        base_model = import_ufo.RestrictModel(base_model)
        restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        base_model.set_parameters_and_couplings(restrict_file)
        base_model.restrict_model(restrict_file)
        
        # Check the information of the CardRule is present and fine:
        self.assertTrue(hasattr(base_model,'rule_card'))
        
        target_zero = [('ckmblock', [1], ''), 
                       ('yukawa', [4], ''), 
                       ('yukawa', [5], ''),
                       ('mass', [11], ''),
                       ('mass', [13], ''),
                       ('mass', [2], ''),
                       ('mass', [4], ''),
                       ('mass', [1], ''),
                       ('mass', [3], ''),
                       ('mass', [5], ''),
                       ('decay', [15], ''),
                       ('decay', [6], '')]
        self.assertEqual(base_model.rule_card.zero, target_zero)
        target_one = []
        self.assertEqual(base_model.rule_card.one, target_one)
        target_identical = [('mass', [25], [23], ''), ('decay', [25], [23], '')]                
        self.assertEqual(base_model.rule_card.identical, target_identical)
        target_rule = []
        self.assertEqual(base_model.rule_card.rule, target_rule)
        
        # test that the rule_card is what we expect
        fsock = StringIO.StringIO()
        base_model.rule_card.write_file(fsock)
        out = fsock.getvalue()
        target ="""<file>######################################################################
## VALIDITY RULE FOR THE PARAM_CARD   ####
######################################################################
<zero>
     ckmblock 1 # 
     yukawa 4 # 
     yukawa 5 # 
     mass 11 # 
     mass 13 # 
     mass 2 # 
     mass 4 # 
     mass 1 # 
     mass 3 # 
     mass 5 # 
     decay 15 # 
     decay 6 # 
</zero>
<one>
</one>
<identical>
     mass 25 : 23 # 
     decay 25 : 23 # 
</identical>
<constraint>
</constraint>
</file>"""

        self.assertEqual(out.split('\n'), target.split('\n'))
        
    def test_check_param(self):
        """check if the check param_card is working""" 
        
        # Load a model and a given restriction file
        sm_path = import_ufo.find_ufo_path('sm')
        base_model = import_ufo.import_full_model(sm_path)
        base_model = import_ufo.RestrictModel(base_model)
        restrict_file = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'restrict_sm.dat')
        base_model.set_parameters_and_couplings(restrict_file)
        base_model.restrict_model(restrict_file)
        
        #
        base_model.rule_card.check_param_card(restrict_file)
        full_card = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_sm.dat')
        self.assertRaises(writter.InvalidParamCard, base_model.rule_card.check_param_card,
                    full_card) 
        
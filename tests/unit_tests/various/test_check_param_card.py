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


class TestBlock(unittest.TestCase):
    """Check the class linked to a block of the param_card"""
    
    def test_block_load_string(self):
        """test that we recognize the different syntax"""

        text = """Block SMINPUTS"""
        b = writter.Block()
        b.load_str(text)
        self.assertEqual(b.name, 'sminputs')
        self.assertEqual(b.scale, None)
        
        text = """Block SMINPUTS # Q=1 #"""
        b = writter.Block()
        b.load_str(text)
        self.assertEqual(b.name, 'sminputs')
        self.assertEqual(b.scale, None)       
        
        text = """Block SMINPUTS  Q=1 #"""
        b = writter.Block()
        b.load_str(text)
        self.assertEqual(b.name, 'sminputs')
        self.assertEqual(b.scale, 1) 
        
    def test_block_str(self):
        """check that we can write correctly the block"""    

        text = """Block SMINPUTS  Q=1 # test"""
        b = writter.Block()
        b.load_str(text)
        target="""###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS Q=1.0 #  test

"""
        self.assertEqual(str(b).split('\n'), target.split('\n'))


    def test_block_append_remove(self):
        """check if we can safely add a parameter"""
        
        text = """Block SMINPUTS  Q=1 # test"""
        b = writter.Block()
        b.load_str(text)
        
        b.append(writter.Parameter(block='sminputs', lhacode=[1,2], value=3))
        b.append(writter.Parameter(block='sminputs', lhacode=[1], value=4))

        self.assertEqual(len(b),2)
        self.assertRaises(AssertionError, b.append, writter.Parameter(block='other'))
                         
        self.assertRaises(AssertionError, 
           b.append, writter.Parameter(block='sminputs', lhacode=[1,2], value=9))
        self.assertEqual(len(b),2)
        
        
        b.remove([1,2])
        self.assertEqual(len(b),1)
        self.assertEqual(b.param_dict.keys(),[(1,)])               


class TestParamCard(unittest.TestCase):
    """ Test the ParamCard Object """
    
    
    def test_mod_card(self):
        """ test that we can modify a param card """

        full_card = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_sm.dat')        
        card = writter.ParamCard(full_card)
        
        # Rename the blocks
        mass = card['mass']
        card.rename_blocks({'mass':'polemass','decay':'width'})
        self.assertTrue(card.has_key('polemass'))
        self.assertTrue(card.has_key('width'))
        self.assertFalse(card.has_key('mass'))
        self.assertFalse(card.has_key('decay'))        
        self.assertEqual(mass, card['polemass'])
        self.assertEqual(mass.name, 'polemass')
    
        # Change the lhacode of a parameter
        param = card['width'].get([23])
        card.mod_param('width', [23], lhacode=[32])
        
        self.assertRaises(KeyError, card['width'].get, [23])
        self.assertEqual(param, card['width'].get([32]))
        self.assertEqual(param.lhacode, [32])
        
        # change the block of a parameter
        card.mod_param('width', [32], block='mass')
        
        self.assertRaises(KeyError, card['width'].get, [32])
        self.assertEqual(param, card['mass'].get([32]))
        self.assertEqual(param.lhacode, [32])
        self.assertEqual(param.block, 'mass')
        
        # change the block of a parameter and lhacode
        card.mod_param('mass', [32], block='polemass', lhacode=[23])
        
        self.assertFalse(card.has_key('mass'))
        self.assertRaises(KeyError, card['polemass'].get, [32])
        self.assertRaises(KeyError, card['width'].get, [32])
        self.assertEqual(param, card['polemass'].get([23]))
        self.assertEqual(param.lhacode, [23])
        self.assertEqual(param.block, 'polemass')        
        
        # change the value / comment
        card.mod_param('polemass', [23], value=2, comment='new')
        self.assertEqual(param.value, 2)
        self.assertEqual(param.comment, 'new')
        
        
        
        
        
        
        
        
        
        
        
        
        
        

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
        """Test that we can write a param_card from the ParamCard object"""
        
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


        for key, item in solution.items():
            for key2, (value, comment) in item.items():
                self.assertEqual(value, float(dict[key].get(eval(key2)).value))
       

        fsock = StringIO.StringIO()
        self.main.write_param_card(fsock, dict)
        output = fsock.getvalue()

        target = """######################################################################
## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####
######################################################################
###################################
## INFORMATION FOR MASS
###################################
BLOCK MASS # 
      11 0.000000e-04 #  me
      13 0.000000e-01 #  mm
      15 1.777000e+00 #  mta
      2 0.000000e-03 #  mu
      4 0.000000e+00 #  mc
      6 1.743000e+02 #  mt
      1 0.000000e-03 #  md
      3 0.000000e-01 #  ms
      5 0.000000e+00 #  mb
      23 9.118800e+01 #  mz
      25 9.118800e+01 #  mh
###################################
## INFORMATION FOR CKMBLOCK
###################################
BLOCK CKMBLOCK # 
      1 0.000000e-01 #  cabi
###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS # 
      1 1.32507000e+02 #  aewm1
      2 1.166390e-05 #  gf
      3 1.180000e-01 #  as
###################################
## INFORMATION FOR YUKAWA
###################################
BLOCK YUKAWA # 
      4 0.00000e+00 #  ymc
      5 0.000000e+00 #  ymb
      6 1.645000e+02 #  ymt
      15 1.777000e+00 #  ymtau
###################################
## INFORMATION FOR DECAY
###################################
DECAY 6 0.000000e+00 #  wt
      5 24 0.99 #  branching ratio
      3 24 0.01 #  branching ratio

DECAY 23 2.441404e+00 #  wz
      5 -5 1 # 

DECAY 24 3.00e+00 #  ww
DECAY 25 2.441404e+00 #  wh
"""
        self.assertEqual(target.split('\n'), output.split('\n'))
        
        
        
        
        dict = self.main.read_param_card([l+'\n' for l in output.split('\n')])
        

        for key, item in solution.items():
            for key2, (value, comment) in item.items():
                self.assertEqual(value, float(dict[key].get(eval(key2)).value))       

 
    
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
    
    def test_make_valid(self):
        """ check that we can modify a param_card following a restriction"""

        # Load a model and a given restriction file
        full_card = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_sm.dat')
        
        restriction = """<file>######################################################################
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
        fsock = StringIO.StringIO()
        writter.make_valid_param_card(full_card, restriction, outputpath=fsock)
        output = fsock.getvalue()
        target = """######################################################################
## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####
######################################################################
###################################
## INFORMATION FOR MASS
###################################
BLOCK MASS # 
      15 1.777000e+00 #  mta
      6 1.743000e+02 #  mt
      5 0.0 #  mb fixed by the model
      23 9.118800e+01 #  mz
      25 91.188 #  mh must be identical to [23]
      11 0 # fixed by the model
      13 0 # fixed by the model
      2 0 # fixed by the model
      4 0 # fixed by the model
      1 0 # fixed by the model
      3 0 # fixed by the model
###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS # 
      1 1.325070e+02 #  aewm1
      2 1.166390e-05 #  gf
      3 1.180000e-01 #  as
###################################
## INFORMATION FOR YUKAWA
###################################
BLOCK YUKAWA # 
      5 0.0 #  ymb fixed by the model
      6 1.645000e+02 #  ymt
      15 1.777000e+00 #  ymtau
      4 0 # fixed by the model
###################################
## INFORMATION FOR DECAY
###################################
DECAY 6 0.0 #  fixed by the model
DECAY 23 2.441404e+00 # 
DECAY 24 2.047600e+00 # 
DECAY 25 2.441404 #  must be identical to [23]
DECAY 15 0 # fixed by the model
###################################
## INFORMATION FOR CKMBLOCK
###################################
BLOCK CKMBLOCK # 
      1 0 # fixed by the model
"""
      
        self.assertEqual(output.split('\n'), target.split('\n'))
        
        
        
        
        
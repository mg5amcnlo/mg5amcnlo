################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
import models.check_param_card as writter


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
BLOCK SMINPUTS Q= 1.000000e+00 #  test

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
                         
        self.assertRaises(writter.InvalidParamCard, 
           b.append, writter.Parameter(block='sminputs', lhacode=[1,2], value=9))
        self.assertEqual(len(b),2)
        
        
        b.remove([1,2])
        self.assertEqual(len(b),1)
        self.assertEqual(b.param_dict.keys(),[(1,)])               


class TestParamCard(unittest.TestCase):
    """ Test the ParamCard Object """

    def test_mod_param(self):
        """ test that we can modify a param card """

        full_card = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_sm.dat')        
        card = writter.ParamCard(full_card)
        
        self.assertFalse(card.has_param('mass', [999]))
        self.assertFalse(card.has_param('new', [24]))
        self.assertFalse(card.has_param('new', [23]))
        card.copy_param('mass',[23], 'new', [24])
        self.assertFalse(card.has_param('mass', [24]))
        self.assertTrue(card.has_param('mass', [23]))
        self.assertTrue(card.has_param('new', [24]))
        self.assertFalse(card.has_param('new', [23]))
        card.copy_param('mass',[23], 'new')
        card.copy_param('mass',[23], lhacode=[999])
        
        self.assertEqual(len(card['new']), 2)
        self.assertTrue(card.has_param('mass', [999]))                        
        self.assertTrue(card.has_param('new', [24]))
        self.assertTrue(card.has_param('new', [23]))
        self.assertTrue(card.has_param('mass', [23]))
        
        card.remove_param('new', [23])
        card.remove_param('new', [24])
        card.remove_param('mass',[999])
        
        self.assertFalse(card.has_param('mass', [999]))
        self.assertFalse(card.has_param('new', [24]))
        self.assertFalse(card.has_param('new', [23]))
        self.assertTrue(card.has_param('mass', [23]))
        
        self.assertFalse('new' in card.keys())
        
        card.mod_param('mass', [23], 'new', [25], 43)
        card.mod_param('decay', [23], 'new', [26], 43)
        
        self.assertEqual(len(card['new']), 2)
        self.assertTrue(card.has_param('new', [25]))                        
        self.assertTrue(card.has_param('new', [26]))
        self.assertFalse(card.has_param('new', [23]))
        self.assertFalse(card.has_param('mass', [23]))
        self.assertEqual(card['new'].get([25]).value, 43)
        
        card.mod_param('new', [25], 'mass', [23])
        card.mod_param('new', [26], 'decay', [23])       
                
        self.assertFalse('new' in card.keys())       
        
                

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
        self.assertEqual(param.lhablock, 'mass')
        
       
        # change the block of a parameter and lhacode
        card.mod_param('mass', [32], block='polemass', lhacode=[35])
        
        self.assertFalse(card.has_key('mass'))
        self.assertRaises(KeyError, card['polemass'].get, [32])
        self.assertRaises(KeyError, card['width'].get, [32])
        self.assertEqual(param, card['polemass'].get([35]))
        self.assertEqual(param.lhacode, [35])
        self.assertEqual(param.lhablock, 'polemass')        
        
        # change the value / comment
        card.mod_param('polemass', [35], value=2, comment='new')
        self.assertEqual(param.value, 2)
        self.assertEqual(param.comment, 'new')
        self.assertRaises(writter.InvalidParamCard, card.mod_param, 
                                             *('polemass', [35], 'width', [24]))
        
    
    
    
class TestParamCardIterator(unittest.TestCase):
    """ Test the ParamCard Object """
    
    def test_paramcard_scan(self):
        full_card = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'param_card_sm.dat')        
        card = writter.ParamCard(full_card)
        
        # create a simple 1D scan
        card['mass'].get(6).value = "scan:[1,2,3,4,5]"
        mh = card['mass'].get(25).value # to check that param_card are independant
        
        itercard = writter.ParamCardIterator(card)
        card['mass'].get(25).value = 25.0
        for i, new_card in enumerate(itercard):
            self.assertEqual(new_card['mass'].get(6).value, i+1)
            self.assertEqual(new_card['mass'].get(25).value, mh)
        self.assertEqual(i, 4)
        
        # create a 1D scan with two parameter
        card['mass'].get(6).value = "scan1:[1,2,3,4,5]"
        card['mass'].get(25).value = "scan1:[0,10,20,30,40]"
        itercard = writter.ParamCardIterator(card)
        for i, new_card in enumerate(itercard):
            self.assertEqual(new_card['mass'].get(6).value, i+1)
            self.assertEqual(new_card['mass'].get(25).value, 10*i)
        self.assertEqual(i, 4)
        
        # create a 2D scan with two parameter
        card['mass'].get(6).value = "scan:[1,2,3]"
        card['mass'].get(25).value = "scan:[0,10,20,30]"
        all_possibilities = [(1,0), (1,10),(1,20),(1,30),
                             (2,0), (2,10),(2,20),(2,30),
                             (3,0), (3,10),(3,20),(3,30)
                             ]
        itercard = writter.ParamCardIterator(card)
        for i, new_card in enumerate(itercard):
            choice = (new_card['mass'].get(6).value, new_card['mass'].get(25).value)
            self.assertIn(choice, all_possibilities)
            all_possibilities.remove(choice)
            
        self.assertEqual(i, 11)                    
        self.assertFalse(all_possibilities)
        
         
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
<opposite>
</opposite>
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
                    'wolfenstein': {'[1]': (0.0, 'cabi'),
                                    '[2]': (0.0, 'cabi'),
                                    '[3]': (0.0, 'cabi'),
                                    '[4]': (0.0, 'cabi')}, 
                    'sminputs': {'[3]': (0.118, 'as'), 
                                 '[1]': (132.507, 'aewm1'), 
                                 '[2]': (1.16639e-05, 'gf')}, 
                    'mass': {'[13]': (0.0, 'mm'), 
                             '[23]': (91.188, 'mz'), 
                             '[15]': (1.777, 'mta'), 
                             '[11]': (0.0, 'me'), 
                             '[6]': (172.0, 'mt'), 
                             '[25]': (91.188, 'mh'), 
                             '[4]': (0.0, 'mc'), 
                             '[5]': (0.0, 'mb'), }, 
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
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS # 
      1 1.325070e+02 # aewm1
      2 1.166390e-05 # gf
      3 1.180000e-01 # as
###################################
## INFORMATION FOR MASS
###################################
BLOCK MASS # 
      4 0.000000e+00 # mc
      5 0.000000e+00 # mb
      6 1.720000e+02 # mt
      11 0.000000e+00 # me
      13 0.000000e+00 # mm
      15 1.777000e+00 # mta
      23 9.118800e+01 # mz
      25 9.118800e+01 # mh
###################################
## INFORMATION FOR WOLFENSTEIN
###################################
BLOCK WOLFENSTEIN # 
      1 0.000000e+00 # lamws
      2 0.000000e+00 # aws
      3 0.000000e+00 # rhows
      4 0.000000e+00 # etaws
###################################
## INFORMATION FOR YUKAWA
###################################
BLOCK YUKAWA # 
      4 0.000000e+00 # ymc
      5 0.000000e+00 # ymb
      6 1.645000e+02 # ymt
      11 0.000000e+00 # yme
      13 0.000000e+00 # ymm
      15 1.777000e+00 # ymtau
###################################
## INFORMATION FOR DECAY
###################################
DECAY 6 0.000000e+00 # wt
      9.900000e-01 2 5 24 # branching ratio
      1.000000e-02 2 3 24 # branching ratio

DECAY 15 0.000000e+00 # 
DECAY 23 2.441404e+00 # wz
      1.000000e+00 2 -5 5 # 

DECAY 24 3.000000e+00 # ww
DECAY 25 2.441404e+00 # wh
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


        target_zero =[('wolfenstein', [1], ''),
                      ('wolfenstein', [2], ''), 
                      ('wolfenstein', [3], ''), 
                      ('wolfenstein', [4], ''), 
                      ('yukawa', [4], ''), 
                      ('yukawa', [5], ''), 
                      ('yukawa', [11], ''), 
                      ('yukawa', [13], ''), 
                      ('mass', [4], ''), 
                      ('mass', [5], ''), 
                      ('mass', [11], ''), 
                      ('mass', [13], ''), 
                      ('decay', [6], ''), 
                      ('decay', [15], '')]

        self.assertEqual(base_model.rule_card.zero, target_zero)
        target_one = []
        self.assertEqual(base_model.rule_card.one, target_one)
        target_identical = [('mass', [25], [23], '')]                
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
     wolfenstein 1 # 
     wolfenstein 2 # 
     wolfenstein 3 # 
     wolfenstein 4 # 
     yukawa 4 # 
     yukawa 5 # 
     yukawa 11 # 
     yukawa 13 # 
     mass 4 # 
     mass 5 # 
     mass 11 # 
     mass 13 # 
     decay 6 # 
     decay 15 # 
</zero>
<one>
</one>
<identical>
     mass 25 : 23 # 
</identical>
<opposite>
</opposite>
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
      15 1.777000e+00 # mta
      6 1.743000e+02 # mt
      5 0.000000e+00 # mb fixed by the model
      23 9.118800e+01 # mz
      25 9.118800e+01 # mh must be identical to [23]
      11 0.000000e+00 # fixed by the model
      13 0.000000e+00 # fixed by the model
      2 0.000000e+00 # fixed by the model
      4 0.000000e+00 # fixed by the model
      1 0.000000e+00 # fixed by the model
      3 0.000000e+00 # fixed by the model
###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS # 
      1 1.325070e+02 # aewm1
      2 1.166390e-05 # gf
      3 1.180000e-01 # as
###################################
## INFORMATION FOR YUKAWA
###################################
BLOCK YUKAWA # 
      5 0.000000e+00 # ymb fixed by the model
      6 1.645000e+02 # ymt
      15 1.777000e+00 # ymtau
      4 0.000000e+00 # fixed by the model
###################################
## INFORMATION FOR DECAY
###################################
DECAY 6 0.000000e+00 # fixed by the model
DECAY 23 2.441404e+00 # 
DECAY 24 2.047600e+00 # 
DECAY 25 2.441404e+00 # must be identical to [23]
DECAY 15 0.000000e+00 # fixed by the model
###################################
## INFORMATION FOR CKMBLOCK
###################################
BLOCK CKMBLOCK # 
      1 0.000000e+00 # fixed by the model
"""
      
        self.assertEqual(output.split('\n'), target.split('\n'))
        
        
class TestConvertSLAH(unittest.TestCase):
    """ Test the ParamCardRule Object"""
    
    sps1a = os.path.join(_file_path, os.path.pardir,
                                     'input_files', 'sps1a_param_card.dat')
    
    output = '/tmp/mg5param.dat'
     
     
    def tearDown(self):
        
        if os.path.exists(self.output):
            os.remove(self.output)
        
    def test_convert_to_mg5(self):
        """take the slah1 and convert it to mg5"""

        target = """######################################################################
## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####
######################################################################
###################################
## INFORMATION FOR DCINFO
###################################
BLOCK DCINFO #  decay program information
      1 sdecay # decay calculator
      2 1.1a # version number
###################################
## INFORMATION FOR SPINFO
###################################
BLOCK SPINFO #  spectrum calculator information
      1 softsusy # spectrum calculator
      2 2.0.5 # version number
###################################
## INFORMATION FOR MODSEL
###################################
BLOCK MODSEL #  model selection
      1 1 sugra # 
      2 1 # fake line for the formating line point of view
###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS #  standard model inputs
      1 1.279340e+02 # alpha_em^-1(m_z)^msbar
      3 1.180000e-01 # alpha_s(m_z)^msbar
      5 4.250000e+00 # mb(mb)^msbar
###################################
## INFORMATION FOR MINPAR
###################################
BLOCK MINPAR #  input parameters - minimal models
      1 1.000000e+02 # m0
      2 2.500000e+02 # m12
      3 1.000000e+01 # tanb
      4 1.000000e+00 # sign(mu)
      5 -1.000000e+02 # a0
###################################
## INFORMATION FOR MASS
###################################
BLOCK MASS #  mass spectrum
      5 4.889917e+00 # b-quark pole mass calculated from mb(mb)_msbar
      6 1.750000e+02 # mt pole mass (not read by me)
      24 7.982901e+01 # w+
      25 1.108991e+02 # h
      35 3.999601e+02 # h
      36 3.995839e+02 # a
      37 4.078790e+02 # h+
      1000001 5.684411e+02 # ~d_l
      2000001 5.452285e+02 # ~d_r
      1000002 5.611190e+02 # ~u_l
      2000002 5.492593e+02 # ~u_r
      1000003 5.684411e+02 # ~s_l
      2000003 5.452285e+02 # ~s_r
      1000004 5.611190e+02 # ~c_l
      2000004 5.492593e+02 # ~c_r
      1000005 5.130652e+02 # ~b_1
      2000005 5.437267e+02 # ~b_2
      1000006 3.996685e+02 # ~t_1
      2000006 5.857858e+02 # ~t_2
      1000011 2.029157e+02 # ~e_l
      2000011 1.441028e+02 # ~e_r
      1000012 1.852583e+02 # ~nu_el
      1000013 2.029157e+02 # ~mu_l
      2000013 1.441028e+02 # ~mu_r
      1000014 1.852583e+02 # ~nu_mul
      1000015 1.344909e+02 # ~tau_1
      2000015 2.068678e+02 # ~tau_2
      1000016 1.847085e+02 # ~nu_taul
      1000021 6.077137e+02 # ~g
      1000022 9.668807e+01 # ~chi_10
      1000023 1.810882e+02 # ~chi_20
      1000025 -3.637560e+02 # ~chi_30
      1000035 3.817294e+02 # ~chi_40
      1000024 1.816965e+02 # ~chi_1+
      1000037 3.799393e+02 # ~chi_2+
###################################
## INFORMATION FOR NMIX
###################################
BLOCK NMIX #  neutralino mixing matrix
      1 1 9.863644e-01 # n_11
      1 2 -5.311036e-02 # n_12
      1 3 1.464340e-01 # n_13
      1 4 -5.311861e-02 # n_14
      2 1 9.935054e-02 # n_21
      2 2 9.449493e-01 # n_22
      2 3 -2.698467e-01 # n_23
      2 4 1.561507e-01 # n_24
      3 1 -6.033880e-02 # n_31
      3 2 8.770049e-02 # n_32
      3 3 6.958775e-01 # n_33
      3 4 7.102270e-01 # n_34
      4 1 -1.165071e-01 # n_41
      4 2 3.107390e-01 # n_42
      4 3 6.492260e-01 # n_43
      4 4 -6.843778e-01 # n_44
###################################
## INFORMATION FOR UMIX
###################################
BLOCK UMIX #  chargino mixing matrix u
      1 1 9.168349e-01 # u_11
      1 2 -3.992666e-01 # u_12
      2 1 3.992666e-01 # u_21
      2 2 9.168349e-01 # u_22
###################################
## INFORMATION FOR VMIX
###################################
BLOCK VMIX #  chargino mixing matrix v
      1 1 9.725578e-01 # v_11
      1 2 -2.326612e-01 # v_12
      2 1 2.326612e-01 # v_21
      2 2 9.725578e-01 # v_22
###################################
## INFORMATION FOR HMIX
###################################
BLOCK HMIX Q= 4.670342e+02 #  drbar higgs parameters
      1 3.576810e+02 # mu(q)mssm drbar
      2 9.748624e+00 # tan beta(q)mssm drba
      4 1.664391e+05 # ma^2(q)mssm drbar
###################################
## INFORMATION FOR GAUGE
###################################
BLOCK GAUGE Q= 4.670342e+02 #  the gauge couplings
      3 1.101787e+00 # g3(q) msbar
###################################
## INFORMATION FOR YU
###################################
BLOCK YU Q= 4.670342e+02 #  the yukawa couplings
      3 3 8.928445e-01 # y_t(q) drbar
###################################
## INFORMATION FOR YD
###################################
BLOCK YD Q= 4.670342e+02 #  the yukawa couplings
      3 3 1.388402e-01 # y_b(q) drbar
###################################
## INFORMATION FOR YE
###################################
BLOCK YE Q= 4.670342e+02 #  the yukawa couplings
      3 3 1.008908e-01 # y_tau(q) drbar
###################################
## INFORMATION FOR MSOFT
###################################
BLOCK MSOFT Q= 4.670342e+02 #  the soft susy breaking masses at the scale q
      1 1.013965e+02 # m_1(q)
      2 1.915042e+02 # m_2(q)
      3 5.882630e+02 # m_3(q)
      21 3.233749e+04 # mh1^2(q)
      22 -1.288001e+05 # mh2^2(q)
###################################
## INFORMATION FOR DECAY
###################################
DECAY 23 2.411433e+00 # z width (sm calculation)
DECAY 24 2.002822e+00 # w width (sm calculation)
DECAY 6 1.561950e+00 # top decays
      1.000000e+00 2 5 24 # br(t ->  b    w+)
      0.000000e+00 2 5 37 # br(t ->  b    h+)
      0.000000e+00 2 1000006 1000022 # br(t -> ~t_1 ~chi_10)
      0.000000e+00 2 1000006 1000023 # br(t -> ~t_1 ~chi_20)
      0.000000e+00 2 1000006 1000025 # br(t -> ~t_1 ~chi_30)
      0.000000e+00 2 1000006 1000035 # br(t -> ~t_1 ~chi_40)
      0.000000e+00 2 1000022 2000006 # br(t -> ~t_2 ~chi_10)
      0.000000e+00 2 1000023 2000006 # br(t -> ~t_2 ~chi_20)
      0.000000e+00 2 1000025 2000006 # br(t -> ~t_2 ~chi_30)
      0.000000e+00 2 1000035 2000006 # br(t -> ~t_2 ~chi_40)

DECAY 25 1.986108e-03 # h decays
      1.456430e-01 2 -15 15 # br(h1 -> tau- tau+)
      8.190707e-01 2 -5 5 # br(h1 -> b bb)
      3.363382e-02 2 -24 24 # br(h1 -> w+ w-)
      1.652515e-03 2 23 23 # br(h1 -> z z)

DECAY 35 5.748014e-01 # h decays
      1.390727e-01 2 -15 15 # br(h -> tau- tau+)
      4.841109e-02 2 -6 6 # br(h -> t tb)
      7.895001e-01 2 -5 5 # br(h -> b bb)
      3.876812e-03 2 -24 24 # br(h -> w+ w-)
      1.804548e-03 2 23 23 # br(h -> z z)
      0.000000e+00 2 -37 24 # br(h -> w+ h-)
      0.000000e+00 2 -24 37 # br(h -> w- h+)
      0.000000e+00 2 -37 37 # br(h -> h+ h-)
      1.733481e-02 2 25 25 # br(h -> h h)
      0.000000e+00 2 36 36 # br(h -> a a)

DECAY 36 6.321785e-01 # a decays
      1.266597e-01 2 -15 15 # br(a -> tau- tau+)
      1.510815e-01 2 -6 6 # br(a -> t tb)
      7.194061e-01 2 -5 5 # br(a -> b bb)
      2.852612e-03 2 23 25 # br(a -> z h)
      0.000000e+00 2 23 35 # br(a -> z h)
      0.000000e+00 2 -37 24 # br(a -> w+ h-)
      0.000000e+00 2 -24 37 # br(a -> w- h+)

DECAY 37 5.469628e-01 # h+ decays
      1.494351e-01 2 -15 16 # br(h+ -> tau+ nu_tau)
      8.468117e-01 2 -5 6 # br(h+ -> t bb)
      3.753154e-03 2 24 25 # br(h+ -> w+ h)
      0.000000e+00 2 24 35 # br(h+ -> w+ h)
      0.000000e+00 2 24 36 # br(h+ -> w+ a)

DECAY 1000021 5.506754e+00 # gluino decays
      2.084542e-02 2 -1 1000001 # br(~g -> ~d_l  db)
      2.084542e-02 2 -1000001 1 # br(~g -> ~d_l* d )
      5.070753e-02 2 -1 2000001 # br(~g -> ~d_r  db)
      5.070753e-02 2 -2000001 1 # br(~g -> ~d_r* d )
      2.897878e-02 2 -2 1000002 # br(~g -> ~u_l  ub)
      2.897878e-02 2 -1000002 2 # br(~g -> ~u_l* u )
      4.468728e-02 2 -2 2000002 # br(~g -> ~u_r  ub)
      4.468728e-02 2 -2000002 2 # br(~g -> ~u_r* u )
      2.084542e-02 2 -3 1000003 # br(~g -> ~s_l  sb)
      2.084542e-02 2 -1000003 3 # br(~g -> ~s_l* s )
      5.070753e-02 2 -3 2000003 # br(~g -> ~s_r  sb)
      5.070753e-02 2 -2000003 3 # br(~g -> ~s_r* s )
      2.897878e-02 2 -4 1000004 # br(~g -> ~c_l  cb)
      2.897878e-02 2 -1000004 4 # br(~g -> ~c_l* c )
      4.468728e-02 2 -4 2000004 # br(~g -> ~c_r  cb)
      4.468728e-02 2 -2000004 4 # br(~g -> ~c_r* c )
      1.058402e-01 2 -5 1000005 # br(~g -> ~b_1  bb)
      1.058402e-01 2 -1000005 5 # br(~g -> ~b_1* b )
      5.565748e-02 2 -5 2000005 # br(~g -> ~b_2  bb)
      5.565748e-02 2 -2000005 5 # br(~g -> ~b_2* b )
      4.806428e-02 2 -6 1000006 # br(~g -> ~t_1  tb)
      4.806428e-02 2 -1000006 6 # br(~g -> ~t_1* t )
      0.000000e+00 2 -6 2000006 # br(~g -> ~t_2  tb)
      0.000000e+00 2 -2000006 6 # br(~g -> ~t_2* t )

DECAY 1000006 2.021596e+00 # stop1 decays
      1.929476e-01 2 6 1000022 # br(~t_1 -> ~chi_10 t )
      1.174692e-01 2 6 1000023 # br(~t_1 -> ~chi_20 t )
      0.000000e+00 2 6 1000025 # br(~t_1 -> ~chi_30 t )
      0.000000e+00 2 6 1000035 # br(~t_1 -> ~chi_40 t )
      6.757477e-01 2 5 1000024 # br(~t_1 -> ~chi_1+ b )
      1.383548e-02 2 5 1000037 # br(~t_1 -> ~chi_2+ b )
      0.000000e+00 2 6 1000021 # br(~t_1 -> ~g      t )
      0.000000e+00 2 37 1000005 # br(~t_1 -> ~b_1    h+)
      0.000000e+00 2 37 2000005 # br(~t_1 -> ~b_2    h+)
      0.000000e+00 2 24 1000005 # br(~t_1 -> ~b_1    w+)
      0.000000e+00 2 24 2000005 # br(~t_1 -> ~b_2    w+)

DECAY 2000006 7.373133e+00 # stop2 decays
      2.968256e-02 2 6 1000022 # br(~t_2 -> ~chi_10 t )
      8.680354e-02 2 6 1000023 # br(~t_2 -> ~chi_20 t )
      4.184084e-02 2 6 1000025 # br(~t_2 -> ~chi_30 t )
      1.932816e-01 2 6 1000035 # br(~t_2 -> ~chi_40 t )
      2.196324e-01 2 5 1000024 # br(~t_2 -> ~chi_1+ b )
      2.022061e-01 2 5 1000037 # br(~t_2 -> ~chi_2+ b )
      0.000000e+00 2 6 1000021 # br(~t_2 -> ~g      t )
      3.663977e-02 2 25 1000006 # br(~t_2 -> ~t_1    h )
      0.000000e+00 2 35 1000006 # br(~t_2 -> ~t_1    h )
      0.000000e+00 2 36 1000006 # br(~t_2 -> ~t_1    a )
      0.000000e+00 2 37 1000005 # br(~t_2 -> ~b_1    h+)
      0.000000e+00 2 37 2000005 # br(~t_2 -> ~b_2    h+)
      1.899131e-01 2 23 1000006 # br(~t_2 -> ~t_1    z )
      0.000000e+00 2 24 1000005 # br(~t_2 -> ~b_1    w+)
      0.000000e+00 2 24 2000005 # br(~t_2 -> ~b_2    w+)

DECAY 1000005 3.736276e+00 # sbottom1 decays
      4.433071e-02 2 5 1000022 # br(~b_1 -> ~chi_10 b )
      3.563199e-01 2 5 1000023 # br(~b_1 -> ~chi_20 b )
      5.160838e-03 2 5 1000025 # br(~b_1 -> ~chi_30 b )
      1.041051e-02 2 5 1000035 # br(~b_1 -> ~chi_40 b )
      4.458301e-01 2 -1000024 6 # br(~b_1 -> ~chi_1- t )
      0.000000e+00 2 -1000037 6 # br(~b_1 -> ~chi_2- t )
      0.000000e+00 2 5 1000021 # br(~b_1 -> ~g      b )
      0.000000e+00 2 -37 1000006 # br(~b_1 -> ~t_1    h-)
      0.000000e+00 2 -37 2000006 # br(~b_1 -> ~t_2    h-)
      1.379480e-01 2 -24 1000006 # br(~b_1 -> ~t_1    w-)
      0.000000e+00 2 -24 2000006 # br(~b_1 -> ~t_2    w-)

DECAY 2000005 8.015663e-01 # sbottom2 decays
      2.862006e-01 2 5 1000022 # br(~b_2 -> ~chi_10 b )
      1.403159e-01 2 5 1000023 # br(~b_2 -> ~chi_20 b )
      5.326356e-02 2 5 1000025 # br(~b_2 -> ~chi_30 b )
      7.487481e-02 2 5 1000035 # br(~b_2 -> ~chi_40 b )
      1.797343e-01 2 -1000024 6 # br(~b_2 -> ~chi_1- t )
      0.000000e+00 2 -1000037 6 # br(~b_2 -> ~chi_2- t )
      0.000000e+00 2 5 1000021 # br(~b_2 -> ~g      b )
      0.000000e+00 2 25 1000005 # br(~b_2 -> ~b_1    h )
      0.000000e+00 2 35 1000005 # br(~b_2 -> ~b_1    h )
      0.000000e+00 2 36 1000005 # br(~b_2 -> ~b_1    a )
      0.000000e+00 2 -37 1000006 # br(~b_2 -> ~t_1    h-)
      0.000000e+00 2 -37 2000006 # br(~b_2 -> ~t_2    h-)
      0.000000e+00 2 23 1000005 # br(~b_2 -> ~b_1    z )
      2.656108e-01 2 -24 1000006 # br(~b_2 -> ~t_1    w-)
      0.000000e+00 2 -24 2000006 # br(~b_2 -> ~t_2    w-)

DECAY 1000002 5.477195e+00 # sup_l decays
      6.652410e-03 2 2 1000022 # br(~u_l -> ~chi_10 u)
      3.190515e-01 2 2 1000023 # br(~u_l -> ~chi_20 u)
      8.449291e-04 2 2 1000025 # br(~u_l -> ~chi_30 u)
      1.034852e-02 2 2 1000035 # br(~u_l -> ~chi_40 u)
      6.494995e-01 2 1 1000024 # br(~u_l -> ~chi_1+ d)
      1.360317e-02 2 1 1000037 # br(~u_l -> ~chi_2+ d)
      0.000000e+00 2 2 1000021 # br(~u_l -> ~g      u)

DECAY 2000002 1.152973e+00 # sup_r decays
      9.863774e-01 2 2 1000022 # br(~u_r -> ~chi_10 u)
      8.466406e-03 2 2 1000023 # br(~u_r -> ~chi_20 u)
      1.238947e-03 2 2 1000025 # br(~u_r -> ~chi_30 u)
      3.917226e-03 2 2 1000035 # br(~u_r -> ~chi_40 u)
      0.000000e+00 2 1 1000024 # br(~u_r -> ~chi_1+ d)
      0.000000e+00 2 1 1000037 # br(~u_r -> ~chi_2+ d)
      0.000000e+00 2 2 1000021 # br(~u_r -> ~g      u)

DECAY 1000001 5.312788e+00 # sdown_l decays
      2.323180e-02 2 1 1000022 # br(~d_l -> ~chi_10 d)
      3.102351e-01 2 1 1000023 # br(~d_l -> ~chi_20 d)
      1.523348e-03 2 1 1000025 # br(~d_l -> ~chi_30 d)
      1.488498e-02 2 1 1000035 # br(~d_l -> ~chi_40 d)
      6.064525e-01 2 -1000024 2 # br(~d_l -> ~chi_1- u)
      4.367232e-02 2 -1000037 2 # br(~d_l -> ~chi_2- u)
      0.000000e+00 2 1 1000021 # br(~d_l -> ~g      d)

DECAY 2000001 2.858123e-01 # sdown_r decays
      9.865296e-01 2 1 1000022 # br(~d_r -> ~chi_10 d)
      8.445104e-03 2 1 1000023 # br(~d_r -> ~chi_20 d)
      1.211721e-03 2 1 1000025 # br(~d_r -> ~chi_30 d)
      3.813561e-03 2 1 1000035 # br(~d_r -> ~chi_40 d)
      0.000000e+00 2 -1000024 2 # br(~d_r -> ~chi_1- u)
      0.000000e+00 2 -1000037 2 # br(~d_r -> ~chi_2- u)
      0.000000e+00 2 1 1000021 # br(~d_r -> ~g      d)

DECAY 1000004 5.477195e+00 # scharm_l decays
      6.652410e-03 2 4 1000022 # br(~c_l -> ~chi_10 c)
      3.190515e-01 2 4 1000023 # br(~c_l -> ~chi_20 c)
      8.449291e-04 2 4 1000025 # br(~c_l -> ~chi_30 c)
      1.034852e-02 2 4 1000035 # br(~c_l -> ~chi_40 c)
      6.494995e-01 2 3 1000024 # br(~c_l -> ~chi_1+ s)
      1.360317e-02 2 3 1000037 # br(~c_l -> ~chi_2+ s)
      0.000000e+00 2 4 1000021 # br(~c_l -> ~g      c)

DECAY 2000004 1.152973e+00 # scharm_r decays
      9.863774e-01 2 4 1000022 # br(~c_r -> ~chi_10 c)
      8.466406e-03 2 4 1000023 # br(~c_r -> ~chi_20 c)
      1.238947e-03 2 4 1000025 # br(~c_r -> ~chi_30 c)
      3.917226e-03 2 4 1000035 # br(~c_r -> ~chi_40 c)
      0.000000e+00 2 3 1000024 # br(~c_r -> ~chi_1+ s)
      0.000000e+00 2 3 1000037 # br(~c_r -> ~chi_2+ s)
      0.000000e+00 2 4 1000021 # br(~c_r -> ~g      c)

DECAY 1000003 5.312788e+00 # sstrange_l decays
      2.323180e-02 2 3 1000022 # br(~s_l -> ~chi_10 s)
      3.102351e-01 2 3 1000023 # br(~s_l -> ~chi_20 s)
      1.523348e-03 2 3 1000025 # br(~s_l -> ~chi_30 s)
      1.488498e-02 2 3 1000035 # br(~s_l -> ~chi_40 s)
      6.064525e-01 2 -1000024 4 # br(~s_l -> ~chi_1- c)
      4.367232e-02 2 -1000037 4 # br(~s_l -> ~chi_2- c)
      0.000000e+00 2 3 1000021 # br(~s_l -> ~g      s)

DECAY 2000003 2.858123e-01 # sstrange_r decays
      9.865296e-01 2 3 1000022 # br(~s_r -> ~chi_10 s)
      8.445104e-03 2 3 1000023 # br(~s_r -> ~chi_20 s)
      1.211721e-03 2 3 1000025 # br(~s_r -> ~chi_30 s)
      3.813561e-03 2 3 1000035 # br(~s_r -> ~chi_40 s)
      0.000000e+00 2 -1000024 4 # br(~s_r -> ~chi_1- c)
      0.000000e+00 2 -1000037 4 # br(~s_r -> ~chi_2- c)
      0.000000e+00 2 3 1000021 # br(~s_r -> ~g      s)

DECAY 1000011 2.136822e-01 # selectron_l decays
      5.731554e-01 2 11 1000022 # br(~e_l -> ~chi_10 e-)
      1.645226e-01 2 11 1000023 # br(~e_l -> ~chi_20 e-)
      0.000000e+00 2 11 1000025 # br(~e_l -> ~chi_30 e-)
      0.000000e+00 2 11 1000035 # br(~e_l -> ~chi_40 e-)
      2.623220e-01 2 -1000024 12 # br(~e_l -> ~chi_1- nu_e)
      0.000000e+00 2 -1000037 12 # br(~e_l -> ~chi_2- nu_e)

DECAY 2000011 2.161216e-01 # selectron_r decays
      1.000000e+00 2 11 1000022 # br(~e_r -> ~chi_10 e-)
      0.000000e+00 2 11 1000023 # br(~e_r -> ~chi_20 e-)
      0.000000e+00 2 11 1000025 # br(~e_r -> ~chi_30 e-)
      0.000000e+00 2 11 1000035 # br(~e_r -> ~chi_40 e-)
      0.000000e+00 2 -1000024 12 # br(~e_r -> ~chi_1- nu_e)
      0.000000e+00 2 -1000037 12 # br(~e_r -> ~chi_2- nu_e)

DECAY 1000013 2.136822e-01 # smuon_l decays
      5.731554e-01 2 13 1000022 # br(~mu_l -> ~chi_10 mu-)
      1.645226e-01 2 13 1000023 # br(~mu_l -> ~chi_20 mu-)
      0.000000e+00 2 13 1000025 # br(~mu_l -> ~chi_30 mu-)
      0.000000e+00 2 13 1000035 # br(~mu_l -> ~chi_40 mu-)
      2.623220e-01 2 -1000024 14 # br(~mu_l -> ~chi_1- nu_mu)
      0.000000e+00 2 -1000037 14 # br(~mu_l -> ~chi_2- nu_mu)

DECAY 2000013 2.161216e-01 # smuon_r decays
      1.000000e+00 2 13 1000022 # br(~mu_r -> ~chi_10 mu-)
      0.000000e+00 2 13 1000023 # br(~mu_r -> ~chi_20 mu-)
      0.000000e+00 2 13 1000025 # br(~mu_r -> ~chi_30 mu-)
      0.000000e+00 2 13 1000035 # br(~mu_r -> ~chi_40 mu-)
      0.000000e+00 2 -1000024 14 # br(~mu_r -> ~chi_1- nu_mu)
      0.000000e+00 2 -1000037 14 # br(~mu_r -> ~chi_2- nu_mu)

DECAY 1000015 1.483273e-01 # stau_1 decays
      1.000000e+00 2 15 1000022 # br(~tau_1 -> ~chi_10  tau-)
      0.000000e+00 2 15 1000023 # br(~tau_1 -> ~chi_20  tau-)
      0.000000e+00 2 15 1000025 # br(~tau_1 -> ~chi_30  tau-)
      0.000000e+00 2 15 1000035 # br(~tau_1 -> ~chi_40  tau-)
      0.000000e+00 2 -1000024 16 # br(~tau_1 -> ~chi_1-  nu_tau)
      0.000000e+00 2 -1000037 16 # br(~tau_1 -> ~chi_2-  nu_tau)
      0.000000e+00 2 -37 1000016 # br(~tau_1 -> ~nu_taul h-)
      0.000000e+00 2 -24 1000016 # br(~tau_1 -> ~nu_taul w-)

DECAY 2000015 2.699061e-01 # stau_2 decays
      5.966530e-01 2 15 1000022 # br(~tau_2 -> ~chi_10  tau-)
      1.545368e-01 2 15 1000023 # br(~tau_2 -> ~chi_20  tau-)
      0.000000e+00 2 15 1000025 # br(~tau_2 -> ~chi_30  tau-)
      0.000000e+00 2 15 1000035 # br(~tau_2 -> ~chi_40  tau-)
      2.488102e-01 2 -1000024 16 # br(~tau_2 -> ~chi_1-  nu_tau)
      0.000000e+00 2 -1000037 16 # br(~tau_2 -> ~chi_2-  nu_tau)
      0.000000e+00 2 -37 1000016 # br(~tau_2 -> ~nu_taul h-)
      0.000000e+00 2 -24 1000016 # br(~tau_2 -> ~nu_taul w-)
      0.000000e+00 2 25 1000015 # br(~tau_2 -> ~tau_1 h)
      0.000000e+00 2 35 1000015 # br(~tau_2 -> ~tau_1 h)
      0.000000e+00 2 36 1000015 # br(~tau_2 -> ~tau_1 a)
      0.000000e+00 2 23 1000015 # br(~tau_2 -> ~tau_1 z)

DECAY 1000012 1.498816e-01 # snu_el decays
      9.777008e-01 2 12 1000022 # br(~nu_el -> ~chi_10 nu_e)
      8.115549e-03 2 12 1000023 # br(~nu_el -> ~chi_20 nu_e)
      0.000000e+00 2 12 1000025 # br(~nu_el -> ~chi_30 nu_e)
      0.000000e+00 2 12 1000035 # br(~nu_el -> ~chi_40 nu_e)
      1.418369e-02 2 11 1000024 # br(~nu_el -> ~chi_1+ e-)
      0.000000e+00 2 11 1000037 # br(~nu_el -> ~chi_2+ e-)

DECAY 1000014 1.498816e-01 # snu_mul decays
      9.777008e-01 2 14 1000022 # br(~nu_mul -> ~chi_10 nu_mu)
      8.115549e-03 2 14 1000023 # br(~nu_mul -> ~chi_20 nu_mu)
      0.000000e+00 2 14 1000025 # br(~nu_mul -> ~chi_30 nu_mu)
      0.000000e+00 2 14 1000035 # br(~nu_mul -> ~chi_40 nu_mu)
      1.418369e-02 2 13 1000024 # br(~nu_mul -> ~chi_1+ mu-)
      0.000000e+00 2 13 1000037 # br(~nu_mul -> ~chi_2+ mu-)

DECAY 1000016 1.475190e-01 # snu_taul decays
      9.859945e-01 2 16 1000022 # br(~nu_taul -> ~chi_10 nu_tau)
      6.251296e-03 2 16 1000023 # br(~nu_taul -> ~chi_20 nu_tau)
      0.000000e+00 2 16 1000025 # br(~nu_taul -> ~chi_30 nu_tau)
      0.000000e+00 2 16 1000035 # br(~nu_taul -> ~chi_40 nu_tau)
      7.754175e-03 2 15 1000024 # br(~nu_taul -> ~chi_1+ tau-)
      0.000000e+00 2 15 1000037 # br(~nu_taul -> ~chi_2+ tau-)
      0.000000e+00 2 -1000015 -37 # br(~nu_taul -> ~tau_1+ h-)
      0.000000e+00 2 -2000015 -37 # br(~nu_taul -> ~tau_2+ h-)
      0.000000e+00 2 -1000015 -24 # br(~nu_taul -> ~tau_1+ w-)
      0.000000e+00 2 -2000015 -24 # br(~nu_taul -> ~tau_2+ w-)

DECAY 1000024 1.704145e-02 # chargino1+ decays
      0.000000e+00 2 -1 1000002 # br(~chi_1+ -> ~u_l   db)
      0.000000e+00 2 -1 2000002 # br(~chi_1+ -> ~u_r   db)
      0.000000e+00 2 -1000001 2 # br(~chi_1+ -> ~d_l*  u )
      0.000000e+00 2 -2000001 2 # br(~chi_1+ -> ~d_r*  u )
      0.000000e+00 2 -3 1000004 # br(~chi_1+ -> ~c_l   sb)
      0.000000e+00 2 -3 2000004 # br(~chi_1+ -> ~c_r   sb)
      0.000000e+00 2 -1000003 4 # br(~chi_1+ -> ~s_l*  c )
      0.000000e+00 2 -2000003 4 # br(~chi_1+ -> ~s_r*  c )
      0.000000e+00 2 -5 1000006 # br(~chi_1+ -> ~t_1   bb)
      0.000000e+00 2 -5 2000006 # br(~chi_1+ -> ~t_2   bb)
      0.000000e+00 2 -1000005 6 # br(~chi_1+ -> ~b_1*  t )
      0.000000e+00 2 -2000005 6 # br(~chi_1+ -> ~b_2*  t )
      0.000000e+00 2 -11 1000012 # br(~chi_1+ -> ~nu_el  e+  )
      0.000000e+00 2 -13 1000014 # br(~chi_1+ -> ~nu_mul  mu+ )
      0.000000e+00 2 -15 1000016 # br(~chi_1+ -> ~nu_tau1 tau+)
      0.000000e+00 2 -1000011 12 # br(~chi_1+ -> ~e_l+    nu_e)
      0.000000e+00 2 -2000011 12 # br(~chi_1+ -> ~e_r+    nu_e)
      0.000000e+00 2 -1000013 14 # br(~chi_1+ -> ~mu_l+   nu_mu)
      0.000000e+00 2 -2000013 14 # br(~chi_1+ -> ~mu_r+   nu_mu)
      9.251611e-01 2 -1000015 16 # br(~chi_1+ -> ~tau_1+  nu_tau)
      0.000000e+00 2 -2000015 16 # br(~chi_1+ -> ~tau_2+  nu_tau)
      7.483888e-02 2 24 1000022 # br(~chi_1+ -> ~chi_10  w+)
      0.000000e+00 2 24 1000023 # br(~chi_1+ -> ~chi_20  w+)
      0.000000e+00 2 24 1000025 # br(~chi_1+ -> ~chi_30  w+)
      0.000000e+00 2 24 1000035 # br(~chi_1+ -> ~chi_40  w+)
      0.000000e+00 2 37 1000022 # br(~chi_1+ -> ~chi_10  h+)
      0.000000e+00 2 37 1000023 # br(~chi_1+ -> ~chi_20  h+)
      0.000000e+00 2 37 1000025 # br(~chi_1+ -> ~chi_30  h+)
      0.000000e+00 2 37 1000035 # br(~chi_1+ -> ~chi_40  h+)

DECAY 1000037 2.486895e+00 # chargino2+ decays
      0.000000e+00 2 -1 1000002 # br(~chi_2+ -> ~u_l   db)
      0.000000e+00 2 -1 2000002 # br(~chi_2+ -> ~u_r   db)
      0.000000e+00 2 -1000001 2 # br(~chi_2+ -> ~d_l*  u )
      0.000000e+00 2 -2000001 2 # br(~chi_2+ -> ~d_r*  u )
      0.000000e+00 2 -3 1000004 # br(~chi_2+ -> ~c_l   sb)
      0.000000e+00 2 -3 2000004 # br(~chi_2+ -> ~c_r   sb)
      0.000000e+00 2 -1000003 4 # br(~chi_2+ -> ~s_l*  c )
      0.000000e+00 2 -2000003 4 # br(~chi_2+ -> ~s_r*  c )
      0.000000e+00 2 -5 1000006 # br(~chi_2+ -> ~t_1   bb)
      0.000000e+00 2 -5 2000006 # br(~chi_2+ -> ~t_2   bb)
      0.000000e+00 2 -1000005 6 # br(~chi_2+ -> ~b_1*  t )
      0.000000e+00 2 -2000005 6 # br(~chi_2+ -> ~b_2*  t )
      2.009688e-02 2 -11 1000012 # br(~chi_2+ -> ~nu_el  e+  )
      2.009688e-02 2 -13 1000014 # br(~chi_2+ -> ~nu_mul  mu+ )
      2.745074e-02 2 -15 1000016 # br(~chi_2+ -> ~nu_tau1 tau+)
      5.204061e-02 2 -1000011 12 # br(~chi_2+ -> ~e_l+    nu_e)
      0.000000e+00 2 -2000011 12 # br(~chi_2+ -> ~e_r+    nu_e)
      5.204061e-02 2 -1000013 14 # br(~chi_2+ -> ~mu_l+   nu_mu)
      0.000000e+00 2 -2000013 14 # br(~chi_2+ -> ~mu_r+   nu_mu)
      2.828599e-04 2 -1000015 16 # br(~chi_2+ -> ~tau_1+  nu_tau)
      5.667293e-02 2 -2000015 16 # br(~chi_2+ -> ~tau_2+  nu_tau)
      2.315133e-01 2 23 1000024 # br(~chi_2+ -> ~chi_1+  z )
      6.767151e-02 2 24 1000022 # br(~chi_2+ -> ~chi_10  w+)
      2.936548e-01 2 24 1000023 # br(~chi_2+ -> ~chi_20  w+)
      0.000000e+00 2 24 1000025 # br(~chi_2+ -> ~chi_30  w+)
      0.000000e+00 2 24 1000035 # br(~chi_2+ -> ~chi_40  w+)
      1.784788e-01 2 25 1000024 # br(~chi_2+ -> ~chi_1+  h )
      0.000000e+00 2 35 1000024 # br(~chi_2+ -> ~chi_1+  h )
      0.000000e+00 2 36 1000024 # br(~chi_2+ -> ~chi_1+  a )
      0.000000e+00 2 37 1000022 # br(~chi_2+ -> ~chi_10  h+)
      0.000000e+00 2 37 1000023 # br(~chi_2+ -> ~chi_20  h+)
      0.000000e+00 2 37 1000025 # br(~chi_2+ -> ~chi_30  h+)
      0.000000e+00 2 37 1000035 # br(~chi_2+ -> ~chi_40  h+)

DECAY 1000022 0.000000e+00 # neutralino1 decays
DECAY 1000023 2.077700e-02 # neutralino2 decays
      0.000000e+00 2 23 1000022 # br(~chi_20 -> ~chi_10   z )
      0.000000e+00 2 -24 1000024 # br(~chi_20 -> ~chi_1+   w-)
      0.000000e+00 2 -1000024 24 # br(~chi_20 -> ~chi_1-   w+)
      0.000000e+00 2 -24 1000037 # br(~chi_20 -> ~chi_2+   w-)
      0.000000e+00 2 -1000037 24 # br(~chi_20 -> ~chi_2-   w+)
      0.000000e+00 2 25 1000022 # br(~chi_20 -> ~chi_10   h )
      0.000000e+00 2 35 1000022 # br(~chi_20 -> ~chi_10   h )
      0.000000e+00 2 36 1000022 # br(~chi_20 -> ~chi_10   a )
      0.000000e+00 2 -37 1000024 # br(~chi_20 -> ~chi_1+   h-)
      0.000000e+00 2 -1000024 37 # br(~chi_20 -> ~chi_1-   h+)
      0.000000e+00 2 -37 1000037 # br(~chi_20 -> ~chi_2+   h-)
      0.000000e+00 2 -1000037 37 # br(~chi_20 -> ~chi_2-   h+)
      0.000000e+00 2 -2 1000002 # br(~chi_20 -> ~u_l      ub)
      0.000000e+00 2 -1000002 2 # br(~chi_20 -> ~u_l*     u )
      0.000000e+00 2 -2 2000002 # br(~chi_20 -> ~u_r      ub)
      0.000000e+00 2 -2000002 2 # br(~chi_20 -> ~u_r*     u )
      0.000000e+00 2 -1 1000001 # br(~chi_20 -> ~d_l      db)
      0.000000e+00 2 -1000001 1 # br(~chi_20 -> ~d_l*     d )
      0.000000e+00 2 -1 2000001 # br(~chi_20 -> ~d_r      db)
      0.000000e+00 2 -2000001 1 # br(~chi_20 -> ~d_r*     d )
      0.000000e+00 2 -4 1000004 # br(~chi_20 -> ~c_l      cb)
      0.000000e+00 2 -1000004 4 # br(~chi_20 -> ~c_l*     c )
      0.000000e+00 2 -4 2000004 # br(~chi_20 -> ~c_r      cb)
      0.000000e+00 2 -2000004 4 # br(~chi_20 -> ~c_r*     c )
      0.000000e+00 2 -3 1000003 # br(~chi_20 -> ~s_l      sb)
      0.000000e+00 2 -1000003 3 # br(~chi_20 -> ~s_l*     s )
      0.000000e+00 2 -3 2000003 # br(~chi_20 -> ~s_r      sb)
      0.000000e+00 2 -2000003 3 # br(~chi_20 -> ~s_r*     s )
      0.000000e+00 2 -6 1000006 # br(~chi_20 -> ~t_1      tb)
      0.000000e+00 2 -1000006 6 # br(~chi_20 -> ~t_1*     t )
      0.000000e+00 2 -6 2000006 # br(~chi_20 -> ~t_2      tb)
      0.000000e+00 2 -2000006 6 # br(~chi_20 -> ~t_2*     t )
      0.000000e+00 2 -5 1000005 # br(~chi_20 -> ~b_1      bb)
      0.000000e+00 2 -1000005 5 # br(~chi_20 -> ~b_1*     b )
      0.000000e+00 2 -5 2000005 # br(~chi_20 -> ~b_2      bb)
      0.000000e+00 2 -2000005 5 # br(~chi_20 -> ~b_2*     b )
      0.000000e+00 2 -11 1000011 # br(~chi_20 -> ~e_l-     e+)
      0.000000e+00 2 -1000011 11 # br(~chi_20 -> ~e_l+     e-)
      2.950720e-02 2 -11 2000011 # br(~chi_20 -> ~e_r-     e+)
      2.950720e-02 2 -2000011 11 # br(~chi_20 -> ~e_r+     e-)
      0.000000e+00 2 -13 1000013 # br(~chi_20 -> ~mu_l-    mu+)
      0.000000e+00 2 -1000013 13 # br(~chi_20 -> ~mu_l+    mu-)
      2.950720e-02 2 -13 2000013 # br(~chi_20 -> ~mu_r-    mu+)
      2.950720e-02 2 -2000013 13 # br(~chi_20 -> ~mu_r+    mu-)
      4.409856e-01 2 -15 1000015 # br(~chi_20 -> ~tau_1-   tau+)
      4.409856e-01 2 -1000015 15 # br(~chi_20 -> ~tau_1+   tau-)
      0.000000e+00 2 -15 2000015 # br(~chi_20 -> ~tau_2-   tau+)
      0.000000e+00 2 -2000015 15 # br(~chi_20 -> ~tau_2+   tau-)
      0.000000e+00 2 -12 1000012 # br(~chi_20 -> ~nu_el    nu_eb)
      0.000000e+00 2 -1000012 12 # br(~chi_20 -> ~nu_el*   nu_e )
      0.000000e+00 2 -14 1000014 # br(~chi_20 -> ~nu_mul   nu_mub)
      0.000000e+00 2 -1000014 14 # br(~chi_20 -> ~nu_mul*  nu_mu )
      0.000000e+00 2 -16 1000016 # br(~chi_20 -> ~nu_tau1  nu_taub)
      0.000000e+00 2 -1000016 16 # br(~chi_20 -> ~nu_tau1* nu_tau )

DECAY 1000025 1.915985e+00 # neutralino3 decays
      1.132266e-01 2 23 1000022 # br(~chi_30 -> ~chi_10   z )
      2.119692e-01 2 23 1000023 # br(~chi_30 -> ~chi_20   z )
      2.953298e-01 2 -24 1000024 # br(~chi_30 -> ~chi_1+   w-)
      2.953298e-01 2 -1000024 24 # br(~chi_30 -> ~chi_1-   w+)
      0.000000e+00 2 -24 1000037 # br(~chi_30 -> ~chi_2+   w-)
      0.000000e+00 2 -1000037 24 # br(~chi_30 -> ~chi_2-   w+)
      2.130765e-02 2 25 1000022 # br(~chi_30 -> ~chi_10   h )
      0.000000e+00 2 35 1000022 # br(~chi_30 -> ~chi_10   h )
      0.000000e+00 2 36 1000022 # br(~chi_30 -> ~chi_10   a )
      1.245383e-02 2 25 1000023 # br(~chi_30 -> ~chi_20   h )
      0.000000e+00 2 35 1000023 # br(~chi_30 -> ~chi_20   h )
      0.000000e+00 2 36 1000023 # br(~chi_30 -> ~chi_20   a )
      0.000000e+00 2 -37 1000024 # br(~chi_30 -> ~chi_1+   h-)
      0.000000e+00 2 -1000024 37 # br(~chi_30 -> ~chi_1-   h+)
      0.000000e+00 2 -37 1000037 # br(~chi_30 -> ~chi_2+   h-)
      0.000000e+00 2 -1000037 37 # br(~chi_30 -> ~chi_2-   h+)
      0.000000e+00 2 -2 1000002 # br(~chi_30 -> ~u_l      ub)
      0.000000e+00 2 -1000002 2 # br(~chi_30 -> ~u_l*     u )
      0.000000e+00 2 -2 2000002 # br(~chi_30 -> ~u_r      ub)
      0.000000e+00 2 -2000002 2 # br(~chi_30 -> ~u_r*     u )
      0.000000e+00 2 -1 1000001 # br(~chi_30 -> ~d_l      db)
      0.000000e+00 2 -1000001 1 # br(~chi_30 -> ~d_l*     d )
      0.000000e+00 2 -1 2000001 # br(~chi_30 -> ~d_r      db)
      0.000000e+00 2 -2000001 1 # br(~chi_30 -> ~d_r*     d )
      0.000000e+00 2 -4 1000004 # br(~chi_30 -> ~c_l      cb)
      0.000000e+00 2 -1000004 4 # br(~chi_30 -> ~c_l*     c )
      0.000000e+00 2 -4 2000004 # br(~chi_30 -> ~c_r      cb)
      0.000000e+00 2 -2000004 4 # br(~chi_30 -> ~c_r*     c )
      0.000000e+00 2 -3 1000003 # br(~chi_30 -> ~s_l      sb)
      0.000000e+00 2 -1000003 3 # br(~chi_30 -> ~s_l*     s )
      0.000000e+00 2 -3 2000003 # br(~chi_30 -> ~s_r      sb)
      0.000000e+00 2 -2000003 3 # br(~chi_30 -> ~s_r*     s )
      0.000000e+00 2 -6 1000006 # br(~chi_30 -> ~t_1      tb)
      0.000000e+00 2 -1000006 6 # br(~chi_30 -> ~t_1*     t )
      0.000000e+00 2 -6 2000006 # br(~chi_30 -> ~t_2      tb)
      0.000000e+00 2 -2000006 6 # br(~chi_30 -> ~t_2*     t )
      0.000000e+00 2 -5 1000005 # br(~chi_30 -> ~b_1      bb)
      0.000000e+00 2 -1000005 5 # br(~chi_30 -> ~b_1*     b )
      0.000000e+00 2 -5 2000005 # br(~chi_30 -> ~b_2      bb)
      0.000000e+00 2 -2000005 5 # br(~chi_30 -> ~b_2*     b )
      5.572205e-04 2 -11 1000011 # br(~chi_30 -> ~e_l-     e+)
      5.572205e-04 2 -1000011 11 # br(~chi_30 -> ~e_l+     e-)
      1.252668e-03 2 -11 2000011 # br(~chi_30 -> ~e_r-     e+)
      1.252668e-03 2 -2000011 11 # br(~chi_30 -> ~e_r+     e-)
      5.572205e-04 2 -13 1000013 # br(~chi_30 -> ~mu_l-    mu+)
      5.572205e-04 2 -1000013 13 # br(~chi_30 -> ~mu_l+    mu-)
      1.252668e-03 2 -13 2000013 # br(~chi_30 -> ~mu_r-    mu+)
      1.252668e-03 2 -2000013 13 # br(~chi_30 -> ~mu_r+    mu-)
      5.262792e-03 2 -15 1000015 # br(~chi_30 -> ~tau_1-   tau+)
      5.262792e-03 2 -1000015 15 # br(~chi_30 -> ~tau_1+   tau-)
      6.728146e-03 2 -15 2000015 # br(~chi_30 -> ~tau_2-   tau+)
      6.728146e-03 2 -2000015 15 # br(~chi_30 -> ~tau_2+   tau-)
      3.189205e-03 2 -12 1000012 # br(~chi_30 -> ~nu_el    nu_eb)
      3.189205e-03 2 -1000012 12 # br(~chi_30 -> ~nu_el*   nu_e )
      3.189205e-03 2 -14 1000014 # br(~chi_30 -> ~nu_mul   nu_mub)
      3.189205e-03 2 -1000014 14 # br(~chi_30 -> ~nu_mul*  nu_mu )
      3.202459e-03 2 -16 1000016 # br(~chi_30 -> ~nu_tau1  nu_taub)
      3.202459e-03 2 -1000016 16 # br(~chi_30 -> ~nu_tau1* nu_tau )

DECAY 1000035 2.585851e+00 # neutralino4 decays
      2.153693e-02 2 23 1000022 # br(~chi_40 -> ~chi_10   z )
      1.855000e-02 2 23 1000023 # br(~chi_40 -> ~chi_20   z )
      0.000000e+00 2 23 1000025 # br(~chi_40 -> ~chi_30   z )
      2.495414e-01 2 -24 1000024 # br(~chi_40 -> ~chi_1+   w-)
      2.495414e-01 2 -1000024 24 # br(~chi_40 -> ~chi_1-   w+)
      0.000000e+00 2 -24 1000037 # br(~chi_40 -> ~chi_2+   w-)
      0.000000e+00 2 -1000037 24 # br(~chi_40 -> ~chi_2-   w+)
      6.932133e-02 2 25 1000022 # br(~chi_40 -> ~chi_10   h )
      0.000000e+00 2 35 1000022 # br(~chi_40 -> ~chi_10   h )
      0.000000e+00 2 36 1000022 # br(~chi_40 -> ~chi_10   a )
      1.476023e-01 2 25 1000023 # br(~chi_40 -> ~chi_20   h )
      0.000000e+00 2 35 1000023 # br(~chi_40 -> ~chi_20   h )
      0.000000e+00 2 36 1000023 # br(~chi_40 -> ~chi_20   a )
      0.000000e+00 2 25 1000025 # br(~chi_40 -> ~chi_30   h )
      0.000000e+00 2 35 1000025 # br(~chi_40 -> ~chi_30   h )
      0.000000e+00 2 36 1000025 # br(~chi_40 -> ~chi_30   a )
      0.000000e+00 2 -37 1000024 # br(~chi_40 -> ~chi_1+   h-)
      0.000000e+00 2 -1000024 37 # br(~chi_40 -> ~chi_1-   h+)
      0.000000e+00 2 -37 1000037 # br(~chi_40 -> ~chi_2+   h-)
      0.000000e+00 2 -1000037 37 # br(~chi_40 -> ~chi_2-   h+)
      0.000000e+00 2 -2 1000002 # br(~chi_40 -> ~u_l      ub)
      0.000000e+00 2 -1000002 2 # br(~chi_40 -> ~u_l*     u )
      0.000000e+00 2 -2 2000002 # br(~chi_40 -> ~u_r      ub)
      0.000000e+00 2 -2000002 2 # br(~chi_40 -> ~u_r*     u )
      0.000000e+00 2 -1 1000001 # br(~chi_40 -> ~d_l      db)
      0.000000e+00 2 -1000001 1 # br(~chi_40 -> ~d_l*     d )
      0.000000e+00 2 -1 2000001 # br(~chi_40 -> ~d_r      db)
      0.000000e+00 2 -2000001 1 # br(~chi_40 -> ~d_r*     d )
      0.000000e+00 2 -4 1000004 # br(~chi_40 -> ~c_l      cb)
      0.000000e+00 2 -1000004 4 # br(~chi_40 -> ~c_l*     c )
      0.000000e+00 2 -4 2000004 # br(~chi_40 -> ~c_r      cb)
      0.000000e+00 2 -2000004 4 # br(~chi_40 -> ~c_r*     c )
      0.000000e+00 2 -3 1000003 # br(~chi_40 -> ~s_l      sb)
      0.000000e+00 2 -1000003 3 # br(~chi_40 -> ~s_l*     s )
      0.000000e+00 2 -3 2000003 # br(~chi_40 -> ~s_r      sb)
      0.000000e+00 2 -2000003 3 # br(~chi_40 -> ~s_r*     s )
      0.000000e+00 2 -6 1000006 # br(~chi_40 -> ~t_1      tb)
      0.000000e+00 2 -1000006 6 # br(~chi_40 -> ~t_1*     t )
      0.000000e+00 2 -6 2000006 # br(~chi_40 -> ~t_2      tb)
      0.000000e+00 2 -2000006 6 # br(~chi_40 -> ~t_2*     t )
      0.000000e+00 2 -5 1000005 # br(~chi_40 -> ~b_1      bb)
      0.000000e+00 2 -1000005 5 # br(~chi_40 -> ~b_1*     b )
      0.000000e+00 2 -5 2000005 # br(~chi_40 -> ~b_2      bb)
      0.000000e+00 2 -2000005 5 # br(~chi_40 -> ~b_2*     b )
      9.648354e-03 2 -11 1000011 # br(~chi_40 -> ~e_l-     e+)
      9.648354e-03 2 -1000011 11 # br(~chi_40 -> ~e_l+     e-)
      3.756845e-03 2 -11 2000011 # br(~chi_40 -> ~e_r-     e+)
      3.756845e-03 2 -2000011 11 # br(~chi_40 -> ~e_r+     e-)
      9.648354e-03 2 -13 1000013 # br(~chi_40 -> ~mu_l-    mu+)
      9.648354e-03 2 -1000013 13 # br(~chi_40 -> ~mu_l+    mu-)
      3.756845e-03 2 -13 2000013 # br(~chi_40 -> ~mu_r-    mu+)
      3.756845e-03 2 -2000013 13 # br(~chi_40 -> ~mu_r+    mu-)
      2.682152e-03 2 -15 1000015 # br(~chi_40 -> ~tau_1-   tau+)
      2.682152e-03 2 -1000015 15 # br(~chi_40 -> ~tau_1+   tau-)
      1.622898e-02 2 -15 2000015 # br(~chi_40 -> ~tau_2-   tau+)
      1.622898e-02 2 -2000015 15 # br(~chi_40 -> ~tau_2+   tau-)
      2.537965e-02 2 -12 1000012 # br(~chi_40 -> ~nu_el    nu_eb)
      2.537965e-02 2 -1000012 12 # br(~chi_40 -> ~nu_el*   nu_e )
      2.537965e-02 2 -14 1000014 # br(~chi_40 -> ~nu_mul   nu_mub)
      2.537965e-02 2 -1000014 14 # br(~chi_40 -> ~nu_mul*  nu_mu )
      2.547244e-02 2 -16 1000016 # br(~chi_40 -> ~nu_tau1  nu_taub)
      2.547244e-02 2 -1000016 16 # br(~chi_40 -> ~nu_tau1* nu_tau )

###################################
## INFORMATION FOR USQMIX
###################################
BLOCK USQMIX # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      4 4 1.000000e+00 # 
      5 5 1.000000e+00 # 
      3 3 5.536450e-01 # o_{11}
      3 6 8.327528e-01 # o_{12}
      6 3 8.327528e-01 # o_{21}
      6 6 -5.536450e-01 # o_{22}
###################################
## INFORMATION FOR DSQMIX
###################################
BLOCK DSQMIX # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      4 4 1.000000e+00 # 
      5 5 1.000000e+00 # 
      3 3 9.387379e-01 # o_{11}
      3 6 3.446319e-01 # o_{12}
      6 3 -3.446319e-01 # o_{21}
      6 6 9.387379e-01 # o_{22}
###################################
## INFORMATION FOR SELMIX
###################################
BLOCK SELMIX # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      4 4 1.000000e+00 # 
      5 5 1.000000e+00 # 
      3 3 2.824872e-01 # o_{11}
      3 6 9.592711e-01 # o_{12}
      6 3 9.592711e-01 # o_{21}
      6 6 -2.824872e-01 # o_{22}
###################################
## INFORMATION FOR FRALPHA
###################################
BLOCK FRALPHA # 
      1 -1.138252e-01 # mixing angle in the neutral higgs boson sector
###################################
## INFORMATION FOR VCKM
###################################
BLOCK VCKM # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      3 3 1.000000e+00 # 
###################################
## INFORMATION FOR SNUMIX
###################################
BLOCK SNUMIX # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      3 3 1.000000e+00 # 
###################################
## INFORMATION FOR UPMNS
###################################
BLOCK UPMNS # 
      1 1 1.000000e+00 # 
      2 2 1.000000e+00 # 
      3 3 1.000000e+00 # 
###################################
## INFORMATION FOR TE
###################################
BLOCK TE # 
      1 1 0.000000e+00 # T_e(Q) DRbar
      2 2 0.000000e+00 # T_mu(Q) DRbar
      3 3 -2.540197e+01 # T_tau(Q) DRbar
###################################
## INFORMATION FOR TU
###################################
BLOCK TU # 
      1 1 0.000000e+00 # T_u(Q) DRbar
      2 2 0.000000e+00 # T_c(Q) DRbar
      3 3 -4.447525e+02 # T_t(Q) DRbar
###################################
## INFORMATION FOR TD
###################################
BLOCK TD # 
      1 1 0.000000e+00 # T_d(Q) DRbar
      2 2 0.000000e+00 # T_s(Q) DRbar
      3 3 -1.106937e+02 # T_b(Q) DRbar
###################################
## INFORMATION FOR MSL2
###################################
BLOCK MSL2 # 
      1 1 3.815567e+04 # mel(q)
      2 2 3.815567e+04 # mmul(q)
      3 3 3.782868e+04 # mtaul(q)
###################################
## INFORMATION FOR MSE2
###################################
BLOCK MSE2 # 
      1 1 1.863063e+04 # mer(q)
      2 2 1.863063e+04 # mmur(q)
      3 3 1.796764e+04 # mtaur(q)
###################################
## INFORMATION FOR MSQ2
###################################
BLOCK MSQ2 # 
      1 1 2.998367e+05 # mql1(q)
      2 2 2.998367e+05 # mql2(q)
      3 3 2.487654e+05 # mql3(q)
###################################
## INFORMATION FOR MSU2
###################################
BLOCK MSU2 # 
      1 1 2.803821e+05 # mur(q)
      2 2 2.803821e+05 # mcr(q)
      3 3 1.791371e+05 # mtr(q)
###################################
## INFORMATION FOR MSD2
###################################
BLOCK MSD2 # 
      1 1 2.736847e+05 # mdr(q)
      2 2 2.736847e+05 # msr(q)
      3 3 2.702620e+05 # mbr(q)
"""



        fsock = StringIO.StringIO()
        writter.convert_to_mg5card(self.sps1a, fsock)
        output = fsock.getvalue()
        self.assertEqual(output.split('\n'), target.split('\n'))
        
        
        
        
        
        

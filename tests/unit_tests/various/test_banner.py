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
        
        
        
        
        
        
        
        

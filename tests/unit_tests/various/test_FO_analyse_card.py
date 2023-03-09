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
import os
import sys
import tests.unit_tests as unittest

import madgraph.various.FO_analyse_card as FO_analyse_card


class TestFOAnalyseCard(unittest.TestCase):
    """Check the class linked to a block of the param_card"""

    def setUp(self):
        if not hasattr(self, 'card_default') or not hasattr(self, 'card_analyse'):
            text_default = \
"""
################################################################################
#                                                                               
# This file contains the settings for analyses to be linked to aMC@NLO fixed
# order runs. Analyse files are meant to be put (or linked) inside
# PROCDIR/SubProcesses (PROCDIR is the name of the exported process directory)
#                                                                               
################################################################################
FO_EXTRALIBS   =                    # Needed extra-libraries (FastJet is already linked). 
FO_EXTRAPATHS  =                    # (Absolute) path to the extra-libraries. 
FO_INCLUDEPATHS=                    # (Absolute) Path to the dirs containing header files neede by C++.
                                    # Directory names are separated by white spaces
FO_ANALYSE     = analysis_fixed_order.o # User's analysis and histogramming routines 
                                    # (please use .o as extension and white spaces to separate files)
"""
            TestFOAnalyseCard.card_default = FO_analyse_card.FOAnalyseCard(text_default, testing = True) 

            text_analyse = \
"""
################################################################################
#                                                                               
# This file contains the settings for analyses to be linked to aMC@NLO fixed
# order runs. Analyse files are meant to be put (or linked) inside
# PROCDIR/SubProcesses (PROCDIR is the name of the exported process directory)
#                                                                               
################################################################################
FO_EXTRALIBS   = libdummy1.a dummy2                   # Needed extra-libraries (FastJet is already linked). 
FO_EXTRAPATHS  = /path/to/dummy1 /path/to/dummy2                   # (Absolute) path to the extra-libraries. 
FO_INCLUDEPATHS= /path/to/dummy1/include /path/to/dummy2/include                   # (Absolute) Path to the dirs containing header files neede by C++.
                                    # Directory names are separated by white spaces
FO_ANALYSE     = analysis_fixed_order.o dummy1.o dummy2.o # User's analysis and histogramming routines 
                                    # (please use .o as extension and white spaces to separate files)
"""
            TestFOAnalyseCard.card_analyse = FO_analyse_card.FOAnalyseCard(text_analyse, testing = True) 


    def test_analyse_card_default(self):
        """test that the default card is correctly written"""
        goal = \
"""FO_EXTRALIBS=
FO_EXTRAPATHS= 
FO_INCLUDEPATHS=
FO_ANALYSE=analysis_fixed_order.o 
"""
        text = self.card_default.write_card('')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_analyse_card_analyse(self):
        """test that the card with extra libraries is correctly written"""
        goal = \
"""FO_EXTRALIBS=-ldummy1 -ldummy2
FO_EXTRAPATHS=-Wl,-rpath,/path/to/dummy1 -Wl,-rpath,/path/to/dummy2 -L/path/to/dummy1 -L/path/to/dummy2
FO_INCLUDEPATHS=-I/path/to/dummy1/include -I/path/to/dummy2/include
FO_ANALYSE=analysis_fixed_order.o dummy1.o dummy2.o 
"""
        text = self.card_analyse.write_card('')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


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

import madgraph.various.shower_card as shower_card


class TestShowerCard(unittest.TestCase):
    """Check the class linked to a block of the param_card"""

    def setUp(self):
        if not hasattr(self, 'card') or not hasattr(self, 'card_analyse'):
            text = \
"""################################################################################
#                                                                               
# This file contains the settings for partons showers to be used in aMC@NLO     
#   mind the format:   variable = name           #         comment              
#                                                                               
################################################################################
#
# SHOWER SETTINGS:
#
ue_enabled = T      # underlying event on (T)/ off (F) (is MSTP(81) for PY6)
hadronize  = T      # hadronization on/off (is MSTP(111) for PY6) 
maxprint   = 2      # maximum number of events to be printed in the log
maxerrs    = 0.01   # maximum tolerated fraction of errors
b_stable   = F      # b hadrons are stable
pi_stable  = T      # pi0's are stable
wp_stable  = F      # w+'s are stable
wm_stable  = F      # w-'s are stable
z_stable   = F      # z0's are stable
h_stable   = F      # Higgs bosons are stable
tap_stable = F      # tau+'s are stable
tam_stable = F      # tau-'s are stable
mup_stable = F      # mu+'s are stable
mum_stable = F      # mu-'s are stable
rnd_seed   = 0      # random seed (0 is default)
rnd_seed2  = 0      # 2nd random seed (only for HW6, 0 is default)
lambda_5   = -1     # lambda_5 value, -1 for default (only for HW6/PY6)
is_4lep    = F      # true if it is 4 lepton production (only for PY6)
is_bbar    = F      # true if it is b-b~ production (only for HW6)
modbos_1   = 5      # decay mode for the first Boson (only for HW6)
modbos_2   = 5      # decay mode for the second Boson (only for HW6)
################################################################################
#
# HERWIG++ PATHS
# The following lines need to be set only for Herwig++. Use the absolute paths
# to the directories where the packages are installed, containing lib/, include/,
# and share/ subfolders
#
################################################################################
HWPPPATH   = None   # Path to the dir where Herwig++ is installed
THEPEGPATH = None   # Path to the dir where ThePeg is installed
HEPMCPATH  = None   # Path to the dir where HepMC is installed
################################################################################
#
# EXTRA LIBRARIES/ANALYSES
# The following lines need to be changed if the user does not want to create the
# StdHEP/HEPMC file, but to directly run his/her own analyse.
# Please note that this works only for HW6 and PY6, and that the analysis should
# be in the HWAnalyzer/ (or PYAnalyzer/) folder. 
# Please use files in those folders as examples.
#
################################################################################
EXTRALIBS   = stdhep Fmcfio         # Needed extra-libraries (not LHAPDF). 
                                    #   Default: "stdhep Fmcfio"
EXTRAPATHS  = ../lib                # Path to the extra-libraries. 
                                    #   Default: "../lib"
INCLUDEPATHS=                       # Path to the dirs containing header files neede by C++.
                                    # Directory names are separated by white spaces
ANALYSE     =                       # User's analysis and histogramming routines 
                                    # (please use .o as extension and use spaces to separate files)
"""
            TestShowerCard.card = shower_card.ShowerCard(text, testing = True) 

            text_analyse = \
"""################################################################################
#                                                                               
# This file contains the settings for partons showers to be used in aMC@NLO     
#   mind the format:   variable = name           #         comment              
#                                                                               
################################################################################
#
# SHOWER SETTINGS:
#
ue_enabled = T      # underlying event on (T)/ off (F) (is MSTP(81) for PY6)
hadronize  = T      # hadronization on/off (is MSTP(111) for PY6) 
maxprint   = 2      # maximum number of events to be printed in the log
maxerrs    = 0.01   # maximum tolerated fraction of errors
b_stable   = F      # b hadrons are stable
pi_stable  = T      # pi0's are stable
wp_stable  = F      # w+'s are stable
wm_stable  = F      # w-'s are stable
z_stable   = F      # z0's are stable
h_stable   = F      # Higgs bosons are stable
tap_stable = F      # tau+'s are stable
tam_stable = F      # tau-'s are stable
mup_stable = F      # mu+'s are stable
mum_stable = F      # mu-'s are stable
rnd_seed   = 0      # random seed (0 is default)
rnd_seed2  = 0      # 2nd random seed (only for HW6, 0 is default)
lambda_5   = -1     # lambda_5 value, -1 for default (only for HW6/PY6)
is_4lep    = F      # true if it is 4 lepton production (only for PY6)
is_bbar    = F      # true if it is b-b~ production (only for HW6)
modbos_1   = 5      # decay mode for the first Boson (only for HW6)
modbos_2   = 5      # decay mode for the second Boson (only for HW6)
################################################################################
#
# HERWIG++ PATHS
# The following lines need to be set only for Herwig++. Use the absolute paths
# to the directories where the packages are installed, containing lib/, include/,
# and share/ subfolders
#
################################################################################
HWPPPATH   = None   # Path to the dir where Herwig++ is installed
THEPEGPATH = None   # Path to the dir where ThePeg is installed
HEPMCPATH  = None   # Path to the dir where HepMC is installed
################################################################################
#
# EXTRA LIBRARIES/ANALYSES
# The following lines need to be changed if the user does not want to create the
# StdHEP/HEPMC file, but to directly run his/her own analyse.
# Please note that this works only for HW6 and PY6, and that the analysis should
# be in the HWAnalyzer/ (or PYAnalyzer/) folder. 
# Please use files in those folders as examples.
#
################################################################################
EXTRALIBS   = stdhep Fmcfio         # Needed extra-libraries (not LHAPDF). 
                                    #   Default: "stdhep Fmcfio"
EXTRAPATHS  = ../lib                # Path to the extra-libraries. 
                                    #   Default: "../lib"
INCLUDEPATHS=                       # Path to the dirs containing header files neede by C++.
                                    # Directory names are separated by white spaces
ANALYSE     =myanalyse.o            # User's analysis and histogramming routines 
                                    # (please use .o as extension and use spaces to separate files)
"""
            TestShowerCard.card_analyse = shower_card.ShowerCard(text_analyse, testing = True) 
    
    def test_shower_card_hwpp(self):
        """test that the hwpp card is correctly written"""
        goal = \
"""UE_HWPP=.TRUE.
HADRONIZE_HWPP=.TRUE.
MAXPR_HWPP=2
ERR_FR_HWPP=0.010
B_STABLE_HWPP=.FALSE.
PI_STABLE_HWPP=.TRUE.
WP_STABLE_HWPP=.FALSE.
WM_STABLE_HWPP=.FALSE.
Z_STABLE_HWPP=.FALSE.
H_STABLE_HWPP=.FALSE.
TAUP_STABLE_HWPP=.FALSE.
TAUM_STABLE_HWPP=.FALSE.
MUP_STABLE_HWPP=.FALSE.
MUM_STABLE_HWPP=.FALSE.
RNDEVSEED_HWPP=0
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWPPUTI=""
"""
        text = self.card.write_card('HERWIGPP', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_hwpp_analyse(self):
        """test that the hwpp card is correctly written"""
        goal = \
"""UE_HWPP=.TRUE.
HADRONIZE_HWPP=.TRUE.
MAXPR_HWPP=2
ERR_FR_HWPP=0.010
B_STABLE_HWPP=.FALSE.
PI_STABLE_HWPP=.TRUE.
WP_STABLE_HWPP=.FALSE.
WM_STABLE_HWPP=.FALSE.
Z_STABLE_HWPP=.FALSE.
H_STABLE_HWPP=.FALSE.
TAUP_STABLE_HWPP=.FALSE.
TAUM_STABLE_HWPP=.FALSE.
MUP_STABLE_HWPP=.FALSE.
MUM_STABLE_HWPP=.FALSE.
RNDEVSEED_HWPP=0
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWPPUTI="myanalyse.o"
"""
        text = self.card_analyse.write_card('HERWIGPP', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_hw6(self):
        """test that the hw6 card is correctly written"""
        goal = \
"""LHSOFT=.TRUE.
MAXPR_HW=2
ERR_FR_HW=0.010
B_STABLE_HW=.FALSE.
PI_STABLE_HW=.TRUE.
WP_STABLE_HW=.FALSE.
WM_STABLE_HW=.FALSE.
Z_STABLE_HW=.FALSE.
H_STABLE_HW=.FALSE.
TAUP_STABLE_HW=.FALSE.
TAUM_STABLE_HW=.FALSE.
MUP_STABLE_HW=.FALSE.
MUM_STABLE_HW=.FALSE.
RNDEVSEED1_HW=0
RNDEVSEED2_HW=0
LAMBDAHERW=-1.000
IS_BB_HW=.FALSE.
MODBOS_1=5
MODBOS_2=5
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWUTI="mcatnlo_hwan_stdhep.o"
"""
        text = self.card.write_card('HERWIG6', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_hw6_analyse(self):
        """test that the hw6 card is correctly written"""
        goal = \
"""LHSOFT=.TRUE.
MAXPR_HW=2
ERR_FR_HW=0.010
B_STABLE_HW=.FALSE.
PI_STABLE_HW=.TRUE.
WP_STABLE_HW=.FALSE.
WM_STABLE_HW=.FALSE.
Z_STABLE_HW=.FALSE.
H_STABLE_HW=.FALSE.
TAUP_STABLE_HW=.FALSE.
TAUM_STABLE_HW=.FALSE.
MUP_STABLE_HW=.FALSE.
MUM_STABLE_HW=.FALSE.
RNDEVSEED1_HW=0
RNDEVSEED2_HW=0
LAMBDAHERW=-1.000
IS_BB_HW=.FALSE.
MODBOS_1=5
MODBOS_2=5
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWUTI="myanalyse.o"
"""
        text = self.card_analyse.write_card('HERWIG6', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_py6(self):
        """test that the py6 card is correctly written"""
        goal = \
"""MSTP_81=1
MSTP_111=1
MAXPR_PY=2
ERR_FR_PY=0.010
B_STABLE_PY=.FALSE.
PI_STABLE_PY=.TRUE.
WP_STABLE_PY=.FALSE.
WM_STABLE_PY=.FALSE.
Z_STABLE_PY=.FALSE.
H_STABLE_PY=.FALSE.
TAUP_STABLE_PY=.FALSE.
TAUM_STABLE_PY=.FALSE.
MUP_STABLE_PY=.FALSE.
MUM_STABLE_PY=.FALSE.
RNDEVSEED_PY=0
LAMBDAPYTH=-1.000
IS_4L_PY=.FALSE.
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
PYUTI="mcatnlo_pyan_stdhep.o"
"""
        text = self.card.write_card('PYTHIA6Q', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_py6_analyse(self):
        """test that the py6 card is correctly written"""
        goal = \
"""MSTP_81=1
MSTP_111=1
MAXPR_PY=2
ERR_FR_PY=0.010
B_STABLE_PY=.FALSE.
PI_STABLE_PY=.TRUE.
WP_STABLE_PY=.FALSE.
WM_STABLE_PY=.FALSE.
Z_STABLE_PY=.FALSE.
H_STABLE_PY=.FALSE.
TAUP_STABLE_PY=.FALSE.
TAUM_STABLE_PY=.FALSE.
MUP_STABLE_PY=.FALSE.
MUM_STABLE_PY=.FALSE.
RNDEVSEED_PY=0
LAMBDAPYTH=-1.000
IS_4L_PY=.FALSE.
HWPPPATH=
THEPEGPATH=
HEPMCPATH=
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
PYUTI="myanalyse.o"
"""
        text = self.card_analyse.write_card('PYTHIA6Q', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)

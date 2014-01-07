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
nevents    = -1     # number of events to shower. If negative, shower all events
ue_enabled = F      # underlying event on (T)/ off (F) (is MSTP(81) 0/1 for PY6)
pdfcode    = 0      # pdf code: 0 = internal pdf, 1 = same as NLO, other = lhaglue
hadronize  = T      # hadronization on/off (is MSTP(111) 0/1 for PY6) 
maxprint   = 2      # maximum number of events to be printed in the log
maxerrs    = 0.1    # maximum tolerated fraction of errors
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
lambda_5   = -1     # lambda_5 value, -1 for default
b_mass     = -1     # b mass, -1 for default
is_4lep    = F      # true if it is 4 lepton production (only for PY6)
is_bbar    = F      # true if it is b-b~ production (only for HW6)
modbos_1   = 5      # decay mode for the first Boson (only for HW6)
modbos_2   = 5      # dec mode for boson 2 (only for HW6, overridden by DM_ below)
################################################################################
# DECAY CHANNELS
# Write down the decay channels for the resonances, to be performed by the shower.
# The syntax (for a two-body decay) is
# DM_I = M > D1 D2 @ BR @ ME
# where I = 1, ..., 99, M is the decaying resonance, D1, D2 are the decay products
# (up to D5 if  such a decay is supported by the shower), BR is the branching ratio
# (only used by the HERWIG6 shower, ignored otherwise) and ME is the type of matrix
# element to be used in the decay (only used by HERWIG6, ignored otherwise).
# BR's are correctly understood by HERWIG6 only if they add up to one and only if
# no more than three modes are required for a given resonance.
# ME corresponds to the third entry of subroutine HWMODK, see the relevant manual.
# Examples of syntax:
# Z -> e+ e- or mu+ mu- with BR = 0.5 each
# DM_1 = 23 > -11 11 @ 0.5d0 @ 100
# DM_2 = 23 > -13 13 @ 0.5d0 @ 100
# H -> tau+ tau- with BR = 1
# DM_3 = 25 > -15 15 @ 1.0d0 @ 0
# t -> nu_e e+ b @ 1d0
# DM_4 = 6 > 12 -11 5 @ 1d0 @ 100
# WARNING: for HERWIG6 the order of decay products in >2-body decays IS RELEVANT.
# WARNING: 1 -> n decays (with n > 2) are handled by PYTHIA6 and PYTHIA8 through
# a sequence of 1 -> 2 decays.
################################################################################

################################################################################
#
# EXTRA LIBRARIES/ANALYSES
# The following lines need to be changed if the user does not want to create the
# StdHEP/HEPMC file, but to directly run his/her own analyse.
# Please note that this works only for HW6 and PY6, and that the analysis should
# be in the HWAnalyzer/ (or PYAnalyzer/) folder. 
# Please use files in those folders as examples.
# "None" and an empty value are equivalent.
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
nevents    = -1     # number of events to shower. If negative, shower all events
ue_enabled = F      # underlying event on (T)/ off (F) (is MSTP(81) 0/1 for PY6)
pdfcode    = 0      # pdf code: 0 = internal pdf, 1 = same as NLO, other = lhaglue
hadronize  = T      # hadronization on/off (is MSTP(111) 0/1 for PY6) 
maxprint   = 2      # maximum number of events to be printed in the log
maxerrs    = 0.1    # maximum tolerated fraction of errors
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
lambda_5   = -1     # lambda_5 value, -1 for default
b_mass     = -1     # b mass, -1 for default
is_4lep    = F      # true if it is 4 lepton production (only for PY6)
is_bbar    = F      # true if it is b-b~ production (only for HW6)
modbos_1   = 5      # decay mode for the first Boson (only for HW6)
modbos_2   = 5      # dec mode for boson 2 (only for HW6, overridden by DM_ below)
################################################################################
# DECAY CHANNELS
# Write down the decay channels for the resonances, to be performed by the shower.
# The syntax (for a two-body decay) is
# DM_I = M > D1 D2 @ BR @ ME
# where I = 1, ..., 99, M is the decaying resonance, D1, D2 are the decay products
# (up to D5 if  such a decay is supported by the shower), BR is the branching ratio
# (only used by the HERWIG6 shower, ignored otherwise) and ME is the type of matrix
# element to be used in the decay (only used by HERWIG6, ignored otherwise).
# BR's are correctly understood by HERWIG6 only if they add up to one and only if
# no more than three modes are required for a given resonance.
# ME corresponds to the third entry of subroutine HWMODK, see the relevant manual.
# Examples of syntax:
# Z -> e+ e- or mu+ mu- with BR = 0.5 each
# DM_1 = 23 > -11 11 @ 0.5d0 @ 100
# DM_2 = 23 > -13 13 @ 0.5d0 @ 100
# H -> tau+ tau- with BR = 1
# DM_3 = 25 > -15 15 @ 1.0d0 @ 0
# t -> nu_e e+ b @ 1d0
# DM_4 = 6 > 12 -11 5 @ 1d0 @ 100
# WARNING: for HERWIG6 the order of decay products in >2-body decays IS RELEVANT.
# WARNING: 1 -> n decays (with n > 2) are handled by PYTHIA6 and PYTHIA8 through
# a sequence of 1 -> 2 decays.
################################################################################

################################################################################
#
# EXTRA LIBRARIES/ANALYSES
# The following lines need to be changed if the user does not want to create the
# StdHEP/HEPMC file, but to directly run his/her own analyse.
# Please note that this works only for HW6 and PY6, and that the analysis should
# be in the HWAnalyzer/ (or PYAnalyzer/) folder. 
# Please use files in those folders as examples.
# "None" and an empty value are equivalent.
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
            TestShowerCard.card_analyse = shower_card.ShowerCard(text_analyse, testing = True) 


    def test_shower_card_py8(self):
        """test that the py8 card is correctly written"""
        goal = \
"""NEVENTS=-1
UE_PY8=.FALSE.
PDFCODE=0
HADRONIZE_PY8=.TRUE.
MAXPR_PY8=2
ERR_FR_PY8=0.100
B_STABLE_PY8=.FALSE.
PI_STABLE_PY8=.TRUE.
WP_STABLE_PY8=.FALSE.
WM_STABLE_PY8=.FALSE.
Z_STABLE_PY8=.FALSE.
H_STABLE_PY8=.FALSE.
TAUP_STABLE_PY8=.FALSE.
TAUM_STABLE_PY8=.FALSE.
MUP_STABLE_PY8=.FALSE.
MUM_STABLE_PY8=.FALSE.
RNDEVSEED_PY8=0
LAMBDAPYTH=-1.000
B_MASS=-1.000
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
PY8UTI=""
"""
        text = self.card.write_card('PYTHIA8', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)

    def test_shower_card_py8_analyse(self):
        """test that the py8 card is correctly written"""
        goal = \
"""NEVENTS=-1
UE_PY8=.FALSE.
PDFCODE=0
HADRONIZE_PY8=.TRUE.
MAXPR_PY8=2
ERR_FR_PY8=0.100
B_STABLE_PY8=.FALSE.
PI_STABLE_PY8=.TRUE.
WP_STABLE_PY8=.FALSE.
WM_STABLE_PY8=.FALSE.
Z_STABLE_PY8=.FALSE.
H_STABLE_PY8=.FALSE.
TAUP_STABLE_PY8=.FALSE.
TAUM_STABLE_PY8=.FALSE.
MUP_STABLE_PY8=.FALSE.
MUM_STABLE_PY8=.FALSE.
RNDEVSEED_PY8=0
LAMBDAPYTH=-1.000
B_MASS=-1.000
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
PY8UTI=""
"""
        text = self.card_analyse.write_card('PYTHIA8', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)
    
    def test_shower_card_hwpp(self):
        """test that the hwpp card is correctly written"""
        goal = \
"""NEVENTS=-1
UE_HWPP=.FALSE.
PDFCODE=0
HADRONIZE_HWPP=.TRUE.
MAXPR_HWPP=2
ERR_FR_HWPP=0.100
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
LAMBDAHERW=-1.000
B_MASS=-1.000
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
"""NEVENTS=-1
UE_HWPP=.FALSE.
PDFCODE=0
HADRONIZE_HWPP=.TRUE.
MAXPR_HWPP=2
ERR_FR_HWPP=0.100
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
LAMBDAHERW=-1.000
B_MASS=-1.000
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWPPUTI=""
"""
        text = self.card_analyse.write_card('HERWIGPP', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_hw6(self):
        """test that the hw6 card is correctly written"""
        goal = \
"""NEVENTS=-1
LHSOFT=.FALSE.
PDFCODE=0
MAXPR_HW=2
ERR_FR_HW=0.100
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
B_MASS=-1.000
IS_BB_HW=.FALSE.
MODBOS_1=5
MODBOS_2=5
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
"""NEVENTS=-1
LHSOFT=.FALSE.
PDFCODE=0
MAXPR_HW=2
ERR_FR_HW=0.100
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
B_MASS=-1.000
IS_BB_HW=.FALSE.
MODBOS_1=5
MODBOS_2=5
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
HWUTI="mcatnlo_hwan_stdhep.o"
"""
        text = self.card_analyse.write_card('HERWIG6', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)


    def test_shower_card_py6(self):
        """test that the py6 card is correctly written"""
        goal = \
"""NEVENTS=-1
MSTP_81=0
PDFCODE=0
MSTP_111=1
MAXPR_PY=2
ERR_FR_PY=0.100
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
B_MASS=-1.000
IS_4L_PY=.FALSE.
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
"""NEVENTS=-1
MSTP_81=0
PDFCODE=0
MSTP_111=1
MAXPR_PY=2
ERR_FR_PY=0.100
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
B_MASS=-1.000
IS_4L_PY=.FALSE.
EXTRALIBS="stdhep Fmcfio"
EXTRAPATHS="../lib"
INCLUDEPATHS=
PYUTI="mcatnlo_pyan_stdhep.o"
"""
        text = self.card_analyse.write_card('PYTHIA6Q', '')
        for a, b in zip(text.split('\n'), goal.split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(text, goal)

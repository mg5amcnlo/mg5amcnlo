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
"""#***********************************************************************
#                        MadGraph5_aMC@NLO                             *
#                                                                      *
#                      shower_card.dat aMC@NLO                         *
#                                                                      *
#  This file is used to set the parameters for the shower.             *
#                                                                      *
#  Some notation/conventions:                                          *
#                                                                      *
#   Lines starting with a hash (#) are info or comments                *
#                                                                      *
#   mind the format:   variable    = value     # comment               *
#***********************************************************************
#
#****************
# Shower settings
#****************
#
#***********************************************************************
# Number of events, jobs, errors, and random seeds                     *
#***********************************************************************
nevents      = -1     # N evts to shower (< 0 = all)
nsplit_jobs  = 1      # N jobs to run in parallel (< 100!!)
combine_td   = T      # combine the topdrawer files if nsplit_jobs > 1
maxprint     = 2      # N evts to print in the log
maxerrs      = 0.1    # max fraction of errors
rnd_seed     = 0      # 1st random seed (0 = default)
rnd_seed2    = 0      # 2nd random seed (0 = default) !ONLY FOR HWERIG6!
#***********************************************************************
# PDFs and non-perturbative modelling                                  *
#***********************************************************************
pdfcode      = 0      # 0 = internal, 1 = same as NLO, other = lhaglue
ue_enabled   = F      # underlying event
hadronize    = T      # hadronisation on/off        !IGNORED BY HERWIG6!
lambda_5     = -1     # Lambda_5 (< 0 = default)    !IGNORED BY PYTHIA8!
#***********************************************************************
# Stable or unstable particles                                         *
#***********************************************************************
b_stable     = F      # set B hadrons stable
pi_stable    = T      # set pi0's stable
wp_stable    = F      # set w+'s stable
wm_stable    = F      # set w-'s stable
z_stable     = F      # set z0's stable
h_stable     = F      # set Higgs' stable
tap_stable   = F      # set tau+'s stable
tam_stable   = F      # set tau-'s stable
mup_stable   = F      # set mu+'s stable
mum_stable   = F      # set mu-'s stable
#***********************************************************************
# Mass of the b quark                                                  *
#***********************************************************************
b_mass       = -1     # b mass, (< 0 = default)
#***********************************************************************
# Special settings                                                     *
#***********************************************************************
is_4lep      = F    # T if 4-lepton production        !ONLY FOR PYTHIA6!
is_bbar      = F    # T if bb~ production             !ONLY FOR HERWIG6!
#***********************************************************************
# Decay channels                                                       *
# Write down the decay channels for the resonances, to be performed by *
# the shower.                                                          *
# The syntax (for a two-body decay) is                                 *
# DM_I = M > D1 D2 @ BR @ ME                                           *
# where I < 100, M is the decaying resonance, D1, D2 are the decay     *
# products (up to D5 if such a decay is supported by the shower), BR   *
# is the branching ratio (only used by the HERWIG6 shower, ignored     *
# otherwise) and ME is the type of matrix element to be used in the    *
# decay (only used by HERWIG6, ignored otherwise).                     *
# BR's are correctly understood by HERWIG6 only if they add up to 1    *
# and only if no more than three modes are required for a given        *
# resonance.                                                           *
# ME corresponds to the third entry of subroutine HWMODK, see the      *
# relevant manual.                                                     *
#                                                                      *
# WARNING: in HERWIG6, the order of decay products in > 2-body decays  *
# IS RELEVANT.                                                         *
# WARNING: in PYTHIA6, turning hadronisation off disables top decays   *
# WARNING: in PYTHIA6 and PYTHIA8, 1 -> n decays (with n > 2) are      *
# handled through a sequence of 1 -> 2 decays.                         *
#                                                                      *
# Examples of syntax:                                                  *
# Z -> e+ e- or mu+ mu- with BR = 0.5 each                             *
# DM_1 = 23 > -11 11 @ 0.5d0 @ 100                                     *
# DM_2 = 23 > -13 13 @ 0.5d0 @ 100                                     *
# H -> tau+ tau- with BR = 1                                           *
# DM_3 = 25 > -15 15 @ 1.0d0 @ 0                                       *
# t -> nu_e e+ b with BR = 1 (HERWIG)                                  *
# DM_4 = 6 > 12 -11 5 @ 1d0 @ 100                                      *
# t -> nu_e e+ b with BR = 1 (PYTHIA)                                  *
# DM_5 = 6 > 24 5 @ 1d0 @ 100                                          *
# DM_6 = 24 > 12 -11 @ 1d0 @ 100                                       *
#***********************************************************************

#***********************************************************************
# Extra Libraries/analyses                                             *
# The following lines need to be changed if the user does not want to  *
# create a StdHEP/HepMC file, but to directly run an own analysis (to  *
# be placed in HWAnalyzer or analogous MCatNLO subfolders).            *
# Please use files in those folders as examples.                       *
#***********************************************************************
EXTRALIBS    = stdhep Fmcfio     # Extra-libraries (not LHAPDF) 
                                 # Default: "stdhep Fmcfio"
                                 # PYTHIA > 8.200 may require library dl
EXTRAPATHS   = ../lib            # Path to the extra-libraries
                                 # Default: "../lib"
INCLUDEPATHS =                   # Path to header files needed by c++
                                 # Dir names separated by white spaces
ANALYSE      =                   # User's analysis and histogramming
                                 # routines (please use .o as extension
                                 # and use spaces to separate files)

"""
            TestShowerCard.card = shower_card.ShowerCard(text, testing = True) 

            text_analyse = \
"""#***********************************************************************
#                        MadGraph5_aMC@NLO                             *
#                                                                      *
#                      shower_card.dat aMC@NLO                         *
#                                                                      *
#  This file is used to set the parameters for the shower.             *
#                                                                      *
#  Some notation/conventions:                                          *
#                                                                      *
#   Lines starting with a hash (#) are info or comments                *
#                                                                      *
#   mind the format:   variable    = value     # comment               *
#***********************************************************************
#
#****************
# Shower settings
#****************
#
#***********************************************************************
# Number of events, jobs, errors, and random seeds                     *
#***********************************************************************
nevents      = -1     # N evts to shower (< 0 = all)
nsplit_jobs  = 1      # N jobs to run in parallel (< 100!!)
combine_td   = T      # combine the topdrawer files if nsplit_jobs > 1
maxprint     = 2      # N evts to print in the log
maxerrs      = 0.1    # max fraction of errors
rnd_seed     = 0      # 1st random seed (0 = default)
rnd_seed2    = 0      # 2nd random seed (0 = default) !ONLY FOR HWERIG6!
#***********************************************************************
# PDFs and non-perturbative modelling                                  *
#***********************************************************************
pdfcode      = 0      # 0 = internal, 1 = same as NLO, other = lhaglue
ue_enabled   = F      # underlying event
hadronize    = T      # hadronisation on/off        !IGNORED BY HERWIG6!
lambda_5     = -1     # Lambda_5 (< 0 = default)    !IGNORED BY PYTHIA8!
#***********************************************************************
# Stable or unstable particles                                         *
#***********************************************************************
b_stable     = F      # set B hadrons stable
pi_stable    = T      # set pi0's stable
wp_stable    = F      # set w+'s stable
wm_stable    = F      # set w-'s stable
z_stable     = F      # set z0's stable
h_stable     = F      # set Higgs' stable
tap_stable   = F      # set tau+'s stable
tam_stable   = F      # set tau-'s stable
mup_stable   = F      # set mu+'s stable
mum_stable   = F      # set mu-'s stable
#***********************************************************************
# Mass of the b quark                                                  *
#***********************************************************************
b_mass       = -1     # b mass, (< 0 = default)
#***********************************************************************
# Special settings                                                     *
#***********************************************************************
is_4lep      = F    # T if 4-lepton production        !ONLY FOR PYTHIA6!
is_bbar      = F    # T if bb~ production             !ONLY FOR HERWIG6!
#***********************************************************************
# Decay channels                                                       *
# Write down the decay channels for the resonances, to be performed by *
# the shower.                                                          *
# The syntax (for a two-body decay) is                                 *
# DM_I = M > D1 D2 @ BR @ ME                                           *
# where I < 100, M is the decaying resonance, D1, D2 are the decay     *
# products (up to D5 if such a decay is supported by the shower), BR   *
# is the branching ratio (only used by the HERWIG6 shower, ignored     *
# otherwise) and ME is the type of matrix element to be used in the    *
# decay (only used by HERWIG6, ignored otherwise).                     *
# BR's are correctly understood by HERWIG6 only if they add up to 1    *
# and only if no more than three modes are required for a given        *
# resonance.                                                           *
# ME corresponds to the third entry of subroutine HWMODK, see the      *
# relevant manual.                                                     *
#                                                                      *
# WARNING: in HERWIG6, the order of decay products in > 2-body decays  *
# IS RELEVANT.                                                         *
# WARNING: in PYTHIA6, turning hadronisation off disables top decays   *
# WARNING: in PYTHIA6 and PYTHIA8, 1 -> n decays (with n > 2) are      *
# handled through a sequence of 1 -> 2 decays.                         *
#                                                                      *
# Examples of syntax:                                                  *
# Z -> e+ e- or mu+ mu- with BR = 0.5 each                             *
# DM_1 = 23 > -11 11 @ 0.5d0 @ 100                                     *
# DM_2 = 23 > -13 13 @ 0.5d0 @ 100                                     *
# H -> tau+ tau- with BR = 1                                           *
# DM_3 = 25 > -15 15 @ 1.0d0 @ 0                                       *
# t -> nu_e e+ b with BR = 1 (HERWIG)                                  *
# DM_4 = 6 > 12 -11 5 @ 1d0 @ 100                                      *
# t -> nu_e e+ b with BR = 1 (PYTHIA)                                  *
# DM_5 = 6 > 24 5 @ 1d0 @ 100                                          *
# DM_6 = 24 > 12 -11 @ 1d0 @ 100                                       *
#***********************************************************************

#***********************************************************************
# Extra Libraries/analyses                                             *
# The following lines need to be changed if the user does not want to  *
# create a StdHEP/HepMC file, but to directly run an own analysis (to  *
# be placed in HWAnalyzer or analogous MCatNLO subfolders).            *
# Please use files in those folders as examples.                       *
#***********************************************************************
EXTRALIBS    = stdhep Fmcfio     # Extra-libraries (not LHAPDF) 
                                 # Default: "stdhep Fmcfio"
                                 # PYTHIA > 8.200 may require library dl
EXTRAPATHS   = ../lib            # Path to the extra-libraries
                                 # Default: "../lib"
INCLUDEPATHS =                   # Path to header files needed by c++
                                 # Dir names separated by white spaces
ANALYSE      =                   # User's analysis and histogramming
                                 # routines (please use .o as extension
                                 # and use spaces to separate files)

"""
            TestShowerCard.card_analyse = shower_card.ShowerCard(text_analyse, testing = True) 


    def test_shower_card_py8(self):
        """test that the py8 card is correctly written"""
        goal = \
"""NEVENTS=-1
MAXPR_PY8=2
ERR_FR_PY8=0.100
RNDEVSEED_PY8=0
PDFCODE=0
UE_PY8=.FALSE.
HADRONIZE_PY8=.TRUE.
LAMBDAPYTH=-1.000
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
MAXPR_PY8=2
ERR_FR_PY8=0.100
RNDEVSEED_PY8=0
PDFCODE=0
UE_PY8=.FALSE.
HADRONIZE_PY8=.TRUE.
LAMBDAPYTH=-1.000
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
MAXPR_HWPP=2
ERR_FR_HWPP=0.100
RNDEVSEED_HWPP=0
PDFCODE=0
UE_HWPP=.FALSE.
HADRONIZE_HWPP=.TRUE.
LAMBDAHERW=-1.000
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
MAXPR_HWPP=2
ERR_FR_HWPP=0.100
RNDEVSEED_HWPP=0
PDFCODE=0
UE_HWPP=.FALSE.
HADRONIZE_HWPP=.TRUE.
LAMBDAHERW=-1.000
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
MAXPR_HW=2
ERR_FR_HW=0.100
RNDEVSEED1_HW=0
RNDEVSEED2_HW=0
PDFCODE=0
LHSOFT=.FALSE.
LAMBDAHERW=-1.000
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
B_MASS=-1.000
IS_BB_HW=.FALSE.
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
MAXPR_HW=2
ERR_FR_HW=0.100
RNDEVSEED1_HW=0
RNDEVSEED2_HW=0
PDFCODE=0
LHSOFT=.FALSE.
LAMBDAHERW=-1.000
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
B_MASS=-1.000
IS_BB_HW=.FALSE.
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
MAXPR_PY=2
ERR_FR_PY=0.100
RNDEVSEED_PY=0
PDFCODE=0
MSTP_81=0
MSTP_111=1
LAMBDAPYTH=-1.000
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
MAXPR_PY=2
ERR_FR_PY=0.100
RNDEVSEED_PY=0
PDFCODE=0
MSTP_81=0
MSTP_111=1
LAMBDAPYTH=-1.000
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

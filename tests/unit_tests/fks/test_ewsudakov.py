##############################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import absolute_import
from __future__ import print_function
from cmd import Cmd
from six.moves import zip
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as amcatnlocmd
import os
import madgraph.fks.fks_helas_objects as fks_helas
import copy
import madgraph.iolibs.save_load_object as save_load_object


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestSudakov(unittest.TestCase):
    """ a suite of extra tests for the ew stuff """
    
    def setUp(self):
        self.interface = mgcmd.MasterCmd()
        self.interface.do_import('model loop_qcd_qed_sm_Gmu_forSudakov')

    def test_generate_ewsud_ttbar(self):
        """check that the generate command works as expected.
        In particular the correct number of matrix elements needed for the sudakov 
        piece
        """
        cmd = 'u u~ > t t~ [LOonly=QCD] --ewsudakov'

        self.interface.do_generate(cmd)

        fksprocess = self.interface._fks_multi_proc['born_processes'][0]
        self.assertEqual(len(fksprocess.sudakov_amps), 2)
        types = set([amp['type'] for amp in fksprocess.sudakov_amps])
        self.assertEqual(list(types), ['ipm2']) # only ipm2 type-amplitudes for this process


    def test_generate_ewsud_ww(self):
        """check that the generate command works as expected.
        In particular the correct number of matrix elements needed for the sudakov 
        piece
        """
        cmd = 'u u~ > w+ w- [LOonly=QCD] --ewsudakov'

        self.interface.do_generate(cmd)

        fksprocess = self.interface._fks_multi_proc['born_processes'][0]
        self.assertEqual(len(fksprocess.sudakov_amps), 29)
        types = [amp['type'] for amp in fksprocess.sudakov_amps]
        self.assertEqual(types.count('goldstone'), 3) # 3 goldstone amps (gw,wg,ww)
        self.assertEqual(types.count('ipm2'), 26) # 26


    def test_generate_ewsud_zz(self):
        """check that the generate command works as expected.
        In particular the correct number of matrix elements needed for the sudakov 
        piece
        """
        cmd = 'u u~ > z z [LOonly=QCD] --ewsudakov'

        self.interface.do_generate(cmd)

        fksprocess = self.interface._fks_multi_proc['born_processes'][0]
        self.assertEqual(len(fksprocess.sudakov_amps), 9)
        types = [amp['type'] for amp in fksprocess.sudakov_amps]
        self.assertEqual(types.count('goldstone'), 0) # 0 goldstone amps (no coupling to light quarks)
        self.assertEqual(types.count('ipm2'), 7) # 
        self.assertEqual(types.count('cew'), 2) # cew amplitude for the z-a mixing 





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
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as amcatnlocmd
import os


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestAMCatNLOEW(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    interface = mgcmd.MasterCmd()

    def test_generate_fks_ew(self):
        """check that the generate command works as expected.
        In particular the correct number of born diagrams, real-emission processes
        and diagrams is checked"""
        cmd_list = [
            'u u~ > d d~ QED=0 QCD=2 [real=QCD]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED QCD]',
            'u u~ > d d~  [real=QCD]',
            'u u~ > d d~  [real=QED]',
            'u u~ > d d~  [real=QED QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED QCD]']

        # expected born_orders
        born_orders_list = [{'QED':0, 'QCD':2},
                            {'QED':0, 'QCD':2},
                            {'QED':0, 'QCD':2},
                            {'WEIGHTED': 2},
                            {'WEIGHTED': 2},
                            {'WEIGHTED': 2},
                            {'QED':2, 'QCD':2},
                            {'QED':2, 'QCD':2},
                            {'QED':2, 'QCD':2}]

        # perturbation couplings
        pert_couplings_list = [['QCD'],
                               ['QED'],
                               ['QED','QCD'],
                               ['QCD'],
                               ['QED'],
                               ['QED','QCD'],
                               ['QCD'],
                               ['QED'],
                               ['QED','QCD']]

        # expected squared_orders (should take into
        #  account the perturbation
        squared_orders_list = [{'QED':0, 'QCD':6},
                            {'QED':2, 'QCD':4},
                            {'QED':2, 'QCD':6},
                            {'WEIGHTED': 6},
                            {'WEIGHTED': 8},
                            {'WEIGHTED': 8},
                            {'QED':4, 'QCD':6},
                            {'QED':6, 'QCD':4},
                            {'QED':6, 'QCD':6}]

        # number of expected born diagrams
        # 1 QCD diagram and 3 EW ones
        nborndiag_list = [1, 4, 4, 1, 4, 4, 4, 4, 4]

        # number of expected real emission processes
        # for QED perturbations also the gluon emissions have to be generated, 
        # as their alpha x alpha_s^2 contribution has to be included
        nrealproc_list = [3, 6, 6, 3, 6, 6, 6, 6, 6]

        # number of expected real emission diagrams
        # u u~ > d d~ g has 5 with QED=0, 17 with QED=2
        # u u~ > d d~ a has 4 with QED=1, 17 with QED=3
        #
        # for e.g. 'u u~ > d d~ QED=0 QCD=2 [real=QED]'
        # the real emissions are ordered as follows:
        #   u u~ > d d~ a [ QED ] QED=2 QCD=4
        #   a u~ > d d~ u~ [ QED ] QED=2 QCD=4
        #   u u~ > d d~ g [ QED ] QED=2 QCD=4
        #   g u~ > d d~ u~ [ QED ] QED=2 QCD=4
        #   u a > d d~ u [ QED ] QED=2 QCD=4
        #   u g > d u d~ [ QED ] QED=2 QCD=4
        nrealdiags_list = [[5, 5, 5],
                           [4, 4, 17, 17, 4, 17],
                           [4, 4, 17, 17, 4, 17],
                           [5, 5, 5],
                           [4, 4, 17, 17, 4, 17],
                           [4, 4, 17, 17, 4, 17],
                           [17, 17, 17, 17, 17, 17],
                           [17, 17, 17, 17, 17, 17],
                           [17, 17, 17, 17, 17, 17]]

        for cmd, born_orders, squared_orders, pert_couplings, nborndiag, nrealproc, nrealdiags in \
                zip(cmd_list, born_orders_list, squared_orders_list, pert_couplings_list, nborndiag_list, 
                        nrealproc_list, nrealdiags_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]
            # check that the extra_cnt_amp_list is empty
            self.assertEqual(0, len(fksprocess.extra_cnt_amp_list))

            self.assertEqual(born_orders, fksprocess.born_amp['process']['born_orders'])
            self.assertEqual(squared_orders, fksprocess.born_amp['process']['squared_orders'])
            self.assertEqual(pert_couplings, fksprocess.born_amp['process']['perturbation_couplings'])

            self.assertEqual(len(fksprocess.born_amp['diagrams']), nborndiag)
            self.assertEqual(len(fksprocess.real_amps), nrealproc)
            for amp, n in zip(fksprocess.real_amps, nrealdiags):
                # check that the fks_j_from i have also been set
                self.assertNotEqual(amp.fks_j_from_i, {})
                self.assertEqual(n, len(amp.amplitude['diagrams']))


    def test_generate_fks_ew_extra_moms(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed"""
        cmd_list = [
            'u u~ > g g [real=QED QCD]']

        len_extra_cnt_amp_list = [0]
        for cmd, len_extra_cnt in zip(cmd_list, len_extra_cnt_amp_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]

            self.assertEqual(len_extra_cnt, len(fksprocess.extra_cnt_amp_list))

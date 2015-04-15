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
import madgraph.fks.fks_helas_objects as fks_helas
import copy


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestAMCatNLOEW(unittest.TestCase):
    """ a suite of extra tests for the ew stuff """
    
    interface = mgcmd.MasterCmd()

    def test_generate_fks_ew(self):
        """check that the generate command works as expected.
        In particular the correct number of born diagrams, real-emission processes
        and diagrams is checked"""
        cmd_list = [
            'u u~ > d d~ QED=0 QCD=2 [real=QCD]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED]',
            'u u~ > d d~ QED=0 QCD=2 [real=QED QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QCD]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED]',
            'u u~ > d d~ QCD=2 QED=2 [real=QED QCD]']

        # expected born_orders
        born_orders_list = [{'QED':0, 'QCD':2},
                            {'QED':0, 'QCD':2},
                            {'QED':0, 'QCD':2},
                            {'QED':2, 'QCD':2},
                            {'QED':2, 'QCD':2},
                            {'QED':2, 'QCD':2}]

        # perturbation couplings (always set to [QED, QCD]
        pert_couplings_list = 9*[['QED','QCD']]

        # expected squared_orders (should take into
        #  account the perturbation
        squared_orders_list = [{'QED':0, 'QCD':6},
                            {'QED':2, 'QCD':4},
                            {'QED':2, 'QCD':6},
                            {'QED':4, 'QCD':6},
                            {'QED':6, 'QCD':4},
                            {'QED':6, 'QCD':6}]

        # number of expected born diagrams
        # 1 QCD diagram and 3 EW ones
        nborndiag_list = [1, 4, 4, 4, 4, 4]

        # number of expected real emission processes
        # for QED perturbations also the gluon emissions have to be generated, 
        # as their alpha x alpha_s^2 contribution has to be included
        nrealproc_list = [3, 6, 6, 6, 6, 6]

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
                # and that no extra counterterm is needed
                for info in amp.fks_infos:
                    self.assertEqual(info['extra_cnt_index'], -1)
                    self.assertEqual(len(info['underlying_born']), 1)
                    self.assertEqual(len(info['splitting_type']), 1)


    def test_generate_fks_ew_extra_cnts_ttx_full(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed"""

        self.interface.do_define('p p a')
        self.interface.do_generate('p p > t t~ QED=2 QCD=2 [real=QCD QED]')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        #there should be 12 processes: 4 qqbar + 4qbarq + gg + ga + ag + aa  
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']),12 )
        for proc in self.interface._fks_multi_proc['born_processes']:
            init_id = [l['id'] for l in proc.born_amp['process']['legs'] if not l['state']]
            # gg initial state, 2 extra cnts
            if init_id == [21, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 2)
                # 18 real emission MEs
                self.assertEqual(len(proc.real_amps), 18)
                # 22 real FKS configurations
                self.assertEqual(len(sum([real.fks_infos for real in proc.real_amps], [])), 22)
                for real in proc.real_amps:
                    init_real_id = real.pdgs[:2]
                    if any([idd in quarks for idd in init_real_id]):
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # ag initial state, 1 extra cnt
            elif init_id == [21, 22] or init_id == [22, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 1)
                # 14 real emission MEs (add mu/e leptons, photon splitting into quarks
                # must not be incuded here)
                self.assertEqual(len(proc.real_amps), 14)
                # 17 real FKS configurations
                self.assertEqual(len(sum([real.fks_infos for real in proc.real_amps], [])), 17)
                for real in proc.real_amps:
                    init_real_id = real.pdgs[:2]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in init_real_id]) and 22 in init_real_id:
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # all others no cnts
            # aa 
            elif init_id == [22, 22]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # 14 real emission MEs (add mu/e leptons, photon splitting into quarks
                # must not be incuded here)
                self.assertEqual(len(proc.real_amps), 10)
                for real in proc.real_amps:
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # qq
            else:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                for real in proc.real_amps:
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)


    def test_generate_fks_ew_extra_cnts_ttx_qed2qcd1(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        In this case the extra counterterms/splittings should not be 
        included in the gg since it is only needed for counterterms"""

        self.interface.do_define('p p a')
        self.interface.do_generate('p p > t t~ QED=2 QCD=1 [real=QCD QED]')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        #there should be 12 processes: 4 qqbar + 4qbarq + gg + ga + ag + aa  
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']),12 )
        for proc in self.interface._fks_multi_proc['born_processes']:
            init_id = [l['id'] for l in proc.born_amp['process']['legs'] if not l['state']]
            # gg initial state, no extra cnts
            if init_id == [21, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # 1 real emission MEs (only extra photon radiation
                self.assertEqual(len(proc.real_amps), 1)
                # 2 real FKS configurations
                self.assertEqual(len(sum([real.fks_infos for real in proc.real_amps], [])), 2)
                for real in proc.real_amps:
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # ag initial state, 1 extra cnt
            elif init_id == [21, 22] or init_id == [22, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 1)
                # 22 real emission MEs (add mu/e leptons, photon splitting into quarks
                # MUST be incuded here)
                self.assertEqual(len(proc.real_amps), 22)
                # 25 real FKS configurations
                self.assertEqual(len(sum([real.fks_infos for real in proc.real_amps], [])), 25)
                for real in proc.real_amps:
                    init_real_id = real.pdgs[:2]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in init_real_id]) and 22 in init_real_id:
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # all others no cnts
            # aa 
            elif init_id == [22, 22]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # 14 real emission MEs (add mu/e leptons, photon splitting into quarks
                # must not be incuded here)
                self.assertEqual(len(proc.real_amps), 10)
                for real in proc.real_amps:
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            # qq
            else:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                for real in proc.real_amps:
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)


    def test_combine_equal_processes_qcd_qed(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected"""
        newinterface = mgcmd.MasterCmd()
        newinterface2 = mgcmd.MasterCmd()
        # generate the processes
        self.interface.do_generate('u u~ > t t~ QED=0 QCD=2 [real=QCD QED]')
        newinterface.do_generate('c c~ > t t~ QED=0 QCD=2 [real=QCD QED]')
        newinterface2.do_generate('d d~ > t t~ QED=0 QCD=2 [real=QCD QED]')

        fksproc1 = self.interface._fks_multi_proc
        fksproc2 = newinterface._fks_multi_proc
        fksproc3 = newinterface2._fks_multi_proc
        fksme1 = fks_helas.FKSHelasMultiProcess(fksproc1)['matrix_elements'][0]
        fksme2 = fks_helas.FKSHelasMultiProcess(fksproc2)['matrix_elements'][0]
        fksme3 = fks_helas.FKSHelasMultiProcess(fksproc3)['matrix_elements'][0]

        self.assertNotEqual(fksme2,fksme3)
        
        # check the reals, they should be in the same order
        for i1, r1, in enumerate(fksme1.real_processes): 
            for i2, r2, in enumerate(fksme2.real_processes): 
                if i1 == i2:
                    self.assertEqual(r1,r2)
                else:
                    self.assertNotEqual(r1,r2)
        self.assertEqual(fksme1, fksme2)

        fksme1.add_process(fksme2)

        self.assertEqual(len(fksme1.born_me['processes']), 2)
        for real in fksme1.real_processes:
            self.assertEqual(len(real.matrix_element['processes']), 2)


    def test_combine_equal_processes_qcd_qed_virt(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected. 
        This test also cehck that equality works for virtuals.
        In particular b-initiate processes have same trees but different loops (w/top)"""
        # generate the processes

        self.interface.do_import('model loop_qcd_qed_sm-no_widths')

        self.interface.do_generate('u u~ > g g QED=0 QCD=2 [QCD QED]')
        fksproc1 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('c c~ > g g QED=0 QCD=2 [QCD QED]')
        fksproc2 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('d d~ > g g QED=0 QCD=2 [QCD QED]')
        fksproc3 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('b b~ > g g QED=0 QCD=2 [QCD QED]')
        fksproc4 = copy.copy(self.interface._fks_multi_proc)

        # this is to avoid effects on other tests
        self.interface.do_import('model sm')

        fksme1 = fks_helas.FKSHelasMultiProcess(fksproc1)['matrix_elements'][0]
        fksme2 = fks_helas.FKSHelasMultiProcess(fksproc2)['matrix_elements'][0]
        fksme3 = fks_helas.FKSHelasMultiProcess(fksproc3)['matrix_elements'][0]
        fksme4 = fks_helas.FKSHelasMultiProcess(fksproc4)['matrix_elements'][0]

        # check that the u and d initiated are not equal
        self.assertNotEqual(fksme2,fksme3)

        # check that the b-initiated is different from all other processes
        self.assertNotEqual(fksme1,fksme4)
        self.assertNotEqual(fksme2,fksme4)
        self.assertNotEqual(fksme3,fksme4)
        
        # check that the u and c initiated are equal
        self.assertEqual(fksme1, fksme2)



    def test_combine_equal_processes_qcd(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected"""
        newinterface = mgcmd.MasterCmd()
        newinterface2 = mgcmd.MasterCmd()
        # generate the processes
        self.interface.do_generate('u u~ > t t~ QED=0 QCD=2 [real=QCD]')
        newinterface.do_generate('d d~ > t t~ QED=0 QCD=2 [real=QCD]')

        fksproc1 = self.interface._fks_multi_proc
        fksproc2 = newinterface._fks_multi_proc
        fksme1 = fks_helas.FKSHelasMultiProcess(fksproc1)['matrix_elements'][0]
        fksme2 = fks_helas.FKSHelasMultiProcess(fksproc2)['matrix_elements'][0]
        
        # check the reals, they should be in the same order
        for i1, r1, in enumerate(fksme1.real_processes): 
            for i2, r2, in enumerate(fksme2.real_processes): 
                if i1 == i2:
                    self.assertEqual(r1,r2)
                else:
                    self.assertNotEqual(r1,r2)
        self.assertEqual(fksme1, fksme2)

        fksme1.add_process(fksme2)

        self.assertEqual(len(fksme1.born_me['processes']), 2)
        for real in fksme1.real_processes:
            self.assertEqual(len(real.matrix_element['processes']), 2)

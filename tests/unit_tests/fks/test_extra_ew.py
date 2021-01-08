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

class TestAMCatNLOEW(unittest.TestCase):
    """ a suite of extra tests for the ew stuff """
    
    def setUp(self):
        self.interface = mgcmd.MasterCmd()

    def test_generate_fks_ew(self):
        """check that the generate command works as expected.
        In particular the correct number of born diagrams, real-emission processes
        and diagrams is checked"""
        cmd_list = [
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QCD]',
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QED]',
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QED QCD]',
            'u u~ > d d~ QCD^2=4 QED^2=4 [real=QCD]',
            'u u~ > d d~ QCD^2=4 QED^2=4 [real=QED]',
            'u u~ > d d~ QCD^2=4 QED^2=4 [real=QED QCD]']

        # exp[ected splitting types
        split_type_list = [['QCD'],
                           ['QED','QCD'],
                           ['QED','QCD'],
                           ['QED','QCD'],
                           ['QED','QCD'],
                           ['QED','QCD']]

        # expected born_orders
        born_orders_list = [{'QED':0, 'QCD':4},
                            {'QED':0, 'QCD':4},
                            {'QED':0, 'QCD':4},
                            {'QED':4, 'QCD':4},
                            {'QED':4, 'QCD':4},
                            {'QED':4, 'QCD':4}]

        # perturbation couplings (always set to [QED, QCD]
        pert_couplings_list = 9*[['QCD','QED']]

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
        # u u~ > d d~ g has 5 with QED^2=0, 17 with QED^2=4
        # u u~ > d d~ a has 4 with QED^2=2, 17 with QED^2=6
        #
        # for e.g. 'u u~ > d d~ QED^2=0 QCD^2=2 [real=QED]'
        # the real emissions are ordered as follows:
        #   u u~ > d d~ a [ QED ] QED^2=4 QCD^2=8
        #   a u~ > d d~ u~ [ QED ] QED^2=4 QCD^2=8
        #   u u~ > d d~ g [ QED ] QED^2=4 QCD^2=8
        #   g u~ > d d~ u~ [ QED ] QED^2=4 QCD^2=8
        #   u a > d d~ u [ QED ] QED^2=4 QCD^2=8
        #   u g > d u d~ [ QED ] QED^2=4 QCD^2=8
        nrealdiags_list = [[5, 5, 5],
                           [4, 4, 17, 17, 4, 17],
                           [4, 4, 17, 17, 4, 17],
                           [17, 17, 17, 17, 17, 17],
                           [17, 17, 17, 17, 17, 17],
                           [17, 17, 17, 17, 17, 17]]

        for cmd, born_orders, squared_orders, pert_couplings, nborndiag, nrealproc, nrealdiags, split in \
                zip(cmd_list, born_orders_list, squared_orders_list, pert_couplings_list, nborndiag_list, 
                        nrealproc_list, nrealdiags_list, split_type_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]

            # check that the extra_cnt_amp_list is empty
            self.assertEqual(0, len(fksprocess.extra_cnt_amp_list))

            self.assertEqual(born_orders, fksprocess.born_amp['process']['born_sq_orders'])
            self.assertEqual(squared_orders, fksprocess.born_amp['process']['squared_orders'])
            self.assertEqual(sorted(pert_couplings), sorted(fksprocess.born_amp['process']['perturbation_couplings']))

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


    def test_generate_fks_ew_noorders(self):
        """check that the generate command works as expected when no (or just some)
        orders are set.
        """
        cmd_list = [
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QED]',
            'u u~ > d d~ QCD^2=4 [real=QED]',
            'u u~ > d d~ [real=QED]',
            \
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QED QCD]',
            'u u~ > d d~ QCD^2=4 [real=QED QCD]',
            'u u~ > d d~ [real=QED QCD]',
            \
            'u u~ > d d~ QED^2=0 QCD^2=4 [real=QCD]',
            'u u~ > d d~ QCD^2=4 [real=QCD]',
            'u u~ > d d~ [real=QCD]']

        # expected born_orders
        born_orders_list = 9*[{'QED':0, 'QCD':4}]

        # perturbation couplings (always set to [QED, QCD]
        pert_couplings_list = 9*[['QCD','QED']]

        # expected squared_orders (should take into
        #  account the perturbation
        squared_orders_list = 3*[{'QED':2, 'QCD':4}] + \
                              3*[{'QED':2, 'QCD':6}] + \
                              3*[{'QED':0, 'QCD':6}]

        # number of expected born diagrams
        # 1 QCD diagram and 3 EW ones
        nborndiag_list = [4, 4, 4, 4, 4, 4, 1, 1, 1]

        # number of expected real emission processes
        # for QED perturbations also the gluon emissions have to be generated, 
        # as their alpha x alpha_s^2 contribution has to be included
        nrealproc_list = [6, 6, 6, 6, 6, 6, 3, 3, 3]

        # number of expected real emission diagrams
        # u u~ > d d~ g has 5 with QED^2=0, 17 with QED^2=4
        # u u~ > d d~ a has 4 with QED^2=2, 17 with QED^2=6
        #
        # for e.g. 'u u~ > d d~ QED^2=0 QCD^2=4 [real=QED]'
        # the real emissions are ordered as follows:
        #   u u~ > d d~ a [ QED ] QED^2=4 QCD^2=8
        #   a u~ > d d~ u~ [ QED ] QED^2=4 QCD^2=8
        #   u u~ > d d~ g [ QED ] QED^2=4 QCD^2=8
        #   g u~ > d d~ u~ [ QED ] QED^2=4 QCD^2=8
        #   u a > d d~ u [ QED ] QED^2=4 QCD^2=8
        #   u g > d u d~ [ QED ] QED^2=4 QCD^2=8
        nrealdiags_list = 3*[[4, 4, 17, 17, 4, 17]] + \
                          3*[[4, 4, 17, 17, 4, 17]] + \
                          3*[[5, 5, 5]]

        for cmd, born_orders, squared_orders, pert_couplings, nborndiag, nrealproc, nrealdiags in \
                zip(cmd_list, born_orders_list, squared_orders_list, pert_couplings_list, nborndiag_list, 
                        nrealproc_list, nrealdiags_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]

            # no orders should be specified, only squared orders
            self.assertEqual(fksprocess.born_amp['process']['orders'], {})

            # check that the extra_cnt_amp_list is empty
            self.assertEqual(0, len(fksprocess.extra_cnt_amp_list))

            self.assertEqual(born_orders, fksprocess.born_amp['process']['born_sq_orders'])
            self.assertEqual(squared_orders, fksprocess.born_amp['process']['squared_orders'])
            self.assertEqual(sorted(pert_couplings), sorted(fksprocess.born_amp['process']['perturbation_couplings']))

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

        self.interface.do_set('include_lepton_initiated_processes True')
        self.interface.do_define('p p a')
        self.interface.do_generate('p p > t t~ QED^2=4 QCD^2=4 [real=QCD QED]')
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
        # avoid border effects
        self.interface.do_set('include_lepton_initiated_processes False')


    def test_generate_fks_ew_extra_cnts_ttx_qed2qcd1(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        In this case the extra counterterms/splittings should not be 
        included in the gg since it is only needed for counterterms"""

        self.interface.do_set('include_lepton_initiated_processes True')
        self.interface.do_define('p p a')
        self.interface.do_generate('p p > t t~ QED^2=4 QCD^2=2 [real=QCD QED]')
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
        # avoid border effects
        self.interface.do_set('include_lepton_initiated_processes False')


# other tests for the extra cnts for dijet
# integrate one LO contribution with one kind of correction
    def test_generate_fks_ew_dijet_qed0qcd2_qcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QCD corrections to the leftmost blob.
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        self.interface.do_generate('u u~ > jj jj QED^2=0 QCD^2=4 [real=QCD]')
        quarks = [-1,-2,-3,-4,1,2,3,4]
        # just one born processes u u~ > g g
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 1)
        for proc in self.interface._fks_multi_proc['born_processes']:
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(final_id, [21, 21])

            # there should never be extra cnts
            self.assertEqual(len(proc.extra_cnt_amp_list), 0) #( 21, 22)
            # make sure there are qqbar splittings in the fs
            foundqq = False
            for real in proc.real_amps:
                final_real_id = real.pdgs[2:]
                # the gluon has to split in order to need for an extra cnt
                if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                    foundqq = True
                self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            self.assertTrue(foundqq)


    def test_generate_fks_ew_dijet_qed2qcd0_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED corrections to the rightmost blob.
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=0 [real=QED]')
        quarks = [-1,-2,-3,-4,1,2,3,4]
        # just one born processes u u~ > a a 
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 1)
        for proc in self.interface._fks_multi_proc['born_processes']:
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(final_id, [22, 22])

            # there should never be extra cnts
            self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
            # make sure there are qqbar splittings in the fs
            foundqq = False
            for real in proc.real_amps:
                final_real_id = real.pdgs[2:]
                # the gluon has to split in order to need for an extra cnt
                if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                    foundqq = True
                self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
            self.assertTrue(foundqq)


    def test_generate_fks_ew_dijet_qed0qcd2_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED corrections to the leftmost blob,
        or, equivalently, QCD corrections to the central blob
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        for orders in ['QED^2=0 QCD^2=4 [real=QED]', 'QED^2=2 QCD^2=2 [real=QCD]']:
            self.interface.do_generate('u u~ > jj jj %s' % orders) 
            # two born processes u u~ > g g and g a
            self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
            target_ids = [[21,21], [22,21]]
            for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
                final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
                self.assertEqual(ids, final_id)

                # there should never be extra cnts
                if final_id == [21, 21]:
                    # there should never be extra cnts
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                    # make sure there are qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        final_real_id = real.pdgs[2:]
                        # the gluon has to split
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                            foundqq = True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    self.assertTrue(foundqq)

                elif final_id == [22, 21]:
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                    # make sure there are qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        # check that real emissions with qqbar splitting has only one fks info
                        if len(real.fks_infos) > 1:
                            self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                        final_real_id = real.pdgs[2:]
                        # the gluon has to split
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                            foundqq=True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                        # make sure no qqbar splitting comes from the photon. 
                        if real.fks_infos[0]['ij'] == 3:
                            self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                            self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                    self.assertTrue(foundqq)


    def test_generate_fks_ew_dijet_qed1qcd1_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED corrections to the central blob,
        or, equivalently, QCD corrections to the rightmost blob
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        for orders in ['QED^2=2 QCD^2=2 [real=QED]', 'QED^2=4 QCD^2=0 [real=QCD]']:
            self.interface.do_generate('u u~ > jj jj %s' % orders) 
            # two born processes u u~ > g a and a a
            self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
            target_ids = [[22,21], [22,22]]
            for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
                final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
                self.assertEqual(ids, final_id)

                # there should never be extra cnts
                if final_id == [22, 21]:
                    # there should never be extra cnts
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                    # make sure there are qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        final_real_id = real.pdgs[2:]
                        # the gluon has to split
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                            foundqq = True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    self.assertTrue(foundqq)

                elif final_id == [22, 22]:
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                    # make sure there are qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        # check that real emissions with qqbar splitting has only one fks info
                        if len(real.fks_infos) > 1:
                            self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                        # make sure no qqbar splitting comes from the photon. 
                        if real.fks_infos[0]['ij'] > 2:
                            self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                            self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)


# integrate one LO contribution with both corrections
    def test_generate_fks_ew_dijet_qed0qcd2_qedqcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED and QCD corrections to the leftmost blob
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=0 QCD^2=4 [real=QED QCD]') 
        # two born processes u u~ > g g and g a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
        target_ids = [[21,21], [22,21]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            # there should never be extra cnts
            if final_id == [21, 21]:
                # there should never be extra cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(foundqq)

            elif final_id == [22, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        foundqq=True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the photon. 
                    if real.fks_infos[0]['ij'] == 3:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)


    def test_generate_fks_ew_dijet_qed1qcd1_qedqcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED and QCD corrections to the central blob
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=2 QCD^2=2 [real=QED QCD]') 
        # all three born processes u u~ > g g, g a and a a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 3)
        target_ids = [[21,21], [22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            # there should never be extra cnts
            # for g g and a a no qqbar splitting should be there
            if final_id in [[21, 21], [22, 22]]:
                # there should never be extra cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                # make sure there are no qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(not foundqq)

            elif final_id == [22, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # make sure there are qqbar splittings in the fs
                foundqqg = False
                foundqqa = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 3:
                        foundqqa=True
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        foundqqg=True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                # make sure qqbar splittings come from both the photon and the gluon
                self.assertTrue(foundqqg)
                self.assertTrue(foundqqa)


    def test_generate_fks_ew_dijet_qed2qcd0_qedqcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED and QCD corrections to the rightmost blob,
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=0 [real=QCD QED]') 
        # two born processes u u~ > g a and a a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
        target_ids = [[22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            # there should never be extra cnts
            if final_id == [22, 21]:
                # there should never be extra cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the gluon. 
                    if real.fks_infos[0]['ij'] == 4:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)

            elif final_id == [22, 22]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)


# integrate two LO contributions together with one kind of correction 
    def test_generate_fks_ew_dijet_qed1qcd2_qcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QCD corrections to the leftmost and central blobs
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=2 QCD^2=4 [real=QCD]') 
        # two born processes u u~ > g g and g a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
        target_ids = [[21,21], [22,21]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            # there should never be extra cnts
            if final_id == [21, 21]:
                # there should never be extra cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(foundqq)

            elif final_id == [22, 21]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        foundqq=True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the photon. 
                    if real.fks_infos[0]['ij'] == 3:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)


    def test_generate_fks_ew_dijet_qed1qcd2_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED corrections to the leftmost and central blobs
        or, equivalently, QCD corrections to the rightmost and central blob
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        for orders in ['QED^2=2 QCD^2=4 [real=QED]', 'QED^2=4 QCD^2=2 [real=QCD]']:
            self.interface.do_generate('u u~ > jj jj %s' % orders) 
            # all three born processes u u~ > g g , g a , aa
            self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 3)
            target_ids = [[21,21], [22,21], [22,22]]
            for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
                final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
                self.assertEqual(ids, final_id)

                # there should never be extra cnts
                if final_id == [21, 21]:
                    # there should never be extra cnts
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                    # make sure there  are no qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        final_real_id = real.pdgs[2:]
                        # there should not be any g > qq splitting here (it should be inside g a)
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                            foundqq = True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    self.assertTrue(not foundqq)

                elif final_id == [22, 21]:
                    # there should never be extra cnts
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                    # make sure there are qqbar splittings in the fs
                    foundqq = False
                    for real in proc.real_amps:
                        # check that real emissions with qqbar splitting has only one fks info
                        if len(real.fks_infos) > 1:
                            self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                        final_real_id = real.pdgs[2:]
                        # both the gluon and photon have to split
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 3:
                            foundqqa=True
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                            foundqqg=True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    self.assertTrue(foundqqa)
                    self.assertTrue(foundqqg)

                elif final_id == [22, 22]:
                    # there should never be extra cnts
                    self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                    foundqq = False
                    for real in proc.real_amps:
                        final_real_id = real.pdgs[2:]
                        # there should not be any g > qq splitting here (it should be inside g a)
                        if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                            foundqq = True
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    self.assertTrue(not foundqq)


    def test_generate_fks_ew_dijet_qed2qcd1_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QED corrections to the rightmost and central blobs,
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=2 [real=QED]') 
        # two born processes u u~ > g a and a a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 2)
        target_ids = [[22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            # there should never be extra cnts
            if final_id == [22, 21]:
                # there should never be extra cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the gluon. 
                    if real.fks_infos[0]['ij'] == 4:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)

            elif final_id == [22, 22]:
                self.assertEqual(len(proc.extra_cnt_amp_list), 0)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)


# integrate three LO contributions together with one kind of correction 
    def test_generate_fks_ew_dijet_qed2qcd2_qcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QCD corrections to the three blobs
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=4 [real=QCD]') 
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 3)
        target_ids = [[21,21], [22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            if final_id == [21, 21]:
                # gg state, should have extra_cnts with g a in the final state
                self.assertEqual(len(proc.extra_cnt_amp_list), 1) #( 21, 22)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(foundqq)

            elif final_id == [22, 21]:
                # ag state, should not have extra_cnts with a a in the final state 
                #(it should appear at order alpha^3, which is not inlcuded here
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) #( 21, 22)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        foundqq=True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the photon. 
                    if real.fks_infos[0]['ij'] == 3:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)

            elif final_id == [22, 22]:
                # aa state, should have no extra_cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) #( 22, 22)
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from final state photons. 
                    if real.fks_infos[0]['ij'] in [3, 4]:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)


    def test_generate_fks_ew_dijet_qed2qcd2_qed(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        considering the NLO QCD corrections to the tree blobs
        No extra cnts should be there"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=4 [real=QED]') 
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 3)
        target_ids = [[21,21], [22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            if final_id == [22, 22]:
                # aa state, should have no extra_cnts (it goes into g a)
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) #( 21, 22)
                # make sure there are no qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(not foundqq)

            elif final_id == [22, 21]:
                # ag state, should not have extra_cnts with a a in the final state 
                #(it should appear at order alpha^3, which is not inlcuded here
                self.assertEqual(len(proc.extra_cnt_amp_list), 1) #( 21, 22)
                # make sure there are qqbar splittings in the fs
                foundqqg = False
                foundqqa = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        foundqqg=True
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the photon. 
                    if real.fks_infos[0]['ij'] == 3:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                        if any([idd in quarks for idd in final_real_id]):
                            foundqqa=True
                self.assertTrue(foundqqg)
                self.assertTrue(foundqqa)

            elif final_id == [21, 21]:
                # gg state, should have no extra_cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) 
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from final state photons. 
                    if real.fks_infos[0]['ij'] in [3, 4]:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)


# integrate all LO and NLO contributions together
    def test_generate_fks_ew_dijet_qed2qcd2_qedqcd(self):
        """check that the generate command works as expected.
        for processes which feature g/a > qqbar splitting.
        Check if the extra countertersm are found when needed.
        Here we will use a simplified dijet case, u u~ > g/a g/a
        with all born orders and all corrections"""

        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_define('jj = g a')
        quarks = [-1,-2,-3,-4,1,2,3,4]

        self.interface.do_generate('u u~ > jj jj QED^2=4 QCD^2=4 [real=QCD QED]')
        # three born processes u u~ > g g ; a g ; a a
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 3)
        target_ids = [[21,21], [22,21], [22,22]]
        for ids, proc in zip(target_ids, self.interface._fks_multi_proc['born_processes']):
            final_id = [l['id'] for l in proc.born_amp['process']['legs'] if l['state']]
            self.assertEqual(ids, final_id)

            if final_id == [21, 21]:
                # gg state, should have extra_cnts with g a in the final state
                self.assertEqual(len(proc.extra_cnt_amp_list), 1) #( 21, 22)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] > 2:
                        foundqq = True
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                self.assertTrue(foundqq)

            elif final_id == [22, 21]:
                # ag state, should have extra_cnts with a a in the final state
                self.assertEqual(len(proc.extra_cnt_amp_list), 1) #( 21, 22)
                # make sure there are qqbar splittings in the fs
                foundqq = False
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    # the gluon has to split in order to need for an extra cnt
                    if any([idd in quarks for idd in final_real_id]) and real.fks_infos[0]['ij'] == 4:
                        self.assertNotEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                        foundqq=True
                    else:
                        self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from the photon. 
                    if real.fks_infos[0]['ij'] == 3:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)
                self.assertTrue(foundqq)

            elif final_id == [22, 22]:
                # aa state, should have no extra_cnts
                self.assertEqual(len(proc.extra_cnt_amp_list), 0) #( 22, 22)
                for real in proc.real_amps:
                    # check that real emissions with qqbar splitting has only one fks info
                    if len(real.fks_infos) > 1:
                        self.assertTrue(not any([real.pdgs[ii['i']-1] in quarks for ii in real.fks_infos]))
                    final_real_id = real.pdgs[2:]
                    self.assertEqual(real.fks_infos[0]['extra_cnt_index'], -1)
                    # make sure no qqbar splitting comes from final state photons. 
                    if real.fks_infos[0]['ij'] in [3, 4]:
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['i']-1] in quarks)
                        self.assertTrue(not real.pdgs[real.fks_infos[0]['j']-1] in quarks)

    def test_generate_fks_2to1_no_finalstate_confs(self):
        """check that in the case of a 2->1 process no final state 
        FKS configurations are generated
        """
        self.interface.do_set('include_lepton_initiated_processes False')
        self.interface.do_generate('u d~ > w+ [QED]')
        self.assertEqual(len(self.interface._fks_multi_proc['born_processes']), 1)

        # there should be 4 configurations, all due to initial-state splitting
        nconfs = 0
        for real in self.interface._fks_multi_proc['born_processes'][0].real_amps:
            for info in real.fks_infos:
                self.assertTrue(info['j'] in [1,2])
                nconfs+= 1
        self.assertEqual(nconfs, 4)


    def test_combine_equal_processes_qcd_qed(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected"""
        newinterface = mgcmd.MasterCmd()
        newinterface2 = mgcmd.MasterCmd()
        # generate the processes
        self.interface.do_generate('u u~ > t t~ QED^2=0 QCD^2=4 [real=QCD QED]')
        newinterface.do_generate('c c~ > t t~ QED^2=0 QCD^2=4 [real=QCD QED]')
        newinterface2.do_generate('d d~ > t t~ QED^2=0 QCD^2=4 [real=QCD QED]')

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

    def load_fksME(self, save_name, model_name, process_def, force=False):
        """tries to recover the fksME from a pickle file, otherwise regenerate it."""

        if not force and os.path.isfile(pjoin(root_path,'input_files',save_name)):
            return save_load_object.load_from_file(pjoin(root_path,'input_files',save_name))
        else:
            self.interface.do_import('model %s'%model_name)
            print(( "Regenerating %s ..."%process_def))
            self.interface.do_generate(process_def)
            proc = copy.copy(self.interface._fks_multi_proc)
            me = fks_helas.FKSHelasMultiProcess(proc)['matrix_elements'][0]
            save_load_object.save_to_file(pjoin(root_path,'input_files',save_name),me)
            return me

    def notest_special_dijet_equal_process_qcd_qed_virt_test(self):
        fksme1 = self.load_fksME('aa_emep_dijet.pkl','loop_qcd_qed_sm-no_widths','a a > e- e+ QED^2=4 QCD^2=4 [QCD QED]')
        fksme2 = self.load_fksME('aa_mummup_dijet.pkl','loop_qcd_qed_sm-no_widths','a a > mu- mu+ QED^2=4 QCD^2=4 [QCD QED]')
        self.assertEqual(fksme1,fksme2)

    def test_combine_equal_processes_dijet_qcd_qed_virt(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected. 
        This test also cehck that equality works for virtuals.
        In particular b-initiate processes have same trees but different loops (w/top)"""
        # generate the processes

        self.interface.do_import('model loop_qcd_qed_sm-no_widths')

        self.interface.do_generate('u u~ > g g QED^2=0 QCD^2=4 [QCD QED]')
        fksproc1 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('c c~ > g g QED^2=0 QCD^2=4 [QCD QED]')
        fksproc2 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('d d~ > g g QED^2=0 QCD^2=4 [QCD QED]')
        fksproc3 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('b b~ > g g QED^2=0 QCD^2=4 [QCD QED]')
        fksproc4 = copy.copy(self.interface._fks_multi_proc)

        fksme1 = fks_helas.FKSHelasMultiProcess(fksproc1)['matrix_elements'][0]
        fksme2 = fks_helas.FKSHelasMultiProcess(fksproc2)['matrix_elements'][0]
        fksme3 = fks_helas.FKSHelasMultiProcess(fksproc3)['matrix_elements'][0]
        fksme4 = fks_helas.FKSHelasMultiProcess(fksproc4)['matrix_elements'][0]

##        fksme1 = self.load_fksME('uux_gg_dijet.pkl','loop_qcd_qed_sm-no_widths','u u~ > g g QED^2=0 QCD^2=4 [QCD QED]')
##        fksme2 = self.load_fksME('ccx_gg_dijet.pkl','loop_qcd_qed_sm-no_widths','c c~ > g g QED^2=0 QCD^2=4 [QCD QED]')
##        fksme3 = self.load_fksME('ddx_gg_dijet.pkl','loop_qcd_qed_sm-no_widths','d d~ > g g QED^2=0 QCD^2=4 [QCD QED]')
##        fksme4 = self.load_fksME('bbx_gg_dijet.pkl','loop_qcd_qed_sm-no_widths','b b~ > g g QED^2=0 QCD^2=4 [QCD QED]')
        # check that the u and d initiated are not equal
        self.assertNotEqual(fksme2,fksme3)

        # check that the b-initiated is different from all other processes
        self.assertNotEqual(fksme1,fksme4)
        self.assertNotEqual(fksme2,fksme4)
        self.assertNotEqual(fksme3,fksme4)
        
        # check that the u and c initiated are equal
        self.assertEqual(fksme1, fksme2)

        # this is to avoid effects on other tests
        self.interface.do_import('model sm')

    def notest_combine_equal_processes_pp_hpwmbbx_virt(self):
        """Makes sure the uux and ccx channel of the process can be merged."""

        if os.path.isfile(pjoin(root_path,'input_files','uux_hpwmbbx.pkl')) and\
           os.path.isfile(pjoin(root_path,'input_files','ccx_hpwmbbx.pkl')) and\
           os.path.isfile(pjoin(root_path,'input_files','ddx_hpwmbbx.pkl')):
            uux_me = save_load_object.load_from_file(pjoin(root_path,'input_files','uux_hpwmbbx.pkl'))
            ccx_me = save_load_object.load_from_file(pjoin(root_path,'input_files','ccx_hpwmbbx.pkl'))
            ddx_me = save_load_object.load_from_file(pjoin(root_path,'input_files','ddx_hpwmbbx.pkl'))            
        else:
            self.interface.do_set('complex_mass_scheme True')
            self.interface.do_import('model 2HDMCMStIIymbMSbar')
            print ("Regenerating u u~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD] ...")
            self.interface.do_generate('u u~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD]')
            uux_proc = copy.copy(self.interface._fks_multi_proc)
            uux_me = fks_helas.FKSHelasMultiProcess(uux_proc)['matrix_elements'][0]
            save_load_object.save_to_file(pjoin(root_path,'input_files','uux_hpwmbbx.pkl'),uux_me)            
            print ("Regenerating c c~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD] ...")
            self.interface.do_generate('c c~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD]')
            ccx_proc = copy.copy(self.interface._fks_multi_proc)
            ccx_me = fks_helas.FKSHelasMultiProcess(ccx_proc)['matrix_elements'][0]
            save_load_object.save_to_file(pjoin(root_path,'input_files','ccx_hpwmbbx.pkl'),ccx_me)
            uux_me = save_load_object.load_from_file(pjoin(root_path,'input_files','uux_hpwmbbx.pkl'))
            ccx_me = save_load_object.load_from_file(pjoin(root_path,'input_files','ccx_hpwmbbx.pkl'))
            print ("Regenerating d d~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD] ...") 
            self.interface.do_generate('d d~ > h+ w- b b~ / h1 h2 h3 QED^2=2 YB=1 YT=1 QCD^2=2 [QCD]')
            ddx_proc = copy.copy(self.interface._fks_multi_proc)
            ddx_me = fks_helas.FKSHelasMultiProcess(ddx_proc)['matrix_elements'][0]
            save_load_object.save_to_file(pjoin(root_path,'input_files','ddx_hpwmbbx.pkl'),ddx_me)  

        self.assertEqual(uux_me,ccx_me)
        self.assertEqual(uux_me,ddx_me)

    def test_combine_equal_processes_dy_qed_virt(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected. 
        This test also cehck that equality works for virtuals.
        In particular b-initiate processes have same trees but different loops (w/top)"""
        # generate the processes

        self.interface.do_import('model loop_qcd_qed_sm-no_widths')

        self.interface.do_generate('u u~ > e+ e- QED^2=4 QCD^2=0 [QED]')
        fksproc1 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('c c~ > e+ e- QED^2=4 QCD^2=0 [QED]')
        fksproc2 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('d d~ > e+ e- QED^2=4 QCD^2=0 [QED]')
        fksproc3 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('b b~ > e+ e- QED^2=4 QCD^2=0 [QED]')
        fksproc4 = copy.copy(self.interface._fks_multi_proc)

        self.interface.do_generate('s s~ > e+ e- QED^2=4 QCD^2=0 [QED]')
        fksproc5 = copy.copy(self.interface._fks_multi_proc)

        fksme1 = fks_helas.FKSHelasMultiProcess(fksproc1)['matrix_elements'][0]
        fksme2 = fks_helas.FKSHelasMultiProcess(fksproc2)['matrix_elements'][0]
        fksme3 = fks_helas.FKSHelasMultiProcess(fksproc3)['matrix_elements'][0]
        fksme4 = fks_helas.FKSHelasMultiProcess(fksproc4)['matrix_elements'][0]
        fksme5 = fks_helas.FKSHelasMultiProcess(fksproc5)['matrix_elements'][0]

        # check that the u and d initiated are not equal
        self.assertNotEqual(fksme2,fksme3)

        # check that the b-initiated is different from all other processes
        self.assertNotEqual(fksme1,fksme4)
        self.assertNotEqual(fksme2,fksme4)
        self.assertNotEqual(fksme3,fksme4)
        
        # check that the u and c initiated are equal
        self.assertEqual(fksme1, fksme2)

        # check that the d and s initiated are equal
        self.assertEqual(fksme3, fksme5)

        # this is to avoid effects on other tests
        self.interface.do_import('model sm')


    def test_combine_equal_processes_qcd(self):
        """check that two processes with the same matrix-elements are equal
        and check that the add_process function works as expected"""
        newinterface = mgcmd.MasterCmd()
        newinterface2 = mgcmd.MasterCmd()
        # generate the processes
        self.interface.do_generate('u u~ > t t~ QED^2=0 QCD^2=4 [real=QCD]')
        newinterface.do_generate('d d~ > t t~ QED^2=0 QCD^2=4 [real=QCD]')

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


    def test_include_lep_split(self):
        """test that the include_lepton_initiated_processes options works as expected"""
        interface = mgcmd.MasterCmd()
        leptons=[11,13,15]
        quarks=[1,2,3,4,5]
        # default should be set to false
        # initial state leptons
        self.interface.do_generate('a a > w+ w- QED^2=4 QCD^2=0 [real=QED]')
        fksproc = self.interface._fks_multi_proc['born_processes'][0]
        reals=fksproc.reals
        for real in reals[0]+reals[1]:
            self.assertTrue(all([lep not in [abs(l['id']) for l in real['leglist']] for lep in leptons]))

        # chack that this does not affect final state photons
        self.interface.do_generate('u u~ > g a QED^2=2 QCD^2=2 [real=QCD]')
        fksproc = self.interface._fks_multi_proc['born_processes'][0]
        reals=fksproc.reals
        self.assertTrue(any([leptons[0] in [abs(l['id']) for l in real['leglist']] for real in reals[2]]))

        # now set it to true
        self.interface.do_set('include_lepton_initiated_processes True')
        # initial state leptons
        self.interface.do_generate('a a > w+ w- QED^2=4 QCD^2=0 [real=QED]')
        fksproc = self.interface._fks_multi_proc['born_processes'][0]
        reals=fksproc.reals
        self.assertTrue(any([leptons[0] in [abs(l['id']) for l in real['leglist']] for real in reals[0]]))
        self.assertTrue(any([leptons[0] in [abs(l['id']) for l in real['leglist']] for real in reals[1]]))
        # avoid border effects
        self.interface.do_set('include_lepton_initiated_processes False')

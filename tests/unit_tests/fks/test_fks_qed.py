################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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

"""Testing modules for FKS_process class"""

from __future__ import absolute_import
import sys
import os
from six.moves import zip
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_helas_objects as fks_helas_objects
import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
import madgraph.iolibs.helas_call_writers as helas_call_writers
import copy
import array


class TestFKSQED(unittest.TestCase):
    """a class to test FKS Processes, with QED perturbation"""

    def setUp(self):
        # the model
        if not hasattr(self, 'mymodel'):
            TestFKSQED.mymodel = import_ufo.import_model('sm')
            #remove third and fourth generation of quarks, and the muon
            particles_to_rm = [3, 4, 13]
            for p in particles_to_rm:
                part = TestFKSQED.mymodel.get('particle_dict')[p]
                TestFKSQED.mymodel['particles'].remove(part)
                bad_vertices = [vert for vert in TestFKSQED.mymodel['interactions'] \
                        if part in vert['particles']]
                for vert in bad_vertices:
                    TestFKSQED.mymodel['interactions'].remove(vert)

        # a partonic process
        if not hasattr(self, 'fksmultiproc_qqqq'):
            leglist = MG.MultiLegList()
            leglist.append(MG.MultiLeg({'ids': [1], 'state': False}))
            leglist.append(MG.MultiLeg({'ids': [-1], 'state': False}))
            leglist.append(MG.MultiLeg({'ids': [2], 'state': True}))
            leglist.append(MG.MultiLeg({'ids': [-2], 'state': True}))

            procdef_dict = {'legs': leglist, 
                           'orders':{},
                           'squared_orders':{'QCD':4, 'QED':2},
                           'model': self.mymodel,
                           'id': 1,
                           'NLO_mode': 'real',
                           'required_s_channels':[],
                           'forbidden_s_channels':[],
                           'forbidden_particles':[],
                           'is_decay_chain': False,
                           'perturbation_couplings':['QED'],
                           'decay_chains': MG.ProcessList(),
                           'overall_orders': {},
                           'born_sq_orders':{'QCD':4, 'QED':0}}

            TestFKSQED.fksmultiproc_qqqq = \
                    fks_base.FKSMultiProcess(MG.ProcessDefinition(procdef_dict))

        #dijet production
        if not hasattr(self, 'fksmultiproc_ppjj'):
            multi_ids = [1, -1, 2, -2, 22, 21]
            p = MG.MultiLeg({'ids': multi_ids, 'state': False})
            j = MG.MultiLeg({'ids': multi_ids, 'state': True})
            leglist = MG.MultiLegList([p, p, j, j])

            procdef_dict = {'legs': leglist, 
                           'orders':{},
                           'model': self.mymodel,
                           'id': 1,
                           'NLO_mode': 'real',
                           'required_s_channels':[],
                           'forbidden_s_channels':[],
                           'forbidden_particles':[],
                           'is_decay_chain': False,
                           'perturbation_couplings':['QED'],
                           'decay_chains': MG.ProcessList(),
                           'overall_orders': {},
                           'squared_orders': {'QCD':4, 'QED':2},
                           'born_sq_orders':{'QCD':4, 'QED':0}}
             
            TestFKSQED.fksmultiproc_ppjj = \
                    fks_base.FKSMultiProcess(MG.ProcessDefinition(procdef_dict))

        #quark/electron production, to test the init_lep_split flag 
        if not hasattr(self, 'fksmultiproc_uuee_wlepotns') or \
           not hasattr(self, 'fksmultiproc_uuee_nolepotns'):
            multi_ids = [1, -1, 11, -11]
            p = MG.MultiLeg({'ids': multi_ids, 'state': False})
            j = MG.MultiLeg({'ids': multi_ids, 'state': True})
            leglist = MG.MultiLegList([p, p, j, j])

            procdef_dict = {'legs': leglist, 
                           'orders':{},
                           'model': self.mymodel,
                           'id': 1,
                           'NLO_mode': 'real',
                           'required_s_channels':[],
                           'forbidden_s_channels':[],
                           'forbidden_particles':[],
                           'is_decay_chain': False,
                           'perturbation_couplings':['QED'],
                           'decay_chains': MG.ProcessList(),
                           'overall_orders': {},
                           'squared_orders': {'QCD':0, 'QED':6},
                           'born_sq_orders':{'QCD':0, 'QED':4}}
             
            TestFKSQED.fksmultiproc_uuee_wlepotns = \
                    fks_base.FKSMultiProcess(MG.ProcessDefinition(procdef_dict), {'init_lep_split': True})
            TestFKSQED.fksmultiproc_uuee_nolepotns = \
                    fks_base.FKSMultiProcess(MG.ProcessDefinition(procdef_dict), {'init_lep_split': False})


    def test_qqtoqqQed(self):
        """test the dijet QED corrections for a subprocess"""

        fksmultiproc = self.fksmultiproc_qqqq
        fksproc = fksmultiproc['born_processes'][0]
        # check perturbation
     #   self.assertEqual(fksproc.perturbation, 'QED')
        # check correct number of diagrams at born (4)
        # 3 s channels (a, z, g)
        # 1 t channel (w)
        self.assertEqual(len(fksproc.born_amp['diagrams']), 4)
        # check correct colors and charges
        self.assertEqual(fksproc.get_colors(), [3,-3,3,-3])
        self.assertEqual(len(fksproc.get_charges()), 4)
        for q1, q2 in zip(fksproc.get_charges(),[-1./3., 1./3., 2./3., -2./3.]):
            self.assertAlmostEqual(q1, q2)
        # check real emissions:
        # from the first and second leg, one can have the final and initial
        # state photon splitting,
        # as well as the final and initial state gluon splittng 
        # (for order consistency)
        self.assertEqual(len(fksproc.reals[0]), 4)
        self.assertEqual(len(fksproc.reals[1]), 4)
        # from the two final state legs, one can either radiate a photon or
        # a gluon
        self.assertEqual(len(fksproc.reals[2]), 2)
        self.assertEqual(len(fksproc.reals[3]), 2)


    def test_uuee_leptons(self):
        """test the correct usage of the init_lep_split flag"""

        fksmultiproc_wlep = self.fksmultiproc_uuee_wlepotns
        fksmultiproc_nolep = self.fksmultiproc_uuee_nolepotns
        #check that the correct orders are set for the real amplitudes

        #there should be 16 born processes with leptons and 12 without 
        # (remeber, without leptons means no born process with BOTH initial-
        # state leptons, so things like u e- > u e- are kept)
        self.assertEqual(len(fksmultiproc_wlep['born_processes']), 20)
        self.assertEqual(len(fksmultiproc_nolep['born_processes']), 14)

        pdg_list_wlep = [born.get_pdg_codes() for born in fksmultiproc_wlep['born_processes']]
        pdg_list_nolep = [born.get_pdg_codes() for born in fksmultiproc_nolep['born_processes']]

        # check that in fksmultiproc_nolep the only real emissions for the processes
        # with leptons in the initial states (u e) have (u a) in the initial state
        # and check that the real emission of the other processes are identical
        # as for the case with leptons
        checked_nolep = 0
        for ib, born in enumerate(fksmultiproc_nolep['born_processes']):
            if [abs(pdg) for pdg in born.get_pdg_codes()[0:2]] in [[1, 11], [11, 1]]:
                checked_nolep += 1
                for real in born.real_amps:
                    self.assertTrue(list(real.pdgs)[0:2] in [[1, 22], [-1, 22], [22, 1], [22, -1]])
            else:
                born2 = fksmultiproc_wlep['born_processes'][pdg_list_wlep.index(pdg_list_nolep[ib])]
                self.assertEqual(len(born.real_amps), len(born2.real_amps))

        # there should be 8 processes with one initial state lepton in fksmultiproc_nolep
        self.assertEqual(checked_nolep, 8)
        

    def test_pptojj_2flav(self):
        """test QED corrections for dijet production"""

        fksmultiproc = self.fksmultiproc_ppjj
        #check that the correct orders are set for the real amplitudes
        born_pdg_list = [amp.get_pdg_codes() for amp in fksmultiproc['born_processes']]

        #there should be 59 born processes
        self.assertEqual(len(fksmultiproc['born_processes']), 59)

        for born in fksmultiproc['born_processes']:
            for real in born.real_amps:
                # check that no missing borns are there
                self.assertEqual(len(real.missing_borns), 0)
                # check that reals with 3 or more external gluons,
                # with 2 or more photons or with any external lepton
                # have 0 diagrams, 
                # because they are not NLO QED corrections as requested,
                # while other reals should have diagrams
                if real.pdgs.tolist().count(21) >= 3 or \
                   real.pdgs.tolist().count(22) >= 2 or \
                   real.pdgs.tolist().count(11) + real.pdgs.tolist().count(-11) > 0:
                    self.assertEqual(len(real.amplitude['diagrams']), 0)
                else:
                    self.assertNotEqual(len(real.amplitude['diagrams']), 0)


    def test_pptojj_2flav_helas(self):
        """test the creation of the FKSHelasMultiProcess for dijet QED corrections"""
        helasmultiproc = fks_helas_objects.FKSHelasMultiProcess(self.fksmultiproc_ppjj)


    def test_qqtoqq_helas(self):
        """test the creation of the FKSHelasMultiProcess for dijet QED corrections"""
        helasmultiproc = fks_helas_objects.FKSHelasMultiProcess(self.fksmultiproc_qqqq)
        # should have only one matrix_element (subprocess)
        self.assertEqual(len(helasmultiproc['matrix_elements']), 1)
        # this subprocess should have only one born
        helasproc = helasmultiproc['matrix_elements'][0]
        # there should be 6 color_links
        self.assertEqual(len(helasproc.color_links), 6)



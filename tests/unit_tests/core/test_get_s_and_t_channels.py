################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Testing modules for the get_s_and_t_channels function"""

import sys
import os
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import madgraph.core.base_objects as MG
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import models.import_ufo as import_ufo
import tests.unit_tests as unittest


#===============================================================================
# TestGetSandTchannels
#===============================================================================
class TestGetSandTchannels(unittest.TestCase):
    """Class to test the get_s_and_t_channels function for many amplitude, both
    inverting and non inverting the t-channel orders"""

    def setUp(self):
        self.base_model = import_ufo.import_model('sm')

    def test_get_s_and_t_ub_tdg(self):
        """test that for the single-top (ub>tdg) the s-and t-channels are correctly
        returned"""

        myleglist = MG.LegList()
        myleglist.append(MG.Leg({'id':2, 'state':False}))
        myleglist.append(MG.Leg({'id':5, 'state':False}))
        myleglist.append(MG.Leg({'id':6, 'state':True}))
        myleglist.append(MG.Leg({'id':1, 'state':True}))
        myleglist.append(MG.Leg({'id':21, 'state':True}))
        proc = MG.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'orders':{'QCD':1, 'QED':2}})
        me = helas_objects.HelasMatrixElement(diagram_generation.Amplitude(proc))

        #without flipping: s-and-t channel legs
        # note that leg 2 never appears
        target = [ [[1,4,-1], [-1,3,-2], [-2,5,-3]],
                   [[5,3,-1], [1,4,-2], [-2,-1,-3]],
                   [[1,5,-1], [-1,4,-2], [-2,3,-3]],
                   [[5,4,-1], [1,-1,-2], [-2,3,-3]] ]

        #if we flip the s-and-t channel legs should be
        # note that leg 1 never appears
        target_flip = [ [[2,5,-1], [-1,3,-2], [-2,4,-3]],
                        [[5,3,-1], [2,-1,-2], [-2,4,-3]],
                        [[2,3,-1], [-1,4,-2], [-2,5,-3]],
                        [[5,4,-1], [2,3,-2], [-2,-1,-3]] ]

        for id, diag in enumerate(me.get('diagrams')):
            s_ch,t_ch = diag.get('amplitudes')[0].get_s_and_t_channels(ninitial = 2, 
                        model=self.base_model, new_pdg = 7, reverse_t_ch = False)
            self.assertEqual( [ [l['number'] for l in v['legs']] for v in s_ch] + \
                              [ [l['number'] for l in v['legs']] for v in t_ch] , 
                              target[id])

        for id, diag in enumerate(me.get('diagrams')):
            s_ch,t_ch = diag.get('amplitudes')[0].get_s_and_t_channels(ninitial = 2, 
                        model=self.base_model, new_pdg = 7, reverse_t_ch = True)
            self.assertEqual( [ [l['number'] for l in v['legs']] for v in s_ch] + \
                              [ [l['number'] for l in v['legs']] for v in t_ch] , 
                              target_flip[id])


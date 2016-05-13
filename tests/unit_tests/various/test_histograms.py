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
from __future__ import division

import madgraph.various.histograms as histograms
import os
import unittest
import copy
import tests.IOTests as IOTests
import madgraph.various.misc as misc

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_HwU_source = os.path.join(_file_path,os.pardir,'input_files','MADatNLO.HwU')
pjoin = os.path.join

class TestHistograms(unittest.TestCase):
    """Test that Histograms are correctly read, parsed, manipulated, written
    out."""
    
    def setUp(self):
        """ Load the histograms"""

        # load the base histograms
        self.histo_list = histograms.HwUList(_HwU_source)
    
    def test_histograms_operations(self):
        """ We test that basic operations are correctly handled """
        
        histo_list = copy.copy(self.histo_list)
    
        # "Testing 'Hist1 - 2.0 + 2.0 == Hist1'"
        my_hist = histo_list[0]+2.0
        my_hist = my_hist-2.0
        self.assertTrue(abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)]))<1.0e-14)

        # "Testing 'Hist1 - Hist2 + Hist2 == Hist1'"
        my_hist = histo_list[0]+histo_list[1]
        my_hist = my_hist-histo_list[1]
        self.assertTrue(abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)]))<1.0e-14)
        
        #"Testing 'Hist1 * 2.0 / 2.0 == Hist1'"
        my_hist = histo_list[0]*2.0
        my_hist = my_hist/2.0
        self.assertTrue(abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)]))<1.0e-14)

        #"Testing 'Hist1 * Hist2 / Hist2 == Hist1'"
        my_hist = histo_list[0]*histo_list[1]
        my_hist = my_hist/histo_list[1]
        self.assertTrue(abs(2.0-(my_hist.bins[0].wgts['central']/histo_list[0].bins[0].wgts['central'])-\
        (my_hist.bins[0].wgts[('scale',1.0,2.0)]/histo_list[0].bins[0].wgts[('scale',1.0,2.0)]))<1.0e-14)
    
    def test_output_reload(self):
        """ Outputs existing HwU histograms in the gnuplot format and makes sure
        that they remain identical when reloading them."""
        
        one_histo = histograms.HwUList([copy.copy(self.histo_list[0])])
        
        with misc.TMP_directory() as tmpdir:
            one_histo.output(pjoin(tmpdir,'OUT'), format='gnuplot')
            new_histo = histograms.HwUList(pjoin(tmpdir,'OUT.HwU'))
        
        one_histo = one_histo[0][0]
        one_histo.trim_auxiliary_weights()
        new_histo = new_histo[0]
        self.assertEqual(one_histo.type, new_histo.type)
        self.assertEqual(one_histo.title,new_histo.title)
        self.assertEqual(one_histo.x_axis_mode,new_histo.x_axis_mode)
        self.assertEqual(one_histo.y_axis_mode,new_histo.y_axis_mode)
        self.assertEqual(one_histo.bins.weight_labels,
                                                  new_histo.bins.weight_labels)
        self.assertEqual(len(one_histo.bins),len(new_histo.bins))
        for i, bin in enumerate(one_histo.bins):
             self.assertEqual(set(bin.wgts.keys()),
                                             set(new_histo.bins[i].wgts.keys()))
             for label, wgt in bin.wgts.items():
                 self.assertEqual(wgt,new_histo.bins[i].wgts[label])

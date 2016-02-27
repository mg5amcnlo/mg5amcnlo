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

"""Unit test library to test the histograms.py functions"""

import StringIO
import copy
import fractions
import os 
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd
import madgraph.various.histograms as histograms

pjoin = os.path.join
_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                                                 'input_files')

#===============================================================================
# IOHistogramTest
#===============================================================================
class IOHistogramTest(IOTests.IOTestManager):
    """Test class for the histograms.py module"""

    @IOTests.createIOTest()
    def testIO_DJR_histograms(self):
        """ target: MLM_djrs_output.HwU
            target: MLM_djrs_output.gnuplot
            target: CKKWL_djrs_output.HwU
            target: CKKWL_djrs_output.gnuplot
        """
        # MLM first
        histos = histograms.HwUList(pjoin(_input_file_path,'MLM_djrs.dat'), 
                                             consider_reweights='ALL', run_id=0)
        histo_output_options = {
          'format':'gnuplot', 
          'uncertainties':['scale','pdf','statistical','merging_scale','alpsfact'], 
          'ratio_correlations':True,
          'arg_string':'Automatic plotting from MG5aMC', 
          'jet_samples_to_keep':None,
          'use_band':['merging','alpsfact'],
          'auto_open':False}
        histos.output(pjoin(self.IOpath,'MLM_djrs_output'),**histo_output_options)
        histos_2 = histograms.HwUList(pjoin(self.IOpath,'MLM_djrs_output.HwU'))
        histos_2.output(pjoin(self.IOpath,'MLM_djrs_output_2'),
                                                         **histo_output_options)
        self.assertEqual(open(pjoin(self.IOpath,'MLM_djrs_output.HwU')).read(),
                            open(pjoin(self.IOpath,'MLM_djrs_output_2.HwU')).read())

        # then CKKWL
        histos = histograms.HwUList(pjoin(_input_file_path,'CKKWL_djrs.dat'), 
                                              consider_reweights='ALL', run_id=0)
        histo_output_options = {
          'format':'gnuplot', 
          'uncertainties':['scale','pdf','statistical','merging_scale'], 
          'ratio_correlations':True,
          'arg_string':'Automatic plotting from MG5aMC', 
          'jet_samples_to_keep':None,
          'use_band':['merging'],
          'auto_open':False}
        histos.output(pjoin(self.IOpath,'CKKWL_djrs_output'),**histo_output_options)
        histos_2 = histograms.HwUList(pjoin(self.IOpath,'CKKWL_djrs_output.HwU'))
        histos_2.output(pjoin(self.IOpath,'CKKWL_djrs_output_2'),
                                                         **histo_output_options)
        self.assertEqual(open(pjoin(self.IOpath,'CKKWL_djrs_output.HwU')).read(),
                      open(pjoin(self.IOpath,'CKKWL_djrs_output_2.HwU')).read()) 
        

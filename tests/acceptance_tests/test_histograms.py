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
_HwU_source = os.path.join(_file_path,'input_files','MADatNLO.HwU')
pjoin = os.path.join

class IOTest_Histogram(IOTests.IOTestManager):
    """To compare the gnuplot and HwU output of a full-fledged output."""
    
    @IOTests.createIOTest()
    def testIO_gnuplot_histo_output(self):
        """ target: HistoOut.HwU
            target: HistoOut.gnuplot
        """
        import sys
        # run in an external version of python due to potential segfault
        line='''if 1:
          import os,sys;
          sys.path=%s;
          _file_path = '%s';
          _HwU_source = os.path.join(_file_path,'input_files','MADatNLO.HwU')
          pjoin = os.path.join

          import madgraph.various.histograms as histograms;
          histo_list = histograms.HwUList(_HwU_source);
          histo_list.output(pjoin('%s','HistoOut'), format = 'gnuplot');
        ''' % (sys.path, _file_path, self.IOpath)

        import os
        os.system('echo "%s" | python' % line) 


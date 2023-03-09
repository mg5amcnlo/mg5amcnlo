################################################################################
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

import os
import time

import madgraph.iolibs.files as files
import tests.unit_tests as unittest

class TestFilesGestion(unittest.TestCase):
    """Check the validity of the pickle gestion routine"""
    
    def test_is_uptodate(self):
        '''check if is_update works'''
        
        
        filespath = ['/tmp/mg5/0.txt','/tmp/mg5/1.txt']
        os.system('mkdir /tmp/mg5 > /dev/null 2>&1')
        for i, path in enumerate(filespath):
            os.system('touch %s' % path)
            if i + 1 != len(filespath):
                time.sleep(1)
        
        self.assertTrue(files.is_uptodate(filespath[1], [filespath[0]]))
        self.assertFalse(files.is_uptodate(filespath[0], [filespath[1]]))
        self.assertTrue(files.is_uptodate(filespath[1], [filespath[0], \
                                                                 filespath[0]]))
        
        self.assertTrue(files.is_uptodate(filespath[1], [filespath[1]]))
        self.assertTrue(files.is_uptodate(filespath[1]))
        self.assertFalse(files.is_uptodate(filespath[0]))
        self.assertFalse(files.is_uptodate('/xxx/yyyy'))
        self.assertTrue(files.is_uptodate(filespath[1], ['/tmp/mg5']))
        
        self.assertRaises(AssertionError, files.is_uptodate, \
                                                      filespath[1], '/tmp/mg5')
        
        
        
        
        
         

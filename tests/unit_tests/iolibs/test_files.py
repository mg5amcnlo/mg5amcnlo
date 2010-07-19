################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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

import os
import time

import madgraph.iolibs.files as files
import tests.unit_tests as unittest

class TestFilesGestion(unittest.TestCase):
    """Check the validity of the pickle gestion routine"""
    
    def test_is_update(self):
        '''check if is_update works'''
        
        
        filespath = ['/tmp/mg5/0.txt','/tmp/mg5/1.txt']
        os.system('mkdir /tmp/mg5 &> /dev/null')
        for i, path in enumerate(filespath):
            os.system('touch %s' % path)
            if i + 1 != len(filespath):
                time.sleep(1)
        
        self.assertTrue(files.is_update(filespath[1], [filespath[0]]))
        self.assertFalse(files.is_update(filespath[0], [filespath[1]]))
        self.assertTrue(files.is_update(filespath[1], [filespath[0], \
                                                                 filespath[0]]))
        
        self.assertTrue(files.is_update(filespath[1], [filespath[1]]))
        self.assertTrue(files.is_update(filespath[1]))
        self.assertFalse(files.is_update(filespath[0]))
        self.assertFalse(files.is_update('/xxx/yyyy'))
        self.assertTrue(files.is_update(filespath[1], ['/tmp/mg5']))
        
        self.assertRaises(AssertionError, files.is_update, \
                                                      filespath[1], '/tmp/mg5')
        
        
        
        
        
         
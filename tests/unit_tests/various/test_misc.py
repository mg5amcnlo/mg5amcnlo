################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest
import madgraph.various.misc as misc
class TEST_misc(unittest.TestCase):
    
    def test_equal(self):
        
        eq = misc.equal
        
        self.assertFalse(eq(1,2,1))
        self.assertTrue(eq(1,1.2,1))
        self.assertFalse(eq(1, 1.2, 2))
        
        self.assertFalse(eq(10,20,1))
        self.assertTrue(eq(10,12,1))
        self.assertFalse(eq(10, 12, 2))        
        
        self.assertTrue(eq(100,1e2))
        self.assertFalse(eq(100,1e2 + 1e-3))
        self.assertTrue(eq(100,1e2 + 1e-4))
        self.assertTrue(eq(100,1e2 + 1e-5))

        self.assertFalse(eq(0.1,0.2,1))
        self.assertTrue(eq(0.1,0.12,1))
        self.assertFalse(eq(0.10, 0.12, 2))         

        self.assertFalse(eq(-0.1,-0.2,1))
        self.assertTrue(eq(-0.1,-0.12,1))
        self.assertFalse(eq(-0.10, -0.12, 2))         
        
        self.assertFalse(eq(-10,-20,1))
        self.assertTrue(eq(-10,-12,1))
        self.assertFalse(eq(-10, -12, 2)) 

        self.assertTrue(eq(-100,-1e2))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-100,-1e2 + 1e-4))
        self.assertTrue(eq(-100,-1e2 + 1e-5))
        
        self.assertTrue(eq(1,1.0))
        self.assertTrue(eq(1,1.0, 1))
        self.assertTrue(eq(1,1.0, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 7))
        self.assertTrue(eq(1,1.0 - 2e-8, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 8))
        self.assertFalse(eq(1,1.0 + 2e-7, 8))
        self.assertTrue(eq(9,9.0 + 2e-8, 7))
        self.assertTrue(eq(9,9.0 - 2e-8, 7))
        self.assertTrue(eq(9,9.0 + 2e-8, 8))
        self.assertFalse(eq(9,9.0 + 2e-8, 9))
        self.assertFalse(eq(1,-1.0))
        self.assertTrue(eq(0 ,0e-5))
        

        self.assertTrue(eq(80.419, 80.419002))
        for i in range(1,4):
            self.assertTrue(eq(81.966005, 81.891469,i))
        for i in range(4,7):
            self.assertFalse(eq(81.966005, 81.891469,i))
        
        # Check negative number
        self.assertTrue(eq(-1,-1))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 8))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 8))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 8))
        self.assertFalse(eq(-1,-1.0 + 1e-8, 9))
        self.assertFalse(eq(-1,-1.0 + 2e-8, 9))
        self.assertFalse(eq(-1,-1.0 - 2e-8, 9))

        self.assertFalse(eq(-1,1.0))
        self.assertTrue(eq(-0 ,-0e-5))
        
        self.assertTrue(eq(-100,-1e2))
        self.assertTrue(eq(-100,-1e2 + 1e-6))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-80.419, -80.419002))
        
        # check with 0
        self.assertTrue(eq(0 ,1e-11))  
        self.assertTrue(eq(0 ,1e-8))
        self.assertTrue(eq(0 ,1e-7))
        self.assertFalse(eq(0 ,1e-6))          
        self.assertFalse(eq(0 ,1e-1))

        self.assertTrue(eq(1e-11, 0))  
        self.assertTrue(eq(1e-8, 0))
        self.assertTrue(eq(1e-7, 0))
        self.assertFalse(eq(1e-6, 0))          
        self.assertFalse(eq(1e-1, 0))
        
        self.assertFalse(eq(0 ,1e-11, zero_limit=False))  
        self.assertFalse(eq(0 ,1e-8, zero_limit=False))
        self.assertFalse(eq(0 ,1e-7, zero_limit=False))
        self.assertFalse(eq(0 ,1e-6, zero_limit=False))          
        self.assertFalse(eq(0 ,1e-1, zero_limit=False))
        
        self.assertFalse(eq(1e-11, 0, zero_limit=False))  
        self.assertFalse(eq(1e-8, 0, zero_limit=False))
        self.assertFalse(eq(1e-7, 0, zero_limit=False))
        self.assertFalse(eq(1e-6, 0, zero_limit=False))          
        self.assertFalse(eq(1e-1, 0, zero_limit=False))         

    def test_ordered_set(self):

        set = misc.OrderedSet

        a = set(['a'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 0)

        a = set(['a', 'b'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 1)


class TEST_pythia8_merging_plugin(unittest.TestCase):
    """main164 of Pythia8 8.318 loads its merging hook as a plugin (a stub
    Pythia8 installation here)."""

    # how main164.cc of Pythia8 8.318 requests its merging hook
    main164_with_merging_plugin = (
        '  if (doMerging) {\n'
        '    pythia.readString("Init:plugins += {libpythia8amcatnloHooks.so"\n'
        '      "::aMCatNLOHooks}");\n'
        '  }\n')

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.py8 = os.path.join(self.tmpdir, 'pythia8')
        self.examples = os.path.join(self.py8, 'share', 'Pythia8', 'examples')
        os.makedirs(self.examples)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write_merging_plugin(self, pythia8_path):
        library = os.path.join(pythia8_path, 'lib', 'libpythia8amcatnloHooks.so')
        os.makedirs(os.path.dirname(library))
        open(library, 'w').close()
        return library

    def test_merging_plugin(self):
        """main164 of Pythia8 8.318 loads its merging hook as a plugin: the
        entry for Init:plugins, with the absolute path of the library"""

        self.assertEqual(misc.get_pythia8_merging_plugin(self.py8), None)
        library = self.write_merging_plugin(self.py8)
        # up to Pythia8 8.317, main164 sets the hook itself
        with open(os.path.join(self.examples, 'main164.cc'), 'w') as fsock:
            fsock.write('  if (doMerging) pythia.setUserHooksPtr(mergingHooks);\n')
        self.assertEqual(misc.get_pythia8_merging_plugin(self.py8), None)

        with open(os.path.join(self.examples, 'main164.cc'), 'w') as fsock:
            fsock.write(self.main164_with_merging_plugin)
        self.assertEqual(misc.get_pythia8_merging_plugin(self.py8),
                         '%s::aMCatNLOHooks' % os.path.abspath(library))
        os.remove(library)
        self.assertEqual(misc.get_pythia8_merging_plugin(self.py8), None)

    def test_merging_plugin_path_the_card_cannot_hold(self):
        """a ',' or a space in the path of the library: its bare name"""

        source_tree = os.path.join(self.tmpdir, 'pythia 8,318')
        os.makedirs(os.path.join(source_tree, 'examples'))
        with open(os.path.join(source_tree, 'examples', 'main164.cc'), 'w') as fsock:
            fsock.write(self.main164_with_merging_plugin)
        self.write_merging_plugin(source_tree)
        self.assertEqual(misc.get_pythia8_merging_plugin(source_tree),
                         'libpythia8amcatnloHooks.so::aMCatNLOHooks')

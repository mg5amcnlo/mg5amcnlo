################################################################################
#
# Copyright (c) 2013 The MadGraph Development team and Contributors
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

import unittest
import madgraph.madweight.MW_fct as permutation

class TestPermutation(unittest.TestCase):
    """ """

    def test_all_permutation(self):
        """check that the modification to cuts.f is what we expect."""
        get_all_permutations = permutation.get_all_permutations
        
        
        output = get_all_permutations(['s','s','s'])
        self.assertEqual(len(output), 6)
        solution =  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for sol in solution:
            self.assertTrue(sol in output)
       
        output = get_all_permutations(['b','b','j','j']) 
        self.assertEqual(len(output), 4)
        solution = [[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 4, 3], [2, 1, 4, 3]]
        for sol in solution:
            self.assertTrue(sol in output)

        output = get_all_permutations(['b','b','j','j','c','c'])        
        self.assertEqual(len(output), 8)
        solution = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5], [2, 1, 3, 4, 5, 6], [2, 1, 3, 4, 6, 5], [1, 2, 4, 3, 5, 6], [1, 2, 4, 3, 6, 5], [2, 1, 4, 3, 5, 6], [2, 1, 4, 3, 6, 5]]
        for sol in solution:
            self.assertTrue(sol in output)
            
        output = get_all_permutations(['b','b','j','b'])   
        self.assertEqual(len(output), 6)
        solution = [[1, 2, 3, 4], [1, 4, 3, 2], [2, 1, 3, 4], [2, 4, 3, 1], [4, 1, 3, 2], [4, 2, 3, 1]]
        for sol in solution:
            self.assertTrue(sol in  output)        
        
    def test_permutation_from_id(self):
        """check that the modification to cuts.f is what we expect."""
        get_perm = permutation.get_perms_from_id
        
        bjet_is_jet = True
        ids = [3, 2, 1]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 6)
        solution =  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for sol in solution:
           self.assertTrue(sol in output)
        
        bjet_is_jet = True
        ids = [5, 2, 1]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 6)
        solution =  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for sol in solution:
           self.assertTrue(sol in output)        

        bjet_is_jet = False
        ids = [5, 2, 1]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 2)
        solution =  [[1, 2, 3], [1, 3, 2]]
        for sol in solution:
           self.assertTrue(sol in output)  
            
        # tt~ fully lept
        bjet_is_jet = False
        ids = [5, 11,-12, -5, -11, 12]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 2)
        solution =  [[1, 2, 3, 4, 5, 6], [4, 2, 3, 1, 5, 6]]
        for sol in solution:
           self.assertTrue(sol in output)  
            
        # tt~ semi lept
        bjet_is_jet = False
        ids = [5, 11,-12, -5, 3, -2]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 4)
        solution =  [[1, 2, 3, 4, 5, 6], [4, 2, 3, 1, 5, 6],[1, 2, 3, 4, 6, 5], [4, 2, 3, 1, 6, 5]]
        for sol in solution:
           self.assertTrue(sol in output)      
            
        # tt~ semi lept
        bjet_is_jet = True
        ids = [5, 11,-12, -5, 3, -2]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 24)
        solution =  [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5], [1, 2, 3, 5, 4, 6], [1, 2, 3, 5, 6, 4], [1, 2, 3, 6, 4, 5], [1, 2, 3, 6, 5, 4], [4, 2, 3, 1, 5, 6], [4, 2, 3, 1, 6, 5], [4, 2, 3, 5, 1, 6], [4, 2, 3, 5, 6, 1], [4, 2, 3, 6, 1, 5], [4, 2, 3, 6, 5, 1], [5, 2, 3, 1, 4, 6], [5, 2, 3, 1, 6, 4], [5, 2, 3, 4, 1, 6], [5, 2, 3, 4, 6, 1], [5, 2, 3, 6, 1, 4], [5, 2, 3, 6, 4, 1], [6, 2, 3, 1, 4, 5], [6, 2, 3, 1, 5, 4], [6, 2, 3, 4, 1, 5], [6, 2, 3, 4, 5, 1], [6, 2, 3, 5, 1, 4], [6, 2, 3, 5, 4, 1]]
        for sol in solution:
           self.assertTrue(sol in output)                                   
        
        # tt~ ful
        bjet_is_jet = True
        ids = [5, 1,-2, -5, 3, -2]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 720)
        for i,sol in enumerate(output):
            self.assertFalse(sol in output[:i])         
            self.assertEqual(set(sol), set([1,2,3,4,5,6]))
        
        # generate p p > t1 t1~ , ( t1 > t n1 , ( t > b w+ , w+ > j j ) ), ( t1~ > t~ n1 , ( t~ > b~ w- , w- > mu- vm~ ) )
        # tt~ ful
        bjet_is_jet = True
        ids = [1000022, 5, 11, -12, 1000022, -5, 3, -2]
        output = get_perm(ids, bjet_is_jet)
        self.assertEqual(len(output), 24)
        solution =  [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 8, 7], [1, 2, 3, 4, 5, 7, 6, 8], [1, 2, 3, 4, 5, 7, 8, 6], [1, 2, 3, 4, 5, 8, 6, 7], [1, 2, 3, 4, 5, 8, 7, 6], [1, 6, 3, 4, 5, 2, 7, 8], [1, 6, 3, 4, 5, 2, 8, 7], [1, 6, 3, 4, 5, 7, 2, 8], [1, 6, 3, 4, 5, 7, 8, 2], [1, 6, 3, 4, 5, 8, 2, 7], [1, 6, 3, 4, 5, 8, 7, 2], [1, 7, 3, 4, 5, 2, 6, 8], [1, 7, 3, 4, 5, 2, 8, 6], [1, 7, 3, 4, 5, 6, 2, 8], [1, 7, 3, 4, 5, 6, 8, 2], [1, 7, 3, 4, 5, 8, 2, 6], [1, 7, 3, 4, 5, 8, 6, 2], [1, 8, 3, 4, 5, 2, 6, 7], [1, 8, 3, 4, 5, 2, 7, 6], [1, 8, 3, 4, 5, 6, 2, 7], [1, 8, 3, 4, 5, 6, 7, 2], [1, 8, 3, 4, 5, 7, 2, 6], [1, 8, 3, 4, 5, 7, 6, 2]]
        for sol in solution:
           self.assertTrue(sol in output)
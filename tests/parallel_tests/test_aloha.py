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
"""Unit test Library for testing the Creation of Helas Amplitude created from 
the output of the Feynman Rules."""
from __future__ import division

import math
import os
import time
import tempfile as tempfile
from functools import wraps

import aloha
import aloha.aloha_object as aloha_obj
import aloha.aloha_lib as aloha_lib
from aloha.aloha_lib import *
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
import models.sm.object_library as object_library
import tests.unit_tests as unittest
import madgraph.various.misc as misc

def set_global(loop=False, unitary=True, mp=False, cms=False):

    def deco_set(f):
        @wraps(f)
        def deco_f_set(*args, **opt):
            old_loop = aloha.loop_mode
            old_gauge = aloha.unitary_gauge
            old_mp = aloha.mp_precision
            old_cms = aloha.complex_mass
            aloha.loop_mode = loop
            aloha.unitary_gauge = unitary
            aloha.mp_precision = mp
            aloha.complex_mass = cms
            aloha_lib.KERNEL.clean()
            try:
                out =  f(*args, **opt)
            except:
                aloha.loop_mode = old_loop
                aloha.unitary_gauge = old_gauge
                aloha.mp_precision = old_mp
                aloha.complex_mass = old_cms
                raise
            aloha.loop_mode = old_loop
            aloha.unitary_gauge = old_gauge
            aloha.mp_precision = old_mp
            aloha.complex_mass = old_cms
            aloha_lib.KERNEL.clean()
            return out
        return deco_f_set
    return deco_set



class TestVariable(unittest.TestCase):

    def setUp(self):
        self.var1 = 2 * aloha_lib.Variable('var1')
        self.var2 = 3 * aloha_lib.Variable('var2')
        self.var3 = 11 * aloha_lib.Variable('var3')
        
    
    def test_short_power(self):
        """check that the power is correctly update"""
        
        a = aloha_lib.Variable('P3_0')
        b = a ** 2 * a ** 2

        b = b.simplify()
        self.assertTrue(isinstance(b, aloha_lib.MultVariable))
        self.assertEqual(4, len(b))
        for i in b:
            self.assertEqual(a[0], i)
    
    def testsumvarvar (self):
        """ test the sum of two Variable Object"""
        
        #Sum of Two Variable
        sum = self.var1 + self.var2
        
        #check sanity
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var1 in sum)        
        self.assertTrue(self.var2 in sum)
        self.assertEquals(len(sum),2)
        
        #test prefactor, constant term treatment
        self.assertEquals(sum.prefactor,1)
        self.assertTrue(self.var1 in sum)
        #for term in sum:
        #    if term == self.var1:
        #        #self.assertEqual(term.prefactor, 2)
        #        self.assertFalse(term is self.var1)
        #    elif term == self.var2:
        #        #self.assertEqual(term.prefactor, 3)
        #        self.assertFalse(term is self.var2)
                
        #self.assertEquals(self.var1.prefactor, 2) #prefactor is preserve
        #self.assertEquals(self.var2.prefactor, 3)   
         
    def testrsumvarvar (self):
        """ test the sum of two Variable Object (inverse order)"""
        
        #Sum of Two Variable
        sum = self.var2 + self.var1        
        #check sanity
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var1 in sum)        
        self.assertTrue(self.var2 in sum)
        self.assertEquals(len(sum),2)
        
        #test prefactor, constant term treatment
        self.assertEquals(sum.prefactor,1)
        #self.assertTrue(self.var1 in sum)
        #for term in sum:
        #    if term == self.var1:
        #        self.assertEqual(term.prefactor, 2)
        #        self.assertFalse(term is self.var1)
        #    elif term == self.var2:
        #        self.assertEqual(term.prefactor, 3)
        #        self.assertFalse(term is self.var2)
                                  
        #self.assertEquals(self.var1.prefactor,2) #prefactor is preserve
        #self.assertEquals(self.var2.prefactor,3)   
 
    def testsumvarint(self):
        """ test the sum of one Variable with an integer"""

        sum = self.var1 + 4
        self.assertEqual(sum.__class__, aloha_lib.AddVariable)
        return
    
    def testsumvaradd(self):
        """ test the sum of one Variable with an AddVariable"""        

        add = aloha_lib.AddVariable()
        add.append(self.var1)
        add.append(self.var2)

        sum = self.var3 + add
        
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var3 in sum)
        self.assertEquals(len(sum), 3)
        #for data in sum:
        #    if data == self.var3:
        #        self.assertFalse(data is self.var3)
        #    else:
        #        self.assertTrue(data is self.var1 or data is self.var2)
                    
                
        #test prefactor- constant_term
        #self.assertEquals(sum.prefactor, 1)
        #self.assertEquals(self.var1.prefactor,2)
        
    def testsumvarmult(self):
        """ test the sum of one Variable with an MultVariable"""        
        
        mult = self.var1 * self.var2 
        sum = self.var3 + mult
        
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var3 in sum)

                
        #test prefactor- constant_term
        #self.assertEquals(sum.prefactor, 1)
        #self.assertEquals(self.var3.prefactor, 11)
         
    def testmultvarvar(self):
        """product of Two Variable"""
        
        prod = self.var1 * self.var2
        prod.simplify()
        #check sanity
        self.assertEquals(prod.__class__,aloha_lib.MultVariable)
        self.assertEquals(len(prod),2)
        
        
        self.assertEquals(prod.prefactor,6)


    def testmultvarAdd(self):
        """product of Variable with an AddVariable"""
        
        add = self.var1 + self.var2
        prod = self.var3 * add
        #sanity check
        self.assertEquals(prod.__class__, aloha_lib.AddVariable)
        self.assertEquals(len(prod), 2)
        prod.simplify()
        #check prefactor of each term
        for term in prod:
            if prod.prefactor * term.prefactor == 22:
                self.assertEqual(set(term), set([self.var1[0], self.var3[0]]))
            elif prod.prefactor * term.prefactor == 33:
                self.assertEqual(set(term),set([self.var2[0], self.var3[0]]))
            else:
                raise Exception('not valid term')
                
    
    def testmultvarMult(self):
        """product of Variable with an MultVariable"""
        
        var1 = 2
        var2 = 3*aloha_lib.Variable('y')
        mult = var1 * var2
        prod = self.var1 * mult
        prod.simplify()
        #Sanity
        self.assertEquals(prod.__class__, aloha_lib.MultVariable)
        self.assertEquals(len(prod), 2)
        
        #check prefactor
        self.assertEquals(prod.prefactor, 12)
        
               
    def testmultvarint(self):
        """product of Var with an integer"""
        
        prod1 = self.var1 * 2
        prod2 = 2 * self.var2
        prod1.simplify()
        prod2.simplify()
        
        #check prefactor - constant term
        self.assertEquals(prod1.prefactor, 4)
        self.assertEquals(prod2.prefactor, 6)

class TestAddVariable(unittest.TestCase):

    def setUp(self):
        """Initialize basic object"""
        self.var1 = 2 * aloha_lib.Variable('var1')
        self.var2 = 3 * aloha_lib.Variable('var2')
        self.add1 = aloha_lib.AddVariable()
        self.add1 += self.var1
        self.add1 += self.var2

        self.var3 = 11 * aloha_lib.Variable( 'var3')
        self.var4 = 4 * aloha_lib.Variable( 'var4')
        self.add2 = aloha_lib.AddVariable()
        self.add2 += self.var3
        self.add2 += self.var4        
    
    def testsumaddint(self):
        """Test the sum of an Add variable with an integer"""
        
        add2 = self.add1 + 5
        self.assertEqual(type(add2), aloha_lib.AddVariable)
        self.assertEqual(len(add2), 3)
        for term in add2:
            if term == self.var1:
                self.assertTrue(term.prefactor, 2)
            elif term == self.var2:
                self.assertTrue(term.prefactor, 3)
            else:
                self.assertEqual(term, 5)
            
        return
                
    def testsumaddmult(self):
        """Test the sum of an AddVariable with a MultVariable."""
        
        var1 = aloha_lib.Variable('v2')
        var2 = aloha_lib.Variable('v3')
        mult = var1 * var2 + 2
        sum = self.add1 + mult
        #Sanity Check
        self.assertEquals(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 4)
        self.assertTrue(2 in sum)
        self.assertTrue('v2 * v3' in str(sum))
        
        #check new term 
        for term in sum:
            if term.__class__ == aloha_lib.AddVariable:
                self.assertTrue(term.prefactor, 6)
                self.assertTrue(term.constant_term, 0)
                
    def testsumaddvar(self):
        """Test the sum of an AddVariable with a Variable."""
        
        var3 = 11 * aloha_lib.Variable( 'var3')
        sum = self.add1 + var3
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var1 in sum)
        self.assertTrue(self.var2 in sum)
        self.assertTrue(self.var3 in sum)        
        self.assertEquals(len(sum), 3)
        for data in sum:
            if data == self.var1:
                self.assertEquals(data.prefactor,2)
            elif data == self.var2:
                self.assertEquals(data.prefactor,3)
            elif data == self.var3:
                self.assertEquals(data.prefactor,11)
                
        #test prefactor- constant_term
        self.assertEquals(sum.prefactor, 1)
    
    def testsumaddadd(self):
        """Test the sum of two add object"""
        
        sum = self.add1 + self.add2
        
        self.assertEquals(sum.__class__, aloha_lib.AddVariable)
        self.assertEquals(len(sum), 4)
        
        self.assertTrue(self.var1 in sum)
        self.assertTrue(self.var2 in sum)
        self.assertTrue(self.var3 in sum)
        self.assertTrue(self.var4 in sum)
        
        for data in sum:
            if data == self.var1:
                self.assertEquals(data.prefactor, 2)
            elif data == self.var2:
                self.assertEquals(data.prefactor, 3)
            elif data == self.var3:
                self.assertEquals(data.prefactor, 11)
            elif data == self.var4:
                self.assertEquals(data.prefactor, 4)
        #test prefactor- constant_term
        self.assertEquals(sum.prefactor, 1)
        
    def testmultaddint(self):
        """test the multiplication of an AddVariable by a Integer"""
        
        prod1 = 3 * self.add1
        prod2 = self.add2 * 2
        
        self.assertEquals(prod1.__class__, aloha_lib.AddVariable)
        self.assertEquals(prod2.__class__, aloha_lib.AddVariable)
        self.assertFalse(prod1 is self.add1)
        self.assertFalse(prod2 is self.add2)
        self.assertEquals(len(prod1), 2)
        self.assertEquals(len(prod2), 2)
        
        self.assertEquals(prod1.prefactor, 3)
        self.assertEquals(prod2.prefactor, 2)
                
        for data in prod1:
            if 'var1' in str(data):
                self.assertEquals(prod1.prefactor * data.prefactor, 6)
            elif 'var2' in str(data):
                self.assertEquals(prod1.prefactor * data.prefactor, 9)
        for data in prod2:
            if 'var3' in data:
                self.assertEquals(prod2.prefactor * data.prefactor, 22)
            elif 'var4' in data:
                self.assertEquals(prod2.prefactor * data.prefactor, 8)

    
    def testmultadd_legacy(self):
        """ int * AddVariable doens't change the content of AddVariable """
        
        var1 = aloha_obj.P(1,2)
        var2 = aloha_obj.P(2,2)
        prod = var1 * var2
        #assert(prod.__class__, aloha_lib.MultLorentz)
        var3 = aloha_obj.Metric(1,2)
        
        sum = (var3 + var1 * var2)    
        new_sum = 2 * sum
        
        self.assertEqual(new_sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(new_sum), 2)
        self.assertEqual(new_sum.prefactor, 2)
        for term in new_sum:
            self.assertEqual(term.prefactor, 1)

            if str(term).startswith('(_ETA_'):
                self.assertFalse('var3' in term)
                self.assertTrue(var3 is term)
            else:
                self.assertEqual(term.__class__, aloha_lib.MultLorentz)
                self.assertEqual(prod, term)
                self.assertFalse(prod is term) 
    
    def testmultaddvar(self):
        """Test the multiplication of an Addvariable with a Variable"""
        
        var3 = 11 * aloha_lib.Variable('var3')
        prod = self.add1 * var3
        #sanity check
        self.assertEquals(prod.__class__, aloha_lib.AddVariable)
        self.assertEquals(len(prod), 2)
        
        #check prefactor of each term
        for term in prod:
            if 'var1' not in str(term):
                self.assertEquals(prod.prefactor * term.prefactor, 33)
            elif 'var2' not in str(term):
                self.assertEquals(prod.prefactor * term.prefactor, 22)
            else:
                raise Exception('not valid term')
                
    
    def testmultaddvar_legacy(self):
        """Test that the legacy is preserve for Add/var multiplication"""
        
        p1 = aloha_obj.P(1,1)
        p2 = aloha_obj.P(1,2)
        p3 = aloha_obj.P(3,3)
        
        #make (p1+p2)*p3
        add= p1+p2
        result= add *p3 
        
        self.assertEqual(result.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(result), 2)
        for term in result:
            self.assertTrue('_P^3_3' in str(term))
            self.assertEqual(term.__class__,aloha_obj.P.mult_class)
        
        
        
        
    def testmultaddmult(self):
        """Test the multiplication of an AddVariable with a MultVariable."""
        
        var3 = 2 * aloha_lib.Variable( 'var3')
        var4 = 1 * aloha_lib.Variable( 'var4')
        prod = self.add1 * (var3 *var4)
        
        self.assertEqual(prod.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(prod), 2)
        
        for data in prod:
            if 'var1' in str(data):
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(prod.prefactor * data.prefactor, 4)
            else:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(prod.prefactor * data.prefactor, 6)
        self.assertEqual(prod.prefactor, 2)
        
                
    def testmultaddadd(self):
        """Test the multiplication between two AddVariable."""
        
        prod = self.add1 * self.add2
        self.assertEqual(prod.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(prod), 4)
        
        for data in prod:
            sdata = str(data)
            if 'var1' in sdata and 'var3' in sdata:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 22)
            elif 'var1' in sdata:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 8)
            elif 'var2' in sdata and 'var3' in sdata:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 33)
            else:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 12)
        
    def test_short_replace(self):
        """test that the replace command works"""
        
        id = self.var1.get_id()
        new = self.add1.replace(id, self.add2)
        
        self.assertEqual(len(new),3)
        self.assertEqual(set([a.prefactor for a in new]),set([3,22,8]))
        for i in new:
            self.assertNotEqual(i,id)
        
        self.setUp()
        id = self.var1.get_id()
        new = self.add1.replace(id, self.var1 * self.var2)
        self.assertEqual(len(new),2)
        self.assertEqual(set([a.prefactor for a in new]),set([12,3]))

        self.setUp()
        add2 = 2 * self.add2
        id  = self.var1.get_id()
        new = self.var1.replace(id, add2)
        self.assertEqual(len(new),2)
        self.assertEqual(set([a.prefactor for a in new]),set([11,4]))
        for i in new:
            self.assertNotEqual(i,id)
        
        self.setUp()
        add1 = 5 * self.add1
        add2 = 2 * self.add2
        id  = self.var1.get_id()
        new = add1.replace(id, add2)
        self.assertEqual(len(new),3)
        self.assertEqual(set([a.prefactor*new.prefactor for a in new]),set([15,220,80]))
        for i in new:
            self.assertNotEqual(i,id)
        
           
    def test_short_factorization(self):
        """test the factorization"""
        
        p1 = aloha_lib.Variable('p1')
        p2 = aloha_lib.Variable('p2')        
        p3 = aloha_lib.Variable('p3')
        p4 = aloha_lib.Variable('p4')
        p5 = aloha_lib.Variable('p5')
        
        
        sum = p1 * p2 + p1 * p3
        sum = sum.factorize()
        self.assertEqual(sum.__class__,aloha_lib.MultContainer)
        self.assertEqual(len(sum),2)
        for fact in sum:
            if isinstance(fact, str):
                self.assertEqual(str(fact), 'p1')
            else:
                self.assertEqual(str(fact), '( (p2) + (p3) )') 
        
        
        sum = p1 * p2 + p1 * p3 + 2 * p1 + 2 *p1 * p2 * p4
        sum = sum.factorize()
        #Should return p1*(p2(2*p4 + 1) + p3 + 2)
        self.assertEqual(sum.__class__,aloha_lib.MultContainer)
        self.assertEqual(len(sum),2)
        self.assertEqual(str(sum), '(p1 * ( (p2 * 2 * ( 0.5 + (p4) )) + ( (p3) + 2 ) ))')

    def test_short_factorization2(self):
        """test the factorization with power and constant"""
        
        p1 = aloha_lib.Variable('p1')
        p2 = aloha_lib.Variable('p2')        
        p3 = aloha_lib.Variable('p3')
                
        sum = ( -2 * p1 **2 + -2 * p2 + 2 * ( p3 * p2 ) )
        sum = sum.factorize()
        #Should return p2*(2*p3-2)-2*p1**2
        self.assertEqual(str(sum), '( (p2 * 2 * ( -1.0 + (p3) )) + (-2 * p1 * p1) )')

    def test_short_factorization3(self):
        """test factorization with prefactor"""
        
        p1 = aloha_lib.Variable('p1')
        p2 = aloha_lib.Variable('p2')
        
        sum =2 * p2**2 + 2* p1 * p2
        sum = sum.factorize()
        #should be p2 (2 * p1 + 2 * p2)
        self.assertEqual(str(sum), '(2 * p2 * ( (p2) + (p1) ))')
        
    
    def test_short_factorization4(self):
        """test the factorization with constant factor"""
        
        P1_0 = aloha_lib.Variable('p1')
        P1_1 = aloha_lib.Variable('p2')
        P1_2 = aloha_lib.Variable('p3')
        P1_3 = aloha_lib.Variable('p4')        
        OM1  = aloha_lib.Variable('om1') 
        
        expr1 = ( -1j * ( P1_3 * P1_1 * OM1 ) + 1j * ( P1_0**2 * P1_3 * P1_1 * OM1**2 ) + -1j * ( P1_1**3 * P1_3 * OM1**2 ) + -1j * ( P1_2**2 * P1_3 * P1_1 * OM1**2 ) + -1j * ( P1_3**3 * P1_1 * OM1**2 ) )

        p1, p2, p3, p4, om1 = 1,2,3,4,5
        value = eval(str(expr1))
        
        expr1 = expr1.factorize()
        self.assertEqual(eval(str(expr1)), value)


    def test_short_factorization5(self):
        """check that P [gamma + P/M] == (/p+M) [Onshell]"""

        P1_0 = aloha_lib.Variable('p1')
        P1_1 = aloha_lib.Variable('p2')
        P1_2 = aloha_lib.Variable('p3')
        P1_3 = aloha_lib.Variable('p4')        
        M1  = aloha_lib.Variable('m1') 
    
        p1, p2, p3, p4, m1 = 1,2,3,4,5
    
        data = (P1_0**2 * M1 - P1_1**2 * M1 + M1)
        value = eval(str(data))
        data2 = data.factorize()
        self.assertEqual(eval(str(data2)), value)
        



    




    
class TestMultVariable(unittest.TestCase):

    def setUp(self):
        self.var1 = 2*aloha_lib.Variable( 'var1')
        self.var2 = 3*aloha_lib.Variable( 'var2')
        self.var3 = 4*aloha_lib.Variable( 'var3')
        self.var4 = 5*aloha_lib.Variable( 'var4')
        
        self.mult1 = self.var1 * self.var2
        self.mult2 = self.var3 * self.var4
    
    def testequality(self):
        """test the definition of Equality"""

        #test with mult obj
        
        self.assertNotEqual(self.mult1, self.mult2)
                
        #test with other type of obj
        self.assertNotEqual(self.mult1, 32)
        self.assertNotEqual(self.mult1, self.var1)
        prov = self.var1 + self.var2
        self.assertNotEqual(self.mult1, prov )
        
                
    def testsummultmul(self):
        """Test the sum of two MultVariable"""
        
        sum = self.mult1 + self.mult2 
        self.assertEqual(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum),2)
        self.assertEqual(sum.prefactor, 1)
        
        for term in sum:
            if 'var1' in str(term):
                self.assertEqual(term.prefactor, 6)
                #self.assertFalse(term is self.mult1)
            else:
                self.assertEqual(term.prefactor, 20)
                #self.assertFalse(term is self.mult2)
                
        sum =  self.mult1 - self.mult1
        sum = sum.simplify()
        self.assertEqual(sum.__class__, int)
        self.assertEqual(sum,0)

        
    def testdealingwithpower1(self):
        """Check that the power is correctly set in a product"""
        
        p1 = aloha_lib.Variable('p1')
        p2 = aloha_lib.Variable('p2')
        
        prod = p1 * p1
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)       
        prod = prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(len(prod), 2)
        self.assertEqual(len(p1), 1)
        
        prod *= p1
        prod = prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(len(prod), 3)
        self.assertEqual(len(p1), 1)
        
        prod *= p2
        prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(prod.count(p1.get_id()), 3)
        self.assertEqual(prod.count(p2.get_id()), 1)                                
        
        prod *= p1
        prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(prod.count(p1.get_id()), 4)
        self.assertEqual(prod.count(p2.get_id()), 1)  
                                 
                                
    def r_testdealingwithpower2(self):
        """Check that the power is correctly set in a product"""       
        
        p1 = aloha_lib.Variable('p1')
        p2 = aloha_lib.Variable('p2')
        p3 = aloha_lib.Variable('p3')
        p4 = aloha_lib.Variable('p2')
        p5 = aloha_lib.Variable('p5')
        sum1 = p1 + p2
        sum2 = p4 + p3

        prod = p3 * sum2 * sum1
        self.assertEqual(prod.__class__, aloha_lib.AddVariable)
        for term in sum1 + sum2:
            self.assertEqual(term.power, 1)
        
        obj1 = 0
        for term in prod:
            if p2 == term:
                self.assertEqual(term.power, 2)
            elif p1 in term and p2 in term and p3 in term:
                self.assertEqual(term[0].power, 1)
                self.assertEqual(term[1].power, 1)        
                self.assertEqual(term[2].power, 1)
                if not obj1:
                    obj1= term[1]
                else:
                    self.assertFalse(obj1 is term[1])
            elif p2 in term and p3 in term:
                self.assertEqual(term[0].power+term[1].power, 3)
                if not obj1:
                    obj1= term[1]
                else:
                    self.assertFalse(obj1 is term[1])                
        
    def testdealingwithpower3(self):
        """Check that the power is correctly set in a product in the full chain"""
        
        F1_1, F1_2, F1_3, F1_4 = 1,2,3,4
        
        P1_0, P1_1, P1_2, P1_3 = 12, 0, 0, 12
        P2_0, P2_1, P2_2, P2_3 = 12, 0, 12, 0
        P3_0, P3_1, P3_2, P3_3 = 20, 0, 12, 12
        M1, M2, M3 = 0, 0, 100 
        
        F2_1, F2_2, F2_3, F2_4 = 5,5,6,7
        T3_1, T3_2, T3_3, T3_4 = 8,9,10,11
        T3_5, T3_6, T3_7, T3_8 = 8,9,10,11
        T3_9, T3_10, T3_11, T3_12 = 8,9,10,11
        T3_13, T3_14, T3_15, T3_16 = 8,9,10,11
        
        
        
        p1 = aloha_obj.P('mu',2)
        gamma1 = aloha_obj.Gamma('mu','a','b')
        metric = aloha_obj.Spin2('nu','rho',3)
        p2 = aloha_obj.P('rho',2)
        gamma2 = aloha_obj.Gamma('nu','b','c')
        F1 = aloha_obj.Spinor('c',1) 
        
         
        lor1 = p1 * gamma1 * gamma2 * F1
        lor2 = metric * p2
        lor1.simplify()
        new_lor = lor1.expand()
        lor2.simplify()
        new_lor2 = lor2.expand()
        
        expr = new_lor * new_lor2
        self.assertEqual((-864+288j), eval(str(expr.get_rep([0]))))
        self.assertEqual((288+864j), eval(str(expr.get_rep([1]))))
        self.assertEqual((2016+288j), eval(str(expr.get_rep([2]))))
        self.assertEqual((-288+2016j), eval(str(expr.get_rep([3]))))
        
    
    def test_short_obj_are_not_modified(self):
        """Check that a sum-product-... doesn't change part of the objects"""
        
        sum = self.mult1 + self.mult2
        #for term in sum:
        #    self.assertFalse(term is self.mult1)
        #    self.assertFalse(term is self.mult2)
            
        
        sum2 = sum - (self.mult1 + self.mult2)
        #for term in sum:
        #    for term2 in sum2:
        #        self.assertFalse(term is term2)
        
        sum2 = sum2.simplify()
        
        #check that sum2 is zero
        self.assertEqual(sum2, 0)
        
        #check that the sum is not modify in this game      
        self.assertEqual(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 2)
        self.assertEqual(sum.prefactor, 1)
        
        for data in sum:
            self.assertEqual(len(data), 2)
            if 'var1' in str(data):
                self.assertEqual(data.prefactor, 6)
                self.assertTrue('var2' in str(data))
            else:
                self.assertEqual(data.prefactor, 20)
                self.assertTrue('var3' in str(data))
                self.assertTrue('var4' in str(data))
            
    def testsummultint(self):
        """Test the sum of a MultVariable object with a number"""
        
        add = self.mult1 + 2
        self.assertEqual(add.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(add), 2)
        for term in add:
            if term.__class__ == aloha_lib.MultVariable:
                self.assertEqual(term.prefactor, 6)
                self.assertEqual(len(term), 2)
                #self.assertFalse(term is self.mult1)
            else:
                self.assertEqual(term.__class__, int)
                self.assertEqual(term, 2)
        
        return
        
    def testsummultadd(self):
        """Test the sum of an MultVariable with a AddVariable."""
        
        var1 = 2 * aloha_lib.Variable('xxx')
        var2 = 3 * aloha_lib.Variable('yyy')
        add = var1 + var2
                
        sum = self.mult2 + add
        #Sanity Check
        self.assertEquals(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 3)
        self.assertTrue(var1 in sum)
        self.assertTrue(var2 in sum)
        
        #check new term 
        for term in sum:
            if 'xxx' in str(term):
                self.assertEqual(term.prefactor, 2)
            elif 'yyy' in str(term):
                self.assertEqual(term.prefactor, 3)
            elif term.__class__ == aloha_lib.MultVariable:
                self.assertEqual(term.prefactor, 20)
                self.assertTrue('var3' in str(term))
                self.assertTrue('var4' in str(term))
                
            self.assertEqual(sum.prefactor, 1)
            
    def testsummulvar(self):
        """Test the sum of a MultVariable with a Variable"""
        
        
        var = 3 * aloha_lib.Variable('xxx')
        sum = self.mult2 + var
        sum.simplify()
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(var in sum)
        self.assertEquals(len(sum), 2)
        for term in sum:
            self.assertTrue(term.prefactor in [3,20])

                
        #test prefactor- constant_term
        self.assertEquals(sum.prefactor, 1)
        self.assertEquals(var.prefactor, 3)
        self.assertEquals(self.mult2.prefactor, 20)
        
    def testmultmultint(self):
        """Test the multiplication of an MultVariable with an integer"""
        
        prod1 = self.mult1 * 2
        
        self.assertEqual(prod1.__class__, aloha_lib.MultVariable)
        self.assertEqual(len(prod1), 2)
        self.assertFalse(prod1 is self.mult1)
        self.assertEqual(prod1.prefactor, 12)
        for fact in prod1:
            if fact == self.var1:
                self.assertEqual(fact.prefactor, 1)
            if fact == self.var2:
                self.assertEqual(fact.prefactor, 1)
                            
        prod2 = 2 * self.mult1

        self.assertEqual(prod2.__class__, aloha_lib.MultVariable)
        self.assertEqual(len(prod2), 2)
        self.assertEqual(prod2.prefactor, 12)
        for fact in prod1:
            if fact == self.var1:
                self.assertEqual(fact.prefactor, 1)
            if fact == self.var2:
                self.assertEqual(fact.prefactor, 1)
        
                
    def testmultmultmult(self):
        """test the multiplication of two MultVariable"""
        
        prod1 = self.mult1 * self.mult2
        self.assertEqual(prod1.__class__, aloha_lib.MultVariable)
        self.assertEqual(len(prod1), 4)
        self.assertTrue('var1' in str(prod1))
        self.assertTrue('var2' in str(prod1))
        self.assertTrue('var3' in str(prod1))
        self.assertTrue('var4' in str(prod1))        
        self.assertEqual(prod1.prefactor, 120)

        
        
                

        

        
        
class testLorentzObject(unittest.TestCase):
    """ Class to test the Operation linked to a Lorentz Object"""
    
    def setUp(self):
        aloha_lib.KERNEL.clean()
        self.p1= aloha_obj.P(1,2)
        self.p2= aloha_obj.P(1,3)
        self.p3= aloha_obj.P(2,2)
        self.p4= aloha_obj.P(2,3)
                
    def testbasicoperation(self):       
        """Test the sum/product run correctly on High level object.
        Those test will be basic since everything should derive from particle
        """
       
        new = self.p1 * self.p2 + self.p3 * self.p4       
        self.assertEqual(new.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(new), 2)
       
        new2 =  aloha_obj.Gamma(1,2,3) * aloha_obj.P(1,2) 

        self.assertEqual(new2.__class__, aloha_lib.MultLorentz)
        self.assertEqual(len(new2), 2)         
        
        new2 += aloha_obj.Gamma(1,2,3) * aloha_obj.P(1,3)
        self.assertEqual(new2.__class__, aloha_lib.AddVariable) 
        self.assertEqual(len(new2), 2)
        self.assertNotEqual(new, new2)
    
    def test_short_power(self):
        """ Test that we can take a square of an object --fully auto contracted"""

        product = self.p2 * self.p2
        power = self.p2**2

        self.assertEqual(power.__class__, aloha_lib.MultLorentz)        
        self.assertEqual(product, power)
        power = power.expand(veto=range(100))

        keys= power.keys()
        keys.sort()
        self.assertEqual(keys, [(0,)])
        solution = '( (P3_0 * P3_0) + (-1 * P3_1 * P3_1) + (-1 * P3_2 * P3_2) + (-1 * P3_3 * P3_3) )'
        
        self.assertEqual(str(power[(0,)]), solution)
        
    def test_short_equality(self):
        """test the equality of Lorentz Object"""
        
        self.assertEqual(self.p1,self.p1)
        self.assertNotEqual(self.p1,self.p2)
        self.assertNotEqual(self.p1,self.p3)
        self.assertNotEqual(self.p1,self.p4)
        self.assertEqual(self.p1, aloha_obj.P(1,2))
        
        self.assertNotEqual(self.p1, aloha_obj.Gamma(1,2,3))
        
        new = aloha_obj.Gamma(1,2,3) * aloha_obj.P(1,2)
        new2 = aloha_obj.Gamma(1,2,3) * aloha_obj.P(1,2)
        self.assertEqual(new, new2)
        
        #Check that sum indices are  consider for equality
        new3 = aloha_obj.Gamma(3,2,3) * aloha_obj.P(3,2)
        self.assertNotEqual(new, new3)
        
        new4 = aloha_obj.P(3,2) * aloha_obj.Gamma(3,2,3)
        self.assertNotEqual(new, new4)
        self.assertNotEqual(new3, new4)
        
        new5 = aloha_obj.P(4,2) * aloha_obj.Gamma(4,2,3)
        self.assertNotEqual(new, new5)
        self.assertNotEqual(new3, new5)
        self.assertNotEqual(new4, new5)
        
        new6 = aloha_obj.P(3,2) * aloha_obj.Gamma(3,3,2)       
        self.assertNotEqual(new, new6)
        
        new7 = aloha_obj.P(3,4) * aloha_obj.Gamma(3,2,3)    
        self.assertNotEqual(new, new7)
        
        #Test contraction on spin
        new = aloha_obj.Gamma(3,3,2) * aloha_obj.Gamma(2,2,4) * \
                                                    aloha_obj.P(3,3) * aloha_obj.P(2,4)
        new2 = aloha_obj.Gamma(3,3,2) * aloha_obj.Gamma(2,2,4) * \
                                                    aloha_obj.P(3,4) * aloha_obj.P(2,3)
        self.assertNotEqual(new,new2)
    
        new3 = aloha_obj.P(1,3) * aloha_obj.Gamma(1,3,1) * aloha_obj.P(4,4) * \
                                                        aloha_obj.Gamma(4,1,4)
        self.assertNotEqual(new, new3)
        self.assertNotEqual(new2, new3)
                                                            
        new4 = aloha_obj.P(1,3) * aloha_obj.Gamma(1,3,2) * aloha_obj.P(4,4) * \
                                                        aloha_obj.Gamma(4,1,4)
        self.assertNotEqual(new,new4)
    
    def testexpand(self):
        """Test if the expansion from HighLevel to LowLevel works correctly"""
        
        #expand a single object
        obj = aloha_obj.P(1,2)
        low_level = obj.expand()

        keys= low_level.keys()
        keys.sort()
        self.assertEqual(keys, [(0,),(1,),(2,),(3,)])
        self.assertEqual(str(low_level[(0,)]), '(P2_0)')
        self.assertEqual(str(low_level[(1,)]), '(P2_1)')
        self.assertEqual(str(low_level[(2,)]), '(P2_2)')
        self.assertEqual(str(low_level[(3,)]), '(P2_3)')

        
        #expand a product
        obj = aloha_obj.P(1,2) * aloha_obj.P(2,3)
        low_level = obj.expand()
        
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind), \
                             aloha_lib.Variable('P3_%s' % ind[0]) *aloha_lib.Variable('P2_%s' % ind[1])) 
                             
        
        #expand a sum
        obj = aloha_obj.P(1,2) + aloha_obj.P(1,3)
        low_level = obj.expand()
        
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind), \
                             aloha_lib.Variable('P2_%s' % ind[0]) + \
                             aloha_lib.Variable('P3_%s' % ind[0]))
            
        #expand zero
        obj = aloha_obj.P(1,2) - aloha_obj.P(1,2)
        obj = obj.simplify()
        self.assertEqual(obj, 0)
        #low_level = obj.expand()
        #pass_in_check = 0
        #for ind in low_level.listindices():
        #    pass_in_check += 1
        #    self.assertEqual(low_level.get_rep(ind), 0)
        #self.assertEqual(pass_in_check, 1)      
             
        #expand zero without first simplification
        obj = aloha_obj.P(1,2) - aloha_obj.P(1,2)
        low_level = obj.expand().simplify()
        pass_in_check = 0 
        for ind in low_level.listindices():
            pass_in_check += 1
            self.assertEqual(low_level.get_rep(ind), 0)
        self.assertEqual(pass_in_check, 4)  
        
        #expand standard frac variable -> Different treatment now
        #obj = aloha_obj.P(1,2) / aloha_obj.P(1,2)   
        #obj = obj.expand()
        #result = {(0,): aloha_lib.Variable('P2_0',[2]), \
        #                            (1,): aloha_lib.Variable('P2_1',[2]), \
        #                            (2,): aloha_lib.Variable('P2_2',[2]), \
        #                            (3,): aloha_lib.Variable('P2_3',[2])}
        #for i in range(3):
        #    self.assertEqual(result[tuple([i])], obj.numerator[tuple([i])])
        #    self.assertEqual(result[tuple([i])], obj.denominator[tuple([i])])
               
        
        # Check for the prefactor
        obj = 36 * aloha_obj.P(1,2)
        obj = obj.expand()
        for ind in obj.listindices():
            expression = obj.get_rep(ind)
            self.assertEqual(expression.prefactor, 36)
             
        # Check for the prefactor
        obj = 36 * aloha_obj.P(1,2) * aloha_obj.P(2,2)
        obj = obj.expand()
        for ind in obj.listindices():
            expression = obj.get_rep(ind)
            self.assertEqual(expression.prefactor, 36)  
  
    def test_short_expand_veto(self):
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
        analytical = (P(-1, t)* P(-1,t)-1) * (Metric(alpha, beta))
        analytical= analytical.expand(veto=range(100))
    
    
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1.0/48#(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
    
        analytical2 = (P(-1, t)* P(-1,t)-1) * (Metric(alpha, beta))
        analytical2= analytical2.expand()
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            try:
                exec('%s = %s' % (name, cexpr))   
            except:
                pass
        
        for ind in analytical.listindices():
            data1 = analytical.get_rep(ind)
            data2 = analytical2.get_rep(ind)
            self.assertAlmostEqual(eval(str( data1 )),eval(str(data2)))
        
        
        
    def testTraceofObject(self):
        """Check that we can output the trace of an object"""
        
        obj = aloha_obj.Gamma(1,1,1)
        obj.expand()
        obj.simplify()      

    def testscalarmanipulation(self):
        """Deal correctly with Scalar type of LorentzObject"""
        
        obj= aloha_obj.Mass(3) 
        obj = obj.simplify()
        low_level = obj.expand()
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind).__class__, aloha_lib.MultVariable)
            self.assertEqual(str(low_level.get_rep(ind)), '(M3)')
                                
        obj= aloha_obj.Mass(3) * aloha_obj.P(1,2)
        obj = obj.simplify()
        low_level = obj.expand()
        self.assertEqual(low_level.__class__, aloha_lib.LorentzObjectRepresentation)
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind).__class__, aloha_lib.MultVariable)
            self.assertEqual(low_level.get_rep(ind),  
                                 aloha_lib.Variable('P2_%s' % ind[0])* aloha_lib.Variable('M3'))
    
    
    def test_short_spin32propagator(self):
        """check various property of the spin3/2 propagator"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        Gamma = aloha_obj.Gamma
        Identity = aloha_obj.Identity
        t = 1
        mu, nu, s0, s1, s2 = 2,3,4,5,6
        
        zero = P(mu,t) * aloha_obj.Spin3halfPropagatorout(mu,nu,s1,s2, t)
        zero = zero.expand(veto=range(100))
        P1_0, P1_1, P1_2, P1_3 = 2,0,0,0
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    

        zero = P(mu,t) * aloha_obj.Spin3halfPropagatorin(mu,nu,s1,s2, t)
        zero = zero.expand(veto=range(100))
        P1_0, P1_1, P1_2, P1_3 = 2,0,0,0
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0) 
     
    def test_short_mass_overmass(self):
        """check various property of the spin3/2 propagator"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        Gamma = aloha_obj.Gamma
        Identity = aloha_obj.Identity
        Gamma = aloha_obj.Gamma
        PSlash = aloha_obj.PSlash
        Mass = aloha_obj.Mass
        OverMass2 = aloha_obj.OverMass2
        Identity = aloha_obj.Identity
        t = 1
        mu, nu, s0, s1, s2 = 2,3,4,5,6
        Spin3halfPropagator =  lambda nu, s1, s2, part: (P(-1,part)**2 - Mass(part)*Mass(part)) * \
                             (Mass(part) * Identity(-3, s2) )  
        
        #- 1/3 * (PSlash(s1,-2,part) + Identity(s1, -2) * Mass(part))* \
        #                     (PSlash(-2,-3, part) - Identity(-2,-3) * Mass(part)) * \
        #                     (P(-1,part)**2 - Mass(part)*Mass(part))
        #                     (Mass(part) * Identity(-3, s2) )
                                     
        zero = Spin3halfPropagator(nu,s1,s2, t)
        zero = zero.expand(veto=range(100))
        P1_0, P1_1, P1_2, P1_3 = 2,0,0,0
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M99 = M1
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)     
 
 
        

    def test_short_part_spin32propagator(self):
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        Gamma = aloha_obj.Gamma
        Identity = aloha_obj.Identity
        Mass = aloha_obj.Mass
        Pslash = aloha_obj.PSlash
        part = 1
        mu, nu, s0, s1, s2,s3 = 2,3,4,5,6,7
        
        
        paranthesis = (Gamma(mu,s1,s2) + Identity(s1, s2) *  P(mu, part) * Mass(part) * OM(part)) * Gamma(nu,s2,s3)
        paranthesis = Gamma(mu,s1,s2) * Gamma(nu,s2,s3) + Identity(s1, s2) *  P(mu, part) * Mass(part) * OM(part) * Gamma(nu,s2,s3)
        #paranthesis =  Gamma(mu,s1,s2) * Gamma(nu,s2,s3)
        goal = (Pslash(s1,s2,part) + Mass(part) * Identity(s1,s2)  ) * Gamma(nu, s2, s3)
        #goal = Pslash(s1,s2,part) * Gamma(nu, s2, s3)
        goal2= P(mu,part) * paranthesis 
        goal2 =  P(mu,part) * Gamma(mu,s1,s2) * Gamma(nu,s2,s3) + Identity(s1, s2) *  P(mu,part) * P(mu, part) * Mass(part) * OM(part) * Gamma(nu,s2,s3)
        zero = goal2 - goal
        
        #zero = zero.simplify()
        zero=zero.expand()
        P1_0, P1_1, P1_2, P1_3 = 20,3,4,5
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            try:
                exec('%s = %s' % (name, cexpr))   
            except:
                pass
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertEqual(eval(str(data)),0)


    
    
    def test_short_spin2propagator(self):
        """Check that the two definition are coherent"""
        
        obj = aloha_obj
        t = 1
        mu, nu, rho, sigma = 1,2,3,4
        propa = (obj.Metric(mu,rho) - obj.OverMass2(t) * obj.P(mu,t) *obj.P(rho,t) ) *\
                (obj.Metric(nu,sigma) - obj.OverMass2(t) * obj.P(nu,t) *obj.P(sigma,t) )
        propa = propa + \
                (obj.Metric(mu,sigma) - obj.OverMass2(t) * obj.P(mu,t) *obj.P(sigma,t) ) *\
                (obj.Metric(nu,rho) - obj.OverMass2(t) * obj.P(nu,t) *obj.P(rho,t) )
        propa = propa - 2/3 * \
                (obj.Metric(mu,nu) - obj.OverMass2(t) * obj.P(mu,t) *obj.P(nu,t) ) *\
                (obj.Metric(rho,sigma) - obj.OverMass2(t) * obj.P(rho,t) *obj.P(sigma,t) )
        
        prop = aloha_obj.Spin2Propagator(mu,nu,rho,sigma, t)
        zero = propa - 2 * prop
        
        
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/36
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)    
 
    def test_short_spin2propagator2(self):
        """test the spin2 propagator is coherent with it's expanded expression"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
        
        propa = 1/2 *( Metric(mu, alpha)* Metric(nu, beta) +\
                       Metric(mu, beta) * Metric(nu, alpha) - \
                       Metric(mu, nu) * Metric(alpha, beta))
        
        propa = propa - 1/2 * OM(t) * \
                  (Metric(mu,alpha)* P(nu, t) * P(beta, t) + \
                   Metric(nu, beta) * P(mu, t) * P(alpha, t) + \
                   Metric(mu, beta) * P(nu, t) * P(alpha, t) + \
                   Metric(nu, alpha) * P(mu, t) * P(beta , t) )
        
        propa = propa + 1/6 * Metric(mu, nu) * Metric(alpha, beta)
        propa = propa + 4/6 * OM(t) * OM(t) * P(mu,t) * P(nu, t) * P(alpha,t) * P(beta,t)
        propa = propa + 2/6 * OM(t) * Metric(mu, nu) *  P(alpha,t) * P(beta,t)
        propa = propa + 2/6 * OM(t) * Metric(alpha, beta) *  P(mu,t) * P(nu,t)     
        
             
        zero = propa - aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 11
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    

    def test_short_spin2propagator3(self):
        """test the spin2 propagator property (contraction gives zero)"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
             
        zero = P(mu,t) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        
        zero = zero.expand(veto=range(100)).simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    
        
        zero = Metric(mu,nu) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        zero = zero.expand(veto=range(100)).simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0) 
    
    def test_short_spin2propagator4(self):
        """test the spin2 propagator is correctly contracted (even offshell)"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
        aloha = complex(0,1)*Metric(mu,nu) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        analytical = complex(0, 1/3) * (OM(t) * P(-1, t)* P(-1,t) - 1) * (Metric(alpha, beta) + 2 * OM(t) * P(alpha,t)*P(beta,t))
        
        
        aloha = aloha.expand().simplify().factorize() 
        analytical= analytical.expand().simplify().factorize()

        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1.0/48#(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            try:
                exec('%s = %s' % (name, cexpr))   
            except:
                pass
        
        for ind in analytical.listindices():
            data1 = aloha.get_rep(ind)
            data2 = analytical.get_rep(ind)
            self.assertAlmostEqual(eval(str( data1 )),eval(str(data2)))
            
    def test_short_spin2propagator5(self):
        """test the spin2 propagator is correctly contracted --part by part --"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OverMass2 = aloha_obj.OverMass2
        Spinor = aloha_obj.Spinor
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        P1_0,P1_1,P1_2,P1_3 = 1000, 3, 4, 1000
        P2_0,P2_1,P2_2,P2_3 = 1000, 3, 6, -1000
        P3_0,P3_1,P3_2,P3_3 = 2000, 2, 6, 9
        
        F1_1, F1_2, F1_3, F1_4  = -44.7213595499958, 62,34,23
        F2_1, F2_2, F2_3, F2_4  = 12, 44, 72, -45 
        OM1,OM2,OM3 = 0 , 0, 1.0 / 500**2
        M3 = 500
        

        #part 1 
        p1 = 0.5j * ( Metric(1003,'I2') * Metric(2003,'I3') * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2))
        p1e = p1.expand(veto=range(100)).simplify().factorize()
        
        solp1 = complex(0,1/2) * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2)
        zero = p1e - solp1.expand(veto=range(100)).simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        #part 2
        p2 =   0.5j * ( Metric(1003,'I3') * Metric(2003,'I2') * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p2e = p2.expand(veto=range(100)).simplify().factorize()
        zero = p2e - solp1.expand(veto=range(100)).simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        # part 3 -and part 8
        p3 = complex(0,-1/3) * ( Metric(1003,2003)**2 * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2) )
        p3e = p3.expand(veto=range(100)).simplify()
        solp3 = complex(0,-4/3) * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2)
        zero = p3e - solp3.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
            
        # part 4
        p4 = -0.5j * ( Metric(1003,'I2') * P(2003,3) * P('I3',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p4e = p4.expand(veto=range(100)).simplify()
        solp4 = complex(0,-1/2) * OverMass2(3) * P('I2',3) * P('I3',3) * Spinor(-1,1) * Spinor(-1,2)
        zero = p4e - solp4.expand(veto=range(100))
        
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        # part 5
        p5 = -0.5j * ( Metric(2003,'I3') * P(1003,3) * P('I2',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p5e = p5.expand(veto=range(100)).simplify()
        zero = p5e - solp4.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)   
        
        #part 6    
        p6 = -0.5j * ( Metric(1003,'I3') * P(2003,3) * P('I2',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )   
        p6e = p6.expand(veto=range(100)).simplify()
        zero = p6e - solp4.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        #part 7
        p7= -0.5j * ( Metric(2003,'I2') * P(1003,3) * P('I3',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p7e = p7.expand(veto=range(100)).simplify()
        zero = p7e - solp4.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        # part 9
        p9 = complex(0,1/3) * ( OverMass2(3) * P('I2',3) * P('I3',3) * Metric(1003,2003)**2 * Spinor(-1,1) * Spinor(-1,2) )
        p9e = p9.expand(veto=range(100)).simplify()
        solp9 = complex(0,4/3) * ( OverMass2(3) * P('I2',3) * P('I3',3) * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p9e - solp9.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
            
        # part 10
        p10 = complex(0,1/3) * ( OverMass2(3) * P(1003,3) * P(2003,3) * Metric('I2','I3') * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p10e = p10.expand(veto=range(100)).simplify()
        solp10 = complex(0,1/3) * ( OverMass2(3) * P(-1,3) **2 * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p10e - solp10.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        
        # part 11
        p11 = complex(0,2/3) * ( OverMass2(3)**2 * P('I2',3) * P('I3',3) * P(1003,3) * P(2003,3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p11e = p11.expand(veto=range(100)).simplify()
        solp11 = complex(0,2/3) * ( OverMass2(3)**2 * P(-1,3) **2 * P('I2',3) * P('I3',3)  * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p11e - solp11.expand(veto=range(100)).simplify()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
            
        # full
        full = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p9 + p10 + p11
        fulle = full.expand(veto=range(100))
        solfull = complex(0,1/3) * ((OverMass2(3) * P(-1, 3)**2 - 1) * (Metric('I2','I3') + 2 * OverMass2(3) * P('I2',3)*P('I3',3)) * Spinor(-1,1) * Spinor(-1,2))  
        solfullbis = 2 * solp1 + solp3 + 4 * solp4 + solp9 +solp10 + solp11
        # first sanity
        zero = solfullbis.expand(veto=range(100)) - solfull.expand(veto=range(100))
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0,6)
        
        
        zero = fulle - solfull.expand(veto=range(100))
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0,6)
        
        
        
class TestLorentzObjectRepresentation(unittest.TestCase):
    """Class to test the operation in the LorentzObjectRepresentation"""
    
    def setUp(self):
        aloha_lib.KERNEL.clean()
        #for lorentz manipulation
        self.p1nu = aloha_obj.P(1,1)
        self.p1nu = self.p1nu.expand()
        self.p1mu = aloha_obj.P(2,1)
        self.p1mu = self.p1mu.expand()   
        self.p2nu = aloha_obj.P(1,2)
        self.p2nu = self.p2nu.expand()
        self.p2mu = aloha_obj.P(2,2)
        self.p2mu = self.p2mu.expand()
        
        #for lorentz - spin manipulation
        self.gamma_nu_ij = aloha_obj.Gamma(1,1,2)
        self.gamma_nu_ij = self.gamma_nu_ij.expand()
        self.gamma_nu_ji = aloha_obj.Gamma(1,2,1)
        self.gamma_nu_ji = self.gamma_nu_ji.expand()
        self.gamma_mu_ij = aloha_obj.Gamma(2,1,2)    
        self.gamma_mu_ij = self.gamma_mu_ij.expand()
        self.gamma_nu_jk = aloha_obj.Gamma(1,2,3)
        self.gamma_nu_jk = self.gamma_nu_jk.expand()
        self.gamma_mu_jk = aloha_obj.Gamma(2,2,3)
        self.gamma_mu_jk = self.gamma_mu_jk.expand()
        self.gamma_nu_kl = aloha_obj.Gamma(1,3,4)
        self.gamma_nu_kl = self.gamma_nu_kl.expand()
        self.gamma_mu_kl = aloha_obj.Gamma(2,3,4)
        self.gamma_mu_kl = self.gamma_mu_kl.expand()    
        self.gamma_mu_ki = aloha_obj.Gamma(2,3,1)
        self.gamma_mu_ki = self.gamma_mu_ki.expand()     
   
    def testlistindices(self):
        """test that we return the correct list of indices"""
        
        #only lorentz indices
        test1 = aloha_lib.LorentzObjectRepresentation([],[1,2],[])
        
        already_use=[]
        for ind in test1.listindices():
            self.assertFalse(ind in already_use, '%s appear two times' % ind)
            already_use.append(list(ind))
            for value in ind:
                self.assertTrue(value >= 0)
                self.assertTrue(value < 4)
        self.assertEqual(len(already_use), 16)
        
        #only spin indices
        test1 = aloha_lib.LorentzObjectRepresentation([],[],[1,2,3])
        
        already_use=[]
        for ind in test1.listindices():
            self.assertFalse(ind in already_use, '%s appear two times' % ind)
            already_use.append(list(ind))
            for value in ind:
                self.assertTrue(value >= 0)
                self.assertTrue(value < 4)
        self.assertEqual(len(already_use), 64)
        
        #mix of indices        
        test1 = aloha_lib.LorentzObjectRepresentation([],[1],[1,2,3])
        
        already_use=[]
        for ind in test1.listindices():
            self.assertFalse(ind in already_use, '%s appear two times' % ind)
            already_use.append(list(ind))
            for value in ind:
                self.assertTrue(value >= 0)
                self.assertTrue(value < 4)
        self.assertEqual(len(already_use), 256)
        
        #only one indice        
        test1 = aloha_lib.LorentzObjectRepresentation([],[1],[])
        
        already_use=[]
        for ind in test1.listindices():
            self.assertFalse(ind in already_use, '%s appear two times' % ind)
            already_use.append(list(ind))
            for value in ind:
                self.assertTrue(value >= 0)
                self.assertTrue(value < 4)
        self.assertEqual(len(already_use), 4)
        
        #no indices        
        test1 = aloha_lib.LorentzObjectRepresentation(38,[],[])
        
        already_use=[]
        for ind in test1.listindices():
            self.assertEqual(ind,[0])
            already_use.append(list(ind))
        self.assertEqual(len(already_use), 1)    
        
    def test_short_split(self):
        """check that we can split correctly an expression"""
        p2rho = aloha_obj.P(3,2)
        p2rho = p2rho.expand()
        expr = self.p1mu *self.gamma_mu_ij *self.gamma_nu_ji * p2rho
        #expr = expr.expand()
        ids = [aloha_lib.KERNEL['P2_%s'%i] for i in [0,1,2,3]] 
        data = expr.split(ids)
        self.assertTrue((0,0,0,0) not in data)
        self.assertEqual(len(data),4)
        

    def testgetrepresentation(self):
        """Check the way to find representation"""
        
        data={(0,0):1, (0,1):2, (0,2):3, (0,3):4,
              (1,0):2, (1,1):4, (1,2):6, (1,3):8,
              (2,0):3, (2,1):6, (2,2):9, (2,3):12,
              (3,0):4, (3,1):8, (3,2):12, (3,3):16
              }
                
        repr1 = aloha_lib.LorentzObjectRepresentation(data, [1], [1])
        repr2 = aloha_lib.LorentzObjectRepresentation(data, [1, 2], [])
        repr3 = aloha_lib.LorentzObjectRepresentation(data, [], [1, 2])
        
        for ind in repr1.listindices():
            self.assertEquals(repr1.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr2.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr3.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            
        
        #check the dealing with scalar
        repr4 = aloha_lib.LorentzObjectRepresentation(49, [], [])
        for ind in repr4.listindices():
            self.assertEquals(repr4.get_rep(ind), 49)


    def test_short_sum_with4ind(self):
        """ check non standard operation with contraction of ()*() """
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        OverMass2 = OM
        F = aloha_obj.Spinor
        Identity = aloha_obj.Identity
        
        mu, nu, alpha, beta, part = 1,2,4,5,3
        
        
        obj1a = 3*( Metric(mu, alpha)* Metric(nu, beta) )
        
        
        
        obj1b=        -5 * OverMass2(part) * (\
                                Metric(mu, beta) * P(nu, part) * P(alpha, part) )

        obj1 = obj1a + obj1b
        
        # check part by part
        obj1a_rep = obj1a.simplify().expand().simplify()
        assert obj1a_rep.lorentz_ind == [2,5,1,4] , "test not valid if condition not met"
        self.assertEqual(str(obj1a_rep.get_rep([1,0,0,0])), '0')
        
        obj1b_rep = obj1b.simplify().expand().simplify()
        assert obj1b_rep.lorentz_ind == [4,2,1,5] , "test not valid if condition not met"
        self.assertEqual(str(obj1b_rep.get_rep([0,1,0,0])), '(-5 * P3_0 * P3_1 * OM3)')       
        
        obj1_rep = obj1.simplify().expand().simplify()
        
        assert obj1_rep.lorentz_ind == [2,5,1,4] , "test not valid if condition not met"
        self.assertEqual(str(obj1_rep.get_rep([1,0,0,0])), '(-5 * P3_0 * P3_1 * OM3)')
        
    
        
        eta = Metric(1,2)
        eta_rep = eta.expand()
        
        final = obj1_rep * eta_rep
        final = final.simplify()
        
        solution = obj1 * eta
        solution_rep = solution.simplify().expand().simplify()
        
        P3_0,P3_1,P3_2,P3_3 = 2, 2, 5, 7
        OM3 = 8
        for ind in final.listindices():
            val1 = eval(str(final.get_rep(ind)))
            val2 = eval(str(solution_rep.get_rep(ind)))
            self.assertAlmostEqual(val1, val2, msg='not equal data for ind: %s, %s != %s' % (ind, val1, val2))



         
    def testsetrepresentation(self):
        """Check the way to set a representation"""
        
        goal=[[1, 2, 3 , 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]
        
        repr1 = aloha_lib.LorentzObjectRepresentation([], [1], [1])
        repr2 = aloha_lib.LorentzObjectRepresentation([], [1, 2], [])
        repr3 = aloha_lib.LorentzObjectRepresentation([], [], [1, 2])
        
        for ind in repr1.listindices():
            repr1.set_rep(ind, (ind[0]+1)*(ind[1]+1))
            repr2.set_rep(ind, (ind[0]+1)*(ind[1]+1))
            repr3.set_rep(ind, (ind[0]+1)*(ind[1]+1))

        for ind in repr1.listindices():
            self.assertEquals(repr1.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr2.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr3.get_rep(ind), (ind[0]+1)*(ind[1]+1))

        for ind in repr1.listindices():
            self.assertEquals(repr1.get_rep(ind), goal[ind[0]][ind[1]])
            self.assertEquals(repr2.get_rep(ind), goal[ind[0]][ind[1]])
            self.assertEquals(repr3.get_rep(ind), goal[ind[0]][ind[1]])
            
            self.assertEquals(repr1.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr2.get_rep(ind), (ind[0]+1)*(ind[1]+1))
            self.assertEquals(repr3.get_rep(ind), (ind[0]+1)*(ind[1]+1))    
            
                    
    def testtensorialproductlorentz(self):
        """Test that two object have correct product"""
        
        product = self.p1nu * self.p2mu
        
        #check global
        self.assertTrue(isinstance(product, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(product.lorentz_ind, [1,2])
        self.assertEqual(product.spin_ind, [])
#        self.assertEqual(product.tag, set(['P1','P2']))
        
        #check the representation
        for ind in product.listindices():
            rep = product.get_rep(ind)
            self.assertEqual(rep.__class__, aloha_lib.MultVariable)
            self.assertEqual(len(rep), 2)
            for data in rep:
                name = str(aloha_lib.KERNEL.objs[data])
                if not( name == 'P1_%s' % ind[0] or name == 'P2_%s' % ind[1]):
                    raise Exception('invalid product')
            self.assertNotEqual(str(aloha_lib.KERNEL.objs[rep[0]]), str(aloha_lib.KERNEL.objs[rep[1]]))
        
        
    def testtensorialproductspin(self):
        """test the product in spin indices"""
        
        product1 = self.gamma_nu_ij * self.gamma_mu_kl
        
        #check global
        self.assertTrue(isinstance(product1, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(product1.lorentz_ind, [1,2])
        self.assertEqual(product1.spin_ind, [1,2,3,4])

        
        #check the representation
        for ind in product1.listindices():
            rep = product1.get_rep(ind)
            
            fact1 = self.gamma_nu_ij.get_rep([ind[0],ind[2],ind[3]])
            fact2 = self.gamma_mu_kl.get_rep([ind[1],ind[4],ind[5]])
            self.assertEqual(rep, fact1 * fact2)
            
        
        #Check with a lorentz contraction
        product2 = self.gamma_nu_ij * self.gamma_nu_kl
        
        #check global
        self.assertTrue(isinstance(product2, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(product2.lorentz_ind, [])
        self.assertEqual(product2.spin_ind, [1,2,3,4])
#        self.assertEqual(product2.tag, set([]))
        
        #check the representation
        for ind in product2.listindices():
            rep = product2.get_rep(ind)
            
            sol = product1.get_rep([0,0] + ind) - product1.get_rep([1,1] + ind) - \
                    product1.get_rep([2,2] + ind) -product1.get_rep([3,3] + ind)

            product1.get_rep([2,2] + ind),product1.get_rep([3,3] + ind)            
            self.assertEqual(rep, sol)
            
 
    def testspincontraction(self):
        """Test the spin contraction"""
        prod0 = self.gamma_mu_ij * self.gamma_nu_kl
        prod1 = self.gamma_mu_ij * self.gamma_nu_jk
        
        #check global
        self.assertTrue(isinstance(prod1, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod1.lorentz_ind, [2, 1])
        self.assertEqual(prod1.spin_ind, [1,3])
        
        for ind in prod1.listindices():

            rep = prod1.get_rep(ind)
            sol = prod0.get_rep([ind[0], ind[1], ind[2], 0, 0, ind[3]]) + \
                prod0.get_rep([ind[0], ind[1], ind[2], 1, 1, ind[3]]) + \
                prod0.get_rep([ind[0], ind[1], ind[2], 2, 2, ind[3]]) + \
                prod0.get_rep([ind[0], ind[1], ind[2], 3, 3, ind[3]]) 
            self.assertEqual(rep, sol)
        
        
        prod2 = self.gamma_mu_ij * self.gamma_mu_jk
 
        #check global
        self.assertTrue(isinstance(prod2, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod2.lorentz_ind, [])
        self.assertEqual(prod2.spin_ind, [1,3])

        for ind in prod2.listindices():

            rep = prod2.get_rep(ind)
            sol = prod1.get_rep([0, 0, ind[0], ind[1]])  \
                        - prod1.get_rep([1, 1, ind[0], ind[1]]) + \
                        - prod1.get_rep([2, 2, ind[0], ind[1]]) + \
                        - prod1.get_rep([3, 3, ind[0], ind[1]])
                        
            self.assertEqual(rep, sol)         
        
        #test 3-> scalar
        prod3 = self.gamma_nu_ij * self.gamma_nu_ji 
 
        #check global
        self.assertTrue(isinstance(prod3, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod3.lorentz_ind, [])
        self.assertEqual(prod3.spin_ind, [])            

        for ind in prod3.listindices():
            
            rep = prod3.get_rep(ind)
            sol = prod2.get_rep([0,0])  \
                      + prod2.get_rep([1,1])  \
                      + prod2.get_rep([2,2])  \
                      + prod2.get_rep([3,3])                 
            self.assertEqual(rep, sol)         

        #test 4-> scalar
        prod3 =  self.gamma_nu_ji * self.gamma_nu_ij
 
        #check global
        self.assertTrue(isinstance(prod3, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod3.lorentz_ind, [])
        self.assertEqual(prod3.spin_ind, [])            

        for ind in prod3.listindices():
            
            rep = prod3.get_rep(ind)
            sol = prod2.get_rep([0,0])  \
                      + prod2.get_rep([1,1])  \
                      + prod2.get_rep([2,2])  \
                      + prod2.get_rep([3,3])                 
            self.assertEqual(rep, sol)         



    def testEinsteinsum(self):
        """Test the Einstein summation"""
        
        prod1 = self.p1nu * self.p2mu * self.p2nu

        #check global
        self.assertTrue(isinstance(prod1, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod1.lorentz_ind, [2])
        self.assertEqual(prod1.spin_ind, [])
#        self.assertEqual(prod1.tag, set(['P1','P2']))
        
        #check the representation
        for ind in prod1.listindices():
            rep = prod1.get_rep(ind)
            self.assertEqual(rep.__class__, aloha_lib.AddVariable)
            self.assertEqual(len(rep), 4)
            for data in rep:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                power = [data.count(data2) for i,data2 in enumerate(data) 
                         if data2 not in data[:i]]
                power.sort()
                if len(power) == 2:
                    self.assertEqual(power, [1,2])
                else:
                    self.assertEqual(power, [1,1,1])

        
        # Returning a scalar
        prod2 = self.p1nu * self.p2nu

        #check global
        self.assertTrue(isinstance(prod2, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(prod2.lorentz_ind, [])
        self.assertEqual(prod2.spin_ind, [])
#        self.assertEqual(prod2.tag, set(['P1','P2']))
        
        #check the representation
        for ind in prod2.listindices():
            rep = prod2.get_rep(ind)
            self.assertEqual(rep.__class__, aloha_lib.AddVariable)
            self.assertEqual(len(rep), 4)
            for data in rep:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(len(data), 2)
                self.assertNotEqual(data[0], data[1])
      
    def testeinsteinsum2(self):
        
        class L_gamma_in_lorentz(aloha_lib.LorentzObject):
            """ local representation """
            
            def __init__(self, name, l1, l2):
                """ (name, s1,s2)"""
                
                aloha_lib.LorentzObject.__init__(self,name,[l1,l2], [])
            
            representation = aloha_lib.LorentzObjectRepresentation(
                            {(0,0): 0, (0,1): 0, (0,2): 0, (0,3):-1,
                             (1,0): 0, (1,1): 0, (1,2): -1, (1,3):0,
                             (2,0): 0, (2,1): 1, (2,2): 0, (2,3):0,
                             (3,0): 1, (3,1): 0, (3,2): 0, (3,3):0},
                                        [1,2], [])
 
        class gamma_in_lorentz(aloha_lib.FactoryLorentz):
            object_class = L_gamma_in_lorentz
        
            @classmethod
            def get_unique_name(self, s1, s2):
                return 'gamma_l_%s_%s' % (s1,s2)
        
    
            
#            create_representation = lambda : representation
            
        obj = gamma_in_lorentz(1,2)
        
        
        obj2 = obj.expand()
        self.assertEqual(obj2.get_rep((0,3)), -1)
        self.assertEqual(obj2.get_rep((1,2)), -1)
        self.assertEqual(obj2.get_rep((2,1)), 1)
        self.assertEqual(obj2.get_rep((3,0)), 1)
                        
        new= obj * aloha_obj.P(2,2)
        new = new.simplify()
        new = new.expand()
        new = new.simplify()
        self.assertEqual(new.__class__, aloha_lib.LorentzObjectRepresentation)
        self.assertEqual(new.lorentz_ind, [1])
        self.assertEqual(new.get_rep([3]), aloha_lib.Variable('P2_0'))
        self.assertEqual(new.get_rep([2]), aloha_lib.Variable('P2_1'))
        self.assertEqual(new.get_rep([1]), aloha_lib.Variable('P2_2'))
        self.assertEqual(new.get_rep([0]), aloha_lib.Variable('P2_3')) 
        self.assertEqual(new.get_rep([0]).prefactor, 1)
        self.assertEqual(new.get_rep([1]).prefactor, 1)   
        self.assertEqual(new.get_rep([2]).prefactor, -1)                  
        self.assertEqual(new.get_rep([3]).prefactor, 1)
        
    def testspinsum(self):
        

        class L_gamma_in_spin(aloha_lib.LorentzObject):
            """ local representation """
            
            def __init__(self, name, spin1, spin2):
                """ (name, s1,s2)"""
                
                aloha_lib.LorentzObject.__init__(self,name,[], [spin1, spin2])
            
            representation = aloha_lib.LorentzObjectRepresentation(
                            {(0,0): 0, (0,1): 0, (0,2): 0, (0,3):-1,
                             (1,0): 0, (1,1): 0, (1,2): -1, (1,3):0,
                             (2,0): 0, (2,1): 1, (2,2): 0, (2,3):0,
                             (3,0): 1, (3,1): 0, (3,2): 0, (3,3):0},
                                        [], [1,2])
            
        class gamma_in_spin(aloha_lib.FactoryLorentz):
            object_class = L_gamma_in_spin
        
            @classmethod
            def get_unique_name(self, s1, s2):
                return 'gamma_s_%s_%s' % (s1,s2)
#            create_representation = lambda : representation
        
        
        obj = gamma_in_spin(1,2)
        
        obj2 = obj.expand()
        self.assertEqual(obj2.get_rep((0,3)), -1)
        self.assertEqual(obj2.get_rep((1,2)), -1)
        self.assertEqual(obj2.get_rep((2,1)), 1)
        self.assertEqual(obj2.get_rep((3,0)), 1)
                        
        new= obj * aloha_obj.Spinor(2,2)
        new = new.simplify()
        new = new.expand()
        new = new.simplify()
        self.assertEqual(new.__class__, aloha_lib.LorentzObjectRepresentation)
        self.assertEqual(new.spin_ind, [1])
        self.assertEqual(new.get_rep([3]), aloha_lib.Variable('F2_1'))
        self.assertEqual(new.get_rep([2]), aloha_lib.Variable('F2_2'))
        self.assertEqual(new.get_rep([1]), aloha_lib.Variable('F2_3'))
        self.assertEqual(new.get_rep([0]), aloha_lib.Variable('F2_4')) 
        self.assertEqual(new.get_rep([0]).prefactor, -1)
        self.assertEqual(new.get_rep([1]).prefactor, -1)   
        self.assertEqual(new.get_rep([2]).prefactor, 1)                  
        self.assertEqual(new.get_rep([3]).prefactor, 1)       
      
      
    def test_short_sumofLorentzObj(self):
        """ Check the sumation of LorentzObject"""
        
        sum = self.p1nu + self.p2nu
        
        #check global
        self.assertTrue(isinstance(sum, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(sum.lorentz_ind, [1])
        self.assertEqual(sum.spin_ind, [])
#        self.assertEqual(sum.tag, set(['P1','P2']))
        
        #check the representation
        for ind in sum.listindices():
            rep = sum.get_rep(ind)
            self.assertEqual(rep.__class__, aloha_lib.AddVariable)
            self.assertEqual(len(rep), 2)
            for data in rep:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
        
        ##
        ## check more complex with indices in wrong order
        ##
        
        sum = self.p1nu * self.p2mu + self.p1mu * self.p2nu

        #check global
        self.assertTrue(isinstance(sum, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(sum.lorentz_ind, [2, 1])
        self.assertEqual(sum.spin_ind, [])
#        tag = set(list(sum.tag))

        #check the representation
        for ind in sum.listindices():
            rep = sum.get_rep(ind)
            if rep.prefactor == 1:
                self.assertEqual(rep.__class__, aloha_lib.AddVariable)
                self.assertEqual(len(rep), 2)
                for data in rep:
                    self.assertEqual(data.__class__, aloha_lib.MultVariable)
                    self.assertEqual(data.prefactor, 1)
            else:
                self.assertEqual(rep.__class__, aloha_lib.MultVariable)
                self.assertEqual(len(rep), 2)
                self.assertEqual(rep.prefactor,2)
        sum2 = sum - (self.p1nu * self.p2mu +  self.p2nu * self.p1mu)
        sum2 = sum2.simplify()
        for ind in sum2.listindices():
            rep = sum2.get_rep(ind)
            self.assertEqual(rep, 0)
        
            
        #check sum is unchanged
        self.assertTrue(isinstance(sum, aloha_lib.LorentzObjectRepresentation))
        self.assertEquals(sum.lorentz_ind, [2, 1])
        self.assertEqual(sum.spin_ind, [])
#        self.assertEqual(sum.tag, tag)
        for ind in sum.listindices():
            rep = sum.get_rep(ind)
            if rep.prefactor == 1:
                self.assertEqual(rep.__class__, aloha_lib.AddVariable)
                self.assertEqual(len(rep), 2)
                for data in rep:
                    self.assertEqual(data.__class__, aloha_lib.MultVariable)
                    self.assertEqual(data.prefactor,1)
            else:
                self.assertEqual(rep.__class__, aloha_lib.MultVariable)
                self.assertEqual(len(rep), 2)
                self.assertEqual(rep.prefactor,2)
        self.assertEqual(sum, self.p1nu * self.p2mu + self.p1mu * self.p2nu)
        
        sumbis = self.p1nu * self.p2mu + self.p1mu * self.p2nu 
        for ind in sumbis.listindices():
            self.assertEqual(sumbis.get_rep(ind),sum.get_rep(ind))
             
        sum -= sumbis
        sum = sum.simplify()
        for ind in sum.listindices():
            rep = sum.get_rep(ind)
            self.assertEqual(rep, 0)        
        self.assertEqual(sum,sum2)
        
        #check wrong sum
        self.assertRaises( \
            aloha_lib.LorentzObjectRepresentation.LorentzObjectRepresentationError, \
            aloha_lib.LorentzObjectRepresentation.__add__,self.p1nu,self.p2mu)
        
    
        

class TestSomeObjectProperty(unittest.TestCase):
    """Test that some property pass correctly for Object"""
        
    def testmassisdiffaswidth(self):
        """Ensure that a mass object is different of a width object"""
            
        mass = aloha_obj.Mass(1)
        width = aloha_obj.Width(1)
        self.assertNotEqual(mass, width)
        self.assertNotEqual(mass * mass, mass * width)
        
            
        mass = mass.expand()
        width = width.expand()
        self.assertNotEqual(mass, width)
        self.assertNotEqual(mass * mass, mass * width)
            
        mass = mass.simplify()
        width = width.simplify()
        self.assertNotEqual(mass, width)
        self.assertNotEqual(mass * mass, mass * width)
 
        mass = aloha_obj.Mass(1)
        width = aloha_obj.Width(1)
        sum = mass * mass + mass * width
        sum.simplify()
        self.assertEqual(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 2)
        
    def testIdentityMatrix(self):
        """ Test the Identity Matrix"""
        Identity = aloha_obj.Identity
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Metric = aloha_obj.Metric
        
        #Test that Identity is idenpotent
        obj1 = Identity(1,2).expand()
        obj2 = Identity(1,3).expand() * Identity(3,2).expand()
        self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)  
        self.assertEqual(obj1.spin_ind, obj2.spin_ind)  
        self.assertEqual(obj1, obj2)          
        
        #Test at low level
        obj1 = Gamma(1,1,2).expand()
        obj2 = Identity(1,3).expand() * Gamma(1,3,2).expand()
        self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)  
        self.assertEqual(obj1.spin_ind, obj2.spin_ind)  
        self.assertEqual(obj1, obj2)       
                  
        #Gamma = Identity * Gamma
        obj1 = Gamma(1,1,2)
        obj2 = Identity(1,3) * Gamma(1,3,2)
        obj1 = obj1.simplify().expand().simplify()
        obj2 = obj2.simplify().expand().simplify()
        self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)  
        self.assertEqual(set(obj1.spin_ind), set(obj2.spin_ind))
        for ind in obj1.listindices():
            if obj1.spin_ind == obj2.spin_ind:
                mapind = lambda ind : ind
            else:
                mapind = lambda ind : [ind[0],ind[2],ind[1]]
            self.assertEqual(obj1.get_rep(ind),obj2.get_rep(mapind(ind)))

        
        #self.assertEqual(obj1, obj2)
        
        #Gamma = Identity * Identity * Gamma
        #at low level
        obj1 = Gamma(1,1,2).expand()
        obj2 = Identity(3,4).expand() * Gamma(1,4,2).expand()
        obj3 = Identity(1,3).expand() *obj2
        self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)
        self.assertEqual(obj1.lorentz_ind, obj3.lorentz_ind)
        self.assertEqual(obj2.spin_ind, [3,2])          
        self.assertEqual(obj1.spin_ind, obj3.spin_ind)
        for ind in obj1.listindices():
            self.assertEqual(obj1.get_rep(ind),obj3.get_rep(ind))
        self.assertEqual(obj1, obj3)
        
        #at High Level        
        obj1 = Gamma(1,1,2)
        obj2 = Identity(1,3) * Identity(3,4) 
        obj3 = obj2 * Gamma(1,4,2)
        obj1 = obj1.simplify().expand().simplify()
        obj2 = obj2.simplify().expand().simplify()
        obj3 = obj3.simplify().expand().simplify()        
        #self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)
        self.assertEqual(obj1.lorentz_ind, obj3.lorentz_ind)
        self.assertEqual(set(obj2.spin_ind), set([1,4]))          
        self.assertEqual(set(obj1.spin_ind), set(obj3.spin_ind))
        for ind in obj1.listindices():
            if obj1.spin_ind == obj3.spin_ind:
                mapind = lambda ind : ind
            else:
                mapind = lambda ind : [ind[0],ind[2],ind[1]]
            self.assertEqual(obj1.get_rep(ind),obj3.get_rep(mapind(ind)))
        #self.assertEqual(obj1, obj3)  
              
        #at High Level        
        obj1 = Gamma(1,1,2)
        obj2 = Identity(1,3) * Identity(3,4) * Gamma(1,4,2) 
        obj1 = obj1.simplify().expand().simplify()
        obj2 = obj2.simplify().expand().simplify()    
        self.assertEqual(obj1.lorentz_ind, obj2.lorentz_ind)
        self.assertEqual(set(obj2.spin_ind), set([1,2]))          
        self.assertEqual(set(obj1.spin_ind), set(obj2.spin_ind))
        for ind in obj1.listindices():
            if obj1.spin_ind == obj2.spin_ind:
                mapind = lambda ind : ind
            else:
                mapind = lambda ind : [ind[0],ind[2],ind[1]]            
            self.assertEqual(obj1.get_rep(ind),obj2.get_rep(mapind(ind)))
        #self.assertEqual(obj1, obj2)         


    def testgammaproperty(self):
        """ Check constitutive properties of Gamma """
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric        


        # Sigma_mu_nu = 1/2 [Gamma_mu, Gamma_nu]
        term1 = Sigma('mu','nu','a','b')
        commutator = Gamma('mu','a','c') * Gamma('nu','c','b') - Gamma('nu','a','c') * Gamma('mu','c','b') 
        expr = term1 - 0.25j* commutator
        zero = expr.expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, '%s != 0.0for %s' % \
                             (zero.get_rep(ind), ind))  

        # Gamma_mu* Gamma_mu = 4 * Id
        fact1 = aloha_obj.Gamma('mu', 'a', 'b')
        fact2 = aloha_obj.Gamma('mu', 'b', 'c')
        fact1 = fact1.expand()
        fact2 = fact2.expand()
        
        result = 4 * aloha_obj.Identity('a','c')
        result = result.expand().simplify()
        prod = fact1 * fact2  
        self.assertEqual(prod, result)

        # gamma_product Gamma_mu * Gamma_nu = - Gamma_nu * Gamma_mu
        prod_gam = Gamma(1,1,2) * Gamma(2,2,3)
        prod_gam = prod_gam.simplify().expand().simplify()
        for ind in prod_gam.listindices():
            if ind[0] != ind[1]:
                self.assertEqual(prod_gam.get_rep(ind), 
                    -1 * prod_gam.get_rep((ind[1],ind[0],ind[2],ind[3])),ind)
        
        prod_gam2 = Gamma(2,1,2) * Gamma(1,2,3)
        self.assertNotEqual(prod_gam, prod_gam2)
    
        # Sigma_mu_nu * Sigma_mu_nu = 3* Id
        sigma_cont  = Sigma(1,2,1,2) * Sigma(1,2,2,1) 
        sigma_cont = sigma_cont.expand().simplify()
        self.assertEqual(sigma_cont.get_rep((0,)), 12)

        # Sigma_mu_nu * Gamma_nu = 3/2i * Gamma_nu # Trace
        prod = Sigma(1,2,'a','b') * Gamma(2,'b','a')
        prod = prod.expand().simplify()
        self.assertEqual(prod.get_rep((0,)), 0)

        # Sigma_mu_nu * Gamma_nu = 3/2i * Gamma_nu # Full
        zero = Sigma(1,2,'a','b') * Gamma(2,'b','c') - complex(0,3/2) * Gamma(1,'a','c')
        zero = zero.expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, '%s != 0.0for %s' % \
                             (zero.get_rep(ind), ind))  


    def test_short_other(self):
        """ test that all object are defined"""        
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric  
    
    def test_short_projector(self):
        """test that projector are correctly define"""
        
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Metric = aloha_obj.Metric
        Id = aloha_obj.Identity 
        
        zero = Metric(1003,2003)*ProjM(2,1) + Metric(1003,2003)*ProjP(2,1)- Metric(1003,2003)*Id(2,1)
        zero = zero.expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, '%s != 0.0for %s' % \
                             (zero.get_rep(ind), ind))  
    
    
    def test_short_Pslashproperty(self):
        """Test Pslash"""
    
        Gamma = aloha_obj.Gamma
        P = aloha_obj.P
        M = aloha_obj.Mass
        PSlash = aloha_obj.PSlash
        Identity = aloha_obj.Identity
        
        
        ps1 = PSlash(1,2,3).simplify().expand().simplify()
        
        ps2 = Gamma(-1,1,2) * P(-1,3)
        ps2 = ps2.simplify().expand().simplify()
        zero = ps1 - ps2
        zero = zero.simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, '%s != 0.0for %s' % \
                             (zero.get_rep(ind), ind)) 
            
        
        #checking that (/p + m)(/p-m)=0 (for onshell)
        expr = (PSlash(1,2,1)+ M(1)*Identity(1,2))*(PSlash(2,3,1)-M(1)*Identity(2,3))
        expr = expr.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)

        for ind in expr.listindices():
            data = expr.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)  
        
        #checking that (/p + m)(/p-m)(P)=0 (for onshell)
        expr = (PSlash(1,2,1)+ M(1)*Identity(1,2))*(PSlash(2,3,1)-M(1)*Identity(2,3))*(Gamma(4,3,4)*Identity(3,4) * P(4,1))
        expr = expr.expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        for ind in expr.listindices():
            data = expr.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)  
            
        # check that /P2 /P3 + /P3 /P2 = 2 P2 * P3        
        expr1 = PSlash(1,-1,2) * PSlash(-1,2,3) + PSlash(1,-1,3) * PSlash(-1,2,2)
        expr2 = 2 * P(-1,2) * P(-1,3) * Identity(1,2)
        expr1 = expr1.simplify().expand(veto=range(len(aloha_lib.KERNEL)))
        expr2 = expr2.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        for ind in expr1.listindices():
            P2_0, P2_1, P2_2, P2_3 = 7,2,3,5
            P3_0, P3_1, P3_2, P3_3 = 73,23,30,51
            data1 = expr1.get_rep(ind)
            data2 = expr2.get_rep(ind)
            self.assertAlmostEqual(eval(str(data1)), eval(str(data2))) 
            if data1:
                data1 = data1.simplify()
                self.assertAlmostEqual(eval(str(data1)), eval(str(data2))) 
                
    def testGammaAlgebraDefinition(self):
        """Test the coherence between gamma/gamma5/sigma/projector"""
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric
        
        #Gamma5 = i *Gamma0 * Gamma1 * Gamma2 * Gamma3 
        gamma5 = complex(0,1) * Gamma(0,1,2) * Gamma(1,2,3) * Gamma(2,3,4) * \
                                                                    Gamma(3,4,5)
        self.assertEqual(gamma5.__class__,aloha_lib.MultLorentz)
        self.assertEqual(gamma5.prefactor, complex(0,1))
        
        gamma5_2 = Gamma5(1,5)
        
        gamma5 = gamma5.expand().simplify()
        gamma5_2 = gamma5_2.expand().simplify()
        
        for ind in gamma5_2.listindices():
            component1 = gamma5.get_rep([0,1,2,3] + ind)
            component2 = gamma5_2.get_rep(ind)
            self.assertEqual(component1, component2)
        
        #ProjP = (1+ Gamma5)/2
        
        projp = 1/2 * (Identity(1,2) + Gamma5(1,2))
        projp = projp.simplify()
        projp = projp.expand()
        projp = projp.simplify()
        
        projp2 = ProjP(1,2)
        projp2 = projp2.simplify()
        projp2 = projp2.expand()
        projp2 = projp2.simplify()         

        self.assertEqual(projp,projp2)
        
        #ProjM = (1 - Gamma5)/2
        
        projm = 1/2 * (Identity(1,2) - Gamma5(1,2))
        projm = projm.simplify()
        projm = projm.expand()
        projm = projm.simplify()
        
        projm2 = ProjM(1,2)
        projm2 = projm2.simplify()
        projm2 = projm2.expand()
        projm2 = projm2.simplify()         

        self.assertEqual(projm,projm2)
        
        
        # Identity = ProjP + ProjM
        identity= ProjM(1,2) + ProjP(1,2)
        identity = identity.simplify().expand().simplify()
        
        identity2 = Identity(1,2)
        identity2 = identity2.simplify().expand().simplify()
        
        self.assertEqual(identity,identity2)

        # Gamma* ProjP + Gamma* ProjM =Gamma
        part1 = Gamma(1,1,2) * ProjP(2,3)  + Gamma(1,1,2) * ProjM(2,3)
        part2 = Gamma(1,1,3)
        
        
        part1 = part1.simplify().expand().simplify()
        part2 = part2.simplify().expand().simplify()
        
        zero = part1 - part2
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, '%s != 0.0for %s' % \
                             (zero.get_rep(ind), ind)) 


          
        #metric_mu_nu = 1/2 {Gamma_nu, Gamma_mu} 
        metric = 1/2 * (Gamma(1,1,2)*Gamma(2,2,3) + Gamma(2,1,2)*Gamma(1,2,3))
        metric = metric.simplify().expand().simplify() 
        
        metric2 = Metric(1,2) * Identity(1,3)
        metric2 = metric2.simplify().expand().simplify()
        for ind in metric.listindices(): 
            self.assertEqual(metric.get_rep(ind), metric2.get_rep(ind))
        self.assertEqual(metric, metric2)

       

        sigma = complex(0, 1/4) * (Gamma(1,3,2)*Gamma(2,2,1) - Gamma(2,3,2)*Gamma(1,2,1))
        sigma2 = sigma.expand()
        
        zero = Sigma(1,2,3,1) - sigma
        zero = zero.expand()
        for ind in zero.listindices(): 
            self.assertEqual(zero.get_rep(ind), 0)        
        
        mu, nu, rho, sigma = 1,2,3,4
        commutator = Sigma(mu,nu,1,2) * Sigma(rho, sigma,2,3) - Sigma(rho,sigma,1,2) * Sigma(mu, nu,2,3) 
        algebra = -1j * Metric(mu,rho) * Sigma(nu,sigma,1,3) + \
                  1j * Metric(nu,rho) * Sigma(mu,sigma,1,3) + \
                  -1j * Metric(nu,sigma) * Sigma(mu,rho,1,3) + \
                  1j * Metric(mu,sigma) * Sigma(nu,rho,1,3)
        
        zero = commutator - algebra
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices(): 
            self.assertEqual(zero.get_rep(ind), 0)         
        
        
    def test_short_complex_equality(self):
        """Is this really equal?"""
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric        
        Epsilon = aloha_obj.Epsilon
        P = aloha_obj.P
        PSlash = aloha_obj.PSlash
        
        object2_paper = P(-1,2) * P(-1,3) * Gamma(3,2,1) - P(3,2) * P(-1,3) * Gamma(-1,2,1) \
                  - complex(0,1) * Epsilon(3,-1,-2,-3) * P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*Gamma5(-4,1)
        object2 = Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjM(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) - complex(0,1)*P(-1,2)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1) - Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjP(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) - complex(0,1)*P(-1,2)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1)        
        
        zero = - complex(0,1) * object2_paper - object2
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices(): 
            self.assertEqual(zero.get_rep(ind), 0)  

        object1_paper = 2 * P(-1,2) * P(-1, 3) * Gamma(3,2,1) - P(3,3) * P(-1,2) * Gamma(-1,2,1)\
                        - P(-2,3) * Gamma(3,2,-3)* P(-4,2)*Gamma(-4,-3,-5)*Gamma(-2,-5,1)
        
        object1 = P(-1,2)*P(3,3)*Gamma(-1,2,1) - (P(-2,2)*P(-1,3)*Gamma(-2,-3,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4))/2. + (P(-2,2)*P(-1,3)*Gamma(-2,-4,-3)*Gamma(-1,-3,1)*Gamma(3,2,-4))/2. - P(-1,2)*P(-1,3)*Gamma(3,2,1)
              
        object1 = object1.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        object1_paper = object1_paper.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        P3_0, P3_1, P3_2, P3_3 = 1,2,3,4
        P2_0, P2_1, P2_2, P2_3 = 10,20,30,50
        
    
        for ind in object1.listindices():
            a = -object1.get_rep(ind) - object1_paper.get_rep(ind)
            if a:
                a.simplify()
            self.assertEqual(eval(str(object1.get_rep(ind))), eval(str(object1_paper.get_rep(ind)))) 


        
        object1_paper = 2 * P(-1,2) * P(-1, 3) * Gamma(3,2,1) - P(3,3) * P(-1,2) * Gamma(-1,2,1)\
                        - P(-2,3) * Gamma(3,2,-3)* P(-4,2)*Gamma(-4,-3,-5)*Gamma(-2,-5,1)
        
        object1 = P(-1,2)*P(3,3)*Gamma(-1,2,1) - (P(-2,2)*P(-1,3)*Gamma(-2,-3,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4))/2. + (P(-2,2)*P(-1,3)*Gamma(-2,-4,-3)*Gamma(-1,-3,1)*Gamma(3,2,-4))/2. - P(-1,2)*P(-1,3)*Gamma(3,2,1)
              
        zero = - 1 * object1_paper - object1
        
        zero = zero.simplify()
        zero = zero.expand(veto=range(len(aloha_lib.KERNEL)))
        zero = zero.simplify()
        P3_0, P3_1, P3_2, P3_3 = 1,2,3,4
        P2_0, P2_1, P2_2, P2_3 = 10,20,30,50
        
        
        
        for ind in zero.listindices():
            try:
                self.assertEqual(zero.get_rep(ind), 0) 
            except Exception as error:
                error.message = '%s (for component %s) is not zero' % (zero.get_rep(ind),ind)
                raise AssertionError, error.message
        
        object1_paper = 2 * P(-1,2) * P(-1, 3) * Gamma(3,2,1) - P(3,3) * P(-1,2) * Gamma(-1,2,1)\
                        - P(-2,3) * Gamma(3,2,-3)* P(-4,2)*Gamma(-4,-3,-5)*Gamma(-2,-5,1)
        
        
        object2_paper = P(-1,2) * P(-1,3) * Gamma(3,2,1) - P(3,2) * P(-1,3) * Gamma(-1,2,1) \
                - complex(0,1) * Epsilon(3,-1,-2,-3) * P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*Gamma5(-4,1)

        zero =   object1_paper - object2_paper
        zero = zero.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        for ind in zero.listindices(): 
            self.assertEqual(zero.get_rep(ind), 0)   
            
            
        # Expression provided by FR compare to kentaru model
        object_fr = complex(0,1)*Epsilon(3,4,-1,-2)*P(-1,2)*Gamma(-2,2,-3)*ProjM(-3,1) + P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - complex(0,1)*Epsilon(3,4,-1,-2)*P(-1,2)*Gamma(-2,2,-3)*ProjP(-3,1) + P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)
        object_kent = (Gamma(3,2,-1)*Gamma(4,-1,-10) - Identity(2,-10)*Metric(3,4)) * PSlash(-10,1,2)
        
        zero = object_fr - object_kent
        zero = zero.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        for ind in zero.listindices(): 
            self.assertEqual(zero.get_rep(ind), 0)  
            
        # Same for three point interactions
        object_fr =Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjM(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) - complex(0,1)*P(-1,2)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1) - Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjP(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) - complex(0,1)*P(-1,2)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1) 
        object_kent = P(-1,2)*P(3,3)*Gamma(-1,2,1) - (P(-2,2)*P(-1,3)*Gamma(-2,-3,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4))/2. + (P(-2,2)*P(-1,3)*Gamma(-2,-4,-3)*Gamma(-1,-3,1)*Gamma(3,2,-4))/2. - P(-1,2)*P(-1,3)*Gamma(3,2,1)
                 
        zero = object_fr - complex(0,1)* object_kent
        zero = zero.simplify().expand(veto=range(len(aloha_lib.KERNEL))).simplify()
        
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0)              
        
        
        
    def test_short_parity_for_epsilon(self):

        # usefull shortcut
        Epsilon = aloha_obj.Epsilon
        # test some value
        eps = Epsilon(1,2,3,4)
        eps=aloha_lib.KERNEL.objs[eps[0]] # take the variable object
        
        indices = ((l1, l2, l3, 6 - l1- l2 -l3)
                                 for l1 in range(4) \
                                 for l2 in range(4) if l2 != l1\
                                 for l3 in range(4) if l3 not in [l1,l2])
        for index in indices:
            val1 = eps.give_parity(index)
            val2 = -1 * aloha_obj.give_sign_perm([0,1,2,3], index)
        
            self.assertEqual(val1, val2, 'not same parity for perm %s' % (index,))

    def testEpsilonProperty(self):
        """Test the property of the epsilon object"""
        
        # usefull shortcut
        Epsilon = aloha_obj.Epsilon

        # test some value
        eps = Epsilon(1,2,3,4)
        eps = eps.expand().simplify()
        self.assertEqual(eps.get_rep([0,1,2,3]), -1)
        self.assertEqual(eps.get_rep([0,1,2,2]), 0) 
        self.assertEqual(eps.get_rep([0,1,3,2]), 1) 
        self.assertEqual(eps.get_rep([0,1,1,2]), 0) 
        self.assertEqual(eps.get_rep([0,0,2,2]), 0) 
        self.assertEqual(eps.get_rep([1,2,3,0]), 1) 
        self.assertEqual(eps.get_rep([1,2,0,3]), -1) 
        self.assertEqual(eps.get_rep([1,0,2,3]), 1) 

        # Test the full contraction of two Epsilon
        contraction = Epsilon(1,2,3,4) * Epsilon(1,2,3,4)
        
        contraction = contraction.simplify().expand().simplify()
        self.assertEqual(contraction.get_rep([0]), -24)
        
        # Test the anti-symmetry of the Epsilon
        momentum1 = aloha_obj.P(1,1) #first index lorentz, second part number
        momentum2 = aloha_obj.P(2,1)
        momentum3 = aloha_obj.P(3,1)
        momentum4 = aloha_obj.P(4,1)
        eps = Epsilon(1,2,3,4)
        
        product = eps * momentum1 * momentum2
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))
        
        product = eps * momentum1 * momentum3
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))        
               
        product = eps * momentum1 * momentum4
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))
                    
        product = eps * momentum2 * momentum3
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))
                    
        product = eps * momentum2 * momentum4
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))
                    
        product = eps * momentum3 * momentum4
        product = product.simplify().expand().simplify()
        for ind in product.listindices():
            self.assertEqual(product.get_rep(ind), 0, 'not zero %s for %s' 
                             % (product.get_rep(ind),ind ))
          
        # Epsilon_{mu nu rho alpha} * Epsilon^{mu nu rho beta} = -6 * Metric(alpha,beta)
        fact1 = aloha_obj.Epsilon('a', 'b', 'c', 'd')
        fact2 = aloha_obj.Epsilon('a', 'b', 'c', 'e')
         
        result = -6 * aloha_obj.Metric('d','e')
        result = result.expand().simplify()
        prod = fact1 * fact2
        prod = prod.expand().simplify()

        self.assertEqual(prod, result)
        
        #  Epsilon_{mu nu rho alpha} = - Epsilon_{nu mu rho alpha} 
        ep1 = aloha_obj.Epsilon('a', 'b', 'c', 'd')
        ep2 = aloha_obj.Epsilon('b', 'a', 'c', 'd')
        zero = ep1 + ep2
        zero = zero.expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))        
  
    def testCAlgebraDefinition(self):
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric
        C = aloha_obj.C
        
        #Check basic property of the C function
        # C^-1= -C         
        product = C(1,2) *-1*C(2,3)
        identity = Identity(1,3)
        
        product = product.simplify().expand().simplify()
        identity = identity.simplify().expand().simplify()
        self.assertEqual(product, identity)
        
        # C^T = -C
        first = C(1,2)
        second = -1 * C(2,1)
        first = first.simplify().expand().simplify()
        second = second.simplify().expand().simplify()        
        
        self.assertEqual(first, second)
        
        # C is a real matrix
        for indices in first.listindices():
            value = complex(first.get_rep(indices))
            self.assertEqual(value, value.conjugate())
        
        # C* Gamma5 * C^-1 =  Gamma5^T
        zero = C(1,2) * Gamma5(2,3) * C(3,4) + Gamma5(4,1)
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))
            
        # C* Gamma_mu * C^-1 =  Gamma_mu^T
        zero = C(1,2) * Gamma('mu',2,3) * C(3,4) - Gamma('mu',4,1)
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))               

        # C* Sigma_mu_nu * C^-1 =  Sigma_mu_nu^T
        zero = C(1,2) * Sigma('mu','nu',2,3) * C(3,4) - Sigma('mu','nu',4,1)
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))
                    

    def testConjugateOperator(self):
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        C = aloha_obj.C    
        
        # Check the sign given in Denner
        
        def conjugate(A):
            # contract on 1,2 return on indices 51 52
            return C(51, 2) * A * C(52,1)
        
        
        # check C * 1 * C^ -1 = 1
        A = Identity(1,2)
        AC = conjugate(A)
        A2 = Identity(51,52) 
        zero = AC - A2 
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))        
        
        # check C * Gamma_mu^T * C^ -1 = - Gamma_mu
        A = Gamma('mu',1,2)
        AC = conjugate(A)
        A2 = -1 * Gamma('mu',51,52) 
        zero = AC - A2 
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))         
        
        # check C * (Gamma_mu * Gamma5)^T * C^ -1 =  Gamma_mu * Gamma5
        A = Gamma('mu',1,21) * Gamma5(21,2)
        AC = conjugate(A)
        A2 = Gamma('mu',51,22) * Gamma5(22,52) 
        zero = AC - A2 
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))           

        # check goldstino interaction
        A = -(P(-1,3)*Gamma(-1,-2,1)*Gamma(3,2,-2)) + P(3,3)*Identity(2,1)
        AC = conjugate(A)
        A2 = -(P(-1,3)*Gamma(-1,-2,51)*Gamma(3,52,-2)) + P(3,3)*Identity(52,51) 
        zero = AC + A2 
        zero = zero.simplify()
        zero = zero.expand()
        zero = zero.simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))         
    
        # check goldstino interaction
        A = -(Gamma('nu',-2,1)*Gamma('mu',2,-2)) + Metric('mu','nu') * Identity(2,1)
        AC = conjugate(A)
        A2 = -(Gamma('nu',-2,51)*Gamma('mu',52,-2)) + Metric('mu','nu') * Identity(52,51) 
        zero = AC + A2 
        zero = zero.simplify().expand().simplify()
        for ind in zero.listindices():
            self.assertEqual(zero.get_rep(ind), 0, 'not zero %s for %s' 
                             % (zero.get_rep(ind),ind ))    
    
    def testemptyisFalse(self):

        false = aloha_lib.AddVariable([])
        if false:
            raise AssertionError, 'empty list are not False'
        
        false = aloha_lib.MultVariable([])
        if false:
            raise AssertionError, 'empty list are not False'      
          

class TestSimplify(unittest.TestCase):
    """Check that the simplification works correctly"""        

    def testsimplifyMultLorentz(self):
        
        # For Standard Product : No Simplification
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.Gamma(3, 4, 5)
        
        simp = prod.simplify()
        self.assertEqual(simp, prod)
        
        # Look if Multiply by Propagator
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.SpinorPropagatorout(1, 2 ,3)
        simp = prod.simplify()
        
        self.assertEqual(simp.__class__, aloha_lib.AddVariable)
        simp = simp.expand()
        simp = simp.simplify()
                
        
class test_aloha_creation(unittest.TestCase):
    """ test the creation of one aloha routine from the create_aloha routine """
    
    
    class Lorentz(object):

        require_args=['name','spins','structure']
    
        def __init__(self, name, spins, structure='external', **opt):
            args = (name, spins, structure)
                
            assert(len(self.require_args) == len (args))
    
            for i, name in enumerate(self.require_args):
                setattr(self, name, args[i])
    
            for (option, value) in opt.items():
                setattr(self, option, value)
            
    def test_short_aloha_VVS(self):
        """ Test the VVS creation of vertex """
        
        VVS_15 = self.Lorentz(name = 'VVS_15',
                 spins = [ 3, 3, 1 ],
                 structure = 'Metric(1,2)')

        abstract = create_aloha.AbstractRoutineBuilder(VVS_15).compute_routine(3)
        
        self.assertEqual(abstract.expr.nb_lor, 0)
        self.assertEqual(abstract.expr.nb_spin, 0)
        
    def test_short_aloha_ZPZZ(self):
        """ Check the validity of Funny Zp coupling to z z """
                
        ZPZZ = self.Lorentz(name = 'ZPZZ',
                 spins = [ 3, 3, 3 ],
                 structure = 'P(-1,1)*Epsilon(3,1,2,-2)*P(-1,1)*P(-2,2)-Epsilon(3,1,2,-2)*P(-1,2)*P(-1,2)*P(-2,1)-Epsilon(3,2,-1,-2)*P(1,1)*P(-1,2)*P(-2,1)+Epsilon(3,1,-1,-2)*P(2,2)*P(-1,2)*P(-2,1)')
    
        abstract_ZP = create_aloha.AbstractRoutineBuilder(ZPZZ).compute_routine(0, factorize=False)
        expr = abstract_ZP.expr

        V2_1, V2_2, V2_3, V2_4  = 1, 2, 3, 4
        V1_1, V1_2, V1_3, V1_4  = 5, 6, 7, 8
        V3_1, V3_2, V3_3, V3_4  = 9, 100, 11, 13
        OM1,OM2,OM3 = 9,11,13
        j = complex(0,1)
        P1_0,P1_1,P1_2,P1_3 = 10, 11, 12, 19
        P2_0,P2_1,P2_2,P2_3 = 101, 111, 121, 134
        P3_0,P3_1,P3_2,P3_3 = 1001, 1106, 1240, 1320
        for name, cexpr in abstract_ZP.contracted.items():
            exec('%s = %s' % (name, cexpr))

        for ind in expr.listindices():
            self.assertEqual(eval(str(expr.get_rep(ind))), 178727040j)

    def test_short_regular_expression_propa(self):

        mod_numerator = create_aloha.AbstractRoutineBuilder.mod_propagator_expression
            

        text = '1*complex(1,1) * ( - Metric(1, 2) + P(1, id) * P(2, id) / (Mass(id) * Mass(id)) )'
        self.assertEqual(mod_numerator({1:3}, text),
               '1*complex(1,1) * ( - Metric(3, 2) + P(3, id) * P(2, id) / (Mass(id) * Mass(id)) )')

        text = '1*complex(0,1) * ( - Metric(1, 2) + P(1, id) * P1(2, id) / (Mass(id) * Mass(id)) )'
        self.assertEqual(mod_numerator({1:3}, text),
               '1*complex(0,1) * ( - Metric(3, 2) + P(3, id) * P1(2, id) / (Mass(id) * Mass(id)) )')

        text = "complex(0,1) * ( - Metric(1, 2) + P(1, id) * P(2, id) / (Mass(id) * Mass(id)) )"
        tag = {'1': 3, '2': 'I2', 'id': 3}
        self.assertEqual(mod_numerator(tag, text),
               "complex(0,1) * ( - Metric(3, 'I2') + P(3, 3) * P('I2', 3) / (Mass(3) * Mass(3)) )")          


        text = "P('mu', id) * P('mu', id) - Mass(id) * Mass(id) + complex(0,1) * Mass(id) * Width(id)"
        tag = {'1': 3, '2': 'I2', 'id': 3}
        self.assertEqual(mod_numerator(tag, text),
               "P('mu', 3) * P('mu', 3) - Mass(3) * Mass(3) + complex(0,1) * Mass(3) * Width(3)")

    def test_short_use_of_library_spin2(self):
        """ check that use the library or the usual definition is the same """
        
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        F = aloha_obj.Spinor
        Identity = aloha_obj.Identity
        t = 3
        mu, nu, alpha, beta = 1003,2003,'I2','I3' 
        
        # One Expand:
        import time
        start = time.time()
        one_exp = Metric(mu,nu) * Identity(1,2)* aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)  * F(1,1) * F(2,2)
        one_exp = one_exp.simplify().expand().simplify()#.factorize()

        # Separate Expand:
        start = time.time()
        two_exp = Metric(mu,nu) * Identity(1,2)  * F(1,1) * F(2,2)
        two_exp = two_exp.simplify().expand().simplify()
        
        two_exp = two_exp * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t).expand().simplify()
        two_exp = two_exp.simplify()#.factorize()
        #self.assertEqual(two_exp.lorentz_ind, one_exp.lorentz_ind)

        P1_0,P1_1,P1_2,P1_3 = 1000, 3, 4, 1000
        P2_0,P2_1,P2_2,P2_3 = 1000, 3, 6, -1000
        P3_0,P3_1,P3_2,P3_3 = 2000, 2, 6, 9
        
        F1_1, F1_2, F1_3, F1_4  = 1, 62,34,23
        F2_1, F2_2, F2_3, F2_4  = 12, 44, 72, -45 
        OM1,OM2,OM3 = 0 , 0, 1.0 / 500**2
        M3 = 500
        
        #for name, cexpr in one_exp.contracted.items():
        #    exec('%s = %s' % (name, cexpr))
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            try:
                exec('%s = %s' % (name, cexpr))            
            except:
                pass
        for ind in one_exp.listindices():
            self.assertAlmostEqual(eval(str(one_exp.get_rep(ind))), eval(str(two_exp.get_rep(ind))))

    def test_short_aloha_FFT2(self):
        """ test the FFT2 creation of vertex"""

        FFT2 = self.Lorentz(name = 'FFT2',
                 spins = [2, 2, 5],
        structure="Metric(1003,2003)*ProjP(1,2)+Metric(1003,2003)*ProjM(1,2)"
        )
        abstract_FFT = create_aloha.AbstractRoutineBuilder(FFT2).compute_routine(3, factorize=False)
        expr = abstract_FFT.expr
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        F = aloha_obj.Spinor
        result = complex(0,1/3) * (OM(3) * P(-1, 3)**2 - 1) * (Metric('I2','I3') + 2 * OM(3) * P('I2',3)*P('I3',3))
        result = result * F(-2,1) * F(-2,2)
        
        zero = expr - result.expand()
        zero = zero.simplify()
        
        P1_0,P1_1,P1_2,P1_3 = 1000, 3, 4, 1000
        P2_0,P2_1,P2_2,P2_3 = 1000, 3, 6, -1000
        P3_0,P3_1,P3_2,P3_3 = 2000, 2, 6, 9
        
        F1_1, F1_2, F1_3, F1_4  = -44.7213595499958, 62,34,23
        F2_1, F2_2, F2_3, F2_4  = 12, 44, 72, -45 
        OM1,OM2,OM3 = 0 , 0, 1.0 / 500**2
        M3 = 500
        
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            exec('%s = %s' % (name, cexpr)) 
        
        for ind in zero.listindices():
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)
             
    def test_short_aloha_get_rank(self):
        """ test the FFV creation of vertex """
        
        FFV_4 = self.Lorentz(name = 'FFV_4',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')     

        abs = create_aloha.AbstractRoutineBuilder(FFV_4)
        routine = abs.compute_routine(2, ['L1'], factorize=False)
        rank = routine.get_info('rank')
        self.assertEqual(rank, 1)
        
        FFV_4 = self.Lorentz(name = 'FFV_4',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')     

        abs = create_aloha.AbstractRoutineBuilder(FFV_4)
        routine = abs.compute_routine(3, ['L1','P0'], factorize=False)
        rank_massless = routine.get_info('rank')
        routine = abs.compute_routine(3, ['L1'], factorize=False)
        rank_massive = routine.get_info('rank')
        self.assertEqual(rank_massive, 2)
        self.assertEqual(rank_massless, 0)

        UUT1 = self.Lorentz(name = 'UUT1',
           spins = [ -1, -1, 5 ],
               structure = 'P(1003,2)*P(2003,1) + P(1003,1)*P(2003,2) + P(-1,2)*P(-1,2)*Metric(1003,2003)')
        abs = create_aloha.AbstractRoutineBuilder(UUT1)
        routine = abs.compute_routine(1, ['L2'], factorize=False)
        rank = routine.get_info('rank')
        self.assertEqual(rank, 2) 
   
    def test_short_aloha_FFV(self):
        """ test the FFV creation of vertex """
        
        FFV_M = self.Lorentz(name = 'FFV_4',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')        
        
        FFV_P = self.Lorentz(name = 'FFV_5',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')
        
        FFV = self.Lorentz(name = 'FFV',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,2)')
        
        
        abstract_M = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3, factorize=False)       
        abstract_P = create_aloha.AbstractRoutineBuilder(FFV_P).compute_routine(3, factorize=False)       
        abstract = create_aloha.AbstractRoutineBuilder(FFV).compute_routine(3, factorize=False)
        

        F2_1, F2_2, F2_3, F2_4  = 1, 2, 3, 4
        F1_1, F1_2, F1_3, F1_4  = 5, 6, 7, 8
        OM3 = 0
        j = complex(0,1)
        P3_0,P3_1,P3_2,P3_3 = 10, 11, 12, 13
        
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            exec('%s = %s' % (name, cexpr)) 
        
        for ind in abstract.expr.listindices():                        
            self.assertAlmostEqual(eval(str(abstract.expr.get_rep(ind))) -
                             eval(str(abstract_M.expr.get_rep(ind))) -
                             eval(str(abstract_P.expr.get_rep(ind)))
                             ,0)
        zero = abstract_M.expr + abstract_P.expr - abstract.expr
        zero.simplify()
        for ind in abstract.expr.listindices():
            self.assertEqual(eval(str(zero.get_rep(ind))),0,'fail')

    def test_short_aloha_FFV_MG4(self):
        """ test the FFV creation of vertex against MG4 """ 
        
        aloha_lib.KERNEL.clean()
        self.assertEqual(len(aloha_lib.KERNEL), 0)
        
        FFV_M = self.Lorentz(name = 'FFV_4',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')        
        
        FFV_P = self.Lorentz(name = 'FFV_5',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')
          
        abstract_M = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3, factorize=False)       
          
        F2_1, F2_2, F2_3, F2_4  = 1, 2, 3, 4
        F1_1, F1_2, F1_3, F1_4  = 5, 6, 7, 8
        OM3 = 0
        j = complex(0,1)
        P3_0,P3_1,P3_2,P3_3 = 10, 11, 12, 13
        
          
        #tested solution again MG4
        s1 = -j*((OM3*(P3_0*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+((F1_3*F2_1)+(F1_4*F2_2)))        
        s2 = -j*((OM3*(P3_1*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-(F1_4*F2_1)-(F1_3*F2_2)))
        s3 = -j*((OM3*(P3_2*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-1*j*(F1_4*F2_1)+1*j*(F1_3*F2_2)))
        s4 = -j*((OM3*(P3_3*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-(F1_3*F2_1)+(F1_4*F2_2)))

        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            exec('%s = %s' % (name, cexpr)) 

        self.assertEqual(s1, eval(str(abstract_M.expr.get_rep([0]))))
        self.assertEqual(s2, eval(str(abstract_M.expr.get_rep([1]))))    
        self.assertEqual(s3, eval(str(abstract_M.expr.get_rep([2]))))    
        self.assertEqual(s4, eval(str(abstract_M.expr.get_rep([3]))))                                   

        FFV_6 = self.Lorentz(name = 'FFV_6',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')
        
        
        abstract_P = create_aloha.AbstractRoutineBuilder(FFV_P).compute_routine(3, factorize=False)       

        abstract_6 = create_aloha.AbstractRoutineBuilder(FFV_6).compute_routine(3, factorize=False)
         
        zero = abstract_6.expr - abstract_M.expr - \
                                                    2* abstract_P.expr   
        for name, cexpr in aloha_lib.KERNEL.reduced_expr2.items():
            exec('%s = %s' % (name, cexpr)) 
        for ind in zero.listindices():
            self.assertEqual(eval(str(zero.get_rep(ind))),0)
        
    def test_short_aloha_symmetries_and_get_info(self):
        """ test that the symmetries of particles works """
    
        # Check that full identification symmetry works
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        helas_suite.look_for_symmetries()
        solution = {'VVVV2': {2: 1 ,4: 3}, 'SSS1': {2: 1, 3: 2}, 'VVSS1': {2: 1, 4: 3}, 'VVS1': {2: 1},'SSSS1': {2: 1, 3: 2, 4: 3}}  
        self.assertEqual(solution, helas_suite.symmetries)
        
        # check that the get_info work
        
        start = time.time()
        rank = helas_suite.get_info('rank', 'VVVV2', 2, ['L1', 'P0'], cached=True)
        time1 = time.time() - start # time1 is expected to be O(1e-2)
        self.assertEqual(rank, 0)
        
        start = time.time()
        rank = helas_suite.get_info('rank', 'VVVV2', 2, ['L1', 'P0'])
        time2 = time.time() - start # time2 is expected to be O(1e-6)
        
        self.assertEqual(rank, 0)
        self.assertTrue(100 * time2 < time1) # if this is not the case this is
                                             # clearly wrong.
        
        
        # check for correct behavior if wrong input:
        # 1) check that it fail for non loop routine
        self.assertRaises(AssertionError, helas_suite.get_info, 'rank', 'VVVV2', 0, []) 
        # 2) check that unknow information fails.
        self.assertRaises(create_aloha.ALOHAERROR, helas_suite.get_info, 'SW', 'VVVV2', 2, ['L1'])
        # 3) check that appropriate error is raise for invalid input
        self.assertRaises(AssertionError, helas_suite.get_info, 'rank', 'VVVV2', 1, ['L1'])
        self.assertRaises(AssertionError, helas_suite.get_info, 'rank', 'VVVV2', 0, ['L1'])
        
        
    def test_short_has_symmetries(self):
        """Check that functions returning symmetries works"""
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        helas_suite.look_for_symmetries()
        
        base = helas_suite.has_symmetries('SSS1', 3)
        self.assertEqual(base, 1)

        base = helas_suite.has_symmetries('SSS1', 3, valid_output=(1, 2))
        self.assertEqual(base, 1)
        
        base = helas_suite.has_symmetries('SSS1', 3, valid_output=(1,))
        self.assertEqual(base, 1)
        
        base = helas_suite.has_symmetries('SSS1', 3, valid_output=(2,))
        self.assertEqual(base, 2)   
        
        base = helas_suite.has_symmetries('VVS1', 3, valid_output=(3,))
        self.assertEqual(base, None)
        
        base = helas_suite.has_symmetries('VVS1', 3, valid_output=(1, 2))
        self.assertEqual(base, None)   

    def test_short_aloha_multiple_lorentz(self):
        """ check if the detection of multiple lorentz work """
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        helas_suite.look_for_multiple_lorentz_interactions()
        solution = {'FFV2': [('FFV3',), ('FFV4',), ('FFV5',)], 'FFS1': [('FFS3',)]}
        self.assertEqual(solution, helas_suite.multiple_lor)
        

    def test_short_aloha_multiple_lorentz_and_symmetry(self):
        """ check if the detection of multiple lorentz work """
        
        aloha_lib.KERNEL.clean()
        VVS1 = self.Lorentz(name = 'VVS1',
                 spins = [ 3, 3, 1 ],
                 structure = 'Metric(1,2)')

        #VVS2 = self.Lorentz(name = 'VVS2',
        #         spins = [ 3, 3, 1 ],
        #         structure = 'Metric(2,1)')
        
        abstract = create_aloha.AbstractRoutineBuilder(VVS1).compute_routine(1)
        abstract.add_symmetry(2)
        abstract.add_combine(('VVS2',))
        
        text =  abstract.write(None, 'Fortran')

        goal = """subroutine VVS1_1(V2, S3, COUP, M1, W1,V1)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 V2(*)
 complex*16 S3(*)
 real*8 P1(0:3)
 real*8 M1
 complex*16 TMP0
 real*8 W1
 complex*16 denom
 real*8 OM1
 complex*16 COUP
 complex*16 V1(6)
entry VVS1_2(V2, S3, COUP, M1, W1,V1)

    OM1 = 0d0
    if (M1.ne.0d0) OM1=1d0/M1**2
    V1(1) = +V2(1)+S3(1)
    V1(2) = +V2(2)+S3(2)
P1(0) = -dble(V1(1))
P1(1) = -dble(V1(2))
P1(2) = -dimag(V1(2))
P1(3) = -dimag(V1(1))
 TMP0 = (V2(3)*P1(0)-V2(4)*P1(1)-V2(5)*P1(2)-V2(6)*P1(3))
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI* W1))
    V1(3)= denom*S3(3)*(-CI*(V2(3))+CI*(P1(0)*OM1*TMP0))
    V1(4)= denom*S3(3)*(-CI*(V2(4))+CI*(P1(1)*OM1*TMP0))
    V1(5)= denom*S3(3)*(-CI*(V2(5))+CI*(P1(2)*OM1*TMP0))
    V1(6)= denom*S3(3)*(-CI*(V2(6))+CI*(P1(3)*OM1*TMP0))
end



subroutine VVS1_2_1(V2, S3, COUP1, COUP2, M1, W1,V1)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 V2(*)
 complex*16 COUP2
 complex*16 S3(*)
 real*8 P1(0:3)
 real*8 M1
 real*8 W1
 complex*16 COUP1
 complex*16 denom
 integer*4 i
 complex*16 Vtmp(6)
 real*8 OM1
 complex*16 V1(6)
entry VVS1_2_2(V2, S3, COUP1, COUP2, M1, W1,V1)

    call VVS1_1(V2,S3,COUP1,M1,W1,V1)
    call VVS2_1(V2,S3,COUP2,M1,W1,Vtmp)
 do i = 3, 6
        V1(i) = V1(i) + Vtmp(i)
 enddo
end

"""
        self.assertEqual(text.split('\n'),goal.split('\n')) 
        text_h, text_cpp =  abstract.write(None, 'CPP')
    
        goal_h = """#ifndef VVS1_1_guard
#define VVS1_1_guard
#include <complex>
using namespace std;

void VVS1_1(complex<double> V2[], complex<double> S3[], complex<double> COUP, double M1, double W1,complex<double> V1[]);
void VVS1_2(complex<double> V2[], complex<double> S3[], complex<double> COUP, double M1, double W1,complex<double> V1[]);
#endif

#ifndef VVS1_2_1_guard
#define VVS1_2_1_guard
#include <complex>
using namespace std;

void VVS1_2_1(complex<double> V2[], complex<double> S3[], complex<double> COUP1, complex<double> COUP2, double M1, double W1,complex<double> V1[]);
void VVS1_2_2(complex<double> V2[], complex<double> S3[], complex<double> COUP1, complex<double> COUP2, double M1, double W1,complex<double> V1[]);
#endif

"""
        goal_cpp = """#include "VVS1_1.h"

void VVS1_1(complex<double> V2[], complex<double> S3[], complex<double> COUP, double M1, double W1,complex<double> V1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  TMP0;
 complex<double>  denom;
 double  OM1;
    OM1 = 0.;
    if (M1 != 0.)
 OM1=1./pow(M1,2);
    V1[0] = +V2[0]+S3[0];
    V1[1] = +V2[1]+S3[1];
P1[0] = -V1[0].real();
P1[1] = -V1[1].real();
P1[2] = -V1[1].imag();
P1[3] = -V1[0].imag();
 TMP0 = (V2[2]*P1[0]-V2[3]*P1[1]-V2[4]*P1[2]-V2[5]*P1[3]);
    denom = COUP/(pow(P1[0],2)-pow(P1[1],2)-pow(P1[2],2)-pow(P1[3],2) - M1 * (M1 -cI* W1));
    V1[2]= denom*S3[2]*(-cI*(V2[2])+cI*(P1[0]*OM1*TMP0));
    V1[3]= denom*S3[2]*(-cI*(V2[3])+cI*(P1[1]*OM1*TMP0));
    V1[4]= denom*S3[2]*(-cI*(V2[4])+cI*(P1[2]*OM1*TMP0));
    V1[5]= denom*S3[2]*(-cI*(V2[5])+cI*(P1[3]*OM1*TMP0));
}

void VVS1_2(complex<double> V2[], complex<double> S3[], complex<double> COUP, double M1, double W1,complex<double> V1[])
{

 VVS1_1(V2,S3,COUP,M1,W1,V1);
}
void VVS1_2_1(complex<double> V2[], complex<double> S3[], complex<double> COUP1, complex<double> COUP2, double M1, double W1,complex<double> V1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  denom;
 int i;
 complex<double>  Vtmp[6];
 double  OM1;
    VVS1_1(V2,S3,COUP1,M1,W1,V1);
    VVS2_1(V2,S3,COUP2,M1,W1,Vtmp);
 i= 2;
while (i < 6)
{
 V1[i] = V1[i] + Vtmp[i];
 i++;
}
}
void VVS1_2_2(complex<double> V2[], complex<double> S3[], complex<double> COUP1, complex<double> COUP2, double M1, double W1,complex<double> V1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  denom;
 int i;
 complex<double>  Vtmp[6];
 double  OM1;
    VVS1_1(V2,S3,COUP1,M1,W1,V1);
    VVS2_1(V2,S3,COUP2,M1,W1,Vtmp);
 i= 2;
while (i < 6)
{
 V1[i] = V1[i] + Vtmp[i];
 i++;
}
}
"""
        self.assertEqual(text_h.split('\n'),goal_h.split('\n'))
        self.assertEqual(text_cpp.split('\n'),goal_cpp.split('\n'))
        
        
        text =  abstract.write(None, 'Python')

        goal = """import cmath
import wavefunctions
def VVS1_1(V2,S3,COUP,M1,W1):
    OM1 = 0.0
    if (M1): OM1=1.0/M1**2
    V1 = wavefunctions.WaveFunction(size=6)
    V1[0] = +V2[0]+S3[0]
    V1[1] = +V2[1]+S3[1]
    P1 = [-complex(V1[0]).real, -complex(V1[1]).real, -complex(V1[1]).imag, -complex(V1[0]).imag]
    TMP0 = (V2[2]*P1[0]-V2[3]*P1[1]-V2[4]*P1[2]-V2[5]*P1[3])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1[2]= denom*S3[2]*(-1j*(V2[2])+1j*(P1[0]*OM1*TMP0))
    V1[3]= denom*S3[2]*(-1j*(V2[3])+1j*(P1[1]*OM1*TMP0))
    V1[4]= denom*S3[2]*(-1j*(V2[4])+1j*(P1[2]*OM1*TMP0))
    V1[5]= denom*S3[2]*(-1j*(V2[5])+1j*(P1[3]*OM1*TMP0))
    return V1


import cmath
import wavefunctions
def VVS1_2(V2,S3,COUP,M1,W1):

    return VVS1_1(V2,S3,COUP,M1,W1)
import cmath
import wavefunctions
def VVS1_2_1(V2,S3,COUP1,COUP2,M1,W1):
    V1 = VVS1_1(V2,S3,COUP1,M1,W1)
    tmp = VVS2_1(V2,S3,COUP2,M1,W1)
    for i in range(2,6):
        V1[i] += tmp[i]
    return V1

import cmath
import wavefunctions
def VVS1_2_2(V2,S3,COUP1,COUP2,M1,W1):
    V1 = VVS1_1(V2,S3,COUP1,M1,W1)
    tmp = VVS2_1(V2,S3,COUP2,M1,W1)
    for i in range(2,6):
        V1[i] += tmp[i]
    return V1

"""
        self.assertEqual(text.split('\n'),goal.split('\n'))
        
        
    def test_short_full_sm_aloha(self):
        """test that the full SM seems to work"""
        # Note that this test check also some of the routine define inside this
        #because of use of some global.
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        self.assertEqual(helas_suite.look_for_conjugate(), {})
        start = time.time()
        helas_suite.compute_all()
        timing = time.time()-start
        if timing > 5:
            print "WARNING ALOHA SLOW (taking %s s for the full sm)" % timing
        lorentz_index = {1:0, 2:0,3:1}
        spin_index = {1:0, 2:1, 3:0}
        error = 'wrong contraction for %s'
        for (name, output_part), abstract in helas_suite.items():
            if not output_part:
                self.assertEqual(abstract.expr.nb_lor, 0, error % name)
                self.assertEqual(abstract.expr.nb_spin, 0, error % abstract.expr.spin_ind)
                continue
            helas = self.find_helas(name, helas_suite.model)
            lorentz_solution = lorentz_index[helas.spins[output_part -1]]
            self.assertEqual(abstract.expr.nb_lor, lorentz_solution)
            spin_solution = spin_index[helas.spins[output_part -1]]
            self.assertEqual(abstract.expr.nb_spin, spin_solution, \
                             error % name)
            
    def test_short_multiple_lorentz_subset(self):
        """test if we create the correct set of routine/files for multiple lorentz"""
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        requested_routines=[(('FFV1',) , (), 0), 
                            (('FFV1','FFV2') , ('C1',), 0)]
        
        helas_suite.compute_subset(requested_routines)

        # Check that the 3 base routines are created
        # FFV1, FFV1C1, FFV2C1
        self.assertEqual(len(helas_suite), 3)
        
        # Check that FFV1C1 are correctly connected to the associate
        # lorentz
        linked = helas_suite[('FFV1C1',0)].combined
        self.assertEqual(linked, [('FFV2',)])        
        linked = helas_suite[('FFV1',0)].combined
        self.assertEqual(linked, [])
        
        # Check that the file are correctly written
        with misc.TMP_directory(prefix='mg5') as path:
            helas_suite.write(path, 'Fortran')
            
            content = set(os.listdir(path))
            self.assertEqual(content, set(['FFV1_0.f',
                                           'FFV1C1_0.f','FFV2C1_0.f']))
            
            # Check the content of FFV1__FFV2C1_0.f
            fsock = open('%s/FFV1C1_0.f' % path)
            goal = """
          SUBROUTINE FFV1_2C1_0(F2, F1, V3, COUP1, COUP2,VERTEX)
          IMPLICIT NONE
          COMPLEX*16 F1(*)
          COMPLEX*16 F2(*)
          COMPLEX*16 V3(*)
          COMPLEX*16 COUP1
          COMPLEX*16 COUP2
          COMPLEX*16 VERTEX
          COMPLEX*16 TMP
          CALL FFV1C1_0(F2,F1,V3,COUP1,VERTEX)
          CALL FFV2C1_0(F2,F1,V3,COUP2,TMP)
          VERTEX = VERTEX + TMP
          END"""
    
            data = [ l.strip() for l in fsock.read().split('\n')]
            for line in goal.split('\n'):
                    self.assertTrue(line.strip() in data)
        
        
        
    
    def test_short_mssm_subset_creation(self):
        """ test the creation of subpart of ALOHA routines 
        including clash routines """
        helas_suite = create_aloha.AbstractALOHAModel('mssm')
        
        requested_routines=[(('FFV1',) , (), 0), 
                            (('FFV1',), (), 2),
                            (('FFV1',), ('C1',), 0),
                            (('FFV2',), ('C1',), 3),
                            (('VVV1',), (), 3)]
        
        helas_suite.compute_subset(requested_routines)        
        self.assertEqual(len(helas_suite), 5)
        
        # Apply basic check for coherence
        error = 'wrong contraction for %s'
        for (name, output_part), abstract in helas_suite.items():
            if not output_part:
                self.assertEqual(abstract.expr.nb_lor, 0, error % name)
                self.assertEqual(abstract.expr.nb_spin, 0, error % abstract.expr.spin_ind)
            elif name in ['FFV2C1','VVV1']:
                self.assertEqual(abstract.expr.nb_lor, 1, error % name)
                self.assertEqual(abstract.expr.nb_spin, 0, error % name)
            elif name in ['FFV1']:
                self.assertEqual(abstract.expr.nb_lor, 0, error % name)
                self.assertEqual(abstract.expr.nb_spin, 1, error % name)
            else:
                raise Exception, 'not expected routine %s' % name
            
    def find_helas(self, name, model):
        for lorentz in model.all_lorentz:
            if lorentz.name == name:
                return lorentz
            
        raise Exception('the test is confuse by name %s' % name)     

    def test_short_aloha_FFVC(self):
        """ test the FFV creation of vertex """
        from models.mssm.object_library import Lorentz

        FFV = Lorentz(name = 'FFV',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,2)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        conjg_builder= builder.define_conjugate_builder()
        amp = builder.compute_routine(0)
        conjg_amp = conjg_builder.compute_routine(0)
        
        # Check correct contraction
        self.assertEqual(conjg_amp.expr.nb_lor, 0)
        self.assertEqual(conjg_amp.expr.nb_spin, 0)
      
        # Check expr are different
        self.assertNotEqual(str(amp.expr), str(conjg_amp.expr))
        self.assertEqual(amp.name, conjg_amp.name)
        self.assertEqual(amp.tag + ['C1'], conjg_amp.tag)
        
    def test_short_aloha_expr_FFFF3(self):
        """Test analytical expression for four fermion (provide by Tim M).
        it's particularity is about to have contraction A and 4*A """
        
        from models.mssm.object_library import Lorentz
        FFFF = Lorentz(name = 'FFFF3',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,3)*Gamma(-2,-3,1)*Gamma(-1,2,-3)*Gamma(-1,4,-4) - Gamma(-1,-2,3)*Gamma(-1,4,-2)*Identity(1,2) - Gamma(-1,-2,1)*Gamma(-1,2,-2)*Identity(3,4) + 4*Identity(1,2)*Identity(3,4)')
        builder = create_aloha.AbstractRoutineBuilder(FFFF)
        conjg_builder= builder.define_conjugate_builder()
        amp = conjg_builder.compute_routine(2)


        F1_1, F1_2, F1_3, F1_4 = 1,2,3,4
        F2_1, F2_2, F2_3, F2_4 = 5,5,6,7
        F3_1, F3_2, F3_3, F3_4 = 10,20,32,44
        F4_1, F4_2, F4_3, F4_4 = 53,56,65,76
        P1_0, P1_1, P1_2, P1_3 = 1,2,3,6
        M1 = P1_0**2 - P1_1**2 - P1_2**2 - P1_3**2
        # For V4:
        cImag = complex(0,1)

        for name, expr in amp.contracted.items():
            exec('%s = %s' % (name,expr))       

        for i in range(100):
            ufo_value = [eval(str(amp.expr.get_rep([i]))) for i in range(4)]

        #computed with 1.4.8.4 // 1.5.3 // 1.5.4
        solution = [(-518016-1383424j), (317568-1604608j), (162600-4898488j), (-31800-8538056j)]
        for out,sol in zip(ufo_value,solution):
            self.assertAlmostEqual(out, sol)

    def test_short_aloha_expr_VVS1(self):
        """Test analytical expression for VVS from SILH. 
        This checks that P(-1,1)**2 is correct."""
        
        
        aloha_lib.KERNEL.clean()
        from models.mssm.object_library import Lorentz
        VVS1 = Lorentz(name = 'VVS1',
                 spins = [ 3, 3, 1 ],
                 structure = 'P(1,1)*P(2,1) - P(-1,1)**2*Metric(1,2)')
        builder = create_aloha.AbstractRoutineBuilder(VVS1)
        amp = builder.compute_routine(0)

        solution = """subroutine VVS1_0(V1, V2, S3, COUP,vertex)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 V2(*)
 complex*16 TMP2
 complex*16 S3(*)
 complex*16 TMP1
 real*8 P1(0:3)
 complex*16 TMP0
 complex*16 vertex
 complex*16 COUP
 complex*16 V1(*)
 complex*16 TMP3
P1(0) = dble(V1(1))
P1(1) = dble(V1(2))
P1(2) = dimag(V1(2))
P1(3) = dimag(V1(1))
TMP1 = (P1(0)*V1(3)-P1(1)*V1(4)-P1(2)*V1(5)-P1(3)*V1(6))
TMP0 = (V2(3)*P1(0)-V2(4)*P1(1)-V2(5)*P1(2)-V2(6)*P1(3))
TMP3 = (P1(0)*P1(0)-P1(1)*P1(1)-P1(2)*P1(2)-P1(3)*P1(3))
TMP2 = (V2(3)*V1(3)-V2(4)*V1(4)-V2(5)*V1(5)-V2(6)*V1(6))
vertex = COUP*S3(3)*(-CI*(TMP0*TMP1)+CI*(TMP2*TMP3))
end


"""
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = [l.strip() for l in solution.split('\n')]
        split_routine = [l.strip() for l in routine.split('\n')]
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))


        V1_1, V1_2, V1_3, V1_4 = 1,2,3,4
        V2_1, V2_2, V2_3, V2_4 = 5,5,6,7
        S3_1, S3_2, S3_3, S3_4 = 10,20,32,44
        P1_0, P1_1, P1_2, P1_3 = 1,2,3,6
        M1 = P1_0**2 - P1_1**2 - P1_2**2 - P1_3**2
        # For V4:
        cImag = complex(0,1)

        for name, expr in amp.contracted.items():
            exec('%s = %s' % (name,expr))       


        ufo_value = eval(str(amp.expr.get_rep([0])))
        self.assertAlmostEqual(ufo_value, 1080j)


        
    def test_short_aloha_expr_FFV2C1(self):
        """Test analytical expression for fermion clash routine"""
        
        from models.mssm.object_library import Lorentz
        FFV = Lorentz(name = 'FFV2',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,\'s1\')*ProjM(\'s1\',1)')
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        conjg_builder= builder.define_conjugate_builder()
        amp = conjg_builder.compute_routine(0, factorize=False)

        self.assertEqual(amp.expr.nb_spin, 0)
        F1_1, F1_2, F1_3, F1_4 = 1,2,3,4
        F2_1, F2_2, F2_3, F2_4 = 5,5,6,7
        V3_1, V3_2, V3_3, V3_4 = 8,9,10,11
        # For V4:
        cImag = complex(0,1)

        for name, expr in amp.contracted.items():
            exec('%s = %s' % (name,expr))
            
        ufo_value = eval(str(amp.expr.get_rep([0])))
    
        #v4_value = ( (F2_1*F1_3+F2_2*F1_4)*V3_1 \
        #            -(F2_1*F1_4+F2_2*F1_3)*V3_2 \
        #            +(F2_1*F1_4-F2_2*F1_3)*V3_3*cImag \
        #            -(F2_1*F1_3-F2_2*F1_4)*V3_4       )
        v4_value = ( (F1_1*F2_3+F1_2*F2_4)*V3_1 \
                    -(F1_1*F2_4+F1_2*F2_3)*V3_2 \
                    +(F1_1*F2_4-F1_2*F2_3)*V3_3*cImag \
                    -(F1_1*F2_3-F1_2*F2_4)*V3_4       )

        self.assertEqual(complex(0,-1)*ufo_value, v4_value)

        FFV = Lorentz(name = 'FFV2',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,\'s1\')*ProjP(\'s1\',1)')
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        conjg_builder= builder.define_conjugate_builder()
        amp = conjg_builder.compute_routine(0, factorize=False)
        for name, expr in amp.contracted.items():
            exec('%s = %s' % (name,expr))
                    
        ufo_value = eval(str(amp.expr.get_rep([0])))
        self.assertNotEqual(complex(0,1)*ufo_value, v4_value)
        v4_value = (F1_3*F2_1+F1_4*F2_2)*V3_1 \
                          +(F1_3*F2_2+F1_4*F2_1)*V3_2 \
                          -(F1_3*F2_2-F1_4*F2_1)*V3_3*cImag \
                          +(F1_3*F2_1-F1_4*F2_2)*V3_4
               
        self.assertEqual(complex(0,-1)*ufo_value, v4_value)
        
    def test_short_aloha_expr_FFFF(self):
        """Test analytical expression for fermion clash routine"""
        
        from models.mssm.object_library import Lorentz
        FFFF = Lorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Identity(2,1)*Identity(4,3)')
        
        builder = create_aloha.AbstractRoutineBuilder(FFFF)
        conjg_builder= builder.define_conjugate_builder()
        conjg_builder= conjg_builder.define_conjugate_builder(pairs=2)
        amp = conjg_builder.compute_routine(0)

        self.assertEqual(builder.conjg,[])
        self.assertEqual(amp.expr.nb_spin, 0)
        self.assertEqual(amp.expr.nb_lor, 0)

        conjg_builder= builder.define_conjugate_builder(pairs=1)
        amp = conjg_builder.compute_routine(0)

        self.assertEqual(amp.expr.nb_spin, 0)
        self.assertEqual(amp.expr.nb_lor, 0)   
        
        conjg_builder= builder.define_conjugate_builder(pairs=2)
        amp = conjg_builder.compute_routine(0)

        self.assertEqual(amp.expr.nb_spin, 0)
        self.assertEqual(amp.expr.nb_lor, 0)        

        

class UFOLorentz(object):
    """ simple UFO LORENTZ OBJECT """
    
    def __init__(self, name='',spins=[],structure='1'):
        """fake lorentz initialization"""
        self.name = name
        self.spins=spins
        self.structure = structure
        
class AbstractRoutineBuilder(create_aloha.AbstractRoutineBuilder):
    
    
    def compute_routine(self, mode):
        """avoid computation"""
        self.outgoing = mode
        self.expr = aloha_obj.C(1,2)
        self.expr.tag=[]
        return self.define_simple_output()

class TestAlohaWriter(unittest.TestCase):
    """ simple unittest of the writer more test are in test_export_v4
    and test_export_pythia"""
    
    def old_test_reorder_call_listFFVV(self):
        
        FFVV = UFOLorentz(name = 'FFVV',
               spins = [ 2, 2, 3, 3])
        
        abstract = AbstractRoutineBuilder(FFVV).compute_routine(1)
        abstract.add_symmetry(2)
        
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        new_call = writer.reorder_call_list(call_list, 1, 2)
        self.assertEqual(['F2', 'V3', 'V4'], new_call)

    def old_test_reorder_call_listFVVV(self):
        FVVV = UFOLorentz(name = 'FVVV',
               spins = [ 2, 3, 3, 3])
        
        abstract = AbstractRoutineBuilder(FVVV).compute_routine(2)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        self.assertEqual(['F1', 'V3', 'V4'], call_list)
        #vertex UAAW
        #vertex_3 receives UAW with label 134
        #vertex_2 expects UAW => need label 134 
        new_call = writer.reorder_call_list(call_list, 2, 3)
        self.assertEqual(['F1', 'V3', 'V4'], new_call)
        
        #vertex UAWA
        #vertex_4 receives UAW with label 134 
        #vertex_2 expects UWA => need label 143
        new_call = writer.reorder_call_list(call_list, 2, 4)
        self.assertEqual(['F1', 'V4', 'V3'], new_call)                  
    
    def old_test_reorder_call_listVVVV(self):
        VVVV = UFOLorentz(name = 'VVVV',
               spins = [ 3, 3, 3, 3])
    
            
        abstract = AbstractRoutineBuilder(VVVV).compute_routine(1)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        # Vertex AAW+W-
        # vertex_2 receives AW+W- with label 234
        # vertex_1 ask for AW+W- so should be label 234
        
        new_call = writer.reorder_call_list(call_list, 1, 2)
        self.assertEqual(['V2', 'V3', 'V4'], new_call)
        
        # Vertex Aw+AW-
        #vertex_3 receives AW+W-  with label 234
        #vertex_1 ask for w+Aw- so should be call with 324
        new_call = writer.reorder_call_list(call_list, 1, 3)
        self.assertEqual(['V3', 'V2', 'V4'], new_call) 
        # Vertex Aw+w-A
        #vertex_4 receives Aw+w-  with label 234
        #vertex_1 ask for w+w-A so should be call with 342        
        new_call = writer.reorder_call_list(call_list, 1, 4)
        self.assertEqual(['V3', 'V4', 'V2'], new_call)        
        
        abstract = create_aloha.AbstractRoutineBuilder(VVVV).compute_routine(2)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        self.assertEqual(['V1', 'V3', 'V4'], call_list)
        # Vertex W+AAW-
        # vertex3 receives W+AW- with label 134
        # vertex2 ask for W+AW- so we should use label 134
        new_call = writer.reorder_call_list(call_list, 2, 3)
        self.assertEqual(['V1', 'V3', 'V4'], new_call)
        # Vertex W+AW-A
        # vertex4 receives W+AW- with label 134
        # vertex2 ask for W+W-A so we should use label 143        
        new_call = writer.reorder_call_list(call_list, 2, 4)
        self.assertEqual(['V1', 'V4', 'V3'], new_call)

        abstract = create_aloha.AbstractRoutineBuilder(VVVV).compute_routine(3)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        self.assertEqual(['V1', 'V2', 'V4'], call_list)
        # Vertex W+W-AA
        # vertex4 receives W+W-A with label 124
        # vertex3 ask for W+W-A so we should use label 124
        new_call = writer.reorder_call_list(call_list, 3, 4)
        self.assertEqual(['V1', 'V2', 'V4'], new_call)

    def old_test_reorder_call_listUVVS(self):
        UVVS = UFOLorentz(name = 'UVVS',
               spins = [ 2, 3, 3, 1])
    
        
        abstract = AbstractRoutineBuilder(UVVS).compute_routine(2)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        # Vertex UAAH
        # vertex_3 receives UAH with label 134
        # vertex_2 ask for UAH so should be label 134
        
        new_call = writer.reorder_call_list(call_list, 2, 3)
        self.assertEqual(['F1', 'V3', 'S4'], new_call)
        
        UVVS = UFOLorentz(name = 'UVVS',
               spins = [ 2, 3, 3, 1])
    
        
        abstract = AbstractRoutineBuilder(UVVS).compute_routine(2)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        call_list= writer.calllist['CallList']
        # Vertex UAAH
        # vertex_3 receives UAH with label 134
        # vertex_2 ask for UAH so should be label 134
        
        new_call = writer.reorder_call_list(call_list, 2, 3)
        self.assertEqual(['F1', 'V3', 'S4'], new_call)        
    
    
    def test_short_change_number_format_fortran(self):
        """ Check that the number are correctly written in fortranwriter """
        
        SSS = UFOLorentz(name = 'SSS',
               spins = [ 1, 1, 1])
    
        
        abstract = AbstractRoutineBuilder(SSS).compute_routine(0)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        
        numbers = [complex(0,1), complex(0,1/2), 3*complex(1.0,3), complex(1,0)]
        numbers +=[0, 1, 2, -3, 3.0, 3.00, 1.01, 2000, 1/3, 1/4, 3/4, math.pi,
                   100*math.pi]
 
        solution = ['CI', '1d0/2d0 * CI', '(3d0 + 9d0*CI)', '1d0', '0d0', '1d0', '2d0', '-3d0', '3d0', '3d0', '101d0/100d0', '2000d0', '1d0/3d0', '1d0/4d0', '3d0/4d0','3.14159265359d0', '314.159265359d0']
#        converted = [writer.change_number_format(number) for number in numbers]
        for i, number in enumerate(numbers):
            value = writer.change_number_format(number)
            self.assertEqual(value, solution[i])
        #map(self.assertEqual, converted, solution)
 
    def test_short_change_number_format_python(self):
        """ Check that the number are correctly written in fortranwriter """
        
        SSS = UFOLorentz(name = 'SSS',
               spins = [ 1, 1, 1])
    
        
        abstract = AbstractRoutineBuilder(SSS).compute_routine(0)
        writer = aloha_writers.ALOHAWriterForPython(abstract, '/tmp')
        
        numbers = [complex(0,1), complex(0,1/2), 3*complex(1.0,3), complex(1,0)]
        numbers +=[0, 1, 2, -3, 3.0, 3.00, 1.01, 2000, 1/3, 1/4, 3/4, math.pi, 1.001,
                   100*math.pi]
 
        solution = ['1j', '1j/2', '(3+9j)', '1', '0', '1', '2', '-3', '3', '3', '101/100', '2000', '1/3', '1/4', '3/4','3.14159265359','1.001','314.159265359']
#        converted = [writer.change_number_format(number) for number in numbers]
        for i, number in enumerate(numbers):
            value = writer.change_number_format(number)
            self.assertEqual(value, solution[i]) 

    def test_short_change_number_format_cpp(self):
        """ Check that the number are correctly written in fortranwriter """
        
        SSS = UFOLorentz(name = 'SSS',
               spins = [ 1, 1, 1])
    
        
        abstract = AbstractRoutineBuilder(SSS).compute_routine(0)
        writer = aloha_writers.ALOHAWriterForCPP(abstract, '/tmp')
        
        numbers = [complex(0,1), complex(0,1/2), 3*complex(1.0,3), complex(1,0)]
        numbers +=[0, 1, 2, -3, 3.0, 3.00, 1.01, 2000, 1/3, 1/4, 3/4, math.pi]
 
        solution = ['cI', '1./2. * cI', '(3. + 9.*cI)', '1.', '0.', '1.', '2.', '-3.', '3.', '3.', '101./100.', '2000.', '1./3.', '1./4.', '3./4.', '3.141592654']
#        converted = [writer.change_number_format(number) for number in numbers]
        for i, number in enumerate(numbers):
            value = writer.change_number_format(number)
            self.assertEqual(value, solution[i]) 
 
    def test_short_pythonwriter(self):
        """ test that python writer works """
        
        solution ="""import cmath
import wavefunctions
def SSS1_1(S2,S3,COUP,M1,W1):
    S1 = wavefunctions.WaveFunction(size=3)
    S1[0] = +S2[0]+S3[0]
    S1[1] = +S2[1]+S3[1]
    P1 = [-complex(S1[0]).real, -complex(S1[1]).real, -complex(S1[1]).imag, -complex(S1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    S1[2]= denom*1j * S3[2]*S2[2]
    return S1


import cmath
import wavefunctions
def SSS1_2(S2,S3,COUP,M1,W1):

    return SSS1_1(S2,S3,COUP,M1,W1)
import cmath
import wavefunctions
def SSS1_3(S2,S3,COUP,M1,W1):

    return SSS1_1(S2,S3,COUP,M1,W1)
"""
        
        SSS = UFOLorentz(name = 'SSS1',
                 spins = [ 1, 1, 1 ],
                 structure = '1')        
        builder = create_aloha.AbstractRoutineBuilder(SSS)
        amp = builder.compute_routine(1)
        amp.add_symmetry(2)
        amp.add_symmetry(3)
        
        routine = amp.write(output_dir=None, language='Python')

        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

    
    @set_global()
    def test_short_pythonwriter_spin3half(self):
        """ test that python writer works """


        solution ="""import cmath
import wavefunctions
def RFSC1_1(R1,S3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = +R1[0]+S3[0]
    F2[1] = +R1[1]+S3[1]
    P2 = [-complex(F2[0]).real, -complex(F2[1]).real, -complex(F2[1]).imag, -complex(F2[0]).imag]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*1j * S3[2]*(P2[0]*-1*(R1[7]+R1[14]+1j*(R1[11])-R1[2])+(P2[1]*(R1[3]+R1[15]+1j*(R1[10])-R1[6])+(P2[2]*(-1j*(R1[6])+1j*(R1[3]+R1[15])-R1[10])-P2[3]*(R1[7]+R1[14]+1j*(R1[11])-R1[2]))))
    F2[3]= denom*1j * S3[2]*(P2[0]*(R1[3]+R1[15]+1j*(R1[10])-R1[6])+(P2[1]*-1*(R1[7]+R1[14]+1j*(R1[11])-R1[2])+(P2[2]*(-1j*(R1[2])+1j*(R1[7]+R1[14])-R1[11])-P2[3]*(R1[3]+R1[15]+1j*(R1[10])-R1[6]))))
    F2[4]= denom*1j * M2*S3[2]*(R1[7]+R1[14]+1j*(R1[11])-R1[2])
    F2[5]= denom*-1j * M2*S3[2]*(R1[3]+R1[15]+1j*(R1[10])-R1[6])
    return F2


"""
        
        RFS = UFOLorentz(name = 'RFS',
                 spins = [ 4, 2, 1 ],
                 structure = 'Gamma(1,2,-1)*ProjM(-1,1)')        
        builder = create_aloha.AbstractRoutineBuilder(RFS)
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        
        routine = amp.write(output_dir=None, language='Python')
        
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution = """import cmath
import wavefunctions
def RFSC1_0(F2,R1,S3,COUP):
    TMP0 = (F2[4]*(R1[7]+R1[14]+1j*(R1[11])-R1[2])-F2[5]*(R1[3]+R1[15]+1j*(R1[10])-R1[6]))
    vertex = COUP*-1j * TMP0*S3[2]
    return vertex   
    
    
"""

        
        amp = builder.compute_routine(0)
        routine = amp.write(output_dir=None, language='Python')
        split_solution = [l.strip() for l in solution.split('\n')]
        split_routine = [l.strip() for l in routine.split('\n')]
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))
        
        solution = """import cmath
import wavefunctions
def RFSC1_2(F2,S3,COUP,M1,W1):
    OM1 = 0.0
    if (M1): OM1=1.0/M1**2
    R1 = wavefunctions.WaveFunction(size=18)
    R1[0] = +F2[0]+S3[0]
    R1[1] = +F2[1]+S3[1]
    P1 = [-complex(R1[0]).real, -complex(R1[1]).real, -complex(R1[1]).imag, -complex(R1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    R1[2]= denom*1j/3 * M1*S3[2]*(OM1*(P1[0]*(F2[4]*(M1*M1*OM1*(P1[3]-P1[0])+(+2*(P1[0])-P1[3]))+F2[5]*(M1*M1*OM1*(P1[1]-1j*(P1[2]))+(+1j*(P1[2])-P1[1])))-F2[4]*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]))-F2[4])
    R1[6]= denom*-1j/3 * M1*S3[2]*(OM1*(P1[1]*(F2[4]*(M1*M1*OM1*(P1[0]-P1[3])+(P1[3]-P1[0]))+F2[5]*(M1*M1*OM1*(+1j*(P1[2])-P1[1])+(+2*(P1[1])-1j*(P1[2]))))+F2[5]*(P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0]))+F2[5])
    R1[10]= denom*-1/3 * M1*S3[2]*(OM1*(P1[2]*(F2[4]*(M1*M1*OM1*(-1j*(P1[3])+1j*(P1[0]))+(-1j*(P1[0])+1j*(P1[3])))+F2[5]*(M1*-M1*OM1*(P1[2]+1j*(P1[1]))+(+2*(P1[2])+1j*(P1[1]))))+F2[5]*(P1[1]*P1[1]+P1[3]*P1[3]-P1[0]*P1[0]))+F2[5])
    R1[14]= denom*-1j/3 * M1*S3[2]*(OM1*(P1[3]*(F2[4]*(M1*M1*OM1*(P1[0]-P1[3])+(+2*(P1[3])-P1[0]))+F2[5]*(M1*M1*OM1*(+1j*(P1[2])-P1[1])+(P1[1]-1j*(P1[2]))))+F2[4]*(P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0]))+F2[4])
    R1[3]= denom*1j/3 * M1*S3[2]*(OM1*(P1[0]*(F2[4]*(M1*M1*OM1*(P1[1]+1j*(P1[2]))+(-1j*(P1[2])-P1[1]))+F2[5]*(M1*-M1*OM1*(P1[0]+P1[3])+(P1[3]+2*(P1[0]))))-F2[5]*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]))-F2[5])
    R1[7]= denom*-1j/3 * M1*S3[2]*(OM1*(P1[1]*(F2[4]*(M1*-M1*OM1*(P1[1]+1j*(P1[2]))+(+2*(P1[1])+1j*(P1[2])))+F2[5]*(M1*M1*OM1*(P1[0]+P1[3])+(-P1[0]-P1[3])))+F2[4]*(P1[3]*P1[3]+P1[2]*P1[2]-P1[0]*P1[0]))+F2[4])
    R1[11]= denom*1/3 * M1*S3[2]*(OM1*(P1[2]*(F2[4]*(M1*M1*OM1*(+1j*(P1[1])-P1[2])+(-1j*(P1[1])+2*(P1[2])))+F2[5]*(M1*-M1*OM1*(+1j*(P1[0]+P1[3]))+(+1j*(P1[0]+P1[3]))))+F2[4]*(P1[3]*P1[3]+P1[1]*P1[1]-P1[0]*P1[0]))+F2[4])
    R1[15]= denom*1j/3 * M1*S3[2]*(OM1*(P1[3]*(F2[4]*(M1*M1*OM1*(P1[1]+1j*(P1[2]))+(-1j*(P1[2])-P1[1]))+F2[5]*(M1*-M1*OM1*(P1[0]+P1[3])+(P1[0]+2*(P1[3]))))+F2[5]*(P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0]))+F2[5])
    R1[4]= denom*1j * S3[2]*(F2[4]*(OM1*(P1[0]*(M1*M1*(OM1*-1/3*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0])+ -5/3)+(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0]))+1/3*(P1[3]*M1*M1))+(+7/3*(P1[0])-1/3*(P1[3])))+F2[5]*(M1*1/3 * M1*OM1*(P1[1]-1j*(P1[2]))+(-1/3*(P1[1])+1j/3*(P1[2]))))
    R1[8]= denom*1j * S3[2]*(F2[4]*(OM1*(P1[1]*(M1*M1*(OM1*-1/3*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0])+ -5/3)+(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0]))-1j/3*(P1[2]*M1*M1))+(+7/3*(P1[1])+1j/3*(P1[2])))+F2[5]*(M1*1/3 * M1*OM1*(P1[0]+P1[3])+(-1/3*(P1[0]+P1[3]))))
    R1[12]= denom*1/3 * S3[2]*(F2[4]*(OM1*(P1[2]*(M1*M1*(OM1*-1*(-1j*(P1[0]*P1[0])+1j*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]))+ -5j)+(+3j*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2])-3j*(P1[0]*P1[0])))-P1[1]*M1*M1)+(P1[1]+7j*(P1[2])))+F2[5]*(M1*M1*OM1*(P1[0]+P1[3])+(-P1[0]-P1[3])))
    R1[16]= denom*1j * S3[2]*(F2[4]*(OM1*(P1[3]*(M1*M1*(OM1*-1/3*(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0])+ -5/3)+(P1[3]*P1[3]+P1[1]*P1[1]+P1[2]*P1[2]-P1[0]*P1[0]))+1/3*(P1[0]*M1*M1))+(+7/3*(P1[3])-1/3*(P1[0])))+F2[5]*(M1*1/3 * M1*OM1*(+1j*(P1[2])-P1[1])+(-1j/3*(P1[2])+1/3*(P1[1]))))
    R1[5]= denom*1j * S3[2]*(F2[5]*(OM1*(P1[0]*(M1*M1*(OM1*-1/3*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0])+ -5/3)+(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0]))-1/3*(P1[3]*M1*M1))+(+7/3*(P1[0])+1/3*(P1[3])))+F2[4]*(M1*1/3 * M1*OM1*(P1[1]+1j*(P1[2]))+(-1j/3*(P1[2])-1/3*(P1[1]))))
    R1[9]= denom*1j * S3[2]*(F2[5]*(OM1*(P1[1]*(M1*M1*(OM1*-1/3*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0])+ -5/3)+(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0]))+1j/3*(P1[2]*M1*M1))+(-1j/3*(P1[2])+7/3*(P1[1])))+F2[4]*(M1*1/3 * M1*OM1*(P1[0]-P1[3])+(-1/3*(P1[0])+1/3*(P1[3]))))
    R1[13]= denom*-1/3 * S3[2]*(F2[5]*(OM1*(P1[2]*(M1*M1*(OM1*(-1j*(P1[0]*P1[0])+1j*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]))+ 5j)+(+3j*(P1[0]*P1[0])-3j*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3])))-P1[1]*M1*M1)+(P1[1]-7j*(P1[2])))+F2[4]*(M1*M1*OM1*(P1[0]-P1[3])+(P1[3]-P1[0])))
    R1[17]= denom*1j * S3[2]*(F2[5]*(OM1*(P1[3]*(M1*M1*(OM1*-1/3*(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0])+ -5/3)+(P1[1]*P1[1]+P1[2]*P1[2]+P1[3]*P1[3]-P1[0]*P1[0]))-1/3*(P1[0]*M1*M1))+(+7/3*(P1[3])+1/3*(P1[0])))+F2[4]*(M1*1/3 * M1*OM1*(P1[1]+1j*(P1[2]))+(-1j/3*(P1[2])-1/3*(P1[1]))))
    return R1
        
    
"""

        
        amp = builder.compute_routine(2)
        routine = amp.write(output_dir=None, language='Python')
        split_solution = [l.strip() for l in solution.split('\n')]
        split_routine = [l.strip() for l in routine.split('\n')]
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))        
        

    def test_short_Fortranwriter_spin3half(self):
        """ test that python writer works """
        
        aloha_lib.KERNEL.clean()

        solution ="""subroutine RFSC1_1(R1, S3, COUP, M2, W2,F2)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 R1(*)
 complex*16 S3(*)
 complex*16 F2(6)
 real*8 P2(0:3)
 real*8 W2
 real*8 M2
 complex*16 denom
 complex*16 COUP
    F2(1) = +R1(1)+S3(1)
    F2(2) = +R1(2)+S3(2)
P2(0) = -dble(F2(1))
P2(1) = -dble(F2(2))
P2(2) = -dimag(F2(2))
P2(3) = -dimag(F2(1))
    denom = COUP/(P2(0)**2-P2(1)**2-P2(2)**2-P2(3)**2 - M2 * (M2 -CI* W2))
    F2(3)= denom*CI * S3(3)*(P2(0)*-1d0*(R1(8)+R1(15)+CI*(R1(12))-R1(3))+(P2(1)*(R1(4)+R1(16)+CI*(R1(11))-R1(7))+(P2(2)*(-CI*(R1(7))+CI*(R1(4)+R1(16))-R1(11))-P2(3)*(R1(8)+R1(15)+CI*(R1(12))-R1(3)))))
    F2(4)= denom*CI * S3(3)*(P2(0)*(R1(4)+R1(16)+CI*(R1(11))-R1(7))+(P2(1)*-1d0*(R1(8)+R1(15)+CI*(R1(12))-R1(3))+(P2(2)*(-CI*(R1(3))+CI*(R1(8)+R1(15))-R1(12))-P2(3)*(R1(4)+R1(16)+CI*(R1(11))-R1(7)))))
    F2(5)= denom*CI * M2*S3(3)*(R1(8)+R1(15)+CI*(R1(12))-R1(3))
    F2(6)= denom*-CI * M2*S3(3)*(R1(4)+R1(16)+CI*(R1(11))-R1(7))
end


"""
        
        RFS = UFOLorentz(name = 'RFS',
                 spins = [ 4, 2, 1 ],
                 structure = 'Gamma(1,2,-1)*ProjM(-1,1)')        
        builder = create_aloha.AbstractRoutineBuilder(RFS)
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution = """subroutine RFSC1_0(F2, R1, S3, COUP,vertex)
implicit none 
complex*16 CI
parameter (CI=(0d0,1d0))
complex*16 R1(*)
complex*16 S3(*)
complex*16 TMP0
complex*16 F2(*)
complex*16 vertex
complex*16 COUP
TMP0 = (F2(5)*(R1(8)+R1(15)+CI*(R1(12))-R1(3))-F2(6)*(R1(4)+R1(16)+CI*(R1(11))-R1(7)))
vertex = COUP*-CI * TMP0*S3(3)
end
    
    
"""

        
        amp = builder.compute_routine(0)
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = [l.strip() for l in solution.split('\n')]
        split_routine = [l.strip() for l in routine.split('\n')]
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution = """subroutine RFSC1_2(F2, S3, COUP, M1, W1,R1)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 F2(*)
 complex*16 S3(*)
 real*8 P1(0:3)
 real*8 M1
 complex*16 R1(18)
 real*8 W1
 complex*16 denom
 real*8 OM1
 complex*16 COUP
    OM1 = 0d0
    if (M1.ne.0d0) OM1=1d0/M1**2
    R1(1) = +F2(1)+S3(1)
    R1(2) = +F2(2)+S3(2)
P1(0) = -dble(R1(1))
P1(1) = -dble(R1(2))
P1(2) = -dimag(R1(2))
P1(3) = -dimag(R1(1))
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI* W1))
    R1(3)= denom*1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(0)*(F2(5)*(M1*M1*OM1*(P1(3)-P1(0))+(+2d0*(P1(0))-P1(3)))+F2(6)*(M1*M1*OM1*(P1(1)-CI*(P1(2)))+(+CI*(P1(2))-P1(1))))-F2(5)*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)))-F2(5))
    R1(4)= denom*1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(0)*(F2(5)*(M1*M1*OM1*(P1(1)+CI*(P1(2)))+(-CI*(P1(2))-P1(1)))+F2(6)*(M1*-M1*OM1*(P1(0)+P1(3))+(P1(3)+2d0*(P1(0)))))-F2(6)*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)))-F2(6))
    R1(5)= denom*CI * S3(3)*(F2(5)*(OM1*(P1(0)*(M1*M1*(OM1*-1d0/3d0*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0))+ -5d0/3d0)+(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0)))+1d0/3d0*(P1(3)*M1*M1))+(+7d0/3d0*(P1(0))-1d0/3d0*(P1(3))))+F2(6)*(M1*1d0/3d0 * M1*OM1*(P1(1)-CI*(P1(2)))+(-1d0/3d0*(P1(1))+1d0/3d0 * CI*(P1(2)))))
    R1(6)= denom*CI * S3(3)*(F2(6)*(OM1*(P1(0)*(M1*M1*(OM1*-1d0/3d0*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0))+ -5d0/3d0)+(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0)))-1d0/3d0*(P1(3)*M1*M1))+(+7d0/3d0*(P1(0))+1d0/3d0*(P1(3))))+F2(5)*(M1*1d0/3d0 * M1*OM1*(P1(1)+CI*(P1(2)))+(-1d0/3d0 * CI*(P1(2))-1d0/3d0*(P1(1)))))
    R1(7)= denom*-1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(1)*(F2(5)*(M1*M1*OM1*(P1(0)-P1(3))+(P1(3)-P1(0)))+F2(6)*(M1*M1*OM1*(+CI*(P1(2))-P1(1))+(+2d0*(P1(1))-CI*(P1(2)))))+F2(6)*(P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0)))+F2(6))
    R1(8)= denom*-1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(1)*(F2(5)*(M1*-M1*OM1*(P1(1)+CI*(P1(2)))+(+2d0*(P1(1))+CI*(P1(2))))+F2(6)*(M1*M1*OM1*(P1(0)+P1(3))+(-P1(0)-P1(3))))+F2(5)*(P1(3)*P1(3)+P1(2)*P1(2)-P1(0)*P1(0)))+F2(5))
    R1(9)= denom*CI * S3(3)*(F2(5)*(OM1*(P1(1)*(M1*M1*(OM1*-1d0/3d0*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0))+ -5d0/3d0)+(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0)))-1d0/3d0 * CI*(P1(2)*M1*M1))+(+7d0/3d0*(P1(1))+1d0/3d0 * CI*(P1(2))))+F2(6)*(M1*1d0/3d0 * M1*OM1*(P1(0)+P1(3))+(-1d0/3d0*(P1(0)+P1(3)))))
    R1(10)= denom*CI * S3(3)*(F2(6)*(OM1*(P1(1)*(M1*M1*(OM1*-1d0/3d0*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0))+ -5d0/3d0)+(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0)))+1d0/3d0 * CI*(P1(2)*M1*M1))+(-1d0/3d0 * CI*(P1(2))+7d0/3d0*(P1(1))))+F2(5)*(M1*1d0/3d0 * M1*OM1*(P1(0)-P1(3))+(-1d0/3d0*(P1(0))+1d0/3d0*(P1(3)))))
    R1(11)= denom*-1d0/3d0 * M1*S3(3)*(OM1*(P1(2)*(F2(5)*(M1*M1*OM1*(-CI*(P1(3))+CI*(P1(0)))+(-CI*(P1(0))+CI*(P1(3))))+F2(6)*(M1*-M1*OM1*(P1(2)+CI*(P1(1)))+(+2d0*(P1(2))+CI*(P1(1)))))+F2(6)*(P1(1)*P1(1)+P1(3)*P1(3)-P1(0)*P1(0)))+F2(6))
    R1(12)= denom*1d0/3d0 * M1*S3(3)*(OM1*(P1(2)*(F2(5)*(M1*M1*OM1*(+CI*(P1(1))-P1(2))+(-CI*(P1(1))+2d0*(P1(2))))+F2(6)*(M1*-M1*OM1*(+CI*(P1(0)+P1(3)))+(+CI*(P1(0)+P1(3)))))+F2(5)*(P1(3)*P1(3)+P1(1)*P1(1)-P1(0)*P1(0)))+F2(5))
    R1(13)= denom*1d0/3d0 * S3(3)*(F2(5)*(OM1*(P1(2)*(M1*M1*(OM1*-1d0*(-CI*(P1(0)*P1(0))+CI*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)))+ -5d0 * CI)+(+3d0 * CI*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2))-3d0 * CI*(P1(0)*P1(0))))-P1(1)*M1*M1)+(P1(1)+7d0 * CI*(P1(2))))+F2(6)*(M1*M1*OM1*(P1(0)+P1(3))+(-P1(0)-P1(3))))
    R1(14)= denom*-1d0/3d0 * S3(3)*(F2(6)*(OM1*(P1(2)*(M1*M1*(OM1*(-CI*(P1(0)*P1(0))+CI*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)))+ 5d0 * CI)+(+3d0 * CI*(P1(0)*P1(0))-3d0 * CI*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3))))-P1(1)*M1*M1)+(P1(1)-7d0 * CI*(P1(2))))+F2(5)*(M1*M1*OM1*(P1(0)-P1(3))+(P1(3)-P1(0))))
    R1(15)= denom*-1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(3)*(F2(5)*(M1*M1*OM1*(P1(0)-P1(3))+(+2d0*(P1(3))-P1(0)))+F2(6)*(M1*M1*OM1*(+CI*(P1(2))-P1(1))+(P1(1)-CI*(P1(2)))))+F2(5)*(P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0)))+F2(5))
    R1(16)= denom*1d0/3d0 * CI * M1*S3(3)*(OM1*(P1(3)*(F2(5)*(M1*M1*OM1*(P1(1)+CI*(P1(2)))+(-CI*(P1(2))-P1(1)))+F2(6)*(M1*-M1*OM1*(P1(0)+P1(3))+(P1(0)+2d0*(P1(3)))))+F2(6)*(P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0)))+F2(6))
    R1(17)= denom*CI * S3(3)*(F2(5)*(OM1*(P1(3)*(M1*M1*(OM1*-1d0/3d0*(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0))+ -5d0/3d0)+(P1(3)*P1(3)+P1(1)*P1(1)+P1(2)*P1(2)-P1(0)*P1(0)))+1d0/3d0*(P1(0)*M1*M1))+(+7d0/3d0*(P1(3))-1d0/3d0*(P1(0))))+F2(6)*(M1*1d0/3d0 * M1*OM1*(+CI*(P1(2))-P1(1))+(-1d0/3d0 * CI*(P1(2))+1d0/3d0*(P1(1)))))
    R1(18)= denom*CI * S3(3)*(F2(6)*(OM1*(P1(3)*(M1*M1*(OM1*-1d0/3d0*(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0))+ -5d0/3d0)+(P1(1)*P1(1)+P1(2)*P1(2)+P1(3)*P1(3)-P1(0)*P1(0)))-1d0/3d0*(P1(0)*M1*M1))+(+7d0/3d0*(P1(3))+1d0/3d0*(P1(0))))+F2(5)*(M1*1d0/3d0 * M1*OM1*(P1(1)+CI*(P1(2)))+(-1d0/3d0 * CI*(P1(2))-1d0/3d0*(P1(1)))))
end


"""


        amp = builder.compute_routine(2)
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = [l.strip() for l in solution.split('\n')]
        split_routine = [l.strip() for l in routine.split('\n')]
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

    def test_short_pythonwriter_C(self):
        """ test that python writer works """
        aloha_lib.KERNEL.clean()
        solution ="""import cmath
import wavefunctions
def FFV1C1_1(F1,V3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = +F1[0]+V3[0]
    F2[1] = +F1[1]+V3[1]
    P2 = [-complex(F2[0]).real, -complex(F2[1]).real, -complex(F2[1]).imag, -complex(F2[0]).imag]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*-1j*(F1[2]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]-1j*(V3[4]))+(P2[2]*(V3[4]+1j*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+(F1[3]*(P2[0]*(V3[3]+1j*(V3[4]))+(P2[1]*-1*(V3[2]+V3[5])+(P2[2]*-1*(+1j*(V3[2]+V3[5]))+P2[3]*(V3[3]+1j*(V3[4])))))+M2*(F1[4]*(V3[2]+V3[5])+F1[5]*(V3[3]+1j*(V3[4])))))
    F2[3]= denom*1j*(F1[2]*(P2[0]*(+1j*(V3[4])-V3[3])+(P2[1]*(V3[2]-V3[5])+(P2[2]*(-1j*(V3[2])+1j*(V3[5]))+P2[3]*(V3[3]-1j*(V3[4])))))+(F1[3]*(P2[0]*(V3[2]+V3[5])+(P2[1]*-1*(V3[3]+1j*(V3[4]))+(P2[2]*(+1j*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[4]*(+1j*(V3[4])-V3[3])+F1[5]*(V3[5]-V3[2]))))
    F2[4]= denom*1j*(F1[4]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(+1j*(V3[4])-V3[3])+(P2[2]*-1*(V3[4]+1j*(V3[3]))-P2[3]*(V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]+1j*(V3[4]))+(P2[1]*(V3[5]-V3[2])+(P2[2]*(-1j*(V3[2])+1j*(V3[5]))-P2[3]*(V3[3]+1j*(V3[4])))))+M2*(F1[2]*(V3[5]-V3[2])+F1[3]*(V3[3]+1j*(V3[4])))))
    F2[5]= denom*-1j*(F1[4]*(P2[0]*(+1j*(V3[4])-V3[3])+(P2[1]*(V3[2]+V3[5])+(P2[2]*-1*(+1j*(V3[2]+V3[5]))+P2[3]*(+1j*(V3[4])-V3[3]))))+(F1[5]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]+1j*(V3[4]))+(P2[2]*(V3[4]-1j*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+M2*(F1[2]*(+1j*(V3[4])-V3[3])+F1[3]*(V3[2]+V3[5]))))
    return F2


"""
        
        FFV = UFOLorentz(name = 'FFV1',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,1)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution ="""import cmath
import wavefunctions
def FFV1C1_2(F2,V3,COUP,M1,W1):
    F1 = wavefunctions.WaveFunction(size=6)
    F1[0] = +F2[0]+V3[0]
    F1[1] = +F2[1]+V3[1]
    P1 = [-complex(F1[0]).real, -complex(F1[1]).real, -complex(F1[1]).imag, -complex(F1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    F1[2]= denom*-1j*(F2[2]*(P1[0]*(V3[2]+V3[5])+(P1[1]*-1*(V3[3]+1j*(V3[4]))+(P1[2]*(+1j*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]-1j*(V3[4]))+(P1[1]*(V3[5]-V3[2])+(P1[2]*(-1j*(V3[5])+1j*(V3[2]))+P1[3]*(+1j*(V3[4])-V3[3]))))+M1*(F2[4]*(V3[2]-V3[5])+F2[5]*(+1j*(V3[4])-V3[3]))))
    F1[3]= denom*1j*(F2[2]*(P1[0]*-1*(V3[3]+1j*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(+1j*(V3[2]+V3[5]))-P1[3]*(V3[3]+1j*(V3[4])))))+(F2[3]*(P1[0]*(V3[5]-V3[2])+(P1[1]*(V3[3]-1j*(V3[4]))+(P1[2]*(V3[4]+1j*(V3[3]))+P1[3]*(V3[5]-V3[2]))))+M1*(F2[4]*(V3[3]+1j*(V3[4]))-F2[5]*(V3[2]+V3[5]))))
    F1[4]= denom*1j*(F2[4]*(P1[0]*(V3[5]-V3[2])+(P1[1]*(V3[3]+1j*(V3[4]))+(P1[2]*(V3[4]-1j*(V3[3]))+P1[3]*(V3[5]-V3[2]))))+(F2[5]*(P1[0]*(V3[3]-1j*(V3[4]))+(P1[1]*-1*(V3[2]+V3[5])+(P1[2]*(+1j*(V3[2]+V3[5]))+P1[3]*(V3[3]-1j*(V3[4])))))+M1*(F2[2]*-1*(V3[2]+V3[5])+F2[3]*(+1j*(V3[4])-V3[3]))))
    F1[5]= denom*-1j*(F2[4]*(P1[0]*-1*(V3[3]+1j*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-1j*(V3[5])+1j*(V3[2]))+P1[3]*(V3[3]+1j*(V3[4])))))+(F2[5]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(+1j*(V3[4])-V3[3])+(P1[2]*-1*(V3[4]+1j*(V3[3]))-P1[3]*(V3[2]+V3[5]))))+M1*(F2[2]*(V3[3]+1j*(V3[4]))+F2[3]*(V3[2]-V3[5]))))
    return F1


"""

        amp = builder.compute_routine(2)
        
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution = """import cmath
import wavefunctions
def FFV1C1_1(F1,V3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = +F1[0]+V3[0]
    F2[1] = +F1[1]+V3[1]
    P2 = [-complex(F2[0]).real, -complex(F2[1]).real, -complex(F2[1]).imag, -complex(F2[0]).imag]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*-1j*(F1[2]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]-1j*(V3[4]))+(P2[2]*(V3[4]+1j*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+(F1[3]*(P2[0]*(V3[3]+1j*(V3[4]))+(P2[1]*-1*(V3[2]+V3[5])+(P2[2]*-1*(+1j*(V3[2]+V3[5]))+P2[3]*(V3[3]+1j*(V3[4])))))+M2*(F1[4]*(V3[2]+V3[5])+F1[5]*(V3[3]+1j*(V3[4])))))
    F2[3]= denom*1j*(F1[2]*(P2[0]*(+1j*(V3[4])-V3[3])+(P2[1]*(V3[2]-V3[5])+(P2[2]*(-1j*(V3[2])+1j*(V3[5]))+P2[3]*(V3[3]-1j*(V3[4])))))+(F1[3]*(P2[0]*(V3[2]+V3[5])+(P2[1]*-1*(V3[3]+1j*(V3[4]))+(P2[2]*(+1j*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[4]*(+1j*(V3[4])-V3[3])+F1[5]*(V3[5]-V3[2]))))
    F2[4]= denom*1j*(F1[4]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(+1j*(V3[4])-V3[3])+(P2[2]*-1*(V3[4]+1j*(V3[3]))-P2[3]*(V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]+1j*(V3[4]))+(P2[1]*(V3[5]-V3[2])+(P2[2]*(-1j*(V3[2])+1j*(V3[5]))-P2[3]*(V3[3]+1j*(V3[4])))))+M2*(F1[2]*(V3[5]-V3[2])+F1[3]*(V3[3]+1j*(V3[4])))))
    F2[5]= denom*-1j*(F1[4]*(P2[0]*(+1j*(V3[4])-V3[3])+(P2[1]*(V3[2]+V3[5])+(P2[2]*-1*(+1j*(V3[2]+V3[5]))+P2[3]*(+1j*(V3[4])-V3[3]))))+(F1[5]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]+1j*(V3[4]))+(P2[2]*(V3[4]-1j*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+M2*(F1[2]*(+1j*(V3[4])-V3[3])+F1[3]*(V3[2]+V3[5]))))
    return F2


import cmath
import wavefunctions
def FFV1_2C1_1(F1,V3,COUP1,COUP2,M2,W2):
    F2 = FFV1C1_1(F1,V3,COUP1,M2,W2)
    tmp = FFV2C1_1(F1,V3,COUP2,M2,W2)
    for i in range(2,6):
        F2[i] += tmp[i]
    return F2

"""



        FFV = UFOLorentz(name = 'FFV1',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,1)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV) 
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        amp.add_combine(('FFV2',))
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))


       
    def test_short_pythonwriter_4_fermion(self):
        """ test that python writer works """
        aloha_lib.KERNEL.clean()
        solution ="""import cmath
import wavefunctions
def FFFF1_1(F2,F3,F4,COUP,M1,W1):
    F1 = wavefunctions.WaveFunction(size=6)
    F1[0] = +F2[0]+F3[0]+F4[0]
    F1[1] = +F2[1]+F3[1]+F4[1]
    P1 = [-complex(F1[0]).real, -complex(F1[1]).real, -complex(F1[1]).imag, -complex(F1[0]).imag]
    TMP0 = (F4[2]*F3[2]+F4[3]*F3[3]+F4[4]*F3[4]+F4[5]*F3[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    F1[2]= denom*-1j * TMP0*(F2[4]*(P1[0]+P1[3])+(F2[5]*(P1[1]+1j*(P1[2]))-F2[2]*M1))
    F1[3]= denom*1j * TMP0*(F2[4]*(+1j*(P1[2])-P1[1])+(F2[5]*(P1[3]-P1[0])+F2[3]*M1))
    F1[4]= denom*1j * TMP0*(F2[2]*(P1[3]-P1[0])+(F2[3]*(P1[1]+1j*(P1[2]))+F2[4]*M1))
    F1[5]= denom*-1j * TMP0*(F2[2]*(+1j*(P1[2])-P1[1])+(F2[3]*(P1[0]+P1[3])-F2[5]*M1))
    return F1


"""
        
        FFFF = UFOLorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Identity(1,2)*Identity(4,3)')       
        builder = create_aloha.AbstractRoutineBuilder(FFFF)
        amp = builder.compute_routine(1)
        
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))
        
        solution ="""import cmath
import wavefunctions
def FFFF1C1_1(F1,F3,F4,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = +F1[0]+F3[0]+F4[0]
    F2[1] = +F1[1]+F3[1]+F4[1]
    P2 = [-complex(F2[0]).real, -complex(F2[1]).real, -complex(F2[1]).imag, -complex(F2[0]).imag]
    TMP0 = (F4[2]*F3[2]+F4[3]*F3[3]+F4[4]*F3[4]+F4[5]*F3[5])
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*-1j * TMP0*(F1[4]*(P2[0]+P2[3])+(F1[5]*(P2[1]+1j*(P2[2]))-F1[2]*M2))
    F2[3]= denom*1j * TMP0*(F1[4]*(+1j*(P2[2])-P2[1])+(F1[5]*(P2[3]-P2[0])+F1[3]*M2))
    F2[4]= denom*1j * TMP0*(F1[2]*(P2[3]-P2[0])+(F1[3]*(P2[1]+1j*(P2[2]))+F1[4]*M2))
    F2[5]= denom*-1j * TMP0*(F1[2]*(+1j*(P2[2])-P2[1])+(F1[3]*(P2[0]+P2[3])-F1[5]*M2))
    return F2


"""
        
        FFFF = UFOLorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Identity(2,1)*Identity(4,3)')       
        builder = create_aloha.AbstractRoutineBuilder(FFFF)
        builder.apply_conjugation(1)
        amp = builder.compute_routine(1)
        
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))
       
        solution ="""import cmath
import wavefunctions
def FFFF1C2_1(F2,F4,F3,COUP,M1,W1):
    F1 = wavefunctions.WaveFunction(size=6)
    F1[0] = +F2[0]+F3[0]+F4[0]
    F1[1] = +F2[1]+F3[1]+F4[1]
    P1 = [-complex(F1[0]).real, -complex(F1[1]).real, -complex(F1[1]).imag, -complex(F1[0]).imag]
    TMP0 = (F4[2]*F3[2]+F4[3]*F3[3]+F4[4]*F3[4]+F4[5]*F3[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    F1[2]= denom*-1j * TMP0*(F2[4]*(P1[0]+P1[3])+(F2[5]*(P1[1]+1j*(P1[2]))-F2[2]*M1))
    F1[3]= denom*1j * TMP0*(F2[4]*(+1j*(P1[2])-P1[1])+(F2[5]*(P1[3]-P1[0])+F2[3]*M1))
    F1[4]= denom*1j * TMP0*(F2[2]*(P1[3]-P1[0])+(F2[3]*(P1[1]+1j*(P1[2]))+F2[4]*M1))
    F1[5]= denom*-1j * TMP0*(F2[2]*(+1j*(P1[2])-P1[1])+(F2[3]*(P1[0]+P1[3])-F2[5]*M1))
    return F1


"""
        
        FFFF = UFOLorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Identity(2,1)*Identity(4,3)')       
        builder = create_aloha.AbstractRoutineBuilder(FFFF)
        builder.apply_conjugation(2)
        amp = builder.compute_routine(1)
        
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))       

    def test_short_pythonwriter_Plorentz(self):
        """ test that python writer works """
        
        aloha_lib.KERNEL.clean()
        solution ="""import cmath
import wavefunctions
def FFV13C1_0(F2,F1,V3,COUP):
    P2 = [complex(F2[0]).real, complex(F2[1]).real, complex(F2[1]).imag, complex(F2[0]).imag]
    P3 = [complex(V3[0]).real, complex(V3[1]).real, complex(V3[1]).imag, complex(V3[0]).imag]
    TMP5 = (F1[4]*(F2[2]*(P2[1]*(P3[2]*-1*(V3[5]+V3[2])+V3[4]*(P3[3]+P3[0]))+(P2[2]*(P3[1]*(V3[5]+V3[2])-V3[3]*(P3[3]+P3[0]))+(P3[1]*-V3[4]*(P2[3]+P2[0])+P3[2]*V3[3]*(P2[3]+P2[0]))))+F2[3]*(P2[0]*(P3[3]*(V3[4]+1j*(V3[3]))-V3[5]*(P3[2]+1j*(P3[1])))+(P2[3]*(P3[0]*-1*(V3[4]+1j*(V3[3]))+V3[2]*(P3[2]+1j*(P3[1])))+(P3[0]*V3[5]*(P2[2]+1j*(P2[1]))-P3[3]*V3[2]*(P2[2]+1j*(P2[1]))))))+F1[5]*(F2[2]*(P2[0]*(P3[3]*(V3[4]-1j*(V3[3]))+V3[5]*(+1j*(P3[1])-P3[2]))+(P2[3]*(P3[0]*(+1j*(V3[3])-V3[4])+V3[2]*(P3[2]-1j*(P3[1])))+(P3[0]*V3[5]*(P2[2]-1j*(P2[1]))+P3[3]*V3[2]*(+1j*(P2[1])-P2[2]))))+F2[3]*(P2[1]*(P3[2]*(V3[2]-V3[5])+V3[4]*(P3[3]-P3[0]))+(P2[2]*(P3[1]*(V3[5]-V3[2])+V3[3]*(P3[0]-P3[3]))+(P3[1]*V3[4]*(P2[0]-P2[3])+P3[2]*V3[3]*(P2[3]-P2[0]))))))
    TMP4 = (P2[0]*P3[0]-P2[1]*P3[1]-P2[2]*P3[2]-P2[3]*P3[3])
    TMP7 = -1*(F1[4]*(F2[2]*(V3[2]+V3[5])+F2[3]*(V3[3]-1j*(V3[4])))+F1[5]*(F2[2]*(V3[3]+1j*(V3[4]))+F2[3]*(V3[2]-V3[5])))
    TMP6 = -1*(F1[4]*(F2[2]*(P3[0]+P3[3])+F2[3]*(P3[1]-1j*(P3[2])))+F1[5]*(F2[2]*(P3[1]+1j*(P3[2]))+F2[3]*(P3[0]-P3[3])))
    TMP1 = (V3[2]*P2[0]-V3[3]*P2[1]-V3[4]*P2[2]-V3[5]*P2[3])
    TMP0 = (F1[2]*(F2[4]*(P2[1]*(P3[2]*(V3[2]-V3[5])+V3[4]*(P3[3]-P3[0]))+(P2[2]*(P3[1]*(V3[5]-V3[2])+V3[3]*(P3[0]-P3[3]))+(P3[1]*V3[4]*(P2[0]-P2[3])+P3[2]*V3[3]*(P2[3]-P2[0]))))+F2[5]*(P2[0]*(P3[3]*-1*(V3[4]+1j*(V3[3]))+V3[5]*(P3[2]+1j*(P3[1])))+(P2[3]*(P3[0]*(V3[4]+1j*(V3[3]))-V3[2]*(P3[2]+1j*(P3[1])))+(P3[0]*-V3[5]*(P2[2]+1j*(P2[1]))+P3[3]*V3[2]*(P2[2]+1j*(P2[1]))))))+F1[3]*(F2[4]*(P2[0]*(P3[3]*(+1j*(V3[3])-V3[4])+V3[5]*(P3[2]-1j*(P3[1])))+(P2[3]*(P3[0]*(V3[4]-1j*(V3[3]))+V3[2]*(+1j*(P3[1])-P3[2]))+(P3[0]*V3[5]*(+1j*(P2[1])-P2[2])+P3[3]*V3[2]*(P2[2]-1j*(P2[1])))))+F2[5]*(P2[1]*(P3[2]*-1*(V3[5]+V3[2])+V3[4]*(P3[3]+P3[0]))+(P2[2]*(P3[1]*(V3[5]+V3[2])-V3[3]*(P3[3]+P3[0]))+(P3[1]*-V3[4]*(P2[3]+P2[0])+P3[2]*V3[3]*(P2[3]+P2[0]))))))
    TMP3 = -1*(F1[2]*(F2[4]*(V3[2]-V3[5])+F2[5]*(+1j*(V3[4])-V3[3]))+F1[3]*(F2[4]*-1*(V3[3]+1j*(V3[4]))+F2[5]*(V3[2]+V3[5])))
    TMP2 = (F1[2]*(F2[4]*(P3[3]-P3[0])+F2[5]*(P3[1]-1j*(P3[2])))+F1[3]*(F2[4]*(P3[1]+1j*(P3[2]))-F2[5]*(P3[0]+P3[3])))
    vertex = COUP*(TMP1*(TMP2+TMP6)+(TMP4*-1*(TMP3+TMP7)+(-1j*(TMP0)+1j*(TMP5))))
    return vertex


"""
        
        FFV = UFOLorentz(name = 'FFV13',
                spins = [ 2, 2, 3 ],
        structure = 'Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjM(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) - complex(0,1)*P(-1\
,2)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1) - Epsilon(3,-1,-2,-3)*P(-2,2)*P(-1,3)*Gamma(-3,2,-4)*ProjP(-4,1) + complex(0,1)*P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) - comple\
x(0,1)*P(-1,2)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1)')
      
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        builder.apply_conjugation()
        amp = builder.compute_routine(0)
        
        routine = amp.write(output_dir=None, language='Python')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(len(split_routine), len(split_solution))
        self.assertEqual(split_solution, split_routine)
            
    
    @set_global(loop=True, unitary=False, mp=True, cms=False)
    def R_test_aloha_Loop_feynmangauge(self):
        """Test the definition of the momenta"""
        aloha_lib.KERNEL.clean()

        FFV_M = UFOLorentz(name = 'FFVM',
             spins = [ 2, 2, 3 ],
             structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)') 
        
        abstract = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3,'L')
        text = abstract.write('/tmp')
        
        target = """subroutine FFVML_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 V3(8)
 real*8 W3
 real*8 M3
 complex*16 F1(*)
 complex*16 F2(*)
 complex*16 COUP
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
    V3(5)= COUP*-CI*(F2(5)*F1(7)+F2(6)*F1(8))
    V3(6)= COUP*-CI*(-F2(6)*F1(7)-F2(5)*F1(8))
    V3(7)= COUP*-CI*(-CI*(F2(5)*F1(8))+CI*(F2(6)*F1(7)))
    V3(8)= COUP*-CI*(F2(6)*F1(8)-F2(5)*F1(7))
end


subroutine MP_FFVML_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*32 CI
 parameter (CI=(0q0,1q0))
 complex*32 V3(8)
 real*16 W3
 real*16 M3
 complex*32 F1(*)
 complex*32 F2(*)
 complex*32 COUP
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
    V3(5)= COUP*-CI*(F2(5)*F1(7)+F2(6)*F1(8))
    V3(6)= COUP*-CI*(-F2(6)*F1(7)-F2(5)*F1(8))
    V3(7)= COUP*-CI*(-CI*(F2(5)*F1(8))+CI*(F2(6)*F1(7)))
    V3(8)= COUP*-CI*(F2(6)*F1(8)-F2(5)*F1(7))
end


"""
        self.assertEqual(text.split('\n'), target.split('\n'))
            
    @set_global(loop=False, unitary=True, mp=True, cms=False)
    def test_short_aloha_get_name(self):

        FFV_M = UFOLorentz(name = 'FFVM',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')  
        abstract = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3)
        
        name = aloha_writers.get_routine_name(abstract=abstract, tag=['P0','C1'])
        
        self.assertEqual(name, 'FFVMC1P0_3')

        name = aloha_writers.get_routine_name(abstract=abstract, tag=['C1','P0'])
        
        self.assertEqual(name, 'FFVMC1P0_3')
    
    @set_global(loop=True, unitary=True, mp=True, cms=False)
    def test_short_aloha_MP_mode(self):
        """ """
        aloha_lib.KERNEL.clean()

        FFV_M = UFOLorentz(name = 'FFVM',
             spins = [ 2, 2, 3 ],
             structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)') 
        
        abstract = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3)
        text = abstract.write('/tmp')

        # Not performed the Fortran formatting
        target = """subroutine FFVM_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 denom
 complex*16 V3(8)
 real*8 W3
 complex*16 TMP0
 real*8 M3
 complex*16 F1(*)
 complex*16 P3(0:3)
 complex*16 F2(*)
 real*8 OM3
 complex*16 COUP
    OM3 = 0d0
    if (M3.ne.0d0) OM3=1d0/M3**2
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
P3(0) = -V3(1)
P3(1) = -V3(2)
P3(2) = -V3(3)
P3(3) = -V3(4)
 TMP0 = (F1(7)*(F2(5)*(P3(0)+P3(3))+F2(6)*(P3(1)-CI*(P3(2))))+F1(8)*(F2(5)*(P3(1)+CI*(P3(2)))+F2(6)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(5)= denom*-CI*(F2(5)*F1(7)+F2(6)*F1(8)-P3(0)*OM3*TMP0)
    V3(6)= denom*-CI*(-F2(6)*F1(7)-F2(5)*F1(8)-P3(1)*OM3*TMP0)
    V3(7)= denom*-CI*(-CI*(F2(5)*F1(8))+CI*(F2(6)*F1(7))-P3(2)*OM3*TMP0)
    V3(8)= denom*-CI*(F2(6)*F1(8)-F2(5)*F1(7)-P3(3)*OM3*TMP0)
end


subroutine MP_FFVM_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*32 CI
 parameter (CI=(0q0,1q0))
 complex*32 denom
 complex*32 V3(8)
 real*16 W3
 complex*32 TMP0
 real*16 M3
 complex*32 F1(*)
 complex*32 P3(0:3)
 complex*32 F2(*)
 real*16 OM3
 complex*32 COUP
    OM3 = 0q0
    if (M3.ne.0q0) OM3=1q0/M3**2
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
P3(0) = -V3(1)
P3(1) = -V3(2)
P3(2) = -V3(3)
P3(3) = -V3(4)
 TMP0 = (F1(7)*(F2(5)*(P3(0)+P3(3))+F2(6)*(P3(1)-CI*(P3(2))))+F1(8)*(F2(5)*(P3(1)+CI*(P3(2)))+F2(6)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(5)= denom*-CI*(F2(5)*F1(7)+F2(6)*F1(8)-P3(0)*OM3*TMP0)
    V3(6)= denom*-CI*(-F2(6)*F1(7)-F2(5)*F1(8)-P3(1)*OM3*TMP0)
    V3(7)= denom*-CI*(-CI*(F2(5)*F1(8))+CI*(F2(6)*F1(7))-P3(2)*OM3*TMP0)
    V3(8)= denom*-CI*(F2(6)*F1(8)-F2(5)*F1(7)-P3(3)*OM3*TMP0)
end


"""
        target2="""subroutine FFVM_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 denom
 complex*16 V3(8)
 real*8 W3
 complex*16 TMP0
 real*8 M3
 complex*16 F1(*)
 complex*16 P3(0:3)
 complex*16 F2(*)
 real*8 OM3
 complex*16 COUP
    OM3 = 0d0
    if (M3.ne.0d0) OM3=1d0/M3**2
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
P3(0) = -V3(1)
P3(1) = -V3(2)
P3(2) = -V3(3)
P3(3) = -V3(4)
 TMP0 = (F1(7)*(F2(5)*(P3(0)+P3(3))+F2(6)*(P3(1)-CI*(P3(2))))+F1(8)*(F2(5)*(P3(1)+CI*(P3(2)))+F2(6)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(5)= denom*-CI*(F1(7)*F2(5)+F1(8)*F2(6)-P3(0)*OM3*TMP0)
    V3(6)= denom*-CI*(-F1(7)*F2(6)-F1(8)*F2(5)-P3(1)*OM3*TMP0)
    V3(7)= denom*-CI*(-CI*(F1(8)*F2(5))+CI*(F1(7)*F2(6))-P3(2)*OM3*TMP0)
    V3(8)= denom*-CI*(F1(8)*F2(6)-F1(7)*F2(5)-P3(3)*OM3*TMP0)
end


subroutine MP_FFVM_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*32 CI
 parameter (CI=(0q0,1q0))
 complex*32 denom
 complex*32 V3(8)
 real*16 W3
 complex*32 TMP0
 real*16 M3
 complex*32 F1(*)
 complex*32 P3(0:3)
 complex*32 F2(*)
 real*16 OM3
 complex*32 COUP
    OM3 = 0q0
    if (M3.ne.0q0) OM3=1q0/M3**2
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
    V3(3) = +F1(3)+F2(3)
    V3(4) = +F1(4)+F2(4)
P3(0) = -V3(1)
P3(1) = -V3(2)
P3(2) = -V3(3)
P3(3) = -V3(4)
 TMP0 = (F1(7)*(F2(5)*(P3(0)+P3(3))+F2(6)*(P3(1)-CI*(P3(2))))+F1(8)*(F2(5)*(P3(1)+CI*(P3(2)))+F2(6)*(P3(0)-P3(3))))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(5)= denom*-CI*(F1(7)*F2(5)+F1(8)*F2(6)-P3(0)*OM3*TMP0)
    V3(6)= denom*-CI*(-F1(7)*F2(6)-F1(8)*F2(5)-P3(1)*OM3*TMP0)
    V3(7)= denom*-CI*(-CI*(F1(8)*F2(5))+CI*(F1(7)*F2(6))-P3(2)*OM3*TMP0)
    V3(8)= denom*-CI*(F1(8)*F2(6)-F1(7)*F2(5)-P3(3)*OM3*TMP0)
end


"""
        try:
            self.assertEqual(text.split('\n'), target.split('\n'))         
        except Exception:
            self.assertEqual(text.split('\n'), target2.split('\n'))         
    def test_short_fortranwriter_C(self):
        """ test that python writer works """

        solution = """subroutine FFV1C1_1(F1, V3, COUP, M2, W2,F2)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 F2(6)
 complex*16 V3(*)
 real*8 P2(0:3)
 real*8 W2
 complex*16 F1(*)
 real*8 M2
 complex*16 denom
 complex*16 COUP
    F2(1) = +F1(1)+V3(1)
    F2(2) = +F1(2)+V3(2)
P2(0) = -dble(F2(1))
P2(1) = -dble(F2(2))
P2(2) = -dimag(F2(2))
P2(3) = -dimag(F2(1))
    denom = COUP/(P2(0)**2-P2(1)**2-P2(2)**2-P2(3)**2 - M2 * (M2 -CI* W2))
    F2(3)= denom*-CI*(F1(3)*(P2(0)*(V3(6)-V3(3))+(P2(1)*(V3(4)-CI*(V3(5)))+(P2(2)*(V3(5)+CI*(V3(4)))+P2(3)*(V3(6)-V3(3)))))+(F1(4)*(P2(0)*(V3(4)+CI*(V3(5)))+(P2(1)*-1d0*(V3(3)+V3(6))+(P2(2)*-1d0*(+CI*(V3(3)+V3(6)))+P2(3)*(V3(4)+CI*(V3(5))))))+M2*(F1(5)*(V3(3)+V3(6))+F1(6)*(V3(4)+CI*(V3(5))))))
    F2(4)= denom*CI*(F1(3)*(P2(0)*(+CI*(V3(5))-V3(4))+(P2(1)*(V3(3)-V3(6))+(P2(2)*(-CI*(V3(3))+CI*(V3(6)))+P2(3)*(V3(4)-CI*(V3(5))))))+(F1(4)*(P2(0)*(V3(3)+V3(6))+(P2(1)*-1d0*(V3(4)+CI*(V3(5)))+(P2(2)*(+CI*(V3(4))-V3(5))-P2(3)*(V3(3)+V3(6)))))+M2*(F1(5)*(+CI*(V3(5))-V3(4))+F1(6)*(V3(6)-V3(3)))))
    F2(5)= denom*CI*(F1(5)*(P2(0)*(V3(3)+V3(6))+(P2(1)*(+CI*(V3(5))-V3(4))+(P2(2)*-1d0*(V3(5)+CI*(V3(4)))-P2(3)*(V3(3)+V3(6)))))+(F1(6)*(P2(0)*(V3(4)+CI*(V3(5)))+(P2(1)*(V3(6)-V3(3))+(P2(2)*(-CI*(V3(3))+CI*(V3(6)))-P2(3)*(V3(4)+CI*(V3(5))))))+M2*(F1(3)*(V3(6)-V3(3))+F1(4)*(V3(4)+CI*(V3(5))))))
    F2(6)= denom*-CI*(F1(5)*(P2(0)*(+CI*(V3(5))-V3(4))+(P2(1)*(V3(3)+V3(6))+(P2(2)*-1d0*(+CI*(V3(3)+V3(6)))+P2(3)*(+CI*(V3(5))-V3(4)))))+(F1(6)*(P2(0)*(V3(6)-V3(3))+(P2(1)*(V3(4)+CI*(V3(5)))+(P2(2)*(V3(5)-CI*(V3(4)))+P2(3)*(V3(6)-V3(3)))))+M2*(F1(3)*(+CI*(V3(5))-V3(4))+F1(4)*(V3(3)+V3(6)))))
end


"""        
        FFV = UFOLorentz(name = 'FFV1',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,1)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution="""subroutine FFV1C1_2(F2, V3, COUP, M1, W1,F1)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 F2(*)
 complex*16 V3(*)
 real*8 P1(0:3)
 real*8 M1
 real*8 W1
 complex*16 F1(6)
 complex*16 denom
 complex*16 COUP
    F1(1) = +F2(1)+V3(1)
    F1(2) = +F2(2)+V3(2)
P1(0) = -dble(F1(1))
P1(1) = -dble(F1(2))
P1(2) = -dimag(F1(2))
P1(3) = -dimag(F1(1))
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI* W1))
    F1(3)= denom*-CI*(F2(3)*(P1(0)*(V3(3)+V3(6))+(P1(1)*-1d0*(V3(4)+CI*(V3(5)))+(P1(2)*(+CI*(V3(4))-V3(5))-P1(3)*(V3(3)+V3(6)))))+(F2(4)*(P1(0)*(V3(4)-CI*(V3(5)))+(P1(1)*(V3(6)-V3(3))+(P1(2)*(-CI*(V3(6))+CI*(V3(3)))+P1(3)*(+CI*(V3(5))-V3(4)))))+M1*(F2(5)*(V3(3)-V3(6))+F2(6)*(+CI*(V3(5))-V3(4)))))
    F1(4)= denom*CI*(F2(3)*(P1(0)*-1d0*(V3(4)+CI*(V3(5)))+(P1(1)*(V3(3)+V3(6))+(P1(2)*(+CI*(V3(3)+V3(6)))-P1(3)*(V3(4)+CI*(V3(5))))))+(F2(4)*(P1(0)*(V3(6)-V3(3))+(P1(1)*(V3(4)-CI*(V3(5)))+(P1(2)*(V3(5)+CI*(V3(4)))+P1(3)*(V3(6)-V3(3)))))+M1*(F2(5)*(V3(4)+CI*(V3(5)))-F2(6)*(V3(3)+V3(6)))))
    F1(5)= denom*CI*(F2(5)*(P1(0)*(V3(6)-V3(3))+(P1(1)*(V3(4)+CI*(V3(5)))+(P1(2)*(V3(5)-CI*(V3(4)))+P1(3)*(V3(6)-V3(3)))))+(F2(6)*(P1(0)*(V3(4)-CI*(V3(5)))+(P1(1)*-1d0*(V3(3)+V3(6))+(P1(2)*(+CI*(V3(3)+V3(6)))+P1(3)*(V3(4)-CI*(V3(5))))))+M1*(F2(3)*-1d0*(V3(3)+V3(6))+F2(4)*(+CI*(V3(5))-V3(4)))))
    F1(6)= denom*-CI*(F2(5)*(P1(0)*-1d0*(V3(4)+CI*(V3(5)))+(P1(1)*(V3(3)-V3(6))+(P1(2)*(-CI*(V3(6))+CI*(V3(3)))+P1(3)*(V3(4)+CI*(V3(5))))))+(F2(6)*(P1(0)*(V3(3)+V3(6))+(P1(1)*(+CI*(V3(5))-V3(4))+(P1(2)*-1d0*(V3(5)+CI*(V3(4)))-P1(3)*(V3(3)+V3(6)))))+M1*(F2(3)*(V3(4)+CI*(V3(5)))+F2(4)*(V3(3)-V3(6)))))
end


"""
        amp = builder.compute_routine(2)
        
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

    def test_short_Cppwriter_C(self):
        """ test that python writer works """

 
        solution_h = """#ifndef FFV1C1_1_guard
#define FFV1C1_1_guard
#include <complex>
using namespace std;

void FFV1C1_1(complex<double> F1[], complex<double> V3[], complex<double> COUP, double M2, double W2,complex<double> F2[]);
#endif

"""
        solution_c="""#include "FFV1C1_1.h"

void FFV1C1_1(complex<double> F1[], complex<double> V3[], complex<double> COUP, double M2, double W2,complex<double> F2[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P2[4];
 complex<double>  denom;
    F2[0] = +F1[0]+V3[0];
    F2[1] = +F1[1]+V3[1];
P2[0] = -F2[0].real();
P2[1] = -F2[1].real();
P2[2] = -F2[1].imag();
P2[3] = -F2[0].imag();
    denom = COUP/(pow(P2[0],2)-pow(P2[1],2)-pow(P2[2],2)-pow(P2[3],2) - M2 * (M2 -cI* W2));
    F2[2]= denom*-cI*(F1[2]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]-cI*(V3[4]))+(P2[2]*(V3[4]+cI*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+(F1[3]*(P2[0]*(V3[3]+cI*(V3[4]))+(P2[1]*-1.*(V3[2]+V3[5])+(P2[2]*-1.*(+cI*(V3[2]+V3[5]))+P2[3]*(V3[3]+cI*(V3[4])))))+M2*(F1[4]*(V3[2]+V3[5])+F1[5]*(V3[3]+cI*(V3[4])))));
    F2[3]= denom*cI*(F1[2]*(P2[0]*(+cI*(V3[4])-V3[3])+(P2[1]*(V3[2]-V3[5])+(P2[2]*(-cI*(V3[2])+cI*(V3[5]))+P2[3]*(V3[3]-cI*(V3[4])))))+(F1[3]*(P2[0]*(V3[2]+V3[5])+(P2[1]*-1.*(V3[3]+cI*(V3[4]))+(P2[2]*(+cI*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[4]*(+cI*(V3[4])-V3[3])+F1[5]*(V3[5]-V3[2]))));
    F2[4]= denom*cI*(F1[4]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(+cI*(V3[4])-V3[3])+(P2[2]*-1.*(V3[4]+cI*(V3[3]))-P2[3]*(V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[5]-V3[2])+(P2[2]*(-cI*(V3[2])+cI*(V3[5]))-P2[3]*(V3[3]+cI*(V3[4])))))+M2*(F1[2]*(V3[5]-V3[2])+F1[3]*(V3[3]+cI*(V3[4])))));
    F2[5]= denom*-cI*(F1[4]*(P2[0]*(+cI*(V3[4])-V3[3])+(P2[1]*(V3[2]+V3[5])+(P2[2]*-1.*(+cI*(V3[2]+V3[5]))+P2[3]*(+cI*(V3[4])-V3[3]))))+(F1[5]*(P2[0]*(V3[5]-V3[2])+(P2[1]*(V3[3]+cI*(V3[4]))+(P2[2]*(V3[4]-cI*(V3[3]))+P2[3]*(V3[5]-V3[2]))))+M2*(F1[2]*(+cI*(V3[4])-V3[3])+F1[3]*(V3[2]+V3[5]))));
}

"""

        solution2_c = """#include "FFV1C1_1.h"

void FFV1C1_1(complex<double> F1[], complex<double> V3[], complex<double> COUP, double M2, double W2,complex<double> F2[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P2[4];
 complex<double>  denom;
    F2[0] = +F1[0]+V3[0];
    F2[1] = +F1[1]+V3[1];
P2[0] = -F2[0].real();
P2[1] = -F2[1].real();
P2[2] = -F2[1].imag();
P2[3] = -F2[0].imag();
    denom = COUP/(pow(P2[0],2)-pow(P2[1],2)-pow(P2[2],2)-pow(P2[3],2) - M2 * (M2 -cI* W2));
    F2[2]= 
    F2[3]= 
    F2[4]= 
    F2[5]= 
}

"""
        
        FFV = UFOLorentz(name = 'FFV1',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,1)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        builder.apply_conjugation()
        amp = builder.compute_routine(1)
        routine = amp.write(output_dir=None, language='CPP')
        
        split_solution = solution_h.split('\n')
        split_routine = routine[0].split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))
        
        split_solution = solution_c.split('\n')
        split_solution2 = solution2_c.split('\n')
        split_routine = routine[1].split('\n')
        for i in range(len(split_routine)):
            try:
                self.assertEqual(split_solution[i], split_routine[i])
            except:
                self.assertEqual(split_solution2[i], split_routine[i])
        self.assertEqual(len(split_routine), len(split_solution))

        solution_h = """#ifndef FFV1C1_2_guard
#define FFV1C1_2_guard
#include <complex>
using namespace std;

void FFV1C1_2(complex<double> F2[], complex<double> V3[], complex<double> COUP, double M1, double W1,complex<double> F1[]);
#endif

"""

        solution_c = """#include "FFV1C1_2.h"

void FFV1C1_2(complex<double> F2[], complex<double> V3[], complex<double> COUP, double M1, double W1,complex<double> F1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  denom;
    F1[0] = +F2[0]+V3[0];
    F1[1] = +F2[1]+V3[1];
P1[0] = -F1[0].real();
P1[1] = -F1[1].real();
P1[2] = -F1[1].imag();
P1[3] = -F1[0].imag();
    denom = COUP/(pow(P1[0],2)-pow(P1[1],2)-pow(P1[2],2)-pow(P1[3],2) - M1 * (M1 -cI* W1));
    F1[2]= denom*-cI*(F2[2]*(P1[0]*(V3[2]+V3[5])+(P1[1]*-1.*(V3[3]+cI*(V3[4]))+(P1[2]*(+cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]-cI*(V3[4]))+(P1[1]*(V3[5]-V3[2])+(P1[2]*(-cI*(V3[5])+cI*(V3[2]))+P1[3]*(+cI*(V3[4])-V3[3]))))+M1*(F2[4]*(V3[2]-V3[5])+F2[5]*(+cI*(V3[4])-V3[3]))));
    F1[3]= denom*cI*(F2[2]*(P1[0]*-1.*(V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(+cI*(V3[2]+V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[5]-V3[2])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(V3[4]+cI*(V3[3]))+P1[3]*(V3[5]-V3[2]))))+M1*(F2[4]*(V3[3]+cI*(V3[4]))-F2[5]*(V3[2]+V3[5]))));
    F1[4]= denom*cI*(F2[4]*(P1[0]*(V3[5]-V3[2])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(V3[4]-cI*(V3[3]))+P1[3]*(V3[5]-V3[2]))))+(F2[5]*(P1[0]*(V3[3]-cI*(V3[4]))+(P1[1]*-1.*(V3[2]+V3[5])+(P1[2]*(+cI*(V3[2]+V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+M1*(F2[2]*-1.*(V3[2]+V3[5])+F2[3]*(+cI*(V3[4])-V3[3]))));
    F1[5]= denom*-cI*(F2[4]*(P1[0]*-1.*(V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[5])+cI*(V3[2]))+P1[3]*(V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(+cI*(V3[4])-V3[3])+(P1[2]*-1.*(V3[4]+cI*(V3[3]))-P1[3]*(V3[2]+V3[5]))))+M1*(F2[2]*(V3[3]+cI*(V3[4]))+F2[3]*(V3[2]-V3[5]))));
}

"""
        solution2_c="""#include "FFV1C1_2.h"

void FFV1C1_2(complex<double> F2[], complex<double> V3[], complex<double> COUP, double M1, double W1,complex<double> F1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  denom;
    F1[0] = +F2[0]+V3[0];
    F1[1] = +F2[1]+V3[1];
P1[0] = -F1[0].real();
P1[1] = -F1[1].real();
P1[2] = -F1[1].imag();
P1[3] = -F1[0].imag();
    denom = COUP/(pow(P1[0],2)-pow(P1[1],2)-pow(P1[2],2)-pow(P1[3],2) - M1 * (M1 -cI* W1));
    F1[2]= denom*((-cI)*((F2[2])*(((P1[0])*(((V3[2])+(V3[5])))+((P1[1])*(-1.*((V3[3])+cI*((V3[4]))))+((P1[2])*((+cI*((V3[3]))-(V3[4])))-(P1[3])*(((V3[2])+(V3[5])))))))+((F2[3])*(((P1[0])*(((V3[3])+(-cI)*((V3[4]))))+((P1[1])*(((V3[5])-(V3[2])))+((P1[2])*((+(-cI)*((V3[5]))+cI*((V3[2]))))+(P1[3])*((+cI*((V3[4]))-(V3[3])))))))+(M1)*(((F2[4])*(((V3[2])-(V3[5])))+(F2[5])*((+cI*((V3[4]))-(V3[3]))))))));
    F1[3]= denom*(cI*((F2[2])*(((P1[0])*(-1.*((V3[3])+cI*((V3[4]))))+((P1[1])*(((V3[2])+(V3[5])))+((P1[2])*((+cI*((V3[2])+(V3[5]))))-(P1[3])*(((V3[3])+cI*((V3[4]))))))))+((F2[3])*(((P1[0])*(((V3[5])-(V3[2])))+((P1[1])*(((V3[3])+(-cI)*((V3[4]))))+((P1[2])*(((V3[4])+cI*((V3[3]))))+(P1[3])*(((V3[5])-(V3[2])))))))+(M1)*(((F2[4])*(((V3[3])+cI*((V3[4]))))-(F2[5])*(((V3[2])+(V3[5]))))))));
    F1[4]= denom*(cI*((F2[4])*(((P1[0])*(((V3[5])-(V3[2])))+((P1[1])*(((V3[3])+cI*((V3[4]))))+((P1[2])*(((V3[4])+(-cI)*((V3[3]))))+(P1[3])*(((V3[5])-(V3[2])))))))+((F2[5])*(((P1[0])*(((V3[3])+(-cI)*((V3[4]))))+((P1[1])*(-1.*((V3[2])+(V3[5])))+((P1[2])*((+cI*((V3[2])+(V3[5]))))+(P1[3])*(((V3[3])+(-cI)*((V3[4]))))))))+(M1)*(((F2[2])*(-1.*((V3[2])+(V3[5])))+(F2[3])*((+cI*((V3[4]))-(V3[3]))))))));
    F1[5]= denom*((-cI)*((F2[4])*(((P1[0])*(-1.*((V3[3])+cI*((V3[4]))))+((P1[1])*(((V3[2])-(V3[5])))+((P1[2])*((+(-cI)*((V3[5]))+cI*((V3[2]))))+(P1[3])*(((V3[3])+cI*((V3[4]))))))))+((F2[5])*(((P1[0])*(((V3[2])+(V3[5])))+((P1[1])*((+cI*((V3[4]))-(V3[3])))+((P1[2])*(-1.*((V3[4])+cI*((V3[3]))))-(P1[3])*(((V3[2])+(V3[5])))))))+(M1)*(((F2[2])*(((V3[3])+cI*((V3[4]))))+(F2[3])*(((V3[2])-(V3[5]))))))));
}

"""
        amp = builder.compute_routine(2)
        
        routine = amp.write(output_dir=None, language='CPP') 

        split_solution = solution_h.split('\n')
        split_routine = routine[0].split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        split_solution = solution_c.split('\n')
        split_solution2 = solution2_c.split('\n')
        split_routine = routine[1].split('\n')
        for i in range(len(split_routine)):
            try:
                self.assertEqual(split_solution[i], split_routine[i])
            except:
                self.assertEqual(split_solution2[i], split_routine[i])
        self.assertEqual(len(split_routine), len(split_solution))

    @set_global(cms=True)
    def test_short_pythonwriter_complex_mass_scheme(self):
        """ test that python writer works """
        
        solution ="""import cmath
import wavefunctions
def SSS1_1(S2,S3,COUP,M1):
    S1 = wavefunctions.WaveFunction(size=3)
    S1[0] = +S2[0]+S3[0]
    S1[1] = +S2[1]+S3[1]
    P1 = [-complex(S1[0]).real, -complex(S1[1]).real, -complex(S1[1]).imag, -complex(S1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1**2)
    S1[2]= denom*1j * S3[2]*S2[2]
    return S1


import cmath
import wavefunctions
def SSS1_2(S2,S3,COUP,M1):

    return SSS1_1(S2,S3,COUP,M1)
import cmath
import wavefunctions
def SSS1_3(S2,S3,COUP,M1):

    return SSS1_1(S2,S3,COUP,M1)
"""
        
        SSS = UFOLorentz(name = 'SSS1',
                 spins = [ 1, 1, 1 ],
                 structure = '1')        
        builder = create_aloha.AbstractRoutineBuilder(SSS)
        amp = builder.compute_routine(1)
        amp.add_symmetry(2)
        amp.add_symmetry(3)
        
        routine = amp.write(output_dir=None, language='Python')
        
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

    @set_global(cms=True)
    def test_short_F77writer_complex_mass_scheme(self):
        """ test that python writer works """
        
        solution = """subroutine SSS1_1(S2, S3, COUP, M1,S1)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 S3(*)
 real*8 P1(0:3)
 complex*16 S1(3)
 complex*16 denom
 complex*16 COUP
 complex*16 M1
 complex*16 S2(*)
entry SSS1_2(S2, S3, COUP, M1,S1)

entry SSS1_3(S2, S3, COUP, M1,S1)

    S1(1) = +S2(1)+S3(1)
    S1(2) = +S2(2)+S3(2)
P1(0) = -dble(S1(1))
P1(1) = -dble(S1(2))
P1(2) = -dimag(S1(2))
P1(3) = -dimag(S1(1))
    denom = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1**2)
    S1(3)= denom*CI * S3(3)*S2(3)
end




"""
        SSS = UFOLorentz(name = 'SSS1',
                 spins = [ 1, 1, 1 ],
                 structure = '1')        
        builder = create_aloha.AbstractRoutineBuilder(SSS)
        amp = builder.compute_routine(1)
        amp.add_symmetry(2)
        amp.add_symmetry(3)
        
        routine = amp.write(output_dir=None, language='Fortran')
        
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))


    @set_global(cms=True)
    def test_short_Cwriter_complex_mass_scheme(self):
        """ test that python writer works """
        
        assert aloha.complex_mass
        
        solution_h="""#ifndef SSS1_1_guard
#define SSS1_1_guard
#include <complex>
using namespace std;

void SSS1_1(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[]);
void SSS1_2(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[]);
void SSS1_3(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[]);
#endif

"""     
        SSS = UFOLorentz(name = 'SSS1',
                 spins = [ 1, 1, 1 ],
                 structure = '1')        
        builder = create_aloha.AbstractRoutineBuilder(SSS)
        amp = builder.compute_routine(1)
        amp.add_symmetry(2)
        amp.add_symmetry(3)
        
        routine_h, routine_c = amp.write(output_dir=None, language='CPP')
        
        split_solution = solution_h.split('\n')
        split_routine = routine_h.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

        solution_c = """#include "SSS1_1.h"

void SSS1_1(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[])
{
 complex<double> cI = complex<double>(0.,1.);
 double  P1[4];
 complex<double>  denom;
    S1[0] = +S2[0]+S3[0];
    S1[1] = +S2[1]+S3[1];
P1[0] = -S1[0].real();
P1[1] = -S1[1].real();
P1[2] = -S1[1].imag();
P1[3] = -S1[0].imag();
    denom = COUP/(pow(P1[0],2)-pow(P1[1],2)-pow(P1[2],2)-pow(P1[3],2) - pow(M1,2));
    S1[2]= denom*cI * S3[2]*S2[2];
}

void SSS1_2(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[])
{

 SSS1_1(S2,S3,COUP,M1,S1);
}
void SSS1_3(complex<double> S2[], complex<double> S3[], complex<double> COUP, complex<double> M1,complex<double> S1[])
{

 SSS1_1(S2,S3,COUP,M1,S1);
}
"""
        split_solution = solution_c.split('\n')
        split_routine = routine_c.split('\n')
        self.assertEqual(split_solution, split_routine)
        self.assertEqual(len(split_routine), len(split_solution))

    @set_global(unitary=False)
    def test_short_F77writer_feynman(self):
        """ test that python writer works """
        
        solution = """subroutine FFV1_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 F2(*)
 complex*16 V3(6)
 real*8 W3
 real*8 P3(0:3)
 real*8 M3
 complex*16 F1(*)
 complex*16 denom
 complex*16 COUP
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
P3(0) = -dble(V3(1))
P3(1) = -dble(V3(2))
P3(2) = -dimag(V3(2))
P3(3) = -dimag(V3(1))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(3)= denom*-CI*(F2(5)*F1(3)+F2(6)*F1(4)+F2(3)*F1(5)+F2(4)*F1(6))
    V3(4)= denom*-CI*(F2(4)*F1(5)+F2(3)*F1(6)-F2(6)*F1(3)-F2(5)*F1(4))
    V3(5)= denom*-CI*(-CI*(F2(6)*F1(3)+F2(3)*F1(6))+CI*(F2(5)*F1(4)+F2(4)*F1(5)))
    V3(6)= denom*-CI*(F2(6)*F1(4)+F2(3)*F1(5)-F2(5)*F1(3)-F2(4)*F1(6))
end


"""
        solution2 = """subroutine FFV1_3(F1, F2, COUP, M3, W3,V3)
implicit none
 complex*16 CI
 parameter (CI=(0d0,1d0))
 complex*16 F2(*)
 complex*16 V3(6)
 real*8 W3
 real*8 P3(0:3)
 real*8 M3
 complex*16 F1(*)
 complex*16 denom
 complex*16 COUP
    V3(1) = +F1(1)+F2(1)
    V3(2) = +F1(2)+F2(2)
P3(0) = -dble(V3(1))
P3(1) = -dble(V3(2))
P3(2) = -dimag(V3(2))
P3(3) = -dimag(V3(1))
    denom = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI* W3))
    V3(3)= denom*((-CI)*((F2(5))*(F1(3))+(F2(6))*(F1(4))+(F2(3))*(F1(5))+(F2(4))*(F1(6))))
    V3(4)= denom*((-CI)*((F2(4))*(F1(5))+(F2(3))*(F1(6))-(F2(6))*(F1(3))-(F2(5))*(F1(4))))
    V3(5)= denom*((-CI)*(+(-CI)*((F2(6))*(F1(3))+(F2(3))*(F1(6)))+CI*((F2(5))*(F1(4))+(F2(4))*(F1(5)))))
    V3(6)= denom*((-CI)*((F2(6))*(F1(4))+(F2(3))*(F1(5))-(F2(5))*(F1(3))-(F2(4))*(F1(6))))
end


"""
        
        SSS = UFOLorentz(name = 'FFV1',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,1)')        
        builder = create_aloha.AbstractRoutineBuilder(SSS)
        amp = builder.compute_routine(3)
        
        routine = amp.write(output_dir=None, language='Fortran')
        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        try:
            self.assertEqual(split_solution, split_routine)
        except Exception:
            split_solution = solution2.split('\n')
            self.assertEqual(split_solution, split_routine)
            
        self.assertEqual(len(split_routine), len(split_solution))




    def test_short_python_routine_are_exec(self):
        """ check if the python routine can be call """
        
        aloha_lib.KERNEL.clean()
        FFV2 = UFOLorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,\'s1\')*ProjM(\'s1\',1)')
            
        builder = create_aloha.AbstractRoutineBuilder(FFV2)
        builder.apply_conjugation()
        amp = builder.compute_routine(0)
        routine = amp.write(output_dir=None, language='Python')
        
        solution = """import cmath
import wavefunctions
def FFV2C1_0(F2,F1,V3,COUP):
    TMP0 = -1*(F1[2]*(F2[4]*(V3[2]-V3[5])+F2[5]*(+1j*(V3[4])-V3[3]))+F1[3]*(F2[4]*-1*(V3[3]+1j*(V3[4]))+F2[5]*(V3[2]+V3[5])))
    vertex = COUP*-1j * TMP0
    return vertex


""" 

        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution,split_routine)
        self.assertEqual(len(split_routine), len(split_solution))


                 
            
class test_aloha_wavefunctions(unittest.TestCase):
    """ test the python wavefunctions against hardcoded value obtained with 
    the HELAS version (fortran)"""  
    
    def test_short_IR(self):
        """check that spin32 wavefunctions IR returns correct results"""

        import aloha.template_files.wavefunctions as wf


        P = [   500.000000000000      ,   110.924284443833      ,   444.830789488121      ,  -199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =           3
        IC =           -1
        Results = wf.WaveFunction(spin=4)
        Results[           2 ] =  complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           3 ] =  complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           4 ] =  complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           5 ] =  complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           6 ] =  complex( -17.1704816996215     , -6.14297559929837     )
        Results[           7 ] =  complex( -1.18355774685631     ,  11.8924011995262     )
        Results[           8 ] =  complex( 1.717048169962149E-015, 6.142975599298373E-016)
        Results[           9 ] =  complex( 1.183557746856311E-016,-1.189240119952622E-015)
        Results[          10 ] =  complex(  6.14297559929837     , -5.93237215925678     )
        Results[          11 ] =  complex( -4.74632700655791     , -2.96552335078786     )
        Results[          12 ] =  complex(-6.142975599298373E-016, 5.932372159256782E-016)
        Results[          13 ] =  complex( 4.746327006557908E-016, 2.965523350787862E-016)
        Results[          14 ] =  complex(  4.14908109764417     , -16.6387282060841     )
        Results[          15 ] =  complex( -11.2381095403647     ,  0.00000000000000     )
        Results[          16 ] =  complex(-4.149081097644176E-016, 1.663872820608413E-015)
        Results[          17 ] =  complex( 1.123810954036470E-015,  0.00000000000000     )
        Results[          0  ] =  complex( 500.000000000000     ,  -199.552929930879     )
        Results[          1  ] =  complex( 110.924284443833     , 444.830789488121     )


        results = wf.irxxxx(P, M, NHEL, IC)
        
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])

        P = [   500.000000000000      ,   110.924284443833      ,   444.830789488121      ,  -199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =           1
        IC =           -1
        Results[           2 ] = complex( 2.612564516658314E+016,-1.047695860614032E+017)
        Results[           3 ] = complex(-7.076334621694114E+016,  0.00000000000000     )
        Results[           4 ] = complex( -2.61256451665832     ,  10.4769586061403     )
        Results[           5 ] = complex(  7.07633462169412     ,  0.00000000000000     )
        Results[           6 ] = complex( 5.795936991473430E+015,-2.324298273067542E+016)
        Results[           7 ] = complex(-1.569874708793080E+016,-1.047695860614033E-015)
        Results[           8 ] = complex(  5.91714722339943     ,  4.64859654613508     )
        Results[           9 ] = complex( 0.527184900927847     ,  10.4769586061403     )
        Results[          10 ] = complex( 2.324298273067542E+016,-9.320947536407534E+016)
        Results[          11 ] = complex(-6.295543032900642E+016, 2.612564516658316E-016)
        Results[          12 ] = complex( -4.64859654613508     ,  11.5655604511210     )
        Results[          13 ] = complex(  2.11412745966097     , -2.61256451665832     )
        Results[          14 ] = complex(-1.042689807865234E+016, 4.181415573239674E+016)
        Results[          15 ] = complex( 2.824206613860756E+016,  0.00000000000000     )
        Results[          16 ] = complex(-0.527184900927846     ,  2.11412745966097     )
        Results[          17 ] = complex( -12.7247478494156     ,-4.959591483691100E-016)
        Results[          0  ] = complex( 500.000000000000     ,  -199.552929930879     )
        Results[          1  ] = complex( 110.924284443833     , 444.830789488121     )

        results = wf.irxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])
       
        P = [   500.000000000000      ,   110.924284443833      ,   444.830789488121      ,  -199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =          -1
        IC =           -1
        Results[           2 ] = complex(  1.71214570486549     , -6.86608103385749     )
        Results[           3 ] = complex(  10.7977847259741     ,  0.00000000000000     )
        Results[           4 ] = complex(-1.712145704865492E+016, 6.866081033857482E+016)
        Results[           5 ] = complex(-1.079778472597411E+017,  0.00000000000000     )
        Results[           6 ] = complex( -10.0381105772710     , -3.04646050245605     )
        Results[           7 ] = complex(  3.07880047236341     ,  6.86608103385749     )
        Results[           8 ] = complex(-3.798370743515730E+015, 1.523230251228026E+016)
        Results[           9 ] = complex(-2.395473088614451E+016,-6.866081033857490E-016)
        Results[           10 ] = complex(  3.04646050245605     , -1.41919226194686     )
        Results[           11 ] = complex(  12.3466673836539     , -1.71214570486549     )
        Results[          12 ] = complex(-1.523230251228026E+016, 6.108488493960483E+016)
        Results[          13 ] = complex(-9.606374208755685E+016, 1.712145704865494E-016)
        Results[          14 ] = complex( -3.07880047236341     ,  12.3466673836539     )
        Results[          15 ] = complex(  2.17886641065001     , 1.239897870922775E-016)
        Results[          16 ] = complex( 6.833273837489574E+015,-2.740293174898197E+016)
        Results[          17 ] = complex( 4.309459157662050E+016,  2.23360143578611     )
        Results[          0  ] = complex( 500.000000000000     ,  -199.552929930879     )
        Results[          1  ] = complex( 110.924284443833     , 444.830789488121     )

        results = wf.irxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])

        P = [   500.000000000000      ,   110.924284443833      ,   444.830789488121      ,  -199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =          -3
        IC =           -1
        Results[           2 ] = complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           3 ] = complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           4 ] = complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           5 ] = complex(  0.00000000000000     ,  0.00000000000000     )
        Results[           6 ] = complex(-1.182541783415517E-015,-1.729019455726596E-016)
        Results[           7 ] = complex(-1.805991723756818E-016,-1.814662461463105E-015)
        Results[           8 ] = complex(  11.8254178341552     ,  1.72901945572659     )
        Results[           9 ] = complex(  1.80599172375682     ,  18.1466246146310     )
        Results[           10 ] = complex( 1.729019455726595E-016, 5.322822608256632E-016)
        Results[           11 ] = complex(-7.242424220410862E-016, 4.525094480903831E-016)
        Results[          12 ] = complex( -1.72901945572659     , -5.32282260825663     )
        Results[          13 ] = complex(  7.24242422041086     , -4.52509448090383     )
        Results[          14 ] = complex(-2.719102757147013E-016, 1.090420039422019E-015)
        Results[          15 ] = complex(-1.714824044241181E-015,  0.00000000000000     )
        Results[          16 ] = complex(  2.71910275714701     , -10.9042003942202     )
        Results[          17 ] = complex(  17.1482404424118     , 4.959591483691100E-016)
        Results[          0 ] = complex( 500.000000000000     ,  -199.552929930879     )
        Results[          1 ] = complex( 110.924284443833     , 444.830789488121     )

        results = wf.irxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])


    def test_short_OR(self):
        """check that spin32 wavefunctions IR returns correct results"""

        import aloha.template_files.wavefunctions as wf       
        Results = wf.WaveFunction(spin=4)
        
        P = [   500.000000000000      ,  -110.924284443833      ,  -444.830789488121      ,   199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =           3
        IC =           1
        Results[           2 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           3 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           4 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           5 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           6 ] = complex (  1.80599172375682     ,  18.1466246146310     )
        Results[           7 ] = complex ( -11.8254178341552     , -1.72901945572660     )
        Results[           8 ] = complex ( 1.805991723756816E-016, 1.814662461463104E-015)
        Results[           9 ] = complex (-1.182541783415517E-015,-1.729019455726596E-016)
        Results[           10 ] = complex (  7.24242422041086     , -4.52509448090383     )
        Results[           11 ] = complex (  1.72901945572660     ,  5.32282260825663     )
        Results[          12 ] = complex ( 7.242424220410857E-016,-4.525094480903828E-016)
        Results[          13 ] = complex ( 1.729019455726596E-016, 5.322822608256632E-016)
        Results[          14 ] = complex (  17.1482404424118     ,  0.00000000000000     )
        Results[          15 ] = complex ( -2.71910275714701     ,  10.9042003942202     )
        Results[          16 ] = complex ( 1.714824044241180E-015,  0.00000000000000     )
        Results[          17 ] = complex (-2.719102757147013E-016, 1.090420039422019E-015)
        Results[          0 ] = complex (  500.000000000000     ,  199.552929930879     )
        Results[          1 ] = complex ( -110.924284443833     , -444.830789488121     )
        
        
        
        results = wf.orxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])
        
        
        P = [   500.000000000000      ,  -110.924284443833      ,  -444.830789488121      ,   199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =           1
        IC =           1
        Results[           2 ] = complex ( 1.079778472597411E+017,  0.00000000000000     )
        Results[           3 ] = complex (-1.712145704865493E+016, 6.866081033857489E+016)
        Results[           4 ] = complex (  10.7977847259741     ,  0.00000000000000     )
        Results[           5 ] = complex ( -1.71214570486549     ,  6.86608103385749     )
        Results[           6 ] = complex (-2.395473088614451E+016,-6.866081033857491E-016)
        Results[           7 ] = complex ( 3.798370743515733E+015,-1.523230251228028E+016)
        Results[           8 ] = complex ( -3.07880047236341     , -6.86608103385749     )
        Results[           9 ] = complex ( -10.0381105772710     , -3.04646050245606     )
        Results[           10 ] = complex (-9.606374208755686E+016, 1.712145704865494E-016)
        Results[           11 ] = complex ( 1.523230251228027E+016,-6.108488493960490E+016)
        Results[          12 ] = complex ( -12.3466673836539     ,  1.71214570486549     )
        Results[          13 ] = complex (  3.04646050245605     , -1.41919226194687     )
        Results[          14 ] = complex ( 4.309459157662049E+016,-2.753126328964350E-032)
        Results[          15 ] = complex (-6.833273837489577E+015, 2.740293174898198E+016)
        Results[          16 ] = complex ( -2.17886641065001     ,  0.00000000000000     )
        Results[          17 ] = complex ( -3.07880047236341     ,  12.3466673836539     )
        Results[          0 ] = complex (  500.000000000000     ,  199.552929930879     )
        Results[          1 ] = complex ( -110.924284443833     , -444.830789488121     )
        
        
        results = wf.orxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])        
        
        
        P = [   500.000000000000      ,  -110.924284443833      ,  -444.830789488121      ,   199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =          -1
        IC =           1
        Results[           2 ] = complex ( -7.07633462169412     ,-2.479795741845550E-016)
        Results[           3 ] = complex ( -2.61256451665831     ,  10.4769586061403     )
        Results[           4 ] = complex (-7.076334621694121E+016, -4.46720287157222     )
        Results[           5 ] = complex (-2.612564516658315E+016, 1.047695860614032E+017)
        Results[           6 ] = complex ( 0.527184900927848     ,  10.4769586061403     )
        Results[           7 ] = complex ( -5.91714722339944     , -4.64859654613509     )
        Results[           8 ] = complex ( 1.569874708793082E+016,  1.11680071789306     )
        Results[           9 ] = complex ( 5.795936991473431E+015,-2.324298273067542E+016)
        Results[           10 ] = complex (  2.11412745966097     , -2.61256451665831     )
        Results[           11 ] = complex (  4.64859654613508     , -11.5655604511209     )
        Results[          12 ] = complex ( 6.295543032900648E+016,  2.23360143578611     )
        Results[          13 ] = complex ( 2.324298273067542E+016,-9.320947536407533E+016)
        Results[          14 ] = complex ( -12.7247478494156     ,-1.239897870922775E-016)
        Results[          15 ] = complex ( 0.527184900927848     , -2.11412745966098     )
        Results[          16 ] = complex (-2.824206613860757E+016, -1.11680071789306     )
        Results[          17 ] = complex (-1.042689807865234E+016, 4.181415573239673E+016)
        Results[          0 ] = complex (  500.000000000000     ,  199.552929930879     )
        Results[          1 ] = complex ( -110.924284443833     , -444.830789488121     )
        
        results = wf.orxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])        
        
        P = [   500.000000000000      ,  -110.924284443833      ,  -444.830789488121      ,   199.552929930879      ]
        M =  1.000000000000000E-013
        NHEL =          -3
        IC =           1
        Results[           2 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           3 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           4 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           5 ] = complex (  0.00000000000000     ,  0.00000000000000     )
        Results[           6 ] = complex ( 1.183557746856311E-016,-1.189240119952622E-015)
        Results[           7 ] = complex (-1.717048169962148E-015,-6.142975599298370E-016)
        Results[           8 ] = complex (  1.18355774685631     , -11.8924011995262     )
        Results[           9 ] = complex ( -17.1704816996215     , -6.14297559929837     )
        Results[           10 ] = complex ( 4.746327006557909E-016, 2.965523350787862E-016)
        Results[           11 ] = complex ( 6.142975599298370E-016,-5.932372159256777E-016)
        Results[          12 ] = complex (  4.74632700655791     ,  2.96552335078786     )
        Results[          13 ] = complex (  6.14297559929837     , -5.93237215925678     )
        Results[          14 ] = complex ( 1.123810954036471E-015, 5.506252657928701E-032)
        Results[          15 ] = complex ( 4.149081097644174E-016,-1.663872820608412E-015)
        Results[          16 ] = complex (  11.2381095403647     ,  0.00000000000000     )
        Results[          17 ] = complex (  4.14908109764417     , -16.6387282060841     )
        Results[          0 ] = complex (  500.000000000000     ,  199.552929930879     )
        Results[          1 ] = complex ( -110.924284443833     , -444.830789488121     )
        
        
        results = wf.orxxxx(P, M, NHEL, IC)
        for i in range(18):
            self.assertAlmostEqual(results[i], Results[i])

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
"""Unit test Library for testing the Creation of Helas Amplitude created from 
the output of the Feynman Rules."""
from __future__ import division

import math
import os
import time
import aloha.aloha_object as aloha_obj
import aloha.aloha_lib as aloha_lib
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
import models.sm.object_library as object_library
import tests.unit_tests as unittest


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.var1 = aloha_lib.Variable(2, 'var1')
        self.var2 = aloha_lib.Variable(3, 'var2')
        self.var3 = aloha_lib.Variable(11, 'var3')
        
    
    def test_power(self):
        """check that the power is correctly update"""
        
        
        a = aloha_lib.ScalarVariable('P3_0')
        b = a ** 2 * a **2

        b = b.simplify()
        self.assertEqual(b.power,4)
    
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
        for term in sum:
            if term == self.var1:
                self.assertEqual(term.prefactor, 2)
                self.assertFalse(term is self.var1)
            elif term == self.var2:
                self.assertEqual(term.prefactor, 3)
                self.assertFalse(term is self.var2)
                
        self.assertEquals(self.var1.prefactor, 2) #prefactor is preserve
        self.assertEquals(self.var2.prefactor, 3)   
         
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
        self.assertTrue(self.var1 in sum)
        for term in sum:
            if term == self.var1:
                self.assertEqual(term.prefactor, 2)
                self.assertFalse(term is self.var1)
            elif term == self.var2:
                self.assertEqual(term.prefactor, 3)
                self.assertFalse(term is self.var2)
                                  
        self.assertEquals(self.var1.prefactor,2) #prefactor is preserve
        self.assertEquals(self.var2.prefactor,3)   
 
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
        for data in sum:
            if data == self.var3:
                self.assertFalse(data is self.var3)
            else:
                self.assertTrue(data is self.var1 or data is self.var2)
                    
                
        #test prefactor- constant_term
        self.assertEquals(sum.prefactor, 1)
        self.assertEquals(self.var1.prefactor,2)
        
    def testsumvarmult(self):
        """ test the sum of one Variable with an MultVariable"""        
        
        mult = aloha_lib.MultVariable()
        mult.append(self.var1)
        mult.append(self.var2) 
        sum = self.var3 + mult
        
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(self.var3 in sum)
        self.assertEquals(len(sum), 2)
        for data in sum:
            if data == self.var3:
                self.assertFalse(data is self.var3)
                self.assertEqual(data.prefactor, self.var3.prefactor)
                
        #test prefactor- constant_term
        self.assertEquals(sum.prefactor, 1)
        self.assertEquals(self.var3.prefactor, 11)
         
    def testmultvarvar(self):
        """product of Two Variable"""
        
        prod = self.var1 * self.var2
        
        #check sanity
        self.assertEquals(prod.__class__,aloha_lib.MultVariable)
        self.assertTrue(self.var1 in prod) # presence of single term        
        self.assertTrue(self.var2 in prod) # presence of single term
        self.assertEquals(len(prod),2)
        
        
        self.assertEquals(prod.prefactor,6)


    def testmultvarAdd(self):
        """product of Variable with an AddVariable"""
        
        add = self.var1 + self.var2
        prod = self.var3 * add
        #sanity check
        self.assertEquals(prod.__class__, aloha_lib.AddVariable)
        self.assertEquals(len(prod), 2)
        
        #check prefactor of each term
        for term in prod:
            if self.var1 not in term:
                self.assertEquals(term.prefactor, 33)
            elif self.var2 not in term:
                self.assertEquals(term.prefactor, 22)
            else:
                raise Exception('not valid term')
                
    
    def testmultvarMult(self):
        """product of Variable with an MultVariable"""
        
        var1 = aloha_lib.Variable(2)
        var2 = aloha_lib.Variable(3,'y')
        mult = aloha_lib.MultVariable()
        mult.append(var1)
        mult.append(var2)
        
        prod = self.var1 * mult
        
        #Sanity
        self.assertEquals(prod.__class__, aloha_lib.MultVariable)
        self.assertEquals(len(prod), 3)
        
        #check prefactor
        self.assertEquals(prod.prefactor, 12)
        
               
    def testmultvarint(self):
        """product of Var with an integer"""
        
        prod1 = self.var1 * 2
        prod2 = 2 * self.var2

        #SanityCheck
        self.assertTrue(prod1, aloha_lib.Variable)
        self.assertTrue(prod2, aloha_lib.Variable)
        self.assertEquals(prod1, self.var1)
        self.assertEquals(prod2, self.var2)
        self.assertFalse(prod1 is self.var1)
        self.assertFalse(prod2 is self.var2)
        
        #check prefactor - constant term
        self.assertEquals(prod1.prefactor, 4)
        self.assertEquals(prod2.prefactor, 6)

class TestAddVariable(unittest.TestCase):

    def setUp(self):
        """Initialize basic object"""
        self.var1 = aloha_lib.Variable(2, 'var1')
        self.var2 = aloha_lib.Variable(3, 'var2')
        self.add1 = aloha_lib.AddVariable()
        self.add1.append(self.var1)
        self.add1.append(self.var2)

        self.var3 = aloha_lib.Variable(11, 'var3')
        self.var4 = aloha_lib.Variable(4, 'var4')
        self.add2 = aloha_lib.AddVariable()
        self.add2.append(self.var3)
        self.add2.append(self.var4)        
    
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
                self.assertEqual(type(term), aloha_lib.ConstantObject)
                self.assertEqual(term.value, 5)
            
        return
                
    def testsumaddmult(self):
        """Test the sum of an AddVariable with a MultVariable."""
        
        var1 = aloha_lib.Variable(2)
        var2 = aloha_lib.Variable(3)
        mult = aloha_lib.MultVariable()
        mult.append(var1)
        mult.append(var2)
        mult.constant_term =2
                
        sum = self.add1 + mult
        
        #Sanity Check
        self.assertEquals(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 3)
        self.assertTrue(mult in sum)
        
        #check new term 
        for term in sum:
            if term.__class__ == aloha_lib.AddVariable:
                self.assertTrue(term.prefactor, 6)
                self.assertTrue(term.constant_term, 0)
                
    def testsumaddvar(self):
        """Test the sum of an AddVariable with a Variable."""
        
        var3 = aloha_lib.Variable(11, 'var3')
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
        
        self.assertEquals(prod1.prefactor, 1)
        self.assertEquals(prod2.prefactor, 1)
                
        for data in prod1:
            if data == self.var1:
                self.assertEquals(data.prefactor, 6)
            elif data == self.var2:
                self.assertEquals(data.prefactor, 9)
        for data in prod2:
            if data == self.var3:
                self.assertEquals(data.prefactor, 22)
            elif data == self.var4:
                self.assertEquals(data.prefactor, 8)

    
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
        self.assertEqual(new_sum.prefactor, 1)
        for term in new_sum:
            self.assertEqual(term.prefactor, 2)
            if term.__class__ == aloha_obj.Metric:
                self.assertFalse(var3 is term)
            else:
                self.assertEqual(term.__class__, aloha_lib.MultLorentz)
                self.assertEqual(prod, term)
                self.assertFalse(prod is term) 
    
    def testmultaddvar(self):
        """Test the multiplication of an Addvariable with a Variable"""
        
        var3 = aloha_lib.Variable(11, 'var3')
        prod = self.add1 * var3
        #sanity check
        self.assertEquals(prod.__class__, aloha_lib.AddVariable)
        self.assertEquals(len(prod), 2)
        
        #check prefactor of each term
        for term in prod:
            if self.var1 not in term:
                self.assertEquals(term.prefactor, 33)
            elif self.var2 not in term:
                self.assertEquals(term.prefactor, 22)
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
            self.assertTrue(p3 in term)
            self.assertEqual(term.__class__,aloha_obj.P.mult_class)
        
        
        
        
    def testmultaddmult(self):
        """Test the multiplication of an AddVariable with a MultVariable."""
        
        var3 = aloha_lib.Variable(2, 'var3')
        var4 = aloha_lib.Variable(1, 'var4')
        prod = self.add1 * (var3 *var4)
        
        self.assertEqual(prod.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(prod), 2)
        
        for data in prod:
            if self.var1 in data:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 4)
            else:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 6)
        self.assertEqual(prod.prefactor, 1)
        
                
    def testmultaddadd(self):
        """Test the multiplication between two AddVariable."""
        
        prod = self.add1 * self.add2
        self.assertEqual(prod.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(prod), 4)
        
        for data in prod:
            if self.var1 in data and self.var3 in data:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 22)
            elif self.var1 in data:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 8)
            elif self.var2 in data and self.var3 in data:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 33)
            else:
                self.assertEqual(data.__class__, aloha_lib.MultVariable)
                self.assertEqual(data.prefactor, 12)
        

    def test_factorization(self):
        """test the factorization"""
        
        p1 = aloha_lib.ScalarVariable('p1')
        p2 = aloha_lib.ScalarVariable('p2')        
        p3 = aloha_lib.ScalarVariable('p3')
        p4 = aloha_lib.ScalarVariable('p4')
        p5 = aloha_lib.ScalarVariable('p5')
        
        
        sum = p1 * p2 + p1 * p3
        sum = sum.factorize()
        self.assertEqual(sum.__class__,aloha_lib.MultVariable)
        self.assertEqual(len(sum),2)
        for fact in sum:
            if isinstance(fact, aloha_lib.Variable):
                self.assertEqual(fact, p1)
            else:
                self.assertEqual(fact, p3 + p2) 
        
        
        sum = p1 * p2 + p1 * p3 + 2 * p1 + 2 *p1 * p2 * p4
        sum = sum.factorize()
        #Should return p1*(p2(2*p4 + 1) + p3 + 2)
        self.assertEqual(sum.__class__,aloha_lib.MultVariable)
        self.assertEqual(len(sum),2)
        for fact in sum:
            if isinstance(fact, aloha_lib.Variable):
                self.assertEqual(fact, p1)
            else:
                self.assertEqual(fact.__class__,aloha_lib.AddVariable)
                self.assertEqual(len(fact), 2)
                for term in fact:
                    if isinstance(term, aloha_lib.AddVariable):
                        self.assertEqual(term[0], p3)
                        self.assertEqual(term[1], aloha_lib.ConstantObject(2))
                    else:
                        self.assertEqual(term.__class__, aloha_lib.MultVariable)
                        self.assertTrue(p2 in term)
                        self.assertTrue(len(term),2)
                        for data in term:
                            if data == p2: 
                                pass
                            else:
                                self.assertEqual(data.__class__, aloha_lib.AddVariable)
                                self.assertTrue(p4 in data)
                                self.assertTrue(aloha_lib.ConstantObject(1) in data)
                                for term2 in data:
                                    if term2 == p4:
                                        self.assertEqual(term2.prefactor,2)
                                    else:
                                        self.assertEqual(term2, \
                                                    aloha_lib.ConstantObject(1))
                                        
    def test_factorization2(self):
        """test the factorization with power and constant"""
        
        p1 = aloha_lib.ScalarVariable('p1')
        p2 = aloha_lib.ScalarVariable('p2')        
        p3 = aloha_lib.ScalarVariable('p3')
                
        sum = ( -2 * p1 **2 + -2 * p2 + 2 * ( p3 * p2 ) )
        sum = sum.factorize()
        #Should return p2*(2*p3-2)-2*p1**2
         
        self.assertEqual(sum.__class__,aloha_lib.AddVariable)
        self.assertEqual(len(sum),2)
        for term in sum:
            if term == p1:
                self.assertEqual(term.power, 2)
                self.assertEqual(term.prefactor, -2)
                continue
            self.assertEqual(term.__class__,aloha_lib.MultVariable)
            self.assertEqual(len(term), 2)
            for fact in term:
                if fact == p2:
                    self.assertEqual(fact.power, 1)
                    self.assertEqual(fact.prefactor, 1)
                    continue
                self.assertEqual(fact.__class__,aloha_lib.AddVariable)
                self.assertEqual(len(fact), 2)
                for term2 in fact:
                    if term2 == p3:
                        self.assertEqual(term2.power, 1)
                        self.assertEqual(term2.prefactor, 2)
                        continue
                    self.assertEqual(term2.__class__, aloha_lib.ConstantObject)
                    self.assertEqual(term2, aloha_lib.ConstantObject(-2) )

    def test_factorization3(self):
        """test factorization with prefactor"""
        
        p1 = aloha_lib.ScalarVariable('p1')
        p2 = aloha_lib.ScalarVariable('p2')
        
        sum =2 * p2**2 + 2* p1 * p2
        sum = sum.factorize()
        #should be p2 (2 * p1 + 2 * p2)
        self.assertEqual(sum.__class__,aloha_lib.MultVariable)
        self.assertEqual(len(sum),2)
        for fact in sum:
            if p2 == fact:
                self.assertEqual(fact.prefactor, 1)
                self.assertEqual(fact.power, 1)
            else:
                self.assertEqual(sum.__class__,aloha_lib.MultVariable)
                self.assertEqual(len(sum),2)
                for term in fact:
                    self.assertEqual(term.prefactor, 2)
                    self.assertEqual(term.power, 1)
    
    def test_factorization4(self):
        """test the factorization with constant factor"""
        
        P1_0 = aloha_lib.ScalarVariable('p1')
        P1_1 = aloha_lib.ScalarVariable('p2')
        P1_2 = aloha_lib.ScalarVariable('p3')
        P1_3 = aloha_lib.ScalarVariable('p4')        
        OM1  = aloha_lib.ScalarVariable('om1') 
        
        expr1 = ( -1j * ( P1_3 * P1_1 * OM1 ) + 1j * ( P1_0**2 * P1_3 * P1_1 * OM1**2 ) + -1j * ( P1_1**3 * P1_3 * OM1**2 ) + -1j * ( P1_2**2 * P1_3 * P1_1 * OM1**2 ) + -1j * ( P1_3**3 * P1_1 * OM1**2 ) )

        p1, p2, p3, p4, om1 = 1,2,3,4,5
        value = eval(str(expr1))
        
        expr1 = expr1.factorize()
        self.assertEqual(eval(str(expr1)), value)


    def test_factorization5(self):
        """check that P [gamma + P/M] == (/p+M) [Onshell]"""

        P1_0 = aloha_lib.ScalarVariable('p1')
        P1_1 = aloha_lib.ScalarVariable('p2')
        P1_2 = aloha_lib.ScalarVariable('p3')
        P1_3 = aloha_lib.ScalarVariable('p4')        
        M1  = aloha_lib.ScalarVariable('m1') 
    
        p1, p2, p3, p4, m1 = 1,2,3,4,5
    
        data = (P1_0**2 * M1 - P1_1**2 * M1 + M1)
        value = eval(str(data))
        data2 = data.factorize()
        self.assertEqual(eval(str(data2)), value)
        
        #check that original object is still un-touched
        self.assertEqual(eval(str(data)), value)



    




    
class TestMultVariable(unittest.TestCase):

    def setUp(self):
        self.var1 = aloha_lib.Variable(2, 'var1')
        self.var2 = aloha_lib.Variable(3, 'var2')
        self.var3 = aloha_lib.Variable(4, 'var3')
        self.var4 = aloha_lib.Variable(5, 'var4')
        
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
            if self.var1 in term:
                self.assertEqual(term.prefactor, 6)
                self.assertFalse(term is self.mult1)
            else:
                self.assertEqual(term.prefactor, 20)
                self.assertFalse(term is self.mult2)
                
        sum =  self.mult1 - self.mult1
        sum = sum.simplify()
        self.assertEqual(sum.__class__, aloha_lib.ConstantObject)
        self.assertEqual(len(sum),0)
        self.assertEqual(sum.prefactor, 1)
        
    def testdealingwithpower1(self):
        """Check that the power is correctly set in a product"""
        
        p1 = aloha_lib.ScalarVariable('p1')
        p2 = aloha_lib.ScalarVariable('p2')
        
        prod = p1 * p1
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)       
        prod = prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.ScalarVariable)
        self.assertEqual(prod.power, 2)
        self.assertEqual(p1.power, 1)
        
        prod *= p1
        prod = prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.ScalarVariable)
        self.assertEqual(prod.power, 3)
        self.assertEqual(p1.power, 1)
        
        prod *= p2
        prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(prod[0].power, 3)
        self.assertEqual(prod[1].power, 1)       
        self.assertEqual(p1.power, 1)
        self.assertEqual(p2.power, 1)                                
        
        prod *= p1
        prod.simplify()
        self.assertEqual(prod.__class__, aloha_lib.MultVariable)
        self.assertEqual(prod[0].power, 4)
        self.assertEqual(prod[1].power, 1)       
        self.assertEqual(p1.power, 1)
        self.assertEqual(p2.power, 1)                                   
                                
    def testdealingwithpower2(self):
        """Check that the power is correctly set in a product"""       
        
        p1 = aloha_lib.ScalarVariable('p1', [])
        p2 = aloha_lib.ScalarVariable('p2', [])
        p3 = aloha_lib.ScalarVariable('p3', [])
        p4 = aloha_lib.ScalarVariable('p2', [])
        p5 = aloha_lib.ScalarVariable('p5', [])
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
        
    
    def test_obj_are_not_modified(self):
        """Check that a sum-product-... doesn't change part of the objects"""
        
        sum = self.mult1 + self.mult2
        for term in sum:
            self.assertFalse(term is self.mult1)
            self.assertFalse(term is self.mult2)
            
        
        sum2 = sum - (self.mult1 + self.mult2)
        #for term in sum:
        #    for term2 in sum2:
        #        self.assertFalse(term is term2)
        
        sum2 = sum2.simplify()
        
        #check that sum2 is zero
        self.assertEqual(len(sum2), 0)
        self.assertEqual(sum2.__class__, aloha_lib.ConstantObject)
        self.assertEqual(sum2, 0)       
        
        #check that the sum is not modify in this game      
        self.assertEqual(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 2)
        self.assertEqual(sum.prefactor, 1)
        
        for data in sum:
            self.assertEqual(len(data), 2)
            if self.var1 in data:
                self.assertEqual(data.prefactor, 6)
                self.assertTrue(self.var2 in data)
            else:
                self.assertEqual(data.prefactor, 20)
                self.assertTrue(self.var3 in data)
                self.assertTrue(self.var4 in data)
            
    def testsummultint(self):
        """Test the sum of a MultVariable object with a number"""
        
        add = self.mult1 + 2
        self.assertEqual(add.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(add), 2)
        for term in add:
            if term.__class__ == aloha_lib.MultVariable:
                self.assertEqual(term.prefactor, 6)
                self.assertEqual(len(term), 2)
                self.assertFalse(term is self.mult1)
            else:
                self.assertEqual(term.__class__, aloha_lib.ConstantObject)
                self.assertEqual(term.value, 2)
        
        return
        
    def testsummultadd(self):
        """Test the sum of an MultVariable with a AddVariable."""
        
        var1 = aloha_lib.Variable(2,'xxx')
        var2 = aloha_lib.Variable(3,'yyy')
        add = var1 + var2
                
        sum = self.mult2 + add
        #Sanity Check
        self.assertEquals(sum.__class__, aloha_lib.AddVariable)
        self.assertEqual(len(sum), 3)
        self.assertTrue(var1 in sum)
        self.assertTrue(var2 in sum)
        
        #check new term 
        for term in sum:
            if term.__class__ == aloha_lib.MultVariable:
                self.assertEqual(term.prefactor, 20)
                self.assertTrue(self.var3 in term)
                self.assertTrue(self.var4 in term)
            elif term == var1:
                self.assertEqual(term.prefactor, 2)
            elif term == var2:
                self.assertEqual(term.prefactor, 3)
                
            self.assertEqual(sum.prefactor, 1)
            
    def testsummulvar(self):
        """Test the sum of a MultVariable with a Variable"""
        
        
        var = aloha_lib.Variable(3,'xxx')
        sum = self.mult2 + var
        
        self.assertEquals(sum.__class__,aloha_lib.AddVariable)
        self.assertTrue(var in sum)
        self.assertEquals(len(sum), 2)
        for term in sum:
            if term == var:
                self.assertEquals(term.prefactor, 3)
                self.assertFalse(term is var)
            else:
                self.assertTrue(self.var3 in term)
                self.assertTrue(self.var4 in term)
                self.assertEqual(term.prefactor, 20)
                
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
        self.assertTrue(self.var1 in prod1)
        self.assertTrue(self.var2 in prod1)
        self.assertTrue(self.var3 in prod1)
        self.assertTrue(self.var4 in prod1)        
        self.assertEqual(prod1.prefactor, 120)

        for fact in prod1:
            self.assertEqual(fact.prefactor, 1)
        
        
                
class TestFracVariable(unittest.TestCase):
    """ Class to test the Operation linked to a FracVariable """
    
    def setUp(self):
        """ some building block """
        
        self.p1 = aloha_obj.P(1,2)
        self.p2 = aloha_obj.P(1,3)
        self.mass1 = aloha_obj.Mass(2)
        self.mass2 = aloha_obj.Mass(3)
        self.frac1 = aloha_lib.FracVariable(self.p1, self.mass1)
        self.frac2 = aloha_lib.FracVariable(self.p2, self.mass2)
        
    def testcreation(self):
        """ test if we can create FracVariable Object with division"""
        
        #
        # First check the creation at Lorentz Object
        #
        frac1= self.p1 / self.mass1
        self.assertEqual(frac1.__class__, aloha_lib.FracVariable)
        self.assertEqual(frac1, self.frac1)
        # Verif that the object are different
        self.assertFalse(frac1.numerator is self.p1)
        self.assertFalse(frac1.denominator is self.mass1)
        
        sum = self.p1 +self.p2
        frac2 = sum / self.mass1
        self.assertEqual(frac2.__class__, aloha_lib.FracVariable)        
        self.assertEqual(frac2.numerator, sum)
        self.assertEqual(frac2.denominator, self.mass1)
        # Verif that the object are different
        self.assertFalse(frac2.numerator is sum)
        self.assertFalse(frac2.denominator is self.mass1)
        
        prod = self.p1 * self.p2
        frac3 = prod / self.mass1
        self.assertEqual(frac3.__class__, aloha_lib.FracVariable)        
        self.assertEqual(frac3.numerator, prod)
        self.assertEqual(frac3.denominator, self.mass1)        
        # Verif that the object are different
        self.assertFalse(frac3.numerator is prod)
        self.assertFalse(frac3.denominator is self.mass1)
        
        frac4 = 2 / self.mass1
        self.assertEqual(frac4.__class__, aloha_lib.FracVariable)
        self.assertTrue(isinstance(frac4.numerator, int))
        self.assertEqual(frac4.numerator, 2)
        self.assertTrue(isinstance(frac4.denominator,aloha_lib.Variable))
        self.assertEqual(frac4.denominator, self.mass1)
        self.assertFalse(frac4.denominator is self.mass1)
        
        sum = (self.mass1 + self.mass2)
        frac4 = 2 / sum
        self.assertEqual(frac4.__class__, aloha_lib.FracVariable)
        self.assertTrue(isinstance(frac4.numerator, int))
        self.assertEqual(frac4.numerator, 2)
        self.assertTrue(isinstance(frac4.denominator,aloha_lib.AddVariable))
        self.assertEqual(frac4.denominator, sum)
        self.assertFalse(frac4.denominator is sum)        
        
        prod = self.mass1 * self.mass2
        frac4 = 2 / prod
        self.assertEqual(frac4.__class__, aloha_lib.FracVariable)
        self.assertTrue(isinstance(frac4.numerator, int))
        self.assertEqual(frac4.numerator, 2)
        self.assertTrue(isinstance(frac4.denominator,aloha_lib.MultVariable))
        self.assertTrue(isinstance(frac4.denominator,aloha_lib.MultLorentz))
        self.assertEqual(frac4.denominator, prod)
        self.assertFalse(frac4.denominator is prod)  
        

    def testmultiplacation(self):
        """Frac Variable can be multiply by any object"""
        
        #product with Variable
        prod1 = self.frac1 * self.p1
        self.assertEqual(prod1.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod1.numerator, self.p1 * self.p1)
        self.assertEqual(prod1.denominator, self.mass1)
        
        prod2 = self.p1 * self.frac1
        self.assertEqual(prod2.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod2.numerator, self.p1 * self.p1)
        self.assertEqual(prod2.denominator, self.mass1)        
    
        #product with MultVariable
        prod = self.p1 * self.p2
        prod2 = prod * self.frac1
        self.assertEqual(prod2.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod2.numerator, self.p1 * self.p2 * self.p1)
        self.assertEqual(prod2.denominator, self.mass1)          
        
        prod3 = self.frac1 * prod
        self.assertEqual(prod3.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod3.numerator, self.p1 * self.p2 * self.p1)
        self.assertEqual(prod3.denominator, self.mass1)
        
        # Product with SumVariable
        sum = self.p1 +self.p2
        prod2 = sum * self.frac1
        self.assertEqual(prod2.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod2.numerator, sum * self.p1)
        self.assertEqual(prod2.denominator, self.mass1) 
        
        prod3 = self.frac1 * sum
        self.assertEqual(prod3.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod3.numerator, sum * self.p1)
        self.assertEqual(prod3.denominator, self.mass1) 
               
               
        # Product with FracVariable
        prod = self.frac1 * self.frac2
        self.assertEqual(prod.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod.numerator.__class__, aloha_lib.MultLorentz)
        self.assertEqual(prod.numerator, self.p2 * self.p1)
        self.assertEqual(prod.denominator, self.mass1 * self.mass2)
               
        prod3 = self.frac2 * self.frac1
        self.assertEqual(prod3.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod3.numerator, self.p2 * self.p1)
        self.assertEqual(prod3.denominator, self.mass1 * self.mass2)       
        
    
    def testdivision(self):
        """ Test division with a FracVariable """ 
        
        #divide with Variable
        prod1 = self.frac1 / self.p1
        self.assertEqual(prod1.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod1.numerator, self.p1)
        self.assertEqual(prod1.denominator, self.p1 * self.mass1)
        
        #divide with a MultVariable
        prod= self.p1 * self.p2
        prod3 = self.frac1 / prod
        self.assertEqual(prod3.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod3.numerator, self.p1)
        self.assertEqual(prod3.denominator, self.mass1 * self.p1 * self.p2)
        
        # divide with AddVariable
        sum = self.p1 +self.p2
        prod2 = self.frac1 / sum
        self.assertEqual(prod2.__class__, aloha_lib.FracVariable)
        self.assertEqual(prod2.numerator, self.p1)
        self.assertEqual(prod2.denominator, sum * self.mass1) 
        

        
        
class testLorentzObject(unittest.TestCase):
    """ Class to test the Operation linked to a Lorentz Object"""
    
    def setUp(self):
        
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
    
    def test_power(self):
        """ Test that we can take a square of an object --fully auto contracted"""
        
        product = self.p2 * self.p2
        power = self.p2**2

        self.assertEqual(power.__class__, aloha_lib.MultLorentz)        
        self.assertEqual(product, power)
        power = power.expand()

        keys= power.keys()
        keys.sort()
        self.assertEqual(keys, [(0,)])
        solution = '( ( P3_0**2 ) + -1 * ( P3_1**2 ) + -1 * ( P3_2**2 ) + -1 * ( P3_3**2 ) )'
        self.assertEqual(str(power[(0,)]), solution)
        
    def test_equality(self):
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
        
        #Check that sum indices are not consider for equality
        new3 = aloha_obj.Gamma(3,2,3) * aloha_obj.P(3,2)
        self.assertEqual(new, new3)
        
        new4 = aloha_obj.P(3,2) * aloha_obj.Gamma(3,2,3)
        self.assertEqual(new, new4)
        self.assertEqual(new3, new4)
        
        new5 = aloha_obj.P(4,2) * aloha_obj.Gamma(4,2,3)
        self.assertEqual(new, new5)
        self.assertEqual(new3, new5)
        self.assertEqual(new4, new5)
        
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
        self.assertEqual(new, new3)
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
        self.assertEqual(low_level[(0,)], aloha_lib.ScalarVariable('P2_0',[2]))
        self.assertEqual(low_level[(1,)], aloha_lib.ScalarVariable('P2_1',[2]))
        self.assertEqual(low_level[(2,)], aloha_lib.ScalarVariable('P2_2',[2]))
        self.assertEqual(low_level[(3,)], aloha_lib.ScalarVariable('P2_3',[2]))

        
        #expand a product
        obj = aloha_obj.P(1,2) * aloha_obj.P(2,3)
        low_level = obj.expand()
        
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind), \
                             aloha_lib.ScalarVariable('P2_%s' % ind[1]) * \
                             aloha_lib.ScalarVariable('P3_%s' % ind[0]))
        
        #expand a sum
        obj = aloha_obj.P(1,2) + aloha_obj.P(1,3)
        low_level = obj.expand()
        
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind), \
                             aloha_lib.ScalarVariable('P2_%s' % ind[0]) + \
                             aloha_lib.ScalarVariable('P3_%s' % ind[0]))
            
        #expand zero
        obj = aloha_obj.P(1,2) - aloha_obj.P(1,2)
        obj = obj.simplify()
        low_level = obj.expand()
        pass_in_check = 0
        for ind in low_level.listindices():
            pass_in_check += 1
            self.assertEqual(low_level.get_rep(ind), 0)
        self.assertEqual(pass_in_check, 1)      
             
        #expand zero without first simplification
        obj = aloha_obj.P(1,2) - aloha_obj.P(1,2)
        low_level = obj.expand().simplify()
        pass_in_check = 0 
        for ind in low_level.listindices():
            pass_in_check += 1
            self.assertEqual(low_level.get_rep(ind), 0)
        self.assertEqual(pass_in_check, 4)  
        
        #expand standard frac variable
        obj = aloha_obj.P(1,2) / aloha_obj.P(1,2)   
        obj = obj.expand()
        result = {(0,): aloha_lib.ScalarVariable('P2_0',[2]), \
                                    (1,): aloha_lib.ScalarVariable('P2_1',[2]), \
                                    (2,): aloha_lib.ScalarVariable('P2_2',[2]), \
                                    (3,): aloha_lib.ScalarVariable('P2_3',[2])}
        for i in range(3):
            self.assertEqual(result[tuple([i])], obj.numerator[tuple([i])])
            self.assertEqual(result[tuple([i])], obj.denominator[tuple([i])])
        
        
        #expand standard frac variable with number numerator
        obj = 1 / aloha_obj.P(1,2)   
        obj = obj.expand()
        self.assertEqual(obj.numerator, 1)    
        for i in range(3):
            self.assertEqual(result[tuple([i])], obj.denominator[tuple([i])])        
        
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
            self.assertEqual(low_level.get_rep(ind).__class__, aloha_lib.ScalarVariable)
            self.assertEqual(low_level.get_rep(ind), aloha_lib.ScalarVariable('M3',[3])) 
                                
        obj= aloha_obj.Mass(3) * aloha_obj.P(1,2)
        obj = obj.simplify()
        low_level = obj.expand()
        self.assertEqual(low_level.__class__, aloha_lib.LorentzObjectRepresentation)
        for ind in low_level.listindices():
            self.assertEqual(low_level.get_rep(ind).__class__, aloha_lib.MultVariable)
            self.assertEqual(low_level.get_rep(ind), aloha_lib.ScalarVariable('M3') 
                                * aloha_lib.ScalarVariable('P2_%s' % ind[0]))
    
    
    def test_spin32propagator(self):
        """check various property of the spin3/2 propagator"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        Gamma = aloha_obj.Gamma
        Identity = aloha_obj.Identity
        t = 1
        mu, nu, s0, s1, s2 = 2,3,4,5,6
        
        zero = P(mu,t) * aloha_obj.Spin3halfPropagator(mu,nu,s1,s2, t)
        zero = zero.expand()
        P1_0, P1_1, P1_2, P1_3 = 2,0,0,0
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    
     
    def test_mass_overmass(self):
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
        zero = zero.expand()
        P1_0, P1_1, P1_2, P1_3 = 2,0,0,0
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        M99 = M1
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)     
 
 
        

    def test_part_spin32propagator(self):
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
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertEqual(eval(str(data)),0)


    
    
    def test_spin2propagator(self):
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
        zero = 1j * propa - 2 * prop
        
        
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/36
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)    
 
    def test_spin2propagator2(self):
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
        
             
        zero = 1j*propa - aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 11
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    

    def test_spin2propagator3(self):
        """test the spin2 propagator property (contraction gives zero)"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
             
        zero = P(mu,t) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)    
        
        zero = Metric(mu,nu) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        zero = zero.expand().simplify() 
        
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0) 
    
    def test_spin2propagator4(self):
        """test the spin2 propagator is correctly contracted (even offshell)"""
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        t = 1
        mu, nu, alpha, beta = 1,2,3,4
        
        aloha = Metric(mu,nu) * aloha_obj.Spin2Propagator(mu,nu,alpha,beta, t)
        analytical = complex(0, 1/3) * (OM(t) * P(-1, t)**2 - 1) * (Metric(alpha, beta) + 2 * OM(t) * P(alpha,t)*P(beta,t))
        
        
        zero = aloha.expand().simplify().factorize() - analytical.expand().simplify()

        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        OM1 = 1/48#(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
            
    def test_spin2propagator5(self):
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
        p1e = p1.expand().simplify().factorize()
        
        solp1 = complex(0,1/2) * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2)
        zero = p1e - solp1.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        #part 2
        p2 =   0.5j * ( Metric(1003,'I3') * Metric(2003,'I2') * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p2e = p2.expand().simplify().factorize()
        zero = p2e - solp1.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        # part 3 -and part 8
        p3 = complex(0,-1/3) * ( Metric(1003,2003)**2 * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2) )
        p3e = p3.expand().simplify().factorize()
        solp3 = complex(0,-4/3) * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2)
        zero = p3e - solp3.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
            
        # part 4
        p4 = -0.5j * ( Metric(1003,'I2') * P(2003,3) * P('I3',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p4e = p4.expand().simplify().factorize()
        solp4 = complex(0,-1/2) * OverMass2(3) * P('I2',3) * P('I3',3) * Spinor(-1,1) * Spinor(-1,2)
        zero = p4e - solp4.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
        
        # part 5
        p5 = -0.5j * ( Metric(2003,'I3') * P(1003,3) * P('I2',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p5e = p5.expand().simplify().factorize()
        zero = p5e - solp4.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)   
        
        #part 6    
        p6 = -0.5j * ( Metric(1003,'I3') * P(2003,3) * P('I2',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )   
        p6e = p6.expand().simplify().factorize()
        zero = p6e - solp4.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        #part 7
        p7= -0.5j * ( Metric(2003,'I2') * P(1003,3) * P('I3',3) * OverMass2(3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p7e = p7.expand().simplify().factorize()
        zero = p7e - solp4.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        # part 9
        p9 = complex(0,1/3) * ( OverMass2(3) * P('I2',3) * P('I3',3) * Metric(1003,2003)**2 * Spinor(-1,1) * Spinor(-1,2) )
        p9e = p9.expand().simplify().factorize()
        solp9 = complex(0,4/3) * ( OverMass2(3) * P('I2',3) * P('I3',3) * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p9e - solp9.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
            
        # part 10
        p10 = complex(0,1/3) * ( OverMass2(3) * P(1003,3) * P(2003,3) * Metric('I2','I3') * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p10e = p10.expand().simplify().factorize()
        solp10 = complex(0,1/3) * ( OverMass2(3) * P(-1,3) **2 * Metric('I2','I3') * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p10e - solp10.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0) 
        
        
        # part 11
        p11 = complex(0,2/3) * ( OverMass2(3)**2 * P('I2',3) * P('I3',3) * P(1003,3) * P(2003,3) * Metric(1003,2003) * Spinor(-1,1) * Spinor(-1,2) )
        p11e = p11.expand().simplify().factorize()
        solp11 = complex(0,2/3) * ( OverMass2(3)**2 * P(-1,3) **2 * P('I2',3) * P('I3',3)  * Spinor(-1,1) * Spinor(-1,2) ) 
        zero = p11e - solp11.expand().simplify().factorize()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0)
            
        # full
        full = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p9 + p10 + p11
        fulle = full.expand()
        solfull = complex(0,1/3) * ((OverMass2(3) * P(-1, 3)**2 - 1) * (Metric('I2','I3') + 2 * OverMass2(3) * P('I2',3)*P('I3',3)) * Spinor(-1,1) * Spinor(-1,2))  
        solfullbis = 2 * solp1 + solp3 + 4 * solp4 + solp9 +solp10 + solp11
        # first sanity
        zero = solfullbis.expand() - solfull.expand()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0,6)
        
        
        zero = fulle - solfull.expand()
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str( data )),0,6)
        
        
        
class TestLorentzObjectRepresentation(unittest.TestCase):
    """Class to test the operation in the LorentzObjectRepresentation"""
    
    #for lorentz manipulation
    p1nu = aloha_obj.P(1,1)
    p1nu = p1nu.expand()
    p1mu = aloha_obj.P(2,1)
    p1mu = p1mu.expand()   
    p2nu = aloha_obj.P(1,2)
    p2nu = p2nu.expand()
    p2mu = aloha_obj.P(2,2)
    p2mu = p2mu.expand()
    
    #for lorentz - spin manipulation
    gamma_nu_ij = aloha_obj.Gamma(1,1,2)
    gamma_nu_ij = gamma_nu_ij.expand()
    gamma_nu_ji = aloha_obj.Gamma(1,2,1)
    gamma_nu_ji = gamma_nu_ji.expand()
    gamma_mu_ij = aloha_obj.Gamma(2,1,2)    
    gamma_mu_ij = gamma_mu_ij.expand()
    gamma_nu_jk = aloha_obj.Gamma(1,2,3)
    gamma_nu_jk = gamma_nu_jk.expand()
    gamma_mu_jk = aloha_obj.Gamma(2,2,3)
    gamma_mu_jk = gamma_mu_jk.expand()
    gamma_nu_kl = aloha_obj.Gamma(1,3,4)
    gamma_nu_kl = gamma_nu_kl.expand()
    gamma_mu_kl = aloha_obj.Gamma(2,3,4)
    gamma_mu_kl = gamma_mu_kl.expand()    
    gamma_mu_ki = aloha_obj.Gamma(2,3,1)
    gamma_mu_ki = gamma_mu_ki.expand()     
   
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


    def test_sum_with4ind(self):
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
        self.assertEqual(str(obj1b_rep.get_rep([0,1,0,0])), '-5 * ( P3_0 * P3_1 * OM3 )')       
        
        obj1_rep = obj1.simplify().expand().simplify()
        
        assert obj1_rep.lorentz_ind == [4,2,1,5] , "test not valid if condition not met"
        self.assertEqual(str(obj1_rep.get_rep([0,1,0,0])), '-5 * ( P3_0 * P3_1 * OM3 )')
        
    
        
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
                if not( data.variable == 'P1_%s' % ind[0] or data.variable == \
                                                            'P2_%s' % ind[1]):
                    raise Exception('invalid product')
            self.assertNotEqual(rep[0].variable, rep[1].variable)
        
        
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
                power = [data2.power for data2 in data]
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
                self.assertNotEqual(data[0].variable, data[1].variable)
      
    def testeinsteinsum2(self):
        
        class gamma_in_lorentz(aloha_lib.LorentzObject):
            """ local representation """
            
            def __init__(self, l1, l2, prefactor=1, constant=0):
                aloha_lib.LorentzObject.__init__(self,[l1, l2], [])
            
            representation = aloha_lib.LorentzObjectRepresentation(
            {(0,0): 0, (0,1): 0, (0,2): 0, (0,3):-1,
             (1,0): 0, (1,1): 0, (1,2): -1, (1,3):0,
             (2,0): 0, (2,1): 1, (2,2): 0, (2,3):0,
             (3,0): 1, (3,1): 0, (3,2): 0, (3,3):0
             }, [1,2], [])
            
#            create_representation = lambda : representation
            
        obj = gamma_in_lorentz([1,2],[],[])
        
        
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
        self.assertEqual(new.get_rep([3]), aloha_lib.ScalarVariable('P2_0'))
        self.assertEqual(new.get_rep([2]), aloha_lib.ScalarVariable('P2_1'))
        self.assertEqual(new.get_rep([1]), aloha_lib.ScalarVariable('P2_2'))
        self.assertEqual(new.get_rep([0]), aloha_lib.ScalarVariable('P2_3')) 
        self.assertEqual(new.get_rep([0]).prefactor, 1)
        self.assertEqual(new.get_rep([1]).prefactor, 1)   
        self.assertEqual(new.get_rep([2]).prefactor, -1)                  
        self.assertEqual(new.get_rep([3]).prefactor, 1)
        
    def testspinsum(self):
        
        class gamma_in_spin(aloha_lib.LorentzObject):
            """ local representation """
            
            def __init__(self, s1, s2, prefactor=1, constant=0):
                aloha_lib.LorentzObject.__init__(self, [], [s1, s2])
            
            representation = aloha_lib.LorentzObjectRepresentation(
                            {(0,0): 0, (0,1): 0, (0,2): 0, (0,3):-1,
                             (1,0): 0, (1,1): 0, (1,2): -1, (1,3):0,
                             (2,0): 0, (2,1): 1, (2,2): 0, (2,3):0,
                             (3,0): 1, (3,1): 0, (3,2): 0, (3,3):0},
                                        [], [1,2])
            
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
        self.assertEqual(new.get_rep([3]), aloha_lib.ScalarVariable('F2_1'))
        self.assertEqual(new.get_rep([2]), aloha_lib.ScalarVariable('F2_2'))
        self.assertEqual(new.get_rep([1]), aloha_lib.ScalarVariable('F2_3'))
        self.assertEqual(new.get_rep([0]), aloha_lib.ScalarVariable('F2_4')) 
        self.assertEqual(new.get_rep([0]).prefactor, -1)
        self.assertEqual(new.get_rep([1]).prefactor, -1)   
        self.assertEqual(new.get_rep([2]).prefactor, 1)                  
        self.assertEqual(new.get_rep([3]).prefactor, 1)       
      
      
    def test_sumofLorentzObj(self):
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
                self.assertEqual(data.__class__, aloha_lib.ScalarVariable)
        
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


    def test_other(self):
        """ test that all object are defined"""        
        Gamma = aloha_obj.Gamma
        Gamma5 = aloha_obj.Gamma5
        Sigma = aloha_obj.Sigma
        ProjM = aloha_obj.ProjM
        ProjP = aloha_obj.ProjP
        Identity = aloha_obj.Identity
        Metric = aloha_obj.Metric  
    
    def test_projector(self):
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
    
    
    def test_Pslashproperty(self):
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
        expr = (PSlash(1,2,1)+ M(1))*(PSlash(2,3,1)-M(1))
        P1_0, P1_1, P1_2, P1_3 = 7,2,3,5
        M1 = math.sqrt(P1_0 **2 - P1_1 **2 -P1_2 **2 -P1_3 **2)

        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)  
         
        #checking that (/p + m)(/p-m)(P)=0 (for onshell)
        expr = (PSlash(1,2,1)+ M(1))*(PSlash(2,3,1)-M(1))*(Gamma(4,3,4)*Identity(3,4) * P(4,1))
        for ind in zero.listindices():
            data = zero.get_rep(ind)
            self.assertAlmostEqual(eval(str(data)), 0)  

    
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
        
        
        
        
        

            
#    def test_Spin2Contraction(self): 
#        """check spin2 contraction"""
#        
#        T = aloha_obj.Spin2
#        metric = aloha_obj.Metric
#        F = aloha_obj.Spinor
#        Id = aloha_obj.Identity 
#        
##        obj = T(1,2,3) * metric(1,2) *F(1,1) * F(2,2) * Id(1,2)    
 #       obj = obj.expand().simplify().factorize()
 ##       print obj
  #      obj = metric(1,2) * Id(1,2) * F(1,1) * F(2,2) * T(1,2,3)    
  #      obj = obj.expand().simplify().factorize()
  #      #self.assertEqual(str(obj), '')
        
        
    def test_parity_for_epsilon(self):

        # usefull shortcut
        Epsilon = aloha_obj.Epsilon
        # test some value
        eps = Epsilon(1,2,3,4)
        
        indices = ((l1, l2, l3, 6 - l1- l2 -l3)
                                 for l1 in range(4) \
                                 for l2 in range(4) if l2 != l1\
                                 for l3 in range(4) if l3 not in [l1,l2])
        for index in indices:
            val1 = eps.give_parity(index)
            val2 = aloha_obj.give_sign_perm([0,1,2,3], index)
        
            self.assertEqual(val1, val2, 'not same parity for perm %s' % (index,))

    def testEpsilonProperty(self):
        """Test the property of the epsilon object"""
        
        # usefull shortcut
        Epsilon = aloha_obj.Epsilon

        # test some value
        eps = Epsilon(1,2,3,4)
        eps = eps.expand().simplify()
        self.assertEqual(eps.get_rep([0,1,2,3]), 1)
        self.assertEqual(eps.get_rep([0,1,2,2]), 0) 
        self.assertEqual(eps.get_rep([0,1,3,2]), -1) 
        self.assertEqual(eps.get_rep([0,1,1,2]), 0) 
        self.assertEqual(eps.get_rep([0,0,2,2]), 0) 
        self.assertEqual(eps.get_rep([1,2,3,0]), -1) 
        self.assertEqual(eps.get_rep([1,2,0,3]), 1) 
        self.assertEqual(eps.get_rep([1,0,2,3]), -1) 

        # Test the full contraction of two Epsilon
        contraction = Epsilon(1,2,3,4) * Epsilon(1,2,3,4)
        
        contraction = contraction.simplify().expand().simplify()
        self.assertEqual(contraction, -24)
        
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
        fact1 = fact1
        fact2 = fact2
         
        result = -6 * aloha_obj.Metric('d','e')
        result = result.expand().simplify()
        prod = fact1 * fact2
        prod = prod.expand().simplify()

        self.assertEqual(prod, result)


  
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
        zero = zero.simplify().expand().simplify()
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
          
class TestConstantObject(unittest.TestCase):
    """Check the different Operation for a Constant Object"""
    
    def testsum(self):
        
        const = aloha_lib.ConstantObject()
        p = aloha_obj.P(1,1)
        
        sum = const + p
        self.assertEqual(type(sum),type(p))
        self.assertEqual(sum, p)
                
        p2 = aloha_obj.P(1,2) 
        add = p + p2   


class TestSimplify(unittest.TestCase):
    """Check that the simplification works correctly"""        

    def testsimplifyMultLorentz(self):
        
        # For Standard Product : No Simplification
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.Gamma(3, 4, 5)
        
        simp = prod.simplify()
        self.assertEqual(simp, prod)
        
        # Look if Multiply by Propagator
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.SpinorPropagator(1, 2 ,3) / \
                                            aloha_obj.DenominatorPropagator(3) 
        simp = prod.simplify()
        
        self.assertEqual(simp.__class__, aloha_lib.FracVariable)
        self.assertEqual(simp.denominator.__class__, aloha_lib.AddVariable)
        simp = simp.expand()
        simp = simp.simplify()
        self.assertEqual(simp.denominator.__class__, aloha_lib.LorentzObjectRepresentation) 
        denominator = simp.denominator.get_rep([0])     
        for data in denominator:
            if aloha_lib.ScalarVariable('P3_0') == data:
                self.assertEqual(data.prefactor, 1)
                self.assertEqual(data.power, 2)
            elif aloha_lib.ScalarVariable('P3_1') == data:
                self.assertEqual(data.prefactor, -1)
                self.assertEqual(data.power, 2)
            elif aloha_lib.ScalarVariable('P3_2') == data:
                self.assertEqual(data.prefactor, -1)
                self.assertEqual(data.power, 2)                
            elif aloha_lib.ScalarVariable('P3_3') == data:
                self.assertEqual(data.prefactor, -1)
                self.assertEqual(data.power, 2)
            elif aloha_lib.ScalarVariable('M3') == data:
                self.assertEqual(data.prefactor, -1)
                self.assertEqual(data.power, 2)                
            elif aloha_lib.ScalarVariable('W3') in data:
                self.assertEqual(data.prefactor, complex(0,1))

        
        
    def testsimplifyFracVariable(self):
        
        # For Standard Product : No Simplification
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.Gamma(3, 4, 5)
        
        simp = prod.simplify()
        self.assertEqual(simp, prod)
        
        # Look if Multiply by Propagator
        
        prod = aloha_obj.Gamma(1, 2, 3) * aloha_obj.SpinorPropagator(1, 2 ,3) / \
                                            aloha_obj.DenominatorPropagator(3) 
        simp = prod.simplify()
        
        self.assertEqual(simp.__class__, aloha_lib.FracVariable)
        self.assertEqual(simp.denominator.__class__, aloha_lib.AddVariable)
        
        
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
            
    def test_aloha_VVS(self):
        """ Test the VVS creation of vertex """
        
        VVS_15 = self.Lorentz(name = 'VVS_15',
                 spins = [ 3, 3, 1 ],
                 structure = 'Metric(1,2)')

        abstract = create_aloha.AbstractRoutineBuilder(VVS_15).compute_routine(3)
        
        self.assertEqual(abstract.expr.numerator.nb_lor, 0)
        self.assertEqual(abstract.expr.numerator.nb_spin, 0)
        
    def test_aloha_ZPZZ(self):
        """ Check the validity of Funny Zp coupling to z z """
                
        ZPZZ = self.Lorentz(name = 'ZPZZ',
                 spins = [ 3, 3, 3 ],
                 structure = 'P(-1,1)*Epsilon(3,1,2,-2)*P(-1,1)*P(-2,2)-Epsilon(3,1,2,-2)*P(-1,2)*P(-1,2)*P(-2,1)-Epsilon(3,2,-1,-2)*P(1,1)*P(-1,2)*P(-2,1)+Epsilon(3,1,-1,-2)*P(2,2)*P(-1,2)*P(-2,1)')
    
        abstract_ZP = create_aloha.AbstractRoutineBuilder(ZPZZ).compute_routine(0)
        expr = abstract_ZP.expr

        V2_1, V2_2, V2_3, V2_4  = 1, 2, 3, 4
        V1_1, V1_2, V1_3, V1_4  = 5, 6, 7, 8
        V3_1, V3_2, V3_3, V3_4  = 9, 100, 11, 13
        OM1,OM2,OM3 = 9,11,13
        j = complex(0,1)
        P1_0,P1_1,P1_2,P1_3 = 10, 11, 12, 19
        P2_0,P2_1,P2_2,P2_3 = 101, 111, 121, 134
        P3_0,P3_1,P3_2,P3_3 = 1001, 1106, 1240, 1320

        for ind in expr.listindices():
            self.assertEqual(eval(str(expr.get_rep(ind))), -178727040j)


    def test_use_of_library_spin2(self):
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
        
        for ind in one_exp.listindices():
            data = one_exp.get_rep(ind) - two_exp.get_rep(ind)
            data.simplify()
            self.assertAlmostEqual(eval(str(one_exp.get_rep(ind))), eval(str(two_exp.get_rep(ind))))
                
    def test_aloha_FFT2(self):
        """ test the FFT2 creation of vertex"""

        FFT2 = self.Lorentz(name = 'FFT2',
                 spins = [2, 2, 5],
        structure="Metric(1003,2003)*ProjP(1,2)+Metric(1003,2003)*ProjM(1,2)"
        )
        abstract_FFT = create_aloha.AbstractRoutineBuilder(FFT2).compute_routine(3)
        expr = abstract_FFT.expr
        
        Metric = aloha_obj.Metric
        P = aloha_obj.P
        OM = aloha_obj.OverMass2
        F = aloha_obj.Spinor
        result = complex(0,1/3) * (OM(3) * P(-1, 3)**2 - 1) * (Metric('I2','I3') + 2 * OM(3) * P('I2',3)*P('I3',3))
        result = result * F(-2,1) * F(-2,2)
        
        zero = expr.numerator - result.expand()
        zero = zero.simplify()
        
        P1_0,P1_1,P1_2,P1_3 = 1000, 3, 4, 1000
        P2_0,P2_1,P2_2,P2_3 = 1000, 3, 6, -1000
        P3_0,P3_1,P3_2,P3_3 = 2000, 2, 6, 9
        
        F1_1, F1_2, F1_3, F1_4  = -44.7213595499958, 62,34,23
        F2_1, F2_2, F2_3, F2_4  = 12, 44, 72, -45 
        OM1,OM2,OM3 = 0 , 0, 1.0 / 500**2
        M3 = 500
        
        
        for ind in zero.listindices():
            self.assertAlmostEqual(eval(str(zero.get_rep(ind))),0)
             


    def test_aloha_FFV(self):
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
        
        
        abstract_M = create_aloha.AbstractRoutineBuilder(FFV_M).compute_routine(3)       
        abstract_P = create_aloha.AbstractRoutineBuilder(FFV_P).compute_routine(3)       
        abstract = create_aloha.AbstractRoutineBuilder(FFV).compute_routine(3)
        
        zero = abstract_M.expr.numerator + abstract_P.expr.numerator - \
                            abstract.expr.numerator
        F2_1, F2_2, F2_3, F2_4  = 1, 2, 3, 4
        F1_1, F1_2, F1_3, F1_4  = 5, 6, 7, 8
        OM3 = 9
        j = complex(0,1)
        P3_0,P3_1,P3_2,P3_3 = 10, 11, 12, 13
            
        for ind in zero.listindices():
            self.assertEqual(eval(str(zero.get_rep(ind))),0)
            
        #tested solution again MG4
        s1 = -j*((OM3*(P3_0*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+((F1_3*F2_1)+(F1_4*F2_2)))        
        s2 = -j*((OM3*(P3_1*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-(F1_4*F2_1)-(F1_3*F2_2)))
        s3 = -j*((OM3*(P3_2*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-1*j*(F1_4*F2_1)+1*j*(F1_3*F2_2)))
        s4 = -j*((OM3*(P3_3*((F2_1*((F1_3*(-P3_0-P3_3))+(F1_4*(-P3_1-1*j*P3_2))))+(F2_2*((F1_3*(-P3_1+1*j*P3_2))+(F1_4*(-P3_0+P3_3)))))))+(-(F1_3*F2_1)+(F1_4*F2_2)))
        
        self.assertEqual(s1, eval(str(abstract_M.expr.numerator.get_rep([0]))))
        self.assertEqual(s2, eval(str(abstract_M.expr.numerator.get_rep([1]))))    
        self.assertEqual(s3, eval(str(abstract_M.expr.numerator.get_rep([2]))))    
        self.assertEqual(s4, eval(str(abstract_M.expr.numerator.get_rep([3]))))                                   

        FFV_6 = self.Lorentz(name = 'FFV_6',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

        abstract_6 = create_aloha.AbstractRoutineBuilder(FFV_6).compute_routine(3)
         
        zero = abstract_6.expr.numerator - abstract_M.expr.numerator - \
                                                    2* abstract_P.expr.numerator   
        
        for ind in zero.listindices():
            self.assertEqual(eval(str(zero.get_rep(ind))),0)
        
    def test_aloha_symmetries(self):
        """ test that the symmetries of particles works """
    
        # Check that full identification symmetry works
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        helas_suite.look_for_symmetries()
        solution = {'VVVV2': {2: 1 ,4: 3}, 'SSS1': {2: 1, 3: 2}, 'VVSS1': {2: 1, 4: 3}, 'VVS1': {2: 1}}  
        self.assertEqual(solution, helas_suite.symmetries)
        
    def test_has_symmetries(self):
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

    def test_aloha_multiple_lorentz(self):
        """ check if the detection of multiple lorentz work """
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        helas_suite.look_for_multiple_lorentz_interactions()
        solution = {'FFV2': [('FFV3',), ('FFV4',), ('FFV5',)]}
        self.assertEqual(solution, helas_suite.multiple_lor)
        

    def test_aloha_multiple_lorentz_and_symmetry(self):
        """ check if the detection of multiple lorentz work """
        
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

        goal =""" subroutine VVS1_1(V2, S3, COUP, M1, W1, V1)
implicit none 
double complex V1(*)
double complex V2(*)
double complex S3(*)
double complex COUP
double complex denom
double precision M1, W1
double complex OM1
double precision P1(0:3)

V1(5)= V2(5)+S3(2)
V1(6)= V2(6)+S3(3)
P1(0) = - dble(V1(5))
P1(1) = - dble(V1(6))
P1(2) = - dimag(V1(6))
P1(3) = - dimag(V1(5))
OM1 = 0d0
if (M1 .ne. 0d0) OM1=1d0/M1**2

denom =1d0/(( (M1*( -M1+(0, 1)*W1))+( (P1(0)**2)-(P1(1)**2)-(P1(2)**2)-(P1(3)**2))))
V1(1)= COUP*denom*(S3(1)*( (OM1*( (0, 1)*(V2(1)*P1(0))+(0, -1)*(V2(2)*P1(1))+(0, -1)*(V2(3)*P1(2))+(0, -1)*(V2(4)*P1(3)))*P1(0))+(0, -1)*V2(1)))
V1(2)= COUP*denom*(S3(1)*( (OM1*( (0, 1)*(V2(1)*P1(0))+(0, -1)*(V2(2)*P1(1))+(0, -1)*(V2(3)*P1(2))+(0, -1)*(V2(4)*P1(3)))*P1(1))+(0, -1)*V2(2)))
V1(3)= COUP*denom*(S3(1)*( (OM1*( (0, 1)*(V2(1)*P1(0))+(0, -1)*(V2(2)*P1(1))+(0, -1)*(V2(3)*P1(2))+(0, -1)*(V2(4)*P1(3)))*P1(2))+(0, -1)*V2(3)))
V1(4)= COUP*denom*(S3(1)*( (OM1*( (0, 1)*(V2(1)*P1(0))+(0, -1)*(V2(2)*P1(1))+(0, -1)*(V2(3)*P1(2))+(0, -1)*(V2(4)*P1(3)))*P1(3))+(0, -1)*V2(4)))
end



 subroutine VVS1_2(V2, S3, COUP, M1, W1, V1)
implicit none 
double complex V1(*)
double complex V2(*)
double complex S3(*)
double complex COUP
double complex denom
double precision M1, W1
double complex OM1
double precision P1(0:3)
call VVS1_1(V2,S3,COUP,M1,W1,V1)
end


 subroutine VVS1_2_1(V2, S3, COUP1,COUP2, M1, W1, V1)
implicit none 
double complex V1(*)
double complex V2(*)
double complex S3(*)
double complex COUP1,COUP2
double complex denom
double precision M1, W1
double complex OM1
double precision P1(0:3)
 double complex TMP(6)
 integer i

 CALL VVS1_1(V2, S3, COUP1, M1, W1, V1)
 CALL VVS2_1(V2, S3, COUP2, M1, W1, TMP)
 do i=1,4
                V1(i) = V1(i) + tmp(i)
                enddo
end

 subroutine VVS1_2_2(V2, S3, COUP1,COUP2, M1, W1, V1)
implicit none 
double complex V1(*)
double complex V2(*)
double complex S3(*)
double complex COUP1,COUP2
double complex denom
double precision M1, W1
double complex OM1
double precision P1(0:3)
 double complex TMP(6)
 integer i

 CALL VVS1_1(V2, S3, COUP1, M1, W1, V1)
 CALL VVS2_1(V2, S3, COUP2, M1, W1, TMP)
 do i=1,4
                V1(i) = V1(i) + tmp(i)
                enddo
end

"""
        self.assertEqual(text.split('\n'),goal.split('\n')) 
   
        text_h, text_cpp =  abstract.write(None, 'CPP')
    
        goal_h = """#ifndef VVS1_1_guard
#define VVS1_1_guard
#include <complex>
using namespace std;

void VVS1_1(complex<double> V2[],complex<double> S3[],complex<double> COUP, double M1, double W1, complex<double>V1[]);

void VVS1_2(complex<double> V2[],complex<double> S3[],complex<double> COUP, double M1, double W1, complex<double>V1[]);

#endif

#ifndef VVS1_2_1_guard
#define VVS1_2_1_guard
#include <complex>
using namespace std;

void VVS1_2_1(complex<double> V2[],complex<double> S3[],complex<double> COUP1, complex <double>COUP2, double M1, double W1, complex<double>V1[]);

void VVS1_2_2(complex<double> V2[],complex<double> S3[],complex<double> COUP1, complex <double>COUP2, double M1, double W1, complex<double>V1[]);

#endif"""
        goal_cpp = """#include "VVS1_1.h"

void VVS1_1(complex<double> V2[],complex<double> S3[],complex<double> COUP, double M1, double W1, complex<double>V1[]){
complex<double> denom;
complex<double> OM1;
double P1[4];
V1[4]= V2[4]+S3[1];
V1[5]= V2[5]+S3[2];
P1[0] = -V1[4].real();
P1[1] = -V1[5].real();
P1[2] = -V1[5].imag();
P1[3] = -V1[4].imag();
OM1 = 0;
if (M1 != 0) OM1= 1./pow(M1,2);
denom =1./(( (M1*( -M1+complex<double>(0., 1.)*W1))+( (pow(P1[0],2))-(pow(P1[1],2))-(pow(P1[2],2))-(pow(P1[3],2)))));
V1[0]= COUP*denom*(S3[0]*( (OM1*( complex<double>(0., 1.)*(V2[0]*P1[0])+complex<double>(0., -1.)*(V2[1]*P1[1])+complex<double>(0., -1.)*(V2[2]*P1[2])+complex<double>(0., -1.)*(V2[3]*P1[3]))*P1[0])+complex<double>(0., -1.)*V2[0]));
V1[1]= COUP*denom*(S3[0]*( (OM1*( complex<double>(0., 1.)*(V2[0]*P1[0])+complex<double>(0., -1.)*(V2[1]*P1[1])+complex<double>(0., -1.)*(V2[2]*P1[2])+complex<double>(0., -1.)*(V2[3]*P1[3]))*P1[1])+complex<double>(0., -1.)*V2[1]));
V1[2]= COUP*denom*(S3[0]*( (OM1*( complex<double>(0., 1.)*(V2[0]*P1[0])+complex<double>(0., -1.)*(V2[1]*P1[1])+complex<double>(0., -1.)*(V2[2]*P1[2])+complex<double>(0., -1.)*(V2[3]*P1[3]))*P1[2])+complex<double>(0., -1.)*V2[2]));
V1[3]= COUP*denom*(S3[0]*( (OM1*( complex<double>(0., 1.)*(V2[0]*P1[0])+complex<double>(0., -1.)*(V2[1]*P1[1])+complex<double>(0., -1.)*(V2[2]*P1[2])+complex<double>(0., -1.)*(V2[3]*P1[3]))*P1[3])+complex<double>(0., -1.)*V2[3]));
}

void VVS1_2(complex<double> V2[],complex<double> S3[],complex<double> COUP, double M1, double W1, complex<double>V1[]){
VVS1_1(V2,S3,COUP,M1,W1,V1);
}

#include "VVS1_2_1.h"

void VVS1_2_1(complex<double> V2[],complex<double> S3[],complex<double> COUP1, complex<double>COUP2, double M1, double W1, complex<double>V1[]){
complex<double> tmp[6];
 int i = 0;

VVS1_1(V2, S3, COUP1, M1, W1, V1);
VVS2_1(V2, S3, COUP2, M1, W1, tmp);
 while (i < 4)
                {
                V1[i] = V1[i] + tmp[i];
                i++;
                }
}

#include "VVS1_2_2.h"

void VVS1_2_2(complex<double> V2[],complex<double> S3[],complex<double> COUP1, complex<double>COUP2, double M1, double W1, complex<double>V1[]){
complex<double> tmp[6];
 int i = 0;

VVS1_1(V2, S3, COUP1, M1, W1, V1);
VVS2_1(V2, S3, COUP2, M1, W1, tmp);
 while (i < 4)
                {
                V1[i] = V1[i] + tmp[i];
                i++;
                }
}

"""
        
        self.assertEqual(text_h.split('\n'),goal_h.split('\n'))
        self.assertEqual(text_cpp.split('\n'),goal_cpp.split('\n'))
        
        
        text =  abstract.write(None, 'Python')
        goal = """import wavefunctions
def VVS1_1(V2, S3, COUP, M1, W1):
    V1 = wavefunctions.WaveFunction(size=6)
    V1[4] = V2[4]+S3[1]
    V1[5] = V2[5]+S3[2]
    P1 = [-complex(V1[4]).real, \\
            - complex(V1[5]).real, \\
            - complex(V1[5]).imag, \\
            - complex(V1[4]).imag]
    OM1 = 0.0
    if (M1): OM1=1.0/M1**2
    denom =1.0/(( (M1*( -M1+1j*W1))+( (P1[0]**2)-(P1[1]**2)-(P1[2]**2)-(P1[3]**2))))
    V1[0]= COUP*denom*(S3[0]*( (OM1*( 1j*(V2[0]*P1[0])-1j*(V2[1]*P1[1])-1j*(V2[2]*P1[2])-1j*(V2[3]*P1[3]))*P1[0])-1j*V2[0]))
    V1[1]= COUP*denom*(S3[0]*( (OM1*( 1j*(V2[0]*P1[0])-1j*(V2[1]*P1[1])-1j*(V2[2]*P1[2])-1j*(V2[3]*P1[3]))*P1[1])-1j*V2[1]))
    V1[2]= COUP*denom*(S3[0]*( (OM1*( 1j*(V2[0]*P1[0])-1j*(V2[1]*P1[1])-1j*(V2[2]*P1[2])-1j*(V2[3]*P1[3]))*P1[2])-1j*V2[2]))
    V1[3]= COUP*denom*(S3[0]*( (OM1*( 1j*(V2[0]*P1[0])-1j*(V2[1]*P1[1])-1j*(V2[2]*P1[2])-1j*(V2[3]*P1[3]))*P1[3])-1j*V2[3]))
    return V1
    
    
def VVS1_2(V2, S3, COUP, M1, W1):
    return VVS1_1(V2,S3,COUP,M1,W1)

def VVS1_2_1(V2, S3, COUP1,COUP2, M1, W1):


    V1 = VVS1_1(V2, S3, COUP1, M1, W1)
    tmp = VVS2_1(V2, S3, COUP2, M1, W1)
    for i in range(4):
        V1[i] += tmp[i]
    return V1

def VVS1_2_2(V2, S3, COUP1,COUP2, M1, W1):


    V1 = VVS1_1(V2, S3, COUP1, M1, W1)
    tmp = VVS2_1(V2, S3, COUP2, M1, W1)
    for i in range(4):
        V1[i] += tmp[i]
    return V1

"""
        self.assertEqual(text.split('\n'),goal.split('\n'))
        
        
    def test_full_sm_aloha(self):
        """test that the full SM seems to work"""
        # Note that this test check also some of the routine define inside this
        #because of use of some global.
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        self.assertEqual(helas_suite.look_for_conjugate(), {})
        start = time.time()
        helas_suite.compute_all()
        timing = time.time()-start
        if timing > 10:
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
            self.assertEqual(abstract.expr.numerator.nb_lor, lorentz_solution)
            spin_solution = spin_index[helas.spins[output_part -1]]
            self.assertEqual(abstract.expr.numerator.nb_spin, spin_solution, \
                             error % name)
            
    def test_multiple_lorentz_subset(self):
        """test if we create the correct set of routine/files for multiple lorentz"""
        
        helas_suite = create_aloha.AbstractALOHAModel('sm')
        requested_routines=[(('FFV1',) , (), 0), 
                            (('FFV1','FFV2') , (1,), 0)]
        
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
        os.system('rm -r /tmp/mg5 &> /dev/null; mkdir /tmp/mg5 &> /dev/null')
        helas_suite.write('/tmp/mg5', 'Fortran')
        
        content = set(os.listdir('/tmp/mg5'))
        self.assertEqual(content, set(['FFV1_0.f',
                                       'FFV1C1_0.f','FFV2C1_0.f',
                                       'FFV1C1_2_0.f']))
        
        # Check the content of FFV1__FFV2C1_0.f
        fsock = open('/tmp/mg5/FFV1C1_2_0.f')
        goal = """C     This File is Automatically generated by ALOHA 
C     
      SUBROUTINE FFV1C1_2_0(F1,F2,V3,COUP1,COUP2,VERTEX)
      IMPLICIT NONE
      DOUBLE COMPLEX F1(*)
      DOUBLE COMPLEX F2(*)
      DOUBLE COMPLEX V3(*)
      DOUBLE COMPLEX COUP1,COUP2
      DOUBLE COMPLEX VERTEX
      DOUBLE COMPLEX TMP


      CALL FFV1C1_0(F1, F2, V3, COUP1, VERTEX)
      CALL FFV2C1_0(F1, F2, V3, COUP2, TMP)
      VERTEX = VERTEX + TMP
      END


"""
        self.assertEqual(fsock.read().split('\n'), goal.split('\n'))
        
        
        
        
        
    
    def test_mssm_subset_creation(self):
        """ test the creation of subpart of ALOHA routines 
        including clash routines """
        helas_suite = create_aloha.AbstractALOHAModel('mssm')
        
        requested_routines=[(('FFV1',) , (), 0), 
                            (('FFV1',), (), 2),
                            (('FFV1',), (1,), 0),
                            (('FFV2',), (1,), 3),
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
                self.assertEqual(abstract.expr.numerator.nb_lor, 1, error % name)
                self.assertEqual(abstract.expr.numerator.nb_spin, 0, error % name)
            elif name in ['FFV1']:
                self.assertEqual(abstract.expr.numerator.nb_lor, 0, error % name)
                self.assertEqual(abstract.expr.numerator.nb_spin, 1, error % name)
            else:
                raise Exception, 'not expected routine %s' % name
            
    def find_helas(self, name, model):
        for lorentz in model.all_lorentz:
            if lorentz.name == name:
                return lorentz
            
        raise Exception('the test is confuse by name %s' % name)
        
    def test_aloha_FFVC(self):
        """ test the FFV creation of vertex """
        from models.mssm.object_library import Lorentz

        FFV = Lorentz(name = 'FFV',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,2)')        
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        amp = builder.compute_routine(0)
        conjg_builder= builder.define_conjugate_builder()
        conjg_amp = conjg_builder.compute_routine(0)
    
        # Check correct contraction
        self.assertEqual(conjg_amp.expr.nb_lor, 0)
        self.assertEqual(conjg_amp.expr.nb_spin, 0)
      
        # Check expr are different
        self.assertNotEqual(str(amp.expr), str(conjg_amp.expr))
        self.assertNotEqual(amp.name, conjg_amp.name)
        
        
    def test_aloha_expr_FFV2C1(self):
        """Test analytical expression for fermion clash routine"""
        from models.mssm.object_library import Lorentz
        FFV = Lorentz(name = 'FFV2',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,\'s1\')*ProjM(\'s1\',1)')
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        conjg_builder= builder.define_conjugate_builder()
        amp = conjg_builder.compute_routine(0)

        self.assertEqual(amp.expr.nb_spin, 0)
        F1_1, F1_2, F1_3, F1_4 = 1,2,3,4
        F2_1, F2_2, F2_3, F2_4 = 5,5,6,7
        V3_1, V3_2, V3_3, V3_4 = 8,9,10,11
        # For V4:
        cImag = complex(0,1)

        ufo_value = eval(str(amp.expr.get_rep([0])))
    
        v4_value = ( (F2_1*F1_3+F2_2*F1_4)*V3_1 \
                    -(F2_1*F1_4+F2_2*F1_3)*V3_2 \
                    +(F2_1*F1_4-F2_2*F1_3)*V3_3*cImag \
                    -(F2_1*F1_3-F2_2*F1_4)*V3_4       )

        self.assertEqual(complex(0,-1)*ufo_value, v4_value)

        FFV = Lorentz(name = 'FFV2',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,2,\'s1\')*ProjP(\'s1\',1)')
        builder = create_aloha.AbstractRoutineBuilder(FFV)
        conjg_builder= builder.define_conjugate_builder()
        amp = conjg_builder.compute_routine(0)
        
        ufo_value = eval(str(amp.expr.get_rep([0])))
        self.assertNotEqual(complex(0,1)*ufo_value, v4_value)
        v4_value = (F2_3*F1_1+F2_4*F1_2)*V3_1 \
                          +(F2_3*F1_2+F2_4*F1_1)*V3_2 \
                          -(F2_3*F1_2-F2_4*F1_1)*V3_3*cImag \
                          +(F2_3*F1_1-F2_4*F1_2)*V3_4
               
        self.assertEqual(complex(0,-1)*ufo_value, v4_value)
        
    def test_aloha_expr_FFFF(self):
        """Test analytical expression for fermion clash routine"""
        
        from models.mssm.object_library import Lorentz
        FFFF = Lorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Identity(1,2)*Identity(4,3)')
        
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
    
    
    def test_change_number_format_fortran(self):
        """ Check that the number are correctly written in fortranwriter """
        
        SSS = UFOLorentz(name = 'SSS',
               spins = [ 1, 1, 1])
    
        
        abstract = AbstractRoutineBuilder(SSS).compute_routine(0)
        writer = aloha_writers.ALOHAWriterForFortran(abstract, '/tmp')
        
        numbers = [complex(0,1), complex(0,1/2), 3*complex(1.0,3), complex(1,0)]
        numbers +=[0, 1, 2, -3, 3.0, 3.00, 1.00001, 2000, 24300.1, 1/3, 1/4, 3/4]
 
        solution = ['(0, 1)', '(0, 0.5)', '(3, 9)', '1', '0', '1', '2', '-3', '3', '3', '1.00001', '2000', '24300.1', '0.333333333', '0.25', '0.75']
        converted = [writer.change_number_format(number) for number in numbers]
        map(self.assertEqual, converted, solution)
 
 
    def test_pythonwriter(self):
        """ test that python writer works """
        
        solution ="""import wavefunctions
def SSS1_1(S2, S3, COUP, M1, W1):
    S1 = wavefunctions.WaveFunction(size=3)
    S1[1] = S2[1]+S3[1]
    S1[2] = S2[2]+S3[2]
    P1 = [-complex(S1[1]).real, \\
            - complex(S1[2]).real, \\
            - complex(S1[2]).imag, \\
            - complex(S1[1]).imag]
    denom =1.0/(( (M1*( -M1+1j*W1))+( (P1[0]**2)-(P1[1]**2)-(P1[2]**2)-(P1[3]**2))))
    S1[0]= COUP*denom*1j*(S3[0]*S2[0])
    return S1
    
    
def SSS1_2(S2, S3, COUP, M1, W1):
    return SSS1_1(S2,S3,COUP,M1,W1)

def SSS1_3(S2, S3, COUP, M1, W1):
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
        
        
    def test_python_routine_are_exec(self):
        """ check if the python routine can be call """
            
        FFV2 = UFOLorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,\'s1\')*ProjM(\'s1\',1)')
            
        builder = create_aloha.AbstractRoutineBuilder(FFV2)
        builder.apply_conjugation()
        amp = builder.compute_routine(0)
        routine = amp.write(output_dir=None, language='Python')

        
        solution = """import wavefunctions
def FFV2C1_0(F1,F2,V3,COUP):
    vertex = COUP*( (F1[3]*( (F2[0]*( -1j*V3[1]-V3[2]))+(F2[1]*( 1j*V3[0]+1j*V3[3]))))+(F1[2]*( (F2[0]*( 1j*V3[0]-1j*V3[3]))+(F2[1]*( -1j*V3[1]+V3[2])))))
    return vertex
    
    
""" 

        split_solution = solution.split('\n')
        split_routine = routine.split('\n')
        self.assertEqual(split_solution,split_routine)
        self.assertEqual(len(split_routine), len(split_solution))
            
            

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
##   Diagram of Class
##
##    Variable (vartype:0)<--- ScalarVariable 
##                          |
##                          +- LorentzObject 
##                                
##
##    list <--- AddVariable (vartype :1)   
##           |
##           +- MultVariable  <--- MultLorentz (vartype:2) 
##           
##    list <--- LorentzObjectRepresentation (vartype :4) <-- ConstantObject
##                                                               (vartype:5)
##
##    FracVariable (vartype:3)
##
################################################################################
"""This module develop basic structure for the symbolic treatment of the Helas
Amplitude.

The Symbolic treatment are based on a series of (scalar) variable of the form
a * x. 'a' is a (complex) number and x is an abstract variable.
Those variable can be multiply/add to each other and will then create a list of 
those variable in 'MultVariable','AddVariable'. 

As We will work on two different level, we have two concrete representation of
Variable: 
    - 'ScalarVariable' in order to represent a number (mass, component of 
        impulsion,...
    - 'LorentzObject' in order to store HighLevel object with lorentz/spin 
        indices (as Gamma, Impulsion,...)

In order to pass from HighLevel Object (deriving from LorentzObject) to concrete
definition based on ScalarVariable, we define a new class 
'LorentzObjectRepresentation' 

Each class defines the attribute "typevar" which is unique, this fastenize the 
type recognition of the object."""
from __future__ import division
from numbers import Number

#number below this number will be consider as ZERO
precision_cut = 1e-14

USE_TAG=set() #global to check which tag are used

depth=-1

#===============================================================================
# FracVariable
#=============================================================================== 
class FracVariable(object):
    """A representation of a fraction. This object simply have 
           - a numerator (self.numerator)
           - a denominator (self.denominator)  
    The numerator/denominator can be of any type of Object.
    
    All call function simply retranslate the call on the numerator/denominator
    """
    
    # In order to fastenize type
    vartype = 3
    
    def __init__(self, numerator, denominator, *opt):
        """ initialize the frac variable """
        
        self.numerator = numerator
        self.denominator = denominator
        #if  isinstance(self.numerator, Number):
        #    self.tag = self.denominator.tag
        #else:
        #    self.tag = self.numerator.tag.union(self.denominator.tag)
    
    def copy(self):
        """return a copy of the frac variable"""
        num = self.numerator.copy()
        den = self.denominator.copy()
        return FracVariable(num,den)
    
        
        
    def simplify(self):
        """apply rule of simplification"""
        
        if not isinstance(self.numerator, Number):
            self.numerator = self.numerator.simplify()
        self.denominator = self.denominator.simplify()
        
        if isinstance(self.denominator, ConstantObject):
            self.numerator /= self.denominator.constant_term
            return self.numerator
        else:
            return self
    
    def factorize(self):
        """made the factorization"""
        if hasattr(self.numerator, 'vartype'):
            self.numerator = self.numerator.factorize()
        self.denominator = self.denominator.factorize()
        return self
        
    
    def expand(self):
        """Expand the content information"""
        
        if isinstance(self.numerator, Number):
            return FracVariable(self.numerator, self.denominator.expand())
        return FracVariable(self.numerator.expand(), self.denominator.expand())

    def __eq__(self, obj):
        """Define the Equality of two Object"""
        
        return (self.numerator == obj.numerator) and \
                                        (self.denominator == obj.denominator)
    
    def __mul__(self, obj):
        """multiply by an object"""
        
        if isinstance(obj, FracVariable):
            return FracVariable(self.numerator * obj.numerator, \
                                 self.denominator * obj.denominator)
        else:
            return FracVariable(self.numerator * obj, self.denominator)
    
    __rmul__ = __mul__    
    
    def __div__(self, obj):
        """Deal with division"""
        # Multiply the denominator by one to ensure that the new object didn't 
        #have a pointer through the old one
        return FracVariable(self.numerator * 1, self.denominator * obj)
    

        
    __truediv__ = __div__
    
    def __add__(self, obj):
        """We don't need to deal with addition-substraction on this type of Object"""
        if isinstance(obj, Number):
            if obj:
                self.numerator += obj * self.denominator
            return self
        assert(obj.vartype == 3)
        
        if self.denominator != obj.denominator:
            return NotImplemented('The Denominator should be the Same')
        else:
            new = FracVariable(self.numerator + obj.numerator, self.denominator)
            return new
     
    def __str__(self):
        if self.vartype == 4: #isinstance(self.numerator, LorentzObjectRepresentation):
            text = 'number of lorentz index :' + str(len(self.numerator.lorentz_ind)) + '\n'
            text += str(self.numerator.lorentz_ind)
            text += 'number of spin index :' + str(len(self.numerator.spin_ind)) + '\n'
            #text += 'other info ' + str(self.numerator.tag) + '\n'
            for ind in self.numerator.listindices():
                ind = tuple(ind)
                text += str(ind) + ' --> '
                if self.numerator.get_rep(ind) == 0:
                    text += '0\n'
                else:
                    text += '[ ' + str(self.numerator.get_rep(ind)) + ' ] / [ ' + \
                                        str(self.denominator.get_rep([0])) + ']\n'
            return text
        else:
            return '%s / %s' % (self.numerator, self.denominator)    
    
#===============================================================================
# AddVariable
#===============================================================================        
class AddVariable(list):
    """ A list of Variable/ConstantObject/... This object represent the operation
    between those object."""    
    
    #variable to fastenize class recognition
    vartype = 1
    
    def __init__(self, old_data=[], prefactor=1):
        """ initialization of the object with default value """
                
        self.prefactor = prefactor
        #self.tag = set()
        list.__init__(self, old_data)
        
    def copy(self):
        """ return a deep copy of the object"""
        return self.__class__(self, self.prefactor)
 
    def simplify(self, short=False):
        """ apply rule of simplification """
        
        # deal with one length object
        if len(self) == 1:
            return self.prefactor * self[0].simplify()
        
        if short:
            return self
        
        # check if more than one constant object
        constant = 0        
        # simplify complex number/Constant object
        for term in self[:]:
            if not hasattr(term, 'vartype'):
                constant += term
                self.remove(term)
            elif term.vartype == 5:
                constant += term.prefactor * term.value
                self.remove(term)
                

        # contract identical term and suppress null term
        varlen = len(self)
        i = -1
        while i > -varlen:
            j = i - 1
            while j >= -varlen:                
                if self[i] == self[j]:
                    self[i].prefactor += self[j].prefactor
                    del self[j]
                    varlen -= 1 
                else:
                    j -= 1
            if abs(self[i].prefactor) < precision_cut:
                del self[i]
                varlen -= 1
            else:
                i -= 1
                
        if constant:
            self.append(ConstantObject(constant))
            varlen += 1
             
        # deal with one/zero length object
        if varlen == 1:
            return self.prefactor * self[0].simplify()
        elif varlen == 0: 
            return ConstantObject()
        
        
        
        return self
    
    def expand(self):
        """Pass from High level object to low level object"""
        
        if not self:
            return self
        
        new = self[0].expand()
        for item in self[1:]:
            obj = item.expand()
            new += obj

        return new
        
    
    def __mul__(self, obj):
        """define the multiplication of 
            - a AddVariable with a number
            - a AddVariable with an AddVariable
        other type of multiplication are define via the symmetric operation base
        on the obj  class."""
        
        if not hasattr(obj, 'vartype'): #  obj is a number
            if not obj:
                return 0
            new = self.__class__([], self.prefactor)
            new[:] = [obj * term for term in self]
            return new
        
        elif obj.vartype == 1: # obj is an AddVariable
            new = AddVariable()
            for term in self:
                new += term * obj
            return new
        else:
            #force the program to look at obj + self
            return NotImplemented
    
    def __add__(self, obj):
        """Define all the different addition."""
        
 
        if not obj: # obj is zero
            return self
        #DECOMENT FOR SPIN2 PROPAGATOR COMPUTATION
        elif isinstance(obj, Number):
            obj = ConstantObject(obj)
            new = AddVariable(self, self.prefactor)
            new.append(obj)
            return new 
        elif not obj.vartype: # obj is a Variable
            new = AddVariable(self, self.prefactor)
            obj = obj.copy()
            new.append(obj)
            return new
        
        elif obj.vartype == 2: # obj is a MultVariable
            new = AddVariable(self, self.prefactor)
            obj = obj.__class__(obj, obj.prefactor)
            new.append(obj)
            return new     
           
        elif obj.vartype == 1: # obj is a AddVariable
            new = AddVariable(list.__add__(self, obj)) 
            return new
        else:
            #force to look at obj + self
            return NotImplemented

    def __div__(self, obj):
        """ Implement division"""
        
        if not hasattr(obj, 'vartype'): #obj is a Number
            factor = 1 / obj 
            return factor * self
        else:
            new_num = AddVariable(self, self.prefactor)
            new_denom = obj.copy()
            
            return FracVariable(new_num, new_denom)
 
    def __rdiv__(self, obj):
        """Deal division in a inverse way"""
        
        new = AddVariable(self, self.prefactor)
        if not hasattr(obj, 'vartype'):
            return FracVariable(obj, new)
        else:
            return NotImplemented  
        
    def __sub__(self, obj):
        return self + (-1) * obj
 
    __radd__ = __add__
    __iadd__ = __add__
    __rmul__ = __mul__ 
    __truediv__ = __div__  
    __rtruediv__ = __rdiv__
    
    def __rsub__(self, obj):
        return (-1) * self + obj   
                
    def append(self, obj):
        """ add a newVariable in the Multiplication list """
        if obj.prefactor:
            list.append(self, obj)
            
    def __eq__(self, obj):
        """Define The Equality"""

        if self.__class__ != obj.__class__:
            return False
        
        for term in self:
            self_prefactor = [term2.prefactor for term2 in self if term2 == term]
            obj_prefactor = [term2.prefactor for term2 in obj if term2 == term]
            if len(self_prefactor) != len(obj_prefactor):
                return False
            self_prefactor.sort()
            obj_prefactor.sort()
            if self_prefactor != obj_prefactor:
                return False
        
        #Pass all the test
        return True
    
    def __ne__(self, obj):
        """Define the unequality"""
        return not self.__eq__(obj)
        
    def __str__(self):
        text = ''
        if self.prefactor != 1:
            text += str(self.prefactor) + ' * '
        text += '( '
        text += ' + '.join([str(item) for item in self])
        text += ' )'
        return text
    
 
    def count_term(self):
        # Count the number of appearance of each variable and find the most 
        #present one in order to factorize her
        count = {}
        max, maxvar = 0, None
        for term in self:
            if term.vartype == 2: # term is MultVariable -> look inside
                for var in term:
                    count[var.variable] = count.setdefault(var.variable, 0) + 1
                    if count[var.variable] > max:
                        #update the maximum if reached
                        max, maxvar = max + 1, var
            else: #term is Varible -> direct update
                count[term.variable] = count.setdefault(term.variable, 0) + 1
                if count[term.variable] > max:
                    max, maxvar = max + 1, term
        
        return max, maxvar
    
    def factorize(self):
        """ try to factorize as much as possible the expression """

        #import aloha 
        #aloha.depth += 1 #global variable for debug
        max, maxvar = self.count_term()
        maxvar = maxvar.__class__(maxvar.variable)
        if max <= 1:
            #no factorization possible
            #aloha.depth -=1
            return self
        else:
            # split in MAXVAR * NEWADD + CONSTANT
            #print " " * 4 * aloha.depth + "start fact", self
            newadd = AddVariable()
            constant = AddVariable()
            #fill NEWADD and CONSTANT
            for term in self:
                if maxvar == term:
                    term = term.copy()
                    term.power -= 1
                    if term.power:
                        newadd.append(term.simplify())
                    else:
                        newadd.append(ConstantObject(term.prefactor))
                elif term.vartype == 2 : #isinstance(term, MultVariable):
                    if maxvar in term:
                        newterm = MultVariable([], term.prefactor)
                        for fact in term:
                            if fact == maxvar:
                                if fact.power > 1:
                                    newfact = fact.copy()
                                    newfact.power -= 1
                                    newterm.append(newfact)
                            else:
                                newterm.append(fact)
                        newadd.append(newterm.simplify())
                    else:
                        constant.append(term)
                else:
                    constant.append(term)

            #factorize the result
            try:
                cur_len = len(newadd)
            except:
                cur_len = 0
            if cur_len > 1:
                try:
                    newadd = newadd.factorize()
                except Exception, error:
                    raise
                    #raise Exception('fail to factorize: %s' % error)
            else:
                #take away the useless AddVariable to going back to a Variable class
                newadd = newadd[0]


            # recombine the factor. First ensure that the power of the object is
            #one. Then recombine
            if maxvar.power > 1:
                maxvar = maxvar.copy()
                maxvar.power = 1
            
            
            
            if newadd.vartype == 2: # isinstance(newadd, MultVariable):
                newadd.append(maxvar)
            else:
                newadd = MultVariable([maxvar, newadd])

            #simplify structure:
            if len(constant) == 1:
                constant = constant[0]
            if len(newadd) == 1:
                newadd = newadd[0] 
                        
            if constant:
                constant = constant.factorize()
                #aloha.depth -=1
                # ' ' * 4 * aloha.depth + 'return', AddVariable([newadd, constant])
                return AddVariable([newadd, constant])
            else:
                if constant.vartype == 5 and constant != 0:
                    #aloha.depth -=1
                    return AddVariable([newadd, constant])
                #aloha.depth -=1
                #print ' ' * 4 * aloha.depth + 'return:', newadd
                return newadd
        
#===============================================================================
# MultVariable
#===============================================================================
class MultVariable(list):
    """ A list of Variable with multiplication as operator between themselves.
    """

    add_class = AddVariable # define the class for addition of MultClass
    vartype = 2 # for istance check optimization
    
    def __init__(self, old_data=[], prefactor=1):
        """ initialization of the object with default value """
        
        self.prefactor = prefactor
        #self.tag = set()
        list.__init__(self, old_data)
        
    def copy(self):
        """ return a copy """
        return self.__class__(self, self.prefactor)
        
    def simplify(self):
        """ simplify the product"""
        
        #Ask to Simplify each term
        self[:] = [fact.simplify() for fact in self]
            
        #Check that we have at least two different factor if not returning
        #the content of the MultVariable
        if len(self) == 1:
            return self.prefactor * self[0]
        return self           
    
    def factorize(self):
        """Try to factorize this (nothing to do)"""
        return self
    
    def expand(self):
        """Pass from High Level object to low level object """
        
        out = self[0].expand()
        for fact in self[1:]:
            out *= fact.expand()
        out *= self.prefactor
        return out
     
    #Defining rule of Multiplication    
    def __mul__(self, obj):
        """Define the multiplication with different object"""
        
        if not hasattr(obj, 'vartype'): # should be a number
            return self.__class__(self, self.prefactor * obj)
        
        elif not obj.vartype: # obj is a Variable
            return NotImplemented # Use the one ov Variable
        
        elif obj.vartype == 2: # obj is a MultVariable
            new = self.__class__(self, self.prefactor)
            for fact in obj:
                new.append(fact)
            new.prefactor *= obj.prefactor
            return new  
                
        elif obj.vartype == 1: # obj is an AddVariable
            new = self.add_class(obj, obj.prefactor)
            new[:] = [data.__mul__(self) for data in new]
            return new

        else: 
            #look at  obj * self
            return NotImplemented
                
                
    def __add__(self, obj):
        """ define the adition with different object"""
        
        if not obj:
            return self
        #DECOMENT FOR SPIN2 PROPAGATOR COMPUTATION
        elif isinstance(obj, Number):
            obj = ConstantObject(obj)
            self.add_class()
            new = self.add_class()
            new.append(obj)
            new.append(self.__class__(self, self.prefactor))
            return new                      
        elif obj.vartype == 2: # obj is MultVariable
            new = self.add_class()
            new.append(self.__class__(self, self.prefactor))
            new.append(self.__class__(obj, obj.prefactor))
            return new
        else:
            #call the implementation of addition implemented in obj
            return NotImplemented
               
    def __sub__(self, obj):
        return self + (-1) * obj
    
    def __neg__(self):
        return (-1) * self
    
    def __rsub__(self, obj):
        return (-1) * self + obj
    
    def __div__(self, obj):
        """ define the division """
        if not hasattr(obj, 'vartype'):
            factor = 1 / obj 
            return factor * self
        else:
            return FracVariable(1 * self, 1 * obj)
        
    def __rdiv__(self, obj):
        """Deal division in a inverse way"""
        
        new = self.__class__(self, self.prefactor)
        if not hasattr(obj, 'vartype'):
            return FracVariable(obj, new)
        else:
            return NotImplemented
        
    __radd__ = __add__
    __iadd__ = __add__
    __rmul__ = __mul__    
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
       

    def append(self, obj):
        """ add a newVariable in the Multiplication list and look for her power
        """

        self.prefactor *= obj.prefactor
        if obj in self:
            index = self.index(obj)
            self[index] = self[index].copy()
            self[index].power += obj.power
        else:
            obj.prefactor = 1
            list.append(self, obj)
        
    def __eq__(self, obj):
        """Define When two MultVariable are identical"""
        
        try:    
            if obj.vartype !=2 or len(self) != len(obj):
                return False
        except:
            return False
        else:
            l1=[(var.variable, var.power) for var in self]
            for var in obj:
                if not (var.variable, var.power) in l1:
                    return False
            return True
    
    def __ne__(self, obj):
        """ Define when two Multvariable are not identical"""
        return not self.__eq__(obj)
                    
    def __str__(self):
        """ String representation """
        
        text = ''
        if self.prefactor != 1:
            text += str(self.prefactor) + ' * '
        text += '( '
        text += ' * '.join([str(item) for item in self])
        text += ' )'
        return text

    __rep__ = __str__

#===============================================================================
# Variable
#===============================================================================
class Variable(object):
    """This is the standard object for all the variable linked to expression.
    the variable 'X' is associte to a prefactor and a constant term. Such that
    this object can be reprensentet mathematicaly by a * X + b 
    """
    
    mult_class = MultVariable # Which class for multiplication with Variable
    add_class = AddVariable #which class for addition with Variable
    vartype = 0 # optimization for instance class recognition
    contract_first = 0 # Optimization for the order of the contracting object
                       #object defining contract first to 1 are the starting point
                       #of a chain of contraction following indices.
                       
    class VariableError(Exception):
        """class for error in Variable object"""
        pass
    
    def __init__(self, prefactor=1, variable='x', power=1):
        """ [prefactor] * Variable ** [power]"""

        self.prefactor = prefactor
        self.variable = variable
        self.power = power

    def copy(self):
        """provide an indenpedant copy of the object"""
        
        new = Variable(self.prefactor, self.variable, self.power)
        return new
        
    def simplify(self):
        """Define How to simplify this object."""
        return self
    
    def expand(self):
        """Return a more basic representation of this variable."""
        return self
    
    def factorize(self):
        """ try to factorize this"""
        return self
    
    # Define basic operation (multiplication, addition, soustraction)    
    def __mul__(self, obj):
        """ How to multiply object together 
            product of Variable -> MultVariable    
        """
        
        if not hasattr(obj, 'vartype'): # obj is a number
            if not obj:
                return 0
            out = self.copy()
            out.prefactor *= obj
            return out
        
        elif not obj.vartype: # obj is a Variable
            new = self.mult_class()
            new.append(self.copy())
            new.append(obj.copy())
            return new

        elif obj.vartype == 2: #obj is a MultVariable
            out = self.mult_class()

            for factor in obj:  
                out.append(factor.copy())
            out.append(self.copy())
            out.prefactor *= obj.prefactor
            return out
                
        elif obj.vartype == 1: # obj is a AddVariable
            new = AddVariable()
            
            for term in obj:
                new += self * term
            return new 
        else:
            #apply obj * self
            return NotImplemented
    
    def __pow__(self,power):
        """define power"""
        copy = self.copy()
        copy.power = self.power * power
        return copy

    def __add__(self, obj):
        """ How to make an addition
            Addition of Variable -> AddVariable
        """
        
        if not obj:
            return self
        
        try: 
            type = obj.vartype
        except: #float
            new = self.add_class()
            new.append(self.copy())
            new.append(ConstantObject(obj))
            return new
            
        
        if not type: # obj is a Variable
            new = self.add_class()
            new.append(self.copy())
            new.append(obj.copy())
            return new
        
        elif type == 2: # obj is a MultVariable
            new = self.add_class()
            new.append(self.copy())
            new.append(self.mult_class(obj, obj.prefactor))
            return new           
        
        elif type == 1: # obj is an AddVariable
            new = self.add_class(obj, obj.prefactor)
            new.append(self.copy())
            return new
        else:
            # apply obj + self
            return NotImplemented
       
    def __sub__(self, obj):
        return self + -1 * obj

    def __rsub__(self, obj):
        return (-1) * self + obj

    def __div__(self, obj):
        """ define the division """

        if not hasattr(obj, 'vartype'):
            factor = 1 / obj 
            return factor * self
        else:
            return FracVariable(1 * self, 1 * obj)
        
    def __rdiv__(self, obj):
        """Deal division in a inverse way"""
        
        new = self.copy()
        if not hasattr(obj, 'vartype'):
            return FracVariable(obj, new)
        else:
            return NotImplemented  
        
    __radd__ = __add__
    __iadd__ = __add__
    __rmul__ = __mul__    
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
        
    def __eq__(self, obj):
        """ identical if the variable is the same """
        if hasattr(obj,'vartype'):
            return not obj.vartype and self.variable == obj.variable
        else:
            return False
        
    def __str__(self):
        text = ''
        if self.prefactor != 1:
            text += str(self.prefactor) + ' * '
        text += str(self.variable)
        if self.power != 1:
            text += '**%s' % self.power
        return text
   
    __repr__ = __str__

#===============================================================================
# ScalarVariable
#===============================================================================
class ScalarVariable(Variable):
    """ A concrete symbolic scalar variable
    """
    
    def __init__(self, variable_name, prefactor=1, power=1):
        """ initialization of the object with default value """
        
        Variable.__init__(self, prefactor, variable_name, power)
        
        
    def copy(self):
        """ Define a independant copy of the object"""
        new = ScalarVariable(self.variable, self.prefactor, self.power) 
        return new

#===============================================================================
# AddLorentz
#===============================================================================  
# Not needed at present stage

 
#===============================================================================
# MultLorentz
#===============================================================================  
class MultLorentz(MultVariable):
    """Specific class for LorentzObject Multiplication"""
    
    add_class = AddVariable # Define which class describe the addition
        
    def find_lorentzcontraction(self):
        """return of (pos_object1, indice1) ->(pos_object2,indices2) definng
        the contraction in this Multiplication."""
        
        out = {}
        len_mult = len(self) 
        # Loop over the element
        for i, fact in enumerate(self):
            # and over the indices of this element
            for j in range(len(fact.lorentz_ind)):
                # in order to compare with the other element of the multiplication
                for k in range(i+1,len_mult):
                    fact2 = self[k]
                    try:
                        l = fact2.lorentz_ind.index(fact.lorentz_ind[j])
                    except:
                        pass
                    else:
                        out[(i, j)] = (k, l)
                        out[(k, l)] = (i, j)
        return out
        
    def find_spincontraction(self):
        """return of (pos_object1, indice1) ->(pos_object2,indices2) defining
        the contraction in this Multiplication."""

        out = {}
        len_mult = len(self)
        # Loop over the element
        for i, fact in enumerate(self):
            # and over the indices of this element
            for j in range(len(fact.spin_ind)):
                # in order to compare with the other element of the multiplication
                for k in range(i+1, len_mult):
                    fact2 = self[k]
                    try:
                        l = fact2.spin_ind.index(fact.spin_ind[j])
                    except:                
                        pass
                    else:
                        out[(i, j)] = (k, l)  
                        out[(k, l)] = (i, j)
        
        return out  

    def expand(self):
        """ expand each part of the product and combine them.
            Try to use a smart order in order to minimize the number of uncontracted indices.
        """

        self.unused = self[:] # list of not expanded
        # made in a list the intersting starting point for the computation
        basic_end_point = [var for var in self if var.contract_first] 
        product_term = [] #store result of intermediate chains
        current = None # current point in the working chain
        
        while self.unused:
            #Loop untill we have expand everything
            if not current:
                # First we need to have a starting point
                try: 
                    # look in priority in basic_end_point (P/S/fermion/...)
                    current = basic_end_point.pop()
                except:
                    #take one of the remaining
                    current = self.unused.pop()
                else:
                    #check that this one is not already use
                    if current not in self.unused:
                        current = None
                        continue
                    #remove of the unuse (usualy done in the pop)
                    self.unused.remove(current)
                
                # initialize the new chain
                product_term.append(current.expand())
            
            # We have a point -> find the next one
            var = self.neighboor(current)
            # provide one term which is contracted with current and which is not
            #yet expanded.
            if var:
                product_term[-1] *= var.expand()
                current = var
                self.unused.remove(current)
                continue
        
            current = None

        # Multiply all those current
        out = self.prefactor
        for fact in product_term:
            out *= fact
        return out

    def neighboor(self, home):
        """return one variable which are contracted with var and not yet expanded"""
        
        for var in self.unused:
            if var.has_component(home.lorentz_ind, home.spin_ind):
                return var
        return None

    def check_equivalence(self, obj, i, j, lorentzcontractself, lorentzcontractobj, \
                          spincontractself, spincontractobj, map):
        """check if i and j are compatible up to sum up indices"""
        
        # Fail if not the same class
        if self[i].__class__ != obj[j].__class__:
            return False
        # Fail if not linked to the same particle
        if hasattr(self[i], 'particle') or hasattr(obj[j], 'particle'):
            try:
                samepart = (self[i].particle == obj[j].particle)
            except:
                return False
            
            if not samepart:
                return False
        
        # Check if an assignement already exist for any of the factor consider
        if map.has_key(i):
            if map[i] != j:
                # The self factor does't point on the obj factor under focus 
                return False
        elif j in map.values():
            # the obj factor is already mapped by another variable
            return False
        
        # Check if the tag information is identical
        #if self[i].tag != obj[j].tag:
        #    return False
        
        # Check all lorentz indices
        for k in range(len(self[i].lorentz_ind)):
            # Check if the same indices
            if self[i].lorentz_ind[k] == obj[j].lorentz_ind[k]:
                continue
            # Check if those indices are contracted
            if (i, k) not in lorentzcontractself or \
                                            (j, k) not in lorentzcontractobj:
                return False
            
            #return object-indices of contraction
            i2, k2 = lorentzcontractself[(i, k)]
            j2, l2 = lorentzcontractobj[(j, k)]
            
            # Check that both contract at same position
            if k2 != l2:
                return False
            
            # Check that both object are the same type
            if self[i2].__class__ != obj[j2].__class__:
                return False
            
            # Check if one of the term is already map
            if map.has_key(i2) and map[i2] != j2:
                if map[i2] != j2:
                    return False
            else:
                # if not mapped, map it
                map[i2] = j2

        
        # Do the same but for spin indices
        for k in range(len(self[i].spin_ind)):
            # Check if the same indices
            if self[i].spin_ind[k] == obj[j].spin_ind[k]:
                continue
            #if not check if this indices is sum
            if (i, k) not in spincontractself or \
                                            (j, k) not in spincontractobj:
        
                return False
            
            #return correspondance
            i2, k2 = spincontractself[(i, k)]
            j2, l2 = spincontractobj[(j, k)]
            
            # Check that both contract at same position
            if k2 != l2:
                return False
            
            # Check that both object are the same type
            if self[i2].__class__ != obj[j2].__class__:
                return False
            
            # Check if one of the term is already map
            if map.has_key(i2) and map[i2] != j2:
                if map[i2] != j2:
                    return False
            else:
                map[i2] = j2
                   
        # If pass all the test then this is a possibility                
        return True

    def find_equivalence(self, obj, pos, lorentzcontractself, lorentzcontractobj, \
        spincontractself, spincontractobj, map):
        """Try to achieve to have a mapping between the two representations"""
        possibility = []
        
        for pos2 in range(len(obj)):
            init_map = dict(map)
            if self.check_equivalence(obj, pos, pos2, lorentzcontractself, \
                            lorentzcontractobj, spincontractself, \
                            spincontractobj, init_map):
                init_map[pos] = pos2
                possibility.append(init_map)
                
        if pos + 1 == len(self) and possibility:
                return True
        for map in possibility:
            if self.find_equivalence(obj, pos + 1, lorentzcontractself, \
                                        lorentzcontractobj, spincontractself, \
                                        spincontractobj, map):
                return True
        return False

    def __eq__(self, obj):
        
        # Check Standard Equality
        if MultVariable.__eq__(self, obj):
            return True 
        
        if self.__class__ != obj.__class__ or len(self) != len(obj):
            return False
        
        # Check that the number and the type of contraction are identical both
        #for spin and lorentz indices
        spin_contract_self = self.find_spincontraction()
        spin_contract_obj = obj.find_spincontraction()

        
        #check basic consitency
        if len(spin_contract_self) != len(spin_contract_obj):
            return False
        
        lorentz_contract_self = self.find_lorentzcontraction()
        lorentz_contract_obj = obj.find_lorentzcontraction()        
        #check basic consitency
        if len(lorentz_contract_self) != len(lorentz_contract_obj):
            return False

        # Try to achieve to have a mapping between the two representations
        mapping = {}
        a = self.find_equivalence(obj, 0, lorentz_contract_self, \
                              lorentz_contract_obj, spin_contract_self, \
                              spin_contract_obj, mapping)
        return a
            
#===============================================================================
# LorentzObject
#===============================================================================
class LorentzObject(Variable):
    """ A symbolic Object for All Helas object. All Helas Object Should 
    derivated from this class"""
    
    mult_class = MultLorentz # The class for the multiplication
    add_class = AddVariable # The class for the addition
    
    def __init__(self, lorentz_indices, spin_indices, prefactor=1, other_indices=[],
                                                                    variable=''):
        """ initialization of the object with default value """
        
        self.lorentz_ind = lorentz_indices
        self.spin_ind = spin_indices
        USE_TAG.update(set(other_indices))
        
        # Automatic variable_name creation. (interesting for debugging) and
        #help to compare object
        if not variable:
            variable = self.__class__.__name__
            if hasattr(self, 'particle'):
                variable += str(self.particle)
            if lorentz_indices:
                variable += '^{%s}' % ','.join([str(ind) for ind in lorentz_indices])
            if spin_indices:
                variable += '_{%s}' % ','.join([str(ind) for ind in spin_indices]) 
            
        #call the initialization of the basic class
        Variable.__init__(self, prefactor, variable)
    
    def __pow__(self, power):
        """ definition of an auto-contracted object """

        assert power == 2, "Lorentz object cann't have power higher than two"
        
        new = MultLorentz()
        new.append(self)
        new.append(self.copy())
        return new
    
    
    def copy(self):
        """return a shadow copy of the object. This is performed in calling 
        again the __init__ instance"""
        
        #computing of the argument depending of the class
        if self.__class__ == LorentzObject:
            arg = [self.lorentz_ind] + [self.spin_ind, self.prefactor ] 
        elif hasattr(self, 'particle'):
            arg = self.lorentz_ind + self.spin_ind + [self.particle] + \
                              [self.prefactor]
        else:
            arg = self.lorentz_ind + self.spin_ind + \
                              [self.prefactor]

        #call the init routine
        new = self.__class__(*arg)
        new.power = self.power
        return new
    
    def expand(self):
        """Expand the content information into LorentzObjectRepresentation."""

        try:
            self.representation
        except:
            self.create_representation()
        
        if self.power == 1:
            if self.prefactor == 1:
                return self.representation
            else:
                return self.prefactor * self.representation
        elif self.power == 2:
            # not possible to have power beyon 2 except for scalar object
            return self.prefactor * self.representation * self.representation
        else:
            assert self.lorentz_ind == self.spin_ind == []
            name =  self.representation.get_rep([0]).variable
            new = ScalarVariable(name, prefactor=self.prefactor,
                                                               power=self.power)
            return LorentzObjectRepresentation(
                                new, self.lorentz_ind, self.spin_ind)
        
    def create_representation(self):
        raise self.VariableError("This Object %s doesn't have define representation" % self.__class__.__name__)
    
    def __eq__(self, obj):
        """redifine equality"""
        
        return (self.__class__ == obj.__class__ and \
                    self.lorentz_ind == obj.lorentz_ind and \
                    self.spin_ind == obj.spin_ind and \
                    self.variable == obj.variable)
        
    def has_component(self, lor_list, spin_list):
        """check if this Lorentz Object have some of those indices"""
        

        for i in lor_list:
            if i in self.lorentz_ind:
                return True

        for i in spin_list:
            if i in self.spin_ind:
                return True



#===============================================================================
# IndicesIterator
#===============================================================================   
class IndicesIterator:
    """Class needed for the iterator"""
             
    def __init__(self, len):
        """ create an iterator looping over the indices of a list of len "len"
        with each value can take value between 0 and 3 """
        
        self.len = len # number of indices
        if len:
            # initialize the position. The first position is -1 due to the method 
            #in place which start by rising an index before returning smtg
            self.data = [-1] + [0] * (len - 1)
        else:
            # Special case for Scalar object
            self.data = 0
            self.next = self.nextscalar
                
    def __iter__(self):
        return self

    def next(self):
        for i in range(self.len):
            if self.data[i] < 3:
                self.data[i] += 1
                return self.data
            else:
                self.data[i] = 0
        raise StopIteration
            
    def nextscalar(self):
        if self.data:
            raise StopIteration
        else:
            self.data = 1
            return [0]

#===============================================================================
# LorentzObjectRepresentation
#===============================================================================            
class LorentzObjectRepresentation(dict):
    """A concrete representation of the LorentzObject."""

    vartype = 4 # Optimization for instance recognition
    
    class LorentzObjectRepresentationError(Exception):
        """Specify error for LorentzObjectRepresentation"""

    def __init__(self, representation, lorentz_indices, spin_indices):
        """ initialize the lorentz object representation"""

        self.lorentz_ind = lorentz_indices #lorentz indices
        self.nb_lor = len(lorentz_indices) #their number
        self.spin_ind = spin_indices #spin indices
        self.nb_spin = len(spin_indices) #their number
        self.nb_ind = self.nb_lor + self.nb_spin #total number of indices
        
        #self.tag = set(other_indices) #some information
        
        #store the representation
        if self.lorentz_ind or self.spin_ind:
            dict.__init__(self, representation) 
        else:
            self[(0,)] = representation
            
    def copy(self):
        return self

    def __add__(self, obj, fact=1):
        assert(obj.vartype == 4 == self.vartype) # are LorentzObjectRepresentation
        
        if self.lorentz_ind != obj.lorentz_ind or self.spin_ind != obj.spin_ind:
            # if the order of indices are different compute a mapping 
            switch_order = []
            for value in self.lorentz_ind:
                try:
                    index = obj.lorentz_ind.index(value)
                except:
                    raise self.LorentzObjectRepresentationError("Invalid" + \
                    "addition. Object doen't have the same lorentz indices:" + \
                                "%s != %s" % (self.lorentz_ind, obj.lorentz_ind))
                else:
                    switch_order.append(index)
            for value in self.spin_ind:
                try:
                    index = obj.spin_ind.index(value)
                except:
                    raise self.LorentzObjectRepresentationError("Invalid" + \
                                "addition. Object doen't have the same indices %s != %s" % (self.spin_ind, obj.spin_ind) )
                else:
                    switch_order.append(self.nb_lor + index)            
            switch = lambda ind : tuple([ind[switch_order[i]] for i in range(len(ind))])
        else:
            # no mapping needed (define switch as identity)
            switch = lambda ind : (ind)
   
        
        assert tuple(self.lorentz_ind+self.spin_ind) == tuple(switch(obj.lorentz_ind+obj.spin_ind)), '%s!=%s' % (self.lorentz_ind+self.spin_ind, switch(obj.lorentz_ind+self.spin_ind))
        assert tuple(self.lorentz_ind) == tuple(switch(obj.lorentz_ind)), '%s!=%s' % (tuple(self.lorentz_ind), switch(obj.lorentz_ind))
        
        # define an empty representation
        new = LorentzObjectRepresentation({}, obj.lorentz_ind, obj.spin_ind)
        
        # loop over all indices and fullfill the new object         
        if fact == 1:
            for ind in self.listindices():
                value = obj.get_rep(ind) + self.get_rep(switch(ind))
                #value = self.get_rep(ind) + obj.get_rep(switch(ind))
                new.set_rep(ind, value)
        else:
            for ind in self.listindices():
                #permute index for the second object
                value = self.get_rep(switch(ind)) + fact * obj.get_rep(ind)
                #value = fact * obj.get_rep(switch(ind)) + self.get_rep(ind)
                new.set_rep(ind, value)            

        return new
    
    __iadd__ = __add__    

    def __sub__(self, obj):
        return self.__add__(obj, fact= -1)
    
    def __rsub__(self, obj):
        return obj.__add__(self, fact= -1)
    
    def __isub__(self, obj):
        return self.__add__(obj, fact= -1)
    
    
    def __mul__(self, obj):
        """multiplication performing directly the einstein/spin sommation.
        """
        
        if not hasattr(obj, 'vartype') or not self.vartype or obj.vartype==5:
            out = LorentzObjectRepresentation({}, self.lorentz_ind, self.spin_ind)
            for ind in out.listindices():
                out.set_rep(ind, obj * self.get_rep(ind))
            return out
        elif obj.vartype == 3 :
            out = self * obj.numerator
            out /= obj.denominator
            return out


        assert(obj.__class__ == LorentzObjectRepresentation), '%s is not valid class for this operation' %type(obj)
        
        # compute information on the status of the index (which are contracted/
        #not contracted
        l_ind, sum_l_ind = self.compare_indices(self.lorentz_ind, \
                                                                obj.lorentz_ind)
        s_ind, sum_s_ind = self.compare_indices(self.spin_ind, \
                                                                   obj.spin_ind)      
        if not(sum_l_ind or sum_s_ind):
            # No contraction made a tensor product
            return self.tensor_product(obj)
       
        # elsewher made a spin contraction
        # create an empty representation but with correct indices
        new_object = LorentzObjectRepresentation({}, l_ind, s_ind)
        #loop and fullfill the representation
        for indices in new_object.listindices():
            #made a dictionary (pos -> index_value) for how call the object
            dict_l_ind = self.pass_ind_in_dict(indices[:len(l_ind)], l_ind)
            dict_s_ind = self.pass_ind_in_dict(indices[len(l_ind):], s_ind)
            #add the new value
            new_object.set_rep(indices, \
                               self.contraction(obj, sum_l_ind, sum_s_ind, \
                                                 dict_l_ind, dict_s_ind))
        
        return new_object

    @staticmethod
    def pass_ind_in_dict(indices, key):
        """made a dictionary (pos -> index_value) for how call the object"""
        if not key:
            return {}
        out = {}
        for i, ind in enumerate(indices):
            out[key[i]] = ind
        return out

    @staticmethod
    def compare_indices(list1, list2):
        """return two list, the first one contains the position of non summed
        index and the second one the position of summed index."""
        
        #init object
        are_unique = []
        are_sum = []
        # loop over the first list and check if they are in the second list
        for indice in list1:
            if indice in list2:
                are_sum.append(indice)
            else:
                are_unique.append(indice)
        # loop over the second list for additional unique item
        for indice in list2:
            if indice not in are_sum:
                are_unique.append(indice)        

        # return value
        return are_unique, are_sum

    def contraction(self, obj, l_sum, s_sum, l_dict, s_dict):
        """ make the Lorentz/spin contraction of object self and obj.
        l_sum/s_sum are the position of the sum indices
        l_dict/s_dict are dict given the value of the fix indices (indices->value)
        """
        
        out = 0 # initial value for the output
        len_l = len(l_sum) #store len for optimization
        len_s = len(s_sum) # same
        
        # loop over the possibility for the sum indices and update the dictionary
        # (indices->value)
        for l_value in IndicesIterator(len_l):
            l_dict.update(self.pass_ind_in_dict(l_value, l_sum))
            for s_value in IndicesIterator(len_s): 
                #s_dict_final = s_dict.copy()
                s_dict.update(self.pass_ind_in_dict(s_value, s_sum))               
                 
                #return the indices in the correct order
                self_ind = self.combine_indices(l_dict, s_dict)
                obj_ind = obj.combine_indices(l_dict, s_dict)
                
                # call the object
                factor = self.get_rep(self_ind) 
                factor *= obj.get_rep(obj_ind)
                
                
                if factor:
                    #compute the prefactor due to the lorentz contraction
                    factor *= (-1) ** (len(l_value) - l_value.count(0))
                    out += factor                        
        return out

    def combine_indices(self, l_dict, s_dict):
        """return the indices in the correct order following the dicts rules"""
        
        out = []
        # First for the Lorentz indices
        for value in self.lorentz_ind:
            out.append(l_dict[value])
        # Same for the spin
        for value in self.spin_ind:
            out.append(s_dict[value])
            
        return out     

    def tensor_product(self, obj):
        """ return the tensorial product of the object"""
        assert(obj.vartype == 4) #isinstance(obj, LorentzObjectRepresentation))

        new_object = LorentzObjectRepresentation({}, \
                                           self.lorentz_ind + obj.lorentz_ind, \
                                           self.spin_ind + obj.spin_ind)

        #some shortcut
        lor1 = self.nb_lor
        lor2 = obj.nb_lor
        spin1 = self.nb_spin
        spin2 = obj.nb_spin
        
        #define how to call build the indices first for the first object
        if lor1 == 0 == spin1:
            #special case for scalar
            selfind = lambda indices: [0]
        else:
            selfind = lambda indices: indices[:lor1] + \
                                        indices[lor1 + lor2: lor1 + lor2 + spin1]
        
        #then for the second
        if lor2 == 0 == spin2:
            #special case for scalar
            objind = lambda indices: [0]
        else:
            objind = lambda indices: indices[lor1: lor1 + lor2] + \
                                        indices[lor1 + lor2 + spin1:]

        # loop on the indices and assign the product
        for indices in new_object.listindices():
            new_object.set_rep(indices, self.get_rep(tuple(selfind(indices))) * 
                                        obj.get_rep(tuple(objind(indices))))
        
        return new_object
    
  
    def __div__(self, obj):
        """ define division 
        Only division by scalar!!!"""
        
        out = LorentzObjectRepresentation({}, self.lorentz_ind, self.spin_ind)
        try:
            obj.vartype
        except:
            for ind in out.listindices():
                out.set_rep(ind, self.get_rep(ind) / obj)    
        else:
            for ind in out.listindices():
                out.set_rep(ind, self.get_rep(ind) / obj.get_rep([0]))
                
        return out
  
    __rmul__ = __mul__
    __imul__ = __mul__
    __truediv__ = __div__
    __rtruediv__ = __div__
    __rdiv__ = __div__     

    def factorize(self):
        """Try to factorize each component"""
        for ind, fact in self.items(): 
            if fact:
                self.set_rep(ind, fact.factorize())
        return self

    def simplify(self):
        """Check if we can simplify the object (check for non treated Sum)"""
        
        #Look for internal simplification
        for ind, term in self.items():
            if hasattr(term, 'vartype'):
                self[ind] = term.simplify()
        
        #no additional simplification    
        return self        
                        
    def listindices(self):
        """Return an iterator in order to be able to loop easily on all the 
        indices of the object."""
        return IndicesIterator(self.nb_ind)
        
    def get_rep(self, indices):
        """return the value/Variable associate to the indices"""

        return self[tuple(indices)]
    
    def set_rep(self, indices, value):
        """assign 'value' at the indices position"""
 
        self[tuple(indices)] = value
    
    def __eq__(self, obj):
        """Check that two representation are identical"""
        
        if self.__class__ != obj.__class__:
            if self.nb_spin == 0 == self.nb_lor and \
                isinstance(obj, Number): 
                return self.get_rep([0]) == obj
            else:
                return False
        if len(self.lorentz_ind) != len(obj.lorentz_ind):
            return False
        if len(self.spin_ind) != len(obj.spin_ind):
            return False
        
        for ind in self.listindices():
            self_comp = self.get_rep(ind)
            try:
                obj_comp = obj.get_rep(ind)
            except:
                return False
            
            if self_comp != obj_comp:
                return False
            if hasattr(self_comp, 'vartype'):
                if self_comp.prefactor != obj_comp.prefactor:
                    return False

        
        #Pass all the test
        return True
        
    def __str__(self):
        """ string representation """
        text = 'number of lorentz index :' + str(self.nb_lor) + '\n'
        text += 'number of spin index :' + str(self.nb_spin) + '\n'
        #text += 'other info ' + str(self.tag) + '\n'
        for ind in self.listindices():
            ind = tuple(ind)
            text += str(ind) + ' --> '
            text += str(self.get_rep(ind)) + '\n'
        return text

#===============================================================================
# ConstantObject
#===============================================================================
class ConstantObject(LorentzObjectRepresentation):
    
    vartype = 5
    lorentz_ind = []
    spin_ind = []
    nb_ind = 0
    #tag = []
    variable = '0'
    prefactor = 1    
    power = 1
    
    def __init__(self, var=0):

        self.value = var
        if var:
            self.variable = str(var)
#            self.vartype = 0 #act as a Variable

    def copy(self):
        """ return a copy of the object """
        return ConstantObject(self.value)
       
    def __add__(self, obj):
        """Addition with a constant"""

        if not self.value:
            return obj

        if not hasattr(obj, 'vartype'):
            return ConstantObject(self.value + obj)
        elif obj.vartype == 0:
            new = obj.add_class()
            new.append(obj.copy())
            new.append(self)
            return new
        elif obj.vartype == 5:
            return ConstantObject(self.value + obj.value)
        elif obj.vartype == 2:
            new = obj.add_class()
            new.append(obj)
            new.append(self)
            return new
        elif obj.vartype == 1:
            new = obj.__class__()
                        
            # Define in the symmetric
            return NotImplemented
        else:
            return  obj

    __radd__ = __add__
    __iadd__ = __add__

    def __mul__(self, obj):
        """Zero is absorbant"""
        if self.value:
            return self.value * obj
        else:
            return self
    __rmul__ = __mul__
    __imul__ = __mul__
    
    def get_rep(self, ind):
        """return representation"""
        return self.value
    
    def __eq__(self, obj):
        if type(obj) == ConstantObject:
            return obj.value == self.value
        elif self.value:
            return False
        elif obj:
            return False
        else:
            return True
    
    def expand(self):
        return self
    
    def __str__(self):
        return str(self.value)

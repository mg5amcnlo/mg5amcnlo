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
##   Diagram of Class
##
##    Variable (vartype:0)<--- ScalarVariable 
##                          |
##                          +- LorentzObject 
##                                
##
##    list <--- AddVariable (vartype :1)   
##           
##    array <--- MultVariable  <--- MultLorentz (vartype:2) 
##           
##    list <--- LorentzObjectRepresentation (vartype :4) <-- ConstantObject
##                                                               (vartype:5)
##
##    FracVariable (vartype:3)
##
##    MultContainer (vartype:6)
##
################################################################################
##
##   Variable is in fact Factory wich adds a references to the variable name
##   Into the KERNEL (Of class Computation) instantiate a real variable object
##   (of class C_Variable, DVariable for complex/real) and return a MUltVariable
##   with a single element.
##
##   Lorentz Object works in the same way.
##
################################################################################


from __future__ import division
from array import array
import collections
from fractions import Fraction
import numbers
import re
import aloha # define mode of writting

class defaultdict(collections.defaultdict):

    def __call__(self, *args):
        return defaultdict(int)

class Computation(dict):
    """ a class to encapsulate all computation. Limit side effect """
    
    def __init__(self):
        self.objs = []
        self.use_tag = set()
        self.id = -1
        self.reduced_expr = {}
        self.fct_expr = {}
        self.reduced_expr2 = {}
        self.inverted_fct = {}
        self.has_pi = False # logical to check if pi is used in at least one fct
        self.unknow_fct = []
        dict.__init__(self)

    def clean(self):
        self.__init__()
        self.clear()
        
    def add(self, name, obj):
        self.id += 1
        self.objs.append(obj)
        self[name] = self.id
        return self.id
        
    def get(self, name):
        return self.objs[self[name]]
    
    def add_tag(self, tag):
        self.use_tag.update(tag)
        
    def get_ids(self, variables):
        """return the list of identification number associate to the 
        given variables names. If a variable didn't exists, create it (in complex).
        """
        out = []
        for var in variables:
            try:
                id = self[var]
            except KeyError:
                assert var not in ['M','W']
                id = Variable(var).get_id()
            out.append(id)
        return out
        
        
    def add_expression_contraction(self, expression):

        str_expr = str(expression)
        if str_expr in self.reduced_expr:
            out, tag = self.reduced_expr[str_expr]
            self.add_tag((tag,))
            return out          
        if expression == 0:
            return 0
        new_2 = expression.simplify()
        if new_2 == 0:
            return 0
        # Add a new variable
        tag = 'TMP%s' % len(self.reduced_expr)
        new = Variable(tag)
        self.reduced_expr[str_expr] = [new, tag]  
        new_2 = new_2.factorize()
        self.reduced_expr2[tag] = new_2
        self.add_tag((tag,))
        #self.unknow_fct = []
        #return expression
        return new
    
    known_fct = ['/', 'log', 'pow', 'sin', 'cos', 'asin', 'acos', 'tan', 'cot', 'acot',
                 'theta_function', 'exp']
    def add_function_expression(self, fct_tag, *args):

        if not (fct_tag.startswith('cmath.') or fct_tag in self.known_fct or
                                       (fct_tag, len(args)) in self.unknow_fct):
            self.unknow_fct.append( (fct_tag, len(args)) )
        
        argument = []
        for expression in args:
            if isinstance(expression, (MultLorentz, AddVariable, LorentzObject)):
                try:
                    expr = expression.expand().get_rep([0])
                except KeyError, error:
                    if error.args != ((0,),):
                        raise
                    else:
                        raise aloha.ALOHAERROR, '''Error in input format. 
    Argument of function (or denominator) should be scalar.
    We found %s''' % expression
                new = expr.simplify()
                if not isinstance(new, numbers.Number):
                    new = new.factorize()
                argument.append(new)
            else:
                argument.append(expression)
        for arg in argument:
            val = re.findall(r'''\bFCT(\d*)\b''', str(arg))
            for v in val:
                self.add_tag(('FCT%s' % v,))

        # check if the function is a pure numerical function.
        if (fct_tag.startswith('cmath.') or fct_tag in self.known_fct) and \
                    all(isinstance(x, numbers.Number) for x in argument):
            print "start the hack"
            import cmath
            if fct_tag.startswith('cmath.'):
                module = ''
            else:
                module = 'cmath.'
            try:
                return str(eval("%s%s(%s)" % (module,fct_tag, ','.join(`x` for x in argument))))
            except Exception, error:
                print error
                print "cmath.%s(%s)" % (fct_tag, ','.join(`x` for x in argument))
        if str(fct_tag)+str(argument) in self.inverted_fct:
            tag = self.inverted_fct[str(fct_tag)+str(argument)]
            v = tag.split('(')[1][:-1]
            self.add_tag(('FCT%s' % v,))
            return tag
        else:
            id = len(self.fct_expr)
            tag = 'FCT%s' % id
            self.inverted_fct[str(fct_tag)+str(argument)] = 'FCT(%s)' % id
            self.fct_expr[tag] = (fct_tag, argument) 
            self.reduced_expr2[tag] = (fct_tag, argument)
            self.add_tag((tag,))
            return 'FCT(%s)' % id
        
KERNEL = Computation()

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
        
    def simplify(self):
        """ apply rule of simplification """

        # deal with one length object
        if len(self) == 1:
            return self.prefactor * self[0].simplify()
        constant = 0
        items = {}
        pos = -1
        for term in self[:]:
            pos += 1 # current position in the real self
            if not hasattr(term, 'vartype'):
                if isinstance(term, dict):
                    # allow term of type{(0,):x}
                    assert term.values() == [0]
                    term = term[(0,)]
                constant += term
                del self[pos]
                pos -= 1
                continue
            tag = tuple(term.sort())
            if tag in items:
                orig_prefac = items[tag].prefactor # to assume to zero 0.33333 -0.3333
                items[tag].prefactor += term.prefactor
                if items[tag].prefactor and \
                    abs(items[tag].prefactor) / (abs(orig_prefac)+abs(term.prefactor)) < 1e-8:
                    items[tag].prefactor = 0
                del self[pos]
                pos -=1
            else:
                items[tag] = term.__class__(term, term.prefactor)
                self[pos] = items[tag]
        
        # get the optimized prefactor
        countprefact = defaultdict(int)
        nbplus, nbminus = 0,0
        if constant not in [0, 1,-1]:
            countprefact[constant] += 1
            if constant.real + constant.imag > 0:
                nbplus += 1
            else:
                nbminus += 1  
             
        for var in items.values():
            if var.prefactor == 0:
                self.remove(var)
            else:
                nb = var.prefactor
                if nb in [1,-1]:
                    continue
                countprefact[abs(nb)] +=1
                if nb.real + nb.imag > 0:
                    nbplus += 1
                else:
                    nbminus += 1
        if countprefact and max(countprefact.values()) >1:
            fact_prefactor = sorted(countprefact.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            fact_prefactor = 1
        if nbplus < nbminus:
                fact_prefactor *= -1
        self.prefactor *= fact_prefactor

        if fact_prefactor != 1:
            for i,a in enumerate(self):
                try:
                    a.prefactor /= fact_prefactor
                except AttributeError:
                    self[i] /= fact_prefactor  
                    
        if constant:
            self.append(constant/ fact_prefactor  )
            
        # deal with one/zero length object
        varlen = len(self)
        if varlen == 1:
            if hasattr(self[0], 'vartype'):
                return self.prefactor * self[0].simplify()
            else:
                #self[0] is a number
                return self.prefactor * self[0]
        elif varlen == 0:
            return 0 #ConstantObject()
        return self
    
    def split(self, variables_id):
        """return a dict with the key being the power associated to each variables
           and the value being the object remaining after the suppression of all
           the variable"""

        out = defaultdict(int)
        for obj in self:
            for key, value in obj.split(variables_id).items():
                out[key] += self.prefactor * value
        return out
    
    def contains(self, variables):
        """returns true if one of the variables is in the expression"""
        
        return any((v in obj for obj in self for v in variables  ))
    
    
    def get_all_var_names(self):
        
        out = []
        for term in self:
            if hasattr(term, 'get_all_var_names'):
                out += term.get_all_var_names()
        return out
            
    
    
    def replace(self, id, expression):
        """replace one object (identify by his id) by a given expression.
           Note that expression cann't be zero.
           Note that this should be canonical form (this should contains ONLY
           MULTVARIABLE) --so this should be called before a factorize.
        """
        new = self.__class__()
        
        for obj in self:
            assert isinstance(obj, MultVariable)
            tmp = obj.replace(id, expression)
            new += tmp
        new.prefactor = self.prefactor
        return new


    def expand(self, veto=[]):
        """Pass from High level object to low level object"""
        
        if not self:
            return self
        if self.prefactor == 1:
            new = self[0].expand(veto)
        else:
            new = self.prefactor * self[0].expand(veto)
        
        for item in self[1:]:
            if self.prefactor == 1:
                try:
                    new += item.expand(veto)
                except AttributeError:
                    new = new + item
                
            else:
                new += (self.prefactor) * item.expand(veto)
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
            return self.__class__(self, self.prefactor*obj)        
        elif obj.vartype == 1: # obj is an AddVariable
            new = self.__class__([],self.prefactor * obj.prefactor)
            new[:] = [i*j for i in self for j in obj]
            return new
        else:
            #force the program to look at obj + self
            return NotImplemented
        
    def __imul__(self, obj):
        """define the multiplication of 
            - a AddVariable with a number
            - a AddVariable with an AddVariable
        other type of multiplication are define via the symmetric operation base
        on the obj  class."""

        if not hasattr(obj, 'vartype'): #  obj is a number
            if not obj:
                return 0
            self.prefactor *= obj
            return self        
        elif obj.vartype == 1: # obj is an AddVariable
            new = self.__class__([], self.prefactor * obj.prefactor)
            new[:] = [i*j for i in self for j in obj]
            return new
        else:
            #force the program to look at obj + self
            return NotImplemented        

    def __neg__(self):
        self.prefactor *= -1
        return self

    def __add__(self, obj):
        """Define all the different addition."""

        if not hasattr(obj, 'vartype'):
            if not obj: # obj is zero
                return self
            new = self.__class__(self, self.prefactor)
            new.append(obj/self.prefactor)
            return new         
        elif obj.vartype == 2: # obj is a MultVariable
            new = AddVariable(self, self.prefactor)
            if self.prefactor == 1:
                new.append(obj)
            else:
                new.append((1/self.prefactor)*obj)
            return new     
        elif obj.vartype == 1: # obj is a AddVariable
            new = AddVariable(self, self.prefactor)
            for item in obj:
                new.append(obj.prefactor/self.prefactor * item) 
            return new
        else:
            #force to look at obj + self
            return NotImplemented

    def __iadd__(self, obj):
        """Define all the different addition."""

        if not hasattr(obj, 'vartype'):
            if not obj: # obj is zero
                return self
            self.append(obj/self.prefactor)
            return self
        elif obj.vartype == 2: # obj is a MultVariable
            if self.prefactor == 1:
                self.append(obj)
            else:
                self.append((1/self.prefactor)*obj)
            return self
        elif obj.vartype == 1: # obj is a AddVariable
            for item in obj:
                self.append(obj.prefactor/self.prefactor * item) 
            return self
        else:
            #force to look at obj + self
            return NotImplemented

    def __sub__(self, obj):
        return self + (-1) * obj
 
    def __rsub__(self, obj):
        return (-1) * self + obj
    
    __radd__ = __add__
    __rmul__ = __mul__ 


    def __div__(self, obj):
        return self.__mul__(1/obj)
    
    __truediv__ = __div__

    def __rdiv__(self, obj):
        return self.__rmult__(1/obj)

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
        count = defaultdict(int)
        correlation = defaultdict(defaultdict(int))
        for i,term in enumerate(self):
            try:
                set_term = set(term)
            except TypeError: 
                #constant term
                continue           
            for val1 in set_term:
                count[val1] +=1
                # allow to find optimized factorization for identical count
                for val2 in set_term:
                    correlation[val1][val2] += 1

        maxnb = max(count.values()) if count else 0
        possibility = [v for v,val in count.items() if val == maxnb]
        if maxnb == 1:
            return 1, None
        elif len(possibility) == 1:
            return maxnb, possibility[0]
            #import random
            #return maxnb, random.sample(possibility,1)[0]
            
            #return maxnb, possibility[0]
        max_wgt, maxvar = 0, None
        for var in possibility:
            wgt = sum(w**2 for w in correlation[var].values())/len(correlation[var])
            if wgt > max_wgt:
                maxvar = var
                max_wgt = wgt
                str_maxvar = str(KERNEL.objs[var])
            elif wgt == max_wgt:
                # keep the one with the lowest string expr
                new_str = str(KERNEL.objs[var])
                if new_str < str_maxvar:
                    maxvar = var
                    str_maxvar = new_str
        return maxnb, maxvar
    
    def factorize(self):
        """ try to factorize as much as possible the expression """

        max, maxvar = self.count_term()
        if max <= 1:
            #no factorization possible
            return self
        else:
            # split in MAXVAR * NEWADD + CONSTANT
            newadd = AddVariable()
            constant = AddVariable() 
            #fill NEWADD and CONSTANT
            for term in self:
                try:
                    term.remove(maxvar)
                except Exception:
                    constant.append(term)
                else:
                    if len(term):
                        newadd.append(term)
                    else: 
                        newadd.append(term.prefactor)
        newadd = newadd.factorize()
        
        # optimize the prefactor 
        if isinstance(newadd, AddVariable):        
            countprefact = defaultdict(int)
            nbplus, nbminus = 0,0
            for nb in [a.prefactor for a in newadd if hasattr(a, 'prefactor')]:
                countprefact[abs(nb)] +=1
                if nb.real + nb.imag > 0:
                    nbplus += 1
                else:
                    nbminus += 1
    
            newadd.prefactor = sorted(countprefact.items(), key=lambda x: x[1], reverse=True)[0][0]
            if nbplus < nbminus:
                newadd.prefactor *= -1
            if newadd.prefactor != 1:
                for i,a in enumerate(newadd):
                    try:
                        a.prefactor /= newadd.prefactor
                    except AttributeError:
                        newadd[i] /= newadd.prefactor
        
        
        if len(constant) > 1:
            constant = constant.factorize()
        elif constant:
            constant = constant[0]
        else:
            out = MultContainer([KERNEL.objs[maxvar], newadd])
            out.prefactor = self.prefactor
            if newadd.prefactor != 1:
                out.prefactor *= newadd.prefactor
                newadd.prefactor = 1 
            return out
        out = AddVariable([MultContainer([KERNEL.objs[maxvar], newadd]), constant],
                          self.prefactor)
        return out

class MultContainer(list):
    
    vartype = 6
    
    def __init__(self,*args):
        self.prefactor =1
        list.__init__(self, *args)
    
    def __str__(self):
        """ String representation """
        if self.prefactor !=1:
            text = '(%s * %s)' % (self.prefactor, ' * '.join([str(t) for t in self]))
        else:
            text = '(%s)' % (' * '.join([str(t) for t in self]))
        return text
    
    def factorize(self):        
        self[:] = [term.factorize() for term in self]

              
class MultVariable(array):
    """ A list of Variable with multiplication as operator between themselves.
    Represented by array for speed optimization
    """
    vartype=2
    addclass = AddVariable
    
    def __new__(cls, old=[], prefactor=1):
        return array.__new__(cls, 'i', old)
    
    
    def __init__(self, old=[], prefactor=1):
        """ initialization of the object with default value """        
        #array.__init__(self, 'i', old) <- done already in new !!
        self.prefactor = prefactor
        assert isinstance(self.prefactor, (float,int,long,complex))
    
    def get_id(self):
        assert len(self) == 1
        return self[0]
    
    def sort(self):
        a = list(self)
        a.sort()
        self[:] = array('i',a)
        return self
    
    def simplify(self):
        """ simplify the product"""
        if not len(self):
            return self.prefactor
        return self  
    
    def split(self, variables_id):
        """return a dict with the key being the power associated to each variables
           and the value being the object remaining after the suppression of all
           the variable"""

        key = tuple([self.count(i) for i in variables_id])
        arg = [id for id in self if id not in variables_id]
        self[:] = array('i', arg)
        return SplitCoefficient([(key,self)])
        
    def replace(self, id, expression):
        """replace one object (identify by his id) by a given expression.
           Note that expression cann't be zero.
        """
        assert hasattr(expression, 'vartype') , 'expression should be of type Add or Mult'
        
        if expression.vartype == 1: # AddVariable
            nb = self.count(id)
            if not nb:
                return self
            for i in range(nb):
                self.remove(id)
            new = self
            for i in range(nb):
                new *= expression
            return new             
        elif expression.vartype == 2: # MultLorentz
            # be carefull about A -> A * B
            nb = self.count(id)
            for i in range(nb):
                self.remove(id)
                self.__imul__(expression)
            return self           
#        elif expression.vartype == 0: # Variable
#           new_id = expression.id 
#            assert new_id != id
#            while 1:
#                try:
#                    self.remove(id)
#                except ValueError:
#                    break
#                else:
#                    self.append(new_id)
#            return self
        else:
            raise Exception, 'Cann\'t replace a Variable by %s' % type(expression)
        
    
    def get_all_var_names(self):
        """return the list of variable used in this multiplication"""
        return ['%s' % KERNEL.objs[n] for n in self]
        


    #Defining rule of Multiplication    
    def __mul__(self, obj):
        """Define the multiplication with different object"""
        
        if not hasattr(obj, 'vartype'): # should be a number
            if obj:
                return self.__class__(self, obj*self.prefactor)
            else:
                return 0
        elif obj.vartype == 1: # obj is an AddVariable
            new = obj.__class__([], self.prefactor*obj.prefactor)
            old, self.prefactor = self.prefactor, 1
            new[:] = [self * term for term in obj]
            self.prefactor = old
            return new
        elif obj.vartype == 4:
            return NotImplemented

        return self.__class__(array.__add__(self, obj), self.prefactor * obj.prefactor)
    
    __rmul__ = __mul__
        
    def __imul__(self, obj):
        """Define the multiplication with different object"""

        if not hasattr(obj, 'vartype'): # should be a number
            if obj:
                self.prefactor *= obj
                return self
            else:
                return 0
        elif obj.vartype == 1: # obj is an AddVariable
            new = obj.__class__([], self.prefactor * obj.prefactor)
            self.prefactor = 1
            new[:] = [self * term for term in obj]
            return new
        elif obj.vartype == 4:
            return NotImplemented

        self.prefactor *= obj.prefactor
        return array.__iadd__(self, obj)
    
    def __pow__(self,value):
        out = 1
        for i in range(value):
            out *= self
        return out
        

    def __add__(self, obj):
        """ define the adition with different object"""

        if not obj:
            return self
        elif not  hasattr(obj, 'vartype') or  obj.vartype == 2: 
            new = self.addclass([self, obj])
            return new                      
        else:
            #call the implementation of addition implemented in obj
            return NotImplemented
    __radd__ = __add__
    __iadd__ = __add__   
    
    def __sub__(self, obj):
        return self + (-1) * obj
    
    def __neg__(self):
        self.prefactor *=-1
        return self
    
    def __rsub__(self, obj):
        return (-1) * self + obj
    
    def __idiv__(self,obj):
        """ ONLY NUMBER DIVISION ALLOWED"""
        assert not hasattr(obj, 'vartype')
        self.prefactor  /= obj
        return self
    
    __div__ = __idiv__
    __truediv__ = __div__ 

    
    def __str__(self):
        """ String representation """
        t = ['%s' % KERNEL.objs[n] for n in self]
        if self.prefactor != 1:
            text = '(%s * %s)' % (self.prefactor,' * '.join(t))
        else:
            text = '(%s)' % (' * '.join(t))
        return text
        
    __rep__ = __str__
    
    def factorize(self):
        return self


#===============================================================================
# FactoryVar
#===============================================================================
class C_Variable(str):
    vartype=0
    type = 'complex'
    
class R_Variable(str):
    vartype=0
    type = 'double'

class ExtVariable(str):
    vartype=0
    type = 'parameter'


class FactoryVar(object):
    """This is the standard object for all the variable linked to expression.
    """
    mult_class = MultVariable # The class for the multiplication   
                      
    def __new__(cls, name, baseclass, *args):
        """Factory class return a MultVariable."""
        
        if name in KERNEL:
            return cls.mult_class([KERNEL[name]]) 
        else:
            obj = baseclass(name, *args)
            id = KERNEL.add(name, obj)
            obj.id = id
            return cls.mult_class([id])

class Variable(FactoryVar):
    
    def __new__(self, name, type=C_Variable):
        return FactoryVar(name, type)

class DVariable(FactoryVar):
    
    def __new__(self, name):
        
        if aloha.complex_mass:
            #some parameter are pass to complex
            if name[0] in ['M','W'] or name.startswith('OM'):
                return FactoryVar(name, C_Variable)
        if aloha.loop_mode and name.startswith('P'):
            return FactoryVar(name, C_Variable)        
        #Normal case:
        return FactoryVar(name, R_Variable)




#===============================================================================
# Object for Analytical Representation of Lorentz object (not scalar one)
#===============================================================================


#===============================================================================
# MultLorentz
#===============================================================================  
class MultLorentz(MultVariable):
    """Specific class for LorentzObject Multiplication"""
    
    add_class = AddVariable # Define which class describe the addition

    def find_lorentzcontraction(self):
        """return of (pos_object1, indice1) ->(pos_object2,indices2) defining
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
                    except Exception:
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
                    except Exception:                
                        pass
                    else:
                        out[(i, j)] = (k, l)  
                        out[(k, l)] = (i, j)
        
        return out
    
    def neighboor(self, home):
        """return one variable which are contracted with var and not yet expanded"""
        
        for var in self.unused:
            obj = KERNEL.objs[var]
            if obj.has_component(home.lorentz_ind, home.spin_ind):
                return obj
        return None
    

        

    def expand(self, veto=[]):
        """ expand each part of the product and combine them.
            Try to use a smart order in order to minimize the number of uncontracted indices.
            Veto forbids the use of sub-expression if it contains some of the variable in the 
            expression. Veto contains the id of the vetoed variables
        """

        self.unused = self[:] # list of not expanded
        # made in a list the interesting starting point for the computation
        basic_end_point = [var for var in self if KERNEL.objs[var].contract_first] 
        product_term = [] #store result of intermediate chains
        current = None # current point in the working chain
        
        while self.unused:
            #Loop untill we have expand everything
            if not current:
                # First we need to have a starting point
                try: 
                    # look in priority in basic_end_point (P/S/fermion/...)
                    current = basic_end_point.pop()
                except Exception:
                    #take one of the remaining
                    current = self.unused.pop()
                else:
                    #check that this one is not already use
                    if current not in self.unused:
                        current = None
                        continue
                    #remove of the unuse (usualy done in the pop)
                    self.unused.remove(current)
                cur_obj = KERNEL.objs[current] 
                # initialize the new chain
                product_term.append(cur_obj.expand())
            
            # We have a point -> find the next one
            var_obj = self.neighboor(product_term[-1])
            # provide one term which is contracted with current and which is not
            #yet expanded.
            if var_obj:
                product_term[-1] *= var_obj.expand()
                cur_obj = var_obj
                self.unused.remove(cur_obj.id)
                continue
        
            current = None


        # Multiply all those current 
        # For Fermion/Vector only one can carry index.
        out = self.prefactor
        for fact in product_term[:]:
            if hasattr(fact, 'vartype') and fact.lorentz_ind == fact.spin_ind == []:
                scalar = fact.get_rep([0])
                if hasattr(scalar, 'vartype') and scalar.vartype == 1:
                    if not veto or not scalar.contains(veto):
                        scalar = scalar.simplify()
                        prefactor = 1

                        if hasattr(scalar, 'vartype') and scalar.prefactor not in  [1,-1]:
                            prefactor = scalar.prefactor
                            scalar.prefactor = 1
                        new = KERNEL.add_expression_contraction(scalar)
                        fact.set_rep([0], prefactor * new)
            out *= fact
        return out

    def __copy__(self):
        """ create a shadow copy """
        new = MultLorentz(self)
        new.prefactor = self.prefactor
        return new

#===============================================================================
# LorentzObject
#===============================================================================
class LorentzObject(object):
    """ A symbolic Object for All Helas object. All Helas Object Should 
    derivated from this class"""
    
    contract_first = 0
    mult_class = MultLorentz # The class for the multiplication
    add_class = AddVariable # The class for the addition

    def __init__(self, name, lor_ind, spin_ind, tags=[]):
        """ initialization of the object with default value """
        assert isinstance(lor_ind, list)
        assert isinstance(spin_ind, list)
        
        self.name = name
        self.lorentz_ind = lor_ind
        self.spin_ind = spin_ind
        KERNEL.add_tag(set(tags))
        
    def expand(self):
        """Expand the content information into LorentzObjectRepresentation."""

        try:
            return self.representation
        except Exception:
            self.create_representation()
        return self.representation
    
    def create_representation(self):
        raise self.VariableError("This Object %s doesn't have define representation" % self.__class__.__name__)

    def has_component(self, lor_list, spin_list):
        """check if this Lorentz Object have some of those indices"""

        if any([id in self.lorentz_ind for id in lor_list]) or \
                               any([id in self.spin_ind for id in spin_list]):
            return True
        

    
    def __str__(self):
        return '%s' % self.name

class FactoryLorentz(FactoryVar):
    """ A symbolic Object for All Helas object. All Helas Object Should 
    derivated from this class"""
    
    mult_class = MultLorentz # The class for the multiplication
    object_class = LorentzObject # Define How to create the basic object.
    
    def __new__(cls, *args):
        name = cls.get_unique_name(*args)
        return FactoryVar.__new__(cls, name, cls.object_class, *args)
    
    @classmethod
    def get_unique_name(cls, *args):
        """default way to have a unique name"""
        return '_L_%(class)s_%(args)s' % \
                    {'class':cls.__name__,
                     'args': '_'.join(args)
                     }


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
                
        #store the representation
        if self.lorentz_ind or self.spin_ind:
            dict.__init__(self, representation) 
        elif isinstance(representation,dict):
            if len(representation) == 0:
                self[(0,)] = 0
            elif len(representation) == 1 and (0,) in representation:
                self[(0,)] = representation[(0,)]
            else:
                raise self.LorentzObjectRepresentationError("There is no key of (0,) in representation.")                    
        else:
            if isinstance(representation,dict):
                try:
                    self[(0,)] = representation[(0,)]
                except Exception:
                    if representation:
                        raise LorentzObjectRepresentation.LorentzObjectRepresentationError("There is no key of (0,) in representation.")
                    else:
                        self[(0,)] = 0
            else:
                self[(0,)] = representation

    def __str__(self):
        """ string representation """
        text = 'lorentz index :' + str(self.lorentz_ind) + '\n'
        text += 'spin index :' + str(self.spin_ind) + '\n'
        #text += 'other info ' + str(self.tag) + '\n'
        for ind in self.listindices():
            ind = tuple(ind)
            text += str(ind) + ' --> '
            text += str(self.get_rep(ind)) + '\n'
        return text

    def get_rep(self, indices):
        """return the value/Variable associate to the indices"""
        return self[tuple(indices)]
    
    def set_rep(self, indices, value):
        """assign 'value' at the indices position"""
 
        self[tuple(indices)] = value

    def listindices(self):
        """Return an iterator in order to be able to loop easily on all the 
        indices of the object."""
        return IndicesIterator(self.nb_ind)

    @staticmethod
    def get_mapping(l1,l2, switch_order=[]):
        shift = len(switch_order)
        for value in l1:
            try:
                index = l2.index(value)
            except Exception:
                raise LorentzObjectRepresentation.LorentzObjectRepresentationError(
                        "Invalid addition. Object doen't have the same lorentz "+ \
                        "indices : %s != %s" % (l1, l2))
            else:
                switch_order.append(shift + index)
        return switch_order
    
    
    def __add__(self, obj, fact=1):
        
        if not obj:
            return self
        
        if not hasattr(obj, 'vartype'):
            assert self.lorentz_ind == []
            assert self.spin_ind == []
            new = self[(0,)] + obj * fact
            out = LorentzObjectRepresentation(new, [], [])
            return out

        assert(obj.vartype == 4 == self.vartype) # are LorentzObjectRepresentation
        
        if self.lorentz_ind != obj.lorentz_ind or self.spin_ind != obj.spin_ind:
            # if the order of indices are different compute a mapping 
            switch_order = []
            self.get_mapping(self.lorentz_ind, obj.lorentz_ind, switch_order)
            self.get_mapping(self.spin_ind, obj.spin_ind, switch_order)
            switch = lambda ind : tuple([ind[switch_order[i]] for i in range(len(ind))])
        else:
            # no mapping needed (define switch as identity)
            switch = lambda ind : (ind)
   
        # Some sanity check  
        assert tuple(self.lorentz_ind+self.spin_ind) == tuple(switch(obj.lorentz_ind+obj.spin_ind)), '%s!=%s' % (self.lorentz_ind+self.spin_ind, switch(obj.lorentz_ind+self.spin_ind))
        assert tuple(self.lorentz_ind) == tuple(switch(obj.lorentz_ind)), '%s!=%s' % (tuple(self.lorentz_ind), switch(obj.lorentz_ind))
        
        # define an empty representation
        new = LorentzObjectRepresentation({}, obj.lorentz_ind, obj.spin_ind)
        
        # loop over all indices and fullfill the new object 
        if fact == 1:
            for ind in self.listindices():
                value = obj.get_rep(ind) + self.get_rep(switch(ind))
                new.set_rep(ind, value)
        else:
            for ind in self.listindices():
                value = fact * obj.get_rep(switch(ind)) + self.get_rep(ind)
                new.set_rep(ind, value)
                                    
        return new

    def __iadd__(self, obj, fact=1):
        
        if not obj:
            return self
        
        assert(obj.vartype == 4 == self.vartype) # are LorentzObjectRepresentation
        
        if self.lorentz_ind != obj.lorentz_ind or self.spin_ind != obj.spin_ind:
            
            # if the order of indices are different compute a mapping 
            switch_order = []
            self.get_mapping(obj.lorentz_ind, self.lorentz_ind, switch_order)
            self.get_mapping(obj.spin_ind, self.spin_ind, switch_order)
            switch = lambda ind : tuple([ind[switch_order[i]] for i in range(len(ind))])
        else:
            # no mapping needed (define switch as identity)
            switch = lambda ind : (ind)
   
        # Some sanity check  
        assert tuple(switch(self.lorentz_ind+self.spin_ind)) == tuple(obj.lorentz_ind+obj.spin_ind), '%s!=%s' % (switch(self.lorentz_ind+self.spin_ind), (obj.lorentz_ind+obj.spin_ind))
        assert tuple(switch(self.lorentz_ind) )== tuple(obj.lorentz_ind), '%s!=%s' % (switch(self.lorentz_ind), tuple(obj.lorentz_ind))
        
        # loop over all indices and fullfill the new object         
        if fact == 1:
            for ind in self.listindices():
                self[tuple(ind)] += obj.get_rep(switch(ind))
        else:
            for ind in self.listindices():
                self[tuple(ind)] += fact * obj.get_rep(switch(ind))             
        return self

    def __sub__(self, obj):
        return self.__add__(obj, fact= -1)
    
    def __rsub__(self, obj):
        return obj.__add__(self, fact= -1)
    
    def __isub__(self, obj):
        return self.__add__(obj, fact= -1)
    
    def __neg__(self):
        self *= -1
        return self
    
    def __mul__(self, obj):
        """multiplication performing directly the einstein/spin sommation.
        """
        
        if not hasattr(obj, 'vartype'):
            out = LorentzObjectRepresentation({}, self.lorentz_ind, self.spin_ind)
            for ind in out.listindices():
                out.set_rep(ind, obj * self.get_rep(ind))
            return out

        # Sanity Check
        assert(obj.__class__ == LorentzObjectRepresentation), \
                           '%s is not valid class for this operation' %type(obj)
                           
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
        
    __rmul__ = __mul__
    __imul__ = __mul__

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
                factor = obj.get_rep(obj_ind) * self.get_rep(self_ind)
                    
                if factor:
                    #compute the prefactor due to the lorentz contraction
                    try:
                        factor.prefactor *= (-1) ** (len(l_value) - l_value.count(0))
                    except Exception:
                        factor *= (-1) ** (len(l_value) - l_value.count(0))
                    out += factor                        
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
            
            fac1 = self.get_rep(tuple(selfind(indices)))
            fac2 = obj.get_rep(tuple(objind(indices)))            
            new_object.set_rep(indices, fac1 * fac2)
        
        return new_object

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

    @staticmethod
    def compare_indices(list1, list2):
        """return two list, the first one contains the position of non summed
        index and the second one the position of summed index."""
        #init object
        
        # equivalent set call --slightly slower
        #return list(set(list1) ^ set(list2)), list(set(list1) & set(list2))
        

        are_unique, are_sum = [], []
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
    
    @staticmethod
    def pass_ind_in_dict(indices, key):
        """made a dictionary (pos -> index_value) for how call the object"""
        if not key:
            return {}
        out = {}
        for i, ind in enumerate(indices):
            out[key[i]] = ind
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
    
    def split(self, variables_id):
        """return a dict with the key being the power associated to each variables
           and the value being the object remaining after the suppression of all
           the variable"""
           
        out = SplitCoefficient()
        zero_rep = {}
        for ind in self.listindices():
            zero_rep[tuple(ind)] = 0
        
        for ind in self.listindices():
            # There is no function split if the element is just a simple number
            if isinstance(self.get_rep(ind), numbers.Number):
                if tuple([0]*len(variables_id)) in out:
                    out[tuple([0]*len(variables_id))][tuple(ind)] += self.get_rep(ind)
                else:
                    out[tuple([0]*len(variables_id))] = \
                                 LorentzObjectRepresentation(dict(zero_rep), 
                                                         self.lorentz_ind, self.spin_ind)
                    out[tuple([0]*len(variables_id))][tuple(ind)] += self.get_rep(ind)
                continue

            for key, value in self.get_rep(ind).split(variables_id).items():
                if key in out:
                    out[key][tuple(ind)] += value
                else:
                    out[key] = LorentzObjectRepresentation(dict(zero_rep), 
                                                self.lorentz_ind, self.spin_ind)
                    out[key][tuple(ind)] += value
        
        return out
                    
           
        

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
            self.data = True
            return [0]

class SplitCoefficient(dict):
    
    def __init__(self, *args, **opt):
        dict.__init__(self, *args, **opt)
        self.tag=set()
        
    def get_max_rank(self):
        """return the highest rank of the coefficient"""
        
        return max([max(arg[:4]) for arg in self])

      
if '__main__' ==__name__:
    
    import cProfile
    def create():
        for i in range(10000):
            LorentzObjectRepresentation.compare_indices(range(i%10),[4,3,5])       
        
    cProfile.run('create()')

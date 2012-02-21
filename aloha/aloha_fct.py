################################################################################
#
# Copyright (c) 2012 The ALOHA Development team and Contributors
#
# This file is a part of the ALOHA project, an application which 
# automatically generates HELAS ROUTINES
#
# It is subject to the ALOHA license which should accompany this 
# distribution.
#
#
################################################################################
from aloha.aloha_object import *
import cmath

class WrongFermionFlow(Exception):
    pass

################################################################################    
##  CHECK FLOW VALIDITY OF A LORENTZ STRUCTURE
################################################################################    
def check_flow_validity(expression, nb_fermion):
    """Check that the fermion flow follows the UFO convention
       1) Only one flow is defined and is 1 -> 2, 3 -> 4, ...
       2) that 1/3/... are on the left side of any Gamma matrices
    """
    
    assert nb_fermion != 0 and (nb_fermion % 2) == 0
    
    # Need to expand the expression in order to have a simple sum of expression
    expr = eval(expression)
    expr = expr.simplify()
    #expr is now a valid AddVariable object if they are a sum or
    if expr.vartype != 1: # not AddVariable 
        expr = [expr] # put in a list to allow comparison
    
    for term in expr:
        if term.vartype == 0: # Single object
            if not term.spin_ind in [[1,2], [2,1]]:
                raise WrongFermionFlow, 'Fermion should be the first particles of any interactions'
            if isinstance(term, (Gamma, Gamma5, Sigma)):
                if not term.spin_ind == [2,1]:
                    raise WrongFermionFlow, 'Not coherent Incoming/outcoming fermion flow'
        
        elif term.vartype == 2: # product of object
            link, rlink = {}, {}
            for obj in term:
                if not obj.spin_ind:
                    continue
                ind1, ind2 = obj.spin_ind
                if isinstance(obj, (Gamma, Sigma)):
                    if (ind1 in range(1, nb_fermion+1) and ind1 % 2 == 1) or \
                       (ind2 in range(2, nb_fermion+1) and ind2 % 2 == 0 ):
                        raise WrongFermionFlow, 'Not coherent Incoming/outcoming fermion flow'
                if ind1 not in link.keys():
                    link[ind1] = ind2
                else:
                    rlink[ind1] = ind2
                if ind2 not in link.keys():
                    link[ind2] = ind1
                else: 
                    rlink[ind2] = ind1                    
            for i in range(1, nb_fermion,2):
                old = []
                pos = i
                while 1:
                    old.append(pos)
                    if pos in link.keys() and link[pos] not in old:
                        pos = link[pos]
                    elif pos in rlink.keys() and rlink[pos] not in old:
                        pos = rlink[pos]
                    elif pos != i+1:
                        raise WrongFermionFlow, 'Not coherent Incoming/outcoming fermion flow'
                    elif pos == i+1:
                        break
        

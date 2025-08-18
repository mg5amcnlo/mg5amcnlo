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
from __future__ import absolute_import
from aloha.aloha_object import *
import aloha.aloha_lib as aloha_lib
import cmath
from six.moves import range

import logging
logger = logging.getLogger('aloha')

class WrongFermionFlow(Exception):
    pass

################################################################################    
##  CHECK FLOW VALIDITY OF A LORENTZ STRUCTURE
################################################################################
def get_fermion_flow(expression, nb_fermion):
    """Get the fermion flow follows the UFO convention
        {I1:O1, I2:O2,...}"""
        
    assert nb_fermion != 0 and (nb_fermion % 2) == 0     
    # Need to expand the expression in order to have a simple sum of expression
    try:
        expr = eval(expression)
    except NameError as error:
        # handle formfactors
        logger.debug('Error in get_fermion_flow: %s -> handle like formfactor' % error)
        import re
        var_name = re.findall(r"'([^']*)'", str(error))[0]
        expr = expression.replace(var_name, '1') 
        return get_fermion_flow(expr, nb_fermion)
    except Exception as error:
        print(expression, error)
        return
    expr = expr.simplify()
    #expr is now a valid AddVariable object if they are a sum or
    if expr.vartype != 1: # not AddVariable 
        expr = [expr] # put in a list to allow comparison

    out = {}
    for term in expr:
        if term.vartype == 0: # Single object
            if not term.spin_ind in [[1,2], [2,1]]:
                raise WrongFermionFlow('Fermion should be the first particles of any interactions')
            if isinstance(term, (Gamma, Gamma5, Sigma)):
                if term.spin_ind == [2,1]:
                    out[1] = 2
                else:
                    out[2] = 1
            elif isinstance(term, Identity):
                out[1] = 2
                    
        elif term.vartype == 2: # product of object
            link, rlink = {}, {}
            for obj in term:
                obj = aloha_lib.KERNEL.objs[obj]
                if not obj.spin_ind:
                    continue
                ind1, ind2 = obj.spin_ind
                if ind1 not in list(link.keys()):
                    link[ind1] = ind2
                else:
                    raise WrongFermionFlow('a spin indices should appear only once on the left indices of an object: %s' % expr)
                if ind2 not in list(rlink.keys()):
                    rlink[ind2] = ind1
                else: 
                    raise WrongFermionFlow('a spin indices should appear only once on the left indices of an object: %s' % expr)             
             
            for i in range(1, nb_fermion):
                if i in list(out.keys()) or i in list(out.values()):
                    continue
                old = []
                pos = i
                while 1:
                    old.append(pos)
                    if pos in list(link.keys()) and link[pos] not in old:
                        pos = link[pos]
                    elif pos in list(rlink.keys()) and rlink[pos] not in old:
                        pos = rlink[pos]
                    else:
                        if pos in list(link.keys()) and i in list(rlink.keys()):
                            out[i] = pos
                            break
                        elif pos in list(rlink.keys()) and i in list(link.keys()):
                            out[pos] = i
                            break
                        else:
                            raise WrongFermionFlow('incoherent IO state: %s' % expr)
    if not len(out) == nb_fermion //2:
        raise WrongFermionFlow('Not coherent Incoming/outcoming fermion flow')
    return out


    
def check_flow_validity(expression, nb_fermion):
    """Check that the fermion flow follows the UFO convention
       1) Only one flow is defined and is 1 -> 2, 3 -> 4, ...
       2) that 1/3/... are on the left side of any Gamma matrices
    """
    
    assert nb_fermion != 0 and (nb_fermion % 2) == 0
    
    # Need to expand the expression in order to have a simple sum of expression
    try:
        expr = eval(expression)
    except Exception:
        return
    expr = expr.simplify()
    #expr is now a valid AddVariable object if they are a sum or
    if expr.vartype != 1: # not AddVariable 
        expr = [expr] # put in a list to allow comparison

    for term in expr:
        if term.vartype == 0: # Single object
            if not term.spin_ind in [[1,2], [2,1]]:
                raise WrongFermionFlow('Fermion should be the first particles of any interactions')
            if isinstance(term, (Gamma, Gamma5, Sigma)):
                if not term.spin_ind == [2,1]:
                    raise WrongFermionFlow('Not coherent Incoming/outcoming fermion flow')
        
        elif term.vartype == 2: # product of object
            link, rlink = {}, {}
            for obj in term:
                obj = aloha_lib.KERNEL.objs[obj]
                if not obj.spin_ind:
                    continue
                ind1, ind2 = obj.spin_ind
                if isinstance(obj, (Gamma, Sigma)):
                    if (ind1 in range(1, nb_fermion+1) and ind1 % 2 == 1) or \
                       (ind2 in range(2, nb_fermion+1) and ind2 % 2 == 0 ):
                        raise WrongFermionFlow('Not coherent Incoming/outcoming fermion flow')
                if ind1 not in list(link.keys()):
                    link[ind1] = ind2
                else:
                    rlink[ind1] = ind2
                if ind2 not in list(link.keys()):
                    link[ind2] = ind1
                else: 
                    rlink[ind2] = ind1                    
            for i in range(1, nb_fermion,2):
                old = []
                pos = i
                while 1:
                    old.append(pos)
                    if pos in list(link.keys()) and link[pos] not in old:
                        pos = link[pos]
                    elif pos in list(rlink.keys()) and rlink[pos] not in old:
                        pos = rlink[pos]
                    elif pos != i+1:
                        raise WrongFermionFlow('Not coherent Incoming/outcoming fermion flow')
                    elif pos == i+1:
                        break
   
def guess_routine_from_name(names):
    """ return (UFO name, tag , offshell) from a given name """
    
    output =[]
    for name in names:
        if name.startswith('MP_'):
            name = name[3:]
            tags = ['MP_']
        else: 
            tags = []
        
        data = name.split('_')
        if len(data) == 2:
            main, offshell = data
            multiple = []            
        else:
            main, multiple, offshell = data[0], data[1:-1],data[-1]
        
        # search for tag allow tag [L, L$, C$, MP]
        allow_tag = ['C1','C2','C3','C4','C5','C6','C7']
        allow_tag += ['L%s' % i for i in range(1,20)]
        allow_tag += ['P%s' % i for i in range(0,20)]
        allow_tag += ['L']
        tags = []
        
        len_tag = -1
        while len(tags) != len_tag:
            len_tag = len(tags)
            for tag in allow_tag:
                if multiple and multiple[-1].endswith(tag):
                    multiple[-1] = multiple[-1][:-len(tag)]
                    tags.append(tag)
                    break
                elif main.endswith(tag):
                    main = main[:-len(tag)]
                    tags.append(tag)
                    break
        
        
        # create the correct lorentz
        lorentz = [main]
        if multiple:
            base = main
            while base[-1].isdigit():
                base = base[:-1]
            for nb in multiple:
                lorentz.append('%s%s' % (base, nb))
        
        # add in the results
        output.append((tuple(lorentz), tuple(tags), int(offshell)))
    return output
         

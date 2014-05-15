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


class Rule(object):
    """ """
    
    def __init__(self, name, default, data,first=True, inverted_display=False):
        """ """
        self.name = name
        self.default=default
        self.status=default
        self.lhablock = data[0].lower()
        self.lhaid = data[1]
        self.value = data[2]
        self.first = first
        if inverted_display:
            self.display = lambda x: not x
        else:
            self.display = lambda x: x
            
class Category(list):
    """A container for the different rules"""
    
    def __init__(self, name, *args, **opt):
        """store a title for those restriction category"""
        
        self.name = name
        list.__init__(self, *args, **opt)
        
    def add_options(self, name='', default='', inverted_display=False, rules=[]):
        first=True
        for arg in rules:
            current_rule = Rule(name, default, arg, first, inverted_display) 
            self.append(current_rule)
            first=False


        
        
        
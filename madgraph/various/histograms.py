#! /usr/bin/env python3
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
"""Module for the handling of histograms, including Monte-Carlo error per bin
and scale/PDF uncertainties."""

from __future__ import division

from __future__ import absolute_import
import array
import collections
import errno
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
import copy
import fractions
import hashlib
import html as html_module
import itertools
import json
import logging
import math
import mmap
import os
import pprint
import random
import re
import sys
import tempfile

import subprocess
import xml.dom.minidom as minidom
from xml.parsers.expat import ExpatError as XMLParsingError
import io
file = io.IOBase

_NUMPY_NOT_CHECKED = object()
_numpy_module = _NUMPY_NOT_CHECKED


def _optional_numpy():
    """Load NumPy lazily for large aggregation jobs, with a pure-Python fallback."""

    global _numpy_module
    if _numpy_module is _NUMPY_NOT_CHECKED:
        try:
            import numpy
        except ImportError:
            _numpy_module = None
        else:
            _numpy_module = numpy
    return _numpy_module

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path)) 
sys.path.append(os.path.join(root_path,os.pardir))
try:
    # import from madgraph directory
    import madgraph.various.misc as misc
    from madgraph import MadGraph5Error
    logger = logging.getLogger("madgraph.various.histograms")

except ImportError as error:
    # import from madevent directory
    import internal.misc as misc    
    from internal import MadGraph5Error
    logger = logging.getLogger("internal.histograms")

# I copy the Physics object list here so as not to add a whole dependency to
# base_objects which is annoying when using this histograms module from the
# bin/internal location of a process output (i.e. outside an MG5_aMC env.)

#===============================================================================
# PhysicsObjectList
#===============================================================================
class histograms_PhysicsObjectList(list):
    """A class to store lists of physics object."""

    class PhysicsObjectListError(Exception):
        """Exception raised if an error occurs in the definition
        or execution of a physics object list."""
        pass

    def __init__(self, init_list=None):
        """Creates a new particle list object. If a list of physics 
        object is given, add them."""

        list.__init__(self)

        if init_list is not None:
            for object in init_list:
                self.append(object)
                
    def append(self, object):
        """Appends an element, but test if valid before."""
        
        assert self.is_valid_element(object), \
            "Object %s is not a valid object for the current list" % repr(object)

        list.append(self, object)
        

    def is_valid_element(self, obj):
        """Test if object obj is a valid element for the list."""
        return True

    def __str__(self):
        """String representation of the physics object list object. 
        Outputs valid Python with improved format."""

        mystr = '['

        for obj in self:
            mystr = mystr + str(obj) + ',\n'

        mystr = mystr.rstrip(',\n')

        return mystr + ']'
#===============================================================================

class Bin(object):
    """A class to store Bin related features and function.
    """
  
    def __init__(self, boundaries=(0.0,0.0), wgts=None, n_entries = 0):
        """ Initializes an empty bin, necessarily with boundaries. """

        self.boundaries = boundaries
        self.n_entries  = n_entries
        if not wgts:
            self.wgts       = {'central':0.0}
        else:
            self.wgts       = wgts
  
    def __setattr__(self, name, value):
        if name=='boundaries':
            if not isinstance(value, tuple):
                raise MadGraph5Error("Argument '%s' for bin property "
                    "'boundaries' must be a tuple."%str(value))
            else:
                for coordinate in value:
                    if isinstance(coordinate, tuple):
                        for dim in coordinate:
                            if not isinstance(dim, float):
                                raise MadGraph5Error("Coordinate '%s' of the bin "
                                  "boundary '%s' must be a float."%
                                  (str(dim), str(value)))
                    elif not isinstance(coordinate, float):
                        raise MadGraph5Error("Element '%s' of the bin boundaries "
                                     "specified must be a float."%str(coordinate))
        elif name=='wgts':
            if not isinstance(value, MutableMapping):
                raise MadGraph5Error("Argument '%s' for bin uncertainty "
                       "'wgts' must be a mutable mapping."%str(value))
            # Dense accumulator views have already validated their backing
            # storage. Iterating over all their values here would turn each
            # lightweight bin construction back into an O(number of weights)
            # operation.
            if not getattr(value, '_validated_histogram_weights', False):
                for val in value.values():
                    if not isinstance(val,float):
                        raise MadGraph5Error("The bin weight value '%s' is not a "
                                             "float."%str(val))
   
        super(Bin, self).__setattr__(name,value)
        
    def get_weight(self, key='central'):
        """ Accesses a specific weight from this bin."""
        try:
            return self.wgts[key]
        except KeyError:
            raise MadGraph5Error("Weight with ID '%s' is not defined for"+\
                                                            " this bin"%str(key))
                                                            
    def set_weight(self, wgt, key='central'):
        """ Accesses a specific weight from this bin."""
        
        # an assert is used here in this intensive function, so as to avoid 
        # slow-down when not in debug mode.
        assert(isinstance(wgt, float))
           
        try:
            self.wgts[key] = wgt
        except KeyError:
            raise MadGraph5Error("Weight with ID '%s' is not defined for"+\
                                                            " this bin"%str(key))                                                

    def addEvent(self, weights = 1.0):
        """ Add an event to this bin. """
        
        
        if isinstance(weights, float):
            weights = {'central': weights}
        
        for key in weights:
            if key == 'stat_error':
                continue
            try:
                self.wgts[key] += weights[key]
            except KeyError:
                raise MadGraph5Error('The event added defines the weight '+
                  '%s which was not '%key+'registered in this histogram.')
        
        self.n_entries += 1
        
        #if 'stat_error' not in weights and 'central' in w:
        #    self.wgts['stat_error'] = self.wgts['central']/math.sqrt(float(self.n_entries))
        #else:
        #    self.wgts['stat_error'] = math.sqrt( self.wgts['stat_error']**2 + 
        #                                              weights['stat_error']**2 )

    def nice_string(self, order=None, short=True):
        """ Nice representation of this Bin. 
        One can order the weight according to the argument if provided."""
        
        res     = ["Bin boundaries : %s"%str(self.boundaries)]
        if not short:
            res.append("Bin weights    :")
            if order is None:
                label_list = list(self.wgts.keys())
            else:
                label_list = order
        
            for label in label_list:
                try:
                    res.append(" -> '%s' : %4.3e"%(str(label),self.wgts[label]))
                except KeyError:
                    pass
        else:
            res.append("Central weight     : %4.3e"%self.get_weight())
        
        return '\n'.join(res)

    def alter_weights(self, func):
        """ Apply a given function to all bin weights."""
        self.wgts = func(self.wgts)

    @classmethod
    def combine(cls, binA, binB, func):
        """ Function to combine two bins. The 'func' is such that it takes
        two weight dictionaries and merge them into one."""
    
        res_bin = cls()
        if binA.boundaries != binB.boundaries:
            raise MadGraph5Error('The two bins to combine have'+\
         ' different boundaries, %s!=%s.'%(str(binA.boundaries),str(binB.boundaries)))
        res_bin.boundaries = binA.boundaries
        
        try:
            res_bin.wgts = func(binA.wgts, binB.wgts)
        except Exception as e:
            raise MadGraph5Error("When combining two bins, the provided"+\
              " function '%s' triggered the following error:\n\"%s\"\n"%\
              (func.__name__,str(e))+" when combining the following two bins:\n"+\
              binA.nice_string(short=False)+"\n and \n"+binB.nice_string(short=False))

        return res_bin

class BinList(histograms_PhysicsObjectList):
    """ A class implementing features related to a list of Bins. """

    def __init__(self, list = [], bin_range = None, 
                                     weight_labels = None):
        """ Initialize a list of Bins. It is possible to define the range
        as a list of three floats: [min_x, max_x, bin_width]"""
        
        self.weight_labels = weight_labels
        if bin_range:
            # Set the default weight_labels to something meaningful
            if not self.weight_labels:
                self.weight_labels = ['central', 'stat_error']
            if len(bin_range)!=3 or any(not isinstance(f, float) for f in bin_range):
                raise MadGraph5Error("The range argument to build a BinList"+\
                  " must be a list of exactly three floats.")
            if bin_range[2] <= 0.0:
                raise MadGraph5Error("The bin width used to build a BinList"+\
                  " must be strictly positive, not '%s'."%str(bin_range[2]))
            current = bin_range[0]
            while current < bin_range[1]:
                self.append(Bin(boundaries =
                            (current, min(current+bin_range[2],bin_range[1])),
                            wgts = dict((wgt,0.0) for wgt in self.weight_labels)))
                current += bin_range[2]
        else:
            super(BinList, self).__init__(list)

    def is_valid_element(self, obj):
        """Test whether specified object is of the right type for this list."""

        return isinstance(obj, Bin)
    
    def __setattr__(self, name, value):
        if name=='weight_labels':
            if not value is None and not isinstance(value, list):
                raise MadGraph5Error("Argument '%s' for BinList property '%s'"\
                                           %(str(value),name)+' must be a list.')
            elif not value is None:
                for label in value:
                    if all((not isinstance(label,cls)) for cls in \
                                                      [str, int, float, tuple]):
                        raise MadGraph5Error("Element '%s' of the BinList property '%s'"\
                                 %(str(value),name)+' must be a string, an '+\
                                        'integer, a float or a tuple of float.')
                    if isinstance(label, tuple):
                        if len(label)>=1:
                            if not isinstance(label[0], (float, str)):
                                    raise MadGraph5Error("Argument "+\
                            "'%s' for BinList property '%s'"%(str(value),name)+\
           ' can be a tuple, but its first element must be a float or string.')
                        for elem in label[1:]:
                            if not isinstance(elem, (float,int,str)):
                                raise MadGraph5Error("Argument "+\
                            "'%s' for BinList property '%s'"%(str(value),name)+\
           ' can be a tuple, but its elements past the first one must be either floats, integers or strings')
                                
   
        super(BinList, self).__setattr__(name, value)    
            
    def append(self, object):
        """Appends an element, but test if valid before."""
        
        super(BinList,self).append(object)    
        # Assign the weight labels to those of the first bin added
        if len(self)==1 and self.weight_labels is None:
            self.weight_labels = list(object.wgts.keys())

    def nice_string(self, short=True):
        """ Nice representation of this BinList."""
        
        res     = ["Number of bin in the list : %d"%len(self)]
        res.append("Registered weight labels  : [%s]"%(', '.join([
                                  str(label) for label in self.weight_labels])))
        if not short: 
            for i, bin in enumerate(self):
                res.append('Bin number %d :'%i)
                res.append(bin.nice_string(order=self.weight_labels, short=short))
        
        return '\n'.join(res)

class Histogram(object):
    """A mother class for all specific implementations of Histogram conventions
    """
    
    allowed_dimensions   = None
    allowed_types        = []  
    allowed_axis_modes  = ['LOG','LIN'] 

    def __init__(self, title = "NoName", n_dimensions = 2, type=None,
                 x_axis_mode = 'LIN', y_axis_mode = 'LOG', bins=None):
        """ Initializes an empty histogram, possibly specifying 
              > a title 
              > a number of dimensions
              > a bin content
        """
        
        self.title       = title
        self.dimension   = n_dimensions
        if not bins:
            self.bins    = BinList([])
        else:
            self.bins    = bins
        self.type        = type
        self.x_axis_mode = x_axis_mode
        self.y_axis_mode = y_axis_mode        
   
    def __setattr__(self, name, value):
        if name=='title':
            if not isinstance(value, str):
                raise MadGraph5Error("Argument '%s' for the histogram property "
                                     "'title' must be a string."%str(value))
        elif name=='dimension':
            if not isinstance(value, int):
                raise MadGraph5Error("Argument '%s' for histogram property "
                                    "'dimension' must be an integer."%str(value))
            if self.allowed_dimensions and value not in self.allowed_dimensions:
                raise MadGraph5Error("%i-Dimensional histograms not supported "\
                         %value+"by class '%s'. Supported dimensions are '%s'."\
                              %(self.__class__.__name__,self.allowed_dimensions))
        elif name=='bins':
            if not isinstance(value, BinList):
                raise MadGraph5Error("Argument '%s' for histogram property "
                                     "'bins' must be a BinList."%str(value))
            else:
                for bin in value:
                    if not isinstance(bin, Bin):
                        raise MadGraph5Error("Element '%s' of the "%str(bin)+\
                                  " histogram bin list specified must be a bin.")
        elif name=='type':
            if value is not None and not isinstance(value, str):
                raise MadGraph5Error("Argument '%s' for histogram property "
                                     "'type' must be a string or None."%str(value))
            if value is not None and self.allowed_types and \
                                             value not in self.allowed_types:
                raise MadGraph5Error("Argument '%s' for histogram"%str(value)+\
                             " property 'type' must be a string in %s or None."\
                                         %([str(t) for t in self.allowed_types]))
        elif name in ['x_axis_mode','y_axis_mode']:
            if not value in self.allowed_axis_modes:
                raise MadGraph5Error("Attribute '%s' of the histogram"%str(name)+\
                  " must be in [%s], ('%s' given)"%(str(self.allowed_axis_modes),
                                                                     str(value)))
                                        
        super(Histogram, self).__setattr__(name,value)
    
    def nice_string(self, short=True):
        """ Nice representation of this histogram. """
        
        res = ['<%s> histogram:'%self.__class__.__name__]
        res.append(' -> title        : "%s"'%self.title)
        res.append(' -> dimensions   : %d'%self.dimension)
        if not self.type is None:
            res.append(' -> type         : %s'%self.type)
        else:        
            res.append(' -> type         : None')
        res.append(' -> (x, y)_axis  : ( %s, %s)'%\
              (tuple([('Linear' if mode=='LIN' else 'Logarithmic') for mode in \
                                        [self.x_axis_mode, self.y_axis_mode]])))
        if short:
            res.append(' -> n_bins       : %s'%len(self.bins))
            res.append(' -> weight types : [ %s ]'%
                (', '.join([str(label) for label in self.bins.weight_labels]) \
                          if (not self.bins.weight_labels is None) else 'None'))
        
        else:
            res.append(' -> Bins content :')
            res.append(self.bins.nice_string(short))

        return '\n'.join(res) 
    
    def alter_weights(self, func):
        """ Apply a given function to all bin weights."""
        
        for bin in self.bins:
            bin.alter_weights(func)
    
    @classmethod
    def combine(cls, histoA, histoB, func):
        """ Function to combine two Histograms. The 'func' is such that it takes
        two weight dictionaries and merge them into one."""
        
        res_histogram = copy.copy(histoA)
        if histoA.title != histoB.title:
            res_histogram.title = "[%s]__%s__[%s]"%(histoA.title,func.__name__,
                                                                   histoB.title)
        else:
            res_histogram.title = histoA.title
    
        res_histogram.bins = BinList([])
        if len(histoA.bins)!=len(histoB.bins):
            raise MadGraph5Error('The two histograms to combine have a '+\
         'different number of bins, %d!=%d.'%(len(histoA.bins),len(histoB.bins)))

        if histoA.dimension!=histoB.dimension:
            raise MadGraph5Error('The two histograms to combine have a '+\
         'different dimensions, %d!=%d.'%(histoA.dimension,histoB.dimension))            
        res_histogram.dimension = histoA.dimension

        lazy_operation = LazyCombinedWeightView.supports(func)
        left_labels = list(histoA.bins.weight_labels or [])
        right_labels = list(histoB.bins.weight_labels or [])
        if lazy_operation:
            if len(left_labels) != len(set(left_labels)) or \
                         len(right_labels) != len(set(right_labels)):
                raise MadGraph5Error('Histogram weight labels must be unique '
                                     'before histograms can be combined.')
            if set(left_labels) != set(right_labels):
                missing = set(left_labels)-set(right_labels)
                extra = set(right_labels)-set(left_labels)
                raise MadGraph5Error('The two histograms to combine have '
                    'different weight labels (missing: %s; extra: %s).'%
                    (sorted(missing, key=str), sorted(extra, key=str)))
        lazy_labels = tuple(left_labels)
        lazy_label_set = frozenset(lazy_labels)
    
        for i, bin in enumerate(histoA.bins):
            other_bin = histoB.bins[i]
            if getattr(bin.wgts, '_lazy_histogram_weights', False) and \
               getattr(other_bin.wgts, '_lazy_histogram_weights', False) and \
               lazy_operation:
                if bin.boundaries != other_bin.boundaries:
                    raise MadGraph5Error('The two bins to combine have different '
                        'boundaries, %s!=%s.'%(str(bin.boundaries),
                                              str(other_bin.boundaries)))
                res_histogram.bins.append(Bin(bin.boundaries,
                    LazyCombinedWeightView(bin.wgts, other_bin.wgts, func,
                        lazy_labels, label_set=lazy_label_set)))
            else:
                res_histogram.bins.append(Bin.combine(bin, other_bin,func))
        
        # Reorder the weight labels as in the original histogram and add at the
        # end the new ones which resulted from the combination, in a sorted order
        res_histogram.bins.weight_labels = [label for label in histoA.bins.\
                weight_labels if label in res_histogram.bins.weight_labels] + \
                sorted([label for label in res_histogram.bins.weight_labels if\
                         label not in histoA.bins.weight_labels], key=str)
                
        
        return res_histogram

    # ==================================================
    #  Some handy function for Histogram combination
    # ==================================================
    @staticmethod
    def MULTIPLY(wgtsA, wgtsB):
        """ Apply the multiplication to the weights of two bins."""
        
        new_wgts = {}

        new_wgts['stat_error'] = math.sqrt(
          (wgtsA['stat_error']*wgtsB['central'])**2+
          (wgtsA['central']*wgtsB['stat_error'])**2)
        
        for label, wgt in wgtsA.items():    
            if label=='stat_error':
                continue
            new_wgts[label] = wgt*wgtsB[label]

        return new_wgts

    @staticmethod
    def DIVIDE(wgtsA, wgtsB):
        """ Apply the division to the weights of two bins."""
        
        new_wgts = {}
        if wgtsB['central'] == 0.0:
            new_wgts['stat_error'] = 0.0
        else: 
            # d(x/y) = ( (dx/y)**2 + ((x*dy)/(y**2))**2 )**0.5
            new_wgts['stat_error'] = math.sqrt(wgtsA['stat_error']**2+
            ((wgtsA['central']*wgtsB['stat_error'])/
                             wgtsB['central'])**2)/abs(wgtsB['central'])
        
        for label, wgt in wgtsA.items():
            if label=='stat_error':
                continue
            if wgtsB[label]==0.0 and wgt==0.0:
                new_wgts[label] = 0.0
            elif wgtsB[label]==0.0:
#               This situation is most often harmless and just happens in regions
#               with low statistics, so I'll bypass the warning here.
#                logger.debug('Warning:: A bin with finite weight was divided '+\
#                                                  'by a bin with zero weight.')
                new_wgts[label] = 0.0
            else:
                new_wgts[label] = wgt/wgtsB[label]

        return new_wgts        
    
    @staticmethod
    def OPERATION(wgtsA, wgtsB, wgt_operation, stat_error_operation):
        """ Apply the operation to the weights of two bins. Notice that we 
        assume here the two dict operands to have the same weight labels.
        The operation is a function that takes two floats as input."""

        new_wgts = {}
        for label, wgt in wgtsA.items():
            if label!='stat_error':
                new_wgts[label] = wgt_operation(wgt, wgtsB[label])
            else:
                new_wgts[label] = stat_error_operation(wgt, wgtsB[label])
#                if new_wgts[label]>1.0e+10:
#                    print "stat_error_operation is ",stat_error_operation.__name__
#                    print " inputs were ",wgt, wgtsB[label]
#                    print "for label", label
        
        return new_wgts


    @staticmethod
    def SINGLEHISTO_OPERATION(wgts, wgt_operation, stat_error_operation):
        """ Apply the operation to the weights of a *single* bins.
        The operation is a function that takes a single float as input."""
        
        new_wgts = {}
        for label, wgt in wgts.items():
            if label!='stat_error':
                new_wgts[label] = wgt_operation(wgt)
            else:
                new_wgts[label] = stat_error_operation(wgt)
        
        return new_wgts

    @staticmethod
    def ADD(wgtsA, wgtsB):
        """ Implements the addition using OPERATION above. """
        return Histogram.OPERATION(wgtsA, wgtsB, 
                         (lambda a,b: a+b),
                         (lambda a,b: math.sqrt(a**2+b**2)))
        
    @staticmethod
    def SUBTRACT(wgtsA, wgtsB):
        """ Implements the subtraction using OPERATION above. """
        
        return Histogram.OPERATION(wgtsA, wgtsB, 
                         (lambda a,b: a-b),
                         (lambda a,b: math.sqrt(a**2+b**2)))

    @staticmethod
    def RESCALE(factor):
        """ Implements the rescaling using SINGLEHISTO_OPERATION above. """
        
        def rescaler(wgts):
            return Histogram.SINGLEHISTO_OPERATION(wgts,(lambda a: a*factor),
                                                      (lambda a: a*abs(factor)))

        return rescaler

    @staticmethod
    def OFFSET(offset):
        """ Implements the offset using SINGLEBIN_OPERATION above. """
        def offsetter(wgts):
            return Histogram.SINGLEHISTO_OPERATION(
                                        wgts,(lambda a: a+offset),(lambda a: a))

        return offsetter
    
    def __add__(self, other):
        """ Overload the plus function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.ADD)
        elif isinstance(other, int) or isinstance(other, float):
            result = copy.deepcopy(self)
            result.alter_weights(Histogram.OFFSET(float(other)))
            return result
        else:
            return NotImplemented

    def __sub__(self, other):
        """ Overload the subtraction function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.SUBTRACT)
        elif isinstance(other, int) or isinstance(other, float):
            result = copy.deepcopy(self)
            result.alter_weights(Histogram.OFFSET(-float(other)))
            return result
        else:
            return NotImplemented
    
    def __mul__(self, other):
        """ Overload the multiplication function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.MULTIPLY)
        elif isinstance(other, int) or isinstance(other, float):
            result = copy.deepcopy(self)
            result.alter_weights(Histogram.RESCALE(float(other)))
            return result
        else:
            return NotImplemented

    def __div__(self, other):
        """ Overload the multiplication function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.DIVIDE)
        elif isinstance(other, int) or isinstance(other, float):
            result = copy.deepcopy(self)
            result.alter_weights(Histogram.RESCALE(1.0/float(other)))
            return result
        else:
            return NotImplemented

    __truediv__ = __div__


class LazyCombinedWeightView(MutableMapping):
    """Compute a derived bin (sum/product/ratio) only when a label is read."""

    _validated_histogram_weights = True
    _lazy_histogram_weights = True

    def __init__(self, left, right, operation, labels, overlay=None,
                 deleted=None, label_set=None):
        self.left = left
        self.right = right
        self.operation = operation
        self.labels = labels
        self.label_set = frozenset(labels) if label_set is None else label_set
        self.overlay = {} if overlay is None else overlay
        self.deleted = set() if deleted is None else deleted

    @staticmethod
    def supports(operation):
        return operation in [Histogram.ADD, Histogram.SUBTRACT,
                             Histogram.MULTIPLY, Histogram.DIVIDE] or \
               getattr(operation, '_histogram_lazy_operation', None) == \
                                                    'ratio_no_correlations'

    def _derived_value(self, label):
        left = self.left
        right = self.right
        if self.operation is Histogram.ADD:
            if label == 'stat_error':
                return math.sqrt(left[label]**2+right[label]**2)
            return left[label]+right[label]
        if self.operation is Histogram.SUBTRACT:
            if label == 'stat_error':
                return math.sqrt(left[label]**2+right[label]**2)
            return left[label]-right[label]
        if self.operation is Histogram.MULTIPLY:
            if label == 'stat_error':
                return math.sqrt((left[label]*right['central'])**2+
                                 (left['central']*right[label])**2)
            return left[label]*right[label]
        if self.operation is Histogram.DIVIDE:
            if label == 'stat_error':
                if right['central'] == 0.0:
                    return 0.0
                return math.sqrt(left[label]**2+
                    ((left['central']*right[label])/right['central'])**2)/\
                    abs(right['central'])
            if right[label] == 0.0:
                return 0.0
            return left[label]/right[label]
        if getattr(self.operation, '_histogram_lazy_operation', None) == \
                                                    'ratio_no_correlations':
            if right['central'] == 0.0:
                return 0.0
            return left[label]/right['central']
        raise KeyError(label)

    def __getitem__(self, label):
        if label in self.deleted:
            raise KeyError(label)
        if label in self.overlay:
            return self.overlay[label]
        if label not in self.label_set:
            raise KeyError(label)
        return float(self._derived_value(label))

    def __setitem__(self, label, value):
        self.deleted.discard(label)
        self.overlay[label] = float(value)

    def __delitem__(self, label):
        if label not in self:
            raise KeyError(label)
        self.overlay.pop(label, None)
        self.deleted.add(label)

    def __iter__(self):
        for label in self.labels:
            if label not in self.deleted:
                yield label
        for label in self.overlay:
            if label not in self.deleted and label not in self.label_set:
                yield label

    def __len__(self):
        return sum(1 for _ in self)

    def __deepcopy__(self, memo):
        result = self.__class__(copy.deepcopy(self.left, memo),
            copy.deepcopy(self.right, memo), self.operation, self.labels,
            copy.deepcopy(self.overlay, memo), copy.deepcopy(self.deleted, memo),
            label_set=self.label_set)
        memo[id(self)] = result
        return result


class LazyRebinnedWeightView(MutableMapping):
    """Compute a rebinned weight from source bins without a per-weight dict."""

    _validated_histogram_weights = True
    _lazy_histogram_weights = True

    def __init__(self, sources, labels, overlay=None, deleted=None,
                 label_set=None):
        self.sources = sources
        self.labels = labels
        self.label_set = frozenset(labels) if label_set is None else label_set
        self.overlay = {} if overlay is None else overlay
        self.deleted = set() if deleted is None else deleted

    def __getitem__(self, label):
        if label in self.deleted:
            raise KeyError(label)
        if label in self.overlay:
            return self.overlay[label]
        if label not in self.label_set:
            raise KeyError(label)
        if label == 'stat_error':
            return math.sqrt(sum(source[label]**2 for source in self.sources))
        return float(sum(source[label] for source in self.sources))

    def __setitem__(self, label, value):
        self.deleted.discard(label)
        self.overlay[label] = float(value)

    def __delitem__(self, label):
        if label not in self:
            raise KeyError(label)
        self.overlay.pop(label, None)
        self.deleted.add(label)

    def __iter__(self):
        for label in self.labels:
            if label not in self.deleted:
                yield label
        for label in self.overlay:
            if label not in self.deleted and label not in self.label_set:
                yield label

    def __len__(self):
        return sum(1 for _ in self)

    def __deepcopy__(self, memo):
        result = self.__class__(copy.deepcopy(self.sources, memo), self.labels,
            copy.deepcopy(self.overlay, memo), copy.deepcopy(self.deleted, memo),
            label_set=self.label_set)
        memo[id(self)] = result
        return result

class HwU(Histogram):
    """A concrete implementation of an histogram plots using the HwU format for
    reading/writing histogram content."""
    
    allowed_dimensions         = [2]
    allowed_types              = []   

    output_formats_implemented = ['HwU','gnuplot','matplotlib','html']
    # Lists the mandatory named weights that must be specified for each bin and
    # what corresponding label we assign them to in the Bin weight dictionary,
    # (if any).
    mandatory_weights  = {'xmin':'boundary_xmin', 'xmax':'boundary_xmax', 
                                   'central value':'central', 'dy':'stat_error'}
    
    # ========================
    #  Weight name parser RE's
    # ========================
    # This marks the start of the line that defines the name of the weights
    weight_header_start_re = re.compile('^##.*')
    # This is the format of a weight name specifier. It is much more complicated
    # than necessary because the HwU standard allows for spaces from within
    # the name of a weight
    weight_header_re = re.compile(
                       r'&\s*(?P<wgt_name>(\S|(\s(?!\s*(&|$))))+)(\s(?!(&|$)))*')
    
    # ================================
    #  Histo weight specification RE's
    # ================================
    # The start of a plot
    histo_start_re = re.compile(r'^\s*<histogram>\s*(?P<n_bins>\d+)\s*"\s*'+
                                   r'(?P<histo_name>(\S|(\s(?!\s*")))+)\s*"\s*$')
    # A given weight specifier
    a_float_re = r'[\+|-]?\d+(\.\d*)?([EeDd][\+|-]?\d+)?'
    histo_bin_weight_re = re.compile(r'(?P<weight>%s|NaN)'%a_float_re,re.IGNORECASE)
    a_int_re = r'[\+|-]?\d+'
    
    # The end of a plot
    histo_end_re = re.compile(r'^\s*<\\histogram>\s*$')
    # A scale type of weight
    weight_label_scale = re.compile(r'^\s*mur\s*=\s*(?P<mur_fact>%s)'%a_float_re+\
                   r'\s*muf\s*=\s*(?P<muf_fact>%s)\s*$'%a_float_re,re.IGNORECASE)
    weight_label_PDF = re.compile(r'^\s*PDF\s*=\s*(?P<PDF_set>\d+)\s*$')
    weight_label_PDF_XML = re.compile(r'^\s*pdfset\s*=\s*(?P<PDF_set>\d+)\s*$')
    weight_label_TMS = re.compile(r'^\s*TMS\s*=\s*(?P<Merging_scale>%s)\s*$'%a_float_re)
    weight_label_alpsfact = re.compile(r'^\s*alpsfact\s*=\s*(?P<alpsfact>%s)\s*$'%a_float_re,
                                                                  re.IGNORECASE)

    weight_label_scale_adv = re.compile(r'^\s*dyn\s*=\s*(?P<dyn_choice>%s)'%a_int_re+\
                                        r'\s*mur\s*=\s*(?P<mur_fact>%s)'%a_float_re+\
                                        r'\s*muf\s*=\s*(?P<muf_fact>%s)\s*$'%a_float_re,re.IGNORECASE)
    weight_label_PDF_adv = re.compile(r'^\s*PDF\s*=\s*(?P<PDF_set>\d+)\s+(?P<PDF_set_cen>\S+)\s*$')
    
    
    class ParseError(MadGraph5Error):
        """a class for histogram data parsing errors"""
    
    @classmethod
    def get_HwU_wgt_label_type(cls, wgt_label):
        """ From the format of the weight label given in argument, it returns
        a string identifying the type of standard weight it is."""

        if isinstance(wgt_label,str):
            return 'UNKNOWN_TYPE'
        if isinstance(wgt_label,tuple):
            if len(wgt_label)==0:
                return 'UNKNOWN_TYPE'
            if isinstance(wgt_label[0],float):
                return 'murmuf_scales'
            if isinstance(wgt_label[0],str):
                return wgt_label[0]
        if isinstance(wgt_label,float):
            return 'merging_scale'
        if isinstance(wgt_label,int):
            return 'pdfset'
        # No clue otherwise
        return 'UNKNOWN_TYPE'
    
    
    def __init__(self, file_path=None, weight_header=None,
                raw_labels=False, consider_reweights='ALL', selected_central_weight=None, **opts):
        """ Read one plot from a file_path or a stream. Notice that this
        constructor only reads one, and the first one, of the plots specified.
        If file_path was a path in argument, it would then close the opened stream.
        If file_path was a stream in argument, it would leave it open.
        The option weight_header specifies an ordered list of weight names 
        to appear in the file specified.
        The option 'raw_labels' specifies that one wants to import the
        histogram data with no treatment of the weight labels at all
        (this is used for the matplotlib output)."""
        
        super(HwU, self).__init__(**opts)

        self.dimension = 2
        
        if file_path is None:
            return
        elif isinstance(file_path, str):
            stream = open(file_path,'r')
        elif isinstance(file_path, io.IOBase):
            stream = file_path
        elif isinstance(file_path, file):
            stream = file_path
        else:
            raise MadGraph5Error("Argument file_path '%s' for HwU init"\
            %str(file_path)+"ialization must be either a file path or a stream.")

        try:
            # Attempt to find the weight headers if not specified.
            if not weight_header:
                weight_header = HwU.parse_weight_header(
                                                  stream, raw_labels=raw_labels)

            if not self.parse_one_histo_from_stream(stream, weight_header,
                      consider_reweights=consider_reweights,
                      selected_central_weight=selected_central_weight,
                      raw_labels=raw_labels):
                # Mark unsuccessful initialization with an empty bin source.
                super(Histogram,self).__setattr__('bins',None)
        finally:
            if isinstance(file_path, str):
                stream.close()

    def addEvent(self, x_value, weights = 1.0):
        """ Add an event to the current plot. """
        
        for bin in self.bins:
            if bin.boundaries[0] <= x_value < bin.boundaries[1]:
                bin.addEvent(weights = weights)
    
    def get(self, name):
        
        if name == 'bins':
            return [b.boundaries[0] for b in self.bins]
        else:
            return [b.wgts[name] for b in self.bins]
    
    def add_line(self, names):
        """add a column to the HwU. name can be a list"""
        
        if isinstance(names, str):
            names = [names]
        else:
            names = list(names)
        #check if all the entry are new 
        for name in names[:]:
            if name in self.bins[0].wgts:
                logger.warning("name: %s is already defines in HwU.")
                names.remove(name)
        #
        for name in names:
            self.bins.weight_labels.append(name)
            for bin in self.bins:
                bin.wgts[name] = 0
            
    def get_uncertainty_band(self, selector, mode=0):
        """return two list of entry one with the minimum and one with the maximum value.
           selector can be:
               - a regular expression on the label name
               - a function returning T/F (applying on the label name)
               - a list of labels
               - a keyword
        """     
        
        # find the set of weights to consider
        if isinstance(selector, str):
            if selector == 'QCUT':
                selector = r'^Weight_MERGING=[\d]*[.]?\d*$'
            elif selector == 'SCALE':
                selector = r'(MUF=\d*[.]?\d*_MUR=([^1]\d*|1\d+)_PDF=\d*)[.]?\d*|(MUF=([^1]\d*|1\d+)[.]?\d*_MUR=\d*[.]?\d*_PDF=\d*)'
            elif selector == 'ALPSFACT':
                selector = r'ALPSFACT'
            elif selector == 'PDF':
                selector = r'(?:MUF=1_MUR=1_PDF=|MU(?:F|R)="1.0" MU(?:R|F)="1.0" PDF=")(\d*)'
                if not mode:
#                    pdfs=[]
##                    for n in self.bins[0].wgts:
#                        misc.sprint( n)
#                        if re.search(selector,n, re.IGNORECASE):
#                            pdfs.append(int(re.findall(selector, n)[0]))
                    pdfs = [int(re.findall(selector, n)[0]) for n in self.bins[0].wgts if re.search(selector,n, re.IGNORECASE)]
                    min_pdf, max_pdf = min(pdfs), max(pdfs) 
                    if max_pdf - min_pdf > 100:
                        mode == 'min/max'
                    elif  max_pdf <= 90000:
                        mode = 'hessian'
                    else:
                        mode = 'gaussian'
            selections = [n for n in self.bins[0].wgts if re.search(selector,n, re.IGNORECASE)]
        elif hasattr(selector, '__call__'):
            selections = [n for n in self.bins[0].wgts if selector(n)]
        elif isinstance(selector, (list, tuple)):
            selections = selector
        
        # find the way to find the minimal/maximal curve
        if not mode:
            mode = 'min/max'
        
        # build the collection of values
        values = []
        for s in selections:
            values.append(self.get(s))
        
        #sanity check
        if not len(values):
            return [0] * len(self.bins), [0]* len(self.bins)
        elif len(values) ==1:
            return values[0], values[0]
        
        
        # Start the real work
        if mode == 'min/max':
            min_value, max_value = [], []
            for i in range(len(values[0])):
                data = [values[s][i] for s in range(len(values))]
                min_value.append(min(data))
                max_value.append(max(data))
        elif mode == 'gaussian':
            # use Gaussian method (NNPDF)
            min_value, max_value = [], []
            for i in range(len(values[0])):
                pdf_stdev = 0.0
                data = [values[s][i] for s in range(len(values))]
                sdata = sum(data)/len(data)
                sdata2 = sum(x**2 for x in data)/len(data)
                pdf_stdev = math.sqrt(max(sdata2 -sdata**2,0.0))
                min_value.append(sdata - pdf_stdev)
                max_value.append(sdata + pdf_stdev)                 

        elif mode == 'hessian':
            # For old PDF this is based on the set ordering ->
            # need to order the PDF sets.  ``selector`` can be an explicit
            # list, in which case using it as a regular expression (as the old
            # code did) raises a TypeError.
            def get_pdf_number(label):
                if isinstance(selector, str):
                    match = re.findall(selector, label, re.IGNORECASE)
                    if match:
                        value = match[0]
                        if isinstance(value, tuple):
                            value = next((item for item in value if item), '')
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            pass
                match = re.search(r'PDF\s*=\s*(\d+)', str(label), re.IGNORECASE)
                if match:
                    return int(match.group(1))
                numbers = re.findall(r'\d+', str(label))
                if numbers:
                    return int(numbers[-1])
                raise MadGraph5Error("Could not determine the PDF member number "
                                     "from weight label '%s'."%str(label))

            pdfs = [(get_pdf_number(name), name) for name in selections]
            pdfs.sort()
            
            # check if the central was put or not in this sets:
            if len(pdfs) % 2:
                # adding the central automatically
                pdf1 = pdfs[0][0]
                central = pdf1 -1
                name = str(pdfs[0][1]).replace(str(pdf1), str(central))
                if name in self.bins[0].wgts:
                    central = self.get(name)
                else:
                    central = self.get('central')
            else:
                central = self.get(pdfs.pop(0)[1])
            
            #rebuilt the collection of values but this time ordered correctly
            values = []
            for _, name in pdfs:
                values.append(self.get(name))

            if len(values) < 2:
                return central, central
            if len(values) % 2:
                logger.warning("Ignoring the unpaired final Hessian PDF member '%s'.",
                               pdfs[-1][1])
                values.pop()
                pdfs.pop()
                
            #Do the computation
            min_value, max_value = [], []
            for i in range(len(values[0])):
                pdf_up = 0
                pdf_down = 0
                cntrl_val = central[i]
                for s in range(int((len(pdfs))/2)):
                    pdf_up   += max(0.0,values[2*s][i]   - cntrl_val,
                                        values[2*s+1][i] - cntrl_val)**2
                    pdf_down   += max(0.0,cntrl_val - values[2*s][i],
                                          cntrl_val - values[2*s+1][i])**2  
                                          
                min_value.append(cntrl_val - math.sqrt(pdf_down))
                max_value.append(cntrl_val + math.sqrt(pdf_up))                  
                                          

                
        
        return min_value, max_value            
    
    def get_formatted_header(self, weight_labels=None):
        """ Return a HwU formatted header for the weight label definition."""

        res = '##& xmin & xmax & '
        weight_labels = self.bins.weight_labels if weight_labels is None \
                                                        else weight_labels
        
        if 'central' in weight_labels:
            res += 'central value & dy & '
        
        others = []
        for label in weight_labels:
            if label in ['central', 'stat_error']:
                continue
            label_type = HwU.get_HwU_wgt_label_type(label)
            if label_type == 'UNKNOWN_TYPE':
                others.append(label)
            elif label_type == 'scale':
                others.append('muR=%6.3f muF=%6.3f'%(label[1],label[2]))
            elif label_type == 'scale_adv':
                others.append('dyn=%i muR=%6.3f muF=%6.3f'%(label[1],label[2],label[3]))
            elif label_type == 'merging_scale':
                others.append('TMS=%4.2f'%label[1])
            elif label_type == 'pdf':
                others.append('PDF=%i'%(label[1]))
            elif label_type == 'pdf_adv':
                others.append('PDF=%i %s'%(label[1],label[2]))
            elif label_type == 'alpsfact':
                others.append('alpsfact=%d'%label[1])

        return res+' & '.join(others)

    def iter_HwU_source(self, print_header=True, weight_labels=None):
        """Yield HwU source line by line without retaining a text copy."""

        registered_labels = list(self.bins.weight_labels)
        weight_labels = registered_labels if weight_labels is None \
                                         else list(weight_labels)
        if len(registered_labels) != len(set(registered_labels)) or \
                         len(weight_labels) != len(set(weight_labels)):
            raise MadGraph5Error("Histogram '%s' has duplicate weight columns."%
                                 self.title)
        if set(weight_labels) != set(registered_labels):
            raise MadGraph5Error("The requested HwU column schema is not "
                                 "compatible with histogram '%s'."%self.title)
        if ('central' in registered_labels) != \
                                      ('stat_error' in registered_labels):
            raise MadGraph5Error("Histogram '%s' must define central and "
                                 "statistical-error columns together."%self.title)
        if print_header:
            yield self.get_formatted_header(weight_labels=weight_labels)
            yield ''
        yield '<histogram> %s "%s"'%(len(self.bins),
                                     self.get_HwU_histogram_name(format='HwU'))
        expected_labels = set(registered_labels)
        for bin_index, hist_bin in enumerate(self.bins):
            if set(hist_bin.wgts) != expected_labels:
                raise MadGraph5Error("Bin %d of histogram '%s' does not match "
                                     "its registered weight columns."%
                                     (bin_index+1, self.title))
            boundaries = list(hist_bin.boundaries)
            if len(boundaries) != 2 or not all(
                            math.isfinite(value) for value in boundaries):
                raise MadGraph5Error("Bin %d of histogram '%s' has invalid "
                                     "boundaries."%(bin_index+1, self.title))
            if boundaries[0] > boundaries[1]:
                raise MadGraph5Error("Bin %d of histogram '%s' has decreasing "
                                     "boundaries."%(bin_index+1, self.title))
            values = [hist_bin.wgts[label] for label in weight_labels]
            if not all(math.isfinite(value) for value in values):
                raise MadGraph5Error("Bin %d of histogram '%s' contains a "
                                     "non-finite weight."%
                                     (bin_index+1, self.title))
            if 'central' in hist_bin.wgts:
                line = ' '.join('%+16.7e'%wgt for wgt in boundaries+
                    [hist_bin.wgts['central'], hist_bin.wgts['stat_error']])
            else:
                line = ' '.join('%+16.7e'%wgt for wgt in boundaries)
            line += ' '.join('%+16.7e'%hist_bin.wgts[key] for key in
                weight_labels if key not in ['central','stat_error'])
            yield line
        yield r'<\histogram>'

    def get_HwU_source(self, print_header=True, weight_labels=None):
        """Return the HwU source as a list (legacy convenience API)."""

        return list(self.iter_HwU_source(print_header=print_header,
                                         weight_labels=weight_labels))
    
    def output(self, path=None, format='HwU', print_header=True):
        """ Ouput this histogram to a file, stream or string if path is kept to
        None. The supported format are for now. Chose whether to print the header
        or not."""
        
        if not format in HwU.output_formats_implemented:
            raise MadGraph5Error("The specified output format '%s'"%format+\
                             " is not yet supported. Supported formats are %s."\
                                                 %HwU.output_formats_implemented)

        if format in ['gnuplot', 'matplotlib', 'html']:
            if not isinstance(path, str):
                raise MadGraph5Error("A path is required for %s output."%format)
            HwUList([copy.deepcopy(self)]).output(path, format=format)
            return True

        if format == 'HwU':
            str_output_list = self.get_HwU_source(print_header=print_header)

        if path is None:
            return '\n'.join(str_output_list)
        elif isinstance(path, str):
            target = _AtomicTextOutput(path)
            try:
                target.stream.write('\n'.join(str_output_list))
                target.commit()
            except BaseException:
                target.abort()
                raise
        elif isinstance(path, file):
            path.write('\n'.join(str_output_list))
        
        # Successful writeout
        return True

    def test_plot_compability(self, other, consider_type=True,
                                                consider_unknown_weight_labels=True):
        """ Test whether the defining attributes of self are identical to histo,
        typically to make sure that they are the same plots but from different
        runs, and they can be summed safely. We however don't want to 
        overload the __eq__ because it is still a more superficial check."""
       
        this_known_weight_labels = [label for label in self.bins.weight_labels if
                                   HwU.get_HwU_wgt_label_type(label)!='UNKNOWN_TYPE']
        other_known_weight_labels = [label for label in other.bins.weight_labels if
                                   HwU.get_HwU_wgt_label_type(label)!='UNKNOWN_TYPE']
        this_unknown_weight_labels = [label for label in self.bins.weight_labels if
                                   HwU.get_HwU_wgt_label_type(label)=='UNKNOWN_TYPE']
        other_unknown_weight_labels = [label for label in other.bins.weight_labels if
                                   HwU.get_HwU_wgt_label_type(label)=='UNKNOWN_TYPE']

        if self.title != other.title or \
           set(this_known_weight_labels) != set(other_known_weight_labels) or \
           (set(this_unknown_weight_labels) != set(other_unknown_weight_labels) and\
                                                 consider_unknown_weight_labels) or \
           (self.type != other.type and consider_type) or \
           self.x_axis_mode != other.x_axis_mode or \
           self.y_axis_mode != other.y_axis_mode or \
           len(self.bins) != len(other.bins) or \
           any(b1.boundaries!=b2.boundaries for (b1,b2) in \
                                                     zip(self.bins,other.bins)):
            return False
        
        return True
           
            
    
    @classmethod
    def parse_weight_header(cls, stream, raw_labels=False):
        """ Read a given stream until it finds a header specifying the weights
        and then returns them."""
        
        for line in stream:
            if cls.weight_header_start_re.match(line):
                header = [h.group('wgt_name') for h in 
                                            cls.weight_header_re.finditer(line)]
                if any((name not in header) for name in cls.mandatory_weights):
                    raise HwU.ParseError("The mandatory weight names %s were"\
                     %str(list(cls.mandatory_weights.keys()))+" are not all present"+\
                     " in the following HwU header definition:\n   %s"%line)
                
                # Apply replacement rules specified in mandatory_weights
                if raw_labels:
                    # If using raw labels, then just change the name of the 
                    # labels corresponding to the bin edges
                    header = [ (h if h not in ['xmin','xmax'] else 
                                     cls.mandatory_weights[h]) for h in header ]
                    # And return it with no further modification
                    return header
                else:
                    header = [ (h if h not in cls.mandatory_weights else 
                                     cls.mandatory_weights[h]) for h in header ]
                
                # We use a special rule for the weight labeled as a 
                # muR=2.0 muF=1.0 scale specification, in which case we store
                # it as a tuple
                for i, h in enumerate(header):
                    scale_wgt   = HwU.weight_label_scale.match(h)
                    PDF_wgt     = HwU.weight_label_PDF.match(h)
                    Merging_wgt = HwU.weight_label_TMS.match(h)
                    alpsfact_wgt = HwU.weight_label_alpsfact.match(h)
                    scale_wgt_adv = HwU.weight_label_scale_adv.match(h)
                    PDF_wgt_adv   = HwU.weight_label_PDF_adv.match(h)
                    if scale_wgt_adv:
                        header[i] = ('scale_adv',
                                     int(scale_wgt_adv.group('dyn_choice')),
                                     float(scale_wgt_adv.group('mur_fact')),
                                     float(scale_wgt_adv.group('muf_fact')))
                    elif scale_wgt:
                        header[i] = ('scale',
                                     float(scale_wgt.group('mur_fact')),
                                     float(scale_wgt.group('muf_fact')))
                    elif PDF_wgt_adv:
                        header[i] = ('pdf_adv',
                                     int(PDF_wgt_adv.group('PDF_set')),
                                     PDF_wgt_adv.group('PDF_set_cen'))
                    elif PDF_wgt:
                        header[i] = ('pdf',int(PDF_wgt.group('PDF_set')))
                    elif Merging_wgt:
                        header[i] = ('merging_scale',float(Merging_wgt.group('Merging_scale')))
                    elif alpsfact_wgt:
                        header[i] = ('alpsfact',float(alpsfact_wgt.group('alpsfact')))                      

                return header
            
        raise HwU.ParseError("The weight headers could not be found.")
    
    
    def process_histogram_name(self, histogram_name):
        """ Parse the histogram name for tags which would set its various
        attributes."""
        
        for i, tag in enumerate(histogram_name.split('|')):
            if i==0:
                self.title = tag.strip()
            else:
                stag = tag.split('@')
                if len(stag)==1 and stag[0].startswith('#'): continue
                if len(stag)!=2:
                    raise MadGraph5Error('Specifier in title must have the'+\
            " syntax @<attribute_name>:<attribute_value>, not '%s'."%tag.strip())
                # Now list all supported modifiers here
                stag = [t.strip().upper() for t in stag]
                if stag[0] in ['T','TYPE']:
                    self.type = stag[1]
                elif stag[0] in ['X_AXIS', 'X']:
                    self.x_axis_mode = stag[1]                    
                elif stag[0] in ['Y_AXIS', 'Y']:
                    self.y_axis_mode = stag[1] 
                elif stag[0] in ['JETSAMPLE', 'JS']:
                    self.jetsample = int(stag[1])
                else:
                    raise MadGraph5Error("Specifier '%s' not recognized."%stag[0])                    
        
    def get_HwU_histogram_name(self, format='human'):
        """ Returns the histogram name in the HwU syntax or human readable."""
        
        type_map = {'NLO':'NLO', 'LO':'LO', 'AUX':'auxiliary histogram'}
        
        if format=='human':
            res = self.title
            if not self.type is None:
                try:
                    res += ', %s'%type_map[self.type]
                except KeyError:
                    res += ', %s'%str('NLO' if self.type.split()[0]=='NLO' else
                                                                      self.type)
            if hasattr(self,'jetsample'):
                if self.jetsample==-1:
                    res += ', all jet samples'
                else:
                    res += ', Jet sample %d'%self.jetsample
                    
            return res

        elif format=='human-no_type':
            res = self.title
            return res

        elif format=='HwU':
            res = [self.title]
            res.append('|X_AXIS@%s'%self.x_axis_mode)
            res.append('|Y_AXIS@%s'%self.y_axis_mode)
            if hasattr(self,'jetsample'):
                res.append('|JETSAMPLE@%d'%self.jetsample)            
            if self.type:
                res.append('|TYPE@%s'%self.type)
            return ' '.join(res)
        
    def parse_one_histo_from_stream(self, stream, all_weight_header,
                consider_reweights='ALL', raw_labels=False, selected_central_weight=None):
        """ Reads *one* histogram from a stream, with the mandatory specification
        of the ordered list of weight names. Return True or False depending
        on whether the starting definition of a new plot could be found in this
        stream."""
        n_bins = 0
        
        if consider_reweights=='ALL' or raw_labels:
            weight_header = all_weight_header 
        else:
            new_weight_header = []
            # Filter the weights to consider based on the user selection
            for wgt_label in all_weight_header:
                if wgt_label in ['central','stat_error','boundary_xmin','boundary_xmax'] or\
                    HwU.get_HwU_wgt_label_type(wgt_label) in consider_reweights:
                        new_weight_header.append(wgt_label)
            weight_header = new_weight_header                   
        
        # Find the starting point of the stream
        for line in stream:
            start = HwU.histo_start_re.match(line)
            if not start is None:
                self.process_histogram_name(start.group('histo_name'))
                # We do not want to include auxiliary diagrams which would be
                # recreated anyway.
                if self.type == 'AUX':
                    continue
                n_bins           = int(start.group('n_bins'))
                # Make sure to exclude the boundaries from the weight
                # specification
                self.bins        = BinList(weight_labels = [ wgt_label for
                                wgt_label in weight_header if wgt_label not in
                                             ['boundary_xmin','boundary_xmax']])
                break
        
        # Now look for the bin weights definition
        for line_bin in stream:
            bin_weights = {}
            boundaries = [0.0,0.0]
            for j, weight in \
                      enumerate(HwU.histo_bin_weight_re.finditer(line_bin)):
                #if (j == len(weight_header)):
                #    continue
                if selected_central_weight == all_weight_header[j]:
                    bin_weights['central'] = float(weight.group('weight'))
                if all_weight_header[j] == 'boundary_xmin':
                    boundaries[0] = float(weight.group('weight'))
                elif all_weight_header[j] == 'boundary_xmax':
                    boundaries[1] = float(weight.group('weight'))
                elif all_weight_header[j] == 'central' and not selected_central_weight is None:
                    continue                      
                elif all_weight_header[j] in weight_header:
                    bin_weights[all_weight_header[j]] = \
                                           float(weight.group('weight'))

            # For the HwU format, we know that exactly two 'weights'
            # specified in the weight_header are in fact the boundary 
            # coordinate, so we must subtract two.    
            if len(bin_weights)<(len(weight_header)-2):
                raise HwU.ParseError(" There are only %i weights"\
                    %len(bin_weights)+" specified and %i were expected."%\
                                                      (len(weight_header)-2))
            self.bins.append(Bin(tuple(boundaries), bin_weights))
            if len(self.bins)==n_bins:
                break

        if len(self.bins)!=n_bins:
            raise HwU.ParseError("%i bin specification "%len(self.bins)+\
                               "were found and %i were expected."%n_bins)

        # Now jump to the next <\histo> tag.
        for line_end in stream:
            if HwU.histo_end_re.match(line_end):
                # Finally, remove all the auxiliary weights, but only if not
                # asking for raw labels
                if not raw_labels:
                    self.trim_auxiliary_weights()
                # End of successful parsing this histogram, so return True.
                return True

        # Could not find a plot definition starter in this stream, return False
        return False
    
    def trim_auxiliary_weights(self):
        """ Remove all weights which are auxiliary (whose name end with '@aux')
        so that they are not included (they will be regenerated anyway)."""
        
        for i, wgt_label in enumerate(self.bins.weight_labels):
            if isinstance(wgt_label, str) and wgt_label.endswith('@aux'):
                for bin in self.bins:
                    try:
                        del bin.wgts[wgt_label]
                    except KeyError:
                        pass
        self.bins.weight_labels = [wgt_label for wgt_label in 
            self.bins.weight_labels if (not isinstance(wgt_label, str) 
           or (isinstance(wgt_label, str) and not wgt_label.endswith('@aux')) )]

    def set_uncertainty(self, type='all_scale',lhapdfconfig='lhapdf-config'):
        """ Adds a weight to the bins which is the envelope of the scale
        uncertainty, for the scale specified which can be either 'mur', 'muf',
        'all_scale' or 'PDF'."""

        type = type.upper()

        if type=='MUR':
            new_wgt_label  = 'delta_mur'
            scale_position = 1
        elif type=='MUF':
            new_wgt_label = 'delta_muf'
            scale_position = 2
        elif type=='ALL_SCALE':
            new_wgt_label = 'delta_mu'
            scale_position = -1
        elif type=='PDF':
            new_wgt_label = 'delta_pdf'
            scale_position = -2            
        elif type=='MERGING':
            new_wgt_label = 'delta_merging'
        elif type=='ALPSFACT':
            new_wgt_label = 'delta_alpsfact'
        else:
            raise MadGraph5Error(' The function set_uncertainty can'+\
              " only handle the scales 'mur', 'muf', 'all_scale', 'pdf',"+\
              "'merging' or 'alpsfact'.")       
        
        wgts_to_consider=[]
        label_to_consider=[]
        if type == 'MERGING':
            # It is a list of list because we consider only the possibility of
            # a single "central value" in this case, so the outtermost list is
            # always of length 1.
            wgts_to_consider.append([ label for label in self.bins.weight_labels if \
                            HwU.get_HwU_wgt_label_type(label)=='merging_scale' ])
            label_to_consider.append('none')

        elif type == 'ALPSFACT':
            # It is a list of list because we consider only the possibility of
            # a single "central value" in this case, so the outtermost list is
            # always of length 1.
            wgts_to_consider.append([ label for label in self.bins.weight_labels if \
                            HwU.get_HwU_wgt_label_type(label)=='alpsfact' ])
            label_to_consider.append('none')            
        elif scale_position > -2:
            ##########: advanced scale
            dyn_scales=[label[1] for label in self.bins.weight_labels if \
                        HwU.get_HwU_wgt_label_type(label)=='scale_adv']
            # remove doubles in list but keep the order!
            dyn_scales=[scale for n,scale in enumerate(dyn_scales) if scale not in dyn_scales[:n]]
            for dyn_scale in dyn_scales:
                wgts=[label for label in self.bins.weight_labels if \
                      HwU.get_HwU_wgt_label_type(label)=='scale_adv' and label[1]==dyn_scale]
                if wgts:
                    wgts_to_consider.append(wgts)
                    label_to_consider.append(dyn_scale)
            ##########: normal scale
            wgts=[label for label in self.bins.weight_labels if \
                                 HwU.get_HwU_wgt_label_type(label)=='scale']
            ## this is for the 7-point variations (excludes mur/muf = 4, 1/4)
            #wgts_to_consider = [ label for label in self.bins.weight_labels if \
            #                     isinstance(label,tuple) and label[0]=='scale' and \
            #                                  not (0.5 in label and 2.0 in label)]
            if wgts:
                wgts_to_consider.append(wgts)
                label_to_consider.append('none')
            ##########: remove renormalisation OR factorisation scale dependence...

            if scale_position > -1:
                wgts_to_consider = [
                    [label for label in wgts
                                      if label[-scale_position] == 1.0]
                    for wgts in wgts_to_consider]
        elif scale_position == -2:
            ##########: advanced PDF
            pdf_sets=[label[2] for label in self.bins.weight_labels if \
                        HwU.get_HwU_wgt_label_type(label)=='pdf_adv']
            # remove doubles in list but keep the order!
            pdf_sets=[ii for n,ii in enumerate(pdf_sets) if ii not in pdf_sets[:n]]
            for pdf_set in pdf_sets:
                wgts=[label for label in self.bins.weight_labels if \
                      HwU.get_HwU_wgt_label_type(label)=='pdf_adv' and label[2]==pdf_set]
                if wgts:
                    wgts_to_consider.append(wgts)
                    label_to_consider.append(pdf_set)
            ##########: normal PDF
            wgts = [ label for label in self.bins.weight_labels if \
                     HwU.get_HwU_wgt_label_type(label)=='pdf']  
            if wgts:
                wgts_to_consider.append(wgts)
                label_to_consider.append('none')

        if len(wgts_to_consider)==0 or all(len(wgts)==0 for wgts in wgts_to_consider):
            # No envelope can be constructed, it is not worth adding the weights
            return (None,[None])

        # find and import python version of lhapdf if doing PDF uncertainties
        if type=='PDF':
            use_lhapdf=False
            try:
                lhapdf_libdir=subprocess.check_output(
                    [lhapdfconfig,'--libdir'], stderr=subprocess.STDOUT
                ).decode(errors='ignore').strip()
            except:
                use_lhapdf=False
            else:
                try:
                    candidates=[dirname for dirname in os.listdir(lhapdf_libdir) \
                                if os.path.isdir(os.path.join(lhapdf_libdir,dirname))]
                except OSError:
                    candidates=[]
                for candidate in candidates:
                    if os.path.isfile(os.path.join(lhapdf_libdir,candidate,'site-packages','lhapdf.so')):
                        sys.path.insert(0,os.path.join(lhapdf_libdir,candidate,'site-packages'))
                        try:
                            import lhapdf
                            use_lhapdf=True
                            break
                        except ImportError:
                            sys.path.pop(0)
                            continue
                   
                if not use_lhapdf:
                    try:
                        candidates=[dirname for dirname in os.listdir(lhapdf_libdir+'64') \
                                    if os.path.isdir(os.path.join(lhapdf_libdir+'64',dirname))]
                    except OSError:
                        candidates=[]
                    for candidate in candidates:
                        if os.path.isfile(os.path.join(lhapdf_libdir+'64',candidate,'site-packages','lhapdf.so')):
                            sys.path.insert(0,os.path.join(lhapdf_libdir+'64',candidate,'site-packages'))
                            try:
                                import lhapdf
                                use_lhapdf=True
                                break
                            except ImportError:
                                sys.path.pop(0)
                                continue
            
            if not use_lhapdf:
                try:
                    import lhapdf
                    use_lhapdf=True
                except ImportError:
                    logger.warning("Failed to access python version of LHAPDF: "\
                                   "cannot compute PDF uncertainty from the "\
                                   "weights in the histograms. The weights in the HwU data files " \
                                   "still cover all PDF set members, "\
                                   "but the automatic computation of the uncertainties from "\
                                   "those weights might not be correct. \n "\
                                   "If the python interface to LHAPDF is available on your system, try "\
                                   "adding its location to the PYTHONPATH environment variable and the"\
                                   "LHAPDF library location to LD_LIBRARY_PATH (linux) or DYLD_LIBRARY_PATH (mac os x).")
            
            if type=='PDF' and use_lhapdf:
                lhapdf.setVerbosity(0)

        # Place the new weight label last before the first tuple
        position=[]
        labels=[]
        for i,label in enumerate(label_to_consider):
            wgts=wgts_to_consider[i]
            if label != 'none':
                new_wgt_labels=['%s_cen %s @aux' % (new_wgt_label,label),
                                '%s_min %s @aux' % (new_wgt_label,label),
                                '%s_max %s @aux' % (new_wgt_label,label)]
            else:
                new_wgt_labels=['%s_cen @aux' % new_wgt_label,
                                '%s_min @aux' % new_wgt_label,
                                '%s_max @aux' % new_wgt_label]
            try:
                pos=[(not isinstance(lab, str)) for lab in \
                            self.bins.weight_labels].index(True)
                position.append(pos)
                labels.append(label)
                self.bins.weight_labels = self.bins.weight_labels[:pos]+\
                                          new_wgt_labels + self.bins.weight_labels[pos:]
            except ValueError:
                pos=len(self.bins.weight_labels)
                position.append(pos)
                labels.append(label)
                self.bins.weight_labels.extend(new_wgt_labels)

            if type=='PDF' and use_lhapdf and label != 'none':
                p=lhapdf.getPDFSet(label)

            # Now add the corresponding weight to all Bins
            for bin in self.bins:
                if type!='PDF': 
                    bin.wgts[new_wgt_labels[0]] = bin.wgts[wgts[0]]
                    bin.wgts[new_wgt_labels[1]] = min(bin.wgts[label] \
                                                  for label in wgts)
                    bin.wgts[new_wgt_labels[2]] = max(bin.wgts[label] \
                                                  for label in wgts)
                elif type=='PDF' and use_lhapdf and label != 'none' and len(wgts) > 1:
                    pdfs   = [bin.wgts[pdf] for pdf in sorted(wgts)]
                    ep=p.uncertainty(pdfs,-1)
                    bin.wgts[new_wgt_labels[0]] = ep.central
                    bin.wgts[new_wgt_labels[1]] = ep.central-ep.errminus
                    bin.wgts[new_wgt_labels[2]] = ep.central+ep.errplus
                elif type=='PDF' and use_lhapdf and label != 'none' and len(wgts) == 1:
                    bin.wgts[new_wgt_labels[0]] = bin.wgts[wgts[0]]
                    bin.wgts[new_wgt_labels[1]] = bin.wgts[wgts[0]]
                    bin.wgts[new_wgt_labels[2]] = bin.wgts[wgts[0]]
                else:
                    pdfs   = [bin.wgts[pdf] for pdf in sorted(wgts)]
                    pdf_up     = 0.0
                    pdf_down   = 0.0
                    cntrl_val  = bin.wgts['central']
                    pdf_set_id = wgts[0][1]
                    if len(pdfs) <= 2:
                        pdf_up = max(pdfs)
                        pdf_down = min(pdfs)
                    elif pdf_set_id <= 90000:
                        # use Hessian method (CTEQ & MSTW)
                        for i in range(int((len(pdfs)-1)/2)):
                            pdf_up   += max(0.0,pdfs[2*i+1]-cntrl_val,
                                                  pdfs[2*i+2]-cntrl_val)**2
                            pdf_down += max(0.0,cntrl_val-pdfs[2*i+1],
                                                   cntrl_val-pdfs[2*i+2])**2
                        pdf_up   = cntrl_val + math.sqrt(pdf_up)
                        pdf_down = cntrl_val - math.sqrt(pdf_down)
                    elif pdf_set_id in range(90200, 90303) or \
                         pdf_set_id in range(90400, 90433) or \
                         pdf_set_id in range(90700, 90801) or \
                         pdf_set_id in range(90900, 90931) or \
                         pdf_set_id in range(91200, 91303) or \
                         pdf_set_id in range(91400, 91433) or \
                         pdf_set_id in range(91700, 91801) or \
                         pdf_set_id in range(91900, 91931) or \
                         pdf_set_id in range(92000, 92031):
                        # PDF4LHC15 Hessian sets
                        pdf_stdev = 0.0
                        for pdf in pdfs[1:]:
                            pdf_stdev += (pdf - cntrl_val)**2
                        pdf_stdev = math.sqrt(pdf_stdev)
                        pdf_up   = cntrl_val+pdf_stdev
                        pdf_down = cntrl_val-pdf_stdev
                    elif pdf_set_id in range(244400, 244501) or \
                         pdf_set_id in range(244600, 244701) or \
                         pdf_set_id in range(244800, 244901) or \
                         pdf_set_id in range(245000, 245101) or \
                         pdf_set_id in range(245200, 245301) or \
                         pdf_set_id in range(245400, 245501) or \
                         pdf_set_id in range(245600, 245701) or \
                         pdf_set_id in range(245800, 245901) or \
                         pdf_set_id in range(246000, 246101) or \
                         pdf_set_id in range(246200, 246301) or \
                         pdf_set_id in range(246400, 246501) or \
                         pdf_set_id in range(246600, 246701) or \
                         pdf_set_id in range(246800, 246901) or \
                         pdf_set_id in range(247000, 247101) or \
                         pdf_set_id in range(247200, 247301) or \
                         pdf_set_id in range(247400, 247501):
                        # use Gaussian (68%CL) method (NNPDF)
                        pdf_stdev = 0.0
                        pdf_diff = sorted([abs(pdf-cntrl_val) for pdf in pdfs[1:]])
                        pdf_stdev = pdf_diff[67]
                        pdf_up   = cntrl_val+pdf_stdev
                        pdf_down = cntrl_val-pdf_stdev
                    else:
                        # use Gaussian (one sigma) method (NNPDF)
                        pdf_stdev = 0.0
                        for pdf in pdfs[1:]:
                            pdf_stdev += (pdf - cntrl_val)**2
                        pdf_stdev = math.sqrt(pdf_stdev/float(len(pdfs)-2))
                        pdf_up   = cntrl_val+pdf_stdev
                        pdf_down = cntrl_val-pdf_stdev
                    # Finally add them to the corresponding new weight
                    bin.wgts[new_wgt_labels[0]] = bin.wgts[wgts[0]]
                    bin.wgts[new_wgt_labels[1]] = pdf_down
                    bin.wgts[new_wgt_labels[2]] = pdf_up

        # And return the position in self.bins.weight_labels of the first
        # of the two new weight label added.
        return (position,labels)
    
    def select_central_weight(self, selected_label):
        """ Select a specific merging scale for the central value of this Histogram. """
        if selected_label not in self.bins.weight_labels:
            raise MadGraph5Error("Selected weight label '%s' could not be found in this HwU."%selected_label)
        
        for bin in self.bins:
            bin.wgts['central']=bin.wgts[selected_label]                    
    
    def rebin(self, n_rebin):
        """ Rebin the x-axis so as to merge n_rebin consecutive bins into a 
        single one. """
        
        if n_rebin < 1 or not isinstance(n_rebin, int):
            raise MadGraph5Error("The argument 'n_rebin' of the HwU function"+\
              " 'rebin' must be larger or equal to 1, not '%s'."%str(n_rebin))
        elif n_rebin==1:
            return
        
        if self.type and 'NOREBIN' in self.type.upper():
            return

        rebinning_list = list(range(0,len(self.bins),n_rebin))+[len(self.bins),]
        concat_list = [self.bins[rebinning_list[i]:rebinning_list[i+1]] for \
                                              i in range(len(rebinning_list)-1)]
        
        new_bins = copy.copy(self.bins)
        del new_bins[:]
        lazy_labels = tuple(self.bins.weight_labels)
        lazy_label_set = frozenset(lazy_labels)

        for bins_to_merge in concat_list:
            if len(bins_to_merge)==0:
                continue
            if all(getattr(hist_bin.wgts, '_lazy_histogram_weights', False)
                   for hist_bin in bins_to_merge):
                new_bins.append(Bin(
                    boundaries=(bins_to_merge[0].boundaries[0],
                                bins_to_merge[-1].boundaries[1]),
                    wgts=LazyRebinnedWeightView(
                        [hist_bin.wgts for hist_bin in bins_to_merge],
                        lazy_labels, label_set=lazy_label_set)))
                continue
            new_bins.append(Bin(boundaries=(bins_to_merge[0].boundaries[0],
              bins_to_merge[-1].boundaries[1]),wgts={'central':0.0}))
            for weight in self.bins.weight_labels:
                if weight != 'stat_error':
                    new_bins[-1].wgts[weight] = \
                                      sum(b.wgts[weight] for b in bins_to_merge)
                else:
                    new_bins[-1].wgts['stat_error'] = \
                        math.sqrt(sum(b.wgts['stat_error']**2 for b in\
                                                                 bins_to_merge))

        self.bins = new_bins
    
    @classmethod
    def get_x_optimal_range(cls, histo_list, weight_labels=None):
        """ Function to determine the optimal x-axis range when plotting 
        together the histos in histo_list and considering the weights 
        weight_labels"""

        # If no list of weight labels to consider is given, use them all.
        if weight_labels is None:
            weight_labels = histo_list[0].bins.weight_labels

        x_min = None
        x_max = None
        for histo in histo_list:
            for hist_bin in histo.bins:
                if not any(hist_bin.wgts[label] != 0.0
                           for label in weight_labels):
                    continue
                for boundary in hist_bin.boundaries:
                    x_min = boundary if x_min is None else min(x_min, boundary)
                    x_max = boundary if x_max is None else max(x_max, boundary)

        if x_min is None:
            for histo in histo_list:
                for hist_bin in histo.bins:
                    for boundary in hist_bin.boundaries:
                        x_min = boundary if x_min is None else min(x_min, boundary)
                        x_max = boundary if x_max is None else max(x_max, boundary)
            if x_min is None:
                raise MadGraph5Error("The histograms with title '%s'"\
                                  %histo_list[0].title+" seems to have no bins.")
        
        return (x_min, x_max)
    
    @classmethod
    def get_y_optimal_range(cls,histo_list, labels=None, 
                                                   scale='LOG', Kratio = False):
        """ Function to determine the optimal y-axis range when plotting 
        together the histos in histo_list and considering the weights 
        weight_labels. The option Kratio is present to allow for the couple of
        tweaks necessary for the the K-factor ratio histogram y-range."""

        # If no list of weight labels to consider is given, use them all.
        if labels is None:
            weight_labels = histo_list[0].bins.weight_labels
        else:
            weight_labels = labels
        
        # Keep exact ranges, but cap the values retained for quantiles.  Sorting
        # every bin value can otherwise dominate memory for very finely binned
        # plots.  Reservoir sampling is deterministic here so generated plot
        # ranges remain reproducible.
        all_weights = []
        sample_limit = 100000
        sample_random = random.Random(0)
        n_weights_seen = 0
        exact_min = None
        exact_max = None

        def add_weight(value):
            nonlocal n_weights_seen, exact_min, exact_max
            n_weights_seen += 1
            if exact_min is None or value < exact_min:
                exact_min = value
            if exact_max is None or value > exact_max:
                exact_max = value
            if len(all_weights) < sample_limit:
                all_weights.append(value)
            else:
                position = sample_random.randrange(n_weights_seen)
                if position < sample_limit:
                    all_weights[position] = value

        for histo in histo_list:
            for bin in histo.bins:
                for label in weight_labels:
                    # Filter out bin weights at *exactly* because they often
                    # come from pathological division by zero for empty bins.
                    if Kratio and bin.wgts[label]==0.0:
                        continue
                    if scale!='LOG':
                        add_weight(bin.wgts[label])
                        if label == 'stat_error':
                            add_weight(-bin.wgts[label])
                    elif bin.wgts[label] != 0.0:
                        # Logarithmic plots render negative contributions by
                        # their magnitude using a dashed line, so their range
                        # must participate in autoscaling as well.
                        add_weight(abs(bin.wgts[label]))
                        
        
        all_weights.sort()
        if len(all_weights)!=0:
            partial_max = all_weights[int(len(all_weights)*0.95)]
            partial_min = all_weights[int(len(all_weights)*0.05)]
            max         = exact_max
            min         = exact_min
        else:
            if scale!='LOG':
                return (0.0,1.0)
            else:
                return (1.0,10.0)

        y_max = 0.0
        y_min = 0.0

        # If the maximum is too far from the 90% max, then take the partial max
        if (max-partial_max)>2.0*(partial_max-partial_min):
            y_max = partial_max
        else:
            y_max = max
        
        # If the maximum is too far from the 90% max, then take the partial max
        if (partial_min - min)>2.0*(partial_max-partial_min) and min != 0.0:
            y_min = partial_min
        else:
            y_min = min

        if Kratio:
            median = all_weights[len(all_weights)//2]
            spread = (y_max-y_min)
            if abs(y_max-median)<spread*0.05 or abs(median-y_min)<spread*0.05:
                y_max = median + spread/2.0
                y_min = median - spread/2.0
            if y_min != y_max:
                return ( y_min , y_max )

        # Enforce the maximum if there is 5 bins or less
        if len(histo_list[0].bins) <= 5:
            y_min = min
            y_max = max   

        # Finally make sure the range has finite length
        if y_min == y_max:
            if max == min:
                if scale == 'LOG' and y_min > 0.0:
                    y_min /= 10.0
                    y_max *= 10.0
                else:
                    y_min -= 1.0
                    y_max += 1.0
            else:
                y_min = min
                y_max = max         
        
        return ( y_min , y_max )


class DenseHistogramRecord(object):
    """One parsed histogram block in compact, row-major storage.

    Instances are deliberately short lived: the streaming aggregator consumes
    one and releases it before the parser advances to the next histogram block.
    """

    __slots__ = ('title', 'type', 'dimension', 'x_axis_mode', 'y_axis_mode',
                 'has_jetsample', 'jetsample', 'weight_labels', 'boundaries',
                 'values', 'n_bins')

    def __init__(self, histo, weight_labels, boundaries, values):
        weight_labels = tuple(weight_labels)
        if len(weight_labels) != len(set(weight_labels)):
            raise HwU.ParseError("Histogram '%s' has duplicate weight labels."%
                                 histo.title)
        if len(boundaries) % 2:
            raise HwU.ParseError("Histogram '%s' has an incomplete bin boundary."%
                                 histo.title)
        n_bins = len(boundaries)//2
        if len(values) != n_bins*len(weight_labels):
            raise HwU.ParseError("Histogram '%s' has %d values; %d were expected."%
                (histo.title, len(values), n_bins*len(weight_labels)))
        self.title = histo.title
        self.type = histo.type
        self.dimension = histo.dimension
        self.x_axis_mode = histo.x_axis_mode
        self.y_axis_mode = histo.y_axis_mode
        self.has_jetsample = hasattr(histo, 'jetsample')
        self.jetsample = histo.jetsample if self.has_jetsample else None
        self.weight_labels = weight_labels
        self.boundaries = boundaries
        self.values = values
        self.n_bins = n_bins

    @classmethod
    def from_hwu(cls, histo):
        """Create a compact record from the non-streaming XML reader."""

        labels = list(histo.bins.weight_labels)
        boundaries = array.array('d')
        values = array.array('d')
        for hist_bin in histo.bins:
            bin_boundaries = [float(value) for value in hist_bin.boundaries]
            bin_values = [float(hist_bin.wgts[label]) for label in labels]
            if len(bin_boundaries) != 2:
                raise HwU.ParseError("Histogram '%s' has a bin with %d boundaries; "
                                     "2 were expected."%
                                     (histo.title, len(bin_boundaries)))
            if not all(math.isfinite(value) for value in
                                               bin_boundaries+bin_values):
                raise HwU.ParseError("A non-finite value was found in '%s'."%
                                     histo.title)
            if bin_boundaries[0] > bin_boundaries[1]:
                raise HwU.ParseError("A bin in '%s' has decreasing boundaries."%
                                     histo.title)
            boundaries.extend(bin_boundaries)
            values.extend(bin_values)
        return cls(histo, labels, boundaries, values)


def iter_hwu_dense(file_path, accepted_types_order=None,
                   consider_reweights='ALL', merging_scale=None,
                   run_id=None, accepted_titles=None, raw_labels=False):
    """Yield standard HwU input one histogram block at a time.

    The standard HwU path never constructs per-bin dictionaries.  Pythia8 XML
    remains supported through the existing parser because its document-level
    metadata does not follow the block-oriented HwU layout.
    """

    accepted_types_order = accepted_types_order or []
    accepted_titles = accepted_titles or []
    stream = open(file_path, 'r')
    try:
        prefix = stream.read(512)
        stream.seek(0)
        if prefix.lstrip().startswith('<'):
            if prefix.lstrip().startswith('<histogram>'):
                raise HwU.ParseError('The HwU weight header is missing before '
                                     'the first histogram block.')
            xml_histograms = HwUList(stream, run_id=run_id,
                merging_scale=merging_scale,
                accepted_types_order=accepted_types_order,
                consider_reweights=consider_reweights,
                raw_labels=raw_labels)
            for histo in xml_histograms:
                if accepted_titles and not any(title in histo.title
                                                for title in accepted_titles):
                    continue
                yield DenseHistogramRecord.from_hwu(histo)
            return

        weight_header = HwU.parse_weight_header(stream, raw_labels=raw_labels)
        if len(weight_header) != len(set(weight_header)):
            raise HwU.ParseError('The HwU weight header contains duplicate labels.')

        selected_central = None
        if merging_scale is not None:
            for label in weight_header:
                if HwU.get_HwU_wgt_label_type(label) == 'merging_scale' and \
                                               float(label[1]) == merging_scale:
                    selected_central = label
                    break
            if selected_central is None:
                raise MadGraph5Error("No weight could be found in the input "
                    "HwU for the selected merging scale '%4.2f'."%merging_scale)

        retained = []
        for position, label in enumerate(weight_header):
            if consider_reweights != 'ALL' and not raw_labels and \
               label not in ['central', 'stat_error', 'boundary_xmin',
                              'boundary_xmax'] and \
               HwU.get_HwU_wgt_label_type(label) not in consider_reweights:
                continue
            if not raw_labels and isinstance(label, str) and \
                                                   label.endswith('@aux'):
                continue
            retained.append((label, position))

        boundary_positions = {}
        output_columns = []
        for label, position in retained:
            if label in ['boundary_xmin', 'boundary_xmax']:
                boundary_positions[label] = position
            else:
                output_columns.append((label, position))
        if selected_central is not None:
            selected_position = weight_header.index(selected_central)
            output_columns = [(label, selected_position if label == 'central'
                               else position) for label, position in output_columns]

        output_labels = [label for label, _ in output_columns]
        if len(set(output_labels)) != len(output_labels):
            raise HwU.ParseError('The HwU weight header contains duplicate labels.')
        if 'boundary_xmin' not in boundary_positions or \
                                      'boundary_xmax' not in boundary_positions:
            raise HwU.ParseError('The HwU header does not define both bin boundaries.')

        numpy = _optional_numpy()
        numpy_output_positions = numpy.asarray(
            [position for _, position in output_columns], dtype=numpy.intp) \
            if numpy is not None else None

        for line in stream:
            start = HwU.histo_start_re.match(line)
            if start is None:
                if line.lstrip().startswith('<histogram>'):
                    raise HwU.ParseError('A malformed histogram opening tag was '
                                         'found: %s'%line.strip())
                continue

            n_bins = int(start.group('n_bins'))
            metadata = HwU()
            metadata.process_histogram_name(start.group('histo_name'))
            boundaries = array.array('d')
            values = numpy.empty((n_bins, len(output_columns)),
                                 dtype=numpy.float64) if numpy is not None \
                     else array.array('d')
            rows_read = 0
            while rows_read < n_bins:
                line_bin = next(stream, None)
                if line_bin is None:
                    raise HwU.ParseError("Only %d of %d bins were found for '%s'."%
                                         (rows_read, n_bins, metadata.title))
                stripped = line_bin.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if HwU.histo_end_re.match(line_bin):
                    raise HwU.ParseError("Only %d of %d bins were found for '%s' "
                                         "before its closing tag."%
                                         (rows_read, n_bins, metadata.title))
                if HwU.histo_start_re.match(line_bin):
                    raise HwU.ParseError("A new histogram starts before all %d "
                                         "bins of '%s' were read."%
                                         (n_bins, metadata.title))
                numeric_text = line_bin.split('#', 1)[0].strip()
                if not numeric_text:
                    continue
                fields = numeric_text.replace('D', 'E').replace('d', 'e').split()
                if len(fields) != len(weight_header):
                    raise HwU.ParseError("There are %d columns in bin %d of "
                        "'%s'; exactly %d were expected."%(len(fields),
                        rows_read+1, metadata.title, len(weight_header)))
                if numpy is not None:
                    try:
                        # ``asarray`` rejects partial tokens such as ``1.0x``;
                        # ``fromstring`` silently accepted those on some NumPy
                        # versions.
                        parsed = numpy.asarray(fields, dtype=numpy.float64)
                    except (TypeError, ValueError, OverflowError):
                        raise HwU.ParseError("A non-numeric value was found in bin "
                            "%d of '%s'."%(rows_read+1, metadata.title))
                    if not numpy.isfinite(parsed).all():
                        raise HwU.ParseError("A non-finite value was found in bin "
                            "%d of '%s'."%(rows_read+1, metadata.title))
                else:
                    try:
                        parsed = [float(value) for value in fields]
                    except (TypeError, ValueError, OverflowError):
                        raise HwU.ParseError("A non-numeric value was found in bin "
                            "%d of '%s'."%(rows_read+1, metadata.title))
                    if not all(math.isfinite(value) for value in parsed):
                        raise HwU.ParseError("A non-finite value was found in bin "
                            "%d of '%s'."%(rows_read+1, metadata.title))
                lower = float(parsed[boundary_positions['boundary_xmin']])
                upper = float(parsed[boundary_positions['boundary_xmax']])
                if lower > upper:
                    raise HwU.ParseError("Bin %d of '%s' has decreasing boundaries "
                                         "(%s > %s)."%
                                         (rows_read+1, metadata.title,
                                          lower, upper))
                boundaries.append(lower)
                boundaries.append(upper)
                if numpy is not None:
                    values[rows_read, :] = parsed[numpy_output_positions]
                else:
                    values.extend(float(parsed[position])
                                  for _, position in output_columns)
                rows_read += 1

            for end_line in stream:
                stripped = end_line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if HwU.histo_end_re.match(end_line):
                    break
                raise HwU.ParseError("Unexpected content after the %d bins of "
                                     "'%s'; a closing histogram tag was expected."%
                                     (n_bins, metadata.title))
            else:
                raise HwU.ParseError("The closing histogram tag for '%s' is missing."%
                                     metadata.title)

            if metadata.type == 'AUX':
                continue
            if accepted_types_order and metadata.type not in accepted_types_order:
                continue
            if accepted_titles and not any(title in metadata.title
                                            for title in accepted_titles):
                continue
            if numpy is not None:
                values = values.reshape(-1)
            yield DenseHistogramRecord(metadata, output_labels, boundaries, values)
    finally:
        stream.close()


class _SpillArena(object):
    """Shared file-backed mmap arena used by many dense buffers."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise MadGraph5Error('Spill arena capacity must be positive.')
        self.capacity = capacity
        self.used = 0
        self._file = None
        self._mapping = None
        try:
            self._file = tempfile.TemporaryFile()
            self._file.truncate(capacity)
            self._mapping = mmap.mmap(self._file.fileno(), capacity)
            # mmap keeps its own OS handle.  Closing the Python file object
            # avoids retaining a second descriptor for the arena.
            self._file.close()
            self._file = None
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def allocate(self, n_bytes):
        if n_bytes < 0 or self.used+n_bytes > self.capacity:
            raise MadGraph5Error('Dense spill arena is out of space.')
        start = self.used
        self.used += n_bytes
        return memoryview(self._mapping)[start:self.used]

    @property
    def remaining(self):
        return self.capacity-self.used

    def close(self):
        if self._mapping is not None:
            mapping = self._mapping
            try:
                mapping.close()
            except BufferError:
                raise
            else:
                self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _DenseDoubleBuffer(object):
    """Fixed-size double buffer backed by RAM or mmap spill storage."""

    def __init__(self, size, use_mmap=False, numpy=None, backing_view=None):
        if not isinstance(size, int) or size < 0:
            raise MadGraph5Error("Dense buffer size must be a non-negative integer.")
        self.size = size
        self.numpy = numpy
        self._file = None
        self._mapping = None
        self.data = None
        try:
            if backing_view is not None:
                if len(backing_view) != size*8:
                    raise MadGraph5Error('Dense buffer backing has the wrong size.')
                if numpy is not None:
                    self.data = numpy.frombuffer(backing_view,
                                      dtype=numpy.float64, count=size)
                else:
                    self.data = backing_view.cast('d')
            elif use_mmap and size:
                self._file = tempfile.TemporaryFile()
                self._file.truncate(size*8)
                self._mapping = mmap.mmap(self._file.fileno(), size*8)
                # mmap owns the backing storage after construction; avoid
                # retaining the temporary file's additional descriptor.
                self._file.close()
                self._file = None
                if numpy is not None:
                    self.data = numpy.frombuffer(self._mapping,
                                      dtype=numpy.float64, count=size)
                else:
                    self.data = memoryview(self._mapping).cast('d')
            elif numpy is not None:
                self.data = numpy.zeros(size, dtype=numpy.float64)
            else:
                self.data = array.array('d', [0.0])*size
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def close(self):
        if isinstance(self.data, memoryview):
            self.data.release()
        # Drop ndarray/memoryview references before closing mmap backing.
        self.data = None
        if self._mapping is not None:
            mapping = self._mapping
            try:
                mapping.close()
            except BufferError:
                # An external exported view still exists.  Retain the mapping
                # so a later close can retry rather than silently leaking it.
                raise
            else:
                self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DenseWeightView(MutableMapping):
    """Dictionary-compatible view of one row in a dense accumulator."""

    _validated_histogram_weights = True
    _lazy_histogram_weights = True

    def __init__(self, accumulator, bin_index, overlay=None, deleted=None):
        self._accumulator = accumulator
        self._bin_index = bin_index
        self._overlay = {} if overlay is None else overlay
        self._deleted = set() if deleted is None else deleted

    def __getitem__(self, key):
        if key in self._deleted:
            raise KeyError(key)
        if key in self._overlay:
            return self._overlay[key]
        return self._accumulator.get_value(self._bin_index, key)

    def __setitem__(self, key, value):
        if not isinstance(value, float):
            value = float(value)
        self._deleted.discard(key)
        self._overlay[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._overlay.pop(key, None)
        self._deleted.add(key)

    def __iter__(self):
        for label in self._accumulator.weight_labels:
            if label not in self._deleted:
                yield label
        for label in self._overlay:
            if label not in self._deleted and \
                                  label not in self._accumulator.label_positions:
                yield label

    def __len__(self):
        return sum(1 for _ in self)

    def __deepcopy__(self, memo):
        result = self.__class__(self._accumulator, self._bin_index,
                                copy.deepcopy(self._overlay, memo),
                                copy.deepcopy(self._deleted, memo))
        memo[id(self)] = result
        return result


class DenseHistogramAccumulator(object):
    """Numerically stable dense sum of compatible histogram blocks."""

    def __init__(self, record, allocate, numpy=None):
        self.title = record.title
        self.type = record.type
        self.dimension = record.dimension
        self.x_axis_mode = record.x_axis_mode
        self.y_axis_mode = record.y_axis_mode
        self.has_jetsample = record.has_jetsample
        self.jetsample = record.jetsample
        self.weight_labels = record.weight_labels
        self.label_positions = dict((label, index) for index, label in
                                    enumerate(self.weight_labels))
        self.boundaries = array.array('d', record.boundaries)
        self.n_bins = record.n_bins
        self.n_weights = len(self.weight_labels)
        self.numpy = numpy
        self._closed = False
        self._buffers = []
        size = self.n_bins*self.n_weights
        try:
            self._sums = allocate(size)
            self._buffers.append(self._sums)
            self._compensations = allocate(size)
            self._buffers.append(self._compensations)
            self._variances = allocate(self.n_bins)
            self._buffers.append(self._variances)
            self._variance_compensations = allocate(self.n_bins)
            self._buffers.append(self._variance_compensations)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _ensure_open(self):
        if self._closed:
            raise MadGraph5Error("Histogram accumulator '%s' is closed."%
                                 self.title)

    @staticmethod
    def _kahan_add(total, compensation, index, value):
        if not math.isfinite(value):
            raise MadGraph5Error('A non-finite value was encountered while '
                                 'aggregating histogram data.')
        corrected = value-compensation[index]
        updated = total[index]+corrected
        new_compensation = (updated-total[index])-corrected
        if not math.isfinite(updated) or not math.isfinite(new_compensation):
            raise MadGraph5Error('Floating-point overflow while aggregating '
                                 'histogram data.')
        compensation[index] = new_compensation
        total[index] = updated

    def add(self, record, factor=1.0):
        self._ensure_open()
        try:
            factor = float(factor)
        except (TypeError, ValueError, OverflowError):
            raise MadGraph5Error("Histogram aggregation factor '%s' is not a "
                                 "number."%str(factor))
        if not math.isfinite(factor):
            raise MadGraph5Error('Histogram aggregation factors must be finite.')
        if record.weight_labels != self.weight_labels:
            if set(record.weight_labels) != set(self.weight_labels):
                missing = set(self.weight_labels)-set(record.weight_labels)
                extra = set(record.weight_labels)-set(self.weight_labels)
                raise MadGraph5Error("Incompatible weights for histogram '%s' "
                    "(missing: %s; extra: %s)."%(self.title, list(missing),
                    list(extra)))
            record_positions = dict((label, index) for index, label in
                                    enumerate(record.weight_labels))
            source_positions = [record_positions[label]
                                for label in self.weight_labels]
        else:
            source_positions = None
        if record.n_bins != self.n_bins or record.boundaries != self.boundaries:
            raise MadGraph5Error("Histogram '%s' has incompatible bin boundaries."%
                                 self.title)

        stat_position = self.label_positions.get('stat_error')
        if self.numpy is not None:
            source = self.numpy.frombuffer(record.values,
                dtype=self.numpy.float64).reshape(self.n_bins, self.n_weights)
            sums = self._sums.data.reshape(self.n_bins, self.n_weights)
            compensations = self._compensations.data.reshape(
                                               self.n_bins, self.n_weights)
            # Keep temporary arrays for one chunk near 32--40 MiB in total.
            # In particular, reordering columns must not copy a full large
            # histogram before accumulation even starts.
            chunk_bins = max(1, (1 << 20)//max(1, self.n_weights))
            for start in range(0, self.n_bins, chunk_bins):
                end = min(self.n_bins, start+chunk_bins)
                source_chunk = source[start:end]
                if source_positions is not None:
                    source_chunk = source_chunk[:, source_positions]
                with self.numpy.errstate(over='ignore', invalid='ignore'):
                    scaled = source_chunk*factor
                    if not self.numpy.isfinite(scaled).all():
                        raise MadGraph5Error("Floating-point overflow while "
                            "scaling histogram '%s'."%self.title)
                    variance_updated = None
                    variance_compensation = None
                    if stat_position is not None:
                        errors = scaled[:, stat_position]
                        variance_values = errors*errors
                        old_variances = self._variances.data[start:end]
                        old_compensations = \
                            self._variance_compensations.data[start:end]
                        variance_correction = (variance_values-
                                                old_compensations)
                        variance_updated = old_variances+variance_correction
                        variance_compensation = (
                            variance_updated-old_variances)-variance_correction
                        if not self.numpy.isfinite(variance_updated).all() or \
                           not self.numpy.isfinite(variance_compensation).all():
                            raise MadGraph5Error("Floating-point overflow while "
                                "combining statistical errors for histogram "
                                "'%s'."%self.title)
                        scaled[:, stat_position] = 0.0

                    sum_chunk = sums[start:end]
                    compensation_chunk = compensations[start:end]
                    corrected = scaled-compensation_chunk
                    updated = sum_chunk+corrected
                    new_compensations = (updated-sum_chunk)-corrected
                    if not self.numpy.isfinite(updated).all() or \
                               not self.numpy.isfinite(new_compensations).all():
                        raise MadGraph5Error("Floating-point overflow while "
                            "combining histogram '%s'."%self.title)

                if variance_updated is not None:
                    self._variance_compensations.data[start:end] = \
                                                       variance_compensation
                    self._variances.data[start:end] = variance_updated
                compensation_chunk[:] = new_compensations
                sum_chunk[:] = updated
            return

        positions = range(self.n_weights) if source_positions is None \
                                            else source_positions
        for bin_index in range(self.n_bins):
            source_offset = bin_index*self.n_weights
            target_offset = source_offset
            for target_position, source_position in enumerate(positions):
                value = record.values[source_offset+source_position]*factor
                if target_position == stat_position:
                    self._kahan_add(self._variances,
                                    self._variance_compensations, bin_index,
                                    value*value)
                else:
                    self._kahan_add(self._sums, self._compensations,
                                    target_offset+target_position, value)

    def get_value(self, bin_index, label):
        self._ensure_open()
        try:
            position = self.label_positions[label]
        except KeyError:
            raise KeyError(label)
        if label == 'stat_error':
            return math.sqrt(max(0.0, self._variances[bin_index]))
        return float(self._sums[bin_index*self.n_weights+position])

    def to_hwu(self):
        self._ensure_open()
        histo = HwU()
        histo.title = self.title
        histo.type = self.type
        histo.dimension = self.dimension
        histo.x_axis_mode = self.x_axis_mode
        histo.y_axis_mode = self.y_axis_mode
        if self.has_jetsample:
            histo.jetsample = self.jetsample
        histo.bins = BinList(weight_labels=list(self.weight_labels))
        for bin_index in range(self.n_bins):
            boundaries = (float(self.boundaries[2*bin_index]),
                          float(self.boundaries[2*bin_index+1]))
            histo.bins.append(Bin(boundaries,
                                  DenseWeightView(self, bin_index)))
        return histo

    def close(self):
        if self._closed:
            return
        first_error = None
        for buffer_ in self._buffers:
            try:
                buffer_.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        self._closed = True


class StreamingHwUAggregator(object):
    """Aggregate many HwU files with memory independent of the file count."""

    _spill_initial_arena_size = 16*1024*1024
    _spill_arena_size = 256*1024*1024

    def __init__(self, memory_limit=256*1024*1024):
        try:
            self.memory_limit = int(memory_limit)
        except (TypeError, ValueError, OverflowError):
            raise MadGraph5Error("Histogram memory limit '%s' is not an integer."%
                                 str(memory_limit))
        if self.memory_limit < 0:
            raise MadGraph5Error('Histogram memory limit must be non-negative.')
        self.resident_bytes = 0
        self.numpy = _optional_numpy()
        self.accumulators = collections.OrderedDict()
        self.expected_keys = None
        self.n_files = 0
        self.accepted_types_order = []
        self._spill_arenas = []
        self._closed = False
        self._failed_error = None

    def _ensure_usable(self):
        if self._failed_error is not None:
            raise MadGraph5Error('This streaming histogram aggregator cannot be '
                'reused because a previous input failed: %s. Create a new '
                'aggregator.'%str(self._failed_error))
        if self._closed:
            raise MadGraph5Error('This streaming histogram aggregator is closed.')

    def _release_resources(self):
        first_error = None
        for accumulator in self.accumulators.values():
            try:
                accumulator.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        for arena in self._spill_arenas:
            try:
                arena.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        self.accumulators.clear()
        self._spill_arenas = []
        self.resident_bytes = 0

    def _allocate_spill(self, n_bytes):
        if self._spill_arenas and \
                           self._spill_arenas[-1].remaining >= n_bytes:
            return self._spill_arenas[-1].allocate(n_bytes)
        capacity = max(self._spill_initial_arena_size, n_bytes)
        if self._spill_arenas:
            capacity = max(capacity, min(self._spill_arena_size,
                                         2*self._spill_arenas[-1].capacity))
        arena = _SpillArena(capacity)
        self._spill_arenas.append(arena)
        return arena.allocate(n_bytes)

    def _allocate(self, size):
        self._ensure_usable()
        n_bytes = size*8
        use_mmap = self.resident_bytes+n_bytes > self.memory_limit
        if not use_mmap:
            self.resident_bytes += n_bytes
            return _DenseDoubleBuffer(size, numpy=self.numpy)
        return _DenseDoubleBuffer(size, numpy=self.numpy,
                                  backing_view=self._allocate_spill(n_bytes))

    @staticmethod
    def _identity_base(record):
        return (record.title, record.type, record.dimension,
                record.x_axis_mode, record.y_axis_mode,
                record.has_jetsample, record.jetsample, record.n_bins,
                hashlib.sha1(record.boundaries).digest(),
                frozenset(record.weight_labels))

    def add_file(self, file_path, factor=1.0, accepted_types_order=None,
                 consider_reweights='ALL', merging_scale=None, run_id=None,
                 accepted_titles=None):
        """Add one file, permanently invalidating this object on failure.

        Aggregation updates dense sums in place.  Invalidating on any failure
        prevents a caller from accidentally emitting partially updated results.
        """

        self._ensure_usable()
        try:
            factor = float(factor)
        except (TypeError, ValueError, OverflowError):
            raise MadGraph5Error("Histogram aggregation factor '%s' is not a "
                                 "number."%str(factor))
        if not math.isfinite(factor):
            raise MadGraph5Error('Histogram aggregation factors must be finite.')
        try:
            self._add_file(file_path, factor=factor,
                accepted_types_order=accepted_types_order,
                consider_reweights=consider_reweights,
                merging_scale=merging_scale, run_id=run_id,
                accepted_titles=accepted_titles)
        except BaseException as error:
            self._failed_error = error
            try:
                self._release_resources()
            except Exception:
                logger.debug('Failed to release histogram aggregation buffers.',
                             exc_info=True)
            raise

    def _add_file(self, file_path, factor=1.0, accepted_types_order=None,
                  consider_reweights='ALL', merging_scale=None, run_id=None,
                  accepted_titles=None):
        if self.n_files == 0:
            self.accepted_types_order = list(accepted_types_order or [])
        occurrences = collections.defaultdict(int)
        seen = set()
        for record in iter_hwu_dense(file_path,
                accepted_types_order=accepted_types_order,
                consider_reweights=consider_reweights,
                merging_scale=merging_scale, run_id=run_id,
                accepted_titles=accepted_titles):
            base = self._identity_base(record)
            occurrence = occurrences[base]
            occurrences[base] += 1
            # Reuse ``base``: hashing all bin boundaries is significant for
            # large files and the occurrence suffix already makes this key
            # unique within the file.
            key = base+(occurrence,)
            seen.add(key)
            if self.expected_keys is not None and key not in self.expected_keys:
                raise MadGraph5Error("Histograms in '%s' are not compatible "
                    "and cannot be combined: unexpected histogram '%s' "
                    "(type %s)."%(file_path, record.title, record.type))
            if key not in self.accumulators:
                self.accumulators[key] = DenseHistogramAccumulator(record,
                                            self._allocate, numpy=self.numpy)
            self.accumulators[key].add(record, factor=factor)

        if self.expected_keys is None:
            if not seen:
                raise MadGraph5Error("No histograms were found in '%s'."%file_path)
            self.expected_keys = set(seen)
        elif seen != self.expected_keys:
            missing = self.expected_keys-seen
            raise MadGraph5Error("Histograms in '%s' are not compatible and "
                "cannot be combined: %d expected histogram(s) are missing: %s."%
                (file_path, len(missing), list(missing)[:5]))
        self.n_files += 1

    def to_hwu_list(self, n_rebin=1):
        """Return lightweight views valid until this aggregator is closed.

        For long-lived independent results, write them with :meth:`output` and
        reload the resulting HwU file instead of retaining every weight in RAM.
        """

        self._ensure_usable()
        result = HwUList([])
        for group in self.iter_hwu_groups(n_rebin=n_rebin):
            result.extend(group)
        return result

    def _ordered_accumulators(self):
        self._ensure_usable()
        accumulators = list(self.accumulators.values())
        title_order = collections.OrderedDict()
        for accumulator in accumulators:
            title_order.setdefault(accumulator.title, len(title_order))

        def ordering_key(accumulator):
            if self.accepted_types_order:
                type_position = self.accepted_types_order.index(accumulator.type)
            else:
                type_position = {'NLO':1, 'LO':2, None:3, 'AUX':5}.get(
                                                       accumulator.type, 4)
            return (title_order[accumulator.title], type_position)

        return sorted(accumulators, key=ordering_key)

    @staticmethod
    def _accumulator_plot_key(accumulator):
        boundary_hash = hashlib.sha1(accumulator.boundaries).digest()
        return (accumulator.title, accumulator.dimension,
                accumulator.x_axis_mode, accumulator.y_axis_mode,
                accumulator.n_bins, boundary_hash,
                frozenset(accumulator.weight_labels))

    def iter_hwu_groups(self, n_rebin=1):
        """Yield one lightweight plot group at a time."""

        self._ensure_usable()
        grouped = collections.OrderedDict()
        for accumulator in self._ordered_accumulators():
            key = self._accumulator_plot_key(accumulator)
            grouped.setdefault(key, []).append(accumulator)

        for accumulators in grouped.values():
            group = HwUList([])
            for accumulator in accumulators:
                histo = accumulator.to_hwu()
                if n_rebin > 1:
                    histo.rebin(n_rebin)
                group.append(histo)
            yield group

    def common_weight_schema(self):
        self._ensure_usable()
        accumulators = list(self.accumulators.values())
        if not accumulators:
            raise MadGraph5Error('No histograms have been aggregated.')
        schema = list(accumulators[0].weight_labels)
        if len(schema) != len(set(schema)):
            raise MadGraph5Error('Histogram weight columns must be unique.')
        expected = set(schema)
        for accumulator in accumulators[1:]:
            labels = list(accumulator.weight_labels)
            if len(labels) != len(set(labels)) or set(labels) != expected:
                raise MadGraph5Error("Histograms written to one HwU file must "
                    "have the same weight columns; '%s' has a different schema."%
                    accumulator.title)
        return schema

    def output(self, path, n_rebin=1, **options):
        """Render aggregated data without materializing every bin at once."""

        self._ensure_usable()
        proxy = HwUList([])
        return proxy.output(path,
            _histogram_groups=self.iter_hwu_groups(n_rebin=n_rebin),
            _weight_schema=self.common_weight_schema(), **options)

    def __len__(self):
        self._ensure_usable()
        return len(self.accumulators)

    def close(self):
        if self._closed:
            return
        self._release_resources()
        self._closed = True

    def __enter__(self):
        self._ensure_usable()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _LineStreamSink(object):
    """List-like line sink which does not retain generated HwU data."""

    def __init__(self, stream):
        self.stream = stream
        self.first = True

    def append(self, line):
        if not self.first:
            self.stream.write('\n')
        self.stream.write(str(line))
        self.first = False

    def extend(self, lines):
        for line in lines:
            self.append(line)


class _TeeLineSink(object):
    """Forward list-like line writes to multiple streaming sinks."""

    def __init__(self, *sinks):
        self.sinks = sinks

    def append(self, line):
        for sink in self.sinks:
            sink.append(line)

    def extend(self, lines):
        for line in lines:
            self.append(line)


def _json_for_html(value):
    """Serialize JSON without permitting an embedded closing script tag."""

    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).\
        replace('&', r'\u0026').replace('<', r'\u003c').\
        replace('>', r'\u003e')


class _EscapedJSONStringLineSink(object):
    """Stream lines as the contents of one safely embedded JSON string."""

    def __init__(self, stream):
        self.stream = stream
        self.first = True

    def append(self, line):
        if not self.first:
            self.stream.write(r'\n')
        encoded = _json_for_html(str(line))
        self.stream.write(encoded[1:-1])
        self.first = False

    def extend(self, lines):
        for line in lines:
            self.append(line)


class _HTMLHistogramWriter(object):
    """Stream a self-contained HTML histogram document."""

    def __init__(self, stream, output_base_name, arg_string):
        self.stream = stream
        title = html_module.escape(output_base_name+' histograms', quote=True)
        command = html_module.escape(
            arg_string.replace('\r', ' ').replace('\n', ' '), quote=True)
        stream.write(_HTML_DOCUMENT_PREFIX % {
            'title': title,
            'command': command})
        self.data_sink = _EscapedJSONStringLineSink(stream)

    def finish(self, plot_specs):
        self.stream.write('"</script>\n')
        self.stream.write(
            '<script id="plot-specs" type="application/json">')
        self.stream.write(_json_for_html(plot_specs))
        self.stream.write('</script>\n')
        self.stream.write(_HTML_SCRIPT_BODY)


class _AtomicTextOutput(object):
    """Write a text output beside its target and replace it only on success."""

    def __init__(self, target_path):
        self.target_path = target_path
        target_directory = os.path.dirname(os.path.abspath(target_path))
        prefix = '.'+os.path.basename(target_path)+'.'
        self.temporary_path = None
        descriptor = None
        for unused_attempt in range(100):
            candidate = os.path.join(target_directory, prefix+
                                     os.urandom(12).hex()+'.tmp')
            try:
                descriptor = os.open(candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            except OSError as error:
                if error.errno == errno.EEXIST:
                    continue
                raise
            self.temporary_path = candidate
            break
        if descriptor is None:
            raise MadGraph5Error("Could not create a temporary output beside '%s'."%
                                 target_path)
        try:
            self.stream = os.fdopen(descriptor, 'w')
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(self.temporary_path)
            except OSError:
                pass
            self.temporary_path = None
            raise
        self._committed = False
        try:
            self._existing_mode = os.stat(target_path).st_mode & 0o7777
        except OSError:
            self._existing_mode = None

    def commit(self):
        if self._committed:
            return
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
        if self._existing_mode is not None:
            os.chmod(self.temporary_path, self._existing_mode)
        os.replace(self.temporary_path, self.target_path)
        self.temporary_path = None
        self._committed = True

    def abort(self):
        try:
            if not self.stream.closed:
                self.stream.close()
        except Exception:
            pass
        if self.temporary_path is not None:
            try:
                os.unlink(self.temporary_path)
            except OSError:
                pass
            self.temporary_path = None

    def __del__(self):
        try:
            if not self._committed:
                self.abort()
        except Exception:
            pass

class HwUList(histograms_PhysicsObjectList):
    """ A class implementing features related to a list of Hwu Histograms. """
    
    # Define here the number of line color schemes defined. If you need more,
    # simply define them in the gnuplot header and increase the number below.
    # It must be <= 9.
    number_line_colors_defined = 8
    
    def is_valid_element(self, obj):
        """Test wether specified object is of the right type for this list."""

        return isinstance(obj, HwU) or isinstance(obj, HwUList)

    def __init__(self, file_path, weight_header=None, run_id=None,
            merging_scale=None, accepted_types_order=[], consider_reweights='ALL',
                                                         raw_labels=False, **opts):
        """ Read one plot from a file_path or a stream. 
        This constructor reads all plots specified in target file.
        File_path can be a path or a stream in the argument.
        The option weight_header specifies an ordered list of weight names 
        to appear in the file or stream specified. It accepted_types_order is 
        empty, no filter is applied, otherwise only histograms of the specified  
        types will be kept, and in this specified order for a given identical 
        title. The option 'consider_reweights' selects whether one wants to 
        include all the extra scale/pdf/merging variation weights. Possible values
        are 'ALL' or a list of the return types of the function get_HwU_wgt_label_type().
        The option 'raw_labels' specifies that one wants to import the
        histogram data with no treatment of the weight labels at all
        (this is used for the matplotlib output).
        """
        
        if isinstance(file_path, str):
            stream = open(file_path,'r')
        elif isinstance(file_path, file):
            stream = file_path
        else:
            return super(HwUList,self).__init__(file_path, **opts)

        try:
            # Try to read it in XML format
            self.parse_histos_from_PY8_XML_stream(stream, run_id, 
                    merging_scale, accepted_types_order,
                    consider_reweights=consider_reweights,
                    raw_labels=raw_labels)  
        except XMLParsingError:
            # Rewinding the stream
            stream.seek(0)
            # Attempt to find the weight headers if not specified        
            if not weight_header:
                weight_header = HwU.parse_weight_header(stream,raw_labels=raw_labels)
        
            # Select a specific merging scale if asked for:
            selected_label = None
            if not merging_scale is None: 
                for label in weight_header:
                    if HwU.get_HwU_wgt_label_type(label)=='merging_scale':
                        if float(label[1])==merging_scale:
                            selected_label = label
                            break
                if selected_label is None:
                    raise MadGraph5Error("No weight could be found in the input HwU "+\
                      "for the selected merging scale '%4.2f'."%merging_scale)

            new_histo = HwU(stream, weight_header,raw_labels=raw_labels,
                            consider_reweights=consider_reweights,
                            selected_central_weight=selected_label)
#            new_histo.select_central_weight(selected_label)           
            while not new_histo.bins is None:
                if accepted_types_order==[] or \
                                         new_histo.type in accepted_types_order:
                    self.append(new_histo)
                new_histo = HwU(stream, weight_header, raw_labels=raw_labels,
                                consider_reweights=consider_reweights,
                                selected_central_weight=selected_label)
                
        #    if not run_id is None:
        #        logger.debug("The run_id '%s' was specified, but "%run_id+
        #                    "format of the HwU plot source is the MG5aMC"+
        #                    " so that the run_id information is ignored.")

        # Order the histograms according to their type.
        titles_order = [h.title for h in self]
        def ordering_function(histo):
            title_position = titles_order.index(histo.title)
            if accepted_types_order==[]:
                type_precedence = {'NLO':1,'LO':2,None:3,'AUX':5}
                try:
                    ordering_key = (title_position,type_precedence[histo.type])
                except KeyError:
                    ordering_key = (title_position,4)
            else:
                ordering_key = (title_position,
                                         accepted_types_order.index(histo.type))
            return ordering_key

        # The command below is to first order them in alphabetical order, but it
        # is often better to keep the order of the original HwU source.
#        self.sort(key=lambda histo: '%s_%d'%(histo.title,
#                                                  type_order.index(histo.type)))
        self.sort(key=ordering_function)
    
        # Explicitly close the opened stream for clarity.
        if isinstance(file_path, str):
            stream.close()

    def get_hist_names(self):
        """return a list of all the names of define histograms"""

        output = []
        for hist in self:
            output.append(hist.get_HwU_histogram_name())
        return output
    
    def get_wgt_names(self):
        """ return the list of all weights define in each histograms"""
        
        return self[0].bins.weight_labels

    @staticmethod
    def _plot_group_key(histo):
        """Return the compatibility key used to group curves in linear time."""

        boundary_hash = hashlib.sha1()
        for hist_bin in histo.bins:
            for boundary in hist_bin.boundaries:
                boundary_hash.update(('%+.17e;'%boundary).encode('ascii'))
        return (histo.title, histo.dimension, histo.x_axis_mode,
                histo.y_axis_mode, len(histo.bins), boundary_hash.digest(),
                frozenset(histo.bins.weight_labels))

    @classmethod
    def _group_for_output(cls, histograms):
        """Group compatible histograms without the former quadratic scan."""

        groups = collections.OrderedDict()
        for histo in histograms:
            key = cls._plot_group_key(histo)
            if key not in groups:
                groups[key] = HwUList([])
            groups[key].append(histo)
        return HwUList(groups.values())

    @staticmethod
    def _common_weight_schema(histograms):
        """Validate and return the single column schema required by HwU."""

        schema = list(histograms[0].bins.weight_labels)
        if len(schema) != len(set(schema)):
            raise MadGraph5Error("Histogram '%s' has duplicate weight columns."%
                                 histograms[0].title)
        expected = set(schema)
        for histo in histograms[1:]:
            labels = list(histo.bins.weight_labels)
            if len(labels) != len(set(labels)) or set(labels) != expected:
                raise MadGraph5Error("Histograms written to one HwU file must "
                    "have the same weight columns; '%s' has a different schema."%
                    histo.title)
        return schema
        
    
    def get(self, name):
        """return the HWU histograms related to a given name"""
        for hist in self:
            if hist.get_HwU_histogram_name() == name:
                return hist

        raise NameError("no histogram with name: %s" % name)
    
    def parse_histos_from_PY8_XML_stream(self, stream, run_id=None, 
            merging_scale=None, accepted_types_order=[], 
            consider_reweights='ALL', raw_labels=False):
        """Initialize the HwU histograms from an XML stream. Only one run is 
        used: the first one if run_id is None or the specified run otherwise.
        Accepted type order is a filter to select histograms of only a certain
        type. The option 'consider_reweights' selects whether one wants to 
        include all the extra scale/pdf/merging variation weights.
        Possible values are 'ALL' or a list of the return types of the
        function get_HwU_wgt_label_type()."""
        
        run_nodes = minidom.parse(stream).getElementsByTagName("run")
        all_nodes = dict((int(node.getAttribute('id')),node) for
                                                              node in run_nodes)
        selected_run_node = None
        weight_header     = None
        if run_id is None:
            if len(run_nodes)>0:
                selected_run_node = all_nodes[min(all_nodes.keys())]
        else:
            try:
                selected_run_node = all_nodes[int(run_id)]
            except:
                selected_run_node = None
        
        if selected_run_node is None:
            if run_id is None:
                raise MadGraph5Error('No histogram was found in the specified XML source.')
            else:
                raise MadGraph5Error("Histogram with run_id '%d' was not found in the "%run_id+\
                                                         "specified XML source.")
 
        # If raw weight label are asked for, then simply read the weight_labels
        # directly as specified in the XML header
        if raw_labels:
            # Filter empty weights coming from the split
            weight_label_list = [wgt.strip() for wgt in 
                str(selected_run_node.getAttribute('header')).split(';') if
                                                      not re.match(r'^\s*$',wgt)]
            ordered_weight_label_list = [w for w in weight_label_list if w not\
                                                             in ['xmin','xmax']]
            # Remove potential repetition of identical weight labels
            filtered_ordered_weight_label_list = []
            for wgt_label in ordered_weight_label_list:
                if wgt_label not in filtered_ordered_weight_label_list:
                    filtered_ordered_weight_label_list.append(wgt_label)
    
            selected_weights = dict([ (wgt_pos, 
             [wgt if wgt not in ['xmin','xmax'] else HwU.mandatory_weights[wgt]])
                 for wgt_pos, wgt in enumerate(weight_label_list) if wgt in 
                            filtered_ordered_weight_label_list+['xmin','xmax']])

            return self.retrieve_plots_from_XML_source(selected_run_node,
                   selected_weights, filtered_ordered_weight_label_list,
                                                                raw_labels=True)

        # Now retrieve the header and save all weight labels as dictionaries
        # with key being properties and their values as value. If the property
        # does not defined a value, then put None as a value
        all_weights = []
        for wgt_position, wgt_label in \
            enumerate(str(selected_run_node.getAttribute('header')).split(';')):
            if not re.match(r'^\s*$',wgt_label) is None:
                continue
            all_weights.append({'POSITION':wgt_position})
            for wgt_item in wgt_label.strip().split('_'):
                property = wgt_item.strip().split('=')
                if len(property) == 2:
                    all_weights[-1][property[0].strip()] = property[1].strip()
                elif len(property)==1:
                    all_weights[-1][property[0].strip()] = None
                #else:
                #    misc.sprint(all_weights)
                #    raise MadGraph5Error("The weight label property %s could not be parsed."%wgt_item)
        
        # Now make sure that for all weights, there is 'PDF', 'MUF' and 'MUR' 
        # and 'MERGING' defined. If absent we specify '-1' which implies that
        # the 'default' value was used (whatever it was).
        # Also cast them in the proper type
        for wgt_label in all_weights:
            for mandatory_attribute in ['PDF','MUR','MUF','MERGING','ALPSFACT']:
                if mandatory_attribute not in wgt_label or wgt_label[mandatory_attribute] is None:
                    wgt_label[mandatory_attribute] = '-1'
                elif mandatory_attribute=='PDF':
                    wgt_label[mandatory_attribute] = int(wgt_label[mandatory_attribute])
                elif mandatory_attribute in ['MUR','MUF','MERGING','ALPSFACT']:
                    wgt_label[mandatory_attribute] = float(wgt_label[mandatory_attribute])                

        # If merging cut is negative, then pick only the one of the central scale
        # If not specified, then take them all but use the PDF and scale weight
        # of the central merging_scale for the variation.
        if not all_weights:
            raise MadGraph5Error('No weights were found in the HwU XML source.')
        if merging_scale is None or merging_scale < 0.0:
            merging_scale_chosen = all_weights[2]['MERGING']
        else:
            merging_scale_chosen = merging_scale

        # Central weight parameters are enforced to be those of the third weight
        central_PDF  = all_weights[2]['PDF']
        # Assume central scale is one, unless specified.
        central_MUR   = all_weights[2]['MUR'] if all_weights[2]['MUR']!=-1.0 else 1.0
        central_MUF   = all_weights[2]['MUF'] if all_weights[2]['MUF']!=-1.0 else 1.0
        central_alpsfact = all_weights[2]['ALPSFACT'] if all_weights[2]['ALPSFACT']!=-1.0 else 1.0
        
        # Dictionary of selected weights with their position as key and the
        # list of weight labels they correspond to.
        selected_weights = {}
        # Treat the first four weights in a special way:
        if 'xmin' not in all_weights[0] or \
           'xmax' not in all_weights[1] or \
           'Weight' not in all_weights[2] or \
           'WeightError' not in all_weights[3]:
            raise MadGraph5Error('The first weight entries in the XML HwU '+\
              ' source are not the standard expected ones  (xmin, xmax, sigmaCentral, errorCentral)')
        selected_weights[0] = ['xmin']
        selected_weights[1] = ['xmax']

# ===========  BEGIN HELPER FUNCTIONS ===========
        def get_difference_to_central(weight):
            """ Return the list of properties which differ from the central weight.
            This disregards the merging scale value for which any central value
            can be picked anyway."""
            
            differences = []
            # If the tag 'Weight' is in the weight label, then this is 
            # automatically considered as the Event weight (central) for which
            # only the merging scale can be different
            if 'Weight' in weight:
                return set([])
            if weight['MUR'] not in [central_MUR, -1.0] or \
               weight['MUF'] not in [central_MUF, -1.0]:
                differences.append('mur_muf_scale')
            if weight['PDF'] not in [central_PDF,-1]:
                differences.append('pdf')
            if weight['ALPSFACT'] not in [central_alpsfact, -1]:
                differences.append('ALPSFACT')
            return set(differences) 

        def format_weight_label(weight):
            """ Print the weight attributes in a nice order."""
            
            all_properties = list(weight.keys())
            all_properties.pop(all_properties.index('POSITION'))
            ordered_properties = []
            # First add the attributes without value
            for property in all_properties:
                if weight[property] is None:
                    ordered_properties.append(property)
            
            ordered_properties.sort()
            all_properties = [property for property in all_properties if 
                                                   not weight[property] is None]
            
            # then add PDF, MUR, MUF and MERGING if present
            for property in ['PDF','MUR','MUF','ALPSFACT','MERGING']:
                all_properties.pop(all_properties.index(property))
                if weight[property]!=-1:
                    ordered_properties.append(property)

            ordered_properties.extend(sorted(all_properties))
            
            return '_'.join('%s%s'\
                    %(key,'' if weight[key] is None else '=%s'%str(weight[key])) for 
                                                      key in ordered_properties)
# ===========  END HELPER FUNCTIONS ===========
        
        
        # The central value is not necessarily the 3rd one if a different merging
        # cut was selected.
        if float(all_weights[2]['MERGING']) == merging_scale_chosen:
            selected_weights[2]=['central value']
        else:
            for weight_position, weight in enumerate(all_weights):
                # Check if that weight corresponds to a central weight 
                # (conventional label for central weight is 'Weight'
                if get_difference_to_central(weight)==set([]):
                    # Check if the merging scale matches this time
                    if weight['MERGING']==merging_scale_chosen:
                        selected_weights[weight_position] = ['central value']
                        break
            # Make sure a central value was found, throw a warning if found
            if 'central value' not in sum(list(selected_weights.values()),[]):
                central_merging_scale = all_weights[2]['MERGING']
                logger.warning('Could not find the central weight for the'+\
                ' chosen merging scale (%f).\n'%merging_scale_chosen+\
                'MG5aMC will chose the original central scale provided which '+\
                'correspond to a merging scale of %s'%("'inclusive'" if 
                   central_merging_scale in [0.0,-1.0] else '%f'%central_merging_scale))
                selected_weights[2]=['central value']
    
        # The error is always the third entry for now.
        selected_weights[3]=['dy']

        # Now process all other weights  
        for weight_position, weight in enumerate(all_weights[4:]):
            # Apply special transformation for the weight label:
            # scale variation are stored as:
            #   ('scale', mu_r, mu_f)    for  scale variation
            #   ('pdf',PDF)              for PDF variation 
            #   ('merging_scale',float)  for merging scale
            #   ('type',value)           for all others (e.g. alpsfact)
            variations = get_difference_to_central(weight)            
            # We know select the 'diagonal' variations where each parameter
            # is varied one at a time.
            
            # Accept also if both pdf and mur_muf_scale differ because
            # the PDF used for the Event weight is often unknown but the
            # mu_r and mu_f variational weight specify it. Same story for
            # alpsfact.
            wgt_label = None
            if variations in [set(['mur_muf_scale']),set(['pdf','mur_muf_scale'])]:
                wgt_label = ('scale',weight['MUR'],weight['MUF'])
            elif variations in [set(['ALPSFACT']),set(['pdf','ALPSFACT'])]:
                wgt_label = ('alpsfact',weight['ALPSFACT'])
            elif variations == set(['pdf']):
                wgt_label = ('pdf',weight['PDF'])             
            elif variations == set([]):
                # Unknown weight (might turn out to be taken as a merging variation weight below)
                wgt_label = format_weight_label(weight)

            # Compound variations are deliberately not part of the diagonal
            # uncertainty sets.  Previously they accidentally reused the label
            # from the preceding weight and could ultimately try to hash the
            # weight dictionary itself.
            if wgt_label is None:
                continue

            # Make sure the merging scale matches the chosen one
            if weight['MERGING'] != merging_scale_chosen:
                # If a merging_scale was specified, then ignore all other weights
                if merging_scale:
                    continue
                # Otherwise consider them also, but for now only if it is for
                # the central value parameter (central PDF, central mu_R and mu_F)
                if variations == set([]):
                    # We choose to store the merging variation weight labels as floats
                    wgt_label = ('merging_scale', weight['MERGING'])
            # Make sure that the weight label does not already exist. If it does,
            # this means that the source has redundant information and that
            # there is no need to specify it again.
            if wgt_label in sum(list(selected_weights.values()),[]):
                continue

            # Now register the selected weight
            try:
                selected_weights[weight_position+4].append(wgt_label)
            except KeyError:
                selected_weights[weight_position+4]=[wgt_label,]
        
        if merging_scale and merging_scale > 0.0 and \
                                       len(sum(list(selected_weights.values()),[]))==4:
            logger.warning('No additional variation weight was found for the '+\
                                       'chosen merging scale %f.'%merging_scale)

        # Make sure to use the predefined keywords for the mandatory weight labels
        for wgt_pos in selected_weights:
            for i, weight_label in enumerate(selected_weights[wgt_pos]):
                try:
                    selected_weights[wgt_pos][i] = HwU.mandatory_weights[weight_label]
                except KeyError:
                    pass

        # Keep only the weights asked for
        if consider_reweights!='ALL':
            new_selected_weights = {}
            for wgt_position, wgt_labels in selected_weights.items():
                for wgt_label in wgt_labels:
                    if wgt_label in ['central','stat_error','boundary_xmin','boundary_xmax'] or\
                       HwU.get_HwU_wgt_label_type(wgt_label) in consider_reweights:
                        try:
                            new_selected_weights[wgt_position].append(wgt_label)
                        except KeyError:
                            new_selected_weights[wgt_position] = [wgt_label]
            selected_weights = new_selected_weights                         

        # Cache the list of selected weights to be defined at each line
        weight_label_list = sum(list(selected_weights.values()),[])

        # The weight_label list to set to self.bins 
        ordered_weight_label_list = ['central','stat_error']
        for weight_label in weight_label_list:
            if not isinstance(weight_label, str):
                ordered_weight_label_list.append(weight_label)
        for weight_label in weight_label_list:
            if weight_label in ['central','stat_error','boundary_xmin','boundary_xmax']:
                continue
            if isinstance(weight_label, str):
                ordered_weight_label_list.append(weight_label)
       
        # Now that we know the desired weights, retrieve all plots from the
        # XML source node.
        return self.retrieve_plots_from_XML_source(selected_run_node,
                  selected_weights, ordered_weight_label_list, raw_labels=False)

    def retrieve_plots_from_XML_source(self, xml_node,
                  selected_weights, ordered_weight_label_list,raw_labels=False):
        """Given an XML node and the selected weights and their ordered list,
        import all histograms from the specified XML node."""

        # We now start scanning all the plots
        for multiplicity_node in xml_node.getElementsByTagName("jethistograms"):
            multiplicity = int(multiplicity_node.getAttribute('njet'))
            for histogram in multiplicity_node.getElementsByTagName("histogram"):
                # We only consider the histograms with all the weight information
                if histogram.getAttribute("weight")!='all':
                    continue
                new_histo = HwU()
                hist_name = '%s %s'%(str(histogram.getAttribute('name')),
                                            str(histogram.getAttribute('unit')))
                # prepend the jet multiplicity to the histogram name
                new_histo.process_histogram_name('%s |JETSAMPLE@%d'%(hist_name,multiplicity))
                # We do not want to include auxiliary diagrams which would be
                # recreated anyway.
                if new_histo.type == 'AUX':
                     continue
                # Make sure to exclude the boundaries from the weight
                # specification
                # Order the weights so that the unreckognized ones go last
                new_histo.bins = BinList(weight_labels = ordered_weight_label_list)
                hist_data = str(histogram.childNodes[0].data)
                for line in hist_data.split('\n'):
                    if line.strip()=='':
                        continue
                    bin_weights = {}
                    boundaries = [0.0,0.0]
                    for j, weight in \
                              enumerate(HwU.histo_bin_weight_re.finditer(line)):
                        try:
                            for wgt_label in selected_weights[j]:
                                if wgt_label == 'boundary_xmin':
                                    boundaries[0] = float(weight.group('weight'))
                                elif wgt_label == 'boundary_xmax':
                                    boundaries[1] = float(weight.group('weight'))                            
                                else:
                                    if weight.group('weight').upper()=='NAN':
                                        raise MadGraph5Error("Some weights are found to be 'NAN' in histogram with name '%s'"%hist_name+\
    " and jet sample multiplicity %d."%multiplicity)
                                    else:
                                        bin_weights[wgt_label] = \
                                                   float(weight.group('weight'))
                        except KeyError:
                            continue
                    # For this check, we subtract two because of the bin boundaries
                    if len(bin_weights)!=len(ordered_weight_label_list):
                        raise MadGraph5Error('Not all defined weights were found in the XML source.\n'+\
                         '%d found / %d expected.'%(len(bin_weights),len(ordered_weight_label_list))+\
                         '\nThe missing ones are: %s.'%\
                         str(list(set(ordered_weight_label_list)-set(bin_weights.keys())))+\
                         "\nIn plot with title '%s' and jet sample multiplicity %d."%\
                                                       (hist_name, multiplicity))
            
                    new_histo.bins.append(Bin(tuple(boundaries), bin_weights))

#                    if bin_weights['central']!=0.0:
#                          print '---------'
#                          print 'multiplicity =',multiplicity
#                          print 'central =', bin_weights['central']
#                          print 'PDF     = ', [(key,bin_weights[key]) for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='pdf']
#                          print 'PDF min/max =',min(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='pdf'),max(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='pdf')
#                          print 'scale   = ', [(key,bin_weights[key]) for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='scale']
#                          print 'scale min/max =',min(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='scale'),max(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='scale')
#                          print 'merging = ', [(key,bin_weights[key]) for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='merging_scale'] 
#                          print 'merging min/max =',min(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='merging_scale'),max(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='merging_scale')
#                          print 'alpsfact = ', [(key,bin_weights[key]) for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='alpsfact'] 
#                          print 'alpsfact min/max =',min(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='alpsfact'),max(bin_weights[key] for key in bin_weights if HwU.get_HwU_wgt_label_type(key)=='alpsfact')
#                          print '---------'
#                          stop

                # Finally remove auxiliary weights
                if not raw_labels:
                    new_histo.trim_auxiliary_weights()
                
                # And add it to the list
                self.append(new_histo)
        
    def output(self, path, format='gnuplot',number_of_ratios = -1, 
          uncertainties=['scale','pdf','statistical','merging_scale','alpsfact'],
          use_band = None,
          ratio_correlations=True, arg_string='', 
          jet_samples_to_keep=None,
          auto_open=True,
          lhapdfconfig='lhapdf-config',
          assigned_colours=None,
          _histogram_groups=None,
          _weight_schema=None):
        """ Ouput this histogram to a file, stream or string if path is kept to
        None. The supported format are for now. Chose whether to print the header
        or not."""
        
        if len(self)==0 and _histogram_groups is None:
            raise MadGraph5Error('No histograms stored in the list yet.')
        
        if not format in HwU.output_formats_implemented:
            raise MadGraph5Error("The specified output format '%s'"%format+\
                             " is not yet supported. Supported formats are %s."\
                                                 %HwU.output_formats_implemented)

        if not isinstance(path, str) or not os.path.basename(path) or \
                         os.path.basename(path).lower().endswith(
                    ('.hwu', '.ps', '.gnuplot', '.pdf', '.py', '.html')):
            raise MadGraph5Error("The path argument of the output function of"+\
              " the HwUList instance must be file path without its extension.")

        if _histogram_groups is None:
            weight_schema = self._common_weight_schema(self)
        else:
            if _weight_schema is None:
                raise MadGraph5Error('A weight schema is required for streamed '
                                     'histogram output.')
            weight_schema = list(_weight_schema)
            if len(weight_schema) != len(set(weight_schema)):
                raise MadGraph5Error('Histogram weight columns must be unique.')

        output_base_name = os.path.basename(path)
        hwu_target = _AtomicTextOutput(path+'.HwU')
        HwU_stream = hwu_target.stream
        HwU_output_list = _LineStreamSink(HwU_stream)
        # If the format is just the raw HwU source, then simply write them
        # out all in sequence.
        if format == 'HwU':
            groups = _histogram_groups if _histogram_groups is not None \
                                           else [self]
            histogram_index = 0
            try:
                for group in groups:
                    for histo in group:
                        HwU_output_list.extend(histo.iter_HwU_source(
                            print_header=(histogram_index == 0),
                            weight_labels=weight_schema))
                        HwU_output_list.extend(['',''])
                        histogram_index += 1
                if histogram_index == 0:
                    raise MadGraph5Error('No histograms were provided for output.')
                hwu_target.commit()
            except BaseException:
                hwu_target.abort()
                raise
            return

        if format == 'matplotlib':
            script_target = _AtomicTextOutput(path+'.py')
            try:
                self._output_matplotlib(path, output_base_name, HwU_stream,
                    script_target.stream, weight_schema,
                    number_of_ratios=number_of_ratios,
                    uncertainties=uncertainties,
                    use_band=use_band,
                    ratio_correlations=ratio_correlations,
                    arg_string=arg_string,
                    jet_samples_to_keep=jet_samples_to_keep,
                    auto_open=auto_open,
                    lhapdfconfig=lhapdfconfig,
                    assigned_colours=assigned_colours,
                    histogram_groups=_histogram_groups)
                hwu_target.commit()
                script_target.commit()
            except BaseException:
                hwu_target.abort()
                script_target.abort()
                raise
            return

        if format == 'html':
            html_target = _AtomicTextOutput(path+'.html')
            try:
                self._output_html(output_base_name, HwU_stream,
                    html_target.stream, weight_schema,
                    number_of_ratios=number_of_ratios,
                    uncertainties=uncertainties,
                    use_band=use_band,
                    ratio_correlations=ratio_correlations,
                    arg_string=arg_string,
                    jet_samples_to_keep=jet_samples_to_keep,
                    lhapdfconfig=lhapdfconfig,
                    assigned_colours=assigned_colours,
                    histogram_groups=_histogram_groups)
                hwu_target.commit()
                html_target.commit()
            except BaseException:
                hwu_target.abort()
                html_target.abort()
                raise
            return
        
        # Now we consider that we are attempting a gnuplot output.
        if format == 'gnuplot':
            gnuplot_target = _AtomicTextOutput(path+'.gnuplot')
            gnuplot_stream = gnuplot_target.stream
            # Keep Matplotlib and direct HTML renderers beside the default
            # GnuPlot card. All renderers use the same prepared HwU data, so
            # streamed histogram groups still only need to be visited once.
            script_target = _AtomicTextOutput(path+'.py')
            html_target = _AtomicTextOutput(path+'.html')
            html_writer = _HTMLHistogramWriter(
                         html_target.stream, output_base_name, arg_string)

        # Group compatible curves without changing the source list.
        matching_histo_lists = _histogram_groups if \
            _histogram_groups is not None else self._group_for_output(self)

        # the histogram colours:
        coli=['col1','col2','col3','col4','col5','col6','col7','col8']
        colours={coli[0] : "#009e73",
                 coli[1] : "#0072b2",
                 coli[2] : "#d55e00",
                 coli[3] : "#f0e442",
                 coli[4] : "#56b4e9",
                 coli[5] : "#cc79a7",
                 coli[6] : "#e69f00",
                 coli[7] : "black"}
        if assigned_colours:
            for index, item in enumerate(assigned_colours[:len(coli)]):
                if (item != None): colours[coli[index]]=item
        matplotlib_colours = [colours[colour] for colour in coli]

        replace_dict=colours
        replace_dict['arg_string']=arg_string
        replace_dict['output_base_name']=output_base_name

        # Write the gnuplot header
        gnuplot_output_list_v4 = [
"""
################################################################################
#
# This gnuplot file was generated by MadGraph5_aMC@NLO project, a program which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond. It also perform the
# integration and/or generate events for these processes, at LO and NLO accuracy.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
# %(arg_string)s
reset

set lmargin 10
set rmargin 0
set terminal postscript portrait enhanced mono dashed lw 1.0 "Helvetica" 9 
# The pdf terminal offers transparency support, but you will have to adapt things a bit
#set terminal pdf enhanced font "Helvetica 12" lw 1.0 dashed size 29.7cm, 21cm
set key font ",9"
set key samplen "2"
set output "%(output_base_name)s.ps"

# This is the "PODO" color palette of gnuplot v.5, but with the order
# changed: palette of colors selected to be easily distinguishable by
# color-blind individuals with either protanopia or deuteranopia. Bang
# Wong [2011] Nature Methods 8, 441.

set style line  1 lt 1 lc rgb "%(col1)s" lw 2.5
set style line 11 lt 2 lc rgb "%(col1)s" lw 2.5
set style line 21 lt 4 lc rgb "%(col1)s" lw 2.5
set style line 31 lt 6 lc rgb "%(col1)s" lw 2.5
set style line 41 lt 8 lc rgb "%(col1)s" lw 2.5

set style line  2 lt 1 lc rgb "%(col2)s" lw 2.5
set style line 12 lt 2 lc rgb "%(col2)s" lw 2.5
set style line 22 lt 4 lc rgb "%(col2)s" lw 2.5
set style line 32 lt 6 lc rgb "%(col2)s" lw 2.5
set style line 42 lt 8 lc rgb "%(col2)s" lw 2.5

set style line  3 lt 1 lc rgb "%(col3)s" lw 2.5
set style line 13 lt 2 lc rgb "%(col3)s" lw 2.5
set style line 23 lt 4 lc rgb "%(col3)s" lw 2.5
set style line 33 lt 6 lc rgb "%(col3)s" lw 2.5
set style line 43 lt 8 lc rgb "%(col3)s" lw 2.5

set style line  4 lt 1 lc rgb "%(col4)s" lw 2.5
set style line 14 lt 2 lc rgb "%(col4)s" lw 2.5
set style line 24 lt 4 lc rgb "%(col4)s" lw 2.5
set style line 34 lt 6 lc rgb "%(col4)s" lw 2.5
set style line 44 lt 8 lc rgb "%(col4)s" lw 2.5

set style line  5 lt 1 lc rgb "%(col5)s" lw 2.5
set style line 15 lt 2 lc rgb "%(col5)s" lw 2.5
set style line 25 lt 4 lc rgb "%(col5)s" lw 2.5
set style line 35 lt 6 lc rgb "%(col5)s" lw 2.5
set style line 45 lt 8 lc rgb "%(col5)s" lw 2.5

set style line  6 lt 1 lc rgb "%(col6)s" lw 2.5
set style line 16 lt 2 lc rgb "%(col6)s" lw 2.5
set style line 26 lt 4 lc rgb "%(col6)s" lw 2.5
set style line 36 lt 6 lc rgb "%(col6)s" lw 2.5
set style line 46 lt 8 lc rgb "%(col6)s" lw 2.5

set style line  7 lt 1 lc rgb "%(col7)s" lw 2.5
set style line 17 lt 2 lc rgb "%(col7)s" lw 2.5
set style line 27 lt 4 lc rgb "%(col7)s" lw 2.5
set style line 37 lt 6 lc rgb "%(col7)s" lw 2.5
set style line 47 lt 8 lc rgb "%(col7)s" lw 2.5

set style line  8 lt 1 lc rgb "%(col8)s" lw 2.5
set style line 18 lt 2 lc rgb "%(col8)s" lw 2.5
set style line 28 lt 4 lc rgb "%(col8)s" lw 2.5
set style line 38 lt 6 lc rgb "%(col8)s" lw 2.5
set style line 48 lt 7 lc rgb "%(col8)s" lw 2.5


set style line 999 lt 1 lc rgb "gray" lw 2.5

safe(x,y,a) = (y == 0.0 ? a : x/y)

set style data histeps
set key invert

"""%(replace_dict)
]
        
        gnuplot_output_list_v5 = [
"""
################################################################################
#
# This gnuplot file was generated by MadGraph5_aMC@NLO project, a program which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond. It also perform the
# integration and/or generate events for these processes, at LO and NLO accuracy.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
# %(arg_string)s
reset

set lmargin 10
set rmargin 0
set terminal postscript portrait enhanced color "Helvetica" 9 
# The pdf terminal offers transparency support, but you will have to adapt things a bit
#set terminal pdf enhanced font "Helvetica 12" lw 1.0 dashed size 29.7cm, 21cm
set key font ",9"
set key samplen "2"
set output "%(output_base_name)s.ps"

# This is the "PODO" color palette of gnuplot v.5, but with the order
# changed: palette of colors selected to be easily distinguishable by
# color-blind individuals with either protanopia or deuteranopia. Bang
# Wong [2011] Nature Methods 8, 441.

set style line   1 lt 1 lc rgb "%(col1)s" lw 1.3
set style line 101 lt 1 lc rgb "%(col1)s" lw 1.3 dt (6,3)
set style line  11 lt 2 lc rgb "%(col1)s" lw 1.3 dt (6,3)
set style line  21 lt 4 lc rgb "%(col1)s" lw 1.3 dt (3,2)
set style line  31 lt 6 lc rgb "%(col1)s" lw 1.3 dt (2,1)
set style line  41 lt 8 lc rgb "%(col1)s" lw 1.3 dt (4,3)

set style line   2 lt 1 lc rgb "%(col2)s" lw 1.3
set style line 102 lt 1 lc rgb "%(col2)s" lw 1.3 dt (6,3)
set style line  12 lt 2 lc rgb "%(col2)s" lw 1.3 dt (6,3)
set style line  22 lt 4 lc rgb "%(col2)s" lw 1.3 dt (3,2)
set style line  32 lt 6 lc rgb "%(col2)s" lw 1.3 dt (2,1)
set style line  42 lt 8 lc rgb "%(col2)s" lw 1.3 dt (4,3)

set style line   3 lt 1 lc rgb "%(col3)s" lw 1.3
set style line 103 lt 1 lc rgb "%(col3)s" lw 1.3 dt (6,3)
set style line  13 lt 2 lc rgb "%(col3)s" lw 1.3 dt (6,3)
set style line  23 lt 4 lc rgb "%(col3)s" lw 1.3 dt (3,2)
set style line  33 lt 6 lc rgb "%(col3)s" lw 1.3 dt (2,1)
set style line  43 lt 8 lc rgb "%(col3)s" lw 1.3 dt (4,3)

set style line   4 lt 1 lc rgb "%(col4)s" lw 1.3
set style line 104 lt 1 lc rgb "%(col4)s" lw 1.3 dt (6,3)
set style line  14 lt 2 lc rgb "%(col4)s" lw 1.3 dt (6,3)
set style line  24 lt 4 lc rgb "%(col4)s" lw 1.3 dt (3,2)
set style line  34 lt 6 lc rgb "%(col4)s" lw 1.3 dt (2,1)
set style line  44 lt 8 lc rgb "%(col4)s" lw 1.3 dt (4,3)

set style line   5 lt 1 lc rgb "%(col5)s" lw 1.3
set style line 105 lt 1 lc rgb "%(col5)s" lw 1.3 dt (6,3)
set style line  15 lt 2 lc rgb "%(col5)s" lw 1.3 dt (6,3)
set style line  25 lt 4 lc rgb "%(col5)s" lw 1.3 dt (3,2)
set style line  35 lt 6 lc rgb "%(col5)s" lw 1.3 dt (2,1)
set style line  45 lt 8 lc rgb "%(col5)s" lw 1.3 dt (4,3)

set style line   6 lt 1 lc rgb "%(col6)s" lw 1.3
set style line 106 lt 1 lc rgb "%(col6)s" lw 1.3 dt (6,3)
set style line  16 lt 2 lc rgb "%(col6)s" lw 1.3 dt (6,3)
set style line  26 lt 4 lc rgb "%(col6)s" lw 1.3 dt (3,2)
set style line  36 lt 6 lc rgb "%(col6)s" lw 1.3 dt (2,1)
set style line  46 lt 8 lc rgb "%(col6)s" lw 1.3 dt (4,3)

set style line   7 lt 1 lc rgb "%(col7)s" lw 1.3
set style line 107 lt 1 lc rgb "%(col7)s" lw 1.3 dt (6,3)
set style line  17 lt 2 lc rgb "%(col7)s" lw 1.3 dt (6,3)
set style line  27 lt 4 lc rgb "%(col7)s" lw 1.3 dt (3,2)
set style line  37 lt 6 lc rgb "%(col7)s" lw 1.3 dt (2,1)
set style line  47 lt 8 lc rgb "%(col7)s" lw 1.3 dt (4,3)

set style line   8 lt 1 lc rgb "%(col8)s" lw 1.3
set style line 108 lt 1 lc rgb "%(col8)s" lw 1.3 dt (6,3)
set style line  18 lt 2 lc rgb "%(col8)s" lw 1.3 dt (6,3)
set style line  28 lt 4 lc rgb "%(col8)s" lw 1.3 dt (3,2)
set style line  38 lt 6 lc rgb "%(col8)s" lw 1.3 dt (2,1)
set style line  48 lt 8 lc rgb "%(col8)s" lw 1.3 dt (4,3)


set style line 999 lt 1 lc rgb "gray" lw 1.3

safe(x,y,a) = (y == 0.0 ? a : x/y)

set style data histeps
set key invert

"""%(replace_dict)
]
        
        # determine the gnuplot version
        try:
            p = subprocess.Popen(['gnuplot', '--version'], \
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            # assume that version 4 of gnuplot is the default if
            # gnuplot could not be found
            gnuplot_output_list=gnuplot_output_list_v5
        else:
            output, _ = p.communicate()
            output = output.decode(errors='ignore')
            if not output:
                gnuplot_output_list=gnuplot_output_list_v5
            elif int(output.split()[1].split('.')[0]) < 5 :
                gnuplot_output_list=gnuplot_output_list_v4
            else:
                gnuplot_output_list=gnuplot_output_list_v5


        # Now output each group one by one
        # Block position keeps track of the gnuplot data_block index considered
        block_position = 0
        plot_specs = []
        ephemeral_groups = _histogram_groups is not None
        histogram_output = _TeeLineSink(HwU_output_list,
                                         html_writer.data_sink)
        output_schema_state = {'base_labels': list(weight_schema),
                               'labels': None}
        for histo_group in matching_histo_lists:
            # Plot preparation adds ratios and auxiliary uncertainty columns.
            # Retain the prepared group just long enough to describe the
            # matching standalone Matplotlib plot.
            prepared_group = histo_group if ephemeral_groups \
                              else copy.deepcopy(histo_group)
            first_block = block_position
            block_position = prepared_group.output_group(histogram_output,
                    gnuplot_output_list, block_position,output_base_name+'.HwU',
                    number_of_ratios=number_of_ratios, 
                    uncertainties = uncertainties,
                    use_band = use_band,
                    ratio_correlations = ratio_correlations,
                    jet_samples_to_keep=jet_samples_to_keep,
                    lhapdfconfig = lhapdfconfig,
                    _copy_group=False,
                    _output_schema_state=output_schema_state)
            plot_specs.append(self._get_matplotlib_plot_spec(
                prepared_group, first_block, uncertainties, use_band,
                jet_samples_to_keep, matplotlib_colours))

        if block_position == 0:
            raise MadGraph5Error('No histograms were provided for output.')

        # Now write the tail of the gnuplot command file
        gnuplot_output_list.extend([
          "unset multiplot",
          '!ps2pdf "%s.ps" &> /dev/null'%output_base_name])
        if auto_open:
            gnuplot_output_list.append(
                                 '!open "%s.pdf" &> /dev/null'%output_base_name)
        
        # Replace all final files only after all groups were prepared.
        gnuplot_stream.write('\n'.join(gnuplot_output_list))
        script_target.stream.write(self._get_matplotlib_script(
                         output_base_name, plot_specs, auto_open, arg_string))
        html_writer.finish(plot_specs)
        hwu_target.commit()
        gnuplot_target.commit()
        script_target.commit()
        html_target.commit()

        logger.debug("Histograms have been written out at "+\
            "%s.[HwU|gnuplot|py|html]' and can be viewed with GnuPlot, "%\
            output_base_name+"Matplotlib, or a web browser.")

    def _output_matplotlib(self, path, output_base_name, hwu_stream,
          script_stream, weight_schema,
          number_of_ratios=-1,
          uncertainties=['scale','pdf','statistical','merging_scale','alpsfact'],
          use_band=None,
          ratio_correlations=True,
          arg_string='',
          jet_samples_to_keep=None,
          auto_open=True,
          lhapdfconfig='lhapdf-config',
          assigned_colours=None,
          histogram_groups=None):
        """Write HwU data and a standalone Matplotlib rendering script."""

        ephemeral_groups = histogram_groups is not None
        matching_histo_lists = histogram_groups if histogram_groups is not None \
                                           else self._group_for_output(self)

        colours = ['#009e73','#0072b2','#d55e00','#f0e442',
                   '#56b4e9','#cc79a7','#e69f00','black']
        if assigned_colours:
            for index, colour in enumerate(assigned_colours[:len(colours)]):
                if colour is not None:
                    colours[index] = colour

        hwu_output = _LineStreamSink(hwu_stream)
        plot_specs = []
        block_position = 0
        output_schema_state = {'base_labels': list(weight_schema),
                               'labels': None}
        for histo_group in matching_histo_lists:
            # Plot preparation adds ratios and auxiliary uncertainty columns.
            # Copy one group only, keeping completed groups out of memory.
            prepared_group = histo_group if ephemeral_groups \
                              else copy.deepcopy(histo_group)
            first_block = block_position
            block_position = prepared_group.output_group(
                hwu_output, [], block_position, output_base_name+'.HwU',
                number_of_ratios=number_of_ratios,
                uncertainties=uncertainties,
                use_band=use_band,
                ratio_correlations=ratio_correlations,
                jet_samples_to_keep=jet_samples_to_keep,
                lhapdfconfig=lhapdfconfig,
                _copy_group=False,
                _output_schema_state=output_schema_state)
            plot_specs.append(self._get_matplotlib_plot_spec(
                prepared_group, first_block, uncertainties, use_band,
                jet_samples_to_keep, colours))

        if block_position == 0:
            raise MadGraph5Error('No histograms were provided for output.')

        script = self._get_matplotlib_script(output_base_name, plot_specs,
                                               auto_open, arg_string)
        script_stream.write(script)

        logger.debug("Histograms have been written out at '%s.[HwU|py]' and can "
                     "now be rendered by invoking Python."%output_base_name)

    def _output_html(self, output_base_name, hwu_stream, html_stream,
          weight_schema,
          number_of_ratios=-1,
          uncertainties=['scale','pdf','statistical','merging_scale','alpsfact'],
          use_band=None,
          ratio_correlations=True,
          arg_string='',
          jet_samples_to_keep=None,
          lhapdfconfig='lhapdf-config',
          assigned_colours=None,
          histogram_groups=None):
        """Write HwU data and a self-contained native HTML/SVG page."""

        ephemeral_groups = histogram_groups is not None
        matching_histo_lists = histogram_groups if histogram_groups is not None \
                                           else self._group_for_output(self)

        colours = ['#009e73','#0072b2','#d55e00','#f0e442',
                   '#56b4e9','#cc79a7','#e69f00','black']
        if assigned_colours:
            for index, colour in enumerate(assigned_colours[:len(colours)]):
                if colour is not None:
                    colours[index] = colour

        html_writer = _HTMLHistogramWriter(
                              html_stream, output_base_name, arg_string)
        hwu_output = _TeeLineSink(_LineStreamSink(hwu_stream),
                                  html_writer.data_sink)
        plot_specs = []
        block_position = 0
        output_schema_state = {'base_labels': list(weight_schema),
                               'labels': None}
        for histo_group in matching_histo_lists:
            prepared_group = histo_group if ephemeral_groups \
                              else copy.deepcopy(histo_group)
            first_block = block_position
            block_position = prepared_group.output_group(
                hwu_output, [], block_position, output_base_name+'.HwU',
                number_of_ratios=number_of_ratios,
                uncertainties=uncertainties,
                use_band=use_band,
                ratio_correlations=ratio_correlations,
                jet_samples_to_keep=jet_samples_to_keep,
                lhapdfconfig=lhapdfconfig,
                _copy_group=False,
                _output_schema_state=output_schema_state)
            plot_specs.append(self._get_matplotlib_plot_spec(
                prepared_group, first_block, uncertainties, use_band,
                jet_samples_to_keep, colours))

        if block_position == 0:
            raise MadGraph5Error('No histograms were provided for output.')

        html_writer.finish(plot_specs)
        logger.debug("Histograms have been written out at '%s.[HwU|html]' and "
                     "can be viewed in a web browser."%output_base_name)

    @staticmethod
    def _get_matplotlib_curve_label(histo, index):
        """Return the legend label for a central Matplotlib curve."""

        label = []
        if histo.type:
            label.append('NLO' if histo.type.split()[0]=='NLO' else histo.type)
        elif not hasattr(histo, 'jetsample'):
            label.append('central value (%d)'%(index+1))
        if hasattr(histo, 'jetsample'):
            if histo.jetsample == -1:
                label.append('all jet samples')
            else:
                label.append('jet sample %d'%histo.jetsample)
        return ', '.join(label) if label else 'central value'

    @staticmethod
    def _get_matplotlib_uncertainties(histo, uncertainties):
        """Map generated auxiliary uncertainty weights to HwU columns."""

        uncertainty_names = [
            ('scale', 'delta_mu', 'scale'),
            ('pdf', 'delta_pdf', 'PDF'),
            ('merging_scale', 'delta_merging', 'merging scale'),
            ('alpsfact', 'delta_alpsfact', 'alpsfact')]
        label_positions = dict((label, index+2) for index, label in
                                      enumerate(histo.bins.weight_labels))
        result = []
        for uncertainty_type, prefix, display_name in uncertainty_names:
            if uncertainty_type not in uncertainties:
                continue
            central_prefix = prefix+'_cen'
            for label in histo.bins.weight_labels:
                if not isinstance(label, str) or not label.endswith(' @aux'):
                    continue
                if label == central_prefix+' @aux':
                    suffix = ''
                elif label.startswith(central_prefix+' '):
                    suffix = label[len(central_prefix)+1:-len(' @aux')]
                else:
                    continue
                suffix_part = (' '+suffix) if suffix else ''
                minimum_label = prefix+'_min'+suffix_part+' @aux'
                maximum_label = prefix+'_max'+suffix_part+' @aux'
                if minimum_label not in label_positions or \
                                           maximum_label not in label_positions:
                    continue
                label_name = display_name
                if suffix and suffix != 'none':
                    label_name += ' (%s)'%suffix
                result.append({
                    'type': uncertainty_type,
                    'label': label_name,
                    'central': label_positions[label],
                    'minimum': label_positions[minimum_label],
                    'maximum': label_positions[maximum_label]})
        return result

    @classmethod
    def _get_matplotlib_plot_spec(cls, histo_group, first_block,
          uncertainties, use_band, jet_samples_to_keep, colours):
        """Build the serializable instructions used by the generated script."""

        main_histograms = [histo for histo in histo_group
                                                if histo.type != 'AUX']
        ratio_histograms = [histo for histo in histo_group
                                                if histo.type == 'AUX']
        main_curves = []
        available_uncertainties = []
        for index, histo in enumerate(main_histograms):
            show_uncertainties = not (
                index != 0 and hasattr(histo, 'jetsample') and
                histo.jetsample != -1 and not (
                    jet_samples_to_keep and len(jet_samples_to_keep)==1 and
                    jet_samples_to_keep[0] == histo.jetsample))
            curve_uncertainties = cls._get_matplotlib_uncertainties(
                                                    histo, uncertainties)
            if not show_uncertainties:
                curve_uncertainties = []
            for uncertainty in curve_uncertainties:
                if uncertainty['type'] not in available_uncertainties:
                    available_uncertainties.append(uncertainty['type'])
            main_curves.append({
                'block': first_block+index,
                'label': cls._get_matplotlib_curve_label(histo, index),
                'color': colours[index%len(colours)],
                'statistical': 'statistical' in uncertainties,
                'uncertainties': curve_uncertainties})

        if use_band is None:
            if len(available_uncertainties) <= 1:
                selected_bands = list(available_uncertainties)
            elif 'scale' in available_uncertainties:
                selected_bands = ['scale']
            else:
                selected_bands = [available_uncertainties[0]]
        else:
            selected_bands = list(use_band)

        ratio_curves = []
        for index, histo in enumerate(ratio_histograms):
            ratio_label = histo.title.strip() or 'ratio %d'%(index+1)
            if ratio_label.endswith('1/K-factor'):
                ratio_label = '1/K-factor'
            elif ratio_label.endswith('K-factor'):
                ratio_label = 'K-factor'
            ratio_curves.append({
                'block': first_block+len(main_histograms)+index,
                'label': ratio_label,
                'color': colours[(len(main_histograms)+index)%len(colours)],
                'statistical': 'statistical' in uncertainties,
                'uncertainties': cls._get_matplotlib_uncertainties(
                                                    histo, uncertainties)})

        for curve in main_curves+ratio_curves:
            for uncertainty in curve['uncertainties']:
                uncertainty['band'] = uncertainty['type'] in selected_bands

        return {
            'title': histo_group[0].get_HwU_histogram_name(
                                                    format='human-no_type'),
            'x_axis_mode': histo_group[0].x_axis_mode,
            'y_axis_mode': histo_group[0].y_axis_mode,
            'main': main_curves,
            'ratios': ratio_curves,
            'relative_uncertainties': bool(available_uncertainties or
                                           'statistical' in uncertainties)}

    @staticmethod
    def _get_matplotlib_script(output_base_name, plot_specs, auto_open,
                                                                  arg_string):
        """Return the standalone Python source for Matplotlib rendering."""

        arg_string = arg_string.replace('\r', ' ').replace('\n', ' ')
        header = """#!/usr/bin/env python3
# Matplotlib histogram renderer generated by MadGraph5_aMC@NLO.
# Original command: %s
import os
import math
import subprocess
import sys

DATA_FILE = %r
PDF_FILE = %r
OPEN_AFTER_RENDER = %r
PLOTS = %s

"""%(arg_string, output_base_name+'.HwU', output_base_name+'.pdf',
       auto_open, pprint.pformat(plot_specs, width=100))
        return header+_MATPLOTLIB_SCRIPT_BODY

    def output_group(self, HwU_out, gnuplot_out, block_position, HwU_name,
          number_of_ratios = -1, 
          uncertainties = ['scale','pdf','statistical','merging_scale','alpsfact'],
          use_band = None,
          ratio_correlations = True, 
          jet_samples_to_keep=None,
          lhapdfconfig='lhapdf-config',
          _copy_group=True,
          _output_schema_state=None):
        
        """ This functions output a single group of histograms with either one
        histograms untyped (i.e. type=None) or two of type 'NLO' and 'LO' 
        respectively."""

        if not self:
            raise MadGraph5Error('Cannot output an empty histogram group.')
        if _copy_group:
            self = copy.deepcopy(self)
        canonical_labels = list(self[0].bins.weight_labels)
        canonical_set = set(canonical_labels)
        if len(canonical_labels) != len(canonical_set):
            raise MadGraph5Error("Histogram '%s' has duplicate weight columns."%
                                 self[0].title)
        if _output_schema_state is not None and \
                         _output_schema_state.get('base_labels') is not None:
            base_labels = list(_output_schema_state['base_labels'])
            if len(base_labels) != len(set(base_labels)) or \
                                           set(base_labels) != canonical_set:
                raise MadGraph5Error("Histogram group '%s' has a different base "
                                     "column schema."%self[0].title)
            canonical_labels = base_labels
            self[0].bins.weight_labels = list(canonical_labels)
        for histo in self[1:]:
            labels = list(histo.bins.weight_labels)
            if len(labels) != len(set(labels)) or set(labels) != canonical_set:
                raise MadGraph5Error("Histograms in plot group '%s' have "
                                     "incompatible weight schemas."%self[0].title)
            histo.bins.weight_labels = list(canonical_labels)
        
        # This function returns the main central plot line, making sure that
        # negative distribution are displayed in dashed style
        def get_main_central_plot_lines(HwU_name, block_position, color_index,
                                                  title, show_mc_uncertainties):
            """ Returns two plot lines, one for the negative contributions in
            dashed and one with the positive ones in solid."""
            
            template = "'%(hwu)s' index %(ind)d using (($1+$2)/2):%(data)s%(stat_col)s%(stat_err)s%(ls)s%(title)s"
            template_no_stat = "'%(hwu)s' index %(ind)d using (($1+$2)/2):%(data)s%(ls)s%(title)s"        
            rep_dic = {'hwu':HwU_name,
                       'ind':block_position,
                       'ls':' ls %d'%color_index,
                       'title':" title '%s'"%title,
                       'stat_col': ':4',
                       'stat_err': ' w yerrorbar',
                       'data':'3',
                       'linetype':''}

            # This would be the original output
            # return [template_no_stat%rep_dic]+\
            #               ([template%rep_dic] if show_mc_uncertainties else [])
            
            # The use of 1/0 is just a trick to prevent the line to display
            res = []
            rep_dic['data'] = '($3 < 0 ? 1/0 : $3)'
            res.append(template_no_stat%rep_dic)
            rep_dic['title'] = " title ''"
            if show_mc_uncertainties:
                res.append(template%rep_dic)                
            rep_dic['data'] = '($3 >= 0 ? 1/0 : abs($3))'
            rep_dic['ls']  = ' ls %d'%(100+color_index)            
            res.append(template_no_stat%rep_dic)
            if show_mc_uncertainties:
                res.append(template%rep_dic)
            return res
        
        # This bool can be modified later to decide whether to use uncertainty
        # bands or not
        # ========
        def get_uncertainty_lines(HwU_name, block_position, 
                           var_pos, color_index,title, ratio=False, band=False):
            """ Return a string line corresponding to the plotting of the
            uncertainty. Band is to chose wether to display uncertainty with
            a band or two lines."""

            # This perl substitution regular expression copies each line of the
            # HwU source and swap the x1 and x2 coordinate of the second copy.
            # So if input is:
            #
            #  blabla
            #  +0.01e+01 0.3 4 5 6
            #  +0.03e+01 0.5 7 8 9
            #  ...
            # 
            # The output will be
            #
            #  blabla
            #  +0.01e+01 0.3 4 5 6
            #  0.3 +0.01e+01 4 5 6
            #  +0.03e+01 0.5 7 8 9
            #  0.5 +0.03e+01 7 8 9
            #  ...
            #
            copy_swap_re = r"perl -pe 's/^\s*(?<x1>[\+|-]?\d+(\.\d*)?([EeDd][\+|-]?\d+)?)\s*(?<x2>[\+|-]?\d+(\.\d*)?([EeDd][\+|-]?\d+)?)(?<rest>.*)\n/ $+{x1} $+{x2} $+{rest}\n$+{x2} $+{x1} $+{rest}\n/g'"
            # Gnuplot escapes the antislash, so we must esacape then once more O_o.
            # Gnuplot doesn't have raw strings, what a shame...
            copy_swap_re = copy_swap_re.replace('\\','\\\\')
            # For the ratio, we must divide by the central value
            position = '(safe($%d,$3,1.0)-1.0)' if ratio else '%d'
            if not band:
                return ["'%s' index %d using (($1+$2)/2):%s ls %d title '%s'"\
                 %(HwU_name,block_position, position%(var_pos),color_index,title),
                        "'%s' index %d using (($1+$2)/2):%s ls %d title ''"\
                 %(HwU_name,block_position, position%(var_pos+1),color_index)]
            else:
                return [' "<%s %s" index %d using 1:%s:%s with filledcurve ls %d fs transparent solid 0.2 title \'%s\''%\
            (copy_swap_re,HwU_name,block_position,
                       position%var_pos,position%(var_pos+1),color_index,title)]
        # ========
                
        
        layout_geometry = [(0.0, 0.5,  1.0, 0.4 ),
                           (0.0, 0.35, 1.0, 0.15),
                           (0.0, 0.2,  1.0, 0.15)]
        layout_geometry.reverse()
   
        # Group histograms which just differ by jet multiplicity and add their 
        # sum as first plot
        matching_histo_lists = HwUList([HwUList([self[0]])])
        for histo in self[1:]:
            matched = False
            for histo_list in matching_histo_lists:
                if hasattr(histo, 'jetsample') and histo.jetsample >= 0 and \
                                               histo.type == histo_list[0].type:
                    matched = True
                    histo_list.append(histo)
                    break
            if not matched:
                matching_histo_lists.append(HwUList([histo]))
        
        # For each group of histograms with different jet multiplicities, we
        # define one at the beginning which is the sum.
        self[:] = []
        for histo_group in matching_histo_lists:
            # First create a plot that sums all jet multiplicities for each type
            # (that is, only if jet multiplicities are defined)
            if len(histo_group)==1:
                self.append(histo_group[0])
                continue
            # If there is already a histogram summing them, then don't create
            # a copy of it.
            if any(hist.jetsample==-1 for hist in histo_group if 
                                                    hasattr(hist, 'jetsample')):
                self.extend(histo_group)
                continue
            summed_histogram = copy.copy(histo_group[0])
            for histo in histo_group[1:]:
                summed_histogram = summed_histogram + histo
            summed_histogram.jetsample = -1
            self.append(summed_histogram)
            self.extend(histo_group)

        # Remove the curve of individual jet samples if they are not desired
        if not jet_samples_to_keep is None:
            self[:] = [histo for histo in self if (not hasattr(histo,'jetsample')) or (histo.jetsample == -1) or
                                 (histo.jetsample in jet_samples_to_keep)]

        # This function is to create the ratio histograms if the user turned off
        # correlations.
        def ratio_no_correlations(wgtsA, wgtsB):
            new_wgts = {}
            for label, wgt in wgtsA.items():
                if wgtsB['central']==0.0 and wgt==0.0:
                    new_wgts[label] = 0.0
                    continue
                elif wgtsB['central']==0.0:
                    # It is ok to skip the warning here.
#                   logger.debug('Warning:: A bin with finite weight '+
#                                      'was divided by a bin with zero weight.')
                    new_wgts[label] = 0.0
                    continue
                new_wgts[label] = (wgtsA[label]/wgtsB['central'])
            return new_wgts
        ratio_no_correlations._histogram_lazy_operation = \
                                                    'ratio_no_correlations'
        
        # First compute the ratio of all the histograms from the second to the
        # number_of_ratios+1 ones in the list to the first histogram.
        n_histograms = len(self)
        ratio_histos = HwUList([])
        # A counter to keep track of the number of ratios included
        n_ratios_included = 0
        for i, histo in enumerate(self[1:]):
            if not hasattr(histo,'jetsample') or histo.jetsample==self[0].jetsample:
                n_ratios_included += 1
            else:
                continue
            
            if number_of_ratios >=0 and n_ratios_included > number_of_ratios:
                break
            
            if ratio_correlations:
                ratio_histos.append(histo/self[0])
            else:
                ratio_histos.append(self[0].__class__.combine(histo, self[0],
                                                         ratio_no_correlations))
            if self[0].type=='NLO' and histo.type=='LO':
                ratio_histos[-1].title += '1/K-factor'
            elif self[0].type=='LO' and histo.type=='NLO':
                ratio_histos[-1].title += 'K-factor'
            else:
                ratio_histos[-1].title += ' %s/%s'%(
                              histo.type if histo.type else '(%d)'%(i+2),
                              self[0].type if self[0].type else '(1)')
            # By setting its type to aux, we make sure this histogram will be
            # filtered out if the .HwU file output here would be re-loaded later.
            ratio_histos[-1].type       = 'AUX'
        self.extend(ratio_histos)

        # Compute scale variation envelope for all diagrams
        if 'scale' in uncertainties:
            (mu_var_pos,mu)  = self[0].set_uncertainty(type='all_scale')
        else:
            (mu_var_pos,mu) = (None,[None])
        
        if 'pdf' in uncertainties: 
            (PDF_var_pos,pdf) = self[0].set_uncertainty(type='PDF',lhapdfconfig=lhapdfconfig)
        else:
            (PDF_var_pos,pdf) = (None,[None])
        
        if 'merging_scale' in uncertainties:
            (merging_var_pos,merging) = self[0].set_uncertainty(type='merging')
        else:
            (merging_var_pos,merging) = (None,[None])
        if 'alpsfact' in uncertainties: 
            (alpsfact_var_pos,alpsfact) = self[0].set_uncertainty(type='alpsfact')
        else:
            (alpsfact_var_pos,alpsfact) = (None,[None])

        uncertainties_present =  list(uncertainties)
        if PDF_var_pos is None and 'pdf' in uncertainties_present:
            uncertainties_present.remove('pdf')
        if mu_var_pos is None and 'scale' in uncertainties_present:
            uncertainties_present.remove('scale')
        if merging_var_pos is None and 'merging_scale' in uncertainties_present:
            uncertainties_present.remove('merging_scale')
        if alpsfact_var_pos is None and 'alpsfact' in uncertainties_present:
            uncertainties_present.remove('alpsfact')
        no_uncertainties = len(uncertainties_present)==0
        
        # If the 'use_band' option is None we should adopt a default which is
        try:
            uncertainties_present.remove('statistical')
        except:
            pass
        if use_band is None:
            # For clarity, it is better to only use bands only for one source
            # of uncertainty
            if len(uncertainties_present)==0:
                use_band = []                
            elif len(uncertainties_present)==1:
                use_band = uncertainties_present
            elif 'scale' in uncertainties_present:
                use_band = ['scale']
            else:
                use_band = [uncertainties_present[0]]

        for histo in self[1:]:
            if (not mu_var_pos is None) and \
                          mu_var_pos != histo.set_uncertainty(type='all_scale')[0]:
               raise MadGraph5Error('Not all histograms in this group specify'+\
                 ' scale uncertainties. It is required to be able to output them'+\
                 ' together.')
            if (not PDF_var_pos is None) and\
                               PDF_var_pos != histo.set_uncertainty(type='PDF',\
                                                                    lhapdfconfig=lhapdfconfig)[0]:
               raise MadGraph5Error('Not all histograms in this group specify'+\
                 ' PDF uncertainties. It is required to be able to output them'+\
                 ' together.')
            if (not merging_var_pos is None) and\
                            merging_var_pos != histo.set_uncertainty(type='merging')[0]:
               raise MadGraph5Error('Not all histograms in this group specify'+\
                 ' merging uncertainties. It is required to be able to output them'+\
                 ' together.')
            if (not alpsfact_var_pos is None) and\
                            alpsfact_var_pos != histo.set_uncertainty(type='alpsfact')[0]:
               raise MadGraph5Error('Not all histograms in this group specify'+\
                 ' alpsfact uncertainties. It is required to be able to output them'+\
                 ' together.')

        # HwU has one global column header.  Uncertainty preparation above may
        # append auxiliary columns, so validate the final schema both within
        # this plot group and across all groups written to the same file.
        final_labels = list(self[0].bins.weight_labels)
        if len(final_labels) != len(set(final_labels)):
            raise MadGraph5Error("Histogram '%s' has duplicate output columns."%
                                 self[0].title)
        for histo in self[1:]:
            if list(histo.bins.weight_labels) != final_labels:
                raise MadGraph5Error("Histograms in plot group '%s' produced "
                                     "different output columns."%self[0].title)
        if _output_schema_state is not None:
            previous_labels = _output_schema_state.get('labels')
            if previous_labels is None:
                _output_schema_state['labels'] = final_labels
            elif previous_labels != final_labels:
                raise MadGraph5Error("Histogram group '%s' produced a different "
                    "final column schema from earlier groups in this HwU file."%
                    self[0].title)

        # Now output the corresponding HwU histogram data
        for i, histo in enumerate(self):
            # Print the header the first time only
            HwU_out.extend(histo.iter_HwU_source(\
                                     print_header=(block_position==0 and i==0)))
            HwU_out.extend(['',''])

        # First the global gnuplot header for this histogram group
        global_header =\
r"""
################################################################################
### Rendering of the plot titled '%(title)s'
################################################################################

set multiplot
set label "%(title)s" font ",13" at graph 0.04, graph 1.05
set xrange [%(xmin).4e:%(xmax).4e]
set bmargin 0 
set tmargin 0
set xtics nomirror
set ytics nomirror
set mytics %(mxtics)d
%(set_xtics)s
set key horizontal noreverse maxcols 1 width -4 
set label front 'MadGraph5\_aMC\@NLO' font "Courier,11" rotate by 90 at graph 1.02, graph 0.04
"""
        
        # Now the header for each subhistogram
        subhistogram_header = \
"""#-- rendering subhistograms '%(subhistogram_type)s'
%(unset label)s
%(set_format_y)s
%(set_yscale)s
set yrange [%(ymin).4e:%(ymax).4e]
set origin %(origin_x).4e, %(origin_y).4e
set size %(size_x).4e, %(size_y).4e
set mytics %(mytics)d
%(set_ytics)s
%(set_format_x)s
%(set_ylabel)s
%(set_histo_label)s
plot \\"""
        replacement_dic = {}

        replacement_dic['title'] = self[0].get_HwU_histogram_name(format='human-no_type')
        # Determine what weight to consider when computing the optimal 
        # range for the y-axis.
        wgts_to_consider = ['central']
        if not mu_var_pos is None:
            for mu_var in mu_var_pos:
                wgts_to_consider.append(self[0].bins.weight_labels[mu_var])
                wgts_to_consider.append(self[0].bins.weight_labels[mu_var+1])
                wgts_to_consider.append(self[0].bins.weight_labels[mu_var+2])
        if not PDF_var_pos is None:
            for PDF_var in PDF_var_pos:
                wgts_to_consider.append(self[0].bins.weight_labels[PDF_var])
                wgts_to_consider.append(self[0].bins.weight_labels[PDF_var+1])
                wgts_to_consider.append(self[0].bins.weight_labels[PDF_var+2])                
        if not merging_var_pos is None:
            for merging_var in merging_var_pos:
                wgts_to_consider.append(self[0].bins.weight_labels[merging_var])
                wgts_to_consider.append(self[0].bins.weight_labels[merging_var+1])
                wgts_to_consider.append(self[0].bins.weight_labels[merging_var+2])                
        if not alpsfact_var_pos is None:
            for alpsfact_var in alpsfact_var_pos: 
                wgts_to_consider.append(self[0].bins.weight_labels[alpsfact_var])
                wgts_to_consider.append(self[0].bins.weight_labels[alpsfact_var+1])
                wgts_to_consider.append(self[0].bins.weight_labels[alpsfact_var+2])

        (xmin, xmax) = HwU.get_x_optimal_range(self[:n_histograms],\
                                               weight_labels = wgts_to_consider)
        replacement_dic['xmin'] = xmin
        replacement_dic['xmax'] = xmax
        replacement_dic['mxtics'] = 10
        replacement_dic['set_xtics'] = 'set xtics auto'
        
        # Add the global header which is now ready
        gnuplot_out.append(global_header%replacement_dic)
        
        # Now add the main plot
        replacement_dic['subhistogram_type'] = '%s and %s results'%(
                 str(self[0].type),str(self[1].type)) if len(self)>1 else \
                                                         'single diagram output'
        y_axis_mode = self[0].y_axis_mode
        (ymin, ymax) = HwU.get_y_optimal_range(self[:n_histograms],
                   labels = wgts_to_consider, scale=y_axis_mode)
            
        # Already add a margin on upper bound.
        if y_axis_mode=='LOG':
            ymax += 10.0 * ymax
            ymin -= 0.1 * ymin
        else:
            ymax += 0.3 * (ymax - ymin)
            ymin -= 0.3 * (ymax - ymin)

        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        replacement_dic['unset label'] = ''
        (replacement_dic['origin_x'], replacement_dic['origin_y'],
         replacement_dic['size_x'], replacement_dic['size_y']) = layout_geometry.pop()
        replacement_dic['mytics'] = 10
        # Use default choise for the main histogram
        replacement_dic['set_ytics'] = 'set ytics auto'
        replacement_dic['set_format_x'] = "set format x ''" if \
          (len(self)-n_histograms>0 or not no_uncertainties) else "set format x"
        replacement_dic['set_ylabel'] = 'set ylabel "{/Symbol s} per bin [pb]"' 
        replacement_dic['set_yscale'] = "set logscale y" if \
                             y_axis_mode=='LOG' else 'unset logscale y'
        replacement_dic['set_format_y'] = "set format y '10^{%T}'" if \
                                y_axis_mode=='LOG' else 'unset format'
                                
        replacement_dic['set_histo_label'] = ""
        gnuplot_out.append(subhistogram_header%replacement_dic)
        
        # Now add the main layout
        plot_lines = []
        uncertainty_plot_lines = []
        n=-1

        for i, histo in enumerate(self[:n_histograms]):
            n=n+1
            color_index = n%self.number_line_colors_defined+1
            # Label to appear for the lower curves 
            title = []
            if histo.type is None and not hasattr(histo, 'jetsample'):
                title.append('%d'%(i+1))
            else:
                if histo.type:
                    title.append('NLO' if \
                                   histo.type.split()[0]=='NLO' else histo.type)
                if hasattr(histo, 'jetsample'):
                    if histo.jetsample!=-1:
                        title.append('jet sample %d'%histo.jetsample)
                    else:
                        title.append('all jet samples')
                        
            title = ', '.join(title)
            # Label for the first curve in the upper plot
            if histo.type is None and not hasattr(histo, 'jetsample'):
                major_title = 'central value for plot (%d)'%(i+1)
            else:
                major_title = []
                if not histo.type is None:
                    major_title.append(histo.type)
                if hasattr(histo, 'jetsample'):
                    if histo.jetsample!=-1:
                        major_title.append('jet sample %d'%histo.jetsample)
                    else:
                        major_title.append('all jet samples')
                else:
                    major_title.append('central value')
                major_title = ', '.join(major_title)                    
            
            if not mu[0] in ['none',None]:
                major_title += r', dynamical\_scale\_choice=%s'%mu[0]
            if not pdf[0] in ['none',None]:
                major_title += ', PDF=%s'%pdf[0].replace('_',r'\_')

            # Do not show uncertainties for individual jet samples (unless first
            # or specified explicitely and uniquely)
            if not (i!=0 and hasattr(histo,'jetsample') and histo.jetsample!=-1 and \
               not (jet_samples_to_keep and len(jet_samples_to_keep)==1 and 
                    jet_samples_to_keep[0] == histo.jetsample)):
                
                uncertainty_plot_lines.append({})
                
                # We decide to show uncertainties in the main plot only if they
                # are part of a monocolor band. Otherwise, they will only be 
                # shown in the first subplot. Notice that plotting '1/0'
                # is just a trick so as to have only the key printed with no
                # line
                
                # Show scale variation for the first central value if available
                if not mu_var_pos is None and len(mu_var_pos)>0:
                    if 'scale' in use_band:
                      uncertainty_plot_lines[-1]['scale'] = get_uncertainty_lines(
                        HwU_name, block_position+i, mu_var_pos[0]+4, color_index+10,
                        '%s, scale variation'%title, band='scale' in use_band)
                    else:
                      uncertainty_plot_lines[-1]['scale'] = \
      ["1/0 ls %d title '%s'"%(color_index+10,'%s, scale variation'%title)]
                # And now PDF_variation if available
                if not PDF_var_pos is None and len(PDF_var_pos)>0:
                    if 'pdf' in use_band:
                        uncertainty_plot_lines[-1]['pdf'] = get_uncertainty_lines(
                     HwU_name,block_position+i, PDF_var_pos[0]+4, color_index+20,
                             '%s, PDF variation'%title, band='pdf' in use_band)
                    else:
                        uncertainty_plot_lines[-1]['pdf'] = \
        ["1/0 ls %d title '%s'"%(color_index+20,'%s, PDF variation'%title)]
                # And now merging variation if available
                if not merging_var_pos is None and len(merging_var_pos)>0:
                    if 'merging_scale' in use_band:
                        uncertainty_plot_lines[-1]['merging_scale'] = get_uncertainty_lines(
                     HwU_name,block_position+i, merging_var_pos[0]+4, color_index+30,
                '%s, merging scale variation'%title, band='merging_scale' in use_band)
                    else:
                        uncertainty_plot_lines[-1]['merging_scale'] = \
        ["1/0 ls %d title '%s'"%(color_index+30,'%s, merging scale variation'%title)]
                # And now alpsfact variation if available
                if not alpsfact_var_pos is None and len(alpsfact_var_pos)>0:
                    if 'alpsfact' in use_band:
                        uncertainty_plot_lines[-1]['alpsfact'] = get_uncertainty_lines(
                     HwU_name,block_position+i, alpsfact_var_pos[0]+4, color_index+40,
                    '%s, alpsfact variation'%title, band='alpsfact' in use_band)
                    else:
                        uncertainty_plot_lines[-1]['alpsfact'] = \
        ["1/0 ls %d title '%s'"%(color_index+40,'%s, alpsfact variation'%title)]

#            plot_lines.append(
# "'%s' index %d using (($1+$2)/2):3 ls %d title '%s'"\
# %(HwU_name,block_position+i,color_index, major_title))
#            if 'statistical' in uncertainties:
#                plot_lines.append(
# "'%s' index %d using (($1+$2)/2):3:4 w yerrorbar ls %d title ''"\
# %(HwU_name,block_position+i,color_index))
            plot_lines.extend(
                get_main_central_plot_lines(HwU_name, block_position+i,
                      color_index, major_title, 'statistical' in uncertainties))

            # Add additional central scale/PDF curves
            if not mu_var_pos is None:
                for j,mu_var in enumerate(mu_var_pos):
                    if j!=0:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                        plot_lines.append(
"'%s' index %d using (($1+$2)/2):%d ls %d title '%s'"\
%(HwU_name,block_position+i,mu_var+3,color_index,\
r'%s dynamical\_scale\_choice=%s' % (title,mu[j])))
            # And now PDF_variation if available
            if not PDF_var_pos is None:
                for j,PDF_var in enumerate(PDF_var_pos):
                    if j!=0:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                        plot_lines.append(
"'%s' index %d using (($1+$2)/2):%d ls %d title '%s'"\
%(HwU_name,block_position+i,PDF_var+3,color_index,\
'%s PDF=%s' % (title,pdf[j].replace('_',r'\_'))))

        # Now add the uncertainty lines, those not using a band so that they
        # are not covered by those using a band after we reverse plo_lines
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if not uncertainty_type in use_band:
                    plot_lines.extend(lines)
        # then those using a band
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if uncertainty_type in use_band:
                    plot_lines.extend(lines)
    
        # Reverse so that bands appear first
        plot_lines.reverse()

        # Add the plot lines
        gnuplot_out.append(',\\\n'.join(plot_lines))

        # Now we can add the scale variation ratio
        replacement_dic['subhistogram_type'] = 'Relative scale and PDF uncertainty'

        if 'statistical' in uncertainties: 
            wgts_to_consider.append('stat_error')

        # This function is just to temporarily create the scale ratio histogram with 
        # the hwu.combine function. 
        def rel_scale(wgtsA, wgtsB):
            new_wgts = {}
            for label, wgt in wgtsA.items():
                if label in wgts_to_consider:
                    if wgtsB['central']==0.0 and wgt==0.0:
                        new_wgts[label] = 0.0
                        continue
                    elif wgtsB['central']==0.0:
#                       It is ok to skip the warning here.
#                        logger.debug('Warning:: A bin with finite weight '+
#                                       'was divided by a bin with zero weight.')
                        new_wgts[label] = 0.0
                        continue
                    new_wgts[label] = (wgtsA[label]/wgtsB['central'])
                    if label != 'stat_error':
                        new_wgts[label] -= 1.0
                else:
                    new_wgts[label] = wgtsA[label]
            return new_wgts

        histos_for_subplots = [(i,histo) for i, histo in enumerate(self[:n_histograms]) if
            (  not (i!=0 and hasattr(histo,'jetsample') and histo.jetsample!=-1 and \
               not (jet_samples_to_keep and len(jet_samples_to_keep)==1 and 
                    jet_samples_to_keep[0] == histo.jetsample)) )]

        # Notice even though a ratio histogram is created here, it
        # is not actually used to plot the quantity in gnuplot, but just to
        # compute the y range. 
        (ymin, ymax) = HwU.get_y_optimal_range([histo[1].__class__.combine(
                    histo[1],histo[1],rel_scale) for histo in histos_for_subplots],
                                                  labels = wgts_to_consider,  scale='LIN')

        # Add a margin on upper and lower bound.
        ymax = ymax + 0.2 * (ymax - ymin)
        ymin = ymin - 0.2 * (ymax - ymin)
        replacement_dic['unset label'] = 'unset label'
        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        if not no_uncertainties:
            (replacement_dic['origin_x'], replacement_dic['origin_y'],
         replacement_dic['size_x'], replacement_dic['size_y']) = layout_geometry.pop()
        replacement_dic['mytics'] = 2
#        replacement_dic['set_ytics'] = 'set ytics %f'%((int(10*(ymax-ymin))/10)/3.0)
        replacement_dic['set_ytics'] = 'set ytics auto'
        replacement_dic['set_format_x'] = "set format x ''" if \
                                    len(self)-n_histograms>0 else "set format x"
        replacement_dic['set_ylabel'] = 'set ylabel "%s rel.unc."'\
                              %('(1)' if self[0].type==None else '%s'%('NLO' if \
                              self[0].type.split()[0]=='NLO' else self[0].type))
        replacement_dic['set_yscale'] = "unset logscale y"
        replacement_dic['set_format_y'] = 'unset format'
                                

        tit='Relative uncertainties w.r.t. central value'
        if n_histograms > 1:
            tit=tit+'s'
#        if (not mu_var_pos is None and 'scale' not in use_band):
#            tit=tit+', scale is dashed'
#        if (not PDF_var_pos is None and 'pdf' not in use_band):
#            tit=tit+', PDF is dotted'
        replacement_dic['set_histo_label'] = \
         'set label "%s" font ",9" front at graph 0.03, graph 0.13' % tit
        # Simply don't add these lines if there are no uncertainties.
        # This meant uncessary extra work, but I no longer car at this point
        if not no_uncertainties:
            gnuplot_out.append(subhistogram_header%replacement_dic)
        
        # Now add the first subhistogram
        plot_lines = []
        uncertainty_plot_lines = []
        n=-1
        for (i,histo) in histos_for_subplots:
            n=n+1
            k=n
            color_index = n%self.number_line_colors_defined+1
            # Plot uncertainties
            if not mu_var_pos is None:
                for j,mu_var in enumerate(mu_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0:
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    # Add the central line only if advanced scale variation
                    if j>0 or mu[j]!='none':
                        plot_lines.append(
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls %d title ''"\
                      %(HwU_name,block_position+i,mu_var+3,color_index))
                    uncertainty_plot_lines[-1]['scale'] = get_uncertainty_lines(
                     HwU_name, block_position+i, mu_var+4, color_index+10,'',
                                             ratio=True, band='scale' in use_band)
            if not PDF_var_pos is None:
                for j,PDF_var in enumerate(PDF_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0:
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    # Add the central line only if advanced pdf variation                            
                    if j>0 or pdf[j]!='none':
                        plot_lines.append(
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls %d title ''"\
                      %(HwU_name,block_position+i,PDF_var+3,color_index))
                    uncertainty_plot_lines[-1]['pdf'] = get_uncertainty_lines(
                    HwU_name, block_position+i, PDF_var+4, color_index+20,'',
                                        ratio=True, band='pdf' in use_band)
            if not merging_var_pos is None:
                for j,merging_var in enumerate(merging_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0:
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    if j>0 or merging[j]!='none':                    
                       plot_lines.append(
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls %d title ''"\
                       %(HwU_name,block_position+i,merging_var+3,color_index))
                    uncertainty_plot_lines[-1]['merging_scale'] = get_uncertainty_lines(
                    HwU_name, block_position+i, merging_var+4, color_index+30,'',
                                    ratio=True, band='merging_scale' in use_band)
            if not alpsfact_var_pos is None:
                for j,alpsfact_var in enumerate(alpsfact_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0:
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    if j>0 or alpsfact[j]!='none':
                        plot_lines.append(
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls %d title ''"\
                       %(HwU_name,block_position+i,alpsfact_var+3,color_index))
                    uncertainty_plot_lines[-1]['alpsfact'] = get_uncertainty_lines(
                    HwU_name, block_position+i, alpsfact_var+4, color_index+40,'',
                                   ratio=True, band='alpsfact' in use_band)
                
            if 'statistical' in uncertainties:
                   plot_lines.append(
    "'%s' index %d using (($1+$2)/2):(0.0):(safe($4,$3,0.0)) w yerrorbar ls %d title ''"%\
    (HwU_name,block_position+i,color_index))

        plot_lines.append("0.0 ls 999 title ''")
        
        # Now add the uncertainty lines, those not using a band so that they
        # are not covered by those using a band after we reverse plo_lines
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if not uncertainty_type in use_band:
                    plot_lines.extend(lines)
        # then those using a band
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if uncertainty_type in use_band:
                    plot_lines.extend(lines)
    
        # Reverse so that bands appear first
        plot_lines.reverse()
        # Add the plot lines
        if not no_uncertainties:
            gnuplot_out.append(',\\\n'.join(plot_lines))

        # We finish here when no ratio plot are asked for.
        if len(self)-n_histograms==0:
            # Now add the tail for this group
            gnuplot_out.extend(['','unset label','',
'################################################################################'])
            # Return the starting data_block position for the next histogram group
            return block_position+len(self)

        # We can finally add the last subhistograms for the ratios.
        ratio_name_long='('
        for i, histo in enumerate(self[:n_histograms]):
            if i==0: continue
            ratio_name_long+='%d'%(i+1) if histo.type is None else ('NLO' if \
                                    histo.type.split()[0]=='NLO' else histo.type)
        ratio_name_long+=')/'
        ratio_name_long+=('(1' if self[0].type==None else '(%s'%('NLO' if \
            self[0].type.split()[0]=='NLO' else self[0].type))+' central value)'

        ratio_name_short = 'ratio w.r.t. '+('1' if self[0].type==None else '%s'%('NLO' if \
                                            self[0].type.split()[0]=='NLO' else self[0].type))
            
        replacement_dic['subhistogram_type'] = '%s ratio'%ratio_name_long
        replacement_dic['set_ylabel'] = 'set ylabel "%s"'%ratio_name_short

        (ymin, ymax) = HwU.get_y_optimal_range(self[n_histograms:], 
               labels = wgts_to_consider, scale='LIN',Kratio = True)    
        
        # Add a margin on upper and lower bound.
        ymax = ymax + 0.2 * (ymax - ymin)
        ymin = ymin - 0.2 * (ymax - ymin)
        replacement_dic['unset label'] = 'unset label'
        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        (replacement_dic['origin_x'], replacement_dic['origin_y'],
         replacement_dic['size_x'], replacement_dic['size_y']) = layout_geometry.pop()
        replacement_dic['mytics'] = 2
#        replacement_dic['set_ytics'] = 'set ytics %f'%((int(10*(ymax-ymin))/10)/10.0)
        replacement_dic['set_ytics'] = 'set ytics auto'
        replacement_dic['set_format_x'] = "set format x"
        replacement_dic['set_yscale'] = "unset logscale y"
        replacement_dic['set_format_y'] = 'unset format'
        replacement_dic['set_histo_label'] = \
        'set label "%s" font ",9" at graph 0.03, graph 0.13'%ratio_name_long
#        'set label "NLO/LO (K-factor)" font ",9" at graph 0.82, graph 0.13'
        gnuplot_out.append(subhistogram_header%replacement_dic)

        uncertainty_plot_lines = []
        plot_lines = []

        # Some crap to get the colors right I suppose...
        n=-1
        n=n+1
        if not mu_var_pos is None:
            for j,mu_var in enumerate(mu_var_pos):
                if j!=0: n=n+1
        if not PDF_var_pos is None:
            for j,PDF_var in enumerate(PDF_var_pos):
                if j!=0: n=n+1
        if not merging_var_pos is None:
            for j,merging_var in enumerate(merging_var_pos):
                if j!=0: n=n+1
        if not alpsfact_var_pos is None:
            for j,alpsfact_var in enumerate(alpsfact_var_pos):
                if j!=0: n=n+1

        for i_histo_ratio, histo_ration in enumerate(self[n_histograms:]):
            n=n+1
            k=n
            block_ratio_pos = block_position+n_histograms+i_histo_ratio
            color_index     = n%self.number_line_colors_defined+1
            # Now add the subhistograms
            plot_lines.append(
    "'%s' index %d using (($1+$2)/2):3 ls %d title ''"%\
    (HwU_name,block_ratio_pos,color_index))
            if 'statistical' in uncertainties:
                plot_lines.append(
    "'%s' index %d using (($1+$2)/2):3:4 w yerrorbar ls %d title ''"%\
    (HwU_name,block_ratio_pos,color_index))

            # Then the scale variations
            if not mu_var_pos is None:
                for j,mu_var in enumerate(mu_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0:
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    # Only print out the additional central value for advanced scale variation                        
                    if j>0 or mu[j]!='none':
                      plot_lines.append(
    "'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
    %(HwU_name,block_ratio_pos,mu_var+3,color_index))
                    uncertainty_plot_lines[-1]['scale'] = get_uncertainty_lines(
                       HwU_name, block_ratio_pos, mu_var+4, color_index+10,'',
                                                       band='scale' in use_band)
            if not PDF_var_pos is None:
                for j,PDF_var in enumerate(PDF_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0: 
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    # Only print out the additional central value for advanced pdf variation
                    if j>0 or pdf[j]!='none':                        
                      plot_lines.append(
    "'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
    %(HwU_name,block_ratio_pos,PDF_var+3,color_index))
                    uncertainty_plot_lines[-1]['pdf'] = get_uncertainty_lines(
                      HwU_name, block_ratio_pos, PDF_var+4, color_index+20,'',
                                                       band='pdf' in use_band)
            if not merging_var_pos is None:
                for j,merging_var in enumerate(merging_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0: 
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    if j>0 or merging[j]!='none':
                        plot_lines.append(
    "'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
    %(HwU_name,block_ratio_pos,merging_var+3,color_index))
                    uncertainty_plot_lines[-1]['merging_scale'] = get_uncertainty_lines(
                    HwU_name, block_ratio_pos, merging_var+4, color_index+30,'',
                                                     band='merging_scale' in use_band)
            if not alpsfact_var_pos is None:
                for j,alpsfact_var in enumerate(alpsfact_var_pos):
                    uncertainty_plot_lines.append({})
                    if j==0: 
                        color_index = k%self.number_line_colors_defined+1
                    else:
                        n=n+1
                        color_index = n%self.number_line_colors_defined+1
                    if j>0 or alpsfact[j]!='none':
                        plot_lines.append(
    "'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
    %(HwU_name,block_ratio_pos,alpsfact_var+3,color_index))
                    uncertainty_plot_lines[-1]['alpsfact'] = get_uncertainty_lines(
                      HwU_name, block_ratio_pos, alpsfact_var+4, color_index+40,'',
                                                     band='alpsfact' in use_band)             

        # Now add the uncertainty lines, those not using a band so that they
        # are not covered by those using a band after we reverse plo_lines
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if not uncertainty_type in use_band:
                    plot_lines.extend(lines)
        # then those using a band
        for one_plot in uncertainty_plot_lines:
            for uncertainty_type, lines in one_plot.items():
                if uncertainty_type in use_band:
                    plot_lines.extend(lines)
       
        plot_lines.append("1.0 ls 999 title ''")

        # Reverse so that bands appear first
        plot_lines.reverse()
        # Add the plot lines
        gnuplot_out.append(',\\\n'.join(plot_lines))
        
        # Now add the tail for this group
        gnuplot_out.extend(['','unset label','',
'################################################################################'])

        # Return the starting data_block position for the next histogram group
        return block_position+len(self)
    
################################################################################
## HTML and matplotlib related functions
################################################################################
def plot_ratio_from_HWU(path, ax, hwu_variable, hwu_numerator, hwu_denominator, *args, **opts):
    """INPUT:
       - path can be a path to HwU or an HwUList instance
       - ax is the matplotlib frame where to do the plot
       - hwu_variable is the histograms to consider
       - hwu_numerator is the numerator of the ratio plot
       - hwu_denominator is the denominator of the ratio plot
       OUTPUT:
       - adding the curves to the plot
       - return the HwUList
    """

    if isinstance(path, str):
        hwu = HwUList(path, raw_labels=True)
    else:
        hwu = path

    if 'hwu_denominator_path' in opts:
        if isinstance(opts['hwu_denominator_path'],str):
            hwu2 = HwUList(opts['hwu_denominator_path'], raw_labels=True)
        else:
            hwu2 = opts['hwu_denominator_path']
        del opts['hwu_denominator_path']
    else:
        hwu2 = hwu


    select_hist = hwu.get(hwu_variable)
    select_hist2 = hwu2.get(hwu_variable)
    bins = select_hist.get('bins')
    num = select_hist.get(hwu_numerator)
    denom = select_hist2.get(hwu_denominator)
    ratio = [num[i]/denom[i] if denom[i] else 1 for i in range(len(bins))]
    if 'drawstyle' not in opts:
        opts['drawstyle'] = 'steps'
    ax.plot(bins, ratio, *args, **opts)
    return hwu

def plot_from_HWU(path, ax, hwu_variable, hwu_central, *args, **opts):
    """INPUT:
       - path can be a path to HwU or an HwUList instance
       - ax is the matplotlib frame where to do the plot
       - hwu_variable is the histograms to consider
       - hwu_central is the central curve to consider
       - hwu_error is the error band to consider (optional: Default is no band)
       - hwu_error_mode is how to compute the error band (optional)
       OUTPUT:
       - adding the curves to the plot
       - return the HwUList
       - return the line associated to the central (can be used to get the color)
    """

#   Handle optional parameter
    if 'hwu_error' in opts:
        hwu_error = opts['hwu_error']
        del opts['hwu_error']
    else:
        hwu_error = None

    if 'hwu_error_mode' in opts:
        hwu_error_mode = opts['hwu_error_mode']
        del opts['hwu_error_mode']
    else:
        hwu_error_mode = None

    if 'hwu_mult' in opts:
        hwu_mult = opts['hwu_mult']
        del opts['hwu_mult']
    else:
        hwu_mult = 1

    if isinstance(path, str):
        hwu = HwUList(path, raw_labels=True)
    else:
        hwu = path


    select_hist = hwu.get(hwu_variable)
    bins = select_hist.get('bins')
    central_value = select_hist.get(hwu_central)
    if hwu_mult != 1:
       central_value = [hwu_mult*b for b in central_value]
    if 'drawstyle' not in opts:
        opts['drawstyle'] = 'steps'
    H, = ax.plot(bins, central_value, *args, **opts)

    # Add error band
    if hwu_error:
        if not 'hwu_error_mode' in opts:
            opts['hwu_error_mode']=None
        h_min, h_max = select_hist.get_uncertainty_band(hwu_error, mode=hwu_error_mode)
        if hwu_mult != 1:
            h_min = [hwu_mult*b for b in h_min] 
            h_max = [hwu_mult*b for b in h_max] 
        fill_between_steps(bins, h_min, h_max, ax=ax, facecolor=H.get_color(),
                           alpha=0.5, edgecolor=H.get_color(),hatch='/')

    return hwu, H


_HTML_DOCUMENT_PREFIX = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f4f6f8; color: #17212b; }
header { padding: 1.25rem max(1rem, calc((100%% - 72rem) / 2));
         background: #263746; color: white; }
header h1 { margin: 0 0 .35rem; font-size: 1.5rem; }
header p { margin: .2rem 0; opacity: .86; overflow-wrap: anywhere; }
nav { max-width: 72rem; margin: 1rem auto; padding: 0 1rem; }
nav details { background: white; border: 1px solid #d6dce1;
              border-radius: .4rem; padding: .6rem .8rem; }
nav ol { columns: 2 20rem; margin-bottom: 0; }
nav a { color: #006b9c; }
main { max-width: 72rem; margin: 0 auto; padding: 0 1rem 2rem; }
article { margin: 1rem 0; padding: 1rem; background: white;
          border: 1px solid #d6dce1; border-radius: .5rem;
          box-shadow: 0 1px 3px rgb(0 0 0 / 8%%); }
article h2 { margin: 0 0 .6rem; font-size: 1.15rem; }
svg { display: block; width: 100%%; height: auto; background: white; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1rem;
          margin: .35rem 0 0; font-size: .78rem; }
.legend-item { display: inline-flex; align-items: center; gap: .35rem; }
.swatch { display: inline-block; width: 1.6rem; height: .55rem;
          border-top-width: 2px; border-top-style: solid; }
.error { padding: 1rem; color: #9c1c1c; background: #fff0f0;
         border: 1px solid #e4a7a7; white-space: pre-wrap; }
noscript { display: block; max-width: 72rem; margin: 1rem auto;
           padding: 1rem; background: #fff0f0; color: #7a1212; }
@media (prefers-color-scheme: dark) {
  body { background: #12171c; color: #e7ebef; }
  article, nav details { background: #202830; border-color: #44515d; }
  svg { background: #fff; }
  nav a { color: #72c7ef; }
}
@media print {
  body { background: white; }
  header, nav { display: none; }
  article { break-after: page; border: 0; box-shadow: none; }
}
</style>
</head>
<body>
<header>
  <h1>%(title)s</h1>
  <p>Direct HTML/SVG histogram output generated by MadGraph5_aMC@NLO.</p>
  <p><code>%(command)s</code></p>
</header>
<nav id="contents"><details open><summary>Histograms</summary><ol></ol></details></nav>
<main id="plots"><p id="loading">Rendering histograms...</p></main>
<noscript>JavaScript is required to draw the embedded histogram data.</noscript>
<script id="hwu-source" type="application/json">"'''


_HTML_SCRIPT_BODY = r'''<script>
(function () {
    'use strict';

    var SVG_NS = 'http://www.w3.org/2000/svg';
    var plotSerial = 0;

    function svgElement(name, attributes, text) {
        var element = document.createElementNS(SVG_NS, name);
        Object.keys(attributes || {}).forEach(function (key) {
            element.setAttribute(key, String(attributes[key]));
        });
        if (text !== undefined) {
            element.textContent = text;
        }
        return element;
    }

    function parseBlocks(source) {
        var blocks = [];
        var current = null;
        var expectedRows = null;
        source.split(/\r?\n/).forEach(function (line) {
            var stripped = line.trim();
            if (stripped.indexOf('<histogram>') === 0) {
                if (current !== null) {
                    throw new Error('A histogram starts before the previous block closes.');
                }
                var match = stripped.match(/^<histogram>\s+(\d+)/);
                if (!match) {
                    throw new Error('Malformed histogram opening tag.');
                }
                expectedRows = Number(match[1]);
                current = [];
                return;
            }
            if (stripped === '<\\histogram>') {
                if (current === null) {
                    throw new Error('Histogram closing tag without an opening tag.');
                }
                if (current.length !== expectedRows) {
                    throw new Error('Histogram declares ' + expectedRows +
                                    ' rows but contains ' + current.length + '.');
                }
                blocks.push(current);
                current = null;
                expectedRows = null;
                return;
            }
            if (current === null || !stripped || stripped.charAt(0) === '#') {
                return;
            }
            var data = stripped.split('#', 1)[0].trim();
            if (!data) {
                return;
            }
            var row = data.split(/\s+/).map(function (field) {
                return Number(field.replace(/[dD]/, 'e'));
            });
            if (!row.length || row.some(function (value) {
                    return !Number.isFinite(value);
                })) {
                throw new Error('A histogram row contains invalid numeric data.');
            }
            current.push(row);
        });
        if (current !== null) {
            throw new Error('A histogram block is missing its closing tag.');
        }
        if (!blocks.length) {
            throw new Error('No histogram blocks were found.');
        }
        return blocks;
    }

    function blockAt(blocks, index) {
        if (!Number.isInteger(index) || index < 0 || index >= blocks.length) {
            throw new Error('Histogram block ' + index + ' is missing.');
        }
        return blocks[index];
    }

    function column(block, position) {
        return block.map(function (row) {
            if (position < 0 || position >= row.length) {
                throw new Error('Histogram column ' + position + ' is missing.');
            }
            return row[position];
        });
    }

    function edges(block) {
        var result = column(block, 0);
        result.push(block[block.length - 1][1]);
        return result;
    }

    function centres(block) {
        return block.map(function (row) { return (row[0] + row[1]) / 2; });
    }

    function relative(values, central) {
        return values.map(function (value, index) {
            return central[index] ? value / central[index] - 1 : 0;
        });
    }

    function dashFor(type) {
        return {
            scale: '8 4',
            pdf: '2 3',
            merging_scale: '8 3 2 3',
            alpsfact: '5 2 1 2'
        }[type] || '6 3';
    }

    function finiteDomain(values, logarithmic, reference, padding) {
        var selected = values.filter(function (value) {
            return Number.isFinite(value) && (!logarithmic || value > 0);
        });
        if (reference !== null && reference !== undefined &&
                Number.isFinite(reference) && (!logarithmic || reference > 0)) {
            selected.push(reference);
        }
        if (!selected.length) {
            return logarithmic ? [0.1, 10] : [-1, 1];
        }
        var minimum = Math.min.apply(null, selected);
        var maximum = Math.max.apply(null, selected);
        if (minimum === maximum) {
            if (logarithmic) {
                minimum /= 10;
                maximum *= 10;
            } else {
                var width = Math.abs(minimum) || 1;
                minimum -= width;
                maximum += width;
            }
        } else if (padding) {
            if (logarithmic) {
                var low = Math.log10(minimum);
                var high = Math.log10(maximum);
                var logPadding = (high - low) * 0.06;
                minimum = Math.pow(10, low - logPadding);
                maximum = Math.pow(10, high + logPadding);
            } else {
                var linearPadding = (maximum - minimum) * 0.07;
                minimum -= linearPadding;
                maximum += linearPadding;
            }
        }
        return [minimum, maximum];
    }

    function tickValues(domain, logarithmic) {
        var ticks = [];
        var index;
        if (!logarithmic) {
            for (index = 0; index <= 5; index += 1) {
                ticks.push(domain[0] + (domain[1] - domain[0]) * index / 5);
            }
            return ticks;
        }
        var first = Math.ceil(Math.log10(domain[0]));
        var last = Math.floor(Math.log10(domain[1]));
        var count = last - first + 1;
        var step = Math.max(1, Math.ceil(count / 7));
        for (index = first; index <= last; index += step) {
            ticks.push(Math.pow(10, index));
        }
        if (!ticks.length) {
            for (index = 0; index <= 4; index += 1) {
                ticks.push(Math.pow(10, Math.log10(domain[0]) +
                    (Math.log10(domain[1]) - Math.log10(domain[0])) *
                    index / 4));
            }
        }
        return ticks;
    }

    function formatNumber(value) {
        if (value === 0) {
            return '0';
        }
        var magnitude = Math.abs(value);
        if (magnitude >= 10000 || magnitude < 0.001) {
            return value.toExponential(1);
        }
        return Number(value.toPrecision(4)).toString();
    }

    function addLegend(article, label, colour, dash, band) {
        if (!label || article._legendKeys.has(label)) {
            return;
        }
        article._legendKeys.add(label);
        var item = document.createElement('span');
        item.className = 'legend-item';
        var swatch = document.createElement('span');
        swatch.className = 'swatch';
        if (band) {
            swatch.style.background = colour;
            swatch.style.opacity = '0.25';
            swatch.style.borderTopColor = colour;
        } else {
            swatch.style.borderTopColor = colour;
            swatch.style.borderTopStyle =
                dash && dash.indexOf('2 3') >= 0 ? 'dotted' :
                dash ? 'dashed' : 'solid';
        }
        var caption = document.createElement('span');
        caption.textContent = label;
        item.appendChild(swatch);
        item.appendChild(caption);
        article.querySelector('.legend').appendChild(item);
    }

    function panelEdges(plot, blocks) {
        var curves = plot.main.length ? plot.main : plot.ratios;
        if (!curves.length) {
            throw new Error('A plot has no curves.');
        }
        return edges(blockAt(blocks, curves[0].block));
    }

    function mainPanel(plot, blocks) {
        var range = [];
        plot.main.forEach(function (curve) {
            var block = blockAt(blocks, curve.block);
            var central = column(block, 2);
            var displayed = plot.y_axis_mode === 'LOG' ?
                central.map(function (value) { return Math.abs(value); }) :
                central.slice();
            range = range.concat(displayed);
            if (curve.statistical) {
                column(block, 3).forEach(function (error, index) {
                    range.push(displayed[index] - Math.abs(error));
                    range.push(displayed[index] + Math.abs(error));
                });
            }
            curve.uncertainties.forEach(function (uncertainty) {
                range = range.concat(column(block, uncertainty.minimum));
                range = range.concat(column(block, uncertainty.maximum));
            });
        });
        return {
            edges: panelEdges(plot, blocks),
            range: range,
            logarithmic: plot.y_axis_mode === 'LOG',
            reference: null,
            ylabel: 'cross section per bin [pb]',
            draw: function (api, article) {
                plot.main.forEach(function (curve) {
                    var block = blockAt(blocks, curve.block);
                    curve.uncertainties.forEach(function (uncertainty) {
                        if (!uncertainty.band) {
                            return;
                        }
                        api.band(edges(block),
                            column(block, uncertainty.minimum),
                            column(block, uncertainty.maximum), curve.color);
                        addLegend(article, curve.label + ', ' +
                            uncertainty.label + ' variation',
                            curve.color, '', true);
                    });
                });
                plot.main.forEach(function (curve) {
                    var block = blockAt(blocks, curve.block);
                    curve.uncertainties.forEach(function (uncertainty) {
                        if (uncertainty.band) {
                            return;
                        }
                        var dash = dashFor(uncertainty.type);
                        api.step(edges(block),
                            column(block, uncertainty.minimum),
                            curve.color, dash, 1);
                        api.step(edges(block),
                            column(block, uncertainty.maximum),
                            curve.color, dash, 1);
                        addLegend(article, curve.label + ', ' +
                            uncertainty.label + ' variation',
                            curve.color, dash, false);
                    });
                });
                plot.main.forEach(function (curve) {
                    var block = blockAt(blocks, curve.block);
                    var central = column(block, 2);
                    if (plot.y_axis_mode === 'LOG') {
                        api.step(edges(block), central.map(function (value) {
                            return value > 0 ? value : null;
                        }), curve.color, '', 1.7);
                        api.step(edges(block), central.map(function (value) {
                            return value < 0 ? Math.abs(value) : null;
                        }), curve.color, '7 4', 1.7);
                    } else {
                        api.step(edges(block), central,
                                 curve.color, '', 1.7);
                    }
                    addLegend(article, curve.label, curve.color, '', false);
                    if (curve.statistical) {
                        var displayed = plot.y_axis_mode === 'LOG' ?
                            central.map(function (value) {
                                return Math.abs(value);
                            }) : central;
                        api.errors(centres(block), displayed,
                                   column(block, 3), curve.color);
                    }
                });
            }
        };
    }

    function uncertaintyPanel(plot, blocks) {
        var range = [0];
        plot.main.forEach(function (curve) {
            var block = blockAt(blocks, curve.block);
            var central = column(block, 2);
            curve.uncertainties.forEach(function (uncertainty) {
                range = range.concat(relative(
                    column(block, uncertainty.minimum), central));
                range = range.concat(relative(
                    column(block, uncertainty.maximum), central));
            });
            if (curve.statistical) {
                column(block, 3).forEach(function (error, index) {
                    var value = central[index] ?
                        Math.abs(error) / Math.abs(central[index]) : 0;
                    range.push(-value);
                    range.push(value);
                });
            }
        });
        return {
            edges: panelEdges(plot, blocks),
            range: range,
            logarithmic: false,
            reference: 0,
            ylabel: 'relative uncertainty',
            draw: function (api, article) {
                api.reference(0);
                plot.main.forEach(function (curve) {
                    var block = blockAt(blocks, curve.block);
                    var central = column(block, 2);
                    curve.uncertainties.forEach(function (uncertainty) {
                        var minimum = relative(
                            column(block, uncertainty.minimum), central);
                        var maximum = relative(
                            column(block, uncertainty.maximum), central);
                        var label = curve.label + ', ' + uncertainty.label;
                        if (uncertainty.band) {
                            api.band(edges(block), minimum, maximum,
                                     curve.color);
                            addLegend(article, label, curve.color, '', true);
                        } else {
                            var dash = dashFor(uncertainty.type);
                            api.step(edges(block), minimum,
                                     curve.color, dash, 1);
                            api.step(edges(block), maximum,
                                     curve.color, dash, 1);
                            addLegend(article, label, curve.color, dash, false);
                        }
                    });
                    if (curve.statistical) {
                        var errors = column(block, 3).map(
                            function (error, index) {
                                return central[index] ?
                                    Math.abs(error) /
                                    Math.abs(central[index]) : 0;
                            });
                        api.errors(centres(block),
                            errors.map(function () { return 0; }),
                            errors, curve.color);
                    }
                });
            }
        };
    }

    function ratioPanel(plot, blocks) {
        var range = [1];
        plot.ratios.forEach(function (curve) {
            var block = blockAt(blocks, curve.block);
            var central = column(block, 2);
            range = range.concat(central);
            if (curve.statistical) {
                column(block, 3).forEach(function (error, index) {
                    range.push(central[index] - Math.abs(error));
                    range.push(central[index] + Math.abs(error));
                });
            }
            curve.uncertainties.forEach(function (uncertainty) {
                range = range.concat(column(block, uncertainty.minimum));
                range = range.concat(column(block, uncertainty.maximum));
            });
        });
        return {
            edges: panelEdges(plot, blocks),
            range: range,
            logarithmic: false,
            reference: 1,
            ylabel: 'ratio',
            draw: function (api, article) {
                api.reference(1);
                plot.ratios.forEach(function (curve) {
                    var block = blockAt(blocks, curve.block);
                    curve.uncertainties.forEach(function (uncertainty) {
                        var label = curve.label + ', ' + uncertainty.label;
                        if (uncertainty.band) {
                            api.band(edges(block),
                                column(block, uncertainty.minimum),
                                column(block, uncertainty.maximum),
                                curve.color);
                            addLegend(article, label, curve.color, '', true);
                        } else {
                            var dash = dashFor(uncertainty.type);
                            api.step(edges(block),
                                column(block, uncertainty.minimum),
                                curve.color, dash, 1);
                            api.step(edges(block),
                                column(block, uncertainty.maximum),
                                curve.color, dash, 1);
                            addLegend(article, label, curve.color, dash, false);
                        }
                    });
                    var central = column(block, 2);
                    api.step(edges(block), central,
                             curve.color, '', 1.5);
                    addLegend(article, curve.label, curve.color, '', false);
                    if (curve.statistical) {
                        api.errors(centres(block), central,
                                   column(block, 3), curve.color);
                    }
                });
            }
        };
    }

    function renderPanel(svg, article, plot, specification,
                         offset, panelHeight, showXAxis) {
        var left = 82;
        var right = 18;
        var top = offset + 18;
        var bottom = offset + panelHeight - (showXAxis ? 46 : 28);
        var width = 900 - left - right;
        var height = bottom - top;
        var xLogarithmic = plot.x_axis_mode === 'LOG' &&
            specification.edges.every(function (value) { return value > 0; });
        var xDomain = finiteDomain(
            specification.edges, xLogarithmic, null, false);
        var yDomain = finiteDomain(
            specification.range, specification.logarithmic,
            specification.reference, true);

        function scaled(value, domain, logarithmic) {
            if (!Number.isFinite(value) || (logarithmic && value <= 0)) {
                return null;
            }
            var low = logarithmic ? Math.log10(domain[0]) : domain[0];
            var high = logarithmic ? Math.log10(domain[1]) : domain[1];
            var current = logarithmic ? Math.log10(value) : value;
            return (current - low) / (high - low);
        }
        function x(value) {
            var result = scaled(value, xDomain, xLogarithmic);
            return result === null ? null : left + result * width;
        }
        function y(value) {
            var result = scaled(
                value, yDomain, specification.logarithmic);
            return result === null ? null : bottom - result * height;
        }

        var clipId = 'plot-clip-' + (plotSerial += 1);
        var definitions = svgElement('defs');
        var clip = svgElement('clipPath', {id: clipId});
        clip.appendChild(svgElement('rect', {
            x: left, y: top, width: width, height: height
        }));
        definitions.appendChild(clip);
        svg.appendChild(definitions);
        svg.appendChild(svgElement('rect', {
            x: left, y: top, width: width, height: height,
            fill: '#fff', stroke: '#9aa5af', 'stroke-width': 0.8
        }));

        var grid = svgElement('g', {
            stroke: '#dce1e5', 'stroke-width': 0.7
        });
        tickValues(yDomain, specification.logarithmic).forEach(
            function (value) {
                var position = y(value);
                grid.appendChild(svgElement('line', {
                    x1: left, x2: left + width,
                    y1: position, y2: position
                }));
                svg.appendChild(svgElement('text', {
                    x: left - 8, y: position + 4,
                    'text-anchor': 'end', fill: '#28323b',
                    'font-size': 11
                }, formatNumber(value)));
            });
        tickValues(xDomain, xLogarithmic).forEach(function (value) {
            var position = x(value);
            grid.appendChild(svgElement('line', {
                x1: position, x2: position,
                y1: top, y2: bottom
            }));
            if (showXAxis) {
                svg.appendChild(svgElement('text', {
                    x: position, y: bottom + 18,
                    'text-anchor': 'middle', fill: '#28323b',
                    'font-size': 11
                }, formatNumber(value)));
            }
        });
        svg.appendChild(grid);

        var dataLayer = svgElement('g', {
            'clip-path': 'url(#' + clipId + ')'
        });
        svg.appendChild(dataLayer);

        function drawStep(stepEdges, values, colour, dash, lineWidth) {
            var path = '';
            var active = false;
            values.forEach(function (value, index) {
                var x0 = x(stepEdges[index]);
                var x1 = x(stepEdges[index + 1]);
                var position = value === null ? null : y(value);
                if (x0 === null || x1 === null || position === null) {
                    active = false;
                    return;
                }
                path += (active ? ' L ' : ' M ') + x0 + ' ' + position;
                path += ' L ' + x1 + ' ' + position;
                active = true;
            });
            if (path) {
                var attributes = {
                    d: path, fill: 'none', stroke: colour,
                    'stroke-width': lineWidth || 1.4,
                    'vector-effect': 'non-scaling-stroke'
                };
                if (dash) {
                    attributes['stroke-dasharray'] = dash;
                }
                dataLayer.appendChild(svgElement('path', attributes));
            }
        }

        function drawBand(bandEdges, minimum, maximum, colour) {
            minimum.forEach(function (low, index) {
                var high = maximum[index];
                var x0 = x(bandEdges[index]);
                var x1 = x(bandEdges[index + 1]);
                var y0 = y(Math.min(low, high));
                var y1 = y(Math.max(low, high));
                if ([x0, x1, y0, y1].some(function (value) {
                        return value === null;
                    })) {
                    return;
                }
                dataLayer.appendChild(svgElement('rect', {
                    x: Math.min(x0, x1), y: Math.min(y0, y1),
                    width: Math.abs(x1 - x0),
                    height: Math.max(0.6, Math.abs(y1 - y0)),
                    fill: colour, opacity: 0.18
                }));
            });
        }

        function drawErrors(xValues, central, errors, colour) {
            var path = '';
            xValues.forEach(function (value, index) {
                var horizontal = x(value);
                var low = y(central[index] - Math.abs(errors[index]));
                var high = y(central[index] + Math.abs(errors[index]));
                if (horizontal === null || low === null || high === null) {
                    return;
                }
                path += ' M ' + horizontal + ' ' + low +
                        ' L ' + horizontal + ' ' + high +
                        ' M ' + (horizontal - 2.5) + ' ' + low +
                        ' L ' + (horizontal + 2.5) + ' ' + low +
                        ' M ' + (horizontal - 2.5) + ' ' + high +
                        ' L ' + (horizontal + 2.5) + ' ' + high;
            });
            if (path) {
                dataLayer.appendChild(svgElement('path', {
                    d: path, fill: 'none', stroke: colour,
                    'stroke-width': 0.8,
                    'vector-effect': 'non-scaling-stroke'
                }));
            }
        }

        function drawReference(value) {
            var position = y(value);
            if (position !== null) {
                dataLayer.appendChild(svgElement('line', {
                    x1: left, x2: left + width,
                    y1: position, y2: position,
                    stroke: '#737d86', 'stroke-width': 0.9
                }));
            }
        }

        specification.draw({
            step: drawStep,
            band: drawBand,
            errors: drawErrors,
            reference: drawReference
        }, article);

        svg.appendChild(svgElement('text', {
            x: 18, y: top + height / 2,
            transform: 'rotate(-90 18 ' + (top + height / 2) + ')',
            'text-anchor': 'middle', fill: '#28323b',
            'font-size': 12
        }, specification.ylabel));
        if (showXAxis) {
            svg.appendChild(svgElement('text', {
                x: left + width / 2, y: bottom + 38,
                'text-anchor': 'middle', fill: '#28323b',
                'font-size': 12
            }, 'bin'));
        }
    }

    function renderPlot(plot, blocks, index) {
        var article = document.createElement('article');
        article.id = 'histogram-' + (index + 1);
        article._legendKeys = new Set();
        var heading = document.createElement('h2');
        heading.textContent = plot.title;
        article.appendChild(heading);
        var svg = svgElement('svg', {
            viewBox: '0 0 900 400',
            role: 'img',
            'aria-label': plot.title
        });
        var legend = document.createElement('div');
        legend.className = 'legend';
        article.appendChild(svg);
        article.appendChild(legend);

        var panels = [mainPanel(plot, blocks)];
        if (plot.relative_uncertainties) {
            panels.push(uncertaintyPanel(plot, blocks));
        }
        if (plot.ratios.length) {
            panels.push(ratioPanel(plot, blocks));
        }
        var heights = panels.map(function (_, panelIndex) {
            return panelIndex === 0 ? 390 : 205;
        });
        var totalHeight = heights.reduce(function (sum, value) {
            return sum + value;
        }, 0);
        svg.setAttribute('viewBox', '0 0 900 ' + totalHeight);
        var offset = 0;
        panels.forEach(function (panel, panelIndex) {
            renderPanel(svg, article, plot, panel, offset,
                        heights[panelIndex], panelIndex === panels.length - 1);
            offset += heights[panelIndex];
        });
        return article;
    }

    try {
        var source = JSON.parse(
            document.getElementById('hwu-source').textContent);
        var plots = JSON.parse(
            document.getElementById('plot-specs').textContent);
        var blocks = parseBlocks(source);
        var main = document.getElementById('plots');
        var loading = document.getElementById('loading');
        if (loading) {
            loading.remove();
        }
        var contents = document.querySelector('#contents ol');
        plots.forEach(function (plot, index) {
            var item = document.createElement('li');
            var link = document.createElement('a');
            link.href = '#histogram-' + (index + 1);
            link.textContent = plot.title;
            item.appendChild(link);
            contents.appendChild(item);
            main.appendChild(renderPlot(plot, blocks, index));
        });
    } catch (error) {
        var target = document.getElementById('plots');
        target.textContent = '';
        var message = document.createElement('div');
        message.className = 'error';
        message.textContent = 'Could not render histogram data:\n' +
                              (error && error.message ? error.message : error);
        target.appendChild(message);
    }
}());
</script>
</body>
</html>
'''


_MATPLOTLIB_SCRIPT_BODY = r'''SOURCE_LINESTYLES = {
    'scale': '--',
    'pdf': ':',
    'merging_scale': '-.',
    'alpsfact': (0, (3, 1, 1, 1)),
}


def _load_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
        from matplotlib.backends.backend_pdf import PdfPages
    except (ImportError, RuntimeError) as error:
        raise SystemExit(
            'Matplotlib and its dependencies are required to render this '
            'histogram script: %s' % error)
    return pyplot, PdfPages


def _index_hwu_blocks(path):
    """Return (byte offset, declared bin count) for every histogram block."""
    offsets = []
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            stripped = line.lstrip()
            if not stripped.startswith(b'<histogram>'):
                continue
            fields = stripped.split(None, 2)
            if len(fields) < 2 or fields[0] != b'<histogram>':
                raise SystemExit('A malformed histogram opening tag was found.')
            try:
                n_bins = int(fields[1])
            except ValueError:
                raise SystemExit('A histogram has an invalid bin count.')
            if n_bins < 0:
                raise SystemExit('A histogram has a negative bin count.')
            offsets.append((stream.tell(), n_bins))
    if not offsets:
        raise SystemExit("No histogram blocks were found in '%s'." % path)
    return offsets


def _read_hwu_block(path, block, positions):
    """Read selected columns of one block into compact column arrays."""
    if isinstance(block, (tuple, list)):
        if len(block) != 2:
            raise SystemExit('Invalid histogram block descriptor.')
        offset, expected_rows = block
    else:
        # Backward compatibility for callers which stored old integer offsets.
        offset, expected_rows = block, None
    positions = list(positions)
    if any(not isinstance(position, int) or position < 0
           for position in positions):
        raise SystemExit('Histogram column positions must be non-negative integers.')
    positions = sorted(set(positions))
    columns = dict((position, []) for position in positions)
    rows_read = 0
    closing_tag_found = False
    with open(path, 'rb') as stream:
        stream.seek(offset)
        for line in stream:
            stripped = line.strip()
            if stripped == b'<\\histogram>':
                closing_tag_found = True
                break
            if not stripped or stripped.startswith(b'#'):
                continue
            if stripped.startswith(b'<histogram>'):
                raise SystemExit('A new histogram starts before the previous '
                                 'block is closed.')
            fields = stripped.split(b'#', 1)[0].split()
            if not fields:
                continue
            if positions and positions[-1] >= len(fields):
                raise SystemExit('Histogram block has %d columns but column %d '
                                 'was requested.' % (len(fields), positions[-1]))
            try:
                values = [float(fields[position].replace(b'D', b'E').replace(
                          b'd', b'e')) for position in positions]
            except ValueError:
                raise SystemExit('A non-numeric value was found in histogram '
                                 'row %d.'%(rows_read+1))
            if not all(math.isfinite(value) for value in values):
                raise SystemExit('A non-finite value was found in histogram '
                                 'row %d.'%(rows_read+1))
            for position, value in zip(positions, values):
                columns[position].append(value)
            rows_read += 1
    if not closing_tag_found:
        raise SystemExit('A histogram block is missing its closing tag.')
    if expected_rows is not None and rows_read != expected_rows:
        raise SystemExit('Histogram block contains %d rows but declares %d.'%
                         (rows_read, expected_rows))
    if rows_read == 0:
        raise SystemExit('An empty histogram block was encountered.')
    return {'columns': columns}


def _required_columns(plot):
    """Map each block on one PDF page to the columns it actually uses."""
    required = {}
    for curve in plot['main'] + plot['ratios']:
        positions = required.setdefault(curve['block'], set([0, 1, 2]))
        if curve['statistical']:
            positions.add(3)
        for uncertainty in curve['uncertainties']:
            positions.update([uncertainty['minimum'], uncertainty['maximum']])
    return required


def _load_plot_blocks(path, offsets, plot):
    blocks = {}
    for block_index, positions in _required_columns(plot).items():
        if block_index >= len(offsets):
            raise SystemExit('Histogram block %d is missing.' % block_index)
        blocks[block_index] = _read_hwu_block(
            path, offsets[block_index], positions)
    return blocks


def _column(block, position):
    return block['columns'][position]


def _edges(block):
    return _column(block, 0) + [_column(block, 1)[-1]]


def _centres(block):
    return [(lower + upper) / 2.0 for lower, upper in
            zip(_column(block, 0), _column(block, 1))]


def _step_values(values):
    return list(values) + [values[-1]]


def _draw_step(axis, edges, values, **options):
    return axis.step(edges, _step_values(values), where='post', **options)


def _draw_band(axis, edges, minimum, maximum, **options):
    lower = [min(low, high) for low, high in zip(minimum, maximum)]
    upper = [max(low, high) for low, high in zip(minimum, maximum)]
    return axis.fill_between(edges, _step_values(lower), _step_values(upper),
                             step='post', **options)


def _relative(values, central):
    return [(value / reference - 1.0) if reference else 0.0
            for value, reference in zip(values, central)]


def _finish_axis(axis):
    axis.grid(True, which='both', alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique and label != '_nolegend_':
            unique[label] = handle
    if unique:
        axis.legend(list(unique.values()), list(unique.keys()), fontsize='small')


def _draw_main(axis, plot, blocks):
    positive_edges = True
    for curve in plot['main']:
        block = blocks[curve['block']]
        edges = _edges(block)
        centres = _centres(block)
        central = _column(block, 2)
        positive_edges = positive_edges and all(value > 0.0 for value in edges)
        if plot['y_axis_mode'] == 'LOG':
            # Match the gnuplot convention: positive contributions are solid,
            # while negative contributions are shown by their magnitude using
            # the same colour and a dashed line.
            positive = [value if value > 0.0 else float('nan')
                        for value in central]
            negative = [abs(value) if value < 0.0 else float('nan')
                        for value in central]
            _draw_step(axis, edges, positive, color=curve['color'],
                       label=curve['label'], linewidth=1.5)
            if any(value < 0.0 for value in central):
                _draw_step(axis, edges, negative, color=curve['color'],
                           linestyle='--', label='_nolegend_', linewidth=1.5)
            displayed_central = [abs(value) if value else float('nan')
                                 for value in central]
        else:
            _draw_step(axis, edges, central, color=curve['color'],
                       label=curve['label'], linewidth=1.5)
            displayed_central = central
        if curve['statistical']:
            axis.errorbar(centres, displayed_central,
                          yerr=[abs(value) for value in _column(block, 3)],
                          fmt='none', color=curve['color'], capsize=1.5,
                          linewidth=0.8)
        for uncertainty in curve['uncertainties']:
            minimum = _column(block, uncertainty['minimum'])
            maximum = _column(block, uncertainty['maximum'])
            label = '%s, %s variation' % (curve['label'], uncertainty['label'])
            if uncertainty['band']:
                _draw_band(axis, edges, minimum, maximum,
                           color=curve['color'], alpha=0.18, label=label)
            else:
                linestyle = SOURCE_LINESTYLES.get(uncertainty['type'], '--')
                _draw_step(axis, edges, minimum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0, label=label)
                _draw_step(axis, edges, maximum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0,
                           label='_nolegend_')

    # Match gnuplot: the histogram's axis mode, rather than empty bins,
    # determines whether the main y axis is logarithmic.
    if plot['y_axis_mode'] == 'LOG':
        axis.set_yscale('log')
    if plot['x_axis_mode'] == 'LOG' and positive_edges:
        axis.set_xscale('log')
    axis.set_ylabel(r'$\sigma$ per bin [pb]')
    axis.text(1.01, 0.02, 'MadGraph5_aMC@NLO', transform=axis.transAxes,
              rotation=90, va='bottom', family='monospace', fontsize='small')
    _finish_axis(axis)


def _draw_relative_uncertainties(axis, plot, blocks):
    for curve in plot['main']:
        block = blocks[curve['block']]
        edges = _edges(block)
        centres = _centres(block)
        central = _column(block, 2)
        for uncertainty in curve['uncertainties']:
            minimum = _relative(_column(block, uncertainty['minimum']), central)
            maximum = _relative(_column(block, uncertainty['maximum']), central)
            label = '%s, %s' % (curve['label'], uncertainty['label'])
            if uncertainty['band']:
                _draw_band(axis, edges, minimum, maximum,
                           color=curve['color'], alpha=0.18, label=label)
            else:
                linestyle = SOURCE_LINESTYLES.get(uncertainty['type'], '--')
                _draw_step(axis, edges, minimum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0, label=label)
                _draw_step(axis, edges, maximum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0,
                           label='_nolegend_')
        if curve['statistical']:
            statistical = [(abs(error) / abs(value)) if value else 0.0
                           for error, value in zip(_column(block, 3), central)]
            axis.errorbar(centres, [0.0] * len(centres), yerr=statistical,
                          fmt='none', color=curve['color'], capsize=1.5,
                          linewidth=0.8)
    axis.axhline(0.0, color='0.45', linewidth=0.8)
    axis.set_ylabel('rel. unc.')
    _finish_axis(axis)


def _draw_ratios(axis, plot, blocks):
    for curve in plot['ratios']:
        block = blocks[curve['block']]
        edges = _edges(block)
        centres = _centres(block)
        central = _column(block, 2)
        _draw_step(axis, edges, central, color=curve['color'],
                   label=curve['label'], linewidth=1.3)
        if curve['statistical']:
            axis.errorbar(centres, central,
                          yerr=[abs(value) for value in _column(block, 3)],
                          fmt='none', color=curve['color'], capsize=1.5,
                          linewidth=0.8)
        for uncertainty in curve['uncertainties']:
            minimum = _column(block, uncertainty['minimum'])
            maximum = _column(block, uncertainty['maximum'])
            label = '%s, %s' % (curve['label'], uncertainty['label'])
            if uncertainty['band']:
                _draw_band(axis, edges, minimum, maximum,
                           color=curve['color'], alpha=0.18, label=label)
            else:
                linestyle = SOURCE_LINESTYLES.get(uncertainty['type'], '--')
                _draw_step(axis, edges, minimum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0, label=label)
                _draw_step(axis, edges, maximum, color=curve['color'],
                           linestyle=linestyle, linewidth=1.0,
                           label='_nolegend_')
    axis.axhline(1.0, color='0.45', linewidth=0.8)
    axis.set_ylabel('ratio')
    _finish_axis(axis)


def _open_pdf(path):
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except (AttributeError, OSError):
        pass


def main(pdf_path=None):
    pyplot, PdfPages = _load_matplotlib()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, DATA_FILE)
    if pdf_path is None:
        pdf_path = os.path.join(script_directory, PDF_FILE)
    else:
        pdf_path = os.path.abspath(pdf_path)
    block_offsets = _index_hwu_blocks(data_path)

    with PdfPages(pdf_path) as pdf:
        for plot in PLOTS:
            # Load only this page's blocks and only the columns referenced by
            # its central curves and selected uncertainties.  The data is
            # released when the loop advances to the next plot.
            blocks = _load_plot_blocks(data_path, block_offsets, plot)
            row_count = 1 + int(plot['relative_uncertainties']) + \
                        int(bool(plot['ratios']))
            heights = [4.0]
            if plot['relative_uncertainties']:
                heights.append(1.5)
            if plot['ratios']:
                heights.append(1.5)
            figure, axes = pyplot.subplots(
                row_count, 1, sharex=True, figsize=(8.3, 5.8+1.4*(row_count-1)),
                gridspec_kw={'height_ratios': heights})
            if row_count == 1:
                axes = [axes]
            else:
                axes = list(axes)

            axis_index = 0
            _draw_main(axes[axis_index], plot, blocks)
            axis_index += 1
            if plot['relative_uncertainties']:
                _draw_relative_uncertainties(axes[axis_index], plot, blocks)
                axis_index += 1
            if plot['ratios']:
                _draw_ratios(axes[axis_index], plot, blocks)
            axes[-1].set_xlabel('bin')
            figure.suptitle(plot['title'])
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            pdf.savefig(figure)
            pyplot.close(figure)

    print("Wrote Matplotlib histogram PDF to '%s'." % pdf_path)
    if OPEN_AFTER_RENDER:
        _open_pdf(pdf_path)
    return pdf_path


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
'''


def _matplotlib_available():
    """Return whether Matplotlib is installed for this Python interpreter."""

    try:
        from importlib.util import find_spec
        return find_spec('matplotlib') is not None
    except Exception:
        return False


def render_histogram_output(path, stdout=None, stderr=None):
    """Render an extensionless HwU output path with the available backend.

    GnuPlot remains the preferred renderer.  The generated Matplotlib script
    is used only when no ``gnuplot`` executable is available on ``PATH``.
    Return ``(None, None)`` without rendering when neither backend exists.
    """

    output_path = os.path.abspath(path)
    output_directory = os.path.dirname(output_path)
    output_base_name = os.path.basename(output_path)
    try:
        gnuplot = misc.which('gnuplot')
    except (KeyError, OSError):
        gnuplot = None
    if gnuplot:
        command = [os.path.abspath(gnuplot), output_base_name+'.gnuplot']
        backend = 'gnuplot'
    else:
        if not _matplotlib_available():
            return None, None
        command = [sys.executable, output_path+'.py']
        backend = 'matplotlib'
    return backend, subprocess.call(command, cwd=output_directory,
                                     stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    main_doc = \
    """ For testing and standalone use. Usage:
        python histograms.py <.HwU input_file_path_1> <.HwU input_file_path_2> ... --out=<output_file_path.format> <options>
        Where <options> can be a list of the following: 
           '--help'          See this message.
           '--gnuplot' or '' output GnuPlot, Matplotlib, and direct HTML renderers; prefer GnuPlot for PDF rendering.
           '--matplotlib'    output the histograms and a Python script which renders them to PDF.
           '--html'          output the histograms directly in a standalone HTML/SVG page.
           '--HwU'           to output the histograms read to the raw HwU source.
           '--types=<type1>,<type2>,...' to keep only the type<i> when importing histograms.
           '--titles=<title1>,<title2>,...' to keep only the titles which have any of 'title<i>' in them (not necessarily equal to them)
           '--n_ratios=<integer>' Specifies how many curves must be considerd for the ratios.
           '--no_open'       Turn off automatic processing/opening of plotting output.
           '--show_full'     to show the complete output of what was read.
           '--show_short'    to show a summary of what was read.
           '--simple_ratios' to turn off correlations and error propagation in the ratio.
           '--colours=<colour1>,<colour2>,...' to assign a non-default colour to GnuPlot histograms (max 8 colours)
           '--sum'           To sum all identical histograms together
           '--average'       To average over all identical histograms
           '--rebin=<n>'     Rebin the plots by merging n-consecutive bins together.  
           '--memory_limit=<MiB>' Maximum resident memory used by dense --sum/--average buffers before using temporary mmap storage (default: 256).
           '--assign_types=<type1>,<type2>,...' to assign a type to all histograms of the first, second, etc... files loaded.
           '--multiply=<fact1>,<fact2>,...' to multiply all histograms of the first, second, etc... files by the fact1, fact2, etc...
           '--no_suffix'     Do no add any suffix (like '#1, #2, etc..) to the histograms types.
           '--lhapdf-config=<PATH_TO_LHAPDF-CONFIG>' give path to lhapdf-config to compute PDF certainties using LHAPDF (only for lhapdf6)
           '--jet_samples=[int1,int2]' Specifies what jet samples to keep. 'None' is the default and keeps them all.
           '--central_only'  This option specifies to disregard all extra weights, so as to make it possible
                             to take the ratio of plots with different extra weights specified.             
           '--keep_all_weights' This option specifies to keep in the HwU produced all the weights, even
                                those which are not known (i.e. that is scale, PDF or merging variation)
        For chosing what kind of variation you want to see on your plot, you can use the following options
           '--no_<type>'                   Turn off the plotting of variations of the chosen type
           '--only_<type>'                 Turn on only the plotting of variations of the chosen type
           '--variations=['<type1>',...]'  Turn on only the plotting of the variations of the list of chosen types
           '--band=['<type1>',...]'        Chose for which variations one should use uncertainty bands as opposed to lines
        The types can be: pdf, scale, stat, merging or alpsfact
        For the last two options one can use ...=all to automatically select all types.
        
        When parsing an XML-formatted plot source output by the Pythia8 driver, the file names can be appended 
        options as suffixes separated by '|', as follows:
           python histograms.py <XML_source_file_name>@<option1>@<option2>@etc..
        These options can be
           'run_id=<integer>'      Specifies the run_ID from which the plots should be loaded.
                                   By default, the first run is considered and the ones that follow are ignored.
           'merging_scale=<float>' This option allows to specify to import only the plots corresponding to a specific 
                                   value for the merging scale.
                                   A value of -1 means that only the weights with the same merging scale as the central weight are kept.
                                   By default, all weights are considered.
    """

    possible_options=['--help', '--gnuplot', '--matplotlib', '--html', '--HwU', '--types','--n_ratios',\
                      '--no_open','--show_full','--show_short','--simple_ratios','--sum','--average','--rebin',  \
                      '--assign_types','--multiply','--no_suffix', '--out', '--jet_samples', 
                      '--no_scale','--no_pdf','--no_stat','--no_merging','--no_alpsfact',
                      '--only_scale','--only_pdf','--only_stat','--only_merging','--only_alpsfact',
                      '--variations','--band','--central_only', '--lhapdf-config','--titles',
                      '--keep_all_weights','--colours','--memory_limit']
    n_ratios   = -1
    uncertainties = ['scale','pdf','statistical','merging_scale','alpsfact']
    # The list of type of uncertainties for which to use bands. None is a 'smart' default
    use_band      = None
    auto_open = True
    ratio_correlations = True
    consider_reweights = ['pdf','scale','murmuf_scales','merging_scale','alpsfact']

    def log(msg):
        print("histograms.py :: %s"%str(msg))
    
    if '--help' in sys.argv or len(sys.argv)==1:
        log('\n\n%s'%main_doc)
        sys.exit(0)

    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            if arg.split('=')[0] not in possible_options:
                log('WARNING: option "%s" not valid. It will be ignored' % arg)

    arg_string=' '.join(sys.argv)

    OutName = ""
    for arg in sys.argv[1:]:
        if arg.startswith('--out='):
            OutName = arg[6:]

    accepted_types = []
    for arg in sys.argv[1:]:
        if arg.startswith('--types='):
            accepted_types = [(type if type!='None' else None) for type in \
                                                             arg[8:].split(',')]

    accepted_titles = []
    for arg in sys.argv[1:]:
        if arg.startswith('--titles='):
            accepted_titles = [(type if type!='None' else None) for type in \
                                                             arg[9:].split(',')]

    assigned_types = []
    for arg in sys.argv[1:]:
        if arg.startswith('--assign_types='):
            assigned_types = [(type if type!='None' else None) for type in \
                                                             arg[15:].split(',')]

    assigned_colours = []
    for arg in sys.argv[1:]:
        if arg.startswith('--colours='):
            assigned_colours = [(colour if colour!='None' else None) for colour in \
                                                             arg[10:].split(',')]

    jet_samples_to_keep = None
    
    lhapdfconfig = 'lhapdf-config'
    for arg in sys.argv[1:]:
        if arg.startswith('--lhapdf-config='):
            lhapdfconfig = arg[16:]

    no_suffix = False
    if '--no_suffix' in sys.argv:
        no_suffix = True
    
    if '--central_only' in sys.argv:
        consider_reweights = []

    if '--keep_all_weights' in sys.argv:
        consider_reweights = 'ALL'

    for arg in sys.argv[1:]:
        if arg.startswith('--n_ratios='):
            n_ratios = int(arg[11:])

    if '--no_open' in sys.argv:
        auto_open = False

    variation_type_map={'scale':'scale','merging':'merging_scale','pdf':'pdf',
                        'stat':'statistical','alpsfact':'alpsfact'}

    for arg in sys.argv:
        try:
            opt, value = arg.split('=')
        except ValueError:
            continue
        if opt=='--jet_samples':
            jet_samples_to_keep = eval(value)
        if opt=='--variations':
            uncertainties=[variation_type_map[type] for type in eval(value,
                         dict([(key,key) for key in variation_type_map.keys()]+
                                          [('all',list(variation_type_map.keys()))]))]
        if opt=='--band':
            use_band=[variation_type_map[type] for type in eval(value,
                         dict([(key,key) for key in variation_type_map.keys()]+
        [('all',[type for type in variation_type_map.keys() if type!='stat'])]))]

    if '--simple_ratios' in sys.argv:
        ratio_correlations = False
    
    for arg in sys.argv:
        if arg.startswith('--no_') and arg not in ['--no_open','--no_suffix']:
            uncertainties.remove(variation_type_map[arg[5:]])
        if arg.startswith('--only_'):
            uncertainties= [variation_type_map[arg[7:]]]
            break         

    # Now remove from the weights considered all those not deemed necessary
    # in view of which uncertainties are selected
    if isinstance(consider_reweights, list):
        naming_map={'pdf':'pdf','scale':'scale',
                  'merging_scale':'merging_scale','alpsfact':'alpsfact'}
        for key in naming_map:
            if (not key in uncertainties) and (naming_map[key] in consider_reweights):
                consider_reweights.remove(naming_map[key])

    n_files    = len([_ for _ in sys.argv[1:] if not _.startswith('--')])
    histo_norm = [1.0]*n_files

    for arg in sys.argv[1:]:
        if arg.startswith('--multiply='):
            histo_norm = [(float(fact) if fact!='' else 1.0) for fact in \
                                                arg[11:].split(',')]

    if '--average' in sys.argv:
        histo_norm = [hist/float(n_files) for hist in histo_norm]

    n_rebin = 1
    for arg in sys.argv[1:]:
        if arg.startswith('--rebin='):
            n_rebin = int(arg[8:])

    try:
        memory_limit_mib = float(os.environ.get(
                                      'MG5_HISTOGRAM_MEMORY_MB', '256'))
    except ValueError:
        raise MadGraph5Error('MG5_HISTOGRAM_MEMORY_MB must be a number.')
    for arg in sys.argv[1:]:
        if arg.startswith('--memory_limit='):
            memory_limit_mib = float(arg.split('=', 1)[1])
    if memory_limit_mib < 0.0:
        raise MadGraph5Error('--memory_limit must be non-negative.')

    streaming_aggregation = any(option in sys.argv
                                for option in ['--sum', '--average'])
    dense_aggregator = StreamingHwUAggregator(
                              memory_limit=int(memory_limit_mib*1024*1024)) \
                       if streaming_aggregation else None
        
    log("=======")
    histo_list = HwUList([])
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            break
        log("Loading histograms from '%s'."%arg)
        if OutName=="":
            OutName = os.path.basename(arg).split('.')[0]+'_output'
        # Make sure to process the potential XML options appended to the filename
        file_specification = arg.split('@')
        filename = file_specification.pop(0)
        file_options = {}
        for option in file_specification:
            opt, value = option.split('=')
            if opt=='run_id':
                file_options[opt]=int(value)
            elif opt=='merging_scale':
                file_options[opt]=float(value)
            else:
                log("Unreckognize file option '%s'."%option)
                sys.exit(1)

        if streaming_aggregation:
            dense_aggregator.add_file(filename, factor=histo_norm[i],
                accepted_types_order=accepted_types,
                consider_reweights=consider_reweights,
                accepted_titles=accepted_titles, **file_options)
            continue

        new_histo_list = HwUList(filename, accepted_types_order=accepted_types,
                          consider_reweights=consider_reweights, **file_options)
        # We filter now the diagrams whose title doesn't match the constraints
        if len(accepted_titles)>0:
            new_histo_list = HwUList(histo for histo in new_histo_list if
                                 any(t in histo.title for t in accepted_titles))
        for histo in new_histo_list:
            if no_suffix or n_files==1 or \
                              any(_ in sys.argv for _ in ['--sum','--average']):
                continue
            if not histo.type is None:
                histo.type += '|'
            else:
                histo.type = ''
            # Firs option is to give a bit of the name of the source HwU file.     
            #histo.type += " %s, #%d"%\
            #                       (os.path.basename(arg).split('.')[0][:3],i+1)
            # But it is more elegant to give just the number.
            # Overwrite existing number if present. We assume here that one never
            # uses the '#' in its custom-defined types, which is a fair assumptions.
            try:
                suffix = assigned_types[i]
            except IndexError:
                suffix = "#%d"%(i+1)
            try:
                histo.type = histo.type[:histo.type.index('#')] + suffix
            except ValueError:
                histo.type += suffix

        if i==0 or all(_ not in ['--sum','--average'] for _ in sys.argv):
            for j,hist in enumerate(new_histo_list):
                new_histo_list[j]=hist*histo_norm[i]
            histo_list.extend(new_histo_list)
            continue
        
        if any(_ in sys.argv for _ in ['--sum','--average']):
            for j, hist in enumerate(new_histo_list):
                 # First make sure the plots have the same weight labels and such
                 if not hist.test_plot_compability(histo_list[j]):
                     raise MadGraph5Error("Histograms at position %d from the"%j+
                       " input files are not compatible and cannot be combined.")
                 # Now let the histogram module do the magic and add them.
                 histo_list[j] += hist*histo_norm[i]
    histogram_count = len(dense_aggregator) if streaming_aggregation \
                                              else len(histo_list)
    log("A total of %i histograms were found."%histogram_count)
    log("=======")
    
    if n_rebin > 1 and not streaming_aggregation:
        for hist in histo_list:
            hist.rebin(n_rebin)

    def output_histograms(path, **options):
        if streaming_aggregation:
            return dense_aggregator.output(path, n_rebin=n_rebin, **options)
        return histo_list.output(path, **options)

    if '--html' in sys.argv:
        output_histograms(OutName, format='html',
            number_of_ratios=n_ratios,
            uncertainties=uncertainties,
            ratio_correlations=ratio_correlations,
            arg_string=arg_string,
            jet_samples_to_keep=jet_samples_to_keep,
            use_band=use_band,
            lhapdfconfig=lhapdfconfig,
            assigned_colours=assigned_colours)
        log("%d histograms have been output directly at "
            "'%s.[HwU|html]'."%(histogram_count, OutName))
        sys.exit(0)

    if '--matplotlib' in sys.argv:
        output_histograms(OutName, format='matplotlib',
            number_of_ratios=n_ratios,
            uncertainties=uncertainties,
            ratio_correlations=ratio_correlations,
            arg_string=arg_string,
            jet_samples_to_keep=jet_samples_to_keep,
            use_band=use_band,
            auto_open=auto_open,
            lhapdfconfig=lhapdfconfig,
            assigned_colours=assigned_colours)
        log("%d histograms have been output in the matplotlib format at "
            "'%s.[HwU|py]'."%(histogram_count, OutName))
        if auto_open:
            return_code = subprocess.call([sys.executable, OutName+'.py'])
            if return_code != 0:
                log("Automatic processing of the matplotlib script failed. "
                    "Try the command by hand:\n%s %s.py"%
                    (sys.executable, OutName))
        sys.exit(0)

    if '--gnuplot' in sys.argv or all(
                arg not in ['--HwU','--matplotlib','--html'] for arg in sys.argv):
        # Where the magic happens:
        output_histograms(OutName, format='gnuplot',
            number_of_ratios = n_ratios, 
            uncertainties=uncertainties, 
            ratio_correlations=ratio_correlations,
            arg_string=arg_string, 
            jet_samples_to_keep=jet_samples_to_keep,
            use_band=use_band,
            auto_open=auto_open,
            lhapdfconfig=lhapdfconfig,
            assigned_colours=assigned_colours)
        # Tell the user that everything went for the best
        log("%d histograms have been output at " % histogram_count+\
                             "'%s.[HwU|gnuplot|py|html]'." % OutName)
        if auto_open:
            try:
                backend, return_code = render_histogram_output(
                                      OutName, stderr=subprocess.PIPE)
            except Exception:
                if misc.which('gnuplot'):
                    command = 'gnuplot %s.gnuplot'%OutName
                else:
                    command = '%s %s.py'%(sys.executable, OutName)
                log("Automatic processing of the histogram output failed. "+
                    "Try the command by hand:\n%s"%command)
            else:
                if backend is None:
                    log("Neither GnuPlot nor Matplotlib is available; no PDF "
                        "was created. The HTML plots remain available.")
                    sys.exit(0)
                if return_code == 0:
                    sys.exit(0)
                if backend == 'gnuplot':
                    command = 'gnuplot %s.gnuplot'%OutName
                else:
                    command = '%s %s.py'%(sys.executable, OutName)
                log("Automatic processing with %s failed. Try the command "
                    "by hand:\n%s"%(backend, command))

    if '--HwU' in sys.argv:
        log("Histograms data has been output in the HwU format at "+\
                                              "'%s.HwU'."%OutName)
        output_histograms(OutName, format='HwU')
        sys.exit(0)
    
    if '--show_short' in sys.argv or '--show_full' in sys.argv:
        if streaming_aggregation:
            histograms_to_show = itertools.chain.from_iterable(
                dense_aggregator.iter_hwu_groups(n_rebin=n_rebin))
        else:
            histograms_to_show = histo_list
        for i, histo in enumerate(histograms_to_show):
            if i!=0:
                log('-------')
            log(histo.nice_string(short=(not '--show_full' in sys.argv)))
    log("=======")

######## Routine from https://gist.github.com/thriveth/8352565 
######## To fill for histograms data in matplotlib
def fill_between_steps(x, y1, y2=0, h_align='right', ax=None, **kwargs):
    ''' Fills a hole in matplotlib: fill_between for step plots.
    Parameters :
    ------------
    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.
    **kwargs will be passed to the matplotlib fill_between() function.
    '''
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()


    # First, duplicate the x values
    #duplicate the info # xx = numpy.repeat(2)[1:] 
    xx= []; [(xx.append(d),xx.append(d)) for d in x]; xx = xx[1:]
    # Now: the average x binwidth
    xstep = x[1] -x[0]
    # Now: add one step at end of row.
    xx.append(xx[-1] + xstep)

    # Make it possible to change step alignment.
    if h_align == 'mid':
        xx = [X-xstep/2. for X in xx]
    elif h_align == 'right':
        xx = [X-xstep for X in xx]

    # Also, duplicate each y coordinate in both arrays
    yy1 = []; [(yy1.append(d),yy1.append(d)) for d in y1]
    if isinstance(y1, list):
        yy2 = []; [(yy2.append(d),yy2.append(d)) for d in y2]
    else:
        yy2=y2
    if len(yy2) != len(yy1):
        yy2 = []; [(yy2.append(d),yy2.append(d)) for d in y2]
        
    # now to the plotting part:
    ax.fill_between(xx, yy1, y2=yy2, **kwargs)

    return ax
######## end routine from https://gist.github.com/thriveth/835256

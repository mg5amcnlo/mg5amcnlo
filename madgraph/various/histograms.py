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

import array
import copy
import fractions
import itertools
import logging
import math
import os
import re
import sys

logger = logging.getLogger("madgraph.various.histograms")

try:
    # import from madgraph directory
    import madgraph.various.misc as misc
    from madgraph import MadGraph5Error

except ImportError, error:
    logger.debug(error)
    # import from madevent directory
    import internal.misc as misc    
    from internal import MadGraph5Error

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
  
    def __init__(self, boundaries=(0.0,0.0), wgts={'central':0.0}):
        """ Initializes an empty bin, necessarily with boundaries. """

        self.boundaries = boundaries
        self.wgts       = wgts
  
    def __setattr__(self, name, value):
        if name=='boundaries':
            if not isinstance(value, tuple):
                raise MadGraph5Error, "Argument '%s' for bin property "+\
                                        "'boundaries' must be a tuple."%str(value)
            else:
                for coordinate in value:
                    if isinstance(coordinate, tuple):
                        for dim in coordinate:
                            if not isinstance(dim, float):
                                raise MadGraph5Error, "Coordinate '%s' of the bin"+\
                                  " boundary '%s' must be a float."%str(dim,value)
                    elif not isinstance(coordinate, float):
                        raise MadGraph5Error, "Element '%s' of the bin boundaries"+\
                                          " specified must be a float."%str(bound)
        elif name=='wgts':
            if not isinstance(value, dict):
                raise MadGraph5Error, "Argument '%s' for bin uncertainty "+\
                                          "'wgts' must be a dictionary."%str(value)
            if not 'central' in value.keys(): 
                raise MadGraph5Error, "The keys of the dictionary specifying "+\
                    "the weights of the bin must include the keyword 'central'."\
                                                                         %str(value)
            for val in value.values():
                if not isinstance(val,float):
                    raise MadGraph5Error, "The bin weight value '%s' is not a "+\
                                                                 "float."%str(val)   
   
        super(Bin, self).__setattr__(name,value)
        
    def get_weight(self, key='central'):
        """ Accesses a specific weight from this bin."""
        try:
            return self.wgts[key]
        except KeyError:
            raise MadGraph5Error, "Weight with ID '%s' is not defined for"+\
                                                            " this bin"%str(key)
                                                            
    def set_weight(self, wgt, key='central'):
        """ Accesses a specific weight from this bin."""
        
        # an assert is used here in this intensive function, so as to avoid 
        # slow-down when not in debug mode.
        assert(isinstance(wgt, float))
           
        try:
            self.wgts[key] = wgt
        except KeyError:
            raise MadGraph5Error, "Weight with ID '%s' is not defined for"+\
                                                            " this bin"%str(key)                                                

    def nice_string(self, order=None, short=True):
        """ Nice representation of this Bin. 
        One can order the weight according to the argument if provided."""
        
        res     = ["Bin boundaries : %s"%str(self.boundaries)]
        if not short:
            res.append("Bin weights    :")
            if order is None:
                label_list = self.wgts.keys()
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
            raise MadGraph5Error, 'The two bins to combine have'+\
         ' different boundaries, %s!=%s.'%(str(binA.boundaries),str(binB.boundaries))
        res_bin.boundaries = binA.boundaries
        
        try:
            res_bin.wgts = func(binA.wgts, binB.wgts)
        except Exception as e:
            raise MadGraph5Error, "When combining two bins, the provided"+\
              " function '%s' triggered the following error:\n\"%s\"\n"%\
              (func.__name__,str(e))+" when combining the following two bins:\n"+\
              binA.nice_string(short=False)+"\n and \n"+binB.nice_string(short=False)

        return res_bin

class BinList(histograms_PhysicsObjectList):
    """ A class implementing features related to a list of Bins. """

    def __init__(self, list = [], weight_labels = None):
        """ Initialize a list of Bins """
        
        self.weight_labels = weight_labels
        super(BinList, self).__init__(list)

    def is_valid_element(self, obj):
        """Test whether specified object is of the right type for this list."""

        return isinstance(obj, Bin)
    
    def __setattr__(self, name, value):
        if name=='weight_labels':
            if not value is None and not isinstance(value, list):
                raise MadGraph5Error, "Argument '%s' for BinList property '%s'"\
                                           %(str(value),name)+' must be a list.'
            elif not value is None:
                for label in value:
                    if all((not isinstance(label,cls)) for cls in [str, int, tuple]):
                        raise MadGraph5Error, "Argument '%s' for BinList property '%s'"\
                                 %(str(value),name)+' must be a string, an '+\
                                                  'integer or a tuple of float.'
                    if isinstance(label, tuple):
                        for elem in label:
                            if not isinstance(elem, float):
                                raise MadGraph5Error, "Argument "+\
                            "'%s' for BinList property '%s'"%(str(value),name)+\
                           ' can be a tuple, but it must be filled with floats.'
                                
   
        super(BinList, self).__setattr__(name, value)    
            
    def append(self, object):
        """Appends an element, but test if valid before."""
        
        super(BinList,self).append(object)    
        # Assign the weight labels to those of the first bin added
        if len(self)==1 and self.weight_labels is None:
            self.weight_labels = object.wgts.keys()

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
                 x_axis_mode = 'LIN', y_axis_mode = 'LOG'):
        """ Initializes an empty histogram, possibly specifying 
              > a title 
              > a number of dimensions
              > a bin content
        """
        
        self.title       = title
        self.dimension   = n_dimensions
        self.bins        = BinList([])
        self.type        = type
        self.x_axis_mode = x_axis_mode
        self.y_axis_mode = y_axis_mode        
   
    def __setattr__(self, name, value):
        if name=='title':
            if not isinstance(value, str):
                raise MadGraph5Error, "Argument '%s' for the histogram property "+\
                                          "'title' must be a string."%str(value)
        elif name=='dimension':
            if not isinstance(value, int):
                raise MadGraph5Error, "Argument '%s' for histogram property "+\
                                    "'dimension' must be an integer."%str(value)
            if self.allowed_dimensions and value not in self.allowed_dimensions:
                raise MadGraph5Error, "%i-Dimensional histograms not supported "\
                         %value+"by class '%s'. Supported dimensions are '%s'."\
                              %(self.__class__.__name__,self.allowed_dimensions)
        elif name=='bins':
            if not isinstance(value, BinList):
                raise MadGraph5Error, "Argument '%s' for histogram property "+\
                                        "'bins' must be a BinList."%str(value)
            else:
                for bin in value:
                    if not isinstance(bin, Bin):
                        raise MadGraph5Error, "Element '%s' of the "%str(bin)+\
                                  " histogram bin list specified must be a bin."
        elif name=='type':
            if not (value is None or value in self.allowed_types or 
                                                        self.allowed_types==[]):
                raise MadGraph5Error, "Argument '%s' for histogram"%str(value)+\
                             " property 'type' must be a string in %s or None."\
                                         %([str(t) for t in self.allowed_types])
        elif name in ['x_axis_mode','y_axis_mode']:
            if not value in self.allowed_axis_modes:
                raise MadGraph5Error, "Attribute '%s' of the histogram"%str(name)+\
                  " must be in [%s], ('%s' given)"%(str(self.allowed_axis_modes),
                                                                     str(value))
                                        
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
            raise MadGraph5Error, 'The two histograms to combine have a '+\
         'different number of bins, %d!=%d.'%(len(histoA.bins),len(histoB.bins))

        if histoA.dimension!=histoB.dimension:
            raise MadGraph5Error, 'The two histograms to combine have a '+\
         'different dimensions, %d!=%d.'%(histoA.dimension,histoB.dimension)            
        res_histogram.dimension = histoA.dimension
    
        for i, bin in enumerate(histoA.bins):
            res_histogram.bins.append(Bin.combine(bin, histoB.bins[i],func))
        
        # Reorder the weight labels as in the original histogram and add at the
        # end the new ones which resulted from the combination, in a sorted order
        res_histogram.bins.weight_labels = [label for label in histoA.bins.\
                weight_labels if label in res_histogram.bins.weight_labels] + \
                sorted([label for label in res_histogram.bins.weight_labels if\
                                       label not in histoA.bins.weight_labels])
                
        
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
                             wgtsB['central'])**2)/wgtsB['central']
        
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
                if new_wgts[label]>1.0e+10:
                    print "stat_error_operation is ",stat_error_operation.__name__
                    print " inputs were ",wgt, wgtsB[label]
                    print "for label", label
        
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
                                                           (lambda a: a*factor))

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
            self.alter_weights(Histogram.OFFSET(float(other)))
            return self
        else:
            return NotImplemented, 'Histograms can only be added to other '+\
              ' histograms or scalars.'

    def __sub__(self, other):
        """ Overload the subtraction function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.SUBTRACT)
        elif isinstance(other, int) or isinstance(other, float):
            self.alter_weights(Histogram.OFFSET(-float(other)))
            return self
        else:
            return NotImplemented, 'Histograms can only be subtracted to other '+\
              ' histograms or scalars.'
    
    def __mul__(self, other):
        """ Overload the multiplication function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.MULTIPLY)
        elif isinstance(other, int) or isinstance(other, float):
            self.alter_weights(Histogram.RESCALE(float(other)))
            return self
        else:
            return NotImplemented, 'Histograms can only be multiplied to other '+\
              ' histograms or scalars.'

    def __div__(self, other):
        """ Overload the multiplication function. """
        if isinstance(other, Histogram):
            return self.__class__.combine(self,other,Histogram.DIVIDE)
        elif isinstance(other, int) or isinstance(other, float):
            self.alter_weights(Histogram.RESCALE(1.0/float(other)))
            return self
        else:
            return NotImplemented, 'Histograms can only be divided with other '+\
              ' histograms or scalars.'

    __truediv__ = __div__

class HwU(Histogram):
    """A concrete implementation of an histogram plots using the HwU format for
    reading/writing histogram content."""
    
    allowed_dimensions         = [2]
    allowed_types              = []   

    # For now only HwU output format is implemented.
    output_formats_implemented = ['HwU','gnuplot'] 
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
                       '&\s*(?P<wgt_name>(\S|(\s(?!\s*(&|$))))+)(\s(?!(&|$)))*')
    
    # ================================
    #  Histo weight specification RE's
    # ================================
    # The start of a plot
    histo_start_re = re.compile('^\s*<histogram>\s*(?P<n_bins>\d+)\s*"\s*'+
                                   '(?P<histo_name>(\S|(\s(?!\s*")))+)\s*"\s*$')
    # A given weight specifier
    a_float_re = '[\+|-]?\d+(\.\d*)?([EeDd][\+|-]?\d+)?'
    histo_bin_weight_re = re.compile('(?P<weight>%s)'%a_float_re)
    # The end of a plot
    histo_end_re = re.compile(r'^\s*<\\histogram>\s*$')
    # A scale type of weight
    weight_label_scale = re.compile('^\s*mur\s*=\s*(?P<mur_fact>%s)'%a_float_re+\
                   '\s*muf\s*=\s*(?P<muf_fact>%s)\s*$'%a_float_re,re.IGNORECASE)
    weight_label_PDF = re.compile('^\s*PDF\s*=\s*(?P<PDF_set>\d+)\s*$')
    
    class ParseError(MadGraph5Error):
        """a class for histogram data parsing errors"""
    
    def __init__(self, file_path=None, weight_header=None):
        """ Read one plot from a file_path or a stream. Notice that this
        constructor only reads one, and the first one, of the plots specified.
        If file_path was a path in argument, it would then close the opened stream.
        If file_path was a stream in argument, it would leave it open.
        The option weight_header specifies an ordered list of weight names 
        to appear in the file specified."""
        
        super(HwU, self).__init__()

        self.dimension = 2
        
        if file_path is None:
            return
        elif isinstance(file_path, str):
            stream = open(file_path,'r')
        elif isinstance(file_path, file):
            stream = file_path
        else:
            raise MadGraph5Error, "Argument file_path '%s' for HwU init"\
            %str(file_path)+"ialization must be either a file path or a stream."

        # Attempt to find the weight headers if not specified        
        if not weight_header:
            weight_header = HwU.parse_weight_header(stream)

        if not self.parse_one_histo_from_stream(stream, weight_header):
            # Indicate that the initialization of the histogram was unsuccessful
            # by setting the BinList property to None.
            super(Histogram,self).__setattr__('bins',None)
        
        # Explicitly close the opened stream for clarity.
        if isinstance(file_path, str):
            stream.close()
    
    def get_formatted_header(self):
        """ Return a HwU formatted header for the weight label definition."""

        res = '##& xmin & xmax & central value & dy & '
        
        others = []
        for label in self.bins.weight_labels:
            if label in ['central', 'stat_error']:
                continue
            if isinstance(label, str):
                others.append(label)
            elif isinstance(label, tuple):
                others.append('muR=%4.2f muF=%4.2f'%(label[0],label[1]))
            elif isinstance(label, int):
                others.append('PDF= %d'%label)
        
        return res+' & '.join(others)

    def get_HwU_source(self, print_header=True):
        """ Returns the string representation of this histogram using the
        HwU standard."""

        res = []
        if print_header:
            res.append(self.get_formatted_header())
            res.extend([''])
        res.append('<histogram> %s "%s"'%(len(self.bins),
                                     self.get_HwU_histogram_name(format='HwU')))
        for bin in self.bins:
            res.append(' '.join('%+16.7e'%wgt for wgt in list(bin.boundaries)+
                                  [bin.wgts['central'],bin.wgts['stat_error']]))
            res[-1] += ' '.join('%+16.7e'%bin.wgts[key] for key in 
                self.bins.weight_labels if key not in ['central','stat_error'])
        res.append('<\histogram>')
        return res
    
    def output(self, path=None, format='HwU', print_header=True):
        """ Ouput this histogram to a file, stream or string if path is kept to
        None. The supported format are for now. Chose whether to print the header
        or not."""
        
        if not format in HwU.output_formats_implemented:
            raise MadGraph5Error, "The specified output format '%s'"%format+\
                             " is not yet supported. Supported formats are %s."\
                                                 %HwU.output_formats_implemented

        if format == 'HwU':
            str_output_list = self.get_HwU_source(print_header=print_header)

        if path is None:
            return '\n'.join(str_output_list)
        elif isinstance(path, str):
            stream = open(path,'w')
            stream.write('\n'.join(str_output_list))
            stream.close()
        elif isinstance(path, file):
            path.write('\n'.join(str_output_list))
        
        # Successful writeout
        return True

    def test_plot_compability(self, other, consider_type=True):
        """ Test whether the defining attributes of self are identical to histo,
        typically to make sure that they are the same plots but from different
        runs, and they can be summed safely. We however don't want to 
        overload the __eq__ because it is still a more superficial check."""
        
        if self.title != other.title or \
           self.bins.weight_labels != other.bins.weight_labels or \
           (self.type != other.type and consider_type) or \
           self.x_axis_mode != self.x_axis_mode or \
           self.y_axis_mode != self.y_axis_mode or \
           any(b1.boundaries!=b2.boundaries for (b1,b2) in \
                                                     zip(self.bins,other.bins)):
            return False
        
        return True
           
            
    
    @classmethod
    def parse_weight_header(cls, stream):
        """ Read a given stream until it finds a header specifying the weights
        and then returns them."""
        
        for line in stream:
            if cls.weight_header_start_re.match(line):
                header = [h.group('wgt_name') for h in 
                                            cls.weight_header_re.finditer(line)]
                if any((name not in header) for name in cls.mandatory_weights):
                    raise HwU.ParseError, "The mandatory weight names %s were"\
                     %str(cls.mandatory_weights.keys())+" are not all present"+\
                     " in the following HwU header definition:\n   %s"%line
                
                # Apply replacement rules specified in mandatory_weights
                header = [ (h if h not in cls.mandatory_weights else 
                                     cls.mandatory_weights[h]) for h in header ]
                
                # We use a special rule for the weight labeled as a 
                # muR=2.0 muF=1.0 scale specification, in which case we store
                # it as a tuple
                for i, h in enumerate(header):
                    scale_wgt = HwU.weight_label_scale.match(h)
                    PDF_wgt   = HwU.weight_label_PDF.match(h)
                    if scale_wgt:
                        header[i] = (float(scale_wgt.group('mur_fact')),
                                     float(scale_wgt.group('muf_fact')))
                    elif PDF_wgt:
                        header[i] = int(PDF_wgt.group('PDF_set'))

                return header
            
        raise HwU.ParseError, "The weight headers could not be found."
    
    
    def process_histogram_name(self, histogram_name):
        """ Parse the histogram name for tags which would set its various
        attributes."""
        
        for i, tag in enumerate(histogram_name.split('|')):
            if i==0:
                self.title = tag.strip()
            else:
                stag = tag.split('@')
                if len(stag)!=2:
                    raise MadGraph5Error, 'Specifier in title must have the'+\
            " syntax @<attribute_name>:<attribute_value>, not '%s'."%tag.strip()
                # Now list all supported modifiers here
                stag = [t.strip().upper() for t in stag]
                if stag[0] in ['T','TYPE']:
                    self.type = stag[1]
                elif stag[0] in ['X_AXIS', 'X']:
                    self.x_axis_mode = stag[1]                    
                elif stag[0] in ['Y_AXIS', 'Y']:
                    self.y_axis_mode = stag[1] 
                else:
                    raise MadGraph5Error, "Specifier '%s' not recognized."%stag[0]                    
        
    def get_HwU_histogram_name(self, format='human'):
        """ Returns the histogram name in the HwU syntax or human readable."""
        
        type_map = {'NLO':'NLO', 'LO':'LO', 'AUX':'auxiliary histogram', None:''}
        
        if format=='human':
            res = self.title
            try:
                res += ', %s'%type_map[self.type]
            except KeyError:
                res += ', %s'%str(self.type)                
            return res

        elif format=='HwU':
            res = [self.title]
            res.append('|X_AXIS@%s'%self.x_axis_mode)
            res.append('|Y_AXIS@%s'%self.y_axis_mode)
            if self.type:
                res.append('|TYPE@%s'%self.type)
            return ' '.join(res)
        
    def parse_one_histo_from_stream(self, stream, weight_header):
        """ Reads *one* histogram from a stream, with the mandatory specification
        of the ordered list of weight names. Return True or False depending
        on whether the starting definition of a new plot could be found in this
        stream."""
        n_bins = 0
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
                if j == len(weight_header):
                    raise ParseError, " There is more bin weights"+\
                      " specified than expected (%i)"%len(weight_header)
                if weight_header[j] == 'boundary_xmin':
                    boundaries[0] = float(weight.group('weight'))
                elif weight_header[j] == 'boundary_xmax':
                    boundaries[1] = float(weight.group('weight'))                            
                else:
                    bin_weights[weight_header[j]] = \
                                           float(weight.group('weight'))

            # For the HwU format, we know that exactly two 'weights'
            # specified in the weight_header are in fact the boundary 
            # coordinate, so we must subtract two.    
            if len(bin_weights)<(len(weight_header)-2):
                raise HwU.ParseError, " There are only %i weights"\
                    %len(bin_weights)+" specified and %i were expected."%\
                                                      (len(weight_header)-2)
            self.bins.append(Bin(tuple(boundaries), bin_weights))
            if len(self.bins)==n_bins:
                break

        if len(self.bins)!=n_bins:
            raise HwU.ParseError, "%i bin specification "%len(self.bins)+\
                               "were found and %i were expected."%n_bins

        # Now jump to the next <\histo> tag.
        for line_end in stream:
            if HwU.histo_end_re.match(line_end):
                # Finally, remove all the auxiliary weights
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

    def set_uncertainty(self, type='all_scale'):
        """ Adds a weight to the bins which is the envelope of the scale
        uncertainty, for the scale specified which can be either 'mur', 'muf',
        'all_scale' or 'PDF'."""

        if type.upper()=='MUR':
            new_wgt_label  = 'delta_mur'
            scale_position = 0
        elif type.upper()=='MUF':
            new_wgt_label = 'delta_muf'
            scale_position = 1
        elif type.upper()=='ALL_SCALE':
            new_wgt_label = 'delta_mu'
            scale_position = -1
        elif type.upper()=='PDF':
            new_wgt_label = 'delta_pdf'
            scale_position = -2
        else:
            raise MadGraph5Error, ' The function set_uncertainty can'+\
              " only handle the scales 'mur', 'muf', 'all_scale' or 'pdf'."       
        
        if scale_position > -2:
            wgts_to_consider = [ label for label in self.bins.weight_labels if \
                                                      isinstance(label, tuple) ]
            if scale_position > -1:
                wgts_to_consider = [ lab for lab in wgts_to_consider if \
                    (lab[scale_position]!=1.0 and all(k==1.0 for k in \
                                lab[:scale_position]+lab[scale_position+1:]) ) ]
        elif scale_position == -2:
            wgts_to_consider = [ label for label in self.bins.weight_labels if \
                                                      isinstance(label, int) ]            
        
        if len(wgts_to_consider)==0:
            # No envelope can be constructed, it is not worth adding the weights
            return None
        else:
            # Place the new weight label last before the first tuple
            new_wgt_labels  = ['%s_min @aux'%new_wgt_label,
                                                    '%s_max @aux'%new_wgt_label]
            try:
                position = [(not isinstance(lab, str)) for lab in \
                                            self.bins.weight_labels].index(True)
                self.bins.weight_labels = self.bins.weight_labels[:position]+\
                  new_wgt_labels + self.bins.weight_labels[position:]
            except ValueError:
                position = len(self.bins.weight_labels)
                self.bins.weight_labels.extend(new_wgt_labels)

            # Now add the corresponding weight to all Bins
            for bin in self.bins:
                if type!='PDF':
                    bin.wgts[new_wgt_labels[0]] = min(bin.wgts[label] \
                                                  for label in wgts_to_consider)
                    bin.wgts[new_wgt_labels[1]] = max(bin.wgts[label] \
                                                  for label in wgts_to_consider)
                else:
                    pdfs   = [bin.wgts[pdf] for pdf in sorted(wgts_to_consider)]
                    pdf_up     = 0.0
                    pdf_down   = 0.0
                    cntrl_val  = bin.wgts['central']
                    if pdfs[-1] <= 90000:
                        # use Hessian method (CTEQ & MSTW)
                        if len(pdfs)>2:
                            for i in range(int((len(pdfs)-1)/2)):
                                pdf_up   += max(0.0,pdfs[2*i+1]-cntrl_val,
                                                      pdfs[2*i+2]-cntrl_val)**2
                                pdf_down += max(0.0,cntrl_val-pdfs[2*i+1],
                                                       cntrl_val-pdfs[2*i+2])**2
                            pdf_up   = cntrl_val + math.sqrt(pdf_up)
                            pdf_down = cntrl_val - math.sqrt(pdf_down)
                        else:
                            pdf_up   = bin.wgts[pdfs[0]]
                            pdf_down = bin.wgts[pdfs[0]]
                    else:
                        # use Gaussian method (NNPDF)
                        pdf_stdev = 0.0
                        for pdf in pdfs[1:]:
                            pdf_stdev += (pdf - cntrl_val)**2
                        pdf_stdev = math.sqrt(pdf_stdev/float(len(pdfs)-2))
                        pdf_up   = cntrl_val+pdf_stdev
                        pdf_down = cntrl_val-pdf_stdev
                    # Finally add them to the corresponding new weight
                    bin.wgts[new_wgt_labels[0]] = pdf_down
                    bin.wgts[new_wgt_labels[1]] = pdf_up
            
            # And return the position in self.bins.weight_labels of the first
            # of the two new weight label added.
            return position                 
    
    @classmethod
    def get_x_optimal_range(cls, histo_list, weight_labels=None):
        """ Function to determine the optimal x-axis range when plotting 
        together the histos in histo_list and considering the weights 
        weight_labels"""

        # If no list of weight labels to consider is given, use them all.
        if weight_labels is None:
            weight_labels = histo_list[0].bins.weight_labels

        all_boundaries = sum([ list(bin.boundaries) for histo in histo_list \
                                                    for bin in histo.bins if \
             (sum(abs(bin.wgts[label]) for label in weight_labels) > 0.0)]  ,[])

        x_min = min(all_boundaries)
        x_max = max(all_boundaries)
        
        return (x_min, x_max)
    
    @classmethod
    def get_y_optimal_range(cls,histo_list, weight_labels=None, scale='LOG',
                                                                 Kratio = False):
        """ Function to determine the optimal y-axis range when plotting 
        together the histos in histo_list and considering the weights 
        weight_labels. The option Kratio is present to allow for the couple of
        tweaks necessary for the the K-factor ratio histogram y-range."""

        # If no list of weight labels to consider is given, use them all.
        if weight_labels is None:
            weight_labels = histo_list[0].bins.weight_labels
        
        all_weights = []
        for histo in histo_list:
            for bin in histo.bins:
                for label in weight_labels:
                    # Filter out bin weights at *exactly* because they often
                    # come from pathological division by zero for empty bins.
                    if Kratio and bin.wgts[label]==0.0:
                        continue
                    if scale!='LOG':
                        all_weights.append(bin.wgts[label])
                        if label == 'stat_error':
                            all_weights.append(-bin.wgts[label])         
                    elif bin.wgts[label]>0.0:
                        all_weights.append(bin.wgts[label])
        
        sum([ [bin.wgts[label] for label in weight_labels if \
                             (scale!='LOG' or bin.wgts[label]!=0.0)] \
                           for histo in histo_list for bin in histo.bins],  [])

        all_weights.sort()
        partial_max = all_weights[int(len(all_weights)*0.95)]
        partial_min = all_weights[int(len(all_weights)*0.05)]
        max         = all_weights[-1]
        min         = all_weights[0]
        
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

        # Make sure the range has finite length
        if y_min == y_max:
            if max == min:
                y_min -= 1.0
                y_max += 1.0
            else:
                y_min = min
                y_max = max

        # Finally enforce the maximum if there is 5 bins or less
        if len(histo_list[0].bins) <= 5:
            y_min = min
            y_max = max            
        
        return ( y_min , y_max )
    
class HwUList(histograms_PhysicsObjectList):
    """ A class implementing features related to a list of Hwu Histograms. """
    
    # Define here the number of line color schemes defined. If you need more,
    # simply define them in the gnuplot header and increase the number below.
    # It must be <= 9.
    number_line_colors_defined = 4 
    
    def is_valid_element(self, obj):
        """Test wether specified object is of the right type for this list."""

        return isinstance(obj, HwU) or isinstance(obj, HwUList)
    
    def __init__(self, file_path, weight_header=None, **opts):
        """ Read one plot from a file_path or a stream. 
        This constructor reads all plots specified in target file.
        File_path can be a path or a stream in the argument.
        The option weight_header specifies an ordered list of weight names 
        to appear in the file or stream specified."""
        
        if isinstance(file_path, str):
            stream = open(file_path,'r')
        elif isinstance(file_path, file):
            stream = file_path
        else:
            return super(HwUList,self).__init__(file_path, **opts)

        # Attempt to find the weight headers if not specified        
        if not weight_header:
            weight_header = HwU.parse_weight_header(stream)

        new_histo = HwU(stream, weight_header)
        while not new_histo.bins is None:
            self.append(new_histo)
            new_histo = HwU(stream, weight_header)            
            
        # Explicitly close the opened stream for clarity.
        if isinstance(file_path, str):
            stream.close()

    def output(self, path, format='gnuplot'):
        """ Ouput this histogram to a file, stream or string if path is kept to
        None. The supported format are for now. Chose whether to print the header
        or not."""
        
        if len(self)==0:
            return MadGraph5Error, 'No histograms stored in the list yet.'
        
        if not format in HwU.output_formats_implemented:
            raise MadGraph5Error, "The specified output format '%s'"%format+\
                             " is not yet supported. Supported formats are %s."\
                                                 %HwU.output_formats_implemented

        if isinstance(path, str) and '.' not in os.path.basename(path):
            output_base_name = os.path.basename(path)
            HwU_stream       = open(path+'.HwU','w')
        else:
            raise MadGraph5Error, "The path argument of the output function of"+\
              " the HwUList instance must be file path without its extension."

        HwU_output_list = []
        # If the format is just the raw HwU source, then simply write them
        # out all in sequence.
        if format == 'HwU':
            HwU_output_list.extend(self[0].get_HwU_source(print_header=True))
            for histo in self[1:]:
                HwU_output_list.extend(histo.get_HwU_source())
                HwU_output_list.extend(['',''])
            stream.write('\n'.join(HwU_output_list))
            stream.close()
            return
        
        # Now we consider that we are attempting a gnuplot output.
        if format == 'gnuplot':
            gnuplot_stream = open(path+'.gnuplot','w')

        # First order the histograms according to their type.
        titles_order = [h.title for h in self]
        def ordering_function(histo):
            title_position = titles_order.index(histo.title)
            if histo.type is None:
                return (title_position,3)
            if histo.type.startswith('NLO'):
                return (title_position,1)
            elif histo.type.startswith('LO'):
                return (title_position,2)
            elif histo.type.startswith('AUX'):
                return (title_position,5)
            else:
                return (title_position,4)

        # The command below is to first order them in alphabetical order, but it
        # is often better to keep the order of the original HwU source.
#        self.sort(key=lambda histo: '%s_%d'%(histo.title,
#                                                  type_order.index(histo.type)))
        self.sort(key=ordering_function)

        # Now group all the identified matching histograms in a list
        matching_histo_lists = HwUList([HwUList([self[0]])])
        for histo in self[1:]:
            matched = False
            for histo_list in matching_histo_lists:
                if histo.test_plot_compability(histo_list[0],consider_type=False):
                    histo_list.append(histo)
                    matched = True
                    break
            if not matched:
                matching_histo_lists.append(HwUList([histo]))

        self[:] = matching_histo_lists

        # Write the gnuplot header
        gnuplot_output_list = [
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
reset

set lmargin 10
set rmargin 0
set terminal postscript portrait enhanced mono dashed lw 1.0 "Helvetica" 9 
set key font ",9"
set key samplen "2"
set output "%s.ps"

set style line 1 lt 1 lc rgb "#006D4F" lw 1.8
set style line 11 lt 2 lc rgb "#006D4F" lw 1.8
set style line 21 lt 4 lc rgb "#006D4F" lw 1.8

set style line 2 lt 1 lc rgb "#B90091" lw 1.8
set style line 12 lt 2 lc rgb "#B90091" lw 1.8
set style line 22 lt 4 lc rgb "#B90091" lw 1.8

set style line 3 lt 1 lc rgb "#2818B1" lw 1.8
set style line 13 lt 2 lc rgb "#2818B1" lw 1.8
set style line 23 lt 4 lc rgb "#2818B1" lw 1.8

set style line 4 lt 1 lc rgb "#A67B00" lw 1.8
set style line 14 lt 2 lc rgb "#A67B00" lw 1.8
set style line 24 lt 4 lc rgb "#A67B00" lw 1.8

set style line 999 lt 1 lc rgb "gray" lw 1.8

safe(x,y,a) = (y == 0.0 ? a : x/y)

set style data histeps

"""%output_base_name
]
        
        # Now output each group one by one
        # Block position keeps track of the gnuplot data_block index considered
        block_position = 0
        for histo_group in self:
            # Output this group
            block_position = histo_group.output_group(HwU_output_list, 
                    gnuplot_output_list, block_position,output_base_name+'.HwU')

        # Now write the tail of the gnuplot command file
        gnuplot_output_list.extend([
          "unset multiplot",
          '!ps2pdf "%s.ps"'%output_base_name,
          '!open "%s.pdf"'%output_base_name])
        
        # Now write result to stream and close it
        gnuplot_stream.write('\n'.join(gnuplot_output_list))
        HwU_stream.write('\n'.join(HwU_output_list))
        gnuplot_stream.close()
        HwU_stream.close()

        logger.debug("Histograms have been written out at "+\
                                 "%s.[HwU|gnuplot]' and can "%output_base_name+\
                                         "now be rendered by invoking gnuplot.")

    def output_group(self, HwU_out, gnuplot_out, block_position, HwU_name):
        """ This functions output a single group of histograms with either one
        histograms untyped (i.e. type=None) or two of type 'NLO' and 'LO' 
        respectively."""
        
        # A sanity check to make sure that there is either one histogram
        # of type 'None' or two of type 'NLO' and 'LO' respectively, since
        # these are the only combinations supported sofar
#        if not (len(self)==1 or \
#                       (len(self)==2 and [h.type for h in self]==['NLO','LO'])):
#            raise MadGraph5Error, "The set of histogram types that can be"+\
#              "output can only contain a single histogram or exactly two with"+\
#              " types ['NLO','LO'], in this order, not %s"\
#                                             %(str([str(h.type) for h in self]))

        # First compute the ratio of NLO and LO histograms if present
        n_histograms = len(self)
        if len(self)>=2:
            self.append(self[0]/self[1])
            if self[0].type=='NLO' and self[1].type=='LO':
                self[-1].title += ' K-factor'
            else:
                self[-1].title += ' %s/%s'%(
                                        self[0].type if self[0].type else '#1',
                                         self[1].type if self[1].type else '#2')
            # By setting its type to aux, we make sure this histogram will be
            # filtered out if the .HwU file output here would be re-loaded later.
            self[-1].type       = 'AUX'

        # Compute scale variation envelope for all diagrams
        mu_var_pos  = self[0].set_uncertainty(type='all_scale')
        PDF_var_pos = self[0].set_uncertainty(type='PDF')
        for histo in self[1:]:
            if mu_var_pos != histo.set_uncertainty(type='all_scale'):
               raise MadGraph5Error, 'Not all histograms in this group specify'+\
                 ' scale dependencies. It is required to be able to output them'+\
                 ' together.'
            if PDF_var_pos != histo.set_uncertainty(type='PDF'):
               raise MadGraph5Error, 'Not all histograms in this group specify'+\
                 ' PDF dependencies. It is required to be able to output them'+\
                 ' together.'
        
        # Now output the corresponding HwU histogram data
        for i, histo in enumerate(self):
            # Print the header the first time only
            HwU_out.extend(histo.get_HwU_source(\
                                     print_header=(block_position==0 and i==0)))
            HwU_out.extend(['',''])

        # First the global gnuplot header for this histogram group
        global_header =\
"""
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
set key horizontal noreverse maxcols 1 width -4 at graph 0.92, graph 0.9
set label front 'MadGraph5\_aMC\@NLO' font "Courier,11" rotate by 90 at graph 1.02, graph 0.04
"""
        
        # Now the header for each subhistogram
        subhistogram_header = \
"""#-- rendering subhistogram '%(subhistogram_type)s'
%(unset label)s
%(set_format_y)s
set yrange [%(ymin).4e:%(ymax).4e]
set origin %(origin_x).4e, %(origin_y).4e
set size %(size_x).4e, %(size_y).4e
set mytics %(mytics)d
%(set_ytics)s
%(set_format_x)s
%(set_yscale)s
%(set_ylabel)s
%(set_histo_label)s
plot \\"""
        replacement_dic = {}

        replacement_dic['title'] = self[0].get_HwU_histogram_name(format='human')
        # Determine what weight to consider when computing the optimal 
        # range for the y-axis.
        wgts_to_consider = ['central']
        if not mu_var_pos is None:
            wgts_to_consider.append(self[0].bins.weight_labels[mu_var_pos])
            wgts_to_consider.append(self[0].bins.weight_labels[mu_var_pos+1])
        if not PDF_var_pos is None:
            wgts_to_consider.append(self[0].bins.weight_labels[PDF_var_pos])
            wgts_to_consider.append(self[0].bins.weight_labels[PDF_var_pos+1])
        (xmin, xmax) = HwU.get_x_optimal_range(self[:2],\
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
        (ymin, ymax) = HwU.get_y_optimal_range(self[:2],
                   weight_labels = wgts_to_consider, scale=self[0].y_axis_mode)

        # Force a linear scale if the detected range is negative
        if ymin< 0.0:
            self[0].y_axis_mode = 'LIN'
            
        # Already add a margin on upper bound.
        if self[0].y_axis_mode=='LOG':
            ymax += 10.0 * (ymax - ymin)
            ymin -= 0.9 * ymin
        else:
            ymax += 0.2 * (ymax - ymin)
            ymin -= 0.2 * (ymax - ymin)

        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        replacement_dic['unset label'] = ''
        replacement_dic['origin_x'] = 0.0
        replacement_dic['origin_y'] = 0.5        
        replacement_dic['size_x'] = 1.0
        replacement_dic['size_y'] = 0.4
        replacement_dic['mytics'] = 10
        # Use default choise for the main histogram
        replacement_dic['set_ytics'] = 'set ytics auto'
        replacement_dic['set_format_x'] = "set format x ''"
        replacement_dic['set_ylabel'] = 'set ylabel "{/Symbol s} per bin [pb]"' 
        replacement_dic['set_yscale'] = "set logscale y" if \
                              self[0].y_axis_mode=='LOG' else 'unset logscale y'
        replacement_dic['set_format_y'] = "set format y '10^{%T}'" if \
                                self[0].y_axis_mode=='LOG' else 'unset format'
                                
        replacement_dic['set_histo_label'] = ""
        gnuplot_out.append(subhistogram_header%replacement_dic)
        
        # Now add the main subhistogram
        plot_lines = []
        for i, histo in enumerate(self[:n_histograms]):
            color_index = i%self.number_line_colors_defined+1
            plot_lines.extend([
"'%s' index %d using (($1+$2)/2):3 ls %d title '%s'"\
%(HwU_name,block_position+i,i+1,histo.get_HwU_histogram_name(format='human')
if i==0 else (histo.type if histo.type else 'central value for plot #%d'%(i+1))),
"'%s' index %d using (($1+$2)/2):3:4 w yerrorbar ls %d title ''"\
%(HwU_name,block_position+i,color_index)])
            # And show scale variation if available
            if not mu_var_pos is None:
                plot_lines.extend([
"'%s' index %d using (($1+$2)/2):%d ls %d title '%s'"\
%(HwU_name,block_position+i,mu_var_pos+3,color_index+10,'%s scale variation'\
%('%d'%(i+1) if histo.type is None else histo.type)),
"'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
%(HwU_name,block_position+i,mu_var_pos+4,color_index+10),
                ])
            # And now PDF_variation if available
            if not PDF_var_pos is None:
                plot_lines.extend([
"'%s' index %d using (($1+$2)/2):%d ls %d title '%s'"\
%(HwU_name,block_position+i,PDF_var_pos+3,color_index+20,'%s PDF variation'\
%('%d'%(i+1) if histo.type is None else histo.type)),
"'%s' index %d using (($1+$2)/2):%d ls %d title ''"\
%(HwU_name,block_position+i,PDF_var_pos+4,color_index+20),
                ])

        # Add the plot lines
        gnuplot_out.append(',\\\n'.join(plot_lines))

        # Now we can add the scale variation ratio
        replacement_dic['subhistogram_type'] = 'Relative scale and PDF uncertainty'

        # This function is just to temporarily create the ratio histogram with 
        # the hwu.combine function. Notice hoewer that the histogram hence created
        # is not used to actuall plot the quantity in gnuplot   
        def rel_scale(wgtsA, wgtsB):
            new_wgts = {}
            for label, wgt in wgtsA.items():
                if label in wgts_to_consider+['stat_error']:
                    if wgtsB['central']==0.0 and wgt==0.0:
                        new_wgts[label] = 0.0
                        continue
                    elif wgtsB[label]==0.0:
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

        (ymin, ymax) = HwU.get_y_optimal_range(
          [self[0].__class__.combine(self[0],self[0],rel_scale),], 
          weight_labels = wgts_to_consider+['stat_error'],  scale='LIN')

        # Add a margin on upper and lower bound.
        ymax = min(ymax + 0.2 * (ymax - ymin), 10.0)
        ymin = max(ymin - 0.2 * (ymax - ymin), -10.0)
        replacement_dic['unset label'] = 'unset label'
        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        replacement_dic['origin_x'] = 0.0
        replacement_dic['origin_y'] = 0.4        
        replacement_dic['size_x'] = 1.0
        replacement_dic['size_y'] = 0.1
        replacement_dic['mytics'] = 2
#        replacement_dic['set_ytics'] = 'set ytics %f'%((int(10*(ymax-ymin))/10)/3.0)
        replacement_dic['set_ytics'] = 'set ytics auto'
        replacement_dic['set_format_x'] = "set format x ''" if len(self)>1 else \
                                                                  "set format x"
        replacement_dic['set_ylabel'] = 'set ylabel "%s rel.unc."'\
                                       %(self[0].type if self[0].type else '#1')
        replacement_dic['set_yscale'] = "unset logscale y"
        replacement_dic['set_format_y'] = 'unset format'
                                
        replacement_dic['set_histo_label'] = \
         'set label "Relative uncertainties" font ",9" at graph 0.03, graph 0.13'
#        'set label "Relative uncertainties" font ",9" at graph 0.79, graph 0.13'
        gnuplot_out.append(subhistogram_header%replacement_dic)
        
        # Now add the first subhistogram
        plot_lines = ["0.0 ls 999 title ''",
"'%s' index %d using (($1+$2)/2):(0.0):(safe($4,$3,0.0)) w yerrorbar ls 1 title ''"%\
(HwU_name,block_position)]
        # Then the scale variations
        if not mu_var_pos is None:
            plot_lines.extend([
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls 11 title ''"\
%(HwU_name,block_position,mu_var_pos+3),
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls 11 title ''"\
%(HwU_name,block_position,mu_var_pos+4)
            ])
        if not PDF_var_pos is None:
            plot_lines.extend([
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls 21 title ''"\
%(HwU_name,block_position,PDF_var_pos+3),
"'%s' index %d using (($1+$2)/2):(safe($%d,$3,1.0)-1.0) ls 21 title ''"\
%(HwU_name,block_position,PDF_var_pos+4)
            ])
        
        # Add the plot lines
        gnuplot_out.append(',\\\n'.join(plot_lines))

        # We finish here for a single diagram output
        if len(self)<=1:
            # Now add the tail for this group
            gnuplot_out.extend(['','unset label','',
'################################################################################'])
            # Return the starting data_block position for the next histogram group
            return block_position+len(self)

        # We can finally add the last subhistogram for the K-factor
        # Now we can add the scale variation ratio
        if self[0].type=='NLO' and self[0].type=='LO':
            ratio_name_long = 'NLO/LO (K-factor)'
            ratio_name_short = 'NLO/LO'
        else:
            ratio_name_long = '%s/%s'%(
                '#1' if self[0].type==None else '%s'%self[0].type,
                '#2' if self[1].type==None else '%s'%self[1].type)
            ratio_name_short = ratio_name_long
            
        ratio_name = '%s/%s'
        replacement_dic['subhistogram_type'] = '%s ratio'%ratio_name_long
        replacement_dic['set_ylabel'] = 'set ylabel "%s"'%ratio_name_short

        (ymin, ymax) = HwU.get_y_optimal_range([self[-1],], 
                   weight_labels = wgts_to_consider+['stat_error'], scale='LIN',
                   Kratio = True)        
        
        # Add a margin on upper and lower bound.
        ymax = min(ymax + 0.15 * (ymax - ymin), 10.0)
        ymin = max(ymin - 0.15 * (ymax - ymin), -10.0)
        replacement_dic['unset label'] = 'unset label'
        replacement_dic['ymin'] = ymin
        replacement_dic['ymax'] = ymax
        replacement_dic['origin_x'] = 0.0
        replacement_dic['origin_y'] = 0.3      
        replacement_dic['size_x'] = 1.0
        replacement_dic['size_y'] = 0.1
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
        
        # Now add the first subhistogram
        plot_lines = ["1.0 ls 999 title ''",
"'%s' index %d using (($1+$2)/2):3 ls 1 title ''"%\
(HwU_name,block_position+n_histograms),
"'%s' index %d using (($1+$2)/2):3:4 w yerrorbar ls 1 title ''"%\
(HwU_name,block_position+n_histograms)]
        # Then the scale variations
        if not mu_var_pos is None:
            plot_lines.extend([
"'%s' index %d using (($1+$2)/2):%d ls 11 title ''"\
%(HwU_name,block_position+n_histograms,mu_var_pos+3),
"'%s' index %d using (($1+$2)/2):%d ls 11 title ''"\
%(HwU_name,block_position+n_histograms,mu_var_pos+4)
            ])
        if not PDF_var_pos is None:
            plot_lines.extend([
"'%s' index %d using (($1+$2)/2):%d ls 21 title ''"\
%(HwU_name,block_position+n_histograms,PDF_var_pos+3),
"'%s' index %d using (($1+$2)/2):%d ls 21 title ''"\
%(HwU_name,block_position+n_histograms,PDF_var_pos+4)
            ])
        
        # Add the plot lines
        gnuplot_out.append(',\\\n'.join(plot_lines))
        
        # Now add the tail for this group
        gnuplot_out.extend(['','unset label','',
'################################################################################'])

        # Return the starting data_block position for the next histogram group
        return block_position+len(self)

if __name__ == "__main__":
    main_doc = \
    """ For testing and standalone use. Usage:
        python histograms.py <.HwU input_file_path_1> <.HwU input_file_path_2> ... --out=<output_file_path.format> <option>
        Where <option> can be *one* of the following: 
           '--gnuplot' or '' output the histogram read to gnuplot
           '--HwU'           to output the histogram read to the raw HwU source.
           '--types=<type1>,<type2>,...' to keep only the type<i> when importing histograms.
           '--help'          See this message.
           '--show_full'     to show the complete output of what was read.
           '--show_short'    to show a summary of what was read.
    """
    
    def log(msg):
        print "histograms.py :: %s"%str(msg)
    
    if '--help' in sys.argv:
        log('\n\n%s'%main_doc)
        sys.exit(0)

    if not any(arg.startswith('--out=') for arg in sys.argv):
        log('Please specify an output name with --out=<OutputName>.')
        sys.exit(0)
    else:
        for arg in sys.argv[1:]:
            if arg.startswith('--out='):
                OutName = arg[6:]

    accepted_types = []
    for arg in sys.argv[1:]:
        if arg.startswith('--types='):
            accepted_types = [(type if type!='None' else None) for type in \
                                                             arg[8:].split(',')]

    log("=======")
    histo_list = HwUList([])
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            break
        log("Loading histograms from '%s'."%arg)
        new_histo_list = HwUList(arg)
        if accepted_types!=[]:
            new_histo_list = [h for h in new_histo_list if h.type in accepted_types]
        if i!=0:
            for histo in new_histo_list:
                if not histo.type is None:
                    histo.type += " #%d"%(i+1)
                else:
                    histo.type = "#%d"%(i+1)

        histo_list.extend(new_histo_list)
    log("A total of %i histograms were found."%len(histo_list))
    log("=======")

    if '--gnuplot' in sys.argv or all(arg not in ['--HwU'] for arg in sys.argv):
        histo_list.output(OutName, format='gnuplot')
        log("%d histograms have been output in " % len(histo_list)+\
                "the gnuplot format at '%s.[HwU|gnuplot]'." % OutName)
        sys.exit(0)

    if '--HwU' in sys.argv:
        log("Histograms data has been output in the HwU format at "+\
                                              "'%s.HwU'."%OutName)
        histo_list.output(OutName, format='HwU')
        sys.exit(0)
    
    if '--show_short' in sys.argv or '--show_full' in sys.argv:
        for i, histo in enumerate(histo_list):
            if i!=0:
                log('-------')
            log(histo.nice_string(short=(not '--show_full' in sys.argv)))
    log("=======")
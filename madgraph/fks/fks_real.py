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

"""Definitions of the objects needed for the implementation of MadFKS"""

import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.fks.fks_common as fks_common
import copy
import logging
import fractions

logger = logging.getLogger('madgraph.fks_real')
#from math import min

class FKSMultiProcessFromReals(diagram_generation.MultiProcess): #test written
    """a multi process class that contains informations on the born processes 
    and the reals"""
    
    def default_setup(self):
        """Default values for all properties"""
        super(FKSMultiProcessFromReals, self).default_setup()

        self['real_processes'] = FKSProcessFromRealsList()
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcessFromReals, self).get_sorted_keys()
        keys += ['real_processes']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'real_processes':
            if not isinstance(value, FKSProcessFromRealsList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for real_processes " % str(value)
                                                                
        return super(FKSMultiProcessFromReals,self).filter(name, value)
    
    def __init__(self, *arguments):
        """initialize the original multiprocess, then generates the amps for the 
        borns, then geneare the born processes and the reals"""
                
        super(FKSMultiProcessFromReals, self).__init__(*arguments)
        
        amps = self.get('amplitudes')
        born_amplist = []
        born_amp_id_list = []
        for amp in amps:
            real_proc = FKSProcessFromReals(amp)
            self['real_processes'].append(real_proc)
            
            
class FKSBornProcess(object):
    """contains informations about a born process
    -- born amplitude
    -- i/j_fks (in the real process leglist)
    -- ijglu -> 0 if ij is not a gluon, ij[number] otherwise
    -- need_color_link -> if color links are needed (ie i_fks is a gluon)
    -- color link list"""
    
    def __init__(self, real_proc, leg_i, leg_j, leg_ij, perturbed_orders = ['QCD']):
        """initialize the born process starting from the real process and the
        combined leg"""

        self.i_fks = leg_i.get('number')
        self.j_fks = leg_j.get('number')
        if leg_ij.get('spin') == 3:
            self.ijglu = leg_ij.get('number')
        else:
            self.ijglu = 0
        
        self.need_color_links = leg_i.get('color') == 8 and leg_i.get('massless')
                
        self.process = copy.copy(real_proc)
#        orders = copy.copy(real_proc.get('orders'))
        
#        for order in perturbed_orders:
#            orders[order] += -1
#        self.process.set('orders', orders)
        
        born_legs = self.reduce_real_leglist(leg_i, leg_j, leg_ij)
        self.process.set('legs', MG.LegList(born_legs))
        self.amplitude = diagram_generation.Amplitude(self.process)
        self.color_links = []
    
    
    def reduce_real_leglist(self, leg_i, leg_j, leg_ij):
        """removes from the leglist of self.process leg_i, leg_j 
        and inserts leg_ij"""
        red_leglist = fks_common.to_fks_legs(
                        copy.deepcopy(self.process.get('legs')),
                                      self.process.get('model'))
#        print " ".join(("%d" % l['id']) for l in red_leglist)
#        print "LEGS ", red_leglist
#        print "I   ", leg_i
#        print "J   ", leg_j
        red_leglist.remove(leg_i)
        red_leglist.remove(leg_j)
        red_leglist.insert(leg_ij.get('number')-1, leg_ij)
        for n, leg in enumerate(red_leglist):
            red_leglist[n].set('number', n+1)
        return red_leglist

    def find_color_links(self): #test written
        """finds all the possible color links between two legs of the born.
        Uses the find_color_links function in fks_common"""
        if self.need_color_links:
            self.color_links = fks_common.find_color_links(\
                              self.process.get('legs'))
        return self.color_links
 
#===============================================================================
# FKS Process
#===============================================================================

class FKSProcessFromRealsList(MG.PhysicsObjectList):
    """class to handle lists of FKSProcesses"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromReals for the list."""
        return isinstance(obj, FKSProcessFromReals)
   
   
class FKSProcessFromReals(object):
    """the class for an FKS process, starting from reals """ 
    
    def __init__(self, start_proc = None): #test written
        """initialization: starts either from an anplitude or a process,
        then init the needed variables:
        --real_proc/real_amp
        --model
        --leglist, nlegs
        --pdg codes, colors ###, ipos, j_from_i to be written in fks.inc
        --borns"""
        
        self.borns = []
        self.leglist = []
        self.pdg_codes = []
        self.colors = []
        self.nlegs = 0
        self.fks_ipos = []
        self.fks_j_from_i = {}
        self.real_proc = None
        self.real_amp = None
        self.model = None
 
        if start_proc:
            if isinstance(start_proc, MG.Process):
                self.real_proc = start_proc 
                self.real_amp = diagram_generation.Amplitude(self.real_proc)
            elif isinstance(start_proc, diagram_generation.Amplitude):
                self.real_amp = start_proc
                self.real_proc = start_proc.get('process') 
            self.model = self.real_proc.get('model')   

            self.leglist = fks_common.to_fks_legs(self.real_proc.get('legs'), self.model)
            self.nlegs = len(self.leglist)
            for leg in self.leglist:
                self.pdg_codes.append(leg['id'])
                self.colors.append(leg['color'])
            
            self.find_borns()
            
    def get_fks_inc_string(self):
        """returns the list of configurations corrresponding to the various 
        underlying borns"""
        string = "integer fks_configs, ipos, jpos \n\
data fks_configs / %(nconfs)d / \n\
data fks_i(%(nconfs)d), fks_j(%(nconfs)d) \n\
INTEGER FKS_IPOS(0:NEXTERNAL) \n\
INTEGER FKS_J_FROM_I(NEXTERNAL, 0:NEXTERNAL) \n\
INTEGER PARTICLE_TYPE(NEXTERNAL), PDG_TYPE(NEXTERNAL) \n" %{'nconfs' : 
                                                        len(self.borns)}
        for n, born in enumerate(self.borns):
            replace= {'conf_num' : n+1, 'i_fks':born.i_fks, 'j_fks':born.j_fks}
            string +=" \n\
c     FKS configuration number  %(conf_num)d \n\
data fks_i(  %(conf_num)d  ) /  %(i_fks)d  / \n\
data fks_j(  %(conf_num)d  ) /  %(j_fks)d  / \n\
      " % replace
        
        ii = sorted(set([b.i_fks for b in self.borns ]))
        ipos_dict = {'n_ipos' : len(ii), 
                     'string_ipos' : ', '.join(["%d" % i for i in ii]) }
        string += "\n\
data (fks_ipos(ipos), ipos = 0, %(n_ipos)d)  / %(n_ipos)d, %(string_ipos)s /\n" \
                % ipos_dict
        for i in ii:
            jj = []
            for b in self.borns:
                if b.i_fks == i:
                    jj.append(b.j_fks)
            j_dict = {'n_j' : len(jj), 
                     'string_j' : ', '.join(["%d" % j for j in jj]) }
            string += "\n\
data (fks_j_from_i(jpos), jpos = 0, %(n_j)d)  / %(n_j)d, %(string_j)s /" \
                % j_dict
        string += "\n\
C\n\
C     Particle type:\n\
C     octet = 8, triplet = 3, singlet = 1\n\
DATA (PARTICLE_TYPE(IPOS), IPOS=1, NEXTERNAL) / %s / \n" % \
                                ', '.join(["%d" % i for i in self.colors])
        string += "\n\
C\n\
C     Particle type according to PDG:\n\
C\n\
      DATA (PDG_TYPE(IPOS), IPOS=1, NEXTERNAL) / %s / \n " % \
                                ', '.join(["%d" % i for i in self.pdg_codes])
        
        return string
             
                 
    def find_borns(self): #test written
        """finds the underlying borns for a given fks real process"""
        dict ={}
        for i in self.leglist:
            if i.get('state'):
                for j in self.leglist:
                    if j.get('number') != i.get('number') :
                        ijlist = fks_common.combine_ij(i, j, self.model, dict)
                        for ij in ijlist:
                            born = FKSBornProcess(self.real_proc, i, j, ij)
                            if born.amplitude.get('diagrams'):
                                self.borns.append(born)

        

###class FKSDirectory(MG.PhysicsObject):
###    
###    def __init__(self, born_amplitude = None, i_fks = 0, ij_isglu = 0):
###
###        self.born_amp = born_amplitude
###        self.spectators = []
###        self.soft_borns = {}
###        self.ifks = i_fks['number'] 
###        self.ijglu = ij_isglu
###        
###        if born_amplitude:
###            part_i = born_amplitude['process']['model'].get('particle_dict')[i_fks['id']]
###            
###            #check that the unresolved particle is a gluon, otherwise no soft sing.
###            if part_i['color'] == 8 and part_i['mass'].lower() == 'zero':
###            #loop over possible color links
###                for m in born_amplitude['process']['legs']: 
###                    for n in born_amplitude['process']['legs']:
###                        #print m['number'], n['number']
###                        part_m =  born_amplitude['process']['model'].get('particle_dict')[m['id']]
###                        part_n =  born_amplitude['process']['model'].get('particle_dict')[n['id']]
###
###                        if part_m['color'] != 1 and part_n['color'] != 1:
###                            if m != n or part_m['mass'].lower() != 'zero':
###                                self.spectators.append([m,n])
###                                link = self.define_link(m,n)
###                                helas = FKSHelasMultiProcess(self.born_amp, link)
###                                self.soft_borns[(m['number'], n['number'])] = \
###                                            helas
###    
###
###def insert_link( col_basis, link):
###    """returns a new color basis obtained from col_basis by inserting the 
###    given color link"""
###    min_index = min(sum([sum([sum([[i for i in col] for col in str],[]) \
###                    for str in dic.values() ],[]) for dic in col_basis],[]))
###    if min_index > 0:
###        min_index = -1000
###    #prepare the color string corresponding to the link
###    c_link = color_algebra.ColorString()
###    replace = []
###
###    iglu = 2* min_index
###    min_index -= 1
###    #the second-to-last index of the t is the triplet, the last is the anti-triplet        
###    if link[0] != link[1]:
###        #two different particles
###        for part in link:
###            if part['color'] == 3:
###                c_link.product(color_algebra.ColorString([
###                               color_algebra.T(iglu,part['number'], min_index)]))
###                c_link.coeff = c_link.coeff * (-1)
###            elif part['color'] == -3:
###                c_link.product(color_algebra.ColorString([
###                               color_algebra.T(iglu,min_index,part['number'])]))
###            elif part['color'] == 8:
###                c_link.product(color_algebra.ColorString(init_list = [
###                               color_algebra.f(min_index,iglu,part['number'])], 
###                               is_imaginary =True))
###            #sixtets.....
###            replace.append([part['number'], min_index])
###            min_index -= 1
###
###    else:
###        part = link[0]
###        #same particle (eg: top quark)
###        if part['color'] == 3:
###            c_link = color_algebra.ColorString(
###                      [ color_algebra.T(iglu, iglu, part['number'], min_index -1)
###                      ])
###        elif part['color'] == -3:
###            c_link = color_algebra.ColorString(
###                      [ color_algebra.T(iglu, iglu, min_index-1, part['number'])
###                      ])
###        replace.append([part['number'], min_index -1] )
###        c_link.coeff = c_link.coeff* fractions.Fraction(1,2)
###
###    
###    for dict in col_basis:
###        for string in dict.values():
###  
###            for col in string:
###                for ind in col:
###                    for pair in replace:
###                        if ind == pair[0]:
###                            col[col.index(ind)] = pair[1]
###            string.product(c_link)
###    return col_basis
###    
###    
###class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
###    """class inherited from HMP to werite out the FKS matrix element"""
###    
###    def __init__(self, argument=None, link=[0,0]):
###        """Allow initialization with AmplitudeList"""
###        
###        if isinstance(argument, diagram_generation.AmplitudeList):
###
###            super(FKSHelasMultiProcess, self).__init__()
###            self.generate_matrix_elementsFKS(argument, link)
###        elif isinstance(argument, diagram_generation.MultiProcess):
###            super(FKSHelasMultiProcess, self).__init__()
###            self.generate_matrix_elementsFKS(argument.get('amplitudes'), link)
###        elif isinstance(argument, diagram_generation.Amplitude):
###            super(FKSHelasMultiProcess, self).__init__()
###            self.generate_matrix_elementsFKS(\
###                diagram_generation.AmplitudeList([argument]), link)
###        elif argument:
###            #call the mother routine
###            super(FKSHelasMultiProcess, self).__init__(argument, link)
###        else:
###            # call the mother routine
###            super(FKSHelasMultiProcess, self).__init__()
###            
###                
###        
###    def generate_matrix_elementsFKS(self, amplitudes, link):
###        """Generate the HelasMatrixElements for the Amplitudes,
###        identifying processes with identical matrix elements, as
###        defined by HelasMatrixElement.__eq__"""
###
###        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
###                  "%s is not valid AmplitudeList" % repr(amplitudes)
###
###        # Keep track of already generated color objects, to reuse as
###        # much as possible
###        list_colorize = []
###        self.list_color_basis = []
###        self.list_color_basis_link = []
###        self.list_color_matrices = []
###
###        matrix_elements = self.get('matrix_elements')
###
###        while amplitudes:
###    
###            # Pop the amplitude to save memory space
###            amplitude = amplitudes.pop(0)
###            if isinstance(amplitude, diagram_generation.DecayChainAmplitude):
###                self.matrix_element_list = HelasDecayChainProcess(amplitude).\
###                                      combine_decay_chain_processes()
###            else:
###                logger.info("Generating Helas calls for %s" % \
###                         amplitude.get('process').nice_string().\
###                                           replace('Process', 'process'))
###                self.matrix_element_list = [helas_objects.HelasMatrixElement(amplitude,
###                                                          gen_color=False)]
###            for matrix_element in self.matrix_element_list:
###                assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
###                          "Not a HelasMatrixElement: %s" % matrix_element
###
###                try:
###                    # If an identical matrix element is already in the list,
###                    # then simply add this process to the list of
###                    # processes for that matrix element
###                    other_processes = matrix_elements[\
###                    matrix_elements.index(matrix_element)].get('processes')
###                    logger.info("Combining process with %s" % \
###                      other_processes[0].nice_string().replace('Process: ', ''))
###                    other_processes.extend(matrix_element.get('processes'))
###                except ValueError:
###                    # Otherwise, if the matrix element has any diagrams,
###                    # add this matrix element.
###
###                    if matrix_element.get('processes') and \
###                           matrix_element.get('diagrams'):
###
###                        matrix_elements.append(matrix_element)
###    
###                        # Always create an empty color basis, and the
###                        # list of raw colorize objects (before
###                        # simplification) associated with amplitude
###                        col_basis = color_amp.ColorBasis()
###                        col_basis_link = color_amp.ColorBasis()
###                        new_amp = matrix_element.get_base_amplitude()
###                        matrix_element.set('base_amplitude', new_amp)
###                        colorize_obj = col_basis.create_color_dict_list(new_amp)
###                         
###                        #print colorize_obj
###                        colorize_link = insert_link(\
###                          col_basis_link.create_color_dict_list(new_amp), link)
###                        
###
###                        # Create color basis and color
###                        # matrix accordingly
###                        list_colorize.append(colorize_obj)
###                        col_basis.build()
###                        #print col_basis                       
###
###                        col_basis_link.build()
###                        self.list_color_basis.append(col_basis)
###                        self.list_color_basis_link.append(col_basis_link)
###                        #print "BASIS", col_basis 
###                        #print "BASIS LINK", col_basis_link 
####                        print "in fksHMP "
####                        print "col basis orig ", col_basis
####                        print "col basis link ", col_basis_link
###                                                
###                        col_matrix = color_amp.ColorMatrix(col_basis, col_basis_link)
###
####                        print "COLOR MATRIX", col_matrix
###                        self.list_color_matrices.append(col_matrix)
###                        
###                        col_index = -1
###                        logger.info(\
###                              "Processing color information for %s" % \
###                              matrix_element.get('processes')[0].nice_string().\
###                                             replace('Process', 'process'))
###
###                matrix_element.set('color_basis', self.list_color_basis[col_index])
###                #print list_color_basis[col_index]
###                #print col_basis_link
###                matrix_element.set('color_matrix',
###                                   self.list_color_matrices[col_index])
###            #    print list_color_matrices[col_index]
###            
###    
###    
###    
###    
###    
###    
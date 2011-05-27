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

"""Definitions of the Helas objects needed for the implementation of MadFKS 
from born"""


import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_real as fks_real
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_real_helas_objects')




class FKSHelasMultiProcessFromReals(helas_objects.HelasMultiProcess):
    """class to generate the helas calls for a FKSMultiProcess,
    starting from real emission"""
    
    def __init__(self, fksmulti, gen_color =True, decay_ids =[]):
        """Initialization from a FKSMultiProcess"""
        #super(FKSHelasMultiProcessFromReals, self).__init__()
        self['matrix_elements'] = self.generate_matrix_elements_fks(
                                fksmulti['real_processes'], 
                                gen_color, decay_ids)
        
    
    def get_used_lorentz(self):
        """Return a list of (lorentz_name, conjugate, outgoing) with
        all lorentz structures used by this HelasMultiProcess."""

        helas_list = []

        for me in self.get('matrix_elements'):
            helas_list.extend(me.get_used_lorentz())

        return list(set(helas_list))
    

    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        coupling_list = []

        for me in self.get('matrix_elements'):
            coupling_list.extend([c for l in me.get_used_couplings() for c in l])

        return list(set(coupling_list))
    
    
    def get_matrix_elements(self):
        """Extract the list of matrix elements"""

        return self.get('matrix_elements')        
        

    def generate_matrix_elements_fks(self, fksprocs, gen_color = True,
                                 decay_ids = []):
        """Generate the HelasMatrixElements for the amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__. Returns a
        HelasMatrixElementList and an amplitude map (used by the
        SubprocessGroup functionality). decay_ids is a list of decayed
        particle ids, since those should not be combined even if
        matrix element is identical."""

        assert isinstance(fksprocs, fks_real.FKSProcessFromRealsList), \
                  "%s is not valid FKSProcessFromRealsList" % \
                   repr(fksprocs)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        list_color_links =[]
        list_color_basis = []
        list_color_matrices = []
        born_me_list = []
        me_id_list = []

        matrix_elements = FKSHelasProcessFromRealsList()

        while fksprocs:
            # Pop the amplitude to save memory space
            proc = fksprocs.pop(0)
#            if isinstance(proc, diagram_generation.DecayChainAmplitude):
#                matrix_element_list = HelasDecayChainProcess(amplitude).\
#                                      combine_decay_chain_processes()
#            else:
            logger.info("Generating Helas calls for FKS process %s" % \
                         proc.real_amp.get('process').nice_string().\
                                           replace('Process', 'process'))
            matrix_element_list = [FKSHelasProcessFromReals(proc, born_me_list,
                                                           me_id_list,
                                                          decay_ids=decay_ids,
                                                          gen_color=False)]
            for matrix_element in matrix_element_list:
                assert isinstance(matrix_element, FKSHelasProcessFromReals), \
                          "Not a FKSHelasProcessFromReals: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other = \
                          matrix_elements[matrix_elements.index(matrix_element)]
                    other.add_process(matrix_element)
                    logger.info("Combining process with %s" % \
                      other.real_matrix_element.get('processes')[0].nice_string().replace('Process: ', ''))
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.real_matrix_element.get('processes') and \
                           matrix_element.real_matrix_element.get('diagrams'):
                        matrix_elements.append(matrix_element)

                        if not gen_color:
                            continue

                        # Always create an empty color basis, and the
                        # list of raw colorize objects (before
                        # simplification) associated with amplitude
                        col_basis = color_amp.ColorBasis()
                        new_amp = matrix_element.real_matrix_element.get_base_amplitude()
                        matrix_element.real_matrix_element.set('base_amplitude', new_amp)
                        colorize_obj = col_basis.create_color_dict_list(new_amp)

                        try:
                            # If the color configuration of the ME has
                            # already been considered before, recycle
                            # the information
                            col_index = list_colorize.index(colorize_obj)
                            logger.info(\
                              "Reusing existing color information for %s" % \
                              matrix_element.real_matrix_element.get('processes')[0].nice_string().\
                                                 replace('Process', 'process'))
                            
                            
                        except ValueError:
                            # If not, create color basis and color
                            # matrix accordingly
                            list_colorize.append(colorize_obj)
                            col_basis.build()
                            list_color_basis.append(col_basis)
                            col_matrix = color_amp.ColorMatrix(col_basis)
                            list_color_matrices.append(col_matrix)
                            col_index = -1
    
                if gen_color:
                    matrix_element.real_matrix_element.set('color_basis',
                                       list_color_basis[col_index])
                    matrix_element.real_matrix_element.set('color_matrix',
                                       list_color_matrices[col_index])
                    
        return matrix_elements    


class FKSHelasProcessFromRealsList(MG.PhysicsObjectList):
    """class to handle lists of FKSHelasProcessesFromReals"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromReals for the list."""
        return isinstance(obj, FKSHelasProcessFromReals)   
    
    
class FKSHelasProcessFromReals(object):
    """class to generate the Helas calls for a FKSProcessFromReals. Contains:
    -- real emission ME
    -- list of FKSHelasBornProcesses
    -- fks_inc string (content of file fks.inc)
    """
    
    def __init__(self, fksproc=None, me_list =[], me_id_list=[], **opts):
        """ constructor, starts from a FKSProcessFromReals, sets borns and ME"""
        
        if fksproc != None:
            self.born_processes = []
            for proc in fksproc.borns:
                self.born_processes.append(
                        FKSHelasBornProcess(proc, me_list, me_id_list, **opts))
            self.real_matrix_element = helas_objects.HelasMatrixElement(
                                    fksproc.real_amp, **opts)
            self.fks_inc_string = fksproc.get_fks_inc_string()
    
    def get(self, key):
        """the get function references to real_matrix_element"""
        return self.real_matrix_element.get(key)
    
    def get_used_lorentz(self):
        """the get_used_lorentz function references to real_matrix_element and
        to the borns"""
        lorentz_list = self.real_matrix_element.get_used_lorentz()
        for born in self.born_processes:
            lorentz_list.extend(born.matrix_element.get_used_lorentz())
        return list(set(lorentz_list))
#        return lorentz_list
    
    def get_used_couplings(self):
        """the get_used_couplings function references to real_matrix_element and
        to the borns"""
        coupl_list = self.real_matrix_element.get_used_couplings()
        for born in self.born_processes:
            coupl_list.extend([c for l in\
                        born.matrix_element.get_used_couplings() for c in l])
        #return list(set(coupl_list))
        return coupl_list    
    
    def __eq__(self, other):
        """the equality between two FKSHelasProcesses is defined up to the 
        color links"""
                    
        if self.real_matrix_element != other.real_matrix_element:
            return False
        borns2 = copy.copy(other.born_processes)
        for born in  self.born_processes:
            try:
                borns2.remove(born)
            except:
                return False  
        return borns2 == []
    
    def add_process(self, other): 
        """adds processes from born and reals of other to itself"""
        self.real_matrix_element.get('processes').extend(
                other.real_matrix_element.get('processes'))
        for born1, born2 in zip(self.born_processes, other.born_processes):
            born1.matrix_element.get('processes').extend(
                born2.matrix_element.get('processes'))  


class FKSHelasBornProcess(object): #test written
    """class to generate the Helas calls for a FKSRealProcess
    contains:
    -- i/j fks
    -- ijglu
    -- matrix element
    -- color links
    -- leg permutation<<REMOVED"""
    
    def __init__(self, fksbornproc=None, me_list = [], me_id_list =[], **opts):
        """constructor, starts from a fksrealproc and then calls the
        initialization for HelasMatrixElement.
        Sets i/j fks and the permutation"""
        
        if fksbornproc != None:
            self.i_fks = fksbornproc.i_fks
            self.j_fks = fksbornproc.j_fks
            self.ijglu = fksbornproc.ijglu

            self.matrix_element = helas_objects.HelasMatrixElement(
                                    fksbornproc.amplitude, **opts)
            self.color_links = fksbornproc.find_color_links()

    
    def __eq__(self, other):
        """Equality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""

        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        """Inequality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return not self.__eq__(other)
    
    
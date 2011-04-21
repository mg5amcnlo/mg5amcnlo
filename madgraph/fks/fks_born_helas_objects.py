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

"""Definitions of the Helas objects needed for the implementation of MadFKS"""


import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.fks.fks_born as fks_born
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_born_helas_objects')


class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
    """class to generate the helas calls for a FKSMultiProcess"""
    
    def __init__(self, fksmulti, gen_color =True, decay_ids =[]):
        """Initialization from a FKSMultiProcess"""
        self['matrix_elements'] = self.generate_matrix_elements_fks(
                                fksmulti['born_processes'], 
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
            coupling_list.extend(me.get_used_couplings())

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

        assert isinstance(fksprocs, fks_born.FKSProcessFromBornList), \
                  "%s is not valid FKSProcessFromBornList" % \
                   repr(FKSProcessFromBornList)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        list_color_links =[]
        list_color_basis = []
        list_color_matrices = []
        real_me_list = []
        me_id_list = []

        matrix_elements = FKSHelasProcessFromBornList()

        while fksprocs:
            # Pop the amplitude to save memory space
            proc = fksprocs.pop(0)
#            if isinstance(proc, diagram_generation.DecayChainAmplitude):
#                matrix_element_list = HelasDecayChainProcess(amplitude).\
#                                      combine_decay_chain_processes()
#            else:
            logger.info("Generating Helas calls for FKS process %s" % \
                         proc.born_amp.get('process').nice_string().\
                                           replace('Process', 'process'))
            matrix_element_list = [FKSHelasProcessFromBorn(proc, real_me_list,
                                                           me_id_list,
                                                          decay_ids=decay_ids,
                                                          gen_color=False)]
            for matrix_element in matrix_element_list:
                assert isinstance(matrix_element, FKSHelasProcessFromBorn), \
                          "Not a FKSHelasProcessFromBorn: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other = \
                          matrix_elements[matrix_elements.index(matrix_element)]
                    other.add_process(matrix_element)
                    logger.info("Combining process with %s" % \
                      other.born_matrix_element.get('processes')[0].nice_string().replace('Process: ', ''))
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.born_matrix_element.get('processes') and \
                           matrix_element.born_matrix_element.get('diagrams'):
                        matrix_elements.append(matrix_element)

                        if not gen_color:
                            continue

                        # Always create an empty color basis, and the
                        # list of raw colorize objects (before
                        # simplification) associated with amplitude
                        col_basis = color_amp.ColorBasis()
                        new_amp = matrix_element.born_matrix_element.get_base_amplitude()
                        matrix_element.born_matrix_element.set('base_amplitude', new_amp)
                        colorize_obj = col_basis.create_color_dict_list(new_amp)

                        try:
                            # If the color configuration of the ME has
                            # already been considered before, recycle
                            # the information
                            col_index = list_colorize.index(colorize_obj)
                            logger.info(\
                              "Reusing existing color information for %s" % \
                              matrix_element.born_matrix_element.get('processes')[0].nice_string().\
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
                            list_color_links.append(
                                    FKSHelasProcessFromBorn.insert_color_links(
                                            col_basis, colorize_obj, 
                                            proc.color_links))
                            logger.info(\
                              "Processing color information for %s" % \
                              matrix_element.born_matrix_element.get('processes')[0].nice_string().\
                                             replace('Process', 'process'))
                            

                if gen_color:
                    matrix_element.born_matrix_element.set('color_basis',
                                       list_color_basis[col_index])
                    matrix_element.born_matrix_element.set('color_matrix',
                                       list_color_matrices[col_index])
                    matrix_element.color_links = list_color_links[col_index]
                    
        return matrix_elements    


class FKSHelasProcessFromBornList(MG.PhysicsObjectList):
    """class to handle lists of FKSHelasProcessesFromBorn"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromBorn for the list."""
        return isinstance(obj, FKSHelasProcessFromBorn)
    
    
    
class FKSHelasProcessFromBorn(object):
    """class to generate the Helas calls for a FKSProcessFromBorn. Contains:
    -- born ME
    -- list of FKSHelasRealProcesses
    -- color links"""
    
    def __init__(self, fksproc=None, me_list =[], me_id_list=[], **opts):#test written
        """ constructor, starts from a FKSProcessFromBorn, 
        sets reals and color links"""
        
        if fksproc != None:
            self.color_links = []
            self.real_processes = []
            for proc in fksproc.real_amps:
                self.real_processes.append(
                        FKSHelasRealProcess(proc, me_list, me_id_list, **opts))
            self.born_matrix_element = helas_objects.HelasMatrixElement(
                                    fksproc.born_amp, **opts)
    
    def __eq__(self, other):
        """the equality between two FKSHelasProcesses is defined up to the 
        color links"""
                    
        if self.born_matrix_element != other.born_matrix_element:
            return False
#        print "same born ",self.born_matrix_element.get('processes')[0].nice_string()
#        print other.born_matrix_element.get('processes')[0].nice_string()
        if len(self.real_processes) != len(other.real_processes):
#            print "Wrong length of reals"
            return False
        reals2 = copy.copy(other.real_processes)
        for real in  self.real_processes:
            try:
                reals2.remove(real)
#                print "removed real ",real.matrix_element.get('processes')[0].nice_string(),\
#                "  i_fks ",real.i_fks, "  j_fks ",real.j_fks
            except:
#                print "Failed remove ",real.matrix_element.get('processes')[0].nice_string(),\
#                "  i_fks ",real.i_fks, "  j_fks ",real.j_fks
#                for r in reals2:
#                    print "left ", r.matrix_element.get('processes')[0].nice_string()
                return False  

        return True
    
    def add_process(self, other): #test written
        """adds processes from born and reals of other to itself"""
        self.born_matrix_element.get('processes').extend(
                other.born_matrix_element.get('processes'))
        for real1, real2 in zip(self.real_processes, other.real_processes):
            real1.matrix_element.get('processes').extend(
                real2.matrix_element.get('processes'))  
            
    @staticmethod
    def insert_color_links(col_basis, col_obj, links):
        """insert the color links in col_obj: returns a list of dictionaries
        (one for each link) with the following entries:
        --link: the numbers of the linked legs
        --link_basis: the linked color basis
        --link_matrix: the color matrix created from the original basis and the linked one
        """   
        
        assert isinstance(col_basis, color_amp.ColorBasis)
        assert isinstance(col_obj, list)
        result =[]
        for link in links:
            this = {}
            #define the link
            l =[]
            for leg in link['legs']:
                l.append(leg.get('number'))
            this['link'] = l
            
            #replace the indices in col_obj of the linked legs according to
            #   link['replacements']
            # and extend the color strings
            
            this_col_obj = []
            for old_dict in col_obj:
                dict = copy.copy(old_dict)
                for k, string in dict.items():
                    dict[k]=string.create_copy()
                    for col in dict[k]:
                        for ind in col:
                            for pair in link['replacements']:
                                if ind == pair[0]:
                                    col[col.index(ind)] = pair[1]
                    dict[k].extend(link['string'])
                this_col_obj.append(dict)
            basis_link = color_amp.ColorBasis()
            for ind, dict in enumerate(this_col_obj):
                basis_link.update_color_basis(dict, ind)
            
            this['link_basis'] = basis_link
            this['link_matrix'] = color_amp.ColorMatrix(col_basis,basis_link)
                
            result.append(this)
                
        return result
            
    
class FKSHelasRealProcess(object): #test written
    """class to generate the Helas calls for a FKSRealProcess
    contains:
    -- i/j fks
    -- matrix element
    -- leg permutation<<REMOVED"""
    
    def __init__(self, fksrealproc=None, me_list = [], me_id_list =[], **opts):
        """constructor, starts from a fksrealproc and then calls the
        initialization for HelasMatrixElement.
        Sets i/j fks and the permutation"""
        
        if fksrealproc != None:
            pdgs = fksrealproc.pdgs
 ##           self.permutation= fksrealproc.permutation
            self.i_fks = fksrealproc.i_fks
            self.j_fks = fksrealproc.j_fks
       #     print "in FKSHelasRealProc  i ", self.i_fks, "   j ", self.j_fks
            
            try:
                matrix_element = copy.copy(me_list[me_id_list.index(pdgs)])
                matrix_element.set('processes', 
                                   copy.copy(matrix_element.get('processes')))
                self.matrix_element = matrix_element
            except ValueError:
                self.matrix_element = helas_objects.HelasMatrixElement(
                                    fksrealproc.amplitude, **opts)
                me_list.append(self.matrix_element)
                me_id_list.append(pdgs)
                
       #     for p in self.matrix_element.get('processes'):
       #         print p.nice_string()
    
    def __eq__(self, other):
        """Equality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        """Inequality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return not self.__eq__(other)



        
        
    
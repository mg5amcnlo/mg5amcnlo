################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Definitions of the Helas objects needed for the implementation of MadFKS 
from born"""


import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.loop.loop_helas_objects as loop_helas_objects
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_helas_objects')


class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
    """class to generate the helas calls for a FKSMultiProcess"""

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSHelasMultiProcess, self).get_sorted_keys()
        keys += ['real_matrix_elements', ['has_isr'], ['has_fsr']]
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'real_matrix_elements':
            if not isinstance(value, helas_objects.HelasMultiProcess):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for real_matrix_element " % str(value)                             
    
    def __init__(self, fksmulti, loop_optimized = False, gen_color =True, decay_ids =[]):
        """Initialization from a FKSMultiProcess"""

        #swhich the other loggers off
        loggers_off = [logging.getLogger('madgraph.diagram_generation'),
                       logging.getLogger('madgraph.helas_objects')]
        old_levels = [logg.level for logg in loggers_off]
        for logg in loggers_off:
            logg.setLevel(logging.WARNING)

        self.loop_optimized = loop_optimized

        logger.info('Generating real emission matrix-elements...')
        self['real_matrix_elements'] = self.generate_matrix_elements(
                copy.copy(fksmulti['real_amplitudes']), combine_matrix_elements = False)

        self['matrix_elements'] = self.generate_matrix_elements_fks(
                                fksmulti, 
                                gen_color, decay_ids)
        self['initial_states']=[]

        self['has_isr'] = fksmulti['has_isr']
        self['has_fsr'] = fksmulti['has_fsr']
        self['has_loops'] = len(self.get_virt_matrix_elements()) > 0 

        for i, logg in enumerate(loggers_off):
            logg.setLevel(old_levels[i])
        
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


    def get_virt_matrix_elements(self):
        """Extract the list of virtuals matrix elements"""
        return [me.virt_matrix_element for me in self.get('matrix_elements') \
                if me.virt_matrix_element]        
        

    def generate_matrix_elements_fks(self, fksmulti, gen_color = True,
                                 decay_ids = []):
        """Generate the HelasMatrixElements for the amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__. Returns a
        HelasMatrixElementList and an amplitude map (used by the
        SubprocessGroup functionality). decay_ids is a list of decayed
        particle ids, since those should not be combined even if
        matrix element is identical."""

        fksprocs = fksmulti['born_processes']
        assert isinstance(fksprocs, fks_base.FKSProcessList), \
                  "%s is not valid FKSProcessList" % \
                   repr(fksprocs)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        list_color_links =[]
        list_color_basis = []
        list_color_matrices = []
        real_me_list = []
        me_id_list = []

        matrix_elements = FKSHelasProcessList()

        for i, proc in enumerate(fksprocs):
            logger.info("Generating Helas calls for FKS %s (%d / %d)" % \
              (proc.born_amp.get('process').nice_string(print_weighted = False).\
                                                  replace('Process', 'process'),
                i + 1, len(fksprocs)))
            matrix_element_list = [FKSHelasProcess(proc, self['real_matrix_elements'],
                                                           fksmulti['real_amplitudes'],
                                                          loop_optimized = self.loop_optimized,
                                                          decay_ids=decay_ids,
                                                          gen_color=False)]
            for matrix_element in matrix_element_list:
                assert isinstance(matrix_element, FKSHelasProcess), \
                          "Not a FKSHelasProcess: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other = \
                          matrix_elements[matrix_elements.index(matrix_element)]
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
                              matrix_element.born_matrix_element.get('processes')\
                              [0].nice_string(print_weighted=False).\
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

                            logger.info(\
                              "Processing color information for %s" % \
                              matrix_element.born_matrix_element.\
                              get('processes')[0].nice_string(print_weighted=False).\
                                             replace('Process', 'process'))
                        matrix_element.born_matrix_element.set('color_basis',
                                           list_color_basis[col_index])
                        matrix_element.born_matrix_element.set('color_matrix',
                                           list_color_matrices[col_index])                    
                else:
                    # this is in order not to handle valueErrors coming from other plaeces,
                    # e.g. from the add_process function
                    other.add_process(matrix_element)

        for me in matrix_elements:
            me.set_color_links()
        return matrix_elements    


class FKSHelasProcessList(MG.PhysicsObjectList):
    """class to handle lists of FKSHelasProcesses"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcess for the list."""
        return isinstance(obj, FKSHelasProcess)
    
    
class FKSHelasProcess(object):
    """class to generate the Helas calls for a FKSProcess. Contains:
    -- born ME
    -- list of FKSHelasRealProcesses
    -- color links"""
    
    def __init__(self, fksproc=None, real_me_list =[], real_amp_list=[], 
            loop_optimized = False, **opts):#test written
        """ constructor, starts from a FKSProcess, 
        sets reals and color links. Real_me_list and real_amp_list are the lists of pre-genrated
        matrix elements in 1-1 correspondence with the amplitudes"""
        
        if fksproc != None:
            self.born_matrix_element = helas_objects.HelasMatrixElement(
                                    fksproc.born_amp, **opts)
            self.real_processes = []
            self.orders = fksproc.born_proc.get('orders')
            self.perturbation = fksproc.perturbation
            real_amps_new = []
            # combine for example u u~ > t t~ and d d~ > t t~
            for proc in fksproc.real_amps:
                fksreal_me = FKSHelasRealProcess(proc, real_me_list, real_amp_list, **opts)
                try:
                    other = self.real_processes[self.real_processes.index(fksreal_me)]
                    other.matrix_element.get('processes').extend(\
                            fksreal_me.matrix_element.get('processes') )
                except ValueError:
                    if fksreal_me.matrix_element.get('processes') and \
                            fksreal_me.matrix_element.get('diagrams'):
                        self.real_processes.append(fksreal_me)
                        real_amps_new.append(proc)
            fksproc.real_amps = real_amps_new
            if fksproc.virt_amp:
                self.virt_matrix_element = \
                  loop_helas_objects.LoopHelasMatrixElement(fksproc.virt_amp, 
                          optimized_output = loop_optimized)
            else: 
                self.virt_matrix_element = None
#            self.color_links_info = fksproc.find_color_links()
            self.color_links = []

    def set_color_links(self):
        """this function computes and returns the color links, it should be called
        after the initialization and the setting of the color basis"""
        if not self.color_links:
            legs = self.born_matrix_element.get('base_amplitude').get('process').get('legs')
            model = self.born_matrix_element.get('base_amplitude').get('process').get('model')
            color_links_info = fks_common.find_color_links(fks_common.to_fks_legs(legs, model),
                        symm = True,pert = self.perturbation)
            col_basis = self.born_matrix_element.get('color_basis')
            self.color_links = fks_common.insert_color_links(col_basis,
                                col_basis.create_color_dict_list(
                                    self.born_matrix_element.get('base_amplitude')),
                                color_links_info)    

    def get_fks_info_list(self):
        """Returns the list of the fks infos for all processes in the format
        {n_me, pdgs, fks_info}, where n_me is the number of real_matrix_element the configuration
        belongs to"""
        info_list = []
        for n, real in enumerate(self.real_processes):
            pdgs = [l['id'] for l in real.matrix_element.get_base_amplitude()['process']['legs']]
            for info in real.fks_infos:
                info_list.append({'n_me' : n + 1,'pdgs' : pdgs, 'fks_info' : info})
        return info_list
        

    def get_lh_pdg_string(self):
        """Returns the pdgs of the legs in the form "i1 i2 -> f1 f2 ...", which may
        be useful (eg. to be written in a B-LH order file)"""

        initial = ''
        final = ''
        for leg in self.born_matrix_element.get('processes')[0].get('legs'):
            if leg.get('state'):
                final += '%d ' % leg.get('id')
            else:
                initial += '%d ' % leg.get('id')
        return initial + '-> ' + final


    def get(self, key):
        """the get function references to born_matrix_element"""
        return self.born_matrix_element.get(key)
    
    def get_used_lorentz(self):
        """the get_used_lorentz function references to born, reals
        and virtual matrix elements"""
        lorentz_list = self.born_matrix_element.get_used_lorentz()
        for real in self.real_processes:
            lorentz_list.extend(real.matrix_element.get_used_lorentz())
        if self.virt_matrix_element:
            lorentz_list.extend(self.virt_matrix_element.get_used_lorentz())

        return list(set(lorentz_list))
    
    def get_used_couplings(self):
        """the get_used_couplings function references to born, reals
        and virtual matrix elements"""
        coupl_list = self.born_matrix_element.get_used_couplings()
        for real in self.real_processes:
            coupl_list.extend([c for c in\
                        real.matrix_element.get_used_couplings()])
        if self.virt_matrix_element:
            coupl_list.extend(self.virt_matrix_element.get_used_couplings())
        return coupl_list    
    
    def __eq__(self, other):
        """the equality between two FKSHelasProcesses is defined up to the 
        color links"""
        selftag = helas_objects.IdentifyMETag.create_tag(self.born_matrix_element.get('base_amplitude'))
        othertag = helas_objects.IdentifyMETag.create_tag(other.born_matrix_element.get('base_amplitude'))
                    
        if self.born_matrix_element != other.born_matrix_element or \
                selftag != othertag:
            return False

        reals2 = copy.copy(other.real_processes)
        for real in  self.real_processes:
            try:
                reals2.remove(real)
            except ValueError:
                return False  
        if not reals2:
            return True
        else: 
            return False
    
    def add_process(self, other): #test written, ppwj
        """adds processes from born and reals of other to itself. Note that 
        corresponding real processes may not be in the same order. This is 
        taken care of by constructing the list of self_reals."""
        self.born_matrix_element.get('processes').extend(
                other.born_matrix_element.get('processes'))
        if self.virt_matrix_element and other.virt_matrix_element:
            self.virt_matrix_element.get('processes').extend(
                    other.virt_matrix_element.get('processes'))
        self_reals = [real.matrix_element for real in self.real_processes]
        for oth_real in other.real_processes:
            this_real = self.real_processes[self_reals.index(oth_real.matrix_element)]
            #need to store pdg lists rather than processes in order to keep mirror processes different
            this_pdgs = [[leg['id'] for leg in proc['legs']] \
                    for proc in this_real.matrix_element['processes']]
            for oth_proc in oth_real.matrix_element['processes']:
                oth_pdgs = [leg['id'] for leg in oth_proc['legs']]
                if oth_pdgs not in this_pdgs:
                    this_real.matrix_element['processes'].append(oth_proc)
                    this_pdgs.append(oth_pdgs)

 #                       if p not in self.real_processes[\
 #                       self_reals.index(oth_real.matrix_element)].matrix_element['processes']])
            
    
class FKSHelasRealProcess(object): #test written
    """class to generate the Helas calls for a FKSRealProcess
    contains:
    -- colors
    -- charges
    -- i/j/ij fks, ij refers to the born leglist
    -- ijglu
    -- need_color_links
    -- fks_j_from_i
    -- matrix element
    -- is_to_integrate
    -- leg permutation<<REMOVED"""
    
    def __init__(self, fksrealproc=None, real_me_list = [], real_amp_list =[], **opts):
        """constructor, starts from a fksrealproc and then calls the
        initialization for HelasMatrixElement.
        Sets i/j fks and the permutation.
        real_me_list and real_amp_list are the lists of pre-generated matrix elements in 1-1 
        correspondance with the amplitudes"""
        
        if fksrealproc != None:
            self.isfinite = False
            self.colors = fksrealproc.colors
            self.charges = fksrealproc.charges
            self.fks_infos = fksrealproc.fks_infos
            self.is_to_integrate = fksrealproc.is_to_integrate

            if len(real_me_list) != len(real_amp_list):
                raise fks_common.FKSProcessError(
                        'not same number of amplitudes and matrix elements: %d, %d' % \
                                (len(real_amp_list), len(real_me_list)))
            if real_me_list and real_amp_list:
                self.matrix_element = copy.deepcopy(real_me_list[real_amp_list.index(fksrealproc.amplitude)])
                self.matrix_element['processes'] = copy.deepcopy(self.matrix_element['processes'])
            else:
                logger.info('generating matrix element...')
                self.matrix_element = helas_objects.HelasMatrixElement(
                                                  fksrealproc.amplitude, **opts)
                #generate the color for the real
                self.matrix_element.get('color_basis').build(
                                    self.matrix_element.get('base_amplitude'))
                self.matrix_element.set('color_matrix',
                                 color_amp.ColorMatrix(
                                    self.matrix_element.get('color_basis')))
            #self.fks_j_from_i = fksrealproc.find_fks_j_from_i()
            self.fks_j_from_i = fksrealproc.fks_j_from_i

    def get_nexternal_ninitial(self):
        """Refers to the matrix_element function"""
        return self.matrix_element.get_nexternal_ninitial()
    
    def __eq__(self, other):
        """Equality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        """Inequality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return not self.__eq__(other)



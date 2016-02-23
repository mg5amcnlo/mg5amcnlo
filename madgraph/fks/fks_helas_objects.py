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
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import copy
import logging
import array
import multiprocessing
import signal
import tempfile
import cPickle
import itertools
import os

logger = logging.getLogger('madgraph.fks_helas_objects')


#functions to be used in the ncores_for_proc_gen mode
def async_generate_real(args):
    i = args[0]
    real_amp = args[1]

    #amplitude generation
    amplitude = real_amp.generate_real_amplitude()
    helasreal = helas_objects.HelasMatrixElement(amplitude)
    logger.info('Generating real %s' % \
            real_amp.process.nice_string(print_weighted=False).replace('Process', 'process'))

    # Keep track of already generated color objects, to reuse as
    # much as possible
    list_colorize = []
    list_color_basis = []
    list_color_matrices = []
    
    # Now this keeps track of the color matrices created from the loop-born
    # color basis. Keys are 2-tuple with the index of the loop and born basis
    # in the list above and the value is the resulting matrix.
    dict_loopborn_matrices = {}
    # The dictionary below is simply a container for convenience to be 
    # passed to the function process_color.
    color_information = { 'list_colorize' : list_colorize,
                          'list_color_basis' : list_color_basis,
                          'list_color_matrices' : list_color_matrices,
                          'dict_loopborn_matrices' : dict_loopborn_matrices}

    helas_objects.HelasMultiProcess.process_color(helasreal,color_information)

    outdata = [amplitude,helasreal]

    output = tempfile.NamedTemporaryFile(delete = False)   
    cPickle.dump(outdata,output,protocol=2)
    output.close()
    
    return [output.name,helasreal.get_num_configs(),helasreal.get_nexternal_ninitial()[0]]


def async_generate_born(args):
    i = args[0]
    born = args[1]
    born_pdg_list = args[2]
    loop_orders = args[3]
    pdg_list = args[4]
    loop_optimized = args[5]
    OLP = args[6]
    realmapout = args[7]

    logger.info('Generating born %s' % \
            born.born_proc.nice_string(print_weighted=False).replace('Process', 'process'))
    
    #load informations on reals from temp files
    helasreal_list = []
    for amp in born.real_amps:
        idx = pdg_list.index(amp.pdgs)
        infilename = realmapout[idx]
        infile = open(infilename,'rb')
        realdata = cPickle.load(infile)
        infile.close()
        amp.amplitude = realdata[0]
        helasreal_list.append(realdata[1])
        
    born.link_born_reals()
        
    for amp in born.real_amps:
        amp.find_fks_j_from_i(born_pdg_list)        
    
    # generate the virtuals if needed
    has_loops = False
    if born.born_proc.get('NLO_mode') == 'all' and OLP == 'MadLoop':
        myproc = copy.copy(born.born_proc)
        # take the orders that are actually used by the matrix element
        myproc['orders'] = loop_orders
        myproc['legs'] = fks_common.to_legs(copy.copy(myproc['legs']))
        myamp = loop_diagram_generation.LoopAmplitude(myproc)
        if myamp.get('diagrams'):
            has_loops = True
            born.virt_amp = myamp
        
    helasfull = FKSHelasProcess(born, helasreal_list,
                                loop_optimized = loop_optimized,
                                decay_ids=[],
                                gen_color=False)

    processes = helasfull.born_matrix_element.get('processes')
    
    metag = helas_objects.IdentifyMETag.create_tag(helasfull.born_matrix_element.get('base_amplitude'))
    
    outdata = helasfull
    
    output = tempfile.NamedTemporaryFile(delete = False)   
    cPickle.dump(outdata,output,protocol=2)
    output.close()
    
    return [output.name,metag,has_loops,processes]


def async_finalize_matrix_elements(args):

    i = args[0]
    mefile = args[1]
    duplist = args[2]
    
    infile = open(mefile,'rb')
    me = cPickle.load(infile)
    infile.close()    

    #set unique id based on position in unique me list
    me.get('processes')[0].set('uid', i)

    # Always create an empty color basis, and the
    # list of raw colorize objects (before
    # simplification) associated with amplitude
    col_basis = color_amp.ColorBasis()
    new_amp = me.born_matrix_element.get_base_amplitude()
    me.born_matrix_element.set('base_amplitude', new_amp)
    colorize_obj = col_basis.create_color_dict_list(new_amp)

    col_basis.build()
    col_matrix = color_amp.ColorMatrix(col_basis)

    me.born_matrix_element.set('color_basis',col_basis)
    me.born_matrix_element.set('color_matrix',col_matrix)
    
    for iother,othermefile in enumerate(duplist):
        infileother = open(othermefile,'rb')
        otherme = cPickle.load(infileother)
        infileother.close()
        me.add_process(otherme)
        
    me.set_color_links()    
        
    initial_states=[]
    for fksreal in me.real_processes:
        # Pick out all initial state particles for the two beams
            initial_states.append(sorted(list(set((p.get_initial_pdg(1),p.get_initial_pdg(2)) for \
                                              p in fksreal.matrix_element.get('processes')))))
    
    if me.virt_matrix_element:
        has_virtual = True
    else:
        has_virtual = False
     
    #data to write to file
    outdata = me

    output = tempfile.NamedTemporaryFile(delete = False)   
    cPickle.dump(outdata,output,protocol=2)
    output.close()
    
    #data to be returned to parent process (filename plus small objects only)
    return [output.name,initial_states,me.get_used_lorentz(),me.get_used_couplings(),has_virtual]


class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
    """class to generate the helas calls for a FKSMultiProcess"""

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSHelasMultiProcess, self).get_sorted_keys()
        keys += ['real_matrix_elements', ['has_isr'], ['has_fsr'], 
                 'used_lorentz', 'used_couplings', 'max_configs', 'max_particles', 'processes']
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

        self['used_lorentz'] = []
        self['used_couplings'] = []
        self['processes'] = []

        self['max_particles'] = -1
        self['max_configs'] = -1

        if not fksmulti['ncores_for_proc_gen']:
            # generate the real ME's if they are needed.
            # note that it may not be always the case, e.g. it the NLO_mode is LOonly
            if fksmulti['real_amplitudes']:
                logger.info('Generating real emission matrix-elements...')
                self['real_matrix_elements'] = self.generate_matrix_elements(
                        copy.copy(fksmulti['real_amplitudes']), combine_matrix_elements = False)
            else:
                self['real_matrix_elements'] = helas_objects.HelasMatrixElementList()

            self['matrix_elements'] = self.generate_matrix_elements_fks(
                                    fksmulti, 
                                    gen_color, decay_ids)
            self['initial_states']=[]
            self['has_loops'] = len(self.get_virt_matrix_elements()) > 0 

        else: 
            self['has_loops'] = False
            #more efficient generation
            born_procs = fksmulti.get('born_processes')
            born_pdg_list = [[l['id'] for l in born.born_proc['legs']] \
            for born in born_procs ]
            loop_orders = {}
            for  born in born_procs:
                for coup, val in fks_common.find_orders(born.born_amp).items():
                    try:
                        loop_orders[coup] = max([loop_orders[coup], val])
                    except KeyError:
                        loop_orders[coup] = val        
            pdg_list = []        
            real_amp_list = []
            for born in born_procs:
                for amp in born.real_amps:
                    if not pdg_list.count(amp.pdgs):
                        pdg_list.append(amp.pdgs)
                        real_amp_list.append(amp)
                        
            #generating and store in tmp files all output corresponding to each real_amplitude
            real_out_list = []
            realmapin = []
            for i,real_amp in enumerate(real_amp_list):
                realmapin.append([i,real_amp])

            # start the pool instance with a signal instance to catch ctr+c
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            if fksmulti['ncores_for_proc_gen'] < 0: # use all cores
                pool = multiprocessing.Pool(maxtasksperchild=1)
            else:
                pool = multiprocessing.Pool(processes=fksmulti['ncores_for_proc_gen'],maxtasksperchild=1)
            signal.signal(signal.SIGINT, original_sigint_handler)

            logger.info('Generating real matrix elements...')
            try:
                # the very large timeout passed to get is to be able to catch
                # KeyboardInterrupts
                realmapout = pool.map_async(async_generate_real,realmapin).get(9999999)
            except KeyboardInterrupt:
                pool.terminate()
                raise KeyboardInterrupt 

            realmapfiles = []
            for realout in realmapout:
                realmapfiles.append(realout[0])

            logger.info('Generating born and virtual matrix elements...')
            #now loop over born and consume reals, generate virtuals
            bornmapin = []
            OLP=fksmulti['OLP']
            for i,born in enumerate(born_procs):
                bornmapin.append([i,born,born_pdg_list,loop_orders,pdg_list,loop_optimized,OLP,realmapfiles])

            try:
                bornmapout = pool.map_async(async_generate_born,bornmapin).get(9999999)
            except KeyboardInterrupt:
                pool.terminate()
                raise KeyboardInterrupt 

            #remove real temp files
            for realtmp in realmapout:
                os.remove(realtmp[0])
                
            logger.info('Collecting infos and finalizing matrix elements...')
            unique_me_list = []
            duplicate_me_lists = []
            for bornout in bornmapout:
                mefile = bornout[0]
                metag = bornout[1]
                has_loops = bornout[2]
                self['has_loops'] = self['has_loops'] or has_loops
                processes = bornout[3]
                self['processes'].extend(processes)
                unique = True
                for ime2,bornout2 in enumerate(unique_me_list):
                    mefile2 = bornout2[0]
                    metag2 = bornout2[1]
                    if metag==metag2:
                        duplicate_me_lists[ime2].append(mefile)
                        unique = False
                        break;
                if unique:
                    unique_me_list.append(bornout)
                    duplicate_me_lists.append([])
            
            memapin = []
            for i,bornout in enumerate(unique_me_list):
                mefile = bornout[0]
                memapin.append([i,mefile, duplicate_me_lists[i]])

            try:
                memapout = pool.map_async(async_finalize_matrix_elements,memapin).get(9999999)
            except KeyboardInterrupt:
                pool.terminate()
                raise KeyboardInterrupt 

            #remove born+virtual temp files
            for bornout in bornmapout:
                mefile = bornout[0]
                os.remove(mefile)

            pool.close()
            pool.join()

            #set final list of matrix elements (paths to temp files)
            matrix_elements = []
            for meout in memapout:
                matrix_elements.append(meout[0])
  
            self['matrix_elements']=matrix_elements
  
            #cache information needed for output which will not be available from
            #the matrix elements later
            initial_states = []
            for meout in memapout:
                me_initial_states = meout[1]
                for state in me_initial_states:
                    initial_states.append(state)
                              
            # remove doubles from the list
            checked = []
            for e in initial_states:
                if e not in checked:
                    checked.append(e)
            initial_states=checked

            self['initial_states']=initial_states
            
            helas_list = []
            for meout in memapout:
                helas_list.extend(meout[2])
            self['used_lorentz']=list(set(helas_list))        
            
            coupling_list = []
            for meout in memapout:
                coupling_list.extend([c for l in meout[3] for c in l])
            self['used_couplings'] = list(set(coupling_list)) 
            
            has_virtuals = False
            for meout in memapout:
                if meout[4]:
                    has_virtuals = True
                    break
            self['has_virtuals'] = has_virtuals
            
            configs_list = []
            for meout in realmapout:
                configs_list.append(meout[1])
            self['max_configs'] = max(configs_list)
            
            nparticles_list = []
            for meout in realmapout:
                nparticles_list.append(meout[2])
            self['max_particles'] = max(nparticles_list)        

        self['has_isr'] = fksmulti['has_isr']
        self['has_fsr'] = fksmulti['has_fsr']

        logger.info('... Done')

        for i, logg in enumerate(loggers_off):
            logg.setLevel(old_levels[i])
        
    def get_used_lorentz(self):
        """Return a list of (lorentz_name, conjugate, outgoing) with
        all lorentz structures used by this HelasMultiProcess."""

        if not self['used_lorentz']:
            helas_list = []
            for me in self.get('matrix_elements'):
                helas_list.extend(me.get_used_lorentz())
            self['used_lorentz'] = list(set(helas_list))

        return self['used_lorentz']


    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        if not self['used_couplings']:
            coupling_list = []
            for me in self.get('matrix_elements'):
                coupling_list.extend([c for l in me.get_used_couplings() for c in l])
            self['used_couplings'] = list(set(coupling_list))

        return self['used_couplings']


    def get_processes(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        if not self['processes']:
            process_list = []
            for me in self.get('matrix_elements'):
                process_list.extend(me.born_matrix_element.get('processes'))
            self['processes'] = process_list

        return self['processes']


    def get_max_configs(self):
        """Return max_configs"""

        if self['max_configs'] < 0:
            try:
                self['max_configs'] = max([me.get_num_configs() \
                                  for me in self['real_matrix_elements']])
            except ValueError:
                self['max_configs'] = max([me.born_matrix_element.get_num_configs() \
                                  for me in self['matrix_elements']])

        return self['max_configs']


    def get_max_particles(self):
        """Return max_paricles"""

        if self['max_particles'] < 0:
            self['max_particles'] = max([me.get_nexternal_ninitial()[0] \
                                for me in self['matrix_elements']])

        return self['max_particles']
    

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
            if fksproc.ncores_for_proc_gen:
                # new NLO (multicore) generation mode 
                for real_me, proc in itertools.izip(real_me_list,fksproc.real_amps):
                    fksreal_me = FKSHelasRealProcess(proc, real_me, **opts)
                    try:
                        other = self.real_processes[self.real_processes.index(fksreal_me)]
                        other.matrix_element.get('processes').extend(\
                                fksreal_me.matrix_element.get('processes') )
                    except ValueError:
                        if fksreal_me.matrix_element.get('processes') and \
                                fksreal_me.matrix_element.get('diagrams'):
                            self.real_processes.append(fksreal_me)
                            real_amps_new.append(proc)
            else:
                #old mode
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

    def get_nexternal_ninitial(self):
        """the nexternal_ninitial function references to the real emissions if they have been
        generated, otherwise to the born"""
        if self.real_processes:
            (nexternal, ninitial) = self.real_processes[0].matrix_element.get_nexternal_ninitial()
        else:
            (nexternal, ninitial) = self.born_matrix_element.get_nexternal_ninitial()
            nexternal += 1
        return (nexternal, ninitial)
    
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

            # real_me_list is a list in the old NLO generation mode;
            # in the new one it is a matrix element
            if type(real_me_list) == list and len(real_me_list) != len(real_amp_list):
                raise fks_common.FKSProcessError(
                        'not same number of amplitudes and matrix elements: %d, %d' % \
                                (len(real_amp_list), len(real_me_list)))
            if type(real_me_list) == list and real_me_list and real_amp_list:
                self.matrix_element = copy.deepcopy(real_me_list[real_amp_list.index(fksrealproc.amplitude)])
                self.matrix_element['processes'] = copy.deepcopy(self.matrix_element['processes'])

            elif type(real_me_list) == helas_objects.HelasMatrixElement: 
                #new NLO generation mode
                self.matrix_element = real_me_list

            else:

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



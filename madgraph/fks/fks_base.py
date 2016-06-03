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

"""Definitions of the objects needed for the implementation of MadFKS"""

import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.fks.fks_common as fks_common
import copy
import logging
import array
import madgraph.various.misc as misc
from madgraph import InvalidCmd

logger = logging.getLogger('madgraph.fks_base')


class NoBornException(Exception): pass

#===============================================================================
# FKS Process
#===============================================================================
class FKSMultiProcess(diagram_generation.MultiProcess): #test written
    """A multi process class that contains informations on the born processes 
    and the reals.
    """
    
    def default_setup(self):
        """Default values for all properties"""
        super(FKSMultiProcess, self).default_setup()
        self['real_amplitudes'] = diagram_generation.AmplitudeList()
        self['pdgs'] = []
        self['born_processes'] = FKSProcessList()
        if not 'OLP' in self.keys():
            self['OLP'] = 'MadLoop'
            self['ncores_for_proc_gen'] = 0
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcess, self).get_sorted_keys()
        keys += ['born_processes', 'real_amplitudes', 'real_pdgs', 'has_isr', 
                 'has_fsr', 'OLP', 'ncores_for_proc_gen']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'born_processes':
            if not isinstance(value, FKSProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for born_processes " % str(value)                             

        if name == 'real_amplitudes':
            if not isinstance(value, diagram_generation.AmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for real_amplitudes " % str(value)
                                                  
        if name == 'real_pdgs':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for real_amplitudes " % str(value)
        
        if name == 'OLP':
            if not isinstance(value,str):
                raise self.PhysicsObjectError, \
                    "%s is not a valid string for OLP " % str(value)

        if name == 'ncores_for_proc_gen':
            if not isinstance(value,int):
                raise self.PhysicsObjectError, \
                    "%s is not a valid value for ncores_for_proc_gen " % str(value)
                                                     
        return super(FKSMultiProcess,self).filter(name, value)
    
    def __init__(self, procdef=None, options={}):
        """Initializes the original multiprocess, then generates the amps for the 
        borns, then generate the born processes and the reals.
        Real amplitudes are stored in real_amplitudes according on the pdgs of their
        legs (stored in pdgs, so that they need to be generated only once and then reicycled
        """

        #swhich the other loggers off
        loggers_off = [logging.getLogger('madgraph.diagram_generation'), 
                       logging.getLogger('madgraph.loop_diagram_generation')]
        old_levels = [logg.level for logg in loggers_off]
        for logg in loggers_off:
            logg.setLevel(logging.WARNING)
        
        # OLP option
        olp='MadLoop'
        if 'OLP' in options.keys():
            olp = options['OLP']
            del options['OLP']

        ncores_for_proc_gen = 0
        # ncores_for_proc_gen has the following meaning
        #   0 : do things the old way
        #   > 0 use ncores_for_proc_gen
        #   -1 : use all cores
        if 'ncores_for_proc_gen' in options.keys():
            ncores_for_proc_gen = options['ncores_for_proc_gen']
            del options['ncores_for_proc_gen']

        try:
            # Now generating the borns for the first time.
            super(FKSMultiProcess, self).__init__(procdef, **options)

        except diagram_generation.NoDiagramException as error:
            # If no born, then this process most likely does not have any.
            raise NoBornException, "Born diagrams could not be generated for the "+\
               self['process_definitions'][0].nice_string().replace('Process',\
               'process')+". Notice that aMC@NLO does not handle loop-induced"+\
               " processes yet, but you can still use MadLoop if you want to "+\
               "only generate them."+\
               " For this, use the 'virt=' mode, without multiparticle labels."

        self['OLP'] = olp
        self['ncores_for_proc_gen'] = ncores_for_proc_gen
            
        #check process definition(s):
        # a process such as g g > g g will lead to real emissions 
        #   (e.g: u g > u g g ) which will miss some corresponding born,
        #   leading to non finite results
        perturbation = []
        for procdef in self['process_definitions']:
            soft_particles = []
            # do not warn for decay processes
            if [ i['state'] for i in procdef['legs']].count(False) == 1:
                continue
            for pert in procdef['perturbation_couplings']:
                if pert not in perturbation:
                    perturbation.append(pert)
                soft_particles.extend(\
                        fks_common.find_pert_particles_interactions(\
                    procdef['model'], pert)['soft_particles'])
                soft_particles_string = ', '.join( \
                    [procdef['model'].get('particle_dict')[id][\
                    {True:'name', False:'antiname'}[id >0] ] \
                    for id in sorted(soft_particles, reverse=True)])
            for leg in procdef['legs']:
                if any([id in soft_particles for id in leg['ids']]) \
                        and sorted(leg['ids']) != soft_particles:
                    logger.warning(('%s can have real emission processes ' + \
            'which are not finite.\nTo avoid this, please use multiparticles ' + \
            'when generating the process and be sure to include all the following ' + \
            'particles in the multiparticle definition:\n %s' ) \
               % (procdef.nice_string(), soft_particles_string) )
                    break
        for procdef in self['process_definitions']:
            procdef.set('orders', diagram_generation.MultiProcess.find_optimal_process_orders(procdef))

        amps = self.get('amplitudes')
        for i, amp in enumerate(amps):
            logger.info("Generating FKS-subtracted matrix elements for born process%s (%d / %d)" \
                % (amp['process'].nice_string(print_weighted=False).replace(\
                                                                 'Process', ''),
                 i + 1, len(amps)))

            born = FKSProcess(amp, ncores_for_proc_gen = self['ncores_for_proc_gen'])
            self['born_processes'].append(born)
            born.generate_reals(self['pdgs'], self['real_amplitudes'])

        if not self['ncores_for_proc_gen']:
            # old generation mode 

            born_pdg_list = [[l['id'] for l in born.born_proc['legs']] \
                    for born in self['born_processes'] ]

            for born in self['born_processes']:
                for real in born.real_amps:
                    real.find_fks_j_from_i(born_pdg_list)
            if amps:
                if self['process_definitions'][0].get('NLO_mode') == 'all':
                    self.generate_virtuals()
                
                elif not self['process_definitions'][0].get('NLO_mode') in ['all', 'real','LOonly']:
                    raise fks_common.FKSProcessError(\
                       "Not a valid NLO_mode for a FKSMultiProcess: %s" % \
                       self['process_definitions'][0].get('NLO_mode'))

                # now get the total number of diagrams
                n_diag_born = sum([len(amp.get('diagrams')) 
                         for amp in self.get_born_amplitudes()])
                n_diag_real = sum([len(amp.get('diagrams')) 
                         for amp in self.get_real_amplitudes()])
                n_diag_virt = sum([len(amp.get('loop_diagrams')) 
                         for amp in self.get_virt_amplitudes()])

                if n_diag_virt == 0 and n_diag_real ==0 and \
                        not self['process_definitions'][0].get('NLO_mode') == 'LOonly':
                    raise fks_common.FKSProcessError(
                            'This process does not have any correction up to NLO in %s'\
                            %','.join(perturbation))

                logger.info(('Generated %d subprocesses with %d real emission diagrams, ' + \
                            '%d born diagrams and %d virtual diagrams') % \
                                    (len(self['born_processes']), n_diag_real, n_diag_born, n_diag_virt))

            for i, logg in enumerate(loggers_off):
                logg.setLevel(old_levels[i])

        self['has_isr'] = any([proc.isr for proc in self['born_processes']])
        self['has_fsr'] = any([proc.fsr for proc in self['born_processes']])

    def add(self, other):
        """combines self and other, extending the lists of born/real amplitudes"""
        self['process_definitions'].extend(other['process_definitions'])
        self['amplitudes'].extend(other['amplitudes'])
        self['born_processes'].extend(other['born_processes'])
        self['real_amplitudes'].extend(other['real_amplitudes'])
        self['pdgs'].extend(other['pdgs'])
        self['has_isr'] = self['has_isr'] or other['has_isr']
        self['has_fsr'] = self['has_fsr'] or other['has_fsr']
        self['OLP'] = other['OLP']
        self['ncores_for_proc_gen'] = other['ncores_for_proc_gen']

    def get_born_amplitudes(self):
        """return an amplitudelist with the born amplitudes"""
        return diagram_generation.AmplitudeList([born.born_amp \
                for born in self['born_processes']])

    def get_virt_amplitudes(self):
        """return an amplitudelist with the virt amplitudes"""
        return diagram_generation.AmplitudeList([born.virt_amp \
                for born in self['born_processes'] if born.virt_amp])

    def get_real_amplitudes(self):
        """return an amplitudelist with the real amplitudes"""
        return self.get('real_amplitudes')


    def generate_virtuals(self):
        """For each process among the born_processes, creates the corresponding
        virtual amplitude"""
        
        # If not using MadLoop, then the LH order file generation and processing
        # will be entirely done during the output, so nothing must be done at
        # this stage yet.
        if self['OLP']!='MadLoop':
            logger.info("The loop matrix elements will be generated by "+\
                                     '%s at the output stage only.'%self['OLP'])
            return

        # determine the orders to be used to generate the loop
        loop_orders = {}
        for  born in self['born_processes']:
            for coup, val in fks_common.find_orders(born.born_amp).items():
                try:
                    loop_orders[coup] = max([loop_orders[coup], val])
                except KeyError:
                    loop_orders[coup] = val

        for i, born in enumerate(self['born_processes']):
            logger.info('Generating virtual matrix elements using MadLoop:')
            myproc = copy.copy(born.born_proc)
            # take the orders that are actually used bu the matrix element
            myproc['orders'] = loop_orders
            myproc['legs'] = fks_common.to_legs(copy.copy(myproc['legs']))
            logger.info('Generating virtual matrix element with MadLoop for process%s (%d / %d)' \
                    % (myproc.nice_string(print_weighted = False).replace(\
                                                             'Process', ''),
                        i + 1, len(self['born_processes'])))
            myamp = loop_diagram_generation.LoopAmplitude(myproc)
            if myamp.get('diagrams'):
                born.virt_amp = myamp


class FKSRealProcess(object): 
    """Contains information about a real process:
    -- fks_infos (list containing the possible fks configs for a given process
    -- amplitude 
    -- is_to_integrate
    -- leg permutation<<REMOVED!.
    """
    
    def __init__(self, born_proc, leglist, ij, ijglu,
                 perturbed_orders = ['QCD']): #test written
        """Initializes the real process based on born_proc and leglist.
        Stores the fks informations into the list of dictionaries fks_infos
        """      
        self.fks_infos = []
        for leg in leglist:
            if leg.get('fks') == 'i':
                i_fks = leg.get('number')
                # i is a gluon or a photon
                need_color_links = leg.get('massless') \
                        and leg.get('spin') == 3 \
                        and leg.get('self_antipart')
            if leg.get('fks') == 'j':
                j_fks = leg.get('number')
        self.fks_infos.append({'i' : i_fks, 
                               'j' : j_fks, 
                               'ij' : ij, 
                               'ij_glu': ijglu, 
                               'need_color_links' : need_color_links})

        self.process = copy.copy(born_proc)
        orders = copy.copy(born_proc.get('orders'))
        # compute the weighted order if not present
        if not 'WEIGHTED' in orders:
            orders['WEIGHTED'] = sum([v * born_proc.get('model').get('order_hierarchy')[o] \
                        for o, v in orders.items()])

        for order in perturbed_orders:
            try:
                orders[order] +=1
            except KeyError:
                pass
            orders['WEIGHTED'] += born_proc.get('model').get('order_hierarchy')[order]

        self.process.set('orders', orders)
        legs = [(leg.get('id'), leg) for leg in leglist]
        self.pdgs = array.array('i',[s[0] for s in legs])
        if 'QCD' in perturbed_orders:
            self.colors = [leg['color'] for leg in leglist]
            # in QCD, charges are irrelevant !
            self.charges = [0. for leg in leglist]
            self.perturbation = 'QCD'
        else:
            self.colors = [leg['color'] for leg in leglist]
            self.charges = [leg['charge'] for leg in leglist]
            self.perturbation = 'QED'
        self.process.set('legs', MG.LegList(leglist))
        self.process.set('legs_with_decays', MG.LegList())
        self.amplitude = diagram_generation.Amplitude()
        self.is_to_integrate = True
        self.is_nbody_only = False
        self.fks_j_from_i = {}


    def generate_real_amplitude(self):
        """generates the real emission amplitude starting from self.process"""
        self.amplitude = diagram_generation.Amplitude(self.process)
        return self.amplitude


    def find_fks_j_from_i(self, born_pdg_list): #test written
        """Returns a dictionary with the entries i : [j_from_i], if the born pdgs are in 
        born_pdg_list"""
        fks_j_from_i = {}
        dict = {}
        for i in self.process.get('legs'):
            fks_j_from_i[i.get('number')] = []
            if i.get('state'):
                for j in [l for l in self.process.get('legs') if \
                        l.get('number') != i.get('number')]:
                    ijlist = fks_common.combine_ij(i, j, self.process.get('model'), dict,\
                                                   pert=self.perturbation)
                    for ij in ijlist:
                        born_leglist = fks_common.to_fks_legs(
                                      copy.deepcopy(self.process.get('legs')), 
                                      self.process.get('model'))
                        born_leglist.remove(i)
                        born_leglist.remove(j)
                        born_leglist.insert(ij.get('number') - 1, ij)
                        born_leglist.sort(pert = self.perturbation)
                        if [l['id'] for l in born_leglist] in born_pdg_list:
                            fks_j_from_i[i.get('number')].append(\
                                                    j.get('number'))                                

        self.fks_j_from_i = fks_j_from_i
        return fks_j_from_i

        
    def get_leg_i(self): #test written
        """Returns leg corresponding to i fks.
        An error is raised if the fks_infos list has more than one entry"""
        if len(self.fks_infos) > 1:
            raise fks_common.FKSProcessError(\
                    'get_leg_i should only be called before combining processes')
        return self.process.get('legs')[self.fks_infos[0]['i'] - 1]

    def get_leg_j(self): #test written
        """Returns leg corresponding to j fks.
        An error is raised if the fks_infos list has more than one entry"""
        if len(self.fks_infos) > 1:
            raise fks_common.FKSProcessError(\
                    'get_leg_j should only be called before combining processes')
        return self.process.get('legs')[self.fks_infos[0]['j'] - 1]


class FKSProcessList(MG.PhysicsObjectList):
    """Class to handle lists of FKSProcesses."""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcess for the list."""
        return isinstance(obj, FKSProcess)

            
class FKSProcess(object):
    """The class for a FKS process. Starts from the born process and finds
    all the possible splittings."""  
    
    def __init__(self, start_proc = None, remove_reals = True, ncores_for_proc_gen=0):
        """initialization: starts either from an amplitude or a process,
        then init the needed variables.
        remove_borns tells if the borns not needed for integration will be removed
        from the born list (mainly used for testing)
        ncores_for_proc_gen has the following meaning
           0 : do things the old way
           > 0 use ncores_for_proc_gen
           -1 : use all cores
        """
                
        self.splittings = {}
        self.reals = []
        self.fks_dirs = []
        self.leglist = []
        self.myorders = {}
        self.pdg_codes = []
        self.colors = [] # color
        self.charges = [] # charge
        self.nlegs = 0
        self.fks_ipos = []
        self.fks_j_from_i = {}
        self.real_amps = []
        self.remove_reals = remove_reals
        self.nincoming = 0
        self.virt_amp = None
        self.perturbation = 'QCD'
        self.ncores_for_proc_gen = ncores_for_proc_gen

        if not remove_reals in [True, False]:
            raise fks_common.FKSProcessError(\
                    'Not valid type for remove_reals in FKSProcess')

        if start_proc:
            if isinstance(start_proc, MG.Process):
                pertur = start_proc['perturbation_couplings']
                if pertur:
                    self.perturbation = sorted(pertur)[0]
                self.born_proc = fks_common.sort_proc(start_proc,pert = self.perturbation)
                # filter in Amplitude will legs sorted in bornproc
                bornproc = copy.copy(self.born_proc) # deepcopy might let T -> array
                assert bornproc==self.born_proc
                self.born_amp = diagram_generation.Amplitude(bornproc)
            elif isinstance(start_proc, diagram_generation.Amplitude):
                pertur = start_proc.get('process')['perturbation_couplings']
                if pertur:
                    self.perturbation = sorted(pertur)[0]
                self.born_proc = fks_common.sort_proc(start_proc.get('process'),\
                                                      pert = self.perturbation)
                # filter in Amplitude will legs sorted in bornproc
                bornproc = copy.copy(self.born_proc)
                assert bornproc == self.born_proc
                self.born_amp = diagram_generation.Amplitude(bornproc)
            else:
                raise fks_common.FKSProcessError(\
                    'Not valid start_proc in FKSProcess')
            self.born_proc.set('legs_with_decays', MG.LegList())

            self.leglist = fks_common.to_fks_legs(
                                    self.born_proc['legs'], self.born_proc['model'])
            self.nlegs = len(self.leglist)
            self.pdg_codes = [leg.get('id') for leg in self.leglist]
            if self.perturbation == 'QCD':
                self.colors = [leg.get('color') for leg in self.leglist]
                # in QCD, charge is irrelevant but impact the combine process !
                self.charges = [0. for leg in self.leglist]
                color = 'color'
                zero = 1
            elif self.perturbation == 'QED':
                self.colors = [leg.get('color') for leg in self.leglist]
                self.charges = [leg.get('charge') for leg in self.leglist]
                color = 'charge'
                zero = 0.
            # special treatment of photon is needed !
            self.isr = set([leg.get(color) for leg in self.leglist if not leg.get('state')]) != set([zero])
            self.fsr = set([leg.get(color) for leg in self.leglist if leg.get('state')]) != set([zero])
            for leg in self.leglist:
                if not leg['state']:
                    self.nincoming += 1
            self.orders = self.born_amp['process']['orders']
            # this is for cases in which the user specifies e.g. QED=0
            if sum(self.orders.values()) == 0:
                self.orders = fks_common.find_orders(self.born_amp)
                
            self.ndirs = 0
            # generate reals, when the mode is not LOonly
            # when is LOonly it is supposed to be a 'fake' NLO process
            # e.g. to be used in merged sampels at high multiplicities
            if self.born_proc['NLO_mode'] != 'LOonly':
                for order in self.born_proc.get('perturbation_couplings'):
                    self.find_reals(order)


    def generate_real_amplitudes(self, pdg_list, real_amp_list):
        """generates the real amplitudes for all the real emission processes, using pdgs and real_amps
        to avoid multiple generation of the same amplitude"""

        for amp in self.real_amps:
            try:
                amp.amplitude = real_amp_list[pdg_list.index(amp.pdgs)]
            except ValueError:
                pdg_list.append(amp.pdgs)
                real_amp_list.append(amp.generate_real_amplitude())


    def combine_real_amplitudes(self):
        """combines real emission processes if the pdgs are the same, combining the lists 
        of fks_infos"""
        pdgs = []
        real_amps = []
        old_real_amps = copy.copy(self.real_amps)
        for amp in old_real_amps:
            try:
                real_amps[pdgs.index(amp.pdgs)].fks_infos.extend(amp.fks_infos)
            except ValueError:
                real_amps.append(amp)
                pdgs.append(amp.pdgs)

        self.real_amps = real_amps


    def generate_reals(self, pdg_list, real_amp_list, combine=True): #test written
        """For all the possible splittings, creates an FKSRealProcess.
        It removes double counted configorations from the ones to integrates and
        sets the one which includes the bosn (is_nbody_only).
        if combine is true, FKS_real_processes having the same pdgs (i.e. real amplitude)
        are combined together
        """

        born_proc = copy.copy(self.born_proc)
        born_proc['orders'] = self.orders
        for i, list in enumerate(self.reals):
            # i is a gluon or photon,i.e. g(a) > ffx or gg
            if self.leglist[i]['massless'] and self.leglist[i]['spin'] == 3:
                ijglu = i + 1
            else:
                ijglu = 0
            for l in list:
                ij = self.leglist[i].get('number')
                self.real_amps.append(FKSRealProcess( \
                        born_proc, l, ij, ijglu,\
                        perturbed_orders = [self.perturbation]))
        self.find_reals_to_integrate()
        if combine:
            self.combine_real_amplitudes()
        if not self.ncores_for_proc_gen:
            self.generate_real_amplitudes(pdg_list, real_amp_list)
            self.link_born_reals()


    def link_born_reals(self):
        """create the rb_links in the real matrix element to find 
        which configuration in the real correspond to which in the born
        """
        for real in self.real_amps:
            for info in real.fks_infos:
                info['rb_links'] = fks_common.link_rb_configs(\
                        self.born_amp, real.amplitude,
                        info['i'], info['j'], info['ij'])


    def find_reals(self, pert_order):
        """finds the FKS real configurations for a given process"""
        if range(len(self.leglist)) != [l['number']-1 for l in self.leglist]:
            raise fks_common.FKSProcessError('Disordered numbers of leglist')

        if [ i['state'] for i in self.leglist].count(False) == 1:
            decay_process=True
        else:
            decay_process=False

        for i in self.leglist:
            i_i = i['number'] - 1
            self.reals.append([])
            if decay_process and not i['state']:
                self.splittings[i_i]=[]
            else:
                self.splittings[i_i] = fks_common.find_splittings(i, self.born_proc['model'], {}, pert_order)
            for split in self.splittings[i_i]:
                self.reals[i_i].append(
                            fks_common.insert_legs(self.leglist, i, split,pert=pert_order))
                


    def find_reals_to_integrate(self): #test written
        """Finds double countings in the real emission configurations, sets the 
        is_to_integrate variable and if "self.remove_reals" is True removes the 
        not needed ones from the born list.
        """
        #find the initial number of real configurations
        ninit = len(self.real_amps)
        remove = self.remove_reals
        
        for m in range(ninit):
            for n in range(m + 1, ninit):
                real_m = self.real_amps[m]
                real_n = self.real_amps[n]
                if len(real_m.fks_infos) > 1 or len(real_m.fks_infos) > 1:
                    raise fks_common.FKSProcessError(\
                    'find_reals_to_integrate should only be called before combining processes')

                i_m = real_m.fks_infos[0]['i']
                j_m = real_m.fks_infos[0]['j']
                i_n = real_n.fks_infos[0]['i']
                j_n = real_n.fks_infos[0]['j']
                if j_m > self.nincoming and j_n > self.nincoming:
                    if (real_m.get_leg_i()['id'] == real_n.get_leg_i()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_j()['id']) \
                        or \
                       (real_m.get_leg_i()['id'] == real_n.get_leg_j()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_i()['id']):
                        if i_m > i_n:
                            print real_m.get_leg_i()['id'], real_m.get_leg_j()['id']
                            if real_m.get_leg_i()['id'] == -real_m.get_leg_j()['id']:
                                self.real_amps[m].is_to_integrate = False
                            else:
                                self.real_amps[n].is_to_integrate = False
                        elif i_m == i_n and j_m > j_n:
                            print real_m.get_leg_i()['id'], real_m.get_leg_j()['id']
                            if real_m.get_leg_i()['id'] == -real_m.get_leg_j()['id']:
                                self.real_amps[m].is_to_integrate = False
                            else:
                                self.real_amps[n].is_to_integrate = False
                        # in case of g(a) > ffx splitting, keep the lowest ij
                        elif i_m == i_n and j_m == j_n and \
                          not real_m.get_leg_j()['self_antipart'] and \
                          not real_m.get_leg_i()['self_antipart']:
                            if real_m.fks_infos[0]['ij'] > real_n.fks_infos[0]['ij']:
                                real_m.is_to_integrate = False
                            else:
                                real_n.is_to_integrate = False
                        else:
                            if real_m.get_leg_i()['id'] == -real_m.get_leg_j()['id']:
                                self.real_amps[n].is_to_integrate = False
                            else:
                                self.real_amps[m].is_to_integrate = False
                         # self.real_amps[m].is_to_integrate = False
                elif j_m <= self.nincoming and j_n == j_m:
                    if real_m.get_leg_i()['id'] == real_n.get_leg_i()['id'] and \
                       real_m.get_leg_j()['id'] == real_n.get_leg_j()['id']:
                        if i_m > i_n:
                            self.real_amps[n].is_to_integrate = False
                        else:
                            self.real_amps[m].is_to_integrate = False
        if remove:
            newreal_amps = []
            for real in self.real_amps:
                if real.is_to_integrate:
                    newreal_amps.append(real)
            self.real_amps = newreal_amps

    

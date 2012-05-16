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
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_real as fks_real
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_born')


#===============================================================================
# FKS Process
#===============================================================================
class FKSMultiProcessFromBorn(diagram_generation.MultiProcess): #test written
    """A multi process class that contains informations on the born processes 
    and the reals.
    """
    
    def default_setup(self):
        """Default values for all properties"""
        super(FKSMultiProcessFromBorn, self).default_setup()
        self['born_processes'] = FKSProcessFromBornList()
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcessFromBorn, self).get_sorted_keys()
        keys += ['born_processes']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'born_processes':
            if not isinstance(value, FKSProcessFromBornList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for born_processes " % str(value)                             
        return super(FKSMultiProcessFromBorn,self).filter(name, value)
    
    def __init__(self,  *arguments):
        """Initializes the original multiprocess, then generates the amps for the 
        borns, then geneare the born processes and the reals.
        """

        #swhich the other loggers off
        loggers_off = [logging.getLogger('madgraph.diagram_generation'), 
                       logging.getLogger('madgraph.loop_diagram_generation')]
        old_levels = [logg.getEffectiveLevel() for logg in loggers_off]
        for logg in loggers_off:
            logg.setLevel(logging.WARNING)

        super(FKSMultiProcessFromBorn, self).__init__(*arguments)   
        amps = self.get('amplitudes')
        for amp in amps:
            born = FKSProcessFromBorn(amp)
            self['born_processes'].append(born)
            born.generate_reals()

        if self['process_definitions'][0].get('NLO_mode') == 'all':
            logger.info('Generating virtual matrix elements:')
            self.generate_virtuals()
        
        elif not self['process_definitions'][0].get('NLO_mode') in ['all', 'real']:
            raise fks_common.FKSProcessError(), \
               "Not a valid NLO_mode for a FKSMultiProcess: %s" % \
               self['process_definitions'][0].get('NLO_mode')

        # now get the total number of diagrams
        n_diag_born = sum([len(amp.get('diagrams')) 
                 for amp in self.get_born_amplitudes()])
        n_diag_real = sum([len(amp.get('diagrams')) 
                 for amp in self.get_real_amplitudes()])
        n_diag_virt = sum([len(amp.get('diagrams')) 
                 for amp in self.get_virt_amplitudes()])

        logger.info(('Generated %d subprocesses with %d real emission diagrams, ' + \
                    '%d born diagrams and %d virtual diagrams') % \
                            (len(self['born_processes']), n_diag_real, n_diag_born, n_diag_virt))

        for i, logg in enumerate(loggers_off):
            logg.setLevel(old_levels[i])


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
        return diagram_generation.AmplitudeList([real.amplitude \
                           for born in self['born_processes'] \
                           for real in born.real_amps])


    def generate_virtuals(self):
        """For each process among the born_processes, creates the corresponding
        virtual amplitude"""

        for born in self['born_processes']:
                myproc = copy.copy(born.born_proc)
                if 'WEIGHTED' in myproc['orders'].keys():
                    del myproc['orders']['WEIGHTED']
                myproc['legs'] = fks_common.to_legs(copy.copy(myproc['legs']))
                logger.info('Generating virtual matrix element for process%s' \
                        % myproc.nice_string().replace('Process', ''))
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
                need_color_links = leg.get('massless') \
                        and leg.get('spin') == 3 \
                        and leg.get('color') == 8
            if leg.get('fks') == 'j':
                j_fks = leg.get('number')
        self.fks_infos.append({'i' : i_fks, 
                               'j' : j_fks, 
                               'ij' : ij, 
                               'ij_glu': ijglu, 
                               'need_color_links' : need_color_links})

        self.process = copy.copy(born_proc)
        orders = copy.copy(born_proc.get('orders'))
        for order in perturbed_orders:
            orders[order] +=1
            if order == 'QCD':
                orders['WEIGHTED'] +=1
            else: 
                orders['WEIGHTED'] +=2

        self.process.set('orders', orders)

        legs = [(leg.get('id'), leg) for leg in leglist]
        self.pdgs = array.array('i',[s[0] for s in legs]) 
        self.colors = [leg['color'] for leg in leglist]
        self.process.set('legs', MG.LegList(leglist))
        self.amplitude = diagram_generation.Amplitude()
        self.is_to_integrate = True
        self.is_nbody_only = False


    def generate_real_amplitude(self):
        """generates the real emission amplitude starting from self.process"""
        self.amplitude = diagram_generation.Amplitude(self.process)


    def find_fks_j_from_i(self): #test written
        """Returns a dictionary with the entries i : [j_from_i]."""
        fks_j_from_i = {}
        dict = {}
        for i in self.process.get('legs'):
            fks_j_from_i[i.get('number')] = []
            if i.get('state'):
                for j in self.process.get('legs'):
                    if j.get('number') != i.get('number') :
                        ijlist = fks_common.combine_ij(i, j, self.process.get('model'), dict)
                        for ij in ijlist:
                            born = fks_real.FKSBornProcess(self.process, i, j, ij)
                            if born.amplitude.get('diagrams'):
                                fks_j_from_i[i.get('number')].append(\
                                                        j.get('number'))                                
        return fks_j_from_i

        
    def get_leg_i(self): #test written
        """Returns leg corresponding to i fks.
        An error is raised if the fks_infos list has more than one entry"""
        if len(self.fks_infos) > 1:
            raise fks_common.FKSProcessError(), \
                    'get_leg_i should only be called before combining processes'
        return self.process.get('legs')[self.fks_infos[0]['i'] - 1]

    def get_leg_j(self): #test written
        """Returns leg corresponding to j fks.
        An error is raised if the fks_infos list has more than one entry"""
        if len(self.fks_infos) > 1:
            raise fks_common.FKSProcessError(), \
                    'get_leg_j should only be called before combining processes'
        return self.process.get('legs')[self.fks_infos[0]['j'] - 1]


class FKSProcessFromBornList(MG.PhysicsObjectList):
    """Class to handle lists of FKSProcessesFromBorn."""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromBorn for the list."""
        return isinstance(obj, FKSProcessFromBorn)

            
class FKSProcessFromBorn(object):
    """The class for a FKS process. Starts from the born process and finds
    all the possible splittings."""  
    
    def __init__(self, start_proc = None, remove_reals = True):
        """initialization: starts either from an amplitude or a process,
        then init the needed variables.
        remove_borns tells if the borns not needed for integration will be removed
        from the born list (mainly used for testing)"""
                
        self.splittings = {}
        self.reals = []
        self.fks_dirs = []
        self.leglist = []
        self.myorders = {}
        self.pdg_codes = []
        self.colors = []
        self.nlegs = 0
        self.fks_ipos = []
        self.fks_j_from_i = {}
        self.color_links = []
        self.real_amps = []
        self.remove_reals = remove_reals
        self.nincoming = 0
        self.virt_amp = None

        if not remove_reals in [True, False]:
            raise fks_common.FKSProcessError(), \
                    'Not valid type for remove_reals in FKSProcessFromBorn'
        
        if start_proc:
            if isinstance(start_proc, MG.Process):
                self.born_proc = fks_common.sort_proc(start_proc) 
                self.born_amp = diagram_generation.Amplitude(self.born_proc)
            elif isinstance(start_proc, diagram_generation.Amplitude):
                self.born_proc = fks_common.sort_proc(start_proc.get('process'))
                self.born_amp = diagram_generation.Amplitude(self.born_proc)
            else:
                raise fks_common.FKSProcessError(), \
                    'Not valid start_proc in FKSProcessFromBorn'

            logger.info("Generating FKS-subtracted matrix elements for born process%s" \
                % self.born_proc.nice_string().replace('Process', '')) 

#            self.model = self.born_proc['model']
            self.leglist = fks_common.to_fks_legs(
                                    self.born_proc['legs'], self.born_proc['model'])
            self.nlegs = len(self.leglist)
            self.pdg_codes = [leg.get('id') for leg in self.leglist]
            self.colors = [leg.get('color') for leg in self.leglist]
            for leg in self.leglist:
                if not leg['state']:
                    self.nincoming += 1
            # find the correct qcd/qed orders from born_amp
            orders = fks_common.find_orders(self.born_amp)
            self.born_proc['orders'] = orders
                
            self.ndirs = 0
            for order in self.born_proc.get('perturbation_couplings'):
                self.find_reals(order)
            self.find_color_links()



    def find_color_links(self): #test written
        """Finds all the possible color links between two legs of the born.
        Uses the find_color_links function in fks_common.
        """
        self.color_links = fks_common.find_color_links(self.leglist)
        return self.color_links


    def generate_real_amplitudes(self):
        """generates the real amplitudes for all the real emission processes"""
        for amp in self.real_amps:
            amp.generate_real_amplitude()


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


        
    def generate_reals(self, combine=True): #test written
        """For all the possible splittings, creates an FKSRealProcess.
        It removes double counted configorations from the ones to integrates and
        sets the one which includes the bosn (is_nbody_only).
        if combine is true, FKS_real_processes having the same pdgs (i.e. real amplitude)
        are combined together
        """

        for i, list in enumerate(self.reals):
            if self.leglist[i]['massless'] and self.leglist[i]['spin'] == 3:
                ijglu = i + 1
            else:
                ijglu = 0
            for l in list:
                ij = self.leglist[i].get('number')
                self.real_amps.append(FKSRealProcess( \
                        self.born_proc, l, ij, ijglu))
        self.find_reals_to_integrate()
        if combine:
            self.combine_real_amplitudes()
        self.generate_real_amplitudes()
#        self.find_real_nbodyonly()


    def find_reals(self, pert_order):
        """finds the FKS real configurations for a given process"""
        for i in self.leglist:
            i_i = i['number'] - 1
            self.reals.append([])
            self.splittings[i_i] = fks_common.find_splittings(i, self.born_proc['model'], {}, pert_order)
            for split in self.splittings[i_i]:
                self.reals[i_i].append(
                            fks_common.insert_legs(self.leglist, i, split))


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
                    raise fks_common.FKSProcessError(), \
                    'find_reals_to_integrate should only be called before combining processes'

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
                            self.real_amps[n].is_to_integrate = False
                        elif i_m == i_n and j_m > j_n:
                            self.real_amps[n].is_to_integrate = False
                        else:
                            self.real_amps[m].is_to_integrate = False
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

    

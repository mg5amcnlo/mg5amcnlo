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

from __future__ import absolute_import
from __future__ import print_function
import madgraph
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
from six.moves import range

logger = logging.getLogger('madgraph.fks_base')

if madgraph.ordering:
    set = misc.OrderedSet

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

        if not 'OLP' in list(self.keys()):
            self['OLP'] = 'MadLoop'
            self['ncores_for_proc_gen'] = 0

        self['loop_filter'] = None
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcess, self).get_sorted_keys()
        keys += ['born_processes', 'real_amplitudes', 'real_pdgs', 'has_isr', 
                 'has_fsr', 'spltting_types', 'OLP', 'ncores_for_proc_gen', 
                 'loop_filter']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'born_processes':
            if not isinstance(value, FKSProcessList):
                raise self.PhysicsObjectError("%s is not a valid list for born_processes " % str(value))                             

        if name == 'real_amplitudes':
            if not isinstance(value, diagram_generation.AmplitudeList):
                raise self.PhysicsObjectError("%s is not a valid list for real_amplitudes " % str(value))
                                                  
        if name == 'real_pdgs':
            if not isinstance(value, list):
                raise self.PhysicsObjectError("%s is not a valid list for real_amplitudes " % str(value))
        
        if name == 'OLP':
            if not isinstance(value,str):
                raise self.PhysicsObjectError("%s is not a valid string for OLP " % str(value))

        if name == 'ncores_for_proc_gen':
            if not isinstance(value,int):
                raise self.PhysicsObjectError("%s is not a valid value for ncores_for_proc_gen " % str(value))
                                                     
        return super(FKSMultiProcess,self).filter(name, value)


    def check_ij_confs(self):
        """check that there is no duplicate FKS ij configuration"""
        ijconfs_dict = {}
        for born in self['born_processes']:
            # the copy.copy is needed as duplicate configurations will be removed on the fly
            for real in copy.copy(born.real_amps):
                pdgs = ' '.join([ '%d' % pdg for pdg in real.pdgs])
                for info in copy.copy(real.fks_infos):
                    ij = [info['i'], info['j']]
                    try:
                        if ij in ijconfs_dict[pdgs]:
                            logger.debug('Duplicate FKS configuration found for %s : ij = %s' %
                                    (real.process.nice_string(), str(ij)))
                            #remove the configuration
                            born.real_amps[born.real_amps.index(real)].fks_infos.remove(info)
                        else:
                            ijconfs_dict[pdgs].append(ij)
                    except KeyError:
                        ijconfs_dict[pdgs] = [ij]
                # check if any FKS configuration remains for the real emission, otherwise
                # remove it
                if not born.real_amps[born.real_amps.index(real)].fks_infos:
                    logger.debug('Removing real %s from born %s' % \
                            (real.process.nice_string(), born.born_amp['process'].nice_string()))
                    born.real_amps.remove(real)

    
    def __init__(self, procdef=None, options={}):
        """Initializes the original multiprocess, then generates the amps for the 
        borns, then generate the born processes and the reals.
        Real amplitudes are stored in real_amplitudes according on the pdgs of their
        legs (stored in pdgs, so that they need to be generated only once and then reicycled
        """


        if 'nlo_mixed_expansion' in options:
            self['nlo_mixed_expansion'] = options['nlo_mixed_expansion']
            del options['nlo_mixed_expansion']
        else:
            self['nlo_mixed_expansion'] = True


        #swhich the other loggers off
        loggers_off = [logging.getLogger('madgraph.diagram_generation'), 
                       logging.getLogger('madgraph.loop_diagram_generation')]
        old_levels = [logg.level for logg in loggers_off]
        for logg in loggers_off:
            logg.setLevel(logging.WARNING)
        
        self['real_amplitudes'] = diagram_generation.AmplitudeList()
        self['pdgs'] = []

        # OLP option
        olp='MadLoop'
        if 'OLP' in list(options.keys()):
            olp = options['OLP']
            del options['OLP']

        self['init_lep_split']=False
        if 'init_lep_split' in list(options.keys()):
            self['init_lep_split']=options['init_lep_split']
            del options['init_lep_split']

        ncores_for_proc_gen = 0
        # ncores_for_proc_gen has the following meaning
        #   0 : do things the old way
        #   > 0 use ncores_for_proc_gen
        #   -1 : use all cores
        if 'ncores_for_proc_gen' in list(options.keys()):
            ncores_for_proc_gen = options['ncores_for_proc_gen']
            del options['ncores_for_proc_gen']

        try:
            # Now generating the borns for the first time.
            super(FKSMultiProcess, self).__init__(procdef, **options)

        except diagram_generation.NoDiagramException as error:
            # If no born, then this process most likely does not have any.
            raise NoBornException("Born diagrams could not be generated for the "+\
               self['process_definitions'][0].nice_string().replace('Process',\
               'process')+". Notice that aMC@NLO does not handle loop-induced"+\
               " processes yet, but you can still use MadLoop if you want to "+\
               "only generate them."+\
               " For this, use the 'virt=' mode, without multiparticle labels.")

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
                    logger.warning('Use of multiparticles is non-trivial for NLO '+ \
                         'process generation and depends on the orders included, '+ \
                         'the process considered, as well as the PDF set chosen. '+ \
                         'See appendix D of arXiv:1804.10017 [hep-ph] for some '+ \
                         'guidance.')
                    break

        amps = self.get('amplitudes')

        # get the list of leptons from the model, in order to discard 
        # lepton-initiated processes unless the init_lep_split flag is specified
        if self['process_definitions']:
            leptons = self['process_definitions'][0]['model'].get_lepton_pdgs()
        else:
            leptons = []

        #generate reals, but combine them after having combined the borns
        for i, amp in enumerate(amps):
            # skip amplitudes with two initial leptons unless the init_lep_split option is True
            if not self['init_lep_split'] and \
                    all([l['id'] in leptons \
                    for l in [ll for ll in amp.get('process').get('legs') if not ll['state']]]):
                logger.info(('Discarding process%s.\n  If you want to include it, set the \n' + \
                             '  \'include_lepton_initiated_processes\' option to True') % \
                             amp.get('process').nice_string().replace('Process', ''))
                continue

            logger.info("Generating FKS-subtracted matrix elements for born process%s (%d / %d)" \
                % (amp['process'].nice_string(print_weighted=False, print_perturbated=False).replace('Process', ''),
                   i + 1, len(amps)))

            born = FKSProcess(amp, ncores_for_proc_gen = self['ncores_for_proc_gen'], \
                                   init_lep_split=self['init_lep_split'])
            self['born_processes'].append(born)

            born.generate_reals(self['pdgs'], self['real_amplitudes'], combine = False)

            # finally combine the real amplitudes
            born.combine_real_amplitudes()

        if not self['ncores_for_proc_gen']:
            # old generation mode 

            born_pdg_list = [[l['id'] for l in born.get_leglist()] \
                    for born in self['born_processes'] ]

            for born in self['born_processes']:
                for real in born.real_amps:
                    real.find_fks_j_from_i(born_pdg_list)
            if amps:
                if self['process_definitions'][0].get('NLO_mode') in ['all']:
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
        return diagram_generation.AmplitudeList([
                born.born_amp for \
                born in self['born_processes']])


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

        if not self['nlo_mixed_expansion']:
            # determine the orders to be used to generate the loop
            loop_orders = {}
            for  born in self['born_processes']:
                for coup, val in fks_common.find_orders(born.born_amp).items():
                    try:
                        loop_orders[coup] = max([loop_orders[coup], val])
                    except KeyError:
                        loop_orders[coup] = val


        for i, born in enumerate(self['born_processes']):
            myproc = copy.copy(born.born_amp['process'])
            #misc.sprint(born.born_proc)
            #misc.sprint(myproc.input_string())
            #misc.sprint(myproc['orders'])
            # if [orders] are not specified, then
            # include all particles in the loops
            # i.e. allow all orders to be perturbed
            # (this is the case for EW corrections, where only squared oders 
            # are imposed)
            if not self['nlo_mixed_expansion']:
                myproc['orders'] = loop_orders
            elif not myproc['orders']:
                    myproc['perturbation_couplings'] = myproc['model']['coupling_orders']
            # take the orders that are actually used bu the matrix element
            myproc['legs'] = fks_common.to_legs(copy.copy(myproc['legs']))
            logger.info('Generating virtual matrix element with MadLoop for process%s (%d / %d)' \
                    % (myproc.nice_string(print_weighted= False, print_perturbated= False).replace(\
                                                             'Process', ''),
                        i + 1, len(self['born_processes'])))
            try:
                myamp = loop_diagram_generation.LoopAmplitude(myproc,  
                                                loop_filter=self['loop_filter'])
                born.virt_amp = myamp
            except InvalidCmd:
                logger.debug('invalid command for loop')
                pass


class FKSRealProcess(object): 
    """Contains information about a real process:
    -- fks_infos (list containing the possible fks configs for a given process
    -- amplitude 
    -- is_to_integrate
    """
    
    def __init__(self, born_proc, leglist, ij, ij_id, born_pdgs, splitting_type,
                 perturbed_orders = ['QCD']): #test written
        """Initializes the real process based on born_proc and leglist.
        Stores the fks informations into the list of dictionaries fks_infos
        """      
        #safety check
        assert type(splitting_type) == list and not type(splitting_type) == str 
        self.fks_infos = []
        for leg in leglist:
            if leg.get('fks') == 'i':
                i_fks = leg.get('number')
                # i is a gluon or a photon
                need_color_links = leg.get('massless') \
                        and leg.get('spin') == 3 \
                        and leg.get('self_antipart') \
                        and leg.get('color') == 8
                need_charge_links = leg.get('massless') \
                        and leg.get('spin') == 3 \
                        and leg.get('self_antipart') \
                        and leg.get('color') == 1
            if leg.get('fks') == 'j':
                j_fks = leg.get('number')
        self.fks_infos.append({'i': i_fks, 
                               'j': j_fks, 
                               'ij': ij, 
                               'ij_id': ij_id, 
                               'underlying_born': born_pdgs,
                               'splitting_type': splitting_type,
                               'need_color_links': need_color_links,
                               'need_charge_links': need_charge_links,
                               'extra_cnt_index': -1})

        self.process = copy.copy(born_proc)
        self.process['perturbation_couplings'] = \
                copy.copy(born_proc['perturbation_couplings'])
        for o in splitting_type:
            if o not in self.process['perturbation_couplings']:
                self.process['perturbation_couplings'].append(o)
        # set the orders to empty, to force the use of the squared_orders
        self.process['orders'] = copy.copy(born_proc['orders'])

        legs = [(leg.get('id'), leg) for leg in leglist]
        self.pdgs = array.array('i',[s[0] for s in legs])
        self.colors = [leg['color'] for leg in leglist]
        self.particle_tags = [leg['is_tagged'] for leg in leglist]
        if not self.process['perturbation_couplings'] == ['QCD']:
            self.charges = [leg['charge'] for leg in leglist]
        else:
            self.charges = [0.] * len(leglist)
        self.perturbation = 'QCD'
        self.process.set('legs', MG.LegList(leglist))
        self.process.set('legs_with_decays', MG.LegList())
        self.amplitude = diagram_generation.Amplitude()
        self.is_to_integrate = True
        self.is_nbody_only = False
        self.fks_j_from_i = {}
        self.missing_borns = []


    def generate_real_amplitude(self):
        """generates the real emission amplitude starting from self.process"""
        self.amplitude = diagram_generation.Amplitude(self.process)
        return self.amplitude


    def find_fks_j_from_i(self, born_pdg_list): #test written
        """Returns a dictionary with the entries i : [j_from_i], if the born pdgs are in 
        born_pdg_list"""
        fks_j_from_i = {}
        for i in self.process.get('legs'):
            fks_j_from_i[i.get('number')] = []
            if i.get('state'):
                for j in [l for l in self.process.get('legs') if \
                        l.get('number') != i.get('number')]:
                    for pert_order in self.process.get('perturbation_couplings'):
                        ijlist = fks_common.combine_ij(i, j, self.process.get('model'), {},\
                                                       pert=pert_order)
                        for ij in ijlist:
                            born_leglist = fks_common.to_fks_legs(
                                          copy.deepcopy(self.process.get('legs')), 
                                          self.process.get('model'))
                            born_leglist.remove(i)
                            born_leglist.remove(j)
                            born_leglist.insert(ij.get('number') - 1, ij)
                            born_leglist.sort(pert = self.perturbation)
                            if [leg['id'] for leg in born_leglist] in born_pdg_list \
                               and not j.get('number') in fks_j_from_i[i.get('number')]:
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


#helper functions

    def get_colors(self):
        """return the list of color representations 
        for each leg in born_amp"""
        return [leg.get('color') for \
                    leg in self.born_amp['process']['legs']]                    


    def get_charges(self):
        """return the list of charges
        for each leg in born_amp"""
        return [leg.get('charge') for \
                    leg in self.born_amp['process']['legs']]                    


    def get_nlegs(self):
        """return the number of born legs"""
        return len(self.born_amp['process']['legs'])


    def get_born_nice_string(self):
        """Return the nice string for the born process.
        """
        return self.born_amp['process'].nice_string()


    def get_pdg_codes(self):
        """return the list of the pdg codes
        of each leg in born_amp"""
        return [leg.get('id') for \
                    leg in self.born_amp['process']['legs']]                    


    def get_is_tagged(self):
        """return the list of the 'is_tagged' keys
        of each leg in born_amp"""
        return [leg.get('is_tagged') for \
                    leg in self.born_amp['process']['legs']]                    


    def get_leglist(self):
        """return the leg list
        for the born amp"""
        return fks_common.to_fks_legs( \
                self.born_amp['process']['legs'], \
                self.born_amp['process']['model'])


###############################################################################
    
    def __init__(self, start_proc = None, remove_reals = True, ncores_for_proc_gen=0, init_lep_split = False):
        """initialization: starts either from an amplitude or a process,
        then init the needed variables.
        remove_borns tells if the borns not needed for integration will be removed
        from the born list (mainly used for testing)
        ncores_for_proc_gen has the following meaning
           0 : do things the old way
           > 0 use ncores_for_proc_gen
           -1 : use all cores
        """
                
        self.reals = []
        self.myorders = {}
        self.real_amps = []
        self.remove_reals = remove_reals
        self.init_lep_split = init_lep_split
        self.nincoming = 0
        self.virt_amp = None
        self.perturbation = 'QCD'
        self.born_amp = diagram_generation.Amplitude()
        self.extra_cnt_amp_list = diagram_generation.AmplitudeList()
        self.ncores_for_proc_gen = ncores_for_proc_gen

        if not remove_reals in [True, False]:
            raise fks_common.FKSProcessError(\
                    'Not valid type for remove_reals in FKSProcess')

        if start_proc:
            #initilaize with process definition (for test purporses)
            if isinstance(start_proc, MG.Process):
                pertur = start_proc['perturbation_couplings']
                if pertur:
                    self.perturbation = sorted(pertur)[0]
                self.born_amp = diagram_generation.Amplitude(\
                                copy.copy(fks_common.sort_proc(\
                                        start_proc, pert = self.perturbation)))
            #initialize with an amplitude
            elif isinstance(start_proc, diagram_generation.Amplitude):
                pertur = start_proc.get('process')['perturbation_couplings']
                self.born_amp = diagram_generation.Amplitude(\
                                copy.copy(fks_common.sort_proc(\
                                    start_proc['process'], 
                                    pert = self.perturbation)))
            else:
                raise fks_common.FKSProcessError(\
                    'Not valid start_proc in FKSProcess')
            self.born_amp['process'].set('legs_with_decays', MG.LegList())

            # special treatment of photon is needed !
            #MZ to be fixed
            ###self.isr = set([leg.get(color) for leg in self.leglist if not leg.get('state')]) != set([zero])
            ###self.fsr = set([leg.get(color) for leg in self.leglist if leg.get('state')]) != set([zero])
            self.isr = False
            self.fsr = False
            #######
            self.nincoming = len([l for l in self.born_amp['process']['legs'] \
                                  if not l['state']])
                
            self.ndirs = 0
            # generate reals, when the mode is not LOonly
            # when is LOonly it is supposed to be a 'fake' NLO process
            # e.g. to be used in merged sampels at high multiplicities
            if self.born_amp['process']['NLO_mode'] != 'LOonly':
                self.find_reals()


    def generate_real_amplitudes(self, pdg_list, real_amp_list):
        """generates the real amplitudes for all the real emission processes, using pdgs and real_amps
        to avoid multiple generation of the same amplitude.
        Amplitude without diagrams are discarded at this stage"""

        no_diags_amps = []
        for amp in self.real_amps:
            try:
                amp.amplitude = real_amp_list[pdg_list.index(amp.pdgs)]
            except ValueError:
                amplitude = amp.generate_real_amplitude()
                if amplitude['diagrams']:
                    pdg_list.append(amp.pdgs)
                    real_amp_list.append(amplitude)
                else:
                    no_diags_amps.append(amp)
        
        for amp in no_diags_amps:
            self.real_amps.remove(amp)



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
        #copy the born process
        born_proc = copy.copy(self.born_amp['process'])
        born_pdgs = self.get_pdg_codes()
        leglist = self.get_leglist()
        extra_cnt_pdgs = []
        for i, real_list in enumerate(self.reals):
            # i is the born leg which splits
            # keep track of the id of the mother (will be used to constrct the
            # spin-correlated borns)
            ij_id = leglist[i].get('id')
            ij = leglist[i].get('number')
            for real_dict in real_list:
                nmom = 0

                # check first if other counterterms need to be generated
                # (e.g. g/a > q qbar)
                # this is quite a tricky business, as double counting
                # the singular configuration must be avoided. 
                # Let real, born, cnt be the real emission, the born process
                # (that will give the name to the P0_** dir) and the 
                # extra counterterm (obtained by the born process replacing
                # ij with the extra mother). 
                # If there are extra mothers, first check that
                # 1) born, at order born[squared_orders] - 
                #    2 * (the perturbation type of the real emission) has diagrams
                # 2) cnt at order born[squared_orders] - 
                #    2 * (the perturbation type of the extra mom) has diagrams

                cnt_amp = diagram_generation.Amplitude()
                born_cnt_amp = diagram_generation.Amplitude()
                mom_cnt = 0
                cnt_ord = None

                # check condition 1) above (has_coll_sing_born)
                born_proc_coll_sing = copy.copy(born_proc)
                born_proc_coll_sing['squared_orders'] = copy.copy(born_proc['squared_orders'])
                if born_proc_coll_sing['squared_orders'][real_dict['perturbation'][0]] < 2:
                    has_coll_sing_born = False
                else:
                    born_proc_coll_sing['squared_orders'][real_dict['perturbation'][0]] += -2
                    has_coll_sing_born = bool(diagram_generation.Amplitude(born_proc_coll_sing)['diagrams'])

                # check that there is at most one extra mother
                allmothers = []
                for order, mothers in real_dict['extra_mothers'].items():
                    allmothers += mothers
                    if mothers:
                        cnt_ord = order

                if len(allmothers) > 1:
                    raise fks_common.FKSProcessError(\
                            'Error, more than one extra mother has been found: %d', len(allmothers))
                # here we are sure to have just one extra mother
                    
                has_coll_sing_cnt = False
                if allmothers:
                    mom_cnt = allmothers[0]

                    # generate a new process with the mother particle 
                    # replaced by the new mother and with the
                    # squared orders changed accordingly

                    cnt_process = copy.copy(born_proc)
                    cnt_process['legs'] = copy.deepcopy(born_proc['legs'])
                    cnt_process['legs'][i]['id'] = mom_cnt
                    cnt_process['legs'] = fks_common.to_fks_legs(
                            cnt_process['legs'], cnt_process['model'])
                    cnt_process['squared_orders'] = \
                            copy.copy(born_proc['squared_orders'])

                    # check if the cnt amplitude exists with the current 
                    # squared orders (i.e. if it will appear as a P0 dir)
                    # if it does not exist, then no need to worry about anything, as all further
                    # checks will have stricter orders than here
                    
                    cnt_process_for_amp = copy.copy(cnt_process)
                    cnt_process_for_amp['squared_orders'] = copy.copy(cnt_process['squared_orders'])
                    cnt_amp = diagram_generation.Amplitude(cnt_process_for_amp)

                    if bool(cnt_amp['diagrams']) and \
                       cnt_process['squared_orders'][cnt_ord] >= 2:

                        # check condition 2) above (has_coll_sing_cnt)
                        # MZMZ17062014 beware that the Amplitude reorders the legs
                        cnt_process['squared_orders'][cnt_ord] += -2
                        born_cnt_amp = diagram_generation.Amplitude(cnt_process)
                        has_coll_sing_cnt = bool(born_cnt_amp['diagrams'])

                # remember there is at most one mom
                # now, one of these cases can occur
                #  a) no real collinear singularity exists (e.g. just the interference
                #     has to be integrated for the real emission). Add the real process to
                #     this born process if ij_id < mom_cnt. No extra_cnt is needed
                #  b) a collinear singularity exists, with the underlying born being the born
                #     process of this dir, while no singularity is there for the extra_cnt.
                #     In this case keep the real emission in this directory, without any 
                #     extra_cnt
                #  c) a collinear singularity exists, with the underlying born being the 
                #     extra_cnt, while no singularity is there for the born of the dir.
                #     In this case skip the real emission, it will be included in the
                #     directory of the extra cnt 
                #  d) a collinear singularity exists, and both the process-dir born and
                #     the extra cnt are needed to subtract it. Add the real process to
                #     this born process if ij_id < mom_cnt and keeping the extra_cnt 
                #
                # in all cases, remember that mom_cnt is set to 0 if no extra mother is there

                # the real emission has to be skipped if mom_cnt(!=0) < ij_id and either a) or d)
                if mom_cnt and mom_cnt < ij_id:
                    if ((not has_coll_sing_born and not has_coll_sing_cnt) or \
                        (has_coll_sing_born and has_coll_sing_cnt)):
                        continue

                # the real emission has also to be skipped in case of c)
                if has_coll_sing_cnt and not has_coll_sing_born:
                    continue

                # if we arrive here, we need to keep this real mession
                ij = leglist[i].get('number')
                self.real_amps.append(FKSRealProcess( \
                        born_proc, real_dict['leglist'], ij, ij_id, \
                        [born_pdgs], 
                        real_dict['perturbation'], \
                        perturbed_orders = born_proc['perturbation_couplings']))

                # keep the extra_cnt if needed
                if has_coll_sing_cnt:
                    # for the moment we just check the pdgs, regardless of any
                    # permutation in the final state
                    try:
                        indx = extra_cnt_pdgs.index([l['id'] for l in cnt_process['legs']])
                    except ValueError:
                        extra_cnt_pdgs.append([l['id'] for l in cnt_process['legs']])
                        assert cnt_amp != None
                        self.extra_cnt_amp_list.append(cnt_amp)
                        indx = len(self.extra_cnt_amp_list) - 1
                      
                    # update the fks infos
                    self.real_amps[-1].fks_infos[-1]['extra_cnt_index'] = indx
                    self.real_amps[-1].fks_infos[-1]['underlying_born'].append(\
                                            [l['id'] for l in cnt_process['legs']])
                    self.real_amps[-1].fks_infos[-1]['splitting_type'].append(cnt_ord)

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

        # check that all splitting types are of ['QCD'] type, otherwise return
        for real in self.real_amps:
            for info in real.fks_infos:
                if info['splitting_type'] != ['QCD']:
                    logger.info('link_born_real: skipping because not all splittings are QCD')
                    return


        for real in self.real_amps:
            for info in real.fks_infos:
                info['rb_links'] = fks_common.link_rb_configs(\
                        self.born_amp, real.amplitude,
                        info['i'], info['j'], info['ij'])



    def find_reals(self, pert_orders = []):
        """finds the FKS real configurations for a given process.
        self.reals[i] is a list of dictionaries corresponding to the real 
        emissions obtained splitting leg i.
        The dictionaries contain the leglist, the type (order) of the
        splitting and extra born particles which can give the same
        splitting (e.g. gluon/photon -> qqbar).
        If pert orders is empty, all the orders of the model will be used
        """
        
        model = self.born_amp['process']['model']
        # if [orders] are not specified, then
        # include all kind of splittings
        # i.e. allow all orders to be perturbed
        # (this is the case for EW corrections, where only squared oders 
        # are imposed)
        if not pert_orders:
            if not self.born_amp['process']['orders']:
                pert_orders = model['coupling_orders']
            else:
                pert_orders = self.born_amp['process']['perturbation_couplings']

        leglist = self.get_leglist()
        if list(range(len(leglist))) != [l['number']-1 for l in leglist]:
            raise fks_common.FKSProcessError('Disordered numbers of leglist')

        if [ i['state'] for i in leglist].count(False) == 1:
            decay_process=True
        else:
            decay_process=False

        # count the number of initial-state leptons
        ninit_lep = [l['id'] in model.get_lepton_pdgs() and not l['state'] for l in leglist].count(True)

        for i in leglist:
            i_i = i['number'] - 1
            self.reals.append([])

            # for 2->1 processes, map only the initial-state singularities
            # this is because final-state mapping preserves shat, which
            # is not possible in 2->1
            if len(leglist) == 3 and not decay_process and i['state']: 
                continue
            for pert_order in pert_orders:
                # no splittings for initial states in decay processes
                if decay_process and not i['state']:
                    splittings=[]
                # if there are leptons in the initial state and init_lep_split is False,
                # only split initial state leptons; do nothing for any other particle
                elif not self.init_lep_split and ninit_lep >= 1 and \
                  (i['state'] or i['id'] not in model.get_lepton_pdgs()):
                    splittings=[]
                else:
                    splittings = fks_common.find_splittings( \
                        i, model, {}, pert_order, \
                        include_init_leptons=self.init_lep_split)
                for split in splittings:
                    # find other 'mother' particles which can end up in the same splitting
                    extra_mothers = {}
                    for pert in pert_orders:
                        extra_mothers[pert] = fks_common.find_mothers(split[0], split[1], model, pert=pert,
                                        mom_mass=model.get('particle_dict')[i['id']]['mass'].lower())

                    #remove the current mother from the extra mothers    
                    if i['state']:
                        extra_mothers[pert_order].remove(i['id'])
                    else:
                        extra_mothers[pert_order].remove(model.get('particle_dict')[i['id']].get_anti_pdg_code())

                    self.reals[i_i].append({
                        'leglist': fks_common.insert_legs(leglist, i, split ,pert=pert_order),
                        'perturbation': [pert_order], 
                        'extra_mothers': extra_mothers})


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
                i_n = real_n.fks_infos[-1]['i']
                j_n = real_n.fks_infos[0]['j']
                ij_id_m = real_m.fks_infos[0]['ij_id']
                ij_id_n = real_n.fks_infos[0]['ij_id']
                if j_m > self.nincoming and j_n > self.nincoming:
                    # make sure i and j in the two real emissions have the same mother 
                    if (ij_id_m != ij_id_n):
                        continue
                    if (real_m.get_leg_i()['id'] == real_n.get_leg_i()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_j()['id']) \
                        or \
                       (real_m.get_leg_i()['id'] == real_n.get_leg_j()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_i()['id']):
                        if i_m > i_n:
                            if real_m.get_leg_i()['id'] == -real_m.get_leg_j()['id']:
                                self.real_amps[m].is_to_integrate = False
                            else:
                                self.real_amps[n].is_to_integrate = False
                        elif i_m == i_n and j_m > j_n:
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

    

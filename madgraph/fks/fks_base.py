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

from madgraph import InvalidCmd

logger = logging.getLogger('madgraph.fks_base')


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
        self['born_processes'] = FKSProcessList()
        if not 'OLP' in self.keys():
            self['OLP'] = 'MadLoop'
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcess, self).get_sorted_keys()
        keys += ['born_processes', 'real_amplitudes', 'real_pdgs', 'has_isr', 
                 'has_fsr', 'OLP']
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
                                                     
        return super(FKSMultiProcess,self).filter(name, value)

    
    def __init__(self, *arguments, **options):
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
        
        self['real_amplitudes'] = diagram_generation.AmplitudeList()
        self['pdgs'] = []
        
        if 'OLP' in options.keys():
            self['OLP']=options['OLP']
            del options['OLP']

        try:
            # Now generate the borns for the first time:
            #---the first step just sets the internal variables according to the 
            #process definition without generating the amplitudes
            super(FKSMultiProcess, self).__init__(*arguments, generate_amps = False)

            #---process definitions are corrected in order to be multi-perturbation
            # compilant
            for procdef in self['process_definitions']:
                if not procdef['orders']:
                    procdef.set('orders', diagram_generation.MultiProcess.find_optimal_process_orders(procdef))

                # set the squared orders consistently with the perturbation specified by the user
                if not procdef['squared_orders']:
                    for ord, val in procdef['orders'].items():
                        procdef['squared_orders'][ord] = 2 * val

                # then increase the orders which are perturbed
                for pert in procdef['perturbation_couplings']:
                    # if orders have been specified increase them
                    if procdef['orders'].keys() != ['WEIGHTED']:
                        try:
                            procdef['orders'][pert] += 2
                        except KeyError:
                            procdef['orders'][pert] = 2
                        try:
                            procdef['squared_orders'][pert] += 2
                        except KeyError:
                            procdef['squared_orders'][pert] = 2

                # update also the WEIGHTED entry
                if 'WEIGHTED' in procdef['orders'].keys():
                    procdef['orders']['WEIGHTED'] += 1 * \
                            max([procdef.get('model').get('order_hierarchy')[ord] for \
                            ord in procdef['perturbation_couplings']])

                    procdef['squared_orders']['WEIGHTED'] += 2 * \
                            max([procdef.get('model').get('order_hierarchy')[ord] for \
                            ord in procdef['perturbation_couplings']])


            #---now generate the amplitudes
            self.get('amplitudes')

        except InvalidCmd as error:
            # If no born, then this process most likely does not have any.
            raise InvalidCmd, "Born diagrams could not be generated for the "+\
               self['process_definitions'][0].nice_string().replace('Process',\
               'process')+". Notice that aMC@NLO does not handle loop-induced"+\
               " processes yet, but you can still use MadLoop if you want to "+\
               "only generate them."+\
               " For this, use the 'virt=' mode, without multiparticle labels." 

        #check process definition(s):
        # a process such as g g > g g will lead to real emissions 
        #   (e.g: u g > u g g ) which will miss some corresponding born,
        #   leading to non finite results
        perturbation = []
        for procdef in self['process_definitions']:
            soft_particles = []
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

        amps = self.get('amplitudes')
        #generate reals, but combine them after having combined the borns
        for i, amp in enumerate(amps):
            logger.info("Generating FKS-subtracted matrix elements for born process%s (%d / %d)" \
                % (amp['process'].nice_string(print_weighted=False).replace(\
                                                                 'Process', ''),
                 i + 1, len(amps)))

            born = FKSProcess(amp)
            self['born_processes'].append(born)
            born.generate_reals(self['pdgs'], self['real_amplitudes'], combine = False)

        # now combine the borns, identifying photons and gluons
        self.combine_born_amplitudes()

        # finally combine the real amplitudes
        for born in self['born_processes']:
            born.combine_real_amplitudes()


        born_pdg_list = []
        for born in self['born_processes']:
            born_pdg_list.extend(born.get_pdg_codes())

        missing_born_pdg_list = []
        missing_born_amp_list = []
        for born in self['born_processes']:
            for real in born.real_amps:
                real.find_fks_j_from_i(born_pdg_list)
                real.find_missing_borns(born_pdg_list, \
                                        missing_born_pdg_list, \
                                        missing_born_amp_list)


        if amps:
            if self['process_definitions'][0].get('NLO_mode') == 'all':
                self.generate_virtuals()
            
            elif not self['process_definitions'][0].get('NLO_mode') in ['all', 'real']:
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

            if n_diag_virt == 0 and n_diag_real ==0:
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


    def combine_born_amplitudes(self):
        """combine together born amplitudes identifying photons and gluons.
        This is needed for consistent subtraction of QED/mixed singularities
        """
        pdg_list = []
        new_borns = FKSProcessList()
        for this in self['born_processes']:
            try:
                other = new_borns[pdg_list.index(this.get_pdg_codes_ag())]
                other.born_amp_list.extend(this.born_amp_list)
                other.real_amps.extend(this.real_amps)
            except ValueError:
                pdg_list.append(this.get_pdg_codes_ag())
                new_borns.append(this)

        self['born_processes'] = new_borns


    def add(self, other):
        """combines self and other, extending the lists of born/real amplitudes"""
        self['born_processes'].extend(other['born_processes'])
        self['real_amplitudes'].extend(other['real_amplitudes'])
        self['pdgs'].extend(other['pdgs'])
        self['has_isr'] = self['has_isr'] or other['has_isr']
        self['has_fsr'] = self['has_fsr'] or other['has_fsr']
        self['OLP'] = other['OLP']


    def get_born_amplitudes(self):
        """return an amplitudelist with the born amplitudes"""
        return diagram_generation.AmplitudeList([amp for \
                born in self['born_processes'] for \
                amp in born.born_amp_list])


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

        for i, born in enumerate(self['born_processes']):
            logger.info('Generating virtual matrix elements using MadLoop:')
            myproc = copy.copy(born.born_proc)
            # take the orders that are actually used bu the matrix element
            myproc['orders'] = fks_common.find_orders(born.born_amp)
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
    """
    
    def __init__(self, born_proc, leglist, ij, ij_id, born_pdgs, splitting_type,
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
                               'need_charge_links': need_charge_links})

        self.process = copy.copy(born_proc)
        legs = [(leg.get('id'), leg) for leg in leglist]
        self.pdgs = array.array('i',[s[0] for s in legs])
        self.colors = [leg['color'] for leg in leglist]
        self.charges = [leg['charge'] for leg in leglist]
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


    def find_missing_borns(self, born_pdg_list, missing_pdgs, missing_amps):
        """returns a list of dictionaries {i:, j:, born_amp:} corresponding to
        borns which have not been generated but need to be included as 
        counterterms"""
        missing_borns = []
        for i in self.process.get('legs'):
            if i.get('state'):
                for j in self.process.get('legs'):
                    if j.get('number') != i.get('number'):
                        for pert_ord in self.process.get('perturbation_couplings'):
                            ijlist = fks_common.combine_ij(i, j, self.process.get('model'), {},\
                                                       pert=pert_ord)
                            for ij in ijlist:
                                born_leglist = fks_common.to_fks_legs(
                                              copy.deepcopy(self.process.get('legs')), 
                                              self.process.get('model'))
                                born_leglist.remove(i)
                                born_leglist.remove(j)
                                born_leglist.insert(ij.get('number') - 1, ij)
                                born_leglist.sort(pert = self.perturbation)
                                pdgs = [l['id'] for l in born_leglist]
                                if pdgs not in born_pdg_list:
                                    #this is a missing process, check now that it has diagrams
                                    # (at order real- order splitting)
                                    try:
                                        amp = missing_amps[missing_pdgs.index(pdgs)]
                                    except ValueError:
                                        missing_pdgs.append(pdgs)
                                        newproc = copy.deepcopy(self.process)
                                        newproc['legs'] = born_leglist
                                        try:
                                            newproc['orders'][pert_ord] = \
                                                max(newproc['orders'][pert_ord] - 1, 0)
                                        except KeyError:
                                            pass
                                        amp = diagram_generation.Amplitude(newproc)
                                        missing_amps.append(amp)
                                    if amp['diagrams']:
                                        missing_borns.append(\
                                            {'i': i.get('number'),
                                             'j': j.get('number'),
                                             'born_amp': amp})
                                                            
        self.missing_borns = missing_borns
        return missing_borns


    def find_fks_j_from_i(self, born_pdg_list): #test written
        """Returns a dictionary with the entries i : [j_from_i], if the born pdgs are in 
        born_pdg_list"""
        fks_j_from_i = {}
        for i in self.process.get('legs'):
            fks_j_from_i[i.get('number')] = []
            if i.get('state'):
                for j in self.process.get('legs'):
                    if j.get('number') != i.get('number') :
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


#helper functions

    def get_colors(self):
        """return the list of color representations 
        for each amp in born_amp list"""
        return [[leg.get('color') for \
                    leg in amp['process']['legs']] for \
                    amp in self.born_amp_list]
                    

    def get_charges(self):
        """return the list of charges
        for each amp in born_amp list"""
        return [[leg.get('charge') for \
                    leg in amp['process']['legs']] for \
                    amp in self.born_amp_list]


    def get_nlegs(self):
        """return the number of born legs
        for each amp in born_amp list"""
        return [[len(amp['process']['legs'])] for \
                    amp in self.born_amp_list]


    def get_born_nice_string(self):
        """Return the nice string for the born process.
        If there is only one amplitude, return its nice string
        else (if there are many with g <-> a identified) replace
        g / a by V
        """
        string = self.born_amp_list[0]['process'].nice_string()
        if len(self.born_amp_list) > 1:
            string.replace('a', 'V').replace('g', 'V')
        return string


    def get_pdg_codes(self):
        """return the list of pdg codes
        for each amp in born_amp list"""
        return [[leg.get('id') for \
                    leg in amp['process']['legs']] for \
                    amp in self.born_amp_list]


    def get_pdg_codes_ag(self):
        """return the list of pdg codes replacing the photon pdg (22) with the 
        gluon one (21)
        for each amp in born_amp list"""
        a_id = 22
        g_id = 21
        return [[leg.get('id') if leg.get('id') != a_id else g_id for \
                    leg in amp['process']['legs']] for \
                    amp in self.born_amp_list]

     
    def get_leglist(self):
        """return the leg list
        for each amp in born_amp list"""
        return [fks_common.to_fks_legs( \
                amp['process']['legs'], \
                amp['process']['model']) for \
                amp in self.born_amp_list]


###############################################################################
    
    def __init__(self, start_proc = None, remove_reals = True):
        """initialization: starts either from an amplitude or a process,
        then init the needed variables.
        remove_borns tells if the borns not needed for integration will be removed
        from the born list (mainly used for testing)"""
                
        self.splittings = {}
        self.reals = []
        self.myorders = {}
        self.real_amps = []
        self.remove_reals = remove_reals
        self.nincoming = 0
        self.virt_amp = None
        self.perturbation = 'QCD'
        self.born_amp_list = diagram_generation.AmplitudeList()

        if not remove_reals in [True, False]:
            raise fks_common.FKSProcessError(\
                    'Not valid type for remove_reals in FKSProcess')

        if start_proc:
            #initilaize with process definition (for test purporses)
            if isinstance(start_proc, MG.Process):
                pertur = start_proc['perturbation_couplings']
                if pertur:
                    self.perturbation = sorted(pertur)[0]
                self.born_amp_list.append(diagram_generation.Amplitude(\
                                copy.copy(fks_common.sort_proc(\
                                        start_proc, pert = self.perturbation))))
            #initialize with an amplitude
            elif isinstance(start_proc, diagram_generation.Amplitude):
                pertur = start_proc.get('process')['perturbation_couplings']
                self.born_amp_list.append(diagram_generation.Amplitude(\
                                copy.copy(fks_common.sort_proc(\
                                    start_proc['process'], 
                                    pert = self.perturbation))))
            else:
                raise fks_common.FKSProcessError(\
                    'Not valid start_proc in FKSProcess')
            self.born_amp_list[0]['process'].set('legs_with_decays', MG.LegList())

            logger.info("Generating FKS-subtracted matrix elements for born process%s" \
                % self.born_amp_list[0]['process'].nice_string(print_weighted=False).replace(\
                                                                 'Process', '')) 

            # special treatment of photon is needed !
            #MZ to be fixed
            ###self.isr = set([leg.get(color) for leg in self.leglist if not leg.get('state')]) != set([zero])
            ###self.fsr = set([leg.get(color) for leg in self.leglist if leg.get('state')]) != set([zero])
            self.isr = False
            self.fsr = False
            print 'MZ fix isr/fsr' 
            #######
            for leg in self.born_amp_list[0]['process']['legs']:
                if not leg['state']:
                    self.nincoming += 1
                
            self.find_reals(self.born_amp_list[0]['process'].get('perturbation_couplings'))


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

        born_proc = copy.copy(self.born_amp_list[0]['process'])
        born_pdgs = self.get_pdg_codes()
        # safety check
        if len(born_pdgs) > 1:
            raise FKSProcessError('generate_reals should be called before ' + \
                    ' combining born amplitudes')
        leglist = self.get_leglist()[0]
        for i, real_list in enumerate(self.reals):
            # i is a gluon or photon,i.e. g(a) > ffx or gg
            ij_id = leglist[i].get('id')
            ij = leglist[i].get('number')
            for real_dict in real_list:
                ij = leglist[i].get('number')
                self.real_amps.append(FKSRealProcess( \
                        born_proc, real_dict['leglist'], ij, ij_id, \
                        born_pdgs[0], 
                        real_dict['perturbation'], \
                        perturbed_orders = born_proc['perturbation_couplings']))
        self.find_reals_to_integrate()
        if combine:
            self.combine_real_amplitudes()
        self.generate_real_amplitudes(pdg_list, real_amp_list)
        print 'MZ, fix LINK'
        #self.link_born_reals()


    def link_born_reals(self):
        """create the rb_links in the real matrix element to find 
        which configuration in the real correspond to which in the born
        """
        for real in self.real_amps:
            for info in real.fks_infos:
                info['rb_links'] = fks_common.link_rb_configs(\
                        self.born_amp, real.amplitude,
                        info['i'], info['j'], info['ij'])


    def find_reals(self, pert_orders):
        """finds the FKS real configurations for a given process.
        self.reals[i] is a list of dictionaries corresponding to the real 
        emissions obtained splitting leg i.
        The dictionaries contain the leglist and the type (order) of the
        splitting
        """
        leglist = self.get_leglist()[0]
        if range(len(leglist)) != [l['number']-1 for l in leglist]:
            raise fks_common.FKSProcessError('Disordered numbers of leglist')
        for i in leglist:
            i_i = i['number'] - 1
            self.reals.append([])
            for pert_order in pert_orders:
                self.splittings[i_i] = fks_common.find_splittings(i, self.born_amp_list[0]['process']['model'], {}, pert_order)
                for split in self.splittings[i_i]:
                    self.reals[i_i].append({
                        'leglist': fks_common.insert_legs(leglist, i, split,pert=pert_order),
                        'perturbation': pert_order})
                


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

    

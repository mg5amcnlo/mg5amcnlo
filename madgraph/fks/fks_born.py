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
import madgraph.fks.fks_real as fks_real
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_born')


#===============================================================================
# FKS Process
#===============================================================================
class FKSMultiProcessFromBorn(diagram_generation.MultiProcess): #test written
    """a multi process class that contains informations on the born processes 
    and the reals"""
    
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
    
    def __init__(self, amp_list, *arguments):
        """initialize the original multiprocess, then generates the amps for the 
        borns, then geneare the born processes and the reals"""
                
        super(FKSMultiProcessFromBorn, self).__init__(*arguments)   
#        amps = self.get('amplitudes')
        amps = amp_list
        real_amplist = []
        real_amp_id_list = []
        for amp in amps:
            born = FKSProcessFromBorn(amp)
            self['born_processes'].append(born)
            born.generate_reals(real_amplist, real_amp_id_list)


class FKSRealProcess(object): 
    """contains information about a real process:
    -- i/j/ij fks (ij refers to the born leglist)
    -- ijglu
    -- amplitude 
    -- is_to_integrate
    -- need_color_links
    -- leg permutation<<REMOVED!"""
    
    def __init__(self, born_proc, leglist, ij, ijglu, amplist, amp_id_list,
                 perturbed_orders = ['QCD']): #test written
        """initialize the real process based on born_proc and leglist,
        then checks if the amplitude has already been generated before in amplist,
        if not it generates the amplitude and appends to amplist.
        amp_id_list contains the arrays of the pdg codes of the amplitudes in amplist"""      
        for leg in leglist:
            if leg.get('fks') == 'i':
                self.i_fks = leg.get('number')
                self.need_color_links = leg.get('massless') \
                        and leg.get('spin') == 3 \
                        and leg.get('color') == 8
            if leg.get('fks') == 'j':
                self.j_fks = leg.get('number')
        self.ijglu = ijglu
        self.ij = ij
        self.process = copy.copy(born_proc)
        orders = copy.copy(born_proc.get('orders'))
        for order in perturbed_orders:
            orders[order] +=1
            if order == 'QCD':
                orders['WEIGHTED'] +=1
            else: 
                orders['WEIGHTED'] +=2

        self.process.set('orders', orders)
        legs = [(leg.get('id'), leg) for leg in \
                         leglist ]
        pdgs = array.array('i',[s[0] for s in legs]) 
        self.process.set('legs', MG.LegList(leglist))
        self.amplitude = diagram_generation.Amplitude(self.process)
        self.pdgs = pdgs
        self.colors = [leg['color'] for leg in leglist]
        self.is_to_integrate = True
        self.is_nbody_only = False

    def find_fks_j_from_i(self):
        """returns a dictionary with the entries i : [j_from_i]"""
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
        """returns leg corresponding to i_fks"""
        return self.process.get('legs')[self.i_fks - 1]

    def get_leg_j(self): #test written
        """returns leg corresponding to i_fks"""
        return self.process.get('legs')[self.j_fks - 1]


class FKSProcessFromBornList(MG.PhysicsObjectList):
    """class to handle lists of FKSProcesses"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromBorn for the list."""
        return isinstance(obj, FKSProcessFromBorn)

            
class FKSProcessFromBorn(object):
    """the class for an FKS process. Starts from the born process """  

    
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
        
        if start_proc:
            if isinstance(start_proc, MG.Process):
                self.born_proc = fks_common.sort_proc(start_proc) 
                self.born_amp = diagram_generation.Amplitude(self.born_proc)
            elif isinstance(start_proc, diagram_generation.Amplitude):
                self.born_proc = fks_common.sort_proc(start_proc.get('process'))
                self.born_amp = diagram_generation.Amplitude(self.born_proc)

            self.model = self.born_proc['model']

            self.leglist = fks_common.to_fks_legs(
                                    self.born_proc['legs'], self.model)
            self.nlegs = len(self.leglist)
            for leg in self.leglist:
                self.pdg_codes.append(leg['id'])
                self.colors.append(leg['color'])
                if not leg['state']:
                    self.nincoming += 1
            # find the correct qcd/qed orders from born_amp
            orders = {}
            for dia in self.born_amp.get('diagrams'):
                for o, val in dia['orders'].items():
                    if o in orders.keys():
                        orders[o] = max(orders[o], val)
                    else:
                        orders[o] =val
            self.born_proc['orders'] = orders
                
            self.ndirs = 0
            self.fks_config_string = ""

            self.find_reals()

            self.find_color_links()


    def link_rb_confs(self):
        """links the configurations of the born amp with those of the real amps.
        Uses the function defined in fks_common"""
        links = []
        for real in self.real_amps:
            links.append(fks_common.link_rb_conf(self.born_amp, real.amplitude, 
                                                 real.i_fks, real.j_fks, real.ij) )

        return links


    def find_color_links(self): #test written
        """finds all the possible color links between two legs of the born.
        Uses the find_color_links function in fks_common"""
        self.color_links = fks_common.find_color_links(self.leglist)
        return self.color_links

        
    def generate_reals(self, amplist, amp_id_list): #test written
        """for all the possible splittings, creates an FKSRealProcess, keeping
        track of all the already generated processes through amplist and amp_id_list
        It removes double counted configorations from the ones to integrates and
        sets the one which includes the bosn (is_nbody_only)"""

        for i, list in enumerate(self.reals):
            if self.leglist[i]['massless'] and self.leglist[i]['spin'] == 3:
                ijglu = i+1
            else:
                ijglu = 0
            for l in list:
                ij = self.leglist[i].get('number')
                self.real_amps.append(FKSRealProcess(\
                        self.born_proc, l, ij, ijglu, amplist, amp_id_list))
        self.find_reals_to_integrate()
        self.find_real_nbodyonly()


    
    def find_reals(self):
        """finds the FKS real configurations for a given process"""
        for i in self.leglist:
            i_i = i['number'] -1
            self.reals.append([])
            self.splittings[i_i] = fks_common.find_splittings(i, self.model, {})
            for split in self.splittings[i_i]:
                self.reals[i_i].append(
                            fks_common.insert_legs(self.leglist, i, split))

    def find_reals_to_integrate(self): #test written
        """finds double countings in the real emission configurations, sets the 
        is_to_integrate variable and if "self.remove_reals" is True removes the 
        not needed ones from the born list"""
        #find the initial number of real configurations
        ninit = len(self.real_amps)
        remove = self.remove_reals
        
        for m in range(ninit):
            for n in range(m+1, ninit):
                real_m = self.real_amps[m]
                real_n = self.real_amps[n]
                if real_m.j_fks > self.nincoming and \
                   real_n.j_fks > self.nincoming:
                    if (real_m.get_leg_i()['id'] == real_n.get_leg_i()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_j()['id']) \
                        or \
                       (real_m.get_leg_i()['id'] == real_n.get_leg_j()['id'] \
                        and \
                        real_m.get_leg_j()['id'] == real_n.get_leg_i()['id']):
                        if real_m.i_fks > real_n.i_fks:
                            self.real_amps[n].is_to_integrate = False
                        elif real_m.i_fks == real_n.i_fks and \
                             real_m.j_fks > real_n.j_fks:
                            self.real_amps[n].is_to_integrate = False
                        else:
                            self.real_amps[m].is_to_integrate = False
                elif real_m.j_fks <= self.nincoming and \
                     real_n.j_fks == real_m.j_fks:
                    if real_m.get_leg_i()['id'] == real_n.get_leg_i()['id'] and \
                       real_m.get_leg_j()['id'] == real_n.get_leg_j()['id']:
                        if real_m.i_fks > real_n.i_fks:
                            self.real_amps[n].is_to_integrate = False
                        else:
                            self.real_amps[m].is_to_integrate = False
        if remove:
            newreal_amps = []
            for real in self.real_amps:
                if real.is_to_integrate:
                    newreal_amps.append(real)
            self.real_amps = newreal_amps
    
    def find_real_nbodyonly(self):
        """finds the real emission configuration that includes the nbody contribution
        in the virt0 and born0 running mode. By convention it is the real emission 
        that has the born legs + 1 extra particle (gluon if pert==QCD) and for
        which i/j_fks are the largest"""
        imax = 0
        jmax = 0
        chosen = -1
        for n, real in enumerate(self.real_amps):
            equal_legs = 0
            born_pdgs = copy.copy(self.pdg_codes)
            real_pdgs = list(copy.copy(real.pdgs))
            
            if born_pdgs[:2] == real_pdgs[:2]: #same initial state:
                same_parts = True
                for p in born_pdgs:
                    try:
                        real_pdgs.remove(p)
                    except:
                        same_parts = False
                        break
                if same_parts and real.i_fks >= imax and real.j_fks >= jmax and \
                        real.is_to_integrate:
                    chosen = n
        
        if chosen >= 0:
            self.real_amps[chosen].is_nbody_only = True
        else:
            raise fks_common.FKSProcessError, \
                  "%s \n Error, nbodyonly configuration not found" % \
                  self.born_proc.input_string()
            

            
            
        

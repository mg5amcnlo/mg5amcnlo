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
import copy
import logging
import array

logger = logging.getLogger('madgraph.fks_born')



def legs_to_color_link_string(leg1, leg2):
    """returns a dictionary containing:
    --string: the color link between the two particles, to be appended to
        the old color string
    --replacements: a pair of lists containing the replacements of the color 
        indices in the old string to match the link"""
    dict={}
    min_index = -3000
    iglu = min_index*2
    string = color_algebra.ColorString()
    replacements = []
    if leg1 != leg2:
        for leg in [leg1,leg2]:
            min_index -= 1
            num = leg.get('number')
            replacements.append([num, min_index])
            icol =1
            if not leg.get('state'):
                icol =-1
            if leg.get('color') * icol == 3:
                string.append(color_algebra.T(iglu, num, min_index))
            elif leg.get('color') * icol == -3:
                string.append(color_algebra.T(iglu, min_index, num))
            elif leg.get('color') * icol == 8:
                string.append(color_algebra.f(iglu, num, min_index))
    else:
        icol =1
        if not leg1.get('state'):
            icol =-1
        num = leg1.get('number')
        replacements.append([num, min_index -1])
        if leg1.get('color') * icol == 3:
            string.append(color_algebra.T(iglu, num, min_index-2))
            string.append(color_algebra.T(iglu, min_index-2, min_index-1))
        elif leg1.get('color') * icol == -3:
            string.append(color_algebra.T(iglu, min_index-2, num))
            string.append(color_algebra.T(iglu, min_index-1, min_index-2))    
    
    dict['replacements'] = replacements
    dict['string'] = string
    return dict


class FKSLeglist(MG.LegList):
    """list of FKSLegs"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSLeg for the list."""
        return isinstance(obj, FKSLeg)

class FKSLeg(MG.Leg):
    """a class for FKS legs: it inherits from the ususal leg class, with two
    extra keys in the dictionary: 
    -'fks', whose value can be 'i', 'j' or 'n' (for "normal" particles) 
    -'color', which gives the color of the leg
    -'massless', boolean, true if leg is massless
    -'spin' which gives the spin of leg"""

    def default_setup(self):
        """Default values for all properties"""
        super(FKSLeg, self).default_setup()

        self['fks'] = 'n'
        self['color'] = 0
        self['massless'] = True
        self['spin'] = 0
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSLeg, self).get_sorted_keys()
        keys += ['fks', 'color', 'massless', 'spin']
        return keys

    
    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'fks':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for leg fks flag" % str(value)

        if name in ['color', 'spin']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid leg color " % \
                                                                    str(value)

        if name == 'massless':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid boolean for leg flag massless" % \
                                                                    str(value)
                                                                
        return super(FKSLeg,self).filter(name, value)
 
    
    def __lt__(self, other):
        #two initial state legs are sorted by their number:
        if (not self.get('state') and not other.get('state')):
            return self.get('number') < other.get('number')
        
        #an initial state leg comes before a final state leg
        if (self.get('state') or other.get('state')) and \
          not (self.get('state') and other.get('state')):
            return other.get('state')
        
        #two final state particles are ordered by increasing color
        elif self.get('state') and other.get('state'):
            if abs(self.get('color')) != abs(other.get('color')):
                return abs(self.get('color')) < abs(other.get('color'))
        #particles of the same color are ordered according to the pdg code
            else:
                if abs(self.get('id')) != abs(other.get('id')):
                    return abs(self.get('id')) < abs(other.get('id'))
                elif self.get('id') != other.get('id') :
        #for the same flavour qqbar pair, first take the quark 
                    return self.get('id') > other.get('id')
        # i fks > j fks > n fks        
                else: 
                    return not self.get('fks') < other.get('fks') 
                    
                         

#===============================================================================
# FKS Process
#===============================================================================
class FKSMultiProcess(diagram_generation.MultiProcess): #test written
    """a multi process class that contains informations on the born processes 
    and the reals"""
    
    def default_setup(self):
        """Default values for all properties"""
        super(FKSMultiProcess, self).default_setup()

        self['born_processes'] = FKSProcessFromBornList()
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSMultiProcess, self).get_sorted_keys()
        keys += ['born_processes']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'born_processes':
            if not isinstance(value, FKSProcessFromBornList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list for born_processes " % str(value)
                                                                
        return super(FKSMultiProcess,self).filter(name, value)
    
    def __init__(self, *arguments):
        """initialize the original multiprocess, then generates the amps for the 
        borns, then geneare the born processes and the reals"""
        super(FKSMultiProcess, self).__init__(*arguments)
        
        amps = self.get('amplitudes')
        real_amplist = []
        real_amp_id_list = []
        for amp in amps:
            born = FKSProcessFromBorn(amp)
            self['born_processes'].append(born)
            born.generate_reals(real_amplist, real_amp_id_list)


class FKSRealProcess(): #test written
    """contains information about a real process:
    -- i/j fks
    -- amplitude
    -- leg permutation"""
    
    def __init__(self, born_proc, leglist, amplist, amp_id_list):
        """initialize the real process based on born_proc and leglist,
        then checks if the amplitude has already been generated before in amplist,
        if not it generates the amplitude and appends to amplist.
        amp_id_list contains the arrays of the pdg codes of the amplitudes in amplist"""
        
        for leg in leglist:
            if leg.get('fks') == 'i':
                self.i_fks = leg.get('number')
            if leg.get('fks') == 'j':
                self.j_fks = leg.get('number')
                
        print "in FKSRealProcess BEFORE permutation", [l['id'] for l in leglist], "i ", self.i_fks,\
        "   j ", self.j_fks

        self.process = copy.copy(born_proc)
        orders = copy.copy(born_proc.get('orders'))
        for n, o in orders.items():
            if n != 'QCD':
                orders[n] = o
            else:
                orders[n] = o +1
        self.process.set('orders', orders)
        sorted_legs = [(leg.get('id'), leg) for leg in \
                         leglist if not leg.get('state')] + \
                         sorted([(leg.get('id'), leg) for leg in \
                                 leglist if leg.get('state')])
        pdgs = array.array('i',[s[0] for s in sorted_legs]) 
        self.permutation= [leg[1].get('number') for leg in sorted_legs]
        self.process.set('legs', MG.LegList([l[1] for l in sorted_legs]))
        for i, leg in enumerate(sorted_legs):
            leg[1].set('number', i+1)
        try:
            self.amplitude = amplist[amp_id_list.index(pdgs)]
        except ValueError:
            self.amplitude = diagram_generation.Amplitude(self.process)
            amplist.append(self.amplitude)
            amp_id_list.append(pdgs)
        self.pdgs = pdgs
        print "in FKSRealProcess after permutation", [l['id'] for l in leglist], "i ", self.i_fks,\
        "   j ", self.j_fks

            

class FKSProcessFromBornList(MG.PhysicsObjectList):
    """class to handle lists of FKSProcesses"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcessFromBorn for the list."""
        return isinstance(obj, FKSProcessFromBorn)

            
class FKSProcessFromBorn(object):
    """the class for an FKS process. Starts from the born process """  
    
    def __init__(self, start_proc = None):
        
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
        
        if start_proc:
            if isinstance(start_proc, MG.Process):
                self.born_proc = start_proc 
                self.born_amp = diagram_generation.Amplitude(self.born_proc)
            elif isinstance(start_proc, diagram_generation.Amplitude):
                self.born_amp = start_proc
                self.born_proc = start_proc.get('process')           
            self.model = self.born_proc['model']
            self.leglist = self.to_fks_legs(self.born_proc['legs'])
            self.nlegs = len(self.leglist)
            print [l.get('id') for l in self.leglist]
#           self.leglist = self.born_proc.get('legs')
#            for leg in self.leglist:
#                self.pdg_codes.append(leg['id'])
#                part = self.born_proc['model'].get('particle_dict')[leg['id']]
#                if part['is_part']:
#                    self.colors.append(part['color'])
#                else:
#                    self.colors.append(- part['color'])
                
            self.ndirs = 0
            self.fks_config_string = ""
            self.qcd_part = self.find_qcd_particles(self.born_proc['model'])
            self.brem_part = self.find_brem_particles(self.born_proc['model'])
            self.qcd_inter = self.find_qcd_interactions(self.born_proc['model'])
            self.find_reals()
            self.find_color_links()
            self.real_amps = []


    def find_color_links(self): #test written
        """finds all the possible color links between two legs of the born"""
        for leg1 in self.leglist:
            for leg2 in self.leglist:
                #legs must be colored and different, unless massive
                if (leg1.get('color') != 1 and leg2.get('color') != 1) \
                  and (leg1 != leg2 or not leg1.get('massless')):
                    self.color_links.append({
                        'legs' : [leg1, leg2],
                        'string' : legs_to_color_link_string(leg1, leg2)['string'],
                        'replacements' : legs_to_color_link_string(leg1, leg2)['replacements']})
             

    def generate_reals(self, amplist, amp_id_list): #test written
        """for all the possible splittings, creates an FKSRealProcess, keeping
        track of all the already generated processes through amplist and amp_id_list"""
        for l in sum(self.reals, []):
            self.real_amps.append(FKSRealProcess(\
                        self.born_proc, l, amplist, amp_id_list))

    def to_fks_leg(self, leg): #test written
        """given a leg or a dict with leg properties, 
        adds color, spin and massless entries"""
        fksleg =FKSLeg(leg)
        part = self.model.get('particle_dict')[leg['id']]
        fksleg['color'] = part.get_color()
        fksleg['massless'] = part['mass'].lower() == 'zero'
        fksleg['spin'] = part.get('spin')      
        return fksleg
    
    def to_fks_legs(self, leglist): #test written
        """given leglist, sets color and massless entries according to the model 
        variable.
        return a FKSLeglist"""
        fkslegs = FKSLeglist()     
        for leg in leglist:
            fkslegs.append(self.to_fks_leg(leg))
        return fkslegs      
            
    def find_qcd_particles(self,model): #test written
        """finds the QCD (i.e. colored) particles in the given model"""
        qcd_part = {}
        for i, p in model.get('particle_dict').items():
            if p['color'] != 1 :
                qcd_part[i] = p
        return qcd_part
    
    def find_brem_particles(self,model): #test_written
        """finds the particles that can be radiated (i.e. colored and massless) 
        in the given model"""
        brem_part = {}
        for i, p in model.get('particle_dict').items():
            if p['color'] != 1 and p['mass'].lower() == 'zero':
                brem_part[i] = p
        return brem_part
    
    def find_qcd_interactions(self, model, pert = ['QCD']):  #test_written
        """finds the interactions for which QCD order is >=1 in the given model"""
        qcd_inter = MG.InteractionList()
        for i, ii in model.get('interaction_dict').items():
            if any([p in ii['orders'].keys() for p in pert]) \
               and len(ii['particles']) ==3 :
                masslist = [p.get('mass').lower() for p in ii.get('particles')]
                masslist.remove('zero')
                if len(set(masslist)) == 1:
                    qcd_inter.append(ii)
        return sorted(qcd_inter)
    
    def add_numbers(self, legs):
        """sorts and adds the numbers to the leg list"""
        for i, leg in enumerate(legs):
            leg['number'] = i+1
        return legs
    
    def find_reals(self):
        """finds the FKS real configurations for a given process"""
        for i in self.leglist:
            i_i = i['number'] -1
            self.reals.append([])
            self.splittings[i_i] = self.find_splittings(i)
            for split in self.splittings[i_i]:
                self.reals[i_i].append(self.add_numbers(self.insert_legs(i, split)))
                print [l.get('id') for l in self.reals[i_i][-1]]
                print [l.get('fks') for l in self.reals[i_i][-1]]
                
    
    def insert_legs(self, leg, split):
        """returns the born process with leg splitted into split. """
        real = MG.Process(self.born_proc)
        leglist = copy.deepcopy(self.leglist)         
        #find the position of the first final state leg
        for i in range(len(leglist)):
            if leglist[-i-1].get('state'):
                firstfinal = len(leglist) -i -1
        real.set('orders', self.myorders)
        i = leglist.index(leg)
        leglist.remove(leg)

        for sleg in split:            
            leglist.insert(i, sleg)
            #keep track of the number for initial state legs
            if not sleg.get('state') and not leg.get('state'):
                leglist[i]['number'] = leg['number']
            i+= 1
            if i < firstfinal :
                i = firstfinal
        return leglist        
            
    def find_splittings(self, leg):
        """find the possible splittings corresponding to leg"""
        splittings = []
#check that the leg is a qcd leg
        if leg['id'] in self.qcd_part.keys():
            part = self.qcd_part[leg['id']]
            antipart = self.qcd_part[part.get_anti_pdg_code()]
            for ii in self.qcd_inter:
#check which interactions contain leg and at least one "brem" particles:
                parts = copy.deepcopy(ii['particles'])
                nbrem = 0
                if part in parts:
                    #pops the ANTI-particle of part from the interaction
                    parts.pop(parts.index(antipart))
                    for p in parts:
                        if p.get_pdg_code() in self.brem_part.keys():
                            nbrem += 1
                    if nbrem >=1:
                        #splittings.extend(sorted(self.split_leg(leg, parts)))
                        splittings.extend(self.split_leg(leg, parts))
        return splittings
                        
                        
    def split_leg(self, leg, parts): #test written
        """splits the leg into parts, and returns the two new legs"""
        #for an outgoing leg take the antiparticles
        split = []
        #for a final state particle one can have only a splitting
        if leg['state'] :
            split.append([])
            for part in parts:
                split[-1].append(self.to_fks_leg({'state' : True, \
                                 'id' : part.get_pdg_code()}))
                self.ij_final(split[-1])
                
                
        #while for an initial state particle one can have two splittings 
        # if the two partons are different
        else:
            if parts[0] != parts[1]:
                for part in parts:
                    cparts = copy.deepcopy(parts)
                    split.append([\
                              self.to_fks_leg({'state' : False, 
                                      'id' : cparts.pop(cparts.index(part)).get_pdg_code(),
                                      'fks' : 'j'}),
                              self.to_fks_leg({'state' : True,
                   'id' : cparts[0].get_anti_pdg_code(),
                   'fks' : 'i'})\
                              ])
            else:
                split.append([\
                              self.to_fks_leg({'state' : False, 
                                      'id' : parts[0].get_pdg_code(),
                                      'fks' : 'j'}),
                              self.to_fks_leg({'state' : True, 
                                      'id' : parts[1].get_anti_pdg_code(),
                                      'fks' : 'i'}) ])
        return split
    
    def ij_final(self, pair):
        """given a pair of legs in the final state, assigns the i/j fks id
        NOTE: the j partons is always put before the i one"""
        #if a massless bosonic particle is in the pair, it is i
        #else by convention the anti-particle is labeled i
        #the order of the splitting is [ j, i]
        if len(pair) ==2:
            for i in range(len(pair)):
                set =0
                if (pair[i]['massless'] and pair[i]['spin'] %2 ==1) \
                 or (pair[i]['color'] == -3 and pair[1-i]['color'] == 3) \
                 and not set:
                    pair[i]['fks'] = 'i'
                    pair[1-i]['fks'] = 'j'
                    #check that first j then i
                    if i < 1-i:
                        pair.reverse()
                    set =1
                        


    def combine_ij(self, i, j):  #test written
        """checks whether partons i and j can be combined together and if so 
        combines them into ij"""
        part_i = self.real_proc['model'].get('particle_dict')[i['id']]
        part_j = self.real_proc['model'].get('particle_dict')[j['id']]
#        print "PART I", part_i
        ij = None
        num = copy.copy(min(i['number'], j['number']))
        if part_i['color'] != 1 and part_j['color'] != 1 and i['state']:
            #check if we have a massless gluon
            if part_i['color'] == 8 and part_i['mass'].lower() == 'zero':
                #ij should be the same as j
                #ij = copy.deepcopy(j)
                ij = MG.Leg(j)
                ij['number'] = num
            
            #check if we have two color triplets -> quarks
            elif part_i['color'] == 3 and part_j['color'] == 3 and part_i['mass'].lower() =='zero':
                #both final state -> i is anti-j
                if j['state']: 
                    if part_i['pdg_code'] == part_j['pdg_code'] and not part_i['is_part'] and part_j['is_part']:
                      #  print "PDG ", part_i['pdg_code'] , part_j['pdg_code'], i, j
                        #ij is an outgoing gluon
                        ij = MG.Leg()
                        ij['id'] = 21
                        ij['number'] = num 
                        ij['state'] = True
                        ij['from_group'] = True
                   #     { 'id' : 21, 'number' : 0, 'state' : True, 'from_group' : True }
                
                else:
                #initial and final state ->i is j
                    if part_i['pdg_code'] == part_j['pdg_code'] and (part_i['is_part'] == part_j['is_part']):
                        #ij is an outgoing gluon
                        ij = MG.Leg()
                        ij['id'] = 21
                        ij['number'] = num 
                        ij['state'] = False
                        ij['from_group'] = True
           
            # initial gluon and final massless quark
            elif part_i['color'] == 3 and part_i['mass'].lower() == 'zero' \
                and part_j['color'] == 8 and not j['state']:
                #ij is the same as i but crossed to the initial state
                #ij = copy.deepcopy(i)
                ij = MG.Leg(i)                
                ij['id'] = - ij['id']
                ij['state'] = False
        
        return ij
 
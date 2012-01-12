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

"""Definitions of the objects needed both for MadFKS from real 
and MadFKS from born"""

import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import copy
import logging
import array
import fractions
    
    
    
class FKSProcessError(Exception):
    """Exception for MadFKS"""
    pass


def find_orders(amp): #test_written
    """take an amplitude as input, and returns a dictionary with the
    order of the couplings"""
    assert isinstance(amp, diagram_generation.Amplitude)
    orders = {}
    for diag in amp.get('diagrams'):
        for order, value in diag.get('orders').items():
            try:
                orders[order] = max(orders[order], value)
            except KeyError:
                orders[order] = value
    return orders

def find_splittings(leg, model, dict, pert='QCD'): #test written
    """find the possible splittings corresponding to leg"""
    if dict == {}:
        dict = find_pert_particles_interactions(model, pert)
    splittings = []
#check that the leg is a qcd leg

    if leg.get('id') in dict['pert_particles']:
        part = model.get('particle_dict')[leg.get('id')]
        antipart = model.get('particle_dict')[part.get_anti_pdg_code()]
        for ii in dict['interactions']:
#check which interactions contain leg and at least one soft particles:
            parts = copy.deepcopy(ii['particles'])
            nsoft = 0
            if part in parts:
                #pops the ANTI-particle of part from the interaction
                parts.pop(parts.index(antipart))
                for p in parts:
                    if p.get_pdg_code() in dict['soft_particles']:
                        nsoft += 1
                if nsoft >=1:
                    splittings.extend(split_leg(leg, parts, model))
    return splittings

def split_leg(leg, parts, model): #test written
    """splits the leg into parts, and returns the two new legs"""
    #for an outgoing leg take the antiparticles
    split = []
    #for a final state particle one can have only a splitting
    if leg['state'] :
        split.append([])
        for part in parts:
            split[-1].append(to_fks_leg({'state' : True, \
                                 'id' : part.get_pdg_code()},model))
            ij_final(split[-1])
    #while for an initial state particle one can have two splittings 
    # if the two partons are different
    else:
        if parts[0] != parts[1]:
            for part in parts:
                cparts = copy.deepcopy(parts)
                split.append([\
                          to_fks_leg({'state' : False, 
                                  'id' : cparts.pop(cparts.index(part)).get_pdg_code(),
                                  'fks' : 'j'}, model),
                          to_fks_leg({'state' : True,
                                  'id' : cparts[0].get_anti_pdg_code(),
                                  'fks' : 'i'}, model)\
                          ])
        else:
            split.append([\
                            to_fks_leg({'state' : False, 
                                  'id' : parts[0].get_pdg_code(),
                                  'fks' : 'j'}, model),
                            to_fks_leg({'state' : True, 
                                  'id' : parts[1].get_anti_pdg_code(),
                                  'fks' : 'i'}, model)])
    return split

def ij_final(pair):
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

def insert_legs(leglist_orig, leg, split):
    """returns a new leglist with leg splitted into split."""
    leglist = copy.deepcopy(leglist_orig)         
    #find the position of the first final state leg
    for i in range(len(leglist)):
        if leglist[-i-1].get('state'):
            firstfinal = len(leglist) -i -1
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
            
    leglist.sort()
    for i, leg in enumerate(leglist):
        leg['number'] = i+1        
    return leglist 


def combine_ij( i, j, model, dict, pert='QCD'): #test written
    """checks whether FKSlegs i and j can be combined together in the given model
    and with given perturbation order and if so combines them into ij. 
    If dict is empty it is initialized with find_pert_particles_interactions"""
    if dict == {}:
        dict = find_pert_particles_interactions(model, pert)
    ij = []
    num = copy.copy(min(i.get('number'), j.get('number')))
    
    # we do not want j being a massiless vector unless also i is or j is initial
    not_double_counting = (j.get('spin') == 3 and j.get('massless') and 
                           i.get('spin') == 3 and i.get('massless')) or \
                           j.get('spin') != 3 or not j.get('massless') or \
                           not j.get('state')

    #if i and j are a final state particle and antiparticle pair,
    # then we want i to be antipart and j to be 
    if j.get('state') and j.get('id') == - i.get('id'):  
        not_double_counting = not_double_counting and j.get('id') >0
                          
    if i.get('id') in dict['soft_particles'] and \
       j.get('id') in dict['pert_particles'] and i.get('state') \
       and not_double_counting:
        for int in dict['interactions']:
            parts= copy.copy(int['particles'])
                #remove i
            try:
                parts.remove(model.get('particle_dict')[i.get('id')])
            except ValueError:
                continue

            #remove j if final state, anti j if initial state

            if j.get('state'):
                j_id = j.get('id')
            else:
                j_id = model.get('particle_dict')[j.get('id')].get_anti_pdg_code()
            try:
                parts.remove(model.get('particle_dict')[j_id])
            except ValueError:
                continue
            
            #ij is what remains if j is initial, the anti of if j is final
            if j.get('state'):
                ij.append(MG.Leg({
                    'id': parts[0].get_anti_pdg_code(),
                    'state': True,
                    'number': num}))
            else:
                ij.append(MG.Leg({
                    'id': parts[0].get_pdg_code(),
                    'state': False,
                    'number': num}))
    return to_fks_legs(ij, model)       


def find_pert_particles_interactions(model, pert_order = 'QCD'): #test written
    """given a model and pert_order, returns a dictionary with as entries:
    --interactions : the interactions of order pert_order
    --pert_particles : pdgs of particles taking part to interactions
    --soft_particles : pdgs of massless particles in pert_particles"""
    qcd_inter = MG.InteractionList()
    pert_parts = []
    soft_parts = []
    for i, ii in model.get('interaction_dict').items():
        # i want interections of pert_order: 1 (from LO to NLO), 
        # without any other orders
        if ii.get('orders') =={pert_order:1} and len(ii['particles']) ==3 :
            masslist = [p.get('mass').lower() for p in ii['particles'] ]
                # check that there is at least a massless particle, and that the 
                # remaining ones have the same mass 
                # (otherwise the real emission final state will not be degenerate
                # with the born one
            masslist.remove('zero')
            if len(set(masslist)) == 1:
                qcd_inter.append(ii)
                for pp in ii['particles']:
                    pert_parts.append(pp.get_pdg_code())
                    if pp['mass'].lower() == 'zero':
                        soft_parts.append(pp.get_pdg_code())

    return {'interactions' : sorted(qcd_inter), 
            'pert_particles': sorted(set(pert_parts)),
            'soft_particles': sorted(set(soft_parts))}    


def insert_color_links(col_basis, col_obj, links): #test written
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
        # and extend-> product the color strings
            
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
                dict[k].product(link['string'])
            this_col_obj.append(dict)
        basis_link = color_amp.ColorBasis()
        for ind, dict in enumerate(this_col_obj):
            basis_link.update_color_basis(dict, ind)
   
        this['link_basis'] = basis_link
        this['link_matrix'] = color_amp.ColorMatrix(col_basis,basis_link)               
        result.append(this)
    basis_orig = color_amp.ColorBasis()
    for ind, dict in enumerate(col_obj):
            basis_orig.update_color_basis(dict, ind)
    
    for link in result:
        link['orig_basis'] = basis_orig
    return result



def find_color_links(leglist): #test written
    """finds all the possible color links between any two legs of the born"""

    color_links = []
    for leg1 in leglist:
        for leg2 in leglist:
            #legs must be colored and different, unless massive
                if (leg1.get('color') != 1 and leg2.get('color') != 1) \
                  and (leg1 != leg2 or not leg1.get('massless')):
                    col_dict = legs_to_color_link_string(leg1,leg2)
                    color_links.append({
                        'legs' : [leg1, leg2],
                        'string' : col_dict['string'],
                        'replacements' : col_dict['replacements']})
    return color_links
             

def legs_to_color_link_string(leg1, leg2): #test written, all cases
    """given two FKSlegs, returns a dictionary containing:
    --string: the color link between the two particles, to be appended to
        the old color string
        extra minus or 1/2 factor are included as it was done in MadDipole
    --replacements: a pair of lists containing the replacements of the color 
        indices in the old string to match the link """
    #the second-to-last index of the t is the triplet,
    # the last is the anti-triplet

    legs = FKSLegList([leg1, leg2]) 
    dict={}
    min_index = -3000
    iglu = min_index*2
    string = color_algebra.ColorString()
    replacements = []
    if leg1 != leg2:
        for leg in legs:
            min_index -= 1
            num = leg.get('number')
            replacements.append([num, min_index])
            icol =1
            if not leg.get('state'):
                icol =-1
            if leg.get('color') * icol == 3:
                string.product(color_algebra.ColorString([
                               color_algebra.T(iglu, num, min_index)]))
                string.coeff = string.coeff * (-1)
            elif leg.get('color') * icol == -3:
                string.product(color_algebra.ColorString([
                               color_algebra.T(iglu, min_index, num)]))
            elif leg.get('color') == 8:
                string.product(color_algebra.ColorString(init_list = [
                               color_algebra.f(min_index,iglu,num)], 
                               is_imaginary =True))

                
#                if not leg.get('state'):
#                    string.coeff = string.coeff* (-1)
#        if leg1.get('color') == 8 and leg2.get('color') == 8:
#            string.coeff = string.coeff *(-1)
    else:
        icol =1
        if not leg1.get('state'):
            icol =-1
        num = leg1.get('number')
        replacements.append([num, min_index -1])
        if leg1.get('color') * icol == 3:
            string = color_algebra.ColorString(
                      [ color_algebra.T(iglu, iglu, num, min_index -1)
                      ])
        elif leg1.get('color') * icol == -3:
            string = color_algebra.ColorString(
                      [ color_algebra.T(iglu, iglu, min_index-1, num)
                      ])
        string.coeff = string.coeff * fractions.Fraction(1,2) 
    dict['replacements'] = replacements
    dict['string'] = string
      
    return dict

def sort_proc(process):
    """given a process, returns the same process but with sorted fkslegs"""
    leglist = to_fks_legs(process.get('legs'), process.get('model'))
    leglist.sort()
    for n, leg in enumerate(leglist):
        leg['number'] = n+1
    process['legs'] = leglist

    return process

def to_leg(fksleg):
    """Given a FKSLeg, returns the original Leg"""
    leg = MG.Leg( \
        {'id': fksleg.get('id'),
         'number': fksleg.get('number'),
         'state': fksleg.get('state'),
         'from_group': fksleg.get('from_group'),
#         'onshell': fksleg.get('onshell') 
          } )
    return leg

def to_legs(fkslegs):
    """Given a FKSLegList, returns the corresponding LegList"""
    leglist = MG.LegList()
    for leg in fkslegs:
        leglist.append(to_leg(leg))
    return leglist


def to_fks_leg(leg, model): #test written
    """given a leg or a dict with leg properties, 
    adds color, spin and massless entries, according to model"""
    fksleg =FKSLeg(leg)
    part = model.get('particle_dict')[leg['id']]
    fksleg['color'] = part.get_color()
    fksleg['massless'] = part['mass'].lower() == 'zero'
    fksleg['spin'] = part.get('spin')      
    return fksleg

    
def to_fks_legs(leglist, model): #test written
    """given leglist, sets color and massless entries according to the model 
    variable.
    return a FKSLeglist"""
    fkslegs = FKSLegList()     
    for leg in leglist:
        fkslegs.append(to_fks_leg(leg, model))
    return fkslegs     


class FKSLegList(MG.LegList):
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
            else:
        #if same color, put massive particles first
                if (self.get('massless') or other.get('massless')) and \
                  not (self.get('massless') and other.get('massless')):
                    return other.get('massless')
                else:
#3                    if (self.get('id') != other.get('id')):
##                        return self.get('id') < other.get('id')
##                    else:
##                        return self.get('number') < other.get('number')
                    return self.get('number') < other.get('number')
        return True
         


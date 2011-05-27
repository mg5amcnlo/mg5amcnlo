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



def combine_ij( i, j, model, dict, pert='QCD'): #test written
    """checks whether FKSlegs i and j can be combined together in the given model
    and with given perturbation order and if so combines them into ij. 
    If dict is empty it is initialized with find_pert_particles_interactions"""
    if dict == {}:
        dict = find_pert_particles_interactions(model, pert)
    ij = []
    num = copy.copy(min(i.get('number'), j.get('number')))

    
    # we do not want j being a massiless vector unless also i is
    not_double_counting = (j.get('spin') == 3 and j.get('massless') and 
                           i.get('spin') == 3 and i.get('massless')) or \
                          j.get('spin') != 3 or not j.get('massless')

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
                    color_links.append({
                        'legs' : [leg1, leg2],
                        'string' : legs_to_color_link_string(leg1, leg2)['string'],
                        'replacements' : legs_to_color_link_string(leg1, leg2)['replacements']})
    return color_links
             

def legs_to_color_link_string(leg1, leg2): #test written, all cases
    """returns a dictionary containing:
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
 
    
#    def __lt__(self, other):
#        #two initial state legs are sorted by their number:
#        if (not self.get('state') and not other.get('state')):
#            return self.get('number') < other.get('number')
#        
#        #an initial state leg comes before a final state leg
#        if (self.get('state') or other.get('state')) and \
#          not (self.get('state') and other.get('state')):
#            return other.get('state')
#        
#        #two final state particles are ordered by increasing color
#        elif self.get('state') and other.get('state'):
#            if abs(self.get('color')) != abs(other.get('color')):
#                return abs(self.get('color')) < abs(other.get('color'))
#        #particles of the same color are ordered according to the pdg code
#            else:
#                if abs(self.get('id')) != abs(other.get('id')):
#                    return abs(self.get('id')) < abs(other.get('id'))
#                elif self.get('id') != other.get('id') :
#        #for the same flavour qqbar pair, first take the quark 
#                    return self.get('id') > other.get('id')
#        # i fks > j fks > n fks        
#                else: 
#                    return not self.get('fks') < other.get('fks') 
                    
         
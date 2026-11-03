################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Definitions of the objects needed both for MadFKS from real 
and MadFKS from born"""

from __future__ import absolute_import
import madgraph
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod
from operator import itemgetter
import copy
import logging
import array
import fractions
if madgraph.ordering:
    set = misc.OrderedSet    
    
class FKSProcessError(Exception):
    """Exception for MadFKS"""
    pass


class FKSDiagramTag(diagram_generation.DiagramTag): #test written
    """Modified diagram tags to be used to link born and real configurations.
    """

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the default end link for a leg: ((id, number), number).
        Note that the number is not taken into account if tag comparison,
        but is used only to extract leg permutations.
        """
        return [((leg.get('id'), leg.get('number')), leg.get('number'))]


def get_qed_qcd_orders_from_weighted(nexternal, hierarchy, weighted):
    """computes the QED/QCD orders from the knowledge of the n of ext particles
    and of the weighted orders"""
    qed_w = hierarchy['QED']
    qcd_w = hierarchy['QCD']
    # n vertices = nexternal - 2 =QED + QCD
    # weighted = qed_w*QED + qcd_w*QCD

    QED = (weighted - qcd_w * (nexternal - 2) ) / (qed_w - qcd_w)
    QCD = (weighted - qed_w * QED) / qcd_w
    return int(QED), int(QCD)


def link_rb_configs(born_amp, real_amp, i, j, ij):
    """finds the real configurations that match the born ones, i.e.  for
    each born configuration, the real configuration that has the ij ->
    i j splitting.  i, j and ij are integers, and refer to the leg
    position in the real process (i, j) and in the born process (ij).
    """
    # find diagrams with 3 point functions and use them as configurations
    id_ij = born_amp['process']['legs'][ij - 1]['id']
    nlegs_b = len(born_amp['process']['legs'])
    nlegs_r = len(real_amp['process']['legs'])
    if nlegs_r - nlegs_b != 1:
        raise FKSProcessError('Inconsistent number of born and real legs: %d %d' % (nlegs_b, nlegs_r))

    #find the mapping of real legs onto the born ones
    shift_dict = {}
    for ir in range(1, nlegs_r + 1):
        shift = 0
        if ir > j:
            shift += 1
        if ir > i:
            shift += 1
        if ir > ij and ij <= max(i,j):
            shift -= 1
        shift_dict[ir] = ir - shift

# now find the configurations
    minvert = min([max([len(vert.get('legs')) \
                        for vert in diag.get('vertices')]) \
                        for diag in born_amp.get('diagrams')])

    born_confs = []
    real_confs = []

    k=0
    for diag in born_amp.get('diagrams'):
        if any([len(vert.get('legs')) > minvert for vert in
                diag.get('vertices')]):
            continue
        else:
            born_confs.append({'number' : k, 'diagram' : diag})
            k=k+1

    k=0
    for diag in real_amp.get('diagrams'):
        if any([len(vert.get('legs')) > minvert \
                for vert in diag.get('vertices')]):
            continue
        else:
            real_confs.append({'number': k, 'diagram': diag})
            k=k+1

    good_diags = []

    # find the real diagrams that have i and j attached to the same vertex
    # check that the order of the interaction is QCD
    # check also that the id of the third leg is id_ij
    real_confs_new = copy.deepcopy(real_confs)
    for diag in real_confs_new:
        for vert in diag['diagram'].get('vertices'):
            vert_legs = [l.get('number') for l in vert.get('legs')]
            vert_ids = [l.get('id') for l in vert.get('legs')]
            if (i in vert_legs and not j in vert_legs) or \
               (j in vert_legs and not i in vert_legs):
                   break

            if i in vert_legs and j in vert_legs:
                vert_ids.remove(vert_ids[vert_legs.index(i)])
                vert_legs.remove(i)
                vert_ids.remove(vert_ids[vert_legs.index(j)])
                vert_legs.remove(j)
                last_leg = vert_legs[0]
                #check absolute value in order not to worry about 
                #incoming/outgoing particles
                if abs(vert_ids[0]) == abs(id_ij):
                    diag['diagram']['vertices'].remove(vert)
                    good_diags.append({'diagram': diag['diagram'], 
                                      'leg_ij': last_leg,
                                      'number': diag['number']})
                break #no need to continue once the vertex is found

    # now good_diags contains the real_confs which had the splitting, 
    #  with the vertex corresponding to the splitting removed

    # The legs in the born and real diags are ordered according to 
    #  the same criterion. Once we removed i-j, we have to re-label the
    #  real legs to match the born numbering.

    # now relabel the legs according to shift_dict
    # replace from lower to higher leg number, in order not to 
    # overwrite
    for ir in range(1, nlegs_r + 1):
        for good_diag in good_diags:
            for vert in good_diag['diagram'].get('vertices'):
                for l in vert.get('legs'):
                    if l.get('number') == ir:
                        l.set('number', shift_dict[l.get('number')])

    # this is to handle cases in which only one diagrams has to be linked
    if len(good_diags) == 1 and len(born_confs) == 1:
        return [{'real_conf': good_diags[0]['number'],
                          'born_conf': born_confs[0]['number']}]

    # this is a rather ugly fix for some problems with 2->1 processes
    # count the occurrence of the leg numbers:
    if nlegs_b ==3:
        for diag in good_diags:
            counts = []
            for il in range(nlegs_b):
                counts.append([l['number'] for v in diag['diagram']['vertices'] for l in v['legs']].count(il+1))
            # normally all the occurrences should be odd (and !=0)
            # (each entry should be 1 for the external leg +2 for each internal one)
            even_list = [c / 2 * 2 == c for c in counts]
            if any(even_list):
                if not even_list.count(True) == 2:
                    raise FKSProcessError('Linking: Don\'t know what to do in this case')
                # find the leg number corresponding to the largest even count
                ilmax = counts.index(max([c for c in counts if even_list[counts.index(c)]]))
                ilmin = counts.index(min([c for c in counts if even_list[counts.index(c)]]))

                # finally do the replacement in the first occurrence of ilmax
                replaced = False
                for vert in diag['diagram']['vertices']:
                    for leg in vert['legs']:
                        if leg['number'] == ilmax + 1 and not replaced:
                            leg['number'] = ilmin + 1
                            replaced = True

    # now create the tags
    born_tags = [FKSDiagramTag(d['diagram'], 
                               born_amp.get('process').get('model')) \
                               for d in born_confs]


    # Dropping a duplicated tag has to drop its diagram with it: the two are
    # walked in lockstep below, real_tags.index giving the position popped out
    # of good_diags, so a tag left without its diagram misaligns everything
    # after it. Which of two equal tags survives is decided by the order they
    # come in, so keeping them apart also made the result order dependent.
    real_tags = []
    kept_diags = []
    for d in good_diags:
        tag = FKSDiagramTag(d['diagram'], real_amp.get('process').get('model'))
        if not tag in real_tags:
            real_tags.append(tag)
            kept_diags.append(d)
    good_diags = kept_diags

    # and compare them
    if len(born_tags) != len(real_tags):
        print('\n'.join([str(r) for r in real_tags]) + '\n')
        raise FKSProcessError('Cannot map born/real configurations between \
                %s and %s (i,j=%d,%d): not same number of configurations: %d %d' % \
                (born_amp.get('process').nice_string().replace('Process:',''), 
                 real_amp.get('process').nice_string().replace('Process:',''),
                               i,j,
                               len(born_tags),
                               len(real_tags)))
    
    links = []
    for ib, btag in enumerate(born_tags):
        try:
            ir = real_tags.index(btag)
            links.append({'real_conf': good_diags[ir]['number'],
                          'born_conf': born_confs[ib]['number']})
            real_tags.remove(btag)
            good_diags.pop(ir)
        except ValueError:
            print(real_tags, i, j, ij)
            print('\n'.join( d['diagram'].nice_string() for d in good_diags))
            raise FKSProcessError('Linking %s to %s: could not link born diagram %s' % \
                 (born_amp.get('process').nice_string().replace('Process:',''), 
                  real_amp.get('process').nice_string().replace('Process:',''),
                                  born_confs[ib]['diagram'].nice_string()) )

    return links



def find_orders(amp): #test_written
    """Takes an amplitude as input, and returns a dictionary with the
    orders of the couplings.
    """
    assert isinstance(amp, diagram_generation.Amplitude)
    orders = {}
    for diag in amp.get('diagrams'):
        for order, value in diag.get('orders').items():
            if value != 0 or order in list(amp['process']['orders'].keys()):
                try:
                    orders[order] = max(orders[order], value)
                except KeyError:
                    orders[order] = value
    return orders


def find_splittings(leg, model, dict, pert='QCD', include_init_leptons=True): #test written
    """Finds the possible splittings corresponding to leg
    """

    leptons = model.get_lepton_pdgs()

    if dict == {}:
        dict = find_pert_particles_interactions(model, pert)
    splittings = []

    if leg.get('id') in dict['pert_particles']:
        part = model.get('particle_dict')[leg.get('id')]
        antipart = model.get('particle_dict')[part.get_anti_pdg_code()]
        for ii in dict['interactions']:
#check which interactions contain leg and at least one soft particles:
            parts = copy.deepcopy(ii['particles'])
            nsoft = 0
            if part in parts:
                #try to pop the ANTI-particle of part from the interaction
                #if not there, pop part
                try:
                    parts.pop(parts.index(antipart))
                except ValueError:
                    parts.pop(parts.index(part))
                for p in parts:
                    if p.get_pdg_code() in dict['soft_particles']:
                        nsoft += 1
                if nsoft >= 1:
                    for split in split_leg(leg, parts, model):
                        # check if a final-state leg is tagged, that
                        # the same particles appear also in the two daughters
                        if 'is_tagged' in leg.keys() and leg['is_tagged'] and \
                           leg['state'] and \
                           leg['id'] != split[0]['id'] and \
                           leg['id'] != split[1]['id']:
                            continue

                        # UPC: only photon -> f fbar initial splitting is allowed
                        if 'is_tagged' in leg.keys() and leg['is_tagged'] and \
                            not leg['state'] and leg['id'] != 22:
                            if (split[0]['id'] == 22 and split[0]['state']):
                                continue
                            if (split[1]['id'] == 22 and split[1]['state']):
                                continue

                        # add the splitting, but check if there is 
                        # an initial-state lepton if the flag
                        # include_init_leptons is False
                        if include_init_leptons or \
                           not (any([l['id'] in leptons for l in split if not l['state']])):
                            splittings.append(split)
    return splittings


def find_mothers(leg1, leg2, model, dict={}, pert='', mom_mass=''):
    """Find the possible mothers of leg1, leg2.
    If mom_mass is passed, only the mothers with mom_mass are returned
    """
    if pert:
        if dict == {}:
            dict = find_pert_particles_interactions(model, pert)
    interactions = dict['interactions']
    mothers = []

    for inte in interactions:
        # loop over interactions which contain leg1 and leg2
        # and add the third particle to mothers
        pdgs = [p.get_pdg_code() for p in inte['particles']]
        try:
            for l in [leg1, leg2]:
                if not l['state']:
                    pdgs.remove(l['id'])
                else:
                    pdgs.remove(model.get('particle_dict')[l['id']].get_anti_pdg_code())
        except ValueError:
            continue
        if mom_mass and \
           mom_mass.lower() == model.get('particle_dict')[pdgs[0]]['mass'].lower():
            mothers.append(pdgs[0])

    return mothers


def carry_polarization(mother, split): #test written
    """Carries the polarization of the born leg ij onto the daughter j of one
    splitting (split is ordered [j, i]).

    In the singular limit the real's leg j *is* the born's leg ij: the two
    matrix elements are subtracted against each other, so they have to fix the
    same helicity state or the subtraction is comparing a fixed-helicity Born
    with a helicity-summed real. This is only meaningful when j carries the
    same identity as the mother, which for a massive QCD emitter in the FINAL
    state is always the case (the only splitting is t -> t g).

    Two families of mother do not satisfy it, and both are refused at
    generation time by the polarization guard in
    madgraph_interface.CheckValidForCmd.check_process_format. This function
    raises rather than silently dropping the polarization, so that it stays a
    backstop if that guard is ever relaxed:
      - a MASSLESS coloured final-state mother. g -> q q~ turns the polarized
        gluon into a quark pair, so there is no leg with the mother's identity
        to transport the projection to.
      - ANY coloured INITIAL-state mother, whatever its mass. The initial-state
        splitting is read backwards, g -> q(-> born) q~: the polarized quark
        that enters the born is an *internal* line of the real emission, so
        there is no external leg to carry the projection at all. The raise is
        all-or-nothing per leg: the same initial-state quark also has the
        well-defined q -> q g splitting, but split_leg is called for every
        splitting of the leg and the backward one raises, aborting the whole
        call, so the initial state is refused wholesale rather than channel
        by channel.
    """
    pol = mother.get('polarization')
    if not pol:
        return
    j_leg = split[0]
    if j_leg['id'] != mother['id'] or j_leg['state'] != mother['state']:
        raise FKSProcessError(
            'Cannot carry the polarization %s of the born leg with id %d '
            'through the splitting into ids %s: the FKS emitter j is a '
            'different particle, so the born and the real would not fix the '
            'same helicity state.' % \
            (pol, mother['id'], [l['id'] for l in split]))
    # ... and it has to be j and not i: for g -> g g in the final state both
    # daughters have the mother's identity and there is nothing to pick
    # between them away from the soft limit. (Only split[1] is inspected;
    # find_splittings can hand split_leg a 3-particle parts list -- the
    # 4-gluon vertex -- whose split[2] would go unchecked. That is gluon-only,
    # so a polarized mother reaching it is already refused by the guard.)
    if len(split) > 1 and split[1]['id'] == mother['id'] and \
       split[1]['state'] == mother['state']:
        raise FKSProcessError(
            'Cannot carry the polarization %s of the born leg with id %d '
            'through the splitting into ids %s: both daughters have the '
            'identity of the mother, so which one inherits the polarization '
            'is not defined.' % \
            (pol, mother['id'], [l['id'] for l in split]))
    j_leg['polarization'] = copy.copy(pol)


def split_leg(leg, parts, model): #test written
    """Splits the leg into parts, and returns the two new legs.
    """
    #for an outgoing leg take the antiparticles
    split = []
    #for a final state particle one can have only a splitting
    if leg['state'] :
        split.append([])
        for part in parts:
            split[-1].append(to_fks_leg({'state': True, \
                                 'id': part.get_pdg_code()},model))
            ij_final(split[-1])
    #while for an initial state particle one can have two splittings 
    # if the two partons are different
    else:
        if parts[0] != parts[1]:
            for part in parts:
                cparts = copy.deepcopy(parts)
                split.append([\
                          to_fks_leg({'state': False, 
                                  'id': cparts.pop(cparts.index(part)).get_pdg_code(),
                                  'fks': 'j'}, model),
                          to_fks_leg({'state': True,
                                  'id': cparts[0].get_anti_pdg_code(),
                                  'fks': 'i'}, model)\
                          ])
        else:
            split.append([\
                            to_fks_leg({'state': False, 
                                  'id': parts[0].get_pdg_code(),
                                  'fks': 'j'}, model),
                            to_fks_leg({'state': True, 
                                  'id': parts[1].get_anti_pdg_code(),
                                  'fks': 'i'}, model)])
    # the born leg ij is replaced by the daughter j (see insert_legs), so any
    # polarization it carries has to follow it there
    for s in split:
        carry_polarization(leg, s)
    return split


def ij_final(pair):
    """given a pair of legs in the final state, assigns the i/j fks id
    NOTE: the j partons is always put before the i one
    """
    #if a massless bosonic particle is in the pair, it is i
    #else by convention the anti-particle is labeled i
    #the order of the splitting is [ j, i]
    if len(pair) == 2:
        for i in range(len(pair)):
            set = 0
            if (pair[i]['massless'] and pair[i]['self_antipart']) or \
             (not pair[i]['is_part'] and pair[1-i]['is_part'] and\
              (pair[i]['spin']+pair[1-i]['spin'])%2==0) and not set:
                pair[i]['fks'] = 'i'
                pair[1-i]['fks'] = 'j'
                #check that first j then i
                if i < 1 - i:
                    pair.reverse()
                set = 1

def insert_legs(leglist_orig, leg, split,pert='QCD'):
    """Returns a new leglist with leg split into split.
    The convention is to remove leg ij, replace it with leg j, and put
    i at the end of the group of legs with the same color(charge) representation
    """
    if pert =='QCD':
        color = 'color'
    elif pert == 'QED':
        color = 'charge'
    else:
        raise FKSProcessError("Only QCD or QED is allowed not  %s" % pert)
    # the deepcopy statement is crucial
    leglist = FKSLegList(copy.deepcopy(leglist_orig))         
    #find the position of the first final state leg
    for i in range(len(leglist)):
        if leglist[-i - 1].get('state'):
            firstfinal = len(leglist) - i - 1
    # replace leg with leg_j  (split[0])
    leglist[leglist.index(leg)] = split[0]
    # and find where to insert i  (split[1])
    col_maxindex = {}
    mass_col_maxindex = {}
    for col in set([l[color] for l in leglist[firstfinal:] if l['massless']]):
        col_maxindex[col] = max([0] + [leglist.index(l) for l in leglist[firstfinal:]\
                                        if l[color] == col and l['massless']])
    for col in set([abs(l[color]) for l in leglist[firstfinal:] if not l['massless']]):
        mass_col_maxindex[col] = max([0] + [leglist.index(l) for l in leglist[firstfinal:]\
                                             if abs(l[color]) == col and not l['massless']])
    #no need to keep info on particles with color > i
    if pert == 'QCD':
        for col in copy.copy(list(col_maxindex.keys())):
            if abs(col) > abs(split[1][color]):
                del col_maxindex[col]
###        for col in copy.copy(mass_col_maxindex.keys()):
###            if abs(col) > abs(split[1][color]):
###                del mass_col_maxindex[col]
    #also remove antiquarks if i is a quark or a fermion
    if split[1]['is_part'] and not split[1]['self_antipart']:
    # In old MADFKS5, the line below was used instead. It is however equivalent in principle.
    # We can remove this comment and the line below altogether after validation and complete
    # merge of the EW branch in aMC@NLO trunk.
    #if split[1][color] > 0:
        try:
            del col_maxindex[-split[1][color]]
        except KeyError:
            pass
    #so now the maximum of the max_col entries should be the position to insert leg i
    leglist.insert(max(list(col_maxindex.values()) + list(mass_col_maxindex.values()) + [firstfinal - 1] ) + 1, split[1])
###    leglist.insert(max(col_maxindex.values() + [firstfinal - 1] ) + 1, split[1])
#    for sleg in split:            
#        leglist.insert(i, sleg)
#        #keep track of the number for initial state legs
#        #if not sleg.get('state') and not leg.get('state'):
#        leglist[i]['number'] = leg['number']
#        i += 1
#        if i < firstfinal:
#            i = firstfinal
#            
#    leglist.sort()
    for i, leg in enumerate(leglist):
        leg['number'] = i + 1        
    return leglist 


def combine_ij( i, j, model, dict, pert='QCD'): #test written
    """checks whether FKSlegs i and j can be combined together in the given model
    and with given perturbation order and if so combines them into ij. 
    If dict is empty it is initialized with find_pert_particles_interactions
    """
    if dict == {}:
        dict = find_pert_particles_interactions(model, pert)
    ij = []
    num = copy.copy(min(i.get('number'), j.get('number')))

    # we do not want j being a massless vector unless also i is or j is initial
    not_double_counting = (j.get('spin') == 3 and j.get('massless') and 
                           i.get('spin') == 3 and i.get('massless')) or \
                           j.get('spin') != 3 or not j.get('massless') or \
                           not j.get('state')

    #if i and j are a final state particle and antiparticle pair,
    # then we want i to be antipart and j to be 
    if j.get('state') and j.get('id') == - i.get('id'):  
        not_double_counting = not_double_counting and j.get('id') >0

    if i.get('id') in dict['soft_particles'] and \
       j.get('id') in dict['pert_particles'] and \
       i.get('state') and not_double_counting:
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
    --soft_particles : pdgs of massless particles in pert_particles
    """
    #ghost_list = [82, -82] # make sure ghost_list is non-empty
    ghost_list = []
    ghost_list += [ p.get_pdg_code() for p in model.get('particles') 
                                            if p.get('ghost') or p.get('goldstone')]

    qcd_inter = MG.InteractionList()
    pert_parts = []
    soft_parts = []
    for i, ii in model.get('interaction_dict').items():
        # i want interactions of pert_order: 1 (from LO to NLO),
        # without any other orders
        # and of "base" type
        if ii.get('type') != 'base': continue

        if ii.get('orders') == {pert_order:1} and len(ii['particles']) == 3 :
            masslist = [p.get('mass').lower() for p in ii['particles']]
            
            # require that at least one particle be soft and of even spin for the interaction to be IR singular
            found_soft_even_spin_particle = False
            for p in ii['particles']:
                if p.get('mass').lower()=='zero':
                    if p.get('spin')%2==1:
                        found_soft_even_spin_particle = True
                        break
            if not found_soft_even_spin_particle:
                continue
            
            # check that there is at least a massless particle, and that the 
            # remaining ones have the same mass 
            # (otherwise the real emission final state will not be degenerate
            # with the born one
            try:
                masslist.remove('zero')
            except ValueError:
                continue
            if len(set(masslist)) == 1 and not \
                    any( [ p.get_pdg_code() in ghost_list or \
                           p.get_anti_pdg_code() in ghost_list for p in ii['particles']]) :
                qcd_inter.append(ii)
                for pp in ii['particles']:
                    pert_parts.append(pp.get_pdg_code())
                    if pp['mass'].lower() == 'zero':
                        soft_parts.append(pp.get_pdg_code())

    return {'interactions': sorted(qcd_inter, key=misc.cmp_to_key(misc.dict_cmp)),
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
            new_dict = dict(old_dict)
            for k, string in new_dict.items():
                new_dict[k] = string.create_copy()
                for col in new_dict[k]:
                    for ind in col:
                        for pair in link['replacements']:
                            if ind == pair[0]:
                                col[col.index(ind)] = pair[1]
                new_dict[k].product(link['string'])
            this_col_obj.append(new_dict)
        basis_link = color_amp.ColorBasis()
        for ind, new_dict in enumerate(this_col_obj):
            basis_link.update_color_basis(new_dict, ind)
   
        this['link_basis'] = basis_link
        this['link_matrix'] = color_amp.ColorMatrix(col_basis,basis_link)               
        result.append(this)
    basis_orig = color_amp.ColorBasis()
    for ind, new_dict in enumerate(col_obj):
            basis_orig.update_color_basis(new_dict, ind)
    
    for link in result:
        link['orig_basis'] = basis_orig
    return result



def born_basis_link_matrices(col_basis, links, Nc=3):
    """Colour-link matrices expressed in the Born colour basis itself.

    For each link (as returned by find_color_links) return the square matrix
        S_ij = sum_k M_ik L_kj
    such that the colour-linked Born is  sum_ij conj(JAMP_i) S_ij JAMP_j,
    with the very same JAMPs as the Born. M is the rectangular matrix between
    the Born basis and the link basis (insert_color_links' link_matrix) and L
    expresses each link-basis element in terms of the Born basis elements:
    inserting the link into the Born basis elements (used as "diagrams" with
    unit coefficient) gives JAMP_link_k = sum_j L_kj JAMP_j by linearity,
    which is exactly what the diagram-based link basis computes.

    Rows/columns follow sorted(col_basis), the ordering of the Born JAMPs and
    colour matrix. Entries are exact complex rationals, returned as
    (real, imag) pairs of Fractions in a dict {(i, j): (re, im)}."""

    born_keys = sorted(col_basis.keys())
    # the Born basis elements, each one a "diagram" with coefficient 1
    col_obj = []
    for key in born_keys:
        string = color_algebra.ColorString()
        string.from_immutable(key)
        col_obj.append({(0,): string})

    def cmul(a, b):
        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

    result = []
    for link in insert_color_links(col_basis, col_obj, links):
        link_keys = sorted(link['link_basis'].keys())
        # L[k][j]: coefficient of Born element j in link-basis element k
        L = []
        for key in link_keys:
            row = {}
            for entry in link['link_basis'][key]:
                j, coeff, is_imag, nc_power = entry[0], entry[2], entry[3], \
                                              entry[4]
                val = fractions.Fraction(coeff) * \
                      fractions.Fraction(Nc) ** nc_power
                val = (fractions.Fraction(0), val) if is_imag else \
                      (val, fractions.Fraction(0))
                old = row.get(j, (0, 0))
                row[j] = (old[0] + val[0], old[1] + val[1])
            L.append(row)
        S = {}
        for i in range(len(born_keys)):
            for k in range(len(link_keys)):
                m_val, m_imag = link['link_matrix'].col_matrix_fixed_Nc[(i, k)]
                m = (fractions.Fraction(0), m_val) if m_imag else \
                    (fractions.Fraction(m_val), fractions.Fraction(0))
                if m == (0, 0):
                    continue
                for j, l in L[k].items():
                    prod = cmul(m, l)
                    old = S.get((i, j), (0, 0))
                    S[(i, j)] = (old[0] + prod[0], old[1] + prod[1])
        result.append({'link': link['link'], 'matrix': S,
                       'ncolor': len(born_keys)})
    return result


def find_color_links(leglist, symm = False,pert = 'QCD'): #test written
    """Finds all the possible color(charge) links between any 
    two legs of the born.
    If symm is true, only half of the color links are generated, those
    for which leg1['number'] <= leg2['number']
    """
    if pert == 'QCD':
        color = 'color'
        zero = 1
    elif pert == 'QED':
        color = 'charge'
        zero = 0.
    else:
        raise FKSProcessError("Only QCD or QED is allowed not %s" % pert)
    color_links = []
    for leg1 in leglist:
        for leg2 in leglist:
            #legs must be colored(charged) and different, unless massive
                if (leg1.get(color) != zero and leg2.get(color) != zero) \
                  and (leg1 != leg2 or not leg1.get('massless')):
                    if not symm or leg1['number'] <= leg2['number']:
                        col_dict = legs_to_color_link_string(leg1,leg2,pert = pert)
                        color_links.append({
                            'legs': [leg1, leg2],
                            'string': col_dict['string'],
                            'replacements': col_dict['replacements']})

    return color_links
             

def legs_to_color_link_string(leg1, leg2, pert = 'QCD'): #test written, all cases
    """given two FKSlegs, returns a dictionary containing:
    --string: the color link between the two particles, to be appended to
        the old color string
        extra minus or 1/2 factor are included as it was done in MadDipole
    --replacements: a pair of lists containing the replacements of the color 
        indices in the old string to match the link 
    """
    #the second-to-last index of the t is the triplet,
    # the last is the anti-triplet

    legs = FKSLegList([leg1, leg2]) 
    dict = {}
    min_index = -3000
    iglu = min_index*2
    string = color_algebra.ColorString()
    replacements = []
    if pert == 'QCD':
        if leg1 != leg2:
            for leg in legs:
                min_index -= 1
                num = leg.get('number')
                replacements.append([num, min_index])
                icol = 1
                if not leg.get('state'):
                    icol = - 1
                if leg.get('color') * icol == 3:
                    string.product(color_algebra.ColorString([
                               color_algebra.T(iglu, num, min_index)]))
                    string.coeff = string.coeff * (-1)
                elif leg.get('color') * icol == - 3:
                    string.product(color_algebra.ColorString([
                               color_algebra.T(iglu, min_index, num)]))
                elif leg.get('color') == 8:
                    string.product(color_algebra.ColorString(init_list = [
                               color_algebra.f(min_index,iglu,num)], 
                               is_imaginary =True))

        else:
            icol = 1
            if not leg1.get('state'):
                icol = - 1
            num = leg1.get('number')
            replacements.append([num, min_index -1])
            if leg1.get('color') * icol == 3:
                string = color_algebra.ColorString(
                     [color_algebra.T(iglu, iglu, num, min_index -1)])
            elif leg1.get('color') * icol == - 3:
                string = color_algebra.ColorString(
                     [color_algebra.T(iglu, iglu, min_index-1, num)])
            elif leg1.get('color') == 8:
                string = color_algebra.ColorString(init_list = [
                               color_algebra.f(min_index-1,iglu,min_index)], 
                               is_imaginary =True)
                string.product(color_algebra.ColorString(init_list = [
                               color_algebra.f(min_index,iglu,num)], 
                               is_imaginary =True))
            string.coeff = string.coeff * fractions.Fraction(1, 2)

    elif pert == 'QED':
        for leg in legs:
            # make it a fraction
            string.coeff = string.coeff * fractions.Fraction(leg['charge']*3.)*\
            fractions.Fraction(1,3)            
    else:
        raise FKSProcessError("Only QCD or QED is allowed not %s"% pert)
    
    dict['replacements'] = replacements
    dict['string'] = string  
    return dict


def get_user_leg_order(process, pert = 'QCD'):
    """Return, for each position of the FKS-sorted leg list, the number the
    leg carried in the process as the user wrote it.

    sort_proc() both permutes and renumbers the legs, so after it has run the
    user's numbering is gone for good. me_frame in the run_card is expressed
    in the user's numbering, so this mapping is what lets the fortran get back
    to it; it is written out as frame_map_born in frame_info.inc.

    The returned list is 0-indexed by FKS position: entry i is the user number
    of the leg sitting at FKS position i+1.

    Call this *before* sort_proc. to_fks_legs() copies the legs, so this does
    not disturb the process it is given.
    """
    leglist = to_fks_legs(process.get('legs'), process.get('model'))
    leglist.sort(pert = pert)
    return [leg['number'] for leg in leglist]


def me_frame_characteristics(processes, density=False,
                             splitting_types=('QCD',)):
    """What the run_card me_frame may mean for these NLO processes, in the
    leg numbering of the process as the user wrote it (the numbering of
    me_frame). Returns (cm_legs, unsafe_legs, default_legs), sorted lists.

    default_legs: the default me_frame of the run_card. For a collision
    with an initial-state QCD splitting, the massive final-state legs: e.g.
    [3, 4] for p p > w+{0} w- [QCD], [3] for p p > w+{0} j [QCD] (the
    Collins-Soper frame of the W). Infrared safe and the same in every FKS
    sector, unlike the partonic c.m. (see cm_legs). Empty otherwise
    (e+ e-, decays), meaning the partonic c.m., [1, 2], is kept. It does
    not depend on whether real emission is computed, so that [LOonly=QCD]
    and [QCD] samples of a merged set share one frame.

    cm_legs: the legs whose rest frame stands for the partonic c.m., which
    is how me_frame = [1, 2] is read.
      - A decay: the decaying particle, [1].
      - A collision with no initial-state QCD splitting (e+ e-): [1, 2].
        Every FKS sector then hands the real emission its momenta in its
        own c.m., so the partonic c.m. is one well defined frame.
      - A collision with one: the Born final state. FKS hands the real of
        an initial-state sector its momenta in the c.m. of that sector's
        underlying Born, and that differs between sectors by a boost along
        z, so 'the frame the momenta come in' depends on the subtraction
        partition (+9.5% on p p > w+{0} j [QCD] when only fks_b changes).
        The real's own c.m. is no way out: it moves by a finite boost when
        an initial-state parton goes collinear, so it is not infrared safe.
        The rest frame of the Born final state is the partonic c.m. for
        the Born, the virtual and every counterterm, and for the real it
        is a function of its physical momenta alone.

    unsafe_legs: the legs whose momentum changes by a finite amount in an
    infrared limit, so that a frame containing them is not infrared safe
    once real emission is computed:
      - massless final-state legs charged under the perturbation (a jet for
        [QCD]): a collinear splitting shares their momentum;
      - initial-state legs of a collision charged under it (the partons of
        p p for [QCD]): the real emission and the reduced Born carry
        different momentum fractions even in the collinear limit.
    Never the decaying particle of a 1 -> n process, whatever its charge:
    FKS does not split it (fks_base.find_reals), and its momentum is the
    total momentum in every configuration. Nor a beam that is not charged
    under the perturbation (e+ e- with [QCD]). When cm_legs contains an
    unsafe leg, me_frame = [1, 2] has no infrared-safe NLO definition.

    splitting_types: the kinds of splitting the real emission was
    generated with (proc_characteristics), 'QCD' and/or 'QED'.

    All three are empty when no massive particle is polarised and no
    spin-density matrix is asked for (output --density, density=True): the
    matrix elements are then Lorentz invariant, and the frame does not
    matter.
    Legs of a multiparticle label count as coloured or massless if any of
    its particles is, and as massive if all of them are. If only some
    processes have an initial-state QCD splitting (a photon in the
    proton), the Born final state is used for all: it is the partonic c.m.
    of every Born too.
    """
    # what splits is what FKS generated splittings for (the run's
    # splitting_types), not perturbation_couplings: with the mixed expansion
    # e+ e- > t t~ [QCD] carries ['QCD', 'QED'] there
    def splits(p):
        """the particle radiates or splits under the perturbation"""
        return ('QCD' in splitting_types and p.get('color') != 1) or \
               ('QED' in splitting_types and (p.get('charge') != 0 or
                                              p.get('pdg_code') == 22))

    # the run_card default: one rule for LO and NLO, so that they agree
    default_legs = banner_mod.default_me_frame_for_process(
                        processes, density=density, radiates=splits)
    default_legs = default_legs if isinstance(default_legs, list) else []

    cm_legs = set()
    cm_legs_isr = set()
    unsafe_legs = set()
    for proc in processes:
        model = proc.get('model')
        legs = proc.get('legs')
        particles = []
        for leg in legs:
            ids = leg.get('ids') if 'ids' in leg else [leg.get('id')]
            particles.append([model.get_particle(i) for i in ids])

        if not density and not any(
                leg.get('polarization') and
                any(p.get('mass').lower() != 'zero' for p in parts)
                for leg, parts in zip(legs, particles)):
            continue

        initial = [n for n, leg in enumerate(legs, 1) if not leg.get('state')]
        final = [n for n, leg in enumerate(legs, 1) if leg.get('state')]
        # [LOonly=...]: no real emission, so nothing to be infrared safe for
        has_real = proc.get('NLO_mode') != 'LOonly'
        initial_radiates = any(splits(p) for n in initial
                               for p in particles[n - 1])

        if len(initial) == 2 and has_real and initial_radiates:
            cm_legs_isr.update(final)
        else:
            cm_legs.update(initial)

        if has_real:
            unsafe_legs.update(
                n for n in final if any(splits(p) and
                                        p.get('mass').lower() == 'zero'
                                        for p in particles[n - 1]))
            if len(initial) == 2:
                unsafe_legs.update(n for n in initial
                                   if any(splits(p) for p in particles[n - 1]))

    return (sorted(cm_legs_isr or cm_legs), sorted(unsafe_legs),
            default_legs)


def sort_proc(process,pert = 'QCD'):
    """Given a process, this function returns the same process 
    but with sorted FKSLegs.
    """
    leglist = to_fks_legs(process.get('legs'), process.get('model'))
    leglist.sort(pert = pert)
    for n, leg in enumerate(leglist):
        leg['number'] = n + 1
    process['legs'] = leglist
    # add this line to pass ./test_managers.py -p A test_check_ppzjj
    process['legs_with_decays']=MG.LegList()

    return process


def to_leg(fksleg):
    """Given a FKSLeg, returns the original Leg.
    """
    leg = MG.Leg( \
        {'id': fksleg.get('id'),
         'number': fksleg.get('number'),
         'state': fksleg.get('state'),
         'from_group': fksleg.get('from_group'),
         'polarization': fksleg.get('polarization')
          })
    return leg


def to_legs(fkslegs):
    """Given a FKSLegList, returns the corresponding LegList.
    """
    leglist = MG.LegList()
    for leg in fkslegs:
        leglist.append(to_leg(leg))
    return leglist


def to_fks_leg(leg, model): #test written
    """Given a leg or a dict with Leg entries, 
    adds color, spin and massless entries, according to model"""
    fksleg = FKSLeg(leg)
    part = model.get('particle_dict')[leg['id']]
    fksleg['color'] = part.get_color()
    fksleg['charge'] = part.get_charge()
    fksleg['massless'] = part['mass'].lower() == 'zero'
    fksleg['spin'] = part.get('spin')
    fksleg['is_part'] = part.get('is_part')
    fksleg['self_antipart'] = part.get('self_antipart')      
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


    def sort(self,pert='QCD'):
        """Sorting routine, sorting chosen to be optimal for madfks"""
        sorted_leglist = FKSLegList()
        #find initial state legs
        initial_legs = FKSLegList([l for l in copy.copy(self) if not l['state']])
        #find final state legs
        final_legs = FKSLegList([l for l in copy.copy(self) if l['state']])
        if len(initial_legs) == 1:
            sorted_leglist.extend(initial_legs)
        elif len(initial_legs) == 2:
            if initial_legs[0]['number'] > initial_legs[1]['number']:
                initial_legs.reverse()
            sorted_leglist.extend(initial_legs)
        else: 
            raise FKSProcessError('Too many initial legs')
        #find color representations
        # order according to spin and mass
        #find massive and massless legs 
        massive_legs = [l for l in final_legs if not l['massless']]
        massless_legs = [l for l in final_legs if l['massless']]

        for leglist in [massive_legs, massless_legs]:
            spins = sorted(set([abs(l['spin']) for l in leglist]))
            for spin in spins:
                spin_legs = FKSLegList([l for l in leglist if abs(l['spin']) == spin])
                init_pdg_legs = []
                if len(initial_legs) == 2:
                #put first legs which have the same abs(pdg) of the initial ones
                    for j in range(len(set([ abs(l['id']) for l in initial_legs]))):
                        pdg = abs(initial_legs[j]['id'])
                        init_pdg_legs = [l for l in spin_legs if abs(l['id']) == pdg]
                        if init_pdg_legs:
                            # sort in order to put first quarks then antiparticles,
                            #  and to put fks partons as n j i
                            init_pdg_legs.sort(key = itemgetter('id'), reverse=True)
                            sorted_leglist.extend(FKSLegList(init_pdg_legs))

                    init_pdgs = [ abs(l['id']) for l in initial_legs]
                    other_legs = [l for l in spin_legs if not abs(l['id']) in init_pdgs]
                    other_legs.sort(key = itemgetter('id'), reverse=True)
                    sorted_leglist.extend(FKSLegList(other_legs))
                else:
                    #llist.sort(key = itemgetter('id'), reverse=True)
                    sorted_leglist.extend(FKSLegList(spin_legs))

        for i, l in enumerate(sorted_leglist):
            self[i] = l


class FKSLeg(MG.Leg):
    """a class for FKS legs: it inherits from the usual leg class, with two
    extra keys in the dictionary: 
    -'fks', whose value can be 'i', 'j' or 'n' (for "normal" particles) 
    -'color', which gives the color of the leg
    -'charge', which gives the charge of the leg
    -'massless', boolean, true if leg is massless
    -'spin' which gives the spin of leg
    -'is_tagged', boolean, true if leg is tagged in the final state
    -'is_part', boolean, true if leg is an particle
    -'self_antipart', boolean, true if leg is an self-conjugated particle
    """

    def default_setup(self):
        """Default values for all properties"""
        super(FKSLeg, self).default_setup()

        self['fks'] = 'n'
        self['color'] = 0
        self['charge'] = 0.
        self['massless'] = True
        self['spin'] = 0
        self['is_tagged'] = False
        self['is_part'] = True
        self['self_antipart'] = False
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSLeg, self).get_sorted_keys()
        keys += ['fks', 'color','charge', 'massless', 'spin','is_tagged','is_part','self_antipart',]
        return keys

    
    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'fks':
            if not isinstance(value, str):
                raise self.PhysicsObjectError("%s is not a valid string for leg fks flag" \
                                                        % str(value))
        if name in ['color', 'spin']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError("%s is not a valid leg %s flag" % \
                                                 (str(value), name))
                                                 
        if name in ['massless','is_tagged','self_antipart','is_part']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError("%s is not a valid boolean for leg flag %s" % \
                                                                    (str(value), name))
        if name == 'charge':
            if not isinstance(value, float):
                raise self.PhysicsObjectError("%s is not a valid float for leg flag charge" \
                    % str(value))                                                           
        return super(FKSLeg,self).filter(name, value)

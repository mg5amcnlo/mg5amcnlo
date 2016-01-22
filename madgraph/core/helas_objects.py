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

"""Definitions of objects used to generate language-independent Helas
calls: HelasWavefunction, HelasAmplitude, HelasDiagram for the
generation of wavefunctions and amplitudes, HelasMatrixElement and
HelasMultiProcess for generation of complete matrix elements for
single and multiple processes; and HelasModel, which is the
language-independent base class for the language-specific classes for
writing Helas calls, found in the iolibs directory"""

import array
import copy
import collections
import logging
import itertools
import math

import aloha

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_color_amp as loop_color_amp
import madgraph.core.color_algebra as color
import madgraph.various.misc as misc

from madgraph import InvalidCmd, MadGraph5Error

#===============================================================================
# 
#===============================================================================

logger = logging.getLogger('madgraph.helas_objects')

#===============================================================================
# DiagramTag class to identify matrix elements
#===============================================================================

class IdentifyMETag(diagram_generation.DiagramTag):
    """DiagramTag daughter class to identify processes with identical
    matrix elements. Need to compare leg number, color, lorentz,
    coupling, state, spin, self_antipart, mass, width, color, decay
    and is_part.

    Note that we also need to check that the processes agree on
    has_mirror_process, process id, and
    identical_particle_factor. Don't allow combining decay chains.

    We also don't want to combine processes with different possibly
    onshell s-channel propagators (i.e., with non-zero width and
    onshell=True or None) since we want the right propagator written
    in the event file in the end. This is done by the vertex."""

    # dec_number is used to separate between decay chains.
    # This is needed since we don't want to merge different decays,
    # in order to get the right factor for identical/non-identical particles
    dec_number = 1
    
    @classmethod
    def create_tag(cls, amplitude, identical_particle_factor = 0):
        """Create a tag which identifies identical matrix elements"""
        process = amplitude.get('process')
        ninitial = process.get_ninitial()
        model = process.get('model')
        dc = 0
        if process.get('is_decay_chain'):
            dc = cls.dec_number
            cls.dec_number += 1
        if not identical_particle_factor:
            identical_particle_factor = process.identical_particle_factor()
        if process.get('perturbation_couplings') and \
                process.get('NLO_mode') not in ['virt', 'loop','noborn']:
            sorted_tags = sorted([IdentifyMETagFKS(d, model, ninitial) for d in \
                                      amplitude.get('diagrams')])
        elif process.get('NLO_mode')=='noborn':
            # For loop-induced processes, make sure to create the Tag based on
            # the contracted diagram
            sorted_tags = sorted([cls(d.get_contracted_loop_diagram(model,
             amplitude.get('structure_repository')), model, ninitial) for d in \
                                                     amplitude.get('diagrams')])
        else:
            sorted_tags = sorted([cls(d, model, ninitial) for d in \
                                                     amplitude.get('diagrams')])

        # Do not use this for loop diagrams as for now the HelasMultiProcess
        # always contain only exactly one loop amplitude.
        if sorted_tags and not isinstance(amplitude, \
                                         loop_diagram_generation.LoopAmplitude):
            # Need to keep track of relative permutations for all diagrams,
            # to make sure we indeed have different matrix elements,
            # and not just identical diagrams disregarding particle order.
            # However, identical particles should be treated symmetrically.
            exts = sorted_tags[0].get_external_numbers()            
            comp_dict = IdentifyMETag.prepare_comp_dict(process, exts)
            perms = [array.array('H',
                     sum([comp_dict[n] for n in p.get_external_numbers()], []))
                     for p in sorted_tags[1:]]
        else:
            perms = []

        return [amplitude.get('has_mirror_process'),
                process.get('id'),
                process.get('is_decay_chain'),
                identical_particle_factor,
                dc,
                perms,
                sorted_tags]

    @staticmethod
    def prepare_comp_dict(process, numbers):
        """Prepare a dictionary from leg number to [positions] in such
           a way that identical particles are treated in a symmetric way"""

        # Crate dictionary from leg number to position
        start_perm_dict = dict([(n,i) for (i,n) in enumerate(numbers)])

        # Ids for process legs            
        legs = [l.get('id') for l in sorted(process.get_legs_with_decays())]
        if process.get('is_decay_chain'):
            legs.insert(0,process.get('legs')[0].get('id'))
        ninitial = len([l for l in process.get('legs') if \
                                not l.get('state')])
        # Dictionary from leg id to position for final-state legs
        id_num_dict = {}
        for n in start_perm_dict.keys():
            if n > ninitial:
                id_num_dict.setdefault(legs[n-1], []).append(\
                    start_perm_dict[n])
        # Make each entry in start_perm_dict independent of position of
        # identical particles by including all positions
        for n in start_perm_dict.keys():
            if n <= ninitial:
                # Just turn it into a list
                start_perm_dict[n] = [start_perm_dict[n]]
            else:
                # Turn into list of positions for identical particles
                start_perm_dict[n] = sorted(id_num_dict[legs[n-1]])

        return start_perm_dict

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the end link for a leg needed to identify matrix
        elements: ((leg numer, state, spin, self_antipart, mass,
        width, color, decay and is_part), number)."""

        part = model.get_particle(leg.get('id'))

        # For legs with decay chains defined, include leg id (don't combine)
        if leg.get('onshell'): id = leg.get('id')
        else: id = 0
        # For FS legs, don't care about number (but do for IS legs)
        if leg.get('state'): number = 0
        else: number = leg.get('number')
        # Include also onshell, since this specifies forbidden s-channel
        return [((number, id, part.get('spin'), leg.get('onshell'),
                  part.get('is_part'), part.get('self_antipart'),
                  part.get('mass'), part.get('width'), part.get('color')),
                 leg.get('number'))]
        
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify matrix elements:
        interaction color, lorentz, coupling, and wavefunction spin,
        self_antipart, mass, width, color, decay and is_part, plus PDG
        code if possible onshell s-channel prop. Note that is_part
        and PDG code needs to be flipped if we move the final vertex around."""

        if vertex.get('id') in [0,-1]:
            return ((vertex.get('id'),),)

        if vertex.get('id') == -2:
            ret_list = vertex.get('loop_tag')
        else:
            inter = model.get_interaction(vertex.get('id'))
            coup_keys = sorted(inter.get('couplings').keys())
            ret_list = tuple([(key, inter.get('couplings')[key]) for key in \
                          coup_keys] + \
                         [str(c) for c in inter.get('color')] + \
                         inter.get('lorentz')+sorted(inter.get('orders')))
                   
        if last_vertex:
            return ((ret_list,),)
        else:
            part = model.get_particle(vertex.get('legs')[-1].get('id'))
            # If we have possibly onshell s-channel particles with
            # identical properties but different PDG code, split the
            # processes to ensure that we write the correct resonance
            # in the event file
            s_pdg = vertex.get_s_channel_id(model, ninitial)
            if s_pdg and (part.get('width').lower() == 'zero' or \
               vertex.get('legs')[-1].get('onshell') == False):
                s_pdg = 0
            return (((part.get('spin'), part.get('color'),
                     part.get('self_antipart'),
                     part.get('mass'), part.get('width'), s_pdg),
                    ret_list),)

    @staticmethod
    def flip_vertex(new_vertex, old_vertex, links):
        """Move the wavefunction part of vertex id appropriately"""

        if len(new_vertex[0]) == 1 and len(old_vertex[0]) == 2:
            # We go from a last link to next-to-last link - add propagator info
            return ((old_vertex[0][0],new_vertex[0][0]),)
        elif len(new_vertex[0]) == 2 and len(old_vertex[0]) == 1:
            # We go from next-to-last link to last link - remove propagator info
            return ((new_vertex[0][1],),)
        # We should not get here
        assert(False)


class IdentifyMETagFKS(IdentifyMETag):
    """on top of what IdentifyMETag, the diagram tags also have the charge 
    difference along the fermionic flow in them for initial state legs."""

    def __init__(self, diagram, model = None, ninitial = 2):
        self.flow_charge_diff = diagram.get_flow_charge_diff(model)
        super(IdentifyMETagFKS, self).__init__(diagram, model, ninitial)

    def __eq__(self, other):
        return super(IdentifyMETag, self).__eq__(other) and \
                self.flow_charge_diff == other.flow_charge_diff


class IdentifyMETagMadSpin(IdentifyMETag):
    """Should ensure that the splitting is the same with and without decay 
       So we want to combine processes with different possibly
       onshell s-channel propagators. This was done by the vertex.
    """
    
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify matrix elements:
        interaction color, lorentz, coupling, and wavefunction spin,
        self_antipart, mass, width, color, decay and is_part.
        BUT NOT PDG code for possible onshell s-channel prop."""
        if last_vertex:
            return IdentifyMETag.vertex_id_from_vertex(vertex, last_vertex, model, ninitial)
        import random
        data = IdentifyMETag.vertex_id_from_vertex(vertex, last_vertex, model, ninitial)
        ((spin, color, selfanti, mass, width, pdg), ret_list) = data
        return ((spin, color, selfanti, mass, width, random.random()), ret_list) 

        
#===============================================================================
# DiagramTag class to create canonical order configs
#===============================================================================

class CanonicalConfigTag(diagram_generation.DiagramTag):
    """DiagramTag daughter class to create canonical order of
    config. Need to compare leg number, mass, width, and color.
    Also implement find s- and t-channels from the tag.
    Warning! The sorting in this tag must be identical to that of
       IdentifySGConfigTag in diagram_symmetry.py (apart from leg number)
       to make sure symmetry works!"""


    def get_s_and_t_channels(self, ninitial, model, new_pdg, max_final_leg = 2):
        """Get s and t channels from the tag, as two lists of vertices
        ordered from the outermost s-channel and in/down towards the highest
        number initial state leg. 
        Algorithm: Start from the final tag. Check for final leg number for 
        all links and move in the direction towards leg 2 (or 1, if 1 and 2
        are in the same direction).
        """

        final_leg = min(ninitial, max_final_leg)

        # Look for final leg numbers in all links
        done = [l for l in self.tag.links if \
                l.end_link and l.links[0][1][0] == final_leg]
        while not done:
            # Identify the chain closest to final_leg
            right_num = -1
            for num, link in enumerate(self.tag.links):
                if len(link.vertex_id) == 3 and \
                        link.vertex_id[1][-1] == final_leg:
                    right_num = num
            if right_num == -1:
                # We need to look for leg number 1 instead
                for num, link in enumerate(self.tag.links):
                    if len(link.vertex_id) == 3 and \
                       link.vertex_id[1][-1] == 1:
                        right_num = num
            if right_num == -1:
                # This should never happen
                raise diagram_generation.DiagramTag.DiagramTagError, \
                    "Error in CanonicalConfigTag, no link with number 1 or 2."

            # Now move one step in the direction of right_link
            right_link = self.tag.links[right_num]
            # Create a new link corresponding to moving one step
            new_links = list(self.tag.links[:right_num]) + \
                                           list(self.tag.links[right_num + 1:])
 
            new_link = diagram_generation.DiagramTagChainLink(\
                                           new_links,
                                           self.flip_vertex(\
                                             self.tag.vertex_id,
                                             right_link.vertex_id,
                                             new_links))

            # Create a new final vertex in the direction of the right_link
            other_links = list(right_link.links) + [new_link]
            other_link = diagram_generation.DiagramTagChainLink(\
                                             other_links,
                                             self.flip_vertex(\
                                                 right_link.vertex_id,
                                                 self.tag.vertex_id,
                                                 other_links))

            self.tag = other_link
            done = [l for l in self.tag.links if \
                    l.end_link and l.links[0][1][0] == final_leg]

        # Construct a diagram from the resulting tag
        diagram = self.diagram_from_tag(model)

        # Go through the vertices and add them to s-channel or t-channel
        schannels = base_objects.VertexList()
        tchannels = base_objects.VertexList()

        for vert in diagram.get('vertices')[:-1]:
            if vert.get('legs')[-1].get('number') > ninitial:
                schannels.append(vert)
            else:
                tchannels.append(vert)

        # Need to make sure leg number 2 is always last in last vertex
        lastvertex = diagram.get('vertices')[-1]
        legs = lastvertex.get('legs')
        leg2 = [l.get('number') for l in legs].index(final_leg)
        legs.append(legs.pop(leg2))
        if ninitial == 2:
            # Last vertex always counts as t-channel        
            tchannels.append(lastvertex)
        else:
            legs[-1].set('id', 
                     model.get_particle(legs[-1].get('id')).get_anti_pdg_code())
            schannels.append(lastvertex)

        # Split up multiparticle vertices using fake s-channel propagators
        multischannels = [(i, v) for (i, v) in enumerate(schannels) \
                          if len(v.get('legs')) > 3]
        multitchannels = [(i, v) for (i, v) in enumerate(tchannels) \
                          if len(v.get('legs')) > 3]
        
        increase = 0
        for channel in multischannels + multitchannels:
            newschannels = []
            vertex = channel[1]
            while len(vertex.get('legs')) > 3:
                # Pop the first two legs and create a new
                # s-channel from them
                popped_legs = \
                           base_objects.LegList([vertex.get('legs').pop(0) \
                                                    for i in [0,1]])
                popped_legs.append(base_objects.Leg({\
                    'id': new_pdg,
                    'number': min([l.get('number') for l in popped_legs]),
                    'state': True,
                    'onshell': None}))

                new_vertex = base_objects.Vertex({
                    'id': vertex.get('id'),
                    'legs': popped_legs})

                # Insert the new s-channel before this vertex
                if channel in multischannels:
                    schannels.insert(channel[0]+increase, new_vertex)
                    # Account for previous insertions
                    increase += 1
                else:
                    schannels.append(new_vertex)
                legs = vertex.get('legs')
                # Insert the new s-channel into vertex
                legs.insert(0, copy.copy(popped_legs[-1]))
                # Renumber resulting leg according to minimum leg number
                legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

        # Finally go through all vertices, sort the legs and replace
        # leg number with propagator number -1, -2, ...
        number_dict = {}
        nprop = 0
        for vertex in schannels + tchannels:
            # Sort the legs
            legs = vertex.get('legs')[:-1]
            if vertex in schannels:
                legs.sort(lambda l1, l2: l2.get('number') - \
                          l1.get('number'))
            else:
                legs.sort(lambda l1, l2: l1.get('number') - \
                          l2.get('number'))
            for ileg,leg in enumerate(legs):
                newleg = copy.copy(leg)
                try:
                    newleg.set('number', number_dict[leg.get('number')])
                except KeyError:
                    pass
                else:
                    legs[ileg] = newleg                    
            nprop = nprop - 1
            last_leg = copy.copy(vertex.get('legs')[-1])
            number_dict[last_leg.get('number')] = nprop
            last_leg.set('number', nprop)
            legs.append(last_leg)
            vertex.set('legs', base_objects.LegList(legs))

        return schannels, tchannels

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the end link for a leg needed to identify configs: 
        ((leg numer, mass, width, color), number)."""

        part = model.get_particle(leg.get('id'))

        # use charge to forbid identification of neutrino and lepton
        # the main reason is that the cuts are different on such particles.
        if part.get('color') != 1:
            charge = 0
        else:
            charge = abs(part.get('charge'))

        return [((leg.get('number'), part.get('spin'), part.get('color'), charge,
                  part.get('mass'), part.get('width')),
                 (leg.get('number'),leg.get('id'),leg.get('state')))]
        
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify configs:
        interaction color, mass, width. Also provide propagator PDG code.
        The third element of the tuple vertex_id serves to store potential
        necessary information regarding the shrunk loop."""
                
        if isinstance(vertex,base_objects.ContractedVertex):
            inter = None
            # I don't add here the 'loop_tag' because it is heavy and in principle
            # not necessary. It can however be added in the future if proven
            # necessary.
            loop_info = {'PDGs':vertex.get('PDGs'),
                         'loop_orders':vertex.get('loop_orders')}
        else:
            # Not that it is going to be used here, but it might be eventually
            inter = model.get_interaction(vertex.get('id'))
            loop_info = {}

        if last_vertex:
            return ((0,),
                    (vertex.get('id'),
                     min([l.get('number') for l in vertex.get('legs')])),
                     loop_info)
        else:
            part = model.get_particle(vertex.get('legs')[-1].get('id'))
            return ((part.get('color'),
                     part.get('mass'), part.get('width')),
                    (vertex.get('id'), 
                     vertex.get('legs')[-1].get('onshell'),
                     vertex.get('legs')[-1].get('number')),
                     loop_info)

    @staticmethod
    def flip_vertex(new_vertex, old_vertex, links):
        """Move the wavefunction part of propagator id appropriately"""

        # Find leg numbers for new vertex
        min_number = min([l.vertex_id[1][-1] for l in links if not l.end_link]\
                         + [l.links[0][1][0] for l in links if l.end_link])

        if len(new_vertex[0]) == 1 and len(old_vertex[0]) > 1:
            # We go from a last link to next-to-last link 
            return (old_vertex[0], 
                    (new_vertex[1][0], old_vertex[1][1], min_number), new_vertex[2])
        elif len(new_vertex[0]) > 1 and len(old_vertex[0]) == 1:
            # We go from next-to-last link to last link - remove propagator info
            return (old_vertex[0], (new_vertex[1][0], min_number), new_vertex[2])

        # We should not get here
        raise diagram_generation.DiagramTag.DiagramTagError, \
              "Error in CanonicalConfigTag, wrong setup of vertices in link."
        
    @staticmethod
    def leg_from_link(link):
        """Return a leg from a link"""

        if link.end_link:
            # This is an external leg, info in links
            leg = base_objects.Leg({'number':link.links[0][1][0],
                                     'id':link.links[0][1][1],
                                     'state':link.links[0][1][2],
                                     'onshell':None})
            return leg
        # This shouldn't happen
        assert False

    @classmethod
    def vertex_from_link(cls, legs, vertex_id, model):
        """Return a vertex given a leg list and a vertex id"""
        
        vert_ID = cls.id_from_vertex_id(vertex_id)
        if vert_ID != -2:
            vertex = base_objects.Vertex({'legs': legs,
                                        'id': vert_ID})
        else:
            vertex = base_objects.ContractedVertex({'legs': legs,'id': -2})
            for key, value in enumerate(cls.loop_info_from_vertex_id(vertex_id)):
                # For now all the info put in the vertex_id is also an attribute
                # of the ContractedVertex instance, i.e. 'PDGs' and 'loop_tag'
                if key in vertex:
                    vertex.set(key,value) 

        if len(vertex_id[1]) == 3:
            vertex.get('legs')[-1].set('onshell', vertex_id[1][1])
        return vertex

    @staticmethod
    def id_from_vertex_id(vertex_id):
        """Return the numerical vertex id from a link.vertex_id"""

        return vertex_id[1][0]


#===============================================================================
# HelasWavefunction
#===============================================================================
class HelasWavefunction(base_objects.PhysicsObject):
    """HelasWavefunction object, has the information necessary for
    writing a call to a HELAS wavefunction routine: the PDG number,
    all relevant particle information, a list of mother wavefunctions,
    interaction id, all relevant interaction information, fermion flow
    state, wavefunction number
    """
    
    supported_analytical_info = ['wavefunction_rank','interaction_rank']


    @staticmethod
    def spin_to_size(spin):
        """ Returns the size of a wavefunction (i.e. number of element) carrying
        a particle with spin 'spin' """

        sizes = {1:1,2:4,3:4,4:16,5:16}
        try:
            return sizes[abs(spin)]
        except KeyError:
            raise MadGraph5Error, "L-cut particle has spin %d which is not supported."%spin

    def default_setup(self):
        """Default values for all properties"""

        # Properties related to the particle propagator
        # For an electron, would have the following values
        # pdg_code = 11
        # name = 'e-'
        # antiname = 'e+'
        # spin = '1'   defined as 2 x spin + 1  
        # color = '1'  1= singlet, 3 = triplet, 8=octet
        # mass = 'zero'
        # width = 'zero'
        # is_part = 'true'    Particle not antiparticle
        # self_antipart='false'   gluon, photo, h, or majorana would be true
        self['particle'] = base_objects.Particle()
        self['antiparticle'] = base_objects.Particle()
        self['is_part'] = True
        # Properties related to the interaction generating the propagator
        # For an e- produced from an e+e-A vertex would have the following
        # proporties:
        # interaction_id = the id of the interaction in the model
        # pdg_codes = the pdg_codes property of the interaction, [11, -11, 22]
        # inter_color = the 'color' property of the interaction: []
        # lorentz = the 'lorentz' property of the interaction: ('')
        # couplings = the coupling names from the interaction: {(0,0):'MGVX12'}
        self['interaction_id'] = 0
        self['pdg_codes'] = []
        self['orders'] = {}
        self['inter_color'] = None
        self['lorentz'] = []
        self['coupling'] = ['none']
        # The color index used in this wavefunction
        self['color_key'] = 0
        # Properties relating to the leg/vertex
        # state = initial/final (for external bosons),
        #         intermediate (for intermediate bosons),
        #         incoming/outgoing (for fermions)
        # leg_state = initial/final for initial/final legs
        #             intermediate for non-external wavefunctions
        # number_external = the 'number' property of the corresponding Leg,
        #                   corresponds to the number of the first external
        #                   particle contributing to this leg
        # fermionflow = 1    fermions have +-1 for flow (bosons always +1),
        #                    -1 is used only if there is a fermion flow clash
        #                    due to a Majorana particle 
        # is_loop = logical true if this function builds a loop or belong
        #           to an external structure.
        self['state'] = 'initial'
        self['leg_state'] = True
        self['mothers'] = HelasWavefunctionList()
        self['number_external'] = 0
        self['number'] = 0
        self['me_id'] = 0
        self['fermionflow'] = 1
        self['is_loop'] = False
        # Some analytical information about the interaction and wavefunction
        # can be cached here in the form of a dictionary.
        # An example of a key of this dictionary is 'rank' which is used in the
        # open loop method and stores the rank of the polynomial describing the 
        # loop numerator up to this loop wavefunction.
        # See the class variable 'supported_analytical_info' for the list of
        # supporter analytical information
        self['analytic_info'] = {}
        # The lcut_size stores the size of the L-Cut wavefunction at the origin
        # of the loopwavefunction.
        self['lcut_size']=None
        # The decay flag is used in processes with defined decay chains,
        # to indicate that this wavefunction has a decay defined
        self['decay'] = False
        # The onshell flag is used in processes with defined decay
        # chains, to indicate that this wavefunction is decayed and
        # should be onshell (True), as well as for forbidden s-channels (False).
        # Default is None
        self['onshell'] = None
        # conjugate_indices is a list [1,2,...] with fermion lines
        # that need conjugates. Default is "None"
        self['conjugate_indices'] = None

    # Customized constructor
    def __init__(self, *arguments):
        """Allow generating a HelasWavefunction from a Leg
        """

        if len(arguments) > 2:
            if isinstance(arguments[0], base_objects.Leg) and \
                   isinstance(arguments[1], int) and \
                   isinstance(arguments[2], base_objects.Model):
                super(HelasWavefunction, self).__init__()
                leg = arguments[0]
                interaction_id = arguments[1]
                model = arguments[2]
                # decay_ids is the pdg codes for particles with decay
                # chains defined
                decay_ids = []
                if len(arguments) > 3:
                    decay_ids = arguments[3]
                self.set('particle', leg.get('id'), model)
                self.set('number_external', leg.get('number'))
                self.set('number', leg.get('number'))
                self.set('is_loop', leg.get('loop_line'))
                self.set('state', {False: 'initial', True: 'final'}[leg.get('state')])
                if leg.get('onshell') == False:
                    # Denotes forbidden s-channel
                    self.set('onshell', leg.get('onshell'))
                self.set('leg_state', leg.get('state'))
                # Need to set 'decay' to True for particles which will be
                # decayed later, in order to not combine such processes
                # although they might have identical matrix elements before
                # the decay is applied
                if self['state'] == 'final' and self.get('pdg_code') in decay_ids:
                    self.set('decay', True)

                # Set fermion flow state. Initial particle and final
                # antiparticle are incoming, and vice versa for
                # outgoing
                if self.is_fermion():
                    if leg.get('state') == False and \
                           self.get('is_part') or \
                           leg.get('state') == True and \
                           not self.get('is_part'):
                        self.set('state', 'incoming')
                    else:
                        self.set('state', 'outgoing')
                self.set('interaction_id', interaction_id, model)
        elif arguments:
            super(HelasWavefunction, self).__init__(arguments[0])
        else:
            super(HelasWavefunction, self).__init__()

    def filter(self, name, value):
        """Filter for valid wavefunction property values."""

        if name in ['particle', 'antiparticle']:
            if not isinstance(value, base_objects.Particle):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a particle" % (name, repr(value))            

        if name == 'is_part':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a boolean" % (name, repr(value))

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer " % str(value) + \
                        " for wavefunction interaction id"

        if name == 'pdg_codes':
            #Should be a list of strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name == 'orders':
            #Should be a dict with valid order names ask keys and int as values
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict for coupling orders" % \
                                                                    str(value)
            for order in value.keys():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(order)
                if not isinstance(value[order], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value[order])


        if name == 'inter_color':
            # Should be None or a color string
            if value and not isinstance(value, color.ColorString):
                    raise self.PhysicsObjectError, \
                            "%s is not a valid Color String" % str(value)

        if name == 'lorentz':
            #Should be a list of string
            if not isinstance(value, list):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            for name in value:
                if not isinstance(name, str):
                    raise self.PhysicsObjectError, \
                        "%s doesn't contain only string" % str(value)

        if name == 'coupling':
            #Should be a list of string
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid coupling string" % str(value)
            for name in value:
                if not isinstance(name, str):
                    raise self.PhysicsObjectError, \
                        "%s doesn't contain only string" % str(value)
            if len(value) == 0:
                raise self.PhysicsObjectError, \
                        "%s should have at least one value" % str(value)

        if name == 'color_key':
            if value and not isinstance(value, int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid integer" % str(value)

        if name == 'state':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for wavefunction state" % \
                                                                    str(value)
            if value not in ['incoming', 'outgoing',
                             'intermediate', 'initial', 'final']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction " % str(value) + \
                        "state (incoming|outgoing|intermediate)"
        if name == 'leg_state':
            if value not in [False, True]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction " % str(value) + \
                        "state (incoming|outgoing|intermediate)"
        if name in ['fermionflow']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)
            if not value in [-1, 1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid sign (must be -1 or 1)" % str(value)

        if name in ['number_external', 'number']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value) + \
                        " for wavefunction number"

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for wavefunction" % \
                      str(value)

        if name in ['decay']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value) + \
                        " for decay"
        
        if name in ['onshell']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value) + \
                        " for onshell"

        if name in ['is_loop']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value) + \
                        " for is_loop"
                        
        if name == 'conjugate_indices':
            if not isinstance(value, tuple) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid tuple" % str(value) + \
                        " for conjugate_indices"

        if name == 'rank':
            if not isinstance(value, int) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid int" % str(value) + \
                        " for the rank"

        if name == 'lcut_size':
            if not isinstance(value, int) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid int" % str(value) + \
                        " for the lcut_size"

        return True

    # Enhanced get function, where we can directly call the properties of the particle
    def get(self, name):
        """When calling any property related to the particle,
        automatically call the corresponding property of the particle."""

        # Set conjugate_indices if it's not already set
        if name == 'conjugate_indices' and self[name] == None:
            self['conjugate_indices'] = self.get_conjugate_index()
        
        if name == 'lcut_size' and self[name] == None:
            self['lcut_size'] = self.get_lcut_size()  

        if name in ['spin', 'mass', 'width', 'self_antipart']:
            return self['particle'].get(name)
        elif name == 'pdg_code':
            return self['particle'].get_pdg_code()
        elif name == 'color':
            return self['particle'].get_color()
        elif name == 'name':
            return self['particle'].get_name()
        elif name == 'antiname':
            return self['particle'].get_anti_name()
        elif name == 'me_id':
            out = super(HelasWavefunction, self).get(name)
            if out: 
                return out
            else:
                return super(HelasWavefunction, self).get('number')
        else:
            return super(HelasWavefunction, self).get(name)
        

    # Enhanced set function, where we can append a model
    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        assert len(arguments) >1, "Too few arguments for set"

        name = arguments[0]
        value = arguments[1]

        if len(arguments) > 2 and \
               isinstance(value, int) and \
               isinstance(arguments[2], base_objects.Model):
            model = arguments[2]
            if name == 'interaction_id':
                self.set('interaction_id', value)
                if value > 0:
                    inter = model.get('interaction_dict')[value]
                    self.set('pdg_codes',
                             [part.get_pdg_code() for part in \
                              inter.get('particles')])
                    self.set('orders', inter.get('orders'))
                    # Note that the following values might change, if
                    # the relevant color/lorentz/coupling is not index 0
                    if inter.get('color'):
                        self.set('inter_color', inter.get('color')[0])
                    if inter.get('lorentz'):
                        self.set('lorentz', [inter.get('lorentz')[0]])
                    if inter.get('couplings'):
                        self.set('coupling', [inter.get('couplings').values()[0]])
                return True
            elif name == 'particle':
                self.set('particle', model.get('particle_dict')[value])
                self.set('is_part', self['particle'].get('is_part'))
                if self['particle'].get('self_antipart'):
                    self.set('antiparticle', self['particle'])
                else:
                    self.set('antiparticle', model.get('particle_dict')[-value])
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasWavefunction, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['particle', 'antiparticle', 'is_part',
                'interaction_id', 'pdg_codes', 'orders', 'inter_color', 
                'lorentz', 'coupling', 'color_key', 'state', 'number_external',
                'number', 'fermionflow', 'mothers', 'is_loop']

    # Helper functions

    def flip_part_antipart(self):
        """Flip between particle and antiparticle."""
        part = self.get('particle')
        self.set('particle', self.get('antiparticle'))
        self.set('antiparticle', part)
    
    def is_anticommutating_ghost(self):
        """ Return True if the particle of this wavefunction is a ghost"""
        return self.get('particle').get('ghost')
    
    def is_fermion(self):
        return self.get('spin') % 2 == 0

    def is_boson(self):
        return not self.is_fermion()

    def is_majorana(self):
        return self.is_fermion() and self.get('self_antipart')

    def get_analytic_info(self, info, alohaModel=None):
        """ Returns a given analytic information about this loop wavefunction or
        its characterizing interaction. The list of available information is in
        the HelasWavefunction class variable 'supported_analytical_info'. An
        example of analytic information is the 'interaction_rank', corresponding
        to the power of the loop momentum q brought by the interaction
        and propagator from which this loop wavefunction originates. This 
        is done in a general way by having aloha analyzing the lorentz structure
        used.
        Notice that if one knows that this analytic information has already been
        computed before (for example because a call to compute_analytic_information
        has been performed before, then alohaModel is not required since 
        the result can be recycled."""
        # This function makes no sense if not called for a loop interaction.
        # At least for now
        assert(self.get('is_loop'))
        # Check that the required information is supported
        assert(info in self.supported_analytical_info)
                
        # Try to recycle the information
        try:
            return self['analytic_info'][info]
        except KeyError:
            # It then need be computed and for this, an alohaModel is necessary
            if alohaModel is None:
                raise MadGraph5Error,"The analytic information %s has"%info+\
                " not been computed yet for this wavefunction and an"+\
                " alohaModel was not specified, so that the information"+\
                " cannot be retrieved."
        
        result = None
        
        if info=="interaction_rank" and len(self['mothers'])==0:
            # It is of course zero for an external particle
            result = 0

        elif info=="interaction_rank":
            # To get advanced analytic information about the interaction, aloha
            # has to work as if in the optimized output, namely treat individually
            # the loop momentum from the rest of the contributions.
            # The 'L' tag is therefore always followed by an integer representing
            # the place of the loop wavefunction in the mothers.
            # For this, optimized_output is set to True below, no matter what.
            aloha_info = self.get_aloha_info(True)
            # aloha_info[0] is the tuple of all lorent structures for this lwf,
            # aloha_info[1] are the tags and aloha_info[2] is the outgoing number.
            max_rank = max([ alohaModel.get_info('rank', lorentz,
                                 aloha_info[2], aloha_info[1], cached=True) 
                                                for lorentz in aloha_info[0] ])
            result = max_rank

        elif info=="wavefunction_rank":
            # wavefunction_rank is the sum of all the interaction rank
            # from the history of open-loop call.
            loop_mothers=[wf for wf in self['mothers'] if wf['is_loop']]
            if len(loop_mothers)==0:
                # It is an external loop wavefunction
                result = 0
            elif len(loop_mothers)==1:
                result=loop_mothers[0].get_analytic_info('wavefunction_rank',
                                                                     alohaModel)
                result = result+self.get_analytic_info('interaction_rank',
                                                                     alohaModel)
            else:
                raise MadGraph5Error, "A loop wavefunction has more than one loop"+\
                    " mothers."
                    
        # Now cache the resulting analytic info
        self['analytic_info'][info] = result
        
        return result

    def compute_analytic_information(self, alohaModel):
        """ Make sure that all analytic pieces of information about this 
        wavefunction are computed so that they can be recycled later, typically
        without the need of specifying an alohaModel."""
        
        for analytic_info in self.supported_analytical_info:
            self.get_analytic_info(analytic_info, alohaModel)

    def to_array(self):
        """Generate an array with the information needed to uniquely
        determine if a wavefunction has been used before: interaction
        id and mother wavefunction numbers."""

        # Identification based on interaction id
        array_rep = array.array('i', [self['interaction_id']])
        # Need the coupling key, to distinguish between
        # wavefunctions from the same interaction but different
        # color structures
        array_rep.append(self['color_key'])
        # Also need to specify if it is a loop wf
        array_rep.append(int(self['is_loop']))
        
        # Finally, the mother numbers
        array_rep.extend([mother['number'] for \
                          mother in self['mothers']])
        
        return array_rep

    def get_pdg_code(self):
        """Generate the corresponding pdg_code for an outgoing particle,
        taking into account fermion flow, for mother wavefunctions"""

        return self.get('pdg_code')

    def get_anti_pdg_code(self):
        """Generate the corresponding pdg_code for an incoming particle,
        taking into account fermion flow, for mother wavefunctions"""

        if self.get('self_antipart'):
            #This is its own antiparticle e.g. gluon
            return self.get('pdg_code')

        return - self.get('pdg_code')

    def set_scalar_coupling_sign(self, model):
        """Check if we need to add a minus sign due to non-identical
        bosons in HVS type couplings"""

        inter = model.get('interaction_dict')[self.get('interaction_id')]
        if [p.get('spin') for p in \
                   inter.get('particles')] == [3, 1, 1]:
            particles = inter.get('particles')
            #                   lambda p1, p2: p1.get('spin') - p2.get('spin'))
            if particles[1].get_pdg_code() != particles[2].get_pdg_code() \
                   and self.get('pdg_code') == \
                   particles[1].get_anti_pdg_code()\
                   and not self.get('coupling')[0].startswith('-'):
                # We need a minus sign in front of the coupling
                self.set('coupling', ['-%s'%c for c in self.get('coupling')])

    def set_octet_majorana_coupling_sign(self):
        """For octet Majorana fermions, need an extra minus sign in
        the FVI (and FSI?) wavefunction in UFO models."""

        # Add minus sign to coupling of color octet Majorana
        # particles to g for FVI vertex
        if self.get('color') == 8 and \
               self.get_spin_state_number() == -2 and \
               self.get('self_antipart') and \
               [m.get('color') for m in self.get('mothers')] == [8, 8] and \
               not self.get('coupling')[0].startswith('-'):
            self.set('coupling', ['-%s' % c for c in self.get('coupling')])
        
    def set_state_and_particle(self, model):
        """Set incoming/outgoing state according to mother states and
        Lorentz structure of the interaction, and set PDG code
        according to the particles in the interaction"""

        assert isinstance(model, base_objects.Model), \
                  "%s is not a valid model for call to set_state_and_particle" \
                  % repr(model)

        # leg_state is final, unless there is exactly one initial 
        # state particle involved in the combination -> t-channel
        if len(filter(lambda mother: mother.get('leg_state') == False,
                      self.get('mothers'))) == 1:
            leg_state = False
        else:
            leg_state = True
        self.set('leg_state', leg_state)

        # Start by setting the state of the wavefunction
        if self.is_boson():
            # For boson, set state to intermediate
            self.set('state', 'intermediate')
        else:
            # For fermion, set state to same as other fermion (in the
            # right way)
            mother = self.find_mother_fermion()

            if self.get('self_antipart'):
                self.set('state', mother.get_with_flow('state'))
                self.set('is_part', mother.get_with_flow('is_part'))
            else:
                self.set('state', mother.get('state'))
                self.set('fermionflow', mother.get('fermionflow'))
                # Check that the state is compatible with particle/antiparticle
                if self.get('is_part') and self.get('state') == 'incoming' or \
                   not self.get('is_part') and self.get('state') == 'outgoing':
                    self.set('state', {'incoming':'outgoing',
                                      'outgoing':'incoming'}[self.get('state')])
                    self.set('fermionflow', -self.get('fermionflow'))
        return True

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   wf_number):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers. This can happen when there is a
        Majorana particle in the diagram, which can flip the fermion
        flow. This is detected either by a wavefunctions or an
        amplitude, with 2 fermion mothers with same state.

        In this case, we need to follow the fermion lines of the
        mother wavefunctions until we find the outermost Majorana
        fermion. For all fermions along the line up to (but not
        including) the Majorana fermion, we need to flip incoming <->
        outgoing and particle id. For all fermions after the Majorana
        fermion, we need to flip the fermionflow property (1 <-> -1).

        The reason for this is that in the Helas calls, we need to
        keep track of where the actual fermion flow clash happens
        (i.e., at the outermost Majorana), as well as having the
        correct fermion flow for all particles along the fermion line.

        This is done by the mothers using
        HelasWavefunctionList.check_and_fix_fermion_flow, which in
        turn calls the recursive function
        check_majorana_and_flip_flow to trace the fermion lines.
        """

        # Use the HelasWavefunctionList helper function
        # Have to keep track of wavefunction number, since we might
        # need to add new wavefunctions.
        self.set('mothers', self.get('mothers').sort_by_pdg_codes(\
            self.get('pdg_codes'), self.get_anti_pdg_code())[0])

        wf_number = self.get('mothers').\
                         check_and_fix_fermion_flow(wavefunctions,
                                                    diagram_wavefunctions,
                                                    external_wavefunctions,
                                                    self,
                                                    wf_number)

        return self, wf_number

    def check_majorana_and_flip_flow(self, found_majorana,
                                     wavefunctions,
                                     diagram_wavefunctions,
                                     external_wavefunctions,
                                     wf_number, force_flip_flow=False,
                                     number_to_wavefunctions=[]):
        """Recursive function. Check for Majorana fermion. If found,
        continue down to external leg, then flip all the fermion flows
        on the way back up, in the correct way:
        Only flip fermionflow after the last Majorana fermion; for
        wavefunctions before the last Majorana fermion, instead flip
        particle identities and state. Return the new (or old)
        wavefunction, and the present wavefunction number.

        Arguments:
          found_majorana: boolean
          wavefunctions: HelasWavefunctionList with previously
                         defined wavefunctions
          diagram_wavefunctions: HelasWavefunctionList with the wavefunctions
                         already defined in this diagram
          external_wavefunctions: dictionary from legnumber to external wf
          wf_number: The present wavefunction number
        """

        if not found_majorana:
            found_majorana = self.get('self_antipart')

        new_wf = self
        flip_flow = False
        flip_sign = False

        # Stop recursion at the external leg
        mothers = copy.copy(self.get('mothers'))
        if not mothers:
            if force_flip_flow:
                flip_flow = True
            elif not self.get('self_antipart'):
                flip_flow = found_majorana
            else:
                flip_sign = found_majorana
        else:
            # Follow fermion flow up through tree
            fermion_mother = self.find_mother_fermion()

            if fermion_mother.get_with_flow('state') != \
                                           self.get_with_flow('state'):
                new_mother = fermion_mother
            else:
                # Perform recursion by calling on mother
                new_mother, wf_number = fermion_mother.\
                                        check_majorana_and_flip_flow(\
                                           found_majorana,
                                           wavefunctions,
                                           diagram_wavefunctions,
                                           external_wavefunctions,
                                           wf_number,
                                           force_flip_flow)

            # If this is Majorana and mother has different fermion
            # flow, we should flip the particle id and flow state.
            # Otherwise, if mother has different fermion flow, flip
            # flow
            flip_sign = new_mother.get_with_flow('state') != \
                        self.get_with_flow('state') and \
                        self.get('self_antipart')
            flip_flow = new_mother.get_with_flow('state') != \
                        self.get_with_flow('state') and \
                        not self.get('self_antipart')

            # Replace old mother with new mother
            mothers[mothers.index(fermion_mother)] = new_mother

        # Flip sign if needed
        if flip_flow or flip_sign:
            if self in wavefunctions:
                # Need to create a new copy, since we don't want to change
                # the wavefunction for previous diagrams
                new_wf = copy.copy(self)
                # Update wavefunction number
                wf_number = wf_number + 1
                new_wf.set('number', wf_number)
                try:
                    # In call from insert_decay, we want to replace
                    # also identical wavefunctions in the same diagram
                    old_wf_index = diagram_wavefunctions.index(self)
                    old_wf = diagram_wavefunctions[old_wf_index]
                    if self.get('number') == old_wf.get('number'):
                        # The wavefunction and old_wf are the same -
                        # need to reset wf_number and new_wf number
                        wf_number -= 1
                        new_wf.set('number', old_wf.get('number'))
                    diagram_wavefunctions[old_wf_index] = new_wf
                except ValueError:
                    # Make sure that new_wf comes before any wavefunction
                    # which has it as mother
                    if len(self['mothers']) == 0:
                        #insert at the beginning
                        if diagram_wavefunctions:
                            wf_nb = diagram_wavefunctions[0].get('number')
                            for w in diagram_wavefunctions:
                                w.set('number', w.get('number') + 1)
                            new_wf.set('number', wf_nb)
                            diagram_wavefunctions.insert(0, new_wf)
                        else:
                            diagram_wavefunctions.insert(0, new_wf)
                    else:
                        for i, wf in enumerate(diagram_wavefunctions):
                            if self in wf.get('mothers'):
                                # Update wf numbers
                                new_wf.set('number', wf.get('number'))
                                for w in diagram_wavefunctions[i:]:
                                    w.set('number', w.get('number') + 1)
                                # Insert wavefunction
                                diagram_wavefunctions.insert(i, new_wf)
                                break
                        else:
                            # For loop processes, care is needed since
                            # some loop wavefunctions in the diag_wfs might have
                            # the new_wf in their mother, so we want to place 
                            # new_wf as early as possible in the list.
                            # We first look if any mother of the wavefunction
                            # we want to add appears in the diagram_wavefunctions
                            # list. If it doesn't, max_mother_index is -1.
                            # If it does, then max_mother_index is the maximum
                            # index in diagram_wavefunctions of those of the 
                            # mothers present in this list.
                            max_mother_index = max([-1]+
                                [diagram_wavefunctions.index(wf) for wf in 
                                        mothers if wf in diagram_wavefunctions])
                            
                            # We want to insert this new_wf as early as 
                            # possible in the diagram_wavefunctions list so that
                            # we are guaranteed that it will be placed *before*
                            # wavefunctions that have new_wf as a mother.
                            # We therefore place it at max_mother_index+1.
                            if max_mother_index<len(diagram_wavefunctions)-1:
                                new_wf.set('number',diagram_wavefunctions[
                                              max_mother_index+1].get('number'))
                            for wf in diagram_wavefunctions[max_mother_index+1:]:
                                wf.set('number',wf.get('number')+1)
                            diagram_wavefunctions.insert(max_mother_index+1,
                                                                         new_wf)

            # Set new mothers
            new_wf.set('mothers', mothers)
                    
            # Now flip flow or sign
            if flip_flow:
                # Flip fermion flow
                new_wf.set('fermionflow', -new_wf.get('fermionflow'))

            if flip_sign:
                # Flip state and particle identity
                # (to keep particle identity * flow state)
                new_wf.set('state', filter(lambda state: \
                                           state != new_wf.get('state'),
                                           ['incoming', 'outgoing'])[0])
                new_wf.set('is_part', not new_wf.get('is_part'))
            try:
                # Use the copy in wavefunctions instead.
                # Remove this copy from diagram_wavefunctions
                new_wf_number = new_wf.get('number')
                new_wf = wavefunctions[wavefunctions.index(new_wf)]
                diagram_wf_numbers = [w.get('number') for w in \
                                                          diagram_wavefunctions]
                index = diagram_wf_numbers.index(new_wf_number)
                diagram_wavefunctions.pop(index)
                # We need to decrease the wf number for later
                # diagram wavefunctions
                for wf in diagram_wavefunctions:
                    if wf.get('number') > new_wf_number:
                        wf.set('number', wf.get('number') - 1)
                # Since we reuse the old wavefunction, reset wf_number
                wf_number = wf_number - 1

                # Need to replace wavefunction in number_to_wavefunctions
                # (in case this wavefunction is in another of the dicts) 
                for n_to_wf_dict in number_to_wavefunctions:
                    if new_wf in n_to_wf_dict.values():
                        for key in n_to_wf_dict.keys():
                            if n_to_wf_dict[key] == new_wf:
                                n_to_wf_dict[key] = new_wf          
 
                if self.get('is_loop'):
                    # fix a bug for the g g > go go g [virt=QCD]
                    # when there is a wf which is replaced, we need to propagate
                    # the change in all wavefunction of that diagrams which could
                    # have this replaced wavefunction in their mothers. This
                    # plays the role of the 'number_to_wavefunction' dictionary
                    # used for tree level.
                    for wf in diagram_wavefunctions:
                        for i,mother_wf in enumerate(wf.get('mothers')):
                            if mother_wf.get('number')==new_wf_number:
                                wf.get('mothers')[i]=new_wf

            except ValueError:
                pass

        # Return the new (or old) wavefunction, and the new
        # wavefunction number
        return new_wf, wf_number

    def get_fermion_order(self):
        """Recursive function to get a list of fermion numbers
        corresponding to the order of fermions along fermion lines
        connected to this wavefunction, in the form [n1,n2,...] for a
        boson, and [N,[n1,n2,...]] for a fermion line"""

        # End recursion if external wavefunction
        if not self.get('mothers'):
            if self.is_fermion():
                return [self.get('number_external'), []]
            else:
                return []

        # Pick out fermion mother
        fermion_mother = None
        if self.is_fermion():
            fermion_mother = self.find_mother_fermion()

        other_fermions = [wf for wf in self.get('mothers') if \
                          wf.is_fermion() and wf != fermion_mother]

        # Pick out bosons
        bosons = filter(lambda wf: wf.is_boson(), self.get('mothers'))

        fermion_number_list = []

        if self.is_fermion():
            # Fermions return the result N from their mother
            # and the list from bosons, so [N,[n1,n2,...]]
            mother_list = fermion_mother.get_fermion_order()
            fermion_number_list.extend(mother_list[1])

        # If there are fermion line pairs, append them as
        # [NI,NO,n1,n2,...]
        fermion_numbers = [f.get_fermion_order() for f in other_fermions]
        for iferm in range(0, len(fermion_numbers), 2):
            fermion_number_list.append(fermion_numbers[iferm][0])
            fermion_number_list.append(fermion_numbers[iferm+1][0])
            fermion_number_list.extend(fermion_numbers[iferm][1])
            fermion_number_list.extend(fermion_numbers[iferm+1][1])

        for boson in bosons:
            # Bosons return a list [n1,n2,...]
            fermion_number_list.extend(boson.get_fermion_order())

        if self.is_fermion():
            return [mother_list[0], fermion_number_list]

        return fermion_number_list

    def needs_hermitian_conjugate(self):
        """Returns true if any of the mothers have negative
        fermionflow"""

        return self.get('conjugate_indices') != ()

    def get_with_flow(self, name):
        """Generate the is_part and state needed for writing out
        wavefunctions, taking into account the fermion flow"""

        if self.get('fermionflow') > 0:
            # Just return (spin, state)
            return self.get(name)

        # If fermionflow is -1, need to flip particle identity and state
        if name == 'is_part':
            return not self.get('is_part')
        if name == 'state':
            return filter(lambda state: state != self.get('state'),
                          ['incoming', 'outgoing'])[0]
        return self.get(name)
    
    def get_external_helas_call_dict(self):
        """ Returns a dictionary for formatting this external wavefunction
        helas call """
        
        if self['mothers']:
            raise MadGraph5Error, "This function should be called only for"+\
                                                    " external wavefunctions."
        return_dict = {}
        if self.get('is_loop'):
            return_dict['conjugate'] = ('C' if self.needs_hermitian_conjugate() \
                                                                        else '')
            return_dict['lcutspinletter'] = self.get_lcutspinletter()
        return_dict['number'] = self.get('number')
        return_dict['me_id'] = self.get('me_id')
        return_dict['number_external'] = self.get('number_external')
        return_dict['mass'] = self.get('mass')
        if self.is_boson():
            return_dict['state_id'] = (-1) ** (self.get('state') == 'initial')
        else:
            return_dict['state_id'] = -(-1) ** self.get_with_flow('is_part')
        return_dict['number_external'] = self.get('number_external')
        
        return return_dict

    def get_helas_call_dict(self, index=1, OptimizedOutput=False,
                                            specifyHel=True,**opt):
        """ return a dictionary to be used for formatting
        HELAS call. The argument index sets the flipping while optimized output
        changes the wavefunction specification in the arguments."""

        if index == 1:
            flip = 0
        else:
            flip = 1

        output = {}
        if self.get('is_loop') and OptimizedOutput:
            output['vertex_rank']=self.get_analytic_info('interaction_rank')
            output['lcut_size']=self.get('lcut_size')
            output['out_size']=self.spin_to_size(self.get('spin'))
        
        loop_mother_found=False
        for ind, mother in enumerate(self.get('mothers')):
            # temporary START
            # This temporary modification is only because aloha has the convention
            # of putting the loop polynomial mother wavefunction first in the 
            # list of argument of the helas call. I this convention is changed
            # to be the 'natural' order in the interaction, this would not be
            # needed anymore.
            if OptimizedOutput and self.get('is_loop'):
                if mother.get('is_loop'):
                    i=0
                else:
                    if loop_mother_found:
                        i=ind
                    else:
                        i=ind+1
            else:
                i=ind
            # temporary END
            nb = mother.get('me_id') - flip
            output[str(i)] = nb
            if not OptimizedOutput:
                if mother.get('is_loop'):
                    output['WF%d'%i] = 'L(1,%d)'%nb
                else:
                    output['WF%d'%i] = '(1,WE(%d)'%nb
            else:
                if mother.get('is_loop'):
                    output['loop_mother_number']=nb
                    output['loop_mother_rank']=\
                                   mother.get_analytic_info('wavefunction_rank')
                    output['in_size']=self.spin_to_size(mother.get('spin'))
                    output['WF%d'%i] = 'PL(0,%d)'%nb
                    loop_mother_found=True
                else:
                    output['WF%d'%i] = 'W(1,%d'%nb
            if not mother.get('is_loop'):
                if specifyHel:
                    output['WF%d'%i]=output['WF%d'%i]+',H)'
                else:
                    output['WF%d'%i]=output['WF%d'%i]+')'
                    
        #fixed argument
        for i, coup in enumerate(self.get_with_flow('coupling')):
            # We do not include the - sign in front of the coupling of loop
            # wavefunctions (only the loop ones, the tree ones are treated normally)
            # in the non optimized output because this sign was already applied to
            # the coupling passed in argument when calling the loop amplitude.
            if not OptimizedOutput and self.get('is_loop'):
                output['coup%d'%i] = coup[1:] if coup.startswith('-') else coup  
            else:
                output['coup%d'%i] = coup
              
        output['out'] = self.get('me_id') - flip
        output['M'] = self.get('mass')
        output['W'] = self.get('width')
        output['propa'] = self.get('particle').get('propagator')
        if output['propa'] not in ['', None]:
            output['propa'] = 'P%s' % output['propa']
        # optimization
        if aloha.complex_mass: 
            if (self.get('width') == 'ZERO' or self.get('mass') == 'ZERO'):
                #print self.get('width'), self.get('mass')
                output['CM'] = '%s' % self.get('mass') 
            else: 
                output['CM'] ='CMASS_%s' % self.get('mass')
        output.update(opt)
        return output
    
    def get_spin_state_number(self, flip=False):
        """Returns the number corresponding to the spin state, with a
        minus sign for incoming fermions. For flip=True, this 
        spin_state_number is suited for find the index in the interaction 
        of a MOTHER wavefunction. """

        state_number = {'incoming':-1 if not flip else 1,
                        'outgoing': 1 if not flip else -1,
                        'intermediate': 1, 'initial': 1, 'final': 1}
        return self.get('fermionflow') * \
                  state_number[self.get('state')] * \
                  self.get('spin')

    def find_mother_fermion(self):
        """Return the fermion mother which is fermion flow connected to
        this fermion"""

        if not self.is_fermion():
            return None

        part_number = self.find_outgoing_number()
        mother_number = (part_number-1)//2*2

        return HelasMatrixElement.sorted_mothers(self)[mother_number]

    def find_outgoing_number(self):
        "Return the position of the resulting particles in the interactions"
        # First shot: just the index in the interaction
        
        if self.get('interaction_id') == 0:
            return 0
        
        return self.find_leg_index(self.get_anti_pdg_code(),\
                                                   self.get_spin_state_number())
        
    def find_leg_index(self, pdg_code, spin_state):
        """ Find the place in the interaction list of the given particle with
        pdg 'pdg_code' and spin 'spin_stat'. For interactions with several identical particles (or 
        fermion pairs) the outgoing index is always the first occurence.
        """
        wf_indices = self.get('pdg_codes')
        wf_index = wf_indices.index(pdg_code)

        # If fermion, then we need to correct for I/O status
        if spin_state % 2 == 0:
            if wf_index % 2 == 0 and spin_state < 0:
                # Outgoing particle at even slot -> increase by 1
                wf_index += 1
            elif wf_index % 2 == 1 and spin_state > 0:
                # Incoming particle at odd slot -> decrease by 1
                wf_index -= 1
        return wf_index + 1
    
    def get_call_key(self):
        """Generate the (spin, number, C-state) tuple used as key for
        the helas call dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()
        res.append(self.get_spin_state_number())
        res.append(self.find_outgoing_number())

        if self['is_loop']:
            res.append(self.get_loop_index())
            if not self.get('mothers'):
                res.append(self.get('is_part'))

        # Check if we need to append a charge conjugation flag
        if self.needs_hermitian_conjugate():
            res.append(self.get('conjugate_indices'))

        return (tuple(res), tuple(self.get('lorentz')))

    def get_base_vertices(self, wf_dict, vx_list = [], optimization = 1):
        """Recursive method to get a base_objects.VertexList
        corresponding to this wavefunction and its mothers."""

        vertices = base_objects.VertexList()

        mothers = self.get('mothers')

        if not mothers:
            return vertices

        # Add vertices for all mothers
        for mother in mothers:
            # This is where recursion happens
            vertices.extend(mother.get_base_vertices(\
                                                wf_dict, vx_list,optimization))

        vertex = self.get_base_vertex(wf_dict, vx_list, optimization)

        try:
            index = vx_list.index(vertex)
            vertex = vx_list[index]
        except ValueError:
            pass
        
        vertices.append(vertex)

        return vertices

    def get_base_vertex(self, wf_dict, vx_list = [], optimization = 1):
        """Get a base_objects.Vertex corresponding to this
        wavefunction."""

        # Generate last vertex
        legs = base_objects.LegList()

        # We use the onshell flag to indicate whether this outgoing
        # leg corresponds to a decaying (onshell) particle, forbidden
        # s-channel, or regular
        try:
            if self.get('is_loop'):
                # Loop wavefunction should always be redefined
                raise KeyError
            lastleg = wf_dict[(self.get('number'),self.get('onshell'))]
        except KeyError:            
            lastleg = base_objects.Leg({
                'id': self.get_pdg_code(),
                'number': self.get('number_external'),
                'state': self.get('leg_state'),
                'onshell': self.get('onshell'),
                'loop_line':self.get('is_loop')
                })

            if optimization != 0 and not self.get('is_loop'):
                wf_dict[(self.get('number'),self.get('onshell'))] = lastleg

        for mother in self.get('mothers'):           
            try:
                if mother.get('is_loop'):
                # Loop wavefunction should always be redefined
                    raise KeyError
                leg = wf_dict[(mother.get('number'),False)]
            except KeyError:
                leg = base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'onshell': None,
                    'loop_line':mother.get('is_loop'),
                    'onshell': None
                    })
                if optimization != 0 and not mother.get('is_loop'):
                    wf_dict[(mother.get('number'),False)] = leg
            legs.append(leg)

        legs.append(lastleg)

        vertex = base_objects.Vertex({
            'id': self.get('interaction_id'),
            'legs': legs})

        return vertex

    def get_color_indices(self):
        """Recursive method to get the color indices corresponding to
        this wavefunction and its mothers."""

        if not self.get('mothers'):
            return []

        color_indices = []

        # Add color indices for all mothers
        for mother in self.get('mothers'):
            # This is where recursion happens
            color_indices.extend(mother.get_color_indices())
        # Add this wf's color index
        color_indices.append(self.get('color_key'))

        return color_indices
    
    def get_aloha_info(self, optimized_output=True):
        """Returns the tuple (lorentz_name, tag, outgoing_number) providing
        the necessary information to compute_subset of create_aloha to write
        out the HELAS-like routines."""
        
        # In principle this function should not be called for the case below,
        # or if it does it should handle specifically the None returned value.
        if self.get('interaction_id') in [0,-1]:
            return None

        tags = ['C%s' % w for w in self.get_conjugate_index()]
        if self.get('is_loop'): 
            if not optimized_output:
                tags.append('L')
            else:
                tags.append('L%d'%self.get_loop_index())

        if self.get('particle').get('propagator') not in ['', None]:
            tags.append('P%s' % str(self.get('particle').get('propagator')))

        return (tuple(self.get('lorentz')),tuple(tags),self.find_outgoing_number())

    def get_lcutspinletter(self):
        """Returns S,V or F depending on the spin of the mother loop particle.
        Return '' otherwise."""
        
        if self['is_loop'] and not self.get('mothers'):
            if self.get('spin') == 1:
                if self.get('particle').get('is_part'):
                    return 'S'
                else:
                    return 'AS'
            if self.get('spin') == 2:
                if self.get('particle').get('is_part'):
                    return 'F'
                else:
                    return 'AF'
            if self.get('spin') == 3:
                return 'V'
            else:
                raise MadGraph5Error,'L-cut particle type not supported'
        else:
            return ''

    def get_s_and_t_channels(self, ninitial, mother_leg, reverse_t_ch = False):
        """Returns two lists of vertices corresponding to the s- and
        t-channels that can be traced from this wavefunction, ordered 
        from the outermost s-channel and in/down towards the highest 
        (if not reverse_t_ch) or lowest (if reverse_t_ch) number initial 
        state leg. mother_leg corresponds to self but with
        correct leg number = min(final state mothers)."""

        schannels = base_objects.VertexList()
        tchannels = base_objects.VertexList()

        mother_leg = copy.copy(mother_leg)

        (startleg, finalleg) = (1,2)
        if reverse_t_ch: (startleg, finalleg) = (2,1)

        # Add vertices for all s-channel mothers
        final_mothers = filter(lambda wf: wf.get('number_external') > ninitial,
                               self.get('mothers'))

        for mother in final_mothers:
            schannels.extend(mother.get_base_vertices({}, optimization = 0))

        # Extract initial state mothers
        init_mothers = filter(lambda wf: wf.get('number_external') <= ninitial,
                              self.get('mothers'))

        assert len(init_mothers) < 3 , \
                   "get_s_and_t_channels can only handle up to 2 initial states"

        if len(init_mothers) == 1:
            # This is an s-channel or t-channel leg, or the initial
            # leg of a decay process. Add vertex and continue stepping
            # down towards external initial state
            legs = base_objects.LegList()
            mothers = final_mothers + init_mothers

            for mother in mothers:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'onshell': mother.get('onshell')
                    }))

            if init_mothers[0].get('number_external') == startleg and \
                   not init_mothers[0].get('leg_state') and ninitial > 1:
                # If this is t-channel going towards external leg 1,
                # mother_leg is resulting wf
                legs.append(mother_leg)
            else:
                # For decay processes or if init_mother is an s-channel leg
                # or we are going towards external leg 2, mother_leg
                # is one of the mothers (placed next-to-last)
                legs.insert(-1, mother_leg)
                # Need to switch direction of the resulting s-channel
                legs[-1].set('id', init_mothers[0].get_anti_pdg_code())
                
            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

            vertex = base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs})

            # Add s- and t-channels from init_mother
            new_mother_leg = legs[-1]
            if init_mothers[0].get('number_external') == startleg and \
                   not init_mothers[0].get('leg_state') and \
                   ninitial > 1:
                # Mother of next vertex is init_mothers[0]
                # (next-to-last in legs)
                new_mother_leg = legs[-2]

            mother_s, tchannels = \
                      init_mothers[0].get_s_and_t_channels(ninitial,
                                                           new_mother_leg,
                                                           reverse_t_ch)
            if ninitial == 1 or init_mothers[0].get('leg_state') == True:
                # This vertex is s-channel
                schannels.append(vertex)
            elif init_mothers[0].get('number_external') == startleg:
                # If init_mothers is going towards external leg 1, add
                # to t-channels, at end
                tchannels.append(vertex)
            else:
                # If init_mothers is going towards external leg 2, add to
                # t-channels, at start
                tchannels.insert(0, vertex)

            schannels.extend(mother_s)

        elif len(init_mothers) == 2:
            # This is a t-channel junction. Start with the leg going
            # towards external particle 1, and then do external
            # particle 2
            init_mothers1 = filter(lambda wf: wf.get('number_external') == \
                                   startleg,
                                   init_mothers)[0]
            init_mothers2 = filter(lambda wf: wf.get('number_external') == \
                                   finalleg,
                                   init_mothers)[0]

            # Create vertex
            legs = base_objects.LegList()
            for mother in final_mothers + [init_mothers1, init_mothers2]:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'onshell': mother.get('onshell')
                    }))
            legs.insert(0, mother_leg)

            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

            vertex = base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs})

            # Add s- and t-channels going down towards leg 1
            mother_s, tchannels = \
                      init_mothers1.get_s_and_t_channels(ninitial, legs[-2],
                                                         reverse_t_ch)
            schannels.extend(mother_s)

            # Add vertex
            tchannels.append(vertex)

            # Add s- and t-channels going down towards leg 2
            mother_s, mother_t = \
                      init_mothers2.get_s_and_t_channels(ninitial, legs[-1],
                                                         reverse_t_ch)
            schannels.extend(mother_s)
            tchannels.extend(mother_t)

        # Sort s-channels according to number
        schannels.sort(lambda x1,x2: x2.get('legs')[-1].get('number') - \
                       x1.get('legs')[-1].get('number'))

        return schannels, tchannels

    def get_struct_external_leg_ids(self):
        """ Return a set containing the ids of all the non-loop outter-most
        external legs attached to the loop at the interaction point of this 
        loop wavefunction """
        
        if not self.get('mothers'):
            return set([self.get('number_external'),])

        res=set([])
        for wf in self.get('mothers'):
            if not wf['is_loop']:
                res=res.union(wf.get_struct_external_leg_ids())
        return res
#
    def get_loop_index(self):
        """Return the index of the wavefunction in the mothers which is the 
        loop one"""
        
        if not self.get('mothers'):
            return 0
        
        try:
            loop_wf_index=\
                       [wf['is_loop'] for wf in self.get('mothers')].index(True)
        except ValueError:
            raise MadGraph5Error, "The loop wavefunctions should have exactly"+\
                                                " one loop wavefunction mother."

        if self.find_outgoing_number()-1<=loop_wf_index:
            # If the incoming loop leg is placed after the outgoing one we
            # need to increment once more its index in the interaction list (
            # because the outgoing loop leg is not part of the mother wf list)
            return loop_wf_index+2
        else:
            # Basic increment of +1 because aloha counts particles in the 
            # interaction starting at 1.
            return loop_wf_index+1

    def get_lcut_size(self):
        """ Return the size (i.e number of elements) of the L-Cut wavefunction
        this loop wavefunction originates from. """
        
        if not self['is_loop']:
            return 0
        
        # Obtain the L-cut wavefunction this loop wavefunction comes from.
        # (I'm using two variable instead of one in order to have only one call
        #  to get_loop_mother())
        last_loop_wf=self
        last_loop_wf_loop_mother=last_loop_wf.get_loop_mother()
        while last_loop_wf_loop_mother:
            last_loop_wf=last_loop_wf_loop_mother
            last_loop_wf_loop_mother=last_loop_wf_loop_mother.get_loop_mother()
        
        # Translate its spin into a wavefunction size.
        return self.spin_to_size(last_loop_wf.get('spin'))
        
    def get_loop_mother(self):
        """ Return the mother of type 'loop', if any. """
        
        if not self.get('mothers'):
            return None
        loop_wfs=[wf for wf in self.get('mothers') if wf['is_loop']]
        if loop_wfs:
            if len(loop_wfs)==1:
                return loop_wfs[0]
            else:
                raise MadGraph5Error, "The loop wavefunction must have either"+\
                  " no mothers, or exactly one mother with type 'loop'."
        else:
            return None
        
    def get_conjugate_index(self):
        """Return the index of the particle that should be conjugated."""

        if not any([(wf.get('fermionflow') < 0 or wf.is_majorana()) for wf in \
                    self.get('mothers')]) and \
                    (not self.get('interaction_id') or \
                    self.get('fermionflow') >= 0):
            return ()
        
        # Pick out first sorted mothers, then fermions
        mothers, self_index = \
                      self.get('mothers').sort_by_pdg_codes(self.get('pdg_codes'),
                                                            self.get_anti_pdg_code())
        fermions = HelasWavefunctionList([wf for wf in mothers if wf.is_fermion()])

        # Insert this wavefunction in list (in the right place)
        if self.is_fermion():
            me = copy.copy(self)
            # Flip incoming/outgoing to make me equivalent to mother
            # as needed by majorana_conjugates
            me.set('state', [state for state in ['incoming', 'outgoing'] \
                             if state != me.get('state')][0])
            fermions.insert(self_index, me)

        # Initialize indices with indices due to Majoranas with wrong order
        indices = fermions.majorana_conjugates()

        # Check for fermions with negative fermion flow
        for i in range(0,len(fermions), 2):
            if fermions[i].get('fermionflow') < 0 or \
               fermions[i+1].get('fermionflow') < 0:
                indices.append(i/2 + 1)

        return tuple(sorted(indices))

    def get_vertex_leg_numbers(self, 
              veto_inter_id=base_objects.Vertex.ID_to_veto_for_multichanneling,
              max_n_loop=0):
        """Get a list of the number of legs in vertices in this diagram"""

        if not self.get('mothers'):
            return []

        if max_n_loop == 0:
            max_n_loop = base_objects.Vertex.max_n_loop_for_multichanneling

        vertex_leg_numbers = [len(self.get('mothers')) + 1] if \
            (self.get('interaction_id') not in veto_inter_id) or\
            (self.get('interaction_id')==-2 and len(self.get('mothers'))+1 > 
                                                             max_n_loop) else []
        for mother in self.get('mothers'):
            vertex_leg_numbers.extend(mother.get_vertex_leg_numbers(
                                                 veto_inter_id = veto_inter_id))

        return vertex_leg_numbers

    # Overloaded operators

    def __eq__(self, other):
        """Overloading the equality operator, to make comparison easy
        when checking if wavefunction is already written, or when
        checking for identical processes. Note that the number for
        this wavefunction, the pdg code, and the interaction id are
        irrelevant, while the numbers for the mothers are important.
        """

        if not isinstance(other, HelasWavefunction):
            return False

        # Check relevant directly defined properties
        if self['number_external'] != other['number_external'] or \
           self['fermionflow'] != other['fermionflow'] or \
           self['color_key'] != other['color_key'] or \
           self['lorentz'] != other['lorentz'] or \
           self['coupling'] != other['coupling'] or \
           self['state'] != other['state'] or \
           self['onshell'] != other['onshell'] or \
           self.get('spin') != other.get('spin') or \
           self.get('self_antipart') != other.get('self_antipart') or \
           self.get('mass') != other.get('mass') or \
           self.get('width') != other.get('width') or \
           self.get('color') != other.get('color') or \
           self['decay'] != other['decay'] or \
           self['decay'] and self['particle'] != other['particle']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return sorted([mother['number'] for mother in self['mothers']]) == \
               sorted([mother['number'] for mother in other['mothers']])

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

#===============================================================================
# Start of the legacy of obsolete functions of the HelasWavefunction class.
#===============================================================================

    def LEGACY_get_interaction_q_power(self):
        """ Returns the power of the loop momentum q brought by the interaction
        and propagator from which this loop wavefunction originates. This 
        is done in a SM ad-hoc way, but it should be promoted to be general in 
        the future, by reading the lorentz structure of the interaction.
        This function is now rendered obsolete by the use of the function
        get_analytical_info. It is however kept for legacy."""
        rank=0
        # First add the propagator power for a fermion of spin 1/2.
        # For the bosons, it is assumed to be in Feynman gauge so that the
        # propagator does not bring in any power of the loop momentum.
        if self.get('spin')==2:
            rank=rank+1

        # Treat in an ad-hoc way the higgs effective theory
        spin_cols = [(self.get('spin'),abs(self.get('color')))]+\
              [(w.get('spin'),abs(w.get('color'))) for w in self.get('mothers')]
        # HGG effective vertex
        if sorted(spin_cols) == sorted([(1,1),(3,8),(3,8)]):
            return rank+2
        # HGGG effective vertex
        if sorted(spin_cols) == sorted([(1,1),(3,8),(3,8),(3,8)]):
            return rank+1
        # HGGGG effective vertex
        if sorted(spin_cols) == sorted([(1,1),(3,8),(3,8),(3,8),(3,8)]):
            return rank
            
        # Now add a possible power of the loop momentum depending on the
        # vertex creating this loop wavefunction. For now we don't read the
        # lorentz structure but just use an SM ad-hoc rule that only 
        # the feynman rules for a three point vertex with only bosons bring
        # in one power of q.
        if self.is_boson() and len([w for w in self.get('mothers') \
                                                           if w.is_boson()])==2:
            rank=rank+1
        return rank
    
#===============================================================================
# End of the legacy of obsolete functions of the HelasWavefunction class.
#===============================================================================

#===============================================================================
# HelasWavefunctionList
#===============================================================================
class HelasWavefunctionList(base_objects.PhysicsObjectList):
    """List of HelasWavefunction objects. This class has the routine
    check_and_fix_fermion_flow, which checks for fermion flow clashes
    among the mothers of an amplitude or wavefunction.
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasWavefunction for the list."""

        return isinstance(obj, HelasWavefunction)

    # Helper functions

    def to_array(self):
        return array.array('i', [w['number'] for w in self])

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   my_wf,
                                   wf_number,
                                   force_flip_flow=False,
                                   number_to_wavefunctions=[]):

        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)). If found, we need to trace back through the
        mother structure (only looking at fermions), until we find a
        Majorana fermion. Then flip fermion flow along this line all
        the way from the initial clash to the external fermion (in the
        right way, see check_majorana_and_flip_flow), and consider an
        incoming particle with fermionflow -1 as outgoing (and vice
        versa). Continue until we have N(incoming) = N(outgoing).
        
        Since the wavefunction number might get updated, return new
        wavefunction number.
        """

        # Clash is defined by whether any of the fermion lines are clashing
        fermion_mother = None

        # Keep track of clashing fermion wavefunctions
        clashes = []
        
        # First check the fermion mother on the same fermion line
        if my_wf and my_wf.is_fermion():
            fermion_mother = my_wf.find_mother_fermion()
            if my_wf.get_with_flow('state') != \
                   fermion_mother.get_with_flow('state'):
                clashes.append([fermion_mother])

        # Now check all other fermions
        other_fermions = [w for w in self if \
                          w.is_fermion() and w != fermion_mother]

        for iferm in range(0, len(other_fermions), 2):
            if other_fermions[iferm].get_with_flow('state') == \
               other_fermions[iferm+1].get_with_flow('state'):
                clashes.append([other_fermions[iferm],
                                other_fermions[iferm+1]])

        if not clashes:
            return wf_number

        # If any of the mothers have negative fermionflow, we need to
        # take this mother first.
        for clash in clashes:
            neg_fermionflow_mothers = [m for m in clash if \
                                       m.get('fermionflow') < 0]

            if not neg_fermionflow_mothers:
                neg_fermionflow_mothers = clash

            for mother in neg_fermionflow_mothers:

                # Call recursive function to check for Majorana fermions
                # and flip fermionflow if found

                found_majorana = False
                state_before = mother.get_with_flow('state')
                new_mother, wf_number = mother.check_majorana_and_flip_flow(\
                                                    found_majorana,
                                                    wavefunctions,
                                                    diagram_wavefunctions,
                                                    external_wavefunctions,
                                                    wf_number,
                                                    force_flip_flow,
                                                    number_to_wavefunctions)

                if new_mother.get_with_flow('state') == state_before:
                    # Fermion flow was not flipped, try next mother
                    continue

                # Replace old mother with new mother
                mother_index = self.index(mother)
                self[self.index(mother)] = new_mother
                clash_index = clash.index(mother)
                clash[clash.index(mother)] = new_mother

                # Fermion flow was flipped, abort loop
                break
            
            if len(clash) == 1 and clash[0].get_with_flow('state') != \
                   my_wf.get_with_flow('state') or \
                   len(clash) == 2 and clash[0].get_with_flow('state') == \
                   clash[1].get_with_flow('state'):
                # No Majorana fermion in any relevant legs - try again,
                # but simply use the first relevant leg
                force_flip_flow = True
                wf_number = self.check_and_fix_fermion_flow(\
                                       wavefunctions,
                                       diagram_wavefunctions,
                                       external_wavefunctions,
                                       my_wf,
                                       wf_number,
                                       force_flip_flow,
                                       number_to_wavefunctions)
                # Already ran for all clashes, abort loop
                break

        return wf_number

    def insert_own_mothers(self):
        """Recursively go through a wavefunction list and insert the
        mothers of all wavefunctions, return the result.
        Assumes that all wavefunctions have unique numbers."""

        res = copy.copy(self)
        # Recursively build up res
        for wf in self:
            index = res.index(wf)
            res = res[:index] + wf.get('mothers').insert_own_mothers() \
                  + res[index:]

        # Make sure no wavefunctions occur twice, by removing doublets
        # from the back
        i = len(res) - 1
        while res[:i]:
            if res[i].get('number') in [w.get('number') for w in res[:i]]:
                res.pop(i)
            i = i - 1

        return res

    def sort_by_pdg_codes(self, pdg_codes, my_pdg_code = 0):
        """Sort this HelasWavefunctionList according to the cyclic
        order of the pdg codes given. my_pdg_code is the pdg code of
        the daughter wavefunction (or 0 if daughter is amplitude)."""

        if not pdg_codes:
            return self, 0

        pdg_codes = copy.copy(pdg_codes)

        # Remove the argument wavefunction code from pdg_codes

        my_index = -1
        if my_pdg_code:
            # Remember index of my code
            my_index = pdg_codes.index(my_pdg_code)
            pdg_codes.pop(my_index)
        
        mothers = copy.copy(self)
        # Sort according to interaction pdg codes

        mother_codes = [ wf.get_pdg_code() for wf \
                         in mothers ]    
        if pdg_codes == mother_codes:
            # Already sorted - skip sort below
            return mothers, my_index

        sorted_mothers = []
        for i, code in enumerate(pdg_codes):
            index = mother_codes.index(code)
            mother_codes.pop(index)
            mother = mothers.pop(index)
            sorted_mothers.append(mother)

        if mothers:
            raise base_objects.PhysicsObject.PhysicsObjectError

        return HelasWavefunctionList(sorted_mothers), my_index

    def majorana_conjugates(self):
        """Returns a list [1,2,...] of fermion lines that need
         conjugate wfs due to wrong order of I/O Majorana particles
         compared to interaction order (or empty list if no Majorana
         particles).  This is crucial if the Lorentz structure depends
         on the direction of the Majorana particles, as in MSSM with
         goldstinos."""

        if len([m for m in self if m.is_majorana()]) < 2:
            return []

        conjugates = []
        
        # Check if the order for Majorana fermions is correct
        for i in range(0, len(self), 2):
            if self[i].is_majorana() and self[i+1].is_majorana() \
                   and self[i].get_pdg_code() != \
                   self[i+1].get_pdg_code():
                # Check if mother I/O order is correct (IO)
                if self[i].get_spin_state_number() > 0 and \
                   self[i + 1].get_spin_state_number() < 0:
                    # Order is wrong, we need a conjugate here
                    conjugates.append(True)
                else:
                    conjugates.append(False)
            elif self[i].is_fermion():
                # For non-Majorana case, always False
                conjugates.append(False)

        # Return list 1,2,... for which indices are needed
        conjugates = [i+1 for (i,c) in enumerate(conjugates) if c]

        return conjugates

    
    def check_wavefunction_numbers_order(self, applyChanges=False, raiseError=True):
        """ This function only serves as an internal consistency check to 
        make sure that when setting the 'wavefunctions' attribute of the
        diagram, their order is consistent, in the sense that all mothers 
        of any given wavefunction appear before that wavefunction.
        This function returns True if there was no change and the original 
        wavefunction list was consistent and False otherwise.
        The option 'applyChanges' controls whether the function should substitute
        the original list (self) with the new corrected one. For now, this function
        is only used for self-consistency checks and the changes are not applied."""
    
        if len(self)<2:
            return True
    
        def RaiseError():
            raise self.PhysicsObjectListError, \
      "This wavefunction list does not have a consistent wavefunction ordering."+\
      "\n  Wf numbers: %s"%str([wf['number'] for wf in diag_wfs])+\
      "\n  Wf mothers: %s"%str([[mother['number'] for mother in wf['mothers']] \
                                                  for wf in diag_wfs])
    
        # We want to work on a local copy of the wavefunction list attribute
        diag_wfs = copy.copy(self)
        
        # We want to keep the original wf numbering (but beware that this
        # implies changing the 'number' attribute of some wf if this function
        # was used for actual reordering and not just self-consistency check)
        wfNumbers = [wf['number'] for wf in self]
        
        exitLoop=False
        while not exitLoop:
            for i, wf in enumerate(diag_wfs):
                if i==len(diag_wfs)-1:
                    exitLoop=True
                    break
                found=False
                # Look at all subsequent wfs in the list placed after wf at 
                # index i. None of them should have wf as its mother
                for w in diag_wfs[i+1:]:
                    if w['number'] in [mwf['number'] for mwf in wf.get('mothers')]:
                        # There is an inconsisent order so we must move this
                        # mother w *before* wf which is placed at i.
                        diag_wfs.remove(w)
                        diag_wfs.insert(i,w)
                        found=True
                        if raiseError: RaiseError()
                        if not applyChanges:
                            return False
                        break
                if found:
                    break
    
        if diag_wfs!=self:
            # After this, diag_wfs is the properly re-ordered and
            # consistent list that should be used, where each mother appear
            # before its daughter
            for i,wf in enumerate(diag_wfs):
                wf.set('number', wfNumbers[i])
            
            # Replace this wavefunction list by corrected one
            del self[:]
            self.extend(diag_wfs)
            
            # The original list was inconsistent, so it returns False.
            return False
        
        # The original list was consistent, so it returns True
        return True
    
    @staticmethod
    def extract_wavefunctions(mothers):
        """Recursively extract the wavefunctions from mothers of mothers"""

        wavefunctions = copy.copy(mothers)
        for wf in mothers:
            wavefunctions.extend(HelasWavefunctionList.\
                                 extract_wavefunctions(wf.get('mothers')))

        return wavefunctions

#===============================================================================
# HelasAmplitude
#===============================================================================
class HelasAmplitude(base_objects.PhysicsObject):
    """HelasAmplitude object, has the information necessary for
    writing a call to a HELAS amplitude routine:a list of mother wavefunctions,
    interaction id, amplitude number
    """

    def default_setup(self):
        """Default values for all properties"""

        # Properties related to the interaction generating the propagator
        self['interaction_id'] = 0
        # Base for born amplitude, the 'type' argument for the CT-vertices
        # and 'loop' for the HelasAmplitudes in a LoopHelasAmplitude.
        self['type'] = 'base'
        self['pdg_codes'] = []
        self['orders'] = {}
        self['inter_color'] = None
        self['lorentz'] = []
        self['coupling'] = ['none']
        # The Lorentz and color index used in this amplitude
        self['color_key'] = 0
        # Properties relating to the vertex
        self['number'] = 0
        self['fermionfactor'] = 0
        self['color_indices'] = []
        self['mothers'] = HelasWavefunctionList()
        # conjugate_indices is a list [1,2,...] with fermion lines
        # that need conjugates. Default is "None"
        self['conjugate_indices'] = None

    # Customized constructor
    def __init__(self, *arguments):
        """Allow generating a HelasAmplitude from a Vertex
        """

        if len(arguments) > 1:
            if isinstance(arguments[0], base_objects.Vertex) and \
               isinstance(arguments[1], base_objects.Model):
                super(HelasAmplitude, self).__init__()
                self.set('interaction_id',
                         arguments[0].get('id'), arguments[1])
        elif arguments:
            super(HelasAmplitude, self).__init__(arguments[0])
        else:
            super(HelasAmplitude, self).__init__()

    def filter(self, name, value):
        """Filter for valid property values."""

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for interaction id" % \
                        str(value)

        if name == 'pdg_codes':
            #Should be a list of integers
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name == 'orders':
            #Should be a dict with valid order names ask keys and int as values
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict for coupling orders" % \
                                                                    str(value)
            for order in value.keys():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(order)
                if not isinstance(value[order], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value[order])

        if name == 'inter_color':
            # Should be None or a color string
            if value and not isinstance(value, color.ColorString):
                    raise self.PhysicsObjectError, \
                            "%s is not a valid Color String" % str(value)

        if name == 'lorentz':
            #Should be a list of string
            if not isinstance(value, list):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid list of string" % str(value)
            for name in value:
                if not isinstance(name, str):
                    raise self.PhysicsObjectError, \
                        "%s doesn't contain only string" % str(value)
                        
        if name == 'coupling':
            #Should be a list of string
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                      "%s is not a valid coupling (list of string)" % str(value)
            
            for name in value:
                if not isinstance(name, str):
                    raise self.PhysicsObjectError, \
                        "%s doesn't contain only string" % str(value)
            if not len(value):
                raise self.PhysicsObjectError, \
                                      'coupling should have at least one value'

        if name == 'color_key':
            if value and not isinstance(value, int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid integer" % str(value)

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for amplitude number" % \
                        str(value)

        if name == 'fermionfactor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for fermionfactor" % \
                        str(value)
            if not value in [-1, 0, 1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermion factor (-1, 0 or 1)" % \
                        str(value)

        if name == 'color_indices':
            #Should be a list of integers
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for amplitude" % \
                      str(value)

        if name == 'conjugate_indices':
            if not isinstance(value, tuple) and value != None:
                raise self.PhysicsObjectError, \
                        "%s is not a valid tuple" % str(value) + \
                        " for conjugate_indices"

        return True

    def __str__(self):
        """ practicle way to represent an HelasAmplitude"""
        
        mystr = '{\n'
        for prop in self.get_sorted_keys():
            if isinstance(self[prop], str):
                mystr = mystr + '    \'' + prop + '\': \'' + \
                        self[prop] + '\',\n'
            elif isinstance(self[prop], float):
                mystr = mystr + '    \'' + prop + '\': %.2f,\n' % self[prop]
            elif isinstance(self[prop], int):
                mystr = mystr + '    \'' + prop + '\': %s,\n' % self[prop]                
            elif prop != 'mothers':
                mystr = mystr + '    \'' + prop + '\': ' + \
                       str(self[prop]) + ',\n'
            else:
                info = [m.get('pdg_code') for m in self['mothers']]
                mystr += '    \'%s\': %s,\n' % (prop, info) 
                
        mystr = mystr.rstrip(',\n')
        mystr = mystr + '\n}'

        return mystr

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'fermionfactor' and not self[name]:
            self.calculate_fermionfactor()

        # Set conjugate_indices if it's not already set
        if name == 'conjugate_indices' and self[name] == None:
            self['conjugate_indices'] = self.get_conjugate_index()

        return super(HelasAmplitude, self).get(name)

    # Enhanced set function, where we can append a model

    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        assert len(arguments) > 1, "Too few arguments for set"

        name = arguments[0]
        value = arguments[1]

        if len(arguments) > 2 and \
               isinstance(value, int) and \
               isinstance(arguments[2], base_objects.Model):
            if name == 'interaction_id':
                self.set('interaction_id', value)
                if value > 0:
                    inter = arguments[2].get('interaction_dict')[value]
                    self.set('pdg_codes',
                             [part.get_pdg_code() for part in \
                              inter.get('particles')])
                    self.set('orders', inter.get('orders'))
                    # Note that the following values might change, if
                    # the relevant color/lorentz/coupling is not index 0
                    if inter.get('type'):
                        self.set('type', inter.get('type'))
                    if inter.get('color'):
                        self.set('inter_color', inter.get('color')[0])
                    if inter.get('lorentz'):
                        self.set('lorentz', [inter.get('lorentz')[0]])
                    if inter.get('couplings'):
                        self.set('coupling', [inter.get('couplings').values()[0]])
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasAmplitude, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['interaction_id', 'pdg_codes', 'orders', 'inter_color', 
                'lorentz', 'coupling', 'color_key', 'number', 'color_indices',
                'fermionfactor', 'mothers']

    # Helper functions

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   wf_number):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers. For documentation, check
        HelasWavefunction.check_and_fix_fermion_flow.
        """

        self.set('mothers', self.get('mothers').sort_by_pdg_codes(\
            self.get('pdg_codes'), 0)[0])
                 
        return self.get('mothers').check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   None,
                                   wf_number)


    def needs_hermitian_conjugate(self):
        """Returns true if any of the mothers have negative
        fermionflow"""

        return self.get('conjugate_indices') != ()

    def get_epsilon_order(self):
        """Based on the type of the amplitude, determines to which epsilon
        order it contributes"""
        
        if '1eps' in self['type']:
            return 1
        elif '2eps' in self['type']:
            return 2
        else:
            return 0

    def get_call_key(self):
        """Generate the (spin, state) tuples used as key for the helas call
        dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        # The call is different depending on the type of vertex. 
        # For example, base would give AMP(%d), R2 would give AMPL(0,%d)
        # and a single pole UV counter-term would give AMPL(1,%d).
        # Also for loop amplitudes, one must have the tag 'loop'
        if self['type']!='base':
            res.append(self['type'])

        # Check if we need to append a charge conjugation flag
        if self.needs_hermitian_conjugate():
            res.append(self.get('conjugate_indices'))

        return (tuple(res), tuple(self.get('lorentz')))

    def calculate_fermionfactor(self):
        """Calculate the fermion factor for the diagram corresponding
        to this amplitude"""

        # Pick out fermion mothers
        fermions = [wf for wf in self.get('mothers') if wf.is_fermion()]
        assert len(fermions) % 2 == 0

        # Pick out bosons
        bosons = filter(lambda wf: wf.is_boson(), self.get('mothers'))

        fermion_number_list = []

        # If there are fermion line pairs, append them as
        # [NI,NO,n1,n2,...]
        fermion_numbers = [f.get_fermion_order() for f in fermions]

        # Apply the right sign correction for anti-commutating ghost loops
        if self.get('type')=='loop':
            # Fetch the second l-cut wavefunctions
            lcuf_wf_2=[m for m in self.get('mothers') if m['is_loop'] and \
                                                    len(m.get('mothers'))==0][0]
            ghost_factor = -1 if lcuf_wf_2.is_anticommutating_ghost() else 1
        else:
            # no ghost at tree level
            ghost_factor = 1

        fermion_loop_factor = 1

        # Now put together the fermion line merging in this amplitude
        if self.get('type')=='loop' and len(fermion_numbers)>0:
            # Remember that the amplitude closing the loop is always a 2-point
            # "fake interaction" attached on the second l-cut wavefunction.
            # So len(fermion_numbers) is either be 0 or 2.
            lcut_wf2_number = lcuf_wf_2.get('number_external')
            assert len(fermion_numbers)==2, "Incorrect number of fermions"+\
                " (%d) for the amp. closing the loop."%len(fermion_numbers)
            # Fetch the first l-cut wavefunctions
            lcuf_wf_1=[m for m in self.get('mothers') if m['is_loop'] and \
                                                     len(m.get('mothers'))>0][0]
            while len(lcuf_wf_1.get('mothers'))>0:
                lcuf_wf_1 = lcuf_wf_1.get_loop_mother()
            lcut_wf1_number = lcuf_wf_1.get('number_external')
            
            # We must now close the loop fermion flow, if there is any.
            # This means merging the two lists representing the fermion flow of
            # each of the two l-cut fermions into one. Example for the process
            # g g > go go [virt=QCD] in the MSSM.
            # Loop diagram 21 has the fermion_number_list
            # [[3, [5, 4]], [6, []]]
            # and 22 has 
            # [[6, []], [4, [3, 5]]]
            # Which should be merged into [3,4] both times
            
            # Here, iferm_to_replace is the position of the fermion line 
            # pairing which is *not* [6,[]] in the above example.
            iferm_to_replace = (fermion_numbers.index([lcut_wf2_number,[]])+1)%2
        
            if fermion_numbers[iferm_to_replace][0]==lcut_wf1_number:
                # We have a closed loop fermion flow here, so we must simply 
                # add a minus sign (irrespectively of whether the closed loop 
                # fermion flow goes clockwise or counter-clockwise) and not
                # consider the fermion loop line in the fermion connection list.
                fermion_number_list.extend(fermion_numbers[iferm_to_replace][1])
                fermion_loop_factor = -1
            else:
                # The fermion flow escape the loop in this case.
                fermion_number_list = \
                                 copy.copy(fermion_numbers[iferm_to_replace][1])
                # We must find to which external fermion the lcut_wf1 is 
                # connected (i.e. 5 being connected to 3(resp. 4) in the example 
                # of diagram 22 (resp. 21) above)
                i_connected_fermion = fermion_number_list.index(lcut_wf1_number)
                fermion_number_list[i_connected_fermion] = \
                                            fermion_numbers[iferm_to_replace][0]
        else:
            for iferm in range(0, len(fermion_numbers), 2):
                fermion_number_list.append(fermion_numbers[iferm][0])
                fermion_number_list.append(fermion_numbers[iferm+1][0])
                fermion_number_list.extend(fermion_numbers[iferm][1])
                fermion_number_list.extend(fermion_numbers[iferm+1][1])
                
                
        # Bosons are treated in the same way for a bosonic loop than for tree
        # level kind of amplitudes.
        for boson in bosons:
            # Bosons return a list [n1,n2,...]
            fermion_number_list.extend(boson.get_fermion_order())

#         if not hasattr(HelasAmplitude,"counter"):
#             HelasAmplitude.counter=1
#             print "MMMMME"
#         save1 = copy.deepcopy(fermion_number_list)
#         save2 = copy.deepcopy(fermion_number_list2)
#         save3 = copy.deepcopy(fermion_number_list)
#         save4 = copy.deepcopy(fermion_number_list2)
#         if HelasAmplitude.counter<500000 and self.get('type')=='loop' and \
#           HelasAmplitude.sign_flips_to_order(save1)*HelasAmplitude.sign_flips_to_order(save2)==-1:
#             print "Before %i=%s"%(HelasAmplitude.counter,str(fermion_numbers_save))
#             print "FOOOOR %i=%s"%(HelasAmplitude.counter,str(fermion_number_list))
#             print "NEW %i=%s"%(HelasAmplitude.counter,str(fermion_number_list2))
#             print "Relative sign =%d"%(HelasAmplitude.sign_flips_to_order(save3)*HelasAmplitude.sign_flips_to_order(save4))
#         HelasAmplitude.counter=self.counter+1

        #fermion_number_list = fermion_number_list2

        fermion_factor = HelasAmplitude.sign_flips_to_order(fermion_number_list)

        self['fermionfactor'] = fermion_factor*ghost_factor*fermion_loop_factor
#        print "foooor %i ="%HelasAmplitude.counter, fermion_factor, self.get('type')

    @staticmethod
    def sign_flips_to_order(fermions):
        """Gives the sign corresponding to the number of flips needed
        to place the fermion numbers in order"""

        # Perform bubble sort on the fermions, and keep track of
        # the number of flips that are needed

        nflips = 0

        for i in range(len(fermions) - 1):
            for j in range(i + 1, len(fermions)):
                if fermions[j] < fermions[i]:
                    fermions[i], fermions[j] = fermions[j], fermions[i]
                    nflips = nflips + 1

        return (-1) ** nflips

    def get_aloha_info(self, optimized_output=True):
        """Returns the tuple (lorentz_name, tag, outgoing_number) providing
        the necessary information to compute_subset of create_aloha to write
        out the HELAS-like routines."""
        
        # In principle this function should not be called for the case below,
        # or if it does it should handle specifically the None returned value.
        if self.get('interaction_id') in [0,-1]:
            return None

        tags = ['C%s' % w for w in self.get_conjugate_index()]

        return (tuple(self.get('lorentz')),tuple(tags),self.find_outgoing_number())

    def get_base_diagram(self, wf_dict, vx_list = [], optimization = 1):
        """Return the base_objects.Diagram which corresponds to this
        amplitude, using a recursive method for the wavefunctions."""

        vertices = base_objects.VertexList()

        # Add vertices for all mothers
        for mother in self.get('mothers'):
            vertices.extend(mother.get_base_vertices(wf_dict, vx_list,
                                                     optimization))
        # Generate last vertex
        vertex = self.get_base_vertex(wf_dict, vx_list, optimization)

        vertices.append(vertex)

        return base_objects.Diagram({'vertices': vertices})

    def get_base_vertex(self, wf_dict, vx_list = [], optimization = 1):
        """Get a base_objects.Vertex corresponding to this amplitude."""
                
        # Generate last vertex
        legs = base_objects.LegList()
        for mother in self.get('mothers'):
            try:
                if mother.get('is_loop'):
                    # Loop wavefunction should always be redefined
                    raise KeyError
                leg = wf_dict[(mother.get('number'),False)]
            except KeyError:
                leg = base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'onshell': None,
                    'loop_line':mother.get('is_loop')
                    })
                if optimization != 0 and not mother.get('is_loop'):
                    wf_dict[(mother.get('number'),False)] = leg
            legs.append(leg)

        return base_objects.Vertex({
            'id': self.get('interaction_id'),
            'legs': legs})

    def get_s_and_t_channels(self, ninitial, model, new_pdg, reverse_t_ch = False):
        """Returns two lists of vertices corresponding to the s- and
        t-channels of this amplitude/diagram, ordered from the outermost
        s-channel and in/down towards the highest number initial state
        leg."""

        # Create a CanonicalConfigTag to ensure that the order of
        # propagators is canonical
        wf_dict = {}
        max_final_leg = 2
        if reverse_t_ch:
            max_final_leg = 1
        # Note that here we do not specify a FDStructure repository, so that
        # each loop diagram will recreate them. This is ok at this point because
        # we do not need to have a canonical ID for the FD structures.
        tag = CanonicalConfigTag(self.get_base_diagram(wf_dict).
                                      get_contracted_loop_diagram(model), model)

        return tag.get_s_and_t_channels(ninitial, model, new_pdg, max_final_leg)


    def get_color_indices(self):
        """Get the color indices corresponding to
        this amplitude and its mothers, using a recursive function."""

        if not self.get('mothers'):
            return []

        color_indices = []

        # Add color indices for all mothers
        for mother in self.get('mothers'):
            # This is where recursion happens
            color_indices.extend(mother.get_color_indices())

        # Add this amp's color index
        if self.get('interaction_id') not in [0,-1]:
            color_indices.append(self.get('color_key'))

        return color_indices

    def find_outgoing_number(self):
        """Return 0. Needed to treat HelasAmplitudes and
        HelasWavefunctions on same footing."""

        return 0

    def get_conjugate_index(self):
        """Return the index of the particle that should be conjugated."""

        if not any([(wf.get('fermionflow') < 0 or wf.is_majorana()) for wf in \
                    self.get('mothers')]):
            return ()
        
        # Pick out first sorted mothers, then fermions
        mothers, self_index = \
                      self.get('mothers').sort_by_pdg_codes(self.get('pdg_codes'))
        fermions = HelasWavefunctionList([wf for wf in mothers if wf.is_fermion()])

        # Initialize indices with indices due to Majoranas with wrong order
        indices = fermions.majorana_conjugates()

        # Check for fermions with negative fermion flow
        for i in range(0,len(fermions), 2):
            if fermions[i].get('fermionflow') < 0 or \
               fermions[i+1].get('fermionflow') < 0:
                indices.append(i/2 + 1)
                
        return tuple(sorted(indices))

    def get_vertex_leg_numbers(self, 
              veto_inter_id=base_objects.Vertex.ID_to_veto_for_multichanneling,
              max_n_loop=0):
        """Get a list of the number of legs in vertices in this diagram,
        This function is only used for establishing the multi-channeling, so that
        we exclude from it all the fake vertices and the vertices resulting from
        shrunk loops (id=-2)"""

        if max_n_loop == 0:
            max_n_loop = base_objects.Vertex.max_n_loop_for_multichanneling

        vertex_leg_numbers = [len(self.get('mothers'))] if \
                             (self['interaction_id'] not in veto_inter_id) or \
          (self['interaction_id']==-2 and len(self.get('mothers'))>max_n_loop) \
                                                                         else []
        for mother in self.get('mothers'):
            vertex_leg_numbers.extend(mother.get_vertex_leg_numbers(
                                                 veto_inter_id = veto_inter_id))

        return vertex_leg_numbers

    def get_helas_call_dict(self,index=1,OptimizedOutput=False,
                                 specifyHel=True,**opt):
        """ return a dictionary to be used for formatting
        HELAS call."""
        
        if index == 1:
            flip = 0
        else:
            flip = 1
        
        output = {}
        for i, mother in enumerate(self.get('mothers')):
            nb = mother.get('me_id') - flip 
            output[str(i)] = nb
            if mother.get('is_loop'):
                output['WF%d' % i ] = 'L(1,%d)'%nb
            else:
                if specifyHel:
                    output['WF%d' % i ] = '(1,WE(%d),H)'%nb
                else:
                    output['WF%d' % i ] = '(1,WE(%d))'%nb                    
                
        #fixed argument
        for i, coup in enumerate(self.get('coupling')):
            output['coup%d'%i] = str(coup)

        output['out'] = self.get('number') - flip
        output['propa'] = ''
        output.update(opt)
        return output



    def set_coupling_color_factor(self):
        """Check if there is a mismatch between order of fermions
        w.r.t. color"""
        mothers = self.get('mothers')

        # Sort mothers according to pdg codes if fermions with indentical
        # color but not identical pdg code. Needed for antisymmetric
        # color eps^{ijk}.
        for imo in range(len(mothers)-1):
            if mothers[imo].get('color') != 1 and \
               mothers[imo].is_fermion() and \
               mothers[imo].get('color') == mothers[imo+1].get('color') and \
               mothers[imo].get('spin') == mothers[imo+1].get('spin') and \
               mothers[imo].get('pdg_code') != mothers[imo+1].get('pdg_code'):
                mothers, my_index = \
                         mothers.sort_by_pdg_codes(self.get('pdg_codes'))
                break

        if mothers != self.get('mothers') and \
                                       not self.get('coupling').startswith('-'):
            # We have mismatch between fermion order for color and lorentz
            self.set('coupling', '-'+self.get('coupling'))

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.

    def __eq__(self, other):
        """Comparison between different amplitudes, to allow check for
        identical processes.
        """

        if not isinstance(other, HelasAmplitude):
            return False

        # Check relevant directly defined properties
        if self['lorentz'] != other['lorentz'] or \
           self['coupling'] != other['coupling'] or \
           self['number'] != other['number']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return sorted([mother['number'] for mother in self['mothers']]) == \
               sorted([mother['number'] for mother in other['mothers']])

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

#===============================================================================
# HelasAmplitudeList
#===============================================================================
class HelasAmplitudeList(base_objects.PhysicsObjectList):
    """List of HelasAmplitude objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasAmplitude for the list."""

        return isinstance(obj, HelasAmplitude)


#===============================================================================
# HelasDiagram
#===============================================================================
class HelasDiagram(base_objects.PhysicsObject):
    """HelasDiagram: list of HelasWavefunctions and a HelasAmplitude,
    plus the fermion factor associated with the corresponding diagram.
    """

    def default_setup(self):
        """Default values for all properties"""

        self['wavefunctions'] = HelasWavefunctionList()
        # In the optimized output the loop wavefunctions can be recycled as
        # well. If so, those are put in the list below, and are not mixed with
        # the tree wavefunctions above in order to keep the original structure.
        self['loop_wavefunctions'] = HelasWavefunctionList()
        # One diagram can have several amplitudes, if there are
        # different Lorentz or color structures associated with this
        # diagram
        self['amplitudes'] = HelasAmplitudeList()
        self['number'] = 0

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'wavefunctions' or name == 'loop_wavefunctions':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasWavefunctionList object" % \
                        str(value)
      
        if name == 'amplitudes':
            if not isinstance(value, HelasAmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasAmplitudeList object" % \
                        str(value)

        return True
                
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['wavefunctions', 'loop_wavefunctions', 'amplitudes']

    def calculate_orders(self):
        """Calculate the actual coupling orders of this diagram"""

        wavefunctions = HelasWavefunctionList.extract_wavefunctions(\
                                       self.get('amplitudes')[0].get('mothers'))

        coupling_orders = {}
        for wf in wavefunctions + [self.get('amplitudes')[0]]:
            if not wf.get('orders'): continue
            for order in wf.get('orders').keys():
                try:
                    coupling_orders[order] += wf.get('orders')[order]
                except Exception:
                    coupling_orders[order] = wf.get('orders')[order]

        return coupling_orders

    def get_vertex_leg_numbers(self, 
              veto_inter_id=base_objects.Vertex.ID_to_veto_for_multichanneling,
              max_n_loop=0):
        """Get a list of the number of legs in vertices in this diagram"""

        if max_n_loop == 0:
            max_n_loop = base_objects.Vertex.max_n_loop_for_multichanneling

        return self.get('amplitudes')[0].get_vertex_leg_numbers(
                             veto_inter_id=veto_inter_id, max_n_loop=max_n_loop)

    def get_regular_amplitudes(self):
        """ For regular HelasDiagrams, it is simply all amplitudes.
        It is overloaded in LoopHelasDiagram"""
        
        return self['amplitudes']

#===============================================================================
# HelasDiagramList
#===============================================================================
class HelasDiagramList(base_objects.PhysicsObjectList):
    """List of HelasDiagram objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasDiagram for the list."""

        return isinstance(obj, HelasDiagram)

#===============================================================================
# HelasMatrixElement
#===============================================================================
class HelasMatrixElement(base_objects.PhysicsObject):
    """HelasMatrixElement: list of processes with identical Helas
    calls, and the list of HelasDiagrams associated with the processes.

    If initiated with an Amplitude, HelasMatrixElement calls
    generate_helas_diagrams, which goes through the diagrams of the
    Amplitude and generates the corresponding Helas calls, taking into
    account possible fermion flow clashes due to Majorana
    particles. The optional optimization argument determines whether
    optimization is used (optimization = 1, default), for maximum
    recycling of wavefunctions, or no optimization (optimization = 0)
    when each diagram is written independently of all previous
    diagrams (this is useful for running with restricted memory,
    e.g. on a GPU). For processes with many diagrams, the total number
    or wavefunctions after optimization is ~15% of the number of
    amplitudes (diagrams).

    By default, it will also generate the color information (color
    basis and color matrix) corresponding to the Amplitude.
    """

    def default_setup(self):
        """Default values for all properties"""

        self['processes'] = base_objects.ProcessList()
        self['diagrams'] = HelasDiagramList()
        self['identical_particle_factor'] = 0
        self['color_basis'] = color_amp.ColorBasis()
        self['color_matrix'] = color_amp.ColorMatrix(color_amp.ColorBasis())
        # base_amplitude is the Amplitude to be used in color
        # generation, drawing etc. For decay chain processes, this is
        # the Amplitude which corresponds to the combined process.
        self['base_amplitude'] = None
        # has_mirror_process is True if the same process but with the
        # two incoming particles interchanged has been generated
        self['has_mirror_process'] = False

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'processes':
            if not isinstance(value, base_objects.ProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessList object" % str(value)
        if name == 'diagrams':
            if not isinstance(value, HelasDiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasDiagramList object" % str(value)
        if name == 'identical_particle_factor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid int object" % str(value)
        if name == 'color_basis':
            if not isinstance(value, color_amp.ColorBasis):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ColorBasis object" % str(value)
        if name == 'color_matrix':
            if not isinstance(value, color_amp.ColorMatrix):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ColorMatrix object" % str(value)
        if name == 'base_amplitude':
            if value != None and not \
                   isinstance(value, diagram_generation.Amplitude):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Amplitude object" % str(value)
        if name == 'has_mirror_process':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid boolean" % str(value)
        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['processes', 'identical_particle_factor',
                'diagrams', 'color_basis', 'color_matrix',
                'base_amplitude', 'has_mirror_process']

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'base_amplitude' and not self[name]:
            self['base_amplitude'] = self.get_base_amplitude()

        return super(HelasMatrixElement, self).get(name)

    # Customized constructor
    def __init__(self, amplitude=None, optimization=1,
                 decay_ids=[], gen_color=True):
        """Constructor for the HelasMatrixElement. In particular allows
        generating a HelasMatrixElement from an Amplitude, with
        automatic generation of the necessary wavefunctions
        """

        if amplitude != None:
            if isinstance(amplitude, diagram_generation.Amplitude):
                super(HelasMatrixElement, self).__init__()
                self.get('processes').append(amplitude.get('process'))
                self.set('has_mirror_process',
                         amplitude.get('has_mirror_process'))
                self.generate_helas_diagrams(amplitude, optimization, decay_ids)
                self.calculate_fermionfactors()
                self.calculate_identical_particle_factor()
                if gen_color and not self.get('color_basis'):
                    self.process_color()
            else:
                # In this case, try to use amplitude as a dictionary
                super(HelasMatrixElement, self).__init__(amplitude)
        else:
            super(HelasMatrixElement, self).__init__()

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    def __eq__(self, other):
        """Comparison between different matrix elements, to allow check for
        identical processes.
        """

        if not isinstance(other, HelasMatrixElement):
            return False

        # If no processes, this is an empty matrix element
        if not self['processes'] and not other['processes']:
            return True

        # Should only check if diagrams and process id are identical
        # Except in case of decay processes: then also initial state
        # must be the same
        if self['processes'] and not other['processes'] or \
               self['has_mirror_process'] != other['has_mirror_process'] or \
               self['processes'] and \
               self['processes'][0]['id'] != other['processes'][0]['id'] or \
               self['processes'][0]['is_decay_chain'] or \
               other['processes'][0]['is_decay_chain'] or \
               self['identical_particle_factor'] != \
                           other['identical_particle_factor'] or \
               self['diagrams'] != other['diagrams']:
            return False
        return True

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

    def process_color(self):
        """ Perform the simple color processing from a single matrix element 
        (without optimization then). This is called from the initialization
        and pulled out here in order to have the correct treatment in daughter
        classes."""
        logger.debug('Computing the color basis')
        self.get('color_basis').build(self.get('base_amplitude'))
        self.set('color_matrix',
          color_amp.ColorMatrix(self.get('color_basis')))
        
    def generate_helas_diagrams(self, amplitude, optimization=1,decay_ids=[]):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions and amplitudes. Choose between default
        optimization (= 1, maximum recycling of wavefunctions) or no
        optimization (= 0, no recycling of wavefunctions, useful for
        GPU calculations with very restricted memory).

        Note that we need special treatment for decay chains, since
        the end product then is a wavefunction, not an amplitude.
        """
        
        assert  isinstance(amplitude, diagram_generation.Amplitude), \
                    "Missing or erraneous arguments for generate_helas_diagrams"
        assert isinstance(optimization, int), \
                    "Missing or erraneous arguments for generate_helas_diagrams"
        self.optimization = optimization

        diagram_list = amplitude.get('diagrams')
        process = amplitude.get('process')

        model = process.get('model')
        if not diagram_list:
            return

        # All the previously defined wavefunctions
        wavefunctions = []
        # List of minimal information for comparison with previous
        # wavefunctions
        wf_mother_arrays = []
        # Keep track of wavefunction number
        wf_number = 0

        # Generate wavefunctions for the external particles
        external_wavefunctions = dict([(leg.get('number'),
                                        HelasWavefunction(leg, 0, model,
                                                          decay_ids)) \
                                       for leg in process.get('legs')])

        # Initially, have one wavefunction for each external leg.
        wf_number = len(process.get('legs'))

        # For initial state bosons, need to flip part-antipart
        # since all bosons should be treated as outgoing
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.is_boson() and wf.get('state') == 'initial' and \
               not wf.get('self_antipart'):
                wf.set('is_part', not wf.get('is_part'))

        # For initial state particles, need to flip PDG code (if has
        # antipart)
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.get('leg_state') == False and \
               not wf.get('self_antipart'):
                wf.flip_part_antipart()

        # Now go through the diagrams, looking for undefined wavefunctions

        helas_diagrams = HelasDiagramList()

        # Keep track of amplitude number and diagram number
        amplitude_number = 0
        diagram_number = 0

        for diagram in diagram_list:

            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()

            vertices = copy.copy(diagram.get('vertices'))

            # Single out last vertex, since this will give amplitude
            lastvx = vertices.pop()

            # Go through all vertices except the last and create
            # wavefunctions
            for vertex in vertices:

                # In case there are diagrams with multiple Lorentz/color 
                # structures, we need to keep track of the wavefunctions
                # for each such structure separately, and generate
                # one HelasDiagram for each structure.
                # We use the array number_to_wavefunctions to keep
                # track of this, with one dictionary per chain of
                # wavefunctions
                # Note that all wavefunctions relating to this diagram
                # will be written out before the first amplitude is written.
                new_number_to_wavefunctions = []
                new_color_lists = []
                for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                     color_lists):
                    legs = copy.copy(vertex.get('legs'))
                    last_leg = legs.pop()
                    # Generate list of mothers from legs
                    mothers = self.getmothers(legs, number_wf_dict,
                                              external_wavefunctions,
                                              wavefunctions,
                                              diagram_wavefunctions)
                    inter = model.get('interaction_dict')[vertex.get('id')]

                    # Now generate new wavefunction for the last leg

                    # Need one amplitude for each color structure,
                    done_color = {} # store link to color
                    for coupl_key in sorted(inter.get('couplings').keys()):
                        color = coupl_key[0]
                        if color in done_color:
                            wf = done_color[color]
                            wf.get('coupling').append(inter.get('couplings')[coupl_key])
                            wf.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                            continue
                        wf = HelasWavefunction(last_leg, vertex.get('id'), model)
                        wf.set('coupling', [inter.get('couplings')[coupl_key]])
                        if inter.get('color'):
                            wf.set('inter_color', inter.get('color')[coupl_key[0]])
                        done_color[color] = wf
                        wf.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        wf.set('color_key', color)
                        wf.set('mothers', mothers)
                        # Need to set incoming/outgoing and
                        # particle/antiparticle according to the fermion flow
                        # of mothers
                        wf.set_state_and_particle(model)
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        # Also need to keep track of the wavefunction number.
                        wf, wf_number = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wf_number)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError, error:
                            # Update wf number
                            wf_number = wf_number + 1
                            wf.set('number', wf_number)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wf_number
                                wf_number = wf_number - 1
                            except ValueError:
                                diagram_wavefunctions.append(wf)

                        new_number_wf_dict[last_leg.get('number')] = wf

                        # Store the new copy of number_wf_dict
                        new_number_to_wavefunctions.append(\
                                                        new_number_wf_dict)
                        # Add color index and store new copy of color_lists
                        new_color_list = copy.copy(color_list)
                        new_color_list.append(coupl_key[0])
                        new_color_lists.append(new_color_list)

                number_to_wavefunctions = new_number_to_wavefunctions
                color_lists = new_color_lists

            # Generate all amplitudes corresponding to the different
            # copies of this diagram
            helas_diagram = HelasDiagram()
            diagram_number = diagram_number + 1
            helas_diagram.set('number', diagram_number)
            for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                  color_lists):
                # Now generate HelasAmplitudes from the last vertex.
                if lastvx.get('id'):
                    inter = model.get_interaction(lastvx.get('id'))
                    keys = sorted(inter.get('couplings').keys())
                    pdg_codes = [p.get_pdg_code() for p in \
                                 inter.get('particles')]
                else:
                    # Special case for decay chain - amplitude is just a
                    # placeholder for replaced wavefunction
                    inter = None
                    keys = [(0, 0)]
                    pdg_codes = None

                # Find mothers for the amplitude
                legs = lastvx.get('legs')
                mothers = self.getmothers(legs, number_wf_dict,
                                          external_wavefunctions,
                                          wavefunctions,
                                          diagram_wavefunctions).\
                                             sort_by_pdg_codes(pdg_codes, 0)[0]
                # Need to check for clashing fermion flow due to
                # Majorana fermions, and modify if necessary
                wf_number = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              None,
                                              wf_number,
                                              False,
                                              number_to_wavefunctions)
                done_color = {}
                for i, coupl_key in enumerate(keys):
                    color = coupl_key[0]
                    if inter and color in done_color.keys():
                        amp = done_color[color]
                        amp.get('coupling').append(inter.get('couplings')[coupl_key])
                        amp.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                        continue
                    amp = HelasAmplitude(lastvx, model)
                    if inter:
                        amp.set('coupling', [inter.get('couplings')[coupl_key]])
                        amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        if inter.get('color'):
                            amp.set('inter_color', inter.get('color')[color])
                        amp.set('color_key', color)
                        done_color[color] = amp
                    amp.set('mothers', mothers)
                    amplitude_number = amplitude_number + 1
                    amp.set('number', amplitude_number)
                    # Add the list with color indices to the amplitude
                    new_color_list = copy.copy(color_list)
                    if inter:
                        new_color_list.append(color)
                        
                    amp.set('color_indices', new_color_list)

                    # Add amplitude to amplitdes in helas_diagram
                    helas_diagram.get('amplitudes').append(amp)

            # After generation of all wavefunctions and amplitudes,
            # first sort the wavefunctions according to number
            diagram_wavefunctions.sort(lambda wf1, wf2: \
                                       wf1.get('number') - wf2.get('number'))
            
            # Then make sure that all mothers come before daughters
            iwf = len(diagram_wavefunctions) - 1
            while iwf > 0:
                this_wf = diagram_wavefunctions[iwf]
                moved = False
                for i,wf in enumerate(diagram_wavefunctions[:iwf]):
                    if this_wf in wf.get('mothers'):
                        diagram_wavefunctions.pop(iwf)
                        diagram_wavefunctions.insert(i, this_wf)
                        this_wf.set('number', wf.get('number'))
                        for w in diagram_wavefunctions[i+1:]:
                            w.set('number',w.get('number')+1)
                        moved = True
                        break
                if not moved: iwf -= 1

            # Finally, add wavefunctions to diagram
            helas_diagram.set('wavefunctions', diagram_wavefunctions)

            if optimization:
                wavefunctions.extend(diagram_wavefunctions)
                wf_mother_arrays.extend([wf.to_array() for wf \
                                         in diagram_wavefunctions])
            else:
                wf_number = len(process.get('legs'))

            # Append this diagram in the diagram list
            helas_diagrams.append(helas_diagram)
        

        self.set('diagrams', helas_diagrams)

        # Sort all mothers according to the order wanted in Helas calls
        for wf in self.get_all_wavefunctions():
            wf.set('mothers', HelasMatrixElement.sorted_mothers(wf))

        for amp in self.get_all_amplitudes():
            amp.set('mothers', HelasMatrixElement.sorted_mothers(amp))
            amp.set('color_indices', amp.get_color_indices())

              
    def reuse_outdated_wavefunctions(self, helas_diagrams):
        """change the wavefunctions id used in the writer to minimize the 
           memory used by the wavefunctions."""
           
        if not self.optimization:
            for diag in helas_diagrams:
                for wf in diag['wavefunctions']:
                    wf.set('me_id',wf.get('number'))
            return helas_diagrams

        # First compute the first/last appearance of each wavefunctions
        # first takes the line number and return the id of the created wf
        # last_lign takes the id of the wf and return the line number
        last_lign={}
        first={}
        pos=0
        for diag in helas_diagrams:
            for wf in diag['wavefunctions']:
                pos+=1
                for wfin in wf.get('mothers'):
                    last_lign[wfin.get('number')] = pos
                    assert wfin.get('number') in first.values()
                first[pos] = wf.get('number')
            for amp in diag['amplitudes']:
                pos+=1
                for wfin in amp.get('mothers'):
                    last_lign[wfin.get('number')] = pos
        
        # last takes the line number and return the last appearing wf at
        #that particular line
        last=collections.defaultdict(list)
        for nb, pos in last_lign.items():
            last[pos].append(nb)
        tag = list(set(last.keys()+first.keys())) 
        tag.sort() #lines number where something happen (new in/out) 

        # Create the replacement id dictionary
        outdated = [] # wf id which ar not use any more at this stage
        replace = {}  # replacement directory
        max_wf = 0
        for nb in tag:
            if outdated and nb in first:
                replace[first[nb]] = outdated.pop()
            elif nb in first:
                assert first[nb] not in replace, '%s already assigned' % first[nb]
                max_wf += 1
                replace[first[nb]] = max_wf
            if nb in last:
                for value in last[nb]:
                    outdated.append(replace[value])

                   
        #replace the id
        for diag in helas_diagrams:
            for wf in diag['wavefunctions']:
                wf.set('me_id', replace[wf.get('number')])
        
        return helas_diagrams

    def restore_original_wavefunctions(self):
        """This restore the original memory print and revert
           change the wavefunctions id used in the writer to minimize the 
           memory used by the wavefunctions."""
        
        helas_diagrams = self.get('diagrams')
        
        for diag in helas_diagrams:
            for wf in diag['wavefunctions']:
                wf.set('me_id',wf.get('number'))
        
        return helas_diagrams


    def insert_decay_chains(self, decay_dict):
        """Iteratively insert decay chains decays into this matrix
        element.        
        * decay_dict: a dictionary from external leg number
          to decay matrix element.
        """

        # First need to reset all legs_with_decays
        for proc in self.get('processes'):
            proc.set('legs_with_decays', base_objects.LegList())

        # We need to keep track of how the
        # wavefunction numbers change
        replace_dict = {}
        for number in decay_dict.keys():
            # Find all wavefunctions corresponding to this external
            # leg number
            replace_dict[number] = [wf for wf in \
                          filter(lambda wf: not wf.get('mothers') and \
                                 wf.get('number_external') == number,
                                 self.get_all_wavefunctions())]

        # Keep track of wavefunction and amplitude numbers, to ensure
        # unique numbers for all new wfs and amps during manipulations
        numbers = [self.get_all_wavefunctions()[-1].get('number'),
                   self.get_all_amplitudes()[-1].get('number')]

        # Check if there are any Majorana particles present in any diagrams
        got_majoranas = False
        for wf in self.get_all_wavefunctions() + \
            sum([d.get_all_wavefunctions() for d in \
                 decay_dict.values()], []):
            if wf.get('fermionflow') < 0 or \
                   wf.get('self_antipart') and wf.is_fermion():
                got_majoranas = True

        # Now insert decays for all legs that have decays
        for number in decay_dict.keys():

                self.insert_decay(replace_dict[number],
                                  decay_dict[number],
                                  numbers,
                                  got_majoranas)

        # Remove all diagrams that surpass overall coupling orders
        overall_orders = self.get('processes')[0].get('overall_orders')
        if overall_orders:
            ndiag = len(self.get('diagrams'))
            idiag = 0
            while self.get('diagrams')[idiag:]:
                diagram = self.get('diagrams')[idiag]
                orders = diagram.calculate_orders()
                remove_diagram = False
                for order in orders.keys():
                    try:
                        if orders[order] > \
                               overall_orders[order]:
                            remove_diagram = True
                    except KeyError:
                        pass
                if remove_diagram:
                    self.get('diagrams').pop(idiag)
                else:
                    idiag += 1

            if len(self.get('diagrams')) < ndiag:
                # We have removed some diagrams - need to go through
                # diagrams, renumber them and set new wavefunctions
                wf_numbers = []
                ndiagrams = 0
                for diagram in self.get('diagrams'):
                    ndiagrams += 1
                    diagram.set('number', ndiagrams)
                    # Extract all wavefunctions contributing to this amplitude
                    diagram_wfs = HelasWavefunctionList()
                    for amplitude in diagram.get('amplitudes'):
                        wavefunctions = \
                          sorted(HelasWavefunctionList.\
                               extract_wavefunctions(amplitude.get('mothers')),
                                 lambda wf1, wf2: wf1.get('number') - \
                                                  wf2.get('number'))
                        for wf in wavefunctions:
                            # Check if wavefunction already used, otherwise add
                            if wf.get('number') not in wf_numbers and \
                                   wf not in diagram_wfs:
                                diagram_wfs.append(wf)
                                wf_numbers.append(wf.get('number'))
                    diagram.set('wavefunctions', diagram_wfs)

        # Final cleaning out duplicate wavefunctions - needed only if
        # we have multiple fermion flows, i.e., either multiple replaced
        # wavefunctions or  majorana fermions and multiple diagrams
        flows = reduce(lambda i1, i2: i1 * i2,
                       [len(replace_dict[i]) for i in decay_dict.keys()], 1)
        diagrams = reduce(lambda i1, i2: i1 * i2,
                               [len(decay_dict[i].get('diagrams')) for i in \
                                decay_dict.keys()], 1)

        if flows > 1 or (diagrams > 1 and got_majoranas):

            # Clean out any doublet wavefunctions

            earlier_wfs = []

            earlier_wf_arrays = []

            mothers = self.get_all_wavefunctions() + self.get_all_amplitudes()
            mother_arrays = [w['mothers'].to_array() \
                             for w in mothers]

            for diagram in self.get('diagrams'):

                if diagram.get('number') > 1:
                    earlier_wfs.extend(self.get('diagrams')[\
                        diagram.get('number') - 2].get('wavefunctions'))

                i = 0
                diag_wfs = diagram.get('wavefunctions')


                # Remove wavefunctions and replace mothers
                while diag_wfs[i:]:
                    try:
                        new_wf = earlier_wfs[\
                            earlier_wfs.index(diag_wfs[i])]
                        wf = diag_wfs.pop(i)

                        self.update_later_mothers(wf, new_wf, mothers,
                                                  mother_arrays)
                    except ValueError:
                        i = i + 1

        # When we are done with all decays, set wavefunction and
        # amplitude numbers
        for i, wf in enumerate(self.get_all_wavefunctions()):
            wf.set('number', i + 1)
        for i, amp in enumerate(self.get_all_amplitudes()):
            amp.set('number', i + 1)
            # Update fermion factors for all amplitudes
            amp.calculate_fermionfactor()
            # Update color indices
            amp.set('color_indices', amp.get_color_indices())

        # Calculate identical particle factors for
        # this matrix element
        self.identical_decay_chain_factor(decay_dict.values())
        

    def insert_decay(self, old_wfs, decay, numbers, got_majoranas):
        """Insert a decay chain matrix element into the matrix element.
        * old_wfs: the wavefunctions to be replaced.
          They all correspond to the same external particle, but might
          have different fermion flow directions
        * decay: the matrix element for the decay chain
        * numbers: the present wavefunction and amplitude number,
          to allow for unique numbering
          
        Note that:
        1) All amplitudes and all wavefunctions using the decaying wf
           must be copied as many times as there are amplitudes in the
           decay matrix element
        2) In the presence of Majorana particles, we must make sure
           to flip fermion flow for the decay process if needed.

        The algorithm is the following:
        1) Multiply the diagrams with the number of diagrams Ndiag in
           the decay element
        2) For each diagram in the decay element, work on the diagrams
           which corresponds to it
        3) Flip fermion flow for the decay wavefunctions if needed
        4) Insert all auxiliary wavefunctions into the diagram (i.e., all 
           except the final wavefunctions, which directly replace the
           original final state wavefunctions)
        4) Replace the wavefunctions recursively, so that we always replace
           each old wavefunctions with Namp new ones, where Namp is
           the number of amplitudes in this decay element
           diagram. Do recursion for wavefunctions which have this
           wavefunction as mother. Simultaneously replace any
           amplitudes which have this wavefunction as mother.
        """

        len_decay = len(decay.get('diagrams'))

        number_external = old_wfs[0].get('number_external')

        # Insert the decay process in the process
        for process in self.get('processes'):
            process.get('decay_chains').append(\
                   decay.get('processes')[0])

        # We need one copy of the decay element diagrams for each
        # old_wf to be replaced, since we need different wavefunction
        # numbers for them
        decay_elements = [copy.deepcopy(d) for d in \
                          [ decay.get('diagrams') ] * len(old_wfs)]

        # Need to replace Particle in all wavefunctions to avoid
        # deepcopy
        for decay_element in decay_elements:
            for idiag, diagram in enumerate(decay.get('diagrams')):
                wfs = diagram.get('wavefunctions')
                decay_diag = decay_element[idiag]
                for i, wf in enumerate(decay_diag.get('wavefunctions')):
                    wf.set('particle', wfs[i].get('particle'))
                    wf.set('antiparticle', wfs[i].get('antiparticle'))

        for decay_element in decay_elements:

            # Remove the unwanted initial state wavefunctions from decay
            for decay_diag in decay_element:
                for wf in filter(lambda wf: wf.get('number_external') == 1,
                                 decay_diag.get('wavefunctions')):
                    decay_diag.get('wavefunctions').remove(wf)

            decay_wfs = sum([d.get('wavefunctions') for d in decay_element], [])

            # External wavefunction offset for new wfs
            incr_new = number_external - \
                       decay_wfs[0].get('number_external')

            for wf in decay_wfs:
                # Set number_external for new wavefunctions
                wf.set('number_external', wf.get('number_external') + incr_new)
                # Set unique number for new wavefunctions
                numbers[0] = numbers[0] + 1
                wf.set('number', numbers[0])

        # External wavefunction offset for old wfs, only the first
        # time this external wavefunction is replaced
        (nex, nin) = decay.get_nexternal_ninitial()
        incr_old = nex - 2 # due to the incoming particle
        wavefunctions = self.get_all_wavefunctions()
        for wf in wavefunctions:
            # Only modify number_external for wavefunctions above old_wf
            if wf.get('number_external') > number_external:
                wf.set('number_external',
                       wf.get('number_external') + incr_old)

        # Multiply the diagrams by Ndiag

        diagrams = HelasDiagramList()
        for diagram in self.get('diagrams'):
            new_diagrams = [copy.copy(diag) for diag in \
                            [ diagram ] * (len_decay - 1)]
            # Update diagram number
            diagram.set('number', (diagram.get('number') - 1) * \
                        len_decay + 1)

            for i, diag in enumerate(new_diagrams):
                # Set diagram number
                diag.set('number', diagram.get('number') + i + 1)
                # Copy over all wavefunctions
                diag.set('wavefunctions',
                         copy.copy(diagram.get('wavefunctions')))
                # Copy over the amplitudes
                amplitudes = HelasAmplitudeList(\
                                [copy.copy(amp) for amp in \
                                 diag.get('amplitudes')])
                # Renumber amplitudes
                for amp in amplitudes:
                    numbers[1] = numbers[1] + 1
                    amp.set('number', numbers[1])
                diag.set('amplitudes', amplitudes)
            # Add old and new diagram to diagrams
            diagrams.append(diagram)
            diagrams.extend(new_diagrams)

        self.set('diagrams', diagrams)

        # Now we work by decay process diagram, parameterized by numdecay
        for numdecay in range(len_decay):

            # Pick out the diagrams which correspond to this decay diagram
            diagrams = [self.get('diagrams')[i] for i in \
                        range(numdecay, len(self.get('diagrams')), len_decay)]

            # Perform replacement for each of the wavefunctions in old_wfs
            for decay_element, old_wf in zip(decay_elements, old_wfs):

                decay_diag = decay_element[numdecay]

                # Find the diagrams which have old_wf
                my_diagrams = filter(lambda diag: (old_wf.get('number') in \
                                            [wf.get('number') for wf in \
                                            diag.get('wavefunctions')]),
                                     diagrams)

                # Ignore possibility for unoptimizated generation for now
                if len(my_diagrams) > 1:
                    raise self.PhysicsObjectError, \
                          "Decay chains not yet prepared for GPU"

                for diagram in my_diagrams:

                    if got_majoranas:
                        # If there are Majorana particles in any of
                        # the matrix elements, we need to check for
                        # fermion flow

                        # Earlier wavefunctions, will be used for fermion flow
                        index = [d.get('number') for d in diagrams].\
                                index(diagram.get('number'))
                        earlier_wavefunctions = \
                                      sum([d.get('wavefunctions') for d in \
                                           diagrams[:index]], [])

                        # Don't want to affect original decay
                        # wavefunctions, so need to deepcopy
                        decay_diag_wfs = copy.deepcopy(\
                                                decay_diag.get('wavefunctions'))
                        # Need to replace Particle in all
                        # wavefunctions to avoid deepcopy
                        for i, wf in enumerate(decay_diag.get('wavefunctions')):
                            decay_diag_wfs[i].set('particle', \
                                                  wf.get('particle'))
                            decay_diag_wfs[i].set('antiparticle', \
                                                  wf.get('antiparticle'))

                        # Complete decay_diag_wfs with the mother wavefunctions
                        # to allow for independent fermion flow flips
                        decay_diag_wfs = decay_diag_wfs.insert_own_mothers()

                        # These are the wavefunctions which directly replace old_wf
                        final_decay_wfs = [amp.get('mothers')[1] for amp in \
                                              decay_diag.get('amplitudes')]

                        # Since we made deepcopy, need to syncronize
                        for i, wf in enumerate(final_decay_wfs):
                            final_decay_wfs[i] = \
                                               decay_diag_wfs[decay_diag_wfs.index(wf)]
                            
                        # Remove final wavefunctions from decay_diag_wfs,
                        # since these will be replaced separately by
                        # replace_wavefunctions
                        for wf in final_decay_wfs:
                            decay_diag_wfs.remove(wf)

                        # Check fermion flow direction
                        if old_wf.is_fermion() and \
                               old_wf.get_with_flow('state') != \
                                     final_decay_wfs[0].get_with_flow('state'):

                            # Not same flow state - need to flip flow of wf

                            for i, wf in enumerate(final_decay_wfs):

                                # We use the function
                                # check_majorana_and_flip_flow, as in the
                                # helas diagram generation.  Since we have
                                # different flow, there is already a Majorana
                                # particle along the fermion line.

                                final_decay_wfs[i], numbers[0] = \
                                                wf.check_majorana_and_flip_flow(\
                                                         True,
                                                         earlier_wavefunctions,
                                                         decay_diag_wfs,
                                                         {},
                                                         numbers[0])

                        # Remove wavefunctions which are already present in
                        # earlier_wavefunctions
                        i = 0
                        earlier_wavefunctions = \
                            sum([d.get('wavefunctions') for d in \
                                 self.get('diagrams')[:diagram.get('number') - 1]], \
                                [])
                        earlier_wf_numbers = [wf.get('number') for wf in \
                                              earlier_wavefunctions]

                        i = 0
                        mother_arrays = [w.get('mothers').to_array() for \
                                         w in final_decay_wfs]
                        while decay_diag_wfs[i:]:
                            wf = decay_diag_wfs[i]
                            try:
                                new_wf = earlier_wavefunctions[\
                                    earlier_wf_numbers.index(wf.get('number'))]
                                # If the wavefunctions are not identical,
                                # then we should keep this wavefunction,
                                # and update its number so it is unique
                                if wf != new_wf:
                                    numbers[0] = numbers[0] + 1
                                    wf.set('number', numbers[0])
                                    continue
                                decay_diag_wfs.pop(i)
                                pres_mother_arrays = [w.get('mothers').to_array() for \
                                                      w in decay_diag_wfs[i:]] + \
                                                      mother_arrays
                                self.update_later_mothers(wf, new_wf,
                                                          decay_diag_wfs[i:] + \
                                                          final_decay_wfs,
                                                          pres_mother_arrays)
                            except ValueError:
                                i = i + 1

                        # Since we made deepcopy, go through mothers and make
                        # sure we are using the ones in earlier_wavefunctions
                        for decay_wf in decay_diag_wfs + final_decay_wfs:
                            mothers = decay_wf.get('mothers')
                            for i, wf in enumerate(mothers):
                                try:
                                    mothers[i] = earlier_wavefunctions[\
                                        earlier_wf_numbers.index(wf.get('number'))]
                                except ValueError:
                                    pass
                    else:
                        # If there are no Majorana particles, the
                        # treatment is much simpler
                        decay_diag_wfs = \
                                       copy.copy(decay_diag.get('wavefunctions'))

                        # These are the wavefunctions which directly
                        # replace old_wf
                        final_decay_wfs = [amp.get('mothers')[1] for amp in \
                                              decay_diag.get('amplitudes')]

                        # Remove final wavefunctions from decay_diag_wfs,
                        # since these will be replaced separately by
                        # replace_wavefunctions
                        for wf in final_decay_wfs:
                            decay_diag_wfs.remove(wf)


                    diagram_wfs = diagram.get('wavefunctions')

                    old_wf_index = [wf.get('number') for wf in \
                                    diagram_wfs].index(old_wf.get('number'))

                    diagram_wfs = diagram_wfs[0:old_wf_index] + \
                                  decay_diag_wfs + diagram_wfs[old_wf_index:]

                    diagram.set('wavefunctions', HelasWavefunctionList(diagram_wfs))

                    # Set the decay flag for final_decay_wfs, to
                    # indicate that these correspond to decayed
                    # particles
                    for wf in final_decay_wfs:
                        wf.set('onshell', True)

                    if len_decay == 1 and len(final_decay_wfs) == 1:
                        # Can use simplified treatment, by just modifying old_wf
                        self.replace_single_wavefunction(old_wf,
                                                         final_decay_wfs[0])
                    else:
                        # Multiply wavefunctions and amplitudes and
                        # insert mothers using a recursive function
                        self.replace_wavefunctions(old_wf,
                                                   final_decay_wfs,
                                                   diagrams,
                                                   numbers)
            # Now that we are done with this set of diagrams, we need
            # to clean out duplicate wavefunctions (i.e., remove
            # identical wavefunctions which are already present in
            # earlier diagrams)
            for diagram in diagrams:

                # We can have duplicate wfs only from previous copies of
                # this diagram
                earlier_wfs = sum([d.get('wavefunctions') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number') - numdecay - 1:\
                                     diagram.get('number') - 1]], [])

                later_wfs = sum([d.get('wavefunctions') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number'):]], [])

                later_amps = sum([d.get('amplitudes') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number') - 1:]], [])

                i = 0
                diag_wfs = diagram.get('wavefunctions')

                # Remove wavefunctions and replace mothers, to make
                # sure we only have one copy of each wavefunction
                # number

                mother_arrays = [w.get('mothers').to_array() for \
                                 w in later_wfs + later_amps]

                while diag_wfs[i:]:
                    try:
                        index = [w.get('number') for w in earlier_wfs].\
                                index(diag_wfs[i].get('number'))
                        wf = diag_wfs.pop(i)
                        pres_mother_arrays = [w.get('mothers').to_array() for \
                                              w in diag_wfs[i:]] + \
                                              mother_arrays
                        self.update_later_mothers(wf, earlier_wfs[index],
                                              diag_wfs[i:] + later_wfs + later_amps,
                                              pres_mother_arrays)
                    except ValueError:
                        i = i + 1

    def update_later_mothers(self, wf, new_wf, later_wfs, later_wf_arrays):
        """Update mothers for all later wavefunctions"""

        daughters = filter(lambda tup: wf.get('number') in tup[1],
                              enumerate(later_wf_arrays))

        for (index, mothers) in daughters:
            try:
                # Replace mother
                later_wfs[index].get('mothers')[\
                    mothers.index(wf.get('number'))] = new_wf
            except ValueError:
                pass

    def replace_wavefunctions(self, old_wf, new_wfs,
                              diagrams, numbers):
        """Recursive function to replace old_wf with new_wfs, and
        multiply all wavefunctions or amplitudes that use old_wf

        * old_wf: The wavefunction to be replaced
        * new_wfs: The replacing wavefunction
        * diagrams - the diagrams that are relevant for these new
          wavefunctions.
        * numbers: the present wavefunction and amplitude number,
          to allow for unique numbering
        """

        # Pick out the diagrams which have the old_wf
        my_diagrams = filter(lambda diag: old_wf.get('number') in \
                         [wf.get('number') for wf in diag.get('wavefunctions')],
                         diagrams)

        # Replace old_wf with new_wfs in the diagrams
        for diagram in my_diagrams:

            diagram_wfs = diagram.get('wavefunctions')

            old_wf_index = [wf.get('number') for wf in \
                            diagram.get('wavefunctions')].index(old_wf.get('number'))
            diagram_wfs = diagram_wfs[:old_wf_index] + \
                          new_wfs + diagram_wfs[old_wf_index + 1:]

            diagram.set('wavefunctions', HelasWavefunctionList(diagram_wfs))

        # Now need to take care of amplitudes and wavefunctions which
        # are daughters of old_wf (only among the relevant diagrams)

        # Pick out diagrams with amplitudes which are daughters of old_wf
        amp_diagrams = filter(lambda diag: old_wf.get('number') in \
                          sum([[wf.get('number') for wf in amp.get('mothers')] \
                               for amp in diag.get('amplitudes')], []),
                              diagrams)

        for diagram in amp_diagrams:

            # Amplitudes in this diagram that are daughters of old_wf
            daughter_amps = filter(lambda amp: old_wf.get('number') in \
                                [wf.get('number') for wf in amp.get('mothers')],
                                diagram.get('amplitudes'))

            new_amplitudes = copy.copy(diagram.get('amplitudes'))

            # Loop over daughter_amps, to multiply each amp by the
            # number of replacement wavefunctions and substitute mothers
            for old_amp in daughter_amps:
                # Create copies of this amp
                new_amps = [copy.copy(amp) for amp in \
                            [ old_amp ] * len(new_wfs)]
                # Replace the old mother with the new ones
                for i, (new_amp, new_wf) in enumerate(zip(new_amps, new_wfs)):
                    mothers = copy.copy(new_amp.get('mothers'))
                    old_wf_index = [wf.get('number') for wf in mothers].index(\
                         old_wf.get('number'))
                    # Update mother
                    mothers[old_wf_index] = new_wf
                    new_amp.set('mothers', mothers)
                    # Update amp numbers for replaced amp
                    numbers[1] = numbers[1] + 1
                    new_amp.set('number', numbers[1])

                # Insert the new amplitudes in diagram amplitudes
                index = [a.get('number') for a in new_amplitudes].\
                                   index(old_amp.get('number'))
                new_amplitudes = new_amplitudes[:index] + \
                                 new_amps + new_amplitudes[index + 1:]

            # Replace diagram amplitudes with the new ones
            diagram.set('amplitudes', HelasAmplitudeList(new_amplitudes))

        # Find wavefunctions that are daughters of old_wf
        daughter_wfs = filter(lambda wf: old_wf.get('number') in \
                              [wf1.get('number') for wf1 in wf.get('mothers')],
                              sum([diag.get('wavefunctions') for diag in \
                                   diagrams], []))

        # Loop over daughter_wfs, multiply them and replace mothers
        for daughter_wf in daughter_wfs:

            # Pick out the diagrams where daughter_wf occurs
            wf_diagrams = filter(lambda diag: daughter_wf.get('number') in \
                                 [wf.get('number') for wf in \
                                  diag.get('wavefunctions')],
                                 diagrams)

            if len(wf_diagrams) > 1:
                raise self.PhysicsObjectError, \
                      "Decay chains not yet prepared for GPU"

            for diagram in wf_diagrams:

                # Now create new wfs with updated mothers
                replace_daughters = [ copy.copy(wf) for wf in \
                                      [daughter_wf] * len(new_wfs) ]

                index = [wf.get('number') for wf in \
                         daughter_wf.get('mothers')].index(old_wf.get('number'))

                # Replace the old mother with the new ones, update wf numbers
                for i, (new_daughter, new_wf) in \
                        enumerate(zip(replace_daughters, new_wfs)):
                    mothers = copy.copy(new_daughter.get('mothers'))
                    mothers[index] = new_wf
                    new_daughter.set('mothers', mothers)
                    numbers[0] = numbers[0] + 1
                    new_daughter.set('number', numbers[0])

                # This is where recursion happens.  We need to replace
                # the daughter wavefunction, and fix amplitudes and
                # wavefunctions which have it as mothers.

                self.replace_wavefunctions(daughter_wf,
                                           replace_daughters,
                                           diagrams,
                                           numbers)

    def replace_single_wavefunction(self, old_wf, new_wf):
        """Insert decay chain by simply modifying wavefunction. This
        is possible only if there is only one diagram in the decay."""

        for key in old_wf.keys():
            old_wf.set(key, new_wf[key])

    def identical_decay_chain_factor(self, decay_chains):
        """Calculate the denominator factor from identical decay chains"""

        final_legs = [leg.get('id') for leg in \
                      filter(lambda leg: leg.get('state') == True, \
                              self.get('processes')[0].get('legs'))]

        # Leg ids for legs being replaced by decay chains
        decay_ids = [decay.get('legs')[0].get('id') for decay in \
                     self.get('processes')[0].get('decay_chains')]

        # Find all leg ids which are not being replaced by decay chains
        non_decay_legs = filter(lambda id: id not in decay_ids,
                                final_legs)

        # Identical particle factor for legs not being decayed
        identical_indices = {}
        for id in non_decay_legs:
            if id in identical_indices:
                identical_indices[id] = \
                                    identical_indices[id] + 1
            else:
                identical_indices[id] = 1
        non_chain_factor = reduce(lambda x, y: x * y,
                                  [ math.factorial(val) for val in \
                                    identical_indices.values() ], 1)

        # Identical particle factor for decay chains
        # Go through chains to find identical ones
        chains = copy.copy(decay_chains)
        iden_chains_factor = 1
        while chains:
            ident_copies = 1
            first_chain = chains.pop(0)
            i = 0
            while i < len(chains):
                chain = chains[i]
                if HelasMatrixElement.check_equal_decay_processes(\
                                                 first_chain, chain):
                    ident_copies = ident_copies + 1
                    chains.pop(i)
                else:
                    i = i + 1
            iden_chains_factor = iden_chains_factor * \
                                 math.factorial(ident_copies)

        self['identical_particle_factor'] = non_chain_factor * \
                                    iden_chains_factor * \
                                    reduce(lambda x1, x2: x1 * x2,
                                    [me.get('identical_particle_factor') \
                                     for me in decay_chains], 1)

    def calculate_fermionfactors(self):
        """Generate the fermion factors for all diagrams in the matrix element
        """
        for diagram in self.get('diagrams'):
            for amplitude in diagram.get('amplitudes'):
                amplitude.get('fermionfactor')

    def calculate_identical_particle_factor(self):
        """Calculate the denominator factor for identical final state particles
        """

        self["identical_particle_factor"] = self.get('processes')[0].\
                                            identical_particle_factor()

    def get_base_amplitude(self):
        """Generate a diagram_generation.Amplitude from a
        HelasMatrixElement. This is used to generate both color
        amplitudes and diagram drawing."""

        # Need to take care of diagram numbering for decay chains
        # before this can be used for those!

        optimization = 1
        if len(filter(lambda wf: wf.get('number') == 1,
                      self.get_all_wavefunctions())) > 1:
            optimization = 0

        model = self.get('processes')[0].get('model')

        wf_dict = {}
        vx_list = []
        diagrams = base_objects.DiagramList()
        for diag in self.get('diagrams'):
            diagrams.append(diag.get('amplitudes')[0].get_base_diagram(\
                wf_dict, vx_list, optimization))

        for diag in diagrams:
            diag.calculate_orders(self.get('processes')[0].get('model'))
            
        return diagram_generation.Amplitude({\
            'process': self.get('processes')[0],
            'diagrams': diagrams})

    # Helper methods

    def getmothers(self, legs, number_to_wavefunctions,
                   external_wavefunctions, wavefunctions,
                   diagram_wavefunctions):
        """Generate list of mothers from number_to_wavefunctions and
        external_wavefunctions"""

        mothers = HelasWavefunctionList()

        for leg in legs:
            try:
                # The mother is an existing wavefunction
                wf = number_to_wavefunctions[leg.get('number')]
            except KeyError:
                # This is an external leg, pick from external_wavefunctions
                wf = external_wavefunctions[leg.get('number')]
                number_to_wavefunctions[leg.get('number')] = wf
                if not wf in wavefunctions and not wf in diagram_wavefunctions:
                    diagram_wavefunctions.append(wf)
            mothers.append(wf)

        return mothers

    def get_num_configs(self):
        """Get number of diagrams, which is always more than number of
        configs"""

        model = self.get('processes')[0].\
                get('model')
        
        next, nini = self.get_nexternal_ninitial()
        return sum([d.get_num_configs(model, nini) for d in \
                    self.get('base_amplitude').get('diagrams')])

    def get_number_of_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME"""

        out =  max([wf.get('me_id') for wfs in self.get('diagrams') 
                                    for wf in wfs.get('wavefunctions')])
        if out: 
            return out
        return sum([ len(d.get('wavefunctions')) for d in \
                       self.get('diagrams')])
        
    def get_all_wavefunctions(self):
        """Gives a list of all wavefunctions for this ME"""

        return sum([d.get('wavefunctions') for d in \
                       self.get('diagrams')], [])

    def get_all_amplitudes(self):
        """Gives a list of all amplitudes for this ME"""

        return sum([d.get('amplitudes') for d in \
                       self.get('diagrams')], [])

    def get_external_wavefunctions(self):
        """Gives the external wavefunctions for this ME"""

        external_wfs = filter(lambda wf: not wf.get('mothers'),
                              self.get('diagrams')[0].get('wavefunctions'))

        external_wfs.sort(lambda w1, w2: w1.get('number_external') - \
             w2.get('number_external'))

        i = 1
        while i < len(external_wfs):
            if external_wfs[i].get('number_external') == \
               external_wfs[i - 1].get('number_external'):
                external_wfs.pop(i)
            else:
                i = i + 1
        return external_wfs

    def get_number_of_amplitudes(self):
        """Gives the total number of amplitudes for this ME"""

        return sum([ len(d.get('amplitudes')) for d in \
                       self.get('diagrams')])

    def get_nexternal_ninitial(self):
        """Gives (number or external particles, number of
        incoming particles)"""

        external_wfs = filter(lambda wf: not wf.get('mothers'),
                                                   self.get_all_wavefunctions())

        return (len(set([wf.get('number_external') for wf in \
                         external_wfs])),
                len(set([wf.get('number_external') for wf in \
                         filter(lambda wf: wf.get('leg_state') == False,
                                external_wfs)])))

    def get_external_masses(self):
        """Gives the list of the strings corresponding to the masses of the
        external particles."""

        mass_list=[]
        external_wfs = sorted(filter(lambda wf: wf.get('leg_state') != \
                              'intermediate', self.get_all_wavefunctions()),\
                              key=lambda w: w['number_external'])
        external_number=1
        for wf in external_wfs:
            if wf.get('number_external')==external_number:
                external_number=external_number+1
                mass_list.append(wf.get('particle').get('mass'))
        
        return mass_list
        
    def get_helicity_combinations(self):
        """Gives the number of helicity combinations for external
        wavefunctions"""

        if not self.get('processes'):
            return None

        model = self.get('processes')[0].get('model')

        return reduce(lambda x, y: x * y,
                      [ len(model.get('particle_dict')[wf.get('pdg_code')].\
                            get_helicity_states())\
                        for wf in self.get_external_wavefunctions() ], 1)

    def get_helicity_matrix(self, allow_reverse=True):
        """Gives the helicity matrix for external wavefunctions"""

        if not self.get('processes'):
            return None

        process = self.get('processes')[0]
        model = process.get('model')

        return apply(itertools.product, [ model.get('particle_dict')[\
                                  wf.get('pdg_code')].get_helicity_states(allow_reverse)\
                                  for wf in self.get_external_wavefunctions()])

    def get_hel_avg_factor(self):
        """ Calculate the denominator factor due to the average over initial
        state spin only """
        
        model = self.get('processes')[0].get('model')
        initial_legs = filter(lambda leg: leg.get('state') == False, \
                              self.get('processes')[0].get('legs'))
        
        return reduce(lambda x, y: x * y,
                      [ len(model.get('particle_dict')[leg.get('id')].\
                                   get_helicity_states())\
                        for leg in initial_legs ])
        
    def get_denominator_factor(self):
        """Calculate the denominator factor due to:
        Averaging initial state color and spin, and
        identical final state particles"""

        model = self.get('processes')[0].get('model')

        initial_legs = filter(lambda leg: leg.get('state') == False, \
                              self.get('processes')[0].get('legs'))

        color_factor = reduce(lambda x, y: x * y,
                              [ model.get('particle_dict')[leg.get('id')].\
                                    get('color')\
                                for leg in initial_legs ])
            
        spin_factor = reduce(lambda x, y: x * y,
                             [ len(model.get('particle_dict')[leg.get('id')].\
                                   get_helicity_states())\
                               for leg in initial_legs ])

        return spin_factor * color_factor * self['identical_particle_factor']

    def generate_color_amplitudes(self, color_basis, diagrams):
        """ Return a list of (coefficient, amplitude number) lists,
        corresponding to the JAMPs for the HelasDiagrams and color basis passed
        in argument. The coefficients are given in the format (fermion factor, 
        colorcoeff (frac), imaginary, Nc power). """

        if not color_basis:
            # No color, simply add all amplitudes with correct factor
            # for first color amplitude
            col_amp = []
            for diagram in diagrams:
                for amplitude in diagram.get('amplitudes'):
                    col_amp.append(((amplitude.get('fermionfactor'),
                                    1, False, 0),
                                    amplitude.get('number')))
            return [col_amp]

        # There is a color basis - create a list of coefficients and
        # amplitude numbers

        col_amp_list = []
        for i, col_basis_elem in \
                enumerate(sorted(color_basis.keys())):

            col_amp = []
            for diag_tuple in color_basis[col_basis_elem]:
                res_amps = filter(lambda amp: \
                          tuple(amp.get('color_indices')) == diag_tuple[1],
                          diagrams[diag_tuple[0]].get('amplitudes'))
                if not res_amps:
                    raise self.PhysicsObjectError, \
                          """No amplitude found for color structure
                            %s and color index chain (%s) (diagram %i)""" % \
                            (col_basis_elem,
                             str(diag_tuple[1]),
                             diag_tuple[0])

                for res_amp in res_amps:
                    col_amp.append(((res_amp.get('fermionfactor'),
                                     diag_tuple[2],
                                     diag_tuple[3],
                                     diag_tuple[4]),
                                    res_amp.get('number')))

            col_amp_list.append(col_amp)

        return col_amp_list

    def get_color_amplitudes(self):
        """Return a list of (coefficient, amplitude number) lists,
        corresponding to the JAMPs for this matrix element. The
        coefficients are given in the format (fermion factor, color
        coeff (frac), imaginary, Nc power)."""
        
        return self.generate_color_amplitudes(self['color_basis'],self['diagrams'])

    def sort_split_orders(self, split_orders):
        """ Sort the 'split_orders' list given in argument so that the orders of
        smaller weights appear first. Do nothing if not all split orders have
        a hierarchy defined."""
        order_hierarchy=\
                    self.get('processes')[0].get('model').get('order_hierarchy')
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            split_orders.sort(key=lambda order: order_hierarchy[order])

    def get_split_orders_mapping_for_diagram_list(self, diag_list, split_orders,
            get_amp_number_function = lambda amp: amp.get('number'),
            get_amplitudes_function = lambda diag: diag.get('amplitudes')):
        """ This a helper function for get_split_orders_mapping to return, for 
        the HelasDiagram list given in argument, the list amp_orders detailed in
        the description of the 'get_split_orders_mapping' function.
        """

        order_hierarchy=\
                    self.get('processes')[0].get('model').get('order_hierarchy')
        # Find the list of amplitude numbers and what amplitude order they
        # correspond to. For its construction, amp_orders is a dictionary with
        # is then translated into an ordered list of tuples before being returned.
        amp_orders={}
        for diag in diag_list:
            diag_orders=diag.calculate_orders()
            # Add the WEIGHTED order information
            diag_orders['WEIGHTED']=sum(order_hierarchy[order]*value for \
                                           order, value in  diag_orders.items())
            # Complement the missing split_orders with 0
            for order in split_orders:
                if not order in diag_orders.keys():
                    diag_orders[order]=0
            key = tuple([diag_orders[order] for order in split_orders])
            try:
                amp_orders[key].extend([get_amp_number_function(amp) for amp in \
                                                 get_amplitudes_function(diag)])
            except KeyError:
                amp_orders[key] = [get_amp_number_function(amp) for amp in \
                                                  get_amplitudes_function(diag)]
        amp_orders=[(order[0],tuple(order[1])) for order in amp_orders.items()]
        # Again make sure a proper hierarchy is defined and order the list
        # according to it if possible
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            # Order the contribution starting from the minimum WEIGHTED one
            amp_orders.sort(key= lambda elem: 
                         sum([order_hierarchy[split_orders[i]]*order_power for \
                                         i, order_power in enumerate(elem[0])]))
        
        return amp_orders

    def get_split_orders_mapping(self):
        """This function returns two lists, squared_orders, amp_orders.
        If process['split_orders'] is empty, the function returns two empty lists.
        
        squared_orders : All possible contributing squared orders among those
            specified in the process['split_orders'] argument. The elements of
            the list are tuples of the format format (OrderValue1,OrderValue2,...) 
            with OrderValue<i> correspond to the value of the <i>th order in
            process['split_orders'] (the others are summed over and therefore 
            left unspecified).
            Ex for dijet with process['split_orders']=['QCD','QED']: 
                => [(4,0),(2,2),(0,4)]
        
        amp_orders : Exactly as for squared order except that this list specifies
            the contributing order values for the amplitude (i.e. not 'squared').
            Also, the tuple describing the amplitude order is nested with a 
            second one listing all amplitude numbers contributing to this order.
            Ex for dijet with process['split_orders']=['QCD','QED']: 
                => [((2, 0), (2,)), ((0, 2), (1, 3, 4))]

        Keep in mind that the orders of the element of the list is important as
        it dicatates the order of the corresponding "order indices" in the
        code output by the exporters.
        """
        
        split_orders=self.get('processes')[0].get('split_orders')
        # If no split_orders are defined, then return the obvious
        if len(split_orders)==0:
            return (),()
        
        # First make sure that the 'split_orders' are ordered according to their
        # weight.
        self.sort_split_orders(split_orders)
            
        amp_orders = self.get_split_orders_mapping_for_diagram_list(\
                                              self.get('diagrams'),split_orders)
            
        # Now we construct the interference splitting order matrix for 
        # convenience
        squared_orders = []
        for i, amp_order in enumerate(amp_orders):
            for j in range(0,i+1):
                key = tuple([ord1 + ord2 for ord1,ord2 in \
                                            zip(amp_order[0],amp_orders[j][0])])
                # We don't use the set structure here since keeping the
                # construction order guarantees us that the squared_orders will
                # also be ordered according to the WEIGHT.
                if not key in squared_orders:
                    squared_orders.append(key)

        return squared_orders, amp_orders
            
            

    def get_used_lorentz(self):
        """Return a list of (lorentz_name, conjugate_tag, outgoing) with
        all lorentz structures used by this HelasMatrixElement."""

        output = []
        for wa in self.get_all_wavefunctions() + self.get_all_amplitudes():
            if wa.get('interaction_id') in [0,-1]:
                continue
            output.append(wa.get_aloha_info());

        return output

    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        tmp = [wa.get('coupling') for wa in \
                self.get_all_wavefunctions() + self.get_all_amplitudes() \
                if wa.get('interaction_id') not in [0,-1]]
        #some coupling have a minus one associated -> need to remove those
        return [ [t] if not t.startswith('-') else [t[1:]] for t2 in tmp for t in t2]


    def get_mirror_processes(self):
        """Return a list of processes with initial states interchanged
        if has mirror processes"""

        if not self.get('has_mirror_process'):
            return []
        processes = base_objects.ProcessList()
        for proc in self.get('processes'):
            legs = copy.copy(proc.get('legs'))
            legs[0:2] = [legs[1],legs[0]]
            decay_legs = copy.copy(proc.get('legs_with_decays'))
            decay_legs[0:2] = [decay_legs[1],decay_legs[0]]
            process = copy.copy(proc)
            process.set('legs', legs)
            process.set('legs_with_decays', decay_legs)
            processes.append(process)
        return processes

    @staticmethod
    def check_equal_decay_processes(decay1, decay2):
        """Check if two single-sided decay processes
        (HelasMatrixElements) are equal.

        Note that this has to be called before any combination of
        processes has occured.
        
        Since a decay processes for a decay chain is always generated
        such that all final state legs are completely contracted
        before the initial state leg is included, all the diagrams
        will have identical wave function, independently of the order
        of final state particles.
        
        Note that we assume that the process definitions have all
        external particles, corresponding to the external
        wavefunctions.
        """
        
        assert len(decay1.get('processes')) == 1 == len(decay2.get('processes')), \
                  "Can compare only single process HelasMatrixElements"

        assert len(filter(lambda leg: leg.get('state') == False, \
                      decay1.get('processes')[0].get('legs'))) == 1 and \
               len(filter(lambda leg: leg.get('state') == False, \
                      decay2.get('processes')[0].get('legs'))) == 1, \
                  "Call to check_decay_processes_equal requires " + \
                  "both processes to be unique"

        # Compare bulk process properties (number of external legs,
        # identity factors, number of diagrams, number of wavefunctions
        # initial leg, final state legs
        if len(decay1.get('processes')[0].get("legs")) != \
           len(decay2.get('processes')[0].get("legs")) or \
           len(decay1.get('diagrams')) != len(decay2.get('diagrams')) or \
           decay1.get('identical_particle_factor') != \
           decay2.get('identical_particle_factor') or \
           sum(len(d.get('wavefunctions')) for d in \
               decay1.get('diagrams')) != \
           sum(len(d.get('wavefunctions')) for d in \
               decay2.get('diagrams')) or \
           decay1.get('processes')[0].get('legs')[0].get('id') != \
           decay2.get('processes')[0].get('legs')[0].get('id') or \
           sorted([leg.get('id') for leg in \
                   decay1.get('processes')[0].get('legs')[1:]]) != \
           sorted([leg.get('id') for leg in \
                   decay2.get('processes')[0].get('legs')[1:]]):
            return False

        # Run a quick check to see if the processes are already
        # identical (i.e., the wavefunctions are in the same order)
        if [leg.get('id') for leg in \
            decay1.get('processes')[0].get('legs')] == \
           [leg.get('id') for leg in \
            decay2.get('processes')[0].get('legs')] and \
            decay1 == decay2:
            return True

        # Now check if all diagrams are identical. This is done by a
        # recursive function starting from the last wavefunction
        # (corresponding to the initial state), since it is the
        # same steps for each level in mother wavefunctions

        amplitudes2 = copy.copy(reduce(lambda a1, d2: a1 + \
                                       d2.get('amplitudes'),
                                       decay2.get('diagrams'), []))

        for amplitude1 in reduce(lambda a1, d2: a1 + d2.get('amplitudes'),
                                  decay1.get('diagrams'), []):
            foundamplitude = False
            for amplitude2 in amplitudes2:
                if HelasMatrixElement.check_equal_wavefunctions(\
                   amplitude1.get('mothers')[-1],
                   amplitude2.get('mothers')[-1]):
                    foundamplitude = True
                    # Remove amplitude2, since it has already been matched
                    amplitudes2.remove(amplitude2)
                    break
            if not foundamplitude:
                return False

        return True

    @staticmethod
    def check_equal_wavefunctions(wf1, wf2):
        """Recursive function to check if two wavefunctions are equal.
        First check that mothers have identical pdg codes, then repeat for
        all mothers with identical pdg codes."""

        # End recursion with False if the wavefunctions do not have
        # the same mother pdgs
        if sorted([wf.get('pdg_code') for wf in wf1.get('mothers')]) != \
           sorted([wf.get('pdg_code') for wf in wf2.get('mothers')]):
            return False

        # End recursion with True if these are external wavefunctions
        # (note that we have already checked that the pdgs are
        # identical)
        if not wf1.get('mothers') and not wf2.get('mothers'):
            return True

        mothers2 = copy.copy(wf2.get('mothers'))

        for mother1 in wf1.get('mothers'):
            # Compare mother1 with all mothers in wf2 that have not
            # yet been used and have identical pdg codes
            equalmothers = filter(lambda wf: wf.get('pdg_code') == \
                                  mother1.get('pdg_code'),
                                  mothers2)
            foundmother = False
            for mother2 in equalmothers:
                if HelasMatrixElement.check_equal_wavefunctions(\
                    mother1, mother2):
                    foundmother = True
                    # Remove mother2, since it has already been matched
                    mothers2.remove(mother2)
                    break
            if not foundmother:
                return False

        return True

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. The order of the particles in the interaction
        2. Cyclic reordering of particles in same spin group
        3. Fermions ordered IOIOIO... according to the pairs in
           the interaction."""

        assert isinstance(arg, (HelasWavefunction, HelasAmplitude)), \
            "%s is not a valid HelasWavefunction or HelasAmplitude" % repr(arg)

        if not arg.get('interaction_id'):
            return arg.get('mothers')

        my_pdg_code = 0
        my_spin = 0
        if isinstance(arg, HelasWavefunction):
            my_pdg_code = arg.get_anti_pdg_code()
            my_spin = arg.get_spin_state_number()

        sorted_mothers, my_index = arg.get('mothers').sort_by_pdg_codes(\
            arg.get('pdg_codes'), my_pdg_code)

        # If fermion, partner is the corresponding fermion flow partner
        partner = None
        if isinstance(arg, HelasWavefunction) and arg.is_fermion():
            # Fermion case, just pick out the fermion flow partner
            if my_index % 2 == 0:
                # partner is after arg
                partner_index = my_index
            else:
                # partner is before arg
                partner_index = my_index - 1
            partner = sorted_mothers.pop(partner_index)
            # If partner is incoming, move to before arg
            if partner.get_spin_state_number() > 0:
                my_index = partner_index
            else:
                my_index = partner_index + 1

        # Reorder fermions pairwise according to incoming/outgoing
        for i in range(0, len(sorted_mothers), 2):
            if sorted_mothers[i].is_fermion():
                # This is a fermion, order between this fermion and its brother
                if sorted_mothers[i].get_spin_state_number() > 0 and \
                   sorted_mothers[i + 1].get_spin_state_number() < 0:
                    # Switch places between outgoing and incoming
                    sorted_mothers = sorted_mothers[:i] + \
                                      [sorted_mothers[i+1], sorted_mothers[i]] + \
                                      sorted_mothers[i+2:]
                elif sorted_mothers[i].get_spin_state_number() < 0 and \
                   sorted_mothers[i + 1].get_spin_state_number() > 0:
                    # This is the right order
                    pass
            else:
                # No more fermions in sorted_mothers
                break
            
        # Put back partner into sorted_mothers
        if partner:
            sorted_mothers.insert(partner_index, partner)

        # Next sort according to spin_state_number
        return HelasWavefunctionList(sorted_mothers)

#===============================================================================
# HelasMatrixElementList
#===============================================================================
class HelasMatrixElementList(base_objects.PhysicsObjectList):
    """List of HelasMatrixElement objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasMatrixElement for the list."""

        return isinstance(obj, HelasMatrixElement)
    
    def remove(self,obj):
        pos = (i for i in xrange(len(self)) if self[i] is obj)
        for i in pos:
            del self[i]
            break

#===============================================================================
# HelasDecayChainProcess
#===============================================================================
class HelasDecayChainProcess(base_objects.PhysicsObject):
    """HelasDecayChainProcess: If initiated with a DecayChainAmplitude
    object, generates the HelasMatrixElements for the core process(es)
    and decay chains. Then call combine_decay_chain_processes in order
    to generate the matrix elements for all combined processes."""

    def default_setup(self):
        """Default values for all properties"""

        self['core_processes'] = HelasMatrixElementList()
        self['decay_chains'] = HelasDecayChainProcessList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'core_processes':
            if not isinstance(value, HelasMatrixElementList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMatrixElementList object" % \
                        str(value)

        if name == 'decay_chains':
            if not isinstance(value, HelasDecayChainProcessList):
                raise self.PhysicsObjectError, \
                     "%s is not a valid HelasDecayChainProcessList object" % \
                     str(value)

        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['core_processes', 'decay_chains']

    def __init__(self, argument=None):
        """Allow initialization with DecayChainAmplitude"""

        if isinstance(argument, diagram_generation.DecayChainAmplitude):
            super(HelasDecayChainProcess, self).__init__()
            self.generate_matrix_elements(argument)
        elif argument:
            # call the mother routine
            super(HelasDecayChainProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(HelasDecayChainProcess, self).__init__()

    def nice_string(self, indent = 0):
        """Returns a nicely formatted string of the matrix element processes."""

        mystr = ""

        for process in self.get('core_processes'):
            mystr += process.get('processes')[0].nice_string(indent) + "\n"

        if self.get('decay_chains'):
            mystr += " " * indent + "Decays:\n"
        for dec in self.get('decay_chains'):
            mystr += dec.nice_string(indent + 2) + "\n"

        return  mystr[:-1]

    def generate_matrix_elements(self, dc_amplitude):
        """Generate the HelasMatrixElements for the core processes and
        decay processes (separately)"""

        assert isinstance(dc_amplitude, diagram_generation.DecayChainAmplitude), \
                        "%s is not a valid DecayChainAmplitude" % dc_amplitude


        # Extract the pdg codes of all particles decayed by decay chains
        # since these should not be combined in a MultiProcess
        decay_ids = dc_amplitude.get_decay_ids()

        matrix_elements = HelasMultiProcess.generate_matrix_elements(\
                               dc_amplitude.get('amplitudes'),
                               False,
                               decay_ids)

        self.set('core_processes', matrix_elements)

        while dc_amplitude.get('decay_chains'):
            # Pop the amplitude to save memory space
            decay_chain = dc_amplitude.get('decay_chains').pop(0)
            self['decay_chains'].append(HelasDecayChainProcess(\
                decay_chain))
            

    def combine_decay_chain_processes(self, combine=True):
        """Recursive function to generate complete
        HelasMatrixElements, combining the core process with the decay
        chains.

        * If the number of decay chains is the same as the number of
        decaying particles, apply each decay chain to the corresponding
        final state particle.
        * If the number of decay chains and decaying final state particles
        don't correspond, all decays applying to a given particle type are
        combined (without double counting).
        * combine allow to merge identical ME
        """

        # End recursion when there are no more decay chains
        if not self['decay_chains']:
            # Just return the list of matrix elements
            return self['core_processes']

        # decay_elements is a list of HelasMatrixElementLists with
        # all decay processes
        decay_elements = []

        for decay_chain in self['decay_chains']:
            # This is where recursion happens
            decay_elements.append(decay_chain.combine_decay_chain_processes(combine))

        # Store the result in matrix_elements
        matrix_elements = HelasMatrixElementList()
        # Store matrix element tags in me_tags, for precise comparison
        me_tags = []
        # Store external id permutations
        permutations = []
        
        # List of list of ids for the initial state legs in all decay
        # processes
        decay_is_ids = [[element.get('processes')[0].get_initial_ids()[0] \
                         for element in elements]
                         for elements in decay_elements]

        while self['core_processes']:
            # Pop the process to save memory space
            core_process = self['core_processes'].pop(0)
            # Get all final state legs that have a decay chain defined
            fs_legs = filter(lambda leg: any([any([id == leg.get('id') for id \
                            in is_ids]) for is_ids in decay_is_ids]),
                            core_process.get('processes')[0].get_final_legs())
            # List of ids for the final state legs
            fs_ids = [leg.get('id') for leg in fs_legs]
            # Create a dictionary from id to (index, leg number)
            fs_numbers = {}
            fs_indices = {}
            for i, leg in enumerate(fs_legs):
                fs_numbers[leg.get('id')] = \
                    fs_numbers.setdefault(leg.get('id'), []) + \
                    [leg.get('number')]
                fs_indices[leg.get('id')] = \
                    fs_indices.setdefault(leg.get('id'), []) + \
                    [i]

            decay_lists = []
            # Loop over unique final state particle ids
            for fs_id in set(fs_ids):
                # decay_list has the leg numbers and decays for this
                # fs particle id:
                # decay_list = [[[n1,d1],[n2,d2]],[[n1,d1'],[n2,d2']],...]

                decay_list = []

                # Two cases: Either number of decay elements is same
                # as number of decaying particles: Then use the
                # corresponding decay for each particle. Or the number
                # of decay elements is different: Then use any decay
                # chain which defines the decay for this particle.

                chains = []
                if len(fs_legs) == len(decay_elements) and \
                       all([fs in ids for (fs, ids) in \
                             zip(fs_ids, decay_is_ids)]):
                    # The decay of the different fs parts is given
                    # by the different decay chains, respectively.
                    # Chains is a list of matrix element lists
                    for index in fs_indices[fs_id]:
                        chains.append(filter(lambda me: \
                                             me.get('processes')[0].\
                                             get_initial_ids()[0] == fs_id,
                                             decay_elements[index]))

                if len(fs_legs) != len(decay_elements) or not chains or not chains[0]:
                    # In second case, or no chains are found
                    # (e.g. because the order of decays is reversed),
                    # all decays for this particle type are used
                    chain = sum([filter(lambda me: \
                                        me.get('processes')[0].\
                                        get_initial_ids()[0] == fs_id,
                                        decay_chain) for decay_chain in \
                                 decay_elements], [])

                    chains = [chain] * len(fs_numbers[fs_id])

                red_decay_chains = []
                for prod in itertools.product(*chains):

                    # Now, need to ensure that we don't append
                    # duplicate chain combinations, e.g. (a>bc, a>de) and
                    # (a>de, a>bc)
                    
                    # Remove double counting between final states
                    if sorted([p.get('processes')[0] for p in prod],
                              lambda x1, x2: x1.compare_for_sort(x2)) \
                              in red_decay_chains:
                        continue
                    
                    # Store already used combinations
                    red_decay_chains.append(\
                    sorted([p.get('processes')[0] for p in prod],
                              lambda x1, x2: x1.compare_for_sort(x2)))

                    # Add the decays to the list
                    decay_list.append(zip(fs_numbers[fs_id], prod))

                decay_lists.append(decay_list)

            # Finally combine all decays for this process,
            # and combine them, decay by decay
            for decays in itertools.product(*decay_lists):
                
                # Generate a dictionary from leg number to decay process
                decay_dict = dict(sum(decays, []))

                # Make sure to not modify the original matrix element
                model_bk = core_process.get('processes')[0].get('model')
                # Avoid Python copying the complete model every time
                for i, process in enumerate(core_process.get('processes')):
                    process.set('model',base_objects.Model())
                matrix_element = copy.deepcopy(core_process)
                # Avoid Python copying the complete model every time
                for i, process in enumerate(matrix_element.get('processes')):
                    process.set('model', model_bk)
                    core_process.get('processes')[i].set('model', model_bk)
                # Need to replace Particle in all wavefunctions to avoid
                # deepcopy
                org_wfs = core_process.get_all_wavefunctions()
                for i, wf in enumerate(matrix_element.get_all_wavefunctions()):
                    wf.set('particle', org_wfs[i].get('particle'))
                    wf.set('antiparticle', org_wfs[i].get('antiparticle'))

                # Insert the decay chains
                logger.info("Combine %s with decays %s" % \
                            (core_process.get('processes')[0].nice_string().\
                             replace('Process: ', ''), \
                             ", ".join([d.get('processes')[0].nice_string().\
                                        replace('Process: ', '') \
                                        for d in decay_dict.values()])))
                    
                matrix_element.insert_decay_chains(decay_dict)    
                
                if combine:
                    me_tag = IdentifyMETag.create_tag(\
                            matrix_element.get_base_amplitude(),
                            matrix_element.get('identical_particle_factor'))
                try:
                    if not combine:
                        raise ValueError
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    me_index = me_tags.index(me_tag)
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.get('processes') and \
                           matrix_element.get('diagrams'):
                        matrix_elements.append(matrix_element)
                        if combine:
                            me_tags.append(me_tag)
                            permutations.append(me_tag[-1][0].\
                                            get_external_numbers())
                else: # try
                    other_processes = matrix_elements[me_index].get('processes')
                    logger.info("Combining process with %s" % \
                      other_processes[0].nice_string().replace('Process: ', ''))
                    for proc in matrix_element.get('processes'):
                        other_processes.append(HelasMultiProcess.\
                              reorder_process(proc,
                                   permutations[me_index],
                                   me_tag[-1][0].get_external_numbers()))

        return matrix_elements

#===============================================================================
# HelasDecayChainProcessList
#===============================================================================
class HelasDecayChainProcessList(base_objects.PhysicsObjectList):
    """List of HelasDecayChainProcess objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasDecayChainProcess for the list."""

        return isinstance(obj, HelasDecayChainProcess)

#===============================================================================
# HelasMultiProcess
#===============================================================================
class HelasMultiProcess(base_objects.PhysicsObject):
    """HelasMultiProcess: If initiated with an AmplitudeList,
    generates the HelasMatrixElements for the Amplitudes, identifying
    processes with identical matrix elements"""

    def default_setup(self):
        """Default values for all properties"""

        self['matrix_elements'] = HelasMatrixElementList()
        
    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'matrix_elements':
            if not isinstance(value, HelasMatrixElementList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMatrixElementList object" % str(value)
        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['matrix_elements']

    def __init__(self, argument=None, combine_matrix_elements=True,
                 matrix_element_opts={}, compute_loop_nc = False):
        """Allow initialization with AmplitudeList. Matrix_element_opts are 
        potential options to be passed to the constructor of the 
        HelasMatrixElements created. By default it is none, but when called from
        LoopHelasProcess, this options will contain 'optimized_output'."""


        if isinstance(argument, diagram_generation.AmplitudeList):
            super(HelasMultiProcess, self).__init__()
            self.set('matrix_elements', self.generate_matrix_elements(argument,
                            combine_matrix_elements = combine_matrix_elements,
                            matrix_element_opts=matrix_element_opts,
                            compute_loop_nc = compute_loop_nc))
        elif isinstance(argument, diagram_generation.MultiProcess):
            super(HelasMultiProcess, self).__init__()
            self.set('matrix_elements',
                     self.generate_matrix_elements(argument.get('amplitudes'),
                             combine_matrix_elements = combine_matrix_elements,
                             matrix_element_opts = matrix_element_opts,
                             compute_loop_nc = compute_loop_nc))
        elif isinstance(argument, diagram_generation.Amplitude):
            super(HelasMultiProcess, self).__init__()
            self.set('matrix_elements', self.generate_matrix_elements(\
                             diagram_generation.AmplitudeList([argument]),
                             combine_matrix_elements = combine_matrix_elements,
                             matrix_element_opts = matrix_element_opts,
                             compute_loop_nc = compute_loop_nc))
        elif argument:
            # call the mother routine
            super(HelasMultiProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(HelasMultiProcess, self).__init__()

    def get_used_lorentz(self):
        """Return a list of (lorentz_name, conjugate, outgoing) with
        all lorentz structures used by this HelasMultiProcess."""
        helas_list = []

        for me in self.get('matrix_elements'):
            helas_list.extend(me.get_used_lorentz())
                
        return list(set(helas_list))

    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""
        
        coupling_list = []

        for me in self.get('matrix_elements'):
            coupling_list.extend([c for l in me.get_used_couplings() for c in l])
        
        return list(set(coupling_list))
    
    def get_matrix_elements(self):
        """Extract the list of matrix elements"""

        return self.get('matrix_elements')

    #===========================================================================
    # generate_matrix_elements
    #===========================================================================

    @classmethod
    def process_color(cls,matrix_element, color_information, compute_loop_nc=None):
        """ Process the color information for a given matrix
        element made of a tree diagram. compute_loop_nc is dummy here for the
        tree-level Nc and present for structural reasons only."""
        
        if compute_loop_nc:
            raise MadGraph5Error, "The tree-level function 'process_color' "+\
             " of class HelasMultiProcess cannot be called with a value for compute_loop_nc"
        
        # Define the objects stored in the contained color_information
        for key in color_information:
            exec("%s=color_information['%s']"%(key,key))
        
        # Always create an empty color basis, and the
        # list of raw colorize objects (before
        # simplification) associated with amplitude
        col_basis = color_amp.ColorBasis()
        new_amp = matrix_element.get_base_amplitude()
        matrix_element.set('base_amplitude', new_amp)
        
        colorize_obj = col_basis.create_color_dict_list(\
                         matrix_element.get('base_amplitude'))
        #list_colorize = []
        #list_color_basis = []
        #list_color_matrices = []
        
        try:
            # If the color configuration of the ME has
            # already been considered before, recycle
            # the information
            col_index = list_colorize.index(colorize_obj)
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
              matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                             replace('Process', 'process'))
        else: # Found identical color
            logger.info(\
              "Reusing existing color information for %s" % \
              matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                                 replace('Process', 'process'))

        matrix_element.set('color_basis',
                               list_color_basis[col_index])
        matrix_element.set('color_matrix',
                               list_color_matrices[col_index])

    # Below is the type of HelasMatrixElement which should be created by this
    # HelasMultiProcess class
    matrix_element_class = HelasMatrixElement

    @classmethod
    def generate_matrix_elements(cls, amplitudes, gen_color = True,
        decay_ids = [], combine_matrix_elements = True, 
        compute_loop_nc = False, matrix_element_opts = {}):
        """Generate the HelasMatrixElements for the amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__. Returns a
        HelasMatrixElementList and an amplitude map (used by the
        SubprocessGroup functionality). decay_ids is a list of decayed
        particle ids, since those should not be combined even if
        matrix element is identical. 
        The compute_loop_nc sets wheter independent tracking of Nc power coming
        from the color loop trace is necessary or not (it is time consuming).
        Matrix_element_opts are potential additional options to be passed to 
        the HelasMatrixElements constructed."""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "%s is not valid AmplitudeList" % type(amplitudes)

        combine = combine_matrix_elements
        if 'mode' in matrix_element_opts and matrix_element_opts['mode']=='MadSpin':
            combine = False
            del matrix_element_opts['mode']

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

        # List of valid matrix elements
        matrix_elements = HelasMatrixElementList()
        # List of identified matrix_elements
        identified_matrix_elements = []
        # List of amplitude tags, synchronized with identified_matrix_elements
        amplitude_tags = []
        # List of the external leg permutations for the amplitude_tags,
        # which allows to reorder the final state particles in the right way
        # for maximal process combination
        permutations = []
        for amplitude in amplitudes:
            if isinstance(amplitude, diagram_generation.DecayChainAmplitude):
                # Might get multiple matrix elements from this amplitude
                tmp_matrix_element_list = HelasDecayChainProcess(amplitude).\
                                          combine_decay_chain_processes(combine)
                # Use IdentifyMETag to check if matrix elements present
                matrix_element_list = []
                for matrix_element in tmp_matrix_element_list:
                    assert isinstance(matrix_element, HelasMatrixElement), \
                              "Not a HelasMatrixElement: %s" % matrix_element

                    # If the matrix element has no diagrams,
                    # remove this matrix element.
                    if not matrix_element.get('processes') or \
                           not matrix_element.get('diagrams'):
                        continue

                    # Create IdentifyMETag
                    amplitude_tag = IdentifyMETag.create_tag(\
                                    matrix_element.get_base_amplitude())
                    try:
                        if not combine:
                            raise ValueError
                        me_index = amplitude_tags.index(amplitude_tag)
                    except ValueError:
                        # Create matrix element for this amplitude
                        matrix_element_list.append(matrix_element)
                        if combine_matrix_elements:
                            amplitude_tags.append(amplitude_tag)
                            identified_matrix_elements.append(matrix_element)
                            permutations.append(amplitude_tag[-1][0].\
                                                get_external_numbers())
                    else: # try
                        # Identical matrix element found
                        other_processes = identified_matrix_elements[me_index].\
                                          get('processes')
                        # Reorder each of the processes
                        # Since decay chain, only reorder legs_with_decays
                        for proc in matrix_element.get('processes'):
                            other_processes.append(cls.reorder_process(\
                                    proc,
                                    permutations[me_index],
                                    amplitude_tag[-1][0].get_external_numbers()))
                        logger.info("Combined %s with %s" % \
                                    (matrix_element.get('processes')[0].\
                                     nice_string().\
                                     replace('Process: ', 'process '),
                                     other_processes[0].nice_string().\
                                     replace('Process: ', 'process ')))
                        # Go on to next matrix element
                        continue
            else: # not DecayChainAmplitude
                # Create tag identifying the matrix element using
                # IdentifyMETag. If two amplitudes have the same tag,
                # they have the same matrix element
                amplitude_tag = IdentifyMETag.create_tag(amplitude)
                try:
                    me_index = amplitude_tags.index(amplitude_tag)
                except ValueError:
                    # Create matrix element for this amplitude
                    logger.info("Generating Helas calls for %s" % \
                            amplitude.get('process').nice_string().\
                                           replace('Process', 'process'))
                    # Correctly choose the helas matrix element class depending
                    # on the type of 'HelasMultiProcess' this static function
                    # is called from.
                    matrix_element_list = [cls.matrix_element_class(amplitude,
                                                          decay_ids=decay_ids,
                                                          gen_color=False,
                                                         **matrix_element_opts)]
                    me = matrix_element_list[0]
                    if me.get('processes') and me.get('diagrams'):
                        # Keep track of amplitude tags
                        if combine_matrix_elements:
                            amplitude_tags.append(amplitude_tag)
                            identified_matrix_elements.append(me)
                            permutations.append(amplitude_tag[-1][0].\
                                                get_external_numbers())
                    else:
                        matrix_element_list = []
                else:
                    # Identical matrix element found
                    other_processes = identified_matrix_elements[me_index].\
                                      get('processes')
                    other_processes.append(cls.reorder_process(\
                        amplitude.get('process'),
                        permutations[me_index],
                        amplitude_tag[-1][0].get_external_numbers()))
                    logger.info("Combined %s with %s" % \
                                (other_processes[-1].nice_string().\
                                 replace('Process: ', 'process '),
                                 other_processes[0].nice_string().\
                                 replace('Process: ', 'process ')))
                    # Go on to next amplitude
                    continue
            
            # Deal with newly generated matrix elements
            for matrix_element in copy.copy(matrix_element_list):
                assert isinstance(matrix_element, HelasMatrixElement), \
                          "Not a HelasMatrixElement: %s" % matrix_element

                # Add this matrix element to list
                matrix_elements.append(matrix_element)

                if not gen_color:
                    continue

                # The treatment of color is quite different for loop amplitudes
                # than for regular tree ones. So the function below is overloaded
                # in LoopHelasProcess
                cls.process_color(matrix_element,color_information,\
                                                compute_loop_nc=compute_loop_nc)                    

        if not matrix_elements:
            raise InvalidCmd, \
                  "No matrix elements generated, check overall coupling orders"

        return matrix_elements

    @staticmethod
    def reorder_process(process, org_perm, proc_perm):
        """Reorder the legs in the process according to the difference
        between org_perm and proc_perm"""

        leglist = base_objects.LegList(\
                  [copy.copy(process.get('legs_with_decays')[i]) for i in \
                   diagram_generation.DiagramTag.reorder_permutation(\
                       proc_perm, org_perm)])
        new_proc = copy.copy(process)
        new_proc.set('legs_with_decays', leglist)

        if not new_proc.get('decay_chains'):
            new_proc.set('legs', leglist)

        return new_proc

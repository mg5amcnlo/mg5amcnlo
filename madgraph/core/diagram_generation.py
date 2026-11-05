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
"""Classes for diagram generation. Amplitude performs the diagram
generation, DecayChainAmplitude keeps track of processes with decay
chains, and MultiProcess allows generation of processes with
multiparticle definitions. DiagramTag allows to identify diagrams
based on relevant properties.
"""

from __future__ import absolute_import
#force filter to be a generator # like in py3

import array
import copy
import fractions
import itertools
import logging

import madgraph
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color_algebra
import madgraph.various.misc as misc
import madgraph.fks.fks_tag as fks_tag
from madgraph import InvalidCmd, MadGraph5Error
logger = logging.getLogger('madgraph.diagram_generation')

if madgraph.ordering:
    set = misc.OrderedSet

class NoDiagramException(InvalidCmd): pass

#===============================================================================
# Interference of two amplitude sets: 'left [LIxtree=QCD] right' (loop-induced
# times tree, see loop_diagram_generation) and 'left [treextree] right'
#===============================================================================
INTERFERENCE_ORDER = base_objects.INTERFERENCE_ORDER
# Output formats able to select squared split orders, hence to return the
# interference alone (the others would give |A_left + A_right|^2): the Fortran
# ones (GET_MATRIX) and the madmatrix ones (mg7, and standalone for the C++
# matrix element), whose color sum pairs the amplitude split orders on every
# backend, CPU and GPU (PR #244)
TREE_INTERFERENCE_FORMATS = ['madevent', 'standalone_fortran', 'mg7',
                             'standalone']

def register_interference_order(model):
    """Add the hidden coupling order tagging the right-hand amplitudes of an
    interference process to the model, in memory and idempotently. Its
    hierarchy is 0, so that the WEIGHTED order of every diagram is unchanged,
    and no interaction carries it."""

    if INTERFERENCE_ORDER not in model.get('order_hierarchy'):
        model.get('order_hierarchy')[INTERFERENCE_ORDER] = 0
    coupling_orders = model.get('coupling_orders')
    if INTERFERENCE_ORDER not in coupling_orders:
        coupling_orders.add(INTERFERENCE_ORDER)

def prepare_interference_process(procdef):
    """Complete the ProcessDefinition of an interference process
    ('left [LIxtree=QCD] right' or 'left [treextree] right', see
    MadGraphCmd.extract_interference_process) before its generation:

    - the hidden order INTERFERENCE_ORDER is registered in the model;
    - the left-hand process gets the squared-order constraint INTERF^2==1
      and INTERF as split order, so that only the products of a left-hand
      amplitude with a right-hand one are kept, also in the generated code;
    - the right-hand process gets, when none is given, the minimal WEIGHTED
      order exactly as 'generate <right-hand process>' would.

    The right-hand diagrams themselves are generated per subprocess, with the
    legs of that subprocess: Amplitude.add_interference_diagrams (treextree)
    and LoopAmplitude.set_interference_trees (LIxtree)."""

    mode = procdef.get_interference_mode()
    if not mode:
        return
    if any(value < 0 for value in procdef.get('squared_orders').values()):
        raise InvalidCmd("Negative squared-order constraints are not "
                         "supported for %s interference processes." % mode)
    model = procdef.get('model')
    register_interference_order(model)

    squared_orders = dict(procdef.get('squared_orders'))
    squared_orders[INTERFERENCE_ORDER] = 1
    procdef.set('squared_orders', squared_orders)
    sqorders_types = dict(procdef.get('sqorders_types'))
    sqorders_types[INTERFERENCE_ORDER] = '=='
    procdef.set('sqorders_types', sqorders_types)
    if INTERFERENCE_ORDER not in procdef.get('split_orders'):
        procdef.set('split_orders',
                    list(procdef.get('split_orders')) + [INTERFERENCE_ORDER])

    right = procdef.get('interference_process')
    if not right.get('orders') and not right.get('born_sq_orders'):
        right.set('orders', MultiProcess.find_optimal_process_orders(right))
    right.check_expansion_orders()

def specialise_interference_process(process):
    """The right-hand process of an interference subprocess, with the
    external legs of that subprocess. It is also stored back in the
    subprocess, which then shows the tree process it interferes with."""

    right = process.get('interference_process')
    legs = base_objects.LegList([copy.copy(leg) for leg in process.get('legs')])
    if isinstance(right, base_objects.ProcessDefinition):
        right_process = right.get_process_with_legs(legs)
    else:
        # already specialised to these legs (regeneration)
        right_process = copy.copy(right)
        right_process.set('legs', legs)
    right_process.set('interference_mode', '')
    right_process.set('interference_process', None)
    # the process number is the one of the whole (left-hand) definition
    right_process.set('id', 0)
    process.set('interference_process', right_process)
    return right_process

#===============================================================================
# DiagramTag mother class
#===============================================================================

class DiagramTag(object):
    r"""Class to tag diagrams based on objects with some __lt__ measure, e.g.
    PDG code/interaction id (for comparing diagrams from the same amplitude),
    or Lorentz/coupling/mass/width (for comparing AMPs from different MEs).
    Algorithm: Create chains starting from external particles:
    1 \        / 6
    2 /\______/\ 7
    3_ /  |   \_ 8
    4 /   5    \_ 9
                \ 10
    gives ((((9,10,id910),8,id9108),(6,7,id67),id910867)
           (((1,2,id12),(3,4,id34)),id1234),
           5,id91086712345)
    where idN is the id of the corresponding interaction. The ordering within
    chains is based on chain length (depth; here, 1234 has depth 3, 910867 has
    depth 4, 5 has depht 0), and if equal on the ordering of the chain elements.
    The determination of central vertex is based on minimizing the chain length
    for the longest subchain. 
    This gives a unique tag which can be used to identify diagrams
    (instead of symmetry), as well as identify identical matrix elements from
    different processes."""

    class DiagramTagError(Exception):
        """Exception for any problems in DiagramTags"""
        pass

    def __init__(self, diagram, model=None, ninitial=2):
        """Initialize with a diagram. Create DiagramTagChainLinks according to
        the diagram, and figure out if we need to shift the central vertex."""

        # wf_dict keeps track of the intermediate particles
        leg_dict = {}
        # Create the chain which will be the diagram tag
        for vertex in diagram.get('vertices'):
            # Only add incoming legs
            legs = vertex.get('legs')[:-1]
            lastvx = vertex == diagram.get('vertices')[-1]
            if lastvx:
                # If last vertex, all legs are incoming
                legs = vertex.get('legs')
            # Add links corresponding to the relevant legs
            link = DiagramTagChainLink([leg_dict.setdefault(leg.get('number'),
                          DiagramTagChainLink(self.link_from_leg(leg, model))) \
                                        for leg in legs],
                                        self.vertex_id_from_vertex(vertex,
                                                                   lastvx,
                                                                   model,
                                                                   ninitial))
            # Add vertex to leg_dict if not last one
            if not lastvx:
                leg_dict[vertex.get('legs')[-1].get('number')] = link

        # The resulting link is the hypothetical result
        self.tag = link

        # Now make sure to find the central vertex in the diagram,
        # defined by the longest leg being as short as possible
        done = max([l.depth for l in self.tag.links]) == 0
        while not done:
            # Identify the longest chain in the tag
            longest_chain = self.tag.links[0]
            # Create a new link corresponding to moving one step
            new_link = DiagramTagChainLink(self.tag.links[1:],
                                           self.flip_vertex(\
                                               self.tag.vertex_id,
                                               longest_chain.vertex_id,
                                               self.tag.links[1:]))
            # Create a new final vertex in the direction of the longest link
            other_links = list(longest_chain.links) + [new_link]
            other_link = DiagramTagChainLink(other_links,
                                             self.flip_vertex(\
                                                 longest_chain.vertex_id,
                                                 self.tag.vertex_id,
                                                 other_links))
            
            if other_link.links[0] < self.tag.links[0]:
                # Switch to new tag, continue search
                self.tag = other_link
            else:
                # We have found the central vertex
                done = True

    def get_external_numbers(self):
        """Get the order of external particles in this tag"""
        return self.tag.get_external_numbers()

    def diagram_from_tag(self, model):
        """Output a diagram from a DiagramTag. Note that each daughter
        class must implement the static functions id_from_vertex_id
        (if the vertex id is something else than an integer) and
        leg_from_link (to pass the correct info from an end link to a
        leg)."""

        # Create the vertices, starting from the final vertex
        diagram = base_objects.Diagram({'vertices': \
                                        self.vertices_from_link(self.tag,
                                                                model,
                                                                True)})
        diagram.calculate_orders(model)
        return diagram

    @classmethod
    def vertices_from_link(cls, link, model, first_vertex = False):
        """Recursively return the leg corresponding to this link and
        the list of all vertices from all previous links"""            

        if link.end_link:
            # This is an end link and doesn't correspond to a vertex
            return cls.leg_from_link(link), []

        # First recursively find all daughter legs and vertices
        leg_vertices = [cls.vertices_from_link(l, model) for l in link.links]
        # The daughter legs are in the first entry
        legs = base_objects.LegList(sorted([l for l,v in leg_vertices],
                                           key= lambda l: l.get('number'), reverse=True))

        # The daughter vertices are in the second entry
        vertices = base_objects.VertexList(sum([v for l, v in leg_vertices],
                                               []))
        
        if not first_vertex:
            # This corresponds to a wavefunction with a resulting leg
            # Need to create the resulting leg from legs and vertex id
            last_leg = cls.leg_from_legs(legs,link.vertex_id,model)
            legs.append(last_leg)
        
        # Now create and append this vertex
        vertices.append(cls.vertex_from_link(legs,
                                        link.vertex_id,
                                        model))
        if first_vertex:
            # Return list of vertices
            return vertices
        else:
            # Return leg and list of vertices
            return last_leg, vertices

    @classmethod
    def legPDGs_from_vertex_id(cls, vertex_id,model):
        """Returns the list of external PDGs of the interaction corresponding 
        to this vertex_id."""
        
        # In case we have to deal with a regular vertex, we return the list
        # external PDGs as given by the model information on that integer 
        # vertex id.
        if (len(vertex_id)>=3 and 'PDGs' in vertex_id[2]):
            return vertex_id[2]['PDGs']
        else:
            return [part.get_pdg_code() for part in model.get_interaction(
                             cls.id_from_vertex_id(vertex_id)).get('particles')]

    @classmethod
    def leg_from_legs(cls,legs, vertex_id, model):
        """Return a leg from a leg list and the model info"""

        pdgs = list(cls.legPDGs_from_vertex_id(vertex_id, model))
        
        # Extract the resulting pdg code from the interaction pdgs
        for pdg in [leg.get('id') for leg in legs]:
            pdgs.remove(pdg)

        assert len(pdgs) == 1
        # Prepare the new leg properties
        pdg = model.get_particle(pdgs[0]).get_anti_pdg_code()
        number = min([l.get('number') for l in legs])
        # State is False for t-channel, True for s-channel
        state = (len([l for l in legs if l.get('state') == False]) != 1)
        # Note that this needs to be done before combining decay chains
        onshell= False

        return base_objects.Leg({'id': pdg,
                                 'number': number,
                                 'state': state,
                                 'onshell': onshell})

    @classmethod
    def vertex_from_link(cls, legs, vertex_id, model):
        """Return a vertex given a leg list and a vertex id"""

        vert_ID = cls.id_from_vertex_id(vertex_id)
        if vert_ID != -2:
            return base_objects.Vertex({'legs': legs,
                                        'id': vert_ID})
        else:
            contracted_vert = base_objects.ContractedVertex({'legs': legs,'id': -2})
            for key, value in cls.loop_info_from_vertex_id(vertex_id):
                if key in contracted_vert:
                    contracted_vert.set(key,value)    

    @staticmethod
    def leg_from_link(link):
        """Return a leg from a link"""

        if link.end_link:
            # This is an external leg, info in links
            return base_objects.Leg({'number':link.links[0][1],
                                     'id':link.links[0][0][0],
                                     'state':(link.links[0][0][1] == 0),
                                     'onium': {},
                                     'onshell':False})

        # This shouldn't happen
        assert False

    @staticmethod
    def id_from_vertex_id(vertex_id):
        """Return the numerical vertex id from a link.vertex_id"""

        return vertex_id[0][0]
    
    @staticmethod
    def loop_info_from_vertex_id(vertex_id):
        """Return the loop_info stored in this vertex id. Notice that the
        IdentifyME tag does not store the loop_info, but should normally never
        need access to it."""

        return vertex_id[2]

    @staticmethod
    def reorder_permutation(perm, start_perm):
        """Reorder a permutation with respect to start_perm. Note that
        both need to start from 1."""
        if perm == start_perm:
            return list(range(len(perm)))
        order = [i for (p,i) in \
                 sorted([(p,i) for (i,p) in enumerate(perm)])]
        return [start_perm[i]-1 for i in order]

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the default end link for a leg: ((id, state, onium), number).
        Note that the number is not taken into account if tag comparison,
        but is used only to extract leg permutations."""
        if leg.get('state'):
            # Identify identical final state particles
            return [((leg.get('id'), 0), leg.get('number'))]
        else:
            # Distinguish identical initial state particles
            return [((leg.get('id'), leg.get('number'), leg.get('onium')), leg.get('number'))]

    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the default vertex id: just the interaction id
           Note that in the vertex id, like the leg, only the first entry is
           taken into account in the tag comparison, while the second is for 
           storing information that is not to be used in comparisons and the 
           third for additional info regarding the shrunk loop vertex."""

        if isinstance(vertex,base_objects.ContractedVertex):
#            return (vertex.get('id'),(),{'PDGs':vertex.get('PDGs')})
            return ((vertex.get('id'),vertex.get('loop_tag')),(),
                                                    {'PDGs':vertex.get('PDGs')})
        else:
            return ((vertex.get('id'),()),(),{})

    @staticmethod
    def flip_vertex(new_vertex, old_vertex, links):
        """Returns the default vertex flip: just the new_vertex"""
        return new_vertex

    def __eq__(self, other):
        """Equal if same tag"""
        if type(self) != type(other):
            return False
        return self.tag == other.tag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.tag)

    def __lt__(self, other):
        return self.tag < other.tag

    def __gt__(self, other):
        return self.tag > other.tag

    __repr__ = __str__

class DiagramTagChainLink(object):
    """Chain link for a DiagramTag. A link is a tuple + vertex id + depth,
    with a comparison operator defined"""

    def __init__(self, objects, vertex_id = None):
        """Initialize, either with a tuple of DiagramTagChainLinks and
        a vertex_id (defined by DiagramTag.vertex_id_from_vertex), or
        with an external leg object (end link) defined by
        DiagramTag.link_from_leg"""

        if vertex_id == None:
            # This is an end link, corresponding to an external leg
            self.links = tuple(objects)
            self.vertex_id = (0,)
            self.depth = 0
            self.end_link = True
            return
        # This is an internal link, corresponding to an internal line
        self.links = tuple(sorted(list(tuple(objects)), reverse=True))
        self.vertex_id = vertex_id
        # depth = sum(depth for links) + max(1, len(self.links)-1)
        # in order to get depth 2 for a 4-particle vertex
        self.depth = sum([l.depth for l in self.links],
                         max(1, len(self.links)-1))
        self.end_link = False

    def get_external_numbers(self):
        """Get the permutation of external numbers (assumed to be the
        second entry in the end link tuples)"""

        if self.end_link:
            #misc.sprint(self.links[0][1])
            return [self.links[0][1]]

        return sum([l.get_external_numbers() for l in self.links], [])

    def __lt__(self, other):
        """Compare self with other in the order:
        1. depth 2. len(links) 3. vertex id 4. measure of links"""

        if self == other:
            return False

        if self.depth != other.depth:
            return self.depth < other.depth

        if len(self.links) != len(other.links):
            return len(self.links) < len(other.links)

        if self.vertex_id[0] != other.vertex_id[0]:
            if isinstance(self.vertex_id[0], int) and isinstance(other.vertex_id[0], tuple):
                return True
            elif isinstance(self.vertex_id[0], tuple) and isinstance(other.vertex_id[0], int):
                return False
            elif isinstance(self.vertex_id[0], str) and isinstance(other.vertex_id[0], tuple):
                return True
            elif isinstance(self.vertex_id[0], tuple) and isinstance(other.vertex_id[0], str):
                return False            
            else:
                try:
                    return self.vertex_id[0] < other.vertex_id[0]
                except TypeError as error:
                    if error.args == ("'<' not supported between instances of 'tuple' and 'str'",):
                        return False
                    elif error.args == ("'<' not supported between instances of 'str' and 'tuple'",):
                        return True
                    else:
                        raise Exception
                    

        for i, link in enumerate(self.links):
            if i > len(other.links) - 1:
                return False
            if link != other.links[i]:
                return link < other.links[i]

    def __gt__(self, other):
        return self != other and not self.__lt__(other)

    def __eq__(self, other):
        """For end link,
        consider equal if self.links[0][0] == other.links[0][0],
        i.e., ignore the leg number (in links[0][1])."""

        if self.end_link and other.end_link and self.depth == other.depth \
           and self.vertex_id == other.vertex_id:
            return self.links[0][0] == other.links[0][0]
        
        return self.end_link == other.end_link and self.depth == other.depth \
            and self.vertex_id[0] == other.vertex_id[0] \
            and self.links == other.links 

    def __ne__(self, other):
        return not self.__eq__(other)


    def __str__(self):
        if self.end_link:
            return str(self.links)
        return "%s, %s; %d" % (str(self.links),
                               str(self.vertex_id),
                               self.depth)

    __repr__ = __str__

#===============================================================================
# Unrolling of quartic vertices into pairs of cubic vertices
#===============================================================================

class UnrollDiagramTag(DiagramTag):
    """DiagramTag which keeps every external leg distinct.

    The default DiagramTag deliberately identifies identical final state
    particles, which is what is wanted when comparing different processes. To
    recognise a diagram inside its own amplitude we need the leg numbers, so
    that e.g. the t- and u-channel of g g > g g do not collide."""

    @staticmethod
    def link_from_leg(leg, model):
        return [((leg.get('id'), leg.get('number')), leg.get('number'))]


def get_unrollable_quartic_vertices(model):
    """Find the quartic interactions which factorise into two cubic ones.

    A quartic vertex qualifies when its four legs carry the same
    self-conjugate particle and each of its colour structures is a product of
    two structure constants sharing a single summed index -- the four gluon
    vertex being the canonical example. Such a colour structure splits the
    four colour indices into two pairs, and that splitting is precisely the
    one produced by two cubic vertices joined by an internal line.

    Returns {quartic_id: (cubic_id, pairings)} where pairings[icolor] is the
    pair of 2-tuples of colour index positions separated by the summed index.
    """

    try:
        return model._unrollable_quartic_vertices
    except AttributeError:
        pass

    # Cubic candidates: three identical legs with a single f(0,1,2) structure
    cubic_by_pdg = {}
    for inter in model.get('interactions'):
        parts = inter.get('particles')
        if len(parts) != 3:
            continue
        pdgs = misc.make_unique([p.get_pdg_code() for p in parts])
        if len(pdgs) != 1:
            continue
        color = inter.get('color')
        if len(color) != 1 or len(color[0]) != 1:
            continue
        if not isinstance(color[0][0], color_algebra.f) or \
           sorted(color[0][0]) != [0, 1, 2]:
            continue
        cubic_by_pdg[pdgs[0]] = inter.get('id')

    res = {}
    for inter in model.get('interactions'):
        parts = inter.get('particles')
        if len(parts) != 4:
            continue
        pdgs = misc.make_unique([p.get_pdg_code() for p in parts])
        if len(pdgs) != 1 or pdgs[0] not in cubic_by_pdg:
            continue
        pairings = []
        for col_str in inter.get('color'):
            pairing = _f_pair_split(col_str)
            if pairing is None:
                break
            pairings.append(pairing)
        else:
            if pairings:
                res[inter.get('id')] = (cubic_by_pdg[pdgs[0]], pairings)

    try:
        model._unrollable_quartic_vertices = res
    except AttributeError:
        pass
    return res


def _f_pair_split(col_str):
    """If col_str is f(..)*f(..) with a single shared summed index and the
    four remaining indices being 0,1,2,3, return the two pairs of external
    colour index positions it separates, otherwise None."""

    if len(col_str) != 2 or not all(isinstance(obj, color_algebra.f)
                                    for obj in col_str):
        return None
    shared = [i for i in col_str[0] if i in col_str[1]]
    if len(shared) != 1 or shared[0] >= 0:
        return None
    pairs = tuple(tuple(sorted(i for i in obj if i != shared[0]))
                  for obj in col_str)
    if sorted(pairs[0] + pairs[1]) != [0, 1, 2, 3]:
        return None
    return pairs


def glued_vertices(diagram):
    """Vertices of the diagram with the trailing identity vertex glued into
    the one before it.

    The identity vertex only states that its two legs are the two ends of one
    and the same line, so it is dropped once the diagram is complete, by
    handing that line to the vertex before it. Until that is done the same
    diagram can be written in several ways, and comparing two of them is
    meaningless. Returns None when there is nothing to glue."""

    vertices = diagram.get('vertices')
    if len(vertices) <= 1 or vertices[-1].get('id') != 0:
        return None

    vertices = copy.copy(vertices)
    lastvx = vertices.pop()
    nexttolastvertex = copy.copy(vertices.pop())
    legs = copy.copy(nexttolastvertex.get('legs'))
    ntlnumber = legs[-1].get('number')
    lastleg = [leg for leg in lastvx.get('legs')
               if leg.get('number') != ntlnumber][0]
    # Reset onshell in case we have forbidden s-channels
    if lastleg.get('onshell') == False:
        lastleg.set('onshell', None)
    # Replace the last leg of nexttolastvertex
    legs[-1] = lastleg
    nexttolastvertex.set('legs', legs)
    vertices.append(nexttolastvertex)
    return vertices


def get_unrollable_cubic_ids(model):
    """Interaction ids of the cubic vertices a factorisable quartic vertex
    unrolls into -- the three gluon vertex for the four gluon one."""

    return frozenset(cubic_id for cubic_id, pairings in
                     get_unrollable_quartic_vertices(model).values())


def colour_index_order(vertex, is_last, model):
    """Return the vertex legs in the order in which color_amp.ColorBasis maps
    them onto the colour indices of the interaction.

    This mirrors ColorBasis.add_vertex: the outgoing leg of an internal
    vertex is flipped to its antiparticle and moved to the front, then the
    legs are sorted following the particle order of the interaction. Note
    that the sorting can move the outgoing leg away from the front again, so
    its position is returned along with the legs.

    Returns (legs, outgoing_position), the position being None for the last
    vertex of a diagram, or None if the vertex does not match its
    interaction."""

    inter = model.get_interaction(vertex.get('id'))
    if inter is None:
        return None

    legs = vertex.get('legs')
    entries = []
    for index, leg in enumerate(legs):
        part = model.get('particle_dict')[leg.get('id')]
        outgoing = index == len(legs) - 1 and not is_last
        entries.append((leg, part.get_anti_pdg_code() if outgoing
                        else part.get_pdg_code(), outgoing))

    if not is_last:
        entries.insert(0, entries.pop(-1))

    ordered = []
    for pdg in [p.get_pdg_code() for p in inter.get('particles')]:
        for index, entry in enumerate(entries):
            if entry[1] == pdg:
                ordered.append(entries.pop(index))
                break
        else:
            return None

    out_position = None
    for index, entry in enumerate(ordered):
        if entry[2]:
            out_position = index

    return [entry[0] for entry in ordered], out_position


def split_quartic_vertex(vertex, is_last, pairing, cubic_id, model):
    """Replace a quartic vertex by the two cubic vertices that the given
    colour structure factorises into, joined by a new internal line."""

    ordered, out_position = colour_index_order(vertex, is_last, model)
    first, second = pairing
    if not is_last and out_position in first:
        # the pair the outgoing leg does not belong to is the one replaced by
        # the new internal line
        first, second = second, first

    combined = sorted([ordered[i] for i in first],
                      key=lambda leg: leg.get('number'))
    new_leg = base_objects.Leg({
        'id': combined[0].get('id'),
        'number': min(leg.get('number') for leg in combined),
        'state': len([l for l in combined if not l.get('state')]) != 1,
        'from_group': True})

    first_vx = base_objects.Vertex({
        'legs': base_objects.LegList(combined + [new_leg]),
        'id': cubic_id})
    if is_last:
        rest = sorted([ordered[i] for i in second] + [new_leg],
                      key=lambda leg: leg.get('number'))
    else:
        # the outgoing leg has to stay last
        rest = sorted([ordered[i] for i in second if i != out_position] +
                      [new_leg], key=lambda leg: leg.get('number')) + \
               [ordered[out_position]]
    second_vx = base_objects.Vertex({'legs': base_objects.LegList(rest),
                                     'id': cubic_id})

    return [first_vx, second_vx]


# Placeholder standing for the single summed index of a factorisable quartic
# colour structure, which is not a leg of the vertex.
_SUMMED = 'summed'


def reroot_diagram(diagram, root, model):
    """Return the same diagram decomposed around another final vertex.

    A diagram is a tree, and which of its vertices is written last decides
    which internal lines become currents: everything on the far side of the
    final vertex is built up as a wavefunction, while the final vertex only
    produces an amplitude. Re-rooting therefore changes which currents exist
    without changing the diagram -- same vertices, same lines, same physics --
    which is what lets a quartic current find the cubic current it has to be
    summed with.

    root is the index of the vertex to end on. Returns None if the diagram
    does not decompose into a tree, which should not happen.
    """

    vertices = diagram.get('vertices')
    last = len(vertices) - 1
    if root == last:
        return diagram

    # Split every vertex into the external legs it holds and the internal
    # lines it shares with another vertex. A line is recognised by its number
    # being live, since a vertex reuses the smallest incoming number for the
    # leg it produces.
    live = {}
    externals = [[] for _ in vertices]
    lines = []
    for i, vertex in enumerate(vertices):
        incoming = vertex.get('legs') if i == last else vertex.get('legs')[:-1]
        for leg in incoming:
            producer = live.pop(leg.get('number'), None)
            if producer is None:
                externals[i].append(leg)
            else:
                lines.append((producer, i,
                              vertices[producer].get('legs')[-1]))
        if i != last:
            live[vertex.get('legs')[-1].get('number')] = i
    if live:
        return None

    neighbours = {}
    line_of = {}
    for producer, consumer, leg in lines:
        neighbours.setdefault(producer, []).append(consumer)
        neighbours.setdefault(consumer, []).append(producer)
        line_of[(producer, consumer)] = (leg, True)
        line_of[(consumer, producer)] = (leg, False)

    # Walk out from the new root so that every vertex is emitted after the
    # ones now feeding it.
    order = []
    parent = {root: None}
    def visit(node):
        for other in neighbours.get(node, []):
            if other == parent[node]:
                continue
            parent[other] = node
            visit(other)
        order.append(node)
    visit(root)
    if len(order) != len(vertices):
        return None

    produced = {}
    res = base_objects.VertexList()
    for node in order:
        incoming = list(externals[node])
        for other in neighbours.get(node, []):
            if other != parent[node]:
                incoming.append(produced[other])
        legs = base_objects.LegList(incoming)
        if node != root:
            leg, forwards = line_of[(node, parent[node])]
            part = model.get('particle_dict')[leg.get('id')]
            outgoing = base_objects.Leg({
                # the line keeps its particle if it already pointed this way,
                # and is flipped when the re-rooting reverses it
                'id': leg.get('id') if forwards or part.get('self_antipart')
                      else -leg.get('id'),
                'number': min(l.get('number') for l in incoming),
                'state': len([l for l in incoming
                              if not l.get('state')]) != 1,
                'from_group': True})
            produced[node] = outgoing
            legs.append(outgoing)
        res.append(base_objects.Vertex({'legs': legs,
                                        'id': vertices[node].get('id')}))

    return base_objects.Diagram({'vertices': res})


def diagram_colour_signature(diagram, model, color_chain, unrollable):
    """Canonical signature of the colour string of one colour structure choice.

    Returns (key, coeff), where key identifies the product of colour objects
    up to the antisymmetry of the structure constants and coeff collects the
    resulting sign together with the rational prefactors. Two contributions
    sharing a key have proportional colour factors and can be summed, the
    relative weight being the ratio of their coeff.

    Every line is labelled by the set of external legs it separates rather
    than by its leg number, so that the signature can be compared between
    diagrams which number their internal lines differently. Returns None when
    the colour structure is not of a supported form.
    """

    vertices = diagram.get('vertices')
    last = len(vertices) - 1

    # A leg number is reused by the vertex which produces it, so an external
    # leg is one consumed while no internal line of that number is alive.
    alive = {}
    externals = []
    for i, vertex in enumerate(vertices):
        legs = vertex.get('legs')
        incoming = legs if i == last else legs[:-1]
        for leg in incoming:
            if alive.pop(leg.get('number'), None) is None:
                externals.append(leg.get('number'))
        if i != last:
            alive[legs[-1].get('number')] = True
    externals = frozenset(externals)
    if not externals:
        return None
    reference = min(externals)

    def normalise(side):
        """Root independent label of the line splitting side from its rest."""
        return side if reference not in side else externals - side

    factors = []
    coeff = fractions.Fraction(1, 1)
    imaginary = False
    live = {}
    for i, vertex in enumerate(vertices):
        inter = model.get_interaction(vertex.get('id'))
        if inter is None:
            return None
        order = colour_index_order(vertex, i == last, model)
        if order is None:
            return None
        ordered, out_position = order
        sides = [None] * len(ordered)
        below = frozenset()
        # the outgoing leg is only known once all the incoming ones are read,
        # since a vertex reuses the smallest incoming leg number for it
        for pos in range(len(ordered)):
            if pos == out_position:
                continue
            number = ordered[pos].get('number')
            side = live.pop(number, None)
            if side is None:
                side = frozenset([number])
            sides[pos] = side
            below = below | side
        if out_position is not None:
            # the outgoing leg carries everything that is not below it
            sides[out_position] = externals - below
            live[ordered[out_position].get('number')] = below

        labels = dict(enumerate(normalise(side) for side in sides))
        if vertex.get('id') in unrollable:
            pairing = unrollable[vertex.get('id')][1][color_chain[i]]
            labels[_SUMMED] = normalise(sides[pairing[0][0]] |
                                        sides[pairing[0][1]])

        if not inter.get('color'):
            continue
        col_str = inter.get('color')[color_chain[i]]
        coeff *= col_str.coeff
        imaginary ^= col_str.is_imaginary
        for obj in col_str:
            try:
                indices = [labels[j if j >= 0 else _SUMMED] for j in obj]
            except KeyError:
                return None
            if isinstance(obj, color_algebra.f):
                # totally antisymmetric: sort and keep track of the parity
                order = sorted(range(len(indices)),
                               key=lambda k: sorted(indices[k]))
                coeff *= _permutation_sign(order)
                indices = [indices[k] for k in order]
            factors.append((obj.__class__.__name__,
                            tuple(tuple(sorted(index)) for index in indices)))

    return (tuple(sorted(factors)), imaginary), coeff


def _permutation_sign(order):
    """Signature of a permutation given as a list of positions."""

    sign = 1
    seen = [False] * len(order)
    for start in range(len(order)):
        if seen[start]:
            continue
        length = 0
        pos = start
        while not seen[pos]:
            seen[pos] = True
            pos = order[pos]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign

#===============================================================================
# Amplitude
#===============================================================================
class Amplitude(base_objects.PhysicsObject):
    """Amplitude: process + list of diagrams (ordered)
    Initialize with a process, then call generate_diagrams() to
    generate the diagrams for the amplitude
    """

    # Interaction ids of the cubic vertices which the seed rule forbids to
    # share a line, see generate_diagrams. Empty -- so the rule is inactive --
    # unless madgraph.merge_quartic_vertices is set.
    seed_forbidden_cubic_ids = frozenset()
    # Links recorded while the seed was expanded, see expand_seed_diagrams
    quartic_unroll_tags = {}

    def default_setup(self):
        """Default values for all properties"""

        self['process'] = base_objects.Process()
        self['diagrams'] = None
        # has_mirror_process is True if the same process but with the
        # two incoming particles interchanged has been generated
        self['has_mirror_process'] = False

    def __init__(self, argument=None, diagram_filter=False):
        """Allow initialization with Process; the diagrams are then generated,
        selected by diagram_filter (see apply_user_filter) if it is set"""
        if isinstance(argument, base_objects.Process):
            super(Amplitude, self).__init__()
            self.set('process', argument)
            self.generate_diagrams(diagram_filter=diagram_filter)
        elif argument != None:
            # call the mother routine
            super(Amplitude, self).__init__(argument)
        else:
            # call the mother routine
            super(Amplitude, self).__init__()

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'process':
            if not isinstance(value, base_objects.Process):
                raise self.PhysicsObjectError("%s is not a valid Process object" % str(value))
        if name == 'diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError("%s is not a valid DiagramList object" % str(value))
        if name == 'has_mirror_process':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError("%s is not a valid boolean" % str(value))
        return True

    def get(self, name):
        """Get the value of the property name."""

        if name == 'diagrams' and self[name] == None:
            # Have not yet generated diagrams for this process
            if self['process']:
                self.generate_diagrams()

        return super(Amplitude, self).get(name)
#        return Amplitude.__bases__[0].get(self, name)  #return the mother routine


    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['process', 'diagrams', 'has_mirror_process']

    def get_number_of_diagrams(self):
        """Returns number of diagrams for this amplitude"""
        return len(self.get('diagrams'))

    def get_amplitudes(self):
        """Return an AmplitudeList with just this amplitude.
        Needed for DecayChainAmplitude."""

        return AmplitudeList([self])

    def nice_string(self, indent=0):
        """Returns a nicely formatted string of the amplitude content."""
        return self.get('process').nice_string(indent) + "\n" + \
               self.get('diagrams').nice_string(indent)

    def nice_string_processes(self, indent=0):
        """Returns a nicely formatted string of the amplitude process."""
        return self.get('process').nice_string(indent)

    def get_ninitial(self):
        """Returns the number of initial state particles in the process."""
        return self.get('process').get_ninitial()

    def has_loop_process(self):
        """ Returns whether this amplitude has a loop process."""
        
        return self.get('process').get('perturbation_couplings')

    def generate_diagrams(self, returndiag=False, diagram_filter=False):
        """Generate diagrams. Algorithm:

        1. Define interaction dictionaries:
          * 2->0 (identity), 3->0, 4->0, ... , maxlegs->0
          * 2 -> 1, 3 -> 1, ..., maxlegs-1 -> 1 

        2. Set flag from_group=true for all external particles.
           Flip particle/anti particle for incoming particles.

        3. If there is a dictionary n->0 with n=number of external
           particles, create if possible the combination [(1,2,3,4,...)] 
           with *at least two* from_group==true. This will give a
           finished (set of) diagram(s) (done by reduce_leglist)

        4. Create all allowed groupings of particles with at least one
           from_group==true (according to dictionaries n->1):
           [(1,2),3,4...],[1,(2,3),4,...],...,
                          [(1,2),(3,4),...],...,[(1,2,3),4,...],... 
           (done by combine_legs)

        5. Replace each group with a (list of) new particle(s) with number 
           n = min(group numbers). Set from_group true for these
           particles and false for all other particles. Store vertex info.
           (done by merge_comb_legs)

        6. Stop algorithm when at most 2 particles remain.
           Return all diagrams (lists of vertices).

        7. Repeat from 3 (recursion done by reduce_leglist)

        8. Replace final p=p vertex
        
        Be aware that the resulting vertices have all particles outgoing,
        so need to flip for incoming particles when used.

        SPECIAL CASE: For A>BC... processes which are legs in decay
        chains, we need to ensure that BC... combine first, giving A=A
        as a final vertex. This case is defined by the Process
        property is_decay_chain = True.
        This function can also be called by the generate_diagram function
        of LoopAmplitudes, in which case the generated diagrams here must not
        be directly assigned to the 'diagrams' attributed but returned as a
        DiagramList by the function. This is controlled by the argument
        returndiag.
        """

        process = self.get('process')
        model = process.get('model')
        legs = process.get('legs')
        # Make sure orders is the minimum of orders and overall_orders
        for key in process.get('overall_orders').keys():
            try:
                process.get('orders')[key] = \
                                 min(process.get('orders')[key],
                                     process.get('overall_orders')[key])
            except KeyError:
                process.get('orders')[key] = process.get('overall_orders')[key]
                
        assert model.get('particles'), \
           "particles are missing in model: %s" %  model.get('particles')

        assert model.get('interactions'), \
               "interactions are missing in model"

        res = base_objects.DiagramList()
        # First check that the number of fermions is even
        if len([leg for leg in legs if model.get('particle_dict')[\
                        leg.get('id')].is_fermion()]) % 2 == 1:
            if not returndiag:
                self['diagrams'] = res
                raise InvalidCmd('The number of fermion is odd')
            else:
                return False, res

        # Then check same number of incoming and outgoing fermions (if
        # no Majorana particles in model)
        if not model.get('got_majoranas') and \
           len([leg for leg in legs if leg.is_incoming_fermion(model)]) != \
           len([leg for leg in legs if leg.is_outgoing_fermion(model)]):
            if not returndiag:
                self['diagrams'] = res
                raise InvalidCmd('The numbers of incoming/outgoing fermions are different')
            else:
                return False, res

        # Finally check that charge (conserve by all interactions) of the process
        #is globally conserve for this process.
        for charge in model.get('conserved_charge'):
            totals = set([0.0])
            for leg in legs:
                part = model.get('particle_dict')[leg.get('id')]
                try:
                    value = part.get(charge)
                except (AttributeError, base_objects.PhysicsObject.PhysicsObjectError):
                    try:
                        value = getattr(part, charge)
                    except AttributeError:
                        value = 0

                values = value if isinstance(value, tuple) else (value,)
                signed_values = []
                for val in values:
                    if (leg.get('id') != part['pdg_code']) != leg['state']:
                        signed_values.append(-float(val))
                    else:
                        signed_values.append(float(val))
                totals = set(round(prev + val, 12) for prev in totals for val in signed_values)

            if not any(abs(total) <= 1e-10 for total in totals):
                if not returndiag:
                    self['diagrams'] = res
                    raise InvalidCmd('No %s conservation for this process ' % charge)
                    return res
                else:
                    raise InvalidCmd('No %s conservation for this process ' % charge)
                    return res, res

        if not returndiag:
            logger.info("Trying %s " % process.nice_string().replace('Process', 'process'))

        # Give numbers to legs in process
        for i in range(0, len(process.get('legs'))):
            # Make sure legs are unique
            leg = copy.copy(process.get('legs')[i])
            process.get('legs')[i] = leg
            if leg.get('number') == 0:
                leg.set('number', i + 1)

            
        # Copy leglist from process, so we can flip leg identities
        # without affecting the original process
        leglist = self.copy_leglist(process.get('legs'))

        for leg in leglist:
            # For the first step, ensure the tag from_group 
            # is true for all legs
            leg.set('from_group', True)

            # Need to flip part-antipart for incoming particles, 
            # so they are all outgoing
            if leg.get('state') == False:
                part = model.get('particle_dict')[leg.get('id')]
                leg.set('id', part.get_anti_pdg_code())

        # Calculate the maximal multiplicity of n-1>1 configurations
        # to restrict possible leg combinations
        max_multi_to1 = max([len(key) for key in \
                             model.get('ref_dict_to1').keys()])

        # Seed rule: when the quartic vertices are to be put back afterwards
        # by unrolling (see unroll_quartic_vertices), generate only the
        # diagrams unrolling cannot produce. Unrolling a quartic vertex always
        # yields two cubic vertices sharing the line which replaced it, so a
        # diagram can be reconstructed exactly when two of its cubic vertices
        # share a line -- and the seed is what is left over.
        # Left off for a decay chain, whose identity vertex is kept rather
        # than glued in, and for loop amplitudes, whose diagram set is not the
        # one the unrolling reasons about. 'auto' also leaves it off below
        # merge_quartic_min_legs, where it is measured to buy nothing.
        self.seed_forbidden_cubic_ids = frozenset()
        if madgraph.merge_quartic_vertices and not self.has_loop_process() \
           and not process.get('is_decay_chain') \
           and (madgraph.merge_quartic_vertices != 'auto' or
                len(process.get('legs')) >= madgraph.merge_quartic_min_legs):
            self.seed_forbidden_cubic_ids = get_unrollable_cubic_ids(model)


        # Reduce the leg list and return the corresponding
        # list of vertices

        # For decay processes, generate starting from final-state
        # combined only as the last particle. This allows to use these
        # in decay chains later on.
        is_decay_proc = process.get_ninitial() == 1
        if is_decay_proc:
            part = model.get('particle_dict')[leglist[0].get('id')]
            # For decay chain legs, we want everything to combine to
            # the initial leg. This is done by only allowing the
            # initial leg to combine as a final identity.
            ref_dict_to0 = {(part.get_pdg_code(),part.get_anti_pdg_code()):[0],
                            (part.get_anti_pdg_code(),part.get_pdg_code()):[0]}
            # Need to set initial leg from_group to None, to make sure
            # it can only be combined at the end.
            leglist[0].set('from_group', None)
            reduced_leglist = self.reduce_leglist(leglist,
                                                  max_multi_to1,
                                                  ref_dict_to0,
                                                  is_decay_proc,
                                                  process.get('orders'))
        else:
            reduced_leglist = self.reduce_leglist(leglist,
                                                  max_multi_to1,
                                                  model.get('ref_dict_to0'),
                                                  is_decay_proc,
                                                  process.get('orders'))
        
        #In LoopAmplitude the function below is overloaded such that it
        #converts back all DGLoopLegs to Legs. In the default tree-level
        #diagram generation, this does nothing.
        self.convert_dgleg_to_leg(reduced_leglist)
         
        if reduced_leglist:
            for vertex_list in reduced_leglist:
                res.append(self.create_diagram(base_objects.VertexList(vertex_list)))

        # Put back the diagrams the seed rule left out
        if self.seed_forbidden_cubic_ids:
            res = self.expand_seed_diagrams(res)

        # Record whether or not we failed generation before required
        # s-channel propagators are taken into account
        failed_crossing = not res

        # Required s-channels is a list of id-lists. Select the
        # diagrams where all required s-channel propagators in any of
        # the lists are present (i.e., the different lists correspond
        # to "or", while the elements of the list correspond to
        # "and").
        if process.get('required_s_channels') and \
               process.get('required_s_channels')[0]:
            # We shouldn't look at the last vertex in each diagram,
            # since that is the n->0 vertex
            lastvx = -1
            # For decay chain processes, there is an "artificial"
            # extra vertex corresponding to particle 1=1, so we need
            # to exclude the two last vertices.
            if is_decay_proc: lastvx = -2
            ninitial = len([leg for leg in process.get('legs') if leg.get('state') == False])
            # Check required s-channels for each list in required_s_channels
            old_res = res
            res = base_objects.DiagramList()
            for id_list in process.get('required_s_channels'):
                res_diags = [diagram for diagram in old_res if all([req_s_channel in \
                               [vertex.get_s_channel_id(\
                               process.get('model'), ninitial) \
                               for vertex in diagram.get('vertices')[:lastvx]] \
                               for req_s_channel in \
                               id_list])]
                # Add diagrams only if not already in res
                res.extend([diag for diag in res_diags if diag not in res])

        # Remove all diagrams with a "double" forbidden s-channel propagator
        # is present.
        # Note that we shouldn't look at the last vertex in each
        # diagram, since that is the n->0 vertex
        if process.get('forbidden_s_channels'):
            ninitial = len([leg for leg in process.get('legs') if leg.get('state') == False])
            if ninitial == 2:
                res = base_objects.DiagramList(\
                [diagram for diagram in res if not any([vertex.get_s_channel_id(\
                           process.get('model'), ninitial) \
                                in process.get('forbidden_s_channels')
                                for vertex in diagram.get('vertices')[:-1]])])
            else:
                # split since we need to avoid that the initial particle is forbidden 
                # as well. 
                newres= []
                for diagram in res:
                    leg1 = 1
                    #check the latest vertex to see if the leg 1 is inside if it 
                    #is we need to inverse the look-up and allow the first s-channel
                    # of the associate particles.
                    vertex =  diagram.get('vertices')[-1]
                    if any([l['number'] ==1 for l in vertex.get('legs')]):
                        leg1 = [l['number'] for l in vertex.get('legs') if l['number'] !=1][0]
                    to_loop = list(range(len(diagram.get('vertices'))-1))
                    if leg1 >1:   
                        to_loop.reverse()
                    for i in to_loop:
                        vertex = diagram.get('vertices')[i]
                        if leg1:
                            if any([l['number'] ==leg1 for l in vertex.get('legs')]):
                                leg1 = 0 
                                continue
                        if vertex.get_s_channel_id(process.get('model'), ninitial)\
                                         in process.get('forbidden_s_channels'):
                            break
                    else:
                        newres.append(diagram)
                res = base_objects.DiagramList(newres)
                

        # Mark forbidden (onshell) s-channel propagators, to forbid onshell
        # generation.
        if process.get('forbidden_onsh_s_channels'):
            ninitial = len([leg for leg in process.get('legs') if leg.get('state') == False])
            
            verts = base_objects.VertexList(sum([[vertex for vertex \
                                                  in diagram.get('vertices')[:-1]
                                           if vertex.get_s_channel_id(\
                                               process.get('model'), ninitial) \
                                           in process.get('forbidden_onsh_s_channels')] \
                                               for diagram in res], []))
            for vert in verts:
                # Use onshell = False to indicate that this s-channel is forbidden
                newleg = copy.copy(vert.get('legs').pop(-1))
                newleg.set('onshell', False)
                vert.get('legs').append(newleg)

        # Set actual coupling orders for each diagram
        for diagram in res:
            diagram.calculate_orders(model)
            
        # Filter the diagrams according to the squared coupling order
        # constraints and possible the negative one. Remember that OrderName=-n
        # means that the user wants to include everything up to the N^(n+1)LO
        # contribution in that order and at most one order can be restricted
        # in this way. We shall do this only if the diagrams are not asked to
        # be returned, as it is the case for NLO because it this case the
        # interference are not necessarily among the diagrams generated here only.
        # 'left [treextree] right': the squared-order constraints are about
        # the products with the right-hand diagrams, added at the end (see
        # add_interference_diagrams); only the amplitude-level constraints of
        # this (left-hand) process apply here
        treextree = not returndiag and \
                         process.get_interference_mode() == 'treextree'
        if treextree:
            for name, (value, operator) in \
                                  process.get('constrained_orders').items():
                res.filter_constrained_orders(name, value, operator)
        elif not returndiag and len(res)>0:
            res = self.apply_squared_order_constraints(res)

        if diagram_filter:
            res = self.apply_user_filter(res, diagram_filter)

        # 4 gluon specials demo (dedicated aloha not implemented yet)
        #if True:
        #    misc.sprint('apply 4 gluon specials')
        #    #res = self.apply_4gluon_specials(res)

        # Replace final id=0 vertex if necessary
        if not process.get('is_decay_chain'):
            for diagram in res:
                # "glue together" last and next-to-last vertex, by replacing
                # the (incoming) last leg of the next-to-last vertex with the
                # (outgoing) leg in the last vertex
                vertices = glued_vertices(diagram)
                if vertices is not None:
                    diagram.set('vertices', vertices)

        if res and not returndiag:
            logger.info("Process has %d diagrams" % len(res))

        # Trim down number of legs and vertices used to save memory
        self.trim_diagrams(diaglist=res)

        # Sort process legs according to leg number
        pertur = 'QCD'
        if self.get('process')['perturbation_couplings']:
            pertur = sorted(self.get('process')['perturbation_couplings'])[0]
        self.get('process').get('legs').sort(pert=pertur)

        if treextree:
            res = self.add_interference_diagrams(res, diagram_filter)

        # Set diagrams to res if not asked to be returned
        if not returndiag:
           self['diagrams'] = res
           return not failed_crossing
        else:
           return not failed_crossing, res

    def add_interference_diagrams(self, left_diagrams, diagram_filter=False):
        """For an interference process 'left [treextree] right', add to the
        diagrams of this (left-hand) process those of the right-hand process
        with the same external legs, each tagged with one unit of the hidden
        order INTERFERENCE_ORDER, and apply the squared-order constraints of
        the process to the whole list: with INTERF^2==1 (see
        prepare_interference_process) only the products of a left-hand and a
        right-hand diagram, 2 Re(A_left A_right^*), contribute. All diagrams of
        both sides are kept as integration channels. A user diagram filter
        ('--diagram_filter') applies to both sides. Returns the new diagram
        list, empty if either side has no diagram."""

        process = self.get('process')
        register_interference_order(process.get('model'))
        right_process = specialise_interference_process(process)

        right_diagrams = base_objects.DiagramList()
        if left_diagrams:
            right_amplitude = Amplitude()
            right_amplitude.set('process', right_process)
            try:
                right_amplitude.generate_diagrams(diagram_filter=diagram_filter)
                right_diagrams = right_amplitude.get('diagrams')
            except InvalidCmd:
                pass
        if not left_diagrams or not right_diagrams:
            logger.info("No diagram for %s: no interference with %s." % (
                (right_process if left_diagrams else process).nice_string(
                                                               prefix=False),
                (process if left_diagrams else right_process).nice_string(
                                                               prefix=False)))
            return base_objects.DiagramList()

        for diagram in right_diagrams:
            orders = dict(diagram.get('orders'))
            orders[INTERFERENCE_ORDER] = 1
            diagram.set('orders', orders)

        res = base_objects.DiagramList(list(left_diagrams) + list(right_diagrams))
        while True:
            new_res = res.apply_positive_sq_orders(res,
                                          process.get('squared_orders'),
                                          process.get('sqorders_types'))
            if len(new_res) == len(res):
                break
            res = new_res
        logger.info("Interference of %d with %d diagrams (%d contributing)" %
                    (len(left_diagrams), len(right_diagrams), len(res)))
        return res

    def apply_squared_order_constraints(self, diag_list):
        """Applies the user specified squared order constraints on the diagram
        list in argument."""

        res = copy.copy(diag_list)                  
        
        # Apply the filtering  on constrained amplitude (== and >)
        # No need to iterate on this one
        for name, (value, operator) in self['process'].get('constrained_orders').items():
            res.filter_constrained_orders(name, value, operator)
            
        # Iterate the filtering since the applying the constraint on one
        # type of coupling order can impact what the filtering on a previous
        # one (relevant for the '==' type of constraint).
        while True: 
            new_res = res.apply_positive_sq_orders(res, 
                                          self['process'].get('squared_orders'), 
                                              self['process']['sqorders_types'])
            # Exit condition
            if len(res)==len(new_res):
                break
            elif (len(new_res)>len(res)):
                raise MadGraph5Error(
                 'Inconsistency in function apply_squared_order_constraints().')
            # Actualizing the list of diagram for the next iteration
            res = new_res
            


        # Now treat the negative squared order constraint (at most one)
        neg_orders = [(order, value) for order, value in \
                       self['process'].get('squared_orders').items() if value<0]
        if len(neg_orders)==1:
            neg_order, neg_value = neg_orders[0]
            # Now check any negative order constraint
            res, target_order = res.apply_negative_sq_order(res, neg_order,\
                  neg_value, self['process']['sqorders_types'][neg_order])
            # Substitute the negative value to this positive one so that
            # the resulting computed constraints appears in the print out
            # and at the output stage we no longer have to deal with 
            # negative valued target orders
            self['process']['squared_orders'][neg_order]=target_order
        elif len(neg_orders)>1:
            raise InvalidCmd('At most one negative squared order constraint'+\
                                   ' can be specified, not %s.'%str(neg_orders))

        return res

    def apply_user_filter(self, diag_list, diagram_filter=True):
        """Applies the user diagram filter on the diagram list in argument.
        diagram_filter is either a function remove_diag(diag, model) provided
        by a plugin (--diagram_filter=PLUGINNAME) or True to use the one of
        PLUGIN/user_filter.py."""

        if callable(diagram_filter):
            # function provided by a plugin (--diagram_filter=PLUGINNAME)
            remove_diag = diagram_filter
        elif diagram_filter is True:
            remove_diag = misc.plugin_import('user_filter', 
                                             'user filter required to be defined in PLUGIN/user_filter.py with the function remove_diag(ONEDIAG) which returns True if the diagram has to be removed',
                                             fcts=['remove_diag'])
        else:
            raise MadGraph5Error('Invalid diagram filter %r: expected True or '
                                 'a function remove_diag(diag, model).' % (diagram_filter,))

        res = diag_list.__class__()                
        nb_removed = 0
        model = self['process']['model'] 
        for diag in diag_list:
            if remove_diag(diag, model):
                nb_removed +=1
            else:
                res.append(diag)

        if nb_removed:
            logger.warning('Diagram filter is ON and removed %s diagrams for this subprocess.' % nb_removed)
            
        return res

    def expand_seed_diagrams(self, seed):
        """Put back the diagrams the seed rule left out of the generation.

        A seed diagram stands for itself and for every diagram obtained by
        replacing any subset of its quartic vertices by the pair of cubic
        vertices one of their colour structures factorises into. Several seeds
        reach the same diagram, hence the dedup, and the result is the full
        diagram set again -- same diagrams, same count, so nothing
        user-visible moves.

        What the detour buys is the decomposition: an unrolled diagram is the
        seed with one vertex taken apart, so it is rooted exactly like the
        diagram it has to be summed with and shares every current except the
        one being summed.

        Also records the link between a quartic contribution and the diagram
        it unrolls to, see get_quartic_unroll_links.
        """

        model = self.get('process').get('model')
        unrollable = get_unrollable_quartic_vertices(model)
        ninitial = self.get_ninitial()

        def canonical_tag(diagram):
            return str(UnrollDiagramTag(diagram, model, ninitial))

        res = base_objects.DiagramList()
        seen = set()
        tag_of = []
        last_seen = {}
        clock = [0]

        def touch(tag):
            # A diagram is placed at its *last* discovery, so that it lands
            # after every seed which can reach it -- and so after every
            # quartic current which can be summed into it. Seeing it again
            # only moves it later, it is never generated twice.
            clock[0] += 1
            last_seen[tag] = clock[0]

        self.quartic_unroll_tags = {}
        todo = []
        for diagram in seed:
            # The identity vertex is glued in right away rather than at the
            # end of generate_diagrams. Until it is, the same diagram has
            # several spellings and comparing two of them is meaningless, and
            # a quartic vertex sitting just before it is not yet the last one
            # -- which is what decides how its colour structures are indexed.
            vertices = glued_vertices(diagram)
            if vertices is not None:
                diagram = self.create_diagram(vertices)
            tag = canonical_tag(diagram)
            seen.add(tag)
            touch(tag)
            res.append(diagram)
            tag_of.append(tag)
            todo.append((diagram, tag))

        # Unrolling is confluent, so taking the diagrams it produces through
        # the same treatment adds nothing to the set -- but it does give the
        # link for the quartic vertices they have left.
        cursor = 0
        while cursor < len(todo):
            diagram, own_tag = todo[cursor]
            cursor += 1
            vertices = diagram.get('vertices')
            positions = [i for i, vertex in enumerate(vertices)
                         if vertex.get('id') in unrollable]
            if not positions:
                continue
            # None leaves the vertex alone, the other entries are the colour
            # structures carrying a coupling, as kept by ColorBasis.colorize
            allowed = [[None] + sorted(misc.make_unique(
                [key[0] for key in model.get_interaction(
                    vertices[p].get('id')).get('couplings')]))
                       for p in positions]

            for keys in itertools.product(*allowed):
                choice = dict((position, key) for position, key
                              in zip(positions, keys) if key is not None)
                if not choice:
                    continue
                unrolled = self.unrolled_diagram(diagram, choice, unrollable)
                tag = canonical_tag(unrolled)
                touch(tag)
                if tag not in seen:
                    seen.add(tag)
                    res.append(unrolled)
                    tag_of.append(tag)
                    todo.append((unrolled, tag))
                if len(choice) == len(positions):
                    chain = tuple(choice.get(i, 0)
                                  for i in range(len(vertices)))
                    self.quartic_unroll_tags[(own_tag, chain)] = tag

        order = sorted(range(len(res)), key=lambda i: (last_seen[tag_of[i]], i))
        if madgraph.merge_quartic_vertices == 'slots':
            # Reversed, each diagram lands at its *first* discovery instead.
            # That keeps far fewer currents alive -- 199 slots against 259 at
            # seven gluons -- at the price of the current sums, since a target
            # then comes before the quartic currents which feed it. The trade
            # a gpu wants, where the wavefunction store is per thread.
            order.reverse()
        return base_objects.DiagramList([res[i] for i in order])

    def get_quartic_unroll_links(self, diaglist=None):
        """Return {(diagram index, colour chain): target index} recorded while
        the seed was expanded, resolved against the diagram list as it stands.

        This is the same map as the one unroll_quartic_vertices reconstructs
        from the colour algebra, but obtained for free: the two diagrams are
        one and the same seed unrolled differently. Empty when the diagrams
        were not generated from a seed."""

        if not self.quartic_unroll_tags:
            return {}

        model = self.get('process').get('model')
        if diaglist is None:
            diaglist = self.get('diagrams')
        ninitial = self.get_ninitial()
        index = dict((str(UnrollDiagramTag(diagram, model, ninitial)), i)
                     for i, diagram in enumerate(diaglist))

        res = {}
        for (seed_tag, chain), tag in self.quartic_unroll_tags.items():
            if seed_tag in index and tag in index:
                res[(index[seed_tag], chain)] = index[tag]
        return res

    def unroll_quartic_vertices(self, diaglist=None):
        """Link every quartic vertex contribution to the cubic diagram it
        merges with.

        Each colour structure of an unrollable quartic vertex (see
        get_unrollable_quartic_vertices) splits its four legs into two pairs,
        which is exactly what two cubic vertices joined by an internal line
        do. Replacing every quartic vertex of a diagram by that pair of cubic
        vertices therefore turns it into a diagram already present in the
        amplitude, and the two carry the same colour factor up to a rational
        coefficient. Their amplitudes can hence be summed before the colour
        algebra is applied.

        Returns {(diagram_index, colour_chain): (target_index, coeff)} where
        colour_chain follows the convention of color_amp.ColorBasis.colorize
        (one colour structure index per vertex) and the amplitude of the
        source is coeff times a contribution to the target diagram. Diagrams
        without any quartic vertex are left out.

        To line the result up with HelasAmplitude, call this on the
        *reconstructed* amplitude, HelasMatrixElement.get_base_amplitude(),
        and not on the amplitude the matrix element was built from. That is
        the one helas_objects itself colorizes, and the vertex order of the
        reconstructed diagrams -- hence the position of a colour structure
        index inside the chain -- can differ from the generated ones. The two
        happen to agree up to six gluons and start to differ at seven.
        """

        model = self.get('process').get('model')
        if diaglist is None:
            diaglist = self.get('diagrams')

        unrollable = get_unrollable_quartic_vertices(model)
        if not unrollable:
            return {}

        quartic_positions = []
        for diag in diaglist:
            quartic_positions.append([i for i, vx in
                                      enumerate(diag.get('vertices'))
                                      if vx.get('id') in unrollable])
        if not any(quartic_positions):
            return {}

        ninitial = self.get_ninitial()
        # Diagrams free of quartic vertices are the possible merge targets
        target_index = {}
        for i, diag in enumerate(diaglist):
            if quartic_positions[i]:
                continue
            target_index[str(UnrollDiagramTag(diag, model, ninitial))] = i

        res = {}
        for i, diag in enumerate(diaglist):
            positions = quartic_positions[i]
            if not positions:
                continue
            # only the colour structures which carry a coupling contribute,
            # matching what color_amp.ColorBasis.colorize keeps
            allowed = [sorted(misc.make_unique(
                [key[0] for key in model.get_interaction(
                    diag.get('vertices')[p].get('id')).get('couplings')]))
                       for p in positions]
            for keys in itertools.product(*allowed):
                choice = dict(zip(positions, keys))
                unrolled = self.unrolled_diagram(diag, choice, unrollable)
                try:
                    target = target_index[
                        str(UnrollDiagramTag(unrolled, model, ninitial))]
                except KeyError:
                    # No cubic partner: the merge is not available, which can
                    # happen when the partner was removed by a diagram filter
                    # or by a forbidden s-channel.
                    continue
                chain = tuple(choice.get(p, 0)
                              for p in range(len(diag.get('vertices'))))
                source_sig = diagram_colour_signature(diag, model, chain,
                                                      unrollable)
                target_chain = (0,) * len(diaglist[target].get('vertices'))
                target_sig = diagram_colour_signature(diaglist[target], model,
                                                      target_chain, unrollable)
                if source_sig is None or target_sig is None or \
                   source_sig[0] != target_sig[0]:
                    continue
                res[(i, chain)] = (target, source_sig[1] / target_sig[1])

        return res

    def unrolled_diagram(self, diagram, choice, unrollable):
        """Return a copy of diagram where the quartic vertices listed in
        choice (position -> colour structure index) have been replaced by the
        two cubic vertices that colour structure factorises into."""

        model = self.get('process').get('model')
        vertices = base_objects.VertexList()
        for i, vertex in enumerate(diagram.get('vertices')):
            if i not in choice:
                vertices.append(vertex)
                continue
            cubic_id, pairings = unrollable[vertex.get('id')]
            vertices.extend(split_quartic_vertex(
                vertex, i == len(diagram.get('vertices')) - 1,
                pairings[choice[i]], cubic_id, model))

        return base_objects.Diagram({'vertices': vertices})

    def apply_4gluon_specials(self, diag_list):

        res = diag_list.__class__()
        for diag in diag_list:
            keep = True
            for vertex in diag.get('vertices'):
                if [21,21,21] == [abs(leg.get('id')) for leg in vertex.get('legs')]:
                    misc.sprint([hasattr(leg, 'g3') for leg in vertex.get('legs')])
                    if any([hasattr(leg, 'g3') for leg in vertex.get('legs')]):
                        keep = False
                        break
                    else:
                        vertex.get('legs')[-1].g3 = True
                        misc.sprint([hasattr(leg, 'g3') for leg in vertex.get('legs')])
                if [21,21] == [abs(leg.get('id')) for leg in vertex.get('legs')]:
                    if all([hasattr(leg, 'g3') for leg in vertex.get('legs')]):
                        keep = False
                        break
            if keep:
                res.append(diag)
        return res


    def create_diagram(self, vertexlist):
        """ Return a Diagram created from the vertex list. This function can be
            overloaded by daughter classes."""
        return base_objects.Diagram({'vertices':vertexlist})

    def convert_dgleg_to_leg(self, vertexdoublelist):
        """ In LoopAmplitude, it converts back all DGLoopLegs into Legs.
            In Amplitude, there is nothing to do. """

        return True

    def copy_leglist(self, legs):
        """ Simply returns a copy of the leg list. This function is
            overloaded in LoopAmplitude so that a DGLoopLeg list is returned.
            The DGLoopLeg has some additional parameters only useful during
            loop diagram generation"""

        return base_objects.LegList(\
                [ copy.copy(leg) for leg in legs ])

    def reduce_leglist(self, curr_leglist, max_multi_to1, ref_dict_to0,
                       is_decay_proc = False, coupling_orders = None,
                       cubic_legs = frozenset()):
        """Recursive function to reduce N LegList to N-1
           For algorithm, see doc for generate_diagrams.

           cubic_legs holds the numbers of the legs of curr_leglist which were
           produced by a vertex the seed rule keeps apart, and is only ever
           non-empty when that rule is active.
        """

        # Result variable which is a list of lists of vertices
        # to be added
        res = []

        # Stop condition. If LegList is None, that means that this
        # diagram must be discarded
        if curr_leglist is None:
            return None

        # Extract ref dict information
        model = self.get('process').get('model')
        ref_dict_to1 = self.get('process').get('model').get('ref_dict_to1')


        # If all legs can be combined in one single vertex, add this
        # vertex to res and continue.
        # Special treatment for decay chain legs

        if curr_leglist.can_combine_to_0(ref_dict_to0, is_decay_proc):
            # Extract the interaction id associated to the vertex 
            
            vertex_ids = self.get_combined_vertices(curr_leglist,
                       copy.copy(ref_dict_to0[tuple(sorted([leg.get('id') for \
                                                       leg in curr_leglist]))]))

            final_vertices = [base_objects.Vertex({'legs':curr_leglist,
                                                   'id':vertex_id}) for \
                              vertex_id in vertex_ids]
            # Check for coupling orders. If orders < 0, skip vertex
            for final_vertex in final_vertices:
                if self.joins_two_cubics(final_vertex, cubic_legs):
                    continue
                if self.reduce_orders(coupling_orders, model,
                                      [final_vertex.get('id')]) != False:
                    res.append([final_vertex])
        # Stop condition 2: if the leglist contained exactly two particles,
        # return the result, if any, and stop.
        if len(curr_leglist) == 2:
            if res:
                return res
            else:
                return None

        # Create a list of all valid combinations of legs
        comb_lists = self.combine_legs(curr_leglist,
                                       ref_dict_to1, max_multi_to1)

        # Create a list of leglists/vertices by merging combinations
        leg_vertex_list = self.merge_comb_legs(comb_lists, ref_dict_to1)

        # Consider all the pairs
        for leg_vertex_tuple in leg_vertex_list:

            # Remove forbidden particles
            if self.get('process').get('forbidden_particles') and \
                any([abs(vertex.get('legs')[-1].get('id')) in \
                self.get('process').get('forbidden_particles') \
                for vertex in leg_vertex_tuple[1]]):
                    continue

            # Check for coupling orders. If couplings < 0, skip recursion.
            new_coupling_orders = self.reduce_orders(coupling_orders,
                                                     model,
                                                     [vertex.get('id') for vertex in \
                                                      leg_vertex_tuple[1]])
            if new_coupling_orders == False:
                # Some coupling order < 0
                continue

            # Seed rule: drop the combinations putting two of the cubic
            # vertices to be kept apart on the same line
            if any(self.shares_line_with_cubic(vertex.get('id'),
                                               vertex.get('legs')[:-1],
                                               cubic_legs)
                   for vertex in leg_vertex_tuple[1]):
                continue

            # This is where recursion happens
            # First, reduce again the leg part
            reduced_diagram = self.reduce_leglist(leg_vertex_tuple[0],
                                                  max_multi_to1,
                                                  ref_dict_to0,
                                                  is_decay_proc,
                                                  new_coupling_orders,
                                                  self.mark_cubic_legs(\
                                                      cubic_legs,
                                                      leg_vertex_tuple[1]))
            # If there is a reduced diagram
            if reduced_diagram:
                vertex_list_list = [list(leg_vertex_tuple[1])]
                vertex_list_list.append(reduced_diagram)
                expanded_list = expand_list_list(vertex_list_list)
                res.extend(expanded_list)

        return res

    def shares_line_with_cubic(self, vertex_id, incoming, cubic_legs):
        """True when vertex_id is one of the cubic vertices the seed rule
        keeps apart and one of the lines coming in was produced by another
        one of them."""

        if not self.seed_forbidden_cubic_ids or \
           vertex_id not in self.seed_forbidden_cubic_ids:
            return False
        return any(leg.get('number') in cubic_legs for leg in incoming)

    def joins_two_cubics(self, vertex, cubic_legs):
        """True when the vertex closing the diagram puts two of the cubic
        vertices the seed rule keeps apart on the same line.

        The closing vertex is either a real n->0 interaction, whose incoming
        lines are simply the legs left over, or the identity vertex, which
        states that its two legs are the two ends of one and the same line --
        and that line joins the two vertices which produced them."""

        if not self.seed_forbidden_cubic_ids:
            return False
        legs = vertex.get('legs')
        if vertex.get('id'):
            return self.shares_line_with_cubic(vertex.get('id'), legs,
                                               cubic_legs)
        return all(leg.get('number') in cubic_legs for leg in legs)

    def mark_cubic_legs(self, cubic_legs, vertices):
        """Update the set of leg numbers standing for a line produced by one
        of the cubic vertices the seed rule keeps apart.

        A number is dropped as soon as the line is consumed, since a vertex
        reuses the smallest number coming in for the leg it produces."""

        if not self.seed_forbidden_cubic_ids:
            return cubic_legs
        consumed = frozenset(leg.get('number') for vertex in vertices
                             for leg in vertex.get('legs')[:-1])
        produced = frozenset(vertex.get('legs')[-1].get('number')
                             for vertex in vertices if vertex.get('id') in \
                             self.seed_forbidden_cubic_ids)
        return (cubic_legs - consumed) | produced

    def reduce_orders(self, coupling_orders, model, vertex_id_list):
        """Return False if the coupling orders for any coupling is <
        0, otherwise return the new coupling orders with the vertex
        orders subtracted. If coupling_orders is not given, return
        None (which counts as success).
        WEIGHTED is a special order, which corresponds to the sum of
        order hierarchies for the couplings.
        We ignore negative constraints as these cannot be taken into
        account on the fly but only after generation."""

        if not coupling_orders:
            return None

        present_couplings = copy.copy(coupling_orders)
        for id in vertex_id_list:
            # Don't check for identity vertex (id = 0)
            if not id:
                continue
            inter = model.get("interaction_dict")[id]
            for coupling in inter.get('orders').keys():
                # Note that we don't consider a missing coupling as a
                # constraint
                if coupling in present_couplings and \
                                                 present_couplings[coupling]>=0:
                    # Reduce the number of couplings that are left
                    present_couplings[coupling] -= \
                             inter.get('orders')[coupling]
                    if present_couplings[coupling] < 0:
                        # We have too many couplings of this type
                        return False
            # Now check for WEIGHTED, i.e. the sum of coupling hierarchy values
            if 'WEIGHTED' in present_couplings and \
                                               present_couplings['WEIGHTED']>=0:
                weight = sum([model.get('order_hierarchy')[c]*n for \
                              (c,n) in inter.get('orders').items()])
                present_couplings['WEIGHTED'] -= weight
                if present_couplings['WEIGHTED'] < 0:
                        # Total coupling weight too large
                        return False
                
        return present_couplings

    def combine_legs(self, list_legs, ref_dict_to1, max_multi_to1):
        """Recursive function. Take a list of legs as an input, with
        the reference dictionary n-1->1, and output a list of list of
        tuples of Legs (allowed combinations) and Legs (rest). Algorithm:

        1. Get all n-combinations from list [123456]: [12],..,[23],..,[123],..

        2. For each combination, say [34]. Check if combination is valid.
           If so:

           a. Append [12[34]56] to result array

           b. Split [123456] at index(first element in combination+1),
              i.e. [12],[456] and subtract combination from second half,
              i.e.: [456]-[34]=[56]. Repeat from 1. with this array

        3. Take result array from call to 1. (here, [[56]]) and append
           (first half in step b - combination) + combination + (result
           from 1.) = [12[34][56]] to result array

        4. After appending results from all n-combinations, return
           resulting array. Example, if [13] and [45] are valid
           combinations:
            [[[13]2456],[[13]2[45]6],[123[45]6]] 
        """

        res = []

        # loop over possible combination lengths (+1 is for range convention!)
        for comb_length in range(2, max_multi_to1 + 1):

            # Check the considered length is not longer than the list length
            if comb_length > len(list_legs):
                return res

            # itertools.combinations returns all possible combinations
            # of comb_length elements from list_legs
            for comb in itertools.combinations(list_legs, comb_length):

                # Check if the combination is valid
                if base_objects.LegList(comb).can_combine_to_1(ref_dict_to1):

                    # Identify the rest, create a list [comb,rest] and
                    # add it to res
                    res_list = copy.copy(list_legs)
                    for leg in comb:
                        res_list.remove(leg)
                    res_list.insert(list_legs.index(comb[0]), comb)
                    res.append(res_list)

                    # Now, deal with cases with more than 1 combination

                    # First, split the list into two, according to the
                    # position of the first element in comb, and remove
                    # all elements form comb
                    res_list1 = list_legs[0:list_legs.index(comb[0])]
                    res_list2 = list_legs[list_legs.index(comb[0]) + 1:]
                    for leg in comb[1:]:
                        res_list2.remove(leg)

                    # Create a list of type [comb,rest1,rest2(combined)]
                    res_list = res_list1
                    res_list.append(comb)
                    # This is where recursion actually happens, 
                    # on the second part
                    for item in self.combine_legs(res_list2,
                                                  ref_dict_to1,
                                                  max_multi_to1):
                        final_res_list = copy.copy(res_list)
                        final_res_list.extend(item)
                        res.append(final_res_list)

        return res


    def merge_comb_legs(self, comb_lists, ref_dict_to1):
        """Takes a list of allowed leg combinations as an input and returns
        a set of lists where combinations have been properly replaced
        (one list per element in the ref_dict, so that all possible intermediate
        particles are included). For each list, give the list of vertices
        corresponding to the executed merging, group the two as a tuple.
        """

        res = []

        for comb_list in comb_lists:

            reduced_list = []
            vertex_list = []

            for entry in comb_list:

                # Act on all leg combinations
                if isinstance(entry, tuple):

                    # Build the leg object which will replace the combination:
                    # 1) leg ids is as given in the ref_dict
                    leg_vert_ids = copy.copy(ref_dict_to1[\
                        tuple(sorted([leg.get('id') for leg in entry]))])
                    # 2) number is the minimum of leg numbers involved in the
                    # combination
                    number = min([leg.get('number') for leg in entry])
                    # 3) state is final, unless there is exactly one initial 
                    # state particle involved in the combination -> t-channel
                    if len([leg for leg in entry if leg.get('state') == False]) == 1:
                        state = False
                    else:
                        state = True
                    # 4) from_group is True, by definition

                    # Create and add the object. This is done by a
                    # separate routine, to allow overloading by
                    # daughter classes
                    new_leg_vert_ids = []
                    if leg_vert_ids:
                        new_leg_vert_ids = self.get_combined_legs(entry,
                                                                  leg_vert_ids,
                                                                  number,
                                                                  state)
                    
                    reduced_list.append([l[0] for l in new_leg_vert_ids])


                    # Create and add the corresponding vertex
                    # Extract vertex ids corresponding to the various legs
                    # in mylegs
                    vlist = base_objects.VertexList()
                    for (myleg, vert_id) in new_leg_vert_ids:
                        # Start with the considered combination...
                        myleglist = base_objects.LegList(list(entry))
                        # ... and complete with legs after reducing
                        myleglist.append(myleg)
                        # ... and consider the correct vertex id
                        vlist.append(base_objects.Vertex(
                                         {'legs':myleglist,
                                          'id':vert_id}))

                    vertex_list.append(vlist)

                # If entry is not a combination, switch the from_group flag
                # and add it
                else:
                    cp_entry = copy.copy(entry)
                    # Need special case for from_group == None; this
                    # is for initial state leg of decay chain process
                    # (see Leg.can_combine_to_0)
                    if cp_entry.get('from_group') != None:
                        cp_entry.set('from_group', False)
                    reduced_list.append(cp_entry)

            # Flatten the obtained leg and vertex lists
            flat_red_lists = expand_list(reduced_list)
            flat_vx_lists = expand_list(vertex_list)

            # Combine the two lists in a list of tuple
            for i in range(0, len(flat_vx_lists)):
                res.append((base_objects.LegList(flat_red_lists[i]), \
                            base_objects.VertexList(flat_vx_lists[i])))

        return res

    def get_combined_legs(self, legs, leg_vert_ids, number, state):
        """Create a set of new legs from the info given. This can be
        overloaded by daughter classes."""

        mylegs = [(base_objects.Leg({'id':leg_id,
                                    'number':number,
                                    'state':state,
                                    'from_group':True}),
                   vert_id)\
                  for leg_id, vert_id in leg_vert_ids]
        

        return mylegs
                          
    def get_combined_vertices(self, legs, vert_ids):
        """Allow for selection of vertex ids. This can be
        overloaded by daughter classes."""

        return vert_ids
                          
    def trim_diagrams(self, decay_ids=[], diaglist=None):
        """Reduce the number of legs and vertices used in memory.
        When called by a diagram generation initiated by LoopAmplitude, 
        this function should not trim the diagrams in the attribute 'diagrams'
        but rather a given list in the 'diaglist' argument."""

        legs = []
        vertices = []

        if diaglist is None:
            diaglist=self.get('diagrams')

        # Flag decaying legs in the core process by onshell = True
        process = self.get('process')
        for leg in process.get('legs'):
            if leg.get('state') and leg.get('id') in decay_ids:
                leg.set('onshell', True)
        
        for diagram in diaglist:
            # Keep track of external legs (leg numbers already used)
            leg_external = set()
            for ivx, vertex in enumerate(diagram.get('vertices')):
                for ileg, leg in enumerate(vertex.get('legs')):
                    # Ensure that only external legs get decay flag
                    if leg.get('state') and leg.get('id') in decay_ids and \
                           leg.get('number') not in leg_external:
                        # Use onshell to indicate decaying legs,
                        # i.e. legs that have decay chains
                        leg = copy.copy(leg)
                        leg.set('onshell', True)
                    try:
                        index = legs.index(leg)
                    except ValueError:
                        vertex.get('legs')[ileg] = leg
                        legs.append(leg)
                    else: # Found a leg
                        vertex.get('legs')[ileg] = legs[index]
                    leg_external.add(leg.get('number'))
                try:
                    index = vertices.index(vertex)
                    diagram.get('vertices')[ivx] = vertices[index]
                except ValueError:
                    vertices.append(vertex)

#===============================================================================
# AmplitudeList
#===============================================================================
class AmplitudeList(base_objects.PhysicsObjectList):
    """List of Amplitude objects
    """

    def has_any_loop_process(self):
        """ Check the content of all processes of the amplitudes in this list to
        see if there is any which defines perturbation couplings. """
        
        for amp in self:
            if amp.has_loop_process():
                return True

    def is_valid_element(self, obj):
        """Test if object obj is a valid Amplitude for the list."""

        return isinstance(obj, Amplitude)
    
#===============================================================================
# DecayChainAmplitude
#===============================================================================
class DecayChainAmplitude(Amplitude):
    """A list of amplitudes + a list of decay chain amplitude lists;
    corresponding to a ProcessDefinition with a list of decay chains
    """

    def default_setup(self):
        """Default values for all properties"""

        self['amplitudes'] = AmplitudeList()
        self['decay_chains'] = DecayChainAmplitudeList()

    def __init__(self, argument = None, collect_mirror_procs = False,
                 ignore_six_quark_processes = False, loop_filter=None, diagram_filter=False):
        """Allow initialization with Process and with ProcessDefinition"""
 
        if isinstance(argument, base_objects.Process):
            super(DecayChainAmplitude, self).__init__()
            from madgraph.loop.loop_diagram_generation import LoopMultiProcess
            if argument['perturbation_couplings']:
                MultiProcessClass=LoopMultiProcess
            else:
                MultiProcessClass=MultiProcess                             
            if isinstance(argument, base_objects.ProcessDefinition):
                self['amplitudes'].extend(\
                  MultiProcessClass.generate_multi_amplitudes(argument,
                                                    collect_mirror_procs,
                                                    ignore_six_quark_processes,
                                                    loop_filter=loop_filter,
                                                    diagram_filter=diagram_filter))
            else:
                self['amplitudes'].append(\
                  MultiProcessClass.get_amplitude_from_proc(argument,
                                                       loop_filter=loop_filter,
                                                       diagram_filter=diagram_filter))
                # Clean decay chains from process, since we haven't
                # combined processes with decay chains yet
                process = copy.copy(self.get('amplitudes')[0].get('process'))
                process.set('decay_chains', base_objects.ProcessList())
                self['amplitudes'][0].set('process', process)

            for process in argument.get('decay_chains'):
                if process.get('perturbation_couplings'):
                    raise MadGraph5Error("Decay processes can not be perturbed")
                process.set('overall_orders', argument.get('overall_orders'))
                if not process.get('is_decay_chain'):
                    process.set('is_decay_chain',True)
                if not process.get_ninitial() == 1:
                    raise InvalidCmd("Decay chain process must have exactly one" + \
                          " incoming particle")
                self['decay_chains'].append(\
                    DecayChainAmplitude(process, collect_mirror_procs,
                                        ignore_six_quark_processes,
                                        diagram_filter=diagram_filter))

            # Flag decaying legs in the core diagrams by onshell = True
            decay_ids = sum([[a.get('process').get('legs')[0].get('id') \
                              for a in dec.get('amplitudes')] for dec in \
                             self['decay_chains']], [])
            decay_ids = set(decay_ids)
            for amp in self['amplitudes']:
                amp.trim_diagrams(decay_ids)                    

            # Check that all decay ids are present in at least some process
            for amp in self['amplitudes']:
                for l in amp.get('process').get('legs'):
                    if l.get('id') in decay_ids:
                        decay_ids.remove(l.get('id'))
            
            if decay_ids:
                model = amp.get('process').get('model')
                names = [model.get_particle(id).get('name') for id in decay_ids]
                
                logger.warning(
                 "$RED Decay without corresponding particle in core process found.\n" + \
                 "Decay information for particle(s) %s is discarded.\n" % ','.join(names) + \
                 "Please check your process definition carefully. \n" + \
                 "This warning usually means that you forgot parentheses in presence of subdecay.\n" + \
                 "Example of correct syntax: p p > t t~, ( t > w+ b, w+ > l+ vl)")

                # Remove unused decays from the process list
                for dc in reversed(self['decay_chains']):
                    for a in reversed(dc.get('amplitudes')):
                        # Remove the amplitudes from this decay chain
                        if a.get('process').get('legs')[0].get('id') in decay_ids:
                            dc.get('amplitudes').remove(a)
                    if not dc.get('amplitudes'):
                        # If no amplitudes left, remove the decay chain
                        self['decay_chains'].remove(dc)
                    
            # Finally, write a fat warning if any decay process has
            # the decaying particle (or its antiparticle) in the final state
            bad_procs = []
            for dc in self['decay_chains']:
                for amp in dc.get('amplitudes'):
                    legs = amp.get('process').get('legs')
                    fs_parts = [abs(l.get('id')) for l in legs if 
                                l.get('state')]
                    is_part = [l.get('id') for l in legs if not 
                               l.get('state')][0]
                    if abs(is_part) in fs_parts:
                        bad_procs.append(amp.get('process'))

            if bad_procs:
                logger.warning(
                    "$RED Decay(s) with particle decaying to itself:\n" + \
                     '\n'.join([p.nice_string() for p in bad_procs]) + \
                 "\nPlease check your process definition carefully. \n")
                    

        elif argument != None:
            # call the mother routine
            super(DecayChainAmplitude, self).__init__(argument)
        else:
            # call the mother routine
            super(DecayChainAmplitude, self).__init__()

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'amplitudes':
            if not isinstance(value, AmplitudeList):
                raise self.PhysicsObjectError("%s is not a valid AmplitudeList" % str(value))
        if name == 'decay_chains':
            if not isinstance(value, DecayChainAmplitudeList):
                raise self.PhysicsObjectError("%s is not a valid DecayChainAmplitudeList object" % \
                        str(value))
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['amplitudes', 'decay_chains']

    # Helper functions

    def get_number_of_diagrams(self):
        """Returns number of diagrams for this amplitude"""
        return sum(len(a.get('diagrams')) for a in self.get('amplitudes')) \
               + sum(d.get_number_of_diagrams() for d in \
                                        self.get('decay_chains'))

    def nice_string(self, indent = 0):
        """Returns a nicely formatted string of the amplitude content."""
        mystr = ""
        for amplitude in self.get('amplitudes'):
            mystr = mystr + amplitude.nice_string(indent) + "\n"

        if self.get('decay_chains'):
            mystr = mystr + " " * indent + "Decays:\n"
        for dec in self.get('decay_chains'):
            mystr = mystr + dec.nice_string(indent + 2) + "\n"

        return  mystr[:-1]

    def nice_string_processes(self, indent = 0):
        """Returns a nicely formatted string of the amplitude processes."""
        mystr = ""
        for amplitude in self.get('amplitudes'):
            mystr = mystr + amplitude.nice_string_processes(indent) + "\n"

        if self.get('decay_chains'):
            mystr = mystr + " " * indent + "Decays:\n"
        for dec in self.get('decay_chains'):
            mystr = mystr + dec.nice_string_processes(indent + 2) + "\n"

        return  mystr[:-1]

    def get_ninitial(self):
        """Returns the number of initial state particles in the process."""
        return self.get('amplitudes')[0].get('process').get_ninitial()

    def get_decay_ids(self):
        """Returns a set of all particle ids for which a decay is defined"""

        decay_ids = []

        # Get all amplitudes for the decay processes
        for amp in sum([dc.get('amplitudes') for dc \
                        in self['decay_chains']], []):
            # For each amplitude, find the initial state leg
            decay_ids.append(amp.get('process').get_initial_ids()[0])
            
        # Return a list with unique ids
        return misc.make_unique(decay_ids)
    
    def has_loop_process(self):
        """ Returns whether this amplitude has a loop process."""
        return self['amplitudes'].has_any_loop_process()
    
    def get_amplitudes(self):
        """Recursive function to extract all amplitudes for this process"""

        amplitudes = AmplitudeList()

        amplitudes.extend(self.get('amplitudes'))
        for decay in self.get('decay_chains'):
            amplitudes.extend(decay.get_amplitudes())

        return amplitudes
            

#===============================================================================
# DecayChainAmplitudeList
#===============================================================================
class DecayChainAmplitudeList(base_objects.PhysicsObjectList):
    """List of DecayChainAmplitude objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid DecayChainAmplitude for the list."""

        return isinstance(obj, DecayChainAmplitude)

    
#===============================================================================
# MultiProcess
#===============================================================================
class MultiProcess(base_objects.PhysicsObject):
    """MultiProcess: list of process definitions
                     list of processes (after cleaning)
                     list of amplitudes (after generation)
    """

    def default_setup(self):
        """Default values for all properties"""

        self['process_definitions'] = base_objects.ProcessDefinitionList()
        # self['amplitudes'] can be an AmplitudeList or a
        # DecayChainAmplitudeList, depending on whether there are
        # decay chains in the process definitions or not.
        self['amplitudes'] = AmplitudeList()
        # Flag for whether to combine IS mirror processes together
        self['collect_mirror_procs'] = False
        # List of quark flavors where we ignore processes with at
        # least 6 quarks (three quark lines)
        self['ignore_six_quark_processes'] = []
        # Allow to use the model parameter numerical value for optimization.
        #This is currently use for 1->N generation(check mass).
        self['use_numerical'] = False
        
    def __init__(self, argument=None, collect_mirror_procs = False,
                 ignore_six_quark_processes = [], optimize=False,
                 loop_filter=None, diagram_filter=None, merge_crossing=False):
        """Allow initialization with ProcessDefinition or
        ProcessDefinitionList
        optimize allows to use param_card information. (useful for 1-.N)"""

        if isinstance(argument, base_objects.ProcessDefinition):
            super(MultiProcess, self).__init__()
            self['process_definitions'].append(argument)
        elif isinstance(argument, base_objects.ProcessDefinitionList):
            super(MultiProcess, self).__init__()
            self['process_definitions'] = argument
        elif argument != None:
            # call the mother routine
            super(MultiProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(MultiProcess, self).__init__()

        self['collect_mirror_procs'] = collect_mirror_procs
        self['ignore_six_quark_processes'] = ignore_six_quark_processes
        self['use_numerical'] = optimize
        self['loop_filter'] = loop_filter
        self['diagram_filter'] = diagram_filter # True (PLUGIN/user_filter.py) or a plugin function
        self['merge_crossing'] = merge_crossing
        
        if isinstance(argument, base_objects.ProcessDefinition) or \
               isinstance(argument, base_objects.ProcessDefinitionList):
            # Generate the diagrams
            self.get('amplitudes')


    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'process_definitions':
            if not isinstance(value, base_objects.ProcessDefinitionList):
                raise self.PhysicsObjectError("%s is not a valid ProcessDefinitionList object" % str(value))

        if name == 'amplitudes':
            if not isinstance(value, AmplitudeList):
                raise self.PhysicsObjectError("%s is not a valid AmplitudeList object" % str(value))

        if name in ['collect_mirror_procs']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError("%s is not a valid boolean" % str(value))

        if name == 'ignore_six_quark_processes':
            if not isinstance(value, list):
                raise self.PhysicsObjectError("%s is not a valid list" % str(value))

        return True

    def get(self, name):
        """Get the value of the property name."""

        if (name == 'amplitudes') and not self[name]:
            for process_def in self.get('process_definitions'):
                if process_def.get('decay_chains'):
                    # This is a decay chain process
                    # Store amplitude(s) as DecayChainAmplitude
                    self['amplitudes'].append(\
                        DecayChainAmplitude(process_def,
                                       self.get('collect_mirror_procs'),
                                       self.get('ignore_six_quark_processes'),
                                       diagram_filter=self['diagram_filter']))
                else:
                    self['amplitudes'].extend(\
                       self.generate_multi_amplitudes(process_def,
                                       self.get('collect_mirror_procs'),
                                       self.get('ignore_six_quark_processes'),
                                       self['use_numerical'],
                                       loop_filter=self['loop_filter'],
                                       diagram_filter=self['diagram_filter'],
                                       merge_crossing=self['merge_crossing']))

        return MultiProcess.__bases__[0].get(self, name) # call the mother routine

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['process_definitions', 'amplitudes']
    
    def get_model(self):
        
        return self['process_definitions'][0]['model']

    @classmethod
    def generate_multi_amplitudes(cls,process_definition,
                                  collect_mirror_procs = False,
                                  ignore_six_quark_processes = [],
                                  use_numerical=False,
                                  loop_filter=None,
                                  diagram_filter=False,
                                  merge_crossing=False):
        """Generate amplitudes in a semi-efficient way.
        Make use of crossing symmetry for processes that fail diagram
        generation, but not for processes that succeed diagram
        generation.  Doing so will risk making it impossible to
        identify processes with identical amplitudes.
        """

        assert isinstance(process_definition, base_objects.ProcessDefinition), \
                                    "%s not valid ProcessDefinition object" % \
                                    repr(process_definition)

        # Set automatic coupling orders if born_sq_orders are not specified
        # otherwise skip
        if not process_definition['born_sq_orders']:
            process_definition.set('orders', MultiProcess.\
                               find_optimal_process_orders(process_definition,
                                                           diagram_filter))
        # Check for maximum orders from the model
        process_definition.check_expansion_orders()

        processes = base_objects.ProcessList()
        amplitudes = AmplitudeList()

        # failed_procs and success_procs are sorted processes that have
        # already failed/succeeded based on crossing symmetry
        failed_procs = []
        success_procs = []
        # Complete processes, for identification of mirror processes
        non_permuted_procs = []
        # permutations keeps the permutations of the crossed processes
        permutations = []

        # Store the diagram tags for processes, to allow for
        # identifying identical matrix elements already at this stage.
        model = process_definition['model']
        
        islegs_orig = [leg for leg in process_definition['legs'] \
                 if leg['state'] == False]
        fslegs = [leg for leg in process_definition['legs'] \
                 if leg['state'] == True]        
        
        isids = [leg['ids'] for leg in process_definition['legs'] \
                 if leg['state'] == False]
        fsids = [leg['ids']  for leg in process_definition['legs'] \
                 if leg['state'] == True]
        polids = [tuple(leg['polarization'])  for leg in process_definition['legs'] \
                 if leg['state'] == True]

        masses = {id: model.get_particle(abs(id)).get('mass')  for leg in process_definition['legs'] for id in leg['ids']} 

        # keep track of the 'is_tagged' property of the legs if needed
        try:
            fstags = [leg['is_tagged'] for leg in process_definition['legs'] \
                 if 'is_tagged' in leg.keys() and leg['state'] == True]

        except KeyError:
            fstags = []

        try:
            istags = [leg['is_tagged'] for leg in process_definition['legs'] \
                 if 'is_tagged' in leg.keys() and leg['state'] == False]

        except KeyError:
            istags = []

        # Generate all combinations for the initial state
        for prod in itertools.product(*isids):
            if any(istags):
                if not all(istags):
                    raise MadGraph5Error("Tagging only one initial-state particle is not allowed")
                islegs = [\
                        fks_tag.TagLeg({'id':id, 'state': False, 
                                        'polarization': isleg['polarization'],
                                        'onium': isleg['onium'],
                                        'offshell': isleg['offshell'], 
                                        'is_tagged': tag}) \
                        for id, isleg, tag in zip(prod, islegs_orig, istags)]
            else:
                def get_flavor(beamid,id):
                    flavor = []
                    if abs(id) in model.get('merged_particles'):
                        for f in islegs_orig[beamid]['flavor']:
                            # multi-particle store the flavor for many id -> need to filter the one we are looking at
                            # (same merged group AND same particle/antiparticle
                            # sign: in `define qa = u d~` the u belongs to the
                            # Q leg and the d~ to the Qx one)
                            if abs(f) in model.get('merged_particles')[abs(id)] \
                                    and f * id > 0:
                                flavor.append(-1*f)
                    return flavor

                islegs = [\
                        base_objects.Leg({'id':id, 'state': False,
                                          'polarization': islegs_orig[i]['polarization'],
                                          'flavor': get_flavor(i,id),
                                          'onium': islegs_orig[i]['onium'],
                                          'offshell': islegs_orig[i]['offshell']}) \
                    for i,id in enumerate(prod)]

            # check for longitudinal photon
            invalid = False
            for i, l in enumerate(islegs):
                if 0 in l['polarization'] and  masses[l['id']] == "ZERO":
                    l['polarization'] = [l for l in l['polarization'] if l != 0]
                    if len(l['polarization']) == 0:
                        invalid = True
                        break
            if invalid:
                continue

            # Generate all combinations for the final state, and make
            # sure to remove double counting

            red_fsidlist = set()

            def get_fs_flavor(id, fsleg):
                flavor = []
                if abs(id) in model.get('merged_particles'):
                    for f in fsleg['flavor']:
                        # multi-particle store the flavor for many id -> need to filter the one we are looking at
                        # (same merged group AND same particle/antiparticle sign)
                        if abs(f) in model.get('merged_particles')[abs(id)] \
                                and f * id > 0:
                            flavor.append(f)
                return flavor

            for prod in itertools.product(*fsids):
                # The per-leg flavor restriction is part of the tag: with
                # `define l+ = u d~` / `define l- = u~ d`, (Q, Qx) restricted to
                # (u, u~) and (Qx, Q) restricted to (d~, d) are NOT permutations
                # of each other even though their merged ids are.
                fsflavors = [tuple(sorted(get_fs_flavor(id, fsleg)))
                             for id, fsleg in zip(prod, fslegs)]
                tag = zip(prod, polids, fsflavors)
                tag = sorted(tag)
                # Remove double counting between final states
                if tuple(tag) in red_fsidlist:
                    continue
                
                red_fsidlist.add(tuple(tag))
                # Generate leg list for process
                leg_list = [copy.copy(leg) for leg in islegs]
                
                if not fstags: 
                    leg_list.extend([\
                            base_objects.Leg({'id':id, 'state': True,
                                              'polarization': fsleg['polarization'],
                                              'flavor': get_fs_flavor(id, fsleg),
                                              'onium': fsleg['onium'],
                                              'offshell': fsleg['offshell']}) \
                            for id, fsleg in zip(prod, fslegs)])
                else:
                    leg_list.extend([\
                            fks_tag.TagLeg({'id':id, 'state': True, 'polarization': fsleg['polarization'], 'onium': fsleg['onium'], 'is_tagged': tag}) \
                            for id, fsleg, tag in zip(prod, fslegs, fstags)])


                legs = base_objects.LegList(leg_list)


                # check for longitudinal photon
                invalid = False
                for l in legs[len(islegs):]:
                    if 0 in l['polarization'] and  masses[l['id']] == "ZERO" and not l['offshell']:
                        l['polarization'] =list(l['polarization'])
                        l['polarization'].remove(0)
                        if len(l['polarization']) == 0:
                            invalid = True
                            break
                if invalid:
                    continue

                # Check for crossed processes
                sorted_legs = sorted([(l,i+1) for (i,l) in \
                                   enumerate(legs.get_outgoing_id_list(model))])
                permutation = [l[1] for l in sorted_legs]
                
                sorted_legs = array.array('i', [l[0] for l in sorted_legs])

                # Check for six-quark processes
                if ignore_six_quark_processes and \
                       len([i for i in sorted_legs if abs(i) in \
                            ignore_six_quark_processes]) >= 6:
                    continue
                    
                # Check if crossed process has already failed,
                # in that case don't check process
                if sorted_legs in failed_procs:
                    continue

                # If allowed check mass validity [assume 1->N]
                if use_numerical:
                    # check that final state has lower mass than initial state
                    initial_mass = abs(model['parameter_dict'][model.get_particle(legs[0].get('id')).get('mass')])
                    if initial_mass == 0:
                         continue
                    for leg in legs[1:]:
                        m = model['parameter_dict'][model.get_particle(leg.get('id')).get('mass')]
                        initial_mass -= abs(m)
                    if initial_mass.real <= 0:
                        continue

                # Setup process
                process = process_definition.get_process_with_legs(legs) 
                
                fast_proc = \
                          array.array('i',[leg.get('id') for leg in legs])
                if collect_mirror_procs and \
                        process_definition.get_ninitial() == 2:
                    # Check if mirrored process is already generated
                    mirror_proc = \
                              array.array('i', [fast_proc[1], fast_proc[0]] + \
                                          list(fast_proc[2:]))
                    # The mirror flag means "the same matrix element with
                    # the beams swapped", so the per-leg flavor restrictions
                    # of the two beams must swap as well: with `define qa =
                    # u d~` and `define qb = u~ d`, the mirror of Q Qx > z
                    # [u u~] is u~ u, not the requested Qx Q > z [d~ d].
                    mirror_amp = None
                    for iproc, other_proc in enumerate(non_permuted_procs):
                        if other_proc != mirror_proc:
                            continue
                        other_legs = amplitudes[iproc].get('process').get('legs')
                        swapped = [other_legs[1], other_legs[0]] + \
                                  list(other_legs[2:])
                        if all(sorted(l1.get('flavor')) == sorted(l2.get('flavor'))
                               for l1, l2 in zip(legs, swapped)):
                            mirror_amp = amplitudes[iproc]
                            break
                    if mirror_amp is not None:
                        # Mirror process found
                        mirror_amp.set('has_mirror_process', True)
                        logger.info("Process %s added to mirror process %s" % \
                                    (process.base_string(),
                                     mirror_amp.get('process').base_string()))
                        continue
                        
                # Check for successful crossings, unless we have specified
                # properties that break crossing symmetry
                # (an interference process carries a right-hand process with
                # its own s-channel constraints: no reuse by crossing either)
                if not process.get('required_s_channels') and \
                   not process.get('forbidden_onsh_s_channels') and \
                   not process.get('forbidden_s_channels') and \
                   not process.get('is_decay_chain') and not diagram_filter \
                   and not process.get_interference_mode():
                    try:
                        crossed_index = success_procs.index(sorted_legs)
                        # The relabeling of legs for loop amplitudes is cumbersome
                        # and does not save so much time. It is disable here and
                        # we use the key 'loop_diagrams' to decide whether
                        # it is an instance of LoopAmplitude.
                        if 'loop_diagrams' in amplitudes[crossed_index]:
                            raise ValueError
                    except ValueError:
                        # No crossing found, just continue
                        pass
                    else:
                        # merge_crossing relies on the merged matrix element
                        # covering every crossing; with a per-leg flavor
                        # restriction it does not (z > Qx Q [d~ d] from
                        # `define l+ = u d~` is no crossing of z > Q Qx
                        # [u u~] flavor-wise), so keep such a process.
                        if not merge_crossing or \
                                any(l.get('flavor') for l in legs):
                            # Found crossing - reuse amplitude
                            amplitude = MultiProcess.cross_amplitude(\
                                amplitudes[crossed_index],
                                process,
                                permutations[crossed_index],
                                permutation)
                            amplitudes.append(amplitude)
                            success_procs.append(sorted_legs)
                            permutations.append(permutation)
                            non_permuted_procs.append(fast_proc)
                            logger.info("Crossed process found for %s, reuse diagrams." % \
                                        process.base_string())
                        else:
                            logger.info("Crossed process found for %s, do not generate diagrams." % \
                                        process.base_string())
                        continue
                    
                # Create new amplitude
                amplitude = cls.get_amplitude_from_proc(process,
                                                        loop_filter=loop_filter)

                try:
                    result = amplitude.generate_diagrams(diagram_filter=diagram_filter)
                except InvalidCmd as error:
                    failed_procs.append(sorted_legs)
                else:
                    # Succeeded in generating diagrams
                    if amplitude.get('diagrams'):
                        amplitudes.append(amplitude)
                        success_procs.append(sorted_legs)
                        permutations.append(permutation)
                        non_permuted_procs.append(fast_proc)
                    elif not result:
                        # Diagram generation failed for all crossings
                        failed_procs.append(sorted_legs)
 
        # Raise exception if there are no amplitudes for this process
        if not amplitudes:
            if len(failed_procs) == 1 and 'error' in locals():
                raise error
            else:
                raise NoDiagramException("No amplitudes generated from process %s. Please enter a valid process" % \
                  process_definition.nice_string())
        

        # Return the produced amplitudes
        return amplitudes

    @classmethod
    def get_amplitude_from_proc(cls,proc,**opts):
        """ Return the correct amplitude type according to the characteristics of
            the process proc. The only option that could be specified here is
            loop_filter and it is of course not relevant for a tree amplitude."""
            
        return Amplitude({"process": proc})
        

    @staticmethod
    def find_optimal_process_orders(process_definition, diagram_filter=False):
        """Find the minimal WEIGHTED order for this set of processes.

        The algorithm:

        1) Check the coupling hierarchy of the model. Assign all
        particles to the different coupling hierarchies so that a
        particle is considered to be in the highest hierarchy (i.e.,
        with lowest value) where it has an interaction.
        
        2) Pick out the legs in the multiprocess according to the
        highest hierarchy represented (so don't mix particles from
        different hierarchy classes in the same multiparticles!)

        3) Find the starting maximum WEIGHTED order as the sum of the
        highest n-2 weighted orders

        4) Pick out required s-channel particle hierarchies, and use
        the highest of the maximum WEIGHTED order from the legs and
        the minimum WEIGHTED order extracted from 2*s-channel
        hierarchys plus the n-2-2*(number of s-channels) lowest
        leg weighted orders.

        5) Run process generation with the WEIGHTED order determined
        in 3)-4) - # final state gluons, with all gluons removed from
        the final state

        6) If no process is found, increase WEIGHTED order by 1 and go
        back to 5), until we find a process which passes. Return that
        order.

        7) Continue 5)-6) until we reach (n-2)*(highest hierarchy)-1.
        If still no process has passed, return
        WEIGHTED = (n-2)*(highest hierarchy)
        """

        assert isinstance(process_definition, base_objects.ProcessDefinition), \
                                    "%s not valid ProcessDefinition object" % \
                                    repr(process_definition)

        processes = base_objects.ProcessList()
        amplitudes = AmplitudeList()

        # If there are already couplings defined, return
        if process_definition.get('orders') or \
                process_definition.get('overall_orders') or \
                process_definition.get('NLO_mode')=='virt':
            return process_definition.get('orders')

        # If this is a decay process (and not a decay chain), return
        if process_definition.get_ninitial() == 1 and not \
                process_definition.get('is_decay_chain'):
            return process_definition.get('orders')

        logger.info("Checking for minimal orders which gives processes.")
        logger.info("Please specify coupling orders to bypass this step.")

        # Calculate minimum starting guess for WEIGHTED order
        max_order_now, particles, hierarchy = \
                                       process_definition.get_minimum_WEIGHTED()
        coupling = 'WEIGHTED'

        model = process_definition.get('model')
        
        # Extract the initial and final leg ids
        isids = [leg['ids'] for leg in \
                 [leg for leg in process_definition['legs'] if leg['state'] == False]]
        fsids = [leg['ids'] for leg in \
                 [leg for leg in process_definition['legs'] if leg['state'] == True]]

        max_WEIGHTED_order = \
                        (len(fsids + isids) - 2)*int(model.get_max_WEIGHTED())
        # get the definition of the WEIGHTED
        hierarchydef = process_definition['model'].get('order_hierarchy')
        tmp = []
        hierarchy = list(hierarchydef.items())
        hierarchy.sort()
        for key, value in hierarchydef.items():
            # an order of hierarchy 0 (like the hidden INTERFERENCE_ORDER)
            # does not enter WEIGHTED
            if value == 0:
                continue
            elif value>1:
                tmp.append('%s*%s' % (value,key))
            else:
                tmp.append('%s' % key)
        wgtdef = '+'.join(tmp)
        # Run diagram generation with increasing max_order_now until
        # we manage to get diagrams
        while max_order_now < max_WEIGHTED_order:
            logger.info("Trying coupling order WEIGHTED<=%d: WEIGHTED IS %s" % (max_order_now, wgtdef))

            oldloglevel = logger.level
            logger.setLevel(logging.WARNING)

            # failed_procs are processes that have already failed
            # based on crossing symmetry
            failed_procs = []
            # Generate all combinations for the initial state        
            for prod in itertools.product(*isids):
                islegs = [ base_objects.Leg({'id':id, 'state': False}) \
                        for id in prod]

                # Generate all combinations for the final state, and make
                # sure to remove double counting

                red_fsidlist = []

                for prod in itertools.product(*fsids):

                    # Remove double counting between final states
                    if tuple(sorted(prod)) in red_fsidlist:
                        continue

                    red_fsidlist.append(tuple(sorted(prod)));

                    # Remove gluons from final state if QCD is among
                    # the highest coupling hierarchy
                    nglue = 0
                    if 21 in particles[0]:
                        nglue = len([id for id in prod if id == 21])
                        prod = [id for id in prod if id != 21]

                    # Generate leg list for process
                    leg_list = [copy.copy(leg) for leg in islegs]

                    leg_list.extend([\
                            base_objects.Leg({'id':id, 'state': True}) \
                            for id in prod])

                    legs = base_objects.LegList(leg_list)

                    # Set summed coupling order according to max_order_now
                    # subtracting the removed gluons
                    coupling_orders_now = {coupling: max_order_now - \
                                           nglue * model['order_hierarchy']['QCD']}

                    # Setup process
                    process = base_objects.Process({\
                              'legs':legs,
                              'model':model,
                              'id': process_definition.get('id'),
                              'orders': coupling_orders_now,
                              'required_s_channels': \
                                 process_definition.get('required_s_channels'),
                              'forbidden_onsh_s_channels': \
                                 process_definition.get('forbidden_onsh_s_channels'),
                              'sqorders_types': \
                                 process_definition.get('sqorders_types'),
                              'squared_orders': \
                                 process_definition.get('squared_orders'),
                              'split_orders': \
                                 process_definition.get('split_orders'), 
                              'forbidden_s_channels': \
                                 process_definition.get('forbidden_s_channels'),
                              'forbidden_particles': \
                                 process_definition.get('forbidden_particles'),
                              'is_decay_chain': \
                                 process_definition.get('is_decay_chain'),
                              'overall_orders': \
                                 process_definition.get('overall_orders'),
                              'split_orders': \
                                 process_definition.get('split_orders')})

                    # Check for couplings with given expansion orders
                    process.check_expansion_orders()

                    # Check for crossed processes
                    sorted_legs = sorted(legs.get_outgoing_id_list(model))
                    # Check if crossed process has already failed
                    # In that case don't check process
                    if tuple(sorted_legs) in failed_procs and not process_definition.get('forbidden_s_channels'):
                        continue

                    amplitude = Amplitude({'process': process})
                    try:
                        amplitude.generate_diagrams(diagram_filter=diagram_filter)
                    except InvalidCmd as error:
                        failed_procs.append(tuple(sorted_legs))
                    else:
                        if amplitude.get('diagrams'):
                            # We found a valid amplitude. Return this order number
                            logger.setLevel(oldloglevel)
                            return {coupling: max_order_now}
                        else:
                            failed_procs.append(tuple(sorted_legs))
            # No processes found, increase max_order_now
            max_order_now += 1
            logger.setLevel(oldloglevel)

        # If no valid processes found with nfinal-1 couplings, return maximal
        return {coupling: max_order_now}

    @staticmethod
    def cross_amplitude(amplitude, process, org_perm, new_perm):
        """Return the amplitude crossed with the permutation new_perm"""
        # Create dict from original leg numbers to new leg numbers
        perm_map = dict(list(zip(org_perm, new_perm)))
        # Initiate new amplitude
        new_amp = copy.copy(amplitude)
        # Number legs
        for i, leg in enumerate(process.get('legs')):
            leg.set('number', i+1)
        # Set process
        new_amp.set('process', process)
        # Now replace the leg numbers in the diagrams
        diagrams = base_objects.DiagramList([d.renumber_legs(perm_map,
                                             process.get('legs'),) for \
                                             d in new_amp.get('diagrams')])
        new_amp.set('diagrams', diagrams)
        new_amp.trim_diagrams()

        # Make sure to reset mirror process
        new_amp.set('has_mirror_process', False)
        
        return new_amp
        
#===============================================================================
# Global helper methods
#===============================================================================

def expand_list(mylist):
    """Takes a list of lists and elements and returns a list of flat lists.
    Example: [[1,2], 3, [4,5]] -> [[1,3,4], [1,3,5], [2,3,4], [2,3,5]]
    """

    # Check that argument is a list
    assert isinstance(mylist, list), "Expand_list argument must be a list"

    res = []

    tmplist = []
    for item in mylist:
        if isinstance(item, list):
            tmplist.append(item)
        else:
            tmplist.append([item])

    for item in itertools.product(*tmplist):
        res.append(list(item))

    return res

def expand_list_list(mylist):
    """Recursive function. Takes a list of lists and lists of lists
    and returns a list of flat lists.
    Example: [[1,2],[[4,5],[6,7]]] -> [[1,2,4,5], [1,2,6,7]]
    """

    res = []

    if not mylist or len(mylist) == 1 and not mylist[0]:
        return [[]]

    # Check the first element is at least a list
    assert isinstance(mylist[0], list), \
              "Expand_list_list needs a list of lists and lists of lists"

    # Recursion stop condition, one single element
    if len(mylist) == 1:
        if isinstance(mylist[0][0], list):
            return mylist[0]
        else:
            return mylist

    if isinstance(mylist[0][0], list):
        for item in mylist[0]:
            # Here the recursion happens, create lists starting with
            # each element of the first item and completed with 
            # the rest expanded
            for rest in expand_list_list(mylist[1:]):
                reslist = copy.copy(item)
                reslist.extend(rest)
                res.append(reslist)
    else:
        for rest in expand_list_list(mylist[1:]):
            reslist = copy.copy(mylist[0])
            reslist.extend(rest)
            res.append(reslist)


    return res

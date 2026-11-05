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

"""Classes, methods and functions required to write QCD color information 
for a diagram and build a color basis, and to square a QCD color string for
squared diagrams and interference terms."""

from __future__ import absolute_import
import collections
import copy
import fractions
import itertools
import itertools
import logging
import operator
import re
import array
import math
import madgraph
import madgraph.core.color_algebra as color_algebra
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.base_objects as base_objects
import madgraph.various.misc as misc
from functools import reduce

if madgraph.ordering:
    set = misc.OrderedSet

logger = logging.getLogger('madgraph.color_amp')

#===============================================================================
# Del Duca-Dixon-Maltoni (adjoint) color basis
#===============================================================================
# For a process whose color structure is purely adjoint (all colored external
# legs are octets) the historical trace basis Tr(1,sigma(2),...,sigma(n)) has
# (n-1)! elements while, thanks to the Jacobi identity, the color factor of any
# such amplitude can be written on the (n-2)! "half-ladder" (multi-peripheral)
# structures
#     F(sigma) = f(1,sigma(2),x1) f(x1,sigma(3),x2) ... f(x(n-3),sigma(n-1),n)
# where legs 1 and n are kept fixed at the two ends of the ladder. This is the
# Del Duca-Dixon-Maltoni basis. Using it divides the number of JAMPs by (n-1)
# and the size of the color matrix by (n-1)^2.
#
# Module level switch selecting the color basis used for fully adjoint
# processes. Set through 'set color_basis' in the MG5 interface (the exporters
# which need a color flow decomposition, i.e. anything writing leshouche
# information, must keep the trace basis).
ddm_basis = False
# Whether the trace basis must be built next to the DDM one. Needed by the
# output formats which have to assign a color flow to an event: the color sum
# then runs over the (n-2)! DDM structures while the color flow probabilities
# keep using the (n-1)! trace ones, obtained from the DDM JAMPs through the
# Kleiss-Kuijf relations.
ddm_flow_basis = False


def set_ddm_basis(value, with_flow=False):
    """Set the module wide switch selecting the DDM color basis."""

    global ddm_basis, ddm_flow_basis
    ddm_basis = bool(value)
    ddm_flow_basis = ddm_basis and bool(with_flow)


class DDMError(Exception):
    """Raised when a color string cannot be mapped onto the DDM basis. Always
    caught by ColorBasis.build, which then falls back on the trace basis."""


def ddm_half_ladder(perm, first, last):
    """Return the immutable color string of the DDM half-ladder structure
        f(first,perm[0],-1) f(-1,perm[1],-2) ... f(-(m-1),perm[m-1],last)
    for the ordered tuple perm of the (n-2) legs sitting between the two fixed
    ends first and last."""

    if len(perm) == 1:
        col_objs = [color_algebra.f(first, perm[0], last)]
    else:
        col_objs = [color_algebra.f(first, perm[0], -1)]
        col_objs.extend([color_algebra.f(-(i + 1), leg, -(i + 2)) \
                         for i, leg in enumerate(perm[1:-1])])
        col_objs.append(color_algebra.f(-(len(perm) - 1), perm[-1], last))

    return color_algebra.ColorString(col_objs).to_immutable()


def _reorder_sign(stored, wanted):
    """Signature of the permutation bringing the three indices of an f object
    from the order 'stored' to the order 'wanted'. f is totally antisymmetric,
    so f(stored) = _reorder_sign(stored,wanted) * f(wanted)."""

    perm = [stored.index(index) for index in wanted]
    sign = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                sign = -sign

    return sign


class _ColorTree(object):
    """A product of f objects seen as a tree: the f's are the nodes, the summed
    (negative) indices the internal edges and the external (positive) indices
    the leaves. Provides the reduction onto the DDM half-ladder basis."""

    def __init__(self, col_str):
        """Build the tree from a ColorString made of f objects only. Raise
        DDMError as soon as the string is not a fully adjoint color tree."""

        self.nodes = []
        for col_obj in col_str:
            if col_obj.__class__.__name__ == 'ColorOne':
                continue
            if type(col_obj) is not color_algebra.f:
                raise DDMError("%s is not an f object" % str(col_obj))
            self.nodes.append(tuple(col_obj))

        if not self.nodes:
            raise DDMError("empty color string")

        # Locate each index. External indices must appear once, summed ones
        # exactly twice and in two different nodes.
        self.where = collections.defaultdict(list)
        for i, node in enumerate(self.nodes):
            if node[0] == node[1] or node[1] == node[2] or node[0] == node[2]:
                raise DDMError("f object %s has repeated indices" % str(node))
            for index in node:
                self.where[index].append(i)

        self.externals = []
        nb_internal = 0
        for index, nodes in self.where.items():
            if index > 0:
                if len(nodes) != 1:
                    raise DDMError("external index %i appears %i times" % \
                                                        (index, len(nodes)))
                self.externals.append(index)
            else:
                if len(nodes) != 2:
                    raise DDMError("summed index %i appears %i times" % \
                                                        (index, len(nodes)))
                nb_internal += 1

        # A connected graph with V nodes and V-1 edges is a tree
        if nb_internal != len(self.nodes) - 1:
            raise DDMError("color structure is not a tree")

    def _neighbour(self, index, node):
        """The node sharing the summed index 'index' with node 'node'."""

        first, second = self.where[index]
        return second if first == node else first

    def _spine(self, first, last):
        """The list of nodes on the path going from the leaf 'first' to the
        leaf 'last'."""

        if first not in self.externals or last not in self.externals:
            raise DDMError("legs %s and %s are not both external here" % \
                                                              (first, last))
        start = self.where[first][0]
        end = self.where[last][0]

        # Depth first search on the node tree, keeping track of the path
        stack = [(start, None, [start])]
        while stack:
            node, from_index, path = stack.pop()
            if node == end:
                return path
            for index in self.nodes[node]:
                if index > 0 or index == from_index:
                    continue
                stack.append((self._neighbour(index, node), index,
                              path + [self._neighbour(index, node)]))

        raise DDMError("color structure is not connected")

    def _subtree(self, index, from_node):
        """Expansion of the adjoint matrix associated to the subtree hanging on
        the edge 'index' of node 'from_node', as a list of
        (sign, ordered tuple of legs). A single leaf gives the generator
        itself, and a node with two children A and B gives the commutator
        [M_B, M_A] (which is where the (n-2)! counting comes from)."""

        if index > 0:
            return [(1, (index,))]

        node_i = self._neighbour(index, from_node)
        node = self.nodes[node_i]
        children = list(node)
        children.remove(index)
        alpha, beta = children
        sign = _reorder_sign(node, (index, alpha, beta))

        exp_a = self._subtree(alpha, node_i)
        exp_b = self._subtree(beta, node_i)

        result = []
        for (sa, wa), (sb, wb) in itertools.product(exp_a, exp_b):
            result.append((sign * sa * sb, wb + wa))
            result.append((-sign * sa * sb, wa + wb))

        return result

    def reduce_to_ddm(self, first, last):
        """Decompose the tree onto the DDM half-ladder basis with the legs
        'first' and 'last' at the two ends. Returns {ordered legs: coefficient}
        where the keys are the (n-2) other external legs in ladder order."""

        spine = self._spine(first, last)

        # Split each node of the spine into (incoming, hanging, outgoing)
        global_sign = 1
        hanging = []
        for pos, node_i in enumerate(spine):
            node = self.nodes[node_i]
            if pos == 0:
                in_index = first
            else:
                in_index = [i for i in node if i in self.nodes[spine[pos - 1]]][0]
            if pos == len(spine) - 1:
                out_index = last
            else:
                out_index = [i for i in node if i in self.nodes[spine[pos + 1]]][0]
            off_index = [i for i in node if i not in (in_index, out_index)][0]
            global_sign *= _reorder_sign(node, (in_index, off_index, out_index))
            hanging.append(self._subtree(off_index, node_i))

        # The whole structure is the matrix product M_m ... M_1 between the
        # ends, and (M_i1 ... M_ik) contracted between 'last' and 'first' is the
        # half-ladder with the legs in the reversed order.
        result = collections.defaultdict(int)
        for combination in itertools.product(*reversed(hanging)):
            sign = global_sign
            word = []
            for term_sign, term_word in combination:
                sign *= term_sign
                word.extend(term_word)
            result[tuple(reversed(word))] += sign

        return dict((perm, coeff) for perm, coeff in result.items() if coeff)


def reduce_to_ddm(col_str, first, last):
    """Decompose the ColorString col_str (a product of f objects) onto the DDM
    half-ladder basis, returning a ColorFactor whose strings are the basis
    elements. Raise DDMError if col_str is not a fully adjoint color tree."""

    decomposition = _ColorTree(col_str).reduce_to_ddm(first, last)

    col_fact = color_algebra.ColorFactor()
    for perm, coeff in decomposition.items():
        new_str = color_algebra.ColorString()
        new_str.from_immutable(ddm_half_ladder(perm, first, last))
        new_str.coeff = col_str.coeff * coeff
        new_str.is_imaginary = col_str.is_imaginary
        new_str.Nc_power = col_str.Nc_power
        new_str.loop_Nc_power = col_str.loop_Nc_power
        col_fact.append(new_str)

    return col_fact


#===============================================================================
# ColorBasis
#===============================================================================
class ColorBasis(dict):
    """The ColorBasis object is a dictionary created from an amplitude. Keys
    are the different color structures present in the amplitude. Values have
    the format (diag,(index c1, index c2,...), coeff, is_imaginary, Nc_power) 
    where diag is the diagram index, (index c1, index c2,...) the list of 
    indices corresponding to the chose color parts for each vertex in the 
    diagram, coeff the corresponding coefficient (a fraction), is_imaginary
    if this contribution is real or complex, and Nc_power the Nc power."""

    # Memo of simplified color structures, keyed on the canonical string alone:
    # coeff is divided out on store and re-applied on a hit, loop_Nc_power is
    # stamped per use, is_imaginary/Nc_power are fixed by the structure itself.
    _canonical_dict = {}

    # Dictionary store the raw colorize information
    _list_color_dict = []

    # Whether relabel_canonical may take its shortcut, per canonical form
    _fast_relabel_dict = {}

    # Color objects whose canonical form is fully determined by
    # permute_immutable (Tr is cyclic, T is an open chain, ColorOne is empty).
    fast_relabel_objects = frozenset(['Tr', 'T', 'ColorOne'])

    # Legs at the two ends of the DDM half-ladders (None for the trace basis)
    _ddm_ends = None

    # Trace basis built next to a DDM one, carrying the color flows
    _flow_basis = None

    # Dictionary to save the DDM decompositions already done
    _ddm_dict = {}


    class ColorBasisError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a color basis object."""
        pass

    def colorize(self, diagram, model):
        """Takes a diagram and a model and outputs a dictionary with keys being
        color coefficient index tuples and values a color string (before 
        simplification)."""

        # The smallest value used to create new summed indices
        min_index = -1000
        # The dictionary to be output
        res_dict = {}
        # The dictionary for book keeping of replaced indices
        repl_dict = {}

        for i, vertex in enumerate(diagram.get('vertices')):
            min_index, res_dict = self.add_vertex(vertex, diagram, model,
                            repl_dict, res_dict, min_index)

        # if the process has no QCD particles
        # Return a list filled with ColorOne if all entries are empty ColorString()
        empty_colorstring = color_algebra.ColorString()
        if all(cs == empty_colorstring for cs in res_dict.values()):
            res_dict = dict((key, color_algebra.ColorString(
                               [color_algebra.ColorOne()])) for key in res_dict)
                    
        return res_dict

    

    def add_vertex(self, vertex, diagram, model,
                   repl_dict, res_dict, min_index, id0_rep=[]):
        """Update repl_dict, res_dict and min_index for normal vertices.
        Returns the min_index reached and the result dictionary in a tuple.
        If the id0_rep list is not None, perform the requested replacement on the
        last leg number before going further."""

        # Create a list of (color,leg number) pairs for the vertex, where color
        # can be negative for anti particles

        color_num_pairs = []
        pdg_codes = []
                
        for index, leg in enumerate(vertex.get('legs')):
            curr_num = leg.get('number')
            curr_part = model.get('particle_dict')[leg.get('id')]
            curr_color = curr_part.get_color()
            curr_pdg = curr_part.get_pdg_code()

            # If this is the next-to-last vertex and the last vertex is
            # the special identity id=0, start by applying the replacement rule
            # on the last vertex.
            if index == len(vertex.get('legs')) - 1 and \
                    curr_num in id0_rep:
                    curr_num = id0_rep[id0_rep.index(curr_num) - 1]

            # If this is the last leg and not the last vertex 
            # flip color. If it is not the last, AND not the next-to-last
            # before an id=0 vertex, replace last index by a new summed index.
            if index == len(vertex.get('legs')) - 1 and \
                vertex != diagram.get('vertices')[-1]:
                curr_color = curr_part.get_anti_color()
                curr_pdg = curr_part.get_anti_pdg_code()
                if not id0_rep:
                    if not ( diagram.get('vertices')[-1].get('id')==-1 and \
                    vertex == diagram.get('vertices')[-2]):
                        repl_dict[curr_num] = min_index
                        min_index = min_index - 1
                    else:                  
                        repl_dict[curr_num] = \
                          max(l.get('number') for l in \
                                        diagram.get('vertices')[-1].get('legs'))

            # Take into account previous replacements
            try:
                curr_num = repl_dict[curr_num]
            except KeyError:
                pass

            color_num_pairs.append((curr_color, curr_num))
            pdg_codes.append(curr_pdg)

        if vertex != diagram.get('vertices')[-1]:
            # Put the resulting wavefunction first, to make
            # wavefunction call more natural
            last_color_num = color_num_pairs.pop(-1)
            color_num_pairs.insert(0, last_color_num)
            last_pdg = pdg_codes.pop(-1)
            pdg_codes.insert(0, last_pdg)

        # Order the legs according to the interaction particles
        if vertex.get('id')!=-1:
            interaction_pdgs = [p.get_pdg_code() for p in \
                                model.get_interaction(vertex.get('id')).\
                                get('particles')]
        else:
            interaction_pdgs = [l.get('id') for l in vertex.get('legs')]

        sorted_color_num_pairs = []
        #print "interactions_pdg=",interaction_pdgs
        #print "pdg_codes=",pdg_codes        
        for i, pdg in enumerate(interaction_pdgs):
            index = pdg_codes.index(pdg)
            pdg_codes.pop(index)
            sorted_color_num_pairs.append(color_num_pairs.pop(index))

        if color_num_pairs:
            raise base_objects.PhysicsObject.PhysicsObjectError

        color_num_pairs = sorted_color_num_pairs

        # Create a list of associated leg number following the same order
        list_numbers = [p[1] for p in color_num_pairs]

        # ... and the associated dictionary for replacement
        match_dict = dict(enumerate(list_numbers))

        if vertex['id'] == -1:
            return (min_index, res_dict)

        # Update the result dict using the current vertex ColorString object
        # If more than one, create different entries
        inter_color = model.get_interaction(vertex['id'])['color']
        inter_indices = [key[0] for key in \
                        model.get_interaction(vertex['id'])['couplings'].keys()]
        
        # For colorless vertices, return a copy of res_dict
        # Where one 0 has been added to each color index chain key
        if not inter_color:
            new_dict = {}
            for k, v in res_dict.items():
                new_key = tuple(list(k) + [0])
                new_dict[new_key] = v
            # If there is no result until now, create an empty CS...
            if not new_dict:
                new_dict[(0,)] = color_algebra.ColorString()
            return (min_index, new_dict)

        new_res_dict = {}
        for i, col_str in \
                enumerate(inter_color):

            # Ignore color string if it doesn't correspond to any coupling
            if i not in inter_indices:
                continue
            
            # Build the new element
            assert type(col_str) == color_algebra.ColorString 
            mod_col_str = col_str.create_copy()

            # Replace summed (negative) internal indices
            list_neg = []
            for col_obj in mod_col_str:
                list_neg.extend([ind for ind in col_obj if ind < 0])
            internal_indices_dict = {}
            # This notation is to remove duplicates
            for index in misc.make_unique(list_neg):
                internal_indices_dict[index] = min_index
                min_index = min_index - 1
            mod_col_str.replace_indices(internal_indices_dict)

            # Replace other (positive) indices using the match_dic
            mod_col_str.replace_indices(match_dict)

            # If we are considering the first vertex, simply create
            # new entries

            if not res_dict:
                new_res_dict[tuple([i])] = mod_col_str
            #... otherwise, loop over existing elements and multiply
            # the color strings
            else:
                for ind_chain, col_str_chain in res_dict.items():
                    new_col_str_chain = col_str_chain.create_copy()
                    new_col_str_chain.product(mod_col_str)
                    new_res_dict[tuple(list(ind_chain) + [i])] = \
                        new_col_str_chain

        return (min_index, new_res_dict)


    def _fast_relabel_possible(self, col_fact):
        """The shortcut is only attempted on color factors made of objects
        whose canonical form is known, and whose indices are all distinct
        within a string so that no contraction identity can fire."""

        for col_str in col_fact:
            indices = []
            for name, idx in col_str.to_immutable():
                if name not in self.fast_relabel_objects:
                    return False
                indices.extend(idx)
            if len(indices) != len(dict.fromkeys(indices)):
                return False
        return True

    @staticmethod
    def _canonicalize_strings(col_fact):
        """Put every color string of col_fact back in canonical form in place
        and drop the vanishing ones, mirroring what ColorFactor.simplify does
        for an expression which is already simplified."""

        for col_str in col_fact:
            immutable = col_str.to_immutable()
            canonical = permute_immutable(immutable, {})
            if canonical != immutable:
                col_str.from_immutable(canonical)
                col_str.immutable = None
                col_str.canonical = None
        return color_algebra.ColorFactor([col_str for col_str in col_fact \
                                          if col_str.coeff != 0])

    def relabel_canonical(self, col_fact, canonical_rep):
        """Return col_fact, which is an already simplified color factor with
        relabelled indices, put back in canonical form. Equivalent to
        col_fact.simplify().simplify(); which of the two is used is decided
        once per canonical representation by running both and comparing."""

        verdict = self._fast_relabel_dict.get(canonical_rep)
        if verdict is True:
            return self._canonicalize_strings(col_fact)
        if verdict is False:
            return col_fact.simplify().simplify()

        # First time this color structure is recycled: check the shortcut
        # against the full simplification before trusting it.
        slow = col_fact.create_copy().simplify().simplify()
        if not self._fast_relabel_possible(col_fact):
            self._fast_relabel_dict[canonical_rep] = False
            return slow
        fast = self._canonicalize_strings(col_fact)
        verdict = len(fast) == len(slow) and \
                  all(f.to_immutable() == s.to_immutable() and
                      f.coeff == s.coeff and
                      f.is_imaginary == s.is_imaginary and
                      f.Nc_power == s.Nc_power and
                      f.loop_Nc_power == s.loop_Nc_power
                      for f, s in zip(fast, slow))
        self._fast_relabel_dict[canonical_rep] = verdict
        return fast if verdict else slow

    def get_ddm_ends(self):
        """If every color structure of this basis is a fully adjoint color tree
        over one and the same set of external legs -- i.e. if the process is a
        pure multi-gluon one -- return the two legs to be put at the ends of the
        DDM half-ladders. Return None otherwise."""

        legs = None
        for colorize_dict in self._list_color_dict:
            for col_str in colorize_dict.values():
                externals = []
                for col_obj in col_str:
                    if col_obj.__class__.__name__ == 'ColorOne':
                        continue
                    if type(col_obj) is not color_algebra.f:
                        return None
                    externals.extend([i for i in col_obj if i > 0])
                if len(externals) != len(set(externals)):
                    return None
                externals = sorted(externals)
                if legs is None:
                    legs = externals
                    if len(legs) < 3:
                        return None
                elif legs != externals:
                    return None

        if legs is None:
            return None

        return (legs[0], legs[-1])

    def update_color_basis_ddm(self, colorize_dict, index):
        """Same as update_color_basis, but decomposing the color structures on
        the (n-2)! DDM half-ladder basis instead of the (n-1)! trace one."""

        first, last = self._ddm_ends

        for col_chain, col_str in colorize_dict.items():
            # The decomposition only depends on the tree structure, so
            # normalize the summed indices to make the cache hit as often as
            # possible.
            repl_dict = {}
            for col_obj in col_str:
                for i in col_obj:
                    if i < 0 and i not in repl_dict:
                        repl_dict[i] = -len(repl_dict) - 1
            canonical_str = col_str.create_copy()
            canonical_str.replace_indices(repl_dict)
            canonical_rep = canonical_str.to_immutable()

            try:
                decomposition = self._ddm_dict[canonical_rep]
            except KeyError:
                decomposition = _ColorTree(canonical_str).reduce_to_ddm(first,
                                                                        last)
                self._ddm_dict[canonical_rep] = decomposition

            for perm, coeff in decomposition.items():
                basis_entry = (index,
                               col_chain,
                               col_str.coeff * coeff,
                               col_str.is_imaginary,
                               col_str.Nc_power,
                               col_str.loop_Nc_power)
                immutable_col_str = ddm_half_ladder(perm, first, last)
                try:
                    self[immutable_col_str].append(basis_entry)
                except KeyError:
                    self[immutable_col_str] = [basis_entry]

    def build_flow_basis(self):
        """Build, next to the DDM basis, the trace basis which is the one
        carrying the color flow information. Only the basis is built, not its
        (n-1)!^2 color matrix, since the color sum stays in the DDM basis."""

        flow_basis = ColorBasis()
        flow_basis._list_color_dict = self._list_color_dict
        for index, color_dict in enumerate(self._list_color_dict):
            flow_basis.update_color_basis(color_dict, index)

        self._flow_basis = flow_basis

    def get_flow_basis(self):
        """The color basis carrying the color flow information: the trace basis
        built next to the DDM one, or simply self for a trace basis."""

        return self._flow_basis if self._flow_basis else self

    def get_flow_projection(self):
        """Return the Kleiss-Kuijf relations giving each trace JAMP as a linear
        combination of the DDM ones, i.e. the coefficients of the expansion of
        the half-ladders on the trace basis, transposed. The format is the one
        of get_color_amplitudes, so that the same writers can be used: a list
        (one entry per element of the flow basis) of
        ((1, coefficient, is_imaginary, Nc power), DDM basis index+1)."""

        if not self._flow_basis:
            raise ColorBasis.ColorBasisError(
                              "No flow basis attached to this color basis")

        flow_index = dict((struct, i) for i, struct in \
                          enumerate(sorted(self._flow_basis.keys())))
        projection = [[] for i in range(len(flow_index))]

        for i, struct in enumerate(sorted(self.keys())):
            col_str = color_algebra.ColorString()
            col_str.from_immutable(struct)
            for cs in color_algebra.ColorFactor([col_str]).full_simplify():
                try:
                    row = flow_index[cs.to_immutable()]
                except KeyError:
                    raise ColorBasis.ColorBasisError(
                        "The half-ladder %s expands on the trace structure %s "
                        "which is not part of the flow basis" % \
                        (str(col_str), str(cs)))
                projection[row].append(((1, cs.coeff, cs.is_imaginary,
                                         cs.Nc_power), i + 1))

        return projection

    def update_color_basis(self, colorize_dict, index):
        """Update the current color basis by adding information from
        the colorize dictionary (produced by the colorize routine)
        associated to diagram with index index. Keep track of simplification
        results for maximal optimization."""
        import madgraph.various.misc as misc
        # loop over possible color chains
        for col_chain, col_str in colorize_dict.items():
            # Create a canonical immutable representation of the string
            canonical_rep, rep_dict = col_str.to_canonical()
            try:
                # If this representation has already been considered,
                # recycle the result.                               
                col_fact = self._canonical_dict[canonical_rep].create_copy()
            except KeyError:
                # If the representation is really new

                # Create and simplify a color factor for the considered chain
                col_fact = color_algebra.ColorFactor([col_str])
                col_fact = col_fact.full_simplify()

                # Here we need to force a specific order for the summed indices
                # in case we have K6 or K6bar Clebsch Gordan coefficients
                for colstr in col_fact: colstr.order_summation()

                # Save the result for further use
                canonical_col_fact = col_fact.create_copy()
                canonical_col_fact.replace_indices(rep_dict)
                # Remove overall coefficient
                for cs in canonical_col_fact:
                    cs.coeff = cs.coeff / col_str.coeff
                    # not a property of the canonical structure
                    cs.loop_Nc_power = 0
                self._canonical_dict[canonical_rep] = canonical_col_fact
            else:
                # If this representation has already been considered,
                # adapt the result
                # Note that we have to replace back
                # the indices to match the initial convention. 
                col_fact.replace_indices(self._invert_dict(rep_dict))
                # Since the initial coeff of col_str is not taken into account
                # for matching, we have to multiply col_fact by it.
                for cs in col_fact:
                    cs.coeff = cs.coeff * col_str.coeff
                # Must simplify up to two times at NLO (since up to two traces
                # can appear with a loop) to put traces in a canonical ordering.
                # If it still causes issue, just do a full_simplify(), it would
                # not bring any heavy additional computational load.
                #
                # What is recycled here is an already simplified color factor
                # to which nothing but a relabelling of the indices has been
                # applied. A relabelled simplified expression is still
                # simplified, so this only has to put every color string back
                # in canonical form, which relabel_canonical does directly
                # instead of running the full simplification machinery over
                # every term. The equivalence of the two is checked once per
                # canonical representation, and the slow path is kept for any
                # color structure where it does not hold.
                col_fact = self.relabel_canonical(col_fact, canonical_rep)

                # Here we need to force a specific order for the summed indices
                # in case we have K6 or K6bar Clebsch Gordan coefficients
                for colstr in col_fact: colstr.order_summation()

            # The memo does not carry loop_Nc_power; set it on both paths.
            for cs in col_fact:
                cs.loop_Nc_power = col_str.loop_Nc_power

            # loop over color strings in the resulting color factor
            for col_str in col_fact:
                immutable_col_str = col_str.to_immutable()
                # if the color structure is already present in the present basis
                # update it
                basis_entry = (index,
                                col_chain,
                                col_str.coeff,
                                col_str.is_imaginary,
                                col_str.Nc_power,
                                col_str.loop_Nc_power)
                try:
                    self[immutable_col_str].append(basis_entry)
                except KeyError:
                    self[immutable_col_str] = [basis_entry]

    def OniumColorString(self, pid_charges, pid_numbers, charge, offset=10000):
        """Do the color projection for a given onium"""
        if len(pid_numbers) != 2 or len(pid_charges) != 2:
            raise ColorBasis.ColorBasisError("Only the projection of two particles is supported")
        if pid_charges==(1,-1):
            if charge != 1:
                raise ColorBasis.ColorBasisError("Only color singlet is possible when charges=%d,%d"%pid_charges)
            projector_str = color_algebra.ColorString([color_algebra.ColorOne()])
        elif pid_charges==(3,-3) or pid_charges==(-3,3):
            if pid_charges==(-3,3):
                pidnums=(pid_numbers[1],pid_numbers[0])
            else:
                pidnums=(pid_numbers[0],pid_numbers[1])
            if charge == 1:
                # color singlet
                projector_str = color_algebra.ColorString([color_algebra.T(pidnums[1], pidnums[0])])
            elif charge == 8:
                # color octet
                projector_str = color_algebra.ColorString([color_algebra.T(pidnums[0],pidnums[1],pidnums[0])])
            else:
                raise ColorBasis.ColorBasisError("Only color singlet/octet is possible when charges=%d,%d"%pid_charges)
        else:
            raise ColorBasis.ColorBasisError("Unknown charges=%d,%d for color projection of onium"%pid_charges)

        return projector_str

    def OniaColorProjection(self, colorize_dict, pid_color_numbers):
        """Color projection for onia"""
        for (pid_charges,pid_numbers,charge) in pid_color_numbers:
            OniumCS=self.OniumColorString(pid_charges, pid_numbers, charge)
            for col_str in colorize_dict.values():
                col_str.product(OniumCS)

    def create_color_dict_list(self, amplitude):
        """Returns a list of colorize dict for all diagrams in amplitude. Also
        update the _list_color_dict object accordingly """

        list_color_dict = []

        for diagram in amplitude.get('diagrams'):
            colorize_dict = self.colorize(diagram,
                                          amplitude.get('process').get('model'))

            onium = False
            pid_color_numbers = []
            for l in amplitude.get('process').get('legs'):
                if l.get('onium'):
                    if not onium:
                        onium = True
                        for part in amplitude.get('process').get('model').get('particles'):
                            if part.get('pdg_code') == l.get('id'): color = part.get('color')
                            if part.get('pdg_code') == -l.get('id'): color = -part.get('color')
                        onium_color = color
                        onium_number = l.get('number')
                    else:
                        onium = False
                        for part in amplitude.get('process').get('model').get('particles'):
                            if part.get('pdg_code') == l.get('id'): color = part.get('color')
                            if part.get('pdg_code') == -l.get('id'): color = -part.get('color')
                        pid_color_numbers.append([(onium_color,color),(onium_number,l.get('number')),l.get('onium').get('C')])

            if pid_color_numbers: self.OniaColorProjection(colorize_dict, pid_color_numbers)

            list_color_dict.append(colorize_dict)

        self._list_color_dict = list_color_dict

        return list_color_dict

    def build(self, amplitude=None):
        """Build the a color basis object using information contained in
        amplitude (otherwise use info from _list_color_dict).
        Returns a list of color """

        if amplitude:
            self.create_color_dict_list(amplitude)

        if ddm_basis:
            self._ddm_ends = self.get_ddm_ends()
        if self._ddm_ends:
            try:
                for index, color_dict in enumerate(self._list_color_dict):
                    self.update_color_basis_ddm(color_dict, index)
                if ddm_flow_basis:
                    self.build_flow_basis()
                return
            except DDMError as error:
                logger.debug('Falling back on the trace color basis: %s', error)
                self.clear()
                self._ddm_ends = None
                self._flow_basis = None

        for index, color_dict in enumerate(self._list_color_dict):
            self.update_color_basis(color_dict, index)

    def __init__(self, *args):
        """Initialize a new color basis object, either empty or filled (0
        or 1 arguments). If one arguments is given, it's interpreted as
        an amplitude."""

        assert len(args) < 2, "Object ColorBasis must be initialized with 0 or 1 arguments"


        dict.__init__(self)

        # Dictionary to save simplifications already done in a canonical form
        self._canonical_dict = {}

        # Dictionary store the raw colorize information
        self._list_color_dict = []

        # Whether relabel_canonical may take its shortcut, per canonical form
        self._fast_relabel_dict = {}

        # Legs at the two ends of the DDM half-ladders, None when the basis is
        # the standard trace one
        self._ddm_ends = None

        # Trace basis built next to a DDM one, carrying the color flows
        self._flow_basis = None

        # Dictionary to save the DDM decompositions already done
        self._ddm_dict = {}


        if args:
            assert isinstance(args[0], diagram_generation.Amplitude), \
                        "%s is not a valid Amplitude object" % str(args[0])
                        
            self.build(*args)

    def __str__(self):
        """Returns a nicely formatted string for display"""

        my_str = ""
        for k, v in self.items():
            for name, indices in k:
                my_str = my_str + name + str(indices)
            my_str = my_str + ': '
            for contrib in v:
                imag_str = ''
                if contrib[3]:
                    imag_str = 'I'
                my_str = my_str + '(diag:%i, chain:%s, coeff:%s%s, Nc:%i) ' % \
                                    (contrib[0], contrib[1], contrib[2],
                                     imag_str, contrib[4])
            my_str = my_str + '\n'
        return my_str

    def _invert_dict(self, mydict):
        """Helper method to invert dictionary dict"""

        return dict([v, k] for k, v in mydict.items())

    @staticmethod
    def get_color_flow_string(my_color_string, octet_indices):
        """Return the color_flow_string (i.e., composed only of T's with 2 
        indices) associated to my_color_string. Take a list of the external leg
        color octet state indices as an input. Returns only the leading N 
        contribution!"""
        # Create a new color factor to allow for simplification
        my_cf = color_algebra.ColorFactor([my_color_string])

        # Add one T per external octet
        for indices in octet_indices:
            if indices[0] == -6:
                # Add a K6 which contracts the antisextet index to a
                # pair of antitriplets
                my_cf[0].append(color_algebra.K6(indices[1],
                                                 indices[2],
                                                 indices[3]))
            if indices[0] == 6:
                # Add a K6Bar which contracts the sextet index to a
                # pair of triplets
                my_cf[0].append(color_algebra.K6Bar(indices[1],
                                                    indices[2],
                                                    indices[3]))
            if abs(indices[0]) == 8:
                # Add a T which contracts the octet to a
                # triplet-antitriplet pair
                my_cf[0].append(color_algebra.T(indices[1],
                                                indices[2],
                                                indices[3]))
        # Simplify the whole thing
        with misc.TMP_variable(color_algebra.Epsilon, 'rule_eps_aeps_nosum', False):
            my_cf = my_cf.full_simplify()

        # If the result is empty, just return
        if not my_cf:
            return my_cf

        # Return the string with the highest N coefficient 
        # (leading N decomposition), and the value of this coeff
        max_coeff = max([cs.Nc_power for cs in my_cf])

        res_cs = [cs for cs in my_cf if cs.Nc_power == max_coeff]

        # If more than one string at leading N...
        if len(res_cs) > 1 and any([not cs.near_equivalent(res_cs[0]) \
                                    for cs in res_cs]):
            raise ColorBasis.ColorBasisError("More than one color string with leading N coeff: %s" % str(res_cs))

        res_cs = res_cs[0]

        # If the result string does not contain only T's with two indices
        # and Epsilon/EpsilonBar objects
        for col_obj in res_cs:
            if not isinstance(col_obj, color_algebra.T) and \
                   not col_obj.__class__.__name__.startswith('Epsilon'):
                raise ColorBasis.ColorBasisError("Color flow decomposition %s contains non T/Epsilon elements" % \
                                                                    str(res_cs))
            if isinstance(col_obj, color_algebra.T) and len(col_obj) != 2:
                raise ColorBasis.ColorBasisError("Color flow decomposition %s contains T's w/o 2 indices" % \
                                                                    str(res_cs))

        return res_cs

    def color_flow_decomposition(self, repr_dict, ninitial):
        """Returns the color flow decomposition of the current basis, i.e. a 
        list of dictionaries (one per color basis entry) with keys corresponding
        to external leg numbers and values tuples containing two color indices
        ( (0,0) for singlets, (X,0) for triplet, (0,X) for antitriplet and 
        (X,Y) for octets). Other color representations are not yet supported 
        here (an error is raised). Needs a dictionary with keys being external
        leg numbers, and value the corresponding color representation."""

        if self._ddm_ends:
            raise ColorBasis.ColorBasisError(
                "A DDM color basis has no single color flow per basis element."
                " Use 'set color_basis trace' for this output format.")

        # Offsets used to introduce fake quark indices for gluons
        offset1 = 1000
        offset2 = 2000
        offset3 = 3000

        res = []

        for col_basis_entry in sorted(self.keys()):

            res_dict = {}
            fake_repl = []

            # Rebuild a color string from a CB entry
            col_str = color_algebra.ColorString()
            col_str.from_immutable(col_basis_entry)
            for (leg_num, leg_repr) in repr_dict.items():
                # By default, assign a (0,0) color flow
                res_dict[leg_num] = [0, 0]
                # Raise an error if external legs contain non supported repr
                if abs(leg_repr) not in [1, 3, 6, 8]:
                    raise ColorBasis.ColorBasisError("Particle ID=%i has an unsupported color representation" % leg_repr)

                # Build the fake indices replacements for octets
                if abs(leg_repr) == 8:
                    fake_repl.append((leg_repr, leg_num,
                                      offset1 + leg_num,
                                      offset2 + leg_num))
                # Build the fake indices for sextets
                elif leg_repr in [-6, 6]:
                    fake_repl.append((leg_repr, leg_num,
                                      offset1 + leg_num,
                                      offset3 + leg_num))

            # Get the actual color flow
            col_str_flow = self.get_color_flow_string(col_str, fake_repl)

            # Offset for color flow
            offset = 500

            for col_obj in col_str_flow:
                if isinstance(col_obj, color_algebra.T):
                    # For T, all color indices should be the same
                    offset = offset + 1
                for i, index in enumerate(col_obj):
                    if isinstance(col_obj, color_algebra.Epsilon):
                        # Epsilon contracts with antitriplets,
                        i = 0
                        # ...and requires all different color indices
                        offset = offset+1
                    elif isinstance(col_obj, color_algebra.EpsilonBar):
                        # EpsilonBar contracts with antitriplets
                        i = 1
                        # ...and requires all different color indices
                        offset = offset+1
                    if index < offset1:
                        res_dict[index][i] = offset
                    elif index > offset1 and index < offset2:
                        res_dict[index - offset1][i] = offset
                    elif index > offset2 and index < offset3:
                        res_dict[index - offset2][i] = offset
                    elif index > offset3:
                        # For color sextets, use negative triplet
                        # number to represent antitriplet and vice
                        # versa, allowing for two triplet or two
                        # antitriplet numbers representing the color
                        # sextet.
                        res_dict[index - offset3][1-i] = -offset

            # Reverse ordering for initial state to stick to the (weird)
            # les houches convention

            for key in res_dict.keys():
                if key <= ninitial:
                    res_dict[key].reverse()

            res.append(res_dict)

        return res


#===============================================================================
# Permutation symmetry of a color basis
#===============================================================================
def permute_immutable(struct, perm):
    """Apply the index permutation perm (a dict {old_index: new_index}) to the
    immutable representation of a color structure, and bring the result back to
    the canonical form used as a ColorBasis key: traces are cyclic, so they are
    rotated to start on their smallest index, and the color objects are sorted
    exactly as ColorString.to_immutable does."""

    res = []
    for name, indices in struct:
        new_indices = tuple([perm.get(i, i) for i in indices])
        if name == 'Tr' and len(new_indices) > 1:
            # Tr is cyclic: rotate so that the smallest index comes first
            start = min(range(len(new_indices)), key=new_indices.__getitem__)
            new_indices = new_indices[start:] + new_indices[:start]
        res.append((name, new_indices))
    res.sort()
    return tuple(res)


class ColorBasisSymmetry(object):
    """Permutations of the external color indices which map a color basis (or a
    pair of color bases, for an asymmetric color matrix) onto itself.

    A color matrix entry is the full contraction of two color structures, so it
    depends only on the *relative* labelling of the indices: relabelling the
    indices consistently in both structures leaves the entry unchanged. Hence
    for any such permutation P,

        C[P(i)][P(j)] = C[i][j]

    and only one row per orbit of P-action on the basis has to be computed.

    Note that the permutations found here are not required to be physical
    permutations of identical particles: any index relabelling that maps the
    basis onto itself is a symmetry of the color matrix. For g g > n g this
    finds the full S_(n+2) rather than only the S_n of the final state, which
    collapses the whole matrix to a single row."""

    # Indices above this value are summed indices introduced internally
    # (order_summation starts at 10000, colorize uses values below -1000);
    # only genuine external indices are permuted.
    max_external_index = 1000

    def __init__(self, keys1, keys2=None):
        """keys1/keys2 are the *sorted* lists of color basis keys, i.e. exactly
        the ordering used to index the color matrix."""

        self.keys1 = keys1
        self.keys2 = keys2 if keys2 is not None else keys1
        # permutation of the basis indices induced by each accepted generator
        self.generators1 = []
        self.generators2 = []
        # representative of the orbit each row belongs to, and how to get there
        # in one step: (parent row, index of the generator mapping it to here)
        self.row_rep = list(range(len(keys1)))
        self.row_parent = [None] * len(keys1)
        self.representatives = list(range(len(keys1)))

        if not keys1 or not self.keys2:
            return

        self._find_generators()
        self._build_orbits()

    def _external_indices(self, keys):
        """Return the sorted list of indices which may be permuted. A plain
        list is used rather than a set since 'set' is shadowed by an ordered
        variant in this module when reproducible ordering is requested."""

        indices = {}
        for struct in keys:
            for _, idx in struct:
                for i in idx:
                    if 0 < i < self.max_external_index:
                        indices[i] = True
        return sorted(indices)

    def _index_signature(self, keys):
        """Group indices by the way they appear in the basis: two indices can
        only be exchanged if they occupy the same kind of slots. This is only
        used to avoid testing hopeless candidates; every candidate is verified
        explicitly afterwards."""

        sig = collections.defaultdict(collections.Counter)
        for struct in keys:
            for name, idx in struct:
                for pos, i in enumerate(idx):
                    sig[i][(name, len(idx), pos)] += 1
        return sig

    def _find_generators(self):
        """Find transpositions of external indices mapping every basis onto
        itself, and store the induced permutation of the basis indices."""

        candidates = self._external_indices(self.keys1)
        if self.keys2 is not self.keys1:
            other = dict((i, True) for i in self._external_indices(self.keys2))
            candidates = [i for i in candidates if i in other]
        if len(candidates) < 2:
            return

        sig1 = self._index_signature(self.keys1)
        sig2 = self._index_signature(self.keys2) \
                                if self.keys2 is not self.keys1 else sig1

        pos1 = dict((k, i) for i, k in enumerate(self.keys1))
        pos2 = pos1 if self.keys2 is self.keys1 else \
                             dict((k, i) for i, k in enumerate(self.keys2))

        for a, b in itertools.combinations(candidates, 2):
            if sig1[a] != sig1[b] or sig2[a] != sig2[b]:
                continue
            perm = {a: b, b: a}
            induced1 = self._induced_permutation(self.keys1, pos1, perm)
            if induced1 is None:
                continue
            if self.keys2 is self.keys1:
                induced2 = induced1
            else:
                induced2 = self._induced_permutation(self.keys2, pos2, perm)
                if induced2 is None:
                    continue
            # A transposition is its own inverse, and so is the permutation it
            # induces on the basis. Rows are gathered from their parent with
            # the generator itself rather than with its inverse, so make sure
            # of it instead of assuming it.
            if any(induced1[induced1[i]] != i for i in range(len(induced1))) or \
               any(induced2[induced2[i]] != i for i in range(len(induced2))):
                continue
            self.generators1.append(induced1)
            self.generators2.append(induced2)

    @staticmethod
    def _induced_permutation(keys, positions, perm):
        """Return the permutation of the basis indices induced by the index
        permutation perm, or None if the basis is not mapped onto itself."""

        induced = [0] * len(keys)
        seen = [False] * len(keys)
        for i, struct in enumerate(keys):
            try:
                j = positions[permute_immutable(struct, perm)]
            except KeyError:
                return None
            if seen[j]:
                return None
            seen[j] = True
            induced[i] = j
        return induced

    def _build_orbits(self):
        """Breadth-first exploration of each orbit, recording for every row the
        representative it comes from and the generator that reaches it from its
        parent, so that the row can be obtained by a single gather."""

        if not self.generators1:
            return

        n = len(self.keys1)
        self.row_rep = [-1] * n
        self.representatives = []
        for start in range(n):
            if self.row_rep[start] != -1:
                continue
            self.representatives.append(start)
            self.row_rep[start] = start
            self.row_parent[start] = None
            queue = collections.deque([start])
            while queue:
                current = queue.popleft()
                for gen_index, induced in enumerate(self.generators1):
                    image = induced[current]
                    if self.row_rep[image] == -1:
                        self.row_rep[image] = start
                        self.row_parent[image] = (current, gen_index)
                        queue.append(image)

    def has_symmetry(self):
        """True if the symmetry actually reduces the number of rows."""

        return bool(self.generators1) and \
                              len(self.representatives) < len(self.keys1)

    def spanning_generators(self):
        """Indices of a subset of the generators which still reaches every
        line of every orbit. Anything which writes the permutations out has to
        store one array of basis indices per generator, so dropping those which
        connect nothing new makes a large difference."""

        parent = list(range(len(self.keys1)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        kept = []
        for index, induced in enumerate(self.generators1):
            used = False
            for i, j in enumerate(induced):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    used = True
            if used:
                kept.append(index)
        return kept

    def spanning_tree(self, gen_indices=None):
        """Describe every line as one generator applied to another line:
        returns the orbit representatives, the representative of each line, the
        (parent line, position in gen_indices) pair reaching each line, and the
        generators actually used. Following the parents back to the
        representative gives the permutation relating the two lines."""

        if gen_indices is None:
            gen_indices = self.spanning_generators()
        gens = [self.generators1[i] for i in gen_indices]

        n = len(self.keys1)
        representative = [-1] * n
        parent = [None] * n
        representatives = []
        for start in range(n):
            if representative[start] != -1:
                continue
            representatives.append(start)
            representative[start] = start
            queue = collections.deque([start])
            while queue:
                current = queue.popleft()
                for local, induced in enumerate(gens):
                    image = induced[current]
                    if representative[image] == -1:
                        representative[image] = start
                        parent[image] = (current, local)
                        queue.append(image)
        return representatives, representative, parent, gens


#===============================================================================
# ColorMatrix
#===============================================================================
class _ColorMatrixView(object):
    """Read-only mapping presenting one of the two representations stored by a
    ColorMatrix (the ColorFactor one, or the fixed Nc one) as the dictionary
    keyed by (i1, i2) that it used to be."""

    def __init__(self, matrix, entry):
        self._matrix = matrix
        self._entry = entry

    def __getitem__(self, key):
        return self._matrix._get_entry(key)[self._entry]

    def __len__(self):
        return len(self._matrix)

    def __contains__(self, key):
        try:
            self[key]
        except (KeyError, IndexError, TypeError):
            return False
        return True

    def __iter__(self):
        return iter(self._matrix)

    def keys(self):
        return list(self)

    def values(self):
        return [self[key] for key in self]

    def items(self):
        return [(key, self[key]) for key in self]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def __eq__(self, other):
        if isinstance(other, _ColorMatrixView):
            if len(self) != len(other):
                return False
            return all(self[key] == other[key] for key in self)
        if isinstance(other, dict):
            return dict(self.items()) == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result


class ColorMatrix(dict):
    """A color matrix, meaning a mapping with pairs (i,j) as keys where i
    and j refer to elements of color basis objects. Values are Color Factor
    objects. The fixed Nc representation is available through the
    col_matrix_fixed_Nc attribute.

    The matrix is not stored entry by entry. A color matrix entry is the full
    contraction of two color structures and therefore only depends on the
    relative labelling of the color indices, so entries repeat massively: the
    distinct values are stored once and the (i,j) grid only keeps an index into
    them. On top of that, index permutations mapping the color basis onto
    itself (see ColorBasisSymmetry) relate whole rows to each other, so only
    one row per orbit is actually computed; the others are a gather away."""

    _col_basis1 = None
    _col_basis2 = None
    col_matrix_fixed_Nc = {}
    _ddm_expansions = None

    def __init__(self, col_basis, col_basis2=None,
                 Nc=3, Nc_power_min=None, Nc_power_max=None):
        """Initialize a color matrix with one or two color basis objects. If
        only one color basis is given, the other one is assumed to be equal.
        As options, any value of Nc and minimal/maximal power of Nc can also be
        provided. Note that the min/max power constraint is applied
        only at the end, so that it does NOT speed up the calculation."""

        # Distinct entries, as (result, result_fixed_Nc) pairs, and the (i1,i2)
        # grid of indices into that list, stored row-major in a compact array.
        self._values = []
        self._val_index = array.array('i')
        self._sorted_keys1 = []
        self._sorted_keys2 = []
        # Set by setup_ddm_entries for a DDM (half-ladder) color basis
        self._ddm_expansions = None
        self.col_matrix_fixed_Nc = _ColorMatrixView(self, 1)

        self._col_basis1 = col_basis
        if col_basis2:
            self._col_basis2 = col_basis2
            self.build_matrix(Nc, Nc_power_min, Nc_power_max)
        else:
            self._col_basis2 = col_basis
            # If the two color basis are equal, assumes the color matrix is
            # symmetric
            self.build_matrix(Nc, Nc_power_min, Nc_power_max, is_symmetric=True)

    #===========================================================================
    # Mapping interface
    #===========================================================================
    def _get_entry(self, key):
        """Return the (result, result_fixed_Nc) pair for the (i1, i2) key."""

        i1, i2 = key
        n1, n2 = len(self._sorted_keys1), len(self._sorted_keys2)
        if not 0 <= i1 < n1 or not 0 <= i2 < n2:
            raise KeyError(key)
        return self._values[self._val_index[i1 * n2 + i2]]

    def __getitem__(self, key):
        return self._get_entry(key)[0]

    def __len__(self):
        return len(self._sorted_keys1) * len(self._sorted_keys2)

    def __bool__(self):
        return bool(self._sorted_keys1) and bool(self._sorted_keys2)

    __nonzero__ = __bool__

    def __contains__(self, key):
        try:
            self._get_entry(key)
        except (KeyError, IndexError, TypeError):
            return False
        return True

    def __iter__(self):
        for i1 in range(len(self._sorted_keys1)):
            for i2 in range(len(self._sorted_keys2)):
                yield (i1, i2)

    def keys(self):
        return list(self)

    def values(self):
        return [self[key] for key in self]

    def items(self):
        return [(key, self[key]) for key in self]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def __eq__(self, other):
        if isinstance(other, ColorMatrix):
            if self._sorted_keys1 != other._sorted_keys1 or \
                                self._sorted_keys2 != other._sorted_keys2:
                return False
            return all(self._get_entry(key) == other._get_entry(key)
                       for key in self)
        if isinstance(other, dict):
            return dict(self.items()) == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None

    @property
    def inverted_col_matrix(self):
        """Dictionary mapping each fixed Nc value to the list of (i1,i2) it
        appears at. Kept for backward compatibility, built on demand."""

        inverted = {}
        for key in self:
            inverted.setdefault(self._get_entry(key)[1], []).append(key)
        return inverted

    #===========================================================================
    # Construction
    #===========================================================================
    def _value_index(self, struct1, struct2, canonical_dict,
                     Nc, Nc_power_min, Nc_power_max):
        """Return the index in self._values of the entry for the two given
        color structures, computing it if it is seen for the first time."""

        # Fix indices in struct2 knowing summed indices in struct1
        # to avoid duplicates
        new_struct2 = self.fix_summed_indices(struct1, struct2)

        # Build a canonical representation of the two immutable struct
        canonical_entry, dummy = \
                    color_algebra.ColorString().to_canonical(struct1 + \
                                                            new_struct2)

        try:
            # If this has already been calculated, use the result
            return canonical_dict[canonical_entry]
        except KeyError:
            pass

        # Otherwise calculate the result
        result, result_fixed_Nc = self.create_new_entry(struct1,
                                                        new_struct2,
                                                        Nc_power_min,
                                                        Nc_power_max,
                                                        Nc)
        index = len(self._values)
        self._values.append((result, result_fixed_Nc))
        canonical_dict[canonical_entry] = index
        return index

    def build_matrix(self, Nc=3,
                     Nc_power_min=None,
                     Nc_power_max=None,
                     is_symmetric=False):
        """Create the matrix using internal color basis objects. Use the stored
        color basis objects and takes Nc and Nc_min/max parameters as __init__.
        If is_symmetric is True, the matrix is assumed to be symmetric so that
        only half of it needs to be computed."""

        self._sorted_keys1 = sorted(self._col_basis1.keys())
        if self._col_basis2 is self._col_basis1:
            self._sorted_keys2 = self._sorted_keys1
        else:
            self._sorted_keys2 = sorted(self._col_basis2.keys())

        keys1, keys2 = self._sorted_keys1, self._sorted_keys2
        n1, n2 = len(keys1), len(keys2)
        self._values = []
        self._val_index = array.array('i', [0]) * (n1 * n2) if n1 * n2 else \
                          array.array('i')
        if not n1 or not n2:
            return

        if getattr(self._col_basis1, '_ddm_ends', None) and \
           getattr(self._col_basis2, '_ddm_ends', None):
            self.setup_ddm_entries()

        canonical_dict = {}
        symmetry = ColorBasisSymmetry(keys1,
                            None if keys2 is keys1 else keys2)

        if not symmetry.has_symmetry():
            # No index permutation maps the basis onto itself: fall back to the
            # plain scan, using the symmetry of the matrix itself if available.
            for i1, struct1 in enumerate(keys1):
                for i2, struct2 in enumerate(keys2):
                    if is_symmetric and i2 < i1:
                        continue
                    index = self._value_index(struct1, struct2, canonical_dict,
                                              Nc, Nc_power_min, Nc_power_max)
                    self._val_index[i1 * n2 + i2] = index
                    if is_symmetric:
                        self._val_index[i2 * n2 + i1] = index
            return

        # One row per orbit is computed explicitly; every other row is the
        # image of an already known one under a single generator.
        done = [False] * n1
        for rep in symmetry.representatives:
            struct1 = keys1[rep]
            offset = rep * n2
            for i2, struct2 in enumerate(keys2):
                self._val_index[offset + i2] = \
                        self._value_index(struct1, struct2, canonical_dict,
                                          Nc, Nc_power_min, Nc_power_max)
            done[rep] = True

        # Breadth-first replay of the orbit exploration: a row whose parent is
        # already filled is obtained by permuting the parent's columns.
        remaining = [i for i in range(n1) if not done[i]]
        while remaining:
            progressed = False
            still_missing = []
            for row in remaining:
                parent, gen_index = symmetry.row_parent[row]
                if not done[parent]:
                    still_missing.append(row)
                    continue
                induced2 = symmetry.generators2[gen_index]
                src = parent * n2
                dest = row * n2
                val_index = self._val_index
                for i2 in range(n2):
                    val_index[dest + i2] = val_index[src + induced2[i2]]
                done[row] = True
                progressed = True
            assert progressed, "Color matrix orbit exploration made no progress"
            remaining = still_missing

    def setup_ddm_entries(self):
        """Switch create_new_entry over to the assembly used for a DDM
        (half-ladder) color basis.

        Contracting two half-ladders head on is exponentially expensive, since
        the simplification rules turn every one of the 2(n-2) f objects into a
        pair of traces. Each ladder is instead expanded once on the trace basis
        (2^(n-2) traces) and the entry is assembled from trace-trace products,
        which are recycled between all the entries. Everything else, including
        the orbit symmetry of the basis, is left to build_matrix."""

        self._ddm_expansions = {}
        self._ddm_half_dict = {}
        self._ddm_trace_dict = {}

    def get_ddm_trace_expansion(self, struct):
        """Expansion of one half-ladder on the trace basis, as a list of
        (immutable trace, coefficient, is_imaginary, Nc power)."""

        try:
            return self._ddm_expansions[struct]
        except KeyError:
            pass

        col_str = color_algebra.ColorString()
        col_str.from_immutable(struct)
        expansion = [(cs.to_immutable(), cs.coeff, cs.is_imaginary,
                      cs.Nc_power) for cs in \
                     color_algebra.ColorFactor([col_str]).full_simplify()]

        self._ddm_expansions[struct] = expansion
        return expansion

    def create_new_entry_ddm(self, struct1, struct2,
                             Nc_power_min, Nc_power_max, Nc):
        """create_new_entry for two half-ladders, through their trace
        expansions."""

        contraction = collections.defaultdict(fractions.Fraction)
        for trace, coeff, is_imaginary, Nc_power in \
                                     self.get_ddm_trace_expansion(struct1):
            self.accumulate_number(contraction,
                                   (coeff, is_imaginary, Nc_power),
                                   self.get_half_ladder_contraction(trace,
                                                                    struct2))

        result = color_algebra.ColorFactor()
        for (is_imaginary, Nc_power), coeff in contraction.items():
            if not coeff:
                continue
            if Nc_power_min is not None and Nc_power < Nc_power_min:
                continue
            if Nc_power_max is not None and Nc_power > Nc_power_max:
                continue
            result.append(color_algebra.ColorString([], coeff, is_imaginary,
                                                    Nc_power))

        return result, result.set_Nc(Nc)

    @staticmethod
    def accumulate_number(target, factor, numbers):
        """Add factor*numbers to target, where a number is a dictionary
        {(is_imaginary, Nc power): coefficient} and factor a single
        (coefficient, is_imaginary, Nc power) triplet."""

        coeff, is_imaginary, Nc_power = factor
        for (other_imaginary, other_power), other_coeff in numbers.items():
            new_coeff = coeff * other_coeff
            if is_imaginary and other_imaginary:
                new_coeff = -new_coeff
                new_imaginary = False
            else:
                new_imaginary = is_imaginary or other_imaginary
            target[(new_imaginary, Nc_power + other_power)] += new_coeff

    def get_half_ladder_contraction(self, trace, struct2):
        """Contraction of the single trace \'trace\' with the complex conjugate
        of the half-ladder \'struct2\'."""

        canonical_rep, dummy = \
            color_algebra.ColorString().to_canonical(trace + struct2)
        try:
            return self._ddm_half_dict[canonical_rep]
        except KeyError:
            pass

        result = collections.defaultdict(fractions.Fraction)
        for trace2, coeff, is_imaginary, Nc_power in \
                                      self.get_ddm_trace_expansion(struct2):
            # complex conjugation of the coefficient of the second ladder
            if is_imaginary:
                coeff = -coeff
            self.accumulate_number(result, (coeff, is_imaginary, Nc_power),
                                   self.get_trace_contraction(trace, trace2))

        self._ddm_half_dict[canonical_rep] = result
        return result

    def get_trace_contraction(self, trace1, trace2):
        """Contraction of two single traces, as a dictionary
        {(is_imaginary, Nc power): coefficient}."""

        canonical_rep, dummy = \
            color_algebra.ColorString().to_canonical(trace1 + trace2)
        try:
            return self._ddm_trace_dict[canonical_rep]
        except KeyError:
            pass

        col_str = color_algebra.ColorString()
        col_str.from_immutable(trace1)
        col_str2 = color_algebra.ColorString()
        col_str2.from_immutable(trace2)
        col_str.product(col_str2.complex_conjugate())

        result = collections.defaultdict(fractions.Fraction)
        for cs in color_algebra.ColorFactor([col_str]).full_simplify():
            assert not len(cs), \
                "Trace contraction %s did not simplify to a number" % str(cs)
            result[(cs.is_imaginary, cs.Nc_power)] += cs.coeff

        self._ddm_trace_dict[canonical_rep] = result
        return result


    def create_new_entry(self, struct1, struct2,
                         Nc_power_min, Nc_power_max, Nc):
        """ Create a new product result, and result with fixed Nc for two color
        basis entries. Implement Nc power limits."""

        if self._ddm_expansions is not None:
            return self.create_new_entry_ddm(struct1, struct2,
                                             Nc_power_min, Nc_power_max, Nc)

        # Create color string objects corresponding to color basis
        # keys
        col_str = color_algebra.ColorString()
        col_str.from_immutable(struct1)

        col_str2 = color_algebra.ColorString()
        col_str2.from_immutable(struct2)

        # Complex conjugate the second one and multiply the two
        col_str.product(col_str2.complex_conjugate())
        if __debug__:
            #check that no index is repeating more than twice
            nb_indices = collections.defaultdict(int)
            for col_obj in col_str:
                for index in col_obj[:]:
                    nb_indices[index] += 1
            assert all([nb <= 2 for nb in nb_indices.values()]), \
                        "Color string %s has indices appearing more than twice: %s" % \
                        (str(col_str), nb_indices)


        # Create a color factor to store the result and simplify it
        # taking into account the limit on Nc
        col_fact = color_algebra.ColorFactor([col_str])
        result = col_fact.full_simplify()

        # Keep only terms with Nc_max >= Nc power >= Nc_min
        if Nc_power_min is not None:
            result[:] = [col_str for col_str in result \
                         if col_str.Nc_power >= Nc_power_min]
        if Nc_power_max is not None:
            result[:] = [col_str for col_str in result \
                         if col_str.Nc_power <= Nc_power_max]

        # Calculate the fixed Nc representation
        result_fixed_Nc = result.set_Nc(Nc)

        return result, result_fixed_Nc

    def __str__(self):
        """Returns a nicely formatted string with the fixed Nc representation
        of the current matrix (only the real part)"""

        mystr = '\n\t' + '\t'.join([str(i) for i in \
                                    range(len(self._col_basis2))])

        for i1 in range(len(self._col_basis1)):
            mystr = mystr + '\n' + str(i1) + '\t'
            mystr = mystr + '\t'.join(['%i/%i' % \
                        (self.col_matrix_fixed_Nc[(i1, i2)][0].numerator,
                        self.col_matrix_fixed_Nc[(i1, i2)][0].denominator) \
                        for i2 in range(len(self._col_basis2))])

        return mystr

    def _fixed_Nc_row(self, line_index):
        """Return the fixed Nc entries of one line of the matrix."""

        n2 = len(self._sorted_keys2)
        offset = line_index * n2
        values = self._values
        val_index = self._val_index
        return [values[val_index[offset + i2]][1] for i2 in range(n2)]

    def get_line_denominators(self):
        """Get a list with the denominators for the different lines in
        the color matrix"""

        den_list = []
        for i1 in range(len(self._sorted_keys1)):
            den_list.append(self.lcmm(*[entry[0].denominator for entry in \
                                        self._fixed_Nc_row(i1)]))

        return den_list

    def get_line_numerators(self, line_index, den):
        """Returns a list of numerator for line line_index, assuming a common
        denominator den."""

        return [entry[0].numerator * den / entry[0].denominator \
                for entry in self._fixed_Nc_row(line_index)]

    @classmethod
    def fix_summed_indices(self, struct1, struct2):
        """Returns a copy of the immutable Color String representation struct2
        where summed indices are modified to avoid duplicates with those
        appearing in struct1. Assumes internal summed indices are negative."""

        # First, determines what is the smallest index appearing in struct1
        list1 = sum((list(elem[1]) for elem in struct1),[])
        list2 = sum((list(elem[1]) for elem in struct2),[])
        if not list1:
            min_index = -1
        else:
           min_index = min(list1) - 1

        # Second, determines the summed indices in struct2 and create a
        # replacement dictionary
        repl_dict = {}
        #list2 = reduce(operator.add,
        #               [list(elem[1]) for elem in struct1])
        for summed_index in misc.make_unique([i for i in list2 \
                                      if list2.count(i) == 2]):
            repl_dict[summed_index] = min_index
            min_index -= 1

        # Three, create a new immutable struct by doing replacements in struct2
        return_list = []
        for elem in struct2:
            fix_elem = [elem[0], []]
            for index in elem[1]:
                try:
                    fix_elem[1].append(repl_dict[index])
                except Exception:
                    fix_elem[1].append(index)
            return_list.append((elem[0], tuple(fix_elem[1])))

        return tuple(return_list)

    @staticmethod
    def lcm(a, b):
        """Return lowest common multiple."""
        return a * b // math.gcd(a, b)
        
    @staticmethod
    def lcmm(*args):
        """Return lcm of args."""
        if args:
            return reduce(ColorMatrix.lcm, args)
        else:
            return 1

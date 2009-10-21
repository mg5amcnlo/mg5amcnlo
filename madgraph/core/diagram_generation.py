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

import copy
import itertools

import madgraph.core.base_objects as base_objects

"""Amplitude object, which is what does the job for the diagram
generation algorithm.
"""

#===============================================================================
# Amplitude
#===============================================================================
class Amplitude(base_objects.PhysicsObject):
    """Amplitude: process + list of diagrams (ordered)
    Initialize with a process, then call generate_diagrams() to
    generate the diagrams for the amplitude
    """

    def default_setup(self):
        """Default values for all properties"""

        self['process'] = base_objects.Process()
        self['diagrams'] = base_objects.DiagramList()

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'process':
            if not isinstance(value, base_objects.Process):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Process object" % str(value)
        if name == 'diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList object" % str(value)
        return True

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['process', 'diagrams']

    def generate_diagrams(self):
        """Generate diagrams. For algorithm explanation, see documentation.
        """
        model = self['process'].get('model')

        for leg in self['process'].get('legs'):
            # For the first step, ensure the tag from_group 
            # is true for all legs
            leg.set('from_group', True)

            # Need to flip part-antipart for incoming particles, 
            # so they are all outgoing
            if leg.get('state') == 'initial':
                part = model.get('particle_dict')[leg.get('id')]
                leg.set('id', part.get_anti_pdg_code())

        # Calculate the maximal multiplicity of n-1>1 configurations
        # to restrict possible leg combinations
        max_multi_to1 = max([len(key) for key in \
                             model.get('ref_dict_to1').keys()])


        # Reduce the leg list and return the corresponding
        # list of vertices
        return self.reduce_leglist(self['process'].get('legs'),
                                               max_multi_to1)

    def reduce_leglist(self, curr_leglist, max_multi_to1):
        """Recursive function to reduce N LegList to N-1
        """

        # Result variable which is a list of lists of vertices
        # to be added
        res = []

        # Stop condition. If LegList is None, that means that this
        # diagram must be discarded
        if curr_leglist is None:
            return None

        # Extract ref dict information
        ref_dict_to0 = self['process'].get('model').get('ref_dict_to0')
        ref_dict_to1 = self['process'].get('model').get('ref_dict_to1')

        # If all legs can be combined in one single vertex, add this
        # vertex to res and continue
        if curr_leglist.can_combine_to_0(ref_dict_to0):
            # Extract the interaction id associated to the vertex 
            vertex_id = ref_dict_to0[tuple([leg.get('id') for \
                                                         leg in curr_leglist])]

            final_vertex = base_objects.Vertex({'legs':curr_leglist,
                                                'id':vertex_id})
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

        for leg_vertex_tuple in leg_vertex_list:
            # This is where recursion happens
            reduced_diagram = self.reduce_leglist(leg_vertex_tuple[0],
                                                  max_multi_to1)
            if reduced_diagram:
                vertex_list = list(leg_vertex_tuple[1])
                vertex_list.append(reduced_diagram)
                res.extend(self.expand_list(vertex_list))

        return res


    def combine_legs(self, list_legs, ref_dict_to1, max_multi_to1):
        """Take a list of legs as an input, with the reference dictionary n-1>1,
        and output a list of list of tuples of Legs (allowed combinations) 
        and Legs (rest). For algorithm, see documentation.
        """

        res = []

        # loop over possible combination lengths (+1 is for range convention!)
        for comb_length in range(2, max_multi_to1 + 1):

            # itertools.combinations returns all possible combinations
            # of comb_length elements from list_legs
            for comb in itertools.combinations(list_legs, comb_length):

                # Check if the combination is valid
                if base_objects.LegList(comb).can_combine_to_1(ref_dict_to1):

                    # Identify the rest, create a list [comb,rest] and
                    # add it to res
                    # TO BE CHANGED TO CONSERVE ORDERING ?
                    res_list = copy.copy(list_legs)
                    for leg in comb:
                        res_list.remove(leg)
                    res_list.insert(0, comb)
                    res.append(res_list)

                    # Now, deal with cases with more than 1 combination
                    # TO BE CHANGED TO CONSERVE ORDERING ?

                    # First, split the list into two, according to the
                    # position of the first element in comb, and remove
                    # all elements form comb
                    res_list1 = list_legs[0:list_legs.index(comb[0])]
                    res_list2 = list_legs[list_legs.index(comb[0]) + 1:]
                    for leg in comb[1:]:
                        res_list2.remove(leg)

                    # Create a list of type [comb,rest1,rest2(combined)]
                    res_list = [comb]
                    res_list.extend(res_list1)
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
                    leg_ids = [elem[0] for elem in \
                           ref_dict_to1[tuple([leg.get('id') \
                                               for leg in entry])]]
                    # 2) number is the minimum of leg numbers involved in the
                    # combination
                    number = min([leg.get('number') for leg in entry])
                    # 3) state is final, unless there is exactly one initial 
                    # state particle involved in the combination -> t-channel
                    if len(filter(lambda leg: leg.get('state') == 'initial',
                                  entry)) == 1:
                        state = 'initial'
                    else:
                        state = 'final'
                    # 4) from_group is True, by definition
                    from_group = True
                    # Create and add the object
                    mylegs = base_objects.LegList([base_objects.Leg(
                                    {'id':leg_id,
                                     'number':number,
                                     'state':state,
                                     'from_group':from_group}) \
                                    for leg_id in leg_ids])
                    reduced_list.append(mylegs)
                    vlist = base_objects.VertexList()

                    # Create and add the corresponding vertex
                    # Extract vertex ids corresponding to the various legs
                    # in mylegs
                    vert_ids = [elem[1] for elem in \
                           ref_dict_to1[tuple([leg.get('id') \
                                               for leg in entry])]]
                    for myleg in mylegs:
                        # Start with the considered combination...
                        myleglist = base_objects.LegList(list(entry))
                        # ... and complete with legs after reducing
                        myleglist.append(myleg)
                        # ... and consider the correct vertex id
                        vlist.append(base_objects.Vertex(
                                         {'legs':myleglist,
                                          'id':vert_ids[mylegs.index(myleg)]}))
                    vertex_list.append(vlist)

                # If entry is not a combination, switch the from_group flag
                # and add it
                else:
                    entry.set('from_group', False)
                    reduced_list.append(entry)

            # Flatten the obtained leg and vertex lists
            flat_red_lists = self.expand_list(reduced_list)
            flat_vx_lists = self.expand_list(vertex_list)

            # Combine the two lists in a list of tuple
            for i in range(0, len(flat_vx_lists)):
                res.append((base_objects.LegList(flat_red_lists[i]), \
                            base_objects.VertexList(flat_vx_lists[i])))

        return res

    def expand_list(self, mylist):
        """Takes a list of lists and elements and returns a list of flat lists.
        Example: [[1,2], 3, [4,5]] -> [[1,3,4], [1,3,5], [2,3,4], [2,3,5]]
        """

        res = []
        # Make things such the first element is always a list
        # to simplify the algorithm
        if not isinstance(mylist[0], list):
            mylist[0] = [mylist[0]]

        # Recursion stop condition, one single element
        if len(mylist) == 1:
            # [[1,2,3]] should give [[1],[2],[3]]
            return [[item] for item in mylist[0]]

        for item in mylist[0]:
            # Here the recursion happens, create lists starting with
            # each element of the first item and completed with 
            # the rest expanded
            for rest in self.expand_list(mylist[1:]):
                reslist = [item]
                reslist.extend(rest)
                res.append(reslist)

        return res


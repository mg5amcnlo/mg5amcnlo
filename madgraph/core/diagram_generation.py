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

import itertools
import copy

import madgraph.core.base_objects as base_objects

"""Amplitude object, which is what does the job for the diagram
generation algorithm
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

        return ['diagrams']

    def generate_diagrams(self,proc, ref_dict_to0, ref_dict_to1):
        """Generate diagrams. For algorithm, see wiki page
        """

        for leg in proc['legs']:
            leg.set('from_group', True)

        max_multi_to1 = max([len(key) for key in ref_dict_to1.keys()])

        return self.reduce_diagram(proc.get('legs'), ref_dict_to0, ref_dict_to1, max_multi_to1)


    def reduce_diagram(self,curr_proc, ref_dict_to0, ref_dict_to1, max_multi_to1):
        """Recursive function to reduce N diagrams to N-1
        """
        res = []

        if curr_proc is None:
            return None

        if len(curr_proc) == 2:
            if curr_proc[0].get('id') == curr_proc[1].get('id') and \
                   curr_proc[0].get('from_group') and \
                   curr_proc[1].get('from_group'):
                return [[base_objects.Vertex({'legs':curr_proc,
                                              'id':0})]]
            else:
                return None

        final_vertex = curr_proc.passesTo0(ref_dict_to0)
        if final_vertex:
            res.append([final_vertex])

        comb_lists = self.combine_legs(curr_proc,
                                       ref_dict_to1, max_multi_to1)

        leg_vertex_list = self.reduce_legs(comb_lists, ref_dict_to1)

        for leg_vertex_tuple in leg_vertex_list:
            # print 'vertex=', leg_vertex_tuple[1]
            reduced_diagram = self.reduce_diagram(leg_vertex_tuple[0],
                                             ref_dict_to0,
                                             ref_dict_to1,
                                             max_multi_to1)
            if reduced_diagram:
                vertex_list = list(leg_vertex_tuple[1])
                vertex_list.append(reduced_diagram)
                res.extend(self.expand_list(vertex_list))

        return res


    def combine_legs(self,list_legs, ref_dict_to1, max_multi_to1):
        """Take a list of legs as an input, with the reference dictionary n-1>1,
        and output a list of list of tuples of Legs (allowed combinations) 
        and Legs (rest). For algorithm, see wiki page.
        """

        res = []

        for comb_length in range(2, max_multi_to1 + 1):
            for comb in itertools.combinations(list_legs, comb_length):
                newleg = base_objects.LegList(comb).passesTo1(ref_dict_to1)
                if newleg:
                    res_list = copy.copy(list_legs)
                    for leg in comb:
                        res_list.remove(leg)
                    res_list.insert(0, comb)
                    res.append(res_list)
                    res_list1 = list_legs[0:list_legs.index(comb[0])]
                    res_list2 = list_legs[list_legs.index(comb[0]) + 1:]
                    for leg in comb[1:]:
                        res_list2.remove(leg)
                    res_list = [comb]
                    res_list.extend(res_list1)
                    # This is where recursion happens
                    for item in self.combine_legs(res_list2, ref_dict_to1, max_multi_to1):
                        res_list3 = copy.copy(res_list)
                        res_list3.extend(item)
                        res.append(res_list3)
        return res


    def reduce_legs(self,comb_lists, ref_dict_to1):
        """Takes a list of allowed leg combinations as an input and returns
        a set of lists where combinations have been properly replaced
        (one list per element in the ref_dict, so that all possible intermediate
        particles are included).
        """

        res = []

        for comb_list in comb_lists:
            reduced_list = []
            vertex_list = []
            for entry in comb_list:
                if isinstance(entry, tuple):
                    ids = ref_dict_to1[tuple([leg.get('id') for leg in entry])]
                    number = min([leg.get('number') for leg in entry])
                    if len(filter(lambda leg: leg.get('state') == 'initial',
                                  entry)) == 1:
                        state = 'initial'
                    else:
                        state = 'final'
                    from_group = True
                    mylegs = base_objects.LegList([base_objects.Leg({'id':id,
                                                                     'number':number,
                                                                     'state':state,
                                                                     'from_group':from_group}) for id in ids])
                    reduced_list.append(mylegs)
                    vlist = base_objects.VertexList()
                    for myleg in mylegs:
                        myleglist = base_objects.LegList(list(entry))
                        myleglist.append(myleg)
                        vlist.append(base_objects.Vertex({'legs':myleglist,
                        # Change id here
                                                          'id':0}))
                    vertex_list.append(vlist)
                else:
                    entry.set('from_group', False)
                    reduced_list.append(entry)

            flat_red_lists = self.expand_list(reduced_list)
            flat_vx_lists = self.expand_list(vertex_list)
            for i in range(0, len(flat_vx_lists)):
                res.append((base_objects.LegList(flat_red_lists[i]), \
                            base_objects.VertexList(flat_vx_lists[i])))

        return res

    def expand_list(self,mylist):
        """Takes a list of lists and elements and returns a list of flat lists.
        Example: [[1,2], 3, [4,5]] -> [[1,3,4], [1,3,5], [2,3,4], [2,3,5]]
        """

        res = []

        if len(mylist) == 1:
            if isinstance(mylist[0], list):
                return [[item] for item in mylist[0]]
            else:
                return [[mylist[0]]]

        if isinstance(mylist[0], list):
            for item in mylist[0]:
                # Here recursion happens
                for rest in self.expand_list(mylist[1:]):
                    reslist = [item]
                    reslist.extend(rest)
                    res.append(reslist)
        else:
            # Here recursion happens
            for rest in self.expand_list(mylist[1:]):
                reslist = [mylist[0]]
                reslist.extend(rest)
                res.append(reslist)
        return res

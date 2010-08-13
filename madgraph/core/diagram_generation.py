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
"""Classes for diagram generation. Amplitude performs the diagram
generation, DecayChainAmplitude keeps track of processes with decay
chains, and MultiProcess allows generation of processes with
multiparticle definitions.
"""

import copy
import itertools
import logging

import madgraph.core.base_objects as base_objects

from madgraph import MadGraph5Error
logger = logging.getLogger('madgraph.diagram_generation')

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
        self['diagrams'] = None

    def __init__(self, argument=None):
        """Allow initialization with Process"""

        if isinstance(argument, base_objects.Process):
            super(Amplitude, self).__init__()
            self.set('process', argument)
            self.generate_diagrams()
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
                raise self.PhysicsObjectError, \
                        "%s is not a valid Process object" % str(value)
        if name == 'diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList object" % str(value)
        return True

    def get(self, name):
        """Get the value of the property name."""

        if name == 'diagrams' and self[name] == None:
            # Have not yet generated diagrams for this process
            if self['process']:
                self.generate_diagrams()

        return Amplitude.__bases__[0].get(self, name)  #return the mother routine


    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['process', 'diagrams']

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

    def generate_diagrams(self):
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

        Be aware that the resulting vertices have all particles outgoing,
        so need to flip for incoming particles when used.

        SPECIAL CASE: For A>BC... processes which are legs in decay
        chains, we need to ensure that BC... combine first, giving A=A
        as a final vertex. This case is defined by the Process
        property is_decay_chain = True.
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

        assert model.get('particles') and model.get('interactions'), \
                  "%s is missing particles or interactions" % repr(model)

        res = base_objects.DiagramList()

        # First check that the number of fermions is even
        if len(filter(lambda leg: model.get('particle_dict')[\
                        leg.get('id')].is_fermion(), legs)) % 2 == 1:
            self['diagrams'] = res
            return res

        # Then check same number of incoming and outgoing fermions (if
        # no Majorana particles in model)
        if not model.get('got_majoranas') and \
           len(filter(lambda leg: leg.is_incoming_fermion(model), legs)) != \
           len(filter(lambda leg: leg.is_outgoing_fermion(model), legs)):
            self['diagrams'] = res
            return res

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
        leglist = base_objects.LegList(\
                       [ copy.copy(leg) for leg in process.get('legs') ])

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


        # Reduce the leg list and return the corresponding
        # list of vertices

        # For decay processes, generate starting from final-state
        # particles and make sure the initial state particle is
        # combined only as the last particle. This allows to use these
        # in decay chains later on.
        is_decay_proc = process.get_ninitial() == 1
        if is_decay_proc:
            part = model.get('particle_dict')[leglist[0].get('id')]
            # For decay chain legs, we want everything to combine to
            # the initial leg. This is done by only allowing the
            # initial leg to combine as a final identity.
            ref_dict_to0 = {(part.get_pdg_code(),part.get_anti_pdg_code()):0,
                            (part.get_anti_pdg_code(),part.get_pdg_code()):0}
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

        for vertex_list in reduced_leglist:
            res.append(base_objects.Diagram(
                            {'vertices':base_objects.VertexList(vertex_list)}))

        # Record whether or not we failed generation before required
        # s-channel propagators are taken into account
        failed_crossing = not res

        # Select the diagrams where all required s-channel propagators
        # are present.
        # Note that we shouldn't look at the last vertex in each
        # diagram, since that is the n->0 vertex
        if process.get('required_s_channels'):
            lastvx = -1
            # For decay chain processes, there is an "artificial"
            # extra vertex corresponding to particle 1=1, so we need
            # to exclude the two last vertexes.
            if is_decay_proc: lastvx = -2
            ninitial = len(filter(lambda leg: leg.get('state') == False,
                                  process.get('legs')))
            res = base_objects.DiagramList(\
                filter(lambda diagram: \
                       all([req_s_channel in \
                            [vertex.get_s_channel_id(\
                            process.get('model'), ninitial) \
                            for vertex in diagram.get('vertices')[:lastvx]] \
                            for req_s_channel in \
                            process.get('required_s_channels')]), res))

        # Select the diagrams where no forbidden s-channel propagators
        # are present.
        # Note that we shouldn't look at the last vertex in each
        # diagram, since that is the n->0 vertex
        if process.get('forbidden_s_channels'):
            ninitial = len(filter(lambda leg: leg.get('state') == False,
                              process.get('legs')))
            res = base_objects.DiagramList(\
                filter(lambda diagram: \
                       not any([vertex.get_s_channel_id(\
                                process.get('model'), ninitial) \
                                in process.get('forbidden_s_channels')
                                for vertex in diagram.get('vertices')[:-1]]),
                       res))

        # Set diagrams to res
        self['diagrams'] = res

        # Trim down number of legs and vertices used to save memory
        self.trim_diagrams()

        # Set actual coupling orders for each diagram
        for diagram in self['diagrams']:
            diagram.calculate_orders(model)

        if res:
            logger.info("Process has %d diagrams" % len(res))

        # Sort process legs according to leg number
        self.get('process').get('legs').sort()
        
        return not failed_crossing

    def reduce_leglist(self, curr_leglist, max_multi_to1, ref_dict_to0,
                       is_decay_proc = False, coupling_orders = None):
        """Recursive function to reduce N LegList to N-1
           For algorithm, see doc for generate_diagrams.
        """

        # Result variable which is a list of lists of vertices
        # to be added
        res = []

        # Stop condition. If LegList is None, that means that this
        # diagram must be discarded
        if curr_leglist is None:
            return None

        # Extract ref dict information
        model = self['process'].get('model')
        ref_dict_to1 = self['process'].get('model').get('ref_dict_to1')


        # If all legs can be combined in one single vertex, add this
        # vertex to res and continue.
        # Special treatment for decay chain legs
        if curr_leglist.can_combine_to_0(ref_dict_to0, is_decay_proc):
            # Extract the interaction id associated to the vertex 
            vertex_id = ref_dict_to0[tuple(sorted([leg.get('id') for \
                                                   leg in curr_leglist]))]

            final_vertex = base_objects.Vertex({'legs':curr_leglist,
                                                'id':vertex_id})
            # Check for coupling orders. If orders < 0, skip vertex
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
            if self['process'].get('forbidden_particles') and \
                any([abs(vertex.get('legs')[-1].get('id')) in \
                self['process'].get('forbidden_particles') \
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

            # This is where recursion happens
            # First, reduce again the leg part
            reduced_diagram = self.reduce_leglist(leg_vertex_tuple[0],
                                                  max_multi_to1,
                                                  ref_dict_to0,
                                                  is_decay_proc,
                                                  new_coupling_orders)
            # If there is a reduced diagram
            if reduced_diagram:
                vertex_list_list = [list(leg_vertex_tuple[1])]
                vertex_list_list.append(reduced_diagram)
                expanded_list = expand_list_list(vertex_list_list)
                res.extend(expanded_list)

        return res

    def reduce_orders(self, coupling_orders, model, vertex_id_list):
        """Return False if the coupling orders for any coupling is <
        0, otherwise return the new coupling orders with the vertex
        orders subtracted. If coupling_orders is not given, return
        None (which counts as success)"""

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
                if coupling in present_couplings:
                    # Reduce the number of couplings that are left
                    present_couplings[coupling] = \
                             present_couplings[coupling] - \
                             inter.get('orders')[coupling]
                    if present_couplings[coupling] < 0:
                        # We have too many couplings of this type
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
                    leg_ids = [elem[0] for elem in \
                           ref_dict_to1[tuple(sorted([leg.get('id') \
                                               for leg in entry]))]]
                    # 2) number is the minimum of leg numbers involved in the
                    # combination
                    number = min([leg.get('number') for leg in entry])
                    # 3) state is final, unless there is exactly one initial 
                    # state particle involved in the combination -> t-channel
                    if len(filter(lambda leg: leg.get('state') == False,
                                  entry)) == 1:
                        state = False
                    else:
                        state = True
                    # 4) from_group is True, by definition

                    # Create and add the object
                    mylegs = [base_objects.Leg(
                                    {'id':leg_id,
                                     'number':number,
                                     'state':state,
                                     'from_group':True}) \
                                    for leg_id in leg_ids]
                    reduced_list.append(mylegs)


                    # Create and add the corresponding vertex
                    # Extract vertex ids corresponding to the various legs
                    # in mylegs
                    vert_ids = [elem[1] for elem in \
                           ref_dict_to1[tuple(sorted([leg.get('id') \
                                               for leg in entry]))]]
                    vlist = base_objects.VertexList()
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

    def trim_diagrams(self):
        """Reduce the number of legs and vertices used in memory."""

        legs = []
        vertices = []

        for diagram in self.get('diagrams'):
            for ivx, vertex in enumerate(diagram.get('vertices')):
                for ileg, leg in enumerate(vertex.get('legs')):
                    leg.set('from_group', False)
                    try:
                        index = legs.index(leg)
                        vertex.get('legs')[ileg] = legs[index]
                    except ValueError:
                        legs.append(leg)
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

    def __init__(self, argument=None):
        """Allow initialization with Process and with ProcessDefinition"""

        if isinstance(argument, base_objects.Process):
            super(DecayChainAmplitude, self).__init__()
            if isinstance(argument, base_objects.ProcessDefinition):
                self['amplitudes'].extend(\
                MultiProcess.generate_multi_amplitudes(argument))
            else:
                self['amplitudes'].append(Amplitude(argument))
                # Clean decay chains from process, since we haven't
                # combined processes with decay chains yet
                process = copy.copy(self['amplitudes'][0].get('process'))
                process.set('decay_chains', base_objects.ProcessList())
                self['amplitudes'][0].set('process', process)
            for process in argument.get('decay_chains'):
                process.set('overall_orders', argument.get('overall_orders'))
                if not process.get('is_decay_chain'):
                    process.set('is_decay_chain',True)
                if not process.get_ninitial() == 1:
                    raise MadGraph5Error,\
                          "Decay chain process must have exactly one" + \
                          " incoming particle"
                self['decay_chains'].append(\
                    DecayChainAmplitude(process))
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
                raise self.PhysicsObjectError, \
                        "%s is not a valid AmplitudeList" % str(value)
        if name == 'decay_chains':
            if not isinstance(value, DecayChainAmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DecayChainAmplitudeList object" % \
                        str(value)
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

    def get_decay_ids(self):
        """Returns a set of all particle ids for which a decay is defined"""

        decay_ids = []

        # Get all amplitudes for the decay processes
        for amp in sum([dc.get('amplitudes') for dc \
                        in self['decay_chains']], []):
            # For each amplitude, find the initial state leg
            decay_ids.append(amp.get('process').get_initial_ids()[0])
            
        # Return a list with unique ids
        return list(set(decay_ids))
    
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

    def __init__(self, argument=None):
        """Allow initialization with ProcessDefinition or
        ProcessDefinitionList"""

        if isinstance(argument, base_objects.ProcessDefinition):
            super(MultiProcess, self).__init__()
            self['process_definitions'].append(argument)
            self.get('amplitudes')
        elif isinstance(argument, base_objects.ProcessDefinitionList):
            super(MultiProcess, self).__init__()
            self['process_definitions'] = argument
            self.get('amplitudes')
        elif argument != None:
            # call the mother routine
            super(MultiProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(MultiProcess, self).__init__()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'process_definitions':
            if not isinstance(value, base_objects.ProcessDefinitionList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessDefinitionList object" % str(value)

        if name == 'process':
            if not isinstance(value, base_objects.ProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessList object" % str(value)

        return True

    def get(self, name):
        """Get the value of the property name."""

        if (name == 'amplitudes') and not self[name]:
            for process_def in self.get('process_definitions'):
                if process_def.get('decay_chains'):
                    # This is a decay chain process
                    # Store amplitude(s) as DecayChainAmplitude
                    self['amplitudes'].append(\
                        DecayChainAmplitude(process_def))
                else:
                    self['amplitudes'].extend(\
                        MultiProcess.generate_multi_amplitudes(process_def))

        return MultiProcess.__bases__[0].get(self, name) # call the mother routine

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['process_definitions', 'amplitudes']

    @staticmethod
    def generate_multi_amplitudes(process_definition):
        """Generate amplitudes in a semi-efficient way.
        Make use of crossing symmetry for processes that fail diagram
        generation, but not for processes that succeed diagram
        generation.  Doing so will risk making it impossible to
        identify processes with identical amplitudes.
        """

        assert isinstance(process_definition, base_objects.ProcessDefinition), \
                                    "%s not valid ProcessDefinition object" % \
                                    repr(process_definition)

        processes = base_objects.ProcessList()
        amplitudes = AmplitudeList()

        # failed_procs are processes that have already failed
        # based on crossing symmetry
        failed_procs = []
        
        model = process_definition['model']
        
        isids = [leg['ids'] for leg in \
                 filter(lambda leg: leg['state'] == False, process_definition['legs'])]
        fsids = [leg['ids'] for leg in \
                 filter(lambda leg: leg['state'] == True, process_definition['legs'])]

        # Generate all combinations for the initial state
        
        for prod in apply(itertools.product, isids):
            islegs = [\
                    base_objects.Leg({'id':id, 'state': False}) \
                    for id in prod]

            # Generate all combinations for the final state, and make
            # sure to remove double counting

            red_fsidlist = []

            for prod in apply(itertools.product, fsids):

                # Remove double counting between final states
                if tuple(sorted(prod)) in red_fsidlist:
                    continue
                
                red_fsidlist.append(tuple(sorted(prod)));
                
                # Generate leg list for process
                leg_list = [copy.copy(leg) for leg in islegs]
                
                leg_list.extend([\
                        base_objects.Leg({'id':id, 'state': True}) \
                        for id in prod])
                
                legs = base_objects.LegList(leg_list)

                # Setup process
                process = base_objects.Process({\
                              'legs':legs,
                              'model':process_definition.get('model'),
                              'id': process_definition.get('id'),
                              'orders': process_definition.get('orders'),
                              'required_s_channels': \
                                 process_definition.get('required_s_channels'),
                              'forbidden_s_channels': \
                                 process_definition.get('forbidden_s_channels'),
                              'forbidden_particles': \
                                 process_definition.get('forbidden_particles'),
                              'is_decay_chain': \
                                 process_definition.get('is_decay_chain'),
                              'overall_orders': \
                                 process_definition.get('overall_orders')})

                # Check for crossed processes
                sorted_legs = sorted(legs.get_outgoing_id_list(model))
                # Check if crossed process has already failed
                # In that case don't check process
                if tuple(sorted_legs) in failed_procs:
                    continue

                amplitude = Amplitude({"process": process})
                if not amplitude.generate_diagrams():
                    # Add process to failed_procs
                    failed_procs.append(tuple(sorted_legs))
                if amplitude.get('diagrams'):
                    amplitudes.append(amplitude)

        # Raise exception if there are no amplitudes for this process
        if not amplitudes:
            raise MadGraph5Error, \
            "No amplitudes generated from process %s. Please enter a valid process" % \
                  process_definition.nice_string()
        

        # Return the produced amplitudes
        return amplitudes
            
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

    for item in apply(itertools.product, tmplist):
        res.append(list(item))

    return res

def expand_list_list(mylist):
    """Recursive function. Takes a list of lists and lists of lists
    and returns a list of flat lists.
    Example: [[1,2],[[4,5],[6,7]]] -> [[1,2,4,5], [1,2,6,7]]
    """

    res = []
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


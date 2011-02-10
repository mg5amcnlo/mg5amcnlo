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
 
"""Classes for generation of color-ordered
diagrams. ColorOrderedAmplitude performs the diagram generation,
ColorOrderedLeg has extra needed flags.

Additions to the regular diagram generation algorithm:

1) Only one set of color triplet lines are allowed (taken from the
first diagram in the process stripped of gluons)

2) Gluon lines are allowed to be merged only with neighboring gluon
lines.
"""

import array
import copy
import itertools
import logging

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
from madgraph import MadGraph5Error

logger = logging.getLogger('madgraph.color_ordered_amplitudes')


#===============================================================================
# ColorOrderedLeg
#===============================================================================
class ColorOrderedLeg(base_objects.Leg):

    """Leg with two additional flags: fermion line number, and color
    ordering number. These will disallow all non-correct orderings of
    colored fermions and gluons. So far, only color triplets
    and gluons are treated (i.e., SM)"""

    def default_setup(self):
        """Default values for all properties"""

        super(ColorOrderedLeg, self).default_setup()
        # Flag to keep track of color ordering
        # For colored particles: {Cycle: (upper, lower)}
        # For color singlets: {}
        self['color_ordering'] = {}
    
    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'color_ordering':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict object" % str(value)
        else:
            super(ColorOrderedLeg, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id', 'number', 'state', 'from_group',
                'color_ordering']

#===============================================================================
# ColorOrderedAmplitude
#===============================================================================
class ColorOrderedAmplitude(diagram_generation.Amplitude):
    """Amplitude: process + list of diagrams (ordered)
    Initialize with a process, then call generate_diagrams() to
    generate the diagrams for the amplitude
    """

    def __init__(self, argument=None):
        """Allow color-ordered initialization with Process"""

        if isinstance(argument, base_objects.Process):
            super(ColorOrderedAmplitude, self).__init__()
            self.set('process', argument)
            self.setup_process()
        elif argument != None:
            # call the mother routine
            super(ColorOrderedAmplitude, self).__init__(argument)
        else:
            # call the mother routine
            super(ColorOrderedAmplitude, self).__init__()

    def default_setup(self):
        """Add number of gluon flag to Amplitude members"""
        
        super(ColorOrderedAmplitude, self).default_setup()
        self['color_flows'] = ColorOrderedFlowList()

    def setup_process(self):
        """Add negative singlet gluon to model. Setup
        ColorOrderedFlows corresponding to all color flows of this
        process. Each ordering of unique particles with color triplets
        pairwise combined as (3 3bar)...(3 3bar)... etc. corresponds
        to a unique color flow."""

        process = self.get('process')
        model = process.get('model')
        legs = base_objects.LegList([copy.copy(l) for l in \
                                     process.get('legs')])

        # Add color negative singlets to model corresponding to all
        # color octets, with only 3-3bar-8 interactions.
        # Color factor: -1/6 * Id(1,2)

        # Set leg numbers for the process
        for i, leg in enumerate(legs):
            leg.set('number', i+1)
            # Reverse initial state legs to have all legs outgoing
            if not leg.get('state'):
                leg.set('id', model.get_particle(leg.get('id')).\
                        get_anti_pdg_code())

        # Create list of leg (number, id, color) to find unique color flows
        order_legs = [(l.get('number'), l.get('id'),
                       model.get_particle(l.get('id')).get_color()) \
                      for l in legs]

        # Identify particle types: Color singlets (can connect
        # anywhere), color triplets, anti-triplets and octets
        triplet_legs = [l for l in order_legs if l[2] == 3]
        anti_triplet_legs = [l for l in order_legs if l[2] == -3]
        octet_legs = [l for l in order_legs if abs(l[2]) == 8]
        singlet_legs = [l for l in order_legs if abs(l[2]) == 1]

        if len(triplet_legs) != len(anti_triplet_legs):
            # Need triplets in pairs for valid amplitude
            return

        if len(triplet_legs+anti_triplet_legs + octet_legs+singlet_legs) != \
               len(legs):
            raise MadGraph5Error, \
                  "Non-triplet/octet/singlet found in color ordered amplitude"
        
        # Determine all unique permutations of particles corresponding
        # to color flows. Valid color flows have color triplet pairs
        # prganized as (3 3bar)...(3 3bar)...

        # Extract all unique permutations of legs
        leg_perms = []
        if triplet_legs:
            first_leg = [triplet_legs.pop(0)]
        else:
            first_leg = [octet_legs.pop(0)]

        for perm in itertools.permutations(sorted(triplet_legs + \
                                                  anti_triplet_legs + \
                                                  octet_legs,
                                                  lambda l1, l2: l1[0] - l2[0])):
            # Permutations with same flavor ordering are thrown out
            if [p[1] for p in perm] in \
               [[p[1] for p in leg_perm] for leg_perm in leg_perms]:
                continue

            # Also remove permutations where a triplet and
            # anti-triplet are not next to each other and ordered as 
            # (3 3bar)...(3 3bar)...
            trip = [(i,p) for (i,p) in enumerate(first_leg + list(perm)) \
                    if abs(p[2])==3]
            failed = False
            for i in range(0,len(trip),2):
                if trip[i][0] != trip[i+1][0]-1 or \
                       trip[i][1][2]-trip[i+1][1][2] != 6:
                    failed = True
                    break
            if failed:
                continue

            leg_perms.append(perm)

        # Create color flow amplitudes corresponding to all resulting
        # permutations
        color_flows = ColorOrderedFlowList()
        colegs = base_objects.LegList([ColorOrderedLeg(l) for l in legs])

        # Set color flow flags (color_ordering) so that every
        # chain (3bar 8 8 .. 8 3) has a unique color ordering
        # group, and each entry in the chain has a unique color flow
        # flag {chain:(n,n)}

        # For > 2 triplet pairs, we need to remove double counting due
        # to different ordering of color chains. Idea: make list of
        # [(pdg,group,color flow) for all combinations of groups
        # except the last (which is fixed to the first triplet)

        used_flows = []

        for perm in leg_perms:
            colegs = base_objects.LegList([ColorOrderedLeg(l) for l in legs])
            # Keep track of number of triplets
            ichain = 0
            ileg = 0
            # Set color ordering flags for all colored legs
            for perm_leg in list(perm) + first_leg:
                leg = colegs[perm_leg[0]-1]
                if perm_leg[2] == -3:
                    ichain += 1
                    ileg = 0
                ileg += 1
                leg.set('color_ordering', {ichain: (ileg, ileg)})
            if ichain > 2:
                # Make sure we don't have double counting between
                # different orders of chains
                failed = False
                for perm in itertools.permutations(range(1, ichain+1), ichain): 
                    this_flow = sorted(sum([[(leg.get('id'), perm[i],
                                           leg.get('color_ordering')[i]) \
                                          for leg in colegs if i in \
                                          leg.get('color_ordering')] for \
                                         i in range(len(perm))], []))
                    if this_flow in used_flows:
                        failed = True
                        break
                    used_flows.append(this_flow)
                if failed:
                    continue
            # Restore initial state leg identities
            for leg in colegs:
                if not leg.get('state'):
                    leg.set('id', model.get_particle(leg.get('id')).\
                            get_anti_pdg_code())
            coprocess = copy.copy(process)
            coprocess.set('legs', colegs)
            # Create the color ordered flow
            flow = ColorOrderedFlow(coprocess)
            if flow.get('diagrams'):
                color_flows.append(flow)

        self.set('color_flows', color_flows)

#===============================================================================
# ColorOrderedFlow
#===============================================================================
class ColorOrderedFlow(diagram_generation.Amplitude):
    """Amplitude: color flow process + list of diagrams (ordered)
    Initialize with a set of processes which together specify a color
    ordered process, then call generate_diagrams() to generate the
    diagrams for the color flow.
    """

    def __init__(self, argument=None):
        """Allow color-ordered initialization with Process"""

        if isinstance(argument, base_objects.Process):
            super(ColorOrderedFlow, self).__init__()
            self.set('process', argument)
            # Set max color order for all color ordering groups
            legs = argument.get('legs')
            groups = set(sum([l.get('color_ordering').keys() for l in legs],[]))
            self.set('max_color_orders', dict( \
                [(g, max([l.get('color_ordering')[g][0] for l \
                      in legs if g in l.get('color_ordering')]))\
                 for g in groups]))
            self.generate_diagrams()
        elif argument != None:
            # call the mother routine
            super(ColorOrderedFlow, self).__init__(argument)
        else:
            # call the mother routine
            super(ColorOrderedFlow, self).__init__()

    def default_setup(self):
        """Add number of gluon flag to Amplitude members"""
        
        super(ColorOrderedFlow, self).default_setup()
        self['max_color_orders'] = {}

    def get_combined_legs(self, legs, leg_vert_ids, number, state):
        """Determine if the combination of legs is valid with color
        ordering. Algorithm: A combination is valid if

        1) all particles with the same color ordering group have
        adjacent color flow numbers

        2) No color octet is dead-ending (if there are other color groups)

        3) Resulting legs have: a) no uncompleted groups (if color singlet),
           b) exactly one uncompleted group (if color triplet),
           c) no uncompleted group, and 1 or 2 groups (if color octet).
        """

        # Find all color ordering groups
        groups = set(sum([l.get('color_ordering').keys() for l in legs],[]))
        model = self.get('process').get('model')
        new_leg_colors = dict([(leg_id,
                                model.get_particle(leg_id).get('color')) \
                           for (leg_id, vert_id) in leg_vert_ids])
        old_leg_colors = dict([(leg_id,
                                model.get_particle(leg_id).get('color')) \
                               for leg_id in set([l.get('id') for l in legs])])
        color_orderings = dict([(leg_id, {}) for (leg_id, vert_id) in \
                                leg_vert_ids])

        for group in groups:
            # sort the legs with color ordering number
            color_legs = [(i, l.get('color_ordering')[group]) for \
                          (i, l) in enumerate(legs) \
                          if group in l.get('color_ordering')]

            # Check for unattached color octet legs
            if len(groups) > 1 and len(color_legs) == 1 and \
               old_leg_colors[legs[color_legs[0][0]].get('id')] == 8 and \
                   len(legs[color_legs[0][0]].get('color_ordering')) == 1:
                return []            

            color_legs.sort(lambda l1, l2: l1[1][0] - l2[1][0])

            color_ordering = (color_legs[0][1][0], color_legs[-1][1][1])

            # Check that we don't try to combine legs with
            # non-adjacent color ordering (allowing to wrap around
            # cyclically)
            lastmax = color_legs[0][1][1]
            ngap = 0
            # First check if there is a gap between last and first
            if (color_legs[0][1][0] > 1 or \
                color_legs[-1][1][1] < self.get('max_color_orders')[group]) and \
                color_legs[-1][1][1] + 1 != color_legs[0][1][0]:
                ngap = 1
            # Check if there are gaps between other legs
            for leg in color_legs[1:]:
                if leg[1][0] != lastmax + 1:
                    ngap += 1
                    color_ordering = (leg[1][0], lastmax)
                if ngap == 2:
                    # For adjacent legs, only allow one gap
                    return []
                lastmax = leg[1][1]
            # Set color ordering for new legs
            for leg_id in color_orderings.keys():
                color_orderings[leg_id][group] = color_ordering
                
        # Check validity of resulting color orderings for the legs
        ileg = 0
        while ileg < len(leg_vert_ids):
            leg_id, vert_id = leg_vert_ids[ileg]
            if abs(new_leg_colors[leg_id]) == 1:
                # Color singlets need all groups to be completed
                if any([color_orderings[leg_id][group] != \
                        (1, self.get('max_color_orders')[group]) \
                        for group in groups]):
                    leg_vert_ids.remove((leg_id, vert_id))
                else:
                    ileg += 1
                color_orderings[leg_id] = {}
            elif abs(new_leg_colors[leg_id]) == 3:
                # Color triplets should have exactly one color ordering
                color_orderings[leg_id] = \
                    dict([(group, color_orderings[leg_id][group]) for \
                          group in groups if \
                          color_orderings[leg_id][group] != \
                           (1, self.get('max_color_orders')[group])])
                if len(color_orderings[leg_id].keys()) != 1:
                    leg_vert_ids.remove((leg_id, vert_id))                    
                else:
                    ileg += 1
            elif abs(new_leg_colors[leg_id]) == 8:
                # Color octets should have no completed groups,
                # and 1 or 2 orderings
                if any([color_orderings[leg_id][group] == \
                        (1, self.get('max_color_orders')[group]) \
                        for group in groups]) or \
                        len(color_orderings[leg_id]) < 1 or \
                        len(color_orderings[leg_id]) > 2:
                    leg_vert_ids.remove((leg_id, vert_id))
                else:
                    ileg += 1

        # Return all legs that have valid color_orderings
        mylegs = [(ColorOrderedLeg({'id':leg_id,
                                   'number':number,
                                   'state':state,
                                   'from_group':True,
                                   'color_ordering': color_orderings[leg_id]}),
                   vert_id) for (leg_id, vert_id) in leg_vert_ids]

        return mylegs

    def get_combined_vertices(self, legs, vert_ids):
        """Check that all color-ordering groups are pairwise
        connected, i.e., we do not allow groups that are disjuct."""

        # Find all color ordering groups
        groups = set(sum([l.get('color_ordering').keys() for l in legs],[]))
        # Extract legs colored under each group
        group_legs = {}
        for group in groups:
            group_legs[group] = set([l.get('number') for l in legs \
                                     if group in l.get('color_ordering')])
        # Check that all groups are pair-wise connected by particles
        if len(groups) > 1:
            for g1, g2 in itertools.combinations(groups, 2):
                if not group_legs[g1].intersection(group_legs[g2]):
                    return []

        return vert_ids

#===============================================================================
# AmplitudeList
#===============================================================================
class ColorOrderedFlowList(diagram_generation.AmplitudeList):
    """List of ColorOrderedFlow objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid Amplitude for the list."""

        return isinstance(obj, ColorOrderedFlow)

#===============================================================================
# COHelasWavefunction
#===============================================================================
class COHelasWavefunction(helas_objects.HelasWavefunction):
    """COHelasWavefunction object, a HelasWavefunction with added
    fields for comparison
    """

    def default_setup(self):
        """Default values for all properties"""

        super(COHelasWavefunction, self).default_setup()
        # Comparison array, with wf numbers, interaction id and
        # lorentz-color index
        self['compare_array'] = []
        # external wavefunction numbers
        self['external_numbers'] = array.array('I')

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'compare_array':
            if not isinstance(value, list) and not \
                   isinstance(value, array.array):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list object" % str(value)
        elif name == 'external_numbers':
            if not isinstance(value, array.array):
                raise self.PhysicsObjectError, \
                        "%s is not a valid array object" % str(value)
        else:
            super(COHelasWavefunction, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return super(COHelasWavefunction, self).get_sorted_keys() + \
               ['compare_array', 'external_numbers']

    def get(self, name):
        """Enhanced get function to initialize compare_array and external_numbers."""

        if name in ['compare_array', 'external_numbers'] and not self[name]:
            self.create_arrays()
            
        return super(COHelasWavefunction, self).get(name)        

    def create_arrays(self):
        """Create the comparison arrays compare_array and external_numbers"""

        if not self.get('mothers'):
            # This is an external wavefunction
            self.set('external_numbers',
                     array.array('I',[self.get('number')]))
            self.set('compare_array', array.array('I',
                                                self['external_numbers']))
        else:
            self.set('external_numbers',
                     array.array('I',
                                 sorted(sum([list(m.get('external_numbers')) \
                                        for m in self.get('mothers')], []))))
            self.set('compare_array', [[m.get('external_numbers') for \
                                        m in self.get('mothers')],
                                       self.get('interaction_id'),
                                       self.get('coupl_key')])

#===============================================================================
# COHelasWavefunction
#===============================================================================
class COHelasAmplitude(helas_objects.HelasAmplitude):
    """COHelasAmplitude object, a HelasAmplitude with added
    fields for comparison
    """

    def default_setup(self):
        """Default values for all properties"""

        super(COHelasAmplitude, self).default_setup()
        # external wavefunction numbers
        self['external_numbers'] = []

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'external_numbers':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list object" % str(value)
        else:
            super(COHelasAmplitude, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return super(COHelasAmplitude, self).get_sorted_keys() + \
               ['external_numbers']

    def get(self, name):
        """Enhanced get function to initialize compare_array and external_numbers."""

        if name in ['external_numbers'] and not self[name]:
            self.create_arrays()
            
        return super(COHelasAmplitude, self).get(name)        

    def create_arrays(self):
        """Create the comparison arrays compare_array and external_numbers"""

        self.set('external_numbers', [sorted([list(m.get('external_numbers')) \
                                              for m in self.get('mothers')]),
                                      self.get('interaction_id'),
                                      self.get('coupl_key')])

#===============================================================================
# BGHelasCurrent
#===============================================================================
class BGHelasCurrent(COHelasWavefunction):
    """BGHelasCurrent object, which combines HelasWavefunctions into
    Behrends-Giele currents
    """

    def create_arrays(self):
        """Create the comparison arrays compare_array and external_numbers"""

        # external_numbers is the set of external numbers of the mothers
        self.set('external_numbers',
                 array.array('I', sorted(list(set(\
                                  sum([list(c.get('external_numbers')) for c \
                                       in self.get('mothers')],
                                      []))))))
        self.set('compare_array', [self['external_numbers'],
                                   self.get('pdg_code')])
        
    def get_call_key(self):
        """Generate the ('sum', spins) tuple used as key for
        the helas call dictionaries in HelasModel"""

        res = [m.get('spin') for m in self.get('mothers')]

        return ('sum', tuple(res))

    
#===============================================================================
# BGHelasMatrixElement
#===============================================================================
class BGHelasMatrixElement(helas_objects.HelasMatrixElement):
    """BGHelasMatrixElement: Behrends-Giele version of a
    HelasMatrixElement, starting from a ColorOrderedFlow.

    After regular helas diagram generation, performs the following:

    1. Go through all wavefunctions and amplitudes to select only
    those with the correct color structure

    2. Go through all wavefunctions to combine the wavefunctions with
    the same external particle numbers using BGHelasCurrents

    3. Go through all amplitudes, and use only one amplitude per set
    of BGHelasCurrents
    """

    def generate_helas_diagrams(self, amplitude, optimization=1,
                                decay_ids=[]):
        """Generate Behrends-Giele diagrams for a color ordered amplitude
        """
        
        assert  isinstance(amplitude, ColorOrderedFlow), \
                    "Missing or erraneous arguments for generate_helas_diagrams"
        
        # First generate full set of wavefunctions and amplitudes
        super(BGHelasMatrixElement, self).generate_helas_diagrams(amplitude,
                                                                  optimization,
                                                                  decay_ids)

        # Go through and change wavefunctions into COHelasWavefunction
        all_wavefunctions = self.get_all_wavefunctions()
        co_wavefunctions = helas_objects.HelasWavefunctionList(\
            [COHelasWavefunction(wf) for wf in all_wavefunctions])
        # Replace all mothers with the co_wavefunctions
        for wf in co_wavefunctions:
            wf.get('mothers')[:] = \
                                [co_wavefunctions[all_wavefunctions.index(w)] \
                                 for w in wf.get('mothers')]
        # Same thing for amplitudes
        co_amplitudes = helas_objects.HelasAmplitudeList(\
            [COHelasAmplitude(wf) for wf in self.get_all_amplitudes()])
        for amp in co_amplitudes:
            amp.get('mothers')[:] = \
                                [co_wavefunctions[all_wavefunctions.index(w)] \
                                 for w in amp.get('mothers')]

        # Sort wavefunctions according to len(external_number)
        co_wavefunctions.sort(lambda w1,w2: len(w1.get('external_numbers')) - \
                              len(w2.get('external_numbers')))
        
        # TODO: Remove irrelevant wavefunctions and
        # amplitudes (based on color)

        # Go through wavefunctions and make all possible combinations
        combined_wavefunctions = helas_objects.HelasWavefunctionList()
        while co_wavefunctions:
            # Pick out all wavefunctions with the same external numbers
            combine_functions = [w for w in co_wavefunctions if \
                                 w.get('external_numbers') == \
                                 co_wavefunctions[0].get('external_numbers') \
                                 and w.get('pdg_code') == \
                                 co_wavefunctions[0].get('pdg_code')]
            if len(combine_functions) == 1:
                # Just add the wavefunction to combined_wavefunctions
                wf = combine_functions.pop(0)
                combined_wavefunctions.append(wf)
                # Remove used wavefunctions from co_wavefunctions
                co_wavefunctions.remove(wf)
            else:
                # Combine wavefunctions to a current
                combine_wf = BGHelasCurrent(combine_functions[0])
                combine_wf.set('mothers', helas_objects.HelasWavefunctionList())
                while combine_functions:
                    wf = combine_functions.pop(0)
                    # Remove used wavefunctions from co_wavefunctions
                    co_wavefunctions.remove(wf)
                    # Check if a wavefunction which uses the same
                    # combined wfs is already present in the current
                    if wf.get('compare_array') in \
                       [m.get('compare_array') for m in \
                        combine_wf.get('mothers')]:
                        continue
                    # Replace the wavefunction mothers in this
                    # wavefunction with corresponding currents
                    for i, m in enumerate(wf.get('mothers')):
                        try:
                            ind = [c.get('compare_array') for c in \
                                   combined_wavefunctions].index(\
                                           [m.get('external_numbers'),
                                            m.get('pdg_code')])
                        except ValueError:
                            continue
                        wf.get('mothers')[i] = combined_wavefunctions[ind]
                    # Add the resulting wavefunction to
                    # combined_wavefunctions and to combine_wf
                    combined_wavefunctions.append(wf)
                    combine_wf.get('mothers').append(wf)
                # Add combine_wf to combined_wavefunctions
                combined_wavefunctions.append(combine_wf)

        # Set wf number for all wavefunctions
        for i, wf in enumerate(combined_wavefunctions):
            wf.set('number', i+1)
        
        # Do the same thing for amplitudes
        combined_amplitudes = helas_objects.HelasAmplitudeList()

        while co_amplitudes:
            # Pick out all amplitudes with the same external number mothers
            amp = co_amplitudes.pop(0)
            remove_amps = [a for a in co_amplitudes if \
                                 a.get('external_numbers') == \
                                 amp.get('external_numbers')]
            for a in remove_amps:
                co_amplitudes.remove(a)
            
            # Replace the amplitude mothers in this
            # amplitude with corresponding currents
            for i, m in enumerate(amp.get('mothers')):
                try:
                    ind = [c.get('compare_array') for c in \
                           combined_wavefunctions].index(\
                                   [m.get('external_numbers'),
                                    m.get('pdg_code')])
                except ValueError:
                    continue
                amp.get('mothers')[i] = combined_wavefunctions[ind]

            combined_amplitudes.append(amp)

        # Set amplitude number for all amplitudes
        for i, amp in enumerate(combined_amplitudes):
            amp.set('number', i+1)
        
        diagram = helas_objects.HelasDiagram()
        diagram.set('wavefunctions', combined_wavefunctions)
        diagram.set('amplitudes', combined_amplitudes)
        diagram.set('number', 1)

        self.set('diagrams', helas_objects.HelasDiagramList([diagram]))
        

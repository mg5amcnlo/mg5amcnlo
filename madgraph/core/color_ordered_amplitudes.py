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

import copy
import itertools
import logging

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
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
        process. Each ordering of color triplets corresponds to a
        unique color flow."""

        process = self.get('process')
        model = process.get('model')
        legs = base_objects.LegList([copy.copy(l) for l in \
                                     process.get('legs')])

        # Add color negative singlet gluon to model - TO BE DONE

        # Set leg numbers for the process
        for i, leg in enumerate(legs):
            leg.set('number', i+1)
            # Reverse initial state legs
            if not leg.get('state') and \
               not model.get_particle(leg.get('id')).get('self_antipart'):
                leg.set('id', -leg.get('id'))

        # First set flags to get only relevant combinations
        order_legs = [(l.get('number'), l.get('id'),
                       model.get_particle(l.get('id')).get_color()) \
                      for l in legs]

        # Identify particle types: Color singlets (can connect
        # anywhere), colored particles
        triplet_legs = [l for l in order_legs if l[2] == 3]
        anti_triplet_legs = [l for l in order_legs if l[2] == -3]
        octet_legs = [l for l in order_legs if abs(l[2]) == 8]
        singlet_legs = [l for l in order_legs if abs(l[2]) == 1]

        if len(triplet_legs+anti_triplet_legs + octet_legs+singlet_legs) != \
               len(legs):
            raise MadGraph5Error, \
                  "Non-triplet/octet/singlet found in color ordered amplitude"
        
        # Insert all orderings of triplet legs into octet legs. Each
        # ordering corresponds to a color flow.

        # Extract all unique combinations of legs
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
            # anti-triplet are not next to each other
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

        # Create color flow amplitudes corresponding to all unique combinations
        color_flows = ColorOrderedFlowList()
        for perm in leg_perms:
            legs = base_objects.LegList([ColorOrderedLeg(l) for l in legs])
            # Keep track of number of triplets
            num_triplets = 0
            ileg = 1
            legs[first_leg[0][0]-1].set('color_ordering', {0:(1,1)})
            # Set color ordering flags for all colored legs
            for perm_leg in perm:
                leg = legs[perm_leg[0]-1]
                if perm_leg[2] == 3:
                    num_triplets += 1
                    ileg = 0
                ileg += 1
                leg.set('color_ordering', {num_triplets: (ileg, ileg)})
            # Restore initial state leg identities
            for leg in legs:
                if not leg.get('state') and \
                   not model.get_particle(leg.get('id')).get('self_antipart'):
                    leg.set('id', -leg.get('id'))
            coprocess = copy.copy(process)
            coprocess.set('legs', legs)
            color_flows.append(ColorOrderedFlow(coprocess))

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
            print "Color flow: ",[(l.get('number'),l.get('id'),l.get('color_ordering')) \
                        for l in argument.get('legs')]
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
        """Create a set of new legs from the info given."""

        # Find all color ordering groups
        groups = set(sum([l.get('color_ordering').keys() for l in legs],[]))
        model = self.get('process').get('model')
        leg_colors = dict([(leg_id, model.get_particle(leg_id).get('color')) \
                           for (leg_id, vert_id) in leg_vert_ids])
        color_orderings = dict([(leg_id, {}) for (leg_id, vert_id) in \
                                leg_vert_ids])
        failed_legs = []
        for group in groups:
            # sort the legs with color ordering number
            color_legs = [l.get('color_ordering')[group] for l in legs \
                         if group in l.get('color_ordering')]

            color_legs.sort(lambda l1, l2: l1[0] - l2[0])

            color_ordering = (color_legs[0][0], color_legs[-1][1])

            # Check that we don't try to combine legs with
            # non-adjacent color ordering (allowing to wrap around
            # cyclically)
            lastmax = color_legs[0][1]
            ngap = 0
            # First check if there is a gap between last and first
            if (color_legs[0][0] > 1 or \
                color_legs[-1][1] < self.get('max_color_orders')[group]) and \
                color_legs[-1][1] + 1 != color_legs[0][0]:
                ngap = 1
            for leg in color_legs[1:]:
                if leg[0] != lastmax + 1:
                    ngap += 1
                    color_ordering = (leg[0], lastmax)
                if ngap == 2:
                    return []
                lastmax = leg[1]

            # Color ordering [first, last] counts as color singlet
            # Set color ordering for legs
            for leg_id, vert_id in leg_vert_ids:
            # Color ordering [first, last] counts as color singlet, so
            # is not allowed for colored propagator
                if abs(leg_colors[leg_id]) != 1:
                    if color_ordering == \
                           (1, self.get('max_color_orders')[group]):
                        failed_legs.append(leg_id)
                    else:
                        color_orderings[leg_id][group] = color_ordering

        # Return all legs that have valid color_orderings
        mylegs = [(ColorOrderedLeg({'id':leg_id,
                                   'number':number,
                                   'state':state,
                                   'from_group':True,
                                   'color_ordering': color_orderings[leg_id]}),
                   vert_id) \
                  for leg_id, vert_id in leg_vert_ids if \
                  leg_id not in failed_legs and \
                  (abs(leg_colors[leg_id]) == 1 or \
                   abs(leg_colors[leg_id]) == 3 and \
                   len(color_orderings[leg_id]) == 1 or \
                   abs(leg_colors[leg_id]) == 8 and \
                   len(color_orderings[leg_id]) > 0 and \
                   len(color_orderings[leg_id]) <= 2)]

        print 'legs: ',[(l.get('number'),l.get('id'),l.get('color_ordering')) \
                        for l in legs]
        print 'newlegs: ',[(l.get('number'),l.get('id'),l.get('color_ordering')) \
                        for l,v in mylegs]

        return mylegs

    def get_combined_vertices(self, legs, vert_ids):
        """Allow for selection of vertex ids. This can be
        overloaded by daughter classes."""

        #return get_combined_legs(self, legs,
        #                         [(0, v) for v in vert_ids], 0, False)

        # Combine legs by color order groups
        
        # Find all color ordering groups
        groups = set(sum([l.get('color_ordering').keys() for l in legs],[]))
        # Extract legs colored under each group
        group_legs = {}
        for group in groups:
            group_legs[group] = set([l.get('number') for l in legs \
                                     if group in l.get('color_ordering')])
        # Check that all groups are pair-wise connected by particles
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


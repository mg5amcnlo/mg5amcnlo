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
"""Classes for diagram generation with loop features.
"""

import array
import copy
import itertools
import logging

import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

from madgraph import MadGraph5Error
logger = logging.getLogger('madgraph.loop_diagram_generation')

#===============================================================================
# LoopAmplitude
#===============================================================================
class LoopAmplitude(diagram_generation.Amplitude):
    """NLOAmplitude: process + list of diagrams (ordered)
    Initialize with a process, then call generate_diagrams() to
    generate the diagrams for the amplitude
    """

    def default_setup(self):
        """Default values for all properties"""
        
        # The 'diagrams' entry from the mother class is inherited but will not
        # be used in NLOAmplitude, because it is split into the four following
        # different categories of diagrams.
        super(LoopAmplitude, self).default_setup()
        self['born_diagrams'] = None        
        self['loop_diagrams'] = None
        # This is in principle equal to self['born_diagram']==[] but it can be 
        # that for some reason the born diagram can be generated but do not
        # contribute.
        # This will decide wether the virtual is squared against the born or
        # itself.
        self['has_born'] = True
        # This where the structures obtained for this amplitudes are stored
        self['structure_repository'] = loop_base_objects.FDStructureList() 

        # A list that registers what Lcut particle have already been
        # employed in order to forbid them as loop particles in the 
        # subsequent diagram generation runs.
        self.lcutpartemployed=[]

    def __init__(self, argument=None):
        """Allow initialization with Process"""
        
        if isinstance(argument, base_objects.Process):
            super(LoopAmplitude, self).__init__()
            self.set('process', argument)
            self.generate_diagrams()
        elif argument != None:
            # call the mother routine
            super(LoopAmplitude, self).__init__(argument)
        else:
            # call the mother routine
            super(LoopAmplitude, self).__init__()

    def get_sorted_keys(self):
        """Return diagram property names as a nicely sorted list."""

        return ['process', 'diagrams', 'has_mirror_process', 'born_diagrams',
                'loop_diagrams','has_born',
                'structure_repository']

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList" % str(value)
            for diag in value:
                if not isinstance(diag,loop_base_objects.LoopDiagram):
                    raise self.PhysicsObjectError, \
                        "%s contains a diagram which is not an NLODiagrams." % str(value)
        if name == 'born_diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList" % str(value)
            for diag in value:
                if not isinstance(diag,loop_base_objects.LoopDiagram):
                    raise self.PhysicsObjectError, \
                        "%s contains a diagram which is not an NLODiagrams." % str(value)
        if name == 'loop_diagrams':
            if not isinstance(value, base_objects.DiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid DiagramList" % str(value)
            for diag in value:
                if not isinstance(diag,loop_base_objects.LoopDiagram):
                    raise self.PhysicsObjectError, \
                        "%s contains a diagram which is not an NLODiagrams." % str(value)
        if name == 'has_born':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value)
        if name == 'structure_repository':
            if not isinstance(value, loop_base_objects.FDStructureList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value)

        else:
            return super(LoopAmplitude, self).filter(name, value)

        return True

    def set(self, name, value):
        """Redefine set for the particular case of diagrams"""

        if name == 'diagrams':
            if self.filter(name, value):
                self['born_diagrams']=base_objects.DiagramList([diag for diag in value if \
                                                                diag['type']==0])
                self['loop_diagrams']=base_objects.DiagramList([diag for diag in value if \
                                                                diag['type']>0])
        else:
            return super(LoopAmplitude, self).set(name, value)

        return True

    def get(self, name):
        """Redefine get for the particular case of '*_diagrams' property"""

        if name == 'diagrams':
            if self['process']:            
                if self['born_diagrams'] == None:
                    self.generate_born_diagrams()
                if self['loop_diagrams'] == None:
                    self.generate_loop_diagrams()

        if name == 'born_diagrams':
            if self['born_diagrams'] == None:
                # Have not yet generated born diagrams for this process
                if self['process']:
                    self.generate_born_diagrams()
            
        if name=='diagrams':
            return (self['born_diagrams']+self['loop_diagrams'])
        else:
            return LoopAmplitude.__bases__[0].get(self, name)  #return the mother routine

    def generate_diagrams(self):
        """ Generates all diagrams relevant to this Loop Process """

        # Description of the algorithm to guess the leading contribution.
        # The summed weighted order of each diagram will be compared to 'target_weighted_order'
        # which acts as a threshold to decide which diagram to keep.
        # Here is an example on how MG5 sets the 'target_weighted_order'.
        #
        # In the sm process uu~ > dd~ [QCD, QED] with hierarch QCD=1, QED=2 we
        # would have at leading order contribution like
        #   (QED=4) , (QED=2, QCD=2) , (QCD=4)
        # leading to a summed weighted order of respectively 
        #   (4*2=8) , (2*2+2*1=6) , (4*1=4)
        # at NLO in QCD and QED we would have the following possible contributions
        #  (QED=6), (QED=4,QCD=2), (QED=2,QCD=4) and (QCD=6)
        # which translate into the following weighted orders, respectively
        #  12, 10, 8 and 6
        # So, now we take the largest weighted order at born level, 4, and add two
        # times the largest weight in the hierarchy among the order for which we
        # consider loop perturbation, in this case 2*2 wich gives us a 
        # target_weighted_order of 8. based on this we will now keep all born 
        # contributions and exclude the NLO contributions (QED=6) and (QED=4,QCD=2)

        print "valinfo:: START loop diag gen"

        # First generate the born diagram
        bornsuccessful = self.generate_born_diagrams()

        # The decision of wether the virtual must be squared against the born or the
        # virtual is now simply made based on wether there are borns or not.
        self['has_born'] = self['born_diagrams']!=[]
        hierarchy=self['process']['model']['order_hierarchy']            

        print "valinfo:: user input orders= ",self['process']['orders']

        # Now, we can further specify the orders for the loop amplitude. Those specified
        # by the user of course remain the same, increased by two if they are perturbed.
        # It is a temporary change that will be reverted after loop diagram generation.
        user_orders=copy.copy(self['process']['orders'])

        # Now if the user specified some squared order, we can use them as an upper bound
        # for the loop diagram generation
        if self['process']['squared_orders']:
            for order in self['process']['squared_orders']:
                if order.upper()!='WEIGHT' and order not in self['process']['orders']:
                    # If there is no born, the min order will simply be 0 as it should.
                    self['process']['orders'][order]=self['process']['squared_orders'][order]-\
                                                      self['born_diagrams'].get_min_order(order)
        
        # If the user had not specified any squared order, we will use the guessed weighted order
        # to assign a bound to the loop diagram order. Later we will check if the order deduced from
        # the max order appearing in the born diagrams is a better upper bound.
        # Of course, if no hierarchy is defined we directly jump to the step above.
        elif hierarchy and self['has_born']:
            
            if 'WEIGHTED' not in [key.upper() for key in self['process']['squared_orders'].keys()]:
                # Then we guess it from the born
                self['process']['squared_orders']['WEIGHTED']= \
                     2*(self['born_diagrams'].get_min_weighted_order(self['process']['model'])+
                        max([hierarchy[order] for order in self['process']['perturbation_couplings']]))

            # Now we know that the remaining weighted orders which can fit in
            # the loop diagram is (self['target_weighted_order']-min_born_weighted_order)
            # so for each perturbed order we just have to take that number divided
            # by its hierarchy weight to have the maximum allowed order for the 
            # loop diagram generation. Of course, we don't overwrite any order
            # already defined by the user.
            remaining_weight=self['process']['squared_orders']['WEIGHTED']-\
                               self['born_diagrams'].get_min_weighted_order(self['process']['model'])
            for order in hierarchy:
                if order not in self['process']['orders']:
                    self['process']['orders'][order]=int(remaining_weight/hierarchy[order])

        # Now for the remaining orders for which the user has not set squared orders neither
        # amplitude orders, then we use the max order encountered in the born (and add 2 if
        # this is a perturbed order). It might be that this upper bound is better than the one
        # guessed from the hierarchy
        for order in self['process']['model']['couplings']:
            neworder=self['born_diagrams'].get_max_order(order)
            if order in self['process']['perturbation_couplings']:
                neworder+=2
            if order not in self['process']['orders'].keys() or neworder<self['process']['orders'][order]:
                self['process']['orders'][order]=neworder

        # Finally we enforce the use of the orders specified for the born (augmented by two if perturbed) by
        # the user, no matter what was the best guess performed above.
        for order in user_orders.keys():
            if order in self['process']['perturbation_couplings']:
                self['process']['orders'][order]=user_orders[order]+2
            else:
                self['process']['orders'][order]=user_orders[order]
                
        print "valinfo:: orders used for loop generation = ",self['process']['orders']
        # Now we can generate the loop particles.
        totloopsuccessful=self.generate_loop_diagrams()
        # Reset the orders to their original specification by the user
        self['process']['orders']=user_orders

        # If there was no born, we will guess the WEIGHT squared order only now, based on the
        # minimum weighted order of the loop contributions, if it was not specified by the user
        if not self['has_born'] and not self['process']['squared_orders'] and hierarchy:
            # The WEIGHT squared order in this case is simply:
            # 2*(min(loop_diag_weighted_order)+
            #    max(order_weight_in_pert_list)-min(order_weight_in_pert_list))          
            pert_order_weights=[hierarchy[order] for order in \
                                self['process']['perturbation_couplings']]
            self['process']['squared_orders']['WEIGHT']=2*(\
              self['loop_diagrams'].get_min_weighted_order(self['process']['model'])+\
              max(pert_order_weights)-min(pert_order_weights))

        print "valinfo:: squared orders = ",self['process']['squared_orders']
        print "valinfo:: total number of loop_diagrams after diag gen. =", len(self['loop_diagrams'])

        # Now select only the loops corresponding to the perturbative orders asked for.
        self['loop_diagrams']=base_objects.DiagramList([diag for diag in self['loop_diagrams'] if \
                                set(diag.get_loop_orders(self['process']['model']).keys()).\
                                difference(set(self['process']['perturbation_couplings']))==set([])])

        print "valinfo:: total number of loop_diagrams pert recognition. =", len(self['loop_diagrams'])

        print "valinfo:: END loop diag gen"

        return (bornsuccessful or totloopsuccessful)

    def generate_born_diagrams(self):
        """ Generates all born diagrams relevant to this NLO Process """
            
        bornsuccessful, self['born_diagrams'] = \
          super(LoopAmplitude, self).generate_diagrams(True)
            
        return bornsuccessful

    def generate_loop_diagrams(self):
        """ Generates all loop diagrams relevant to this NLO Process """  
        
        # Reinitialize the loop diagram container
        self['loop_diagrams']=base_objects.DiagramList()
        totloopsuccessful=False

                    
        # Make sure to start with an empty l-cut particle list.
        self.lcutpartemployed=[]

        for order in self['process']['perturbation_couplings']:
            print "valinfo:: pert orders =",order           
            for part in [particle for particle in self['process']['model']['particles'] \
                         if particle.is_perturbing(order)]:
                if part.get_pdg_code() not in self.lcutpartemployed:
                    # First get rid of all the previously defined l-cut particles.
                    self['process']['legs']=[leg for leg in self['process']['legs'] if \
                                         not leg['loop_line']]

                    # Now create the two L-cut particles to add to the process.
                    # Remember that in the model only the particles should be tagged as 
                    # contributing to the a perturbation. Never the anti-particle.
                    print "valinfo:: using L-cut part=",part.get_name()
                    lcutone=base_objects.Leg({'id': part.get_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    lcuttwo=base_objects.Leg({'id': part.get_anti_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    self['process']['legs'].append(lcutone)
                    self['process']['legs'].append(lcuttwo)
                
                    # We generate the diagrams now
                    loopsuccessful, lcutdiaglist = super(LoopAmplitude, self).generate_diagrams(True)
                    self['loop_diagrams']+=lcutdiaglist

                    # Update the list of already employed L-cut particles such that we
                    # never use them again in loop particles
                    self.lcutpartemployed.append(part.get_pdg_code())
                    self.lcutpartemployed.append(part.get_anti_pdg_code())

                    print "valinfo:: len(loop_diagrams) for this L-cut part. =",len(lcutdiaglist)
                    # Accordingly update the totloopsuccessful tag
                    if loopsuccessful:
                        totloopsuccessful=True
        
        # Reset the l-cut particle list
        self.lcutpartemployed=[]

        return loopsuccessful

    def create_diagram(self, vertexlist):
        """ Return a LoopDiagram created."""
        return loop_base_objects.LoopDiagram({'vertices':vertexlist})

    def copy_leglist(self, leglist):
        """ Returns a DGLoopLeg list instead of the default copy_leglist
            defined in base_objects.Amplitude """

        dgloopleglist=base_objects.LegList()
        for leg in leglist:
            dgloopleglist.append(loop_base_objects.DGLoopLeg(leg))
        
        return dgloopleglist

    def convert_dgleg_to_leg(self, vertexdoublelist):
        """ Overloaded here to convert back all DGLoopLegs into Legs. """

        for vertexlist in vertexdoublelist:
            for vertex in vertexlist:
                if isinstance(vertex['legs'][0],loop_base_objects.DGLoopLeg):
                    continue
                vertex['legs'][:]=[leg.convert_to_leg() for leg in vertex['legs']]
        return True
    
    def get_combined_legs(self, legs, leg_vert_ids, number, state):
        """Create a set of new legs from the info given."""
      
        looplegs=[leg for leg in legs if leg['loop_line']]
        
        # Get rid of all tadpoles
        if(len([1 for leg in looplegs if leg['depth']==0])==2):
            return []

        # Correctly propagate the loopflow
        loopline=(len(looplegs)==1)    

        mylegs = []
        for i, (leg_id, vert_id) in enumerate(leg_vert_ids):
            # We can now create the set of possible merged legs.
            # However, we make sure that its PDG is not in the list of L-cut particles 
            # we already explored. If it is, we simply reject the diagram.
            if not loopline or not (leg_id in self.lcutpartemployed):             
                # Reminder: The only purpose of the "depth" flag is to get rid of (some, not all)
                # of the wave-function renormalization already during diagram generation.
                # We reckognize a wf renormalization diagram as follows:
                if len(legs)==2 and len(looplegs)==2:
                    # We have candidate
                    depths=(looplegs[0]['depth'],looplegs[1]['depth'])
                    if (0 in depths) and (-1 not in depths) and depths!=(0,0):   
                        # Check that the PDG of the outter particle in the wavefunction renormalization
                        # bubble is equal to the one of the inner particle.
                        if abs(leg_id) in depths:
                            continue

                depth=-1
                # When creating a loop leg from exactly to external legs, we set the depth
                # to the absolute value of the PDG of the external non-loop line.
                if len(legs)==2 and loopline and (legs[0]['depth'],legs[1]['depth'])==(0,0):
                    if not legs[0]['loop_line']:
                        depth=abs(legs[0]['id'])
                    else:
                        depth=abs(legs[1]['id'])
                # In case of two point interactions among two same particle (i.e same abs(PDG))
                # we propagate the existing depth
                if len(legs)==1 and abs(legs[0]['id'])==abs(leg_id):
                    depth=legs[0]['depth']
                # In all other cases we set the depth to -1 since no wave-function renormalization
                # diagram can arise from this side of the diagram construction.
                
                mylegs.append((loop_base_objects.DGLoopLeg({'id':leg_id,
                                    'number':number,
                                    'state':state,
                                    'from_group':True,
                                    'depth': depth,
                                    'loop_line': loopline}),
                                    vert_id))
        return mylegs

    def get_combined_vertices(self, legs, vert_ids):
        """Allow for selection of vertex ids."""
        
        looplegs=[leg for leg in legs if leg['loop_line']]
        nonlooplegs=[leg for leg in legs if not leg['loop_line']]
        
        # Get rid of all tadpoles
        if(len([1 for leg in looplegs if leg['depth']==0])==2):
            return []

        # Get rid of some wave-function renormalization diagrams already during
        # diagram generation already.In a similar manner as in get_combined_legs.
        if(len(legs)==3 and len(looplegs)==2):
            depths=(looplegs[0]['depth'],looplegs[1]['depth'])                    
            if (0 in depths) and (-1 not in depths) and depths!=(0,0):
                if nonlooplegs[0]['id'] in depths:
                    return []

        return vert_ids

#===============================================================================
# LoopMultiProcess
#===============================================================================
class LoopMultiProcess(diagram_generation.MultiProcess):
    """LoopMultiProcess: MultiProcess with loop features.
    """

    def get_amplitude_from_proc(self, proc):
        """ Return the correct amplitude type according to the characteristics of
            the process proc """
        return LoopAmplitude({"process": proc})


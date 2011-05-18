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


        # This is a bool controlling the verbosity of this subroutine only
        loopGenInfo=False

        if loopGenInfo: print "LoopGenInfo:: Start of loop diagram generation."

        # First generate the born diagram
        bornsuccessful = self.generate_born_diagrams()

        # The decision of wether the virtual must be squared against the born or the
        # virtual is now simply made based on wether there are borns or not.
        self['has_born'] = self['born_diagrams']!=[]
        hierarchy=self['process']['model']['order_hierarchy']            

        if loopGenInfo: print "LoopGenInfo:: User input born orders                  = ",self['process']['orders']
        if loopGenInfo: print "LoopGenInfo:: User input squared orders               = ",self['process']['squared_orders']
        if loopGenInfo: print "LoopGenInfo:: User input perturbation                 = ",self['process']['perturbation_couplings']

        # Now, we can further specify the orders for the loop amplitude. Those specified
        # by the user of course remain the same, increased by two if they are perturbed.
        # It is a temporary change that will be reverted after loop diagram generation.
        user_orders=copy.copy(self['process']['orders'])

        # Now if the user specified some squared order, we can use them as an upper bound
        # for the loop diagram generation
        for order, value in self['process']['squared_orders'].items():
            if order.upper()!='WEIGHT' and order not in self['process']['orders']:
                # If there is no born, the min order will simply be 0 as it should.                    
                bornminorder=self['born_diagrams'].get_min_order(order)
                if value>=0:
                    self['process']['orders'][order]=value-bornminorder 
                elif self['has_born']:
                    # This means the user want the leading if order=-1 or N^n Leading term if 
                    # order=-n. If there is a born diag, we can infer the necessary
                    # maximum order in the loop: bornminorder+2*(n-1).
                    # If there is no born diag, then we cannot say anything.
                    self['process']['orders'][order]=bornminorder+2*(-value-1)
        
        # If the user had not specified any fixed squared order other than WEIGHTED, 
        # we will use the guessed weighted order to assign a bound to the loop diagram order.
        # Later we will check if the order deduced from the max order appearing in the born 
        # diagrams is a better upper bound.
        # Of course, if no hierarchy is defined we directly jump to the next step.
        if not [1 for elem in self['process']['squared_orders'].items() if \
                elem[0].upper()!='WEIGHTED'] and hierarchy and self['has_born']:

            min_born_wgt=self['born_diagrams'].get_min_weighted_order(self['process']['model'])
            if 'WEIGHTED' not in [key.upper() for key in self['process']['squared_orders'].keys()]:
                # Then we guess it from the born
                self['process']['squared_orders']['WEIGHTED']= 2*(min_born_wgt+ \
                        max([hierarchy[order] for order in self['process']['perturbation_couplings']]))

            # Now we know that the remaining weighted orders which can fit in
            # the loop diagram is (self['target_weighted_order']-min_born_weighted_order)
            # so for each perturbed order we just have to take that number divided
            # by its hierarchy weight to have the maximum allowed order for the 
            # loop diagram generation. Of course, we don't overwrite any order
            # already defined by the user.
            trgt_wgt=self['process']['squared_orders']['WEIGHTED']-min_born_wgt
            # We also need the minimum number of vertices in the born.
            min_nvert=min([len([1 for vert in diag['vertices'] if vert['id']!=0]) \
                               for diag in self['born_diagrams']])
            # And the minimum weight for the ordered declared as perturbed
            min_pert=min([hierarchy[order] for order in self['process']['perturbation_couplings']])

            for order, value in hierarchy.items():
                if order not in self['process']['orders']:
                    # The four cases below come from a study of the maximal order needed in the 
                    # loop for the weighted order needed and the number of vertices available.
                    if order in self['process']['perturbation_couplings']:
                        if value!=1:
                            self['process']['orders'][order]=int((trgt_wgt-min_nvert-2)/(value-1))
                        else:
                            self['process']['orders'][order]=int(trgt_wgt)
                    else:
                        if value!=1:
                            self['process']['orders'][order]=int((trgt_wgt-min_nvert-2*min_pert)/(value-1))
                        else:
                            self['process']['orders'][order]=int(trgt_wgt-2*min_pert)

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
                
        if loopGenInfo: print "LoopGenInfo:: Orders used for loop generation         = ",self['process']['orders']        
        
        # Now we can generate the loop particles.
        totloopsuccessful=self.generate_loop_diagrams()
        # Reset the orders to their original specification by the user
        self['process']['orders']=user_orders

        # If there was no born, we will guess the WEIGHT squared order only now, based on the
        # minimum weighted order of the loop contributions, if it was not specified by the user
        if not self['has_born'] and not self['process']['squared_orders'] and hierarchy:         
            pert_order_weights=[hierarchy[order] for order in \
                                self['process']['perturbation_couplings']]
            self['process']['squared_orders']['WEIGHTED']=2*(\
              self['loop_diagrams'].get_min_weighted_order(self['process']['model'])+\
              max(pert_order_weights)-min(pert_order_weights))

        if loopGenInfo: print "LoopGenInfo:: Squared orders after treatment          = ",self['process']['squared_orders']
        if loopGenInfo: print "LoopGenInfo:: #Diags after diagram generation         = ",len(self['loop_diagrams'])        

        # Now select only the loops corresponding to the perturbative orders asked for.
        # First defined what are the set of particles allowed to run in the loop.
        allowedpart=[]
        for part in self['process']['model']['particles']:
            for order in self['process']['perturbation_couplings']:
                if part.is_perturbing(order):
                    allowedpart.append(part.get_pdg_code())
                    break
        
        newloopselection=base_objects.DiagramList()
        for diag in self['loop_diagrams']:
            # Now collect what are the coupling orders building the loop which are also perturbed order.        
            loop_orders=diag.get_loop_orders(self['process']['model'])
            pert_loop_order=set(loop_orders.keys()).intersection(set(self['process']['perturbation_couplings']))
            # Then make sure that the particle running in the loop for all diagrams belong to the set above.
            # Also make sure that there is at least one coupling order building the loop which is in the list
            # of the perturbed order.
            if (diag.get_loop_line_types()-set(allowedpart))==set() and \
              sum([loop_orders[order] for order in pert_loop_order])>=2:
                newloopselection.append(diag)
        self['loop_diagrams']=newloopselection   

        if loopGenInfo: print "LoopGenInfo:: #Diags after perturbation recognition   = ",len(self['loop_diagrams'])
        
        # The loop diagrams are filtered according to the 'squared_order' specification
        if self['has_born']:
            # If there are born diagrams the selection is simple
            self.check_squared_orders(self['process']['squared_orders'])
        else:
            # In case there is no born, we must make the selection of the loop diagrams based on themselves.
            # The minimum of the different orders used for the selections can possibly increase, after some
            # loop diagrams are selected out. So this check must be iterated until the number of diagrams
            # remaining is stable
            while True:
                nloopdiag_remaining=len(self['loop_diagrams'])
                self.check_squared_orders(self['process']['squared_orders'])
                if len(self['loop_diagrams'])==nloopdiag_remaining:
                    break

        if loopGenInfo: print "LoopGenInfo:: #Diags after squared_orders constraints = ",len(self['loop_diagrams'])                
        # Now the loop diagrams are tagged and filtered for redundancy.
        tag_selected=[]
        loop_basis=base_objects.DiagramList()
        for i, diag in enumerate(self['loop_diagrams']):
            diag.tag(self['structure_repository'],len(self['process']['legs'])+1,len(self['process']['legs'])+2,\
                     self['process'])
            # Make sure not to consider wave-function renormalization, tadpoles, or redundant diagrams
            if not diag.is_wvf_correction(self['structure_repository'],self['process']['model']) \
               and not diag.is_tadpole() and diag['canonical_tag'] not in tag_selected:
                loop_basis.append(diag)
                tag_selected.append(diag['canonical_tag'])
        self['loop_diagrams']=loop_basis

        if loopGenInfo: print "================================================================== "
        if loopGenInfo: print "|| LoopGenInfo:: #Diags after filtering                  = ",len(self['loop_diagrams'])," ||"
        if loopGenInfo: print "================================================================== "        
        if loopGenInfo: print "LoopGenInfo:: # of different structures identified    = ",len(self['structure_repository'])
        if loopGenInfo: print "LoopGenInfo:: End of loop diagram generation."             

        return (bornsuccessful or totloopsuccessful)

    def generate_born_diagrams(self):
        """ Generates all born diagrams relevant to this NLO Process """
            
        bornsuccessful, self['born_diagrams'] = \
          super(LoopAmplitude, self).generate_diagrams(True)
            
        return bornsuccessful

    def generate_loop_diagrams(self):
        """ Generates all loop diagrams relevant to this NLO Process """  
       
        # This is a bool controlling the verbosity of this subroutine only
        partGenInfo=False

        # Reinitialize the loop diagram container
        self['loop_diagrams']=base_objects.DiagramList()
        totloopsuccessful=False
                    
        # Make sure to start with an empty l-cut particle list.
        self.lcutpartemployed=[]

        for order in self['process']['perturbation_couplings']:
            if partGenInfo: print "partGenInfo:: Perturbation coupling generated         = ",order
            for part in [particle for particle in self['process']['model']['particles'] \
                         if particle.is_perturbing(order)]:
                if part.get_pdg_code() not in self.lcutpartemployed:
                    # First create the two L-cut particles to add to the process.
                    # Remember that in the model only the particles should be tagged as 
                    # contributing to the a perturbation. Never the anti-particle.
                    # We chose here a specific orientation for the loop momentum flow:
                    # It goes OUT of lcutone and IN lcuttwo.
                    if partGenInfo: print "partGenInfo:: L-cut particle generated                = ",part.get_name()
                    lcutone=base_objects.Leg({'id': part.get_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    lcuttwo=base_objects.Leg({'id': part.get_anti_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    self['process']['legs'].append(lcutone)
                    # WARNING, it is important for the tagging to notice here that lcuttwo
                    # is the last leg in the process list of legs and will therefore carry
                    # the highest 'number' attribute as required to insure that it will 
                    # never be 'propagated' to any output leg.
                    self['process']['legs'].append(lcuttwo)
                
                    # We generate the diagrams now
                    loopsuccessful, lcutdiaglist = super(LoopAmplitude, self).generate_diagrams(True)
                    
                    # Now get rid of all the previously defined l-cut particles.
                    self['process']['legs']=base_objects.LegList([leg for leg in self['process']['legs'] \
                                            if not leg['loop_line']])

                    # The correct L-cut type is specified
                    for diag in lcutdiaglist:
                        diag.set('type',part.get_pdg_code())
                    self['loop_diagrams']+=lcutdiaglist

                    # Update the list of already employed L-cut particles such that we
                    # never use them again in loop particles
                    self.lcutpartemployed.append(part.get_pdg_code())
                    self.lcutpartemployed.append(part.get_anti_pdg_code())

                    if partGenInfo: print "partGenInfo:: #Diags generated w/ this L-cut particle = ",len(lcutdiaglist)
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
                if not isinstance(vertex['legs'][0],loop_base_objects.DGLoopLeg):
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

        #Ease the access to the model
        model=self['process']['model']

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
                        if leg_id in depths:
                            continue
                
                # If depth is not 0 because of being an external leg and not the propagated PDG, then we
                # set it to -1 so that from that point we are sure the diagram will not be reckognized as
                # a wave-function renormalization.
                depth=-1
                # When creating a loop leg from exactly two external legs, we set the depth
                # to the PDG of the external non-loop line.
                if len(legs)==2 and loopline and (legs[0]['depth'],legs[1]['depth'])==(0,0):
                    if not legs[0]['loop_line']:
                        depth=legs[0]['id']
                    else:
                        depth=legs[1]['id']
                # In case of two point interactions among two same particle
                # we propagate the existing depth
                if len(legs)==1 and legs[0]['id']==leg_id:
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

    # Helper function

    def check_squared_orders(self, sq_order_constrains):
        """ Filters loop diagrams according to the constraints on the squared orders
            in argument and wether the process has a born or not. """

        diagRef=base_objects.DiagramList()
        if self['has_born']:
            diagRef=self['born_diagrams']
        else:
            diagRef=self['loop_diagrams']

        for order, value in sq_order_constrains.items():
            if order.upper()=='WEIGHTED':
                max_wgt=value-diagRef.get_min_weighted_order(self['process']['model'])
                self['loop_diagrams']=base_objects.DiagramList([diag for diag in self['loop_diagrams'] if \
                                        diag.get_weighted_order(self['process']['model'])<=max_wgt])
            else:
                max_order = 0
                if value>=0:
                    # Fixed squared order
                    max_order=value-diagRef.get_min_order(order)
                else:
                    # ask for the N^(-value) Leading Order in tha coupling
                    max_order=diagRef.get_min_order(order)+2*(-value-1)                    
                self['loop_diagrams']=base_objects.DiagramList([diag for diag in self['loop_diagrams'] if \
                                        diag.get_order(order)<=max_order])

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


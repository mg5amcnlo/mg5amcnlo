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
        self['loop_UVCT_diagrams'] = base_objects.DiagramList()
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
                if not isinstance(diag,loop_base_objects.LoopDiagram) and \
                   not isinstance(diag,loop_base_objects.LoopUVCTDiagram):
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
            super(LoopAmplitude, self).filter(name, value)

        return True

    def set(self, name, value):
        """Redefine set for the particular case of diagrams"""

        if name == 'diagrams':
            if self.filter(name, value):
                self['born_diagrams']=base_objects.DiagramList([diag for diag in value if \
                  not isinstance(diag,loop_base_objects.LoopUVCTDiagram) and diag['type']==0])
                self['loop_diagrams']=base_objects.DiagramList([diag for diag in value if \
                  not isinstance(diag,loop_base_objects.LoopUVCTDiagram) and diag['type']!=0])
                self['loop_UVCT_diagrams']=base_objects.DiagramList([diag for diag in value if \
                  isinstance(diag,loop_base_objects.LoopUVCTDiagram)])
                
        else:
            return super(LoopAmplitude, self).set(name, value)

        return True

    def get(self, name):
        """Redefine get for the particular case of '*_diagrams' property"""

        if name == 'diagrams':
            if self['process'] and self['loop_diagrams'] == None:            
                self.generate_diagrams()
            return base_objects.DiagramList(self['born_diagrams']+\
                                            self['loop_diagrams']+\
                                            self['loop_UVCT_diagrams'])
        
        if name == 'born_diagrams':
            if self['born_diagrams'] == None:
                # Have not yet generated born diagrams for this process
                if self['process']['has_born']:
                    if self['process']:
                        self.generate_born_diagrams()
                else:
                        self['born_diagrams']=base_objects.DiagramList()                  
            
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
        
        old_logger_level=logger.getEffectiveLevel()
        if loopGenInfo:
            # Disable the basic info if using the extended one
            logger.setLevel(logging.WARNING)

        logger.info("Generating %s " % self['process'].nice_string().replace('Process', 'process'))

        if loopGenInfo: print "LoopGenInfo:: Loop diagram generation for process",self['process'].nice_string()

        # First generate the born diagram if the user asked for it
        if self['process']['has_born']:
            bornsuccessful = self.generate_born_diagrams()
            if loopGenInfo: print "LoopGenInfo:: # born diagrams after first generation  = ",len(self['born_diagrams'])        
        else:
            self['born_diagrams'] = base_objects.DiagramList()
            bornsuccessful = True
            if loopGenInfo: print "LoopGenInfo:: # born diagrams generation skipped by user request."        

        # The decision of wether the virtual must be squared against the born or the
        # virtual is made based on wether there are borns or not unless the user
        # already asked for the loop squard.
        if self['process']['has_born']:
            self['process']['has_born'] = self['born_diagrams']!=[]
        hierarchy=self['process']['model']['order_hierarchy']            

        if loopGenInfo: print "LoopGenInfo:: User input born orders                  = ",self['process']['orders']
        if loopGenInfo: print "LoopGenInfo:: User input squared orders               = ",self['process']['squared_orders']
        if loopGenInfo: print "LoopGenInfo:: User input perturbation                 = ",self['process']['perturbation_couplings']

        # Now, we can further specify the orders for the loop amplitude. Those specified
        # by the user of course remain the same, increased by two if they are perturbed.
        # It is a temporary change that will be reverted after loop diagram generation.
        user_orders=copy.copy(self['process']['orders'])

        # Before proceeding with the loop contribution. We must make sure that
        # the born diagram generated factorize the same power of all coupling
        # orders which are not perturbed. If this is not true, then it is very
        # cumbersome to get the real radiation contribution correct and consistent
        # with the computations of the virtuals.
        if self['process']['has_born']:
            for order in self['process']['model']['coupling_orders']:
                if order not in self['process']['perturbation_couplings']:
                    order_power=self['born_diagrams'][0].get_order(order)
                    for diag in self['born_diagrams'][1:]:
                        if diag.get_order(order)!=order_power:
                            logger.warning("All born diagrams do not factorize "+\
                              "the same power of the order %s which is not"%order+\
                              " perturbed. This is potentially dangerous as the "+\
                              " real-emission diagrams will not be consistent "+\
                              "with these virtual contributions.")
                            break

        # Now if the user specified some squared order, we can use them as an upper bound
        # for the loop diagram generation
        for order, value in self['process']['squared_orders'].items():
            if order.upper()!='WEIGHTED' and order not in self['process']['orders']:
                # If there is no born, the min order will simply be 0 as it should.                    
                bornminorder=self['born_diagrams'].get_min_order(order)
                if value>=0:
                    self['process']['orders'][order]=value-bornminorder 
                elif self['process']['has_born']:
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
        if (not [1 for elem in self['process']['squared_orders'].items() if \
                elem[0].upper()!='WEIGHTED']) and hierarchy and self['process']['has_born']:

            min_born_wgt=self['born_diagrams'].get_min_order('WEIGHTED')
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
            for order in self['process']['model']['coupling_orders']:
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
        
        # Make sure to warn the user if we already possibly excluded mixed order loops
        # by smartly setting up the orders
        if self['process']['has_born']:
            for order in self['process']['model']['coupling_orders']:
                if order not in self['process']['perturbation_couplings']:
                    if self['process']['orders'][order]< \
                       self['born_diagrams'].get_max_order(order):
                        logger.warning("Some loop diagrams contributing to this process"+\
                          " might be discarded because they are not pure "+\
                          '+'.join(self['process']['perturbation_couplings'])+\
                          " perturbation. Make sure there are none or that you did not"+\
                          " want to include them.")
                        break
                    
        # Now we can generate the loop particles.
        totloopsuccessful=self.generate_loop_diagrams()
        
        # If there is no born neither loop diagrams, return now.
        if not self['process']['has_born'] and not self['loop_diagrams']:
            return False

        # We add here the UV renormalization contribution represented
        # here as a LoopUVCTDiagram.
        # It is done before the square order selection as it is possible that
        # some UV-renormalization diagrams are removed as well.
        if self['process']['has_born']:
            self.setUVCT()
        if loopGenInfo: print "LoopGenInfo:: #UVCTDiags generated                    = ",len(self['loop_UVCT_diagrams'])                        
        
        # Reset the orders to their original specification by the user
        self['process']['orders'].clear()
        self['process']['orders'].update(user_orders)

        # If there was no born, we will guess the WEIGHT squared order only now, based on the
        # minimum weighted order of the loop contributions, if it was not specified by the user
        if not self['process']['has_born'] and not self['process']['squared_orders'] and hierarchy:         
            pert_order_weights=[hierarchy[order] for order in \
                                self['process']['perturbation_couplings']]
            self['process']['squared_orders']['WEIGHTED']=2*(\
              self['loop_diagrams'].get_min_order('WEIGHTED')+\
              max(pert_order_weights)-min(pert_order_weights))

        if loopGenInfo: print "LoopGenInfo:: Squared orders after treatment          = ",self['process']['squared_orders']
        if loopGenInfo: print "LoopGenInfo:: #Diags after diagram generation         = ",len(self['loop_diagrams'])        

        # Now select only the loops corresponding to the perturbative orders asked for.
        # First define what are the set of particles allowed to run in the loop.
        allowedpart=[]
        for part in self['process']['model']['particles']:
            for order in self['process']['perturbation_couplings']:
                if part.is_perturbating(order,self['process']['model']):
                    allowedpart.append(part.get_pdg_code())
                    break
        
        newloopselection=base_objects.DiagramList()
        warned=False
        for diag in self['loop_diagrams']:
            # Now collect what are the coupling orders building the loop which are also perturbed order.        
            loop_orders=diag.get_loop_orders(self['process']['model'])
            pert_loop_order=set(loop_orders.keys()).intersection(set(self['process']['perturbation_couplings']))
            # Then make sure that the particle running in the loop for all diagrams belong to the set above.
            # Also make sure that there is at least one coupling order building the loop which is in the list
            # of the perturbed order.
            valid_diag=True
            if (diag.get_loop_line_types()-set(allowedpart))!=set():
                valid_diag=False
                if not warned:
                    logger.warning("Some loop diagrams contributing to this process"+\
                      " are discarded because they are not pure "+\
                      '+'.join(self['process']['perturbation_couplings'])+\
                      " perturbation. Make sure you did not want to include them.")
                    warned=True
            if sum([loop_orders[order] for order in pert_loop_order])<2:
                valid_diag=False
            if valid_diag:
                newloopselection.append(diag)

        self['loop_diagrams']=newloopselection

        # The loop diagrams are filtered according to the 'squared_order' 
        # specification
        if self['process']['has_born']:
            # If there are born diagrams the selection is simple
            self.check_squared_orders(self['process']['squared_orders'])
        else:
            # In case there is no born, we must make the selection of the loop 
            # diagrams based on themselves. The minimum of the different orders 
            # used for the selections can possibly increase, after some loop 
            # diagrams are selected out. So this check must be iterated until 
            # the number of diagrams remaining is stable
            while True:
                nloopdiag_remaining=len(self['loop_diagrams'])
                self.check_squared_orders(self['process']['squared_orders'])
                if len(self['loop_diagrams'])==nloopdiag_remaining:
                    break

        if loopGenInfo: print "LoopGenInfo:: #Diags after squared_orders constraints = ",len(self['loop_diagrams'])                
        if loopGenInfo: print "LoopGenInfo:: # born diagrams after s.o constraints   = ",len(self['born_diagrams'])        
        if loopGenInfo: print "LoopGenInfo:: #UVCTDiags after s.o. constraints       = ",len(self['loop_UVCT_diagrams'])

        # Now the loop diagrams are tagged and filtered for redundancy.
        tag_selected=[]
        loop_basis=base_objects.DiagramList()
        for diag in self['loop_diagrams']:
            diag.tag(self['structure_repository'],len(self['process']['legs'])+1\
                                ,len(self['process']['legs'])+2,self['process'])
            # Make sure not to consider wave-function renormalization, tadpoles, or redundant diagrams
            if not diag.is_wf_correction(self['structure_repository'],self['process']['model']) \
               and not diag.is_tadpole() and diag['canonical_tag'] not in tag_selected:
                loop_basis.append(diag)
                tag_selected.append(diag['canonical_tag'])
        self['loop_diagrams']=loop_basis

        # If there is no born neither loop diagrams after filtering, return now.
        if not self['process']['has_born'] and not self['loop_diagrams']:
            return False

        # Set the necessary UV/R2 CounterTerms for each loop diagram generated
        self.setLoopCT_vertices()
        
        nLoopDiag = 0
        nCT={'UV':0,'R2':0}
        for ldiag in self['loop_UVCT_diagrams']:
            nCT[ldiag['type'][:2]]+=len(ldiag['UVCT_couplings'])
        for ldiag in self['loop_diagrams']:
            nLoopDiag+=1
            nCT['UV']+=len(ldiag.get_CT(self['process']['model'],'UV'))
            nCT['R2']+=len(ldiag.get_CT(self['process']['model'],'R2'))         
            
        logger.info("Generated %d loop diagrams"%nLoopDiag+\
                    " with %d R2"%nCT['R2']+\
                    " and %d UV counterterms"%nCT['UV'])
        
        if loopGenInfo: print "================================================================== "
        if loopGenInfo: print "|| LoopGenInfo:: #Diags after filtering                  = ",len(self['loop_diagrams'])," ||"
        if loopGenInfo: print "================================================================== "        
        if loopGenInfo: print "LoopGenInfo:: # of different structures identified    = ",len(self['structure_repository'])
        if loopGenInfo: print "LoopGenInfo:: End of loop diagram generation."             

        if loopGenInfo:
            logger.setLevel(old_logger_level)

        return (bornsuccessful or totloopsuccessful)

    def generate_born_diagrams(self):
        """ Generates all born diagrams relevant to this NLO Process """

        bornsuccessful, self['born_diagrams'] = \
          super(LoopAmplitude, self).generate_diagrams(True)
        logger.info("Generated %d born diagrams" % len(self['born_diagrams']))
            
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
            lcutPart=[particle for particle in self['process']['model']['particles'] \
                         if (particle.is_perturbating(order, self['process']['model']) and particle.get_pdg_code() not in 
                             self['process']['forbidden_particles'])]
            #print "lcutPart=",[part.get('name') for part in lcutPart]
            for part in lcutPart:
                if part.get_pdg_code() not in self.lcutpartemployed:
                    # First create the two L-cut particles to add to the process.
                    # Remember that in the model only the particles should be tagged as 
                    # contributing to the a perturbation. Never the anti-particle.
                    # We chose here a specific orientation for the loop momentum flow
                    # , say going IN lcutone and OUT lcuttwo. We also define here
                    # the 'positive' loop fermion flow by always setting lcutone
                    # to be a particle and lcuttwo the corresponding anti-particle
                    if partGenInfo: print "partGenInfo:: L-cut particle generated                = ",part.get_name()
                    lcutone=base_objects.Leg({'id': part.get_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    lcuttwo=base_objects.Leg({'id': part.get_anti_pdg_code(),
                                              'state': True,
                                              'loop_line': True})
                    self['process'].get('legs').extend([lcutone,lcuttwo])
                    # WARNING, it is important for the tagging to notice here that lcuttwo
                    # is the last leg in the process list of legs and will therefore carry
                    # the highest 'number' attribute as required to insure that it will 
                    # never be 'propagated' to any output leg.
                                    
                    # We generate the diagrams now
                    loopsuccessful, lcutdiaglist = super(LoopAmplitude, self).generate_diagrams(True)

                    # Now get rid of all the previously defined l-cut particles.
                    leg_to_remove=[leg for leg in self['process']['legs'] \
                                            if leg['loop_line']]
                    for leg in leg_to_remove:
                        self['process']['legs'].remove(leg)

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

    def setUVCT(self):
        """ Scan all born diagrams and add for each all the corresponding UV 
        counterterms. It creates one LoopUVCTDiagram per born diagram and set
        of possible coupling_order (so that QCD and QED wavefunction corrections
        are not in the same LoopUVCTDiagram for example). Notice that this takes
        care only of the UV counterterm which factorize with the born and the
        other contributions like the UV mass renormalization are added in the
        function setLoopCTVertices"""


        #  ============================================
        #     Vertex renormalization
        #  ============================================

        # The following lists the UV interactions potentially giving UV counterterms
        # (The UVmass interactions is accounted for like the R2s)
        UVCTvertex_interactions = base_objects.InteractionList()
        for inter in self['process']['model']['interactions'].get_UV():
            if not inter.is_UVmass() and len(inter['particles'])>1 and \
              inter.is_perturbating(self['process']['perturbation_couplings']) \
              and (set(inter['orders'].keys()).intersection(\
               set(self['process']['perturbation_couplings'])))!=set([]) and \
              (any([set(loop_parts).intersection(set(self['process']\
                   ['forbidden_particles']))==set([]) for loop_parts in \
                   inter.get('loop_particles')]) or \
                   inter.get('loop_particles')==[[]]):
                UVCTvertex_interactions.append(inter)
        
        # Temporarly give the tagging order 'UVCT_SPECIAL' to those interactions
        self['process']['model'].get('order_hierarchy')['UVCT_SPECIAL']=0
        self['process']['model'].get('coupling_orders').add('UVCT_SPECIAL')
        for inter in UVCTvertex_interactions:
            neworders=copy.copy(inter.get('orders'))
            neworders['UVCT_SPECIAL']=1
            inter.set('orders',neworders)
        # Refresh the model interaction dictionary while including those special 
        # interactions
        self['process']['model'].actualize_dictionaries(useUVCT=True)
        
        # Generate the UVCTdiagrams (born diagrams with 'UVCT_SPECIAL'=0 order 
        # will be generated along)
        self['process']['orders']['UVCT_SPECIAL']=1      
        
        UVCTsuccessful, UVCTdiagrams = \
          super(LoopAmplitude, self).generate_diagrams(True)
        
        for UVCTdiag in UVCTdiagrams:
            if UVCTdiag.get_order('UVCT_SPECIAL')==1:
                newUVCTDiag = loop_base_objects.LoopUVCTDiagram({\
                  'vertices':copy.deepcopy(UVCTdiag['vertices'])})
                UVCTinter = newUVCTDiag.get_UVCTinteraction(self['process']['model'])
                newUVCTDiag.set('type',UVCTinter.get('type'))
                # This interaction counter-term must be accounted for as many times
                # as they are list of loop_particles defined and allowed for by
                # the process.
                newUVCTDiag.get('UVCT_couplings').append((len([1 for loop_parts \
                  in UVCTinter.get('loop_particles') if set(loop_parts).intersection(\
                  set(self['process']['forbidden_particles']))==set([])])) if
                  loop_parts!=[[]] else  1)
                self['loop_UVCT_diagrams'].append(newUVCTDiag)

        # Remove the additional order requirement in the born orders for this
        # process
        del self['process']['orders']['UVCT_SPECIAL']
        # Remove the fake order added to the selected UVCT interactions
        del self['process']['model'].get('order_hierarchy')['UVCT_SPECIAL']
        self['process']['model'].get('coupling_orders').remove('UVCT_SPECIAL')
        for inter in UVCTvertex_interactions:
            del inter.get('orders')['UVCT_SPECIAL']     
        # Revert the model interaction dictionaries to default
        self['process']['model'].actualize_dictionaries(useUVCT=False)
        
        # Set the correct orders to the loop_UVCT_diagrams
        for UVCTdiag in self['loop_UVCT_diagrams']:
            UVCTdiag.calculate_orders(self['process']['model'])        
        
        #  ============================================
        #     Wavefunction renormalization
        #  ============================================
        
        if not self['process']['has_born']:
            return UVCTsuccessful

        # We now scan each born diagram, adding the necessary wavefunction
        # renormalizations
        for bornDiag in self['born_diagrams']:
            # This dictionary takes for keys the tuple 
            # (('OrderName1',power1),...,('OrderNameN',powerN) representing
            # the power brought by the counterterm and the value is the
            # corresponding LoopUVCTDiagram.
            # The last entry is of the form ('EpsilonOrder', value) to put the 
            # contribution of each different EpsilonOrder to different
            # LoopUVCTDiagrams.
            LoopUVCTDiagramsAdded={}
            for leg in self['process']['legs']:
                counterterm=self['process']['model'].get_particle(abs(leg['id'])).\
                            get('counterterm')
                for key, value in counterterm.items():
                    if key[0] in self['process']['perturbation_couplings']:
                        for laurentOrder, CTCoupling in value.items():
                            # Create the order key of the UV counterterm
                            orderKey=[(key[0],2),]
                            orderKey.sort()
                            orderKey.append(('EpsilonOrder',-laurentOrder))
                            CTCouplings=[CTCoupling for loop_parts in key[1] if
                              set(loop_parts).intersection(set(self['process']\
                              ['forbidden_particles']))==set([])]
                            if CTCouplings!=[]:
                                try:
                                    LoopUVCTDiagramsAdded[tuple(orderKey)].get(\
                                      'UVCT_couplings').extend(CTCouplings)
                                except KeyError:
                                    LoopUVCTDiagramsAdded[tuple(orderKey)]=\
                                      loop_base_objects.LoopUVCTDiagram({\
                                        'vertices':copy.deepcopy(bornDiag['vertices']),
                                        'type':'UV'+('' if laurentOrder==0 else 
                                          str(-laurentOrder)+'eps'),
                                        'UVCT_orders':{key[0]:2},
                                        'UVCT_couplings':CTCouplings})

            for LoopUVCTDiagram in LoopUVCTDiagramsAdded.values():
                LoopUVCTDiagram.calculate_orders(self['process']['model'])
                self['loop_UVCT_diagrams'].append(LoopUVCTDiagram)

        return UVCTsuccessful

    def setLoopCT_vertices(self):
        """ Scan each loop diagram and recognizes what are the R2/UVmass CounterTerms
            associated to them """

        # We first create a base dictionary with as a key (tupleA,tupleB). For each
        # R2/UV interaction, tuple B is the ordered tuple of the loop particles 
        # (not anti-particles, so that the PDG is always positive!) listed in its 
        # loop_particles attribute. Tuple A is the ordered tuple of external particles PDGs.
        # making up this interaction. The values of the dictionary are a list of the 
        # interaction ID having the same key above.
        CT_interactions = {}
        for inter in self['process']['model']['interactions']:
            if inter.is_UVmass() or inter.is_R2() and len(inter['particles'])>1 \
               and inter.is_perturbating(self['process']['perturbation_couplings']):
                # This interaction might have several possible loop particles yielding
                # the same counterterm. So we add this interaction ID for each entry
                # in the list loop_particles.
                for lparts in inter['loop_particles']:      
                    keya=copy.copy(lparts)
                    keya.sort()
                    keyb=[part.get_pdg_code() for part in inter['particles']]
                    keyb.sort()
                    key=(tuple(keyb),tuple(keya))
                    try:
                        CT_interactions[key].append(inter['id'])
                    except KeyError:
                        CT_interactions[key]=[inter['id'],]
        
        # The dictionary CT_added keeps track of what are the CounterTerms already
        # added and prevents us from adding them again. For instance, the fermion
        # boxes with four external gluons exist in 6 copies (with different 
        # crossings of the external legs each time) and the corresponding R2 must
        # be added only once. The key of this dictionary characterizing the loop
        # is (tupleA,tupleB). Tuple A is made from the list of the ID of the external
        # structures attached to this loop and tuple B from list of the pdg of the
        # particles building this loop.
        
        CT_added = {}

        for diag in self['loop_diagrams']:
            # First build the key from this loop for the CT_interaction dictionary 
            # (Searching Key) and the key for the CT_added dictionary (tracking Key)
            searchingKeyA=[]
            searchingKeyB=[]
            trackingKeyA=[]
            for tagElement in diag['canonical_tag']:
                for structID in tagElement[1]:
                    trackingKeyA.append(structID)
                    searchingKeyA.append(self['process']['model'].get_particle(\
                        self['structure_repository'][structID]['binding_leg']['id']).\
                        get_pdg_code())
                searchingKeyB.append(self['process']['model'].get_particle(\
                    tagElement[0]).get('pdg_code'))
            searchingKeyA.sort()
            # We do not repeat particles present many times in the loop
            searchingKeyB=list(set(searchingKeyB))
            searchingKeyB.sort()
            trackingKeyA.sort()
            # There are two kind of keys, the ones defining the loop
            # particles and the ones only specifying the external legs.
            searchingKeySimple=(tuple(searchingKeyA),())
            searchingKeyLoopPart=(tuple(searchingKeyA),tuple(searchingKeyB))
            trackingKeySimple=(tuple(trackingKeyA),())
            trackingKeyLoopPart=(tuple(trackingKeyA),tuple(searchingKeyB))
            # Now we look for a CT which might correspond to this loop by looking
            # for its searchingKey in CT_interactions

            #print "I have the following CT_interactions=",CT_interactions  
            try:
                CTIDs=copy.copy(CT_interactions[searchingKeySimple])
            except KeyError:
                CTIDs=[]
            try:
                CTIDs.extend(copy.copy(CT_interactions[searchingKeyLoopPart]))
            except KeyError:
                pass
            if not CTIDs:
                continue
            # We have found some CT interactions corresponding to this loop
            # so we must make sure we have not included them already
            try:    
                usedIDs=copy.copy(CT_added[trackingKeySimple])
            except KeyError:
                usedIDs=[]
            try:    
                usedIDs.extend(copy.copy(CT_added[trackingKeyLoopPart]))
            except KeyError:
                pass    

            for CTID in CTIDs:
                # Make sure it has not been considered yet and that the loop orders match
                if CTID not in usedIDs and diag.get_loop_orders(self['process']['model'])==\
                  self['process']['model']['interaction_dict'][CTID]['orders']:
                    # Create the amplitude vertex corresponding to this CT
                    # and add it to the LoopDiagram treated.
                    CTleglist = base_objects.LegList()
                    for tagElement in diag['canonical_tag']:
                        for structID in tagElement[1]:
                            CTleglist.append(self['structure_repository'][structID]['binding_leg'])
                    CTVertex = base_objects.Vertex({'id':CTID, \
                                                    'legs':CTleglist})
                    diag['CT_vertices'].append(CTVertex)
                    # Now add this CT vertex to the CT_added dictionary so that
                    # we are sure it will not be double counted
                    if self['process']['model']['interaction_dict'][CTID]['loop_particles']==[[]]:
                        try:
                            CT_added[trackingKeySimple].append(CTID)
                        except KeyError:
                            CT_added[trackingKeySimple] = [CTID, ]
                    else:
                        try:
                            CT_added[trackingKeyLoopPart].append(CTID)
                        except KeyError:
                            CT_added[trackingKeyLoopPart] = [CTID, ]

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
        AllLoopDiagrams=base_objects.DiagramList(self['loop_diagrams']+\
                                                 self['loop_UVCT_diagrams'])
        if self['process']['has_born']:
            diagRef=self['born_diagrams']
        else:
            diagRef=AllLoopDiagrams

        for order, value in sq_order_constrains.items():
            if order.upper()=='WEIGHTED':
                max_wgt=value-diagRef.get_min_order('WEIGHTED')
                AllLoopDiagrams=base_objects.DiagramList([diag for diag in\
                  AllLoopDiagrams if diag.get_order('WEIGHTED')<=max_wgt])
            else:
                max_order = 0
                if value>=0:
                    # Fixed squared order
                    max_order=value-diagRef.get_min_order(order)
                else:
                    # ask for the N^(-value) Leading Order in tha coupling
                    max_order=diagRef.get_min_order(order)+2*(-value-1)                    
                AllLoopDiagrams=base_objects.DiagramList([diag for diag in \
                    AllLoopDiagrams if diag.get_order(order)<=max_order])
        
        self['loop_diagrams']=[diag for diag in AllLoopDiagrams if not \
            isinstance(diag,loop_base_objects.LoopUVCTDiagram)]
        self['loop_UVCT_diagrams']=[diag for diag in AllLoopDiagrams if \
            isinstance(diag,loop_base_objects.LoopUVCTDiagram)]
        
#===============================================================================
# LoopMultiProcess
#===============================================================================
class LoopMultiProcess(diagram_generation.MultiProcess):
    """LoopMultiProcess: MultiProcess with loop features.
    """

    @classmethod
    def get_amplitude_from_proc(cls, proc):
        """ Return the correct amplitude type according to the characteristics of
            the process proc """
        return LoopAmplitude({"process": proc})


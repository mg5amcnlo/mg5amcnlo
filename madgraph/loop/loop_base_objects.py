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

"""Definitions of all basic objects with extra features to treat loop diagrams"""

import copy
import itertools
import logging
import numbers
import os
import re
import madgraph.core.color_algebra as color
import madgraph.core.base_objects as base_objects
from madgraph import MadGraph5Error, MG5DIR

logger = logging.getLogger('madgraph.loop_base_objects')

#===============================================================================
# LoopDiagram
#===============================================================================
class LoopDiagram(base_objects.Diagram):
    """LoopDiagram: Contains an additional tag to uniquely identify the diagram
       if it contains a loop. Also has many additional functions useful only
       for loop computations.
       """

    def default_setup(self):
        """Default values for all properties"""

        super(LoopDiagram,self).default_setup()
        # This tag uniquely define a loop particle. It is not used for born, 
        # R2 and UV diagrams. It is only a list of integers, so not too
        # heavy to store. It is necessary to store it because it useful both
        # during diagram generation and HELAS output.
        self['tag'] = None
        # This information is in principle recoverable from the VertexList but
        # it is more information to store it as a single integer. It is 0 for
        # a born diagram and it is the (positive) PDG of the (particle, not
        # anti-particle) L-cut particle for a loop diagram.
        self['type'] = 0

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'tag':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid tag" % str(value)
            else:
                for item in value:
                    if (len(item)!=2 or not isinstance(item[0],int) or \
                        not isinstance(item[1],int)):
                        raise self.PhysicsObjectError, \
                            "%s is not a valid tag" % str(value)

        if name == 'type':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)

        else:
            return super(LoopDiagram, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        
        return ['vertices', 'orders', 'type', 'tag']

    def nice_string(self):
        """Returns a nicely formatted string of the diagram content."""
        mystr=''
        if self['tag']:
            mystr = mystr+'tag: '+str(self['tag'])+'\n'
        if self['vertices']:
            mystr = mystr+'('
            for vert in self['vertices']:
                mystr = mystr + '('
                for leg in vert['legs'][:-1]:
                    mystr = mystr + str(leg['number']) + '(%s)' % str(leg['id']) + ','

                if self['vertices'].index(vert) < len(self['vertices']) - 1:
                    # Do not want ">" in the last vertex
                    mystr = mystr[:-1] + '>'
                mystr = mystr + str(vert['legs'][-1]['number']) + '(%s)' % str(vert['legs'][-1]['id']) + ','
                mystr = mystr + 'id:' + str(vert['id']) + '),'
            mystr = mystr[:-1] + ')'
            mystr += " (%s)" % ",".join(["%s=%d" % (key, self['orders'][key]) \
                                        for key in self['orders'].keys()])
            return mystr
        else:
            return '()'

    def get_loop_orders(self,model):
        """ Return a dictionary one entry per type of order appearing in the interactions building the loop flow
            The corresponding keys are the number of type this order appear in the diagram. """
        
        loop_orders = {}
        for vertex in self['vertices']:
            # We do not count the identity vertex
            if vertex['id']!=0 and len([1 for leg in vertex['legs'] if leg['loop_line']])==2:
                vertex_orders = model.get_interaction(vertex['id'])['orders']
                for order in vertex_orders.keys():
                    if order in loop_orders.keys():
                        loop_orders[order]+=vertex_orders[order]
                    else:
                        loop_orders[order]=vertex_orders[order]
        return loop_orders

    def is_fermion_loop(self, model):
        """ Return none if there is no loop or if a tag has not yet been set and returns True if this graph contains
            a purely fermionic loop and False if not. """

        if(self['tag']):
            for part in self['tag']:
                if not model.get('particle_dict')[part[0].get('id')].is_fermion():
                    return False
            return True
        else:
            return False

    def is_tadpole(self):
        """ Return None if there is no loop or if a tag has not yet been set and returns True if this graph contains
            a tadpole loop and False if not. """

        if(self['tag']):
            if(len(self['tag'])==1):
               return True
            else:
               return False
        else:
            return None

    def is_wvf_correction(self, struct_rep):
        """ Return None if there is no loop or if a tag has not yet been set and returns True if this graph contains
            a wave-function correction and False if not. """

        if(self['tag']):
            if(len(self['tag'])==2):
                # Makes sure only one current flows off each side of the bubble
                if(len(self['tag'][0][1])==1 and len(self['tag'][1][1])==1):   
                    # Checks that at least one of the two structure is external
                    if(struct_rep[self['tag'][0][1][0]].is_external() or struct_rep[self['tag'][1][1][0]].is_external()):
                        return True
            return False
        else:
            return None

    def tag(self, struct_rep, start, end):
        """ Construct the tag of the diagram providing the loop structure of it. """

        if(self.next_loop_leg(struct_rep,-1,-1,start,end)):
            # We then construct the canonical_tag such that it is a cyclic permutation of tag such that the first loop 
            # vertex appearing in canonical_tag is the one carrying the structure with the lowest ID. This is a safe
            # procedure because a given structure can only appear once in a diagram since FDStructures are characterized
            # by the particle numbers and a given particle number can only appear once in a diagram.
            imin=-2
            minStructID=-2
            for i, part in enumerate(self['tag']):
                if minStructID==-2 or min(part[1])<minStructID:
                    minStructID=min(part[1])
                    imin=i
            if not imin==-2:
                self['canonical_tag']=self['tag'][imin:]+self['tag'][:imin]
            else:
                raise self.PhysicsObjectError, \
                      "Error during the construction of the canonical tag."                   
            return True
        else:
            raise self.PhysicsObjectError, \
                  "Loop diagram tagging failed."
            return False

    def next_loop_leg(self, structRep, fromVert, fromPos, currLeg, endLeg):
        """ Finds a loop leg and what is the next one. Also identify and tag the FD structure attached in
            between these two loop legs. It adds the corresponding tuple to the diagram tag and calls iself 
            again to treat the next loop leg. Return True when tag successfully computed."""

        nextLoopLeg=None
        legPos=-2
        vertPos=-2
        FDStructureIDList=[]

        vertRange=range(len(self['vertices']))
        # If we just start the iterative procedure, then from_vert=-1 and we must look for the "start" loop leg
        # in the entire vertices list
        if not fromVert == -1:
           if fromPos == -1:
               # If the last loop leg was the vertex output (i.e. last in the vertex leg list) then we
               # must look for it in the vertices located after the one where it was found (i.e. from_vert).
               vertRange=vertRange[fromVert+1:]
           else:
               # If the last loop leg was in the vertex inputs (i.e. not last in the vertex leg list) then we
               # must look where it in the vertices located before where it was found (i.e. from_vert).
               vertRange=vertRange[:fromVert]
        # Look in the vertices in vertRange if it can finds the loop leg asked for.
        for i in vertRange:
            # If the last loop leg was an output of its vertex, we must look for it in the INPUTS of the vertices
            # before. However, it it was an input of its vertex we must look in the OUTPUT of the vertices forehead
            legRange=range(len(self['vertices'][i].get('legs')))
            if fromPos == -1:
                # In the last vertex of the list, all entries are input
                if not i==len(self['vertices'])-1:
                    legRange=legRange[:-1]
            else:
                # If looking for an output, then skip the last vertex of the list which only has inputs.
                if i==len(self['vertices'])-1:
                    continue
                else:
                    legRange=legRange[-1:]
            for j in legRange:
                if self['vertices'][i].get('legs')[j].same(currLeg):
                    vertPos=i
                    # Once it has found it, it will look for all the OTHER legs in the vertex and contruct
                    # FDStructure for those who are not loop legs and recognize the next loop leg.
                    for k in filter(lambda ind: not ind==j, range(len(self['vertices'][i].get('legs')))):
                        pos=-2
                        # pos gives the direction in which to look for nextLoopLeg from vertPos. It is after
                        # vertPos (i.e. then pos=-1) only when the next loop leg was found to be the output
                        # (i.e. so positioned last in the vertex leg list) of the vertex at vertPos. Note that
                        # for the last vertex in the list, all entries are i.
                        if not i==len(self['vertices'])-1 \
                           and k==len(self['vertices'][i].get('legs'))-1:
                            pos=-1
                        else:
                            pos=k

                        if self['vertices'][i].get('legs')[k].get('loop_line'):
                            if not nextLoopLeg:
                                nextLoopLeg=self['vertices'][i].get('legs')[k]
                                legPos=pos
                            else:
                                raise self.PhysicsObjectError, \
                                      " An interaction has more than two loop legs."
                        else:
                             FDStruct=FDStructure()
                             # Lauch here the iterative construction of the FDStructure constructing the 
                             # four-vector current of leg at position k in vertex i.
                             canonical = self.construct_FDStructure(i,pos,self['vertices'][i].get('legs')[k],FDStruct)
                             if not canonical:
                                 raise self.PhysicsObjectError, \
                                       "Failed to reconstruct a FDStructure."

                             # The branch was directly an external leg, so it the canonical repr of this struct is
                             # simply ((legID),0).
                             if isinstance(canonical,int):
                                 FDStruct.set('canonical',(((canonical,),0),))
                             elif isinstance(canonical,tuple):
                                 FDStruct.set('canonical',canonical)
                             else:                                      
                                 raise self.PhysicsObjectError, \
                                       "Non-proper behavior of the construct_FDStructure function"

                             # First check if this structure exists in the dictionary of the structures already obtained in the 
                             # diagrams for this process
                             myStructID=-1
                             myFDStruct=structRep.get_struct(FDStruct.get('canonical'))
                             if not myFDStruct:
                                 # It is a new structure that must be added to dictionary struct Rep
                                 myStructID=len(structRep)
                                 # A unique ID is given to the Struct we add to the dictionary.
                                 FDStruct.set('id',myStructID)
                                 structRep.append(FDStruct)
                             else:
                                 # We get here the ID of the FDstruct recognised which has already been added to the dictionary.
                                 # Note that using the unique ID for the canonical tag of the tree cut-loop diagrams has pros and
                                 # cons. In particular, it makes shorter diagram tags yielding shorter selection but at the same
                                 # time is makes the recovery of the full FDStruct object from it's ID more cumbersome.
                                 myStructID=myFDStruct.get('id')

                             FDStructureIDList.append(myStructID) 

                    #Now that we have found loop leg curr_leg, we can get out of the two searching loop
                    break
            if nextLoopLeg:
                break

        if FDStructureIDList:
            # The FDStructure list can be empty in case of an identity vertex       
            self['tag'].append((copy.copy(currLeg),sorted(FDStructureIDList)))
        
        if nextLoopLeg:
            # Returns true if we reached the end loop leg or continues the iterative procedure.
            if nextLoopLeg.same(endLeg):
                return True
            else:
                return self.next_loop_leg(structRep, vertPos, legPos, nextLoopLeg, endLeg)
        else:
            # Returns False in case of a malformed diagram where it has been unpossible to find
            # the loop leg looked for.
            return False

    def construct_FDStructure(self, fromVert, fromPos, currLeg, FDStruct):
        """ Construct iteratively a Feynman Diagram structure attached to a Loop, given at each step
            a vertex and the position of the leg this function is called from. At the same time, it constructs
            a canonical representation of the structure which is a tuple with each element corresponding to
            a 2-tuple ((external_parent_legs),vertex_ID). The external parent legs tuple is ordered as growing
            and the construction of the canonical representation is such that the 2-tuples appear in a fixed order.
            This functions returns a tuple of 2-tuple like above for the vertex where currLeg was found or false if fails.

            To illustrate this algorithm, we take a concrete example, the following structure:
                                                                       
                                                  4 5 6 7
                                               1 3 \/2 \/  <- Vertex ID, left=73 and right=99
                                               \ / | \ /   <- Vertex ID, left=34 and right=42 
                                                |  |4 | 
                                                1\ | /2
                                                  \|/      <- Vertex ID=72
                                                   |
                                                   |1

            For this structure with external legs (1,2,3,5,6,7) and current created 1, the canonical tag will be
            
            (((1,2,3,4,5,6,7),72),((1,3),34),((2,6,7),42),((6,7),99),((4,5),73))
 
        """

        nextLeg = None
        legPos=-2
        vertPos=-2

        vertRange=range(len(self['vertices']))

        # Say we are at the begining of the structure reconstruction algotithm of the structure above, with currLeg=1 so 
        # it was found in the vertex ID=72 with legs (1,1,4,2). Then, this function will call itself on
        # the particles 1,4 and 2. Each of these calls will return a list of 2-tuples or a simple integer being the leg ID
        # for the case of an external line, like leg 4 in our example.
        # So the two lists of 2-tuples returned will be put in the list "reprBuffer". 
        # In fact the 2-tuple are nested in another 2-tuple with the first element being the legID of the current vertex.
        # This helps the sorting of these 2-tuple in a growing order of their originating legID.
        # In this example, once the procedure is finished with vertex ID=72, reprBuffer would be:
        #  [(((1,3),34),),(((5,6),73),),(((2,7,8),42),((7,8),99))]  (Still needs to be sorted and later transformed to a tuple)
        # The 2-tuple corresponding to the mother vertex (so ID=72 in the example) is constructed in vertBuffer (the parent lines
        # list is progressevely filled with the identified external particle of each leg). and will be put in front of vertBuffer
        # and then transformed to a tuple to form the output of the function.
        vertBuffer=[]

        # Each of the parent legs identified for this vertex are put in the first element of a list called here parentBufer.
        # The second element stores the vertex ID where currLeg was found.
        parentBuffer=[[],0]

        # If fromPos == -1 then the leg was an output of its vertex so we must look for it in the vertices
        # following fromVert. If the leg was an input of its vertex then we must look for it in the vertices
        # preceding fromVert.
        if fromPos == -1:
               # If the last loop leg was the vertex output (i.e. last in the vertex leg list) then we
               # must look for it in the vertices located after the one where it was found (i.e. from_vert).
            vertRange=vertRange[fromVert+1:]
        else:
               # If the last loop leg was in the vertex inputs (i.e. not last in the vertex leg list) then we
               # must look where it in the vertices located before where it was found (i.e. from_vert).
            vertRange=vertRange[:fromVert]
        
        # The variable below serves two purposes:
        # 1) It labels the position of the particle in the vertex (-1 = output)
        # 2) If at the end equals to -2, then it means that the particle looked for has not been found.
        pos=-2

        # Look in the vertices in vertRange if it can find the parents of currLeg
        for i in vertRange:
            # We must look in the output of these vertices if the leg was previously found as an input of its
            # vertex. In case it was an output of its vertices, then we must look in the inputs of these vertices.
            # Remember that the last vertex of the list has only inputs.
            legRange=range(len(self['vertices'][i].get('legs')))
            if fromPos == -1:
                # In the last vertex of the list, all entries are input
                if not i==len(self['vertices'])-1:
                    legRange=legRange[:-1]
            else:
                # If looking for an output, then skip the last vertex of the list which only has inputs.
                if i==len(self['vertices'])-1:
                    continue
                else:
                    legRange=legRange[-1:]

            # Now search over the leg range for currLeg
            for j in legRange:
                if self['vertices'][i].get('legs')[j].same(currLeg):
                    # The id of the vertex where currLeg was found is stored in the second element of parentBuffer.
                    parentBuffer[1]=self['vertices'][i].get('id')
                    # We can add this vertex to the FDStructure vertex list, in the "right" order so that 
                    # a virtual particle in the inputs of some vertex appears always AFTER the vertex where
                    # this particle was the output.
                    if fromPos == -1:
                        FDStruct.get('vertices').append(copy.copy(self['vertices'][i]))
                    else:
                        FDStruct.get('vertices').insert(0,copy.copy(self['vertices'][i]))
                     
                    # Now we must continue the iterative procedure for each of the other leg of the vertex found.
                    for k in filter(lambda ind: not ind==j, range(len(self['vertices'][i].get('legs')))):
                        # If we found currLeg in an identity vertex we directly skip it for what regards the
                        # construction of the cannonical representation
                        if not self['vertices'][i].get('id'):
                            return self.construct_FDStructure(i, k, self['vertices'][i].get('legs')[k], FDStruct)

                        if k==len(self['vertices'][i].get('legs'))-1 \
                           and not i==len(self['vertices'])-1:
                            pos=-1
                        else:
                            pos=k
                        # We get here the structure of each branch of the actual vertex.    
                        branch=self.construct_FDStructure(i, pos, self['vertices'][i].get('legs')[k], FDStruct)
                        if not branch:
                            raise self.PhysicsObjectError, \
                                  "Failed to reconstruct a FDStructure."
                        # That means that this branch was an external leg.  
                        if isinstance(branch,int):
                            parentBuffer[0].append(branch)
                        # If it is a list it means that the branch contains at least one further vertex.
                        elif isinstance(branch,tuple):
                            parentBuffer[0]+=list(branch[0][0])
                            vertBuffer.append(branch)
                        else:
                            raise self.PhysicsObjectError, \
                                  "Non-proper behavior of the construct_FDStructure function"


        if(pos == -2): 
            if(not fromPos == -1):
                # In this case, the leg has not been found. It is an external leg.
                FDStruct.get('external_legs').append(copy.copy(currLeg))
                return currLeg.get('number')
            else:
                raise self.PhysicsObjectError, \
                                  " A structure is malformed."
        else:
            # In this case a vertex with currLeg has been found and we must return the list of tuple described above.
            # First let's sort the list so that the branches comes in a fixed order which is irrelevant but not trivial here.
            # First comes the branches involving the smallest number of vertices. Among those who have an equal number of
            # vertices, those with the smallest ID for the external legs come first.
            vertBuffer.sort()
            # Now flatten the list to have a list of tuple instead of a list of tuple made of tuples.
            # In the above example, this corresponds to go from
            # [(((1,3),34),),(((5,6),73),),(((2,7,8),42),((7,8),99))]
            # to
            # [((1,3),34),((5,6),73),((2,7,8),42),((7,8),99)]
            vertBufferFlat=[]
            for t in vertBuffer:
                for u in t:
                    vertBufferFlat.append(u)
                    
            # Sort the parent lines
            parentBuffer[0].sort()
            # Add the 2-tuple corresponding to the vertex where currLeg was found.
            vertBufferFlat.insert(0,(tuple(parentBuffer[0]),parentBuffer[1]))
            return tuple(vertBufferFlat)

#===============================================================================
# LoopModel
#===============================================================================
class LoopModel(base_objects.Model):
    """A class to store all the model information with advanced feature
       to compute loop process."""
    
    def default_setup(self):
       super(LoopModel,self).default_setup()         
       self['perturbation_couplings'] = []

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'perturbation_couplings':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a list" % \
                                                            type(value)
            for order in value:
                if not isinstance(value, str):
                    raise self.PhysicsObjectError, \
                        "Object of type %s is not a string" % \
                                                            type(order)
        else:
            return super(LoopModel,self).filter(name,value)

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['name', 'particles', 'parameters', 'interactions', 'couplings',
                'lorentz','perturbation_couplings','conserved_charge']

#===============================================================================
# DGLoopLeg
#===============================================================================
class DGLoopLeg(base_objects.Leg):
    """A class only used during the loop diagram generation. Exactly like leg
       except for a few other parameters only useful during the loop diagram
       generation."""
    
    def __init__(self,argument=None):
        """ Allow for initializing a DGLoopLeg of a Leg """
        if not isinstance(argument, base_objects.Leg):
            if argument:
                super(DGLoopLeg,self).__init__(argument)
            else:
                super(DGLoopLeg,self).__init__()
        else:
            super(DGLoopLeg,self).__init__()
            for key in argument.get_sorted_keys():
                self.set(key,argument[key])

    def default_setup(self):
       super(DGLoopLeg,self).default_setup()         
       self['depth'] = 0

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'depth':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a int" % \
                                                            type(value)
        else:
            return super(DGLoopLeg,self).filter(name,value)

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['id', 'number', 'state', 'from_group','loop_line','depth']
    
    def convert_to_leg(self):
        """ Converts a DGLoopLeg back to a Leg. Basically removes the extra
            attributes """

        aleg=base_objects.Leg()
        for key in aleg.get_sorted_keys():
            aleg.set(key,self[key])

        return aleg

#===============================================================================
# FDStructure
#===============================================================================
class FDStructure(base_objects.PhysicsObject):
    """FDStructure:
    list of vertices (ordered). This is part of a diagram.
    """

    def default_setup(self):
        """Default values for all properties""" 

        self['vertices'] = VertexList()
        self['id'] = -1 
        self['external_legs'] = LegList()
        self['canonical'] = ()

    def is_external(self):
        """Returns wether the structure is simply made of an external particle only"""
        if (len(self['canonical'])==1 and self['canonical'][0][1]==0):
            return True
        else:
            return False

    def filter(self, name, value):
        """Filter for valid FDStructure property values."""

        if name == 'vertices':
            if not isinstance(value, VertexList):
                raise self.PhysicsObjectError, \
        "%s is not a valid VertexList object" % str(value)

        if name == 'id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
        "id %s is not an integer" % repr(value)

        if name == 'external_legs':
            if not isinstance(value, LegList):
                raise self.PhysicsObjectError, \
        "external_legs %s is not a valid Leg List" % str(value)

        if name == 'canonical':
            if not isinstance(value, tuple):
                raise self.PhysicsObjectError, \
        "canonical %s is not a valid tuple" % str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id','external_legs','canonical','vertices']

    def nice_string(self):
        """Returns a nicely formatted string of the diagram content."""

        mystr=''

        if not self['id']==-1:
            mystr=mystr+'id: '+str(self['id'])+' , '
        else:
            return '()'

        if self['canonical']:
            mystr=mystr+'canonical_repr.: '+str(self['canonical'])+' , '

        if self['external_legs']:
            mystr=mystr+'external_legs: { '
            for leg in self['external_legs'][:-1]:
                mystr = mystr + str(leg['number']) + '(%s)' % str(leg['id']) + ', '
            mystr = mystr + str(self['external_legs'][-1]['number']) + '(%s)' % str(self['external_legs'][-1]['id']) + ' }'
        return mystr

#===============================================================================
# FDStructureList
#===============================================================================
class FDStructureList(base_objects.PhysicsObjectList):
    """List of FDStructure objects
    """

    def is_valid_element(self, obj):
         """Test if object obj is a valid Diagram for the list."""

         return isinstance(obj, FDStructure)

    def get_struct(self, ID):
        """Return the FDStructure of the list with the corresponding canonical 
           tag if ID is a tuple or the corresponding ID if ID is an integer.
           It returns the structure if it founds it, or None if it was not found"""
        if isinstance(ID, int):
            for FDStruct in self:
                if FDStruct.get('id')==ID:
                    return FDStruct
            return None
        elif isinstance(ID, tuple):
            for FDStruct in self:
                if FDStruct.get('canonical')==ID:
                    return FDStruct
            return None
        else:
            raise self.PhysicsObjectListError, \
                "The ID %s specified for get_struct is not an integer or tuple" % \
                                                                       repr(object)

    def nice_string(self):
        """Returns a nicely formatted string"""
        mystr = str(len(self)) + ' FD Structures:\n'
        for struct in self:
            mystr = mystr + "  " + struct.nice_string() + '\n'
        return mystr[:-1]


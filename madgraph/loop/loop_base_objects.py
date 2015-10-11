################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Definitions of all basic objects with extra features to treat loop 
   diagrams"""

import copy
import itertools
import logging
import numbers
import os
import re
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.base_objects as base_objects
import madgraph.various.misc as misc
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
    
    # The class variable below select what algorithm is used for choosing where
    # to cut the loops. The possibilities are:
    # 'optimal' -> will use choos_optimal_lcut()
    # 'default' -> will use chose_default_lcut()
    # In principle it is always 'optimal'. But it can be changed by process_check
    # for the purpose of the check permutation command.
    cutting_method = 'optimal'

    def default_setup(self):
        """Default values for all properties"""

        super(LoopDiagram,self).default_setup()
        # This tag specifies the particular structure of this loop, cut at 
        # and ordered in the same way as originally generated. It contains
        # the full information about the loop vertices and the loop legs.
        # It is of the form:
        #   [(Leg,[Structure_IDs],VertexID), (...), ...]
        self['tag'] = []        
        # This tag uniquely define a loop particle. It is not used for born, 
        # R2 and UV diagrams. It is only a list of integers, so not too
        # heavy to store. It is what allows for diagram selection. 
        # It is of the form:
        #   [(LegPDG,[Structure_IDs],VertexID), (...), ...]
        # But ordered in a canonical unambiguous way.
        self['canonical_tag'] = []
        # This information is in principle recoverable from the VertexList but
        # it is faster to store it as a single integer.
        # It is the (positive) PDG of the (particle, not anti-particle) L-cut 
        # particle for a loop diagram.
        self['type'] = 0
        # Loop diagrams can be identified to others which are numerically exactly
        # equivalent. This is the case for example for the closed massles quark
        # loops. In this case, only one copy of the diagram is kept and this
        # multiplier attribute is set the to number of identified diagrams.
        self['multiplier'] = 1
        # This stores the list of amplitudes vertices which give the R2/UV
        # counter-terms to this loop.
        self['CT_vertices'] = base_objects.VertexList()
        # The 'contracted_diagram' is the diagram constructed by the function
        # get_contracted_loop_diagram' which corresponds to the list of vertices
        # to construct the equivalent diagram to this one when the loop is shrunk
        # to one point.
        self['contracted_diagram'] = None

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'tag':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid tag" % str(value)
            else:
                for item in value:
                    if (len(item)!=3 or \
                      not isinstance(item[0],base_objects.Leg) or \
                      not isinstance(item[1],list)) or \
                      not isinstance(item[2],base_objects.Vertex):
                        raise self.PhysicsObjectError, \
                            "%s is not a valid tag" % str(value)

        if name == 'canonical_tag':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid tag" % str(value)
            else:
                for item in value:
                    if (len(item)!=3 or not isinstance(item[0],int) or \
                      not isinstance(item[1],list)) or \
                      not isinstance(item[2],int):
                        raise self.PhysicsObjectError, \
                            "%s is not a valid canonical_tag" % str(value)

        if name == 'CT_vertices':
            if not isinstance(value, base_objects.VertexList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid VertexList object" % str(value)

        if name == 'type':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)

        if name == 'multiplier':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)

        if name == 'contracted_diagram':
            if not isinstance(value, base_objects.Diagram):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Diagram." % str(value)                            

        else:
            super(LoopDiagram, self).filter(name, value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        
        return ['vertices', 'CT_vertices', 'orders', 'type', 'tag']

    def nice_string(self, struct_list=None ):
        """Returns a nicely formatted string of the diagram content."""
        
        # Return the mother nice_string if this LoopDiagram is of born type.
        if self['type']==0:
            return super(LoopDiagram,self).nice_string()
        
        mystr=''
        if not self['vertices']:
            return '()'
        if self['canonical_tag']:
            mystr = mystr+'canonical tag: '+str(self['canonical_tag'])+'\n'
        if self['CT_vertices']:
            mystr = mystr+'CT vertex ids:'
            for ctvx in self['CT_vertices']:
                mystr = mystr +' '+str(ctvx.get('id'))
            mystr = mystr+'\n'
        if self['vertices']:
            mystr = mystr+'Loop vertices: ('
            for vert in self['vertices']:
                mystr = mystr + '('
                for leg in vert['legs'][:-1]:
                    if leg['loop_line']:
                        mystr = mystr + str(leg['number']) + \
                        '(%s*)' % str(leg['id']) + ','
                    else:
                        mystr = mystr + str(leg['number']) + \
                        '(%s)' % str(leg['id']) + ','                        

                if self['vertices'].index(vert) < len(self['vertices']) - 1:
                    # Do not want ">" in the last vertex
                    mystr = mystr[:-1] + '>'
                if vert['legs'][-1]['loop_line']:
                    mystr = mystr + str(vert['legs'][-1]['number']) + \
                    '(%s*)' % str(vert['legs'][-1]['id']) + ','
                else:
                    mystr = mystr + str(vert['legs'][-1]['number']) + \
                    '(%s)' % str(vert['legs'][-1]['id']) + ','                    
                mystr = mystr + 'id:' + str(vert['id']) + '),'
            mystr = mystr[:-1] + ')'
            mystr += " (%s)" % ",".join(["%s=%d" % (key, self['orders'][key]) \
                                        for key in self['orders'].keys()])+"\n"
        if struct_list and self['tag']:
            for i, tag_elem in enumerate(self['tag']):
                for j, struct in enumerate(tag_elem[1]):
                    if len(tag_elem[1])>1:
                        mystr += 'Struct. #'+str(j+1)+\
                                 ' on loop vx #'+str(i+1)+": "+\
                      struct_list[struct].nice_string_vertices()+"\n"
                    else:
                        mystr += 'Struct. on loop vx #'+str(i+1)+": "+\
                      struct_list[struct].nice_string_vertices()+"\n"
        #remove the unecessary last \n on the line
        mystr=mystr[:-1]

        return mystr

    def get_contracted_loop_diagram_without_tag(self, struct_rep=None):
        """This is the old function used without tag which means that no
        canonical loop information can be produced. It will be used for 
        unit test only and moved there when I'll implement them."""

        # Without the tagging information we will have to reconstruct the
        # contracted diagrams with the unordered vertices
        if len(self.get('vertices'))==0:
            raise MadGraph5Error, "Function get_contracted_loop_diagram()"+\
                "called for the first time without specifying struct_rep "+\
                                            "for a diagram already tagged."
                                
        # The leg below will be the outgoing one 
        contracted_vertex_last_loop_leg = None
        # List here the vertices which have to appear after and before the
        # contracted loop vertex.
        vertices_after_contracted_vertex = []
        vertices_before_contracted_vertex = []
        # To know if a given vertex must be placed after or before the
        # the contracted loop vertex, we must now the list of leg numbers
        # which have been "generated" starting from the one outgoing leg of
        # the contracted loop vertex.
        contracted_vertex_leg_daughters_nb = []
                                            
        # We need a different treatment for the amplitude-type vertex 
        # (the last one) for which all legs are incoming.
        for vertex in self.get('vertices')[:-1]:
            # If the interaction had nothing to do with a loop, just add it
            if not any(l['loop_line'] for l in vertex.get('legs')):
                # before the contracted vertex if it didn't need any of 
                # the leg numbers it generated
                if any((l.get('number') in contracted_vertex_leg_daughters_nb) \
                                          for l in vertex.get('legs')[:-1]):
                    vertices_after_contracted_vertex.append(vertex)
                    contracted_vertex_leg_daughters_nb.append(vertex.get('legs')[-1])
                else:
                    vertices_before_contracted_vertex.append(vertex)
            else:
                # Add to the mothers of the contracted vertex
                contracted_vertex.get('legs').extend(
                   [l for l in vertex.get('legs')[:-1] if not l['loop_line']])
                # Extend the list of PDGs making up this interaction.
                # This is useful for the DigramChainTag.
                contracted_vertex.get('PDGs').extend([l.get('id') for l in
                                  vertex.get('legs') if not l['loop_line']])
                # If the outgoing leg is not a loop line but the vertex still
                # has two loop lines as mothers, then it is the vertex that we 
                # must replace by the contracted loop vertex
                if not vertex.get('legs')[-1]['loop_line']:
                    # The contracted vertex is not of amplitude type here
                    contracted_vertex_last_loop_leg = vertex.get('legs')[-1]
                    
        # Treat the last vertex now
        if any(l['loop_line'] for l in self.get('vertices')[-1].get('legs')):
            # Add to the mothers of the contracted vertex
            contracted_vertex.get('legs').extend([l for l in 
                self.get('vertices')[-1].get('legs') if not l['loop_line']])

        else:
            vertices_after_contracted_vertex.append(self.get('vertices')[-1])
            
        
        contracted_diagram_vertices.extend(vertices_before_contracted_vertex)
        if not contracted_vertex_last_loop_leg is None:
            contracted_vertex.get('legs').append(contracted_vertex_last_loop_leg)

        if len(contracted_vertex.get('legs'))==1:
            stop
        contracted_diagram_vertices.append(contracted_vertex)
        contracted_diagram_vertices.extend(vertices_after_contracted_vertex)

        contracted_diagram = base_objects.Diagram(
           {'vertices':contracted_diagram_vertices,'orders':self.get('orders')})

        return contracted_diagram
    
    def build_loop_tag_for_diagram_identification(self, model, FDStrut_rep,
                                            use_FDStructure_ID_for_tag = False):
        """ This function returns what will be used as the 'loop_tag' attribute
        of the ContractedVertex instance in the function 'get_contracted_loop_diagram'.
        It is important since it is what is used by MG5_aMC to decide
        if two processes have *exactly* the same matrix element and can be
        identified. 
        There is no need to characterize the details of the FDStructures attached
        to the loops because these are already compared using the rest of the
        DiagramTag structure. All we need is to identify a structure by its
        external leg numbers."""

        canonical_tag = self['canonical_tag']
     
        # First create a list of objects we want to use to identify the particles
        # running in the loop. We use here the same strategy as in the function
        # 'vertex_id_from_vertex' of IdentifyMETag.
        # However, in addition to what one has in IdentifyMETag, we must also
        # keep track of the attribute 'is_part' since this provides the 
        # direction of the loop flow.
        loop_parts_tagging = [[]]*len(canonical_tag)
        for i, tag_elem in enumerate(canonical_tag):
            loop_part = model.get_particle(tag_elem[0])
            loop_parts_tagging[i] = (loop_part.get('spin'), 
                                     loop_part.get('color'),
                                     loop_part.get('self_antipart'),
                                     loop_part.get('mass'),
                                     loop_part.get('width'),
                                     loop_part.get('is_part'))
        
        # Now create a list of objects which we want to use to uniquely
        # identify each structure attached to the loop for the loop_tag.
        FDStructs_tagging = [[]]*len(canonical_tag)
        for i, tag_elem in enumerate(canonical_tag):
            for struct_ID in tag_elem[1]:
                if not use_FDStructure_ID_for_tag:
                    # The FDStructures will be probed by the rest of the 
                    # DiagramTag, it is therefore not necessary to include any
                    # information regarding the structures in the loop_tag.
                    # However, notice that this means that the same loop attached
                    # to structures (1,2,3,4), in this order, and another one
                    # attached to the same structure but in a different order,
                    # say (1,4,3,2), will share the same DiagramTag (because the
                    # structure are the same but in a different order) since the
                    # loop_tag doesn't account for any information regarding the
                    # structures. This is ok because the Diagram is only intended
                    # for process identifications.
                    pass
#                    FDStructs_tagging[i].extend([leg.get('number') for leg in
#                        FDStrut_rep.get_struct(struct_ID).get('external_legs')])
                else:
                    # For the loop diagram identification (within a given process)
                    # we must account for the FDStructure, and it is then
                    # simplest to just use their ID (since all loop diagrams 
                    # have been tagged with the same FDStructure repository in
                    # this case, so that the FDStructure ID is really unique).
                    # There is no need to use the 'canonical' attribute of the
                    # structure ID.
                    FDStructs_tagging[i].append(struct_ID)

            FDStructs_tagging[i].sort()
            FDStructs_tagging[i] = tuple(FDStructs_tagging[i])
        
        # We want to identify processes together if their diagrams
        # are made of the same interactions which can however have different
        # ID's for different process (i.e. the ID of the 'gdd~' interaction is
        # different than the one of 'gss~'). We again use the same strategy
        # as in the function 'vertex_id_from_vertex' of IdentifyMETag.
        # So we create a list of objects we want to use to tag the loop interactions 
        interactions_tagging = [[]]*len(canonical_tag)
        for i, tag_elem in enumerate(canonical_tag):
            inter = model.get_interaction(tag_elem[2])
            coup_keys = sorted(inter.get('couplings').keys())
            interactions_tagging[i]=(
                tuple((key, inter.get('couplings')[key]) for key in coup_keys),
                tuple(str(c) for c in inter.get('color')),
                tuple(inter.get('lorentz')))

        return tuple(
                 # For each loop vertex, we must identify the following three things
                 zip(
                   # Loop particle identification
                   loop_parts_tagging,
                   # FDStructure identification
                   FDStructs_tagging,
                   # Loop interactions identification
                   interactions_tagging,
                 )
                 # Finally make sure that the loop orders are the same
                 + sorted(self.get_loop_orders(model).items())
               )

    def get_contracted_loop_diagram(self, model, struct_rep=None):
        """ Returns a base_objects.Diagram which correspond to this loop diagram
        with the loop shrunk to a point. If struct_rep is no specified, then
        the tagging will proceed assuming no FDStructure has been identified yet.
        Otherwise, it will possible reuse them an update the repository."""
        
        if self['type']<=0:
            return copy.copy(self)

        if self['contracted_diagram']:
            return self['contracted_diagram']

        # If this loop diagram hasn't been tagged yet, we must do that now.
        # (or if the structure repository is not provided
        if not self['canonical_tag'] or struct_rep is None:
            n_external_legs = len(base_objects.LegList([l for l in 
                               self.get_external_legs() if not l['loop_line']]))
            
            # use natural ordering for loop tagging
            start_in, end_in = n_external_legs +1, n_external_legs+2
            for l in self['vertices'][0]['legs']:
                if l.same(start_in):
                    break
                elif l.same(end_in):
                    start_in, end_in = end_in, start_in
                    break
                
            if struct_rep is None:                
                struct_rep = FDStructureList([])
            self.tag(struct_rep, model, start_in=start_in, end_in=end_in, 
                                                              synchronize=False)

        contracted_diagram_vertices = base_objects.VertexList()
        # We give this vertex the special ID -2 so that whenever MG5_aMC tries
        # to retrieve an information in typically gets from the model interaction
        # it will instead get it from the 'loop_info' provided by the contracted
        # vertex of its corresponding vertex_id in a Tag
        contracted_vertex = base_objects.ContractedVertex({
          'id':-2,
          'loop_orders':self.get_loop_orders(model),
          'loop_tag': self.build_loop_tag_for_diagram_identification(model, struct_rep)
                                                          })

        # Using the 'tag' information, the construction of the contracted diagram
        # quite simple. First scan all structures to add their vertices and at
        # the same time construct the legs of the final vertex which corresponds
        # to the shrunk loop.
        for tagelem in self['tag']:
            contracted_vertex.get('legs').extend([struct_rep[
                 struct_ID].get('binding_leg') for struct_ID in tagelem[1]])
            # Extend the list of PDGs making up this interaction.
            # This is useful for the DigramChainTag.
            contracted_vertex.get('PDGs').extend([struct_rep[struct_ID].
                  get('binding_leg').get('id') for struct_ID in tagelem[1]])     
            contracted_diagram_vertices.extend(sum([struct_rep[
                struct_ID].get('vertices') for struct_ID in tagelem[1]],[]))
    
        # Add the shrunk vertex to the contracted diagram vertices list.
        contracted_diagram_vertices.append(contracted_vertex)

        contracted_diagram = base_objects.Diagram(
           {'vertices':contracted_diagram_vertices,'orders':self.get('orders')})
        
        self['contracted_diagram'] = contracted_diagram
        
        return contracted_diagram

    def get_CT(self,model,string=None):
        """ Returns the CounterTerms of the type passed in argument. If None
            it returns all of them. """
        if string:
            return base_objects.VertexList([vert for vert in \
              self['CT_vertices'] if string in \
              model['interaction_dict'][vert['id']]['type']])
        else:
            return self['CT_vertices']

    def is_fermion_loop(self, model):
        """ Return none if there is no loop or if a tag has not yet been set and
        returns True if this graph contains a purely fermionic loop and False if 
        not. """

        if(self['tag']):
            for part in self['tag']:
                if not model.get('particle_dict')[part[0].get('id')].is_fermion():
                    return False
            return True
        else:
            return False

    def is_tadpole(self):
        """ Return None if there is no loop or if a tag has not yet been set and 
        returns True if this graph contains a tadpole loop and False if not. """

        if(self['tag']):
            if(len(self['tag'])==1):
               return True
            else:
               return False
        else:
            return None

    def is_vanishing_tadpole(self,model):
        """Return None if there is no loop or if a tag has not yet been set and 
        returns True if this graph contains a vanishing tadpole loop and False 
        if not. """

        if not self.is_tadpole():
            return False

        # absorbed by renormalization of vev
        if(len(self['tag'][0][1])<=1):
            return True
        # massless tadpole
        return any([part['mass'].lower()=='zero' for pdg,part in \
                                         model.get('particle_dict').items() if \
                                             pdg==abs(self['tag'][0][0]['id'])])

    def is_wf_correction(self, struct_rep, model):
        """ Return None if there is no loop or if a tag has not yet been set and
        returns True if this graph contains a wave-function correction and False
        if not. """

        if self['tag'] :
            # Makes sure only one current flows off each side of the bubble
            if len(self['tag'])==2 and len(self['tag'][0][1])==1 \
               and len(self['tag'][1][1])==1:   
                # Checks that at least one of the two structure is external
                if struct_rep[self['tag'][0][1][0]].is_external() or \
                                 struct_rep[self['tag'][1][1][0]].is_external():
                    # Check that the two binding legs are of the same nature
                    inLegID=struct_rep[self['tag'][0][1][0]]['binding_leg']['id']
                    outLegID=struct_rep[self['tag'][1][1][0]]['binding_leg']['id']
                    return True
            
            # check a wf correction with tadpole (massive)
            if len(self['tag'])==1 and len(self['tag'][0][1])==2 and \
               (struct_rep[self['tag'][0][1][0]].is_external() or
                                struct_rep[self['tag'][0][1][1]].is_external()):
                return True

            return False
        else:
            return None

    def get_nloopline(self):
        """Return the number of loop lines. """
        if self['tag']:
            return len(self['tag'])
        else:
            return None

    @classmethod
    def compute_weight(cls, FD_ids_list, struct_rep, number_legs):
        """ Computes the weighting function S for this structure 'i' such that
        S(i)>0 for each any i, S(i)!=S(j) if i['external_legs']!=j['external_legs']
        and S(i+j)>max(S(i),S(j)). """
        
        external_numbers=[leg['number'] for id in FD_ids_list for leg in \
                                 struct_rep.get_struct(id).get('external_legs')]
        external_numbers.sort()
        weight=0
        for i, number in enumerate(external_numbers):
            weight=i*number_legs+number
        return weight
    
    @classmethod
    def choose_optimal_lcut(cls,intag,struct_rep, model, external_legs):
        """ This function chooses the place where to cut the loop in order to
        maximize the loop wavefunction recycling in the open loops method.
        This amounts to cut just before the combined structure with smallest 
        weight and then chose the direction to go towards the one with smallest
        weight."""
        
        tag=copy.deepcopy(intag)
        number_legs=len(external_legs)
        
        # Put the smallest weight first
        weights=[cls.compute_weight(t[1],struct_rep,number_legs) for t in tag]
        imin = weights.index(min(weights))
        tag=tag[imin:]+tag[:imin]
        weights=weights[imin:]+weights[:imin]
        
        # Now chose the direction
        rev_tag=cls.mirrored_tag(tag, model)
        # Put it back with the smallest weight first
        rev_tag=rev_tag[-1:]+rev_tag[:-1]
        rev_weights=[cls.compute_weight(t[1],struct_rep,number_legs) for t in rev_tag]
        
        # Finally return the appropriate tag
        if len(tag)==1:
            return tag
        elif len(tag)==2:
            if abs(tag[0][0]['id'])>abs(tag[1][0]['id']):
                return rev_tag
            else:
                return tag
        else:
            if rev_weights[1]<weights[1]:
                return rev_tag
            else:
                return tag

    @classmethod
    def choose_default_lcut(cls,tag, model):
        """ This function chooses where to cut the loop. It returns the
            canonical tag corresponding to this unambiguous choice."""
        # We then construct the canonical_tag such that it is a cyclic
        # permutation of tag such that the first loop vertex appearing in 
        # canonical_tag is the one carrying the structure with the lowest 
        # ID. This is a safe procedure because a given structure can only
        # appear once in a diagram since FDStructures are characterized by 
        # the particle numbers and a given particle number can only appear 
        # once in a diagram.
        canonical_tag=copy.deepcopy(tag)
        canonical_tag=cls.make_canonical_cyclic(canonical_tag)
        canonical_mirrored_tag=copy.deepcopy(canonical_tag)
        canonical_mirrored_tag=cls.mirrored_tag(canonical_mirrored_tag,model)
        # We must put it back in the canonical cyclic order
        canonical_mirrored_tag=canonical_mirrored_tag[-1:]+\
                                 canonical_mirrored_tag[:-1]
        # Now to relieve the remaining ambiguity due to the mirrored L-cut 
        # diagram, we chose among the two equivalent tag 'canonical_tag' and
        # 'canonical_mirrored_tag' the one having the lowest structure ID in 
        # second position (this is equivalent as saying that we always 
        # construct the tag starting next to the lowest structure ID and
        # in the direction of the next-to-lowest structure ID). This is 
        # irrelevant in the case of tadpoles (len(tag)==1) and bubbles made
        # of the same particle. If these bubbles are not made of the same
        # two particle, the tag chosen is the one starting from the biggest 
        # particle id.
        # Remove the redundant bubble diagrams, like [a W- a] and [W+ a W-]
        # add abs when it is a bubble,i.e. len(tag)==2
        if (len(tag)==2 and abs(canonical_mirrored_tag[0][0]['id'])>\
                            abs(canonical_tag[0][0]['id'])) or (len(tag)>2 and \
                              canonical_mirrored_tag[1][1]<canonical_tag[1][1]):
            canonical_tag=canonical_mirrored_tag

        return canonical_tag        

    def tag(self, struct_rep, model, start_in=None, end_in=None, synchronize=True):
        """ Construct the tag of the diagram providing the loop structure 
        of it. """
       
        # Create the container for the new vertices which create the loop flow
        # It is dummy at this stage
        loopVertexList=base_objects.VertexList()
        
        # We create here the list of external legs. It will be used in each call
        # of process_next_loop_leg to generate the FDStructure vertices, so
        # it is more efficient to create it here once only.
        external_legs = base_objects.LegList([l for l in 
                                self.get_external_legs() if not l['loop_line']])
        n_initial = len([1 for leg in external_legs if not leg['state']])

        if start_in is None or end_in is None:
            start_in = len(external_legs)+1
            end_in = len(external_legs)+2                     
                
        # Notice here that start and end can be either the Legs object
        # specification of the two L-cut particles or simply their 'number'.
        if isinstance(start_in,int) and isinstance(end_in,int):
            start=start_in
            end=end_in
        elif isinstance(start_in,base_objects.Leg) and \
                                            isinstance(end_in,base_objects.Leg):
            start=start_in.get('number')
            end=end_in.get('number')
        else:
            raise MadGraph5Error, "In the diagram tag function, 'start' and "+\
                " 'end' must be either integers or Leg objects." 
        
        if self.process_next_loop_leg(struct_rep,-1,-1,start,end,\
                                          loopVertexList, model, external_legs):
            # Possible check here is:
            #mytype=self['type']
            #self.synchronize_loop_vertices_with_tag(process['model'],
            #                                               struct_rep,start,end)
            #assert(loopVertexList==self['vertices'] and mytype==self['type'])
            
            # Different choices of the loop cut can be made suited for different
            # optimizations.
            if self.cutting_method=='default':
                # The default one has no specific property.
                canonical_tag=self.choose_default_lcut(self['tag'],model)
            elif self.cutting_method=='optimal':
                # The choice below is optimized for recycling the loop wavefunction
                # in the open loops method.
                canonical_tag=self.choose_optimal_lcut(self['tag'],struct_rep, 
                                                           model, external_legs)
            else:
                raise MadGraph5Error, 'The cutting method %s is not implemented.'\
                                                            %self.cutting_method
            # The tag of the diagram is now updated with the canonical tag
            self['tag']=canonical_tag
            # We assign here the loopVertexList to the list of vertices 
            # building this loop diagram. Keep in mind the the structures are 
            # factored out.
            if synchronize:
                self.synchronize_loop_vertices_with_tag(model,n_initial,
                                                           struct_rep,start,end)
            # Now we just have to replace, in the canonical_tag, the legs with
            # the corresponding leg PDG since this is the only thing that matter
            # when building a canonical representation for the loop to perform 
            # the selection of the loop basis.
            self['canonical_tag']=[[t[0]['id'],t[1],t[2]] for t in canonical_tag]
            return True
        else:
            raise self.PhysicsObjectError, \
                  "Loop diagram tagging failed."
            return False


    @classmethod
    def generate_loop_vertex(cls,myleglist, model, n_initial, vertID):
        """ Generate a loop vertex from incoming legs myleglist and the 
        interaction with id vertID of the model given in argument """
        # Define easy access point
        ref_dict_to1 = model.get('ref_dict_to1')
        # Now we make sure we can combine those legs together (and 
        # obtain the output particle ID)
        key=tuple(sorted([leg.get('id') for leg in myleglist]))
        if ref_dict_to1.has_key(key):
            for interaction in ref_dict_to1[key]:
                # Find the interaction with the right ID
                if interaction[1]==vertID:
                    # Create the output Leg and add it to the 
                    # existing list 
                    #1) id is like defined by ref_dict_to1
                    legid = interaction[0]
                    # 2) number is the minimum of leg numbers 
                    #    involved in the combination
                    number = min([leg.get('number') for leg in\
                                   myleglist])
                    # 3) state is final, unless there is exactly 
                    #    one initial state particle involved in the
                    #    combination -> t-channel
                    #    For a decay process there is of course no t-channel
                    if n_initial>1 and len(myleglist)>1 and len(filter( \
                        lambda leg: leg.get('state') == False, myleglist)) == 1:
                        state = False
                    else:
                        state = True
                    myleglist.append(base_objects.Leg(\
                                            {'number': number,\
                                             'id': legid,\
                                             'state': state,
                                             'loop_line': True}))
                    # Now we can add the corresponding vertex
                    return base_objects.Vertex({'legs':myleglist,'id':vertID})
        else:
            raise cls.PhysicsObjectError, \
            "An interaction from the original L-cut diagram could"+\
            " not be found when reconstructing the loop vertices."

    def process_next_loop_leg(self, structRep, fromVert, fromPos, currLeg, \
                                  endLeg, loopVertexList, model, external_legs):
        """ Finds a loop leg and what is the next one. Also identify and tag the
        FD structure attached in between these two loop legs. It adds the 
        corresponding tuple to the diagram tag and calls iself again to treat
        the next loop leg. Return True when tag successfully computed."""

        nextLoopLeg=None
        legPos=-2
        vertPos=-2
        FDStructureIDList=[]
        vertFoundID=-1
        n_initial = len([1 for leg in external_legs if not leg['state']])

        # Helper function to process a loop interaction once found
        def process_loop_interaction(i,j,k,pos):
            """For vertex position 'i' and loop leg position 'j'. Find the 
            structure attached to leg k of this loop interaction, tag it and
            update the loop tag."""
            FDStruct=FDStructure()
            # Launch here the iterative construction of the FDStructure 
            # constructing the four-vector current of leg at position k 
            # in vertex i.
            canonical = self.construct_FDStructure(i,pos,\
                               self['vertices'][i].get('legs')[k],FDStruct)

            if not canonical:
                raise self.PhysicsObjectError, \
                      "Failed to reconstruct a FDStructure."
            
            # The branch was directly an external leg, so it the canonical
            # repr of this struct is simply ((legID),0).
            if isinstance(canonical,int):
                FDStruct.set('canonical',(((canonical,),0),))
            elif isinstance(canonical,tuple):
                FDStruct.set('canonical',canonical)
            else:                                      
                raise self.PhysicsObjectError, \
                "Non-proper behavior of the construct_FDStructure function"
            
            # First check if this structure exists in the dictionary of the
            # structures already obtained in the diagrams for this process
            myStructID=-1
            myFDStruct=structRep.get_struct(FDStruct.get('canonical'))
            if not myFDStruct:
                # It is a new structure that must be added to dictionary 
                # struct Rep
                myStructID=len(structRep)
                # A unique ID is given to the Struct we add to the
                # dictionary.
                FDStruct.set('id',myStructID)
                # And we now ask the structure to create its vertices, 
                # starting from the outter legs going inwards towards the 
                # binding leg.
                FDStruct.generate_vertices(model, external_legs)
                structRep.append(FDStruct)
            else:
                # We get here the ID of the FDstruct recognised which has 
                # already been added to the dictionary. Note that using the
                # unique ID for the canonical tag of the tree cut-loop 
                # diagrams has pros and cons. In particular, it makes 
                # shorter diagram tags yielding shorter selection but at
                # the same time it makes the recovery of the full FDStruct 
                # object from it's ID more cumbersome.
                myStructID=myFDStruct.get('id')
            
            FDStructureIDList.append(myStructID) 
  
        # == Code begins ==
        # We will scan the whole vertex list to look for the next loop
        # interaction.
        vertRange=range(len(self['vertices']))
        # If we just start the iterative procedure, then from_vert=-1 and we
        # must look for the "start" loop leg in the entire vertices list
        if not fromVert == -1: 
           if fromPos == -1:
               # If the last loop leg was the vertex output (i.e. last in the 
               # vertex leg list) then we must look for it in the vertices 
               # located after the one where it was found (i.e. from_vert).
               vertRange=vertRange[fromVert+1:]
           else:
               # If the last loop leg was in the vertex inputs (i.e. not last 
               # in the vertex leg list) then we must look where it in the 
               # vertices located before where it was found (i.e. from_vert), 
               # starting from the closest to fromVert (hence the reverse())
               vertRange=vertRange[:fromVert]
               vertRange.reverse()
        # Look in the vertices in vertRange if it can finds the loop leg asked 
        # for.
        for i in vertRange:
            # If the last loop leg was an output of its vertex, we must look for
            # it in the INPUTS of the vertices before. However, it it was an 
            # input of its vertex we must look in the OUTPUT of the vertices 
            # forehead
            legRange=range(len(self['vertices'][i].get('legs')))
            if fromPos == -1:
                # In the last vertex of the list, all entries are input
                if not i==len(self['vertices'])-1:
                    legRange=legRange[:-1]
            else:
                # If looking for an output, then skip the last vertex of the 
                # list which only has inputs.
                if i==len(self['vertices'])-1:
                    continue
                else:
                    legRange=legRange[-1:]
            for j in legRange:
                if self['vertices'][i].get('legs')[j].same(currLeg):
                    vertPos=i    
                    vertFoundID=self['vertices'][i]['id']
                    # If currLeg was just an integer from the first call to
                    # process_next_loop_leg, we can now change it to the Leg 
                    # it really correspond to.
                    if isinstance(currLeg,int):
                        currLeg=base_objects.Leg(self['vertices'][i].get('legs')[j])
                        
                    # We can now process this loop interaction found...
                    for k in filter(lambda ind: not ind==j, \
                                   range(len(self['vertices'][i].get('legs')))):
                        # ..for the structure k
                        # pos gives the direction in which to look for 
                        # nextLoopLeg from vertPos. It is after vertPos
                        # (i.e. then pos=-1) only when the next loop leg was 
                        # found to be the output (i.e. so positioned last in 
                        # the vertex leg list) of the vertex at vertPos. Note that
                        # for the last vertex in the list, all entries are input.
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
                            process_loop_interaction(i,j,k,pos)
                    # Now that we have found loop leg curr_leg, we can get out 
                    # of the two searching loop.
                    break
            if nextLoopLeg:
                break

        # To make sure we found the next loop vertex
        if not nextLoopLeg:
            # Returns False in case of a malformed diagram where it has been 
            # impossible to find the loop leg looked for.
            return False

        # The FDStructureIDList can be empty in case of an identity vertex. 
        # We need to skip the vertex construction and the tag actualization
        # in that case
        if FDStructureIDList and vertFoundID not in [0,-1]:
            # We now have constructed all the FDStructures attached at this
            # vertex of the loop and we have identified the two loop legs. 
            # So we can add the corresponding vertex to loopVertexList
                
            # Create the list of legs from the FDStructures
            myleglist=base_objects.LegList([copy.copy(\
                        structRep[FDindex]['binding_leg']) for FDindex in \
                        FDStructureIDList])

                
            # Add The original loop leg we started from. We either take it 
            # from starting leg (at the first call of process_next_loop_leg)
            # or from the output leg of the latest Leg we added to 
            # loopVertexList. Also, the tag is updated here using the same 
            # rule.
            if loopVertexList:
                self['tag'].append([copy.copy(\
                  loopVertexList[-1]['legs'][-1]),\
                  sorted(FDStructureIDList),vertFoundID])
                myleglist.append(loopVertexList[-1]['legs'][-1])
            else:
                self['tag'].append([copy.copy(currLeg),\
                                     sorted(FDStructureIDList),vertFoundID])
                new_input_leg = copy.copy(currLeg)
                if fromPos!=-1:
                    # In this case the currLeg is an *output* of the current
                    # loop vertex (the last loop vertex must have been a 2-point
                    # dummy one otherwise loopVertexList wouldn't be empty). 
                    # To have this leg as an *input* of the loop vertex we are
                    # constructing with generate_loop_vertex, we must switch 
                    # the id of the new_input_leg to its corresponding anti pdg.
                    new_input_leg.set('id',model.get_particle(
                                   new_input_leg.get('id')).get_anti_pdg_code())
                myleglist.append(new_input_leg)
                    
            # Now depending we reached the last loop vertex or not, we will 
            # create a current (with ref_dict_to1) or a wavefunction plus 
            # a trivial two-point amplitude with interaction id=-1 which
            # plays the role of a conventional amplitude. This allow for
            # having only wavefunctions in the loop and therefore compute
            # the loop lorentz trace easily.
            # WARNING: This is very important here that the endLeg has the 
            # maximal attribute 'number' among all other legs, because this 
            # guarantees that its number is NOT propagated and that as soon 
            # as we reach this number, we reached the EXTERNAL outter leg 
            # which set the end of the tagging algorithm.
            loopVertexList.append(\
               self.generate_loop_vertex(myleglist,model,n_initial,vertFoundID))
            # check that the particle/anti-particle is set correctly

        if nextLoopLeg.same(endLeg):
            # Now we can add the corresponding 'fake' amplitude vertex
            # with flagged id = -1
            # If last vertex was dummy, then recover the original leg
            if vertFoundID not in [0,-1]:
                starting_Leg=copy.copy(myleglist[-1])
                legid=model.get_particle(myleglist[-1]['id']).get_anti_pdg_code()
                state=myleglist[-1].get('state')
            else:
                starting_Leg=copy.copy(currLeg)
                legid=model.get_particle(currLeg['id']).get_anti_pdg_code()
                state=currLeg.get('state')

            loopVertexList.append(base_objects.Vertex(\
                {'legs':base_objects.LegList([starting_Leg,\
                 base_objects.Leg({'number': endLeg,
                                   'id': legid,
                                    'state': state,
                                    'loop_line': True})]),
                 'id':-1}))
            # Returns true since we reached the end loop leg. 
            # Again, it is very important that this end loop leg has the 
            # maximal number (see comment above)
            return True
        else:
            # This is where the recursion happens. We have not reached the 
            # end loop leg yet, so we iterate the procedure.
            return self.process_next_loop_leg(structRep, vertPos, legPos, \
                      nextLoopLeg, endLeg, loopVertexList, model, external_legs)

    def synchronize_loop_vertices_with_tag(self,model,n_initial,struct_rep,
                                         lcut_part_number,lcut_antipart_number):
        """ Construct the loop vertices from the tag of the loop diagram."""
        
        if not self['tag']:
            return
        # Easy access point to the interaction dictionary
        ref_dict_to1 = model.get('ref_dict_to1')
        
        # Create the container for the new vertices which create the loop flow
        loopVertexList=base_objects.VertexList()
        for i, t in enumerate(self['tag']):
            # Tag elements are organized like this 
            # (Incoming_loop_leg,[Structures_ID_list],vertex_ID)
            myleglist=base_objects.LegList([copy.copy(\
                       struct_rep[FDindex]['binding_leg']) for FDindex in t[1]])
            if i==0:
                starting_leg=copy.copy(t[0])
                # Remember here that it is crucial to stick to one convention
                # chosen here to be that the lcut leg 'start_number' always
                # is a particle and the lcut leg 'end_number' always is the
                # corresponding anti-particle. (if not self). This is to ensure
                # a correct evaluation of the fermion number for amplitude.
                # Also and alternatively, it would have been possible at this
                # stage to have the end_number and starting_number set to the 
                # same value while assigning the delta in color and lorentz as
                # the structure of this 2-point closing interaction.
                # There would have been one such interaction per particle in the
                # model so it would be natural to create this interaction when
                # importing the model. This is a cleaner implementation which
                # I will be setting up soon.
                if model.get_particle(starting_leg['id']).get('is_part'):
                    starting_leg['number']=lcut_part_number
                    end_number=lcut_antipart_number
                else:
                    starting_leg['number']=lcut_antipart_number       
                    end_number=lcut_part_number             
                starting_leg['state']=True
            else:
                starting_leg=loopVertexList[-1].get('legs')[-1]
            self['tag'][i][0]=starting_leg
            myleglist.append(starting_leg)
            loopVertexList.append(self.generate_loop_vertex(myleglist,
                                                          model,n_initial,t[2]))
        # Now we can add the corresponding 'fake' amplitude vertex
        # with flagged id = -1
        first_leg=copy.copy(loopVertexList[-1].get('legs')[-1])
        sec_leg_id=model.get_particle(first_leg['id']).get_anti_pdg_code()
        second_leg=base_objects.Leg({'number': end_number,
                                     'id': sec_leg_id,
                                     'state': first_leg.get('state'),
                                     'loop_line': True})
        loopVertexList.append(base_objects.Vertex(\
            {'legs':base_objects.LegList([first_leg,second_leg]),
             'id':-1}))

        self['type'] = abs(first_leg['id'])
        self['vertices'] = loopVertexList    
    
    def construct_FDStructure(self, fromVert, fromPos, currLeg, FDStruct):
        """ Construct iteratively a Feynman Diagram structure attached to a Loop, 
        given at each step a vertex and the position of the leg this function is 
        called from. At the same time, it constructs a canonical representation 
        of the structure which is a tuple with each element corresponding to 
        a 2-tuple ((external_parent_legs),vertex_ID). The external parent legs 
        tuple is ordered as growing and the construction of the canonical 
        representation is such that the 2-tuples appear in a fixed order.
        This functions returns a tuple of 2-tuple like above for the vertex 
        where currLeg was found or false if fails.

        To illustrate this algorithm, we take a concrete example, 
        the following structure:
                                                                       
                      4 5 6 7
                   1 3 \/2 \/  <- Vertex ID, left=73 and right=99
                   \ / | \ /   <- Vertex ID, left=34 and right=42 
                    |  |4 | 
                    1\ | /2
                      \|/      <- Vertex ID=72
                       |
                       |1

        For this structure with external legs (1,2,3,5,6,7) and current created 
        1, the canonical tag will be
            
         (((1,2,3,4,5,6,7),72),((1,3),34),((2,6,7),42),((6,7),99),((4,5),73))
        """
        nextLeg = None
        legPos=-2
        vertPos=-2

        vertRange=range(len(self['vertices']))

        # Say we are at the beginning of the structure reconstruction algorithm 
        # of the structure above, with currLeg=1 so it was found in the vertex 
        # ID=72 with legs (1,1,4,2). Then, this function will call itself on
        # the particles 1,4 and 2. Each of these calls will return a list of 
        # 2-tuples or a simple integer being the leg ID for the case of an 
        # external line, like leg 4 in our example.
        # So the two lists of 2-tuples returned will be put in the list 
        # "reprBuffer". In fact the 2-tuple are nested in another 2-tuple with 
        # the first element being the legID of the current vertex. This helps 
        # the sorting of these 2-tuple in a growing order of their originating
        # legID. In this example, once the procedure is finished with vertex 
        # ID=72, reprBuffer would be:
        #  [(((1,3),34),),(((4,5),73),),(((2,6,7),42),((6,7),99))] 
        # (Still needs to be sorted and later transformed to a tuple)
        # The 2-tuple corresponding to the mother vertex (so ID=72 in the
        # example) is constructed in vertBuffer (the parent lines list is 
        # progressevely filled with the identified external particle of each 
        # leg). and will be put in front of vertBuffer and then transformed to
        # a tuple to form the output of the function.
        vertBuffer=[]

        # Each of the parent legs identified for this vertex are put in the 
        # first element of a list called here parentBufer.
        # The second element stores the vertex ID where currLeg was found.
        parentBuffer=[[],0]

        # If fromPos == -1 then the leg was an output of its vertex so we must 
        # look for it in the vertices following fromVert. If the leg was an
        # input of its vertex then we must look for it in the vertices 
        # preceding fromVert.
        if fromPos == -1:
            # If the last loop leg was the vertex output (i.e. last in the 
            # vertex leg list) then we must look for it in the vertices 
            # located after the one where it was found (i.e. from_vert).
            vertRange=vertRange[fromVert+1:]
        else:
            # If the last loop leg was in the vertex inputs (i.e. not last 
            # in the vertex leg list) then we must look where it in the 
            # vertices located before where it was found (i.e. from_vert) 
            # starting from the clostest to the actual vertex 
            # (hence the reverse())
            vertRange=vertRange[:fromVert]
            vertRange.reverse()
        
        # The variable below serves two purposes:
        # 1) It labels the position of the particle in the vertex (-1 = output)
        # 2) If at the end equals to -2, then it means that the particle looked 
        #    for has not been found.
        pos=-2

        # Helper function
        def process_leg(vertID, legID):
            """ Treats the leg equal to currLeg found in the place located by 
            self['vertices'][vertID].get('legs')[legID]"""
            
            # The id of the vertex where currLeg was found is stored in the 
            # second element of parentBuffer.
            parentBuffer[1]=self['vertices'][vertID].get('id')
            # We can add this vertex to the FDStructure vertex list, in the 
            # "right" order so that a virtual particle in the inputs of some 
            # vertex appears always AFTER the vertex where this particle was the
            # output.
             
            # Now we must continue the iterative procedure for each of the other
            # leg of the vertex found.
            legPos=-2
            for k in [ind for ind in \
                range(len(self['vertices'][vertID].get('legs'))) if ind!=legID]:
                # If we found currLeg in an identity vertex we directly skip it
                # for what regards the construction of the cannonical 
                # representation.
                if not self['vertices'][vertID].get('id'):
                    return self.construct_FDStructure(vertID, k,\
                              self['vertices'][vertID].get('legs')[k], FDStruct)

                if k==len(self['vertices'][vertID].get('legs'))-1 \
                   and not vertID==len(self['vertices'])-1:
                    legPos=-1
                else:
                    legPos=k
                # We get here the structure of each branch of the actual vertex.   
                branch=self.construct_FDStructure(i, legPos, \
                              self['vertices'][vertID].get('legs')[k], FDStruct)
                if not branch:
                    raise self.PhysicsObjectError, \
                          "Failed to reconstruct a FDStructure."
                # That means that this branch was an external leg. 
                if isinstance(branch,int):
                    parentBuffer[0].append(branch)
                # If it is a list it means that the branch contains at least 
                # one further vertex.
                elif isinstance(branch,tuple):
                    parentBuffer[0]+=list(branch[0][0])
                    vertBuffer.append(branch)
                else:
                    raise self.PhysicsObjectError, \
                    "Non-proper behavior of the construct_FDStructure function"
            return legPos

        # == Beginning of the code ==
        # Look the vertices in vertRange if it can find the parents of currLeg
        # once it is found call the function below process_leg
        for i in vertRange:
            # We must look in the output of these vertices if the leg was 
            # previously found as an input of its vertex. In case it was an 
            # output of its vertices, then we must look in the inputs of 
            # these vertices. Remember that the last vertex of the list has only
            # inputs.
            legRange=range(len(self['vertices'][i].get('legs')))
            if fromPos == -1:
                # In the last vertex of the list, all entries are input
                if not i==len(self['vertices'])-1:
                    legRange=legRange[:-1]
            else:
                # If looking for an output, then skip the last vertex of the 
                # list which only has inputs.
                if i==len(self['vertices'])-1:
                    continue
                else:
                    legRange=legRange[-1:]

            # Breaking off a double nested loop using findVert. A neater way of 
            # doing it would be to use exceptions.
            findVert=False
            # Now search over the leg range for currLeg
            for j in legRange:
                if self['vertices'][i].get('legs')[j].same(currLeg):
                    # Now call the function to process the leg found.                
                    pos=process_leg(i,j)
                    # Now that we have found the vertex with currLeg and treated
                    # it, we must get out of the searching loop.
                    findVert=True
                    break;
            if findVert:
                break;

        if(pos == -2): 
            if(not fromPos == -1):
                # In this case, the leg has not been found. It is an external leg.
                FDStruct.get('external_legs').append(copy.copy(currLeg))
                return currLeg.get('number')
            else:
                raise self.PhysicsObjectError, \
                                  " A structure is malformed."
        else:
            # In this case a vertex with currLeg has been found and we must
            # return the list of tuple described above. First let's sort the 
            # list so that the branches comes in a fixed order which is 
            # irrelevant but not trivial here. First comes the branches 
            # involving the smallest number of vertices. Among those who have
            # an equal number of vertices, those with the smallest ID for the
            # external legs come first.
            vertBuffer.sort()
            # Now flatten the list to have a list of tuple instead of a list
            # of tuple made of tuples. In the above example, this corresponds
            # to go from 
            # [(((1,3),34),),(((4,5),73),),(((2,6,7),42),((6,7),99))]
            # to
            # [((1,3),34),((4,5),73),((2,6,7),42),((6,7),99)]
            vertBufferFlat=[]
            for t in vertBuffer:
                for u in t:
                    vertBufferFlat.append(u)
                    
            # Sort the parent lines
            parentBuffer[0].sort()
            # Add the 2-tuple corresponding to the vertex where currLeg was found.
            vertBufferFlat.insert(0,(tuple(parentBuffer[0]),parentBuffer[1]))
            return tuple(vertBufferFlat)

    # Helper function

    def get_starting_loop_line(self):
        """ Return the starting loop line of this diagram, i.e. lcut leg one."""
        for v in self['vertices']:
            for l in v['legs']:
                if l['loop_line']:
                    return l
    
    def get_finishing_loop_line(self):
        """ Return the finishing line of this diagram, i.e. lcut leg two.
        Notice that this function is only available when the loop diagram is 
        constructed with the special two-point vertex with id -1. """
        
        assert self['vertices'][-1].get('id')==-1, "Loop diagrams must finish "+\
          " with vertex with id '-1' for get_finishing_loop_line to be called"
        
        return max(self['vertices'][-1].get('legs'), key=lambda l: l['number'])

    def get_loop_line_types(self):
        """ Return a set with one occurence of each different PDG code of the
        particles running in the loop. By convention, the PDF of the particle,
        not the antiparticle, is stored in this list. Using the tag would be 
        quicker, but we want this function to be available before tagging as 
        well"""
        return set([abs(l['id']) for v in self['vertices'] for l in v['legs'] \
                    if l['loop_line']])

    def get_loop_orders(self,model):
        """ Return a dictionary with one entry per type of order appearing in 
        the interactions building the loop flow. The corresponding keys are the
        number of type this order appear in the diagram. """
        
        loop_orders = {}
        for vertex in self['vertices']:
            # We do not count the identity vertex nor the vertices building the
            # external FDStructures (possibly left over if not synchronized with
            # the tag).
            if vertex['id'] not in [0,-1] and len([1 for leg \
                                     in vertex['legs'] if leg['loop_line']])==2:
                vertex_orders = model.get_interaction(vertex['id'])['orders']
                for order in vertex_orders.keys():
                    if order in loop_orders.keys():
                        loop_orders[order]+=vertex_orders[order]
                    else:
                        loop_orders[order]=vertex_orders[order]
        return loop_orders
    

    @classmethod
    def make_canonical_cyclic(cls,atag):
        """ Perform cyclic permutations on the tag given in parameter such that 
        the structure with the lowest ID appears first."""
        
        if not atag:
            return []

        imin=-2
        minStructID=-2
        for i, part in enumerate(atag):
            if minStructID==-2 or min(part[1])<minStructID:
                minStructID=min(part[1])
                imin=i
        
        atag=atag[imin:]+atag[:imin]

        return atag

    @classmethod
    def mirrored_tag(cls,atag, model):
        """ Performs a mirror operation on A COPY of the tag and returns it. """
        
        if not atag:
            return []

        # Make a local copy (since we will act on the leg object of the tag)
        revTag=[(copy.deepcopy(elem[0]), copy.copy(elem[1]), \
                 copy.copy(elem[2])) for elem in atag]
        
        # reverse it
        revTag.reverse()
        # shift right all legs
        shiftBuff=revTag[-1]
        for i in range(len(revTag)-1):
            revTag[-(i+1)]=[revTag[-(i+2)][0],revTag[-(i+1)][1],revTag[-(i+1)][2]]
        revTag[0]=[shiftBuff[0],revTag[0][1],revTag[0][2]] 
        # When reading the tag in the opposite direction, all particles will 
        # appear as antiparticle and we need to flip their pdg in order to keep
        # the same convention.
        nonselfantipartlegs = [ elem[0] for elem in revTag if not \
             model.get('particle_dict')[elem[0].get('id')]['self_antipart'] ]
        for leg in nonselfantipartlegs:
            leg.set('id',\
              model.get('particle_dict')[leg.get('id')].get_anti_pdg_code())

        return revTag

    
    # Helper functions for the user_filter in the loop diagram generation. They
    # are not used by any other part of MadLoop.
            
    def get_loop_lines_pdgs(self):
        """ Returns the pdgs of the lines running in the loop while not 
        differentiating the particles from the anti-particles """
        
        return [abs(tag_elem[0].get('id')) for tag_elem in self['tag']]
                
    def get_pdgs_attached_to_loop(self,structs):
        """ Returns the pdgs of the lines directly branching off the loop."""
        
        return [structs.get_struct(struct_ID).get('binding_leg').get('id') \
                       for tag_elem in self['tag'] for struct_ID in tag_elem[1]]

#===============================================================================
# LoopDiagram
#===============================================================================

class LoopUVCTDiagram(base_objects.Diagram):
    """ A special kind of LoopDiagram which does not contain a loop but only
    specifies all UV counter-term which factorize the the same given born
    and bringing in the same orders. UV mass renormalization does not belong to
    this class of counter-term for example, and it is added along with the R2 
    interactions."""

    def default_setup(self):
        """Default values for all properties"""

        super(LoopUVCTDiagram,self).default_setup()
        # These attributes store the specifics of the UV counter-term
        # contribution of this diagram
        self['type']='UV'
        self['UVCT_orders']={}
        self['UVCT_couplings']=[]

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'UVCT_couplings':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list" % str(value)
            else:
                for elem in value:
                    if not isinstance(elem, str) and not isinstance(elem, int):
                        raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)
        
        if name == 'UVCT_orders':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary" % str(value)

        if name == 'type':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)
        
        else:
            super(LoopUVCTDiagram, self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        
        return ['vertices', 'UVCT_couplings', 'UVCT_orders', 'type', 'orders']

    def get_UVCTinteraction(self, model):
        """ Finds the UV counter-term interaction present in this UVCTDiagram """
        
        for vert in self['vertices']:
            if vert.get('id') != 0:
                if model.get_interaction(vert.get('id')).is_UV():
                    return model.get_interaction(vert.get('id'))
                
        return None

    def calculate_orders(self, model):
        """Calculate the actual coupling orders of this diagram. Note
        that the special order WEIGTHED corresponds to the sum of
        hierarchies for the couplings."""

        coupling_orders = dict([(c, 0) for c in model.get('coupling_orders')])
        weight = 0
        for couplings in [model.get('interaction_dict')[vertex.get('id')].\
                        get('orders') for vertex in self['vertices'] if \
                        vertex.get('id') != 0]+[self['UVCT_orders']]:
            for coupling in couplings:
                coupling_orders[coupling] += couplings[coupling]
            weight += sum([model.get('order_hierarchy')[c]*n for \
                              (c,n) in couplings.items()])
        coupling_orders['WEIGHTED'] = weight
        self.set('orders', coupling_orders)

    def nice_string(self):
        """Returns a nicely formatted string of the diagram content."""
        res=''
        if self['vertices']:
            res=res+super(LoopUVCTDiagram,self).nice_string()
        if self['UVCT_couplings']:
            res=res+'UV renorm. vertices: '
            res=res+','.join(str(vert) for vert in self['UVCT_couplings'])+'\n'
        if self['UVCT_orders']:
            res=res+'UVCT orders: '
            res=res+','.join(order for order in self['UVCT_orders'].keys())+'\n'      
        if self['type']:
            res=res+'UVCT type: '+self['type']
            
        return res
    
#===============================================================================
# LoopModel
#===============================================================================
class LoopModel(base_objects.Model):
    """A class to store all the model information with advanced feature
       to compute loop process."""
    
    def __init__(self,*args,**opts):
        """Make sure to copy over the attribute map_CTcoup_CTparam if the 
        first instance used is a LoopModel"""

        if len(args)>0 and isinstance(args[0],LoopModel):
            if hasattr(args[0],'map_CTcoup_CTparam'):
                self.map_CTcoup_CTparam = copy.copy(args[0].map_CTcoup_CTparam)

        super(LoopModel,self).__init__(*args,**opts)

    def default_setup(self):
       super(LoopModel,self).default_setup()
       self['perturbation_couplings'] = []
       # The 'coupling_orders_counterterms' has all coupling orders
       # as keys and values are tuple of the form:
       #    (loop_particles, counterterm, laurent_order)
       # where loop_particles are defined as usual:
       #    [[lpartID1, lpartID2, ...], [lpartID1bis, lpartID2bis, ...],...]
       # and the counterterm is a string giving the name of the coupling
       # representing the counterterm and finally 'laurent_order' is to which
       # laurent order this counterterm contributes.
       self['coupling_orders_counterterms']={}
       
       # This attribute is not registered as a key to this object's dictionary
       # because it is not a new physical attribute to the model.
       # It is the mapping between couplings (in values of the dict) and the
       # list of CTparameter names which enter in its expression (in the keys).
       if not hasattr(self,'map_CTcoup_CTparam'):
           self.map_CTcoup_CTparam = {}
       
    
    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'perturbation_couplings':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a list" % \
                                                            type(value)
            for order in value:
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "Object of type %s is not a string" % \
                                                            type(order)
        else:
            super(LoopModel,self).filter(name,value)
        
        return True

    def actualize_dictionaries(self, useUVCT=False):
        """This function actualizes the dictionaries"""

        if useUVCT:
            [self['ref_dict_to0'], self['ref_dict_to1']] = \
              self['interactions'].generate_ref_dict(useR2UV=False,useUVCT=True)
        else:
            [self['ref_dict_to0'], self['ref_dict_to1']] = \
                    self['interactions'].generate_ref_dict() 
        self['ref_dict_to0'].update(
                                self['particles'].generate_ref_dict())

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
            super(DGLoopLeg,self).filter(name,value)
        
        return True

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

        self['vertices'] = base_objects.VertexList()
        self['id'] = -1 
        self['external_legs'] = base_objects.LegList()
        self['canonical'] = ()
        self['binding_leg']= base_objects.Leg()

    def is_external(self):
        """Returns wether the structure is simply made of an external particle
        only"""
        if (len(self['canonical'])==1 and self['canonical'][0][1]==0):
            return True
        else:
            return False

    def filter(self, name, value):
        """Filter for valid FDStructure property values."""

        if name == 'vertices':
            if not isinstance(value, base_objects.VertexList):
                raise self.PhysicsObjectError, \
        "%s is not a valid VertexList object" % str(value)

        if name == 'id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
        "id %s is not an integer" % repr(value)

        if name == 'weight':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
        "weight %s is not an integer" % repr(value)

        if name == 'external_legs':
            if not isinstance(value, base_objects.LegList):
                raise self.PhysicsObjectError, \
        "external_legs %s is not a valid Leg List" % str(value)

        if name == 'binding_leg':
            if not isinstance(value, base_objects.Leg):
                raise self.PhysicsObjectError, \
        "binding_leg %s is not a valid Leg" % str(value)

        if name == 'canonical':
            if not isinstance(value, tuple):
                raise self.PhysicsObjectError, \
        "canonical %s is not a valid tuple" % str(value)

        return True
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['id','external_legs','binding_leg','canonical','vertices']

    def nice_string(self):
        """Returns a nicely formatted string of the structure content."""

        mystr=''

        if not self['id']==-1:
            mystr=mystr+'id: '+str(self['id'])+',\n'
        else:
            return '()'

        if self['canonical']:
            mystr=mystr+'canonical_repr.: '+str(self['canonical'])+',\n'

        if self['external_legs']:
            mystr=mystr+'external_legs: { '
            for leg in self['external_legs'][:-1]:
                mystr = mystr + str(leg['number']) + '(%s)' % str(leg['id']) \
                        + ', '
            mystr = mystr + str(self['external_legs'][-1]['number']) + \
                    '(%s)' % str(self['external_legs'][-1]['id']) + ' },\n'
            mystr = mystr+'binding_leg: '+str(self['binding_leg']['number']) +\
                    '(%s)' % str(self['binding_leg']['id'])
        return mystr

    def nice_string_vertices(self):
        """Returns a nicely formatted string of the structure vertices."""
        mystr=''
        if self['vertices']:
            mystr = mystr+'('
            for vert in self['vertices']:
                mystr = mystr + '('
                for leg in vert['legs'][:-1]:
                    mystr = mystr + str(leg['number']) + \
                            '(%s)' % str(leg['id']) + ','
                mystr = mystr[:-1] + '>'
                mystr = mystr + str(vert['legs'][-1]['number']) +\
                                '(%s)' % str(vert['legs'][-1]['id']) + ','
                mystr = mystr + 'id:' + str(vert['id']) + '),'
            mystr = mystr[:-1] + ')'
            return mystr
        elif len(self['external_legs'])==1:
            return '('+str(self['external_legs'][0]['number'])+\
                                    '('+str(self['external_legs'][0]['id'])+'))'
        else:
            return '()'    


    def generate_vertices(self, model, external_legs=None):
        """ This functions generate the vertices building this structure, 
        starting from the outter legs going towards the binding leg. 
        It uses the interactions dictionaries from the model. """

        if isinstance(model, base_objects.Process):
            assert  external_legs is None
            #retro-compatible way to call the function
            external_legs= model.get('legs')
            model = model['model']
        assert external_legs is not None
        assert isinstance(model, base_objects.Model)
            

        
        # First empty the existing vertices
        self.set('vertices',base_objects.VertexList())

        tag=copy.copy(self['canonical'])

        # Define easy access points
        ref_dict_to1 = model.get('ref_dict_to1')

        if not tag:
            raise self.PhysicsObjectError, \
        "The canonical tag of the FD structure is not set yet, so that the "+\
        "reconstruction of the vertices cannot be performed."

        # Create a local copy of the external legs
        leglist = copy.deepcopy(external_legs)

        # Create a dictionary to get an easy access to a given particle number
        legDict={}
        for leg in leglist:
            legDict[leg['number']]=leg

        # If this structure is directly an external leg, then there is no vertex 
        # to add
        if len(tag)==1 and len(tag[0][0])==1:
            # But we should still define the binding leg
            self['binding_leg']=copy.deepcopy(legDict[tag[0][0][0]])
            return

        # Reverse the tag to start from the outter legs
        tag=list(tag)
        tag.reverse()

        # Change the tuples to lists and convert the particle numbers to their
        # corresponding LegList object
        for i, tagelem in enumerate(tag):
            tag[i]=list(tagelem)
            tag[i][0]=base_objects.LegList([legDict[myleg] for myleg in \
                                            tagelem[0]])

        # For each element of the tag, combine them with the appropriate vertex
        # ID, create and add the corresponding vertex to the structure's vertex
        # list, remove this element of the tag and substitutes the leg number
        # in all other tag's elements by the new leg number created.
        while tag:
            # First get an easy access to the LegList of the first tag element
            # we aim at treating.
            legs=tag[0][0]

            # Now we make sure we can combine those legs together
            key=tuple(sorted([leg.get('id') for leg in legs]))
            if ref_dict_to1.has_key(key):
                for interaction in ref_dict_to1[key]:
                    # Find the interaction with the right ID
                    if interaction[1]==tag[0][1]:
                        # Create the output Leg and add it to the existing list
                        # 1) id is like defined by ref_dict_to1
                        legid = interaction[0]
                        # 2) number is the minimum of leg numbers involved in the
                        # combination
                        number = min([leg.get('number') for leg in legs])
                        # 3) state is final, unless there is exactly one initial 
                        # state particle involved in the combination -> t-channel
                        if len(filter(lambda leg: leg.get('state') == False,
                                  legs)) == 1:
                            state = False
                        else:
                            state = True
                        legs.append(base_objects.Leg({'number': number,\
                                                      'id': legid,\
                                                      'state': state,
                                                      'loop_line': False}))
                        # Now we can add the corresponding vertex
                        self.get('vertices').append(base_objects.Vertex(\
                                             {'legs':legs,'id':interaction[1]}))
                        break

                # In all further elements, we should replace any combination of 
                # the legs just merged here by the new output leg we just created.
                for i, tagelement in enumerate(tag[1:]):
                    Found=False
                    for leg in legs[:-1]:
                        try:
                            tag[i+1][0].remove(leg)
                            Found=True
                        except Exception:
                            pass
                    if Found:
                        tag[i+1][0].append(legs[-1])

                # If we are about to empty the tag we must now set the
                # binding_leg as the last one we produced.
                if len(tag)==1:
                    self['binding_leg']=copy.deepcopy(legs[-1])

                # Now we should remove this first element of the tag that we 
                # just treated
                tag.pop(0)

            else:
                raise self.PhysicsObjectError, \
        "The canonical tag of the FD structure is corrupted because one "+\
        "interaction does not exist."

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
              "The ID %s specified for get_struct is not an integer or tuple"%\
                                                                    repr(object)

    def nice_string(self):
        """Returns a nicely formatted string"""
        mystr = str(len(self)) + ' FD Structures:\n'
        for struct in self:
            mystr = mystr + "  " + struct.nice_string() + '\n'
        return mystr[:-1]


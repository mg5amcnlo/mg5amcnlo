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

"""Definitions of objects inheriting from the classes defined in
helas_objects.py and which have special attributes and function 
devoted to the treatment of Loop processes"""

import array
import copy
import logging
import itertools
import math

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects

#===============================================================================
# 
#===============================================================================

logger = logging.getLogger('madgraph.helas_objects')

#===============================================================================
# LoopHelasAmplitude
#===============================================================================
class LoopHelasAmplitude(helas_objects.HelasAmplitude):
    """LoopHelasAmplitude object, behaving exactly as an amplitude except that
       it also contains loop wave-functions closed on themselves, building an
       amplitude corresponding to the closed loop.
    """

    # Customized constructor
    def __init__(self, *arguments):
        """Constructor for the LoopHelasAmplitude. For now, it works exactly
           as for the HelasMatrixElement one."""
        
        if arguments:
            super(LoopHelasAmplitude, self).__init__(arguments)
        else:
            super(LoopHelasAmplitude, self).__init__()        

    def default_setup(self):
        """Default values for all properties"""
                
        super(LoopHelasAmplitude,self).default_setup()
        
        # Store the wavefunctions building this loop
        self['wavefunctions'] = helas_objects.HelasWavefunctionList()
        self['amplitudes'] = helas_objects.HelasAmplitudeList()

    def filter(self, name, value):
        """Filter for valid LoopHelasAmplitude property values."""

        if name=='wavefunctions':
            if not value.isinstance(helas_objects.HelasWaveFunctionList):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list of HelasWaveFunctions" % str(value)
            for wf in value:
                if not wf['is_loop']:
                    raise self.PhysicsObjectError, \
                      "Wavefunctions from a LoopHelasAmplitude must be from a loop."
        
        elif name=='amplitudes':
            if not value.isinstance(helas_objects.HelasAmplitudeList):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list of HelasAmplitudes" % str(value)
        else:
            return super(LoopHelasAmplitude,self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return LoopHelasAmplitude property names as a nicely sorted list."""

        return super(LoopHelasAmplitude,self).get_sorted_keys()+\
               ['wavefunctions', 'amplitudes']

#===============================================================================
# LoopHelasMatrixElement
#===============================================================================
class LoopHelasMatrixElement(helas_objects.HelasMatrixElement):
    """LoopHelasMatrixElement: list of processes with identical Helas
    calls, and the list of LoopHelasDiagrams associated with the processes.
    It works as for the HelasMatrixElement except for the loop-related features
    which are defined here. """

    def default_setup(self):
        """Default values for all properties"""
        
        super(LoopHelasMatrixElement,self).default_setup()

    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if False:
            pass
        else:
            return super(LoopHelasMatrixElement,self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return super(LoopHelasMatrixElement,self).get_sorted_keys()

    # Customized constructor
    
    def __init__(self, amplitude=None, optimization=1,
                 decay_ids=[], gen_color=True):
        """Constructor for the LoopHelasMatrixElement. For now, it works exactly
           as for the HelasMatrixElement one."""

        super(LoopHelasMatrixElement, self).__init__(amplitude, optimization,\
                                                     decay_ids, gen_color)

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    
    def __eq__(self, other):
        """Comparison between different loop matrix elements. It works exactly as for
           the HelasMatrixElement for now."""

        return super(LoopHelasMatrixElement,self).__eq__(other)

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

    def generate_helas_diagrams(self, amplitude, optimization=1,
                                decay_ids=[]):
        """Starting from a list of LoopDiagrams from the diagram
        generation, generate the corresponding LoopHelasDiagrams, i.e.,
        the wave functions and amplitudes (for the loops and their R2 and UV
        counterterms). Choose between default optimization (= 1, maximum 
        recycling of wavefunctions) or no optimization (= 0, no recycling of
        wavefunctions, useful for GPU calculations with very restricted memory).

        Note that we need special treatment for decay chains, since
        the end product then is a wavefunction, not an amplitude.
        """
        
        assert  isinstance(amplitude, loop_diagram_generation.LoopAmplitude), \
                    "Bad arguments for generate_helas_diagrams in LoopHelasMatrixElement"
        assert isinstance(optimization, int), \
                    "Bad arguments for generate_helas_diagrams in LoopHelasMatrixElement"

        structures = amplitude.get('structure_repository')
        
        process = amplitude.get('process')
        has_born = amplitude.get('has_born')          
        
        model = process.get('model')

        # All the previously defined wavefunctions
        wavefunctions = []
        
        # List of dictionaries from struct ID to wave function,
        # keeps track of the structures already scanned.
        # The key is the struct ID and the value infos is the tuple
        # (wfs, colorlists). 'wfs' is the list of wavefunctions,
        # one for each color-lorentz structure of the FDStructure.
        # Same for the 'colorlists', everything appearing
        # in the same order in these lists
        structID_to_infos = {}
        
        # List of minimal information for comparison with previous
        # wavefunctions
        wf_mother_arrays = []
        # Keep track of wavefunction number
        wf_number = 0

        # Generate wavefunctions for the external particles
        external_wavefunctions = dict([(leg.get('number'),
                                        helas_objects.HelasWavefunction(\
                                        leg, 0, model, decay_ids)) \
                                        for leg in process.get('legs')])

        # Initially, have one wavefunction for each external leg.
        wf_number = len(process.get('legs'))

        # For initial state bosons, need to flip part-antipart
        # since all bosons should be treated as outgoing
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.is_boson() and wf.get('state') == 'initial' and \
               not wf.get('self_antipart'):
                wf.set('is_part', not wf.get('is_part'))

        # For initial state particles, need to flip PDG code (if has
        # antipart)
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.get('leg_state') == False and \
               not wf.get('self_antipart'):
                wf.flip_part_antipart()

        # Now go through the diagrams, looking for undefined wavefunctions

        helas_diagrams = helas_objects.HelasDiagramList()

        # Keep track of amplitude number and diagram number
        amplitude_number = 0
        diagram_number = 0
            
        def process_born_diagram(diagram, wfNumber, amplitudeNumber):
            """ Helper function to process a born diagrams exactly as it is done in 
            HelasMatrixElement for tree-level diagrams."""
            
            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = helas_objects.HelasWavefunctionList()

            vertices = copy.copy(diagram.get('vertices'))

            # Single out last vertex, since this will give amplitude
            lastvx = vertices.pop()

            # Go through all vertices except the last and create
            # wavefunctions
            for vertex in vertices:

                # In case there are diagrams with multiple Lorentz/color 
                # structures, we need to keep track of the wavefunctions
                # for each such structure separately, and generate
                # one HelasDiagram for each structure.
                # We use the array number_to_wavefunctions to keep
                # track of this, with one dictionary per chain of
                # wavefunctions
                # Note that all wavefunctions relating to this diagram
                # will be written out before the first amplitude is written.
                new_number_to_wavefunctions = []
                new_color_lists = []
                for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                     color_lists):
                    legs = copy.copy(vertex.get('legs'))
                    last_leg = legs.pop()
                    # Generate list of mothers from legs
                    mothers = self.getmothers(legs, number_wf_dict,
                                              external_wavefunctions,
                                              wavefunctions,
                                              diagram_wavefunctions)
                    inter = model.get('interaction_dict')[vertex.get('id')]

                    # Now generate new wavefunction for the last leg

                    # Need one amplitude for each color structure,
                    done_color = {} # store link to color
                    for coupl_key in sorted(inter.get('couplings').keys()):
                        color = coupl_key[0]
                        if color in done_color:
                            wf = done_color[color]
                            wf.get('coupling').append(inter.get('couplings')[coupl_key])
                            wf.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                            continue
                        wf = helas_objects.HelasWavefunction(last_leg, \
                               vertex.get('id'), model)
                        wf.set('coupling', [inter.get('couplings')[coupl_key]])
                        if inter.get('color'):
                            wf.set('inter_color', inter.get('color')[coupl_key[0]])
                        done_color[color] = wf
                        wf.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        wf.set('color_key', color)
                        wf.set('mothers', mothers)
                        # Need to set incoming/outgoing and
                        # particle/antiparticle according to the fermion flow
                        # of mothers
                        wf.set_state_and_particle(model)

                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        # Also need to keep track of the wavefunction number.
                        wf, wfNumber = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wfNumber)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)
                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError:
                            # Update wf number
                            wfNumber = wfNumber + 1
                            wf.set('number', wfNumber)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wfNumber
                                wfNumber = wfNumber - 1
                            except ValueError:
                                diagram_wavefunctions.append(wf)

                        new_number_wf_dict[last_leg.get('number')] = wf

                        # Store the new copy of number_wf_dict
                        new_number_to_wavefunctions.append(\
                                                        new_number_wf_dict)
                        # Add color index and store new copy of color_lists
                        new_color_list = copy.copy(color_list)
                        new_color_list.append(coupl_key[0])
                        new_color_lists.append(new_color_list)

                number_to_wavefunctions = new_number_to_wavefunctions
                color_lists = new_color_lists

            # Generate all amplitudes corresponding to the different
            # copies of this diagram
            helas_diagram = helas_objects.HelasDiagram()
                        
            for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                  color_lists):

                # Now generate HelasAmplitudes from the last vertex.
                if lastvx.get('id'):
                    inter = model.get_interaction(lastvx.get('id'))
                    keys = sorted(inter.get('couplings').keys())
                    pdg_codes = [p.get_pdg_code() for p in \
                                 inter.get('particles')]
                else:
                    # Special case for decay chain - amplitude is just a
                    # placeholder for replaced wavefunction
                    inter = None
                    keys = [(0, 0)]
                    pdg_codes = None

                # Find mothers for the amplitude
                legs = lastvx.get('legs')
                mothers = self.getmothers(legs, number_wf_dict,
                                          external_wavefunctions,
                                          wavefunctions,
                                          diagram_wavefunctions).\
                                             sort_by_pdg_codes(pdg_codes, 0)[0]
                # Need to check for clashing fermion flow due to
                # Majorana fermions, and modify if necessary
                wfNumber = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              None,
                                              wfNumber,
                                              False,
                                              number_to_wavefunctions)
                done_color = {}
                for i, coupl_key in enumerate(keys):
                    color = coupl_key[0]
                    if inter and color in done_color.keys():
                        amp = done_color[color]
                        amp.get('coupling').append(inter.get('couplings')[coupl_key])
                        amp.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                        continue
                    amp = helas_objects.HelasAmplitude(lastvx, model)
                    if inter:
                        amp.set('coupling', [inter.get('couplings')[coupl_key]])
                        amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        if inter.get('color'):
                            amp.set('inter_color', inter.get('color')[color])
                        amp.set('color_key', color)
                        done_color[color] = amp
                    amp.set('mothers', mothers)
                    amplitudeNumber = amplitudeNumber + 1
                    amp.set('number', amplitudeNumber)
                    # Add the list with color indices to the amplitude
                    new_color_list = copy.copy(color_list)
                    if inter:
                        new_color_list.append(color)
                        
                    amp.set('color_indices', new_color_list)

                    # Add amplitude to amplitdes in helas_diagram
                    helas_diagram.get('amplitudes').append(amp)

            # After generation of all wavefunctions and amplitudes,
            # add wavefunctions to diagram
            helas_diagram.set('wavefunctions', diagram_wavefunctions)

            # Sort the wavefunctions according to number
            diagram_wavefunctions.sort(lambda wf1, wf2: \
                          wf1.get('number') - wf2.get('number'))

            if optimization:
                wavefunctions.extend(diagram_wavefunctions)
                wf_mother_arrays.extend([wf.to_array() for wf \
                                         in diagram_wavefunctions])
            else:
                wfNumber = len(process.get('legs'))

            # Return the diagram obtained
            return helas_diagram, wfNumber, amplitudeNumber

        def process_struct(sID, diag_wfs, wfNumber):
            """ Scan a structure, create the necessary wavefunctions, add them
            to the diagram wavefunctions list, and return a list of bridge
            wavefunctions (i.e. those attached to the loop) with a list, ordered
            in the same way, of color lists. Each element of these lists
            correspond to one choice of color-lorentz structure of this
            tree-structure #sID. """

            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree structure.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Bridge wavefunctions
            bridg_wfs = helas_objects.HelasWaveFunctionList()

            vertices = copy.copy(structures[sID].get('vertices'))

            # Go through all vertices except the last and create
            # wavefunctions
            for i, vertex in enumerate(vertices):

                # In case there are diagrams with multiple Lorentz/color 
                # structures, we need to keep track of the wavefunctions
                # for each such structure separately, and generate
                # one HelasDiagram for each structure.
                # We use the array number_to_wavefunctions to keep
                # track of this, with one dictionary per chain of
                # wavefunctions
                # Note that all wavefunctions relating to this diagram
                # will be written out before the first amplitude is written.
                new_number_to_wavefunctions = []
                new_color_lists = []
                for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                     color_lists):
                    legs = copy.copy(vertex.get('legs'))
                    last_leg = legs.pop()
                    # Generate list of mothers from legs
                    mothers = self.getmothers(legs, number_wf_dict,
                                              external_wavefunctions,
                                              wavefunctions,
                                              diagram_wavefunctions)
                    inter = model.get('interaction_dict')[vertex.get('id')]

                    # Now generate new wavefunction for the last leg

                    # Need one amplitude for each color structure,
                    done_color = {} # store link to color
                    for coupl_key in sorted(inter.get('couplings').keys()):
                        color = coupl_key[0]
                        if color in done_color:
                            wf = done_color[color]
                            wf.get('coupling').append(inter.get('couplings')[coupl_key])
                            wf.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                            continue
                        wf = HelasWavefunction(last_leg, vertex.get('id'), model)
                        wf.set('coupling', [inter.get('couplings')[coupl_key]])
                        if inter.get('color'):
                            wf.set('inter_color', inter.get('color')[coupl_key[0]])
                        done_color[color] = wf
                        wf.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        wf.set('color_key', color)
                        wf.set('mothers', mothers)
                        # Need to set incoming/outgoing and
                        # particle/antiparticle according to the fermion flow
                        # of mothers
                        wf.set_state_and_particle(model)
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        # Also need to keep track of the wavefunction number.
                        wf, wfNumber = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wfNumber)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError:
                            # Update wf number
                            wfNumber = wfNumber + 1
                            wf.set('number', wfNumber)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wfNumber
                                wfNumber = wfNumber - 1
                            except ValueError:
                                diagram_wavefunctions.append(wf)

                        new_number_wf_dict[last_leg.get('number')] = wf
                        if i==(len(vertices)-1):
                            # Last vertex of the structure so we should define
                            # the bridge wavefunctions.
                            bridg_wfs.append(wf)
                        # Store the new copy of number_wf_dict
                        new_number_to_wavefunctions.append(\
                                                        new_number_wf_dict)
                        # Add color index and store new copy of color_lists
                        new_color_list = copy.copy(color_list)
                        new_color_list.append(coupl_key[0])
                        new_color_lists.append(new_color_list)

                number_to_wavefunctions = new_number_to_wavefunctions
                color_lists = new_color_lists
                
            return (bridge_wfs, color_lists), wfNumber
        
        def getloopmothers(loopWfsIn, structIDs, color_list, diag_wfs, wfNumber):
            """From the incoming loop leg and the list of structures IDs 
            connected to the loop at this point, it generates the list of
            mothers, a list of colorlist and a number_to_wavefunctions
            dictionary list for which each element correspond to one 
            lorentz-color structure of the tree-structure attached to the loop.
            It will launch the reconstruction procedure of the structures 
            which have not been encountered yet."""
    
            # The mothers list and the color lists There is one element in these 
            # lists, in the same order, for each combination of the 
            # lorentz-color tree-structures of the FDStructures attached to
            # this point.
            mothers_list = [loopWfsIn]
            color_lists = [color_list]

            # Scanning of the FD tree-structures attached to the loop at this
            # point.
            for sID in structID:
                try:
                   struct_infos = structID_to_infos[sID]
                except KeyError:
                    # The structure has not been encountered yet, we must
                    # scan it
                    # Not done yet
                    struct_infos, wfNumber = \
                      process_struct(sID, diag_wfs, wfNumber)
                    if optimization:
                        # Only if there is optimization the dictionary is
                        # because otherwise we must always rescan the
                        # structures to correctly add all the necessary
                        # wavefunctions to the diagram wavefunction list
                        structID_to_infos[sID]=(copy.copy(struct_infos[0]),
                                                  copy.copy(struct_infos[1]))
                # The orig object are those already existing before treating
                # this structure
                new_mothers_list = []
                new_color_lists = []
                for mothers, orig_color_list in zip(mothers_list, color_lists):
                    for struct_wf, struct_color_list in struct_infos:
                        new_color_list = copy.copy(orig_color_list)+\
                                         copy.copy(struct_color_list)
                        new_mothers = copy.copy(mothers)
                        new_mothers.append(struct_wf)
                        new_color_lists.append(new_color_list)
                        new_mothers_list.append(new_mothers)
                mothers_list = new_mothers_list       
                color_lists = new_color_lists
                
            return (mothers_list, color_lists), wfNumber
                             
        def process_loop_diagram(diagram, wfNumber, amplitudeNumber):
            """ Helper function to process a the loop diagrams which features
            several different aspects compared to the tree born diagrams."""

            # Initialize here the loop helas diagram we are about to create
            helas_diagram = helas_objects.HelasDiagram()
            
            # Also create here the LoopHelasAmplitude associated to this loop
            # diagram.
            loop_helas_amplitude = LoopHelasAmplitude()

            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the loop.
            # We only need to retain the last loop wavefunctions created
            # This is a list to store all the last loop wavefunctions created
            # due to the possibly many color-lorentz structure of the last
            # loop vertex.
            last_loop_wfs = helas_objects.HelasWavefunctionList()

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]
            
            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = helas_objects.HelasWavefunctionList()

            # Copy the original tag of the loop which contains all the necessary
            # information
            tag = copy.deepcopy(diagram.get('tag'))
            
            # Single out last tag element, since this will give amplitude
            lastTagElem = tag.pop()
            
            # Copy the ct vertices of the loop
            ct_vertices = copy.copy(diagram.get('CT_vertices'))      

            # First create the starting external loop leg
            last_loop_wfs.append(\
              helas_objects.HelasWavefunction(tag[0][0], 0, model, decay_ids))
            diagram_wavefunctions.append(last_loop_wfs[0])

            def process_tag_elem(tagElem, wfNumber, lastloopwfs, colorlists):
                """Treat one tag element of the loop diagram (not the last one
                   which provides an amplitude)"""
                   
                # We go through all the structures generated during the 
                # exploration of the structures attached at this point
                # of the loop. Let's define the new color_lists and
                # last_loop_wfs we will use for next iteration
                new_color_lists = []
                new_last_loop_wfs = helas_objects.HelasWavefunctionList()
                
                # In case there are diagrams with multiple Lorentz/color 
                # structures, we need to keep track of the wavefunctions
                # for each such structure separately, and generate
                # one HelasDiagram for each structure.
                # We use the array number_to_wavefunctions to keep
                # track of this, with one dictionary per chain of
                # wavefunctions
                # Note that all wavefunctions relating to this diagram
                # will be written out before the first amplitude is written.
                vertex=tagElem[2]
                structureIDs=tagElem[1]
                for last_loop_wf, color_list in zip(lastloopwfs,
                                                     colorlists):
                    loopLegOut = copy.copy(vertex.get('legs')[-1])
   
                    # From the incoming loop leg and the struct IDs, it generates
                    # a list of mothers, colorlists and number_to_wavefunctions
                    # dictionary for which each element correspond to one 
                    # lorentz-color structure of the tree-structure attached to
                    # the loop.
                    (motherslist, colorlists), wfNumber = \
                      getloopmothers([last_loop_wf,], structIDs,\
                                           color_list, diagram_wavefunctions)
                    
                    inter = model.get('interaction_dict')[vertex.get('id')]

                    # Now generate new wavefunctions for the last leg

                    for mothers, structcolorlist in zip(motherslist, colorlists):
                        # Need one amplitude for each color structure,
                        done_color = {} # store link to color
                        for coupl_key in sorted(inter.get('couplings').keys()):
                            color = coupl_key[0]
                            if color in done_color:
                                wf = done_color[color]
                                wf.get('coupling').append(inter.get('couplings')[coupl_key])
                                wf.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                                continue
                            wf = helas_objects.HelasWavefunction(loopLegOut, \
                                                 vertex.get('id'), model)
                            wf.set('coupling', [inter.get('couplings')[coupl_key]])
                            if inter.get('color'):
                                wf.set('inter_color', inter.get('color')[coupl_key[0]])
                            done_color[color] = wf
                            wf.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                            wf.set('color_key', color)
                            wf.set('mothers', mothers)
                            # Need to set incoming/outgoing and
                            # particle/antiparticle according to the fermion flow
                            # of mothers
                            wf.set_state_and_particle(model)
                            # Need to check for clashing fermion flow due to
                            # Majorana fermions, and modify if necessary
                            # Also need to keep track of the wavefunction number.
                            wf, wfNumber = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wfNumber)

                            # Store wavefunction
                            try:
                                wf = diagram_wavefunctions[\
                                       diagram_wavefunctions.index(wf)]
                            except ValueError:
                                # Update wf number
                                wfNumber = wfNumber + 1
                                wf.set('number', wfNumber)
                                # We directly add it to the diagram
                                # wavefunctions because loop wavefunctions are
                                # never reused.
                                diagram_wavefunctions.append(wf)

                            # Update the last_loop_wfs list with the loop wf
                            # we just created. 
                            new_last_loop_wfs.append(wf)

                            # Add color index and store new copy of color_lists
                            new_color_list = copy.copy(color_list)+\
                                               copy.copy(structcolorlist)
                            new_color_list.append(coupl_key[0])
                            new_color_lists.append(new_color_list)
                
                # We update the lastloopwfs list and the color_lists for the
                # next iteration, i.e. the treatment of the next loop vertex
                # by returning them to the calling environnement.
                return wfNumber, new_last_loop_wfs, new_color_lists

 
            # Go through all vertices except the last and create
            # wavefunctions
            
            def process_last_tag_Elem(tagElem, wfNumber, amplitudeNumber):
                """Treat the last tag element of the loop diagram (which 
                provides an amplitude)"""
                # First create the other external loop leg closing the loop.
                lastvx=lastTagElem[2]  
                lastvx_structureIDs=lastTagElem[1]
                other_external_loop_wf=HelasWavefunction()
                for leg in [leg for leg in lastvx['legs'] if leg['loop_line']]:
                    if last_loop_wfs[0]['number_external']!=leg['number']:
                        other_external_loop_wf=\
                          HelasWavefunction(leg, 0, model, decay_ids)
                        break
                # It is a loop wf so we add it anyway to the diag ones.
                diagram_wavefunctions.append(other_external_loop_wf)
                
                for last_loop_wf, color_list in zip(last_loop_wfs,color_lists):
                    # Now generate HelasAmplitudes from the last vertex.
                    if not lastvx.get('id'):
                        raise self.PhysicsObjectError, \
                          "A decay chain placeholder was found in a loop diagram." 
                    inter = model.get_interaction(lastvx.get('id'))
                        
                    keys = sorted(inter.get('couplings').keys())
                    pdg_codes = [p.get_pdg_code() for p in \
                                     inter.get('particles')]
     
                    (motherslist, colorlists), wfNumber = getloopmothers(\
                      [other_external_loop_wf,last_loop_wf], \
                      lastvx_structureIDs, color_list, diagram_wavefunctions)
                    
                    for mothers, structcolorlist in zip(motherslist, colorlists):
                        mothers.sort_by_pdg_codes(pdg_codes, 0)[0]      
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        wfNumber = mothers.check_and_fix_fermion_flow(wavefunctions,
                                                  diagram_wavefunctions,
                                                  external_wavefunctions,
                                                  None,
                                                  wfNumber,
                                                  False,
                                                  [])
                        done_color = {}
                        for i, coupl_key in enumerate(keys):
                            color = coupl_key[0]
                            if color in done_color.keys():
                                amp = done_color[color]
                                amp.get('coupling').append(inter.get('couplings')[coupl_key])
                                amp.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                                continue
                            amp = helas_objects.HelasAmplitude(lastvx, model)
                            amp.set('coupling', [inter.get('couplings')[coupl_key]])
                            amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                            if inter.get('color'):
                                amp.set('inter_color', inter.get('color')[color])
                            amp.set('color_key', color)
                            done_color[color] = amp 
                            amp.set('mothers', mothers)
                            amplitudeNumber = amplitudeNumber + 1
                            amp.set('number', amplitudeNumber)
                            # Add the list with color indices to the amplitude
                            new_color_list = copy.copy(color_list)+\
                                               copy.copy(structcolorlist)
                            new_color_list.append(color)
                            amp.set('color_indices', new_color_list)
                            # Add this amplitude to the LoopHelasAmplitude of this
                            # diagram.
                            loop_helas_amplitude.get('amplitudes').append(amp)
                return wfNumber, amplitudeNumber
                
            def process_counterterms(ct_vertices, wfNumber, amplitudeNumber):
                """Process the counterterms vertices defined in this loop
                   diagram."""
                
                structIDs=[]
                for tagElem in tag:
                    structIDs += tagElem[2]
                
                # Here we call getloopmothers without any incoming loop
                # wavefunctions such that the function will return exactly
                # the mother of the counter-term amplitude we wish to create
                # We start with an empty color list as well in this case
                (motherslist, colorlists), wfNumber = getloopmothers([], structIDs, \
                                            [], diagram_wavefunctions, wfNumber)
                          
                for mothers, structcolorlist in zip(motherslist, colorlists):
                    for ct_vertex in ct_vertices:
                        # Now generate HelasAmplitudes from this ct_vertex.
                        inter = model.get_interaction(ct_vertex.get('id'))
                        keys = sorted(inter.get('couplings').keys())
                        pdg_codes = [p.get_pdg_code() for p in \
                                     inter.get('particles')]
                        mothers.sort_by_pdg_codes(pdg_codes, 0)[0]
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        wfNumber = mothers.check_and_fix_fermion_flow(wavefunctions,
                                                  diagram_wavefunctions,
                                                  external_wavefunctions,
                                                  None,
                                                  wfNumber,
                                                  False,
                                                  [])
                        done_color = {}
                        for i, coupl_key in enumerate(keys):
                            color = coupl_key[0]
                            if color in done_color.keys():
                                amp = done_color[color]
                                amp.get('coupling').append(inter.get('couplings')[coupl_key])
                                amp.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                                continue
                            amp = HelasAmplitude(ct_vertex, model)
                            amp.set('coupling', [inter.get('couplings')[coupl_key]])
                            amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                            if inter.get('color'):
                                amp.set('inter_color', inter.get('color')[color])
                            amp.set('color_key', color)
                            done_color[color] = amp
                            amp.set('mothers', mothers)
                            amplitudeNumber = amplitudeNumber + 1
                            amp.set('number', amplitudeNumber)
                            # Add the list with color indices to the amplitude
                            amp_color_list = copy.copy(color_list)
                            amp_color_list.append(color)     
                            amp.set('color_indices', amp_color_list)
                            
                            # Add amplitude to amplitdes in helas_diagram
                            helas_diagram.get('amplitudes').append(amp)
                return wfNumber, amplitudeNumber
            
            for tagElem in tag:
                print 'tagElem=',tagElem
                wfNumber, last_loop_wfs, color_lists = \
                  process_tag_elem(tagElem, wf_number, last_loop_wfs, color_lists)

            # Generate all amplitudes corresponding to the different
            # copies of this diagram
            wfNumber, amplitudeNumber = process_last_tag_Elem(lastTagElem, \
                                          wf_number, amplitude_number)

### ====
###   Perform here further necessary treatment to the loop_helas_amplitude at
###   the end of the function above.                    
### ====

            # Add now the counter-terms vertices
            wfNumber, amplitudeNumber = process_counterterms(ct_vertices)

            # Split the diagram wavefunctions among the loop ones which will go
            # to the LoopHelasAmplitude and the others from the structures which
            # will fill the 'wavefunctions' list of the diagram
            
            struct_wfs=helas_objects.HelasWaveFunctionList(\
                      [wf for wf in diagram_wavefunctions if not wf['is_loop']])
            loop_wfs=helas_objects.HelasWaveFunctionList(\
                          [wf for wf in diagram_wavefunctions if wf['is_loop']])      

            # Sort the wavefunctions according to number
            struct_wfs.sort(lambda wf1, wf2: \
                          wf1.get('number') - wf2.get('number'))
            loop_wfs.sort(lambda wf1, wf2: \
                          wf1.get('number') - wf2.get('number'))
        
            loop_helas_amplitude.set('wavefunctions',loop_wfs)
            helas_diagram.get('amplitudes').append(loop_helas_amplitude)
                     
            # After generation of all wavefunctions and amplitudes,
            # add wavefunctions to diagram
            helas_diagram.set('wavefunctions', struct_wfs)

            # Of course we only allow to reuse the struct wavefunctions but
            # never the loop ones which have to be present and reused in each
            # loop diagram
            if optimization:
                wavefunctions.extend(struct_wfs)
                wf_mother_arrays.extend([wf.to_array() for wf \
                                         in struct_wfs])
            else:
                wfNumber = len(process.get('legs'))

            # Return the diagram obtained
            return helas_diagram, wfNumber, amplitudeNumber      

        # Let's first treat the born diagrams
                
        if has_born:
            for diagram in amplitude.get('born_diagrams'):
                helBornDiag, wf_number, amplitude_number=\
                  process_born_diagram(diagram, wf_number, amplitude_number)
                diagram_number = diagram_number + 1
                helBornDiag.set('number', diagram_number)
                helas_diagrams.append(helBornDiag)
        
        # Now we treat the loop diagrams
        for diagram in amplitude.get('loop_diagrams'):
            loopHelDiag, wf_number, amplitude_number=\
              process_loop_diagram(diagram, wf_number, amplitude_number)
            diagram_number = diagram_number + 1
            loopHelDiag.set('number', diagram_number)
            helas_diagrams.append(loopHelDiag)

        self.set('diagrams', helas_diagrams)

        # Sort all mothers according to the order wanted in Helas calls
        for wf in self.get_all_wavefunctions():
            wf.set('mothers', helas_objects.HelasMatrixElement.sorted_mothers(wf))

        for amp in self.get_all_amplitudes():
            amp.set('mothers', helas_objects.HelasMatrixElement.sorted_mothers(amp))
            amp.set('color_indices', amp.get_color_indices())

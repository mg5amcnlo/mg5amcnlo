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
class LoopHelasAmplitude(base_objects.PhysicsObject):
    """LoopHelasAmplitude object, behaving exactly as an amplitude except that
       it also contains loop wave-functions closed on themselves, building an
       amplitude corresponding to the closed loop.
    """
    
    def default_setup(self):
        """Default values for all properties"""

        super(LoopHelasAmplitude,self).default_setup()

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if False:
            pass
        else:
            return super(LoopHelasAmplitude,self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return super(LoopHelasAmplitude,self).get_sorted_keys()

    # Customized constructor
    def __init__(self, *arguments):
        """Constructor for the LoopHelasMatrixElement. For now, it works exactly
           as for the HelasMatrixElement one."""

        super(LoopHelasAmplitude, self).__init__(arguments)


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

        #born_diagram_list = amplitude.get('born_diagrams')
        loop_diagram_list = amplitude.get('loop_diagrams')
        
        process = amplitude.get('process')
        has_born = amplitude.get('has_born')          
        
        model = process.get('model')
        if not loop_diagram_list:
            return

        # All the previously defined wavefunctions
        wavefunctions = []
        # List of minimal information for comparison with previous
        # wavefunctions
        wf_mother_arrays = []
        # Keep track of wavefunction number
        wf_number = 0

        # Generate wavefunctions for the external particles
        external_wavefunctions = dict([(leg.get('number'),
                                        HelasWavefunction(leg, 0, model,
                                                          decay_ids)) \
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

        helas_diagrams = HelasDiagramList()

        # Keep track of amplitude number and diagram number
        amplitude_number = 0
        diagram_number = 0

        def process_born_diagram(self, diagram):
            """ Helper function to process a born diagrams exactly as it is done in 
            HelasMatrixElement for tree-level diagrams."""
                
            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()

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
                        wf, wf_number = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wf_number)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError:
                            # Update wf number
                            wf_number = wf_number + 1
                            wf.set('number', wf_number)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wf_number
                                wf_number = wf_number - 1
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
            helas_diagram = HelasDiagram()
            diagram_number = diagram_number + 1
            helas_diagram.set('number', diagram_number)
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
                wf_number = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              None,
                                              wf_number,
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
                    amp = HelasAmplitude(lastvx, model)
                    if inter:
                        amp.set('coupling', [inter.get('couplings')[coupl_key]])
                        amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        if inter.get('color'):
                            amp.set('inter_color', inter.get('color')[color])
                        amp.set('color_key', color)
                        done_color[color] = amp
                    amp.set('mothers', mothers)
                    amplitude_number = amplitude_number + 1
                    amp.set('number', amplitude_number)
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
                wf_number = len(process.get('legs'))

            # Append this diagram in the diagram list
            helas_diagrams.append(helas_diagram)
            
        def process_loop_diagram(self, diagram):
            """ Helper function to process a the loop diagrams which features
            several different aspects compared to the tree born diagrams."""

            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()

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
                        wf, wf_number = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wf_number)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError:
                            # Update wf number
                            wf_number = wf_number + 1
                            wf.set('number', wf_number)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wf_number
                                wf_number = wf_number - 1
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
            helas_diagram = HelasDiagram()
            diagram_number = diagram_number + 1
            helas_diagram.set('number', diagram_number)
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
                wf_number = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              None,
                                              wf_number,
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
                    amp = HelasAmplitude(lastvx, model)
                    if inter:
                        amp.set('coupling', [inter.get('couplings')[coupl_key]])
                        amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        if inter.get('color'):
                            amp.set('inter_color', inter.get('color')[color])
                        amp.set('color_key', color)
                        done_color[color] = amp
                    amp.set('mothers', mothers)
                    amplitude_number = amplitude_number + 1
                    amp.set('number', amplitude_number)
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
                wf_number = len(process.get('legs'))

            # Append this diagram in the diagram list
            helas_diagrams.append(helas_diagram)        

        # Let's first treat the born diagrams
        
        if has_born:
            for diagram in amplitude.get('born_diagrams'):
                process_born_diagrams(diagram)
        
        # Now we treat the loop diagrams
        for diagram in amplitude.get('loop_diagrams'):
            process_loop_diagram(diagram)
        
        self.set('diagrams', helas_diagrams)

        # Sort all mothers according to the order wanted in Helas calls
        for wf in self.get_all_wavefunctions():
            wf.set('mothers', HelasMatrixElement.sorted_mothers(wf))

        for amp in self.get_all_amplitudes():
            amp.set('mothers', HelasMatrixElement.sorted_mothers(amp))
            amp.set('color_indices', amp.get_color_indices())

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

import aloha

import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.loop.loop_color_amp as loop_color_amp
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects

#===============================================================================
# 
#===============================================================================

logger = logging.getLogger('madgraph.helas_objects')

#===============================================================================
# LoopUVCTHelasAmplitude
#===============================================================================
class LoopHelasUVCTAmplitude(helas_objects.HelasAmplitude):
    """LoopHelasUVCTAmplitude object, behaving exactly as an amplitude except that
       it also contains additional vertices with coupling constants corresponding
       to the 'UVCTVertices' defined in the 'UVCTVertices ' of the 
       loop_base_objects.LoopUVCTDiagram of the LoopAmplitude. These are stored
       in the additional attribute 'UVCT_interaction_ids' of this class.
    """
    
    # Customized constructor
    def __init__(self, *arguments):
        """Constructor for the LoopHelasAmplitude. For now, it works exactly
           as for the HelasMatrixElement one."""
        
        if arguments:           
            super(LoopHelasUVCTAmplitude, self).__init__(*arguments)
        else:
            super(LoopHelasUVCTAmplitude, self).__init__() 
    
    def default_setup(self):
        """Default values for all properties"""
                
        super(LoopHelasUVCTAmplitude,self).default_setup()
        
        # Store interactions ID of the UV counterterms related to this diagram
        self['UVCT_couplings'] = []
        self['UVCT_orders'] = {}

    def filter(self, name, value):
        """Filter for valid LoopHelasAmplitude property values."""

        if name=='UVCT_couplings':
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list for UVCT_couplings" % str(value)
            for id in value:
                if not isinstance(id, str) and not isinstance(id, int):
                    raise self.PhysicsObjectError, \
                      "%s is not a valid string or integer for UVCT_couplings" % str(value)
                      
        if name == 'UVCT_orders':
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary" % str(value)

        if name == 'type':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)

        else:
            return super(LoopHelasUVCTAmplitude,self).filter(name, value)

    def get_sorted_keys(self):
        """Return LoopHelasAmplitude property names as a nicely sorted list."""

        return super(LoopHelasUVCTAmplitude,self).get_sorted_keys()+\
               ['UVCT_couplings','UVCT_orders','type']

        return True

    def get_call_key(self):
        """ Exactly as a regular HelasAmplitude except that here we must add 
        an entry to mutliply the final result by the coupling constants of the
        interaction in UVCT_couplings if there are any"""
        original_call_key = super(LoopHelasUVCTAmplitude,self).get_call_key()
        
        if self.get_UVCT_couplings()=='1.0d0':
            return original_call_key
        else:
            return (original_call_key[0],original_call_key[1],'UVCT')

    def get_used_UVCT_couplings(self):
        """ Returns a list of the string UVCT_couplings defined for this
        amplitudes. """
        return [coupl for coupl in self['UVCT_couplings'] if \
                isinstance(coupl,str)]

    def get_UVCT_couplings(self):
        """ Returns the string corresponding to the overall UVCT coupling which
        factorize this amplitude """
        if self['UVCT_couplings']==[]:
            return '1.0d0'

        answer=[]
        integer_sum=0
        for coupl in list(set(self['UVCT_couplings'])):
            if isinstance(coupl,int):
                integer_sum+=coupl
            else:
                answer.append(str(len([1 for c in self['UVCT_couplings'] if \
                                   c==coupl]))+'.0d0*'+coupl)
        if integer_sum!=0:
            answer.append(str(integer_sum)+'.0d0')
        if answer==[] and (integer_sum==0 or integer_sum==1):
            return '1.0d0'
        else:
            return '+'.join(answer)

    def get_base_diagram(self, wf_dict, vx_list = [], optimization = 1):
        """Return the loop_base_objects.LoopUVCTDiagram which corresponds to this
        amplitude, using a recursive method for the wavefunctions."""

        vertices = super(LoopHelasUVCTAmplitude,self).get_base_diagram(\
                     wf_dict, vx_list, optimization)['vertices']

        return loop_base_objects.LoopUVCTDiagram({'vertices': vertices, \
                                    'UVCT_couplings': self['UVCT_couplings'], \
                                    'UVCT_orders': self['UVCT_orders'], \
                                    'type': self['type']})
        
    def get_helas_call_dict(self, index=1):
        """ return a dictionary to be used for formatting
        HELAS call. """
        
        
        out = helas_objects.HelasAmplitude.get_helas_call_dict(self, 
                                                                    index=index)
        out['uvct'] = self.get_UVCT_couplings()
        return out

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

    def is_equivalent(self, other):
        """Comparison between different LoopHelasAmplitude in order to recognize
        which ones are equivalent at the level of the file output.
        I decided not to overload the operator __eq__ to be sure not to interfere
        with other functionalities of the code."""

        if(len(self.get('wavefunctions'))!=len(other.get('wavefunctions')) or
           len(self.get('amplitudes'))!=len(other.get('amplitudes')) or
           [len(wf.get('coupling')) for wf in self.get('wavefunctions')]!=
           [len(wf.get('coupling')) for wf in other.get('wavefunctions')] or
           [len(amp.get('coupling')) for amp in self.get('amplitudes')]!=
           [len(amp.get('coupling')) for amp in other.get('amplitudes')]):
            return False
        
        wfArgsToCheck = ['fermionflow','lorentz','state','onshell','spin',\
                         'self_antipart','color']
        for arg in wfArgsToCheck:
            if [wf.get(arg) for wf in self.get('wavefunctions')]!=\
               [wf.get(arg) for wf in other.get('wavefunctions')]:
                return False

        ampArgsToCheck = ['lorentz',]
        for arg in ampArgsToCheck:
            if [amp.get(arg) for amp in self.get('amplitudes')]!=\
               [amp.get(arg) for amp in other.get('amplitudes')]:
                return False
        
        # Finally just check that the loop and external mother wavefunctions
        # of the loop wavefunctions and loop amplitudes arrive at the same places 
        # in both self and other. The characteristics of the mothers is irrelevant,
        # the only thing that matters is that the loop-type and external-type mothers
        # are in the same order.
        if [[m.get('is_loop') for m in lwf.get('mothers')] for lwf in self.get('wavefunctions')]!=\
           [[m.get('is_loop') for m in lwf.get('mothers')] for lwf in other.get('wavefunctions')]:
            return False
        if [[m.get('is_loop') for m in lwf.get('mothers')] for lwf in self.get('amplitudes')]!=\
           [[m.get('is_loop') for m in lwf.get('mothers')] for lwf in other.get('amplitudes')]:
            return False
        
        return True

    def default_setup(self):
        """Default values for all properties"""
                
        super(LoopHelasAmplitude,self).default_setup()
        
        # Store the wavefunctions building this loop
        self['wavefunctions'] = helas_objects.HelasWavefunctionList()
        # In this first version, a LoopHelasAmplitude is always built out of
        # a single amplitude, but later one could imagine resumming many 
        # contribution to one CutTools call and having many HelasAmplitudes.
        self['amplitudes'] = helas_objects.HelasAmplitudeList()
        # The pairing is used for the output to know at each loop interactions
        # how many non-loop mothers are necessary. This list is ordered as the
        # helas calls building the loop
        self['pairing'] = []
        # To keep the 'type' (L-cut particle ID) of the LoopDiagram this
        # Loop amplitude tracks.
        # In principle this info is recoverable from the loop wfs.
        self['type'] = -1
        # To store the symmetry factor of the loop
        self['loopsymmetryfactor'] = 0

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'loopsymmetryfactor' and not self[name]:
            self.calculate_loopsymmetryfactor()

        return super(LoopHelasAmplitude, self).get(name)
        
    def filter(self, name, value):
        """Filter for valid LoopHelasAmplitude property values."""

        if name=='wavefunctions':
            if not isinstance(value, helas_objects.HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list of HelasWaveFunctions" % str(value)
            for wf in value:
                if not wf['is_loop']:
                    raise self.PhysicsObjectError, \
                      "Wavefunctions from a LoopHelasAmplitude must be from a loop."
        
        elif name=='amplitudes':
            if not isinstance(value, helas_objects.HelasAmplitudeList):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list of HelasAmplitudes" % str(value)
        
        elif name=='type':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                  "%s is not a valid integer for the attribute 'type'" % str(value) 
           
        elif name == 'loopsymmetryfactor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for loopsymmetryfactor" % \
                        str(value)
        else:
            return super(LoopHelasAmplitude,self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return LoopHelasAmplitude property names as a nicely sorted list."""

        return super(LoopHelasAmplitude,self).get_sorted_keys()+\
               ['wavefunctions', 'amplitudes']

    def get_base_diagram(self, wf_dict, vx_list = [], optimization = 1):
        """Return the loop_base_objects.LoopDiagram which corresponds to this
        amplitude, using a recursive method for the wavefunctions.
        Remember that this diagram is not tagged and structures are not
        recognized."""

        vertices = self['amplitudes'][0].get_base_diagram(\
                     wf_dict, vx_list, optimization)['vertices']

        return loop_base_objects.LoopDiagram({'vertices': vertices,\
                                              'type':self['type']})   

    def set_mothers_and_pairing(self):
        """ Sets the mothers of this amplitude in the same order as they will
        be used in the arguments of the helas calls building this loop"""
        
        if len(self.get('amplitudes'))!=1:
            self.PhysicsObjectError, \
                  "HelasLoopAmplitude is for now designed to contain only one \
                   HelasAmplitude"
        
        self.set('mothers',helas_objects.HelasWavefunctionList())
        # Keep in mind that the two first wavefunctions are the L-cut particles
        for lwf in self.get('wavefunctions')[2:]:
            mothersList=[wf for wf in lwf.get('mothers') if not wf['is_loop']]
            self['mothers'].extend(mothersList)
            self['pairing'].append(len(mothersList))
        mothersList=[wf for wf in self.get('amplitudes')[0].get('mothers') \
                            if not wf['is_loop']]
        self['mothers'].extend(mothersList)
        self['pairing'].append(len(mothersList))

    def get_masses(self):
        """ Returns the list of the masses of the loop particles as they should
        appear for cuttools (L-cut particles specified last) """
        
        masses=[]
        if not aloha.complex_mass:
            for wf in self.get('wavefunctions')[2:]:
                masses.append(wf.get('mass'))
            masses.append(self.get('wavefunctions')[0].get('mass'))
        else:
            for wf in self.get('wavefunctions')[2:]:
                if (wf.get('width') == 'ZERO' or wf.get('mass') == 'ZERO'):
                    masses.append(wf.get('mass'))
                else: 
                    masses.append('CMASS_%s' % wf.get('mass'))
        return masses

    def get_couplings(self):
        """ Returns the list of the couplings of the different helas objects
        building this HelasLoopAmplitude. They are ordered as they will appear
        in the helas calls."""

        return (sum([wf.get('coupling') for wf in self.get('wavefunctions') if \
             wf.get('coupling')!=['none']],[])\
             +sum([amp.get('coupling') for amp in self.get('amplitudes')],[]))

    def get_call_key(self):
        """ The helas call to a loop is simple and only depends on the number
        of loop lines and mothers. This how it is reflected in the call key. """
        
        return ("LOOP",len(self.get('wavefunctions'))-1,\
                len(self.get('mothers')),len(self.get('coupling')))

    def get_rank(self):
        """ Returns the rank of the loop numerator, i.e. the maximum power to which
        the loop momentum is elevated in the loop numerator. The way of returning it
        here is only valid for the simple interactions of the SM. The completely
        general approach needs to use aloha to gather the information of the
        'lorentz' object stored in each loop vertex and it will be implemented later."""
        
        rank=0
        # First add one power for each fermion propagator
        rank=rank+len([ wf for wf in self.get('wavefunctions') if \
                       wf.get('mothers') and wf.is_fermion()])
        # Add one if the L-cut particle is a fermion
        if True in [ wf.is_fermion() for wf in self.get('wavefunctions') if \
                       not wf.get('mothers')]:
            rank=rank+1
        # Add one for each three-boson vertex
        rank=rank+len([ wf for wf in self.get('wavefunctions') if \
                       wf.is_boson() and len([w for w in wf.get('mothers') \
                         if w.is_boson()])==2])
        # Counting the amplitude as well (there is only one normally)
        rank=rank+len([ amp for amp in self.get('amplitudes') if \
                        len([w for w in amp.get('mothers') \
                         if w.is_boson()])==3])
        return rank

    def calculate_fermionfactor(self):
        """ Overloading of the function of the mother class as it might be necessary
        to modify it for fermion loops."""
        super(LoopHelasAmplitude,self).calculate_fermionfactor()

    def calculate_loopsymmetryfactor(self):
        """ Calculate the loop symmetry factor. For now it is hard-coded function valid
        for the SM only where all symmetry factors are 1 except for the gluon bubble which
        exhibits a factor 2."""
        
        if len(self.get('wavefunctions'))==3 and \
           len([wf for wf in self.get('wavefunctions') if wf.get('pdg_code')==21])==3:
            self['loopsymmetryfactor']=2
        else:
            self['loopsymmetryfactor']=1
        
#===============================================================================
# LoopHelasDiagram
#===============================================================================
class LoopHelasDiagram(helas_objects.HelasDiagram):
    """LoopHelasDiagram object, behaving exactly as a Diagram except that
       it has a couple of additional functions which can reconstruct and
       handle loop amplitudes.
    """

    def get_regular_amplitudes(self):
        """ Quick access to ALL non-loop amplitudes, including those which are
        inside the LoopAmplitudes defined in this diagram."""
        
        ampList=helas_objects.HelasAmplitudeList()
        for loopAmp in self.get_loop_amplitudes():
            ampList.extend(loopAmp['amplitudes'])
        ampList.extend(self.get_ct_amplitudes())
        return ampList
              
    def get_ct_amplitudes(self):
        """ Quick access to the regular amplitudes defined directly in this
            diagram (not in the LoopAmplitudes). Usually they correspond to the
            counter-terms. """
        
        return helas_objects.HelasAmplitudeList([amp for amp in \
          self['amplitudes'] if not isinstance(amp, LoopHelasAmplitude)]) 

    def get_loop_amplitudes(self):
        """ Quick access to the loop amplitudes only"""
        
        return helas_objects.HelasAmplitudeList([amp for amp in \
          self['amplitudes'] if isinstance(amp, LoopHelasAmplitude)])

    def get_loop_UVCTamplitudes(self):
        """ Quick access to the loop amplitudes only"""
        
        return helas_objects.HelasAmplitudeList([amp for amp in \
          self['amplitudes'] if isinstance(amp, LoopHelasUVCTAmplitude)])

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

        # Store separately the color basis for the loop and born diagrams
        self['born_color_basis'] = loop_color_amp.LoopColorBasis()
        self['loop_color_basis'] = loop_color_amp.LoopColorBasis()

    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if name=='born_color_basis' or name=='loop_color_basis':
            if not isinstance(value,color_amp.ColorBasis):
                raise self.PhysicsObjectError, \
                  "%s is not a valid color basis" % str(value)
        else:
            return super(LoopHelasMatrixElement,self).filter(name, value)

        return True

    def get(self,name):
        """Overload in order to return the loop_color_basis when simply asked
        for color_basis. The setter is not updated to avoid side effects."""
        
        if name=='color_basis':
            return self['loop_color_basis']
        else:
            return super(LoopHelasMatrixElement,self).get(name)
        
    def process_color(self):
        """ Perform the simple color processing from a single matrix element 
        (without optimization then). This is called from the initialization
        and overloaded here in order to have the correct treatment """
        
        # Generation of helas objects is assumed to be finished so we can relabel
        # optimaly the 'number' attribute of these objects.
        self.relabel_helas_objects()
        self.get('loop_color_basis').build_loop(self.get('base_amplitude'))
        if self.get('base_amplitude')['process']['has_born']:
            self.get('born_color_basis').build_born(self.get('base_amplitude'))
            self.set('color_matrix',\
                     color_amp.ColorMatrix(self.get('loop_color_basis'),\
                                           self.get('born_color_basis')))  
        else:
            self.set('color_matrix',\
                     color_amp.ColorMatrix(self.get('loop_color_basis')))

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['processes', 'identical_particle_factor',
                'diagrams', 'born_color_basis','loop_color_basis',
                'color_matrix','base_amplitude', 'has_mirror_process']

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
            
        def process_born_diagram(diagram, wfNumber, amplitudeNumber, UVCTdiag=False):
            """ Helper function to process a born diagrams exactly as it is done in 
            HelasMatrixElement for tree-level diagrams. This routine can also
            process LoopUVCTDiagrams, and if so the argument UVCTdiag must be set
            to true"""
            
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
                        wf.set('mothers',mothers)
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
            if not UVCTdiag:
                helas_diagram = helas_objects.HelasDiagram()
            else:
                helas_diagram = LoopHelasDiagram()                
                        
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
                    if not UVCTdiag:
                        amp = helas_objects.HelasAmplitude(lastvx, model)
                    else:
                        amp = LoopHelasUVCTAmplitude(lastvx, model)
                        amp.set('UVCT_orders',diagram.get('UVCT_orders'))
                        amp.set('UVCT_couplings',diagram.get('UVCT_couplings'))
                        amp.set('type',diagram.get('type'))
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
            bridge_wfs = helas_objects.HelasWavefunctionList()

            vertices = copy.copy(structures[sID].get('vertices'))

            # First treat the special case of a structure made solely of one
            # external leg
            if len(vertices)==0:
                binding_leg=copy.copy(structures[sID]['binding_leg'])
                binding_wf = self.getmothers(base_objects.LegList([binding_leg,]),
                                              {},
                                              external_wavefunctions,
                                              wavefunctions,
                                              diag_wfs)
                # Simply return the wf of this external leg along with an 
                # empty color list
                return [(binding_wf[0],[])] ,wfNumber

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
                                              diag_wfs)
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
                        wf = helas_objects.HelasWavefunction(last_leg, vertex.get('id'), model)
                        wf.set('coupling', [inter.get('couplings')[coupl_key]])
                        if inter.get('color'):
                            wf.set('inter_color', inter.get('color')[coupl_key[0]])
                        done_color[color] = wf
                        wf.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                        wf.set('color_key', color)
                        wf.set('mothers',mothers)
                        ###print "in process_struct and adding wf with"
                        ###print "    mothers id:"
                        ###for ii, mot in enumerate(mothers):
                        ###    print "    mother ",ii,"=",mot['number_external'],"("+str(mot.get_pdg_code())+") number=",mot['number']
                        ###print "    and iself =",wf['number_external'],"("+str(wf.get_pdg_code())+") number=",wf['number']                       
                        # Need to set incoming/outgoing and
                        # particle/antiparticle according to the fermion flow
                        # of mothers
                        wf.set_state_and_particle(model)
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        # Also need to keep track of the wavefunction number.
                        wf, wfNumber = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diag_wfs,
                                                   external_wavefunctions,
                                                   wfNumber)
                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diag_wfs[\
                                    diag_wfs.index(wf)]
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
                                diag_wfs.append(wf)

                        new_number_wf_dict[last_leg.get('number')] = wf
                        if i==(len(vertices)-1):
                            # Last vertex of the structure so we should define
                            # the bridge wavefunctions.
                            bridge_wfs.append(wf)
                        # Store the new copy of number_wf_dict
                        new_number_to_wavefunctions.append(\
                                                        new_number_wf_dict)
                        # Add color index and store new copy of color_lists
                        new_color_list = copy.copy(color_list)
                        new_color_list.append(coupl_key[0])
                        new_color_lists.append(new_color_list)

                number_to_wavefunctions = new_number_to_wavefunctions
                color_lists = new_color_lists
            
            ###print "bridg wfs returned="
            ###for wf in bridge_wfs:
            ###    print "    bridge =",wf['number_external'],"("+str(wf.get_pdg_code())+") number=",wf['number']
            
            return zip(bridge_wfs, color_lists), wfNumber
        
        def getloopmothers(loopWfsIn, structIDs, color_list, diag_wfs, wfNumber):
            """From the incoming loop leg(s) and the list of structures IDs 
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
            mothers_list = [loopWfsIn,]
            color_lists = [color_list,]

            # Scanning of the FD tree-structures attached to the loop at this
            # point.
            for sID in structIDs:
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
                        structID_to_infos[sID]=copy.copy(struct_infos)
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
            
            ###print "getloop mothers returned with sID", structIDs
            ###print "len mothers_list=",len(mothers_list)
            ###for wf in mothers_list[0]:
            ###    print "    mother =",wf['number_external'],"("+str(wf.get_pdg_code())+") number=",wf['number']  

            return (mothers_list, color_lists), wfNumber
                             
        def process_loop_diagram(diagram, wavefunctionNumber, amplitudeNumber):
            """ Helper function to process a the loop diagrams which features
            several different aspects compared to the tree born diagrams."""

            # Initialize here the loop helas diagram we are about to create
            helas_diagram = LoopHelasDiagram()

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
            # information with the interaction ID in the tag replaced by the 
            # corresponding vertex
            tag = copy.deepcopy(diagram.get('tag'))
            loop_vertices = copy.deepcopy(diagram.get('vertices'))
            for i in range(len(tag)):
                tag[i][2]=loop_vertices[i]
            
            # Single out last tag element, since this will give amplitude
            lastTagElem = tag.pop()
            
            # Copy the ct vertices of the loop
            ct_vertices = copy.copy(diagram.get('CT_vertices'))      

            # First create the starting external loop leg
            external_loop_wf=helas_objects.HelasWavefunction(\
                               tag[0][0], 0, model, decay_ids)
            wavefunctionNumber=wavefunctionNumber+1
            external_loop_wf.set('number',wavefunctionNumber)
            last_loop_wfs.append(external_loop_wf)
            diagram_wavefunctions.append(external_loop_wf)

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
                structIDs=tagElem[1]
                for last_loop_wf, color_list in zip(lastloopwfs,
                                                     colorlists):
                    loopLegOut = copy.copy(vertex.get('legs')[-1])
   
                    # From the incoming loop leg and the struct IDs, it generates
                    # a list of mothers, colorlists and number_to_wavefunctions
                    # dictionary for which each element correspond to one 
                    # lorentz-color structure of the tree-structure attached to
                    # the loop.
                    (motherslist, colorlists), wfNumber = \
                      getloopmothers(\
                            helas_objects.HelasWavefunctionList([last_loop_wf,]),
                            structIDs,\
                            color_list, diagram_wavefunctions, wfNumber)
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
                            wf.set('mothers',mothers)
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
                other_external_loop_wf=helas_objects.HelasWavefunction()
                wfNumber=wfNumber+1
                for leg in [leg for leg in lastvx['legs'] if leg['loop_line']]:
                    if last_loop_wfs[0]['number_external']!=leg['number']:
                        other_external_loop_wf=\
                          helas_objects.HelasWavefunction(leg, 0, model, decay_ids)
                        other_external_loop_wf.set('number',wfNumber)
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
                      helas_objects.HelasWavefunctionList(\
                                [other_external_loop_wf,last_loop_wf]), \
                      lastvx_structureIDs, color_list, \
                      diagram_wavefunctions, wfNumber)
                    
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
                                amp, loop_amp = done_color[color]
                                amp.get('coupling').append(inter.get('couplings')[coupl_key])
                                amp.get('lorentz').append(inter.get('lorentz')[coupl_key[1]])
                                # Update the coupling attribute of the loop_amp
                                loop_amp.set('coupling',loop_amp.get_couplings())
                                continue
                            amp = helas_objects.HelasAmplitude(lastvx, model)
                            amp.set('coupling', [inter.get('couplings')[coupl_key]])
                            amp.set('lorentz', [inter.get('lorentz')[coupl_key[1]]])
                            if inter.get('color'):
                                amp.set('inter_color', inter.get('color')[color])
                            amp.set('color_key', color) 
                            amp.set('mothers', mothers)
                            ###print "mothers added for amp="
                            ###for wf in mothers:
                            ###    print "    mother =",wf['number_external'],"("+str(wf.get_pdg_code())+") number=",wf['number']                             
                            # Add the list with color indices to the amplitude
                            new_color_list = copy.copy(color_list)+\
                                               copy.copy(structcolorlist)
                            new_color_list.append(color)
                            amp.set('color_indices', new_color_list)
                            # Add this amplitude to the LoopHelasAmplitude of this
                            # diagram.
                            amplitudeNumber = amplitudeNumber + 1
                            amp.set('number', amplitudeNumber)
                            amp.set('type','loop')
                            loop_amp = LoopHelasAmplitude()
                            loop_amp.set('amplitudes',\
                              helas_objects.HelasAmplitudeList([amp,]))
                            # Set the loop wavefunctions building this amplitude
                            # by tracking them from the last loop wavefunction
                            # added and its loop wavefunction among its mothers
                            loop_amp_wfs=helas_objects.HelasWavefunctionList(\
                              [last_loop_wf,])
                            while loop_amp_wfs[-1].get('mothers'):
                              loop_amp_wfs.append([lwf for lwf in \
                                loop_amp_wfs[-1].get('mothers') if lwf['is_loop']][0])
                            # Sort the loop wavefunctions of this amplitude
                            # according to their correct order of creation for 
                            # the HELAS calls (using their 'number' attribute
                            # would work as well, but I want something less naive)
                            # 1) Add the other L-cut particle at the end
                            loop_amp_wfs.append(other_external_loop_wf)
                            # 2) Reverse to have a consistent ordering of creation
                            # of helas wavefunctions.
                            loop_amp_wfs.reverse()
                            # Sort the loop wavefunctions of this amplitude
                            # according to their number.
                            #loop_amp_wfs.sort(lambda wf1, wf2: \
                            #  wf1.get('number') - wf2.get('number'))
                            loop_amp.set('wavefunctions',loop_amp_wfs)
                            loop_amp.set('type',diagram.get('type'))
                            loop_amp.set('number',min([amp.get('number') for amp
                                                       in loop_amp.get('amplitudes')]))
                            loop_amp.set('coupling',loop_amp.get_couplings())
                            helas_diagram.get('amplitudes').append(loop_amp)
                            # Save amp and loop_amp to the corresponding color
                            # in order to reuse them if the vertex has other
                            # lorentz structure linked to the same color one
                            done_color[color] = (amp, loop_amp)
                return wfNumber, amplitudeNumber
                
            def process_counterterms(ct_vertices, wfNumber, amplitudeNumber):
                """Process the counterterms vertices defined in this loop
                   diagram."""
                
                structIDs=[]
                for tagElem in tag:
                    structIDs += tagElem[1]
                structIDs += lastTagElem[1]
                # Here we call getloopmothers without any incoming loop
                # wavefunctions such that the function will return exactly
                # the mother of the counter-term amplitude we wish to create
                # We start with an empty color list as well in this case
                (motherslist, colorlists), wfNumber = getloopmothers(\
                                helas_objects.HelasWavefunctionList(), structIDs, \
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
                            amp = helas_objects.HelasAmplitude(ct_vertex, model)
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
                            amp_color_list = copy.copy(structcolorlist)
                            amp_color_list.append(color)     
                            amp.set('color_indices', amp_color_list)
                            amp.set('type',inter.get('type'))
                            
                            # Add amplitude to amplitdes in helas_diagram
                            helas_diagram.get('amplitudes').append(amp)
                return wfNumber, amplitudeNumber
            
            for tagElem in tag:
                wavefunctionNumber, last_loop_wfs, color_lists = \
                  process_tag_elem(tagElem, wavefunctionNumber, \
                                   last_loop_wfs, color_lists)
                  
            # Generate all amplitudes corresponding to the different
            # copies of this diagram
            wavefunctionNumber, amplitudeNumber = process_last_tag_Elem(lastTagElem, \
                                          wavefunctionNumber, amplitudeNumber)

            # Add now the counter-terms vertices
            if ct_vertices:
                wavefunctionNumber, amplitudeNumber = process_counterterms(\
                  ct_vertices, wavefunctionNumber, amplitudeNumber)

            # Identify among the diagram wavefunctions those from the structures
            # which will fill the 'wavefunctions' list of the diagram
            
            struct_wfs=helas_objects.HelasWavefunctionList(\
                      [wf for wf in diagram_wavefunctions if not wf['is_loop']])     

            # Sort the wavefunctions according to number
            struct_wfs.sort(lambda wf1, wf2: \
                          wf1.get('number') - wf2.get('number'))
                             
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
                wavefunctionNumber = len(process.get('legs'))

            # Return the diagram obtained
            return helas_diagram, wavefunctionNumber, amplitudeNumber      

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

        # We finally turn to the UVCT diagrams
        for diagram in amplitude.get('loop_UVCT_diagrams'):
            loopHelDiag, wf_number, amplitude_number=\
              process_born_diagram(diagram, wf_number, amplitude_number, \
                                   UVCTdiag=True)
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
            
        for loopdiag in self.get_loop_diagrams():
            for loopamp in loopdiag.get_loop_amplitudes():
                loopamp.set_mothers_and_pairing()

    def find_max_loop_coupling(self):
        """ Find the maximum number of loop couplings appearing in any of the
        LoopHelasAmplitude in this LoopHelasMatrixElement"""
        
        return max([len(amp.get('coupling')) for amp in \
            sum([d.get_loop_amplitudes() for d in self.get_loop_diagrams()],[])])

    def relabel_helas_objects(self):
        """After the generation of the helas objects, we can give up on having
        a unique number identifying the helas wavefunction and amplitudes and 
        instead use a labeling which is optimal for the output of the loop process.
        Also we tag all the LoopHelasAmplitude which are identical with the same
        'number' attribute."""

        # Give a unique number to each non-equivalent (at the level of the output)
        # LoopHelasAmplitude
        LoopHelasAmplitudeRecognized=[]
        for lamp in \
         sum([d.get_loop_amplitudes() for d in self.get_loop_diagrams()],[]):
            lamp.set('number',-1)
            for lamp2 in LoopHelasAmplitudeRecognized:
                if lamp.is_equivalent(lamp2):
                # The if statement below would be to turn the optimization off
                # if False:
                    lamp.set('number',lamp2.get('number'))
                    break;
            if lamp.get('number')==-1:
                    lamp.set('number',(len(LoopHelasAmplitudeRecognized)+1))
                    LoopHelasAmplitudeRecognized.append(lamp)

        # Start with the born diagrams
        wfnumber=1
        ampnumber=1
        for borndiag in self.get_born_diagrams():
            for wf in borndiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for amp in borndiag.get('amplitudes'):
                amp.set('number',ampnumber)
                ampnumber=ampnumber+1
        ampnumber=1
        # Now the loop ones
        for loopdiag in self.get_loop_diagrams():
            for wf in loopdiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for loopamp in loopdiag.get_loop_amplitudes():
                loopwfnumber=1
                for loopwf in loopamp['wavefunctions']:
                    loopwf.set('number',loopwfnumber)
                    loopwfnumber=loopwfnumber+1
                for amp in loopamp['amplitudes']:
                    amp.set('number',ampnumber)
                    ampnumber=ampnumber+1
            for ctamp in loopdiag.get_ct_amplitudes():
                    ctamp.set('number',ampnumber)
                    ampnumber=ampnumber+1
        # Finally the loopUVCT ones
        for loopUVCTdiag in self.get_loop_UVCT_diagrams():
            for wf in loopUVCTdiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for amp in loopUVCTdiag.get('amplitudes'):
                amp.set('number',ampnumber)
                ampnumber=ampnumber+1            
    
    def get_number_of_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME, including the
        loop ones"""

        return len(self.get_all_wavefunctions())
    
    def get_number_of_external_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME, excluding the
        loop ones."""
        
        return sum([ len(d.get('wavefunctions')) for d in \
                       self.get('diagrams')])

    def get_all_wavefunctions(self):
        """Gives a list of all wavefunctions for this ME"""

        allwfs=sum([d.get('wavefunctions') for d in self.get('diagrams')], [])
        for d in self['diagrams']:
            if isinstance(d,LoopHelasDiagram):
                for l in d.get_loop_amplitudes():
                    allwfs += l.get('wavefunctions')
                
        return allwfs

    def get_number_of_amplitudes(self):
        """Gives the total number of amplitudes for this ME, including the loop
        ones."""

        return len(self.get_all_amplitudes())

    def get_number_of_external_amplitudes(self):
        """Gives the total number of amplitudes for this ME, excluding those
        inside the loop amplitudes. (So only one is counted per loop amplitude.)
        """
        
        return sum([ len(d.get('amplitudes')) for d in \
                       self.get('diagrams')])

    def get_number_of_loop_amplitudes(self):
        """Gives the total number of helas amplitudes for the loop diagrams of this ME,
        excluding those inside the loop amplitudes, but including the CT-terms.
        (So only one amplitude is counted per loop amplitude.)
        """
        
        return sum([len(d.get('amplitudes')) for d in (self.get_loop_diagrams()+
                    self.get_loop_UVCT_diagrams())])

    def get_number_of_born_amplitudes(self):
        """Gives the total number of amplitudes for the born diagrams of this ME
        """
        
        return sum([len(d.get('amplitudes')) for d in self.get_born_diagrams()])

    def get_all_amplitudes(self):
        """Gives a list of all amplitudes for this ME"""

        allamps=sum([d.get_regular_amplitudes() for d in self.get('diagrams')], [])
        for d in self['diagrams']:
            if isinstance(d,LoopHelasDiagram):
                for l in d.get_loop_amplitudes():
                    allamps += l.get('amplitudes')
                
        return allamps

    def get_born_diagrams(self):
        """Gives a list of the born diagrams for this ME"""

        return helas_objects.HelasDiagramList([hd for hd in self['diagrams'] if\
                 not isinstance(hd,LoopHelasDiagram)])

    def get_loop_diagrams(self):
        """Gives a list of the loop diagrams for this ME"""

        return helas_objects.HelasDiagramList([hd for hd in self['diagrams'] if\
                 isinstance(hd,LoopHelasDiagram) and\
                 len(hd.get_loop_amplitudes())>=1])

    def get_loop_UVCT_diagrams(self):
        """Gives a list of the loop UVCT diagrams for this ME"""
        
        return helas_objects.HelasDiagramList([hd for hd in self['diagrams'] if\
                 isinstance(hd,LoopHelasDiagram) and\
                 len(hd.get_loop_UVCTamplitudes())>=1])

    def get_used_lorentz(self):
        """Return a list of (lorentz_name, tags, outgoing) with
        all lorentz structures used by this LoopHelasMatrixElement."""

        # Loop version of the function which add to the tuple wether it is a loop 
        # structure or not so that aloha knows if it has to produce the subroutine 
        # which removes the denominator in the propagator of the wavefunction created.
        output = []
        for wa in self.get_all_wavefunctions() + self.get_all_amplitudes():
            if wa.get('interaction_id') == 0:
                continue
            
            tags = ['C%s' % w for w in wa.get_conjugate_index()]
            if not ((isinstance(wa,helas_objects.HelasAmplitude) and \
                    wa.get('type')!='loop') or \
                  (isinstance(wa,helas_objects.HelasWavefunction) and \
                   not wa.get('is_loop'))): 
                tags.append('L')
            
            output.append((tuple(wa.get('lorentz')), tuple(tags), 
                                                     wa.find_outgoing_number()))
        return output

    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        answer = super(LoopHelasMatrixElement, self).get_used_couplings()
        for diag in self.get_loop_UVCT_diagrams():
            answer.extend([amp.get_used_UVCT_couplings() for amp in \
              diag.get_loop_UVCTamplitudes()])
        return answer

    def get_color_amplitudes(self):
        """ Just to forbid the usage of this generic function in a
        LoopHelasMatrixElement"""

        raise self.PhysicsObjectError, \
            "Usage of get_color_amplitudes is not allowed in a LoopHelasMatrixElement"

    def get_born_color_amplitudes(self):
        """Return a list of (coefficient, amplitude number) lists,
        corresponding to the JAMPs for this born color basis and the born
        diagrams of this LoopMatrixElement. The coefficients are given in the
        format (fermion factor, color coeff (frac), imaginary, Nc power)."""

        return super(LoopHelasMatrixElement,self).generate_color_amplitudes(\
            self['born_color_basis'],self.get_born_diagrams())

    def get_loop_color_amplitudes(self):
        """Return a list of (coefficient, amplitude number) lists,
        corresponding to the JAMPs for this loop color basis and the loop
        diagrams of this LoopMatrixElement. The coefficients are given in the
        format (fermion factor, color coeff (frac), imaginary, Nc power)."""

        diagrams=self.get_loop_diagrams()
        color_basis=self['loop_color_basis']
        
        if not color_basis:
            # No color, simply add all amplitudes with correct factor
            # for first color amplitude
            col_amp = []
            for diagram in diagrams:
                for amplitude in diagram.get('amplitudes'):
                    col_amp.append(((amplitude.get('fermionfactor'),
                                    1, False, 0),
                                    amplitude.get('number')))
            return [col_amp]

        # There is a color basis - create a list of coefficients and
        # amplitude numbers

        # Remember that with get_base_amplitude of LoopHelasMatrixElement,
        # we get several base_objects.Diagrams for a given LoopHelasDiagram:
        # One for the loop and one for each counter-term.
        # We should then here associate what are the HelasAmplitudes associated
        # to each diagram number using the function 
        # get_helas_amplitudes_loop_diagrams().
        LoopDiagramsHelasAmplitudeList=self.get_helas_amplitudes_loop_diagrams()
        # The HelasLoopAmplitudes should be unfolded to the HelasAmplitudes
        # (only one for the current version) they contain.
        for i, helas_amp_list in enumerate(LoopDiagramsHelasAmplitudeList):
            new_helas_amp_list=helas_objects.HelasAmplitudeList()
            for helas_amp in helas_amp_list:
                if isinstance(helas_amp,LoopHelasAmplitude):
                    new_helas_amp_list.extend(helas_amp['amplitudes'])
                else:
                    new_helas_amp_list.append(helas_amp)
            LoopDiagramsHelasAmplitudeList[i]=new_helas_amp_list

#        print "I get LoopDiagramsHelasAmplitudeList="
#        for i, elem in enumerate(LoopDiagramsHelasAmplitudeList):
#            print "LoopDiagramsHelasAmplitudeList[",i,"]=",[amp.get('number') for amp in LoopDiagramsHelasAmplitudeList[i]]

        col_amp_list = []
        for i, col_basis_elem in \
                enumerate(sorted(color_basis.keys())):

            col_amp = []
#            print "color_basis[col_basis_elem]=",color_basis[col_basis_elem]
            for diag_tuple in color_basis[col_basis_elem]:
                res_amps = filter(lambda amp: \
                          tuple(amp.get('color_indices')) == diag_tuple[1],
                          LoopDiagramsHelasAmplitudeList[diag_tuple[0]])
                if not res_amps:
                    raise self.PhysicsObjectError, \
                          """No amplitude found for color structure
                            %s and color index chain (%s) (diagram %i)""" % \
                            (col_basis_elem,
                             str(diag_tuple[1]),
                             diag_tuple[0])

                for res_amp in res_amps:
                    col_amp.append(((res_amp.get('fermionfactor'),
                                     diag_tuple[2],
                                     diag_tuple[3],
                                     diag_tuple[4]),
                                    res_amp.get('number')))

            col_amp_list.append(col_amp)

        return col_amp_list

    def get_helas_amplitudes_loop_diagrams(self):
        """ When creating the base_objects.Diagram in get_base_amplitudes(),
        each LoopHelasDiagram will lead to one loop_base_objects.LoopDiagram
        for its LoopHelasAmplitude and one other for each of its counter-term
        (with different interaction id). This function return a list for which
        each element is a HelasAmplitudeList corresponding to the HelasAmplitudes
        related to a given loop_base_objects.LoopDiagram generated """

        amplitudes_loop_diagrams=[]

        for diag in self.get_loop_diagrams():
            # We start by adding the loop topology
            amplitudes_loop_diagrams.append(diag.get_loop_amplitudes())
            # Then add a diagram for each counter-term with a different 
            # interactions id. (because it involves a different interaction
            # which possibly brings new color structures).
            # This is strictly speaking not necessary since Counter-Terms
            # cannot in principle bring new color structures into play. 
            # The dictionary ctIDs has the ct interactions ID as keys
            # and a HelasAmplitudeList of the corresponding HelasAmplitude as
            # values.
            ctIDs={}    
            for ctamp in diag.get_ct_amplitudes():
                try:
                    ctIDs[ctamp.get('interaction_id')].append(ctamp)
                except KeyError:
                    ctIDs[ctamp.get('interaction_id')]=\
                      helas_objects.HelasAmplitudeList([ctamp])
            # To have a canonical order of the CT diagrams, we sort them according
            # to their interaction_id value.
            keys=ctIDs.keys()
            keys.sort()
            for key in keys:
                amplitudes_loop_diagrams.append(ctIDs[key])
        
        for diag in self.get_loop_UVCT_diagrams():
            amplitudes_loop_diagrams.append(diag.get_loop_UVCTamplitudes())

        return amplitudes_loop_diagrams

    def get_base_amplitude(self):
        """Generate a loop_diagram_generation.LoopAmplitude from a
        LoopHelasMatrixElement. This is used to generate both color
        amplitudes and diagram drawing."""

        # Need to take care of diagram numbering for decay chains
        # before this can be used for those!

        optimization = 1
        if len(filter(lambda wf: wf.get('number') == 1,
                      self.get_all_wavefunctions())) > 1:
            optimization = 0

        model = self.get('processes')[0].get('model')

        wf_dict = {}
        vx_list = []
        diagrams = base_objects.DiagramList()

        # Start with the born
        for diag in self.get_born_diagrams():
            newdiag=diag.get('amplitudes')[0].get_base_diagram(\
                  wf_dict, vx_list, optimization)
            diagrams.append(loop_base_objects.LoopDiagram({
                  'vertices':newdiag['vertices'],'type':0}))
        
        # Store here the type of the last LoopDiagram encountered to reuse the
        # same value, but negative, for the corresponding counter-terms. 
        # It is not strictly necessary, it only has to be negative.
        type=1
        for HelasAmpList in self.get_helas_amplitudes_loop_diagrams():
            # We use uniformly the class LoopDiagram for the diagrams stored
            # in LoopAmplitude
            if isinstance(HelasAmpList[0],LoopHelasAmplitude):
                diagrams.append(HelasAmpList[0].get_base_diagram(\
                      wf_dict, vx_list, optimization))
                type=diagrams[-1]['type']
            elif isinstance(HelasAmpList[0],LoopHelasUVCTAmplitude):
                diagrams.append(HelasAmpList[0].\
                            get_base_diagram(wf_dict, vx_list, optimization))
            else:
                newdiag=HelasAmpList[0].get_base_diagram(wf_dict, vx_list, optimization)
                diagrams.append(loop_base_objects.LoopDiagram({
                  'vertices':newdiag['vertices'],'type':-type}))

        
        for diag in diagrams:
            diag.calculate_orders(self.get('processes')[0].get('model'))
            
        return loop_diagram_generation.LoopAmplitude({\
            'process': self.get('processes')[0],
            'diagrams': diagrams})

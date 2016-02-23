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

"""Definitions of objects inheriting from the classes defined in
helas_objects.py and which have special attributes and function 
devoted to the treatment of Loop processes"""

import array
import copy
import logging
import itertools
import math

import aloha
import aloha.create_aloha as create_aloha

from madgraph import MadGraph5Error
import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.loop.loop_color_amp as loop_color_amp
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.various.misc as misc

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
        
    def get_helas_call_dict(self, index=1, OptimizedOutput=False,\
                                                        specifyHel=True, **opt):
        """ return a dictionary to be used for formatting
        HELAS call. """
        
        
        out = helas_objects.HelasAmplitude.get_helas_call_dict(self, 
                index=index,OptimizedOutput=OptimizedOutput)
        out['uvct'] = self.get_UVCT_couplings()
        out.update(opt)
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
            super(LoopHelasAmplitude, self).__init__(*arguments)
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
                         'is_part','self_antipart','color']
        for arg in wfArgsToCheck:
            if [wf.get(arg) for wf in self.get('wavefunctions')]!=\
               [wf.get(arg) for wf in other.get('wavefunctions')]:
                return False

        if [wf.find_outgoing_number() for wf in self.get('wavefunctions')]!=\
           [wf.find_outgoing_number() for wf in other.get('wavefunctions')]:
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
        # a single amplitude, it was realized later that one would never need
        # more than one. But until now we kept the structure as such.
        self['amplitudes'] = helas_objects.HelasAmplitudeList()
        # The pairing is used for the output to know at each loop interactions
        # how many non-loop mothers are necessary. This list is ordered as the
        # helas calls building the loop
        self['pairing'] = []
        # To keep the 'type' (L-cut particle ID) of the LoopDiagram this
        # Loop amplitude tracks.
        # In principle this info is recoverable from the loop wfs.
        self['type'] = -1
        # The loop_group_id gives the place of this LoopHelasAmplitude 
        # in the 'loop_groups' attribute of the LoopHelasMatrixElement it belongs
        # to.
        self['loop_group_id']=-1
        # To store the symmetry factor of the loop
        self['loopsymmetryfactor'] = 0
        # Loop diagrams can be identified to others which are numerically exactly
        # equivalent. This is the case for example for the closed massless quark
        # loops. In this case, only one copy of the diagram is kept and this
        # multiplier attribute is set the to number of identified diagrams.
        # At the Helas level, this multiplier is given to each LoopHelasAmplitude
        self['multiplier'] = 1

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

        elif name in ['type','loop_group_id','multiplier','loopsymmetryfactor']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                  "%s is not a valid integer for the attribute '%s'" %(str(value),name)

        else:
            return super(LoopHelasAmplitude,self).filter(name, value)

        return True
    
    def get_sorted_keys(self):
        """Return LoopHelasAmplitude property names as a nicely sorted list."""

        return super(LoopHelasAmplitude,self).get_sorted_keys()+\
               ['wavefunctions', 'amplitudes','loop_group_id']

    def get_lcut_size(self):
        """ Return the wavefunction size (i.e. number of elements) based on the
        spin of the l-cut particle """
                
        return helas_objects.HelasWavefunction.spin_to_size(
                                 self.get_final_loop_wavefunction().get('spin'))

    def get_starting_loop_wavefunction(self):
        """ Return the starting external loop mother of this loop helas amplitude.
        It is the loop wavefunction of the l-cut leg one."""
        
        loop_wf=self.get_final_loop_wavefunction()
        loop_wf_mother=loop_wf.get_loop_mother()
        while loop_wf_mother:
            loop_wf=loop_wf_mother
            loop_wf_mother=loop_wf.get_loop_mother()
        return loop_wf

    def get_final_loop_wavefunction(self):
        """Return the non-external loop mother of the helas amplitude building 
        this loop amplitude"""
        
        final_lwf=[lwf for lwf in self.get('amplitudes')[0].get('mothers') if \
                   lwf.get('mothers')]
        if len(final_lwf)!=1:
            raise MadGraph5Error, 'The helas amplitude building the helas loop'+\
                ' amplitude should be made of exactly one loop wavefunctions'+\
                ' with mothers.'
        return final_lwf[0]

    def get_base_diagram(self, wf_dict, vx_list = [], optimization = 1):
        """Return the loop_base_objects.LoopDiagram which corresponds to this
        amplitude, using a recursive method for the wavefunctions.
        Remember that this diagram is not tagged and structures are not
        recognized."""

        vertices = self['amplitudes'][0].get_base_diagram(\
                     wf_dict, vx_list, optimization)['vertices']

        out = loop_base_objects.LoopDiagram({'vertices': vertices,\
                                              'type':self['type']})  
        
        # The generation of Helas diagram sometimes return that the two
        # loop external wavefunctions have the same external_id due to the 
        # recycling of the first external wavefunctions.
        # i. e. ((5(5*),1(21)>1(5*),id:160),(1(5*),2(21)>1(5*),id:160),(1(5*),3(37)>1(6*),id:21),(1(6*),4(-37)>1(5*),id:22),(5(-5*),1(5*),id:-1))
        # This only problematic when creating diagram with get_base_amplitude and
        # using them for the identifyME tagging
        
        starting_loop_line = out.get_starting_loop_line()
        finishing_loop_line = out.get_finishing_loop_line()
        if starting_loop_line['number'] == finishing_loop_line['number']:
            # This is the problematic case.
            # Since both particles have the same id, the routine get_external_legs
            # is always missing a particle. So we need to add one to have the correct
            # number of external particles (including the l-cut particle)
            nb_external = len(out.get_external_legs()) +1
            if nb_external == starting_loop_line['number']:
                starting_loop_line.set('number', nb_external -1)
            else:
                starting_loop_line.set('number', nb_external)

        
        return out 

    def set_mothers_and_pairing(self):
        """ Sets the mothers of this amplitude in the same order as they will
        be used in the arguments of the helas calls building this loop"""
        
        if len(self.get('amplitudes'))!=1:
            self.PhysicsObjectError, \
                  "HelasLoopAmplitude is for now designed to contain only one \
                   HelasAmplitude"
        
        self.set('mothers',helas_objects.HelasWavefunctionList())
        for lwf in [wf for wf in self.get('wavefunctions') if wf.get('mothers')]:
            mothersList=[wf for wf in lwf.get('mothers') if not wf['is_loop']]
            self['mothers'].extend(mothersList)
            self['pairing'].append(len(mothersList))

    def get_vertex_leg_numbers(self, 
              veto_inter_id=base_objects.Vertex.ID_to_veto_for_multichanneling,
              max_n_loop=0):
        """Get a list of the number of legs in vertices in this diagram"""

        if max_n_loop == 0:
            max_n_loop = base_objects.Vertex.max_n_loop_for_multichanneling

        # There is no need to check for self.get('interaction_id')==-2 when
        # applying the max_n_loop check because we already know that this
        # vertex is a loop one since it is a LoopHelasAmplitude
        vertex_leg_numbers = [len(self.get('mothers'))] if \
                         (self.get('interaction_id') not in veto_inter_id) or \
                                     len(self.get('mothers'))>max_n_loop else []
        for mother in self.get('mothers'):
            vertex_leg_numbers.extend(mother.get_vertex_leg_numbers(
                            veto_inter_id=veto_inter_id, max_n_loop=max_n_loop))

        return vertex_leg_numbers

    def get_denominators(self):
        """ Returns the denominator structure as a tuple (tupleA, tupleB) whose
        elements are of this form ((external_part_ids),mass) where
        external_part_ids are all the leg id building the momentum flowing in 
        the loop, i.e:
               D_i=(q+Sum(p_j,j))^2 - m^2
        """
        
        denoms=[]
        last_loop_wf=self.get_final_loop_wavefunction()
        last_loop_wf_mother=last_loop_wf.get_loop_mother()
        while last_loop_wf_mother:
            denoms.append((tuple(last_loop_wf.get_struct_external_leg_ids()),
                                                      last_loop_wf.get('mass')))
            last_loop_wf=last_loop_wf_mother
            last_loop_wf_mother=last_loop_wf.get_loop_mother()
        denoms.reverse()
        
        return tuple(denoms)

    def get_masses(self):
        """ Returns the list of the masses of the loop particles as they should
        appear for cuttools (L-cut particles specified last) """
        
        masses=[]
        if not aloha.complex_mass:
            for lwf in [wf for wf in self.get('wavefunctions') if wf.get('mothers')]:
                    masses.append(lwf.get('mass'))
        else:
            for lwf in [wf for wf in self.get('wavefunctions') if wf.get('mothers')]:
                if (lwf.get('width') == 'ZERO' or lwf.get('mass') == 'ZERO'):
                    masses.append(lwf.get('mass'))
                else: 
                    masses.append('CMASS_%s' % lwf.get('mass'))
        return masses

    def get_couplings(self):
        """ Returns the list of the couplings of the different helas objects
        building this HelasLoopAmplitude. They are ordered as they will appear
        in the helas calls."""

        return (sum([wf.get('coupling') for wf in self.get('wavefunctions') \
             if wf.get('coupling')!=['none']],[])\
             +sum([amp.get('coupling') for amp in self.get('amplitudes') if \
             amp.get('coupling')!=['none']],[]))

    def get_helas_call_dict(self, OptimizedOutput=False,specifyHel=True,**opt):
        """ return a dictionary to be used for formatting
        HELAS call. """
        output = {}
        output['numLoopLines']='_%d'%(len(self.get('wavefunctions'))-2)
        # Plus one below because fortran array start at 1.
        output['loop_group_id']=self.get('loop_group_id')+1
        output['ampNumber']=self.get('amplitudes')[0].get('number')
        if len(self.get('mothers'))!=len(self.get('pairing')):
            output['numMotherWfs']='_%d'%len(self.get('mothers'))
        else:
            output['numMotherWfs']=''            
        for i, pairing in enumerate(self.get('pairing')):
            output["Pairing%d"%i]=pairing
        output['numCouplings']='_%d'%len(self.get('coupling'))
        output['numeratorNumber']=self.get('number')
        output["LoopRank"]=self.get_analytic_info('wavefunction_rank')
        if OptimizedOutput:
            if self.get('loop_group_id')==-1:
                output['loopNumber']=self.get('number')
            else:
                output['loopNumber']=self.get('loop_group_id')+1               
        else:
            output['loopNumber']=self.get('amplitudes')[0].get('number')
        for i , wf in enumerate(self.get('mothers')):
            output["MotherID%d"%(i+1)]=wf.get('number')
        for i , mass in enumerate(self.get_masses()):
            output["LoopMass%d"%(i+1)]=mass
        for i , coupling in enumerate(self.get('coupling')):
            output["LoopCoupling%d"%(i+1)]=coupling
        output["LoopSymmetryFactor"] = self.get('loopsymmetryfactor')
        output["LoopMultiplier"] = self.get('multiplier')
        output.update(opt)

        return output

    def get_call_key(self):
        """ The helas call to a loop is simple and only depends on the number
        of loop lines and mothers. This how it is reflected in the call key. """
        
        return ("LOOP",len(self.get('wavefunctions'))-2,\
                len(self.get('mothers')),len(self.get('coupling')))

    def get_orders(self):
        """ Compute the orders building this loop amplitude only (not from the
        struct wavefunctions. Uses the cached result if available."""

        if self.get('orders') != {}:
            return self.get('orders')
        else:
            coupling_orders = {}
            last_wf = self.get_final_loop_wavefunction()
            while last_wf.get_loop_mother()!=None:
                for order in last_wf.get('orders').keys():
                    try:
                        coupling_orders[order] += last_wf.get('orders')[order]
                    except Exception:
                        coupling_orders[order] = last_wf.get('orders')[order]
                last_wf = last_wf.get_loop_mother()
            return coupling_orders
    
    def get_analytic_info(self, info, alohaModel=None):
        """ Returns an analytic information of the loop numerator, for example
        the 'wavefunction_rank' i.e. the maximum power to which the loop momentum 
        is elevated in the loop numerator. All analytic pieces of information 
        are for now identical to the one retrieved from the final_loop_wavefunction."""

        return self.get_final_loop_wavefunction().\
                                             get_analytic_info(info, alohaModel)

    def compute_analytic_information(self,alohaModel):
        """ Make sure that all analytic pieces of information about this 
        wavefunction are computed so that they can be recycled later, typically
        without the need of specifying an alohaModel. For now, all analytic
        information about the loop helas amplitude are identical to those of the
        final loop wavefunction."""
        
        self.get_final_loop_wavefunction().compute_analytic_information(\
                                                                     alohaModel)

    def calculate_fermionfactor(self):
        """ The fermion factor is not implemented for this object but in the
        subamplitude"""
        self['fermion_factor']=0
        for amp in self.get('amplitudes'):
            amp.get('fermionfactor')

    def calculate_loopsymmetryfactor(self):
        """ Calculate the loop symmetry factor. For one-loop matrix elements,
        it is always 2 for bubble with identical particles and 1 otherwise."""

        self['loopsymmetryfactor']=1
        
        # Make sure all particles are self-conjugated and identical in the loop
        if len(set([wf.get('pdg_code') for wf in self.get('wavefunctions')]))==1 and \
          not any([not wf.get('self_antipart') for wf in self.get('wavefunctions')]):
            # Now make sure we only include tadpoles or bubble
            if len(self.get('wavefunctions')) in [3,4]:
                self['loopsymmetryfactor']=2
        
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
        # To store the grouping of HelasLoopAmplitudes which share the same
        # denominators.
        # List of (key,value) where keys are tuples corresponding to the 
        # denominator structures (see get_denominators() of LoopHelasAmplitudes)
        # and values are lists of LoopHelasAmplitudes. It is not a dictionary
        # because we want for each LoopHelasAmplitude to assign a 'loop_group_id'
        # which indicates where it is placed in this list
        self['loop_groups'] = []
        
    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if name=='born_color_basis' or name=='loop_color_basis':
            if not isinstance(value,color_amp.ColorBasis):
                raise self.PhysicsObjectError, \
                  "%s is not a valid color basis" % str(value)
        elif name=='loop_groups':
            if not isinstance(value,list):
                raise self.PhysicsObjectError, \
                  "%s is not a valid list"%str(value)
            for (dkey, dvalue) in value:
                if not isinstance(dvalue,helas_objects.HelasAmplitudeList):
                    raise self.PhysicsObjectError, \
                      "%s is not a valid HelasAmplitudeList."%str(dvalue)
                if not isinstance(dkey,tuple):
                    raise self.PhysicsObjectError, \
                      "%s is not a valid tuple."%str(dkey)
        else:
            return super(LoopHelasMatrixElement,self).filter(name, value)

        return True

    def get(self,name):
        """Overload in order to return the loop_color_basis when simply asked
        for color_basis. The setter is not updated to avoid side effects."""
        
        if name=='color_basis':
            return self['loop_color_basis']
        elif name=='loop_groups':
            if not self['loop_groups']:
                self.identify_loop_groups()
            return self['loop_groups']
        else:
            return super(LoopHelasMatrixElement,self).get(name)
        
    def identify_loop_groups(self):
        """ Identify what are the loops sharing the same denominators and put
        them together in the 'loop_groups' attribute of this object. """
        
        identified_denom_structures=[]
        for lamp in [lamp for ldiag in self.get_loop_diagrams() for lamp in \
                     ldiag.get_loop_amplitudes()]:
            denom_structure=lamp.get_denominators()
            try:
                denom_index=identified_denom_structures.index(denom_structure)
                self['loop_groups'][denom_index][1].append(lamp)
            except ValueError:
                denom_index=len(self['loop_groups'])
                self['loop_groups'].append((denom_structure,
                                     helas_objects.HelasAmplitudeList([lamp,])))
                identified_denom_structures.append(denom_structure)
            lamp.set('loop_group_id',denom_index)
        # Now make sure that the loop amplitudes lists in values of the
        # dictionary are ordering in decreasing ranks, so that the first one
        # (later to be the reference amplitude) has the highest rank
        self['loop_groups']=[(group[0],helas_objects.HelasAmplitudeList(
            sorted(group[1],key=lambda lamp: \
            lamp.get_analytic_info('wavefunction_rank'),reverse=True)))
                                               for group in self['loop_groups']]
        # Also, order them so to put first the groups with the smallest
        # reference amplitude number
        self['loop_groups']=sorted(self['loop_groups'],key=lambda group: \
                                                      group[1][0].get('number'))
        self.update_loop_group_ids()

    def reuse_outdated_wavefunctions(self, helas_diagrams):
        """ Make sure never to use this optimization in the loop context."""
        # But just make sure that me_id is simply the number.
        for diag in helas_diagrams:
            for wf in diag['wavefunctions']:
                wf.set('me_id',wf.get('number'))
                
        return helas_diagrams

    def update_loop_group_ids(self):
        """ Make sure that the attribute 'loop_group_id' of all loop amplitudes
        in the 'loop_groups' list is correct given the order of 'loop_groups'"""
        
        for i, group in enumerate(self['loop_groups']):
            for lamp in group[1]:
                lamp.set('loop_group_id',i)

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
                'color_matrix','base_amplitude', 'has_mirror_process',
                'loop_groups']

    # Customized constructor
    def __init__(self, amplitude=None, optimization=1,
                 decay_ids=[], gen_color=True, optimized_output=False):
        """Constructor for the LoopHelasMatrixElement. For now, it works exactly
           as for the HelasMatrixElement one."""
        self.optimized_output=optimized_output
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

        # First make sure that the 'split_orders' are ordered according to their
        # weight.
        self.sort_split_orders(self.get('processes')[0].get('split_orders'))

        # Before starting, and if split_orders are defined in the amplitude
        # process, we must reorder the generated diagrams so as to put together
        # all those which share the same coupling orders. Then, we sort these
        # *group of diagrams* in decreasing WEIGHTED order, so that the
        # leading contributions are placed first (I will therfore be possible
        # to compute them only, saving the time of the rest of the computation)
        amplitude.order_diagrams_according_to_split_orders(\
                                   self.get('processes')[0].get('split_orders'))

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

        # To store the starting external loop wavefunctions needed
        # (They are never output so they are not in the diagrams wavefunctions)
        external_loop_wfs_dict={}

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

        # Initially, have one wavefunction for each external leg.
        wf_number = len(process.get('legs'))
        
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
                if self.optimized_output:
                    # Add one for the starting external loop wavefunctions
                    # which is fixed
                    wfNumber = wfNumber+1

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
            
            # Copy the ct vertices of the loop
            ct_vertices = copy.copy(diagram.get('CT_vertices'))      

            # First create the starting external loop leg
            external_loop_wf=helas_objects.HelasWavefunction(\
                                                tag[0][0], 0, model, decay_ids)

            # When on the optimized output mode, the starting loop wavefunction
            # can be recycled if it has the same pdg because whatever its pdg 
            # it has the same coefficients and loop momentum zero, 
            # so it is in principle not necessary to add it to the 
            # diagram_wavefunction. However, this is necessary for the function
            # check_and_fix_fermion_flow to correctly update the dependances of
            # previous diagrams to an external L-cut majorana wavefunction which
            # needs flipping.
            if not self.optimized_output:
                wavefunctionNumber=wavefunctionNumber+1
                external_loop_wf.set('number',wavefunctionNumber)
                diagram_wavefunctions.append(external_loop_wf)
            else:
                try:
                    external_loop_wf=\
                        external_loop_wfs_dict[external_loop_wf.get('pdg_code')]
                except KeyError:
                    wavefunctionNumber=wavefunctionNumber+1
                    external_loop_wf.set('number',wavefunctionNumber)
                    external_loop_wfs_dict[external_loop_wf.get('pdg_code')]=\
                                                                external_loop_wf
                    diagram_wavefunctions.append(external_loop_wf)

            # Setup the starting point of the reading of the loop flow.
            last_loop_wfs.append(external_loop_wf)

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
                                # Depending on wether we are on the 
                                # loop_optimized_output mode or now we want to
                                # reuse the loop wavefunctions as well.
                                try:
                                    if not self.optimized_output:
                                        raise ValueError
                                    # Use wf_mother_arrays to locate existing
                                    # wavefunction
                                    wf = wavefunctions[wf_mother_arrays.index(\
                                                                 wf.to_array())]
                                    # Since we reuse the old wavefunction, reset
                                    # wfNumber
                                    wfNumber = wfNumber - 1
                                    # To keep track of the number of loop 
                                    # wfs reused
                                    self.lwf_reused += 1
                                except ValueError:
                                    diagram_wavefunctions.append(wf)

                            # Update the last_loop_wfs list with the loop wf
                            # we just created. 
                            new_last_loop_wfs.append(wf)
                            # Add color index and store new copy of color_lists
                            new_color_list = copy.copy(structcolorlist)
                            new_color_list.append(coupl_key[0])
                            new_color_lists.append(new_color_list)
                
                # We update the lastloopwfs list and the color_lists for the
                # next iteration, i.e. the treatment of the next loop vertex
                # by returning them to the calling environnement.
                return wfNumber, new_last_loop_wfs, new_color_lists

 
            # Go through all vertices except the last and create
            # wavefunctions
            
            def create_amplitudes(lastvx, wfNumber, amplitudeNumber):
                """Treat the last tag element of the loop diagram (which 
                provides an amplitude)"""
                # First create the other external loop leg closing the loop.
                # It will not be in the final output, and in this sense, it is
                # a dummy wavefunction, but it is structurally important.
                # Because it is only structurally important, we do not need to
                # add it to the list of the wavefunctions for this ME or this
                # HELAS loop amplitude, nor do we need to update its number.
                other_external_loop_wf=helas_objects.HelasWavefunction()
#                wfNumber=wfNumber+1
                for leg in [leg for leg in lastvx['legs'] if leg['loop_line']]:
                    if last_loop_wfs[0]['number_external']!=leg['number']:
                        other_external_loop_wf=\
                          helas_objects.HelasWavefunction(leg, 0, model, decay_ids)
#                        other_external_loop_wf.set('number',wfNumber)
                        break
#               diagram_wavefunctions.append(other_external_loop_wf)
                
                for last_loop_wf, color_list in zip(last_loop_wfs,color_lists):
                    # Now generate HelasAmplitudes from the last vertex.
                    if lastvx.get('id')!=-1:
                        raise self.PhysicsObjectError, \
                          "The amplitude vertex of a loop diagram must be a "+\
                          "two point vertex with id=-1" 
                    # skip the boson and Dirac fermions
                    # adjust the fermion flow of external majorana loop wfs
                    if other_external_loop_wf.is_majorana():
                        fix_lcut_majorana_fermion_flow(last_loop_wf,\
                                                       other_external_loop_wf)
                    # fix the fermion flow
                    mothers=helas_objects.HelasWavefunctionList(\
                                [last_loop_wf,other_external_loop_wf])
                    wfNumber = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              None,
                                              wfNumber,
                                              False,
                                              []) # number_to_wavefunctions is useless in loop case
                    amp = helas_objects.HelasAmplitude(lastvx, model)
                    amp.set('interaction_id',-1)
                    amp.set('mothers',mothers)
                    #amp.set('mothers', helas_objects.HelasWavefunctionList(\
                    #            [last_loop_wf,other_external_loop_wf]))
                    amp.set('pdg_codes',[last_loop_wf.get_pdg_code(),
                                     other_external_loop_wf.get_pdg_code()])
                            ###print "mothers added for amp="
                            ###for wf in mothers:
                            ###    print "    mother =",wf['number_external'],"("+str(wf.get_pdg_code())+") number=",wf['number']                             
                            # Add the list with color indices to the amplitude

                    amp.set('color_indices', copy.copy(color_list))
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
                    loop_amp.set('wavefunctions',loop_amp_wfs)
                    loop_amp.set('type',diagram.get('type'))
                    loop_amp.set('multiplier',diagram.get('multiplier'))
                    # 'number' is not important as it will be redefined later.
                    loop_amp.set('number',min([amp.get('number') for amp
                                               in loop_amp.get('amplitudes')]))
                    loop_amp.set('coupling',loop_amp.get_couplings())
                    loop_amp.set('orders',loop_amp.get_orders())
                    helas_diagram.get('amplitudes').append(loop_amp)
                    # here we check the two L-cut loop helas wavefunctions are                                                                                                                           
                    # in consistent flow                     
                    check_lcut_fermion_flow_consistency(\
                        loop_amp_wfs[0],loop_amp_wfs[1])
                return wfNumber, amplitudeNumber
            
            def check_lcut_fermion_flow_consistency(lcut_wf1, lcut_wf2):
                """Checks that the two L-cut loop helas wavefunctions have                                                                                                                               
                a consistent fermion flow."""
                if lcut_wf1.is_boson():
                    if lcut_wf1.get('state')!='final' or\
                            lcut_wf2.get('state')!='final':
                        raise MadGraph5Error,\
                            "Inconsistent flow in L-cut bosons."
                elif not lcut_wf1.is_majorana():
                    for lcut_wf in [lcut_wf1,lcut_wf2]:
                        if not ((lcut_wf.get('is_part') and \
                                     lcut_wf.get('state')=='outgoing') or\
                                    (not lcut_wf.get('is_part') and\
                                         lcut_wf.get('state')=='incoming')):
                            raise MadGraph5Error,\
                                "Inconsistent flow in L-cut Dirac fermions."
                elif lcut_wf1.is_majorana():
                    if (lcut_wf1.get('state'), lcut_wf2.get('state')) not in \
                            [('incoming','outgoing'),('outgoing','incoming')]:
                        raise MadGraph5Error,\
                            "Inconsistent flow in L-cut Majorana fermions."
                            
            def fix_lcut_majorana_fermion_flow(last_loop_wf,\
                                               other_external_loop_wf):
                """Fix the fermion flow of the last external Majorana loop 
                wavefunction through the fermion flow of the first external 
                Majorana loop wavefunction."""
                # skip the boson and Dirac fermions
                # if not other_external_loop_wf.is_majorana():return
                loop_amp_wfs=helas_objects.HelasWavefunctionList(\
                                                                [last_loop_wf,])
                while loop_amp_wfs[-1].get('mothers'):
                    loop_amp_wfs.append([lwf for lwf in \
                    loop_amp_wfs[-1].get('mothers') if lwf['is_loop']][0])
                loop_amp_wfs.append(other_external_loop_wf)
                loop_amp_wfs.reverse()
                # loop_amp_wfs[0] is the last external loop wavefunction
                # while loop_amp_wfs[1] is the first external loop wavefunction
                rep={'incoming':'outgoing','outgoing':'incoming'}
                # Check if we need to flip the state of the external L-cut majorana
                other_external_loop_wf['state']=rep[loop_amp_wfs[1]['state']]
                return
                                                            
            def process_counterterms(ct_vertices, wfNumber, amplitudeNumber):
                """Process the counterterms vertices defined in this loop
                   diagram."""
                
                structIDs=[]
                for tagElem in tag:
                    structIDs += tagElem[1]
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
                        mothers = mothers.sort_by_pdg_codes(pdg_codes, 0)[0]
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
            wavefunctionNumber, amplitudeNumber = create_amplitudes(
                         loop_vertices[-1], wavefunctionNumber, amplitudeNumber)

            # Add now the counter-terms vertices
            if ct_vertices:
                wavefunctionNumber, amplitudeNumber = process_counterterms(\
                  ct_vertices, wavefunctionNumber, amplitudeNumber)

            # Identify among the diagram wavefunctions those from the structures
            # which will fill the 'wavefunctions' list of the diagram
            struct_wfs=helas_objects.HelasWavefunctionList(\
                      [wf for wf in diagram_wavefunctions if not wf['is_loop']])
            loop_wfs=helas_objects.HelasWavefunctionList(\
                      [wf for wf in diagram_wavefunctions if wf['is_loop']])     

            # Sort the wavefunctions according to number
            struct_wfs.sort(lambda wf1, wf2: \
                          wf1.get('number') - wf2.get('number'))
                             
            # After generation of all wavefunctions and amplitudes,
            # add wavefunctions to diagram
            helas_diagram.set('wavefunctions', struct_wfs)

            # Of course we only allow to reuse the struct wavefunctions but
            # never the loop ones which have to be present and reused in each
            # loop diagram, UNLESS we are in the loop_optimized_output mode.
            if optimization:
                wavefunctions.extend(struct_wfs)
                wf_mother_arrays.extend([wf.to_array() for wf in struct_wfs])
                if self.optimized_output:
                    wavefunctions.extend(loop_wfs)
                    wf_mother_arrays.extend([wf.to_array() for wf in loop_wfs])
            else:
                wavefunctionNumber = len(process.get('legs'))
                if self.optimized_output:
                    # Add one for the starting external loop wavefunctions
                    # which is fixed
                    wavefunctionNumber = wavefunctionNumber+1

            # And to the loop helas diagram if under the optimized output.
            # In the default output, one use those stored in the loop amplitude
            # since they are anyway not recycled. Notice that we remove the 
            # external L-cut loop wavefunctions from this list since they do 
            # not need to be computed.
            if self.optimized_output:
                loop_wfs = helas_objects.HelasWavefunctionList(
                         [lwf for lwf in loop_wfs if len(lwf.get('mothers'))>0])
                helas_diagram.set('loop_wavefunctions',loop_wfs)

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
        self.lwf_reused=0
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
            # We must add the UVCT_orders to the regular orders of the 
            # LooopHelasUVCTAmplitude
            for lamp in loopHelDiag.get_loop_UVCTamplitudes():
                new_orders = copy.copy(lamp.get('orders'))
                for order, value in lamp.get('UVCT_orders').items():
                    try:
                        new_orders[order] = new_orders[order] + value
                    except KeyError:
                        new_orders[order] = value
                lamp.set('orders', new_orders)
            helas_diagrams.append(loopHelDiag)
 
        self.set('diagrams', helas_diagrams)
         # Check wf order consistency
        if __debug__:
            for diag in self.get('diagrams'):
                # This is just a monitoring function, it will *NOT* affect the
                # wavefunctions list of the diagram, but just raise an Error
                # if the order is inconsistent, namely if a wavefunction in this
                # list has a mother which appears after its position in the list.
                diag.get('wavefunctions').check_wavefunction_numbers_order()

        # Inform how many loop wavefunctions have been reused.
        if self.optimized_output:
            logger.debug('%d loop wavefunctions have been reused'%self.lwf_reused+
            ', for a total of %d ones'%sum([len(ldiag.get('loop_wavefunctions'))
             for ldiag in self.get_loop_diagrams()]))
 
        # Sort all mothers according to the order wanted in Helas calls
        for wf in self.get_all_wavefunctions():
            wf.set('mothers', helas_objects.HelasMatrixElement.sorted_mothers(wf))

        for amp in self.get_all_amplitudes():
            amp.set('mothers', helas_objects.HelasMatrixElement.sorted_mothers(amp))
            # Not really necessary for the LoopHelasAmplitude as the color 
            # indices of the amplitudes should be correct. It is however 
            # cleaner like this. For debugging purposes we leave here an assert.
            gen_colors = amp.get('color_indices')
            amp.set('color_indices', amp.get_color_indices())
            if isinstance(amp,LoopHelasAmplitude):
                assert (amp.get('color_indices')==gen_colors), \
                  "Error in the treatment of color in the loop helas diagram "+\
                  "generation. It could be harmless, but report this bug to be sure."+\
                  " The different keys are %s vs %s."%(str(gen_colors),\
                                                      str(amp.get('color_indices')))
        for loopdiag in self.get_loop_diagrams():
            for loopamp in loopdiag.get_loop_amplitudes():
                loopamp.set_mothers_and_pairing()
 
        # As a final step, we compute the analytic information for the loop
        # wavefunctions and amplitudes building this loop matrix element.
        # Because we want to have the same AlohaModel used for various
        # HelasMatrix elements, we instead perform the call below in the
        # export which will use its AlohaModel for several HelasME's.
        # Hence we comment it here.
        # self.compute_all_analytic_information()
    
    def get_split_orders_mapping(self):
        """This function returns a list and a dictionary:
                        squared_orders, amps_orders
        ===
        The squared_orders lists all contributing squared_orders as tuple whose
        elements are the power at which are elevated the couplings orderered as
        in the 'split_orders'.
        
        squared_orders : All possible contributing squared orders among those
            specified in the process['split_orders'] argument. The elements of
            the list are tuples of the format
             ((OrderValue1,OrderValue2,...),
              (max_contrib_ct_amp_number,
              max_contrib_uvct_amp_number,
              max_contrib_loop_amp_number,
              max_contrib_group_id))
            with OrderValue<i> correspond to the value of the <i>th order in
            process['split_orders'] (the others are summed over and therefore 
            left unspecified).
            Ex for dijet with process['split_orders']=['QCD','QED']: 
                => [((4,0),(8,2,3)),((2,2),(10,3,3)),((0,4),(20,5,4))]
           
        'max_contrib_loop_amp_number': For optimization purposes, it is good to
        know what is the maximum loop amplitude number contributing to any given 
        squared order. The fortran output is structured so that if the user 
        is interested in a given squared order contribution only, then
        all the open loop coefficients for the amplitudes with a number above
        this value can be skipped.
        
        'max_contrib_(uv)ct_amp_number': Same as above but for the 
        (uv)ctamplitude number.
        
        'max_contrib_group_id': The same as above, except this time
        it is for the loop group id used for the loop reduction.
        ===
        The amps_orders is a *dictionary* with keys 
          'born_amp_orders',
          'loop_amp_orders'
        with values being the tuples described below.
        
        If process['split_orders'] is empty, all these tuples are set empty.
        
        'born_amp_orders' : Exactly as for squared order except that this list specifies
            the contributing order values for the amplitude (i.e. not 'squared').
            Also, the tuple describing the amplitude order is nested with a 
            second one listing all amplitude numbers contributing to this order.
            Ex for dijet with process['split_orders']=['QCD','QED']: 
                => [((2, 0), (2,)), ((0, 2), (1, 3, 4))]
            The function returns () if the process has no borns.
        
        'loop_amp_orders' : The same as for born_amp_orders but for the loop
            type of amplitudes only.

        Keep in mind that the orders of the elements of the outter most list is
        important as it dictates the order for the corresponding "order indices" 
        in the fortran code output by the exporters.
        """
    
        split_orders=self.get('processes')[0].get('split_orders')
        # If no split_orders are defined, then return the obvious
        amps_orders = {'born_amp_orders':[],
                       'loop_amp_orders':[]}
        if len(split_orders)==0:
            self.squared_orders = []
            return [],amps_orders
        
        # First make sure that the 'split_orders' are ordered according to their
        # weight.
        self.sort_split_orders(split_orders)
        
        process = self.get('processes')[0]
        # First make sure that the 'split_orders' are ordered according to their
        # weight.
        self.sort_split_orders(split_orders)
        loop_amp_orders = self.get_split_orders_mapping_for_diagram_list(\
             self.get_loop_diagrams(), split_orders, 
             get_amplitudes_function = lambda diag: diag.get_loop_amplitudes(),
             # We chose at this stage to store not only the amplitude numbers but
             # also the reference reduction id in the loop grouping, necessary
             # for returning the max_contrib_ref_amp_numbers.
             get_amp_number_function =  lambda amp: 
               (amp.get('amplitudes')[0].get('number'),amp.get('loop_group_id')))
        ct_amp_orders = self.get_split_orders_mapping_for_diagram_list(\
                self.get_loop_diagrams(), split_orders, 
                get_amplitudes_function = lambda diag: diag.get_ct_amplitudes())
        uvct_amp_orders = self.get_split_orders_mapping_for_diagram_list(\
                self.get_loop_UVCT_diagrams(), split_orders)
    
        # With this function, we just return the contributing amplitude numbers
        # The format is therefore the same as for the born_amp_orders and
        # ct_amp_orders
        amps_orders['loop_amp_orders'] = dict([(lao[0],
                          [el[0] for el in lao[1]]) for lao in loop_amp_orders])
        # Now add there the ct_amp_orders and uvct_amp_orders
        for ct_amp_order in ct_amp_orders+uvct_amp_orders:
            try:
                amps_orders['loop_amp_orders'][ct_amp_order[0]].extend(\
                                                          list(ct_amp_order[1]))
            except KeyError:
                amps_orders['loop_amp_orders'][ct_amp_order[0]] = \
                                                           list(ct_amp_order[1])
        # We must now turn it back to a list
        amps_orders['loop_amp_orders'] = [
            (key, tuple(sorted(amps_orders['loop_amp_orders'][key]))) 
                               for key in amps_orders['loop_amp_orders'].keys()]         
        # and re-sort it to make sure it follows an increasing WEIGHT order.
        order_hierarchy = self.get('processes')[0]\
                                            .get('model').get('order_hierarchy')
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            amps_orders['loop_amp_orders'].sort(key= lambda so: 
                         sum([order_hierarchy[split_orders[i]]*order_power for \
                                           i, order_power in enumerate(so[0])]))

        # Finally the born amp orders
        if process.get('has_born'):
            born_amp_orders = self.get_split_orders_mapping_for_diagram_list(\
                                          self.get_born_diagrams(),split_orders)
        
            amps_orders['born_amp_orders'] = born_amp_orders
        
        # Now we construct the interference splitting order matrix.
        # For this we flatten the list of many individual 2-tuples of the form
        # (amp_number, ref_amp_number) into one big 2-tuple of the form
        # (tuple_of_all_amp_numers, tuple_of_all_ref_amp_numbers).
        loop_orders = [(lso[0],tuple(zip(*list(lso[1])))) for lso in loop_amp_orders]

        # For the reference orders (against which the loop and ct amps are squared)
        # we only need the value of the orders, not the corresponding amp numbers.
        if process.get('has_born'):
            ref_orders = [bao[0] for bao in born_amp_orders]
        else:
            ref_orders = [lao[0] for lao in loop_orders+ct_amp_orders]
        
        # Temporarily we set squared_orders to be a dictionary with keys being
        # the actual contributing squared_orders and the values are the list 
        # [max_contrib_uvctamp_number,max_contrib_ct_amp_number,
        #  max_contrib_loop_amp_number,
        #  max_contrib_ref_amp_number]
        
        # In the event where they would be no contributing amplitude in one of 
        # the four class above, then the list on which the function max will be
        # called will be empty and we need to have the function not crash but
        # return -1 instead. 
        def smax(AmpNumList):
            return -1 if len(AmpNumList)==0 else max(AmpNumList)
            
        squared_orders = {}
        for ref_order in ref_orders:
            for uvct_order in uvct_amp_orders:
                key = tuple([ord1 + ord2 for ord1,ord2 in zip(uvct_order[0],
                                                                    ref_order)])
                try:
                    # Finding the max_contrib_uvct_amp_number
                    squared_orders[key][0] = smax([squared_orders[key][0]]+
                                                            list(uvct_order[1]))
                except KeyError:
                    squared_orders[key] = [smax(list(uvct_order[1])),-1,-1,-1]

            for ct_order in ct_amp_orders:
                key = tuple([ord1 + ord2 for ord1,ord2 in zip(ct_order[0],
                                                                    ref_order)])
                try:
                    # Finding the max_contrib_ct_amp_number
                    squared_orders[key][1] = smax([squared_orders[key][1]]+
                                                              list(ct_order[1]))
                except KeyError:
                    squared_orders[key] = [-1,smax(list(ct_order[1])),-1,-1]

            for loop_order in loop_orders:
                key = tuple([ord1 + ord2 for ord1,ord2 in zip(loop_order[0],
                                                                    ref_order)])
                try:
                    # Finding the max_contrib_loop_amp_number
                    squared_orders[key][2] = smax([squared_orders[key][2]]+
                                                         list(loop_order[1][0]))
                    # Finding the max_contrib_loop_id
                    squared_orders[key][3] = smax([squared_orders[key][3]]+
                                                         list(loop_order[1][1]))
                except KeyError:
                    squared_orders[key] = [-1,-1,smax(list(loop_order[1][0])),
                                                    smax(list(loop_order[1][1]))]

        # To sort the squared_orders, we now turn it into a list instead of a
        # dictionary. Each element of the list as the format
        #   ( squared_so_powers_tuple, 
        #     (max_uvct_amp_number, max_ct_amp_number,
        #      max_loop_amp_number, max_loop_id) )
        squared_orders = [(sqso[0],tuple(sqso[1])) for sqso in \
                                                         squared_orders.items()]
        # Sort the squared orders if the hierarchy defines them all.
        order_hierarchy = self.get('processes')[0].get('model').get('order_hierarchy')
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            squared_orders.sort(key= lambda so: 
                         sum([order_hierarchy[split_orders[i]]*order_power for \
                                           i, order_power in enumerate(so[0])]))

        # Cache the squared_orders information
        self.squared_orders = squared_orders

        return squared_orders, amps_orders

    def get_squared_order_contribs(self):
        """Return the squared_order contributions as returned by the function
        get_split_orders_mapping. It uses the cached value self.squared_orders
        if it was already defined during a previous call to get_split_orders_mapping.
        """

        if not hasattr(self, "squared_orders"):
            self.get_split_orders_mapping()

        return self.squared_orders

    def find_max_loop_coupling(self):
        """ Find the maximum number of loop couplings appearing in any of the
        LoopHelasAmplitude in this LoopHelasMatrixElement"""
        if len(self.get_loop_diagrams())==0:
            return 0
        return max([len(amp.get('coupling')) for amp in \
            sum([d.get_loop_amplitudes() for d in self.get_loop_diagrams()],[])])

    def get_max_loop_vertex_rank(self):
        """ Returns the maximum power of loop momentum brought by a loop
        interaction. For renormalizable theories, it should be no more than one.
        """
        return max([lwf.get_analytic_info('interaction_rank') for lwf in \
                                             self.get_all_loop_wavefunctions()])

    def get_max_loop_rank(self):
        """ Returns the rank of the contributing loop with maximum rank """
        r_list = [lamp.get_analytic_info('wavefunction_rank') for ldiag in \
               self.get_loop_diagrams() for lamp in ldiag.get_loop_amplitudes()]
        if len(r_list)==0:
            return 0
        else:
            return max(r_list)

    def get_max_spin_connected_to_loop(self):
        """Returns the maximum spin that any particle either connected to a loop
        or running in it has, among all the loops contributing to this ME"""

        # Remember that the loop wavefunctions running in the loop are stored in
        # the attribute 'loop_wavefunctions' of the HelasLoopDiagram in the 
        # optimized mode and in the 'wavefunction' attribute of the LoopHelasAmplitude
        # in the default mode.
        return max(
                     max(l.get('spin') for l in lamp.get('mothers')+
                          lamp.get('wavefunctions')+d.get('loop_wavefunctions'))
                     for d in self['diagrams'] if isinstance(d,LoopHelasDiagram) 
                                             for lamp in d.get_loop_amplitudes()
                  )

    def get_max_loop_particle_spin(self):
        """ Returns the spin of the loop particle with maximum spin among all
        the loop contributing to this ME"""
        return max([lwf.get('spin') for lwf in \
                                             self.get_all_loop_wavefunctions()])

    def relabel_loop_amplitudes(self):
        """Give a unique number to each non-equivalent (at the level of the output)
        LoopHelasAmplitude """
        
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

    def relabel_loop_amplitudes_optimized(self):
        """Give a unique number to each LoopHelasAmplitude. These will be the
        number used for the LOOPCOEF array in the optimized output and the
        grouping is done in a further stage by adding all the LOOPCOEF sharing
        the same denominator to a given one using the 'loop_group_id' attribute
        of the LoopHelasAmplitudes. """
        
        lamp_number=1
        for lamp in \
         sum([d.get_loop_amplitudes() for d in self.get_loop_diagrams()],[]):
            lamp.set('number',lamp_number)
            lamp_number += 1

    def relabel_loop_wfs_and_amps(self,wfnumber):
        """ Give the correct number for the default output to the wavefunctions
        and amplitudes building the loops """

        # We want first the CT amplitudes and only then the loop ones.
        CT_ampnumber=1
        loop_ampnumber=self.get_number_of_CT_amplitudes()+1
        loopwfnumber=1
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
                    amp.set('number',loop_ampnumber)
                    loop_ampnumber=loop_ampnumber+1
            for ctamp in loopdiag.get_ct_amplitudes():
                    ctamp.set('number',CT_ampnumber)
                    CT_ampnumber=CT_ampnumber+1
        # Finally the loopUVCT ones
        for loopUVCTdiag in self.get_loop_UVCT_diagrams():
            for wf in loopUVCTdiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for amp in loopUVCTdiag.get('amplitudes'):
                amp.set('number',CT_ampnumber)
                CT_ampnumber=CT_ampnumber+1 

    def relabel_loop_wfs_and_amps_optimized(self, wfnumber):
        """ Give the correct number for the optimized output to the wavefunctions
        and amplitudes building the loops """
        CT_ampnumber=1
        loop_ampnumber=self.get_number_of_CT_amplitudes()+1
        loopwfnumber=1
        # Now the loop ones
        for loopdiag in self.get_loop_diagrams():
            for wf in loopdiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for lwf in loopdiag.get('loop_wavefunctions'):
                lwf.set('number',loopwfnumber)
                loopwfnumber=loopwfnumber+1
            for loopamp in loopdiag.get_loop_amplitudes():
                # Set the number of the starting wavefunction (common to all 
                # diagrams) to one.
                loopamp.get_starting_loop_wavefunction().set('number',0)
                for amp in loopamp['amplitudes']:
                    amp.set('number',loop_ampnumber)
                    loop_ampnumber=loop_ampnumber+1
            for ctamp in loopdiag.get_ct_amplitudes():
                    ctamp.set('number',CT_ampnumber)
                    CT_ampnumber=CT_ampnumber+1
        # Finally the loopUVCT ones
        for loopUVCTdiag in self.get_loop_UVCT_diagrams():
            for wf in loopUVCTdiag.get('wavefunctions'):
                wf.set('number',wfnumber)
                wfnumber=wfnumber+1
            for amp in loopUVCTdiag.get('amplitudes'):
                amp.set('number',CT_ampnumber)
                CT_ampnumber=CT_ampnumber+1 

    def relabel_helas_objects(self):
        """After the generation of the helas objects, we can give up on having
        a unique number identifying the helas wavefunction and amplitudes and 
        instead use a labeling which is optimal for the output of the loop process.
        Also we tag all the LoopHelasAmplitude which are identical with the same
        'number' attribute."""
        
        # Number the LoopHelasAmplitude depending of the type of output
        if self.optimized_output:
            self.relabel_loop_amplitudes_optimized()
        else:
            self.relabel_loop_amplitudes()

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
        
        # Number the HelasWavefunctions and Amplitudes from the loops
        # depending of the type of output
        if self.optimized_output:
            self.relabel_loop_wfs_and_amps_optimized(wfnumber)
            for lwf in [lwf for loopdiag in self.get_loop_diagrams() for \
                                     lwf in loopdiag.get('loop_wavefunctions')]:
                lwf.set('me_id',lwf.get('number'))
        else:
            self.relabel_loop_wfs_and_amps(wfnumber)
            
        # Finally, for loops we do not reuse previously defined wavefunctions to
        # store new ones. So that 'me_id' is always equal to 'number'.
        for wf in self.get_all_wavefunctions():
            wf.set('me_id',wf.get('number'))
        

    def get_number_of_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME, including the
        loop ones"""

        return len(self.get_all_wavefunctions())
    
    def get_number_of_loop_wavefunctions(self):
        """ Gives the total number of loop wavefunctions for this ME."""
        return sum([len(ldiag.get('loop_wavefunctions')) for ldiag in \
                                                      self.get_loop_diagrams()])

    def get_number_of_external_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME, excluding the
        loop ones."""
        
        return sum([ len(d.get('wavefunctions')) for d in self.get('diagrams')])

    def get_all_wavefunctions(self):
        """Gives a list of all wavefunctions for this ME"""

        allwfs=sum([d.get('wavefunctions') for d in self.get('diagrams')], [])
        for d in self['diagrams']:
            if isinstance(d,LoopHelasDiagram):
                for l in d.get_loop_amplitudes():
                    allwfs += l.get('wavefunctions')
                
        return allwfs

    def get_all_loop_wavefunctions(self):
        """Gives a list of all the loop wavefunctions for this ME"""
                
        return helas_objects.HelasWavefunctionList(
                    # In the default output, this is where the loop wavefunction
                    # are placed
                    [lwf for ldiag in self.get_loop_diagrams()
                     for lamp in ldiag.get_loop_amplitudes()
                     for lwf in lamp.get('wavefunctions')]+
                    # In the optimized one they are directly in the 
                    # 'loop_wavefunctions' attribute of the loop diagrams
                    [lwf for ldiag in self.get_loop_diagrams() for lwf in
                     ldiag.get('loop_wavefunctions')])

    def get_nexternal_ninitial(self):
        """Gives (number or external particles, number of
        incoming particles)"""

        external_wfs = filter(lambda wf: 
                    not wf.get('mothers') and not wf.get('is_loop'),
                                                   self.get_all_wavefunctions())

        return (len(set([wf.get('number_external') for wf in \
                         external_wfs])),
                len(set([wf.get('number_external') for wf in \
                         filter(lambda wf: wf.get('leg_state') == False,
                                external_wfs)])))

    def get_number_of_amplitudes(self):
        """Gives the total number of amplitudes for this ME, including the loop
        ones."""

        return len(self.get_all_amplitudes())

    def get_number_of_CT_amplitudes(self):
        """Gives the total number of CT amplitudes for this ME. (i.e the amplitudes
        which are not LoopHelasAmplitudes nor within them.)"""

        return sum([len(d.get_ct_amplitudes()) for d in (self.get_loop_diagrams()+
                    self.get_loop_UVCT_diagrams())])

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

    def compute_all_analytic_information(self, alohaModel=None):
        """Make sure that all analytic pieces of information about all 
        loop wavefunctions and loop amplitudes building this loop helas matrix
        element are computed so that they can be recycled later, typically
        without the need of specifying an alohaModel.
        Notice that for now this function is called at the end of the 
        generat_helas_diagrams function and the alohaModel is created here.
        In principle, it might be better to have this function called by the
        exporter just after export_v4 because at this stage an alohaModel is
        already created and can be specified here instead of being generated.
        This can make a difference for very complicated models."""
        
        if alohaModel is None:
            # Generate it here
            model = self.get('processes')[0].get('model')
            myAlohaModel = create_aloha.AbstractALOHAModel(model.get('name'))
            myAlohaModel.add_Lorentz_object(model.get('lorentz'))
        else:
            # Use the one provided
            myAlohaModel = alohaModel

        for lwf in self.get_all_loop_wavefunctions():
            lwf.compute_analytic_information(myAlohaModel)
        
        for diag in self.get_loop_diagrams():
            for amp in diag.get_loop_amplitudes():
                amp.compute_analytic_information(myAlohaModel)

    def get_used_lorentz(self):
        """Return a list of (lorentz_name, tags, outgoing) with
        all lorentz structures used by this LoopHelasMatrixElement."""

        # Loop version of the function which add to the tuple wether it is a loop 
        # structure or not so that aloha knows if it has to produce the subroutine 
        # which removes the denominator in the propagator of the wavefunction created.
        output = []

        for wa in self.get_all_wavefunctions() + self.get_all_amplitudes():
            if wa.get('interaction_id') in [0,-1]:
                continue
            output.append(wa.get_aloha_info(self.optimized_output));

        return output

    def get_used_helas_loop_amps(self):
        """ Returns the list of the helas loop amplitude of type 
        CALL LOOP_I_J(_K)(...) used for this matrix element """
        
        # In the optimized output, we don't care about the number of couplings
        # in a given loop.
        if self.optimized_output:
            last_relevant_index=3
        else:
            last_relevant_index=4

        return list(set([lamp.get_call_key()[1:last_relevant_index] \
          for ldiag in self.get_loop_diagrams() for lamp in \
                                                  ldiag.get_loop_amplitudes()]))

    def get_used_wl_updates(self):
        """ Returns a list of the necessary updates of the loop wavefunction
        polynomials """
        
        return list(set([(lwf.get_analytic_info('wavefunction_rank')-\
                                    lwf.get_analytic_info('interaction_rank'), 
                                    lwf.get_analytic_info('interaction_rank')) 
                                for ldiag in self.get_loop_diagrams() 
                                for lwf in ldiag.get('loop_wavefunctions')]))
        
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
        dtype=1
        for HelasAmpList in self.get_helas_amplitudes_loop_diagrams():
            # We use uniformly the class LoopDiagram for the diagrams stored
            # in LoopAmplitude
            if isinstance(HelasAmpList[0],LoopHelasAmplitude):
                diagrams.append(HelasAmpList[0].get_base_diagram(\
                      wf_dict, vx_list, optimization))
                dtype=diagrams[-1]['type']
            elif isinstance(HelasAmpList[0],LoopHelasUVCTAmplitude):
                diagrams.append(HelasAmpList[0].\
                            get_base_diagram(wf_dict, vx_list, optimization))
            else:
                newdiag=HelasAmpList[0].get_base_diagram(wf_dict, vx_list, optimization)
                diagrams.append(loop_base_objects.LoopDiagram({
                  'vertices':newdiag['vertices'],'type':-dtype}))

        
        for diag in diagrams:
            diag.calculate_orders(self.get('processes')[0].get('model'))
            
        return loop_diagram_generation.LoopAmplitude({\
            'process': self.get('processes')[0],
            'diagrams': diagrams})

#===============================================================================
# LoopHelasProcess
#===============================================================================
class LoopHelasProcess(helas_objects.HelasMultiProcess):
    """LoopHelasProcess: Analogous of HelasMultiProcess except that it is suited
    for LoopAmplitude and with the peculiarity that it is always treating only 
    one loop amplitude. So this LoopHelasProcess correspond to only one single
    subprocess without multiparticle labels (contrary to HelasMultiProcess)."""
    
    # Type of HelasMatrixElement to be generated by this class of HelasMultiProcess
    matrix_element_class = LoopHelasMatrixElement
    
    def __init__(self, argument=None, combine_matrix_elements=True,
      optimized_output = True, compute_loop_nc = False, matrix_element_opts={}):
        """ Allow for the initialization of the HelasMultiProcess with the
        right argument 'optimized_output' for the helas_matrix_element options.
        """
        
        matrix_element_opts = dict(matrix_element_opts)
        matrix_element_opts.update({'optimized_output' : optimized_output})
        
        super(LoopHelasProcess, self).__init__(argument, combine_matrix_elements,
                        compute_loop_nc = compute_loop_nc, 
                                      matrix_element_opts = matrix_element_opts)

    @classmethod
    def process_color(cls,matrix_element,color_information,compute_loop_nc=False):
        """ Process the color information for a given matrix
        element made of a loop diagrams. It will create a different 
        color matrix depending on wether the process has a born or not.
        The compute_loop_nc sets wheter independent tracking of Nc power coming
        from the color loop trace is necessary or not (it is time consuming).
        """
        if matrix_element.get('processes')[0]['has_born']:
            logger.debug('Computing the loop and Born color basis')
        else:
            logger.debug('Computing the loop color basis')      
        
        # Define the objects stored in the contained color_information
        for key in color_information:
            exec("%s=color_information['%s']"%(key,key))

        # Now that the Helas Object generation is finished, we must relabel
        # the wavefunction and the amplitudes according to what should be
        # used for the output.
        matrix_element.relabel_helas_objects()

        # Always create an empty color basis, and the
        # list of raw colorize objects (before
        # simplification) associated with amplitude
        new_amp = matrix_element.get_base_amplitude()
        matrix_element.set('base_amplitude', new_amp)
        # Process the loop color basis which is needed anyway
        loop_col_basis = loop_color_amp.LoopColorBasis(
                                              compute_loop_nc = compute_loop_nc)
        loop_colorize_obj = loop_col_basis.create_loop_color_dict_list(\
                              matrix_element.get('base_amplitude'),
                              )
        try:
            # If the loop color configuration of the ME has
            # already been considered before, recycle
            # the information
            loop_col_basis_index = list_colorize.index(loop_colorize_obj)
            loop_col_basis = list_color_basis[loop_col_basis_index]
        except ValueError:
            # If not, create color basis accordingly
            list_colorize.append(loop_colorize_obj)
            loop_col_basis.build()
            loop_col_basis_index = len(list_color_basis)
            list_color_basis.append(loop_col_basis)
            logger.info(\
              "Processing color information for %s" % \
              matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                             replace('Process', 'loop process'))
        else: # Found identical color
            logger.info(\
              "Reusing existing color information for %s" % \
              matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                                 replace('Process', 'loop process'))
            
        if new_amp['process']['has_born']:
            born_col_basis = loop_color_amp.LoopColorBasis()
            born_colorize_obj = born_col_basis.create_born_color_dict_list(\
                             matrix_element.get('base_amplitude'))
            try:
                # If the loop color configuration of the ME has
                # already been considered before, recycle
                # the information
                born_col_basis_index = list_colorize.index(born_colorize_obj)
                born_col_basis = list_color_basis[born_col_basis_index]
            except ValueError:
                # If not, create color basis accordingly
                list_colorize.append(born_colorize_obj)
                born_col_basis.build()
                born_col_basis_index = len(list_color_basis)
                list_color_basis.append(born_col_basis)
                logger.info(\
                  "Processing color information for %s" % \
                  matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                             replace('Process', 'born process'))
            else: # Found identical color
                logger.info(\
                  "Reusing existing color information for %s" % \
                  matrix_element.get('processes')[0].nice_string(print_weighted=False).\
                                    replace('Process', 'born process'))                 
            loopborn_matrices_key=(loop_col_basis_index,born_col_basis_index)
        else:
            loopborn_matrices_key=(loop_col_basis_index,loop_col_basis_index)
            

        # Now we try to recycle the color matrix
        try:
            # If the color configuration of the ME has
            # already been considered before, recycle
            # the information
            col_matrix = dict_loopborn_matrices[loopborn_matrices_key]
        except KeyError:
            # If not, create color matrix accordingly
            col_matrix = color_amp.ColorMatrix(\
              list_color_basis[loopborn_matrices_key[0]],
              list_color_basis[loopborn_matrices_key[1]])
            dict_loopborn_matrices[loopborn_matrices_key]=col_matrix
            logger.info(\
              "Creating color matrix  %s" % \
              matrix_element.get('processes')[0].nice_string().\
                             replace('Process', 'loop process'))
        else: # Found identical color
            logger.info(\
              "Reusing existing color matrix for %s" % \
              matrix_element.get('processes')[0].nice_string().\
                                 replace('Process', 'loop process'))
            
        matrix_element.set('loop_color_basis',loop_col_basis)
        if new_amp['process']['has_born']:
            matrix_element.set('born_color_basis',born_col_basis)
        matrix_element.set('color_matrix',col_matrix)

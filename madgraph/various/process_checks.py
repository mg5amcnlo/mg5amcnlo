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
"""Several different checks for processes (and hence models):
permutation tests, gauge invariance tests, lorentz invariance
tests. Also class for evaluation of Python matrix elements,
MatrixElementEvaluator."""

from __future__ import division

import array
import copy
import fractions
import itertools
import logging
import math
import os
import sys
import re
import shutil
import random
import glob
import re
import subprocess
import time
import datetime
import errno
import pickle
# If psutil becomes standard, the RAM check can be performed with it instead
#import psutil

import aloha
import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.iolibs.file_writers as writers

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.rambo as rambo
import madgraph.various.misc as misc
import madgraph.various.progressbar as pbar
import madgraph.various.banner as bannermod
import madgraph.various.progressbar as pbar

import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import models.check_param_card as check_param_card

from madgraph.interface.madevent_interface import MadLoopInitializer
from madgraph.interface.common_run_interface import AskforEditCard
from madgraph import MG5DIR, InvalidCmd, MadGraph5Error

from madgraph.iolibs.files import cp

import StringIO
import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx, txxxxx, irxxxx, orxxxx

ADDED_GLOBAL = []

temp_dir_prefix = "TMP_CHECK"

pjoin = os.path.join

def clean_added_globals(to_clean):
    for value in list(to_clean):
        del globals()[value]
        to_clean.remove(value)

#===============================================================================
# Fake interface to be instancied when using process_checks from tests instead.
#===============================================================================
class FakeInterface(object):
    """ Just an 'option container' to mimick the interface which is passed to the
    tests. We put in only what is now used from interface by the test:
    cmd.options['fortran_compiler']
    cmd.options['complex_mass_scheme']
    cmd._mgme_dir"""
    def __init__(self, mgme_dir = "", complex_mass_scheme = False,
                 fortran_compiler = 'gfortran' ):
        self._mgme_dir = mgme_dir
        self.options = {}
        self.options['complex_mass_scheme']=complex_mass_scheme
        self.options['fortran_compiler']=fortran_compiler

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.process_checks')


# Helper function to boost momentum
def boost_momenta(p, boost_direction=1, beta=0.5):
    """boost the set momenta in the 'boost direction' by the 'beta' 
       factor"""
       
    boost_p = []    
    gamma = 1/ math.sqrt(1 - beta**2)
    for imp in p:
        bosst_p = imp[boost_direction]
        E, px, py, pz = imp
        boost_imp = []
        # Energy:
        boost_imp.append(gamma * E - gamma * beta * bosst_p)
        # PX
        if boost_direction == 1:
            boost_imp.append(-gamma * beta * E + gamma * px)
        else: 
            boost_imp.append(px)
        # PY
        if boost_direction == 2:
            boost_imp.append(-gamma * beta * E + gamma * py)
        else: 
            boost_imp.append(py)    
        # PZ
        if boost_direction == 3:
            boost_imp.append(-gamma * beta * E + gamma * pz)
        else: 
            boost_imp.append(pz) 
        #Add the momenta to the list
        boost_p.append(boost_imp)                   
            
    return boost_p

#===============================================================================
# Helper class MatrixElementEvaluator
#===============================================================================
class MatrixElementEvaluator(object):
    """Class taking care of matrix element evaluation, storing
    relevant quantities for speedup."""

    def __init__(self, model , param_card = None,
                    auth_skipping = False, reuse = True, cmd = FakeInterface()):
        """Initialize object with stored_quantities, helas_writer,
        model, etc.
        auth_skipping = True means that any identical matrix element will be
                        evaluated only once
        reuse = True means that the matrix element corresponding to a
                given process can be reused (turn off if you are using
                different models for the same process)"""
 
        self.cmd = cmd
 
        # Writer for the Python matrix elements
        self.helas_writer = helas_call_writers.PythonUFOHelasCallWriter(model)
    
        # Read a param_card and calculate couplings
        self.full_model = model_reader.ModelReader(model)
        try:
            self.full_model.set_parameters_and_couplings(param_card)
        except MadGraph5Error:
            if isinstance(param_card, (str,file)):
                raise
            logger.warning('param_card present in the event file not compatible.'+
                                                ' We will use the default one.')
            self.full_model.set_parameters_and_couplings()
            
        self.auth_skipping = auth_skipping
        self.reuse = reuse
        self.cmass_scheme = cmd.options['complex_mass_scheme']
        self.store_aloha = []
        self.stored_quantities = {}
        
    #===============================================================================
    # Helper function evaluate_matrix_element
    #===============================================================================
    def evaluate_matrix_element(self, matrix_element, p=None, full_model=None, 
                                gauge_check=False, auth_skipping=None, output='m2',
                                options=None):
        """Calculate the matrix element and evaluate it for a phase space point
           output is either m2, amp, jamp
        """

        if full_model:
            self.full_model = full_model
        process = matrix_element.get('processes')[0]
        model = process.get('model')

        if "matrix_elements" not in self.stored_quantities:
            self.stored_quantities['matrix_elements'] = []
            matrix_methods = {}

        if self.reuse and "Matrix_%s" % process.shell_string() in globals() and p:
            # Evaluate the matrix element for the momenta p
            matrix = eval("Matrix_%s()" % process.shell_string())
            me_value = matrix.smatrix(p, self.full_model)
            if output == "m2":
                return matrix.smatrix(p, self.full_model), matrix.amp2
            else:
                m2 = matrix.smatrix(p, self.full_model)
            return {'m2': m2, output:getattr(matrix, output)}
        if (auth_skipping or self.auth_skipping) and matrix_element in \
               self.stored_quantities['matrix_elements']:
            # Exactly the same matrix element has been tested
            logger.info("Skipping %s, " % process.nice_string() + \
                        "identical matrix element already tested" \
                        )
            return None

        self.stored_quantities['matrix_elements'].append(matrix_element)

        # Create an empty color basis, and the list of raw
        # colorize objects (before simplification) associated
        # with amplitude
        if "list_colorize" not in self.stored_quantities:
            self.stored_quantities["list_colorize"] = []
        if "list_color_basis" not in self.stored_quantities:
            self.stored_quantities["list_color_basis"] = []
        if "list_color_matrices" not in self.stored_quantities:
            self.stored_quantities["list_color_matrices"] = []        

        col_basis = color_amp.ColorBasis()
        new_amp = matrix_element.get_base_amplitude()
        matrix_element.set('base_amplitude', new_amp)
        colorize_obj = col_basis.create_color_dict_list(new_amp)

        try:
            # If the color configuration of the ME has
            # already been considered before, recycle
            # the information
            col_index = self.stored_quantities["list_colorize"].index(colorize_obj)
        except ValueError:
            # If not, create color basis and color
            # matrix accordingly
            self.stored_quantities['list_colorize'].append(colorize_obj)
            col_basis.build()
            self.stored_quantities['list_color_basis'].append(col_basis)
            col_matrix = color_amp.ColorMatrix(col_basis)
            self.stored_quantities['list_color_matrices'].append(col_matrix)
            col_index = -1

        # Set the color for the matrix element
        matrix_element.set('color_basis',
                           self.stored_quantities['list_color_basis'][col_index])
        matrix_element.set('color_matrix',
                           self.stored_quantities['list_color_matrices'][col_index])

        # Create the needed aloha routines
        if "used_lorentz" not in self.stored_quantities:
            self.stored_quantities["used_lorentz"] = []

        me_used_lorentz = set(matrix_element.get_used_lorentz())
        me_used_lorentz = [lorentz for lorentz in me_used_lorentz \
                               if lorentz not in self.store_aloha]

        aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
        aloha_model.add_Lorentz_object(model.get('lorentz'))
        aloha_model.compute_subset(me_used_lorentz)

        # Write out the routines in Python
        aloha_routines = []
        for routine in aloha_model.values():
            aloha_routines.append(routine.write(output_dir = None, 
                                                mode='mg5',
                                                language = 'Python'))
        for routine in aloha_model.external_routines:
            aloha_routines.append(
                     open(aloha_model.locate_external(routine, 'Python')).read())

        # Define the routines to be available globally
        previous_globals = list(globals().keys())
        for routine in aloha_routines:
            exec(routine, globals())
        for key in globals().keys():
            if key not in previous_globals:
                ADDED_GLOBAL.append(key)

        # Add the defined Aloha routines to used_lorentz
        self.store_aloha.extend(me_used_lorentz)
        # Export the matrix element to Python calls
        exporter = export_python.ProcessExporterPython(matrix_element,
                                                       self.helas_writer)
        try:
            matrix_methods = exporter.get_python_matrix_methods(\
                gauge_check=gauge_check)
#            print "I got matrix_methods=",str(matrix_methods.items()[0][1])
        except helas_call_writers.HelasWriterError, error:
            logger.info(error)
            return None
        # If one wants to output the python code generated for the computation
        # of these matrix elements, it is possible to run the following cmd
#       open('output_path','w').write(matrix_methods[process.shell_string()])
        if self.reuse:
            # Define the routines (globally)
            exec(matrix_methods[process.shell_string()], globals())	    
            ADDED_GLOBAL.append('Matrix_%s'  % process.shell_string())
        else:
            # Define the routines (locally is enough)
            exec(matrix_methods[process.shell_string()])
        # Generate phase space point to use
        if not p:
            p, w_rambo = self.get_momenta(process, options)
        # Evaluate the matrix element for the momenta p
        exec("data = Matrix_%s()" % process.shell_string())
        if output == "m2":
            return data.smatrix(p, self.full_model), data.amp2
        else:
            m2 = data.smatrix(p,self.full_model)
            return {'m2': m2, output:getattr(data, output)}
    
    @staticmethod
    def pass_isolation_cuts(pmoms, ptcut=50.0, drcut=0.5):
        """ Check whether the specified kinematic point passes isolation cuts
        """

        def Pt(pmom):
            """ Computes the pt of a 4-momentum"""
            return math.sqrt(pmom[1]**2+pmom[2]**2)

        def DeltaR(p1,p2):
            """ Computes the DeltaR between two 4-momenta"""
            # First compute pseudo-rapidities
            p1_vec=math.sqrt(p1[1]**2+p1[2]**2+p1[3]**2)
            p2_vec=math.sqrt(p2[1]**2+p2[2]**2+p2[3]**2)    
            eta1=0.5*math.log((p1_vec+p1[3])/(p1_vec-p1[3]))
            eta2=0.5*math.log((p2_vec+p2[3])/(p2_vec-p2[3]))
            # Then azimutal angle phi
            phi1=math.atan2(p1[2],p1[1])
            phi2=math.atan2(p2[2],p2[1])
            dphi=abs(phi2-phi1)
            # Take the wraparound factor into account
            dphi=abs(abs(dphi-math.pi)-math.pi)
            # Now return deltaR
            return math.sqrt(dphi**2+(eta2-eta1)**2)

        for i, pmom in enumerate(pmoms[2:]):
            # Pt > 50 GeV
            if Pt(pmom)<ptcut:
                return False
            # Delta_R ij > 0.5
            for pmom2 in pmoms[3+i:]:
                if DeltaR(pmom,pmom2)<drcut:
                    return False
        return True
    
    #===============================================================================
    # Helper function get_momenta
    #===============================================================================
    def get_momenta(self, process, options=None, special_mass=None):
        """Get a point in phase space for the external states in the given
        process, with the CM energy given. The incoming particles are
        assumed to be oriented along the z axis, with particle 1 along the
        positive z axis.
        For the CMS check, one must be able to chose the mass of the special
        resonance particle with id = -1, and the special_mass option allows
        to specify it."""

        if not options:
            energy=1000
            events=None
        else:
            energy = options['energy']
            events = options['events']
            to_skip = 0
            
        if not (isinstance(process, base_objects.Process) and \
                isinstance(energy, (float,int))):
            raise rambo.RAMBOError, "Not correct type for arguments to get_momenta"


        sorted_legs = sorted(process.get('legs'), lambda l1, l2:\
                                            l1.get('number') - l2.get('number'))

        # If an events file is given use it for getting the momentum
        if events:
            ids = [l.get('id') for l in sorted_legs]
            import MadSpin.decay as madspin
            if not hasattr(self, 'event_file'):
                fsock = open(events)
                self.event_file = madspin.Event(fsock)

            skip = 0
            while self.event_file.get_next_event() != 'no_event':
                event = self.event_file.particle
                #check if the event is compatible
                event_ids = [p['pid'] for p in event.values()]
                if event_ids == ids:
                    skip += 1
                    if skip > to_skip:
                        break
            else:
                raise MadGraph5Error, 'No compatible events for %s' % ids
            p = []
            for part in event.values():
                m = part['momentum']
                p.append([m.E, m.px, m.py, m.pz])
            return p, 1

        nincoming = len([leg for leg in sorted_legs if leg.get('state') == False])
        nfinal = len(sorted_legs) - nincoming

        # Find masses of particles
        mass = []
        for l in sorted_legs:
            if l.get('id') != 0:
                mass_string = self.full_model.get_particle(l.get('id')).get('mass')        
                mass.append(self.full_model.get('parameter_dict')[mass_string].real)
            else:
                if isinstance(special_mass, float):
                    mass.append(special_mass)
                else:
                    raise Exception, "A 'special_mass' option must be specified"+\
                 " in get_momenta when a leg with id=-10 is present (for CMS check)"
        #mass = [math.sqrt(m.real) for m in mass]



        # Make sure energy is large enough for incoming and outgoing particles,
#        # Keep the special_mass case separate to be sure that nothing interferes
#        # with the regular usage of get_momenta.
#        if not (any(l.get('id')==0 for l in sorted_legs) and \
#                                               isinstance(special_mass, float)):
        energy = max(energy, sum(mass[:nincoming])*1.2,sum(mass[nincoming:])*1.2)
#        else:
#            incoming_mass = sum([mass[i] for i, leg in enumerate(sorted_legs) \
#                             if leg.get('state') == False and leg.get('id')!=0])
#            outcoming_mass = sum([mass[i] for i, leg in enumerate(sorted_legs) \
#                             if leg.get('state') == True and leg.get('id')!=0])
#            energy = max(energy, incoming_mass*1.2, outcoming_mass*1.2)

        if nfinal == 1:
            p = []
            energy = mass[-1]
            p.append([energy/2,0,0,energy/2])
            p.append([energy/2,0,0,-energy/2])
            p.append([mass[-1],0,0,0])
            return p, 1.0

        e2 = energy**2
        m1 = mass[0]
        p = []

        masses = rambo.FortranList(nfinal)
        for i in range(nfinal):
            masses[i+1] = mass[nincoming + i]

        if nincoming == 1:
            # Momenta for the incoming particle
            p.append([abs(m1), 0., 0., 0.])
            p_rambo, w_rambo = rambo.RAMBO(nfinal, abs(m1), masses)
            # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
            for i in range(1, nfinal+1):
                momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                        p_rambo[(2,i)], p_rambo[(3,i)]]
                p.append(momi)

            return p, w_rambo

        if nincoming != 2:
            raise rambo.RAMBOError('Need 1 or 2 incoming particles')

        if nfinal == 1:
            energy = masses[1]
            if masses[1] == 0.0:
                raise rambo.RAMBOError('The kinematic 2 > 1 with the final'+\
                                          ' state particle massless is invalid')

        e2 = energy**2
        m2 = mass[1]

        mom = math.sqrt((e2**2 - 2*e2*m1**2 + m1**4 - 2*e2*m2**2 - \
                  2*m1**2*m2**2 + m2**4) / (4*e2))
        e1 = math.sqrt(mom**2+m1**2)
        e2 = math.sqrt(mom**2+m2**2)
        # Set momenta for incoming particles
        p.append([e1, 0., 0., mom])
        p.append([e2, 0., 0., -mom])

        if nfinal == 1:
            p.append([energy, 0., 0., 0.])
            return p, 1.

        p_rambo, w_rambo = rambo.RAMBO(nfinal, energy, masses)

        # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
        for i in range(1, nfinal+1):
            momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                    p_rambo[(2,i)], p_rambo[(3,i)]]
            p.append(momi)

        return p, w_rambo

#===============================================================================
# Helper class LoopMatrixElementEvaluator
#===============================================================================

class LoopMatrixElementEvaluator(MatrixElementEvaluator):
    """Class taking care of matrix element evaluation for loop processes."""

    def __init__(self,cuttools_dir=None, output_path=None, tir_dir={}, 
                                            cmd=FakeInterface(),*args,**kwargs):
        """Allow for initializing the MG5 root where the temporary fortran
        output for checks is placed."""
        
        super(LoopMatrixElementEvaluator,self).__init__(*args,cmd=cmd,**kwargs)

        self.mg_root=self.cmd._mgme_dir
        # If no specific output path is specified, then write in MG5 root directory
        if output_path is None:
            self.output_path = self.cmd._mgme_dir
        else:
            self.output_path = output_path
            
        self.cuttools_dir=cuttools_dir
        self.tir_dir=tir_dir
        self.loop_optimized_output = cmd.options['loop_optimized_output']
        # Set proliferate to true if you want to keep the produced directories
        # and eventually reuse them if possible
        self.proliferate=True
        
    #===============================================================================
    # Helper function evaluate_matrix_element for loops
    #===============================================================================
    def evaluate_matrix_element(self, matrix_element, p=None, options=None,
                             gauge_check=False, auth_skipping=None, output='m2', 
                                                  PS_name = None, MLOptions={}):
        """Calculate the matrix element and evaluate it for a phase space point
           Output can only be 'm2. The 'jamp' and 'amp' returned values are just
           empty lists at this point.
           If PS_name is not none the written out PS.input will be saved in 
           the file PS.input_<PS_name> as well."""

        process = matrix_element.get('processes')[0]
        model = process.get('model')
        
        if options and 'split_orders' in options.keys():
            split_orders = options['split_orders']
        else:
            split_orders = -1
        
        if "loop_matrix_elements" not in self.stored_quantities:
            self.stored_quantities['loop_matrix_elements'] = []

        if (auth_skipping or self.auth_skipping) and matrix_element in \
                [el[0] for el in self.stored_quantities['loop_matrix_elements']]:
            # Exactly the same matrix element has been tested
            logger.info("Skipping %s, " % process.nice_string() + \
                        "identical matrix element already tested" )
            return None

        # Generate phase space point to use
        if not p:
            p, w_rambo = self.get_momenta(process, options=options)
        
        if matrix_element in [el[0] for el in \
                                self.stored_quantities['loop_matrix_elements']]:  
            export_dir=self.stored_quantities['loop_matrix_elements'][\
                [el[0] for el in self.stored_quantities['loop_matrix_elements']\
                 ].index(matrix_element)][1]
            logger.debug("Reusing generated output %s"%str(export_dir))
        else:        
            export_dir=pjoin(self.output_path,temp_dir_prefix)
            if os.path.isdir(export_dir):
                if not self.proliferate:
                    raise InvalidCmd("The directory %s already exist. Please remove it."%str(export_dir))
                else:
                    id=1
                    while os.path.isdir(pjoin(self.output_path,\
                                        '%s_%i'%(temp_dir_prefix,id))):
                        id+=1
                    export_dir=pjoin(self.output_path,'%s_%i'%(temp_dir_prefix,id))
            
            if self.proliferate:
                self.stored_quantities['loop_matrix_elements'].append(\
                                                    (matrix_element,export_dir))

            # I do the import here because there is some cyclic import of export_v4
            # otherwise
            import madgraph.loop.loop_exporters as loop_exporters
            if self.loop_optimized_output:
                exporter_class=loop_exporters.LoopProcessOptimizedExporterFortranSA
            else:
                exporter_class=loop_exporters.LoopProcessExporterFortranSA
            
            MLoptions = {'clean': True, 
                       'complex_mass': self.cmass_scheme,
                       'export_format':'madloop', 
                       'mp':True,
                       'SubProc_prefix':'P',
                       'compute_color_flows': not process.get('has_born'),
              'loop_dir': pjoin(self.mg_root,'Template','loop_material'),
                       'cuttools_dir': self.cuttools_dir,
                       'fortran_compiler': self.cmd.options['fortran_compiler'],
                       'output_dependencies': self.cmd.options['output_dependencies']}

            MLoptions.update(self.tir_dir)
            
            FortranExporter = exporter_class(\
                self.mg_root, export_dir, MLoptions)
            FortranModel = helas_call_writers.FortranUFOHelasCallWriter(model)
            FortranExporter.copy_v4template(modelname=model.get('name'))
            FortranExporter.generate_subprocess_directory_v4(matrix_element, FortranModel)
            wanted_lorentz = list(set(matrix_element.get_used_lorentz()))
            wanted_couplings = list(set([c for l in matrix_element.get_used_couplings() \
                                                                    for c in l]))
            FortranExporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
            FortranExporter.finalize_v4_directory(None,"",False,False,compiler=
                       {'fortran':self.cmd.options['fortran_compiler'],
                        'f2py':self.cmd.options['fortran_compiler'],
                        'cpp':self.cmd.options['fortran_compiler']})

        MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                                                      split_orders=split_orders)

        self.fix_MadLoopParamCard(pjoin(export_dir,'Cards'),
           mp = gauge_check and self.loop_optimized_output, MLOptions=MLOptions)
        
        if gauge_check:
            file_path, orig_file_content, new_file_content = \
              self.setup_ward_check(pjoin(export_dir,'SubProcesses'), 
                                       ['helas_calls_ampb_1.f','loop_matrix.f'])
            file = open(file_path,'w')
            file.write(new_file_content)
            file.close()
            if self.loop_optimized_output:
                mp_file_path, mp_orig_file_content, mp_new_file_content = \
                  self.setup_ward_check(pjoin(export_dir,'SubProcesses'), 
                  ['mp_helas_calls_ampb_1.f','mp_compute_loop_coefs.f'],mp=True)
                mp_file = open(mp_file_path,'w')
                mp_file.write(mp_new_file_content)
                mp_file.close()
    
        # Evaluate the matrix element for the momenta p
        finite_m2 = self.get_me_value(process.shell_string_v4(), 0,\
                          export_dir, p, PS_name = PS_name, verbose=False)[0][0]

        # Restore the original loop_matrix.f code so that it could be reused
        if gauge_check:
            file = open(file_path,'w')
            file.write(orig_file_content)
            file.close()
            if self.loop_optimized_output:
                mp_file = open(mp_file_path,'w')
                mp_file.write(mp_orig_file_content)
                mp_file.close()
        
        # Now erase the output directory
        if not self.proliferate:
            shutil.rmtree(export_dir)
        
        if output == "m2": 
            # We do not provide details (i.e. amps and Jamps) of the computed 
            # amplitudes, hence the []
            return finite_m2, []
        else:
            return {'m2': finite_m2, output:[]}

    def fix_MadLoopParamCard(self,dir_name, mp=False, loop_filter=False,
                                 DoubleCheckHelicityFilter=False, MLOptions={}):
        """ Set parameters in MadLoopParams.dat suited for these checks.MP
            stands for multiple precision and can either be a bool or an integer
            to specify the mode."""

        # Instanciate a MadLoopParam card
        file = open(pjoin(dir_name,'MadLoopParams.dat'), 'r')
        MLCard = bannermod.MadLoopParam(file)

        if isinstance(mp,bool):
            mode = 4 if mp else 1
        else:
            mode = mp

        for key, value in MLOptions.items():
            if key == "MLReductionLib":
                if isinstance(value, int):
                    ml_reds = str(value)
                if isinstance(value,list):
                    if len(value)==0:
                        ml_reds = '1'
                    else:
                        ml_reds="|".join([str(vl) for vl in value])
                elif isinstance(value, str):
                    ml_reds = value
                elif isinstance(value, int):
                    ml_reds = str(value)
                else:
                    raise MadGraph5Error, 'The argument %s '%str(value)+\
                      ' in fix_MadLoopParamCard must be a string, integer'+\
                      ' or a list.'
                MLCard.set("MLReductionLib",ml_reds)      
            elif key == 'ImprovePS':
                MLCard.set('ImprovePSPoint',2 if value else -1)
            elif key == 'ForceMP':
                mode = 4
            elif key in MLCard:
                MLCard.set(key,value)
            else:
                raise Exception, 'The MadLoop options %s specified in function'%key+\
                  ' fix_MadLoopParamCard does not correspond to an option defined'+\
                  ' MadLoop nor is it specially handled in this function.'
        if not mode is None:
            MLCard.set('CTModeRun',mode)
            MLCard.set('CTModeInit',mode)
        MLCard.set('UseLoopFilter',loop_filter)
        MLCard.set('DoubleCheckHelicityFilter',DoubleCheckHelicityFilter)
        
        MLCard.write(pjoin(dir_name,os.pardir,'SubProcesses','MadLoopParams.dat'))

    @classmethod
    def get_me_value(cls, proc, proc_id, working_dir, PSpoint=[], PS_name = None,
                          verbose=True, format='tuple', skip_compilation=False):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id and PSpoint if specified.
        If PS_name is not none the written out PS.input will be saved in 
        the file PS.input_<PS_name> as well"""  
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
         
        shell_name = None
        directories = misc.glob('P%i_*' % proc_id, pjoin(working_dir, 'SubProcesses'))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" %proc)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        if verbose: logging.debug("Working on process %s in dir %s" % (proc, shell_name))
        
        dir_name = pjoin(working_dir, 'SubProcesses', shell_name)
        if not skip_compilation:
            # Make sure to recreate the executable and modified sources
            if os.path.isfile(pjoin(dir_name,'check')):
                os.remove(pjoin(dir_name,'check'))
                try:
                    os.remove(pjoin(dir_name,'check_sa.o'))
                    os.remove(pjoin(dir_name,'loop_matrix.o'))
                except OSError:
                    pass
            # Now run make
            devnull = open(os.devnull, 'w')
            retcode = subprocess.call(['make','check'],
                                       cwd=dir_name, stdout=devnull, stderr=devnull)
            devnull.close()
                         
            if retcode != 0:
                logging.info("Error while executing make in %s" % shell_name)
                return ((0.0, 0.0, 0.0, 0.0, 0), [])

        # If a PS point is specified, write out the corresponding PS.input
        if PSpoint:
            misc.write_PS_input(pjoin(dir_name, 'PS.input'),PSpoint)
            # Also save the PS point used in PS.input_<PS_name> if the user
            # wanted so. It is used for the lorentz check. 
            if not PS_name is None:
                misc.write_PS_input(pjoin(dir_name, \
                                                 'PS.input_%s'%PS_name),PSpoint)        
        # Run ./check
        try:
            output = subprocess.Popen('./check',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(pjoin(dir_name,'result.dat')):
                return cls.parse_check_output(file(pjoin(dir_name,\
                                                   'result.dat')),format=format)  
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                           .join(dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

    @classmethod
    def parse_check_output(cls,output,format='tuple'):
        """Parse the output string and return a pair where first four values are 
        the finite, born, single and double pole of the ME and the fourth is the
        GeV exponent and the second value is a list of 4 momenta for all particles 
        involved. Return the answer in two possible formats, 'tuple' or 'dict'."""

        res_dict = {'res_p':[],
                    'born':0.0,
                    'finite':0.0,
                    '1eps':0.0,
                    '2eps':0.0,
                    'gev_pow':0,
                    'export_format':'Default',
                    'accuracy':0.0,
                    'return_code':0,
                    'Split_Orders_Names':[],
                    'Loop_SO_Results':[],
                    'Born_SO_Results':[],
                    'Born_kept':[],
                    'Loop_kept':[]
                    }
        res_p = []
        
        # output is supposed to be a file, if it is its content directly then
        # I change it to be the list of line.
        if isinstance(output,file) or isinstance(output,list):
            text=output
        elif isinstance(output,str):
            text=output.split('\n')
        else:
            raise MadGraph5Error, 'Type for argument output not supported in'+\
                                                          ' parse_check_output.'
        for line in text:
            splitline=line.split()
            if len(splitline)==0:
                continue
            elif splitline[0]=='PS':
                res_p.append([float(s) for s in splitline[1:]])
            elif splitline[0]=='ASO2PI':
                res_dict['alphaS_over_2pi']=float(splitline[1])
            elif splitline[0]=='BORN':
                res_dict['born']=float(splitline[1])
            elif splitline[0]=='FIN':
                res_dict['finite']=float(splitline[1])
            elif splitline[0]=='1EPS':
                res_dict['1eps']=float(splitline[1])
            elif splitline[0]=='2EPS':
                res_dict['2eps']=float(splitline[1])
            elif splitline[0]=='EXP':
                res_dict['gev_pow']=int(splitline[1])
            elif splitline[0]=='Export_Format':
                res_dict['export_format']=splitline[1]
            elif splitline[0]=='ACC':
                res_dict['accuracy']=float(splitline[1])
            elif splitline[0]=='RETCODE':
                res_dict['return_code']=int(splitline[1])
            elif splitline[0]=='Split_Orders_Names':
                res_dict['Split_Orders_Names']=splitline[1:]
            elif splitline[0] in ['Born_kept', 'Loop_kept']:
                res_dict[splitline[0]] = [kept=='T' for kept in splitline[1:]]
            elif splitline[0] in ['Loop_SO_Results', 'Born_SO_Results']:
                # The value for this key of this dictionary is a list of elements
                # with format ([],{}) where the first list specifies the split
                # orders to which the dictionary in the second position corresponds 
                # to.
                res_dict[splitline[0]].append(\
                                         ([int(el) for el in splitline[1:]],{}))
            elif splitline[0]=='SO_Loop':
                res_dict['Loop_SO_Results'][-1][1][splitline[1]]=\
                                                             float(splitline[2])
            elif splitline[0]=='SO_Born':
                res_dict['Born_SO_Results'][-1][1][splitline[1]]=\
                                                             float(splitline[2])
        
        res_dict['res_p'] = res_p

        if format=='tuple':
            return ((res_dict['finite'],res_dict['born'],res_dict['1eps'],
                       res_dict['2eps'],res_dict['gev_pow']), res_dict['res_p'])
        else:
            return res_dict
    
    @staticmethod
    def apply_log_tweak(proc_path, mode):
        """ Changes the file model_functions.f in the SOURCE of the process output
        so as to change how logarithms are analytically continued and see how
        it impacts the CMS check."""
        valid_modes = ['default','recompile']
        if not (mode in valid_modes or (isinstance(mode, list) and
                len(mode)==2 and all(m in ['logp','logm','log'] for m in mode))):
            raise MadGraph5Error("Mode '%s' not reckonized"%mode+
                                                " in function apply_log_tweak.")
        
        model_path = pjoin(proc_path,'Source','MODEL')
        directories = misc.glob('P0_*', pjoin(proc_path,'SubProcesses'))
        if directories and os.path.isdir(directories[0]):
            exe_path = directories[0]
        else:
            raise MadGraph5Error, 'Could not find a process executable '+\
                                                      'directory in %s'%proc_dir
        bu_path = pjoin(model_path, 'model_functions.f__backUp__')
        
        if mode=='default':
            # Restore the default source file model_function.f
            if not os.path.isfile(bu_path):
                raise MadGraph5Error, 'Back up file %s could not be found.'%bu_path
            shutil.move(bu_path, pjoin(model_path, 'model_functions.f'))
            return

        if mode=='recompile':
            try:
                os.remove(pjoin(model_path,'model_functions.o'))
                os.remove(pjoin(proc_path,'lib','libmodel.a'))
            except:
                pass    
            misc.compile(cwd=model_path)
            # Remove the executable to insure proper recompilation
            try:
                os.remove(pjoin(exe_path,'check'))
            except:
                pass
            misc.compile(arg=['check'], cwd=exe_path)
            return
        
        if mode[0]==mode[1]:
            return
        
        # Now change the logs
        mp_prefix = 'MP_'
        target_line = 'FUNCTION %%sREG%s(ARG)'%mode[0].lower()

        # Make sure to create a backup
        if not os.path.isfile(bu_path):
            shutil.copy(pjoin(model_path, 'model_functions.f'), bu_path)            
        model_functions = open(pjoin(model_path,'model_functions.f'),'r')
        
        new_model_functions = []
        has_replaced        = False
        just_replaced       = False
        find_one_replacement= False
        mp_mode             = None
        suffix = {'log':'','logp':r'\s*\+\s*TWOPII','logm':r'\s*\-\s*TWOPII'}
        replace_regex=r'^\s*%%sREG%s\s*=\s*LOG\(ARG\)%s'%(mode[0],suffix[mode[0]])
        for line in model_functions:
            # Make sure to skip split lines after the replacement
            if just_replaced:
                if not re.match(r'\s{6}', line):
                    continue
                else:
                    just_replaced = False
            if mp_mode is None:
                # We are looking for the start of the function
                new_model_functions.append(line)
                if (target_line%mp_prefix).lower() in line.lower():
                    mp_mode       = mp_prefix
                elif (target_line%'').lower() in line.lower():
                    mp_mode       = ''
            else:
                # Now apply the substitution
                if not has_replaced and re.match(replace_regex%mp_mode,line,
                                                                 re.IGNORECASE):
                    # Apply the replacement
                    if mode[0]=='log':
                        if mp_mode=='':
                            new_line =\
"""      if(dble(arg).lt.0.0d0.and.dimag(arg).gt.0.0d0)then
        reg%s=log(arg) %s TWOPII
      else
        reg%s=log(arg)
      endif\n"""%(mode[0],'+' if mode[1]=='logp' else '-',mode[0])
                        else:
                            new_line =\
"""      if(real(arg,kind=16).lt.0.0e0_16.and.imagpart(arg).lt.0.0e0_16)then
        mp_reg%s=log(arg) %s TWOPII
      else
        mp_reg%s=log(arg)
      endif\n"""%(mode[0],'+' if mode[1]=='logp' else '-',mode[0])
                    else:
                        new_line = ' '*6+"%sreg%s=log(arg) %s\n"%(mp_mode,mode[0],
      ('' if mode[1]=='log' else ('+TWOPII' if mode[1]=='logp' else '-TWOPII')))
                    new_model_functions.append(new_line)
                    just_replaced = True
                    has_replaced  = True
                    find_one_replacement = True
                else:
                    new_model_functions.append(line)
                    if re.match(r'^\s*END\s*$',line,re.IGNORECASE):
                        mp_mode      = None
                        has_replaced = False
        
        if not find_one_replacement:
            logger.warning('No replacement was found/performed for token '+
                                                  "'%s->%s'."%(mode[0],mode[1]))
        else:
            open(pjoin(model_path,'model_functions.f'),'w').\
                                             write(''.join(new_model_functions))
        return          
                
    def setup_ward_check(self, working_dir, file_names, mp = False):
        """ Modify loop_matrix.f so to have one external massless gauge boson
        polarization vector turned into its momentum. It is not a pretty and 
        flexible solution but it works for this particular case."""
        
        shell_name = None
        directories = misc.glob('P0_*', working_dir)
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])
        
        dir_name = pjoin(working_dir, shell_name)
        
        # Look, in order, for all the possible file names provided.
        ind=0
        while ind<len(file_names) and not os.path.isfile(pjoin(dir_name,
                                                              file_names[ind])):
            ind += 1
        if ind==len(file_names):
            raise Exception, "No helas calls output file found."
        
        helas_file_name=pjoin(dir_name,file_names[ind])
        file = open(pjoin(dir_name,helas_file_name), 'r')
        
        helas_calls_out=""
        original_file=""
        gaugeVectorRegExp=re.compile(\
         r"CALL (MP\_)?VXXXXX\(P\(0,(?P<p_id>\d+)\),((D)?CMPLX\()?ZERO((,KIND\=16)?\))?,"+
         r"NHEL\(\d+\),[\+\-]1\*IC\(\d+\),W\(1,(?P<wf_id>\d+(,H)?)\)\)")
        foundGauge=False
        # Now we modify the first massless gauge vector wavefunction
        for line in file:
            helas_calls_out+=line
            original_file+=line
            if line.find("INCLUDE 'coupl.inc'") != -1 or \
                             line.find("INCLUDE 'mp_coupl_same_name.inc'") !=-1:
                helas_calls_out+="      INTEGER WARDINT\n"
            if not foundGauge:
                res=gaugeVectorRegExp.search(line)
                if res!=None:
                    foundGauge=True
                    helas_calls_out+="      DO WARDINT=1,4\n"
                    helas_calls_out+="        W(WARDINT+4,"+res.group('wf_id')+")="
                    if not mp:
                        helas_calls_out+=\
                            "DCMPLX(P(WARDINT-1,"+res.group('p_id')+"),0.0D0)\n"
                    else:
                        helas_calls_out+="CMPLX(P(WARDINT-1,"+\
                                       res.group('p_id')+"),0.0E0_16,KIND=16)\n"
                    helas_calls_out+="      ENDDO\n"
        file.close()
        
        return pjoin(dir_name,helas_file_name), original_file, helas_calls_out

#===============================================================================
# Helper class LoopMatrixElementEvaluator
#===============================================================================
class LoopMatrixElementTimer(LoopMatrixElementEvaluator):
    """Class taking care of matrix element evaluation and running timing for 
       loop processes."""

    def __init__(self, *args, **kwargs):
        """ Same as the mother for now """
        LoopMatrixElementEvaluator.__init__(self,*args, **kwargs)
    
    @classmethod
    def get_MadLoop_Params(cls,MLCardPath):
        """ Return a dictionary of the parameter of the MadLoopParamCard.
        The key is the name of the parameter and the value is the corresponding
        string read from the card."""
        
        return bannermod.MadLoopParam(MLCardPath)


    @classmethod
    def set_MadLoop_Params(cls,MLCardPath,params):
        """ Set the parameters in MadLoopParamCard to the values specified in
        the dictionary params.
        The key is the name of the parameter and the value is the corresponding
        string to write in the card."""
        
        MLcard = bannermod.MadLoopParam(MLCardPath)
        for key,value in params.items():
            MLcard.set(key, value, ifnotdefault=False)

        MLcard.write(MLCardPath, commentdefault=True)

    def skip_loop_evaluation_setup(self, dir_name, skip=True):
        """ Edit loop_matrix.f in order to skip the loop evaluation phase.
        Notice this only affects the double precision evaluation which is
        normally fine as we do not make the timing check on mp."""

        file = open(pjoin(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(pjoin(dir_name,'loop_matrix.f'), 'w')
        loop_matrix = re.sub(r"SKIPLOOPEVAL=\S+\)","SKIPLOOPEVAL=%s)"%('.TRUE.' 
                                           if skip else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def boot_time_setup(self, dir_name, bootandstop=True):
        """ Edit loop_matrix.f in order to set the flag which stops the
        execution after booting the program (i.e. reading the color data)."""

        file = open(pjoin(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(pjoin(dir_name,'loop_matrix.f'), 'w')        
        loop_matrix = re.sub(r"BOOTANDSTOP=\S+\)","BOOTANDSTOP=%s)"%('.TRUE.' 
                                    if bootandstop else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def setup_process(self, matrix_element, export_dir, reusing = False,
                                      param_card = None, MLOptions={},clean=True):
        """ Output the matrix_element in argument and perform the initialization
        while providing some details about the output in the dictionary returned. 
        Returns None if anything fails"""
                
        infos={'Process_output': None,
               'HELAS_MODEL_compilation' : None,
               'dir_path' : None,
               'Initialization' : None,
               'Process_compilation' : None}

        if not reusing and clean:
            if os.path.isdir(export_dir):
                clean_up(self.output_path)
                if os.path.isdir(export_dir):
                    raise InvalidCmd(\
                            "The directory %s already exist. Please remove it."\
                                                            %str(export_dir))
        else:
            if not os.path.isdir(export_dir):
                raise InvalidCmd(\
                    "Could not find the directory %s to reuse."%str(export_dir))                           
        

        if not reusing and clean:
            model = matrix_element['processes'][0].get('model')
            # I do the import here because there is some cyclic import of export_v4
            # otherwise
            import madgraph.loop.loop_exporters as loop_exporters
            if self.loop_optimized_output:
                exporter_class=loop_exporters.LoopProcessOptimizedExporterFortranSA
            else:
                exporter_class=loop_exporters.LoopProcessExporterFortranSA
    
            MLoptions = {'clean': True, 
                       'complex_mass': self.cmass_scheme,
                       'export_format':'madloop', 
                       'mp':True,
                       'SubProc_prefix':'P',
        'compute_color_flows':not matrix_element['processes'][0].get('has_born'),
          'loop_dir': pjoin(self.mg_root,'Template','loop_material'),
                       'cuttools_dir': self.cuttools_dir,
                       'fortran_compiler':self.cmd.options['fortran_compiler'],
                       'output_dependencies':self.cmd.options['output_dependencies']}
    
            MLoptions.update(self.tir_dir)

            start=time.time()
            FortranExporter = exporter_class(self.mg_root, export_dir, MLoptions)
            FortranModel = helas_call_writers.FortranUFOHelasCallWriter(model)
            FortranExporter.copy_v4template(modelname=model.get('name'))
            FortranExporter.generate_subprocess_directory_v4(matrix_element, FortranModel)
            wanted_lorentz = list(set(matrix_element.get_used_lorentz()))
            wanted_couplings = list(set([c for l in matrix_element.get_used_couplings() \
                                                                for c in l]))
            FortranExporter.convert_model_to_mg4(self.full_model,wanted_lorentz,wanted_couplings)
            infos['Process_output'] = time.time()-start
            start=time.time()
            FortranExporter.finalize_v4_directory(None,"",False,False,compiler=
                       {'fortran':self.cmd.options['fortran_compiler'],
                        'f2py':self.cmd.options['fortran_compiler'],
                        'cpp':self.cmd.options['fortran_compiler']})
            infos['HELAS_MODEL_compilation'] = time.time()-start
        
        # Copy the parameter card if provided
        if param_card != None:
            if isinstance(param_card, str):
                cp(pjoin(param_card),\
                              pjoin(export_dir,'Cards','param_card.dat'))
            else:
                param_card.write(pjoin(export_dir,'Cards','param_card.dat'))
                
        # First Initialize filters (in later versions where this will hopefully
        # be done at generation time, then it will be able to skip it)
        MadLoopInitializer.fix_PSPoint_in_check(
                 pjoin(export_dir,'SubProcesses'), read_ps = False, npoints = 4)

        self.fix_MadLoopParamCard(pjoin(export_dir,'Cards'),
                            mp = False, loop_filter = True,MLOptions=MLOptions)
        
        shell_name = None
        directories = misc.glob('P0_*', pjoin(export_dir, 'SubProcesses'))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])
        dir_name = pjoin(export_dir, 'SubProcesses', shell_name)
        infos['dir_path']=dir_name

        # Do not refresh the filter automatically as this is very often a waste
        # of time
        if not MadLoopInitializer.need_MadLoopInit(
                                                export_dir, subproc_prefix='P'):
            return infos

        attempts = [3,15]
        # remove check and check_sa.o for running initialization again
        try:
            os.remove(pjoin(dir_name,'check'))
            os.remove(pjoin(dir_name,'check_sa.o'))
        except OSError:
            pass

        nPS_necessary = MadLoopInitializer.run_initialization(dir_name,
                                pjoin(export_dir,'SubProcesses'),infos,\
                                req_files = ['HelFilter.dat','LoopFilter.dat'],
                                attempts = attempts)
        if attempts is None:
            logger.error("Could not compile the process %s,"%shell_name+\
                              " try to generate it via the 'generate' command.")
            return None
        if nPS_necessary is None:
            logger.error("Could not initialize the process %s"%shell_name+\
                                            " with %s PS points."%max(attempts))
            return None
        elif nPS_necessary > min(attempts):
            logger.warning("Could not initialize the process %s"%shell_name+\
              " with %d PS points. It needed %d."%(min(attempts),nPS_necessary))

        return infos

    def time_matrix_element(self, matrix_element, reusing = False,
                       param_card = None, keep_folder = False, options=None,
                       MLOptions = {}):
        """ Output the matrix_element in argument and give detail information
        about the timing for its output and running"""

        # If True, then force three PS points only and skip the test on
        # unpolarized PS point 
        make_it_quick=False

        if options and 'split_orders' in options.keys():
            split_orders = options['split_orders']
        else:
            split_orders = -1

        assert ((not reusing and isinstance(matrix_element, \
                 helas_objects.HelasMatrixElement)) or (reusing and 
                              isinstance(matrix_element, base_objects.Process)))
        if not reusing:
            proc_name = matrix_element['processes'][0].shell_string()[2:]
        else:
            proc_name = matrix_element.shell_string()[2:]
        
        export_dir=pjoin(self.output_path,('SAVED' if keep_folder else '')+\
                                                temp_dir_prefix+"_%s"%proc_name)

        res_timings = self.setup_process(matrix_element,export_dir, \
            reusing, param_card,MLOptions = MLOptions,clean=True)
        
        if res_timings == None:
            return None
        dir_name=res_timings['dir_path']

        def check_disk_usage(path):
            return subprocess.Popen("du -shc -L "+str(path), \
                stdout=subprocess.PIPE, shell=True).communicate()[0].split()[-2]
            # The above is compatible with python 2.6, not the neater version below
            #return subprocess.check_output(["du -shc %s"%path],shell=True).\
            #                                                         split()[-2]

        res_timings['du_source']=check_disk_usage(pjoin(\
                                                 export_dir,'Source','*','*.f'))
        res_timings['du_process']=check_disk_usage(pjoin(dir_name,'*.f'))
        res_timings['du_color']=check_disk_usage(pjoin(dir_name,
                                                  'MadLoop5_resources','*.dat'))
        res_timings['du_exe']=check_disk_usage(pjoin(dir_name,'check'))

        if not res_timings['Initialization']==None:
            time_per_ps_estimate = (res_timings['Initialization']/4.0)/2.0
        elif make_it_quick:
            time_per_ps_estimate = -1.0            
        else:
            # We cannot estimate from the initialization, so we run just a 3
            # PS point run to evaluate it.
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                                  read_ps = False, npoints = 3, hel_config = -1, 
                                                      split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
            time_per_ps_estimate = run_time/3.0
        
        self.boot_time_setup(dir_name,bootandstop=True)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        res_timings['Booting_time'] = run_time
        self.boot_time_setup(dir_name,bootandstop=False)

        # Detect one contributing helicity
        contributing_hel=0
        n_contrib_hel=0
        proc_prefix_file = open(pjoin(dir_name,'proc_prefix.txt'),'r')
        proc_prefix = proc_prefix_file.read()
        proc_prefix_file.close()
        helicities = file(pjoin(dir_name,'MadLoop5_resources',
                                  '%sHelFilter.dat'%proc_prefix)).read().split()
        for i, hel in enumerate(helicities):
            if (self.loop_optimized_output and int(hel)>-10000) or hel=='T':
                if contributing_hel==0:
                    contributing_hel=i+1
                n_contrib_hel += 1
                    
        if contributing_hel==0:
            logger.error("Could not find a contributing helicity "+\
                                     "configuration for process %s."%proc_name)
            return None
        
        res_timings['n_contrib_hel']=n_contrib_hel
        res_timings['n_tot_hel']=len(helicities)
        
        # We aim at a 30 sec run
        if not make_it_quick:
            target_pspoints_number = max(int(30.0/time_per_ps_estimate)+1,50)
        else:
            target_pspoints_number = 10
        
        logger.info("Checking timing for process %s "%proc_name+\
                                    "with %d PS points."%target_pspoints_number)
        
        MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        
        if compile_time == None: return None
        
        res_timings['run_polarized_total']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)

        if make_it_quick:
            res_timings['run_unpolarized_total'] = 1.0
            res_timings['ram_usage'] = 0.0
        else:
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                 read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                          split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name, 
                                                                      checkRam=True)

            if compile_time == None: return None
            res_timings['run_unpolarized_total']=\
                       (run_time-res_timings['Booting_time'])/target_pspoints_number
            res_timings['ram_usage'] = ram_usage       
        
        if not self.loop_optimized_output:
            return res_timings
        
        # For the loop optimized output, we also check the time spent in
        # computing the coefficients of the loop numerator polynomials.
        
        # So we modify loop_matrix.f in order to skip the loop evaluation phase.
        self.skip_loop_evaluation_setup(dir_name,skip=True)

        if make_it_quick:
            res_timings['run_unpolarized_coefs'] = 1.0
        else:
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                 read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                          split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
            if compile_time == None: return None
            res_timings['run_unpolarized_coefs']=\
                       (run_time-res_timings['Booting_time'])/target_pspoints_number
        
        MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_polarized_coefs']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)    

        # Restitute the original file.
        self.skip_loop_evaluation_setup(dir_name,skip=False)
        
        return res_timings

#===============================================================================
# Global helper function run_multiprocs
#===============================================================================

    def check_matrix_element_stability(self, matrix_element,options=None,
                          infos_IN = None, param_card = None, keep_folder = False,
                          MLOptions = {}):
        """ Output the matrix_element in argument, run in for nPoints and return
        a dictionary containing the stability information on each of these points.
        If infos are provided, then the matrix element output is skipped and 
        reused from a previous run and the content of infos.
        """
        
        if not options:
            reusing = False
            nPoints = 100
            split_orders = -1
        else:
            reusing = options['reuse']
            nPoints = options['npoints']
            split_orders = options['split_orders']
        
        assert ((not reusing and isinstance(matrix_element, \
                 helas_objects.HelasMatrixElement)) or (reusing and 
                              isinstance(matrix_element, base_objects.Process)))            

        # Helper functions
        def format_PS_point(ps, rotation=0):
            """ Write out the specified PS point to the file dir_path/PS.input
            while rotating it if rotation!=0. We consider only rotations of 90
            but one could think of having rotation of arbitrary angle too.
            The first two possibilities, 1 and 2 are a rotation and boost 
            along the z-axis so that improve_ps can still work.
            rotation=0  => No rotation
            rotation=1  => Z-axis pi/2 rotation
            rotation=2  => Z-axis pi/4 rotation
            rotation=3  => Z-axis boost            
            rotation=4 => (x'=z,y'=-x,z'=-y)
            rotation=5 => (x'=-z,y'=y,z'=x)"""
            if rotation==0:
                p_out=copy.copy(ps)
            elif rotation==1:
                p_out = [[pm[0],-pm[2],pm[1],pm[3]] for pm in ps]
            elif rotation==2:
                sq2 = math.sqrt(2.0)
                p_out = [[pm[0],(pm[1]-pm[2])/sq2,(pm[1]+pm[2])/sq2,pm[3]] for pm in ps]
            elif rotation==3:
                p_out = boost_momenta(ps, 3)     
            # From this point the transformations will prevent the
            # improve_ps script of MadLoop to work.      
            elif rotation==4:
                p_out=[[pm[0],pm[3],-pm[1],-pm[2]] for pm in ps]
            elif rotation==5:
                p_out=[[pm[0],-pm[3],pm[2],pm[1]] for pm in ps]
            else:
                raise MadGraph5Error("Rotation id %i not implemented"%rotation)
            
            return '\n'.join([' '.join(['%.16E'%pi for pi in p]) for p in p_out])
            
        def pick_PS_point(proc, options):
            """ Randomly generate a PS point and make sure it is eligible. Then
            return it. Users can edit the cuts here if they want."""

            p, w_rambo = self.get_momenta(proc, options)
            if options['events']:
                return p
            # For 2>1 process, we don't check the cuts of course
            while (not MatrixElementEvaluator.pass_isolation_cuts(p) and  len(p)>3):
                p, w_rambo = self.get_momenta(proc, options)
             
            # For a 2>1 process, it would always be the same PS point,
            # so here we bring in so boost along the z-axis, just for the sake
            # of it.
            if len(p)==3:
                p = boost_momenta(p,3,random.uniform(0.0,0.99))
            return p
        
        # Start loop on loop libraries        
        # Accuracy threshold of double precision evaluations above which the
        # PS points is also evaluated in quadruple precision
        accuracy_threshold=1.0e-1
        
        # Number of lorentz transformations to consider for the stability test
        # (along with the loop direction test which is performed by default)
        num_rotations = 1
        
        if "MLReductionLib" not in MLOptions:
            tools=[1]
        else:
            tools=MLOptions["MLReductionLib"]
            tools=list(set(tools)) # remove the duplication ones
            
        # not self-contained tir libraries
        tool_var={'pjfry':2,'golem':4,'samurai':5,'ninja':6}
        for tool in ['pjfry','golem','samurai','ninja']:
            tool_dir='%s_dir'%tool
            if not tool_dir in self.tir_dir:
                continue
            tool_libpath=self.tir_dir[tool_dir]
            tool_libname="lib%s.a"%tool
            if (not isinstance(tool_libpath,str)) or (not os.path.exists(tool_libpath)) \
                or (not os.path.isfile(pjoin(tool_libpath,tool_libname))):
                if tool_var[tool] in tools:
                    tools.remove(tool_var[tool])
        if not tools:
            return None
        
        # Normally, this should work for loop-induced processes as well
        if not reusing:
            process = matrix_element['processes'][0]
        else:
            process = matrix_element
        proc_name = process.shell_string()[2:]
        export_dir=pjoin(self.mg_root,("SAVED" if keep_folder else "")+\
                                                temp_dir_prefix+"_%s"%proc_name)
        
        tools_name={1:'CutTools',2:'PJFry++',3:'IREGI',4:'Golem95',5:'Samurai',
                    6:'Ninja'}
        return_dict={}
        return_dict['Stability']={}
        infos_save={'Process_output': None,
               'HELAS_MODEL_compilation' : None,
               'dir_path' : None,
               'Initialization' : None,
               'Process_compilation' : None} 
       
        for tool in tools:
            tool_name=tools_name[tool]
            # Each evaluations is performed in different ways to assess its stability.
            # There are two dictionaries, one for the double precision evaluation
            # and the second one for quadruple precision (if it was needed).
            # The keys are the name of the evaluation method and the value is the 
            # float returned.
            DP_stability = []
            QP_stability = []
            # The unstable point encountered are stored in this list
            Unstable_PS_points = []
            # The exceptional PS points are those which stay unstable in quad prec.
            Exceptional_PS_points = []
        
            MLoptions={}
            MLoptions["MLReductionLib"]=tool
            clean = (tool==tools[0]) and not nPoints==0
            if infos_IN==None or (tool_name not in infos_IN):
                infos=infos_IN
            else:
                infos=infos_IN[tool_name]

            if not infos:
                infos = self.setup_process(matrix_element,export_dir, \
                                            reusing, param_card,MLoptions,clean)
                if not infos:
                    return None
            
            if clean:
                infos_save['Process_output']=infos['Process_output']
                infos_save['HELAS_MODEL_compilation']=infos['HELAS_MODEL_compilation']
                infos_save['dir_path']=infos['dir_path']
                infos_save['Process_compilation']=infos['Process_compilation']
            else:
                if not infos['Process_output']:
                    infos['Process_output']=infos_save['Process_output']
                if not infos['HELAS_MODEL_compilation']:
                    infos['HELAS_MODEL_compilation']=infos_save['HELAS_MODEL_compilation']
                if not infos['dir_path']:
                    infos['dir_path']=infos_save['dir_path']
                if not infos['Process_compilation']:
                    infos['Process_compilation']=infos_save['Process_compilation']
                    
            dir_path=infos['dir_path']

            # Reuse old stability runs if present
            savefile='SavedStabilityRun_%s%%s.pkl'%tools_name[tool]
            data_i = 0
            
            if reusing:
                # Possibly add additional data than the main one in 0
                data_i=0
                while os.path.isfile(pjoin(dir_path,savefile%('_%d'%data_i))):
                    pickle_path = pjoin(dir_path,savefile%('_%d'%data_i))
                    saved_run = save_load_object.load_from_file(pickle_path)
                    if data_i>0:
                        logger.info("Loading additional data stored in %s."%
                                                               str(pickle_path))
                        logger.info("Loaded data moved to %s."%str(pjoin(
                                   dir_path,'LOADED_'+savefile%('_%d'%data_i))))
                        shutil.move(pickle_path,
                               pjoin(dir_path,'LOADED_'+savefile%('%d'%data_i)))
                    DP_stability.extend(saved_run['DP_stability'])
                    QP_stability.extend(saved_run['QP_stability'])
                    Unstable_PS_points.extend(saved_run['Unstable_PS_points'])
                    Exceptional_PS_points.extend(saved_run['Exceptional_PS_points'])
                    data_i += 1
                                        
            return_dict['Stability'][tool_name] = {'DP_stability':DP_stability,
                              'QP_stability':QP_stability,
                              'Unstable_PS_points':Unstable_PS_points,
                              'Exceptional_PS_points':Exceptional_PS_points}

            if nPoints==0:
                if len(return_dict['Stability'][tool_name]['DP_stability'])!=0:
                    # In case some data was combined, overwrite the pickle
                    if data_i>1:
                        save_load_object.save_to_file(pjoin(dir_path,
                             savefile%'_0'),return_dict['Stability'][tool_name])
                    continue
                else:
                    logger.info("ERROR: Not reusing a directory or any pickled"+
                                " result for tool %s and the number"%tool_name+\
                                             " of point for the check is zero.")
                    return None

            logger.info("Checking stability of process %s "%proc_name+\
                "with %d PS points by %s."%(nPoints,tool_name))
            if infos['Initialization'] != None:
                time_per_ps_estimate = (infos['Initialization']/4.0)/2.0
                sec_needed = int(time_per_ps_estimate*nPoints*4)
            else:
                sec_needed = 0
            
            progress_bar = None
            time_info = False
            if sec_needed>5:
                time_info = True
                logger.info("This check should take about "+\
                            "%s to run. Started on %s."%(\
                            str(datetime.timedelta(seconds=sec_needed)),\
                            datetime.datetime.now().strftime("%d-%m-%Y %H:%M")))
            if logger.getEffectiveLevel()<logging.WARNING and \
                (sec_needed>5 or (reusing and infos['Initialization'] == None)):
                widgets = ['Stability check:', pbar.Percentage(), ' ', 
                                            pbar.Bar(),' ', pbar.ETA(), ' ']
                progress_bar = pbar.ProgressBar(widgets=widgets, maxval=nPoints, 
                                                              fd=sys.stdout)
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
            read_ps = True, npoints = 1, hel_config = -1, split_orders=split_orders)
            # Recompile (Notice that the recompilation is only necessary once) for
            # the change above to take effect.
            # Make sure to recreate the executable and modified sources
            try:
                os.remove(pjoin(dir_path,'check'))
                os.remove(pjoin(dir_path,'check_sa.o'))
            except OSError:
                pass
            # Now run make
            devnull = open(os.devnull, 'w')
            retcode = subprocess.call(['make','check'],
                                   cwd=dir_path, stdout=devnull, stderr=devnull)
            devnull.close()    
            if retcode != 0:
                logging.info("Error while executing make in %s" % dir_path)
                return None
                

            # First create the stability check fortran driver executable if not 
            # already present.
            if not os.path.isfile(pjoin(dir_path,'StabilityCheckDriver.f')):
                # Use the presence of the file born_matrix.f to check if this output
                # is a loop_induced one or not.
                if os.path.isfile(pjoin(dir_path,'born_matrix.f')):
                    checkerName = 'StabilityCheckDriver.f'
                else:
                    checkerName = 'StabilityCheckDriver_loop_induced.f'

                with open(pjoin(self.mg_root,'Template','loop_material','Checks',
                                                checkerName),'r') as checkerFile:
                    with open(pjoin(dir_path,'proc_prefix.txt')) as proc_prefix:
                        checkerToWrite = checkerFile.read()%{'proc_prefix':
                                                                 proc_prefix.read()}
                checkerFile = open(pjoin(dir_path,'StabilityCheckDriver.f'),'w')
                checkerFile.write(checkerToWrite)
                checkerFile.close()                
                #cp(pjoin(self.mg_root,'Template','loop_material','Checks',\
                #    checkerName),pjoin(dir_path,'StabilityCheckDriver.f'))
        
            # Make sure to recompile the possibly modified files (time stamps can be
            # off).
            if os.path.isfile(pjoin(dir_path,'StabilityCheckDriver')):
                os.remove(pjoin(dir_path,'StabilityCheckDriver'))
            if os.path.isfile(pjoin(dir_path,'loop_matrix.o')):
                os.remove(pjoin(dir_path,'loop_matrix.o'))
            misc.compile(arg=['StabilityCheckDriver'], cwd=dir_path, \
                                              mode='fortran', job_specs = False)

            # Now for 2>1 processes, because the HelFilter was setup in for always
            # identical PS points with vec(p_1)=-vec(p_2), it is best not to remove
            # the helicityFilter double check
            if len(process['legs'])==3:
              self.fix_MadLoopParamCard(dir_path, mp=False,
                              loop_filter=False, DoubleCheckHelicityFilter=True)

            StabChecker = subprocess.Popen([pjoin(dir_path,'StabilityCheckDriver')], 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                                   cwd=dir_path)
            start_index = len(DP_stability)
            if progress_bar!=None:
                    progress_bar.start()

            # Flag to know if the run was interrupted or not
            interrupted = False
            # Flag to know wheter the run for one specific PS point got an IOError
            # and must be retried
            retry = 0
            # We do not use a for loop because we want to manipulate the updater.
            i=start_index
            if options and 'events' in options and options['events']:
                # it is necessary to reuse the events from lhe file
                import MadSpin.decay as madspin
                fsock = open(options['events'])
                self.event_file = madspin.Event(fsock)
            while i<(start_index+nPoints):
                # To be added to the returned statistics  
                qp_dict={}
                dp_dict={}
                UPS = None
                EPS = None
                # Pick an eligible PS point with rambo, if not already done
                if retry==0:
                    p = pick_PS_point(process, options)
#               print "I use P_%i="%i,p
                try:
                    if progress_bar!=None:
                        progress_bar.update(i+1-start_index)
                    # Write it in the input file
                    PSPoint = format_PS_point(p,0)
                    dp_res=[]
                    dp_res.append(self.get_me_value(StabChecker,PSPoint,1,
                                                     split_orders=split_orders))
                    dp_dict['CTModeA']=dp_res[-1]
                    dp_res.append(self.get_me_value(StabChecker,PSPoint,2,
                                                     split_orders=split_orders))
                    dp_dict['CTModeB']=dp_res[-1]
                    for rotation in range(1,num_rotations+1):
                        PSPoint = format_PS_point(p,rotation)
                        dp_res.append(self.get_me_value(StabChecker,PSPoint,1,
                                                     split_orders=split_orders))
                        dp_dict['Rotation%i'%rotation]=dp_res[-1]
                        # Make sure all results make sense
                    if any([not res for res in dp_res]):
                        return None
                    dp_accuracy =((max(dp_res)-min(dp_res))/
                                                   abs(sum(dp_res)/len(dp_res)))
                    dp_dict['Accuracy'] = dp_accuracy
                    if dp_accuracy>accuracy_threshold:
                        if tool in [1,6]:
                            # Only CutTools or Ninja can use QP
                            UPS = [i,p]
                            qp_res=[]
                            PSPoint = format_PS_point(p,0)
                            qp_res.append(self.get_me_value(StabChecker,PSPoint,4,
                                                         split_orders=split_orders))
                            qp_dict['CTModeA']=qp_res[-1]
                            qp_res.append(self.get_me_value(StabChecker,PSPoint,5,
                                                         split_orders=split_orders))
                            qp_dict['CTModeB']=qp_res[-1]
                            for rotation in range(1,num_rotations+1):
                                PSPoint = format_PS_point(p,rotation)
                                qp_res.append(self.get_me_value(StabChecker,PSPoint,4,
                                                         split_orders=split_orders))
                                qp_dict['Rotation%i'%rotation]=qp_res[-1]
                            # Make sure all results make sense
                            if any([not res for res in qp_res]):
                                return None
                        
                            qp_accuracy = ((max(qp_res)-min(qp_res))/
                                                   abs(sum(qp_res)/len(qp_res)))
                            qp_dict['Accuracy']=qp_accuracy
                            if qp_accuracy>accuracy_threshold:
                                EPS = [i,p]
                        else:
                            # Simply consider the point as a UPS when not using
                            # CutTools
                            UPS = [i,p]

                except KeyboardInterrupt:
                    interrupted = True
                    break
                except IOError, e:
                    if e.errno == errno.EINTR:
                        if retry==100:
                            logger.error("Failed hundred times consecutively because"+
                                               " of system call interruptions.")
                            raise
                        else:
                            logger.debug("Recovered from a system call interruption."+\
                                        "PSpoint #%i, Attempt #%i."%(i,retry+1))
                            # Sleep for half a second. Safety measure.
                            time.sleep(0.5)                        
                        # We will retry this PS point
                        retry = retry+1
                        # Make sure the MadLoop process is properly killed
                        try:
                            StabChecker.kill()
                        except Exception: 
                            pass
                        StabChecker = subprocess.Popen(\
                               [pjoin(dir_path,'StabilityCheckDriver')], 
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, cwd=dir_path)
                        continue
                    else:
                        raise
                
                # Successfully processed a PS point so,
                #  > reset retry
                retry = 0
                #  > Update the while loop counter variable
                i=i+1
            
                # Update the returned statistics
                DP_stability.append(dp_dict)
                QP_stability.append(qp_dict)
                if not EPS is None:
                    Exceptional_PS_points.append(EPS)
                if not UPS is None:
                    Unstable_PS_points.append(UPS)

            if progress_bar!=None:
                progress_bar.finish()
            if time_info:
                logger.info('Finished check on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))

            # Close the StabChecker process.
            if not interrupted:
                StabChecker.stdin.write('y\n')
            else:
                StabChecker.kill()
        
            #return_dict = {'DP_stability':DP_stability,
            #           'QP_stability':QP_stability,
            #           'Unstable_PS_points':Unstable_PS_points,
            #           'Exceptional_PS_points':Exceptional_PS_points}
        
            # Save the run for possible future use
            save_load_object.save_to_file(pjoin(dir_path,savefile%'_0'),\
                                          return_dict['Stability'][tool_name])

            if interrupted:
                break
        
        return_dict['Process'] =  matrix_element.get('processes')[0] if not \
                                                     reusing else matrix_element
        return return_dict

    @classmethod
    def get_me_value(cls, StabChecker, PSpoint, mode, hel=-1, mu_r=-1.0,
                                                               split_orders=-1):
        """ This version of get_me_value is simplified for the purpose of this
        class. No compilation is necessary. The CT mode can be specified."""

        # Reset the stdin with EOF character without closing it.
        StabChecker.stdin.write('\x1a')
        StabChecker.stdin.write('1\n')
        StabChecker.stdin.write('%d\n'%mode)   
        StabChecker.stdin.write('%s\n'%PSpoint)
        StabChecker.stdin.write('%.16E\n'%mu_r) 
        StabChecker.stdin.write('%d\n'%hel)
        StabChecker.stdin.write('%d\n'%split_orders)

        try:
            while True:
                output = StabChecker.stdout.readline()
                if output != '':
                    last_non_empty = output
                if output==' ##TAG#RESULT_START#TAG##\n':
                    break
                # Break if the checker has crashed for some reason.
                ret_code = StabChecker.poll()
                if not ret_code is None:
                    output = StabChecker.stdout.readline()
                    if output != '':
                        last_non_empty = output
                    error = StabChecker.stderr.readline()
                    raise MadGraph5Error, \
 "The MadLoop stability checker crashed with return code = %d, and last output:\n\nstdout: %s\nstderr: %s\n"%\
                                               (ret_code, last_non_empty, error)
                    
            res = ""
            while True:
                output = StabChecker.stdout.readline()
                if output != '':
                    last_non_empty = output
                if output==' ##TAG#RESULT_STOP#TAG##\n':
                    break
                else:
                    res += output
                ret_code = StabChecker.poll()                
                if not ret_code is None:
                    output = StabChecker.stdout.readline()
                    if output != '':
                        last_non_empty = output
                    error = StabChecker.stderr.readline()
                    raise MadGraph5Error, \
 "The MadLoop stability checker crashed with return code = %d, and last output:\n\nstdout: %s\nstderr: %s\n"%\
                                               (ret_code, last_non_empty, error)

            return cls.parse_check_output(res,format='tuple')[0][0]
        except IOError as e:
            logging.warning("Error while running MadLoop. Exception = %s"%str(e))
            raise e 

def evaluate_helicities(process, param_card = None, mg_root="", 
                                                          cmass_scheme = False):
    """ Perform a python evaluation of the matrix element independently for
    all possible helicity configurations for a fixed number of points N and 
    returns the average for each in the format [[hel_config, eval],...].
    This is used to determine what are the vanishing and dependent helicity 
    configurations at generation time and accordingly setup the output.
    This is not yet implemented at LO."""
    
    # Make sure this function is employed with a single process at LO
    assert isinstance(process,base_objects.Process)
    assert process.get('perturbation_couplings')==[]
    
    N_eval=50
    
    evaluator = MatrixElementEvaluator(process.get('model'), param_card,
                                            auth_skipping = False, reuse = True)
    
    amplitude = diagram_generation.Amplitude(process)
    matrix_element = helas_objects.HelasMatrixElement(amplitude,gen_color=False)
    
    cumulative_helEvals = []
    # Fill cumulative hel progressively with several evaluations of the ME.
    for i in range(N_eval):
        p, w_rambo = evaluator.get_momenta(process) 
        helEvals = evaluator.evaluate_matrix_element(\
                matrix_element, p = p, output = 'helEvals')['helEvals']
        if cumulative_helEvals==[]:
            cumulative_helEvals=copy.copy(helEvals)
        else:
            cumulative_helEvals = [[h[0],h[1]+helEvals[i][1]] for i, h in \
                                                 enumerate(cumulative_helEvals)]
            
    # Now normalize with the total number of evaluations
    cumulative_helEvals = [[h[0],h[1]/N_eval] for h in cumulative_helEvals]
    
    # As we are not in the context of a check command, so we clean the added
    # globals right away
    clean_added_globals(ADDED_GLOBAL)
    
    return cumulative_helEvals
    
def run_multiprocs_no_crossings(function, multiprocess, stored_quantities,
                                opt=None, options=None):
    """A wrapper function for running an iteration of a function over
    a multiprocess, without having to first create a process list
    (which makes a big difference for very large multiprocesses.
    stored_quantities is a dictionary for any quantities that we want
    to reuse between runs."""
    
    model = multiprocess.get('model')
    isids = [leg.get('ids') for leg in multiprocess.get('legs') \
              if not leg.get('state')]
    fsids = [leg.get('ids') for leg in multiprocess.get('legs') \
             if leg.get('state')]
    # Create dictionary between isids and antiids, to speed up lookup
    id_anti_id_dict = {}
    for id in set(tuple(sum(isids+fsids, []))):
        id_anti_id_dict[id] = model.get_particle(id).get_anti_pdg_code()
        id_anti_id_dict[model.get_particle(id).get_anti_pdg_code()] = id        
    sorted_ids = []
    results = []
    for is_prod in apply(itertools.product, isids):
        for fs_prod in apply(itertools.product, fsids):

            # Check if we have already checked the process
            if check_already_checked(is_prod, fs_prod, sorted_ids,
                                     multiprocess, model, id_anti_id_dict):
                continue
            # Generate process based on the selected ids
            process = multiprocess.get_process_with_legs(base_objects.LegList(\
                            [base_objects.Leg({'id': id, 'state':False}) for \
                             id in is_prod] + \
                            [base_objects.Leg({'id': id, 'state':True}) for \
                             id in fs_prod]))

            if opt is not None:
                if isinstance(opt, dict):
                    try:
                        value = opt[process.base_string()]
                    except Exception:
                        continue
                    result = function(process, stored_quantities, value, options=options)
                else:
                    result = function(process, stored_quantities, opt, options=options)
            else:
                result = function(process, stored_quantities, options=options)
                        
            if result:
                results.append(result)
            
    return results

#===============================================================================
# Helper function check_already_checked
#===============================================================================

def check_already_checked(is_ids, fs_ids, sorted_ids, process, model,
                          id_anti_id_dict = {}):
    """Check if process already checked, if so return True, otherwise add
    process and antiprocess to sorted_ids."""

    # Check if process is already checked
    if id_anti_id_dict:
        is_ids = [id_anti_id_dict[id] for id in \
                  is_ids]
    else:
        is_ids = [model.get_particle(id).get_anti_pdg_code() for id in \
                  is_ids]        

    ids = array.array('i', sorted(is_ids + list(fs_ids)) + \
                      [process.get('id')])

    if ids in sorted_ids:
        # We have already checked (a crossing of) this process
        return True

    # Add this process to tested_processes
    sorted_ids.append(ids)

    # Skip adding antiprocess below, since might be relevant too
    return False

#===============================================================================
# Generate a loop matrix element
#===============================================================================
def generate_loop_matrix_element(process_definition, reuse, output_path=None,
                        cmd = FakeInterface(), proc_name=None, loop_filter=None):
    """ Generate a loop matrix element from the process definition, and returns
    it along with the timing information dictionary.
    If reuse is True, it reuses the already output directory if found.
    There is the possibility of specifying the proc_name."""

    assert isinstance(process_definition,
                          (base_objects.ProcessDefinition,base_objects.Process))
    assert process_definition.get('perturbation_couplings')!=[]

    if isinstance(process_definition,base_objects.ProcessDefinition):
        if any(len(l.get('ids'))>1 for l in process_definition.get('legs')):
            raise InvalidCmd("This check can only be performed on single "+
                             " processes. (i.e. without multiparticle labels).")
    
        isids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
                  if not leg.get('state')]
        fsids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
                 if leg.get('state')]
    
        # Now generate a process based on the ProcessDefinition given in argument.
        process = process_definition.get_process(isids,fsids)
    else:
        process = process_definition
    
    if not output_path is None:
        root_path = output_path
    else:
        root_path = cmd._mgme_dir
    # By default, set all entries to None
    timing = {'Diagrams_generation': None,
              'n_loops': None,
              'HelasDiagrams_generation': None,
              'n_loop_groups': None,
              'n_loop_wfs': None,
              'loop_wfs_ranks': None}

    if proc_name:
        proc_dir = pjoin(root_path,proc_name)
    else:
        proc_dir = pjoin(root_path,"SAVED"+temp_dir_prefix+"_%s"%(
                               '_'.join(process.shell_string().split('_')[1:])))
    if reuse and os.path.isdir(proc_dir):
        logger.info("Reusing directory %s"%str(proc_dir))
        # If reusing, return process instead of matrix element
        return timing, process
    
    logger.info("Generating p%s"%process_definition.nice_string()[1:])

    start=time.time()
    try:
        amplitude = loop_diagram_generation.LoopAmplitude(process,
                                                        loop_filter=loop_filter)
    except InvalidCmd:
        # An error about the sanity of the process can be thrown, in which case
        # we return nothing
        return time.time()-start, None        
    if not amplitude.get('diagrams'):
        # Not matrix eleemnt for this process
        return time.time()-start, None

    # Make sure to disable loop_optimized_output when considering loop induced 
    # processes
    loop_optimized_output = cmd.options['loop_optimized_output']
    timing['Diagrams_generation']=time.time()-start
    timing['n_loops']=len(amplitude.get('loop_diagrams'))
    start=time.time()
    
    matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                        optimized_output = loop_optimized_output,gen_color=True)
    # Here, the alohaModel used for analytica computations and for the aloha
    # subroutine output will be different, so that some optimization is lost.
    # But that is ok for the check functionality.
    matrix_element.compute_all_analytic_information()
    timing['HelasDiagrams_generation']=time.time()-start
    
    if loop_optimized_output:
        timing['n_loop_groups']=len(matrix_element.get('loop_groups'))
        lwfs=[l for ldiag in matrix_element.get_loop_diagrams() for l in \
                                                ldiag.get('loop_wavefunctions')]
        timing['n_loop_wfs']=len(lwfs)
        timing['loop_wfs_ranks']=[]
        for rank in range(0,max([l.get_analytic_info('wavefunction_rank') \
                                                             for l in lwfs])+1):
            timing['loop_wfs_ranks'].append(\
                len([1 for l in lwfs if \
                               l.get_analytic_info('wavefunction_rank')==rank]))

    return timing, matrix_element

#===============================================================================
# check profile for loop process (timings + stability in one go)
#===============================================================================
def check_profile(process_definition, param_card = None,cuttools="",tir={},
             options = {}, cmd = FakeInterface(),output_path=None,MLOptions={}):
    """For a single loop process, check both its timings and then its stability
    in one go without regenerating it."""

    if 'reuse' not in options:
        keep_folder=False
    else:
        keep_folder = options['reuse']

    model=process_definition.get('model')

    timing1, matrix_element = generate_loop_matrix_element(process_definition,
                                    keep_folder,output_path=output_path,cmd=cmd)
    reusing = isinstance(matrix_element, base_objects.Process)
    options['reuse'] = reusing
    myProfiler = LoopMatrixElementTimer(cuttools_dir=cuttools,tir_dir=tir,
                                  model=model, output_path=output_path, cmd=cmd)

    if not myProfiler.loop_optimized_output:
        MLoptions={}
    else:
        MLoptions=MLOptions

    timing2 = myProfiler.time_matrix_element(matrix_element, reusing, 
                            param_card, keep_folder=keep_folder,options=options,
                            MLOptions = MLoptions)
    
    if timing2 == None:
        return None, None

    # The timing info is made of the merged two dictionaries
    timing = dict(timing1.items()+timing2.items())
    stability = myProfiler.check_matrix_element_stability(matrix_element,                                            
                            options=options, infos_IN=timing,param_card=param_card,
                                                      keep_folder = keep_folder,
                                                      MLOptions = MLoptions)
    if stability == None:
        return None, None
    else:
        timing['loop_optimized_output']=myProfiler.loop_optimized_output
        stability['loop_optimized_output']=myProfiler.loop_optimized_output
        return timing, stability

#===============================================================================
# check_timing for loop processes
#===============================================================================
def check_stability(process_definition, param_card = None,cuttools="",tir={}, 
                               options=None,nPoints=100, output_path=None,
                               cmd = FakeInterface(), MLOptions = {}):
    """For a single loop process, give a detailed summary of the generation and
    execution timing."""

    if "reuse" in options:
        reuse=options['reuse']
    else:
        reuse=False

    reuse=options['reuse']
    keep_folder = reuse
    model=process_definition.get('model')

    timing, matrix_element = generate_loop_matrix_element(process_definition,
                                        reuse, output_path=output_path, cmd=cmd)
    reusing = isinstance(matrix_element, base_objects.Process)
    options['reuse'] = reusing
    myStabilityChecker = LoopMatrixElementTimer(cuttools_dir=cuttools,tir_dir=tir,
                                    output_path=output_path,model=model,cmd=cmd)

    if not myStabilityChecker.loop_optimized_output:
        MLoptions = {}
    else:
        MLoptions = MLOptions
        if "MLReductionLib" not in MLOptions:
            MLoptions["MLReductionLib"] = []
            if cuttools:
                MLoptions["MLReductionLib"].extend([1])
            if "iregi_dir" in tir:
                MLoptions["MLReductionLib"].extend([3])
            if "pjfry_dir" in tir:
                MLoptions["MLReductionLib"].extend([2])
            if "golem_dir" in tir:
                MLoptions["MLReductionLib"].extend([4])
            if "samurai_dir" in tir:
                MLoptions["MLReductionLib"].extend([5])
            if "ninja_dir" in tir:
                MLoptions["MLReductionLib"].extend([6])

    stability = myStabilityChecker.check_matrix_element_stability(matrix_element, 
                        options=options,param_card=param_card, 
                                                        keep_folder=keep_folder,
                                                        MLOptions=MLoptions)
    
    if stability == None:
        return None
    else:
        stability['loop_optimized_output']=myStabilityChecker.loop_optimized_output
        return stability

#===============================================================================
# check_timing for loop processes
#===============================================================================
def check_timing(process_definition, param_card= None, cuttools="",tir={},
                           output_path=None, options={}, cmd = FakeInterface(),
                                                                MLOptions = {}):                 
    """For a single loop process, give a detailed summary of the generation and
    execution timing."""

    if 'reuse' not in options:
        keep_folder = False
    else:
        keep_folder = options['reuse']
    model=process_definition.get('model')
    timing1, matrix_element = generate_loop_matrix_element(process_definition,
                                  keep_folder, output_path=output_path, cmd=cmd)
    reusing = isinstance(matrix_element, base_objects.Process)
    options['reuse'] = reusing
    myTimer = LoopMatrixElementTimer(cuttools_dir=cuttools,model=model,tir_dir=tir,
                                               output_path=output_path, cmd=cmd)

    if not myTimer.loop_optimized_output:
        MLoptions = {}
    else:
        MLoptions = MLOptions
    timing2 = myTimer.time_matrix_element(matrix_element, reusing, param_card,
                                     keep_folder = keep_folder, options=options,
                                     MLOptions = MLoptions)
    
    if timing2 == None:
        return None
    else:    
        # Return the merged two dictionaries
        res = dict(timing1.items()+timing2.items())
        res['loop_optimized_output']=myTimer.loop_optimized_output
        return res

#===============================================================================
# check_processes
#===============================================================================
def check_processes(processes, param_card = None, quick = [],cuttools="",tir={},
          options=None, reuse = False, output_path=None, cmd = FakeInterface()):
    """Check processes by generating them with all possible orderings
    of particles (which means different diagram building and Helas
    calls), and comparing the resulting matrix element values."""

    cmass_scheme = cmd.options['complex_mass_scheme']
    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes
        model = multiprocess.get('model')

        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model,
               auth_skipping = True, reuse = False, cmd = cmd)
        else:
            evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir, 
                            model=model, auth_skipping = True,
                            reuse = False, output_path=output_path, cmd = cmd)
       
        results = run_multiprocs_no_crossings(check_process,
                                              multiprocess,
                                              evaluator,
                                              quick,
                                              options)

        if "used_lorentz" not in evaluator.stored_quantities:
            evaluator.stored_quantities["used_lorentz"] = []
            
        if multiprocess.get('perturbation_couplings')!=[] and not reuse:
            # Clean temporary folders created for the running of the loop processes
            clean_up(output_path)
            
        return results, evaluator.stored_quantities["used_lorentz"]

    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    if not processes:
        raise InvalidCmd("No processes given")

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    if processes[0].get('perturbation_couplings')==[]:
        evaluator = MatrixElementEvaluator(model, param_card,
               auth_skipping = True, reuse = False, cmd = cmd)
    else:
        evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools, tir_dir=tir,
                                               model=model,param_card=param_card,
                                           auth_skipping = True, reuse = False,
                                           output_path=output_path, cmd = cmd)

    # Keep track of tested processes, matrix elements, color and already
    # initiated Lorentz routines, to reuse as much as possible
    sorted_ids = []
    comparison_results = []

    # Check process by process
    for process in processes:
        
        # Check if we already checked process        
        if check_already_checked([l.get('id') for l in process.get('legs') if \
                                  not l.get('state')],
                                 [l.get('id') for l in process.get('legs') if \
                                  l.get('state')],
                                 sorted_ids, process, model):
            continue
        # Get process result
        res = check_process(process, evaluator, quick, options)
        if res:
            comparison_results.append(res)

    if "used_lorentz" not in evaluator.stored_quantities:
        evaluator.stored_quantities["used_lorentz"] = []
    
    if processes[0].get('perturbation_couplings')!=[] and not reuse:
        # Clean temporary folders created for the running of the loop processes
        clean_up(output_path)    
    
    return comparison_results, evaluator.stored_quantities["used_lorentz"]

def check_process(process, evaluator, quick, options):
    """Check the helas calls for a process by generating the process
    using all different permutations of the process legs (or, if
    quick, use a subset of permutations), and check that the matrix
    element is invariant under this."""

    model = process.get('model')

    # Ensure that leg numbers are set
    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking crossings of %s" % \
                process.nice_string().replace('Process:', 'process'))

    process_matrix_elements = []

    # For quick checks, only test twp permutations with leg "1" in
    # each position
    if quick:
        leg_positions = [[] for leg in process.get('legs')]
        quick = range(1,len(process.get('legs')) + 1)

    values = []

    # Now, generate all possible permutations of the legs
    number_checked=0
    for legs in itertools.permutations(process.get('legs')):
        
        order = [l.get('number') for l in legs]
        if quick:
            found_leg = True
            for num in quick:
                # Only test one permutation for each position of the
                # specified legs
                leg_position = legs.index([l for l in legs if \
                                           l.get('number') == num][0])

                if not leg_position in leg_positions[num-1]:
                    found_leg = False
                    leg_positions[num-1].append(leg_position)

            if found_leg:
                continue
        
        # Further limit the total number of permutations checked to 3 for
        # loop processes.
        if quick and process.get('perturbation_couplings') and number_checked >3:
            continue

        legs = base_objects.LegList(legs)

        if order != range(1,len(legs) + 1):
            logger.info("Testing permutation: %s" % \
                        order)
        
        newproc = copy.copy(process)
        newproc.set('legs',legs)

        # Generate the amplitude for this process
        try:
            if newproc.get('perturbation_couplings')==[]:
                amplitude = diagram_generation.Amplitude(newproc)
            else:
                # Change the cutting method every two times.
                loop_base_objects.cutting_method = 'optimal' if \
                                            number_checked%2 == 0 else 'default'
                amplitude = loop_diagram_generation.LoopAmplitude(newproc)
        except InvalidCmd:
            result=False
        else:
            result = amplitude.get('diagrams')
        # Make sure to re-initialize the cutting method to the original one.
        loop_base_objects.cutting_method = 'optimal'
        
        if not result:
            # This process has no diagrams; go to next process
            logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
            break

        if order == range(1,len(legs) + 1):
            # Generate phase space point to use
            p, w_rambo = evaluator.get_momenta(process, options)

        # Generate the HelasMatrixElement for the process
        if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
            matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                          gen_color=False)
        else:
            matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                               optimized_output=evaluator.loop_optimized_output)

        # The loop diagrams are always the same in the basis, so that the
        # LoopHelasMatrixElement always look alike. One needs to consider
        # the crossing no matter what then.
        if amplitude.get('process').get('has_born'):
            # But the born diagrams will change depending on the order of the
            # particles in the process definition
            if matrix_element in process_matrix_elements:
                # Exactly the same matrix element has been tested
                # for other permutation of same process
                continue

        process_matrix_elements.append(matrix_element)

        res = evaluator.evaluate_matrix_element(matrix_element, p = p, 
                                                                options=options)
        if res == None:
            break

        values.append(res[0])
        number_checked += 1

        # Check if we failed badly (1% is already bad) - in that
        # case done for this process
        if abs(max(values)) + abs(min(values)) > 0 and \
               2 * abs(max(values) - min(values)) / \
               (abs(max(values)) + abs(min(values))) > 0.01:
            break
    
    # Check if process was interrupted
    if not values:
        return None

    # Done with this process. Collect values, and store
    # process and momenta
    diff = 0
    if abs(max(values)) + abs(min(values)) > 0:
        diff = 2* abs(max(values) - min(values)) / \
               (abs(max(values)) + abs(min(values)))

    # be more tolerant with loop processes
    if process.get('perturbation_couplings'):
        passed = diff < 1.e-5
    else:
        passed = diff < 1.e-8        

    return {"process": process,
            "momenta": p,
            "values": values,
            "difference": diff,
            "passed": passed}

def clean_up(mg_root):
    """Clean-up the possible left-over outputs from 'evaluate_matrix element' of
    the LoopMatrixEvaluator (when its argument proliferate is set to true). """
    
    if mg_root is None:
        pass
    
    directories = misc.glob('%s*' % temp_dir_prefix, mg_root)
    if directories != []:
        logger.debug("Cleaning temporary %s* check runs."%temp_dir_prefix)
    for dir in directories:
        # For safety make sure that the directory contains a folder SubProcesses
        if os.path.isdir(pjoin(dir,'SubProcesses')):
            shutil.rmtree(dir)

def format_output(output,format):
    """ Return a string for 'output' with the specified format. If output is 
    None, it returns 'NA'."""
    
    if output!=None:
        return format%output
    else:
        return 'NA'

def output_profile(myprocdef, stability, timing, output_path, reusing=False):
    """Present the results from a timing and stability consecutive check"""

    opt = timing['loop_optimized_output']

    text = 'Timing result for the '+('optimized' if opt else 'default')+\
                                                                    ' output:\n'
    text += output_timings(myprocdef,timing)

    text += '\nStability result for the '+('optimized' if opt else 'default')+\
                                                                    ' output:\n'
    text += output_stability(stability,output_path, reusing=reusing)

    mode = 'optimized' if opt else 'default'
    logFilePath =  pjoin(output_path, 'profile_%s_%s.log'\
                                    %(mode,stability['Process'].shell_string()))        
    logFile = open(logFilePath, 'w')
    logFile.write(text)
    logFile.close()
    logger.info('Log of this profile check was output to file %s'\
                                                              %str(logFilePath))
    return text

def output_stability(stability, output_path, reusing=False):
    """Present the result of a stability check in a nice format.
    The full info is printed out in 'Stability_result_<proc_shell_string>.dat'
    under the MadGraph5_aMC@NLO root folder (output_path)"""
    
    def accuracy(eval_list):
        """ Compute the accuracy from different evaluations."""
        return (2.0*(max(eval_list)-min(eval_list))/
                                             abs(max(eval_list)+min(eval_list)))
    
    def best_estimate(eval_list):
        """ Returns the best estimate from different evaluations."""
        return (max(eval_list)+min(eval_list))/2.0
    
    def loop_direction_test_power(eval_list):
        """ Computes the loop direction test power P is computed as follow:
          P = accuracy(loop_dir_test) / accuracy(all_test)
        So that P is large if the loop direction test is effective.
        The tuple returned is (log(median(P)),log(min(P)),frac)
        where frac is the fraction of events with powers smaller than -3
        which means events for which the reading direction test shows an
        accuracy three digits higher than it really is according to the other
        tests."""
        powers=[]
        for eval in eval_list:
            loop_dir_evals = [eval['CTModeA'],eval['CTModeB']]
            # CTModeA is the reference so we keep it in too
            other_evals = [eval[key] for key in eval.keys() if key not in \
                                                         ['CTModeB','Accuracy']]
            if accuracy(other_evals)!=0.0 and accuracy(loop_dir_evals)!=0.0:
                powers.append(accuracy(loop_dir_evals)/accuracy(other_evals))
        
        n_fail=0
        for p in powers:
            if (math.log(p)/math.log(10))<-3:
                n_fail+=1
                
        if len(powers)==0:
            return (None,None,None)

        return (math.log(median(powers))/math.log(10),
                math.log(min(powers))/math.log(10),
                n_fail/len(powers))
        
    def test_consistency(dp_eval_list, qp_eval_list):
        """ Computes the consistency test C from the DP and QP evaluations.
          C = accuracy(all_DP_test) / abs(best_QP_eval-best_DP_eval)
        So a consistent test would have C as close to one as possible.
        The tuple returned is (log(median(C)),log(min(C)),log(max(C)))"""
        consistencies = []
        for dp_eval, qp_eval in zip(dp_eval_list,qp_eval_list):
            dp_evals = [dp_eval[key] for key in dp_eval.keys() \
                                                             if key!='Accuracy']
            qp_evals = [qp_eval[key] for key in qp_eval.keys() \
                                                             if key!='Accuracy']
            if (abs(best_estimate(qp_evals)-best_estimate(dp_evals)))!=0.0 and \
               accuracy(dp_evals)!=0.0:
                consistencies.append(accuracy(dp_evals)/(abs(\
                              best_estimate(qp_evals)-best_estimate(dp_evals))))

        if len(consistencies)==0:
            return (None,None,None)

        return (math.log(median(consistencies))/math.log(10),
                math.log(min(consistencies))/math.log(10),
                math.log(max(consistencies))/math.log(10))
    
    def median(orig_list):
        """ Find the median of a sorted float list. """
        list=copy.copy(orig_list)
        list.sort()
        if len(list)%2==0:
            return (list[int((len(list)/2)-1)]+list[int(len(list)/2)])/2.0
        else:
            return list[int((len(list)-1)/2)]

    # Define shortcut
    f = format_output   
        
    opt = stability['loop_optimized_output']

    mode = 'optimized' if opt else 'default'
    process = stability['Process']
    res_str = "Stability checking for %s (%s mode)\n"\
                                           %(process.nice_string()[9:],mode)

    logFile = open(pjoin(output_path, 'stability_%s_%s.log'\
                                           %(mode,process.shell_string())), 'w')

    logFile.write('Stability check results\n\n')
    logFile.write(res_str)
    data_plot_dict={}
    accuracy_dict={}
    nPSmax=0
    max_acc=0.0
    min_acc=1.0
    if stability['Stability']:
        toolnames= stability['Stability'].keys()
        toolnamestr="     |     ".join(tn+
                                ''.join([' ']*(10-len(tn))) for tn in toolnames)
        DP_stability = [[eval['Accuracy'] for eval in stab['DP_stability']] \
                        for key,stab in stability['Stability'].items()]
        med_dp_stab_str="     |     ".join([f(median(dp_stab),'%.2e  ') for dp_stab in  DP_stability])
        min_dp_stab_str="     |     ".join([f(min(dp_stab),'%.2e  ') for dp_stab in  DP_stability])
        max_dp_stab_str="     |     ".join([f(max(dp_stab),'%.2e  ') for dp_stab in  DP_stability])
        UPS = [stab['Unstable_PS_points'] for key,stab in stability['Stability'].items()]
        res_str_i  = "\n= Tool (DoublePrec for CT).......   %s\n"%toolnamestr
        len_PS=["%i"%len(evals)+\
             ''.join([' ']*(10-len("%i"%len(evals)))) for evals in DP_stability]
        len_PS_str="     |     ".join(len_PS)
        res_str_i += "|= Number of PS points considered   %s\n"%len_PS_str        
        res_str_i += "|= Median accuracy...............   %s\n"%med_dp_stab_str
        res_str_i += "|= Max accuracy..................   %s\n"%min_dp_stab_str
        res_str_i += "|= Min accuracy..................   %s\n"%max_dp_stab_str
        pmedminlist=[]
        pfraclist=[]
        for key,stab in stability['Stability'].items():
            (pmed,pmin,pfrac)=loop_direction_test_power(stab['DP_stability'])
            ldtest_str = "%s,%s"%(f(pmed,'%.1f'),f(pmin,'%.1f'))
            pfrac_str = f(pfrac,'%.2e')
            pmedminlist.append(ldtest_str+''.join([' ']*(10-len(ldtest_str))))
            pfraclist.append(pfrac_str+''.join([' ']*(10-len(pfrac_str))))
        pmedminlist_str="     |     ".join(pmedminlist)
        pfraclist_str="     |     ".join(pfraclist)
        res_str_i += "|= Overall DP loop_dir test power   %s\n"%pmedminlist_str
        res_str_i += "|= Fraction of evts with power<-3   %s\n"%pfraclist_str
        len_UPS=["%i"%len(upup)+\
                        ''.join([' ']*(10-len("%i"%len(upup)))) for upup in UPS]
        len_UPS_str="     |     ".join(len_UPS)
        res_str_i += "|= Number of Unstable PS points     %s\n"%len_UPS_str
        res_str_i += \
            """
= Legend for the statistics of the stability tests. (all log below ar log_10)
The loop direction test power P is computed as follow:
    P = accuracy(loop_dir_test) / accuracy(all_other_test)
    So that log(P) is positive if the loop direction test is effective.
  The tuple printed out is (log(median(P)),log(min(P)))
  The consistency test C is computed when QP evaluations are available:
     C = accuracy(all_DP_test) / abs(best_QP_eval-best_DP_eval)
  So a consistent test would have log(C) as close to zero as possible.
  The tuple printed out is (log(median(C)),log(min(C)),log(max(C)))\n"""
        res_str+=res_str_i
    for key in stability['Stability'].keys():
        toolname=key
        stab=stability['Stability'][key]
        DP_stability = [eval['Accuracy'] for eval in stab['DP_stability']]
        # Remember that an evaluation which did not require QP has an empty dictionary
        QP_stability = [eval['Accuracy'] if eval!={} else -1.0 for eval in \
                                                      stab['QP_stability']]
        nPS = len(DP_stability)
        if nPS>nPSmax:nPSmax=nPS
        UPS = stab['Unstable_PS_points']
        UPS_stability_DP = [DP_stability[U[0]] for U in UPS]
        UPS_stability_QP = [QP_stability[U[0]] for U in UPS]
        EPS = stab['Exceptional_PS_points']
        EPS_stability_DP = [DP_stability[E[0]] for E in EPS]
        EPS_stability_QP = [QP_stability[E[0]] for E in EPS]
        res_str_i = ""
        # Use nicer name for the XML tag in the log file
        xml_toolname = {'GOLEM95':'GOLEM','IREGI':'IREGI',
                        'CUTTOOLS':'CUTTOOLS','PJFRY++':'PJFRY',
                        'NINJA':'NINJA','SAMURAI':'SAMURAI'}[toolname.upper()]
        if len(UPS)>0:
            res_str_i = "\nDetails of the %d/%d UPS encountered by %s\n"\
                                                        %(len(UPS),nPS,toolname)
            prefix = 'DP' if toolname=='CutTools' else '' 
            res_str_i += "|= %s Median inaccuracy.......... %s\n"\
                                    %(prefix,f(median(UPS_stability_DP),'%.2e'))
            res_str_i += "|= %s Max accuracy............... %s\n"\
                                       %(prefix,f(min(UPS_stability_DP),'%.2e'))
            res_str_i += "|= %s Min accuracy............... %s\n"\
                                       %(prefix,f(max(UPS_stability_DP),'%.2e'))
            (pmed,pmin,pfrac)=loop_direction_test_power(\
                                 [stab['DP_stability'][U[0]] for U in UPS])
            if toolname=='CutTools':
                res_str_i += "|= UPS DP loop_dir test power.... %s,%s\n"\
                                                %(f(pmed,'%.1f'),f(pmin,'%.1f'))
                res_str_i += "|= UPS DP fraction with power<-3. %s\n"\
                                                                %f(pfrac,'%.2e')
                res_str_i += "|= QP Median accuracy............ %s\n"\
                                             %f(median(UPS_stability_QP),'%.2e')
                res_str_i += "|= QP Max accuracy............... %s\n"\
                                                %f(min(UPS_stability_QP),'%.2e')
                res_str_i += "|= QP Min accuracy............... %s\n"\
                                                %f(max(UPS_stability_QP),'%.2e')
                (pmed,pmin,pfrac)=loop_direction_test_power(\
                                     [stab['QP_stability'][U[0]] for U in UPS])
                res_str_i += "|= UPS QP loop_dir test power.... %s,%s\n"\
                                                %(f(pmed,'%.1f'),f(pmin,'%.1f'))
                res_str_i += "|= UPS QP fraction with power<-3. %s\n"%f(pfrac,'%.2e')
                (pmed,pmin,pmax)=test_consistency(\
                                     [stab['DP_stability'][U[0]] for U in UPS],
                                     [stab['QP_stability'][U[0]] for U in UPS])
                res_str_i += "|= DP vs QP stab test consistency %s,%s,%s\n"\
                                     %(f(pmed,'%.1f'),f(pmin,'%.1f'),f(pmax,'%.1f'))
            if len(EPS)==0:    
                res_str_i += "= Number of Exceptional PS points : 0\n"
        if len(EPS)>0:
            res_str_i = "\nDetails of the %d/%d EPS encountered by %s\n"\
                                                        %(len(EPS),nPS,toolname)
            res_str_i += "|= DP Median accuracy............ %s\n"\
                                             %f(median(EPS_stability_DP),'%.2e')
            res_str_i += "|= DP Max accuracy............... %s\n"\
                                                %f(min(EPS_stability_DP),'%.2e')
            res_str_i += "|= DP Min accuracy............... %s\n"\
                                                %f(max(EPS_stability_DP),'%.2e')
            pmed,pmin,pfrac=loop_direction_test_power(\
                                 [stab['DP_stability'][E[0]] for E in EPS])
            res_str_i += "|= EPS DP loop_dir test power.... %s,%s\n"\
                                                %(f(pmed,'%.1f'),f(pmin,'%.1f'))
            res_str_i += "|= EPS DP fraction with power<-3. %s\n"\
                                                                %f(pfrac,'%.2e')
            res_str_i += "|= QP Median accuracy............ %s\n"\
                                             %f(median(EPS_stability_QP),'%.2e')
            res_str_i += "|= QP Max accuracy............... %s\n"\
                                                %f(min(EPS_stability_QP),'%.2e')
            res_str_i += "|= QP Min accuracy............... %s\n"\
                                                %f(max(EPS_stability_QP),'%.2e')
            pmed,pmin,pfrac=loop_direction_test_power(\
                                 [stab['QP_stability'][E[0]] for E in EPS])
            res_str_i += "|= EPS QP loop_dir test power.... %s,%s\n"\
                                                %(f(pmed,'%.1f'),f(pmin,'%.1f'))
            res_str_i += "|= EPS QP fraction with power<-3. %s\n"%f(pfrac,'%.2e')

        logFile.write(res_str_i)

        if len(EPS)>0:
            logFile.write('\nFull details of the %i EPS encountered by %s.\n'\
                                                           %(len(EPS),toolname))
            logFile.write('<EPS_data reduction=%s>\n'%xml_toolname.upper())
            for i, eps in enumerate(EPS):
                logFile.write('\nEPS #%i\n'%(i+1))
                logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in eps[1]]))
                logFile.write('\n  DP accuracy :  %.4e\n'%DP_stability[eps[0]])
                logFile.write('  QP accuracy :  %.4e\n'%QP_stability[eps[0]])
            logFile.write('</EPS_data>\n')
        if len(UPS)>0:
            logFile.write('\nFull details of the %i UPS encountered by %s.\n'\
                                                           %(len(UPS),toolname))
            logFile.write('<UPS_data reduction=%s>\n'%xml_toolname.upper())
            for i, ups in enumerate(UPS):
                logFile.write('\nUPS #%i\n'%(i+1))
                logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in ups[1]]))
                logFile.write('\n  DP accuracy :  %.4e\n'%DP_stability[ups[0]])
                logFile.write('  QP accuracy :  %.4e\n'%QP_stability[ups[0]])
            logFile.write('</UPS_data>\n')

        logFile.write('\nData entries for the stability plot.\n')
        logFile.write('First row is a maximal accuracy delta, second is the '+\
                  'fraction of events with DP accuracy worse than delta.\n')
        logFile.write('<plot_data reduction=%s>\n'%xml_toolname.upper())
    # Set the x-range so that it spans [10**-17,10**(min_digit_accuracy)]
        if max(DP_stability)>0.0:
            min_digit_acc=int(math.log(max(DP_stability))/math.log(10))
            if min_digit_acc>=0:
                min_digit_acc = min_digit_acc+1
            accuracies=[10**(-17+(i/5.0)) for i in range(5*(17+min_digit_acc)+1)]
        else:
            logFile.writelines('%.4e  %.4e\n'%(accuracies[i], 0.0) for i in \
                                                         range(len(accuracies)))      
            logFile.write('</plot_data>\n')
            res_str_i += '\nPerfect accuracy over all the trial PS points. No plot'+\
                                                              ' is output then.'
            logFile.write('Perfect accuracy over all the trial PS points.')
            res_str +=res_str_i
            continue

        accuracy_dict[toolname]=accuracies
        if max(accuracies) > max_acc: max_acc=max(accuracies)
        if min(accuracies) < min_acc: min_acc=min(accuracies)
        data_plot=[]
        for acc in accuracies:
            data_plot.append(float(len([d for d in DP_stability if d>acc]))\
                                                      /float(len(DP_stability)))
        data_plot_dict[toolname]=data_plot
        
        logFile.writelines('%.4e  %.4e\n'%(accuracies[i], data_plot[i]) for i in \
                                                         range(len(accuracies)))
        logFile.write('</plot_data>\n')
        logFile.write('\nList of accuracies recorded for the %i evaluations with %s\n'\
                                                                %(nPS,toolname))
        logFile.write('First row is DP, second is QP (if available).\n\n')
        logFile.write('<accuracies reduction=%s>\n'%xml_toolname.upper())
        logFile.writelines('%.4e  '%DP_stability[i]+('NA\n' if QP_stability[i]==-1.0 \
                             else '%.4e\n'%QP_stability[i]) for i in range(nPS))
        logFile.write('</accuracies>\n')
        res_str+=res_str_i
    logFile.close()
    res_str += "\n= Stability details of the run are output to the file"+\
                          " stability_%s_%s.log\n"%(mode,process.shell_string())
                          
    # Bypass the plotting if the madgraph logger has a FileHandler (like it is
    # done in the check command acceptance test) because in this case it makes
    # no sense to plot anything.
    if any(isinstance(handler,logging.FileHandler) for handler in \
                                        logging.getLogger('madgraph').handlers):
        return res_str

    try:
        import matplotlib.pyplot as plt
        colorlist=['b','r','g','y','m','c']
        for i,key in enumerate(data_plot_dict.keys()):
            color=colorlist[i]
            data_plot=data_plot_dict[key]
            accuracies=accuracy_dict[key]
            plt.plot(accuracies, data_plot, color=color, marker='', linestyle='-',\
                     label=key)
        plt.axis([min_acc,max_acc,\
                               10**(-int(math.log(nPSmax-0.5)/math.log(10))-1), 1])
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Stability plot for %s (%s mode, %d points)'%\
                                           (process.nice_string()[9:],mode,nPSmax))
        plt.ylabel('Fraction of events')
        plt.xlabel('Maximal precision')
        plt.legend()
        if not reusing:
            logger.info('Some stability statistics will be displayed once you '+\
                                                        'close the plot window')
            plt.show()
        else:
            fig_output_file = str(pjoin(output_path, 
                     'stability_plot_%s_%s.png'%(mode,process.shell_string())))
            logger.info('Stability plot output to file %s. '%fig_output_file)
            plt.savefig(fig_output_file)
        return res_str
    except Exception as e:
        if isinstance(e, ImportError):
            res_str += "\n= Install matplotlib to get a "+\
                               "graphical display of the results of this check."
        else:
            res_str += "\n= Could not produce the stability plot because of "+\
                                                "the following error: %s"%str(e)
        return res_str
  
def output_timings(process, timings):
    """Present the result of a timings check in a nice format """
    
    # Define shortcut
    f = format_output
    loop_optimized_output = timings['loop_optimized_output']
    
    res_str = "%s \n"%process.nice_string()
    try:
        gen_total = timings['HELAS_MODEL_compilation']+\
                    timings['HelasDiagrams_generation']+\
                    timings['Process_output']+\
                    timings['Diagrams_generation']+\
                    timings['Process_compilation']+\
                    timings['Initialization']
    except TypeError:
        gen_total = None
    res_str += "\n= Generation time total...... ========== %s\n"%f(gen_total,'%.3gs')
    res_str += "|= Diagrams generation....... %s\n"\
                                       %f(timings['Diagrams_generation'],'%.3gs')
    res_str += "|= Helas Diagrams generation. %s\n"\
                                  %f(timings['HelasDiagrams_generation'],'%.3gs')
    res_str += "|= Process output............ %s\n"\
                                            %f(timings['Process_output'],'%.3gs')
    res_str += "|= HELAS+model compilation... %s\n"\
                                   %f(timings['HELAS_MODEL_compilation'],'%.3gs')
    res_str += "|= Process compilation....... %s\n"\
                                       %f(timings['Process_compilation'],'%.3gs')
    res_str += "|= Initialization............ %s\n"\
                                            %f(timings['Initialization'],'%.3gs')

    res_str += "\n= Helicity sum time / PSpoint ========== %.3gms\n"\
                                    %(timings['run_unpolarized_total']*1000.0)
    if loop_optimized_output:
        coef_time=timings['run_unpolarized_coefs']*1000.0
        loop_time=(timings['run_unpolarized_total']-\
                                        timings['run_unpolarized_coefs'])*1000.0
        total=coef_time+loop_time
        res_str += "|= Coefs. computation time... %.3gms (%d%%)\n"\
                                  %(coef_time,int(round(100.0*coef_time/total)))
        res_str += "|= Loop evaluation (OPP) time %.3gms (%d%%)\n"\
                                  %(loop_time,int(round(100.0*loop_time/total)))
    res_str += "\n= One helicity time / PSpoint ========== %.3gms\n"\
                                    %(timings['run_polarized_total']*1000.0)
    if loop_optimized_output:
        coef_time=timings['run_polarized_coefs']*1000.0
        loop_time=(timings['run_polarized_total']-\
                                        timings['run_polarized_coefs'])*1000.0
        total=coef_time+loop_time        
        res_str += "|= Coefs. computation time... %.3gms (%d%%)\n"\
                                  %(coef_time,int(round(100.0*coef_time/total)))
        res_str += "|= Loop evaluation (OPP) time %.3gms (%d%%)\n"\
                                  %(loop_time,int(round(100.0*loop_time/total)))
    res_str += "\n= Miscellaneous ========================\n"
    res_str += "|= Number of hel. computed... %s/%s\n"\
                %(f(timings['n_contrib_hel'],'%d'),f(timings['n_tot_hel'],'%d'))
    res_str += "|= Number of loop diagrams... %s\n"%f(timings['n_loops'],'%d')
    if loop_optimized_output:
        res_str += "|= Number of loop groups..... %s\n"\
                                               %f(timings['n_loop_groups'],'%d')
        res_str += "|= Number of loop wfs........ %s\n"\
                                                  %f(timings['n_loop_wfs'],'%d')
        if timings['loop_wfs_ranks']!=None:
            for i, r in enumerate(timings['loop_wfs_ranks']):
                res_str += "||= # of loop wfs of rank %d.. %d\n"%(i,r)
    res_str += "|= Loading time (Color data). ~%.3gms\n"\
                                               %(timings['Booting_time']*1000.0)
    res_str += "|= Maximum RAM usage (rss)... %s\n"\
                                  %f(float(timings['ram_usage']/1000.0),'%.3gMb')                                            
    res_str += "\n= Output disk size =====================\n"
    res_str += "|= Source directory sources.. %s\n"%f(timings['du_source'],'%sb')
    res_str += "|= Process sources........... %s\n"%f(timings['du_process'],'%sb')    
    res_str += "|= Color and helicity data... %s\n"%f(timings['du_color'],'%sb')
    res_str += "|= Executable size........... %s\n"%f(timings['du_exe'],'%sb')
    
    return res_str

def output_comparisons(comparison_results):
    """Present the results of a comparison in a nice list format
       mode short: return the number of fail process
    """    
    proc_col_size = 17
    pert_coupl = comparison_results[0]['process']['perturbation_couplings']
    if pert_coupl:
        process_header = "Process [virt="+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = len(process_header) + 1

    for proc in comparison_results:
        if len(proc['process'].base_string()) + 1 > proc_col_size:
            proc_col_size = len(proc['process'].base_string()) + 1

    col_size = 18

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length(process_header, proc_col_size) + \
              fixed_string_length("Min element", col_size) + \
              fixed_string_length("Max element", col_size) + \
              fixed_string_length("Relative diff.", col_size) + \
              "Result"

    for result in comparison_results:
        proc = result['process'].base_string()
        values = result['values']
        
        if len(values) <= 1:
            res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   "    * No permutations, process not checked *" 
            no_check_proc += 1
            no_check_proc_list.append(result['process'].nice_string())
            continue

        passed = result['passed']

        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % min(values), col_size) + \
                   fixed_string_length("%1.10e" % max(values), col_size) + \
                   fixed_string_length("%1.10e" % result['difference'],
                                       col_size)
        if passed:
            pass_proc += 1
            res_str += "Passed"
        else:
            fail_proc += 1
            failed_proc_list.append(result['process'].nice_string())
            res_str += "Failed"

    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc != 0:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)

    return res_str

def fixed_string_length(mystr, length):
    """Helper function to fix the length of a string by cutting it 
    or adding extra space."""
    
    if len(mystr) > length:
        return mystr[0:length]
    else:
        return mystr + " " * (length - len(mystr))
    

#===============================================================================
# check_gauge
#===============================================================================
def check_gauge(processes, param_card = None,cuttools="", tir={}, reuse = False, 
                         options=None, output_path=None, cmd = FakeInterface()):
    """Check gauge invariance of the processes by using the BRS check.
    For one of the massless external bosons (e.g. gluon or photon), 
    replace the polarization vector (epsilon_mu) with its momentum (p_mu)
    """
    cmass_scheme = cmd.options['complex_mass_scheme']
    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes

        model = multiprocess.get('model')        
        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,cmd= cmd,
                                           auth_skipping = True, reuse = False)
        else:
            evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir,
                                           cmd=cmd,model=model, param_card=param_card,
                                           auth_skipping = False, reuse = False,
                                           output_path=output_path)

        if not cmass_scheme and multiprocess.get('perturbation_couplings')==[]:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.
        results = run_multiprocs_no_crossings(check_gauge_process,
                                           multiprocess,
                                           evaluator,
                                           options=options
                                           )
        
        if multiprocess.get('perturbation_couplings')!=[] and not reuse:
            # Clean temporary folders created for the running of the loop processes
            clean_up(output_path)
        
        return results

    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    if processes[0].get('perturbation_couplings')==[]:
        evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = True, reuse = False, 
                                       cmd = cmd)
    else:
        evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir,
                                           model=model, param_card=param_card,
                                           auth_skipping = False, reuse = False,
                                           output_path=output_path, cmd = cmd)
    comparison_results = []
    comparison_explicit_flip = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Check if we already checked process
        #if check_already_checked([l.get('id') for l in process.get('legs') if \
        #                          not l.get('state')],
        ##                         [l.get('id') for l in process.get('legs') if \
        #                          l.get('state')],
        #                         sorted_ids, process, model):
        #    continue
        
        # Get process result
        result = check_gauge_process(process, evaluator,options=options)
        if result:
            comparison_results.append(result)

    if processes[0].get('perturbation_couplings')!=[] and not reuse:
        # Clean temporary folders created for the running of the loop processes
        clean_up(output_path)
            
    return comparison_results


def check_gauge_process(process, evaluator, options=None):
    """Check gauge invariance for the process, unless it is already done."""

    model = process.get('model')

    # Check that there are massless vector bosons in the process
    found_gauge = False
    for i, leg in enumerate(process.get('legs')):
        part = model.get_particle(leg.get('id'))
        if part.get('spin') == 3 and part.get('mass').lower() == 'zero':
            found_gauge = True
            break
    if not found_gauge:
        logger.info("No ward identity for %s" % \
                process.nice_string().replace('Process', 'process'))
        # This process can't be checked
        return None

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking ward identities for %s" % \
                process.nice_string().replace('Process', 'process'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        if process.get('perturbation_couplings')==[]:
            amplitude = diagram_generation.Amplitude(process)
        else:
            amplitude = loop_diagram_generation.LoopAmplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    # Generate the HelasMatrixElement for the process
    if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)
    else:
        matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                               optimized_output=evaluator.loop_optimized_output)

    #p, w_rambo = evaluator.get_momenta(process)

#    MLOptions = {'ImprovePS':True,'ForceMP':True}

#    brsvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = True,
#                                  output='jamp',MLOptions=MLOptions, options=options)

    brsvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = True,
                                                 output='jamp', options=options)

    if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)
          
    mvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = False,
                                               output='jamp', options=options)
    
    if mvalue and mvalue['m2']:
        return {'process':process,'value':mvalue,'brs':brsvalue}

def output_gauge(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format"""

    proc_col_size = 17
    
    pert_coupl = comparison_results[0]['process']['perturbation_couplings']
    
    # Of course, be more tolerant for loop processes
    if pert_coupl:
        threshold=1e-5
    else:
        threshold=1e-10
        
    if pert_coupl:
        process_header = "Process [virt="+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = len(process_header) + 1

    for one_comp in comparison_results:
        proc = one_comp['process'].base_string()
        mvalue = one_comp['value']
        brsvalue = one_comp['brs']
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    col_size = 18

    pass_proc = 0
    fail_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length(process_header, proc_col_size) + \
              fixed_string_length("matrix", col_size) + \
              fixed_string_length("BRS", col_size) + \
              fixed_string_length("ratio", col_size) + \
              "Result"

    for  one_comp in comparison_results:
        proc = one_comp['process'].base_string()
        mvalue = one_comp['value']
        brsvalue = one_comp['brs']
        ratio = (abs(brsvalue['m2'])/abs(mvalue['m2']))
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                    fixed_string_length("%1.10e" % mvalue['m2'], col_size)+ \
                    fixed_string_length("%1.10e" % brsvalue['m2'], col_size)+ \
                    fixed_string_length("%1.10e" % ratio, col_size)
         
        if ratio > threshold:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"
        else:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"

        #check all the JAMP
        # loop over jamp
        # This is not available for loop processes where the jamp list returned
        # is empty.
        if len(mvalue['jamp'])!=0:
            for k in range(len(mvalue['jamp'][0])):
                m_sum = 0
                brs_sum = 0
                # loop over helicity
                for j in range(len(mvalue['jamp'])):
                    #values for the different lorentz boost
                    m_sum += abs(mvalue['jamp'][j][k])**2
                    brs_sum += abs(brsvalue['jamp'][j][k])**2                                            
                        
                # Compare the different helicity  
                if not m_sum:
                    continue
                ratio = abs(brs_sum) / abs(m_sum)
    
                tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
                       fixed_string_length("%1.10e" % m_sum, col_size) + \
                       fixed_string_length("%1.10e" % brs_sum, col_size) + \
                       fixed_string_length("%1.10e" % ratio, col_size)        
                       
                if ratio > 1e-15:
                    if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                        fail_proc += 1
                        pass_proc -= 1
                        failed_proc_list.append(proc)
                    res_str += tmp_str + "Failed"
                elif not proc_succeed:
                     res_str += tmp_str + "Passed"


    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

    if output=='text':
        return res_str
    else:
        return fail_proc
#===============================================================================
# check_lorentz
#===============================================================================
def check_lorentz(processes, param_card = None,cuttools="", tir={}, options=None, \
                 reuse = False, output_path=None, cmd = FakeInterface()):
    """ Check if the square matrix element (sum over helicity) is lorentz 
        invariant by boosting the momenta with different value."""

    cmass_scheme = cmd.options['complex_mass_scheme']
    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes
        model = multiprocess.get('model')
        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model,
                                cmd= cmd, auth_skipping = False, reuse = True)
        else:
            evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir,
                     model=model, auth_skipping = False, reuse = True,
                                             output_path=output_path, cmd = cmd)

        if not cmass_scheme and processes.get('perturbation_couplings')==[]:
            # Set all widths to zero for lorentz check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[\
                                                     particle.get('width')] = 0.

        results = run_multiprocs_no_crossings(check_lorentz_process,
                                           multiprocess,
                                           evaluator,
                                           options=options)
        
        if multiprocess.get('perturbation_couplings')!=[] and not reuse:
            # Clean temporary folders created for the running of the loop processes
            clean_up(output_path)
        
        return results
        
    elif isinstance(processes, base_objects.Process):
        processes = base_objects.ProcessList([processes])
    elif isinstance(processes, base_objects.ProcessList):
        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    if processes[0].get('perturbation_couplings')==[]:
        evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = False, reuse = True, 
                                       cmd=cmd)
    else:
        evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools, tir_dir=tir,
                                               model=model,param_card=param_card,
                                           auth_skipping = False, reuse = True,
                                           output_path=output_path, cmd = cmd)

    comparison_results = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Check if we already checked process
        #if check_already_checked([l.get('id') for l in process.get('legs') if \
        #                          not l.get('state')],
        #                         [l.get('id') for l in process.get('legs') if \
        #                          l.get('state')],
        #                         sorted_ids, process, model):
        #    continue
        
        # Get process result
        result = check_lorentz_process(process, evaluator,options=options)
        if result:
            comparison_results.append(result)

    if processes[0].get('perturbation_couplings')!=[] and not reuse:
        # Clean temporary folders created for the running of the loop processes
        clean_up(output_path)

    return comparison_results


def check_lorentz_process(process, evaluator,options=None):
    """Check gauge invariance for the process, unless it is already done."""

    amp_results = []
    model = process.get('model')

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking lorentz transformations for %s" % \
                process.nice_string().replace('Process:', 'process'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        if process.get('perturbation_couplings')==[]:
            amplitude = diagram_generation.Amplitude(process)
        else:
            amplitude = loop_diagram_generation.LoopAmplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None

     # Generate the HelasMatrixElement for the process
    p, w_rambo = evaluator.get_momenta(process, options)

    # Generate the HelasMatrixElement for the process
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                               gen_color = True)
    else:
        matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                                       optimized_output = evaluator.loop_optimized_output)

    MLOptions = {'ImprovePS':True,'ForceMP':True}
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        data = evaluator.evaluate_matrix_element(matrix_element, p=p, output='jamp',
                                                 auth_skipping = True, options=options)
    else:
        data = evaluator.evaluate_matrix_element(matrix_element, p=p, output='jamp',
                auth_skipping = True, PS_name = 'original', MLOptions=MLOptions,
                                                              options = options)

    if data and data['m2']:
        if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
            results = [data]
        else:
            results = [('Original evaluation',data)]
    else:
        return  {'process':process, 'results':'pass'}

    # The boosts are not precise enough for the loop evaluations and one need the
    # fortran improve_ps function of MadLoop to work. So we only consider the
    # boosts along the z directions for loops or simple rotations.
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        for boost in range(1,4):
            boost_p = boost_momenta(p, boost)
            results.append(evaluator.evaluate_matrix_element(matrix_element,
                                                    p=boost_p,output='jamp'))
    else:
        # We only consider the rotations around the z axis so to have the
        boost_p = boost_momenta(p, 3)
        results.append(('Z-axis boost',
            evaluator.evaluate_matrix_element(matrix_element, options=options,
            p=boost_p, PS_name='zBoost', output='jamp',MLOptions = MLOptions)))
        # We add here also the boost along x and y for reference. In the output
        # of the check, it is now clearly stated that MadLoop improve_ps script
        # will not work for them. The momenta read from event file are not
        # precise enough so these x/yBoost checks are omitted.
        if not options['events']:
            boost_p = boost_momenta(p, 1)
            results.append(('X-axis boost',
                evaluator.evaluate_matrix_element(matrix_element, options=options,
                p=boost_p, PS_name='xBoost', output='jamp',MLOptions = MLOptions)))
            boost_p = boost_momenta(p, 2)
            results.append(('Y-axis boost',
                evaluator.evaluate_matrix_element(matrix_element,options=options,
                p=boost_p, PS_name='yBoost', output='jamp',MLOptions = MLOptions)))
        # We only consider the rotations around the z axis so to have the 
        # improve_ps fortran routine work.
        rot_p = [[pm[0],-pm[2],pm[1],pm[3]] for pm in p]
        results.append(('Z-axis pi/2 rotation',
            evaluator.evaluate_matrix_element(matrix_element,options=options,
            p=rot_p, PS_name='Rotation1', output='jamp',MLOptions = MLOptions)))
        # Now a pi/4 rotation around the z-axis
        sq2 = math.sqrt(2.0)
        rot_p = [[pm[0],(pm[1]-pm[2])/sq2,(pm[1]+pm[2])/sq2,pm[3]] for pm in p]
        results.append(('Z-axis pi/4 rotation',
            evaluator.evaluate_matrix_element(matrix_element,options=options,
            p=rot_p, PS_name='Rotation2', output='jamp',MLOptions = MLOptions)))
            
        
    return {'process': process, 'results': results}

#===============================================================================
# check_gauge
#===============================================================================
def check_unitary_feynman(processes_unit, processes_feynm, param_card=None, 
                               options=None, tir={}, output_path=None,
                               cuttools="", reuse=False, cmd = FakeInterface()):
    """Check gauge invariance of the processes by flipping
       the gauge of the model
    """
    
    mg_root = cmd._mgme_dir
    
    cmass_scheme = cmd.options['complex_mass_scheme']
    
    if isinstance(processes_unit, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess_unit = processes_unit
        model = multiprocess_unit.get('model')

        # Initialize matrix element evaluation
        # For the unitary gauge, open loops should not be used
        loop_optimized_bu = cmd.options['loop_optimized_output']
        if processes_unit.get('squared_orders'):
            if processes_unit.get('perturbation_couplings') in [[],['QCD']]:
                cmd.options['loop_optimized_output'] = True
            else:
                raise InvalidCmd("The gauge test cannot be performed for "+
                  " a process with more than QCD corrections and which"+
                  " specifies squared order constraints.")
        else:
            cmd.options['loop_optimized_output'] = False
            
        aloha.unitary_gauge = True
        if processes_unit.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,
                                       cmd=cmd,auth_skipping = False, reuse = True)
        else:
            evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir,
                                           cmd=cmd, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, 
                                           output_path=output_path,
                                           reuse = False)
        if not cmass_scheme and multiprocess_unit.get('perturbation_couplings')==[]:
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_u = run_multiprocs_no_crossings(get_value,
                                           multiprocess_unit,
                                           evaluator,
                                           options=options)
        
        clean_added_globals(ADDED_GLOBAL)
       # Clear up previous run if checking loop output
        if processes_unit.get('perturbation_couplings')!=[]:
            clean_up(output_path)

        momentum = {}
        for data in output_u:
            momentum[data['process']] = data['p']
        
        multiprocess_feynm = processes_feynm
        model = multiprocess_feynm.get('model')

        # Initialize matrix element evaluation
        aloha.unitary_gauge = False
        # We could use the default output as well for Feynman, but it provides
        # an additional check
        cmd.options['loop_optimized_output'] = True
        if processes_feynm.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,
                                       cmd= cmd, auth_skipping = False, reuse = False)
        else:
            evaluator = LoopMatrixElementEvaluator(cuttools_dir=cuttools,tir_dir=tir,
                                           cmd= cmd, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, 
                                           output_path=output_path,
                                           reuse = False)

        if not cmass_scheme and multiprocess_feynm.get('perturbation_couplings')==[]:
            # Set all widths to zero for gauge check
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_f = run_multiprocs_no_crossings(get_value, multiprocess_feynm,
                                                            evaluator, momentum,
                                                            options=options)  
        output = [processes_unit]        
        for data in output_f:
            local_dico = {}
            local_dico['process'] = data['process']
            local_dico['value_feynm'] = data['value']
            local_dico['value_unit'] = [d['value'] for d in output_u 
                                      if d['process'] == data['process']][0]
            output.append(local_dico)
        
        if processes_feynm.get('perturbation_couplings')!=[] and not reuse:
            # Clean temporary folders created for the running of the loop processes
            clean_up(output_path)

        # Reset the original global variable loop_optimized_output.
        cmd.options['loop_optimized_output'] = loop_optimized_bu

        return output
#    elif isinstance(processes, base_objects.Process):
#        processes = base_objects.ProcessList([processes])
#    elif isinstance(processes, base_objects.ProcessList):
#        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

#===============================================================================
# check_cms
#===============================================================================
def check_complex_mass_scheme(process_line, param_card=None, cuttools="",tir={}, 
         cmd = FakeInterface(), output_path=None, MLOptions = {}, options={}):
    """Check complex mass scheme consistency in the offshell region of s-channels
    detected for this process, by varying the expansion paramer consistently
    with the corresponding width and making sure that the difference between
    the complex mass-scheme and the narrow-width approximation is higher order.
    """

    if not isinstance(process_line, str):
        raise InvalidCmd("Proces definition must be given as a stirng for this check")

    # Generate a list of unique processes in the NWA scheme
    cmd.do_set('complex_mass_scheme False', log=False)
    #cmd.do_import('model loop_qcd_qed_sm-NWA')
    multiprocess_nwa = cmd.extract_process(process_line)

    # Change the option 'recompute_width' to the optimal value if set to 'auto'.
    has_FRdecay = os.path.isfile(pjoin(cmd._curr_model.get('modelpath'),
                                                                   'decays.py'))

    # Proceed with some warning
    missing_perturbations = cmd._curr_model.get_coupling_orders()-\
                             set(multiprocess_nwa.get('perturbation_couplings'))

    if len(multiprocess_nwa.get('perturbation_couplings'))>0 and \
                                                   len(missing_perturbations)>0:
        logger.warning("------------------------------------------------------")
        logger.warning("The process considered does not specify the following "+
           "type of loops to be included : %s"%str(list(missing_perturbations)))
        logger.warning("Consequently, the CMS check will be unsuccessful if the"+
         " process involves any resonating particle whose LO decay is "+
         "mediated by one of these orders.")
        logger.warning("You can use the syntax '[virt=all]' to automatically"+
                                 " include all loops supported by the model.")
        logger.warning("------------------------------------------------------")

    if len(multiprocess_nwa.get('perturbation_couplings'))>0 and \
                                            len(multiprocess_nwa.get('legs'))<=4:
        logger.warning("------------------------------------------------------")
        logger.warning("Processes with four or less external states are typically not"+\
          " sensitive to incorrect Complex Mass Scheme implementations.")
        logger.warning("You can test this sensitivity by making sure that the"+
          " same check on the leading-order counterpart of this process *fails*"+
          " when using the option '--diff_lambda_power=2'.")
        logger.warning("If it does not, then consider adding a massless "+
                                         "gauge vector to the external states.")
        logger.warning("------------------------------------------------------")

    if options['recompute_width']=='auto':
        if multiprocess_nwa.get('perturbation_couplings')!=[]:
            # NLO, so it is necessary to have the correct LO width for the check
            options['recompute_width'] = 'first_time'
        else:
            options['recompute_width'] = 'never'

    # Some warnings
    if options['recompute_width'] in ['first_time', 'always'] and \
                             not has_FRdecay and not 'cached_widths' in options:
        logger.info('The LO widths will need to be recomputed but the '+
         'model considered does not appear to have a decay module.\nThe widths'+
         ' will need to be computed numerically and it will slow down the test.\n'+
         'Consider using a param_card already specifying correct LO widths and'+
         " adding the option --recompute_width=never when doing this check.")

    if options['recompute_width']=='never' and \
        any(order in multiprocess_nwa.get('perturbation_couplings') for order in
                                                   options['expansion_orders']):
        logger.warning('You chose not to recompute the widths while including'+
          ' loop corrections. The check will be successful only if the width'+\
          ' specified in the default param_card is LO accurate (Remember that'+\
          ' the default values of alpha_s and awem1 are set to 0.1 and 10.0'+\
          ' respectively by default).')

    # Reload the model including the decay.py to have efficient MadWidth if
    # possible (this model will be directly given to MadWidth. Notice that 
    # this will not be needed for the CMS run because MadWidth is not supposed
    # to be used there (the widths should be recycled from those of the NWA run).
    if options['recompute_width'] in ['first_time', 'always'] and has_FRdecay:
        modelname = cmd._curr_model.get('modelpath+restriction')
        with misc.MuteLogger(['madgraph'], ['INFO']):
            model = import_ufo.import_model(modelname, decay=True,
                                                      complex_mass_scheme=False)
        multiprocess_nwa.set('model', model)

    run_options = copy.deepcopy(options)

    # Set the seed if chosen by user
    if options['seed'] > 0:
        random.seed(options['seed'])
    
    # Add useful entries
    run_options['param_card'] = param_card
    if isinstance(cmd, FakeInterface):
        raise MadGraph5Error, "Check CMS cannot be run with a FakeInterface."
    run_options['cmd']        = cmd
    run_options['MLOptions']  = MLOptions
    if output_path:
        run_options['output_path'] = output_path
    else:
        run_options['output_path'] = cmd._mgme_dir
    
    # Add the information regarding FR decay for optimal log information
    run_options['has_FRdecay']     = has_FRdecay

    # And one for caching the widths computed along the way
    if 'cached_widths' not in run_options:
        run_options['cached_widths'] = {}
    # Cached param_cards, first is param_card instance, second is
    # param_name dictionary
    run_options['cached_param_card'] = {'NWA':[None,None],'CMS':[None,None]}

    if options['tweak']['name']:
        logger.info("Now running the CMS check for tweak '%s'"\
                                                      %options['tweak']['name'])

    model = multiprocess_nwa.get('model')
    # Make sure all masses are defined as external
    for particle in model.get('particles'):
        mass_param = model.get_parameter(particle.get('mass'))
        if particle.get('mass')!='ZERO' and 'external' not in mass_param.depend:
            if model.get('name') not in ['sm','loop_sm']:
                logger.warning("The mass '%s' of particle '%s' is not an external"%\
              (model.get_parameter(particle.get('mass')).name,particle.get('name'))+\
              " parameter as required by this check. \nMG5_aMC will try to"+\
              " modify the model to remedy the situation. No guarantee.")
            status = model.change_electroweak_mode(set(['mz','mw','alpha']))
            if not status:
                raise InvalidCmd('The EW scheme could apparently not be changed'+\
                  ' so as to have the W-boson mass external. The check cannot'+\
                  ' proceed.')
            break

    veto_orders = [order for order in model.get('coupling_orders') if \
                                       order not in options['expansion_orders']]
    if len(veto_orders)>0:
        logger.warning('You did not define any parameter scaling rule for the'+\
          " coupling orders %s. They will be "%','.join(veto_orders))+\
          "forced to zero in the tests. Consider adding the scaling rule to"+\
          "avoid this. (see option '--cms' in 'help check')"
        for order in veto_orders:
            multiprocess_nwa.get('orders')[order]==0
        multiprocess_nwa.set('perturbation_couplings', [order for order in
        multiprocess_nwa['perturbation_couplings'] if order not in veto_orders])

    if multiprocess_nwa.get('perturbation_couplings')==[]:
        evaluator = MatrixElementEvaluator(model, param_card,
                                   cmd=cmd,auth_skipping = False, reuse = True)
    else:
        evaluator = LoopMatrixElementTimer(cuttools_dir=cuttools,tir_dir=tir,
                                           cmd=cmd, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, 
                                           output_path=output_path,
                                           reuse = False)

    cached_information = []
    output_nwa = run_multiprocs_no_crossings(check_complex_mass_scheme_process,
                                           multiprocess_nwa,
                                           evaluator,
    # This empty list 'opt' will be passed to the check_complex_mass_scheme_process
    # function which will fill it with the specification of the particle for which
    # the the complex mass scheme must be checked. The fact that it is a list
    # at this stage tells the function check_complex_mass_scheme_process that
    # we are doing nwa. It will then be converted to a dictionary when doing cms.
                                           opt = cached_information,
                                           options=run_options)

    # Make sure to start from fresh for LO runs
    clean_added_globals(ADDED_GLOBAL)

    # Generate a list of unique processes in the CMS scheme
    cmd.do_set('complex_mass_scheme True', log=False)
    #cmd.do_import('model loop_qcd_qed_sm__CMS__-CMS')

    multiprocess_cms = cmd.extract_process(process_line)    
    model = multiprocess_cms.get('model')
    # Apply veto
    if len(veto_orders)>0:
        for order in veto_orders:
            multiprocess_cms.get('orders')[order]==0
        multiprocess_cms.set('perturbation_couplings', [order for order in
        multiprocess_cms['perturbation_couplings'] if order not in veto_orders])
        
    if multiprocess_cms.get('perturbation_couplings')==[]:
        evaluator = MatrixElementEvaluator(model, param_card,
                                    cmd=cmd,auth_skipping = False, reuse = True)
    else:
        evaluator = LoopMatrixElementTimer(cuttools_dir=cuttools,tir_dir=tir,
                                           cmd=cmd, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, 
                                           output_path=output_path,
                                           reuse = False)

    output_cms = run_multiprocs_no_crossings(check_complex_mass_scheme_process,
                                   multiprocess_cms,
                                   evaluator,
                                   # We now substituted the cached information
                                   opt = dict(cached_information),
                                   options=run_options)

    if multiprocess_cms.get('perturbation_couplings')!=[] and not options['reuse']:
        # Clean temporary folders created for the running of the loop processes
        clean_up(output_path)

    # Now reformat a bit the output by putting the CMS and NWA results together
    # as values of a dictionary with the process name as key. 
    # Also a 'processes_order' to list all processes in their order of appearance
    result = {'ordered_processes':[],'lambdaCMS':options['lambdaCMS']}
    # Recall what perturbation orders were used
    result['perturbation_orders']=multiprocess_nwa.get('perturbation_couplings')
    for i, proc_res in enumerate(output_nwa):
        result['ordered_processes'].append(proc_res[0])
        result[proc_res[0]] = {
                'NWA':proc_res[1]['resonances_result'],
                'CMS':output_cms[i][1]['resonances_result'],
                'born_order':proc_res[1]['born_order'],
                'loop_order':proc_res[1]['loop_order']}
    
    # As an optimization we propagate the widths as they could be reused when
    # using several tweaks
    options['cached_widths'] = run_options['cached_widths']
    
    # Add widths information to the result
    result['recompute_width'] = options['recompute_width']
    result['has_FRdecay']     = has_FRdecay
    result['widths_computed'] = []
    cached_widths = sorted(options['cached_widths'].items(), key=lambda el: \
                                                                  abs(el[0][0]))
    for (pdg, lambda_value), width in cached_widths:
        if lambda_value != 1.0:
            continue
        result['widths_computed'].append((model.get_particle(pdg).get_name(),
                                                                         width))
        
    # Make sure to clear the python ME definitions generated in LO runs
    clean_added_globals(ADDED_GLOBAL)
    
    return result


# Check CMS for a given process
def check_complex_mass_scheme_process(process, evaluator, opt = [], 
                                                                  options=None):
    """Check CMS for the process in argument. The options 'opt' is quite important.
    When opt is a list, it means that we are doing NWA and we are filling the
    list with the following tuple 
                       ('proc_name',({'ParticlePDG':ParticlePDG,
                                      'FinalStateMothersNumbers':set([]), 
                                      'PS_point_used':[]},...))
    When opt is a dictionary, we are in the CMS mode and it will be reused then.
    """

    # a useful logical to check if we are in LO (python on the flight) or 
    # NLO (output and compilation) mode
    NLO = process.get('perturbation_couplings') != []
    
    def glue_momenta(production, decay):
        """ Merge together the kinematics for the production of particle 
        positioned last in the 'production' array with the 1>N 'decay' kinematic' 
        provided where the decay particle is first."""
        
        from MadSpin.decay import momentum
        
        full = production[:-1]
        
        # Consistency check:
        # target  =  production[decay_number-1]
        # boosted = momentum(decay[0][0],decay[0][1],decay[0][2],decay[0][3])
        # print 'Consistency check ',target==boosted
        for p in decay[1:]:
            bp = momentum(*p).boost(momentum(*production[-1]))
            full.append([bp.E,bp.px,bp.py,bp.pz])
        
        return full
    
    def find_resonances(diagrams):
        """ Find all the resonances in the matrix element in argument """
        
        model = process['model']
        resonances_found = []

        for ll, diag in enumerate(diagrams):
            for amp in diag.get('amplitudes'):
                # 0 specifies the PDG given to the fake s-channels from 
                # vertices with more than four legs
                s_channels, t_channels = amp.\
                 get_s_and_t_channels(process.get_ninitial(), model, 0)
                # The s-channel are given from the outmost ones going inward as
                # vertices, so we must replace parent legs with the outermost ones
                replacement_dict = {}
                for s_channel in s_channels:
                    new_resonance = {
                        'ParticlePDG':s_channel.get('legs')[-1].get('id'),
                        'FSMothersNumbers':[],
                        'PS_point_used':[]}
                    for leg in s_channel.get('legs')[:-1]:
                        if leg.get('number')>0:
                            new_resonance['FSMothersNumbers'].append(
                                                            leg.get('number'))
                        else:
                            try:
                                new_resonance['FSMothersNumbers'].extend(
                                            replacement_dict[leg.get('number')])
                            except KeyError:
                                raise Exception, 'The following diagram '+\
                                              'is malformed:'+diag.nice_string()
                                               
                    replacement_dict[s_channel.get('legs')[-1].get('number')] = \
                                               new_resonance['FSMothersNumbers']
                    new_resonance['FSMothersNumbers'] = set(
                                              new_resonance['FSMothersNumbers'])
                    if new_resonance not in resonances_found:
                        resonances_found.append(new_resonance)

        # Now we setup the phase-space point for each resonance found
        kept_resonances = []
        for resonance in resonances_found:
            # Discard fake s-channels
            if resonance['ParticlePDG'] == 0:
                continue

            # Discard if the particle appears in the final state
            if abs(resonance['ParticlePDG']) in \
                                [abs(l.get('id')) for l in process.get('legs')]:
                continue

            mass_string = evaluator.full_model.get_particle(
                                           resonance['ParticlePDG']).get('mass')
            mass  = evaluator.full_model.get('parameter_dict')[mass_string].real
            # Discard massless s-channels
            if mass==0.0:
                continue
            
            width_string = evaluator.full_model.get_particle(
                                           resonance['ParticlePDG']).get('width')
            width  = evaluator.full_model.get('parameter_dict')[width_string].real

            # Discard stable s-channels
            if width==0.0:
                continue

            final_state_energy = sum(
                evaluator.full_model.get('parameter_dict')[
                evaluator.full_model.get_particle(l.get('id')).get('mass')].real
                for l in process.get('legs') if l.get('number') in 
                                                  resonance['FSMothersNumbers'])
            
            # Choose the offshellness
            special_mass = (1.0 + options['offshellness'])*mass
            
            # Discard impossible kinematics
            if special_mass<final_state_energy:
                raise InvalidCmd('The offshellness specified (%s) is such'\
                  %options['offshellness']+' that the resulting kinematic is '+\
                  'impossible for resonance %s %s.'%(evaluator.full_model.
                           get_particle(resonance['ParticlePDG']).get_name(),
                                      str(list(resonance['FSMothersNumbers']))))
                continue
            
            # Add it to the list of accepted resonances
            kept_resonances.append(resonance)
            
        for resonance in kept_resonances:
            # Chose the PS point for the resonance
            set_PSpoint(resonance, force_other_res_offshell=kept_resonances)

#        misc.sprint(kept_resonances)
#        misc.sprint(len(kept_resonances))
        return tuple(kept_resonances)

    def set_PSpoint(resonance, force_other_res_offshell=[], 
                                allow_energy_increase=1.5, isolation_cuts=True):
        """ Starting from the specified resonance, construct a phase space point
        for it and possibly also enforce other resonances to be onshell. Possibly
        allow to progressively increase enregy by steps of the integer specified
        (negative float to forbid it) and possible enforce default isolation cuts
        as well."""
        
        def invmass(momenta):
            """ Computes the invariant mass of a list of momenta."""
            ptot = [sum(p[i] for p in momenta) for i in range(4)]
            return math.sqrt(ptot[0]**2-ptot[1]**2-ptot[2]**2-ptot[3]**2)
        
        model = evaluator.full_model
        def getmass(pdg):
            """ Returns the mass of a particle given the current model and its
            pdg given in argument."""
            return model.get('parameter_dict')[
                                       model.get_particle(pdg).get('mass')].real

        N_trials                  = 0
        max_trial                 = 1e4
        nstep_for_energy_increase = 1e3
        PS_point_found            = None
        if options['offshellness'] > 0.0:
            offshellness              = options['offshellness']
        else:
            # We must undershoot the offshellness since one needs more 
            # energy than the target mass to have a valid PS point. So we
            # start with an offshellness 4 times larger, and progressively reduce
            # it later
            offshellness = (0.25*(options['offshellness']+1.0))-1.0
        
        # When offshellness is negative, it is progressively decreased every
        # nstep_for_energy_increase attempts (not increased!), so it is more
        # dangerous, and we therefore want the steps to be smaller
        if options['offshellness'] < 0.0:
            energy_increase = math.sqrt(allow_energy_increase)
        else:
            energy_increase = allow_energy_increase
        # Make sure to remove the resonance itself from force_other_res_offshell
        other_res_offshell = [res for res in force_other_res_offshell if 
                                                                 res!=resonance]

        # Now play it smart on finding starting energy and offshellness and 
        # register all resonance masses
        all_other_res_masses = [getmass(res['ParticlePDG'])
                                                  for res in other_res_offshell]
        resonance_mass = getmass(resonance['ParticlePDG'])

        str_res = '%s %s'%(model.get_particle(
                            resonance['ParticlePDG']).get_name(),
                                       str(list(resonance['FSMothersNumbers'])))
        leg_number_to_leg = dict((l.get('number'),l) for l in process.get('legs'))
        # Find what is the minimum possible offshellness given
        # the mass of the daughters of this resonance.
        # This will only be relevant when options['offshellness'] is negative
        daughter_masses = sum(getmass(leg_number_to_leg[\
                 number].get('id')) for number in resonance['FSMothersNumbers'])
        min_offshellnes = 4.0*((daughter_masses*1.2)/resonance_mass)-1.0

        # Compute the minimal energy given the external states, add 20% to leave
        # enough phase-space
        min_energy = max(sum(getmass(l.get('id')) for l in \
                                  process.get('legs') if l.get('state')==True),
                         sum(getmass(l.get('id')) for l in \
                                  process.get('legs') if l.get('state')==False))
        
        # List all other offshellnesses of the potential daughters of this 
        # resonance
        daughter_offshellnesses = [(1.0+options['offshellness'])*mass 
              for i, mass in enumerate(all_other_res_masses) if 
                      other_res_offshell[i]['FSMothersNumbers'].issubset(
                                                 resonance['FSMothersNumbers'])]
        
        if options['offshellness'] >= 0.0:
            
            if len(daughter_offshellnesses)>0:
                max_mass = max(daughter_offshellnesses)
                # A factor two to have enough phase-space
                offshellness = max(2.0*(max_mass/resonance_mass)-1.0,
                                                        options['offshellness'])
            
            max_mass = max([(1.0+options['offshellness'])*mass for mass in \
                  all_other_res_masses]+[(1.0+offshellness)*resonance_mass])
            
            # Account for external_masses too
            # A factor two to have enough phase-space open
            target = max(min_energy*1.2,max_mass*2.0)
            if target > options['energy']:
                logger.warning("The user-defined energy %f seems "%options['energy']+
              " insufficient to reach the minimum propagator invariant mass "+
              "%f required for the chosen offshellness %f."%(max_mass,
               options['offshellness']) + " Energy reset to %f."%target)
                options['energy'] = target
                
        else:
            if len(daughter_offshellnesses) > 0:
                min_mass = min(daughter_offshellnesses)
                # A factor one half to have enough phase-space
                offshellness = min(0.25*(min_mass/resonance_mass)-1.0,
                                                        options['offshellness'])
            
            # Make sure the chosen offshellness leaves enough energy to produce
            # the daughter masses
            if (1.0+offshellness)*resonance_mass < daughter_masses*1.2:
                msg = 'The resonance %s cannot accomodate'%str_res+\
              ' an offshellness of %f because the daughter'%options['offshellness']+\
              ' masses are %f.'%daughter_masses
                if options['offshellness']<min_offshellnes:
                    msg += ' Try again with an offshellness'+\
                  ' smaller (in absolute value) of at least %f.'%min_offshellnes
                else:
                  msg += ' Try again with a smalled offshellness (in absolute value).'
                raise InvalidCmd(msg)
            
            min_mass = min([(1.0+options['offshellness'])*mass for mass in \
                      all_other_res_masses]+[(1.0+offshellness)*resonance_mass])
            # Account for external_masses too
            # A factor two to have enough phase-space open
            if 2.0*min_mass < options['energy']:
                new_energy = max(min_energy*1.2, 2.0*min_mass)
                logger.warning("The user-defined energy %f seems "%options['energy']+
                  " too large to not overshoot the maximum propagator invariant mass "+
                  "%f required for the chosen offshellness %f."%(min_mass,
                   options['offshellness']) + " Energy reset to %f."%new_energy)
                options['energy'] = new_energy  
        
        if options['offshellness'] < 0.0 and options['energy'] >= min_mass:
            logger.debug("The target energy is not compatible with the mass"+
              " of the external states for this process (%f). It is "%min_mass+
                 "unlikely that a valid kinematic configuration will be found.")
        
        if options['offshellness']<0.0 and offshellness<options['offshellness'] or \
           options['offshellness']>0.0 and offshellness>options['offshellness']:
            logger.debug("Offshellness increased to %f"%offshellness+
                " so as to try to find a kinematical configuration with"+
                " offshellness at least equal to %f"%options['offshellness']+
                                                         " for all resonances.")

        start_energy = options['energy']        
        while N_trials<max_trial:
            N_trials += 1
            if N_trials%nstep_for_energy_increase==0:
                if allow_energy_increase > 0.0:
                    old_offshellness = offshellness
                    if offshellness > 0.0:
                        options['energy'] *= energy_increase
                        offshellness      *= energy_increase
                    else:
                        options['energy'] = max(options['energy']/energy_increase, 
                                                                 min_energy*1.2)
                        offshellness = max(min_offshellnes,
                                       ((offshellness+1.0)/energy_increase)-1.0)
                    if old_offshellness!=offshellness:
                        logger.debug('Trying to find a valid kinematic'+\
                           " configuration for resonance '%s'"%str_res+\
                             ' with increased offshellness %f'%offshellness)

            candidate = get_PSpoint_for_resonance(resonance, offshellness)
            pass_offshell_test = True
            for i, res in enumerate(other_res_offshell):
                # Make sure other resonances are sufficiently offshell too
                if offshellness > 0.0:
                    if invmass([candidate[j-1] for j in res['FSMothersNumbers']]) <\
                        ((1.0+options['offshellness'])*all_other_res_masses[i]):
                        pass_offshell_test = False
                        break
                else:
                    if invmass([candidate[j-1] for j in res['FSMothersNumbers']]) >\
                        ((1.0+options['offshellness'])*all_other_res_masses[i]):
                        pass_offshell_test = False
                        break
            if not pass_offshell_test:
                continue
            # Make sure it is isolated
            if isolation_cuts:
                # Set ptcut to 5% of total energy
                if not evaluator.pass_isolation_cuts(candidate,
                      ptcut=0.05*invmass([candidate[0],candidate[1]]), drcut=0.4):
                    continue
            PS_point_found = candidate
            break
        
        # Restore the initial energy setup
        options['energy'] = start_energy

        if PS_point_found is None:
            err_msg = 'Could not find a valid PS point in %d'%max_trial+\
                ' trials. Try increasing the energy, modify the offshellness '+\
                'or relax some constraints.'
            if options['offshellness']<0.0:
                err_msg +='Try with a positive offshellness instead (or a '+\
                                       'negative one of smaller absolute value)'
            raise InvalidCmd, err_msg
        else:
#            misc.sprint('PS point found in %s trials.'%N_trials)
#            misc.sprint(PS_point_found)
            resonance['offshellnesses'] = []
            all_other_res_masses = [resonance_mass] + all_other_res_masses
            other_res_offshell   = [resonance] + other_res_offshell
            for i, res in enumerate(other_res_offshell):
                if i==0:
                    res_str = 'self'
                else:
                    res_str = '%s %s'%(model.get_particle(
                                         res['ParticlePDG']).get_name(),
                                             str(list(res['FSMothersNumbers'])))
                resonance['offshellnesses'].append((res_str,(
                    (invmass([PS_point_found[j-1] for j in 
                       res['FSMothersNumbers']])/all_other_res_masses[i])-1.0)))

            resonance['PS_point_used'] = PS_point_found
        
    def get_PSpoint_for_resonance(resonance, offshellness = options['offshellness']):
        """ Assigns a kinematic configuration to the resonance dictionary 
        given in argument."""            

        # Get the particle mass
        mass_string = evaluator.full_model.get_particle(
                                       resonance['ParticlePDG']).get('mass')
        mass  = evaluator.full_model.get('parameter_dict')[mass_string].real
        
        # Choose the offshellness
        special_mass = (1.0 + offshellness)*mass
        
        # Create a fake production and decay process
        prod_proc = base_objects.Process({'legs':base_objects.LegList(
             copy.copy(leg) for leg in process.get('legs') if 
                   leg.get('number') not in resonance['FSMothersNumbers'])})
        # Add the resonant particle as a final state
        # ID set to 0 since its mass will be forced
        # Number set so as to be first in the list in get_momenta
        prod_proc.get('legs').append(base_objects.Leg({
                'number':max(l.get('number') for l in process.get('legs'))+1,
                'state':True,
                'id':0}))
        # now the decay process
        decay_proc = base_objects.Process({'legs':base_objects.LegList(
           copy.copy(leg) for leg in process.get('legs') if leg.get('number') 
             in resonance['FSMothersNumbers'] and not leg.get('state')==False)})
        # Add the resonant particle as an initial state
        # ID set to 0 since its mass will be forced
        # Number set to -1 as well so as to be sure it appears first in 
        # get_momenta
        decay_proc.get('legs').insert(0,base_objects.Leg({
                'number':-1,
                'state':False,
                'id':0}))
        prod_kinematic = evaluator.get_momenta(prod_proc, options=options,
                                               special_mass=special_mass)[0]
        decay_kinematic = evaluator.get_momenta(decay_proc, options=options, 
                                               special_mass=special_mass)[0]
        momenta = glue_momenta(prod_kinematic,decay_kinematic)
        # Reshuffle the momentum so as to put it back in the order specified
        # in the process definition.
        # First the production momenta, without the special decayed particle
        ordered_momenta = [(prod_proc.get('legs')[i].get('number'),momenta[i])
                for i in range(len(prod_proc.get('legs'))-1)]
        # And then the decay ones.
        ordered_momenta += [(decay_proc.get('legs')[-i].get('number'),
                 momenta[-i]) for i in range(1,len(decay_proc.get('legs')))]

        # Return the PSpoint found in the right order
        return [m[1] for m in sorted(ordered_momenta, key = lambda el: el[0])]
        
        # misc.sprint(resonance['PS_point_used'])        
    
    @misc.mute_logger()
    def get_width(PDG, lambdaCMS, param_card):
        """ Returns the width to use for particle with absolute PDG 'PDG' and
        for the the lambdaCMS value 'lambdaCMS' using the cache if possible."""

        # If an unstable particle is in the external state, then set its width
        # to zero and don't cache the result of course.
        if abs(PDG) in [abs(leg.get('id')) for leg in process.get('legs')]:
            return 0.0

        particle = evaluator.full_model.get_particle(PDG)
        
        # If it is a goldstone or a ghost, return zero as its width should anyway
        # not be independent.
        if particle.get('ghost') or particle.get('goldstone'):
            return 0.0

        # If its width is analytically set to zero, then return zero right away
        if particle.get('width')=='ZERO':
            return 0.0
    
        if (PDG,lambdaCMS) in options['cached_widths']:
            return options['cached_widths'][(PDG,lambdaCMS)]

        if options['recompute_width'] == 'never':
            width = evaluator.full_model.\
                               get('parameter_dict')[particle.get('width')].real
        else:
            # Crash if we are doing CMS and the width was not found and recycled above
            if aloha.complex_mass:
                raise MadGraph5Error, "The width for particle with PDG %d and"%PDG+\
                  " lambdaCMS=%f should have already been "%lambdaCMS+\
                  "computed during the NWA run."

        # Use MadWith
        if options['recompute_width'] in ['always','first_time']:
            particle_name = particle.get_name()
            with misc.TMP_directory(dir=options['output_path']) as path:
                param_card.write(pjoin(path,'tmp.dat'))
                # 2-body decay is the maximum that should be considered for NLO check.
                # The default 1% accuracy is not enough when pushing to small
                # lambdaCMS values, we need 1 per mil at least.
                command = '%s --output=%s'%(particle_name,pjoin(path,'tmp.dat'))+\
                    ' --path=%s --body_decay=2'%pjoin(path,'tmp.dat')+\
                    ' --precision_channel=0.001'
#                misc.sprint(command)
                param_card.write(pjoin(options['output_path'],'tmp.dat'))
                # The MG5 command get_width will change the cmd._curr_model
                # and the cmd._curr_fortran_model which what we specified, so 
                # we must make sure to restore them after it finishes
                orig_model = options['cmd']._curr_model
                orig_fortran_model = options['cmd']._curr_fortran_model
                options['cmd'].do_compute_widths(command, evaluator.full_model)
                # Restore the models
                options['cmd']._curr_model = orig_model
                options['cmd']._curr_fortran_model = orig_fortran_model
                # Restore the width of the model passed in argument since
                # MadWidth will automatically update the width
                evaluator.full_model.set_parameters_and_couplings(
                                                          param_card=param_card)
                try:
                    tmp_param_card = check_param_card.ParamCard(pjoin(path,'tmp.dat'))
                except:
                    raise MadGraph5Error, 'Error occured during width '+\
                       'computation with command:\n   compute_widths %s'%command                   
                width = tmp_param_card['decay'].get(PDG).value
#                misc.sprint('lambdaCMS checked is', lambdaCMS,
#                                                   'for particle',particle_name)
#                misc.sprint('Width obtained :', width)
#                if lambdaCMS != 1.0:
#                    misc.sprint('Naively expected (lin. scaling) :',
#                                  options['cached_widths'][(PDG,1.0)]*lambdaCMS)
                
        if options['recompute_width'] in ['never','first_time']:
            # Assume linear scaling of the width
            for lam in options['lambdaCMS']:
                options['cached_widths'][(PDG,lam)]=width*(lam/lambdaCMS)
        else:
            options['cached_widths'][(PDG,lambdaCMS)] = width
        
        return options['cached_widths'][(PDG,lambdaCMS)]
            
    def get_order(diagrams, diagsName):
        """Compute the common summed of coupling orders used for this cms check
         in the diagrams specified. When inconsistency occurs, use orderName
         in the warning message if throwm."""
         
        orders = set([])
        for diag in diagrams:
            diag_orders = diag.calculate_orders()
            orders.add(sum((diag_orders[order] if order in diag_orders else 0)
                                      for order in options['expansion_orders']))
            if len(orders)>1:
                logger.warning(msg%('%s '%diagsName,str(orders)))
                return min(list(orders))
            else:
                return list(orders)[0]
         
    MLoptions = copy.copy(options['MLOptions'])
    # Make sure double-check helicities is set to False
    MLoptions['DoubleCheckHelicityFilter'] = False
    
    # Apply the seed tweak if present
    for tweak in options['tweak']['custom']:
        if tweak.startswith('seed'):
            try:
                new_seed = int(tweak[4:])
            except ValueError:
                raise MadGraph5Error, "Seed '%s' is not of the right format 'seed<int>'."%tweak
            random.seed(new_seed)
                
    mode = 'CMS' if aloha.complex_mass else 'NWA'
    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Running CMS check for process %s  (now doing %s scheme)" % \
                  ( process.nice_string().replace('Process:', 'process'), mode))
    
    proc_dir = None
    resonances = None
    warning_msg = "All %sdiagrams do not share the same sum of orders "+\
            "%s; found %%s."%(','.join(options['expansion_orders']))+\
            " This potentially problematic for the CMS check."
    if NLO:
        # We must first create the matrix element, export it and set it up.
        # If the reuse option is specified, it will be recycled.
        
        if options['name']=='auto':
            proc_name = "%s%s_%s%s__%s__"%(('SAVED' if options['reuse'] else ''),
         temp_dir_prefix, '_'.join(process.shell_string().split('_')[1:]), 
         ('_' if process.get('perturbation_couplings') else '')+
         '_'.join(process.get('perturbation_couplings')),mode)
        else:
            proc_name = "%s%s_%s__%s__"%(('SAVED' if options['reuse'] else ''),
                                          temp_dir_prefix,options['name'], mode)
        # Generate the ME
        timing, matrix_element = generate_loop_matrix_element(process, 
                options['reuse'], output_path=options['output_path'], 
                           cmd = options['cmd'], proc_name=proc_name, 
                                             loop_filter=options['loop_filter'])
        if matrix_element is None:
            # No diagrams for this process
            return None

        reusing = isinstance(matrix_element, base_objects.Process)
        proc_dir = pjoin(options['output_path'],proc_name)

        # Export the ME
        infos = evaluator.setup_process(matrix_element, proc_dir, 
                        reusing = reusing, param_card = options['param_card'], 
                                                            MLOptions=MLoptions)
        # Make sure the right MLoptions are set
        evaluator.fix_MadLoopParamCard(pjoin(proc_dir,'Cards'),
                              mp = None, loop_filter = True,MLOptions=MLoptions)
        
        # Make sure to start from fresh if previous run was stopped
        tmp_card_backup = pjoin(proc_dir,'Cards','param_card.dat__TemporaryBackup__')
        if os.path.isfile(tmp_card_backup):
            # Run was stopped mid-way, we must then restore the original card
            logger.info("Last run in process '%s' apparently aborted."%proc_dir+\
                           " Now reverting 'param_card.dat' to its original value.")
            shutil.copy(tmp_card_backup, pjoin(proc_dir, 'Cards','param_card.dat'))
        else:
            # Create a temporary backup which will be cleaned if the run ends properly
            shutil.copy(pjoin(proc_dir,'Cards','param_card.dat'), tmp_card_backup)
        # Now do the same with model_functions.f
        tmp_modelfunc_backup = pjoin(proc_dir,'Source','MODEL',
                                         'model_functions.f__TemporaryBackup__')
        if os.path.isfile(tmp_modelfunc_backup):
            # Run was stopped mid-way, we must then restore the model functions
            logger.info("Last run in process '%s' apparently aborted."%proc_dir+\
                    " Now reverting 'model_functions.f' to its original value.")
            shutil.copy(tmp_modelfunc_backup, pjoin(proc_dir,'Source','MODEL',
                                                           'model_functions.f'))
            evaluator.apply_log_tweak(proc_dir, 'recompile')
        else:
            # Create a temporary backup which will be cleaned if the run ends properly
            shutil.copy(pjoin(proc_dir,'Source','MODEL','model_functions.f'),
                                                           tmp_modelfunc_backup)

        # Make sure to setup correctly the helicity
        MadLoopInitializer.fix_PSPoint_in_check(pjoin(proc_dir,'SubProcesses'),
              read_ps = True, npoints = 1, hel_config = options['helicity'], 
                                           split_orders=options['split_orders'])
   
        # And recompile while making sure to recreate the executable and 
        # modified sources
        for dir in misc.glob('P*_*', pjoin(proc_dir,'SubProcesses')):
            if not (re.search(r'.*P\d+_\w*$', dir) or not os.path.isdir(dir)):
                continue
            try:
                os.remove(pjoin(dir,'check'))
                os.remove(pjoin(dir,'check_sa.o'))
            except OSError:
                pass
            # Now run make
            with open(os.devnull, 'w') as devnull:
                retcode = subprocess.call(['make','check'],
                                   cwd=dir, stdout=devnull, stderr=devnull)                     
            if retcode != 0:
                raise MadGraph5Error, "Compilation error with "+\
                                                        "'make check' in %s"%dir

        # Now find all the resonances of the ME, if not saved from a previous run
        pkl_path = pjoin(proc_dir,'resonance_specs.pkl')
        if reusing:
            # We recover the information from the pickle dumped during the
            # original run
            if not os.path.isfile(pkl_path):
                raise InvalidCmd('The folder %s could'%proc_dir+\
                 " not be reused because the resonance specification file "+
                                            "'resonance_specs.pkl' is missing.")
            else:
                proc_name, born_order, loop_order, resonances = \
                                       save_load_object.load_from_file(pkl_path)
                # Make sure to rederive the phase-space point since parameters
                # such as masses, seed, offshellness could have affected it
                for res in resonances:
                    set_PSpoint(res, force_other_res_offshell=resonances)

            # Second run (CMS), we can reuse the information if it is a dictionary
            if isinstance(opt, list):
                opt.append((proc_name, resonances))
            else:
                resonances = opt
        else:
            helas_born_diagrams = matrix_element.get_born_diagrams()
            if len(helas_born_diagrams)==0:
                logger.warning('The CMS check for loop-induced process is '+\
                                  'not yet available (nor is it very interesting).')
                return None
            born_order = get_order(helas_born_diagrams,'Born')
            loop_order = get_order(matrix_element.get_loop_diagrams(),'loop')

            # Second run (CMS), we can reuse the information if it is a dictionary
            if isinstance(opt, list):
                opt.append((process.base_string(),find_resonances(helas_born_diagrams)))
                resonances = opt[-1][1]
            else:
                resonances = opt
            # Save the resonances to a pickle file in the output directory so that
            # it can potentially be reused.
            save_load_object.save_to_file(pkl_path, (process.base_string(),
                                             born_order, loop_order,resonances))

    else:
        # The LO equivalent
        try:
            amplitude = diagram_generation.Amplitude(process)
        except InvalidCmd:
            logging.info("No diagrams for %s" % \
                            process.nice_string().replace('Process', 'process'))
            return None
        if not amplitude.get('diagrams'):
            # This process has no diagrams; go to next process
            logging.info("No diagrams for %s" % \
                             process.nice_string().replace('Process', 'process'))
            return None

        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                                 gen_color=True)
        diagrams = matrix_element.get('diagrams')
        born_order = get_order(diagrams,'Born')
        # Loop order set to -1 indicates an LO result
        loop_order = -1
        # Find all the resonances of the ME, if not already given in opt
        if isinstance(opt, list):
            opt.append((process.base_string(),find_resonances(diagrams)))
            resonances = opt[-1][1]
        else:
            resonances= opt
    
    if len(resonances)==0:
        logger.info("No resonance found for process %s."\
                                                         %process.base_string())
        return None
    
    # Cache the default param_card for NLO
    if not options['cached_param_card'][mode][0]:
        if NLO:
            param_card = check_param_card.ParamCard(
                                       pjoin(proc_dir,'Cards','param_card.dat'))
        else:
            param_card = check_param_card.ParamCard(
                     StringIO.StringIO(evaluator.full_model.write_param_card()))
        options['cached_param_card'][mode][0] = param_card
        name2block, _ = param_card.analyze_param_card()
        options['cached_param_card'][mode][1] = name2block
        
    else:
        param_card = options['cached_param_card'][mode][0]
        name2block = options['cached_param_card'][mode][1]

    # Already add the coupling order for this sqaured ME.
    if loop_order != -1 and (loop_order+born_order)%2 != 0:
        raise MadGraph5Error, 'The summed squared matrix element '+\
                              " order '%d' is not even."%(loop_order+born_order)
    result = {'born_order':born_order, 
              'loop_order': (-1 if loop_order==-1 else (loop_order+born_order)/2),
              'resonances_result':[]}

    # Create a physical backup of the param_card
    if NLO:
        try:
            shutil.copy(pjoin(proc_dir,'Cards','param_card.dat'),
                             pjoin(proc_dir,'Cards','param_card.dat__backUp__'))
        except:
            pass

    # Apply custom tweaks
    had_log_tweaks=False
    if NLO:
        for tweak in options['tweak']['custom']:
            if tweak.startswith('seed'):
                continue
            try:
                logstart, logend = tweak.split('->')
            except:
                raise Madgraph5Error, "Tweak '%s' not reckognized."%tweak
            if logstart in ['logp','logm', 'log'] and \
               logend in ['logp','logm', 'log']:
                if NLO:
                    evaluator.apply_log_tweak(proc_dir, [logstart, logend])
                    had_log_tweaks = True
            else:
                raise Madgraph5Error, "Tweak '%s' not reckognized."%tweak
        if had_log_tweaks:
            evaluator.apply_log_tweak(proc_dir, 'recompile')

    # Select what resonances should be run
    if options['resonances']=='all':
        resonances_to_run = resonances
    elif isinstance(options['resonances'],int):
        resonances_to_run = resonances[:options['resonances']]    
    elif isinstance(options['resonances'],list):
        resonances_to_run = []
        for res in resonances:
            for res_selection in options['resonances']:
                if abs(res['ParticlePDG'])==res_selection[0] and \
                                 res['FSMothersNumbers']==set(res_selection[1]):
                    resonances_to_run.append(res)
                    break
    else:
        raise InvalidCmd("Resonance selection '%s' not reckognized"%\
                                                     str(options['resonances']))

    # Display progressbar both for LO and NLO for now but not when not showing
    # the plots
    if NLO and options['show_plot']:
        widgets = ['ME evaluations:', pbar.Percentage(), ' ', 
                                                pbar.Bar(),' ', pbar.ETA(), ' ']
        progress_bar = pbar.ProgressBar(widgets=widgets, 
                maxval=len(options['lambdaCMS'])*len(resonances_to_run), fd=sys.stdout)
        progress_bar.update(0)
        # Flush stdout to force the progress_bar to appear
        sys.stdout.flush()
    else:
        progress_bar = None

    for resNumber, res in enumerate(resonances_to_run):
        # First add a dictionary for this resonance to the result with already
        # one key specifying the resonance
        result['resonances_result'].append({'resonance':res,'born':[]})
        if NLO:
            result['resonances_result'][-1]['finite'] = []
        # Now scan the different lambdaCMS values
        for lambdaNumber, lambdaCMS in enumerate(options['lambdaCMS']):
            # Setup the model for that value of lambdaCMS
            # The copy constructor below creates a deep copy
            new_param_card = check_param_card.ParamCard(param_card)
            # Change all specified parameters
            for param, replacement in options['expansion_parameters'].items():
                # Replace the temporary prefix used for evaluation of the 
                # substitution expression 
                orig_param = param.replace('__tmpprefix__','')
                if orig_param not in name2block:
                    # It can be that some parameter ar in the NWA model but not
                    # in the CMS, such as the Yukawas for example.
                    # logger.warning("Unknown parameter '%s' in mode '%s'."%(param,mode))
                    continue
                for block, lhaid in name2block[orig_param]:
                    orig_value = float(param_card[block].get(lhaid).value)
                    new_value  = eval(replacement,
                                       {param:orig_value,'lambdacms':lambdaCMS})
                    new_param_card[block].get(lhaid).value=new_value

            # Apply these changes already (for the purpose of Width computation.
            # although it is optional since we now provide the new_param_card to
            # the width computation function.). Also in principle this matters
            # only in the CMS and there the widths would be reused from their 
            # prior computation within NWA with zero widths. So, all in all,
            # the line below is really not crucial, but semantically, it ought
            # to be there.
            evaluator.full_model.set_parameters_and_couplings(
                                                      param_card=new_param_card)
            # Now compute or recyle all widths
            for decay in new_param_card['decay'].keys():
                if mode=='CMS':
                    new_width = get_width(abs(decay[0]), lambdaCMS, 
                                                                 new_param_card)
                else:
                    new_width = 0.0
                new_param_card['decay'].get(decay).value= new_width

            # Apply these changes for the purpose of the final computation
            evaluator.full_model.set_parameters_and_couplings(
                                                      param_card=new_param_card)
            if NLO:
                new_param_card.write(pjoin(proc_dir,'Cards','param_card.dat'))
                # Write the recomputed widths so that it can potentially be
                # used for future runs (here for the model in the CMS format)
                if lambdaCMS==1.0 and mode=='CMS' and \
                          options['recompute_width'] in ['always','first_time']:
                    new_param_card.write(pjoin(proc_dir,
                                    'Cards','param_card.dat_recomputed_widths'))
                    
            # If recomputing widths with MadWidths, we want to do it within
            # the NWA models with zero widths.
            if mode=='NWA' and (options['recompute_width']=='always' or (
                  options['recompute_width']=='first_time' and lambdaCMS==1.0)):
                # The copy constructor below creates a deep copy
                tmp_param_card = check_param_card.ParamCard(new_param_card)
                # We don't use the result here, it is just so that it is put
                # in the cache and reused in the CMS run that follows.
                for decay in new_param_card['decay'].keys():
                    particle_name = evaluator.full_model.get_particle(\
                                                       abs(decay[0])).get_name()
                    new_width = get_width(abs(decay[0]),lambdaCMS,new_param_card)
                    tmp_param_card['decay'].get(decay).value = new_width
                    if not options['has_FRdecay'] and new_width != 0.0 and \
                      (abs(decay[0]),lambdaCMS) not in options['cached_widths']:
                        logger.info('Numerically computed width of particle'+\
                                        ' %s for lambda=%.4g : %-9.6gGeV'%
                                            (particle_name,lambdaCMS,new_width))

                # Write the recomputed widths so that it can potentially be
                # used for future runs (here the model in the NWA format)
                if lambdaCMS==1.0 and NLO:
                    tmp_param_card.write(pjoin(proc_dir,
                                    'Cards','param_card.dat_recomputed_widths'))
            
            # Apply the params tweaks
            for param, replacement in options['tweak']['params'].items():
               # Replace the temporary prefix used for evaluation of the 
               # substitution expression 
               orig_param = param.replace('__tmpprefix__','')
               # Treat the special keyword 'allwidths'
               if orig_param.lower() == 'allwidths':
                    # Apply the rule to all widhts
                    for decay in new_param_card['decay'].keys():
                        orig_value = float(new_param_card['decay'].get(decay).value)
                        new_value = eval(replacement,
                                       {param:orig_value,'lambdacms':lambdaCMS})
                        new_param_card['decay'].get(decay).value = new_value
                    continue
               if orig_param not in name2block:
                   # It can be that some parameter are in the NWA model but not
                   # in the CMS, such as the Yukawas for example.
                   continue
               for block, lhaid in name2block[orig_param]:
                   orig_value = float(new_param_card[block].get(lhaid).value)
                   new_value  = eval(replacement,
                                       {param:orig_value,'lambdacms':lambdaCMS})
                   new_param_card[block].get(lhaid).value=new_value
            
            if options['tweak']['params']:
                # Apply the tweaked param_card one last time
                evaluator.full_model.set_parameters_and_couplings(
                                                      param_card=new_param_card)
                if NLO:
                    new_param_card.write(pjoin(proc_dir,'Cards','param_card.dat'))

            # Finally ready to compute the matrix element
            if NLO:
                ME_res = LoopMatrixElementEvaluator.get_me_value(process, 0, 
                       proc_dir, PSpoint=res['PS_point_used'], verbose=False, 
                                           format='dict', skip_compilation=True)
                # Notice that there is much more information in ME_res. It can
                # be forwarded to check_complex_mass_scheme in this result
                # dictionary if necessary for the analysis. (or even the full 
                # dictionary ME_res can be added).
                result['resonances_result'][-1]['born'].append(ME_res['born'])
                result['resonances_result'][-1]['finite'].append(
                      ME_res['finite']*ME_res['born']*ME_res['alphaS_over_2pi'])
            else:
                ME_res = evaluator.evaluate_matrix_element(matrix_element,
                    p=res['PS_point_used'], auth_skipping=False, output='m2')[0]
                result['resonances_result'][-1]['born'].append(ME_res)
            if not progress_bar is None:
                progress_bar.update(resNumber*len(options['lambdaCMS'])+\
                                                               (lambdaNumber+1))
                # Flush to force the printout of the progress_bar to be updated
                sys.stdout.flush()

    # Restore the original continued log definition if necessary
    log_reversed = False
    for tweak in options['tweak']['custom']:
        if tweak.startswith('log') and had_log_tweaks:
            if log_reversed:
                continue
            if NLO:
                evaluator.apply_log_tweak(proc_dir, 'default')
                evaluator.apply_log_tweak(proc_dir, 'recompile')                
                log_reversed = True

    # Restore the original model parameters
    evaluator.full_model.set_parameters_and_couplings(param_card=param_card)
    if NLO:
        try:
            shutil.copy(pjoin(proc_dir,'Cards','param_card.dat__backUp__'),
                                      pjoin(proc_dir,'Cards','param_card.dat'))
        except:
            param_card.write(pjoin(proc_dir,'Cards','param_card.dat'))
    
    # All should have been restored properly, so we can now clean the temporary
    # backups
    try:
        os.remove(pjoin(proc_dir,'Cards','param_card.dat__TemporaryBackup__'))
        os.remove(pjoin(proc_dir,'Source','MODEL',
                                        'model_functions.f__TemporaryBackup__'))
    except:
        pass

    return (process.nice_string().replace('Process:', '').strip(),result)

def get_value(process, evaluator, p=None, options=None):
    """Return the value/momentum for a phase space point"""
    
    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking %s in %s gauge" % \
        ( process.nice_string().replace('Process:', 'process'),
                               'unitary' if aloha.unitary_gauge else 'feynman'))

    legs = process.get('legs')
    # Generate a process with these legs
    # Generate the amplitude for this process
    try:
        if process.get('perturbation_couplings')==[]:
            amplitude = diagram_generation.Amplitude(process)
        else:
            amplitude = loop_diagram_generation.LoopAmplitude(process)
    except InvalidCmd:
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    
    if not amplitude.get('diagrams'):
        # This process has no diagrams; go to next process
        logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
        return None
    
    if not p:
        # Generate phase space point to use
        p, w_rambo = evaluator.get_momenta(process, options)
        
    # Generate the HelasMatrixElement for the process
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = True)
    else:
        matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude, 
           gen_color = True, optimized_output = evaluator.loop_optimized_output)

    mvalue = evaluator.evaluate_matrix_element(matrix_element, p=p,
                                                  output='jamp',options=options)
    
    if mvalue and mvalue['m2']:
        return {'process':process.base_string(),'value':mvalue,'p':p}

def output_lorentz_inv_loop(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format for loop 
    processes. It detail the results from each lorentz transformation performed.
    """

    process = comparison_results[0]['process']
    results = comparison_results[0]['results']
    # Rotations do not change the reference vector for helicity projection,
    # the loop ME are invarariant under them with a relatively good accuracy.
    threshold_rotations = 1e-6
    # This is typically not the case for the boosts when one cannot really 
    # expect better than 1e-5. It turns out that this is even true in 
    # quadruple precision, for an unknown reason so far.
    threshold_boosts =  1e-3
    res_str = "%s" % process.base_string()
    
    transfo_col_size = 17
    col_size = 18
    transfo_name_header = 'Transformation name'

    if len(transfo_name_header) + 1 > transfo_col_size:
        transfo_col_size = len(transfo_name_header) + 1
    
    misc.sprint(results)
    for transfo_name, value in results:
        if len(transfo_name) + 1 > transfo_col_size:
            transfo_col_size = len(transfo_name) + 1
        
    res_str += '\n' + fixed_string_length(transfo_name_header, transfo_col_size) + \
      fixed_string_length("Value", col_size) + \
      fixed_string_length("Relative diff.", col_size) + "Result"
    
    ref_value = results[0]
    res_str += '\n' + fixed_string_length(ref_value[0], transfo_col_size) + \
                   fixed_string_length("%1.10e" % ref_value[1]['m2'], col_size)
    # Now that the reference value has been recuperated, we can span all the 
    # other evaluations
    all_pass = True
    for res in results[1:]:
        threshold = threshold_boosts if 'BOOST' in res[0].upper() else \
                                                             threshold_rotations
        rel_diff = abs((ref_value[1]['m2']-res[1]['m2'])\
                                       /((ref_value[1]['m2']+res[1]['m2'])/2.0))
        this_pass = rel_diff <= threshold
        if not this_pass: 
            all_pass = False
        res_str += '\n' + fixed_string_length(res[0], transfo_col_size) + \
                   fixed_string_length("%1.10e" % res[1]['m2'], col_size) + \
                   fixed_string_length("%1.10e" % rel_diff, col_size) + \
                   ("Passed" if this_pass else "Failed")
    if all_pass:
        res_str += '\n' + 'Summary: passed'
    else:
        res_str += '\n' + 'Summary: failed'
    
    return res_str

def output_lorentz_inv(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format
        if output='fail' return the number of failed process -- for test-- 
    """

    # Special output for loop processes
    if comparison_results[0]['process']['perturbation_couplings']!=[]:
        return output_lorentz_inv_loop(comparison_results, output)

    proc_col_size = 17

    threshold=1e-10
    process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = len(process_header) + 1
    
    for proc, values in comparison_results:
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    col_size = 18

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length(process_header, proc_col_size) + \
              fixed_string_length("Min element", col_size) + \
              fixed_string_length("Max element", col_size) + \
              fixed_string_length("Relative diff.", col_size) + \
              "Result"

    for one_comp in comparison_results:
        proc = one_comp['process'].base_string()
        data = one_comp['results']
        
        if data == 'pass':
            no_check_proc += 1
            no_check_proc_list.append(proc)
            continue

        values = [data[i]['m2'] for i in range(len(data))]
        
        min_val = min(values)
        max_val = max(values)
        diff = (max_val - min_val) / abs(max_val) 
        
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % min_val, col_size) + \
                   fixed_string_length("%1.10e" % max_val, col_size) + \
                   fixed_string_length("%1.10e" % diff, col_size)
                   
        if diff < threshold:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"
        else:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"

        #check all the JAMP
        # loop over jamp
        # Keep in mind that this is not available for loop processes where the
        # jamp list is empty
        if len(data[0]['jamp'])!=0:
            for k in range(len(data[0]['jamp'][0])):
                sum = [0] * len(data)
                # loop over helicity
                for j in range(len(data[0]['jamp'])):
                    #values for the different lorentz boost
                    values = [abs(data[i]['jamp'][j][k])**2 for i in range(len(data))]
                    sum = [sum[i] + values[i] for i in range(len(values))]
    
                # Compare the different lorentz boost  
                min_val = min(sum)
                max_val = max(sum)
                if not max_val:
                    continue
                diff = (max_val - min_val) / max_val 
            
                tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
                           fixed_string_length("%1.10e" % min_val, col_size) + \
                           fixed_string_length("%1.10e" % max_val, col_size) + \
                           fixed_string_length("%1.10e" % diff, col_size)
                       
                if diff > 1e-10:
                    if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                        fail_proc += 1
                        pass_proc -= 1
                        failed_proc_list.append(proc)
                    res_str += tmp_str + "Failed"
                elif not proc_succeed:
                 res_str += tmp_str + "Passed" 
            
            
        
    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)
    
    if output == 'text':
        return res_str        
    else: 
        return fail_proc

def output_unitary_feynman(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format
        if output='fail' return the number of failed process -- for test-- 
    """
    
    proc_col_size = 17
    
    # We use the first element of the comparison_result list to store the
    # process definition object
    pert_coupl = comparison_results[0]['perturbation_couplings']
    comparison_results = comparison_results[1:]
    
    if pert_coupl:
        process_header = "Process [virt="+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"
    
    if len(process_header) + 1 > proc_col_size:
        proc_col_size = len(process_header) + 1
    
    for data in comparison_results:
        proc = data['process']
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    col_size = 18

    res_str = fixed_string_length(process_header, proc_col_size) + \
              fixed_string_length("Unitary", col_size) + \
              fixed_string_length("Feynman", col_size) + \
              fixed_string_length("Relative diff.", col_size) + \
              "Result"

    for one_comp in comparison_results:
        proc = one_comp['process']
        data = [one_comp['value_unit'], one_comp['value_feynm']]
        
        
        if data[0] == 'pass':
            no_check_proc += 1
            no_check_proc_list.append(proc)
            continue
        
        values = [data[i]['m2'] for i in range(len(data))]
        
        min_val = min(values)
        max_val = max(values)
        # when max_val is also negative
        # diff will be negative if there is no abs
        diff = (max_val - min_val) / abs(max_val) 
        
        res_str += '\n' + fixed_string_length(proc, proc_col_size) + \
                   fixed_string_length("%1.10e" % values[0], col_size) + \
                   fixed_string_length("%1.10e" % values[1], col_size) + \
                   fixed_string_length("%1.10e" % diff, col_size)
                   
        if diff < 1e-8:
            pass_proc += 1
            proc_succeed = True
            res_str += "Passed"
        else:
            fail_proc += 1
            proc_succeed = False
            failed_proc_list.append(proc)
            res_str += "Failed"

        #check all the JAMP
        # loop over jamp
        # This is not available for loop processes where the jamp list returned
        # is empty.
        if len(data[0]['jamp'])>0:
            for k in range(len(data[0]['jamp'][0])):
                sum = [0, 0]
                # loop over helicity
                for j in range(len(data[0]['jamp'])):
                    #values for the different lorentz boost
                    values = [abs(data[i]['jamp'][j][k])**2 for i in range(len(data))]
                    sum = [sum[i] + values[i] for i in range(len(values))]
    
                # Compare the different lorentz boost  
                min_val = min(sum)
                max_val = max(sum)
                if not max_val:
                    continue
                diff = (max_val - min_val) / max_val 
            
                tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , col_size) + \
                           fixed_string_length("%1.10e" % sum[0], col_size) + \
                           fixed_string_length("%1.10e" % sum[1], col_size) + \
                           fixed_string_length("%1.10e" % diff, col_size)
                       
                if diff > 1e-10:
                    if not len(failed_proc_list) or failed_proc_list[-1] != proc:
                        fail_proc += 1
                        pass_proc -= 1
                        failed_proc_list.append(proc)
                    res_str += tmp_str + "Failed"
                elif not proc_succeed:
                     res_str += tmp_str + "Passed" 
                
            
        
    res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                (pass_proc, pass_proc + fail_proc,
                 fail_proc, pass_proc + fail_proc)

    if fail_proc != 0:
        res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)
    if no_check_proc:
        res_str += "\nNot checked processes: %s" % ', '.join(no_check_proc_list)
    
    
    if output == 'text':
        return res_str        
    else: 
        return fail_proc

def CMS_save_path(extension, cms_res, used_model, opts, output_path=None):
    """Creates a suitable filename for saving these results."""
    
    if opts['name']=='auto' and opts['analyze']!='None':
        # Reuse the same name then
        return '%s.%s'%(os.path.splitext(opts['analyze'].split(',')[0])\
                                                                  [0],extension)
    # if a name is specified, use it
    if opts['name']!='auto':
        basename = opts['name']
    else:
        prefix = 'cms_check_'
        # Use process name if there is only one process            
        if len(cms_res['ordered_processes'])==1:
            proc = cms_res['ordered_processes'][0]
            replacements = {' ':'','+':'p','-':'m','~':'x', '>':'_','=':'eq'}
            # Remove the perturbation couplings:
            try:
                proc=proc[:proc.index('[')]
            except ValueError:
                pass
    
            for key, value in replacements.items():
                proc = proc.replace(key,value)
    
            basename =prefix+proc+'_%s_'%used_model.get('name')+\
                      ( ('_'+'_'.join(cms_res['perturbation_orders'])) if \
                                      cms_res['perturbation_orders']!=[] else '')
        # Use timestamp otherwise
        else:
            basename = prefix+datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
            
    suffix = '_%s'%opts['tweak']['name'] if opts['tweak']['name']!='' else '' 
    if output_path:     
        return pjoin(output_path,'%s%s.%s'%(basename,suffix,extension))
    else:
        return '%s%s.%s'%(basename,suffix,extension)

def output_complex_mass_scheme(result,output_path, options, model, output='text'):
    """ Outputs nicely the outcome of the complex mass scheme check performed
    by varying the width in the offshell region of resonances found for eahc process.
    Output just specifies whether text should be returned or a list of failed
    processes. Use 'concise_text' for a consise report of the results."""
    
    pert_orders=result['perturbation_orders']
    
    ######## CHECK PARAMETERS #########
    #
    # diff_lambda_power choses the power by which one should divide the difference
    # curve. The test should only work with 1, but it is useful for the LO
    # check to see the difference has O(\lambda) contribution by setting this
    # parameter to 2. If the Born does not have O(\lambda) contributions
    # (i.e. if the test still pas with diff_lambda_power=2) then the NLO test
    # will not be sensitive to the CMS implementation details.
    diff_lambda_power = options['diff_lambda_power']
    # DISLAIMER:
    # The CMS check is non trivial to automate and it is actually best done
    # manually by looking at plots for various implementation of the CMS.
    # The automatic check performed here with the default parameters below
    # should typically capture the main features of the CMS implementation.
    # There will always be exceptions however.
    #
    if 'has_FRdecay' in result:
        has_FRdecay = result['has_FRdecay']
    else:
        has_FRdecay = False
    # be tighter at LO
    if not pert_orders:
        CMS_test_threshold = 1e-3
    else:
        # AT NLO, a correct cancellation is typically of the order of 2% with
        # a lowest lambda value of 10^-4. It is clear that the threshold should
        # scale with the minimum lambda value because any little offset in the
        # LO width value for example (acceptable when less than one 1% if the
        # widths were computed numerically) will lead to an inaccuracy of the 
        # cancellation scaling with lambda. 
        if not has_FRdecay and ('recomputed_with' not in result or \
                          result['recompute_width'] in ['always','first_time']):
            CMS_test_threshold = 2e-2*(1.0e-4/min(result['lambdaCMS']))
        else:
            # If the widths were not computed numerically, then the accuracy of
            # the cancellation should be better.
            CMS_test_threshold = 2e-2*(1.0e-5/min(result['lambdaCMS']))
            
    # This threshold sets how flat the diff line must be when approaching it from
    # the right to start considering its value. Notice that it cannot be larger
    # than the CMS_test_threshold
    consideration_threshold = min(CMS_test_threshold/10.0, 0.05)
    # Number of values groupes with the median technique to avoid being
    # sensitive to unstabilities
    group_val = 3
    # Starting from which value, relative to the averaged diff, should one consider
    # the asymptotic diff median to be exactly 0.0 in which case one would use this
    # average instead of this asymptotic median. u d~ > e+ ve LO exhibit a \
    # difference at zero for example.
    diff_zero_threshold = 1e-3
    # Plotting parameters. Specify the lambda range to plot. 
    # lambda_range = [-1,-1] returns the default automatic setup
    lambda_range = options['lambda_plot_range']
    ##################################
    
#   One can print out the raw results by uncommenting the line below
#    misc.sprint(result)
#    for i, res in enumerate(result['a e- > e- ve ve~ [ virt = QCD QED ]']['CMS']):
#    for i, res in enumerate(result['u d~ > e+ ve a [ virt = QCD QED ]']['CMS']):
#        if res['resonance']['FSMothersNumbers'] == set([3, 4]):
#            misc.sprint(res['resonance']['PS_point_used'])
#    stop
    
    res_str           = ''
    # Variables for the concise report
    concise_str       = ''
    concise_data      = '%%(process)-%ds%%(asymptot)-15s%%(cms_check)-25s%%(status)-25s\n'
    concise_repl_dict = {'Header':{'process':'Process',
                                   'asymptot':'Asymptot',
                                   'cms_check':'Deviation to asymptot',
                                   'status':'Result'}}
    
    ####### BEGIN helper functions

    # Chose here whether to use Latex particle names or not
    # Possible values are 'none', 'model' or 'built-in'
    useLatexParticleName = 'built-in'
    name2tex = {'e+':r'e^+','w+':r'W^+','a':r'\gamma','g':'g',
                'e-':r'e^-','w-':r'W^-','z':'Z','h':'H',
                'mu+':r'\mu^+',
                'mu-':r'\mu^-',
                'ta+':r'\tau^+',
                'ta-':r'\tau^-'}
    for p in ['e','m','t']:
        d = {'e':'e','m':r'\mu','t':r'\tau'}
        name2tex['v%s'%p]=r'\nu_{%s}'%d[p]
        name2tex['v%s~'%p]=r'\bar{\nu_{%s}}'%d[p]
        
    for p in ['u','d','c','s','b','t']:
        name2tex[p]=p
        name2tex['%s~'%p]=r'\bar{%s}'%p
      
    def format_particle_name(particle, latex=useLatexParticleName):
        p_name = particle
        if latex=='model':
            try:
                texname = model.get_particle(particle).get('texname')
                if texname and texname!='none':
                    p_name = r'$\displaystyle %s$'%texname
            except:
                pass
        elif latex=='built-in':
            try:
                p_name = r'$\displaystyle %s$'%name2tex[particle]
            except:
                pass
        return p_name

    def resonance_str(resonance, latex=useLatexParticleName):
        """ Provides a concise string to characterize the resonance """
        particle_name = model.get_particle(resonance['ParticlePDG']).get_name()
        mothersID=['%d'%n for n in sorted(resonance['FSMothersNumbers'])]
        return r"%s [%s]"%(format_particle_name(particle_name,latex=latex),
                                                            ','.join(mothersID))

    def format_title(process, resonance):
        """ Format the plot title given the process and resonance """
        
        process_string = []
        for particle in process.split():
            if particle=='$$':
                process_string.append(r'\$\$')
                continue
            if particle=='>':
                process_string.append(r'$\displaystyle \rightarrow$')
                continue
            process_string.append(format_particle_name(particle))              

        if resonance=='':
            return r'CMS check for %s' %(' '.join(process_string))
        else:
            return r'CMS check for %s ( resonance %s )'\
                                           %(' '.join(process_string),resonance)

    def guess_lambdaorder(ME_values_list, lambda_values, expected=None,
                                                           proc=None, res=None):
        """ Guess the lambda scaling from a list of ME values and return it.
        Also compare with the expected result if specified and trigger a 
        warning if not in agreement."""
        # guess the lambdaCMS power in the amplitude squared
        bpowers = []
        for i, lambdaCMS in enumerate(lambda_values[1:]):
            bpowers.append(round(math.log(ME_values_list[0]/ME_values_list[i+1],\
                                                   lambda_values[0]/lambdaCMS)))

        # Pick the most representative power
        bpower = sorted([(el, bpowers.count(el)) for el in set(bpowers)],
                                 key = lambda elem: elem[1], reverse=True)[0][0]
        if not expected:
            return bpower
        if bpower != expected:
            logger.warning('The apparent scaling of the squared amplitude'+
             'seems inconsistent w.r.t to detected value '+
             '(%i vs %i). %i will be used.'%(expected,bpower,bpower)+
             ' This happend for process %s and resonance %s'%(proc, res))
        return bpower    

    def check_stability(ME_values, lambda_values, lambda_scaling, values_name):
        """ Checks if the values passed in argument are stable and return the 
        stability check outcome warning if it is not precise enough. """

        values = sorted([
            abs(val*(lambda_values[0]/lambda_values[i])**lambda_scaling) for \
                                                i, val in enumerate(ME_values)])
        median = values[len(values)//2]
        max_diff = max(abs(values[0]-median),abs(values[-1]-median))
        stability = max_diff/median
        stab_threshold = 1e-2
        if stability >= stab_threshold:
            return "== WARNING: Stability check failed for '%s' with stability %.2e.\n"\
                                                       %(values_name, stability)
        else:
            return None
    ####### END helper functions
    if options['analyze']=='None':
        if options['reuse']:
            save_path = CMS_save_path('pkl', result, model, options, 
                                                        output_path=output_path)
            buff = "\nThe results of this check have been stored on disk and its "+\
              "analysis can be rerun at anytime with the MG5aMC command:\n   "+\
            "      check cms --analyze=%s\n"%save_path
            res_str += buff
            concise_str += buff
            save_load_object.save_to_file(save_path, result)
        elif len(result['ordered_processes'])>0:
            buff = "\nUse the following synthax if you want to store "+\
                    "the raw results on disk.\n"+\
                    "    check cms -reuse <proc_def> <options>\n"
            res_str += buff
            concise_str += buff            
    
    ############################
    # Numerical check first    #
    ############################
    
    checks = []
    for process in result['ordered_processes']:
        checks.extend([(process,resID) for resID in \
                                            range(len(result[process]['CMS']))])
        
    if options['reuse']:
        logFile = open(CMS_save_path(
                    'log', result, model, options, output_path=output_path),'w')
    
    lambdaCMS_list=result['lambdaCMS']
    
    # List of failed processes
    failed_procs = []

    # A bar printing function helper. Change the length here for esthetics
    bar = lambda char: char*47

    # Write out the widths used if information is present:
    if 'widths_computed' in result:
        res_str += '\n%s%s%s\n'%(bar('='),' Widths ',bar('='))
        if result['recompute_width'] == 'never':
            res_str += '| Widths extracted from the param_card.dat'
        else:
            res_str += '| Widths computed %s'%('analytically' if has_FRdecay 
                                                             else 'numerically')
            if result['recompute_width'] == 'first_time':
                res_str += ' for \lambda = 1'
            elif result['recompute_width'] == 'always':
                res_str += ' for all \lambda values'
        res_str += " using mode '--recompute_width=%s'.\n"%result['recompute_width']
        for particle_name, width in result['widths_computed']:
            res_str += '| %-10s = %-11.6gGeV\n'%('Width(%s)'%particle_name,width)
        res_str += '%s%s%s\n'%(bar('='),'='*8,bar('='))

    # Doing the analysis to printout to the MG5 interface and determine whether
    # the test is passed or not
    # Number of last points to consider for the stability test
    nstab_points=group_val
    # Store here the asymptot detected for each difference curve
    differences_target = {}
    for process, resID in checks:
        # Reinitialize the concise result replacement dictionary 
        # (only one resonance is indicated in this one, no matter what.)
        concise_repl_dict[process] = {'process':process,
                                      'asymptot':'N/A',
                                      'cms_check':'N/A',
                                      'status':'N/A'}
        proc_res = result[process]
        cms_res = proc_res['CMS'][resID]
        nwa_res = proc_res['NWA'][resID]
        resonance = resonance_str(cms_res['resonance'], latex='none')
        cms_born=cms_res['born']
        nwa_born=nwa_res['born']
        # Starting top thick bar
        res_str += '\n%s%s%s\n'%(bar('='),'='*8,bar('='))
        # Centered process and resonance title
        proc_title = "%s (resonance %s)"%(process,resonance)
        centering = (bar(2)+8-len(proc_title))//2
        res_str += "%s%s\n"%(' '*centering,proc_title)
        # Starting bottom thin bar
        res_str += '%s%s%s\n'%(bar('-'),'-'*8,bar('-'))
        # Reminder if diff_lambda_power is not 1
        
        if diff_lambda_power!=1:
            res_str += "== WARNING diff_lambda_power is not 1 but    = %g\n"%diff_lambda_power
            res_str += '%s%s%s\n'%(bar('-'),'-'*8,bar('-'))

        born_power = guess_lambdaorder(nwa_born,lambdaCMS_list,
               expected=proc_res['born_order'], proc=process, res=resonance)
        stab_cms_born = check_stability(cms_born[-nstab_points:], 
                         lambdaCMS_list[-nstab_points:], born_power, 'CMS Born')
        if stab_cms_born:
            res_str += stab_cms_born
        stab_nwa_born = check_stability(nwa_born[-nstab_points:], 
                         lambdaCMS_list[-nstab_points:], born_power, 'NWA Born')
        if stab_nwa_born:
            res_str += stab_nwa_born
        # Write out the phase-space point
        res_str += "== Kinematic configuration in GeV (E,px,pypz)\n"
        for i, p in enumerate(cms_res['resonance']['PS_point_used']):
            res_str += "  | p%-2.d = "%(i+1)
            for pi in p:
                res_str += '%-24.17g'%pi if pi<0.0 else ' %-23.17g'%pi 
            res_str += "\n"
        # Write out the offshellnesses specification
        res_str += "== Offshellnesses of all detected resonances\n"
        for res_name, offshellness in cms_res['resonance']['offshellnesses']:
            res_str += "  | %-15s = %f\n"%(res_name, offshellness)
        res_str += '%s%s%s\n'%(bar('-'),'-'*8,bar('-'))

        if not pert_orders:
            res_str += "== Born scaling lambda^n_born. nborn         = %d\n"%born_power
        else:
            cms_finite=cms_res['finite']
            nwa_finite=nwa_res['finite']
            loop_power = guess_lambdaorder(nwa_finite,lambdaCMS_list,
                   expected=proc_res['loop_order'], proc=process, res=resonance)  
            res_str += "== Scaling lambda^n. nborn, nloop            = %d, %d\n"\
                                                        %(born_power,loop_power)
            stab_cms_finite = check_stability(cms_finite[-nstab_points:], 
                                  lambdaCMS_list[-nstab_points:], loop_power, 'CMS finite')
            if stab_cms_finite:
                res_str += stab_cms_finite
            stab_nwa_finite = check_stability(nwa_finite[-nstab_points:], 
                      lambdaCMS_list[-nstab_points:], loop_power, 'NWA finite')
            if stab_nwa_finite:
                res_str += stab_nwa_finite
        # Now organize data
        CMSData     = []
        NWAData     = []
        DiffData    = []
        for idata, lam in enumerate(lambdaCMS_list):
            if not pert_orders:
                new_cms=cms_born[idata]/(lam**born_power)
                new_nwa=nwa_born[idata]/(lam**born_power)
            else:
                new_cms=(cms_finite[idata]+cms_born[idata]-nwa_born[idata])/(lam*nwa_born[idata])
                new_nwa=nwa_finite[idata]/(lam*nwa_born[idata])
            new_diff=(new_cms-new_nwa)/(lam**diff_lambda_power)
            CMSData.append(new_cms)
            NWAData.append(new_nwa)
            DiffData.append(new_diff)

                    
        # NWA Born median

        # Find which values to start the test at by looking at the CMSdata scaling
        # First compute the median of the middle 60% of entries in the plot
        trim_range=int(((1.0-0.6)/2.0)*len(DiffData))
        low_diff_median = sorted(DiffData[trim_range:-trim_range])\
                                               [(len(DiffData)-2*trim_range)//2]
        
        # Now walk the values from the right of the diff plot until we reaches
        # values stable with respect to the CMS_tale_median. This value will
        # be limit of the range considered for the CMS test. Do it in a way which
        # is insensitive to instabilities, by considering medians of group_val 
        # consecutive points.
        current_median = 0
        # We really want to select only the very stable region
        scan_index = 0
        reference = abs(sorted(NWAData)[len(NWAData)//2])
        if low_diff_median!= 0.0:
            if abs(reference/low_diff_median)<diff_zero_threshold:
                reference = abs(low_diff_median)
        while True:
            scanner = DiffData[scan_index:group_val+scan_index]
            current_median = sorted(scanner)[len(scanner)//2]
            # Useful for debugging
            #misc.sprint(scanner,current_median,abs(current_median-low_diff_median)/reference,reference,consideration_threshold)
            if abs(current_median-low_diff_median)/reference<\
                                                        consideration_threshold:
                break;
            scan_index += 1
            if (group_val+scan_index)>=len(DiffData):
                # this should not happen, but in this case we arbitrarily take
                # half of the data
                logger.warning('The median scanning failed during the CMS check '+
                  'for process %s'%proc_title+\
                  'This is means that the difference plot has not stable'+\
                  'intermediate region and MG5_aMC will arbitrarily consider the'+\
                                                     'left half of the values.')
                scan_index = -1
                break;
        
        if scan_index == -1:
            cms_check_data_range = len(DiffData)//2
        else:
            cms_check_data_range = scan_index + group_val

        res_str += "== Data range considered (min, max, n_val)   = (%.1e, %.1e, %d)\n"\
                  %(lambdaCMS_list[-1],lambdaCMS_list[scan_index],
                                                 len(lambdaCMS_list)-scan_index)
        # Now setup the list of values affecting the CMScheck
        CMScheck_values = DiffData[cms_check_data_range:]

        # For the purpose of checking the stability of the tale, we now do
        # the consideration_threshold scan from the *left* and if we finsih
        # before the end, it means that there is an unstable region.
        if scan_index >= 0:            
            # try to find the numerical instability region
            scan_index = len(CMScheck_values)
            used_group_val = max(3,group_val)
            unstability_found = True
            while True:
                scanner = CMScheck_values[scan_index-used_group_val:scan_index]
                maxdiff =  max(abs(scan-low_diff_median) for scan in scanner)
                if maxdiff/reference<consideration_threshold:
                    break;
                if (scan_index-used_group_val)==0:
                    # this only happens when no stable intermediate region can be found
                    # Set scan_index to -99 so as to prevent warning
                    unstability_found = False
                    break;
                # Proceed to th next block of data
                scan_index -= 1

            # Now report here the unstability found
            if unstability_found:
                unstab_check=CMScheck_values[scan_index:]
                relative_array = [val > CMScheck_values[scan_index-1] for 
                                                            val in unstab_check]
                upper = relative_array.count(True)
                lower = relative_array.count(False)
                if not ((lower==0 and upper>=0) or (lower>=0 and upper==0)):
                    logger.warning(
"""For process %s, a numerically unstable region was detected starting from lambda < %.1e.
Look at the plot in this region (and possibly throw more points using the option --lambdaCMS).
If this is indeed a stability issue, then either decrease MLStabThreshold in MadLoop or decrease the
minimum value of lambda to be considered in the CMS check."""\
                %(proc_title, lambdaCMS_list[cms_check_data_range+scan_index-1]))
        
        # Now apply the same same technique, as above but to the difference plot
        # Now we will use low_diff_median instead of diff_tale_median
        #diff_tale_median = sorted(CMScheck_values)[len(CMScheck_values)//2]
        scan_index = 0
        max_diff = 0.0
        res_str += "== Ref. value used in the ratios (Born NWA)  = %s\n"\
                                                             %('%.3g'%reference)
        res_str += "== Asymptotic difference value detected      = %s\n"\
                                                       %('%.3g'%low_diff_median)
        concise_repl_dict[process]['asymptot'] = '%.3e'%low_diff_median

        # Pass information to the plotter for the difference target
        differences_target[(process,resID)]= low_diff_median
#        misc.sprint('Now doing resonance %s.'%res_str)
        while True:
            current_vals = CMScheck_values[scan_index:scan_index+group_val]      
            max_diff = max(max_diff, abs(low_diff_median-
                     sorted(current_vals)[len(current_vals)//2])/reference)
            if (scan_index+group_val)>=len(CMScheck_values):
                break
            scan_index += 1
        
        # Now use the CMS check result
        cms_check = (max_diff*100.0, '>' if max_diff>CMS_test_threshold else '<',
                                                       CMS_test_threshold*100.0) 
        res_str += "== CMS check result (threshold)              = %.3g%% (%s%.3g%%)\n"%cms_check
        concise_repl_dict[process]['cms_check'] = \
            "%-10s (%s%.3g%%)"%('%.3g%%'%cms_check[0],cms_check[1],cms_check[2])

        if max_diff>CMS_test_threshold:
            failed_procs.append((process,resonance))
        res_str += "%s %s %s\n"%(bar('='),
                 'FAILED' if max_diff>CMS_test_threshold else 'PASSED',bar('='))
        concise_repl_dict[process]['status'] = 'Failed' if max_diff>CMS_test_threshold \
                                                                   else 'Passed'

    if output=='concise_text':
        # Find what is the maximum size taken by the process string
        max_proc_size = max(
                 [len(process) for process in result['ordered_processes']]+[10])
        # Re-initialize the res_str so as to contain only the minimal report
        res_str = concise_str
        res_str += '\n'+concise_data%(max_proc_size+4)%concise_repl_dict['Header']
        for process in result['ordered_processes']:
            res_str += (concise_data%(max_proc_size+4)%concise_repl_dict[process])

    if len(checks):
        res_str += "Summary: %i/%i passed"%(len(checks)-len(failed_procs),len(checks))+\
                    ('.\n' if not failed_procs else ', failed checks are for:\n')
    else:
        return "\nNo CMS check to perform, the process either has no diagram or does not "+\
                                  "not feature any massive s-channel resonance."
        
    for process, resonance in failed_procs:
        res_str += ">  %s, %s\n"%(process, resonance)

    if output=='concise_text':
        res_str += '\nMore detailed information on this check available with the command:\n'
        res_str += '  MG5_aMC>display checks\n'

    ############################
    # Now we turn to the plots #
    ############################
    if not options['show_plot']:
        if options['reuse']:
            logFile.write(res_str)
            logFile.close()
        if output.endswith('text'):
            return res_str
        else:
            return failed_procs
        
    fig_output_file = CMS_save_path('pdf', result, model, options, 
                                                        output_path=output_path)
    base_fig_name = fig_output_file[:-4]
    suffix = 1
    while os.path.isfile(fig_output_file):
        fig_output_file = '%s__%d__.pdf'%(base_fig_name,suffix)
        suffix+=1

    process_data_plot_dict={}
    
    # load possible additional results. The second element of the tuple is
    # the dataset name.
    all_res = [(result, None)]
    for i, add_res in enumerate(options['analyze'].split(',')[1:]):
        specs =re.match(r'^(?P<filename>.*)\((?P<title>.*)\)$', add_res)
        if specs:
            filename = specs.group('filename')
            title    = specs.group('title')
        else:
            filename = add_res
            title    = '#%d'%(i+1)

        new_result = save_load_object.load_from_file(filename)
        if new_result is None:
            raise InvalidCmd('The complex mass scheme check result'+
                             " file below could not be read.\n     %s"%filename)
        if len(new_result['ordered_processes'])!=len(result['ordered_processes']) \
                      or len(new_result['lambdaCMS'])!=len(result['lambdaCMS']):
            raise self.InvalidCmd('The complex mass scheme check result'+
                      " file below does not seem compatible.\n     %s"%filename)
        all_res.append((new_result,title))
    
    # Prepare the data
    for process, resID in checks:
        data1=[] # for subplot 1,i.e. CMS and NWA
        data2=[] # for subplot 2,i.e. diff
        info ={} # info to be passed to the plotter
        for res in all_res:
            proc_res = res[0][process]
            cms_res = proc_res['CMS'][resID]
            nwa_res = proc_res['NWA'][resID]
            resonance = resonance_str(cms_res['resonance'])
            if options['resonances']!=1:
                info['title'] = format_title(process, resonance)
            else:
                info['title'] = format_title(process, '')
            # Born result
            cms_born=cms_res['born']
            nwa_born=nwa_res['born']
            if len(cms_born) != len(lambdaCMS_list) or\
                 len(nwa_born) != len(lambdaCMS_list):
                raise MadGraph5Error, 'Inconsistent list of results w.r.t. the'+\
                                ' lambdaCMS values specified for process %s'%process
            if pert_orders:
                cms_finite=cms_res['finite'] 
                nwa_finite=nwa_res['finite']
                if len(cms_finite) != len(lambdaCMS_list) or\
                    len(nwa_finite) != len(lambdaCMS_list):
                    raise MadGraph5Error, 'Inconsistent list of results w.r.t. the'+\
                                ' lambdaCMS values specified for process %s'%process
        
            bpower = guess_lambdaorder(nwa_born,lambdaCMS_list,
                    expected=proc_res['born_order'], proc=process, res=resonance)

            CMSData  = []
            NWAData  = []
            DiffData = []   
            for idata, lam in enumerate(lambdaCMS_list):
                if not pert_orders:
                    new_cms = cms_born[idata]/lam**bpower
                    new_nwa = nwa_born[idata]/lam**bpower
                else:
                    new_cms=cms_finite[idata]+cms_born[idata]-nwa_born[idata]
                    new_nwa=nwa_finite[idata]
                    new_cms /= lam*nwa_born[idata]
                    new_nwa /= lam*nwa_born[idata]
                new_diff=(new_cms-new_nwa)/(lam**diff_lambda_power)
                CMSData.append(new_cms)
                NWAData.append(new_nwa)
                DiffData.append(new_diff)
            if res[1] is None:
                if not pert_orders:
                    data1.append([r'$\displaystyle CMS\;=\;\mathcal{M}_{CMS}^{(0)}/\lambda^%d$'%bpower,CMSData])
                    data1.append([r'$\displaystyle NWA\;=\;\mathcal{M}_{NWA}^{(0)}/\lambda^%d$'%bpower,NWAData])
                else:
                    data1.append([r'$\displaystyle CMS\;=\;(\mathcal{M}^{(1)}_{CMS}+\mathcal{M}_{CMS}^{(0)}-\mathcal{M}^{(0)}_{NWA})/(\lambda\cdot\mathcal{M}^{(0)}_{NWA})$',CMSData])
                    data1.append([r'$\displaystyle NWA\;=\;\mathcal{M}^{(1)}_{NWA}/(\lambda\cdot\mathcal{M}^{(0)}_{NWA})$',NWAData])
                data2.append([r'$\displaystyle\Delta\;=\;(CMS-NWA)/\lambda%s$'\
                    %('' if diff_lambda_power==1 else r'^{%g}'%diff_lambda_power)
                                                                     ,DiffData])
                data2.append([r'Detected asymptot',[differences_target[(process,resID)] 
                                                for i in range(len(lambdaCMS_list))]])
            else:
                data1.append([r'$\displaystyle CMS$  %s'%res[1].replace('_',' '),CMSData])
                data1.append([r'$\displaystyle NWA$  %s'%res[1].replace('_',' '),NWAData])
                data2.append([r'$\displaystyle\Delta$  %s'%res[1].replace('_',' '),DiffData])
                
        process_data_plot_dict[(process,resID)]=(data1,data2, info)

    # Now turn to the actual plotting
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        logger.info('Rendering plots... (this can take some time because of the latex labels)')

        res_str += \
"""\n-----------------------------------------------------------------------------------------------
| In the plots, the Complex Mass Scheme check is successful if the normalized difference      |
| between the CMS and NWA result (lower inset) tends to a constant when \lambda goes to zero. |
-----------------------------------------------------------------------------------------------\n"""

        # output the figures
        if lambda_range[1]>0:
            min_lambda_index = -1
            for i, lam in enumerate(lambdaCMS_list):
                if lam<=lambda_range[1]:
                    min_lambda_index = i
                    break
        else:
            min_lambda_index = 0
        if lambda_range[0]>0:
            max_lambda_index = -1
            for i, lam in enumerate(lambdaCMS_list):
                if lam<=lambda_range[0]:
                    max_lambda_index=i-1
                    break
        else:
            max_lambda_index=len(lambdaCMS_list)-1
    
        if max_lambda_index==-1 or min_lambda_index==-1 or \
                                             min_lambda_index==max_lambda_index:
            raise InvalidCmd('Invalid lambda plotting range: (%.1e,%.1e)'%\
                                              (lambda_range[0],lambda_range[1]))
        # Trim lambda values
        if lambda_range[0]>0.0 or lambda_range[1]>0.0:
            lambdaCMS_list = lambdaCMS_list[min_lambda_index:max_lambda_index+1]

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        pp=PdfPages(fig_output_file)
        if len(checks)==0 or len(process_data_plot_dict[checks[0]][1])<=7:
            colorlist=['b','r','g','k','c','m','y']
        else:
            import matplotlib.colors as colors
            import matplotlib.cm as mplcm
            import matplotlib.colors as colors
            
            # Nice color maps here are 'gist_rainbow'
            cm = plt.get_cmap('gist_rainbow')
            cNorm  = colors.Normalize(vmin=0, vmax=(len(data2)-1))
            scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
            # use vmax=(len(data1)-1)*0.9 to remove pink at the end of the spectrum
            colorlist = [scalarMap.to_rgba(i*0.9) for i in range(len(data2))]
            # Or it is also possible to alternate colors so as to make them 
            # as distant as possible to one another
            # colorlist = sum([
            #    [scalarMap.to_rgba(i),scalarMap.to_rgba(i+len(data2)//2)]
            #                                  for i in range(len(data2)//2)],[])

        legend_size = 10
        for iproc, (process, resID) in enumerate(checks):
            data1,data2, info=process_data_plot_dict[(process,resID)]
            # Trim dataplot if necessary
            if lambda_range[0]>0.0 or lambda_range[1]>0.0:
                for i in range(len(data1)):
                    data1[i][1]=data1[i][1][min_lambda_index:max_lambda_index+1]
                for i in range(len(data2)):
                    data2[i][1]=data2[i][1][min_lambda_index:max_lambda_index+1]
            plt.figure(iproc+1)
            plt.subplot(211)
            minvalue=1e+99
            maxvalue=-1e+99
            for i, d1 in enumerate(data1):
                # Use the same color for NWA and CMS curve but different linestyle
                color=colorlist[i//2]
                data_plot=d1[1]
                minvalue=min(min(data_plot),minvalue)
                maxvalue=max(max(data_plot),maxvalue)           
                plt.plot(lambdaCMS_list, data_plot, color=color, marker='', \
                        linestyle=('-' if i%2==0 else '--'), 
                        label=(d1[0] if (i%2==0 or i==1) else '_nolegend_'))
            ymin = minvalue-(maxvalue-minvalue)/5.
            ymax = maxvalue+(maxvalue-minvalue)/5.

            plt.yscale('linear')
            plt.xscale('log')
            plt.title(info['title'],fontsize=12,y=1.08)
            plt.ylabel(r'$\displaystyle \mathcal{M}$')
            #plt.xlabel('lambdaCMS')
            if ymax*len(data1)-sum(max(d1[1][-len(d1[1])//2:]) \
                                 for d1 in data1) > 0.5*(ymax-ymin)*len(data1):
                plt.legend(prop={'size':legend_size},loc='upper left', frameon=False)
            else:
                plt.legend(prop={'size':legend_size},loc='lower left', frameon=False)
                
            plt.axis([min(lambdaCMS_list),max(lambdaCMS_list), ymin, ymax])
            
            plt.subplot(212)
            minvalue=1e+99
            maxvalue=-1e+99
            
            try:
                asymptot_index = [d2[0] for d2 in data2].index('Detected asymptot')
                plt.plot(lambdaCMS_list, data2[asymptot_index][1], 
                               color='0.75', marker='', linestyle='-', label='')
            except ValueError:
                pass
            
            color_ID = -1
            for d2 in data2:
                # Special setup for the reference asymptot straight line
                if d2[0]=='Detected asymptot':
                    continue
                color_ID += 1
                color=colorlist[color_ID]
                data_plot=d2[1]
                minvalue=min(min(data_plot),minvalue)
                maxvalue=max(max(data_plot),maxvalue)
                plt.plot(lambdaCMS_list, data_plot, color=color, marker='',\
                                                     linestyle='-', label=d2[0])
            ymin = minvalue-(maxvalue-minvalue)/5.
            ymax = maxvalue+(maxvalue-minvalue)/5.

            plt.yscale('linear')
            plt.xscale('log')
            plt.ylabel(r'$\displaystyle \Delta$')
            plt.xlabel(r'$\displaystyle \lambda$')
            # The unreadable stuff below is just to check if the left of the 
            # plot is stable or not
            sd = [sorted(d2[1][-len(d2[1])//2:]) for d2 in data2]
            left_stability = sum(abs(s[0]-s[-1]) for s in sd)
            sd = [sorted(d2[1][:-len(d2[1])//2]) for d2 in data2]
            right_stability = sum(abs(s[0]-s[-1]) for s in sd)
            left_stable =  False if right_stability==0.0 else \
                                            (left_stability/right_stability)<0.1
 
            if left_stable:
                if ymax*len(data2)-sum(max(d2[1][-len(d2[1])//2:]) \
                                 for d2 in data2) > 0.5*(ymax-ymin)*len(data2):
                    plt.legend(prop={'size':legend_size},loc='upper left', frameon=False)
                else:
                    plt.legend(prop={'size':legend_size},loc='lower left', frameon=False)                
            else:
                if ymax*len(data2)-sum(max(d2[1][:-len(d2[1])//2]) \
                                  for d2 in data2) > 0.5*(ymax-ymin)*len(data2):
                    plt.legend(prop={'size':legend_size},loc='upper right', frameon=False)
                else:
                    plt.legend(prop={'size':legend_size},loc='lower right', frameon=False)

            plt.axis([min(lambdaCMS_list),max(lambdaCMS_list),\
              minvalue-(maxvalue-minvalue)/5., maxvalue+(maxvalue-minvalue)/5.])
            
            plt.savefig(pp,format='pdf')

        pp.close()
        
        if len(checks)>0:
            logger.info('Complex Mass Scheme check plot output to file %s. '%fig_output_file)
            
            if sys.platform.startswith('linux'):
                misc.call(["xdg-open", fig_output_file])
            elif sys.platform.startswith('darwin'):
                misc.call(["open", fig_output_file])
        
        plt.close("all")
    
    except Exception as e:
        if isinstance(e, ImportError):
            res_str += "\n= Install matplotlib to get a "+\
                "graphical display of the results of the cms check."
        else:
            general_error = "\n= Could not produce the cms check plot because of "+\
                                                    "the following error: %s"%str(e)
            try:
                import Tkinter
                if isinstance(e, Tkinter.TclError):
                    res_str += "\n= Plots are not generated because your system"+\
                                          " does not support graphical display."
                else:
                    res_str += general_error
            except:
                res_str += general_error
    
    if options['reuse']:
        logFile.write(res_str)
        logFile.close()

    if output.endswith('text'):
        return res_str
    else:
        return failed_procs

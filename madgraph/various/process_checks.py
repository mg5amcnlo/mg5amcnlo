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
# If psutil becomes standard, the RAM check can be performed with it instead
#import psutil

import aloha
import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writers
import models.import_ufo as import_ufo
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.rambo as rambo
import madgraph.various.misc as misc
import madgraph.various.progressbar as pbar

import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_base_objects as loop_base_objects

from madgraph import MG5DIR, InvalidCmd, MadGraph5Error

from madgraph.iolibs.files import cp

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
# Helper class for timing and RAM flashing of subprocesses.
#===============================================================================
class ProcessTimer:
  def __init__(self,*args,**opts):
    self.cmd_args = args
    self.cmd_opts = opts
    self.execution_state = False

  def execute(self):
    self.max_vms_memory = 0
    self.max_rss_memory = 0

    self.t1 = None
    self.t0 = time.time()
    self.p = subprocess.Popen(*self.cmd_args,**self.cmd_opts)
    self.execution_state = True

  def poll(self):
    if not self.check_execution_state():
      return False

    self.t1 = time.time()
    flash = subprocess.Popen("ps -p %i -o rss"%self.p.pid,
                                              shell=True,stdout=subprocess.PIPE)
    stdout_list = flash.communicate()[0].split('\n')
    rss_memory = int(stdout_list[1])
    # for now we ignore vms
    vms_memory = 0

    # This is the neat version using psutil
#    try:
#      pp = psutil.Process(self.p.pid)
#
#      # obtain a list of the subprocess and all its descendants
#      descendants = list(pp.get_children(recursive=True))
#      descendants = descendants + [pp]
#
#      rss_memory = 0
#      vms_memory = 0
#
#      # calculate and sum up the memory of the subprocess and all its descendants 
#      for descendant in descendants:
#        try:
#          mem_info = descendant.get_memory_info()
#
#          rss_memory += mem_info[0]
#          vms_memory += mem_info[1]
#        except psutil.error.NoSuchProcess:
#          # sometimes a subprocess descendant will have terminated between the time
#          # we obtain a list of descendants, and the time we actually poll this
#          # descendant's memory usage.
#          pass
#
#    except psutil.error.NoSuchProcess:
#      return self.check_execution_state()

    self.max_vms_memory = max(self.max_vms_memory,vms_memory)
    self.max_rss_memory = max(self.max_rss_memory,rss_memory)

    return self.check_execution_state()

  def is_running(self):
    # Version with psutil
#    return psutil.pid_exists(self.p.pid) and self.p.poll() == None
    return self.p.poll() == None

  def check_execution_state(self):
    if not self.execution_state:
      return False
    if self.is_running():
      return True
    self.executation_state = False
    self.t1 = time.time()
    return False

  def close(self,kill=False):

    if self.p.poll() == None:
        if kill:
            self.p.kill()
        else:
            self.p.terminate()

    # Again a neater handling with psutil
#    try:
#      pp = psutil.Process(self.p.pid)
#      if kill:
#        pp.kill()
#      else:
#        pp.terminate()
#    except psutil.error.NoSuchProcess:
#      pass

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
            logger.warning('param_card present in the event file not compatible. We will use the default one.')
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
            # print "I got matrix_methods=",str(matrix_methods.items()[0][1])
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
    
    #===============================================================================
    # Helper function get_momenta
    #===============================================================================
    def get_momenta(self, process, options=None):
        """Get a point in phase space for the external states in the given
        process, with the CM energy given. The incoming particles are
        assumed to be oriented along the z axis, with particle 1 along the
        positive z axis."""

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
        mass_strings = [self.full_model.get_particle(l.get('id')).get('mass') \
                         for l in sorted_legs]        
        mass = [self.full_model.get('parameter_dict')[m] for m in mass_strings]
        mass = [m.real for m in mass]
        #mass = [math.sqrt(m.real) for m in mass]



        # Make sure energy is large enough for incoming and outgoing particles
        energy = max(energy, sum(mass[:nincoming]) + 200.,
                     sum(mass[nincoming:]) + 200.)

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
            FortranExporter.finalize_v4_directory(None,"",False,False,'gfortran')

        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
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

        if isinstance(mp,bool):
            mode = 4 if mp else 1
        else:
            mode = mp
        # Read the existing option card
        file = open(pjoin(dir_name,'MadLoopParams.dat'), 'r')
        MLParams = file.read()
        file.close()
        # Additional option specifications
        for key in MLOptions.keys():
            if key == "ImprovePS":
                MLParams = re.sub(r"#ImprovePS\n\S+","#ImprovePS\n%s"%(\
                            '.TRUE.' if MLOptions[key] else '.FALSE.'),MLParams)
            elif key == "ForceMP":
                if MLOptions[key]:
                    mode = 4
            elif key == "MLReductionLib":
                value=MLOptions[key]
                if isinstance(value,list):
                    if value:
                        mlred="|".join([str(vl) for vl in value])
                    else:
                        mlred="1"
                    MLParams = re.sub(r"#MLReductionLib\n\S+","#MLReductionLib\n%s"\
                        %(mlred),MLParams)
                else:
                    MLParams = re.sub(r"#MLReductionLib\n\S+","#MLReductionLib\n%d"\
                            %(MLOptions[key] if MLOptions[key] else 1),MLParams)
            else:
                logger.error("Key %s is not a valid MadLoop option."%key)

        # Mandatory option specifications
        MLParams = re.sub(r"#CTModeRun\n-?\d+","#CTModeRun\n%d"%mode, MLParams)
        MLParams = re.sub(r"#CTModeInit\n-?\d+","#CTModeInit\n%d"%mode, MLParams)
        MLParams = re.sub(r"#UseLoopFilter\n\S+","#UseLoopFilter\n%s"%(\
                               '.TRUE.' if loop_filter else '.FALSE.'),MLParams)
        MLParams = re.sub(r"#DoubleCheckHelicityFilter\n\S+",
                            "#DoubleCheckHelicityFilter\n%s"%('.TRUE.' if 
                             DoubleCheckHelicityFilter else '.FALSE.'),MLParams)

        # Write out the modified MadLoop option card
        file = open(pjoin(dir_name,'MadLoopParams.dat'), 'w')
        file.write(MLParams)
        file.close()

    @classmethod
    def fix_PSPoint_in_check(cls, dir_path, read_ps = True, npoints = 1,
                             hel_config = -1, mu_r=0.0, split_orders=-1):
        """Set check_sa.f to be reading PS.input assuming a working dir dir_name.
        if hel_config is different than -1 then check_sa.f is configured so to
        evaluate only the specified helicity.
        If mu_r > 0.0, then the renormalization constant value will be hardcoded
        directly in check_sa.f, if is is 0 it will be set to Sqrt(s) and if it
        is < 0.0 the value in the param_card.dat is used.
        If the split_orders target (i.e. the target squared coupling orders for 
        the computation) is != -1, it will be changed in check_sa.f via the
        subroutine CALL SET_COUPLINGORDERS_TARGET(split_orders)."""

        file_path = dir_path
        if not os.path.isfile(dir_path) or \
                                   not os.path.basename(dir_path)=='check_sa.f':
            file_path = pjoin(dir_path,'check_sa.f')
            if not os.path.isfile(file_path):
                directories = [d for d in glob.glob(pjoin(dir_path,'P*_*')) \
                         if (re.search(r'.*P\d+_\w*$', d) and os.path.isdir(d))]
                if len(directories)>0:
                     file_path = pjoin(directories[0],'check_sa.f')
        if not os.path.isfile(file_path):
            raise MadGraph5Error('Could not find the location of check_sa.f'+\
                                  ' from the specified path %s.'%str(file_path))    

        file = open(file_path, 'r')
        check_sa = file.read()
        file.close()
        
        file = open(file_path, 'w')
        check_sa = re.sub(r"READPS = \S+\)","READPS = %s)"%('.TRUE.' if read_ps \
                                                      else '.FALSE.'), check_sa)
        check_sa = re.sub(r"NPSPOINTS = \d+","NPSPOINTS = %d"%npoints, check_sa)
        if hel_config != -1:
            check_sa = re.sub(r"SLOOPMATRIX\S+\(\S+,MATELEM,",
                      "SLOOPMATRIXHEL_THRES(P,%d,MATELEM,"%hel_config, check_sa)
        else:
            check_sa = re.sub(r"SLOOPMATRIX\S+\(\S+,MATELEM,",
                                        "SLOOPMATRIX_THRES(P,MATELEM,",check_sa)
        if mu_r > 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","MU_R=%s"%\
                                        (("%.17e"%mu_r).replace('e','d')),check_sa)
        elif mu_r < 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","",check_sa)
        
        if split_orders > 0:
            check_sa = re.sub(r"SET_COUPLINGORDERS_TARGET\(-?\d+\)",
                     "SET_COUPLINGORDERS_TARGET(%d)"%split_orders,check_sa) 
        
        file.write(check_sa)
        file.close()

    def get_me_value(self, proc, proc_id, working_dir, PSpoint=[], \
                                                  PS_name = None, verbose=True):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id and PSpoint if specified.
        If PS_name is not none the written out PS.input will be saved in 
        the file PS.input_<PS_name> as well"""  
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
         
        shell_name = None
        directories = glob.glob(pjoin(working_dir, 'SubProcesses',
                                  'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" %proc)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        if verbose: logging.debug("Working on process %s in dir %s" % (proc, shell_name))
        
        dir_name = pjoin(working_dir, 'SubProcesses', shell_name)
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
                return self.parse_check_output(file(pjoin(dir_name,\
                                                  'result.dat')),format='tuple')  
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
    
    def setup_ward_check(self, working_dir, file_names, mp = False):
        """ Modify loop_matrix.f so to have one external massless gauge boson
        polarization vector turned into its momentum. It is not a pretty and 
        flexible solution but it works for this particular case."""
        
        shell_name = None
        directories = glob.glob(pjoin(working_dir,'P0_*'))
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
    def make_and_run(cls, dir_name,checkRam=False):
        """ Compile the check program in the directory dir_name.
        Return the compilation and running time. """

        # Make sure to recreate the executable and modified source
        # (The time stamps are sometimes not actualized if it is too fast)
        if os.path.isfile(pjoin(dir_name,'check')):
            os.remove(pjoin(dir_name,'check'))
            os.remove(pjoin(dir_name,'check_sa.o'))
            os.remove(pjoin(dir_name,'loop_matrix.o'))            
        # Now run make
        devnull = open(os.devnull, 'w')
        start=time.time()
        retcode = subprocess.call(['make','check'],
                                   cwd=dir_name, stdout=devnull, stderr=devnull)
        compilation_time = time.time()-start
                     
        if retcode != 0:
            logging.info("Error while executing make in %s" % dir_name)
            return None, None, None

        if not checkRam:
            start=time.time()
            retcode = subprocess.call('./check',
                                   cwd=dir_name, stdout=devnull, stderr=devnull)
            run_time = time.time()-start
            ram_usage = None
        else:
            ptimer = ProcessTimer(['./check'], cwd=dir_name, shell=False, \
                                 stdout=devnull, stderr=devnull, close_fds=True)
            try:
                ptimer.execute()
                #poll as often as possible; otherwise the subprocess might 
                # "sneak" in some extra memory usage while you aren't looking
                # Accuracy of .2 seconds is enough for the timing.
                while ptimer.poll():
                    time.sleep(.2)
            finally:
                #make sure that we don't leave the process dangling.
                ptimer.close()
            # Notice that ptimer.max_vms_memory is also available if needed.
            ram_usage = ptimer.max_rss_memory
            # Unfortunately the running time is less precise than with the
            # above version
            run_time = (ptimer.t1 - ptimer.t0)
            retcode = ptimer.p.returncode

        devnull.close()

        if retcode != 0:
            logging.warning("Error while executing ./check in %s" % dir_name)
            return None, None, None

        return compilation_time, run_time, ram_usage
    
    @classmethod
    def get_MadLoop_Params(cls,MLCardPath):
        """ Return a dictionary of the parameter of the MadLoopParamCard.
        The key is the name of the parameter and the value is the corresponding
        string read from the card."""
        
        res = {}
        # Not elegant, but the file is small anyway, so no big deal.
        MLCard_lines = open(MLCardPath).readlines()
        try:
            for i, line in enumerate(MLCard_lines):
                if line.startswith('#'):
                    res[line.split()[0][1:]]=MLCard_lines[i+1].split()[0]
            return res
        except IndexError:
            raise MadGraph5Error, 'The MadLoop param card %s is '%MLCardPath+\
                                                           'not well formatted.'

    @classmethod
    def set_MadLoop_Params(cls,MLCardPath,params):
        """ Set the parameters in MadLoopParamCard to the values specified in
        the dictionary params.
        The key is the name of the parameter and the value is the corresponding
        string to write in the card."""
        
        # Not elegant, but the file is small anyway, so no big deal.
        MLCard_lines = open(MLCardPath).readlines()
        newCard_lines = []
        modified_Params = []
        param_to_modify=None
        for i, line in enumerate(MLCard_lines):
            if not param_to_modify is None:
                modified_Params.append(param_to_modify)
                newCard_lines.append(params[param_to_modify]+'\n')
                param_to_modify = None
            else:
                if line.startswith('#') and \
                   line.split()[0][1:] in params.keys():
                    param_to_modify = line.split()[0][1:]
                newCard_lines.append(line)
        if not param_to_modify is None:
            raise MadGraph5Error, 'The MadLoop param card %s is '%MLCardPath+\
                                                           'not well formatted.'
        
        left_over = set(params.keys())-set(modified_Params)
        if left_over != set([]):
            raise MadGraph5Error, 'The following parameters could not be '+\
                             'accessed in MadLoopParams.dat : %s'%str(left_over)

        newCard=open(MLCardPath,'w')
        newCard.writelines(newCard_lines)
        newCard.close()

    @classmethod    
    def run_initialization(cls, run_dir=None, SubProc_dir=None, infos=None,\
                            req_files = ['HelFilter.dat','LoopFilter.dat'],
                            attempts = [3,15]):
        """ Run the initialization of the process in 'run_dir' with success 
        characterized by the creation of the files req_files in this directory.
        The directory containing the driving source code 'check_sa.f'.
        The list attempt gives the successive number of PS points the 
        initialization should be tried with before calling it failed.
        Returns the number of PS points which were necessary for the init.
        Notice at least run_dir or SubProc_dir must be provided."""
        
        # If the user does not want detailed info, then set the dictionary
        # to a dummy one.
        if infos is None:
            infos={}
        
        if SubProc_dir is None and run_dir is None:
            raise MadGraph5Error, 'At least one of [SubProc_dir,run_dir] must'+\
                                           ' be provided in run_initialization.'
        
        # If the user does not specify where is check_sa.f, then it is assumed
        # to be one levels above run_dir
        if SubProc_dir is None:
            SubProc_dir = os.path.abspath(pjoin(run_dir,os.pardir))
            
        if run_dir is None:
            directories =[ dir for dir in glob.glob(pjoin(SubProc_dir,\
                                             'P[0-9]*')) if os.path.isdir(dir) ]
            if directories:
                run_dir = directories[0]
            else:
                raise MadGraph5Error, 'Could not find a valid running directory'+\
                                                      ' in %s.'%str(SubProc_dir)

        to_attempt = copy.copy(attempts)
        to_attempt.reverse()
        my_req_files = copy.copy(req_files)
        
        # Make sure that LoopFilter really is needed.
        MLCardPath = pjoin(SubProc_dir,os.pardir,'Cards',\
                                                            'MadLoopParams.dat')
        if not os.path.isfile(MLCardPath):
            raise MadGraph5Error, 'Could not find MadLoopParams.dat at %s.'\
                                                                     %MLCardPath
        if 'FALSE' in cls.get_MadLoop_Params(MLCardPath)['UseLoopFilter'].upper():
            try:
                my_req_files.pop(my_req_files.index('LoopFilter.dat'))
            except ValueError:
                pass
        
        def need_init():
            """ True if init not done yet."""
            proc_prefix_file = open(pjoin(run_dir,'proc_prefix.txt'),'r')
            proc_prefix = proc_prefix_file.read()
            proc_prefix_file.close()
            return any([not os.path.exists(pjoin(run_dir,'MadLoop5_resources',
                            proc_prefix+fname)) for fname in my_req_files]) or \
                         not os.path.isfile(pjoin(run_dir,'check')) or \
                         not os.access(pjoin(run_dir,'check'), os.X_OK)
    
        curr_attempt = 1
        while to_attempt!=[] and need_init():
            curr_attempt = to_attempt.pop()+1
            # Plus one because the filter are written on the next PS point after
            # initialization is performed.
            cls.fix_PSPoint_in_check(run_dir, read_ps = False, 
                                                         npoints = curr_attempt)
            compile_time, run_time, ram_usage = cls.make_and_run(run_dir)
            if compile_time==None:
                logging.error("Failed at running the process in %s."%run_dir)
                attempts = None
                return None
            # Only set process_compilation time for the first compilation.
            if 'Process_compilation' not in infos.keys() or \
                                             infos['Process_compilation']==None:
                infos['Process_compilation'] = compile_time
            infos['Initialization'] = run_time
        
        if need_init():
            return None
        else:
            return curr_attempt-1

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
                                                param_card = None,MLOptions={},clean=True):
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
            FortranExporter.finalize_v4_directory(None,"",False,False,'gfortran')
            infos['HELAS_MODEL_compilation'] = time.time()-start
        
        # Copy the parameter card if provided
        if param_card != None:
            if isinstance(param_card, str):
                cp(pjoin(param_card),\
                              pjoin(export_dir,'Cards','param_card.dat'))
            else:
                param_card.write(pjoin(export_dir,'Cards','param_card.dat'))
                
        # First Initialize filters (in later versions where this will be done
        # at generation time, it can be skipped)
        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                                                   read_ps = False, npoints = 4)
        self.fix_MadLoopParamCard(pjoin(export_dir,'Cards'),
                            mp = False, loop_filter = True,MLOptions=MLOptions)
        
        shell_name = None
        directories = glob.glob(pjoin(export_dir, 'SubProcesses','P0_*'))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])
        dir_name = pjoin(export_dir, 'SubProcesses', shell_name)
        infos['dir_path']=dir_name

        attempts = [3,15]
        # remove check and check_sa.o for running initialization again
        try:
            os.remove(pjoin(dir_name,'check'))
            os.remove(pjoin(dir_name,'check_sa.o'))
        except OSError:
            pass
        
        nPS_necessary = self.run_initialization(dir_name,
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
        
        # Normally, this should work for loop-induced processes as well
#        if not matrix_element.get('processes')[0]['has_born']:
#            return None

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
                                    reusing, param_card,MLOptions = MLOptions)
        
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
        res_timings['du_color']=check_disk_usage(pjoin(dir_name,'*.dat'))
        res_timings['du_exe']=check_disk_usage(pjoin(dir_name,'check'))

        if not res_timings['Initialization']==None:
            time_per_ps_estimate = (res_timings['Initialization']/4.0)/2.0
        else:
            # We cannot estimate from the initialization, so we run just a 3
            # PS point run to evaluate it.
            self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                                  read_ps = False, npoints = 3, hel_config = -1, 
                                                      split_orders=split_orders)
            compile_time, run_time, ram_usage = self.make_and_run(dir_name)
            time_per_ps_estimate = run_time/3.0
        
        self.boot_time_setup(dir_name,bootandstop=True)
        compile_time, run_time, ram_usage = self.make_and_run(dir_name)
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
        target_pspoints_number = max(int(30.0/time_per_ps_estimate)+1,5)

        logger.info("Checking timing for process %s "%proc_name+\
                                    "with %d PS points."%target_pspoints_number)
        
        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_polarized_total']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)

        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                      split_orders=split_orders)
        compile_time, run_time, ram_usage = self.make_and_run(dir_name, 
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

        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                      split_orders=split_orders)
        compile_time, run_time, ram_usage = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_unpolarized_coefs']=\
                   (run_time-res_timings['Booting_time'])/target_pspoints_number
        
        self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = self.make_and_run(dir_name)
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

            def pass_cuts(p):
                """ Defines the cut a PS point must pass"""
                for i, pmom in enumerate(p[2:]):
                    # Pt > 50 GeV
                    if Pt(pmom)<50.0:
                        return False
                    # Delta_R ij > 0.5
                    for pmom2 in p[3+i:]:
                        if DeltaR(pmom,pmom2)<0.5:
                            return False
                return True
            p, w_rambo = self.get_momenta(proc, options)
            if options['events']:
                return p
            # For 2>1 process, we don't check the cuts of course
            while (not pass_cuts(p) and  len(p)>3):
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
        tool_var={'pjfry':2,'golem':4}
        for tool in ['pjfry','golem']:
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
        
        tools_name={1:'CutTools',2:'PJFry++',3:'IREGI',4:'Golem95'}
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
            clean=(tool==tools[0])
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
                    logger.info("ERROR: Not reusing a directory and the number"+\
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
            self.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
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
                        if tool==1:
                            # Only CutTools can use QP
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
                if output==' ##TAG#RESULT_START#TAG##\n':
                    break
            res = ""
            while True:
                output = StabChecker.stdout.readline()
                if output==' ##TAG#RESULT_STOP#TAG##\n':
                    break
                else:
                    res += output
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
                                                         cmd = FakeInterface()):
    """ Generate a loop matrix element from the process definition, and returns
    it along with the timing information dictionary.
    If reuse is True, it reuses the already output directory if found."""

    assert isinstance(process_definition,base_objects.ProcessDefinition)
    assert process_definition.get('perturbation_couplings')!=[]
    
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
    
    if any(len(l.get('ids'))>1 for l in process_definition.get('legs')):
        raise InvalidCmd("This check can only be performed on single "+
                         " processes. (i.e. without multiparticle labels).")

    isids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
              if not leg.get('state')]
    fsids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
             if leg.get('state')]

    # Now generate a process based on the ProcessDefinition given in argument.
    process = process_definition.get_process(isids,fsids)
    
    proc_dir = pjoin(root_path,"SAVED"+temp_dir_prefix+"_%s"%(
                               '_'.join(process.shell_string().split('_')[1:])))
    if reuse and os.path.isdir(proc_dir):
        logger.info("Reusing directory %s"%str(proc_dir))
        # If reusing, return process instead of matrix element
        return timing, process
    
    logger.info("Generating p%s"%process_definition.nice_string()[1:])

    start=time.time()
    amplitude = loop_diagram_generation.LoopAmplitude(process)
    # Make sure to disable loop_optimized_output when considering loop induced 
    # processes
    loop_optimized_output = cmd.options['loop_optimized_output']
    if not amplitude.get('process').get('has_born'):
        loop_optimized_output = False
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
    
    if not reusing and not matrix_element.get('processes')[0].get('has_born'):
        myProfiler.loop_optimized_output=False
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
    if not reusing and not matrix_element.get('processes')[0].get('has_born'):
        myStabilityChecker.loop_optimized_output=False
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
    if not reusing and not matrix_element.get('processes')[0].get('has_born'):
        myTimer.loop_optimized_output=False
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
                if not amplitude.get('process').get('has_born'):
                    evaluator.loop_optimized_output = False   
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
    
    directories = glob.glob(pjoin(mg_root, '%s*'%temp_dir_prefix))
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
            for i, eps in enumerate(EPS):
                logFile.write('\nEPS #%i\n'%(i+1))
                logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in eps[1]]))
                logFile.write('\n  DP accuracy :  %.3e\n'%DP_stability[eps[0]])
                logFile.write('  QP accuracy :  %.3e\n'%QP_stability[eps[0]])
        if len(UPS)>0:
            logFile.write('\nFull details of the %i UPS encountered by %s.\n'\
                                                           %(len(UPS),toolname))
            for i, ups in enumerate(UPS):
                logFile.write('\nUPS #%i\n'%(i+1))
                logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in ups[1]]))
                logFile.write('\n  DP accuracy :  %.3e\n'%DP_stability[ups[0]])
                logFile.write('  QP accuracy :  %.3e\n'%QP_stability[ups[0]])

        logFile.write('\nData entries for the stability plot.\n')
        logFile.write('First row is a maximal accuracy delta, second is the '+\
                  'fraction of events with DP accuracy worse than delta.\n\n')
    # Set the x-range so that it spans [10**-17,10**(min_digit_accuracy)]
        if max(DP_stability)>0.0:
            min_digit_acc=int(math.log(max(DP_stability))/math.log(10))
            if min_digit_acc>=0:
                min_digit_acc = min_digit_acc+1
            accuracies=[10**(-17+(i/5.0)) for i in range(5*(17+min_digit_acc)+1)]
        else:
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
        
        logFile.writelines('%.3e  %.3e\n'%(accuracies[i], data_plot[i]) for i in \
                                                         range(len(accuracies)))
        logFile.write('\nList of accuracies recorded for the %i evaluations with %s\n'\
                                                                %(nPS,toolname))
        logFile.write('First row is DP, second is QP (if available).\n\n')
        logFile.writelines('%.3e  '%DP_stability[i]+('NA\n' if QP_stability[i]==-1.0 \
                             else '%.3e\n'%QP_stability[i]) for i in range(nPS))
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
        colorlist=['b','r','g','y']
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
    res_str += "\n= Unpolarized time / PSpoint. ========== %.3gms\n"\
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

    res_str += "\n= Polarized time / PSpoint... ========== %.3gms\n"\
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
            if not amplitude.get('process').get('has_born'):
                evaluator.loop_optimized_output = False
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

    #MLOptions = {'ImprovePS':True,'ForceMP':True}

    #brsvalue = evaluator.evaluate_matrix_element(matrix_element, p=p, gauge_check = True,
    #                                             output='jamp',MLOptions=MLOptions)
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
            if not amplitude.get('process').get('has_born'):
                evaluator.loop_optimized_output = False 
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
            if not amplitude.get('process').get('has_born'):
                evaluator.loop_optimized_output = False
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
        diff = (max_val - min_val) / max_val 
        
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





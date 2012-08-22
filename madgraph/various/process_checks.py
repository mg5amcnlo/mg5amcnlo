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
import re
import shutil
import glob
import re
import subprocess
import time
import datetime

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


import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects

from madgraph import MG5DIR, InvalidCmd

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx, txxxxx, irxxxx, orxxxx

ADDED_GLOBAL = []

loop_optimized_output = False

def clean_added_globals(to_clean):
    for value in list(to_clean):
        del globals()[value]
        to_clean.remove(value)

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.process_checks')

#===============================================================================
# Helper class MatrixElementEvaluator
#===============================================================================
class MatrixElementEvaluator(object):
    """Class taking care of matrix element evaluation, storing
    relevant quantities for speedup."""

    def __init__(self, model, param_card = None,
                 auth_skipping = False, reuse = True, cmass_scheme = False):
        """Initialize object with stored_quantities, helas_writer,
        model, etc.
        auth_skipping = True means that any identical matrix element will be
                        evaluated only once
        reuse = True means that the matrix element corresponding to a
                given process can be reused (turn off if you are using
                different models for the same process)"""
 
        # Writer for the Python matrix elements
        self.helas_writer = helas_call_writers.PythonUFOHelasCallWriter(model)
    
        # Read a param_card and calculate couplings
        self.full_model = model_reader.ModelReader(model)
        self.full_model.set_parameters_and_couplings(param_card)

        self.auth_skipping = auth_skipping
        self.reuse = reuse
        self.cmass_scheme = cmass_scheme
        self.store_aloha = []
        self.stored_quantities = {}
        
    #===============================================================================
    # Helper function evaluate_matrix_element
    #===============================================================================
    def evaluate_matrix_element(self, matrix_element, p=None, full_model=None, 
                                gauge_check=False, auth_skipping=None, output='m2'):
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

        if self.reuse:
            # Define the routines (globally)
            exec(matrix_methods[process.shell_string()], globals())
            ADDED_GLOBAL.append('Matrix_%s'  % process.shell_string())
        else:
            # Define the routines (locally is enough)
            exec(matrix_methods[process.shell_string()])

        # Generate phase space point to use
        if not p:
            p, w_rambo = self.get_momenta(process)

        # Evaluate the matrix element for the momenta p
        exec("data = Matrix_%s()" % process.shell_string())
        if output == "m2": 
            return data.smatrix(p, self.full_model), data.amp2
        else:
            m2 = data.smatrix(p, self.full_model)
            return {'m2': m2, output:getattr(data, output)}
    
    #===============================================================================
    # Helper function get_momenta
    #===============================================================================
    def get_momenta(self, process, energy = 1000.):
        """Get a point in phase space for the external states in the given
        process, with the CM energy given. The incoming particles are
        assumed to be oriented along the z axis, with particle 1 along the
        positive z axis."""

        if not (isinstance(process, base_objects.Process) and \
                isinstance(energy, float)):
            raise rambo.RAMBOError, "Not correct type for arguments to get_momenta"

        sorted_legs = sorted(process.get('legs'), lambda l1, l2:\
                             l1.get('number') - l2.get('number'))

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

        e2 = energy**2
        m1 = mass[0]

        p = []

        masses = rambo.FortranList(nfinal)
        for i in range(nfinal):
            masses[i+1] = mass[nincoming + i]
        
        if nincoming == 1:

            # Momenta for the incoming particle
            p.append([m1, 0., 0., 0.])

            p_rambo, w_rambo = rambo.RAMBO(nfinal, m1, masses)

            # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
            for i in range(1, nfinal+1):
                momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                        p_rambo[(2,i)], p_rambo[(3,i)]]
                p.append(momi)

            return p, w_rambo

        if nincoming != 2:
            raise rambo.RAMBOError('Need 1 or 2 incoming particles')

        if nfinal == 1:
            energy = masses[0]

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

    def __init__(self, mg_root=None, cuttools_dir=None, *args, **kwargs):
        """Allow for initializing the MG5 root where the temporary fortran
        output for checks is placed."""
        
        super(LoopMatrixElementEvaluator,self).__init__(*args, **kwargs)
        
        self.mg_root=mg_root
        self.cuttools_dir=cuttools_dir
        # Set proliferate to true if you want to keep the produced directories
        # and eventually reuse them if possible
        self.proliferate=True
        
    #===============================================================================
    # Helper function evaluate_matrix_element for loops
    #===============================================================================
    def evaluate_matrix_element(self, matrix_element, p=None, full_model=None, 
                                gauge_check=False, auth_skipping=None, output='m2'):
        """Calculate the matrix element and evaluate it for a phase space point
           Output can only be 'm2. The 'jamp' and 'amp' returned values are just
           empty lists at this point.
        """
        if full_model:
            self.full_model = full_model

        process = matrix_element.get('processes')[0]
        model = process.get('model')

        # For now, only accept the process if it has a born
        if not matrix_element.get('processes')[0]['has_born']:
            if output == "m2":
                return 0.0, []
            else:
                return {'m2': 0.0, output:[]}
            
        if "matrix_elements" not in self.stored_quantities:
            self.stored_quantities['matrix_elements'] = []

        if (auth_skipping or self.auth_skipping) and matrix_element in \
                [el[0] for el in self.stored_quantities['matrix_elements']]:
            # Exactly the same matrix element has been tested
            logger.info("Skipping %s, " % process.nice_string() + \
                        "identical matrix element already tested" \
                        )
            return None

        # Generate phase space point to use
        if not p:
            p, w_rambo = self.get_momenta(process)
        
        if matrix_element in [el[0] for el in \
                                     self.stored_quantities['matrix_elements']]:  
            export_dir=self.stored_quantities['matrix_elements'][\
                [el[0] for el in self.stored_quantities['matrix_elements']\
                 ].index(matrix_element)][1]
            logger.debug("Reusing generated output %s"%str(export_dir))
        else:        
            export_dir=os.path.join(self.mg_root,'TMP_DIR_FOR_THE_CHECK_CMD')
            if os.path.isdir(export_dir):
                if not self.proliferate:
                    raise InvalidCmd("The directory %s already exist. Please remove it."%str(export_dir))
                else:
                    id=1
                    while os.path.isdir(os.path.join(self.mg_root,\
                                        'TMP_DIR_FOR_THE_CHECK_CMD_%i'%id)):
                        id+=1
                    export_dir=os.path.join(self.mg_root,'TMP_DIR_FOR_THE_CHECK_CMD_%i'%id)
            
            if self.proliferate:
                self.stored_quantities['matrix_elements'].append(\
                                                    (matrix_element,export_dir))

            # I do the import here because there is some cyclic import of export_v4
            # otherwise
            import madgraph.loop.loop_exporters as loop_exporters
            if loop_optimized_output:
                exporter=loop_exporters.LoopProcessOptimizedExporterFortranSA
            else:
                exporter=loop_exporters.LoopProcessExporterFortranSA

            FortranExporter = exporter(\
                self.mg_root, export_dir, clean=True,
                complex_mass_scheme = self.cmass_scheme, mp=True,
                loop_dir=os.path.join(self.mg_root, 'Template/loop_material'),\
                cuttools_dir=self.cuttools_dir)
            FortranModel = helas_call_writers.FortranUFOHelasCallWriter(model)
            FortranExporter.copy_v4template(modelname=model.get('name'))
            FortranExporter.generate_subprocess_directory_v4(matrix_element, FortranModel)
            wanted_lorentz = list(set(matrix_element.get_used_lorentz()))
            wanted_couplings = list(set([c for l in matrix_element.get_used_couplings() \
                                                                    for c in l]))
            FortranExporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
            FortranExporter.finalize_v4_directory(None,"",False,False,'gfortran')

        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'))
        self.fix_MadLoopParamCard(os.path.join(export_dir,'SubProcesses'),
                                     mp = gauge_check and loop_optimized_output)
        
        if gauge_check:
            file_path, orig_file_content, new_file_content = \
              self.setup_ward_check(os.path.join(export_dir,'SubProcesses'), 
                                       ['helas_calls_ampb_1.f','loop_matrix.f'])
            file = open(file_path,'w')
            file.write(new_file_content)
            file.close()
            if loop_optimized_output:
                mp_file_path, mp_orig_file_content, mp_new_file_content = \
                  self.setup_ward_check(os.path.join(export_dir,'SubProcesses'), 
                  ['mp_helas_calls_ampb_1.f','mp_compute_loop_coefs.f'],mp=True)
                mp_file = open(mp_file_path,'w')
                mp_file.write(mp_new_file_content)
                mp_file.close()

        # Evaluate the matrix element for the momenta p        
        finite_m2 = self.get_me_value(process.shell_string_v4(), 0,\
                                               export_dir, p,verbose=False)[0][0]

        # Restore the original loop_matrix.f code so that it could be reused
        if gauge_check:
            file = open(file_path,'w')
            file.write(orig_file_content)
            file.close()
            if loop_optimized_output:
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

    def fix_MadLoopParamCard(self,dir_name, mp=False, loop_filter=False):
        """ Set parameters in MadLoopParams.dat suited for these checks.MP
            stands for multiple precision and can either be a bool or an integer
            to specify the mode."""

        if isinstance(mp,bool):
            mode = 4 if mp else 1
        else:
            mode = mp
        file = open(os.path.join(dir_name,'MadLoopParams.dat'), 'r')
        MLParams = file.read()
        file.close()
        file = open(os.path.join(dir_name,'MadLoopParams.dat'), 'w')
        MLParams = re.sub(r"#CTModeRun\n-?\d+","#CTModeRun\n%d"%mode, MLParams)
        MLParams = re.sub(r"#CTModeInit\n-?\d+","#CTModeInit\n%d"%mode, MLParams)
        MLParams = re.sub(r"#UseLoopFilter\n\S+","#UseLoopFilter\n%s"%(\
                               '.TRUE.' if loop_filter else '.FALSE.'),MLParams)                
        MLParams = re.sub(r"#DoubleCheckHelicityFilter\n\S+",
                                 "#DoubleCheckHelicityFilter\n.FALSE.",MLParams)
        file.write(MLParams)
        file.close()

    def fix_PSPoint_in_check(self, dir_name, read_ps = True, npoints = 1,
                             hel_config = -1):
        """Set check_sa.f to be reading PS.input assuming a working dir dir_name.
        if hel_config is different than -1 then check_sa.f is configured so to
        evaluate only the specified helicity."""

        file = open(os.path.join(dir_name,'check_sa.f'), 'r')
        check_sa = file.read()
        file.close()
        
        file = open(os.path.join(dir_name,'check_sa.f'), 'w')
        check_sa = re.sub(r"READPS = \S+\)","READPS = %s)"%('.TRUE.' if read_ps \
                                                      else '.FALSE.'), check_sa)
        check_sa = re.sub(r"NPSPOINTS = \d+","NPSPOINTS = %d"%npoints, check_sa)
        if hel_config != -1:
            check_sa = re.sub(r"SLOOPMATRIX\S+\)","SLOOPMATRIXHEL(P,%d,MATELEM)"\
                                                          %hel_config, check_sa)
        else:
            check_sa = re.sub(r"SLOOPMATRIX\S+\)","SLOOPMATRIX(P,MATELEM)",\
                                                                       check_sa)
        file.write(check_sa)
        file.close()

    def get_me_value(self, proc, proc_id, working_dir, PSpoint=[], verbose=True):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id and PSpoint if specified."""  
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir, 'SubProcesses',
                                  'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" %proc)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        if verbose: logging.debug("Working on process %s in dir %s" % (proc, shell_name))
        
        dir_name = os.path.join(working_dir, 'SubProcesses', shell_name)
        # Make sure to recreate the executable and modified sources
        if os.path.isfile(os.path.join(dir_name,'check')):
            os.remove(os.path.join(dir_name,'check'))
            try:
                os.remove(os.path.join(dir_name,'check_sa.o'))
                os.remove(os.path.join(dir_name,'loop_matrix.o'))
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
            PSfile = open(os.path.join(dir_name, 'PS.input'), 'w')
            PSfile.write('\n'.join([' '.join(['%.16E'%pi for pi in p]) \
                                  for p in PSpoint]))
            PSfile.close()
        
        # Run ./check
        try:
            output = subprocess.Popen('./check',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(dir_name,'result.dat')):
                return self.parse_check_output(file(os.path.join(dir_name,\
                                                                 'result.dat')))  
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                                  .join(dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

    def parse_check_output(self,output):
        """Parse the output string and return a pair where first four values are 
        the finite, born, single and double pole of the ME and the fourth is the
        GeV exponent and the second value is a list of 4 momenta for all particles 
        involved."""

        res_p = []
        value = [0.0,0.0,0.0,0.0]
        gev_pow = 0

        for line in output:
            splitline=line.split()
            if splitline[0]=='PS':
                res_p.append([float(s) for s in splitline[1:]])
            elif splitline[0]=='BORN':
                value[1]=float(splitline[1])
            elif splitline[0]=='FIN':
                value[0]=float(splitline[1])
            elif splitline[0]=='1EPS':
                value[2]=float(splitline[1])
            elif splitline[0]=='2EPS':
                value[3]=float(splitline[1])
            elif splitline[0]=='EXP':
                gev_pow=int(splitline[1])

        return ((value[0],value[1],value[2],value[3],gev_pow), res_p)
    
    def setup_ward_check(self, working_dir, file_names, mp = False):
        """ Modify loop_matrix.f so to have one external massless gauge boson
        polarization vector turned into its momentum. It is not a pretty and 
        flexible solution but it works for this particular case."""
        
        shell_name = None
        directories = glob.glob(os.path.join(working_dir,'P0_*'))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])
        
        dir_name = os.path.join(working_dir, shell_name)
        
        # Look, in order, for all the possible file names provided.
        ind=0
        while ind<len(file_names) and not os.path.isfile(os.path.join(dir_name,
                                                              file_names[ind])):
            ind += 1
        if ind==len(file_names):
            raise Exception, "No helas calls output file found."
        
        helas_file_name=os.path.join(dir_name,file_names[ind])  
        file = open(os.path.join(dir_name,helas_file_name), 'r')
        
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
        
        return os.path.join(dir_name,helas_file_name), original_file, helas_calls_out

#===============================================================================
# Helper class LoopMatrixElementEvaluator
#===============================================================================
class LoopMatrixElementTimer(LoopMatrixElementEvaluator):
    """Class taking care of matrix element evaluation and running timing for 
       loop processes."""
   
    def __init__(self, *args, **kwargs):
        """ Same as the mother for now """
        LoopMatrixElementEvaluator.__init__(self,*args, **kwargs)

    def make_and_run(self, dir_name):
        """ Compile the check program in the directory dir_name.
        Return the compilation and running time. """

        # Make sure to recreate the executable and modified source
        # (The time stamps are sometimes not actualized if it is too fast)
        if os.path.isfile(os.path.join(dir_name,'check')):
            os.remove(os.path.join(dir_name,'check'))
            os.remove(os.path.join(dir_name,'check_sa.o'))
            os.remove(os.path.join(dir_name,'loop_matrix.o'))            
        # Now run make
        devnull = open(os.devnull, 'w')
        start=time.time()
        retcode = subprocess.call(['make','check'],
                                   cwd=dir_name, stdout=devnull, stderr=devnull)
        compilation_time = time.time()-start
                     
        if retcode != 0:
            logging.info("Error while executing make in %s" % dir_name)
            return None, None

        start=time.time()
        retcode = subprocess.call('./check',
                                   cwd=dir_name, stdout=devnull, stderr=devnull)
        run_time = time.time()-start
        devnull.close()
        
        if retcode != 0:
            logging.warning("Error while executing ./check in %s" % dir_name)
            return None, None

        return compilation_time, run_time
        
    def skip_loop_evaluation_setup(self, dir_name, skip=True):
        """ Edit loop_matrix.f in order to skip the loop evaluation phase.
        Notice this only affects the double precision evaluation which is
        normally fine as we do not make the timing check on mp."""

        file = open(os.path.join(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(os.path.join(dir_name,'loop_matrix.f'), 'w')
        loop_matrix = re.sub(r"SKIPLOOPEVAL=\S+\)","SKIPLOOPEVAL=%s)"%('.TRUE.' 
                                           if skip else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def boot_time_setup(self, dir_name, bootandstop=True):
        """ Edit loop_matrix.f in order to set the flag which stops the
        execution after booting the program (i.e. reading the color data)."""

        file = open(os.path.join(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(os.path.join(dir_name,'loop_matrix.f'), 'w')        
        loop_matrix = re.sub(r"BOOTANDSTOP=\S+\)","BOOTANDSTOP=%s)"%('.TRUE.' 
                                    if bootandstop else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def setup_process(self, matrix_element, model, export_dir):
        """ Output the matrix_element in argument and perform the initialization
        while providing some details about the output in the dictionary returned. 
        Returns None if anything fails"""

        infos={}

        if os.path.isdir(export_dir):
            raise InvalidCmd(\
            "The directory %s already exist. Please remove it."%str(export_dir))

        # I do the import here because there is some cyclic import of export_v4
        # otherwise
        import madgraph.loop.loop_exporters as loop_exporters
        if loop_optimized_output:
            exporter=loop_exporters.LoopProcessOptimizedExporterFortranSA
        else:
            exporter=loop_exporters.LoopProcessExporterFortranSA
            
        start=time.time()
        FortranExporter = exporter(\
            self.mg_root, export_dir, clean=True,
            complex_mass_scheme = self.cmass_scheme, mp=True,
            loop_dir=os.path.join(self.mg_root, 'Template/loop_material'),\
            cuttools_dir=self.cuttools_dir)
        FortranModel = helas_call_writers.FortranUFOHelasCallWriter(model)
        FortranExporter.copy_v4template(modelname=model.get('name'))
        FortranExporter.generate_subprocess_directory_v4(matrix_element, FortranModel)
        wanted_lorentz = list(set(matrix_element.get_used_lorentz()))
        wanted_couplings = list(set([c for l in matrix_element.get_used_couplings() \
                                                                for c in l]))
        FortranExporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
        infos['Process_output'] = time.time()-start
        start=time.time()
        FortranExporter.finalize_v4_directory(None,"",False,False,'gfortran')
        infos['HELAS_MODEL_compilation'] = time.time()-start
        
        # First Initialize filters (in later versions where this will be done
        # at generation time, it can be skipped)
        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
                                                   read_ps = False, npoints = 4)
        self.fix_MadLoopParamCard(os.path.join(export_dir,'SubProcesses'),
                                                 mp = False, loop_filter = True)
        
        shell_name = None
        directories = glob.glob(os.path.join(export_dir, 'SubProcesses','P0_*'))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])
        dir_name = os.path.join(export_dir, 'SubProcesses', shell_name)
        infos['dir_path']=dir_name

        compile_time, run_time = self.make_and_run(dir_name)
        if compile_time==None:
            logging.error("Failed at running the process %s."%shell_name)
            return None
        
        if not os.path.exists(os.path.join(dir_name,'HelFilter.dat')) or not \
               os.path.exists(os.path.join(dir_name,'LoopFilter.dat')):
            logger.warning("Could not initialize the process %s"%shell_name+\
                            " with 4 PS points. Now trying with 15 PS points.")
            self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
                                                  read_ps = False, npoints = 15)
            compile_time, run_time = self.make_and_run(dir_name)
            if compile_time == None: return None
            if not os.path.exists(os.path.join(dir_name,'HelFilter.dat')) or \
               not os.path.exists(os.path.join(dir_name,'LoopFilter.dat')):
                logger.error("Could not initialize the process %s"%shell_name+\
                            " with 15 PS points.")
                return None
        
        infos['Initialization'] = run_time
        infos['Process_compilation'] = compile_time

        return infos

    def time_matrix_element(self, matrix_element, model):
        """ Output the matrix_element in argument and give detail information
        about the timing for its output and running"""
        
        # Normally, this should work for loop-induced processes as well
#        if not matrix_element.get('processes')[0]['has_born']:
#            return None
        proc_name = matrix_element['processes'][0].shell_string()
        export_dir=os.path.join(self.mg_root,'TMP_DIR_FOR_THE_CHECK_CMD')
        res_timings = self.setup_process(matrix_element, model,export_dir)
        if not res_timings:
            return None
        dir_name=infos['dir_path']

        def check_disk_usage(path):
            return subprocess.check_output(["du -shc %s"%path],shell=True).\
                                                                     split()[-2]

        res_timings['du_source']=check_disk_usage(os.path.join(\
                                                 export_dir,'Source','*','*.f'))
        res_timings['du_process']=check_disk_usage(os.path.join(dir_name,'*.f'))
        res_timings['du_color']=check_disk_usage(os.path.join(dir_name,'*.dat'))
        res_timings['du_exe']=check_disk_usage(os.path.join(dir_name,'check'))

        time_per_ps_estimate = (res_timings['Initialization']/4.0)/2.0
        
        self.boot_time_setup(dir_name,bootandstop=True)
        compile_time, run_time = self.make_and_run(dir_name)
        res_timings['Booting_time'] = run_time
        self.boot_time_setup(dir_name,bootandstop=False)

        # Detect one contributing helicity
        contributing_hel=0
        n_contrib_hel=0
        helicities = file(os.path.join(dir_name,'HelFilter.dat')).read().split()
        for i, hel in enumerate(helicities):
            if (loop_optimized_output and int(hel)>-10000) or hel=='T':
                if contributing_hel==0:
                    contributing_hel=i+1
                n_contrib_hel += 1
                    
        if contributing_hel==0:
            logger.error("Could not find a contributing helicity "+\
                                     "configuration for process %s."%proc_name)
            return None
        
        res_timings['n_contrib_hel']=n_contrib_hel
        res_timings['n_tot_hel']=len(helicities)
        
        # We aim at a 15 sec run
        target_pspoints_number = max(int(15.0/time_per_ps_estimate)+1,10)

        logger.info("Checking timing for process %s "%proc_name+\
                                    "with %d PS points."%target_pspoints_number)
        
        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number*2, \
                                                  hel_config = contributing_hel)
        compile_time, run_time = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_polarized_total']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)

        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number, hel_config = -1)
        compile_time, run_time = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_unpolarized_total']=\
                   (run_time-res_timings['Booting_time'])/target_pspoints_number
        
        if not loop_optimized_output:
            return res_timings
        
        # For the loop optimized output, we also check the time spent in
        # computing the coefficients of the loop numerator polynomials.
        
        # So we modify loop_matrix.f in order to skip the loop evaluation phase.
        self.skip_loop_evaluation_setup(dir_name,skip=True)

        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number, hel_config = -1)
        compile_time, run_time = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_unpolarized_coefs']=\
                   (run_time-res_timings['Booting_time'])/target_pspoints_number
        
        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
             read_ps = False, npoints = target_pspoints_number*2, \
                                                  hel_config = contributing_hel)
        compile_time, run_time = self.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_polarized_coefs']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)    

        # Restitute the original file.
        self.skip_loop_evaluation_setup(dir_name,skip=False)
        
        return res_timings

#===============================================================================
# Global helper function run_multiprocs
#===============================================================================

    def check_matrix_element_stability(self, matrix_element, model, nPoints):
        """ Output the matrix_element in argument, run in for nPoints and return
        a dictionary containing the stability information on each of these points
        """
        
        # Accuracy threshold of double precision evaluations above which the
        # PS points is also evaluated in quadruple preicsion
        accuracy_threshold=1.0e-3
        
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
        
        # Normally, this should work for loop-induced processes as well
#        if not matrix_element.get('processes')[0]['has_born']:
#            return None
        
        process = matrix_element['processes'][0]
        proc_name = process.shell_string()
        export_dir=os.path.join(self.mg_root,'TMP_DIR_FOR_THE_CHECK_CMD')
        infos = self.setup_process(matrix_element, model,export_dir)
        if not infos:
            return None
        dir_path=infos['dir_path']
        
        logger.info("Checking stability of process %s "%proc_name+\
                    "with %d PS points."%nPoints)
        time_per_ps_estimate = (infos['Initialization']/4.0)/2.0
        sec_needed = int(time_per_ps_estimate*nPoints*2)
        if sec_needed>15:     
            logger.info("This check should take about "+\
                        "%s to run. Started on %s."%(\
                            str(datetime.timedelta(seconds=sec_needed)),\
                            datetime.datetime.now().strftime("%d-%m-%Y %H:%M")))
        self.fix_PSPoint_in_check(os.path.join(export_dir,'SubProcesses'),
                                   read_ps = True, npoints = 1, hel_config = -1)
        # Recompile (Notice that the recompilation is only necessary once) for
        # the change above to take effect.
        # Make sure to recreate the executable and modified sources
        try:
            os.remove(os.path.join(dir_path,'check'))
            os.remove(os.path.join(dir_path,'check_sa.o'))
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
        
        def pick_PS_point(proc):
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
            p, w_rambo = self.get_momenta(proc)
            while not pass_cuts(p):
                p, w_rambo = self.get_momenta(proc)
            return p                
        
        for i in range(nPoints):
            # Pick an eligible PS point with rambo
            p = pick_PS_point(process)
#            print "I use P_%i="%i,p
            # Write it in the input file
            PSfile = open(os.path.join(dir_path, 'PS.input'), 'w')
            PSfile.write('\n'.join([' '.join(['%.16E'%pi for pi in pmom]) \
                                                                for pmom in p]))
            PSfile.close()
            dp_res=[]
            dp_res.append(self.get_me_value(1,\
                              os.path.join(export_dir,'SubProcesses'),dir_path))
            dp_res.append(self.get_me_value(2,\
                              os.path.join(export_dir,'SubProcesses'),dir_path))
            # Make sure all results make sense
            if any([not res for res in dp_res]):
                return None
            
            dp_accuracy = (max(dp_res)-min(dp_res))/abs(sum(dp_res)/len(dp_res))
            DP_stability.append({'CTMode1':dp_res[0],'CTMode2':dp_res[1],
                                                        'Accuracy':dp_accuracy})
            if dp_accuracy>accuracy_threshold:
                Unstable_PS_points.append([i,p])
                qp_res=[]
                qp_res.append(self.get_me_value(4,\
                                  os.path.join(export_dir,'SubProcesses'),dir_path))
                qp_res.append(self.get_me_value(5,\
                                  os.path.join(export_dir,'SubProcesses'),dir_path))
                # Make sure all results make sense
                if any([not res for res in qp_res]):
                    return None
                
                qp_accuracy = (max(qp_res)-min(qp_res))/abs(sum(qp_res)/len(qp_res))
                QP_stability.append({'CTMode4':qp_res[0],'CTMode5':qp_res[1],
                                                        'Accuracy':qp_accuracy})
                if qp_accuracy>accuracy_threshold:
                    Exceptional_PS_points.append([i,p])
            else:
                QP_stability.append({})
        
        return {'DP_stability':DP_stability,
                'QP_stability':QP_stability,
                'Unstable_PS_points':Unstable_PS_points,
                'Exceptional_PS_points':Exceptional_PS_points}

    def get_me_value(self,mode,param_card_path,dir_path):
        """ This version of get_me_value is simplified for the purpose of this
        class. No compilation is necessary. The CT mode can be specified."""

        self.fix_MadLoopParamCard(param_card_path,mp=mode,loop_filter=True)
        try:
            output = subprocess.Popen('./check',
                        cwd=dir_path,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(dir_path,'result.dat')):
                return self.parse_check_output(file(os.path.join(dir_path,\
                                                           'result.dat')))[0][0]
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                                  .join(dir_path,'result.dat')))
                return None
        except IOError:
            logging.warning("Error while executing ./check in %s"%str(dir_path))
            return None    

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
               auth_skipping = False, reuse = True, cmass_scheme = cmass_scheme)
    
    amplitude = diagram_generation.Amplitude(newproc)
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
                                opt=None):
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
            process = base_objects.Process({\
                'legs': base_objects.LegList(\
                        [base_objects.Leg({'id': id, 'state':False}) for \
                         id in is_prod] + \
                        [base_objects.Leg({'id': id, 'state':True}) for \
                         id in fs_prod]),
                'model':multiprocess.get('model'),
                'id': multiprocess.get('id'),
                'orders': multiprocess.get('orders'),
                'required_s_channels': \
                              multiprocess.get('required_s_channels'),
                'forbidden_s_channels': \
                              multiprocess.get('forbidden_s_channels'),
                'forbidden_particles': \
                              multiprocess.get('forbidden_particles'),
                'perturbation_couplings': \
                              multiprocess.get('perturbation_couplings'),
                'is_decay_chain': \
                              multiprocess.get('is_decay_chain'),
                'overall_orders': \
                              multiprocess.get('overall_orders')})
            if opt is not None:
                if isinstance(opt, dict):
                    try:
                        value = opt[process.base_string()]
                    except:
                        continue
                    result = function(process, stored_quantities, value)
                else:
                    result = function(process, stored_quantities, opt)
            else:
                result = function(process, stored_quantities)
                        
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
# check_timing for loop processes
#===============================================================================
def check_stability(process_definition, mg_root="",cuttools="",
                    cmass_scheme = False, nPoints=100):
    """For a single loop process, give a detailed summary of the generation and
    execution timing."""
    
    assert isinstance(process_definition,base_objects.ProcessDefinition)
    assert process_definition.get('perturbation_couplings')!=[]

    model=process_definition.get('model')
    
    if any(len(l.get('ids'))>1 for l in process_definition.get('legs')):
        raise InvalidCmd("The stability check can only be performed on single "+
                         " processes. (i.e. without multiparticle labels).")

    isids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
              if not leg.get('state')]
    fsids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
             if leg.get('state')]

    # Now generate a process based on the ProcessDefinition given in argument.
    process = process_definition.get_process(isids,fsids)

    logger.info("Generating p%s"%process_definition.nice_string()[1:])

    amplitude = loop_diagram_generation.LoopAmplitude(process)
    matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                        optimized_output = loop_optimized_output,gen_color=True)

    myStabilityChecker = LoopMatrixElementTimer(mg_root=mg_root, 
                           cuttools_dir=cuttools, model=model, 
                           cmass_scheme = cmass_scheme)
    
    stability = myStabilityChecker.check_matrix_element_stability(matrix_element, 
                                                          model,nPoints=nPoints)
    
    if stability == None:
        return None
    else:
        clean_up(mg_root)
        stability['Process']=process
        return stability

#===============================================================================
# check_timing for loop processes
#===============================================================================
def check_timing(process_definition, mg_root="",cuttools="",cmass_scheme = False):
    """For a single loop process, give a detailed summary of the generation and
    execution timing."""
    
    assert isinstance(process_definition,base_objects.ProcessDefinition)
    assert process_definition.get('perturbation_couplings')!=[]
    
    model=process_definition.get('model')
    
    if any(len(l.get('ids'))>1 for l in process_definition.get('legs')):
        raise InvalidCmd("The timing check can only be performed on single "+
                         " processes. (i.e. without multiparticle labels).")

    isids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
              if not leg.get('state')]
    fsids = [leg.get('ids')[0] for leg in process_definition.get('legs') \
             if leg.get('state')]

    # Now generate a process based on the ProcessDefinition given in argument.
    process = process_definition.get_process(isids,fsids)
    
    logger.info("Generating p%s"%process_definition.nice_string()[1:])
    
    timing1 = {}
    start=time.time()
    amplitude = loop_diagram_generation.LoopAmplitude(process)
    timing1['Diagrams_generation']=time.time()-start
    timing1['n_loops']=len(amplitude.get('loop_diagrams'))
    start=time.time()
    matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                        optimized_output = loop_optimized_output,gen_color=True)
    timing1['HelasDiagrams_generation']=time.time()-start
    
    if loop_optimized_output:
        timing1['n_loop_groups']=len(matrix_element.get('loop_groups'))
        lwfs=[l for ldiag in matrix_element.get_loop_diagrams() for l in \
                                                ldiag.get('loop_wavefunctions')]
        timing1['n_loop_wfs']=len(lwfs)
        timing1['loop_wfs_ranks']=[]
        for rank in range(0,max([l.get('rank') for l in lwfs])+1):
            timing1['loop_wfs_ranks'].append(\
                                  len([1 for l in lwfs if l.get('rank')==rank]))
    
    myTimer = LoopMatrixElementTimer(mg_root=mg_root, cuttools_dir=cuttools, 
                                       model=model, cmass_scheme = cmass_scheme)
    timing2 = myTimer.time_matrix_element(matrix_element, model)
    
    if timing2 == None:
        return None
    else:
        clean_up(mg_root)
    
    # Return the merged two dictionaries
    return dict(timing1.items()+timing2.items())

#===============================================================================
# check_processes
#===============================================================================

def check_processes(processes, param_card = None, quick = [],
                    mg_root="",cuttools="",cmass_scheme = False):
    """Check processes by generating them with all possible orderings
    of particles (which means different diagram building and Helas
    calls), and comparing the resulting matrix element values."""


    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes
        model = multiprocess.get('model')

        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model,
               auth_skipping = True, reuse = False, cmass_scheme = cmass_scheme)
        else:
            evaluator = LoopMatrixElementEvaluator(mg_root=mg_root,
                            cuttools_dir=cuttools, model=model, auth_skipping = True,
                            reuse = False, cmass_scheme = cmass_scheme)
       
        results = run_multiprocs_no_crossings(check_process,
                                              multiprocess,
                                              evaluator,
                                              quick)

        if "used_lorentz" not in evaluator.stored_quantities:
            evaluator.stored_quantities["used_lorentz"] = []
            
        if multiprocess.get('perturbation_couplings')!=[]:
            # Clean temporary folders created for the running of the loop processes
            clean_up(mg_root)
            
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
               auth_skipping = True, reuse = False, cmass_scheme = cmass_scheme)
    else:
        evaluator = LoopMatrixElementEvaluator(mg_root=mg_root, 
                                           cuttools_dir=cuttools, model=model,
                                           param_card=param_card,
                                           auth_skipping = True, reuse = False,
                                           cmass_scheme = cmass_scheme)

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
        res = check_process(process, evaluator, quick)
        if res:
            comparison_results.append(res)

    if "used_lorentz" not in evaluator.stored_quantities:
        evaluator.stored_quantities["used_lorentz"] = []
    
    if processes[0].get('perturbation_couplings')!=[]:
        # Clean temporary folders created for the running of the loop processes
        clean_up(mg_root)    
    
    return comparison_results, evaluator.stored_quantities["used_lorentz"]

def check_process(process, evaluator, quick):
    """Check the helas calls for a process by generating the process
    using all different permutations of the process legs (or, if
    quick, use a subset of permutations), and check that the matrix
    element is invariant under this."""

    model = process.get('model')

    # Ensure that leg numbers are set
    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking %s" % \
                process.nice_string().replace('Process', 'process'))

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

        newproc = base_objects.Process({'legs':legs,
            'orders':copy.copy(process.get('orders')),
            'model':process.get('model'),
            'id':copy.copy(process.get('id')),
            'uid':process.get('uid'),
            'required_s_channels':copy.copy(process.get('required_s_channels')),
            'forbidden_s_channels':copy.copy(process.get('forbidden_s_channels')),
            'forbidden_particles':copy.copy(process.get('forbidden_particles')),
            'is_decay_chain':process.get('is_decay_chain'),
            'overall_orders':copy.copy(process.get('overall_orders')),
            'decay_chains':process.get('decay_chains'),
            'perturbation_couplings':copy.copy(process.get('perturbation_couplings')),
            'squared_orders':copy.copy(process.get('squared_orders')),
            'has_born':process.get('has_born')})

        # Generate the amplitude for this process
        try:
            if newproc.get('perturbation_couplings')==[]:
                amplitude = diagram_generation.Amplitude(newproc)
            else:
                amplitude = loop_diagram_generation.LoopAmplitude(newproc)                
        except InvalidCmd:
            result=False
        else:
            result = amplitude.get('diagrams')

        if not result:
            # This process has no diagrams; go to next process
            logging.info("No diagrams for %s" % \
                         process.nice_string().replace('Process', 'process'))
            break

        if order == range(1,len(legs) + 1):
            # Generate phase space point to use
            p, w_rambo = evaluator.get_momenta(process)

        # Generate the HelasMatrixElement for the process
        if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
            matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                          gen_color=False)
        else:
            matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                                         optimized_output=loop_optimized_output)

        if matrix_element in process_matrix_elements:
            # Exactly the same matrix element has been tested
            # for other permutation of same process
            continue

        process_matrix_elements.append(matrix_element)

        res = evaluator.evaluate_matrix_element(matrix_element, p = p)
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

    directories = glob.glob(os.path.join(mg_root, 'TMP_DIR_FOR_THE_CHECK_CMD*'))
    for dir in directories:
        shutil.rmtree(dir)

def output_stability(stability, mg_root):
    """Present the result of a stability check in a nice format.
    The full info is printed out in 'Stability_result_<proc_shell_string>.dat'
    under the MadGraph root folder (mg_root)"""
    
#    return "Stability="+str(stability)
    DP_stability = [eval['Accuracy'] for eval in stability['DP_stability']]
    # Remember that an evaluation which did not require QP has an empty dictionary
    QP_stability = [eval['Accuracy'] if eval!={} else -1.0 for eval in \
                                                      stability['QP_stability']]
    process = stability['Process']
    nPS = len(DP_stability)
    UPS = stability['Unstable_PS_points']
    UPS_stability_DP = [DP_stability[U[0]] for U in UPS]
    UPS_stability_QP = [QP_stability[U[0]] for U in UPS]
    EPS = stability['Exceptional_PS_points']
    EPS_stability_DP = [DP_stability[E[0]] for E in EPS]
    EPS_stability_QP = [QP_stability[E[0]] for E in EPS]
    
    res_str = "%i PS points evaluated for %s \n"%(nPS,process.nice_string())
    res_str += "\n= Double precision results\n"
    res_str += "|= Average accuracy............ %.3e\n"%(sum(DP_stability)/nPS)
    res_str += "|= Max accuracy................ %.3e\n"%min(DP_stability)
    res_str += "|= Min accuracy................ %.3e\n"%max(DP_stability)
    res_str += "\n= Number of Unstable PS points    : %i\n"%len(UPS)
    if len(UPS)>0:
        res_str += "|= DP Average inaccuracy....... %.3e\n"%\
                                                (sum(UPS_stability_DP)/len(UPS))
        res_str += "|= DP Max accuracy............. %.3e\n"%min(UPS_stability_DP)
        res_str += "|= DP Min accuracy............. %.3e\n"%max(UPS_stability_DP)
        res_str += "|= QP Average accuracy......... %.3e\n"%\
                                                (sum(UPS_stability_QP)/len(UPS))
        res_str += "|= QP Max accuracy............. %.3e\n"%min(UPS_stability_QP)
        res_str += "|= QP Min accuracy............. %.3e\n"%max(UPS_stability_QP)
    res_str += "\n= Number of Exceptional PS points : %i\n"%len(EPS)
    if len(EPS)>0:
        res_str += "|= DP Average accuracy......... %.3e\n"%\
                                                (sum(EPS_stability_DP)/len(EPS))
        res_str += "|= DP Max accuracy............. %.3e\n"%min(EPS_stability_DP)
        res_str += "|= DP Min accuracy............. %.3e\n"%max(EPS_stability_DP)
        res_str += "|= QP Average accuracy......... %.3e\n"%\
                                                (sum(EPS_stability_QP)/len(EPS))
        res_str += "|= QP Max accuracy............. %.3e\n"%min(EPS_stability_QP)
        res_str += "|= QP Min accuracy............. %.3e\n"%max(EPS_stability_QP)

    logFile = open(os.path.join(mg_root, 'stability_%s.log'\
                                                  %process.shell_string()), 'w')
    logFile.write('Stability check results\n\n')
    logFile.write(res_str)
    res_str += "\n= Full details of the run are output in the file"+\
               " stability_%s.log\n"%process.shell_string()
    if len(EPS)>0:
        logFile.write('\nFull details of the %i EPS encountered.\n'%len(EPS))
        for i, eps in enumerate(EPS):
            logFile.write('\nEPS #%i\n'%(i+1))
            logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in eps[1]]))
            logFile.write('\n  DP accuracy :  %.3e\n'%DP_stability[eps[0]])
            logFile.write('  QP accuracy :  %.3e\n'%QP_stability[eps[0]])
    if len(UPS)>0:
        logFile.write('\nFull details of the %i UPS encountered.\n'%len(UPS))
        for i, ups in enumerate(UPS):
            logFile.write('\nUPS #%i\n'%(i+1))
            logFile.write('\n'.join(['  '+' '.join(['%.16E'%pi for pi in p]) \
                                                              for p in ups[1]]))
            logFile.write('\n  DP accuracy :  %.3e\n'%DP_stability[ups[0]])
            logFile.write('  QP accuracy :  %.3e\n'%QP_stability[ups[0]])
    
    accuracies=[10**(-i/5.0) for i in range(5*17)]
    data_plot=[]
    for acc in accuracies:
        data_plot.append(float(len([d for d in DP_stability if d>acc]))\
                                                      /float(len(DP_stability)))
    logFile.write('\nData entries for the stability plot.\n')
    logFile.write('First row is a maximal accuracy delta, second is the '+\
                  'fraction of events with DP accuracy worse than delta.\n\n')
    logFile.writelines('%.3e  %.3e\n'%(accuracies[i], data_plot[i]) for i in \
                                                         range(len(accuracies)))
    logFile.write('\nList of accuracies recorded for the %i evaluations.\n'%nPS)
    logFile.write('First row is DP, second is QP.\n\n')
    logFile.writelines('%.3e  '%DP_stability[i]+('NA\n' if QP_stability[i]==-1.0 \
                             else '%.3e\n'%QP_stability[i]) for i in range(nPS))
    logFile.close()
    try:
        import matplotlib.pyplot as plt
        plt.plot(accuracies, data_plot, color='b', marker='o', linestyle='-')
        plt.axis([min(accuracies),max(accuracies),10**(-int(math.log(nPS)/math.log(10))-1), 1])
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Stability plot for %s'%process.nice_string())
        plt.ylabel('Fraction of events')
        plt.xlabel('Maximal precision')
        logger.info('Some stability statistics will be displayed once you '+\
                                                        'close the plot window')
        plt.show()
        return res_str
    except ImportError:
        res_str += "\n= Install matplotlib to get a "+\
                   "graphical display of the results of this check."
        return res_str
  
def output_timings(process, timings, loop_optimized_output):
    """Present the result of a timings check in a nice format """
    
    res_str = "%s \n"%process.nice_string()
    gen_total = timings['HELAS_MODEL_compilation']+\
                timings['HelasDiagrams_generation']+\
                timings['Process_output']+\
                timings['Diagrams_generation']+\
                timings['Process_compilation']+\
                timings['Initialization']
    res_str += "\n= Generation time total...... ========== %.3gs\n"%gen_total
    res_str += "|= Diagrams generation....... %.3gs\n"\
                                                 %timings['Diagrams_generation']
    res_str += "|= Helas Diagrams generation. %.3gs\n"\
                                            %timings['HelasDiagrams_generation']
    res_str += "|= Process output............ %.3gs\n"\
                                            %timings['Process_output']
    res_str += "|= HELAS+model compilation... %.3gs\n"\
                                            %timings['HELAS_MODEL_compilation']
    res_str += "|= Process compilation....... %.3gs\n"\
                                            %timings['Process_compilation']
    res_str += "|= Initialization............ %.3gs\n"\
                                            %timings['Initialization']
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
    res_str += "|= Loading time (Color data). ~%.3gms\n"\
                                               %(timings['Booting_time']*1000.0)
    res_str += "|= Number of hel. computed... %d/%d\n"\
                                %(timings['n_contrib_hel'],timings['n_tot_hel'])
    res_str += "|= Number of loop diagrams... %d\n"%timings['n_loops']
    if loop_optimized_output:
        res_str += "|= Number of loop groups..... %d\n"%timings['n_loop_groups']
        res_str += "|= Number of loop wfs........ %d\n"%timings['n_loop_wfs']
        for i, r in enumerate(timings['loop_wfs_ranks']):
            res_str += "||= # of loop wfs of rank %d.. %d\n"%(i,r)
    res_str += "\n= Output disk size =====================\n"
    res_str += "|= Source directory sources.. %s\n"%timings['du_source']
    res_str += "|= Process sources........... %s\n"%timings['du_process']    
    res_str += "|= Color and helicity data... %s\n"%timings['du_color']  
    res_str += "|= Executable size........... %s\n"%timings['du_exe'] 
    
    return res_str

def output_comparisons(comparison_results):
    """Present the results of a comparison in a nice list format
       mode short: return the number of fail process
    """
    
    proc_col_size = 17

    pert_coupl = comparison_results[0]['process']['perturbation_couplings']
    if pert_coupl:
        process_header = "Process ["+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = process_header + 1

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
def check_gauge(processes, param_card = None, mg_root="",cuttools="",
                cmass_scheme=False):
    """Check gauge invariance of the processes by using the BRS check.
    For one of the massless external bosons (e.g. gluon or photon), 
    replace the polarization vector (epsilon_mu) with its momentum (p_mu)
    """

    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes

        model = multiprocess.get('model')
        
        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping = True, reuse = False)
        else:
            evaluator = LoopMatrixElementEvaluator(mg_root=mg_root, cuttools_dir=cuttools,
                                           cmass_scheme= cmass_scheme,
                                           model=model, param_card=param_card,
                                           auth_skipping = False, reuse = False)

        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        results = run_multiprocs_no_crossings(check_gauge_process,
                                           multiprocess,
                                           evaluator)
        
        if multiprocess.get('perturbation_couplings')!=[]:
            # Clean temporary folders created for the running of the loop processes
            clean_up(mg_root)
        
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
                                       auth_skipping = True, reuse = False)
    else:
        evaluator = LoopMatrixElementEvaluator(mg_root=mg_root, cuttools_dir=cuttools,
                                           model=model, param_card=param_card,
                                           auth_skipping = False, reuse = False)
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
        result = check_gauge_process(process, evaluator)
        if result:
            comparison_results.append(result)

    if processes[0].get('perturbation_couplings')!=[]:
        # Clean temporary folders created for the running of the loop processes
        clean_up(mg_root)
            
    return comparison_results


def check_gauge_process(process, evaluator):
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
        # This process can't be checked
        return None

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking gauge %s" % \
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
                                         optimized_output=loop_optimized_output)
        
    brsvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = True,
                                                 output='jamp')

    if not isinstance(amplitude,loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = False)
          
    mvalue = evaluator.evaluate_matrix_element(matrix_element, gauge_check = False,
                                               output='jamp')
    
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
        process_header = "Process ["+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = process_header + 1

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
def check_lorentz(processes, param_card = None, mg_root="",cuttools="",
                  cmass_scheme=False):
    """ Check if the square matrix element (sum over helicity) is lorentz 
        invariant by boosting the momenta with different value."""
    
    if isinstance(processes, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess = processes

        model = multiprocess.get('model')
        
        # Initialize matrix element evaluation
        if multiprocess.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model,
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping = False, reuse = True)
        else:
            evaluator = LoopMatrixElementEvaluator(mg_root=mg_root,
                                           cmass_scheme= cmass_scheme,
                                           cuttools_dir=cuttools, model=model,
                                           auth_skipping = False, reuse = True)

        if not cmass_scheme:
            # Set all widths to zero for lorentz check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[\
                                                     particle.get('width')] = 0.
        results = run_multiprocs_no_crossings(check_lorentz_process,
                                           multiprocess,
                                           evaluator)
        
        if multiprocess.get('perturbation_couplings')!=[]:
            # Clean temporary folders created for the running of the loop processes
            clean_up(mg_root)
        
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
                                       cmass_scheme= cmass_scheme,
                                       auth_skipping = False, reuse = True)
    else:
        evaluator = LoopMatrixElementEvaluator(mg_root=mg_root, 
                                           cuttools_dir=cuttools, model=model,
                                           param_card=param_card,
                                           cmass_scheme= cmass_scheme,
                                           auth_skipping = False, reuse = True)

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
        result = check_lorentz_process(process, evaluator)
        if result:
            comparison_results.append(result)

    if processes[0].get('perturbation_couplings')!=[]:
        # Clean temporary folders created for the running of the loop processes
        clean_up(mg_root)

    return comparison_results


def check_lorentz_process(process, evaluator):
    """Check gauge invariance for the process, unless it is already done."""

    amp_results = []
    model = process.get('model')

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)

    logger.info("Checking lorentz %s" % \
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

    # Generate phase space point to use
    p, w_rambo = evaluator.get_momenta(process)

    # Generate the HelasMatrixElement for the process
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = True)
    else:
        matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude,
                                       optimized_output = loop_optimized_output)

    data = evaluator.evaluate_matrix_element(matrix_element, p=p, output='jamp',
                                             auth_skipping = True)

    if data and data['m2']:
        results = [data]
    else:
        return  {'process':process, 'results':'pass'}
    
    for boost in range(1,4):
        boost_p = boost_momenta(p, boost)
        results.append(evaluator.evaluate_matrix_element(matrix_element,
                                                         p=boost_p,
                                                         output='jamp'))
        
        
    return {'process': process, 'results': results}


#===============================================================================
# check_gauge
#===============================================================================
def check_unitary_feynman(processes_unit, processes_feynm, param_card=None, 
                                   mg_root="", cuttools="", cmass_scheme=False):
    """Check gauge invariance of the processes by flipping
       the gauge of the model
    """

    if isinstance(processes_unit, base_objects.ProcessDefinition):
        # Generate a list of unique processes
        # Extract IS and FS ids
        multiprocess_unit = processes_unit
        results = []
        model = multiprocess_unit.get('model')
        
        # Initialize matrix element evaluation
        aloha.unitary_gauge = True
        if processes_unit.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,
                                       cmass_scheme= cmass_scheme,
                                       auth_skipping = False, reuse = True)
        else:
            evaluator = LoopMatrixElementEvaluator(mg_root=mg_root,
                                           cmass_scheme= cmass_scheme,
                                           cuttools_dir=cuttools, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, reuse = True)

                
        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_u = run_multiprocs_no_crossings(get_value,
                                           multiprocess_unit,
                                           evaluator)
        
        clean_added_globals(ADDED_GLOBAL)
        
        momentum = {}
        for data in output_u:
            momentum[data['process']] = data['p']
        
        multiprocess_feynm = processes_feynm
        model = multiprocess_feynm.get('model')
        
        # Initialize matrix element evaluation
        aloha.unitary_gauge = False
        if processes_feynm.get('perturbation_couplings')==[]:
            evaluator = MatrixElementEvaluator(model, param_card,
                                       cmass_scheme= cmass_scheme,
                                       auth_skipping = False, reuse = False)
        else:
            evaluator = LoopMatrixElementEvaluator(mg_root=mg_root,
                                           cmass_scheme= cmass_scheme,
                                           cuttools_dir=cuttools, model=model,
                                           param_card=param_card,
                                           auth_skipping = False, reuse = False)
                
        if not cmass_scheme:
            # Set all widths to zero for gauge check
            logger.info('Set All width to zero for non complex mass scheme checks')
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.

        output_f = run_multiprocs_no_crossings(get_value,
                                           multiprocess_feynm,
                                           evaluator, momentum)  
        
        output = []
        for data in output_f:
            local_dico = {}
            local_dico['process'] = data['process']
            local_dico['value_feynm'] = data['value']
            local_dico['value_unit'] = [d['value'] for d in output_u 
                                      if d['process'] == data['process']][0]
            output.append(local_dico)

        if processes_feynm.get('perturbation_couplings')!=[]:
            # Clean temporary folders created for the running of the loop processes
            clean_up(mg_root)        

        return output
#    elif isinstance(processes, base_objects.Process):
#        processes = base_objects.ProcessList([processes])
#    elif isinstance(processes, base_objects.ProcessList):
#        pass
    else:
        raise InvalidCmd("processes is of non-supported format")

    assert False
    assert processes, "No processes given"

    model = processes[0].get('model')

    # Initialize matrix element evaluation
    evaluator = MatrixElementEvaluator(model, param_card,
                                       auth_skipping = True, reuse = False)

    comparison_results = []
    comparison_explicit_flip = []

    # For each process, make sure we have set up leg numbers:
    for process in processes:
        # Get process result
        result = check_gauge_process(process, evaluator)
        if result:
            comparison_results.append(result)
        
        
            
    return comparison_results


def get_value(process, evaluator, p=None):
    """Return the value/momentum for a phase space point"""

    model = process.get('model')

    for i, leg in enumerate(process.get('legs')):
        leg.set('number', i+1)


    logger.info("Checking gauge %s" % \
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
    
    if not p:
        # Generate phase space point to use
        p, w_rambo = evaluator.get_momenta(process)
        
    # Generate the HelasMatrixElement for the process
    # Generate the HelasMatrixElement for the process
    if not isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
        matrix_element = helas_objects.HelasMatrixElement(amplitude,
                                                      gen_color = True)
    else:
        matrix_element = loop_helas_objects.LoopHelasMatrixElement(amplitude, 
                                                              gen_color = False)    
      
    mvalue = evaluator.evaluate_matrix_element(matrix_element, p=p,
                                                                  output='jamp')
    
    if mvalue and mvalue['m2']:
        return {'process':process.base_string(),'value':mvalue,'p':p}



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

def output_lorentz_inv(comparison_results, output='text'):
    """Present the results of a comparison in a nice list format
        if output='fail' return the number of failed process -- for test-- 
    """

    proc_col_size = 17

    pert_coupl = comparison_results[0]['process']['perturbation_couplings']
    # Of course, be more tolerant for loop processes
    if pert_coupl:
        threshold=1e-5
    else:
        threshold=1e-10
    if pert_coupl:
        process_header = "Process ["+" ".join(pert_coupl)+"]"
    else:
        process_header = "Process"

    if len(process_header) + 1 > proc_col_size:
        proc_col_size = process_header + 1
    
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
    for data in comparison_results:
        proc = data['process']
        if len(proc) + 1 > proc_col_size:
            proc_col_size = len(proc) + 1

    col_size = 17

    pass_proc = 0
    fail_proc = 0
    no_check_proc = 0

    failed_proc_list = []
    no_check_proc_list = []

    res_str = fixed_string_length("Process", proc_col_size) + \
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
        
            tmp_str = '\n' + fixed_string_length('   JAMP %s'%k , proc_col_size) + \
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
    
    
    print res_str
    if output == 'text':
        return res_str        
    else: 
        return fail_proc





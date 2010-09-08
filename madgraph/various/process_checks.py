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

"""Unit test library for the export Python format routines"""

import copy
import fractions
import itertools
import logging
import math
import os
import re

import aloha.aloha_writers as aloha_writers
import aloha.create_aloha as create_aloha

import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writer
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.rambo as rambo

from madgraph import MG5DIR

import models.model_reader as model_reader
import aloha.template_files.wavefunctions as wavefunctions
from aloha.template_files.wavefunctions import \
     ixxxxx, oxxxxx, vxxxxx, sxxxxx

#===============================================================================
# Logger for process_checks
#===============================================================================

logger = logging.getLogger('madgraph.various.process_checks')

#===============================================================================
# Global helper function get_momenta
#===============================================================================
def get_momenta(process, model, energy = 1000.):
    """Get a point in phase space for the external states in the given
    process, with the CM energy given. The incoming particles are
    assumed to be oriented along the z axis, with particle 1 along the
    positive z axis."""

    if not (isinstance(process, base_objects.Process) and \
            isinstance(model, model_reader.ModelReader) and \
            isinstance(energy, float)):
        raise rambo.RAMBOError, "Not correct type for arguments to get_momenta"

    sorted_legs = sorted(process.get('legs'), lambda l1, l2:\
                         l1.get('number') - l2.get('number'))

    nincoming = len([leg for leg in sorted_legs if leg.get('state') == False])
    nfinal = len(sorted_legs) - nincoming

    # Find masses of particles
    mass_strings = [model.get_particle(l.get('id')).get('mass') \
                     for l in sorted_legs]
    mass = [abs(model.get('parameter_dict')[m]) for m in mass_strings]

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
        for i in range(nfinal):
            momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                    p_rambo[(2,i)], p_rambo[(3,i)]]
            p.append(momi)

    if nincoming == 2:

        m2 = mass[1]

        mom = math.sqrt((e2**2 - 2*e2*m1**2 + m1**4 - 2*e2*m2**2 - \
                  2*m1**2*m2**2 + m2**4) / (4*e2))
        e1 = math.sqrt(mom**2+m1**2)
        e2 = math.sqrt(mom**2+m2**2)
        # Set momenta for incoming particles
        p.append([e1, 0., 0., mom])
        p.append([e2, 0., 0., -mom])
             
        p_rambo, w_rambo = rambo.RAMBO(nfinal, energy, masses)

        # Reorder momenta from px,py,pz,E to E,px,py,pz scheme
        for i in range(1, nfinal+1):
            momi = [p_rambo[(4,i)], p_rambo[(1,i)],
                    p_rambo[(2,i)], p_rambo[(3,i)]]
            p.append(momi)

        return p, w_rambo

#===============================================================================
# MatrixElementChecker
#===============================================================================
class MatrixElementChecker(object):
    """Class for checking matrix elements, by generating diagrams with
    all possible orderings of the legs in the matrix element and then
    comparing the matrix element value between the different
    orderings, for a given (random) phase space point."""

    @staticmethod
    def check_processes(processes, param_card = None):
        """Check processes by generating them with all possible
        orderings of particles (which means different diagram building
        and Helas calls)"""

        amplitudes = diagram_generation.AmplitudeList()
        matrix_elements = helas_objects.HelasMatrixElementList()

        if isinstance(processes, base_objects.Process):
            processes = base_objects.ProcessList([processes])
        elif isinstance(processes, base_objects.ProcessList):
            pass
        elif isinstance(processes, diagram_generation.Amplitude):
            amplitudes.append(processes)
            processes = base_objects.ProcessList([processes.get('process')])
        elif isinstance(processes, diagram_generation.AmplitudeList):
            amplitudes = processes
            processes = base_objects.ProcessList([a.get('process') for \
                                                  a in processes])
        elif isinstance(processes, helas_objects.HelasMatrixElement):
            matrix_elements.append(processes)
            processes = base_objects.ProcessList(\
                                                [processes.get('processes')[0]])
        elif isinstance(processes, helas_objects.HelasMatrixElementList):
            matrix_elements = processes
            processes = base_objects.ProcessList([me.get('processes')[0] for \
                                                  me in processes])
        elif isinstance(processes, helas_objects.HelasMultiProcess):
            matrix_elements = processes.get('matrix_elements')
            processes = base_objects.ProcessList([me.get('processes')[0] for \
                                                  me in processes.get(\
                                                          'matrix_elements')])
        else:
            raise MadGraph5Error("processes is of non-supported format")

        if not processes:
            raise MadGraph5Error("No processes given")
            
        model = processes[0].get('model')

        # Read a param_card and calculate couplings
        full_model = model_reader.ModelReader(model)
        
        full_model.set_parameters_and_couplings(param_card)

        # Write the matrix element(s) in Python
        helas_writer = helas_call_writer.PythonUFOHelasCallWriter(model)

        comparison_results = []
        tested_processes = []
        used_lorentz = []

        # For each process, make sure we have set up leg numbers:
        for iproc, process in enumerate(processes):
            # Check if process is already checked
            ids = [l.get('id') for l in process.get('legs') if l.get('state')]
            ids.extend([model.get_particle(l.get('id')).get_anti_pdg_code() \
                        for l in process.get('legs') if not l.get('state')])
            ids = sorted(ids) + [process.get('id')]

            if ids in tested_processes:
                # We have already tested a version of this process, continue
                continue

            # Add this process to tested_processes
            tested_processes.append(ids)
            
            logger.info("Testing process %s" % process.nice_string())

            # Generate phase space point to use
            p, w_rambo = get_momenta(process, full_model)
            # Initiate the value array
            values = []
            for i, leg in enumerate(process.get('legs')):
                leg.set('number', i+1)

            # Now, generate all possible permutations of the legs
            for legs in itertools.permutations(process.get('legs')):
                logger.info("Testing permutation: %s" % \
                            ([l.get('number') for l in legs]))

                legs = base_objects.LegList(legs)

                if legs == process.get('legs') and \
                        len(matrix_elements) >= iproc + 1:
                    matrix_element = matrix_element[iproc]
                else:
                    if legs == process.get('legs') and \
                            len(amplitudes) >= iproc + 1:
                        amplitude = amplitudes[iproc]
                    else:
                        # Generate a process with these legs
                        newproc = copy.copy(process)
                        newproc.set('legs', legs)
                        # Generate the amplitude for this process
                        amplitude = diagram_generation.Amplitude(newproc)
                        if not amplitude.get('diagrams'):
                            # This process has no diagrams; go to next process
                            break
                        
                    # Generate the HelasMatrixElement for the process
                    matrix_element = helas_objects.HelasMatrixElement(amplitude)

                # Create the needed aloha routines
                me_used_lorentz = set(matrix_element.get_used_lorentz())
                me_used_lorentz = [lorentz for lorentz in me_used_lorentz \
                                   if lorentz not in used_lorentz]

                aloha_model = create_aloha.AbstractALOHAModel(model.get('name'))
                aloha_model.compute_subset(me_used_lorentz)
                if aloha_model:
                    logger.info("Generating ALOHA functions %s" % \
                                ",".join([str(k) for k in aloha_model.keys()]))

                # Write out the routines in Python
                aloha_routines = []
                for routine in aloha_model.values():
                    aloha_routines.append(routine.write(output_dir = None,
                                                        language = 'Python').\
                          replace('import wavefunctions',
                                  'import aloha.template_files.wavefunctions' +\
                                  ' as wavefunctions'))

                # Define the routines to be available globally
                for routine in aloha_routines:
                    exec("\n".join(routine.split("\n")[:-1]), globals())

                # Add the defined Aloha routines to used_lorentz
                used_lorentz.extend(me_used_lorentz)

                # Export the matrix element to Python calls
                exporter = export_python.ProcessExporterPython(matrix_element,
                                                               helas_writer)
                matrix_methods = exporter.get_python_matrix_methods()

                # Define the routines (locally is enough)
                for matrix_method in matrix_methods.values():
                    exec(matrix_method)

                # Evaluate the matrix element for the momenta p
                values.append(eval("Matrix_%s().smatrix(p, full_model)" % \
                                   process.shell_string()))
                
            # Done with this process. Collect values, and store
            # process and momenta
            if not values[0]:
                passed = max(values) - min(values) < 1.e-10
            else:
                passed = abs(max(values) - min(values))/abs(max(values)) < 1.e-10

            comparison_results.append({"process": process,
                                       "momenta": p,
                                       "values": values,
                                       "passed": passed})

        return comparison_results

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
from madgraph.core import base_objects
"""Methods and classes to import v4 format model files."""

import fractions
import logging
import os
import re

from madgraph import InvalidCmd, MG4DIR, ReadWrite

import madgraph.core.color_algebra as color
import madgraph.iolibs.files as files
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.various.misc as misc

from madgraph.core.base_objects import Particle, ParticleList
from madgraph.core.base_objects import Interaction, InteractionList

logger = logging.getLogger('madgraph.import_v4')

#===============================================================================
# import_v4model
#===============================================================================
def import_model(model_path, mgme_dir = MG4DIR, absolute=True):
    """create a model from a MG4 model directory."""

    # Check for a valid directory
    model_path_old = model_path
    model_path = find_model_path(model_path, mgme_dir, absolute)

    files_list = [os.path.join(model_path, 'particles.dat'),\
                  os.path.join(model_path, 'interactions.dat')]
    
    for filepath in files_list:
        if not os.path.isfile(filepath):
            if not absolute:
                raise InvalidCmd,  "%s directory is not a valid v4 model" % \
                                                                    (model_path)
            else:
                return import_model(model_path_old, mgme_dir, False)
                                                                
    # use pickle files if defined
    if files.is_uptodate(os.path.join(model_path, 'model.pkl'), files_list):
        model = save_load_object.load_from_file( \
                                          os.path.join(model_path, 'model.pkl'))
        if model.has_key('version_tag') and model.get('version_tag') == os.path.realpath(model_path) + str(misc.get_pkg_info()):
            return model, model_path

    model = base_objects.Model()    
    model.set('particles',files.read_from_file( \
                                  os.path.join(model_path, 'particles.dat'),
                                  read_particles_v4))
    
    model.set('interactions',files.read_from_file( \
                                  os.path.join(model_path, 'interactions.dat'),
                                  read_interactions_v4,
                                  model['particles']))
    
    model.set('name', os.path.split(model_path)[-1])  

    # save in a pickle files to fasten future usage
    if ReadWrite:
        try:
            save_load_object.save_to_file(os.path.join(model_path, 'model.pkl'), model)
        except Exception:
            logger.warning("fail to write %s. This is perfectly fine will just prevent speed boost in future load of this model" %\
                           os.path.join(model_path, 'model.pkl'))
    return model, model_path  

    
def find_model_path(model_path, mgme_dir, absolute=True):
    """Find the path to the model, starting with path model_path."""

    # treat simple case (model_path is a valid path/ mgme_dir doesn't exist)
    if os.path.isdir(model_path) and absolute:
        return model_path
    elif mgme_dir and os.path.isdir(os.path.join(mgme_dir, 'models',
                                                 model_path + "_v4")):
        model_path = os.path.join(mgme_dir, 'models', model_path + "_v4")
    elif mgme_dir and os.path.isdir(os.path.join(mgme_dir, 'Models', model_path)):
        model_path = os.path.join(mgme_dir, 'Models', model_path)
    elif not mgme_dir:
        error_text = "Path %s is not a valid pathname\n" % model_path
        error_text += "and no MG_ME installation detected in order to search in Models"
        raise InvalidCmd(error_text)

    # Try to build the valid path
    path_possibilities = [os.path.join(mgme_dir, 'Models', model_path),
                         os.path.join(mgme_dir, 'models', model_path + "_v4"),    
                         os.path.join(mgme_dir, 'models', model_path)            
                         ]

    for path in path_possibilities:
        if os.path.exists(path) and \
                        not os.path.exists(os.path.join(path, 'particles.py')):
            return path
    
    # No valid path found
    raise InvalidCmd("Path %s is not a valid pathname" % model_path)

#===============================================================================
# read_particles_v4
#===============================================================================
def read_particles_v4(fsock):
    """Read a list of particle from stream fsock, using the old v4 format"""

    spin_equiv = {'s': 1,
                  'f': 2,
                  'v': 3,
                  't': 5}

    color_equiv = {'s': 1,
                   't': 3,
                   '6': 6,
                   'o': 8}

    line_equiv = {'d': 'dashed',
                  's': 'straight',
                  'w': 'wavy',
                  'c': 'curly'}

    logger.info('load particles')

    mypartlist = ParticleList()

    for line in fsock:
        mypart = Particle()

        if line.find("MULTIPARTICLES") != -1:
            break # stop scanning if old MULTIPARTICLES tag found

        line = line.split("#", 2)[0] # remove any comment
        line = line.strip() # makes the string clean

        if line != "":
            values = line.split()
            if len(values) != 9:
                # Not the right number tags on the line
                raise ValueError, \
                    "Unvalid initialization string:" + line
            else:
                try:
                    mypart.set('name', values[0].lower())
                    mypart.set('antiname', values[1].lower())

                    if mypart['name'] == mypart['antiname']:
                        mypart['self_antipart'] = True

                    if values[2].lower() in spin_equiv.keys():
                        mypart.set('spin',
                                   spin_equiv[values[2].lower()])
                    else:
                        raise ValueError, "Invalid spin %s" % \
                                values[2]

                    if values[3].lower() in line_equiv.keys():
                        mypart.set('line',
                                   line_equiv[values[3].lower()])
                    else:
                        raise ValueError, \
                                "Invalid line type %s" % values[3]

                    mypart.set("mass", values[4])
                    mypart.set("width", values[5])

                    if values[6].lower() in color_equiv.keys():
                        mypart.set('color',
                                   color_equiv[values[6].lower()])
                    else:
                        raise ValueError, \
                            "Invalid color rep %s" % values[6]

                    #mypart.set("texname", values[7])
                    mypart.set("pdg_code", int(values[8]))

                    mypart.set('charge', 0.)
                    #mypart.set('antitexname', mypart.get('texname'))

                except (Particle.PhysicsObjectError, ValueError), why:
                    logger.warning("Warning: %s, particle ignored" % why)
                else:
                    mypartlist.append(mypart)

    return mypartlist


#===============================================================================
# read_interactions_v4
#===============================================================================
def read_interactions_v4(fsock, ref_part_list):
    """Read a list of interactions from stream fsock, using the old v4 format.
    Requires a ParticleList object as an input to recognize particle names."""

    logger.info('load interactions')
    myinterlist = InteractionList()

    if not isinstance(ref_part_list, ParticleList):
        raise ValueError, \
            "Object %s is not a valid ParticleList" % repr(ref_part_list)

    for line in fsock:
        myinter = Interaction()

        line = line.split("#", 2)[0] # remove any comment
        line = line.strip() # makes the string clean

        if line != "": # skip blank
            values = line.split()
            part_list = ParticleList()

            try:
                for str_name in values:
                    curr_part = ref_part_list.get_copy(str_name.lower())
                    if isinstance(curr_part, Particle):
                        # Look at the total number of strings, stop if 
                        # anyway not enough, required if a variable name 
                        # corresponds to a particle! (eg G)
                        if len(values) >= 2 * len(part_list) + 1:
                            part_list.append(curr_part)
                        else: break
                    # also stops if string does not correspond to 
                    # a particle name
                    else: break

                if len(part_list) < 3:
                    raise Interaction.PhysicsObjectError, \
                        "Vertex with less than 3 known particles found."

                # Flip part/antipart of first part for FFV, FFS, FFT vertices
                # according to v4 convention
                spin_array = [part['spin'] for part in part_list]
                if spin_array[:2] == [2, 2] and \
                   not part_list[0].get('self_antipart'):
                    part_list[0]['is_part'] = not part_list[0]['is_part']

                myinter.set('particles', part_list)

                # Give color structure
                # Order particles according to color
                # Don't consider singlets
                color_parts = sorted(enumerate(part_list), lambda p1, p2:\
                                     p1[1].get_color() - p2[1].get_color())
                color_ind = [(i, part.get_color()) for i, part in \
                             color_parts if part.get_color() !=1]
                colors = [c for i,c in color_ind]
                ind = [i for i,c in color_ind]

                # Set color empty by default
                myinter.set('color', [])
                if not colors:
                    # All color singlets - set empty
                    pass
                elif colors == [-3, 3]:
                    # triplet-triplet-singlet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(ind[1], ind[0])])])
                elif colors == [8, 8]:
                    # octet-octet-singlet coupling
                    my_cs = color.ColorString(\
                        [color.Tr(ind[0], ind[1])])
                    my_cs.coeff = fractions.Fraction(2)
                    myinter.set('color', [my_cs])
                elif colors == [-3, 3, 8]:
                    # triplet-triplet-octet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(ind[2], ind[1], ind[0])])])
                elif colors == [8, 8, 8]:
                    # Triple glue coupling
                    my_color_string = color.ColorString(\
                        [color.f(ind[0], ind[1], ind[2])])
                    my_color_string.is_imaginary = True
                    myinter.set('color', [my_color_string])
                elif colors == [-3, 3, 8, 8]:
                    my_cs1 = color.ColorString(\
                        [color.T(ind[2], ind[3], ind[1], ind[0])])
                    my_cs2 = color.ColorString(\
                        [color.T(ind[3], ind[2], ind[1], ind[0])])
                    myinter.set('color', [my_cs1, my_cs2])
                elif colors == [8, 8, 8, 8]:
                    # 4-glue coupling
                    cs1 = color.ColorString([color.f(0, 1, -1),
                                                   color.f(2, 3, -1)])
                    #cs1.coeff = fractions.Fraction(-1)
                    cs2 = color.ColorString([color.f(2, 0, -1),
                                                   color.f(1, 3, -1)])
                    #cs2.coeff = fractions.Fraction(-1)
                    cs3 = color.ColorString([color.f(1, 2, -1),
                                                   color.f(0, 3, -1)])
                    #cs3.coeff = fractions.Fraction(-1)
                    myinter.set('color', [cs1, cs2, cs3])
#   The following line are expected to be correct but not physical validations
#    have been performed. So we keep it commented for the moment.                    
#                elif colors == [3, 3, 3]:
#                    my_color_string = color.ColorString(\
#                        [color.Epsilon(ind[0], ind[1], ind[2])])
#                    myinter.set('color', [my_color_string])                    
#                elif colors == [-3, -3, -3]:
#                    my_color_string = color.ColorString(\
#                        [color.EpsilonBar(ind[0], ind[1], ind[2])])
#                    myinter.set('color', [my_color_string])
                else:
                    logger.warning(\
                        "Color combination %s not yet implemented." % \
                        repr(colors))

                # Set the Lorentz structure. Default for 3-particle
                # vertices is empty string, for 4-particle pair of
                # empty strings
                myinter.set('lorentz', [''])

                pdg_codes = sorted([part.get_pdg_code() for part in part_list])

                # WWWW and WWVV
                if pdg_codes == [-24, -24, 24, 24]:
                    myinter.set('lorentz', ['WWWW'])
                elif spin_array == [3, 3, 3, 3] and \
                             24 in pdg_codes and - 24 in pdg_codes:
                    myinter.set('lorentz', ['WWVV'])

                # gggg
                if pdg_codes == [21, 21, 21, 21]:
                    myinter.set('lorentz', ['gggg1', 'gggg2', 'gggg3'])

                # go-go-g
                # Using the special fvigox routine provides the minus
                # sign necessary for octet Majorana-vector interactions
                if spin_array == [2, 2, 3] and colors == [8, 8, 8] and \
                   part_list[0].get('self_antipart') and \
                   part_list[1].get('self_antipart'):
                    myinter.set('lorentz', ['go'])

                # If extra flag, add this to Lorentz    
                if len(values) > 3 * len(part_list) - 4:
                    myinter.get('lorentz')[0] = \
                                      myinter.get('lorentz')[0]\
                                      + values[3 * len(part_list) - 4].upper()

                # Use the other strings to fill variable names and tags

                # Couplings: special treatment for 4-vertices, where MG4 used
                # two couplings, while MG5 only uses one (with the exception
                # of the 4g vertex, which needs special treatment)
                # DUM0 and DUM1 are used as placeholders by FR, corresponds to 1
                if len(part_list) == 3 or \
                   values[len(part_list) + 1] in ['DUM', 'DUM0', 'DUM1']:
                    # We can just use the first coupling, since the second
                    # is a dummy
                    myinter.set('couplings', {(0, 0):values[len(part_list)]})
                    if myinter.get('lorentz')[0] == 'WWWWN':
                        # Should only use one Helas amplitude for electroweak
                        # 4-vector vertices with FR. I choose W3W3NX.
                        myinter.set('lorentz', ['WWVVN'])
                elif values[len(part_list)] in ['DUM', 'DUM0', 'DUM1']:
                    # We can just use the second coupling, since the first
                    # is a dummy
                    myinter.set('couplings', {(0, 0):values[len(part_list)+1]})
                elif pdg_codes == [21, 21, 21, 21]:
                    # gggg
                    myinter.set('couplings', {(0, 0):values[len(part_list)],
                                              (1, 1):values[len(part_list)],
                                              (2, 2):values[len(part_list)]})
                elif myinter.get('lorentz')[0] == 'WWWW':
                    # Need special treatment of v4 SM WWWW couplings since 
                    # MG5 can only have one coupling per Lorentz structure
                    myinter.set('couplings', {(0, 0):\
                                              'sqrt(' + 
                                              values[len(part_list)] + \
                                             '**2+' + \
                                              values[len(part_list) + 1] + \
                                              '**2)'})
                else: #if myinter.get('lorentz')[0] == 'WWVV':
                    # Need special treatment of v4 SM WWVV couplings since 
                    # MG5 can only have one coupling per Lorentz structure
                    myinter.set('couplings', {(0, 0):values[len(part_list)] + \
                                             '*' + \
                                              values[len(part_list) + 1]})
                    #raise Interaction.PhysicsObjectError, \
                    #    "Only FR-style 4-vertices implemented."
                
                # SPECIAL TREATMENT OF COLOR
                # g g sq sq (two different color structures, same Lorentz)
                if spin_array == [3, 3, 1, 1] and colors == [-3, 3, 8, 8]:
                    myinter.set('couplings', {(0, 0):values[len(part_list)],
                                              (1, 0):values[len(part_list)]})

                # Coupling orders - needs to be fixed
                order_list = values[2 * len(part_list) - 2: \
                                    3 * len(part_list) - 4]

                def count_duplicates_in_list(dupedlist):
                    """return a dictionary with key the element of dupeList and
                    with value the number of times that they are in this list"""
                    unique_set = set(item for item in dupedlist)
                    ret_dict = {}
                    for item in unique_set:
                        ret_dict[item] = dupedlist.count(item)
                    return ret_dict

                myinter.set('orders', count_duplicates_in_list(order_list))

                myinter.set('id', len(myinterlist) + 1)

                myinterlist.append(myinter)

            except Interaction.PhysicsObjectError, why:
                logger.error("Interaction ignored: %s" % why)

    return myinterlist

#===============================================================================
# read_proc_card.dat (mg4 format)
#===============================================================================
def read_proc_card_v4(fsock):
    """A simple function reading the files in fsock and returning a 
    ProcCardv4Reader object. This function authorize to have the same syntax as
    for the other files treatment"""

    reader = ProcCardv4Reader(fsock)
    return reader

class ParticleError(InvalidCmd):
    """ A class to carch the error"""
    pass

class WrongFileFormat(InvalidCmd): 
    """A specific class error for wrong V4 proc_card"""
    pass

class ProcCardv4Reader(object):
    """read a proc_card.dat in the mg4 format and creates the equivalent routine
    for mg5"""
    
    #tag in the proc_card.dat which split the proc_card content
        
    # line pattern (remove comment at the end of the line)
    pat_line = re.compile(r"""^\s*(?P<info>[^\#]*?)\s*(\#|$)""", re.DOTALL)
    
    def __init__(self, fsock):
        """init the variable"""

        self.process = [] # List of ProcessInfo
        self.model = ""   # name of the model
        self.multipart = [] # list of the mg4 definition of multiparticle
        self.particles_name = set() # set of authorize particle name
        self.couplings_name = set() # set of mandatory couplings
        self.process_path = os.path.realpath(os.path.join(
                                        os.path.dirname(fsock.name), os.pardir))
        
        # Reading the files and store the information in string format.
        self.analyze_v4_proc_card(fsock)

    
    def analyze_v4_proc_card(self, fsock):
        """read the file and fullfill the variable with mg4 line"""
        
        proc_card = fsock.read()

        # store process information
        process_open = False
        
        process_re = re.search(\
            r"^# Begin\s+PROCESS.*?^(?P<process>.*)^# End\s+PROCESS",
            proc_card, re.MULTILINE|re.DOTALL)

        if not process_re:
            raise WrongFileFormat('No valid Begin...End PROCESS tags')

        model_re = re.search(\
            r"^# Begin\s+MODEL.*?^(?P<model>.+?)(\s+|$)^# End\s+MODEL",
            proc_card, re.MULTILINE|re.DOTALL)

        if not model_re:
            raise WrongFileFormat('No valid Begin...End MODEL tags')

        multiparticles_re = re.search(\
            r"^# Begin\s+MULTIPARTICLES.*?^(?P<multiparticles>.*)^# End\s+MULTIPARTICLES",
            proc_card, re.MULTILINE|re.DOTALL)

        if not multiparticles_re:
            raise WrongFileFormat('No valid Begin...End MULTIPARTICLES tags')

        process_lines = process_re.group('process').split('\n')

        for line in process_lines:
            # an 'end_coup' stop the current process, 
            #    'done' finish the list of process
            analyze_line = self.pat_line.search(line)
            if analyze_line:
                data = analyze_line.group('info') #skip the comment
                if not data:
                    continue
                if not process_open and 'done' not in data:
                    process_open = True
                    self.process.append(ProcessInfo(data))
                elif 'end_coup' in data:
                    process_open = False
                elif 'done' not in data:
                    self.process[-1].add_coupling(data)
         
        self.model = model_re.group('model')
                
        multiparticles_lines = multiparticles_re.group('multiparticles').split('\n')

        for line in multiparticles_lines:
            analyze_line = self.pat_line.search(line)
            if analyze_line:
                line = analyze_line.group('info') #skip the comment
                if not line:
                    continue
                data = line.split()
                self.particles_name.add(data[0].lower())
                self.multipart.append(line)
        
    
    def extract_command_lines(self, model):
        """Return the MG5 command line corresponding to this proc_card 
        the MG5 command import model is skipped (since the model should be 
        loaded -it is one of the argument-)"""
         
        # extract useful information of the model
        self.extract_info_from_model(model)
        
        # use the model information for the splitting in particles of the mg4
        #process line.
        for process in self.process:
            process.analyze_process(self.particles_name)
        
        #Now we are in position to write the lines call
        lines = []    
        #first write the lines associate to the multiparticls definition
        if self.multipart:
            lines.append('# Define multiparticle labels')
        for multipart in self.multipart:
            data = self.separate_particle(multipart, self.particles_name)
            lines.append('define ' + ' '.join(data))
        
        # secondly define the lines associate with diagram
        if self.process:
            lines.append('# Specify process(es) to run')
        for i, process in enumerate(self.process):
            if i == 0:
                lines.append('generate %s' % \
                                  process.mg5_process_line(self.couplings_name))
            else:
                lines.append('add process %s' % \
                                  process.mg5_process_line(self.couplings_name))
        
        #finally export the madevent output
        lines.append('# Output processes to MadEvent directory')
        lines.append('output -f')
        
        return lines
        
        
    def extract_info_from_model(self, model):
        """ creates the self.particles_name (list of all valid name)
            and self.couplings_name (list of all couplings)"""
        
        # add in self.particles_name (it contains normally the mulpart name 
        #already) all the valid name of particle of the model    
        for particle in model['particles']:
            self.particles_name.add(particle['name'])
            self.particles_name.add(particle['antiname'])

        # add in self.couplings_name the couplings name of the model
        for interaction in model['interactions']:
            for coupling in interaction['orders'].keys():
                self.couplings_name.add(coupling)

    
    @staticmethod
    def separate_particle(line, possible_str):
        """ for a list of concatanate variable return a list of particle name"""

        line = line.lower() # Particle name are not case sensitive
        out = []            # list of the particles
        # The procedure to find particles is the following
        #  - check if the combination of 4 string form a valid particle name
        #    if it is, move of 4 characters and check for the next particles.
        #    if not try with 3, 2, 1 
        #    if still not -> exit.
        
        pos = 0        # current starting position 
        old_pos = -1   # check that we don't have infinite loop    
        line += '     '  #add 4 blank for security
        while pos < len(line) - 4:
            #Check for infinite loop
            if pos == old_pos:
                logging.error('Invalid particle name: %s' % \
                              line[pos:pos + 4].rstrip())
                raise ParticleError('Invalid particle name %s' %
                                     line[pos:pos + 4].rstrip())
            old_pos = pos
            # check for pointless character
            if line[pos] in [' ', '\n', '\t']:
                pos += 1
                continue
            
            # try to find a match at 4(then 3/2/1) characters
            for i in range(4, 0, -1):
                if line[pos:pos + i] in possible_str:
                    out.append(line[pos:pos + i])
                    pos = pos + i
                    break
                
        return out
    
class ProcessInfo(object):
    """This is the basic object for storing process information"""
    
    def __init__(self, line):
        """Initialize information"""
            
        self.particles = [] # list tuple (level, particle)
        self.couplings = {} # coupling -> max_order
        self.decays = []    # ProcessInfo of the decays
        self.tag = ''       # tag of the process
        self.s_forbid = []  # list of particles forbids in s channel
        self.forbid = []     # list of particles forbids
        self.line = line    # initialization line
        
        self.is_mg5_valid = False
        #some shortcut
        self.separate_particle = ProcCardv4Reader.separate_particle
    
    def analyze_process(self, particles_name):
        """Add a line information
            two format are possible (decay chains or not)
            pp>h>WWj /a $u @3
            pp>(h>WW)j /a $u @3
        """

        line = self.line
        #extract the tag
        if '@' in line:
            split = line.split('@')
            line = split[0]
            self.tag = split[1]
            

        # check if we have a MG5 format
        if '/mg5/' in line:
            self.line = line.replace('/mg5/','')
            self.is_mg5_valid = True
            return
        if ',' in line or '=' in line:
            self.is_mg5_valid = True
            return

        # extract (S-)forbidden particle
        pos_forbid = line.find('/')
        pos_sforbid = line.find('$')
        
        # Select the restrictions (pos is -1 if not defined)
        #and remove the restrictions from the line
        if pos_forbid != -1 and pos_sforbid != -1:
            if  pos_forbid > pos_sforbid :
                self.forbid = self.separate_particle(line[pos_forbid + 1:], \
                                                                 particles_name)
                self.s_forbid = self.separate_particle(\
                               line[pos_sforbid + 1:pos_forbid], particles_name)
                line = line[:min(pos_forbid, pos_sforbid)]
            else:
                self.forbid = self.separate_particle(\
                               line[pos_forbid + 1:pos_sforbid], particles_name)
                self.s_forbid = self.separate_particle(line[pos_sforbid + 1:], \
                                                           particles_name)
                line = line[:min(pos_forbid, pos_sforbid)]
        # Same but if they are no S-forbidden particles
        elif pos_forbid != -1:
            self.forbid = self.separate_particle(line[pos_forbid + 1:], \
                                                                 particles_name)
            line = line[:pos_forbid]
        # Same but if they are no forbidden particles
        elif pos_sforbid != -1:
            self.s_forbid = self.separate_particle(line[pos_sforbid + 1:], \
                                                                 particles_name)
            line = line[:pos_sforbid]
            
        # Deal with decay chains, returns lines whitout the decay (and treat 
        #the different decays.
        if '(' in line:
            line = self.treat_decay_chain(line, particles_name)
            
        #define the level of each particle
        level_content = line.split('>')
        for level, data in enumerate(level_content):
            particles = self.separate_particle(data, particles_name)
            if particles:
                [self.particles.append((level, name)) for name in particles]
            
            
    def treat_decay_chain(self, line, particles_name):
        """Split the information of the decays into a tree of ProcessInfo."""
            
        level = 0 #depth of the decay chain
        out_line = '' # core process
        for character in line:
            if character == '(':
                level += 1
                if level == 1:
                    decay_line = "" # initialize a new decay info
                else:
                    decay_line += '('
                continue
            elif character == ')':
                level -= 1
                if level == 0: #store the information
                    self.decays.append(ProcessInfo(decay_line))
                    self.decays[-1].add_restrictions(self.forbid, self.s_forbid,
                                                                 None)
                    self.decays[-1].analyze_process(particles_name)
                    out_line += decay_line[:decay_line.find('>')]
                else:
                    decay_line += ')'
                continue
            elif level:
                decay_line += character
            else:
                out_line += character
        return out_line
        
    def add_coupling(self, line):
        """Add the coupling information to the process"""
        data = line.split('=')
        self.couplings[data[0]] = int(data[1])
        
    
    def add_restrictions(self, forbid, s_forbid, couplings):
        """Associate some restriction to this diagram"""
        
        self.forbid = forbid
        self.s_forbid = s_forbid
        self.couplings = couplings

    def mg5_process_line(self, model_coupling):
        """Return a valid mg5 format for this process """
        
        if self.is_mg5_valid:
            return self.line
        
        text = ''
        # Write the process
        cur_level = 0
        for level, particle in self.particles:
            if level > cur_level:
                text += '> '
                cur_level += 1
            text += '%s ' % particle

        # Write the constraints
        if self.s_forbid:
            text += '$ ' + ' '.join(self.s_forbid) + ' '
        if self.forbid:
            text += '/ ' + ' '.join(self.forbid) + ' '

        #treat decay_chains
        for decay in self.decays:
            decay_text = decay.mg5_process_line(model_coupling)
            if ',' in decay_text:
                text = text.rstrip() + ', (%s) ' % decay_text.strip()
            else:
                text = text.rstrip() + ', %s ' % decay_text.strip()
        
        # write the tag
        if self.tag:
            text += '@%s ' % self.tag

        if self.couplings:
            if not self.tag:
                text += '@0 '
            #write the rules associate to the couplings
            text += self.mg5_couplings_line(model_coupling, len(self.particles))
        
        return text.rstrip()
    
    def mg5_couplings_line(self, model_coupling, nb_part):
        """Return the assignment of coupling for this process"""

        out = ''
        for coupling in model_coupling:
            if self.couplings.has_key(coupling):
                # Need coupling for all cases, since might be decay chain
                out += '%s=%s ' % (coupling, self.couplings[coupling])
            else:
                # if not define put to zero (mg4 default)
                out += '%s=0 ' % coupling
        
        return out 
    
    
    
    

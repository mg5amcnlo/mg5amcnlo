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

"""Methods and classes to import v4 format model files."""

import fractions
import logging
import os
import re


import madgraph.core.color_algebra as color
from madgraph.core.base_objects import Particle, ParticleList
from madgraph.core.base_objects import Interaction, InteractionList

logger = logging.getLogger('import_model_v4')

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
                   'o': 8}

    line_equiv = {'d': 'dashed',
                  's': 'straight',
                  'w': 'wavy',
                  'c': 'curly'}


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
                    mypart.set('name', values[0])
                    mypart.set('antiname', values[1])

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

                    mypart.set("texname", values[7])
                    mypart.set("pdg_code", int(values[8]))

                    mypart.set('charge', 0.)
                    mypart.set('antitexname', mypart.get('texname'))

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
                    curr_part = ref_part_list.find_name(str_name)
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

                # Flip part/antipart of first part for FFV and FFS vertices
                # according to v4 convention
                spin_array = [part['spin'] for part in part_list]
                if spin_array in [[2, 2, 1], [2, 2, 3]]  and \
                   not part_list[0].get('self_antipart'):
                    part_list[0]['is_part'] = not part_list[0]['is_part']

                myinter.set('particles', part_list)

                # Give color structure
                # Order particles according to color
                # Don't consider singlets
                color_parts = sorted(part_list, lambda p1, p2:\
                                            p1.get_color() - p2.get_color())
                color_parts = filter(lambda p: p.get_color() != 1, color_parts)
                colors = [part.get_color() for part in color_parts]

                # Set color empty by default
                myinter.set('color', [])
                if not colors:
                    # All color singlets - set empty
                    pass
                elif colors == [-3, 3]:
                    # triplet-triplet-singlet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(1, 0)])])
                elif colors == [8, 8]:
                    # octet-octet-singlet coupling
                    my_cs = color.ColorString(\
                        [color.Tr(0, 1)])
                    my_cs.coeff = fractions.Fraction(2)
                    myinter.set('color', [my_cs])
                elif colors == [-3, 3, 8]:
                    # triplet-triplet-octet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(2, 1, 0)])])
                elif colors == [8, 8, 8]:
                    # Triple glue coupling
                    my_color_string = color.ColorString(\
                        [color.f(0, 1, 2)])
                    my_color_string.is_imaginary = True
                    myinter.set('color', [my_color_string])
                elif colors == [-3, 3, 8, 8]:
                    my_cs1 = color.ColorString(\
                        [color.T(2, 3, 1, 0)])
                    my_cs2 = color.ColorString(\
                        [color.T(3, 2, 1, 0)])
                    myinter.set('color', [my_cs1, my_cs2])

                elif colors == [8, 8, 8, 8]:
                    # 4-glue / glue-glue-gluino-gluino coupling
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
                if spin_array == [2, 2, 3] and colors == [8, 8, 8]:
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
                elif myinter.get('lorentz')[0] == 'WWVV':
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

                def count_duplicates_in_list(dupedList):
                    uniqueSet = set(item for item in dupedList)
                    ret_dict = {}
                    for item in uniqueSet:
                        ret_dict[item] = dupedList.count(item)
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

    modeldir = os.path.dirname(fsock.name) + '../../Models'
    reader = Reader_proc_card(fsock, modeldir)
    return reader



class Reader_proc_card():
    """ read a proc_card.dat in the mg4 format and creates the equivalent routine \
    for mg5"""
    
    #tag in order 
    begin_process = "# Begin PROCESS # This is TAG. Do not modify this line\n"
    end_process = "# End PROCESS  # This is TAG. Do not modify this line\n"
    begin_model = "# Begin MODEL  # This is TAG. Do not modify this line\n"
    end_model = "# End   MODEL  # This is TAG. Do not modify this line\n"
    begin_multipart = "# Begin MULTIPARTICLES # This is TAG. Do not modify this line\n"
    end_multipart = "# End  MULTIPARTICLES # This is TAG. Do not modify this line\n" 
        
   # line pattern
    pat_line =re.compile(r"""^\s*(?P<info>[^\#]*[\w+-~])\s*[\#]*""",re.DOTALL)
    
    def __init__(self, fsock, model_dir='../Models'):
        """init the variable"""

        self.process = [] 
        self.model = ""
        self.model_dir = model_dir
        self.multipart = []
        self.particles_name = set()
        self.couplings_name = set()
        self.mg5_process = []
        
        # Reading the files and first storing of information
        #whithout any knowledge of the model
        self.collect_info(fsock)

    
    def collect_info(self, fp):
        """read the file and fullfill the variable with mg4 line"""
        
        # skip the introduction
        for line in iter(fp.readline, self.begin_process):
            pass
        
        # store process information
        process_open = False
        for line in iter(fp.readline, self.end_process):
            analyze_line = self.pat_line.search(line)
            if analyze_line:
                data = analyze_line.group('info')
                if not process_open and 'done' not in data:
                    process_open = True
                    self.process.append(Process_info(data))
                elif 'end_coup' in data:
                    process_open = False
                elif 'done' not in data:
                    self.process[-1].add_coupling(data)
         
        #skip comment
        for line in iter(fp.readline, self.begin_model):
            pass        
        
        #load the model
        for line in iter(fp.readline, self.end_model):
            analyze_line = self.pat_line.search(line)
            if analyze_line:
                model = analyze_line.group('info')
                self.model = model
                
        #skip comment
        for line in iter(fp.readline, self.begin_multipart):
            pass
        
        #store multipart information
        for line in iter(fp.readline, self.end_multipart):
            data=line.split()
            self.particles_name.add(data[0].lower())
            self.multipart.append(line)   
        
    
    def treat_data(self, model):
        
        # extract useful information of the model
        self.extract_info_from_model(model)
        
        #use to develop the data for the processes
        for process in self.process:
            process.analyze_process(self.particles_name)
        
        #Now we are in position to write the lines call
        lines = []    
        #first write the lines associate to the multiparton definition
        for multipart in self.multipart:
            data = self.separate_particle(multipart, self.particles_name)
            lines.append('define '+' '.join(data))
        
        # secondly define the lines associate with diagram
        for i,process in enumerate(self.process):
            if i == 0:
                lines.append('generate %s'% process.repr(self.couplings_name))                
            else:
                lines.append('add process %s' % process.repr(self.couplings_name))
        
        #finally export the madevent output
        lines.append('export v4madevent .')
        
        return lines
        
        
    def extract_info_from_model(self, model):
        """ creates the self.particles_name (list of all valid name)
            and self.couplings_name (list of all couplings)"""
            
        for particle in model['particles']:
            self.particles_name.add(particle['name'])
            self.particles_name.add(particle['antiname'])

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
        #    if it is, move of 4 caracter and check for the next particle i
        #    if not try with 3, 2, 1 
        #    if still not -> exit.
        
        pos = 0        # current starting position 
        old_pos = -1   # check that we don't have infinite loop    
        line += '     '  #add 4 blank for security
        while pos < len(line) - 4:
            if pos == old_pos:
                logging.error('Invalid set of caracter: %s' % line[pos:pos+4])
                return
            old_pos = pos
            # check for pontless caracter
            if line[pos] in [' ','\n','\t']:
                pos += 1
                continue
            
            # try to find a match at 4(then 3/2/1) caracter
            for i in range(4,0,-1):
                if line[pos:pos+i] in possible_str:
                    out.append(line[pos:pos + i])
                    pos = pos + i
                    break
                
        return out
    
class Process_info():
    """ this is the basic object of a process"""
    
    
    
    def __init__(self, line):
        """ initialize information"""
            
        self.particles = [] # list tuple (level, particle)
        self.couplings = {}
        self.decays = []
        self.tag = ''
        self.s_forbid = []
        self.forbid =[]
        self.line = line
        self.separate_particle = Reader_proc_card.separate_particle
    
    def analyze_process(self, particles_name):
        """add a line information
            two format are possible (decay chains or not
            pp>h>WWj /a $u @3
            pp>(h>WW)j /a $u @3
        """
        line = self.line
        #extract the tag
        if '@' in line:
            split = line.split('@')
            line = split[0]
            self.tag = split[1]
            

        # ensure that we deal with old format
        if ',' in line:
            logging.error('not accepting new format yet')
            return
            
        # extract (S-)forbidden particle
        pos_forbid = line.find('/')
        pos_sforbid = line.find('$')
        
        #select the restriction (pos is -1 if not defined)
        if pos_forbid != -1 and pos_sforbid != -1:
            if  pos_forbid > pos_sforbid :
                self.forbid = self.separate_particle(line[pos_forbid:], \
                                                                 particles_name)
                self.sforbid = self.separate_particle(\
                                    line[pos_sforbid:pos_forbid], particles_name)
                line = line[:min(pos_forbid,pos_sforbid)]
            else:
                self.forbid = self.separate_particle(\
                                   line[pos_forbid:pos_sforbid], particles_name)
                self.s_forbid = self.separate_particle(line[pos_sforbid:], \
                                                           particles_name)
                line = line[:min(pos_forbid,pos_sforbid)]
        elif pos_forbid != -1:
            self.forbid = self.separate_particle(line[pos_forbid:], \
                                                                 particles_name)
            line = line[:pos_forbid]
        elif pos_sforbid != -1:
            self.sforbid = self.separate_particle(line[pos_sforbid:], \
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
            [self.particles.append((level,name)) for name in particles]
            
            
    def treat_decay_chain(self, line, particles_name):
        """ split the information of the decays """
            
        level = 0 #depth of the decay chain
        out_line = '' # core process
        for caracter in line:
            if caracter == '(':
                level += 1
                if level == 1:
                    decay_line = "" # initialize a new decay info
                else:
                    decay_line += '('
                continue
            elif caracter == ')':
                level -=1
                if level == 0: #store the information
                    self.decays.append(Process_info(decay_line))
                    self.decays[-1].put_constraints(self.forbid, self.s_forbid, \
                                                                 self.couplings)
                    self.decays[-1].analyze_process(particles_name)
                    out_line += decay_line[:decay_line.find('>')]
                else:
                    decay_line += ')'
                continue
            elif level:
                decay_line += caracter
            else:
                out_line += caracter
        return out_line
        
    def add_coupling(self, line):
        data = line.split('=')
        self.couplings[data[0]]=int(data[1])
        
    
    def put_constraints(self,forbid, s_forbid, couplings):
        """ associate some restriction to this diagram"""
        
        self.forbid = forbid
        self.s_forbid = s_forbid
        self.couplings = couplings

    def repr(self, model_coupling):
        
        text = ''
        # Write the process
        cur_level = 0
        for level, particle in self.particles:
            if level > cur_level:
                text += '> '
                cur_level +=1
            text += '%s ' % particle

        # Write the constraints
        if self.forbid:
            text+='/ '+' '.join(self.forbid)
        if self.s_forbid:
            text+='$ '+' '.join(self.s_forbid)
        
        #write the rules associate to the couplings
        text += self.repr_couplings(model_coupling, len(self.particles))
        
        # write the tag
        if self.tag:
            text += '@%s ' % self.tag

        #treat decay_chains
        for decay in self.decays:
            decay_text = decay.repr(model_coupling)
            if ',' in decay_text:
                text +=', (%s) ' % decay_text
            else:
                text += ', %s ' % decay_text
        
        return text
    
    def repr_couplings(self, model_coupling, nb_part):
        """return the assignement of coupling for this process"""

        out=''
        for coupling in model_coupling:
            if self.couplings.has_key(coupling):
                if self.couplings[coupling] < nb_part - 2:
                    out += '%s=%s ' %(coupling, self.couplings[coupling])
            else:
                out += '%s=0 ' % coupling
        
        return out 
    
    
    
    
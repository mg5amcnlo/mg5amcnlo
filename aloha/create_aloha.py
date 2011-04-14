################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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
from __future__ import division
import cmath
import copy
import cPickle
import glob
import logging
import numbers
import os
import re
import shutil
import sys
import time

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(root_path)
from aloha.aloha_object import *
import aloha.aloha_writers as aloha_writers
import aloha.aloha_lib as aloha_lib
try:
    import madgraph.iolibs.files as files
except:
    import aloha.files as files
    
aloha_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('ALOHA')

_conjugate_gap = 50
_spin2_mult = 1000

class ALOHAERROR(Exception): pass

class AbstractRoutine(object):
    """ store the result of the computation of Helicity Routine
    this is use for storing and passing to writer """
    
    def __init__(self, expr, outgoing, spins, name, infostr):
        """ store the information """

        self.spins = spins
        self.expr = expr
        self.name = name
        self.outgoing = outgoing
        self.infostr = infostr
        self.symmetries = []
        
    def add_symmetry(self, outgoing):
        """ add an outgoing """
        
        if not outgoing in self.symmetries:
            self.symmetries.append(outgoing)
        
    def write(self, output_dir, language='Fortran', mode='self'):
        """ write the content of the object """
        return getattr(aloha_writers, 'ALOHAWriterFor%s' % language)(self, output_dir).write(mode=mode)



class AbstractRoutineBuilder(object):
    """ Launch the creation of the Helicity Routine"""
    
    aloha_lib = None
    counter = 0
    
    class AbstractALOHAError(Exception):
        """ An error class for ALOHA"""
    
    def __init__(self, lorentz):
        """ initialize the run
        lorentz: the lorentz information analyzed (UFO format)
        language: define in which language we write the output
        modes: 0 for  all incoming particles 
              >0 defines the outgoing part (start to count at 1)
        """

        self.spins = lorentz.spins
        self.name = lorentz.name
        self.conjg = []
        self.outgoing = None
        self.lorentz_expr = lorentz.structure        
        self.routine_kernel = None
        self.spin2_massless = False
        
    
    def compute_routine(self, mode, factorize=True):
        """compute the expression and return it"""
        self.outgoing = mode
        self.expr = self.compute_aloha_high_kernel(mode, factorize)
        return self.define_simple_output()
    
    def define_all_conjugate_builder(self, pair_list):
        """ return the full set of AbstractRoutineBuilder linked to fermion 
        clash"""
        
        solution = []

        for i, pair in enumerate(pair_list):
            new_builder = self.define_conjugate_builder(pair)
            solution.append(new_builder)
            solution += new_builder.define_all_conjugate_builder(pair_list[i+1:])
        return solution
                   
    def define_conjugate_builder(self, pairs=1):
        """ return a AbstractRoutineBuilder for the conjugate operation.
        If they are more than one pair of fermion. Then use pair to claim which 
        one is conjugated"""
        
        new_builder = copy.copy(self)
        new_builder.conjg = self.conjg[:]
        try:
            for index in pairs:
                new_builder.apply_conjugation(index) 
        except TypeError:
            new_builder.apply_conjugation(pairs) 
        return new_builder
    
    def apply_conjugation(self, pair=1):
        """ apply conjugation on self object"""
        
        
        old_id = 2 * pair - 1
        new_id = _conjugate_gap + old_id
        
        if self.routine_kernel is None:
            self.kernel_tag = set()
            self.routine_kernel = eval(self.lorentz_expr)
           
        self.routine_kernel = \
             C(new_id, old_id + 1) * self.routine_kernel * C(new_id + 1, old_id)
        self.name += 'C'

        if pair:
            self.name += str(pair)
        self.conjg.append(pair)

    
    
    def define_simple_output(self):
        """ define a simple output for this AbstractRoutine """
    
        infostr = str(self.lorentz_expr)        
        return AbstractRoutine(self.expr, self.outgoing, self.spins, self.name, \
                                                                        infostr)
        
    def change_sign_for_outcoming_fermion(self):
        """change the sign of P for outcoming fermion in order to 
        correct the mismatch convention between HELAS and FR"""
        
        flip_sign = []
        for i in range(1,len(self.spins),2):
            if self.spins[i] == 2:
                flip_sign.append(str(i))
        
        if not flip_sign:
            return self.lorentz_expr
        momentum_pattern = re.compile(r'\bP\(([\+\-\d]+),(%s)\)' % '|'.join(flip_sign))
        lorentz_expr = momentum_pattern.sub(r'P(\1,\2, -1)', self.lorentz_expr)
        return lorentz_expr
        
        
    def compute_aloha_high_kernel(self, mode, factorize=True):
        """compute the abstract routine associate to this mode """
        
        # reset tag for particles
        aloha_lib.USE_TAG=set()
        #multiply by the wave functions
        nb_spinor = 0
        if not self.routine_kernel:
            AbstractRoutineBuilder.counter += 1
            logger.info('aloha creates %s routines' % self.name)
            try:
                lorentz = self.change_sign_for_outcoming_fermion()       
                lorentz = eval(lorentz)
            except NameError:
                logger.error('unknow type in Lorentz Evaluation')
                raise ALOHAERROR, 'unknow type in Lorentz Evaluation' 
            else:
                self.routine_kernel = lorentz
                self.kernel_tag = set(aloha_lib.USE_TAG)
        else:
            lorentz = self.routine_kernel
            aloha_lib.USE_TAG = set(self.kernel_tag) 
        for (i, spin ) in enumerate(self.spins):
            id = i + 1
            #Check if this is the outgoing particle
            if id == self.outgoing:
                if spin == 1: 
                    lorentz *= complex(0,1)
                elif spin == 2:
                    # shift the tag if we multiply by C matrices
                    if (id+1) // 2 in self.conjg: 
                        id += _conjugate_gap
                    nb_spinor += 1
                    if nb_spinor %2:
                        lorentz *= SpinorPropagator(id, 'I2', self.outgoing)
                    else:
                        lorentz *= SpinorPropagator('I2', id, self.outgoing) 
                elif spin == 3 :
                    lorentz *= VectorPropagator(id, 'I2', id)
                elif spin == 5 :
                    lorentz *= 1 # delayed evaluation (fastenize the code)
                    #if self.spin2_massless:
                    #    lorentz *= Spin2masslessPropagator(_spin2_mult + id, \
                    #                         2 * _spin2_mult + id,'I2','I3')
                    #else:
                    #    lorentz *= Spin2Propagator(_spin2_mult + id, \
                    #                         2 * _spin2_mult + id,'I2','I3', id)
                else:
                    raise self.AbstractALOHAError(
                                'The spin value %s is not supported yet' % spin)
            else:
                # This is an incoming particle
                if spin == 1:
                    lorentz *= Scalar(id)
                elif spin == 2:
                    # shift the tag if we multiply by C matrices
                    if (id+1) // 2 in self.conjg:
                        spin_id = id + _conjugate_gap
                    else:
                        spin_id = id
                    nb_spinor += 1
                    lorentz *= Spinor(spin_id, id)
                elif spin == 3:        
                    lorentz *= Vector(id, id)
                elif spin == 5:
                    lorentz *= Spin2(1 * _spin2_mult + id, 2 * _spin2_mult + id, id)
                else:
                    raise self.AbstractALOHAError(
                                'The spin value %s is not supported yet' % spin)                    

        # If no particle OffShell
        if self.outgoing:
            lorentz /= DenominatorPropagator(self.outgoing)
            #lorentz.tag.add('OM%s' % self.outgoing )  
            #lorentz.tag.add('P%s' % self.outgoing)  
        else:
            lorentz *= complex(0,-1)

          
        lorentz = lorentz.simplify()

        lorentz = lorentz.expand()
        if self.outgoing and self.spins[self.outgoing-1] == 5:
            if not self.aloha_lib:
                AbstractRoutineBuilder.load_library()
            if self.spin2_massless:
                lorentz *= self.aloha_lib[('Spin2PropMassless', id)]
            else:
                lorentz *= self.aloha_lib[('Spin2Prop', id)]
            aloha_lib.USE_TAG.add('OM%d' % id)
            aloha_lib.USE_TAG.add('P%d' % id)       
        
        lorentz = lorentz.simplify()
        if factorize:
            lorentz = lorentz.factorize()
        lorentz.tag = set(aloha_lib.USE_TAG)
        return lorentz         
                        
    def define_lorentz_expr(self, lorentz_expr):
        """Define the expression"""
        
        self.expr = lorentz_expr
    
    def define_routine_kernel(self, lorentz=None):
        """Define the kernel at low level"""
        
        if not lorentz:
            logger.info('compute kernel %s' % self.counter)
            AbstractRoutineBuilder.counter += 1  
            lorentz = eval(self.lorentz_expr)
                 
            if isinstance(lorentz, numbers.Number):
                self.routine_kernel = lorentz
                return lorentz
            lorentz = lorentz.simplify()
            lorentz = lorentz.expand()
            lorentz = lorentz.simplify()        
        
        self.routine_kernel = lorentz
        return lorentz

    
    @staticmethod
    def get_routine_name(name, outgoing):
        """return the name of the """
        
        name = '%s_%s' % (name, outgoing) 
        return name
            
    @classmethod
    def load_library(cls):
    # load the library
        try:
            fsock = open(os.path.join(aloha_path, 'ALOHALib.pkl'), 'r')
        except IOError:
            cls.aloha_lib = create_library()
        else:
            #cls.aloha_lib = create_library()
            cls.aloha_lib = cPickle.load(fsock)
        

class AbstractALOHAModel(dict):
    """ A class to build and store the full set of Abstract ALOHA Routine"""

    def __init__(self, model_name, write_dir=None, format='Fortran'):
        """ load the UFO model and init the dictionary """
        
        # Extract the model name if combined with restriction
        model_name_pattern = re.compile("^(?P<name>.+)-(?P<rest>[\w\d_]+)$")
        model_name_re = model_name_pattern.match(model_name)
        if model_name_re:
            name = model_name_re.group('name')
            rest = model_name_re.group("rest")
            if rest == 'full' or \
               os.path.isfile(os.path.join(root_path, "models", name,
                                           "restrict_%s.dat" % rest)):
                model_name = model_name_re.group("name")

        # load the UFO model
        try:
            python_pos = model_name 
            __import__(python_pos)
        except:
            python_pos = 'models.%s' % model_name 
            __import__(python_pos)
        self.model = sys.modules[python_pos]
        # find the position on the disk
        self.model_pos = os.path.dirname(self.model.__file__)

        # list the external routine
        self.external_routines = [] 

        # init the dictionary
        dict.__init__(self)
        self.symmetries = {}
        
        # check the mass of spin2 (if any)
        self.massless_spin2 = self.has_massless_spin2()
        
        if write_dir:
            self.main(write_dir,format=format)
            
    def main(self, output_dir, format='Fortran'):
        """ Compute if not already compute. 
            Write file in models/MY_MODEL/MY_FORMAT.
            copy the file to output_dir
        """
        ext = {'Fortran':'f','Python':'py','CPP':'h'}
        
        
        # Check if a pickle file exists
        if not self.load():
            self.compute_all()
        logger.info(' %s aloha routine' % len(self))
            
        # Check that output directory exists
        if not output_dir:
            output_dir = os.path.join(self.model_pos, format.lower())
            logger.debug('aloha output dir is %s' % output_dir) 
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Check that all routine are generated at default places:
        for (name, outgoing), abstract in self.items():
            routine_name = AbstractRoutineBuilder.get_routine_name(name, outgoing)
            if not glob.glob(os.path.join(output_dir, routine_name) + '.' + ext[format]):
                abstract.write(output_dir, format) 
            else:
                logger.info('File for %s already present, skip the writing of this file' % routine_name)
                   
        
    def save(self, filepos=None):
        """ save the current model in a pkl file """
        
        logger.info('save the aloha abstract routine in a pickle file')
        if not filepos:
            filepos = os.path.join(self.model_pos,'aloha.pkl') 
        
        fsock = open(filepos, 'w')
        cPickle.dump(dict(self), fsock)
        
    def load(self, filepos=None):
        """ reload the pickle file """
        if not filepos:
            filepos = os.path.join(self.model_pos,'aloha.pkl') 
        if os.path.exists(filepos):
            fsock = open(filepos, 'r')
            self.update(cPickle.load(fsock))        
            return True
        else:
            return False
        
    def get(self, lorentzname, outgoing):
        """ return the AbstractRoutine with a given lorentz name, and for a given
        outgoing particle """
        
        try:
            return self[(lorentzname, outgoing)]
        except:
            logger.warning('(%s, %s) is not a valid key' % 
                                                       (lorentzname, outgoing) )
            return None
    
    def set(self, lorentzname, outgoing, abstract_routine):
        """ add in the dictionary """
    
        self[(lorentzname, outgoing)] = abstract_routine
    
    def compute_all(self, save=True, wanted_lorentz = []):
        """ define all the AbstractRoutine linked to a model """

        # Search identical particles in the vertices in order to avoid
        #to compute identical contribution
        self.look_for_symmetries()
        conjugate_list = self.look_for_conjugate()
        if not wanted_lorentz:
            wanted_lorentz = [l.name for l in self.model.all_lorentz]
        for lorentz in self.model.all_lorentz:
            if not lorentz.name in wanted_lorentz:
                # Only include the routines we ask for
                continue
            if -1 in lorentz.spins:
                # No Ghost in ALOHA
                continue 
            
            if lorentz.structure == 'external':
                self.external_routines.append(lorentz.name)
                continue
            
            builder = AbstractRoutineBuilder(lorentz)
            # add information for spin2mass
            if 5 in lorentz.spins and self.massless_spin2 is not None:
                builder.spin2_massless = self.massless_spin2 
            self.compute_aloha(builder)
            if lorentz.name in conjugate_list:
                conjg_builder_list= builder.define_all_conjugate_builder(\
                                                   conjugate_list[lorentz.name])
                for conjg_builder in conjg_builder_list:
                    # No duplication of conjugation:
                    assert conjg_builder_list.count(conjg_builder) == 1
                    self.compute_aloha(conjg_builder, lorentz.name)
                    
                    
        if save:
            self.save()
    

    def compute_subset(self, data):
        """ create the requested ALOHA routine. 
        data should be a list of tuple (lorentz, conjugate, outgoing)
        conjugate should be a tuple with the pair number to conjugate.
        outgoing a tuple of the requested routines."""
        
        # Search identical particles in the vertices in order to avoid
        #to compute identical contribution
        self.look_for_symmetries()
        
        # reorganize the data (in order to use optimization for a given lorentz
        #structure
        request = {}
        for l_name, conjugate, outgoing in data:
            try:
                request[l_name][conjugate].append(outgoing)
            except:
                try:
                    request[l_name][conjugate] = [outgoing]
                except:
                    request[l_name] = {conjugate: [outgoing]}
        # Loop on the structure to build exactly what is request
        for l_name in request:
            lorentz = eval('self.model.lorentz.%s' % l_name)
            if lorentz.structure == 'external':
                if lorentz.name not in self.external_routines:
                    self.external_routines.append(lorentz.name)
                continue
            
            builder = AbstractRoutineBuilder(lorentz)
            # add information for spin2mass
            if 5 in lorentz.spins and self.massless_spin2 is not None:
                builder.spin2_massless = self.massless_spin2 
            
            for conjg in request[l_name]:
                #ensure that routines are in rising order (for symetries)
                routines = sorted(request[l_name][conjg])
                if not conjg:
                    # No need to conjugate -> compute directly
                    self.compute_aloha(builder, routines=routines)
                else:
                    # Define the high level conjugate routine
                    conjg_builder = builder.define_conjugate_builder(conjg)
                    # Compute routines
                    self.compute_aloha(conjg_builder, symmetry=lorentz.name,
                                        routines=routines)
            
                        
    def compute_aloha(self, builder, symmetry=None, routines=None):
        """ define all the AbstractRoutine linked to a given lorentz structure
        symmetry authorizes to use the symmetry of anoter lorentz structure.
        routines to define only a subset of the routines."""

        name = builder.name
        if not symmetry:
            symmetry = name
        if not routines:
            routines = range(len(builder.spins) + 1)
  
        # Create the routines
        for outgoing in routines:
            symmetric = self.has_symmetries(symmetry, outgoing, valid_output=routines)
            if symmetric:
                self.get(symmetry, symmetric).add_symmetry(outgoing)
            else:
                wavefunction = builder.compute_routine(outgoing)
                #Store the information
                self.set(name, outgoing, wavefunction)

    def compute_aloha_without_kernel(self, builder, symmetry=None, routines=None):
        """define all the AbstractRoutine linked to a given lorentz structure
        symmetry authorizes to use the symmetry of anoter lorentz structure.
        routines to define only a subset of the routines. 
        Compare to compute_aloha, each routines are computed independently.
        """

        name = builder.name
        if not routines:
            routines = range(len(builder.spins) + 1 )         
        
        for outgoing in routines:
            builder.routine_kernel = None
            wavefunction = builder.compute_routine(outgoing)
            self.set(name, outgoing, wavefunction)


    def write(self, output_dir, language):
        """ write the full set of Helicity Routine in output_dir"""

        for abstract_routine in self.values():
            abstract_routine.write(output_dir, language)
        
        for routine in self.external_routines:
            self.locate_external(routine, language, output_dir)
        
        #self.write_aloha_file_inc(output_dir)
    
    def locate_external(self, name, language, output_dir=None):
        """search a valid external file and copy it to output_dir directory"""
        
        language_to_ext = {'Python': 'py',
                           'Fortran' : 'f',
                           'CPP': 'C'}
        ext = language_to_ext[language]
         
        if os.path.exists(os.path.join(self.model_pos, '%s.%s' % (name, ext))):
            filepos = '%s/%s.%s' % (self.model_pos, name, ext)

        elif os.path.exists(os.path.join(root_path, 'aloha', 'template_files', 
                                                       '%s.%s' %(name, ext))):
            filepos = '%s/aloha/template_files/%s.%s' % (root_path, name, ext)
        else:
            path1 = self.model_pos
            path2 = os.path.join(root_path, 'aloha', 'template_files', )
            raise ALOHAERROR, 'No external routine \"%s.%s\" in directories\n %s\n %s' % \
                        (name, ext, path1, path2)
        
        if output_dir:
            files.cp(filepos, output_dir)
        return filepos
                    
        

    def look_for_symmetries(self):
        """Search some symmetries in the vertices.
        We search if some identical particles are in a vertices in order
        to avoid to compute symmetrical contributions"""
        
        for vertex in self.model.all_vertices:
            for i, part1 in enumerate(vertex.particles):
                for j in range(i-1,-1,-1):
                    part2 = vertex.particles[j]
                    if part1.name == part2.name and \
                                        part1.color == part2.color == 1 and\
                                        part1.spin != 2:
                        for lorentz in vertex.lorentz:
                            if self.symmetries.has_key(lorentz.name):
                                if self.symmetries[lorentz.name].has_key(i+1):
                                    self.symmetries[lorentz.name][i+1] = max(self.symmetries[lorentz.name][i+1], j+1)
                                else:
                                    self.symmetries[lorentz.name][i+1] = j+1
                            else:
                                self.symmetries[lorentz.name] = {i+1:j+1}
                        break
                    
    def has_massless_spin2(self):
        """Search if the spin2 particles are massless or not"""
        
        massless = None
        for particle in self.model.all_particles:
            if particle.spin == 5:
                if massless is None:
                    massless = (particle.mass == 'Zero')
                elif massless != (particle.mass == 'Zero'):
                    raise ALOHAERROR, 'All spin 2 should be massive or massless'
        return massless     
                    
    def has_symmetries(self, l_name, outgoing, out=None, valid_output=None):
        """ This returns out if no symmetries are available, otherwise it finds 
        the lowest equivalent outgoing by recursivally calling this function.
        auth is a list of authorize output, if define"""

        try:
            equiv = self.symmetries[l_name][outgoing]
        except:
            return out
        else:
            if not valid_output or equiv in valid_output:
                return self.has_symmetries(l_name, equiv, out=equiv, 
                                                      valid_output=valid_output)
            else:
                return self.has_symmetries(l_name, equiv, out=out,              
                                                      valid_output=valid_output)
        
    def look_for_conjugate(self):
        """ create a list for the routine needing to be conjugate """

        # Check if they are majorana in the model.
        need = False
        for particle in self.model.all_particles:
            if particle.spin == 2 and particle.selfconjugate:
                need = True
                break

        if not need:
            for interaction in self.model.all_vertices:
                fermions = [p for p in interaction.particles if p.spin == 2]
                for i in range(0, len(fermions), 2):
                    if fermions[i].pdg_code * fermions[i+1].pdg_code > 0:
                        # This is a fermion flow violating interaction
                        need = True
                        break

        # No majorana particles    
        if not need:
            return {}
        
        conjugate_request = {}
        # Check each vertex if they are fermion and/or majorana
        for vertex in self.model.all_vertices:
            for i in range(0, len(vertex.particles), 2):
                part1 = vertex.particles[i]
                if part1.spin !=2:
                    # deal only with fermion
                    break
                # check if this pair contains a majorana
                if part1.selfconjugate:
                    continue
                part2 = vertex.particles[i + 1]
                if part2.selfconjugate:
                    continue
                
                # No majorana => add the associate lorentz structure
                for lorentz in vertex.lorentz:
                    try:
                        conjugate_request[lorentz.name].add(i//2+1)
                    except:
                        conjugate_request[lorentz.name] = set([i//2+1])
        
        for elem in conjugate_request:
            conjugate_request[elem] = list(conjugate_request[elem])
        
        return conjugate_request
            
        
            
def write_aloha_file_inc(aloha_dir,file_ext, comp_ext):
    """find the list of Helicity routine in the directory and create a list 
    of those files (but with compile extension)"""

    aloha_files = []
    
    # Identify the valid files
    alohafile_pattern = re.compile(r'''^[STFV]*[_C\d]*_\d%s''' % file_ext)
    for filename in os.listdir(aloha_dir):
        if os.path.isfile(os.path.join(aloha_dir, filename)):
            if alohafile_pattern.match(filename):
                aloha_files.append(filename.replace(file_ext, comp_ext))

    text="ALOHARoutine = "
    text += ' '.join(aloha_files)
    text +='\n'
    file(os.path.join(aloha_dir, 'aloha_file.inc'), 'w').write(text) 


            
def create_library():
    
    def create(obj):
        """ """
        obj= obj.simplify()
        obj = obj.expand()
        obj = obj.simplify()
        return obj        
    
    # avoid to add tag in global
    old_tag = set(aloha_lib.USE_TAG)
    
    lib = {} # key: (name, part_nb, special) -> object
    for i in range(1, 10):
        logger.info('step %s/9' % i)
        #lib[('Scalar', i)] = create( Scalar(i) )
        #lib[('ScalarProp', i)] = complex(0,1)
        #lib[('Denom', i )] = create( DenominatorPropagator(i) )
        #lib[('Spinor', i )] = create( Spinor(i, i) )
        #lib[('SpinorProp', i, 0)] = create( SpinorPropagator(i, 'I2', i) )
        #lib[('SpinorProp', i, 1)] = create( SpinorPropagator('I2', i, i) )
        #lib[('Vector', i)] = create( Vector(i+1, i+1) )
        #lib[('VectorProp', i)] = create( VectorPropagator(i,'I2', i) )
        #lib[('Spin2', i )] = create( Spin2(10*i+1, 10*i+2, i) )
        lib[('Spin2Prop',i)] = create( Spin2Propagator(_spin2_mult + i, \
                                             2 * _spin2_mult + i,'I2','I3', i) )
        lib[('Spin2PropMassless',i)] = create( Spin2masslessPropagator(
                             _spin2_mult + i, 2 * _spin2_mult + i,'I2','I3'))
    logger.info('writing Spin2 lib')         
    fsock = open(os.path.join(aloha_path, 'ALOHALib.pkl'),'wb')
    cPickle.dump(lib, fsock, -1)
    aloha_lib.USE_TAG = old_tag
    return lib

if '__main__' == __name__:       
    logging.basicConfig(level=0)
    #create_library()
    import profile       
    #model 
      
    start = time.time()
    def main():
        alohagenerator = AbstractALOHAModel('sm') 
        alohagenerator.compute_all(save=False)
        return alohagenerator
    def write(alohagenerator):
        alohagenerator.write('/tmp/', 'Python')
    alohagenerator = main()
    logger.info('done in %s s' % (time.time()-start))
    write(alohagenerator)
    #profile.run('main()')
    #profile.run('write(alohagenerator)')
    stop = time.time()
    logger.info('done in %s s' % (stop-start))
  








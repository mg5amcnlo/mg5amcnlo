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


aloha_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('ALOHA')

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
        
        self.symmetries.append(outgoing)
        
    def write(self, output_dir, language='Fortran'):
        """ write the content of the object """

        getattr(aloha_writers, 'ALOHAWriterFor%s' % language)(self, output_dir).write()



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
        self.outgoing = None
        self.lorentz_expr = lorentz.structure        
        self.routine_kernel = None
        
    
    def compute_routine(self, mode):
        """compute the expression and return it"""
        self.outgoing = mode
        self.expr = self.compute_aloha_high_kernel(mode)
        return self.define_simple_output()
    
    def define_simple_output(self):
        """ define a simple output for this AbstractRoutine """
        
        infostr = str(self.lorentz_expr)        
        return AbstractRoutine(self.expr, self.outgoing, self.spins, self.name, \
                                                                        infostr)
        
        
    def compute_aloha_high_kernel(self, mode):
        """compute the abstract routine associate to this mode """
        
        #multiply by the wave functions
        nb_spinor = 0
        if not self.routine_kernel:
            AbstractRoutineBuilder.counter += 1
            logger.info('aloha creates %s routines' % self.name)
            try:       
                lorentz = eval(self.lorentz_expr)
            except NameError:
                print 'unknow type in Lorentz Evaluation'
                raise
            else:
                self.routine_kernel = lorentz
                
        else:
            lorentz = self.routine_kernel

        for (i, spin ) in enumerate(self.spins):
            id = i + 1
            
            #Check if this is the outgoing particle
            if id == self.outgoing:
                if spin == 1: 
                    lorentz *= complex(0,1)
                elif spin == 2:
                    nb_spinor += 1
                    if nb_spinor %2:
                        lorentz *= SpinorPropagator(id, 'I2', id)
                    else:
                        lorentz *= SpinorPropagator('I2', id, id) 
                elif spin == 3 :
                    lorentz *= VectorPropagator(id, 'I2', id)
                elif spin == 5 :
                    lorentz *= 1 # delayed evaluation (fastenize the code)
                else:
                    raise self.AbstractALOHAError(
                                'The spin value %s is not supported yet' % spin)
            else:
                # This is an incoming particle
                if spin == 1:
                    lorentz *= Scalar(id)
                elif spin == 2:
                    nb_spinor += 1
                    lorentz *= Spinor(id, id)
                elif spin == 3:        
                    lorentz *= Vector(id, id)
                elif spin == 5:
                    lorentz *= Spin2(10*id+1, 10*id+2, 'I2', 'I3', id)
                else:
                    raise self.AbstractALOHAError(
                                'The spin value %s is not supported yet' % spin)                    

        # If no particle OffShell
        if self.outgoing:
            lorentz /= DenominatorPropagator(self.outgoing)
            lorentz.tag.add('OM%s' % self.outgoing )  
            lorentz.tag.add('P%s' % self.outgoing)  
            lorentz.tag.add('W%s' % self.outgoing)  
        else:
            lorentz *= complex(0,-1)

          
        #lorentz = lorentz.simplify()
        lorentz = lorentz.expand()

        if self.spins[self.outgoing-1] == 5:
            if not self.aloha_lib:
                AbstractRoutineBuilder.load_library()
            lorentz *= self.aloha_lib[('Spin2Prop', id)]

        
        lorentz = lorentz.simplify()
        lorentz = lorentz.factorize()
        return lorentz         
        
    def compute_aloha_low_kernel(self, mode):
        """ compute the abstract routine associate to this mode """
        
        if not self.routine_kernel:
            self.routine_kernel = self.define_routine_kernel()
        
        if not AbstractRoutineBuilder.aloha_lib:
            AbstractRoutineBuilder.load_library()
        
        #multiply by the wave functions
        nb_spinor = 0
        lorentz = self.routine_kernel
        for (i, spin ) in enumerate(self.spins):
            id = i + 1
            
            #Check if this is the outgoing particle
            if id == self.outgoing:
                if spin == 1 : 
                    lorentz *= self.aloha_lib[('ScalarProp', id)]
                elif spin == 2 :
                    nb_spinor += 1
                    lorentz *= self.aloha_lib[('SpinorProp', id, nb_spinor %2)]
                elif spin == 3 :
                    lorentz *= self.aloha_lib[('VectorProp', id)]
                elif spin == 5 :
                    lorentz *= self.aloha_lib[('Spin2Prop', id)]
                else:
                    raise self.AbstractRoutineError(
                                'The spin value %s is not supported yet' % spin)
            else:
                # This is an incoming particle
                if spin == 1:
                    lorentz *= self.aloha_lib[('Scalar', id)]
                elif spin == 2:
                    nb_spinor += 1
                    lorentz *= self.aloha_lib[('Spinor', id)]
                elif spin == 3:        
                    lorentz *= self.aloha_lib[('Vector', id)]
                elif spin == 5:
                    lorentz *= self.aloha_lib[('Spin2', id)]
                else:
                    raise self.AbstractRoutineError(
                                'The spin value %s is not supported yet' % spin)                    
            
        # If no particle OffShell
        if self.outgoing:
            lorentz /= self.aloha_lib[('Denom', id)]
            lorentz.tag.add('OM%s' % self.outgoing )  
            lorentz.tag.add('P%s' % self.outgoing)  
            lorentz.tag.add('W%s' % self.outgoing)  
        else:
            lorentz *= complex(0,-1)
            
        lorentz = lorentz.simplify()
        lorentz = lorentz.factorize()
        return lorentz 
                
    def define_lorentz_expr(self, lorentz_expr):
        """Define the expression"""
        
        self.expr = lorentz_expr
    
    def define_routine_kernel(self, lorentz=None):
        """Define the kernel at low level"""
        
        if not lorentz:
            logger.info('compute kernel %s' % self.counter)
            AbstractRoutineBuilder.counter += 1  
            try:        
                lorentz = eval(self.lorentz_expr)
            except NameError, why:
                print dir(why)
                print why.args
                raise
                
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
        fsock = open(os.path.join(aloha_path, 'ALOHALib.pkl'), 'r')
        cls.aloha_lib = cPickle.load(fsock)
        

class AbstractALOHAModel(dict):
    """ A class to build and store the full set of Abstract ALOHA Routine"""

    files_from_template = ['makefile', 'sxxxxx.f','ixxxxx.f', 'oxxxxx.f',
                           'vxxxxx.f', 'txxxxx.f', 'pxxxxx.f']

    def __init__(self, model_name, write_dir=None):
        """ load the UFO model and init the dictionary """
        
        # load the UFO model
        python_pos = 'models.%s' % model_name 
        __import__(python_pos)
        self.model = sys.modules[python_pos]
        
        # find the position on the disk
        self.model_pos = os.path.dirname(self.model.__file__)

        #init the dictionary
        dict.__init__(self)
        self.symmetries = {}
        
        if write_dir:
            self.main(write_dir)
            
    def main(self, output_dir, format='Fortran'):
        """ Compute if not already compute. 
            Write file in models/MY_MODEL/MY_FORMAT.
            copy the file to output_dir
        """
        
        # Check if a pickle file exists
        if not self.load():
            self.compute_all()
        logger.info(' %s aloha routine' % len(self))
            
        # Check that output directory exists
        aloha_dir = os.path.join(self.model_pos, format.lower())
        logger.debug('aloha output dir is %s' %aloha_dir) 
        if not os.path.exists(aloha_dir):
            os.mkdir(aloha_dir)
        
        # Check that all routine are generated at default places:
        for (name, outgoing), abstract in self.items():
            routine_name = AbstractRoutineBuilder.get_routine_name(name, outgoing)
            if not glob.glob(os.path.join(aloha_dir, routine_name) + '.*'):
                abstract.write(output_dir, format)
        
        # Check that makefile and default file are up-to-date
        self.insertTemplate(output_dir, format)
        # Check aloha_file.inc
        self.write_aloha_file_inc(output_dir)
        
        # Copy model_routine in PROC
        
        
        
        
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
    
    def compute_all(self, save=True):
        """ define all the AbstractRoutine linked to a model """

        # Search identical particles in the vertices in order to avoid
        #to compute identical contribution
        self.look_for_symmetries()
        
        for lorentz in self.model.all_lorentz:
            if -1 in lorentz.spins:
                # No Ghost in ALOHA
                continue
            self.compute_for_lorentz(lorentz)
            if self.need_conjugate(lorentz):
                conjugate_lorentz = -1 * C('c1',1) * lorentz * C(2, 'c2')
                conjugate_lorentz.name += 'C'
                self.complex_mode = True
                self.compute_for_lorentz(conjugate_lorentz)
                self.complex_mode = False # <- automatic?
        
        if save:
            self.save()
        
    def compute_for_lorentz(self, lorentz):
        """ """
        self.compute_lorentz_with_kernel(lorentz)
        
    def compute_lorentz_without_kernel(self, lorentz):
        """define all the AbstractRoutine"""
        
        name = lorentz.name
        for outgoing in range(len(lorentz.spins) + 1 ):
            builder = AbstractRoutineBuilder(lorentz)
            wavefunction = builder.compute_routine(outgoing)
            self.set(name, outgoing, wavefunction)
        
        
    def compute_lorentz_with_kernel(self, lorentz):
        """ define all the AbstractRoutine linked to a given lorentz structure"""
        
        name = lorentz.name
        # first compute the amplitude contribution
        builder = AbstractRoutineBuilder(lorentz)
        wavefunction = builder.compute_routine(0)
        self.set(name, 0, wavefunction)
        
        # Create the routine associate to an externel particles
        for outgoing in range(1, len(lorentz.spins) + 1 ):
            symmetric = self.has_symmetries(lorentz.name, outgoing)
            if symmetric:
                self.get(lorentz.name, symmetric).add_symmetry(outgoing)
            else:
                wavefunction = builder.compute_routine(outgoing)
                #Store the information
                self.set(name, outgoing, wavefunction)

    def write(self, output_dir, language):
        """ write the full set of Helicity Routine in output_dir"""
        
        for abstract_routine in self.values():
            abstract_routine.write(output_dir, language)
        
        self.write_aloha_file_inc(output_dir)
        

    def look_for_symmetries(self):
        """Search some symmetries in the vertices.
        We search if some identical particles are in a vertices in order
        to avoid to compute symmetrical contributions"""
        
        for vertex in self.model.all_vertices:
            for i, part1 in enumerate(vertex.particles):
                for j in range(i):
                    part2 = vertex.particles[j]
                    if part1.name == part2.name:
                        for lorentz in vertex.lorentz:
                            if self.symmetries.has_key(lorentz.name):
                                self.symmetries[lorentz.name][i+1] = j+1
                            else:
                                self.symmetries[lorentz.name] = {i+1:j+1}
                        break
                    
    def has_symmetries(self, l_name, outgoing, out=None):
        """ This returns out if no symmetries are available, otherwise it finds 
        the lowest equivalent outgoing by recursivally calling this function"""
    
        try:
            equiv = self.symmetries[l_name][outgoing]
        except:
            return out
        else:
            return self.has_symmetries(l_name, equiv, out=equiv)
    
    def need_conjugate(self, lorentz):
        return False
        
def write_aloha_file_inc(aloha_dir,file_ext, comp_ext):
    """find the list of Helicity routine in the directory and create a list 
    of those files (but with compile extension)"""

    aloha_files = []
    
    # Identify the valid files
    alohafile_pattern = re.compile(r'''^[STFV]*[_\d]*_\d%s''' % file_ext)
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
    
    
    lib = {} # key: (name, part_nb, special) -> object
    for i in range(1, 10):
        logger.info('step %s/9' % i)
        lib[('Scalar', i)] = create( Scalar(i) )
        lib[('ScalarProp', i)] = complex(0,1)
        lib[('Denom', i )] = create( DenominatorPropagator(i) )
        lib[('Spinor', i )] = create( Spinor(i, i) )
        lib[('SpinorProp', i, 0)] = create( SpinorPropagator(i, 'I2', i) )
        lib[('SpinorProp', i, 1)] = create( SpinorPropagator('I2', i, i) )
        lib[('Vector', i)] = create( Vector(i+1, i+1) )
        lib[('VectorProp', i)] = create( VectorPropagator(i,'I2', i) )
        lib[('Spin2', i )] = create( Spin2(10*i+1, 10*i+2, i) )
        lib[('Spin2Prop',i)] = create( Spin2Propagator(10*i+1, \
                                            10*i+2,'I2','I3', i) )
    logger.info('writing')         
    fsock = open('./ALOHALib.pkl','wb')
    cPickle.dump(lib, fsock, -1)
    logger.info('done')
    
if '__main__' == __name__:       
    logging.basicConfig(level=0)
    #create_library()
    import profile       
    #model 
      
    start = time.time()
    def main():
        alohagenerator = AbstractALOHAModel('sm') 
        alohagenerator.compute_all()
    def write(alohagenerator):
        alohagenerator.write('/tmp/', 'Fortran')
    main()
    #profile.run('main()')
    #profile.run('write(alohagenerator)')
    stop = time.time()
    logger.info('done in %s s' % (stop-start))
  








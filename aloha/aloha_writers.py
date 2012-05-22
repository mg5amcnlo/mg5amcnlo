try:
    import madgraph.iolibs.file_writers as writers 
except:
    import aloha.file_writers as writers

import aloha
import aloha.aloha_lib as aloha_lib
import os
import re 
from numbers import Number
from collections import defaultdict
from fractions import Fraction
# fast way to deal with string
from cStringIO import StringIO
# Look at http://www.skymind.com/~ocrow/python_string/ 
# For knowing how to deal with long strings efficiently.



class WriteALOHA: 
    """ Generic writing functions """ 
    
    power_symbol = '**'
    change_number_format = str
    extension = ''
    type_to_variable = {2:'F',3:'V',5:'T',1:'S'}
    type_to_size = {'S':3, 'T':18, 'V':6, 'F':6}
    
            
    def __init__(self, abstract_routine, dirpath):

        name = get_routine_name(abstract = abstract_routine)

        if dirpath:
            self.dir_out = dirpath
            self.out_path = os.path.join(dirpath, name + self.extension)
        else:
            self.out_path = None
            self.dir_out = None

        self.routine = abstract_routine
        self.tag = self.routine.tag
        self.name = name
        self.particles =  [self.type_to_variable[spin] for spin in \
                          abstract_routine.spins]
        
        self.offshell = abstract_routine.outgoing # position of the outgoing in particle list        
        self.outgoing = self.offshell             # expected position for the argument list
        if 'C%s' %((self.outgoing + 1) // 2) in self.tag:
            #flip the outgoing tag if in conjugate
            self.outgoing = self.outgoing + self.outgoing % 2 - (self.outgoing +1) % 2
        
        self.outname = '%s%s' % (self.particles[self.offshell -1], \
                                                               self.outgoing)
        
        #initialize global helper routine
        self.declaration = Declaration_list()

                                   
                                       
    def pass_to_HELAS(self, indices, start=0):
        """find the Fortran HELAS position for the list of index""" 
        
        
        if len(indices) == 1:
            return indices[0] + start

        ind_name = self.routine.expr.lorentz_ind

        if ind_name == ['I3', 'I2']:
            return  4 * indices[1] + indices[0] + start 
        elif len(indices) == 2: 
            return  4 * indices[0] + indices[1] + start 
        else:
            raise Exception, 'WRONG CONTRACTION OF LORENTZ OBJECT for routine %s: %s' \
                    % (self.name, ind_name)                                 
                                 
    def get_header_txt(self): 
        """ Prototype for language specific header""" 
        raise Exception, 'THis function should be overwritten'
        return ''
    
    def get_declaration_txt(self):
        """ Prototype for how to write the declaration of variable"""
        return ''

    def define_content(self): 
        """Prototype for language specific body""" 
        pass

    def get_momenta_txt(self):
        """ Prototype for the definition of the momenta"""
        raise Exception, 'THis function should be overwritten'

    def get_momentum_conservation_sign(self):
        """find the sign associated to the momentum conservation"""

        # help data 
        signs = []
        nb_fermion =0
        
        #compute global sign
        if not self.offshell % 2 and self.particles[self.offshell -1] == 'F': 
            global_sign = 1
        else:
            global_sign = -1
        
        flipped = [2*(int(c[1:])-1) for c in self.tag if c.startswith('C')]
        for index, spin in enumerate(self.particles):
            assert(spin in ['S','F','V','T'])  
                  
            #compute the sign
            if spin != 'F':
                sign = -1 * global_sign
            elif nb_fermion % 2 == 0:
                sign = global_sign
                nb_fermion += 1
                if index in flipped:
                    sign *= -1
            else: 
                sign = -1 * global_sign
                nb_fermion += 1
                if index-1 in flipped:
                    sign *= -1
            
            # No need to include the outgoing particles in the definitions
            if index == self.outgoing -1:
                signs.append('0*')
                continue     
                
            if sign == 1:    
                signs.append('+')
            else:
                signs.append('-')
        return signs


    def get_P_sign(self, index):

        type = self.particles[index - 1]
        energy_pos = self.type_to_size[type] -1
        sign = 1
        if self.outgoing == index and type in ['V','S']:
            sign = -1
        if 'C%s' % ((index +1) // 2)  in self.tag: 
            if index == self.outgoing:
                pass
            elif index % 2 and index -1 != self.outgoing:
                pass
            elif index % 2 == 1 and index + 1  != self.outgoing:
                pass
            else:
                sign *= -1
        
        if sign == -1 :
            return '-'
        else:
            return ''
        
        
        
        
    
    def get_foot_txt(self):
        """Prototype for language specific footer"""
        return ''
    
    def define_argument_list(self, couplings=['COUP']):
        """define a list with the string of object required as incoming argument"""

        call_arg = [] #incoming argument of the routine

        conjugate = [2*(int(c[1:])-1) for c in self.tag if c[0] == 'C']
        
        for index,spin in enumerate(self.particles):
            if self.offshell == index + 1:
                continue
            
            if index in conjugate:
                index2, spin2 = index+1, self.particles[index+1]
                call_arg.append(('complex','%s%d' % (spin2, index2 +1))) 
                #call_arg.append('%s%d' % (spin, index +1)) 
            elif index-1 in conjugate:
                index2, spin2 = index-1, self.particles[index-1]
                call_arg.append(('complex','%s%d' % (spin2, index2 +1))) 
            else:
                call_arg.append(('complex','%s%d' % (spin, index +1)))
        
        for coup in couplings:       
            call_arg.append(('complex', coup))              
            self.declaration.add(('complex',coup))
            
        if self.offshell:
            if aloha.complex_mass:
                call_arg.append(('complex','M%s' % self.outgoing))              
                self.declaration.add(('complex','M%s' % self.outgoing))
            else:
                call_arg.append(('double','M%s' % self.outgoing))              
                self.declaration.add(('double','M%s' % self.outgoing))                
                call_arg.append(('double','W%s' % self.outgoing))              
                self.declaration.add(('double','W%s' % self.outgoing))
            
        self.call_arg = call_arg
                
        return call_arg

    def write(self, mode=None):
                         
        self.mode = mode
        
        self.define_argument_list()
        core_text = self.define_expression()    
        out = StringIO()
        
        out.write(self.get_header_txt())
        out.write(self.get_declaration_txt())
        out.write(self.get_momenta_txt())
        out.write(core_text)
        out.write(self.get_foot_txt())

        for elem in self.routine.symmetries:
            out.write('\n')
            out.write(self.define_symmetry(elem))

        text = out.getvalue()
        
        if self.out_path:        
            writer = self.writer(self.out_path)
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer.write_comments(commentstring)
            writer.write(text)
        print text
        return text + '\n'





    
    def write_indices_part(self, indices, obj): 
        """Routine for making a string out of indices objects"""
        
        text = 'output(%s)' % indices
        return text                 
        
    def write_obj(self, obj, prefactor=True):
        """Calls the appropriate writing routine"""
        
        try:
            vartype = obj.vartype
        except:
            return self.change_number_format(obj)

        # The order is from the most current one to the les probable one
        if vartype == 1 : # AddVariable
            return self.write_obj_Add(obj, prefactor)
        elif vartype == 2 : # MultVariable
            return self.write_MultVariable(obj, prefactor)
        elif vartype == 6 : # MultContainer
            return self.write_MultContainer(obj, prefactor) 
        elif vartype == 0 : # MultContainer
            return self.write_variable(obj)               
        else: 
            raise Exception('Warning unknown object: %s' % obj.vartype)

    def write_MultVariable(self, obj, prefactor=True):
        """Turn a multvariable into a string"""
        
        mult_list = [self.write_variable_id(id) for id in obj]
        data = {'factors': '*'.join(mult_list)}
        if prefactor and obj.prefactor != 1:
            if obj.prefactor != -1:
                text = '%(prefactor)s * %(factors)s'
                data['prefactor'] = self.change_number_format(obj.prefactor)
            else:
                text = '-%(factors)s'
        else:
            text = '%(factors)s'
        return text % data

    def write_MultContainer(self, obj, prefactor=True):
        """Turn a multvariable into a string"""

        mult_list = [self.write_obj(id) for id in obj]
        data = {'factors': '*'.join(mult_list)}
        if prefactor and obj.prefactor != 1:
            if obj.prefactor != -1:
                text = '%(prefactor)s * %(factors)s'
                data['prefactor'] = self.change_number_format(obj.prefactor)
            else:
                text = '-%(factors)s'
        else:
            text = '%(factors)s'
        return text % data
         
    
    def write_obj_Add(self, obj, prefactor=True):
        """Turns addvariable into a string"""

        data = defaultdict(list)
        number = []
        [data[p.prefactor].append(p) if hasattr(p, 'prefactor') else number.append(p)
             for p in obj]

        file_str = StringIO()
        
        if prefactor and obj.prefactor != 1:
            file_str.write(self.change_number_format(obj.prefactor))
            file_str.write('*(')
        else:
            file_str.write('(')
        first=True
        for value, obj_list in data.items():
            add= '+'
            if value not in  [-1,1]:
                nb_str = self.change_number_format(value)
                if nb_str[0] in ['+','-']:
                    file_str.write(nb_str)
                else:
                    file_str.write('+')
                    file_str.write(nb_str)
                file_str.write('*(')
            elif value == -1:
                add = '-' 
                file_str.write('-')
            elif not first:
                file_str.write('+')
            else:
                file_str.write('')
            first = False
            file_str.write(add.join([self.write_obj(obj, prefactor=False) 
                                                          for obj in obj_list]))
            if value not in [1,-1]:
                file_str.write(')')
                
        file_str.write(')')
        return file_str.getvalue()
                
    def write_variable(self, obj):
        return self.change_var_format(obj)
    
    def write_variable_id(self, id):
        
        obj = aloha_lib.KERNEL.objs[id]
        return self.write_variable(obj)   
    
    def change_var_format(self, obj):
        """format the way to write the variable and add it to the declaration list
        """

        str_var = str(obj)
        
        self.declaration.add((obj.type, str_var))        
        return str_var


    
    def make_call_list(self, outgoing=None):
        """find the way to write the call of the functions"""

        if outgoing is None:
            outgoing = self.offshell

        call_arg = [] #incoming argument of the routine

        conjugate = [2*(int(c[1:])-1) for c in self.tag if c[0] == 'C']
        
        for index,spin in enumerate(self.particles):
            if self.offshell == index + 1:
                continue
            
            if index in conjugate:
                index2, spin2 = index+1, self.particles[index+1]
                call_arg.append('%s%d' % (spin2, index2 +1)) 
                #call_arg.append('%s%d' % (spin, index +1)) 
            elif index-1 in conjugate:
                index2, spin2 = index-1, self.particles[index-1]
                call_arg.append('%s%d' % (spin2, index2 +1)) 
            else:
                call_arg.append('%s%d' % (spin, index +1)) 
                
        return call_arg

    
    def make_declaration_list(self):
        """ make the list of declaration nedded by the header """
        
        declare_list = []
        
        
        for index, spin in enumerate(self.particles):
            # First define the size of the associate Object 
            declare_list.append(self.declare_dict[spin] % (index + 1) ) 
 
        return declare_list
 
 
 
 
     
class ALOHAWriterForFortran(WriteALOHA): 
    """routines for writing out Fortran"""
    
    extension = '.f'
    writer = writers.FortranWriter

    type2def = {}    
    type2def['int'] = 'integer*4'
    if aloha.mp_precision:
        type2def['double'] = 'real*16'
        type2def['complex'] = 'complex*32'
        format = 'q0'
    else:
        type2def['double'] = 'real*8'
        type2def['complex'] = 'complex*16'
        
        format = 'd0'
    
    
    def get_header_txt(self, name=None, couplings=['COUP']):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        out = StringIO()
        # define the type of function and argument
        
        arguments = [arg for format, arg in self.define_argument_list(couplings)]
        if not self.offshell:
            output = 'vertex'
            self.declaration.add(('complex','vertex'))
        else:
            output = '%(spin)s%(id)d' % {
                     'spin': self.particles[self.offshell -1],
                     'id': self.outgoing}
            self.declaration.add(('list_complex', output))
        
        out.write('subroutine %(name)s(%(args)s,%(output)s)\n' % \
                  {'output':output, 'name': name, 'args': ', '.join(arguments)})
        
        return out.getvalue() 
    
    def get_declaration_txt(self):
        """ Prototype for how to write the declaration of variable"""
        
        out = StringIO()
        out.write('implicit none\n')
        print self.call_arg
        argument_var = [name for type,name in self.call_arg]
        print argument_var
        print self.declaration
        for type, name in self.declaration:
            if type.startswith('list'):
                type = type[5:]
                #determine the size of the list
                if name in argument_var:
                    size ='*'
                elif name.startswith('P'):
                    size='0:3'
                elif name[0] in ['F','V']:
                    if aloha.loop_mode:
                        size = 8
                    else:
                        size = 6
                elif name[0] == 'S':
                    if aloha.loop_mode:
                        size = 5
                    else:
                        size = 3
                elif name[0] in ['R','T']: 
                    if aloha.loop_mode:
                        size = 20
                    else:
                        size = 18
                else:
                    size = '*'
    
                out.write(' %s %s(%s)\n' % (self.type2def[type], name, size))
            else:
                out.write(' %s %s\n' % (self.type2def[type], name))

        print out.getvalue()
        return out.getvalue()
        
    def get_momenta_txt(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
                    
        out = StringIO()
        
        # Define all the required momenta
        p = [] # a list for keeping track how to write the momentum
        
        signs = self.get_momentum_conservation_sign()
        
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
                out.write("    OM{0} = {1}\n    if (M{0}.ne.{1}) OM{0}={2}/M{0}**2\n".format( 
                         i+1, self.change_number_format(0), self.change_number_format(1)))
            
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                energy_pos = self.type_to_size[type] -2
                p.append('%s%s%s({0})' % (signs[i],type,i+1))    
                
            if self.declaration.is_used('P%s' % (i+1)):
                self.get_one_momenta_def(i+1, out)
                
        # define the resulting momenta
        if self.offshell:
            energy_pos = out_size -2
            type = self.particles[self.outgoing-1]
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 2
            for i in range(size_p):
                out.write('    %s%s(%s) = %s\n' % (type,self.outgoing, energy_pos+1+i, 
                                             ''.join(p).format(energy_pos+1+i)))
            
            
            self.get_one_momenta_def(self.outgoing, out)

        
        # Returning result
        return out.getvalue()

    def get_one_momenta_def(self, i, strfile):
        
        type = self.particles[i-1]
        energy_pos = self.type_to_size[type] -2
        
        if aloha.loop_mode:
            template ='P%(i)d(%(j)d) = %(sign)s%(type)s%(i)d(%(nb)d)\n'
        else:
            template ='P%(i)d(%(j)d) = %(sign)s%(operator)s(%(type)s%(i)d(%(nb2)d))\n'

        nb2 = energy_pos + 1
        for j in range(4):
            if not aloha.loop_mode:
                nb = energy_pos + j
                if j == 0: 
                    assert not aloha.mp_precision 
                    operator = 'dble' # not suppose to pass here in mp
                elif j == 1: 
                    nb2 += 1
                elif j == 2:
                    assert not aloha.mp_precision 
                    operator = 'dimag' # not suppose to pass here in mp
                elif j ==3:
                    nb2 -= 1
            else:
                operator =''
                nb = energy_pos + 1+ j
                nb2 = energy_pos + 1 + j
            strfile.write(template % {'j':j,'type': type, 'i': i, 
                        'nb': nb, 'nb2': nb2, 'operator':operator,
                        'sign': self.get_P_sign(i)})  
            
              
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
        if '_' in name:
            type = name.type
            decla = name.split('_',1)[0]
            name = name.replace('_', '(', 1) + ')'
            self.declaration.add(('list_%s' % type, decla))
        else:
            self.declaration.add((name.type, name.split('_',1)[0]))
        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
        return name
  
    def change_number_format(self, number):
        """Formating the number"""

        def isinteger(x):
            try:
                return int(x) == x
            except TypeError:
                return False

        if isinteger(number):
            out = '%s%s' % (str(int(number)),self.format)
        elif isinstance(number, complex):
            if number.imag:
                out = '(%s, %s)' % (self.change_number_format(number.real), \
                                    self.change_number_format(number.imag))
            else:
                out = '%s' % (self.change_number_format(number.real))
        else:
            tmp = Fraction(number)
            tmp = tmp.limit_denominator(100)
            out = '%s%s/%s%s' % (tmp.numerator, self.format, tmp.denominator, self.format)
        return out
    
    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            for name,obj in self.routine.contracted.items():
                out.write(' %s = %s\n' % (name, self.write_obj(obj)))

        numerator = self.routine.expr
        
        if not self.offshell:
            out.write(' vertex = COUP*%s\n' % self.write_obj(numerator.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            if 'L' not in self.tag:
                coeff = 'denom'
                out.write('    denom = COUP/(P%(i)s(0)**2-P%(i)s(1)**2-P%(i)s(2)**2-P%(i)s(3)**2 - M%(i)s * (M%(i)s -(0,1)* W%(i)s))\n' % \
                      {'i': self.outgoing})
                self.declaration.add(('complex','denom'))
            else:
                coeff = 'COUP'
                
            for ind in numerator.listindices():
                out.write('    %s(%d)= %s*%s\n' % (self.outname, 
                                        self.pass_to_HELAS(ind)+1, coeff,
                                        self.write_obj(numerator.get_rep(ind))))
        return out.getvalue()

    def define_symmetry(self, new_nb, couplings=['COUP']):
        number = self.offshell
        arguments = [name for format, name in self.define_argument_list()]
        new_name = self.name.rsplit('_')[0] + '_%s' % new_nb
        return '%s\n    call %s(%s)' % \
            (self.get_header_txt(new_name, couplings), self.name, ','.join(arguments))

    def get_foot_txt(self):
        return 'end\n\n' 

    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
            
        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        # write head - momenta - body - foot
        text = StringIO()
        routine = StringIO()
        data = {} # for the formating of the line
                    
        # write header 
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        text.write(self.get_header_txt(name=name, couplings=new_couplings))
  
        # Define which part of the routine should be called
        data['addon'] = ''.join(self.tag) + '_%s' % self.offshell

        # how to call the routine
        argument = [name for format, name in self.define_argument_list(new_couplings)]
        index= argument.index('COUP1')
        data['before_coup'] = ','.join(argument[:index])
        data['after_coup'] = ','.join(argument[index+len(lor_names)+1:])
        if data['after_coup']:
            data['after_coup'] = ',' + data['after_coup']
            
        lor_list = (self.routine.name,) + lor_names
        line = "    call %(name)s%(addon)s(%(before_coup)s,%(coup)s%(after_coup)s,%(out)s)\n"
        main = '%(spin)s%(id)d' % {'spin': self.particles[self.offshell -1],
                           'id': self.outgoing}
        for i, name in enumerate(lor_list):
            data['name'] = name
            data['coup'] = 'COUP%d' % (i+1)
            if i == 0:
                if  not offshell: 
                    data['out'] = 'vertex'
                else:
                    data['out'] = main
            elif i==1:
                if self.offshell:
                    type = self.particles[self.offshell-1]
                    self.declaration.add(('list_complex','%stmp' % type))
                else:
                    type = ''
                    self.declaration.add(('complex','tmp'))
                data['out'] = '%stmp' % type
            routine.write(line % data)
            if i:
                if not offshell:
                    routine.write( '    vertex = vertex + tmp\n')
                else:
                    size = self.type_to_size[self.particles[offshell -1]] -2
                    routine.write(" do i = 1, %s\n" % size)
                    routine.write("        %(main)s(i) = %(main)s(i) + %(tmp)s(i)\n" %\
                               {'main': main, 'tmp': data['out']})
                    routine.write(' enddo\n')
                    self.declaration.add(('int','i'))
                   
        text.write(self.get_declaration_txt())
        text.write(routine.getvalue())
        text.write(self.get_foot_txt())

        #ADD SYMETRY
        if sym:
            for elem in self.routine.symmetries:
                text.write(self.write_combined(lor_names, mode, elem))


        text = text.getvalue()
        if self.out_path:        
            writer = self.writer(self.out_path,'a')
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer.write_comments(commentstring)
            writer.write(text)
        return text

class ALOHAWriterForFortranQP(ALOHAWriterForFortran): 
    """routines for writing out Fortran"""
    
    type2def = {}    
    type2def['int'] = 'integer*4'
    type2def['double'] = 'real*16'
    type2def['complex'] = 'complex*32'
    format = 'q0'
    

class ALOHAWriterForFortranLoop(ALOHAWriterForFortran): 
    """routines for writing out Fortran"""

    def __init__(self, abstract_routine, dirpath):

        
        ALOHAWriterForFortran.__init__(self, abstract_routine, dirpath)
        print 'pass here in LOOP'
        # position of the outgoing in particle list        
        self.l_id = [int(c[1:]) for c in abstract_routine.tag if c[0] == 'L'][0] 
        self.l_helas_id = self.l_id   # expected position for the argument list
        if 'C%s' %((self.outgoing + 1) // 2) in abstract_routine.tag:
            #flip the outgoing tag if in conjugate
            self.l_helas_id += self.l_id % 2 - (self.l_id +1) % 2 
       
            
        print 'define ', self.l_helas_id
        

    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            for name,obj in self.routine.contracted.items():
                out.write(' %s = %s\n' % (name, self.write_obj(obj)))


        OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                              self.offshell)

        for key,expr in self.routine.expr.items():
            arg = self.get_loop_argument(key)
            for ind in expr.listindices():
                data = expr.get_rep(ind)
                if data:
                    out.write('    COEFF(%s,%s)= denom*%s\n' % ( 
                                    self.pass_to_HELAS(ind)+1, ','.join(arg), 
                                    self.write_obj(data)))
                else:
                    out.write('    COEFF(%s,%s)= %s\n' % ( 
                                    self.pass_to_HELAS(ind)+1, ','.join(arg), 
                                    self.write_obj(data)))                    
        return out.getvalue()
    
    def define_argument_list(self, couplings=['COUP']):
        """define a list with the string of object required as incoming argument"""


        conjugate = [2*(int(c[1:])-1) for c in self.tag if c[0] == 'C']
        call_arg = [('list_complex', 'P%s'% self.l_helas_id)] #incoming argument of the routine
        self.declaration.add(call_arg[0])
        
        for index,spin in enumerate(self.particles):
            if self.offshell == index + 1:
                continue
            if self.l_helas_id == index + 1:
                continue
            
            if index in conjugate:
                index2, spin2 = index+1, self.particles[index+1]
                call_arg.append(('complex','%s%d' % (spin2, index2 +1))) 
                #call_arg.append('%s%d' % (spin, index +1)) 
            elif index-1 in conjugate:
                index2, spin2 = index-1, self.particles[index-1]
                call_arg.append(('complex','%s%d' % (spin2, index2 +1))) 
            else:
                call_arg.append(('complex','%s%d' % (spin, index +1)))
        
        for coup in couplings:       
            call_arg.append(('complex', coup))              
            self.declaration.add(('complex',coup))
            
        if self.offshell:
            if aloha.complex_mass:
                call_arg.append(('complex','M%s' % self.outgoing))              
                self.declaration.add(('complex','M%s' % self.outgoing))
            else:
                call_arg.append(('double','M%s' % self.outgoing))              
                self.declaration.add(('double','M%s' % self.outgoing))                
                call_arg.append(('double','W%s' % self.outgoing))              
                self.declaration.add(('double','W%s' % self.outgoing))
            
        self.call_arg = call_arg
                
        return call_arg

    def get_momenta_txt(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
                    
        out = StringIO()
        
        # Define all the required momenta
        p = [] # a list for keeping track how to write the momentum
        size = []
        
        signs = self.get_momentum_conservation_sign()
        
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
                out.write("    OM{0} = {1}\n    if (M{0}.ne.{1}) OM{0}={2}/M{0}**2\n".format( 
                         i+1, self.change_number_format(0), self.change_number_format(1)))
            
            if i+1 == self.outgoing:
                out_type = 'P'
                continue
            elif i+1 == self.l_helas_id:
                p.append('%sP%s({%s})' % (signs[i],i+1,len(size))) 
                size.append(0)
                continue
            elif self.offshell:
                p.append('%s%s%s({%s})' % (signs[i],type,i+1,len(size)))
                size.append(self.type_to_size[type] -1)
                
            if self.declaration.is_used('P%s' % (i+1)):
                    self.get_one_momenta_def(i+1, out)
                
        # define the resulting momenta
        if self.offshell:
            type = self.particles[self.outgoing-1]
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 2
            for i in range(size_p):
                out.write('    %s%s(%s) = %s\n' % (type,self.outgoing, i, 
                                             ''.join(p).format(*[s+i for s in size])))

        
        # Returning result
        return out.getvalue()
  

    def get_loop_argument(self, key):
        """return the position for the argument in the HELAS convention"""
        
        loop_momentum = key[:4]
        basis = key[4:]
        
        loop_pos = sum([loop_momentum[i] * (i+1) for i in range(4)])
        basis_pos = sum([basis[i] * (i+1) for i in range(len(basis))])
        return (str(loop_pos), str(basis_pos))
        

        
        
        
        
    def get_header_txt(self, name=None, couplings=['COUP']):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        out = StringIO()
        # define the type of function and argument
        
        arguments = [arg for format, arg in self.define_argument_list(couplings)]
        self.declaration.add(('list_complex', 'P%s'% self.outgoing))
        self.declaration.add(('list_complex', 'P%s'% self.l_helas_id))        
        self.declaration.add(('list_complex', 'coeff'))
        out.write('subroutine %(name)s(%(args)s, P%(out)s, COEFF)\n' % \
                  {'name': name, 'args': ', '.join(arguments),
                   'out':self.outgoing})
        
        return out.getvalue() 

class ALOHAWriterForFortranLoopQP(ALOHAWriterForFortranQP): 
    """routines for writing out Fortran"""        

def get_routine_name(name=None, outgoing=None, tag=None, abstract=None):
    """ build the name of the aloha function """
    
    assert (name and outgoing) or abstract

    if tag is None:
        tag = abstract.tag

    if name is None:
        name = abstract.name + ''.join(tag)
    
    if outgoing is None:
        outgoing = abstract.outgoing

    
    return '%s_%s' % (name, outgoing)

def combine_name(name, other_names, outgoing, tag=None):
    """ build the name for combined aloha function """
    

    # Two possible scheme FFV1C1_2_X or FFV1__FFV2C1_X
    # If they are all in FFVX scheme then use the first
    p=re.compile('^(?P<type>[FSVT]+)(?P<id>\d+)')
    routine = ''
    if p.search(name):
        base, id = p.search(name).groups()
        if tag is not None:
            routine = name + ''.join(tag)
        else:
            routine = name
        for s in other_names:
            try:
                base2,id2 = p.search(s).groups()
            except:
                routine = ''
                break # one matching not good -> other scheme
            if base != base2:
                routine = ''
                break  # one matching not good -> other scheme
            else:
                routine += '_%s' % id2
    if routine:
        return routine +'_%s' % outgoing

    if tag is not None:
        addon = ''.join(tag)
    else:
        addon = ''
        if 'C' in name:
            short_name, addon = name.split('C',1)
            try:
                addon = 'C' + str(int(addon))
            except:
                addon = ''
            else:
                name = short_name

    return '_'.join((name,) + tuple(other_names)) + addon + '_%s' % outgoing
 

class ALOHAWriterForCPP(WriteALOHA): 
    """Routines for writing out helicity amplitudes as C++ .h and .cc files."""
    
    def __init__(self,*args,**err):
        raise NotImplemented
#    
#    declare_dict = {'S':'double complex S%d[3]',
#                    'F':'double complex F%d[6]',
#                    'V':'double complex V%d[6]',
#                    'T':'double complex T%s[18]'}
#    
#    def define_header(self, name=None):
#        """Define the headers for the C++ .h and .cc files. This include
#            - function tag
#            - definition of variable
#            - momentum conservation
#        """
#        
#        if name is None:
#            name = self.name
#            
#        #Width = self.collected['width']
#        #Mass = self.collected['mass']
#        
#        CallList = self.calllist['CallList'][:]
#        
#        local_declare = []
#        OffShell = self.offshell
#        OffShellParticle = OffShell -1 
#        # Transform function call variables to C++ format
#        for i, call in enumerate(CallList):
#            CallList[i] = "complex<double> %s[]" % call
#        #if Mass:
#        #    Mass[0] = "double %s" % Mass[0]
#        #if Width:
#        #    Width[0] = "double %s" % Width[0]
#        
#        # define the type of function and argument
#        if not OffShell:
#            str_out = 'void %(name)s(%(args)s, complex<double>& vertex)' % \
#               {'name': name,
#                'args': ','.join(CallList + ['complex<double> COUP'])}
#        else: 
#            str_out = 'void %(name)s(%(args)s, double M%(number)d, double W%(number)d, complex<double>%(out)s%(number)d[])' % \
#              {'name': name,
#               'args': ','.join(CallList+ ['complex<double> COUP']),
#               'out': self.particles[self.outgoing - 1],
#               'number': self.outgoing 
#               }
#
#        h_string = str_out + ";\n\n"
#        cc_string = str_out + "{\n"
#        if self.routine.contracted:
#            lstring = []
#            for tag in self.routine.contracted['order']:
#                name, obj, nb = self.routine.contracted[tag]
#                lstring.append('complex<double> %s;\n' % name)
#            lstring.append('')
#            str_out += '\n'.join(lstring)
#
#
#        return {'h_header': h_string, 'cc_header': cc_string}
#            
#    def define_momenta(self):
#        """Write the expressions for the momentum of the outgoing
#        particle."""
#
#        momenta = self.collected['momenta']
#        overm = self.collected['om']
#        momentum_conservation = self.calllist['Momentum']
#        
#        str_out = ''
#        # Declare auxiliary variables
#        if self.offshell:
#            str_out += 'complex<double> denom;\n'
#        if len(overm) > 0: 
#            str_out += 'complex<double> %s;\n' % ','.join(overm)
#        if len(momenta) > 0:
#            str_out += 'double %s[4];\n' % '[4],'.join(momenta)
#
#        # Energy
#        if self.offshell: 
#            offshelltype = self.particles[self.offshell -1]
#            offshell_size = self.type_to_size[offshelltype]            
#            #Implement the conservation of Energy Impulsion
#            for i in range(-2,0):
#                str_out += '%s%d[%d]= ' % (offshelltype, self.outgoing,
#                                           offshell_size + i)
#                
#                pat=re.compile(r'^[-+]?(?P<spin>\w)')
#                for elem in momentum_conservation:
#                    spin = pat.search(elem).group('spin') 
#                    str_out += '%s[%d]' % (elem, self.type_to_size[spin] + i)  
#                str_out += ';\n'
#        
#        # Momentum
#        for mom in momenta:
#            #Mom is in format PX with X the number of the particle
#            index = int(mom[1:])
#            
#            type = self.particles[index - 1]
#            energy_pos = self.type_to_size[type] - 2
#            sign = 1
#            if self.offshell == index and type in ['V', 'S']:
#                sign = -1
#            if 'C%s' % ((index +1) // 2)  in self.tag: 
#                if index == self.outgoing:
#                    pass
#                elif index % 2 and index -1 != self.outgoing:
#                    pass
#                elif index %2 == 1 and index + 1  != self.outgoing:
#                    pass
#                else:
#                    sign *= -1
#            
#            if sign == -1 :
#                sign = '-'
#            else:
#                sign = ''
#                   
#            str_out += '%s[0] = %s%s%d[%d].real();\n' % (mom, sign, type, index, energy_pos)
#            str_out += '%s[1] = %s%s%d[%d].real();\n' % (mom, sign, type, index, energy_pos + 1)
#            str_out += '%s[2] = %s%s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos + 1)
#            str_out += '%s[3] = %s%s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos)            
#            
#        # Definition for the One Over Mass**2 terms
#        for elem in overm:
#            #Mom is in format OMX with X the number of the particle
#            index = int(elem[2:])
#            str_out += 'OM%d = 0;\n' % (index)
#            str_out += 'if (M%d != 0) OM%d' % (index, index) + '= 1./pow(M%d,2);\n' % (index) 
#        
#        # Returning result
#        return str_out
#        
#        
#    def change_var_format(self, name): 
#        """Format the variable name to C++ format"""
#        
#        if '_' in name:
#            name = name.replace('_','[',1) +']'
#        outstring = ''
#        counter = 0
#        for elem in re.finditer('[FVTSfvts][0-9]\[[0-9]\]',name):
#            outstring += name[counter:elem.start()+2]+'['+str(int(name[elem.start()+3:elem.start()+4])-1)+']'
#            counter = elem.end()
#        outstring += name[counter:]
#        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
#        return outstring
#    
#    def change_number_format(self, number):
#        """Format numbers into C++ format"""
#        if isinstance(number, complex):
#            if number.real == int(number.real) and \
#                   number.imag == int(number.imag):
#                out = 'complex<double>(%d., %d.)' % \
#                      (int(number.real), int(number.imag))
#            else:
#                out = 'complex<double>(%.9f, %.9f)' % \
#                      (number.real, number.imag)                
#        else:
#            if number == int(number):
#                out = '%d.' % int(number)
#            else:
#                out = '%.9f' % number
#        return out
#    
#    def define_expression(self):
#        """Write the helicity amplitude in C++ format"""
#        OutString = '' 
#
#        if self.routine.contracted:        
#            string = ''
#            for tag in self.routine.contracted['order']:
#                name, obj, nb = self.routine.contracted[tag]
#                string += '%s = %s ! used %s times\n' % (name, self.write_obj(obj),nb)
#            string = string.replace('+-', '-')
#        
#        if not self.offshell:
#            for ind in self.obj.listindices():
#                string = 'vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
#                string = string.replace('+-', '-')
#                OutString = OutString + string + ';\n'
#        else:
#            OffShellParticle = self.particles[self.offshell-1]+'%s'%(self.outgoing)
#            numerator = self.obj.numerator
#            denominator = self.obj.denominator
#            for ind in denominator.listindices():
#                denom = self.write_obj(denominator.get_rep(ind))
#            string = 'denom =' + '1./(' + denom + ')'
#            string = string.replace('+-', '-')
#            OutString = OutString + string + ';\n'
#            for ind in numerator.listindices():
#                string = '%s[%d]= COUP*denom*' % (OffShellParticle, self.pass_to_HELAS(ind))
#                string += self.write_obj(numerator.get_rep(ind))
#                string = string.replace('+-', '-')
#                OutString = OutString + string + ';\n' 
#        OutString = re.sub('(?P<variable>[A-Za-z]+[0-9]\[*[0-9]*\]*)\*\*(?P<num>[0-9])','pow(\g<variable>,\g<num>)',OutString)
#        return OutString 
#
#    remove_double = re.compile('complex<double> (?P<name>[\w]+)\[\]')
#    def define_symmetry(self, new_nb):
#        """Write the call for symmetric routines"""
#        calls = self.calllist['CallList']
#        
#        for i, call in enumerate(calls):
#            if self.remove_double.match(call):
#                calls[i] = self.remove_double.match(call).group('name')
#                
#        # For the call, need to remove the type specification
#        #calls = [self.remove_double.match(call).group('name') for call in \
#        #         calls]
#        number = self.offshell 
#        Outstring = self.name+'('+','.join(calls)+',COUP,M%s,W%s,%s%s);\n' \
#                         %(number,number,self.particles[self.offshell-1],number)
#        return Outstring
#    
#    def define_foot(self):
#        """Return the end of the function definition"""
#
#        return '}\n\n' 
#
#    def write_h(self, header, compiler_cmd=True):
#        """Return the full contents of the .h file"""
#
#        h_string = ''
#        if compiler_cmd:
#            h_string = '#ifndef '+ self.name + '_guard\n'
#            h_string += '#define ' + self.name + '_guard\n'
#            h_string += '#include <complex>\n'
#            h_string += 'using namespace std;\n\n'
#
#        h_header = header['h_header']
#
#        h_string += h_header
#
#        for elem in self.symmetries: 
#            symmetryhead = h_header.replace( \
#                             self.name,self.name[0:-1]+'%s' %(elem))
#            h_string += symmetryhead
#
#        if compiler_cmd:
#            h_string += '#endif\n\n'
#
#        return h_string
#
#    def write_combined_h(self, lor_names, offshell=None, compiler_cmd=True):
#        """Return the content of the .h file linked to multiple lorentz call."""
#        
#        name = combine_name(self.abstractname, lor_names, offshell, self.tag)
#        text= ''
#        if compiler_cmd:
#            text = '#ifndef '+ name + '_guard\n'
#            text += '#define ' + name + '_guard\n'
#            text += '#include <complex>\n'
#            text += 'using namespace std;\n\n'
#        
#        # write header 
#        header = self.define_header(name=name)
#        h_header = header['h_header']
#        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
#        h_string = h_header.replace('COUP', ', complex <double>'.join(new_couplings))
#        text += h_string 
#        
#        for elem in self.symmetries: 
#            text += h_string.replace(name, name[0:-1]+str(elem))
#        
#        if compiler_cmd:
#            text += '#endif'
#        
#        return text
#
#    def write_cc(self, header, compiler_cmd=True):
#        """Return the full contents of the .cc file"""
#
#        cc_string = ''
#        if compiler_cmd:
#            cc_string = '#include \"%s.h\"\n\n' % self.name
#        cc_header = header['cc_header']
#        cc_string += cc_header
#        cc_string += self.define_momenta()
#        cc_string += self.define_expression()
#        cc_string += self.define_foot()
#
#        for elem in self.symmetries: 
#            symmetryhead = cc_header.replace( \
#                             self.name,self.name[0:-1]+'%s' %(elem))
#            symmetrybody = self.define_symmetry(elem)
#            cc_string += symmetryhead
#            cc_string += symmetrybody
#            cc_string += self.define_foot()
#
#        return cc_string
#
#    def write_combined_cc(self, lor_names, offshell=None, compiler_cmd=True, sym=True):
#        "Return the content of the .cc file linked to multiple lorentz call."
#        
#        # Set some usefull command
#        if offshell is None:
#            offshell = self.offshell
#            
#        name = combine_name(self.abstractname, lor_names, offshell, self.tag)
#
#        text = ''
#        if compiler_cmd:
#            text += '#include "%s.h"\n\n' % name
#           
#        # write header 
#        header = self.define_header(name=name)['cc_header']
#        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
#        text += header.replace('COUP', ', complex<double>'.join(new_couplings))
#        # define the TMP for storing output        
#        if not offshell:
#            text += 'complex<double> tmp;\n'
#        else:
#            spin = self.particles[offshell -1] 
#            text += 'complex<double> tmp[%s];\n int i = 0;' % self.type_to_size[spin]         
#
#        # Define which part of the routine should be called
#        addon = ''
#        if 'C' in self.name:
#            short_name, addon = name.split('C',1)
#            if addon.split('_')[0].isdigit():
#                addon = 'C' +self.name.split('C',1)[1]
#            elif all([n.isdigit() for n in addon.split('_')[0].split('C')]):
#                addon = 'C' +self.name.split('C',1)[1]
#            else:
#                addon = '_%s' % self.offshell
#        else:
#            addon = '_%s' % self.offshell
#
#        # how to call the routine
#        if not offshell:
#            main = 'vertex'
#            call_arg = '%(args)s, %(COUP)s, %(LAST)s' % \
#                    {'args': ', '.join(self.calllist['CallList']), 
#                     'COUP':'COUP%d',
#                     'spin': self.particles[self.offshell -1],
#                     'LAST': '%s'}
#        else:
#            main = '%(spin)s%(id)d' % \
#                          {'spin': self.particles[self.offshell -1],
#                           'id': self.outgoing}
#            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d, %(LAST)s' % \
#                    {'args': ', '.join(self.calllist['CallList']), 
#                     'COUP':'COUP%d',
#                     'id': self.outgoing,
#                     'LAST': '%s'}
#
#        # make the first call
#        line = "%s%s("+call_arg+");\n"
#        text += '\n\n' + line % (self.name, '', 1, main)
#        
#        # make the other call
#        for i,lor in enumerate(lor_names):
#            text += line % (lor, addon, i+2, 'tmp')
#            if not offshell:
#                text += ' vertex = vertex + tmp;\n'
#            else:
#                size = self.type_to_size[spin] -2
#                text += """ while (i < %(id)d)
#                {
#                %(main)s[i] = %(main)s[i] + tmp[i];
#                i++;
#                }\n""" %  {'id': size, 'main':main}
#                
#
#        text += self.define_foot()
#        
#        if sym:
#            for elem in self.symmetries:
#                text += self.write_combined_cc(lor_names, elem, 
#                                                      compiler_cmd=compiler_cmd,
#                                                      sym=False)
#            
#            
#        if self.out_path:
#            # Prepare a specific file
#            path = os.path.join(os.path.dirname(self.out_path), name+'.f')
#            writer = writers.FortranWriter(path)
#            writer.downcase = False 
#            commentstring = 'This File is Automatically generated by ALOHA \n'
#            writer.write_comments(commentstring)
#            writer.writelines(text)
#        
#        return text
#
#
#    
#    def write(self, mode='self', **opt):
#        """Write the .h and .cc files"""
#
#
#        # write head - momenta - body - foot
#        header = self.define_header()
#        h_text = self.write_h(header, **opt)
#        cc_text = self.write_cc(header, **opt)
#        
#        # write in two file
#        if self.out_path:
#            writer_h = writers.CPPWriter(self.out_path + ".h")
#            writer_cc = writers.CPPWriter(self.out_path + ".cc")
#            commentstring = 'This File is Automatically generated by ALOHA \n'
#            commentstring += 'The process calculated in this file is: \n'
#            commentstring += self.comment + '\n'
#            writer_h.write_comments(commentstring)
#            writer_cc.write_comments(commentstring)
#            writer_h.writelines(h_text)
#            writer_cc.writelines(cc_text)
#            
#        return h_text, cc_text
# 
# 
# 
#    def write_combined(self, lor_names, mode='self', offshell=None, **opt):
#        """Write the .h and .cc files associated to the combined file"""
#        
#        # Set some usefull command
#        if offshell is None:
#            offshell = self.offshell
#        
#        name = combine_name(self.abstractname, lor_names, offshell, self.tag)
#        
#        h_text = self.write_combined_h(lor_names, offshell, **opt)
#        cc_text = self.write_combined_cc(lor_names, offshell, sym=True, **opt)
#        
#        if self.out_path:
#            # Prepare a specific file
#            path = os.path.join(os.path.dirname(self.out_path), name)
#            commentstring = 'This File is Automatically generated by ALOHA \n'
#            
#            writer_h = writers.CPPWriter(path + ".h")
#            writer_h.write_comments(commentstring)
#            writer_h.writelines(h_text)
#            
#            writer_cc = writers.CPPWriter(path + ".cc")
#            writer_cc.write_comments(commentstring)
#            writer_cc.writelines(cc_text)
#        else:
#            return h_text, cc_text
#        
#        return h_text, cc_text
#        
class ALOHAWriterForPython(WriteALOHA):
    """ A class for returning a file/a string for python evaluation """
    
    extension = '.py'
    writer = writers.PythonWriter
    
    @staticmethod
    def change_number_format(obj):
        if obj.real == 0 and obj.imag:
            if int(obj.imag) == obj.imag: 
                return '%ij' % obj.imag
            else:
                return '%sj' % str(obj.imag)
        else: 
            return str(obj)
    
    @staticmethod
    def shift_indices(match):
        """shift the indices for non impulsion object"""
        if match.group('var').startswith('P'):
            shift = 0
        else: 
            shift = -1
            
        return '%s[%s]' % (match.group('var'), int(match.group('num')) + shift)

    def change_var_format(self, name): 
        """Formatting the variable name to Python format
        start to count at zero. 
        No neeed to define the variable in python -> no need to keep track of 
        the various variable
        """
        
        if '_' not in name:
            self.declaration.add(('', name))
        else:
            self.declaration.add(('', name.split('_',1)[0]))
        name = re.sub('(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
        
        return name
    
    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            for name,obj in self.routine.contracted.items():
                out.write('    %s = %s\n' % (name, self.write_obj(obj)))

        numerator = self.routine.expr
        
        if not self.offshell:
            out.write('    vertex = COUP*%s\n' % self.write_obj(numerator.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)

            if not 'L' in self.tag:
                coeff = 'denom'
                out.write('    denom = COUP/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s * (M%(i)s -1j* W%(i)s))\n' % {'i': self.outgoing})
            else:
                coeff = 'COUP'
                
            for ind in numerator.listindices():
                out.write('    %s[%d]= %s*%s\n' % (self.outname, 
                                        self.pass_to_HELAS(ind), coeff, 
                                        self.write_obj(numerator.get_rep(ind))))
        return out.getvalue()
    
    def get_foot_txt(self):
        if not self.offshell:
            return '    return vertex\n\n'
        else:
            return '    return %s\n\n' % (self.outname)
            
    
    def get_header_txt(self, name=None, couplings=['COUP']):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        out = StringIO()
        
        if self.mode == 'mg5':
            out.write('import aloha.template_files.wavefunctions as wavefunctions\n')
        else:
            out.write('import wavefunctions\n')
        
        
        # define the type of function and argument
        
        arguments = [arg for format, arg in self.define_argument_list(couplings)]       
        out.write('def %(name)s(%(args)s):\n' % \
                                    {'name': name, 'args': ','.join(arguments)})
          
        return out.getvalue()     

    def get_momenta_txt(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
                    
        out = StringIO()
        
        # Define all the required momenta
        p1,p2 = [], [] # a list for keeping track how to write the momentum
        
        signs = self.get_momentum_conservation_sign()
        
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
                out.write("    OM{0} = 0.0\n    if (M{0}): OM{0}=1.0/M{0}**2\n".format( (i+1) ))
            
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                energy_pos = self.type_to_size[type] -2
                p1.append('%s%s%s[%s]' % (signs[i],type,i+1, energy_pos))
                p2.append('%s%s%s[%s]' % (signs[i],type,i+1, energy_pos+1))      
            
            if self.declaration.is_used('P%s' % (i+1)):
                energy_pos = self.type_to_size[type] -2
                out.write('''    P%(i)d = [%(sign)scomplex(%(type)s%(i)d[%(nb)d]).real, %(sign)scomplex(%(type)s%(i)d[%(nb2)d]).real, %(sign)scomplex(%(type)s%(i)d[%(nb2)d]).imag, %(sign)scomplex(%(type)s%(i)d[%(nb)d]).imag]\n''' % \
                {'type': type, 'i': i+1, 'nb': energy_pos, 'nb2': energy_pos + 1,
                 'sign': self.get_P_sign(i+1)})               
                
        # define the resulting momenta
        if self.offshell:
            energy_pos = out_size -2
            type = self.particles[self.outgoing-1]
            out.write('    %s = wavefunctions.WaveFunction(size=%s)\n' % \
                                                       (self.outname, out_size))
            
            out.write('    %s%s[%s] = %s\n' % (type,self.outgoing, energy_pos, ''.join(p1)))
            out.write('    %s%s[%s] = %s\n' % (type,self.outgoing, energy_pos+1, ''.join(p2)))
            
            out.write('''    P%(i)d = [%(sign)scomplex(%(type)s%(i)d[%(nb)d]).real, %(sign)scomplex(%(type)s%(i)d[%(nb2)d]).real, %(sign)scomplex(%(type)s%(i)d[%(nb2)d]).imag, %(sign)scomplex(%(type)s%(i)d[%(nb)d]).imag]\n''' % \
                {'type': out_type, 'i': self.outgoing, 'nb': energy_pos, 
                 'nb2': energy_pos + 1, 'sign': self.get_P_sign(self.offshell)}) 
        
        # Returning result
        return out.getvalue()

    def define_symmetry(self, new_nb, couplings=['COUP']):
        number = self.offshell
        arguments = [name for format, name in self.define_argument_list()]
        new_name = self.name.rsplit('_')[0] + '_%s' % new_nb
        return '%s\n    return %s(%s)' % \
            (self.get_header_txt(new_name, couplings), self.name, ','.join(arguments))

    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        # write head - momenta - body - foot
        text = StringIO()
        data = {} # for the formating of the line
                    
        # write header 
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        text.write(self.get_header_txt(name=name, couplings=new_couplings))
  
        # Define which part of the routine should be called
        data['addon'] = ''.join(self.tag) + '_%s' % self.offshell

        # how to call the routine
        argument = [name for format, name in self.define_argument_list(new_couplings)]
        index= argument.index('COUP1')
        data['before_coup'] = ','.join(argument[:index])
        data['after_coup'] = ','.join(argument[index+len(lor_names)+1:])
        if data['after_coup']:
            data['after_coup'] = ',' + data['after_coup']
            
        lor_list = (self.routine.name,) + lor_names
        line = "    %(out)s = %(name)s%(addon)s(%(before_coup)s,%(coup)s%(after_coup)s)\n"
        main = '%(spin)s%(id)d' % {'spin': self.particles[self.offshell -1],
                           'id': self.outgoing}
        for i, name in enumerate(lor_list):
            data['name'] = name
            data['coup'] = 'COUP%d' % (i+1)
            if i == 0:
                if  not offshell: 
                    data['out'] = 'vertex'
                else:
                    data['out'] = main
            elif i==1:
                data['out'] = 'tmp'
            text.write(line % data)
            if i:
                if not offshell:
                    text.write( '    vertex += tmp\n')
                else:
                    size = self.type_to_size[self.particles[offshell -1]] -2
                    text.write("    for i in range(%(id)d):\n" % {'id': size})
                    text.write("        %(main)s[i] += tmp[i]\n" %{'main': main})
        
        text.write(self.get_foot_txt())

        #ADD SYMETRY
        if sym:
            for elem in self.routine.symmetries:
                text.write(self.write_combined(lor_names, mode, elem))

        text = text.getvalue()
        if self.out_path:        
            writer = self.writer(self.out_path)
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer.write_comments(commentstring)
            writer.write(text)


        return text


class Declaration_list(set):

    def is_used(self, var):
        if hasattr(self, 'var_name'):
            return var in self.var_name
        self.var_name = [name for type,name in self]
        return var in self.var_name
            

class WriterFactory(object):
    
    def __new__(cls, data, language, outputdir, tags):
        
        language = language.lower()
        if isinstance(data.expr, aloha_lib.SplitCoefficient):
            assert language == 'fortran'
            if 'MP' in tags:
                return ALOHAWriterForFortranLoopQP(data, outputdir)
            else:
                return ALOHAWriterForFortranLoop(data, outputdir)
        
        if language == 'fortran':
            if 'MP' in tags:
                return ALOHAWriterForFortranQP(data, outputdir)
            else:
                return ALOHAWriterForFortran(data, outputdir)
        elif language == 'python':
            return ALOHAWriterForPython(data, outputdir)
        elif language == 'cpp':
            raise Exception, 'CPP output not yet implemented'
            return ALOHAWriterForCPP(data, outputdir)
        else:
            raise Exception, 'Unknown output format'






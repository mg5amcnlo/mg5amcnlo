from __future__ import absolute_import
from six.moves import range
try:
    import madgraph.iolibs.file_writers as writers 
    import madgraph.various.q_polynomial as q_polynomial
    import madgraph.various.misc as misc
except Exception:
    import aloha.file_writers as writers
    import aloha.q_polynomial as q_polynomial
    import aloha.misc as misc

import aloha
import aloha.aloha_lib as aloha_lib
import cmath
import os
import re 
from numbers import Number
from collections import defaultdict
from fractions import Fraction
# fast way to deal with string
from six import StringIO
# Look at http://www.skymind.com/~ocrow/python_string/ 
# For knowing how to deal with long strings efficiently.
import itertools

KERNEL = aloha_lib.KERNEL
pjoin = os.path.join

class WriteALOHA: 
    """ Generic writing functions """ 
    
    power_symbol = '**'
    change_number_format = str
    extension = ''
    type_to_variable = {2:'F',3:'V',5:'T',1:'S',4:'R', -1:'S'}
    type_to_size = {'S':3, 'T':18, 'V':6, 'F':6,'R':18}


            
    def __init__(self, abstract_routine, dirpath):
        if aloha.loop_mode:
            self.momentum_size = 4
        else:
            self.momentum_size = 2

        if aloha.unitary_gauge == 3: #FD gauge
            # need to check for non goldstone scalar
            self.type_to_size = {'S':7, 'T':18, 'V':7, 'F':6,'R':18}
        else:
            self.type_to_size = {'S':3, 'T':18, 'V':6, 'F':6,'R':18}

        self.has_model_parameter = False
        
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
        self.outname = '%s%s' % (self.particles[self.outgoing -1], \
                                                               self.outgoing)
        #initialize global helper routine
        self.declaration = Declaration_list()
                                   
                                       
    def pass_to_HELAS(self, indices, start=0):
        """find the Fortran HELAS position for the list of index""" 
        
        
        if len(indices) == 1:
            return indices[0] + start + self.momentum_size

        try:
            # When the expr is not a SplitCoefficient
            ind_name = self.routine.expr.lorentz_ind
        except:
            # When the expr is a loop one, i.e. with SplitCoefficient
            if len(set([tuple(expr.lorentz_ind) for expr in self.routine.expr.values()]))!=1:
                raise Exception('All SplitCoefficients do not share the same indices names.')
            for expr in self.routine.expr.values():
              ind_name = expr.lorentz_ind
              break

        if ind_name == ['I3', 'I2']:
            return  4 * indices[1] + indices[0] + start + self.momentum_size
        elif len(indices) == 2: 
            return  4 * indices[0] + indices[1] + start + self.momentum_size
        else:
            raise Exception('WRONG CONTRACTION OF LORENTZ OBJECT for routine %s: %s' \
                    % (self.name, ind_name))                                 
                                 
    def get_header_txt(self,mode=''): 
        """ Prototype for language specific header""" 
        raise Exception('THis function should be overwritten')
        return ''
    
    def get_declaration_txt(self):
        """ Prototype for how to write the declaration of variable"""
        return ''

    def define_content(self): 
        """Prototype for language specific body""" 
        pass

    def get_momenta_txt(self):
        """ Prototype for the definition of the momenta"""
        raise Exception('THis function should be overwritten')

    def get_momentum_conservation_sign(self):
        """find the sign associated to the momentum conservation"""

        # help data 
        signs = []
        nb_fermion =0
        
        #compute global sign

        global_sign = -1
        
        flipped = [2*(int(c[1:])-1) for c in self.tag if c.startswith('C')]
        for index, spin in enumerate(self.particles):
            assert(spin in ['S','F','V','T', 'R'])  
                  
            #compute the sign
            if 1:#spin != 'F':
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
        if self.outgoing == index:
            sign = -1
        #if 'C%s' % ((index +1) // 2)  in self.tag: 
        #    if index == self.outgoing:
        #        pass
        #       elif index % 2 and index -1 != self.outgoing:
#                pass
#            elif index % 2 == 1 and index + 1  != self.outgoing:
#                pass
#            else:
#                sign *= -1
        
        if sign == -1 :
            return '-'
        else:
            return ''
        
        
        
        
    
    def get_foot_txt(self):
        """Prototype for language specific footer"""
        return ''
    
    def define_argument_list(self, couplings=None):
        """define a list with the string of object required as incoming argument"""

        call_arg = [] #incoming argument of the routine

        conjugate = [2*(int(c[1:])-1) for c in self.tag if c[0] == 'C']
        

        for index,spin in enumerate(self.particles):
            if self.offshell == index + 1:
                continue
            
            if index in conjugate:
                index2, spin2 = index+1, self.particles[index+1]
                call_arg.append(('list_complex','%s%d' % (spin2, index2 +1))) 
                #call_arg.append('%s%d' % (spin, index +1)) 
            elif index-1 in conjugate:
                index2, spin2 = index-1, self.particles[index-1]
                call_arg.append(('list_complex','%s%d' % (spin2, index2 +1))) 
            else:
                call_arg.append(('list_complex','%s%d' % (spin, index +1)))
        
        # couplings
        if  couplings is None:
            detected_couplings = [name for type, name in self.declaration if name.startswith('COUP')]
            detected_couplings.sort(key=lambda x: int(x[4:]) if x[4:] else 0)
            if detected_couplings:
                couplings = detected_couplings
            else:
                couplings = ['COUP']
                
        for coup in couplings:   
            call_arg.append(('complex', coup))              
            self.declaration.add(('complex',coup))
            
        if self.offshell:
            if 'P1N' in self.tag:
                pass
            elif aloha.complex_mass:
                call_arg.append(('complex','M%s' % self.outgoing))              
                self.declaration.add(('complex','M%s' % self.outgoing))
            else:
                call_arg.append(('double','M%s' % self.outgoing))              
                self.declaration.add(('double','M%s' % self.outgoing))                
                call_arg.append(('double','W%s' % self.outgoing))              
                self.declaration.add(('double','W%s' % self.outgoing))
        
        assert len(call_arg) == len(set([a[1] for a in call_arg]))
        assert len(self.declaration) == len(set([a[1] for a in self.declaration])), self.declaration
        self.call_arg = call_arg
        return call_arg

    def write(self, mode=None):
                         
        self.mode = mode
        
        core_text = self.define_expression()    
        self.define_argument_list()
        out = StringIO()
        out.write(self.get_header_txt(mode=self.mode))
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
            writer.writelines(text)
            
        return text + '\n'

    
    def write_indices_part(self, indices, obj): 
        """Routine for making a string out of indices objects"""
        
        text = 'output(%s)' % indices
        return text                 
        
    def write_obj(self, obj, prefactor=True):
        """Calls the appropriate writing routine"""
        
        try:
            vartype = obj.vartype
        except Exception:
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
            formatted = self.change_number_format(obj.prefactor)
            if formatted.startswith(('+','-')):
                file_str.write('(%s)' % formatted)
            else:
                file_str.write(formatted)
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
        if number:
            total = sum(number)
            file_str.write('+ %s' % self.change_number_format(total))

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
    
    def get_fct_format(self, fct):
        """Put the function in the correct format"""
        if not hasattr(self, 'fct_format'):
            one = self.change_number_format(1)
            self.fct_format = {'csc' : '{0}/cos(dble(%s))'.format(one),
                   'sec': '{0}/sin(dble(%s))'.format(one),
                   'acsc': 'asin({0}/(dble(%s)))'.format(one),
                   'asec': 'acos({0}/(%s))'.format(one),
                   're': ' dble(%s)',
                   'im': 'imag(%s)',
                   'cmath.sqrt':'sqrt(dble(%s))', 
                   'sqrt': 'sqrt(dble(%s))',
                   'complexconjugate': 'conjg(dcmplx(%s))',
                   '/' : '{0}/(%s)'.format(one),
                   'pow': '(%s)**(%s)',
                   'log': 'log(dble(%s))',
                   'asin': 'asin(dble(%s))',
                   'acos': 'acos(dble(%s))',
                   'abs': 'std::abs(%s)',
                   'fabs': 'std::abs(%s)',
                   'math.abs': 'std::abs(%s)',
                   'cmath.abs': 'std::abs(%s)',
                   '':'(%s)'
                   }
            
        if fct in self.fct_format:
            return self.fct_format[fct]
        else:
            self.declaration.add(('fct', fct))
            return '{0}(%s)'.format(fct)
            

    
    def get_header_txt(self, name=None, couplings=None, **opt):
        """Define the Header of the fortran file. 
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
                     'spin': self.particles[self.outgoing -1],
                     'id': self.outgoing}
            self.declaration.add(('list_complex', output))
        
        out.write('subroutine %(name)s(%(args)s,%(output)s)\n' % \
                  {'output':output, 'name': name, 'args': ', '.join(arguments)})
        
        return out.getvalue() 
    
    def get_declaration_txt(self):
        """ Prototype for how to write the declaration of variable
            Include the symmetry line (entry FFV_2)
        """
        
        out = StringIO()
        out.write('implicit none\n')
        # Check if we are in formfactor mode
        if self.has_model_parameter:
            out.write(' include "../vector.inc"\n') 
            out.write(' include "../MODEL/input.inc"\n')
            out.write(' include "../MODEL/coupl.inc"\n')
        argument_var = [name for type,name in self.call_arg]
        # define the complex number CI = 0+1j
        if 'MP' in self.tag:
            out.write(' complex*32 CI\n')
            if KERNEL.has_pi:
                out.write(' REAL ( KIND = 16 ) PI\n')
        else:
            out.write(' complex*16 CI\n')
            if KERNEL.has_pi:
                out.write(' double precision PI\n')
        out.write(' parameter (CI=(%s,%s))\n' % 
                    (self.change_number_format(0),self.change_number_format(1)))
        if KERNEL.has_pi:
            out.write(' parameter (PI=%s)\n' % self.change_number_format(cmath.pi))
        
        if aloha.unitary_gauge == 3: # FG gauge 
            self.declaration.add(('int','i'))
            out.write(" COMPLEX*16 CZERO\n")
            out.write("PARAMETER (CZERO=(0D0,0D0)) \n")

        for type, name in self.declaration.tolist():
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
                    elif aloha.unitary_gauge == 3 and name[0] in 'V':
                        size = 7
                    else:
                        size = 6
                elif name[0] == 'S':
                    if aloha.loop_mode:
                        size = 5
                    elif aloha.unitary_gauge == 3: # FD gauge 
                        # Need to fix since this need to be dependent if S is a goldstone or not
                        size = 7
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
            elif type == 'fct':
                if name.upper() in ['EXP','LOG','SIN','COS','ASIN','ACOS']:
                    continue
                out.write(' %s %s\n' % (self.type2def['complex'], name))
                out.write(' external %s\n' % (name))
            else:
                out.write(' %s %s\n' % (self.type2def[type], name))
                
        # Add the lines corresponding to the symmetry
        
        #number = self.offshell
        #arguments = [name for format, name in self.define_argument_list()]
        #new_name = self.name.rsplit('_')[0] + '_%s' % new_nb
        #return '%s\n    call %s(%s)' % \
        #    (self.get_header_txt(new_name, couplings), self.name, ','.join(arguments))
        couplings = [name for type, name in self.declaration if name.startswith('COUP') ]
        couplings.sort(key=lambda x: int(x[4:]) if x[4:] else 0)
        for elem in self.routine.symmetries:
            new_name = self.name.rsplit('_',1)[0] + '_%s' % elem
            out.write('%s\n' % self.get_header_txt(new_name, couplings).replace('subroutine','entry'))
                    

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
                p.append('{0}{1}{2}(%(i)s)'.format(signs[i],type,i+1,type))    
                
            if self.declaration.is_used('P%s' % (i+1)):
                self.get_one_momenta_def(i+1, out)
        
        # define the resulting momenta
        bypass = False
        if 'P1N' in self.tag:
            if  not self.declaration.is_used('P%s' % (self.outgoing)):
                bypass = True

        if self.offshell and not bypass:

            energy_pos = out_size -2
            type = self.particles[self.outgoing-1]
            
            for i in range(self.momentum_size):
                dict_energy = {'i':1+i}
                out.write('    %s%s(%s) = %s\n' % (type,self.outgoing, 1+i, 
                                             ''.join(p) % dict_energy))
            if self.declaration.is_used('P%s' % self.outgoing):
                self.get_one_momenta_def(self.outgoing, out)

            if "P1T" in self.tag or "P1L" in self.tag:
                for i in range(1,4):
                    P = "P%s" % (self.outgoing)
                    value = ["1d-30", "0d0", "1d-15"]
                    out.write("  IF (ABS(%(P)s(0))*1e-10.gt.ABS(%(P)s(%(i)s))) %(P)s(%(i)s)=%(val)s\n"
                              % {"P": P, "i":i, 'val':value[i-1]})
            i = self.outgoing -1
            if self.declaration.is_used('Tnorm%s' % (i+1)):
                out.write("    TNORM{0} = DSQRT(P{0}(1)*P{0}(1)+P{0}(2)*P{0}(2)+P{0}(3)*P{0}(3))\n".format(
                        i+1))
            if self.declaration.is_used('TnormZ%s' % (i+1)):
                out.write("    TNORMZ{0} =  TNORM{0} - P{0}(3)\n".format(
                        i+1))

            if self.declaration.is_used('FWP%s' % (i+1)):
                out.write("     FWP{0} = DSQRT(-P{0}(0) + TNORM{0})\n"\
                          .format(i+1))
            if self.declaration.is_used('FWM%s' % (i+1)):
                out.write("     FWM{0} = DSQRT(-P{0}(0) - TNORM{0})\n"\
                          .format(i+1))
                #out.write("     FWM{0} = M{0}/FWP{0}\n".format(i+1))
        
        if self.offshell and aloha.unitary_gauge == 3: # FD gauge
            type = self.particles[self.outgoing-1]
            if type in ["S","V"]: 
                out.write("      DO I = 3, 7\n")
                out.write(" %(type)s%(out)s(I) = CZERO \n" % {'type': type, 'out':self.outgoing}) 
                out.write(" ENDDO\n")      

        # Returning result
        return out.getvalue()

    def get_one_momenta_def(self, i, strfile):
        
        type = self.particles[i-1]
        
        if aloha.loop_mode:
            template ='P%(i)d(%(j)d) = %(sign)s%(type)s%(i)d(%(nb)d)\n'
        else:
            template ='P%(i)d(%(j)d) = %(sign)s%(operator)s(%(type)s%(i)d(%(nb2)d))\n'

        nb2 = 1
        for j in range(4):
            if not aloha.loop_mode:
                nb = j + 1
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
                nb = 1+ j
                nb2 = 1 + j
            strfile.write(template % {'j':j,'type': type, 'i': i, 
                        'nb': nb, 'nb2': nb2, 'operator':operator,
                        'sign': self.get_P_sign(i)})  
    
    def shift_indices(self, match):
        """shift the indices for non impulsion object"""
        if match.group('var').startswith('P'):
            shift = 0
        else:
            shift =  self.momentum_size 
            if aloha.unitary_gauge ==3 and match.group('var').startswith('S'):
                shift += 4
        return '%s(%s)' % (match.group('var'), int(match.group('num')) + shift)
              
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
        if isinstance(name, aloha_lib.ExtVariable):
            # external parameter nothing to do but handling model prefix
            self.has_model_parameter = True
            if name.lower() in ['pi', 'as', 'mu_r', 'aewm1','g']:
                return name
            if name.startswith(aloha.aloha_prefix):
                return name
            return '%s%s' % (aloha.aloha_prefix, name)
        
        if '_' in name:
            vtype = name.type
            decla = name.split('_',1)[0]
            self.declaration.add(('list_%s' % vtype, decla))
        else:
            self.declaration.add((name.type, name))
        name = re.sub(r'(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
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
                if number.real:
                    out = '(%s + %s*CI)' % (self.change_number_format(number.real), \
                                    self.change_number_format(number.imag))
                else:
                    if number.imag == 1:
                        out = 'CI'
                    elif number.imag == -1:
                        out = '-CI'
                    else: 
                        out = '%s * CI' % self.change_number_format(number.imag)
            else:
                out = '%s' % (self.change_number_format(number.real))
        else:
            tmp = Fraction(str(number))
            tmp = tmp.limit_denominator(100)
            if not abs(tmp - number) / abs(tmp + number) < 1e-8:
                if 'e' in str(number):
                    out = str(number).replace('e','d')
                else:
                    out = '%s%s' % (number, self.format)
            else:
                out = '%s%s/%s%s' % (tmp.numerator, self.format, tmp.denominator, self.format)
        return out
    
    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            all_keys = list(self.routine.contracted.keys())
            all_keys.sort()
            for name in all_keys:
                obj = self.routine.contracted[name]
                out.write(' %s = %s\n' % (name, self.write_obj(obj)))
                self.declaration.add(('complex', name))
                
        
        def sort_fct(a, b):
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            elif a < b:
                return -1
            else:
                return +1
            
        keys = list(self.routine.fct.keys())        
        keys.sort(key=misc.cmp_to_key(sort_fct))
        for name in keys:
            fct, objs = self.routine.fct[name]
            format = ' %s = %s\n' % (name, self.get_fct_format(fct))
            try:
                text = format % ','.join([self.write_obj(obj) for obj in objs])
            except TypeError:
                text = format % tuple([self.write_obj(obj) for obj in objs])
            finally:
                out.write(text)
        

        numerator = self.routine.expr

        if not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)

        if not self.offshell:
            if coup_name == 'COUP':
                formatted = self.write_obj(numerator.get_rep([0]))
                if formatted.startswith(('+','-')):
                    out.write(' vertex = COUP*(%s)\n' % formatted)
                else:
                    out.write(' vertex = COUP*%s\n' % formatted)
            else:
                out.write(' vertex = %s\n' % self.write_obj(numerator.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            is_loop = False
            if 'L' in self.tag:
                if self.tag.count('L') == 1 and 'PL' in self.tag:
                    is_loop = False
                else:
                    is_loop = True
                    
            if not is_loop:
                coeff = 'denom*'    
                if not aloha.complex_mass:
                    if self.routine.denominator:
                        if 'P1N' not in self.tag:
                            out.write('    denom = %(COUP)s/(%(denom)s)\n' % {'COUP': coup_name,\
                                'denom':self.write_obj(self.routine.denominator)}) 
                    else:
                        out.write('    denom = %(COUP)s/(P%(i)s(0)**2-P%(i)s(1)**2-P%(i)s(2)**2-P%(i)s(3)**2 - M%(i)s * (M%(i)s -CI* W%(i)s))\n' % \
                                  {'i': self.outgoing, 'COUP': coup_name})
                else:
                    if self.routine.denominator:
                        if 'P1N' not in self.tag:
                            raise Exception('modify denominator are not compatible with complex mass scheme', self.tag)                
                    if 'P1N' not in self.tag:
                        out.write('    denom = %(COUP)s/(P%(i)s(0)**2-P%(i)s(1)**2-P%(i)s(2)**2-P%(i)s(3)**2 - M%(i)s**2)\n' % \
                      {'i': self.outgoing, 'COUP': coup_name})
                if 'P1N' not in self.tag:
                    self.declaration.add(('complex','denom'))
                if aloha.loop_mode:
                    ptype = 'list_complex'
                else:
                    ptype = 'list_double'
                if 'P1N' not in self.tag:
                    self.declaration.add((ptype,'P%s' % self.outgoing))
                else:
                    coeff = '%(COUP)s*' % {'COUP': coup_name}  
            else:
                if coup_name == 'COUP':
                    coeff = 'COUP*'
                else:
                    coeff = ''
            to_order = {}  
            for ind in numerator.listindices():
                formatted = self.write_obj(numerator.get_rep(ind))
                if formatted.startswith(('+','-')):
                    if '*' in formatted:
                        formatted = '(%s)*%s' % tuple(formatted.split('*',1))
                    else:
                        if formatted.startswith('+'):
                            formatted = formatted[1:]
                        else:
                            formatted = '(-1)*%s' % formatted[1:]
                shift = 1
                if aloha.unitary_gauge == 3 and self.outname[0] == "S":
                    shift = 5
                to_order[self.pass_to_HELAS(ind)] = \
                        '    %s(%d)= %s%s\n' % (self.outname, self.pass_to_HELAS(ind)+shift, 
                        coeff, formatted)
            key = list(to_order.keys())
            key.sort()
            for i in key:
                out.write(to_order[i])

        txt = out.getvalue() 
        # in rare case FCT/TMP might not be needed (multiply by zero)
        # This is detected here and in such case we remove those FCT/TMP
        # from the routine block are recall this routine
        found=False
        # detection for FCT
        keys = list(self.routine.fct.keys())        
        keys.sort(key=misc.cmp_to_key(sort_fct))
        for name in keys:
            if txt.count(name) == 1:
                del self.routine.fct[name]
                found = True
        #detection for TMP variable
        all_keys = list(self.routine.contracted.keys())
        all_keys.sort()
        for name in all_keys:
            if txt.count(name) == 1:
                del self.routine.contracted[name]
                self.declaration.discard(('complex', name))
                found = True
        if found:
            # retry when removing the useless part.
            return self.define_expression()

        return txt

    def define_symmetry(self, new_nb, couplings=None):
        return ''
        #number = self.offshell
        #arguments = [name for format, name in self.define_argument_list()]
        #new_name = self.name.rsplit('_')[0] + '_%s' % new_nb
        #return '%s\n    call %s(%s)' % \
        #    (self.get_header_txt(new_name, couplings), self.name, ','.join(arguments))

    def get_foot_txt(self, combine=False):
        text = ' ' 
    
        if not combine and aloha.unitary_gauge == 3: # FD gauge
            if self.outgoing and 'P1N' not in self.tag:
                name = self.particles[self.outgoing-1]
                if name.startswith(('V','S')):
                    # need to be smarter for Higgs
                    text += 'CALL MULTIPLY_PROPAGATOR_FACTOR(%(name)s%(i)s,%(mass)s%(i)s, %(name)s%(i)s)\n' %\
                    {'name':name, 'mass': 'M%s' % name[1:], 'i': self.outgoing }


        text += 'end\n\n' 
        return text

    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        self.name = name
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
        main = '%(spin)s%(id)d' % {'spin': self.particles[self.outgoing -1],
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
                    type = self.particles[self.outgoing-1]
                    self.declaration.add(('list_complex','%stmp' % type))
                else:
                    type = ''
                    self.declaration.add(('complex','%stmp' % type))
                data['out'] = '%stmp' % type
            routine.write(line % data)
            if i:
                if not offshell:
                    routine.write( '    vertex = vertex + tmp\n')
                else:
                    size = self.type_to_size[self.particles[self.outgoing -1]] -2
                    routine.write(" do i = %s, %s\n" % (self.momentum_size+1, self.momentum_size+size))
                    routine.write("        %(main)s(i) = %(main)s(i) + %(tmp)s(i)\n" %\
                               {'main': main, 'tmp': data['out']})
                    routine.write(' enddo\n')
                    self.declaration.add(('int','i'))

        self.declaration.discard(('complex','COUP'))
        for name in aloha_lib.KERNEL.reduced_expr2:
            self.declaration.discard(('complex', name))
        
        #clean pointless declaration
        #self.declaration.discard
        
        
        text.write(self.get_declaration_txt())
        text.write(routine.getvalue())
        text.write(self.get_foot_txt(combine=True))


        text = text.getvalue()
        if self.out_path:        
            writer = self.writer(self.out_path,'a')
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer.write_comments(commentstring)
            writer.writelines(text)
        return text


class QP(object): 
    """routines for writing out Fortran"""
    
    type2def = {}    
    type2def['int'] = 'integer*4'
    type2def['double'] = 'real*16'
    type2def['complex'] = 'complex*32'
    format = 'q0'
    
class ALOHAWriterForFortranQP(QP, ALOHAWriterForFortran):
    
    def __init__(self, *arg):
        return ALOHAWriterForFortran.__init__(self, *arg)
    
class ALOHAWriterForFortranLoop(ALOHAWriterForFortran): 
    """routines for writing out Fortran"""

    def __init__(self, abstract_routine, dirpath):
        
        ALOHAWriterForFortran.__init__(self, abstract_routine, dirpath)
        # position of the outgoing in particle list        
        self.l_id = [int(c[1:]) for c in abstract_routine.tag if c[0] == 'L'][0] 
        self.l_helas_id = self.l_id   # expected position for the argument list
        if 'C%s' %((self.l_id + 1) // 2) in abstract_routine.tag:
            #flip the outgoing tag if in conjugate
            self.l_helas_id += self.l_id % 2 - (self.l_id +1) % 2
         

    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            for name,obj in self.routine.contracted.items():
                out.write(' %s = %s\n' % (name, self.write_obj(obj)))
                self.declaration.add(('complex', name))

        if not 'Coup(1)' in self.routine.infostr:
            coup = True
        else:
            coup = False

        rank = self.routine.expr.get_max_rank()
        poly_object = q_polynomial.Polynomial(rank)
        nb_coeff = q_polynomial.get_number_of_coefs_for_rank(rank)
        size = self.type_to_size[self.particles[self.l_id-1]] - 2
        for K in range(size):
            for J in range(nb_coeff):
                data = poly_object.get_coef_at_position(J)
                arg = [data.count(i) for i in range(4)] # momentum
                arg += [0] * (K) + [1] + [0] * (size-1-K) 
                try:
                    expr = self.routine.expr[tuple(arg)]
                except KeyError:
                    expr = None
                for ind in list(self.routine.expr.values())[0].listindices():
                    if expr:
                        data = expr.get_rep(ind)
                    else:
                        data = 0
                    if data and coup:
                        out.write('    COEFF(%s,%s,%s)= coup*%s\n' % ( 
                                    self.pass_to_HELAS(ind)+1-self.momentum_size,
                                    J, K+1, self.write_obj(data)))
                    else:
                        out.write('    COEFF(%s,%s,%s)= %s\n' % ( 
                                    self.pass_to_HELAS(ind)+1-self.momentum_size,
                                    J, K+1, self.write_obj(data)))


        return out.getvalue()
    
    def get_declaration_txt(self):
        """ Prototype for how to write the declaration of variable"""
        
        out = StringIO()
        out.write('implicit none\n')
        # define the complex number CI = 0+1j
        if 'MP' in self.tag:
            out.write(' complex*32 CI\n')
        else:
            out.write(' complex*16 CI\n')
        out.write(' parameter (CI=(%s,%s))\n' % 
                    (self.change_number_format(0),self.change_number_format(1)))
        argument_var = [name for type,name in self.call_arg]
        for type, name in self.declaration:
            if type.startswith('list'):
                type = type[5:]
                #determine the size of the list
                if name.startswith('P'):
                    size='0:3'
                elif name in argument_var:
                    size ='*'
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
                elif name == 'coeff':
                    out.write("include 'coef_specs.inc'\n")
                    size = 'MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE'
    
                out.write(' %s %s(%s)\n' % (self.type2def[type], name, size))
            elif type == 'fct':
                if name.upper() in ['EXP','LOG','SIN','COS','ASIN','ACOS']:
                    continue
                out.write(' %s %s\n' % (self.type2def['complex'], name))
                out.write(' external %s\n' % (name))
            else:
                out.write(' %s %s\n' % (self.type2def[type], name))

        return out.getvalue()
    
    
    def define_argument_list(self, couplings=None):
        """define a list with the string of object required as incoming argument"""

        conjugate = [2*(int(c[1:])-1) for c in self.tag if c[0] == 'C']
        call_arg = []
        #incoming argument of the routine
        call_arg.append( ('list_complex', 'P%s'% self.l_helas_id) )
        
        self.declaration.add(call_arg[0])
        
        for index,spin in enumerate(self.particles):
            if self.outgoing == index + 1:
                continue
            if self.l_helas_id == index + 1:
                continue
            call_arg.append(('complex','%s%d' % (spin, index +1)))
            self.declaration.add(('list_complex', call_arg[-1][-1])) 
        
        # couplings
        if couplings is None:
            detected_couplings = [name for type, name in self.declaration if name.startswith('COUP')]
            #coup_sort = lambda x,y: int(x[4:])-int(y[4:])  
            detected_couplings.sort(key=lambda x: int(x[4:]) if x[4:] else 0 )
            if detected_couplings:
                couplings = detected_couplings
            else:
                couplings = ['COUP']
                
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
        """Define the Header of the ortran file. This include
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
                size.append(1)
                
            if self.declaration.is_used('P%s' % (i+1)):
                    self.get_one_momenta_def(i+1, out)
                
        # define the resulting momenta
        if self.offshell:
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 2
            for i in range(size_p):
                out.write('    P%s(%s) = %s\n' % (self.outgoing, i, 
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
        

        
        
        
        
    def get_header_txt(self, name=None, couplings=None, **opt):
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

class ALOHAWriterForFortranLoopQP(QP, ALOHAWriterForFortranLoop): 
    """routines for writing out Fortran"""     
    
    def __init__(self, *arg):
        return ALOHAWriterForFortranLoop.__init__(self, *arg)   

def get_routine_name(name=None, outgoing=None, tag=None, abstract=None):
    """ build the name of the aloha function """

    assert (name and outgoing is not None) or abstract

    if tag is None:
        tag = list(abstract.tag)
    else:
        tag=list(tag)

    if name is None:
        prefix=''
        if 'MP' in tag:
            prefix = 'MP_'
            tag.remove('MP')
        if any(t.startswith('P') for t in tag):
            #put the propagator tag at the end
            propa = [t for t in tag if t.startswith('P')][0]
            tag.remove(propa)
            tag.append(propa)
        name = prefix + abstract.name + ''.join(tag)
    
    if outgoing is None:
        outgoing = abstract.outgoing

    return '%s_%s' % (name, outgoing)

def combine_name(name, other_names, outgoing, tag=None, unknown_propa=False):
    """ build the name for combined aloha function """

    def myHash(target_string):
        suffix = ''
        if '%(propa)s' in target_string:
            target_string = target_string.replace('%(propa)s','')
            suffix = '%(propa)s'
            
        if len(target_string)<50:
            return '%s%s' % (target_string, suffix)
        else:
            return 'ALOHA_%s%s' % (str(hash(target_string.lower())).replace('-','m'), suffix)

    if tag and any(t.startswith('P') for t in tag[:-1]):
        # propagator need to be the last entry for the tag
        for i,t  in enumerate(tag):
            if t.startswith('P'):
                tag.pop(i)
                tag.append(t)
                break

    # Two possible scheme FFV1C1_2_X or FFV1__FFV2C1_X
    # If they are all in FFVX scheme then use the first
    p=re.compile(r'^(?P<type>[RFSVT]{2,})(?P<id>\d+)$')
    routine = ''
    if p.search(name):
        base, id = p.search(name).groups()
        routine = name
        for s in other_names:
            try:
                base2,id2 = p.search(s).groups()
            except Exception:
                routine = ''
                break # one matching not good -> other scheme
            if base != base2:
                routine = ''
                break  # one matching not good -> other scheme
            else:
                routine += '_%s' % id2
    
    if routine:
        if tag is not None:
            routine += ''.join(tag)
        if unknown_propa and outgoing:
            routine += '%(propa)s'
        if outgoing is not None:
            return myHash(routine)+'_%s' % outgoing
#            return routine +'_%s' % outgoing
        else:
            return myHash(routine)
#            return routine

    if tag is not None:
        addon = ''.join(tag)
    else:
        addon = ''
        if 'C' in name:
            short_name, addon = name.split('C',1)
            try:
                addon = 'C' + str(int(addon))
            except Exception:
                addon = ''
            else:
                name = short_name
    if unknown_propa:
        addon += '%(propa)s'

#    if outgoing is not None:
#        return '_'.join((name,) + tuple(other_names)) + addon + '_%s' % outgoing
#    else:
#        return '_'.join((name,) + tuple(other_names)) + addon

    if outgoing is not None:
        return myHash('_'.join((name,) + tuple(other_names))) + addon + '_%s' % outgoing
    else:
        return myHash('_'.join((name,) + tuple(other_names))) + addon

class ALOHAWriterForCPP(WriteALOHA): 
    """Routines for writing out helicity amplitudes as C++ .h and .cc files."""
    
    extension = '.c'
    prefix =''
    writer = writers.CPPWriter

    type2def = {}    
    type2def['int'] = 'int '
    type2def['double'] = 'double '
    type2def['complex'] = 'std::complex<double> '
    type2def['pointer_vertex'] = '&' # using complex<double> & vertex)
    type2def['pointer_coup'] = ''
    #variable overwritten by gpu
    realoperator = '.real()'
    imagoperator = '.imag()'
    ci_definition = 'static std::complex<double> cI = std::complex<double>(0.,1.);\n'
    
    
    def change_number_format(self, number):
        """Formating the number"""

        def isinteger(x):
            try:
                return int(x) == x
            except TypeError:
                return False

        if isinteger(number):
            out = '%s.' % (str(int(number)))
        elif isinstance(number, complex):
            if number.imag:
                if number.real:
                    out = '(%s + %s*cI)' % (self.change_number_format(number.real), \
                                    self.change_number_format(number.imag))
                else:
                    if number.imag == 1:
                        out = 'cI'
                    elif number.imag == -1:
                        out = '-cI'
                    else: 
                        out = '%s * cI' % self.change_number_format(number.imag)
            else:
                out = '%s' % (self.change_number_format(number.real))
        else:
            tmp = Fraction(str(number))
            tmp = tmp.limit_denominator(100)
            if not abs(tmp - number) / abs(tmp + number) < 1e-8:
                out = '%.9f' % (number)
            else:
                out = '%s./%s.' % (tmp.numerator, tmp.denominator)
        return out
    
    
    def shift_indices(self, match):
        """shift the indices for non impulsion object"""
        if match.group('var').startswith('P'):
            shift = 0
        else:
            shift =  self.momentum_size - 1
        return '%s[%s]' % (match.group('var'), int(match.group('num')) + shift)
              
    
    def change_var_format(self, name): 
        """Format the variable name to C++ format"""
        
        if '_' in name:
            type = name.type
            decla = name.split('_',1)[0]
            self.declaration.add(('list_%s' % type, decla))
        else:
            self.declaration.add((name.type, name.split('_',1)[0]))
        name = re.sub(r'(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
        return name
            
    def get_fct_format(self, fct):
        """Put the function in the correct format"""
        if not hasattr(self, 'fct_format'):
            one = self.change_number_format(1)
            self.fct_format = {'csc' : '{0}/cos(%s)'.format(one),
                   'sec': '{0}/sin(%s)'.format(one),
                   'acsc': 'asin({0}/(%s))'.format(one),
                   'asec': 'acos({0}/(%s))'.format(one),
                   're': ' real(%s)',
                   'im': 'imag(%s)',
                   'cmath.sqrt':'sqrt(%s)', 
                   'sqrt': 'sqrt(%s)',
                   'complexconjugate': 'conj(dcmplx(%s))',
                   '/' : '{0}/(%s)'.format(one),
                   'abs': 'std::abs(%s)'
                   }
            
        if fct in self.fct_format:
            return self.fct_format[fct]
        else:
            self.declaration.add(('fct', fct))
            return '{0}(%s)'.format(fct)
    
    
    
    
    def get_header_txt(self, name=None, couplings=None,mode=''):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        if mode=='':
            mode = self.mode
        
        
        
        out = StringIO()
        # define the type of function and argument
        if not 'no_include' in mode:
            out.write('#include \"%s.h\"\n\n' % self.name)
        args = []
        for format, argname in self.define_argument_list(couplings):
            if format.startswith('list'):
                type = self.type2def[format[5:]]
                list_arg = '[]'
            else:
                type = self.type2def[format]
                list_arg = ''
            if argname.startswith('COUP'):
                point = self.type2def['pointer_coup']
                args.append('%s%s%s%s'% (type,point, argname, list_arg))
            else:
                args.append('%s%s%s'% (type, argname, list_arg))
                
        if not self.offshell:
            output = '%(doublec)s %(pointer_vertex)s vertex' % {
                'doublec':self.type2def['complex'],
                'pointer_vertex': self.type2def['pointer_vertex']}
            #self.declaration.add(('complex','vertex'))
        else:
            output = '%(doublec)s %(spin)s%(id)d[]' % {
                     'doublec': self.type2def['complex'],
                     'spin': self.particles[self.outgoing -1],
                     'id': self.outgoing}
            self.declaration.add(('list_complex', output))
        
        out.write('%(prefix)s void %(name)s(%(args)s,%(output)s)' % \
                  {'prefix': self.prefix,
                      'output':output, 'name': name, 'args': ', '.join(args)})
        if 'is_h' in mode:
            out.write(';\n')
        else:
            out.write('\n{\n')

        return out.getvalue() 

    def get_declaration_txt(self, add_i=True):
        """ Prototype for how to write the declaration of variable
            Include the symmetry line (entry FFV_2)
        """
        
        out = StringIO()
        argument_var = [name for type,name in self.call_arg]
        # define the complex number CI = 0+1j
        if add_i:
            out.write(self.ci_definition)
                    
        for type, name in self.declaration.tolist():
            if type.startswith('list'):
                type = type[5:]
                if name.startswith('P'):
                    size = 4
                elif not 'tmp' in name:
                    continue
                    #should be define in the header
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
    
                out.write(' %s %s[%s];\n' % (self.type2def[type], name, size))
            elif (type, name) not in self.call_arg:
                if type == 'parameter':
                    model_name =self.routine.model.__name__ 
                    if '.' in model_name:
                        model_name = model_name.split('.')[-1]
                    out.write('std::complex<double> %s = Parameters_%s::getInstance()->mdl_%s;' % (name,self.routine.model.__name__,name))
                    out.write('std::complex<double> mdl_%s = %s;' % (name,name))
                elif type != 'fct':
                    out.write(' %s %s;\n' % (self.type2def[type], name))               

        return out.getvalue()

    def get_foot_txt(self):
        """Prototype for language specific footer"""
        return '}\n'

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
                out.write("    OM{0} = {1};\n    if (M{0} != {1})\n OM{0}={2}/(M{0}*M{0});\n".format( 
                         i+1, self.change_number_format(0), self.change_number_format(1)))
            
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                p.append('{0}{1}{2}[%(i)s]'.format(signs[i],type,i+1,type))    
                
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
                dict_energy = {'i':i}
                out.write('    %s%s[%s] = %s;\n' % (type,self.outgoing, i, 
                                             ''.join(p) % dict_energy))
            if self.declaration.is_used('P%s' % self.outgoing):
                self.get_one_momenta_def(self.outgoing, out)

        
        # Returning result
        return out.getvalue()

    def get_one_momenta_def(self, i, strfile):
        
        type = self.particles[i-1]
        
        if aloha.loop_mode:
            template ='P%(i)d[%(j)d] = %(sign)s%(type)s%(i)d[%(nb)d];\n'
        else:
            template ='P%(i)d[%(j)d] = %(sign)s%(type)s%(i)d[%(nb2)d]%(operator)s;\n'

        nb2 = 0
        for j in range(4):
            if not aloha.loop_mode:
                nb = j 
                if j == 0: 
                    assert not aloha.mp_precision 
                    operator = self.realoperator # not suppose to pass here in mp
                elif j == 1: 
                    nb2 += 1
                elif j == 2:
                    assert not aloha.mp_precision 
                    operator = self.imagoperator # not suppose to pass here in mp
                elif j ==3:
                    nb2 -= 1
            else:
                operator =''
                nb = j
                nb2 = j
            strfile.write(template % {'j':j,'type': type, 'i': i, 
                        'nb': nb, 'nb2': nb2, 'operator':operator,
                        'sign': self.get_P_sign(i)})

            
    def define_expression(self):
        """Write the helicity amplitude in C++ format"""
        
        out = StringIO()

        if self.routine.contracted:
            keys = sorted(self.routine.contracted.keys())
            for name in keys:
                obj = self.routine.contracted[name]
                out.write(' %s = %s;\n' % (name, self.write_obj(obj)))
                self.declaration.add(('complex', name))
        
        def sort_fct(a, b):
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            elif a < b:
                return -1
            else:
                return +1
            
        keys = list(self.routine.fct.keys())        
        keys.sort(key=misc.cmp_to_key(sort_fct))
        for name in keys:
            fct, objs = self.routine.fct[name]
            format = ' %s = %s;\n' % (name, self.get_fct_format(fct))
            out.write(format % ','.join([self.write_obj(obj) for obj in objs]))
            
        

        numerator = self.routine.expr
        if not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)
        if not self.offshell:
            if coup_name == 'COUP':
                mydict = {'num': self.write_obj(numerator.get_rep([0]))}
                for c in ['coup', 'vertex']:
                    if self.type2def['pointer_%s' %c] in ['*']:
                        mydict['pre_%s' %c] = '(*'
                        mydict['post_%s' %c] = ')'
                    else:
                        mydict['pre_%s' %c] = ''
                        mydict['post_%s'%c] = ''
                out.write(' %(pre_vertex)svertex%(post_vertex)s = %(pre_coup)sCOUP%(post_coup)s*%(num)s;\n' %\
                            mydict)
            else:
                mydict= {}
                if self.type2def['pointer_vertex'] in ['*']:
                    mydict['pre_vertex'] = '(*'
                    mydict['post_vertex'] = ')'
                else:
                    mydict['pre_vertex'] = ''
                    mydict['post_vertex'] = ''                 
                mydict['data'] = self.write_obj(numerator.get_rep([0]))
                out.write(' %(pre_vertex)svertex%(post_vertex)s = %(data)s;\n' % 
                          mydict)
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            if 'L' not in self.tag:
                coeff = 'denom'
                mydict = {}
                if self.type2def['pointer_coup'] in ['*']:
                    mydict['pre_coup'] = '(*'
                    mydict['post_coup'] = ')'
                else:
                    mydict['pre_coup'] = ''
                    mydict['post_coup'] = ''
                mydict['coup'] = coup_name
                mydict['i'] = self.outgoing
                if not aloha.complex_mass:
                    if self.routine.denominator:
                        if self.routine.denominator == "1":
                            out.write('    denom = %(pre_coup)s%(coup)s%(post_coup)s;\n' % \
                                  mydict) 
                        else:
                            mydict['denom'] = self.write_obj(self.routine.denominator)
                            out.write('    denom = %(pre_coup)s%(coup)s%(post_coup)s/(%(denom)s);\n' % \
                                  mydict) 
                    else:
                        out.write('    denom = %(pre_coup)s%(coup)s%(post_coup)s/((P%(i)s[0]*P%(i)s[0])-(P%(i)s[1]*P%(i)s[1])-(P%(i)s[2]*P%(i)s[2])-(P%(i)s[3]*P%(i)s[3]) - M%(i)s * (M%(i)s -cI* W%(i)s));\n' % \
                                  mydict)
                else:
                    if self.routine.denominator:
                        raise Exception('modify denominator are not compatible with complex mass scheme')                

                    out.write('    denom = %(pre_coup)s%(coup)s%(post_coup)s/((P%(i)s[0]*P%(i)s[0])-(P%(i)s[1]*P%(i)s[1])-(P%(i)s[2]*P%(i)s[2])-(P%(i)s[3]*P%(i)s[3]) - (M%(i)s*M%(i)s));\n' % \
                              mydict)

                self.declaration.add(('complex','denom'))
                if aloha.loop_mode:
                    ptype = 'list_complex'
                else:
                    ptype = 'list_double'
                self.declaration.add((ptype,'P%s' % self.outgoing))
            else:
                coeff = 'COUP'
                
            for ind in numerator.listindices():
                out.write('    %s[%d]= %s*%s;\n' % (self.outname, 
                                        self.pass_to_HELAS(ind), coeff,
                                        self.write_obj(numerator.get_rep(ind))))
        return out.getvalue()
        
    remove_double = re.compile(r'std::complex<double> (?P<name>[\w]+)\[\]')
    def define_symmetry(self, new_nb, couplings=None):
        """Write the call for symmetric routines"""
        number = self.offshell
        arguments = [name for format, name in self.define_argument_list()]
        new_name = self.name.rsplit('_')[0] + '_%s' % new_nb
        output = '%(spin)s%(id)d' % {
                     'spin': self.particles[self.offshell -1],
                     'id': self.outgoing}
        return  '%s\n %s(%s,%s);\n}' % \
            (self.get_header_txt(new_name, couplings, mode='no_include'), 
             self.name, ','.join(arguments), output)
    
    def get_h_text(self,couplings=None):
        """Return the full contents of the .h file"""

        h_string = StringIO()
        if not self.mode == 'no_include':
            h_string.write('#ifndef '+ self.name + '_guard\n')
            h_string.write('#define ' + self.name + '_guard\n')
            h_string.write('#include <complex>\n\n')

        h_header = self.get_header_txt(mode='no_include__is_h', couplings=couplings)
        h_string.write(h_header)

        for elem in self.routine.symmetries: 
            symmetryhead = h_header.replace( \
                             self.name,self.name[0:-1]+'%s' %(elem))
            h_string.write(symmetryhead)

        if not self.mode == 'no_include':
            h_string.write('#endif\n\n')

        return h_string.getvalue()


    def write_combined_cc(self, lor_names, offshell=None, sym=True, mode=''):
        "Return the content of the .cc file linked to multiple lorentz call."

        # Set some usefull command
        if offshell is None:
            offshell = self.offshell
              
        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        self.name = name
        # write head - momenta - body - foot
        text = StringIO()
        routine = StringIO()
        data = {} # for the formating of the line
                   
        # write header 
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        text.write(self.get_header_txt(name=name, couplings=new_couplings, mode=mode))
  
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
        line = "    %(name)s%(addon)s(%(before_coup)s,%(coup)s%(after_coup)s,%(out)s);\n"
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
                    self.declaration.add(('complex','%stmp' % type))
                data['out'] = '%stmp' % type
            routine.write(line % data)
            if i:
                if not offshell:
                    routine.write( '    vertex = vertex + tmp;\n')
                else:
                    size = self.type_to_size[self.particles[offshell -1]] -2
                    routine.write(""" i= %s;\nwhile (i < %s)\n{\n""" % (self.momentum_size, self.momentum_size+size))
                    routine.write(" %(main)s[i] = %(main)s[i] + %(tmp)s[i];\n i++;\n" %\
                               {'main': main, 'tmp': data['out']})
                    routine.write('}\n')
                    self.declaration.add(('int','i'))
        self.declaration.discard(('complex','COUP'))
        self.declaration.discard(('complex', 'denom'))
        if self.outgoing:
            self.declaration.discard(('list_double', 'P%s' % self.outgoing))
            self.declaration.discard(('double', 'OM%s' % self.outgoing))
        for name in aloha_lib.KERNEL.reduced_expr2:
            self.declaration.discard(('complex', name))
        
        #clean pointless declaration
        #self.declaration.discard
        text.write(self.get_declaration_txt(add_i=False))
        text.write(routine.getvalue())
        text.write(self.get_foot_txt())

        text = text.getvalue()
        return text

    
    def write(self, **opt):
        """Write the .h and .cc files"""

        cc_text = WriteALOHA.write(self, **opt)
        h_text = self.get_h_text()
        
        # write in two file
        if self.out_path:
            writer_h = writers.CPPWriter(self.out_path[:-len(self.extension)] + ".h")
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer_h.write_comments(commentstring)
            writer_h.writelines(h_text)
            
        return h_text, cc_text
 
 
 
    def write_combined(self, lor_names, mode='', offshell=None, **opt):
        """Write the .h and .cc files associated to the combined file"""

        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
        
        if mode == 'self':
            # added to another file
            self.mode = 'no_include'
        

        
        #h_text = self.write_combined_h(lor_names, offshell, **opt)
        cc_text, h_text = StringIO() , StringIO() 
        cc_text.write(self.write_combined_cc(lor_names, offshell, mode=mode,**opt))
        couplings = ['COUP%d' % (i+1) for i in range(len(lor_names)+1)]
        
        if mode == 'self':
            self.mode = 'self'
        h_text.write(self.get_h_text(couplings=couplings))
        
        #ADD SYMETRY
        if sym:
            for elem in self.routine.symmetries:
                self.mode = 'no_include'
                cc_text.write( self.write_combined_cc(lor_names, elem))

        
        if self.out_path:
            # Prepare a specific file
            path = os.path.join(os.path.dirname(self.out_path), self.name)
            commentstring = 'This File is Automatically generated by ALOHA \n'
            
            writer_h = writers.CPPWriter(path + ".h")
            writer_h.write_comments(commentstring)
            writer_h.writelines(h_text.getvalue())
            
            writer_cc = writers.CPPWriter(path + ".cc")
            writer_cc.write_comments(commentstring)
            writer_cc.writelines(cc_text.getvalue())
        
        return h_text.getvalue(), cc_text.getvalue()
        
        
class ALOHAWriterForGPU(ALOHAWriterForCPP):
    
    extension = '.cu'
    prefix ='__device__'
    realoperator = '.real()'
    imagoperator = '.imag()'
    ci_definition = 'cxtype cI = cxtype(0., 1.);\n'
    
    type2def = {}    
    type2def['int'] = 'int '
    type2def['double'] = 'fptype '
    type2def['complex'] = 'cxtype '
    type2def['pointer_vertex'] = '*' # using complex<double> * vertex)
    type2def['pointer_coup'] = ''
    
    def get_header_txt(self, name=None, couplings=None, mode=''):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        text = StringIO()
        #if not 'is_h' in mode:
        #    text.write('__device__=__forceinclude__\n')
        text.write(ALOHAWriterForCPP.get_header_txt(self, name, couplings, mode))
        return text.getvalue()
        
    def get_h_text(self,couplings=None):
        """Return the full contents of the .h file"""

        h_string = StringIO()
        if not self.mode == 'no_include':
            h_string.write('#ifndef '+ self.name + '_guard\n')
            h_string.write('#define ' + self.name + '_guard\n')
            h_string.write('#include "mgOnGpuTypes.h"\n')
            h_string.write('using namespace std;\n\n')

        h_header = self.get_header_txt(mode='no_include__is_h', couplings=couplings)
        h_string.write(h_header)

        for elem in self.routine.symmetries: 
            symmetryhead = h_header.replace( \
                             self.name,self.name[0:-1]+'%s' %(elem))
            h_string.write(symmetryhead)

        if not self.mode == 'no_include':
            h_string.write('#endif\n\n')

        
        return h_string.getvalue()
    
    
    def write_obj_Add_test(self, obj, prefactor=True):
        """Turns addvariable into a string"""

        data = defaultdict(list)
        number = []
        [data[p.prefactor].append(p) if hasattr(p, 'prefactor') else number.append(p)
             for p in obj]

        file_str = StringIO()
        
        if prefactor and obj.prefactor != 1:
            formatted = self.change_number_format(obj.prefactor)
            if formatted.startswith(('+','-')):
                file_str.write('(%s)' % formatted)
            else:
                file_str.write(formatted)
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
        if number:
            total = sum(number)
            file_str.write('+ %s' % self.change_number_format(total))

        file_str.write(')')
        return file_str.getvalue()    
    
    def write_MultVariable_test(self, obj, prefactor=True):
        """Turn a multvariable into a string"""
        
        mult_list = [self.write_variable_id(id) for id in obj]
        
        tmp = mult_list[0]
        for obj in mult_list[1:]:
            tmp = 'cuCmul(%s,%s)' % (obj, tmp)
        
        
        data = {'factors': tmp}
        if prefactor and obj.prefactor != 1:
            if obj.prefactor != -1:
                text = '%(prefactor)s * %(factors)s'
                data['prefactor'] = self.change_number_format(obj.prefactor)
            else:
                text = '-%(factors)s'
        else:
            text = '%(factors)s'
        return text % data
    
    
    def get_header_txt(self, name=None, couplings=None,mode=''):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        if mode=='':
            mode = self.mode
        
        
        
        out = StringIO()
        # define the type of function and argument
        if not 'no_include' in mode:
            out.write('#include \"%s.h\"\n\n' % self.name)
        args = []
        for format, argname in self.define_argument_list(couplings):
            if format.startswith('list'):
                type = self.type2def[format[5:]]
                list_arg = '[]'
            else:
                type = self.type2def[format]
                list_arg = ''
            if argname.startswith('COUP'):
                point = self.type2def['pointer_coup']
                args.append('%s%s%s%s'% (type,point, argname, list_arg))
            else:
                args.append('%s%s%s'% (type, argname, list_arg))
                
        if not self.offshell:
            output = '%(doublec)s %(pointer_vertex)s vertex' % {
                'doublec':self.type2def['complex'],
                'pointer_vertex': self.type2def['pointer_vertex']}
            #self.declaration.add(('complex','vertex'))
        else:
            output = '%(doublec)s %(spin)s%(id)d[]' % {
                     'doublec': self.type2def['complex'],
                     'spin': self.particles[self.outgoing -1],
                     'id': self.outgoing}
            self.declaration.add(('list_complex', output))
        
        out.write('%(prefix)s void %(name)s(const %(args)s, %(output)s)' % \
                  {'prefix': self.prefix,
                      'output':output, 'name': name, 'args': ', const '.join(args)})
        if 'is_h' in mode:
            out.write(';\n')
        else:
            out.write('\n{\n')

        return out.getvalue() 

class ALOHAWriterForPython(WriteALOHA):
    """ A class for returning a file/a string for python evaluation """
    
    extension = '.py'
    writer = writers.PythonWriter
    
    @staticmethod
    def change_number_format(obj, pure_complex=''):
        change_number_format = ALOHAWriterForPython.change_number_format
        if obj.real == 0 and obj.imag:
            if int(obj.imag) == obj.imag: 
                return '%ij' % obj.imag
            else:
                return change_number_format(obj.imag, pure_complex='j')
        elif obj.imag != 0:
            return '(%s+%s)' % (change_number_format(obj.real),
                               change_number_format(obj.imag, pure_complex='j')) 
        elif obj.imag == 0: 
            if int(obj.real) == obj:
                return '%i%s' % (obj.real,pure_complex)
            obj = obj.real
            tmp = Fraction(str(obj))
            tmp = tmp.limit_denominator(100)
            if not abs(tmp - obj) / abs(tmp + obj) < 1e-8:
                out = str(obj)
            elif tmp.denominator != 1:
                out = '%i%s/%i' % (tmp.numerator, pure_complex, tmp.denominator)
            else:
                out = '%i%s' % (tmp.numerator, pure_complex)
        return out 
    
    
    def shift_indices(self, match):
        """shift the indices for non impulsion object"""
        if match.group('var').startswith('P'):
            shift = 0
        else:
            shift = -1 + self.momentum_size
            
        return '%s[%s]' % (match.group('var'), int(match.group('num')) + shift)

    def change_var_format(self, name): 
        """Formatting the variable name to Python format
        start to count at zero. 
        No neeed to define the variable in python -> no need to keep track of 
        the various variable
        """
        
        if '_' not in name:
            self.declaration.add((name.type, name))
        else:
            self.declaration.add(('', name.split('_',1)[0]))
        name = re.sub(r'(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
        
        return name

    def get_fct_format(self, fct):
        """Put the function in the correct format"""
        if not hasattr(self, 'fct_format'):
            one = self.change_number_format(1)
            self.fct_format = {'csc' : '{0}/cmath.cos(%s)'.format(one),
                   'sec': '{0}/cmath.sin(%s)'.format(one),
                   'acsc': 'cmath.asin({0}/(%s))'.format(one),
                   'asec': 'cmath.acos({0}/(%s))'.format(one),
                   're': ' complex(%s).real',
                   'im': 'complex(%s).imag',
                   'cmath.sqrt': 'cmath.sqrt(%s)',
                   'sqrt': 'cmath.sqrt(%s)',
                   'pow': 'pow(%s, %s)',
                   'complexconjugate': 'complex(%s).conjugate()',
                   '/' : '{0}/%s'.format(one),
                   'abs': 'cmath.fabs(%s)'
                   }
            
        if fct in self.fct_format:
            return self.fct_format[fct]
        elif hasattr(cmath, fct):
            self.declaration.add(('fct', fct))
            return 'cmath.{0}(%s)'.format(fct)
        else:
            raise Exception("Unable to handle function name %s (no special rule defined and not in cmath)" % fct)
    
    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            keys = list( self.routine.contracted.keys())
            keys.sort()
            
            for name in keys:
                obj = self.routine.contracted[name]
                out.write('    %s = %s\n' % (name, self.write_obj(obj)))

        def sort_fct(a, b):
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            elif a < b:
                return -1
            else:
                return +1
            
        keys = list(self.routine.fct.keys())        
        keys.sort(key=misc.cmp_to_key(sort_fct))
        for name in keys:
            fct, objs = self.routine.fct[name]
            format = '    %s = %s\n' % (name, self.get_fct_format(fct))
            try:
                text = format % ','.join([self.write_obj(obj) for obj in objs])
            except TypeError:
                text = format % tuple([self.write_obj(obj) for obj in objs])
            finally:
                out.write(text)



        numerator = self.routine.expr
        if not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)

        if not self.offshell:
            if coup_name == 'COUP':
                out.write('    vertex = COUP*%s\n' % self.write_obj(numerator.get_rep([0])))
            else:
                out.write('    vertex = %s\n' % self.write_obj(numerator.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)

            if not 'L' in self.tag:
                coeff = 'denom'
                if not aloha.complex_mass:
                    if self.routine.denominator:
                        out.write('    denom = %(COUP)s/(%(denom)s)\n' % {'COUP': coup_name,\
                                'denom':self.write_obj(self.routine.denominator)}) 
                    else:
                        out.write('    denom = %(coup)s/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s * (M%(i)s -1j* W%(i)s))\n' % 
                          {'i': self.outgoing,'coup':coup_name})
                else:
                    if self.routine.denominator:
                        raise Exception('modify denominator are not compatible with complex mass scheme')                
                    
                    out.write('    denom = %(coup)s/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s**2)\n' % 
                          {'i': self.outgoing,'coup':coup_name})                    
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
            
    
    def get_header_txt(self, name=None, couplings=None, mode=''):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
           
        out = StringIO()
        out.write("import cmath\n")
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
        p = [] # a list for keeping track how to write the momentum
        
        signs = self.get_momentum_conservation_sign()
        
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
               out.write("    OM{0} = 0.0\n    if (M{0}): OM{0}=1.0/M{0}**2\n".format( (i+1) ))
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                p.append('{0}{1}{2}[%(i)s]'.format(signs[i],type,i+1))  
                
            if self.declaration.is_used('P%s' % (i+1)):
                self.get_one_momenta_def(i+1, out)             
             
        # define the resulting momenta
        if self.offshell:
            type = self.particles[self.outgoing-1]
            out.write('    %s%s = wavefunctions.WaveFunction(size=%s)\n' % (type, self.outgoing, out_size))
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 2
            for i in range(size_p):
                dict_energy = {'i':i}
    
                out.write('    %s%s[%s] = %s\n' % (type,self.outgoing, i, 
                                             ''.join(p) % dict_energy))
            
            self.get_one_momenta_def(self.outgoing, out)

               
        # Returning result
        return out.getvalue()

    def get_one_momenta_def(self, i, strfile):
        """return the string defining the momentum"""

        type = self.particles[i-1]
        
        main = '    P%d = [' % i
        if aloha.loop_mode:
            template ='%(sign)s%(type)s%(i)d[%(nb)d]'
        else:
            template ='%(sign)scomplex(%(type)s%(i)d[%(nb2)d])%(operator)s'

        nb2 = 0
        strfile.write(main)
        data = []
        for j in range(4):
            if not aloha.loop_mode:
                nb = j
                if j == 0: 
                    assert not aloha.mp_precision 
                    operator = '.real' # not suppose to pass here in mp
                elif j == 1: 
                    nb2 += 1
                elif j == 2:
                    assert not aloha.mp_precision 
                    operator = '.imag' # not suppose to pass here in mp
                elif j ==3:
                    nb2 -= 1
            else:
                operator =''
                nb = j
                nb2 = j
            data.append(template % {'j':j,'type': type, 'i': i, 
                        'nb': nb, 'nb2': nb2, 'operator':operator,
                        'sign': self.get_P_sign(i)}) 
            
        strfile.write(', '.join(data))
        strfile.write(']\n')


    def define_symmetry(self, new_nb, couplings=None):
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
                    text.write("    for i in range(%s,%s):\n" % (self.momentum_size, self.momentum_size+size))
                    text.write("        %(main)s[i] += tmp[i]\n" %{'main': main})
        
        text.write(self.get_foot_txt())

        #ADD SYMETRY
        if sym:
            for elem in self.routine.symmetries:
                text.write(self.write_combined(lor_names, mode, elem))

        text = text.getvalue()
        if self.out_path:        
            writer = self.writer(self.out_path, 'a')
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.infostr + '\n'
            writer.write_comments(commentstring)
            writer.writelines(text)


        return text


class Declaration_list(set):

    def is_used(self, var):
        if hasattr(self, 'var_name'):
            return var in self.var_name
        self.var_name = [name for type,name in self]
        return var in self.var_name
    
    def add(self,obj):
        if __debug__:
            type, name = obj
            samename = [t for t,n in self if n ==name]
            for type2 in samename:
                assert type2 == type, '%s is defined with two different type "%s" and "%s"' % \
                            (name, type2, type)
            
        set.add(self,obj)
        
    def tolist(self):

        out = list(self)
        out.sort(key=lambda n:n[1])
        return out
    
        

class WriterFactory(object):
    
    def __new__(cls, data, language, outputdir, tags):
        try:
            language = language.lower()
        except AttributeError:
            pass

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
            return ALOHAWriterForCPP(data, outputdir)
        elif language in ['gpu','cudac']:
            return ALOHAWriterForGPU(data, outputdir)
        elif issubclass(language, WriteALOHA):
            return language(data, outputdir)
        else:
            raise Exception('Unknown output format')


    
#unknow_fct_template = """
#cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#       double complex %(fct_name)s(%(args)s)
#       implicit none
#c      Include Model parameter / coupling
#       include \"../MODEL/input.inc\"
#       include \"../MODEL/coupl.inc\"
#c      Defintion of the arguments       
#%(definitions)s
#       
#c      enter HERE the code corresponding to your function.
#c      The output value should be put to the %(fct_name)s variable.
#
#
#       return
#       end
#cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#
#"""
#        
#def write_template_fct(fct_name, nb_args, output_dir):
#        """create a template for function not recognized by ALOHA"""
#
#        dico = {'fct_name' : fct_name,
#                'args': ','.join(['S%i' %(i+1) for i in range(nb_args)]),
#                'definitions': '\n'.join(['       double complex S%i' %(i+1) for i in range(nb_args)])}
#
#        ff = open(pjoin(output_dir, 'additional_aloha_function.f'), 'a')
#        ff.write(unknow_fct_template % dico)
#        ff.close()


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
        self.name = name
        self.particles =  [self.type_to_variable[spin] for spin in \
                          abstract_routine.spins]
        
        self.offshell = abstract_routine.outgoing # position of the outgoing in particle list        
        self.outgoing = self.offshell             # expected position for the argument list
        if 'C%s' %((self.outgoing + 1) // 2) in self.routine.tag:
            #flip the outgoing tag if in conjugate
            self.outgoing = self.outgoing + self.outgoing % 2 - (self.outgoing +1) % 2
            
        #initialize global helper routine
        self.declaration = Declaration_list()
        self.define_argument_list()
                                   
                                       
    def pass_to_HELAS(self, indices, start=0):
        """find the Fortran HELAS position for the list of index""" 
        
        
        if len(indices) == 1:
            return indices[0] + start
        
        ind_name = self.obj.numerator.lorentz_ind 
        if ind_name == ['I3', 'I2']:
            return  4 * indices[1] + indices[0] + start 
        elif len(indices) == 2: 
            return  4 * indices[0] + indices[1] + start 
        else:
            raise Exception, 'WRONG CONTRACTION OF LORENTZ OBJECT for routine %s: %s' \
                    % (self.name, indices)                                 
                                 
    def define_header(self): 
        """ Prototype for language specific header""" 
        pass

    def define_content(self): 
        """Prototype for language specific body""" 
        pass
    
    def define_foot(self):
        """Prototype for language specific footer"""
        pass
    
    def define_argument_list(self):
        """define a list with the string of object required as incoming argument"""

        call_arg = [] #incoming argument of the routine

        conjugate = [2*(int(c[1:])-1) for c in self.routine.tag if c[0] == 'C']
        
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
                 
        call_arg.append(('complex','COUP'))              
        self.declaration.add(('complex','COUP'))
            
        if self.offshell:
            if aloha.complex_mass:
                call_arg.append(('complex','M%s' % self.offshell))              
                self.declaration.add(('complex','M%s' % self.offshell))
            else:
                call_arg.append(('double','M%s' % self.offshell))              
                self.declaration.add(('double','M%s' % self.offshell))                
                call_arg.append(('double','W%s' % self.offshell))              
                self.declaration.add(('double','W%s' % self.offshell))
            
        self.call_arg = call_arg
                
        return call_arg

    def write(self, mode=None):
                         
        self.mode = mode
            
        core_text = self.define_expression()    
        out = StringIO()
        
        out.write(self.get_header_txt())
        out.write(self.get_declaration_txt())
        out.write(self.get_momenta_txt())
        out.write(core_text)
        out.write(self.get_foot_text())

        for elem in self.symmetries:
            out.write('\n')
            out.write(self.define_symmetry(elem))

        text = out.getvalue()
        
        if self.out_path:        
            writer = self.writer(self.out_path)
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.routine.comment + '\n'
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
                data['prefactor'] = self.change_number_format(self.prefactor)
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
                data['prefactor'] = self.change_number_format(self.prefactor)
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
        
        for value, obj_list in data.items():
            add= '+'
            if value not in  [-1,1]:
                file_str.write(self.change_number_format(value))
                file_str.write('*(')
            elif value == -1:
                add = '-' 
                file_str.write('-')
            else:
                file_str.write('+')
                
            file_str.write(add.join([self.write_obj(obj, prefactor=False) 
                                                          for obj in obj_list]))
            if value not in [1,-1]:
                file_str.write(')')
                
        if prefactor and obj.prefactor != 1:
             file_str.write(')')
                
    def write_variable(self, obj):
        return self.change_var_format(obj)
    
    def write_variable_id(self, id):
        
        obj = aloha_lib.KERNEL.objs[id]
        return self.write_variable(obj)   
    
    def change_var_format(self, obj):
        """format the way to write the variable and add it to the declaration list
        """
        print obj, type(obj)
        str_var = str(obj)
        self.declaration.add((obj.type, str_var))        
        return str_var


    
    def make_call_list(self, outgoing=None):
        """find the way to write the call of the functions"""

        if outgoing is None:
            outgoing = self.offshell

        call_arg = [] #incoming argument of the routine

        conjugate = [2*(int(c[1:])-1) for c in self.routine.tag if c[0] == 'C']
        
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
    declare_dict = {'S':'double complex S%d(*)',
                    'F':'double complex F%d(*)',
                    'V':'double complex V%d(*)',
                    'T':'double complex T%s(*)'}
    
    def define_header(self, name=None):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        
        if not name:
            name = self.name
             
        Momenta = self.collected['momenta']
        OverM = self.collected['om']
        
        CallList = self.calllist['CallList']
        declare_list = self.calllist['DeclareList']
        if 'double complex COUP' in declare_list:
            alredy_update = True
        else:
            alredy_update = False
        
        if not alredy_update:    
            declare_list.append('double complex COUP')

        # define the type of function and argument        
        if not self.offshell:
            str_out = 'subroutine %(name)s(%(args)s,vertex)\n' % \
               {'name': name,
                'args': ','.join(CallList+ ['COUP']) }
            if not alredy_update: 
                declare_list.append('double complex vertex') 
        else:
            if not alredy_update:
                declare_list.append('double complex denom')
                declare_list.append('double precision M%(id)d, W%(id)d' % 
                                                          {'id': self.outgoing})
            call_arg = '%(args)s, COUP, M%(id)d, W%(id)d, %(spin)s%(id)d' % \
                    {'args': ', '.join(CallList), 
                     'spin': self.particles[self.offshell -1],
                     'id': self.outgoing}
            str_out = ' subroutine %s(%s)\n' % (name, call_arg) 

        # Forcing implicit None
        str_out += 'implicit none \n'
        
        # Declare all the variable
        for elem in declare_list:
            str_out += elem + '\n'
        if len(OverM) > 0: 
            str_out += 'double complex ' + ','.join(OverM) + '\n'
        if len(Momenta) > 0:
            str_out += 'double precision ' + '(0:3),'.join(Momenta) + '(0:3)\n'

        # Define the contracted variable
        if self.routine.contracted:
            lstring = []
            for tag in self.routine.contracted['order']:
                name, obj, nb = self.routine.contracted[tag]
                lstring.append('double complex %s' % name)
            lstring.append('')
            str_out += '\n'.join(lstring)


        # Add entry for symmetry
        #str_out += '\n'
        #for elem in self.symmetries:
        #    CallList2 = self.reorder_call_list(CallList, self.offshell, elem)
        #    call_arg = '%(args)s, C, M%(id)d, W%(id)d, %(spin)s%(id)d' % \
        #            {'args': ', '.join(CallList2), 
        #             'spin': self.particles[self.offshell -1],
        #             'id': self.offshell}
        #    
        #    
        #    str_out += ' entry %(name)s(%(args)s)\n' % \
        #                {'name': get_routine_name(self.abstractname, elem),
        #                 'args': call_arg}

        return str_out
      
    def define_momenta(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            -definition of the impulsion"""
        # Definition of the Momenta
        
        momenta = self.collected['momenta']
        overm = self.collected['om']
        momentum_conservation = self.calllist['Momentum']
        
        str_out = ''
        # Conservation of Energy Impulsion
        if self.offshell: 
            offshelltype = self.particles[self.offshell -1]
            offshell_size = self.type_to_size[offshelltype]            
            #Implement the conservation of Energy Impulsion
            for i in range(-1,1):
                str_out += '%s%d(%d)= ' % (offshelltype, self.outgoing, \
                                                              offshell_size + i)
                
                pat=re.compile(r'^[-+]?(?P<spin>\w)')
                for elem in momentum_conservation:
                    spin = pat.search(elem).group('spin') 
                    str_out += '%s(%d)' % (elem, self.type_to_size[spin] + i)  
                str_out += '\n'  
                    
        # Momentum
        for mom in momenta:
            #Mom is in format PX with X the number of the particle
            index = int(mom[1:])
            type = self.particles[index - 1]
            energy_pos = self.type_to_size[type] -1
            sign = 1
            if self.offshell == index and type in ['V','S']:
                sign = -1
            if 'C%s' % ((index +1) // 2)  in self.routine.tag: 
                if index == self.outgoing:
                    pass
                elif index % 2 and index -1 != self.outgoing:
                    pass
                elif index % 2 == 1 and index + 1  != self.outgoing:
                    pass
                else:
                    sign *= -1
            
            if sign == -1 :
                sign = '-'
            else:
                sign = ''
                            
            str_out += '%s(0) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)
            str_out += '%s(1) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(2) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(3) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)            
            
                   
        # Definition for the One Over Mass**2 terms
        for elem in overm:
            #Mom is in format OMX with X the number of the particle
            index = int(elem[2:])
            str_out += 'OM%d = 0d0\n' % (index)
            str_out += 'if (M%d .ne. 0d0) OM%d' % (index, index) + '=1d0/M%d**2\n' % (index) 
        
        # Returning result
        return str_out
        
        
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
        if '_' in name:
            name = name.replace('_', '(', 1) + ')'
        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
        return name
  
    zero_pattern = re.compile(r'''0+$''')
    def change_number_format(self, number):
        """Formating the number"""

        def isinteger(x):
            try:
                return int(x) == x
            except TypeError:
                return False
            

        if isinteger(number):
            out = str(int(number))
        elif isinstance(number, complex):
            if number.imag:
                out = '(%s, %s)' % (self.change_number_format(number.real) , \
                                    self.change_number_format(number.imag))
            else:
                out = self.change_number_format(number.real)
        else:
            out = '%.9f' % number
            out = self.zero_pattern.sub('', out)
        return out
        
    
    def define_expression(self):
        OutString = ''
        
        if self.routine.contracted:
            string = ''
            for tag in self.routine.contracted['order']:
                name, obj, nb = self.routine.contracted[tag]
                string += '%s = %s ! used %s times\n' % (name, self.write_obj(obj),nb)
            string = string.replace('+-', '-')
            OutString += string
            
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'Vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                string = re.sub('\((?P<num>[+-]*[0-9])(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
                string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>.*(0d0,1d0)', string)
                OutString = OutString + string + '\n'
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.outgoing)
            numerator = self.obj.numerator
            denominator = self.obj.denominator
            for ind in denominator.listindices():
                denom = self.write_obj(denominator.get_rep(ind))
            string = 'denom =' + '1d0/(' + denom + ')'
            string = string.replace('+-', '-')
            string = re.sub('\((?P<num>[+-]*[0-9])\+(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
            string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>*(0d0,1d0)', string)
            OutString = OutString + string + '\n'
            for ind in numerator.listindices():
                string = '%s(%d)= COUP*denom*' % (OffShellParticle, self.pass_to_HELAS(ind, start=1))
                string += self.write_obj(numerator.get_rep(ind))
                string = string.replace('+-', '-')
                string = re.sub('\((?P<num>[+-][0-9])\+(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
                string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>*(0d0,1d0)', string)
                OutString = OutString + string + '\n' 
        return OutString 
    
    def define_symmetry(self, new_nb):
        number = self.offshell 
        calls = self.calllist['CallList']
                                                                
        Outstring = 'call '+self.name+'('+','.join(calls)+',COUP,M%s,W%s,%s%s)\n' \
                         %(number,number,self.particles[number-1],number)
        return Outstring
    
    def define_foot(self):
        return 'end\n\n' 

    def write(self, mode='self'):
                         
        # write head - momenta - body - foot
        text = self.define_header()+'\n'
        text += self.define_momenta()+'\n'
        text += self.define_expression()
        text += self.define_foot()
        
        sym_text = []
        for elem in self.symmetries: 
            symmetryhead = self.define_header().replace( \
                             self.name,self.name[0:-1]+'%s' %(elem))
            symmetrybody = self.define_symmetry(elem)
            
            sym_text.append(symmetryhead + symmetrybody + self.define_foot())
            
            
        if self.out_path:
            writer = writers.FortranWriter(self.out_path)
            writer.downcase = False 
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.comment + '\n'
            writer.write_comments(commentstring)
            writer.writelines(text)
            for text in sym_text:
                writer.write_comments('\n%s\n' % ('#'*65))
                writer.writelines(text)
        else:
            for stext in sym_text:
                text += '\n\n' + stext

        return text + '\n'
            
    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1 
            offshell = self.offshell
        else:
            sym = None  # deactivate symetry
            
        name = combine_name(self.abstractname, lor_names, offshell, self.routine.tag)

                 
        # write header 
        header = self.define_header(name=name)
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        text = header.replace('COUP', ','.join(new_couplings))
        # define the TMP for storing output        
        if not offshell:
            text += ' double complex TMP\n'
        else:
            spin = self.particles[offshell -1] 
            text += ' double complex TMP(%s)\n integer i' % self.type_to_size[spin]         
        
        # Define which part of the routine should be called
        addon = ''.join(self.routine.tag) + '_%s' % self.offshell
        
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

        # how to call the routine
        if not offshell:
            main = 'vertex'
            call_arg = '%(args)s, %(COUP)s, %(LAST)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'spin': self.particles[self.offshell -1],
                     'LAST': '%s'}
        else:
            main = '%(spin)s%(id)d' % \
                          {'spin': self.particles[offshell -1],
                           'id': self.outgoing}
            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d, %(LAST)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'id': self.outgoing,
                     'LAST': '%s'}

        # make the first call
        line = " CALL %s%s("+call_arg+")\n"
        text += '\n\n' + line % (self.name, '', 1, main)
        
        # make the other call
        for i,lor in enumerate(lor_names):
            text += line % (lor, addon, i+2, 'TMP')
            if not offshell:
                text += ' vertex = vertex + tmp\n'
            else:
                size = self.type_to_size[spin] -2
                text += """ do i=1,%(id)d
                %(main)s(i) = %(main)s(i) + tmp(i)
                enddo\n""" %  {'id': size, 'main':main}
                

        text += self.define_foot()
        
        #ADD SYMETRY
        if sym:
            for elem in self.symmetries:
                text += self.write_combined(lor_names, mode, elem)
            
            
        if self.out_path:
            # Prepare a specific file
            path = os.path.join(os.path.dirname(self.out_path), name+'.f')
            writer = writers.FortranWriter(path)
            writer.downcase = False 
            commentstring = 'This File is Automatically generated by ALOHA \n'
            writer.write_comments(commentstring)
            writer.writelines(text)
        
        return text
    
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
    
    declare_dict = {'S':'double complex S%d[3]',
                    'F':'double complex F%d[6]',
                    'V':'double complex V%d[6]',
                    'T':'double complex T%s[18]'}
    
    def define_header(self, name=None):
        """Define the headers for the C++ .h and .cc files. This include
            - function tag
            - definition of variable
            - momentum conservation
        """
        
        if name is None:
            name = self.name
            
        #Width = self.collected['width']
        #Mass = self.collected['mass']
        
        CallList = self.calllist['CallList'][:]
        
        local_declare = []
        OffShell = self.offshell
        OffShellParticle = OffShell -1 
        # Transform function call variables to C++ format
        for i, call in enumerate(CallList):
            CallList[i] = "complex<double> %s[]" % call
        #if Mass:
        #    Mass[0] = "double %s" % Mass[0]
        #if Width:
        #    Width[0] = "double %s" % Width[0]
        
        # define the type of function and argument
        if not OffShell:
            str_out = 'void %(name)s(%(args)s, complex<double>& vertex)' % \
               {'name': name,
                'args': ','.join(CallList + ['complex<double> COUP'])}
        else: 
            str_out = 'void %(name)s(%(args)s, double M%(number)d, double W%(number)d, complex<double>%(out)s%(number)d[])' % \
              {'name': name,
               'args': ','.join(CallList+ ['complex<double> COUP']),
               'out': self.particles[self.outgoing - 1],
               'number': self.outgoing 
               }

        h_string = str_out + ";\n\n"
        cc_string = str_out + "{\n"
        if self.routine.contracted:
            lstring = []
            for tag in self.routine.contracted['order']:
                name, obj, nb = self.routine.contracted[tag]
                lstring.append('complex<double> %s;\n' % name)
            lstring.append('')
            str_out += '\n'.join(lstring)


        return {'h_header': h_string, 'cc_header': cc_string}
            
    def define_momenta(self):
        """Write the expressions for the momentum of the outgoing
        particle."""

        momenta = self.collected['momenta']
        overm = self.collected['om']
        momentum_conservation = self.calllist['Momentum']
        
        str_out = ''
        # Declare auxiliary variables
        if self.offshell:
            str_out += 'complex<double> denom;\n'
        if len(overm) > 0: 
            str_out += 'complex<double> %s;\n' % ','.join(overm)
        if len(momenta) > 0:
            str_out += 'double %s[4];\n' % '[4],'.join(momenta)

        # Energy
        if self.offshell: 
            offshelltype = self.particles[self.offshell -1]
            offshell_size = self.type_to_size[offshelltype]            
            #Implement the conservation of Energy Impulsion
            for i in range(-2,0):
                str_out += '%s%d[%d]= ' % (offshelltype, self.outgoing,
                                           offshell_size + i)
                
                pat=re.compile(r'^[-+]?(?P<spin>\w)')
                for elem in momentum_conservation:
                    spin = pat.search(elem).group('spin') 
                    str_out += '%s[%d]' % (elem, self.type_to_size[spin] + i)  
                str_out += ';\n'
        
        # Momentum
        for mom in momenta:
            #Mom is in format PX with X the number of the particle
            index = int(mom[1:])
            
            type = self.particles[index - 1]
            energy_pos = self.type_to_size[type] - 2
            sign = 1
            if self.offshell == index and type in ['V', 'S']:
                sign = -1
            if 'C%s' % ((index +1) // 2)  in self.routine.tag: 
                if index == self.outgoing:
                    pass
                elif index % 2 and index -1 != self.outgoing:
                    pass
                elif index %2 == 1 and index + 1  != self.outgoing:
                    pass
                else:
                    sign *= -1
            
            if sign == -1 :
                sign = '-'
            else:
                sign = ''
                   
            str_out += '%s[0] = %s%s%d[%d].real();\n' % (mom, sign, type, index, energy_pos)
            str_out += '%s[1] = %s%s%d[%d].real();\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s[2] = %s%s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s[3] = %s%s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos)            
            
        # Definition for the One Over Mass**2 terms
        for elem in overm:
            #Mom is in format OMX with X the number of the particle
            index = int(elem[2:])
            str_out += 'OM%d = 0;\n' % (index)
            str_out += 'if (M%d != 0) OM%d' % (index, index) + '= 1./pow(M%d,2);\n' % (index) 
        
        # Returning result
        return str_out
        
        
    def change_var_format(self, name): 
        """Format the variable name to C++ format"""
        
        if '_' in name:
            name = name.replace('_','[',1) +']'
        outstring = ''
        counter = 0
        for elem in re.finditer('[FVTSfvts][0-9]\[[0-9]\]',name):
            outstring += name[counter:elem.start()+2]+'['+str(int(name[elem.start()+3:elem.start()+4])-1)+']'
            counter = elem.end()
        outstring += name[counter:]
        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
        return outstring
    
    def change_number_format(self, number):
        """Format numbers into C++ format"""
        if isinstance(number, complex):
            if number.real == int(number.real) and \
                   number.imag == int(number.imag):
                out = 'complex<double>(%d., %d.)' % \
                      (int(number.real), int(number.imag))
            else:
                out = 'complex<double>(%.9f, %.9f)' % \
                      (number.real, number.imag)                
        else:
            if number == int(number):
                out = '%d.' % int(number)
            else:
                out = '%.9f' % number
        return out
    
    def define_expression(self):
        """Write the helicity amplitude in C++ format"""
        OutString = '' 

        if self.routine.contracted:        
            string = ''
            for tag in self.routine.contracted['order']:
                name, obj, nb = self.routine.contracted[tag]
                string += '%s = %s ! used %s times\n' % (name, self.write_obj(obj),nb)
            string = string.replace('+-', '-')
        
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + ';\n'
        else:
            OffShellParticle = self.particles[self.offshell-1]+'%s'%(self.outgoing)
            numerator = self.obj.numerator
            denominator = self.obj.denominator
            for ind in denominator.listindices():
                denom = self.write_obj(denominator.get_rep(ind))
            string = 'denom =' + '1./(' + denom + ')'
            string = string.replace('+-', '-')
            OutString = OutString + string + ';\n'
            for ind in numerator.listindices():
                string = '%s[%d]= COUP*denom*' % (OffShellParticle, self.pass_to_HELAS(ind))
                string += self.write_obj(numerator.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + ';\n' 
        OutString = re.sub('(?P<variable>[A-Za-z]+[0-9]\[*[0-9]*\]*)\*\*(?P<num>[0-9])','pow(\g<variable>,\g<num>)',OutString)
        return OutString 

    remove_double = re.compile('complex<double> (?P<name>[\w]+)\[\]')
    def define_symmetry(self, new_nb):
        """Write the call for symmetric routines"""
        calls = self.calllist['CallList']
        
        for i, call in enumerate(calls):
            if self.remove_double.match(call):
                calls[i] = self.remove_double.match(call).group('name')
                
        # For the call, need to remove the type specification
        #calls = [self.remove_double.match(call).group('name') for call in \
        #         calls]
        number = self.offshell 
        Outstring = self.name+'('+','.join(calls)+',COUP,M%s,W%s,%s%s);\n' \
                         %(number,number,self.particles[self.offshell-1],number)
        return Outstring
    
    def define_foot(self):
        """Return the end of the function definition"""

        return '}\n\n' 

    def write_h(self, header, compiler_cmd=True):
        """Return the full contents of the .h file"""

        h_string = ''
        if compiler_cmd:
            h_string = '#ifndef '+ self.name + '_guard\n'
            h_string += '#define ' + self.name + '_guard\n'
            h_string += '#include <complex>\n'
            h_string += 'using namespace std;\n\n'

        h_header = header['h_header']

        h_string += h_header

        for elem in self.symmetries: 
            symmetryhead = h_header.replace( \
                             self.name,self.name[0:-1]+'%s' %(elem))
            h_string += symmetryhead

        if compiler_cmd:
            h_string += '#endif\n\n'

        return h_string

    def write_combined_h(self, lor_names, offshell=None, compiler_cmd=True):
        """Return the content of the .h file linked to multiple lorentz call."""
        
        name = combine_name(self.abstractname, lor_names, offshell, self.routine.tag)
        text= ''
        if compiler_cmd:
            text = '#ifndef '+ name + '_guard\n'
            text += '#define ' + name + '_guard\n'
            text += '#include <complex>\n'
            text += 'using namespace std;\n\n'
        
        # write header 
        header = self.define_header(name=name)
        h_header = header['h_header']
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        h_string = h_header.replace('COUP', ', complex <double>'.join(new_couplings))
        text += h_string 
        
        for elem in self.symmetries: 
            text += h_string.replace(name, name[0:-1]+str(elem))
        
        if compiler_cmd:
            text += '#endif'
        
        return text

    def write_cc(self, header, compiler_cmd=True):
        """Return the full contents of the .cc file"""

        cc_string = ''
        if compiler_cmd:
            cc_string = '#include \"%s.h\"\n\n' % self.name
        cc_header = header['cc_header']
        cc_string += cc_header
        cc_string += self.define_momenta()
        cc_string += self.define_expression()
        cc_string += self.define_foot()

        for elem in self.symmetries: 
            symmetryhead = cc_header.replace( \
                             self.name,self.name[0:-1]+'%s' %(elem))
            symmetrybody = self.define_symmetry(elem)
            cc_string += symmetryhead
            cc_string += symmetrybody
            cc_string += self.define_foot()

        return cc_string

    def write_combined_cc(self, lor_names, offshell=None, compiler_cmd=True, sym=True):
        "Return the content of the .cc file linked to multiple lorentz call."
        
        # Set some usefull command
        if offshell is None:
            offshell = self.offshell
            
        name = combine_name(self.abstractname, lor_names, offshell, self.routine.tag)

        text = ''
        if compiler_cmd:
            text += '#include "%s.h"\n\n' % name
           
        # write header 
        header = self.define_header(name=name)['cc_header']
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        text += header.replace('COUP', ', complex<double>'.join(new_couplings))
        # define the TMP for storing output        
        if not offshell:
            text += 'complex<double> tmp;\n'
        else:
            spin = self.particles[offshell -1] 
            text += 'complex<double> tmp[%s];\n int i = 0;' % self.type_to_size[spin]         

        # Define which part of the routine should be called
        addon = ''
        if 'C' in self.name:
            short_name, addon = name.split('C',1)
            if addon.split('_')[0].isdigit():
                addon = 'C' +self.name.split('C',1)[1]
            elif all([n.isdigit() for n in addon.split('_')[0].split('C')]):
                addon = 'C' +self.name.split('C',1)[1]
            else:
                addon = '_%s' % self.offshell
        else:
            addon = '_%s' % self.offshell

        # how to call the routine
        if not offshell:
            main = 'vertex'
            call_arg = '%(args)s, %(COUP)s, %(LAST)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'spin': self.particles[self.offshell -1],
                     'LAST': '%s'}
        else:
            main = '%(spin)s%(id)d' % \
                          {'spin': self.particles[self.offshell -1],
                           'id': self.offshell}
            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d, %(LAST)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'id': self.offshell,
                     'LAST': '%s'}

        # make the first call
        line = "%s%s("+call_arg+");\n"
        text += '\n\n' + line % (self.name, '', 1, main)
        
        # make the other call
        for i,lor in enumerate(lor_names):
            text += line % (lor, addon, i+2, 'tmp')
            if not offshell:
                text += ' vertex = vertex + tmp;\n'
            else:
                size = self.type_to_size[spin] -2
                text += """ while (i < %(id)d)
                {
                %(main)s[i] = %(main)s[i] + tmp[i];
                i++;
                }\n""" %  {'id': size, 'main':main}
                

        text += self.define_foot()
        
        if sym:
            for elem in self.symmetries:
                text += self.write_combined_cc(lor_names, elem, 
                                                      compiler_cmd=compiler_cmd,
                                                      sym=False)
            
            
        if self.out_path:
            # Prepare a specific file
            path = os.path.join(os.path.dirname(self.out_path), name+'.f')
            writer = writers.FortranWriter(path)
            writer.downcase = False 
            commentstring = 'This File is Automatically generated by ALOHA \n'
            writer.write_comments(commentstring)
            writer.writelines(text)
        
        return text


    
    def write(self, mode='self', **opt):
        """Write the .h and .cc files"""


        # write head - momenta - body - foot
        header = self.define_header()
        h_text = self.write_h(header, **opt)
        cc_text = self.write_cc(header, **opt)
        
        # write in two file
        if self.out_path:
            writer_h = writers.CPPWriter(self.out_path + ".h")
            writer_cc = writers.CPPWriter(self.out_path + ".cc")
            commentstring = 'This File is Automatically generated by ALOHA \n'
            commentstring += 'The process calculated in this file is: \n'
            commentstring += self.comment + '\n'
            writer_h.write_comments(commentstring)
            writer_cc.write_comments(commentstring)
            writer_h.writelines(h_text)
            writer_cc.writelines(cc_text)
            
        return h_text, cc_text
 
 
 
    def write_combined(self, lor_names, mode='self', offshell=None, **opt):
        """Write the .h and .cc files associated to the combined file"""

        # Set some usefull command
        if offshell is None:
            offshell = self.offshell
        
        name = combine_name(self.abstractname, lor_names, offshell, self.routine.tag)
        
        h_text = self.write_combined_h(lor_names, offshell, **opt)
        cc_text = self.write_combined_cc(lor_names, offshell, sym=True, **opt)
        
        if self.out_path:
            # Prepare a specific file
            path = os.path.join(os.path.dirname(self.out_path), name)
            commentstring = 'This File is Automatically generated by ALOHA \n'
            
            writer_h = writers.CPPWriter(path + ".h")
            writer_h.write_comments(commentstring)
            writer_h.writelines(h_text)
            
            writer_cc = writers.CPPWriter(path + ".cc")
            writer_cc.write_comments(commentstring)
            writer_cc.writelines(cc_text)
        else:
            return h_text, cc_text
        
        return h_text, cc_text
        
class ALOHAWriterForPython(WriteALOHA):
    """ A class for returning a file/a string for python evaluation """
    
    extension = '.py'
    
    def __init__(self, abstract_routine, dirpath=None):
        """ standard init but if dirpath is None the write routine will
        return a string (which can be evaluated by an exec"""
        
        WriteALOHA.__init__(self, abstract_routine, dirpath)
        self.outname = '%s%s' % (self.particles[self.offshell -1], \
                                                               self.outgoing)
    
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
            
        name = re.sub('(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
        # call the mother to add the 
        #return WriteALOHA.change_var_format(self, name)
        
        return name
    
    def define_expression(self):
        """Define the functions in a 100% way """

        out = StringIO()

        if self.routine.contracted:
            for obj,name in self.routine.contracted.values():
                out.write('%s = %s\n' % (name, self.write_obj(obj)))
        
        if not self.offshell:
            out.write('vertex = COUP*%s\n' % self.write_obj(self.obj.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            numerator = self.routine.expr
            
            out.write('denom = 1.0/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s * (M%(i)s -1j* W%(i)s))\n' % {'i': self.offshell})

            for ind in numerator.listindices():
                out.write('%s[%d]= COUP*denom*%s\n' % (self.outname, 
                                        self.pass_to_HELAS(ind), 
                                        self.write_obj(numerator.get_rep(ind))))
        return out.getvalue()
    
    def define_foot(self):
        if not self.offshell:
            return '    return vertex\n\n'
        else:
            return '    return %s\n\n' % (self.outname)
            
    
    def define_header(self, name=None):
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
        str_out.write('def %(name)s(%(args)s):\n' % \
                               {'name': name, 'args': ','.join(CallList) })
          
        return str_out     

    def define_momenta(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
                    
        out = StringIO()
        
        # Define all the required momenta
        p1,p2 = [], [] # a list for keeping track how to write the momentum
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
                out.write("    OM{1} = 0.0\nif (M{1}): OM{1}=1.0/M{1}**2\n".format( (index) ))
            
            if i+1 == self.offshell:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                energy_pos = self.type_to_size[type] -2
                p1.append('%s%s[%s]' % (type,i+1, energy_pos))
                p1.append('%s%s[%s]' % (type,i+1, energy_pos+1))      
            
            if self.declaration.is_used('P%s' % (i+1)):
                energy_pos = self.type_to_size[type] -2
                out.write('''    P%(i)d = [complex(%(type)s%(i)d[%(nb)]).real,
                complex(%(type)s%(i)d[%(nb2)]).real,
                complex(%(type)s%(i)d[%(nb2)]).imag,
                complex(%(type)s%(i)d[%(nb)]).imag]\n''' % \
                {'type': type, 'i': i+1, 'nb': energy_pos, 'nb2': energy_pos + 1})               
                
        # define the resulting momenta
        if self.offshell:
            energy_pos = out_size -2
            out.write('    %s = wavefunctions.WaveFunction(size=%s)\n' % \
                                                       (self.outname, out_size))
            out.write('    %s%s[%s] = %s\n' % (type,i+1, energy_pos, '+'.join(p1)))
            out.write('    %s%s[%s] = %s\n' % (type,i+1, energy_pos+1, '+'.join(p2)))
            
            out.write('''    P%(i)d = [complex(%(type)s%(i)d[%(nb)]).real,
                complex(%(type)s%(i)d[%(nb2)]).real,
                complex(%(type)s%(i)d[%(nb2)]).imag,
                complex(%(type)s%(i)d[%(nb)]).imag]\n''' % \
                {'type': out_type, 'i': self.offshell, 'nb': energy_pos, 
                                                         'nb2': energy_pos + 1})
            
        
        # Returning result
        return out_size.get_value()

    def define_symmetry(self, new_nb):
        number = self.offshell 
        calls = self.calllist['CallList']
        Outstring = '    return '+self.name+'('+','.join(calls)+',COUP,M%s,W%s)\n' \
                         %(number,number)
        return Outstring        


    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
        name = combine_name(self.abstractname, lor_names, offshell, self.routine.tag)

        # write head - momenta - body - foot
        text = ''
        #if mode == 'mg5':
        #    text = 'import aloha.template_files.wavefunctions as wavefunctions\n'
        #else:
        #    text = 'import wavefunctions\n'
                    
                 
        # write header 
        header = self.define_header(name=name)
        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        header = header.replace('COUP', ','.join(new_couplings))
        
        text += header
  
        # Define which part of the routine should be called
        addon = ''.join(self.routine.tag) + '_%s' % self.offshell

        # how to call the routine
        if not offshell:
            main = 'vertex'
            call_arg = '%(args)s, %(COUP)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'spin': self.particles[self.offshell -1]}
        else:
            main = '%(spin)s%(id)d' % \
                          {'spin': self.particles[self.offshell -1],
                           'id': self.outgoing}
            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'id': self.outgoing}

        # make the first call
        line = "    %s = %s%s("+call_arg+")\n"
        text += '\n\n' + line % (main, self.name, '', 1)
        
        # make the other call
        for i,name in enumerate(lor_names):
            text += line % ('tmp',name, addon, i+2)
            if not offshell:
               text += '    vertex += tmp\n'
            else:
                size = self.type_to_size[self.particles[offshell -1]] -2
                text += "    for i in range(%(id)d):\n" % {'id': size}
                text += "        %(main)s[i] += tmp[i]\n" %{'main': main}
        
        text += '    '+self.define_foot()

        #ADD SYMETRY
        if sym:
            for elem in self.symmetries:
                text += self.write_combined(lor_names, mode, elem)
            
        return text

class Declaration_list(set):

    def is_used(var):
        if hasattr(self, 'var_name'):
            return var in self.var_name
        self.var_name = [name for type,name in self]
        return var in self.var_name
            
            
        
        






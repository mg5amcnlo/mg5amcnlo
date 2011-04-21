try:
    import madgraph.iolibs.file_writers as writers 
except:
    import aloha.file_writers as writers
    
import os
import re 
from numbers import Number

class WriteALOHA: 
    """ Generic writing functions """ 
    
    power_symbol = '**'
    change_var_format = str
    change_number_format = str
    extension = ''
    type_to_variable = {2:'F',3:'V',5:'T',1:'S'}
    type_to_size = {'S':3, 'T':18, 'V':6, 'F':6}
    
    def __init__(self, abstract_routine, dirpath):


        name = get_routine_name(abstract_routine.name, abstract_routine.outgoing)
        if dirpath:
            self.dir_out = dirpath
            self.out_path = os.path.join(dirpath, name + self.extension)
        else:
            self.out_path = None
            self.dir_out = None

        self.obj = abstract_routine.expr
        self.particles =  [self.type_to_variable[spin] for spin in \
                          abstract_routine.spins]
        self.namestring = name
        self.abstractname = abstract_routine.name
        self.comment = abstract_routine.infostr
        self.offshell = abstract_routine.outgoing 
        self.symmetries = abstract_routine.symmetries

        #prepare the necessary object
        self.collect_variables() # Look for the different variables
        self.make_all_lists()    # Compute the expression for the call ordering
                                 #the definition of objects,..
                                 
                                 
                                 
                                 
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
            raise Exception, 'WRONG CONTRACTION OF LORENTZ OBJECT'                                 
                                 
    def collect_variables(self):
        """Collects Momenta,Mass,Width into lists"""
         
        MomentaList = set()
        OverMList = set()
        for elem in self.obj.tag:
            if elem.startswith('P'):
                MomentaList.add(elem)
            elif elem.startswith('O'):
                OverMList.add(elem) 

        MomentaList = list(MomentaList)
        OverMList = list(OverMList)
        
        self.collected = {'momenta':MomentaList, 'om':OverMList}
        
        return self.collected

    def define_header(self): 
        """ Prototype for language specific header""" 
        pass

    def define_content(self): 
        """Prototype for language specific body""" 
        pass
    
    def define_foot(self):
        """Prototype for language specific footer"""
        pass
    
    def write_indices_part(self, indices, obj): 
        """Routine for making a string out of indices objects"""
        
        text = 'output(%s)' % indices
        return text                 
        
    def write_obj(self, obj):
        """Calls the appropriate writing routine"""

        try:
            vartype = obj.vartype
        except:
            return self.change_number_format(obj)

        if vartype == 2 : # MultVariable
            return self.write_obj_Mult(obj)
        elif not vartype: # Variable
            return self.write_obj_Var(obj)
        elif vartype == 1 : # AddVariable
            return self.write_obj_Add(obj)
        elif vartype == 5: # ConstantObject
            return self.change_number_format(obj.value)
        else: 
            raise Exception('Warning unknown object: %s' % obj.vartype)

    def write_obj_Mult(self, obj):
        """Turn a multvariable into a string""" 
        mult_list = [self.write_obj(factor) for factor in obj] 
        text = '(' 
        if obj.prefactor != 1:
            if obj.prefactor != -1:
                text = self.change_number_format(obj.prefactor) + '*' + text 
            else:
                text = '-' + text
        return text + '*'.join(mult_list) + ')'
    
    def write_obj_Add(self, obj):
        """Turns addvariable into a string"""
        mult_list = [self.write_obj(factor) for factor in obj]
        prefactor = ''
        if obj.prefactor == 1:
            prefactor = ''
        elif obj.prefactor == -1:
            prefactor = '-'
        else:
            prefactor = '%s*' % self.change_number_format(obj.prefactor)

        return '(%s %s)' % (prefactor, '+'.join(mult_list))

        
    def write_obj_Var(self, obj):
        text = ''
        if obj.prefactor != 1:
            if obj.prefactor != -1: 
                text = self.change_number_format(obj.prefactor) + '*' + text
            else:
                text = '-' + text
        text += self.change_var_format(obj.variable)
        if obj.power != 1:
            text = text + self.power_symbol + str(obj.power)
        return text

    def make_all_lists(self):
        """ Make all the list for call ordering, conservation impulsion, 
        basic declaration"""
        
        DeclareList = self.make_declaration_list()
        CallList = self.make_call_list()
        MomentumConserve = self.make_momentum_conservation()

        self.calllist =  {'CallList':CallList,'DeclareList':DeclareList, \
                           'Momentum':MomentumConserve}

    
    def make_call_list(self, outgoing=None):
        """find the way to write the call of the functions"""

        if outgoing is None:
            outgoing = self.offshell

        call_arg = [] #incoming argument of the routine
        

        call_arg = ['%s%d' % (spin, index +1) 
                                 for index,spin in enumerate(self.particles)
                                 if outgoing != index +1]
                
        return call_arg

#    def reorder_call_list(self, call_list, old, new):
#        """ restore the correct order for symmetries """
#        raise
#        #spins = self.particles
#        #assert(0 < old < new)
#        #old, new = old -1, new -1 # pass in real position in particles list
#        #assert(spins[old] == spins[new])
#        #spin =spins[old]
#        
#        new_call = call_list[:]
#        #val = new_call.pop(old)
#        #new_call.insert(new - 1, val)
#        return new_call
    
        
    def make_momentum_conservation(self):
        """ compute the sign for the momentum conservation """
        
        if not self.offshell:
            return []
        # How Convert  sign to a string
        sign_dict = {1: '+', -1: '-'}
        # help data 
        momentum_conserve = []
        nb_fermion =0
        
        #compute global sign
        if not self.offshell % 2 and self.particles[self.offshell -1] == 'F': 
            global_sign = 1
        else:
            global_sign = -1
        
        
        for index, spin in enumerate(self.particles): 
            assert(spin in ['S','F','V','T'])  
      
            #compute the sign
            if spin != 'F':
                sign = -1 * global_sign
            elif nb_fermion % 2 == 0:
                sign = global_sign
                nb_fermion += 1
            else: 
                sign = -1 * global_sign
                nb_fermion += 1
            
            # No need to include the outgoing particles in the definitions
            if index == self.offshell -1:
                continue 
            
            # write the
            momentum_conserve.append('%s%s%d' % (sign_dict[sign], spin, \
                                                                     index + 1))
        
        # Remove the
        if momentum_conserve[0][0] == '+':
            momentum_conserve[0] = momentum_conserve[0][1:]
        
        return momentum_conserve
    
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
    declare_dict = {'S':'double complex S%d(3)',
                    'F':'double complex F%d(6)',
                    'V':'double complex V%d(6)',
                    'T':'double complex T%s(18)'}
    
    def define_header(self, name=None):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        
        if not name:
            name = self.namestring
             
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
                                                          {'id': self.offshell})
            call_arg = '%(args)s, COUP, M%(id)d, W%(id)d, %(spin)s%(id)d' % \
                    {'args': ', '.join(CallList), 
                     'spin': self.particles[self.offshell -1],
                     'id': self.offshell}
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
                str_out += '%s%d(%d)= ' % (offshelltype, self.offshell, \
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
            sign = ''
            if self.offshell == index and type in ['V','S']:
                sign = '-'
                            
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
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'Vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                string = re.sub('\((?P<num>[+-]*[0-9])(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
                string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>.*(0d0,1d0)', string)
                OutString = OutString + string + '\n'
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
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
                                                                
        Outstring = 'call '+self.namestring+'('+','.join(calls)+',COUP,M%s,W%s,%s%s)\n' \
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
                             self.namestring,self.namestring[0:-1]+'%s' %(elem))
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
            
        name = combine_name(self.abstractname, lor_names, offshell)

                 
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
        addon = ''
        if 'C' in self.namestring:
            short_name, addon = name.split('C',1)
            if addon.split('_')[0].isdigit():
                addon = 'C' +self.namestring.split('C',1)[1]
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
                          {'spin': self.particles[offshell -1],
                           'id': self.offshell}
            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d, %(LAST)s' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'id': self.offshell,
                     'LAST': '%s'}

        # make the first call
        line = " CALL %s%s("+call_arg+")\n"
        text += '\n\n' + line % (self.namestring, '', 1, main)
        
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
    
def get_routine_name(name,outgoing):
    """ build the name of the aloha function """
    
    return '%s_%s' % (name, outgoing) 

def combine_name(name, other_names, outgoing):
    """ build the name for combined aloha function """

    # Two possible scheme FFV1C1_2_X or FFV1__FFV2C1_X
    # If they are all in FFVX scheme then use the first
    p=re.compile('^(?P<type>[FSVT]+)(?P<id>\d+)')
    routine = ''
    if p.search(name):
        base, id = p.search(name).groups()
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
            name = self.namestring
            
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
               'out': self.particles[OffShellParticle],
               'number': OffShellParticle + 1 
               }

        h_string = str_out + ";\n\n"
        cc_string = str_out + "{\n"

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
                str_out += '%s%d[%d]= ' % (offshelltype, self.offshell,
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
            sign = ''
            if self.offshell == index and type in ['V', 'S']:
                sign = '-'
                   
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
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + ';\n'
        else:
            OffShellParticle = self.particles[self.offshell-1]+'%s'%(self.offshell)
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
        Outstring = self.namestring+'('+','.join(calls)+',COUP,M%s,W%s,%s%s);\n' \
                         %(number,number,self.particles[self.offshell-1],number)
        return Outstring
    
    def define_foot(self):
        """Return the end of the function definition"""

        return '}\n\n' 

    def write_h(self, header, compiler_cmd=True):
        """Return the full contents of the .h file"""

        h_string = ''
        if compiler_cmd:
            h_string = '#ifndef '+ self.namestring + '_guard\n'
            h_string += '#define ' + self.namestring + '_guard\n'
            h_string += '#include <complex>\n'
            h_string += 'using namespace std;\n\n'

        h_header = header['h_header']

        h_string += h_header

        for elem in self.symmetries: 
            symmetryhead = h_header.replace( \
                             self.namestring,self.namestring[0:-1]+'%s' %(elem))
            h_string += symmetryhead

        if compiler_cmd:
            h_string += '#endif\n\n'

        return h_string

    def write_combined_h(self, lor_names, offshell=None, compiler_cmd=True):
        """Return the content of the .h file linked to multiple lorentz call."""
        
        name = combine_name(self.abstractname, lor_names, offshell)
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
            cc_string = '#include \"%s.h\"\n\n' % self.namestring
        cc_header = header['cc_header']
        cc_string += cc_header
        cc_string += self.define_momenta()
        cc_string += self.define_expression()
        cc_string += self.define_foot()

        for elem in self.symmetries: 
            symmetryhead = cc_header.replace( \
                             self.namestring,self.namestring[0:-1]+'%s' %(elem))
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
            
        name = combine_name(self.abstractname, lor_names, offshell)

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
        if 'C' in self.namestring:
            short_name, addon = name.split('C',1)
            if addon.split('_')[0].isdigit():
                addon = 'C' +self.namestring.split('C',1)[1]
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
        text += '\n\n' + line % (self.namestring, '', 1, main)
        
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
        
        name = combine_name(self.abstractname, lor_names, offshell)
        
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
                                                               self.offshell)
    
        
    def change_var_format(self, name): 
        """Formatting the variable name to Python format
        start to count at zero"""
        
        def shift_indices(match):
            """shift the indices for non impulsion object"""
            if match.group('var').startswith('P'):
                shift = 0
            else: 
                shift = -1
            
            return '%s[%s]' % (match.group('var'), \
                                                int(match.group('num')) + shift)
            
        
        name = re.sub('(?P<var>\w*)_(?P<num>\d+)$', shift_indices , name)
        return name
    
    def define_expression(self):
        """Define the functions in a 100% way """
        
        OutString = ''
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'vertex = COUP*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + '\n'
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            numerator = self.obj.numerator
            denominator = self.obj.denominator
            for ind in denominator.listindices():
                denom = self.write_obj(denominator.get_rep(ind))
            string = 'denom =' + '1.0/(' + denom + ')'
            string = string.replace('+-', '-')
            OutString += string + '\n'
            for ind in numerator.listindices():
                string = '%s[%d]= COUP*denom*' % (self.outname, self.pass_to_HELAS(ind))
                string += self.write_obj(numerator.get_rep(ind))
                string = string.replace('+-', '-')
                OutString += string + '\n' 
        return OutString 
    
    def define_foot(self):
        if not self.offshell:
            return 'return vertex\n\n'
        else:
            return 'return %s\n\n' % (self.outname)
            
    
    def define_header(self, name=None):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.namestring
           
        Momenta = self.collected['momenta']
        OverM = self.collected['om']
        
        CallList = self.calllist['CallList']

        str_out = ''
        # define the type of function and argument        
        if not self.offshell:
            str_out += 'def %(name)s(%(args)s):\n' % \
                {'name': name,
                 'args': ','.join(CallList+ ['COUP']) }
        else:
            str_out += 'def %(name)s(%(args)s, COUP, M%(id)d, W%(id)d):\n' % \
                {'name': name,
                 'args': ', '.join(CallList), 
                     'id': self.offshell}            
        return str_out     

    def make_declaration_list(self):
        """ make the list of declaration nedded by the header """
        return []

    def define_momenta(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
            
        # Definition of the Momenta
        momenta = self.collected['momenta']
        overm = self.collected['om']
        momentum_conservation = self.calllist['Momentum']
        
        str_out = ''
        
        # Definition of the output
        if self.offshell:
            offshelltype = self.particles[self.offshell -1]
            offshell_size = self.type_to_size[offshelltype]            
            str_out += '%s = wavefunctions.WaveFunction(size=%s)\n' % \
                                                (self.outname, offshell_size)
        # Conservation of Energy Impulsion
        if self.offshell:     
            #Implement the conservation of Energy Impulsion
            for i in range(-2,0):
                str_out += '%s[%d] = ' % (self.outname, offshell_size + i)
                
                pat=re.compile(r'^[-+]?(?P<spin>\w)')
                for elem in momentum_conservation:
                    spin = pat.search(elem).group('spin') 
                    str_out += '%s[%d]' % (elem, self.type_to_size[spin] + i)  
                str_out += '\n'  
                    
        # Momentum
        for mom in momenta:
            #Mom is in format PX with X the number of the particle
            index = int(mom[1:])
            type = self.particles[index - 1]
            energy_pos = self.type_to_size[type] -2
            sign = ''
            if self.offshell == index and type in ['V','S']:
                sign = '-'

            str_out += '%s = [%scomplex(%s%d[%d]).real, \\\n' % (mom, sign, type, index, energy_pos)
            str_out += '        %s complex(%s%d[%d]).real, \\\n' % ( sign, type, index, energy_pos + 1)
            str_out += '        %s complex(%s%d[%d]).imag, \\\n' % ( sign, type, index, energy_pos + 1)
            str_out += '        %s complex(%s%d[%d]).imag]\n' % ( sign, type, index, energy_pos) 
                   
        # Definition for the One Over Mass**2 terms
        for elem in overm:
            #Mom is in format OMX with X the number of the particle
            index = int(elem[2:])
            str_out += 'OM%d = 0.0\n' % (index)
            str_out += 'if (M%d): OM%d' % (index, index) + '=1.0/M%d**2\n' % (index) 
        
        # Returning result
        return str_out    

    def define_symmetry(self, new_nb):
        number = self.offshell 
        calls = self.calllist['CallList']
        Outstring = 'return '+self.namestring+'('+','.join(calls)+',COUP,M%s,W%s)\n' \
                         %(number,number)
        return Outstring        

    def write(self,mode='self'):
                         
        # write head - momenta - body - foot
        if mode == 'mg5':
            text = 'import aloha.template_files.wavefunctions as wavefunctions\n'
        else:
            text = 'import wavefunctions\n'
        text += self.define_header()
        content = self.define_momenta()
        content += self.define_expression()
        content += self.define_foot()
        
        # correct identation
        text += '    ' +content.replace('\n','\n    ')
        
        for elem in self.symmetries:
            text +='\n' + self.define_header().replace( \
                             self.namestring,self.namestring[0:-1]+'%s' %(elem))
            text += '    ' +self.define_symmetry(elem)
            
            
        if self.out_path:
            ff = open(self.out_path,'w').write(text)

        return text + '\n'

    def write_combined(self, lor_names, mode='self', offshell=None):
        """Write routine for combine ALOHA call (more than one coupling)"""
        
        # Set some usefull command
        if offshell is None:
            sym = 1
            offshell = self.offshell  
        else:
            sym = None
        name = combine_name(self.abstractname, lor_names, offshell)

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
        addon = ''
        if 'C' in self.namestring:
            short_name, addon = name.split('C',1)
            if addon.split('_')[0].isdigit():
                addon = 'C' +self.namestring.split('C',1)[1]
            else:
                addon = '_%s' % self.offshell
        else:
            addon = '_%s' % self.offshell

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
                           'id': self.offshell}
            call_arg = '%(args)s, %(COUP)s, M%(id)d, W%(id)d' % \
                    {'args': ', '.join(self.calllist['CallList']), 
                     'COUP':'COUP%d',
                     'id': self.offshell}

        # make the first call
        line = "    %s = %s%s("+call_arg+")\n"
        text += '\n\n' + line % (main, self.namestring, '', 1)
        
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
        
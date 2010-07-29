try:
    import madgraph.iolibs.file_writers as Writer 
except:
    import aloha.writer as Writer
    
import os
import re 
from numbers import Number

class WriteHelas: 
    """ Generic writing functions """ 
    
    power_symbol = '**'
    change_var_format = str
    change_number_format = str
    extension = ''
    type_to_variable = {2:'f',3:'v',5:'t',1:'s'}
    type_to_size = {'s':3, 't':18, 'v':6, 'f':6}
    #type_to_pos = {'s':2, 't':17, 'v':5, 'f':5}
    
    def __init__(self, abstracthelas, dirpath):

        self.obj = abstracthelas.expr
        helasname = get_helas_name(abstracthelas.name, abstracthelas.outgoing)
        self.out_path = os.path.join(dirpath, helasname + self.extension)
        self.dir_out = dirpath
        self.particles =  [self.type_to_variable[spin] for spin in \
                          abstracthelas.spins]
        self.namestring = helasname
        self.comment = abstracthelas.infostr
        self.offshell = abstracthelas.outgoing 
        self.symmetries = abstracthelas.symmetries

        #prepare the necessary object
        self.collect_variables() # Look for the different variables
        self.make_all_lists()   # Compute the expression for the call ordering
                                 #the definition of objects,...

    def collect_variables(self):
        """Collects Momenta,Mass,Width into lists"""
         
        MomentaList = set()
        MassList = set()
        WidthList = set()
        OverMList = set()
        for elem in self.obj.tag:
            if elem.startswith('P'):
                MomentaList.add(elem)
            elif elem.startswith('M'):
                MassList.add(elem)
            elif elem.startswith('W'):
                WidthList.add(elem)
            elif elem.startswith('O'):
                OverMList.add(elem) 

        MomentaList = list(MomentaList)
        MassList = list(MassList)
        WidthList = list(WidthList)
        OverMList = list(OverMList)
        
        self.collected = {'momenta':MomentaList, 'width':WidthList, \
                          'mass':MassList, 'om':OverMList}
        
        return self.collected

    def define_header(self): 
        """ Prototype for language specific header""" 
        pass

    def define_content(self): 
        """Prototype for language specific body""" 
        pass
    
    def define_foote (self):
        """Prototype for language specific footer"""
        pass
    
    def write_indices_part(self, indices, obj): 
        """Routine for making a string out of indice objects"""
        
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

    
    def make_call_list(self):
        """find the way to write the call of the functions"""

        # particle type counter
        nb_type = {'s':0, 'f':0, 'v':0, 't':0}        
        call_arg = [] #incoming argument of the routine
        
        # update the type counter + make call_arg for amplitude
        for index,spin in enumerate(self.particles):
            nb_type[spin] += 1
            call_arg.append('%s%d' % (spin, index +1))
            
        # reorder call_arg if not amplitude
        if self.offshell:
            part_pos = self.offshell -1 
            out_type = self.particles[part_pos]
            
            #order is FVST #look at the border of the cycling move
            # start/stop are the index of the group of spin where to perform
            #cycling ordering.
            if out_type == 'f':
                start = 0
                stop = nb_type['f']
            elif out_type == 'v':
                start = nb_type['f']
                stop = start + nb_type['v']
            elif out_type == 's':
                start = nb_type['f'] + nb_type['v']
                stop = start + nb_type['s']
            elif out_type == 't':
                start = nb_type['f'] + nb_type['v']+ nb_type['s']
                stop = start + nb_type['t']
            else:
                raise NotImplemented, 'Only type FVST are supported' 
            
            #reorganize the order and suppress the output from this part
            call_arg = self.new_order(call_arg, part_pos, start, stop)
        
        return call_arg
            
    @ staticmethod
    def new_order(call_list, remove, start, stop):
        """ create the new order for the calling using cycling order"""
        
        assert(start <= remove <= stop <= len(call_list))
        
        new_list= call_list[:start]
        for i in range(remove+1, stop):
            new_list.append(call_list[i])
        for i in range(start, remove):
            new_list.append(call_list[i])
        new_list += call_list[stop:]
        
        return new_list
        
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
        if not self.offshell % 2 and self.particles[self.offshell -1] == 'f': 
            global_sign = 1
        else:
            global_sign = -1
        
        
        for index, spin in enumerate(self.particles): 
            assert(spin in ['s','f','v','t'])  
      
            #compute the sign
            if spin != 'f':
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
 
    
        
class HelasWriterForFortran(WriteHelas): 
    """routines for writing out Fortran"""

    extension = '.f'
    declare_dict = {'s':'double complex S%d(3)',
                    'f':'double complex F%d(6)',
                    'v':'double complex V%d(6)',
                    't':'double complex T%s(18)'}
    
    def define_header(self):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
            
        Momenta = self.collected['momenta']
        Width = self.collected['width']
        Mass = self.collected['mass']
        OverM = self.collected['om']
        
        CallList = self.calllist['CallList']
        DeclareList = self.calllist['DeclareList']
        DeclareList.append('double complex C')
        
        local_declare = []
        OffShell = self.offshell
        OffShellParticle = OffShell -1 
        
        
        # define the type of function and argument
        if not OffShell:
            str_out = 'subroutine %(name)s(%(args)s,vertex)\n' % \
               {'name': self.namestring,
                'args': ','.join(CallList+ ['C'] + Mass + Width) } 
            local_declare.append('double complex vertex\n') 
        else: 
            local_declare.append('double complex denom\n')
            str_out = 'subroutine %(name)s(%(args)s, %(out)s%(number)d)\n' % \
               {'name': self.namestring,
                'args': ','.join(CallList+ ['C'] + Mass + Width), 
                'out': self.particles[OffShellParticle],
                'number': OffShellParticle + 1 
                }
                                 
        # Forcing implicit None
        str_out += 'implicit none \n'
        
        # Declare all the variable
        for elem in DeclareList + local_declare:
            str_out += elem + '\n'
        if len(Mass + Width) > 0:
            str_out += 'double precision ' + ','.join(Mass + Width) + '\n'
        if len(OverM) > 0: 
            str_out += 'double complex ' + ','.join(OverM) + '\n'
        if len(Momenta) > 0:
            str_out += 'double precision ' + '(0:3),'.join(Momenta) + '(0:3)\n'

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
            if self.offshell == index and (type == 'v' or type == 's'):
                sign = '-'
                
            str_out += '%s(0) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)
            str_out += '%s(1) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(2) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(3) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)            
            
                   
        # Definition for the One Over Mass**2 terms
        for elem in overm:
            #Mom is in format OMX with X the number of the particle
            index = int(elem[2:])
            str_out += 'om%d = 0d0\n' % (index)
            str_out += 'if (m%d .ne. 0d0) om%d' % (index, index) + '=1d0/dcmplx(m%d**2,-w%d*m%d)\n' % (index, index, index) 
        
        # Returning result
        return str_out
        
        
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
        if '_' in name:
            name = name.replace('_', '(', 1) + ')'
        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
        return name
    
    def change_number_format(self, number):
        """Formating the number"""
        if isinstance(number, complex):
            out = '(%.9fd0, %.9fd0)' % (number.real, number.imag)
        else:
            out = '%.9f' % number
        return out
    
    def define_expression(self):
        OutString = ''
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'Vertex = C*' + self.write_obj(self.obj.get_rep(ind))
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
            counter = 1
            for ind in numerator.listindices():
                string = '%s(%d)= C*denom*' % (OffShellParticle, counter)
                string += self.write_obj(numerator.get_rep(ind))
                string = string.replace('\+-', '-')
                string = re.sub('\((?P<num>[+-][0-9])\+(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
                string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>*(0d0,1d0)', string)
                OutString = OutString + string + '\n' 
                counter += 1
        return OutString 
    
    def define_symmetry(self):
        calls = self.calllist['CallList']
        number = self.offshell 
        Outstring = 'call '+self.namestring+'('+','.join(calls)+',C,M%s,W%s,%s%s)' \
                         %(number,number,self.particles[self.offshell-1],number)
        return Outstring
    
    def define_foot(self):
        return 'end' 

    def write(self):
                
        writer = Writer.FortranWriter(self.out_path)
        writer.downcase = False 
        commentstring = 'This File is Automatically generated by ALOHA \n'
        commentstring += 'The process calculated in this file is: \n'
        commentstring += self.comment + '\n'
        writer.write_comments(commentstring)
         
        # write head - momenta - body - foot
        writer.writelines(self.define_header())
        writer.writelines(self.define_momenta())
        writer.writelines(self.define_expression())
        writer.writelines(self.define_foot())
        
        for elem in self.symmetries: 
            symmetryhead = self.define_header().replace( \
                             self.namestring,self.namestring[0:-1]+'%s' %(elem))
            symmetrybody = self.define_symmetry()
            writer.write_comments('\n%s\n' % ('#'*65))
            writer.writelines(symmetryhead)
            writer.writelines(symmetrybody)
            writer.writelines(self.define_foot())
        
def get_helas_name(name,outgoing):
    """ build the name of the helas function """
    
    return '%s_%s' % (name, outgoing) 

class HelasWriterForCPP(WriteHelas): 
    """routines for writing out Fortran"""
    
    extension = '.cc'
    
    def __init__(self, abstracthelas, dirpath):

        WriteHelas.__init__(self, abstracthelas, dirpath)
        
        helasname = get_helas_name(abstracthelas.name, abstracthelas.outgoing)
        self.out_head = os.path.join(dirpath,helasname + '.h')
        
    def make_call_lists(self):
        """ """
        
        MomentumConserve = []
        DeclareDict = {'F':'complex<double > F', 'V':'complex<double>  V', \
                                'S':'complex<double>  S', 'T':'complex<double> T'}
        Counter = 0
        ScalarNumber = 0
        FermionNumber = 0
        VectorNumber = 0
        TensorNumber = 0
        VectorList = []
        TensorList = []
        ScalarList = []
        FermionList = []
        for index, elem in enumerate(self.particles):
            
            # First define the size of the associate Object 
            if elem == 'S':
                ScalarList.append('%s%d[3]' % (DeclareDict['S'], index + 1))
                ScalarNumber += 1
            elif elem == 'T':
                TensorList.append('%s%d[18]' % (DeclareDict['T'], index + 1))
                TensorNumber +=1
            elif elem == 'F':
                FermionList.append('%s%d[6]' % (DeclareDict[elem[0]], index + 1)) 
                FermionNumber +=1 
            else: 
                VectorList.append('%s%d[6]' % (DeclareDict[elem[0]], index + 1)) 
                VectorNumber +=1 
            # Define Momentum Conservation
            if elem in ['V', 'S', 'T']:
                MomentumConserve.append('-%s%d' % (elem[0], index + 1))
            elif elem == 'F' and Counter %2 == 0:
                MomentumConserve.append('-F%d' % (index + 1))
                Counter += 1 
            else: 
                MomentumConserve.append('+F%d' % (index + 1))
                Counter += 1
        # reorder call list 
        if self.offshell:
            OffShellParticle = self.offshell - 1 
            PermList = []
            if OffShellParticle < FermionNumber:
                FermionList.pop(OffShellParticle)
            elif OffShellParticle < (FermionNumber + VectorNumber):
                for i in range(len(VectorList)):
                    Shift = FermionNumber + VectorNumber - 1 - OffShellParticle
                    PermList.append(i - Shift) 
                VectorList = [VectorList[i] for i in PermList] 
                VectorList.pop()
            elif OffShellParticle < (FermionNumber + VectorNumber + ScalarNumber):
                for i in range(len(ScalarList)):
                    Shift = FermionNumber + VectorNumber + ScalarNumber - 1 - OffShellParticle
                    PermList.append(i - Shift) 
                ScalarList = [ScalarList[i] for i in PermList] 
                ScalarList.pop()
            elif OffShellParticle < (FermionNumber + VectorNumber + ScalarNumber):
                for i in range(len(VectorList)):
                    Shift = len(self.particles) - 1 - OffShellParticle
                    PermList.append(i - Shift) 
                TensorList = [TensorList[i] for i in PermList] 
                TensorList.pop()
        DeclareList = FermionList + VectorList + ScalarList + TensorList
        DeclareList.append('complex<double> C')
        
        self.calllist = {'DeclareList':DeclareList,'Momentum':MomentumConserve}
        return self.calllist
    
    def define_header(self):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
            - momentum conservation
            -definition of the impulsion"""
        AddedDeclare = [] 
        Momenta = self.collected['momenta']
        Width = self.collected['width']
        Mass = self.collected['mass']
        OverM = self.collected['om']
        for index,elem in enumerate(Mass):
            Mass[index] = 'double '+elem 
        for index,elem in enumerate(Width): 
            Width[index] = 'double '+elem
            
        DeclareList = self.calllist['DeclareList']
        OffShellParticle = self.offshell-1
        OffShell = self.offshell
        MomentumConserve = self.calllist['Momentum']
        headerstring = '#ifndef '+self.namestring+'_guard \n #define '+self.namestring+'_guard\n'
        string = '#include <complex>\n using namespace std;\n'
        headerstring += '#include <complex>\n using namespace std;\n'
        # define the type of function and argument
        if not OffShell:
            headerstring += 'void ' + self.namestring + '('+','.join(DeclareList) + ',complex<double> &vertex);\n'
            string += 'void ' + self.namestring + '(' + ','.join(DeclareList + Mass + Width) + ',complex<double> &vertex){\n'
        else: 
            AddedDeclare.append('complex<double> denom')
            temp = self.particles[self.offshell-1]+'%s' %(self.offshell)
            if temp[0] in ['V','F']: 
                entries = 6
            elif temp[0] == 'S':
                entries = 3
            elif temp[0] == 'T':
                entries = 18
            OutputParticle = 'complex<double> '+temp+'[%s]'%(entries)
            headerstring += 'void ' + self.namestring + '(' + ','.join(DeclareList + Mass+Width) \
                    + ',' + OutputParticle+');\n' 
            string += 'void ' + self.namestring + '(' + ','.join(DeclareList + Mass+Width) \
                    + ',' + OutputParticle+'){\n' 
        headerstring += '#endif'
        # Declare all the variable
        for elem in AddedDeclare:
            string += elem + ';\n'
        if len(OverM) > 0: 
            string += 'complex<double> ' + ','.join(OverM) + ';\n'
        if len(Momenta) > 0:
            string += 'double ' + '[4],'.join(Momenta) + '[4];\n'
        if OffShell: 
            NegSign = MomentumConserve.pop(OffShellParticle) 
            NegSignBool = re.match('-F', NegSign) 
            if NegSignBool:
                NegString = '(' 
            else:
                NegString = '-('
                
            #Implement the conservation of momenta             
            if re.match('S', self.particles[OffShellParticle]):
                firstnumber = 1
            elif re.match('T', self.particles[OffShellParticle]):
                firstnumber = 16
            else: 
                firstnumber = 4
            MomString = ''
            for i in range(2): 
                MomString += '%s%d[%d]=' % (self.particles[OffShellParticle], OffShellParticle + 1,firstnumber+i) + NegString 
                for elem in MomentumConserve:
                    if re.match('-S', elem):
                        MomString = MomString + elem + '[%d]'%(1+i)
                    elif re.match('-T', elem):
                        MomString = MomString + elem + '[%d]'%(16+i) 
                    else:
                        MomString = MomString + elem + '[%d]'%(4+i)
                MomString = MomString + ');\n'
            string += MomString
            
        # Definition of the Momenta
        for mom in Momenta:
             
            index = int(mom[-1])
            
            type = self.particles[index-1]
            energy_pos=self.type_to_size[type]
            sign = ''
            if OffShellParticle == index -1 and type !='S':
                sign='-'
                
            string += '%s[0] = %s %s%d[%d].real();\n' % (mom, sign, type, index, energy_pos)
            string += '%s[1] = %s %s%d[%d].real();\n' % (mom, sign, type, index, energy_pos+1)
            string += '%s[2] = %s %s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos+1)
            string += '%s[3] = %s %s%d[%d].imag();\n' % (mom, sign, type, index, energy_pos)            
            
                   
        # Definition for the One Over Mass**2 terms
        for elem in OverM:
            index = int(elem[-1])
            string = string + 'OM%d = 0.;\n' % (index)
            string = string + 'if (M%d != 0.){ OM%d' % (index, index) + '=1./complex<double> (pow(M%d,2),-W%d*M%d);}\n' % (index, index, index) 
        
        # Returning result
        return {'headerfile':headerstring,'head':string}
        
        
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
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
        """Formating the number"""
        if isinstance(number, complex):
            out = 'complex<double> (%.9f, %.9f)' % (number.real, number.imag)
        else:
            out= '%.9f' % number
        return out
    
    
    def define_expression(self):
        OutString = '' 
        if not self.offshell:
            for ind in self.obj.listindices():
                string = 'vertex = C*' + self.write_obj(self.obj.get_rep(ind))
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
            counter = 0
            for ind in numerator.listindices():
                string = '%s[%d]= C*denom*' % (OffShellParticle, counter)
                string += self.write_obj(numerator.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + ';\n' 
                counter += 1
        OutString = re.sub('(?P<variable>[A-Za-z]+[0-9]\[*[0-9]*\]*)\*\*(?P<num>[0-9])','pow(\g<variable>,\g<num>)',OutString)
        return OutString 

    def define_foot(self):
        return '}' 

    def write(self):
        
        #prepare the necessary object
        self.collect_variables() # Look for the different variables
        self.make_all_lists()   # Compute the expression for the call ordering
                                #the definition of objects,...

        
        
        hWriter = Writer.CPPWriter(self.out_head)
        ccWriter = Writer.CPPWriter(self.out_path)
#        commentstring = 'c   This File Automatically generated by MadGraph 5/FeynRules HELAS writer \n'
#        commentstring += 'c   The process calculated in this file is: \n'
#        commentstring = commentstring + 'c    ' + self.comment + '\n'
        headerfile = self.define_header()['headerfile'] 
        head = self.define_header()['head']
        body = self.define_expression()
        foot = self.define_foot()
        out = head + body + foot
        for line in headerfile: 
                hWriter.write(line)
        for line in out:
                ccWriter.write(line)

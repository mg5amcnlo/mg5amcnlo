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
    type_to_variable = {2:'F',3:'V',5:'T',1:'S'}
    type_to_pos = {'S':2, 'T':17, 'V':5, 'F':5}
    
    def __init__(self, abstracthelas, dirpath):

        self.obj = abstracthelas.expr
        helasname = get_helas_name(abstracthelas.name, abstracthelas.outgoing)
        self.out_path = os.path.join(dirpath, helasname + self.extension)
        self.dir_out = dirpath
        self.particles = abstracthelas.spins
        self.namestring = helasname
        self.comment = abstracthelas.infostr
        self.offshell = abstracthelas.outgoing 
        self.symmetries = abstracthelas.symmetries
         
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
        
class HelasWriterForFortran(WriteHelas): 
    """routines for writing out Fortran"""

    extension = '.f'

    def make_call_lists(self):
        """ """
        
        CallList = []
        DeclareList = ['double complex C']
        MomentumConserve = []
        DeclareDict = {'F':'double complex f', 'V':'double complex V', \
                                'S':'double complex s', 'T':'double complex T'}
        TypeToVariable = {2:'F',3:'V',5:'T',1:'S'}
        FermionNumber = 0
        VectorNumber = 0
        ScalarNumber = 0
        TensorNumber = 0
        Counter = 0
        OffShell = self.offshell 
        FermiList = []
        VectorList = []
        ScalarList = []
        TensorList = [] 
        for index, elem in enumerate(self.particles):
            
            # First define the size of the associate Object 
            if elem == 1:
                DeclareList.append('%s%d(3)' % (DeclareDict['S'], index + 1))
                ScalarList.append('%s%d' % ('S', index + 1))
                ScalarNumber += 1 
            elif elem == 5:
                DeclareList.append('%s%d(18)' % (DeclareDict['T'], index + 1))
                TensorList.append('%s%d' % ('T', index + 1))
                TensorNumber += 1
            elif elem == 3:
                DeclareList.append('%s%d(6)' % (DeclareDict['V'], index + 1))  
                VectorList.append('%s%d' % ('V', index + 1))
                VectorNumber += 1 
            elif elem == 2:
                DeclareList.append('%s%d(6)' % (DeclareDict['F'], index + 1))  
                FermiList.append('%s%d' % ('F', index + 1))  
                FermionNumber += 1 
            # Define the Calllist
            if index != (OffShell-1):
                CallList.append('%s%d' % (TypeToVariable[elem], index + 1))
                
            # Define Momentum Conservation
            if TypeToVariable[elem] in ['V', 'S', 'T']:
                MomentumConserve.append('-%s%d' % (TypeToVariable[elem], index + 1))
            elif TypeToVariable[elem] == 'F' and Counter % 2 == 0:
                MomentumConserve.append('-F%d' % (index + 1))
                Counter += 1 
            else: 
                MomentumConserve.append('+F%d' % (index + 1))
                Counter += 1
                    
        # Reorder calllist cyclically. 
        if OffShell:
            OffShellParticle = OffShell -1 
            PermList = []
            if OffShellParticle < FermionNumber:
                for i in range(FermionNumber):
                    PermList.append(OffShellParticle + i - FermionNumber + 1) 
                FermiList = [FermiList[i] for i in PermList] 
                FermiList.pop()
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
            CallList = FermiList + VectorList + ScalarList + TensorList
            
        self.calllist =  {'CallList':CallList,'DeclareList':DeclareList, \
                           'Momentum':MomentumConserve}
    
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
                'out': self.type_to_variable[self.particles[OffShellParticle]],
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
            NegSign = momentum_conservation.pop(self.offshell -1) 
            if NegSign.startswith('-F'):
                NegString = '(' 
            else:
                NegString = '-('

            offshelltype = self.type_to_variable[self.particles[self.offshell -1]]
            #Implement the conservation of Energy Impulsion 
            if 'S' == offshelltype:
                str_out += '%s%d(2)=%s' % (offshelltype, self.offshell, NegString) 
            elif 'T' == offshelltype:
                str_out += '%s%d(17)=%s' % (offshelltype, self.offshell, NegString) 
            else: 
                str_out += '%s%d(5)=%s' % (offshelltype, self.offshell, NegString) 
            for elem in momentum_conservation:
                if elem.startswith('-S'):
                    str_out += elem + '(2)' 
                elif elem.startswith('-T'):
                    str_out += elem + '(17)' 
                else:
                    str_out += elem + '(5)'
                     
            str_out += ')\n'
            if 'S' == offshelltype:
                str_out += '%s%d(3)=%s' % (offshelltype, self.offshell, NegString) 
            elif 'T' == offshelltype:
                str_out += '%s%d(18)=%s' % (offshelltype, self.offshell, NegString) 
            else: 
                str_out += '%s%d(6)=%s' % (offshelltype, self.offshell, NegString)
                 
            for elem in momentum_conservation:
                if elem.startswith('-S'):
                    str_out += elem + '(3)' 
                elif elem.startswith('-T'):
                    str_out += elem + '(18)' 
                else:
                    str_out += elem + '(6)'
            str_out +=  ')\n' 

        
        # Momentum
        for mom in momenta:
            index = int(mom[-1])
            
            type = self.type_to_variable[self.particles[index - 1]]
            energy_pos = self.type_to_pos[type]
            sign = ''
            if self.offshell == index and (type == 'V' or type == 'S'):
                sign = '-'
                
            str_out += '%s(0) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)
            str_out += '%s(1) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(2) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos + 1)
            str_out += '%s(3) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)            
            
                   
        # Definition for the One Over Mass**2 terms
        for elem in overm:
            index = int(elem[-1])
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
            OffShellParticle = '%s%d' % (self.type_to_variable[self.particles[self.offshell-1]], self.offshell)
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
        Outstring = 'call '+self.namestring+'('+','.join(calls)+',C,M%s,W%s,%s%s)'%(number,number,self.type_to_variable[self.particles[self.offshell-1]],number)
        return Outstring
    
    def define_foot(self):
        return 'end' 

    def write(self):
        
        #prepare the necessary object
        self.collect_variables() # Look for the different variables
        self.make_call_lists()   # Compute the expression for the call ordering
                                 #the definition of objects,...
        
        
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

        self.obj = abstracthelas.expr
        helasname = get_helas_name(abstracthelas.name, abstracthelas.outgoing)
        self.out_path = os.path.join(dirpath, helasname + self.extension)
        self.out_head = os.path.join(dirpath,helasname + '.h')
        self.dir_out = dirpath
        self.particles =[]
        for elem in abstracthelas.spins: 
                self.particles.append(self.type_to_variable[elem])
        self.namestring = helasname
        self.comment = abstracthelas.infostr
        self.offshell = abstracthelas.outgoing 
        self.symmetries = abstracthelas.symmetries 
        
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
                range = 6
            elif temp[0] == 'S':
                range = 3
            elif temp[0] == 'T':
                range = 18
            OutputParticle = 'complex<double> '+temp+'[%s]'%(range)
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
            energy_pos=self.type_to_pos[type]
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

try:
    import madgraph.iolibs.export_v4 as FortranWriter 
except:
    import aloha.Writter as FortranWriter
import aloha.helasamp_object as Helas
import aloha.helasamp_lib as Helas_Lib
import re 
from numbers import Number
class WriteHelas: 
    """ Generic writing functions """ 
    
    power_symbol = '**'
    change_var_format = str
    change_number_format =str
    
    def __init__(self, object, particlelist, out_path, comment):
        self.obj = object
        self.out = open(out_path + '.f', 'w')
        self.particles = particlelist 
        self.namestring = out_path
        self.comment = comment
        
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
        return {'momenta':MomentaList, 'width':WidthList, 'mass':MassList, 'om':OverMList}

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

        if isinstance(obj, Number):
            return self.change_number_format(obj)
        elif obj.vartype == 2 : #isinstance(obj, Helas_Lib.MultVariable):
            return self.write_obj_Mult(obj)
        elif not obj.vartype: #isinstance(obj, Helas_Lib.Variable):
            return self.write_obj_Var(obj)
        elif obj.vartype == 1 : #isinstance(obj, Helas_Lib.AddVariable):
            return self.write_obj_Add(obj)
        elif obj.vartype == 5: #ConstantObject
            return self.change_number_format(obj.value)
        else: 
            print 'Warning unknow object',obj.vartype
            return str(obj)

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
        text = '(' 
        if obj.prefactor != 1:
            if obj.prefactor != -1:
                text = self.change_number_format(obj.prefactor) + '*' + text 
            else: 
                text = '-' + text
        return text + '+'.join(mult_list) + ')'

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

    def make_call_lists(self):
        """ """
        
        CallList = []
        DeclareList = ['double precision C']
        MomentumConserve = []
        DeclareDict = {'F':'double complex f', 'V':'double complex V', \
                                'S':'double complex s', 'T':'double complex T'}
        FermionNumber = 0
        VectorNumber =0
        ScalarNumber = 0
        TensorNumber = 0
        OnShell = 1 
        Counter = 0
        OffShellParticle = 999
        FermiList = []
        VectorList = []
        ScalarList = []
        TensorList =[] 
        for index, elem in enumerate(self.particles):
            
            # First define the size of the associate Object 
            if elem[0] == 'S':
                DeclareList.append('%s%d(3)' % (DeclareDict['S'], index + 1))
                ScalarList.append('%s%d' % ('S', index + 1))
                ScalarNumber += 1 
            elif elem[0] == 'T':
                DeclareList.append('%s%d(18)' % (DeclareDict['T'], index + 1))
                TensorList.append('%s%d' % ('T', index + 1))
                TensorNumber += 1
            elif elem[0] == 'V':
                DeclareList.append('%s%d(6)' % (DeclareDict[elem[0]], index + 1))  
                VectorList.append('%s%d' % ('V', index + 1))
                VectorNumber +=1 
            elif elem[0] == 'F':
                DeclareList.append('%s%d(6)' % (DeclareDict[elem[0]], index + 1))  
                FermiList.append('%s%d' % ('F', index + 1))  
                FermionNumber +=1 
            # Define the Calllist
            if elem[1]:
                CallList.append('%s%d' % (elem[0], index + 1))
            else: 
                OnShell = 0
                OffShellParticle = index
                
            # Define Momentum Conservation
            if elem[0] in ['V', 'S', 'T']:
                MomentumConserve.append('-%s%d' % (elem[0], index + 1))
            elif elem[0] == 'F' and Counter %2 == 0:
                MomentumConserve.append('-F%d' % (index + 1))
                Counter += 1 
            else: 
                MomentumConserve.append('+F%d' % (index + 1))
                Counter += 1
        # Reorder calllist cyclically. 
        if not OnShell:
            PermList = []
            if OffShellParticle< FermionNumber:
                for i in range(FermionNumber):
                    PermList.append(OffShellParticle+i-FermionNumber+1) 
                FermiList = [FermiList[i] for i in PermList] 
                FermiList.pop()
            elif OffShellParticle< (FermionNumber+VectorNumber):
                for i in range(len(VectorList)):
                    Shift = FermionNumber+VectorNumber-1-OffShellParticle
                    PermList.append(i-Shift) 
                VectorList = [VectorList[i] for i in PermList] 
                VectorList.pop()
            elif OffShellParticle< (FermionNumber+VectorNumber+ScalarNumber):
                for i in range(len(ScalarList)):
                    Shift = FermionNumber+VectorNumber+ScalarNumber-1-OffShellParticle
                    PermList.append(i-Shift) 
                ScalarList = [ScalarList[i] for i in PermList] 
                ScalarList.pop()
            elif OffShellParticle< (FermionNumber+VectorNumber+ScalarNumber):
                for i in range(len(VectorList)):
                    Shift = len(self.particles)-1-OffShellParticle
                    PermList.append(i-Shift) 
                TensorList = [TensorList[i] for i in PermList] 
                TensorList.pop()
            CallList = FermiList+VectorList+ScalarList+TensorList
        return {'CallList':CallList, 'OnShell':OnShell, 'DeclareList':DeclareList, \
                     'OffShell':OffShellParticle, 'Momentum':MomentumConserve}
    
    def define_header(self):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
            - momentum conservation
            -definition of the impulsion"""
            
        CollectedVariables = self.collect_variables()
        Momenta = CollectedVariables['momenta']
        Width = CollectedVariables['width']
        Mass = CollectedVariables['mass']
        OverM = CollectedVariables['om']
        
        listout = self.make_call_lists()
        CallList = listout['CallList']
        OnShell = listout['OnShell']
        DeclareList = listout['DeclareList']
        OffShellParticle = listout['OffShell']
        MomentumConserve = listout['Momentum']
        
        # define the type of function and argument
        if OnShell:
            string = 'subroutine ' + self.namestring + '(C,' + ','.join(CallList + Mass + Width) + ',vertex)\n'
            DeclareList.append('double complex vertex') 
        else: 
            DeclareList.append('double complex denom')
            string = 'subroutine ' + self.namestring + '(C,' + ','.join(CallList + Mass + Width) \
                    + ',' + self.particles[OffShellParticle][0] + '%d)\n' % (OffShellParticle + 1)
        
        # Forcing implicit None
        string += 'implicit none \n'
        
        # Declare all the variable
        for elem in DeclareList:
            string += elem + '\n'
        if len(Mass + Width) > 0:
            string += 'double precision ' + ','.join(Mass + Width) + '\n'
        if len(OverM) > 0: 
            string += 'double complex ' + ','.join(OverM) + '\n'
        if len(Momenta) > 0:
            string += 'double precision ' + '(0:3),'.join(Momenta) + '(0:3)\n'
        if not OnShell: 
            NegSign = MomentumConserve.pop(OffShellParticle) 
            NegSignBool = re.match('-F', NegSign) 
            if NegSignBool:
                NegString = '(' 
            else:
                NegString = '-('
                
            # Implement better routine here!!
            #Implement the conservation of Energy Impulsion 
            if re.match('S', self.particles[OffShellParticle][0]):
                MomString = '%s%d(2)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString 
            elif re.match('T', self.particles[OffShellParticle][0]):
                MomString = '%s%d(17)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString 
            else: 
                MomString = '%s%d(5)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString 
            for elem in MomentumConserve:
                if re.match('-S', elem):
                    MomString = MomString + elem + '(2)' 
                elif re.match('-T', elem):
                    MomString = MomString + elem + '(17)' 
                else:
                    MomString = MomString + elem + '(5)' 
            MomString = MomString + ')\n'
            if re.match('S', self.particles[OffShellParticle][0]):
                MomString = MomString + '%s%d(3)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString 
            elif re.match('T', self.particles[OffShellParticle][0]):
                MomString = MomString + '%s%d(18)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString 
            else: 
                MomString = MomString + '%s%d(6)=' % (self.particles[OffShellParticle][0], OffShellParticle + 1) + NegString
            for elem in MomentumConserve:
                if re.match('-S', elem):
                    MomString = MomString + elem + '(3)' 
                elif re.match('-T', elem):
                    MomString = MomString + elem + '(18)' 
                else:
                    MomString = MomString + elem + '(6)'
            MomString = MomString + ')\n' 
            string += MomString
            
        # Definition of the Momenta
        type_to_pos={'S':2,'T':17,'V':5,'F':5}
        for mom in Momenta:
             
            index = int(mom[-1])
            
            type = self.particles[index-1][0]
            energy_pos=type_to_pos[type]
            sign = ''
            if OffShellParticle == index -1 and type !='S':
                sign='-'
                
            string += '%s(0) = %s dble(%s%d(%d))\n' %  (mom, sign, type, index, energy_pos)
            string += '%s(1) = %s dble(%s%d(%d))\n' % (mom, sign, type, index, energy_pos+1)
            string += '%s(2) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos+1)
            string += '%s(3) = %s dimag(%s%d(%d))\n' % (mom, sign, type, index, energy_pos)            
            
                   
        # Definition for the One Over Mass**2 terms
        for elem in OverM:
            index = int(elem[-1])
            string = string + 'om%d = 0d0\n' % (index)
            string = string + 'if (m%d .ne. 0d0) om%d' % (index, index) + '=1d0/dcmplx(m%d**2,-w%d*m%d)\n' % (index, index, index) 
        
        # Returning result
        return string
        
        
    def change_var_format(self, name): 
        """Formatting the variable name to Fortran format"""
        
        if '_' in name:
            name = name.replace('_','(',1) +')'
        #name = re.sub('\_(?P<num>\d+)$', '(\g<num>)', name)
        return name
    
    def change_number_format(self, number):
        """Formating the number"""
        if isinstance(number, complex):
            out = '(%.9fd0, %.9fd0)' % (number.real, number.imag)
        else:
            out= '%.9f' % number
        return out
    
    
    def define_expression(self):
        OnShell = 1
        OutString = ''
        for index, elem in enumerate(self.particles):
            if elem[1]:
                OnShell = 0
                OffShellParticle = '%s%d' % (elem[0], index + 1)
        if OnShell:
            for ind in self.obj.listindices():
                string = 'Vertex = C*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                string = re.sub('\((?P<num>[+-]*[0-9])(?P<num2>[+-][0-9])[Jj]\)\.', '(\g<num>d0,\g<num2>d0)', string)
                string = re.sub('(?P<num>[0-9])[Jj]\.', '\g<num>.*(0d0,1d0)', string)
                OutString = OutString + string + '\n'
        else:
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

    def define_foot(self):
        return 'end' 

    def write(self):
        writer = FortranWriter.FortranWriter()
        writer.downcase = False 
        commentstring = 'c   This File Automatically generated by MadGraph 5/FeynRules HELAS writer \n'
        commentstring += 'c   The process calculated in this file is: \n'
        commentstring = commentstring + 'c    ' + self.comment + '\n' 
        head = self.define_header()
        body = self.define_expression()
        foot = self.define_foot()
        out = commentstring + head + body + foot
        for lines in out.split('\n'):
            writer.write_fortran_line(self.out, lines) 

if __name__ == '__main__':
    # Input as coming from FR!!!
    obj = Helas.Mass(1, 1) * Helas.P(1, 1)
#    test = Mass(1)+Mass(2)
#    test = test.simplify().expand()
    # Analysis
#    obj = obj.simplify() # -> Simplify sum    
    # expand 
#    print obj.simplify().expand()
    List = [('F', 1), ('V', 1), ('F', 1)]
    Write = HelasWriterForFortran(obj.expand(), List, 'test')
    Write.write()
#    for ind in test.listindices(): 
#        print Write.write_obj(test.get_rep(ind))
#        print test.get_rep(ind).prefactor
#    print Write.write_obj(test)
    print 'all is done'
#    print low_level
    #

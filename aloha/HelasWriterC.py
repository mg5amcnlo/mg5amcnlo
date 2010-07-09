import aloha.helasamp_object as Helas
import aloha.helasamp_lib as Helas_Lib
import re 
import os
from numbers import Number
from aloha.WriteHelas import * 

class HelasWriterForCpp(WriteHelas): 
    """routines for writing out Fortran"""

    def __init__(self, object, particlelist, out_path, comment):
        out_pathcc = out_path+'.cc'
        WriteHelas. __init__(self, object, particlelist, out_pathcc, comment)
        self.outheader = out_path + '.h'
        self.namestring = os.path.basename(out_path)
        self.out = out_pathcc
    def make_call_lists(self):
        """ """
        
        DeclareList = ['double C']
        MomentumConserve = []
        DeclareDict = {'F':'complex<double > F', 'V':'complex<double>  V', \
                                'S':'complex<double>  S', 'T':'complex<double> T'}
        OnShell = 1 
        Counter = 0
        OffShellParticle = 999
        
        for index, elem in enumerate(self.particles):
            
            # First define the size of the associate Object 
            if elem[0] == 'S':
                DeclareList.append('%s%d[3]' % (DeclareDict['S'], index + 1))
            elif elem[0] == 'T':
                DeclareList.append('%s%d[18]' % (DeclareDict['T'], index + 1))
            else:
                DeclareList.append('%s%d[6]' % (DeclareDict[elem[0]], index + 1)) 
           
            if not elem[1]:
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

        return {'OnShell':OnShell, 'DeclareList':DeclareList, \
                     'OffShell':OffShellParticle, 'Momentum':MomentumConserve}
    
    def define_header(self):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
            - momentum conservation
            -definition of the impulsion"""
        AddedDeclare = [] 
        CollectedVariables = self.collect_variables()
        Momenta = CollectedVariables['momenta']
        Width = CollectedVariables['width']
        Mass = CollectedVariables['mass']
        OverM = CollectedVariables['om']
        for index,elem in enumerate(Mass):
            Mass[index] = 'double '+elem 
        for index,elem in enumerate(Width): 
            Width[index] = 'double '+elem
            
        listout = self.make_call_lists()
        OnShell = listout['OnShell']
        DeclareList = listout['DeclareList']
        OffShellParticle = listout['OffShell']
        MomentumConserve = listout['Momentum']
        headerstring = '#ifndef '+self.namestring+'_guard \n #define '+self.namestring+'_guard\n'
        string = '#include <complex>\n using namespace std;\n'
        headerstring += '#include <complex>\n using namespace std;\n'
        # define the type of function and argument
        if OnShell:
            headerstring += 'void ' + self.namestring + '('+','.join(DeclareList) + ',complex<double> &vertex);\n'
            string += 'void ' + self.namestring + '(' + ','.join(DeclareList + Mass + Width) + ',complex<double> &vertex){\n'
        else: 
            AddedDeclare.append('complex<double> denom')
            OutputParticle = DeclareList.pop(OffShellParticle+1)
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
        if not OnShell: 
            NegSign = MomentumConserve.pop(OffShellParticle) 
            NegSignBool = re.match('-F', NegSign) 
            if NegSignBool:
                NegString = '(' 
            else:
                NegString = '-('
                
            #Implement the conservation of momenta             
            if re.match('S', self.particles[OffShellParticle][0]):
                firstnumber = 1
            elif re.match('T', self.particles[OffShellParticle][0]):
                firstnumber = 16
            else: 
                firstnumber = 4
            MomString = ''
            for i in range(2): 
                MomString += '%s%d[%d]=' % (self.particles[OffShellParticle][0], OffShellParticle + 1,firstnumber+i) + NegString 
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
        type_to_pos={'S':1,'T':16,'V':4,'F':4}
        for mom in Momenta:
             
            index = int(mom[-1])
            
            type = self.particles[index-1][0]
            energy_pos=type_to_pos[type]
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
        OnShell = 1
        OutString = ''
        for index, elem in enumerate(self.particles):
            if not elem[1]:
                OnShell = 0
                OffShellParticle = '%s%d' % (elem[0], index + 1)
        if OnShell:
            for ind in self.obj.listindices():
                string = 'vertex = C*' + self.write_obj(self.obj.get_rep(ind))
                string = string.replace('+-', '-')
                OutString = OutString + string + ';\n'
        else:
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
        hWriter = Writer.CPPWriter(self.outheader)
        ccWriter = Writer.CPPWriter(self.out)
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

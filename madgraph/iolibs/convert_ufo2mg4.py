""" export the UFO model to a valid MG4 fortran model """

import os
import re
import sys


import madgraph.iolibs.export_v4 as export_v4

class convert_model_to_mg4(object):
    """ A converter of the UFO to the MG4 format """

    event_independant = ['G',"g"] #this is compute internally by MG4
    
    def __init__(self, model_path, output_path):
        """ initialization of the objects """
        
        sys.path.append(model_path)
        global couplings, lorentz, parameters, particles, vertices
        import couplings 
        import lorentz
        import parameters
        import particles
        import vertices
        
        self.dir_path = output_path
        self.couplings = []
        self.couplings_event_dependent = []
        self.parameters = []
        self.parameters_event_dependent = []


    def open(self, name, comment='c', format='default'):
        
        file_path = os.path.join(self.dir_path, name)
        
        if format == 'fortran':
            fsock = export_v4.FortranFile(file_path, 'w')
        else:
            fsock = open(file_path, 'w')
            
        file.writelines(fsock, comment * 77 + '\n')
        file.writelines(fsock,'%(comment)s written by the UFO converter\n' % \
                               {'comment': comment + (6 - len(comment)) *  ' '})
        file.writelines(fsock, comment * 77 + '\n\n')
        return fsock       

    
    def write_all(self):
        """ write all the files """
        #write the part related to the external parameter
        self.create_ident_card()
        self.create_param_read()
        
        #separate event by event information
        self.analyze_parameters()
        self.analyze_couplings()
        
        #write the definition of the parameter
        self.create_input()
        self.create_intparam_def()
        
        
        # definition of the coupling.
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        
        # the makefile
        self.create_makeinc()
        self.create_param_write()
        #        
        #self.load_basic_parameter()
 
    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in parameters.all_parameters \
                                              if param.nature == 'external']
        fsock.writelines('\n'.join(external_param))
        
    def create_param_read(self):    
        
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)"""
            
            return template % {'name': parameter.name, \
                                    'value': python_to_fortran(parameter.value)}
        
        fsock = self.open('param_read.inc', format='fortran')
        external_param = [format(param) for param in parameters.all_parameters \
                                              if param.nature == 'external']
        fsock.writelines('\n'.join(external_param))
        
        
    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        
        # Write header
        header = """double precision G
                common/strong/ G
                 
                double complex gal(2)
                common/weak/ gal

                double precision DUM0
                common/FRDUM0/ DUM0

                double precision DUM1
                common/FRDUM1/ DUM1
                """        
        fsock.writelines(header)
        
        # Write the Mass definition/ common block
        masses = [param.name for param in parameters.all_parameters \
                                              if param.lhablock == 'MASS']
        fsock.writelines('double precision '+','.join(masses)+'\n')
        fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
        
        # Write the Width definition/ common block
        widths = [param.name for param in parameters.all_parameters \
                                              if param.lhablock == 'DECAY']
        fsock.writelines('double precision '+','.join(widths)+'\n')
        fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
        
        # Write the Couplings
        coupling_list = [coupl.name for coupl in couplings.all_couplings]       
        fsock.writelines('double complex '+', '.join(coupling_list)+'\n')
        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        
    def create_write_couplings(self):
        
        fsock = self.open('coupl_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' Couplings of Standard_Model'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")
        def format(coupl):
            return 'write(*,2) \'%(name)s = \', %(name)s' % {'name': coupl.name}
        
        # Write the Couplings
        lines = [format(coupl) for coupl in couplings.all_couplings]       
        fsock.writelines('\n'.join(lines))
        
        
    def create_input(self):
        """create input.inc containing the definition of the parameters"""
        
        fsock = self.open('input.inc', format='fortran')
        
        real_parameters = [param[0] for param in self.parameters + 
                            self.parameters_event_dependent if param[2] == 'real'
                            and param[0] != 'G']
        real_parameters += [param.name for param in parameters.all_parameters 
                            if param.nature == "external" and param.type == 'real'
                            and param.lhablock not in ['MASS', 'DECAY']]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        
        complex_parameters = [param[0] for param in self.parameters + 
                            self.parameters_event_dependent if param[2] == 'complex']
        complex_parameters += [param.name for param in parameters.all_parameters 
                            if param.nature == "external" and param.type == 'complex']

        fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
        fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')                
    
    

    def create_intparam_def(self):
        """ create intparam_definition.inc """

        def format(expr):
            return python_to_fortran(expr)

        fsock = self.open('intparam_definition.inc', format='fortran')
        
        fsock.write_comments("Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        
        for name, expr, type in self.parameters:
            fsock.writelines("%s = %s\n" % (name, format(expr)))
        
        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        for name, expr, type in self.parameters_event_dependent:
            fsock.writelines("%s = %s\n" % (name, format(expr)))
           
        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")
        fsock.writelines(""" gal(1) = 1d0
                             gal(2) = 1d0
                         """)

        fsock.write_comments("\nDefinition of DUM symbols\n")
        fsock.writelines(""" DUM0 = 0
                             DUM1 = 1
                         """)
    
    def create_couplings(self):
        
        self.create_couplings_main()
        nb_coup_indep = 1 + len(self.couplings) // 15 
        nb_coup_dep = 1 + len(self.couplings_event_dependent) // 15 
        
        for i in range(nb_coup_indep):
            data = self.couplings[15 * i: min(len(self.couplings), 15 * (i+1))]
            self.create_couplings_part(i + 1, data)
        for i in range(nb_coup_dep):
            data = self.couplings_event_dependent[15 * i: 
                           min(len(self.couplings_event_dependent), 15 * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data)        
        
        
      
    def create_couplings_main(self):

        fsock = self.open('couplings.f', format='fortran')
        
        fsock.writelines("""subroutine coup(readlha)

                            implicit none
                            logical readlha
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            
                            include \'input.inc\'
                            include \'coupl.inc\'
                            include \'intparam_definition.inc\'\n\n
                         """)
        nb_coup_indep = 1 + len(self.couplings) // 15 
        nb_coup_dep = 1 + len(self.couplings_event_dependent) // 15 
        
        fsock.writelines('if (readlha) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        fsock.writelines('''\nendif\n''')
        
        fsock.write_comments('\ncouplings ependent of alphas\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')


    def create_couplings_part(self, nb_file, data):
        
        def format(expr):
            return python_to_fortran(expr)

        
        fsock = self.open('couplings%s.f' % nb_file, format='fortran')
        fsock.writelines("""subroutine coup%s()
        
          implicit none
      
          include 'input.inc'
          include 'coupl.inc'
                        """ % nb_file)
        
        for name, value in data:
            fsock.writelines('%s = %s' % (name, format(value)))
        fsock.writelines('end')


    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o '
        
        nb_coup_indep = 1 + len(self.couplings) // 15 
        nb_coup_dep = 1 + len(self.couplings_event_dependent) // 15
        text += ' '.join(['couplings%s.o' % (i+1) \
                                  for i in range(nb_coup_dep + nb_coup_indep) ])
        fsock.writelines(text)
        
    def create_param_write(self):
        """ create param_write """

        fsock = self.open('param_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' External Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")
        def format(name):
            return 'write(*,*) \'%(name)s = \', %(name)s' % {'name': name}
        
        # Write the external parameter
        lines = [format(param.name) for param in parameters.all_parameters if 
                                                     param.nature == "external"]       
        fsock.writelines('\n'.join(lines))        
        
        fsock.writelines("""write(*,*)  ' Internal Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")        
        lines = [format(data[0]) for data in self.parameters]
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data[0]) for data in self.parameters_event_dependent]
        fsock.writelines('\n'.join(lines))                
        
    def analyze_parameters(self):
        """ separate the parameters needed to be recomputed events by events and
        the others"""
        
        for param in parameters.all_parameters:
            if param.nature == 'external':
                continue
            
            if self.is_event_dependent(param.value, param.name):
                self.parameters_event_dependent.append((param.name, param.value, 
                                                        param.type))
            else:
                self.parameters.append((param.name, param.value, param.type))
        
    def analyze_couplings(self):
        """creates the shortcut for all special function/parameter
        separate the couplings dependant of alphas of the others"""
        
        for coupling in couplings.all_couplings:
            expr = self.shorten_coupling_expr(coupling.value)
            if self.is_event_dependent(expr, coupling.name):
                self.couplings_event_dependent.append((coupling.name, expr))
            else:
                self.couplings.append((coupling.name, expr))
        
        self.parameters = [ param for i, param in enumerate(self.parameters)
                           if param not in self.parameters[:i]]
        self.parameters_event_dependent = [ param for i, param 
                            in enumerate(self.parameters_event_dependent)
                            if param not in self.parameters[:i]]


            
    separator = re.compile(r'''[+,-,*,/()]''')
        
    def is_event_dependent(self, expr, name=''):
        """ """
        
        if name in self.event_independant:  
            return False
        
        
        expr = self.separator.sub(' ',expr)
        
        for subexpr in expr.split():
            if subexpr in ['G','aS','g','as','AS']:
                return True

            if subexpr in [data[0] for data in self.couplings_event_dependent + 
                                               self.parameters_event_dependent]:
                return True
            if subexpr in [data[0] for data in self.parameters_event_dependent]:
                return True
    
    def search_type(self, expr, dep=''):
        """return the type associate to the expression"""
        
        if not dep == 1:
            try:
                type = [data[2] for data in self.parameters_event_dependent if 
                                    data[0] == expr][0]
                return type
            except:
                pass
            
        if not dep == 0: 
            try:
                type = [data[2] for data in self.parameters if 
                                    data[0] == expr][0]
                return type
            except:
                pass             
        
        return 'complex'
    
    complex_number = re.compile(r'''complex\((?P<real>[^,\(\)]+),(?P<imag>[^,\(\)]+)\)''')
    expo_expr = re.compile(r'''(?P<expr>\w+)\s*\*\*\s*(?P<expo>\d+)''')
    sqrt_expr = re.compile(r'''cmath.sqrt\((?P<expr>\w+)\)''')
    conj_expr = re.compile(r'''complexconjugate\((?P<expr>\w+)\)''')
    
    def shorten_coupling_expr(self, expr):
        """ apply the rules for the couplings"""

        expr = self.complex_number.sub(self.shorten_complex, expr)
        expr = self.expo_expr.sub(self.shorten_expo, expr)
        expr = self.sqrt_expr.sub(self.shorten_sqrt, expr)
        expr = self.conj_expr.sub(self.shorten_conjugate, expr)
        
        return expr

    def shorten_complex(self, matchobj):
        """"""
        real = float(matchobj.group('real'))
        imag = float(matchobj.group('imag'))
        if real != 0 and imag !=1:
            self.parameters.append(('R%sI%s__' % (real, imag), \
                                        '(%s,%s )' % (python_to_fortran(real), \
                                            python_to_fortran(imag)), 'complex'))
        
            return 'R%sI%s__' % (real, imag)
        else:
            self.parameters.append(('COMPLEXI', '(0d0, 1d0)', 'complex'))
            return 'COMPLEXI'
        
    def shorten_expo(self, matchobj):
        
        expr = matchobj.group('expr')
        exponent = matchobj.group('expo')
        output = '%s__EXP__%s' % (expr, exponent)
        new_expr = '%s**%s' % (expr,exponent)

        if expr.isdigit():
            self.parameters.append( output, new_expr,'real')
        elif self.is_event_dependent(expr):
            type = self.search_type(expr)
            self.parameters_event_dependent.append((output, new_expr, type))
        else:
            type = self.search_type(expr)
            self.parameters.append((output, new_expr, type))

        return output
        
    def shorten_sqrt(self, matchobj):
        
        expr = matchobj.group('expr')
        output = 'SQRT__%s' % (expr)
        if expr.isdigit():
            self.parameters.append(( output, ' cmath.sqrt(%s) ' %  float(expr),
                                                                       'real' ))
        elif self.is_event_dependent(expr):
            type = self.search_type(expr)
            self.parameters_event_dependent.append((output, ' sqrt(%s) ' %
                                                               expr, type))
        else:
            type = self.search_type(expr)
            self.parameters.append((output, ' sqrt(%s) ' %
                                                               expr, type))
        
        return output        
        
    def shorten_conjugate(self, matchobj):
        
        expr = matchobj.group('expr')
        output = 'CONJG__%s' % (expr)
        if expr.isdigit():
            self.parameters.append(( output, ' complexconjugate(%s) ' %  float(expr),
                                                                       'real' ))
        elif self.is_event_dependent(expr):
            self.parameters_event_dependent.append((output, ' conjg(%s) ' %
                                                               expr, 'complex'))
        else:
            self.parameters.append((output, ' conjg(%s) ' %  expr, 'complex'))
        
        return output            







def export_to_mg4(model_path, dir_path):
    """ all the call for the creation of the output """
    export_obj = convert_model_to_mg4(model_path, dir_path)
    export_obj.write_all()      
        
class python_to_fortran(str):
    
    python_split = re.compile(r'''(\*\*|\*|\s|\+|\-|/|\(|\))''')

    operator = {'+': '+', '-': '-', '*': '*', '/' : '/', '**': '**',
                'cmath.sin' :'sin', 'cmath.cos': 'cos', 'cmath.tan': 'tan',
                'cmath.sqrt':'dsqrt', 'sqrt': 'dsqrt', 
                'cmath.pi':'pi',
                'complexconjugate':'conjg',
                ')': ')', '(':'('
                }
    
    operator_key = operator.keys()
    
    def __new__(cls, input):
        """ test"""
        
        if isinstance(input, str):
            converted = cls.convert_string(input)
        else:
            converted = cls.convert_number(input)
        return super(python_to_fortran, cls).__new__(cls, converted)
    
    @classmethod
    def convert_string(cls, expr):
        """look element by element how to convert the string"""
        
        last_element = ""
        this_element = ""
        converted = ""
        for data in cls.python_split.split(expr):
            if data  == '':
                continue
            data = data.lower()

            last_element, this_element = this_element, data
            
            if data in cls.operator_key:
                converted += cls.operator[data]
                continue
            elif data.isdigit() and last_element == '**':
                converted += data 
                continue
            
            #check if this is a number
            test_data = data.replace('d','e')
            try:
                test_data = float(test_data)    
            except ValueError:
                converted += data
                continue
            else:
                converted += ('%e' % test_data).replace('e', 'd')
        return converted
    
    @classmethod
    def convert_number(cls, value):
        """look element by element how to convert a number"""
        
        return ('%e' % value).replace('e', 'd')

if '__main__' == __name__:
    
    file_dir_path = os.path.dirname(os.path.realpath( __file__ ))
    root_path = os.path.join(file_dir_path, os.pardir, os.pardir)
    model_path = os.path.join(root_path,'models', 'sm')
    out_path = os.path.join( model_path, 'fortran')
    
    export_to_mg4(model_path, out_path) 
    print 'done'

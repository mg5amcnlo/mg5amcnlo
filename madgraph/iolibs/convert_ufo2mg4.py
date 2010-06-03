""" export the UFO model to a valid MG4 fortran model """

import os
import re
import sys


import madgraph.iolibs.export_v4 as export_v4


class CompactifyExpression:
    
    event_independant = ['G', "g",'aS'] #this is compute internally by MG4
    
    params_dep = []     # (name, expression, type)
    params_indep = []   # (name, expression, type)
    
    complex_number = re.compile(r'''complex\((?P<real>[^,\(\)]+),(?P<imag>[^,\(\)]+)\)''')
    expo_expr = re.compile(r'''(?P<expr>\w+)\s*\*\*\s*(?P<expo>\d+)''')
    sqrt_expr = re.compile(r'''cmath.sqrt\((?P<expr>\w+)\)''')
    conj_expr = re.compile(r'''complexconjugate\((?P<expr>\w+)\)''')
    
    def shorten_expr(self, expr):
        """ apply the rules of contraction and fullfill
        self.params_dep and self.params_indep with associate_definition"""

        expr = self.complex_number.sub(self.shorten_complex, expr)
        expr = self.expo_expr.sub(self.shorten_expo, expr)
        expr = self.sqrt_expr.sub(self.shorten_sqrt, expr)
        expr = self.conj_expr.sub(self.shorten_conjugate, expr)
        
        return expr
    
    def add_dep(self, tuple):
        """ add tuple if not already define """
        
        if tuple not in self.params_dep:
            self.params_dep.append(tuple)
    
    def add_indep(self, tuple):
        """ add tuple if not already define """
        
        if tuple not in self.params_indep:
            self.params_indep.append(tuple)

    def shorten_complex(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        real = float(matchobj.group('real'))
        imag = float(matchobj.group('imag'))
        if real == 0 and imag ==1:
            self.add_indep( ('COMPLEXI', '(0d0, 1d0)', 'complex') )
            return 'COMPLEXI'
        else:
            self.add_indep( 
                ('R%sI%s__' % (real, imag), \
                 '(%s,%s )' % (python_to_fortran(real), python_to_fortran(imag)),
                 'complex'
                 )
            )
            return 'R%sI%s__' % (real, imag)
        
        
    def shorten_expo(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        
        expr = matchobj.group('expr')
        exponent = matchobj.group('expo')
        output = '%s__EXP__%s' % (expr, exponent)
        new_expr = '%s**%s' % (expr,exponent)

        if expr.isdigit():
            self.add_indep( ( output, new_expr,'real') )
        elif self.is_event_dependent(expr) and expr !='aS':
            type = self.search_type(expr)
            self.add_dep( (output, new_expr, type) )
        else:
            type = self.search_type(expr)
            self.add_indep( (output, new_expr, type) )

        return output
        
    def shorten_sqrt(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        
        expr = matchobj.group('expr')
        output = 'SQRT__%s' % (expr)
        if expr.isdigit():
            self.add_indep( ( output, ' cmath.sqrt(%s) ' %  float(expr), 'real' ))
        elif self.is_event_dependent(expr) and expr !='aS':
            type = self.search_type(expr)
            self.add_dep( (output, ' sqrt(%s) ' % expr, type) )
        else:
            type = self.search_type(expr)
            self.add_indep( (output, ' sqrt(%s) ' % expr, type) )
        
        return output        
        
    def shorten_conjugate(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        
        expr = matchobj.group('expr')
        output = 'CONJG__%s' % (expr)
        if self.is_event_dependent(expr) and expr !='aS':
            self.add_dep( (output, ' conjg(%s) ' % expr, 'complex') )
        else:
            self.add_indep( (output, ' conjg(%s) ' % expr, 'complex') )
                    
        return output            
    
    #RE expression for is_event_dependent
    separator = re.compile(r'''[+,-,*,/()]''')
    
    def is_event_dependent(self, expr, name=''):
        """check if an expression should be evaluated points by points or not
            name authorizes to bypass the check and force to be in some category
        """
        
        # Treat predefined result
        if name in self.event_independant:  
            return False
        
        # Split the different part of the expression in order to say if a 
        #subexpression should be evaluated points by points or not.
        expr = self.separator.sub(' ',expr)
        
        # look for each subexpression
        for subexpr in expr.split():
            if subexpr in ['G','aS','g','as','AS']:
                return True

            if subexpr in [data[0] for data in self.coups_dep + self.params_dep]:
                return True
    
    def search_type(self, expr, dep=''):
        """return the type associate to the expression"""
        
        for (name, value, type) in self.params_dep:
            if name == expr:
                return type
            
        for (name, value, type) in self.params_indep:
            if name == expr:
                return type
        
        return 'complex'
            


class convert_model_to_mg4(CompactifyExpression):
    """ A converter of the UFO to the MG4 format """
    
    def __init__(self, model_path, output_path):
        """ initialization of the objects """
        
        model_updir, model_name = os.path.split(model_path)
        sys.path.append(model_updir)
        __import__(model_name)
        model = sys.modules[model_name]


        self.model = model
        self.model_name = model_name
        
        self.dir_path = output_path
        
        self.coups_dep = []    # (name, expression, type)
        self.coups_indep = []  # (name, expression, type)
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)

    def open(self, name, comment='c', format='default'):
        """ Open the file name in the correct directory and with a valid
        header."""
        
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
 
    def analyze_parameters(self):
        """ separate the parameters needed to be recomputed events by events and
        the others"""
        
        for param in self.model.all_parameters:
            if param.nature == 'external':
                continue
            expr = self.shorten_expr(param.value)
            if self.is_event_dependent(expr, param.name):
                self.params_dep.append( (param.name, expr, param.type) )
            else:
                self.params_indep.append( (param.name, expr, param.type) )
                

    def analyze_couplings(self):
        """creates the shortcut for all special function/parameter
        separate the couplings dependant of alphas of the others"""
        
        for coupling in self.model.all_couplings:
            expr = self.shorten_expr(coupling.value)
            
            if self.is_event_dependent(expr, coupling.name):
                self.coups_dep.append( (coupling.name, expr, 'complex') )
            else:
                self.coups_indep.append( (coupling.name, expr, 'complex') )
        


    ############################################################################
    ##  ROUTINE CREATING THE FILES  ############################################
    ############################################################################



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
        masses = [param.name for param in self.model.all_parameters \
                                              if param.lhablock == 'MASS']
        fsock.writelines('double precision '+','.join(masses)+'\n')
        fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
        
        # Write the Width definition/ common block
        widths = [param.name for param in self.model.all_parameters \
                                              if param.lhablock == 'DECAY']
        fsock.writelines('double precision '+','.join(widths)+'\n')
        fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
        
        # Write the Couplings
        coupling_list = [coupl.name for coupl in self.model.all_couplings]       
        fsock.writelines('double complex '+', '.join(coupling_list)+'\n')
        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        
    def create_write_couplings(self):
        """ write the file coupl_write.inc """
        
        fsock = self.open('coupl_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' Couplings of %s'  
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""" % self.model_name)
        def format(coupl):
            return 'write(*,2) \'%(name)s = \', %(name)s' % {'name': coupl.name}
        
        # Write the Couplings
        lines = [format(coupl) for coupl in self.model.all_couplings]       
        fsock.writelines('\n'.join(lines))
        
        
    def create_input(self):
        """create input.inc containing the definition of the parameters"""
        
        fsock = self.open('input.inc', format='fortran')
        
        real_parameters = [param[0] for param in self.params_dep + 
                            self.params_indep if param[2] == 'real'
                            and param[0] != 'G']
        
        real_parameters += [param.name for param in self.model.all_parameters 
                            if param.nature == "external" and param.type == 'real'
                            and param.lhablock not in ['MASS', 'DECAY']]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        
        complex_parameters = [param[0] for param in self.params_dep + 
                            self.params_indep if param[2] == 'complex']
        complex_parameters += [param.name for param in self.model.all_parameters 
                            if param.nature == "external" and param.type == 'complex']

        fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
        fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')                
    
    

    def create_intparam_def(self):
        """ create intparam_definition.inc """

        fsock = self.open('intparam_definition.inc', format='fortran')
        
        fsock.write_comments(\
                "Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        
        for name, expr, type in self.params_indep:
            fsock.writelines("%s = %s\n" % (name, python_to_fortran(expr) ))
        
        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        for name, expr, type in self.params_dep:
            fsock.writelines("%s = %s\n" % (name, python_to_fortran(expr) ))
           
        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")
        fsock.writelines(""" gal(1) = 1d0
                             gal(2) = 1d0
                         """)

        fsock.write_comments("\nDefinition of DUM symbols\n")
        fsock.writelines(""" DUM0 = 0
                             DUM1 = 1
                         """)
    
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = 25
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        for i in range(nb_coup_indep):
            data = self.coups_indep[nb_def_by_file * i: 
                             min(len(self.coups_indep), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data)
            
        for i in range(nb_coup_dep):
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data)        
        
        
    def create_couplings_main(self, nb_def_by_file=25):
        """ create couplings.f """

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
        
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        fsock.writelines('if (readlha) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        fsock.writelines('''\nendif\n''')
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')


    def create_couplings_part(self, nb_file, data):
        """ create couplings[nb_file].f containing information coming from data
        """
        
        fsock = self.open('couplings%s.f' % nb_file, format='fortran')
        fsock.writelines("""subroutine coup%s()
        
          implicit none
      
          include 'input.inc'
          include 'coupl.inc'
                        """ % nb_file)
        
        for (name, value, type) in data:
            fsock.writelines('%s = %s' % (name, python_to_fortran(value)))
        fsock.writelines('end')


    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o '
        
        nb_coup_indep = 1 + len(self.coups_dep) // 25 
        nb_coup_dep = 1 + len(self.coups_indep) // 25
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
        lines = [format(param.name) for param in self.model.all_parameters if 
                                                     param.nature == "external"]       
        fsock.writelines('\n'.join(lines))        
        
        fsock.writelines("""write(*,*)  ' Internal Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")        
        lines = [format(data[0]) for data in self.params_indep]
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data[0]) for data in self.params_dep]
        
        fsock.writelines('\n'.join(lines))                
        
 
    
    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in self.model.all_parameters \
                                              if param.nature == 'external']
        fsock.writelines('\n'.join(external_param))
        
    def create_param_read(self):    
        """create param_read"""
        
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)"""
            
            return template % {'name': parameter.name, \
                                    'value': python_to_fortran(parameter.value)}
        
        fsock = self.open('param_read.inc', format='fortran')
        external_param = [format(param) for param in self.model.all_parameters \
                                              if param.nature == 'external']
        fsock.writelines('\n'.join(external_param))

    def search_type(self, expr):
        """return the type associate to the expression"""
        
        for param in self.model.all_parameters:
            if param.name == expr:
                return param.type
        
        return CompactifyExpression.search_type(self, expr)



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

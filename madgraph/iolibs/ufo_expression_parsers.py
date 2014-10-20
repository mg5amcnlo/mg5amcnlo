################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Parsers for algebraic expressions coming from UFO, outputting into
different languages/frameworks (Fortran and Pythia8). Uses the PLY 3.3
Lex + Yacc framework"""

import logging
import os
import re
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir))

from madgraph import MadGraph5Error
import vendor.ply.lex as lex
import vendor.ply.yacc as yacc
import models.check_param_card as check_param_card

logger = logging.getLogger('madgraph.ufo_parsers')

# PLY lexer class

class ModelError(MadGraph5Error):
    """Appropriate Error for a wrong parsing"""

class UFOExpressionParser(object):
    """A base class for parsers for algebraic expressions coming from UFO."""

    parsed_string = ""
    logical_equiv = {}

    def __init__(self, **kw):
        """Initialize the lex and yacc"""

        modname = self.__class__.__name__
        self.debugfile = os.path.devnull
        self.tabmodule = os.path.join(root_path, "iolibs",  modname + "_" + "parsetab.py")
        lex.lex(module=self, debug=0)
        self.y=yacc.yacc(module=self, debug=0, debugfile=self.debugfile,
                  tabmodule=self.tabmodule)
        
    def parse(self, buf):
        """Parse the string buf"""
        self.y.parse(buf)
        return self.parsed_string

    # List of tokens and literals
    tokens = (
        'LOGICAL','LOGICALCOMB','POWER', 'CSC', 'SEC', 'ACSC', 'ASEC',
        'SQRT', 'CONJ', 'RE', 'RE2', 'IM', 'PI', 'COMPLEX', 'FUNCTION', 'IF','ELSE',
        'VARIABLE', 'NUMBER','COND','REGLOG', 'ARG'
        )
    literals = "=+-*/(),"

    # Definition of tokens

    def t_CSC(self, t):
        r'(?<!\w)csc(?=\()'
        return t
    def t_SEC(self, t):
        r'(?<!\w)sec(?=\()'
        return t
    def t_ACSC(self, t):
        r'(?<!\w)acsc(?=\()'
        return t
    def t_ASEC(self, t):
        r'(?<!\w)asec(?=\()'
        return t
    def t_REGLOG(self, t):
        r'(?<!\w)reglog(?=\()'
        return t
    def t_COND(self, t):
        r'(?<!\w)cond(?=\()'
        return t
    def t_ARG(self,t):
        r'(?<!\w)arg(?=\()'
    def t_IF(self, t):
        r'(?<!\w)if\s'
        return t
    def t_ELSE(self, t):
        r'(?<!\w)else\s'
        return t
    def t_LOGICAL(self, t):
        r'==|!=|<=|>=|<|>'
        return t
    def t_LOGICALCOMB(self, t):
        r'(?<!\w)and(?=[\s\(])|(?<!\w)or(?=[\s\(])'
        return t
    def t_SQRT(self, t):
        r'cmath\.sqrt'
        return t
    def t_PI(self, t):
        r'cmath\.pi'
        return t
    def t_CONJ(self, t):
        r'complexconjugate'
        return t
    def t_IM(self, t):
        r'(?<!\w)im(?=\()'
        return t
    def t_RE(self, t):
        r'(?<!\w)re(?=\()'
        return t
    def t_RE2(self, t):
        r'\.real|\.imag'
        return t
    
    def t_COMPLEX(self, t):
        r'(?<!\w)complex(?=\()'
        return t
    def t_FUNCTION(self, t):
        r'(cmath\.){0,1}[a-zA-Z_][0-9a-zA-Z_]*(?=\()'
        return t
    def t_VARIABLE(self, t):
        r'[a-zA-Z_][0-9a-zA-Z_]*'
        return t

    t_NUMBER = r'([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]{0,1}[0-9]+){0,1}j{0,1}'
    t_POWER  = r'\*\*'

    t_ignore = " \t"

    re_cmath_function = re.compile("cmath\.(?P<name>[0-9a-zA-Z_]+)")

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        logger.error("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer
    def build(self,**kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    # Definitions for the PLY yacc parser
    # Parsing rules
    precedence = (
        ('right', 'LOGICALCOMB'),
        ('right', 'LOGICAL'),
        ('right','IF'),
        ('right','ELSE'),
        ('left','='),
        ('left','+','-'),
        ('left','*','/'),
        ('left', 'RE2'),
        ('right','UMINUS'),
        ('left','POWER'),
        ('right','REGLOG'),
        ('right','ARG'),
        ('right','CSC'),
        ('right','SEC'),
        ('right','ACSC'),
        ('right','ASEC'),
        ('right','SQRT'),
        ('right','CONJ'),
        ('right','RE'),
        ('right','IM'),
        ('right','FUNCTION'),
        ('right','COMPLEX'),
        ('right','COND'),
        )

    # Dictionary of parser expressions
    def p_statement_expr(self, p):
        'statement : expression'
        self.parsed_string = p[1]

    def p_expression_binop(self, p):
        '''expression : expression '=' expression
                      | expression '+' expression
                      | expression '-' expression
                      | expression '*' expression
                      | expression '/' expression'''
        p[0] = p[1] + p[2] + p[3]

    def p_expression_logical(self, p):
        '''boolexpression : expression LOGICAL expression'''
        if p[2] not in self.logical_equiv:
            p[0] = p[1] + p[2] + p[3]
        else:
            p[0] = p[1] + self.logical_equiv[p[2]] + p[3]        

    def p_expression_logicalcomb(self, p):
        '''boolexpression : boolexpression LOGICALCOMB boolexpression'''
        if p[2] not in self.logical_equiv:
            p[0] = p[1] + p[2] + p[3]
        else:
            p[0] = p[1] + self.logical_equiv[p[2]] + p[3]

    def p_expression_uminus(self, p):
        "expression : '-' expression %prec UMINUS"
        p[0] = '-' + p[2]

    def p_group_parentheses(self, p):
        "group : '(' expression ')'"
        p[0] = '(' + p[2] +')'

    def p_group_parentheses_boolexpr(self, p):
        "boolexpression : '(' boolexpression ')'"
        p[0] = '(' + p[2] +')'

    def p_expression_group(self, p):
        "expression : group"
        p[0] = p[1]

    def p_expression_function1(self, p):
        "expression : FUNCTION '(' expression ')'"
        p1 = p[1]
        re_groups = self.re_cmath_function.match(p1)
        if re_groups:
            p1 = re_groups.group("name")
        p[0] = p1 + '(' + p[3] + ')'

    def p_expression_function2(self, p):
        "expression : FUNCTION '(' expression ',' expression ')'"
        p1 = p[1]
        re_groups = self.re_cmath_function.match(p1)
        if re_groups:
            p1 = re_groups.group("name")
        p[0] = p1 + '(' + p[3] + ',' + p[5] + ')'

    def p_error(self, p):
        if p:
            raise ModelError("Syntax error at '%s' (%s)." %(p.value,p))
        else:
            logger.error("Syntax error at EOF")
        self.parsed_string = "Error"

class UFOExpressionParserFortran(UFOExpressionParser):
    """A parser for UFO algebraic expressions, outputting
    Fortran-style code."""

    # The following parser expressions need to be defined for each
    # output language/framework
    
    logical_equiv = {'==':'.EQ.',
                     '>=':'.GE.',
                     '<=':'.LE.',
                     '!=':'.NE.',
                     '>':'.GT.',
                     '<':'.LT.',
                     'or':'.OR.',
                     'and':'.AND.'}

    def p_expression_number(self, p):
        "expression : NUMBER"
        if p[1].endswith('j'):
            p[0] = ('DCOMPLX(0d0, %e)' % float(p[1][:-1])).replace('e', 'd')
        else:
            p[0] = ('%e' % float(p[1])).replace('e', 'd')

    def p_expression_variable(self, p):
        "expression : VARIABLE"
        p[0] = p[1].lower()

    def p_expression_power(self, p):
        'expression : expression POWER expression'
        try:
            p3 = float(p[3].replace('d','e'))
            # Chebck if exponent is an integer
            if p3 == int(p3):
                p3 = str(int(p3))
                p[0] = p[1] + "**" + p3
            else:
                p[0] = p[1] + "**" + p[3]
        except Exception:
            p[0] = p[1] + "**" + p[3]

    def p_expression_if(self,p):
        "expression :   expression IF boolexpression ELSE expression "
        p[0] = 'CONDIF(%s,DCMPLX(%s),DCMPLX(%s))' % (p[3], p[1], p[5])
            
    def p_expression_ifimplicit(self,p):
        "expression :   expression IF expression ELSE expression "
        p[0] = 'CONDIF(DCMPLX(%s).NE.(0d0,0d0),DCMPLX(%s),DCMPLX(%s))'\
                                                             %(p[3], p[1], p[5])

    def p_expression_cond(self, p):
        "expression :  COND '(' expression ',' expression ',' expression ')'"
        p[0] = 'COND(DCMPLX('+p[3]+'),DCMPLX('+p[5]+'),DCMPLX('+p[7]+'))'

    def p_expression_complex(self, p):
        "expression : COMPLEX '(' expression ',' expression ')'"
        p[0] = '(' + p[3] + ',' + p[5] + ')'

    def p_expression_func(self, p):
        '''expression : CSC group
                      | SEC group
                      | ACSC group
                      | ASEC group
                      | RE group
                      | IM group
		              | ARG group
                      | SQRT group
                      | CONJ group
                      | REGLOG group'''
        if p[1] == 'csc': p[0] = '1d0/cos' + p[2]
        elif p[1] == 'sec': p[0] = '1d0/sin' + p[2]
        elif p[1] == 'acsc': p[0] = 'asin(1./' + p[2] + ')'
        elif p[1] == 'asec': p[0] = 'acos(1./' + p[2] + ')'
        elif p[1] == 're': p[0] = 'dble' + p[2]
        elif p[1] == 'im': p[0] = 'dimag' + p[2]
        elif p[1] == 'arg': p[0] = 'arg(DCMPLX'+p[2]+')'
        elif p[1] == 'cmath.sqrt' or p[1] == 'sqrt': p[0] = 'sqrt' + p[2]
        elif p[1] == 'complexconjugate': p[0] = 'conjg(DCMPLX' + p[2]+')'
        elif p[1] == 'reglog': p[0] = 'reglog(DCMPLX' + p[2] +')'


    def p_expression_real(self, p):
        ''' expression : expression RE2 '''
        
        if p[2] == '.real':
            if p[1].startswith('('):
                p[0] = 'dble' +p[1]
            else:
                p[0] = 'dble(%s)' % p[1]
        elif p[2] == '.imag':
            if p[1].startswith('('):
                p[0] = 'dimag' +p[1]
            else:
                p[0] = 'dimag(%s)' % p[1]            

    def p_expression_pi(self, p):
        '''expression : PI'''
        p[0] = 'pi'

class UFOExpressionParserMPFortran(UFOExpressionParserFortran):
    """A parser for UFO algebraic expressions, outputting
    Fortran-style code for quadruple precision computation."""

    mp_prefix = check_param_card.ParamCard.mp_prefix

    # The following parser expressions need to be defined for each
    # output language/framework

    def p_expression_number(self, p):
        "expression : NUMBER"
        
        if p[1].endswith('j'):
            p[0] = 'CMPLX(0.000000e+00_16, %e_16 ,KIND=16)' % float(p[1][:-1])
        else:
            p[0] = '%e_16' % float(p[1])

    def p_expression_variable(self, p):
        "expression : VARIABLE"
        # All the multiple_precision variables are defined with the prefix _MP_"
        p[0] = (self.mp_prefix+p[1]).lower()

    def p_expression_power(self, p):
        'expression : expression POWER expression'
        try:
            p3 = float(p[3].replace('_16',''))
            # Check if exponent is an integer
            if p3 == int(p3):
                p3 = str(int(p3))
                p[0] = p[1] + "**" + p3
            else:
                p[0] = p[1] + "**" + p[3]
        except Exception:
            p[0] = p[1] + "**" + p[3]

    def p_expression_if(self,p):
        "expression :   expression IF boolexpression ELSE expression "
        p[0] = 'MP_CONDIF(%s,CMPLX(%s,KIND=16),CMPLX(%s,KIND=16))' % (p[3], p[1], p[5])
            
    def p_expression_ifimplicit(self,p):
        "expression :   expression IF expression ELSE expression "
        p[0] = 'MP_CONDIF(CMPLX(%s,KIND=16).NE.(0.0e0_16,0.0e0_16),CMPLX(%s,KIND=16),CMPLX(%s,KIND=16))'\
                                                             %(p[3], p[1], p[5])

    def p_expression_cond(self, p):
        "expression :  COND '(' expression ',' expression ',' expression ')'"
        p[0] = 'MP_COND(CMPLX('+p[3]+',KIND=16),CMPLX('+p[5]+\
                                          ',KIND=16),CMPLX('+p[7]+',KIND=16))'

    def p_expression_func(self, p):
        '''expression : CSC group
                      | SEC group
                      | ACSC group
                      | ASEC group
                      | RE group
                      | IM group
	                  | ARG group
                      | SQRT group
                      | CONJ group
                      | REGLOG group'''
        if p[1] == 'csc': p[0] = '1e0_16/cos' + p[2]
        elif p[1] == 'sec': p[0] = '1e0_16/sin' + p[2]
        elif p[1] == 'acsc': p[0] = 'asin(1e0_16/' + p[2] + ')'
        elif p[1] == 'asec': p[0] = 'acos(1e0_16/' + p[2] + ')'
        elif p[1] == 're': p[0] = 'real' + p[2]
        elif p[1] == 'im': p[0] = 'imag' + p[2]
        elif p[1] == 'arg': p[0] = 'mp_arg(CMPLX(' + p[2] + ',KIND=16))'
        elif p[1] == 'cmath.sqrt' or p[1] == 'sqrt': p[0] = 'sqrt' + p[2]
        elif p[1] == 'complexconjugate': p[0] = 'conjg(CMPLX(' + p[2] + ',KIND=16))'
        elif p[1] == 'reglog': p[0] = 'mp_reglog(CMPLX(' + p[2] +',KIND=16))'

    def p_expression_real(self, p):
        ''' expression : expression RE2 '''
        
        if p[2] == '.real':
            if p[1].startswith('('):
                p[0] = 'real' +p[1]
            else:
                p[0] = 'real(%s)' % p[1]
        elif p[2] == '.imag':
            if p[1].startswith('('):
                p[0] = 'imag' +p[1]
            else:
                p[0] = 'imag(%s)' % p[1]  


    def p_expression_pi(self, p):
        '''expression : PI'''
        p[0] = self.mp_prefix+'pi'

class UFOExpressionParserCPP(UFOExpressionParser):
    """A parser for UFO algebraic expressions, outputting
    C++-style code."""

    logical_equiv = {'==':'==',
                     '>=':'>=',
                     '<=':'<=',
                     '!=':'!=',
                     '>':'>',
                     '<':'<',
                     'or':'||',
                     'and':'&&'}

    # The following parser expressions need to be defined for each
    # output language/framework

    def p_expression_number(self, p):
        'expression : NUMBER'
        
        if p[1].endswith('j'):
            p[0] = 'std::complex<double>(0., %e)'  % float(p[1][:-1]) 
        else:
            p[0] = ('%e' % float(p[1])).replace('e', 'd')
        
        
        p[0] = p[1]
        # Check number is an integer, if so add "."
        if float(p[1]) == int(float(p[1])) and float(p[1]) < 1000:
            p[0] = str(int(float(p[1]))) + '.'

    def p_expression_variable(self, p):
        'expression : VARIABLE'
        p[0] = p[1]

    def p_expression_if(self,p):
        "expression :   expression IF boolexpression ELSE expression "
        p[0] = '(%s ? %s : %s)' % (p[3], p[1], p[5])
            
    def p_expression_ifimplicit(self,p):
        "expression :   expression IF expression ELSE expression "
        p[0] = '(%s ? %s : %s)' % (p[3], p[1], p[5])

    def p_expression_cond(self, p):
        "expression :  COND '(' expression ',' expression ',' expression ')'"
        p[0] = 'COND('+p[3]+','+p[5]+','+p[7]+')'

    def p_expression_power(self, p):
        'expression : expression POWER expression'
        p1=p[1]
        p3=p[3]
        if p[1][0] == '(' and p[1][-1] == ')':
            p1 = p[1][1:-1]
        if p[3][0] == '(' and p[3][-1] == ')':
            p3 = p[3][1:-1]
        p[0] = 'pow(' + p1 + ',' + p3 + ')'        

    def p_expression_complex(self, p):
        "expression : COMPLEX '(' expression ',' expression ')'"
        p[0] = 'std::complex<double>(' + p[3] + ',' + p[5] + ')'
    
    def p_expression_func(self, p):
        '''expression : CSC group
                      | SEC group
                      | ACSC group
                      | ASEC group
                      | RE group
                      | IM group
		              | ARG group
                      | SQRT group
                      | CONJ group
                      | REGLOG group '''
        if p[1] == 'csc': p[0] = '1./cos' + p[2]
        elif p[1] == 'sec': p[0] = '1./sin' + p[2]
        elif p[1] == 'acsc': p[0] = 'asin(1./' + p[2] + ')'
        elif p[1] == 'asec': p[0] = 'acos(1./' + p[2] + ')'
        elif p[1] == 're': p[0] = 'real' + p[2]
        elif p[1] == 'im': p[0] = 'imag' + p[2]
        elif p[1] == 'arg':p[0] = 'arg' + p[2]
        elif p[1] == 'cmath.sqrt' or p[1] == 'sqrt': p[0] = 'sqrt' + p[2]
        elif p[1] == 'complexconjugate': p[0] = 'conj' + p[2]
        elif p[1] == 'reglog': p[0] = 'reglog' + p[2]

    def p_expression_real(self, p):
        ''' expression : expression RE2 '''
        
        if p[2] == '.real':
            if p[1].startswith('('):
                p[0] = 'real' +p[1]
            else:
                p[0] = 'real(%s)' % p[1]
        elif p[2] == '.imag':
            if p[1].startswith('('):
                p[0] = 'imag' +p[1]
            else:
                p[0] = 'imag(%s)' % p[1]    

    
    def p_expression_pi(self, p):
        '''expression : PI'''
        p[0] = 'M_PI'
           


# Main program, allows to interactively test the parser
if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        print "Please specify a parser: fortran, mpfortran or c++"
        exit()
    if sys.argv[1] == "fortran":
        calc = UFOExpressionParserFortran()
    elif sys.argv[1] == "mpfortran":
        calc = UFOExpressionParserMPFortran()
    elif sys.argv[1] == "c++":
        calc = UFOExpressionParserCPP()
    elif sys.argv[1] == "aloha":
        calc = UFOExpressionParserCPP()
    else:
        print "Please specify a parser: fortran, mpfortran or c++"
        print "You gave", sys.argv[1]
        exit()

    while 1:
        try:
            s = raw_input('calc > ')
        except EOFError:
            break
        if not s: continue
        print calc.parse(s)

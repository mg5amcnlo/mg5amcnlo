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

from __future__ import division

import logging
import numbers
import os
import re
import sys


root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path))

import aloha_lib
from aloha_object import *
import vendor.ply.lex as lex
import vendor.ply.yacc as yacc
from aloha.aloha_lib import KERNEL
logger = logging.getLogger('aloha.parsers')


# PLY lexer class
class UFOExpressionParser(object):
    """A base class for parsers for algebraic expressions coming from UFO."""

    parsed_string = ""

    def __init__(self, **kw):
        """Ininitialize the lex and yacc"""

        modname = self.__class__.__name__
        self.debugfile = os.path.devnull
        self.tabmodule = os.path.join(root_path, "iolibs",  modname + "_" + "parsetab.py")
        lex.lex(module=self, debug=0)
        yacc.yacc(module=self, debug=0, debugfile=self.debugfile,
                  tabmodule=self.tabmodule)
        
    def parse(self, buf):
        """Parse the string buf"""
        yacc.parse(buf)
        return self.parsed_string

    # List of tokens and literals
    tokens = (
        'POWER', 'CSC', 'SEC', 'ACSC', 'ASEC',
        'SQRT', 'CONJ', 'RE', 'IM', 'PI', 'COMPLEX', 'FUNCTION',
        'VARIABLE', 'NUMBER'
        )
    literals = "=+-*/(),'"

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
    def t_COMPLEX(self, t):
        r'(?<!\w)complex(?=\()'
        return t
    def t_FUNCTION(self, t):
        r'(cmath\.){0,1}[a-zA-Z_][0-9a-zA-Z_]*(?=\()'
        return t
    def t_VARIABLE(self, t):
        r'[a-zA-Z_][0-9a-zA-Z_]*'
        return t
    
    t_NUMBER = r'([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]{0,1}[0-9]+){0,1}'
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
        ('left','='),
        ('left','+','-'),
        ('left','*','/'),
        ('right','UMINUS'),
        ('left','POWER'),
        ('right','CSC'),
        ('right','SEC'),
        ('right','ACSC'),
        ('right','ASEC'),
        ('right','SQRT'),
        ('right','CONJ'),
        ('right','RE'),
        ('right','IM'),
        ('right','FUNCTION'),
        ('right','COMPLEX')
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

    def p_expression_uminus(self, p):
        "expression : '-' expression %prec UMINUS"
        p[0] = '-' + p[2]

    def p_group_parentheses(self, p):
        "group : '(' expression ')'"
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
            print p[:]
            raise Exception("Syntax error at '%s' in '%s'" % (p.value, self.f))
        else:
            logger.error("Syntax error at EOF")
        self.parsed_string = "Error"


class ALOHAExpressionParser(UFOExpressionParser):

    aloha_object = ['P','Gamma','Gamma5','Sigma','Mass','PSlash',
                    'OverMass2','Width','Scalar','Spinor','Vector',
                    'Spin2','Spin32','C','Epsilon','Metric','Identity',
                    'ProjM','ProjP','Coup']

    def p_expression_pi(self, p):
        '''expression : PI'''
        KERNEL.has_pi = True
        p[0] = 'Param(\'PI\')'

    def p_expression_power(self, p):
        'expression : expression POWER expression'
        
        obj = p[1]
        if '(' in p[1]:
            obj = p[1].split('(',1)[0]
        
        if obj in self.aloha_object:
            p[0] = ''.join(p[1:])
        else:
             new = aloha_lib.KERNEL.add_function_expression('pow', eval(p[1]), eval(p[3]))
             p[0] = str(new)


    def p_expression_variable(self, p):
        "expression : VARIABLE"
        p[0] = 'Param(\'%s\')' % p[1]
        
    def p_expression_variable2(self, p):
        "expression : '\\'' VARIABLE '\\''"
        p[0] = '\'%s\'' % p[2]

    def p_expression_expression(self, p):
        "expression : '\\'' expression '\\''"
        p[0] = '\'%s\'' % p[2]

    def p_expression_complex(self, p):
        "expression : COMPLEX '(' expression ',' expression ')'"
        p[0] = 'complex(' + p[3] + ',' + p[5] + ')'

    def p_expression_number(self, p):
        "expression : NUMBER"
        p[0] = p[1]
        if float(p[1]) == int(float(p[1])) and float(p[1]) < 1000:
            p[0] = str(int(float(p[1])))

    def p_expression_func(self, p):
        '''expression : CSC group
                      | SEC group
                      | ACSC group
                      | ASEC group
                      | RE group
                      | IM group
                      | SQRT group
                      | CONJ group'''
      
        new = aloha_lib.KERNEL.add_function_expression(p[1], eval(p[2])) 
        p[0] = str(new)

    def p_expression_function1(self, p):
        "expression : FUNCTION '(' expression ')'"
        
        p1 = p[1]
        if p1 in self.aloha_object:
            p[0] = p[1]+'('+p[3]+')'
            return 
        re_groups = self.re_cmath_function.match(p1)
        
        if re_groups:
            p1 = re_groups.group("name")
        new = aloha_lib.KERNEL.add_function_expression(p1, eval(p[3]))
        
        p[0] = str(new)
    
    def p_expression_function2(self, p):
        "expression : FUNCTION '(' expression ',' expression ')'"
        
        if p[1] in self.aloha_object:
            p[0] = p[1]+'('+p[3]+','+ p[5]+')'
            return
        
        p1 = p[1]
        re_groups = self.re_cmath_function.match(p1)
        if re_groups:
            p1 = re_groups.group("name")
        new = aloha_lib.KERNEL.add_function_expression(p1, eval(p[3]), eval(p[5]))
        p[0] = str(new)
    
    def p_expression_function3(self, p):
        "expression : FUNCTION '(' expression ',' expression ',' expression ')'"
        
        if p[1] in self.aloha_object:
            p[0] = p[1]+'('+p[3]+','+ p[5]+','+p[7]+')'
            return
        
        p1 = p[1]
        re_groups = self.re_cmath_function.match(p1)
        if re_groups:
            p1 = re_groups.group("name")

        new = aloha_lib.KERNEL.add_function_expression(p1, eval(p[3]), eval(p[5]),eval(p[7]))
        p[0] = str(new)

    
    def p_expression_binop(self, p):
        '''expression : expression '=' expression
                      | expression '+' expression
                      | expression '-' expression
                      | expression '*' expression
                      | expression '/' expression'''
        if p[2] != '/' or p[3].isdigit() or p[3].endswith('.'):
            p[0] = p[1] + p[2] + p[3]
        else:  
            denom = eval(p[3])
            if isinstance(denom, numbers.Number):
                p[0] = p[1] + '*' + str(1/denom)
            else:
                new = aloha_lib.KERNEL.add_function_expression('/', denom)
                p[0] = p[1] + ' * ' + str(new)
        
    def p_expression_function3(self, p):
        "expression : FUNCTION '(' expression ',' expression ',' expression ')'"
        
        if p[1] in self.aloha_object:
            p[0] = p[1]+'('+p[3]+','+ p[5]+','+ p[7]+')'
            return
        
        args = [eval(p[2*i+1]) for i in [1,2,3]]
        new = aloha_lib.KERNEL.add_function_expression(p[1], *args)
        p[0] = str(new)
        
        
        
    def p_expression_function4(self, p):
        "expression : FUNCTION '(' expression ',' expression ',' expression ',' expression ')'"
        if p[1] in self.aloha_object:
            p[0] = p[1]+'('+p[3]+','+ p[5]+','+ p[7]+','+ p[9]+')'            
            return
        args = [eval(p[2*i+1]) for i in [1,2,3,4]]
        new = aloha_lib.KERNEL.add_function_expression(p[1], *args)
        p[0] = str(new)

# Main program, allows to interactively test the parser
if __name__ == '__main__':


    calc = ALOHAExpressionParser()
    while 1:
        try:
            s = raw_input('calc > ')
        except EOFError:
            break
        if not s: continue
        logger.info(calc.parse(s))
        
    
    
        
            
    
    

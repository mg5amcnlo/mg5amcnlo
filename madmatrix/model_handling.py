# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: O. Mattelaer (Sep 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: O. Mattelaer, J. Teig, A. Valassi, Z. Wettersten (2021-2025).
# Integrated with the MadGraph7 project in Feb 2026.

import os
import sys

import math
import re

# AV - create a plugin-specific logger
import logging
PLUGIN_NAME = __name__.rsplit('.',1)[0]
logger = logging.getLogger('madgraph.%s.model_handling'%PLUGIN_NAME)
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'

from madgraph.iolibs import export_cpp, export_mg7
from madgraph.iolibs import file_writers as writers

import aloha
from aloha import aloha_writers
from aloha import unitary_gauge

from collections import defaultdict
from fractions import Fraction
from six import StringIO

# FD gauge check
fd_gauge = (unitary_gauge == 3)

def strip_banner(file_text, banner_mark):
    # skip leading lines that start with '!'
    start = 0
    file_lines = file_text.split('\n')
    for i, line in enumerate(file_lines):
        if line.startswith(banner_mark):
            start = i + 1
        else:
            break
    return '\n'.join(file_lines[start:])

# AV - define a custom ALOHAWriter
# (NB: enable this via MadMatrixUFOModelConverter.aloha_writer)
class MadMatrixALOHAWriter(aloha_writers.ALOHAWriterForGPU):
    # Class structure information
    #  - object
    #  - WriteALOHA(object) [in aloha/aloha_writers.py]
    #  - ALOHAWriterForCPP(WriteALOHA) [in aloha/aloha_writers.py]
    #  - ALOHAWriterForGPU(ALOHAWriterForCPP) [in aloha/aloha_writers.py]
    #  - MadMatrixALOHAWriter(ALOHAWriterForGPU)
    #      This class

    # AV - keep defaults from aloha_writers.ALOHAWriterForGPU
    ###extension = '.cu'
    ###prefix ='__device__'
    type2def = aloha_writers.ALOHAWriterForGPU.type2def

    # AV - modify C++ code from aloha_writers.ALOHAWriterForGPU
    ###ci_definition = 'cxtype cI = cxtype(0., 1.);\n'
    ci_definition = 'const cxtype cI = cxmake( 0., 1. );\n'
    ###realoperator = '.real()'
    ###imagoperator = '.imag()'
    realoperator = 'cxreal' # NB now a function
    imagoperator = 'cximag' # NB now a function

    # AV - improve formatting
    ###type2def['int'] = 'int '
    type2def['int'] = 'int'
    type2def['int_v'] = 'int'
    ###type2def['double'] = 'fptype '
    type2def['double'] = 'fptype'
    ###type2def['complex'] = 'cxtype '
    type2def['complex'] = 'cxtype'

    # AV - add vector types
    type2def['double_v'] = 'fptype_sv'
    type2def['complex_v'] = 'cxtype_sv'

    type2def['aloha_ref'] = '&'

    # AV - modify C++ code from aloha_writers.ALOHAWriterForGPU
    # AV new option: declare C++ variable type only when they are defined?
    ###nodeclare = False # old behaviour (separate declaration with no initialization)
    nodeclare = True # new behaviour (delayed declaration with initialisation)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outname = 'w%s%s' % (self.particles[self.outgoing-1], self.outgoing)
        self.momentum_size = 0 # for ALOHAOBJ implementation the momentum is separated from the wavefunctions

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    def change_number_format(self, number):
        """Formatting the number"""
        def isinteger(x):
            try:
                return int(x) == x
            except TypeError:
                return False
        if isinteger(number):
            if number == 1: out = 'one' # AV
            elif number == -1: out = '-one' # AV
            elif number == 2: out = 'two' # AV
            elif number == -2: out = '-two' # AV
            else: out = '%s.' % (str(int(number))) # This prints -1 as '-1.'
        elif isinstance(number, complex):
            if number.imag:
                if number.real:
                    out = '( %s + %s * cI )' % (self.change_number_format(number.real), \
                                    self.change_number_format(number.imag))
                else:
                    if number.imag == 1:
                        out = 'cI'
                    elif number.imag == -1:
                        out = '-cI'
                    else:
                        out = '( %s * cI )' % self.change_number_format(number.imag)
            else:
                out = '%s' % (self.change_number_format(number.real))
        else:
            tmp = Fraction(str(number))
            tmp = tmp.limit_denominator(100)
            if not abs(tmp - number) / abs(tmp + number) < 1e-8: out = '%.9f' % (number)
            elif tmp.numerator == 1 and tmp.denominator == 2 : out = 'half' # AV
            elif tmp.numerator == -1 and tmp.denominator == 2 : out = '-half' # AV
            elif tmp.numerator == 1 and tmp.denominator == 4 : out = 'quarter' # AV
            elif tmp.numerator == -1 and tmp.denominator == 4 : out = '-quarter' # AV
            else: out = '%s./%s.' % (tmp.numerator, tmp.denominator)
        return out

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    # [NB: this exists in ALOHAWriterForGPU but essentially falls back to ALOHAWriterForCPP]
    # [NB: no, actually this exists twice(!) in ForGPU and the 2nd version is not trivial! but I keep the ForCPP version]
    # This affects HelAmps_sm.h and HelAmps_sm.cc
    def get_header_txt(self, name=None, couplings=None,mode='', combined=False):
        """Define the Header of the fortran file. This include
            - function tag
            - definition of variable
        """
        if name is None:
            name = self.name
        if fd_gauge and name.count("_") > 1:  # FIXME ugly hack to get right header
            combined = True
        if mode=='':
            mode = self.mode
        out = StringIO()
        # define the type of function and argument
        if not 'no_include' in mode:
            out.write('#include \"%s.h\"\n\n' % self.name)
        args = []
        comment_inputs = [] # AV
        for format, argname in self.define_argument_list(couplings):
            if format.startswith('list'):
                type = self.type2def[format[5:]] # double or complex (instead of list_double or list_complex)
                comment_inputs.append('%s[6]'%argname) # AV (wavefuncsize=6 is hardcoded also in export_cpp...)
                ###if not argname.startswith('COUP'): type = self.type2def[format[5:]+'_v'] # AV vectorize (double_v or complex_v)
                if not argname.startswith('COUP'):
                    type = self.type2def['double'] # AV from cxtype_sv to fptype
                    argname = 'all'+argname
                list_arg = '[]'
            else:
                type = self.type2def[format] + ' ' + self.type2def['aloha_ref']
                list_arg = ''
            misc.sprint(argname,self.tag)
            if argname.startswith('COUP'):
                type = self.type2def['double'] # AV from cxtype_sv to fptype array (running alphas #373)
                if 'M' in self.tag:
                    type = 'FLV_COUPLING_VIEW'
                    argname = argname.replace('COUP','MCOUP')
                    list_arg = ""
                    point = self.type2def['aloha_ref']
                else:
                    argname = 'all'+argname # AV from cxtype_sv to fptype array (running alphas #373)
                    list_arg = '[]' # AV from cxtype_sv to fptype array (running alphas #373)
                    point = self.type2def['pointer_coup']
                args.append('%s %s%s%s'% (type, point, argname, list_arg))
                coeff_n = re.search(r"\d*$", argname).group()
                args.append('double Ccoeff%s'% coeff_n) # OM for 'unary minus' #628
            else:
                args.append('%s %s%s'% (type, argname, list_arg))

        indent = ' ' * len( '  %s( ' % name )
        if not self.offshell:
            ###output = '%(doublec)s%(pointer_vertex)s allvertexes' % {
            ###    'doublec': self.type2def['double'],
            ###    'pointer_vertex': self.type2def['pointer_vertex']}
            output = '%(doublec)s allvertexes[]' % {
                'doublec': self.type2def['double']}
            if combined:
                output = output + ', ' + '\n%(indent)s%(doublec)s alltmp[]' % {'doublec': self.type2def['double'], 'indent': indent}
            comment_output = 'amplitude \'vertex\''
            template = 'template<class W_ACCESS, class A_ACCESS, class C_ACCESS>'
        else:
            alohatype = 'aloha%s' % self.particles[self.outgoing -1]
            output = '%(doublec)s %(aloha_ref)s %(spin)s%(id)d' % {
                     'doublec': self.type2def[alohatype],
                     'spin': self.particles[self.outgoing -1],
                     'aloha_ref': self.type2def['aloha_ref'], 
                     'id': self.outgoing}
            if combined:
                    output = output + ', ' + '\n{indent}{doublec} {aloha_ref} {spin}tmp'.format(
                        doublec=self.type2def[alohatype],
                        aloha_ref= self.type2def['aloha_ref'],
                        spin=self.particles[self.outgoing -1],
                        indent=indent,
                    )
            ###self.declaration.add(('list_complex', output)) # AV BUG FIX - THIS IS NOT NEEDED AND IS WRONG (adds name 'cxtype_sv V3[]')
            comment_output = 'wavefunction \'%s%d[6]\'' % ( self.particles[self.outgoing -1], self.outgoing ) # AV (wavefuncsize=6)
            template = 'template<class W_ACCESS, class C_ACCESS>'
        comment = '// Compute the output %s from the input wavefunctions %s' % ( comment_output, ', '.join(comment_inputs) ) # AV
        out.write('  %(comment)s\n  %(template)s\n  %(prefix)s void\n  %(name)s( const %(args)s,\n%(indent)s%(output)s )%(suffix)s' %
                  {'comment': comment, # AV - add comment
                   'template': template, # AV - add template
                   'prefix': self.prefix + ( ' INLINE' if 'is_h' in mode else '' ), # AV - add INLINE
                   'suffix': ( ' ALWAYS_INLINE' if 'is_h' in mode else '' ), # AV - add ALWAYS_INLINE
                   'indent':indent, 'output':output, 'name': name,
                   'args': (',\n' + indent + 'const ').join(args)}) # AV - add const, add indent
        if 'is_h' in mode:
            out.write(';\n')
            out.write('\n  //--------------------------------------------------------------------------\n') # AV add footer
        else:
            ###out.write('\n{\n')
            out.write('\n  {\n') # AV
        return out.getvalue()

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    # This affects HelAmps_sm.cc
    def get_foot_txt(self, combined=False):
        """Prototype for language specific footer"""
        text = ' '
        if not combined and fd_gauge:
            if self.outgoing and 'P1N' not in self.tag:
                name = self.particles[self.outgoing-1]
                if name.startswith(('S','V')):
                    text += '      multiply_propagator_factor<W_ACCESS>(%(name)s%(i)s,%(mass)s%(i)s, %(name)s%(i)s);\n' % \
                            {'name':name, 'mass': 'M%s' % name[1:], 'i': self.outgoing }
        text +='   mgDebug( 1, __FUNCTION__ );\n'
        text +='    return;\n'
        text += '  }\n\n  //--------------------------------------------------------------------------' # AV
        return text

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    # This affects HelAmps_sm.cc
    def get_declaration_txt(self, add_i=True, combined=False):
        """ Prototype for how to write the declaration of variable
            Include the symmetry line (entry FFV_2)
        """
        out = StringIO()
        out.write('    mgDebug( 0, __FUNCTION__ );\n') # AV
        ###argument_var = [name for type,name in self.call_arg] # UNUSED
        for type, name in self.call_arg:
            ###out.write('    %s %s;\n' % ( type, name ) ) # FOR DEBUGGING
            if type.startswith('aloha'):
                out.write('    const cxtype_sv* w%s = W_ACCESS::kernelAccessConst( %s.w );\n' % ( name, name ) )
            if name.startswith('COUP'): # AV from cxtype_sv to fptype array (running alphas #373)
                if 'M' in self.tag:
                    out.write('    cxtype_sv %s;\n' % name )
                else:
                    out.write('    const cxtype_sv %s = C_ACCESS::kernelAccessConst( all%s );\n' % ( name, name ) )
        if not self.offshell:
            vname = 'vertex'
            access = 'A_ACCESS'
            allvname = 'allvertexes'
        else:
            vname = '%(spin)s%(id)d' % { 'spin': self.particles[self.outgoing -1], 'id': self.outgoing }
            access = 'W_ACCESS'
            allvname = vname+".w"
            vname = "w" + vname
        out.write('    cxtype_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
        if combined:
            if not self.offshell:
                vname = 'tmp'
                allvname = 'alltmp'
                access = 'A_ACCESS'
            else:
                vname = '%(spin)stmp' % { 'spin': self.particles[self.outgoing -1]}
                access = 'W_ACCESS'
                allvname = vname+".w"
                vname = "w" + vname
            out.write('    cxtype_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
        if fd_gauge:
            out.write('    cxtype_sv CZERO=cxzero_sv(); \n')
        # define the complex number CI = 0+1j
        if add_i:
            ###out.write(self.ci_definition)
            out.write('    ' + self.ci_definition) # AV
        codedict = {} # AV allow delayed declaration with initialisation
        for type, name in self.declaration.tolist():
            ###print(name) # FOR DEBUGGING
            ###out.write('    %s %s;\n' % ( type, name ) ) # FOR DEBUGGING
            if type.startswith('list'):
                type = type[5:]
                if name.startswith('P'):
                    size = 4
                elif not 'tmp' in name:
                    continue # should be defined in the header
                elif name[0] in ['F','V']:
                    if aloha.loop_mode:
                        size = 8
                    elif fd_gauge and name[0]=='V':
                        size = 7
                    else:
                        size = 6
                elif name[0] == 'S':
                    if aloha.loop_mode:
                        size = 5
                    elif fd_gauge:
                        size = 7
                    else:
                        size = 3
                elif name[0] in ['R','T']:
                    if aloha.loop_mode:
                        size = 20
                    else:
                        size = 18
                fullname = '%s[%s]'%(name, size) # AV
            elif (type, name) not in self.call_arg:
                fullname = name # AV
            else:
                continue # AV no need to declare the variable
            if fullname.startswith('OM') :
                codedict[fullname] = '%s %s' % (self.type2def[type], fullname) # AV UGLY HACK (OM3 is always a scalar)
            else:
                codedict[fullname] = '%s %s' % (self.type2def[type+'_v'], fullname) # AV vectorize, add to codedict
            ###print(fullname, codedict[fullname]) # FOR DEBUGGING
            if self.nodeclare:
                self.declaration.codedict = codedict # AV new behaviour (delayed declaration with initialisation)
            else:
                out.write('    %s;\n' % codedict[fullname] ) # AV old behaviour (separate declaration with no initialization)
        ###out.write('    // END DECLARATION\n') # FOR DEBUGGING
        return out.getvalue()

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    # This affects 'V1[0] = ' in HelAmps_sm.cc
    def get_momenta_txt(self):
        """Define the Header of the C++ file. This include
            - momentum conservation
            - definition of the impulsion"""
        out = StringIO()
        # Define all the required momenta
        p = [] # a list for keeping track how to write the momentum
        signs = self.get_momentum_conservation_sign()
        for i, type in enumerate(self.particles):
            if self.declaration.is_used( 'OM%s' % (i+1) ):
                declname = 'OM%s' % (i+1)
                if self.nodeclare: declname = 'const ' + self.declaration.codedict[declname]
                out.write('    {3} = ( M{0} != {1} ? {2} / ( M{0} * M{0} ) : {1} );\n'.format( # AV use ternary in OM3
                    i+1, '0.', '1.', declname)) # AV force scalar "1." instead of vector "one", add declaration
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type]
                continue
            elif self.offshell:
                if len(p) == 0 :
                    p.append('{0}{1}{2}.pvec[%(i)s]'.format(signs[i],type,i+1,type)) # AV for clang-format (ugly!)
                else:
                    p.append(' ')
                    p.append('{0} {1}{2}.pvec[%(i)s]'.format(signs[i],type,i+1,type))
            if self.declaration.is_used('P%s' % (i+1)):
                self.get_one_momenta_def(i+1, out)
        # Define the resulting momenta
        if self.offshell:
            energy_pos = out_size -2
            type = self.particles[self.outgoing-1]
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 4
            for i in range(size_p):
                dict_energy = {'i':i}
                out.write( '    %s%s.pvec[%s] = %s;\n' % ( type, self.outgoing, i, ''.join(p) % dict_energy ) )
            if self.declaration.is_used( 'P%s' % self.outgoing ):
                self.get_one_momenta_def( self.outgoing, out )
        if self.offshell and fd_gauge:
            type = self.particles[self.outgoing-1]
            if type in ['S','V']:
                for i in range(1,6):
                    cppindex = i -1
                    out.write("    w%(type)s%(out)s[%(ind)s] = CZERO ;\n" % {'type': type, 'out':self.outgoing, 'ind':cppindex})
        # Returning result
        ###print('."' + out.getvalue() + '"') # AV - FOR DEBUGGING
        return out.getvalue()

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting, add delayed declaration with initialisation)
    # This affects 'P1[0] = ' in HelAmps_sm.cc
    def get_one_momenta_def(self, i, strfile):
        type = self.particles[i-1]
        if aloha.loop_mode:
            ptype = 'complex_v'
        else:
            ptype = 'double_v'
        templateval ='%(sign)s%(type)s%(i)d.pvec[%(j)d]'
        if self.nodeclare: strfile.write('    const %s P%d[4] = { ' % ( self.type2def[ptype], i) ) # AV
        for j in range(4):
            sign = self.get_P_sign(i) if self.get_P_sign(i) else '+' # AV
            if self.nodeclare: template = templateval + ( ', ' if j<3 else '' ) # AV
            else: template ='    P%(i)d[%(j)d] = ' + templateval + ';\n' # AV
            strfile.write(template % {'j':j,'type': type, 'i': i, 'sign': sign}) # AV
        if self.nodeclare: strfile.write(' };\n') # AV

    def get_coupling_def(self):
        """Define the coupling constant"""
        # This is the same as the parent class method, but adapted for CUDACPP types

        nb_coupling = 0 
        for ftype, name in self.declaration:
            if name.startswith('COUP'):
                nb_coupling += 1

        out = StringIO()
        if 'M' not in self.tag:
            if self.particles[0] != 'F':
                return ''
            # no matrix coupling, so a single coupling, so this is diagonal in flavor space
            # but still need to check !
            elif self.outgoing == 0  or self.particles[self.outgoing-1] not in ['F']:
                if not self.outgoing:
                    fail = "*vertex = cxzero_sv();"
                else:
                    fail = 'for(int i=0; i<%s%d.np4; i++) { w%s%d[i] = cxzero_sv(); }' % (self.particles[self.outgoing-1], self.outgoing, self.particles[self.outgoing-1], self.outgoing)

                out.write('    const int & flv_index1 = F1.flv_index;\n')
                out.write('    const int & flv_index2 = F2.flv_index;\n')
                out.write('    if(flv_index1 != flv_index2 || flv_index1 == -1) {\n')
                out.write('      %s\n' % fail)
                out.write('      return;\n')
                out.write('    }\n')
            else:
                incoming = [i+1 for i in range(len(self.particles)) if i+1 != self.outgoing and self.particles[self.outgoing-1] == 'F'][0]
                outgoing = self.outgoing
                out.write('    F%i.flv_index = F%i.flv_index;\n' % (outgoing, incoming))

            return out.getvalue()

        if self.outgoing == 0  or self.particles[self.outgoing-1] not in ['F']:
            if not self.outgoing:
                fail = "*vertex = cxzero_sv();"
            else:
                fail = 'for(int i=0; i<%s%d.np4; i++) { w%s%d[i] = cxzero_sv(); }' % (self.particles[self.outgoing-1], self.outgoing, self.particles[self.outgoing-1], self.outgoing)

            out.write('    const int & flv_index1 = F1.flv_index;\n')
            out.write('    const int & flv_index2 = F2.flv_index;\n')
            if nb_coupling >1:
                for i in range(1,nb_coupling+1):
                    out.write('    int zero_coup%i = 0;\n' % i)
                out.write('    if(flv_index1 != flv_index2 || flv_index1 == -1) {\n')
                out.write('      %s\n' % fail)
                out.write('      return;\n')
                out.write('    }\n')
            out.write('    if(flv_index1 == -1 || flv_index2 == -1) {\n')
            out.write('      %s\n' % fail)
            out.write('      return;\n')
            out.write('    }\n')
            if nb_coupling == 1:
                out.write('    if(MCOUP.partner1[flv_index1] != flv_index2) {\n')
                out.write('      %s\n' % fail)
                out.write('      return;\n')
                out.write('    }\n')
            else:
                for i in range(1,nb_coupling+1):
                    out.write('    if(MCOUP%i.partner1[flv_index1] != flv_index2 || MCOUP%i.partner2[flv_index1] != flv_index2) {\n' %(i,i))
                    out.write('      zero_coup%i = 1;\n' % i)
                    out.write('      COUP%i = cxzero_sv();\n' % i)
                    out.write('    }\n')
            if nb_coupling ==1:
                # the coupling is a complex number but in this case it is represented as a sequence of real numbers
                # so, when we need to shift within the array, we need to double the shift width to account for
                # both real and imaginary parts
                out.write('    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + 2*flv_index1 );\n')
            else:
                for i in range(1,nb_coupling+1):
                    # the coupling is a complex number but in this case it is represented as a sequence of real numbers
                    # so, when we need to shift within the array, we need to double the shift width to account for
                    # both real and imaginary parts
                    out.write('    if(zero_coup%i ==0) { COUP%i = C_ACCESS::kernelAccessConst( MCOUP%i.value + 2*flv_index1 ); }\n' % (i,i,i))
        else:
            incoming = [i+1 for i in range(len(self.particles)) if i+1 != self.outgoing and self.particles[self.outgoing-1] == 'F'][0]
            if incoming %2 == 1:
                outgoing = self.outgoing
                out.write('    int flv_index%i = F%i.flv_index;\n' % (incoming, incoming))
                out.write('    if(flv_index%i == -1) {\n' %(incoming))
                out.write('      for(int i=0; i<F%i.nw6; i++) { wF%i[i] = cxzero_sv(); }\n' % (outgoing, outgoing))
                out.write('      F%i.flv_index = -1;\n' % outgoing)
                out.write('      return;\n')
                out.write('    }\n')
                if nb_coupling == 1:
                    out.write('    int flv_index2 = MCOUP.partner1[flv_index%i];\n' %(incoming))
                else:
                    out.write('    int flv_index2 = MCOUP1.partner1[flv_index%i];\n' %(incoming))
                    for i in range(2,nb_coupling+1):
                        out.write('    if(flv_index2 == -1){flv_index2 = MCOUP%i.partner1[flv_index%i];}' %(i, incoming)) 
                out.write('    if(flv_index2 == -1){\n')
                out.write('      for(int i=0; i<F%i.nw6; i++) { wF%i[i] = cxzero_sv(); }\n' % (outgoing, outgoing))
                out.write('      F%i.flv_index = -1;\n' % outgoing)
                out.write('      return;\n')
                out.write('    }\n')
                out.write('    F%i.flv_index = flv_index2;\n' % outgoing)
            else:
                outgoing = self.outgoing
                out.write('    int flv_index%i = F%i.flv_index;\n' % (incoming,incoming))
                out.write('    if(flv_index%i == -1){\n' %(incoming))
                out.write('      for(int i=0; i<F%i.nw6; i++) { wF%i[i] = cxzero_sv(); }\n' % (outgoing, outgoing))
                out.write('      F%i.flv_index = -1;\n' % outgoing)
                out.write('      return;\n')
                out.write('    }\n')
                if nb_coupling == 1:
                    out.write('    int flv_index1 = MCOUP.partner2[flv_index%i];\n' %(incoming))
                else:
                    out.write('    int flv_index1 = MCOUP1.partner2[flv_index%i];\n' %(incoming))
                    for i in range(2,nb_coupling+1):
                        out.write('    if(flv_index1 == -1) { flv_index1 = MCOUP%i.partner2[flv_index%i]; }' %(i, incoming))
                out.write('    if(flv_index1 == -1){\n')
                out.write('      for(int i=0; i<F%i.nw6; i++) { wF%i[i] = cxzero_sv(); }\n' % (outgoing, outgoing))
                out.write('      F%i.flv_index = -1;\n' % outgoing)
                out.write('      return;\n')
                out.write('    }\n')
                out.write('    F%i.flv_index = flv_index1;\n' % outgoing)
 
            for ftype, name in self.declaration:
                if name.startswith('COUP'):
                    # the coupling is a complex number but in this case it is represented as a sequence of real numbers
                    # so, when we need to shift within the array, we need to double the shift width to account for
                    # both real and imaginary parts
                    out.write('    %s = C_ACCESS::kernelAccessConst( M%s.value + 2*flv_index1 );\n' % (name, name))
        return out.getvalue()

    # AV - modify aloha_writers.ALOHAWriterForCPP method (improve formatting)
    # This is called once per FFV function, i.e. once per WriteALOHA instance?
    # It is called by WriteALOHA.write, after get_header_txt, get_declaration_txt, get_momenta_txt, before get_foot_txt
    # This affects 'denom = COUP' in HelAmps_sm.cc
    # This affects 'V1[2] = ' and 'F1[2] = ' in HelAmps_sm.cc
    # This affects 'TMP0 = ' in HelAmps_sm.cc
    # This affects '( *vertex ) = ' in HelAmps_sm.cc
    def define_expression(self):
        """Write the helicity amplitude in C++ format"""
        out = StringIO()
        ###out.write('    mgDebug( 0, __FUNCTION__ );\n') # AV - NO! move to get_declaration.txt
        if self.routine.contracted:
            keys = sorted(self.routine.contracted.keys())
            for name in keys:
                obj = self.routine.contracted[name]
                # This affects 'TMP0 = ' in HelAmps_sm.cc
                ###out.write(' %s = %s;\n' % (name, self.write_obj(obj)))
                if self.nodeclare:
                    out.write('    const %s %s = %s;\n' %
                              (self.type2def['complex_v'], name, self.write_obj(obj))) # AV
                else:
                    out.write('    %s = %s;\n' % (name, self.write_obj(obj))) # AV
                    self.declaration.add(('complex', name))
        for name, (fct, objs) in self.routine.fct.items():
            format = ' %s = %s;\n' % (name, self.get_fct_format(fct))
            out.write(format % ','.join([self.write_obj(obj) for obj in objs])) # AV not used in eemumu?
        numerator = self.routine.expr
        if not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)
        if not self.offshell:
            if coup_name == 'COUP':
                mydict = {'num': self.write_obj(numerator.get_rep([0]))} # '...(TMP4)-cI...' comes from here
                for c in ['coup', 'vertex']:
                    if self.type2def['pointer_%s' %c] in ['*']:
                        mydict['pre_%s' %c] = '( *'
                        mydict['post_%s' %c] = ' )'
                    else:
                        mydict['pre_%s' %c] = ''
                        mydict['post_%s'%c] = ''
                # This affects '( *vertex ) = ' in HelAmps_sm.cc
                out.write('    %(pre_vertex)svertex%(post_vertex)s = Ccoeff * %(pre_coup)sCOUP%(post_coup)s * %(num)s;\n' % mydict) # OM add Ccoeff (fix #825)
            else:
                mydict= {}
                if self.type2def['pointer_vertex'] in ['*']:
                    mydict['pre_vertex'] = '( *'
                    mydict['post_vertex'] = ' )'
                else:
                    mydict['pre_vertex'] = ''
                    mydict['post_vertex'] = ''
                mydict['data'] = self.write_obj(numerator.get_rep([0]))
                # This affects '( *vertex ) = ' in HelAmps_sm.cc
                out.write('    %(pre_vertex)svertex%(post_vertex)s = %(data)s;\n' % mydict)
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            if 'L' not in self.tag:
                coeff = 'denom'
                mydict = {}
                if self.type2def['pointer_coup'] in ['*']:
                    mydict['pre_coup'] = '(*'
                    mydict['post_coup'] = ')'
                else:
                    mydict['pre_coup'] = ''
                    mydict['post_coup'] = ''
                mydict['coup'] = coup_name
                mydict['i'] = self.outgoing
                if self.nodeclare:
                    mydict['declnamedenom'] = 'const %s denom' % self.type2def['complex_v'] # AV
                else:
                    mydict['declnamedenom'] = 'denom' # AV
                    self.declaration.add(('complex','denom'))
                # Need to add the unary operator before the coupling (OM fix for #825)
                if mydict['coup'] != 'one': # but in case where the coupling is not used (one)
                    mydict['pre_coup'] = 'Ccoeff * %s' % mydict['pre_coup']
                if not aloha.complex_mass:
                    # This affects 'denom = COUP' in HelAmps_sm.cc
                    if self.routine.denominator:
                        if self.routine.denominator == '1':
                            out.write('    %(declnamedenom)s = %(pre_coup)s%(coup)s%(post_coup)s;\n' % mydict) # AV
                        else:
                            mydict['denom'] = self.routine.denominator
                            out.write('    %(declnamedenom)s = %(pre_coup)s%(coup)s%(post_coup)s / ( %(denom)s );\n' % mydict) # AV
                    else:
                        out.write('    %(declnamedenom)s = %(pre_coup)s%(coup)s%(post_coup)s / ( ( P%(i)s[0] * P%(i)s[0] ) - ( P%(i)s[1] * P%(i)s[1] ) - ( P%(i)s[2] * P%(i)s[2] ) - ( P%(i)s[3] * P%(i)s[3] ) - M%(i)s * ( M%(i)s - cI * W%(i)s ) );\n' % mydict) # AV
                else:
                    if self.routine.denominator:
                        raise Exception('modify denominator are not compatible with complex mass scheme')
                    # This affects 'denom = COUP' in HelAmps_sm.cc
                    out.write('    %(declnamedenom)s = %(pre_coup)s%(coup)s%(post_coup)s / ( ( P%(i)s[0] * P%(i)s[0] ) - ( P%(i)s[1] *P%(i)s[1] ) - ( P%(i)s[2] * P%(i)s[2] ) - ( P%(i)s[3] * P%(i)s[3] ) - ( M%(i)s * M%(i)s ) );\n' % mydict) # AV
                ###self.declaration.add(('complex','denom')) # AV moved earlier (or simply removed)
                if aloha.loop_mode: ptype = 'list_complex'
                else: ptype = 'list_double'
                self.declaration.add((ptype,'P%s' % self.outgoing))
            else:
                coeff = 'COUP'
            shift = 1 - 1 #to correspond to the shift in fortran indicies with -1 for C++
            if fd_gauge and "S" in self.outname:
                shift = 5 - 1 #to correspond to the shift in fortran indicies with -1 for C++
            for ind in numerator.listindices():
                # This affects 'V1[2] = ' and 'F1[2] = ' in HelAmps_sm.cc
                ###out.write('    %s[%d]= %s*%s;\n' % (self.outname,
                out.write('    %s[%d] = %s * %s;\n' % (self.outname, # AV
                                        self.pass_to_HELAS(ind) + shift, coeff,
                                        self.write_obj(numerator.get_rep(ind))))
        ###return out.getvalue() # AV
        # AV check if one, two, half or quarter are used and need to be defined (ugly hack for #291: can this be done better?)
        out2 = StringIO()
        if 'one' in out.getvalue(): out2.write('    constexpr fptype one( 1. );\n')
        if 'two' in out.getvalue(): out2.write('    constexpr fptype two( 2. );\n')
        if 'half' in out.getvalue(): out2.write('    constexpr fptype half( 1. / 2. );\n')
        if 'quarter' in out.getvalue(): out2.write('    constexpr fptype quarter( 1. / 4. );\n')
        out2.write( out.getvalue() )
        return out2.getvalue()

    # AV - modify aloha_writers.WriteALOHA method (improve formatting)
    def write_MultVariable(self, obj, prefactor=True):
        """Turn a multvariable into a string"""
        mult_list = [self.write_variable_id(id) for id in obj]
        ###data = {'factors': '*'.join(mult_list)}
        data = {'factors': ' * '.join(mult_list)}
        if prefactor and obj.prefactor != 1:
            if obj.prefactor != -1:
                text = '%(prefactor)s * %(factors)s'
                data['prefactor'] = self.change_number_format(obj.prefactor)
            else:
                text = '-%(factors)s' # AV keep default (this is not used in eemumu)
        else:
            text = '%(factors)s'
        return text % data

    # AV - modify aloha_writers.WriteALOHA method (improve formatting)
    def write_MultContainer(self, obj, prefactor=True):
        """Turn a multvariable into a string"""
        mult_list = [self.write_obj(id) for id in obj]
        ###data = {'factors': '*'.join(mult_list)}
        data = {'factors': ' * '.join(mult_list)} # AV
        if prefactor and obj.prefactor != 1:
            if obj.prefactor != -1:
                text = '%(prefactor)s * %(factors)s'
                data['prefactor'] = self.change_number_format(obj.prefactor)
            else:
                text = '-%(factors)s' # AV keep default (this is not used in eemumu)
        else:
            text = '%(factors)s'
        return text % data

    def shift_indices(self, match):
        """shift the indices for non impulsion object"""
        if match.group('var').startswith('P'):
            shift = 0
            return '%s[%s]' % (match.group('var'), int(match.group('num')) + shift) 
        else:
            shift =  -1
            if fd_gauge and match.group('var').startswith('S'):
                shift += 4
            return 'w%s[%s]' % (match.group('var'), int(match.group('num')) + shift)

    # OM - overload aloha_writers.WriteALOHA and ALOHAWriterForCPP methods (handle 'unary minus' #628)
    def change_var_format(self, obj):
        """ """
        if obj.startswith('COUP'):
            out = super().change_var_format(obj)
            postfix = out[4:]
            return "Ccoeff%s * %s" % (postfix, out) # OM for 'unary minus' #628
        else:
            return super().change_var_format(obj)

    # AV - new method (based on implementation of write_obj and write_MultVariable)
    def objIsSimpleVariable(self, obj) :
        ###print ( obj.vartype, obj.prefactor, len( obj ), obj ) # AV - FOR DEBUGGING
        return ( obj.vartype == 0 ) or ( obj.vartype == 2 and len( obj ) == 1 )

    # AV - modify aloha_writers.WriteALOHA method (improve formatting)
    # This affects 'V1[2] = ' and 'F1[2] = ' in HelAmps_sm.cc
    def write_obj_Add(self, obj, prefactor=True):
        """Turns addvariable into a string"""
        data = defaultdict(list)
        number = []
        [data[p.prefactor].append(p) if hasattr(p, 'prefactor') else number.append(p) for p in obj]
        file_str = StringIO()
        if prefactor and obj.prefactor != 1:
            formatted = self.change_number_format(obj.prefactor)
            if formatted.startswith(('+','-')):
                file_str.write('( %s )' % formatted)
            else:
                file_str.write(formatted)
            file_str.write(' * ( ')
        else:
            file_str.write('( ')
        ###print('."'+file_str.getvalue()+'"') # AV - FOR DEBUGGING
        first=True
        for value, obj_list in data.items():
            ###print('.."' + str(value) + '" "' + str(obj_list) + '"') # AV - FOR DEBUGGING
            add= ' + '
            if value not in  [-1,1]:
                nb_str = self.change_number_format(value)
                ###print('.>"' + nb_str + '"') # AV - FOR DEBUGGING
                if nb_str[0] in ['+', '-']:
                    if first: file_str.write(nb_str)
                    else : file_str.write(' ' + nb_str[0] + ' ' + nb_str[1:])
                elif first and nb_str == '( half * cI )':
                    file_str.write('half * cI')
                elif not first and nb_str == '( -half * cI )':
                    file_str.write(' - half * cI')
                else:
                    file_str.write('+' if first else ' + ')
                    file_str.write(nb_str)
                file_str.write(' * ')
                if len( obj_list ) > 1 or not self.objIsSimpleVariable( obj_list[0] ) : file_str.write('( ')
            elif value == -1:
                add = ' - '
                file_str.write('-' if first else ' - ')
            elif not first:
                file_str.write(' + ')
            else:
                file_str.write('')
            first = False
            # AV comment: write_obj here also adds calls declaration_add (via change_var_format) - example: OM3
            ###print('..."'+file_str.getvalue()+'"') # AV - FOR DEBUGGING
            file_str.write( add.join( [self.write_obj(obj, prefactor=False) for obj in obj_list] ) ) # NB: RECURSIVE! (write_obj_Add calls write_obj...)
            ###print('...."'+file_str.getvalue()+'"') # AV - FOR DEBUGGING
            if value not in [1,-1]:
                if len( obj_list ) > 1 or not self.objIsSimpleVariable( obj_list[0] ) : file_str.write(' )')
        if number:
            total = sum(number)
            file_str.write('+ %s' % self.change_number_format(total))
        file_str.write(' )')
        ###print('....."'+file_str.getvalue()+'"') # AV - FOR DEBUGGING
        return file_str.getvalue()

#------------------------------------------------------------------------------------


    # FS overload of upstream function
    # used mainly for FD gauge
    def write_combined_cc(self, lor_names, offshell=None, sym=True, mode=''):
        """Return the content of the .cc file linked to multiple lorentz call."""
        # Set some usefull command
        if offshell is None:
            offshell = self.offshell

        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        self.name = name
        # write head - momenta - body - foot
        text = StringIO()
        routine = StringIO()
        data = {}  # for the formating of the line


        # write header
        new_couplings = ['COUP%s' % (i + 1) for i in range(len(lor_names) + 1)]
        text.write(self.get_header_txt(name=name, couplings=new_couplings, mode=mode, combined=True))
        # Define which part of the routine should be called
        data['addon'] = ''.join(self.tag) + '_%s' % self.offshell

        # how to call the routine
        argument = [name for format, name in self.define_argument_list(new_couplings)]
        index = argument.index('COUP1')
        data['before_coup'] = ','.join(' ' + arg for arg in argument[:index])
        data['after_coup'] = ','.join(argument[index + len(lor_names) + 1:])
        if data['after_coup']:
            data['after_coup'] = ',' + data['after_coup']

        # to define accesses
        if name.endswith('_0'):
            data['access'] = '<W_ACCESS,A_ACCESS,C_ACCESS>'
        else:
            data['access'] = '<W_ACCESS,C_ACCESS>'

        # names of functions to be combined
        lor_list = (self.routine.name,) + lor_names
        line = "    %(name)s%(addon)s%(access)s(%(before_coup)s,%(coup)s,%(ccoef)s%(after_coup)s,%(out)s);\n"
        main = '%(spin)s%(id)d' % {'spin': self.particles[self.offshell - 1],
                                   'id': self.outgoing}
        for i, name in enumerate(lor_list):
            data['name'] = name
            if 'M' in data['addon']:
                data['coup'] = 'MCOUP%d' % (i + 1)
            else:
                data['coup'] = 'allCOUP%d' % (1+i)

            if i == 0:
                if not offshell:
                    data['out'] = 'allvertexes'
                else:
                    data['out'] = main
            elif i == 1:
                if self.offshell:
                    type = self.particles[self.offshell - 1]
                    self.declaration.add(('list_complex', '%stmp' % type))
                    data['out'] = '%stmp' % type
                else:
                    type = ''
                    self.declaration.add(('complex', '%stmp' % type))
                    data['out'] = 'all%stmp' % type
            data['ccoef'] = 'Ccoeff%d' % (i + 1)
            routine.write(line % data)
            if i:
                if not offshell:
                    routine.write('    (*vertex) = (*vertex) + (*tmp);\n')
                else:
                    size = self.type_to_size[self.particles[offshell - 1]] - 2
                    # routine.write(""" i= %s;\nwhile (i < %s)\n{\n""" % (self.momentum_size, self.momentum_size + size))
                    # routine.write(" %(main)s[i] = %(main)s[i] + %(tmp)s[i];\n i++;\n" % \
                    # {'main': main, 'tmp': data['out']})
                    # routine.write('}\n')
                    routine.write('   //unrolled upstream while\n')
                    for index in range(self.momentum_size,
                                       self.momentum_size + size):  # unrolling the upstream loop -1 for cpp
                        routine.write('    w%(main)s[%(index)d] = w%(main)s[%(index)d] + w%(tmp)s[%(index)d];\n' % \
                                      {'main': main, 'tmp': data['out'], 'index': index})
                    self.declaration.add(('int', 'i'))
        self.declaration.discard(('complex', 'COUP'))
        self.declaration.discard(('complex', 'denom'))
        if self.outgoing:
            self.declaration.discard(('list_double', 'P%s' % self.outgoing))
            self.declaration.discard(('double', 'OM%s' % self.outgoing))

        tmtmt = aloha.aloha_lib.KERNEL.reduced_expr2

        for name in aloha.aloha_lib.KERNEL.reduced_expr2:
            self.declaration.discard(('complex', name))

        # clean pointless declaration
        # self.declaration.discard
        text.write(self.get_declaration_txt(add_i=False, combined=True))
        text.write(routine.getvalue())
        text.write(self.get_foot_txt(combined=True))

        text = text.getvalue()
        return text


from os.path import join as pjoin

# AV - define a custom UFOModelConverter
# (NB: enable this via ProcessExporterMadMatrix.create_model_class in output.py)
class MadMatrixUFOModelConverter(export_cpp.UFOModelConverterGPU):
    # Class structure information
    #  - object
    #  - UFOModelConverterCPP(object) [in madgraph/iolibs/export_cpp.py]
    #  - UFOModelConverterGPU(UFOModelConverterCPP) [in madgraph/iolibs/export_cpp.py]
    #  - MadMatrixUFOModelConverter(UFOModelConverterGPU)
    #      This class

    # AV - change defaults from export_cpp.UFOModelConverterCPP
    # (custom tag to appear in 'This file has been automatically generated for')
    output_name = 'CUDA/C++ standalone'

    # AV - change defaults from export_cpp.UFOModelConverterGPU
    ###cc_ext = 'cu' # create HelAmps_sm.cu
    cc_ext = 'cc' # create HelAmps_sm.cc

    # Update defaults from export_cpp.UFOModelConverterGPU
    ###cc_ext = 'cu'
    # The following are defined in OneProcessExporterMadMatrix
    # _template_path = pjoin(_file_path, "aloha", "template_files") # this one is used specifying classpath in read_template_file
    # template_path = pjoin(_file_path, "madgraph", "iolibs", "template_files")

    aloha_template_h = pjoin('madmatrix','cpp_hel_amps_h.inc')
    aloha_template_cc = pjoin('madmatrix','cpp_hel_amps_cc.inc')
    helas_h = pjoin('madmatrix', 'helas.h') # _template_path
    helas_cc = pjoin('madmatrix', 'helas.cu') # _template_path
    param_template_h = pjoin('madmatrix', 'cpp_model_parameters_h.inc')
    param_template_cc = pjoin('madmatrix', 'cpp_model_parameters_cc.inc')

    # AV - use a custom ALOHAWriter (NB: this is an argument to WriterFactory.__new__, either a string or a class!)
    ###aloha_writer = 'cudac' # WriterFactory will use ALOHAWriterForGPU
    aloha_writer = MadMatrixALOHAWriter # WriterFactory will use ALOHAWriterForGPU

    # use template files from MG5DIR; strip leading copyright lines
    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""
        out = []        
        if not fd_gauge:
            helas_temp_file = self.helas_h
        else:
            helas_temp_file = self.helas_h.replace('.h', '_fd.h')
 
        if ext == 'h':
            file = self.read_template_file(helas_temp_file, classpath=True)
        else:
            file = self.read_template_file(self.helas_cc, classpath=True)
        file = strip_banner(file, banner_mark = "!") # skip first 9 lines in helas.h/cu (copyright including ALOHA)
        out.append( file )
        return out

    # AV - use the plugin's OneProcessExporterMadMatrix template_path and _template_path (for aloha_template_h/cc)
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
        return OneProcessExporterMadMatrix.read_template_file(filename, classpath)

    # AV - overload export_cpp.UFOModelConverterCPP method (improve formatting)
    def write_parameters(self, params):
        res = super().write_parameters(params)
        res = res.replace('std::complex<','cxsmpl<') # custom simplex complex class (with constexpr arithmetics)
        res = res.replace('\n','\n    ')
        res = res.replace(',',', ')
        if res == '' : res = '    // (none)'
        else : res = '    ' + res # add leading '  ' after the '// Model' line
        return res

    def write_flv_couplings(self, params):
        """Write out the lines of independent parameters"""

        def_flv = []
        # For each parameter, write name = expr;
        for coupl in params:
            for key, c in coupl.flavors.items():
                # get first/second index
                k1, k2 = [i for i in key if i!=0]
                def_flv.append('%(name)s.partner1[%(in)i] = %(out)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.partner2[%(out)i] = %(in)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.value[%(in)i] = &%(coupl)s;' % {'name': coupl.name,'in': k1-1, 'coupl': c})

        return "\n  ".join(def_flv)

    # AV - overload export_cpp.UFOModelConverterCPP method (improve formatting)
    def write_set_parameters(self, params):
        res = self.super_write_set_parameters_donotfixMajorana(params)
        res = res.replace('(','( ')
        res = res.replace(')',' )')
        res = res.replace('+',' + ')
        res = res.replace('-',' - ')
        res = res.replace('e + ','e+') # fix exponents
        res = res.replace('e - ','e-') # fix exponents
        res = res.replace('=  + ','= +') # fix leading + in assignmments
        res = res.replace('=  - ','= -') # fix leading - in assignmments
        res = res.replace('*',' * ')
        res = res.replace('/',' / ')
        res = res.replace(',',', ')
        res = res.replace(',  ',', ')
        res = res.replace('std::complex<','cxsmpl<') # custom simplex complex class (with constexpr arithmetics)
        if res == '' : res = '// (none)'
        res = res.replace('\n','\n  ')
        res = res.replace('(  - ','( -') # post-fix for susy
        res = res.replace(',  - ',', -') # post-fix for susy
        res = res.replace('Re+mdl','Re + mdl') # post-fix for smeft
        res = res.replace('Re+0','Re + 0') # post-fix for smeft
        res = res.replace('He-2','He - 2') # post-fix for smeft
        res = res.replace(', - ',', -') # post-fix for smeft
        ###misc.sprint( "'"+res+"'" )
        return res

    # AV - new method (merging write_parameters and write_set_parameters)
    def write_hardcoded_parameters(self, params, deviceparams=dict()):
        majorana_widths = []
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                majorana_widths.append( particle.get('width') )
        ###misc.sprint(params) # for debugging
        pardef = super().write_parameters(params)
        parset = self.super_write_set_parameters_donotfixMajorana(params)
        ###print( '"' + pardef + '"' )
        ###print( '"' + parset + '"' )
        if ( pardef == '' ):
            assert parset == '', "pardef is empty but parset is not: '%s'"%parset # AV sanity check (both are empty)
            res = '// (none)\n'
            return res
	#=== Replace patterns in pardef (left of the assignment '=')
        pardef = pardef.replace('std::complex<','cxsmpl<') # custom simplex complex class (with constexpr arithmetics)
	#=== Replace patterns in parset (right of the assignment '=')
        parset = parset.replace('std::complex<','cxsmpl<') # custom simplex complex class (with constexpr arithmetics)
        parset = parset.replace('sqrt(','constexpr_sqrt(') # constexpr sqrt (based on iterative Newton-Raphson approximation)
        parset = parset.replace('pow(','constexpr_pow(') # constexpr pow
        parset = parset.replace('atan(','constexpr_atan(') # constexpr atan for BSM #627
        parset = parset.replace('sin(','constexpr_sin(') # constexpr sin for BSM #627
        parset = parset.replace('cos(','constexpr_cos(') # constexpr cos for BSM #627
        parset = parset.replace('tan(','constexpr_tan(').replace('aconstexpr_tan(','atan(') # constexpr tan for BSM #627
        parset = parset.replace('(','( ')
        parset = parset.replace(')',' )')
        parset = parset.replace('+',' + ')
        parset = parset.replace('-',' - ')
        parset = parset.replace('e + ','e+') # fix exponents
        parset = parset.replace('e - ','e-') # fix exponents
        parset = parset.replace('=  + ','= +') # fix leading + in assignmments
        parset = parset.replace('=  - ','= -') # fix leading - in assignmments
        parset = parset.replace('*',' * ')
        parset = parset.replace('/',' / ')
        parset = parset.replace(',',', ')
	#=== Compute pardef_lines from pardef (left of the assignment '=')
        pardef_lines = {}
        for line in pardef.split('\n'):
            ###print(line) # for debugging
            type, pars = line.rstrip(';').split(' ') # strip trailing ';'
            for par in pars.split(','):
                ###print(len(pardef_lines), par) # for debugging
                if par in majorana_widths:
                    pardef_lines[par] = ( 'constexpr ' + type + ' ' + par + "_abs" )
                elif par in deviceparams:
                    pardef_lines[par] = ( '__device__ constexpr ' + type + ' ' + par )
                else:
                    pardef_lines[par] = ( 'constexpr ' + type + ' ' + par )
        ###misc.sprint( 'pardef_lines size =', len(pardef_lines), ', keys size =', len(pardef_lines.keys()) )
        ###print( pardef_lines ) # for debugging
        ###for line in pardef_lines: misc.sprint(line) # for debugging
	#=== Compute parset_lines from parset (right of the assignment '=')
        parset_pars = []
        parset_lines = {}
        skipnextline = False
        for iline, line in enumerate(parset.split('\n')):
            ###print(iline, line) # for debugging
            if line.startswith('indices'):
                ###print('WARNING! Skip line with leading "indices" :', line)
                continue # skip line with leading "indices", before slha.get_block_entry (#622)
            par, parval = line.split(' = ')
            ###misc.sprint(len(parset_pars), len(parset_lines), par, parval) # for debugging
            if parval.startswith('slha.get_block_entry'): parval = parval.split(',')[2].lstrip(' ').rstrip(');') + ';'
            parset_pars.append( par )
            parset_lines[par] = parval # includes a trailing ';'
        ###misc.sprint( 'parset_pars size =', len(parset_pars) )
        ###misc.sprint( 'parset_lines size =', len(parset_lines), ', keys size =', len(parset_lines.keys()) )
        ###print( parset_lines ) # for debugging
        ###for line in parset_lines: misc.sprint(line) # for debugging
	#=== Assemble pardef_lines and parset_lines into a single string res and then replace patterns in res
        assert len(pardef_lines) == len(parset_lines), 'len(pardef_lines) != len(parset_lines)' # AV sanity check (same number of parameters)
        res = '    '.join( pardef_lines[par] + ' = ' + parset_lines[par] + '\n' for par in parset_pars ) # no leading '    ' on first row
        res = res.replace(' ;',';')
        res = res.replace('= - ','= -') # post-fix for susy
        res = res.replace('(  - ','( -') # post-fix for susy
        res = res.replace('Re+mdl','Re + mdl') # better post-fix for smeft #633
        res = res.replace('Re+0','Re + 0') # better post-fix for smeft #633
        res = res.replace('He-2','He - 2') # better post-fix for smeft #633
        res = res.replace(',  - ',', -') # post-fix for smeft
        ###print(res); assert(False)
        ###misc.sprint( "'"+res+"'" )
        return res

    # AV - replace export_cpp.UFOModelConverterCPP method (split writing of parameters and fixes for Majorana particles #622)
    def super_write_set_parameters_donotfixMajorana(self, params):
        """Write out the lines of independent parameters"""
        res_strings = []
        # For each parameter, write "name = expr;"
        for param in params:
            res_strings.append( "%s" % param.expr )
        res = "\n".join(res_strings)
        res = res.replace('ABS(','std::abs(') # for SMEFT #614 and #616
        return res

    # AV - replace export_cpp.UFOModelConverterCPP method (eventually split writing of parameters and fixes for Majorana particles #622)
    def super_write_set_parameters_onlyfixMajorana(self, hardcoded): # FIXME! split hardcoded (constexpr) and not-hardcoded code
        """Write out the lines of independent parameters"""
        print( 'super_write_set_parameters_onlyfixMajorana (hardcoded=%s)'%hardcoded )
        res_strings = []
        # Correct width sign for Majorana particles (where the width and mass need to have the same sign)        
        prefix = "  " if hardcoded else "" # hardcoded code goes into Parameters.h and needs two extra leading spaces due to a namespace...
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                if hardcoded:
                    res_strings.append( prefix+"  constexpr int %s_sign = ( %s < 0 ? -1 : +1 );" % ( particle.get('width'), particle.get('mass') ) )
                    res_strings.append( prefix+"  constexpr double %(W)s = %(W)s_sign * %(W)s_abs;" % { 'W' : particle.get('width') } )
                else:
                    res_strings.append( prefix+"  if( %s < 0 )" % particle.get('mass'))
                    res_strings.append( prefix+"    %(width)s = -std::abs( %(width)s );" % {"width": particle.get('width')})
        if len( res_strings ) != 0 : res_strings = [ prefix + "  // Fixes for Majorana particles" ] + res_strings
        if not hardcoded: return '\n' + '\n'.join(res_strings) if res_strings else ''
        else: return '\n' + '\n'.join(res_strings) + '\n' if res_strings else '\n'

    # AV - replace export_cpp.UFOModelConverterCPP method (add hardcoded parameters and couplings)
    def super_generate_parameters_class_files(self):
        """Create the content of the Parameters_model.h and .cc files"""
        # First of all, identify which extra independent parameters must be made available through CPU static and GPU constant memory in BSM models
        # because they are used in the event by event calculation of alphaS-dependent couplings
        bsmparam_indep_real_used = []
        bsmparam_indep_complex_used = []
        for param in self.params_indep: # NB this is now done also for 'sm' processes (no check on model name, see PR #824)
            if param.name == 'mdl_complexi' : continue
            if param.name == 'aS' : continue
            # Add BSM parameters which are needed to compute dependent couplings
            # Note: this seemed enough to fix SUSY processes, but not EFT processes
            for coupdep in self.coups_dep.values():
                if param.name in coupdep.expr:
                ###if ' '+param.name+' ' in coupdep.expr: # this is not enough, see review of PR #824 and mg5amcnlo#90
                    if param.type == 'real':
                        bsmparam_indep_real_used.append( param.name )
                    elif param.type == 'complex':
                        bsmparam_indep_complex_used.append( param.name )
            # Add BSM parameters which are needed to compute dependent parameters
            # Note: this was later added to also fix EFT processes (related to #616)
            for pardep in self.params_dep:
                if param.name in pardep.expr:
                ###if param.name in pardep.expr: # this is not enough, see review of PR #824 and mg5amcnlo#90
                    if param.type == 'real':
                        bsmparam_indep_real_used.append( param.name )
                    elif param.type == 'complex':
                        bsmparam_indep_complex_used.append( param.name )
        # Use dict.fromkeys() instead of set() to ensure a reproducible ordering of parameters (see https://stackoverflow.com/a/53657523)
        bsmparam_indep_real_used = dict.fromkeys( bsmparam_indep_real_used ) 
        bsmparam_indep_complex_used = dict.fromkeys( bsmparam_indep_complex_used ) 
        # Then do everything else
        replace_dict = self.default_replace_dict
        replace_dict['info_lines'] = self.get_mg5_info_lines()
        params_indep = [ line.replace('aS, ','')
                         for line in self.write_parameters(self.params_indep).split('\n') ]
        replace_dict['independent_parameters'] = '// Model parameters independent of aS\n    //double aS; // now retrieved event-by-event (as G) from Fortran (running alphas #373)\n' + '\n'.join( params_indep )
        replace_dict['independent_couplings'] = '// Model couplings independent of aS\n' + self.write_parameters(self.coups_indep).replace('cxsmpl<double>', 'cxtype')
        params_dep = [ '    //' + line[4:] + ' // now computed event-by-event (running alphas #373)' for line in self.write_parameters(self.params_dep).split('\n') ]
        replace_dict['dependent_parameters'] = '// Model parameters dependent on aS\n' + '\n'.join( params_dep )
        coups_dep = [ '    //' + line[4:] + ' // now computed event-by-event (running alphas #373)' for line in self.write_parameters(list(self.coups_dep.values())).replace('cxsmpl<double>', 'cxtype').split('\n') ]
        replace_dict['dependent_couplings'] = '// Model couplings dependent on aS\n' + '\n'.join( coups_dep )
        replace_dict['flavor_independent_couplings'] = \
                                    "// Model flavor couplings independent of aS\n" + \
                                    self.write_parameters([c for c in self.coups_flv_indep])
        replace_dict['flavor_dependent_couplings'] = \
                                    "// Model flavor couplings dependent of aS\n" + \
                                    self.write_parameters([c for c in self.coups_flv_dep])                                    
        set_params_indep = [ line.replace('aS','//aS') + ' // now retrieved event-by-event (as G) from Fortran (running alphas #373)'
                             if line.startswith( '  aS =' ) else
                             line for line in self.write_set_parameters(self.params_indep).split('\n') ]
        replace_dict['set_independent_parameters'] = '\n'.join( set_params_indep )
        replace_dict['set_independent_parameters'] += self.super_write_set_parameters_onlyfixMajorana( hardcoded=False ) # add fixes for Majorana particles only in the aS-indep parameters #622
        replace_dict['set_independent_parameters'] += '\n  // BSM parameters that do not depend on alphaS but are needed in the computation of alphaS-dependent couplings;' # NB this is now done also for 'sm' processes (no check on model name, see PR #824)
        replace_dict['set_flv_couplings'] = self.write_flv_couplings(self.coups_flv_dep+self.coups_flv_indep)    
        if len(bsmparam_indep_real_used) + len(bsmparam_indep_complex_used) > 0:
            for ipar, par in enumerate( bsmparam_indep_real_used ):
                replace_dict['set_independent_parameters'] += '\n  mdl_bsmIndepParam[%i] = %s;' % ( ipar, par )
            for ipar, par in enumerate( bsmparam_indep_complex_used ):
                replace_dict['set_independent_parameters'] += '\n  mdl_bsmIndepParam[%i] = %s.real();' % ( len(bsmparam_indep_real_used) + 2 * ipar, par )
                replace_dict['set_independent_parameters'] += '\n  mdl_bsmIndepParam[%i] = %s.imag();' % ( len(bsmparam_indep_real_used) + 2 * ipar + 1, par )
        else:
            replace_dict['set_independent_parameters'] += '\n  // (none)'
        replace_dict['set_independent_couplings'] = self.write_set_parameters(self.coups_indep)
        replace_dict['set_dependent_parameters'] = self.write_set_parameters(self.params_dep)
        replace_dict['set_dependent_couplings'] = self.write_set_parameters(list(self.coups_dep.values()))
        print_params_indep = [ line.replace('std::cout','//std::cout') + ' // now retrieved event-by-event (as G) from Fortran (running alphas #373)'
                               if '"aS =' in line else
                               line for line in self.write_print_parameters(self.params_indep).split('\n') ]
        replace_dict['print_independent_parameters'] = '\n'.join( print_params_indep )
        replace_dict['print_independent_couplings'] = self.write_print_parameters(self.coups_indep)
        replace_dict['print_dependent_parameters'] = self.write_print_parameters(self.params_dep)
        replace_dict['print_dependent_couplings'] = self.write_print_parameters(list(self.coups_dep.values()))
        if 'include_prefix' not in replace_dict:
            replace_dict['include_prefix'] = ''
        assert super().write_parameters([]) == '', 'super().write_parameters([]) is not empty' # AV sanity check (#622)
        assert self.super_write_set_parameters_donotfixMajorana([]) == '', 'super_write_set_parameters_donotfixMajorana([]) is not empty' # AV sanity check (#622)
        ###misc.sprint(self.params_indep) # for debugging
        hrd_params_indep = [ line.replace('constexpr','//constexpr') + ' // now retrieved event-by-event (as G) from Fortran (running alphas #373)' if 'aS =' in line else line for line in self.write_hardcoded_parameters(self.params_indep, {**bsmparam_indep_real_used, **bsmparam_indep_complex_used}).split('\n') if line != '' ] # use bsmparam_indep_real_used + bsmparam_indep_complex_used as deviceparams (dictionary merge as in https://stackoverflow.com/a/26853961)
        replace_dict['hardcoded_independent_parameters'] = '\n'.join( hrd_params_indep ) + self.super_write_set_parameters_onlyfixMajorana( hardcoded=True ) # add fixes for Majorana particles only in the aS-indep parameters #622
        ###misc.sprint(self.coups_indep) # for debugging
        replace_dict['hardcoded_independent_couplings'] = self.write_hardcoded_parameters(self.coups_indep).replace('cxsmpl<double>', 'cxtype')
        ###misc.sprint(self.params_dep) # for debugging
        hrd_params_dep = [ line.replace('constexpr ','//constexpr ') + ' // now computed event-by-event (running alphas #373)' if line != '' else line for line in self.write_hardcoded_parameters(self.params_dep).split('\n') ]
        replace_dict['hardcoded_dependent_parameters'] = '\n'.join( hrd_params_dep )
        ###misc.sprint(self.coups_dep) # for debugging
        hrd_coups_dep = [ line.replace('constexpr','//constexpr') + ' // now computed event-by-event (running alphas #373)' if line != '' else line for line in self.write_hardcoded_parameters(list(self.coups_dep.values())).replace('cxsmpl<double>', 'cxtype').split('\n') ]
        replace_dict['hardcoded_dependent_couplings'] = '\n'.join( hrd_coups_dep )
        replace_dict['nicoup'] = len( self.coups_indep )
        if len( self.coups_indep ) > 0 :
            iicoup = [ '    //constexpr size_t ixcoup_%s = %d + Parameters_dependentCouplings::ndcoup; // out of ndcoup+nicoup' % (par.name, id) for (id, par) in enumerate(self.coups_indep) ]
            replace_dict['iicoup'] = '\n'.join( iicoup )
        else:
            replace_dict['iicoup'] = '    // NB: there are no aS-independent couplings in this physics process'
        replace_dict['ndcoup'] = len( self.coups_dep )
        if len( self.coups_dep ) > 0 :
            idcoup = [ '    constexpr size_t idcoup_%s = %d;' % (name, id) for (id, name) in enumerate(self.coups_dep) ]
            replace_dict['idcoup'] = '\n'.join( idcoup )
            dcoupdecl = [ '      cxtype_sv %s;' % name for name in self.coups_dep ]
            replace_dict['dcoupdecl'] = '\n'.join( dcoupdecl )
            dcoupsetdpar = []
            # Special handling of G and aS parameters (cudacpp starts from G, while UFO starts from aS)
            # For simplicity, compute these parameters directly from G, rather than from another such parameter
            # (e.g. do not compute mdl_sqrt__aS as sqrt of aS, which would require defining aS first)
            gparameters = { 'aS' : 'G * G / 4. / M_PI',
                            'mdl_sqrt__aS' : 'G / 2. / constexpr_sqrt( M_PI )' }
            gparamcoded = set()
            foundG = False
            for pdep in self.params_dep:
                ###misc.sprint(pdep.type, pdep.name) 
                line = '    ' + self.write_hardcoded_parameters([pdep]).rstrip('\n')
                ###misc.sprint(line)
                if not foundG:
                    # Comment out the default UFO assignment of mdl_sqrt__aS (from aS) and of G (from mdl_sqrt__aS), but keep them for reference
                    # (WARNING! This Python CODEGEN code essentially assumes that this refers precisely and only to mdl_sqrt__aS and G)
                    dcoupsetdpar.append( '    ' + line.replace('constexpr double', '//const fptype_sv') )
                ###elif pdep.name == 'mdl_G__exp__2' : # added for UFO mg5amcnlo#89 (complex in susy, should be double as in heft/smeft), now fixed
                ###    # Hardcode a custom assignment (valid for both SUSY and SMEFT) instead of replacing double or complex by fptype in 'line'
                ###    dcoupsetdpar.append('        const fptype_sv ' + pdep.name + ' = G * G;' )
                ###elif pdep.name == 'mdl_G__exp__3' : # added for UFO mg5amcnlo#89 (complex in smeft, should be double), now fixed, may be removed
                ###    # Hardcode a custom assignment (valid for both SUSY and SMEFT) instead of replacing double or complex by fptype in 'line'
                ###    dcoupsetdpar.append('        const fptype_sv ' + pdep.name + ' = G * G * G;' )
                elif pdep.name in gparameters:
                    # Skip the default UFO assignment from aS (if any?!) of aS and mdl_sqrt__aS, as these are now derived from G
                    # (WARNING! no path to this statement! aS is not in params_dep, while mdl_sqrt__aS is handled in 'if not foundG' above)
                    ###misc.sprint('Skip gparameter:', pdep.name)
                    continue
                else:
                    for gpar in gparameters:
                        if ' ' + gpar + ' ' in line and not gpar in gparamcoded:
                            gparamcoded.add(gpar)
                            dcoupsetdpar.append('        const fptype_sv ' + gpar + ' = ' + gparameters[gpar] + ';' )
                    dcoupsetdpar.append( '    ' + line.replace('constexpr double', 'const fptype_sv') )
                if pdep.name == 'G':
                    foundG = True
                    dcoupsetdpar.append('        // *** NB Compute all dependent parameters, including aS, in terms of G rather than in terms of aS ***')
            replace_dict['dcoupsetdpar'] = '\n'.join( dcoupsetdpar )
            dcoupsetdcoup = [ '    ' + line.replace('constexpr cxsmpl<double> ','out.').replace('mdl_complexi', 'cI') for line in self.write_hardcoded_parameters(list(self.coups_dep.values())).split('\n') if line != '' ]
            replace_dict['dcoupsetdcoup'] = '    ' + '\n'.join( dcoupsetdcoup )
            dcoupaccessbuffer = [ '    fptype* %ss = C_ACCESS::idcoupAccessBuffer( couplings, idcoup_%s );'%( name, name ) for name in self.coups_dep ]
            replace_dict['dcoupaccessbuffer'] = '\n'.join( dcoupaccessbuffer ) + '\n'
            dcoupkernelaccess = [ '    cxtype_sv_ref %ss_sv = C_ACCESS::kernelAccess( %ss );'%( name, name ) for name in self.coups_dep ]
            replace_dict['dcoupkernelaccess'] = '\n'.join( dcoupkernelaccess ) + '\n'
            dcoupcompute = [ '    %ss_sv = couplings_sv.%s;'%( name, name ) for name in self.coups_dep ]
            replace_dict['dcoupcompute'] = '\n'.join( dcoupcompute )
            # Special handling in EFT for fptype=float using SIMD
            dcoupoutfptypev2 = [ '      fptype_v %sr_v;\n      fptype_v %si_v;'%(name,name) for name in self.coups_dep ]
            replace_dict['dcoupoutfptypev2'] = ( '\n' if len(self.coups_dep) > 0 else '' ) + '\n'.join( dcoupoutfptypev2 )
            replace_dict['dcoupsetdpar2'] = replace_dict['dcoupsetdpar'].replace('fptype_sv','fptype')
            dcoupsetdcoup2 = [ '    ' + line.replace('constexpr cxsmpl<double> ','const cxtype ').replace('mdl_complexi', 'cI') for line in self.write_hardcoded_parameters(list(self.coups_dep.values())).split('\n') if line != '' ]
            dcoupsetdcoup2 += [ '        %sr_v[i] = cxreal( %s );\n        %si_v[i] = cximag( %s );'%(name,name,name,name) for name in self.coups_dep ]
            replace_dict['dcoupsetdcoup2'] = '  ' + '\n'.join( dcoupsetdcoup2 )
            dcoupoutdcoup2 = [ '      out.%s = cxtype_v( %sr_v, %si_v );'%(name,name,name) for name in self.coups_dep ]
            replace_dict['dcoupoutdcoup2'] = '\n' + '\n'.join( dcoupoutdcoup2 )
            for par in bsmparam_indep_complex_used:
                replace_dict['dcoupsetdcoup'] = replace_dict['dcoupsetdcoup'].replace( par, '(cxtype)'+par )
                replace_dict['dcoupsetdcoup2'] = replace_dict['dcoupsetdcoup2'].replace( par, '(cxtype)'+par )
        else:
            replace_dict['idcoup'] = '    // NB: there are no aS-dependent couplings in this physics process'
            replace_dict['dcoupdecl'] = '      // (none)'
            replace_dict['dcoupsetdpar'] = '        // (none)'
            replace_dict['dcoupsetdcoup'] = '        // (none)'
            replace_dict['dcoupaccessbuffer'] = ''
            replace_dict['dcoupkernelaccess'] = ''
            replace_dict['dcoupcompute'] = '    // NB: there are no aS-dependent couplings in this physics process'
            # Special handling in EFT for fptype=float using SIMD
            replace_dict['dcoupoutfptypev2'] = ''
            replace_dict['dcoupsetdpar2'] = '        // (none)'
            replace_dict['dcoupsetdcoup2'] = '      // (none)'
            replace_dict['dcoupoutdcoup2'] = ''
        # Require HRDCOD=1 in EFT and special handling in EFT for fptype=float using SIMD
        nbsmparam_indep_all_used = len( bsmparam_indep_real_used ) + 2 * len( bsmparam_indep_complex_used )
        replace_dict['max_flavor'] = max((len(ids) for ids in self.model['merged_particles'].values()), default=1)
        replace_dict['bsmdefine'] = '#define MGONGPUCPP_NBSMINDEPPARAM_GT_0 1' if nbsmparam_indep_all_used > 0 else '#undef MGONGPUCPP_NBSMINDEPPARAM_GT_0'
        replace_dict['nbsmip'] = nbsmparam_indep_all_used # NB this is now done also for 'sm' processes (no check on model name, see PR #824)
        replace_dict['hasbsmip'] = '' if nbsmparam_indep_all_used > 0 else '//'
        replace_dict['bsmip'] = ', '.join( list(bsmparam_indep_real_used) + [ '%s.real(), %s.imag()'%(par,par) for par in bsmparam_indep_complex_used] ) if nbsmparam_indep_all_used > 0 else '(none)'
        replace_dict['eftwarn0'] = ''
        replace_dict['eftwarn1'] = ''
        ###if 'eft' in self.model_name.lower():
        ###    replace_dict['eftwarn0'] = '\n//#warning Support for EFT physics models is still limited for HRDCOD=0 builds (#439 and PR #625)'
        ###    replace_dict['eftwarn1'] = '\n//#warning Support for EFT physics models is still limited for HRDCOD=1 builds (#439 and PR #625)'
        if len( bsmparam_indep_real_used ) + len( bsmparam_indep_complex_used ) == 0:
            replace_dict['eftspecial0'] = '\n      // No special handling of non-hardcoded parameters (no additional BSM parameters needed in constant memory)'
        else:
            replace_dict['eftspecial0'] = ''
            for ipar, par in enumerate( bsmparam_indep_real_used ) : replace_dict['eftspecial0'] += '\n      const double %s = bsmIndepParamPtr[%i];' % ( par, ipar )
            for ipar, par in enumerate( bsmparam_indep_complex_used ) : replace_dict['eftspecial0'] += '\n      const cxsmpl<double> %s = cxsmpl<double>( bsmIndepParamPtr[%i], bsmIndepParamPtr[%i] );' % ( par, 2*ipar, 2*ipar+1 )
        file_h = self.read_template_file(self.param_template_h) % replace_dict
        file_cc = self.read_template_file(self.param_template_cc) % replace_dict
        return file_h, file_cc

    def write_parameter_class_files(self):
        # Rename Parameters_%(model_name).h/cc to Parameters.h/cc
        super().write_parameter_class_files()

        # compute the paths the legacy method wrote
        h_dir = os.path.join(self.dir_path, self.include_dir)
        cc_dir = os.path.join(self.dir_path, self.cc_file_dir)

        src_h = os.path.join(h_dir, "Parameters_%s.h" % self.model_name)
        src_cc = os.path.join(cc_dir, "Parameters_%s.cc" % self.model_name)

        dst_h = os.path.join(h_dir, "Parameters.h")
        dst_cc = os.path.join(cc_dir, "Parameters.cc")

        if os.path.exists(src_h):
            os.replace(src_h, dst_h)
        if os.path.exists(src_cc):
            os.replace(src_cc, dst_cc)

        logger.info("Rename files %s->%s, %s->%s" % (
            os.path.split(src_h)[-1], os.path.split(dst_h)[-1],
            os.path.split(src_cc)[-1], os.path.split(dst_cc)[-1]
        ))

    # AV - overload export_cpp.UFOModelConverterCPP method (improve formatting)
    def generate_parameters_class_files(self):
        ###file_h, file_cc = super().generate_parameters_class_files()
        file_h, file_cc = self.super_generate_parameters_class_files()
        file_h = file_h[:-1] # remove extra trailing '\n'
        file_cc = file_cc[:-1] # remove extra trailing '\n'
        # [NB: there is a minor bug in export_cpp.UFOModelConverterCPP.generate_parameters_class_files
        # ['independent_couplings' contains dependent parameters, 'dependent parameters' contains independent_couplings]
        # [This only affects the order in which they are printed out - which is now reversed in the templates]
        # [This has been reported as bug https://bugs.launchpad.net/mg5amcnlo/+bug/1959192]
        return file_h, file_cc

    # AV - replace export_cpp.UFOModelConverterCPP method (add explicit std namespace)
    def write_print_parameters(self, params):
        """Write out the lines of independent parameters"""
        # For each parameter, write name = expr;
        res_strings = []
        for param in params:
            res_strings.append('std::cout << std::setw( 20 ) << \"%s = \" << std::setiosflags( std::ios::scientific ) << std::setw( 10 ) << %s << std::endl;' % (param.name, param.name)) # AV
        if len(res_strings) == 0 : res_strings.append('// (none)')
        ##return '\n'.join(res_strings)
        return '\n  '.join(res_strings) # AV (why was this not necessary before?)

    # AV - replace export_cpp.UFOModelConverterCPP method (add debug printouts)
    # (This is where the loop over FFV functions takes place - I had a hard time to understand it)
    # (Note also that write_combined_cc seems to never be called for our eemumu and ggttgg examples)
    # The calling sequence is the following (understood via MG5_debug after forcing an error by renaming 'write')
    # - madgraph_interface.py 8369 in finalize => self._curr_exporter.convert_model(self._curr_model
    # - output.py 127 in convert_model => super().convert_model(model, wanted_lorentz, wanted_coupling)
    # - export_cpp.py 2503 in convert_model => model_builder.write_files()
    # - export_cpp.py 128 in write_files => self.write_aloha_routines()
    # - export_cpp.py 392 in write_aloha_routines => h_rout, cc_rout = abstracthelas.write(output_dir=None,
    # - create_aloha.py 97 in write => text = writer.write(mode=mode, **opt)
    #   [this is MadMatrixALOHAWriter.write which defaults to ALOHAWriterForCPP.write]
    #   [therein, cc_text comes from WriteALOHA.write, while h_text comes from get_h_text]
    def write_aloha_routines(self):
        """Generate the hel_amps_model.h and hel_amps_model.cc files, which
        have the complete set of generalized Helas routines for the model"""
        import aloha.create_aloha as create_aloha
        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))
        model_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'HelAmps_%s.h' % self.model_name)
        model_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'HelAmps_%s.%s' % (self.model_name, self.cc_ext))
        replace_dict = {}
        replace_dict['output_name'] = self.output_name
        replace_dict['info_lines'] = self.get_mg5_info_lines()
        replace_dict['namespace'] = self.namespace
        replace_dict['model_name'] = self.model_name
        # Read in the template .h and .cc files, stripped of compiler commands and namespaces
        template_h_files = self.read_aloha_template_files(ext = 'h')
        template_cc_files = self.read_aloha_template_files(ext = 'cc')
        if(fd_gauge):
            aloha_model = create_aloha.AbstractALOHAModel(self.model.get('name'), explicit_combine=False)
        else:
            aloha_model = create_aloha.AbstractALOHAModel(self.model.get('name'), explicit_combine=True)
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        if self.wanted_lorentz:
            aloha_model.compute_subset(self.wanted_lorentz)
        else:
            aloha_model.compute_all(save=False, custom_propa=True)
        for abstracthelas in dict(aloha_model).values():
            print(type(abstracthelas), abstracthelas.name) # AV this is the loop on FFV functions
            h_rout, cc_rout = abstracthelas.write(output_dir=None, language=self.aloha_writer, mode='no_include')
            template_h_files.append(h_rout)
            template_cc_files.append(cc_rout)
        replace_dict['function_declarations'] = '\n'.join(template_h_files)
        replace_dict['function_definitions'] = '\n'.join(template_cc_files)
        replace_dict['nwave'] = 4
        if (fd_gauge): replace_dict['nwave'] += 1
        file_h = self.read_template_file(self.aloha_template_h) % replace_dict
        file_cc = self.read_template_file(self.aloha_template_cc) % replace_dict
        file_cc = strip_banner(file_cc, banner_mark = "!") # skip first 9 lines in cpp_hel_amps_cc.inc (copyright including ALOHA)
        # Write the HelAmps_sm.h and HelAmps_sm.cc files
        ###MadMatrixwriters.CPPWriter(model_h_file).writelines(file_h)
        ###MadMatrixwriters.CPPWriter(model_cc_file).writelines(file_cc)
        ###logger.info('Created files %s and %s in directory' \
        ###            % (os.path.split(model_h_file)[-1],
        ###               os.path.split(model_cc_file)[-1]))
        ###logger.info('%s and %s' % \
        ###            (os.path.split(model_h_file)[0],
        ###             os.path.split(model_cc_file)[0]))
        # Write only the HelAmps_sm.h file
        file_h_lines = file_h.split('\n')
        file_h = '\n'.join( file_h_lines[:-3]) # skip the trailing '//---'
        file_h += file_cc # append the contents of HelAmps_sm.cc directly to HelAmps_sm.h!
        file_h = file_h[:-1] # skip the trailing empty line
        writers.CPPWriter(model_h_file).writelines(file_h, formatting=False)
        logger.info('Created file %s in directory %s' \
                    % (os.path.split(model_h_file)[-1], os.path.split(model_h_file)[0] ) )

    def prepare_couplings(self, wanted_couplings = []):
        super().prepare_couplings(wanted_couplings)
        # the two lines below fix #748, i.e. they re-order the dictionary keys following the order in wanted_couplings

        def all_str(wanted_couplings):
            str_repr = []
            for coup in wanted_couplings:
                if isinstance(coup, base_objects.FLV_Coupling):
                    str_repr.append(coup.name)
                else:
                    str_repr.append(coup)
            return str_repr

        running_wanted_couplings = [value for value in all_str(wanted_couplings) if value in self.coups_dep]
        ordered_dict = [(k, self.coups_dep[k]) for k in running_wanted_couplings]
        self.coups_dep = dict((x, y) for x, y in ordered_dict)

    def get_mg5_info_lines(self):
        return super().get_mg5_info_lines().replace('# ', '//')

#------------------------------------------------------------------------------------

import madgraph.iolibs.files as files
import madgraph.various.misc as misc
import madgraph.iolibs.export_v4 as export_v4
import madgraph.core.base_objects as base_objects
# AV - define a custom OneProcessExporter
# (NB: enable this via ProcessExporterMadMatrix.oneprocessclass in output.py)
# (NB: use this directly also in MadMatrixUFOModelConverter.read_template_file)
# (NB: use this directly also in MadMatrixGPUFOHelasCallWriter.super_get_matrix_element_calls)
class OneProcessExporterMadMatrix(export_mg7.OneProcessExporterMG7):
    # Class structure information
    #  - object
    #  - OneProcessExporterCPP(object) [in madgraph/iolibs/export_cpp.py]
    #  - OneProcessExporterMG7(OneProcessExporterCPP) [in madgraph/iolibs/export_mg7.py]
    #  - OneProcessExporterMadMatrix(OneProcessExporterCPP)
    #      This class

    # AV - change defaults from export_cpp.OneProcessExporterCPP
    cc_ext = 'cc' # create CPPProcess.cc
    process_dir = '.'
    include_dir = '.'
    _template_path = pjoin(_file_path, "aloha", "template_files") # this one is used specifying classpath in read_template_file
    template_path = pjoin(_file_path, "madgraph", "iolibs", "template_files")
    process_template_h = pjoin('madmatrix', 'process_h.inc')
    process_template_cc = pjoin('madmatrix', 'process_cc.inc')
    process_class_template = pjoin('madmatrix', 'process_class.inc')
    process_definition_template = pjoin('madmatrix', 'process_function_definitions.inc')
    process_wavefunction_template = pjoin('madmatrix', 'cpp_process_wavefunctions.inc')
    process_sigmaKin_function_template = pjoin('madmatrix', 'process_sigmaKin_function.inc')
    single_process_template = pjoin('madmatrix', 'process_matrix.inc')
    support_multichannel = False
    multichannel_var = ',fptype& multi_chanel_num, fptype& multi_chanel_denom'
    imaginary_unit = "cxtype(0,1)"

    # AV - overload export_cpp.OneProcessExporterCPP constructor (rename gCPPProcess to CPPProcess)
    def __init__(self, *args, **kwargs):
        ###misc.sprint('Entering OneProcessExporterMadMatrix.__init__')
        for kwarg in kwargs: misc.sprint( 'kwargs[%s] = %s' %( kwarg, kwargs[kwarg] ) )
        super().__init__(*args, **kwargs)
        self.process_class = 'CPPProcess'
        ###if self.in_madevent_mode: proc_id = kwargs['prefix']+1 # madevent+cudacpp (NB: HERE SELF.IN_MADEVENT_MODE DOES NOT WORK!)
        if 'prefix' in kwargs: proc_id = kwargs['prefix']+1 # madevent+cudacpp (ime+1 from ProcessExporterFortranMEGroup.generate_subprocess_directory)
        else: proc_id = 0 # standalone_cudacpp
        ###misc.sprint(proc_id)
        self.proc_id = proc_id

    # AV - overload export_cpp.OneProcessExporterCPP method (indent comments in process_lines)
    def get_process_class_definitions(self, write=True):
        replace_dict = super().get_process_class_definitions(write=False)
        replace_dict['process_lines'] = replace_dict['process_lines'].replace('\n','\n  ')
        ###misc.sprint( replace_dict['nwavefuncs'] ) # NB: this (from export_cpp) is the WRONG value of nwf, e.g. 6 for gg_tt (#644)
        ###misc.sprint( self.matrix_elements[0].get_number_of_wavefunctions() ) # NB: this is a different WRONG value of nwf, e.g. 7 for gg_tt (#644)
        ###replace_dict['nwavefunc'] = self.matrix_elements[0].get_number_of_wavefunctions() # how do I get HERE the right value of nwf, e.g. 5 for gg_tt?
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        replace_dict['nbhel'] = self.matrix_elements[0].get_helicity_combinations() # number of helicity combinations
        replace_dict['ndiagrams'] = len(self.matrix_elements[0].get('diagrams')) # AV FIXME #910: elsewhere matrix_element.get('diagrams') and max(config[0]...
        replace_dict['nmaxflavor'] = len(self.matrix_elements[0].get_external_flavors_with_iden()) # number of flavor combinations
        replace_dict['nwave'] = 4
        if (fd_gauge): replace_dict['nwave'] += 1

        replace_dict['namp'] = len(self.amplitudes.get_all_amplitudes())
        replace_dict['model'] = self.model_name
        
        replace_dict['sizew'] = self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['ncomb'] = len([x for x in self.matrix_elements[0].get_helicity_matrix()])
        
        if( write ): # ZW: added dict return for uses in child exporters. Default argument is True so no need to modify other calls to this function
            file = self.read_template_file(self.process_class_template) % replace_dict 
            file = strip_banner(file, banner_mark = "!") # skip first 8 lines in process_class.inc (copyright)
            return file
        else:
            return replace_dict

    # AV - replace export_cpp.OneProcessExporterCPP method (fix CPPProcess.cc)
    def get_process_function_definitions(self, write=True):
        """The complete class definition for the process"""
        replace_dict = super().get_process_function_definitions(write=False) # defines replace_dict['initProc_lines']
        replace_dict['hardcoded_initProc_lines'] = replace_dict['initProc_lines'].replace( 'm_pars->', 'Parameters::')
        couplings2order_indep = []
        ###replace_dict['ncouplings'] = len(self.couplings2order)
        ###replace_dict['ncouplingstimes2'] = 2 * replace_dict['ncouplings']
        replace_dict['nparams'] = len(self.params2order)
        ###replace_dict['nmodels'] = replace_dict['nparams'] + replace_dict['ncouplings'] # AV unused???
        replace_dict['coupling_list'] = ' '
        replace_dict['hel_amps_cc'] = '#include \"HelAmps_%s.cc\"' % self.model_name # AV
        coupling = [''] * len(self.couplings2order)
        params = [''] * len(self.params2order)
        flv_couplings = [''] * len(self.couporderflv)
        for coup, pos in self.couplings2order.items():
            coupling[pos] = coup
        for para, pos in self.params2order.items():
            params[pos] = para
        coupling_indep = [] # AV keep only the alphas-independent couplings #434
        for coup in coupling:
            keep = True
            # Use the same implementation as in UFOModelConverterCPP.prepare_couplings (assume self.model is the same)
            for key, coup_list in self.model['couplings'].items():
                if "aS" in key and coup in coup_list: keep = False
            if keep: coupling_indep.append( coup ) # AV only indep!
        replace_dict['ncouplings'] = len(coupling_indep) # AV only indep!
        replace_dict['nipc'] = len(coupling_indep)
        if len(coupling_indep) > 0:
            replace_dict['cipcassign'] = 'const cxtype tIPC[nIPC] = { cxmake( m_pars->%s ) };'\
                                         % ( ' ), cxmake( m_pars->'.join(coupling_indep) ) # AV only indep!
            replace_dict['cipcdevice'] = '__device__ __constant__ fptype cIPC[nIPC * 2];'
            replace_dict['cipcstatic'] = 'static fptype cIPC[nIPC * 2];'
            replace_dict['cipc2tipcSym'] = 'gpuMemcpyToSymbol( cIPC, tIPC, nIPC * sizeof( cxtype ) );'
            replace_dict['cipc2tipc'] = 'memcpy( cIPC, tIPC, nIPC * sizeof( cxtype ) );'
            replace_dict['cipcdump'] = '\n    //for ( int i=0; i<nIPC; i++ ) std::cout << std::setprecision(17) << "tIPC[i] = " << tIPC[i] << std::endl;'
            coup_str_hrd = '__device__ const fptype cIPC[nIPC * 2] = { '
            for coup in coupling_indep : coup_str_hrd += '(fptype)Parameters::%s.real(), (fptype)Parameters::%s.imag(), ' % ( coup, coup ) # AV only indep!
            coup_str_hrd = coup_str_hrd[:-2] + ' };'
            replace_dict['cipchrdcod'] = coup_str_hrd
        else:
            replace_dict['cipcassign'] = '//const cxtype tIPC[0] = { ... }; // nIPC=0'
            replace_dict['cipcdevice'] = '__device__ __constant__ fptype* cIPC = nullptr; // unused as nIPC=0'
            replace_dict['cipcstatic'] = 'static fptype* cIPC = nullptr; // unused as nIPC=0'
            replace_dict['cipc2tipcSym'] = '//gpuMemcpyToSymbol( cIPC, tIPC, 0 * sizeof( cxtype ) ); // nIPC=0'
            replace_dict['cipc2tipc'] = '//memcpy( cIPC, tIPC, nIPC * sizeof( cxtype ) ); // nIPC=0'
            replace_dict['cipcdump'] = ''
            replace_dict['cipchrdcod'] = '__device__ const fptype* cIPC = nullptr; // unused as nIPC=0'
        replace_dict['nipd'] = len(params)
        if len(params) > 0:
            replace_dict['cipdassign'] = 'const fptype tIPD[nIPD] = { (fptype)m_pars->%s };'\
                                         %( ', (fptype)m_pars->'.join(params) )
            replace_dict['cipddevice'] = '__device__ __constant__ fptype cIPD[nIPD];'
            replace_dict['cipdstatic'] = 'static fptype cIPD[nIPD];'
            replace_dict['cipd2tipdSym'] = 'gpuMemcpyToSymbol( cIPD, tIPD, nIPD * sizeof( fptype ) );'
            replace_dict['cipd2tipd'] = 'memcpy( cIPD, tIPD, nIPD * sizeof( fptype ) );'
            replace_dict['cipddump'] = '\n    //for ( int i=0; i<nIPD; i++ ) std::cout << std::setprecision(17) << "tIPD[i] = " << tIPD[i] << std::endl;'
            param_str_hrd = '__device__ const fptype cIPD[nIPD] = { '
            for para in params : param_str_hrd += '(fptype)Parameters::%s, ' % ( para )
            param_str_hrd = param_str_hrd[:-2] + ' };'
            replace_dict['cipdhrdcod'] = param_str_hrd
        else:
            replace_dict['cipdassign'] = '//const fptype tIPD[0] = { ... }; // nIPD=0'
            replace_dict['cipddevice'] = '//__device__ __constant__ fptype* cIPD = nullptr; // unused as nIPD=0'
            replace_dict['cipdstatic'] = '//static fptype* cIPD = nullptr; // unused as nIPD=0'
            replace_dict['cipd2tipdSym'] = '//gpuMemcpyToSymbol( cIPD, tIPD, 0 * sizeof( fptype ) ); // nIPD=0'
            replace_dict['cipd2tipd'] = '//memcpy( cIPD, tIPD, nIPD * sizeof( fptype ) ); // nIPD=0'
            replace_dict['cipddump'] = ''
            replace_dict['cipdhrdcod'] = '//__device__ const fptype* cIPD = nullptr; // unused as nIPD=0'

        # flavor couplings
        for flv_coup, pos in self.couporderflv.items():
            flv_couplings[pos] = flv_coup
        replace_dict['nipf'] = len(flv_couplings)
        if len(flv_couplings):
            nMF = max(len(ids) for ids in self.model['merged_particles'].values())
            # we have 3 arrays:
            #  - all partner1 arrays combined
            #  - all partner2 arrays combines
            #  - all value arrays combined
            replace_dict['cipfassign'] = """int tIPF_partner1[nMF * nIPF];
    int tIPF_partner2[nMF * nIPF];
    cxtype tIPF_value[nMF * nIPF];
    const FLV_COUPLING tFLV[nIPF] = { m_pars->%s };
    for (int i = 0; i < nIPF; ++i) {
      memcpy( tIPF_partner1 + i * nMF, tFLV[i].partner1, nMF * sizeof( int ) );
      memcpy( tIPF_partner2 + i * nMF, tFLV[i].partner2, nMF * sizeof( int ) );
      for (int j = 0; j < nMF; ++j)
        tIPF_value[i * nMF + j] = tFLV[i].value[j] ? *tFLV[i].value[j] : cxtype{}; // guard from null pointers
    }""" % ( ', m_pars->'.join(flv_couplings) )
            replace_dict['cipfdevice'] = """__device__ __constant__ int cIPF_partner1[nMF * nIPF];
  __device__ __constant__ int cIPF_partner2[nMF * nIPF];
  __device__ __constant__ fptype cIPF_value[nMF * nIPF * 2];"""
            replace_dict['cipfstatic'] = """static int cIPF_partner1[nMF * nIPF];
  static int cIPF_partner2[nMF * nIPF];
  static fptype cIPF_value[nMF * nIPF * 2];"""
            replace_dict['cipf2tipfSym'] = """gpuMemcpyToSymbol( cIPF_partner1, tIPF_partner1, nMF * nIPF * sizeof( int )    );
    gpuMemcpyToSymbol( cIPF_partner2, tIPF_partner2, nMF * nIPF * sizeof( int )    );
    gpuMemcpyToSymbol( cIPF_value   , tIPF_value   , nMF * nIPF * sizeof( cxtype ) );"""
            replace_dict['cipf2tipf'] = """memcpy( cIPF_partner1, tIPF_partner1, nMF * nIPF * sizeof( int )    );
    memcpy( cIPF_partner2, tIPF_partner2, nMF * nIPF * sizeof( int )    );
    memcpy( cIPF_value   , tIPF_value   , nMF * nIPF * sizeof( cxtype ) );"""
            replace_dict['cipfdump'] = '''
    //for ( int i=0; i < nIPD; i++ ) {
    //  std::cout << std::setprecision(17) << "tIPF[i].partner1 = { ";
    //  for ( int j=0; j < nMF-1; j++ ) std::cout << std::setprecision(17) << tIPF[i].partner1[j] << ", ";
    //  std::cout << std::setprecision(17) << tIPF[i].partner1[nMF-1] << " }" << std::endl;
    //  std::cout << std::setprecision(17) << "tIPF[i].partner2 = { ";
    //  for ( int j=0; j < nMF-1; j++ ) std::cout << std::setprecision(17) << tIPF[i].partner2[j] << ", ";
    //  std::cout << std::setprecision(17) << tIPF[i].partner2[nMF-1] << " }" << std::endl;
    //  std::cout << std::setprecision(17) << "tIPF[i].value = { ";
    //  for ( int j=0; j < nMF-1; j++ ) std::cout << std::setprecision(17) << tIPF[i].value[j] << ", ";
    //  std::cout << std::setprecision(17) << tIPF[i].value[nMF-1] << " }" << std::endl;
    //}
'''
            coup_str_hrd_partner1 = '__device__ const int cIPF_partner1[nMF * nIPF] = { '
            coup_str_hrd_partner2 = '__device__ const int cIPF_partner2[nMF * nIPF] = { '
            coup_str_hrd_value    = '__device__ const fptype cIPF_value[nMF * nIPF * 2] = { '
            for flv_coup in flv_couplings:
                coup_str_hrd_partner1 += ( ('Parameters_%(model_name)s::%(coup)s.param1' % {"model_name": self.model_name, "coup": flv_coup} + '[%d], ') * nMF) % ( *range(nMF), )
                coup_str_hrd_partner2 += ( ('Parameters_%(model_name)s::%(coup)s.param2' % {"model_name": self.model_name, "coup": flv_coup} + '[%d], ') * nMF) % ( *range(nMF), )
                value_string = '(fptype)Parameters_%(model_name)s::%(coup)s.value' % {"model_name": self.model_name, "coup": flv_coup}
                range_ids = [ [ i, i ] for i in range(nMF) ]
                coup_str_hrd_value += ( ( value_string + '[%d].real(), ' + value_string + '[%d].imag(), ' ) * nMF) % ( *[ j for i in range_ids for j in i ], )
            coup_str_hrd_partner1 = coup_str_hrd_partner1[:-2] + ' };'
            coup_str_hrd_partner2 = coup_str_hrd_partner2[:-2] + ' };'
            coup_str_hrd_value    = coup_str_hrd_value[:-2] + ' };'
            replace_dict['cipfhrdcod'] = '%s\n  %s\n  %s' % (coup_str_hrd_partner1, coup_str_hrd_partner2, coup_str_hrd_value)
        else:
            replace_dict['cipfassign'] = ''
            replace_dict['cipfdevice'] = """__device__ __constant__ int* cIPF_partner1 = nullptr; // unused as nIPF=0'
    __device__ __constant__ int* cIPF_partner2 = nullptr; // unused as nIPF=0'
    __device__ __constant__ fptype* cIPF_value = nullptr; // unused as nIPF=0'"""
            replace_dict['cipfstatic'] = """static int* cIPF_partner1 = nullptr; // unused as nIPF=0'
    static int* cIPF_partner2 = nullptr; // unused as nIPF=0'
    static fptype* cIPF_value = nullptr; // unused as nIPF=0'"""
            replace_dict['cipf2tipfSym'] = ''
            replace_dict['cipf2tipf'] = ''
            replace_dict['cipfdump'] = ''
            replace_dict['cipfhrdcod'] = """__device__ const int* cIPF_partner1 = nullptr; // unused as nIPF=0'
    __device__ const int* cIPF_partner2 = nullptr; // unused as nIPF=0'
    __device__ const fptype* cIPF_value = nullptr; // unused as nIPF=0'"""
        # FIXME! Here there should be different code generated depending on MGONGPUCPP_NBSMINDEPPARAM_GT_0 (issue #827)
        replace_dict['all_helicities'] = self.get_helicity_matrix(self.matrix_elements[0])
        replace_dict['all_helicities'] = replace_dict['all_helicities'] .replace('helicities', 'tHel')
        replace_dict['all_flavors'] = self.get_flavor_matrix(self.matrix_elements[0])
        replace_dict['all_flavors'] = replace_dict['all_flavors'].replace('flavors', 'tFlavors')
        color_amplitudes = [me.get_color_amplitudes() for me in self.matrix_elements] # as in OneProcessExporterCPP.get_process_function_definitions
        replace_dict['ncolor'] = len(color_amplitudes[0])
        # broken_symmetry_factor function: use the shared decay-aware symmetry
        # data (same as the Fortran / standalone_cpp exporters) instead of the
        # old simple PID-count version, so identical-particle and decay-chain
        # symmetry factors match across backends.
        _, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        process = self.matrix_elements[0].get('processes')[0]
        sym_data = export_v4.ProcessExporterFortran._get_broken_symmetry_data(
            process, nincoming)
        export_v4.ProcessExporterFortran._fill_broken_sym_replace_dict(
            replace_dict, sym_data)

        file = self.read_template_file(self.process_definition_template) % replace_dict # HACK! ignore write=False case
        if len(params) == 0: # remove cIPD from OpenMP pragma (issue #349)
            file_lines = file.split('\n')
            file_lines = [l.replace('cIPC, cIPD','cIPC') for l in file_lines] # remove cIPD from OpenMP pragma
            file = '\n'.join( file_lines )
        file = strip_banner(file, banner_mark = "!") # skip first 8 lines in process_function_definitions.inc (copyright)
        return file

    # AV - modify export_cpp.OneProcessExporterCPP method (add debug printouts for multichannel #342)
    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        ###misc.sprint('Entering OneProcessExporterMadMatrix.get_sigmaKin_lines')
        replace_dict = super().get_sigmaKin_lines(color_amplitudes, write=False)
        replace_dict['proc_id'] = self.proc_id if self.proc_id>0 else 1
        replace_dict['proc_id_source'] = 'MadMatrix exporter'

        # Extract denominator (avoid to extend size for mirroring)
        den_factors = [str(me.get_denominator_factor()) for me in \
                            self.matrix_elements]
        replace_dict['den_factors'] = ",".join(den_factors)

        replace_dict['madE_var_reset'] = """
        fptype multi_chanel_num = 0.;
        fptype multi_chanel_denom = 0.;
        """
        replace_dict['madE_caclwfcts_call'] = '&multi_chanel_num, &multi_chanel_denom'
        replace_dict['madE_update_answer'] = '   allMEs[iproc*nprocesses + ievt] *= multi_chanel_num/multi_chanel_denom;'

        replace_dict['nb_channel'] = len(self.multi_channel_map)
        replace_dict['nb_color'] = max(1, len(self.matrix_elements[0].get('color_basis')))

        if write:
            file = self.read_template_file(self.process_sigmaKin_function_template) % replace_dict
            file = strip_banner(file, banner_mark = "!") # skip first 8 lines in process_sigmaKin_function.inc (copyright)
            return file, replace_dict
        else:
            return replace_dict

    # AV - modify export_cpp.OneProcessExporterCPP method (fix CPPProcess.cc)
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for CPPProcess.cc"""
        ret_lines = []
        if self.single_helicities:
            ###misc.sprint(type(self.helas_call_writer))
            ###misc.sprint( 'before get_matrix_element_calls', self.matrix_elements[0].get_number_of_wavefunctions() ) # WRONG value of nwf, eg 7 for gg_tt
            helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    self.matrix_elements[0],
                                                    color_amplitudes[0],
                                                    multi_channel_map = self.multi_channel_map
                                                    )
            ###misc.sprint( 'after get_matrix_element_calls', self.matrix_elements[0].get_number_of_wavefunctions() ) # CORRECT value of nwf, eg 5 for gg_tt
            assert len(self.matrix_elements) == 1 or len(self.matrix_elements) == 2 # how to handle if this is not true?
            self.couplings2order = self.helas_call_writer.couplings2order
            self.couporderflv = self.helas_call_writer.couporderflv
            self.params2order = self.helas_call_writer.params2order
            ret_lines.append("""
  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams
  // Also compute running sums over helicities adding jamp2, numerator, denominator
  // (NB: this function no longer handles matrix elements as the color sum has now been moved to a separate function/kernel)
  // In CUDA, this function processes a single event
  // ** NB1: NEW Nov2024! In CUDA this is now a kernel function (it used to be a device function)
  // ** NB2: NEW Nov2024! in CUDA this now takes a channelId array as input (it used to take a scalar channelId as input)
  // In C++, this function processes a single event "page" or SIMD vector (or for two in "mixed" precision mode, nParity=2)
  // *** NB: in C++, calculate_jamps accepts a SCALAR channelId because it is GUARANTEED that all events in a SIMD vector have the same channelId #898
  __global__ void /* clang-format off */
  calculate_jamps( int ihel,
                   const fptype* allmomenta,          // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,    // input: indices of the flavor combinations
#ifdef MGONGPUCPP_GPUIMPL
                   fptype* allJamps,                  // output: jamp[2*ncolor*nevt] buffer for one helicity _within a super-buffer for dcNGoodHel helicities_
                   bool storeChannelWeights,
                   fptype* allNumerators,             // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype* allDenominators,           // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype* colAllJamp2s,              // output: allJamp2s[ncolor][nevt] super-buffer, sum over col/hel (nullptr to disable)
                   const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
                   const bool processAllHelicities    // input: if true, use blockIdx.y to index helicities
#else
                   cxtype_sv* allJamp_sv,             // output: jamp_sv[ncolor] (float/double) or jamp_sv[2*ncolor] (mixed) for this helicity
                   bool storeChannelWeights,
                   fptype* allNumerators,             // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype* allDenominators,           // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype_sv* jamp2_sv,               // output: jamp2[nParity][ncolor][neppV] for color choice (nullptr if disabled)
                   const int ievt00                   // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
#endif
                   )
  //ALWAYS_INLINE // attributes are not permitted in a function definition
  {
#ifdef MGONGPUCPP_GPUIMPL
    using namespace mg5amcGpu;
    using M_ACCESS = DeviceAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = DeviceAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = DeviceAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = DeviceAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = DeviceAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using F_ACCESS = DeviceAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = DeviceAccessNumerators;    // non-trivial access: buffer includes all events
    using DEN_ACCESS = DeviceAccessDenominators;  // non-trivial access: buffer includes all events
#else
    using namespace mg5amcCpu;
    using M_ACCESS = HostAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = HostAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = HostAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = HostAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = HostAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using F_ACCESS = HostAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;    // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators;  // non-trivial access: buffer includes all events
#endif
    mgDebug( 0, __FUNCTION__ );
    //bool debug = true;
#ifndef MGONGPUCPP_GPUIMPL
    //debug = ( ievt00 >= 64 && ievt00 < 80 && ihel == 3 ); // example: debug #831
    //if( debug ) printf( \"calculate_jamps: ievt00=%d ihel=%2d\\n\", ievt00, ihel );
#else
    //const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    //debug = ( ievt == 0 );
    //if( debug ) printf( \"calculate_jamps: ievt=%6d ihel=%2d\\n\", ievt, ihel );
    if (processAllHelicities) {
      int ighel = blockIdx.y;
      ihel = dcGoodHel[ighel];
      allJamps = allJamps + ighel * nevt;
      allNumerators = allNumerators + ighel * nevt * processConfig::ndiagrams;
      allDenominators = allDenominators + ighel * nevt;
    }
#endif /* clang-format on */""")
            nwavefuncs = self.matrix_elements[0].get_number_of_wavefunctions()
            ret_lines.append("""
    // The variable nwf (which is specific to each P1 subdirectory, #644) is only used here
    // It is hardcoded here because various attempts to hardcode it in CPPProcess.h at generation time gave the wrong result...
    static const int nwf = %i; // #wavefunctions = #external (npar) + #internal: e.g. 5 for e+ e- -> mu+ mu- (1 internal is gamma or Z)"""%nwavefuncs )
            ret_lines.append("""
    // Local TEMPORARY variables for a subset of Feynman diagrams in the given CUDA event (ievt) or C++ event page (ipagV)
    // [NB these variables are reused several times (and re-initialised each time) within the same event or event page]
    // ** NB: in other words, amplitudes and wavefunctions still have TRIVIAL ACCESS: there is currently no need
    // ** NB: to have large memory structurs for wavefunctions/amplitudes in all events (no kernel splitting yet)!
    //MemoryBufferWavefunctions w_buffer[nwf]{ neppV };
    // Create memory for both momenta and wavefunctions separately, and later wrap them in ALOHAOBJ
    fptype_sv pvec_sv[nwf][np4];
    cxtype_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams (nw6 is 4: spin wavefunctions, momenta are no more included, see before)
    cxtype_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram

    // Wrap the memory into ALOHAOBJ
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{pvec_sv[iwf], w_sv[iwf]};
    fptype* amp_fp;
    amp_fp = reinterpret_cast<fptype*>( amp_sv );""")
            if fd_gauge:
                ret_lines.append("""
    // special temporary ALOHAOBJ to hold F/Vtmp values in the combined vertex functions while using the FD gauge
    fptype_sv pvec_sv_tmp[1][np4];
    cxtype_sv w_sv_tmp[1][nw6]; 
    ALOHAOBJ aloha_obj_tmp[1];
    aloha_obj_tmp[0] = ALOHAOBJ{pvec_sv_tmp[0], w_sv_tmp[0]};
    
    // special one value to hold tmp vertex value inside the combined vertex functions while using the FD gauge
    cxtype_sv amp_tmp_sv[1]; //to ensure proper aligment for vector instructions
    fptype* amp_tmp_fp;
    amp_tmp_fp = reinterpret_cast<fptype*>( amp_tmp_sv );
    """)
            ret_lines.append("""
    // Local variables for the given CUDA event (ievt) or C++ event page (ipagV)
    // [jamp: sum (for one event or event page) of the invariant amplitudes for all Feynman diagrams in a given color combination]
    cxtype_sv jamp_sv[ncolor] = {}; // all zeros (NB: vector cxtype_v IS initialized to 0, but scalar cxtype is NOT, if "= {}" is missing!)

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes         ===
    // === (for one event in CUDA, for one - or two in mixed mode - SIMD event pages in C++ ===

    // START LOOP ON IPARITY
    for( int iParity = 0; iParity < nParity; ++iParity )
    {
#ifndef MGONGPUCPP_GPUIMPL
      const int ievt0 = ievt00 + iParity * neppV;
#endif""")
            ret_lines += helas_calls
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        #ret_lines.extend([self.get_matrix_single_process(i, me,
        #                                                 color_amplitudes[i],
        #                                                 class_name) \
        #                        for i, me in enumerate(self.matrix_elements)])
        file_extend = []
        for i, me in enumerate(self.matrix_elements):
            file = self.get_matrix_single_process( i, me, color_amplitudes[i], class_name )
            file = strip_banner(file, banner_mark = "!") # skip first 8 lines in process_matrix.inc (copyright)
            file_extend.append( file )
            assert i == 0, "more than one ME in get_all_sigmaKin_lines" # AV sanity check (added for color_sum.cc but valid independently)
        ret_lines.extend( file_extend )
        return '\n'.join(ret_lines)

    # AV - modify export_cpp.OneProcessExporterCPP method (replace '# Process' by '// Process')
    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""
        ###return'\n'.join([ '# ' + process.nice_string().replace('\n', '\n# * ') \
        ###                 for process in matrix_element.get('processes')])
        return'\n'.join([ '// ' + process.nice_string().replace('\n', '\n// * ') \
                         for process in matrix_element.get('processes')])

    # AV - replace the export_cpp.OneProcessExporterCPP method (invert .cc/.cu, add debug printouts)
    def generate_process_files(self):
        """Generate mgOnGpuConfig.h, CPPProcess.cc, CPPProcess.h, check_sa.cc, gXXX.cu links"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.generate_process_files')
        self.edit_mgonGPU()
        self.edit_processidfile() # AV new file (NB this is Sigma-specific, should not be a symlink to Subprocesses)
        self.edit_processConfig() # sub process specific, not to be symlinked from the Subprocesses directory
        self.edit_colorsum() # AV new file (NB this is Sigma-specific, should not be a symlink to Subprocesses)
        self.edit_coloramps()
        self.edit_memorybuffers() # AV new file (NB this is generic in Subprocesses and then linked in Sigma-specific)
        self.edit_memoryaccesscouplings() # AV new file (NB this is generic in Subprocesses and then linked in Sigma-specific)
        super().generate_process_files()
        # NB: symlink of cudacpp.mk to makefile is overwritten by madevent makefile if this exists (#480)
        # NB: this relies on the assumption that cudacpp code is generated before madevent code
        files.ln(pjoin(self.path, "..", "makefile"), self.path, "makefile")

    # AV - replace the export_cpp.OneProcessExporterCPP method (add debug printouts and multichannel handling #473) 
    def edit_mgonGPU(self):
        """Generate mgOnGpuConfig.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_mgonGPU')
        template = open(pjoin(self.template_path,'madmatrix','mgOnGpuConfig.h'),'r').read()
        replace_dict = {}
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        replace_dict['nbhel'] = self.matrix_elements[0].get_helicity_combinations() # number of helicity combinations
        ###replace_dict['nwavefunc'] = self.matrix_elements[0].get_number_of_wavefunctions() # this is the correct P1-specific nwf, now in CPPProcess.h (#644)
        replace_dict['wavefuncsize'] = 6
        ff = open(pjoin(self.path, '..','..','src','mgOnGpuConfig.h'),'w')
        ff.write(template % replace_dict)
        ff.close()

    # AV - new method
    def edit_processidfile(self):
        """Generate epoch_process_id.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_processidfile')
        template = open(pjoin(self.template_path,'madmatrix','epoch_process_id.h'),'r').read()
        replace_dict = {}
        replace_dict['processid'] = self.name
        replace_dict['processid_uppercase'] = self.name.upper()
        ff = open(pjoin(self.path, 'epoch_process_id.h'),'w')
        ff.write(template % replace_dict)
        ff.close()

    # AV - new method
    def edit_colorsum(self):
        """Generate color_sum.cc"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_colorsum')
        template = open(pjoin(self.template_path,'madmatrix','color_sum.cc'),'r').read()
        replace_dict = {}
        # Extract color matrix again (this was also in get_matrix_single_process called within get_all_sigmaKin_lines)
        replace_dict['color_matrix_lines'] = self.get_color_matrix_lines(self.matrix_elements[0])
        ff = open(pjoin(self.path, 'color_sum.cc'),'w')
        ff.write(template % replace_dict)
        ff.close()
        
    def edit_processConfig(self):
        """Generate process_config.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_processConfig')
        template = open(pjoin(self.template_path,'madmatrix','processConfig.h'),'r').read()
        replace_dict = {}
        replace_dict['ndiagrams'] = len(self.matrix_elements[0].get('diagrams'))
        replace_dict['processid_uppercase'] = self.name.upper()
        ff = open(pjoin(self.path, 'processConfig.h'),'w')
        ff.write(template % replace_dict)
        ff.close()

    # AV - new method
    def edit_coloramps(self):
        """Generate coloramps.h"""

        # we don't sort self.multi_channel_map, and we rely on MadSpace sorting
        # so, diagrams there may be unsorted
        config_subproc_map_C = self.multi_channel_map.values() # this has C-indexing
        config_subproc_map = [] # this has Fortran indexing
        for config in config_subproc_map_C:
            config_subproc_map.append([c+1 for c in config])

        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_coloramps')
        template = open(pjoin(self.template_path,'madmatrix','coloramps.h'),'r').read()
        ff = open(pjoin(self.path, 'coloramps.h'),'w')
        # The following five lines from OneProcessExporterCPP.get_sigmaKin_lines (using OneProcessExporterCPP.get_icolamp_lines)
        replace_dict={}

        iconfig_to_diag = {}
        diag_to_iconfig = {}
        iconfig = 0
        for config in config_subproc_map:
            if set(config) == set([0]):
                continue
            iconfig += 1
            iconfig_to_diag[iconfig] = config[0] 
            diag_to_iconfig[config[0]] = iconfig

        misc.sprint(iconfig_to_diag)
        misc.sprint(diag_to_iconfig)

        # Note that if the last diagram is/are not mapped to a channel nb_diag 
        # will be smaller than the true number of diagram. This is fine for color
        # but maybe not for something else.
        nb_diag = max(config[0] for config in config_subproc_map)
        ndigits = str(int(math.log10(nb_diag))+1+1) # the additional +1 is for the -sign
        # Output which diagrams correspond ot a channel to get information for valid color
        lines = []
        for diag in range(1, nb_diag+1):
            if diag in diag_to_iconfig:
                iconfigf = diag_to_iconfig[diag]
                iconfigftxt = '%i'%iconfigf
            else:
                iconfigf = -1
                iconfigftxt = '-1 (diagram with no associated iconfig for single-diagram enhancement)'
            text = '    %(iconfigf){0}i, // DIAGRAM=%(diag)-{0}i --> ICONFIG=%(iconfigftxt)s'.format(ndigits)
            lines.append(text % {'diag':diag-1, 'iconfigf':iconfigf, 'iconfigftxt':iconfigftxt}) # diag - 1 is to follow MadSpace indexing
        replace_dict['channelc2iconfig_lines'] = '\n'.join(lines)

        replace_dict['nb_channel'] = len(self.active_color_map)
        # here I can do the conversion in between the active color map and true, false, and obtain a C++ compatible thing immediately
        replace_dict['nb_diag'] = nb_diag
        nb_color = max(1, len(self.color_basis))
        replace_dict['nb_color'] = nb_color
        # AV extra formatting (e.g. gg_tt was "{{true,true};,{true,false};,{false,true};};")
        ###misc.sprint(replace_dict['is_LC'])
        text=', // ICONFIG=%-{0}i <-- DIAGRAM=%i'.format(ndigits)
        icolamp = []
        for iconfigc, active in enumerate(self.active_color_map):
            icolamp_text = "    { " + ", ".join(["true" if i in active else "false" for i in range(nb_color)]) + "}"
            icolamp_text += text % (iconfigc+1, iconfig_to_diag[iconfigc+1]-1) # diag - 1 is to follow MadSpace indexing
            icolamp.append(icolamp_text)
        replace_dict['is_LC'] = '\n'.join(icolamp)
        ff.write(template % replace_dict)
        ff.close()

    # AV - new method
    def edit_memorybuffers(self):
        """Generate MemoryBuffers.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_memorybuffers')
        template = open(pjoin(self.template_path,'madmatrix','MemoryBuffers.h'),'r').read()
        replace_dict = {}
        replace_dict['model_name'] = self.model_name
        ff = open(pjoin(self.path, '..', 'MemoryBuffers.h'),'w')
        ff.write(template % replace_dict)
        ff.close()

    # AV - new method
    def edit_memoryaccesscouplings(self):
        """Generate MemoryAccessCouplings.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_memoryaccesscouplings')
        template = open(pjoin(self.template_path,'madmatrix','MemoryAccessCouplings.h'),'r').read()
        replace_dict = {}
        replace_dict['model_name'] = self.model_name
        ff = open(pjoin(self.path, '..', 'MemoryAccessCouplings.h'),'w')
        ff.write(template % replace_dict)
        ff.close()

    # AV - overload the export_cpp.OneProcessExporterCPP method (add debug printout and truncate last \n)
    # [*NB export_cpp.UFOModelConverterGPU.write_process_h_file is not called!*]
    def write_process_h_file(self, writer):
        """Generate final CPPProcess.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.write_process_h_file')
        replace_dict = super().write_process_h_file(False)
        try:
            replace_dict['helamps_h'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.h' % self.model_name)).read()
        except FileNotFoundError:
            replace_dict['helamps_h'] = "\n#include \"../../src/HelAmps_%s.h\"" % self.model_name

        replace_dict['process_file_name'] = self.name.upper()

        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file, formatting=False)
        else:
            return replace_dict

    # AV - replace the export_cpp.OneProcessExporterCPP method (replace HelAmps.cu by HelAmps.cc)
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process described by matrix_element"""
        replace_dict = super().write_process_cc_file(False)
        ###replace_dict['hel_amps_def'] = '\n#include \"../../src/HelAmps_%s.cu\"' % self.model_name
        replace_dict['hel_amps_h'] = '#include \"HelAmps_%s.h\"' % self.model_name # AV
        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file, formatting=False)
        else:
            return replace_dict

    # AV - replace the export_cpp.OneProcessExporterCPP method (fix fptype and improve formatting)
    def get_color_matrix_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split rows in chunks of size n."""
        import madgraph.core.color_algebra as color
        if not matrix_element.get('color_matrix'):
            return '\n'.join(['  static constexpr fptype2 colorDenom[1] = {1.};', 'static const fptype2 cf[1][1] = {1.};'])
        else:
            color_denominators = matrix_element.get('color_matrix').\
                                                 get_line_denominators()
            denom_string = '  static constexpr fptype2 colorDenom[ncolor] = { %s }; // 1-D array[%i]' \
                           % ( ', '.join(['%i' % denom for denom in color_denominators]), len(color_denominators) )
            matrix_strings = []
            for index, denominator in enumerate(color_denominators):
                # Then write the numerators for the matrix elements
                num_list = matrix_element.get('color_matrix').get_line_numerators(index, denominator)
                matrix_strings.append('{ %s }' % ', '.join(['%d' % i for i in num_list]))
            matrix_string = '  static constexpr fptype2 colorMatrix[ncolor][ncolor] = '
            if len( matrix_strings ) > 1:
                matrix_string += '{\n    ' + ',\n    '.join(matrix_strings) + ' };'
            else:
                matrix_string += '{ ' + matrix_strings[0] + ' };'
            matrix_string += ' // 2-D array[%i][%i]' % ( len(color_denominators), len(color_denominators) )
            denom_comment = '\n  // The color denominators (initialize all array elements, with ncolor=%i)\n  // [NB do keep \'static\' for these constexpr arrays, see issue #283]\n' % len(color_denominators)
            matrix_comment = '\n  // The color matrix (initialize all array elements, with ncolor=%i)\n  // [NB do keep \'static\' for these constexpr arrays, see issue #283]\n' % len(color_denominators)
            denom_string = denom_comment + denom_string
            matrix_string = matrix_comment + matrix_string
            return '\n'.join([denom_string, matrix_string])

    # AV - replace the export_cpp.OneProcessExporterCPP method (improve formatting)
    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for CPPProcess::initProc"""
        initProc_lines = []
        initProc_lines.append('// Set external particle masses for this matrix element')
        for part in matrix_element.get_external_wavefunctions():
            ###initProc_lines.append('mME.push_back(pars->%s);' % part.get('mass'))
            initProc_lines.append('    m_masses.push_back( m_pars->%s );' % part.get('mass')) # AV
        ###for i, colamp in enumerate(color_amplitudes):
        ###    initProc_lines.append('jamp2_sv[%d] = new double[%d];' % (i, len(colamp))) # AV - this was commented out already
        return '\n'.join(initProc_lines)

    # AV - replace the export_cpp.OneProcessExporterCPP method (fix helicity order and improve formatting)
    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""
        helicity_line = '    static constexpr short helicities[ncomb][npar] = {\n      '; # AV (this is tHel)
        helicity_line_list = []
        for helicities in matrix_element.get_helicity_matrix(allow_reverse=True): # AV was False: different order in Fortran and cudacpp! #569
            helicity_line_list.append( '{ ' + ', '.join(['%d'] * len(helicities)) % tuple(helicities) + ' }' ) # AV
        return helicity_line + ',\n      '.join(helicity_line_list) + ' };' # AV

    def get_flavor_matrix(self, matrix_element):
        """Return the flavor matrix definition lines for this matrix element"""
        # Emitted at namespace scope (file-local linkage), so the host-side table
        # is visible to both the CPPProcess constructor and CPPProcess::flavorPDG.
        flavor_line = '  static constexpr short flavors[nmaxflavor][npar] = {\n    '; # (this is tFlavors)
        flavor_line_list = []
        for flavors in matrix_element.get_external_flavors_with_iden():
            # get only the index 0 one because the other ones have same matrix element
            # additionally they will be used as indices in some cases (e.g. matrix flavor couplings)
            # so we need to subtract 1 because FORTRAN indices starts from 1, and C++ from zero
            cpp_flavors = list(map(lambda f: f-1, flavors[0]))
            flavor_line_list.append( '{ ' + ', '.join(['%d'] * len(cpp_flavors)) % tuple(cpp_flavors) + ' }' )
        out = flavor_line + ',\n    '.join(flavor_line_list) + ' };'
        # Companion table with the true signed PDG ids of each flavor
        # combination (same convention as the PDG lines printed by the
        # Fortran/C++ standalone 'check' drivers); used by CPPProcess::flavorPDG.
        model = matrix_element.get('processes')[0].get('model')
        merged = model.get('merged_particles') or {}
        all_pdgs = [l.get('id') for l in
                    matrix_element.get('processes')[0].get('legs_with_decays')]
        pdg_line_list = []
        for flavors in matrix_element.get_external_flavors_with_iden():
            row = []
            for j, flv in enumerate(flavors[0]):
                raw = all_pdgs[j]
                if abs(raw) in merged:
                    row.append(flv if raw >= 0 else -flv)
                else:
                    row.append(raw)
            pdg_line_list.append( '{ ' + ', '.join('%d' % v for v in row) + ' }' )
        out += ('\n  static constexpr int flavorPDGs[nmaxflavor][npar] = {\n    ' +
                ',\n    '.join(pdg_line_list) + ' };')
        return out

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""
        ret_lines = ""
        return ret_lines

#------------------------------------------------------------------------------------

import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.helas_call_writers as helas_call_writers

# AV - define a custom HelasCallWriter
# (NB: enable this via ProcessExporterMadMatrix.helas_exporter in output.py - this fixes #341)
class MadMatrixUFOHelasCallWriter(helas_call_writers.GPUFOHelasCallWriter):
    """ A Custom HelasCallWriter """

    # Flavor-mask optimization: skip wavefunction/amplitude calls that vanish
    # for the selected flavor (see super_get_matrix_element_calls). Toggled by
    # the output command's --mask=True|False; default on.
    use_flavor_mask = True
    # Class structure information
    #  - object
    #  - dict(object) [built-in]
    #  - PhysicsObject(dict) [in madgraph/core/base_objects.py]
    #  - HelasCallWriter(base_objects.PhysicsObject) [in madgraph/iolibs/helas_call_writers.py]
    #  - UFOHelasCallWriter(HelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - CPPUFOHelasCallWriter(UFOHelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - GPUFOHelasCallWriter(CPPUFOHelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - MadMatrixUFOHelasCallWriter(GPUFOHelasCallWriter)
    #      This class


    def __init__(self, *args, **opts):

        self.wanted_ordered_dep_couplings = []
        self.wanted_ordered_indep_couplings = []
        self.wanted_ordered_flv_couplings = []

        # the following is based on the fact that model = args[0]
        # builds a map FLV_Coupling name : value to be used when writing
        # the HELAS
        model = args[0]
        self.flv_couplings_map = {}
        for interaction in model.interactions:
            all_couplings = interaction["couplings"]
            for coupling in all_couplings.values():
                if not isinstance(coupling, base_objects.FLV_Coupling):
                    continue
                self.flv_couplings_map[coupling.name] = coupling

        super().__init__(*args,**opts)


    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (improve formatting of CPPProcess.cc)
    # [GPUFOHelasCallWriter.format_coupling is called by GPUFOHelasCallWriter.get_external_line/generate_helas_call]
    # [GPUFOHelasCallWriter.get_external_line is called by GPUFOHelasCallWriter.get_external]
    # [GPUFOHelasCallWriter.get_external (adding #ifdef CUDA) is called by GPUFOHelasCallWriter.generate_helas_call]
    # [GPUFOHelasCallWriter.generate_helas_call is called by UFOHelasCallWriter.get_wavefunction_call/get_amplitude_call]
    ###findcoupling = re.compile('pars->([-]*[\d\w_]+)\s*,')
    def format_coupling(self, call):
        """Format the coupling so any minus signs are put in front"""
        import re
        ###print(call) # FOR DEBUGGING
        model = self.get('model')
        newcoup = False
        if not hasattr(self, 'couplings2order'):
            self.couplings2order = {}
            self.params2order = {}
        if not hasattr(self, 'couporderdep'):
            self.couporderdep = {}
            self.couporderindep = {}
            self.couporderflv = {}
        for coup in re.findall(self.findcoupling, call):
            if coup == 'ZERO':
                ###call = call.replace('pars->ZERO', '0.')
                call = call.replace('m_pars->ZERO', '0.') # AV
                continue
            sign = ''
            if coup.startswith('-'):
                sign = '-'
                coup = coup[1:]
            try:
                param = model.get_parameter(coup)
            except KeyError:
                param = False
            if param:
                alias = self.params2order
                aliastxt = 'PARAM'
                name = 'cIPD'
            elif model.is_running_coupling(coup):
                if coup not in self.wanted_ordered_dep_couplings: 
                    self.wanted_ordered_dep_couplings.append(coup)
                alias = self.couporderdep
                aliastxt = 'COUPD'
                name = 'cIPC'
            elif coup.startswith("FLV"):
                if coup not in [coup.name for coup in self.wanted_ordered_flv_couplings]:
                    flv_coup = self.flv_couplings_map[coup]
                    self.wanted_ordered_flv_couplings.append(flv_coup)
                    for indep_coup in set(flv_coup.flavors.values()):
                        if indep_coup not in self.wanted_ordered_indep_couplings:
                            self.wanted_ordered_indep_couplings.append(indep_coup)
                alias = self.couporderflv
                aliastxt = 'flvCOUP'
                name = 'flvCOUPs'
            else:
                if coup not in self.wanted_ordered_indep_couplings: 
                    self.wanted_ordered_indep_couplings.append(coup)
                alias = self.couporderindep
                aliastxt = 'COUPI'
                name = 'cIPC'
            if coup not in alias:
                ###if alias == self.couporderindep: # bug #821! this is incorrectly true when both dictionaries are empty!
                if aliastxt == 'COUPI':
                    if not len(alias):
                        alias[coup] = len(self.couporderdep)
                    else:
                        alias[coup] = alias[list(alias)[-1]]+1
                else:
                    alias[coup] = len(alias) # this works perfectly also for FLV couplings
                ###if alias == self.couporderdep: # bug #821! this is incorrectly true when both dictionaries are empty!
                if aliastxt == 'COUPD':
                    for k in self.couporderindep:
                        self.couporderindep[k] += 1
                newcoup = True
            if name == 'cIPD':
                call = call.replace('m_pars->%s%s' % (sign, coup),
                                    '%s%s[%s]' % (sign, name, alias[coup]))                        
            elif model.is_running_coupling(coup):
                ###call = call.replace('m_pars->%s%s' % (sign, coup),
                ###                    '%scxmake( cIPC[%s], cIPC[%s] )' %
                ###                    (sign, 2*alias[coup],2*alias[coup]+1))
                ###misc.sprint(name, alias[coup])
                # AV from cIPCs to COUP array (running alphas #373)
                # OM fix handling of 'unary minus' #628
                call = call.replace('CI_ACCESS', 'CD_ACCESS')
                call = call.replace('m_pars->%s%s' % (sign, coup),
                                    'COUPs[%s], %s' % (alias[coup], '1.0' if not sign else '-1.0')) 
            elif name == 'flvCOUPs':
                call = call.replace('CD_ACCESS', 'CI_ACCESS')
                call = call.replace('m_pars->%s%s' % (sign, coup),
                                    '%s[%s], %s' % (name, alias[coup], '1.0' if not sign else '-1.0'))
            else:
                call = call.replace('CD_ACCESS', 'CI_ACCESS')
                call = call.replace('m_pars->%s%s' % (sign, coup),
                                    'COUPs[ndcoup + %s], %s' % (alias[coup]-len(self.couporderdep), '1.0' if not sign else '-1.0'))

            if newcoup:
                self.couplings2order = self.couporderdep | self.couporderindep
        model.cudacpp_wanted_ordered_couplings = self.wanted_ordered_dep_couplings + self.wanted_ordered_indep_couplings + self.wanted_ordered_flv_couplings
        return call

    # AV - new method for formatting wavefunction/amplitude calls
    # [It would be too complex to modify them in helas_objects.HelasWavefunction/Amplitude.get_call_key]
    @staticmethod
    def format_call(call):
        return call.replace('(','( ').replace(')',' )').replace(',',', ')

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (improve formatting)
    def super_get_matrix_element_calls(self, matrix_element, color_amplitudes, multi_channel_map):
        """Return a list of strings, corresponding to the Helas calls for the matrix element"""
        import madgraph.core.helas_objects as helas_objects
        import madgraph.loop.loop_helas_objects as loop_helas_objects
        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
               '%s not valid argument for get_matrix_element_calls' % \
               type(matrix_element)
        # Do not reuse the wavefunctions for loop matrix elements
        if isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement):
            return self.get_loop_matrix_element_calls(matrix_element)
        # Restructure data for easier handling
        color = {}
        for njamp, coeff_list in enumerate(color_amplitudes):
            for coeff, namp in coeff_list:
                if namp not in color:
                    color[namp] = {}
                color[namp][njamp] = coeff
        me = matrix_element.get('diagrams')
        matrix_element.reuse_outdated_wavefunctions(me)
        ###misc.sprint(multi_channel_map)
        res = []
        ###res.append('for(int i=0;i<%s;i++){jamp[i] = cxtype(0.,0.);}' % len(color_amplitudes))
        res.append("""//constexpr size_t nxcoup = ndcoup + nicoup; // both dependent and independent couplings (BUG #823)
      constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings (FIX #823)
      const fptype* allCOUPs[nxcoup];
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#pragma nv_diagnostic push
#pragma nv_diag_suppress 186 // e.g. <<warning #186-D: pointless comparison of unsigned integer with zero>>
#endif
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
      //for( size_t iicoup = 0; iicoup < nicoup; iicoup++ )                             // BUG #823
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )                                 // FIX #823
        allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
#ifdef MGONGPUCPP_GPUIMPL
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#pragma nv_diagnostic pop
#endif
      // CUDA kernels take input/output buffers with momenta/MEs for all events
      const fptype* momenta = allmomenta;
      const fptype* COUPs[nxcoup];
      for( size_t ixcoup = 0; ixcoup < nxcoup; ixcoup++ ) COUPs[ixcoup] = allCOUPs[ixcoup];
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      fptype* numerators = &allNumerators[ievt * processConfig::ndiagrams];
      fptype* denominators = allDenominators;
#else
      // C++ kernels take input/output buffers with momenta/MEs for one specific event (the first in the current event page)
      const fptype* momenta = M_ACCESS::ieventAccessRecordConst( allmomenta, ievt0 );
      const fptype* COUPs[nxcoup];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        COUPs[idcoup] = CD_ACCESS::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 ); // dependent couplings, vary event-by-event
      //for( size_t iicoup = 0; iicoup < nicoup; iicoup++ ) // BUG #823
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )     // FIX #823
        COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
      fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * processConfig::ndiagrams );
      fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
#endif
      // Create an array of views over the Flavor Couplings
      FLV_COUPLING_ARRAY<nIPF, nMF> flvCOUPs{ cIPF_partner1, cIPF_partner2, cIPF_value };

      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i = 0; i < ncolor; i++ ) { jamp_sv[i] = cxzero_sv(); }

      // Numerators and denominators for the current event (CUDA) or SIMD event page (C++)
      fptype_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      // Scalar iflavor for the current event
      // for GPU it is an int
      // for SIMD it is also an int, since it is constant across the SIMD vector
#ifdef MGONGPUCPP_GPUIMPL
      const unsigned int iflavor = F_ACCESS::kernelAccessConst( iflavorVec );
#else
      const unsigned int* iflavor_rec = F_ACCESS::ieventAccessRecordConst( iflavorVec, ievt0 );
      const uint_sv iflavor_sv = F_ACCESS::kernelAccessConst( iflavor_rec );
      const unsigned int iflavor = reinterpret_cast<const unsigned int*>(&iflavor_sv)[0];
#endif
""")
        diagrams = matrix_element.get('diagrams')
        diag_to_config = {}
        for config in sorted(multi_channel_map.keys()):
            amp = [a.get('number') for a in \
                              sum([diagrams[idiag].get('amplitudes') for \
                                   idiag in multi_channel_map[config]], [])]
            diag_to_config[amp[0]] = config
        ###misc.sprint(diag_to_config)

        # Flavor masking (--mask): when merged-particle flavors give some
        # diagrams that vanish for some flavor combinations, guard the
        # corresponding wavefunction / amplitude calls with a runtime test on
        # the scalar iflavor so they are skipped entirely (rather than computed
        # and multiplied by a zero coupling). The runtime iflavor indexes the
        # grouped flavors (get_external_flavors_with_iden), so the per-flavor
        # masks set by compute_flavor_masks() are compressed onto those groups.
        # NB: this relies on iflavor being constant across a SIMD vector.
        diag_group_mask, wf_group_mask = self._compute_group_masks(matrix_element)

        def _guard_open(group_mask):
            # Emit the opening of an `if` guard for a non-full grouped mask.
            return 'if( ( 0x%xULL >> iflavor ) & 0x1ULL ) {' % group_mask

        id_amp = 0
        for diagram in matrix_element.get('diagrams'):
            ###print('DIAGRAM %3d: #wavefunctions=%3d, #diagrams=%3d' %
            ###      (diagram.get('number'), len(diagram.get('wavefunctions')), len(diagram.get('amplitudes')) )) # AV - FOR DEBUGGING
            res.append('\n      // *** DIAGRAM %d OF %d ***' % (diagram.get('number'), len(matrix_element.get('diagrams'))) ) # AV
            res.append('\n      // Wavefunction(s) for diagram number %d' % diagram.get('number')) # AV
            for wf in diagram.get('wavefunctions'):
                call = self.get_wavefunction_call(wf)
                if call and call[-1] == '\n': call = call[:-1]
                gmask = wf_group_mask.get(id(wf))
                if gmask is not None:
                    res.append(_guard_open(gmask))
                    res.append(call)
                    res.append('}')
                else:
                    res.append(call)
            if len(diagram.get('wavefunctions')) == 0 : res.append('// (none)') # AV
            res.append('\n      // Amplitude(s) for diagram number %d' % diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                id_amp +=1
                namp = amplitude.get('number')
                amplitude.set('number', 1)
                amp_block = [ self.get_amplitude_call(amplitude) ] # AV new: avoid format_call
                if id_amp in diag_to_config:
                    ###res.append("if( channelId == %i ) numerators_sv += cxabs2( amp_sv[0] );" % diag_to_config[id_amp]) # BUG #472
                    ###res.append("if( channelId == %i ) numerators_sv += cxabs2( amp_sv[0] );" % id_amp) # wrong fix for BUG #472
                    diagnum = diagram.get('number')
                    amp_block.append("if( storeChannelWeights )")
                    amp_block.append("{")
                    amp_block.append("  numerators_sv[%i] += cxabs2( amp_sv[0] );" % (diagnum-1))
                    amp_block.append("  denominators_sv += cxabs2( amp_sv[0] );")
                    amp_block.append("}")
                for njamp, coeff in color[namp].items():
                    scoeff = OneProcessExporterMadMatrix.coeff(*coeff) # AV
                    if scoeff[0] == '+' : scoeff = scoeff[1:]
                    scoeff = scoeff.replace('(','( ')
                    scoeff = scoeff.replace(')',' )')
                    scoeff = scoeff.replace(',',', ')
                    scoeff = scoeff.replace('*',' * ')
                    scoeff = scoeff.replace('/',' / ')
                    if scoeff.startswith('-'): amp_block.append('jamp_sv[%s] -= %samp_sv[0];' % (njamp, scoeff[1:])) # AV
                    else: amp_block.append('jamp_sv[%s] += %samp_sv[0];' % (njamp, scoeff)) # AV
                # The amplitude (and the jamp/channel contributions that read its
                # amp_sv[0]) only contributes for the flavors in the diagram's
                # mask, so guard the whole block as a unit.
                gmask = diag_group_mask.get(id(diagram))
                if gmask is not None:
                    res.append(_guard_open(gmask))
                    res.extend(amp_block)
                    res.append('}')
                else:
                    res.extend(amp_block)
            if len(diagram.get('amplitudes')) == 0 : res.append('// (none)') # AV
        ###res.append('\n    // *** END OF DIAGRAMS ***' ) # AV - no longer needed ('COLOR MATRIX BELOW')
        return res

    # AV/OM - compute per-diagram and per-wavefunction flavor masks, projected
    # onto the grouped flavors used at runtime (iflavor). Returns two dicts
    # keyed by id(object) -> grouped bitmask, containing only objects whose mask
    # is NOT all-ones (i.e. that actually benefit from a guard). Both dicts are
    # empty when masking is disabled (use_flavor_mask False), trivial, or the
    # process has more than 64 flavor groups.
    def _compute_group_masks(self, matrix_element):
        if not getattr(self, 'use_flavor_mask', True):
            return {}, {}
        try:
            allowed = matrix_element.compute_flavor_masks()
        except Exception:
            return {}, {}
        if not allowed or matrix_element.flavor_mask_is_trivial():
            return {}, {}
        groups = [list(g) for g in matrix_element.get_external_flavors_with_iden()]
        n_groups = len(groups)
        if n_groups == 0 or n_groups > 64:
            return {}, {}
        flavor_to_idx = {tuple(f): i for i, f in enumerate(allowed)}
        group_bits = []
        for g in groups:
            bits = 0
            for fl in g:
                bits |= (1 << flavor_to_idx[tuple(fl)])
            group_bits.append(bits)
        full = (1 << n_groups) - 1

        def to_group(perflavor_mask):
            gm = 0
            for gi, gb in enumerate(group_bits):
                if perflavor_mask & gb:
                    gm |= (1 << gi)
            return gm

        diag_group_mask = {}
        for diag in matrix_element.get('diagrams'):
            if 'flavor_mask' not in diag:
                continue
            gm = to_group(diag['flavor_mask'])
            if gm != full:
                diag_group_mask[id(diag)] = gm
        wf_group_mask = {}
        for wf in matrix_element.get_all_wavefunctions():
            if 'flavor_mask' not in wf:
                continue
            gm = to_group(wf['flavor_mask'])
            if gm != full:
                wf_group_mask[id(wf)] = gm
        return diag_group_mask, wf_group_mask

    # AV - overload helas_call_writers.GPUFOHelasCallWriter method (improve formatting)
    def get_matrix_element_calls(self, matrix_element, color_amplitudes, multi_channel_map):
        """Return a list of strings, corresponding to the Helas calls for the matrix element"""
        res = self.super_get_matrix_element_calls(matrix_element, color_amplitudes, multi_channel_map)
        for i, item in enumerate(res):
            ###print(item) # FOR DEBUGGING
            if item.startswith('# Amplitude'): item='//'+item[1:] # AV replace '# Amplitude' by '// Amplitude'
            if not item.startswith('\n') and not item.startswith('#'): res[i]='      '+item
        return res

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (improve formatting)
    # [GPUFOHelasCallWriter.format_coupling is called by GPUFOHelasCallWriter.get_external_line/generate_helas_call]
    # [GPUFOHelasCallWriter.get_external_line is called by GPUFOHelasCallWriter.get_external]
    # [=> GPUFOHelasCallWriter.get_external is called by GPUFOHelasCallWriter.generate_helas_call]
    # [GPUFOHelasCallWriter.generate_helas_call is called by UFOHelasCallWriter.get_wavefunction_call/get_amplitude_call]
    first_get_external = True
    def get_external(self, wf, argument):
        line = self.get_external_line(wf, argument)
        split_line = line.split(',')
        split_line = [ str.lstrip(' ').rstrip(' ') for str in split_line] # AV
        # (AV join using ',': no need to add a space as this is done by format_call later on)
        line = ', '.join(split_line)
        line = line.replace( 'xxx(', 'xxx<M_ACCESS, W_ACCESS>(' )
        line = line.replace( 'w_sv', 'w_fp' )
        # AV2: line2 logic is to have MGONGPU_TEST_DIVERGENCE on the first xxx call
        if self.first_get_external and ( ( 'mzxxx' in line ) or ( 'pzxxx' in line ) or ( 'xzxxx' in line ) ) :
            self.first_get_external = False
            line2 = line.replace('mzxxx','xxxxx').replace('pzxxx','xxxxx').replace('xzxxx','xxxxx')
            line2 = line2[:line2.find('// NB')]
            split_line2 = line2.split(',')
            split_line2 = [ str.lstrip(' ').rstrip(' ') for str in split_line2] # AV
            split_line2.insert(2, '0') # add parameter fmass=0
            line2 = ', '.join(split_line2)
            text = '#if not( defined MGONGPUCPP_GPUIMPL and defined MGONGPU_TEST_DIVERGENCE )\n      %s\n#else\n      if( ( blockDim.x * blockIdx.x + threadIdx.x ) %% 2 == 0 )\n        %s\n      else\n        %s\n#endif\n' # AV
            return text % (line, line, line2)
        text = '%s\n' # AV
        return text % line

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (vectorize w_sv)
    # This is the method that creates the ixxx/oxxx function calls in calculate_wavefunctions
    # [GPUFOHelasCallWriter.get_external_line is called by GPUFOHelasCallWriter.get_external]
    # [GPUFOHelasCallWriter.get_external (adding #ifdef CUDA) is called by GPUFOHelasCallWriter.generate_helas_call]
    # [GPUFOHelasCallWriter.generate_helas_call is called by UFOHelasCallWriter.get_wavefunction_call/get_amplitude_call]
    def get_external_line(self, wf, argument):
        call = ''
        call = call + helas_call_writers.HelasCallWriter.mother_dict[\
            argument.get_spin_state_number()].lower()
        # Fill out with X up to 6 positions
        call = call + 'x' * (6 - len(call))
        # Specify namespace for Helas calls
        call = call + '( momenta,'
        if argument.get('spin') != 1:
            # For non-scalars, need mass and helicity
            call = call + 'm_pars->%s, cHel[ihel][%d],'
        else:
            # AV This seems to be for scalars (spin==1???), pass neither mass nor helicity (#351)
            ###call = call + 'm_pars->%s,'
            call = call
        # Add flavor and the related ALOHA object
        call = call + '%+d, cFlavors[iflavor][%d], aloha_obj[%d], %d );'
        if argument.get('spin') == 1:
            # AV This seems to be for scalars (spin==1???), pass neither mass nor helicity (#351)
            return call % \
                            (
                                ###wf.get('mass'),
                                # For boson, need initial/final here
                                (-1) ** (wf.get('state') == 'initial'),
                                wf.get('me_id')-1,
                                wf.get('number_external')-1,
                                wf.get('number_external')-1)
        elif argument.is_boson():
            ###misc.sprint(call)
            ###misc.sprint( (wf.get('mass'),
            ###                     wf.get('number_external')-1,
            ###                     # For boson, need initial/final here
            ###                     (-1) ** (wf.get('state') == 'initial'),
            ###                     wf.get('me_id')-1,
            ###                     wf.get('number_external')-1))
            return  self.format_coupling(call % \
                            (wf.get('mass'),
                                wf.get('number_external')-1,
                                # For boson, need initial/final here
                                (-1) ** (wf.get('state') == 'initial'),
                                wf.get('number_external')-1,
                                wf.get('me_id')-1,
                                wf.get('number_external')-1))
        else:
            return self.format_coupling(call % \
                            (wf.get('mass'),
                                wf.get('number_external')-1,
                                # For fermions, need particle/antiparticle
                                - (-1) ** wf.get_with_flow('is_part'),
                                wf.get('number_external')-1,
                                wf.get('me_id')-1,
                                wf.get('number_external')-1))

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (vectorize w_sv and amp_sv)
    def generate_helas_call(self, argument):
        """Routine for automatic generation of C++ Helas calls
        according to just the spin structure of the interaction.

        First the call string is generated, using a dictionary to go
        from the spin state of the calling wavefunction and its
        mothers, or the mothers of the amplitude, to difenrentiate wich call is
        done.

        Then the call function is generated, as a lambda which fills
        the call string with the information of the calling
        wavefunction or amplitude. The call has different structure,
        depending on the spin of the wavefunction and the number of
        mothers (multiplicity of the vertex). The mother
        wavefunctions, when entering the call, must be sorted in the
        correct way - this is done by the sorted_mothers routine.

        Finally the call function is stored in the relevant
        dictionary, in order to be able to reuse the function the next
        time a wavefunction with the same Lorentz structure is needed.
        """
        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError('get_helas_call must be called with wavefunction or amplitude')
        call = ''
        call_function = None
        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = '#'
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return
        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just ixxxxx, oxxxxx, vxxxxx or sxxxxx
            call_function = lambda wf: self.get_external(wf, argument)
        else:
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0
            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            flag = []
            if argument.needs_hermitian_conjugate():
                flag = ['C%d' % i for i in argument.get_conjugate_index()]
            # Creating line formatting:
            # (AV NB: in the default code these two branches were identical, use a single branch)
            ###if isinstance(argument, helas_objects.HelasWavefunction): # AV e.g. FFV1P0_3 (output is wavefunction)
            ###    call = '%(routine_name)s(%(wf)s%(coup)s%(mass)s%(out)s);'
            ###else: # AV e.g. FFV1_0 (output is amplitude)
            ###    call = '%(routine_name)s(%(wf)s%(coup)s%(mass)s%(out)s);'
            call = '%(routine_name)s( %(wf)s%(coup)s%(mass)s%(out)s );'
            # compute wf
            arg = {'routine_name': aloha_writers.combine_name('%s' % l[0], l[1:], outgoing, flag, True),
                   'wf': ('aloha_obj[%%(%d)d], ' * len(argument.get('mothers'))) % tuple(range(len(argument.get('mothers')))),
                   'coup': ('m_pars->%%(coup%d)s, ' * len(argument.get('coupling'))) % tuple(range(len(argument.get('coupling'))))
                   }
            # AV FOR PR #434: determine if this call needs aS-dependent or aS-independent parameters
            usesdepcoupl = None
            for coup in argument.get('coupling'):
                if isinstance(coup, base_objects.FLV_Coupling):
                    if usesdepcoupl is None: usesdepcoupl = False
                    elif usesdepcoupl: raise Exception('PANIC! this call seems to use both aS-dependent and aS-independent couplings?')
                    continue
                if coup.startswith('-'): 
                    coup = coup[1:]
                # Use the same implementation as in UFOModelConverterCPP.prepare_couplings (assume self.model is the same)
                for key, coup_list in self.get('model')['couplings'].items():
                    if coup in coup_list:
                        if "aS" in key:
                            if usesdepcoupl is None: usesdepcoupl = True
                            elif not usesdepcoupl: raise Exception('PANIC! this call seems to use both aS-dependent and aS-independent couplings?')
                        else:
                            if usesdepcoupl is None: usesdepcoupl = False
                            elif usesdepcoupl: raise Exception('PANIC! this call seems to use both aS-dependent and aS-independent couplings?')
            # AV FOR PR #434: CI_ACCESS for independent couplings and CD_ACCESS for dependent couplings
            if usesdepcoupl is None: raise Exception('PANIC! could not determine if this call uses aS-dependent or aS-independent couplings?')
            elif usesdepcoupl: caccess = 'CD_ACCESS'
            else: caccess = 'CI_ACCESS'
            ###if arg['routine_name'].endswith( '_0' ) : arg['routine_name'] += '<W_ACCESS, A_ACCESS, C_ACCESS>'
            ###else : arg['routine_name'] += '<W_ACCESS, C_ACCESS>'
            if arg['routine_name'].endswith( '_0' ) : arg['routine_name'] += '<W_ACCESS, A_ACCESS, %s>'%caccess
            else : arg['routine_name'] += '<W_ACCESS, %s>'%caccess
            if isinstance(argument, helas_objects.HelasWavefunction):
                #arg['out'] = 'w_sv[%(out)d]'
                arg['out'] = 'aloha_obj[%(out)d]'
                if fd_gauge and len(l) > 1: #FIXME: this is a hack to avoid a bug in the FD code
                    arg['out'] = arg['out'] + ", aloha_obj_tmp[0]"
                if aloha.complex_mass:
                    arg['mass'] = 'm_pars->%(CM)s, '
                else:
                    arg['mass'] = 'm_pars->%(M)s, m_pars->%(W)s, '
            else:
                #arg['out'] = '&amp_sv[%(out)d]'
                arg['out'] = '&amp_fp[%(out)d]'
                arg['out2'] = 'amp_sv[%(out)d]'
                if fd_gauge and len(l) > 1: #FIXME: this is a hack to avoid a bug in the FD code
                    arg['out'] = arg['out'] + ", &amp_tmp_fp[0]"
                arg['mass'] = ''
            call = call % arg
            # Now we have a line correctly formatted
            call_function = lambda wf: self.format_coupling(
                                         call % wf.get_helas_call_dict(index=0))
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

def combine_name(name, other_names, outgoing, tag=None, unknown_propa=False):
    """ build the name for combined aloha function """

    def myHash(target_string):
        suffix = ''
        if '%(propa)s' in target_string:
            target_string = target_string.replace('%(propa)s', '')
            suffix = '%(propa)s'

        if len(target_string) < 50:
            return '%s%s' % (target_string, suffix)
        else:
            return 'ALOHA_%s%s' % (str(hash(target_string.lower())).replace('-', 'm'), suffix)

    if tag and any(t.startswith('P') for t in tag[:-1]):
        # propagator need to be the last entry for the tag
        for i, t in enumerate(tag):
            if t.startswith('P'):
                tag.pop(i)
                tag.append(t)
                break

    # Two possible scheme FFV1C1_2_X or FFV1__FFV2C1_X
    # If they are all in FFVX scheme then use the first
    p = re.compile(r'^(?P<type>[RFSVT]{2,})(?P<id>\d+)$')
    routine = ''
    if p.search(name):
        base, id = p.search(name).groups()
        routine = name
        for s in other_names:
            try:
                base2, id2 = p.search(s).groups()
            except Exception:
                routine = ''
                break  # one matching not good -> other scheme
            if base != base2:
                routine = ''
                break  # one matching not good -> other scheme
            else:
                routine += '_%s' % id2

    if routine:
        if tag is not None:
            routine += ''.join(tag)
        if unknown_propa and outgoing:
            routine += '%(propa)s'
        if outgoing is not None:
            return myHash(routine) + '_%s' % outgoing
        else:
            return myHash(routine)

    if tag is not None:
        addon = ''.join(tag)
    else:
        addon = ''
        if 'C' in name:
            short_name, addon = name.split('C', 1)
            try:
                addon = 'C' + str(int(addon))
            except Exception:
                addon = ''
            else:
                name = short_name
    if unknown_propa:
        addon += '%(propa)s'

    if outgoing is not None:
        return myHash('_'.join((name,) + tuple(other_names))) + addon + '_%s' % outgoing
    else:
        return myHash('_'.join((name,) + tuple(other_names))) + addon
#------------------------------------------------------------------------------------

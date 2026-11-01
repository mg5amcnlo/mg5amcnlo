# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: O. Mattelaer (Sep 2021) for the MadGraph7 CUDACPP plugin.
# Further modified by: O. Mattelaer, J. Teig, A. Valassi, Z. Wettersten (2021-2025).
# Integrated with the MadGraph7 project in Feb 2026.

import os
import sys

import glob
import math
import re

# AV - create a plugin-specific logger
import logging
PLUGIN_NAME = __name__.rsplit('.',1)[0]
logger = logging.getLogger('madgraph.%s.model_handling'%PLUGIN_NAME)
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'

from madgraph import MadGraph5Error
from madgraph.iolibs import export_cpp, export_mg7
from madgraph.iolibs import file_writers as writers
from madgraph.iolibs import jamp_optimiser

import aloha
from aloha import aloha_writers

# FD gauge check — if set on second output
class _FdGaugeFlag:
    __slots__ = ()
    def __bool__(self):
        return aloha.unitary_gauge == 3
fd_gauge = _FdGaugeFlag()

from collections import defaultdict
from fractions import Fraction

from io import StringIO

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


# Emit the MADARITH_DOUBLEEXPANSION (FPTYPE=e) code paths only when explicitly requested
arith_doubleexpansion = os.environ.get('MADMATRIX_DOUBLEEXPANSION', '').lower() in ('1', 'true', 'yes', 'on')


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
    ci_definition = 'const cxtype_amp_sv cI = cxmake( 0., 1. );\n'
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
    type2def['vertex_sv'] = 'fptype_amp_sv'
    type2def['vertex_v'] = 'fptype_amp_sv'

    type2def['aloha_ref'] = '&'

    # AV - modify C++ code from aloha_writers.ALOHAWriterForGPU
    # AV new option: declare C++ variable type only when they are defined?
    ###nodeclare = False # old behaviour (separate declaration with no initialization)
    nodeclare = True # new behaviour (delayed declaration with initialisation)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # a combined routine written as a wrapper needs a scratch wavefunction
        # as an extra argument (see write_combined_cc)
        self.combined_needs_tmp = True
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
        if not self.combined_needs_tmp:
            # assembled combined routine: it needs no scratch wavefunction
            combined = False
        elif fd_gauge and name.count("_") > 1:  # FIXME ugly hack to get right header
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
                args.append('fptype Ccoeff%s'% coeff_n) # OM for 'unary minus' #628
            else:
                args.append('%s %s%s'% (type, argname, list_arg))

        indent = ' ' * len( '  %s( ' % name )
        if not self.offshell:
            ###output = '%(doublec)s%(pointer_vertex)s allvertexes' % {
            ###    'doublec': self.type2def['double'],
            ###    'pointer_vertex': self.type2def['pointer_vertex']}
            output = 'fptype_amp allvertexes[]'
            if combined:
                output = output + ', ' + '\n%(indent)sfptype_amp alltmp[]' % {'indent': indent}
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
                out.write('    const cxtype_amp_sv* w%s = W_ACCESS::kernelAccessConst( %s.w );\n' % ( name, name ) )
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
        if not self.offshell:
            out.write('    cxtype_amp_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
        else:
            out.write('    cxtype_amp_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
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
            if not self.offshell:
                out.write('    cxtype_amp_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
            else:
                out.write('    cxtype_amp_sv* %s = %s::kernelAccess( %s );\n' % ( vname, access, allvname ) )
        if fd_gauge:
            out.write('    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); \n')
        # define the complex number CI = 0+1j
        if add_i:
            ###out.write(self.ci_definition)
            out.write('    ' + self.ci_definition) # AV
        codedict = {} # AV allow delayed declaration with initialisation
        for type, name in self.declaration.tolist():
            ###print(name) # FOR DEBUGGING
            ###out.write('    %s %s;\n' % ( type, name ) ) # FOR DEBUGGING
            if type == 'fct':
                continue # OM an external function name (e.g. 'pow'): nothing to declare in C++
            if type == 'parameter':
                out.write(self.get_model_parameter_txt(name)) # OM a model parameter used in the body (custom propagators)
                continue
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
                codedict[fullname] = 'fptype_amp %s' % fullname # multiple of scalars 
            else:
                codedict[fullname] = '%s %s' % (self.type2def[type+'_v'], fullname) # AV vectorize, add to codedict
            ###print(fullname, codedict[fullname]) # FOR DEBUGGING
            if self.nodeclare:
                self.declaration.codedict = codedict # AV new behaviour (delayed declaration with initialisation)
            else:
                out.write('    %s;\n' % codedict[fullname] ) # AV old behaviour (separate declaration with no initialization)
        ###out.write('    // END DECLARATION\n') # FOR DEBUGGING
        return out.getvalue()

    # OM - the names of the aS-dependent parameters, filled by write_aloha_routines.
    # They are recomputed event by event from G inside computeDependentCouplings_fromG,
    # so they have no addressable value a HelAmps routine could read.
    dependent_params = ()

    # OM - the body of a routine may refer to a model parameter: this happens for
    # the custom propagators of a UFO model, whose numerator/denominator are
    # written in terms of model parameters (e.g. the width corrections dWT, dWZ,
    # dWW and dWH of SMEFTsim). aloha_writers.ALOHAWriterForCPP reads them from
    # the Parameters singleton; that is host-only code, so on GPUs the value has
    # to be a compile-time constant, i.e. HRDCOD=1 is required there.
    def get_model_parameter_txt(self, name):
        """Define a model parameter which the routine body uses"""
        # ALOHA sees the UFO name ('dWT'), the generated Parameters class uses
        # the prefixed one ('mdl_dWT'), exactly as in the fortran writer
        mdlname = '%s%s' % (aloha.aloha_prefix, name)
        if mdlname in self.dependent_params:
            raise MadGraph5Error(
                'The custom propagator used by routine %s needs the model parameter %s, '
                'which depends on alphaS and is therefore recomputed event by event. '
                'The C++/CUDA backend can only read alphaS-independent parameters inside '
                'a HelAmps routine, so this process can only be generated with '
                '"output madevent" or "output standalone_fortran".' % (self.name, mdlname))
        return ('#ifdef MGONGPU_HARDCODE_PARAM\n'
                '    constexpr auto %(var)s = Parameters::%(mdl)s;\n'
                '#elif !defined MGONGPUCPP_GPUIMPL\n'
                '    const auto %(var)s = Parameters::getInstance()->%(mdl)s;\n'
                '#else\n'
                '#error Model parameter %(mdl)s is used inside a HelAmps routine '
                '(custom propagator) and is not available in device code: rebuild with HRDCOD=1\n'
                '#endif\n') % { 'var': name, 'mdl': mdlname }

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
            if self.has_fd_propagator():
                out.write(self.get_fd_gauge_txt())
        # Returning result
        ###print('."' + out.getvalue() + '"') # AV - FOR DEBUGGING
        return out.getvalue()

    # OM - FD gauge: the propagator factor is written in the routine (it used to
    # be a call to multiply_propagator_factor), split in a part that only
    # depends on the momentum and one that is linear in the wavefunction.
    def get_fd_gauge_txt(self):
        """FD gauge: the 5-momentum q (its 5th component carries the mass) and
        the gauge direction n. Written with the momenta: it does not depend on
        the wavefunction the routine builds."""

        out = StringIO()
        wf = '%s%s' % (self.particles[self.outgoing-1], self.outgoing)
        # the 5th component is -i * m: built from broadcast reals, since a
        # scalar complex has no conversion to the vector type (cxtype_sv)
        out.write('    cxtype_sv FDQ[5] = { %s, cxmake( fptype_sv{ 0 }, -M%s + fptype_sv{ 0 } ) };\n' %
                  (', '.join('cxmake( fpamp_of_mom( -%s.pvec[%d] ), 0. )' % (wf, i) for i in range(4)),
                   self.outgoing))
        out.write('    fptype_sv FDN[5];\n')
        out.write('    define_gauge_dir( FDQ, FDN );\n')
        out.write('    const fptype_sv FDNQ = %s;\n' %
                  ' - '.join('FDN[%d] * FDQ[%d].real()' % (i, i) for i in range(4)))
        return out.getvalue()

    def get_fd_propagator_txt(self):
        """FD gauge: the part of the propagator factor that is linear in the
        wavefunction, w -> w - q * js1 - n * js2, written after the components
        have been built."""

        out = StringIO()
        w = self.outname
        size = self.type_to_size[self.particles[self.outgoing-1]] - 2
        out.write('    const cxtype_sv FDJS1 = ( %s ) / FDNQ;\n' %
                  ' - '.join('FDN[%d] * %s[%d]' % (i, w, i) for i in range(4)))
        out.write('    const cxtype_sv FDJS2 = ( %s - cxconj( FDQ[4] ) * %s[4] ) / FDNQ;\n' %
                  (' - '.join('FDQ[%d] * %s[%d]' % (i, w, i) for i in range(4)), w))
        for i in range(size):
            out.write('    %(w)s[%(i)d] = %(w)s[%(i)d] - FDQ[%(i)d] * FDJS1 - FDN[%(i)d] * FDJS2;\n'
                      % {'w': w, 'i': i})
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
        if self.nodeclare:
            if ptype == 'double_v':
                # narrow momenta
                strfile.write('    const fptype_amp_sv P%d[4] = { ' % i )
                for j in range(4):
                    sign = self.get_P_sign(i) if self.get_P_sign(i) else '+'
                    element = 'fpamp_of_mom(%(sign)s%(type)s%(i)d.pvec[%(j)d])' % {'j':j,'type': type, 'i': i, 'sign': sign}
                    strfile.write(element + (', ' if j<3 else ''))
                strfile.write(' };\n')
                # dP array in denom precision: for the outgoing particle (its
                # propagator denominator) and for every leg a momenta-only TMP uses
                if i == self.outgoing or i in getattr(self, 'dp_needed', ()):
                    strfile.write('    const fptype_denom_sv dP%d[4] = { ' % i )
                    for j in range(4):
                        sign = self.get_P_sign(i) if self.get_P_sign(i) else '+'
                        element = '%(sign)s%(type)s%(i)d.pvec[%(j)d]' % {'j':j,'type': type, 'i': i, 'sign': sign}
                        strfile.write(element + (', ' if j<3 else ''))
                    strfile.write(' };\n')
            else:
                mom_type = self.type2def[ptype]
                strfile.write('    const %s P%d[4] = { ' % ( mom_type, i ) )
                for j in range(4):
                    sign = self.get_P_sign(i) if self.get_P_sign(i) else '+'
                    strfile.write( (templateval + ( ', ' if j<3 else '' )) % {'j':j,'type': type, 'i': i, 'sign': sign} )
                strfile.write(' };\n')
        else:
            for j in range(4):
                sign = self.get_P_sign(i) if self.get_P_sign(i) else '+'
                template ='    P%(i)d[%(j)d] = ' + templateval + ';\n'
                strfile.write(template % {'j':j,'type': type, 'i': i, 'sign': sign})

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
                # A flavored coupling is indexed by the *merged* fermion leg; for
                # a single merged leg the unmerged partner carries flavor index 0
                # and the merged leg can be either F1 or F2 (the cudacpp argument
                # order is not guaranteed to put the merged leg first, unlike the
                # Fortran side). Pick whichever leg is the merged (partner-
                # populated) one. For the two-leg case partner1[flv1]==flv2 holds
                # and flv_sel==flv_index1, reproducing the old behaviour.
                out.write('    int flv_sel = -1;\n')
                out.write('    if(MCOUP.partner1[flv_index1] == flv_index2) flv_sel = flv_index1;\n')
                out.write('    else if(MCOUP.partner1[flv_index2] == flv_index1) flv_sel = flv_index2;\n')
                out.write('    if(flv_sel == -1) {\n')
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
                # the per-flavor stride is C_ACCESS::flv_stride (nx2 for independent
                # couplings = scalar broadcast; nx2*neppC for dependent ones = AOSOA
                # SIMD record), so the same routine body works for both instantiations
                out.write('    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_sel );\n')
            else:
                for i in range(1,nb_coupling+1):
                    # the coupling is a complex number but in this case it is represented as a sequence of real numbers
                    # so, when we need to shift within the array, we need to double the shift width to account for
                    # both real and imaginary parts
                    out.write('    if(zero_coup%i ==0) { COUP%i = C_ACCESS::kernelAccessConst( MCOUP%i.value + C_ACCESS::flv_stride*flv_index1 ); }\n' % (i,i,i))
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
                    out.write('    %s = C_ACCESS::kernelAccessConst( M%s.value + C_ACCESS::flv_stride*flv_index1 );\n' % (name, name))
        return out.getvalue()

    # OM - add the formats which aloha_writers.ALOHAWriterForCPP is missing.
    # Without this 'pow' falls back to std::pow, which has no overload for the
    # (possibly SIMD) complex types: cxpow in HelAmps expands the integer power.
    def get_fct_format(self, fct):
        """Put the function in the correct format"""
        if not hasattr(self, 'fct_format'):
            super().get_fct_format('sqrt') # a known key: builds self.fct_format with no side effect
            self.fct_format['pow'] = 'cxpow( %s, %s )'
        return super().get_fct_format(fct)

    # OM - momenta-only expressions in denominator precision (FPTYPE=v: double)
    _momentum_ref = re.compile(r'(?<![A-Za-z_0-9])P(\d+)\[(\d)\]')
    _tmp_ref = re.compile(r'(?<![A-Za-z_0-9])(TMP\d+)(?![A-Za-z_0-9])')
    _fp_constant = re.compile(r'(?<![A-Za-z_0-9])(one|two|half|quarter)(?![A-Za-z_0-9])')
    _arithmetic_only = re.compile(r'^[\s()+\-*/.0-9eE]*$')

    def momenta_in_denom_precision(self, expr):
        """If the C++ expression *expr* only involves momenta (P1[0], ...),
        numbers and the fptype constants, return it written with the momenta
        and the constants in denominator precision (dP1[0], oned, ...), and
        record the legs whose dP array it needs. Return None otherwise."""
        legs = self._momenta_only(expr)
        if not legs:
            return None
        self.dp_needed.update(legs)
        dexpr = self._momentum_ref.sub(lambda m: 'dP%s[%s]' % m.groups(), expr)
        return self._fp_constant.sub(r'\1d', dexpr)

    def denominator_momentum_tmps(self):
        """The TMPs of this routine's custom propagator denominator (e.g.
        (P.PBar) P^2 in axial gauge) when that denominator is made only of
        momenta-only TMPs: those, and only those, are computed in denominator
        precision. An empty set when there is no such denominator."""
        denominator = getattr(self.routine, 'denominator', None)
        if not self.offshell or 'L' in self.tag or aloha.complex_mass or \
                denominator is None or str(denominator) == '1' or \
                not self.routine.contracted:
            return set()
        denominator = str(denominator)
        tmps = set(self._tmp_ref.findall(denominator))
        rest = self._fp_constant.sub(' ', self._tmp_ref.sub(' ', denominator))
        if not tmps or not self._arithmetic_only.match(rest):
            return set()
        for tmp in tmps:
            obj = self.routine.contracted.get(tmp)
            if obj is None or not self._momenta_only(self.write_obj(obj)):
                return set()
        return tmps

    def _momenta_only(self, expr):
        """The legs whose momenta *expr* uses, if it only involves momenta,
        numbers and the fptype constants; an empty set otherwise."""
        if aloha.loop_mode:
            return set()
        legs = set(int(leg) for leg, _ in self._momentum_ref.findall(expr))
        rest = self._fp_constant.sub(' ', self._momentum_ref.sub(' ', expr))
        if not legs or not self._arithmetic_only.match(rest):
            return set()
        return legs

    def denominator_in_denom_precision(self, denominator):
        """A custom propagator denominator (e.g. (P.PBar) P^2 in axial gauge)
        made only of momenta-only TMPs is evaluated in denominator precision,
        from their dTMP versions. Anything else is returned unchanged."""
        tmps = self._tmp_ref.findall(denominator)
        if not tmps or not all(tmp in self.pure_momentum_tmps for tmp in tmps):
            return denominator
        rest = self._fp_constant.sub(' ', self._tmp_ref.sub(' ', denominator))
        if not self._arithmetic_only.match(rest):
            return denominator
        return self._fp_constant.sub(r'\1d', self._tmp_ref.sub(r'd\1', denominator))

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
        # OM the TMPn computed in denominator precision (those of a custom
        # propagator denominator), and the legs whose momenta they need in that
        # precision (read by get_one_momenta_def, which WriteALOHA.write calls
        # after this method)
        self.pure_momentum_tmps = set()
        self.dp_needed = set()
        denominator_tmps = self.denominator_momentum_tmps() if self.nodeclare else set()
        if self.routine.contracted:
            keys = sorted(self.routine.contracted.keys())
            for name in keys:
                obj = self.routine.contracted[name]
                # This affects 'TMP0 = ' in HelAmps_sm.cc
                ###out.write(' %s = %s;\n' % (name, self.write_obj(obj)))
                if self.nodeclare:
                    expr = self.write_obj(obj)
                    dexpr = None
                    if name in denominator_tmps:
                        dexpr = self.momenta_in_denom_precision(expr)
                    if dexpr is not None:
                        # tmp variable build from pure momenta
                        #   - single precision is used only in the numerator
                        #   - double precision is used in the denominator
                        self.pure_momentum_tmps.add(name)
                        out.write('    const fptype_denom_sv d%s = %s;\n' % (name, dexpr))
                        out.write('    const cxtype_amp_sv %s = fpamp_of_mom( d%s );\n' % (name, name))
                    else:
                        out.write('    const cxtype_amp_sv %s = %s;\n' %
                                  (name, expr)) # AV
                else:
                    out.write('    %s = %s;\n' % (name, self.write_obj(obj))) # AV
                    self.declaration.add(('complex', name))
        # FCTn are numbered in creation order and a later one can use an
        # earlier one (the $ propagator's theta function takes FCT0/FCT1), so
        # define them in that order -- the dict order puts FCT0 last there.
        for name in sorted(self.routine.fct, key=lambda n: (len(n), n)):
            fct, objs = self.routine.fct[name]
            # OM the FCTn variable needs to be defined, not only assigned (and
            # write_combined_parts_cc looks for exactly this 'const <type> FCTn ='
            # form when it merges the structures of an assembled routine)
            if self.nodeclare:
                format = '    const %s %s = %s;\n' % (self.type2def['complex_v'], name, self.get_fct_format(fct))
            else:
                format = '    %s = %s;\n' % (name, self.get_fct_format(fct))
                self.declaration.add(('complex', name))
            args = [self.write_obj(obj) for obj in objs]
            try:
                out.write(format % ','.join(args)) # single-argument formats
            except TypeError:
                out.write(format % tuple(args)) # e.g. 'pow', which takes two
        numerator = self.routine.expr
        if self.coup_name:
            # one term of an assembled routine: its own coupling among COUP1, ...
            coup_name = self.coup_name
        elif not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)
        has_coup = coup_name != self.change_number_format(1)
        # OM the Ccoeff sign carrier goes with the coupling it multiplies
        ccoeff = 'Ccoeff%s' % coup_name[4:]
        if not self.offshell:
            if has_coup:
                mydict = {'num': self.write_obj(numerator.get_rep([0]))} # '...(TMP4)-cI...' comes from here
                for c in ['coup', 'vertex']:
                    if self.type2def['pointer_%s' %c] in ['*']:
                        mydict['pre_%s' %c] = '( *'
                        mydict['post_%s' %c] = ' )'
                    else:
                        mydict['pre_%s' %c] = ''
                        mydict['post_%s'%c] = ''
                # This affects '( *vertex ) = ' in HelAmps_sm.cc
                mydict['coup'] = coup_name
                mydict['ccoeff'] = ccoeff
                mydict['add'] = '%(pre_vertex)svertex%(post_vertex)s + ' % mydict if self.combined_part else ''
                out.write('    %(pre_vertex)svertex%(post_vertex)s = (cxtype_amp_sv)( %(add)s%(ccoeff)s * %(pre_coup)s%(coup)s%(post_coup)s * %(num)s );\n' % mydict) # OM add Ccoeff (fix #825)
            else:
                mydict= {}
                if self.type2def['pointer_vertex'] in ['*']:
                    mydict['pre_vertex'] = '( *'
                    mydict['post_vertex'] = ' )'
                else:
                    mydict['pre_vertex'] = ''
                    mydict['post_vertex'] = ''
                mydict['data'] = self.write_obj(numerator.get_rep([0]))
                mydict['add'] = '%(pre_vertex)svertex%(post_vertex)s + ' % mydict if self.combined_part else ''
                # This affects '( *vertex ) = ' in HelAmps_sm.cc
                out.write('    %(pre_vertex)svertex%(post_vertex)s = (cxtype_amp_sv)%(add)s%(data)s;\n' % mydict)
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)
            if 'L' not in self.tag:
                # each term of an assembled routine has its own denominator
                # (they are 'const' declarations in the same scope): 'denomname'
                # is already at vertex precision, so the wavefunction formulas
                # below use it as it is
                denomsuffix = coup_name[4:] if self.combined_part else ''
                denomname = 'denom%s' % denomsuffix
                coeff = denomname
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
                    mydict['declnamedenom'] = 'const cxtype_amp_sv %s' % denomname
                else:
                    mydict['declnamedenom'] = denomname # AV
                    self.declaration.add(('complex', denomname))
                # Need to add the unary operator before the coupling (OM fix for #825).
                # pre_coup/post_coup wrap %(coup)s, so open the cast in pre_coup and
                # close it in post_coup (the Ccoeff sign carrier multiplies from outside).
                coup_vertex = '%s%s%s' % (mydict['pre_coup'], coup_name, mydict['post_coup'])
                if has_coup: # but in case where the coupling is not used (one)
                    mydict['pre_coup'] = 'static_cast<fptype_denom>(%s) * static_cast<cxtype_denom_sv>(%s' % (ccoeff, mydict['pre_coup'])
                    mydict['post_coup'] = '%s)' % mydict['post_coup']
                else:
                    mydict['pre_coup'] = 'static_cast<fptype_denom>(%s' % mydict['pre_coup']
                    mydict['post_coup'] = '%s)' % mydict['post_coup']
                # Custom denominators: divided at denom precision.
                prop = lambda den: 'static_cast<cxtype_amp_sv>( %s%s%s / %s )' % (mydict['pre_coup'], coup_name, mydict['post_coup'], den)
                # Standard propagator: madmatrix::denom keeps only P^2 - Re(m^2) at denom precision.
                # Qualified: the local variable is also called denom. Without coupling (one): 1 / ( P^2 - m^2 ).
                coup_args = '%s, %s, ' % (ccoeff, coup_vertex) if has_coup else ''
                propagator = lambda mass: 'madmatrix::denom( %sdP%s, %s )' % (coup_args, self.outgoing, mass)
                if not aloha.complex_mass:
                    # This affects 'denom = COUP' in HelAmps_sm.cc
                    if self.routine.denominator:
                        # the P1N routines of the amplitude split carry the
                        # identity propagator, as the integer 1
                        if str(self.routine.denominator) == '1':
                            out.write('    %s = static_cast<cxtype_amp_sv>( %s%s%s );\n' % (mydict['declnamedenom'], mydict['pre_coup'], coup_name, mydict['post_coup']))
                        else:
                            mydict['denom'] = self.denominator_in_denom_precision(
                                                   str(self.routine.denominator))
                            out.write('    %s = %s;\n' % (mydict['declnamedenom'], prop('( %s )' % mydict['denom'])))
                    else:
                        if arith_doubleexpansion:
                            out.write('\n#ifndef MADARITH_DOUBLEEXPANSION\n')
                        out.write('    %s = %s;\n' % (mydict['declnamedenom'], propagator('M%(i)s, W%(i)s' % mydict)))
                        if arith_doubleexpansion:
                            out.write('#endif\n')
                            out.write('#ifdef MADARITH_DOUBLEEXPANSION\n')
                            wtype = self.particles[self.outgoing - 1]
                            coeff_vertex = '%(pre_coup)s%(coup)s%(post_coup)s' % mydict
                            coeff_vertex = coeff_vertex.replace('fptype_denom', 'fptype_amp')
                            out.write('    const MG_ARITHM::Double<fptype_amp> P{0}d{2}[4] = {{ static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[0]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[1]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[2]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[3]) }};\n'.format(self.outgoing, wtype, denomsuffix))
                            out.write('    const MG_ARITHM::Double<fptype_amp> Md{0}{1} = static_cast<MG_ARITHM::Double<fptype_amp>>(M{0});\n'.format(self.outgoing, denomsuffix))
                            out.write('    const fptype_amp_sv PmM2{1} = static_cast<fptype_amp_sv>(( P{0}d{1}[0] * P{0}d{1}[0] ) - ( P{0}d{1}[1] * P{0}d{1}[1] ) - ( P{0}d{1}[2] * P{0}d{1}[2] ) - ( P{0}d{1}[3] * P{0}d{1}[3] ) - ( Md{0}{1} * Md{0}{1} ) );\n'.format(self.outgoing, denomsuffix))
                            out.write('    const fptype_amp_sv iMW{1} = M{0} * W{0};\n'.format(self.outgoing, denomsuffix))
                            out.write('    const cxtype_amp_sv denden%s = cxmake( PmM2%s, iMW%s );\n' % (denomsuffix, denomsuffix, denomsuffix))
                            out.write('    const cxtype_amp_sv %s = %s / denden%s;\n' % (denomname, coeff_vertex, denomsuffix))
                            out.write('#endif\n')
                else:
                    if self.routine.denominator:
                        raise Exception('modify denominator are not compatible with complex mass scheme')
                    # This affects 'denom = COUP' in HelAmps_sm.cc
                    if arith_doubleexpansion:
                        out.write('\n#ifndef MADARITH_DOUBLEEXPANSION\n')
                    # Complex mass: M carries the width.
                    out.write('    %s = %s;\n' % (mydict['declnamedenom'], propagator('M%(i)s' % mydict)))
                    if arith_doubleexpansion:
                        out.write('#endif\n')
                        out.write('#ifdef MADARITH_DOUBLEEXPANSION\n')
                        wtype = self.particles[self.outgoing - 1]
                        coeff_vertex = '%(pre_coup)s%(coup)s%(post_coup)s' % mydict
                        coeff_vertex = coeff_vertex.replace('fptype_denom', 'fptype_amp')
                        out.write('    const MG_ARITHM::Double<fptype_amp> P{0}d{2}[4] = {{ static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[0]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[1]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[2]), static_cast<MG_ARITHM::Double<fptype_amp>>(-{1}{0}.pvec[3]) }};\n'.format(self.outgoing, wtype, denomsuffix))
                        out.write('    const MG_ARITHM::Double<fptype_amp> Md{0}{1} = static_cast<MG_ARITHM::Double<fptype_amp>>(M{0});\n'.format(self.outgoing, denomsuffix))
                        out.write('    const fptype_amp_sv PmM2{1} = static_cast<fptype_amp_sv>(( P{0}d{1}[0] * P{0}d{1}[0] ) - ( P{0}d{1}[1] * P{0}d{1}[1] ) - ( P{0}d{1}[2] * P{0}d{1}[2] ) - ( P{0}d{1}[3] * P{0}d{1}[3] ) - ( Md{0}{1} * Md{0}{1} ) );\n'.format(self.outgoing, denomsuffix))
                        out.write('    const cxtype_amp_sv %s = %s / PmM2%s;\n' % (denomname, coeff_vertex, denomsuffix))
                        out.write('#endif\n')
                ###self.declaration.add(('complex','denom')) # AV moved earlier (or simply removed)
                if aloha.loop_mode: ptype = 'list_complex'
                else: ptype = 'list_double'
                self.declaration.add((ptype,'P%s' % self.outgoing))
            else:
                coeff = 'COUP'
            shift = 1 - 1 #to correspond to the shift in fortran indices with -1 for C++
            if fd_gauge and "S" in self.outname:
                shift = 5 - 1 #to correspond to the shift in fortran indices with -1 for C++
            for ind in numerator.listindices():
                # This affects 'V1[2] = ' and 'F1[2] = ' in HelAmps_sm.cc
                # the slot is fixed by the spin of this structure, the name by
                # the routine the code is written in; a term of an assembled
                # routine adds up into a slot several structures can feed
                # outname carries the 'w' accessor prefix, rename_wf works on
                # the wavefunction name itself
                slot = 'w%s[%d]' % (self.rename_wf(self.outname[1:]),
                                    self.pass_to_HELAS(ind) + shift)
                out.write('    %s = %s%s * %s;\n' % (slot, # AV
                                        '%s + ' % slot if self.combined_part else '',
                                        coeff,
                                        self.write_obj(numerator.get_rep(ind))))
        if self.has_fd_propagator() and not self.combined_part:
            # the FD gauge propagator factor is part of the wavefunction this
            # routine builds, not a post-treatment (a term of an assembled
            # routine gets it once, from the assembler, on the total)
            out.write(self.get_fd_propagator_txt())
        ###return out.getvalue() # AV
        # AV check if one, two, half or quarter are used and need to be defined (ugly hack for #291: can this be done better?)
        out2 = StringIO()
        if 'one' in out.getvalue(): out2.write('    constexpr fptype_amp one( 1. );\n    constexpr fptype_denom oned( 1. );\n')
        if 'two' in out.getvalue(): out2.write('    constexpr fptype_amp two( 2. );\n    constexpr fptype_denom twod( 2. );\n')
        if 'half' in out.getvalue(): out2.write('    constexpr fptype_amp half( 1. / 2. );\n    constexpr fptype_denom halfd( 1. / 2. );\n')
        if 'quarter' in out.getvalue(): out2.write('    constexpr fptype_amp quarter( 1. / 4. );\n    constexpr fptype_denom quarterd( 1. / 4. );\n')
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
            # the slot is fixed by the spin this structure sees on the leg, the
            # name by the routine the code is written in (rename_wf)
            return 'w%s[%s]' % (self.rename_wf(match.group('var')),
                                int(match.group('num')) + shift)

    # OM - overload aloha_writers.WriteALOHA and ALOHAWriterForCPP methods (handle 'unary minus' #628)
    # also wrap momentum references in non-denominator expressions with vertex-precision cast
    def change_var_format(self, obj):
        """ """
        if obj.startswith('COUP'):
            out = super().change_var_format(obj)
            postfix = out[4:]
            return "Ccoeff%s * static_cast<cxtype_amp_sv>( %s )" % (postfix, out) # OM for 'unary minus' #628, AV cast down for non-denom formulas
        else:
            out = super().change_var_format(obj)
            if out == 'denom':
                out = 'static_cast<cxtype_amp_sv>(%s)' % out
            return out

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
        # Set some useful command
        if offshell is None:
            offshell = self.offshell

        # structures acting on different spins (FD gauge: a leg is a vector in
        # one structure and its Goldstone in the next one) can not share a
        # single ALOHA expression, but they can still be assembled into a
        # single routine instead of a wrapper calling each of them
        self.combined_needs_tmp = True
        if hasattr(self.routine, 'get_combined_routines') and \
                       not os.environ.get('MG_ALOHA_COMBINE_WRAPPER'):
            routines = self.routine.get_combined_routines(lor_names)
            if routines:
                self.combined_needs_tmp = False
                return self.write_combined_parts_cc(routines, lor_names, offshell, mode)

        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        self.name = name
        # write head - momenta - body - foot
        text = StringIO()
        routine = StringIO()
        data = {}  # for the formatting of the line


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

    # OM - a combined routine whose structures do not act on the same spins is
    # assembled term by term instead of calling each of them (see
    # AbstractRoutine.get_combined_routines and the fortran writer)
    def write_combined_parts_cc(self, routines, lor_names, offshell, mode=''):
        """Assemble the structures of a combined call into a single routine:
        one set of momenta, one set of declarations, one FD propagator factor,
        and each structure adding its own coupling times its own expression
        into the slots it feeds."""

        name = combine_name(self.routine.name, lor_names, offshell, self.tag)
        self.name = name
        writers_l = [self.__class__(routine, None, options=self.options)
                                                        for routine in routines]
        main = writers_l[0]
        main.name = name
        main.mode = mode if mode else self.mode
        # assembled: no scratch wavefunction in the signature (the .h header is
        # written from self, which write_combined_cc already flagged)
        main.combined_needs_tmp = False
        # one name per leg -- the one the first structure gives it
        legs = ['%s%d' % (spin, i+1) for i, spin in enumerate(main.particles)]

        bodies = []
        for i, writer in enumerate(writers_l):
            writer.name = name
            writer.mode = main.mode
            writer.coup_name = 'COUP%s' % (i+1)
            writer.combined_part = True
            for j, spin in enumerate(writer.particles):
                wf = '%s%d' % (spin, j+1)
                if wf != legs[j]:
                    writer.wf_rename[wf] = legs[j]
            bodies.append(writer.define_expression())

        new_couplings = ['COUP%s' % (i+1) for i in range(len(lor_names)+1)]
        for writer in writers_l[1:]:
            for entry in writer.declaration:
                main.declaration.add(entry)
            # the momenta are declared once for all the structures
            main.dp_needed |= writer.dp_needed

        text = StringIO()
        text.write(main.get_header_txt(name=name, couplings=new_couplings,
                                       mode=main.mode))
        # unlike the wrapper form, the assembled body needs cI
        text.write(main.get_declaration_txt())
        text.write(main.get_momenta_txt())
        text.write(main.get_coupling_def())
        # the terms add up into the output, so it has to start from zero (in FD
        # gauge the momenta already reset the wavefunction)
        if not offshell:
            text.write('    ( *vertex ) = cxzero_sv();\n')
        elif not (fd_gauge and main.particles[main.outgoing-1] in ['S', 'V']):
            for i in range(self.type_to_size[main.particles[main.outgoing-1]] - 2):
                text.write('    %s[%d] = cxzero_sv();\n' % (main.outname, i))
        # the structures share the contraction cache, so the same TMP/FCT (and
        # the same constexpr fptype) is built by several of them: a given name
        # always stands for the same value, keep the first definition only
        declared = re.compile(r'^(?:const|constexpr)\s+\S+\s+'
                              r'((?:d?TMP|FCT)\d+|(?:one|two|half|quarter)d?|cId)\s*[=(]')
        seen = set()
        for body in bodies:
            for line in body.splitlines(True):
                found = declared.match(line.strip())
                if found:
                    if found.group(1) in seen:
                        continue
                    seen.add(found.group(1))
                text.write(line)
        if main.has_fd_propagator():
            text.write(main.get_fd_propagator_txt())
        text.write(main.get_foot_txt())
        return text.getvalue()


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

        # Refuse merged-flavor structures this backend cannot yet generate
        # correctly (single-merged-leg vertices, e.g. MSSM
        # gluino/chargino-squark-quark; event-by-event flavored couplings) with
        # a clear message rather than crashing or emitting wrong/uncompilable
        # code. See docs/mg7_merged_flavor_mssm_design.md.
        self._assert_flv_couplings_supported(params)

        from madgraph.core import base_objects
        def_flv = []
        # For each parameter, write name = expr;
        for coupl in params:
            for key, c in coupl.flavors.items():
                # Same (k1, k2) derivation as the Fortran/C++/Python backends:
                # for a single merged leg the unmerged partner is flavor index 1
                # and the PARTNER/PARTNER2 direction depends on which fermion
                # carries the merged leg (see FLV_Coupling.get_partner_indices).
                # Using this shared helper keeps mg7 consistent with the others
                # -- previously it always used (k, 1), which transposed the
                # table for a single merged leg not in the first position (e.g.
                # the merged neutrino of `w+ ta+ vt~`).
                k1, k2 = base_objects.FLV_Coupling.get_partner_indices(key)
                def_flv.append('%(name)s.partner1[%(in)i] = %(out)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.partner2[%(out)i] = %(in)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.value[%(in)i] = &%(coupl)s;' % {'name': coupl.name,'in': k1-1, 'coupl': c})

        return "\n  ".join(def_flv)

    # AV - overload export_cpp.UFOModelConverterCPP method (improve formatting)
    def write_set_parameters(self, params):
        # Independent, SLHA-card-level parameters (self.params_ext_names) get
        # an inline check for a runtime override (e.g. set by name via the
        # UMAMI set_parameter interface) ahead of their normal SLHA-driven
        # assignment; every other parameter/coupling is unaffected. NB: this
        # must NOT be folded into super_write_set_parameters_donotfixMajorana,
        # since write_hardcoded_parameters() calls that method directly and
        # requires each parameter's expr to remain a single plain assignment.
        res_strings = []
        for param in params:
            # 'aS' is excluded here even though it is SLHA-card-level: its
            # assignment is entirely commented out below (by name-matching on
            # a plain "  aS = ..." line), since this backend always retrieves
            # alphaS event-by-event (as G) rather than from the param card.
            # Wrapping it in an override check would break that name-match.
            if param.name in self.params_ext_names and param.name != 'aS':
                res_strings.append(
                    'if (m_param_overrides.find("%s") != m_param_overrides.end())\n'
                    '%s = m_param_overrides["%s"];\nelse\n%s' %
                    (param.name, param.name, param.name, param.expr))
            else:
                res_strings.append(param.expr)
        res = "\n".join(res_strings)
        res = res.replace('ABS(','std::abs(') # for SMEFT #614 and #616
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
    def used_dependent_parameters(self):
        """The names of the aS-dependent parameters the dependent couplings of
        the process need, directly or through one another.

        computeDependentCouplings_fromG recomputes them event by event from G,
        so writing every aS-dependent parameter of the model there costs work
        per event, and breaks the build when an unused one calls a UFO
        function the generated code does not define: loop_sm's R2/UV
        counterterm parameters use COND, reglog and mdl_complexi, while the
        tree-level couplings need only G."""
        identifier = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
        by_name = dict((param.name, param) for param in self.params_dep)
        todo = [name for coup in self.coups_dep.values()
                for name in identifier.findall(coup.expr) if name in by_name]
        used = set()
        while todo:
            name = todo.pop()
            if name in used:
                continue
            used.add(name)
            todo.extend(other for other in identifier.findall(by_name[name].expr)
                        if other in by_name)
        return used

    def super_generate_parameters_class_files(self):
        """Create the content of the Parameters_model.h and .cc files"""
        params_dep_used = self.used_dependent_parameters()
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
                if pardep.name not in params_dep_used:
                    continue
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
        # Only the independent flavored couplings are serialized via the FLV_COUPLING
        # value[] pointer mechanism in setIndependentCouplings: a dependent (running-alphas)
        # coupling is not addressable as a fixed pointer here (it is computed event-by-event).
        # Dependent flavored couplings are handled separately via cDPF_* + the per-event
        # gather in calculate_jamps (see get_process_function_definitions / Step 3).
        replace_dict['set_flv_couplings'] = self.write_flv_couplings(self.coups_flv_indep)
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
        # NB: unlike the plain (mg7_v5) backend, dependent (aS-dependent) parameters
        # and couplings are NOT persistent members of this Parameters class at all
        # (their declarations are commented out above): they are recomputed fresh
        # every event batch by computeDependentCouplings(), bypassing the Parameters
        # singleton entirely. Only independent parameters/couplings (and never 'aS',
        # whose declaration is likewise commented out) can be registered here.
        # Two separate calls/maps, not one combined list into one map: model
        # parameters (mdl_*) always stay plain cxsmpl<double> (registered
        # into m_complex_model_params below), while independent couplings
        # (GC_*) are cxtype -- FPTYPE-templated, so possibly cxsmpl<float>
        # -- and go into m_complex_params instead. A single combined
        # registration into one map only compiled by accident at FPTYPE=m/d
        # (where cxtype happens to also be cxsmpl<double>) and failed to
        # compile at FPTYPE=f -- see write_register_parameters()'s own doc
        # comment.
        replace_dict['register_parameters'] = \
                               self.write_register_parameters(
                                   [p for p in self.params_indep if p.name != 'aS'],
                                   [n for n in self.params_ext_names if n != 'aS'],
                                   complex_map_name='m_complex_model_params') + \
                               '\n' + \
                               self.write_register_parameters(self.coups_indep, [])
        replace_dict['register_slha_keys'] = \
                               self.write_register_slha_keys(
                                   [e for e in self.params_ext_slha if e[0] != 'aS'])
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
                if foundG and pdep.name not in params_dep_used:
                    # not needed by any dependent coupling of the process
                    continue
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
        # BSM params for aS-dependent couplings as fptype/cxtype
        replace_dict['eftspecial0'] = ''
        if len( bsmparam_indep_real_used ) + len( bsmparam_indep_complex_used ) == 0:
            replace_dict['eftspecial0'] = '\n      // No special handling of non-hardcoded parameters (no additional BSM parameters needed in constant memory)'
        for ipar, par in enumerate( bsmparam_indep_real_used ):
            replace_dict['eftspecial0'] += '\n      const fptype %s = bsmIndepParamPtr[%i];' % ( par, ipar )
        for ipar, par in enumerate( bsmparam_indep_complex_used ):
            ire = len( bsmparam_indep_real_used ) + 2 * ipar
            replace_dict['eftspecial0'] += '\n      const cxtype %s = cxtype( bsmIndepParamPtr[%i], bsmIndepParamPtr[%i] );' % ( par, ire, ire + 1 )
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
        # OM - the writer needs to know which parameters are alphaS-dependent (they
        # cannot be read from the Parameters class inside a routine)
        self.aloha_writer.dependent_params = frozenset(p.name for p in self.params_dep)
        if(fd_gauge):
            aloha_model = create_aloha.AbstractALOHAModel.from_model(self.model, explicit_combine=False)
        else:
            aloha_model = create_aloha.AbstractALOHAModel.from_model(self.model, explicit_combine=True)
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        if self.wanted_lorentz:
            aloha_model.compute_subset(self.wanted_lorentz)
        else:
            aloha_model.compute_all(save=False, custom_propa=True)
        for abstracthelas in dict(aloha_model).values():
            # print(type(abstracthelas), abstracthelas.name) # AV this is the loop on FFV functions
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
        # Add Arithmetics include guarded by MADARITH_DOUBLEEXPANSION (only when that path is generated)
        if arith_doubleexpansion:
            file_h = file_h.replace(
                '#include "mgOnGpuConfig.h"',
                '#include "mgOnGpuConfig.h"\n#ifdef MADARITH_DOUBLEEXPANSION\n#include "Arithmetics/Double.h"\n#endif'
            )
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
    # Below this many colors the SYMM call is not worth setting up and the
    # scalar sum wins (see cpp_blas_wanted_for)
    blas_min_ncolor = 100
    support_multichannel = False
    multichannel_var = ',fptype& multi_chanel_num, fptype& multi_chanel_denom'
    imaginary_unit = "cxtype_amp(0,1)"

    # Build rules (in SubProcesses/) that this P* directory links as its 'makefile'
    p_makefile = 'madmatrix.mk'

    # AV - overload export_cpp.OneProcessExporterCPP constructor (rename gCPPProcess to CPPProcess)
    def __init__(self, *args, **kwargs):
        ###misc.sprint('Entering OneProcessExporterMadMatrix.__init__')
        ###for kwarg in kwargs: misc.sprint( 'kwargs[%s] = %s' %( kwarg, kwargs[kwarg] ) )
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
        # ncolor_flow sits next to ncolor in the class, so it has to be known
        # here as well as in get_process_function_definitions
        self.set_color_flow_lines_cpp(self.matrix_elements[0], replace_dict)
        ###misc.sprint( replace_dict['nwavefuncs'] ) # NB: this (from export_cpp) is the WRONG value of nwf, e.g. 6 for gg_tt (#644)
        ###misc.sprint( self.matrix_elements[0].get_number_of_wavefunctions() ) # NB: this is a different WRONG value of nwf, e.g. 7 for gg_tt (#644)
        ###replace_dict['nwavefunc'] = self.matrix_elements[0].get_number_of_wavefunctions() # how do I get HERE the right value of nwf, e.g. 5 for gg_tt?
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        replace_dict['nbhel'] = self.matrix_elements[0].get_helicity_combinations() # number of helicity combinations
        replace_dict['ndiagrams'] = len(self.matrix_elements[0].get('diagrams')) # AV FIXME #910: elsewhere matrix_element.get('diagrams') and max(config[0]...
        replace_dict['nmaxflavor'] = len(self.matrix_elements[0].get_external_flavors_with_iden()) # number of flavor combinations
        # Only written when the jamps are actually split, so that a process
        # without squared split orders keeps the header it always had
        replace_dict['split_order_constants'] = '' if not self.split_orders_active() else (
            '\n    // Squared split orders (see ProcessData.h, and color_sum_cpu_splitorders'
            '\n    // in backend/<variant>/color_sum.cc for how the jamps are paired)'
            '\n    static constexpr int nampso = ProcessData::nampso;'
            '\n    static constexpr int njampso = ProcessData::njampso; // the jamps of every amplitude order, end to end'
            '\n    static constexpr int nsqampso = ProcessData::nsqampso;')
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
    # backend_separation: cIPD/cIPC/cIPF/bsmIndepParam storage now lives in
    # backend/{cpu,simd,gpu}/SigmaKin.cc. This method still computes the local
    # tIPD/tIPC/tIPF assignment text (genuinely process-specific: which SM
    # parameters/couplings this process uses), but ends each with a call to
    # the corresponding backend setter instead of a storage-declaration
    # variant + direct memcpy/gpuMemcpyToSymbol.
    def get_process_function_definitions(self, write=True):
        """The complete class definition for the process"""
        replace_dict = super().get_process_function_definitions(write=False) # defines replace_dict['initProc_lines']
        replace_dict['hardcoded_initProc_lines'] = self.get_hardcoded_initProc_lines(self.matrix_elements[0])
        # Cached for edit_processtables(): calculate_jamps' jampTmp_sv shared
        # sub-expression scratch is backend-owned storage now, sized from this
        # process-specific count (ProcessTables::nb_tmp_jamp) rather than
        # hardcoded per-process like the rest of calculate_jamps.
        self._nb_tmp_jamp = getattr(self.helas_call_writer, 'nb_tmp_jamp', 0)
        replace_dict['refreshExternalMasses_lines'] = \
                               self.get_refresh_external_masses_lines(self.matrix_elements[0])
        replace_dict['nparams'] = len(self.params2order)
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

        # dependent (running-alphas, event-by-event) flavor couplings -> for ProcessTables.h (Step 3).
        flv_couplings_dep = [''] * len(self.couporderflv_dep)
        for flv_coup, pos in self.couporderflv_dep.items():
            flv_couplings_dep[pos] = flv_coup

        # Cache counts for edit_processdata()/edit_processtables(), which run
        # after generate_process_files() has populated couplings2order etc.
        self._nipc = len(coupling_indep)
        self._nipd = len(params)
        self._nipf = len(flv_couplings)
        self._ndpf = len(flv_couplings_dep)

        if len(coupling_indep) > 0:
            replace_dict['cipcassign'] = ('static constexpr cxtype Parameters::* const cIPC_members[nIPC] = {\n'
                                           '      &Parameters::' + ',\n      &Parameters::'.join(coupling_indep) + '\n'
                                           '    };\n'
                                           '    cxtype tIPC[nIPC];\n'
                                           '    gatherCxtype( m_pars, cIPC_members, tIPC );\n'
                                           '    setIndependentCouplings( tIPC );')
            coup_str_hrd = 'const cxtype tIPC[nIPC] = { cxmake( Parameters::%s ) };\n    setIndependentCouplings( tIPC );'\
                                         % ( ' ), cxmake( Parameters::'.join(coupling_indep) )
            replace_dict['cipchrdassign'] = coup_str_hrd
        else:
            replace_dict['cipcassign'] = '//const cxtype tIPC[0] = { ... }; // nIPC=0'
            replace_dict['cipchrdassign'] = '//const cxtype tIPC[0] = { ... }; // nIPC=0'

        if len(params) > 0:
            replace_dict['cipdassign'] = ('static constexpr double Parameters::* const cIPD_members[nIPD] = {\n'
                                           '      &Parameters::' + ',\n      &Parameters::'.join(params) + '\n'
                                           '    };\n'
                                           '    fptype tIPD[nIPD];\n'
                                           '    gatherFptype( m_pars, cIPD_members, tIPD );\n'
                                           '    setIndependentParams( tIPD );')
            replace_dict['cipdhrdassign'] = 'const fptype tIPD[nIPD] = { (fptype)Parameters::%s };\n    setIndependentParams( tIPD );'\
                                         %( ', (fptype)Parameters::'.join(params) )
        else:
            replace_dict['cipdassign'] = '//const fptype tIPD[0] = { ... }; // nIPD=0'
            replace_dict['cipdhrdassign'] = '//const fptype tIPD[0] = { ... }; // nIPD=0'

        # flavor couplings
        for flv_coup, pos in self.couporderflv.items():
            flv_couplings[pos] = flv_coup
        if len(flv_couplings):
            nMF = max(len(ids) for ids in self.model['merged_particles'].values())
            # we have 3 arrays: all partner1/partner2/value arrays combined
            replace_dict['cipfassign'] = """int tIPF_partner1[nMF * nIPF];
    int tIPF_partner2[nMF * nIPF];
    cxtype tIPF_value[nMF * nIPF];
    const FLV_COUPLING tFLV[nIPF] = { m_pars->%s };
    for (int i = 0; i < nIPF; ++i) {
      memcpy( tIPF_partner1 + i * nMF, tFLV[i].partner1, nMF * sizeof( int ) );
      memcpy( tIPF_partner2 + i * nMF, tFLV[i].partner2, nMF * sizeof( int ) );
      for (int j = 0; j < nMF; ++j)
        tIPF_value[i * nMF + j] = tFLV[i].value[j] ? *tFLV[i].value[j] : cxtype{}; // guard from null pointers
    }
    setFlavorCouplings( tIPF_partner1, tIPF_partner2, tIPF_value );""" % ( ', m_pars->'.join(flv_couplings) )
            # Hardcoded variant: same shape, values come from Parameters:: instead of m_pars->
            hrd_lines = ['int tIPF_partner1[nMF * nIPF];', '    int tIPF_partner2[nMF * nIPF];', '    cxtype tIPF_value[nMF * nIPF];']
            for i, flv_coup in enumerate(flv_couplings):
                base = 'Parameters_%s::%s' % (self.model_name, flv_coup)
                for j in range(nMF):
                    hrd_lines.append('    tIPF_partner1[%d] = %s.param1[%d];' % (i * nMF + j, base, j))
                    hrd_lines.append('    tIPF_partner2[%d] = %s.param2[%d];' % (i * nMF + j, base, j))
                    hrd_lines.append('    tIPF_value[%d] = %s.value[%d] ? *%s.value[%d] : cxtype{};' % (i * nMF + j, base, j, base, j))
            hrd_lines.append('    setFlavorCouplings( tIPF_partner1, tIPF_partner2, tIPF_value );')
            replace_dict['cipfhrdassign'] = '\n    '.join(hrd_lines)
        else:
            replace_dict['cipfassign'] = ''
            replace_dict['cipfhrdassign'] = ''

        # mdl_bsmIndepParam only exists as a symbol at all when the model has
        # BSM params (see MGONGPUCPP_NBSMINDEPPARAM_GT_0, PR #625) - this is a
        # pre-existing macro, not new, and must stay a #ifdef (not a runtime
        # check) since the symbol itself may not exist to name-lookup.
        replace_dict['bsmassign'] = '''#ifdef MGONGPUCPP_NBSMINDEPPARAM_GT_0
    if( Parameters::nBsmIndepParam > 0 ) setBsmIndepParam( m_pars->mdl_bsmIndepParam, Parameters::nBsmIndepParam );
#endif'''
        replace_dict['bsmhrdassign'] = '''#ifdef MGONGPUCPP_NBSMINDEPPARAM_GT_0
    if( Parameters::nBsmIndepParam > 0 ) setBsmIndepParam( Parameters::mdl_bsmIndepParam, Parameters::nBsmIndepParam );
#endif'''

        # ncolor_flow is set by set_color_flow_lines_cpp in
        # get_process_class_definitions (process_class.inc), and the color flow
        # lines go to ColorFlows.inc (edit_colorflows); the broken-symmetry data
        # broken_symmetry_factor reads is in ProcessTables.h (edit_processtables).

        file = self.read_template_file(self.process_definition_template) % replace_dict # HACK! ignore write=False case
        if len(params) == 0: # remove cIPD from OpenMP pragma (issue #349)
            file_lines = file.split('\n')
            file_lines = [l.replace('cIPC, cIPD','cIPC') for l in file_lines] # remove cIPD from OpenMP pragma
            file = '\n'.join( file_lines )
        file = strip_banner(file, banner_mark = "!") # skip first 8 lines in process_function_definitions.inc (copyright)
        return file

    # backend_separation: sigmaKin and everything it calls are backend-owned
    # (backend/<variant>/SigmaKin.cc), so there is no sigmaKin text to render
    # into CPPProcess.cc any more; export_cpp still asks for it.
    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        """Nothing process-specific left to write for sigmaKin (see above)."""
        if self.include_multi_channel and not self.support_multichannel:
            raise Exception("This standalone format does not support madevent interface")
        return ('', {}) if write else {}

    #===========================================================================
    # helicity recycling (--hel_recycling)
    #===========================================================================
    # A wavefunction only depends on the helicities of the external legs it
    # descends from, so every helicity row that agrees on those legs can share
    # it. calculate_jamps rebuilds everything once per row; the recycled build
    # (calculate_all_jamps in backend/{cpu,simd}/SigmaKin.cc) makes each
    # wavefunction once per COPY, skips the amplitudes that vanish at a row and
    # folds the others into per-row tables. The unfolding is madevent's own
    # hel_recycle, through its C++ dialect.
    #
    # Two passes. At generation, the calls are written once more in the form the
    # recycler reads (hr_orig_calls) and the ordinary ones carry a probe hook.
    # Once the directory is buildable, the standalone exporter's finalize builds a
    # probe (BACKEND=scalar FPTYPE=d -DMG_HR_PROBE), which prints the good
    # helicities and, per helicity, the amplitudes that are only a cancellation
    # zero; write_recycled then runs the recycler for those and writes
    # HelRecycling.inc / HelRecyclingCalls.inc, whose presence is what switches
    # SigmaKin.cc over. Any failure leaves the ordinary code in place.
    _hr_warmups = []
    HR_CALLS_MARK = '//HR_CALLS'

    # One function holding the whole unrolled block is what makes a big process
    # unbuildable: the compiler's per-function cost is superlinear, so the
    # recycled block of g g > g g g g in one body takes longer to compile than any
    # build should. It is cut into ordinary functions of about this many
    # statements that calculate_all_jamps calls in order (the fix the fortran
    # recycler applies too); --hel_recycling_chunk=<n> overrides it, 0 keeps the
    # block in one piece. Below HR_INLINE_MAX statements it stays inline: the
    # chunks cost about 9% at run time (g g > g g g: 239k -> 218k events/s, the
    # same with 8 chunks or 38), while its 8148 statements build in 6 s inline.
    # Above, chunks of 200 build fastest for the same run time (g g > g g g g,
    # 136510 statements: 29 s, 52 s and 76 s with chunks of 200, 1000 and 3000
    # for the same run time, against 808 s inline for a run 13% faster).
    HR_CHUNK_STMTS = 200
    HR_INLINE_MAX = 15000
    # The chunks are written into translation units of about this many
    # statements (HelRecyclingChunks_<k>.cc), which the build compiles in
    # parallel: in one SigmaKin.cc, g g > t t~ g g g (2.8M statements) is a
    # single compiler process for the best part of an hour.
    HR_FILE_STMTS = 20000
    # What the recycled calls read of calculate_all_jamps and of SigmaKin.cc: the
    # chunks live in translation units of their own, so the run-time tables of
    # SigmaKin.cc (cIPD, cFlavors, cBWCUTOFF) are passed along too.
    HR_CHUNK_STATE = [
        ('const fptype_momenta* momenta', 'momenta'),
        ('const fptype* const* COUPs', 'COUPs'),
        ('const unsigned int iflavor', 'iflavor'),
        ('ALOHAOBJ* aloha_obj', 'aloha_obj'),
        ('cxtype_amp_sv* amp_sv', 'amp_sv'),
        ('fptype_amp* amp_fp', 'amp_fp'),
        ('ALOHAOBJ& _p1n', '_p1n'),
        ('cxtype_amp_sv* jampAll_sv', 'jampAll_sv'),
        ('cxtype_amp_sv* jampTmpAll_sv', 'jampTmpAll_sv'),
        ('cxtype_amp_sv* ampAll_sv', 'ampAll_sv'),
        ('fptype_amp_sv* numAll_sv', 'numAll_sv'),
        ('const int iParity', 'iParity'),
        ('const bool storeChannelWeights', 'storeChannelWeights'),
        ('const FLV_COUPLING_ARRAY<nIPF, nMF>& flvCOUPs', 'flvCOUPs'),
        ('const FLV_COUPLING_ARRAY<nDPF, nMF, HostAccessCouplings::flv_stride>&'
         ' flvCOUPs_dep', 'flvCOUPs_dep'),
        ('const fptype* cIPD', 'cIPD'),
        ('const short ( *cFlavors )[npar]', 'cFlavors'),
        ('const fptype cBWCUTOFF', 'cBWCUTOFF'),
    ]
    # hidden: every P* library has its own hr_unit_<k>, which must never bind
    # to another one's; noinline: one call per unit is the point
    HR_CHUNK_ATTR = '__attribute__( ( noinline, visibility( "hidden" ) ) )'
    HR_CHUNK_FILE = 'HelRecyclingChunks_%d.cc'
    HR_CHUNK_USING = (
        '    using M_ACCESS [[maybe_unused]] = HostAccessMomenta;\n'
        '    using W_ACCESS [[maybe_unused]] = HostAccessWavefunctions;\n'
        '    using A_ACCESS [[maybe_unused]] = HostAccessAmplitudes;\n'
        '    using CD_ACCESS [[maybe_unused]] = HostAccessCouplings;\n'
        '    using CI_ACCESS [[maybe_unused]] = HostAccessCouplingsFixed;\n'
        '    using CDPF_ACCESS [[maybe_unused]] = HostAccessCouplings;\n'
        '    using F_ACCESS [[maybe_unused]] = HostAccessIflavorVec;\n')

    # Above this many statements of code (hr_count_statements: the tables of
    # the tabulated P1N groups do not count) the recycled block is declined and
    # the directory keeps the plain code. Recycling makes each wavefunction once
    # per copy instead of once per row, so it executes fewer instructions, but
    # as ONE straight run of code per event page, where the plain build loops
    # over the rows through the same few hundred kB. Once that run no longer
    # fits in the caches, every page fetches it from memory again and the
    # instructions per cycle collapse. Measured on an Apple M5 Pro (16 MB L2),
    # check_sa simd_128, recycled over plain, with the amplitudes as
    # straight-line code: 1.4x-2.2x up to 183k statements (6 MB of code),
    # 0.47x at 610k (22 MB), 0.31x at 1.29M (48 MB, g g > t t~ g g g, IPC 0.73
    # against 5.1). The tables (--hel_recycling_tables, on by default) take
    # g g > t t~ g g g down to ~100k statements of code, 3.4 MB, and 1.7x
    # faster than plain; the limit is for what is still bigger than that.
    # --hel_recycling_max=<n> overrides it, 0 for no limit.
    HR_MAX_STMTS = 300000

    def hel_recycling_max_stmts(self):
        """Statements above which the recycled block is declined (0: never),
        from --hel_recycling_max if given."""
        options = getattr(self.helas_call_writer, 'cmd_options', None) or {}
        if 'hel_recycling_max' in options:
            try:
                return int(options['hel_recycling_max'])
            except (TypeError, ValueError):
                logger.warning('--hel_recycling_max must be an integer; '
                               'using the default')
        return self.HR_MAX_STMTS

    def hel_recycling_tables(self):
        """--hel_recycling_tables: write the P1N groups of the recycled block as
        tables (default) or as straight-line code (see CppDialect._p1n_table)."""
        options = getattr(self.helas_call_writer, 'cmd_options', None) or {}
        return str(options.get('hel_recycling_tables', True)).lower() \
                                            not in ('false', '0', 'no')

    @staticmethod
    def hr_count_statements(calls):
        """Statements of code in a recycled call block: the lines hr_chunk_calls
        counts, but for the tables of the tabulated P1N groups, which are data
        and not code (see HelRecyclingTables.h)."""
        return sum(1 for line in calls.split('\n')
                   if line.split('//')[0].strip()
                   and not line.lstrip().startswith('static constexpr'))

    def hel_recycling_chunk_size(self, nstmt):
        """Statements per chunk for a block of `nstmt` statements (0: inline),
        from --hel_recycling_chunk if given."""
        options = getattr(self.helas_call_writer, 'cmd_options', None) or {}
        if 'hel_recycling_chunk' in options:
            try:
                return int(options['hel_recycling_chunk'])
            except (TypeError, ValueError):
                logger.warning('--hel_recycling_chunk must be an integer; '
                               'using the default')
        return self.HR_CHUNK_STMTS if nstmt > self.HR_INLINE_MAX else 0

    @staticmethod
    def hr_split_statements(lines, per_chunk):
        """Cut the calls into groups of about `per_chunk` statements.

        A cut may only fall where nothing is open. Brace depth is not enough to
        say that: `if( storeChannelWeights )` leaves the depth at zero and its
        body opens on the NEXT line, so a cut after it would hand the body to
        another function. The line has to have finished a statement too -- ended
        with a ';' or a '}'."""
        chunks, current, depth, count = [], [], 0, 0
        for line in lines:
            current.append(line)
            code = line.split('//')[0].rstrip()
            depth += code.count('{') - code.count('}')
            if code.strip():
                count += 1
            complete = code.endswith(';') or code.endswith('}')
            if depth == 0 and complete and count >= per_chunk:
                chunks.append(current)
                current, count = [], 0
        if current:
            chunks.append(current)
        return chunks

    def hr_chunk_calls(self, calls):
        """(declarations, translation units, calls) for the recycled call block,
        or ('', [], calls) when it is small enough to stay inline.

        Each unit holds its chunks (static, hr_chunk_<i>) and one function,
        hr_unit_<k>, that calls them in order: calculate_all_jamps calls the
        units, so that its own body stays a handful of calls -- with one call
        per chunk (10570 for g g > t t~ g g g), SigmaKin.cc alone took longer
        to build than all the chunks together."""
        lines = calls.rstrip('\n').split('\n')
        nstmt = sum(1 for line in lines if line.split('//')[0].strip())
        per_chunk = self.hel_recycling_chunk_size(nstmt)
        if per_chunk <= 0 or nstmt <= per_chunk:
            return '', [], calls
        params = ',\n              '.join('[[maybe_unused]] ' + decl
                                          for decl, _ in self.HR_CHUNK_STATE)
        # the declarations SigmaKin.cc sees leave the names out: there, cIPD &co
        # are its own tables
        types = ', '.join(re.sub(r'\b%s\b' % name, '', decl).strip()
                          for decl, name in self.HR_CHUNK_STATE)
        args = ', '.join(name for _, name in self.HR_CHUNK_STATE)
        units, unit, unit_stmts, nchunks = [], [], 0, 0
        for chunk in self.hr_split_statements(lines, per_chunk):
            unit.append(chunk)
            unit_stmts += sum(1 for line in chunk if line.split('//')[0].strip())
            if unit_stmts >= self.HR_FILE_STMTS:
                units.append(unit)
                unit, unit_stmts = [], 0
        if unit:
            units.append(unit)
        decls, texts, body = [], [], []
        for k, chunks in enumerate(units):
            defs, calls_k = [], []
            for chunk in chunks:
                # noinline: folding the chunks back into one body is what they avoid
                defs.append('  __attribute__( ( noinline ) ) static void\n'
                            '  hr_chunk_%d( %s )\n  {\n%s%s\n  }\n'
                            % (nchunks, params, self.HR_CHUNK_USING, '\n'.join(chunk)))
                calls_k.append('    hr_chunk_%d( %s );' % (nchunks, args))
                nchunks += 1
            defs.append('  %s void\n  hr_unit_%d( %s )\n  {\n%s\n  }\n'
                        % (self.HR_CHUNK_ATTR, k, params, '\n'.join(calls_k)))
            texts.append('// --hel_recycling: recycled chunks of calculate_all_jamps, generated by\n'
                         '// madgraph/madevent/hel_recycle.py (see HelRecycling.inc)\n\n'
                         '#include "HelRecyclingChunks.h"\n\n'
                         'namespace madmatrix\n{\n%s}\n' % '\n'.join(defs))
            decls.append('  %s void hr_unit_%d( %s );' % (self.HR_CHUNK_ATTR, k, types))
            body.append('      hr_unit_%d( %s );' % (k, args))
        logger.debug('hel_recycling: %d statements in %d chunks, %d files',
                     nstmt, nchunks, len(texts))
        return '\n'.join(decls) + '\n', texts, '\n'.join(body) + '\n'

    @property
    def hel_recycling(self):
        """Whether the output command asked for --hel_recycling=True (off by
        default here; madevent's own option of the same name is separate)."""
        options = getattr(self.helas_call_writer, 'cmd_options', None) or {}
        return str(options.get('hel_recycling', False)).lower() \
                                                in ('true', '1', 'yes')

    def hel_recycling_refusal(self):
        """Why this process cannot be recycled, or None."""
        if aloha.unitary_gauge == 3:
            return 'the FD gauge (its five-component wavefunctions)'
        if len(self.matrix_elements) != 1:
            return 'more than one matrix element in the directory'
        so = export_v4.split_order_tables(self.matrix_elements[0])
        if so and so['nampso'] > 1:
            return 'split amplitude orders'
        return None

    @staticmethod
    def hr_jamps_inc(post):
        """HelRecyclingJamps.inc: the color flows of one row (slot), with
        jamp_sv the row's color flows -- in the rows mode inside hr_row, once
        per good helicity, in the others at the end of calculate_all_jamps."""
        head = ['// --hel_recycling (%s): the color flows of one row (slot), once every'
                % post['mode'],
                '// amplitude is made (see calculate_all_jamps)']
        if post['mode'] == 'fold':
            head.append('cxtype_amp_sv* jampTmp_sv = jampTmpAll_sv + (size_t)slot * %d;'
                        % max(post['ntmp'], 1))
        elif post['mode'] == 'rows':
            # inside hr_row: numerators_sv is the event's own (nullptr unless
            # storeChannelWeights), jamp_sv the helicity's color flows
            head += ['const cxtype_amp_sv* ampRow = ampAll_sv + slot * %d;'
                     % max(post['namp'], 1)]
            if post['ntmp'] > 0:
                head.append('cxtype_amp_sv jampTmp_sv[%d];' % post['ntmp'])
        return ''.join('      %s\n' % line for line in head + list(post['lines']))

    def hr_orig_calls(self, color_amplitudes):
        """The calls of the matrix element in the form the recycler reads.

        Written before the ordinary calls: the ordinary pass renumbers every
        amplitude to the one scratch it writes, and the number is what an
        hr_orig amplitude is tagged with (and what the color flows and the
        quartic current sums are keyed on). Restored afterwards, so that the
        ordinary pass sees what it always saw."""
        matrix_element = self.matrix_elements[0]
        writer = self.helas_call_writer
        amplitudes = matrix_element.get_all_amplitudes()
        numbers = [(amp, amp.get('number')) for amp in amplitudes]
        writer.hr_orig = True
        writer.hr_jamp_post = dict(writer.HR_POST_EMPTY)
        try:
            calls = writer.get_matrix_element_calls(
                matrix_element, color_amplitudes,
                multi_channel_map=self.multi_channel_map)
            self._hr_jamp_post = writer.hr_jamp_post
            return calls
        finally:
            writer.hr_orig = False
            for amp, number in numbers:
                amp.set('number', number)

    def write_recycled(self, good_hels, zero_perhel):
        """Run the recycler over the hr_orig calls for the good helicities
        (1-based rows of cHel) and the (row, amplitude) pairs that vanish, and
        write HelRecycling.inc and HelRecyclingCalls.inc. Returns the number of
        rows built, or None when the recycled block is declined for its size
        (hel_recycling_max_stmts): then nothing is written and the directory
        keeps the plain code."""
        import madgraph.madevent.hel_recycle as hel_recycle
        matrix_element = self.matrix_elements[0]
        all_hel = [tuple(hel) for hel in
                   matrix_element.get_helicity_matrix(allow_reverse=True)]
        ncomb = len(all_hel)
        kept = sorted(int(hel) for hel in good_hels)
        row_of_hel = [-1] * ncomb
        for compact, full in enumerate(kept):
            row_of_hel[full - 1] = compact
        post = getattr(self, '_hr_jamp_post', None) or \
               MadMatrixUFOHelasCallWriter.HR_POST_EMPTY
        ntmp = post['ntmp'] if post['mode'] == 'fold' else 0
        namp = post['namp'] if post['mode'] == 'rows' else 0
        orig_path = pjoin(self.path, 'HelRecycling_orig.cc')
        tmpl_path = pjoin(self.path, 'HelRecycling_template.cc')
        out_path = pjoin(self.path, 'HelRecycling_out.cc')
        with open(orig_path, 'w') as fsock:
            fsock.write('\n'.join(self._hr_calls) + '\n')
        with open(tmpl_path, 'w') as fsock:
            fsock.write(
                '  // --hel_recycling: generated by madgraph/madevent/hel_recycle.py\n'
                '  constexpr int nHrRows = ${ncomb}; // helicity rows built\n'
                '  constexpr int hrNwf = ${nwavefuncs}; // wavefunction slots of the recycled build\n'
                '  constexpr int hrNtmp = %d; // shared color-flow sub-expressions of one row (fold)\n'
                '  constexpr int hrNamp = %d; // amplitude slots of one row (rows)\n'
                '  static constexpr int cHrRow[ncomb] = { %s }; // row of each helicity (-1: not built)\n'
                '%s'
                '${helicity_lines}\n%s\n${helas_calls}\n'
                % (ntmp, namp, ', '.join(str(row) for row in row_of_hel),
                   '#define MGONGPU_HR_ROWS 1 // the color flows are made row by row in hr_row\n'
                   if post['mode'] == 'rows' else '',
                   self.HR_CALLS_MARK))
        dialect = hel_recycle.CppDialect()
        dialect.TABLES = self.hel_recycling_tables()
        recycler = hel_recycle.HelicityRecycler(
                        [str(hel) for hel in kept], [],
                        sorted((int(h), int(a)) for h, a in zero_perhel),
                        dialect=dialect)
        recycler.hel_filt = True
        recycler.amp_splt = True
        recycler.amp_filt = True
        try:
            recycler.set_input(orig_path)
            recycler.set_output(out_path)
            recycler.set_template(tmpl_path)
            recycler.set_helicity_table(all_hel)
            recycler.generate_output_file()
            with open(out_path) as fsock:
                text = fsock.read()
        finally:
            for path in (orig_path, tmpl_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        header, calls = text.split(self.HR_CALLS_MARK, 1)
        nstmt = self.hr_count_statements(calls)
        limit = self.hel_recycling_max_stmts()
        if limit > 0 and nstmt > limit:
            logger.warning('hel_recycling: %s is written WITHOUT recycling: its '
                'recycled block would be %d statements (%d helicity rows), more '
                'than --hel_recycling_max=%d. Code that size no longer runs from the '
                'caches and is slower than the plain build (2x at 22 MB of code). '
                'Pass --hel_recycling_max=0 to build it anyway.',
                os.path.basename(self.path), nstmt, len(kept), limit)
            return None
        decls, units, calls = self.hr_chunk_calls(calls.lstrip('\n'))
        for stale in glob.glob(pjoin(self.path, self.HR_CHUNK_FILE.replace('%d', '*'))):
            os.remove(stale)
        for k, unit in enumerate(units):
            with open(pjoin(self.path, self.HR_CHUNK_FILE % k), 'w') as fsock:
                fsock.write(unit)
        with open(pjoin(self.path, 'HelRecycling.inc'), 'w') as fsock:
            fsock.write(header + '\n' + decls)
        with open(pjoin(self.path, 'HelRecyclingCalls.inc'), 'w') as fsock:
            fsock.write(calls)
        with open(pjoin(self.path, 'HelRecyclingJamps.inc'), 'w') as fsock:
            fsock.write(self.hr_jamps_inc(post))
        return len(kept)

    # AV - modify export_cpp.OneProcessExporterCPP method (fix CPPProcess.cc)
    # backend_separation: calculate_jamps' prologue (signature, memory-access
    # typedefs) and epilogue (color-choice bookkeeping, jamp output copy) are
    # backend-conditional but process-independent, so they live in
    # backend/{cpu,simd,gpu}/SigmaKin.cc. Only the diagram/vertex-call sequence
    # (helas_calls) is process-specific; it is written here to
    # EvaluateDiagrams.inc, which SigmaKin.cc #includes (as it does the color
    # flows of ColorFlows.inc, see edit_colorflows).
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Write EvaluateDiagrams.inc, the diagram calls backend/<variant>/SigmaKin.cc #includes"""
        recycle = False
        if self.single_helicities and self.hel_recycling:
            refusal = self.hel_recycling_refusal()
            if refusal:
                logger.warning('--hel_recycling is not supported with %s: %s is '
                               'written without it.', refusal,
                               os.path.basename(self.path))
            else:
                recycle = True
                self._hr_calls = self.hr_orig_calls(color_amplitudes[0])
                if self not in OneProcessExporterMadMatrix._hr_warmups:
                    OneProcessExporterMadMatrix._hr_warmups.append(self)
        if self.single_helicities:
            self.helas_call_writer.hr_probe = recycle
            try:
                helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    self.matrix_elements[0],
                                                    color_amplitudes[0],
                                                    multi_channel_map = self.multi_channel_map
                                                    )
            finally:
                self.helas_call_writer.hr_probe = False
            assert len(self.matrix_elements) == 1 or len(self.matrix_elements) == 2 # how to handle if this is not true?
            self.couplings2order = self.helas_call_writer.couplings2order
            self.couporderflv = self.helas_call_writer.couporderflv
            self.couporderflv_dep = self.helas_call_writer.couporderflv_dep
            self.params2order = self.helas_call_writer.params2order
            content = []
            content += helas_calls
        else:
            content = [self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)]
        ff = open(pjoin(self.path, 'EvaluateDiagrams.inc'), 'w')
        ff.write('\n'.join(content))
        ff.close()
        return ''

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
        self.edit_colordata() # AV new file (NB this is Sigma-specific, should not be a symlink to Subprocesses)
        self.edit_colorflows()
        super().generate_process_files()
        # needs to be after get_matrix_element_calls to have nwf ready
        self.edit_processdata()
        self.edit_processtables()
        # The build rules live in SubProcesses/<p_makefile>; SubProcesses/makefile
        # itself is the dispatcher that fans out over all the P* directories.
        # NB: this symlink is overwritten by the madevent makefile if this exists (#480)
        # NB: this relies on the assumption that cudacpp code is generated before madevent code
        files.ln(pjoin(self.path, "..", self.p_makefile), self.path, "makefile")

    def edit_colorflows(self):
        """Generate ColorFlows.inc: the process-specific amplitudes the color
        choice in calculate_jamps picks a color flow among.

        These are not always jamp_sv. When the color sum runs on the (n-2)! DDM
        basis the ncolor_flow trace flows are rebuilt from it (Kleiss-Kuijf),
        and when the jamps are split by amplitude order the flow is taken from
        their sum (see set_color_flow_lines_cpp). backend/<variant>/SigmaKin.cc
        #includes this right after EvaluateDiagrams.inc, in the same scope, and
        reads the result through jampflow_sv[0..ncolor_flow)."""
        replace_dict = {'ncolor': len(self.matrix_elements[0].get_color_amplitudes())}
        self.set_color_flow_lines_cpp(self.matrix_elements[0], replace_dict)
        lines = ['// Color flows for the color choice in calculate_jamps (generated).',
                 '// #included by backend/<variant>/SigmaKin.cc right after EvaluateDiagrams.inc.',
                 replace_dict['jampflow_lines'],
                 '      const auto* jampflow_sv = %s; // the ncolor_flow color flow amplitudes'
                 % replace_dict['jamp_flow'],
                 '']
        ff = open(pjoin(self.path, 'ColorFlows.inc'), 'w')
        ff.write('\n'.join(lines))
        ff.close()

    # separate process constants to one truth file
    def edit_processdata(self):
        """Generate ProcessData.h"""
        template = open(pjoin(self.template_path, 'madmatrix', 'ProcessData.h'), 'r').read()
        me = self.matrix_elements[0]
        replace_dict = {}
        nexternal, nincoming = me.get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        replace_dict['nbhel'] = me.get_helicity_combinations()
        replace_dict['ndiagrams'] = len(me.get('diagrams'))
        replace_dict['nmaxflavor'] = len(me.get_external_flavors_with_iden())
        replace_dict['nwave'] = 4 + (1 if fd_gauge else 0)
        replace_dict['ncolor'] = len(me.get_color_amplitudes())
        so = self.split_orders_info() if self.split_orders_active() else None
        replace_dict['nampso'] = so['nampso'] if so else 1
        replace_dict['nsqampso'] = so['nsqampso'] if so else 1
        replace_dict['nwf'] = me.get_number_of_wavefunctions()
        replace_dict['nproc'] = sum(2 if m.get('has_mirror_process') else 1 for m in self.matrix_elements)
        replace_dict['proc_id'] = self.proc_id if self.proc_id > 0 else 1
        den_factors = [str(m.get_denominator_factor()) for m in self.matrix_elements]
        replace_dict['den_factors'] = ",".join(den_factors)
        # cached by get_process_function_definitions(), which runs earlier in
        # super().generate_process_files()
        replace_dict['nipd'] = self._nipd
        replace_dict['nipc'] = self._nipc
        replace_dict['nipf'] = self._nipf
        replace_dict['ndpf'] = self._ndpf
        replace_dict['processid'] = self.name
        replace_dict['processid_uppercase'] = self.name.upper()
        replace_dict['thel_lines'] = self.get_helicity_matrix(me).replace('helicities', 'tHel')
        replace_dict['tflavors_lines'] = self.get_flavor_matrix(me).replace('flavors', 'tFlavors')
        ff = open(pjoin(self.path, 'ProcessData.h'), 'w')
        ff.write(template % replace_dict)
        ff.close()

    # backend_separation: process-specific compile-time DATA (arrays, not
    # scalars) that backend-owned code needs but can't take as a runtime
    # parameter without losing constexpr-ness (see ProcessTables.h).
    def edit_processtables(self):
        """Generate ProcessTables.h"""
        template = open(pjoin(self.template_path, 'madmatrix', 'ProcessTables.h'), 'r').read()
        replace_dict = {}

        # jampTmp_sv scratch size for calculate_jamps' shared sub-expressions
        # (cached by get_process_function_definitions(), see there).
        replace_dict['nb_tmp_jamp'] = self._nb_tmp_jamp

        # Dependent (event-by-event, running-alphas) flavor couplings: partner
        # indices and the per-flavor idcoup are pure compile-time constants
        # (the complex values are gathered per event page in calculate_jamps).
        flv_couplings_dep = [''] * len(self.couporderflv_dep)
        for flv_coup, pos in self.couporderflv_dep.items():
            flv_couplings_dep[pos] = flv_coup
        if len(flv_couplings_dep):
            nMF = max(len(ids) for ids in self.model['merged_particles'].values())
            flv_map = self.helas_call_writer.flv_couplings_map
            partner1_vals, partner2_vals, idcoup_vals = [], [], []
            for name in flv_couplings_dep:
                coupl = flv_map[name]
                p1 = [-1] * nMF
                p2 = [-1] * nMF
                idc = ['-1'] * nMF
                for key, gc in coupl.flavors.items():
                    nonzero = [i for i in key if i != 0]
                    if len(nonzero) == 2:
                        k1, k2 = nonzero
                    else:
                        # single merged leg: unmerged partner has flavor index 1
                        k1 = nonzero[0]; k2 = 1
                    p1[k1-1] = k2-1
                    p2[k2-1] = k1-1
                    # symbolic idcoup: resolves to the position of this dependent coupling
                    # in the event-by-event allcouplings buffer (== COUPs index), defined in
                    # Parameters_dependentCouplings (Parameters_<model>.h)
                    idc[k1-1] = '(int)Parameters_dependentCouplings::idcoup_%s' % gc
                partner1_vals += [str(v) for v in p1]
                partner2_vals += [str(v) for v in p2]
                idcoup_vals += idc
            cdpfdecl = '__device__ constexpr int cDPF_partner1[nMF * nDPF] = { %s };\n' % ', '.join(partner1_vals)
            cdpfdecl += '  __device__ constexpr int cDPF_partner2[nMF * nDPF] = { %s };\n' % ', '.join(partner2_vals)
            cdpfdecl += '  __device__ constexpr int cDPF_idcoup[nMF * nDPF] = { %s };' % ', '.join(idcoup_vals)
            replace_dict['cdpfdecl'] = cdpfdecl
        else:
            replace_dict['cdpfdecl'] = """__device__ constexpr const int* cDPF_partner1 = nullptr; // unused as nDPF=0
  __device__ constexpr const int* cDPF_partner2 = nullptr; // unused as nDPF=0
  __device__ constexpr const int* cDPF_idcoup = nullptr; // unused as nDPF=0"""

        # broken_symmetry_factor data: same shared decay-aware symmetry data
        # as the Fortran / standalone_cpp exporters.
        _, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        process = self.matrix_elements[0].get('processes')[0]
        sym_data = export_v4.ProcessExporterFortran._get_broken_symmetry_data(
            process, nincoming)
        export_v4.ProcessExporterFortran._fill_broken_sym_replace_dict(
            replace_dict, sym_data)

        ff = open(pjoin(self.path, 'ProcessTables.h'), 'w')
        ff.write(template % replace_dict)
        ff.close()

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

    _blas_available = None
    _blas_flags = ''

    @classmethod
    def blas_is_available(cls):
        """Whether a BLAS carrying SYMM can be linked, asked once. The probe
        goes through gfortran, which is the compiler MG5 already knows it has;
        the flags it settles on are the ones the C++ link line needs too."""

        if cls._blas_available is None:
            import subprocess, tempfile, shutil
            probe = ("      PROGRAM P\n"
                     "      DOUBLE PRECISION A(1,1),B(1,1),C(1,1)\n"
                     "      A=1D0\n      B=1D0\n      C=0D0\n"
                     "      CALL DSYMM('L','U',1,1,1D0,A,1,B,1,0D0,C,1)\n"
                     "      END\n")
            work = tempfile.mkdtemp()
            cls._blas_available = False
            cls._blas_flags = ''
            try:
                src = os.path.join(work, 'p.f')
                open(src, 'w').write(probe)
                for flags in ('-framework Accelerate', '-lblas'):
                    try:
                        out = subprocess.call(
                            ['gfortran', src, '-o', os.path.join(work, 'p')]
                            + flags.split(),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
                    except OSError:
                        break
                    if out == 0:
                        cls._blas_available = True
                        cls._blas_flags = flags
                        break
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return cls._blas_available

    @classmethod
    def blas_available_flags(cls):
        """What a BLAS carrying SYMM needs on the link line, empty when there
        is none. This only asks whether one is there; whether a given process
        takes it is cpp_blas_wanted_for."""

        if not cls.blas_is_available():
            return ''
        return cls._blas_flags

    # AV - new method
    def split_orders_info(self):
        """The squared split-order tables for this process, or None.

        None means the process has no '^2' constraint, and then every hole this
        fills reproduces the code that was written before split orders existed
        -- which is how the default path stays byte-for-byte what it was.
        """
        if not hasattr(self, '_split_orders_info'):
            from madgraph.iolibs.export_v4 import split_order_tables
            self._split_orders_info = split_order_tables(self.matrix_elements[0])
        return self._split_orders_info

    def split_orders_active(self):
        """Whether the jamps carry an amplitude-order index at all.

        A single amplitude order is the same code as none: the pair loop of the
        color sum has one term, so it is left switched off rather than writing
        a 1x1 interference matrix."""
        so = self.split_orders_info()
        return bool(so) and so['nampso'] > 1

    # AV - new method
    @classmethod
    def cpp_blas_wanted_for(cls, ncolor):
        """Whether the C++ color sum goes through a host BLAS: only when one
        carrying SYMM can be linked, and when the color matrix is big enough
        that the call is worth setting up. With BLAS off nothing is written
        out, so color_sum.cc and CPPProcess.cc are character for character the
        files written before any of this existed."""
        if not cls.blas_is_available():
            return False
        return ncolor >= cls.blas_min_ncolor

    def cpp_blas_wanted(self):
        # The BLAS color sum multiplies a single jamp vector per helicity by the
        # color matrix. With the jamps split by amplitude order the sum is a
        # pair loop instead, so the scalar color sum is used for those.
        if self.split_orders_active():
            return False
        return self.cpp_blas_wanted_for(
            max(1, len(self.matrix_elements[0].get('color_basis'))))

    # AV - new method
    def get_sqso_table_lines(self):
        """The interference matrix and the squared-order mask, as C++ tables.

        sqSoIndex[m][n] is the Fortran SQSOINDEX: the squared order a pair of
        amplitude orders lands in. It is symmetric because a squared order is
        the SUM of the two amplitude orders, which is what lets color_sum_cpu
        mask on it and still get a real answer."""
        so = self.split_orders_info()
        lines = []
        lines.append('  // The squared order each pair of amplitude orders'
                     ' contributes to (symmetric)')
        # __device__: the GPU color sum reads them in a kernel (an empty macro
        # on the CPU backends), as for mgOnGpu::icolamp
        lines.append('  __device__ constexpr int sqSoIndex[nampso][nampso] = {')
        lines.append(',\n'.join('    { %s }' % ', '.join(str(i) for i in row)
                                for row in so['sqsoindex']))
        lines.append('  };')
        lines.append('  // The squared orders the process asked for:')
        for k, (name, keep) in enumerate(zip(so['names'], so['chosen'])):
            lines.append('  //   %d) %s%s' % (k, name,
                                              '' if keep else '   [dropped]'))
        lines.append('  __device__ constexpr bool chosenSqso[nsqampso] = { %s };'
                     % ', '.join('true' if k else 'false'
                                 for k in so['chosen']))
        return '\n'.join(lines)

    # generate process specific color matrix + channel/config maps - algo is backend owned
    def edit_colordata(self):
        """Generate ColorData.h"""
        ###misc.sprint('Entering OneProcessExporterMadMatrix.edit_colordata')
        template = open(pjoin(self.template_path,'madmatrix','ColorData.h'),'r').read()
        replace_dict = {}
        # Extract the color matrix
        replace_dict['color_matrix_lines'] = self.get_color_matrix_lines(self.matrix_elements[0])
        # backend/{cpu,simd}/color_sum.cc always compiles the BLAS path (it is
        # only ever built, never process-specific); this constexpr, not this
        # file's %-substitution, is what picks it at compile time per process.
        replace_dict['should_use_blas'] = 'true' if self.cpp_blas_wanted() else 'false'
        # A process whose '^2' constraint leaves more than one amplitude split
        # order pairs its jamps in the color sum instead (color_sum_cpu_splitorders
        # in backend/{cpu,simd}/color_sum.cc, selected at compile time from
        # ProcessData::nampso: those files are compiled once per P* directory).
        if self.split_orders_active():
            replace_dict['sqso_tables'] = self.get_sqso_table_lines()
        else:
            replace_dict['sqso_tables'] = '\n'.join([
                '  __device__ constexpr int sqSoIndex[nampso][nampso] = { { 0 } };',
                '  __device__ constexpr bool chosenSqso[nsqampso] = { true };'])

        # we don't sort self.multi_channel_map, and we rely on MadSpace sorting
        # so, diagrams there may be unsorted
        config_subproc_map_C = self.multi_channel_map.values() # this has C-indexing
        config_subproc_map = [] # this has Fortran indexing
        for config in config_subproc_map_C:
            config_subproc_map.append([c+1 for c in config])

        # The following five lines from OneProcessExporterCPP.get_sigmaKin_lines (using OneProcessExporterCPP.get_icolamp_lines)
        iconfig_to_diag = {}
        diag_to_iconfig = {}
        iconfig = 0
        for config in config_subproc_map:
            if set(config) == set([0]):
                continue
            iconfig += 1
            iconfig_to_diag[iconfig] = config[0] 
            diag_to_iconfig[config[0]] = iconfig


        # Note that if the last diagram is/are not mapped to a channel nb_diag 
        # will be smaller than the true number of diagram. This is fine for color
        # but maybe not for something else.
        nb_diag = max(config[0] for config in config_subproc_map)
        ndigits = str(int(math.log10(nb_diag))+1+1) # the additional +1 is for the -sign
        # Output which diagrams correspond to a channel to get information for valid color
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
        # icolamp is the mask of the color flows allowed for a config, and the
        # selection walks it over ncolor_flow entries, so it is dimensioned on
        # the flow basis -- which is larger than the color basis itself when
        # the color sum runs on the DDM one
        nb_color = max(1, len(self.color_flow_basis))
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
        ff = open(pjoin(self.path, 'ColorData.h'),'w')
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
    def set_color_flow_lines_cpp(self, matrix_element, replace_dict):
        """Fill in replace_dict everything the process template needs to know
        about the color flow basis.

        For a fully adjoint (multi-gluon) process the color sum can be done on
        the (n-2)! Del Duca-Dixon-Maltoni basis, but a color flow still has to
        be picked among the (n-1)! trace structures. The trace jamps are then
        not built from the amplitudes but obtained from the DDM ones through
        the Kleiss-Kuijf relations, which is (n-1) times cheaper."""

        color_basis = matrix_element.get('color_basis')
        flow_basis = color_basis.get_flow_basis() if color_basis else None

        # A color flow is picked from the WHOLE amplitude, so when the jamps are
        # split by amplitude order the flow is built from their sum: the color
        # flow amplitude of flow i is sum_M jamp_M[i], and squaring that gives
        # back exactly today's jamp2 when there is a single amplitude order.
        # (The squared-order mask is deliberately not applied here: this weight
        # only chooses which color flow to write out, it is not an observable.)
        so = self.split_orders_info()
        split = bool(so) and so['nampso'] > 1
        jamp_src = 'jamp_sv'
        so_lines = []
        if split:
            jamp_src = 'jampso_sv'
            so_lines = ['',
                        '      // The color flows of the whole amplitude: the'
                        ' jamps of every',
                        '      // amplitude split order summed back together'
                        ' (see color_sum.cc,',
                        '      // which instead keeps them apart and pairs'
                        ' them)',
                        '      cxtype_sv jampso_sv[ncolor] = {};',
                        '      for( int icol = 0; icol < ncolor; icol++ )',
                        '        for( int iao = 0; iao < nampso; iao++ )',
                        '          jampso_sv[icol] += jamp_sv[iao * ncolor +'
                        ' icol];']

        if flow_basis is None or flow_basis is color_basis:
            replace_dict['ncolor_flow'] = replace_dict['ncolor']
            replace_dict['jampflow_lines'] = '\n'.join(so_lines)
            replace_dict['jamp_flow'] = jamp_src
            replace_dict['jamp_flow_col'] = '%s[icol]' % jamp_src
            return

        projection = color_basis.get_flow_projection()
        lines = so_lines + ['',
                 '      // The color flow jamps, rebuilt from the ones entering',
                 '      // the color sum through the Kleiss-Kuijf relations',
                 '      cxtype_sv jampf_sv[ncolor_flow] = {};']
        for i, coeff_list in enumerate(projection):
            terms = ''.join('%s%s[%d]' % (self.coeff(coefficient[0],
                                                     coefficient[1],
                                                     coefficient[2],
                                                     coefficient[3]),
                                          jamp_src,
                                          number - 1)
                            for coefficient, number in coeff_list)
            lines.append('      jampf_sv[%d] = %s;' % (i, terms if terms
                                                       else 'cxzero_sv()'))

        replace_dict['ncolor_flow'] = max(1, len(flow_basis))
        replace_dict['jampflow_lines'] = '\n'.join(lines)
        replace_dict['jamp_flow'] = 'jampf_sv'
        replace_dict['jamp_flow_col'] = 'jampf_sv[icol]'

        logger.debug('Color sum on %d DDM structures, color flow on %d trace '
                     'structures (%d Kleiss-Kuijf terms)',
                     replace_dict['ncolor'], replace_dict['ncolor_flow'],
                     sum(len(row) for row in projection))

    def get_color_matrix_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split rows in chunks of size n."""
        if not matrix_element.get('color_matrix'):
            return '\n'.join([
                '  static constexpr fptype_colour colorDenom[1] = {1.};',
                '  static constexpr fptype_colour colorMatrix[1][1] = {1.};'])
        else:
            color_denominators = matrix_element.get('color_matrix').\
                                                 get_line_denominators()
            num_lists = [matrix_element.get('color_matrix').
                             get_line_numerators(index, denominator)
                         for index, denominator
                         in enumerate(color_denominators)]
            ncolor = len(color_denominators)
            denom_string = '  static constexpr fptype_colour colorDenom[ncolor] = { %s }; // 1-D array[%i]' \
                           % ( ', '.join(['%i' % denom for denom in color_denominators]), ncolor )
            matrix_strings = ['{ %s }' % ', '.join(['%d' % i for i in num_list])
                              for num_list in num_lists]
            matrix_string = '  static constexpr fptype_colour colorMatrix[ncolor][ncolor] = '
            if len( matrix_strings ) > 1:
                matrix_string += '{\n    ' + ',\n    '.join(matrix_strings) + ' };'
            else:
                matrix_string += '{ ' + matrix_strings[0] + ' };'
            matrix_string += ' // 2-D array[%i][%i]' % ( ncolor, ncolor )
            denom_comment = '\n  // The color denominators (initialize all array elements, with ncolor=%i)\n  // [NB do keep \'static\' for these constexpr arrays, see issue #283]\n' % ncolor
            matrix_comment = '\n  // The color matrix (initialize all array elements, with ncolor=%i)\n  // [NB do keep \'static\' for these constexpr arrays, see issue #283]\n' % ncolor
            denom_string = denom_comment + denom_string
            matrix_string = matrix_comment + matrix_string
            return '\n'.join([denom_string, matrix_string])

    # AV - replace the export_cpp.OneProcessExporterCPP method (improve formatting)
    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """initProc_lines for CPPProcess::initProc (non-hardcoded branch): a generated pointer-to-member table read by gatherFptype() (see Parameters.h)."""
        masses = [part.get('mass') for part in matrix_element.get_external_wavefunctions()]
        return ('// Set external particle masses for this matrix element\n'
                '    static constexpr double Parameters::* const massMembers[npar] = {\n'
                '      &Parameters::' + ',\n      &Parameters::'.join(masses) + '\n'
                '    };\n'
                '    fptype tMasses[npar];\n'
                '    gatherFptype( m_pars, massMembers, tMasses );\n'
                '    m_masses.assign( tMasses, tMasses + npar );')

    def get_hardcoded_initProc_lines(self, matrix_element):
        """initProc_lines for CPPProcess::initProc, MGONGPU_HARDCODE_PARAM branch: Parameters has no instance here, so this stays imperative."""
        initProc_lines = ['// Set external particle masses for this matrix element']
        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append('    m_masses.push_back( Parameters::%s );' % part.get('mass'))
        return '\n'.join(initProc_lines)

    # Companion to get_initProc_lines: the same external-mass list, but pushed into an
    # arbitrary "masses" vector instead of m_masses, for CPPProcess::refreshExternalMasses()
    # (called after umami_set_parameter() overrides a mass so UMAMI_META_MASSES stays in sync;
    # see madgraph/iolibs/template_files/madmatrix/umami.cc).
    def get_refresh_external_masses_lines(self, matrix_element):
        """Get refreshExternalMasses_lines for function definition for CPPProcess::refreshExternalMasses"""
        lines = ['    // Set external particle masses for this matrix element']
        for part in matrix_element.get_external_wavefunctions():
            lines.append('    masses.push_back( m_pars->%s );' % part.get('mass'))
        return '\n'.join(lines)

    # AV - replace the export_cpp.OneProcessExporterCPP method (fix helicity order and improve formatting)
    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""
        helicity_line = '  static constexpr short helicities[ncomb][npar] = {\n    '; # AV (this is tHel)
        helicity_line_list = []
        for helicities in matrix_element.get_helicity_matrix(allow_reverse=True): # AV was False: different order in Fortran and cudacpp! #569
            helicity_line_list.append( '{ ' + ', '.join(['%d'] * len(helicities)) % tuple(helicities) + ' }' ) # AV
        return helicity_line + ',\n    '.join(helicity_line_list) + ' };' # AV

    def get_flavor_matrix(self, matrix_element):
        """Return the flavor matrix definition lines for this matrix element"""
        # Emitted at namespace scope (file-local linkage), so the host-side table
        # is visible to both the CPPProcess constructor and CPPProcess::flavorPDG.
        model = matrix_element.get('processes')[0].get('model')
        merged = model.get('merged_particles') or {}
        all_pdgs = [l.get('id') for l in
                    matrix_element.get('processes')[0].get('legs_with_decays')]

        flavor_line = '  static constexpr short flavors[nmaxflavor][npar] = {\n    '; # (this is tFlavors)
        flavor_line_list = []
        for flavors in matrix_element.get_external_flavors_with_iden():
            # get only the index 0 one because the other ones have same matrix element.
            # These values are used at runtime as 0-based indices into the per-flavor
            # FLV_COUPLING arrays (partner1/partner2/value, size max_flavor).  A
            # merged leg's entry is the *physical PDG* of its flavor, so convert it
            # to the 0-based position inside the merged group; an unmerged leg maps
            # to 0.  Subtracting 1 directly only worked for the light quarks, whose
            # PDG (1..4) equals the position; for a merged lepton/neutrino it gave an
            # out-of-range index (e.g. 15 for nu_tau) -> out-of-bounds read (segfault).
            cpp_flavors = []
            for j, flv in enumerate(flavors[0]):
                raw = all_pdgs[j]
                if abs(raw) in merged:
                    cpp_flavors.append(list(merged[abs(raw)]).index(abs(flv)))
                else:
                    cpp_flavors.append(0)
            flavor_line_list.append( '{ ' + ', '.join('%d' % v for v in cpp_flavors) + ' }' )
        out = flavor_line + ',\n    '.join(flavor_line_list) + ' };'
        # Companion table with the true signed PDG ids of each flavor
        # combination (same convention as the PDG lines printed by the
        # Fortran/C++ standalone 'check' drivers); used by CPPProcess::flavorPDG.
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


# Standalone mode: P*/makefile points at the wrapper that also builds check_sa.exe
# (see ProcessExporterMadMatrixStandalone in output.py)
class OneProcessExporterMadMatrixStandalone(OneProcessExporterMadMatrix):

    p_makefile = 'madmatrix_standalone.mk'

#------------------------------------------------------------------------------------

import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.helas_call_writers as helas_call_writers

# AV - define a custom HelasCallWriter
# (NB: enable this via ProcessExporterMadMatrix.helas_exporter in output.py - this fixes #341)
class MadMatrixUFOHelasCallWriter(helas_call_writers.GPUFOHelasCallWriter,
                                  jamp_optimiser.JampOptimiser):
    """ A Custom HelasCallWriter """

    # Flavor-mask optimization: skip wavefunction/amplitude calls that vanish
    # for the selected flavor (see super_get_matrix_element_calls). Toggled by
    # the output command's --mask=True|False; default on.
    use_flavor_mask = True
    # Write the color flows through the shared sub-expressions the color-flow
    # optimisation finds, instead of one line per (color flow, amplitude) pair
    # (see build_jamp_plan). Toggled by --jamp_optim=True|False.
    jamp_optim = True
    # Look for those sub-expressions by whole orbits of the permutations
    # leaving the color basis invariant (see JampOptimiser). Toggled by
    # --jamp_orbit=True|False.
    jamp_orbit = True
    # --hel_recycling. hr_orig: write the calls the way madevent's hel_recycle
    # reads them back (one call per line, every amplitude tagged //HRAMP with its
    # number and followed by what it adds to the per-row tables, with a ${row}
    # hole); see OneProcessExporterMadMatrix.hr_orig_calls. hr_probe: follow every
    # amplitude of the ordinary calls with MG_HR_PROBE_AMP, which only a warm-up
    # build (-DMG_HR_PROBE) compiles to anything.
    hr_orig = False
    hr_probe = False
    # Class structure information
    #  - object
    #  - dict(object) [built-in]
    #  - PhysicsObject(dict) [in madgraph/core/base_objects.py]
    #  - HelasCallWriter(base_objects.PhysicsObject) [in madgraph/iolibs/helas_call_writers.py]
    #  - UFOHelasCallWriter(HelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - CPPUFOHelasCallWriter(UFOHelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - GPUFOHelasCallWriter(CPPUFOHelasCallWriter) [in madgraph/iolibs/helas_call_writers.py]
    #  - MadMatrixUFOHelasCallWriter(GPUFOHelasCallWriter, JampOptimiser)
    #      This class (JampOptimiser is in madgraph/iolibs/jamp_optimiser.py and
    #      brings the color-flow optimisation shared with the fortran exporter)


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
            self.couporderflv = {}      # independent (fixed) flavored couplings -> flvCOUPs
            self.couporderflv_dep = {}  # dependent (running-alphas) flavored couplings -> flvCOUPs_dep
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
                flv_coup = self.flv_couplings_map[coup]
                # Classify the whole flavored coupling as dependent (running-alphas,
                # event-by-event) or independent, mirroring export_cpp.prepare_couplings
                # (which buckets it into coups_flv_dep/coups_flv_indep using one of its
                # underlying couplings).
                is_flv_dep = model.is_running_coupling(flv_coup.get_one_coupling())
                if coup not in [c.name for c in self.wanted_ordered_flv_couplings]:
                    self.wanted_ordered_flv_couplings.append(flv_coup)
                    # NB: the underlying couplings are added to the *independent* wanted
                    # list even when they are running. prepare_couplings still routes the
                    # running ones into coups_dep (by the 'aS' model key), but keeping
                    # them out of wanted_ordered_dep_couplings preserves the alignment
                    # between couporderdep and the coups_dep/idcoup ordering used by the
                    # ordinary (non-flavored) dependent couplings.
                    for ud_coup in set(flv_coup.flavors.values()):
                        if ud_coup not in self.wanted_ordered_indep_couplings:
                            self.wanted_ordered_indep_couplings.append(ud_coup)
                if is_flv_dep:
                    alias = self.couporderflv_dep
                    aliastxt = 'flvCOUPdep'
                    name = 'flvCOUPs_dep'
                else:
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
            elif name == 'flvCOUPs_dep':
                # dependent (running-alphas) flavored coupling: instantiate the vertex
                # routine with CDPF_ACCESS, the access to the values gathered into
                # flvCOUPs_dep (each backend's SigmaKin.cc defines it: CD_ACCESS on the
                # CPU, a one-event buffer on the GPU, where a thread is one event), so
                # get_coupling_def reads them with the right per-flavor stride.
                call = call.replace('CI_ACCESS', 'CDPF_ACCESS')
                call = call.replace('m_pars->%s%s' % (sign, coup),
                                    '%s[%s], %s' % (name, alias[coup], '1.0' if not sign else '-1.0'))
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
        if 'flvCOUPs_dep[' in call and ('CDPF_ACCESS' not in call or re.search(r'\bCOUPs\[|flvCOUPs\[', call)):
            # one access type per vertex routine: CDPF_ACCESS cannot also read the
            # all-events buffer of an ordinary running coupling (on the GPU)
            raise Exception('a vertex mixing a running flavored coupling with another running coupling is not supported: %s' % call)
        model.cudacpp_wanted_ordered_couplings = self.wanted_ordered_dep_couplings + self.wanted_ordered_indep_couplings + self.wanted_ordered_flv_couplings
        return call

    # AV - new method for formatting wavefunction/amplitude calls
    # [It would be too complex to modify them in helas_objects.HelasWavefunction/Amplitude.get_call_key]
    @staticmethod
    def format_call(call):
        return call.replace('(','( ').replace(')',' )').replace(',',', ')

    # --- Color flows through shared sub-expressions --------------------------
    #
    # Written out as it stands, a color flow is one 'jamp_sv[i] += c*amp_sv[0]'
    # per (color flow, amplitude) pair: eight thousand of them for g g > g g g
    # g. The optimisation in jamp_optimiser finds the partial sums that several
    # flows have in common and returns them as definitions
    #
    #     TMP(i) = <operand> + frac * <operand>
    #
    # where an operand is either an amplitude or an earlier definition, leaving
    # the color flows as a much shorter combination of those definitions.
    #
    # The fortran output has AMP(NGRAPHS) to read the amplitudes back from, so
    # it prints the definitions as they come. Here every amplitude passes
    # through the single slot amp_sv[0] and is gone by the next diagram, so the
    # definitions are accumulated instead: each amplitude is added into the one
    # definition that uses it while it is still there, and the definitions of
    # definitions follow as soon as everything they need has been seen (see
    # jamp_definition_order).

    @staticmethod
    def jamp_number(value):
        """A C++ literal for a real coefficient, kept exact where it can be
        (an integer, or a ratio the compiler divides out itself)."""

        frac = Fraction(value).limit_denominator(10**9)
        if float(frac) == float(value):
            if frac.denominator == 1:
                return '%d.' % frac.numerator
            return '%d. / %d.' % (frac.numerator, frac.denominator)
        text = '%.17g' % value
        if '.' not in text and 'e' not in text and 'n' not in text:
            text += '.'
        return text

    @classmethod
    def jamp_factor(cls, value):
        """(sign, factor) of a JAMP coefficient, the factor being the C++ text
        multiplying the operand (empty when the coefficient is +-1)."""

        number = complex(value)
        if number.imag == 0:
            magnitude, imaginary = number.real, False
        elif number.real == 0:
            magnitude, imaginary = number.imag, True
        else:
            # never seen in practice: a color coefficient is real or imaginary
            return 1, 'cxtype( %s, %s ) * ' % (cls.jamp_number(number.real),
                                               cls.jamp_number(number.imag))
        sign = -1 if magnitude < 0 else 1
        magnitude = abs(magnitude)
        if magnitude == 1:
            return sign, ('cxtype( 0, 1 ) * ' if imaginary else '')
        if imaginary:
            return sign, '%s * cxtype( 0, 1 ) * ' % cls.jamp_number(magnitude)
        return sign, '%s * ' % cls.jamp_number(magnitude)

    @classmethod
    def jamp_statement(cls, target, terms, assign):
        """'target = t1 - t2;' (assign) or 'target += t1 - t2;', from a list of
        (coefficient, operand) terms."""

        pieces = []
        for pos, (value, name) in enumerate(terms):
            sign, factor = cls.jamp_factor(value)
            if pos == 0 and not assign and len(terms) == 1:
                # the common case: keep the sign on the operator, as the
                # expanded output does
                return '%s %s= %s%s;' % (target, '-' if sign < 0 else '+',
                                         factor, name)
            if pos == 0:
                pieces.append('%s%s%s' % ('-' if sign < 0 else '', factor, name))
            else:
                pieces.append('%s %s%s' % ('-' if sign < 0 else '+', factor, name))
        return '%s %s %s;' % (target, '=' if assign else '+=', ' '.join(pieces))

    def split_order_index(self, matrix_element):
        """{amplitude number -> amplitude-order index}, or None.

        None whenever the jamps are a single vector, i.e. no '^2' constraint or
        only one amplitude order, in which case every jamp index below is the
        plain color index it always was."""
        key = id(matrix_element)
        if getattr(self, '_so_key', None) != key:
            from madgraph.iolibs.export_v4 import split_order_tables
            so = split_order_tables(matrix_element)
            self._so_key = key
            self._so_index = so['amp_so'] if so and so['nampso'] > 1 else None
        return self._so_index

    def build_jamp_plan(self, matrix_element, color_amplitudes):
        """Work out how the color flows are built from shared sub-expressions,
        and return (ntmp, captures, combines, final):
          - captures[n] are the lines to write while amplitude n sits in
            amp_sv[0], as (line, target, is_first_write) so that a masked
            amplitude can be told to zero its target first;
          - combines[n] are the definitions ready once amplitude n has been
            added, to write just after it;
          - final are the lines assembling jamp_sv out of the definitions.
        Returns None when there is nothing to share, so that the caller keeps
        the expanded output."""

        if not self.jamp_optim_enabled():
            return None
        # A shared sub-expression is a partial sum of amplitudes. With the jamps
        # split by amplitude order, two amplitudes of different orders must not
        # end up in the same partial sum -- it would be added to one jamp vector
        # and the order information lost. Write the flows out one amplitude at a
        # time instead.
        if self.split_order_index(matrix_element) is not None:
            return None
        all_element = self.jamp_matrix(color_amplitudes)
        if not all_element:
            return None
        new_mat, defs = self.optimise_jamp_matrix(all_element,
                                                  matrix_element=matrix_element)
        if not defs:
            return None
        order, ready = self.jamp_definition_order(defs)
        definition = {i: (amp1, amp2, frac) for i, amp1, amp2, frac, _nb in defs}

        # what each amplitude has to be added into while it is still in amp_sv
        captures = defaultdict(list)      # amplitude -> [(target, coefficient)]
        for i, amp1, amp2, frac, _nb in defs:
            for amp, coefficient in ((amp1, 1), (amp2, frac)):
                if amp > 0:
                    captures[amp].append(('jampTmp_sv[%d]' % (i - 1), coefficient))
        for (jamp, var), factor in sorted(new_mat.items()):
            if var > 0 and factor:
                captures[var].append(('jamp_sv[%d]' % (jamp - 1), factor))

        # the definitions in the order they become available, grouped by the
        # amplitude they are ready after
        ready_after = defaultdict(list)
        for i in order:
            ready_after[ready[i]].append(i)

        started = set()                   # definitions already assigned to
        done = set()                      # definitions holding their full value
        capture_lines = defaultdict(list)
        combine_lines = defaultdict(list)
        for amp in sorted(set(list(captures) + list(ready_after))):
            for target, coefficient in captures.get(amp, []):
                # jamp_sv is zeroed at the top of the event page, so it is only
                # the definitions that have to start with an assignment
                first = target.startswith('jampTmp_sv') and target not in started
                if first:
                    started.add(target)
                capture_lines[amp].append(
                    (self.jamp_statement(target, [(coefficient, 'amp_sv[0]')],
                                         first), target, first))
            for i in ready_after.get(amp, []):
                amp1, amp2, frac = definition[i]
                operands = [operand for operand in (amp1, amp2) if operand < 0]
                assert all(-operand in done for operand in operands), \
                       'a color-flow definition is used before it is complete'
                done.add(i)
                if not operands:
                    continue      # both operands were amplitudes, already added
                terms = [(coefficient, 'jampTmp_sv[%d]' % (-operand - 1))
                         for operand, coefficient in ((amp1, 1), (amp2, frac))
                         if operand < 0]
                target = 'jampTmp_sv[%d]' % (i - 1)
                first = target not in started
                started.add(target)
                combine_lines[amp].append(
                                self.jamp_statement(target, terms, first))
        assert len(started) == len(defs) == len(done), \
               'a color-flow definition is never written'

        # what is left of the color flows: a combination of the definitions
        final = []
        by_jamp = defaultdict(list)
        for (jamp, var), factor in sorted(new_mat.items()):
            if var < 0 and factor:
                by_jamp[jamp].append((factor, 'jampTmp_sv[%d]' % (-var - 1)))
        for jamp in sorted(by_jamp):
            final.append(self.jamp_statement('jamp_sv[%d]' % (jamp - 1),
                                             by_jamp[jamp], False))
        return len(defs), capture_lines, combine_lines, final

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (improve formatting)
    def super_get_matrix_element_calls(self, matrix_element, color_amplitudes, multi_channel_map):
        """Return a list of strings, corresponding to the Helas calls for the matrix element"""
        import madgraph.core.helas_objects as helas_objects
        import madgraph.loop.loop_helas_objects as loop_helas_objects
        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
               '%s not valid argument for get_matrix_element_calls' % \
               type(matrix_element)
        self.nb_tmp_jamp = 0
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
        # Color flows through shared sub-expressions (None to write them out
        # one (color flow, amplitude) pair at a time, as before)
        # In hr_orig mode they are kept too (hr_jamp_mode): either the color
        # code below runs as it is once per helicity row on a table of that
        # row's amplitudes (rows), or every amplitude is folded into its row's
        # sub-expressions and the rest is done once per row (fold). Without
        # them (expanded) each amplitude goes into every color flow of every
        # row it survives in -- 3.4x the plain build's color updates for
        # g g > t t~ g g g, where no row drops.
        jamp_plan = self.cached_jamp_plan(matrix_element, color_amplitudes)
        self.nb_tmp_jamp = jamp_plan[0] if jamp_plan else 0
        hr_mode = self.hr_jamp_mode() if self.hr_orig else None
        if hr_mode == 'fold':
            self.hr_jamp_post = dict(self.HR_POST_EMPTY, mode='fold',
                                     **dict(zip(('ntmp', 'lines'),
                                                self.hr_post_lines(jamp_plan))))
        elif hr_mode == 'rows':
            # the row loop: what calculate_jamps does with each amplitude, read
            # from the row's table instead of amp_sv[0], and in the same order
            hr_rows = []
            hr_namp = max([a.get('number') for a in
                           matrix_element.get_all_amplitudes()] or [1])
            self.hr_jamp_post = dict(self.HR_POST_EMPTY, mode='rows',
                                     ntmp=self.nb_tmp_jamp, namp=hr_namp,
                                     lines=hr_rows)
        elif hr_mode == 'expanded':
            self.hr_jamp_post = dict(self.HR_POST_EMPTY, mode='expanded')
        so_index = self.split_order_index(matrix_element)
        ncolor_jamp = len(color_amplitudes)
        if jamp_plan is not None:
            _ntmp, jamp_captures, jamp_combines, jamp_final = jamp_plan
        me = matrix_element.get('diagrams')
        matrix_element.reuse_outdated_wavefunctions(me)
        ###misc.sprint(multi_channel_map)
        res = []
        ###res.append('for(int i=0;i<%s;i++){jamp[i] = cxtype(0.,0.);}' % len(color_amplitudes))
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
        # The guards wrap a call in an if/{} that the recycler's line-per-call
        # reader cannot unfold, and they are only an optimisation (a masked term
        # is multiplied by a zero coupling anyway): left out of the hr_orig calls.
        diag_group_mask, wf_group_mask = ({}, {}) if self.hr_orig \
                                     else self._compute_group_masks(matrix_element)

        def _guard_open(group_mask):
            # Emit the opening of an `if` guard for a non-full grouped mask.
            return 'if( ( 0x%xULL >> iflavor ) & 0x1ULL ) {' % group_mask

        # OM - the four gluon optimisation (set merge_quartic_vertices). A quartic
        # current and the cubic current carrying the same colour factor are
        # summed into a third one, which the amplitude reads instead, so that
        # one call gets both contributions and the quartic amplitude is never
        # computed. The sum is written as soon as the later of the two
        # currents is made. Unlike Fortran there is no AMP array here, so the
        # amplitudes which cannot be reached this way are simply left alone:
        # their own colour coefficients put them in the right JAMPs, which is
        # why get_color_amplitudes is asked not to drop them.
        sums, sum_uses, sum_folded = matrix_element.get_quartic_current_sums()
        sum_slots = matrix_element.get_quartic_sum_me_ids()
        sum_written = set()
        sum_after = {}
        for isum, (cubic, quartic, coeff) in enumerate(sums):
            sum_after.setdefault(max(cubic.get('number'),
                                     quartic.get('number')), []).append(isum)

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
                for isum in sum_after.get(wf.get('number'), []):
                    # a wavefunction number can be listed by more than one
                    # diagram here, and the sum must only be written once
                    if isum in sum_written:
                        continue
                    sum_written.add(isum)
                    cubic, quartic, coeff = sums[isum]
                    res.append('%s<W_ACCESS>( aloha_obj[%d], aloha_obj[%d],'
                               ' aloha_obj[%d] );'
                               % ('SUMW_1' if coeff == 1 else 'SUBW_1',
                                  cubic.get('me_id') - 1,
                                  quartic.get('me_id') - 1,
                                  sum_slots[isum] - 1))
            if len(diagram.get('wavefunctions')) == 0 : res.append('// (none)') # AV
            res.append('\n      // Amplitude(s) for diagram number %d' % diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                id_amp +=1
                if amplitude.get('number') in sum_folded:
                    continue    # summed into another amplitude as a current
                namp = amplitude.get('number')
                amplitude.set('number', 1)
                # OM - read the current sum in place of the cubic current it
                # was built from, the same way the Fortran writer does
                sum_original = []
                for mother in amplitude.get('mothers'):
                    isum = sum_uses.get(namp, {}).get(mother.get('number'))
                    if isum is None:
                        continue
                    sum_original.append((mother, mother.get('me_id')))
                    mother.set('me_id', sum_slots[isum])
                amp_block = [ self.get_amplitude_call(amplitude) ] # AV new: avoid format_call
                for mother, me_id in sum_original:
                    mother.set('me_id', me_id)
                if hr_mode == 'rows':
                    res.extend(self._hr_orig_store(amp_block[0], namp, hr_namp))
                    hr_rows.extend(self._hr_row_lines(
                        namp, diagram, id_amp in diag_to_config, color,
                        jamp_captures.get(namp, []) if jamp_plan is not None else None,
                        jamp_combines.get(namp, []) if jamp_plan is not None else []))
                    continue
                if self.hr_orig:
                    res.extend(self._hr_orig_amplitude(
                        amp_block[0], namp, diagram, id_amp in diag_to_config, color,
                        jamp_captures.get(namp, []) if hr_mode == 'fold'
                                                 and jamp_plan is not None else None))
                    continue
                if self.hr_probe:
                    amp_block.append('MG_HR_PROBE_AMP( %d );' % namp)
                if id_amp in diag_to_config:
                    ###res.append("if( channelId == %i ) numerators_sv += cxabs2( amp_sv[0] );" % diag_to_config[id_amp]) # BUG #472
                    ###res.append("if( channelId == %i ) numerators_sv += cxabs2( amp_sv[0] );" % id_amp) # wrong fix for BUG #472
                    diagnum = diagram.get('number')
                    amp_block.append("if( storeChannelWeights )")
                    amp_block.append("{")
                    amp_block.append("  NUM_ATOMIC_ADD( numerators_sv[%i], cxabs2( amp_sv[0] ) );" % (diagnum-1))
                    amp_block.append("}")
                # The amplitude (and the jamp/channel contributions that read its
                # amp_sv[0]) only contributes for the flavors in the diagram's
                # mask, so guard the whole block as a unit.
                gmask = diag_group_mask.get(id(diagram))
                before_guard = []
                if jamp_plan is None:
                    # With split orders the jamps are nampso vectors end to end
                    # and this amplitude belongs to exactly one of them, so its
                    # color flows are offset onto that vector. so_index is None
                    # otherwise and the index is the plain color index.
                    jamp_offset = 0
                    if so_index is not None:
                        jamp_offset = so_index.get(namp, 0) * ncolor_jamp
                    for njamp, coeff in color[namp].items():
                        scoeff = OneProcessExporterMadMatrix.coeff(*coeff) # AV
                        if scoeff[0] == '+' : scoeff = scoeff[1:]
                        scoeff = scoeff.replace('(','( ')
                        scoeff = scoeff.replace(')',' )')
                        scoeff = scoeff.replace(',',', ')
                        scoeff = scoeff.replace('*',' * ')
                        scoeff = scoeff.replace('/',' / ')
                        njamp = njamp + jamp_offset
                        if scoeff.startswith('-'): amp_block.append('jamp_sv[%s] -= %samp_sv[0];' % (njamp, scoeff[1:])) # AV
                        else: amp_block.append('jamp_sv[%s] += %samp_sv[0];' % (njamp, scoeff)) # AV
                else:
                    for line, target, first in jamp_captures.get(namp, []):
                        if first and gmask is not None:
                            # the guard can skip the line that opens this
                            # sub-expression, so it has to start from zero
                            before_guard.append('%s = cxzero_sv();' % target)
                            line = line.replace(' = ', ' += ', 1)
                        amp_block.append(line)
                res.extend(before_guard)
                if gmask is not None:
                    res.append(_guard_open(gmask))
                    res.extend(amp_block)
                    res.append('}')
                else:
                    res.extend(amp_block)
                if jamp_plan is not None:
                    # sub-expressions that have now seen every amplitude they
                    # need: they always run, whatever the flavor
                    res.extend(jamp_combines.get(namp, []))
            if len(diagram.get('amplitudes')) == 0 : res.append('// (none)') # AV
        if hr_mode == 'rows' and jamp_plan is not None:
            hr_rows.extend(jamp_final)
        if jamp_plan is not None and not self.hr_orig:
            res.append('\n      // *** COLOR FLOWS FROM THE SHARED SUB-EXPRESSIONS ***')
            res.extend(jamp_final)
        ###res.append('\n    // *** END OF DIAGRAMS ***' ) # AV - no longer needed ('COLOR MATRIX BELOW')
        return res

    # the per-row tables the hr_orig folds write: the color flows and the shared
    # sub-expressions of the color flows, each [nHrRows][nParity][n]
    _HR_JAMP = re.compile(r'\bjamp_sv\[(\d+)\]')
    _HR_JAMP_TMP = re.compile(r'\bjampTmp_sv\[(\d+)\]')
    _HR_ASSIGN = re.compile(r'^(\s*jampTmp_sv\[\d+\]) = ')
    _HR_AMP = re.compile(r'\bamp_sv\[0\]')

    # What the recycled build does with the color flows (--hel_recycling_jamp):
    #  rows: as the fortran recycler does (AMP(NCOMB,NGRAPHS) and a loop over
    #        the rows) -- every amplitude of every row is stored in a per-row
    #        table (ampAll_sv), and the color flows of the plain build, shared
    #        sub-expressions and all, run once per row afterwards from it;
    #  fold: each amplitude is added into its row's shared sub-expressions
    #        (jampTmpAll_sv) as it is made, the rest is done per row after;
    #  expanded: each amplitude is added into every color flow of its row.
    # rows is the default: with the tabulated P1N groups it executes the fewest
    # instructions of the three on every process measured, is the fastest or
    # tied with fold on one thread, and is by far the quickest to build of the
    # two, since its color code is the plain build's, written once (g g > t t~
    # g g g, check_sa simd_128, cycles against plain: rows 1.67x, fold 1.46x,
    # expanded 1.01x; build 31 s / 114 s / 16 s, plain 13 s). With every core
    # busy (mg7, 18 threads) its gain on g g > t t~ g g g shrinks to ~5% while
    # g g > u u~ g g keeps 1.45x. fold and expanded are kept to compare.
    HR_JAMP_MODES = ('rows', 'fold', 'expanded')
    HR_POST_EMPTY = {'mode': 'expanded', 'ntmp': 0, 'namp': 0, 'lines': ()}

    def hr_jamp_mode(self):
        """--hel_recycling_jamp, rows by default."""
        options = getattr(self, 'cmd_options', None) or {}
        mode = str(options.get('hel_recycling_jamp', 'rows')).lower()
        if mode not in self.HR_JAMP_MODES:
            logger.warning('--hel_recycling_jamp must be one of %s; using rows',
                           ', '.join(self.HR_JAMP_MODES))
            mode = 'rows'
        return mode

    def _hr_orig_store(self, call, namp, nslot):
        """One amplitude in hr_orig form for the rows mode: the call, then its
        store into the row's amplitude table -- nothing else is done with it
        until the row loop (_hr_row_lines)."""
        return ['%s //HRAMP %d' % (call.rstrip('\n'), namp), '//HRFOLD %d' % namp,
                'ampAll_sv[( ${row} * nParity + iParity ) * %d + %d] = amp_sv[0];'
                % (nslot, namp - 1), '//HRENDFOLD']

    def _hr_row_lines(self, namp, diagram, channel, color, captures, combines):
        """What calculate_jamps does with amplitude namp, for the row loop of
        the rows mode: amp_sv[0] is the row's ampRow[namp - 1]."""
        operand = 'ampRow[%d]' % (namp - 1)
        out = []
        if channel:
            out.append('if( storeChannelWeights ) numerators_sv[%d] += cxabs2( %s );'
                       % (diagram.get('number') - 1, operand))
        if captures is not None:
            out += [self._HR_AMP.sub(operand, line.strip())
                    for line, _target, _first in captures]
            out += [line.strip() for line in combines]
            return out
        for njamp, coeff in color[namp].items():
            scoeff = OneProcessExporterMadMatrix.coeff(*coeff)
            if scoeff[0] == '+': scoeff = scoeff[1:]
            for old_, new_ in (('(', '( '), (')', ' )'), (',', ', '),
                               ('*', ' * '), ('/', ' / ')):
                scoeff = scoeff.replace(old_, new_)
            if scoeff.startswith('-'):
                out.append('jamp_sv[%s] -= %s%s;' % (njamp, scoeff[1:], operand))
            else:
                out.append('jamp_sv[%s] += %s%s;' % (njamp, scoeff, operand))
        return out

    def cached_jamp_plan(self, matrix_element, color_amplitudes):
        """build_jamp_plan, once per matrix element: the hr_orig pass and the
        ordinary one ask for the same plan, and at high multiplicity finding it
        is most of the generation time."""
        # the objects themselves, not their id(): the cache holds them alive,
        # so an id cannot come back for another matrix element
        flags = (self.jamp_optim_enabled(), self.jamp_orbit_allowed(matrix_element))
        cache = getattr(self, '_jamp_plan_cache', None)
        if cache is None or cache[0] is not matrix_element \
                or cache[1] is not color_amplitudes or cache[2] != flags:
            cache = (matrix_element, color_amplitudes, flags,
                     self.build_jamp_plan(matrix_element, color_amplitudes))
            self._jamp_plan_cache = cache
        return cache[3]

    @classmethod
    def hr_post_lines(cls, jamp_plan):
        """(ntmp, lines): what is left of the color flows of ONE recycled row once
        every amplitude has been folded into it -- the sub-expressions built out
        of other sub-expressions, in an order that has each complete before it
        is read, then the color flows that are combinations of them. They run
        in place on the row's tables, as jampTmp_sv / jamp_sv, which the folds
        have only ever added to (the tables start at zero and an amplitude that
        vanishes at a row is not folded in at all), so every assignment
        becomes an addition. (0, []) when the color flows share nothing."""
        if jamp_plan is None:
            return 0, []
        ntmp, _captures, combines, final = jamp_plan
        lines = [cls._HR_ASSIGN.sub(r'\1 += ', line)
                 for amp in sorted(combines) for line in combines[amp]]
        return ntmp, lines + list(final)

    def _hr_orig_amplitude(self, call, namp, diagram, channel, color, captures=None):
        """One amplitude in hr_orig form: the call, tagged with its number (all
        amplitudes write the one scratch amp_sv[0], so the call itself cannot say
        which it is), then a //HRFOLD block of what it adds to its helicity row's
        tables -- the multichannel numerator and the color flows -- with a
        ${row} hole the recycler stamps out once per row it survives in.

        With captures (the build_jamp_plan lines of this amplitude) the color
        flows are added to as the plain build adds to them: into the row's
        shared sub-expressions (jampTmpAll_sv) and the few color flows it goes
        to directly; hr_post_lines finishes them per row."""
        out = ['%s //HRAMP %d' % (call.rstrip('\n'), namp), '//HRFOLD %d' % namp]
        slot = '( ${row} * nParity + iParity )'
        if channel:
            out += ['if( storeChannelWeights )', '{',
                    '  numAll_sv[%s * ndiagrams + %i] += cxabs2( amp_sv[0] );'
                    % (slot, diagram.get('number') - 1), '}']
        if captures is not None:
            for line, _target, first in captures:
                if first:
                    line = line.replace(' = ', ' += ', 1)
                # the count as a literal: the chunk translation units do not
                # see HelRecycling.inc, where it is hrNtmp
                line = self._HR_JAMP_TMP.sub(
                    r'jampTmpAll_sv[%s * %d + \1]' % (slot, self.nb_tmp_jamp),
                    line.strip())
                line = self._HR_JAMP.sub(
                    r'jampAll_sv[%s * njampso + \1]' % slot, line)
                out.append(line)
            out.append('//HRENDFOLD')
            return out
        for njamp, coeff in color[namp].items():
            scoeff = OneProcessExporterMadMatrix.coeff(*coeff)
            if scoeff[0] == '+': scoeff = scoeff[1:]
            for old_, new_ in (('(', '( '), (')', ' )'), (',', ', '),
                               ('*', ' * '), ('/', ' / ')):
                scoeff = scoeff.replace(old_, new_)
            target = 'jampAll_sv[%s * njampso + %s]' % (slot, njamp)
            if scoeff.startswith('-'):
                out.append('%s -= %samp_sv[0];' % (target, scoeff[1:]))
            else:
                out.append('%s += %samp_sv[0];' % (target, scoeff))
        out.append('//HRENDFOLD')
        return out

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
        # --axial_gauge: the massless external vectors are written out as
        # vxxxxxr, in the axial gauge of the momentum of another external leg
        # (see helas_call_writers.get_axial_gauge_refs)
        self.axial_gauge = helas_call_writers.axial_gauge_requested(
                                            getattr(self, 'cmd_options', None))
        self.axial_gauge_refs = helas_call_writers.get_axial_gauge_refs(
                            matrix_element) if self.axial_gauge else {}
        try:
            res = self.super_get_matrix_element_calls(matrix_element, color_amplitudes, multi_channel_map)
        finally:
            self.axial_gauge_refs = {}
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
    # (cudacpp's MGONGPU_TEST_DIVERGENCE variant of the first ipzxxx/imzxxx/
    # ixzxxx call is gone with them: get_external_line never emits those)
    def get_external(self, wf, argument):
        line = self.get_external_line(wf, argument)
        split_line = line.split(',')
        split_line = [ str.lstrip(' ').rstrip(' ') for str in split_line] # AV
        # (AV join using ',': no need to add a space as this is done by format_call later on)
        line = ', '.join(split_line)
        line = line.replace( 'xxx(', 'xxx<M_ACCESS, W_ACCESS>(' )
        line = line.replace( 'xxxr(', 'xxxr<M_ACCESS, W_ACCESS>(' )
        line = line.replace( 'w_sv', 'w_fp' )
        text = '%s\n' # AV
        return text % line

    # AV - replace helas_call_writers.GPUFOHelasCallWriter method (vectorize w_sv)
    # This is the method that creates the ixxx/oxxx function calls in calculate_wavefunctions
    # [GPUFOHelasCallWriter.get_external_line is called by GPUFOHelasCallWriter.get_external]
    # [GPUFOHelasCallWriter.get_external (adding #ifdef CUDA) is called by GPUFOHelasCallWriter.generate_helas_call]
    # [GPUFOHelasCallWriter.generate_helas_call is called by UFOHelasCallWriter.get_wavefunction_call/get_amplitude_call]
    def get_external_line(self, wf, argument):
        # NB: always the generic ixxxxx/oxxxxx/vxxxxx/sxxxxx, never the
        # z-axis-optimised ipzxxx/imzxxx/ixzxxx (and their o counterparts) that
        # the base GPUFOHelasCallWriter picks for a massless external fermion.
        # Those assume px == py == 0 and E == +-pz for legs 1 and 2, which only
        # holds in the frames where the beams run along z. The matrix element is
        # called on momenta boosted into the frame the run card's me_frame asks
        # for (madevent: boost_to_frame in genps.f; mg7: the boost_to_frame
        # instruction madspace applies before the call), and in e.g. the rest
        # frame of one final-state particle the incoming partons are not along z
        # at all. Do not reintroduce them without gating on me_frame.
        call = ''
        call = call + helas_call_writers.HelasCallWriter.mother_dict[\
            argument.get_spin_state_number()].lower()
        # Fill out with X up to 6 positions
        call = call + 'x' * (6 - len(call))
        # Axial gauge: a vector leg that got a reference (a massless one, see
        # helas_call_writers.get_axial_gauge_refs) is written out as vxxxxxr,
        # which takes the (lightlike) momentum of another external leg as its
        # gauge reference; any other leg keeps vxxxxx
        axial = argument.get('spin') == 3 and self.has_axial_gauge_ref(wf)
        if axial:
            call = call + 'r'
        # Specify namespace for Helas calls
        call = call + '( momenta,'
        if argument.get('spin') != 1:
            # For non-scalars, need mass and helicity. A massive leg marked
            # off-shell ('*', MadSpin's decaying particles) takes the invariant
            # mass of its momentum, as the Fortran output does (offshell_mass in
            # backend/*/SigmaKin.cc, per event in every backend); a massless one
            # keeps its zero mass rather than a rounding error. 'offshell' is
            # part of the call key, but the leg number and the mass are wf's own:
            # argument is whichever wavefunction first had this key.
            if argument.get('offshell') and str(wf.get('mass')).upper() != 'ZERO':
                call = call + 'offshell_mass<M_ACCESS>( m_pars->%%s, momenta, %d ), hel_sv[%%d],' \
                    % (wf.get('number_external') - 1)
            else:
                call = call + 'm_pars->%s, hel_sv[%d],'
        else:
            # AV This seems to be for scalars (spin==1???), pass neither mass nor helicity (#351)
            ###call = call + 'm_pars->%s,'
            call = call
        # Add flavor and the related ALOHA object
        if axial:
            call = call + '%+d, cFlavors[iflavor][%d], aloha_obj[%d], %d, %d );'
        else:
            call = call + '%+d, cFlavors[iflavor][%d], aloha_obj[%d], %d );'
        if axial:
            return self.format_coupling(call % \
                            (wf.get('mass'),
                                wf.get('number_external')-1,
                                # For boson, need initial/final here
                                (-1) ** (wf.get('state') == 'initial'),
                                wf.get('number_external')-1,
                                wf.get('me_id')-1,
                                wf.get('number_external')-1,
                                self.get_axial_gauge_ref(wf)-1))
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
        mothers, or the mothers of the amplitude, to differentiate which call is
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
            if isinstance(argument, helas_objects.HelasWavefunction) and argument.get('onshell') is False:
                flag.append('P1D') # D is for $ syntax -> offshell propagator only
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
                if aloha.complex_mass:
                    arg['mass'] = 'm_pars->%(CM)s, '
                else:
                    arg['mass'] = 'm_pars->%(M)s, m_pars->%(W)s, '
                if argument.get('onshell') is False:
                    # $-excluded propagator: the run card bw_cutoff, set at run
                    # time through umami_set_parameter (see SigmaKin.cc)
                    arg['mass'] += 'cBWCUTOFF, '
            else:
                #arg['out'] = '&amp_sv[%(out)d]'
                arg['out'] = '&amp_fp[%(out)d]'
                arg['out2'] = 'amp_sv[%(out)d]'
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

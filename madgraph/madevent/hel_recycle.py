#!/usr/bin/env python3

import argparse
import atexit
import os
import re
import collections
from string import Template
from copy import copy
from itertools import product
from functools import reduce 

try:
     import madgraph
except:
     import internal.misc as misc
else:
     import madgraph.various.misc as misc
import mmap
try:
    from tqdm import tqdm
except ImportError:
    tqdm = misc.tqdm

    
def get_num_lines(file_path):
    fp = open(file_path, 'r+')
    buf = mmap.mmap(fp.fileno(),0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class Dialect:
    '''The language-specific text layer of the recycler.

    Everything between reading a call and writing it back -- the DAG, the
    helicity unfolding, the good-helicity and dead-amplitude filters, the
    amplitude split -- is language independent and lives outside this class.
    What a dialect owns is only: how to recognise a call and take it apart, how
    to put it back together, which argument holds what, and how a wavefunction
    or an amplitude is named. Subclass it to recycle a backend written in
    another language (see CppDialect for madmatrix).
    '''

    name = 'fortran'
    # the HELAS routines that build an external wavefunction (compared upper case)
    # VXXXXXR is VXXXXX with an axial-gauge reference momentum appended after
    # nsv; it is still an external wavefunction, and its first arguments are
    # the ones read here (P(0,i) and NHEL(i)).
    external_routines = ('OXXXXX', 'IXXXXX', 'VXXXXX', 'SXXXXX', 'VXXXXXR')
    # which argument of a call receives its result
    output_index = -1
    # which argument of an external call carries the helicity
    helicity_index = 2
    # Whether a physical line can be continued on the next one, so that the
    # reader has to hold a line back before it can act on it.
    needs_lookahead = True
    # Whether the source being read carries the things only a fortran
    # matrix_orig.f has: the DATA (NHEL table, the color flow block, the AMP2
    # block, and the two chunkers that cut the written file up afterwards. A
    # backend that hands its helicity table in and keeps its color stage out of
    # the recycled block sets this False.
    fortran_blocks = True

    # -- taking a call apart -------------------------------------------------
    def called_function(self, line):
        '''The name of the routine this line calls, or None.'''
        if 'CALL' not in line.upper():
            return None
        return get_called_function(line)

    def arguments(self, line):
        '''The call's arguments, in order, blanks stripped.'''
        return get_arguments(line)

    def output_position(self, args):
        '''Which argument of this call receives its result.'''
        return self.output_index

    def output_name(self, args):
        '''The name the call writes its result to.'''
        return args[self.output_position(args)].replace(' ', '')

    def is_amplitude(self, function):
        '''Whether a routine name is an amplitude (rather than a current).'''
        return function.split('_')[-1] == '0'

    # -- putting it back together --------------------------------------------
    def apply_args(self, line, all_args):
        '''`line` re-emitted once per argument list in `all_args`.'''
        return apply_args(line, all_args)

    def wavefunction_name(self, num):
        return 'W(%d)' % num

    def amplitude_name(self, num, diag):
        return 'AMP(%d,%d)' % (num, diag)

    # -- reading the meaning of an argument ----------------------------------
    def external_leg(self, args):
        '''The 1-based leg number an external call builds, read off its
        momentum argument.'''
        return int(re.findall(r'P\(0,(\d+)\)', args[0])[0])

    def external_helicity_leg(self, args):
        '''The 1-based leg whose helicity the external call reads, or None when
        the call takes no helicity at all (a scalar).'''
        helicity = args[self.helicity_index]
        if 'NHEL' not in helicity.upper():
            return None
        return int(re.search(r'\(.*?\)', helicity).group()[1:-1])

    def helicity_literal(self, hel):
        '''How a fixed helicity value is written in place of the table read.'''
        return int_to_string(hel)

    def set_external_helicity(self, args, hel):
        '''`args` with the helicity table read replaced by the fixed value
        `hel`. A scalar has no helicity argument to replace; fortran overwrites
        the slot the output name is about to take anyway, which is why this is
        unconditional here.'''
        args[self.helicity_index] = self.helicity_literal(hel)
        return args

    def amplitude_diagram(self, line, args):
        '''The diagram number an amplitude call writes into.'''
        return int(re.search(r'\(.*?\)', self.output_name(args)).group()[1:-1])

    # -- line shape ----------------------------------------------------------
    def is_continuation(self, line):
        '''Whether this physical line continues the previous one.'''
        try:
            return line[5] == '$'
        except IndexError:
            return False

    def join_continuation(self, line, continuation):
        return undo_multiline(line, continuation)

    def wrap(self, line):
        '''Break a written line so the compiler accepts it.'''
        return do_multiline(line)

    def comment(self, text):
        return '! %s' % text

    def helicity_row(self, row, old_row, hel_comb):
        '''One line of the recycled helicity table: the row's index in the
        ORIGINAL table followed by the helicity of each leg.'''
        formatted_hel = [f'{hel}' if hel < 0 else f' {hel}' for hel in hel_comb]
        return ('      DATA (NHEL(I,%d),I=0,%d) /%d,%s/'
                % (row, len(hel_comb), old_row, ','.join(formatted_hel)))

    def skip_line(self, line):
        '''Whether the reader should ignore this line of the original source
        entirely (a block the dialect has already taken for itself).'''
        return False

    def render_external(self, line, wav):
        '''The text one copy of an external wavefunction turns into.'''
        return self.apply_args(line, [wav.args])

    # -- the amplitude split (P1N) -------------------------------------------
    def render_amplitudes(self, line, new_amps, gauge, amp_splt):
        '''The whole text one amplitude call turns into once unfolded over the
        helicity rows that keep it.'''
        if amp_splt:
            return split_amps(line, new_amps, gauge=gauge)
        return apply_args(line, [amp.args for amp in new_amps])


class CppDialect(Dialect):
    '''The C++ the madmatrix backend writes its HELAS calls in.

    Differences from fortran, all of them local to the text:
      - a call is `NAME<TEMPLATE, ARGS>( a, b, ... );`, so the argument list
        starts after a balanced <...> and commas inside it are not separators;
      - the result is not always the last argument: an external ends with its
        leg index (`..., aloha_obj[5], 4 );`);
      - the helicity is `hel_sv[k]` (or `cHel[ihel][k]`), and a scalar external simply has no
        helicity argument at all rather than a slot to overwrite;
      - every amplitude writes the same scratch `&amp_fp[0]`, so the row an
        unfolded amplitude belongs to cannot be carried by its name. It is
        carried by the object, and the fold of that amplitude into the row's
        color flows -- which the exporter emitted next to the call, with a
        ${row} hole -- is written out here, once per row.
    '''

    name = 'cpp'
    fortran_blocks = False
    needs_lookahead = False
    # the fold of one amplitude into the color flows and the multichannel
    # numerators, as the exporter wrote it next to the call: a ${row} hole per
    # diagram number. Filled by read_folds().
    FOLD_BEGIN = '//HRFOLD'
    FOLD_END = '//HRENDFOLD'
    # the same device for a crossed external, whose NSF blend is more than one
    # line; keyed by the leg rather than the diagram
    EXT_BEGIN = '//HREXT'
    EXT_END = '//HRENDEXT'

    _CALL = re.compile(r'^\s*([A-Za-z_]\w*)\s*[<(]')
    # the helicity of leg k: hel_sv[k] (one per SIMD lane), or cHel[ihel][k]
    _HELICITY = re.compile(r'(?:cHel\s*\[\s*ihel\s*\]|hel_sv)\s*\[\s*(\d+)\s*\]')
    _SLOT = re.compile(r'^&?(?:aloha_obj|amp_fp)\s*\[')

    def __init__(self):
        self.folds = {}
        self.externals = {}
        self._in_block = False
        # Where a line's argument list starts and ends, and what it splits into.
        # Every one of these is asked once per COPY of the call -- get_deps,
        # get_new_args and get_obj each re-ask, and so does every rendering --
        # so at high multiplicity the char loops below are the whole cost of the
        # recycling step. The fortran dialect caches for the same reason.
        self._span_cache = {}
        self._args_cache = {}

    # -- taking a call apart -------------------------------------------------
    def called_function(self, line):
        stripped = line.split('//')[0].strip()
        if not stripped.endswith(';'):
            return None
        match = self._CALL.match(stripped)
        if not match or self._arg_start(stripped) is None:
            return None
        return match.group(1)

    def _span(self, line):
        '''(start, end) of the argument list: the first '(' outside a template
        and the ')' that closes it, or (None, None).'''
        try:
            return self._span_cache[line]
        except KeyError:
            pass
        start = None
        depth = 0
        previous = ''
        for i, char in enumerate(line):
            if char == '<':
                depth += 1
            elif char == '>' and previous != '-':   # not the arrow of a->b
                depth -= 1
            elif char == '(' and depth == 0:
                start = i
                break
            previous = char
        end = None
        if start is not None:
            depth = 0
            for i in range(start, len(line)):
                if line[i] == '(':
                    depth += 1
                elif line[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
        span = (start, end)
        if len(self._span_cache) >= self._CACHE_SIZE:
            self._span_cache.clear()
        self._span_cache[line] = span
        return span

    _CACHE_SIZE = 64

    @classmethod
    def _arg_start(cls, line):
        '''Where the argument list opens (kept for callers that only want it).'''
        return cls()._span(line)[0]

    def arguments(self, line):
        try:
            return list(self._args_cache[line])
        except KeyError:
            pass
        start, _end = self._span(line)
        if start is None:
            return ['']
        depth = 0
        args = ['']
        for char in line[start:]:
            if char in '([{':
                depth += 1
                if depth == 1:
                    continue        # the '(' that opens the list
            elif char in ')]}':
                depth -= 1
                if depth == 0:
                    break           # the ')' that closes it
            elif char == ',' and depth == 1:
                args.append('')
                continue
            if char != ' ':
                args[-1] += char
        if len(self._args_cache) >= self._CACHE_SIZE:
            self._args_cache.clear()
        self._args_cache[line] = args
        return list(args)

    def output_position(self, args):
        for i in range(len(args) - 1, -1, -1):
            if self._SLOT.match(args[i]):
                return i
        return -1

    # -- putting it back together --------------------------------------------
    def apply_args(self, line, all_args):
        start, end = self._span(line)
        head, tail = line[:start], line[end + 1:]
        return ''.join('%s( %s )%s' % (head, ', '.join(args), tail.rstrip('\n'))
                       for args in all_args)

    def wavefunction_name(self, num):
        return 'aloha_obj[%d]' % (num - 1)

    def amplitude_name(self, num, diag):
        # every amplitude goes through the one scratch; which row it belongs to
        # is on the object (set_name keeps it), not in the text
        return '&amp_fp[0]'

    # -- reading the meaning of an argument ----------------------------------
    def external_leg(self, args):
        # the leg index follows the output slot: `..., aloha_obj[5], 4 );`,
        # and vxxxxxr adds its reference leg after it (`..., 4, 0 );`)
        return int(args[self.output_position(args) + 1]) + 1

    def external_helicity_leg(self, args):
        for arg in args:
            match = self._HELICITY.search(arg)
            if match:
                return int(match.group(1)) + 1
        return None

    def helicity_literal(self, hel):
        # the external wavefunctions take a helicity per SIMD lane (fptype_amp_sv):
        # the row's own, the same in every lane
        return '( fptype_amp_sv{} + %d )' % hel

    def set_external_helicity(self, args, hel):
        for i, arg in enumerate(args):
            if self._HELICITY.search(arg):
                args[i] = self.helicity_literal(hel)
                break               # a scalar has none: nothing to replace
        return args

    _AMP_TAG = re.compile(r'//\s*HRAMP\s+(\d+)')

    def amplitude_diagram(self, line, args):
        # the scratch every amplitude writes to cannot say which diagram it is,
        # so the exporter tags the call with it
        return int(self._AMP_TAG.search(line).group(1))

    # -- line shape ----------------------------------------------------------
    def is_continuation(self, line):
        return False

    def wrap(self, line):
        return line

    def comment(self, text):
        return '// %s' % text

    def helicity_row(self, row, old_row, hel_comb):
        # The runtime helicity table (cHel) is not the recycler's to write: it
        # already holds every row of the process. What the recycled block needs
        # is which of those rows it built, and in what order -- the exporter
        # writes that from the same list it hands in as good_elements. Left
        # here as a comment so the generated source says it too.
        return '  // recycled row %d is helicity row %d: %s' % (
            row - 1, old_row - 1, ' '.join('%+d' % hel for hel in hel_comb))

    # -- the per-row fold ----------------------------------------------------
    def read_folds(self, path):
        '''Take the tail blocks out of the original source.

        Some of what a call turns into is not itself a call and so cannot be
        unfolded as one, but still has to be written once per copy: the fold of
        an amplitude into ITS helicity row's color flows, and the NSF blend of a
        crossed external. The exporter writes both next to the call they belong
        to, with ${row} / ${hel} / ${out} holes, and they are read here -- ahead
        of the single streaming pass that then has to skip them -- and stamped
        out by render_amplitudes / render_external below.'''
        self.folds, self.externals = {}, {}
        target, key = None, None
        with open(path) as source:
            for line in source:
                stripped = line.strip()
                if stripped.startswith(self.FOLD_BEGIN):
                    target, key = self.folds, int(stripped[len(self.FOLD_BEGIN):])
                    target[key] = []
                elif stripped.startswith(self.EXT_BEGIN):
                    target, key = self.externals, int(stripped[len(self.EXT_BEGIN):])
                    target[key] = []
                elif stripped.startswith((self.FOLD_END, self.EXT_END)):
                    target = None
                elif target is not None:
                    target[key].append(line.rstrip('\n'))
        return self.folds

    def skip_line(self, line):
        stripped = line.strip()
        if stripped.startswith((self.FOLD_BEGIN, self.EXT_BEGIN)):
            self._in_block = True
            return True
        if stripped.startswith((self.FOLD_END, self.EXT_END)):
            self._in_block = False
            return True
        return self._in_block

    # Below this many amplitudes sharing one partial contraction the split is a
    # net loss: the P1N call and its scratch cost more than the vertex
    # evaluations they save. Measured in the fortran backend, same conclusion.
    # MG_HR_P1N_MIN overrides it, which is how the split is exercised on a
    # process whose sharing is too shallow to reach the default.
    P1N_MIN_SHARING = int(os.environ.get('MG_HR_P1N_MIN', 4))
    # the ALOHAOBJ the partial contraction is written into (declared once by
    # the recycled function's template)
    P1N_SCRATCH = '_p1n'
    _SLOT_INDEX = re.compile(r'aloha_obj\[(\d+)\]')

    def is_wavefunction(self, arg):
        return arg.startswith('aloha_obj[')

    def _amp_fold(self, amp):
        row = amp.numbers[0] - 1
        return [Template(fold).safe_substitute(row=row)
                for fold in self.folds.get(amp.diag_num, ())]

    def _whole_amplitude(self, line, amp):
        return [self.apply_args(line, [amp.args]).rstrip('\n')] + self._amp_fold(amp)

    def _p1n_group(self, line, group, peel):
        """One partial contraction shared by a group of amplitudes.

        The vertex is evaluated once with the peeled leg left off -- that is
        what <ROOT>P1N_<peel+1> is -- and each amplitude of the group is then
        the plain dot product of that current with its own peeled wavefunction.

        The flavour test is NOT uniform across peel positions, and getting it
        backwards is silent: a P1N whose output is a BOSON applies the
        amplitude's own flv_index test itself and zeroes the current (and never
        sets the output's own flv_index), so testing again here would read a
        stale index and could kill a non-zero amplitude; one whose output is a
        FERMION propagates the partner's flv_index instead and defers the test,
        so it has to be made here.
        """
        head = line.split('(')[0]
        match = re.match(r'^(\s*)([A-Za-z_]\w*)_0\s*<([^>]*)>', head)
        if not match:
            return None
        indent, root, template = match.groups()
        access = [t.strip() for t in template.split(',')]
        # the vertex writes an amplitude, the current does not: keep the
        # wavefunction and coupling access classes, drop the amplitude one
        access = [access[0], access[-1]]
        spin = root[peel] if peel < len(root) else None
        if spin not in ('F', 'V', 'S'):
            return None                 # spin 2 / 3/2: not split, as in fortran
        ncomp = 1 if spin == 'S' else 4
        args = list(group[0].args)
        args.pop(peel)
        args[-1] = self.P1N_SCRATCH
        lines = ['%s{' % indent,
                 '%s  %sP1N_%d<%s>( %s );'
                 % (indent, root, peel + 1, ', '.join(access), ', '.join(args)),
                 '%s  const cxtype_sv* _pt = %s::kernelAccessConst( %s.w );'
                 % (indent, access[0], self.P1N_SCRATCH)]
        if self.TABLES:
            table = self._p1n_table(group, peel, indent, access[0], ncomp,
                                    spin == 'F')
            if table is not None:
                return lines[:2] + table + ['%s}' % indent]
        dot = ' + '.join('_pt[%d] * _pw[%d]' % (i, i) for i in range(ncomp))
        for amp in group:
            peeled = amp.args[peel]
            lines.append('%s  { const cxtype_sv* _pw = %s::kernelAccessConst( %s.w );'
                         % (indent, access[0], peeled))
            if spin == 'F':
                lines.append('%s    amp_sv[0] = ( %s.flv_index != %s.flv_index'
                             ' || %s.flv_index == -1 ) ? cxzero_sv() : ( %s ); }'
                             % (indent, self.P1N_SCRATCH, peeled,
                                self.P1N_SCRATCH, dot))
            else:
                lines.append('%s    amp_sv[0] = %s; }' % (indent, dot))
            lines.extend(self._amp_fold(amp))
        lines.append('%s}' % indent)
        return lines

    # -- the tabulated split ---------------------------------------------
    # At high multiplicity the recycled block is mostly the members of the P1N
    # groups -- one dot product per member and the member's fold into its row's
    # color flows -- written out as straight-line code: 90% of the 1.29M
    # statements of g g > t t~ g g g. That code is run once per event page and
    # is far bigger than the caches, so it is fetched from memory every time
    # and the processor idles (IPC 0.7). Written as tables instead -- which
    # wavefunction and row each member is, which table entry and coefficient
    # each fold adds to -- the same work is one small loop (hr_p1n_fold in
    # HelRecyclingTables.h) over data read in order. The folds are read back
    # from the text the exporter wrote; a group with a fold this does not
    # recognise keeps the straight-line form.
    TABLES = True               # the exporter's --hel_recycling_tables
    _FOLD_JAMP = re.compile(r'^\s*(jampAll_sv|jampTmpAll_sv|ampAll_sv)\[\( (\d+) \* nParity'
                            r' \+ iParity \) \* (\w+) \+ (\d+)\] ([+-]?)= (.*)amp_sv\[0\];\s*$')
    _FOLD_NUM = re.compile(r'^\s*numAll_sv\[\( (\d+) \* nParity \+ iParity \)'
                           r' \* ndiagrams \+ (\d+)\] \+= cxabs2\( amp_sv\[0\] \);\s*$')
    _FOLD_SKIP = ('if( storeChannelWeights )', '{', '}')
    _COEF_TEXT = re.compile(r'^[-+0-9.eE*/(), ]*$')
    _COEF_NAMES = re.compile(r'\b(?:cxtype_amp|cxtype|fptype)\b')
    # what a fold entry is packed into: kind (2 bits: jampAll_sv, jampTmpAll_sv,
    # numAll_sv, a store into ampAll_sv), how it adds (2 bits: +=, -=, += real *,
    # += complex *), the coefficient in the group's own table (8 bits), the
    # index (20 bits)
    HR_KINDS = {'jampAll_sv': 0, 'jampTmpAll_sv': 1, 'ampAll_sv': 3}
    HR_NUM = 2
    HR_STORE = 3
    HR_MAX_INDEX = 1 << 20
    HR_MAX_COEF = 1 << 8

    @classmethod
    def fold_coefficient(cls, text):
        """The complex value of the C++ factor in front of amp_sv[0] ('' is 1),
        or None for anything but numbers and the complex constructors."""
        text = text.strip()
        if text.endswith('*'):
            text = text[:-1].strip()
        if text in ('', '+'):
            return complex(1)
        if text == '-':
            return complex(-1)
        if not cls._COEF_TEXT.match(cls._COEF_NAMES.sub('', text)):
            return None
        try:
            value = eval(text, {'__builtins__': {}},
                         {'cxtype_amp': complex, 'cxtype': complex, 'fptype': float})
        except Exception:
            return None
        return complex(value)

    def parse_fold(self, amp):
        """[(kind, index, value)] of the fold of `amp` at its row, value None for
        a multichannel numerator, plus the stride of the jampTmpAll_sv rows (0
        when unused); or None when a line is not one of the shapes the exporter
        writes."""
        row = amp.numbers[0] - 1
        entries, ntmp = [], 0
        for line in self._amp_fold(amp):
            stripped = line.strip()
            if stripped in self._FOLD_SKIP:
                continue
            match = self._FOLD_NUM.match(line)
            if match:
                if int(match.group(1)) != row:
                    return None
                entries.append((self.HR_NUM, int(match.group(2)), None))
                continue
            match = self._FOLD_JAMP.match(line)
            if not match:
                return None
            target, frow, stride, index, sign, factor = match.groups()
            if int(frow) != row:
                return None
            if (target == 'ampAll_sv') != (sign == '' and factor.strip() == ''):
                return None             # a store, and only a store
            if target == 'jampAll_sv':
                if stride != 'njampso':
                    return None
            else:
                if not stride.isdigit() or (ntmp and int(stride) != ntmp):
                    return None
                ntmp = int(stride)
            value = self.fold_coefficient(factor)
            if value is None:
                return None
            entries.append((self.HR_KINDS[target], int(index),
                            -value if sign == '-' else value))
        return entries, ntmp

    @classmethod
    def pack_fold(cls, kind, index, value, coefs):
        """One fold entry as the 32 bits hr_fold_amp reads, adding its
        coefficient to `coefs` (the group's table of distinct values) when it
        needs one; None when it does not fit."""
        if index >= cls.HR_MAX_INDEX:
            return None
        op, coef = 0, 0
        if value is not None and kind != cls.HR_STORE:
            if value == 1:
                op = 0
            elif value == -1:
                op = 1
            else:
                op = 2 if value.imag == 0 else 3
                if value not in coefs:
                    coefs.append(value)
                coef = coefs.index(value)
                if coef >= cls.HR_MAX_COEF:
                    return None
        return (kind << 30) | (op << 28) | (coef << 20) | index

    @staticmethod
    def _double(value):
        text = repr(float(value))
        return text if ('.' in text or 'e' in text or 'n' in text) else text + '.'

    def _p1n_table(self, group, peel, indent, w_access, ncomp, fermion):
        """The members of a P1N group and their folds as tables, and the call
        that runs them (after the P1N call itself); None when a member cannot be
        tabulated."""
        members, folds, coefs, ntmp, table = [], [], [], 0, set()
        for amp in group:
            match = self._SLOT_INDEX.fullmatch(amp.args[peel])
            parsed = self.parse_fold(amp)
            if match is None or parsed is None:
                return None
            entries, stride = parsed
            if stride:
                if ntmp and stride != ntmp:
                    return None
                ntmp = stride
            table.update(kind for kind, _index, _value in entries
                         if kind in (1, self.HR_STORE))
            if len(table) > 1:
                return None             # one second table per group
            packed = [self.pack_fold(kind, index, value, coefs)
                      for kind, index, value in entries]
            if any(p is None for p in packed) or len(packed) >= 1 << 16:
                return None
            members.append('{ %s, %d, %d }' % (match.group(1), amp.numbers[0] - 1,
                                               len(packed)))
            folds.extend(packed)
        cvalues = []
        for value in coefs:
            cvalues += [self._double(value.real), self._double(value.imag)]
        return ['%s  static constexpr HrMember hrM[] = { %s };'
                % (indent, ', '.join(members)),
                '%s  static constexpr unsigned int hrF[] = { %s };'
                % (indent, ', '.join('0x%xu' % f for f in folds) or '0u'),
                '%s  static constexpr double hrC[] = { %s };'
                % (indent, ', '.join(cvalues) or '0.'),
                '%s  hr_p1n_fold<%s, %d, %s>( %s, aloha_obj, hrM, %d, hrF, hrC, %d,'
                ' jampAll_sv, %s, numAll_sv, iParity, storeChannelWeights );'
                % (indent, w_access, ncomp, 'true' if fermion else 'false',
                   self.P1N_SCRATCH, len(members), ntmp or 1,
                   'ampAll_sv' if self.HR_STORE in table else 'jampTmpAll_sv')]

    def render_amplitudes(self, line, new_amps, gauge, amp_splt):
        '''One amplitude, once per helicity row that keeps it: the call into the
        shared scratch, then that row's fold. With amp_splt, the rows that share
        every leg but one get the vertex evaluated once between them.'''
        plan = plan_amp_split(new_amps, self.is_wavefunction) if amp_splt else None
        if plan is None:
            out = []
            for amp in new_amps:
                out.extend(self._whole_amplitude(line, amp))
            return '\n'.join(out) + '\n'

        peel, groups = plan
        out = []
        for group in groups:
            split = (self._p1n_group(line, group, peel)
                     if len(group) >= self.P1N_MIN_SHARING else None)
            if split is None:
                for amp in group:
                    out.extend(self._whole_amplitude(line, amp))
            else:
                out.extend(split)
        return '\n'.join(out) + '\n'

    def render_external(self, line, wav):
        '''One copy of an external: the call, then -- under crossing -- the NSF
        blend that finishes it, told which helicity this copy is and which slot
        it was given.'''
        out = self.apply_args(line, [wav.args]).rstrip('\n')
        tail = self.externals.get(wav.mg - 1)
        if not tail:
            return out
        # set_name was given the wavefunction number, 1-based; the slots it
        # indexes are 0-based
        subs = {'hel': self.helicity_literal(wav.hel),
                'out': wav.numbers[0] - 1}
        return '\n'.join([out] + [Template(t).safe_substitute(subs)
                                   for t in tail])



class DAG:

    def __init__(self):
        self.all_wavs = []
        self.external_wavs = []
        self.internal_wavs = []
        # all_wavs holds every wavefunction ever built, dead ones included, and
        # grows to hundreds of thousands of entries at high multiplicity. Every
        # question ever asked of it is keyed on old_name, so bucket it: a linear
        # scan per HELAS line is quadratic in the file, and it was the second
        # cost centre of the whole recycling step after find_path.
        self.by_old_name = {}
        # The externals each wavefunction depends on, which is the ONLY thing
        # anyone ever wanted the graph for -- good_helicity asked it as
        # find_path(dep, ext) over every (dep, external) pair, i.e. a fresh DFS
        # per pair, 114 million of them on g g > 5g. It needs no search at all:
        # the edges recorded by store_wav go straight from a wavefunction to the
        # externals under it (that is what its caller passes as ext_deps, itself
        # already a transitive closure), and externals have no outgoing edges,
        # so the graph is two levels deep by construction and reachability is a
        # set membership. Verified against find_path over every top-level pair
        # of g g > g g g and g g > g g g g g: same answer everywhere, and no
        # path longer than two nodes exists.
        self.ext_closure = {}
        # Bit i of comb_masks[wav] is set when good_wav_combs[i] contains wav;
        # compat_masks[node] is the AND over its ext_closure, so "does some good
        # helicity combination cover this subtree" is one big-int test instead
        # of a rescan of the whole comb list. Rebuilt by set_good_wav_combs.
        self.comb_masks = {}
        self.compat_masks = {}
        self.full_mask = 0

    def store_wav(self, wav, ext_deps=[]):
        self.all_wavs.append(wav)
        nature = wav.nature
        if nature == 'external':
            self.external_wavs.append(wav)
            # An external is its own only external dependency: find_path(w, w)
            # returned the one-element path [w], which is truthy.
            self.ext_closure[wav] = frozenset((wav,))
        else:
            if nature == 'internal':
                self.internal_wavs.append(wav)
            self.ext_closure[wav] = frozenset(ext_deps)
        try:
            self.by_old_name[wav.old_name].append(wav)
        except KeyError:
            self.by_old_name[wav.old_name] = [wav]

    def dependencies(self, old_name):
        return list(self.by_old_name.get(old_name, ()))

    def kill_old(self, old_name):
        # Every wavefunction under this name dies at once, so the bucket can be
        # emptied rather than filtered later: dead wavefunctions are never
        # resurrected, and dependencies() would drop them anyway. The name stays
        # a key so old_names() keeps reporting it, exactly as the scan over
        # all_wavs (which also kept the dead entries) used to.
        bucket = self.by_old_name.get(old_name)
        if bucket:
            for wav in bucket:
                wav.dead = True
            del bucket[:]

    def old_names(self):
        '''The old names ever stored, live or dead. Callers only intersect it
        with a set, which leaves it untouched -- do not mutate the result.'''
        return self.by_old_name.keys()

    def set_good_wav_combs(self, good_wav_combs):
        '''Index the good external-wavefunction combinations as bitmasks. Called
        whenever External.get_gwc rebuilds them, which is what invalidates the
        cached per-node masks.'''
        self.comb_masks = comb_masks = {}
        self.compat_masks = {}
        for i, comb in enumerate(good_wav_combs):
            bit = 1 << i
            for wav in comb:
                comb_masks[wav] = comb_masks.get(wav, 0) | bit
        self.full_mask = (1 << len(good_wav_combs)) - 1

    def compat_mask(self, node):
        '''The combinations that cover every external under `node`. Zero when
        none does -- with no combinations at all that is every node, which is
        how the old "no comb was a superset" answer came out for an empty
        good_wav_combs.'''
        try:
            return self.compat_masks[node]
        except KeyError:
            pass
        mask = self.full_mask
        comb_masks = self.comb_masks
        for ext in self.ext_closure[node]:
            mask &= comb_masks.get(ext, 0)
            if not mask:
                break
        self.compat_masks[node] = mask
        return mask

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        branches = [(key, sorted(item, key=lambda w: w.name))
                    for key, item in self.ext_closure.items()
                    if item and key.nature != 'external']
        print_str = 'With new names:\n\t'
        print_str += '\n\t'.join([f'{key} : {item}' for key, item in branches])
        print_str += '\n\nWith old names:\n\t'
        print_str += '\n\t'.join([f'{key.old_name} : {[i.old_name for i in item]}' for key, item in branches])
        return print_str



class MathsObject:
    '''Abstract class for wavefunctions and Amplitudes'''

    # Store here which externals the last wav/amp depends on, so that get_obj
    # and get_number do not have to recompute what good_helicity just worked out.
    ext_deps = None

    # The language the calls being recycled are written in. Class level like
    # every other bit of state these classes share (good_hel, num_externals,
    # ...), and reset by HelicityRecycler.__init__ along with the rest.
    dialect = Dialect()

    def __init__(self, arguments, old_name, nature):
        self.args = arguments
        self.old_name = old_name
        self.nature = nature
        self.name = None
        self.dead = False
        self.nb_used = 0
        self.linkdag = []

    def set_name(self, *args):
        # what the name was made of, which is more than the name itself can
        # hold in a language whose amplitudes all share one scratch variable
        self.numbers = args
        out = self.dialect.output_position(self.args)
        self.args[out] = self.format_name(*args)
        self.name = self.args[out]

    def format_name(self, *nums):
        pass

    @classmethod
    def get_deps(cls, line, graph):
        old_args = cls.dialect.arguments(line)
        old_name = cls.dialect.output_name(old_args)
        matches = graph.old_names() & set([old.replace(' ','') for old in old_args])
        try:
            matches.remove(old_name)
        except KeyError:
            pass
        old_deps = old_args[0:len(matches)]

        # If we're overwriting a wav clear it from graph
        graph.kill_old(old_name)
        return [graph.dependencies(dep) for dep in old_deps]

    @classmethod
    def good_helicity(cls, wavs, graph, diag_number=None, all_hel=[], bad_hel_amp=[]):
        # The externals under this combination of dependencies: the union of the
        # closures the DAG already recorded, not a search per (dep, external)
        # pair. See DAG.ext_closure.
        closure = graph.ext_closure
        ext_deps = set()
        for dep in wavs:
            ext_deps |= closure[dep]
        cls.ext_deps = ext_deps
        # "Is ext_deps covered by some good combination" -- an AND of the
        # per-dependency masks, which is the same answer as testing every
        # combination for a superset (a combination covers the union exactly when
        # it covers each closure) but does not rescan the comb list, and reuses
        # the mask each dependency was given the first time it was seen.
        mask = graph.full_mask
        for dep in wavs:
            mask &= graph.compat_mask(dep)
            if not mask:
                break
        this_comb_good = bool(mask)

        if diag_number and this_comb_good and cls.ext_deps:

            helicity = dict([(a.get_id(), a.hel) for a in cls.ext_deps])
            this_hel = [helicity[i] for i in range(1, len(helicity)+1)]
            hel_number = 1 + External.all_hel_index[tuple(this_hel)]
            
            if (hel_number,diag_number) in bad_hel_amp:        
                this_comb_good = False
            

            
        return this_comb_good and cls.ext_deps

    @classmethod
    def get_new_args(cls, line, wavs):
        old_args = cls.dialect.arguments(line)
        # Work out if wavs corresponds to an allowed helicity combination
        this_args = copy(old_args)
        wav_names = [w.name for w in wavs]
        this_args[0:len(wavs)] = wav_names
        # This isnt maximally efficient
        # Could take the num from wavs that've been deleted in graph
        return this_args

    @staticmethod
    def get_number():
        pass

    @classmethod
    def get_obj(cls, line, wavs, graph, diag_num = None):
        old_name = cls.dialect.output_name(cls.dialect.arguments(line))
        new_args = cls.get_new_args(line, wavs)
        num = cls.get_number(wavs, graph)
        
        this_obj = cls.call_constructor(new_args, old_name, diag_num)
        this_obj.set_name(num, diag_num)
        if this_obj.nature != 'amplitude':
            graph.store_wav(this_obj, cls.ext_deps)
        return this_obj


    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class External(MathsObject):
    '''Class for storing external wavefunctions'''

    good_hel = []
    nhel_lines = ''
    num_externals = 0
    # Could get this from dag but I'm worried about preserving order
    wavs_same_leg = {}
    good_wav_combs = []
    max_wav_num = 0
    # helicity tuple -> its row in the original NHEL table, filled by
    # HelicityRecycler.set_helicity_table once that table is complete
    all_hel_index = {}

    def __init__(self, arguments, old_name, hel):
        super().__init__(arguments, old_name, 'external')
        self.hel = int(hel)
        self.mg = self.dialect.external_leg(arguments)
        self.hel_ranges = []
        self.raise_num()

    @classmethod
    def raise_num(cls):
        cls.num_externals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        # If graph is passed in Internal it should be done here to so
        # we can set names
        dialect = cls.dialect
        old_args = dialect.arguments(line)
        old_name = dialect.output_name(old_args)
        graph.kill_old(old_name)

        hel_leg = dialect.external_helicity_leg(old_args)
        if hel_leg is not None:
            ext_num = hel_leg - 1
            new_hels = sorted(list(External.hel_ranges[ext_num]), reverse=True)
        else:
            # Spinor must be a scalar so give it hel = 0
            ext_num = dialect.external_leg(old_args) - 1
            new_hels = [0]

        new_wavfuncs = []
        for hel in new_hels:

            this_args = dialect.set_external_helicity(copy(old_args), hel)

            this_wavfunc = External(this_args, old_name, hel)
            this_wavfunc.set_name(len(graph.external_wavs) + len(graph.internal_wavs) +1)

            graph.store_wav(this_wavfunc)
            new_wavfuncs.append(this_wavfunc)
        if ext_num in cls.wavs_same_leg:
            cls.wavs_same_leg[ext_num] += new_wavfuncs
        else:
            cls.wavs_same_leg[ext_num] = new_wavfuncs
        
        cls.max_wav_num = max( cls.max_wav_num, len(graph.external_wavs) + len(graph.internal_wavs))
        return new_wavfuncs

    @classmethod
    def get_gwc(cls):
        num_combs = len(cls.good_hel)
        gwc_old = [[] for x in range(num_combs)]
        gwc=[]
        for n, comb in enumerate(cls.good_hel):
            sols = [[]]
            for leg, wavs in cls.wavs_same_leg.items():
                valid = []
                for wav in wavs:
                    if comb[leg] == wav.hel:
                        valid.append(wav)
                        gwc_old[n].append(wav)
                if len(valid) == 1:
                    for sol in sols:
                        sol.append(valid[0])
                else:
                    tmp = []
                    for w in valid:
                        for sol in sols:
                            tmp2 = list(sol)
                            tmp2.append(w)
                            tmp.append(tmp2)
                    sols = tmp
            gwc += sols

        cls.good_wav_combs = gwc

    def format_name(self, *nums):
        return self.dialect.wavefunction_name(nums[0])

    def get_id(self):
        """ return the id of the particle under consideration """

        try:
           return self.id
        except:
            self.id = self.mg
            return self.id
        
        

class Internal(MathsObject):
    '''Class for storing internal wavefunctions'''

    max_wav_num = 0
    num_internals = 0

    @classmethod
    def raise_num(cls):
        cls.num_internals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        deps = cls.get_deps(line, graph)
        new_wavfuncs = [ cls.get_obj(line, wavs, graph) 
                         for wavs in product(*deps) 
                         if cls.good_helicity(wavs, graph) ]

        return new_wavfuncs


    # There must be a better way
    @classmethod
    def call_constructor(cls, new_args, old_name, diag_num):
        return Internal(new_args, old_name)

    @classmethod
    def get_number(cls, *args):
        num = External.num_externals + Internal.num_internals + 1
        if cls.max_wav_num < num:
            cls.max_wav_num = num
        return num

    def __init__(self, arguments, old_name):
        super().__init__(arguments, old_name, 'internal')
        self.raise_num()


    def format_name(self, *nums):
        return self.dialect.wavefunction_name(nums[0])

class Amplitude(MathsObject):
    '''Class for storing Amplitudes'''

    max_amp_num = 0

    def __init__(self, arguments, old_name, diag_num):
        self.diag_num = diag_num
        super().__init__(arguments, old_name, 'amplitude')


    def format_name(self, *nums):
        return self.dialect.amplitude_name(nums[0], nums[1])

    @classmethod
    def generate_amps(cls, line, graph, all_hel=None, all_bad_hel=[]):
        args = cls.dialect.arguments(line)
        diag_num = cls.dialect.amplitude_diagram(line, args)

        deps = cls.get_deps(line, graph)

        new_amps = [cls.get_obj(line, wavs, graph, diag_num) 
                        for wavs in product(*deps) 
                        if cls.good_helicity(wavs, graph, diag_num, all_hel,all_bad_hel)]

        return new_amps

    @classmethod
    def call_constructor(cls, new_args, old_name, diag_num):
        return Amplitude(new_args, old_name, diag_num)

    @classmethod
    def get_number(cls, *args):
        wavs, graph = args
        amp_num = -1
        exts = graph.external_wavs        
        hel_amp = tuple([w.hel for w in sorted(cls.ext_deps, key=lambda x: x.mg)])
        amp_num  = External.map_hel[hel_amp] +1 # Offset because Fortran counts from 1

        if cls.max_amp_num < amp_num:
            cls.max_amp_num = amp_num 
        return amp_num  

class HelicityRecycler():
    '''Class for recycling helicity'''

    def __init__(self, good_elements, bad_amps=[], bad_amps_perhel=[], gauge='U',
                 dialect=None):

        # The language of the calls being recycled. Shared with the objects
        # through the class attribute they all inherit, like the rest of the
        # per-run state reset just below.
        self.dialect = dialect if dialect is not None else Dialect()
        MathsObject.dialect = self.dialect

        External.good_hel = []
        External.nhel_lines = ''
        External.num_externals = 0
        External.wavs_same_leg = {}
        External.good_wav_combs = []
        External.all_hel_index = {}
        # read back as NWAVEFUNCS: without the reset a P directory recycled
        # after a bigger one in the same process inherits its array size
        External.max_wav_num = 0

        Internal.max_wav_num = 0
        Internal.num_internals = 0

        Amplitude.max_amp_num = 0
        self.last_category = None
        self.good_elements = good_elements
        # Only ever asked "is it in there": sets, not the lists the callers
        # hand in. good_helicity asks bad_amps_perhel once per unfolded copy of
        # every amplitude -- hundreds of thousands of times against 12612 dead
        # (helicity, amplitude) pairs for g g > g g g g in the axial gauge.
        self.bad_amps = set(bad_amps)
        self.bad_amps_perhel = set(map(tuple, bad_amps_perhel))

        # Default file names
        self.input_file = 'matrix_orig.f'
        self.output_file = 'matrix_orig.f'
        self.template_file = 'template_matrix.f'
        
        self.template_dict = {}
        #initialise everything as for zero matrix element
        self.template_dict['helicity_lines'] = '\n'
        self.template_dict['helas_calls'] = []
        self.template_dict['jamp_lines'] = '\n'
        self.template_dict['amp2_lines'] = '\n'
        self.template_dict['ncomb'] = '0'  
        self.template_dict['nwavefuncs'] = '0' 

        self.dag = DAG()

        self.diag_num = 1
        self.got_gwc = False

        self.procedure_name = self.input_file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'

        self.old_out_name = ''
        self.loop_var = 'K'

        self.all_hel = []
        self.hel_filt = True
        self.gauge = gauge

    def set_input(self, file):
        if 'born_matrix' in file:
            print('HelicityRecycler is currently '
                  f'unable to handle {file}')
            exit(1)
        self.procedure_name = file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'
        self.input_file = file
        # a dialect that owns blocks of the source reads them now, before the
        # single streaming pass that has to skip them
        if hasattr(self.dialect, 'read_folds'):
            self.dialect.read_folds(file)

    def set_output(self, file):
        self.output_file = file
        if os.path.islink(self.output_file):
            os.remove(self.output_file)

    def set_template(self, file):
        self.template_file = file

    def function_call(self, line):
        # Check a function is called at all
        function = self.dialect.called_function(line)
        if not function:
            return None

        # Now check for external spinor
        if function.upper() in self.dialect.external_routines:
            return 'external'

        # Now check for internal
        # Wont find a internal when no externals have been found...
        # ... I assume
        if not self.dag.external_wavs:
            return None

        return 'amplitude' if self.dialect.is_amplitude(function) else 'internal'

    # string manipulation

    def add_amp_index(self, matchobj):
        old_pat = matchobj.group()
        new_pat = old_pat.replace('AMP(', 'AMP( %s,' % self.loop_var)
        
        #new_pat = f'{self.loop_var},{old_pat[:-1]}{old_pat[-1]}'
        return new_pat

    def add_indices(self, line):
        '''Add loop_var index to amp and output variable. 
           Also update name of output variable.'''
        # Doesnt work if the AMP arguments contain brackets.
        # The character in front is looked at rather than eaten, so that an
        # AMP( opening the statement is indexed too -- which is what a line
        # like "AMP(31) = AMP(31) + AMP(1)" needs.
        new_line = re.sub(r'(?<![A-Za-z0-9_])AMP\(.*?\)',
                          self.add_amp_index, line)
        new_line = re.sub(r'MATRIX\d+', 'TS(K)', new_line)
        return new_line

    def jamp_finished(self, line):
        # indent_end = re.compile(fr'{self.jamp_indent}END\W')
        # m = indent_end.match(line)
        # if m:
        #     return True
        return 'init_mode' in line.lower() 
        #if f'{self.old_out_name}=0.D0' in line.replace(' ', ''):
        #    return True
        #return False

    def get_old_name(self, line):
        if f'{self.procedure_kind} {self.procedure_name}' in line:
            if 'SUBROUTINE' == self.procedure_kind:
                self.old_out_name = self.dialect.arguments(line)[-1]
            if 'FUNCTION' == self.procedure_kind:
                self.old_out_name = line.split('(')[0].split()[-1]

    def get_amp_stuff(self, line_num, line):

        if 'diagram number' in line:
            self.amp_calc_started = True
        # Check if the calculation of this diagram is finished
        if ('AMP' not in self.dialect.arguments(line)[-1]
                and self.amp_calc_started and list(line)[0] != 'C'):
            # Check if the calculation of all diagrams is finished
            if self.function_call(line) not in ['external',
                                                'internal',
                                                'amplitude']:
                self.jamp_started = True
            self.amp_calc_started = False
        if self.jamp_started:
            self.get_jamp_lines(line)
        if self.in_amp2:
            self.get_amp2_lines(line)
        if self.find_amp2 and line.startswith('      ENDDO'):
            self.in_amp2 = True
            self.find_amp2 = False

    def get_jamp_lines(self, line):
        if self.jamp_finished(line):
            self.jamp_started = False
            self.find_amp2 = True
        elif not line.isspace():
            self.template_dict['jamp_lines'] += f'{line[0:6]}  {self.add_indices(line[6:])}'

    def get_amp2_lines(self, line):
        if line.startswith('      DO I = 1, NCOLOR'):
            self.in_amp2 = False
        elif not line.isspace() and 'DENOM' not in line:
            self.template_dict['amp2_lines'] += f'{line[0:6]}  {self.add_indices(line[6:])}'

    def prepare_bools(self):
        self.amp_calc_started = False
        self.jamp_started = False
        self.find_amp2 = False
        self.in_amp2 = False
        self.nhel_started = False

    def unfold_helicities(self, line, nature):



        #print('deps',line, deps)
        if nature not in  ['external', 'internal', 'amplitude']:
            raise Exception('wrong unfolding')
        
        if nature == 'external':
            new_objs = External.generate_wavfuncs(line, self.dag)
            for obj in new_objs:
                obj.line = self.dialect.render_external(line, obj)
        else:
            deps = Amplitude.get_deps(line, self.dag)
            name2dep = dict([(d.name,d) for d in sum(deps,[])])
            
            
        if nature == 'internal':
            new_objs = Internal.generate_wavfuncs(line, self.dag)
            for obj in new_objs:
                obj.line = self.dialect.apply_args(line, [obj.args])
                obj.linkdag = []
                for name in obj.args:
                    if name in name2dep:
                        name2dep[name].nb_used +=1
                        obj.linkdag.append(name2dep[name])
                
        if nature == 'amplitude':
            args = self.dialect.arguments(line)
            nb_diag = str(self.dialect.amplitude_diagram(line, args))
            if nb_diag not in self.bad_amps:
                new_objs = Amplitude.generate_amps(line, self.dag, self.all_hel, self.bad_amps_perhel)
                out_line = self.apply_amps(line, new_objs)
                for i,obj in enumerate(new_objs):
                    if i == 0: 
                        obj.line = out_line
                        obj.nb_used = 1
                    else:
                        obj.line = ''
                        obj.nb_used = 1
                    obj.linkdag = []
                    for name in obj.args:
                        if name in name2dep:
                            name2dep[name].nb_used +=1
                            obj.linkdag.append(name2dep[name])
            else:
                return ''

          
        return new_objs
        #return f'{line}\n' if nature == 'external' else line

    def apply_amps(self, line, new_objs):
        return self.dialect.render_amplitudes(line, new_objs, self.gauge,
                                             self.amp_splt)

    def get_gwc(self, line, category):

        #self.last_category = 
        if category not in ['external', 'internal', 'amplitude']:
            return
        if self.last_category != 'external':
            self.last_category = category
            return

        External.get_gwc()
        # The only place the combinations change, so the only place the DAG's
        # bitmask index has to be rebuilt.
        self.dag.set_good_wav_combs(External.good_wav_combs)
        self.last_category = category

    def get_good_hel(self, line):
        if 'DATA (NHEL' in line:
            self.nhel_started = True
            this_hel = [int(hel) for hel in line.split('/')[1].split(',')]
            self.all_hel.append(tuple(this_hel))
        elif self.nhel_started:
            self.nhel_started = False
            self.set_helicity_table()

    def set_helicity_table(self, all_hel=None):
        '''Work out, from the full helicity table, which rows survive the
        good-helicity filter and what each external leg's helicities then range
        over. A backend whose table does not come from the source being read
        (there is no DATA (NHEL to parse in C++) hands it in here instead.'''
        if all_hel is not None:
            self.all_hel = [tuple(hel) for hel in all_hel]

        # the value is the 1-based id of the helicity in the original
        # NHEL table: SMATRIX returns it (NHEL(0,I)) as the selected
        # helicity, which get_nhel then reads back for the LHE spin column
        if self.hel_filt:
            hel_ids = [int(i) for i in self.good_elements]
        else:
            hel_ids = range(1, len(self.all_hel)+1)
        External.good_hel = dict([(self.all_hel[i-1], i) for i in hel_ids])

        External.map_hel=dict([(hel,i) for i,hel in  enumerate(External.good_hel)])
        # good_helicity needs the position of a helicity tuple in the FULL
        # table (not the filtered one map_hel indexes) once per amplitude it
        # unfolds; that was a scan of the 128-row table 811 000 times over
        # g g > g g g g g.
        External.all_hel_index = dict([(hel,i) for i,hel in enumerate(self.all_hel)])
        External.hel_ranges = [set() for hel in next(iter(External.good_hel))]
        for comb in External.good_hel:
            for i, hel in enumerate(comb):
                External.hel_ranges[i].add(hel)

        self.counter = 0
        nhel_array = [self.nhel_string(hel)
                      for hel in External.good_hel]
        nhel_lines = '\n'.join(nhel_array)
        self.template_dict['helicity_lines'] += nhel_lines

        self.template_dict['ncomb'] = len(External.good_hel)

    def nhel_string(self, hel_comb):
        old_id = External.good_hel[hel_comb]
        self.counter += 1
        return self.dialect.helicity_row(self.counter, old_id, hel_comb)

    def read_orig(self):

        with open(self.input_file, 'r') as input_file:

            self.prepare_bools()

            lookahead = self.dialect.needs_lookahead
            for line_num, line in tqdm(enumerate(input_file), total=get_num_lines(self.input_file)):
                if lookahead and line_num == 0:
                    line_cache = line
                    continue

                if '!SKIP' in line:
                    continue

                if self.dialect.skip_line(line):
                    continue

                if lookahead:
                    if self.dialect.is_continuation(line):
                        line_cache = self.dialect.join_continuation(line_cache, line)
                        continue
                    line, line_cache = line_cache, line

                if self.dialect.fortran_blocks:
                    self.get_old_name(line)
                    self.get_good_hel(line)
                    self.get_amp_stuff(line_num, line)
                call_type = self.function_call(line)
                self.get_gwc(line, call_type)

                
                if call_type in ['external', 'internal', 'amplitude']:
                    self.template_dict['helas_calls'] += self.unfold_helicities(
                        line, call_type)

        self.template_dict['nwavefuncs'] = max(External.num_externals, Internal.max_wav_num, External.max_wav_num)
        # filter out uselless call
        for i in range(len(self.template_dict['helas_calls'])-1,-1,-1):
            obj = self.template_dict['helas_calls'][i]
            if obj.nb_used == 0:
                obj.line = ''
                for dep in obj.linkdag:
                    dep.nb_used -= 1

        
        
        comment = self.dialect.comment
        self.template_dict['helas_calls'] = '\n'.join(
            ['%s %s' % (obj.line.rstrip(), comment('count %d' % obj.nb_used))
             for obj in self.template_dict['helas_calls']
             if obj.nb_used > 0 and obj.line])

    def read_template(self):
        out_file = open(self.output_file, 'w+')
        with open(self.template_file, 'r') as file:
            for line in file:
                s = Template(line)
                line = s.safe_substitute(self.template_dict)
                line = '\n'.join([self.dialect.wrap(sub_lines)
                                  for sub_lines in line.split('\n')])
                out_file.write(line)
        out_file.close()

    def write_zero_matrix_element(self):
        try:
      	    os.remove(self.output_file)
        except Exception:
            pass
        input_file = self.output_file.replace("_optim.f", "_orig.f")
        os.symlink(input_file, self.output_file)


    def generate_output_file(self):
        if not self.good_elements:
            misc.sprint("No helicity", self.input_file)
            self.write_zero_matrix_element()
            return
        
        atexit.register(self.clean_up)
        self.read_orig()
        self.read_template()
        atexit.unregister(self.clean_up)

    def clean_up(self):
        pass


# get_arguments walks its line character by character, and unfold_helicities
# asks it again for every object it unfolds out of that line: 905 351 calls over
# the 8 143 HELAS lines of g g > g g g g g, all but ~50 000 of them a repeat of
# the line just parsed, and 3.7 s of a 16.8 s (profiled) recycling step. Keep the
# last few answers -- the callers walk the file one line at a time, so a handful
# of slots is all it takes -- and hand out a copy, since a shared mutable list is
# not what a caller that goes on to substitute arguments into it expects.
_ARGUMENT_CACHE = {}
_ARGUMENT_CACHE_SIZE = 32


def get_arguments(line):
    '''Find the substrings separated by commas between the first
    closed set of parentheses in 'line'.
    '''
    try:
        return list(_ARGUMENT_CACHE[line])
    except KeyError:
        pass
    arguments = parse_arguments(line)
    if len(_ARGUMENT_CACHE) >= _ARGUMENT_CACHE_SIZE:
        _ARGUMENT_CACHE.clear()
    _ARGUMENT_CACHE[line] = arguments
    return list(arguments)


def parse_arguments(line):
    '''The uncached get_arguments.'''
    start_idx = None
    call_idx = line.upper().find('CALL ')
    if call_idx != -1:
        start_idx = line.find('(', call_idx)
    if start_idx is None or start_idx == -1:
        start_idx = line.find('(')
    if start_idx == -1:
        return ['']

    bracket_depth = 0
    element = 0
    arguments = ['']
    for i, char in enumerate(line):
        if i < start_idx:
            continue
        if char == '(':
            bracket_depth += 1
            if bracket_depth - 1 == 0:
                # This is the first '('. We don't want to add it to
                # 'arguments'
                continue
        if char == ')':
            bracket_depth -= 1
            if bracket_depth == 0:
                # We've reached the end
                break
        if char == ',' and bracket_depth == 1:
            element += 1
            arguments.append('')
            continue
        if bracket_depth > 0 and char != ' ':
            arguments[element] += char
    return arguments


def apply_args(old_line, all_the_args):
    call_idx = old_line.upper().find('CALL ')
    if call_idx == -1:
        function = (old_line.split('(')[0]).split()[-1]
        old_args = old_line.split(function)[-1]
        new_lines = [old_line.replace(old_args, f'({",".join(x)})\n')
                     for x in all_the_args]
        return ''.join(new_lines)

    call_arg_start = old_line.find('(', call_idx)
    if call_arg_start == -1:
        return old_line

    bracket_depth = 0
    call_arg_end = -1
    for i, char in enumerate(old_line[call_arg_start:], start=call_arg_start):
        if char == '(':
            bracket_depth += 1
        elif char == ')':
            bracket_depth -= 1
            if bracket_depth == 0:
                call_arg_end = i
                break
    if call_arg_end == -1:
        return old_line

    call_head = old_line[:call_arg_start]
    call_tail = old_line[call_arg_end+1:]
    new_lines = [f'{call_head}({",".join(args)}){call_tail}'
                 for args in all_the_args]
    
    return ''.join(new_lines)

def get_called_function(line):
    call_idx = line.upper().find('CALL ')
    if call_idx == -1:
        return None
    after_call = line[call_idx+5:]
    if '(' not in after_call:
        return None
    return after_call.split('(', 1)[0].strip().split()[-1]

def plan_amp_split(new_amps, is_wavefunction):
    """How to split a set of unfolded amplitudes into partial contractions.

    They all evaluate the same vertex at different helicities, so the column
    that carries the MOST distinct wavefunctions is the one worth peeling: the
    contraction of everything else is then shared by every amplitude that
    agrees on the remaining columns. Returns (peeled column, [group, ...]),
    each group a list of amplitudes in their original order, or None when there
    is no wavefunction column to peel.
    """
    if not new_amps:
        return None
    columns = [i for i, a in enumerate(new_amps[0].args) if is_wavefunction(a)]
    if not columns:
        return None
    occur = [collections.defaultdict(int) for _ in columns]
    for amp in new_amps:
        for j, i in enumerate(columns):
            occur[j][amp.args[i]] += 1
    peel = columns[[len(o) for o in occur].index(max(len(o) for o in occur))]
    rest = [o for j, o in enumerate(occur) if columns[j] != peel]

    # Which amplitudes carry a given wavefunction, one bit per amplitude. The
    # selection below was a rescan of every amplitude for every combination of
    # columns -- 12.5 million `w in amp.args` evaluations on g g > g g g g g,
    # the largest single cost left in the recycling step -- and it is an AND of
    # these masks instead. A HELAS call never uses the same wavefunction twice,
    # so a name identifies the column it came from and asking "is w anywhere in
    # this amplitude" is the same question as asking its column.
    amp_masks = {}
    for i, amp in enumerate(new_amps):
        bit = 1 << i
        for a in amp.args:
            amp_masks[a] = amp_masks.get(a, 0) | bit
    all_amps_mask = (1 << len(new_amps)) - 1

    groups = []
    for wfcts in product(*[o.keys() for o in rest]):
        # Select the amplitudes produced by wfcts
        mask = all_amps_mask
        for w in wfcts:
            mask &= amp_masks.get(w, 0)
            if not mask:
                break
        if not mask:
            continue
        # lowest bit first, so the group keeps the order of new_amps
        group = []
        while mask:
            low = mask & -mask
            group.append(new_amps[low.bit_length() - 1])
            mask ^= low
        groups.append(group)
    return peel, groups


def split_amps(line, new_amps, gauge):
    if not new_amps:
        return ''
    call_idx = line.upper().find('CALL ')
    call_arg_start = line.find('(', call_idx) if call_idx != -1 else -1
    called_function = get_called_function(line)
    if call_idx == -1 or call_arg_start == -1 or not called_function:
        return ''
    call_prefix = line[:call_idx]
    call_keyword = line[call_idx:call_idx+5]
    function_root = called_function.split('_0')[0]
    indent = re.match(r'\s*', call_prefix).group(0)
    guard_stmt = call_prefix.strip()
    guarded_call = guard_stmt.upper().startswith('IF')
    call_stmt_prefix = call_prefix if not guarded_call else (indent + '  ')
    fct = '%s%s%s' % (call_stmt_prefix, call_keyword, function_root)
    plan = plan_amp_split(new_amps, lambda a: "W(" in a)
    if plan is None:
        return ''
    to_remove, groups = plan

    lines = []
    for sub_amps in groups:
        if len(sub_amps) ==1:
            lines.append(apply_args(line, [i.args for i in sub_amps]).replace('\n',''))
            
            continue
                         
        # the next line is to make the code nicer 
        sub_amps.sort(key=lambda a: int(a.args[-1][:-1].split(',',1)[1]))
        windices = []
        hel_calculated = []
        iamp = 0
        local_lines = []
        for i,amp in enumerate(sub_amps):
            args = amp.args[:]   
            # Remove wav and get its index
            wcontract = args.pop(to_remove)
            windex = wcontract.split('(')[1].split(')')[0]
            windices.append(windex)
            amp_result,  args[-1]  =  args[-1], 'TMP(1)'
            
            if i ==0:
                # Call the original fct with P1N_...
                # Final arg is replaced with TMP(1)
                spin = function_root[to_remove]
                local_lines.append('%sP1N_%s(%s)' % (fct, to_remove+1, ', '.join(args)))

            hel, iamp = re.findall(r'AMP\((\d+),(\d+)\)', amp_result)[0]
            hel_calculated.append(hel)
            #lines.append(' %(result)s = TMP(3) * W(3,%(w)s) + TMP(4) * W(4,%(w)s)+'
            #             % {'result': amp_result, 'w':  windex}) 
            #lines.append('     &             TMP(5) * W(5,%(w)s)+TMP(6) * W(6,%(w)s)'
            #             % {'result': amp_result, 'w':  windex})
        if spin == "F" or ( spin == "V" and gauge !='FD'):
            suffix = ''
        elif spin == "S":
            suffix = 'S'
        elif spin == "V" and  gauge == "FD":
            suffix = "FD"
        else:
            raise Exception("split amp not supported for spin2, 3/2")

        local_lines.append("""%(call_prefix)s%(call_keyword)sCombineAmp%(suffix)s(%(nb)i,
     & (/%(hel_list)s/), 
     & (/%(w_list)s/),
     & TMP, W, AMP(1,%(iamp)s))""" % {'suffix':suffix,
                                      'call_prefix': call_stmt_prefix,
                                      'call_keyword': call_keyword,
                                      'nb': len(sub_amps),
                                      'hel_list': ','.join(hel_calculated),
                                      'w_list': ','.join(windices),
                                      'iamp': iamp
                                     })
        if guarded_call:
            if not guard_stmt.upper().endswith('THEN'):
                guard_stmt = '%s THEN' % guard_stmt
            lines.append('%s%s' % (indent, guard_stmt))
            lines.extend(local_lines)
            lines.append('%sENDIF' % indent)
        else:
            lines.extend(local_lines)

            
    #lines.append('')
    return '\n'.join(lines)

def get_num(wav):
    name = wav.name
    between_brackets = re.search(r'\(.*?\)', name).group()
    num = int(between_brackets[1:-1].split(',')[-1])    
    return num

def undo_multiline(old_line, new_line):
    new_line = new_line[6:]
    old_line = old_line.replace('\n','')
    return f'{old_line}{new_line}'

def do_multiline(line):
    if "!" in line:
        line,comment  = line.split("!",1)
    else: 
        comment = None
    char_limit = 72
    if len(line) > char_limit:
        split_line = []
        remaining = line
        while len(remaining) > char_limit:
            split_at = remaining.rfind(' ', 0, char_limit + 1)
            # A split which leaves nothing but blanks on the current line --
            # the only space is the statement's own indentation, as for the
            # space-free Kleiss-Kuijf JAMPF lines -- emits an empty physical
            # line, and the continuation which follows it is then attached to
            # the *previous* statement. Break mid-token instead.
            if split_at <= 0 or not remaining[:split_at+1].strip():
                split_line.append(remaining[:char_limit])
                remaining = remaining[char_limit:]
            else:
                split_line.append(remaining[:split_at+1])
                remaining = remaining[split_at+1:]
        split_line.append(remaining)
        indent = ''
        for char in line[6:]:
            if char == ' ':
                indent += char
            else:
                break

        line = f'\n     ${indent}'.join(split_line)
    if not comment:
        return line
    else:
        return f"{line} ! {comment}"
def int_to_string(i):
    if i == 1:
        return '+1'
    if i == 0:
        return ' 0'
    if i == -1:
        return '-1'
    else:
        print(f'How can {i} be a helicity?')
        set_trace()
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The file containing the '
                                          'original matrix calculation')
    parser.add_argument('hel_file', help='The file containing the '
                                         'contributing helicities')
    parser.add_argument('--hf-off', dest='hel_filt', action='store_false', default=True, help='Disable helicity filtering')
    parser.add_argument('--as-off', dest='amp_splt', action='store_false', default=True, help='Disable amplitude splitting')

    args = parser.parse_args()

    with open(args.hel_file, 'r') as file:
        good_elements = file.readline().split()

    recycler = HelicityRecycler(good_elements)

    recycler.hel_filt = args.hel_filt
    recycler.amp_splt = args.amp_splt

    recycler.set_input(args.input_file)
    recycler.set_output('green_matrix.f')
    recycler.set_template('template_matrix1.f')

    recycler.generate_output_file()

if __name__ == '__main__':
    main()

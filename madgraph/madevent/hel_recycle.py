#!/usr/bin/env python3

import argparse
import atexit
import re
from string import Template
from copy import copy
from itertools import product

# Remove
import mmap
import tqdm
def get_num_lines(file_path):
    fp = open(file_path, 'r+')
    buf = mmap.mmap(fp.fileno(),0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class DAG:

    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        self.graph[node] = []

    def add_branch(self, node_i, node_f):
        self.graph[node_i].append(node_f)

    def external_nodes(self):
        exts = [key for key, value in self.graph.items()
                if key.nature == 'external']
        return exts

    def dependencies(self, old_name):
        deps = [key for key, value in self.graph.items()
                if key.old_name == old_name]
        return deps

    def clear_old(self, old_name):
        for key, value in list(self.graph.items()):
            if key.old_name == old_name:
                del self.graph[key]
                continue
            for i in reversed(range(len(value))):
                if value[i].old_name == old_name:
                    del self.graph[key][i]

    def clear_amp(self, diag_num):
        for key, value in list(self.graph.items()):
            if key.nature == 'amplitude' and key.diag_num <= diag_num:
                del self.graph[key]
                continue
            for i in reversed(range(len(value))):
                if (value[i].nature == 'amplitude' and
                        value[i].diag_num <= diag_num):
                    del self.graph[key][i]

    def old_names(self):
        return {key.old_name for key, value in self.graph.items()}

    def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in self.graph:
            return None
        for node in self.graph[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)
                if newpath:
                    return newpath
        return None

    def __str__(self):
        print(self.graph)
        return ''

    def __repr__(self):
        return self.graph


class MathsObject:
    '''Abstract class for wavefunctions and Amplitudes'''

    def __init__(self, arguments, old_name, nature):
        self.args = arguments
        self.old_name = old_name
        self.nature = nature
        self.name = None

    def set_name(self, *args):
        self.args[-1] = self.format_name(*args)
        self.name = self.args[-1]

    def format_name(self, *nums):
        pass

    @staticmethod
    def get_deps(line, graph):
        old_args = get_arguments(line)
        old_name = old_args[-1]
        matches = graph.old_names() & set(old_args)
        try:
            matches.remove(old_name)
        except KeyError:
            pass
        old_deps = old_args[0:len(matches)]

        # If we're overwriting a wav clear it from graph
        graph.clear_old(old_name)
        return [graph.dependencies(dep) for dep in old_deps]

    @staticmethod
    def good_helicity(wavs, graph):
        exts = graph.external_nodes()
        exts_on_path = { i for dep in wavs for i in exts if graph.find_path(i, dep) }
        this_wav_comb = [comb for comb in External.good_wav_combs
                         if exts_on_path.issubset(set(comb))]
        return this_wav_comb and exts_on_path

    @staticmethod
    def get_new_args(line, wavs):
        old_args = get_arguments(line)
        old_name = old_args[-1]
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
        old_name = get_arguments(line)[-1]
        new_args = cls.get_new_args(line, wavs)
        num = cls.get_number(wavs, graph)
        this_obj = cls.call_constructor(new_args, old_name, diag_num)
        this_obj.set_name(num, diag_num)
        graph.add_node(this_obj)
        [graph.add_branch(w, this_obj) for w in wavs]
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
    paired_up_wavs = []
    good_wav_combs = []

    def __init__(self, arguments, old_name):
        super().__init__(arguments, old_name, 'external')
        self.hel = int(self.args[2])
        self.raise_num()

    @classmethod
    def raise_num(cls):
        cls.num_externals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        # If graph is passed in Internal it should be done here to so
        # we can set names
        old_args = get_arguments(line)
        old_name = old_args[-1]

        new_wavfuncs = []

        for new_hel in ['+1', '-1']:

            this_args = copy(old_args)
            this_args[2] = new_hel

            this_wavfunc = External(this_args, old_name)
            this_wavfunc.set_name(len(graph.external_nodes())+1)

            graph.add_node(this_wavfunc)
            new_wavfuncs.append(this_wavfunc)

        cls.paired_up_wavs.append(new_wavfuncs)
        return new_wavfuncs

    @classmethod
    def get_gwc(cls):
        rows = len(cls.good_hel)
        columns = len(cls.good_hel[0])
        # TODO: is it better to have list of sets?
        wav_comb = [[] for x in range(rows)]
        # TODO: CHECK SHAPE OF HEL MAKES SENSE AND SHAPE OF SPINOR_COMB IS SAME
        for i, j in product(range(rows), range(columns)):
            for wav in cls.paired_up_wavs[j]:
                if cls.good_hel[i][j] == wav.hel:
                    wav_comb[i].append(wav)
        cls.good_wav_combs = wav_comb

    @staticmethod
    def format_name(*nums):
        return f'W(1,{nums[0]})'


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


    @staticmethod
    def format_name(*nums):
        return f'W(1,{nums[0]})'

class Amplitude(MathsObject):
    '''Class for storing Amplitudes'''

    max_amp_num = 0

    def __init__(self, arguments, old_name, diag_num):
        self.diag_num = diag_num
        super().__init__(arguments, old_name, 'amplitude')


    @staticmethod
    def format_name(*nums):
        return f'AMP({nums[1]},{nums[0]})'

    @classmethod
    def generate_amps(cls, line, graph):
        old_args = get_arguments(line)
        old_name = old_args[-1]

        amp_index = re.search(r'\(.*?\)', old_name).group()
        diag_num = int(amp_index[1:-1])
        graph.clear_amp(diag_num)

        deps = cls.get_deps(line, graph)

        new_amps = [cls.get_obj(line, wavs, graph, diag_num) 
                        for wavs in product(*deps) 
                        if cls.good_helicity(wavs, graph)]

        return new_amps

    @classmethod
    def call_constructor(cls, new_args, old_name, diag_num):
        return Amplitude(new_args, old_name, diag_num)

    @classmethod
    def get_number(cls, *args):
        wavs, graph = args
        amp_num = -1
        exts = graph.external_nodes()
        # Update good_wavs_combs to not include wavs that 
        # have been overwritten
        good_wav_combs = [{wav for wav in comb if wav in exts} for comb in External.good_wav_combs]
        exts_on_path = { i for dep in wavs for i in exts if graph.find_path(i, dep) }
        for i in range(len(External.good_wav_combs)):
            if good_wav_combs[i] == set(exts_on_path):
                # Offset because Fortran counts from 1
                amp_num = i + 1
        if amp_num < 1:
            print('Failed to find amp_num')
            exit(1)
        if cls.max_amp_num < amp_num:
            cls.max_amp_num = amp_num 
        return amp_num  

class HelicityRecycler():
    '''Class for recycling helicity'''

    def __init__(self, good_elements):

        External.good_hel = []
        External.nhel_lines = ''
        External.num_externals = 0
        External.paired_up_wavs = []
        External.good_wav_combs = []

        Internal.max_wav_num = 0
        Internal.num_internals = 0

        Amplitude.max_amp_num = 0

        self.good_elements = good_elements

        # Default file names
        self.input_file = 'matrix_orig.f'
        self.output_file = 'matrix_orig.f'
        self.template_file = 'template_matrix.f'
        
        self.template_dict = {}
        self.template_dict['helicity_lines'] = '\n'
        self.template_dict['helas_calls'] = '\n'
        self.template_dict['jamp_lines'] = '\n'
        self.template_dict['amp2_lines'] = '\n'

        self.dag = DAG()

        self.diag_num = 1
        self.got_gwc = False

        self.procedure_name = self.input_file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'

        self.old_out_name = ''
        self.loop_var = 'K'

        self.all_hel = []

    def set_input(self, file):
        if 'born_matrix' in file:
            print('HelicityRecycler is currently '
                  f'unable to handle {file}')
            exit(1)
        self.procedure_name = file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'
        self.input_file = file

    def set_output(self, file):
        self.output_file = file

    def set_template(self, file):
        self.template_file = file

    def function_call(self, line):
        # Check a function is called at all
        if not 'CALL' in line:
            return None

        # Now check for spinor
        if ('CALL OXXXXX' in line or 'CALL IXXXXX' in line or 'CALL VXXXXX' in line):
            return 'external'

        # Now check for internal
        # Wont find a internal when no externals have been found...
        # ... I assume
        if not self.dag.external_nodes():
            return None

        # Search for internals by looking for calls to the externals
        # Maybe I should just get a list of all internals?
        matches = self.dag.old_names() & set(get_arguments(line))
        try:
            matches.remove(get_arguments(line)[-1])
        except KeyError:
            pass
        try:
            function = (line.split('(', 1)[0]).split()[-1]
        except IndexError:
            return None
        # What if [-1] is garbage? Then I'm relying on needs changing.
        # Is that OK?
        if (len(matches) in [2, 3]) and (function.split('_')[-1] != '0'):
            return 'internal'
        elif (len(matches) in [3, 4] and (function.split('_')[-1] == '0')):
            return 'amplitude'
        else:
            print(f'Ahhhh what is going on here?\n{line}')

        return None

    # string manipulation

    def add_amp_index(self, matchobj):
        old_pat = matchobj.group()
        new_pat = f'{old_pat[:-1]},{self.loop_var}{old_pat[-1]}'
        return new_pat

    def add_indices(self, line):
        '''Add loop_var index to amp and output variable. 
           Also update name of output variable.'''
        # Doesnt work if the AMP arguments contain brackets
        new_line = re.sub(r'\WAMP\(.*?\)', self.add_amp_index, line)
        return new_line

    def jamp_finished(self, line):
        # indent_end = re.compile(fr'{self.jamp_indent}END\W')
        # m = indent_end.match(line)
        # if m:
        #     return True
        if f'{self.old_out_name}=0.D0' in line.replace(' ', ''):
            return True
        return False

    def get_old_name(self, line):
        if f'{self.procedure_kind} {self.procedure_name}' in line:
            if 'SUBROUTINE' == self.procedure_kind:
                self.old_out_name = get_arguments(line)[-1]
            if 'FUNCTION' == self.procedure_kind:
                self.old_out_name = line.split('(')[0].split()[-1]

    def get_amp_stuff(self, line_num, line):

        if 'diagram number' in line:
            self.amp_calc_started = True
        # Check if the calculation of this diagram is finished
        if ('AMP' not in get_arguments(line)[-1]
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
        elif not line.isspace():
            self.template_dict['amp2_lines'] += f'{line[0:6]}  {self.add_indices(line[6:])}'

    def prepare_bools(self):
        self.amp_calc_started = False
        self.jamp_started = False
        self.find_amp2 = False
        self.in_amp2 = False
        self.nhel_started = False

    def unfold_amp(self, line):
        new_amps = Amplitude.generate_amps(line, self.dag)
        line = apply_args(line, [i.args for i in new_amps])
        return line

    def unfold_internal(self, line):
        new_wavs = Internal.generate_wavfuncs(line, self.dag)
        line = apply_args(line, [i.args for i in new_wavs])
        return line

    def unfold_external(self, line):
        new_wavs = External.generate_wavfuncs(line, self.dag)
        line = apply_args(line, [i.args for i in new_wavs])
        return f'{line}\n'

    def get_gwc(self, line):
        if self.got_gwc:
            return
        num_found = len([line 
                         for line in self.template_dict['helas_calls'].splitlines() 
                         if line.strip() != ''])
        try:
            num_exts = len(External.good_hel[0])
        except IndexError:
            return
        if num_found == 2*num_exts:
            self.got_gwc=True
            External.get_gwc()

    def get_good_hel(self, line):
        if 'DATA (NHEL' in line:
            self.nhel_started = True
            this_hel = [int(hel) for hel in line.split('/')[1].split(',')]
            self.all_hel.append(this_hel)
        elif self.nhel_started:
            self.nhel_started = False
            External.good_hel = [ self.all_hel[int(i)-1] for i in self.good_elements ]
            self.counter = 0
            nhel_array = [self.nhel_string(hel)
                          for hel in External.good_hel]
            nhel_lines = '\n'.join(nhel_array)
            self.template_dict['helicity_lines'] += nhel_lines
            self.template_dict['ncomb'] = len(External.good_hel)

    def nhel_string(self, hel_comb):
        self.counter += 1
        formatted_hel = [f'{hel}' if hel < 0 else f' {hel}' for hel in hel_comb]
        nexternal = len(hel_comb)
        return (f'      DATA (NHEL(I,{self.counter}),I=1,{nexternal}) /{",".join(formatted_hel)}/')

    def read_orig(self):

        with open(self.input_file, 'r') as input_file:

            self.prepare_bools()

            for line_num, line in tqdm.tqdm(enumerate(input_file), total=get_num_lines(self.input_file)):

                self.get_old_name(line)
                self.get_good_hel(line)
                self.get_amp_stuff(line_num, line)
                self.get_gwc(line)

                if self.function_call(line) == 'external':
                    self.template_dict['helas_calls'] += self.unfold_external(
                        line)
                if self.function_call(line) == 'internal':
                    self.template_dict['helas_calls'] += self.unfold_internal(
                        line)
                if self.function_call(line) == 'amplitude':
                    self.template_dict['helas_calls'] += self.unfold_amp(line)

        self.template_dict['nwavefuncs'] = Internal.max_wav_num

    def read_template(self):
        out_file = open(self.output_file, 'w+')
        with open(self.template_file, 'r') as file:
            for line in file:
                s = Template(line)
                line = s.safe_substitute(self.template_dict)
                out_file.write(line)
        out_file.close()

    def generate_output_file(self):
        atexit.register(self.clean_up)
        self.read_orig()
        self.read_template()
        atexit.unregister(self.clean_up)

    def clean_up(self):
        pass


def get_arguments(line):
    '''Find the substrings separated by commas between the first
    closed set of parentheses in 'line'. 
    '''
    bracket_depth = 0
    element = 0
    arguments = ['']
    for char in line:
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
        if bracket_depth > 0:
            arguments[element] += char
    return arguments


def apply_args(old_line, all_the_args):
    function = (old_line.split('(')[0]).split()[-1]
    old_args = old_line.split(function)[-1]
    new_lines = [old_line.replace(old_args, f'({",".join(x)})\n')
                 for x in all_the_args]
    return ''.join(new_lines)
 
def get_num(wav):
    name = wav.name
    between_brackets = re.search(r'\(.*?\)', name).group()
    num = int(between_brackets[1:-1].split(',')[-1])    
    return num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The file containing the '
                                          'original matrix calculation')
    parser.add_argument('hel_file', help='The file containing the '
                                         'contributing helicities')
    args = parser.parse_args()

    with open(args.hel_file, 'r') as file:
        good_elements = file.readline().split()

    recycler = HelicityRecycler(good_elements)

    recycler.set_input(args.input_file)
    recycler.set_output('green_matrix.f')
    recycler.set_template('template_matrix1.f')

    recycler.generate_output_file()

if __name__ == '__main__':
    main()

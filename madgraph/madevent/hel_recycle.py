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

class DAG:

    def __init__(self):
        self.graph = {}
        self.all_wavs = []
        self.external_wavs = []
        self.internal_wavs = []

    def store_wav(self, wav, ext_deps=[]):
        self.all_wavs.append(wav)
        nature = wav.nature
        if nature == 'external':
            self.external_wavs.append(wav)
        if nature == 'internal':
            self.internal_wavs.append(wav)
        for ext in ext_deps:
            self.add_branch(wav, ext)

    def add_branch(self, node_i, node_f):
        try:
            self.graph[node_i].append(node_f)
        except KeyError:
            self.graph[node_i] = [node_f]

    def dependencies(self, old_name):
        deps = [wav for wav in self.all_wavs
                if wav.old_name == old_name and not wav.dead]
        return deps

    def kill_old(self, old_name):
        for wav in self.all_wavs:
            if wav.old_name == old_name:
                wav.dead = True

    def old_names(self):
        return {wav.old_name for wav in self.all_wavs}

    def find_path(self, start, end, path=[]):
        '''Taken from https://www.python.org/doc/essays/graphs/'''

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
        return self.__repr__()

    def __repr__(self):
        print_str = 'With new names:\n\t'
        print_str += '\n\t'.join([f'{key} : {item}' for key, item in self.graph.items() ])
        print_str += '\n\nWith old names:\n\t'
        print_str += '\n\t'.join([f'{key.old_name} : {[i.old_name for i in item]}' for key, item in self.graph.items() ])
        return print_str



class MathsObject:
    '''Abstract class for wavefunctions and Amplitudes'''

    # Store here which externals the last wav/amp depends on.
    # This saves us having to call find_path multiple times.
    ext_deps = None

    def __init__(self, arguments, old_name, nature):
        self.args = arguments
        self.old_name = old_name
        self.nature = nature
        self.name = None
        self.dead = False
        self.nb_used = 0
        self.linkdag = []

    def set_name(self, *args):
        self.args[-1] = self.format_name(*args)
        self.name = self.args[-1]

    def format_name(self, *nums):
        pass

    @staticmethod
    def get_deps(line, graph):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
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
        exts = graph.external_wavs
        cls.ext_deps = { i for dep in wavs for i in exts if graph.find_path(dep, i) }
        this_comb_good = False
        for comb in External.good_wav_combs:
            if cls.ext_deps.issubset(set(comb)):
                this_comb_good = True
                break
            
        if diag_number and this_comb_good and cls.ext_deps:

            helicity = dict([(a.get_id(), a.hel) for a in cls.ext_deps])
            this_hel = [helicity[i] for i in range(1, len(helicity)+1)] 
            hel_number = 1 + all_hel.index(tuple(this_hel))
            
            if (hel_number,diag_number) in bad_hel_amp:        
                this_comb_good = False
            

            
        return this_comb_good and cls.ext_deps

    @staticmethod
    def get_new_args(line, wavs):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
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
        old_name = get_arguments(line)[-1].replace(' ','')
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

    def __init__(self, arguments, old_name):
        super().__init__(arguments, old_name, 'external')
        self.hel = int(self.args[2])
        self.mg = int(arguments[0].split(',')[-1][:-1])
        self.hel_ranges = []
        self.raise_num()

    @classmethod
    def raise_num(cls):
        cls.num_externals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        # If graph is passed in Internal it should be done here to so
        # we can set names
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
        graph.kill_old(old_name)

        if 'NHEL' in old_args[2].upper():
            nhel_index = re.search(r'\(.*?\)', old_args[2]).group()
            ext_num = int(nhel_index[1:-1]) - 1
            new_hels = sorted(list(External.hel_ranges[ext_num]), reverse=True)
            new_hels = [int_to_string(i) for i in new_hels]
        else:
            # Spinor must be a scalar so give it hel = 0
            ext_num = int(re.search(r'\(0,(\d+)\)', old_args[0]).group(1)) -1
            new_hels = [' 0']

        new_wavfuncs = []
        for hel in new_hels:

            this_args = copy(old_args)
            this_args[2] = hel

            this_wavfunc = External(this_args, old_name)
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

    @staticmethod
    def format_name(*nums):
        return f'W(1,{nums[0]})'
    
    def get_id(self):
        """ return the id of the particle under consideration """
        
        try:
           return self.id 
        except:
            self.id =  int(re.findall(r'P\(0,(\d+)\)', self.args[0])[0])
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
        return f'AMP({nums[0]},{nums[1]})'

    @classmethod
    def generate_amps(cls, line, graph, all_hel=None, all_bad_hel=[]):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')

        amp_index = re.search(r'\(.*?\)', old_name).group()
        diag_num = int(amp_index[1:-1])

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

    def __init__(self, good_elements, bad_amps=[], bad_amps_perhel=[], gauge='U'):

        External.good_hel = []
        External.nhel_lines = ''
        External.num_externals = 0
        External.wavs_same_leg = {}
        External.good_wav_combs = []

        Internal.max_wav_num = 0
        Internal.num_internals = 0

        Amplitude.max_amp_num = 0
        self.last_category = None
        self.good_elements = good_elements
        self.bad_amps = bad_amps
        self.bad_amps_perhel = bad_amps_perhel

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

    def set_output(self, file):
        self.output_file = file
        if os.path.islink(self.output_file):
            os.remove(self.output_file)

    def set_template(self, file):
        self.template_file = file

    def function_call(self, line):
        # Check a function is called at all
        if not 'CALL' in line:
            return None

        # Now check for external spinor
        ext_calls = ['CALL OXXXXX', 'CALL IXXXXX', 'CALL VXXXXX', 'CALL SXXXXX']
        if any( call in line for call in ext_calls ):
            return 'external'

        # Now check for internal
        # Wont find a internal when no externals have been found...
        # ... I assume
        if not self.dag.external_wavs:
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
        if (function.split('_')[-1] != '0'):
            return 'internal'
        elif (function.split('_')[-1] == '0'):
            return 'amplitude'
        else:
            print(f'Ahhhh what is going on here?\n{line}')
            set_trace()

        return None

    # string manipulation

    def add_amp_index(self, matchobj):
        old_pat = matchobj.group()
        new_pat = old_pat.replace('AMP(', 'AMP( %s,' % self.loop_var)
        
        #new_pat = f'{self.loop_var},{old_pat[:-1]}{old_pat[-1]}'
        return new_pat

    def add_indices(self, line):
        '''Add loop_var index to amp and output variable. 
           Also update name of output variable.'''
        # Doesnt work if the AMP arguments contain brackets
        new_line = re.sub(r'\WAMP\(.*?\)', self.add_amp_index, line)
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
                obj.line = apply_args(line, [obj.args])
        else:
            deps = Amplitude.get_deps(line, self.dag)
            name2dep = dict([(d.name,d) for d in sum(deps,[])])
            
            
        if nature == 'internal':
            new_objs = Internal.generate_wavfuncs(line, self.dag)
            for obj in new_objs:
                obj.line = apply_args(line, [obj.args])
                obj.linkdag = []
                for name in obj.args:
                    if name in name2dep:
                        name2dep[name].nb_used +=1
                        obj.linkdag.append(name2dep[name])
                
        if nature == 'amplitude':
            nb_diag = re.findall(r'AMP\((\d+)\)', line)[0]
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
        if self.amp_splt:
            return split_amps(line, new_objs, gauge=self.gauge)  
        else: 

            return apply_args(line, [i.args for i in new_objs])

    def get_gwc(self, line, category):

        #self.last_category = 
        if category not in ['external', 'internal', 'amplitude']:
            return
        if self.last_category != 'external':
            self.last_category = category
            return

        External.get_gwc()
        self.last_category = category

    def get_good_hel(self, line):
        if 'DATA (NHEL' in line:
            self.nhel_started = True
            this_hel = [int(hel) for hel in line.split('/')[1].split(',')]
            self.all_hel.append(tuple(this_hel))
        elif self.nhel_started:
            self.nhel_started = False
            
            if self.hel_filt:
                External.good_hel = dict([ (self.all_hel[int(i)-1],int(i)) for i in self.good_elements ])
            else:
                External.good_hel = dict([(v,i) for i,v in enumerate(self.all_hel)])

            External.map_hel=dict([(hel,i) for i,hel in  enumerate(External.good_hel)])
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
        formatted_hel = [f'{hel}' if hel < 0 else f' {hel}' for hel in hel_comb]
        nexternal = len(hel_comb)
        return (f'      DATA (NHEL(I,{self.counter}),I=0,{nexternal}) /{old_id},{",".join(formatted_hel)}/')

    def read_orig(self):

        with open(self.input_file, 'r') as input_file:

            self.prepare_bools()

            for line_num, line in tqdm(enumerate(input_file), total=get_num_lines(self.input_file)):
                if line_num == 0:
                    line_cache = line
                    continue
                
                if '!SKIP' in line:
                    continue
                
                char_5 = ''
                try:
                    char_5 = line[5]
                except IndexError:
                    pass
                if char_5 == '$':
                    line_cache = undo_multiline(line_cache, line)
                    continue

                line, line_cache = line_cache, line

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

        
        
        self.template_dict['helas_calls'] = '\n'.join([f'{obj.line.rstrip()} ! count {obj.nb_used}' 
                                 for obj in self.template_dict['helas_calls']
                                 if obj.nb_used > 0 and obj.line])

    def read_template(self):
        out_file = open(self.output_file, 'w+')
        with open(self.template_file, 'r') as file:
            for line in file:
                s = Template(line)
                line = s.safe_substitute(self.template_dict)
                line = '\n'.join([do_multiline(sub_lines) for sub_lines in line.split('\n')])
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
        if bracket_depth > 0 and char != ' ':
            arguments[element] += char
    return arguments


def apply_args(old_line, all_the_args):
    function = (old_line.split('(')[0]).split()[-1]
    old_args = old_line.split(function)[-1]
    new_lines = [old_line.replace(old_args, f'({",".join(x)})\n')
                 for x in all_the_args]
    
    return ''.join(new_lines)

def split_amps(line, new_amps, gauge):
    if not new_amps:
        return ''
    fct = line.split('(',1)[0].split('_0')[0]
    for i,amp in enumerate(new_amps):
        if i == 0:
            occur = []
            for a in amp.args:
                if "W(1," in a:
                    tmp = collections.defaultdict(int)
                    tmp[a] += 1
                    occur.append(tmp)
        else:
            for i in range(len(occur)):
                a = amp.args[i]
                occur[i][a] +=1
    # Each element in occur is the wavs that appear in a column, with
    # the number of occurences
    nb_wav =  [len(o) for o in occur]
    to_remove = nb_wav.index(max(nb_wav)) 
    # Remove the one that occurs the most
    occur.pop(to_remove)
    
    lines = [] 
    # Get the wavs per column
    wav_name = [o.keys() for o in occur]          
    for wfcts in product(*wav_name):
        # Select the amplitudes produced by wfcts
        sub_amps = [amp for amp in new_amps 
                    if all(w in amp.args for w in wfcts)]
        if not sub_amps:
            continue
        if len(sub_amps) ==1:
            lines.append(apply_args(line, [i.args for i in sub_amps]).replace('\n',''))
            
            continue
                         
        # the next line is to make the code nicer 
        sub_amps.sort(key=lambda a: int(a.args[-1][:-1].split(',',1)[1]))
        windices = []
        hel_calculated = []
        iamp = 0
        for i,amp in enumerate(sub_amps):
            args = amp.args[:]   
            # Remove wav and get its index
            wcontract = args.pop(to_remove)
            windex = wcontract.split(',')[1].split(')')[0]
            windices.append(windex)
            amp_result,  args[-1]  =  args[-1], 'TMP(1)'
            
            if i ==0:
                # Call the original fct with P1N_...
                # Final arg is replaced with TMP(1)
                spin = fct.split(None,1)[1][to_remove]
                lines.append('%sP1N_%s(%s)' % (fct, to_remove+1, ', '.join(args)))

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

        lines.append("""      call CombineAmp%(suffix)s(%(nb)i,
     & (/%(hel_list)s/), 
     & (/%(w_list)s/),
     & TMP, W, AMP(1,%(iamp)s))""" % {'suffix':suffix,
                                      'nb': len(sub_amps),
                                      'hel_list': ','.join(hel_calculated),
                                      'w_list': ','.join(windices),
                                      'iamp': iamp
                                     })

            
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
    num_splits = len(line)//char_limit
    if num_splits != 0 and len(line) != 72:
        split_line = [line[i*char_limit:char_limit*(i+1)] for i in range(num_splits+1)]
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

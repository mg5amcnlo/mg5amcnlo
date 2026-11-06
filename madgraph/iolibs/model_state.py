"""Select the numerical model inputs read by exported tree evaluators.

The UFO converter already defines which values UPDATE_AS_PARAM updates. This
module combines that lifecycle information with reads in the generated code;
declarations and unused model parameters do not become cache inputs.
"""

from collections import defaultdict
from pathlib import Path
import re

from madgraph import MadGraph5Error
from madgraph.iolibs.export_v4 import UFO_model_to_mg4


_NAME = re.compile(r'(?<!%)\b[a-zA-Z][a-zA-Z_0-9]*\b')
_STRING = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_START = re.compile(r'^\s*(?:(?:recursive|pure|elemental|impure|double\s+precision|'
                    r'double\s+complex|real(?:\*\d+|\([^)]*\))?|'
                    r'complex(?:\*\d+|\([^)]*\))?|integer(?:\*\d+|\([^)]*\))?|'
                    r'logical(?:\*\d+|\([^)]*\))?|character(?:\*\d+|\([^)]*\))?)\s+)*'
                    r'(subroutine|function)\s+(\w+)\s*(?:\(([^)]*)\))?', re.I)
_DECL = re.compile(r'^\s*(?:implicit|include|common|save|data|parameter|dimension|'
                   r'external|intrinsic|equivalence|allocatable|intent|optional|'
                   r'public|private|use|import|sequence|type\b|end\s*type\b|'
                   r'integer\b|real\b|complex\b|logical\b|character\b|'
                   r'double\s+(?:precision|complex)\b)', re.I)
_INTRINSICS = set(('abs acos acosh adjustl adjustr aimag aint all allocated anint any '
                  'asin asinh associated atan atan2 atanh bessel_j0 bessel_j1 btest '
                  'ceiling char cmplx conjg cos cosh count cshift dabs dacos dasin '
                  'datan datan2 dble dcmplx dconjg dcos dcosh ddim dexp dim dimag '
                  'dint dlog dlog10 dmax1 dmin1 dmod dnint dot_product dprod dsign '
                  'dsin dsinh dsqrt dtan dtanh epsilon exp exponent floor fraction '
                  'huge iabs iand ibclr ibset ichar idim idint idnint ieor index int '
                  'ior ishft kind lbound len len_trim log log10 matmul max max0 max1 '
                  'maxexponent maxloc maxval merge min min0 min1 minexponent minloc '
                  'minval mod modulo nearest nint not pack precision present product '
                  'radix range real repeat reshape rrspacing scale scan selected_int_kind '
                  'selected_real_kind set_exponent shape sign sin sinh size spacing '
                  'spread sqrt storage_size sum tan tanh tiny transfer transpose trim '
                  'ubound unpack verify if elseif while select case where forall '
                  'read write open close inquire format allocate deallocate nullify '
                  # The generated stochastic-helicity path can call the runtime
                  # random generator. It has no numerical model inputs.
                  'ran2').split())
_ROOTS = {
    'born': ('born.f', 'born_hel.f', 'born_onehel.f'),
    'real': ('matrix_*.f',),
    'extra': ('born_cnt_*.f', 'extra_cnt_wrapper.f'),
    'correlation': ('b_sf_*.f', 'sborn_sf.f'),
}
_RUNTIME_NO_MODEL = {'ran2'}  # Reserved generated-runtime RNG contract above.


def _clean(line):
    return _STRING.sub('', line).split('!', 1)[0].strip().lower()


def _has_preprocessor(path):
    free_form = path.suffix.lower() in ('.f90', '.f95')
    return any(line.lstrip().startswith('#') and
               (free_form or len(line) < 6 or line[5] != '#' or line[:5].strip())
               for line in path.read_text().splitlines())


def _paren_end(text, start):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '(':
            depth += 1
        elif text[index] == ')':
            depth -= 1
            if depth == 0:
                return index
    return len(text)-1


def _arguments(text):
    result, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += (char == '(') - (char == ')')
        if char == ',' and depth == 0:
            result.append(text[start:index].strip())
            start = index+1
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


def _argument_base(text):
    """A definable scalar/array/component aliases its base; expressions do not."""
    match = re.match(r'^(\w+)', text.strip())
    if not match:
        return None
    tail = text.strip()[match.end():].strip()
    while tail:
        if tail.startswith('('):
            tail = tail[_paren_end(tail, 0)+1:].strip()
        elif tail.startswith('%'):
            component = re.match(r'%\s*\w+', tail)
            if not component:
                return None
            tail = tail[component.end():].strip()
        else:
            return None
    return match[1]


def _assignment(line):
    statement = re.sub(r'^\d+\s+', '', line)
    # A one-line IF/WHERE has an executable assignment after its condition.
    while re.match(r'^(?:if|where)\s*\(', statement):
        statement = statement[_paren_end(statement, statement.index('('))+1:].strip()
    depth = 0
    for index, char in enumerate(statement):
        depth += (char == '(') - (char == ')')
        if char == '=' and depth == 0 and statement[index-1:index] not in ('<', '>', '/', '=') \
                and statement[index+1:index+2] != '=':
            left = statement[:index].strip()
            return _argument_base(left), line.find(statement)+statement.find(left)
    return None, -1


def _callsites(line, calls):
    result = []
    for match in re.finditer(r'(?<!%)\b(\w+)\s*\(', line):
        if match[1] in calls:
            first = line.index('(', match.start())
            result.append((match[1], _arguments(line[first+1:_paren_end(line, first)])))
    for name in re.findall(r'\bcall\s+(\w+)\s*$', line):
        result.append((name, []))
    return result


def _rename_used_symbols(line, aliases):
    def replace(match):
        # Keyword argument names belong to the callee, not the caller's USE
        # scope. Keep them while translating the actual argument expression.
        depth = line[:match.start()].count('(')-line[:match.start()].count(')')
        if depth > 0 and re.match(r'\s*=(?!=|>)', line[match.end():]):
            return match[0]
        return aliases.get(match[0], match[0])
    return _NAME.sub(replace, line)


def _use_imports(line):
    match = re.match(r'^use\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)(.*)', line)
    if not match:
        return None
    tail = match[2]
    only = re.search(r'\bonly\s*:', tail)
    imports = {}
    for entry in _arguments(tail[only.end():] if only else tail.lstrip(' ,')):
        pair = re.fullmatch(r'(\w+)\s*=>\s*(\w+)', entry)
        if pair:
            imports[pair[1]] = pair[2]
        elif only and re.fullmatch(r'\w+', entry):
            imports[entry] = entry
    return match[1], imports, only is not None


def _semicolon_statements(line):
    """Fortran statement separators do not split quoted diagnostics/comments."""
    quote, start, index = None, 0, 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == quote:
                if line[index+1:index+2] == quote:
                    index += 2
                    continue
                quote = None
        elif char in ('\"', "'"):
            quote = char
        elif char == '!':
            line = line[:index]
            break
        elif char == ';':
            if line[start:index].strip():
                yield line[start:index]
            start = index+1
        index += 1
    if line[start:].strip():
        yield line[start:]


def _statements(path):
    # Import lazily: born_support consumes this module during finalization.
    from madgraph.iolibs.born_support import statements
    if path.suffix.lower() not in ('.f90', '.f95'):
        for line in statements(path.read_text()):
            yield from _semicolon_statements(line)
        return
    pending = ''
    for raw in path.read_text().splitlines():
        line = re.split(r'!(?=(?:[^\"\']|\"[^\"]*\"|\'[^\']*\')*$)', raw, 1)[0].strip()
        if not line or line.startswith('#'):
            continue
        pending += ' ' + line.lstrip('&')
        if pending.endswith('&'):
            pending = pending[:-1]
        else:
            yield from _semicolon_statements(pending.strip())
            pending = ''
    if pending.strip():
        yield from _semicolon_statements(pending.strip())


class _Sources:
    def __init__(self, root, records, names):
        self.root, self.names = Path(root), names
        self.units, self.by_name, self.by_file = [], defaultdict(list), defaultdict(list)
        self.expanded = {}
        self.module_aliases, self.module_imports = defaultdict(dict), defaultdict(list)
        directories = [self.root/'Source/DHELAS', self.root/'Source/MODEL', self.root/'Source']
        directories += [self.root/'SubProcesses'/record['directory'] for record in records]
        for directory in directories:
            for path in sorted(directory.glob('*')):
                if path.suffix.lower() in ('.f', '.for', '.f90', '.f95') and path.is_file():
                    self._index(path)

    def _index(self, path):
        if path in self.by_file:
            return
        self.by_file[path] = []
        current, module = None, None
        for line in _statements(path):
            clean = _clean(line)
            module_start = re.fullmatch(r'module\s+(\w+)', clean)
            if module_start:
                module = module_start[1]
            elif re.fullmatch(r'end\s*module(?:\s+\w+)?', clean):
                module = None
            imported = _use_imports(clean) if current is None and module else None
            if imported:
                origin, aliases, only = imported
                self.module_aliases[module].update((local, (origin, remote))
                                                   for local, remote in aliases.items())
                if not only:
                    self.module_imports[module].append(origin)
            match = _START.match(clean)
            if match:
                current = dict(name=match[2], arguments=tuple(_NAME.findall(match[3] or '')),
                               path=path, lines=[])
                self.units.append(current)
                self.by_name[match[2]].append(current)
                self.by_file[path].append(current)
            elif re.fullmatch(r'end(?:\s*(?:subroutine|function)(?:\s+\w+)?)?', clean):
                current = None
            elif current is not None:
                current['lines'].append(line)
            else:
                include = re.match(r'''\s*include\s*['"]([^'"]+)['"]''', line, re.I)
                if include:
                    target = self._include(path, include[1])
                    if target is not None:
                        self._index(target)

    def _include(self, path, include):
        candidates = [path.parent/include, self.root/'Source/MODEL'/include,
                      self.root/'Source/DHELAS'/include, self.root/'SubProcesses'/include,
                      self.root/'Source'/include]
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def expand(self, path, lines, stack=()):
        if _has_preprocessor(path):
            raise MadGraph5Error('Cannot determine model inputs in %s: preprocessor dependencies '
                                 'require an explicit source contract.' % path.relative_to(self.root))
        result = []
        for line in lines:
            match = re.match(r'''\s*include\s*['"]([^'"]+)['"]''', line, re.I)
            if match:
                target = self._include(path, match[1])
                if target is None or target.resolve() in stack:
                    # An unavailable include cannot be certified from the
                    # executable text; a reachable one is rejected below.
                    result.append(('missing_include', match[1]))
                else:
                    result.extend(self.expand(target, _statements(target), stack+(target.resolve(),)))
            else:
                result.append(('statement', _clean(line)))
        return result

    def resolve(self, caller, name):
        candidates = self.by_name.get(name, [])
        local = [unit for unit in candidates if unit['path'].parent == caller['path'].parent]
        return local or [unit for unit in candidates
                         if unit['path'].parent in (self.root/'Source/DHELAS', self.root/'Source/MODEL',
                                                    self.root/'Source')]

    def module_symbol(self, module, symbol, seen=None):
        seen = set() if seen is None else set(seen)
        if (module, symbol) in seen:
            return symbol
        seen.add((module, symbol))
        if symbol in self.module_aliases[module]:
            origin, original = self.module_aliases[module][symbol]
            return self.module_symbol(origin, original, seen)
        if symbol in self.names or symbol in self.by_name:
            return symbol
        for origin in self.module_imports[module]:
            resolved = self.module_symbol(origin, symbol, seen)
            if resolved != symbol:
                return resolved
        return symbol

    def module_exports(self, module, seen=None):
        seen = set() if seen is None else set(seen)
        if module in seen:
            return {}
        seen.add(module)
        exports = {}
        for origin in self.module_imports[module]:
            exports.update(self.module_exports(origin, seen))
        exports.update((local, self.module_symbol(module, local))
                       for local in self.module_aliases[module])
        return exports

    def scan(self, unit):
        key = (unit['path'], unit['name'])
        if key in self.expanded:
            return self.expanded[key]
        expanded = self.expand(unit['path'], unit['lines'], (unit['path'].resolve(),))
        arrays, visible, executable, missing, writes = set(), set(), [], [], set()
        externals, intrinsic_declarations = set(), set()
        aliases = {}
        for kind, line in expanded:
            if kind == 'missing_include':
                missing.append(line)
                continue
            if re.match(r'^common\b', line):
                visible.update(_NAME.findall(re.sub(r'/[^/]+/', ' ', line)))
            if re.match(r'^use\b', line):
                # Module variables are visible without local declarations.
                # Only actual executable references below select model fields.
                visible.update(self.names)
                imported = _use_imports(line)
                if imported:
                    module, names, only = imported
                    if not only:
                        aliases.update(self.module_exports(module))
                    aliases.update((local, self.module_symbol(module, remote))
                                   for local, remote in names.items())
            if _DECL.match(line):
                arrays.update(re.findall(r'\b(\w+)\s*\(', line))
                if re.match(r'^external\b', line) or ('::' in line and re.search(r'\bexternal\b', line)):
                    declaration = line.split('::', 1)[-1] if '::' in line else line[len('external'):]
                    externals.update(_NAME.findall(declaration))
                if re.match(r'^intrinsic\b', line):
                    intrinsic_declarations.update(_NAME.findall(line[len('intrinsic'):]))
                if re.search(r'\bintent\s*\(\s*(?:out|inout)\s*\)', line):
                    declaration = line.split('::', 1)[-1] if '::' in line else line[line.index(')')+1:]
                    writes.update(_NAME.findall(declaration))
            elif line:
                executable.append(line)
        reads, calls, sites = set(), set(), []
        for kind, line in expanded:
            if kind == 'statement' and re.match(r'^equivalence\b', line) and \
                    set(_NAME.findall(line)) & (visible & self.names | set(unit['arguments'])):
                raise MadGraph5Error('Cannot prove model input aliases in reachable routine %s in %s: '
                                     'EQUIVALENCE requires an explicit dependency contract.' %
                                     (unit['name'], unit['path'].relative_to(self.root)))
            if kind == 'statement' and '=>' in line and not re.match(r'^use\b', line):
                line = _rename_used_symbols(line, aliases)
                for specification in _arguments(line.split('::', 1)[-1]):
                    if '=>' in specification:
                        source = _argument_base(specification.split('=>', 1)[1].strip())
                        if source in visible & self.names | set(unit['arguments']):
                            raise MadGraph5Error('Cannot prove model input aliases in reachable routine %s in %s: '
                                                 'pointer association requires an explicit dependency contract.' %
                                                 (unit['name'], unit['path'].relative_to(self.root)))
        for line in executable:
            line = _rename_used_symbols(line, aliases)
            # Assignment targets are writes. Subscripts on an array target may
            # themselves read a model input, so retain everything but its name.
            target, target_index = _assignment(line)
            read_line = line
            if target:
                writes.add(target)
                read_line = line[:target_index] + ' ' + line[target_index+len(target):]
            if re.match(r'^read\s*\(', line):
                tail = line[_paren_end(line, line.index('('))+1:]
                writes.update(base for base in map(_argument_base, _arguments(tail)) if base)
            reads.update(set(_NAME.findall(read_line)) & visible & self.names - set(unit['arguments']))
            explicit = set(re.findall(r'\bcall\s+(\w+)\s*(?:\(|$)', line))
            expressions = re.sub(r'%\s*\w+\s*\(', '%component(', line)
            referenced = set(re.findall(r'(?<!%)\b(\w+)\s*\(', expressions))
            expressions = referenced - arrays - self.names
            statement_calls = explicit | {name for name in expressions
                if name not in intrinsic_declarations and name not in _RUNTIME_NO_MODEL and
                (name not in _INTRINSICS or name in externals or self.resolve(unit, name))}
            calls.update(statement_calls)
            sites.extend(_callsites(line, statement_calls))
        result = dict(reads=reads, calls=calls, visible=visible & self.names,
                      missing=missing, sites=sites,
                      global_writes=writes & visible & self.names,
                      argument_writes=writes & set(unit['arguments']))
        self.expanded[key] = result
        return result

    def readset(self, roots):
        reads, seen, effects, units = set(), set(), {}, {}
        pending = list(roots)
        while pending:
            unit = pending.pop()
            key = (unit['path'], unit['name'])
            if key in seen:
                continue
            seen.add(key)
            info = self.scan(unit)
            units[key] = unit
            effects[key] = dict(global_writes=set(info['global_writes']),
                                argument_writes=set(info['argument_writes']))
            reads.update(info['reads'])
            unresolved = []
            for name in sorted(info['calls']):
                targets = self.resolve(unit, name)
                if targets:
                    pending.extend(targets)
                else:
                    unresolved.append(name)
            if info['missing'] or unresolved:
                # A missing body can read COMMONs not visible in its caller;
                # widening the caller's declarations would not prove safety.
                raise MadGraph5Error(
                    'Cannot determine model inputs of reachable routine %s in %s: %s. '
                    'Supply the source body/include or an explicit zero-model-input '
                    'runtime dependency contract.' % (
                        unit['name'], unit['path'].relative_to(self.root),
                        ', '.join(['unavailable callee '+name for name in unresolved] +
                                  ['missing include '+name for name in info['missing']])))
        # Propagate output effects through formal arguments to their actual
        # aliases. Iterate to a fixed point so recursive helper cycles cannot
        # hide a write to a model COMMON in their callers.
        changed = True
        while changed:
            changed = False
            for key, unit in units.items():
                info, effect = self.scan(unit), effects[key]
                globals_before = len(effect['global_writes'])
                arguments_before = len(effect['argument_writes'])
                for name, actuals in info['sites']:
                    for target in self.resolve(unit, name):
                        target_effect = effects[(target['path'], target['name'])]
                        effect['global_writes'].update(target_effect['global_writes'])
                        for index, actual in enumerate(actuals):
                            keyword = re.match(r'^(\w+)\s*=(?!=)(.*)', actual)
                            formal = keyword[1] if keyword else (target['arguments'][index]
                                if index < len(target['arguments']) else None)
                            if formal not in target_effect['argument_writes']:
                                continue
                            base = _argument_base(keyword[2].strip() if keyword else actual)
                            if base in unit['arguments']:
                                effect['argument_writes'].add(base)
                            if base in info['visible']:
                                effect['global_writes'].add(base)
                changed |= (globals_before != len(effect['global_writes']) or
                            arguments_before != len(effect['argument_writes']))
        mutations = []
        for key, effect in effects.items():
            if effect['global_writes']:
                mutations.append(dict(source=str(key[0].relative_to(self.root)), routine=key[1],
                                      fields=sorted(effect['global_writes'])))
        return reads, mutations


def _runtime_assignments(path):
    """Assignment dependencies in the per-point branch of model initialization."""
    dependencies, blocks = defaultdict(set), []
    if not path.is_file():
        return dependencies

    def condition_value(condition):
        condition = condition.replace(' ', '').strip('()')
        return False if condition == 'readlha' else True if condition == '.not.readlha' else None

    for statement in _statements(path):
        line = _clean(statement)
        conditional = re.match(r'^(?:if|elseif)\s*\(', line)
        controls = set()
        if conditional:
            first = line.index('(')
            last = _paren_end(line, first)
            condition = condition_value(line[first+1:last])
            controls.update(_NAME.findall(line[first+1:last]))
            if line[last+1:].strip() == 'then':
                parent = any(block['inactive'] for block in (blocks[:-1] if line.startswith('elseif') else blocks))
                block = dict(parent=parent, condition=condition, inactive=parent or condition is False,
                             controls=controls)
                if line.startswith('elseif'):
                    blocks[-1] = block
                else:
                    blocks.append(block)
                continue
            if condition is False:
                continue
        if re.match(r'^end\s*if\b', line):
            if blocks:
                blocks.pop()
            continue
        if line == 'else' and blocks:
            block = blocks[-1]
            block['inactive'] = block['parent'] or block['condition'] is True
            continue
        if any(block['inactive'] for block in blocks):
            continue
        target, _ = _assignment(line)
        if target and '=>' not in line:
            _, target_index = _assignment(line)
            right = line[line.index('=', target_index+len(target))+1:]
            dependencies[target].update(_NAME.findall(right))
            dependencies[target].update(controls)
            for block in blocks:
                dependencies[target].update(block['controls'])
    return dependencies


def _dynamic_names(model, root):
    """Use the same partition as the Fortran model runtime, including RGE."""
    converter = UFO_model_to_mg4(model, '', {'mp': False})
    converter.refactorize()
    dynamic = {variable.name.lower() for variable in converter.params_dep + converter.coups_dep}
    roots = {name.lower() for name in converter.PS_dependent_key}
    running_entries = set()
    for element in model.get('running_elements') or []:
        for variables in element.run_objects:
            for variable in variables:
                if hasattr(variable, 'lhablock') and hasattr(variable, 'lhacode'):
                    running_entries.add((variable.lhablock.lower(), tuple(variable.lhacode)))
    for variable in converter.params_ext:
        entry = (variable.lhablock.lower(), tuple(variable.lhacode))
        if variable.name.lower() in roots or entry in running_entries:
            dynamic.add(variable.name.lower())
    # The runtime takes G, rather than aS, as its input. These aliases are
    # generated in coupl.inc rather than represented by ordinary UFO variables.
    dynamic.update(('g', 'all_g', 'mu_r', 'all_mu_r'))
    known = {variable.name.lower() for variable in converter.params_ext + converter.params_dep +
             converter.params_indep + converter.coups_dep + converter.coups_indep}
    known.update('mdl__%s__scale' % block.lower() for block in converter.scales)
    known.update(('g', 'all_g', 'mu_r', 'all_mu_r', 'gal'))
    # Header aliases are absent from UFO ModelVariable groups. In particular,
    # GAL is assigned outside READLHA and can run when a custom model evolves
    # its EW inputs. Follow generated RHSs without reclassifying fixed UFO
    # variables whose initializer merely happens to mention a dynamic root.
    dependencies = _runtime_assignments(Path(root)/'Source/MODEL/intparam_definition.inc')
    derived_dynamic = set(dynamic)
    changed = True
    while changed:
        changed = False
        for name, inputs in dependencies.items():
            if name in known - dynamic - {'gal'}:
                continue
            if name not in derived_dynamic and inputs & derived_dynamic:
                derived_dynamic.add(name)
                changed = True
    dynamic.update(derived_dynamic & {'gal'})
    aliases = {name: sorted(dependencies[name]) for name in ('gal', 'g', 'all_g', 'mu_r', 'all_mu_r')
               if name in dependencies}
    return dynamic, known, roots, aliases


def _pack(members):
    sizes, packed = {'real': 0, 'complex': 0}, []
    for name, kind, index in members:
        endpoints = [int(value) for value in str(index).split(':')]
        count = endpoints[-1] - endpoints[0] + 1
        first = sizes[kind] + 1
        sizes[kind] += count
        packed.append((name, kind, str(first) if ':' not in str(index)
                       else '%d:%d' % (first, sizes[kind])))
    return packed, sizes


def build_model_state_layout(root, records, model, members):
    """Return compact fixed/dynamic numerical inputs and an auditable readset.

    ``members`` uses born_support.common_members' (name, kind, index) format.
    Read arrays retain their declared extent, since runtime indices need not be
    constant. Unknown but read numerical fields stay dynamic conservatively.
    """
    root, records, members = Path(root), list(records), list(members)
    names = {name.lower() for name, _, _ in members}
    sources = _Sources(root, records, names)
    consumer_readsets, mutations, used = {}, [], set()
    for record in records:
        directory = root/'SubProcesses'/record['directory']
        consumers = {}
        for category, patterns in _ROOTS.items():
            paths = {path for pattern in patterns for path in directory.glob(pattern)}
            roots = [unit for path in sorted(paths) for unit in sources.by_file[path]]
            readset, reasons = sources.readset(roots)
            consumers[category] = sorted(readset)
            used.update(readset)
            mutations.extend(dict(consumer=record['directory'], category=category, **reason)
                             for reason in reasons)
        consumer_readsets[record['directory']] = consumers
    # Positive G is part of the existing certified helicity/amplitude scaling
    # key only when at least one exported amplitude has nonzero strong degree.
    scaling_g = any(matrix.get('homogeneous') and any(matrix.get('amplitude_degrees', []))
                    for record in records for matrix in record.get('helicity_matrices', []))
    if scaling_g and 'g' in names:
        used.add('g')
    dynamic, known, roots, aliases = _dynamic_names(model, root)
    unknown = used - known
    dynamic.update(unknown)
    for mutation in mutations:
        fixed_writes = set(mutation['fields']) & used - dynamic
        if fixed_writes:
            raise MadGraph5Error('Reachable routine %s in %s mutates fixed model input(s) %s. '
                                 'Fixed model configuration must be read-only during evaluation; '
                                 'declare a dynamic runtime input or remove the mutation.' %
                                 (mutation['routine'], mutation['source'], ', '.join(sorted(fixed_writes))))
    dynamic_writes = sorted({name for mutation in mutations for name in mutation['fields']
                             if name in used & dynamic})
    fixed_members, fixed_sizes = _pack([member for member in members
                                        if member[0].lower() in used - dynamic])
    dynamic_members, dynamic_sizes = _pack([member for member in members
                                            if member[0].lower() in used & dynamic])
    sizes = dict(fixed=fixed_sizes, dynamic=dynamic_sizes)
    manifest = dict(version=1, used=sorted(used), unused=sorted(names-used),
                    fixed=[member[0] for member in fixed_members],
                    dynamic=[member[0] for member in dynamic_members],
                    dynamic_roots=sorted(roots), unclassified_dynamic=sorted(unknown),
                    generated_alias_dependencies=aliases,
                    scaling_g=scaling_g, fallbacks=[], mutations=mutations,
                    dynamic_writes=dynamic_writes,
                    consumer_readsets=consumer_readsets, sizes=sizes)
    return dict(fixed_members=fixed_members, dynamic_members=dynamic_members,
                sizes=sizes, consumer_readsets=consumer_readsets, manifest=manifest)

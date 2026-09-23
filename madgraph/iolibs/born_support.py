"""Output-wide Born providers and labelled MC history metadata.

The registry is prepared before forking exporters. Workers only write their own
records; no worker updates a shared manifest. Generated Fortran has private
provider state and passes model state and results explicitly across the DSO API.
"""

import ast
from fractions import Fraction
import inspect
import itertools
import json
import os
from pathlib import Path
import pickle
import re

from madgraph import MadGraph5Error
from madgraph.core import helas_objects
from madgraph.fks import fks_common


def freeze(value):
    if isinstance(value, dict):
        return tuple((k, freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def load_element(element):
    if isinstance(element, str):
        with open(element, 'rb') as stream:
            return pickle.load(stream)
    return element


def evaluator_tag(amplitude):
    """Use HELAS equivalence without merging native process identities.

    The process number controls grouping/event bookkeeping, not the numerical
    HELAS evaluator. Keep it in the context record, but do not compile the
    same implementation again solely because the user chose another number.
    All other HELAS checks, including permutations and symmetry, remain.
    """
    tag = helas_objects.IdentifyMETag.create_tag(amplitude)
    tag[1] = 0
    return tag


def prepare(exporter, elements):
    """Assign stable context/provider IDs using the existing HELAS tags.

    Keep native contexts even when their evaluator is shared. A tag alone is
    insufficient: the labelled external order, splitting orders, normalization
    and extra-Born roles are part of the compatibility contract as well.
    """
    records = []
    for element in elements:
        me = load_element(element)
        born = me.born_me
        proc = born['processes'][0]
        tag = evaluator_tag(born['base_amplitude'])
        signature = (
            tuple(t.get_external_numbers() for t in tag[-1]),
            freeze(proc['split_orders']), freeze(proc['born_sq_orders']),
            freeze(proc['squared_orders']),
            me.ewsudakov,
            tuple(exporter.get_den_factor_lines(me)),
            tuple(exporter.get_ij_lines(me)),
            freeze([i['fks_info'] for i in me.get_fks_info_list()]),
            tuple(evaluator_tag(e['base_amplitude'])
                  for e in me.extra_cnt_me_list))
        records.append(('P' + proc.shell_string(), tag, signature))
    providers = []
    registry = {}
    for context, (name, tag, signature) in enumerate(sorted(records), 1):
        provider = next((i for i, (t, s, _) in enumerate(providers, 1)
                         if tag == t and signature == s), None)
        if provider is None:
            providers.append((tag, signature, name))
            provider = len(providers)
        registry[name] = dict(context=context, provider=provider,
                              source=providers[provider-1][2])
    exporter.born_support_registry = registry


def worker_record(exporter, me, amp_orders):
    """Write immutable native metadata alongside the worker's generated files."""
    name = Path.cwd().name
    entry = dict(exporter.born_support_registry[name])
    born = me.born_me
    proc = born['processes'][0]
    sq, amps = born.get_split_orders_mapping()
    entry.update(directory=name, nexternal=me.get_nexternal_ninitial()[0],
                 nincoming=born.get_nexternal_ninitial()[1],
                 born_pdgs=[[leg['id'] for leg in process.get_legs_with_decays()]
                            for process in born['processes']],
                 nprocesses=len(born['processes']),
                 ngraphs=born.get_number_of_amplitudes(),
                 ncolor=max(1, len(born['color_basis'])),
                 nhelicity=born.get_helicity_combinations(),
                 namps=len(amps), nsqamps=len(sq),
                 orders=proc['split_orders'], amp_orders=amp_orders,
                 born_orders=sq,
                 links=[link['link'] for link in me.color_links],
                 extra=len(me.extra_cnt_me_list),
                 born_identity=born['identical_particle_factor'], sectors=[],
                 helicity_matrices=[helicity_matrix_metadata(matrix, proc['model'])
                     for matrix in [born]+list(me.extra_cnt_me_list)+
                     [real.matrix_element for real in me.real_processes]])
    for sector, info in enumerate(me.get_fks_info_list(), 1):
        real = me.real_processes[info['n_me']-1]
        fks = info['fks_info']
        entry['sectors'].append(dict(
            sector=sector, real=info['n_me'], fks=fks,
            processes=[fks_common.external_process_identity(p, real.particle_tags)
                       for p in real.matrix_element['processes']],
            real_identity=real.matrix_element['identical_particle_factor'],
            allowed=sorted((i, j) for i, js in real.fks_j_from_i.items()
                           for j in js if i > entry['nincoming'] and i != j)))
    Path('born_support.json').write_text(json.dumps(entry, indent=2, sort_keys=True)+'\n')


def labelled_maps(native, outer, nincoming, anchors=None):
    """Flavour/tag preserving label maps, with incoming ordering fixed.

    With anchors, enumerate their orientations and choose the first compatible
    spectator map. Exchanging identical spectators adds no history, so the
    registry must not enumerate their factorially many permutations.
    """
    def restrictions(identity):
        # Coupling names describe the same physical restrictions regardless of
        # the order chosen for the numerical split-order arrays.
        constraints, decays = identity
        return (constraints[:6]+(tuple(sorted(constraints[6])),)+constraints[7:], decays)

    native_key, outer_key = native[0], outer[0]
    if isinstance(native_key, tuple) and len(native_key) == 2:
        native_key, outer_key = restrictions(native_key), restrictions(outer_key)
    if native_key != outer_key or len(native[1]) != len(outer[1]):
        return
    ns, os_ = native[1], outer[1]
    if ns[:nincoming] != os_[:nincoming]:
        return
    choices = [[i+1 for i, s in enumerate(os_) if i >= nincoming and s == leg]
               for leg in ns[nincoming:]]
    if anchors is not None:
        moving = sorted({leg for leg in anchors if leg > nincoming})
        for labels in itertools.product(*(choices[leg-nincoming-1] for leg in moving)):
            if len(set(labels)) != len(labels):
                continue
            perm = dict(zip(moving,labels))
            used = set(labels)
            for leg in range(nincoming+1,len(ns)+1):
                if leg in perm:
                    continue
                label = next((i for i in choices[leg-nincoming-1] if i not in used),None)
                if label is None:
                    break
                perm[leg] = label
                used.add(label)
            else:
                yield tuple(range(1,nincoming+1))+tuple(perm[leg] for leg in range(nincoming+1,len(ns)+1))
        return
    for final in itertools.product(*choices):
        if len(set(final)) == len(final):
            yield tuple(range(1, nincoming+1)) + final


def resolve_histories(records, strict=True):
    """Resolve physical real flavours independently of grouped-list ordering.

    A row maps every outer flavour to its native PDF entry (zero means that
    native group does not contain that flavour). Identical spectator exchanges
    are one history; ordered emitter/emitted orientations remain distinct.
    """
    result = {}
    for outer in records:
        sectors = []
        for sector in outer['sectors']:
            op = freeze(sector['processes'])
            rows = {}
            ownership = {}
            ambiguous = []
            for native in records:
                if native['nexternal'] != outer['nexternal']:
                    continue
                if set(native['orders']) != set(outer['orders']):
                    continue
                order_map = [outer['orders'].index(o)+1 for o in native['orders']]
                amplitude_map = []
                for powers in native['amp_orders']:
                    reordered = [powers[native['orders'].index(o)] for o in outer['orders']]
                    amplitude_map.append(next((i for i, p in enumerate(outer['amp_orders'], 1)
                                               if list(p) == reordered), 0))
                for ns in native['sectors']:
                    np = freeze(ns['processes'])
                    fi = ns['fks']
                    for ofl, physical in enumerate(op, 1):
                        candidates = {}
                        for nfl, native_physical in enumerate(np, 1):
                            for perm in labelled_maps(native_physical, physical, outer['nincoming'],
                                                      anchors=(fi['i'],fi['j'])):
                                pair = (perm[fi['i']-1], perm[fi['j']-1])
                                # The lexicographically first spectator map is
                                # deterministic; swapping spectators adds no pole.
                                candidates.setdefault(pair, (nfl, perm))
                        for pair, (nfl, perm) in candidates.items():
                            key = (ofl,) + pair
                            identity = (native['context'], ns['sector'])
                            if key in ownership and ownership[key] != identity:
                                if strict:
                                    raise MadGraph5Error('Ambiguous global MC history %s: %s and %s' %
                                                         (key, ownership[key], identity))
                                ambiguous.append((key, ownership[key], identity))
                            ownership[key] = identity
                            rowkey = identity + (perm,)
                            row = rows.setdefault(rowkey, dict(
                                provider=native['provider'], context=native['context'],
                                sector=ns['sector'], i=pair[0], j=pair[1],
                                permutation=perm, flavours=[0]*len(op),
                                orders=order_map, amplitudes=amplitude_map,
                                splitting=fi['splitting_type'],
                                extra=fi['extra_cnt_index']+1))
                            row['flavours'][ofl-1] = nfl
            missing = [(ifl, i, j) for ifl in range(1, len(op)+1)
                       for i, j in sector['allowed'] if (ifl, i, j) not in ownership]
            sectors.append(dict(sector=sector['sector'],
                                histories=[rows[k] for k in sorted(rows)],
                                unresolved=missing, ambiguous=ambiguous))
        result[outer['directory']] = sectors
    return result


def statements(source):
    """Read generated fixed-form Fortran without changing string literals."""
    pending = ''
    for line in source.expandtabs(8).splitlines():
        if not line.strip() or line[0] in 'cC*!':
            continue
        body = line[6:] if len(line) > 6 else ''
        if len(line) > 5 and line[5] not in ' 0':
            pending += ' ' + body
        else:
            if pending:
                yield pending
            pending = (line[:5].strip()+' ' if line[:5].strip() else '')+body
    if pending:
        yield pending


def expand(path, stack=()):
    path = Path(path)
    if path.resolve() in stack:
        raise MadGraph5Error('Recursive Born include: %s' % path)
    result = []
    for statement in statements(path.read_text()):
        match = re.match(r'''\s*include\s*['"]([^'"]+)['"]''', statement, re.I)
        if match:
            result.extend(expand(path.parent / match[1], stack+(path.resolve(),)))
        else:
            result.append(statement)
    return result


def include_dependencies(path, seen=None):
    seen = set() if seen is None else seen
    path = Path(path)
    if path in seen:
        return seen
    seen.add(path)
    for line in statements(path.read_text()):
        match = re.match(r'''\s*include\s*['"]([^'"]+)['"]''', line, re.I)
        if match:
            include_dependencies(path.parent/match[1], seen)
    return seen


def rename(source, names):
    # Split off quoted strings before replacing identifiers. This also preserves
    # diagnostics containing routine names and paths.
    parts = re.split(r'''('(?:[^']|'')*'|"(?:[^"]|"")*")''', source)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'(?<!%)\b[a-zA-Z][a-zA-Z_0-9]*\b',
                          lambda m: names.get(m[0].lower(), m[0]), parts[i])
    return ''.join(parts)


def write_fortran(path, lines):
    # Generated statements are already valid Fortran. In particular, passing
    # arbitrary model FORMAT strings through FortranWriter's line splitter can
    # loop indefinitely. These files are built with unlimited fixed line length.
    physical = []
    for line in lines:
        label = re.match(r'^\s*(\d+)\s+(.*)', line)
        physical.append(('%5s %s' % (label[1], label[2])) if label else '      '+line)
    Path(path).write_text('\n'.join(physical)+'\n')


def common_members(lines):
    """Model transfer uses named, typed members, never a raw COMMON byte copy."""
    declarations = {}
    commons = []
    for line in lines:
        m = re.match(r'\s*(double precision|double complex|real\*8|complex\*16)\s+(.+)', line, re.I)
        if m:
            kind = 'complex' if 'complex' in m[1].lower() else 'real'
            for var in re.split(r',\s*(?![^()]*\))', m[2]):
                vm = re.fullmatch(r'\s*(\w+)(?:\((\d+)\))?\s*', var)
                if not vm:
                    raise MadGraph5Error('Unsupported model declaration for Born transfer: '+line)
                declarations[vm[1].lower()] = (kind, int(vm[2] or 1), vm[2] is not None)
        m = re.match(r'\s*common\s*/(\w+)/\s*(.*)', line, re.I)
        if m:
            commons.extend(v.strip().lower() for v in m[2].split(','))
    result = []
    offsets = dict(real=0, complex=0)
    for name in commons:
        if name not in declarations:
            raise MadGraph5Error('Missing typed Born model-state declaration: '+name)
        kind, size, array = declarations[name]
        lo = offsets[kind]+1
        offsets[kind] += size
        index = '%d:%d' % (lo, offsets[kind]) if array else str(lo)
        result.append((name, kind, index))
    return result, offsets


def helicity_coupling_degrees(model):
    """Prove homogeneity in positive G from the actual UFO expressions.

    Declared QCD orders alone are insufficient (legacy HEFT assigns QCD=0
    to its G**2 Higgs coupling). Unknown functions or inhomogeneous sums
    deliberately return None, selecting the full model-state mask key.
    """
    variables = {v.name.lower(): v for group in ('parameters', 'couplings')
                 for values in model[group].values() for v in values}
    externals = model['parameters'].get(('external',), [])
    external = {v.name.lower() for v in externals}
    strong = {v.name.lower() for v in externals
              if str(getattr(v, 'lhablock', '')).lower() == 'sminputs'
              and list(getattr(v, 'lhacode', [])) == [3]}
    cache, visiting = {}, set()

    def variable(name):
        name = name.lower()
        if name in cache:
            return cache[name]
        if name in visiting or name not in variables:
            raise ValueError('Unknown or cyclic model dependency')
        if name in external:
            return Fraction(2 if name in strong else 0)
        visiting.add(name)
        try:
            value = expression(ast.parse(str(variables[name].expr).strip(), mode='eval').body)
            cache[name] = value
            return value
        finally:
            visiting.remove(name)

    def expression(node):
        # Num also recognizes numeric Constant nodes on Python >= 3.8.
        if isinstance(node, ast.Num):
            return Fraction(0)
        if isinstance(node, ast.Name):
            return variable(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == 'cmath' and node.attr == 'pi':
                return Fraction(0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return expression(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = expression(node.left), expression(node.right)
            if isinstance(node.op, ast.Mult):
                return left+right
            if isinstance(node.op, ast.Div):
                return left-right
            if isinstance(node.op, (ast.Add, ast.Sub)) and left == right:
                return left
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Num):
                return left*Fraction(node.right.n)
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and node.func.value.id == 'cmath' else '')
            degrees = [expression(arg) for arg in node.args]
            if name == 'sqrt' and len(degrees) == 1:
                return degrees[0]/2
            if name in ('complexconjugate', 'conjugate', 're', 'im', 'abs') and len(degrees) == 1:
                return degrees[0]
            if name in ('complex', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                        'log', 'exp', 'sec', 'csc', 'cot') and not any(degrees):
                return Fraction(0)
        raise ValueError('Nonhomogeneous or unsupported model expression')

    result = {}
    for values in model['couplings'].values():
        for coupling in values:
            try:
                degree = variable(coupling.name)
                result[coupling.name.lower()] = int(degree) if degree.denominator == 1 else None
            except (ValueError, SyntaxError, AttributeError, TypeError, ZeroDivisionError):
                result[coupling.name.lower()] = None
    try:
        if variable('g') != 1:
            return dict.fromkeys(result)
    except (ValueError, SyntaxError, AttributeError, TypeError):
        return dict.fromkeys(result)
    return result


def helicity_matrix_metadata(matrix, model):
    """Certify uniform G scaling in each actual squared-order component."""
    degrees = helicity_coupling_degrees(model)
    used = {name.lower(): degrees.get(name.lower())
            for name in matrix.get_used_couplings(output='set')}
    result = dict(couplings=used, homogeneous=False)
    # Form factors can read model state hidden from the matrix CALL arguments.
    used_lorentz = {name for names, _, _ in matrix.get_used_lorentz() for name in names}
    if any(getattr(lorentz, 'formfactors', None) for lorentz in model['lorentz']
           if lorentz.name in used_lorentz) or any(v is None for v in used.values()):
        return result
    # P0 is the built-in massless/Feynman-gauge propagator.
    if any(wavefunction['particle'].get('propagator') not in ('', None, 0)
           for wavefunction in getattr(matrix, 'get_all_wavefunctions', lambda: [])()):
        return result
    cache = {}

    def node_degree(node):
        if id(node) not in cache:
            local = {degrees.get(c.lstrip('-').lower()) for c in node['coupling']}
            if not node['mothers']:
                value = 0
            elif node['interaction_id'] in (0, -1):
                value = sum(node_degree(mother) for mother in node['mothers'])
            else:
                if len(local) != 1 or None in local:
                    raise ValueError('Mixed powers in a HELAS vertex')
                value = local.pop()+sum(node_degree(mother) for mother in node['mothers'])
            cache[id(node)] = value
        return cache[id(node)]

    try:
        amplitudes = {amp['number']: node_degree(amp) for amp in matrix.get_all_amplitudes()}
        _, groups = matrix.get_split_orders_mapping()
        if not groups:
            groups = [((), tuple(amplitudes))]
        orders = {}
        for order, indices in groups:
            powers = {amplitudes[i] for i in indices}
            if len(powers) != 1:
                return result
            orders[order] = powers.pop()
        squared = {}
        for a, da in orders.items():
            for b, db in orders.items():
                key = tuple(x+y for x, y in zip(a, b))
                if key in squared and squared[key] != da+db:
                    return result
                squared[key] = da+db
    except (ValueError, KeyError):
        return result
    result['homogeneous'] = True
    return result


def helicity_state_signature(root, records, model, members):
    """Use only certified tree inputs for masks; amplitude keys stay complete."""
    matrices = [matrix for record in records for matrix in record.get('helicity_matrices', [])]
    if not matrices or not all(matrix['homogeneous'] for matrix in matrices):
        return None
    # Ordinary ALOHA routines depend only on their arguments. Custom HELAS
    # routines with hidden model COMMONs need the conservative full-state key.
    for path in (root/'Source'/'DHELAS').glob('*.f'):
        if any(re.search(r'(?i)\bcommon\b', line) for line in expand(path)):
            return None
    direct = set()
    for record in records:
        source = root/'SubProcesses'/record['directory']
        for pattern in ('born.f', 'born_hel.f', 'born_cnt_*.f', 'matrix_*.f'):
            for path in source.glob(pattern):
                direct.update(re.findall(r'\b[a-zA-Z]\w*\b', '\n'.join(statements(path.read_text())).lower()))
    independent = {v.name.lower() for v in model['parameters'].get(('external',), [])
                   if not ((str(getattr(v, 'lhablock', '')).lower(),
                            tuple(getattr(v, 'lhacode', []))) in
                           [('sminputs', (3,)), ('loop', (1,))])}
    used = {name: degree for matrix in matrices for name, degree in matrix['couplings'].items()}
    by_name = {name: (kind, index) for name, kind, index in members}
    if 'g' not in by_name or by_name['g'][0] != 'real' or ':' in by_name['g'][1]:
        return None
    signature = dict(g=int(by_name['g'][1]), real=[], complex=[], couplings=[])
    for name in sorted(used):
        if name not in by_name or ':' in by_name[name][1]:
            return None
        kind, index = by_name[name]
        signature['couplings'].append((kind, int(index), used[name]))
    for name, kind, index in members:
        if name in used or name not in direct | independent:
            continue
        # A direct G/mu_R read cannot be discarded just because it normally
        # comes from running couplings; keep it exact as any other input.
        indices = list(map(int, index.split(':')))
        signature[kind].extend(range(indices[0], indices[-1]+1))
    return signature


def write_helicity_state_comparator(path, members, signature):
    """Install an output-specific mask predicate, using exact comparisons.

    Positive nonzero G and identical normalized couplings imply a nonzero
    common factor per certified squared order. Division roundoff can cause
    harmless relearning; no tolerance can hide a change in a cancellation.
    """
    if signature is None:
        return
    sizes = {kind: max([int(index.split(':')[-1]) for _, k, index in members if k == kind] or [0])
             for kind in ('real', 'complex')}
    lines = ['  logical function born_helicity_state_equal(a,b)',
             '    use,intrinsic::ieee_arithmetic,only:ieee_is_finite',
             '    type(BornModelState),intent(in)::a,b',
             '    real(8)::ga,gb,pa,pb', '    complex(8)::ca,cb',
             '    born_helicity_state_equal=.false.',
             '    if(.not.allocated(a%real_values).or..not.allocated(b%real_values))return',
             '    if(.not.allocated(a%complex_values).or..not.allocated(b%complex_values))return',
             '    if(size(a%real_values).ne.size(b%real_values))return',
             '    if(size(a%complex_values).ne.size(b%complex_values))return']
    if members:
        lines += ['    if(size(a%%%s_values).ne.%d)return' % (kind, sizes[kind])
                  for kind in ('real', 'complex')]
    lines += [
             '    if(born_model_state_equal(a,b))then',
             '      born_helicity_state_equal=.true.', '      return', '    endif',
             '    ga=a%%real_values(%d)' % signature['g'],
             '    gb=b%%real_values(%d)' % signature['g'],
             '    if(.not.ieee_is_finite(ga).or..not.ieee_is_finite(gb))return',
             '    if(ga.le.0d0.or.gb.le.0d0)return']
    for kind in ('real', 'complex'):
        for index in sorted(set(signature[kind])):
            lines.append('    if(a%%%s_values(%d).ne.b%%%s_values(%d))return' % (kind,index,kind,index))
    for kind, index, degree in sorted(signature['couplings']):
        left, right = ('%s%%%s_values(%d)' % (state,kind,index) for state in ('a','b'))
        lines += ['    pa=ga**(%d)' % degree, '    pb=gb**(%d)' % degree,
                  '    if(.not.ieee_is_finite(pa).or..not.ieee_is_finite(pb))return',
                  '    if(pa.eq.0d0.or.pb.eq.0d0)return',
                  '    ca=%s/pa' % left, '    cb=%s/pb' % right,
                  '    if(.not.ieee_is_finite(real(ca,8)).or..not.ieee_is_finite(aimag(ca)))return',
                  '    if(.not.ieee_is_finite(real(cb,8)).or..not.ieee_is_finite(aimag(cb)))return',
                  '    if(real(ca,8).eq.0d0.and.real(%s,8).ne.0d0)return' % left,
                  '    if(real(cb,8).eq.0d0.and.real(%s,8).ne.0d0)return' % right]
        if kind == 'complex':
            lines += ['    if(aimag(ca).eq.0d0.and.aimag(%s).ne.0d0)return' % left,
                      '    if(aimag(cb).eq.0d0.and.aimag(%s).ne.0d0)return' % right]
        lines.append('    if(ca.ne.cb)return')
    lines += ['    born_helicity_state_equal=.true.', '  end function']
    source = path.read_text()
    source = re.sub(r'  logical function born_helicity_state_equal\(a,b\)[\s\S]*?  end function',
                    '\n'.join(lines), source)
    path.write_text(source)


def routine(source, name):
    lines = list(statements(source))
    start = next(i for i, line in enumerate(lines)
                 if re.match(r'\s*(?:[\w*]+\s+)*?(?:subroutine|function)\s+'+name+r'\b', line, re.I))
    end = next(i for i in range(start+1, len(lines)) if lines[i].strip().lower() == 'end')
    return lines[start:end+1]


def without_routine(lines, name):
    result, skipping = [], False
    for line in lines:
        if re.match(r'\s*subroutine\s+'+name+r'\b', line, re.I):
            skipping = True
        if not skipping:
            result.append(line)
        elif line.strip().lower() == 'end':
            skipping = False
    return result


def model_aware_helicity_filter(lines):
    """Relearn each routine's helicity mask when its provider model changes.

    NTRY is per sector for the main Born and scalar for extra counterterms.
    Keeping the epoch local to each routine also handles a counterterm first
    requested several calls after the main Born discovered the new model.
    """
    result, routine_lines = [], []
    for line in lines:
        routine_lines.append(line)
        if line.strip().lower() != 'end':
            continue
        if any(re.match(r'(?i)\s*ntry(?:\([^)]*\))?\s*=\s*ntry', s)
               for s in routine_lines):
            scalar = any(re.match(r'(?i)\s*ntry\s*=\s*ntry', s)
                         for s in routine_lines)
            for statement in routine_lines:
                # Learn individual order components: sums can cancel between
                # different G powers, and extra Borns need every order too.
                if 'goodhel' in statement.lower():
                    statement = re.sub(r'(?i)borns\(([12]),[01]\)\s*\.ne\.\s*0d0',
                        r'maxval(abs(borns(\1,1:nsqampso))).ne.0d0', statement)
                result.append(statement)
                if re.match(r'(?i)\s*implicit\s+none\s*$', statement):
                    result += ['integer born_model_epoch,helicity_model_epoch',
                               'common/c_born_model_epoch/born_model_epoch',
                               'save helicity_model_epoch',
                               'data helicity_model_epoch/-1/']
                    if scalar:
                        result += ['integer,save::helicity_sector=0']
                if re.match(r'(?i)\s*ntry(?:\([^)]*\))?\s*=\s*ntry', statement):
                    result.pop()
                    changed = 'helicity_model_epoch.ne.born_model_epoch'
                    if scalar:
                        changed += '.or.helicity_sector.ne.nfksprocess'
                    result += ['if(%s)then' % changed,
                               'ntry=0', 'goodhel=.false.',
                               'helicity_model_epoch=born_model_epoch']
                    if scalar:
                        result += ['helicity_sector=nfksprocess']
                    result += ['endif', statement]
        else:
            result += routine_lines
        routine_lines = []
    return result + routine_lines


def optimize_real_helicities(lines):
    """Keep zero-helicity filtering, without empirical equal-helicity reuse.

    Equal squared helicity amplitudes at one momentum are not necessarily
    equal after a boost or at another point. In particular the real template's
    T_IDENT shortcut can change a massive-vector helicity sum under a boost.
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'(?i)\s*if\s*\(\s*ntry\s*\.lt\.\s*2\s*\)\s*then\s*$', line):
            # Replace the entire discovery/reuse conditional by its matrix
            # call. The enclosing GOODHEL predicate and sum remain unchanged.
            depth, end, matrix_call = 1, i+1, None
            while depth and end < len(lines):
                item = lines[end]
                if re.match(r'(?i)\s*call\s+matrix_\w+\s*\(', item):
                    matrix_call = item
                if re.match(r'(?i)\s*if\s*\(.*\)\s*then\s*$', item):
                    depth += 1
                elif re.match(r'(?i)\s*end\s*if\s*$', item):
                    depth -= 1
                end += 1
            if depth or matrix_call is None:
                raise MadGraph5Error('Unrecognized real helicity reuse block')
            result.append(matrix_call)
            i = end
            continue
        if re.match(r'(?i)\s*save\s+t_save\s*,\s*t_ident\s*$', line):
            i += 1
            continue
        line = re.sub(r'(?i),\s*t_ident\(ncomb\)', '', line)
        line = re.sub(r'(?i),\s*t_save\(ncomb,0:nsqampso\)', '', line)
        if re.match(r'(?i)\s*implicit\s+none\s*$', line):
            result += ['use mc_born_types,only:BornModelState,born_helicity_state_equal', line,
                       'type(BornModelState),save::helicity_model_state,helicity_current_state']
        elif re.match(r'(?i)\s*ntry\s*=\s*ntry\s*\+\s*1\s*$', line):
            result += ['call mc_capture_model_state(helicity_current_state)',
                       'if(.not.born_helicity_state_equal(helicity_model_state,helicity_current_state))then',
                       'ntry=0', 'goodhel=.false.',
                       'helicity_model_state%real_values=helicity_current_state%real_values',
                       'helicity_model_state%complex_values=helicity_current_state%complex_values',
                       'endif', line]
        else:
            if 'goodhel' in line.lower():
                line = re.sub(r'(?i)t\(0\)\s*\.ne\.\s*0d0',
                              'maxval(abs(t(1:nsqampso))).ne.0d0', line)
            result.append(line)
        i += 1
    return result


def write_real_helicity_filter(path):
    """Only rewrite the summed routine; retain the generated HELAS source."""
    from madgraph.iolibs.native_histories import write
    source = path.read_text()
    temporary = path.with_suffix('.helicity.tmp')
    def replace(match):
        write(temporary,optimize_real_helicities(list(statements(match[0]))))
        return temporary.read_text().rstrip('\n')
    try:
        source = re.sub(r'^      SUBROUTINE\s+SMATRIX\w+_SPLITORDERS\b[\s\S]*?^      END[ \t]*$',
                        replace,source,flags=re.I|re.M)
        path.write_text(source)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize(exporter):
    """Merge worker records, generate the DSO and replace legacy entry points."""
    root = Path(exporter.dir_path).resolve()
    records = [json.loads(p.read_text()) for p in sorted(
        (root/'SubProcesses').glob('P*/born_support.json'))]
    if not records:
        return
    support = root/'Source'/'BornSupport'
    support.mkdir(exist_ok=True)
    # A fixed-order output may intentionally omit a Born family required for
    # matching, or use splitting modes unsupported by MC@NLO. Preserve export,
    # but make the strict runtime query reject both missing and ambiguous sums.
    histories = resolve_histories(records, strict=False)
    (support/'registry.json').write_text(json.dumps(dict(version=1, contexts=records,
        histories=histories), indent=2, sort_keys=True)+'\n')
    model_lines = expand(root/'Source'/'MODEL'/'coupl.inc')
    model_lines += expand(root/'Source'/'MODEL'/'input.inc')
    model_members, model_sizes = common_members(model_lines)
    model_commons = {m.lower(): 'mcb_model_'+m.lower() for line in model_lines
                     for m in re.findall(r'(?i)common\s*/(\w+)/', line)}
    providers = {}
    for record in records:
        if record['directory'] == record['source']:
            providers[record['provider']] = record
    for provider, record in sorted(providers.items()):
        write_provider(root, support, record, model_members, model_commons)
    for record in records:
        write_metadata(root, support, record)
    write_api(support, records, providers, model_sizes, histories)
    signature = helicity_state_signature(root, records, exporter.model, model_members)
    write_helicity_state_comparator(support/'mc_born_types.f90', model_members, signature)
    (support/'helicity_signature.json').write_text(json.dumps(
        dict(mode='homogeneous_G' if signature else 'full_model_state', signature=signature),
        indent=2, sort_keys=True)+'\n')
    write_build(root, support, providers, model_commons, records)
    for record in records:
        write_wrappers(root, record, model_members)
    from madgraph.iolibs import native_histories
    native_histories.export(root, records, histories)
    for record in records:
        for source in (root/'SubProcesses'/record['directory']).glob('matrix_*.f'):
            write_real_helicity_filter(source)


def write_provider(root, support, record, model_members, model_commons):
    """Private provider implementation; no caller-sized Born COMMON is shared."""
    source = root/'SubProcesses'/record['directory']
    pid = record['provider']
    directory = support/('p%d' % pid)
    directory.mkdir(exist_ok=True)
    write_fortran(directory/'helicity_sampler.f',
                  routine((source/'born_hel.f').read_text(), 'pickhelicitymc'))
    # EW interference routines query these immutable Born order tables in the
    # executable. They contain no amplitudes, mutable state or model functions.
    born_source = (source/'born.f').read_text()
    write_fortran(directory/'order_queries.f', sum((routine(born_source, name)
        for name in ('sqsoindexb', 'sqsoindexb_from_orders',
                     'getordpowfromindex_b', 'get_nsqso_b')), []))
    paths = [source/n for n in ('born.f', 'born_hel.f', 'sborn_sf.f', 'extra_cnt_wrapper.f')]
    paths += sorted(source.glob('b_sf_*.f')) + sorted(source.glob('born_cnt_*.f'))
    lines = []
    for path in paths:
        lines += expand(path)
    lines = without_routine(lines, 'pickhelicitymc')
    # Pure order helpers used by the Born and EW-Sudakov Born-order buffers.
    # The remaining event bookkeeping belongs to the executable.
    helper = (source/'splitorders_stuff.f').read_text()
    ew_helper = next(path for path in (source/'ewsudakov_functions.f',
                                      source/'ewsudakov_functions_dummy.f')
                     if path.exists())
    helper += '\n' + ew_helper.read_text()
    for name in ('orders_to_amp_split_pos', 'amp_split_pos_to_orders',
                 'orders_equal', 'get_lo2_orders'):
        temp = directory/'helper.f'
        write_fortran(temp, routine(helper, name))
        for include in ('orders.inc', 'amp_split_orders.inc'):
            (directory/include).write_text((source/include).read_text())
        lines += expand(temp)
        temp.unlink()
    lines = model_aware_helicity_filter(lines)
    names = dict(model_commons)
    for line in lines:
        names.update((n.lower(), 'mcb%d_' % pid+n.lower()) for n in
                     re.findall(r'(?i)(?:subroutine|function|block\s+data)\s+(\w+)', line))
        for n in re.findall(r'(?i)common\s*/(\w+)/', line):
            names.setdefault(n.lower(), 'mcb%d_' % pid+n.lower())
    write_fortran(directory/'matrix.f', [rename(line, names) for line in lines])
    adapter = provider_adapter(record, source, model_members)
    write_fortran(directory/'adapter.f', [rename(line, names) for line in adapter])


def write_metadata(root, support, record):
    # Flavours and mapping data belong to the native context even when its
    # numerical HELAS evaluator is equivalent to another context's provider.
    source = root/'SubProcesses'/record['directory']
    directory = support/('c%d' % record['context'])
    directory.mkdir(exist_ok=True)
    meta = ['subroutine mcb_metadata_%d(metadata)' % record['context'],
            'use mc_born_types', 'implicit none', 'type(BornMetadata) metadata',
            'integer i,j']
    meta += expand(source/'nexternal.inc') + expand(source/'genps.inc')
    meta += expand(source/'born_nhel.inc')
    meta += ['integer idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc)',
             'integer icolup(2,nexternal-1,max_bcol)']
    meta += expand(source/'born_leshouche.inc')
    topology = expand(source/'born_conf.inc')
    # The legacy include is sparse: e.g. an s-channel leaves its t-channel
    # entry unset. Returning those uninitialized cells would make an immutable
    # metadata query depend on stack contents or compiler initialization flags.
    nconfigs = int(next(re.search(r'(?i)lmaxconfigsb_used\s*=\s*(\d+)', line)[1]
                        for line in topology if re.search(r'(?i)lmaxconfigsb_used\s*=', line)))
    nbranches = int(next(re.search(r'(?i)max_branchb_used\s*=\s*(\d+)', line)[1]
                        for line in topology if re.search(r'(?i)max_branchb_used\s*=', line)))
    for name, default in (('sprop','0'),('tprid','0'),('gforcebw','.false.'),('iforest','0,0')):
        indices = r'\s*IFR\s*,' if name == 'iforest' else ''
        pattern = re.compile(r'(?i)\b'+name+r'\('+indices+r'\s*(-?\d+)\s*,\s*(\d+)\s*\)')
        present = {tuple(map(int, match.groups())) for line in topology
                   for match in pattern.finditer(line) if line.lstrip().lower().startswith('data')}
        for branch in range(-nbranches,0):
            for config in range(1,nconfigs+1):
                if (branch,config) not in present:
                    target = ('(iforest(ifr,%d,%d),ifr=1,2)' if name == 'iforest' else
                              name+'(%d,%d)') % (branch,config)
                    topology.append('data %s /%s/' % (target,default))
    meta += topology + expand(source/'born_coloramps.inc')
    meta += ['metadata%born_ids=idup(:,1:iproc_born)',
             'metadata%born_mothers=mothup(:,:,1:iproc_born)',
             'allocate(metadata%configurations(0:lmaxconfigsb_used))',
             'allocate(metadata%tree(2,-max_branchb_used:-1,lmaxconfigsb_used))',
             'allocate(metadata%sprop(-max_branchb_used:-1,lmaxconfigsb_used))',
             'allocate(metadata%tprid(-max_branchb_used:-1,lmaxconfigsb_used))',
             'allocate(metadata%force_bw(-max_branchb_used:-1,lmaxconfigsb_used))',
             'metadata%born_colours=icolup', 'metadata%configurations=mapconfig',
             'metadata%tree=iforest', 'metadata%sprop=sprop', 'metadata%tprid=tprid',
             'metadata%force_bw=gforcebw',
             'metadata%colour_amplitudes=icolamp', 'end']
    write_fortran(directory/'metadata.f', meta)


def provider_adapter(r, source, members):
    n = r['nexternal']-1
    ng, nc, nh = r['ngraphs'], r['ncolor'], r['nhelicity']
    na, ns = r['namps'], r['nsqamps']
    lines = ['subroutine mcb_provider_%d(context,p,state,request,result,status)' % r['provider'],
             'use mc_born_types', 'implicit none',
             'integer context,status', 'type(BornModelState) state',
             'type(BornRequest) request', 'type(BornResult) result',
             'double precision p(0:3,%d),ans,correlation' % n,
             'integer nfksprocess,i', 'common/c_nfksprocess/nfksprocess',
             'logical calculatedBorn', 'common/ccalculatedBorn/calculatedBorn',
             'type(BornModelState),save::cached_state',
             'double precision,save::cached_p(0:3,%d)=0d0' % n,
             'integer,save::cached_context=0,cached_sector=0,model_epoch=0',
             'integer born_model_epoch', 'common/c_born_model_epoch/born_model_epoch',
             'logical model_changed,cache_hit',
             'double precision amp2(%d),jamp2(0:%d)' % (ng, nc),
             'common/to_amps/amp2,jamp2',
             'double precision amp2b(%d),jamp2b(0:%d,0:%d)' % (ng,nc,na),
             'common/to_amps_born/amp2b,jamp2b',
             'double precision wgt_hel(%d),wgt_hel_split(%d,%d)' % (nh,ns,nh),
             'common/c_born_hel/wgt_hel',
             'common/c_born_hel_split/wgt_hel_split',
             'double complex ans_split(2,0:%d)' % ns]
    lines += expand(source/'orders.inc') + expand(source/'coupl.inc')
    lines += expand(source.parents[1]/'Source'/'MODEL'/'input.inc')
    lines += ['double complex ans_cnt(2,nsplitorders),extra(2,nsplitorders)',
              'common/c_born_cnt/ans_cnt',
              'double precision amp_split_soft(amp_split_size)',
              'common/to_amp_split_soft/amp_split_soft',
              'logical split_type_used(nsplitorders)',
              'common/to_split_type_used/split_type_used',
              'logical need_color_links,need_charge_links',
              'common/c_need_links/need_color_links,need_charge_links',
              'double precision charges(%d)' % n,
              'common/c_charges_born/charges',
              'double complex amp_split_ewsud(amp_split_size)',
              'common/to_amp_split_ewsud/amp_split_ewsud',
              'double complex amp_split_ewsud_lo2(amp_split_size)',
              'common/to_amp_split_ewsud_lo2/amp_split_ewsud_lo2']
    lines += ['status=0', 'nfksprocess=request%sector',
              'if(nfksprocess.lt.1.or.nfksprocess.gt.%d) then' % max(1,len(r['sectors'])),
              'status=2', 'return', 'endif']
    if not r['sectors']:
        # LO-only exports lack the NLO order slots and FKS tables needed by
        # the legacy correlated routines. Reject that unsupported request at
        # the API boundary instead of indexing an empty GOODHEL array or
        # stopping inside the generated order lookup.
        lines += ['if(request%colour.or.request%charge)then',
                  'status=4', 'return', 'endif']
    if r['links']:
        checks = ['((request%%m.eq.%d.and.request%%n.eq.%d).or.(request%%m.eq.%d.and.request%%n.eq.%d))' %
                  (m,n,n,m) for m,n in r['links']]
        lines += ['if(request%colour.and..not.(' + '.or.'.join(checks)+'))then',
                  'status=4','return','endif']
    else:
        lines += ['if(request%colour)then','status=4','return','endif']
    lines += ['model_changed=.not.born_model_state_equal(state,cached_state)',
              'cache_hit=.false.',
              'if(.not.model_changed)cache_hit=context.eq.cached_context.and.'
              'request%sector.eq.cached_sector.and.all(p.eq.cached_p)',
              'if(model_changed)then',
              'if(.not.born_helicity_state_equal(state,cached_state))model_epoch=model_epoch+1',
              'cached_state%real_values=state%real_values',
              'cached_state%complex_values=state%complex_values',
              'endif',
              'born_model_epoch=model_epoch']
    # The model COMMONs are shared by providers inside the DSO. Transfer even
    # on a cache hit: an intervening provider may have installed another state.
    lines += ['%s=state%%%s_values(%s)' % v for v in members]
    flags = [any(o in s['fks']['splitting_type'] for s in r['sectors']) for o in r['orders']]
    lines += ['split_type_used(%d)=%s' % (i, '.true.' if v else '.false.')
              for i,v in enumerate(flags,1)]
    lines += ['need_color_links=request%colour', 'need_charge_links=request%charge',
              'charges=request%charges',
              # Rebuild the inexpensive contractions/order/flow buffers from
              # SAVEAMP on a hit. Never trust a caller's calculatedBorn flag.
              'calculatedBorn=cache_hit', 'ans_cnt=(0d0,0d0)',
              'amp_split_cnt=(0d0,0d0)', 'call sborn(p,ans)',
              'cached_p=p', 'cached_context=context', 'cached_sector=request%sector',
              'result%born=ans', 'result%amplitudes=amp_split',
              'result%counterterms=ans_cnt', 'result%split_counterterms=amp_split_cnt',
              'result%diagrams=amp2', 'result%%flows=jamp2(1:%d)' % nc,
              'result%%flow_orders=jamp2b(1:%d,1:%d)' % (nc,na),
              'if(request%helicities) then',
              'result%has_helicities=.true.',
              'wgt_hel=0d0', 'wgt_hel_split=0d0', 'call sborn_hel(p,ans)',
              'result%helicities=wgt_hel', 'result%helicity_orders=wgt_hel_split', 'endif',
              'if(request%colour.or.request%charge) then',
              'result%has_soft=.true.',
              'call sborn_sf(p,request%m,request%n,correlation)',
              'result%correlation=correlation', 'result%soft=amp_split_soft',
              'result%split_counterterms=amp_split_cnt', 'endif',
              'if(request%extra.gt.0) then',
              'result%has_extra=.true.',
              'call extra_cnt(p,request%extra,extra)',
              'result%extra=extra', 'result%split_counterterms=amp_split_cnt', 'endif',
              # SBORN above initializes the provider's order selection. Keep
              # the one-helicity EW buffers separate from the summed result.
              'if(request%single_helicity.gt.0)then',
              'result%has_single_helicity=.true.',
              'call sborn_onehel(p,request%helicity,request%single_helicity,ans)',
              'result%single_helicity=ans',
              'result%ewsudakov=amp_split_ewsud',
              'result%ewsudakov_lo2=amp_split_ewsud_lo2', 'endif',
              'end']
    return lines


def write_api(support, records, providers, sizes, histories):
    template = Path(__file__).resolve().parents[2]/'Template'/'NLO'/'Source'/'BornSupport'
    (support/'mc_born_types.f90').write_text((template/'mc_born_types.f90').read_text())
    lines = ['module mc_born_support', 'use mc_born_types', 'implicit none',
             'private', 'public :: born_query, born_evaluate, born_model_dimensions',
             'contains', 'subroutine born_model_dimensions(nreal,ncomplex)',
             'integer,intent(out)::nreal,ncomplex',
             'nreal=%d' % sizes['real'], 'ncomplex=%d' % sizes['complex'],
             'end subroutine',
             'subroutine born_query(provider,context,metadata,status,details,require_complete)',
             'integer,intent(in)::provider,context',
             'type(BornMetadata),intent(out)::metadata',
             'integer,intent(out)::status',
             'logical,optional,intent(in)::details,require_complete',
             'status=1', 'select case(context)']
    for r in records:
        lines += ['case(%d)' % r['context'],
                  'if(provider.ne.%d)return' % r['provider'],
                  'metadata%provider=provider', 'metadata%context=context']
        for key in ('nexternal','nincoming','nprocesses','ngraphs','ncolor','nhelicity','namps','nsqamps','extra'):
            lines += ['metadata%%%s=%d' % (key,r[key])]
        lines += ['metadata%%nsectors=%d' % max(1,len(r['sectors'])),
                  'metadata%%supports_correlations=%s' % ('.true.' if r['sectors'] else '.false.'),
                  'metadata%%nsplitorders=%d' % len(r['orders']),
                  'metadata%%namplitudes=%d' % len(r['amp_orders']),
                  'status=0',
                  'if(present(details))then','if(.not.details)then',
                  'if(.not.present(require_complete))return',
                  'if(.not.require_complete)return','endif','endif',
                  'allocate(metadata%fks(4,metadata%nsectors))', 'metadata%fks=0']
        lines += ['call mcb_metadata_%d(metadata)' % r['context'],
                  'metadata%%identical_factor=%sd0' % r['born_identity'],
                  'allocate(metadata%order_names(metadata%nsplitorders))',
                  'allocate(metadata%amplitude_orders(metadata%nsplitorders,metadata%namplitudes))']
        for i, order in enumerate(r['orders'],1):
            lines += ["metadata%%order_names(%d)='%s'" % (i,order)]
        for i, powers in enumerate(r['amp_orders'],1):
            lines += ['metadata%%amplitude_orders(:,%d)=[%s]' % (i,','.join(map(str,powers)))]
        for sector in r['sectors']:
            f = sector['fks']
            lines += ['metadata%%fks(:,%d)=[%d,%d,%d,%d]' %
                      (sector['sector'], f['i'], f['j'], f['ij'], f['extra_cnt_index']+1)]
        tables = histories[r['directory']]
        lines += ['allocate(metadata%complete_histories(metadata%nsectors))',
                  'metadata%complete_histories=.true.',
                  'allocate(metadata%%histories(%d))' % sum(len(s['histories']) for s in tables)]
        index = 0
        for table in tables:
            if table['unresolved'] or table['ambiguous']:
                lines += ['metadata%%complete_histories(%d)=.false.' % table['sector']]
            for history in table['histories']:
                index += 1
                prefix = 'metadata%%histories(%d)%%' % index
                lines += [prefix+'owner_sector='+str(table['sector'])]
                for key in ('provider','context','sector','i','j','extra'):
                    lines += [prefix+key+'='+str(history[key])]
                for key in ('permutation','flavours','orders','amplitudes'):
                    lines += [prefix+key+'=['+','.join(map(str,history[key]))+']']
        lines += ['status=0']
    lines += ['end select', 'if(status.ne.0)return',
              'if(present(require_complete))then',
              'if(require_complete.and..not.all(metadata%complete_histories))status=5',
              'endif', 'end subroutine',
              'subroutine born_evaluate(provider,context,p,state,request,result,status)',
              'integer,intent(in)::provider,context', 'real(8),intent(in)::p(0:,:)',
              'type(BornModelState),intent(in)::state',
              'type(BornRequest),intent(in)::request',
              'type(BornResult),intent(inout)::result',
              'integer,intent(out)::status', 'type(BornMetadata)::metadata',
              'call born_reset_result(result)',
              'call born_query(provider,context,metadata,status,details=.false.)',
              'if(status.ne.0)return', 'status=3',
              'if(size(p,1).ne.4.or.size(p,2).ne.metadata%nexternal-1)return',
              'if(.not.allocated(state%real_values).or..not.allocated(state%complex_values))return',
              'if(size(state%%real_values).ne.%d.or.size(state%%complex_values).ne.%d)return' % (sizes['real'],sizes['complex']),
              'if(.not.allocated(request%charges))return',
              'if(size(request%charges).ne.metadata%nexternal-1)return',
              'if(request%extra.lt.0.or.request%extra.gt.metadata%extra)return',
              'if(request%single_helicity.lt.0.or.request%single_helicity.gt.metadata%nhelicity)return',
              'if(request%single_helicity.gt.0)then',
              'if(.not.allocated(request%helicity))return',
              'if(size(request%helicity).ne.metadata%nexternal-1)return',
              'endif',
              'if(request%colour.and.request%charge)return',
              'if(request%colour.or.request%charge)then',
              'if(min(request%m,request%n).lt.1.or.max(request%m,request%n).ge.metadata%nexternal)return',
              'endif', 'status=0', 'select case(provider)']
    for pid in sorted(providers):
        lines += ['case(%d)' % pid,
                  'call mcb_provider_%d(context,p,state,request,result,status)' % pid]
    lines += ['case default', 'status=1', 'end select', 'end subroutine', 'end module']
    (support/'mc_born_support.f90').write_text('\n'.join(lines)+'\n')


def write_wrappers(root, r, members):
    directory = root/'SubProcesses'/r['directory']
    lines = ['subroutine mc_born_local(p,request,result)',
             'use mc_born_support', 'use mc_born_types', 'implicit none',
             "include 'nexternal.inc'", "include 'coupl.inc'",
             'double precision p(0:3,nexternal-1)',
             'type(BornRequest) request', 'type(BornResult) result',
             'type(BornModelState),save::state', 'integer nr,nc,status,nfksprocess',
             'common/c_nfksprocess/nfksprocess',
             'double precision charges(nexternal-1)', 'common/c_charges_born/charges',
             ]
    lines += expand(root/'Source'/'MODEL'/'input.inc')
    lines += ['call born_model_dimensions(nr,nc)',
              'call born_resize_model_state(state,nr,nc)']
    lines += ['state%%%s_values(%s)=%s' % (kind, index, name) for name,kind,index in members]
    lines += ['request%sector=nfksprocess', 'request%charges=charges',
              'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context']),
              'if(status.ne.0)then', "write(*,*)'Born support evaluation failed',status", 'stop 1',
              'endif', 'end']
    capture = ['subroutine mc_capture_model_state(state)',
               'use mc_born_support','use mc_born_types','implicit none',
               "include 'coupl.inc'",'type(BornModelState) state','integer nr,nc']
    capture += expand(root/'Source'/'MODEL'/'input.inc')
    capture += ['call born_model_dimensions(nr,nc)',
                'call born_resize_model_state(state,nr,nc)']
    capture += ['state%%%s_values(%s)=%s' % (kind,index,name) for name,kind,index in members]
    write_fortran(directory/'born_support.f', lines+capture+['end'])
    declarations = ['use mc_born_types', 'implicit none', "include 'nexternal.inc'",
                    "include 'orders.inc'", 'double precision p(0:3,nexternal-1),ans',
                    'type(BornRequest),save::request', 'type(BornResult),save::result']
    lines = ['subroutine sborn(p,ans)']+declarations+[
        'double precision amp2(%d),jamp2(0:%d)' % (r['ngraphs'],r['ncolor']),
        'common/to_amps/amp2,jamp2', 'double complex ans_cnt(2,nsplitorders)',
        'common/c_born_cnt/ans_cnt', 'double precision wgt_ME_born,wgt_ME_real',
        'common/c_wgt_ME_tree/wgt_ME_born,wgt_ME_real',
        'logical calculatedBorn', 'common/ccalculatedBorn/calculatedBorn',
        'call mc_born_local(p,request,result)', 'ans=result%born',
        'wgt_ME_born=ans', 'amp_split=result%amplitudes',
        'amp_split_cnt=result%split_counterterms', 'ans_cnt=result%counterterms',
        'amp2=result%diagrams', 'jamp2(0)=%dd0' % r['ncolor'],
        'jamp2(1:%d)=result%%flows' % r['ncolor'], 'calculatedBorn=.true.', 'end']
    lines += ['subroutine sborn_onehel(p,nhel,hell,ans)']+declarations+[
        'integer nhel(nexternal-1),hell',
        'double complex amp_split_ewsud(amp_split_size)',
        'common/to_amp_split_ewsud/amp_split_ewsud',
        'double complex amp_split_ewsud_lo2(amp_split_size)',
        'common/to_amp_split_ewsud_lo2/amp_split_ewsud_lo2',
        'request%single_helicity=hell', 'request%helicity=nhel',
        'call mc_born_local(p,request,result)', 'ans=result%single_helicity',
        'amp_split_ewsud=result%ewsudakov',
        'amp_split_ewsud_lo2=result%ewsudakov_lo2', 'end']
    lines += list(statements((root/'Source/BornSupport'/('p%d' % r['provider'])/
                              'order_queries.f').read_text()))
    write_fortran(directory/'born.f', lines)
    lines = ['subroutine sborn_hel(p,ans)']+declarations+[
        'double precision helicities(%d)' % r['nhelicity'],
        'common/c_born_hel/helicities',
        'double precision helicity_orders(%d,%d)' % (r['nsqamps'],r['nhelicity']),
        'common/c_born_hel_split/helicity_orders',
        'request%helicities=.true.', 'call mc_born_local(p,request,result)',
        'helicities=result%helicities', 'helicity_orders=result%helicity_orders',
        'ans=sum(helicities)', 'end']
    lines += list(statements((root/'Source/BornSupport'/('p%d' % r['provider'])/
                              'helicity_sampler.f').read_text()))
    write_fortran(directory/'born_hel.f', lines)
    lines = ['subroutine sborn_sf(p,m,n,ans)']+declarations+[
        'integer m,n', 'logical need_color_links,need_charge_links',
        'common/c_need_links/need_color_links,need_charge_links',
        'double precision amp_split_soft(amp_split_size)',
        'common/to_amp_split_soft/amp_split_soft',
        'request%colour=need_color_links', 'request%charge=need_charge_links',
        'request%m=m', 'request%n=n', 'call mc_born_local(p,request,result)',
        'ans=result%correlation', 'amp_split_soft=result%soft',
        'amp_split_cnt=result%split_counterterms', 'end']
    write_fortran(directory/'sborn_sf.f', lines)
    # The extra-counterterm metadata functions remain local compatibility
    # queries, but the matrix-element implementation is in the provider.
    extra_path = directory/'extra_cnt_wrapper.f'
    extra = extra_path.read_text()
    body = ['subroutine extra_cnt(p,icnt,cnts)']+declarations+[
        'integer icnt', 'double complex cnts(2,nsplitorders)',
        'cnts=(0d0,0d0)', 'if(icnt.le.0)return', 'request%extra=icnt',
        'call mc_born_local(p,request,result)', 'cnts=result%extra',
        'amp_split_cnt=result%split_counterterms', 'end']
    for name in ('get_extra_cnt_color','get_extra_cnt_pdg','get_extra_cnt_charge'):
        body += routine(extra, name)
    write_fortran(extra_path, body)
    for path in list(directory.glob('b_sf_*.f'))+list(directory.glob('born_cnt_*.f')):
        path.unlink()


def write_build(root, support, providers, model_commons, records):
    template = Path(__file__).resolve().parents[2]/'Template'/'NLO'/'Source'/'BornSupport'
    for name in ('makefile', 'exports.map', 'exports.list'):
        (support/name).write_text((template/name).read_text())
    # Only HELAS and model functions are needed. Couplings are transferred, never
    # initialized by opening a param card relative to the process working dir.
    deps = support/'deps'
    deps.mkdir(exist_ok=True)
    dep_objects = []
    rules = []
    script = ('import re, sys\nfrom pathlib import Path\nMadGraph5Error=RuntimeError\n\n'+
              '\n\n'.join(inspect.getsource(f) for f in (statements,expand,rename,write_fortran))+
              '\nnames='+repr(model_commons)+'\n'+
              'write_fortran(sys.argv[2], [rename(line,names) for line in expand(sys.argv[1])])\n')
    (support/'refresh_dependency.py').write_text(script)
    for kind, paths in [('helas', sorted((root/'Source'/'DHELAS').glob('*.f'))),
                        ('model', [root/'Source'/'MODEL'/'model_functions.f'])]:
        for path in paths:
            if not path.exists():
                continue
            name = kind+'_'+path.name
            write_fortran(deps/name, [rename(line, model_commons) for line in expand(path)])
            dep_objects.append('deps/'+Path(name).stem+'.o')
            inputs = sorted(os.path.relpath(p,support) for p in include_dependencies(path))
            rules += ['deps/%s: refresh_dependency.py %s' % (name,' '.join(inputs)),
                      '\t$(PYTHON) refresh_dependency.py %s $@' % os.path.relpath(path,support)]
    # Model functions can be provided in a user file as well.
    # rw_para, couplings and param readers are intentionally not DSO dependencies.
    objects = ['p%d/%s.o' % (pid, name) for pid in sorted(providers)
               for name in ('matrix','adapter')]
    objects += ['c%d/metadata.o' % r['context'] for r in records]
    (support/'providers.mk').write_text('PROVIDER_OBJECTS = '+' '.join(objects)+'\n'+
        'DEPENDENCY_OBJECTS = '+' '.join(dep_objects)+'\n'+ '\n'.join(rules)+'\n')

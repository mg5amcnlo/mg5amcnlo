"""Pack independent full-model numerical fixtures into the compact public API.

The reference matrices deliberately see every original model COMMON. Production
state is built from the generated read set, and fixed snapshots are selected
explicitly when the fixture alternates between model configurations.
"""

import copy
import json
import re


def load_layout(output):
    return json.loads((output/'Source/BornSupport/model_state_layout.json').read_text())


def revision_helpers(output):
    """Use the actual generated cold-update counter in standalone fixtures."""
    from madgraph.iolibs import born_support
    lines = list(born_support.statements(
        (output/'Source/MODEL/couplings.f').read_text()))
    start = next(i for i, line in enumerate(lines) if re.match(
        r'(?i)\s*subroutine\s+mc_born_model_revision_value\b', line))
    return lines[start:]


def indices(index):
    bounds = list(map(int, str(index).split(':')))
    return list(range(bounds[0], bounds[-1]+1))


def full_signature(signature, members, layout):
    """Translate compact indices for the fixture's independent full vectors."""
    if not signature:
        return None
    result = copy.deepcopy(signature)
    original = {name: (kind, index) for name, kind, index in members}
    mapping = {'real': {}, 'complex': {}}
    used = {'real': set(), 'complex': set()}
    for group in ('fixed_members', 'dynamic_members'):
        for name, kind, index in layout[group]:
            old_kind, old_index = original[name]
            assert old_kind == kind
            old = indices(old_index)
            new = indices(index)
            assert len(old) == len(new)
            used[kind].update(old)
            if group == 'dynamic_members':
                mapping[kind].update(zip(new, old))
    result['g'] = mapping['real'][signature['g']]
    for kind in ('real', 'complex'):
        result[kind] = sorted(used[kind])
    result['couplings'] = [[kind, mapping[kind][index], degree]
                           for kind, index, degree in signature['couplings']]
    return result


def adapt_wide_driver(lines, members, sizes, layout):
    """Keep full values solely in the independently compiled test fixture."""
    original = {name: (kind, index) for name, kind, index in members}
    fixed = layout['fixed_sizes']
    dynamic = layout['dynamic_sizes']
    module = [
        'module model_state_fixture', 'use mc_born_support', 'use mc_born_types',
        'implicit none', 'type TestModelValues',
        'real(8),allocatable::real_values(:)',
        'complex(8),allocatable::complex_values(:)', 'end type',
        'type(BornModelConfiguration),save::configurations(64)', 'contains',
        'subroutine prepare_test_state(wide,state)',
        'type(TestModelValues),intent(in)::wide',
        'type(BornModelState),intent(inout)::state',
        'type(BornModelConfiguration)::candidate', 'integer i,slot,status',
        'allocate(candidate%%real_values(%d),candidate%%complex_values(%d))' %
        (fixed['real'], fixed['complex']),
    ]
    for name, kind, index in layout['fixed_members']:
        module.append('candidate%%%s_values(%s)=wide%%%s_values(%s)' %
                      (kind, index, kind, original[name][1]))
    module += ['slot=0', 'do i=1,size(configurations)',
               'if(configurations(i)%epoch.eq.0)cycle',
               'if(all(candidate%real_values.eq.configurations(i)%real_values).and.'
               'all(candidate%complex_values.eq.configurations(i)%complex_values))then',
               'slot=i', 'exit', 'endif', 'enddo', 'if(slot.eq.0)then',
               'do i=1,size(configurations)', 'if(configurations(i)%epoch.ne.0)cycle',
               'slot=i', 'exit', 'enddo', 'if(slot.eq.0)error stop 97',
               'configurations(slot)%real_values=candidate%real_values',
               'configurations(slot)%complex_values=candidate%complex_values',
               'call born_configure_model(configurations(slot),status)',
               'if(status.ne.0)error stop 98', 'else',
               'call born_activate_model(configurations(slot),status)',
               'if(status.ne.0)error stop 99', 'endif',
               'call born_resize_model_state(state,%d,%d)' %
               (dynamic['real'], dynamic['complex']),
               'state%configuration_epoch=configurations(slot)%epoch']
    for name, kind, index in layout['dynamic_members']:
        module.append('state%%%s_values(%s)=wide%%%s_values(%s)' %
                      (kind, index, kind, original[name][1]))
    module += ['end subroutine', 'end module']
    output = list(module)
    for line in lines:
        if re.match(r'(?i)^program\b', line):
            output += [line, 'use model_state_fixture']
            continue
        declaration = re.fullmatch(r'type\(BornModelState\) (state(?:,alternate)?)', line)
        if declaration:
            names = declaration[1].split(',')
            output += [line, 'type(TestModelValues) '+','.join('wide_'+n for n in names)]
            continue
        line = re.sub(r'\b(state|alternate)%', r'wide_\1%', line)
        assignment = re.fullmatch(r'(state|alternate)=(state|alternate)', line)
        if assignment:
            line = 'wide_'+assignment[1]+'=wide_'+assignment[2]
        wanted = []
        if re.search(r'\b(?:call born_evaluate|born_\w+_state_equal)\b', line):
            for name in ('state', 'alternate'):
                if re.search(r'(?<=[,(])'+name+r'(?=[,)])', line):
                    wanted.append(name)
        for name in wanted:
            output.append('call prepare_test_state(wide_%s,%s)' % (name, name))
        output.append(line)
    return output

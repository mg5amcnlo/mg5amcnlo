"""Compiled checks of shared real amplitudes using actual exported evaluators."""

import json
import re

from madgraph.iolibs import born_support as support
from models import import_ufo, model_reader


def check_real_cache(output, run_command):
    records = json.loads((output/'Source/BornSupport/registry.json').read_text())['contexts']
    model = support.expand(output/'Source/MODEL/coupl.inc')
    model += support.expand(output/'Source/MODEL/input.inc')
    members, sizes = support.common_members(model)
    # Boost comparisons require gauge-consistent couplings. Arbitrary unrelated
    # GC values are sufficient for provider equality, but violate Ward identities.
    physical = model_reader.ModelReader(import_ufo.import_model('sm'))
    physical.set_parameters_and_couplings(param_card=str(output/'Cards/param_card.dat'))
    values = {name.lower():complex(value) for dictionary in
              (physical['parameter_dict'],physical['coupling_dict'])
              for name,value in dictionary.items()}
    for name in ('mdl_wt','mdl_ww'):
        values[name] = 0j
    def real(value):
        return '%.17e' % value.real if isinstance(value,complex) else '%.17e' % value
    def literal(value,kind):
        if kind == 'complex':
            return '(%s,%s)' % (real(value.real).replace('e','d'),real(value.imag).replace('e','d'))
        return real(value).replace('e','d')
    total_direct, total_cached = 0, 0
    for record in records:
        if not record['sectors']:
            continue
        sub = output/'SubProcesses'/record['directory']
        work = output/('real_check_'+str(record['context']))
        work.mkdir(exist_ok=True)
        matrices = []
        oracle_names = {}
        wrappers = []
        for source in sorted(sub.glob('matrix_*.f')):
            number = int(source.stem.split('_')[-1])
            name = 'smatrix%d' % number
            support.write_fortran(work/source.name, [support.rename(line, {name:name+'_direct'})
                                  for line in support.statements(source.read_text())])
            matrices.append(work/source.name)
            source_lines = list(support.statements(source.read_text()))
            names = {name.lower():name.lower()+'_oracle' for line in source_lines
                     for name in re.findall(r'(?i)(?:subroutine|function)\s+(\w+)',line)}
            oracle_names[name] = name+'_oracle'
            oracle = [support.rename(line,names) for line in source_lines]
            # Independent all-helicity sum, retaining the actual diagrams and
            # colour/order contractions but bypassing every learned zero mask.
            oracle = [re.sub(r'(?i)\bntry\s*=\s*ntry\s*\+\s*1', 'NTRY=1',line)
                      for line in oracle]
            support.write_fortran(work/(source.stem+'_oracle.f'),oracle)
            matrices.append(work/(source.stem+'_oracle.f'))
            wrappers += ['subroutine %s(p,wgt)' % name, 'implicit none',
                         "include 'nexternal.inc'", 'real(8) p(0:3,nexternal),wgt',
                         'integer calls', 'common/test_real_calls/calls',
                         'calls=calls+1', 'call %s_direct(p,wgt)' % name, 'end']
        # Count actual matrix-evaluator calls, not entries to the cached API.
        support.write_fortran(work/'wrappers.f', wrappers)
        oracle_names['smatrix_real'] = 'smatrix_real_oracle'
        oracle_chooser = [support.rename(line,oracle_names) for line in
                          support.statements((sub/'real_me_chooser.f').read_text())]
        oracle_chooser += ['subroutine smatrix_reference(p,wgt)',
            'use mc_native_context,only:shared_real_active', 'implicit none',
            "include 'nexternal.inc'", 'real(8) p(0:3,nexternal),wgt', 'logical active',
            'active=shared_real_active', 'shared_real_active=.false.',
            'call smatrix_real_oracle(p,wgt)', 'shared_real_active=active', 'end']
        support.write_fortran(work/'oracle_chooser.f',oracle_chooser)
        support.write_fortran(work/'capture.f', support.routine(
            (sub/'born_support.f').read_text(), 'mc_capture_model_state'))
        support.write_fortran(work/'helpers.f', sum((support.routine(
            (sub/'splitorders_stuff.f').read_text(), name) for name in
            ('orders_to_amp_split_pos','amp_split_pos_to_orders','orders_equal')), []))
        lines = ['program check_real_cache', 'use mc_native_context', 'implicit none',
                 "include 'nexternal.inc'", "include 'orders.inc'", "include 'mc_histories.inc'",
                 *model, 'type(BornModelState) state',
                 'integer owner,ihist,i,j,iteration,nfksprocess,calls,before,hits,direct,stage',
                 'common/test_location/owner,ihist,j,iteration,stage',
                 'common/c_nfksprocess/nfksprocess', 'common/test_real_calls/calls',
                 'real(8) p(0:3,nexternal),q(0:3,nexternal),boosted(0:3,nexternal)',
                 'real(8) mass(nexternal),ans,reference(MC_HIST_COUNT),tmp(0:3)',
                 'real(8) reforders(amp_split_size,MC_HIST_COUNT),wgt_me_born,wgt_me_real',
                 'common/c_wgt_me_tree/wgt_me_born,wgt_me_real',
                 'allocate(state%%real_values(%d),state%%complex_values(%d))' %
                 (sizes['real'],sizes['complex']),
                 'state%real_values=0d0', 'state%complex_values=(0d0,0d0)']
        lines += ['state%%%s_values(%s)=%s' % (kind,index,literal(values.get(name,0j),kind))
                  for name,kind,index in members]
        lines += ['calls=0', 'hits=0', 'direct=0', 'do iteration=1,3']
        lines += ['%s=state%%%s_values(%s)' % member for member in members]
        for owner, sector in enumerate(record['sectors'], 1):
            masses = [literal(complex(physical.get_mass(leg[0])), 'real')
                      for leg in sector['processes'][0][1]]
            lines += ['owner=%d' % owner, 'mass=[%s]' % ','.join(masses),
                      'p=0d0', 'p(1:3,3)=[210d0+iteration,130d0,-90d0]',
                      'p(1:3,4)=[-70d0,190d0,-100d0]',
                      'p(1:3,5)=-p(1:3,3)-p(1:3,4)',
                      'do i=3,nexternal', 'p(0,i)=sqrt(sum(p(1:3,i)**2)+mass(i)**2)', 'enddo',
                      'p(0,1)=sum(p(0,3:nexternal))/2d0', 'p(0,2)=p(0,1)',
                      'p(3,1)=p(0,1)', 'p(3,2)=-p(0,2)',
                      'call mc_end_real_point()',
                      'do ihist=MC_HIST_FIRST(owner),MC_HIST_LAST(owner)',
                      'nfksprocess=MC_HIST_NATIVE(ihist)', 'call set_native_history(ihist)',
                      'do i=1,nexternal', 'q(:,i)=p(:,MC_HIST_PERM(i,ihist))', 'enddo',
                      'call smatrix_reference(q,reference(ihist))',
                      'reforders(:,ihist)=amp_split', 'direct=direct+2', 'enddo',
                      'call mc_begin_real_point(p)', 'before=calls',
                      'do ihist=MC_HIST_FIRST(owner),MC_HIST_LAST(owner)',
                      'nfksprocess=MC_HIST_NATIVE(ihist)', 'call set_native_history(ihist)',
                      'do i=1,nexternal', 'q(:,i)=p(:,MC_HIST_PERM(i,ihist))', 'enddo',
                      # Native tilde frames differ longitudinally. Compare the
                      # shared physical R with an independently evaluated boost.
                      'boosted=q', 'boosted(0,:)=cosh(0.3d0)*q(0,:)+sinh(0.3d0)*q(3,:)',
                      'boosted(3,:)=sinh(0.3d0)*q(0,:)+cosh(0.3d0)*q(3,:)',
                      'stage=1', 'call smatrix_reference(boosted,ans)',
                      'call close_real(ans,reference(ihist))',
                      'call mc_end_real_point()', 'call smatrix_real(boosted,ans)',
                      'call close_real(ans,reference(ihist))',
                      'shared_real_active=.true.',
                      'stage=2', 'do j=1,2',
                      'amp_split=-999d0', 'wgt_me_real=-999d0',
                      'call smatrix_real(boosted,ans)',
                      'call close_real(ans,reference(ihist))',
                      'call close_real(wgt_me_real,reference(ihist))',
                      'do i=1,amp_split_size',
                      'call close_real(amp_split(i),reforders(i,ihist))', 'enddo', 'enddo', 'enddo',
                      'hits=hits+3*(MC_HIST_LAST(owner)-MC_HIST_FIRST(owner)+1)-(calls-before)',
                      # Changing a coupling with the point still active must
                      # invalidate reuse. It must also restore the original
                      # result after A -> B -> A model-state changes.
                      'call mc_capture_model_state(state)']
            changed = next(v for v, kind, index in members
                           if kind == 'complex' and v.startswith('gc_') and values.get(v,0j))
            lines += [changed+'='+changed+'*(0.93d0,0.02d0)',
                      'i=calls', 'call smatrix_real(q,ans)', 'if(calls.ne.i+1)stop 93',
                      'call mc_end_real_point()',
                      'stage=3', 'call smatrix_reference(q,tmp(0))', 'call close_real(ans,tmp(0))',
                      'shared_real_active=.true.']
            lines += [name+'=(0d0,0d0)' for name,kind,index in members if name.startswith('gc_')]
            lines += ['call smatrix_real(q,ans)', 'call close_real(ans,0d0)',
                      'call smatrix_reference(q,tmp(0))', 'call close_real(ans,tmp(0))']
            lines += ['%s=state%%%s_values(%s)' % member for member in members]
            lines += ['shared_real_active=.true.', 'stage=4', 'call smatrix_real(q,ans)',
                      'call close_real(ans,reference(MC_HIST_LAST(owner)))',
                      'call mc_end_real_point()', 'call set_native_history(0)']
        lines += ['enddo', 'if(hits.le.0)stop 92',
                  "write(*,*)'PASS real cache',direct,hits", 'end',
                  'subroutine close_real(a,b)', 'use,intrinsic::ieee_arithmetic',
                  'implicit none', 'real(8) a,b',
                  'integer owner,ihist,j,iteration,stage',
                  'common/test_location/owner,ihist,j,iteration,stage',
                  'if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b))stop 90',
                  'if(abs(a-b).gt.1d-11*max(1d-50,abs(a),abs(b)))then',
                  "write(*,*)'Mismatch',a,b,owner,ihist,j,iteration,stage", 'stop 91', 'endif', 'end',
                  'subroutine mc_sync_native_tables()', 'end']
        support.write_fortran(work/'check.f', lines)
        paths = [sub/'mc_native_context.f90', work/'capture.f', sub/'real_me_chooser.f',
                 *matrices, work/'wrappers.f', work/'oracle_chooser.f',work/'helpers.f',work/'check.f',
                 *sorted((output/'Source/DHELAS').glob('*.f')),
                 output/'Source/MODEL/model_functions.f']
        run_command(['gfortran','-O0','-g','-fcheck=all','-ffixed-line-length-none',
                     '-ffree-line-length-none','-I'+str(sub),'-I'+str(output/'lib'),
                     *map(str,paths),'-L'+str(output/'lib'),'-lmc_born_support',
                     '-Wl,-rpath,'+str(output/'lib'),'-o','check'], work)
        result = run_command(['./check'],work)
        row = next(line for line in result.splitlines() if 'PASS real cache' in line)
        direct, hits = map(int, row.split()[-2:])
        total_direct += direct
        total_cached += direct-hits
    return total_direct, total_cached

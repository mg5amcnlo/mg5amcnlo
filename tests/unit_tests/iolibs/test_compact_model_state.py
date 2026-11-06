"""Compiled checks of fixed configurations and compact per-point model state."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs import born_support as support


MODEL_LINES = [
    'double precision fixed_mass,fixed_width',
    'double complex fixed_weak,fixed_pair(2)',
    'common/test_fixed_real/fixed_mass,fixed_width',
    'common/test_fixed_complex/fixed_weak,fixed_pair',
]
FIXED_MEMBERS = [('fixed_mass', 'real', '1'),
                 ('fixed_width', 'real', '2'),
                 ('fixed_weak', 'complex', '1'),
                 ('fixed_pair', 'complex', '2:3')]
DYNAMIC_MEMBERS = [('g', 'real', '1'), ('real_only_coupling', 'complex', '1')]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestCompactModelConfiguration(unittest.TestCase):
    def run_driver(self, body, empty_dynamic=False):
        """Use the production API and installer with two small providers."""
        records = [dict(directory='P%d' % index, provider=index, context=index,
                        nexternal=2, nincoming=1, nprocesses=1, ngraphs=1,
                        ncolor=1, nhelicity=1, namps=1, nsqamps=1, extra=0,
                        orders=['QCD'], amp_orders=[(1,)], sectors=[],
                        born_identity=1) for index in (1, 2)]
        dynamic_sizes = dict(real=0 if empty_dynamic else 1,
                             complex=0 if empty_dynamic else 1)
        layout = dict(fixed_members=FIXED_MEMBERS,
                      sizes=dict(fixed=dict(real=2, complex=3),
                                 dynamic=dynamic_sizes))
        with tempfile.TemporaryDirectory(prefix='mg7_compact_configuration_') as tmp:
            work = Path(tmp)
            support.write_api(work, records,
                              {record['provider']: record for record in records},
                              dynamic_sizes, {record['directory']: [] for record in records},
                              model_layout=layout, model_lines=MODEL_LINES,
                              model_commons={})
            # Count the generated fixed assignments themselves. This catches
            # reinstalling fixed fields on every evaluation or activation.
            api_path = work / 'mc_born_support.f90'
            source = api_path.read_text().replace(
                'implicit none', 'implicit none\ninteger fixed_installs\n'
                'common/test_fixed_installs/fixed_installs', 1)
            source, count = re.subn(
                r'(?im)^(\s*)(fixed_mass\s*=.*)$',
                r'\1fixed_installs=fixed_installs+1\n\1\2', source)
            self.assertEqual(count, 1, 'fixed installation must have one explicit writer')
            api_path.write_text(source)
            stubs = []
            for index in (1, 2):
                stubs.append('''
subroutine mcb_metadata_%d(metadata)
use mc_born_types
implicit none
type(BornMetadata) metadata
end subroutine
subroutine mcb_provider_%d(context,p,state,request,result,status)
use mc_born_types
implicit none
integer context,status,calls(2)
real(8) p(0:3,1),fixed_mass,fixed_width
complex(8) fixed_weak,fixed_pair(2)
type(BornModelState) state
type(BornRequest) request
type(BornResult) result
common/test_provider_calls/calls
common/test_fixed_real/fixed_mass,fixed_width
common/test_fixed_complex/fixed_weak,fixed_pair
calls(%d)=calls(%d)+1
result%%born=%dd0+fixed_mass+fixed_width+real(fixed_weak+sum(fixed_pair),8)
if(size(state%%real_values).gt.0)result%%born=result%%born+state%%real_values(1)
if(size(state%%complex_values).gt.0)result%%born=result%%born+real(state%%complex_values(1),8)
status=BORN_OK
end subroutine
''' % (index, index, index, index, 100 * index))
            (work / 'providers.f90').write_text('\n'.join(stubs))
            (work / 'check.f90').write_text('''program check
use mc_born_types
use mc_born_support
implicit none
type(BornModelConfiguration) config_a,config_b,invalid
type(BornModelState) state,previous
type(BornRequest) request
type(BornResult) result
real(8) p(0:3,1)
integer status,nr,nc,i,calls(2),fixed_installs
integer(8) first_epoch
common/test_provider_calls/calls
common/test_fixed_installs/fixed_installs
calls=0
fixed_installs=0
p=1d0
request%charges=[0d0]
call born_model_configuration_dimensions(nr,nc)
if(nr.ne.2.or.nc.ne.3)stop 1
config_a%real_values=[3d0,1d0]
config_a%complex_values=[(2d0,0d0),(4d0,0d0),(5d0,0d0)]
call born_model_dimensions(nr,nc)
call born_resize_model_state(state,nr,nc)
state%real_values=2d0
state%complex_values=(3d0,0d0)
''' + body + '\nend program\n')
            command = ['gfortran', '-O2', '-fcheck=all', '-fno-automatic',
                       '-ffree-line-length-none', 'mc_born_types.f90',
                       'mc_born_support.f90', 'providers.f90', 'check.f90',
                       '-o', 'check']
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_configuration_snapshots_restore_a_b_a_between_providers(self):
        self.run_driver('''
call born_configure_model(config_a,status)
if(status.ne.BORN_OK.or.config_a%epoch.le.0)stop 10
if(fixed_installs.ne.1)stop 11
first_epoch=config_a%epoch
state%configuration_epoch=config_a%epoch
call born_evaluate(1,1,p,state,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.120d0)stop 12
config_b=config_a
config_b%real_values(1)=7d0
config_b%complex_values(1)=(6d0,0d0)
call born_configure_model(config_b,status)
if(status.ne.BORN_OK.or.config_b%epoch.eq.first_epoch)stop 13
if(config_a%real_values(1).ne.3d0.or.config_a%complex_values(1).ne.(2d0,0d0))stop 14
if(fixed_installs.ne.2)stop 15
state%configuration_epoch=config_b%epoch
call born_evaluate(2,2,p,state,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.228d0)stop 16
call born_activate_model(config_a,status)
if(status.ne.BORN_OK.or.config_a%epoch.ne.first_epoch)stop 17
state%configuration_epoch=config_a%epoch
do i=1,4
  call born_evaluate(1,1,p,state,request,result,status)
  if(status.ne.BORN_OK.or.result%born.ne.120d0)stop 18
  call born_activate_model(config_a,status)
  if(status.ne.BORN_OK)stop 19
  call born_evaluate(2,2,p,state,request,result,status)
  if(status.ne.BORN_OK.or.result%born.ne.220d0)stop 20
enddo
if(fixed_installs.ne.3)stop 21
if(any(calls.ne.[5,5]))stop 22
''')

    def test_unconfigured_or_inactive_state_never_enters_a_provider(self):
        self.run_driver('''
call born_evaluate(1,1,p,state,request,result,status)
if(status.ne.BORN_INVALID_CONFIGURATION.or.any(calls.ne.0))stop 10
call born_configure_model(config_a,status)
if(status.ne.BORN_OK)stop 11
state%configuration_epoch=config_a%epoch
previous=state
config_b=config_a
call born_configure_model(config_b,status)
if(status.ne.BORN_OK)stop 12
result%born=-123d0
call born_evaluate(1,1,p,state,request,result,status)
if(status.ne.BORN_INVALID_CONFIGURATION.or.any(calls.ne.0))stop 13
if(result%born.ne.0d0)stop 14
call born_activate_model(config_a,status)
if(status.ne.BORN_OK)stop 15
call born_evaluate(1,1,p,previous,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.120d0.or.calls(1).ne.1)stop 16
''')

    def test_invalid_configuration_does_not_displace_the_active_snapshot(self):
        self.run_driver('''
call born_configure_model(config_a,status)
if(status.ne.BORN_OK)stop 10
state%configuration_epoch=config_a%epoch
call born_activate_model(invalid,status)
if(status.ne.BORN_INVALID_CONFIGURATION)stop 11
invalid%real_values=[3d0]
invalid%complex_values=config_a%complex_values
call born_configure_model(invalid,status)
if(status.ne.BORN_INVALID_CONFIGURATION)stop 12
call born_evaluate(1,1,p,state,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.120d0)stop 13
if(fixed_installs.ne.1)stop 14
''')

    def test_fixed_only_evaluator_accepts_zero_length_dynamic_vectors(self):
        self.run_driver('''
if(nr.ne.0.or.nc.ne.0)stop 10
call born_configure_model(config_a,status)
if(status.ne.BORN_OK)stop 11
state%configuration_epoch=config_a%epoch
call born_evaluate(1,1,p,state,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.115d0)stop 12
call born_evaluate(2,2,p,state,request,result,status)
if(status.ne.BORN_OK.or.result%born.ne.215d0)stop 13
if(fixed_installs.ne.1)stop 14
''', empty_dynamic=True)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeModelCapture(unittest.TestCase):
    def run_capture_driver(self, body):
        record = dict(directory='P1', provider=1, context=1, nexternal=2,
                      nincoming=1, nprocesses=1, ngraphs=1, ncolor=1,
                      nhelicity=1, namps=1, nsqamps=1, extra=0, orders=['QCD'],
                      amp_orders=[(1,)], sectors=[], born_identity=1)
        layout = dict(fixed_members=FIXED_MEMBERS, dynamic_members=DYNAMIC_MEMBERS,
                      sizes=dict(fixed=dict(real=2, complex=3),
                                 dynamic=dict(real=1, complex=1)))
        private_commons = {name: 'private_' + name
                           for name in ('test_fixed_real', 'test_fixed_complex')}
        native_model = MODEL_LINES + [
            'double precision g,unused_parameter',
            'double complex real_only_coupling,unused_counterterm',
            'common/test_dynamic_real/g,unused_parameter',
            'common/test_dynamic_complex/real_only_coupling,unused_counterterm']
        with tempfile.TemporaryDirectory(prefix='mg7_compact_capture_') as tmp:
            root = Path(tmp)
            model = root / 'Source/MODEL'
            model.mkdir(parents=True)
            generated = root / 'Source/BornSupport'
            (generated / 'p1').mkdir(parents=True)
            sub = root / 'SubProcesses/P1'
            sub.mkdir(parents=True)
            (model / 'input.inc').write_text('')
            support.write_fortran(sub / 'coupl.inc', native_model)
            for name in ('order_queries.f', 'helicity_sampler.f'):
                (generated / 'p1' / name).write_text('')
            support.write_fortran(sub / 'extra_cnt_wrapper.f', [
                line for name in ('get_extra_cnt_color', 'get_extra_cnt_pdg',
                                  'get_extra_cnt_charge')
                for line in ('subroutine %s()' % name, 'end')])
            support.write_api(generated, [record], {1: record}, layout['sizes']['dynamic'],
                              {'P1': []}, model_layout=layout, model_lines=MODEL_LINES,
                              model_commons=private_commons)
            api_path = generated / 'mc_born_support.f90'
            api_source = api_path.read_text().replace(
                'implicit none', 'implicit none\ninteger fixed_installs\n'
                'common/test_fixed_installs/fixed_installs', 1)
            api_source, count = re.subn(
                r'(?im)^(\s*)(fixed_mass\s*=.*)$',
                r'\1fixed_installs=fixed_installs+1\n\1\2', api_source)
            self.assertEqual(count, 1)
            api_path.write_text(api_source)
            support.write_wrappers(root, record, DYNAMIC_MEMBERS, model_layout=layout)
            native_source = (sub / 'born_support.f').read_text()
            capture = support.routine(native_source, 'mc_capture_model_state')
            prepare = support.routine(native_source, 'mc_prepare_model_configuration')
            # Instrument actual generated reads/copies, including array extents.
            # Declarations may mention unused model fields; executable loads may not.
            read_pattern = re.compile(
                r'\b(?:state|configuration|candidate)%(real|complex)_values\((\d+)(?::(\d+))?\)\s*=\s*(\w+)',
                re.I)
            observed = []
            instrumented = []
            for routine in (capture, prepare):
                for line in routine:
                    instrumented.append(line)
                    if line.strip().lower() == 'implicit none':
                        instrumented += [
                            'integer capture_calls,dynamic_bytes,fixed_bytes',
                            'common/test_capture_counts/capture_calls,dynamic_bytes,fixed_bytes']
                    if line.lower().startswith('call mc_prepare_model_configuration('):
                        instrumented.insert(len(instrumented) - 1,
                                            'capture_calls=capture_calls+1')
                    match = read_pattern.search(line)
                    if match:
                        kind, first, last, name = match.groups()
                        count = int(last or first) - int(first) + 1
                        nbytes = (8 if kind.lower() == 'real' else 16) * count
                        counter = 'dynamic_bytes' if line.lower().startswith('state%') else 'fixed_bytes'
                        instrumented.insert(len(instrumented) - 1,
                                            '%s=%s+%d' % (counter, counter, nbytes))
                        observed.append(name.lower())
            self.assertEqual(set(observed), {name for name, _, _ in
                                            FIXED_MEMBERS + DYNAMIC_MEMBERS})
            self.assertNotIn('unused_parameter', observed)
            self.assertNotIn('unused_counterterm', observed)
            support.write_fortran(root / 'capture.f', instrumented)
            support.write_fortran(model / 'couplings.f',
                                  ['subroutine coup()', 'implicit none', 'return', 'end'])
            support.write_model_revision_hook(root)
            hook_source = (model / 'couplings.f').read_text()
            support.write_model_revision_hook(root)
            self.assertEqual((model / 'couplings.f').read_text(), hook_source)
            (root / 'providers.f90').write_text('''
subroutine mcb_metadata_1(metadata)
use mc_born_types
implicit none
type(BornMetadata) metadata
end subroutine
subroutine mcb_provider_1(context,p,state,request,result,status)
use mc_born_types
implicit none
integer context,status
real(8) p(0:3,1),fixed_mass,fixed_width
complex(8) fixed_weak,fixed_pair(2)
type(BornModelState) state
type(BornRequest) request
type(BornResult) result
common/private_test_fixed_real/fixed_mass,fixed_width
common/private_test_fixed_complex/fixed_weak,fixed_pair
result%born=fixed_mass+fixed_width+real(fixed_weak+sum(fixed_pair),8)+ &
            state%real_values(1)+real(state%complex_values(1),8)
status=BORN_OK
end subroutine
''')
            driver = ['program check', 'use mc_born_types', 'use mc_born_support',
                      'implicit none', *native_model,
                      'type(BornModelConfiguration) alternate',
                      'type(BornModelState) state,previous',
                      'type(BornRequest) request', 'type(BornResult) result',
                      'real(8) p(0:3,1)', 'integer status,i,capture_calls,dynamic_bytes,fixed_bytes,fixed_installs',
                      'integer(8) first_epoch,mc_born_model_revision',
                      'external mc_born_model_revision',
                      'common/test_capture_counts/capture_calls,dynamic_bytes,fixed_bytes',
                      'common/test_fixed_installs/fixed_installs',
                      'capture_calls=0', 'dynamic_bytes=0', 'fixed_bytes=0', 'fixed_installs=0',
                      'fixed_mass=3d0', 'fixed_width=1d0', 'fixed_weak=(2d0,0d0)',
                      'fixed_pair=[(4d0,0d0),(5d0,0d0)]', 'g=2d0',
                      'real_only_coupling=(3d0,0d0)', 'unused_parameter=99d0',
                      'unused_counterterm=(99d0,77d0)', 'p=1d0',
                      'request%charges=[0d0]']
            support.write_fortran(root / 'check.f', driver + list(support.statements(body)) + ['end'])
            command = ['gfortran', '-O2', '-fcheck=all', '-fno-automatic',
                       '-ffixed-line-length-none', '-ffree-line-length-none', '-I' + str(sub),
                       str(generated / 'mc_born_types.f90'),
                       str(generated / 'mc_born_support.f90'), 'providers.f90',
                       str(model / 'couplings.f'), 'capture.f', 'check.f', '-o', 'check']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(root / 'check')], cwd=root,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_only_dynamic_bytes_are_captured_on_repeated_real_requests(self):
        self.run_capture_driver('''
      call coup()
      if(mc_born_model_revision().ne.1)stop 10
      call mc_capture_model_state(state)
      first_epoch=state%configuration_epoch
      previous=state
      if(fixed_bytes.ne.64.or.dynamic_bytes.ne.24)stop 11
      do i=1,4
        unused_parameter=unused_parameter+1d0
        unused_counterterm=unused_counterterm+(1d0,1d0)
        call mc_capture_model_state(state)
        if(.not.born_model_state_equal(previous,state))stop 12
      enddo
      if(capture_calls.ne.5.or.dynamic_bytes.ne.5*24.or.fixed_bytes.ne.64)stop 13
      ! A coupling used only by real emission remains part of the dynamic key.
      real_only_coupling=(4d0,1d0)
      call mc_capture_model_state(state)
      if(born_model_state_equal(previous,state))stop 14
      if(state%configuration_epoch.ne.first_epoch)stop 15
      if(dynamic_bytes.ne.6*24.or.fixed_bytes.ne.64)stop 16
      call born_evaluate(1,1,p,state,request,result,status)
      if(status.ne.BORN_OK.or.result%born.ne.21d0)stop 17
''')

    def test_completed_coup_refreshes_fixed_state_and_capture_reactivates_native_config(self):
        self.run_capture_driver('''
      call coup()
      call mc_capture_model_state(state)
      first_epoch=state%configuration_epoch
      alternate%real_values=[100d0,1d0]
      alternate%complex_values=[(2d0,0d0),(4d0,0d0),(5d0,0d0)]
      call born_configure_model(alternate,status)
      if(status.ne.BORN_OK)stop 10
      ! No COUP update occurred: reactivate the native saved configuration.
      call mc_capture_model_state(state)
      if(state%configuration_epoch.ne.first_epoch.or.fixed_bytes.ne.64)stop 11
      call born_evaluate(1,1,p,state,request,result,status)
      if(status.ne.BORN_OK.or.result%born.ne.20d0)stop 12
      fixed_mass=7d0
      fixed_weak=(6d0,0d0)
      call coup()
      if(mc_born_model_revision().ne.2)stop 13
      call mc_capture_model_state(state)
      if(state%configuration_epoch.eq.first_epoch.or.fixed_bytes.ne.128)stop 14
      call born_evaluate(1,1,p,state,request,result,status)
      if(status.ne.BORN_OK.or.result%born.ne.28d0)stop 15
      call mc_capture_model_state(state)
      if(fixed_bytes.ne.128)stop 16
      ! Re-reading configuration A gets a fresh identity and the original value.
      fixed_mass=3d0
      fixed_weak=(2d0,0d0)
      call coup()
      call mc_capture_model_state(state)
      if(fixed_bytes.ne.192.or.state%configuration_epoch.eq.first_epoch)stop 17
      call born_evaluate(1,1,p,state,request,result,status)
      if(status.ne.BORN_OK.or.result%born.ne.20d0)stop 18
''')

    def test_cold_unused_or_dynamic_updates_preserve_fixed_configuration_identity(self):
        self.run_capture_driver('''
      call coup()
      call mc_capture_model_state(state)
      previous=state
      first_epoch=state%configuration_epoch
      if(fixed_bytes.ne.64.or.fixed_installs.ne.1)stop 10
      ! A completed model update can change fields absent from every evaluator.
      unused_parameter=101d0
      unused_counterterm=(3d0,9d0)
      call coup()
      call mc_capture_model_state(state)
      if(.not.born_model_state_equal(previous,state))stop 11
      if(state%configuration_epoch.ne.first_epoch.or.fixed_installs.ne.1)stop 12
      ! A full COUP update of G changes dynamic values, not fixed identity.
      g=3d0
      call coup()
      call mc_capture_model_state(state)
      if(born_model_state_equal(previous,state))stop 13
      if(state%configuration_epoch.ne.first_epoch.or.fixed_installs.ne.1)stop 14
      previous=state
      alternate%real_values=[100d0,1d0]
      alternate%complex_values=[(2d0,0d0),(4d0,0d0),(5d0,0d0)]
      call born_configure_model(alternate,status)
      if(status.ne.BORN_OK.or.fixed_installs.ne.2)stop 15
      ! Revision changed while an external configuration is active: reactivate
      ! the original native ID after comparing the unchanged fixed candidate.
      call coup()
      call mc_capture_model_state(state)
      if(.not.born_model_state_equal(previous,state))stop 16
      if(state%configuration_epoch.ne.first_epoch.or.fixed_installs.ne.3)stop 17
      if(fixed_bytes.ne.4*64.or.dynamic_bytes.ne.4*24)stop 18
      call mc_capture_model_state(state)
      if(fixed_bytes.ne.4*64.or.dynamic_bytes.ne.5*24.or.fixed_installs.ne.3)stop 19
      call born_evaluate(1,1,p,state,request,result,status)
      if(status.ne.BORN_OK.or.result%born.ne.21d0)stop 20
''')

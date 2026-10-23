"""Compile actual generated real MEs at native near-beam roundoff points."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs import born_support as support
from models import import_ufo, model_reader

ROOT = Path(__file__).resolve().parents[3]
POINTS = ROOT/'tests/input_files/real_me_near_beam_points.json'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestRealMEBasis(unittest.TestCase):
    @staticmethod
    def run_command(command, cwd):
        result = subprocess.run(list(map(str, command)), cwd=cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise AssertionError(' '.join(map(str, command))+'\n'+result.stdout)
        return result.stdout

    def test_generated_real_me_uses_stable_invariant_basis(self):
        from madgraph.interface.master_interface import MasterCmd
        with tempfile.TemporaryDirectory(prefix='mg5_real_basis_') as temporary:
            root = Path(temporary)
            interface = MasterCmd()
            def command(text):
                interface.exec_cmd(text, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            command('set automatic_html_opening False --no_save')
            command('set low_mem_multicore_nlo_generation False --no_save')
            command('import model sm')
            fixtures = json.loads(POINTS.read_text())
            for name, process in [('dy', 'u u~ > e+ e-'), ('zz', 'd d~ > z z')]:
                with self.subTest(process=name):
                    command('generate '+process+' QED^2=4 QCD^2=0 [real=QCD]')
                    output = root/name
                    command('output %s -f -nojpeg' % output)
                    selected = [p for p in fixtures if p['process'] == name]
                    directories = sorted((output/'SubProcesses').glob('P*'))
                    self.assertEqual(len(directories), 1)
                    self.check_export(output, directories[0], selected)

    def test_basis_does_not_move_another_leg_onto_the_spinor_pole(self):
        from madgraph.iolibs import native_histories
        # A balanced massless 2 -> 6 state occupies all six Cartesian axes.
        # One fixed ninety-degree rotation merely moves the singularity to
        # another external leg; the selected basis must avoid every pole.
        state = ('restrictions', tuple((21, i >= 2, (), False, None)
                                       for i in range(8)))
        record = dict(context=1, sectors=[dict(real=1, processes=[state])])
        with tempfile.TemporaryDirectory(prefix='mg5_real_basis_axes_') as temporary:
            work = Path(temporary)
            support.write_fortran(work/'nexternal.inc', ['integer,parameter::nexternal=8'])
            support.write_fortran(work/'orders.inc', ['integer,parameter::amp_split_size=1',
                'real(8) amp_split(1)', 'common/to_amp_split/amp_split'])
            native_histories.write_dispatchers(work, record, {1: record}, [(1, 1)], {})
            (work/'mc_native_context.f90').write_text('''module mc_native_context
implicit none
logical,save::shared_real_active=.true.
integer,save::shared_real_epoch=1,active_history=0
real(8),save::shared_real_point(0:3,8)
integer,save::history_permutations(8,1)=0
end module
''')
            program = ['program check', 'use mc_native_context', 'implicit none',
                'real(8) p(0:3,8),wgt', 'integer nfksprocess',
                'common/c_nfksprocess/nfksprocess', 'p=0d0', 'p(0,:)=1d0',
                'p(0,1:2)=3d0', 'p(3,1)=3d0', 'p(3,2)=-3d0',
                'p(1,3)=1d0', 'p(1,4)=-1d0', 'p(2,5)=1d0',
                'p(2,6)=-1d0', 'p(3,7)=1d0', 'p(3,8)=-1d0',
                'shared_real_point=p', 'nfksprocess=1', 'call smatrix_real(p,wgt)',
                'if(any(p.ne.shared_real_point))stop 1', 'end',
                'subroutine smatrix1(p,wgt)', 'implicit none',
                "include 'orders.inc'", 'real(8) p(0:3,8),wgt,det',
                'if(minval((p(0,:)+p(3,:))/p(0,:)).lt.0.1d0)stop 2',
                'if(any(abs(sum(p(1:3,:)**2,dim=1)-p(0,:)**2).gt.1d-14))stop 3',
                'if(maxval(abs(sum(p(:,1:2),dim=2)-sum(p(:,3:8),dim=2))).gt.1d-14)stop 4',
                'det=p(1,3)*(p(2,5)*p(3,7)-p(3,5)*p(2,7))',
                'det=det-p(2,3)*(p(1,5)*p(3,7)-p(3,5)*p(1,7))',
                'det=det+p(3,3)*(p(1,5)*p(2,7)-p(2,5)*p(1,7))',
                'if(abs(det-1d0).gt.1d-14)stop 5', 'wgt=1d0', 'amp_split=wgt', 'end',
                'subroutine mc_capture_model_state(state)', 'use mc_born_types',
                'implicit none', 'type(BornModelState) state',
                'state%real_values=[1d0]', 'state%complex_values=[(1d0,0d0)]', 'end']
            support.write_fortran(work/'check.f', program)
            self.run_command(['gfortran', '-fcheck=all', '-ffixed-line-length-none',
                '-ffree-line-length-none', ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90',
                'mc_native_context.f90', 'real_me_chooser.f', 'check.f', '-o', 'check'], work)
            self.run_command(['./check'], work)

    def check_export(self, output, sub, fixtures):
        work = output/'basis_test'
        work.mkdir()
        routes = []
        for line in (sub/'leshouche_info.dat').read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == 'I':
                routes.append((int(fields[1]), list(map(int, fields[3:]))))
        reordered = []
        for fixture in fixtures:
            ids = fixture['ids']
            sector, labels = next((sector, labels) for sector, labels in routes
                                  if labels[:2] == ids[:2] and sorted(labels[2:]) == sorted(ids[2:]))
            remaining = list(range(2, len(ids)))
            permutation = [0, 1]
            for pid in labels[2:]:
                index = next(i for i in remaining if ids[i] == pid)
                remaining.remove(index)
                permutation.append(index)
            reordered.append((fixture, sector, [fixture['momenta'][i] for i in permutation]))
        model = support.expand(output/'Source/MODEL/coupl.inc')
        model += support.expand(output/'Source/MODEL/input.inc')
        members, sizes = support.common_members(model)
        physical = model_reader.ModelReader(import_ufo.import_model('sm'))
        physical.set_parameters_and_couplings(param_card=str(output/'Cards/param_card.dat'))
        values = {name.lower(): complex(value) for table in
                  (physical['parameter_dict'], physical['coupling_dict'])
                  for name, value in table.items()}
        def literal(value):
            return ('%.17e' % value).replace('e', 'd')
        def member_value(name, kind):
            v = values.get(name, 0j)
            if kind == 'complex':
                return '(%s,%s)' % (literal(v.real), literal(v.imag))
            return literal(v.real)
        (work/'mc_native_context.f90').write_text('''module mc_born_support
contains
subroutine born_model_dimensions(nreal,ncomplex)
integer,intent(out)::nreal,ncomplex
nreal=%d
ncomplex=%d
end subroutine
end module
module mc_native_context
implicit none
logical,save::shared_real_active=.false.
integer,save::shared_real_epoch=0,active_history=0
real(8),save::shared_real_point(0:3,5)
integer,save::history_permutations(5,1)=0
end module
''' % (sizes['real'], sizes['complex']))
        chooser = list(support.statements((sub/'real_me_chooser.f').read_text()))
        # A separate dispatch oracle uses actual matrices without the new basis
        # helper. Its inputs are explicitly rotated in two independent bases.
        oracle = chooser[:next(i for i, x in enumerate(chooser) if x.strip().lower() == 'contains')]
        oracle = [support.rename(x, {'smatrix_real': 'smatrix_real_direct'}) for x in oracle
                  if 'call stable_real_basis' not in x.lower()]
        oracle += ['end']
        support.write_fortran(work/'direct_chooser.f', oracle)
        support.write_fortran(work/'capture.f', support.routine(
            (sub/'born_support.f').read_text(), 'mc_capture_model_state'))
        helpers = sum((support.routine((sub/'splitorders_stuff.f').read_text(), name)
                       for name in ('orders_to_amp_split_pos', 'amp_split_pos_to_orders', 'orders_equal')), [])
        support.write_fortran(work/'helpers.f', helpers)
        matrices, wrappers = [], []
        for source in sorted(sub.glob('matrix_*.f')):
            number = int(source.stem.split('_')[-1])
            entry = 'smatrix%d' % number
            target = work/source.name
            support.write_fortran(target, [support.rename(x, {entry: entry+'_physical'})
                                          for x in support.statements(source.read_text())])
            matrices.append(target)
            wrappers += ['subroutine %s(p,wgt)' % entry, 'implicit none',
                         "include 'nexternal.inc'", 'real(8) p(0:3,nexternal),wgt,last(0:3,nexternal)',
                         'integer calls', 'common/test_eval_calls/calls',
                         'common/test_eval_point/last', 'calls=calls+1', 'last=p',
                         'call %s_physical(p,wgt)' % entry, 'end']
        support.write_fortran(work/'wrappers.f', wrappers)
        program = ['program check', 'use mc_native_context', 'implicit none',
                   "include 'nexternal.inc'", "include 'orders.inc'", *model,
                   'real(8) p(0:3,nexternal),original(0:3,nexternal),a(0:3,nexternal),b(0:3,nexternal),last(0:3,nexternal)',
                   'real(8) w,raw,ref,ref2,saved(amp_split_size),dot0,dot1,scale,maxdiff',
                   'integer nfksprocess,calls,before,i,j', 'common/c_nfksprocess/nfksprocess',
                   'common/test_eval_calls/calls', 'common/test_eval_point/last',
                   'calls=0', 'maxdiff=0d0']
        program += ['%s=%s' % (name, member_value(name, kind)) for name, kind, index in members]
        for index, (fixture, sector, momenta) in enumerate(reordered, 1):
            program += ['! '+fixture['label'], 'nfksprocess=%d' % sector]
            program += ['p(:,%d)=[%s]' % (i+1, ','.join(literal(x) for x in p))
                        for i, p in enumerate(momenta)]
            program += ['original=p', 'shared_real_point=p', 'shared_real_epoch=shared_real_epoch+1',
                        'shared_real_active=.true.', 'call smatrix_real(p,w)',
                        'if(w.le.0d0)stop 10', 'saved=amp_split',
                        # Compare exact bit patterns: no event-vector editing.
                        'if(any(transfer(p,[0_8],size(p)).ne.transfer(original,[0_8],size(p))))stop 11',
                        'if(any(transfer(shared_real_point,[0_8],size(p)).ne.transfer(original,[0_8],size(p))))stop 12',
                        'do i=1,nexternal', 'do j=1,nexternal',
                        'dot0=p(0,i)*p(0,j)-sum(p(1:3,i)*p(1:3,j))',
                        'dot1=last(0,i)*last(0,j)-sum(last(1:3,i)*last(1:3,j))',
                        'scale=max(1d0,abs(p(0,i)*p(0,j))+sum(abs(p(1:3,i)*p(1:3,j))))',
                        'if(abs(dot1-dot0).gt.512d0*epsilon(1d0)*scale)stop 13', 'enddo', 'enddo',
                        'before=calls', 'amp_split=-999d0', 'call smatrix_real(p,ref)',
                        'if(calls.ne.before.or.ref.ne.w.or.any(amp_split.ne.saved))stop 14',
                        'shared_real_active=.false.', 'call smatrix_real_direct(p,raw)',
                        'a(0,:)=p(0,:)', 'a(1,:)=p(3,:)', 'a(2,:)=p(2,:)', 'a(3,:)=-p(1,:)',
                        'b(0,:)=p(0,:)', 'b(1,:)=p(3,:)',
                        'b(2,:)=(p(2,:)-p(1,:))/sqrt(2d0)',
                        'b(3,:)=-(p(1,:)+p(2,:))/sqrt(2d0)',
                        'call smatrix_real_direct(a,ref)', 'call smatrix_real_direct(b,ref2)',
                        'call close(w,ref,maxdiff)', 'call close(w,ref2,maxdiff)',
                        'do i=1,amp_split_size', 'call close(saved(i),amp_split(i),maxdiff)', 'enddo']
            program += ['if(raw.gt.1d-8*ref)stop 15'] if fixture['label'] == 'bad_native' else ['call close(raw,ref,maxdiff)']
        # A genuine zero is preserved; the fix does not replace small MEs.
        program += [name+'=(0d0,0d0)' for name, kind, index in members if name.startswith('gc_')]
        program += ['shared_real_active=.true.', 'call smatrix_real(p,w)', 'if(w.ne.0d0)stop 16',
                    "write(*,*)'PASS stable real basis',maxdiff", 'end',
                    'subroutine close(a,b,maxdiff)', 'use,intrinsic::ieee_arithmetic', 'implicit none',
                    'real(8) a,b,maxdiff,delta', 'if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b))stop 20',
                    'delta=abs(a-b)/max(1d-99,abs(a),abs(b))', 'maxdiff=max(maxdiff,delta)',
                    "if(delta.gt.2d-10)then", "write(*,*)'Mismatch',a,b,delta", 'stop 21', 'endif', 'end']
        support.write_fortran(work/'check.f', program)
        paths = [ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90', work/'mc_native_context.f90',
                 work/'capture.f', sub/'real_me_chooser.f', work/'direct_chooser.f', *matrices,
                 work/'wrappers.f', work/'helpers.f', work/'check.f',
                 *sorted((output/'Source/DHELAS').glob('*.f')), output/'Source/MODEL/model_functions.f']
        self.run_command(['gfortran', '-O0', '-g', '-fcheck=all', '-ffixed-line-length-none',
                          '-ffree-line-length-none', '-I'+str(sub), '-I'+str(output/'lib'),
                          *paths, '-o', work/'check'], work)
        self.assertIn('PASS stable real basis', self.run_command([work/'check'], work))

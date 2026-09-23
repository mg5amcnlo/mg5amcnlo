"""Registry invariants and compiled equivalence with the standalone Born code."""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.iolibs import born_support as support
from madgraph.interface.common_run_interface import CommonRunCmd
from tests.unit_tests.fks.test_momentum_maps import fortran_routine


def physical(ids, tags=None):
    tags = tags or [False]*len(ids)
    return ('restrictions', tuple((pdg, i >= 2, (), tag, None)
                                 for i,(pdg,tag) in enumerate(zip(ids,tags))))


def context(number, processes, i, j, allowed, orders=('QCD','QED')):
    return dict(directory='P%d' % number, provider=number, context=number,
                nexternal=len(processes[0][1]), nincoming=2, orders=orders,
                amp_orders=[(2,2)], sectors=[dict(sector=1,
                    processes=processes, allowed=allowed,
                    fks=dict(i=i, j=j, splitting_type=['QCD'], extra_cnt_index=-1))])


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestStandaloneSudakov(unittest.TestCase):
    def test_equivalent_subprocesses_keep_standalone_born_sources(self):
        from madgraph.interface.master_interface import MasterCmd
        with tempfile.TemporaryDirectory(prefix='mg5_sudakov_export_') as tmp:
            output = Path(tmp)/'output'
            interface = MasterCmd()
            for command in (
                    'set automatic_html_opening False --no_save',
                    'set low_mem_multicore_nlo_generation False --no_save',
                    'import model sm',
                    'generate u u~ > t t~ QED^2=0 QCD^2=4 [LOonly=QCD] --ewsudakov',
                    'add process u~ u > t t~ QED^2=0 QCD^2=4 [LOonly=QCD] --ewsudakov',
                    'output ewsudakovsa %s -f -nojpeg' % output):
                interface.exec_cmd(command, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            subprocesses = sorted((output/'SubProcesses').glob('P*'))
            self.assertEqual(len(subprocesses), 2)
            for sub in subprocesses:
                with self.subTest(subprocess=sub.name):
                    self.assertTrue((sub/'born.f').is_file())
                    # This standalone export has no native FKS tables. Both
                    # Born sources and Sudakov object dependencies must work
                    # without FKS runtime initialization or an f2py install.
                    result = subprocess.run(
                        ['make', 'born.o', 'ewsudakov_functions.o'], cwd=sub,
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)


class TestBornRegistry(unittest.TestCase):
    def test_external_width_particles_from_born_metadata(self):
        """Width removal must survive the replacement of static Born tables."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_pids_') as tmp:
            root = Path(tmp)
            sub = root/'SubProcesses'/'P0_epem_ttx'
            sub.mkdir(parents=True)
            (root/'SubProcesses'/'subproc.mg').write_text(sub.name+'\n')
            table = '      DATA (IDUP(I,1),I=1,4)/ -11, 11, 6, -6 /\n'
            (sub/'born_leshouche.inc').write_text(table)
            run = mock.Mock(me_dir=tmp)
            expected = {'-11', '11', '6', '-6'}
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run), expected)
            (sub/'born_leshouche.inc').write_text(
                '      common/mc_native_born_lh/idup,mothup,icolup\n')
            (sub/'born_support.json').write_text(json.dumps(
                dict(context=1, born_pdgs=[[-11,11,6,-6],[-13,13,6,-6]])))
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run),
                             expected | {'-13', '13'})
            # Outputs generated before born_pdgs was recorded remain usable.
            metadata = root/'Source'/'BornSupport'/'c1'
            metadata.mkdir(parents=True)
            (metadata/'metadata.f').write_text(table)
            (sub/'born_support.json').write_text(json.dumps(dict(context=1)))
            self.assertEqual(CommonRunCmd.get_pid_final_initial_states(run), expected)

    def test_foreign_flavours_and_group_order(self):
        u, d = physical((2,21,23,2)), physical((1,21,23,1))
        a = context(1,[u,d],4,1,[(4,1),(4,2)])
        b = context(2,[d,u],4,2,[(4,1),(4,2)],orders=('QED','QCD'))
        histories = support.resolve_histories([a,b])['P1'][0]
        self.assertEqual(histories['unresolved'], [])
        self.assertEqual(len(histories['histories']), 2)
        foreign = histories['histories'][1]
        self.assertEqual(foreign['flavours'], [2,1])
        self.assertEqual(foreign['orders'], [2,1])

    def test_labelled_identical_orientations(self):
        a = context(1,[physical((2,-2,21,21,21))],5,3,
                    [(i,j) for i in (3,4,5) for j in (3,4,5) if i != j])
        rows = support.resolve_histories([a])['P1'][0]['histories']
        self.assertEqual({(r['i'],r['j']) for r in rows},
                         {(i,j) for i in (3,4,5) for j in (3,4,5) if i != j})
        self.assertEqual(len(rows),6)

    def test_incoming_order_and_tags_preserved(self):
        native = physical((2,21,22,22),[False,False,False,True])
        outer = physical((2,21,22,22),[False,False,True,False])
        self.assertEqual(list(support.labelled_maps(native,outer,2)), [(1,2,4,3)])
        self.assertEqual(list(support.labelled_maps(native,
            physical((21,2,22,22)),2)), [])

    def test_identical_spectators_do_not_expand_factorially(self):
        physical_state = physical((2,-2)+(21,)*10)
        pairs = {(i,j) for i in range(3,13) for j in range(3,13) if i != j}
        a = context(1,[physical_state],12,11,sorted(pairs))
        rows = support.resolve_histories([a])['P1'][0]['histories']
        self.assertEqual(len(rows),90)
        self.assertEqual({(r['i'],r['j']) for r in rows},pairs)
        for row in rows:
            self.assertEqual(sorted(row['permutation']),list(range(1,13)))

    def test_unique_ownership(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1)])
        b = copy.deepcopy(a)
        b.update(context=2,provider=2,directory='P2')
        with self.assertRaisesRegex(MadGraph5Error,'Ambiguous global MC history'):
            support.resolve_histories([a,b])

    def test_missing_is_explicit(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1),(4,2)])
        self.assertEqual(support.resolve_histories([a])['P1'][0]['unresolved'], [(1,4,2)])

    def test_worker_order_does_not_change_registry(self):
        a = context(1,[physical((2,21,23,2))],4,1,[(4,1),(4,2)])
        b = context(2,[physical((2,21,23,2))],4,2,[(4,1),(4,2)])
        self.assertEqual(support.resolve_histories([a,b]),support.resolve_histories([b,a]))

    def test_native_provenance_round_trip(self):
        from madgraph.various.lhe_parser import OneNLOWeight
        # Five legs, native incoming flavours, coupling powers, scales,
        # momentum references, contribution type and native FKS identity.
        line = ('1 0 0 2 3 5 21 2 -11 12 1 404 2 0.1 0.2 '
                '100 100 100 1.2 1 2 13 123 5 1 2 4 1')
        legacy = OneNLOWeight(line)
        self.assertIsNone(legacy.native_provenance)
        native = OneNLOWeight(line+' 3 7 105 2')
        self.assertEqual(native.native_provenance,(3,7,105,2))
        restored = OneNLOWeight(native.__str__(mode='formatted'))
        self.assertEqual(restored.native_provenance,native.native_provenance)
        self.assertEqual(restored.pdgs,native.pdgs)
        self.assertEqual(restored.nfks,123)

    def test_namespacing_preserves_components_and_strings(self):
        text = "call born(p,result%born) ! 'born'"
        self.assertEqual(support.rename(text,{'born':'mcb1_born'}),
                         "call mcb1_born(p,result%born) ! 'born'")

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_extra_helicity_filter_tracks_sector_and_model(self):
        # The extra-counterterm template has one NTRY/SKIP, unlike the main
        # Born's per-sector arrays. Its mother may change between sectors.
        routine = ['subroutine extra_filter(skip_out,tries)', 'implicit none',
            'integer skip_out,tries,ntry,nfksprocess,skip,glu_ij',
            'integer nhel(2,4),ij_values(2)',
            'common/c_nfksprocess/nfksprocess', 'logical goodhel(4)',
            'save ntry,skip,goodhel', 'data ntry/0/',
            'data nhel/-1,-1,1,-1,-1,1,1,1/', 'data ij_values/1,2/',
            'glu_ij=ij_values(nfksprocess)', 'NTRY=NTRY+1',
            'if(ntry.lt.2)then', 'skip=1',
            'do while(nhel(glu_ij,skip).ne.1)', 'skip=skip+1', 'enddo',
            'skip=skip-1', 'endif', 'skip_out=skip', 'tries=ntry', 'end']
        program = ['program check', 'implicit none',
            'integer nfksprocess,born_model_epoch,skip,tries',
            'common/c_nfksprocess/nfksprocess',
            'common/c_born_model_epoch/born_model_epoch',
            'born_model_epoch=1', 'nfksprocess=1', 'call extra_filter(skip,tries)',
            'if(skip.ne.1.or.tries.ne.1)stop 1', 'call extra_filter(skip,tries)',
            'if(skip.ne.1.or.tries.ne.2)stop 2',
            'nfksprocess=2', 'call extra_filter(skip,tries)',
            'if(skip.ne.2.or.tries.ne.1)stop 3',
            'born_model_epoch=2', 'call extra_filter(skip,tries)',
            'if(skip.ne.2.or.tries.ne.1)stop 4', 'end']
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'check.f'
            support.write_fortran(source,program+support.model_aware_helicity_filter(routine))
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                                         'check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_polarized_real_cache_preserves_caller_frame(self):
        from madgraph.iolibs import native_histories
        identity = physical((21,21,6))
        restricted = copy.deepcopy(identity)
        restricted = (restricted[0],restricted[1][:-1]+((6,True,(1,),False,None),))
        record = dict(context=1,sectors=[dict(real=1,processes=[restricted]),
                                        dict(real=2,processes=[identity])])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support.write_fortran(root/'nexternal.inc', ['integer,parameter::nexternal=3'])
            support.write_fortran(root/'orders.inc', ['integer,parameter::amp_split_size=1',
                'real(8) amp_split(1)', 'common/to_amp_split/amp_split'])
            native_histories.write_dispatchers(root,record,{1:record},[(1,1),(1,2)],{})
            (root/'mc_native_context.f90').write_text('''module mc_native_context
implicit none
logical,save::shared_real_active=.true.
integer,save::shared_real_epoch=1,active_history=0
real(8),save::shared_real_point(0:3,3)
integer,save::history_permutations(3,1)=0
end module
''')
            program = ['program check', 'use mc_native_context', 'implicit none',
                'integer nfksprocess,calls', 'common/c_nfksprocess/nfksprocess',
                'common/test_calls/calls', 'real(8) p(0:3,3),wgt',
                'calls=0', 'shared_real_point=1d0', 'p=2d0', 'nfksprocess=1',
                'call smatrix_real(p,wgt)', 'if(wgt.ne.4d0.or.calls.ne.1)stop 1',
                'p(0,1)=3d0', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.5d0.or.calls.ne.2)stop 2', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.5d0.or.calls.ne.2)stop 3', 'nfksprocess=2',
                'call smatrix_real(p,wgt)', 'if(wgt.ne.2d0.or.calls.ne.3)stop 4',
                'p(0,1)=4d0', 'call smatrix_real(p,wgt)',
                'if(wgt.ne.2d0.or.calls.ne.3)stop 5', 'end',
                'subroutine mc_capture_model_state(state)', 'use mc_born_types',
                'implicit none', 'type(BornModelState) state',
                'state%real_values=[1d0]', 'state%complex_values=[(1d0,0d0)]', 'end']
            for number in (1,2):
                program += ['subroutine smatrix%d(p,wgt)' % number, 'implicit none',
                    "include 'orders.inc'", 'real(8) p(0:3,3),wgt,wgt_me_born,wgt_me_real',
                    'integer calls', 'common/test_calls/calls',
                    'common/c_wgt_me_tree/wgt_me_born,wgt_me_real',
                    'calls=calls+1', 'wgt=p(0,1)+p(0,3)',
                    'amp_split=wgt', 'wgt_me_real=wgt', 'end']
            support.write_fortran(root/'check.f',program)
            types = Path(support.__file__).resolve().parents[2]/'Template/NLO/Source/BornSupport/mc_born_types.f90'
            TestBornLibrary.run_command(['gfortran','-fcheck=all','-ffixed-line-length-none',
                '-ffree-line-length-none',str(types),'mc_native_context.f90',
                'real_me_chooser.f','check.f','-o','check'],tmp)
            TestBornLibrary.run_command(['./check'],tmp)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornLibrary(unittest.TestCase):
    """Use actual exports and independently compiled pre-wrapper evaluators."""

    origin = '@loader_path' if sys.platform == 'darwin' else '$ORIGIN'

    @classmethod
    def setUpClass(cls):
        from madgraph.interface import master_interface
        cls.temp = tempfile.TemporaryDirectory(prefix='mg5_born_support_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.outputs = {}
        cls.cache_metrics = {}
        interface = master_interface.MasterCmd()
        def command(text):
            interface.exec_cmd(text,errorhandling=False,printcmd=False,
                               precmd=True,postcmd=True)
        command('set automatic_html_opening False --no_save')
        command('set low_mem_multicore_nlo_generation False --no_save')
        command('import model sm')
        command('define p = g u u~ d d~')
        command('define jj = g a')
        original_finalize = support.finalize

        def capture(exporter):
            root = Path(exporter.dir_path)
            reference = root/'reference'
            reference.mkdir()
            for sub in (root/'SubProcesses').glob('P*'):
                if not sub.is_dir():
                    continue
                target = reference/sub.name
                target.mkdir()
                for pattern in ('*.inc','born.f','born_hel.f','sborn_sf.f',
                                'b_sf_*.f','born_cnt_*.f','extra_cnt_wrapper.f',
                                'splitorders_stuff.f','ewsudakov_functions*.f'):
                    for source in sub.glob(pattern):
                        if source.exists():
                            shutil.copyfile(source,target/source.name)
            # A reused provider deliberately has no second standalone Born
            # implementation. Test its context against the representative's
            # original evaluator while retaining its own metadata includes.
            for name, entry in exporter.born_support_registry.items():
                if name == entry['source']:
                    continue
                for source in (reference/entry['source']).glob('*.f'):
                    target = reference/name/source.name
                    if not target.exists():
                        shutil.copyfile(source,target)
            return original_finalize(exporter)

        for name, process in (
                ('dy','p p > e+ e- QED^2=4 QCD^2=0 [real=QCD]'),
                ('ttbar','p p > t t~ QED^2=0 QCD^2=4 [real=QCD]'),
                ('wjet','p p > w+ j QED^2=2 QCD^2=2 [real=QCD]'),
                ('extra','u u~ > jj jj QED^2=4 QCD^2=4 [real=QCD]'),
                ('qed','e+ e- > mu+ mu- QED^2=4 QCD^2=0 [real=QED]'),
                ('loonly','u d~ > w+ g QED^2=2 QCD^2=2 [LOonly=QCD]'),
                ('shared',('u u~ > e+ e- QED^2=4 QCD^2=0 [real=QCD] @1',
                           'u u~ > e+ e- QED^2=4 QCD^2=0 [real=QCD] @2'))):
            processes = (process,) if isinstance(process,str) else process
            command('generate '+processes[0])
            for additional in processes[1:]:
                command('add process '+additional)
            output = cls.work/name
            with mock.patch.object(support,'finalize',capture):
                command('output %s -f -nojpeg' % output)
            cls.instrument_provider_calls(output)
            cls.run_command(['make','-j2','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],
                            output/'Source/BornSupport')
            cls.outputs[name] = output

    @staticmethod
    def instrument_provider_calls(output):
        """Count actual HELAS calls in test exports, without production counters."""
        directory = output/'Source/BornSupport'
        providers = sorted(directory.glob('p[0-9]*/matrix.f'))
        types = directory/'mc_born_types.f90'
        types.write_text(types.read_text().replace('  implicit none',
            '  implicit none\n  integer(8),save :: born_test_helas_calls(%d)=0' % len(providers), 1))
        for path in providers:
            provider = int(path.parent.name[1:])
            lines = []
            for line in support.statements(path.read_text()):
                if re.match(r'(?i)\s*call\s+(?!mcb\d+_)\w+\s*\(', line):
                    lines.append('born_test_helas_calls(%d)=born_test_helas_calls(%d)+1' %
                                 (provider,provider))
                lines.append(line)
                if re.match(r'(?i)\s*(?:[\w*]+\s+)*?(?:subroutine|function)\s+\w+', line):
                    lines.append('use mc_born_types,only:born_test_helas_calls')
            support.write_fortran(path,lines)

    @staticmethod
    def run_command(command, cwd):
        result = subprocess.run(command,cwd=cwd,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if result.returncode:
            raise AssertionError(' '.join(map(str,command))+'\n'+result.stdout)
        return result.stdout

    def compare(self, output):
        registry = json.loads((output/'Source/BornSupport/registry.json').read_text())
        model_lines = support.expand(output/'Source/MODEL/coupl.inc')
        model_lines += support.expand(output/'Source/MODEL/input.inc')
        members,sizes = support.common_members(model_lines)
        signature = json.loads((output/'Source/BornSupport/helicity_signature.json').read_text())
        self.assertEqual(signature['mode'], 'homogeneous_G')
        for record in registry['contexts']:
            with self.subTest(process=output.name, provider=record['provider']):
                source = output/'reference'/record['directory']
                # Extract only pure order helpers; their other routines depend
                # on event bookkeeping which is not part of this test.
                helpers = (source/'splitorders_stuff.f').read_text()
                helpers += '\n' + next(source.glob('ewsudakov_functions*.f')).read_text()
                support.write_fortran(source/'helpers.f',sum((support.routine(helpers,n)
                    for n in ('orders_to_amp_split_pos','amp_split_pos_to_orders',
                              'orders_equal','get_lo2_orders')),[]))
                helicity_data = [line for line in support.statements((source/'born.f').read_text())
                                 if re.match(r'(?i)data\s*\(nhel\(', line)]
                support.write_fortran(source/'check.f',self.driver(record,members,sizes,model_lines,
                    registry['contexts'], helicity_data, signature['signature']))
                paths = [source/n for n in ('born.f','born_hel.f','sborn_sf.f',
                                           'extra_cnt_wrapper.f','helpers.f','check.f')]
                paths += sorted(source.glob('b_sf_*.f'))+sorted(source.glob('born_cnt_*.f'))
                # HELAS is independent here; only the provider implementation
                # is linked dynamically, with hidden internal symbols.
                paths += sorted((output/'Source/DHELAS').glob('*.f'))
                paths += [output/'Source/MODEL/model_functions.f']
                self.run_command(['gfortran','-O0','-g','-fcheck=all','-ffixed-line-length-none',
                    '-I'+str(source),'-I'+str(output/'lib'),*map(str,paths),
                    '-L'+str(output/'lib'),'-lmc_born_support',
                    '-Wl,-rpath,'+self.origin+'/../../lib','-Wl,-rpath,'+self.origin,
                    '-o','check'],source)
                checked = self.run_command(['./check'],source)
                self.assertIn('PASS Born provider',checked)
                discovery,pruned = map(int,re.search(
                    r'HELAS discovery/pruned\s+(\d+)\s+(\d+)',checked).groups())
                self.cache_metrics[output.name,record['provider']] = (discovery,pruned)
                if output.name == 'dy':
                    self.assertLess(pruned,discovery,
                        'fixed-model helicity filtering must eliminate zero-helicity HELAS calls')
                # A worker only needs the executable and the DSO beside it.
                scratch = output/('scratch_'+str(record['context']))
                scratch.mkdir()
                shutil.copy2(source/'check',scratch/'check')
                for lib in (output/'lib').glob('libmc_born_support.*'):
                    shutil.copy2(lib,scratch/lib.name)
                self.assertIn('PASS Born provider',self.run_command(['./check'],scratch))

    @staticmethod
    def driver(r, members, sizes, model_lines, contexts, helicity_data=(), signature=None):
        n,ng,nc,nh,ns = r['nexternal']-1,r['ngraphs'],r['ncolor'],r['nhelicity'],r['nsqamps']
        lines = ['program check', 'use mc_born_support', 'use mc_born_types', 'implicit none']
        lines += ["include 'orders.inc'"]+model_lines+[
            'type(BornModelState) state,alternate', 'type(BornRequest) request',
            'type(BornResult) result,other', 'type(BornMetadata) metadata',
            'integer nfksprocess,status,sector,iteration,i,j,k,ihel',
            'integer(8) calls_before,calls_first,calls_pruned',
            'integer nhel(%d,%d)' % (n,nh),
            *helicity_data, 'common/c_nfksprocess/nfksprocess',
            'complex(8) ewsud(amp_split_size),ewsud_lo2(amp_split_size)',
            'common/to_amp_split_ewsud/ewsud',
            'common/to_amp_split_ewsud_lo2/ewsud_lo2',
            'logical calculatedBorn', 'common/ccalculatedBorn/calculatedBorn',
            'logical need_color_links,need_charge_links', 'common/c_need_links/need_color_links,need_charge_links',
            'logical split_type_used(nsplitorders)', 'common/to_split_type_used/split_type_used',
            'real(8) p(0:3,%d),pother(0:3,%d),ans,corr,mass,energy,magnitude' % (n,n),
            'real(8) amp2(%d),jamp2(0:%d)' % (ng,nc), 'common/to_amps/amp2,jamp2',
            'real(8) amp2b(%d),jamp2b(0:%d,0:%d)' % (ng,nc,r['namps']),
            'common/to_amps_born/amp2b,jamp2b',
            'complex(8) ans_cnt(2,nsplitorders),extra(2,nsplitorders)', 'common/c_born_cnt/ans_cnt',
            'real(8) hel(%d),helorders(%d,%d)' % (nh,ns,nh),
            'common/c_born_hel/hel', 'common/c_born_hel_split/helorders',
            'real(8) soft(amp_split_size)', 'common/to_amp_split_soft/soft',
            'real(8) charges(%d)' % n, 'common/c_charges_born/charges',
            'allocate(state%%real_values(%d),state%%complex_values(%d))' % (sizes['real'],sizes['complex']),
            'allocate(request%%charges(%d))' % n,
            'state%real_values=1.1d0', 'state%complex_values=(0.7d0,0.2d0)',
            'split_type_used=.true.', 'charges=0.5d0', 'request%charges=charges',
            'request%helicities=.true.',
            'call born_query(%d,%d,metadata,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 10',
            'if(size(metadata%%born_ids,1).ne.%d)stop 14' % n,
            'if(size(metadata%%born_ids,2).ne.%d)stop 18' % r['nprocesses'],
            'if(any(metadata%sprop.ne.0.and.metadata%tprid.ne.0))stop 19',
            'if(lbound(metadata%configurations,1).ne.0)stop 24',
            'if(lbound(metadata%tree,2).ne.-size(metadata%tree,2))stop 25',
            'call born_query(%d,%d,metadata,status,require_complete=.true.)' % (r['provider'],r['context']),
            'if(all(metadata%complete_histories))then',
            'if(status.ne.0)stop 15', 'else', 'if(status.ne.5)stop 16', 'endif',
            'do iteration=1,3',
            'state%real_values=state%real_values*1.03d0',
            'state%complex_values=state%complex_values*(0.9d0,0.1d0)']
        lines += ['%s=state%%%s_values(%s)' % v for v in members]
        lines += ['mass=state%real_values(1)', 'energy=500d0',
            'magnitude=sqrt(energy**2-mass**2)', 'p=0d0',
            'p(:,1)=[energy,0d0,0d0,energy]', 'p(:,2)=[energy,0d0,0d0,-energy]',
            'p(:,3)=[energy,magnitude*0.6d0,0d0,magnitude*0.8d0]',
            'p(:,4)=[energy,-magnitude*0.6d0,0d0,-magnitude*0.8d0]',
            # Rotate transverse momenta while leaving the old cache's E,pz key
            # unchanged. This was not caught by the legacy savemom check.
            'if(mod(iteration,2).eq.0)then', 'p(2,:)=p(1,:)', 'p(1,:)=0d0','endif',
            'do sector=1,%d' % max(1,len(r['sectors'])),
            'nfksprocess=sector', 'request%sector=sector',
            'request%colour=.false.', 'request%charge=.false.', 'request%extra=0',
            'ans_cnt=(0d0,0d0)', 'amp_split_cnt=(0d0,0d0)',
            'calculatedBorn=.false.', 'call sborn(p,ans)',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 11',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 29',
            'if(born_test_helas_calls(%d).ne.calls_before)stop 30' % r['provider'],
            'call close_real(result%born,other%born)',
            'call close_real(ans,result%born)',
            'do i=1,amp_split_size', 'call close_real(amp_split(i),result%amplitudes(i))','enddo',
            'do i=1,%d' % ng, 'call close_real(amp2(i),result%diagrams(i))','enddo',
            'do i=1,%d' % nc, 'call close_real(jamp2(i),result%flows(i))','enddo',
            'do i=1,%d' % nc, 'do j=1,%d' % r['namps'],
            'call close_real(jamp2b(i,j),result%flow_orders(i,j))','enddo','enddo',
            'do j=1,nsplitorders','do k=1,2',
            'call close_complex(ans_cnt(k,j),result%counterterms(k,j))',
            'do i=1,amp_split_size',
            'call close_complex(amp_split_cnt(i,k,j),result%split_counterterms(i,k,j))',
            'enddo','enddo','enddo',
            'call sborn_hel(p,ans)',
            'do i=1,%d' % nh, 'call close_real(hel(i),result%helicities(i))','enddo',
            'do i=1,%d' % nh,'do j=1,%d' % ns,
            'call close_real(helorders(j,i),result%helicity_orders(j,i))','enddo','enddo',
            'do ihel=1,%d' % nh,
            'call sborn_onehel(p,nhel(:,ihel),ihel,ans)',
            'request%single_helicity=ihel', 'request%helicity=nhel(:,ihel)',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 28', 'call close_real(ans,other%single_helicity)',
            'do i=1,amp_split_size',
            'call close_complex(ewsud(i),other%ewsudakov(i))',
            'call close_complex(ewsud_lo2(i),other%ewsudakov_lo2(i))',
            'enddo', 'enddo', 'request%single_helicity=0',
            'pother=p*1.13d0']
        for other in contexts:
            lines += ['request%sector=1',
                'call born_evaluate(%d,%d,pother,state,request,other,status)' % (other['provider'],other['context']),
                'if(status.ne.0)stop 12']
        lines += ['request%sector=sector',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'call close_real(result%born,other%born)']
        for m,n_ in r['links'] if r['sectors'] else []:
            lines += ['calculatedBorn=.false.', 'call sborn(p,ans)',
                'need_color_links=.true.', 'need_charge_links=.false.',
                'call sborn_sf(p,%d,%d,corr)' % (m,n_),
                'request%colour=.true.', 'request%m='+str(m), 'request%n='+str(n_),
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 13', 'call close_real(corr,other%correlation)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 31' % r['provider'],
                'do i=1,amp_split_size','call close_real(soft(i),other%soft(i))','enddo']
        # QED uses the Born charge correlator even for a colourless process.
        # Interleave it with colour/Born calls to check the request flags and
        # private charge state, and compare all native split orders.
        if any('QED' in sector['fks']['splitting_type'] for sector in r['sectors']):
            lines += ['request%colour=.false.', 'request%charge=.true.',
                'need_color_links=.false.', 'need_charge_links=.true.',
                'do j=1,%d' % (n-1), 'do k=j+1,%d' % n,
                'calculatedBorn=.false.', 'call sborn(p,ans)',
                'call sborn_sf(p,j,k,corr)', 'request%m=j', 'request%n=k',
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 26', 'call close_real(corr,other%correlation)',
                'do i=1,amp_split_size','call close_real(soft(i),other%soft(i))','enddo',
                'charges=0.7d0', 'request%charges=charges',
                'call sborn_sf(p,j,k,corr)',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 40', 'call close_real(corr,other%correlation)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 41' % r['provider'],
                'charges=0.5d0', 'request%charges=charges',
                'enddo','enddo','request%charge=.false.', 'need_charge_links=.false.']
        if not r['sectors']:
            lines += ['request%charge=.true.', 'request%m=1', 'request%n=2',
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.BORN_MISSING_CORRELATION)stop 27', 'request%charge=.false.']
        for sector in r['sectors']:
            icnt = sector['fks']['extra_cnt_index']+1
            if icnt:
                lines += ['if(sector.eq.%d)then' % sector['sector'],
                    'request%colour=.false.', 'request%%extra=%d' % icnt,
                    'call extra_cnt(p,%d,extra)' % icnt,
                    'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                    'if(status.ne.0)stop 17',
                    'do j=1,nsplitorders','do k=1,2',
                    'call close_complex(extra(k,j),other%extra(k,j))',
                    'do i=1,amp_split_size',
                    'call close_complex(amp_split_cnt(i,k,j),other%split_counterterms(i,k,j))',
                    'enddo','enddo','enddo','endif']
        lines += ['enddo','enddo',
            'request%sector=1', 'request%colour=.false.', 'request%charge=.false.',
            'request%extra=0', 'request%helicities=.false.',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context']),
            'if(result%has_soft.or.result%has_helicities.or.result%has_extra.or.result%has_single_helicity)stop 42',
            'alternate=state', 'state%complex_values=(0d0,0d0)',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 32', 'call close_real(other%born,0d0)',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'state=alternate',
            'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 33', 'call close_real(result%born,other%born)',
            'if(born_test_helas_calls(%d).le.calls_before)stop 34' % r['provider'],
            'calls_first=born_test_helas_calls(%d)-calls_before' % r['provider'],
            # Transverse components alone must invalidate SAVEAMP; keep model,
            # E and pz identical, then compare with the standalone calculation.
            'pother=p', 'pother(1,:)=p(2,:)', 'pother(2,:)=p(1,:)',
            'calls_before=born_test_helas_calls(%d)' % r['provider'],
            'call born_evaluate(%d,%d,pother,state,request,other,status)' % (r['provider'],r['context']),
            'if(status.ne.0)stop 35',
            'calls_pruned=born_test_helas_calls(%d)-calls_before' % r['provider'],
            'if(calls_pruned.le.0.or.calls_pruned.gt.calls_first)stop 36',
            'nfksprocess=1', 'calculatedBorn=.false.', 'call sborn(pother,ans)',
            'call close_real(ans,other%born)',
            'call born_evaluate(%d,%d,p,state,request,result,status)' % (r['provider'],r['context'])]
        for other in contexts:
            if other['provider'] == r['provider']:
                continue
            lines += [
                'alternate%real_values=state%real_values*1.19d0',
                'alternate%complex_values=state%complex_values*(0.4d0,0.3d0)',
                'call born_evaluate(%d,%d,pother,alternate,request,other,status)' % (other['provider'],other['context']),
                'if(status.ne.0)stop 37',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,p,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 38', 'call close_real(result%born,other%born)',
                'if(born_test_helas_calls(%d).ne.calls_before)stop 39' % r['provider']]
        if signature:
            lines += ['alternate=state',
                'state%%real_values(%d)=2d0*state%%real_values(%d)' % (signature['g'],signature['g'])]
            for kind, index, degree in signature['couplings']:
                lines += ['state%%%s_values(%d)=state%%%s_values(%d)*2d0**(%d)' %
                          (kind,index,kind,index,degree)]
            lines += ['if(.not.born_helicity_state_equal(state,alternate))stop 43',
                'if(born_model_state_equal(state,alternate))stop 44',
                'calls_before=born_test_helas_calls(%d)' % r['provider'],
                'call born_evaluate(%d,%d,pother,state,request,other,status)' % (r['provider'],r['context']),
                'if(status.ne.0)stop 45',
                'if(born_test_helas_calls(%d)-calls_before.ne.calls_pruned)stop 46' % r['provider']]
            lines += ['%s=state%%%s_values(%s)' % v for v in members]
            lines += ['calculatedBorn=.false.', 'nfksprocess=1', 'call sborn(pother,ans)',
                      'call close_real(ans,other%born)']
        lines += ["write(*,*)'PASS Born provider; HELAS discovery/pruned',calls_first,calls_pruned",'end',
            'subroutine close_complex(a,b)','implicit none','complex(8) a,b',
            'call close_real(real(a,8),real(b,8))','call close_real(aimag(a),aimag(b))','end',
            'subroutine close_real(a,b)','use, intrinsic :: ieee_arithmetic','implicit none','real(8) a,b',
            'if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b))stop 21',
            'if(abs(a-b).gt.1d-12*max(1d-50,abs(a),abs(b)))then',
            "write(*,*)'Mismatch',a,b",'stop 22','endif','end',
            'double precision function ran2()', 'stop 23', 'end']
        return lines

    def test_dy(self):
        self.compare(self.outputs['dy'])

    def test_ttbar(self):
        self.compare(self.outputs['ttbar'])

    def test_wjet(self):
        self.compare(self.outputs['wjet'])

    def test_extra_born(self):
        self.compare(self.outputs['extra'])

    def test_qed_charge_correlations(self):
        self.compare(self.outputs['qed'])

    def test_loonly(self):
        self.compare(self.outputs['loonly'])

    def test_real_history_cache(self):
        from tests.unit_tests.iolibs.native_cache_checks import check_real_cache
        for name in ('dy','ttbar','wjet'):
            with self.subTest(process=name):
                check_real_cache(self.outputs[name],self.run_command)

    def test_native_context_switching(self):
        """Bounds-checked native aliases reproduce explicit provider requests."""
        for name in ('dy','ttbar','wjet','loonly'):
            output = self.outputs[name]
            records = json.loads((output/'Source/BornSupport/registry.json').read_text())['contexts']
            model = support.expand(output/'Source/MODEL/coupl.inc')
            model += support.expand(output/'Source/MODEL/input.inc')
            members, sizes = support.common_members(model)
            for record in records:
                directory = output/'SubProcesses'/record['directory']
                lines = ['program check_native','use mc_native_context','implicit none',
                    "include 'nexternal.inc'", "include 'genps.inc'", "include 'orders.inc'",
                    "include 'born_nhel.inc'", "include 'born_conf.inc'",
                    'integer idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc)',
                    'integer icolup(2,nexternal-1,max_bcol)',"include 'born_leshouche.inc'",
                    'integer nfksprocess,status,i,j,k,iteration,c',
                    'common/c_nfksprocess/nfksprocess',
                    'real(8) p(0:3,nexternal-1),ans,soft(amp_split_size),mc_born_flow_weight',
                    'real(8) charges(nexternal-1)', 'common/c_charges_born/charges',
                    'common/to_amp_split_soft/soft',
                    'logical need_color_links,need_charge_links',
                    'common/c_need_links/need_color_links,need_charge_links',
                    'logical split_type(nsplitorders),is_leading_cflow(max_bcol)',
                    'integer num_leading_cflows',
                    'common/c_split_type/split_type',
                    'common/c_leading_cflows/is_leading_cflow,num_leading_cflows',
                    'type(BornModelState) state','type(BornRequest) request',
                    'type(BornResult) reference']+model+[
                    'allocate(state%%real_values(%d),state%%complex_values(%d))' %
                        (sizes['real'],sizes['complex']),
                    'state%real_values=1.1d0','state%complex_values=(0.7d0,0.2d0)',
                    'charges=0.5d0','request%charges=charges',
                    'p=0d0','p(:,1)=[500d0,0d0,0d0,500d0]',
                    'p(:,2)=[500d0,0d0,0d0,-500d0]',
                    'p(:,3)=[500d0,300d0,0d0,400d0]',
                    'p(:,4)=[500d0,-300d0,0d0,-400d0]',
                    'do iteration=1,3','state%complex_values=state%complex_values*(0.9d0,0.1d0)',
                    'p(1:2,:)=-p(1:2,:)']
                lines += ['%s=state%%%s_values(%s)' % v for v in members]
                lines += ['do nfksprocess=1,size(native_sector_ids)',
                    'call activate_native_context(nfksprocess)',
                    'c=active_context', 'request%sector=active_sector',
                    'request%colour=.false.', 'need_color_links=.true.',
                    'need_charge_links=.false.',
                    'call sborn_native(p,ans)',
                    'call born_evaluate(context_providers(c),c,p,state,request,reference,status)',
                    'if(status.ne.0)stop 11','call close_real(ans,reference%born)',
                    'if(mapconfig(0).ne.native_metadata%configurations(0))stop 12',
                    'if(any(idup(:,1:iproc_born).ne.native_metadata%born_ids))stop 13',
                    'split_type=.false.', 'is_leading_cflow=.false.',
                    'call set_QCD_flows',
                    'if(num_leading_cflows.le.0)stop 16',
                    'if(.not.any(is_leading_cflow))stop 17',
                    'do i=1,native_metadata%ncolor',
                    'call close_real(mc_born_flow_weight(i),reference%flows(i))','enddo',
                    'do i=1,native_metadata%namplitudes',
                    'j=native_amplitude_map(i,c)','if(j.eq.0)cycle',
                    'call close_real(amp_split(j),reference%amplitudes(i))','enddo',
                    # A correlated call must not overwrite the caller's accumulated
                    # soft amplitude. This is distinct from the provider's cache.
                    'if(%s)then' % ('.true.' if record['sectors'] else '.false.'),
                    'amp_split=123d0','call sborn_sf_native(p,1,2,ans)',
                    'if(any(amp_split.ne.123d0))stop 14',
                    'request%colour=.true.','request%m=1','request%n=2',
                    'call born_evaluate(context_providers(c),c,p,state,request,reference,status)',
                    'if(status.ne.0)stop 15','call close_real(ans,reference%correlation)',
                    'do i=1,native_metadata%namplitudes',
                    'j=native_amplitude_map(i,c)','if(j.eq.0)cycle',
                    'call close_real(soft(j),reference%soft(i))','enddo','endif',
                    'enddo','call activate_native_context(1)','enddo',
                    "write(*,*)'PASS native contexts'",'end']
                checks = self.driver(record,members,sizes,model,records)
                lines += checks[checks.index('subroutine close_complex(a,b)'):]
                support.write_fortran(directory/'check_native.f',lines)
                # Keep pure order helpers only, as in the standalone comparison.
                support.write_fortran(directory/'native_helpers.f',sum((support.routine(
                    (directory/'splitorders_stuff.f').read_text(),n) for n in
                    ('orders_to_amp_split_pos','amp_split_pos_to_orders')),[]))
                counter = Path(support.__file__).resolve().parents[2]/'Template/NLO/SubProcesses/montecarlocounter.f'
                (directory/'native_flows.f').write_text('\n'.join(
                    fortran_routine(counter, name) for name in
                    ('set_QCD_flows', 'check_QCD_flows')))
                paths = ['mc_native_context.f90','mc_native_runtime.f','mc_native_props.f',
                         'born_support.f','born.f','sborn_sf.f','extra_cnt_wrapper.f',
                         'native_helpers.f','native_flows.f','check_native.f']
                self.run_command(['gfortran','-O0','-g','-fcheck=all',
                    '-ffixed-line-length-none','-ffree-line-length-none','-ffunction-sections',
                    '-I'+str(output/'lib'),*paths,
                    '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                    '-L'+str(output/'lib'),'-lmc_born_support','-Wl,-rpath,'+self.origin+'/../../lib',
                    '-o','check_native'],directory)
                self.assertIn('PASS native contexts',self.run_command(['./check_native'],directory))

    def test_shared_provider_retains_both_contexts(self):
        output = self.outputs['shared']
        support_dir = output/'Source/BornSupport'
        records = json.loads((support_dir/'registry.json').read_text())['contexts']
        self.assertEqual(len(records),2)
        self.assertEqual({r['provider'] for r in records},{1})
        self.assertEqual({r['context'] for r in records},{1,2})
        self.assertEqual(len(list(support_dir.glob('p*/matrix.f'))),1)
        self.assertEqual(len(list(support_dir.glob('c*/metadata.f'))),2)
        for r in records:
            wrapper = output/'SubProcesses'/r['directory']/'born.f'
            self.assertNotIn('SBORN_SPLITORDERS',wrapper.read_text().upper())
        self.compare(output)

    def test_parallel_registry(self):
        from madgraph.interface import master_interface
        interface = master_interface.MasterCmd()
        output = self.work/'dy_parallel'
        for command in ('set automatic_html_opening False --no_save',
                        'set low_mem_multicore_nlo_generation True --no_save',
                        'set nb_core 2 --no_save', 'import model sm',
                        'define p = g u u~ d d~',
                        'generate p p > e+ e- QED^2=4 QCD^2=0 [real=QCD]',
                        'output %s -f -nojpeg' % output):
            interface.exec_cmd(command,errorhandling=False,printcmd=False,
                               precmd=True,postcmd=True)
        registry = Path('Source/BornSupport/registry.json')
        self.assertEqual(json.loads((output/registry).read_text()),
                         json.loads((self.outputs['dy']/registry).read_text()))

    def test_incremental_and_hidden_symbols(self):
        output = self.outputs['ttbar']
        directory = output/'Source/BornSupport'
        self.run_command(['make','-q'],directory)
        os.utime(directory/'p1/matrix.f',None)
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('-c p1/matrix.f',commands)
        self.assertNotIn('-c p2/matrix.f',commands)
        self.run_command(['make','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],directory)
        dependency = next((output/'Source/DHELAS').glob('FFV*.f'))
        os.utime(dependency,None)
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('refresh_dependency.py',commands)
        self.assertNotIn('-c p1/matrix.f',commands)
        self.run_command(['make','FFLAGS=-O0 -g -fcheck=all -fbacktrace'],directory)
        self.run_command(['make','-q'],directory)
        (output/'lib/mc_born_support.mod').unlink()
        commands = self.run_command(['make','-n'],directory)
        self.assertIn('cp mc_born_support.mod',commands)
        self.assertNotIn(' -c ',commands)
        self.run_command(['make'],directory)
        self.run_command(['make','-q'],directory)
        if os.name == 'posix' and shutil.which('nm') and (output/'lib/libmc_born_support.so').exists():
            symbols = self.run_command(['nm','-D','--defined-only',str(output/'lib/libmc_born_support.so')],directory)
            for line in symbols.splitlines():
                self.assertTrue(line.split()[-1].startswith(('__mc_born_support_MOD_','__mc_born_types_MOD_')),line)

    def test_z_relocation(self):
        original = self.outputs['ttbar']
        moved = self.work/'relocated'
        original.rename(moved)
        for executable in moved.glob('reference/P*/check'):
            self.assertIn('PASS Born provider',self.run_command(['./check'],executable.parent))

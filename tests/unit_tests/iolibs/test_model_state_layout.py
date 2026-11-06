"""Readsets and lifecycle must follow the generated evaluator and UFO runtime."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from madgraph.core import base_objects
from madgraph import MadGraph5Error
from madgraph.iolibs.model_state import build_model_state_layout


def model(running=False):
    external = [
        base_objects.ParamCardVariable('aS', .118, 'SMINPUTS', [3]),
        base_objects.ParamCardVariable('MU_R', 91., 'LOOP', [1]),
        base_objects.ParamCardVariable('aEWM1', 128., 'SMINPUTS', [1]),
        base_objects.ParamCardVariable('mdl_mass', 173., 'MASS', [6]),
        base_objects.ParamCardVariable('mdl_width', 1.5, 'DECAY', [6]),
        base_objects.ParamCardVariable('mdl_unused', 7., 'EXTRA', [1]),
    ]
    parameters = {('external',): external,
                  ('aS',): [base_objects.ModelVariable('G', '2*cmath.sqrt(aS*cmath.pi)',
                                                       'real', ('aS',))],
                  (): [base_objects.ModelVariable('mdl_constant', '2', 'real')]}
    couplings = {
        ('aEWM1',): [base_objects.ModelVariable('GC_WEAK', '1/aEWM1', 'complex', ('aEWM1',))],
        ('aS',): [base_objects.ModelVariable('GC_S', 'G', 'complex', ('aS',)),
                  base_objects.ModelVariable('GC_NONHOM', 'G+1', 'complex', ('aS',))],
    }
    elements = []
    if running:
        original = [base_objects.ParamCardVariable('cQQ1', 1., 'DIM6', [1]),
                    base_objects.ParamCardVariable('cQQ8', 2., 'DIM6', [8])]
        external.extend(base_objects.ParamCardVariable('mdl_'+p.name, p.value, p.lhablock,
                                                       p.lhacode) for p in original)
        external.append(base_objects.ParamCardVariable('mdl_muprime', 91., 'LOOP', [2]))
        elements = [SimpleNamespace(run_objects=[original])]
        couplings[('cQQ1',)] = [base_objects.ModelVariable('GC_RUN', 'mdl_cQQ1', 'complex',
                                                         ('cQQ1',))]
    return base_objects.Model(dict(name='state_layout_fixture', parameters=parameters,
                                   couplings=couplings, running_elements=elements))


class TestModelStateLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mg7_model_layout_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root/'SubProcesses/P0_test'
        self.directory.mkdir(parents=True)
        (self.root/'Source/DHELAS').mkdir(parents=True)
        (self.root/'Source/MODEL').mkdir()
        self.members = [
            ('g', 'real', '1'), ('mu_r', 'real', '2'), ('as', 'real', '3'),
            ('mdl_mass', 'real', '4'), ('mdl_width', 'real', '5'),
            ('mdl_unused', 'real', '6'), ('mdl_constant', 'real', '7'),
            ('gc_weak', 'complex', '1'), ('gc_s', 'complex', '2'),
            ('gc_nonhom', 'complex', '3'), ('gal', 'complex', '4:5'),
        ]
        self.write_common()
        self.records = [dict(directory='P0_test')]

    def write(self, path, lines):
        path = Path(path)
        if not path.is_absolute():
            path = self.directory/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join('      '+line+'\n' for line in lines))

    def write_common(self):
        lines = []
        for name, kind, index in self.members:
            size = int(index.split(':')[-1])-int(index.split(':')[0])+1
            declaration = name+('(%d)' % size if ':' in index else '')
            lines += [('double complex ' if kind == 'complex' else 'double precision ')+declaration,
                      'common/model_'+name+'/'+name]
        self.write('coupl.inc', lines)

    def evaluator(self, filename, expression, name=None, extra=()):
        self.write(filename, ['subroutine '+(name or Path(filename).stem)+'(p,ans)',
                              'implicit none', "include 'coupl.inc'",
                              'double precision p(4),ans', *extra,
                              'ans='+expression, 'end'])

    def layout(self, fixture=None):
        return build_model_state_layout(self.root, self.records, fixture or model(), self.members)

    def names(self, layout, part):
        return [member[0] for member in layout[part+'_members']]

    def test_real_only_strong_read_is_not_lost_to_the_born_readset(self):
        self.evaluator('born.f', 'dble(gc_weak)')
        self.evaluator('matrix_1.f', 'g*dble(gc_weak)')
        result = self.layout()
        self.assertEqual(self.names(result, 'fixed'), ['gc_weak'])
        self.assertEqual(self.names(result, 'dynamic'), ['g'])
        self.assertEqual(result['consumer_readsets']['P0_test']['born'], ['gc_weak'])
        self.assertEqual(result['consumer_readsets']['P0_test']['real'], ['g', 'gc_weak'])
        self.assertEqual(result['sizes'], dict(fixed=dict(real=0, complex=1),
                                               dynamic=dict(real=1, complex=0)))

    def test_used_numeric_couplings_are_transferred_without_their_unused_roots(self):
        self.evaluator('born.f', 'dble(gc_weak*gc_s)')
        result = self.layout()
        self.assertEqual(result['fixed_members'], [('gc_weak', 'complex', '1')])
        self.assertEqual(result['dynamic_members'], [('gc_s', 'complex', '1')])
        self.assertNotIn('as', result['manifest']['used'])
        self.assertNotIn('mu_r', result['manifest']['used'])

    def test_nonhomogeneous_coupling_does_not_prevent_pruning(self):
        self.evaluator('born.f', 'dble(gc_nonhom)')
        self.records[0]['helicity_matrices'] = [dict(homogeneous=False)]
        result = self.layout()
        self.assertEqual(result['fixed_members'], [])
        self.assertEqual(result['dynamic_members'], [('gc_nonhom', 'complex', '1')])
        self.assertFalse(result['manifest']['scaling_g'])

    def test_certified_nonzero_scaling_adds_g_but_zero_degree_does_not(self):
        self.evaluator('born.f', 'dble(gc_s)')
        self.records[0]['helicity_matrices'] = [dict(homogeneous=True, amplitude_degrees=[1])]
        self.assertEqual(self.names(self.layout(), 'dynamic'), ['g', 'gc_s'])
        self.records[0]['helicity_matrices'][0]['amplitude_degrees'] = [0]
        self.assertEqual(self.names(self.layout(), 'dynamic'), ['gc_s'])

    def test_mass_width_extra_and_gal_correlation_reads_are_fixed_configuration(self):
        self.evaluator('born.f', 'mdl_mass')
        self.evaluator('born_cnt_1.f', 'mdl_width')
        self.evaluator('sborn_sf.f', 'dble(gal(2))*mdl_constant')
        result = self.layout()
        self.assertEqual(result['fixed_members'], [('mdl_mass', 'real', '1'),
                                                  ('mdl_width', 'real', '2'),
                                                  ('mdl_constant', 'real', '3'),
                                                  ('gal', 'complex', '1:2')])
        self.assertEqual(result['dynamic_members'], [])
        self.assertEqual(result['consumer_readsets']['P0_test']['extra'], ['mdl_width'])
        self.assertEqual(result['consumer_readsets']['P0_test']['correlation'], ['gal', 'mdl_constant'])

    def test_rge_externals_match_lha_identity_despite_the_mdl_prefix(self):
        self.members += [('mdl_cqq1', 'real', '8'), ('mdl_cqq8', 'real', '9'),
                         ('mdl_muprime', 'real', '10'), ('gc_run', 'complex', '6'),
                         ('mdl__dim6__scale', 'real', '11')]
        self.write_common()
        self.evaluator('born.f', 'mdl_cqq1+mdl_cqq8+mdl_muprime+dble(gc_run)+mdl__dim6__scale')
        result = self.layout(model(running=True))
        self.assertEqual(self.names(result, 'dynamic'), ['mdl_cqq1', 'mdl_cqq8', 'mdl_muprime', 'gc_run'])
        self.assertEqual(self.names(result, 'fixed'), ['mdl__dim6__scale'])
        self.assertEqual(result['manifest']['unclassified_dynamic'], [])

    def test_helper_reads_are_transitive_but_declarations_and_dummy_names_are_not_reads(self):
        self.evaluator('born.f', 'ans', extra=('call helper(gc_s,ans)',))
        self.write(self.root/'Source/DHELAS/helper.f', [
            'subroutine helper(g,ans)', 'implicit none', 'double complex g',
            'double precision ans,model_factor', 'external model_factor',
            'ans=dble(g)*model_factor()', 'end'])
        self.write(self.root/'Source/MODEL/factors.f', [
            'double precision function model_factor()', 'implicit none',
            "include 'factor.inc'", 'model_factor=mdl_mass+mdl_width', 'end'])
        self.write(self.root/'Source/MODEL/factor.inc', [
            'double precision mdl_mass,mdl_width,mdl_unused',
            'common/mass_inputs/mdl_mass,mdl_width,mdl_unused'])
        result = self.layout()
        self.assertEqual(self.names(result, 'fixed'), ['mdl_mass', 'mdl_width'])
        self.assertEqual(self.names(result, 'dynamic'), ['gc_s'])
        self.assertEqual(result['manifest']['fallbacks'], [])

    def test_custom_helper_array_is_retained_whole_only_when_read(self):
        self.members += [('hidden_array', 'real', '8:10'), ('unused_array', 'real', '11:14')]
        self.write_common()
        self.evaluator('born.f', 'ans', extra=('call custom_helper(p,ans)',))
        self.write(self.root/'Source/DHELAS/custom.f', [
            'subroutine custom_helper(p,ans)', 'double precision p(4),ans',
            'double precision hidden_array(3),unused_array(4)',
            'common/custom_inputs/hidden_array,unused_array',
            'ans=hidden_array(2)*p(1)', 'end'])
        result = self.layout()
        self.assertEqual(result['dynamic_members'], [('hidden_array', 'real', '1:3')])
        self.assertEqual(result['manifest']['unclassified_dynamic'], ['hidden_array'])
        self.assertNotIn('unused_array', result['manifest']['used'])

    def test_unresolved_hidden_helper_requires_a_source_or_explicit_contract(self):
        self.evaluator('born.f', 'ans', extra=('call helper(p,ans)',))
        self.write(self.root/'Source/DHELAS/helper.f', [
            'subroutine helper(p,ans)', 'double precision p(4),ans,opaque',
            'double precision mdl_mass,mdl_width', 'external opaque',
            'common/custom_inputs/mdl_mass,mdl_width', 'ans=opaque(p)', 'end'])
        with self.assertRaisesRegex(MadGraph5Error,
                                    r'helper in Source/DHELAS/helper.f: unavailable callee opaque'):
            self.layout()

    def test_unused_module_helper_does_not_add_state(self):
        self.evaluator('born.f', 'mdl_constant')
        self.write(self.root/'Source/DHELAS/unreachable.f', [
            'double precision function unrelated()', "include 'coupl.inc'",
            'unrelated=mdl_unused+g', 'end'])
        result = self.layout()
        self.assertEqual(result['fixed_members'], [('mdl_constant', 'real', '1')])
        self.assertEqual(result['dynamic_members'], [])

    def test_logical_function_helper_is_resolved_and_its_numeric_reads_are_kept(self):
        self.evaluator('born.f', 'mdl_constant', extra=('logical available',
                                                     'external available',
                                                     'if(.not.available())return'))
        self.write('splitorders_stuff.f', [
            'logical function available()', "include 'coupl.inc'",
            'available=mdl_mass.gt.0d0', 'end'])
        self.assertEqual(self.names(self.layout(), 'fixed'), ['mdl_mass', 'mdl_constant'])

    def test_direct_fixed_model_common_write_is_rejected(self):
        self.evaluator('born.f', 'mdl_mass', extra=('call corrupt()',))
        self.write(self.root/'Source/DHELAS/corrupt.f', [
            'subroutine corrupt()', 'double precision mdl_mass',
            'common/mass_input/mdl_mass', 'if(mdl_mass.gt.0d0)mdl_mass=2d0', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) mdl_mass'):
            self.layout()

    def test_fixed_model_write_through_two_formal_argument_aliases_is_rejected(self):
        self.evaluator('born.f', 'mdl_mass', extra=('call relay(mdl_mass)',))
        self.write(self.root/'Source/DHELAS/mutate.f', [
            'subroutine relay(x)', 'double precision x', 'call mutate(x)', 'end',
            'subroutine mutate(y)', 'double precision y', 'y=7d0', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) mdl_mass'):
            self.layout()

    def test_fixed_model_write_through_array_element_and_keyword_argument_is_rejected(self):
        self.evaluator('born.f', 'dble(gal(1))', extra=('call mutate(value=gal(2))',))
        self.write(self.root/'Source/DHELAS/mutate.f', [
            'subroutine mutate(value)', 'double complex value', 'value=(0d0,0d0)', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) gal'):
            self.layout()

    def test_fixed_module_input_write_is_rejected(self):
        self.evaluator('born.f', 'dble(gal(1))', extra=('call mutate()',))
        path = self.root/'Source/DHELAS/module_helper.f90'
        path.write_text('module custom_model\ncomplex(8) gal(2)\nend module\n'
                        'subroutine mutate()\nuse custom_model\nimplicit none\n'
                        'gal(2)=(0d0,0d0)\nend subroutine\n')
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) gal'):
            self.layout()

    def test_dynamic_input_write_is_reported_for_conservative_reinstallation(self):
        self.evaluator('born.f', 'g', extra=('call mutate(g)',))
        self.write(self.root/'Source/DHELAS/mutate.f', [
            'subroutine mutate(value)', 'double precision value', 'value=2d0', 'end'])
        result = self.layout()
        self.assertEqual(result['manifest']['dynamic_writes'], ['g'])
        self.assertEqual(result['dynamic_members'], [('g', 'real', '1')])

    def test_helper_body_supplied_by_a_source_include_is_traced(self):
        self.evaluator('born.f', 'model_factor()', extra=('double precision model_factor',
                                                       'external model_factor'))
        self.write(self.root/'Source/MODEL/model_functions.f', ["include 'factor_body.inc'"])
        self.write(self.root/'Source/MODEL/factor_body.inc', [
            'double precision function model_factor()', "include 'factor.inc'",
            'model_factor=mdl_mass', 'end'])
        self.write(self.root/'Source/MODEL/factor.inc', [
            'double precision mdl_mass', 'common/mass_inputs/mdl_mass'])
        self.assertEqual(self.names(self.layout(), 'fixed'), ['mdl_mass'])

    def test_opaque_equivalence_or_pointer_alias_of_a_model_input_is_rejected(self):
        for alias in ('equivalence(mdl_mass,value)', 'value=>mdl_mass'):
            with self.subTest(alias=alias):
                self.evaluator('born.f', 'mdl_mass', extra=('call mutate()',))
                self.write(self.root/'Source/DHELAS/mutate.f', [
                    'subroutine mutate()', 'double precision mdl_mass,value',
                    'common/mass_inputs/mdl_mass', alias, 'value=2d0', 'end'])
                with self.assertRaisesRegex(MadGraph5Error, 'explicit dependency contract'):
                    self.layout()

    def test_external_function_overriding_an_intrinsic_keeps_its_model_reads(self):
        self.evaluator('born.f', 'exp(p(1))', extra=('double precision exp', 'external exp'))
        self.write(self.root/'Source/MODEL/custom.f', [
            'double precision function exp(value)', 'double precision value,mdl_mass',
            'common/mass_input/mdl_mass', 'exp=value+mdl_mass', 'end'])
        self.assertEqual(self.names(self.layout(), 'fixed'), ['mdl_mass'])

    def test_missing_external_override_of_an_intrinsic_is_not_whitelisted(self):
        self.evaluator('born.f', 'exp(p(1))', extra=('double precision exp', 'external exp'))
        with self.assertRaisesRegex(MadGraph5Error, 'unavailable callee exp'):
            self.layout()

    def test_reserved_runtime_rng_contract_needs_no_model_state_or_body(self):
        self.evaluator('born.f', 'ran2()', extra=('double precision ran2', 'external ran2'))
        result = self.layout()
        self.assertEqual(result['fixed_members'], [])
        self.assertEqual(result['dynamic_members'], [])

    def test_hash_in_fixed_form_continuation_column_is_not_a_preprocessor_directive(self):
        self.evaluator('born.f', 'model_factor()', extra=('double precision model_factor',
                                                       'external model_factor'))
        path = self.directory/'splitorders_stuff.f'
        path.write_text('      double precision function model_factor()\n'
                        "      include 'coupl.inc'\n"
                        '      model_factor=mdl_mass+\n'
                        '     # mdl_constant\n'
                        '      end\n')
        self.assertEqual(self.names(self.layout(), 'fixed'), ['mdl_mass', 'mdl_constant'])

    def test_hidden_preprocessor_dependencies_inside_a_reachable_include_are_rejected(self):
        self.evaluator('born.f', 'model_factor()', extra=('double precision model_factor',
                                                       'external model_factor'))
        self.write(self.root/'Source/MODEL/factor.f', [
            'double precision function model_factor()', "include 'opaque.inc'",
            'model_factor=value', 'end'])
        (self.root/'Source/MODEL/opaque.inc').write_text('#include "unavailable.h"\n')
        with self.assertRaisesRegex(MadGraph5Error, 'Source/MODEL/opaque.inc: preprocessor dependencies'):
            self.layout()

    def test_gal_inherits_running_ew_inputs_through_generated_assignment_closure(self):
        self.evaluator('born.f', 'dble(gal(2))')
        self.write(self.root/'Source/MODEL/intparam_definition.inc', [
            'if(readlha)then', 'fixed_initial=mu_r', 'endif',
            'weak_temporary=1d0/aEWM1', 'gal(1)=sqrt(weak_temporary)',
            'gal(2)=gal(1)**2'])
        ordinary = self.layout()
        self.assertEqual(ordinary['fixed_members'], [('gal', 'complex', '1:2')])
        self.assertEqual(ordinary['dynamic_members'], [])
        fixture = model()
        ew_input = next(p for p in fixture['parameters'][('external',)] if p.name == 'aEWM1')
        fixture['running_elements'] = [SimpleNamespace(run_objects=[[ew_input]])]
        running = self.layout(fixture)
        self.assertEqual(running['fixed_members'], [])
        self.assertEqual(running['dynamic_members'], [('gal', 'complex', '1:2')])
        self.assertIn('weak_temporary', running['manifest']['generated_alias_dependencies']['gal'])
        self.assertNotIn('aewm1', running['manifest']['used'])

    def test_gal_initializer_in_readlha_is_fixed_but_per_point_else_is_dynamic(self):
        self.evaluator('born.f', 'dble(gal(1))')
        path = self.root/'Source/MODEL/intparam_definition.inc'
        self.write(path, ['if(readlha)then', 'gal(1)=mu_r', 'gal(2)=1d0', 'endif'])
        self.assertEqual(self.names(self.layout(), 'fixed'), ['gal'])
        self.write(path, ['if(readlha)then', 'gal(1)=1d0', 'else', 'gal(1)=mu_r',
                          'endif', 'gal(2)=1d0'])
        self.assertEqual(self.names(self.layout(), 'dynamic'), ['gal'])

    def test_generated_alias_control_dependency_can_make_constant_rhs_dynamic(self):
        self.evaluator('born.f', 'dble(gal(1))')
        self.write(self.root/'Source/MODEL/intparam_definition.inc', [
            'if(mu_r.gt.100d0)then', 'gal(1)=1d0', 'else', 'gal(1)=2d0',
            'endif', 'gal(2)=1d0'])
        self.assertEqual(self.names(self.layout(), 'dynamic'), ['gal'])

    def test_use_renamed_model_common_input_retains_its_original_field(self):
        self.write('born.f', ['subroutine born(p,ans)',
                             'use visible_model, only: local_mass=>mdl_mass',
                             'implicit none', 'double precision p(4),ans',
                             'ans=local_mass', 'end'])
        self.write(self.root/'Source/MODEL/visible_model.f', [
            'module visible_model', 'double precision mdl_mass',
            'common/model_mdl_mass/mdl_mass', 'end module'])
        result = self.layout()
        self.assertEqual(result['fixed_members'], [('mdl_mass', 'real', '1')])
        self.assertEqual(result['dynamic_members'], [])

    def test_use_renamed_model_input_mutation_through_keyword_alias_is_rejected(self):
        self.write('born.f', ['subroutine born(p,ans)',
                             'use visible_model, only: local_mass=>mdl_mass',
                             'implicit none', 'double precision p(4),ans',
                             'call mutate(local_mass=local_mass)',
                             'ans=local_mass', 'end'])
        self.write(self.root/'Source/MODEL/visible_model.f', [
            'module visible_model', 'double precision mdl_mass',
            'common/model_mdl_mass/mdl_mass', 'end module'])
        self.write(self.root/'Source/DHELAS/mutate.f', [
            'subroutine mutate(local_mass)', 'double precision local_mass',
            'local_mass=2d0', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) mdl_mass'):
            self.layout()

    def test_use_renamed_function_is_traced_to_its_model_reads(self):
        self.write('born.f', ['subroutine born(p,ans)',
                             'use functions, only: local_factor=>model_factor',
                             'implicit none', 'double precision p(4),ans',
                             'ans=local_factor()', 'end'])
        self.write(self.root/'Source/MODEL/functions.f', [
            'module functions', 'contains', 'double precision function model_factor()',
            'double precision mu_r', 'common/rscale/mu_r',
            'model_factor=mu_r', 'end function', 'end module'])
        self.assertEqual(self.names(self.layout(), 'dynamic'), ['mu_r'])

    def test_known_model_aliases_reexported_by_modules_remain_inputs(self):
        self.write(self.root/'Source/MODEL/visible_model.f', [
            'module visible_model', 'double precision mdl_mass',
            'common/model_mdl_mass/mdl_mass', 'end module',
            'module bridge_model', 'use visible_model, only: exported_mass=>mdl_mass',
            'end module', 'module wildcard_model', 'use bridge_model', 'end module'])
        for use, symbol in [('use bridge_model, only: local_mass=>exported_mass', 'local_mass'),
                            ('use bridge_model, only: exported_mass', 'exported_mass'),
                            ('use wildcard_model', 'exported_mass')]:
            with self.subTest(use=use):
                self.write('born.f', ['subroutine born(p,ans)', use, 'implicit none',
                                     'double precision p(4),ans', 'ans='+symbol, 'end'])
                self.assertEqual(self.layout()['fixed_members'], [('mdl_mass', 'real', '1')])

    def test_second_statement_fixed_input_mutation_is_not_hidden_by_a_semicolon(self):
        self.evaluator('born.f', 'mdl_mass; mdl_mass=0d0')
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) mdl_mass'):
            self.layout()

    def test_semicolon_split_also_applies_to_free_form_helper_bodies(self):
        self.evaluator('born.f', 'mdl_mass', extra=('call mutate()',))
        self.write(self.root/'Source/DHELAS/mutate.f90', [
            'subroutine mutate()', 'double precision mdl_mass',
            'common/mass_input/mdl_mass', 'mdl_mass=mdl_mass; mdl_mass=0d0', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, r'mutates fixed model input\(s\) mdl_mass'):
            self.layout()

    def test_quoted_semicolons_and_doubled_quotes_do_not_create_model_reads_or_writes(self):
        self.evaluator('born.f', 'mdl_constant', extra=(
            "write(*,*) 'it''s; mdl_mass=0; gc_s'; write(*,*) \"gal;g=2\" ! mdl_width=0;",
        ))
        result = self.layout()
        self.assertEqual(result['fixed_members'], [('mdl_constant', 'real', '1')])
        self.assertEqual(result['dynamic_members'], [])

    def test_pointer_declaration_initializer_aliasing_a_model_input_is_rejected(self):
        self.write(self.root/'Source/MODEL/visible_model.f', [
            'module visible_model', 'double precision,target::mdl_mass',
            'common/model_mdl_mass/mdl_mass', 'save/model_mdl_mass/', 'end module'])
        self.write('born.f', ['subroutine born(p,ans)', 'use visible_model',
                             'implicit none', 'double precision p(4),ans',
                             'double precision,pointer::alias=>mdl_mass', 'ans=alias', 'end'])
        with self.assertRaisesRegex(MadGraph5Error, 'pointer association requires an explicit dependency contract'):
            self.layout()

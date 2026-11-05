################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for the customize_model command"""

from __future__ import absolute_import
import io
import os

import tests.unit_tests as unittest

import madgraph.interface.madgraph_interface as mg_interface
import madgraph.models.build_restriction_lib as build_restrict_lib
import madgraph.models.check_param_card as check_param_card
import madgraph.models.import_ufo as import_ufo
import madgraph.models.write_param_card as param_writer


def get_option(categories, name):
    for category in categories:
        for option in category:
            if option.name == name:
                return option
    return None


#===============================================================================
# The generic (model independent) options
#===============================================================================
class TestGenericCategories(unittest.TestCase):
    """The mass scheme options are built from the content of the model"""

    @classmethod
    def setUpClass(cls):
        sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(sm_path)
        cls.default_model = import_ufo.import_model(sm_path)

    def test_flavour_scheme_options(self):
        """the three schemes only differ by the mass of the c and b quark"""

        categories = build_restrict_lib.get_generic_categories(self.full_model)
        option = get_option(categories, 'flavour scheme')
        self.assertNotEqual(option, None)
        self.assertEqual(option.labels, ['3F', '4F', '5F'])

        target = {}
        for label, rules in option.choices:
            target[label] = set((block.lower(), tuple(code))
                                for block, code, value in rules)
            # a mass scheme only puts parameters to zero
            self.assertEqual(set(value for block, code, value in rules) - set([0.]),
                             set())

        # u, d and s are massless in the sm UFO: nothing to restrict for 3F
        self.assertEqual(target['3F'], set())
        self.assertEqual(target['4F'], set([('mass', (4,)), ('yukawa', (4,))]))
        self.assertEqual(target['5F'], set([('mass', (4,)), ('yukawa', (4,)),
                                            ('mass', (5,)), ('yukawa', (5,))]))

    def test_lepton_scheme_options(self):
        """the massive leptons are taken from the heaviest one"""

        categories = build_restrict_lib.get_generic_categories(self.full_model)
        option = get_option(categories, 'nb of massive leptons')
        self.assertNotEqual(option, None)
        self.assertEqual(option.labels, ['0', '1', '2', '3'])

        target = {}
        for label, rules in option.choices:
            target[label] = set((block.lower(), tuple(code))
                                for block, code, value in rules)

        self.assertEqual(target['3'], set())
        self.assertEqual(target['2'], set([('mass', (11,)), ('yukawa', (11,))]))
        self.assertEqual(target['1'], set([('mass', (11,)), ('yukawa', (11,)),
                                           ('mass', (13,)), ('yukawa', (13,))]))
        # a massless particle can not decay: the tau width follows its mass
        self.assertEqual(target['0'], set([('mass', (11,)), ('yukawa', (11,)),
                                           ('mass', (13,)), ('yukawa', (13,)),
                                           ('mass', (15,)), ('yukawa', (15,)),
                                           ('decay', (15,))]))

    def test_default_follows_the_loaded_model(self):
        """the default of the options reproduces the model as loaded"""

        categories = build_restrict_lib.get_generic_categories(self.full_model,
                                                            self.default_model)
        # default sm: c massless, b massive / tau massive, e and mu massless
        self.assertEqual(get_option(categories, 'flavour scheme').status, '4F')
        self.assertEqual(get_option(categories, 'nb of massive leptons').status, '1')

        # on the full model everything is massive
        categories = build_restrict_lib.get_generic_categories(self.full_model)
        self.assertEqual(get_option(categories, 'flavour scheme').status, '3F')
        self.assertEqual(get_option(categories, 'nb of massive leptons').status, '3')


#===============================================================================
# The options of the question
#===============================================================================
class TestChoiceOption(unittest.TestCase):
    """A customization option with more than two values"""

    def get_option(self):
        return build_restrict_lib.ChoiceOption('scheme',
                    [('a', [('MASS', [5], 0.0)]), ('b', []), ('c', [])], 'b')

    def test_status(self):
        option = self.get_option()
        self.assertEqual(option.status, 'b')
        option.next_status()
        self.assertEqual(option.status, 'c')
        option.next_status()
        self.assertEqual(option.status, 'a')
        option.set_status('b')
        self.assertEqual(option.status, 'b')
        self.assertRaises(ValueError, option.set_status, 'd')

    def test_an_abbreviation_is_accepted(self):
        """'set flavourscheme 5' is the same as 'set flavourscheme 5F'"""

        option = build_restrict_lib.ChoiceOption('flavour scheme',
                    [('3F', []), ('4F', []), ('5F', [])], '4F')
        option.set_status('5')
        self.assertEqual(option.status, '5F')
        option.set_status('3f')     # case insensitive
        self.assertEqual(option.status, '3F')
        option.set_status(' 4 ')    # and spaces around it
        self.assertEqual(option.status, '4F')
        # an exact label still wins, and an unknown one is still refused
        option.set_status('5F')
        self.assertEqual(option.status, '5F')
        self.assertRaises(ValueError, option.set_status, '9')
        self.assertRaises(ValueError, option.set_status, 'F')

    def test_an_ambiguous_abbreviation_is_refused(self):
        option = build_restrict_lib.ChoiceOption('scheme',
                    [('3F', []), ('3G', [])], '3F')
        self.assertRaises(ValueError, option.set_status, '3')
        self.assertEqual(option.status, '3F') # unchanged
        option.set_status('3G')
        self.assertEqual(option.status, '3G')

    def test_an_exact_label_wins_over_a_prefix(self):
        """with labels '1' and '10', '1' is the first one"""

        option = build_restrict_lib.ChoiceOption('nb', [('1', []), ('10', [])], '1')
        option.set_status('1')
        self.assertEqual(option.status, '1')

    def test_get_rules(self):
        option = self.get_option()
        self.assertEqual(option.get_rules(), [])
        option.set_status('a')
        self.assertEqual(option.get_rules(), [('MASS', [5], 0.0)])

    def test_rule_get_rules(self):
        category = build_restrict_lib.Category('test')
        category.add_options(name='massless b', default=False,
                             rules=[('MASS', [5], 0.0), ('YUKAWA', [5], 0.0)])
        self.assertEqual([r for rule in category for r in rule.get_rules()], [])
        for rule in category:
            rule.status = True
        self.assertEqual([r for rule in category for r in rule.get_rules()],
                         [('mass', [5], 0.0), ('yukawa', [5], 0.0)])


#===============================================================================
# The restriction card built by customize_model
#===============================================================================
class TestCustomizeCard(unittest.TestCase):
    """The way customize_model fills the restriction card"""

    @classmethod
    def setUpClass(cls):
        sm_path = import_ufo.find_ufo_path('sm')
        cls.model = import_ufo.import_full_model(sm_path)
        out = io.StringIO()
        param_writer.ParamCardWriter(cls.model, out)
        cls.text = out.getvalue()

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.card = check_param_card.ParamCard(self.text.split('\n'))
        self.externals = self.cmd.get_external_lhacode(self.model)

    def test_randomize_only_touches_dangerous_values(self):
        """0, 1 and duplicated values are the ones removed by the restriction"""

        before = dict(((block, tuple(param.lhacode)), param.value)
                      for block in self.card for param in self.card[block])
        self.cmd.randomize_param_card(self.card, self.externals)

        for block in self.card:
            for param in self.card[block]:
                key = (block, tuple(param.lhacode))
                old, new = before[key], param.value
                if key not in self.externals:
                    # informative entries of the card: never touched
                    self.assertEqual(old, new)
                elif old in (0., 1.):
                    if block == 'decay' and old == 0.:
                        self.assertEqual(new, 0.) # a massless particle can not decay
                    else:
                        self.assertNotEqual(new, old)
                        self.assertTrue(0 < new < 1)

        # no value of a block can be simplified by the restriction anymore.
        # Only inside a block: that is where the restriction merges parameters,
        # so MASS 15 and YUKAWA 15 may well keep the same value.
        for block in self.card:
            if block == 'loop' or block.startswith(('qnumbers', 'decay_table')):
                continue
            values = [param.value for param in self.card[block]
                      if (block, tuple(param.lhacode)) in self.externals
                      and not (block == 'decay' and param.value == 0.)]
            self.assertEqual(len(values), len(set(values)), block)
            self.assertFalse([v for v in values if v in (0., 1.)], block)

    def test_a_value_shared_between_two_blocks_is_kept(self):
        """MTA and ymtau are both 1.777 in the sm and stay so: the restriction
        never merges two parameters of two different blocks"""

        self.cmd.randomize_param_card(self.card, self.externals)
        self.assertEqual(self.card['mass'].get([15]).value,
                         self.card['yukawa'].get([15]).value)

    def test_randomize_keeps_the_order_of_magnitude(self):
        """MZ must not be replaced by a O(1) value because of a duplicate"""

        self.card['mass'].get([23]).value = 91.188
        self.card['mass'].get([25]).value = 91.188
        self.cmd.randomize_param_card(self.card, self.externals)
        values = sorted([self.card['mass'].get([23]).value,
                         self.card['mass'].get([25]).value])
        self.assertNotEqual(values[0], values[1])
        self.assertEqual(values[0], 91.188)
        self.assertTrue(91.188 < values[1] < 2 * 91.188)

    def test_apply_rules(self):
        """the user choices are written in the card"""

        category = build_restrict_lib.Category('test')
        category.add_options(name='massless b', default=True,
                             rules=[('MASS', [5], 0.0), ('YUKAWA', [5], 0.0)])

        restricted = self.cmd.apply_customize_rules(self.card, [category],
                    set_zero=[('mass', (4,))], set_one=[('sminputs', (3,))],
                    set_equal=[(('yukawa', (6,)), ('mass', (6,)))])

        self.assertEqual(self.card['mass'].get([5]).value, 0.)
        self.assertEqual(self.card['yukawa'].get([5]).value, 0.)
        self.assertEqual(self.card['mass'].get([4]).value, 0.)
        self.assertEqual(self.card['sminputs'].get([3]).value, 1.)
        self.assertEqual(self.card['yukawa'].get([6]).value,
                         self.card['mass'].get([6]).value)
        self.assertEqual(restricted,
                         set([('mass', (5,)), ('yukawa', (5,)), ('mass', (4,)),
                              ('sminputs', (3,)), ('yukawa', (6,))]))

    def test_set_equal_chain_is_resolved(self):
        """'set_equal A B' + 'set_equal B C' has to make the three of them equal,
        whatever the order they were entered in."""

        A, B, C = ('mass', (4,)), ('mass', (5,)), ('mass', (6,))
        for chain in ([(A, B), (B, C)], [(B, C), (A, B)]):
            card = check_param_card.ParamCard(self.text.split('\n'))
            set_equal = mg_interface.MadGraphCmd.resolve_set_equal(chain)
            self.cmd.apply_customize_rules(card, [], [], [], set_equal)
            values = [card['mass'].get([pdg]).value for pdg in (4, 5, 6)]
            self.assertEqual(values[0], values[2])
            self.assertEqual(values[1], values[2])

    def test_set_equal_cycle_does_not_loop(self):
        """'set_equal A B' + 'set_equal B A' is degenerate but must terminate"""

        A, B = ('mass', (4,)), ('mass', (5,))
        set_equal = mg_interface.MadGraphCmd.resolve_set_equal([(A, B), (B, A)])
        card = check_param_card.ParamCard(self.text.split('\n'))
        self.cmd.apply_customize_rules(card, [], [], [], set_equal)
        self.assertEqual(card['mass'].get([4]).value, card['mass'].get([5]).value)

    def test_set_equal_follows_a_restricted_parameter(self):
        """'set_equal A B' with B set to zero does put A to zero as well"""

        self.cmd.apply_customize_rules(self.card, [], set_zero=[('mass', (5,))],
                    set_one=[], set_equal=[(('yukawa', (5,)), ('mass', (5,)))])
        self.assertEqual(self.card['yukawa'].get([5]).value, 0.)


#===============================================================================
# The options of the model itself
#===============================================================================
class TestModelCategories(unittest.TestCase):
    """The options of the build_restrict.py of the model are kept only if they
    do not conflict with the generic ones"""

    @classmethod
    def setUpClass(cls):
        sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(sm_path)
        cls.default_model = import_ufo.import_model(sm_path)

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()

    def test_group_options(self):
        """a category is a flat list of rules: they have to be re-grouped"""

        category = build_restrict_lib.Category('test')
        category.add_options(name='one', default=True,
                             rules=[('MASS', [5], 0.0), ('YUKAWA', [5], 0.0)])
        category.add_options(name='two', default=True, rules=[('MASS', [4], 0.0)])

        options = self.cmd.group_options(category)
        self.assertEqual([len(o) for o in options], [2, 1])
        self.assertEqual([o[0].name for o in options], ['one', 'two'])

    def test_sm_options(self):
        """every option of the sm is now one of the generic ones"""

        categories = self.cmd.get_customize_categories(self.full_model,
                                                       self.default_model)
        names = [option.name for category in categories for option in category
                                                             if option.first]
        self.assertTrue('flavour scheme' in names)
        self.assertTrue('nb of massive leptons' in names)
        self.assertTrue('diagonal ckm' in names)
        # ... so all of them are dropped from the build_restrict.py of the
        # model, and the generic ones are the only ones left
        self.assertEqual(sorted(names), ['diagonal ckm', 'flavour scheme',
                                         'nb of massive leptons'])
        # each of them is proposed once: an option which is superseded by a
        # generic one must not be kept next to it
        self.assertEqual(len(names), len(set(names)))

    def test_sm_ckm_comes_from_the_generic_option(self):
        """the sm declares a 'diagonal ckm' too: only one of them is kept"""

        categories = self.cmd.get_customize_categories(self.full_model,
                                                       self.default_model)
        ckm = [category for category in categories
                        if category.name == 'quark mixing']
        self.assertEqual(len(ckm), 1)
        self.assertEqual(set((rule.lhablock, tuple(rule.lhaid))
                             for rule in ckm[0]),
                         set([('wolfenstein', (1,)), ('wolfenstein', (2,)),
                              ('wolfenstein', (3,)), ('wolfenstein', (4,))]))


#===============================================================================
# The commands of the customize_model question
#===============================================================================
class TestAskforCustomize(unittest.TestCase):
    """The parameter/coupling modifications entered at the question"""

    @classmethod
    def setUpClass(cls):
        sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(sm_path)

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        categories = self.cmd.get_customize_categories(self.full_model,
                                                       self.full_model)
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
                                                categories=categories)

    def test_set_zero_and_set_one(self):
        self.ask.do_set_zero('MB')
        self.ask.do_set_one('lamWS')
        self.assertEqual(self.ask.set_zero, [('MASS', (5,))])
        self.assertEqual(self.ask.set_one, [('Wolfenstein', (1,))])

        # the name is case insensitive
        self.ask.do_set_zero('mta')
        self.assertEqual(self.ask.set_zero, [('MASS', (5,)), ('MASS', (15,))])

        # an unknown (or internal) parameter is refused
        self.ask.do_set_zero('NotAParameter')
        self.ask.do_set_zero('ee') # internal
        self.assertEqual(self.ask.set_zero, [('MASS', (5,)), ('MASS', (15,))])

    def test_a_parameter_has_only_one_status(self):
        """the last command entered for a parameter is the one kept"""

        self.ask.do_set_zero('MB')
        self.ask.do_set_one('MB')
        self.assertEqual(self.ask.set_zero, [])
        self.assertEqual(self.ask.set_one, [('MASS', (5,))])

        self.ask.do_set_equal('MB MT')
        self.assertEqual(self.ask.set_one, [])
        self.assertEqual(self.ask.set_equal, [(('MASS', (5,)), ('MASS', (6,)))])

    def test_set_equal(self):
        self.ask.do_set_equal('ymtau MTA')
        self.assertEqual(self.ask.set_equal, [(('YUKAWA', (15,)), ('MASS', (15,)))])
        # a parameter can not be identified to itself
        self.ask.do_set_equal('MTA MTA')
        self.assertEqual(len(self.ask.set_equal), 1)
        # wrong number of arguments
        self.ask.do_set_equal('MTA')
        self.assertEqual(len(self.ask.set_equal), 1)

    def test_formula_and_coupling(self):
        self.ask.do_formula('MB = MT/2.')
        self.assertEqual(self.ask.new_formula, [('MB', 'MT/2.')])
        # entering it twice only keeps the last one
        self.ask.do_formula('MB = MT/3.')
        self.assertEqual(self.ask.new_formula, [('MB', 'MT/3.')])
        # a parameter can not be defined from itself
        self.ask.do_formula('MB = MB/3.')
        self.assertEqual(self.ask.new_formula, [('MB', 'MT/3.')])
        # an internal parameter can not be redefined that way
        self.ask.do_formula('ee = 0.3')
        self.assertEqual(self.ask.new_formula, [('MB', 'MT/3.')])

        self.ask.do_coupling('GC_1 = 2*ee')
        self.assertEqual(self.ask.new_coupling, [('GC_1', '2*ee')])
        self.ask.do_coupling('NotACoupling = 2*ee')
        self.assertEqual(self.ask.new_coupling, [('GC_1', '2*ee')])

    def test_clear(self):
        self.ask.do_set_zero('MB')
        self.ask.do_formula('MTA = MT/2.')
        self.ask.do_coupling('GC_1 = 2*ee')
        self.ask.do_clear('')
        self.assertEqual(self.ask.set_zero, [])
        self.assertEqual(self.ask.new_formula, [])
        self.assertEqual(self.ask.new_coupling, [])

    def test_set_of_a_choice_option(self):
        """'set flavourscheme 5F' is the scripted version of the question"""

        self.ask.do_set('flavourscheme 5F')
        option = get_option(self.ask.all_categories, 'flavour scheme')
        self.assertEqual(option.status, '5F')
        # an invalid value does not change anything
        self.ask.do_set('flavourscheme 6F')
        self.assertEqual(option.status, '5F')
        # and the boolean options still work
        self.ask.do_set('diagonalckm False')
        self.assertEqual(get_option(self.ask.all_categories, 'diagonal ckm').status,
                         False)

    def test_question_lists_the_modifications(self):
        self.ask.do_set_zero('MB')
        self.ask.do_set_equal('ymtau MTA')
        self.ask.do_coupling('GC_1 = 2*ee')
        question = self.ask.get_question()
        self.assertTrue('MB (MASS [5]) = 0' in question)
        self.assertTrue('ymtau (YUKAWA [15]) = MTA (MASS [15])' in question)
        self.assertTrue('GC_1 = 2*ee' in question)

    def test_answer_is_the_categories(self):
        """ask() returns the options, both in interactive and script mode"""

        self.assertEqual(self.ask.answer, self.ask.all_categories)


#===============================================================================
# The name asked for the new model written by the advanced options
#===============================================================================
class TestAskforModelName(unittest.TestCase):
    """A script which does not answer that question must not have its next
    command used as the name of the model."""

    def setUp(self):
        self.question = mg_interface.AskforModelName('name?', default='sm_custom')

    def test_a_name_is_accepted(self):
        self.assertEqual(
            self.question.special_check_answer_in_input_file('my_model', 'x'),
            'my_model')
        self.assertEqual(
            self.question.special_check_answer_in_input_file('  sm2  ', 'x'),
            'sm2')

    def test_a_command_is_refused(self):
        for line in ['generate p p > t t~', 'output MYDIR', 'sm-restriction',
                     'a/b', '']:
            self.assertEqual(
                self.question.special_check_answer_in_input_file(line, 'x'),
                None, '%s should not be taken as a model name' % line)


#===============================================================================
# customize_model --explain
#===============================================================================
class TestExplainRestriction(unittest.TestCase):
    """The attribution of each removal to one of the user choices"""

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.card = check_param_card.ParamCard("""
Block mass
    4 0.0 # mc
    5 2.0 # mb
    6 2.0 # mt
   23 1.0 # mz
Block yukawa
    5 2.0 # ymb
DECAY 6 0.0 # wt
DECAY 23 0.0 # wz
""".split('\n'))
        self.externals = set([('mass', (4,)), ('mass', (5,)), ('mass', (6,)),
                              ('mass', (23,)), ('yukawa', (5,)),
                              ('decay', (6,)), ('decay', (23,))])

    def test_parameter_classes(self):
        """0/1 leave the card, and so do the duplicates of a same block"""

        dropped, fused = self.cmd.get_parameter_classes(self.card, self.externals)
        # MC = 0, MZ = 1 and the two zero widths become internal parameters
        self.assertEqual(dropped, set([('mass', (4,)), ('mass', (23,)),
                                       ('decay', (6,)), ('decay', (23,))]))
        # MB and MT share their value, ymb only shares it across blocks
        self.assertEqual(fused, set([('mass', (5,)), ('mass', (6,))]))

    def test_zero_widths_are_not_fused(self):
        """two particles with a zero width are not two identical parameters"""

        dropped, fused = self.cmd.get_parameter_classes(self.card, self.externals)
        self.assertFalse([k for k in fused if k[0] == 'decay'])
        # they are zero, so they do leave the card
        self.assertTrue(('decay', (6,)) in dropped)

    def test_only_the_external_parameters_are_considered(self):
        """the informative entries of a param_card are not model parameters"""

        dropped, fused = self.cmd.get_parameter_classes(self.card, set())
        self.assertEqual(dropped, set())
        self.assertEqual(fused, set())


class TestRestrictionGroups(unittest.TestCase):
    """The user choices, as the units --explain attributes a removal to"""

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.category = build_restrict_lib.Category('test')
        self.category.add_options(name='massless b', default=True,
                             rules=[('MASS', [5], 0.0), ('YUKAWA', [5], 0.0)])
        self.category.add_options(name='not selected', default=False,
                             rules=[('MASS', [4], 0.0)])
        self.category.append(build_restrict_lib.ChoiceOption('scheme',
                    [('a', [('MASS', [6], 0.0)]), ('b', [])], 'a'))

    def test_groups(self):
        groups = self.cmd.get_restriction_groups([self.category],
                    set_zero=[('YUKAWA', (6,))], set_one=[],
                    set_equal=[(('MASS', (5,)), ('MASS', (6,)))],
                    lha2name={('yukawa', (6,)): 'ymt', ('mass', (5,)): 'MB',
                              ('mass', (6,)): 'MT'})
        labels = [label for label, keys in groups]
        # an option which is not selected has no rule, so it is not a group
        self.assertEqual(labels, ['massless b', 'scheme = a', 'set_zero ymt',
                                  'set_equal MB MT'])
        keys = dict(groups)
        # the rules of an option are grouped together
        self.assertEqual(keys['massless b'],
                         set([('mass', (5,)), ('yukawa', (5,))]))
        self.assertEqual(keys['scheme = a'], set([('mass', (6,))]))
        self.assertEqual(keys['set_zero ymt'], set([('yukawa', (6,))]))


#===============================================================================
# A scheme is only proposed if the model can be put in it
#===============================================================================
class FakeParam(object):
    def __init__(self, name, lhablock, lhacode):
        self.name, self.lhablock, self.lhacode = name, lhablock, lhacode


class FakeParticle(dict):
    def get(self, name):
        return self[name]


class FakeModel(dict):
    """the few entries build_restriction_lib reads out of a model"""

    def __init__(self, masses):
        """masses: pdg -> mass parameter name ('ZERO' for a massless one).
        A mass is external unless its name starts with 'internal'."""

        particles, external = {}, []
        for pdg, mass in masses.items():
            particles[pdg] = FakeParticle({'pdg_code': pdg, 'mass': mass,
                                           'width': 'ZERO'})
            if mass != 'ZERO' and not mass.startswith('internal'):
                external.append(FakeParam(mass, 'MASS', [pdg]))
        dict.__init__(self, {'particle_dict': particles,
                             'parameters': {('external',): external}})

    def get(self, name):
        return self[name]


class TestReachableSchemes(unittest.TestCase):
    """The flavour/lepton scheme is always proposed, but only with the values
    this model can actually take."""

    def test_all_schemes_when_everything_is_a_parameter(self):
        model = FakeModel({1: 'ZERO', 2: 'ZERO', 3: 'ZERO', 4: 'MC', 5: 'MB'})
        option = build_restrict_lib.get_flavour_scheme_option(model, model)
        self.assertEqual(option.labels, ['3F', '4F', '5F'])
        self.assertFalse('Only' in option.description)

    def test_only_5F_when_c_and_b_are_massless_in_the_ufo(self):
        """the option is still proposed, and says why it has a single value"""

        model = FakeModel({1: 'ZERO', 2: 'ZERO', 3: 'ZERO', 4: 'ZERO', 5: 'ZERO'})
        option = build_restrict_lib.get_flavour_scheme_option(model, model)
        self.assertEqual(option.labels, ['5F'])
        self.assertEqual(option.status, '5F')
        self.assertTrue('Only 5F possible for this model' in option.description)
        self.assertTrue('c, b have no mass in this model' in option.description)
        # cycling on a single value is a no-op, not a crash
        option.next_status()
        self.assertEqual(option.status, '5F')

    def test_3F_refused_when_c_has_no_mass(self):
        model = FakeModel({1: 'ZERO', 2: 'ZERO', 3: 'ZERO', 4: 'ZERO', 5: 'MB'})
        option = build_restrict_lib.get_flavour_scheme_option(model, model)
        self.assertEqual(option.labels, ['4F', '5F'])
        self.assertTrue('3F (c has no mass in this model)' in option.description)

    def test_scheme_refused_when_the_mass_is_internal(self):
        """a mass which is not in the param_card can not be set to zero"""

        model = FakeModel({1: 'ZERO', 2: 'ZERO', 3: 'ZERO',
                           4: 'internal_MC', 5: 'MB'})
        option = build_restrict_lib.get_flavour_scheme_option(model, model)
        self.assertEqual(option.labels, ['3F'])
        self.assertTrue('the mass of c is not in the param_card'
                        in option.description)

    def test_no_option_without_quarks(self):
        self.assertEqual(
            build_restrict_lib.get_flavour_scheme_option(FakeModel({}), FakeModel({})),
            None)

    def test_lepton_scheme_is_restricted_too(self):
        model = FakeModel({11: 'ZERO', 13: 'ZERO', 15: 'MTA'})
        option = build_restrict_lib.get_lepton_scheme_option(model, model)
        self.assertEqual(option.labels, ['0', '1'])
        self.assertTrue('Only 0 and 1 possible for this model'
                        in option.description)

    def test_no_lepton_option_without_leptons(self):
        self.assertEqual(
            build_restrict_lib.get_lepton_scheme_option(FakeModel({}), FakeModel({})),
            None)


class TestQuestionAlwaysShowsTheModifications(unittest.TestCase):
    """The parameter/coupling section is shown even when it is empty, so that
    the set_zero/set_one commands are discoverable."""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))

    def test_empty_section_is_shown(self):
        question = self.ask.get_question()
        self.assertTrue('parameter/coupling modifications (use set_zero/set_one):'
                        in question)
        self.assertTrue('    none' in question)

    def test_section_lists_the_modifications(self):
        self.ask.do_set_zero('MB')
        question = self.ask.get_question()
        self.assertTrue('parameter/coupling modifications (use set_zero/set_one):'
                        in question)
        self.assertFalse('    none' in question)
        self.assertTrue('MB (MASS [5]) = 0' in question)


#===============================================================================
# set_equal only proposes what can really be merged
#===============================================================================
class TestSetEqualCompletion(unittest.TestCase):
    """The restriction only fuses two parameters of a same block (and never two
    widths), so the completion must not propose anything else."""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))

    def complete(self, line):
        return self.ask.complete_set_equal(line.split()[-1] if
                    not line.endswith(' ') else '', line, len(line), len(line))

    def test_widths_are_never_proposed(self):
        """two zero widths are not two identical parameters for the restriction"""

        self.assertFalse('decay' in self.ask.get_identifiable())
        for name in self.complete('set_equal '):
            param = self.ask.external_params[name.lower()]
            self.assertNotEqual(param.lhablock.lower(), 'decay')

    def test_a_block_with_a_single_parameter_is_not_proposed(self):
        """such a parameter could only be merged across blocks, which the
        restriction does not do"""

        for block, names in self.ask.get_identifiable().items():
            self.assertTrue(len(names) > 1, block)

    def test_first_argument(self):
        first = self.complete('set_equal ')
        self.assertTrue('MB' in first)   # MASS has several entries
        self.assertTrue('ymb' in first)  # and so has YUKAWA

    def test_second_argument_stays_in_the_block(self):
        """once MB is given, only the other masses can be proposed"""

        second = self.complete('set_equal MB ')
        self.assertTrue('MT' in second)
        self.assertFalse('MB' in second)     # not itself
        self.assertFalse('ymb' in second)    # not another block
        for name in second:
            self.assertEqual(
                self.ask.external_params[name.lower()].lhablock.lower(), 'mass')

    def test_second_argument_of_an_unknown_parameter(self):
        self.assertEqual(self.complete('set_equal NotAParameter '), [])


#===============================================================================
# 'set' as a shortcut for set_zero/set_one/set_equal
#===============================================================================
class TestSetShortcut(unittest.TestCase):
    """'set yt 0', 'set yt 1' and 'set yt = yb' are the short forms"""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))

    def test_set_to_zero_and_one(self):
        self.ask.do_set('ymb 0')
        self.assertEqual(self.ask.set_zero, [('YUKAWA', (5,))])
        self.ask.do_set('ymt 1')
        self.assertEqual(self.ask.set_one, [('YUKAWA', (6,))])

    def test_set_equal_spacing(self):
        """'=' may be glued to either side, or stand alone"""

        for line in ['ymb = ymt', 'ymb= ymt', 'ymb =ymt', 'ymb=ymt']:
            self.ask.do_clear('')
            self.ask.do_set(line)
            self.assertEqual(self.ask.set_equal,
                             [(('YUKAWA', (5,)), ('YUKAWA', (6,)))], line)

    def test_set_equal_without_the_equal_sign(self):
        self.ask.do_set('ymb ymt')
        self.assertEqual(self.ask.set_equal,
                         [(('YUKAWA', (5,)), ('YUKAWA', (6,)))])

    def test_the_options_still_win(self):
        """an option of the question is not a parameter"""

        self.ask.do_set('diagonalckm False')
        self.assertEqual(get_option(self.ask.all_categories, 'diagonal ckm').status,
                         False)
        self.assertEqual(self.ask.set_zero, [])

    def test_refused_values(self):
        """only 0, 1 and another external parameter make sense"""

        self.ask.do_set('ymb 2')
        self.ask.do_set('ymb = NotAParameter')
        self.ask.do_set('NotAParameter 0')
        self.assertEqual(self.ask.set_zero, [])
        self.assertEqual(self.ask.set_one, [])
        self.assertEqual(self.ask.set_equal, [])

    def test_an_internal_parameter_is_refused(self):
        """'yt' is internal in the sm: it is not in the param_card"""

        self.ask.do_set('yt 0')
        self.assertEqual(self.ask.set_zero, [])

    def test_completion_of_a_parameter(self):
        """the second argument of 'set MB' is 0, 1 or a mass it can merge with"""

        out = self.ask.complete_set('', 'set MB ', 7, 7)
        self.assertTrue('0' in out)
        self.assertTrue('1' in out)
        self.assertTrue('MT' in out)
        self.assertFalse('MB' in out)
        self.assertFalse('ymb' in out)


#===============================================================================
# the --explain modes
#===============================================================================
class TestExplainModes(unittest.TestCase):
    """--explain=life reports each command, --explain=final reports the
    attribution once the question is closed, --explain alone means life.
    The live report is what one gets without asking for it, --explain=off
    is how one asks for no report at all."""

    def test_parse(self):
        parse = mg_interface.parse_explain_mode
        self.assertEqual(parse('--explain'), 'life')
        self.assertEqual(parse('--explain=life'), 'life')
        self.assertEqual(parse('--explain=final'), 'final')
        self.assertEqual(parse('--explain=off'), 'off')
        # 'live' is the spelling one expects, accept it too
        self.assertEqual(parse('--explain=live'), 'life')
        self.assertEqual(parse('--explain=LIFE'), 'life')
        # and so are the other ways of spelling 'off'
        self.assertEqual(parse('--explain=OFF'), 'off')
        self.assertEqual(parse('--explain=none'), 'off')
        self.assertEqual(parse('--explain=False'), 'off')
        self.assertEqual(parse('--explain=no'), 'off')

    def test_parse_of_something_else(self):
        parse = mg_interface.parse_explain_mode
        self.assertEqual(parse('--save=NAME'), None)
        self.assertEqual(parse(''), None)
        # a wrong value is parsed, and refused by check_customize_model
        self.assertEqual(parse('--explain=bogus'), 'bogus')

    def test_check_accepts_the_valid_modes(self):
        cmd = mg_interface.MadGraphCmd()
        for args in [[], ['--explain'], ['--explain=life'], ['--explain=final'],
                     ['--explain=off'], ['--explain=OFF'],
                     ['--explain=live'], ['--save=NAME', '--explain=final'],
                     ['--explain', '--save=NAME']]:
            cmd.check_customize_model(list(args))

    def test_check_refuses_an_unknown_mode(self):
        cmd = mg_interface.MadGraphCmd()
        for args in [['--explain=bogus'], ['--explain=']]:
            self.assertRaises(mg_interface.MadGraph5Error,
                              cmd.check_customize_model, list(args))

    def test_the_last_one_wins(self):
        """'--explain --explain=final' asks for the final report"""

        resolve = mg_interface.resolve_explain_mode
        self.assertEqual(resolve(['--explain', '--explain=final']), 'final')
        self.assertEqual(resolve(['--explain=final', '--explain']), 'life')
        self.assertEqual(resolve(['--explain=final', '--explain=off']), None)

    def test_the_live_report_is_the_default(self):
        """the report used to be opt-in, it now has to be opt-out"""

        resolve = mg_interface.resolve_explain_mode
        self.assertEqual(resolve([]), 'life')
        self.assertEqual(resolve(['--save=NAME']), 'life')
        self.assertEqual(resolve(['--save=NAME', '--all']), 'life')

    def test_off_asks_for_no_report(self):
        """'off' is not a report mode, it is the absence of one"""

        resolve = mg_interface.resolve_explain_mode
        for arg in ['--explain=off', '--explain=OFF', '--explain=none',
                    '--explain=no', '--explain=False']:
            self.assertEqual(resolve(['--save=NAME', arg]), None, arg)


#===============================================================================
# the generic 'diagonal ckm' option
#===============================================================================
class ParamModel(dict):
    """a model with only external parameters, enough for the options which are
    derived from the blocks of the param_card"""

    def __init__(self, params):
        """params: (name, lhablock, lhacode, value)"""

        external, values = [], {}
        for name, lhablock, lhacode, value in params:
            param = FakeParam(name, lhablock, lhacode)
            param.value = value
            external.append(param)
            values[name] = value
        dict.__init__(self, {'particle_dict': {},
                             'parameters': {('external',): external},
                             'parameter_dict': values})

    def get(self, name):
        return self[name]


class TestCKMRules(unittest.TestCase):
    """The parameterisations of the quark mixing a SM-like model can use"""

    def test_wolfenstein(self):
        model = ParamModel([('lamWS', 'Wolfenstein', [1], 0.2253),
                            ('AWS', 'Wolfenstein', [2], 0.808),
                            ('rhoWS', 'Wolfenstein', [3], 0.132),
                            ('etaWS', 'Wolfenstein', [4], 0.341),
                            ('MZ', 'MASS', [23], 91.188)])
        self.assertEqual(build_restrict_lib.get_ckm_rules(model),
                         [('Wolfenstein', [1], 0.0), ('Wolfenstein', [2], 0.0),
                          ('Wolfenstein', [3], 0.0), ('Wolfenstein', [4], 0.0)])

    def test_ckm_matrix(self):
        """the SLHA2 convention: the matrix itself, 1 on the diagonal"""

        model = ParamModel([('RCKM1x1', 'VCKM', [1, 1], 1.0),
                            ('RCKM1x2', 'VCKM', [1, 2], 0.2253),
                            ('RCKM2x1', 'VCKM', [2, 1], -0.2253),
                            ('RCKM2x2', 'VCKM', [2, 2], 1.0)])
        self.assertEqual(sorted(build_restrict_lib.get_ckm_rules(model)),
                         sorted([('VCKM', [1, 1], 1.0), ('VCKM', [1, 2], 0.0),
                                 ('VCKM', [2, 1], 0.0), ('VCKM', [2, 2], 1.0)]))

    def test_cabibbo_angle(self):
        """the 2 generation models give a single mixing angle"""

        model = ParamModel([('cabi', 'CKMBLOCK', [1], 0.2277)])
        self.assertEqual(build_restrict_lib.get_ckm_rules(model),
                         [('CKMBLOCK', [1], 0.0)])

    def test_a_model_without_quark_mixing(self):
        model = ParamModel([('MZ', 'MASS', [23], 91.188)])
        self.assertEqual(build_restrict_lib.get_ckm_rules(model), [])
        self.assertEqual(build_restrict_lib.get_ckm_category(model, model), None)

    def test_default_follows_the_loaded_model(self):
        """the option starts checked when the model already has it applied"""

        mixing = ParamModel([('lamWS', 'Wolfenstein', [1], 0.2253)])
        diagonal = ParamModel([('lamWS', 'Wolfenstein', [1], 0.0)])
        rules = build_restrict_lib.get_ckm_rules(mixing)
        self.assertEqual(build_restrict_lib.is_already_applied(mixing, rules), False)
        self.assertEqual(build_restrict_lib.is_already_applied(diagonal, rules), True)

        category = build_restrict_lib.get_ckm_category(mixing, diagonal)
        self.assertEqual(category.name, 'quark mixing')
        self.assertEqual([rule.name for rule in category], ['diagonal ckm'])
        self.assertEqual(category[0].status, True)

    def test_a_removed_parameter_counts_as_applied(self):
        """a parameter which is not in the param_card anymore was fixed by the
        restriction the model was loaded with"""

        rules = [('Wolfenstein', [1], 0.0)]
        self.assertEqual(build_restrict_lib.is_already_applied(
                                    ParamModel([]), rules), True)


#===============================================================================
# explaining a restriction card which already exists
#===============================================================================
class TestCardRestrictionGroups(unittest.TestCase):
    """What explain_restriction attributes a removal to, when it is fed a card
    instead of the answers to the question."""

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.card = check_param_card.ParamCard("""
Block mass
    4 0.0 # mc
    5 2.0 # mb
    6 2.0 # mt
   23 1.0 # mz
Block yukawa
    5 2.0 # ymb
DECAY 6 0.0 # wt
""".split('\n'))
        self.externals = set([('mass', (4,)), ('mass', (5,)), ('mass', (6,)),
                              ('mass', (23,)), ('yukawa', (5,)), ('decay', (6,))])
        self.lha2name = {('mass', (4,)): 'MC', ('mass', (5,)): 'MB',
                         ('mass', (6,)): 'MT', ('mass', (23,)): 'MZ',
                         ('yukawa', (5,)): 'ymb', ('decay', (6,)): 'WT'}

    def test_groups(self):
        groups = self.cmd.get_card_restriction_groups(self.card, self.externals,
                                                      self.lha2name)
        labels = [label for label, keys in groups]
        # the parameters it sets to zero or one, one group each
        self.assertTrue('MC = 0' in labels)
        self.assertTrue('MZ = 1' in labels)
        self.assertTrue('WT = 0' in labels)
        # and the family which shares a value, as a single group
        self.assertTrue('MB = MT' in labels)
        # ymb has the same value but in another block: not the same family
        self.assertFalse([l for l in labels if 'ymb' in l])
        self.assertEqual(dict(groups)['MB = MT'],
                         set([('mass', (5,)), ('mass', (6,))]))

    def test_a_card_without_restriction(self):
        card = check_param_card.ParamCard("""
Block mass
    5 2.0 # mb
    6 3.0 # mt
""".split('\n'))
        self.assertEqual(self.cmd.get_card_restriction_groups(card,
                    set([('mass', (5,)), ('mass', (6,))]), self.lha2name), [])

    def test_only_the_external_parameters(self):
        self.assertEqual(self.cmd.get_card_restriction_groups(self.card, set(),
                                                    self.lha2name), [])

    def test_name_lookup_is_case_insensitive(self):
        """a UFO spells its blocks as it likes ('Wolfenstein' in the sm)"""

        lha2name = {('wolfenstein', (1,)): 'lamWS'}
        self.assertEqual(mg_interface.MadGraphCmd.name_of_lha(
                                        lha2name, ('Wolfenstein', (1,))), 'lamWS')
        self.assertEqual(mg_interface.MadGraphCmd.name_of_lha(
                                        lha2name, ('WOLFENSTEIN', (1,))), 'lamWS')
        # and an unknown one still says which entry it is
        self.assertEqual(mg_interface.MadGraphCmd.name_of_lha(
                                    lha2name, ('MASS', (5,))), 'MASS [5]')


class TestDisplayInTheQuestion(unittest.TestCase):
    """'display parameters/couplings' while answering the question"""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))

    def test_current_restrictions(self):
        """the display marks what the choices made so far do to a parameter"""

        self.ask.do_set_zero('ymt')
        self.ask.do_set_equal('MB MT')
        self.ask.do_set('flavourscheme 5F')
        current = self.ask.get_current_restrictions()
        self.assertEqual(current[('yukawa', (6,))], '0')
        # the commands win over the options, as they do on the card
        self.assertEqual(current[('mass', (5,))], 'MT')
        # and the options of the question are in there too
        self.assertTrue(('yukawa', (4,)) in current) # 5F: c is massless

    def test_display_does_not_crash(self):
        """the command is a read-only view: whatever the argument"""

        for line in ['parameters', 'parameters yukawa', 'parameters mb',
                     'parameters NotAParameter', 'couplings GC_1',
                     'couplings NotACoupling', 'parameters yt']:
            self.ask.do_display(line)
        # an invalid sub-command is refused, not raised
        self.ask.do_display('')
        self.ask.do_display('something')

    def test_completion(self):
        out = self.ask.complete_display('', 'display ', 8, 8)
        self.assertEqual(sorted(out), ['couplings', 'parameters'])
        out = self.ask.complete_display('', 'display parameters ', 19, 19)
        self.assertTrue('YUKAWA' in out)
        self.assertTrue('MB' in out)


#===============================================================================
# 'display couplings X' unfolds the definition down to the param_card
#===============================================================================
class TestExpandExpression(unittest.TestCase):
    """Following a coupling down to the parameters it is built from is what
    tells why the restriction drops it."""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))

    def test_reaches_the_restricted_parameter(self):
        """GC_15 = I1x33 = yb*conjugate(CKM3x3), and yb comes from ymb"""

        self.ask.do_set('flavourscheme 5F')
        found = self.ask.expand_expression(
                                self.ask.ufo_couplings['gc_15'].value)
        self.assertEqual([name for name, why in found], ['ymb'])
        self.assertTrue(found[0][1].startswith('0'))

    def test_nothing_restricted_on_that_path(self):
        """GC_1 is built from the electric charge only"""

        found = self.ask.expand_expression(
                                self.ask.ufo_couplings['gc_1'].value)
        self.assertEqual(found, [])

    def test_follows_a_set_zero(self):
        self.ask.do_set_zero('ymb')
        found = self.ask.expand_expression(
                                self.ask.ufo_couplings['gc_15'].value)
        self.assertEqual(found, [('ymb', '0')])

    def test_does_not_loop_on_a_repeated_parameter(self):
        """sw2 uses MW and MZ, MW uses MZ: MZ shows twice, MW is expanded once"""

        found = self.ask.expand_expression('sw2 + MW')
        self.assertEqual(found, [])

    def test_depth_is_bounded(self):
        """a pathological model must not blow the stack"""

        self.ask.internal_params['loopy'] = self.ask.internal_params['ee']
        try:
            self.ask.expand_expression('ee', depth=self.ask.EXPAND_MAX_DEPTH)
        finally:
            del self.ask.internal_params['loopy']

    def test_an_exact_name_wins_over_a_substring(self):
        """'display couplings GC_1' is about GC_1, not GC_1, GC_10, GC_100..."""

        self.assertTrue('gc_10' in self.ask.ufo_couplings)
        # nothing to assert on the output, but the command must not list them
        # all: the exact match is the one which is expanded
        self.ask.do_display('couplings GC_1')
        self.ask.do_display('parameters MB')


#===============================================================================
# explain_restriction picks the right card
#===============================================================================
class TestRestrictionToExplain(unittest.TestCase):
    """Which (model, card) pair 'explain_restriction ARG' resolves to"""

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()

    def test_a_model_name(self):
        model_path, card = self.cmd.get_restriction_to_explain(['sm-ckm'])
        self.assertEqual(os.path.basename(model_path), 'sm')
        self.assertEqual(os.path.basename(card), 'restrict_ckm.dat')

    def test_an_unknown_name(self):
        self.assertRaises(mg_interface.MadGraph5Error,
                          self.cmd.get_restriction_to_explain, ['NotAModel'])

    def test_no_model_loaded(self):
        self.assertRaises(mg_interface.MadGraph5Error,
                          self.cmd.get_restriction_to_explain, [])

    def test_a_card_which_is_not_on_disk_anymore(self):
        """customize_model builds the model from a temporary card which it then
        removes: the command must say so instead of failing on the open()"""

        class FakeModel(dict):
            restrict_card = '/does/not/exist/restrict_gone.dat'
            def get(self, name):
                return self[name]
        self.cmd._curr_model = FakeModel({'modelpath': 'whatever'})
        self.assertRaises(mg_interface.MadGraph5Error,
                          self.cmd.get_restriction_to_explain, [])

    def test_the_card_of_the_current_model(self):
        sm_path = import_ufo.find_ufo_path('sm')
        self.cmd._curr_model = import_ufo.import_model(sm_path)
        model_path, card = self.cmd.get_restriction_to_explain([])
        self.assertEqual(os.path.basename(card), 'restrict_default.dat')
        self.assertTrue(os.path.isfile(card))


#===============================================================================
# --all: every coupling instead of the first few
#===============================================================================
class TestFullListing(unittest.TestCase):
    """The lists of couplings of a report are truncated unless --all is given"""

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.names = ['GC_%d' % i for i in [13, 9, 101, 2, 47, 8, 300, 5,
                                            22, 1, 76, 4]]

    def test_natural_order(self):
        """GC_9 before GC_10, not after GC_101"""

        self.assertEqual(mg_interface.natural_key('GC_9') <
                         mg_interface.natural_key('GC_10'), True)
        self.assertEqual(self.cmd.short_list(['GC_10', 'GC_9'], None),
                         'GC_9, GC_10')

    def test_truncated_by_default(self):
        out = self.cmd.short_list(self.names)
        self.assertTrue(out.endswith('... (4 more)'))
        self.assertEqual(out.count('GC_'), 8)

    def test_all_of_them(self):
        out = self.cmd.short_list(self.names, None)
        self.assertFalse('more)' in out)
        self.assertEqual(out.count('GC_'), len(self.names))
        # and in natural order
        self.assertTrue(out.startswith('GC_1, GC_2, GC_4, GC_5, GC_8, GC_9'))

    def test_wrap(self):
        """a long list is cut into lines, keeping every name"""

        lines = self.cmd.wrap_list(self.names, width=30)
        self.assertTrue(len(lines) > 1)
        for line in lines:
            self.assertTrue(len(line) <= 32, line)
        self.assertEqual(sorted(', '.join(lines).split(', ')),
                         sorted(self.names))

    def test_wrap_of_nothing(self):
        self.assertEqual(self.cmd.wrap_list([]), [''])

    def test_the_flag_is_accepted(self):
        for args in [['--all', 'sm-ckm'], ['sm-ckm', '--all']]:
            self.cmd.check_explain_restriction(list(args))
        # '--all' is not a target: alone it still needs a model to explain
        self.assertRaises(mg_interface.MadGraph5Error,
                    self.cmd.check_explain_restriction, ['--all'])
        # and two targets are still refused
        self.assertRaises(mg_interface.MadGraph5Error,
                    self.cmd.check_explain_restriction, ['sm-ckm', 'sm', '--all'])

    def test_completion_shares_the_prefix(self):
        """the readline wrapper strips the common prefix off every candidate,
        so one which does not start with the typed text comes back mangled"""

        self.cmd._curr_model = import_ufo.import_model(
                                            import_ufo.find_ufo_path('sm'))
        for text, begidx in [('loop_sm-', 19), ('loop_', 20), ('', 20),
                             ('restrict_c', 20), ('--', 20)]:
            line = 'explain_restriction %s' % text
            out = self.cmd.complete_explain_restriction(text, line, begidx,
                                                        len(line))
            for name in out:
                self.assertTrue(name.startswith(text),
                                '%r proposed for %r' % (name, text))
        # the option is proposed, and a model restriction is found
        self.assertTrue('--all' in
                self.cmd.complete_explain_restriction('--', 'explain_restriction --',
                                                      20, 22))
        self.assertTrue('loop_sm-ckm' in
                self.cmd.complete_explain_restriction('loop_sm-',
                        'explain_restriction loop_sm-', 19, 28))

    def test_customize_model_accepts_it_too(self):
        """it is the same report, so it takes the same flag"""

        self.cmd.check_customize_model(['--explain=final', '--all'])
        self.cmd.check_customize_model(['--all'])


#===============================================================================
# 'set default': where the values of the param_card come from
#===============================================================================
class TestSetDefault(unittest.TestCase):
    """The parameters keep the value they have in the model which was loaded,
    unless the user points at a card or sets one by hand."""

    @classmethod
    def setUpClass(cls):
        cls.sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(cls.sm_path)

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))
        self.card = os.path.join(self.sm_path, 'restrict_ckm.dat')

    def value_of(self, name):
        return self.ask.get_default_value(self.ask.external_params[name])

    def test_nothing_asked(self):
        """MT is 172 in the UFO and 173 in the restriction cards"""

        self.assertEqual(self.ask.default_source, None)
        self.assertEqual(self.value_of('mt'), 172.)

    def test_a_card(self):
        self.ask.do_set('default %s' % self.card)
        self.assertEqual(self.ask.default_source, self.card)
        self.assertEqual(self.value_of('mt'), 173.)
        # a parameter the card does not define keeps the value it had
        self.assertEqual(self.value_of('mh'), 125.)

    def test_one_parameter(self):
        for line in ['default MH = 130', 'default MH 130']:
            self.ask.do_set('default UFO')
            self.ask.do_set(line)
            self.assertEqual(self.value_of('mh'), 130., line)

    def test_a_parameter_wins_over_the_card(self):
        self.ask.do_set('default %s' % self.card)
        self.ask.do_set('default MT = 171')
        self.assertEqual(self.value_of('mt'), 171.)

    def test_ufo_starts_from_scratch(self):
        """'set default UFO' undoes every 'set default' entered before it"""

        self.ask.do_set('default %s' % self.card)
        self.ask.do_set('default MH = 130')
        self.ask.do_set('default UFO')
        self.assertEqual(self.ask.default_source, 'ufo')
        self.assertEqual(self.ask.default_values, {})
        self.assertEqual(self.value_of('mt'), 172.)
        self.assertEqual(self.value_of('mh'), 125.)

    def test_a_card_starts_from_scratch_too(self):
        self.ask.do_set('default MH = 130')
        self.ask.do_set('default %s' % self.card)
        self.assertEqual(self.ask.default_values, {})
        self.assertEqual(self.value_of('mh'), 125.)

    def test_refused_inputs(self):
        self.ask.do_set('default /not/a/file.dat')
        self.assertEqual(self.ask.default_source, None)
        self.ask.do_set('default NotAParameter = 1')
        self.ask.do_set('default MH = not_a_number')
        self.ask.do_set('default')
        self.assertEqual(self.ask.default_values, {})

    def test_clear_forgets_them(self):
        self.ask.do_set('default %s' % self.card)
        self.ask.do_set('default MH = 130')
        self.ask.do_clear('')
        self.assertEqual(self.ask.default_source, None)
        self.assertEqual(self.ask.default_values, {})

    def test_the_card_is_built_with_them(self):
        """build_default_card is what puts them in the param_card"""

        baseline = self.cmd.get_full_param_card(self.full_model)
        self.ask.do_set('default %s' % self.card)
        self.ask.do_set('default MH = 130')
        param_card = self.cmd.build_default_card(self.full_model, baseline,
                                                 self.ask)
        self.assertEqual(param_card['mass'].get([6]).value, 173.)
        self.assertEqual(param_card['mass'].get([25]).value, 130.)

    def test_ufo_ignores_the_baseline(self):
        baseline = self.cmd.get_full_param_card(self.full_model)
        baseline['mass'].get([6]).value = 999.
        self.ask.do_set('default UFO')
        param_card = self.cmd.build_default_card(self.full_model, baseline,
                                                 self.ask)
        self.assertEqual(param_card['mass'].get([6]).value, 172.)


#===============================================================================
# the values propagated from the model which was loaded
#===============================================================================
class TestLoadedDefaults(unittest.TestCase):
    """customize_model starts from the values of the model as it was loaded,
    and falls back to the UFO for the parameters that model had fixed."""

    @classmethod
    def setUpClass(cls):
        cls.sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(cls.sm_path)

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.externals = self.cmd.get_external_lhacode(self.full_model)

    def test_the_values_of_the_restriction_are_propagated(self):
        """sm-ckm has MT = 173 while the UFO has 172"""

        reference = import_ufo.import_model(self.sm_path + '-ckm')
        card, recovered = self.cmd.get_loaded_default_card(self.full_model,
                                                reference, self.externals)
        self.assertEqual(card['mass'].get([6]).value, 173.)

    def test_a_parameter_that_restriction_had_fixed(self):
        """restrict_ckm.dat sets MC and WTau to zero, so they are gone from the
        model: there is no value to propagate and the UFO one is used"""

        reference = import_ufo.import_model(self.sm_path + '-ckm')
        card, recovered = self.cmd.get_loaded_default_card(self.full_model,
                                                reference, self.externals)
        self.assertTrue(('mass', (4,)) in recovered)
        self.assertTrue(('decay', (15,)) in recovered)
        self.assertFalse(('mass', (6,)) in recovered)
        # and they do keep the value of the UFO
        self.assertEqual(card['mass'].get([4]).value,
                         self.cmd.get_full_param_card(
                                self.full_model)['mass'].get([4]).value)

    def test_an_unrestricted_reference(self):
        """nothing was fixed, so nothing has to be recovered"""

        card, recovered = self.cmd.get_loaded_default_card(self.full_model,
                                            self.full_model, self.externals)
        self.assertEqual(recovered, [])
        self.assertEqual(card['mass'].get([6]).value, 172.)

    def test_no_reference(self):
        card, recovered = self.cmd.get_loaded_default_card(self.full_model,
                                                    None, self.externals)
        self.assertEqual(recovered, [])
        self.assertEqual(card['mass'].get([6]).value, 172.)


#===============================================================================
# 'BLOCK all': act on every parameter of a block of the param_card
#===============================================================================
class TestBlockAssignment(unittest.TestCase):
    """'set decay all 0', 'set_zero yukawa all', 'set default CEFT all 0'"""

    @classmethod
    def setUpClass(cls):
        cls.full_model = import_ufo.import_full_model(
                                            import_ufo.find_ufo_path('sm'))

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.cmd._curr_model = self.full_model
        self.ask = mg_interface.AskforCustomize('', mother_interface=self.cmd,
            categories=self.cmd.get_customize_categories(self.full_model,
                                                         self.full_model))
        self.yukawa = set(key for key in self.ask.lha2name if key[0] == 'yukawa')

    def test_resolve_a_name(self):
        keys, rest = self.ask.resolve_parameters(['MB', '0'])
        self.assertEqual(keys, [('MASS', (5,))])
        self.assertEqual(rest, ['0'])

    def test_resolve_a_block(self):
        keys, rest = self.ask.resolve_parameters(['yukawa', 'all', '0'])
        self.assertEqual(set((b.lower(), c) for b, c in keys), self.yukawa)
        self.assertEqual(rest, ['0'])
        # the block name is case insensitive
        self.assertEqual(len(self.ask.resolve_parameters(['YUKAWA', 'all'])[0]),
                         len(self.yukawa))

    def test_resolve_an_unknown_block(self):
        keys, rest = self.ask.resolve_parameters(['NotABlock', 'all', '0'])
        self.assertEqual(keys, None)
        self.assertEqual(rest, ['0'])

    def test_set_zero_of_a_block(self):
        self.ask.do_set_zero('yukawa all')
        self.assertEqual(set((b.lower(), c) for b, c in self.ask.set_zero),
                         self.yukawa)

    def test_set_one_of_a_block(self):
        self.ask.do_set_one('yukawa all')
        self.assertEqual(set((b.lower(), c) for b, c in self.ask.set_one),
                         self.yukawa)
        self.assertEqual(self.ask.set_zero, [])

    def test_the_set_alias(self):
        self.ask.do_set('decay all 0')
        self.assertTrue(self.ask.set_zero)
        for block, code in self.ask.set_zero:
            self.assertEqual(block.lower(), 'decay')

    def test_set_default_of_a_block(self):
        self.ask.do_set('default wolfenstein all 0.5')
        self.assertEqual(len(self.ask.default_values), 4)
        for value in self.ask.default_values.values():
            self.assertEqual(value, 0.5)
        self.assertEqual(self.ask.get_default_value(
                            self.ask.external_params['lamws']), 0.5)

    def test_a_block_can_not_be_set_equal(self):
        self.ask.do_set('yukawa all MB')
        self.assertEqual(self.ask.set_equal, [])
        self.assertEqual(self.ask.set_zero, [])

    def test_a_block_replaces_a_single_parameter(self):
        """the last command on a parameter is the one which counts"""

        self.ask.do_set_one('ymb')
        self.ask.do_set_zero('yukawa all')
        self.assertEqual(self.ask.set_one, [])
        self.assertEqual(len(self.ask.set_zero), len(self.yukawa))

    def test_the_question_groups_them_back(self):
        self.ask.do_set_zero('yukawa all')
        self.ask.do_set('default wolfenstein all 0.5')
        question = self.ask.get_question()
        self.assertTrue('YUKAWA all = 0' in question)
        self.assertTrue('Wolfenstein all = 0.5' in question)
        # a single parameter is still named
        self.ask.do_set_one('MB')
        self.assertTrue('MB (MASS [5]) = 1' in self.ask.get_question())

    def test_completion(self):
        out = self.ask.complete_set_zero('', 'set_zero ', 9, 9)
        self.assertTrue('YUKAWA' in out)
        self.assertTrue('ymb' in out)
        self.assertEqual(self.ask.complete_set_zero('', 'set_zero yukawa ',
                                                    16, 16), ['all'])


#===============================================================================
# which parameters can not take the value of the model which was loaded
#===============================================================================
class TestRecoveredDefaults(unittest.TestCase):
    """A parameter the loaded restriction had fixed has no value to propagate,
    whether that restriction removed it from the param_card or -- SLHA2 -- kept
    it in there."""

    @classmethod
    def setUpClass(cls):
        cls.sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(cls.sm_path)

    def setUp(self):
        self.cmd = mg_interface.MadGraphCmd()
        self.externals = self.cmd.get_external_lhacode(self.full_model)

    def test_a_kept_but_fixed_parameter(self):
        """an SLHA2 restriction keeps what it fixed in the param_card, so
        'not external anymore' does not catch it"""

        reference = import_ufo.import_model(self.sm_path + '-ckm')
        # pretend the restriction had kept MZ at one, as an SLHA2 one would
        kept = self.cmd.get_external_lhacode(reference)
        self.assertTrue(('mass', (23,)) in kept)
        loaded = self.cmd.get_full_param_card(reference)
        loaded['mass'].get([23]).value = 1.

        dropped, merged = self.cmd.get_parameter_classes(loaded, kept)
        self.assertTrue(('mass', (23,)) in dropped)

    def test_the_warning_skips_what_set_default_gave(self):
        """a parameter the user gave a value to did not fall back on the UFO"""

        class FakeAsk(object):
            default_values = {('DECAY', (15,)): 1e-12}
            default_card_values = None

        param_card = self.cmd.get_full_param_card(self.full_model)
        recovered = [('decay', (15,)), ('mass', (4,))]
        # nothing raises, and only the one which was not given is reported
        self.cmd.warn_recovered_defaults(recovered, set(), param_card,
                    self.sm_path, 'restrict_ckm.dat', FakeAsk())

    def test_the_warning_skips_what_the_card_defines(self):
        class FakeAsk(object):
            default_values = {}
        FakeAsk.default_card_values = check_param_card.ParamCard(
                    os.path.join(self.sm_path, 'restrict_ckm.dat'))

        param_card = self.cmd.get_full_param_card(self.full_model)
        self.cmd.warn_recovered_defaults([('decay', (15,))], set(), param_card,
                    self.sm_path, 'restrict_ckm.dat', FakeAsk())


#===============================================================================
# the question starts from the restriction the model was loaded with
#===============================================================================
class TestStartFromTheLoadedRestriction(unittest.TestCase):
    """customize_model used to start from the full model with the values of
    the loaded one, so a parameter the loaded restriction had fixed came back
    at its UFO value: customizing SMEFTatNLO-NLO and answering `done` gave 2013
    interactions against the 1932 loaded, operators, masses and widths
    switched back on.  The rules now start as that restriction."""

    @classmethod
    def setUpClass(cls):
        sm_path = import_ufo.find_ufo_path('sm')
        cls.full_model = import_ufo.import_full_model(sm_path)

    def ask(self, card, set_scheme=None):
        cmd = mg_interface.MadGraphCmd()
        cmd._curr_model = self.full_model
        categories = cmd.get_customize_categories(self.full_model,
                                                  self.full_model)
        if set_scheme:
            for category in categories:
                for option in category:
                    if option.name == 'flavour scheme':
                        option.status = set_scheme
        return mg_interface.AskforCustomize(
            '', mother_interface=cmd, categories=categories,
            loaded_card=check_param_card.ParamCard(card.split('\n')),
            loaded_name='test')

    CARD = """
Block mass
    5 0.0 # MB
    6 172.0 # MT
   15 1.777 # MTA
   23 91.188 # MZ
Block yukawa
    5 0.0 # ymb
    6 172.0 # ymt
   15 1.777 # ymtau
DECAY 6 0.0 # WT
DECAY 23 1.0 # WZ
"""

    def test_zeros_ones_and_merges_become_rules(self):
        ask = self.ask(self.CARD)
        self.assertIn(('MASS', (5,)), ask.set_zero)
        self.assertIn(('YUKAWA', (5,)), ask.set_zero)
        self.assertIn(('DECAY', (6,)), ask.set_zero)     # a zero width stays
        self.assertIn(('DECAY', (23,)), ask.set_one)
        # the same value inside a block is a merge; ymt = MT is across two
        # blocks, which the restriction does not merge
        self.assertEqual(ask.set_equal, [])
        self.assertIn('test', ask.get_question())

    def test_a_merge_inside_a_block(self):
        card = self.CARD.replace('15 1.777 # MTA', '15 172.0 # MTA')
        ask = self.ask(card)
        self.assertEqual(ask.set_equal, [(('MASS', (15,)), ('MASS', (6,)))])

    def test_opposite_signs_are_left_free(self):
        """`set A = B` has no sign: a pair the restriction merged as A = -B
        cannot be written as a rule"""

        card = self.CARD.replace('15 1.777 # MTA', '15 -172.0 # MTA')
        self.assertEqual(self.ask(card).set_equal, [])

    def test_what_an_option_covers_is_left_to_it(self):
        """a 5F scheme zeroes the b mass itself: not listed twice"""

        ask = self.ask(self.CARD, set_scheme='5F')
        self.assertNotIn(('MASS', (5,)), ask.set_zero)
        self.assertIn(('DECAY', (6,)), ask.set_zero)

    def test_set_free_releases_one(self):
        ask = self.ask(self.CARD)
        ask.do_set('MB free')
        self.assertNotIn(('MASS', (5,)), ask.set_zero)
        self.assertIn(('YUKAWA', (5,)), ask.set_zero)    # the others stay

    def test_without_a_card_nothing_is_prefilled(self):
        cmd = mg_interface.MadGraphCmd()
        cmd._curr_model = self.full_model
        ask = mg_interface.AskforCustomize(
            '', mother_interface=cmd,
            categories=cmd.get_customize_categories(self.full_model,
                                                    self.full_model))
        self.assertEqual((ask.set_zero, ask.set_one, ask.set_equal),
                         ([], [], []))


class TestCustomizeKeepsTheLoadedRestriction(unittest.TestCase):
    """End to end: from sm-no_widths, answering `done` keeps every width at
    zero -- before, they all came back at the values of the UFO."""

    def test_done_rebuilds_the_loaded_model(self):
        import tempfile
        import madgraph.interface.master_interface as master_interface

        cmd = master_interface.MasterCmd()
        cmd.options['crash_on_error'] = False
        cmd.exec_cmd('import model sm-no_widths', printcmd=False)
        loaded = len(cmd._curr_model.get('interactions'))

        handle, path = tempfile.mkstemp(suffix='.mg5')
        try:
            with os.fdopen(handle, 'w') as script:
                script.write('customize_model\ndone\n')
            cmd.exec_cmd('import command %s' % path, printcmd=False)
        finally:
            os.remove(path)

        widths = [param for param in cmd._curr_model.get('parameters').get(
                                                          ('external',), [])
                  if param.lhablock.upper() == 'DECAY']
        self.assertEqual(widths, [], 'a width came back as a free parameter')
        self.assertEqual(len(cmd._curr_model.get('interactions')), loaded)

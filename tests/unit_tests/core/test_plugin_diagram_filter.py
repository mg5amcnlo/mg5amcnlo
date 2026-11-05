################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
################################################################################
"""Tests of the diagram filters provided by plugins: the plugin declares
    new_diagram_filter = remove_diag
and it is selected with --diagram_filter=PLUGINNAME (tree and MadLoop
generation) or --tree_diagram_filter / --loop_diagram_filter=PLUGINNAME (NLO
FKS generation)."""

from __future__ import absolute_import

import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.core.diagram_generation as diagram_generation
import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph import InvalidCmd, MadGraph5Error

NoDiagram = diagram_generation.NoDiagramException

HEADER = """
minimal_mg7_version = (0, 0, 0)
maximal_mg7_version = (99, 0, 0)
latest_validated_mg7_version = (99, 0, 0)
"""

KEEP_ALL = "def remove_diag(diag, model=None):\n    return False\n"
# keeps everything but writes the type of each diagram in the file 'record'
# of the plugin, also from the worker processes of a multicore generation
RECORD_ALL = """import os
def remove_diag(diag, model=None):
    with open(os.path.join(os.path.dirname(__file__), 'record'), 'a') as fsock:
        fsock.write('%s\\n' % type(diag).__name__)
    return False
"""
REMOVE_ALL = "def remove_diag(diag, model=None):\n    return True\n"


class PluginDiagramFilterTest(unittest.TestCase):
    """A temporary plugin directory is put in sys.path; each test writes the
    plugins it needs with add_plugin."""

    counter = 0

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        PluginDiagramFilterTest.counter += 1
        # a different package name each time, python caches the imports
        self.dirname = 'TESTPLUGINDIR%d' % PluginDiagramFilterTest.counter
        self.plugdir = os.path.join(self.tmp, self.dirname)
        os.mkdir(self.plugdir)
        open(os.path.join(self.plugdir, '__init__.py'), 'w').close()
        sys.path.insert(0, self.tmp)

    def tearDown(self):
        sys.path.remove(self.tmp)
        for name in list(misc.plugin_support):
            if name.startswith(self.dirname):
                del misc.plugin_support[name]
        for name in list(sys.modules):
            if name.startswith(self.dirname):
                del sys.modules[name]
        shutil.rmtree(self.tmp)

    def add_plugin(self, name, filter_code=None, init_text=None):
        """plugin 'name' whose new_diagram_filter is remove_diag of
        filter_code (or init_text, the full __init__.py, if given)"""
        path = os.path.join(self.plugdir, name)
        os.mkdir(path)
        if init_text is None:
            with open(os.path.join(path, 'filt.py'), 'w') as fsock:
                fsock.write(filter_code)
            init_text = 'from .filt import remove_diag\n' \
                        'new_diagram_filter = remove_diag\n'
        with open(os.path.join(path, '__init__.py'), 'w') as fsock:
            fsock.write(HEADER + init_text)
        # the version check of the plugin is not what is tested here
        misc.plugin_support['%s.%s' % (self.dirname, name)] = True

    def record(self, name):
        """types of the diagrams seen by the RECORD_ALL plugin 'name'"""
        path = os.path.join(self.plugdir, name, 'record')
        if not os.path.exists(path):
            return []
        return [line.split()[0] for line in open(path)]

    def interface(self, model):
        interface = MGCmd.MasterCmd()
        interface.plugin_path = [self.plugdir]
        interface.exec_cmd('import model %s' % model, printcmd=False,
                           precmd=True)
        return interface

    def generate(self, line, model='sm'):
        interface = self.interface(model)
        interface.exec_cmd('generate %s' % line, printcmd=False, precmd=True,
                           errorhandling=False)
        return interface

    def test_plugin_diagram_filter_lookup(self):
        """the filter of the plugin of that name is returned, the names of the
        plugins providing one are listed"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('nofilter', init_text='x = 1\n')
        fct = misc.plugin_diagram_filter([self.plugdir], 'keepall')
        self.assertTrue(callable(fct))
        self.assertFalse(fct(None))
        self.assertEqual(misc.plugin_diagram_filter([self.plugdir], None),
                         ['keepall'])

    def test_plugin_diagram_filter_errors(self):
        """unknown plugin, plugin without filter, unsupported plugin,
        non-callable filter and empty name raise, with a message telling
        which"""
        self.add_plugin('nofilter', init_text='x = 1\n')
        self.add_plugin('notfunction', init_text='new_diagram_filter = "x"\n')
        self.add_plugin('unsupported', KEEP_ALL)
        misc.plugin_support['%s.unsupported' % self.dirname] = False
        for name, msg in [('unknown', 'No plugin'),
                          ('nofilter', 'does not provide'),
                          ('unsupported', 'not supported'),
                          ('notfunction', 'must be a function'),
                          ('', 'No plugin name')]:
            try:
                misc.plugin_diagram_filter([self.plugdir], name)
            except MadGraph5Error as error:
                self.assertIn(msg, str(error))
            else:
                self.fail('no error for plugin %r' % name)
        # none of them is listed
        self.assertEqual(misc.plugin_diagram_filter([self.plugdir], None), [])

    def test_tree_level(self):
        """--diagram_filter=PLUGINNAME removes the diagrams selected by the
        plugin function (here all or none)"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        interface = self.generate('u u~ > d d~ --diagram_filter=keepall')
        self.assertTrue(len(interface._curr_amps[0].get("diagrams")) > 0)
        interface = self.generate('u u~ > d d~')
        self.assertTrue(len(interface._curr_amps[0].get('diagrams')) > 0)
        self.assertRaises(NoDiagram, self.generate,
                          'u u~ > d d~ --diagram_filter=removeall')

    def test_tree_level_invalid_name(self):
        self.add_plugin('nofilter', init_text='x = 1\n')
        for name in ['unknown', 'nofilter', '']:
            self.assertRaises(InvalidCmd, self.generate,
                              'u u~ > d d~ --diagram_filter=%s' % name)

    def test_tree_level_filter_receives_diagrams(self):
        """the function is called with the diagrams and the model"""
        calls = []
        def record(diag, model=None):
            calls.append((diag, model))
            return False
        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.base_objects as base_objects
        amp = diagram_generation.Amplitude()
        amp.set('process', base_objects.Process())
        diags = base_objects.DiagramList([base_objects.Diagram(),
                                          base_objects.Diagram()])
        res = amp.apply_user_filter(diags, record)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(res), 2)
        self.assertRaises(MadGraph5Error, amp.apply_user_filter, diags,
                          'not a function')

    def test_loop_level(self):
        """--diagram_filter=PLUGINNAME is applied to the loop diagrams of a
        MadLoop process"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        interface = self.generate('g g > h [virt=QCD] '
                                  '--diagram_filter=keepall', 'loop_sm')
        nloops = len(interface._curr_amps[0].get('loop_diagrams'))
        self.assertTrue(nloops > 0)
        # no loop diagram left: no amplitude
        self.assertRaises(NoDiagram, self.generate, 'g g > h [virt=QCD] '
                                  '--diagram_filter=removeall', 'loop_sm')

    def test_loop_level_multiparticle(self):
        """the filter is kept when a MadLoop multiprocess is split into its
        processes"""
        self.add_plugin('rec', RECORD_ALL)
        interface = self.generate('p p > h [virt=QCD] --diagram_filter=rec',
                                  'loop_sm')
        self.assertTrue(interface._curr_amps)
        self.assertIn('LoopDiagram', self.record('rec'))

    def test_loop_induced(self):
        """... and of a loop-induced process, also when reached from the NLO
        mode (no Born diagram), with --loop_diagram_filter"""
        self.add_plugin('removeall', REMOVE_ALL)
        self.add_plugin('rec', RECORD_ALL)
        self.assertRaises(NoDiagram, self.generate,
                          'g g > h [noborn=QCD] --diagram_filter=removeall',
                          'loop_sm')
        self.generate('g g > h QED=1 [QCD] --loop_diagram_filter=rec',
                      'loop_sm')
        self.assertIn('LoopDiagram', self.record('rec'))

    def test_fks_tree_and_loop_filters(self):
        """--tree_diagram_filter acts on the Born (and real) diagrams and
        --loop_diagram_filter on the loop diagrams only; --diagram_filter is
        refused"""
        self.add_plugin('keepall', KEEP_ALL)
        self.add_plugin('removeall', REMOVE_ALL)
        proc = 'u u~ > d d~ [QCD] %s'

        interface = self.generate(proc % ('--tree_diagram_filter=keepall '
                                          '--loop_diagram_filter=keepall'),
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        nvirt = sum(len(amp.get('loop_diagrams'))
                    for amp in fksmulti.get_virt_amplitudes())
        nborn = sum(len(amp.get('diagrams'))
                    for amp in fksmulti.get_born_amplitudes())
        nreal = sum(len(amp.get('diagrams'))
                    for amp in fksmulti.get_real_amplitudes())
        self.assertTrue(nvirt > 0 and nborn > 0 and nreal > 0)

        # no loop diagram left, the tree-level ones are untouched
        interface = self.generate(proc % '--loop_diagram_filter=removeall',
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        self.assertEqual(sum(len(amp.get('loop_diagrams'))
                         for amp in fksmulti.get_virt_amplitudes()), 0)
        self.assertEqual(sum(len(amp.get('diagrams'))
                         for amp in fksmulti.get_born_amplitudes()), nborn)
        self.assertEqual(sum(len(amp.get('diagrams'))
                         for amp in fksmulti.get_real_amplitudes()), nreal)

        # no Born diagram left: the switch to the loop-induced mode refuses a
        # tree-level filter (or, without coupling orders, their automatic
        # determination fails)
        self.assertRaises(InvalidCmd, self.generate,
                          'u u~ > d d~ QCD=2 [QCD] --tree_diagram_filter=removeall',
                          'loop_sm')
        self.assertRaises(MadGraph5Error, self.generate,
                          proc % '--tree_diagram_filter=removeall', 'loop_sm')

        # the loop filter does not touch the loop diagrams of a tree-level
        # filter alone
        interface = self.generate(proc % '--tree_diagram_filter=keepall',
                                  'loop_sm')
        fksmulti = interface._fks_multi_proc
        self.assertEqual(sum(len(amp.get('loop_diagrams'))
                         for amp in fksmulti.get_virt_amplitudes()), nvirt)

        self.assertRaises(InvalidCmd, self.generate,
                          proc % '--diagram_filter', 'loop_sm')

    def test_fks_multicore_virtual(self):
        """with low_mem_multicore_nlo_generation the virtual is generated in
        the worker processes (at output), with the loop filter"""
        self.add_plugin('rec', RECORD_ALL)
        interface = self.interface('loop_sm')
        interface.exec_cmd('set low_mem_multicore_nlo_generation True',
                           printcmd=False, precmd=True)
        interface.exec_cmd('set nb_core 2', printcmd=False, precmd=True)
        interface.exec_cmd('generate u u~ > d d~ [QCD] '
                           '--loop_diagram_filter=rec', printcmd=False,
                           precmd=True, errorhandling=False)
        # nothing seen at generation, the virtual is not generated yet
        self.assertNotIn('LoopDiagram', self.record('rec'))
        interface.exec_cmd('output %s' % os.path.join(self.tmp, 'OUT'),
                           printcmd=False, precmd=True, errorhandling=False)
        self.assertIn('LoopDiagram', self.record('rec'))

    def test_plugin_version_attributes(self):
        """a plugin gives its MadGraph7 version range with minimal_mg7_version
        and maximal_mg7_version; the MG5_aMC@NLO attributes alone are accepted
        without check; a plugin with neither (or only one bound) is
        discarded"""
        import types
        def plugin(name, **attributes):
            module = types.ModuleType('%s.%s' % (self.dirname, name))
            for key, value in attributes.items():
                setattr(module, key, value)
            return module
        cases = [
            ('mg7', dict(minimal_mg7_version=(0, 0, 0),
                         maximal_mg7_version=(1000, 0, 0)), True),
            ('mg5only', dict(minimal_mg5amcnlo_version=(3, 6, 0),
                             maximal_mg5amcnlo_version=(1000, 0, 0),
                             latest_validated_version=(3, 6, 0)), True),
            ('onebound', dict(minimal_mg7_version=(0, 0, 0)), False),
            ('noversion', dict(), False)]
        for name, attributes, expected in cases:
            self.assertEqual(bool(misc.is_plugin_supported(
                                        plugin(name, **attributes))), expected,
                             name)

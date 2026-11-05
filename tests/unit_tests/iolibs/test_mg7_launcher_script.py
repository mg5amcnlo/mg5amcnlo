##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
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
"""bin/generate_events of an mg7 output: which python actually runs it.

A run needs the packages of the environment MadGraph7 was started from
(matplotlib for the plots, the LHAPDF bindings, madspace); the "python3" the
PATH resolves to often has none of them. The launcher therefore pins that
interpreter, and does not trust the shebang to be what gets used.
"""

from __future__ import absolute_import
import os
import stat
import subprocess
import sys
import tempfile
import shutil
import unittest

import madgraph.iolibs.export_cpp as export_cpp


class TestMG7LauncherScript(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='test_mg7_launcher')

    def tearDown(self):
        shutil.rmtree(self.path)

    def write(self, interpreter, body=None):
        """The generated launcher, with its import of the real launcher
        replaced by `body` so that the test does not need madspace."""
        source = export_cpp.mg7_launcher_source(interpreter, '/nowhere')
        if body is not None:
            head, _sep, _tail = source.partition(
                "sys.path.append('/nowhere')")
            source = head + body
        script = os.path.join(self.path, 'generate_events')
        with open(script, 'w') as stream:
            stream.write(source)
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        return script

    def test_the_shebang_is_the_running_interpreter(self):
        source = export_cpp.mg7_launcher_source('/opt/py/bin/python3', '/opt/MG5')
        self.assertTrue(source.startswith('#! /opt/py/bin/python3\n'))
        self.assertIn("_INTERPRETER = '/opt/py/bin/python3'", source)
        self.assertIn("sys.path.append('/opt/MG5')", source)

    def test_no_interpreter_falls_back_on_env(self):
        """nothing to pin (an embedded interpreter): the old behaviour"""
        source = export_cpp.mg7_launcher_source('', '/opt/MG5')
        self.assertTrue(source.startswith('#! /usr/bin/env python3\n'))
        self.assertIn("_INTERPRETER = ''", source)

    def test_symlinks_are_not_resolved(self):
        """a virtualenv's python is a symlink to the base interpreter, which
        does not see the environment's site-packages"""
        base = os.path.join(self.path, 'base_python')
        with open(base, 'w') as stream:
            stream.write('')
        link = os.path.join(self.path, 'venv_python')
        os.symlink(base, link)
        self.assertIn('_INTERPRETER = %r' % link,
                      export_cpp.mg7_launcher_source(link, '/opt/MG5'))

    def test_a_foreign_interpreter_is_replaced(self):
        """started with another python, the script re-execs through the
        pinned one -- this is what the shebang alone cannot guarantee"""
        script = self.write(sys.executable,
                            'print(sys.executable)\n')
        # run it through a python that is *not* the pinned one: a copy of the
        # same binary is a different path, so the re-exec has to happen
        other = shutil.which('python3') or sys.executable
        if os.path.normpath(other) == os.path.normpath(sys.executable):
            other = os.path.realpath(sys.executable)
        if os.path.normpath(other) == os.path.normpath(sys.executable):
            self.skipTest('no second interpreter to start the script with')
        out = subprocess.check_output([other, script], text=True).strip()
        self.assertEqual(os.path.normpath(out),
                         os.path.normpath(sys.executable))

    def test_the_pinned_interpreter_does_not_re_exec_itself(self):
        """no exec loop when it is already the right one"""
        script = self.write(sys.executable, 'print("ran once")\n')
        out = subprocess.check_output([sys.executable, script], text=True)
        self.assertEqual(out.strip(), 'ran once')

    def test_a_vanished_interpreter_is_ignored(self):
        """the directory was moved to a machine without that python: the run
        carries on with whatever started it rather than failing to exec"""
        script = self.write(os.path.join(self.path, 'gone', 'python3'),
                            'print("ran anyway")\n')
        out = subprocess.check_output([sys.executable, script], text=True)
        self.assertEqual(out.strip(), 'ran anyway')

    def test_arguments_survive_the_re_exec(self):
        script = self.write(sys.executable,
                            'print(" ".join(sys.argv[1:]))\n')
        other = os.path.realpath(sys.executable)
        if os.path.normpath(other) == os.path.normpath(sys.executable):
            self.skipTest('no second interpreter to start the script with')
        out = subprocess.check_output([other, script, '-f', '--name=abc'],
                                      text=True)
        self.assertEqual(out.strip(), '-f --name=abc')

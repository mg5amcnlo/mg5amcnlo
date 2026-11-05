################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
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
"""CommonRunCmd.find_model_name reads the model and the processes back from
Cards/proc_card_mg5.dat. Older versions of ProcCard.write cut every line at 70
characters, wherever the cut falls (even inside a token), so the card has to
be read back through ProcCard and not line by line."""

from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import madgraph.interface.common_run_interface as common_run
import madgraph.various.banner as banner

pjoin = os.path.join


class _Cmd(object):
    """the only state find_model_name needs"""

    def __init__(self, me_dir):
        self.me_dir = me_dir

    find_model_name = common_run.CommonRunCmd.find_model_name


class TestFindModelName(unittest.TestCase):

    def setUp(self):
        self.me_dir = tempfile.mkdtemp(prefix='find_model_name_')
        os.mkdir(pjoin(self.me_dir, 'Cards'))
        self.card_path = pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')

    def tearDown(self):
        shutil.rmtree(self.me_dir)

    def write_card(self, lines):
        """write the card as older versions of MG5 did (every line cut at 70
        characters) and return the raw text"""
        card = banner.ProcCard()
        for line in lines:
            card.append(line)
        card.write(self.card_path)
        with open(self.card_path) as stream:
            wrapped = []
            for line in stream.read().split('\n'):
                while len(line) > 70:
                    wrapped.append(line[:70] + '\\')
                    line = line[70:]
                wrapped.append(line)
        raw = '\n'.join(wrapped)
        with open(self.card_path, 'w') as stream:
            stream.write(raw)
        return raw

    def find(self):
        cmd = _Cmd(self.me_dir)
        model = cmd.find_model_name()
        self.assertEqual(cmd.model, model)
        return model, cmd.process

    def test_wrapped_model_path(self):
        # slide the 70-character cut through the model path and the option
        prefix = 'import model '
        cut_in_path = 0
        for path_length in range(70 - len(prefix) - 12, 70 - len(prefix) + 12):
            model_path = '/' + 'm' * (path_length - 1)
            raw = self.write_card(['%s%s -modelname' % (prefix, model_path),
                                   'generate e+ e- > mu+ mu-'])
            if model_path not in raw:
                cut_in_path += 1
            model, process = self.find()
            self.assertEqual(model, model_path, msg='card:\n%s' % raw)
            self.assertEqual(process, ['e+ e- > mu+ mu-'])
        # the regression case is in the sweep: the path itself is cut
        self.assertGreater(cut_in_path, 0)

    def test_wrapped_processes(self):
        # 'generate ' + generate is 70 characters: every pad moves the cut one
        # character back through '/ h z a @1'
        generate = 'p p > e+ ve mu- vm~ e+ ve mu- vm~ j QED<=10 QCD<=1 / h z a @1'
        add_process = 'p p > e+ ve mu- vm~ e+ ve mu- vm~ j j QED<=8 QCD<=2 @2'
        for pad in range(1, 12):
            procs = [generate.replace('/ h', ' ' * pad + '/ h'),
                     add_process.replace('@2', ' ' * pad + '@2')]
            lines = ['generate %s' % procs[0],
                     'add process %s # with jets' % procs[1]]
            raw = self.write_card(['import model sm-no_b_mass'] + lines +
                                  ['output madevent PROC_test'])
            for line in lines:
                self.assertNotIn(line, raw)
            model, process = self.find()
            self.assertEqual(model, 'sm-no_b_mass')
            self.assertEqual(process, procs, msg='card:\n%s' % raw)

    def test_processes_after_last_import_model(self):
        # a hand-edited card, not cleaned by ProcCard on the way out
        with open(self.card_path, 'w') as stream:
            stream.write('import model sm\n'
                         'generate p p > t t~\n'
                         'import model heft # the second model\n'
                         'generate g g > h\n'
                         '# add process g g > h j\n'
                         'add process g g > h j j\n'
                         'output madevent PROC_test\n')
        model, process = self.find()
        self.assertEqual(model, 'heft')
        self.assertEqual(process, ['g g > h', 'g g > h j j'])

    def test_model_v4(self):
        # ProcCard keeps no info['model'] for a v4 model: the name still comes
        # from the import line
        self.write_card(['import model_v4 mssm --debug',
                         'generate e+ e- > n1 n1'])
        model, process = self.find()
        self.assertEqual(model, 'mssm')
        self.assertEqual(process, ['e+ e- > n1 n1'])

    def test_default_model(self):
        self.write_card(['generate e+ e- > mu+ mu-'])
        model, process = self.find()
        self.assertEqual(model, 'sm')
        self.assertEqual(process, ['e+ e- > mu+ mu-'])


if __name__ == '__main__':
    unittest.main()

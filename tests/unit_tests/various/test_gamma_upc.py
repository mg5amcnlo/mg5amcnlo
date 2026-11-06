################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests of the gamma-UPC interface (arXiv:2207.03012) with forward neutron
tagging: pdlabel set, run_card checks, makefiles, generated Fortran, and the
gamma-UPC Fortran library itself (label parser and break-up probabilities)."""

from __future__ import absolute_import

import collections
import io
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import madgraph.various.banner as bannermod
import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join

UPC_RE = re.compile(r'(ed|ch)[fpm](f|[ix0-4]n[ix0-4]n)$')
TAG_CODE = {'i': -2, 'x': -1, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
PB208, AU197, O16, PROTON = (82, 126), (79, 118), (8, 8), (1, 0)


def upc_card(card_class, label, beam1=PB208, beam2=PB208, lpp=2):
    """run_card for an ion-ion UPC run with pdlabel=label"""
    card = card_class()
    card.set('lpp1', lpp, user=True)
    card.set('lpp2', lpp, user=True)
    for i, (Z, N) in ((1, beam1), (2, beam2)):
        card.set('nb_proton%i' % i, Z, user=True)
        card.set('nb_neutron%i' % i, N, user=True)
    card.set('pdlabel', label, user=True)
    return card


class TestGammaUPCLabels(unittest.TestCase):
    """the set of gamma-UPC pdlabels"""

    def test_upc_label_set(self):
        """300 labels, all valid for LO and NLO, and short enough for Fortran"""

        labels = bannermod.UPC_LABELS
        self.assertEqual(len(labels), 300)
        self.assertEqual(len(set(labels)), 300)
        for label in labels:
            self.assertTrue(UPC_RE.match(label), label)
            self.assertLessEqual(len(label), 7, label) # character*7 in Fortran
            self.assertTrue(bannermod.is_upc_label(label))
        for label in ['edff', 'chff', 'edpf', 'chmf', 'edfinin', 'chp0nxn', 'edm4n4n']:
            self.assertIn(label, labels)
        for label in ['nn23lo1', 'lhapdf', 'eva', 'edf', 'edfq1n', 'edf1n2x', 'edf5n5n', 'EDFF']:
            self.assertFalse(bannermod.is_upc_label(label), label)

        lo, nlo = bannermod.RunCardLO(), bannermod.RunCardNLO()
        for name in ['pdlabel', 'pdlabel1', 'pdlabel2']:
            self.assertTrue(set(labels) <= set(lo.allowed_value[name]), name)
        # edpf, edmf, chpf, chmf were missing at NLO
        self.assertTrue(set(labels) <= set(nlo.allowed_value['pdlabel']))

    def test_upc_prefix_reserved(self):
        """no other pdlabel starts with ed or ch: the makefiles link
        libgammaUPC on $(filter ed% ch%, ...) and the generated Fortran
        calls PHOTONPDFSQUARE on PDLABEL(1:2) = 'ed' or 'ch'"""

        for card in [bannermod.RunCardLO(), bannermod.RunCardNLO()]:
            for label in card.allowed_value['pdlabel']:
                if label[:2] in ('ed', 'ch'):
                    self.assertTrue(bannermod.is_upc_label(label), label)

    def test_get_pdf_id_upc(self):
        """the LHE pdf id of every gamma-UPC label is 0"""

        card = bannermod.RunCardLO()
        for label in bannermod.UPC_LABELS:
            self.assertEqual(card.get_pdf_id(label), 0, label)
        self.assertEqual(card.get_pdf_id('nn23lo1'), 247000)


class TestGammaUPCRunCard(unittest.TestCase):
    """run_card handling of the gamma-UPC labels"""

    def test_upc_pdlabel_block_propagation(self):
        """one gamma-UPC label sets pdlabel, pdlabel1 and pdlabel2"""

        card = upc_card(bannermod.RunCardLO, 'nn23lo1')
        card.set('pdlabel1', 'edf1n2n', user=True)
        card.check_validity()
        self.assertEqual(card['pdlabel'], 'edf1n2n')
        self.assertEqual(card['pdlabel1'], 'edf1n2n')
        self.assertEqual(card['pdlabel2'], 'edf1n2n')

        card = upc_card(bannermod.RunCardLO, 'nn23lo1')
        card.set('pdlabel2', 'chm4nxn', user=True)
        card.check_validity()
        self.assertEqual(card['pdlabel'], 'chm4nxn')
        self.assertEqual(card['pdlabel1'], 'chm4nxn')

        card = upc_card(bannermod.RunCardLO, 'chp0nxn')
        card.check_validity()
        self.assertEqual(card['pdlabel1'], 'chp0nxn')
        self.assertEqual(card['pdlabel2'], 'chp0nxn')

    def test_upc_pdlabel_lpp_compat(self):
        """gamma-UPC labels need lpp=2; lpp=2 falls back to edff"""

        card = upc_card(bannermod.RunCardLO, 'edfinin', beam1=PROTON, beam2=PROTON, lpp=1)
        with self.assertRaises(bannermod.InvalidRunCard):
            card.check_validity()

        card = upc_card(bannermod.RunCardLO, 'nn23lo1', beam1=PROTON, beam2=PROTON)
        with misc.MuteLogger(['madevent.cards'], [50]):
            card.check_validity()
        self.assertEqual(card['pdlabel1'], 'edff')
        self.assertEqual(card['pdlabel2'], 'edff')

    def test_upc_neutron_tagging_nuclei(self):
        """the nucleus x neutron-tag matrix, at LO (card check and helper)"""

        allowed = [
            (PB208, PB208, 'edf4n4n'), (AU197, AU197, 'chm3n1n'),
            (O16, O16, 'edf2n2n'), (O16, O16, 'chpxn0n'),
            (PROTON, PROTON, 'edf1n1n'),          # tags ignored for protons
            (PROTON, PB208, 'edf4n1n'), (PB208, PROTON, 'edf2n4n'),
            (PB208, O16, 'edf3n2n'),               # asymmetric
            ((54, 75), (54, 75), 'edfinin'),       # Xe129: no tag, no check
            ((54, 75), (54, 75), 'edff'), ((54, 75), (54, 75), 'edpf'),
            ((92, 146), (92, 146), 'chff'),
        ]
        rejected = [
            (O16, O16, 'edf3n0n'), (O16, O16, 'edf0n4n'),
            (PB208, O16, 'edf2n3n'),               # O16 is beam 2
            ((54, 75), (54, 75), 'edfxnxn'), ((54, 75), PB208, 'edf0nin'),
            ((92, 146), (92, 146), 'edf1n1n'),     # U238 not enabled
            (PB208, (92, 146), 'edm0n0n'),
        ]
        for beam1, beam2, label in allowed:
            card = upc_card(bannermod.RunCardLO, label, beam1, beam2)
            card.check_validity()
            self.assertEqual(card['pdlabel'], label)
            bannermod.check_upc_neutron_tagging(card, label)
        for beam1, beam2, label in rejected:
            card = upc_card(bannermod.RunCardLO, label, beam1, beam2)
            with self.assertRaises(bannermod.InvalidRunCard, msg=(beam1, beam2, label)):
                card.check_validity()

        # the message names the beam and the nucleus
        card = upc_card(bannermod.RunCardLO, 'edf2n3n', PB208, O16)
        with self.assertRaisesRegex(bannermod.InvalidRunCard, r"'3n'.*beam 2.*Z=8, N=8.*at most 2n"):
            bannermod.check_upc_neutron_tagging(card, 'edf2n3n')

    def test_upc_neutron_tagging_nlo(self):
        """the same check at NLO (only in Fortran before)"""

        for beam, label in [(PB208, 'edf4n4n'), (O16, 'chf2nxn'), (PB208, 'edpf'),
                            (PB208, 'chmf'), ((54, 75), 'edfinin')]:
            card = upc_card(bannermod.RunCardNLO, label, beam, beam)
            card.check_validity()
            self.assertEqual(card['pdlabel'], label)
        for beam, label in [(O16, 'edf3n3n'), ((54, 75), 'edf0n0n'), ((92, 146), 'chfxnxn')]:
            card = upc_card(bannermod.RunCardNLO, label, beam, beam)
            with self.assertRaises(bannermod.InvalidRunCard, msg=(beam, label)):
                card.check_validity()

    def test_upc_lpp_sign(self):
        """lpp=-2 is a photon beam too, as in the Fortran ABS(LPP)"""

        card = {'lpp1': -2, 'lpp2': -2, 'nb_proton1': 8, 'nb_neutron1': 8,
                'nb_proton2': 8, 'nb_neutron2': 8}
        with self.assertRaises(bannermod.InvalidRunCard):
            bannermod.check_upc_neutron_tagging(card, 'edf3n3n')
        card['lpp1'] = card['lpp2'] = 1
        bannermod.check_upc_neutron_tagging(card, 'edf3n3n') # not a UPC run

        # NLO: the gamma-UPC label of a (-2,-2) run is kept
        card = upc_card(bannermod.RunCardNLO, 'chf0n0n', PROTON, PROTON, lpp=-2)
        card.check_validity()
        self.assertEqual(card['pdlabel'], 'chf0n0n')

    def test_upc_run_card_write_read(self):
        """a 7-character label survives write/read and reaches the include"""

        card = upc_card(bannermod.RunCardLO, 'nn23lo1')
        card.set('pdlabel1', 'chm4n4n', user=True)
        tmpdir = tempfile.mkdtemp(prefix='upc_card')
        try:
            path = pjoin(tmpdir, 'run_card.dat')
            card.write(path)
            reloaded = bannermod.RunCard(path)
            self.assertIsInstance(reloaded, bannermod.RunCardLO)
            for name in ['pdlabel', 'pdlabel1', 'pdlabel2']:
                self.assertEqual(reloaded[name], 'chm4n4n')
            f = io.StringIO()
            reloaded.write_include_file(None, output_file=f)
            self.assertIn("pdlabel = 'chm4n4n'", f.getvalue())
            self.assertIn("pdsublabel(1) = 'chm4n4n'", f.getvalue())
            self.assertIn("pdsublabel(2) = 'chm4n4n'", f.getvalue())
        finally:
            shutil.rmtree(tmpdir)


class TestGammaUPCSystematics(unittest.TestCase):

    def test_systematics_refuses_upc(self):
        """systematics stops cleanly for every gamma-UPC label"""

        import madgraph.various.systematics as systematics

        tmpdir = tempfile.mkdtemp(prefix='upc_syst')
        try:
            for label in ['edff', 'edpf', 'chf1n2n', 'chm4nxn']:
                card = upc_card(bannermod.RunCardLO, label)
                text = io.StringIO()
                card.write(text)
                path = pjoin(tmpdir, '%s.lhe' % label)
                with open(path, 'w') as fsock:
                    fsock.write('<LesHouchesEvents version="3.0">\n<header>\n<MGRunCard>\n'
                                '<![CDATA[\n%s]]>\n</MGRunCard>\n</header>\n<init>\n'
                                '22 22 2.6e+03 2.6e+03 0 0 0 0 3 1\n1.0 0.1 1.0 1\n'
                                '</init>\n</LesHouchesEvents>\n' % text.getvalue())
                with self.assertRaisesRegex(systematics.SystematicsError, label):
                    systematics.Systematics(path, None)
        finally:
            shutil.rmtree(tmpdir)


class TestGammaUPCMakefiles(unittest.TestCase):
    """the three makefiles link libgammaUPC exactly for the gamma-UPC labels"""

    makefiles = [  # (path, %-template, pdlabel variables)
        ('Template/LO/Source/PDF/makefile', False, ('pdlabel1', 'pdlabel2')),
        ('madgraph/iolibs/template_files/madevent_makefile_source', True, ('pdlabel1', 'pdlabel2')),
        ('Template/NLO/Source/makefile', False, ('pdlabel',)),
    ]

    def setUp(self):
        if not misc.which('make'):
            raise unittest.SkipTest('make not available')
        self.tmpdir = tempfile.mkdtemp(prefix='upc_make')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def gammaupc_condition(path, template):
        """the 'ifneq' line that selects libgammaUPC in that makefile"""

        text = open(pjoin(MG5DIR, path)).read()
        if template:
            text = text % collections.defaultdict(str)
        lines = text.split('\n')
        found = []
        for i, line in enumerate(lines):
            if line.startswith('ifneq') and 'filter' in line:
                block = []
                for nxt in lines[i+1:]:
                    if nxt.startswith('endif'):
                        break
                    block.append(nxt)
                if any('gammaUPC' in l for l in block):
                    found.append(line)
        assert len(found) == 1, (path, found)
        return found[0]

    def make(self, content, *args):
        path = pjoin(self.tmpdir, 'makefile')
        with open(path, 'w') as fsock:
            fsock.write(content)
        out = subprocess.run(['make', '-s', '-f', path] + list(args), cwd=self.tmpdir,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(out.returncode, 0, out.stdout)
        return out.stdout.strip()

    def test_makefile_links_gammaupc(self):

        upc = ['edff', 'chff', 'edpf', 'edmf', 'chpf', 'chmf', 'edf1n2n', 'chp0nxn',
               'edm4n4n', 'chfinin']
        other = ['nn23lo1', 'lhapdf', 'eva', 'iww', 'none', 'mixed', 'cteq6l1',
                 'isronlyll', 'cepc240ll', 'emela', 'ct14q00']
        all_labels = set(bannermod.RunCardLO().allowed_value['pdlabel']) | \
                     set(bannermod.RunCardNLO().allowed_value['pdlabel'])
        for path, template, variables in self.makefiles:
            cond = self.gammaupc_condition(path, template)
            content = cond + '\nR=yes\nelse\nR=no\nendif\nall:\n\t@echo $(R)\n'
            for label in upc + other + ['']:
                args = ['%s=%s' % (v, label) for v in variables]
                expect = 'yes' if label in upc else 'no'
                self.assertEqual(self.make(content, *args), expect, (path, label))
            if len(variables) == 2: # one beam with gamma-UPC is enough
                self.assertEqual(self.make(content, 'pdlabel1=nn23lo1', 'pdlabel2=edf1n2n'), 'yes')

            # every allowed pdlabel, in one call
            pattern = re.search(r'\$\(filter ([^,]+),', cond).group(1)
            content = 'all:\n\t@echo $(foreach l,$(LABELS),$(l):$(if $(filter %s,$(l)),1,0))\n' % pattern
            out = self.make(content, 'LABELS=%s' % ' '.join(sorted(all_labels)))
            result = dict(item.split(':') for item in out.split())
            self.assertEqual(set(result), all_labels)
            for label, linked in result.items():
                self.assertEqual(linked == '1', bannermod.is_upc_label(label), (path, label))


class TestGammaUPCExporters(unittest.TestCase):

    def test_pdf_lines_upc_condition(self):
        """a a > mu+ mu-: LO and NLO call PHOTONPDFSQUARE on the same condition"""

        import models.import_ufo as import_ufo
        import madgraph.core.base_objects as base_objects
        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_v4 as export_v4
        import madgraph.iolibs.export_fks as export_fks

        model = import_ufo.import_model('sm')
        legs = base_objects.LegList([base_objects.Leg({'id': 22, 'state': False}),
                                     base_objects.Leg({'id': 22, 'state': False}),
                                     base_objects.Leg({'id': -13, 'state': True}),
                                     base_objects.Leg({'id': 13, 'state': True})])
        process = base_objects.Process({'legs': legs, 'model': model})
        me = helas_objects.HelasMatrixElement(diagram_generation.Amplitude(process))

        lo = export_v4.ProcessExporterFortranME().get_pdf_lines(me, 2)[2]
        nlo = export_fks.ProcessExporterFortranFKS().get_pdf_lines_mir(me, 2)[2]
        condition = "IF (ABS(LPP(1)).EQ.2.AND.ABS(LPP(2)).EQ.2.AND.(PDLABEL(1:2).EQ.'ed'.OR.PDLABEL(1:2).EQ.'ch'))THEN"
        for lines in [lo, nlo]:
            self.assertIn(condition, lines)
            self.assertIn('PHOTONPDFSQUARE(XBK(1),XBK(2))', lines)


class TestGammaUPCFortran(unittest.TestCase):
    """compile the gamma-UPC library (a few seconds) and test it directly"""

    gammaupc = pjoin(MG5DIR, 'Template', 'Common', 'Source', 'PDF', 'gammaUPC')
    modules = ['tbessk', 'tbessj', 'elliptic_integral', 'nielsen_generalized_polylog',
               'photoabsorption', 'nintlib', 'interpolation', 'fmin_module',
               'OpticalGlauber_Geometry', 'ElasticPhotonPhotonFlux']

    label_driver = """
      program check_labels
      USE ElasticPhotonPhotonFlux
      implicit none
      character*7 label
      integer ios
      do
         read(*,'(a)',iostat=ios) label
         if(ios.ne.0) exit
         call set_gammaUPC_label(label)
         write(*,'(a7,1x,l1,1x,f4.1,2i3)') label,
     $        USE_CHARGEFORMFACTOR4PHOTON, NEUTRON_XSIGMA, neutron_tagging
      enddo
      end
"""

    # P(in1,in2) for Pb208+Pb208 at 5.02 TeV, impact parameter b
    pbreak_driver = """
program check_pbreak
  USE ElasticPhotonPhotonFlux
  implicit none
  include 'run90.inc'
  character(len=8)::arg
  integer::i,j,ib
  real(kind(1d0))::p(-2:4,-2:4),b,bfm(3)=(/15d0,25d0,50d0/)
  nb_proton=(/82,82/); nb_neutron=(/126,126/)
  ebeamMG5=(/522080d0,522080d0/); lpp=(/2,2/)
  call get_command_argument(1,arg)
  read(arg,*) NEUTRON_XSIGMA
  do ib=1,3
     b=bfm(ib)/0.1973d0
     do i=-2,4
        do j=-2,4
           p(i,j)=Pbreak_AB_WoodsSaxon(i,j,b)
        enddo
     enddo
     write(*,'(a,49es24.15)') 'P', ((p(i,j),j=-2,4),i=-2,4)
  enddo
end program check_pbreak
"""

    @classmethod
    def setUpClass(cls):
        cls.fc = misc.which('gfortran')
        if not cls.fc:
            raise unittest.SkipTest('gfortran not available')
        cls.tmpdir = tempfile.mkdtemp(prefix='upc_fortran')
        src = pjoin(cls.tmpdir, 'gammaUPC')
        os.mkdir(src)
        for name in os.listdir(cls.gammaupc):
            if name.endswith(('.f90', '.f', '.inc')) and name != 'test.f90':
                shutil.copy(pjoin(cls.gammaupc, name), src)
        # photonpdfsquare.f includes '../pdf.inc'
        shutil.copy(pjoin(MG5DIR, 'Template', 'NLO', 'Source', 'pdf.inc'), cls.tmpdir)
        for mod in cls.modules:
            cls.compile(src, ['-c', '%s.f90' % mod])
        cls.compile(src, ['-ffixed-line-length-132', '-c', 'photonpdfsquare.f'])
        objects = ['%s.o' % m for m in cls.modules] + ['photonpdfsquare.o']
        with open(pjoin(src, 'check_labels.f'), 'w') as fsock:
            fsock.write(cls.label_driver)
        with open(pjoin(src, 'check_pbreak.f90'), 'w') as fsock:
            fsock.write(cls.pbreak_driver)
        cls.compile(src, ['-ffixed-line-length-132', '-o', 'check_labels', 'check_labels.f'] + objects)
        cls.compile(src, ['-o', 'check_pbreak', 'check_pbreak.f90'] + objects)
        cls.src = src

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'tmpdir', None):
            shutil.rmtree(cls.tmpdir)

    @classmethod
    def compile(cls, cwd, args):
        out = subprocess.run([cls.fc, '-O'] + args, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        if out.returncode:
            raise Exception('compilation failed: %s\n%s' % (' '.join(args), out.stdout))

    def run_exe(self, exe, args=(), stdin=''):
        return subprocess.run([pjoin(self.src, exe)] + list(args), input=stdin, cwd=self.src,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_photonpdfsquare_label_parser(self):
        """set_gammaUPC_label decodes the 300 labels and stops on others"""

        out = self.run_exe('check_labels', stdin='\n'.join(bannermod.UPC_LABELS) + '\n')
        self.assertEqual(out.returncode, 0, out.stdout)
        result = {}
        for line in out.stdout.split('\n'):
            fields = line.split()
            if len(fields) == 5 and bannermod.is_upc_label(fields[0]):
                result[fields[0]] = (fields[1] == 'T', float(fields[2]), int(fields[3]), int(fields[4]))
        self.assertEqual(set(result), set(bannermod.UPC_LABELS))
        for label, got in result.items():
            tags = (-2, -2) if label[3] == 'f' else (TAG_CODE[label[3]], TAG_CODE[label[5]])
            expect = (label[:2] == 'ch', {'f': 0., 'p': 1., 'm': -1.}[label[2]]) + tags
            self.assertEqual(got, expect, label)

        for label in ['edfq1n', 'edg', 'edf1n2x', 'xyff', 'edf1n2', 'edff1', 'edf5n0n']:
            out = self.run_exe('check_labels', stdin=label + '\n')
            self.assertEqual(out.returncode, 2, (label, out.stdout))
            self.assertIn('unknown gamma-UPC pdlabel', out.stdout)

    def test_pbreak_sum_rules(self):
        """break-up probabilities of Pb208+Pb208: 0n+Xn=1 per nucleus, the four
        0n/Xn classes sum to one, P(y,z)=P(y,in)P(in,z), sum_k P(kn) < P(Xn);
        P(y,in) used to stop (in2=-2 rejected). Central and +-1 sigma."""

        X, Z0, I = -1, 0, -2
        for xsigma in ['0', '1', '-1']:
            out = self.run_exe('check_pbreak', [xsigma])
            self.assertEqual(out.returncode, 0, out.stdout)
            rows = [l.split()[1:] for l in out.stdout.split('\n') if l.startswith('P ')]
            self.assertEqual(len(rows), 3, out.stdout)
            for row in rows:
                v = [float(x) for x in row]
                P = lambda i, j: v[(i + 2) * 7 + (j + 2)]
                self.assertEqual(P(I, I), 1.)
                self.assertAlmostEqual(P(Z0, I) + P(X, I), 1., places=12)
                self.assertAlmostEqual(P(I, Z0) + P(I, X), 1., places=12)
                self.assertAlmostEqual(P(Z0, Z0) + P(Z0, X) + P(X, Z0) + P(X, X), 1., places=12)
                for y in range(-1, 5):
                    for z in range(-1, 5):
                        self.assertAlmostEqual(P(y, z), P(y, I) * P(I, z), places=14)
                    self.assertAlmostEqual(P(y, I), P(I, y), places=14)  # symmetric beams
                    self.assertGreater(P(y, I), 0.)
                self.assertLess(sum(P(k, I) for k in range(1, 5)), P(X, I))

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
from __future__ import absolute_import
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as mgcmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as amcatnlocmd
import os
import madgraph.fks.fks_helas_objects as fks_helas
import copy
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from madgraph.iolibs.export_fks import ProcessExporterFortranFKS
from madgraph.fks import fks_common
from tests.unit_tests.fks.test_momentum_maps import fortran_routine
import madgraph.iolibs.save_load_object as save_load_object
import madgraph
import madgraph.various.misc as misc
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class TestAMCatNLOEWTagPh(unittest.TestCase):
    """ a suite of extra tests for the ew stuff """
    
    def setUp(self):
        self.interface = mgcmd.MasterCmd()

    def test_generate_fks_tagph(self):
        """check that the generate command works as expected.
        In particular the correct number of real-emission processes
        and FKS partners"""
        cmd_list = [
            'u u~ > a a [real=QED]',
            'u u~ > !a! !a! [real=QED]',
            'u u~ > !2a!  [real=QED]',
            'u u~ > 2!a! [real=QED]'] 

        nrealproc_list = [9, 3, 3, 3]
        fks_j_list = [[1,2,3], [1,2], [1,2], [1,2]] # 4 is not there in the first set for symmetry


        for cmd, nrealproc, fks_j in \
                zip(cmd_list, nrealproc_list, fks_j_list):
            self.interface.do_generate(cmd)

            fksprocess = self.interface._fks_multi_proc['born_processes'][0]

            self.assertEqual(len(fksprocess.real_amps), nrealproc)
            self.assertEqual(set([r.fks_infos[0]['j'] for r in fksprocess.real_amps]), set(fks_j))
            
    def test_invalid_syntax_tag(self):

        cmd = "u u~ > !a! !a!"
        self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !z! a [real=QED]"
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !t! t~ a [real=QED]"
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        #cmd = "u u~ > !a! a [real=QCD]" # is this valid ? I guess NOT [note this is QCD]
        #self.assertRaises(madgraph.InvalidCmd,  self.interface.do_generate, cmd)

        

class TestTaggedPhotonStatistics(unittest.TestCase):
    """Outer FKS compensation and native relabelling have distinct identities."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg7_tagged_statistics_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = Path(cls.tempdir.name)
        model = cls.work / 'models/loop_qcd_qed_sm'
        shutil.copytree(cls.root / 'models/loop_qcd_qed_sm', model,
                        ignore=shutil.ignore_patterns('*.pkl', '*.pyc', '__pycache__'))
        cls.cases = []
        for count, process in (
                (1, 'u u~ > !a! e+ e- aS=0 aEW=3 [real=QED]'),
                (2, 'u u~ > !a! !a! aS=0 aEW=2 [real=QED]')):
            interface = mgcmd.MasterCmd()
            interface.no_notification()
            for command in (
                    'set automatic_html_opening False --no_save',
                    'set low_mem_multicore_nlo_generation False --no_save',
                    'import model ' + str(model), 'generate ' + process):
                interface.exec_cmd(command, errorhandling=False, printcmd=False,
                                   precmd=True, postcmd=True)
            helas = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
            for born in helas.get_matrix_elements():
                cls.cases.append((count, born))

    def export_tables(self, born, directory):
        directory.mkdir(exist_ok=True)
        previous = os.getcwd()
        try:
            os.chdir(directory)
            object.__new__(ProcessExporterFortranFKS).write_mc_history_files(born)
        finally:
            os.chdir(previous)
        return ((directory / 'fks_symmetry.inc').read_text(),
                (directory / 'mc_histories.inc').read_text())

    @staticmethod
    def values(source, name):
        match = re.search(r'\bDATA\s+' + re.escape(name) + r'\s*/(.*?)/',
                          source, re.I | re.S)
        if match is None:
            raise AssertionError('Missing generated table ' + name)
        return re.sub(r'\n {5}\S', '', match.group(1)).replace('\n', '').split(',')

    def photon_rows(self, count, born):
        rows = []
        for sector, info in enumerate(born.get_fks_info_list()):
            real = born.real_processes[info['n_me'] - 1]
            legs = real.matrix_element['processes'][0].get_legs_with_decays()
            if sum(leg['id'] == 22 and leg['state'] for leg in legs) == count + 1:
                rows.append((sector, info, real, legs))
        self.assertTrue(rows)
        return rows

    def test_generated_compensation_preserves_strict_history_permutations(self):
        for count, born in self.cases:
            with self.subTest(tagged_photons=count):
                symmetry, histories = self.export_tables(born, self.work / ('tables%d' % count))
                fac_i = list(map(int, self.values(symmetry, 'FKS_FAC_I_D')))
                fac_j = list(map(int, self.values(symmetry, 'FKS_FAC_J_D')))
                born_iden = [float(value.upper().replace('D', 'E'))
                             for value in self.values(symmetry, 'FKS_IDEN_BORN_D')]
                real_iden = [float(value.upper().replace('D', 'E'))
                             for value in self.values(symmetry, 'FKS_IDEN_REAL_D')]
                native = list(map(int, self.values(histories, 'MC_HIST_NATIVE')))
                complete = [value.strip().lower() for value in self.values(histories, 'MC_HIST_COMPLETE')]
                for sector, info, real, legs in self.photon_rows(count, born):
                    self.assertEqual(fac_i[sector], count + 1)
                    self.assertEqual(fac_j[sector], 1)
                    self.assertEqual(real.matrix_element['identical_particle_factor'],
                                     2 if count == 1 else 6)
                    self.assertEqual(fac_i[sector] * fac_j[sector] *
                                     born_iden[sector] / real_iden[sector], 1)
                    # Only the untagged photon is a retained FKS parton. Its
                    # omitted permutations affect statistics, not safe ME reuse.
                    self.assertFalse(real.particle_tags[info['fks_info']['i'] - 1])
                    self.assertEqual(native.count(sector + 1), 1)
                    self.assertEqual(complete[sector], '.false.')
                for row, sector in enumerate(native, 1):
                    info = born.get_fks_info_list()[sector - 1]
                    real = born.real_processes[info['n_me'] - 1]
                    permutation = re.search(
                        r'DATA\s*\(MC_HIST_PERM\(MC_HIST_POS,%d\).*?/(.*?)/' % row,
                        histories, re.I | re.S)
                    labels = list(map(int, re.sub(r'\n {5}\S', '',
                                                 permutation.group(1)).replace('\n', '').split(',')))
                    self.assertEqual(labels[:2], [1, 2])
                    for index, label in enumerate(labels):
                        self.assertEqual(real.particle_tags[index], real.particle_tags[label - 1])

    def test_statistical_photon_identity_keeps_polarization_and_shell_constraints(self):
        count, original = self.cases[0]
        for constraint, value in (('polarization', [1]), ('onshell', True)):
            with self.subTest(constraint=constraint):
                born = copy.deepcopy(original)
                rows = self.photon_rows(count, born)
                changed = set()
                for _, _, real, legs in rows:
                    if id(real) in changed:
                        continue
                    changed.add(id(real))
                    for process in real.matrix_element['processes']:
                        for index, leg in enumerate(process.get_legs_with_decays()):
                            if leg['id'] == 22 and real.particle_tags[index]:
                                leg[constraint] = value
                symmetry, _ = self.export_tables(born, self.work / constraint)
                factors = list(map(int, self.values(symmetry, 'FKS_FAC_I_D')))
                for sector, _, _, _ in rows:
                    self.assertEqual(factors[sector], 1)
        process = self.photon_rows(count, original)[0][2].matrix_element['processes'][0]
        tags = list(self.photon_rows(count, original)[0][2].particle_tags)
        ordinary = fks_common.external_process_identity(process, tags)
        tags[0] = True
        self.assertNotEqual(ordinary, fks_common.external_process_identity(process, tags))

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_permutation_averaged_real_partition_and_singular_limits(self):
        template = self.root / 'Template/NLO/SubProcesses'
        for count, born in self.cases:
            with self.subTest(tagged_photons=count):
                rows = self.photon_rows(count, born)
                sector, _, real, legs = rows[0]
                selected = [row for row in rows if row[2] is real]
                work = self.work / ('partition%d' % count)
                symmetry, _ = self.export_tables(born, work)
                factors_i = list(map(int, self.values(symmetry, 'FKS_FAC_I_D')))
                factors_j = list(map(int, self.values(symmetry, 'FKS_FAC_J_D')))
                photons = [index + 1 for index, leg in enumerate(legs)
                           if leg['id'] == 22 and leg['state']]
                pairs = [(i, j) for i, js in sorted(real.fks_j_from_i.items()) for j in js]
                outer = [(info['fks_info']['i'], info['fks_info']['j'],
                          factors_i[index] * factors_j[index])
                         for index, info, _, _ in selected]
                self.compile_partition_driver(work, template, legs, photons, pairs, outer)
                result = subprocess.run([str(work / 'check')], cwd=work,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS tagged photon partitions and limits', result.stdout)

    def compile_partition_driver(self, work, template, legs, photons, pairs, outer):
        def ints(values):
            return ','.join(str(value) for value in values)
        count = len(legs)
        for name, source in {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=%d,nincoming=2)' % count,
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'coupl.inc': '', 'pmass.inc': 'common/to_mass/pmass'}.items():
            (work / name).write_text(''.join('      ' + line + '\n' for line in source.splitlines()))
        shutil.copyfile(template / 'fks_powers.inc', work / 'fks_powers.inc')
        routines = [fortran_routine(template / 'fks_Sij.f', name) for name in
                    ('fks_Sij', 'get_dkl_Sij', 'get_cms_energy',
                     'get_cms_costh_fks', 'costh_fks', 'h_damp')]
        routines.append(fortran_routine(self.root / 'Template/NLO/Source/kin_functions.f', 'dot'))
        (work / 'partition.f').write_text('\n'.join(routines))
        source = PARTITION_DRIVER.replace('@PHOTONS@', ints(photons))
        for key, values in (('PAIR_I', [pair[0] for pair in pairs]),
                            ('PAIR_J', [pair[1] for pair in pairs]),
                            ('OUTER_I', [row[0] for row in outer]),
                            ('OUTER_J', [row[1] for row in outer]),
                            ('FACTORS', [row[2] for row in outer]),
                            ('PDGS', [leg['id'] for leg in legs])):
            source = source.replace('@' + key + '@', ints(values))
        source = source.replace('@NPHOTONS@', str(len(photons))).replace('@NPAIRS@', str(len(pairs)))
        source = source.replace('@NOUTER@', str(len(outer)))
        (work / 'driver.f90').write_text(source)
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-none', '-ffree-line-length-none', '-I', str(work),
            str(template / 'fks_phase_space_data.f'), str(work / 'partition.f'),
            str(self.root / 'HELAS/boostx.F'), str(work / 'driver.f90'),
            '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


PARTITION_DRIVER = '''program check_tagged_partition
use fks_phase_space_data, only: shat,sqrtshat,ybst_til_tocm,p_i_fks_cnt
implicit none
include 'nexternal.inc'
integer,parameter :: photons(@NPHOTONS@)=[@PHOTONS@],pair_i(@NPAIRS@)=[@PAIR_I@],pair_j(@NPAIRS@)=[@PAIR_J@]
integer,parameter :: outer_i(@NOUTER@)=[@OUTER_I@],outer_j(@NOUTER@)=[@OUTER_J@],factors(@NOUTER@)=[@FACTORS@]
integer :: partners(nexternal,0:nexternal),types(nexternal),pdgs(nexternal),ifks,jfks
common/c_fks_inc/partners,types,pdgs
common/fks_indices/ifks,jfks
logical :: radiation(nexternal),charged(nexternal)
common/c_is_aorg/radiation
common/c_is_charged/charged
real(8) :: pmass(nexternal),p(0:3,nexternal),q(0:3,nexternal),total(0:3),rest(0:3),epsilon
common/to_mass/pmass
real(8) :: full,retained,dominant,toy,average,mass,xi,angle,sector
integer :: sample,shift,k,index,partner,pair(2),pair_count
pdgs=[@PDGS@]
types=1
types(1:2)=[3,-3]
radiation=pdgs.eq.22
charged=.not.radiation
pmass=0d0
partners=0
do k=1,size(pair_i)
  index=pair_i(k)
  partners(index,0)=partners(index,0)+1
  partners(index,partners(index,0))=pair_j(k)
enddo
sqrtshat=20d0
shat=sqrtshat**2
ybst_til_tocm=0d0
do sample=0,6
  p=0d0
  p(:,1)=[10d0,0d0,0d0,10d0]
  p(:,2)=[10d0,0d0,0d0,-10d0]
  p(:,photons(size(photons)))=[3d0,-1.2d0,1.5d0,3d0*sqrt(.59d0)]
  if(sample.ge.1.and.sample.le.3)then
    epsilon=10d0**(-2*sample)
    p(:,photons(size(photons)))=p(:,photons(size(photons)))*epsilon/3d0
  elseif(sample.ge.4)then
    epsilon=10d0**(2-sample)
    p(:,photons(size(photons)))=[3d0,3d0*epsilon,0d0,3d0*sqrt(1d0-epsilon**2)]
  endif
  if(size(photons).eq.2)p(:,photons(1))=[2d0,1.2d0,0d0,1.6d0]
  pair_count=0
  do k=3,nexternal
    if(k.eq.photons(size(photons)))cycle
    if(size(photons).eq.2.and.k.eq.photons(1))cycle
    pair_count=pair_count+1
    pair(pair_count)=k
  enddo
  total=sum(p(:,1:2),dim=2)-sum(p(:,3:nexternal),dim=2)
  mass=sqrt(total(0)**2-sum(total(1:3)**2))
  rest=mass/2d0*[1d0,.3d0,.4d0,sqrt(.75d0)]
  call boostx(rest,total,p(:,pair(1)))
  rest(1:3)=-rest(1:3)
  call boostx(rest,total,p(:,pair(2)))
  full=0d0
  dominant=0d0
  do k=1,size(pair_i)
    sector=weight(p,pair_i(k),pair_j(k))
    full=full+sector
    if(pair_i(k).eq.photons(size(photons)))dominant=dominant+sector
  enddo
  if(abs(full-1d0).gt.2d-9)error stop 'full real sector partition'
  average=0d0
  do shift=0,size(photons)-1
    q=p
    do k=1,size(photons)
      q(:,photons(k))=p(:,photons(mod(k-1+shift,size(photons))+1))
    enddo
    retained=0d0
    do k=1,size(outer_i)
      retained=retained+factors(k)*weight(q,outer_i(k),outer_j(k))
    enddo
    toy=1d0
    do k=1,size(photons)
      toy=toy*(1d0+q(0,photons(k))/sqrtshat)*(1d0+q(1,photons(k))**2/shat)
    enddo
    average=average+retained*toy/size(photons)
  enddo
  if(abs(average/toy-full).gt.2d-9)error stop 'tagged photon outer statistical compensation'
  if((sample.eq.3.or.sample.eq.6).and.abs(dominant-1d0).gt.2d-7) &
       error stop 'tagged photon soft or collinear sector limit'
enddo
print *, 'PASS tagged photon partitions and limits'
contains
real(8) function weight(point,i,j)
  real(8),intent(in) :: point(0:3,nexternal)
  integer,intent(in) :: i,j
  real(8) :: xi,y,fks_Sij
  ifks=i
  jfks=j
  xi=2d0*point(0,i)/sqrtshat
  y=sum(point(1:3,i)*point(1:3,j))/sqrt(sum(point(1:3,i)**2)*sum(point(1:3,j)**2))
  p_i_fks_cnt(:,0)=point(:,i)/xi
  p_i_fks_cnt(:,2)=p_i_fks_cnt(:,0)
  weight=fks_Sij(point,i,j,xi,y)
end function
end program
'''

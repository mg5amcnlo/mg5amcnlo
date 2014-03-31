################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
import copy
import subprocess
import shutil
import os

import tests.unit_tests as unittest
import logging

from madgraph import MG4DIR, MG5DIR, MadGraph5Error

import madgraph.core.base_objects as base_objects
import madgraph.iolibs.export_v4 as export_v4
import models.import_ufo as import_ufo
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.ufo_expression_parsers as ufo_expression_parsers
from madgraph.iolibs import save_load_object

import  models.check_param_card as check_param_card 

logger = logging.getLogger('madgraph.test.model')
pjoin = os.path.join

class CheckFileCreate():
    """Check that the files are correctly created"""

    output_path = '/tmp/mg5_model_equivalence' # work only on LINUX but that's ok for the test routine
    created_files =[]

    def setUp(self):
        try:
            os.system('rm -rf %s &> /dev/null' % self.output_path)
        except:
            pass
        os.mkdir(self.output_path)
        
    def tearDown(self):
        os.system('rm -rf %s ' % self.output_path)

    def assertFileContains(self, filename, solution):
        """ Check the content of a file """

        current_value = open(self.give_pos(filename)).read()
        self.assertEqual(current_value, solution)

    def FileContent(self, filename):
        return open(self.give_pos(filename)).read()

    def ReturnFile(self, filename):
        return open(self.give_pos(filename))

    def give_pos(self, filename):
        """ take a name and a change it in order to have a valid path in the output directory """
        return os.path.join(self.output_path, filename)

    def clean_files(self):
        """ suppress all the files linked to this test """
        
        for filename in self.created_files:
            try:
                os.remove(self.give_pos(filename))
            except OSError:
                pass
    


class CompareMG4WithUFOModel(unittest.TestCase):
    """checking if the MG4 model and the UFO model are coherent when they should"""
    
    
    def test_sm_equivalence(self):
        """Test the UFO and MG4 SM model correspond to the same model """
        
        # import UFO model
        sm_path = import_ufo.find_ufo_path('sm')
        ufo_model = import_ufo.import_model(sm_path)
        ufo_model.pass_particles_name_in_mg_default()
        
        # import MG4 model
        model = base_objects.Model()
        v4_path = os.path.join(MG4DIR, 'models', 'sm_v4')
        if not os.path.isdir(v4_path):
            v4_path = os.path.join(MG4DIR, 'Models', 'sm')
            if not os.path.isdir(v4_path):
                raise MadGraph5Error, \
                      "Please provide a valid MG/ME path with -d"

        model.set('particles', files.read_from_file(
               os.path.join(v4_path,'particles.dat'),
               import_v4.read_particles_v4))
        model.set('interactions', files.read_from_file(
            os.path.join(v4_path,'interactions.dat'),
            import_v4.read_interactions_v4,
            model['particles']))
        model.pass_particles_name_in_mg_default()
        
        # Checking the particles
        for particle in model['particles']:
            ufo_particle = ufo_model.get("particle_dict")[particle['pdg_code']]
            self.check_particles(particle, ufo_particle)
        
        # Checking the interactions
        nb_vertex = 0
        ufo_vertices = []
        for ufo_vertex in ufo_model['interactions']:
            pdg_code_ufo = [abs(part['pdg_code']) for part in ufo_vertex['particles']]
            int_name = [part['name'] for part in ufo_vertex['particles']]
            rep = (pdg_code_ufo, int_name)
            pdg_code_ufo.sort()
            ufo_vertices.append(pdg_code_ufo)
        mg4_vertices = []
        for vertex in model['interactions']:
            pdg_code_mg4 = [abs(part['pdg_code']) for part in vertex['particles']]
            pdg_code_mg4.sort()

            try:
                ufo_vertices.remove(pdg_code_mg4)
            except ValueError:
                mg4_vertices.append(pdg_code_mg4)

        self.assertEqual(ufo_vertices, [[25,25,25,25]])
        self.assertEqual(mg4_vertices, [])  

    def test_mssm_equivalence(self):
        """Test the UFO and MG4 MSSM model correspond to the same model """
        
        # import UFO model
        mssm_path = import_ufo.find_ufo_path('mssm')
        ufo_model = import_ufo.import_model(mssm_path)
        #converter = import_ufo.UFOMG5Converter(model)
        #ufo_model = converter.load_model()
        ufo_model.pass_particles_name_in_mg_default()
        
        # import MG4 model
        model = base_objects.Model()
        if not MG4DIR:
            raise MadGraph5Error, "Please provide a valid MG/ME path with -d"
        v4_path = os.path.join(MG4DIR, 'models', 'mssm_v4')
        if not os.path.isdir(v4_path):
            v4_path = os.path.join(MG4DIR, 'Models', 'mssm')
            if not os.path.isdir(v4_path):
                raise MadGraph5Error, \
                      "Please provide a valid MG/ME path with -d"

        model.set('particles', files.read_from_file(
               os.path.join(v4_path,'particles.dat'),
               import_v4.read_particles_v4))
        model.set('interactions', files.read_from_file(
            os.path.join(v4_path,'interactions.dat'),
            import_v4.read_interactions_v4,
            model['particles']))
        
        #model.pass_particles_name_in_mg_default()
        # Checking the particles
        for particle in model['particles']:
            ufo_particle = ufo_model.get("particle_dict")[particle['pdg_code']]
            self.check_particles(particle, ufo_particle)

        # Skip test below until equivalence has been created by Benj and Claude
        return

        
        # Checking the interactions
        nb_vertex = 0
        ufo_vertices = []
        for ufo_vertex in ufo_model['interactions']:
            pdg_code_ufo = [abs(part['pdg_code']) for part in ufo_vertex['particles']]
            int_name = [part['name'] for part in ufo_vertex['particles']]
            rep = (pdg_code_ufo, int_name)
            pdg_code_ufo.sort()
            ufo_vertices.append(pdg_code_ufo)
        mg4_vertices = []
        for vertex in model['interactions']:
            pdg_code_mg4 = [abs(part['pdg_code']) for part in vertex['particles']]
            pdg_code_mg4.sort()

            try:
                ufo_vertices.remove(pdg_code_mg4)
            except ValueError:
                mg4_vertices.append(pdg_code_mg4)

        self.assertEqual(ufo_vertices, [])  
        self.assertEqual(mg4_vertices, [])  
  
            
    
    def check_particles(self, mg4_part, ufo_part):
        """ check that the internal definition for a particle comming from mg4 or
        comming from the UFO are the same """
        
        not_equiv = ['charge', 'mass','width',
                        'texname','antitexname','line']
        
        if abs(mg4_part['pdg_code']) != abs(ufo_part['pdg_code']):
            print '%s non equivalent particle' % mg4_part['name']
            return
        elif mg4_part['pdg_code'] != ufo_part['pdg_code']:
            self.assertFalse(mg4_part.get('is_part') == ufo_part.get('is_part'))
            not_equiv.append('is_part')
            not_equiv.append('pdg_code')
            not_equiv.append('name')
            not_equiv.append('antiname')
            self.assertEqual(mg4_part.get('name'), ufo_part.get('antiname'))
            
            
        
        for name in mg4_part.sorted_keys:
            if name in not_equiv:
                continue
            if name == 'propagator':
                if mg4_part.get('mass') == 'ZERO':
                    if ufo_part.get('propagator') == 0:
                        continue
            
            self.assertEqual(mg4_part.get(name), ufo_part.get(name), 
                    'fail for particle %s different property for %s, %s != %s' %
                    (mg4_part['name'], name, mg4_part.get(name), \
                                                            ufo_part.get(name)))
        
        
    def check_interactions(self, mg4_vertex, ufo_vertex, vname):
        """ check that the internal definition for a particle comming from mg4 or
        comming from the UFO are the same """
                
        # Checking only the color
        mg4_color = mg4_vertex.get('color')
        mg5_color = ufo_vertex.get('color')
        try:
            self.assertEqual(mg4_color, mg5_color) 
        except AssertionError:
            part_name =[part.get('name') for part in mg4_vertex.get('particles')]
            log = 'Potential different color structure for %s.\n' % part_name
            log += '    mg4 color : %s\n' % mg4_color
            log += '    mg5 color : %s\n' % mg5_color 
            logger.info(log)
            if part_name == ['g', 'g', 'g', 'g']:
                pass #too complex
            elif str(mg4_color) == '[]':
                self.assertEqual('[1 ]',str(mg5_color))
            elif len(part_name) == 3:
                if 'g' in part_name:
                    logger.info('and too complex to be tested')
                    pass # too complex
                else:
                    raise 
            else:
                mg5_color = copy.copy(mg5_color)
                for i,col in enumerate(mg5_color):
                    if len(col)==2:
                        simp = mg5_color[i][0].pair_simplify(mg5_color[i][1])
                        if simp:
                            mg5_color[i] = simp[0]
                            continue
                        simp = mg5_color[i][1].pair_simplify(mg5_color[i][0])
                        if simp:
                            mg5_color[i] = simp[0]
                            continue
                self.assertEqual(str(mg4_color), str(mg5_color))


        
    
        
class TestModelCreation(unittest.TestCase, CheckFileCreate):

    created_files = ['couplings.f', 'couplings1.f', 'couplings2.f', 'couplings3.f', 
                     'couplings4.f', 'coupl.inc', 'intparam_definition.inc',
                     'input.inc', 'param_read.f', 'makefile', 'tesprog.f', 
                     'testprog', 'rw_para.f', 'lha_read.f', 'printout.f', 
                     'formats.inc', 'makeinc.inc', 'ident_card.dat', 'libmodel.a',
                     'param_write.inc','coupl_write.inc','param_read.inc',
                     'testprog.f','param_card.dat']

    # clean all the tested files before and after any test
    def setUp(self):
        """ creating the full model from scratch """
        CheckFileCreate.setUp(self)
        os.system('cp %s %s' % (pjoin(MG5DIR,'Template', 'LO','Source',
                                      'make_opts'), '/tmp'))
        
        CheckFileCreate.clean_files(self)
        
        #picklefile = os.path.join(MG5DIR,'models','sm','model.pkl') 
        #if not files.is_uptodate(picklefile):
        #    sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model('sm')
        #else:
        #    model = save_load_object.load_from_file(picklefile)
            
        export_v4.UFO_model_to_mg4(model, self.output_path).build()
        
#    tearDown = CheckFileCreate.clean_files

    def test_all(self):
        """ test all the files"""
        self.check_intparam_definition_creation()
        self.check_compilation()
        
        
    def check_compilation(self):
        """check that the model compile return correct output"""
        #Check the cleaning
        join = lambda p: os.path.join(self.output_path, p)
        self.assertFalse(os.path.exists(self.give_pos('testprog')))
        try:
            os.remove(join('../param_card.inc'))
        except:
            pass
        # prepare for a local compilation
        
        subprocess.call(['python','write_param_card.py'], cwd=os.path.join(MG5DIR,'models','sm'),
                        stdout=subprocess.PIPE)
        files.cp(os.path.join(MG5DIR,'models','sm','param_card.dat'),
                 join('param_card.dat'))
        
        text = file(join('makefile')).read().replace('../../Cards/param_card.dat','param_card.dat')
        open(join('makefile'),'w').write(text)
        #make ../param_card.inc 
        param_card = check_param_card.ParamCard(join('param_card.dat'))
        param_card.write_inc_file(join('../param_card.inc'), join('ident_card.dat'),
                                   join('param_card.dat'))
        
        subprocess.call(['make', 'testprog'], cwd=self.output_path,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertTrue(os.path.exists(self.give_pos('testprog')))
        
        os.chmod(os.path.join(self.output_path, 'testprog'), 0777)
        testprog = subprocess.Popen("./testprog", stdout=subprocess.PIPE,
                            cwd=self.output_path,
                            stderr=subprocess.STDOUT, shell=True)

        solutions = {'ymtau ': [1.7769999504089355], 'I4x33 ': [0.024123685681481218, 0.0], 'MTA ': [1.7769999504089355], 'GC_81 ': [0.0, 67.544], 'GC_5 ': [0.0, 0.094836], 'MZ ': [91.18800354003906], 'GC_27 ': [0.94484, 0.0], 'I1x33 ': [0.024123685681481218, 0.0], 'I3x33 ': [0.9448443987662922, 0.0], 'GC_95 ': [0.66811, 0.0], 'GC_60 ': [-0.37035, 0.0], 'ee__exp__2 ': [0.09483552005165403], 'aEWM1 ': [132.5070037841797], 'ytau ': [0.01020661671581679], 'GC_69 ': [-0.0, -190.38], 'GC_35 ': [-0.0, -0.42671], 'cw__exp__2 ': [0.7777535472599892], 'Gf ': [1.16639e-05], 'GC_59 ': [0.0, 0.08231], 'GC_21 ': [-0.94484, -0.0], 'ee ': [0.307953762847045], 'WZ ': [2.441404104232788], 'sw2 ': [0.22224645274001076], 'WT ': [1.5083359479904175], 'GC_80 ': [-0.0, -33.772], 'GC_57 ': [-0.0, -0.35482], 'sqrt__sw2 ': [0.4714302204356555], 'GC_67 ': [13.239, 0.0], 'GC_76 ': [-29.784, 0.0], 'GC_36 ': [0.0, 0.33188], 'GC_68 ': [-0.0, -63.46], 'GC_56 ': [0.10058, 0.0], 'sw__exp__2 ': [0.22224645274001076], 'GC_3 ': [-0.0, -0.30795], 'GC_54 ': [-0.10058, 0.0], 'WW ': [2.047600030899048], 'GC_70 ': [-26.266, 0.0], 'GC_66 ': [-13.239, 0.0], 'GC_38 ': [-0.0, -0.32662], 'GC_83 ': [-0.0, -0.017058], 'GC_77 ': [-16.545, 0.0], 'gw ': [0.653232969584471], 'MH ': [125.0], 'ymb ': [4.199999809265137], 'complexi ': [0.0, 1.0], 'GC_37 ': [-0.32662, 0.0], 'conjg__CKM1x1 ': [1.0], 'GC_2 ': [0.0, 0.2053], 'GC_51 ': [0.0, 0.28804], 'GC_71 ': [-0.0, -26.266], 'GC_39 ': [0.0, 0.32662], 'GC_82 ': [-0.017058, 0.0], 'GC_55 ': [-0.0, -0.10058], 'GC_78 ': [16.545, 0.0], 'GC_98 ': [-0.0072172, 0.0], 'GC_30 ': [-0.024124, -0.0], 'GC_15 ': [0.024124, 0.0], 'cw ': [0.881903366168873], 'yt ': [0.9448443987662922], 'sqrt__aEW ': [0.08687215260631942], 'vev ': [246.2184581018163], 'GC_79 ': [29.784, 0.0], 'GC_72 ': [0.0, 52.532], 'GC_1 ': [-0.0, -0.10265], 'conjg__CKM3x3 ': [1.0], 'GC_52 ': [-0.0, -0.57609], 'GC_100 ': [0.0, 0.46191], 'GC_65 ': [0.0, 0.27432], 'GC_12 ': [0.0, 1.4828], 'I2x33 ': [0.9448443987662922, 0.0], 'GC_94 ': [-0.0, -0.66811], 'sqrt__aS ': [0.3435112817874484], 'GC_31 ': [-0.0, -0.25774], 'aS ': [0.11800000071525575], 'MW__exp__2 ': [6467.21673128622], 'MZ__exp__4 ': [69143415.65084904], 'yb ': [0.024123685681481218], 'GC_99 ': [-0.0, -0.0072172], 'WH ': [0.006382339168339968], 'GC_96 ': [-0.010207, 0.0], 'MH__exp__2 ': [15625.0], 'GC_63 ': [0.0, 0.12671], 'GC_53 ': [0.0, 0.57609], 'GC_73 ': [26.266, 0.0], 'GC_64 ': [0.0, 0.084653], 'sw ': [0.4714302204356555], 'GC_9 ': [0.053768, 0.0], 'GC_32 ': [-0.0, -0.51548], 'muH ': [88.38834764831844], 'GC_7 ': [-0.053768, 0.0], 'aEW ': [0.0075467708984556505], 'vev__exp__2 ': [60623.52911003587], 'MZ__exp__2 ': [8315.251989618177], 'GC_97 ': [0.010207, 0.0], 'GC_62 ': [0.0, 0.37035], 'GC_74 ': [-24.765, 0.0], 'g1 ': [0.34919218438279087], 'GC_10 ': [-1.2177, 0.0], 'GC_8 ': [0.0, 0.053768], 'CKM3x3 ': [1.0], 'MW ': [80.41900727617956], 'MT ': [172.0], 'GC_33 ': [-0.0, -0.77321], 'GC_6 ': [0.0, 0.18967], 'GC_4 ': [0.0, 0.30795], 'MB ': [4.699999809265137], 'GC_61 ': [0.0, -0.20573], 'ymt ': [164.5], 'GC_75 ': [24.765, 0.0], 'G__exp__2 ': [1.4828317414825511], 'lam ': [0.1288691060169027], 'GC_50 ': [-0.0, -0.28804], 'GC_34 ': [0.0, 0.21336], 'GC_11 ': [0.0, 1.2177], 'sqrt__2 ': [1.4142135623730951], 'GC_58 ': [-0.0, -0.027437]}
        #solutions = {'GC_5 ': [0.0, 0.094836], 'mdl_MW ': [80.41900727617956], 'mdl_yb ': [0.024123685681481218], 'mdl_sw__exp__2 ': [0.22224645274001076], 'mdl_conjg__CKM3x3 ': [1.0], 'GC_56 ': [0.10058, 0.0], 'mdl_MH ': [125.0], 'GC_95 ': [0.66811, 0.0], 'mdl_I4x33 ': [0.024123685681481218, 0.0], 'mdl_complexi ': [0.0, 1.0], 'aEWM1 ': [132.5070037841797], 'GC_69 ': [-0.0, -190.38], 'GC_35 ': [-0.0, -0.42671], 'mdl_Gf ': [1.16639e-05], 'mdl_gw ': [0.653232969584471], 'mdl_conjg__CKM1x1 ': [1.0], 'mdl_sqrt__aEW ': [0.08687215260631942], 'GC_59 ': [0.0, 0.08231], 'GC_21 ': [-0.94484, -0.0], 'GC_4 ': [0.0, 0.30795], 'mdl_cw ': [0.881903366168873], 'GC_80 ': [-0.0, -33.772], 'GC_64 ': [0.0, 0.084653], 'GC_57 ': [-0.0, -0.35482], 'GC_76 ': [-29.784, 0.0], 'GC_67 ': [13.239, 0.0], 'mdl_vev__exp__2 ': [60623.52911003587], 'mdl_I3x33 ': [0.9448443987662922, 0.0], 'GC_36 ': [0.0, 0.33188], 'mdl_I1x33 ': [0.024123685681481218, 0.0], 'GC_81 ': [0.0, 67.544], 'mdl_sw2 ': [0.22224645274001076], 'GC_68 ': [-0.0, -63.46], 'mdl_ytau ': [0.01020661671581679], 'GC_100 ': [0.0, 0.46191], 'GC_3 ': [-0.0, -0.30795], 'GC_54 ': [-0.10058, 0.0], 'GC_70 ': [-26.266, 0.0], 'GC_66 ': [-13.239, 0.0], 'GC_38 ': [-0.0, -0.32662], 'GC_83 ': [-0.0, -0.017058], 'GC_77 ': [-16.545, 0.0], 'GC_27 ': [0.94484, 0.0], 'GC_10 ': [-1.2177, 0.0], 'GC_37 ': [-0.32662, 0.0], 'GC_60 ': [-0.37035, 0.0], 'GC_2 ': [0.0, 0.2053], 'mdl_muH ': [88.38834764831844], 'mdl_MT ': [172.0], 'mdl_WH ': [0.006382339168339968], 'GC_51 ': [0.0, 0.28804], 'GC_71 ': [-0.0, -26.266], 'GC_39 ': [0.0, 0.32662], 'GC_82 ': [-0.017058, 0.0], 'mdl_sw ': [0.4714302204356555], 'GC_55 ': [-0.0, -0.10058], 'GC_61 ': [0.0, -0.20573], 'mdl_cw__exp__2 ': [0.7777535472599892], 'mdl_ymt ': [164.5], 'GC_78 ': [16.545, 0.0], 'mdl_CKM3x3 ': [1.0], 'GC_30 ': [-0.024124, -0.0], 'GC_15 ': [0.024124, 0.0], 'mdl_aEW ': [0.0075467708984556505], 'mdl_sqrt__sw2 ': [0.4714302204356555], 'mdl_I2x33 ': [0.9448443987662922, 0.0], 'GC_72 ': [0.0, 52.532], 'GC_1 ': [-0.0, -0.10265], 'GC_52 ': [-0.0, -0.57609], 'GC_65 ': [0.0, 0.27432], 'GC_12 ': [0.0, 1.4828], 'mdl_ymb ': [4.199999809265137], 'mdl_ee ': [0.307953762847045], 'GC_79 ': [29.784, 0.0], 'mdl_sqrt__2 ': [1.4142135623730951], 'GC_31 ': [-0.0, -0.25774], 'aS ': [0.11800000071525575], 'GC_99 ': [-0.0, -0.0072172], 'mdl_vev ': [246.2184581018163], 'GC_96 ': [-0.010207, 0.0], 'GC_63 ': [0.0, 0.12671], 'GC_53 ': [0.0, 0.57609], 'GC_73 ': [26.266, 0.0], 'mdl_MZ__exp__2 ': [8315.251989618177], 'mdl_WZ ': [2.441404104232788], 'GC_9 ': [0.053768, 0.0], 'mdl_g1 ': [0.34919218438279087], 'GC_32 ': [-0.0, -0.51548], 'mdl_G ': [1.2177157884673053], 'mdl_WT ': [1.5083359479904175], 'GC_7 ': [-0.053768, 0.0], 'mdl_G__exp__2 ': [1.4828317414825511], 'GC_97 ': [0.010207, 0.0], 'GC_62 ': [0.0, 0.37035], 'GC_74 ': [-24.765, 0.0], 'mdl_MZ ': [91.18800354003906], 'mdl_MZ__exp__4 ': [69143415.65084904], 'GC_8 ': [0.0, 0.053768], 'mdl_yt ': [0.9448443987662922], 'GC_98 ': [-0.0072172, 0.0], 'mdl_ee__exp__2 ': [0.09483552005165403], 'mdl_MB ': [4.699999809265137], 'GC_33 ': [-0.0, -0.77321], 'mdl_ymtau ': [1.7769999504089355], 'mdl_WW ': [2.047600030899048], 'GC_6 ': [0.0, 0.18967], 'mdl_MTA ': [1.7769999504089355], 'GC_75 ': [24.765, 0.0], 'GC_94 ': [-0.0, -0.66811], 'mdl_MW__exp__2 ': [6467.21673128622], 'mdl_MH__exp__2 ': [15625.0], 'GC_50 ': [-0.0, -0.28804], 'GC_34 ': [0.0, 0.21336], 'GC_11 ': [0.0, 1.2177], 'mdl_sqrt__aS ': [0.3435112817874484], 'GC_58 ': [-0.0, -0.027437], 'mdl_lam ': [0.1288691060169027]}
        nb_value = 0
        for line in testprog.stdout:
            self.assertTrue('Warning' not in line)
            if '=' not in line:
                continue
            split = line.split('=')
            variable = split[0].lstrip()
            if variable.startswith('mdl_'):
                variable = variable[4:]

            if ',' in line:
                value = eval(split[1])
            else:
                value=[float(numb) for numb in split[1].split()]
            nb_value +=1

            for i, singlevalue in enumerate(value):
                #try:
                    self.assertAlmostEqual(singlevalue,
                                           solutions[variable][i],
                                           places=6,
                        msg='fail to be equal for param %s : %s != %s' % \
                            (variable, singlevalue, solutions[variable][i]))
                #except Exception as error:
                #    print variable
                #    if i == 0:
                #        solutions[variable] = [singlevalue]
                #    else:
                #        solutions[variable].append(singlevalue)
        
        self.assertEqual(nb_value, 116)
        
        

    def check_intparam_definition_creation(self):
        """ test the creation of a valid intparam_definition"""

        # Check that any definition appears only once:
        alreadydefine = []
        for line in self.ReturnFile('intparam_definition.inc'):
            if 'ENDIF' in line:
                self.assertEqual(len(alreadydefine), 34)
            if '=' not in line:
                continue
            new_def = line.split('=')[0].lstrip()
            # Check that is the firsttime that this definition is done
            self.assertFalse(new_def in alreadydefine)
            alreadydefine.append(new_def)
        alreadydefine = [name.lower() for name in alreadydefine]
        alreadydefine.sort()
        solution = ['as ', 'g ', 'gal(1) ', 'gal(2) ', 'mdl_aew ', 'mdl_ckm3x3 ', 'mdl_complexi ', 'mdl_conjg__ckm1x1 ', 'mdl_conjg__ckm3x3 ', 'mdl_cw ', 'mdl_cw__exp__2 ', 'mdl_ee ', 'mdl_ee__exp__2 ', 'mdl_g1 ', 'mdl_g__exp__2 ', 'mdl_gw ', 'mdl_i1x33 ', 'mdl_i2x33 ', 'mdl_i3x33 ', 'mdl_i4x33 ', 'mdl_lam ', 'mdl_mh__exp__2 ', 'mdl_muh ', 'mdl_mw ', 'mdl_mw__exp__2 ', 'mdl_mz__exp__2 ', 'mdl_mz__exp__4 ', 'mdl_sqrt__2 ', 'mdl_sqrt__aew ', 'mdl_sqrt__as ', 'mdl_sqrt__sw2 ', 'mdl_sw ', 'mdl_sw2 ', 'mdl_sw__exp__2 ', 'mdl_vev ', 'mdl_vev__exp__2 ', 'mdl_yb ', 'mdl_yt ', 'mdl_ytau ']
        self.assertEqual(alreadydefine, solution)
        


      

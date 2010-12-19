################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
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

logger = logging.getLogger('madgraph.test.model')


class CheckFileCreate():
    """Check that the files are correctly created"""

    output_path = '/tmp/' # work only on LINUX but that's ok for the test routine
    created_files =[]

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
        ufo_model = import_ufo.import_model('sm')
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

        self.assertEqual(ufo_vertices, [])  
        self.assertEqual(mg4_vertices, [])  

    def test_mssm_equivalence(self):
        """Test the UFO and MG4 MSSM model correspond to the same model """
        
        # import UFO model
        ufo_model = import_ufo.import_model('mssm')
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
        
        model.pass_particles_name_in_mg_default()
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
                        'texname','antitexname']
        
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
        CheckFileCreate.clean_files(self)
        
        picklefile = os.path.join(MG5DIR,'models','sm','model.pkl') 
        if not files.is_uptodate(picklefile):
            model = import_ufo.import_model('sm')
        else:
            model = save_load_object.load_from_file(picklefile)
            
        export_v4.UFO_model_to_mg4(model, self.output_path).build()
        
#    tearDown = CheckFileCreate.clean_files

    def test_all(self):
        """ test all the files"""
        self.check_intparam_definition_creation()
        self.check_compilation()
        
        
    def check_compilation(self):
        """check that the model compile return correct output"""
        #Check the cleaning
        self.assertFalse(os.path.exists(self.give_pos('testprog')))
        subprocess.call(['make', 'testprog'], cwd=self.output_path,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertTrue(os.path.exists(self.give_pos('testprog')))
        
        os.chmod(os.path.join(self.output_path, 'testprog'), 0777)
        testprog = subprocess.Popen("./testprog", stdout=subprocess.PIPE,
                            cwd=self.output_path,
                            stderr=subprocess.STDOUT, shell=True)
        
        
        solutions ={'sqrt__aEW ': [0.086872153846781555], 'ymtau ': [1.7769999999999999], 'GC_5 ': [0.0, 1.2177199999999999], 'sqrt__sw2 ': [0.4714302554840723], 'sw__exp__2 ': [0.22224648578577769], 'GC_11 ': [0.0, 0.46190999999999999], 'GC_3 ': [-0.0, -0.30795], 'aEW ': [0.0075467711139788835], 'MZ__exp__2 ': [8315.2513440000002], 'MZ ': [91.188000000000002], 'WW ': [2.0476000000000001], 'GC_1 ': [-0.0, -0.10265000000000001], 'GC_9 ': [0.0, 0.33188000000000001], 'GC_7 ': [0.0, 0.57608999999999999], 'GC_10 ': [0.0, 0.21335999999999999], 'ee__exp__2 ': [0.094835522759998875], 'aEWM1 ': [132.50700000000001], 'gw ': [0.6532329303475799], 'ytau ': [0.010206617000654717], 'GC_8 ': [-0.0, -0.42670999999999998], 'GC_16 ': [0.0, 0.30795], 'GC_19 ': [0.0, 0.37035000000000001], 'MTA ': [1.7769999999999999], 'sqrt__aS ': [0.34351128074635334], 'MH ': [120.0], 'GC_23 ': [0.0, 67.543689999999998], 'aS ': [0.11799999999999999], 'ymb ': [4.2000000000000002], 'MZ__exp__4 ': [69143404.913893804], 'complexi ': [0.0, 1.0], 'G ': [1.2177157847767195], 'yb ': [0.024123686777011714], 'MW__exp__2 ': [6467.2159543705357], 'Gf ': [1.16639e-05], 'GC_21 ': [-0.0, -175.45394999999999], 'ee ': [0.30795376724436879], 'WZ ': [2.4414039999999999], 'v__exp__2 ': [60623.529110035903], 'v ': [246.21845810181637], 'WH ': [0.005753088], 'cw__exp__2 ': [0.77775351421422245], 'GC_6 ': [0.0, 1.4828300000000001], 'GC_15 ': [0.0, 0.082309999999999994], 'sw2 ': [0.22224648578577766], 'GC_2 ': [0.0, 0.20530000000000001], 'GC_24 ': [-0.0, -0.017059999999999999], 'MB ': [4.7000000000000002], 'WT ': [1.5083359999999999], 'GC_18 ': [0.0, 0.094839999999999994], 'GC_14 ': [-0.0, -0.027439999999999999], 'GC_4 ': [-1.2177199999999999, 0.0], 'MH__exp__2 ': [14400.0], 'ymt ': [164.5], 'G__exp__2 ': [1.4828317324943818], 'MT ': [174.30000000000001], 'GC_13 ': [0.0, 0.28804000000000002], 'lam ': [0.11876576810517747], 'GC_25 ': [-0.0, -0.66810999999999998], 'GC_12 ': [-0.0, -0.28804000000000002], 'sw ': [0.4714302554840723], 'gw__exp__2 ': [0.42671326129048615], 'GC_26 ': [-0.0, -0.0072199999999999999], 'g1 ': [0.34919219678733299], 'GC_22 ': [0.0, 52.532339999999998], 'MW ': [80.419002445756163], 'muH ': [84.852813742385706], 'GC_17 ': [-0.0, -0.35482000000000002], 'cw ': [0.88190334743339216], 'sqrt__2 ': [1.4142135623730951], 'GC_20 ': [0.0, 0.27432000000000001], 'yt ': [0.944844398766292]}





        nb_value = 0
        for line in testprog.stdout:
            self.assertTrue('Warning' not in line)
            if '=' not in line:
                continue
            split = line.split('=')
            variable = split[0].lstrip()
            if ',' in line:
                value = eval(split[1])
            else:
                value=[float(numb) for numb in split[1].split()]
            nb_value +=1
            for i, singlevalue in enumerate(value):
                #try:
                    self.assertAlmostEqual(singlevalue, solutions[variable][i], \
                        msg='fail to be equal for param %s : %s != %s' % \
                            (variable, singlevalue, solutions[variable][i]))
                #except:
                #    print i, singlevalue, [variable]
                #    if i == 0:
                #        solutions[variable] = [singlevalue]
                #    else:
                #        solutions[variable].append(singlevalue)

        self.assertEqual(nb_value, 71)
        
        

    def check_intparam_definition_creation(self):
        """ test the creation of a valid intparam_definition"""

        # Check that any definition appears only once:
        alreadydefine = []
        for line in self.ReturnFile('intparam_definition.inc'):
            if 'ENDIF' in line:
                self.assertEqual(len(alreadydefine), 29)
                
            if '=' not in line:
                continue
            
            new_def = line.split('=')[0].lstrip()
            # Check that is the firsttime that this definition is done
            self.assertFalse(new_def in alreadydefine)
            alreadydefine.append(new_def)
        alreadydefine = [name.lower() for name in alreadydefine]
        alreadydefine.sort()
        solution= ['aew ', 'complexi ', 'cw ', 'cw__exp__2 ', 'dum0 ', 'dum1 ', 'ee ', 'ee__exp__2 ', 'g ', 'g1 ', 'g__exp__2 ', 'gal(1) ', 'gal(2) ', 'gw ', 'gw__exp__2 ', 'lam ', 'mh__exp__2 ', 'muh ', 'mw ', 'mw__exp__2 ', 'mz__exp__2 ', 'mz__exp__4 ', 'sqrt__2 ', 'sqrt__aew ', 'sqrt__as ', 'sqrt__sw2 ', 'sw ', 'sw2 ', 'sw__exp__2 ', 'v ', 'v__exp__2 ', 'yb ', 'yt ', 'ytau ']
        self.assertEqual(len(alreadydefine), len(solution))
        for i in range(len(alreadydefine)):
            self.assertEqual(alreadydefine[i], solution[i])
                                       


      

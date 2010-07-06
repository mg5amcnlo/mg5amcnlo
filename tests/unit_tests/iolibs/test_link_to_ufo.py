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
from madgraph import MG5DIR
import subprocess
import shutil
import unittest
import os

import madgraph.core.base_objects as base_objects
import madgraph.iolibs.convert_ufo2mg4 as ufo2mg4
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4


file_dir_path = os.path.dirname(os.path.realpath( __file__ ))
root_path = os.path.join(file_dir_path, os.pardir, os.pardir, os.pardir)


class CompareMG4WithUFOModel(unittest.TestCase):
    """checking if the MG4 model and the UFO model are coherent when they should"""
    
    
    def test_model_equivalence(self):
        """ test the UFO and MG4 model correspond to the same model """
        
        # import UFO model
        import models.sm as model
        converter = import_ufo.converter_ufo_mg5(model)
        ufo_model = converter.load_model()
        
        # import MG4 model
        model = base_objects.Model()
        model.set('particles', files.read_from_file(
               os.path.join(MG5DIR,'tests','input_files','v4_sm_particles.dat'),
               import_v4.read_particles_v4))
        model.set('interactions', files.read_from_file(
            os.path.join(MG5DIR,'tests','input_files','v4_sm_interactions.dat'),
            import_v4.read_interactions_v4,
            model['particles']))
        
        # Checking the particles
        for particle in model['particles']:
            ufo_particle = ufo_model["particle_dict"][particle['pdg_code']]
            self.check_particles(particle, ufo_particle)
        
        # Checking the interactions
        nb_vertex = 0
        for ufo_vertex in ufo_model['interactions']:
            pdg_code_ufo = [abs(part['pdg_code']) for part in ufo_vertex['particles']]
            int_name = [part['name'] for part in ufo_vertex['particles']]
            rep = (pdg_code_ufo, int_name)
            pdg_code_ufo.sort()
            for vertex in model['interactions']:
                pdg_code_mg4 = [abs(part['pdg_code']) for part in vertex['particles']]
                pdg_code_mg4.sort()
                
                if pdg_code_mg4 == pdg_code_ufo:
                    nb_vertex += 1
                    self.check_interactions(vertex, ufo_vertex, rep )
            
        self.assertEqual(nb_vertex, 67)
            
    
    def check_particles(self, mg4_part, ufo_part):
        """ check that the internal definition for a particle comming from mg4 or
        comming from the UFO are the same """
        
        not_equiv = ['charge', 'mass','width','name','antiname',
                        'texname','antitexname']
        
        if abs(mg4_part['pdg_code']) != abs(ufo_part['pdg_code']):
            print '%s non equivalent particle' % mg4_part['name']
            return
        elif mg4_part['pdg_code'] != ufo_part['pdg_code']:
            self.assertFalse(mg4_part.get('is_part') == ufo_part.get('is_part'))
            not_equiv.append('is_part')
            not_equiv.append('pdg_code')
            
        
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
        
        # don't check this for the moment. Too difficult to compare
        return
        
        # Checking only the color
        name = 'color'
        self.assertEqual(mg4_vertex.get(name), ufo_vertex.get(name), 
            'fail for interactions %s different property for %s, %s != %s' % \
            (vname, name, mg4_vertex.get(name), ufo_vertex.get(name) ) )
        
class TestPythonToFrotran(unittest.TestCase):
    
    def test_convert_str(self):
        """ python to fortran expression is working"""
        
        expr = 'cmath.sqrt(2)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2)' 
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2.)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2.5)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.500000d+00)')

        
        expr = '(ee**2*IMAG/(2.*sw**2) * (cmath.sin(sqrt(2)*ee)/3.'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 
        '(ee**2*imag/(2.000000d+00*sw**2) * (sin(dsqrt(2.000000d+00)*ee)/3.000000d+00')
    
    def test_convert_number(self):
        """ test it can convert number in fortran string"""
        
        expr = 2
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.000000d+00')  
        
        expr = 0.23
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.300000d-01')  
        
        expr = 2.5e6
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.500000d+06')
        
        expr = 0.0000116639  
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '1.166390d-05')        
        
        
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
        
        import models.sm as model
        ufo2mg4.export_to_mg4(model, self.output_path)
        
        
    tearDown = CheckFileCreate.clean_files

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
        
        
        solutions = {'ymtau ': [1.7769999999999999], 'GC_5 ': [0.0, 0.19650000000000001], 'GC_110 ': [0.0, 0.44666], 'GC_134 ': [0.0, 0.45849000000000001], 'GC_81 ': [0.0, -0.0], 'MZ ': [91.188000000000002], 'GC_27 ': [0.0, 0.0], 'GC_56 ': [15.746119999999999, 0.0], 'GC_95 ': [-0.0, -0.0], 'GC_57 ': [-15.746119999999999, 0.0], 'GC_60 ': [-13.817449999999999, 0.0], 'aEWM1 ': [127.90000000000001], 'GC_29 ': [0.0, 0.44666], 'ytau ': [0.010206617000654717], 'GC_69 ': [0.019089999999999999, 0.0], 'GC_16 ': [-0.0, -0.47505999999999998], 'GC_35 ': [0.0, 0.28381000000000001], 'GC_117 ': [0.0, -0.0], 'GC_122 ': [0.0, 0.44666], 'GC_45 ': [29.563569999999999, 0.0], 'CKM31 ': [0.0, 0.0], 'GC_118 ': [-0.0, 0.0], 'MH__EXP__2 ': [14400.0], 'COMPLEXI ': [0.0, 1.0], 'G ': [1.2135809144852661], 'GC_14 ': [0.0, 0.32218000000000002], 'Gf ': [1.16639e-05], 'GC_59 ': [0.0, 0.27432000000000001], 'GC_21 ': [-0.0, -0.32419999999999999], 'ee ': [0.31345100004952897], 'WZ ': [2.4413999999999998], 'ye ': [0.0], 'GC_120 ': [0.0018400000000000001, 0.0], 'GC_4 ': [0.0, 0.31345000000000001], 'CONJG__CKM21 ': [-0.2257725604285693, -0.0], 'WT ': [1.50834], 'GC_18 ': [-25.021280000000001, 0.0], 'GC_85 ': [-0.0, -0.0], 'CONJG__CKM11 ': [0.97418004031982097, -0.0], 'GC_67 ': [0.0, 67.543689999999998], 'GC_114 ': [0.0, -0.0], 'GC_28 ': [-0.0, -0.10352], 'GC_76 ': [0.0079500000000000005, 0.0], 'GC_36 ': [-0.0, -0.56760999999999995], 'GC_17 ': [-0.0, -0.71258999999999995], 'ym ': [0.0], 'GC_115 ': [-0.0, 0.0], 'GC_68 ': [-0.0, -0.019089999999999999], 'GC_103 ': [-0.0, -0.0072199999999999999], 'GC_88 ': [0.0, 0.0], 'GC_20 ': [0.0, 0.21021999999999999], 'GC_130 ': [-0.0, 0.0], 'GC_123 ': [-0.0079500000000000005, 0.0], 'GC_100 ': [1.0011300000000001, 0.0], 'GC_3 ': [-0.0, -0.31345000000000001],  'gw__EXP__2 ': [0.4204345654976559], 'CONJG__CKM22 ': [0.97418004031982097, -0.0], 'GC_93 ': [-0.0, -0.0], 'yd ': [0.0], 'GC_54 ': [0.37035000000000001, 0.0], 'WW ': [2.0476000000000001], 'GC_70 ': [-0.0, -0.0], 'GC_66 ': [25.021280000000001, 0.0], 'GC_38 ': [-0.10162, 0.0], 'MZ__EXP__2 ': [8315.2513440000002], 'GC_83 ': [-0.0, 0.0], 'GC_108 ': [0.0, 0.0], 'GC_26 ': [0.0, 0.10352], 'gw ': [0.64840925772050473], 'GC_44 ': [-29.563569999999999, 0.0], 'GC_19 ': [25.021280000000001, 0.0], 'MH ': [120.0], 'GC_51 ': [0.0, 0.098250000000000004], 'ymb ': [4.7000000000000002], 'GC_37 ': [0.0, 0.56760999999999995], 'yu ': [0.0], 'GC_124 ': [0.0, -0.0], 'GC_47 ': [-0.0, -0.028850000000000001], 'SQRT__aEW ': [0.088422894590285753], 'GC_101 ': [-0.01021, 0.0], 'CONJG__CKM23 ': [0.0, -0.0], 'GC_90 ': [0.0, 0.0], 'GC_129 ': [0.0, -0.0], 'GC_2 ': [0.0, 0.20896999999999999], 'GC_133 ': [-0.0, 0.0], 'CONJG__CKM33 ': [1.0, -0.0], 'CONJG__CKM13 ': [0.0, -0.0], 'GC_71 ': [-0.0, -0.0], 'GC_49 ': [0.0, 0.31345000000000001], 'GC_39 ': [-0.0, -0.10162], 'GC_109 ': [0.0, 0.0], 'GC_82 ': [-0.0, -0.0], 'v__EXP__2 ': [60623.529110035888], 'GC_91 ': [-0.0, -0.0], 'Me ': [0.00051099999999999995], 'SQRT__aS ': [0.34234485537247378], 'GC_78 ': [-0.0, -0.0], 'GC_98 ': [0.0, 0.0], 'GC_131 ': [0.0, 0.0], 'GC_30 ': [0.0, 0.0], 'MW ': [79.825163827442964], 'ymc ': [1.4199999999999999], 'GC_125 ': [0.0, 0.0], 'cw ': [0.87539110220032201], 'yc ': [0.008156103624608722], 'G__EXP__2 ': [1.4727786360028947], 'yt ': [1.0011330012459863], 'ee__EXP__2 ': [0.098251529432049817], 'GC_126 ': [0.0, -0.0], 'CONJG__CKM32 ': [0.0, -0.0], 'GC_72 ': [-0.027, -0.0], 'GC_112 ': [-0.0, 0.0], 'GC_48 ': [0.0, 0.086550000000000002], 'cw__EXP__2 ': [0.76630958181149467], 'GC_1 ': [-0.0, -0.10448], 'GC_52 ': [0.0, -0.19725999999999999], 'MM ': [0.10566], 'GC_136 ': [-1.0011300000000001, 0.0], 'GC_65 ': [-25.021280000000001, 0.0], 'CKM11 ': [0.97418004031982097, 0.0], 'GC_12 ': [0.0, 0.56760999999999995], 'GC_25 ': [0.0, 0.44666], 'ys ': [0.0], 'GC_79 ': [0.0, 0.0], 'GC_41 ': [-25.87969, 0.0], 'GC_31 ': [0.0, 0.0], 'aS ': [0.1172], 'GC_106 ': [-0.0, -0.0], 'yb ': [0.026995554250465494], 'SQRT__2 ': [1.4142135623730951], 'GC_55 ': [0.0, 0.12366000000000001], 'CKM21 ': [-0.2257725604285693, 0.0], 'GC_99 ': [0.0, 0.0], 'GC_127 ': [-0.0, 0.0], 'WH ': [0.0057530899999999998], 'MD ': [0.0050400000000000002], 'CONJG__CKM31 ': [0.0, -0.0], 'GC_132 ': [0.0, -0.0], 'GC_84 ': [0.0, 0.0], 'GC_96 ': [-0.70791000000000004, 0.0], 'MW__EXP__2 ': [6372.0567800781082], 'GC_63 ': [-0.0, -175.45394999999999], 'GC_53 ': [0.0, 0.37035000000000001], 'GC_128 ': [0.0, 0.0], 'GC_89 ': [-0.0, -0.0], 'GC_73 ': [-0.00577, 0.0], 'CKM12 ': [0.2257725604285693, 0.0], 'GC_102 ': [0.01021, 0.0], 'GC_64 ': [0.0, 51.75938], 'GC_13 ': [-0.0, -0.42043000000000003], 'sw ': [0.48341536817575986], 'CKM32 ': [0.0, 0.0], 'CONJG__CKM12 ': [0.2257725604285693, -0.0], 'GC_40 ': [0.10162, 0.0], 'GC_9 ': [-1.2135800000000001, 0.0], 'cabi ': [0.22773599999999999], 'GC_107 ': [0.0, 0.0], 'MU ': [0.0025500000000000002], 'GC_24 ': [0.0, 0.45849000000000001], 'GC_32 ': [0.0, 0.0], 'muH ': [84.852813742385706], 'MZ__EXP__4 ': [69143404.913893804], 'GC_7 ': [0.0, 0.056120000000000003], 'GC_87 ': [-0.0, 0.0], 'aEW ': [0.0078186082877247844], 'MC ': [1.4199999999999999], 'GC_97 ': [-0.0, -0.70791000000000004], 'GC_62 ': [-0.0, -58.484650000000002], 'SQRT__sw2 ': [0.48341536817575986], 'GC_74 ': [-0.0, -0.00577], 'GC_11 ': [0.0, 1.47278], 'GC_116 ': [0.0, 0.0], 'GC_10 ': [0.0, 1.2135800000000001], 'GC_8 ': [0.056120000000000003, 0.0], 'CKM33 ': [1.0, 0.0], 'MTA ': [1.7769999999999999], 'CKM13 ': [0.0, 0.0], 'GC_104 ': [0.0072199999999999999, 0.0], 'GC_23 ': [0.32419999999999999, 0.0], 'GC_15 ': [-0.0, -0.23752999999999999], 'GC_113 ': [0.0, 0.10352], 'CKM23 ': [0.0, 0.0], 'MT ': [174.30000000000001], 'GC_33 ': [0.0, 0.45849000000000001], 'v ': [246.21845810181634], 'GC_135 ': [0.027, -0.0], 'GC_6 ': [-0.056120000000000003, 0.0], 'CKM22 ': [0.97418004031982097, 0.0], 'sw2 ': [0.23369041818850544], 'GC_119 ': [0.0, -0.10352], 'MB ': [4.7000000000000002], 'GC_111 ': [0.0, -0.0], 'GC_61 ': [13.817449999999999, 0.0], 'GC_77 ': [0.0, 0.0], 'GC_92 ': [0.0, 0.0], 'ymt ': [174.30000000000001], 'GC_86 ': [0.0, 0.0], 'GC_75 ': [-0.0018400000000000001, 0.0], 'GC_43 ': [25.87969, 0.0], 'GC_94 ': [-0.0, -0.0], 'lam ': [0.1187657681051775], 'GC_46 ': [-0.0, -33.771839999999997], 'GC_50 ': [-0.0, -0.35583999999999999], 'sw__EXP__2 ': [0.23369041818850547], 'GC_80 ': [-0.0, -0.0], 'GC_34 ': [-0.0, -0.28381000000000001], 'g1 ': [0.35806966653151989], 'GC_22 ': [0.0, 0.32419999999999999], 'GC_121 ': [-0.0, -0.0], 'GC_42 ': [-0.0, -25.87969], 'GC_58 ': [0.0, 0.07782], 'GC_105 ': [-0.0, 0.0], 'MS ': [0.104]}


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
#                try:
                    self.assertAlmostEqual(singlevalue, solutions[variable][i], 7, 'fail to be equal for param %s : %s != %s' % (variable, singlevalue, solutions[variable][i]))
#                except:
#                    print i, singlevalue, [variable]
#                    if i == 0:
#                        solutions[variable] = [singlevalue]
#                    else:
#                        solutions[variable].append(singlevalue)
#        print solutions
        self.assertEqual(nb_value, 213)
        
        

    def check_intparam_definition_creation(self):
        """ test the creation of a valid intparam_definition"""

        # Check that any definition appears only once:
        alreadydefine = []
        for line in self.ReturnFile('intparam_definition.inc'):
            if 'ENDIF' in line:
                self.assertEqual(len(alreadydefine), 53)
                
            if '=' not in line:
                continue
            
            new_def = line.split('=')[0].lstrip()
            # Check that is the firsttime that this definition is done
            self.assertFalse(new_def in alreadydefine)
            alreadydefine.append(new_def)
            
        self.assertEqual(alreadydefine, ['AEW ', 'SQRT__AS ', 'G ', 'YE ', 'YM ', 'YU ', 'YD ', 'YS ', 'CKM11 ', 'CKM12 ', 'CKM13 ', 'CKM21 ', 'CKM22 ', 'CKM23 ', 'CKM31 ', 'CKM32 ', 'CKM33 ', 'MZ__EXP__2 ', 'MZ__EXP__4 ', 'SQRT__2 ', 'MW ', 'SQRT__AEW ', 'EE ', 'MW__EXP__2 ', 'SW2 ', 'CW ', 'SQRT__SW2 ', 'SW ', 'G1 ', 'GW ', 'V ', 'MH__EXP__2 ', 'V__EXP__2 ', 'LAM ', 'YB ', 'YC ', 'YT ', 'YTAU ', 'MUH ', 'COMPLEXI ', 'EE__EXP__2 ', 'GW__EXP__2 ', 'CW__EXP__2 ', 'SW__EXP__2 ', 'CONJG__CKM11 ', 'CONJG__CKM12 ', 'CONJG__CKM13 ', 'CONJG__CKM21 ', 'CONJG__CKM22 ', 'CONJG__CKM23 ', 'CONJG__CKM31 ', 'CONJG__CKM32 ', 'CONJG__CKM33 ', 'G__EXP__2 ', 'GAL(1) ', 'GAL(2) ', 'DUM0 ', 'DUM1 '])


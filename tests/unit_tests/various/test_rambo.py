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

import madgraph.various.rambo as rambo
import aloha.template_files.wavefunctions as wavefunctions
import tests.unit_tests as unittest


class test_rambo(unittest.TestCase):
    """ Test the Rambo module """
    
    def setUp(self):
        self.m2_zero = rambo.FortranList(2)
        self.m1 = rambo.FortranList(1)
        self.m1[1] = 100
        self.m2 = rambo.FortranList(2)
        self.m2[1] , self.m2[2] = 100, 200
        
    def test_rambo_validity_check(self):
        """test if it raise error if wrong input"""

        # not enough energy
        self.assertRaises(rambo.RAMBOError, rambo.RAMBO, 2,150,self.m2)
        
        # not valid mass
        self.assertRaises(AssertionError, rambo.RAMBO, 2,1500,[1,2])
        
        # not valid mass
        self.assertRaises(AssertionError, rambo.RAMBO, 2,1500,self.m1)
    
        # at least 2 particles in final state
        self.assertRaises(AssertionError, rambo.RAMBO, 1, 1500, self.m1)

    def test_massless(self):
        """ Rambo can generate impulsion for massless final state """
        
        P, wgt = rambo.RAMBO(2,150, self.m2_zero)
        for i in range(1,3):
            self.assertAlmostEqual(0., P[(4,i)]**2 - P[(1,i)]**2 - P[(2,i)]**2 - P[(3,i)]**2)
         
    def test_massivecase(self):
        """ Rambo can generate impulsion for massive final state """
        
        P, wgt = rambo.RAMBO(2,500, self.m2)
        for i in range(1,3):
            self.assertAlmostEqual(self.m2[i]**2, P[(4,i)]**2 - P[(1,i)]**2 - P[(2,i)]**2 - P[(3,i)]**2)
        

class test_wavefunctions(unittest.TestCase):
    """ check that the wavefunctions TXXXX, IXXXXX, ... are correctly define"""
    
    def test_T(self):
        """check T wavefunctions against fortran output"""
        
        #input
        P = [624.99999999999989, 83.193213332874592, 333.62309211609107, -149.66469744815910 ]
        M = 500
        
        # first
        NHEL = 2
        NST = 1
        
        solution = [ 
             complex(  624.99999999999989     , -149.66469744815910     ),
             complex(  83.193213332874592     ,  333.62309211609107     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex(-0.46606677614840392     ,-9.36959948731358322E-002),
             complex( 0.13607969464455591     ,-0.17618862729680237     ),
             complex( 4.42705319225060942E-002,-0.44483078948812155     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex( 0.13607969464455591     ,-0.17618862729680237     ),
             complex( 4.57095198364004252E-002, 9.36959948731358322E-002),
             complex( 0.17753457473164128     , 0.11092428444383282     ),
             complex(  0.0000000000000000     ,  0.0000000000000000     ),
             complex( 4.42705319225060942E-002,-0.44483078948812155     ),
             complex( 0.17753457473164128     , 0.11092428444383282     ),
             complex( 0.42035725631200371     ,  0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        # NEXT
        NHEL = 1
        solution = [ 
                    complex(  624.99999999999989     , -149.66469744815910     ),
                    complex(  83.193213332874592     ,  333.62309211609107     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 3.62119349359586659E-002,-0.36385791873139811     ),
                    complex( 0.14521782752280418     , 9.07326566152192315E-002),
                    complex( 0.34383932052304755     ,  0.0000000000000000     ),
                    complex( 3.62119349359586659E-002,-0.36385791873139811     ),
                    complex( 2.67785531406523024E-002,-0.26907119516335065     ),
                    complex( 0.10738789070969987     ,-0.50596916746687115     ),
                    complex( 0.10304644292578606     , 0.24202971253800695     ),
                    complex( 0.14521782752280418     , 9.07326566152192315E-002),
                    complex( 0.10738789070969987     ,-0.50596916746687115     ),
                    complex( 0.43064907243145895     , 0.26907119516335065     ),
                    complex( 0.41323891148317965     ,-6.03532248932644386E-002),
                    complex( 0.34383932052304755     ,  0.0000000000000000     ),
                    complex( 0.10304644292578606     , 0.24202971253800695     ),
                    complex( 0.41323891148317965     ,-6.03532248932644386E-002),
                    complex(-0.45742762557211131     ,  0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        #NEXT
        NHEL = 0
        solution = [ 
                    complex(  624.99999999999989     , -149.66469744815910     ),
                    complex(  83.193213332874592     ,  333.62309211609107     ),
                    complex( 0.45927932677184580     ,  0.0000000000000000     ),
                    complex( 0.16981743560670751     ,  0.0000000000000000     ),
                    complex( 0.68100528507831026     ,  0.0000000000000000     ),
                    complex(-0.30550178438001113     ,  0.0000000000000000     ),
                    complex( 0.16981743560670751     ,  0.0000000000000000     ),
                    complex(-0.32536602932851599     ,  0.0000000000000000     ),
                    complex( 0.33237610537903167     ,  0.0000000000000000     ),
                    complex(-0.14910529404613407     ,  0.0000000000000000     ),
                    complex( 0.68100528507831026     ,  0.0000000000000000     ),
                    complex( 0.33237610537903167     ,  0.0000000000000000     ),
                    complex( 0.92465303140679633     ,  0.0000000000000000     ),
                    complex(-0.59794503971747703     ,  0.0000000000000000     ),
                    complex(-0.30550178438001113     ,  0.0000000000000000     ),
                    complex(-0.14910529404613407     ,  0.0000000000000000     ),
                    complex(-0.59794503971747703     ,  0.0000000000000000     ),
                    complex(-0.14000767530643479     ,  0.0000000000000000     )]
        
        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
        
        NHEL=-1
        solution = [ 
                    complex(  624.99999999999989     , -149.66469744815910     ),
                    complex(  83.193213332874592     ,  333.62309211609107     ),
                   complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(-3.62119349359586659E-002,-0.36385791873139811     ),
                    complex(-0.14521782752280418     , 9.07326566152192315E-002),
                    complex(-0.34383932052304755     ,  0.0000000000000000     ),
                    complex(-3.62119349359586659E-002,-0.36385791873139811     ),
                    complex(-2.67785531406523024E-002,-0.26907119516335065     ),
                    complex(-0.10738789070969987     ,-0.50596916746687115     ),
                    complex(-0.10304644292578606     , 0.24202971253800695     ),
                    complex(-0.14521782752280418     , 9.07326566152192315E-002),
                    complex(-0.10738789070969987     ,-0.50596916746687115     ),
                    complex(-0.43064907243145895     , 0.26907119516335065     ),
                    complex(-0.41323891148317965     ,-6.03532248932644386E-002),
                    complex(-0.34383932052304755     ,  0.0000000000000000     ),
                    complex(-0.10304644292578606     , 0.24202971253800695     ),
                    complex(-0.41323891148317965     ,-6.03532248932644386E-002),
                    complex( 0.45742762557211131     , -0.0000000000000000     )]
        
        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=-2        
        solution = [ 
                    complex(  624.99999999999989     , -149.66469744815910     ),
                    complex(  83.193213332874592     ,  333.62309211609107     ),
                   complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     , -0.0000000000000000     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     , -0.0000000000000000     ),
                    complex(-0.46606677614840392     , 9.36959948731358322E-002),
                    complex( 0.13607969464455591     , 0.17618862729680237     ),
                    complex( 4.42705319225060942E-002, 0.44483078948812155     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 0.13607969464455591     , 0.17618862729680237     ),
                    complex( 4.57095198364004252E-002,-9.36959948731358322E-002),
                    complex( 0.17753457473164128     ,-0.11092428444383282     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 4.42705319225060942E-002, 0.44483078948812155     ),
                    complex( 0.17753457473164128     ,-0.11092428444383282     ),
                    complex( 0.42035725631200371     , -0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        
        # first
        NHEL = 2
        NST = -1
        solution = [
                    complex( -624.99999999999989     ,  149.66469744815910     ),
                    complex( -83.193213332874592     , -333.62309211609107     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex(-0.46606677614840392     , 9.36959948731358322E-002),
                    complex( 0.13607969464455591     , 0.17618862729680237     ),
                    complex( 4.42705319225060942E-002, 0.44483078948812155     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 0.13607969464455591     , 0.17618862729680237     ),
                    complex( 4.57095198364004252E-002,-9.36959948731358322E-002),
                    complex( 0.17753457473164128     ,-0.11092428444383282     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 4.42705319225060942E-002, 0.44483078948812155     ),
                    complex( 0.17753457473164128     ,-0.11092428444383282     ),
                    complex( 0.42035725631200371     ,  0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
 
             # first
        NHEL = 1   
        solution = [
                    complex( -624.99999999999989     ,  149.66469744815910     ),
                    complex( -83.193213332874592     , -333.62309211609107     ),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 3.62119349359586659E-002, 0.36385791873139811     ),
                    complex( 0.14521782752280418     ,-9.07326566152192315E-002),
                    complex( 0.34383932052304755     ,  0.0000000000000000     ),
                    complex( 3.62119349359586659E-002, 0.36385791873139811     ),
                    complex( 2.67785531406523024E-002, 0.26907119516335065     ),
                    complex( 0.10738789070969987     , 0.50596916746687115     ),
                    complex( 0.10304644292578606     ,-0.24202971253800695     ),
                    complex( 0.14521782752280418     ,-9.07326566152192315E-002),
                    complex( 0.10738789070969987     , 0.50596916746687115     ),
                    complex( 0.43064907243145895     ,-0.26907119516335065     ),
                    complex( 0.41323891148317965     , 6.03532248932644386E-002),
                    complex( 0.34383932052304755     ,  0.0000000000000000     ),
                    complex( 0.10304644292578606     ,-0.24202971253800695     ),
                    complex( 0.41323891148317965     , 6.03532248932644386E-002),
                    complex(-0.45742762557211131     ,  0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL = 0  
        solution = [complex( -624.99999999999989     ,  149.66469744815910     ),
                    complex( -83.193213332874592     , -333.62309211609107     ),
                   complex( 0.45927932677184580     ,  0.0000000000000000     ),
                    complex( 0.16981743560670751     ,  0.0000000000000000     ),
                    complex( 0.68100528507831026     ,  0.0000000000000000     ),
                    complex(-0.30550178438001113     ,  0.0000000000000000     ),
                    complex( 0.16981743560670751     ,  0.0000000000000000     ),
                    complex(-0.32536602932851599     ,  0.0000000000000000     ),
                    complex( 0.33237610537903167     ,  0.0000000000000000     ),
                    complex(-0.14910529404613407     ,  0.0000000000000000     ),
                    complex( 0.68100528507831026     ,  0.0000000000000000     ),
                    complex( 0.33237610537903167     ,  0.0000000000000000     ),
                    complex( 0.92465303140679633     ,  0.0000000000000000     ),
                    complex(-0.59794503971747703     ,  0.0000000000000000     ),
                    complex(-0.30550178438001113     ,  0.0000000000000000     ),
                    complex(-0.14910529404613407     ,  0.0000000000000000     ),
                    complex(-0.59794503971747703     ,  0.0000000000000000     ),
                    complex(-0.14000767530643479     ,  0.0000000000000000     )]

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL = -1
        solution = [                    complex( -624.99999999999989     ,  149.66469744815910     ),
                    complex( -83.193213332874592     , -333.62309211609107     ),
                   complex(  0.0000000000000000     ,  0.0000000000000000     ),
                   complex(-3.62119349359586659E-002, 0.36385791873139811     ),
                    complex(-0.14521782752280418     ,-9.07326566152192315E-002),
                    complex(-0.34383932052304755     ,  0.0000000000000000     ),
                    complex(-3.62119349359586659E-002, 0.36385791873139811     ),
                    complex(-2.67785531406523024E-002, 0.26907119516335065     ),
                    complex(-0.10738789070969987     , 0.50596916746687115     ),
                    complex(-0.10304644292578606     ,-0.24202971253800695     ),
                    complex(-0.14521782752280418     ,-9.07326566152192315E-002),
                    complex(-0.10738789070969987     , 0.50596916746687115     ),
                    complex(-0.43064907243145895     ,-0.26907119516335065     ),
                    complex(-0.41323891148317965     , 6.03532248932644386E-002),
                    complex(-0.34383932052304755     ,  0.0000000000000000     ),
                    complex(-0.10304644292578606     ,-0.24202971253800695     ),
                    complex(-0.41323891148317965     , 6.03532248932644386E-002),
                    complex( 0.45742762557211131     , -0.0000000000000000     )]


        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
 
        NHEL = -2
        solution = [                    complex( -624.99999999999989     ,  149.66469744815910     ),
                    complex(-83.193213332874592     , -333.62309211609107),
                    complex(  0.0000000000000000     ,  0.0000000000000000     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex(  0.0000000000000000     , -0.0000000000000000     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex(-0.46606677614840392     ,-9.36959948731358322E-002),
                    complex( 0.13607969464455591     ,-0.17618862729680237     ),
                    complex( 4.42705319225060942E-002,-0.44483078948812155     ),
                    complex(  0.0000000000000000     , -0.0000000000000000     ),
                    complex( 0.13607969464455591     ,-0.17618862729680237     ),
                    complex( 4.57095198364004252E-002, 9.36959948731358322E-002),
                    complex( 0.17753457473164128     , 0.11092428444383282     ),
                    complex( -0.0000000000000000     ,  0.0000000000000000     ),
                    complex( 4.42705319225060942E-002,-0.44483078948812155     ),
                    complex( 0.17753457473164128     , 0.11092428444383282     ),
                    complex( 0.42035725631200371     , -0.0000000000000000     )]     

        values = wavefunctions.txxxxx(P, M, NHEL, NST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
            
    def test_I(self):
        """check I wavefunctions against fortran output"""            
            
        P = [2499.9999999999991, 537.98548101331721, 2157.4401624693646, -967.83657009607577] 
        M = 607.71370000000002 
            
        NHEL=-1
        NRST=-1
        solution = [
            complex( +2499.9999999999991     , -967.83657009607577     ),
            complex( +537.98548101331721     , 2157.4401624693646     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex(  1.7524192771692548     ,  7.0275869209877353     ),
            complex( -38.466940073898407     , -0.0000000000000000     ),
            complex( -14.201895200573350     , -56.952724878721050     )
            ]


        values = wavefunctions.ixxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
            
            
        NHEL=1
        NRST=-1
        solution = [
            complex(  2499.9999999999991     , -967.83657009607577     ),
            complex(  537.98548101331721     , 2157.4401624693646     ),
            complex(  14.201895200573350     , -56.952724878721050     ),
            complex( -38.466940073898407     , -0.0000000000000000     ),
            complex( -1.7524192771692548     ,  7.0275869209877353     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     )]

        values = wavefunctions.ixxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=1
        NRST=1
        solution = [            
            complex(  -2499.9999999999991     , 967.83657009607577     ),
            complex(  -537.98548101331721     , -2157.4401624693646     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex(  1.7524192771692548     ,  7.0275869209877353     ),
            complex(  38.466940073898407     ,  0.0000000000000000     ),
            complex(  14.201895200573350     ,  56.952724878721050     )]

        values = wavefunctions.ixxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=-1
        NRST=1
        solution = [            
            complex(  -2499.9999999999991     , 967.83657009607577     ),
            complex(  -537.98548101331721     , -2157.4401624693646     ),
            complex( -14.201895200573350     ,  56.952724878721050     ),
            complex(  38.466940073898407     ,  0.0000000000000000     ),
            complex( -1.7524192771692548     ,  7.0275869209877353     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     )]

            
        

        values = wavefunctions.ixxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
    
    def test_O(self):
        """check O wavefunctions against fortran output""" 
    
            
        P = [2499.9999999999991, 537.98548101331721, 2157.4401624693646, -967.83657009607577] 
        M = 607.71370000000002     
    
        NHEL=1
        NRST=1
        solution=[
                        complex(  2499.9999999999991     , -967.83657009607577     ),
            complex(  537.98548101331721     ,  2157.4401624693646     ),
            complex(  38.466940073898407     ,  0.0000000000000000     ),
            complex(  14.201895200573350     , -56.952724878721050     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex(  1.7524192771692548     , -7.0275869209877353     )]
 
        values = wavefunctions.oxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
 
 
        NHEL=1
        NRST=-1
        solution=[
                        complex( -2499.9999999999991     ,  967.83657009607577     ),
            complex( -537.98548101331721     , -2157.4401624693646     ),
            complex( -1.7524192771692548     , -7.0275869209877353     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex(  14.201895200573350     ,  56.952724878721050     ),
            complex( -38.466940073898407     , -0.0000000000000000     )]

        values = wavefunctions.oxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=-1
        NRST=1
        solution=[
            complex(  2499.9999999999991     , -967.83657009607577     ),
            complex(  537.98548101331721     ,  2157.4401624693646     ),
            complex( -1.7524192771692548     , -7.0275869209877353     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex( -14.201895200573350     , -56.952724878721050     ),
            complex(  38.466940073898407     ,  0.0000000000000000     ),
]

        values = wavefunctions.oxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])


        NHEL=-1
        NRST=-1
        solution=[
            complex( -2499.9999999999991     ,  967.83657009607577     ),
            complex( -537.98548101331721     , -2157.4401624693646     ),
            complex( -38.466940073898407     , -0.0000000000000000     ),
            complex( -14.201895200573350     ,  56.952724878721050     ),
            complex(  4.7465641991565066     ,  0.0000000000000000     ),
            complex(  1.7524192771692548     , -7.0275869209877353     ),
]

        values = wavefunctions.oxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])
            
    def test_V(self):
        """check V wavefunctions against fortran output"""             
    
        P = [2500, 0, 0, 2500]
        M = 0
        
        NHEL=1
        NRST=1
        solution=[
            complex(  2500.0000000000000     ,  2500.0000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex(-0.70710678118654757     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     , 0.70710678118654757     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
]

        values = wavefunctions.vxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=1
        NRST=-1
        solution=[
            complex( -2500.0000000000000     , -2500.0000000000000     ),
            complex( -0.0000000000000000     , -0.0000000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex(-0.70710678118654757     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     ,-0.70710678118654757     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
]

        values = wavefunctions.vxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=-1
        NRST=-1
        solution=[
            complex( -2500.0000000000000     , -2500.0000000000000     ),
            complex( -0.0000000000000000     , -0.0000000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex( 0.70710678118654757     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     ,-0.70710678118654757     ),
            complex( -0.0000000000000000     ,  0.0000000000000000     )]

        values = wavefunctions.vxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])

        NHEL=-1
        NRST=1
        solution=[
            complex(  2500.0000000000000     ,  2500.0000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     ,  0.0000000000000000     ),
            complex( 0.70710678118654757     ,  0.0000000000000000     ),
            complex(  0.0000000000000000     , 0.70710678118654757     ),
            complex( -0.0000000000000000     ,  0.0000000000000000     ),
]
        
        values = wavefunctions.vxxxxx(P, M, NHEL, NRST)
        self.assertEqual(len(values), len(solution))
        for i in range(len(values)):
            self.assertAlmostEqual(values[i], solution[i])    
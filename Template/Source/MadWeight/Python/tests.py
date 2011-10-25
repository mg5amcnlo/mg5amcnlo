#!/usr/bin/env python 
####################################################################################################
####################################################################################################
##                                                                                                ##
##                                    MOD FILE MODULE                                             ##
##                                                                                                ##
####################################################################################################
####################################################################################################
##                                                                                                ##
##    Author: Mattelaer Olivier                                                                   ##
##    Institution: UCL-CP3                                                                        ##
##    contact: omattelaer@uclouvain.be                                                            ##
##                                                                                                ##
##    last modification: 01/06/10                                                                 ##
##    tag release:       1.4                                                                      ##
##                                                                                                ##
####################################################################################################

  
#########################################################################################################
#  TEST #################################################################################################
#########################################################################################################
import sys
sys.path.append('./Source/MadWeight/Python')
import filecmp
import unittest
import os, shutil

import create_run
import MW_param
import mod_file    
import write_MadWeight as MW_write

#########################################################################################################
#  TEST MOD_FILE   ######################################################################################
#########################################################################################################

class TestMod_file(unittest.TestCase):
    """ Test the the mod routines works correctly on MadWeight """

    def setUp(self):
        """ create a copy of the original file """
        shutil.copyfile('../Template/SubProcesses/cuts.f', './SubProcesses/cuts.bk')

    def tearDown(self):
        os.system('rm -f ./SubProcesses/cuts.mod')
        os.system('rm -f ./SubProcesses/cuts.bk')
        os.system('rm -f ./SubProcesses/cuts.o')


    def test_cuts(self):
        """ test if we can activate/desactivate the cuts """

        self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)

        file_to_mod='./SubProcesses/cuts.bk'
        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        #modify file                                            
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        file_to_mod='./SubProcesses/cuts.mod'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

    def test_mw_cuts(self):

        file_to_mod ='./SubProcesses/cuts.bk'
        rule= './Source/MadWeight/mod_file/mod_cuts'
        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)            
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())
        
        file_to_mod='./SubProcesses/cuts.mod'
        rule = './Source/MadWeight/mod_file/suppress_cuts_MW'
        
        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

    def test_P_BW_cuts(self):

        self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)
        
        file_to_mod='./SubProcesses/cuts.bk'
        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        #modify file                                                                                                                                                                   
        
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        file_to_mod='./SubProcesses/cuts.mod'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())
 
        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        # Next one will Fail but is not supose to be called whitout check of the second 
        #rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        #mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        #self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        #self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)


    def test_MW_BW_cuts(self):

        self.assertEqual(create_run.cut_is_active('cuts.bk'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.bk'),1)

        file_to_mod='./SubProcesses/cuts.bk'
        rule= './Source/MadWeight/mod_file/mod_cuts'
        mod_file.mod_file(file_to_mod,rule, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'),1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        file_to_mod='./SubProcesses/cuts.mod'
        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        #modify file                                                                                                                                                                   
        
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())
            
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 1)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_cuts_MG'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 1)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())

        rule='./Source/MadWeight/mod_file/suppress_BW_cuts'
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""}, write='./SubProcesses/cuts.mod')
        self.assertEqual(create_run.cut_is_active('cuts.mod'), 0)
        self.assertEqual(create_run.bw_cut_is_active('cuts.mod'), 0)
        self.assertFalse('\n$B$' in open('./SubProcesses/cuts.mod').read())


#########################################################################################################
#  TEST DATA.INC   ######################################################################################
#########################################################################################################

class TestMWBuilder(unittest.TestCase):
    """ Test that the main code/data.inc are written correctly"""

    def oneSetUp(self):
        """prepare the runs """
        os.system('./bin/PassToMadWeight')
        os.system('rm ./Cards/proc_card_mg5.dat')
        os.system('ln -s ../Source/MadWeight/test_files/proc_card_mg5.dat ./Cards')
        os.system('./bin/newprocess')
        os.system('./bin/madweight.py -12')
        self.MWparam = MW_param.MW_info('MadWeight_card.dat')
        #MW_write.create_all_fortran_code(self.MWparam)

    def checkSetUp(self):
        return filecmp.cmp('./Source/MadWeight/test_files/proc_card_mg5.dat','./Cards/proc_card_mg5.dat', False)

    def test_autofiles(self):
        if not self.checkSetUp():
            self.oneSetUp()
        else:
            self.MWparam = MW_param.MW_info('MadWeight_card.dat')

        for MWdir in self.MWparam.MW_listdir:
            for filename in ['main_code%s.f', 'data%s.inc','d_choices%s.f','d_choices%s.inc','info_part%s.dat','multi_channel%s.f','transfer_functions%s.f']:
                text = 'diff ./SubProcesses/%s/' + filename + ' ./Source/MadWeight/test_files/'+ filename
                os.system(text % (MWdir, '', MWdir[4]))
                self.assertTrue(filecmp.cmp(os.path.join('./SubProcesses/',MWdir, filename % ''), './Source/MadWeight/test_files/' + filename % MWdir[4], False),
                                'not valid file %s in process %s' % (filename % '', MWdir))
    




print 'WARNING: This test modify this version of the Template. The correct way to launch it is to first made a copy of the Template'
y =raw_input('Do you want to continue? [y/n]')
if y == 'y':
    unittest.main()
else:
    print 'stopped'


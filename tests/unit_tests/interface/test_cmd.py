##############################################################################
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
""" Basic test of the command interface """

import unittest
import madgraph
import madgraph.interface.master_interface as cmd
import madgraph.interface.extended_cmd as ext_cmd
import os


class TestValidCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
    cmd = cmd.MasterCmd()
    
    def wrong(self,*opt):
        self.assertRaises(madgraph.MadGraph5Error, *opt)
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
    
    def test_shell_and_continuation_line(self):
        """ check that the cmd line interpret shell and ; correctly """
        
        #Those tests are important for this type of launch: 
        # cd DIR; ./bin/generate_events 
        try:
            os.remove('/tmp/tmp_file')
        except:
            pass
        
        self.do('! cd /tmp; touch tmp_file')
        self.assertTrue(os.path.exists('/tmp/tmp_file'))
        
        try:
            os.remove('/tmp/tmp_file')
        except:
            pass
        self.do(' ! cd /tmp; touch tmp_file')
        self.assertTrue(os.path.exists('/tmp/tmp_file'))
    
    def test_cleaning_history(self):
        """check that the cleaning of the history command works as expected"""
        
        # Test the call present inside do_generate        
        history="""set cluster_queue 2
        import model mssm
        generate p p > go go 
        add process p p > go go j
        set gauge Feynman
        check p p > go go
        output standalone
        display particles
        generate p p > go go"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history = history
        self.cmd.clean_history(remove_bef_last='generate', keep_switch=True,
                     allow_for_removal= ['generate', 'add process', 'output'])

        goal = """set cluster_queue 2
        import model mssm
        set gauge Feynman
        generate p p > go go"""
        goal = [l.strip() for l in  goal.split('\n')]

        self.assertEqual(self.cmd.history, goal)
        
        # Test the call present in do_import model
        history="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        add process p p > go go j
        set gauge Feynman
        check p p > go go
        output standalone
        display particles
        generate p p > go go
        import heft"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history = history        
        
        self.cmd.clean_history(remove_bef_last='import', keep_switch=True,
                        allow_for_removal=['generate', 'add process', 'output'])

        # Test the call present in do_import model
        goal="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        import model mssm --modelname
        set gauge Feynman
        import heft""" 

        goal = [l.strip() for l in  goal.split('\n')]

        self.assertEqual(self.cmd.history, goal)
        
        
        # Test the call present in do_output
        history="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        output standalone
        launch
        output"""
        history = [l.strip() for l in  history.split('\n')]
        self.cmd.history = history         
        
        self.cmd.clean_history(allow_for_removal = ['output'], keep_switch=True,
                           remove_bef_last='output')

        goal="""set cluster_queue 2
        import model mssm
        define SW = May The Force Be With You
        generate p p > go go 
        import model mssm --modelname
        output"""
        
        goal = [l.strip() for l in  goal.split('\n')]
        self.assertEqual(self.cmd.history, goal)
    
    
    
    
    def test_help_category(self):
        """Check that no help category are introduced by mistake.
           If this test fails, this is due to a un-expected ':' in a command of
           the cmd interface.
        """
        
        category = set()
        categories_nb = {}
        for interface_class in cmd.MasterCmd.__mro__:
            valid_command = [c for c in dir(interface_class) if c.startswith('do_')]
            name = interface_class.__name__
            if name in ['CmdExtended', 'CmdShell', 'Cmd']:
                continue
            for command in valid_command:
                obj = getattr(interface_class, command)
                if obj.__doc__ and ':' in obj.__doc__:
                    cat = obj.__doc__.split(':',1)[0]
                    category.add(cat)
                    if cat in categories_nb:
                        categories_nb[cat] += 1
                    else:
                        categories_nb[cat] = 1
                
        target = set(['Not in help'])
        self.assertEqual(target, category)
        self.assertEqual(categories_nb['Not in help'], 2)
    
    
    
    
    def test_check_generate(self):
        """check if generate format are correctly supported"""
    
        cmd = self.cmd
        
        # valid syntax
        cmd.check_process_format('e+ e- > e+ e-')
        cmd.check_process_format('e+ e- > mu+ mu- QED=0')
        cmd.check_process_format('e+ e- > mu+ ta- / x $y @1')
        cmd.check_process_format('e+ e- > mu+ ta- $ x /y @1')
        cmd.check_process_format('e+ e- > mu+ ta- $ x /y, (e+ > e-, e-> ta) @1')
        
        # unvalid syntax
        self.wrong(cmd.check_process_format, ' e+ e-')
        self.wrong(cmd.check_process_format, ' e+ e- > e+ e-,')
        self.wrong(cmd.check_process_format, ' e+ e- > > e+ e-')
        self.wrong(cmd.check_process_format, ' e+ e- > j / g > e+ e-')        
        self.wrong(cmd.check_process_format, ' e+ e- > j $ g > e+  e-')         
        self.wrong(cmd.check_process_format, ' e+ > j / g > e+ > e-')        
        self.wrong(cmd.check_process_format, ' e+ > j $ g > e+ > e-')
        self.wrong(cmd.check_process_format, ' e+ > e+, (e+ > e- / z, e- > top')   
        self.wrong(cmd.check_process_format, 'e+ > ')
        self.wrong(cmd.check_process_format, 'e+ >')
        
    def test_output_default(self):
        """check that if a export_dir is define before an output
           a new one is propose"""
           
        cmd = self.cmd
        cmd._export_dir = 'tmp'
        cmd._curr_amps = 'dummy'
        cmd._curr_model = {'name':'WHY'}
        cmd.check_output([])
        
        self.assertNotEqual('tmp', cmd._export_dir)


class TestExtendedCmd(unittest.TestCase):
    """test the extension of cmd interface"""
    
    
    def test_the_exit_from_child_cmd(self):
        """ """
        main = ext_cmd.Cmd()
        child = ext_cmd.Cmd()
        main.define_child_cmd_interface(child, interface=False)
        self.assertEqual(main.child, child)
        self.assertEqual(child.mother, main)        
        
        ret = main.do_quit('')
        self.assertEqual(ret, None)
        self.assertEqual(main.child, None)
        ret = main.do_quit('')
        self.assertEqual(ret, True)
        
    def test_the_exit_from_child_cmd2(self):
        """ """
        main = ext_cmd.Cmd()
        child = ext_cmd.Cmd()
        main.define_child_cmd_interface(child, interface=False)
        self.assertEqual(main.child, child)
        self.assertEqual(child.mother, main)        
        
        ret = child.do_quit('')
        self.assertEqual(ret, True)
        self.assertEqual(main.child, None)
        #ret = main.do_quit('')
        #self.assertEqual(ret, True)        
         
    


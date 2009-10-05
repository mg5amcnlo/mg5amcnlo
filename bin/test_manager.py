#!/usr/bin/env python 
##############################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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
##############################################################################
""" Manager for running the test library 

   This library offer a simple way to launch test.
   
   To run a test/class of test/test file/module of test/...
   you just have to launch 
   test_manager.run(NAME)
   or 
   test_manager.run(LIST_OF_NAME)

   the NAME can contain regular expression (in python re standard format)
""" 

import unittest
import re
import os
import inspect
import sys
sys.path+=['.','..']

##############################################################################
def run(expression='', re_opt=0, package='./tests/'):
    """ running the test associated to expression. By default, this launch all 
    test derivated from TestCase. expression can be the name of directory, 
    module, class, function or event standard regular expression (in re format)
    """
    
    #init a test suite
    testsuite = unittest.TestSuite()
    collect = unittest.TestLoader()
    for test_fct in Test_in_module(package=package, expression=expression, \
                                   re_opt=re_opt):
        data=collect.loadTestsFromName(test_fct)
        testsuite.addTest(data)
    unittest.TextTestRunner(verbosity=1).run(testsuite)
    

##############################################################################
class Test_in_module(list):
    """ class introspecting the test module to find the available test.

    The strategy is the following:
    The routine collect_dir looks in all module/file to find the different 
    functions in different test class. This produce a list, on wich external 
    routines can loop on. 
        
    In order to authorize definition and loop on this object on the same time,
    i.e: for test in Test_in_module([opt])-. At each time a loop is started, 
    we check if a collect_dir runned before, and run it if neccessary. And 
    that without any problem with future add-on on this class.
    """

    search_class=unittest.TestCase
    class ERRORTestManager(Exception): pass

    ##########################################################################
    def __init__(self, package='./tests/', expression='', re_opt=0):
        """ initialize global variable for the test """

        list.__init__(self)

        self.package = package
        if self.package[-1] != '/': self.package+='/'
        self.restrict_to(expression, re_opt)
        
    ##########################################################################
    def _check_if_obj_build(self):
        """ Check if a collect is already done 
            Uses to have smart __iter__ and __contain__ functions
        """
        if len(self) == 0:
            self.collect_dir(self.package, checking=True)

    ##########################################################################
    def __iter__(self):
        """ Check that a collect was performed (do it if needed) """
        self._check_if_obj_build()
        return list.__iter__(self)

    ##########################################################################
    def __contains__(self,value):
        """ Check that a collect was performed (do it if needed) """
        self._check_if_obj_build()
        return list.__contains__(self,value)

    ##########################################################################
    def collect_dir(self, directory, checking=True):
        """ Find the file and the subpackage in this package """

        for name in os.listdir(directory):
            local_check = checking

            status = self.status_file(directory + '/' + name)
            if status is None:
                continue

            if checking:
                if self.check_valid(directory + '/' + name):
                    local_check = False    #since now perform all the test

            if status == 'file':
                self.collect_file(directory + '/' + name, local_check)
            elif status == "module":
                self.collect_dir(directory + '/' + name, local_check)
        
    ##########################################################################
    def collect_file(self, file, checking=True):
        """ Find the different class instance derivated of TestCase """

        pyname=self.passin_pyformat(file)
        exec('import '+pyname+' as obj')         

        #look at class
        for name in dir(obj):
            exec('class_=obj.'+name)
            if inspect.isclass(class_) and issubclass(class_,unittest.TestCase):
                if checking:
                    if self.check_valid(name):
                        check_inside=False
                    else:
                        check_inside=True
                else:
                    check_inside=False


                self.collect_function(class_, checking=check_inside, \
                                          base=pyname)

    ##########################################################################
    def collect_function(self, class_, checking=True, base=''):
        """
        Find the different test function in this class
        test functions should start with test
        """

        if not inspect.isclass(class_):
            raise self.ERRORTestManager, 'wrong input class_'
        if not issubclass(class_,unittest.TestCase):
            raise self.ERRORTestManager, 'wrong input class_'

        #devellop the name
        if base:
            base+='.'+class_.__name__
        else:
            base=class_.__name__

        candidate=[base+'.'+name for name in dir(class_) if \
                       name.startswith('test')\
                       and inspect.ismethod(eval('class_.'+name))]

        if not checking:
            self+=candidate
        else:
            self+=[name for name in candidate if self.check_valid(name)]

    ##########################################################################
    def restrict_to(self, expression, re_opt=0):
        """ store in global the expression to fill in order to be a valid test """
        
        if isinstance(expression,list):
            pass
        elif isinstance(expression,basestring):
            if expression=='':
                expression=['.*'] #made an re authorizing all regular name
            else:
                expression = [expression]
        else:
            raise self.ERRORTestManager, 'obj should be list or string'
        
        self.rule=[]
        for expr in expression:
            if not expr.startswith('^'): expr='^'+expr #fix the begin of the re
            if not expr.endswith('$'): expr=expr+'$' #fix the end of the re
            self.rule.append( re.compile(expr, re_opt) )

    ##########################################################################
    def check_valid(self, name):
        """ check if the name correspond to the rule """
        
        if not isinstance(name,basestring):
            raise self.ERRORTestManager, 'check valid take a string argument'
        
        for specific_format in self.format_possibility(name):
            for expr in self.rule:
                if expr.search(specific_format):
                    return True
        return False

    ##########################################################################
    @staticmethod
    def status_file(name):
        """ check if a name is a module/a python file and return the status """
        if os.path.isfile(name):
            if name.endswith('.py') and '__init__' not in name:
                return 'file'
        elif os.path.isdir(name):
            if os.path.isfile(name+'/__init__.py'):
                return 'module'

    ##########################################################################
    @classmethod
    def passin_pyformat(cls,name):
        """ transform a relative position in a python import format """

        if not isinstance(name,basestring):
            raise cls.ERRORTestManager, 'collect_file takes a file position'

        name=name.replace('//','/') #sanity
        #deal with begin/end
        if name.startswith('./'):
            name=name[2:]
        if not name.endswith('.py'):
            raise cls.ERRORTestManager,'Python files should have .py extension'
        else:
            name=name[:-3]

        if name.startswith('/'):
            raise cls.ERRORTestManager, 'path must be relative'
        if '..' in name:
            raise cls.ERRORTestManager,'relative position with \'..\' is' + \
                ' not supported for the moment'
        
        #replace '/' by points -> Python format
        name=name.replace('/','.')        

        #position
        return name

    ##########################################################################
    def format_possibility(self,name):
        """ return the different string derivates from name in order to 
        scan all the different format authorizes for a restrict_to 
        format authorizes:
        1) full file position
        2) name of the file (with extension)
        3) full file position whithour extension
        4) name of the file (whithout extension)
        5) name of the file (but suppose name in python format)
        6) if name is a python file, try with a './' and with package pos
        """

        def add_to_possibility(possibility,val):
            """ add if not exist """
            if val not in possibility: 
                possibility.append(val)
        #end local def
                
        #print name
        #sanity
        if name.startswith('./'): name = name[2:]
        name=name.replace('//','/')
        # init with solution #
        out=[name]

        # add solution 2
        new_pos=name.split('/')[-1]
        add_to_possibility(out,new_pos)
        
        #remove extension and add solution3 and 6
        if name.endswith('.py'):
            add_to_possibility(out,'./'+name)
            add_to_possibility(out,self.package+name)
            name = name[:-3]
        add_to_possibility(out,name)

        #add solution 4
        new_pos=name.split('/')[-1]
        add_to_possibility(out,new_pos)

        #add solution 5
        new_pos=name.split('.')[-1]
        add_to_possibility(out,new_pos)




        return out

##############################################################################
if __name__ == "__main__":

    opt=sys.argv
    if len(opt)==1:
        run()
    elif len(opt)==2:
        run(opt[1])
    else:
        run(opt[1],re_opt=opt[2])

#some example
#    run('iolibs')
#    run('test_test_manager.py')
#    run('./tests/unit_tests/bin/test_test_manager.py')
#    run('IOLibsMiscTest')
#    run('Unittest_on_TestinModule')
#    run('test_check_valid_on_file')
#    run('test_collect_dir.*') # '.*' stands for all possible char (re format)



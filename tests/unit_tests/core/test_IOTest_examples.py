import tests.IOTests as IOTests
import os 
import shutil

# This test modules is to illustrate the various way of easily designing new
# Customized IOTests

# Notice that when you run an IOTest directly from
#  ./test/test_manager.py -p [A|P|U]
# It will crash of pass, without much details provided or possibility of
# automatically updating it. In order to have a better monitoring and updating
# functionalities for these tests, you can run them from
#  ./test/test_manager.py -i [U|R] (|-f|-F) 'testGroup/testName/ALL'
# For more information on the syntax of the above, you can type
#  ./test/test_manager.py help

# Let's start with the simplest implementation.
# Notice that if you plan on having some IOTests in a given class, you must make
# it inherit from IOTests.IOTestManager (which itself inherits from 
# unittest.TestCase). You can still have regular tests in these classes too.

class IOTest_SimpleExamples(IOTests.IOTestManager):
    """We should present here the simplest IOTest implementations"""

    # A couple of things are important in the syntax above:
    #  a) The name of the function must start with testIO_
    #  b) The body of the function should contain the instruction to generate
    #     the files to be compared. It must return a path which serves as the
    #     'base_path' from which starts the relative path of the files to be
    #      compared.
    #  c) The function doc is semantic! After each "target:" marker, you can
    #     specify one file to compare. You can use regular expression for them,
    #     by enclosing them in squared brackets. for example:
    #        target: ../../Source/DHELAS/[.+\.(inc|f)]
    #     Notice that the absolute path from which the path above is the one
    #     returned by the function (see point b).
    #  d) Don't worry about the header of most MG file with the date and version
    #     because the IOTestManager makes sure to parse that and transform it 
    #     automatically in a transparent manner.

    @IOTests.createIOTest()
    def testIO_MySimpleIOTestWrapped(self):
        """ target: aFile.txt
        """
        open('/tmp/aFile.txt','w').write("This is the content to check")
        return os.path.join('/tmp')

    # In this more complicated example you see that more than one file can be
    # specified as a target, while also using regular expression if one wants.
    # Also in the doc you can specify a 'clean:' marker (or several) followed
    # by an absolute path. These paths will be erased after the IOTest is run.
    # Finally, each IOTest has a name and are organized in groups.
    # By default the testName is inferred from the function, i.e. it would be
    # MySimpleIOTestWrapped in this case. The groupName is set by default to
    # the class name defining this test function.
    # As shown above, you can change these defaults with the decorator args.

    @IOTests.createIOTest(groupName='MyTestGroup',testName='MyTestName')
    def testIO_MyCustomNameIOTestWrapped(self):
        """ target: FileC.txt
            target: FolderA/[.+\.txt]
            clean: /tmp/testScratch
        """
        if os.path.isdir('/tmp/testScratch'):
            shutil.rmtree('/tmp/testScratch')
        os.mkdir('/tmp/testScratch')
        os.mkdir('/tmp/testScratch/FolderA')
        open('/tmp/testScratch/FolderA/FileA.txt','w').write("FileA")
        open('/tmp/testScratch/FolderA/FileB.txt','w').write("FileB")
        open('/tmp/testScratch/FileC.txt','w').write("FileC")   
        return os.path.join('/tmp/testScratch')

class IOTestExampleWithSetUp(IOTests.IOTestManager):
    """Here are some slightly more involved examples"""
    
    # This setup function will be called prior the IOTest and regular tests,
    # both when run directly from ./test/test_manager.py -p [A|P|U]  and
    # from ./test/test_manager.py -i [U|R] (|-f|-F)
    
    def setUp(self):
        self.file_content = 'ContentDefinedFromSetup'

    # This test is the same as the simple one except that it uses info
    # defined in setUP

    @IOTests.createIOTest()
    def testIO_IOTestWrappedWithSetUP(self):
        """ target: aFileWithSetup.txt
        """
        open('/tmp/aFileWithSetup.txt','w').write(
                                       self.file_content+" From a wrapped test")
        return os.path.join('/tmp')

    # This test is similar to the simple one above except that it is implemented
    # directly without the decorator. In principle you would not need it, but
    # it might be useful in some cases to have more freedom in defining the 
    # clean up function for example. And it shows a bit more of how things are
    # implemented under the hood.

    def testIO_MyHandwrittenIOTest(self, load_only=False):
        " Example of a handwritten test"
        def MyRun(self):
            open('/tmp/FileX.txt','w').write("FileX "+self.file_content)
            return os.path.join('/tmp')

        def MyClean():
            os.remove('/tmp/FileX.txt')
        MyCustomTest = IOTests.CustomIOTest(['FileX.txt'],MyRun,MyClean)
        self.addIOTest('IOTestExampleWithSetUp','MyHandwrittenIOTest',
                                                                   MyCustomTest)
        if not load_only:
            self.runIOTests(update=False, force=10, verbose=False, 
                    testKeys=[('IOTestExampleWithSetUp','MyHandwrittenIOTest')])

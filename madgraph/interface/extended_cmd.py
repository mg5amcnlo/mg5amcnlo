################################################################################
#
# Copyright (c) 2011 The MadGraph Development team and Contributors
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
"""  A file containing different extension of the cmd basic python library"""


import cmd
import logging
import os
import signal
import traceback
logger = logging.getLogger('cmdprint') # for stdout
logger_stderr = logging.getLogger('fatalerror') # for stderr

#===============================================================================
# CmdExtended
#===============================================================================
class Cmd(cmd.Cmd):
    """Extension of the cmd.Cmd command line.
    This extensions supports line breaking, history, comments,
    internal call to cmdline, path completion,...
    this class should be MG5 independent"""

    #suggested list of command
    next_possibility = {} # command : [list of suggested command]
    
    class InvalidCmd(Exception):
        """expected error for wrong command"""
        pass    
 
        debug_output = 'debug'
        error_debug = """Please report this bug to developers\n
           More information is found in '%s'.\n
           Please attach this file to your report."""
           
        keyboard_stop_msg = """stopping all current operation
            in order to quit the program please enter exit"""
 
    
    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
        self.log = True
        self.history = []
        self.save_line = ''
        cmd.Cmd.__init__(self, *arg, **opt)
        self.__initpos = os.path.abspath(os.getcwd())
        

        
        
    def precmd(self, line):
        """ A suite of additional function needed for in the cmd
        this implement history, line breaking, comment treatment,...
        """
        
        if not line:
            return line
        line = line.lstrip()

        # Update the history of this suite of command,
        # except for useless commands (empty history and help calls)
        if line != "history" and \
            not line.startswith('help') and \
            not line.startswith('#*'):
            self.history.append(line)

        # Check if we are continuing a line:
        if self.save_line:
            line = self.save_line + line 
            self.save_line = ''
        
        # Check if the line is complete
        if line.endswith('\\'):
            self.save_line = line[:-1]
            return '' # do nothing   
        
        # Remove comment
        if '#' in line:
            line = line.split('#')[0]

        # Deal with line splitting
        if ';' in line and not (line.startswith('!') or line.startswith('shell')):
            for subline in line.split(';'):
                stop = self.onecmd(subline)
                stop = self.postcmd(stop, subline)
            return ''
        
        # execute the line command
        return line

    def nice_error_handling(self, error, line):
        """ """ 
        # Make sure that we are at the initial position
        os.chdir(self.__initpos)
        # Create the debug files
        self.log = False
        cmd.Cmd.onecmd(self, 'history %s' % self.debug_output)
        debug_file = open(self.debug_output, 'a')
        traceback.print_exc(file=debug_file)
        # Create a nice error output
        if self.history and line == self.history[-1]:
            error_text = 'Command \"%s\" interrupted with error:\n' % line
        elif self.history:
            error_text = 'Command \"%s\" interrupted in sub-command:\n' %line
            error_text += '\"%s\" with error:\n' % self.history[-1]
        else:
            error_text = ''
        error_text += '%s : %s\n' % (error.__class__.__name__, 
                                                str(error).replace('\n','\n\t'))
        error_text += self.error_debug % self.debug_output
        logger_stderr.critical(error_text)
        #stop the execution if on a non interactive mode
        if self.use_rawinput == False:
            return True 
        return False

    def nice_user_error(self, error, line):
        # Make sure that we are at the initial position
        os.chdir(self.__initpos)
        if line == self.history[-1]:
            error_text = 'Command \"%s\" interrupted with error:\n' % line
        else:
            error_text = 'Command \"%s\" interrupted in sub-command:\n' %line
            error_text += '\"%s\" with error:\n' % self.history[-1] 
        error_text += '%s : %s' % (error.__class__.__name__, 
                                                str(error).replace('\n','\n\t'))
        logger_stderr.error(error_text)
        #stop the execution if on a non interactive mode
        if self.use_rawinput == False:
            return True
        # Remove failed command from history
        self.history.pop()
        return False

    def onecmd(self, line):
        """catch all error and stop properly command accordingly"""
        
        try:
            return cmd.Cmd.onecmd(self, line)
        except self.InvalidCmd as error:
            if __debug__:
                self.nice_error_handling(error, line)
            else:
                self.nice_user_error(error, line)
        except Exception as error:
            self.nice_error_handling(error, line)
        except KeyboardInterrupt:
            print self.keyboard_stop_msg
            
    def exec_cmd(self, line, errorhandling=False):
        """for third party call, call the line with pre and postfix treatment
        without global error handling """

        logger.info(line)
        line = self.precmd(line)
        if errorhandling:
            stop = self.onecmd(line)
        else:
            stop = cmd.Cmd.onecmd(self, line)
        stop = self.postcmd(stop, line)
        return stop      

    def run_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment
        with global error handling"""
        
        return self.exec_cmd(line, errorhandling=True)
    
    def emptyline(self):
        """If empty line, do nothing. Default is repeat previous command."""
        pass
    
    def default(self, line):
        """Default action if line is not recognized"""

        # Faulty command
        logger.warning("Command \"%s\" not recognized, please try again" % \
                                                                line.split()[0])
    # Quit
    def do_quit(self, line):
        """ exit the mainloop() """
        print
        return True
 
    # Aliases
    do_EOF = do_quit
    do_exit = do_quit

    def do_help(self, line):
        """ propose some usefull possible action """
        
        cmd.Cmd.do_help(self,line)
        
        # if not basic help -> simple call is enough    
        if line:
            return

        if len(self.history) == 0:
            last_action_2 = last_action = 'start'
        else:
            last_action_2 = last_action = 'none'
        
        pos = 0
        authorize = self.next_possibility.keys() 
        while last_action_2  not in authorize and last_action not in authorize:
            pos += 1
            if pos > len(self.history):
                last_action_2 = last_action = 'start'
                break
            
            args = self.history[-1 * pos].split()
            last_action = args[0]
            if len(args)>1: 
                last_action_2 = '%s %s' % (last_action, args[1])
            else: 
                last_action_2 = 'none'
        
        print 'Contextual Help'
        print '==============='
        if last_action_2 in authorize:
            options = self.next_possibility[last_action_2]
        elif last_action in authorize:
            options = self.next_possibility[last_action]
        
        text = 'The following command(s) may be useful in order to continue.\n'
        for option in options:
            text+='\t %s \n' % option      
        print text

    @staticmethod
    def list_completion(text, list):
        """Propose completions of text in list"""
        if not text:
            completions = list
        else:
            completions = [ f
                            for f in list
                            if f.startswith(text)
                            ]
        return completions

    @staticmethod
    def path_completion(text, base_dir = None, only_dirs = False, 
                                                                 relative=True):
        """Propose completions of text to compose a valid path"""
        
        if base_dir is None:
            base_dir = os.getcwd()
            
        prefix, text = os.path.split(text)
        base_dir = os.path.join(base_dir, prefix)
        if prefix:
            prefix += os.path.sep
        
        

        if only_dirs:
            completion = [prefix + f
                          for f in os.listdir(base_dir)
                          if f.startswith(text) and \
                          os.path.isdir(os.path.join(base_dir, f)) and \
                          (not f.startswith('.') or text.startswith('.'))
                          ]
        else:
            completion = [ prefix + f
                          for f in os.listdir(base_dir)
                          if f.startswith(text) and \
                          os.path.isfile(os.path.join(base_dir, f)) and \
                          (not f.startswith('.') or text.startswith('.'))
                          ]

            completion = completion + \
                         [prefix + f + os.path.sep
                          for f in os.listdir(base_dir)
                          if f.startswith(text) and \
                          os.path.isdir(os.path.join(base_dir, f)) and \
                          (not f.startswith('.') or text.startswith('.'))
                          ]

        if relative:
            completion += [prefix + f for f in ['.'+os.path.sep, '..'+os.path.sep] if \
                       f.startswith(text) and not prefix.startswith('.')]

        return completion





#===============================================================================
# Question with auto-completion
#===============================================================================
class SmartQuestion(cmd.Cmd):
    """ a class for answering a question with the path autocompletion"""

    def preloop(self):
        """Initializing before starting the main loop"""
        self.prompt = ''
        self.value = None

    def __init__(self,  allow_arg=[], default=None, *arg, **opt):
        self.allow_arg = [str(a) for a in allow_arg]
        self.history_header = ''
        self.default_value = str(default)
        cmd.Cmd.__init__(self, *arg, **opt)

    def completenames(self, text, *ignored):
        signal.alarm(0) # avoid timer if any
        try:
            return Cmd.list_completion(text, self.allow_arg)
        except Exception, error:
            print error
            
    def default(self, line):
        """Default action if line is not recognized"""

        if line == '' and self.default_value is not None:
            self.value = self.default_value
        else:
            self.value = line

    def emptyline(self):
        """If empty line, return default"""
        
        if self.default_value is not None:
            self.value = self.default_value

    def postcmd(self, stop, line):
        
        try:    
            if self.value in self.allow_arg:
                return True
            else:
                raise Exception
        except Exception:
            print """not valid argument. Valid argument are in (%s).""" \
                          % ','.join(self.allow_arg)
            print 'please retry'
            return False
            
    def cmdloop(self, intro=None):
        cmd.Cmd.cmdloop(self, intro)
        return self.value
    
# a function helper
def smart_input(input_text, allow_arg=[], default=None):
    print input_text
    obj = SmartQuestion(allow_arg=allow_arg, default=default)
    return obj.cmdloop()

#===============================================================================
# Question in order to return a path with auto-completion
#===============================================================================
class OneLinePathCompletion(SmartQuestion):
    """ a class for answering a question with the path autocompletion"""


    def completenames(self, text, *ignored):
        signal.alarm(0) # avoid timer if any
        
        return SmartQuestion.completenames(self, text) + Cmd.path_completion(text,'.', only_dirs = False)
            
    def postcmd(self, stop, line):
        
        try:    
            if self.value in self.allow_arg: 
                return True
            elif os.path.isfile(self.value):
                return os.path.relpath(self.value)
            else:
                raise Exception
        except Exception, error:
            print """not valid argument. Valid argument are file path or value in (%s).""" \
                          % ','.join(self.allow_arg)
            print 'please retry'
            return False
            
# a function helper
def raw_path_input(input_text, allow_arg=[], default=None):
    print input_text
    obj = OneLinePathCompletion(allow_arg=allow_arg, default=default )
    return obj.cmdloop()

#===============================================================================
# 
#===============================================================================
class CmdFile(file):
    """ a class for command input file -in order to debug cmd \n problem"""
    
    def __init__(self, name, opt='rU'):
        
        file.__init__(self, name, opt)
        self.text = file.read(self)
        self.close()
        self.lines = self.text.split('\n')
    
    def readline(self, *arg, **opt):
        """readline method treating correctly a line whithout \n at the end
           (add it)
        """
        if self.lines:
            line = self.lines.pop(0)
        else:
            return ''
        
        if line.endswith('\n'):
            return line
        else:
            return line + '\n'
    
    def __next__(self):
        return self.lines.__next__()    
    def __iter__(self):
        return self.lines.__iter__()

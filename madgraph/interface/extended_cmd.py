################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
"""  A file containing different extension of the cmd basic python library"""


import cmd
import logging
import os
import pydoc
import re
import signal
import subprocess
import sys
import traceback
try:
    import readline
    GNU_SPLITTING = ('GNU' in readline.__doc__)
except:
    readline = None
    GNU_SPLITTING = True


logger = logging.getLogger('cmdprint') # for stdout
logger_stderr = logging.getLogger('fatalerror') # for stderr

try:
    import madgraph.various.misc as misc
    from madgraph import MG5DIR
    MADEVENT = False
except ImportError, error:
    try:
        import internal.misc as misc
    except:
        raise error
    MADEVENT = True


pjoin = os.path.join

class TimeOutError(Exception):
    """Class for run-time error"""

def debug(debug_only=True):

    def deco_debug(f):
        
        if debug_only and not __debug__:
            return f
        
        def deco_f(*args, **opt):
            try:
                return f(*args, **opt)
            except Exception, error:
                logger.error(error)
                logger.error(traceback.print_exc(file=sys.stdout))
                return
        return deco_f
    return deco_debug
            

#===============================================================================
# CmdExtended
#===============================================================================
class BasicCmd(cmd.Cmd):
    """Simple extension for the readline"""

    def preloop(self):
        if readline and not 'libedit' in readline.__doc__:
            readline.set_completion_display_matches_hook(self.print_suggestions)

    def deal_multiple_categories(self, dico):
        """convert the multiple category in a formatted list understand by our
        specific readline parser"""

        if 'libedit' in readline.__doc__:
            # No parser in this case, just send all the valid options
            out = []
            for name, opt in dico.items():
                out += opt
            return out

        # check if more than one categories but only one value:
        if all(len(s) <= 1 for s in dico.values() ):
            values = set((s[0] for s in dico.values() if len(s)==1))
            if len(values) == 1:
                return values
                
        # That's the real work
        out = []
        valid=0
        # if the key starts with number order the key with that number.
        for name, opt in dico.items():
            if not opt:
                continue
            name = name.replace(' ', '_')
            valid += 1
            out.append(opt[0].rstrip()+'@@'+name+'@@')
            # Remove duplicate
            d = {}
            for x in opt:
                d[x] = 1    
            opt = list(d.keys())
            opt.sort()
            out += opt

            
        if valid == 1:
            out = out[1:]
        return out
    
    @debug()
    def print_suggestions(self, substitution, matches, longest_match_length) :
        """print auto-completions by category"""
        if not hasattr(self, 'completion_prefix'):
            self.completion_prefix = ''
        longest_match_length += len(self.completion_prefix)
        try:
            if len(matches) == 1:
                self.stdout.write(matches[0]+' ')
                return
            self.stdout.write('\n')
            l2 = [a[-2:] for a in matches]
            if '@@' in l2:
                nb_column = self.getTerminalSize()//(longest_match_length+1)
                pos=0
                for val in self.completion_matches:
                    if val.endswith('@@'):
                        category = val.rsplit('@@',2)[1]
                        category = category.replace('_',' ')
                        self.stdout.write('\n %s:\n%s\n' % (category, '=' * (len(category)+2)))
                        start = 0
                        pos = 0
                        continue
                    elif pos and pos % nb_column ==0:
                        self.stdout.write('\n')
                    self.stdout.write(self.completion_prefix + val + \
                                      ' ' * (longest_match_length +1 -len(val)))
                    pos +=1
                self.stdout.write('\n')
            else:
                # nb column
                nb_column = self.getTerminalSize()//(longest_match_length+1)
                for i,val in enumerate(matches):
                    if i and i%nb_column ==0:
                        self.stdout.write('\n')
                    self.stdout.write(self.completion_prefix + val + \
                                     ' ' * (longest_match_length +1 -len(val)))
                self.stdout.write('\n')
    
            self.stdout.write(self.prompt+readline.get_line_buffer())
            self.stdout.flush()
        except Exception, error:
            if __debug__:
                logger.error(error)
            
    def getTerminalSize(self):
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                                                     '1234'))
            except Exception:
                return None
            return cr
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except Exception:
                pass
        if not cr:
            try:
                cr = (os.environ['LINES'], os.environ['COLUMNS'])
            except Exception:
                cr = (25, 80)
        return int(cr[1])
    
    def complete(self, text, state):
        """Return the next possible completion for 'text'.
         If a command has not been entered, then complete against command list.
         Otherwise try to call complete_<command> to get list of completions.
        """

        if state == 0:
            import readline
            origline = readline.get_line_buffer()
            line = origline.lstrip()
            stripped = len(origline) - len(line)
            begidx = readline.get_begidx() - stripped
            endidx = readline.get_endidx() - stripped
            
            if ';' in line:
                begin, line = line.rsplit(';',1)
                begidx = begidx - len(begin) - 1
                endidx = endidx - len(begin) - 1
                if line[:begidx] == ' ' * begidx:
                    begidx=0

            if begidx>0:
                cmd, args, foo = self.parseline(line)
                if cmd == '':
                    compfunc = self.completedefault
                else:
                    try:
                        compfunc = getattr(self, 'complete_' + cmd)
                    except AttributeError:
                        compfunc = self.completedefault
            else:
                compfunc = self.completenames
                
            # correct wrong splittion with '\ '
            if line and begidx > 2 and line[begidx-2:begidx] == '\ ':
                Ntext = line.split(os.path.sep)[-1]
                self.completion_prefix = Ntext.rsplit('\ ', 1)[0] + '\ '
                to_rm = len(self.completion_prefix) - 1
                Nbegidx = len(line.rsplit(os.path.sep, 1)[0]) + 1
                data = compfunc(Ntext.replace('\ ', ' '), line, Nbegidx, endidx)
                self.completion_matches = [p[to_rm:] for p in data 
                                              if len(p)>to_rm]                
            # correct wrong splitting with '-'
            elif line and line[begidx-1] == '-':
             try:    
                Ntext = line.split()[-1]
                self.completion_prefix = Ntext.rsplit('-',1)[0] +'-'
                to_rm = len(self.completion_prefix)
                Nbegidx = len(line.rsplit(None, 1)[0])
                data = compfunc(Ntext, line, Nbegidx, endidx)
                self.completion_matches = [p[to_rm:] for p in data 
                                              if len(p)>to_rm]
             except Exception, error:
                 print error
            else:
                self.completion_prefix = ''
                self.completion_matches = compfunc(text, line, begidx, endidx)

        self.completion_matches = [ l if l[-1] in [' ','@','=',os.path.sep]
                                    else ((l + ' ') if not l.endswith('\\$') else l[:-2])
                                  for l in self.completion_matches if l] 
        
        try:
            return self.completion_matches[state]
        except IndexError, error:
            # if __debug__:
            #    logger.error('\n Completion ERROR:')
            #    logger.error( error)
            return None    
        
    @staticmethod
    def split_arg(line):
        """Split a line of arguments"""
        
        split = line.split()
        out=[]
        tmp=''
        for data in split:
            if data[-1] == '\\':
                tmp += data[:-1]+' '
            elif tmp:
                tmp += data
                tmp = os.path.expanduser(os.path.expandvars(tmp))
                out.append(tmp)
            else:
                out.append(data)
        return out
    
    @staticmethod
    def list_completion(text, list, line=''):
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
        base_dir = os.path.expanduser(os.path.expandvars(base_dir))
        
        if text == '~':
            text = '~/'
        prefix, text = os.path.split(text)
        prefix = os.path.expanduser(os.path.expandvars(prefix))
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
        
        completion = [a.replace(' ','\ ') for a in completion]
        return completion




class CheckCmd(object):
    """Extension of the cmd object for only the check command"""

    def check_history(self, args):
        """check the validity of line"""
        
        if len(args) > 1:
            self.help_history()
            raise self.InvalidCmd('\"history\" command takes at most one argument')
        
        if not len(args):
            return
        
        if args[0] =='.':
            if not self._export_dir:
                raise self.InvalidCmd("No default directory is defined for \'.\' option")
        elif args[0] != 'clean':
                dirpath = os.path.dirname(args[0])
                if dirpath and not os.path.exists(dirpath) or \
                       os.path.isdir(args[0]):
                    raise self.InvalidCmd("invalid path %s " % dirpath)
    
    def check_save(self, args):
        """check that the line is compatible with save options"""
        
        if len(args) > 2:
            self.help_save()
            raise self.InvalidCmd, 'too many arguments for save command.'
        
        if len(args) == 2:
            if args[0] != 'options':
                self.help_save()
                raise self.InvalidCmd, '\'%s\' is not recognized as first argument.' % \
                                                args[0]
            else:
                args.pop(0)           

class HelpCmd(object):
    """Extension of the cmd object for only the help command"""

    def help_quit(self):
        logger.info("-- terminates the application",'$MG:color:BLUE')
        logger.info("syntax: quit",'$MG:color:BLACK')
    
    help_EOF = help_quit

    def help_history(self):
        logger.info("-- interact with the command history.",'$MG:color:BLUE')
        logger.info("syntax: history [FILEPATH|clean|.] ",'$MG:color:BLACK')
        logger.info(" > If FILEPATH is \'.\' and \'output\' is done,")
        logger.info("   Cards/proc_card_mg5.dat will be used.")
        logger.info(" > If FILEPATH is omitted, the history will be output to stdout.")
        logger.info("   \"clean\" will remove all entries from the history.")
        
    def help_help(self):
        logger.info("-- access to the in-line help",'$MG:color:BLUE')
        logger.info("syntax: help",'$MG:color:BLACK')

    def help_save(self):
        """help text for save"""
        logger.info("-- save options configuration to filepath.",'$MG:color:BLUE')
        logger.info("syntax: save [options]  [FILEPATH]",'$MG:color:BLACK') 
        
    def help_display(self):
        """help for display command"""
        logger.info("-- display a the status of various internal state variables",'$MG:color:BLUE')          
        logger.info("syntax: display " + "|".join(self._display_opts),'$MG:color:BLACK')
        
class CompleteCmd(object):
    """Extension of the cmd object for only the complete command"""

    def complete_display(self,text, line, begidx, endidx):        
        args = self.split_arg(line[0:begidx])
        # Format
        if len(args) == 1:
            return self.list_completion(text, self._display_opts)
        
    def complete_history(self, text, line, begidx, endidx):
        "Complete the history command"

        args = self.split_arg(line[0:begidx])

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))

        if len(args) == 1:
            return self.path_completion(text)

    def complete_save(self, text, line, begidx, endidx):
        "Complete the save command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, ['options'])

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin('.',*[a for a in args if a.endswith(os.path.sep)]),
                                        only_dirs = True)

        # Filename if directory is not given
        if len(args) == 2:
            return self.path_completion(text)

class Cmd(CheckCmd, HelpCmd, CompleteCmd, BasicCmd):
    """Extension of the cmd.Cmd command line.
    This extensions supports line breaking, history, comments,
    internal call to cmdline, path completion,...
    this class should be MG5 independent"""

    #suggested list of command
    next_possibility = {} # command : [list of suggested command]
    history_header = ""
    
    _display_opts = ['options','variable']
    
    class InvalidCmd(Exception):
        """expected error for wrong command"""
        pass    
    
    ConfigurationError = InvalidCmd

    debug_output = 'debug'
    error_debug = """Please report this bug to developers\n
           More information is found in '%(debug)s'.\n
           Please attach this file to your report."""
    config_debug = error_debug
           
    keyboard_stop_msg = """stopping all current operation
            in order to quit the program please enter exit"""
 
    
    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
        self.log = True
        self.history = []
        self.save_line = '' # for line splitting
        cmd.Cmd.__init__(self, *arg, **opt)
        self.__initpos = os.path.abspath(os.getcwd())
        self.child = None # sub CMD interface call from this one
        self.mother = None #This CMD interface was called from another one
        self.inputfile = None # input file (in non interactive mode)
        self.haspiping = not sys.stdin.isatty() # check if mg5 is piped 
        self.stored_line = '' # for be able to treat answer to question in input file 
                              # answer which are not required.
        if not hasattr(self, 'helporder'):
            self.helporder = ['Documented commands']
        
        
    def precmd(self, line):
        """ A suite of additional function needed for in the cmd
        this implement history, line breaking, comment treatment,...
        """
        
        if not line:
            return line
        line = line.lstrip()

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
        if ';' in line: 
            lines = line.split(';')
            for subline in lines:
                if not (subline.startswith("history") or subline.startswith('help') \
                        or subline.startswith('#*')): 
                    self.history.append(subline)           
                stop = self.onecmd(subline)
                stop = self.postcmd(stop, subline)
            return ''
            
        # execute the line command
        self.history.append(line) 
        return line

    def postcmd(self,stop, line):
        """ finishing a command
        This looks if the command add a special post part."""

        if line.strip():
            try:
                cmd, subline = line.split(None, 1)
            except ValueError:
                pass
            else:
                if hasattr(self,'post_%s' %cmd):
                    stop = getattr(self, 'post_%s' % cmd)(stop, subline)
        return stop

    def define_child_cmd_interface(self, obj_instance, interface=True):
        """Define a sub cmd_interface"""

        # We are in a file reading mode. So we need to redirect the cmd
        self.child = obj_instance
        self.child.mother = self

        if self.use_rawinput and interface:
            # We are in interactive mode -> simply call the child
            obj_instance.cmdloop()
            stop = obj_instance.postloop()
            return stop
        if self.inputfile:
            # we are in non interactive mode -> so pass the line information
            obj_instance.inputfile = self.inputfile
        
        obj_instance.haspiping = self.haspiping
        
        if not interface:
            return self.child
 
    #===============================================================================
    # Ask a question with nice options handling
    #===============================================================================    
    def ask(self, question, default, choices=[], path_msg=None, 
            timeout = True, fct_timeout=None, ask_class=None, alias={},**opt):
        """ ask a question with some pre-define possibility
            path info is
        """
        if path_msg:
            path_msg = [path_msg]
        else:
            path_msg = []
            
        if timeout:
            try:
                timeout = self.options['timeout']
            except Exception:
                pass
                    
        # add choice info to the question
        if choices + path_msg:
            question += ' ['
            question += "\033[%dm%s\033[0m, " % (4, default)    
            for data in choices[:9] + path_msg:
                if default == data:
                    continue
                else:
                    question += "%s, " % data
                    
            if len(choices) > 9:
                question += '... , ' 
            question = question[:-2]+']'
        else:
            question += "[\033[%dm%s\033[0m] " % (4, default)    
        if ask_class:
            obj = ask_class  
        elif path_msg:
            obj = OneLinePathCompletion
        else:
            obj = SmartQuestion

        if alias:
            choices += alias.keys()
        
        question_instance = obj(question, allow_arg=choices, default=default, 
                                                   mother_interface=self, **opt)
        if not self.haspiping:
            if hasattr(obj, "haspiping"):
                obj.haspiping = self.haspiping
            
            
        answer = self.check_answer_in_input_file(question_instance, default, path_msg)
        if answer is not None:
            if answer in alias:
                answer = alias[answer]
            if ask_class:
                answer = question_instance.default(answer)
            return answer
        
        question = question_instance.question
        value =   Cmd.timed_input(question, default, timeout=timeout,
                                 fct=question_instance, fct_timeout=fct_timeout)
        
        try:
            if value in alias:
                value = alias[value]
        except TypeError:
            pass
        if value == default and ask_class:
            value = question_instance.default(default)
        return value
 
    def do_import(self, line):
        """Advanced commands: Import command files"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_import(args)
        
        # Execute the card
        self.import_command_file(args[1])
         
    def check_import(self, args):
        """check import command"""
        
        if '-f' in args:
            self.force = True
            args.remove('-f')
        if args[0] != 'command':
            args.set(0, 'command')
        if len(args) != 2:
            raise self.InvalidCmd('import command requires one filepath argument')
        if not os.path.exists(args[1]):
            raise 'No such file or directory %s' % args[1]
        
    
    def check_answer_in_input_file(self, question_instance, default, path=False):
        """Questions can have answer in output file (or not)"""

        if not self.inputfile:
            return None# interactive mode

        line = self.get_stored_line()
        # line define if a previous answer was not answer correctly 
        if not line:
            try:
                line = self.inputfile.next()
            except StopIteration:
                if self.haspiping:
                    logger.debug('piping')
                    self.store_line(line)
                    return None # print the question and use the pipe
                logger.info(question_instance.question)
                logger.warning('The answer to the previous question is not set in your input file')
                logger.warning('Use %s value' % default)
                return str(default)
        
        line = line.replace('\n','').strip()
        if '#' in line: 
            line = line.split('#')[0]
        if not line:
            # Comment or empty line, pass to the next one
            return self.check_answer_in_input_file(question_instance, default, path)
        options = question_instance.allow_arg
        if line in options:
            return line
        elif hasattr(question_instance, 'do_%s' % line.split()[0]):
            #This is a command line, exec it and check next line
            logger.info(line)
            fct = getattr(question_instance, 'do_%s' % line.split()[0])
            fct(' '.join(line.split()[1:]))
            return self.check_answer_in_input_file(question_instance, default, path)
        elif path:
            line = os.path.expanduser(os.path.expandvars(line))
            if os.path.exists(line):
                return line
        # No valid answer provides
        if self.haspiping:
            self.store_line(line)
            return None # print the question and use the pipe
        else:
            logger.info(question_instance.question)
            logger.warning('The answer to the previous question is not set in your input file')
            logger.warning('Use %s value' % default)
            self.store_line(line)
            return str(default)

    def store_line(self, line):
        """store a line of the input file which should be executed by the higher mother"""
        
        if self.mother:
            self.mother.store_line(line)
        else:
            self.stored_line = line

    def get_stored_line(self):
        """return stored line and clean it"""
        if self.mother:
            value = self.mother.get_stored_line()
            self.mother.stored_line = None
        else:
            value = self.stored_line
            self.stored_line = None    
        return value



    def nice_error_handling(self, error, line):
        """ """ 
        # Make sure that we are at the initial position
        if self.child:
            return self.child.nice_error_handling(error, line)
        
        os.chdir(self.__initpos)
        # Create the debug files
        self.log = False
        if os.path.exists(self.debug_output):
            os.remove(self.debug_output)
        try:
            cmd.Cmd.onecmd(self, 'history %s' % self.debug_output.replace(' ', '\ '))
        except Exception, error:
           logger.error(error)

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
        error_text += self.error_debug % {'debug':self.debug_output}
        logger_stderr.critical(error_text)
        
                
        # Add options status to the debug file
        try:
            self.do_display('options', debug_file)
        except Exception, error:
            debug_file.write('Fail to write options with error %s' % error)
        
        #add the cards:
        for card in ['proc_card_mg5.dat','param_card.dat', 'run_card.dat']:
            try:
                ff = open(pjoin(self.me_dir, 'Cards', card))
                debug_file.write(ff.read())
                ff.close()
            except Exception:
                pass
            
        #stop the execution if on a non interactive mode
        if self.use_rawinput == False:
            return True 
        return False



    def nice_user_error(self, error, line):
        if self.child:
            return self.child.nice_user_error(error, line)
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
    
    def nice_config_error(self, error, line):
        if self.child:
            return self.child.nice_user_error(error, line)
        # Make sure that we are at the initial position                                 
        os.chdir(self.__initpos)
        if not self.history or line == self.history[-1]:
            error_text = 'Error detected in \"%s\"\n' % line
        else:
            error_text = 'Error detected in sub-command %s\n' % self.history[-1]
        error_text += 'write debug file %s \n' % self.debug_output
        self.log = False
        cmd.Cmd.onecmd(self, 'history %s' % self.debug_output)
        debug_file = open(self.debug_output, 'a')
        traceback.print_exc(file=debug_file)
        error_text += self.config_debug % {'debug' :self.debug_output}
        error_text += '%s : %s' % (error.__class__.__name__,
                                                str(error).replace('\n','\n\t'))
        logger_stderr.error(error_text)
        
        # Add options status to the debug file
        try:
            self.do_display('options', debug_file)
        except Exception, error:
            debug_file.write('Fail to write options with error %s' % error)
        #stop the execution if on a non interactive mode                                
        if self.use_rawinput == False:
            return True
        # Remove failed command from history                                            
        if self.history:
            self.history.pop()
        return False
    
    def onecmd_orig(self, line, **opt):
        """Interpret the argument as though it had been typed in response
        to the prompt.

        The return value is a flag indicating whether interpretation of
        commands by the interpreter should stop.
        
        This allow to pass extra argument for internal call.
        """
        if '~/' in line and os.environ.has_key('HOME'):
            line = line.replace('~/', '%s/' % os.environ['HOME'])
        line = os.path.expandvars(line)
        cmd, arg, line = self.parseline(line)
        if not line:
            return self.emptyline()
        if cmd is None:
            return self.default(line)
        self.lastcmd = line
        if cmd == '':
            return self.default(line)
        else:
            try:
                func = getattr(self, 'do_' + cmd)
            except AttributeError:
                return self.default(line)
            return func(arg, **opt)


    def onecmd(self, line, **opt):
        """catch all error and stop properly command accordingly"""
        
        try:
            return self.onecmd_orig(line, **opt)
        except self.InvalidCmd as error:            
            if __debug__:
                self.nice_error_handling(error, line)
                self.history.pop()
            else:
                self.nice_user_error(error, line)
        except self.ConfigurationError as error:
            self.nice_config_error(error, line)
        except Exception as error:
            self.nice_error_handling(error, line)
            if self.mother:
                self.do_quit('')
        except KeyboardInterrupt as error:
            self.stop_on_keyboard_stop()
            if __debug__:
                self.nice_config_error(error, line)
            logger.error(self.keyboard_stop_msg)
    
    def stop_on_keyboard_stop(self):
        """action to perform to close nicely on a keyboard interupt"""
        pass # dummy function
            
    def exec_cmd(self, line, errorhandling=False, printcmd=True, 
                                     precmd=False, postcmd=True, **opt):
        """for third party call, call the line with pre and postfix treatment
        without global error handling """

        if printcmd and not line.startswith('#'):
            logger.info(line)
        if self.child:
            current_interface = self.child
        else:
            current_interface = self
        
        if precmd:
            line = current_interface.precmd(line)
        if errorhandling:
            stop = current_interface.onecmd(line, **opt)
        else:
            stop = Cmd.onecmd_orig(current_interface, line, **opt)
        if postcmd:
            stop = current_interface.postcmd(stop, line)
        return stop      

    def run_cmd(self, line):
        """for third party call, call the line with pre and postfix treatment
        with global error handling"""
        
        return self.exec_cmd(line, errorhandling=True, precmd=True)
    
    def emptyline(self):
        """If empty line, do nothing. Default is repeat previous command."""
        pass
    
    def default(self, line, log=True):
        """Default action if line is not recognized"""

        # Faulty command
        if log:
            logger.warning("Command \"%s\" not recognized, please try again" % \
                                                                line.split()[0])
        if line.strip() in ['q', '.q', 'stop']:
            logger.info("If you want to quit mg5 please type \"exit\".")

        if self.history and self.history[-1] == line:        
            self.history.pop()
        



     
    # Write the list of command line use in this session
    def do_history(self, line):
        """write in a file the suite of command that was used"""
        
        args = self.split_arg(line)
        # Check arguments validity
        self.check_history(args)

        if len(args) == 0:
            logger.info('\n'.join(self.history))
            return
        elif args[0] == 'clean':
            self.history = []
            logger.info('History is cleaned')
            return
        elif args[0] == '.':
            output_file = os.path.join(self._export_dir, 'Cards', \
                                       'proc_card_mg5.dat')
            output_file = open(output_file, 'w')
        else:
            output_file = open(args[0], 'w')
            
        # Create the command file
        text = self.get_history_header()
        text += ('\n'.join(self.history) + '\n') 
        
        #write this information in a file
        output_file.write(text)
        output_file.close()

        if self.log:
            logger.info("History written to " + output_file.name)

    def compile(self, *args, **opts):
        """ """
        
        return misc.compile(nb_core=self.options['nb_core'], *args, **opts)

    def avoid_history_duplicate(self, line, no_break=[]):
        """remove all line in history (but the last) starting with line.
        up to the point when a line didn't start by something in no_break.
        (reading in reverse order)"""
        
        new_history = []
        for i in range(1, len(self.history)+1):
            cur_line = self.history[-i]
            if i == 1:
                new_history.append(cur_line)
            elif not any((cur_line.startswith(text) for text in no_break)):
                to_add = self.history[:-i+1]
                to_add.reverse()
                new_history += to_add
                break
            elif cur_line.startswith(line):
                continue
            else:
                new_history.append(cur_line)
            
        new_history.reverse()
        self.history[:] = new_history
        
                        
    def import_command_file(self, filepath):
        # remove this call from history
        if self.history:
            self.history.pop()
        
        # Read the lines of the file and execute them
        if isinstance(filepath, str):
            commandline = open(filepath).readlines()
        else:
            commandline = filepath
        oldinputfile = self.inputfile
        oldraw = self.use_rawinput
        self.inputfile = (l for l in commandline) # make a generator
        self.use_rawinput = False
        # Note using "for line in open(filepath)" is not safe since the file
        # filepath can be overwritten during the run (leading to weird results)
        # Note also that we need a generator and not a list.
        for line in self.inputfile:
            #remove pointless spaces and \n
            line = line.replace('\n', '').strip()
            # execute the line
            if line:
                self.exec_cmd(line, precmd=True)
            stored = self.get_stored_line()
            while stored:
                line = stored
                self.exec_cmd(line, precmd=True)
                stored = self.get_stored_line()

        # If a child was open close it
        if self.child:
            self.child.exec_cmd('quit')        
        self.inputfile = oldinputfile
        self.use_rawinput = oldraw       
        return
    
    def get_history_header(self):
        """Default history header"""
        
        return self.history_header
    
    def postloop(self):
        """ """
        
        args = self.split_arg(self.lastcmd)
        if args and args[0] in ['quit','exit']:
            if 'all' in args:
                return True
            if len(args) >1 and args[1].isdigit():
                if args[1] not in  ['0', '1']:
                    return True
        return False
        
    #===============================================================================
    # Ask a question with a maximum amount of time to answer
    #===============================================================================    
    @staticmethod
    def timed_input(question, default, timeout=None, noerror=True, fct=None,
                    fct_timeout=None):
        """ a question with a maximal time to answer take default otherwise"""
    
        def handle_alarm(signum, frame): 
            raise TimeOutError
        
        signal.signal(signal.SIGALRM, handle_alarm)
    
        if fct is None:
            fct = raw_input
        
        if timeout:
            signal.alarm(timeout)
            question += '[%ss to answer] ' % (timeout)    
        try:
            result = fct(question)
        except TimeOutError:
            if noerror:
                logger.info('\nuse %s' % default)
                if fct_timeout:
                    fct_timeout(True)
                return default
            else:
                signal.alarm(0)
                raise
        finally:
            signal.alarm(0)
        if fct_timeout:
            fct_timeout(False)
        return result



        


    # Quit
    def do_quit(self, line):
        """Not in help: exit the mainloop() """
        
        if self.child:
            self.child.exec_cmd('quit ' + line, printcmd=False)
            return
        elif self.mother:
            self.mother.child = None
            if line == 'all':
                pass
            elif line:
                level = int(line) - 1
                if level:
                    self.mother.lastcmd = 'quit %s' % level
        logger.info(' ')
        return True

    # Aliases
    do_EOF = do_quit
    do_exit = do_quit

    def do_help(self, line):
        """Not in help: propose some usefull possible action """
                
        # if they are an argument use the default help
        if line:
            return cmd.Cmd.do_help(self, line)
        
        
        names = self.get_names()
        cmds = {}
        names.sort()
        # There can be duplicates if routines overridden
        prevname = ''
        for name in names:
            if name[:3] == 'do_':
                if name == prevname:
                    continue
                prevname = name
                cmdname=name[3:]
                try:
                    doc = getattr(self.cmd, name).__doc__
                except Exception:
                    doc = None
                if not doc:
                    doc = getattr(self, name).__doc__
                if not doc:
                    tag = "Documented commands"
                elif ':' in doc:
                    tag = doc.split(':',1)[0]
                else:
                    tag = "Documented commands"
                if tag in cmds:
                    cmds[tag].append(cmdname)
                else:
                    cmds[tag] = [cmdname] 
                    
        self.stdout.write("%s\n"%str(self.doc_leader))
        for tag in self.helporder:
            if tag not in cmds:
                continue
            header = "%s (type help <topic>):" % tag
            self.print_topics(header, cmds[tag],   15,80)
        for name, item in cmds.items():
            if name in self.helporder:
                continue
            if name == "Not in help":
                continue
            header = "%s (type help <topic>):" % name
            self.print_topics(header, item,   15,80)


        ## Add contextual help
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
        
        logger.info('Contextual Help')
        logger.info('===============')
        if last_action_2 in authorize:
            options = self.next_possibility[last_action_2]
        elif last_action in authorize:
            options = self.next_possibility[last_action]
        else:
            return
        text = 'The following command(s) may be useful in order to continue.\n'
        for option in options:
            text+='\t %s \n' % option      
        logger.info(text)

    def do_display(self, line, output=sys.stdout):
        """Advanced commands: basic display"""
        
        args = self.split_arg(line)
        #check the validity of the arguments
        
        if len(args) == 0:
            self.help_display()
            raise self.InvalidCmd, 'display require at least one argument'
        
        if args[0] == "options":
            outstr = "Value of current Options:\n" 
            for key, value in self.options.items():
                outstr += '%25s \t:\t%s\n' %(key,value)
            output.write(outstr)
            
        elif args[0] == "variable":
            outstr = "Value of Internal Variable:\n"
            try:
                var = eval(args[1])
            except Exception:
                outstr += 'GLOBAL:\nVariable %s is not a global variable\n' % args[1]
            else:
                outstr += 'GLOBAL:\n' 
                outstr += misc.nice_representation(var, nb_space=4)
               
            try:
                var = eval('self.%s' % args[1])
            except Exception:
                outstr += 'LOCAL:\nVariable %s is not a local variable\n' % args[1]
            else:
                outstr += 'LOCAL:\n'
                outstr += misc.nice_representation(var, nb_space=4)
            split =  args[1].split('.')
            for i, name in enumerate(split):
                try:
                    __import__('.'.join(split[:i+1]))                    
                    exec('%s=sys.modules[\'%s\']' % (split[i], '.'.join(split[:i+1])))
                except ImportError:
                    try:
                        var = eval(args[1])
                    except Exception, error:
                        outstr += 'EXTERNAL:\nVariable %s is not a external variable\n' % args[1]
                        break
                    else:
                        outstr += 'EXTERNAL:\n'
                        outstr += misc.nice_representation(var, nb_space=4)                        
                else:
                    var = eval(args[1])
                    outstr += 'EXTERNAL:\n'
                    outstr += misc.nice_representation(var, nb_space=4)                        
            
            pydoc.pager(outstr)
    
    
    def do_save(self, line, check=True):
        """Save the configuration file"""
        
        args = self.split_arg(line)
        # Check argument validity
        if check:
            Cmd.check_save(self, args)
            
        # find base file for the configuration
        if'HOME' in os.environ and os.environ['HOME']  and \
        os.path.exists(pjoin(os.environ['HOME'], '.mg5', 'mg5_configuration.txt')):
            base = pjoin(os.environ['HOME'], '.mg5', 'mg5_configuration.txt')
            if hasattr(self, 'me_dir'):
                basedir = self.me_dir
            elif not MADEVENT:
                basedir = MG5DIR
            else:
                basedir = os.getcwd()
        elif MADEVENT:
            # launch via ./bin/madevent
            for config_file in ['me5_configuration.txt', 'amcatnlo_configuration.txt']:
                if os.path.exists(pjoin(self.me_dir, 'Cards', config_file)): 
                    base = pjoin(self.me_dir, 'Cards', config_file)
            basedir = self.me_dir
        else:
            if hasattr(self, 'me_dir'):
                base = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
                if len(args) == 0 and os.path.exists(base):
                    self.write_configuration(base, base, self.me_dir)
            base = pjoin(MG5DIR, 'input', 'mg5_configuration.txt')
            basedir = MG5DIR
            
        if len(args) == 0:
            args.append(base)
        self.write_configuration(args[0], base, basedir, self.options)
        
    def write_configuration(self, filepath, basefile, basedir, to_keep):
        """Write the configuration file"""
        # We use the default configuration file as a template.
        # to ensure that all configuration information are written we 
        # keep track of all key that we need to write.

        logger.info('save configuration file to %s' % filepath)
        to_write = to_keep.keys()
        text = ""
        # Use local configuration => Need to update the path
        for line in file(basefile):
            if '=' in line:
                data, value = line.split('=',1)
            else: 
                text += line
                continue
            data = data.strip()
            if data.startswith('#'):
                key = data[1:].strip()
            else: 
                key = data 
            if '#' in value:
                value, comment = value.split('#',1)
            else:
                comment = ''    
            if key in to_keep:
                value = str(to_keep[key])
            else:
                text += line
                continue
            try:
                to_write.remove(key)
            except Exception:
                pass
            if '_path' in key:       
                # special case need to update path
                # check if absolute path
                if not os.path.isabs(value):
                    value = os.path.realpath(os.path.join(basedir, value))
            text += '%s = %s # %s \n' % (key, value, comment)
        for key in to_write:
            if key in to_keep:
                text += '%s = %s \n' % (key, to_keep[key])
        
        if not MADEVENT:
            text += """\n# MG5 MAIN DIRECTORY\n"""
            text += "mg5_path = %s\n" % MG5DIR         
        
        writer = open(filepath,'w')
        writer.write(text)
        writer.close()
                       

    

class CmdShell(Cmd):
    """CMD command with shell activate"""

    # Access to shell
    def do_shell(self, line):
        "Run a shell command"

        if line.strip() is '':
            self.help_shell()
        else:
            logging.info("running shell command: " + line)
            subprocess.call(line, shell=True)
    
    def complete_shell(self, text, line, begidx, endidx):
        """ add path for shell """

        # Filename if directory is given
        #
        if len(self.split_arg(line[0:begidx])) > 1 and line[begidx - 1] == os.path.sep:
            if not text:
                text = ''
            output = self.path_completion(text,
                                        base_dir=\
                                          self.split_arg(line[0:begidx])[-1])
        else:
            output = self.path_completion(text)
        return output

    def help_shell(self):
        """help for the shell"""
        logger.info("-- run the shell command CMD and catch output",'$MG:color:BLUE')        
        logger.info("syntax: shell CMD (or ! CMD)",'$MG:color:BLACK')




#===============================================================================
# Question with auto-completion
#===============================================================================
class SmartQuestion(BasicCmd):
    """ a class for answering a question with the path autocompletion"""

    def preloop(self):
        """Initializing before starting the main loop"""
        self.prompt = '>'
        self.value = None
        BasicCmd.preloop(self)
        

    def __init__(self, question, allow_arg=[], default=None, 
                                            mother_interface=None, *arg, **opt):
        self.question = question
        self.wrong_answer = 0 # forbids infinite loop
        self.allow_arg = [str(a) for a in allow_arg]
        self.history_header = ''
        self.default_value = str(default)
        self.mother_interface = mother_interface
        cmd.Cmd.__init__(self, *arg, **opt)

    def __call__(self, question, reprint_opt=True, **opts):
        
        self.question = question
        for key,value in opts:
            setattr(self, key, value)
        if reprint_opt:
            print question
        return self.cmdloop()
        

    def completenames(self, text, line, *ignored):
        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        try:
            out = {}
            out[' Options'] = Cmd.list_completion(text, self.allow_arg)
            out[' Recognized command'] = BasicCmd.completenames(self, text)
            
            return self.deal_multiple_categories(out)
        except Exception, error:
            print error

    def get_names(self):
        # This method used to pull in base class attributes
        # at a time dir() didn't do it yet.
        return dir(self)  
    
    def onecmd(self, line, **opt):
        """catch all error and stop properly command accordingly
        Interpret the argument as though it had been typed in response
        to the prompt.

        The return value is a flag indicating whether interpretation of
        commands by the interpreter should stop.
        
        This allow to pass extra argument for internal call.
        """
        try:
            if '~/' in line and os.environ.has_key('HOME'):
                line = line.replace('~/', '%s/' % os.environ['HOME'])
            line = os.path.expandvars(line)
            cmd, arg, line = self.parseline(line)
            if not line:
                return self.emptyline()
            if cmd is None:
                return self.default(line)
            self.lastcmd = line
            if cmd == '':
                return self.default(line)
            else:
                try:
                    func = getattr(self, 'do_' + cmd)
                except AttributeError:
                    return self.default(line)
                return func(arg, **opt)        
        except Exception as error:
            logger.warning(error)  
            
    def reask(self, reprint_opt=True):
        pat = re.compile('\[(\d*)s to answer\]')
        prev_timer = signal.alarm(0) # avoid timer if any
        
        if prev_timer:     
            if pat.search(self.question):
                timeout = int(pat.search(self.question).groups()[0])
            else:
                timeout=20
            print
            signal.alarm(timeout)
        if reprint_opt:
            if not prev_timer:
                self.question = pat.sub('',self.question)
            print self.question
        return False
        
    def default(self, line):
        """Default action if line is not recognized"""

        if line.strip() == '' and self.default_value is not None:
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
            elif str(self.value) == 'EOF':
                self.value = self.default_value
                return True
            elif line and hasattr(self, 'do_%s' % line.split()[0]):
                return self.reask()
            elif self.value == 'repeat':
                return self.reask()
            elif len(self.allow_arg)==0:
                return True
            else: 
                raise Exception
        except Exception,error:
            if self.wrong_answer < 100:
                self.wrong_answer += 1
                logger.warning("""%s not valid argument. Valid argument are in (%s).""" \
                          % (self.value,','.join(self.allow_arg)))
                logger.warning('please retry')
                return False
            else:
                self.value = self.default_value
                return True
                
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
    
    completion_prefix=''

    def completenames(self, text, line, begidx, endidx):
        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        
        try:
            out = {}
            out[' Options'] = Cmd.list_completion(text, self.allow_arg)
            out[' Path from ./'] = Cmd.path_completion(text, only_dirs = False)
            out[' Recognized command'] = BasicCmd.completenames(self, text)
            
            return self.deal_multiple_categories(out)
        except Exception, error:
            print error
            
    def precmd(self, *args):
        """ """
        
        signal.alarm(0)
        return SmartQuestion.precmd(self, *args)
        
    def completedefault(self,text, line, begidx, endidx):
        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        try:
            args = Cmd.split_arg(line[0:begidx])
        except Exception, error:
            print error

        # Directory continuation                 
        if args[-1].endswith(os.path.sep):

            return Cmd.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                               if a.endswith(os.path.sep)]))
        self.completenames(line+text)


    def postcmd(self, stop, line):
        try:    
            if self.value in self.allow_arg: 
                return True
            elif self.value and os.path.isfile(self.value):
                return os.path.relpath(self.value)
            elif self.value and str(self.value) == 'EOF':
                self.value = self.default_value
                return True
            elif line and hasattr(self, 'do_%s' % line.split()[0]):
                # go to retry
                reprint_opt = True 
            elif self.value == 'repeat':
                reprint_opt = True         
            else:
                raise Exception
        except Exception, error:            
            print """not valid argument. Valid argument are file path or value in (%s).""" \
                          % ','.join(self.allow_arg)
            print 'please retry'
            reprint_opt = False 
            
        return self.reask(reprint_opt)

            
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





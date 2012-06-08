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
import pydoc
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
except Exception, error:
    if __debug__:
        print error
    import internal.misc as misc
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
                print error
                print traceback.print_exc(file=sys.stdout)
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
                 print error
            
    def getTerminalSize(self):
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct, os
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                                                     '1234'))
            except:
                return None
            return cr
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            try:
                cr = (env['LINES'], env['COLUMNS'])
            except:
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
        #print self.completion_matches

        self.completion_matches = [ (l[-1] in [' ','@','=',os.path.sep] 
                      and l or (l+' ')) for l in self.completion_matches if l]
        
        try:
            return self.completion_matches[state]
        except IndexError, error:
            #if __debug__:
            #    print '\n Completion ERROR:'
            #    print error
            #    print '\n'
            return None    

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
        logger.info("syntax: quit")
        logger.info("-- terminates the application")
    
    help_EOF = help_quit

    def help_history(self):
        logger.info("syntax: history [FILEPATH|clean|.] ")
        logger.info("   If FILEPATH is \'.\' and \'output\' is done,")
        logger.info("   Cards/proc_card_mg5.dat will be used.")
        logger.info("   If FILEPATH is omitted, the history will be output to stdout.")
        logger.info("   \"clean\" will remove all entries from the history.")
        
    def help_help(self):
        logger.info("syntax: help")
        logger.info("-- access to the in-line help" )

    def help_save(self):
        """help text for save"""
        logger.info("syntax: save [options]  [FILEPATH]") 
        logger.info("-- save options configuration to filepath.")
        
    def help_display(self):
        """help for display command"""
        logger.info("syntax: display " + "|".join(self._display_opts))
        logger.info("-- display a the status of various internal state variables")          
        
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
           More information is found in '%s'.\n
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
        self.stored_line = '' # for be able to treat answer to question in input file 
                             # answer which are not required.

        
        
    def precmd(self, line):
        """ A suite of additional function needed for in the cmd
        this implement history, line breaking, comment treatment,...
        """
        
        if not line:
            return line
        line = line.lstrip()

        # Update the history of this suite of command,
        # except for useless commands (empty history and help calls)
        if ';' in line: 
            lines = line.split(';')
        else:
            lines = [line]
        for l in lines:
            l = l.strip()
            if not (l.startswith("history") or l.startswith('help') or \
                                                            l.startswith('#*')):
                self.history.append(l)

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
            print error
            
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
        error_text += self.error_debug % {'debug': self.debug_output}
        logger_stderr.critical(error_text)
                
        # Add options status to the debug file
        try:
            self.do_display('options', debug_file)
        except Exception, error:
            debug_file.write('Fail to write options with error %s' % error)
        
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


    def onecmd(self, line):
        """catch all error and stop properly command accordingly"""
        
        try:
            return cmd.Cmd.onecmd(self, line)
        except self.InvalidCmd as error:            
            if __debug__:
                self.nice_error_handling(error, line)
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
            #self.nice_error_handling(error, line)
            print self.keyboard_stop_msg
    
    def stop_on_keyboard_stop(self):
        """action to perform to close nicely on a keyboard interupt"""
        pass # dummy function
            
    def exec_cmd(self, line, errorhandling=False, printcmd=True, 
                                                 precmd=False, postcmd=True):
        """for third party call, call the line with pre and postfix treatment
        without global error handling """

        if printcmd:
            logger.info(line)
        if self.child:
            current_interface = self.child
        else:
            current_interface = self
        
        if precmd:
            line = current_interface.precmd(line)
        if errorhandling:
            stop = current_interface.onecmd(line)
        else:
            stop = cmd.Cmd.onecmd(current_interface, line)
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
    
    def default(self, line):
        """Default action if line is not recognized"""

        # Faulty command
        logger.warning("Command \"%s\" not recognized, please try again" % \
                                                                line.split()[0])
        

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

    def correct_splitting(line):
        """if the line finish with a '-' the code splits in a weird way
           on GNU_SPLITTING"""
                
        line = line.lstrip()
        if line[-1] in [' ','\t']:
            return '', line, len(line),len(enidx)
        return text, line, begidx, endidx
        
        
        
     
    # Write the list of command line use in this session
    def do_history(self, line):
        """write in a file the suite of command that was used"""
        
        args = self.split_arg(line)
        # Check arguments validity
        self.check_history(args)

        if len(args) == 0:
            print '\n'.join(self.history)
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

    def clean_history(self, to_keep=['set','add','load'],
                            remove_bef_lb1=None,
                            to_remove=['open','display','launch'],
                            keep_last=False):
        """Remove all commands in arguments from history"""
        
    
        nline = -1
        last = 0
        while nline > -len(self.history):

            if remove_bef_lb1 and self.history[nline].startswith(remove_bef_lb1):
                if last:
                    last = 2
                    self.history.pop(nline)
                    continue
                else:
                    last=1
            if nline == -1 and keep_last:
                nline -=1
                continue
            if any([self.history[nline].startswith(arg) for arg in to_remove]):
                self.history.pop(nline)
                continue
            if last == 2:
                if not any([self.history[nline].startswith(arg) for arg in to_keep]):
                  self.history.pop(nline)
                  continue
                else:
                    nline -= 1  
            else:
                nline -= 1
                
    def import_command_file(self, filepath):
        # remove this call from history
        if self.history:
            self.history.pop()
        
        # Read the lines of the file and execute them
        self.inputfile = open(filepath)
        for line in self.inputfile:
            #remove pointless spaces and \n
            line = line.replace('\n', '').strip()
            # execute the line
            if line:
                self.exec_cmd(line, precmd=True)
            if self.stored_line: # created by intermediate question
                self.exec_cmd(self.stored_line, precmd=True)
                self.stored_line = ''
        # If a child was open close it
        if self.child:
            self.child.exec_cmd('quit')        
                
        return
    
    def get_history_header(self):
        """Default history header"""
        
        return self.history_header

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
        
        if not interface:
            return self.child
    
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
                print '\nuse %s' % default
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


    #===============================================================================
    # Ask a question with nice options handling
    #===============================================================================    
    def ask(self, question, default, choices=[], path_msg=None, 
            timeout = True, fct_timeout=None):
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
            except:
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
                
        if path_msg:
            f = lambda q: raw_path_input(q, allow_arg=choices, default=default)
        else:
            f = lambda q: smart_input(q, allow_arg=choices, default=default)

        answer = self.check_answer_in_input_file(choices, path_msg)
        if answer is not None:
            return answer
        
        value =   Cmd.timed_input(question, default, timeout=timeout,
                                    fct=f, fct_timeout=fct_timeout)

        return value
        
    def check_answer_in_input_file(self, options, path=False):
        """Questions can have answer in output file (or not)"""

        if self.check_stored_line():
            return None # already one line not correctly answer 

        if not self.inputfile:
            return None# interactive mode

        try:
            line = self.inputfile.next()
        except StopIteration:
            return None
        line = line.replace('\n','').strip()
        if '#' in line: 
            line = line.split('#')[0]
        if not line:
            return self.check_answer_in_input_file(options, path)

        if line in options:
            return line
        elif path and os.path.exists(line):
            return line
        else:
            self.store_line(line)
            return None

    def store_line(self, line):
        """store a line of the input file which should be executed by the higher mother"""
        
        if self.mother:
            self.mother.store_line(line)
        else:
            self.stored_line = line

    def check_stored_line(self):
        if self.mother:
            return self.mother.check_stored_line()
        else:
            return self.stored_line

    # Quit
    def do_quit(self, line):
        """ exit the mainloop() """
        
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

        return True

    # Aliases
    do_EOF = do_quit
    do_exit = do_quit


     
    

    def do_help(self, line):
        """ propose some usefull possible action """
                
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
                except:
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
        tag = "Documented commands"
        header = "%s (type help <topic>):" % tag
        self.print_topics(header, cmds[tag],   15,80)
        for name, item in cmds.items():
            if name == "Documented commands":
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
            except:
                outstr += 'GLOBAL:\nVariable %s is not a global variable\n' % args[1]
            else:
                outstr += 'GLOBAL:\n' 
                outstr += misc.nice_representation(var, nb_space=4)
               
            try:
                var = eval('self.%s' % args[1])
            except:
                outstr += 'LOCAL:\nVariable %s is not a local variable\n' % args[1]
            else:
                outstr += 'LOCAL:\n'
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
            base = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
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
        self.write_configuration(args[0], base, basedir)
        
    def write_configuration(self, filepath, basefile, basedir):
        """Write the configuration file"""
        # We use the default configuration file as a template.
        # to ensure that all configuration information are written we 
        # keep track of all key that we need to write.

        logger.info('save configuration file to %s' % filepath)
        to_write = self.options.keys()[:]
        text = ""
        # Use local configuration => Need to update the path
        for line in file(basefile):
            if '#' in line:
                data, comment = line.split('#',1)
            else: 
                data, comment = line, ''
            data = data.split('=')
            if len(data) !=2:
                text += line
                continue
            key = data[0].strip()
            if key in self.options:
                value = str(self.options[key])
            else:
                value = data[1].strip()
            try:
                to_write.remove(key)
            except:
                pass
            if '_path' in key:       
                # special case need to update path
                # check if absolute path
                if value.startswith('./'):
                    value = os.path.realpath(os.path.join(basedir, value))
            text += '%s = %s # %s \n' % (key, value, comment)
        for key in to_write:
            if key in self.options:
                text += '%s = %s \n' % (key,self.options[key])
            else:
                text += '%s = %s \n' % (key,self.options[key])
        writer = open(filepath,'w')
        writer.write(text)
        writer.close()
                       



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
        
        completion = [a.replace(' ','\ ') for a in completion]
        return completion
    

    

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
        
        logger.info("syntax: shell CMD (or ! CMD)")
        logger.info("-- run the shell command CMD and catch output")




#===============================================================================
# Question with auto-completion
#===============================================================================
class SmartQuestion(BasicCmd):
    """ a class for answering a question with the path autocompletion"""

    def preloop(self):
        """Initializing before starting the main loop"""
        self.prompt = ''
        self.value = None
        BasicCmd.preloop(self)

    def __init__(self,  allow_arg=[], default=None, *arg, **opt):
        self.wrong_answer = 0 # forbids infinite loop
        self.allow_arg = [str(a) for a in allow_arg]
        self.history_header = ''
        self.default_value = str(default)
        cmd.Cmd.__init__(self, *arg, **opt)

    def completenames(self, text, line, *ignored):
        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        try:
            return Cmd.list_completion(text, self.allow_arg)
        except Exception, error:
            print error
            
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
            else: 
                raise Exception
        except Exception:
            if self.wrong_answer < 100:
                self.wrong_answer += 1
                print """%s not valid argument. Valid argument are in (%s).""" \
                          % (self.value,','.join(self.allow_arg))
                print 'please retry'
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
    

    def completenames(self, text, line, begidx, endidx):
        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        try:
            out = {}
            out[' Options'] = SmartQuestion.completenames(self, text, line)
            out[' Path from ./'] = Cmd.path_completion(text, only_dirs = False)
            
            return self.deal_multiple_categories(out)
        except Exception, error:
            print error

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





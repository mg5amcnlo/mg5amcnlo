################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Classes to write good-looking output in different languages:
Fortran, C++, etc."""


import re
import collections

class FileWriter(file):
    """Generic Writer class. All writers should inherit from this class."""

    supported_preprocessor_commands = ['if']
    preprocessor_command_re=re.compile(
                          "\s*(?P<command>%s)\s*\(\s*(?P<body>.*)\s*\)\s*{\s*"\
                                   %('|'.join(supported_preprocessor_commands)))
    preprocessor_endif_re=re.compile(\
    "\s*}\s*(?P<endif>else)?\s*(\((?P<body>.*)\))?\s*(?P<new_block>{)?\s*")
    
    class FileWriterError(IOError):
        """Exception raised if an error occurs in the definition
        or the execution of a Writer."""

        pass

    class FilePreProcessingError(IOError):
        """Exception raised if an error occurs in the handling of the
        preprocessor tags '##' in the template file."""
        pass

    def __init__(self, name, opt = 'w'):
        """Initialize file to write to"""

        return file.__init__(self, name, opt)

    def write_line(self, line):
        """Write a line with proper indent and splitting of long lines
        for the language in question."""

        pass

    def write_comment_line(self, line):
        """Write a comment line, with correct indent and line splits,
        for the language in question"""

        pass

    def write_comments(self, lines):
        """Write set of comment lines, with correct indent and line splits,
        for the language in question"""

        splitlines = []
        if isinstance(lines, list):
            for line in lines:
                if not isinstance(line, str):
                    raise self.FileWriterError("%s not string" % repr(line))
                splitlines.extend(line.split('\n'))
        elif isinstance(lines, str):
            splitlines.extend(lines.split('\n'))
        else:
            raise self.FileWriterError("%s not string" % repr(lines))

        for line in splitlines:
            res_lines = self.write_comment_line(line)
            for line_to_write in res_lines:
                self.write(line_to_write)

        pass

    def writelines(self, lines, context={}):
        """Extends the regular file.writeline() function to write out
        nicely formatted code. When defining a context, then the lines
        will be preprocessed to apply possible conditional statements on the
        content of the template depending on the contextual variables specified."""

        splitlines = []
        if isinstance(lines, list):
            for line in lines:
                if not isinstance(line, str):
                    raise self.FileWriterError("%s not string" % repr(line))
                splitlines.extend(line.split('\n'))
        elif isinstance(lines, str):
            splitlines.extend(lines.split('\n'))
        else:
            raise self.FileWriterError("%s not string" % repr(lines))

        if len(context)>0:
            splitlines = self.preprocess_template(splitlines,context=context)

        for line in splitlines:
            res_lines = self.write_line(line)
            for line_to_write in res_lines:
                self.write(line_to_write)
                
    def preprocess_template(self, input_lines, context={}):
        """ This class takes care of applying the pre-processing statements
        starting with ## in the template .inc files, using the contextual
        variables specified in the dictionary 'context' given in input with
        the variable names given as keys and their respective value as values."""
        
        template_lines = []
        if isinstance(input_lines, list):
            for line in input_lines:
                if not isinstance(line, str):
                    raise self.FileWriterError("%s not string" % repr(input_lines))
                template_lines.extend(line.split('\n'))
        elif isinstance(input_lines, str):
            template_lines.extend(input_lines.split('\n'))
        else:
            raise self.FileWriterError("%s not string" % repr(input_lines))
        
        # Setup the contextual environment
        for contextual_variable, value in context.items():
            exec('%s=%s'%(str(contextual_variable),repr(value)))
        
        res = []
        # The variable below tracks the conditional statements structure
        if_stack = []
        for i, line in enumerate(template_lines):
            if not line.startswith('##'):
                if all(if_stack):
                    res.append(line)
                continue
            preproc_command = self.preprocessor_command_re.match(line[2:])
            # Treat the follow up of an if statement
            if preproc_command is None:
                preproc_endif = self.preprocessor_endif_re.match(line[2:])
                if len(if_stack)==0 or preproc_endif is None:
                    raise self.FilePreProcessingError, 'Incorrect '+\
                             'preprocessing command %s at line %d.'%(line,i)
                if preproc_endif.group('new_block') is None:
                    if_stack.pop()
                elif preproc_endif.group('endif')=='else':
                    if_stack[-1]=(not if_stack[-1])
            # Treat an if statement
            elif preproc_command.group('command')=='if':
                try:
                    if_stack.append(eval(preproc_command.group('body'))==True)
                except Exception, e:
                    raise self.FilePreProcessingError, 'Could not evaluate'+\
                      "python expression '%s' given the context %s provided."%\
                            (preproc_command.group('body'),str(context))+\
                                           "\nLine %d of file %s."%(i,self.name)
        
        if len(if_stack)>0:
            raise self.FilePreProcessingError, 'Some conditional statements are'+\
                                                     ' not properly terminated.'
        return res

#===============================================================================
# FortranWriter
#===============================================================================
class FortranWriter(FileWriter):
    """Routines for writing fortran lines. Keeps track of indentation
    and splitting of long lines"""

    class FortranWriterError(FileWriter.FileWriterError):
        """Exception raised if an error occurs in the definition
        or the execution of a FortranWriter."""
        pass

    # Parameters defining the output of the Fortran writer
    keyword_pairs = {'^if.+then\s*$': ('^endif', 2),
                     '^type(?!\s*\()\s*.+\s*$': ('^endtype', 2),
                     '^do(?!\s+\d+)\s+': ('^enddo\s*$', 2),
                     '^subroutine': ('^end\s*$', 0),
                     'function': ('^end\s*$', 0)}
    single_indents = {'^else\s*$':-2,
                      '^else\s*if.+then\s*$':-2}
    number_re = re.compile('^(?P<num>\d+)\s+(?P<rest>.*)')
    line_cont_char = '$'
    comment_char = 'c'
    downcase = False
    line_length = 71
    max_split = 10
    split_characters = "+-*/,) "
    comment_split_characters = " "

    # Private variables
    __indent = 0
    __keyword_list = []
    __comment_pattern = re.compile(r"^(\s*#|c$|(c\s+([^=]|$))|cf2py)", re.IGNORECASE)

    def write_line(self, line):
        """Write a fortran line, with correct indent and line splits"""

        # This Routine is for a single line
        assert(isinstance(line, str) and line.find('\n') == -1)
        
        
        res_lines = []

        # Check if empty line and write it
        if not line.lstrip():
            res_lines.append("\n")
            return res_lines

        # Check if this line is a comment
        if self.__comment_pattern.search(line):
            # This is a comment
            res_lines = self.write_comment_line(line.lstrip()[1:])
            return res_lines

        else:
            # This is a regular Fortran line

            # Strip leading spaces from line
            myline = line.lstrip()

            # Check if line starts with number
            num_group = self.number_re.search(myline)
            num = ""
            if num_group:
                num = num_group.group('num')
                myline = num_group.group('rest')

            # Convert to upper or lower case
            # Here we need to make exception for anything within quotes.
            (myline, part, post_comment) = myline.partition("!")
            # Set space between line and post-comment
            if part:
                part = "  " + part
            # Replace all double quotes by single quotes
            myline = myline.replace('\"', '\'')
            # Downcase or upcase Fortran code, except for quotes
            splitline = myline.split('\'')
            myline = ""
            i = 0
            while i < len(splitline):
                if i % 2 == 1:
                    # This is a quote - check for escaped \'s
                    while  splitline[i] and splitline[i][-1] == '\\':
                        splitline[i] = splitline[i] + '\'' + splitline.pop(i + 1)
                else:
                    # Otherwise downcase/upcase
                    if FortranWriter.downcase:
                        splitline[i] = splitline[i].lower()
                    else:
                        splitline[i] = splitline[i].upper()
                i = i + 1

            myline = "\'".join(splitline).rstrip()

            # Check if line starts with dual keyword and adjust indent 
            if self.__keyword_list and re.search(self.keyword_pairs[\
                self.__keyword_list[-1]][0], myline.lower()):
                key = self.__keyword_list.pop()
                self.__indent = self.__indent - self.keyword_pairs[key][1]

            # Check for else and else if
            single_indent = 0
            for key in self.single_indents.keys():
                if re.search(key, myline.lower()):
                    self.__indent = self.__indent + self.single_indents[key]
                    single_indent = -self.single_indents[key]
                    break

            # Break line in appropriate places
            # defined (in priority order) by the characters in split_characters
            res = self.split_line(" " + num + \
                                  " " * (5 + self.__indent - len(num)) + myline,
                                  self.split_characters,
                                  " " * 5 + self.line_cont_char + \
                                  " " * (self.__indent + 1))

            # Check if line starts with keyword and adjust indent for next line
            for key in self.keyword_pairs.keys():
                if re.search(key, myline.lower()):
                    self.__keyword_list.append(key)
                    self.__indent = self.__indent + self.keyword_pairs[key][1]
                    break

            # Correct back for else and else if
            if single_indent != None:
                self.__indent = self.__indent + single_indent
                single_indent = None

        # Write line(s) to file
        res_lines.append("\n".join(res) + part + post_comment + "\n")

        return res_lines

    def write_comment_line(self, line):
        """Write a comment line, with correct indent and line splits"""
        
        # write_comment_line must have a single line as argument
        assert(isinstance(line, str) and line.find('\n') == -1)

        if line.startswith('F2PY'):
            return ["C%s\n" % line.strip()]
        

        res_lines = []

        # This is a comment
        myline = " " * (5 + self.__indent) + line.lstrip()
        if FortranWriter.downcase:
            self.comment_char = self.comment_char.lower()
        else:
            self.comment_char = self.comment_char.upper()
        myline = self.comment_char + myline
        # Break line in appropriate places
        # defined (in priority order) by the characters in
        # comment_split_characters
        res = self.split_line(myline,
                              self.comment_split_characters,
                              self.comment_char + " " * (5 + self.__indent))

        # Write line(s) to file
        res_lines.append("\n".join(res) + "\n")

        return res_lines

    def split_line(self, line, split_characters, line_start):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters, and start each new line with line_start."""

        res_lines = [line]

        while len(res_lines[-1]) > self.line_length:
            split_at = self.line_length
            for character in split_characters:
                index = res_lines[-1][(self.line_length - self.max_split): \
                                      self.line_length].rfind(character)
                if index >= 0:
                    split_at = self.line_length - self.max_split + index
                    break
            newline = res_lines[-1][split_at:]
            nquotes = self.count_number_of_quotes(newline)
            res_lines.append(line_start + 
              ('//\''+res_lines[-1][(split_at-1):] if nquotes%2==1 else 
               ''+res_lines[-1][split_at:]))
            res_lines[-2] = (res_lines[-2][:(split_at-1)]+'\'' if nquotes%2==1 \
                                                  else res_lines[-2][:split_at])
        return res_lines
    
    def count_number_of_quotes(self, line):
        """ Count the number of real quotes (not escaped ones) in a line. """
        
        splitline = line.split('\'')
        i = 0
        while i < len(splitline):
           if i % 2 == 1:
                # This is a quote - check for escaped \'s
                while  splitline[i] and splitline[i][-1] == '\\':
                    splitline[i] = splitline[i] + '\'' + splitline.pop(i + 1)
           i = i + 1
        return len(splitline)-1

#===============================================================================
# CPPWriter
#===============================================================================
class CPPWriter(FileWriter):
    """Routines for writing C++ lines. Keeps track of brackets,
    spaces, indentation and splitting of long lines"""

    class CPPWriterError(FileWriter.FileWriterError):
        """Exception raised if an error occurs in the definition
        or the execution of a CPPWriter."""
        pass

    # Parameters defining the output of the C++ writer
    standard_indent = 2
    line_cont_indent = 4

    indent_par_keywords = {'^if': standard_indent,
                           '^else if': standard_indent,
                           '^for': standard_indent,
                           '^while': standard_indent,
                           '^switch': standard_indent}
    indent_single_keywords = {'^else': standard_indent}
    indent_content_keywords = {'^class': standard_indent,
                              '^namespace': 0}        
    cont_indent_keywords = {'^case': standard_indent,
                            '^default': standard_indent,
                            '^public': standard_indent,
                            '^private': standard_indent,
                            '^protected': standard_indent}
    
    spacing_patterns = [('\s*\"\s*}', '\"'),
                        ('\s*,\s*', ', '),
                        ('\s*-\s*', ' - '),
                        ('([{(,=])\s*-\s*', '\g<1> -'),
                        ('(return)\s*-\s*', '\g<1> -'),
                        ('\s*\+\s*', ' + '),
                        ('([{(,=])\s*\+\s*', '\g<1> +'),
                        ('\(\s*', '('),
                        ('\s*\)', ')'),
                        ('\{\s*', '{'),
                        ('\s*\}', '}'),
                        ('\s*=\s*', ' = '),
                        ('\s*>\s*', ' > '),
                        ('\s*<\s*', ' < '),
                        ('\s*!\s*', ' !'),
                        ('\s*/\s*', '/'),
                        ('\s*\*\s*', ' * '),
                        ('\s*-\s+-\s*', '-- '),
                        ('\s*\+\s+\+\s*', '++ '),
                        ('\s*-\s+=\s*', ' -= '),
                        ('\s*\+\s+=\s*', ' += '),
                        ('\s*\*\s+=\s*', ' *= '),
                        ('\s*/=\s*', ' /= '),
                        ('\s*>\s+>\s*', ' >> '),
                        ('<\s*double\s*>>\s*', '<double> > '),
                        ('\s*<\s+<\s*', ' << '),
                        ('\s*-\s+>\s*', '->'),
                        ('\s*=\s+=\s*', ' == '),
                        ('\s*!\s+=\s*', ' != '),
                        ('\s*>\s+=\s*', ' >= '),
                        ('\s*<\s+=\s*', ' <= '),
                        ('\s*&&\s*', ' && '),
                        ('\s*\|\|\s*', ' || '),
                        ('\s*{\s*}', ' {}'),
                        ('\s*;\s*', '; '),
                        (';\s*\}', ';}'),
                        (';\s*$}', ';'),
                        ('\s*<\s*([a-zA-Z0-9]+?)\s*>', '<\g<1>>'),
                        ('^#include\s*<\s*(.*?)\s*>', '#include <\g<1>>'),
                        ('(\d+\.{0,1}\d*|\.\d+)\s*[eE]\s*([+-]{0,1})\s*(\d+)',
                         '\g<1>e\g<2>\g<3>'),
                        ('\s+',' ')]
    spacing_re = dict([(key[0], re.compile(key[0])) for key in \
                       spacing_patterns])

    init_array_pattern = re.compile(r"=\s*\{.*\}")
    short_clause_pattern = re.compile(r"\{.*\}")

    comment_char = '//'
    comment_pattern = re.compile(r"^(\s*#\s+|\s*//)")
    start_comment_pattern = re.compile(r"^(\s*/\*)")
    end_comment_pattern = re.compile(r"(\s*\*/)$")

    quote_chars = re.compile(r"[^\\][\"\']|^[\"\']")
    no_space_comment_patterns = re.compile(r"--|\*\*|==|\+\+")
    line_length = 80
    max_split = 40
    split_characters = " "
    comment_split_characters = " "
    
    # Private variables
    __indent = 0
    __keyword_list = collections.deque()
    __comment_ongoing = False

    def write_line(self, line):
        """Write a C++ line, with correct indent, spacing and line splits"""

        # write_line must have a single line as argument
        assert(isinstance(line, str) and line.find('\n') == -1)

        res_lines = []

        # Check if this line is a comment
        if self.comment_pattern.search(line) or \
               self.start_comment_pattern.search(line) or \
               self.__comment_ongoing:
            # This is a comment
            res_lines = self.write_comment_line(line.lstrip())
            return res_lines

        # This is a regular C++ line

        # Strip leading spaces from line
        myline = line.lstrip()

        # Return if empty line
        if not myline:
            return ["\n"]

        # Check if line starts with "{"
        if myline[0] == "{":
            # Check for indent
            indent = self.__indent
            key = ""
            if self.__keyword_list:
                key = self.__keyword_list[-1]
            if key in self.indent_par_keywords:
                indent = indent - self.indent_par_keywords[key]
            elif key in self.indent_single_keywords:
                indent = indent - self.indent_single_keywords[key]
            elif key in self.indent_content_keywords:
                indent = indent - self.indent_content_keywords[key]
            else:
                # This is free-standing block, just use standard indent
                self.__indent = self.__indent + self.standard_indent
            # Print "{"
            res_lines.append(" " * indent + "{" + "\n")
            # Add "{" to keyword list
            self.__keyword_list.append("{")
            myline = myline[1:].lstrip()
            if myline:
                # If anything is left of myline, write it recursively
                res_lines.extend(self.write_line(myline))
            return res_lines

        # Check if line starts with "}"
        if myline[0] == "}":
            # First: Check if no keywords in list
            if not self.__keyword_list:
                raise self.CPPWriterError(\
                                'Non-matching } in C++ output: ' \
                                + myline)                
            # First take care of "case" and "default"
            if self.__keyword_list[-1] in self.cont_indent_keywords.keys():
                key = self.__keyword_list.pop()
                self.__indent = self.__indent - self.cont_indent_keywords[key]
            # Now check that we have matching {
            if not self.__keyword_list.pop() == "{":
                raise self.CPPWriterError(\
                                'Non-matching } in C++ output: ' \
                                + ",".join(self.__keyword_list) + myline)
            # Check for the keyword before and close
            key = ""
            if self.__keyword_list:
                key = self.__keyword_list[-1]
            if key in self.indent_par_keywords:
                self.__indent = self.__indent - \
                                self.indent_par_keywords[key]
                self.__keyword_list.pop()
            elif key in self.indent_single_keywords:
                self.__indent = self.__indent - \
                                self.indent_single_keywords[key]
                self.__keyword_list.pop()
            elif key in self.indent_content_keywords:
                self.__indent = self.__indent - \
                                self.indent_content_keywords[key]
                self.__keyword_list.pop()
            else:
                # This was just a { } clause, without keyword
                self.__indent = self.__indent - self.standard_indent

            # Write } or };  and then recursively write the rest
            breakline_index = 1
            if len(myline) > 1:
                if myline[1] in [";", ","]:
                    breakline_index = 2
                elif myline[1:].lstrip()[:2] == "//":
                    if myline.endswith('\n'):
                        breakline_index = len(myline) - 1
                    else:
                        breakline_index = len(myline)
            res_lines.append("\n".join(self.split_line(\
                                       myline[:breakline_index],
                                       self.split_characters)) + "\n")
            if len(myline) > breakline_index and myline[breakline_index] =='\n':
                breakline_index +=1
            myline = myline[breakline_index:].lstrip()
            
            if myline:
                # If anything is left of myline, write it recursively
                res_lines.extend(self.write_line(myline))
            return res_lines

        # Check if line starts with keyword with parentesis
        for key in self.indent_par_keywords.keys():
            if re.search(key, myline):
                # Step through to find end of parenthesis
                parenstack = collections.deque()
                for i, ch in enumerate(myline[len(key)-1:]):
                    if ch == '(':
                        parenstack.append(ch)
                    elif ch == ')':
                        try:
                            parenstack.pop()
                        except IndexError:
                            # no opening parenthesis left in stack
                            raise self.CPPWriterError(\
                                'Non-matching parenthesis in C++ output' \
                                + myline)
                        if not parenstack:
                            # We are done
                            break
                endparen_index = len(key) + i
                # Print line, make linebreak, check if next character is {
                res_lines.append("\n".join(self.split_line(\
                                      myline[:endparen_index], \
                                      self.split_characters)) + \
                            "\n")
                myline = myline[endparen_index:].lstrip()
                # Add keyword to list and add indent for next line
                self.__keyword_list.append(key)
                self.__indent = self.__indent + \
                                self.indent_par_keywords[key]
                if myline:
                    # If anything is left of myline, write it recursively
                    res_lines.extend(self.write_line(myline))

                return res_lines
                    
        # Check if line starts with single keyword
        for key in self.indent_single_keywords.keys():
            if re.search(key, myline):
                end_index = len(key) - 1
                # Print line, make linebreak, check if next character is {
                res_lines.append(" " * self.__indent + myline[:end_index] + \
                            "\n")
                myline = myline[end_index:].lstrip()
                # Add keyword to list and add indent for next line
                self.__keyword_list.append(key)
                self.__indent = self.__indent + \
                                self.indent_single_keywords[key]
                if myline:
                    # If anything is left of myline, write it recursively
                    res_lines.extend(self.write_line(myline))

                return res_lines
                    
        # Check if line starts with content keyword
        for key in self.indent_content_keywords.keys():
            if re.search(key, myline):
                # Print line, make linebreak, check if next character is {
                if "{" in myline:
                    end_index = myline.index("{")
                else:
                    end_index = len(myline)
                res_lines.append("\n".join(self.split_line(\
                                      myline[:end_index], \
                                      self.split_characters)) + \
                            "\n")
                myline = myline[end_index:].lstrip()
                # Add keyword to list and add indent for next line
                self.__keyword_list.append(key)
                self.__indent = self.__indent + \
                                self.indent_content_keywords[key]
                if myline:
                    # If anything is left of myline, write it recursively
                    res_lines.extend(self.write_line(myline))

                return res_lines
                    
        # Check if line starts with continuous indent keyword
        for key in self.cont_indent_keywords.keys():
            if re.search(key, myline):
                # Check if we have a continuous indent keyword since before
                if self.__keyword_list[-1] in self.cont_indent_keywords.keys():
                    self.__indent = self.__indent - \
                                    self.cont_indent_keywords[\
                                       self.__keyword_list.pop()]
                # Print line, make linebreak
                res_lines.append("\n".join(self.split_line(myline, \
                                      self.split_characters)) + \
                            "\n")
                # Add keyword to list and add indent for next line
                self.__keyword_list.append(key)
                self.__indent = self.__indent + \
                                self.cont_indent_keywords[key]

                return res_lines
                    
        # Check if this line is an array initialization a ={b,c,d};
        if self.init_array_pattern.search(myline):
            res_lines.append("\n".join(self.split_line(\
                                      myline,
                                      self.split_characters)) + \
                        "\n")
            return res_lines

        # Check if this is a short xxx {yyy} type line;
        if self.short_clause_pattern.search(myline):
            lines = self.split_line(myline,
                                        self.split_characters)
            if len(lines) == 1:
                res_lines.append("\n".join(lines) + "\n")
                return res_lines

        # Check if there is a "{" somewhere in the line
        if "{" in myline:
            end_index = myline.index("{")
            res_lines.append("\n".join(self.split_line(\
                                      myline[:end_index], \
                                      self.split_characters)) + \
                        "\n")
            myline = myline[end_index:].lstrip()
            if myline:
                # If anything is left of myline, write it recursively
                res_lines.extend(self.write_line(myline))
            return res_lines

        # Check if there is a "}" somewhere in the line
        if "}" in myline:
            end_index = myline.index("}")
            res_lines.append("\n".join(self.split_line(\
                                      myline[:end_index], \
                                      self.split_characters)) + \
                        "\n")
            myline = myline[end_index:].lstrip()
            if myline:
                # If anything is left of myline, write it recursively
                res_lines.extend(self.write_line(myline))
            return res_lines

        # Write line(s) to file
        res_lines.append("\n".join(self.split_line(myline, \
                                              self.split_characters)) + "\n")

        # Check if this is a single indented line
        if self.__keyword_list:
            if self.__keyword_list[-1] in self.indent_par_keywords:
                self.__indent = self.__indent - \
                            self.indent_par_keywords[self.__keyword_list.pop()]
            elif self.__keyword_list[-1] in self.indent_single_keywords:
                self.__indent = self.__indent - \
                         self.indent_single_keywords[self.__keyword_list.pop()]
            elif self.__keyword_list[-1] in self.indent_content_keywords:
                self.__indent = self.__indent - \
                         self.indent_content_keywords[self.__keyword_list.pop()]

        return res_lines

    def write_comment_line(self, line):
        """Write a comment line, with correct indent and line splits"""

        # write_comment_line must have a single line as argument
        assert(isinstance(line, str) and line.find('\n') == -1)

        res_lines = []

        # This is a comment

        if self.start_comment_pattern.search(line):
            self.__comment_ongoing = True
            line = self.start_comment_pattern.sub("", line)

        if self.end_comment_pattern.search(line):
            self.__comment_ongoing = False
            line = self.end_comment_pattern.sub("", line)
            
        line = self.comment_pattern.sub("", line).strip()
        # Avoid extra space for lines starting with certain multiple patterns
        if self.no_space_comment_patterns.match(line):
            myline = self.comment_char + line
        else:
            myline = self.comment_char + " " + line
        # Break line in appropriate places defined (in priority order)
        # by the characters in comment_split_characters
        res = self.split_comment_line(myline)

        # Write line(s) to file
        res_lines.append("\n".join(res) + "\n")

        return res_lines

    def split_line(self, line, split_characters):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters. Also fix spacing for line."""

        # First split up line if there are comments
        comment = ""
        if line.find(self.comment_char) > -1:
            line, dum, comment = line.partition(self.comment_char)

        # Then split up line if there are quotes
        quotes = self.quote_chars.finditer(line)

        start_pos = 0
        line_quotes = []
        line_no_quotes = []
        for i, quote in enumerate(quotes):
            if i % 2 == 0:
                # Add text before quote to line_no_quotes
                line_no_quotes.append(line[start_pos:quote.start()])
                start_pos = quote.start()
            else:
                # Add quote to line_quotes
                line_quotes.append(line[start_pos:quote.end()])
                start_pos = quote.end()

        line_no_quotes.append(line[start_pos:])
        
        # Fix spacing for line, but only outside of quotes
        line.rstrip()
        for i, no_quote in enumerate(line_no_quotes):
            for key in self.spacing_patterns:
                no_quote = self.spacing_re[key[0]].sub(key[1], no_quote)
            line_no_quotes[i] = no_quote

        # Glue together quotes and non-quotes:
        line = line_no_quotes[0]
        for i in range(len(line_quotes)):
            line += line_quotes[i]
            if len(line_no_quotes) > i + 1:
                 line += line_no_quotes[i+1]

        # Add indent
        res_lines = [" " * self.__indent + line]

        while len(res_lines[-1]) > self.line_length:
            long_line = res_lines[-1]
            split_at = -1
            for character in split_characters:
                index = long_line[(self.line_length - self.max_split): \
                                      self.line_length].rfind(character)
                if index >= 0:
                    split_at = self.line_length - self.max_split + index + 1
                    break
                
            # no valid breaking so find the first breaking allowed:
            if split_at == -1:
                split_at = len(long_line)
                for character in split_characters:
                    split = long_line[self.line_length].find(character)
                    if split > 0:
                        split_at = min(split, split_at)
            if split_at == len(long_line):
                break
                    
            # Don't allow split within quotes
            quotes = self.quote_chars.findall(long_line[:split_at])
            if quotes and len(quotes) % 2 == 1:
                quote_match = self.quote_chars.search(long_line[split_at:])
                if not quote_match:
                    raise self.CPPWriterError(\
                        "Error: Unmatched quote in line " + long_line)
                split_at = quote_match.end() + split_at + 1
                split_match = re.search(self.split_characters,
                                        long_line[split_at:])
                if split_match:
                    split_at = split_at + split_match.start()
                else:
                    split_at = len(long_line) + 1

            # Append new line
            if long_line[split_at:].lstrip():
                # Replace old line
                res_lines[-1] = long_line[:split_at].rstrip()
                res_lines.append(" " * \
                                 (self.__indent + self.line_cont_indent) + \
                                 long_line[split_at:].strip())
            else:
                break

        if comment:
            res_lines[-1] += " " + self.comment_char + comment
            
        return res_lines

    def split_comment_line(self, line):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters."""

        # First fix spacing for line
        line.rstrip()
        res_lines = [" " * self.__indent + line]

        while len(res_lines[-1]) > self.line_length:
            long_line = res_lines[-1]
            split_at = self.line_length
            index = long_line[(self.line_length - self.max_split): \
                                  self.line_length].rfind(' ')
            if index >= 0:
                split_at = self.line_length - self.max_split + index + 1
            
            # Append new line
            if long_line[split_at:].lstrip():
                # Replace old line
                res_lines[-1] = long_line[:split_at].rstrip()
                res_lines.append(" " * \
                                 self.__indent + self.comment_char + " " + \
                                 long_line[split_at:].strip())
            else:
                break
            
        return res_lines

class PythonWriter(FileWriter):
    
    def write_comments(self, text):
        text = '#%s\n' % text.replace('\n','\n#')
        file.write(self, text)
        
class MakefileWriter(FileWriter):
    
    def write_comments(self, text):
        text = '#%s\n' % text.replace('\n','\n#')
        file.write(self, text)
        
    def writelines(self, lines):
        """Extends the regular file.writeline() function to write out
        nicely formatted code"""
        
        self.write(lines)

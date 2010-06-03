################################################################################
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
################################################################################

"""Classes to write good-looking output in different languages:
Fortran, C++, etc."""


import re
import collections

class FileWriter(file):
    """Generic Writer class. All writers should inherit from this class."""

    class FileWriterError(IOError):
        """Exception raised if an error occurs in the definition
        or the execution of a Writer."""

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

    def writelines(self, lines):
        """Extends the regular file.writeline() function to write out
        nicely formatted code"""

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
            res_lines = self.write_line(line)
            for line_to_write in res_lines:
                self.write(line_to_write)

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
                     '^do\s+': ('^enddo\s*$', 2),
                     '^subroutine': ('^end\s*$', 0),
                     'function': ('^end\s*$', 0)}
    single_indents = {'^else\s*$':-2,
                      '^else\s*if.+then\s*$':-2}
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
    __comment_pattern = re.compile(r"^(\s*#|c$|(c\s+([^=]|$)))", re.IGNORECASE)

    def write_line(self, line):
        """Write a fortran line, with correct indent and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.FortranWriterError, \
                  "write_fortran_line must have a single line as argument"

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
                    while splitline[i][len(splitline[i]) - 1] == '\\':
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
            res = self.split_line(" " * (6 + self.__indent) + myline,
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

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.FortranWriterError, \
                  "write_comment_line must have a single line as argument"

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

            res_lines.append(line_start + \
                             res_lines[-1][split_at:])
            res_lines[-2] = res_lines[-2][:split_at]

        return res_lines

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
                        ('\s*;\s*', '; '),
                        ('\s*,\s*', ', '),
                        ('\(\s*', '('),
                        ('\s*\)', ')'),
                        ('(\s*[^!=><])=([^=]\s*)', '\g<1> = \g<2>'),
                        ('(\s*[^/])/([^/]\s*)', '\g<1> / \g<2>'),
                        ('(\s*[^=])==([^=]\s*)', '\g<1> == \g<2>'),
                        ('(\s*[^>])>([^>=]\s*)', '\g<1> > \g<2>'),
                        ('(\s*[^<])<([^<=]\s*)', '\g<1> < \g<2>'),
                        ('\s*!([^=]\s*)', ' !\g<1>'),
                        ('\s*\+\s*', ' + '),
                        ('\s*-\s*', ' - '),
                        ('\s*\*\s*', ' * '),
                        ('\s*>>\s*', ' >> '),
                        ('\s*<<\s*', ' << '),
                        ('\s*!=\s*', ' != '),
                        ('\s*>=\s*', ' >= '),
                        ('\s*<=\s*', ' <= '),
                        ('\s*&&\s*', ' && '),
                        ('\s*\|\|\s*', ' || '),
                        ('\s*{\s*}', ' {}'),
                        ('\s+',' ')]
    spacing_re = dict([(key[0], re.compile(key[0])) for key in \
                       spacing_patterns])

    comment_char = '//'
    comment_pattern = re.compile(r"^(\s*#\s+|\s*//)")
    start_comment_pattern = re.compile(r"^(\s*/\*)")
    end_comment_pattern = re.compile(r"(\s*\*/)$")

    quote_chars = re.compile(r"[^\\]\"")

    line_length = 80
    max_split = 10
    split_characters = " "
    comment_split_characters = " "
    
    # Private variables
    __indent = 0
    __keyword_list = collections.deque()
    __comment_ongoing = False

    def write_line(self, line):
        """Write a C++ line, with correct indent, spacing and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.CPPWriterError, \
                  "write_line must have a single line as argument"

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

        # Don't print empty lines
        if not myline:
            return res_lines

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
                if myline[1] == ";":
                    breakline_index = 2
            res_lines.append(" " * self.__indent + myline[:breakline_index] + \
                        "\n")
            myline = myline[breakline_index + 1:].lstrip()
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
                    
        # Check if there is a "{}" in the line.
        # In that case just print the line
        if re.search("{\s*}", myline):
            res_lines.append("\n".join(self.split_line(\
                                      myline, \
                                      self.split_characters)) + \
                        "\n")
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

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.CPPWriterError, \
                  "write_comment_line must have a single line as argument"

        res_lines = []

        # This is a comment

        if self.start_comment_pattern.search(line):
            self.__comment_ongoing = True
            line = self.start_comment_pattern.sub("", line)

        if self.end_comment_pattern.search(line):
            self.__comment_ongoing = False
            line = self.end_comment_pattern.sub("", line)

        line = self.comment_pattern.sub("", line)
        myline = line.lstrip()
        myline = self.comment_char + " " + myline
        # Break line in appropriate places defined (in priority order)
        # by the characters in comment_split_characters
        res = self.split_line(myline, \
                              self.comment_split_characters)

        # Write line(s) to file
        res_lines.append("\n".join(res) + "\n")

        return res_lines

    def split_line(self, line, split_characters):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters."""

        # First fix spacing for line
        line.rstrip()
        for key in self.spacing_patterns:
            line = self.spacing_re[key[0]].sub(key[1], line)
        res_lines = [" " * self.__indent + line]

        while len(res_lines[-1]) > self.line_length:
            long_line = res_lines[-1]
            split_at = self.line_length
            for character in split_characters:
                index = long_line[(self.line_length - self.max_split): \
                                      self.line_length].rfind(character)
                if index >= 0:
                    split_at = self.line_length - self.max_split + index + 1
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
                res_lines[-1] = long_line[:split_at].lstrip().rstrip()
                res_lines.append(" " * \
                                 (self.__indent + self.line_cont_indent) + \
                                 long_line[split_at:].lstrip().rstrip())
            else:
                break
            
        return res_lines


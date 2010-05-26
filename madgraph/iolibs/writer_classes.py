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

#===============================================================================
# FortranWriter
#===============================================================================
class FortranWriter():
    """Routines for writing fortran lines. Keeps track of indentation
    and splitting of long lines"""

    class FortranWriterError(Exception):
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

    def write_fortran_line(self, fsock, line):
        """Write a fortran line, with correct indent and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.FortranWriterError, \
                  "write_fortran_line must have a single line as argument"

        # Check if this line is a comment
        if self.__comment_pattern.search(line):
            # This is a comment
            myline = " " * (5 + self.__indent) + line.lstrip()[1:].lstrip()
            if FortranWriter.downcase:
                self.comment_char = self.comment_char.lower()
            else:
                self.comment_char = self.comment_char.upper()
            myline = self.comment_char + myline
            part = ""
            post_comment = ""
            # Break line in appropriate places
            # defined (in priority order) by the characters in
            # comment_split_characters
            res = self.split_line(myline,
                                  self.comment_split_characters,
                                  self.comment_char + " " * (5 + self.__indent))
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
                self.__keyword_list[len(self.__keyword_list) - 1]][0],
                                               myline.lower()):
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
        fsock.write("\n".join(res) + part + post_comment + "\n")

        return True

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


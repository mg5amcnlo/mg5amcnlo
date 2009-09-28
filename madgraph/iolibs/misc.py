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

"""A set of functions performing routine administrative I/O tasks."""

import madgraph
import os
import re

def _parse_info_str(info_str):
    """Parse a newline separated list of "param=value" as a dictionnary
    """

    info_dict = {}
    pattern = re.compile("(?P<name>\w*)\s*=\s*(?P<value>.*)",
                         re.IGNORECASE | re.VERBOSE)
    entry_list = info_str.split('\n')
    for entry in entry_list:
        entry = entry.strip()
        if len(entry) == 0: continue
        m = pattern.match(entry)
        if m is not None:
            info_dict[m.group('name')] = m.group('value')
        else:
            raise IOError, "String %s is not a valid info string" % entry

    return info_dict


def get_pkg_info(info_str=None):
    """Returns the current version information of the MadGraph package, 
    as written in the VERSION text file. If the file cannot be found, 
    a dictionnary with empty values is returned. As an option, an info
    string can be passed to be read instead of the file content.
    """

    if info_str is None:
        try:
            # Open the file using the module path as a reference. Notice that 
            # module.__path__ is a list with only one element

            version_file = open(os.path.join(madgraph.__path__[0], "VERSION")
                                , 'r')
            try:
                info_str = version_file.read()
            finally:
                version_file.close()
        except IOError:
            info_str = ""

    return _parse_info_str(info_str)


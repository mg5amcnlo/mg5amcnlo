#!/usr/bin/env python
# Copyright (C) 2021 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, January 2017.

# This is a script to check if the XML documenation pages are well
# formed. Run as "./private/checkXML.py".

import glob
import xml.etree.cElementTree as xml
names = glob.glob('share/Pythia8/xmldoc/*.xml')
names = sorted(names)
for name in names:
    xmldoc = file(name).read().replace('&', 'AMPERSAND')
    name = name.split('/')[-1]
    try: xml.fromstring(xmldoc); print '  ', name
    except Exception, msg: print 'X ', name, msg

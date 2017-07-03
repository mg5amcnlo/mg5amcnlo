#!/usr/bin/env python
# Copyright (C) 2017 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, January 2017.

# This is a script to check if the XML documenation pages are well
# formed.

import glob
import xml.etree.ElementTree as xml
names = glob.glob('../share/Pythia8/xmldoc/*.xml')
names = sorted(names)
for name in names:
    xmldoc = file(name).read().replace('&', 'AMPERSAND')
    name = name.split('/')[-1]
    try: xml.fromstring(xmldoc); print '  ', name
    except Exception, msg: print 'X ', name, msg

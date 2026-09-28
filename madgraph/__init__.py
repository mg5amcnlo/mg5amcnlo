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
from __future__ import absolute_import
class MadGraph5Error(Exception):
    """Exception raised if an exception is find 
    Those Types of error will stop nicely in the cmd interface"""

class InvalidCmd(MadGraph5Error):
    """a class for the invalid syntax call"""

class aMCatNLOError(MadGraph5Error):
    """A MC@NLO error"""

import os
import logging
import shutil
import tempfile
import time
pjoin = os.path.join

#Look for basic file position MG5DIR and MG4DIR
MG5DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                os.path.pardir))
if ' ' in MG5DIR:
   logging.critical('''\033[1;31mpath to MG5: "%s" contains space. 
    This is likely to create code unstability. 
    Please consider changing the path location of the code\033[0m''' % MG5DIR)
   time.sleep(1)
MG4DIR = MG5DIR
ReadWrite = os.access(MG5DIR, os.W_OK) # W_OK is for writing

def atomic_copy(src, dst):
    """Copy src onto dst by rename, so that a concurrent reader of dst always
    sees a whole file (the old one or the new one) and never a truncated one.

    Duplicated from madgraph.various.misc.atomic_copy, which is the version to
    use everywhere else: misc imports this package, so this package cannot
    import misc.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst)),
                               prefix='.%s.' % os.path.basename(dst),
                               suffix='.tmp')
    os.close(fd)
    try:
        shutil.copy(src, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


if ReadWrite:
    # Temporary fix for problem with auto-update
    try:
        tmp_path = pjoin(MG5DIR, 'Template','LO','Source','make_opts')
        #1480375724 is 29/11/16
        if os.path.exists(tmp_path) and os.path.getmtime(tmp_path) < 1480375724:
            # Rename into place rather than remove-then-copy: this file is shared
            # by every MG5aMC process using this installation and is copied into
            # each new output directory (and parsed by make); it must never be
            # observed missing or half-written. NB the old code also referenced
            # shutil without importing it, so the copy raised NameError into the
            # except below and only the os.remove ever took effect.
            atomic_copy(pjoin(MG5DIR, 'Template','LO','Source','.make_opts'),
                        tmp_path)
    except Exception as error:
        pass
  
ADMIN_DEBUG = False  
if os.path.exists(os.path.join(MG5DIR,'bin', 'create_release.py')):
    if os.path.exists(os.path.join(MG5DIR,'.bzr')):
        ADMIN_DEBUG = True

if __debug__ or ADMIN_DEBUG:
    ordering = True
else:
    ordering = False
        

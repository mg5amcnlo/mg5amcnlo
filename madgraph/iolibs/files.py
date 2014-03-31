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

"""Methods and classes dealing with file access."""

import logging
import os
import shutil


logger = logging.getLogger('madgraph.files')

#===============================================================================
# read_from_file
#===============================================================================
def read_from_file(filename, myfunct, *args, **opt):
    """Open a file, apply the function myfunct (with sock as an arg) 
    on its content and return the result. Deals properly with errors and
    returns None if something goes wrong. 
    """

    try:
        sock = open(filename, 'r')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        if opt.has_key('print_error'):
            if not opt['print_error']:
                return None
        logger.error("I/O error on file %s (%s): %s" % (filename,errno, strerror))
        return None

    return ret_value

#===============================================================================
# write_to_file
#===============================================================================
def write_to_file(filename, myfunct, *args, **opts):
    """Open a file for writing, apply the function myfunct (with sock as an arg) 
    on its content and return the result. Deals properly with errors and
    returns None if something goes wrong. 
    """

    try:
        sock = open(filename, 'w')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        if 'log' not in opts or opts['log']:
            logger.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value

#===============================================================================
# append_to_file
#===============================================================================
def append_to_file(filename, myfunct, *args):
    """Open a file for appending, apply the function myfunct (with
    sock as an arg) on its content and return the result. Deals
    properly with errors and returns None if something goes wrong.
    """

    try:
        sock = open(filename, 'a')
        try:
            ret_value = myfunct(sock, *args)
        finally:
            sock.close()
    except IOError, (errno, strerror):
        logger.error("I/O error (%s): %s" % (errno, strerror))
        return None

    return ret_value

#===============================================================================
# check piclke validity
#===============================================================================
def is_uptodate(picklefile, path_list=None, min_time=1343682423):
    """Check if the pickle files is uptodate compare to a list of files. 
    If no files are given, the pickle files is checked against it\' current 
    directory"""
    
    if not os.path.exists(picklefile):
        return False
    
    if path_list is None:
        dirpath = os.path.dirname(picklefile)
        path_list = [ os.path.join(dirpath, file) for file in \
                                                            os.listdir(dirpath)]
    
    assert type(path_list) == list, 'is_update expect a list of files'
    
    pickle_date = os.path.getctime(picklefile)
    if pickle_date < min_time:
        return False
    
    for path in path_list:
        try:
            if os.path.getmtime(path) > pickle_date:
                return False
        except Exception:
            continue
    #all pass
    return True


################################################################################
## helper function for universal file treatment
################################################################################
def format_path(path):
    """Format the path in local format taking in entry a unix format"""
    if path[0] != '/':
        return os.path.join(*path.split('/'))
    else:
        return os.path.sep + os.path.join(*path.split('/'))
    
def cp(path1, path2, log=True, error=False):
    """ simple cp taking linux or mix entry"""
    path1 = format_path(path1)
    path2 = format_path(path2)
    try:
        shutil.copy(path1, path2)
    except IOError, why:
        try: 
            if os.path.exists(path2):
                path2 = os.path.join(path2, os.path.split(path1)[1])
            shutil.copytree(path1, path2)
        except IOError, why:
            if error:
                raise
            if log:
                logger.warning(why)
    except shutil.Error:
        # idetical file
        pass

def rm(path, log=True):
    """removes path, that can be a single element or a list"""
    if type(path) == list:
        for p in path:
            rm(p, log)
    else:
        path = format_path(path)
        try:
            os.remove(path)
        except OSError:
            shutil.rmtree(path, ignore_errors = True)

        
    
def mv(path1, path2):
    """simple mv taking linux or mix format entry"""
    path1 = format_path(path1)
    path2 = format_path(path2)
    try:
        shutil.move(path1, path2)
    except Exception:
        # An error can occur if the files exist at final destination
        if os.path.isfile(path2):
            os.remove(path2)
            shutil.move(path1, path2)
            return
        elif os.path.isdir(path2) and os.path.exists(
                                   os.path.join(path2, os.path.basename(path1))):      
            path2 = os.path.join(path2, os.path.basename(path1))
            os.remove(path2)
            shutil.move(path1, path2)
        else:
            raise
        
def ln(file_pos, starting_dir='.', name='', log=True, cwd=None, abspath=False):
    """a simple way to have a symbolic link without to have to change directory
    starting_point is the directory where to write the link
    file_pos is the file to link
    WARNING: not the linux convention
    """
    file_pos = format_path(file_pos)
    starting_dir = format_path(starting_dir)
    if not name:
        name = os.path.split(file_pos)[1]    

    if cwd:
        if not os.path.isabs(file_pos):
            file_pos = os.path.join(cwd, file_pos)
        if not os.path.isabs(starting_dir):
            starting_dir = os.path.join(cwd, starting_dir)        

    # Remove existing link if necessary
    if os.path.exists(os.path.join(starting_dir, name)):
        os.remove(os.path.join(starting_dir, name))

    if not abspath:
        target = os.path.relpath(file_pos, starting_dir)
    else:
        target = file_pos

    try:
        os.symlink(target, os.path.join(starting_dir, name))
    except Exception:
        if log:
            logger.warning('Could not link %s at position: %s' % (file_pos, \
                                                os.path.realpath(starting_dir)))



     

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
"""A File for splitting"""

import sys
import re
import os
import logging
pjoin = os.path.join

logger = logging.getLogger('madgraph.stdout')

class FOAnalyseCardError(Exception):
    pass

class FOAnalyseCard(dict):
    """A simple handler for the fixed-order analyse card """

    string_vars = ['fo_extralibs', 'fo_extrapaths', 'fo_includepaths', 
                   'fo_analyse', 'fo_analysis_format', 'fo_lhe_min_weight',
                   'fo_lhe_weight_ratio',
                   'fo_lhe_postprocessing']

    
    def __init__(self, card=None, testing=False):
        """ if testing, card is the content"""
        self.testing = testing
        dict.__init__(self)
        self.keylist = self.keys()
            
        if card:
            self.read_card(card)

    
    def read_card(self, card_path):
        """read the FO_analyse_card, if testing card_path is the content"""
        fo_analysis_formats = ['topdrawer','hwu','root','none', 'lhe']
        if not self.testing:
            content = open(card_path).read()
        else:
            content = card_path
        lines = [l for l in content.split('\n') \
                    if '=' in l and not l.startswith('#')] 
        for l in lines:
            args =  l.split('#')[0].split('=')
            key = args[0].strip().lower()
            value = args[1].strip()
            if key in self.string_vars:
                # special treatment for libs: remove lib and .a 
                # (i.e. libfastjet.a -> fastjet)
                if key == 'fo_extralibs':
                    value = value.replace('lib', '').replace('.a', '')
                elif key == 'fo_analysis_format' and value.lower() not in fo_analysis_formats:
                    raise FOAnalyseCardError('Unknown FO_ANALYSIS_FORMAT: %s' % value)
                if value.lower() == 'none':
                    self[key] = ''
                else:
                    self[key] = value
            else:
                raise FOAnalyseCardError('Unknown entry: %s = %s' % (key, value))
            self.keylist.append(key)


    def write_card(self, card_path):
        """write the parsed FO_analyse.dat (to be included in the Makefile) 
        in side card_path.
        if self.testing, the function returns its content"""

        if 'fo_analysis_format' in self and self['fo_analysis_format'].lower() in ['lhe','none']:
            if self['fo_analyse']:
                logger.warning('FO_ANALYSE parameter of the FO_analyse card should be empty for this analysis format. Removing this information.')
                self['fo_analyse'] = ''

        lines = []
        to_add = ''
        for key in self.keylist:
            value = self[key].lower()
            if key in self.string_vars:
                if key == 'fo_analysis_format':
                    if value == 'topdrawer':
                        to_add = 'dbook.o open_output_files_dummy.o HwU_dummy.o'
                    elif value == 'hwu':
                        to_add = 'HwU.o open_output_files_dummy.o'
                    elif value == 'root':
                        to_add = 'rbook_fe8.o rbook_be8.o HwU_dummy.o'
                    elif value == 'lhe':
                        to_add = 'analysis_lhe.o open_output_files_dummy.o write_event.o'
                    else:
                        to_add = 'analysis_dummy.o dbook.o open_output_files_dummy.o HwU_dummy.o'
                        


        for key in self.keylist:
            value = self[key]
            if key in self.string_vars:
                if key == 'fo_extrapaths':
                    # add the -L flag
                    line = '%s=%s' % (key.upper(), 
                            ' '.join(['-Wl,-rpath,' + path for path in value.split()])+' '+' '.join(['-L' + path for path in value.split()]))
                elif key == 'fo_includepaths':
                    # add the -I flag
                    line = '%s=%s' % (key.upper(), 
                            ' '.join(['-I' + path for path in value.split()]))
                elif key == 'fo_extralibs':
                    # add the -l flag
                    line = '%s=%s' % (key.upper(), 
                            ' '.join(['-l' + lib for lib in value.split()]))
                elif key == 'fo_analyse':
                    line = '%s=%s '% (key.upper(), value)
                    line = line + to_add
                else:
                    line = ''
                lines.append(line)
            else:
                raise FOAnalyseCardError('Unknown key: %s = %s' % (key, value))

        if self.testing:
            return ('\n'.join(lines) + '\n')
        else:
            open(card_path, 'w').write(('\n'.join(lines) + '\n'))


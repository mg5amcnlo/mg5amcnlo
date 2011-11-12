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
pjoin = os.path.join

#dict


class split_banner:
    

    pat_begin=re.compile('<(?P<name>\w*)>')
    pat_end=re.compile('</(?P<name>\w*)>')

    tag_to_file={'slha':'param_card.dat',
      'MGRunCard':'run_card.dat',
      'MGPythiaCard':'pythia_card.dat',
      'MGPGSCard' : 'pgs_card.dat',
      'MGDelphesCard':'delphes_card.dat',      
      'MGDelphesTrigger':'delphes_trigger.dat'
      }

    def __init__(self, banner_path, PROC_dir):

        self.pos=banner_path
        self.outpath = pjoin(PROC_dir, 'Cards')

    def split(self):
        self.file=open(self.pos,'r')

        for line in self.file:
            if self.pat_begin.search(line):
                tag = self.pat_begin.search(line).group('name')
                if name in self.tag_to_file:
                    filepath = os.path.join(pjoin(self.outpath, self.tag_to_file[tag]))
                    self.write_card(filepath)

    def write_card(self,pos):
        ff=open(pos,'w')
        for line in self.file:
            if self.pat_end.search(line):
                ff.close()
                return
            else:
                ff.writelines(line)



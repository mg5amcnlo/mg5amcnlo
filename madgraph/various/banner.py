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

try:
    import madgraph.various.misc as misc
    import madgraph.iolibs.file_writers as file_writers
    from madgraph import MG5DIR
    MADEVENT = False
except:
    MADEVENT = True
    import internal.file_writers as file_writers
    MEDIR = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
    MEDIR = os.path.split(MEDIR)[0]

import logging

logger = logging.getLogger('madevent.cards')

#dict
class Banner(dict):
    """ """
    
    def __init__(self, banner_path=None):
        """ """
        dict.__init__(self)
        
        #Look at the version
        if MADEVENT:
            self['mgversion'] = '#%s\n' % open(pjoin(MEDIR, 'MGMEVersion.txt')).read()
        else:
            info = misc.get_pkg_info()
            self['mgversion'] = info['version']+'\n'
            
        if banner_path:
            self.read_banner(banner_path)

    ############################################################################
    #  READ BANNER
    ############################################################################
    pat_begin=re.compile('<(?P<name>\w*)>')
    pat_end=re.compile('</(?P<name>\w*)>')

    tag_to_file={'slha':'param_card.dat',
      'mgruncard':'run_card.dat',
      'mgpythiacard':'pythia_card.dat',
      'mgpgscard' : 'pgs_card.dat',
      'mgdelphescard':'delphes_card.dat',      
      'mgdelphestrigger':'delphes_trigger.dat',
      'mg5proccard':'proc_card_mg5.dat',
      'mgproccard': 'proc_card.dat',
      }
    
    def read_banner(self, input_path):
        """read a banner"""

        text = ''
        store = False
        for line in open(input_path):
            if self.pat_begin.search(line):
                tag = self.pat_begin.search(line).group('name').lower()
                if tag in self.tag_to_file:
                    store = True
                    continue
            if store and self.pat_end.search(line):
                if tag == self.pat_end.search(line).group('name').lower():
                    self[tag] = text
                    text = ''
                    store = False
            if store:
                text += line

                
    def load_basic(self, medir):
        """ Load the proc_card /param_card and run_card """
        
        self.add(pjoin(medir,'Cards', 'param_card.dat'))
        self.add(pjoin(medir,'Cards', 'run_card.dat'))
        if os.path.exists(pjoin(medir, 'SubProcesses', 'procdef_mg5.dat')):
            self.add(pjoin(medir,'SubProcesses', 'procdef_mg5.dat'))
            self.add(pjoin(medir,'Cards', 'proc_card_mg5.dat'))
        else:
            self.add(pjoin(medir,'Cards', 'proc_card.dat'))
    
    
    def change_seed(self, seed):
        """Change the seed value in the banner"""
        #      0       = iseed
        p = re.compile(r'''^\s*\d+\s*=\s*iseed''', re.M)
        new_seed_str = " %s = iseed" % seed
        self['mgruncard'] = p.sub(new_seed_str, self['mgruncard'])
    
    def add_generation_info(self, cross, nb_event):
        """add info on MGGeneration"""
        
        text = """
#  Number of Events        :       %s
#  Integrated weight (pb)  :       %s
""" % (nb_event, cross)
        self['MGGenerationInfo'] = text
    
    ############################################################################
    #  SPLIT BANNER
    ############################################################################
    def split(self, me_dir, proc_card=True):
        """write the banner in the Cards directory.
        proc_card argument is present to avoid the overwrite of proc_card 
        information"""

        for tag, text in self.items():
            if tag == 'mgversion':
                continue
            if not proc_card and tag in ['mg5proccard','mgproccard']:
                continue
            ff = open(pjoin(me_dir, 'Cards', self.tag_to_file[tag]), 'w')
            ff.write(text)
            ff.close()

    ############################################################################
    #  WRITE BANNER
    ############################################################################
    def write(self, output_path):
        """write the banner"""
        
        ff = open(output_path, 'w')
        if MADEVENT:
            ff.write(open(pjoin(MEDIR, 'Source', 'banner_header.txt')).read())
        else:
            ff.write(open(pjoin(MG5DIR,'Template', 'Source', 'banner_header.txt')).read())
        for tag, text in self.items():
            ff.write('<%(tag)s>\n%(text)s\n</%(tag)s>\n' % \
                     {'tag':tag, 'text':text})
        ff.write('</header>\n')    
        ff.write('</LesHouchesEvents>\n')
            
    ############################################################################
    # BANNER
    ############################################################################
    def add(self, path, tag=None):
        """Add the content of the file to the banner"""
        
        if not tag:
            card_name = os.path.basename(path)
            if 'param_card' in card_name:
                tag = 'slha'
            elif 'run_card' in card_name:
                tag = 'MGRunCard'
            elif 'pythia_card' in card_name:
                tag = 'MGPythiaCard'
            elif 'pgs_card' in card_name:
                tag = 'MGPGSCard'
            elif 'delphes_card' in card_name:
                tag = 'MGDelphesCard'
            elif 'delphes_trigger' in card_name:
                tag = 'MGDelphesTrigger'
            elif 'proc_card_mg5' in card_name:
                tag = 'MG5ProcCard'
            elif 'proc_card' in card_name:
                tag = 'MGProcCard'
            elif 'procdef_mg5' in card_name:
                tag = 'MGProcCard'
            else:
                raise Exception, 'Impossible to know the type of the card'

            self[tag.lower()] = open(path).read()





def split_banner(banner_path, me_dir, proc_card=True):
    """a simple way to split a banner"""
    
    banner = Banner(banner_path)
    banner.split(me_dir, proc_card)
    
def recover_banner(results_object, level):
    """as input we receive a gen_crossxhtml.AllResults object.
       This define the current banner and load it
    """
    try:  
        run = results_object.current['run_name']    
        tag = results_object.current['tag'] 
    except Exception:
        return Banner()                                  
    path = results_object.path
    banner_path = pjoin(path,'Events',run,'%s_%s_banner.txt' % (run, tag))
    
    if not os.path.exists(banner_path):
        # security if the banner was remove (or program canceled before created it)
        return Banner()  
    
    banner = Banner(banner_path)
    
    
    
    if level == 'pythia':
        if 'mgpythiacard' in banner:
            del banner['mgpythiacard']
    if level in ['pythia','pgs','delphes']:
        for tag in ['mgpgscard', 'mgdelphescard', 'mgdelphestrigger']:
            if tag in banner:
                del banner[tag]
    return banner
    

class RunCard(dict):
    """A class object for the run_card"""

    def __init__(self, run_card):
        """ """

        for line in file(run_card,'r'):
            line = line.split('#')[0]
            line = line.split('!')[0]
            line = line.split('=')
            if len(line) != 2:
                continue
            self[line[1].strip()] = line[0].replace('\'','').strip()

    def get_default(self, name, default, log_level):
        """return self[name] if exist otherwise default. log control if we 
        put a warning or not if we use the default value"""

        try:
            return self[name]
        except KeyError:
            if log_level:
                logger.log(log_level, 'run_card missed argument %s. Takes default: %s'
                                   % (name, default))
                self[name] = default
                return default
        
    @staticmethod
    def format(format, value):
        """format the value"""
        
        if format == 'bool':
            if str(value) in ['1','T','.true.','True']:
                return '.true.'
            else:
                return '.false.'
            
        elif format == 'int':
            return str(int(value))
        
        elif format == 'float':
            if isinstance(value, str):
                value = value.replace('d','e')
            return ('%.10e' % float(value)).replace('e','d')
        
        elif format == 'str':
            return "'%s'" % value
    

        
    def write(self, output_file, template):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""
        
        text = ""
        for line in file(template,'r'):
            nline = line.split('#')[0]
            nline = nline.split('!')[0]
            comment = line[len(nline):]
            nline = nline.split('=')
            if len(nline) != 2:
                text += line
            else:
                text += '  %s\t= %s %s' % (self[nline[1].strip()],nline[1], comment)        
        
        fsock = open(output_file,'w')
        fsock.write(text)
        fsock.close()


    def write_include_file(self, output_path):
        """writing the run_card.inc file""" 
        
        self.fsock = file_writers.FortranWriter(output_path)    
################################################################################
#      Writing the lines corresponding to the cuts
################################################################################
    
        self.add_line('maxjetflavor', 'int', 4)
        self.add_line('auto_ptj_mjj', 'bool', True)
        self.add_line('cut_decays', 'bool', True)
        # minimum pt
        self.add_line('ptj', 'float', 20)
        self.add_line('ptb', 'float', 20)
        self.add_line('pta', 'float', 20)
        self.add_line('ptl', 'float', 20)
        self.add_line('misset', 'float', 0)
        self.add_line('ptonium', 'float', 0.0)
        # maximal pt
        self.add_line('ptjmax', 'float', -1)
        self.add_line('ptbmax', 'float', -1)
        self.add_line('ptamax', 'float', -1)
        self.add_line('ptlmax', 'float', -1)
        self.add_line('missetmax', 'float', -1)
        # maximal rapidity (absolute value)
        self.add_line('etaj', 'float', 4.0)
        self.add_line('etab', 'float', 4.0)
        self.add_line('etaa', 'float', 4.0)
        self.add_line('etal', 'float', 4.0)
        # minimal rapidity (absolute value)
        self.add_line('etajmin', 'float', 0.0)
        self.add_line('etabmin', 'float', 0.0)
        self.add_line('etaamin', 'float', 0.0)
        self.add_line('etalmin', 'float', 0.0)
        self.add_line('etaonium', 'float', 100.0)
        # Minimul E's
        self.add_line('ej', 'float', 0.0)
        self.add_line('eb', 'float', 0.0)
        self.add_line('ea', 'float', 0.0)
        self.add_line('el', 'float', 0.0)
        # Maximum E's
        self.add_line('ejmax', 'float', -1)
        self.add_line('ebmax', 'float', -1)
        self.add_line('eamax', 'float', -1)
        self.add_line('elmax', 'float', -1)     
        # minimum delta_r
        self.add_line('drjj', 'float', 0.4)     
        self.add_line('drbb', 'float', 0.4)     
        self.add_line('drll', 'float', 0.4)     
        self.add_line('draa', 'float', 0.4)     
        self.add_line('drbj', 'float', 0.4)     
        self.add_line('draj', 'float', 0.4)     
        self.add_line('drjl', 'float', 0.4)     
        self.add_line('drab', 'float', 0.4)     
        self.add_line('drbl', 'float', 0.4)     
        self.add_line('dral', 'float', 0.4)     
        # maximum delta_r
        self.add_line('drjjmax', 'float', -1)
        self.add_line('drbbmax', 'float', -1)
        self.add_line('drllmax', 'float', -1)
        self.add_line('draamax', 'float', -1)
        self.add_line('drbjmax', 'float', -1)
        self.add_line('drajmax', 'float', -1)
        self.add_line('drjlmax', 'float', -1)
        self.add_line('drabmax', 'float', -1)
        self.add_line('drblmax', 'float', -1)
        self.add_line('dralmax', 'float', -1)
        # minimum invariant mass for pairs
        self.add_line('mmjj', 'float', 0.0)
        self.add_line('mmbb', 'float', 0.0)
        self.add_line('mmaa', 'float', 0.0)
        self.add_line('mmll', 'float', 0.0)
        # maximum invariant mall for pairs
        self.add_line('mmjjmax', 'float', -1)
        self.add_line('mmbbmax', 'float', -1)
        self.add_line('mmaamax', 'float', -1)
        self.add_line('mmllmax', 'float', -1)
        #Min Maxi invariant mass for all leptons 
        self.add_line("mmnl", 'float', 0.0)
        self.add_line("mmnlmax", 'float', -1)
        #inclusive cuts
        self.add_line("xptj", 'float', 0.0)
        self.add_line("xptb", 'float', 0.0)
        self.add_line("xpta", 'float', 0.0)
        self.add_line("xptl", 'float', 0.0)
        self.add_line("xmtcentral", 'float', 0.0, fortran_name='xmtc', log=10)
        # WBT cuts
        self.add_line("xetamin", 'float', 0.0)
        self.add_line("deltaeta", 'float', 0.0)
        # Jet measure cuts 
        self.add_line("xqcut", 'float', 0.0)
        self.add_line("d", 'float', 1.0, log=10)
        # Set min pt of one heavy particle 
        self.add_line("ptheavy", 'float', 0.0)
        # Pt of pairs of leptons (CHARGED AND NEUTRALS)
        self.add_line("ptllmin", "float", 0.0)
        self.add_line("ptllmax", "float", -1)
        # Check   the pt's of the jets sorted by pt
        self.add_line("ptj1min", "float", 0.0)
        self.add_line("ptj1max", "float", -1)
        self.add_line("ptj2min", "float", 0.0)
        self.add_line("ptj2max", "float", -1)
        self.add_line("ptj3min", "float", 0.0)
        self.add_line("ptj3max", "float", -1)
        self.add_line("ptj4min", "float", 0.0)
        self.add_line("ptj4max", "float", -1)
        self.add_line("cutuse", "float", 0.0)
        # Check   the pt's of leptons sorted by pt
        self.add_line("ptl1min", "float", 0.0)
        self.add_line("ptl1max", "float", -1)
        self.add_line("ptl2min", "float", 0.0)
        self.add_line("ptl2max", "float", -1)
        self.add_line("ptl3min", "float", 0.0)
        self.add_line("ptl3max", "float", -1)
        self.add_line("ptl4min", "float", 0.0)
        self.add_line("ptl4max", "float", -1)
        # Check  Ht
        self.add_line("ht2min", 'float', 0.0)
        self.add_line("ht3min", 'float', 0.0)
        self.add_line("ht4min", 'float', 0.0)
        self.add_line("ht2max", 'float', -1)
        self.add_line("ht3max", 'float', -1)
        self.add_line("ht4max", 'float', -1)        
        self.add_line("htjmin", 'float', 0.0)
        self.add_line("htjmax", 'float', -1)        
        self.add_line("ihtmin", 'float', 0.0)
        self.add_line("ihtmax", 'float', -1)

################################################################################
#      Writing the lines corresponding to anything but cuts
################################################################################
        # seed
        self.add_line("gridpack","bool", False)
        self.add_line("gridrun",'bool', False, log=10)
        if str(self['gridrun']) in ['1','T','.true','True'] and \
           str(self['gridpack']) in ['1','T','.true','True']:
            self.add_line('gseed', 'int', 0, fortran_name='iseed')
        else:
            self.add_line('iseed', 'int', 0, fortran_name='iseed')
        # Renormalizrion and factorization scales
        self.add_line('fixed_ren_scale', 'bool', True)
        self.add_line('fixed_fac_scale', 'bool', True)
        self.add_line('scale', 'float', 'float', 91.188)
        self.add_line('dsqrt_q2fact1','float', 91.188, fortran_name='sf1')
        self.add_line('dsqrt_q2fact2', 'float', 91.188, fortran_name='sf2')
        self.add_line('scalefact', 'float', 1.0)
        self.add_line('fixed_couplings', 'bool', True, log=10)
        self.add_line('ickkw', 'int', 0)
        self.add_line('chcluster', 'bool', False)
        self.add_line('ktscheme', 'int', 1)
        if int(self['ickkw'])>0:
            self.add_line('alpsfact', 'float', 1.0)
            self.add_line('pdfwgt', 'bool', True)
        if int(self['ickkw'])==2:
            self.add_line('highestmult','int', 0, fortran_name='nhmult')
            self.add_line('issgridfile','str','issudgrid.dat')
        # Collider energy and type
        self.add_line('lpp1', 'int', 1, fortran_name='lpp(1)')
        self.add_line('lpp2', 'int', 1, fortran_name='lpp(2)')
        self.add_line('ebeam1', 'float', 7000, fortran_name='ebeam(1)')
        self.add_line('ebeam2', 'float', 7000, fortran_name='ebeam(2)')
        # Beam polarization
        self.add_line('polbeam1', 'float', 0.0, fortran_name='pb1')
        self.add_line('polbeam2', 'float', 0.0, fortran_name='pb2')
        # BW cutoff (M+/-bwcutoff*Gamma)
        self.add_line('bwcutoff', 'float', 15.0)
        #  Collider pdf
        self.add_line('pdlabel','str','cteq6l1')
        if self['pdlabel'] == 'lhapdf':
            self.add_line('lhaid', 'int', 10042)
        else:
            self.add_line('lhaid', 'int', 10042, log=10)
        
        self.fsock.close()



        
    def add_line(self, card_name, type, default, log=30, fortran_name=None):
        """get the line for the .inc file""" 
         
        value = self.get_default(card_name, default, log)
        if not fortran_name:
            fortran_name = card_name
        self.fsock.writelines(' %s = %s \n' % (fortran_name, self.format(type, value)))





 


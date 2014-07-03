################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
"""A File for splitting"""

import sys
import re
import os

pjoin = os.path.join

try:
    import madgraph.various.misc as misc
    import madgraph.iolibs.file_writers as file_writers
    import madgraph.iolibs.files as files 
    import models.check_param_card as param_card_reader
    from madgraph import MG5DIR
    MADEVENT = False
except ImportError:
    MADEVENT = True
    import internal.file_writers as file_writers
    import internal.files as files
    import internal.check_param_card as param_card_reader
    import internal.misc as misc
    
    MEDIR = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
    MEDIR = os.path.split(MEDIR)[0]

import logging

logger = logging.getLogger('madevent.cards')

#dict
class Banner(dict):
    """ """

    ordered_items = ['mgversion', 'mg5proccard', 'mgproccard', 'mgruncard',
                     'slha', 'mggenerationinfo', 'mgpythiacard', 'mgpgscard',
                     'mgdelphescard', 'mgdelphestrigger','mgshowercard','run_settings']

    capitalized_items = {
            'mgversion': 'MGVersion',
            'mg5proccard': 'MG5ProcCard',
            'mgproccard': 'MGProcCard',
            'mgruncard': 'MGRunCard',
            'mggenerationinfo': 'MGGenerationInfo',
            'mgpythiacard': 'MGPythiaCard',
            'mgpgscard': 'MGPGSCard',
            'mgdelphescard': 'MGDelphesCard',
            'mgdelphestrigger': 'MGDelphesTrigger',
            'mgshowercard': 'MGShowerCard' }
    
    def __init__(self, banner_path=None):
        """ """
        if isinstance(banner_path, Banner):
            return dict.__init__(self, banner_path)     
        else:
            dict.__init__(self)
        
        #Look at the version
        if MADEVENT:
            self['mgversion'] = '#%s\n' % open(pjoin(MEDIR, 'MGMEVersion.txt')).read()
        else:
            info = misc.get_pkg_info()
            self['mgversion'] = info['version']+'\n'
        
        self.lhe_version = None
        

            
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
      'init': '',
      'mggenerationinfo':'',
      'scalesfunctionalform':'',
      'montecarlomasses':'',
      'initrwgt':'',
      'madspin':'madspin_card.dat',
      'mgshowercard':'shower_card.dat',
      'run_settings':''
      }
    
    def read_banner(self, input_path):
        """read a banner"""

        if isinstance(input_path, str):
            input_path = open(input_path)

        text = ''
        store = False
        for line in input_path:
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
                if line.endswith('\n'):
                    text += line
                else:
                    text += '%s%s' % (line, '\n')
                
            #reaching end of the banner in a event file avoid to read full file 
            if "</init>" in line:
                break
            elif "<event>" in line:
                break
    
    def change_lhe_version(self, version):
        """change the lhe version associate to the banner"""
    
        version = float(version)
        if version < 3:
            version = 1
        elif version > 3:
            raise Exception, "Not Supported version"
        self.lhe_version = version
    
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
            if not self.tag_to_file[tag]:
                continue
            ff = open(pjoin(me_dir, 'Cards', self.tag_to_file[tag]), 'w')
            ff.write(text)
            ff.close()


    ############################################################################
    #  WRITE BANNER
    ############################################################################
    def check_pid(self, pid2label):
        """special routine removing width/mass of particles not present in the model
        This is usefull in case of loop model card, when we want to use the non
        loop model."""
        
        if not hasattr(self, 'param_card'):
            self.charge_card('slha')
            
        for tag in ['mass', 'decay']:
            block = self.param_card.get(tag)
            for data in block:
                pid = data.lhacode[0]
                if pid not in pid2label.keys(): 
                    block.remove((pid,))


    ############################################################################
    #  WRITE BANNER
    ############################################################################
    def write(self, output_path, close_tag=True, exclude=[]):
        """write the banner"""
        
        if isinstance(output_path, str):
            ff = open(output_path, 'w')
        else:
            ff = output_path
            
        if MADEVENT:
            header = open(pjoin(MEDIR, 'Source', 'banner_header.txt')).read()
        else:
            header = open(pjoin(MG5DIR,'Template', 'LO', 'Source', 'banner_header.txt')).read()
    
        if not self.lhe_version:
            self.lhe_version = self.get('run_card', 'lhe_version', default=1.0)
            if float(self.lhe_version) < 3:
                self.lhe_version = 1.0
        
        ff.write(header % { 'version':float(self.lhe_version)})


        for tag in [t for t in self.ordered_items if t in self.keys()]:
            if tag in exclude: 
                continue
            capitalized_tag = self.capitalized_items[tag] if tag in self.capitalized_items else tag
            ff.write('<%(tag)s>\n%(text)s\n</%(tag)s>\n' % \
                     {'tag':capitalized_tag, 'text':self[tag].strip()})
        for tag in [t for t in self.keys() if t not in self.ordered_items]:
            if tag in ['init'] or tag in exclude:
                continue
            capitalized_tag = self.capitalized_items[tag] if tag in self.capitalized_items else tag
            ff.write('<%(tag)s>\n%(text)s\n</%(tag)s>\n' % \
                     {'tag':capitalized_tag, 'text':self[tag].strip()})
        
        if not '/header' in exclude:
            ff.write('</header>\n')    

        if 'init' in self and not 'init' in exclude:
            text = self['init']
            ff.write('<%(tag)s>\n%(text)s\n</%(tag)s>\n' % \
                     {'tag':'init', 'text':text.strip()})  
        if close_tag:          
            ff.write('</LesHouchesEvents>\n')
        return ff
        
        
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
            elif 'shower_card' in card_name:
                tag = 'MGShowerCard'
            elif 'madspin_card' in card_name:
                tag = 'madspin'
            elif 'FO_analyse_card' in card_name:
                tag = 'foanalyse'
            else:
                raise Exception, 'Impossible to know the type of the card'

            self.add_text(tag.lower(), open(path).read())

    def add_text(self, tag, text):
        """Add the content of the file to the banner"""
        
        self[tag.lower()] = text
    
    
    def charge_card(self, tag):
        """Build the python object associated to the card"""
        
        if tag == 'param_card':
            tag = 'slha'
        elif tag == 'run_card':
            tag = 'mgruncard' 
        elif tag == 'proc_card':
            tag = 'mg5proccard' 
        elif tag == 'shower_card':
            tag = 'mgshowercard'
        elif tag == 'FO_analyse_card':
            tag = 'foanalyse'

        assert tag in ['slha', 'mgruncard', 'mg5proccard', 'mgshowercard', 'foanalyse'], 'invalid card %s' % tag
        
        if tag == 'slha':
            param_card = self[tag].split('\n')
            self.param_card = param_card_reader.ParamCard(param_card)
            return self.param_card
        elif tag == 'mgruncard':
            run_card = self[tag].split('\n') 
            if 'parton_shower' in self[tag]:
                self.run_card = RunCardNLO(run_card)
            else:
                self.run_card = RunCard(run_card)
            return self.run_card
        elif tag == 'mg5proccard':
            proc_card = self[tag].split('\n')
            self.proc_card = ProcCard(proc_card)
            return self.proc_card
        elif tag =='mgshowercard':
            shower_content = self[tag] 
            if MADEVENT:
                import internal.shower_card as shower_card
            else:
                import madgraph.various.shower_card as shower_card
            self.shower_card = shower_card.ShowerCard(shower_content, True)
            # set testing to false (testing = true allow to init using 
            #  the card content instead of the card path"
            self.shower_card.testing = False
            return self.shower_card
        elif tag =='foanalyse':
            analyse_content = self[tag] 
            if MADEVENT:
                import internal.FO_analyse_card as FO_analyse_card
            else:
                import madgraph.various.FO_analyse_card as FO_analyse_card
            # set testing to false (testing = true allow to init using 
            #  the card content instead of the card path"
            self.FOanalyse_card = FO_analyse_card.FOAnalyseCard(analyse_content, True)
            self.FOanalyse_card.testing = False
            return self.FOanalyse_card
        

    def get_detail(self, tag, *arg, **opt):
        """return a specific """
                
        if tag == 'param_card':
            tag = 'slha'
            attr_tag = 'param_card'
        elif tag == 'run_card':
            tag = 'mgruncard' 
            attr_tag = 'run_card'
        elif tag == 'proc_card':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
        elif tag == 'model':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
            arg = ('model',)
        elif tag == 'generate':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
            arg = ('generate',)
        elif tag == 'shower_card':
            tag = 'mgshowercard'
            attr_tag = 'shower_card'
        assert tag in ['slha', 'mgruncard', 'mg5proccard', 'shower_card'], 'not recognized'
        
        if not hasattr(self, attr_tag):
            self.charge_card(attr_tag) 
        
        card = getattr(self, attr_tag)
        if len(arg) == 1:
            if tag == 'mg5proccard':
                try:
                    return card.info[arg[0]]
                except KeyError, error:
                    if opt['default']:
                        return opt['default']
                    else:
                        raise
            try:
                return card[arg[0]]
            except KeyError:
                if opt['default']:
                    return opt['default']
                else:
                    raise                
        elif len(arg) == 2 and tag == 'slha':
            try:
                return card[arg[0]].get(arg[1:])
            except KeyError:
                if opt['default']:
                    return opt['default']
                else:
                    raise  
        else:
            raise Exception, "Unknow command"
    
    #convenient alias
    get = get_detail
    
    def set(self, card, *args):
        """modify one of the cards"""

        if tag == 'param_card':
            tag = 'slha'
            attr_tag = 'param_card'
        elif tag == 'run_card':
            tag = 'mgruncard' 
            attr_tag = 'run_card'
        elif tag == 'proc_card':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
        elif tag == 'model':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
            arg = ('model',)
        elif tag == 'generate':
            tag = 'mg5proccard' 
            attr_tag = 'proc_card'
            arg = ('generate',)
        elif tag == 'shower_card':
            tag = 'mgshowercard'
            attr_tag = 'shower_card'
        assert tag in ['slha', 'mgruncard', 'mg5proccard', 'shower_card'], 'not recognized'
        
        if not hasattr(self, attr_tag):
            self.charge_card(attr_tag) 
            
        card = getattr(self, attr_tag)
        if len(args) ==2:
            if tag == 'mg5proccard':
                card.info[args[0]] = args[-1]
            else:
                card[args[0]] = args[1]
        else:
            card[args[:-1]] = args[-1]
        
    

    def add_to_file(self, path, seed=None):
        """Add the banner to a file and change the associate seed in the banner"""

        if seed is not None:
            self.set("run_card", "iseed", seed)
            
        ff = self.write("%s.tmp" % path, close_tag=False,
                        exclude=['MGGenerationInfo', '/header', 'init'])
        ff.write("## END BANNER##\n")
        if self.lhe_version >= 3:
        #add the original content
            [ff.write(line) if not line.startswith("<generator name='MadGraph5_aMC@NLO'")
                        else ff.write("<generator name='MadGraph5_aMC@NLO' version='%s'>" % self['mgversion'][:-1])
                        for line in open(path)]
        else:
            [ff.write(line) for line in open(path)]
        ff.write("</LesHouchesEvents>\n")
        ff.close()
        files.mv("%s.tmp" % path, path)


        
def split_banner(banner_path, me_dir, proc_card=True):
    """a simple way to split a banner"""
    
    banner = Banner(banner_path)
    banner.split(me_dir, proc_card)
    
def recover_banner(results_object, level, run=None, tag=None):
    """as input we receive a gen_crossxhtml.AllResults object.
       This define the current banner and load it
    """
    
    if not run:
        try:    
            tag = results_object.current['tag'] 
        except Exception:
            return Banner()
    if not tag:
        try:    
            tag = results_object[run].tags[-1] 
        except Exception,error:
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

    #list of paramater which are allowed BUT not present in the _default file.
    hidden_param = ['lhaid', 'gridrun', 'fixed_couplings']
    true = ['true', 'True','.true.','T', True, 1,'TRUE']

    def __init__(self, run_card):
        """ """
        
        if isinstance(run_card, str):
            run_card = file(run_card,'r')
        else:
            pass # use in banner loading

        for line in run_card:
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
    

        
    def write(self, output_file, template=None):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""
        
        if not template:
            template = output_file
        
        text = ""
        for line in file(template,'r'):                  
            nline = line.split('#')[0]
            nline = nline.split('!')[0]
            comment = line[len(nline):]
            nline = nline.split('=')
            if len(nline) != 2:
                text += line
            elif nline[1].strip() in self:
                text += '  %s\t= %s %s' % (self[nline[1].strip()],nline[1], comment)        
            else:
                logger.info('Adding missing parameter %s to current run_card (with default value)' % nline[1].strip())
                text += line 
        
        for param in self.hidden_param:
            if param in self:
                text += '  %s\t= %s \n' % (self[param],param) 

        fsock = open(output_file,'w')
        fsock.write(text)
        fsock.close()


    def write_include_file(self, output_path):
        """writing the run_card.inc file""" 
        
        self.fsock = file_writers.FortranWriter(output_path)    
################################################################################
#      Writing the lines corresponding to the cuts
################################################################################
        # Frixione photon isolation
        self.add_line('ptgmin', 'float', 0.0)
        self.add_line('R0gamma', 'float', 0.4)
        self.add_line('xn', 'float', 1.0)
        self.add_line('epsgamma', 'float', 1.0)
        self.add_line('isoEM', 'bool', True)
        # Cut that need to be deactivated in presence of isolation
        if 'ptgmin' in self and float(self['ptgmin'])>0:
            if float(self['pta']) > 0:
                logger.warning('pta cut discarded since photon isolation is used')
                self['pta'] = '0'
            if float(self['draj']) > 0:
                logger.warning('draj cut discarded since photon isolation is used')
                self['draj'] = '0' 
    
        self.add_line('maxjetflavor', 'int', 4)
        if int(self['maxjetflavor']) > 6:
            raise Exception, 'maxjetflavor should be lower than 5! (6 is partly supported)'
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
        # kt_ durham
        self.add_line('ktdurham', 'float', -1, fortran_name='kt_durham')
        self.add_line('dparameter', 'float', 0.4, fortran_name='d_parameter')


################################################################################
#      Writing the lines corresponding to anything but cuts
################################################################################
        # lhe output format
        self.add_line("lhe_version", "float", 2.0) #if not specify assume old standard
        # seed
        self.add_line("gridpack","bool", False)
        self.add_line("gridrun",'bool', False, log=10)
        if str(self['gridrun']) in ['1','T','.true','True'] and \
           str(self['gridpack']) in ['1','T','.true','True']:
            self.add_line('gseed', 'int', 0, fortran_name='iseed')
        else:
            self.add_line('iseed', 'int', 0, fortran_name='iseed')
        #number of events
        self.add_line('nevents', 'int', 10000)
        #self.add_line('gevents', 'int', 2000, log=10)
            
        # Renormalizrion and factorization scales
        self.add_line('fixed_ren_scale', 'bool', True)
        self.add_line('fixed_fac_scale', 'bool', True)
        self.add_line('scale', 'float', 'float', 91.188)
        self.add_line('dsqrt_q2fact1','float', 91.188, fortran_name='sf1')
        self.add_line('dsqrt_q2fact2', 'float', 91.188, fortran_name='sf2')
        
        self.add_line('use_syst', 'bool', False)
        #if use_syst is True, some parameter are automatically fixed.
        if self['use_syst'] in self.true:
            value = self.format('float',self.get_default('scalefact', 1.0, 30))
            if value != self.format('float', 1.0):
                logger.warning('Since use_syst=T, We change the value of \'scalefact\' to 1')
                self['scalefact'] = 1.0
        self.add_line('scalefact', 'float', 1.0)
        
        self.add_line('fixed_couplings', 'bool', True, log=10)
        self.add_line('ickkw', 'int', 0)
        self.add_line('chcluster', 'bool', False)
        self.add_line('ktscheme', 'int', 1)
        self.add_line('asrwgtflavor', 'int', 5)
        
        #CKKW TREATMENT!
        if int(self['ickkw'])>0:
            #if use_syst is True, some parameter are automatically fixed.
            if self['use_syst'] in self.true:
                value = self.format('float',self.get_default('alpsfact', 1.0, 30))
                if value != self.format('float', 1.0):
                    logger.warning('Since use_syst=T, We change the value of \'alpsfact\' to 1')
                    self['alpsfact'] = 1.0
            if int(self['maxjetflavor']) == 6:
                raise Exception, 'maxjetflavor at 6 is NOT supported for matching!'
            self.add_line('alpsfact', 'float', 1.0)
            self.add_line('pdfwgt', 'bool', True)
            self.add_line('clusinfo', 'bool', False)
            # check that DRJJ and DRJL are set to 0 and MMJJ
            if self.format('float', self['drjj']) != self.format('float', 0.):
                logger.warning('Since icckw>0, We change the value of \'drjj\' to 0')
            if self.format('float', self['drjl']) != self.format('float', 0.):
                logger.warning('Since icckw>0, We change the value of \'drjl\' to 0')
            if self.format('bool', self['auto_ptj_mjj']) == '.false.':
                #ensure formatting
                mmjj = self['mmjj']
                if isinstance(mmjj,str):
                    mmjj = float(mmjj.replace('d','e'))
                xqcut = self['xqcut']
                if isinstance(xqcut,str):
                    xqcut = float(xqcut.replace('d','e'))
                 
                if mmjj > xqcut:
                    logger.warning('mmjj > xqcut (and auto_ptj_mjj = F). MMJJ set to 0')
                    self.add_line('mmjj','float',0)
                    
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


class RunCardNLO(RunCard):
    """A class object for the run_card for a (aMC@)NLO pocess"""

        
    def write_include_file(self, output_path):
        """writing the run_card.inc file""" 
        
        self.fsock = file_writers.FortranWriter(output_path)    
################################################################################
#      Writing the lines corresponding to the cuts
################################################################################
    
        self.add_line('maxjetflavor', 'int', 4)
        # minimum pt
        self.add_line('ptj', 'float', 20)
        self.add_line('etaj', 'float', -1.0)
        self.add_line('ptl', 'float', 20)
        self.add_line('etal', 'float', -1.0)
        # minimum delta_r
        self.add_line('drll', 'float', 0.4)     
        self.add_line('drll_sf', 'float', 0.4)     
        # minimum invariant mass for pairs
        self.add_line('mll', 'float', 0.0)
        self.add_line('mll_sf', 'float', 0.0)
        #inclusive cuts
        # Jet measure cuts 
        self.add_line("jetradius", 'float', 0.7, log=10)

################################################################################
#      Writing the lines corresponding to anything but cuts
################################################################################
        # seed
        self.add_line('iseed', 'int', 0)
        self.add_line('parton_shower', 'str', 'HERWIG6', fortran_name='shower_mc')
        self.add_line('nevents', 'int', 10000)
        self.add_line('event_norm', 'str', 'average', fortran_name='event_norm')
        # Renormalizrion and factorization scales
        self.add_line('fixed_ren_scale', 'bool', True)
        self.add_line('fixed_fac_scale', 'bool', True)
        self.add_line('fixed_QES_scale', 'bool', True)
        self.add_line('muR_ref_fixed', 'float', 91.188)
        self.add_line('muF1_ref_fixed','float', 91.188)
        self.add_line('muF2_ref_fixed', 'float', 91.188)
        self.add_line('QES_ref_fixed', 'float', 91.188)
        self.add_line('muR_over_ref', 'float', 1.0)
        self.add_line('muF1_over_ref', 'float', 1.0)
        self.add_line('muF2_over_ref', 'float', 1.0)
        self.add_line('QES_over_ref', 'float', 1.0)
        #reweight block
        self.add_line('reweight_scale', 'bool', True, fortran_name='do_rwgt_scale')
        self.add_line('rw_Rscale_up', 'float', 2.0)
        self.add_line('rw_Rscale_down', 'float', 0.5)
        self.add_line('rw_Fscale_up', 'float', 2.0)
        self.add_line('rw_Fscale_down', 'float', 0.5)
        self.add_line('reweight_PDF', 'bool', True, fortran_name='do_rwgt_pdf')
        self.add_line('PDF_set_min', 'int', 21101)
        self.add_line('PDF_set_max', 'int', 21140)
        # FxFx merging stuff
        self.add_line('ickkw', 'int', 0)
        # self.add_line('fixed_couplings', 'bool', True, log=10)
        self.add_line('jetalgo', 'float', 1.0)
        # Collider energy and type
        self.add_line('lpp1', 'int', 1, fortran_name='lpp(1)')
        self.add_line('lpp2', 'int', 1, fortran_name='lpp(2)')
        self.add_line('ebeam1', 'float', 4000, fortran_name='ebeam(1)')
        self.add_line('ebeam2', 'float', 4000, fortran_name='ebeam(2)')
        # BW cutoff (M+/-bwcutoff*Gamma)
        self.add_line('bwcutoff', 'float', 15.0)
        # Photon isolation
        self.add_line('ptgmin', 'float', 10.0)
        self.add_line('etagamma', 'float', -1.0)
        self.add_line('R0gamma', 'float', 0.4)
        self.add_line('xn', 'float', 1.0)
        self.add_line('epsgamma', 'float', 1.0)
        self.add_line('isoEM', 'bool', True)
        #  Collider pdf
        self.add_line('pdlabel','str','cteq6_m')
        if self['pdlabel'] == 'lhapdf':
            self.add_line('lhaid', 'int', 21100)
        else:
            self.add_line('lhaid', 'int', 21100, log=10)
        
        self.fsock.close()

class ProcCard(list):
    """Basic Proccard object"""
    
    history_header = \
        '#************************************************************\n' + \
        '#*                     MadGraph5_aMC@NLO                    *\n' + \
        '#*                                                          *\n' + \
        "#*                *                       *                 *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                    * * * * 5 * * * *                     *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                *                       *                 *\n" + \
        "#*                                                          *\n" + \
        "#*                                                          *\n" + \
        "%(info_line)s" +\
        "#*                                                          *\n" + \
        "#*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
        "#*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        '#*                                                          *\n' + \
        '#************************************************************\n' + \
        '#*                                                          *\n' + \
        '#*               Command File for MadGraph5_aMC@NLO         *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/mg5_aMC  filename                       *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
    
    
    
    
    def __init__(self, init=None):
        """ initialize a basic proc_card"""
        self.info = {'model': 'sm', 'generate':None,
                     'full_model_line':'import model sm'}
        list.__init__(self)
        if init:
            self.read(init)

            
    def read(self, init):
        """read the proc_card and save the information"""
        
        if isinstance(init, str): #path to file
            init = file(init, 'r')
        
        store_line = ''
        for line in init:
            line = line.strip()
            if line.endswith('\\'):
                store_line += line[:-1]
            else:
                self.append(store_line + line)
                store_line = ""
        if store_line:
            raise Exception, "WRONG CARD FORMAT"
    def move_to_last(self, cmd):
        """move an element to the last history."""
        for line in self[:]:
            if line.startswith(cmd):
                self.remove(line)
                list.append(self, line)
    
    def append(self, line):
        """"add a line in the proc_card perform automatically cleaning"""
        
        line = line.strip()
        cmds = line.split()
        if len(cmds) == 0:
            return
        
        list.append(self, line)
        
        # command type:
        cmd = cmds[0]
        
        if cmd == 'output':
            # Remove previous outputs from history
            self.clean(allow_for_removal = ['output'], keep_switch=True,
                           remove_bef_last='output')
        elif cmd == 'generate':
            # Remove previous generations from history
            self.clean(remove_bef_last='generate', keep_switch=True,
                     allow_for_removal= ['generate', 'add process', 'output'])
            self.info['generate'] = ' '.join(cmds[1:])
        elif cmd == 'add' and cmds[1] == 'process' and not self.info['generate']:
            self.info['generate'] = ' '.join(cmds[2:])
        elif cmd == 'import':
            if len(cmds) < 2:
                return
            if cmds[1].startswith('model'):
                self.info['full_model_line'] = line
                self.clean(remove_bef_last='import', keep_switch=True,
                        allow_for_removal=['generate', 'add process', 'add model', 'output'])
                if cmds[1] == 'model':
                    self.info['model'] = cmds[2]
                else:
                    self.info['model'] = None # not UFO model
            elif cmds[1] == 'proc_v4':
                #full cleaning
                self[:] = []
                

    def clean(self, to_keep=['set','add','load'],
                            remove_bef_last=None,
                            to_remove=['open','display','launch', 'check','history'],
                            allow_for_removal=None,
                            keep_switch=False):
        """Remove command in arguments from history.
        All command before the last occurrence of  'remove_bef_last'
        (including it) will be removed (but if another options tells the opposite).                
        'to_keep' is a set of line to always keep.
        'to_remove' is a set of line to always remove (don't care about remove_bef_ 
        status but keep_switch acts.).
        if 'allow_for_removal' is define only the command in that list can be 
        remove of the history for older command that remove_bef_lb1. all parameter
        present in to_remove are always remove even if they are not part of this 
        list.
        keep_switch force to keep the statement remove_bef_??? which changes starts
        the removal mode.
        """

        #check consistency
        if __debug__ and allow_for_removal:
            for arg in to_keep:
                assert arg not in allow_for_removal
            
    
        nline = -1
        removal = False
        #looping backward
        while nline > -len(self):
            switch  = False # set in True when removal pass in True

            #check if we need to pass in removal mode
            if not removal and remove_bef_last:
                    if self[nline].startswith(remove_bef_last):
                        removal = True
                        switch = True  

            # if this is the switch and is protected pass to the next element
            if switch and keep_switch:
                nline -= 1
                continue

            # remove command in to_remove (whatever the status of removal)
            if any([self[nline].startswith(arg) for arg in to_remove]):
                self.pop(nline)
                continue
            
            # Only if removal mode is active!
            if removal:
                if allow_for_removal:
                    # Only a subset of command can be removed
                    if any([self[nline].startswith(arg) 
                                                 for arg in allow_for_removal]):
                        self.pop(nline)
                        continue
                elif not any([self[nline].startswith(arg) for arg in to_keep]):
                    # All command have to be remove but protected
                    self.pop(nline)
                    continue
            
            # update the counter to pass to the next element
            nline -= 1
        
    def __getattr__(self, tag):
        if isinstance(tag, int):
            list.__getattr__(self, tag)
        else:
            return self.info[tag]
            
    def write(self, path):
        """write the proc_card to a given path"""
        
        fsock = open(path, 'w')
        fsock.write(self.history_header)
        for line in self:
            while len(line) > 70:
                sub, line = line[:70]+"\\" , line[70:] 
                fsock.write(sub+"\n")
            else:
                fsock.write(line+"\n")
                
            
                
            
        
    
            
            
        
        
        
        
        
        
        
    

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

from __future__ import division
import collections
import copy
import logging
import numbers
import os
import sys
import re
import math
import StringIO

pjoin = os.path.join

try:
    import madgraph
except ImportError:
    MADEVENT = True
    from internal import MadGraph5Error, InvalidCmd
    import internal.file_writers as file_writers
    import internal.files as files
    import internal.check_param_card as param_card_reader
    import internal.misc as misc
    MEDIR = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
    MEDIR = os.path.split(MEDIR)[0]
else:
    MADEVENT = False
    import madgraph.various.misc as misc
    import madgraph.iolibs.file_writers as file_writers
    import madgraph.iolibs.files as files 
    import models.check_param_card as param_card_reader
    from madgraph import MG5DIR, MadGraph5Error, InvalidCmd


logger = logging.getLogger('madevent.cards')

# A placeholder class to store unknown parameters with undecided format
class UnknownType(str):
    pass

#dict
class Banner(dict):
    """ """

    ordered_items = ['mgversion', 'mg5proccard', 'mgproccard', 'mgruncard',
                     'slha','initrwgt','mggenerationinfo', 'mgpythiacard', 'mgpgscard',
                     'mgdelphescard', 'mgdelphestrigger','mgshowercard',
                     'ma5card_parton','ma5card_hadron','run_settings']

    capitalized_items = {
            'mgversion': 'MGVersion',
            'mg5proccard': 'MG5ProcCard',
            'mgproccard': 'MGProcCard',
            'mgruncard': 'MGRunCard',
            'ma5card_parton' : 'MA5Card_parton',
            'ma5card_hadron' : 'MA5Card_hadron',            
            'mggenerationinfo': 'MGGenerationInfo',
            'mgpythiacard': 'MGPythiaCard',
            'mgpgscard': 'MGPGSCard',
            'mgdelphescard': 'MGDelphesCard',
            'mgdelphestrigger': 'MGDelphesTrigger',
            'mgshowercard': 'MGShowerCard' }
    
    forbid_cdata = ['initrwgt']
    
    def __init__(self, banner_path=None):
        """ """

        if isinstance(banner_path, Banner):
            dict.__init__(self, banner_path)
            self.lhe_version = banner_path.lhe_version
            return     
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
      'pythia8':'pythia8_card.dat',
      'ma5card_parton':'madanalysis5_parton_card.dat',
      'ma5card_hadron':'madanalysis5_hadron_card.dat',      
      'run_settings':''
      }
    
    def read_banner(self, input_path):
        """read a banner"""

        if isinstance(input_path, str):
            if input_path.find('\n') ==-1:
                input_path = open(input_path)
            else:
                def split_iter(string):
                    return (x.groups(0)[0] for x in re.finditer(r"([^\n]*\n)", string, re.DOTALL))
                input_path = split_iter(input_path)
                
        text = ''
        store = False
        for line in input_path:
            if self.pat_begin.search(line):
                if self.pat_begin.search(line).group('name').lower() in self.tag_to_file:
                    tag = self.pat_begin.search(line).group('name').lower()
                    store = True
                    continue
            if store and self.pat_end.search(line):
                if tag == self.pat_end.search(line).group('name').lower():
                    self[tag] = text
                    text = ''
                    store = False
            if store and not line.startswith(('<![CDATA[',']]>')):
                if line.endswith('\n'):
                    text += line
                else:
                    text += '%s%s' % (line, '\n')
                
            #reaching end of the banner in a event file avoid to read full file 
            if "</init>" in line:
                break
            elif "<event>" in line:
                break
    
    def __getattribute__(self, attr):
        """allow auto-build for the run_card/param_card/... """
        try:
            return super(Banner, self).__getattribute__(attr)
        except:
            if attr not in ['run_card', 'param_card', 'slha', 'mgruncard', 'mg5proccard', 'mgshowercard', 'foanalyse']:
                raise
            return self.charge_card(attr)


    
    def change_lhe_version(self, version):
        """change the lhe version associate to the banner"""
    
        version = float(version)
        if version < 3:
            version = 1
        elif version > 3:
            raise Exception, "Not Supported version"
        self.lhe_version = version
    
    def get_cross(self, witherror=False):
        """return the cross-section of the file"""

        if "init" not in self:
            raise Exception
        
        text = self["init"].split('\n')
        cross = 0
        error = 0
        for line in text:
            s = line.split()
            if len(s)==4:
                cross += float(s[0])
                if witherror:
                    error += float(s[1])**2
        if not witherror:
            return cross
        else:
            return cross, math.sqrt(error)
        

    def scale_init_cross(self, ratio):
        """modify the init information with the associate scale"""

        assert "init" in self
        
        all_lines = self["init"].split('\n')
        new_data = []
        new_data.append(all_lines[0])
        for i in range(1, len(all_lines)):
            line = all_lines[i]
            split = line.split()
            if len(split) == 4:
                xsec, xerr, xmax, pid = split 
            else:
                new_data += all_lines[i:]
                break
            pid = int(pid)
            
            line = "   %+13.7e %+13.7e %+13.7e %i" % \
                (ratio*float(xsec), ratio* float(xerr), ratio*float(xmax), pid)
            new_data.append(line)
        self['init'] = '\n'.join(new_data)
    
    def get_pdg_beam(self):
        """return the pdg of each beam"""
        
        assert "init" in self
        
        all_lines = self["init"].split('\n')
        pdg1,pdg2,_ = all_lines[0].split(None, 2)
        return int(pdg1), int(pdg2)
    
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

    def get_lha_strategy(self):
        """get the lha_strategy: how the weight have to be handle by the shower"""
        
        if not self["init"]:
            raise Exception, "No init block define"
        
        data = self["init"].split('\n')[0].split()
        if len(data) != 10:
            misc.sprint(len(data), self['init'])
            raise Exception, "init block has a wrong format"
        return int(float(data[-2]))
        
    def set_lha_strategy(self, value):
        """set the lha_strategy: how the weight have to be handle by the shower"""
        
        if not (-4 <= int(value) <= 4):
            raise Exception, "wrong value for lha_strategy", value
        if not self["init"]:
            raise Exception, "No init block define"
        
        all_lines = self["init"].split('\n')
        data = all_lines[0].split()
        if len(data) != 10:
            misc.sprint(len(data), self['init'])
            raise Exception, "init block has a wrong format"
        data[-2] = '%s' % value
        all_lines[0] = ' '.join(data)
        self['init'] = '\n'.join(all_lines)


    def modify_init_cross(self, cross):
        """modify the init information with the associate cross-section"""
        assert isinstance(cross, dict)
#        assert "all" in cross
        assert "init" in self
        
        cross = dict(cross)
        for key in cross.keys():
            if isinstance(key, str) and key.isdigit() and int(key) not in cross:
                cross[int(key)] = cross[key]
        
        
        all_lines = self["init"].split('\n')
        new_data = []
        new_data.append(all_lines[0])
        for i in range(1, len(all_lines)):
            line = all_lines[i]
            split = line.split()
            if len(split) == 4:
                xsec, xerr, xmax, pid = split 
            else:
                new_data += all_lines[i:]
                break
            if int(pid) not in cross:
                raise Exception
            pid = int(pid)
            if float(xsec):
                ratio = cross[pid]/float(xsec)
            else:
                ratio = 0
            line = "   %+13.7e %+13.7e %+13.7e %i" % \
                (float(cross[pid]), ratio* float(xerr), ratio*float(xmax), pid)
            new_data.append(line)
        self['init'] = '\n'.join(new_data)
                
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


        for tag in [t for t in self.ordered_items if t in self.keys()]+ \
            [t for t in self.keys() if t not in self.ordered_items]:
            if tag in ['init'] or tag in exclude: 
                continue
            capitalized_tag = self.capitalized_items[tag] if tag in self.capitalized_items else tag
            start_data, stop_data = '', ''
            if capitalized_tag not in self.forbid_cdata and \
                                          ('<' in self[tag] or '@' in self[tag]):
                start_data = '\n<![CDATA['
                stop_data = ']]>\n'
            ff.write('<%(tag)s>%(start_data)s\n%(text)s\n%(stop_data)s</%(tag)s>\n' % \
                     {'tag':capitalized_tag, 'text':self[tag].strip(),
                      'start_data': start_data, 'stop_data':stop_data})
        
        
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
            elif 'pythia8_card' in card_name or 'pythia8.cmd' in card_name:
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
            elif 'reweight_card' in card_name:
                tag='reweight_card'
            elif 'madanalysis5_parton_card' in card_name:
                tag='MA5Card_parton'
            elif 'madanalysis5_hadron_card' in card_name:
                tag='MA5Card_hadron'
            else:
                raise Exception, 'Impossible to know the type of the card'

            self.add_text(tag.lower(), open(path).read())

    def add_text(self, tag, text):
        """Add the content of the file to the banner"""

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
            self.run_card = RunCard(self[tag])
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
                
        if tag in ['param_card', 'param']:
            tag = 'slha'
            attr_tag = 'param_card'
        elif tag in ['run_card', 'run']:
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
        assert tag in ['slha', 'mgruncard', 'mg5proccard', 'shower_card'], '%s not recognized' % tag
        
        if not hasattr(self, attr_tag):
            self.charge_card(attr_tag) 

        card = getattr(self, attr_tag)
        if len(arg) == 0:
            return card
        elif len(arg) == 1:
            if tag == 'mg5proccard':
                try:
                    return card.get(arg[0])
                except KeyError, error:
                    if 'default' in opt:
                        return opt['default']
                    else:
                        raise
            try:
                return card[arg[0]]
            except KeyError:
                if 'default' in opt:
                    return opt['default']
                else:
                    raise                
        elif len(arg) == 2 and tag == 'slha':
            try:
                return card[arg[0]].get(arg[1:])
            except KeyError:
                if 'default' in opt:
                    return opt['default']
                else:
                    raise  
        elif len(arg) == 0:
            return card
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
        
    
    @misc.multiple_try()
    def add_to_file(self, path, seed=None, out=None):
        """Add the banner to a file and change the associate seed in the banner"""

        if seed is not None:
            self.set("run_card", "iseed", seed)
        
        if not out:
            path_out = "%s.tmp" % path
        else:
            path_out = out
        
        ff = self.write(path_out, close_tag=False,
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
        if out:
            os.remove(path)
        else:
            files.mv(path_out, path)


        
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
            _run = results_object.current['run_name']   
            _tag = results_object.current['tag'] 
        except Exception:
            return Banner()
    else:
        _run = run
    if not tag:
        try:    
            _tag = results_object[run].tags[-1] 
        except Exception,error:
            return Banner()      
    else:
        _tag = tag
                                          
    path = results_object.path
    banner_path = pjoin(path,'Events',run,'%s_%s_banner.txt' % (run, tag))
    
    if not os.path.exists(banner_path):
        if level != "parton" and tag != _tag:
            return recover_banner(results_object, level, _run, results_object[_run].tags[0])
        elif level == 'parton':
            paths = [pjoin(path,'Events',run, 'unweighted_events.lhe.gz'),
                     pjoin(path,'Events',run, 'unweighted_events.lhe'),
                     pjoin(path,'Events',run, 'events.lhe.gz'),
                     pjoin(path,'Events',run, 'events.lhe')]
            for p in paths:
                if os.path.exists(p):
                    if MADEVENT:
                        import internal.lhe_parser as lhe_parser
                    else:
                        import madgraph.various.lhe_parser as lhe_parser
                    lhe = lhe_parser.EventFile(p)
                    return Banner(lhe.banner)

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
    
class InvalidRunCard(InvalidCmd):
    pass

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
            line = line.rstrip()
            if line.endswith('\\'):
                store_line += line[:-1]
            else:
                tmp = store_line + line
                self.append(tmp.strip())
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
        
    def get(self, tag, default=None):
        if isinstance(tag, int):
            list.__getattr__(self, tag)
        elif tag == 'info' or tag == "__setstate__":
            return default #for pickle
        elif tag == "multiparticles":
            out = []
            for line in self:
                if line.startswith('define'):
                    name, content = line[7:].split('=',1)
                    out.append((name, content))
            return out 
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
 
class InvalidCardEdition(InvalidCmd): pass 
 
class ConfigFile(dict):
    """ a class for storing/dealing with input file.
    """     

    def __init__(self, finput=None, **opt):
        """initialize a new instance. input can be an instance of MadLoopParam,
        a file, a path to a file, or simply Nothing"""                
        
        if isinstance(finput, self.__class__):
            dict.__init__(self, finput)
            assert finput.__dict__.keys()
            for key in finput.__dict__:
                setattr(self, key, copy.copy(getattr(finput, key)) )
            return
        else:
            dict.__init__(self)
        
        # Initialize it with all the default value
        self.user_set = set()
        self.auto_set = set()
        self.system_only = set()
        self.lower_to_case = {}
        self.list_parameter = {} #key -> type of list (int/float/bool/str/...
        self.dict_parameter = {}
        self.comments = {} # comment associated to parameters. can be display via help message
        # store the valid options for a given parameter.
        self.allowed_value = {}
        
        self.default_setup()
        

        # if input is define read that input
        if isinstance(finput, (file, str, StringIO.StringIO)):
            self.read(finput, **opt)

    def default_setup(self):
        pass

    def __copy__(self):
        return self.__class__(self)

    def __add__(self, other):
        """define the sum"""
        assert isinstance(other, dict)
        base = self.__class__(self)
        #base = copy.copy(self)
        base.update((key.lower(),value) for key, value in other.items())
        return base

    def __radd__(self, other):
        """define the sum"""
        new = copy.copy(other)
        new.update((key, value) for key, value in self.items())
        return new
    
    def __contains__(self, key):
        return dict.__contains__(self, key.lower())

    def __iter__(self):
        iter = super(ConfigFile, self).__iter__()
        return (self.lower_to_case[name] for name in iter)
    
    def keys(self):
        return [name for name in self]
    
    def items(self):
        return [(name,self[name]) for name in self]
        
    @staticmethod
    def warn(text, level, raiseerror=False):
        """convenient proxy to raiseerror/print warning"""

        if raiseerror is True:
            raise InvalidCardEdition(text)
        elif raiseerror:
            raise raiseerror(text)

        if isinstance(level,str):
            log = getattr(logger, level.lower())
        elif isinstance(level, int):
            log = lambda t: logger.log(level, t)
        elif level:
            log = level
        
        return log(text)

    def post_set(self, name, value, change_userdefine, raiseerror):
        
        if value is None:
            value = self[name]

        if hasattr(self, 'post_set_%s' % name):
            return getattr(self, 'post_set_%s' % name)(value, change_userdefine, raiseerror)
    
    def __setitem__(self, name, value, change_userdefine=False,raiseerror=False):
        """set the attribute and set correctly the type if the value is a string.
           change_userdefine on True if we have to add the parameter in user_set
        """
                       
        if  not len(self):
            #Should never happen but when deepcopy/pickle
            self.__init__()
                
        name = name.strip()
        lower_name = name.lower() 

        # 0. check if this parameter is a system only one
        if change_userdefine and lower_name in self.system_only:
            text='%s is a private entry which can not be modify by the user. Keep value at %s' % (name,self[name])
            self.warn(text, 'critical', raiseerror)
            return
        
        #1. check if the parameter is set to auto -> pass it to special
        if lower_name in self:
            targettype = type(dict.__getitem__(self, lower_name))
            if targettype != str and isinstance(value, str) and value.lower() == 'auto':
                self.auto_set.add(lower_name)
                if lower_name in self.user_set:
                    self.user_set.remove(lower_name)
                #keep old value.
                self.post_set(lower_name, 'auto', change_userdefine, raiseerror)
                return 
            elif lower_name in self.auto_set:
                self.auto_set.remove(lower_name)
            
        # 2. Find the type of the attribute that we want
        if lower_name in self.list_parameter:
            targettype = self.list_parameter[lower_name]
            
            
            
            if isinstance(value, str):
                # split for each comma/space
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]
                #do not perform split within a " or ' block  
                data = re.split(r"((?<![\\])['\"])((?:.(?!(?<![\\])\1))*.?)\1", str(value))
                new_value = []
                i = 0
                while len(data) > i:
                    current = filter(None, re.split(r'(?:(?<!\\)\s)|,', data[i]))
                    i+=1
                    if len(data) > i+1:
                        if current:
                            current[-1] += '{0}{1}{0}'.format(data[i], data[i+1])
                        else:
                            current = ['{0}{1}{0}'.format(data[i], data[i+1])]
                        i+=2
                    new_value += current
 
                value = new_value                           
                
            elif not hasattr(value, '__iter__'):
                value = [value]
            elif isinstance(value, dict):
                text = "not being able to handle dictionary in card entry"
                return self.warn(text, 'critical', raiseerror)

            #format each entry    
            values =[self.format_variable(v, targettype, name=name) 
                                                                 for v in value]
            
            # ensure that each entry are in the allowed list
            if lower_name in self.allowed_value and '*' not in self.allowed_value[lower_name]:
                new_values = []
                dropped = []
                for val in values:
                    allowed = self.allowed_value[lower_name]
            
                    if val in allowed:
                        new_values.append(val)
                        continue
                    elif isinstance(val, str):
                        val = val.lower()
                        allowed = allowed.lower()
                        if value in allowed:
                            i = allowed.index(value)
                            new_values.append(self.allowed_value[i])
                            continue
                    # no continue -> bad input
                    dropped.append(val)
                    
                if not new_values:

                    text= "value '%s' for entry '%s' is not valid.  Preserving previous value: '%s'.\n" \
                               % (value, name, self[lower_name])
                    text += "allowed values are any list composed of the following entry: %s" % ', '.join([str(i) for i in self.allowed_value[lower_name]])
                    return self.warn(text, 'warning', raiseerror)                    
                elif dropped:               
                    text = "some value for entry '%s' are not valid. Invalid item are: '%s'.\n" \
                               % (value, name, dropped)
                    text += "value will be set to %s" % new_values
                    text += "allowed items in the list are: %s" % ', '.join([str(i) for i in self.allowed_value[lower_name]])        
                    self.warn(text, 'warning')

                values = new_values

            # make the assignment
            dict.__setitem__(self, lower_name, values) 
            if change_userdefine:
                self.user_set.add(lower_name)
            #check for specific action
            return self.post_set(lower_name, None, change_userdefine, raiseerror) 
        elif lower_name in self.dict_parameter:
            targettype = self.dict_parameter[lower_name] 
            full_reset = True #check if we just update the current dict or not
            
            if isinstance(value, str):
                value = value.strip()
                # allowed entry:
                #   name : value   => just add the entry
                #   name , value   => just add the entry
                #   name  value    => just add the entry
                #   {name1:value1, name2:value2}   => full reset
                
                # split for each comma/space
                if value.startswith('{') and value.endswith('}'):
                    new_value = {}
                    for pair in value[1:-1].split(','):
                        if not pair.strip():
                            break
                        x, y = pair.split(':')
                        x, y = x.strip(), y.strip()
                        if x.startswith(('"',"'")) and x.endswith(x[0]):
                            x = x[1:-1] 
                        new_value[x] = y
                    value = new_value
                elif ',' in value:
                    x,y = value.split(',')
                    value = {x.strip():y.strip()}
                    full_reset = False
                    
                elif ':' in value:
                    x,y = value.split(':')
                    value = {x.strip():y.strip()}
                    full_reset = False       
                else:
                    x,y = value.split()
                    value = {x:y}
                    full_reset = False 
            
            if isinstance(value, dict):
                for key in value:
                    value[key] = self.format_variable(value[key], targettype, name=name)
                if full_reset:
                    dict.__setitem__(self, lower_name, value)
                else:
                    dict.__getitem__(self, lower_name).update(value)
            else:
                raise Exception, '%s should be of dict type'% lower_name
            if change_userdefine:
                self.user_set.add(lower_name)
            return self.post_set(lower_name, None, change_userdefine, raiseerror)
        elif name in self:            
            targettype = type(self[name])
        else:
            logger.debug('Trying to add argument %s in %s. ' % (name, self.__class__.__name__) +\
              'This argument is not defined by default. Please consider adding it.')
            suggestions = [k for k in self.keys() if k.startswith(name[0].lower())]
            if len(suggestions)>0:
                logger.debug("Did you mean one of the following: %s"%suggestions)
            self.add_param(lower_name, self.format_variable(UnknownType(value), 
                                                             UnknownType, name))
            self.lower_to_case[lower_name] = name
            if change_userdefine:
                self.user_set.add(lower_name)
            return self.post_set(lower_name, None, change_userdefine, raiseerror)
    
        value = self.format_variable(value, targettype, name=name)
        #check that the value is allowed:
        if lower_name in self.allowed_value and '*' not in self.allowed_value[lower_name]:
            valid = False
            allowed = self.allowed_value[lower_name]
            
            # check if the current value is allowed or not (set valid to True)
            if value in allowed:
                valid=True     
            elif isinstance(value, str):
                value = value.lower()
                allowed = [v.lower() for v in allowed]
                if value in allowed:
                    i = allowed.index(value)
                    value = self.allowed_value[lower_name][i]
                    valid=True
                    
            if not valid:
                # act if not valid:
                text = "value '%s' for entry '%s' is not valid.  Preserving previous value: '%s'.\n" \
                               % (value, name, self[lower_name])
                text += "allowed values are %s\n" % ', '.join([str(i) for i in self.allowed_value[lower_name]])
                if lower_name in self.comments:
                    text += 'type "help %s" for more information' % name
                return self.warn(text, 'warning', raiseerror)

        dict.__setitem__(self, lower_name, value)
        if change_userdefine:
            self.user_set.add(lower_name)
        self.post_set(lower_name, None, change_userdefine, raiseerror)


    def add_param(self, name, value, system=False, comment=False, typelist=None,
                  allowed=[]):
        """add a default parameter to the class"""

        lower_name = name.lower()
        if __debug__:
            if lower_name in self:
                raise Exception("Duplicate case for %s in %s" % (name,self.__class__))
            
        dict.__setitem__(self, lower_name, value)
        self.lower_to_case[lower_name] = name
        if isinstance(value, list):
            if len(value):
                targettype = type(value[0])
            else:
                targettype=typelist
                assert typelist
            if any([targettype != type(v) for v in value]):
                raise Exception, "All entry should have the same type"
            self.list_parameter[lower_name] = targettype
        elif isinstance(value, dict):
            allvalues = value.values()
            if any([type(allvalues[0]) != type(v) for v in allvalues]):
                raise Exception, "All entry should have the same type"   
            self.dict_parameter[lower_name] = type(allvalues[0])  
            if '__type__' in value:
                del value['__type__']
                dict.__setitem__(self, lower_name, value)
        
        if allowed and allowed != ['*']:
            self.allowed_value[lower_name] = allowed
            assert value in allowed or '*' in allowed
        #elif isinstance(value, bool) and allowed != ['*']:
        #    self.allowed_value[name] = [True, False]
                   
        if system:
            self.system_only.add(lower_name)
        if comment:
            self.comments[lower_name] = comment

    def do_help(self, name):
        """return a minimal help for the parameter"""
        
        out = "## Information on parameter %s from class %s\n" % (name, self.__class__.__name__)
        if name.lower() in self:
            out += "## current value: %s (parameter should be of type %s)\n" % (self[name], type(self[name]))
            if name.lower() in self.comments:
                out += '## %s\n' % self.comments[name.lower()].replace('\n', '\n## ')
        else:
            out += "## Unknown for this class\n"
        if name.lower() in self.user_set:
            out += "## This value is considered as been set by the user\n" 
        else:
            out += "## This value is considered as been set by the system\n"
        if name.lower() in self.allowed_value:
            if '*' not in self.allowed_value[name.lower()]:
                out += "Allowed value are: %s\n" % ','.join([str(p) for p in self.allowed_value[name.lower()]])
            else:
                out += "Suggested value are : %s\n " % ','.join([str(p) for p in self.allowed_value[name.lower()] if p!='*'])
        
        logger.info(out)
        return out

    @staticmethod
    def format_variable(value, targettype, name="unknown"):
        """assign the value to the attribute for the given format"""
        
        if not isinstance(value, str):
            # just have to check that we have the correct format
            if isinstance(value, targettype):
                pass # assignement at the end
            elif isinstance(value, numbers.Number) and issubclass(targettype, numbers.Number):
                try:
                    new_value = targettype(value)
                except TypeError:
                    if value.imag/value.real<1e-12:
                        new_value = targettype(value.real)
                    else:
                        raise
                if new_value == value:
                    value = new_value
                else:
                    raise InvalidCmd, "Wrong input type for %s found %s and expecting %s for value %s" %\
                        (name, type(value), targettype, value)
            else:
                raise InvalidCmd, "Wrong input type for %s found %s and expecting %s for value %s" %\
                        (name, type(value), targettype, value)                
        else:
            # We have a string we have to format the attribute from the string
            if targettype == UnknownType:
                # No formatting
                pass
            elif targettype == bool:
                value = value.strip()
                if value.lower() in ['0', '.false.', 'f', 'false', 'off']:
                    value = False
                elif value.lower() in ['1', '.true.', 't', 'true', 'on']:
                    value = True
                else:
                    raise InvalidCmd, "%s can not be mapped to True/False for %s" % (repr(value),name)
            elif targettype == str:
                value = value.strip()
                if value.startswith('\'') and value.endswith('\''):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
            elif targettype == int:
                if value.isdigit():
                    value = int(value)
                elif value[1:].isdigit() and value[0] == '-':
                    value = int(value)
                elif value.endswith(('k', 'M')) and value[:-1].isdigit():
                    convert = {'k':1000, 'M':1000000}
                    value =int(value[:-1]) * convert[value[-1]] 
                else:
                    try:
                        value = float(value.replace('d','e'))
                    except ValueError:
                        raise InvalidCmd, "%s can not be mapped to an integer" % value                    
                    try:
                        new_value = int(value)
                    except ValueError:
                        raise InvalidCmd, "%s can not be mapped to an integer" % value
                    else:
                        if value == new_value:
                            value = new_value
                        else:
                            raise InvalidCmd, "incorect input: %s need an integer for %s" % (value,name)
            elif targettype == float:
                value = value.replace('d','e') # pass from Fortran formatting
                try:
                    value = float(value)
                except ValueError:
                    try:
                        split = re.split('(\*|/)',value)
                        v = float(split[0])
                        for i in range((len(split)//2)):
                            if split[2*i+1] == '*':
                                v *=  float(split[2*i+2])
                            else:
                                v /=  float(split[2*i+2])
                    except:
                        v=0
                        raise InvalidCmd, "%s can not be mapped to a float" % value
                    finally:
                        value = v
            else:
                raise InvalidCmd, "type %s is not handle by the card" % targettype
            
        return value
            
 

    def __getitem__(self, name):
        
        lower_name = name.lower()
        if __debug__:
            if lower_name not in self:
                if lower_name in [key.lower() for key in self] :
                    raise Exception, "Some key are not lower case %s. Invalid use of the class!"\
                                     % [key for key in self if key.lower() != key]
        
        if lower_name in self.auto_set:
            return 'auto'

        return dict.__getitem__(self, name.lower())

    
    def set(self, name, value, changeifuserset=True, user=False, raiseerror=False):
        """convenient way to change attribute.
        changeifuserset=False means that the value is NOT change is the value is not on default.
        user=True, means that the value will be marked as modified by the user 
        (potentially preventing future change to the value) 
        """

        # changeifuserset=False -> we need to check if the user force a value.
        if not changeifuserset:
            if name.lower() in self.user_set:
                #value modified by the user -> do nothing
                return
            
        self.__setitem__(name, value, change_userdefine=user, raiseerror=raiseerror) 
 


class ProcCharacteristic(ConfigFile):
    """A class to handle information which are passed from MadGraph to the madevent
       interface.""" 
     
    def default_setup(self):
        """initialize the directory to the default value"""
        
        self.add_param('loop_induced', False)
        self.add_param('has_isr', False)
        self.add_param('has_fsr', False)
        self.add_param('nb_channel', 0)
        self.add_param('nexternal', 0)
        self.add_param('ninitial', 0)
        self.add_param('grouped_matrix', True)
        self.add_param('has_loops', False)
        self.add_param('bias_module','None')
        self.add_param('max_n_matched_jets', 0)
        self.add_param('colored_pdgs', [1,2,3,4,5])
        self.add_param('complex_mass_scheme', False)
        self.add_param('pdg_initial1', [0])
        self.add_param('pdg_initial2', [0])        

    def read(self, finput):
        """Read the input file, this can be a path to a file, 
           a file object, a str with the content of the file."""
           
        if isinstance(finput, str):
            if "\n" in finput:
                finput = finput.split('\n')
            elif os.path.isfile(finput):
                finput = open(finput)
            else:
                raise Exception, "No such file %s" % finput
            
        for line in finput:
            if '#' in line:
                line = line.split('#',1)[0]
            if not line:
                continue
            
            if '=' in line:
                key, value = line.split('=',1)
                self[key.strip()] = value
         
    def write(self, outputpath):
        """write the file"""

        template ="#    Information about the process      #\n"
        template +="#########################################\n"
        
        fsock = open(outputpath, 'w')
        fsock.write(template)
        
        for key, value in self.items():
            fsock.write(" %s = %s \n" % (key, value))
        
        fsock.close()   
 



class GridpackCard(ConfigFile):
    """an object for the GridpackCard"""
    
    def default_setup(self):
        """default value for the GridpackCard"""
    
        self.add_param("GridRun", True)
        self.add_param("gevents", 2500)
        self.add_param("gseed", 1)
        self.add_param("ngran", -1)  
 
    def read(self, finput):
        """Read the input file, this can be a path to a file, 
           a file object, a str with the content of the file."""
           
        if isinstance(finput, str):
            if "\n" in finput:
                finput = finput.split('\n')
            elif os.path.isfile(finput):
                finput = open(finput)
            else:
                raise Exception, "No such file %s" % finput
        
        for line in finput:
            line = line.split('#')[0]
            line = line.split('!')[0]
            line = line.split('=',1)
            if len(line) != 2:
                continue
            self[line[1].strip()] = line[0].replace('\'','').strip()

    def write(self, output_file, template=None):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""

        if not template:
            if not MADEVENT:
                template = pjoin(MG5DIR, 'Template', 'LO', 'Cards', 
                                                        'grid_card_default.dat')
            else:
                template = pjoin(MEDIR, 'Cards', 'grid_card_default.dat')

                
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
        
        if isinstance(output_file, str):
            fsock =  open(output_file,'w')
        else:
            fsock = output_file
            
        fsock.write(text)
        fsock.close()
        
class PY8Card(ConfigFile):
    """ Implements the Pythia8 card."""

    def add_default_subruns(self, type):
        """ Placeholder function to allow overwriting in the PY8SubRun daughter.
        The initialization of the self.subruns attribute should of course not
        be performed in PY8SubRun."""
        if type == 'parameters':
            if "LHEFInputs:nSubruns" not in self:
                self.add_param("LHEFInputs:nSubruns", 1,
                hidden='ALWAYS_WRITTEN',
                comment="""
    ====================
    Subrun definitions
    ====================
    """)
        if type == 'attributes':
            if not(hasattr(self,'subruns')):
                first_subrun = PY8SubRun(subrun_id=0)
                self.subruns = dict([(first_subrun['Main:subrun'],first_subrun)])

    def default_setup(self):
        """ Sets up the list of available PY8 parameters."""
        
        # Visible parameters
        # ==================
        self.add_param("Main:numberOfEvents", -1)
        # for MLM merging
        # -1.0 means that it will be set automatically by MadGraph5_aMC@NLO
        self.add_param("JetMatching:qCut", -1.0, always_write_to_card=False)
        self.add_param("JetMatching:doShowerKt",False,always_write_to_card=False)
        # -1 means that it is automatically set.
        self.add_param("JetMatching:nJetMax", -1, always_write_to_card=False) 
        # for CKKWL merging
        self.add_param("Merging:TMS", -1.0, always_write_to_card=False)
        self.add_param("Merging:Process", '<set_by_user>', always_write_to_card=False)
        # -1 means that it is automatically set.   
        self.add_param("Merging:nJetMax", -1, always_write_to_card=False)
        # for both merging, chose whether to also consider different merging
        # scale values for the extra weights related to scale and PDF variations.
        self.add_param("SysCalc:fullCutVariation", False)
        # Select the HepMC output. The user can prepend 'fifo:<optional_fifo_path>'
        # to indicate that he wants to pipe the output. Or /dev/null to turn the
        # output off.
        self.add_param("HEPMCoutput:file", 'auto')

        # Hidden parameters always written out
        # ====================================
        self.add_param("Beams:frameType", 4,
            hidden=True,
            comment='Tell Pythia8 that an LHEF input is used.')
        self.add_param("HEPMCoutput:scaling", 1.0e9,
            hidden=True,
            comment='1.0 corresponds to HEPMC weight given in [mb]. We choose here the [pb] normalization.')
        self.add_param("Check:epTolErr", 1e-2,
            hidden=True,
            comment='Be more forgiving with momentum mismatches.')
        # By default it is important to disable any cut on the rapidity of the showered jets
        # during MLML merging and by default it is set to 2.5
        self.add_param("JetMatching:etaJetMax", 1000.0, hidden=True, always_write_to_card=True)

        # Hidden parameters written out only if user_set or system_set
        # ============================================================
        self.add_param("PDF:pSet", 'LHAPDF5:CT10.LHgrid', hidden=True, always_write_to_card=False,
            comment='Reminder: Parameter below is shower tune dependent.')
        self.add_param("SpaceShower:alphaSvalue", 0.118, hidden=True, always_write_to_card=False,
            comment='Reminder: Parameter below is shower tune dependent.')
        self.add_param("TimeShower:alphaSvalue", 0.118, hidden=True, always_write_to_card=False,
            comment='Reminder: Parameter below is shower tune dependent.')
        self.add_param("hadronlevel:all", True, hidden=True, always_write_to_card=False,
            comment='This allows to turn on/off hadronization alltogether.')
        self.add_param("partonlevel:mpi", True, hidden=True, always_write_to_card=False,
            comment='This allows to turn on/off MPI alltogether.')
        self.add_param("Beams:setProductionScalesFromLHEF", False, hidden=True, 
            always_write_to_card=False,
            comment='This parameter is automatically set to True by MG5aMC when doing MLM merging with PY8.')
        
        # for MLM merging
        self.add_param("JetMatching:merge", False, hidden=True, always_write_to_card=False,
          comment='Specifiy if we are merging sample of different multiplicity.')
        self.add_param("SysCalc:qCutList", [10.0,20.0], hidden=True, always_write_to_card=False)
        self['SysCalc:qCutList'] = 'auto'
        self.add_param("SysCalc:qWeed",-1.0,hidden=True, always_write_to_card=False,
          comment='Value of the merging scale below which one does not even write the HepMC event.')
        self.add_param("JetMatching:doVeto", False, hidden=True, always_write_to_card=False,
          comment='Do veto externally (e.g. in SysCalc).')
        self.add_param("JetMatching:scheme", 1, hidden=True, always_write_to_card=False) 
        self.add_param("JetMatching:setMad", False, hidden=True, always_write_to_card=False,
              comment='Specify one must read inputs from the MadGraph banner.') 
        self.add_param("JetMatching:coneRadius", 1.0, hidden=True, always_write_to_card=False)
        self.add_param("JetMatching:nQmatch",4,hidden=True, always_write_to_card=False)
        # for CKKWL merging (common with UMEPS, UNLOPS)
        self.add_param("TimeShower:pTmaxMatch", 2, hidden=True, always_write_to_card=False)
        self.add_param("SpaceShower:pTmaxMatch", 1, hidden=True, always_write_to_card=False)
        self.add_param("SysCalc:tmsList", [10.0,20.0], hidden=True, always_write_to_card=False)
        self['SysCalc:tmsList'] = 'auto'
        self.add_param("Merging:muFac", 91.188, hidden=True, always_write_to_card=False,
                        comment='Set factorisation scales of the 2->2 process.')
        self.add_param("Merging:applyVeto", False, hidden=True, always_write_to_card=False,
          comment='Do veto externally (e.g. in SysCalc).')
        self.add_param("Merging:includeWeightInXsection", True, hidden=True, always_write_to_card=False,
          comment='If turned off, then the option belows forces PY8 to keep the original weight.')                       
        self.add_param("Merging:muRen", 91.188, hidden=True, always_write_to_card=False,
                      comment='Set renormalization scales of the 2->2 process.')
        self.add_param("Merging:muFacInME", 91.188, hidden=True, always_write_to_card=False,
                 comment='Set factorisation scales of the 2->2 Matrix Element.')
        self.add_param("Merging:muRenInME", 91.188, hidden=True, always_write_to_card=False,
               comment='Set renormalization scales of the 2->2 Matrix Element.')
        self.add_param("SpaceShower:rapidityOrder", False, hidden=True, always_write_to_card=False)
        self.add_param("Merging:nQuarksMerge",4,hidden=True, always_write_to_card=False)
        # To be added in subruns for CKKWL
        self.add_param("Merging:mayRemoveDecayProducts", False, hidden=True, always_write_to_card=False)
        self.add_param("Merging:doKTMerging", False, hidden=True, always_write_to_card=False)
        self.add_param("Merging:Dparameter", 0.4, hidden=True, always_write_to_card=False)        
        self.add_param("Merging:doPTLundMerging", False, hidden=True, always_write_to_card=False)

        # Special Pythia8 paremeters useful to simplify the shower.
        self.add_param("BeamRemnants:primordialKT", True, hidden=True, always_write_to_card=False, comment="see http://home.thep.lu.se/~torbjorn/pythia82html/BeamRemnants.html")
        self.add_param("PartonLevel:Remnants", True, hidden=True, always_write_to_card=False, comment="Master switch for addition of beam remnants. Cannot be used to generate complete events")
        self.add_param("Check:event", True, hidden=True, always_write_to_card=False, comment="check physical sanity of the events")
        self.add_param("TimeShower:QEDshowerByQ", True, hidden=True, always_write_to_card=False, comment="Allow quarks to radiate photons for FSR, i.e. branchings q -> q gamma")
        self.add_param("TimeShower:QEDshowerByL", True, hidden=True, always_write_to_card=False, comment="Allow leptons to radiate photons for FSR, i.e. branchings l -> l gamma")
        self.add_param("SpaceShower:QEDshowerByQ", True, hidden=True, always_write_to_card=False, comment="Allow quarks to radiate photons for ISR, i.e. branchings q -> q gamma")
        self.add_param("SpaceShower:QEDshowerByL", True, hidden=True, always_write_to_card=False, comment="Allow leptons to radiate photonsfor ISR, i.e. branchings l -> l gamma")
        self.add_param("PartonLevel:FSRinResonances", True, hidden=True, always_write_to_card=False, comment="Do not allow shower to run from decay product of unstable particle")
        self.add_param("ProcessLevel:resonanceDecays", True, hidden=True, always_write_to_card=False, comment="Do not allow unstable particle to decay.")

        # Add parameters controlling the subruns execution flow.
        # These parameters should not be part of PY8SubRun daughter.
        self.add_default_subruns('parameters')
             
    def __init__(self, *args, **opts):
        # Parameters which are not printed in the card unless they are 
        # 'user_set' or 'system_set' or part of the 
        #  self.hidden_params_to_always_print set.
        self.hidden_param = []
        self.hidden_params_to_always_write = set()
        self.visible_params_to_always_write = set()
        # List of parameters that should never be written out given the current context.
        self.params_to_never_write = set()
        
        # Parameters which have been set by the system (i.e. MG5 itself during
        # the regular course of the shower interface)
        self.system_set = set()
        
        # Add attributes controlling the subruns execution flow.
        # These attributes should not be part of PY8SubRun daughter.
        self.add_default_subruns('attributes')
        
        # Parameters which have been set by the 
        super(PY8Card, self).__init__(*args, **opts)

    def add_param(self, name, value, hidden=False, always_write_to_card=True, 
                                                                  comment=None):
        """ add a parameter to the card. value is the default value and 
        defines the type (int/float/bool/str) of the input.
        The option 'hidden' decides whether the parameter should be visible to the user.
        The option 'always_write_to_card' decides whether it should
        always be printed or only when it is system_set or user_set.
        The option 'comment' can be used to specify a comment to write above
        hidden parameters.
        """
        super(PY8Card, self).add_param(name, value, comment=comment)
        name = name.lower()
        if hidden:
            self.hidden_param.append(name)
            if always_write_to_card:
                self.hidden_params_to_always_write.add(name)
        else:
            if always_write_to_card:
                self.visible_params_to_always_write.add(name)                
        if not comment is None:
            if not isinstance(comment, str):
                raise MadGraph5Error("Option 'comment' must be a string, not"+\
                                                          " '%s'."%str(comment))

    def add_subrun(self, py8_subrun):
        """Add a subrun to this PY8 Card."""
        assert(isinstance(py8_subrun,PY8SubRun))
        if py8_subrun['Main:subrun']==-1:
            raise MadGraph5Error, "Make sure to correctly set the subrun ID"+\
                            " 'Main:subrun' *before* adding it to the PY8 Card."
        if py8_subrun['Main:subrun'] in self.subruns:
            raise MadGraph5Error, "A subrun with ID '%s'"%py8_subrun['Main:subrun']+\
                " is already present in this PY8 card. Remove it first, or "+\
                                                          " access it directly."
        self.subruns[py8_subrun['Main:subrun']] = py8_subrun
        if not 'LHEFInputs:nSubruns' in self.user_set:
            self['LHEFInputs:nSubruns'] = max(self.subruns.keys())
        
    def userSet(self, name, value, **opts):
        """Set an attribute of this card, following a user_request"""
        self.__setitem__(name, value, change_userdefine=True, **opts)
        if name.lower() in self.system_set:
            self.system_set.remove(name.lower())

    def vetoParamWriteOut(self, name):
        """ Forbid the writeout of a specific parameter of this card when the 
        "write" function will be invoked."""
        self.params_to_never_write.add(name.lower())
    
    def systemSet(self, name, value, **opts):
        """Set an attribute of this card, independently of a specific user
        request and only if not already user_set."""
        try:
            force = opts.pop('force')
        except KeyError:
            force = False
        if force or name.lower() not in self.user_set:
            self.__setitem__(name, value, change_userdefine=False, **opts)
            self.system_set.add(name.lower())
    
    def MadGraphSet(self, name, value, **opts):
        """ Sets a card attribute, but only if it is absent or not already
        user_set."""
        try:
            force = opts.pop('force')
        except KeyError:
            force = False
        if name.lower() not in self or (force or name.lower() not in self.user_set):
            self.__setitem__(name, value, change_userdefine=False, **opts)
            self.system_set.add(name.lower())            
    
    def defaultSet(self, name, value, **opts):
            self.__setitem__(name, value, change_userdefine=False, **opts)
        
    @staticmethod
    def pythia8_formatting(value, formatv=None):
        """format the variable into pythia8 card convention.
        The type is detected by default"""
        if not formatv:
            if isinstance(value,UnknownType):
                formatv = 'unknown'                
            elif isinstance(value, bool):
                formatv = 'bool'
            elif isinstance(value, int):
                formatv = 'int'
            elif isinstance(value, float):
                formatv = 'float'
            elif isinstance(value, str):
                formatv = 'str'
            elif isinstance(value, list):
                formatv = 'list'
            else:
                logger.debug("unknow format for pythia8_formatting: %s" , value)
                formatv = 'str'
        else:
            assert formatv
            
        if formatv == 'unknown':
            # No formatting then
            return str(value)
        if formatv == 'bool':
            if str(value) in ['1','T','.true.','True','on']:
                return 'on'
            else:
                return 'off'
        elif formatv == 'int':
            try:
                return str(int(value))
            except ValueError:
                fl = float(value)
                if int(fl) == fl:
                    return str(int(fl))
                else:
                    raise
        elif formatv == 'float':
            return '%.10e' % float(value)
        elif formatv == 'shortfloat':
            return '%.3f' % float(value)        
        elif formatv == 'str':
            return "%s" % value
        elif formatv == 'list':
            if len(value) and isinstance(value[0],float):
                return ','.join([PY8Card.pythia8_formatting(arg, 'shortfloat') for arg in value])
            else:
                return ','.join([PY8Card.pythia8_formatting(arg) for arg in value])
            

    def write(self, output_file, template, read_subrun=False, 
                    print_only_visible=False, direct_pythia_input=False, add_missing=True):
        """ Write the card to output_file using a specific template.
        > 'print_only_visible' specifies whether or not the hidden parameters
            should be written out if they are in the hidden_params_to_always_write
            list and system_set.
        > If 'direct_pythia_input' is true, then visible parameters which are not
          in the self.visible_params_to_always_write list and are not user_set
          or system_set are commented.
        > If 'add_missing' is False then parameters that should be written_out but are absent
        from the template will not be written out."""

        # First list the visible parameters
        visible_param = [p for p in self if p.lower() not in self.hidden_param
                                                  or p.lower() in self.user_set]
        # Filter against list of parameters vetoed for write-out
        visible_param = [p for p in visible_param if p.lower() not in self.params_to_never_write]
        
        # Now the hidden param which must be written out
        if print_only_visible:
            hidden_output_param = []
        else:
            hidden_output_param = [p for p in self if p.lower() in self.hidden_param and
              not p.lower() in self.user_set and
              (p.lower() in self.hidden_params_to_always_write or 
                                                  p.lower() in self.system_set)]
        # Filter against list of parameters vetoed for write-out
        hidden_output_param = [p for p in hidden_output_param if p not in self.params_to_never_write]
        
        if print_only_visible:
            subruns = []
        else:
            if not read_subrun:
                subruns = sorted(self.subruns.keys())
        
        # Store the subruns to write in a dictionary, with its ID in key
        # and the corresponding stringstream in value
        subruns_to_write = {}
        
        # Sort these parameters nicely so as to put together parameters
        # belonging to the same group (i.e. prefix before the ':' in their name).
        def group_params(params):
            if len(params)==0:
                return []
            groups = {}
            for p in params:
                try:
                    groups[':'.join(p.split(':')[:-1])].append(p)
                except KeyError:
                    groups[':'.join(p.split(':')[:-1])] = [p,]
            res =  sum(groups.values(),[])
            # Make sure 'Main:subrun' appears first
            if 'Main:subrun' in res:
                res.insert(0,res.pop(res.index('Main:subrun')))
            # Make sure 'LHEFInputs:nSubruns' appears last
            if 'LHEFInputs:nSubruns' in res:
                res.append(res.pop(res.index('LHEFInputs:nSubruns')))
            return res

        visible_param       = group_params(visible_param)
        hidden_output_param = group_params(hidden_output_param)

        # First dump in a temporary_output (might need to have a second pass
        # at the very end to update 'LHEFInputs:nSubruns')
        output = StringIO.StringIO()
            
        # Setup template from which to read
        if isinstance(template, str):
            if os.path.isfile(template):
                tmpl = open(template, 'r')
            elif '\n' in template:
                tmpl = StringIO.StringIO(template)
            else:
                raise Exception, "File input '%s' not found." % file_input     
        elif template is None:
            # Then use a dummy empty StringIO, hence skipping the reading
            tmpl = StringIO.StringIO()
        elif isinstance(template, (StringIO.StringIO, file)):
            tmpl = template
        else:
            raise MadGraph5Error("Incorrect type for argument 'template': %s"%
                                                    template.__class__.__name__)

        # Read the template
        last_pos = tmpl.tell()
        line     = tmpl.readline()
        started_subrun_reading = False
        while line!='':
            # Skip comments
            if line.strip().startswith('!') or \
               line.strip().startswith('\n') or\
               line.strip() == '':
                output.write(line)
                # Proceed to next line
                last_pos = tmpl.tell()
                line     = tmpl.readline()
                continue
            # Read parameter
            try:
                param_entry, value_entry = line.split('=')
                param = param_entry.strip()
                value = value_entry.strip()
            except ValueError:
                line = line.replace('\n','')
                raise MadGraph5Error, "Could not read line '%s' of Pythia8 card."%\
                                                                            line
            # Read a subrun if detected:
            if param=='Main:subrun':
                if read_subrun:
                    if not started_subrun_reading:
                        # Record that the subrun reading has started and proceed
                        started_subrun_reading = True
                    else:
                        # We encountered the next subrun. rewind last line and exit
                        tmpl.seek(last_pos)
                        break
                else:
                    # Start the reading of this subrun
                    tmpl.seek(last_pos)
                    subruns_to_write[int(value)] = StringIO.StringIO()
                    if int(value) in subruns:
                        self.subruns[int(value)].write(subruns_to_write[int(value)],
                                                      tmpl,read_subrun=True)
                        # Remove this subrun ID from the list
                        subruns.pop(subruns.index(int(value)))
                    else:
                        # Unknow subrun, create a dummy one
                        DummySubrun=PY8SubRun()
                        # Remove all of its variables (so that nothing is overwritten)
                        DummySubrun.clear()
                        DummySubrun.write(subruns_to_write[int(value)],
                                tmpl, read_subrun=True, 
                                print_only_visible=print_only_visible, 
                                direct_pythia_input=direct_pythia_input)

                        logger.info('Adding new unknown subrun with ID %d.'%
                                                                     int(value))
                    # Proceed to next line
                    last_pos = tmpl.tell()
                    line     = tmpl.readline()
                    continue
            
            # Change parameters which must be output
            if param in visible_param:
                new_value = PY8Card.pythia8_formatting(self[param])
                visible_param.pop(visible_param.index(param))
            elif param in hidden_output_param:
                new_value = PY8Card.pythia8_formatting(self[param])
                hidden_output_param.pop(hidden_output_param.index(param))
            else:
                # Just copy parameters which don't need to be specified
                if param.lower() not in self.params_to_never_write:
                    output.write(line)
                else:
                    output.write('! The following parameter was forced to be commented out by MG5aMC.\n')
                    output.write('! %s'%line)
                # Proceed to next line
                last_pos = tmpl.tell()
                line     = tmpl.readline()
                continue
            
            # Substitute the value. 
            # If it is directly the pytia input, then don't write the param if it
            # is not in the list of visible_params_to_always_write and was 
            # not user_set or system_set
            if ((not direct_pythia_input) or
                  (param.lower() in self.visible_params_to_always_write) or
                  (param.lower() in self.user_set) or
                  (param.lower() in self.system_set)):
                template = '%s=%s'
            else:
                # These are parameters that the user can edit in AskEditCards
                # but if neither the user nor the system edited them,
                # then they shouldn't be passed to Pythia
                template = '!%s=%s'

            output.write(template%(param_entry,
                                  value_entry.replace(value,new_value)))
        
            # Proceed to next line
            last_pos = tmpl.tell()
            line     = tmpl.readline()
        
        # If add_missing is False, make sure to empty the list of remaining parameters
        if not add_missing:
            visible_param = []
            hidden_output_param = []
        
        # Now output the missing parameters. Warn about visible ones.
        if len(visible_param)>0 and not template is None:
            output.write(
"""!
! Additional general parameters%s.
!
"""%(' for subrun %d'%self['Main:subrun'] if 'Main:subrun' in self else ''))
        for param in visible_param:
            value = PY8Card.pythia8_formatting(self[param])
            output.write('%s=%s\n'%(param,value))
            if template is None:
                if param=='Main:subrun':
                    output.write(
"""!
!  Definition of subrun %d
!
"""%self['Main:subrun'])
            elif param.lower() not in self.hidden_param:
                logger.debug('Adding parameter %s (missing in the template) to current '+\
                                    'pythia8 card (with value %s)',param, value)

        if len(hidden_output_param)>0 and not template is None:
            output.write(
"""!
! Additional technical parameters%s set by MG5_aMC.
!
"""%(' for subrun %d'%self['Main:subrun'] if 'Main:subrun' in self else ''))
        for param in hidden_output_param:
            if param.lower() in self.comments:
                comment = '\n'.join('! %s'%c for c in 
                          self.comments[param.lower()].split('\n'))
                output.write(comment+'\n')
            output.write('%s=%s\n'%(param,PY8Card.pythia8_formatting(self[param])))
        
        # Don't close the file if we were reading a subrun, but simply write 
        # output and return now
        if read_subrun:
            output_file.write(output.getvalue())
            return

        # Now add subruns not present in the template
        for subrunID in subruns:
            new_subrun = StringIO.StringIO()
            self.subruns[subrunID].write(new_subrun,None,read_subrun=True)
            subruns_to_write[subrunID] = new_subrun

        # Add all subruns to the output, in the right order
        for subrunID in sorted(subruns_to_write):
            output.write(subruns_to_write[subrunID].getvalue())

        # If 'LHEFInputs:nSubruns' is not user_set, then make sure it is
        # updated at least larger or equal to the maximum SubRunID
        if 'LHEFInputs:nSubruns'.lower() not in self.user_set and \
             len(subruns_to_write)>0 and 'LHEFInputs:nSubruns' in self\
             and self['LHEFInputs:nSubruns']<max(subruns_to_write.keys()):
            logger.info("Updating PY8 parameter 'LHEFInputs:nSubruns' to "+
          "%d so as to cover all defined subruns."%max(subruns_to_write.keys()))
            self['LHEFInputs:nSubruns'] = max(subruns_to_write.keys())
            output = StringIO.StringIO()
            self.write(output,template,print_only_visible=print_only_visible)

        # Write output
        if isinstance(output_file, str):
            out = open(output_file,'w')
            out.write(output.getvalue())
            out.close()
        else:
            output_file.write(output.getvalue())
        
    def read(self, file_input, read_subrun=False, setter='default'):
        """Read the input file, this can be a path to a file, 
           a file object, a str with the content of the file.
           The setter option choses the authority that sets potential 
           modified/new parameters. It can be either: 
             'default' or 'user' or 'system'"""
        if isinstance(file_input, str):
            if "\n" in file_input:
                finput = StringIO.StringIO(file_input)
            elif os.path.isfile(file_input):
                finput = open(file_input)
            else:
                raise Exception, "File input '%s' not found." % file_input
        elif isinstance(file_input, (StringIO.StringIO, file)):
            finput = file_input
        else:
            raise MadGraph5Error("Incorrect type for argument 'file_input': %s"%
                                                    file_input.__class__.__name__)

        # Read the template
        last_pos = finput.tell()
        line     = finput.readline()
        started_subrun_reading = False
        while line!='':
            # Skip comments
            if line.strip().startswith('!') or line.strip()=='':
                # proceed to next line
                last_pos = finput.tell()
                line     = finput.readline()
                continue
            # Read parameter
            try:
                param, value = line.split('=',1)
                param = param.strip()
                value = value.strip()
            except ValueError:
                line = line.replace('\n','')
                raise MadGraph5Error, "Could not read line '%s' of Pythia8 card."%\
                                                                          line
            if '!' in value:
                value,_ = value.split('!',1)                                                             
                                                                          
            # Read a subrun if detected:
            if param=='Main:subrun':
                if read_subrun:
                    if not started_subrun_reading:
                        # Record that the subrun reading has started and proceed
                        started_subrun_reading = True
                    else:
                        # We encountered the next subrun. rewind last line and exit
                        finput.seek(last_pos)
                        return
                else:
                    # Start the reading of this subrun
                    finput.seek(last_pos)
                    if int(value) in self.subruns:
                        self.subruns[int(value)].read(finput,read_subrun=True,
                                                                  setter=setter)
                    else:
                        # Unknow subrun, create a dummy one
                        NewSubrun=PY8SubRun()
                        NewSubrun.read(finput,read_subrun=True, setter=setter)
                        self.add_subrun(NewSubrun)

                    # proceed to next line
                    last_pos = finput.tell()
                    line     = finput.readline()
                    continue
            
            # Read parameter. The case of a parameter not defined in the card is
            # handled directly in ConfigFile.

            # Use the appropriate authority to set the new/changed variable
            if setter == 'user':
                self.userSet(param,value)
            elif setter == 'system':
                self.systemSet(param,value)
            else:
                self.defaultSet(param,value)

            # proceed to next line
            last_pos = finput.tell()
            line     = finput.readline()

class PY8SubRun(PY8Card):
    """ Class to characterize a specific PY8 card subrun section. """

    def add_default_subruns(self, type):
        """ Overloading of the homonym function called in the __init__ of PY8Card.
        The initialization of the self.subruns attribute should of course not
        be performed in PY8SubRun."""
        pass

    def __init__(self, *args, **opts):
        """ Initialize a subrun """
        
        # Force user to set it manually.
        subrunID = -1
        if 'subrun_id' in opts:
            subrunID = opts.pop('subrun_id')

        super(PY8SubRun, self).__init__(*args, **opts)
        self['Main:subrun']=subrunID

    def default_setup(self):
        """Sets up the list of available PY8SubRun parameters."""
        
        # Add all default PY8Card parameters
        super(PY8SubRun, self).default_setup()
        # Make sure they are all hidden
        self.hidden_param = [k.lower() for k in self.keys()]
        self.hidden_params_to_always_write = set()
        self.visible_params_to_always_write = set()

        # Now add Main:subrun and Beams:LHEF. They are not hidden.
        self.add_param("Main:subrun", -1)
        self.add_param("Beams:LHEF", "events.lhe.gz")



runblock = collections.namedtuple('block', ('name', 'fields', 'template_on', 'template_off'))
class RunCard(ConfigFile):

    filename = 'run_card'
    blocks = [] 
                                   
    def __new__(cls, finput=None, **opt):
        if cls is RunCard:
            if not finput:
                target_class = RunCardLO
            elif isinstance(finput, cls):
                target_class = finput.__class__
            elif isinstance(finput, str):
                if '\n' not in finput:
                    finput = open(finput).read()
                if 'req_acc_FO' in finput:
                    target_class = RunCardNLO
                else:
                    target_class = RunCardLO
            else:
                return None
            return super(RunCard, cls).__new__(target_class, finput, **opt)
        else:
            return super(RunCard, cls).__new__(cls, finput, **opt)

    def __init__(self, *args, **opts):
        
        # The following parameter are updated in the defaultsetup stage.
        
        #parameter for which no warning should be raised if not define
        self.hidden_param = []
        # in which include file the parameer should be written
        self.includepath = collections.defaultdict(list)
        #some parameter have different name in fortran code
        self.fortran_name = {}
        #parameter which are not supported anymore. (no action on the code)
        self.legacy_parameter = {}
        #a list with all the cuts variable
        self.cuts_parameter = []
        # parameter added where legacy requires an older value.
        self.system_default = {}
        
        self.display_block = [] # set some block to be displayed
        self.warned=False


        super(RunCard, self).__init__(*args, **opts)

    def add_param(self, name, value, fortran_name=None, include=True, 
                  hidden=False, legacy=False, cut=False, system=False, sys_default=None, 
                  **opts):
        """ add a parameter to the card. value is the default value and 
        defines the type (int/float/bool/str) of the input.
        fortran_name defines what is the associate name in the f77 code
        include defines if we have to put the value in the include file
        hidden defines if the parameter is expected to be define by the user.
        legacy:Parameter which is not used anymore (raise a warning if not default)
        cut: defines the list of cut parameter to allow to set them all to off.
        sys_default: default used if the parameter is not in the card
        
        options of **opts:
        - allowed: list of valid options. '*' means anything else should be allowed.
                 empty list means anything possible as well. 
        - comment: add comment for writing/help
        - typelist: type of the list if default is empty
        """

        super(RunCard, self).add_param(name, value, system=system,**opts)
        name = name.lower()
        if fortran_name:
            self.fortran_name[name] = fortran_name
        if legacy:
            self.legacy_parameter[name] = value
            include = False
        self.includepath[include].append(name)
        if hidden or system:
            self.hidden_param.append(name)
        if cut:
            self.cuts_parameter.append(name)
        if sys_default is not None:
            self.system_default[name] = sys_default

        

    def read(self, finput, consistency=True):
        """Read the input file, this can be a path to a file, 
           a file object, a str with the content of the file."""
           
        if isinstance(finput, str):
            if "\n" in finput:
                finput = finput.split('\n')
            elif os.path.isfile(finput):
                finput = open(finput)
            else:
                raise Exception, "No such file %s" % finput
        
        for line in finput:
            line = line.split('#')[0]
            line = line.split('!')[0]
            line = line.rsplit('=',1)
            if len(line) != 2:
                continue
            value, name = line
            name = name.lower().strip()
            if name not in self and ('min' in name or 'max' in name):
                #looks like an entry added by one user -> add it nicely
                self.add_param(name, float(value), hidden=True, cut=True)
            else:
                self.set( name, value, user=True)
        # parameter not set in the run_card can be set to compatiblity value
        if consistency:
                try:
                    self.check_validity()
                except InvalidRunCard, error:
                    if consistency == 'warning':
                        logger.warning(str(error))
                    else:
                        raise
                    
                
    def write(self, output_file, template=None, python_template=False,
                    write_hidden=False):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""

        to_write = set(self.user_set) 
        written = set()
        if not template:
            raise Exception

        # check which optional block to write:
        write_block= []
        for b in self.blocks:
            name = b.name
            # check if the block has to be written
            if name not in self.display_block and \
               not any(f in self.user_set for f in b.fields):
                continue
            write_block.append(b.name)
        
        if python_template and not to_write:
            import string
            text = file(template,'r').read() 
            if self.blocks:
                text = string.Template(text)
                mapping = {}
                for b in self.blocks:
                    if b.name in write_block:
                        mapping[b.name] = b.template_on
                    else:
                        mapping[b.name] = b.template_off
                text = text.substitute(mapping)

            if not self.list_parameter:
                text = text % self
            else:
                data = dict(self)                
                for name in self.list_parameter:
                    if self.list_parameter[name] != str:
                        data[name] = ', '.join(str(v) for v in data[name])
                    else:
                        data[name] = "['%s']" % "', '".join(str(v) for v in data[name])
                text = text % data
        else:                        
            text = ""
            for line in file(template,'r'):                  
                nline = line.split('#')[0]
                nline = nline.split('!')[0]
                comment = line[len(nline):]
                nline = nline.split('=')
                if python_template and nline[0].startswith('$'):
                    block_name = nline[0][1:]
                    this_group = [b for b in self.blocks if b.name == block_name]
                    if not this_group:
                        logger.debug("block %s not defined", block_name)
                        continue
                    else:
                        this_group = this_group[0]
                    if block_name in write_block:
                        text += this_group.on_template % self
                        for name in this_group.fields:
                            written.add(f)
                            if name in to_write:
                                to_write.remove(name)
                    else:
                        text += this_group.template_off % self
                    
                elif len(nline) != 2:
                    text += line
                elif nline[1].strip() in self:
                    name = nline[1].strip().lower()
                    value = self[name]
                    if name in self.list_parameter:
                        if self.list_parameter[name] != str:
                            value = ', '.join([str(v) for v in value])
                        else:
                            value =  "['%s']" % "', '".join(str(v) for v in value)
                    if python_template:
                        text += line % {nline[1].strip():value, name:value}
                        written.add(name)
                    else:
                        if not comment or comment[-1]!='\n':
                            endline = '\n'
                        else:
                            endline = ''
                        text += '  %s\t= %s %s%s' % (value, name, comment, endline)
                        written.add(name)                        

                    if name in to_write:
                        to_write.remove(name)
                else:
                    logger.info('Adding missing parameter %s to current %s (with default value)',
                                 (name, self.filename))
                    written.add(name) 
                    text += line 

            for b in self.blocks:
                if b.name not in write_block:
                    continue
                # check if all attribute of the block have been written already
                if all(f in written for f in b.fields):
                    continue

                to_add = []
                for line in b.template_on.split('\n'):                  
                    nline = line.split('#')[0]
                    nline = nline.split('!')[0]
                    nline = nline.split('=')
                    if len(nline) != 2:
                        to_add.append(line)
                    elif nline[1].strip() in self:
                        name = nline[1].strip().lower()
                        value = self[name]
                        if name in self.list_parameter:
                            value = ', '.join([str(v) for v in value])
                        if name in written:
                            continue #already include before
                        else:
                            to_add.append(line % {nline[1].strip():value, name:value})
                            written.add(name)                        
    
                        if name in to_write:
                            to_write.remove(name)
                    else:
                        raise Exception
                
                if b.template_off in text:
                    text = text.replace(b.template_off, '\n'.join(to_add))
                else:
                    text += '\n'.join(to_add)

        if to_write or write_hidden:
            text+="""#********************************************************************* 
#  Additional hidden parameters
#*********************************************************************
"""            
            if write_hidden:
                #
                # do not write hidden parameter not hidden for this template 
                #
                if python_template:
                    written = written.union(set(re.findall('\%\((\w*)\)s', file(template,'r').read(), re.M)))
                to_write = to_write.union(set(self.hidden_param))
                to_write = to_write.difference(written)

            for key in to_write:
                if key in self.system_only:
                    continue

                comment = self.comments.get(key,'hidden_parameter').replace('\n','\n#')
                text += '  %s\t= %s # %s\n' % (self[key], key, comment)

        if isinstance(output_file, str):
            fsock = open(output_file,'w')
            fsock.write(text)
            fsock.close()
        else:
            output_file.write(text)


    def get_default(self, name, default=None, log_level=None):
        """return self[name] if exist otherwise default. log control if we 
        put a warning or not if we use the default value"""

        lower_name = name.lower()
        if lower_name not in self.user_set:
            if log_level is None:
                if lower_name in self.system_only:
                    log_level = 5
                elif lower_name in self.auto_set:
                    log_level = 5
                elif lower_name in self.hidden_param:
                    log_level = 10
                else:
                    log_level = 20
            if not default:
                default = dict.__getitem__(self, name.lower())
 
            logger.log(log_level, '%s missed argument %s. Takes default: %s'
                                   % (self.filename, name, default))
            self[name] = default
            return default
        else:
            return self[name]   

    
    @staticmethod
    def f77_formatting(value, formatv=None):
        """format the variable into fortran. The type is detected by default"""

        if not formatv:
            if isinstance(value, bool):
                formatv = 'bool'
            elif isinstance(value, int):
                formatv = 'int'
            elif isinstance(value, float):
                formatv = 'float'
            elif isinstance(value, str):
                formatv = 'str'
            else:
                logger.debug("unknow format for f77_formatting: %s" , str(value))
                formatv = 'str'
        else:
            assert formatv
            
        if formatv == 'bool':
            if str(value) in ['1','T','.true.','True']:
                return '.true.'
            else:
                return '.false.'
            
        elif formatv == 'int':
            try:
                return str(int(value))
            except ValueError:
                fl = float(value)
                if int(fl) == fl:
                    return str(int(fl))
                else:
                    raise
                
        elif formatv == 'float':
            if isinstance(value, str):
                value = value.replace('d','e')
            return ('%.10e' % float(value)).replace('e','d')
        
        elif formatv == 'str':
            # Check if it is a list
            if value.strip().startswith('[') and value.strip().endswith(']'):
                elements = (value.strip()[1:-1]).split()
                return ['_length = %d'%len(elements)]+\
                       ['(%d) = %s'%(i+1, elem.strip()) for i, elem in \
                                                            enumerate(elements)]
            else:
                return "'%s'" % value
        

    
    def check_validity(self, log_level=30):
        """check that parameter missing in the card are set to the expected value"""

        for name, value in self.system_default.items():
                self.set(name, value, changeifuserset=False)
        

        for name in self.includepath[False]:
            to_bypass = self.hidden_param + self.legacy_parameter.keys()
            if name not in to_bypass:
                self.get_default(name, log_level=log_level) 

        for name in self.legacy_parameter:
            if self[name] != self.legacy_parameter[name]:
                logger.warning("The parameter %s is not supported anymore this parameter will be ignored." % name)
               
    default_include_file = 'run_card.inc'

    def update_system_parameter_for_include(self):
        """update hidden system only parameter for the correct writtin in the 
        include"""
        return

    def write_include_file(self, output_dir):
        """Write the various include file in output_dir.
        The entry True of self.includepath will be written in run_card.inc
        The entry False will not be written anywhere"""
        
        # ensure that all parameter are coherent and fix those if needed
        self.check_validity()
        
        #ensusre that system only parameter are correctly set
        self.update_system_parameter_for_include()
        
        for incname in self.includepath:
            if incname is True:
                pathinc = self.default_include_file
            elif incname is False:
                continue
            else:
                pathinc = incname
                
            fsock = file_writers.FortranWriter(pjoin(output_dir,pathinc))  
            for key in self.includepath[incname]:                
                #define the fortran name
                if key in self.fortran_name:
                    fortran_name = self.fortran_name[key]
                else:
                    fortran_name = key
                    
                #get the value with warning if the user didn't set it
                value = self.get_default(key)
                # Special treatment for strings containing a list of
                # strings. Convert it to a list of strings
                if isinstance(value, list):
                    # in case of a list, add the length of the list as 0th
                    # element in fortran. Only in case of integer or float
                    # list (not for bool nor string)
                    targettype = self.list_parameter[key]                        
                    if targettype is bool:
                        pass
                    elif targettype is int:
                        line = '%s(%s) = %s \n' % (fortran_name, 0, self.f77_formatting(len(value)))
                        fsock.writelines(line)
                    elif targettype is float:
                        line = '%s(%s) = %s \n' % (fortran_name, 0, self.f77_formatting(float(len(value))))
                        fsock.writelines(line)
                    # output the rest of the list in fortran
                    for i,v in enumerate(value):
                        line = '%s(%s) = %s \n' % (fortran_name, i+1, self.f77_formatting(v))
                        fsock.writelines(line)
                elif isinstance(value, dict):
                    for fortran_name, onevalue in value.items():
                        line = '%s = %s \n' % (fortran_name, self.f77_formatting(onevalue))
                        fsock.writelines(line)                       
                else:
                    line = '%s = %s \n' % (fortran_name, self.f77_formatting(value))
                    fsock.writelines(line)
            fsock.close()   

    @staticmethod
    def get_idbmup(lpp):
        """return the particle colliding pdg code"""
        if lpp in (1,2, -1,-2):
            return math.copysign(2212, lpp)
        elif lpp in (3,-3):
            return math.copysign(11, lpp)
        elif lpp == 0:
            #logger.critical("Fail to write correct idbmup in the lhe file. Please correct those by hand")
            return 0
        else:
            return lpp

    def get_banner_init_information(self):
        """return a dictionary with the information needed to write
        the first line of the <init> block of the lhe file."""
        
        output = {}
        output["idbmup1"] = self.get_idbmup(self['lpp1'])
        output["idbmup2"] = self.get_idbmup(self['lpp2'])
        output["ebmup1"] = self["ebeam1"]
        output["ebmup2"] = self["ebeam2"]
        output["pdfgup1"] = 0
        output["pdfgup2"] = 0
        output["pdfsup1"] = self.get_pdf_id(self["pdlabel"])
        output["pdfsup2"] = self.get_pdf_id(self["pdlabel"])
        return output
    
    def get_pdf_id(self, pdf):
        if pdf == "lhapdf":
            lhaid = self["lhaid"]
            if isinstance(lhaid, list):
                return lhaid[0]
            else:
                return lhaid
        else: 
            return {'none': 0, 
                    'cteq6_m':10000,'cteq6_l':10041,'cteq6l1':10042,
                    'nn23lo':246800,'nn23lo1':247000,'nn23nlo':244800
                    }[pdf]    
    
    def get_lhapdf_id(self):
        return self.get_pdf_id(self['pdlabel'])

    def remove_all_cut(self): 
        """remove all the cut"""

        for name in self.cuts_parameter:
            targettype = type(self[name])
            if targettype == bool:
                self[name] = False
            elif 'min' in name:
                self[name] = 0
            elif 'max' in name:
                self[name] = -1
            elif 'eta' in name:
                self[name] = -1
            else:
                self[name] = 0       

class RunCardLO(RunCard):
    """an object to handle in a nice way the run_card information"""
    
    blocks = [
#    HEAVY ION OPTIONAL BLOCK            
        runblock(name='ion_pdf', fields=('nb_neutron1', 'nb_neutron2','nb_proton1','nb_proton2','mass_ion1', 'mass_ion2'),
            template_on=\
"""#*********************************************************************
# Heavy ion PDF / rescaling of PDF                                   *
#*********************************************************************
  %(nb_proton1)s    = nb_proton1 # number of proton for the first beam
  %(nb_neutron1)s    = nb_neutron1 # number of neutron for the first beam
  %(mass_ion1)s = mass_ion1 # mass of the heavy ion (first beam)
# Note that seting differently the two beams only work if you use 
# group_subprocess=False when generating your matrix-element
  %(nb_proton2)s    = nb_proton2 # number of proton for the second beam
  %(nb_neutron2)s    = nb_neutron2 # number of neutron for the second beam
  %(mass_ion2)s = mass_ion2 # mass of the heavy ion (second beam)  
""",
            template_off='# To see heavy ion options: type "update ion_pdf"'),
              
              
#    BEAM POLARIZATION OPTIONAL BLOCK
        runblock(name='beam_pol', fields=('polbeam1','polbeam2'),
            template_on=\
"""#*********************************************************************
# Beam polarization from -100 (left-handed) to 100 (right-handed)    *
#*********************************************************************
     %(polbeam1)s     = polbeam1 ! beam polarization for beam 1
     %(polbeam2)s     = polbeam2 ! beam polarization for beam 2
""",                                               
            template_off='# To see polarised beam options: type "update beam_pol"'),

#    SYSCALC OPTIONAL BLOCK              
        runblock(name='syscalc', fields=('sys_scalefact', 'sys_alpsfact','sys_matchscale','sys_pdf'),
              template_on=\
"""#**************************************
# Parameter below of the systematics study
#  will be used by SysCalc (if installed)
#**************************************
#
%(sys_scalefact)s = sys_scalefact  # factorization/renormalization scale factor
%(sys_alpsfact)s = sys_alpsfact  # \alpha_s emission scale factors
%(sys_matchscale)s = sys_matchscale # variation of merging scale
# PDF sets and number of members (0 or none for all members).
%(sys_pdf)s = sys_pdf # list of pdf sets. (errorset not valid for syscalc)
# MSTW2008nlo68cl.LHgrid 1  = sys_pdf
#
""", 
    template_off= '# Syscalc is deprecated but to see the associate options type\'update syscalc\''),
    ]
    
    
    
    def default_setup(self):
        """default value for the run_card.dat"""
        
        self.add_param("run_tag", "tag_1", include=False)
        self.add_param("gridpack", False)
        self.add_param("time_of_flight", -1.0, include=False)
        self.add_param("nevents", 10000)        
        self.add_param("iseed", 0)
        self.add_param("python_seed", -1, include=False, hidden=True, comment="controlling python seed [handling in particular the final unweighting].\n -1 means use default from random module.\n -2 means set to same value as iseed")
        self.add_param("lpp1", 1, fortran_name="lpp(1)", allowed=[-1,1,0,2,3,9, -2,-3],
                        comment='first beam energy distribution:\n 0: fixed energy\n 1: PDF from proton\n -1: PDF from anti-proton\n 2:photon from proton, 3:photon from electron, 9: PLUGIN MODE')
        self.add_param("lpp2", 1, fortran_name="lpp(2)", allowed=[-1,1,0,2,3,9],
                       comment='first beam energy distribution:\n 0: fixed energy\n 1: PDF from proton\n -1: PDF from anti-proton\n 2:photon from proton, 3:photon from electron, 9: PLUGIN MODE')
        self.add_param("ebeam1", 6500.0, fortran_name="ebeam(1)")
        self.add_param("ebeam2", 6500.0, fortran_name="ebeam(2)")
        self.add_param("polbeam1", 0.0, fortran_name="pb1", hidden=True,
                                              comment="Beam polarization from -100 (left-handed) to 100 (right-handed) --use lpp=0 for this parameter--")
        self.add_param("polbeam2", 0.0, fortran_name="pb2", hidden=True,
                                              comment="Beam polarization from -100 (left-handed) to 100 (right-handed) --use lpp=0 for this parameter--")
        self.add_param('nb_proton1', 1, hidden=True, allowed=[1,0, 82 , '*'],fortran_name="nb_proton(1)",
                       comment='For heavy ion physics nb of proton in the ion (for both beam but if group_subprocess was False)')
        self.add_param('nb_proton2', 1, hidden=True, allowed=[1,0, 82 , '*'],fortran_name="nb_proton(2)",
                       comment='For heavy ion physics nb of proton in the ion (used for beam 2 if group_subprocess was False)')
        self.add_param('nb_neutron1', 0, hidden=True, allowed=[1,0, 126 , '*'],fortran_name="nb_neutron(1)",
                       comment='For heavy ion physics nb of neutron in the ion (for both beam but if group_subprocess was False)')
        self.add_param('nb_neutron2', 0, hidden=True, allowed=[1,0, 126 , '*'],fortran_name="nb_neutron(2)",
                       comment='For heavy ion physics nb of neutron in the ion (of beam 2 if group_subprocess was False )')        
        self.add_param('mass_ion1', -1.0, hidden=True, fortran_name="mass_ion(1)",
                       allowed=[-1,0, 0.938, 207.9766521*0.938, 0.000511, 0.105, '*'],
                       comment='For heavy ion physics mass in GeV of the ion (of beam 1)')
        self.add_param('mass_ion2', -1.0, hidden=True, fortran_name="mass_ion(2)",
                       allowed=[-1,0, 0.938, 207.9766521*0.938, 0.000511, 0.105, '*'],
                       comment='For heavy ion physics mass in GeV of the ion (of beam 2)')
        
        self.add_param("pdlabel", "nn23lo1", allowed=['lhapdf', 'cteq6_m','cteq6_l', 'cteq6l1','nn23lo', 'nn23lo1', 'nn23nlo']), 
        self.add_param("lhaid", 230000, hidden=True)
        self.add_param("fixed_ren_scale", False)
        self.add_param("fixed_fac_scale", False)
        self.add_param("scale", 91.1880)
        self.add_param("dsqrt_q2fact1", 91.1880, fortran_name="sf1")
        self.add_param("dsqrt_q2fact2", 91.1880, fortran_name="sf2")
        self.add_param("dynamical_scale_choice", -1, comment="\'-1\' is based on CKKW back clustering (following feynman diagram).\n \'1\' is the sum of transverse energy.\n '2' is HT (sum of the transverse mass)\n '3' is HT/2\n '4' is the center of mass energy",
                                                allowed=[-1,0,1,2,3,4])
        
        # Bias module options
        self.add_param("bias_module", 'None', include=False)
        self.add_param('bias_parameters', {'__type__':1.0}, include='BIAS/bias.inc')
                
        #matching
        self.add_param("scalefact", 1.0)
        self.add_param("ickkw", 0, allowed=[0,1],                               comment="\'0\' for standard fixed order computation.\n\'1\' for MLM merging activates alphas and pdf re-weighting according to a kt clustering of the QCD radiation.")
        self.add_param("highestmult", 1, fortran_name="nhmult", hidden=True)
        self.add_param("ktscheme", 1, hidden=True)
        self.add_param("alpsfact", 1.0)
        self.add_param("chcluster", False, hidden=True)
        self.add_param("pdfwgt", True, hidden=True)
        self.add_param("asrwgtflavor", 5,                                       comment = 'highest quark flavor for a_s reweighting in MLM')
        self.add_param("clusinfo", True)
        self.add_param("lhe_version", 3.0)
        self.add_param("event_norm", "average", allowed=['sum','average', 'unity'],
                        include=False, sys_default='sum')
        #cut
        self.add_param("auto_ptj_mjj", False)
        self.add_param("bwcutoff", 15.0)
        self.add_param("cut_decays", False)
        self.add_param("nhel", 0, include=False)
        #pt cut
        self.add_param("ptj", 20.0, cut=True)
        self.add_param("ptb", 0.0, cut=True)
        self.add_param("pta", 10.0, cut=True)
        self.add_param("ptl", 10.0, cut=True)
        self.add_param("misset", 0.0, cut=True)
        self.add_param("ptheavy", 0.0, cut=True,                                comment='this cut apply on particle heavier than 10 GeV')
        self.add_param("ptonium", 1.0, legacy=True)
        self.add_param("ptjmax", -1.0, cut=True)
        self.add_param("ptbmax", -1.0, cut=True)
        self.add_param("ptamax", -1.0, cut=True)
        self.add_param("ptlmax", -1.0, cut=True)
        self.add_param("missetmax", -1.0, cut=True)
        # E cut
        self.add_param("ej", 0.0, cut=True)
        self.add_param("eb", 0.0, cut=True)
        self.add_param("ea", 0.0, cut=True)
        self.add_param("el", 0.0, cut=True)
        self.add_param("ejmax", -1.0, cut=True)
        self.add_param("ebmax", -1.0, cut=True)
        self.add_param("eamax", -1.0, cut=True)
        self.add_param("elmax", -1.0, cut=True)
        # Eta cut
        self.add_param("etaj", 5.0, cut=True)
        self.add_param("etab", -1.0, cut=True)
        self.add_param("etaa", 2.5, cut=True)
        self.add_param("etal", 2.5, cut=True)
        self.add_param("etaonium", 0.6, legacy=True)
        self.add_param("etajmin", 0.0, cut=True)
        self.add_param("etabmin", 0.0, cut=True)
        self.add_param("etaamin", 0.0, cut=True)
        self.add_param("etalmin", 0.0, cut=True)
        # DRJJ
        self.add_param("drjj", 0.4, cut=True)
        self.add_param("drbb", 0.0, cut=True)
        self.add_param("drll", 0.4, cut=True)
        self.add_param("draa", 0.4, cut=True)
        self.add_param("drbj", 0.0, cut=True)
        self.add_param("draj", 0.4, cut=True)
        self.add_param("drjl", 0.4, cut=True)
        self.add_param("drab", 0.0, cut=True)
        self.add_param("drbl", 0.0, cut=True)
        self.add_param("dral", 0.4, cut=True)
        self.add_param("drjjmax", -1.0, cut=True)
        self.add_param("drbbmax", -1.0, cut=True)
        self.add_param("drllmax", -1.0, cut=True)
        self.add_param("draamax", -1.0, cut=True)
        self.add_param("drbjmax", -1.0, cut=True)
        self.add_param("drajmax", -1.0, cut=True)
        self.add_param("drjlmax", -1.0, cut=True)
        self.add_param("drabmax", -1.0, cut=True)
        self.add_param("drblmax", -1.0, cut=True)
        self.add_param("dralmax", -1.0, cut=True)
        # invariant mass
        self.add_param("mmjj", 0.0, cut=True)
        self.add_param("mmbb", 0.0, cut=True)
        self.add_param("mmaa", 0.0, cut=True)
        self.add_param("mmll", 0.0, cut=True)
        self.add_param("mmjjmax", -1.0, cut=True)
        self.add_param("mmbbmax", -1.0, cut=True)                
        self.add_param("mmaamax", -1.0, cut=True)
        self.add_param("mmllmax", -1.0, cut=True)
        self.add_param("mmnl", 0.0, cut=True)
        self.add_param("mmnlmax", -1.0, cut=True)
        #minimum/max pt for sum of leptons
        self.add_param("ptllmin", 0.0, cut=True)
        self.add_param("ptllmax", -1.0, cut=True)
        self.add_param("xptj", 0.0, cut=True)
        self.add_param("xptb", 0.0, cut=True)
        self.add_param("xpta", 0.0, cut=True) 
        self.add_param("xptl", 0.0, cut=True)
        # ordered pt jet 
        self.add_param("ptj1min", 0.0, cut=True)
        self.add_param("ptj1max", -1.0, cut=True)
        self.add_param("ptj2min", 0.0, cut=True)
        self.add_param("ptj2max", -1.0, cut=True)
        self.add_param("ptj3min", 0.0, cut=True)
        self.add_param("ptj3max", -1.0, cut=True)
        self.add_param("ptj4min", 0.0, cut=True)
        self.add_param("ptj4max", -1.0, cut=True)                
        self.add_param("cutuse", 0, cut=True)
        # ordered pt lepton
        self.add_param("ptl1min", 0.0, cut=True)
        self.add_param("ptl1max", -1.0, cut=True)
        self.add_param("ptl2min", 0.0, cut=True)
        self.add_param("ptl2max", -1.0, cut=True)
        self.add_param("ptl3min", 0.0, cut=True)
        self.add_param("ptl3max", -1.0, cut=True)        
        self.add_param("ptl4min", 0.0, cut=True)
        self.add_param("ptl4max", -1.0, cut=True)
        # Ht sum of jets
        self.add_param("htjmin", 0.0, cut=True)
        self.add_param("htjmax", -1.0, cut=True)
        self.add_param("ihtmin", 0.0, cut=True)
        self.add_param("ihtmax", -1.0, cut=True)
        self.add_param("ht2min", 0.0, cut=True) 
        self.add_param("ht3min", 0.0, cut=True)
        self.add_param("ht4min", 0.0, cut=True)
        self.add_param("ht2max", -1.0, cut=True)
        self.add_param("ht3max", -1.0, cut=True)
        self.add_param("ht4max", -1.0, cut=True)
        # photon isolation
        self.add_param("ptgmin", 0.0, cut=True)
        self.add_param("r0gamma", 0.4)
        self.add_param("xn", 1.0)
        self.add_param("epsgamma", 1.0) 
        self.add_param("isoem", True)
        self.add_param("xetamin", 0.0, cut=True)
        self.add_param("deltaeta", 0.0, cut=True)
        self.add_param("ktdurham", -1.0, fortran_name="kt_durham", cut=True)
        self.add_param("dparameter", 0.4, fortran_name="d_parameter", cut=True)
        self.add_param("ptlund", -1.0, fortran_name="pt_lund", cut=True)
        self.add_param("pdgs_for_merging_cut", [21, 1, 2, 3, 4, 5, 6])
        self.add_param("maxjetflavor", 4)
        self.add_param("xqcut", 0.0, cut=True)
        self.add_param("use_syst", True)
        self.add_param('systematics_program', 'systematics', include=False, hidden=True, comment='Choose which program to use for systematics computation: none, systematics, syscalc')
        self.add_param('systematics_arguments', ['--mur=0.5,1,2', '--muf=0.5,1,2', '--pdf=errorset'], include=False, hidden=True, comment='Choose the argment to pass to the systematics command. like --mur=0.25,1,4. Look at the help of the systematics function for more details.')
        
        self.add_param("sys_scalefact", "0.5 1 2", include=False, hidden=True)
        self.add_param("sys_alpsfact", "None", include=False, hidden=True)
        self.add_param("sys_matchscale", "auto", include=False, hidden=True)
        self.add_param("sys_pdf", "errorset", include=False, hidden=True)
        self.add_param("sys_scalecorrelation", -1, include=False, hidden=True)

        #parameter not in the run_card by default
        self.add_param('gridrun', False, hidden=True)
        self.add_param('fixed_couplings', True, hidden=True)
        self.add_param('mc_grouped_subproc', True, hidden=True)
        self.add_param('xmtcentral', 0.0, hidden=True, fortran_name="xmtc")
        self.add_param('d', 1.0, hidden=True)
        self.add_param('gseed', 0, hidden=True, include=False)
        self.add_param('issgridfile', '', hidden=True)
        #job handling of the survey/ refine
        self.add_param('job_strategy', 0, hidden=True, include=False, allowed=[0,1,2], comment='see appendix of 1507.00020 (page 26)')
        self.add_param('survey_splitting', -1, hidden=True, include=False, comment="for loop-induced control how many core are used at survey for the computation of a single iteration.")
        self.add_param('survey_nchannel_per_job', 2, hidden=True, include=False, comment="control how many Channel are integrated inside a single job on cluster/multicore")
        self.add_param('refine_evt_by_job', -1, hidden=True, include=False, comment="control the maximal number of events for the first iteration of the refine (larger means less jobs)")
        
        # parameter allowing to define simple cut via the pdg
        # Special syntax are related to those. (can not be edit directly)
        self.add_param('pt_min_pdg',{'__type__':0.}, include=False)
        self.add_param('pt_max_pdg',{'__type__':0.}, include=False)
        self.add_param('E_min_pdg',{'__type__':0.}, include=False)
        self.add_param('E_max_pdg',{'__type__':0.}, include=False)
        self.add_param('eta_min_pdg',{'__type__':0.}, include=False)
        self.add_param('eta_max_pdg',{'__type__':0.}, include=False)
        self.add_param('mxx_min_pdg',{'__type__':0.}, include=False)
        self.add_param('mxx_only_part_antipart', {'default':False}, include=False)
        
        self.add_param('pdg_cut',[0],  system=True) # store which PDG are tracked
        self.add_param('ptmin4pdg',[0.], system=True) # store pt min
        self.add_param('ptmax4pdg',[-1.], system=True)
        self.add_param('Emin4pdg',[0.], system=True) # store pt min
        self.add_param('Emax4pdg',[-1.], system=True)  
        self.add_param('etamin4pdg',[0.], system=True) # store pt min
        self.add_param('etamax4pdg',[-1.], system=True)   
        self.add_param('mxxmin4pdg',[-1.], system=True)
        self.add_param('mxxpart_antipart', [False], system=True)
            
             
    def check_validity(self):
        """ """
        
        super(RunCardLO, self).check_validity()
        
        #Make sure that nhel is only either 0 (i.e. no MC over hel) or
        #1 (MC over hel with importance sampling). In particular, it can
        #no longer be > 1.
        if 'nhel' not in self.user_set:
            raise InvalidRunCard, "Parameter nhel is not defined in the run_card."
        if self['nhel'] not in [1,0]:
            raise InvalidRunCard, "Parameter nhel can only be '0' or '1', "+\
                                                          "not %s." % self['nhel']
        if int(self['maxjetflavor']) > 6:
            raise InvalidRunCard, 'maxjetflavor should be lower than 5! (6 is partly supported)'
  
        if len(self['pdgs_for_merging_cut']) > 1000:
            raise InvalidRunCard, "The number of elements in "+\
                               "'pdgs_for_merging_cut' should not exceed 1000."
  
        # some cut need to be deactivated in presence of isolation
        if self['ptgmin'] > 0:
            if self['pta'] > 0:
                logger.warning('pta cut discarded since photon isolation is used')
                self['pta'] = 0.0
            if self['draj'] > 0:
                logger.warning('draj cut discarded since photon isolation is used')
                self['draj'] = 0.0   
        
        # special treatment for gridpack use the gseed instead of the iseed        
        if self['gridrun']:
            self['iseed'] = self['gseed']
        
        #Some parameter need to be fixed when using syscalc
        if self['use_syst']:
            if self['scalefact'] != 1.0:
                logger.warning('Since use_syst=T, We change the value of \'scalefact\' to 1')
                self['scalefact'] = 1.0
     
        # CKKW Treatment
        if self['ickkw'] > 0:
            if self['ickkw'] != 1:
                logger.critical('ickkw >1 is pure alpha and only partly implemented.')
                import madgraph.interface.extended_cmd as basic_cmd
                answer = basic_cmd.smart_input('Do you really want to continue', allow_arg=['y','n'], default='n')
                if answer !='y':
                    raise InvalidRunCard, 'ickkw>1 is still in alpha'
            if self['use_syst']:
                # some additional parameter need to be fixed for Syscalc + matching
                if self['alpsfact'] != 1.0:
                    logger.warning('Since use_syst=T, We change the value of \'alpsfact\' to 1')
                    self['alpsfact'] =1.0
            if self['maxjetflavor'] == 6:
                raise InvalidRunCard, 'maxjetflavor at 6 is NOT supported for matching!'
            if self['ickkw'] == 2:
                # add warning if ckkw selected but the associate parameter are empty
                self.get_default('highestmult', log_level=20)                   
                self.get_default('issgridfile', 'issudgrid.dat', log_level=20)
        if self['xqcut'] > 0:
            if self['ickkw'] == 0:
                logger.error('xqcut>0 but ickkw=0. Potentially not fully consistent setup. Be carefull')
                import time
                time.sleep(5)
            if self['drjj'] != 0:
                logger.warning('Since icckw>0, We change the value of \'drjj\' to 0')
                self['drjj'] = 0
            if self['drjl'] != 0:
                logger.warning('Since icckw>0, We change the value of \'drjl\' to 0')
                self['drjl'] = 0    
            if not self['auto_ptj_mjj']:         
                if self['mmjj'] > self['xqcut']:
                    logger.warning('mmjj > xqcut (and auto_ptj_mjj = F). MMJJ set to 0')
                    self['mmjj'] = 0.0 


    
        # check validity of the pdf set
        if self['pdlabel'] == 'lhapdf':
            #add warning if lhaid not define
            self.get_default('lhaid', log_level=20)
   
    def update_system_parameter_for_include(self):
        
        # set the pdg_for_cut fortran parameter
        pdg_to_cut = set(self['pt_min_pdg'].keys() +self['pt_max_pdg'].keys() + 
                         self['e_min_pdg'].keys() +self['e_max_pdg'].keys() +
                         self['eta_min_pdg'].keys() +self['eta_max_pdg'].keys()+
                         self['mxx_min_pdg'].keys() + self['mxx_only_part_antipart'].keys())
        pdg_to_cut.discard('__type__')
        pdg_to_cut.discard('default')
        if len(pdg_to_cut)>25:
            raise Exception, "Maximum 25 different pdgs are allowed for pdg specific cut"
        
        if any(int(pdg)<0 for pdg in pdg_to_cut):
            logger.warning('PDG specific cuts are always applied symmetrically on particle/anti-particle. Always use positve PDG codes')
            raise MadGraph5Error, 'Some PDG specific cuts are defined with negative pdg code'
        
        
        if any(pdg in pdg_to_cut for pdg in [1,2,3,4,5,21,22,11,13,15]):
            raise Exception, "Can not use PDG related cut for light quark/b quark/lepton/gluon/photon"
        
        if pdg_to_cut:
            self['pdg_cut'] = list(pdg_to_cut)
            self['ptmin4pdg'] = []
            self['Emin4pdg'] = []
            self['etamin4pdg'] =[]
            self['ptmax4pdg'] = []
            self['Emax4pdg'] = []
            self['etamax4pdg'] =[]
            self['mxxmin4pdg'] =[]
            self['mxxpart_antipart']  = []
            for pdg in self['pdg_cut']:
                for var in ['pt','e','eta', 'Mxx']:
                    for minmax in ['min', 'max']:
                        if var in ['Mxx'] and minmax =='max':
                            continue
                        new_var = '%s%s4pdg' % (var, minmax)
                        old_var = '%s_%s_pdg' % (var, minmax)
                        default = 0. if minmax=='min' else -1.
                        self[new_var].append(self[old_var][str(pdg)] if str(pdg) in self[old_var] else default)
                #special for mxx_part_antipart
                old_var = 'mxx_only_part_antipart'
                new_var = 'mxxpart_antipart'
                if 'default' in self[old_var]:
                    default = self[old_var]['default']
                    self[new_var].append(self[old_var][str(pdg)] if str(pdg) in self[old_var] else default)
                else:
                    if str(pdg) not in self[old_var]:
                        raise Exception("no default value defined for %s and no value defined for pdg %s" % (old_var, pdg)) 
                    self[new_var].append(self[old_var][str(pdg)])
        else:
            self['pdg_cut'] = [0]
            self['ptmin4pdg'] = [0.]
            self['Emin4pdg'] = [0.]
            self['etamin4pdg'] =[0.]
            self['ptmax4pdg'] = [-1.]
            self['Emax4pdg'] = [-1.]
            self['etamax4pdg'] =[-1.]
            self['mxxmin4pdg'] =[0.] 
            self['mxxpart_antipart'] = [False]
            
                    
           
    def create_default_for_process(self, proc_characteristic, history, proc_def):
        """Rules
          process 1->N all cut set on off.
          loop_induced -> MC over helicity
          e+ e- beam -> lpp:0 ebeam:500
          p p beam -> set maxjetflavor automatically
          more than one multiplicity: ickkw=1 xqcut=30 use_syst=F
         """

        if proc_characteristic['loop_induced']:
            self['nhel'] = 1
        self['pdgs_for_merging_cut'] = proc_characteristic['colored_pdgs']
                    
        if proc_characteristic['ninitial'] == 1:
            #remove all cut
            self.remove_all_cut()
            self['use_syst'] = False
        else:
            # check for beam_id
            beam_id = set()
            for proc in proc_def:
                for oneproc in proc:
                    for leg in oneproc['legs']:
                        if not leg['state']:
                            beam_id.add(leg['id'])
            if any(i in beam_id for i in [1,-1,2,-2,3,-3,4,-4,5,-5,21,22]):
                maxjetflavor = max([4]+[abs(i) for i in beam_id if  -7< i < 7])
                self['maxjetflavor'] = maxjetflavor
                self['asrwgtflavor'] = maxjetflavor
                pass
            elif 11 in beam_id or -11 in beam_id:
                self['lpp1'] = 0
                self['lpp2'] = 0
                self['ebeam1'] = 500
                self['ebeam2'] = 500
                self['use_syst'] = False
            else:
                self['lpp1'] = 0
                self['lpp2'] = 0    
                self['use_syst'] = False            
                
        # Check if need matching
        min_particle = 99
        max_particle = 0
        for proc in proc_def:
            min_particle = min(len(proc[0]['legs']), min_particle)
            max_particle = max(len(proc[0]['legs']), max_particle)
        if min_particle != max_particle:
            #take one of the process with min_particle
            for procmin in proc_def:
                if len(procmin[0]['legs']) != min_particle:
                    continue
                else:
                    idsmin = [l['id'] for l in procmin[0]['legs']]
                    break
            matching = False
            for procmax in proc_def:
                if len(procmax[0]['legs']) != max_particle:
                    continue
                idsmax =  [l['id'] for l in procmax[0]['legs']]
                for i in idsmin:
                    if i not in idsmax:
                        continue
                    else:
                        idsmax.remove(i)
                for j in idsmax:
                    if j not in [1,-1,2,-2,3,-3,4,-4,5,-5,21]:
                        break
                else:
                    # all are jet => matching is ON
                    matching=True
                    break 
            
            if matching:
                self['ickkw'] = 1
                self['xqcut'] = 30
                #self['use_syst'] = False 
                self['drjj'] = 0
                self['drjl'] = 0
                self['sys_alpsfact'] = "0.5 1 2"
                
        # For interference module, the systematics are wrong.
        # automatically set use_syst=F and set systematics_program=none
        no_systematics = False
        for proc in proc_def:
            for oneproc in proc:
                if '^2' in oneproc.nice_string():
                    no_systematics = True
                    break
            else:
                continue
            break
        if no_systematics:
            self['use_syst'] = False
            self['systematics_program'] = 'none'
            
    def write(self, output_file, template=None, python_template=False,
              **opt):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""

        if not template:
            if not MADEVENT:
                template = pjoin(MG5DIR, 'Template', 'LO', 'Cards', 
                                                        'run_card.dat')
                python_template = True
            else:
                template = pjoin(MEDIR, 'Cards', 'run_card_default.dat')
                python_template = False
       
        super(RunCardLO, self).write(output_file, template=template,
                                    python_template=python_template, **opt)            


class InvalidMadAnalysis5Card(InvalidCmd):
    pass

class MadAnalysis5Card(dict):
    """ A class to store a MadAnalysis5 card. Very basic since it is basically
    free format."""
    
    _MG5aMC_escape_tag = '@MG5aMC'
    
    _default_hadron_inputs = ['*.hepmc', '*.hep', '*.stdhep', '*.lhco','*.root']
    _default_parton_inputs = ['*.lhe']
    _skip_analysis         = False
    
    @classmethod
    def events_can_be_reconstructed(cls, file_path):
        """ Checks from the type of an event file whether it can be reconstructed or not."""
        return not (file_path.endswith('.lhco') or file_path.endswith('.lhco.gz') or \
                          file_path.endswith('.root') or file_path.endswith('.root.gz'))
    
    @classmethod
    def empty_analysis(cls):
        """ A method returning the structure of an empty analysis """
        return {'commands':[],
                'reconstructions':[]}

    @classmethod
    def empty_reconstruction(cls):
        """ A method returning the structure of an empty reconstruction """
        return {'commands':[],
                'reco_output':'lhe'}

    def default_setup(self):
        """define the default value""" 
        self['mode']      = 'parton'
        self['inputs']    = []
        # None is the default stdout level, it will be set automatically by MG5aMC
        self['stdout_lvl'] = None
        # These two dictionaries are formated as follows:
        #     {'analysis_name':
        #          {'reconstructions' : ['associated_reconstructions_name']}
        #          {'commands':['analysis command lines here']}    }
        # with values being of the form of the empty_analysis() attribute
        # of this class and some other property could be added to this dictionary
        # in the future.
        self['analyses']       = {}
        # The recasting structure contains on set of commands and one set of 
        # card lines. 
        self['recasting']      = {'commands':[],'card':[]}
        # Add the default trivial reconstruction to use an lhco input
        # This is just for the structure
        self['reconstruction'] = {'lhco_input':
                                        MadAnalysis5Card.empty_reconstruction(),
                                  'root_input':
                                        MadAnalysis5Card.empty_reconstruction()}
        self['reconstruction']['lhco_input']['reco_output']='lhco'
        self['reconstruction']['root_input']['reco_output']='root'        

        # Specify in which order the analysis/recasting were specified
        self['order'] = []

    def __init__(self, finput=None,mode=None):
        if isinstance(finput, self.__class__):
            dict.__init__(self, finput)
            assert finput.__dict__.keys()
            for key in finput.__dict__:
                setattr(self, key, copy.copy(getattr(finput, key)) )
            return
        else:
            dict.__init__(self)
        
        # Initialize it with all the default value
        self.default_setup()
        if not mode is None:
            self['mode']=mode

        # if input is define read that input
        if isinstance(finput, (file, str, StringIO.StringIO)):
            self.read(finput, mode=mode)
    
    def read(self, input, mode=None):
        """ Read an MA5 card"""
        
        if mode not in [None,'parton','hadron']:
            raise MadGraph5Error('A MadAnalysis5Card can be read online the modes'+
                                                         "'parton' or 'hadron'")
        card_mode = mode
        
        if isinstance(input, (file, StringIO.StringIO)):
            input_stream = input
        elif isinstance(input, str):
            if not os.path.isfile(input):
                raise InvalidMadAnalysis5Card("Cannot read the MadAnalysis5 card."+\
                                                    "File '%s' not found."%input)
            if mode is None and 'hadron' in input:
                card_mode = 'hadron'
            input_stream = open(input,'r')
        else:
            raise MadGraph5Error('Incorrect input for the read function of'+\
              ' the MadAnalysis5Card card. Received argument type is: %s'%str(type(input)))

        # Reinstate default values
        self.__init__()
        current_name = 'default'
        current_type = 'analyses'
        for line in input_stream:
            # Skip comments for now
            if line.startswith('#'):
                continue
            if line.endswith('\n'):
                line = line[:-1]
            if line.strip()=='':
                continue
            if line.startswith(self._MG5aMC_escape_tag):
                try:
                    option,value = line[len(self._MG5aMC_escape_tag):].split('=')
                    value = value.strip()
                except ValueError:
                    option = line[len(self._MG5aMC_escape_tag):]
                option = option.strip()
                
                if option=='inputs':
                    self['inputs'].extend([v.strip() for v in value.split(',')])
                
                elif option == 'skip_analysis':
                    self._skip_analysis = True

                elif option=='stdout_lvl':
                    try: # It is likely an int
                        self['stdout_lvl']=int(value)
                    except ValueError:
                        try: # Maybe the user used something like 'logging.INFO'
                            self['stdout_lvl']=eval(value)
                        except:
                            try:
                               self['stdout_lvl']=eval('logging.%s'%value)
                            except:
                                raise InvalidMadAnalysis5Card(
                 "MA5 output level specification '%s' is incorrect."%str(value))
                
                elif option=='analysis_name':
                    current_type = 'analyses'
                    current_name = value
                    if current_name in self[current_type]:
                        raise InvalidMadAnalysis5Card(
               "Analysis '%s' already defined in MadAnalysis5 card"%current_name)
                    else:
                        self[current_type][current_name] = MadAnalysis5Card.empty_analysis()
                
                elif option=='set_reconstructions':
                    try:
                        reconstructions = eval(value)
                        if not isinstance(reconstructions, list):
                            raise
                    except:
                        raise InvalidMadAnalysis5Card("List of reconstructions"+\
                         " '%s' could not be parsed in MadAnalysis5 card."%value)
                    if current_type!='analyses' and current_name not in self[current_type]:
                        raise InvalidMadAnalysis5Card("A list of reconstructions"+\
                                   "can only be defined in the context of an "+\
                                             "analysis in a MadAnalysis5 card.")
                    self[current_type][current_name]['reconstructions']=reconstructions
                    continue
                
                elif option=='reconstruction_name':
                    current_type = 'reconstruction'
                    current_name = value
                    if current_name in self[current_type]:
                        raise InvalidMadAnalysis5Card(
               "Reconstruction '%s' already defined in MadAnalysis5 hadron card"%current_name)
                    else:
                        self[current_type][current_name] = MadAnalysis5Card.empty_reconstruction()

                elif option=='reco_output':
                    if current_type!='reconstruction' or current_name not in \
                                                         self['reconstruction']:
                        raise InvalidMadAnalysis5Card(
               "Option '%s' is only available within the definition of a reconstruction"%option)
                    if not value.lower() in ['lhe','root']:
                        raise InvalidMadAnalysis5Card(
                                  "Option '%s' can only take the values 'lhe' or 'root'"%option)
                    self['reconstruction'][current_name]['reco_output'] = value.lower()
                
                elif option.startswith('recasting'):
                    current_type = 'recasting'
                    try:
                        current_name = option.split('_')[1]
                    except:
                        raise InvalidMadAnalysis5Card('Malformed MA5 recasting option %s.'%option)
                    if len(self['recasting'][current_name])>0:
                        raise InvalidMadAnalysis5Card(
               "Only one recasting can be defined in MadAnalysis5 hadron card")
                
                else:
                    raise InvalidMadAnalysis5Card(
               "Unreckognized MG5aMC instruction in MadAnalysis5 card: '%s'"%option)
                
                if option in ['analysis_name','reconstruction_name'] or \
                                                 option.startswith('recasting'):
                    self['order'].append((current_type,current_name))
                continue

            # Add the default analysis if needed since the user does not need
            # to specify it.
            if current_name == 'default' and current_type == 'analyses' and\
                                          'default' not in self['analyses']:
                    self['analyses']['default'] = MadAnalysis5Card.empty_analysis()
                    self['order'].append(('analyses','default'))

            if current_type in ['recasting']:
                self[current_type][current_name].append(line)
            elif current_type in ['reconstruction']:
                self[current_type][current_name]['commands'].append(line)
            elif current_type in ['analyses']:
                self[current_type][current_name]['commands'].append(line)

        if 'reconstruction' in self['analyses'] or len(self['recasting']['card'])>0:
            if mode=='parton':
                raise InvalidMadAnalysis5Card(
      "A parton MadAnalysis5 card cannot specify a recombination or recasting.")
            card_mode = 'hadron'
        elif mode is None:
            card_mode = 'parton'

        self['mode'] = card_mode
        if self['inputs'] == []:
            if self['mode']=='hadron':
                self['inputs']  = self._default_hadron_inputs
            else:
                self['inputs']  = self._default_parton_inputs
        
        # Make sure at least one reconstruction is specified for each hadron
        # level analysis and that it exists.
        if self['mode']=='hadron':
            for analysis_name, analysis in self['analyses'].items():
                if len(analysis['reconstructions'])==0:
                    raise InvalidMadAnalysis5Card('Hadron-level analysis '+\
                      "'%s' is not specified any reconstruction(s)."%analysis_name)
                if any(reco not in self['reconstruction'] for reco in \
                                                   analysis['reconstructions']):
                    raise InvalidMadAnalysis5Card('A reconstructions specified in'+\
                                 " analysis '%s' is not defined."%analysis_name)
    
    def write(self, output):
        """ Write an MA5 card."""

        if isinstance(output, (file, StringIO.StringIO)):
            output_stream = output
        elif isinstance(output, str):
            output_stream = open(output,'w')
        else:
            raise MadGraph5Error('Incorrect input for the write function of'+\
              ' the MadAnalysis5Card card. Received argument type is: %s'%str(type(output)))
        
        output_lines = []
        if self._skip_analysis:
            output_lines.append('%s skip_analysis'%self._MG5aMC_escape_tag)
        output_lines.append('%s inputs = %s'%(self._MG5aMC_escape_tag,','.join(self['inputs'])))
        if not self['stdout_lvl'] is None:
            output_lines.append('%s stdout_lvl=%s'%(self._MG5aMC_escape_tag,self['stdout_lvl']))
        for definition_type, name in self['order']:
            
            if definition_type=='analyses':
                output_lines.append('%s analysis_name = %s'%(self._MG5aMC_escape_tag,name))
                output_lines.append('%s set_reconstructions = %s'%(self._MG5aMC_escape_tag,
                                str(self['analyses'][name]['reconstructions'])))                
            elif definition_type=='reconstruction':
                output_lines.append('%s reconstruction_name = %s'%(self._MG5aMC_escape_tag,name))
            elif definition_type=='recasting':
                output_lines.append('%s recasting_%s'%(self._MG5aMC_escape_tag,name))

            if definition_type in ['recasting']:
                output_lines.extend(self[definition_type][name])
            elif definition_type in ['reconstruction']:
                output_lines.append('%s reco_output = %s'%(self._MG5aMC_escape_tag,
                                    self[definition_type][name]['reco_output']))                
                output_lines.extend(self[definition_type][name]['commands'])
            elif definition_type in ['analyses']:
                output_lines.extend(self[definition_type][name]['commands'])                
        
        output_stream.write('\n'.join(output_lines))
        
        return
    
    def get_MA5_cmds(self, inputs_arg, submit_folder, run_dir_path=None, 
                                               UFO_model_path=None, run_tag=''):
        """ Returns a list of tuples ('AnalysisTag',['commands']) specifying 
        the commands of the MadAnalysis runs required from this card. 
        At parton-level, the number of such commands is the number of analysis 
        asked for. In the future, the idea is that the entire card can be
        processed in one go from MA5 directly."""
        
        if isinstance(inputs_arg, list):
            inputs = inputs_arg
        elif isinstance(inputs_arg, str):
            inputs = [inputs_arg]
        else:
            raise MadGraph5Error("The function 'get_MA5_cmds' can only take "+\
                            " a string or a list for the argument 'inputs_arg'")
        
        if len(inputs)==0:
            raise MadGraph5Error("The function 'get_MA5_cmds' must have "+\
                                              " at least one input specified'")
        
        if run_dir_path is None:
            run_dir_path = os.path.dirname(inputs_arg)
        
        cmds_list = []
        
        UFO_load = []
        # first import the UFO if provided
        if UFO_model_path:
            UFO_load.append('import %s'%UFO_model_path)
        
        def get_import(input, type=None):
            """ Generates the MA5 import commands for that event file. """
            dataset_name = os.path.basename(input).split('.')[0]
            res = ['import %s as %s'%(input, dataset_name)]
            if not type is None:
                res.append('set %s.type = %s'%(dataset_name, type))
            return res
        
        fifo_status = {'warned_fifo':False,'fifo_used_up':False}
        def warn_fifo(input):
            if not input.endswith('.fifo'):
                return False
            if not fifo_status['fifo_used_up']:
                fifo_status['fifo_used_up'] = True
                return False
            else:
                if not fifo_status['warned_fifo']:
                    logger.warning('Only the first MA5 analysis/reconstructions can be run on a fifo. Subsequent runs will skip fifo inputs.')
                    fifo_status['warned_fifo'] = True
                return True
            
        # Then the event file(s) input(s)
        inputs_load = []
        for input in inputs:
            inputs_load.extend(get_import(input))
        
        submit_command = 'submit %s'%submit_folder+'_%s'
        
        # Keep track of the reconstruction outpus in the MA5 workflow
        # Keys are reconstruction names and values are .lhe.gz reco file paths.
        # We put by default already the lhco/root ones present
        reconstruction_outputs = {
                'lhco_input':[f for f in inputs if 
                                 f.endswith('.lhco') or f.endswith('.lhco.gz')],
                'root_input':[f for f in inputs if 
                                 f.endswith('.root') or f.endswith('.root.gz')]}

        # If a recasting card has to be written out, chose here its path
        recasting_card_path = pjoin(run_dir_path,
       '_'.join([run_tag,os.path.basename(submit_folder),'recasting_card.dat']))

        # Make sure to only run over one analysis over each fifo.
        for definition_type, name in self['order']:
            if definition_type == 'reconstruction':   
                analysis_cmds = list(self['reconstruction'][name]['commands'])
                reco_outputs = []
                for i_input, input in enumerate(inputs):
                    # Skip lhco/root as they must not be reconstructed
                    if not MadAnalysis5Card.events_can_be_reconstructed(input):
                        continue
                    # Make sure the input is not a used up fifo.
                    if warn_fifo(input):
                        continue
                    analysis_cmds.append('import %s as reco_events'%input)
                    if self['reconstruction'][name]['reco_output']=='lhe':
                        reco_outputs.append('%s_%s.lhe.gz'%(os.path.basename(
                               input).replace('_events','').split('.')[0],name))
                        analysis_cmds.append('set main.outputfile=%s'%reco_outputs[-1])
                    elif self['reconstruction'][name]['reco_output']=='root':
                        reco_outputs.append('%s_%s.root'%(os.path.basename(
                               input).replace('_events','').split('.')[0],name))
                        analysis_cmds.append('set main.fastsim.rootfile=%s'%reco_outputs[-1])
                    analysis_cmds.append(
                                 submit_command%('reco_%s_%d'%(name,i_input+1)))
                    analysis_cmds.append('remove reco_events')
                    
                reconstruction_outputs[name]= [pjoin(run_dir_path,rec_out) 
                                                    for rec_out in reco_outputs]
                if len(reco_outputs)>0:
                    cmds_list.append(('_reco_%s'%name,analysis_cmds))

            elif definition_type == 'analyses':
                if self['mode']=='parton':
                    cmds_list.append( (name, UFO_load+inputs_load+
                      self['analyses'][name]['commands']+[submit_command%name]) )
                elif self['mode']=='hadron':
                    # Also run on the already reconstructed root/lhco files if found.
                    for reco in self['analyses'][name]['reconstructions']+\
                                                    ['lhco_input','root_input']:
                        if len(reconstruction_outputs[reco])==0:
                            continue
                        if self['reconstruction'][reco]['reco_output']=='lhe':
                            # For the reconstructed lhe output we must be in parton mode
                            analysis_cmds = ['set main.mode = parton']
                        else:
                            analysis_cmds = []
                        analysis_cmds.extend(sum([get_import(rec_out) for 
                                   rec_out in reconstruction_outputs[reco]],[]))
                        analysis_cmds.extend(self['analyses'][name]['commands'])
                        analysis_cmds.append(submit_command%('%s_%s'%(name,reco)))
                        cmds_list.append( ('%s_%s'%(name,reco),analysis_cmds)  )

            elif definition_type == 'recasting':
                if len(self['recasting']['card'])==0:
                    continue
                if name == 'card':
                    # Create the card here
                    open(recasting_card_path,'w').write('\n'.join(self['recasting']['card']))
                if name == 'commands':
                    recasting_cmds = list(self['recasting']['commands'])
                    # Exclude LHCO files here of course
                    n_inputs = 0
                    for input in inputs:
                        if not MadAnalysis5Card.events_can_be_reconstructed(input):
                            continue
                        # Make sure the input is not a used up fifo.
                        if warn_fifo(input):
                            continue
                        recasting_cmds.extend(get_import(input,'signal'))
                        n_inputs += 1

                    recasting_cmds.append('set main.recast.card_path=%s'%recasting_card_path)
                    recasting_cmds.append(submit_command%'Recasting')
                    if n_inputs>0:
                        cmds_list.append( ('Recasting',recasting_cmds))

        return cmds_list

class RunCardNLO(RunCard):
    """A class object for the run_card for a (aMC@)NLO pocess"""
    
    def default_setup(self):
        """define the default value"""
        
        self.add_param('run_tag', 'tag_1', include=False)
        self.add_param('nevents', 10000)
        self.add_param('req_acc', -1.0, include=False)
        self.add_param('nevt_job', -1, include=False)
        self.add_param('event_norm', 'average')
        #FO parameter
        self.add_param('req_acc_fo', 0.01, include=False)        
        self.add_param('npoints_fo_grid', 5000, include=False)
        self.add_param('niters_fo_grid', 4, include=False)
        self.add_param('npoints_fo', 10000, include=False)        
        self.add_param('niters_fo', 6, include=False)
        #seed and collider
        self.add_param('iseed', 0)
        self.add_param('lpp1', 1, fortran_name='lpp(1)')        
        self.add_param('lpp2', 1, fortran_name='lpp(2)')                        
        self.add_param('ebeam1', 6500.0, fortran_name='ebeam(1)')
        self.add_param('ebeam2', 6500.0, fortran_name='ebeam(2)')        
        self.add_param('pdlabel', 'nn23nlo', allowed=['lhapdf', 'cteq6_m','cteq6_d','cteq6_l','cteq6l1', 'nn23lo','nn23lo1','nn23nlo'])                
        self.add_param('lhaid', [244600],fortran_name='lhaPDFid')
        self.add_param('lhapdfsetname', ['internal_use_only'], system=True)
        #shower and scale
        self.add_param('parton_shower', 'HERWIG6', fortran_name='shower_mc')        
        self.add_param('shower_scale_factor',1.0)
        self.add_param('fixed_ren_scale', False)
        self.add_param('fixed_fac_scale', False)
        self.add_param('mur_ref_fixed', 91.118)                       
        self.add_param('muf1_ref_fixed', -1.0, hidden=True)
        self.add_param('muf_ref_fixed', 91.118)                       
        self.add_param('muf2_ref_fixed', -1.0, hidden=True)
        self.add_param("dynamical_scale_choice", [-1],fortran_name='dyn_scale', comment="\'-1\' is based on CKKW back clustering (following feynman diagram).\n \'1\' is the sum of transverse energy.\n '2' is HT (sum of the transverse mass)\n '3' is HT/2")
        self.add_param('fixed_qes_scale', False, hidden=True)
        self.add_param('qes_ref_fixed', -1.0, hidden=True)
        self.add_param('mur_over_ref', 1.0)
        self.add_param('muf_over_ref', 1.0)                       
        self.add_param('muf1_over_ref', -1.0, hidden=True)                       
        self.add_param('muf2_over_ref', -1.0, hidden=True)
        self.add_param('qes_over_ref', -1.0, hidden=True)
        self.add_param('reweight_scale', [True], fortran_name='lscalevar')
        self.add_param('rw_rscale_down', -1.0, hidden=True)        
        self.add_param('rw_rscale_up', -1.0, hidden=True)
        self.add_param('rw_fscale_down', -1.0, hidden=True)                       
        self.add_param('rw_fscale_up', -1.0, hidden=True)
        self.add_param('rw_rscale', [1.0,2.0,0.5], fortran_name='scalevarR')
        self.add_param('rw_fscale', [1.0,2.0,0.5], fortran_name='scalevarF')
        self.add_param('reweight_pdf', [False], fortran_name='lpdfvar')
        self.add_param('pdf_set_min', 244601, hidden=True)
        self.add_param('pdf_set_max', 244700, hidden=True)
        self.add_param('store_rwgt_info', False)
        self.add_param('systematics_program', 'none', include=False, hidden=True, comment='Choose which program to use for systematics computation: none, systematics')
        self.add_param('systematics_arguments', [''], include=False, hidden=True, comment='Choose the argment to pass to the systematics command. like --mur=0.25,1,4. Look at the help of the systematics function for more details.')
             
        #merging
        self.add_param('ickkw', 0)
        self.add_param('bwcutoff', 15.0)
        #cuts        
        self.add_param('jetalgo', 1.0)
        self.add_param('jetradius', 0.7)         
        self.add_param('ptj', 10.0 , cut=True)
        self.add_param('etaj', -1.0, cut=True)        
        self.add_param('ptl', 0.0, cut=True)
        self.add_param('etal', -1.0, cut=True) 
        self.add_param('drll', 0.0, cut=True)
        self.add_param('drll_sf', 0.0, cut=True)        
        self.add_param('mll', 0.0, cut=True)
        self.add_param('mll_sf', 30.0, cut=True) 
        self.add_param('ptgmin', 20.0, cut=True)
        self.add_param('etagamma', -1.0)        
        self.add_param('r0gamma', 0.4)
        self.add_param('xn', 1.0)                         
        self.add_param('epsgamma', 1.0)
        self.add_param('isoem', True)        
        self.add_param('maxjetflavor', 4, hidden=True)
        self.add_param('iappl', 0)   
        self.add_param('lhe_version', 3, hidden=True, include=False)
        
        #internal variable related to FO_analyse_card
        self.add_param('FO_LHE_weight_ratio',1e-3, hidden=True, system=True)
        self.add_param('FO_LHE_postprocessing',['grouping','random'], 
                       hidden=True, system=True, include=False)
    
        # parameter allowing to define simple cut via the pdg
        self.add_param('pt_min_pdg',{'__type__':0.}, include=False)
        self.add_param('pt_max_pdg',{'__type__':0.}, include=False)
        self.add_param('mxx_min_pdg',{'__type__':0.}, include=False)
        self.add_param('mxx_only_part_antipart', {'default':False}, include=False, hidden=True)
        
        #hidden parameter that are transfer to the fortran code
        self.add_param('pdg_cut',[0], hidden=True, system=True) # store which PDG are tracked
        self.add_param('ptmin4pdg',[0.], hidden=True, system=True) # store pt min
        self.add_param('ptmax4pdg',[-1.], hidden=True, system=True)
        self.add_param('mxxmin4pdg',[0.], hidden=True, system=True)
        self.add_param('mxxpart_antipart', [False], hidden=True, system=True)
        
    def check_validity(self):
        """check the validity of the various input"""
        
        super(RunCardNLO, self).check_validity()

        # for lepton-lepton collisions, ignore 'pdlabel' and 'lhaid'
        if self['lpp1']==0 and self['lpp2']==0:
            if self['pdlabel']!='nn23nlo' or self['reweight_pdf']:
                self['pdlabel']='nn23nlo'
                self['reweight_pdf']=[False]
                logger.info('''Lepton-lepton collisions: ignoring PDF related parameters in the run_card.dat (pdlabel, lhaid, reweight_pdf, ...)''')
        
        # For FxFx merging, make sure that the following parameters are set correctly:
        if self['ickkw'] == 3: 
            # 1. Renormalization and factorization (and ellis-sexton scales) are not fixed       
            scales=['fixed_ren_scale','fixed_fac_scale','fixed_QES_scale']
            for scale in scales:
                if self[scale]:
                    logger.warning('''For consistency in the FxFx merging, \'%s\' has been set to false'''
                                % scale,'$MG:BOLD')
                    self[scale]= False
            #and left to default dynamical scale
            if len(self["dynamical_scale_choice"]) > 1 or self["dynamical_scale_choice"][0] != -1:
                self["dynamical_scale_choice"] = [-1]
                self["reweight_scale"]=[self["reweight_scale"][0]]
                logger.warning('''For consistency in the FxFx merging, dynamical_scale_choice has been set to -1 (default)'''
                                ,'$MG:BOLD')
                
            # 2. Use kT algorithm for jets with pseudo-code size R=1.0
            jetparams=['jetradius','jetalgo']
            for jetparam in jetparams:
                if float(self[jetparam]) != 1.0:
                    logger.info('''For consistency in the FxFx merging, \'%s\' has been set to 1.0'''
                                % jetparam ,'$MG:BOLD')
                    self[jetparam] = 1.0
        elif self['ickkw'] == -1 and (self["dynamical_scale_choice"][0] != -1 or
                                      len(self["dynamical_scale_choice"]) > 1):
                self["dynamical_scale_choice"] = [-1]
                self["reweight_scale"]=[self["reweight_scale"][0]]
                logger.warning('''For consistency with the jet veto, the scale which will be used is ptj. dynamical_scale_choice will be set at -1.'''
                                ,'$MG:BOLD')            
                                
        # For interface to APPLGRID, need to use LHAPDF and reweighting to get scale uncertainties
        if self['iappl'] != 0 and self['pdlabel'].lower() != 'lhapdf':
            raise InvalidRunCard('APPLgrid generation only possible with the use of LHAPDF')
        if self['iappl'] != 0 and not self['reweight_scale']:
            raise InvalidRunCard('APPLgrid generation only possible with including' +\
                                      ' the reweighting to get scale dependence')

        # Hidden values check
        if self['qes_ref_fixed'] == -1.0:
            self['qes_ref_fixed']=self['mur_ref_fixed']
        if self['qes_over_ref'] == -1.0:
            self['qes_over_ref']=self['mur_over_ref']
        if self['muf1_over_ref'] != -1.0 and self['muf1_over_ref'] == self['muf2_over_ref']:
            self['muf_over_ref']=self['muf1_over_ref']
        if self['muf1_over_ref'] == -1.0:
            self['muf1_over_ref']=self['muf_over_ref']
        if self['muf2_over_ref'] == -1.0:
            self['muf2_over_ref']=self['muf_over_ref']
        if self['muf1_ref_fixed'] != -1.0 and self['muf1_ref_fixed'] == self['muf2_ref_fixed']:
            self['muf_ref_fixed']=self['muf1_ref_fixed']
        if self['muf1_ref_fixed'] == -1.0:
            self['muf1_ref_fixed']=self['muf_ref_fixed']
        if self['muf2_ref_fixed'] == -1.0:
            self['muf2_ref_fixed']=self['muf_ref_fixed']
        # overwrite rw_rscale and rw_fscale when rw_(r/f)scale_(down/up) are explicitly given in the run_card for backward compatibility.
        if (self['rw_rscale_down'] != -1.0 and ['rw_rscale_down'] not in self['rw_rscale']) or\
           (self['rw_rscale_up'] != -1.0 and ['rw_rscale_up'] not in self['rw_rscale']):
            self['rw_rscale']=[1.0,self['rw_rscale_up'],self['rw_rscale_down']]
        if (self['rw_fscale_down'] != -1.0 and ['rw_fscale_down'] not in self['rw_fscale']) or\
           (self['rw_fscale_up'] != -1.0 and ['rw_fscale_up'] not in self['rw_fscale']):
            self['rw_fscale']=[1.0,self['rw_fscale_up'],self['rw_fscale_down']]
    
        # PDF reweighting check
        if any(self['reweight_pdf']):
            # check that we use lhapdf if reweighting is ON
            if self['pdlabel'] != "lhapdf":
                raise InvalidRunCard, 'Reweight PDF option requires to use pdf sets associated to lhapdf. Please either change the pdlabel to use LHAPDF or set reweight_pdf to False.'

        # make sure set have reweight_pdf and lhaid of length 1 when not including lhapdf
        if self['pdlabel'] != "lhapdf":
            self['reweight_pdf']=[self['reweight_pdf'][0]]
            self['lhaid']=[self['lhaid'][0]]
            
        # make sure set have reweight_scale and dyn_scale_choice of length 1 when fixed scales:
        if self['fixed_ren_scale'] and self['fixed_fac_scale']:
            self['reweight_scale']=[self['reweight_scale'][0]]
            self['dynamical_scale_choice']=[0]

        # If there is only one reweight_pdf/reweight_scale, but
        # lhaid/dynamical_scale_choice are longer, expand the
        # reweight_pdf/reweight_scale list to have the same length
        if len(self['reweight_pdf']) == 1 and len(self['lhaid']) != 1:
            self['reweight_pdf']=self['reweight_pdf']*len(self['lhaid'])
            logger.warning("Setting 'reweight_pdf' for all 'lhaid' to %s" % self['reweight_pdf'][0])
        if len(self['reweight_scale']) == 1 and len(self['dynamical_scale_choice']) != 1:
            self['reweight_scale']=self['reweight_scale']*len(self['dynamical_scale_choice']) 
            logger.warning("Setting 'reweight_scale' for all 'dynamical_scale_choice' to %s" % self['reweight_pdf'][0])

        # Check that there are no identical elements in lhaid or dynamical_scale_choice
        if len(self['lhaid']) != len(set(self['lhaid'])):
                raise InvalidRunCard, "'lhaid' has two or more identical entries. They have to be all different for the code to work correctly."
        if len(self['dynamical_scale_choice']) != len(set(self['dynamical_scale_choice'])):
                raise InvalidRunCard, "'dynamical_scale_choice' has two or more identical entries. They have to be all different for the code to work correctly."
            
        # Check that lenght of lists are consistent
        if len(self['reweight_pdf']) != len(self['lhaid']):
            raise InvalidRunCard, "'reweight_pdf' and 'lhaid' lists should have the same length"
        if len(self['reweight_scale']) != len(self['dynamical_scale_choice']):
            raise InvalidRunCard, "'reweight_scale' and 'dynamical_scale_choice' lists should have the same length"
        if len(self['dynamical_scale_choice']) > 10 :
            raise InvalidRunCard, "Length of list for 'dynamical_scale_choice' too long: max is 10."
        if len(self['lhaid']) > 25 :
            raise InvalidRunCard, "Length of list for 'lhaid' too long: max is 25."
        if len(self['rw_rscale']) > 9 :
            raise InvalidRunCard, "Length of list for 'rw_rscale' too long: max is 9."
        if len(self['rw_fscale']) > 9 :
            raise InvalidRunCard, "Length of list for 'rw_fscale' too long: max is 9."
    # make sure that the first element of rw_rscale and rw_fscale is the 1.0
        if 1.0 not in self['rw_rscale']:
            logger.warning("'1.0' has to be part of 'rw_rscale', adding it")
            self['rw_rscale'].insert(0,1.0)
        if 1.0 not in self['rw_fscale']:
            logger.warning("'1.0' has to be part of 'rw_fscale', adding it")
            self['rw_fscale'].insert(0,1.0)
        if self['rw_rscale'][0] != 1.0 and 1.0 in self['rw_rscale']:
            a=self['rw_rscale'].index(1.0)
            self['rw_rscale'][0],self['rw_rscale'][a]=self['rw_rscale'][a],self['rw_rscale'][0]
        if self['rw_fscale'][0] != 1.0 and 1.0 in self['rw_fscale']:
            a=self['rw_fscale'].index(1.0)
            self['rw_fscale'][0],self['rw_fscale'][a]=self['rw_fscale'][a],self['rw_fscale'][0]
    # check that all elements of rw_rscale and rw_fscale are diffent.
        if len(self['rw_rscale']) != len(set(self['rw_rscale'])):
                raise InvalidRunCard, "'rw_rscale' has two or more identical entries. They have to be all different for the code to work correctly."
        if len(self['rw_fscale']) != len(set(self['rw_fscale'])):
                raise InvalidRunCard, "'rw_fscale' has two or more identical entries. They have to be all different for the code to work correctly."


    def update_system_parameter_for_include(self):
        
        # set the pdg_for_cut fortran parameter
        pdg_to_cut = set(self['pt_min_pdg'].keys() +self['pt_max_pdg'].keys()+
                         self['mxx_min_pdg'].keys()+ self['mxx_only_part_antipart'].keys())
        pdg_to_cut.discard('__type__')
        pdg_to_cut.discard('default')
        if len(pdg_to_cut)>25:
            raise Exception, "Maximum 25 different PDGs are allowed for PDG specific cut"
        
        if any(int(pdg)<0 for pdg in pdg_to_cut):
            logger.warning('PDG specific cuts are always applied symmetrically on particle/anti-particle. Always use positve PDG codes')
            raise MadGraph5Error, 'Some PDG specific cuts are defined with negative PDG codes'
        
        
        if any(pdg in pdg_to_cut for pdg in [21,22,11,13,15]+ range(self['maxjetflavor']+1)):
            # Note that this will double check in the fortran code
            raise Exception, "Can not use PDG related cuts for massless SM particles/leptons"
        if pdg_to_cut:
            self['pdg_cut'] = list(pdg_to_cut)
            self['ptmin4pdg'] = []
            self['ptmax4pdg'] = []
            self['mxxmin4pdg'] = []
            self['mxxpart_antipart']  = []
            for pdg in self['pdg_cut']:
                for var in ['pt','mxx']:
                    for minmax in ['min', 'max']:
                        if var == 'mxx' and minmax == 'max':
                            continue
                        new_var = '%s%s4pdg' % (var, minmax)
                        old_var = '%s_%s_pdg' % (var, minmax)
                        default = 0. if minmax=='min' else -1.
                        self[new_var].append(self[old_var][str(pdg)] if str(pdg) in self[old_var] else default)
                #special for mxx_part_antipart
                old_var = 'mxx_only_part_antipart'
                new_var = 'mxxpart_antipart'
                if 'default' in self[old_var]:
                    default = self[old_var]['default']
                    self[new_var].append(self[old_var][str(pdg)] if str(pdg) in self[old_var] else default)
                else:
                    if str(pdg) not in self[old_var]:
                        raise Exception("no default value defined for %s and no value defined for pdg %s" % (old_var, pdg)) 
                    self[new_var].append(self[old_var][str(pdg)])
        else:
            self['pdg_cut'] = [0]
            self['ptmin4pdg'] = [0.]
            self['ptmax4pdg'] = [-1.]
            self['mxxmin4pdg'] = [0.]
            self['mxxpart_antipart'] = [False]

    def write(self, output_file, template=None, python_template=False, **opt):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""

        if not template:
            if not MADEVENT:
                template = pjoin(MG5DIR, 'Template', 'NLO', 'Cards', 
                                                        'run_card.dat')
                python_template = True
            else:
                template = pjoin(MEDIR, 'Cards', 'run_card_default.dat')
                python_template = False
       
        super(RunCardNLO, self).write(output_file, template=template,
                                    python_template=python_template, **opt)


    def create_default_for_process(self, proc_characteristic, history, proc_def):
        """Rules
          e+ e- beam -> lpp:0 ebeam:500  
          p p beam -> set maxjetflavor automatically
        """

        # check for beam_id
        beam_id = set()
        for proc in proc_def:
            for leg in proc['legs']:
                if not leg['state']:
                    beam_id.add(leg['id'])
        if any(i in beam_id for i in [1,-1,2,-2,3,-3,4,-4,5,-5,21,22]):
            maxjetflavor = max([4]+[abs(i) for i in beam_id if  -7< i < 7])
            self['maxjetflavor'] = maxjetflavor
            pass
        elif 11 in beam_id or -11 in beam_id:
            self['lpp1'] = 0
            self['lpp2'] = 0
            self['ebeam1'] = 500
            self['ebeam2'] = 500
        else:
            self['lpp1'] = 0
            self['lpp2'] = 0  
            
        if proc_characteristic['ninitial'] == 1:
            #remove all cut
            self.remove_all_cut()
    
    
    
class MadLoopParam(ConfigFile):
    """ a class for storing/dealing with the file MadLoopParam.dat
    contains a parser to read it, facilities to write a new file,...
    """
    
    _ID_reduction_tool_map = {1:'CutTools',
                             2:'PJFry++',
                             3:'IREGI',
                             4:'Golem95',
                             5:'Samurai',
                             6:'Ninja',
                             7:'COLLIER'}
    
    def default_setup(self):
        """initialize the directory to the default value"""
        
        self.add_param("MLReductionLib", "6|7|1")
        self.add_param("IREGIMODE", 2)
        self.add_param("IREGIRECY", True)
        self.add_param("CTModeRun", -1)
        self.add_param("MLStabThres", 1e-3)
        self.add_param("NRotations_DP", 0)
        self.add_param("NRotations_QP", 0)
        self.add_param("ImprovePSPoint", 2)
        self.add_param("CTLoopLibrary", 2)
        self.add_param("CTStabThres", 1e-2)
        self.add_param("CTModeInit", 1)
        self.add_param("CheckCycle", 3)
        self.add_param("MaxAttempts", 10)
        self.add_param("ZeroThres", 1e-9)
        self.add_param("OSThres", 1.0e-8)
        self.add_param("DoubleCheckHelicityFilter", True)
        self.add_param("WriteOutFilters", True)
        self.add_param("UseLoopFilter", False)
        self.add_param("HelicityFilterLevel", 2)
        self.add_param("LoopInitStartOver", False)
        self.add_param("HelInitStartOver", False)
        self.add_param("UseQPIntegrandForNinja", True)        
        self.add_param("UseQPIntegrandForCutTools", True)
        self.add_param("COLLIERMode", 1)
        self.add_param("COLLIERComputeUVpoles", True)
        self.add_param("COLLIERComputeIRpoles", True)
        self.add_param("COLLIERRequiredAccuracy", 1.0e-8)
        self.add_param("COLLIERCanOutput",False)
        self.add_param("COLLIERGlobalCache",-1)
        self.add_param("COLLIERUseCacheForPoles",False)
        self.add_param("COLLIERUseInternalStabilityTest",True)

    def read(self, finput):
        """Read the input file, this can be a path to a file, 
           a file object, a str with the content of the file."""
           
        if isinstance(finput, str):
            if "\n" in finput:
                finput = finput.split('\n')
            elif os.path.isfile(finput):
                finput = open(finput)
            else:
                raise Exception, "No such file %s" % input
        
        previous_line= ''
        for line in finput:
            if previous_line.startswith('#'):
                name = previous_line[1:].split()[0]
                value = line.strip()
                if len(value) and value[0] not in ['#', '!']:
                    self.__setitem__(name, value, change_userdefine=True)
            previous_line = line
        
    
    def write(self, outputpath, template=None,commentdefault=False):
        
        if not template:
            if not MADEVENT:
                template = pjoin(MG5DIR, 'Template', 'loop_material', 'StandAlone', 
                                                   'Cards', 'MadLoopParams.dat')
            else:
                template = pjoin(MEDIR, 'Cards', 'MadLoopParams_default.dat')
        fsock = open(template, 'r')
        template = fsock.readlines()
        fsock.close()
        
        if isinstance(outputpath, str):
            output = open(outputpath, 'w')
        else:
            output = outputpath

        def f77format(value):
            if isinstance(value, bool):
                if value:
                    return '.true.'
                else:
                    return '.false.'
            elif isinstance(value, int):
                return value
            elif isinstance(value, float):
                tmp ='%e' % value
                return tmp.replace('e','d')
            elif isinstance(value, str):
                return value
            else:
                raise Exception, "Can not format input %s" % type(value)
            
        name = ''
        done = set()
        for line in template:
            if name:
                done.add(name)
                if commentdefault and name.lower() not in self.user_set :
                    output.write('!%s\n' % f77format(self[name]))
                else:
                    output.write('%s\n' % f77format(self[name]))
                name=''
                continue
            elif line.startswith('#'):
                name = line[1:].split()[0]
            output.write(line)
        
    
            
        
        
        
        
    

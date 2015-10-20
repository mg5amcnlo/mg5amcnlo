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

from __future__ import division
import copy
import logging
import numbers
import os
import sys
import re
import math

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
    
    def get_cross(self):
        """return the cross-section of the file"""

        if "init" not in self:
            misc.sprint(self.keys())
            raise Exception
        
        text = self["init"].split('\n')
        cross = 0
        for line in text:
            s = line.split()
            if len(s)==4:
                cross += float(s[0])
        return cross
        

    
    def modify_init_cross(self, cross):
        """modify the init information with the associate cross-section"""

        assert isinstance(cross, dict)
#        assert "all" in cross
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
            if int(pid) not in cross:
                raise Exception
            pid = int(pid)
            ratio = cross[pid]/float(xsec)
            line = "   %+13.7e %+13.7e %+13.7e %i" % \
                (float(cross[pid]), ratio* float(xerr), ratio*float(xmax), pid)
            new_data.append(line)
        self['init'] = '\n'.join(new_data)
    
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
            ratio = cross[pid]/float(xsec)
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
            elif 'reweight_card' in card_name:
                tag='reweight_card'
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
        if len(arg) == 1:
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
 
 
class ConfigFile(dict):
    """ a class for storing/dealing with input file.
    """     

    def __init__(self, finput=None):
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
        self.lower_to_case = {}
        self.default_setup()

        
        
        # if input is define read that input
        if isinstance(finput, (file, str)):
            self.read(finput)
    

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
        return [(self.lower_to_case[name], value) for name,value in \
                                               super(ConfigFile, self).items()]
    
    def __setitem__(self, name, value, change_userdefine=False):
        """set the attribute and set correctly the type if the value is a string"""
        if  not len(self):
            #Should never happen but when deepcopy/pickle
            self.__init__()
            
        name = name.strip() 
        # 1. Find the type of the attribute that we want
        if name in self:
            lower_name = name.lower()
            targettype = type(self[name])
        else:
            lower_name = name.lower()          
            logger.debug('Trying to add argument %s in %s. ' % (name, self.__class__.__name__) +\
                        'This argument is not defined by default. Please consider to add it.')
            logger.debug("Did you mean %s", [k for k in self.keys() if k.startswith(name[0].lower())])
            self.add_param(lower_name, self.format_variable(str(value), str, name))
            self.lower_to_case[lower_name] = name
            if change_userdefine:
                self.user_set.add(lower_name)
            return
    
        value = self.format_variable(value, targettype, name=name)
        dict.__setitem__(self, lower_name, value)
        if change_userdefine:
            self.user_set.add(lower_name)

    def add_param(self, name, value):
        """add a default parameter to the class"""

        lower_name = name.lower()
        if __debug__:
            if lower_name in self:
                raise Exception("Duplicate case for %s in %s" % (name,self.__class__))
            
        dict.__setitem__(self, lower_name, value)
        self.lower_to_case[lower_name] = name

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
                    raise Exception, "Wrong input type for %s found %s and expecting %s for value %s" %\
                        (name, type(value), targettype, value)
            else:
                raise Exception, "Wrong input type for %s found %s and expecting %s for value %s" %\
                        (name, type(value), targettype, value)                
        else:
            # We have a string we have to format the attribute from the string
            if targettype == bool:
                value = value.strip()
                if value.lower() in ['0', '.false.', 'f', 'false']:
                    value = False
                elif value.lower() in ['1', '.true.', 't', 'true']:
                    value = True
                else:
                    raise Exception, "%s can not be mapped to True/False for %s" % (repr(value),name)
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
                else:
                    try:
                        value = float(value.replace('d','e'))
                    except ValueError:
                        raise Exception, "%s can not be mapped to an integer" % value                    
                    try:
                        new_value = int(value)
                    except ValueError:
                        raise Exception, "%s can not be mapped to an integer" % value
                    else:
                        if value == new_value:
                            value = new_value
                        else:
                            raise Exception, "incorect input: %s need an integer for %s" % (value,name)
            elif targettype == float:
                value = value.replace('d','e') # pass from Fortran formatting
                try:
                    value = float(value)
                except ValueError:
                    raise Exception, "%s can not be mapped to a float" % value
            else:
                raise Exception, "type %s is not handle by MadLoopParam" % targettype
            
        return value
            
 

    def __getitem__(self, name):
        
        if __debug__:
            if name.lower() not in self:
                if name.lower() in [key.lower() for key in self] :
                    raise Exception, "Some key are not lower case %s. Invalid use of the class!"\
                                     % [key for key in self if key.lower() != key]

        return dict.__getitem__(self, name.lower())

    
    def set(self, name, value, ifnotdefault=True, user=False):
        """convenient way to change attribute.
        ifnotdefault=False means that the value is NOT change is the value is not on default.
        user=True, means that the value will be marked as modified by the user 
        (potentially preventing future change to the value) 
        """

        # ifnotdefault=False -> we need to check if the user force a value.
        if not ifnotdefault:
            if name.lower() in self.user_set:
                #value modified by the user -> do nothing
                return
            
        self.__setitem__(name, value, change_userdefine=user) 
 


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
        
        fsock = open(output_file,'w')
        fsock.write(text)
        fsock.close()
    
class RunCard(ConfigFile):

    def __new__(cls, finput=None):
        if cls is RunCard:
            if not finput:
                target_class = RunCardLO
            elif isinstance(finput, cls):
                target_class = finput.__class__
            elif isinstance(finput, str):
                if '\n' not in finput:
                    finput = open(finput).read()
                if 'fixed_QES_scale' in finput:
                    target_class = RunCardNLO
                else:
                    target_class = RunCardLO
            else:
                return None
            return super(RunCard, cls).__new__(target_class, finput)
        else:
            return super(RunCard, cls).__new__(cls, finput)

    def __init__(self, *args, **opts):
        
        # The following parameter are updated in the defaultsetup stage.
        
        #parameter for which no warning should be raised if not define
        self.hidden_param = []
        # parameter which should not be hardcoded in the config file
        self.not_in_include = []
        #some parameter have different name in fortran code
        self.fortran_name = {}
        #parameter which are not supported anymore. (no action on the code)
        self.legacy_parameter = {}
        #a list with all the cuts variable
        self.cuts_parameter = []
        
        
        super(RunCard, self).__init__(*args, **opts)

    def add_param(self, name, value, fortran_name=None, include=True, 
                  hidden=False, legacy=False, cut=False):
        """ add a parameter to the card. value is the default value and 
        defines the type (int/float/bool/str) of the input.
        fortran_name defines what is the associate name in the f77 code
        include defines if we have to put the value in the include file
        hidden defines if the parameter is expected to be define by the user.
        legacy:Parameter which is not used anymore (raise a warning if not default)
        cut: defines the list of cut parameter to allow to set them all to off.
        """

        super(RunCard, self).add_param(name, value)
        name = name.lower()
        if fortran_name:
            self.fortran_name[name] = fortran_name
        if not include:
            self.not_in_include.append(name)
        if hidden:
            self.hidden_param.append(name)
        if legacy:
            self.legacy_parameter[name] = value
            if include:
                self.not_in_include.append(name)
        if cut:
            self.cuts_parameter.append(name)



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
            value, name = line
            name = name.lower().strip()
            if name not in self and ('min' in name or 'max' in name):
                #looks like an entry added by one user -> add it nicely
                self.add_param(name, float(value), hidden=True, cut=True)
            else:
                self.set( name, value, user=True)
                
    def write(self, output_file, template=None, python_template=False):
        """Write the run_card in output_file according to template 
           (a path to a valid run_card)"""

        to_write = set(self.user_set) 
        if not template:
            raise Exception

        if python_template and not to_write:
            text = file(template,'r').read() % self
        else:
            text = ""
            for line in file(template,'r'):                  
                nline = line.split('#')[0]
                nline = nline.split('!')[0]
                comment = line[len(nline):]
                nline = nline.split('=')
                if len(nline) != 2:
                    text += line
                elif nline[1].strip() in self:
                    if python_template:
                        text += line % {nline[1].strip().lower(): self[nline[1].strip()]}
                    else:
                        text += '  %s\t= %s %s' % (self[nline[1].strip()],nline[1], comment)        
                    if nline[1].strip().lower() in to_write:
                        to_write.remove(nline[1].strip().lower())
                else:
                    logger.info('Adding missing parameter %s to current run_card (with default value)' % nline[1].strip())
                    text += line 

        if to_write:
            text+="""#********************************************************************* 
#  Additional parameter
#*********************************************************************
"""
            
            for key in to_write:
                text += '  %s\t= %s # %s\n' % (self[key], key, 'hidden parameter')

        if isinstance(output_file, str):
            fsock = open(output_file,'w')
            fsock.write(text)
            fsock.close()
        else:
            output_file.write(text)


    def get_default(self, name, default=None, log_level=None):
        """return self[name] if exist otherwise default. log control if we 
        put a warning or not if we use the default value"""

        if name not in self.user_set:
            if log_level is None:
                if name.lower() in self.hidden_param:
                    log_level = 10
                else:
                    log_level = 20
            if not default:
                default = self[name]
            logger.log(log_level, 'run_card missed argument %s. Takes default: %s'
                                   % (name, default))
            self[name] = default
            return default
        else:
            return self[name]   

    @staticmethod
    def format(formatv, value):
        """for retro compatibility"""
        
        logger.debug("please use f77_formatting instead of format")
        return self.f77_formatting(value, formatv=formatv)
    
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
                logger.debug("unknow format for f77_formatting: %s" , value)
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
            return "'%s'" % value

        

    def write_include_file(self, output_file):
        """ """
        
        # ensure that all parameter are coherent and fix those if needed
        self.check_validity()
        
        fsock = file_writers.FortranWriter(output_file)  
        for key in self:
            if key in self.not_in_include:
                continue
            
            #define the fortran name
            if key in self.fortran_name:
                fortran_name = self.fortran_name[key]
            else:
                fortran_name = key
                
            #get the value with warning if the user didn't set it
            value = self.get_default(key) 
            
            line = '%s = %s \n' % (fortran_name, self.f77_formatting(value))
            fsock.writelines(line)
        fsock.close()   


    def get_banner_init_information(self):
        """return a dictionary with the information needed to write
        the first line of the <init> block of the lhe file."""
        
        output = {}
        
        def get_idbmup(lpp):
            """return the particle colliding pdg code"""
            if lpp in (1,2, -1,-2):
                return math.copysign(2212, lpp)
            elif lpp in (3,-3):
                return math.copysign(11, lpp)
            elif lpp == 0:
                logger.critical("Fail to write correct idbmup in the lhe file. Please correct those by hand")
                return 0
            else:
                return lpp
        
        def get_pdf_id(pdf):
            if pdf == "lhapdf":
                return self["lhaid"]
            else: 
                return {'none': 0, 'mrs02nl':20250, 'mrs02nn':20270, 'cteq4_m': 19150,
                        'cteq4_l':19170, 'cteq4_d':19160, 'cteq5_m':19050, 
                        'cteq5_d':19060,'cteq5_l':19070,'cteq5m1':19051,
                        'cteq6_m':10000,'cteq6_l':10041,'cteq6l1':10042,
                        'nn23lo':246800,'nn23lo1':247000,'nn23nlo':244600
                        }[pdf]
            
        output["idbmup1"] = get_idbmup(self['lpp1'])
        output["idbmup2"] = get_idbmup(self['lpp2'])
        output["ebmup1"] = self["ebeam1"]
        output["ebmup2"] = self["ebeam2"]
        output["pdfgup1"] = 0
        output["pdfgup2"] = 0
        output["pdfsup1"] = get_pdf_id(self["pdlabel"])
        output["pdfsup2"] = get_pdf_id(self["pdlabel"])
        return output

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
    """an object to handle in a nice way the run_card infomration"""
    
    def default_setup(self):
        """default value for the run_card.dat"""
        
        self.add_param("run_tag", "tag_1", include=False)
        self.add_param("gridpack", False)
        self.add_param("time_of_flight", -1.0, include=False, hidden=True)
        self.add_param("nevents", 10000)        
        self.add_param("iseed", 0)
        self.add_param("lpp1", 1, fortran_name="lpp(1)")
        self.add_param("lpp2", 1, fortran_name="lpp(2)")
        self.add_param("ebeam1", 6500.0, fortran_name="ebeam(1)")
        self.add_param("ebeam2", 6500.0, fortran_name="ebeam(2)")
        self.add_param("polbeam1", 0.0, fortran_name="pb1")
        self.add_param("polbeam2", 0.0, fortran_name="pb2")
        self.add_param("pdlabel", "nn23lo1")
        self.add_param("lhaid", 230000, hidden=True)
        self.add_param("fixed_ren_scale", False)
        self.add_param("fixed_fac_scale", False)
        self.add_param("scale", 91.1880)
        self.add_param("dsqrt_q2fact1", 91.1880, fortran_name="sf1")
        self.add_param("dsqrt_q2fact2", 91.1880, fortran_name="sf2")
        self.add_param("dynamical_scale_choice", -1)
        
        #matching
        self.add_param("scalefact", 1.0)
        self.add_param("ickkw", 0)
        self.add_param("highestmult", 1, fortran_name="nhmult")
        self.add_param("ktscheme", 1)
        self.add_param("alpsfact", 1.0)
        self.add_param("chcluster", False)
        self.add_param("pdfwgt", True)
        self.add_param("asrwgtflavor", 5)
        self.add_param("clusinfo", True)
        self.add_param("lhe_version", 3.0)
        #cut
        self.add_param("auto_ptj_mjj", True)
        self.add_param("bwcutoff", 15.0)
        self.add_param("cut_decays", False)
        self.add_param("nhel", 0, include=False)
        #pt cut
        self.add_param("ptj", 20.0, cut=True)
        self.add_param("ptb", 0.0, cut=True)
        self.add_param("pta", 10.0, cut=True)
        self.add_param("ptl", 10.0, cut=True)
        self.add_param("misset", 0.0, cut=True)
        self.add_param("ptheavy", 0.0, cut=True)
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
        self.add_param("maxjetflavor", 4)
        self.add_param("xqcut", 0.0, cut=True)
        self.add_param("use_syst", True)
        self.add_param("sys_scalefact", "0.5 1 2", include=False)
        self.add_param("sys_alpsfact", "None", include=False)
        self.add_param("sys_matchscale", "30 50", include=False)
        self.add_param("sys_pdf", "Ct10nlo.LHgrid", include=False)
        self.add_param("sys_scalecorrelation", -1, include=False)
        
        #parameter not in the run_card by default
        self.add_param('gridrun', False, hidden=True)
        self.add_param('fixed_couplings', True, hidden=True)
        self.add_param('mc_grouped_subproc', True, hidden=True)
        self.add_param('xmtcentral', 0.0, hidden=True, fortran_name="xmtc")
        self.add_param('d', 1.0, hidden=True)
        self.add_param('gseed', 0, hidden=True, include=False)
        self.add_param('issgridfile', '', hidden=True)
        #job handling of the survey/ refine
        self.add_param('job_strategy', 0, hidden=True, include=False)
        self.add_param('survey_splitting', -1, hidden=True, include=False)
        self.add_param('refine_evt_by_job', -1, hidden=True, include=False)

 

        
    def check_validity(self):
        """ """
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
        possible_set = ['lhapdf','mrs02nl','mrs02nn', 'mrs0119','mrs0117','mrs0121','mrs01_j', 'mrs99_1','mrs99_2','mrs99_3','mrs99_4','mrs99_5','mrs99_6', 'mrs99_7','mrs99_8','mrs99_9','mrs9910','mrs9911','mrs9912', 'mrs98z1','mrs98z2','mrs98z3','mrs98z4','mrs98z5','mrs98ht', 'mrs98l1','mrs98l2','mrs98l3','mrs98l4','mrs98l5', 'cteq3_m','cteq3_l','cteq3_d', 'cteq4_m','cteq4_d','cteq4_l','cteq4a1','cteq4a2', 'cteq4a3','cteq4a4','cteq4a5','cteq4hj','cteq4lq', 'cteq5_m','cteq5_d','cteq5_l','cteq5hj','cteq5hq', 'cteq5f3','cteq5f4','cteq5m1','ctq5hq1','cteq5l1', 'cteq6_m','cteq6_d','cteq6_l','cteq6l1', 'nn23lo','nn23lo1','nn23nlo']
        if self['pdlabel'] not in possible_set:
            raise InvalidRunCard, 'Invalid PDF set (argument of pdlabel): %s. Possible choice are:\n %s' % (self['pdlabel'], ', '.join(possible_set))
        if self['pdlabel'] == 'lhapdf':
            #add warning if lhaid not define
            self.get_default('lhaid', log_level=20)
   
        for name in self.legacy_parameter:
            if self[name] != self.legacy_parameter[name]:
                logger.warning("The parameter %s is not supported anymore this parameter will be ignored." % name)
                

            
        
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
            
        if proc_characteristic['ninitial'] == 1:
            #remove all cut
            self.remove_all_cut()
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
            else:
                self['lpp1'] = 0
                self['lpp2'] = 0                
                
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
                self['use_syst'] = False 
                self['drjj'] = 0
                self['drjl'] = 0
                self['sys_alpsfact'] = "0.5 1 2"
                
            
    def write(self, output_file, template=None, python_template=False):
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
                                    python_template=python_template)            


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
        self.add_param('pdlabel', 'nn23nlo')                
        self.add_param('lhaid', 244600)
        #shower and scale
        self.add_param('parton_shower', 'HERWIG6', fortran_name='shower_mc')        
        self.add_param('shower_scale_factor',1.0)
        self.add_param('fixed_ren_scale', False)
        self.add_param('fixed_fac_scale', False)
        self.add_param('mur_ref_fixed', 91.118)                       
        self.add_param('muf1_ref_fixed', 91.118)
        self.add_param('muf2_ref_fixed', 91.118)
        self.add_param("dynamical_scale_choice", -1)
        self.add_param('fixed_qes_scale', False)
        self.add_param('qes_ref_fixed', 91.118)
        self.add_param('mur_over_ref', 1.0)
        self.add_param('muf1_over_ref', 1.0)                       
        self.add_param('muf2_over_ref', 1.0)
        self.add_param('qes_over_ref', 1.0)
        self.add_param('reweight_scale', True, fortran_name='do_rwgt_scale')
        self.add_param('rw_rscale_down', 0.5)        
        self.add_param('rw_rscale_up', 2.0)
        self.add_param('rw_fscale_down', 0.5)                       
        self.add_param('rw_fscale_up', 2.0)
        self.add_param('reweight_pdf', False, fortran_name='do_rwgt_pdf')
        self.add_param('pdf_set_min', 244601)
        self.add_param('pdf_set_max', 244700)
        #merging
        self.add_param('ickkw', 0)
        self.add_param('bwcutoff', 15.0)
        #cuts        
        self.add_param('jetalgo', 1.0)
        self.add_param('jetradius', 0.7, hidden=True)         
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
        self.add_param('maxjetflavor', 4)
        self.add_param('iappl', 0)   
    
        self.add_param('lhe_version', 3, hidden=True, include=False)
    
    def check_validity(self):
        """check the validity of the various input"""
        
        # For FxFx merging, make sure that the following parameters are set correctly:
        if self['ickkw'] == 3: 
            # 1. Renormalization and factorization (and ellis-sexton scales) are not fixed       
            scales=['fixed_ren_scale','fixed_fac_scale','fixed_QES_scale']
            for scale in scales:
                if self[scale]:
                    logger.warning('''For consistency in the FxFx merging, \'%s\' has been set to false'''
                                % scale,'$MG:color:BLACK')
                    self[scale]= False
            #and left to default dynamical scale
            if self["dynamical_scale_choice"] != -1:
                self["dynamical_scale_choice"] = -1
                logger.warning('''For consistency in the FxFx merging, dynamical_scale_choice has been set to -1 (default)'''
                                ,'$MG:color:BLACK')
                
                
            # 2. Use kT algorithm for jets with pseudo-code size R=1.0
            jetparams=['jetradius','jetalgo']
            for jetparam in jetparams:
                if float(self[jetparam]) != 1.0:
                    logger.info('''For consistency in the FxFx merging, \'%s\' has been set to 1.0'''
                                % jetparam ,'$MG:color:BLACK')
                    self[jetparam] = 1.0
        elif self['ickkw'] == -1 and self["dynamical_scale_choice"] != -1:
                self["dynamical_scale_choice"] = -1
                self["dynamical_scale_choice"] = -1
                logger.warning('''For consistency with the jet veto, the scale which will be used is ptj. dynamical_scale_choice will be set at -1.'''
                                ,'$MG:color:BLACK')            
            
                                
        # For interface to APPLGRID, need to use LHAPDF and reweighting to get scale uncertainties
        if self['iappl'] != 0 and self['pdlabel'].lower() != 'lhapdf':
            raise InvalidRunCard('APPLgrid generation only possible with the use of LHAPDF')
        if self['iappl'] != 0 and not self['reweight_scale']:
            raise InvalidRunCard('APPLgrid generation only possible with including' +\
                                      ' the reweighting to get scale dependence')

        # check that the pdf is set correctly
        possible_set = ['lhapdf','mrs02nl','mrs02nn', 'mrs0119','mrs0117','mrs0121','mrs01_j', 'mrs99_1','mrs99_2','mrs99_3','mrs99_4','mrs99_5','mrs99_6', 'mrs99_7','mrs99_8','mrs99_9','mrs9910','mrs9911','mrs9912', 'mrs98z1','mrs98z2','mrs98z3','mrs98z4','mrs98z5','mrs98ht', 'mrs98l1','mrs98l2','mrs98l3','mrs98l4','mrs98l5', 'cteq3_m','cteq3_l','cteq3_d', 'cteq4_m','cteq4_d','cteq4_l','cteq4a1','cteq4a2', 'cteq4a3','cteq4a4','cteq4a5','cteq4hj','cteq4lq', 'cteq5_m','cteq5_d','cteq5_l','cteq5hj','cteq5hq', 'cteq5f3','cteq5f4','cteq5m1','ctq5hq1','cteq5l1', 'cteq6_m','cteq6_d','cteq6_l','cteq6l1', 'nn23lo','nn23lo1','nn23nlo']
        if self['pdlabel'] not in possible_set:
            raise InvalidRunCard, 'Invalid PDF set (argument of pdlabel) possible choice are:\n %s' % ','.join(possible_set)
    

        # PDF reweighting check
        if self['reweight_pdf']:
            # check that we use lhapdf if reweighting is ON
            if self['pdlabel'] != "lhapdf":
                raise InvalidRunCard, 'Reweight PDF option requires to use pdf sets associated to lhapdf. Please either change the pdlabel or set reweight_pdf to False.'
            
            # check that the number of pdf set is coherent for the reweigting:    
            if (self['pdf_set_max'] - self['pdf_set_min'] + 1) % 2:
                raise InvalidRunCard, "The number of PDF error sets must be even" 
        

    def write(self, output_file, template=None, python_template=False):
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
                                    python_template=python_template)


    def create_default_for_process(self, proc_characteristic, history, proc_def):
        """Rules
          e+ e- beam -> lpp:0 ebeam:500  
          p p beam -> set maxjetflavor automatically
        """

        # check for beam_id
        beam_id = set()
        for proc in proc_def:
            for leg in proc[0]['legs']:
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
    

            
    def default_setup(self):
        """initialize the directory to the default value"""
        
        self.add_param("MLReductionLib", "1|3|2")
        self.add_param("IREGIMODE", 2)
        self.add_param("IREGIRECY", True)
        self.add_param("CTModeRun", -1)
        self.add_param("MLStabThres", 1e-3)
        self.add_param("NRotations_DP", 1)
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
                template = pjoin(MEDIR, 'SubProcesses', 'MadLoop5_resources',
                                                           'MadLoopParams.dat' )
                if not os.path.exists(template):
                    template = pjoin(MEDIR, 'Cards', 'MadLoopParams.dat')
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
        
        
        
        
        
        
        
    
